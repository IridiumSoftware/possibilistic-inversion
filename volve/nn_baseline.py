"""
volve/nn_baseline.py - Neural-network comparator on F-15A 1D Vp(z).

A small MLP trained as a SUPERVISED inverter: input = the 1215-element
F-15A pick-time vector, output = the 45-element Vp(z) profile.

Training data are SYNTHETIC: 5000 smooth random Vp(z) profiles drawn
from the same prior as phase 4 + their corresponding eikonal-forward
pick times under the F-15A geometry. The network learns the inverse
mapping. At test time we feed the REAL F-15A picks.

Uncertainty: MC dropout. Inference is repeated 100 times with dropout
ACTIVE (training mode); the spread of predictions across runs is the
Monte-Carlo-dropout uncertainty band.

HONEST SCOPE.
  - NOT a "state-of-the-art tomography NN." A small MLP, 3 hidden
    layers, ~300k params. Standard supervised regression.
  - NOT trained on Volve. Trained on synthetic profiles drawn from
    the phase-4 prior with the F-15A eikonal forward.
  - MC dropout is a known approximation to Bayesian uncertainty.
    The dropout band is the network's "epistemic" uncertainty under
    the assumption the synthetic prior is right.
  - The point is a comparator under MATCHED PHYSICS + MATCHED PRIOR,
    not to beat posdec or MCMC.

Run:  uv run python -m volve.nn_baseline
"""

from dataclasses import dataclass
from pathlib import Path
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import lasio

from scipy.ndimage import gaussian_filter1d

from volve.inversion_1d import (
    load_picks, depth_grid_for_picks, depth_centers,
)
from volve.inversion_eikonal import (
    forward_eikonal_1d, grid_dimensions, pick_grid_coords,
)


OUT_CERT = "volve/picks/nn_certificate.json"
OUT_NPZ = "volve/picks/nn_predictions.npz"
OUT_FIG = "volve_nn_baseline.png"

N_TRAIN = 5000
N_VAL = 500
N_EPOCHS = 80
BATCH_SIZE = 64
LR = 1e-3
DROPOUT_P = 0.20
MC_DROPOUT_RUNS = 200

# Prior settings - same as phase 4
VP_MIN_KMS = 1.5
VP_MAX_KMS = 5.5
SMOOTH_CORR_M = 250.0


# --- synthetic profile generator -----------------------------------------

def _synth_vp(rng, n_bins, bin_thick_m):
    """Draw a smooth random Vp(z) profile from the phase-4 prior."""
    z_frac = np.linspace(0.0, 1.0, n_bins)
    trend = 1.7 + 2.8 * z_frac
    sigma = max(1.0, SMOOTH_CORR_M / bin_thick_m)
    noise = gaussian_filter1d(rng.standard_normal(n_bins), sigma=sigma) * 0.5
    return np.clip(trend + noise, VP_MIN_KMS, VP_MAX_KMS)


# --- training data generation --------------------------------------------

def _generate_dataset(n_samples, picks, grid, nz, nx, cell_m,
                      ix_recv, iz_recv, seed):
    rng = np.random.default_rng(seed)
    n_bins = grid.size - 1
    bin_thick = float(grid[1] - grid[0])
    X = np.zeros((n_samples, picks.n()), dtype=np.float32)
    Y = np.zeros((n_samples, n_bins), dtype=np.float32)
    for i in range(n_samples):
        vp = _synth_vp(rng, n_bins, bin_thick)
        t_pred, _ = forward_eikonal_1d(
            vp, picks, nz, nx, cell_m,
            ix_recv=ix_recv, iz_recv=iz_recv, compute_jacobian=False)
        X[i] = t_pred.astype(np.float32)
        Y[i] = vp.astype(np.float32)
        if (i + 1) % 500 == 0:
            print(f"    generated {i + 1}/{n_samples}")
    return X, Y


# --- NN ------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, n_in, n_out, hidden=(512, 256, 128),
                 dropout_p=DROPOUT_P):
        super().__init__()
        dims = [n_in] + list(hidden) + [n_out]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_p))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --- sonic loader --------------------------------------------------------

def _load_sonic_f15a():
    las = lasio.read("volve/data/15_9-F-15 A/TZV_DEPTH_MD_COMPUTED_1.LAS")
    tvd = np.asarray(las["TVD"], dtype=float)
    dt = np.asarray(las["DT-EDIT"], dtype=float)
    m = np.isfinite(tvd) & np.isfinite(dt) & (dt > 0) & (dt < 500)
    return tvd[m], 304.8 / dt[m]


# --- main ----------------------------------------------------------------

def main():
    torch.manual_seed(20260608)
    np.random.seed(20260608)
    t0 = time.time()

    picks = load_picks("volve/picks/picks_z.csv")
    grid = depth_grid_for_picks(picks)
    z_centers = depth_centers(grid)
    n_bins = grid.size - 1
    n_pick = picks.n()
    print(f"picks: {n_pick}, bins: {n_bins}")

    nz, nx, cell_m = grid_dimensions(picks)
    ix_recv, iz_recv = pick_grid_coords(picks, cell_m)
    print(f"FMM grid: {nz}x{nx} cells of {cell_m:.0f} m")

    print(f"generating {N_TRAIN} train + {N_VAL} val synthetic samples...")
    t_data = time.time()
    X_train, Y_train = _generate_dataset(
        N_TRAIN, picks, grid, nz, nx, cell_m, ix_recv, iz_recv,
        seed=20260608)
    X_val, Y_val = _generate_dataset(
        N_VAL, picks, grid, nz, nx, cell_m, ix_recv, iz_recv,
        seed=20260609)
    print(f"  data generation: {time.time() - t_data:.0f} s")

    # Standardize inputs
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0) + 1e-6
    Xn = (X_train - x_mean) / x_std
    Xn_val = (X_val - x_mean) / x_std

    Xn_t = torch.from_numpy(Xn).float()
    Y_t = torch.from_numpy(Y_train).float()
    Xn_val_t = torch.from_numpy(Xn_val).float()
    Y_val_t = torch.from_numpy(Y_val).float()

    model = MLP(n_pick, n_bins)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MLP: {n_params:,} parameters")

    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print(f"training {N_EPOCHS} epochs, batch {BATCH_SIZE}...")
    t_train = time.time()
    n_batches = N_TRAIN // BATCH_SIZE
    val_losses = []
    for epoch in range(N_EPOCHS):
        model.train()
        idx = torch.randperm(N_TRAIN)
        epoch_loss = 0.0
        for b in range(n_batches):
            sel = idx[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            xb = Xn_t[sel]
            yb = Y_t[sel]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
        epoch_loss /= n_batches
        model.eval()
        with torch.no_grad():
            val_pred = model(Xn_val_t)
            val_loss = float(loss_fn(val_pred, Y_val_t).item())
        val_losses.append(val_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:3d}: train MSE={epoch_loss:.4f} "
                  f"val MSE={val_loss:.4f}")
    print(f"training: {time.time() - t_train:.0f} s")

    # Predict on real F-15A picks (with MC dropout)
    print(f"MC dropout inference: {MC_DROPOUT_RUNS} runs")
    real_picks_x = picks.times_s.astype(np.float32)
    real_xn = (real_picks_x - x_mean) / x_std
    real_xn_t = torch.from_numpy(real_xn).float().unsqueeze(0)
    model.train()    # enable dropout
    preds = np.zeros((MC_DROPOUT_RUNS, n_bins), dtype=np.float32)
    with torch.no_grad():
        for r in range(MC_DROPOUT_RUNS):
            preds[r] = model(real_xn_t).squeeze(0).numpy()
    # Clip to envelope
    preds = np.clip(preds, VP_MIN_KMS, VP_MAX_KMS)

    pred_med = np.median(preds, axis=0)
    pred_p05 = np.percentile(preds, 5.0, axis=0)
    pred_p95 = np.percentile(preds, 95.0, axis=0)
    pred_p025 = np.percentile(preds, 2.5, axis=0)
    pred_p975 = np.percentile(preds, 97.5, axis=0)

    # Sonic calibration
    sonic_tvd, sonic_vp = _load_sonic_f15a()
    half = 0.5 * float(grid[1] - grid[0])
    sonic_bin_mean = np.full(n_bins, np.nan)
    for j in range(n_bins):
        sel = (sonic_tvd >= z_centers[j] - half) & \
              (sonic_tvd < z_centers[j] + half)
        if sel.any():
            sonic_bin_mean[j] = float(np.mean(sonic_vp[sel]))
    ok = np.isfinite(sonic_bin_mean)
    in90 = (sonic_bin_mean >= pred_p05) & (sonic_bin_mean <= pred_p95)
    in95 = (sonic_bin_mean >= pred_p025) & (sonic_bin_mean <= pred_p975)
    n_ok = int(ok.sum())
    n_in90 = int(in90[ok].sum())
    n_in95 = int(in95[ok].sum())
    elapsed = time.time() - t0
    print()
    print("NN MC-dropout sonic calibration vs DT-EDIT:")
    print(f"  sonic-bin coverage : {n_ok}/{n_bins}")
    print(f"  inside 90% band    : {n_in90}/{n_ok} "
          f"({100 * n_in90 / n_ok:.1f}%)")
    print(f"  inside 95% band    : {n_in95}/{n_ok} "
          f"({100 * n_in95 / n_ok:.1f}%)")
    print(f"\ntotal elapsed: {elapsed:.0f} s")

    # Figure
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.fill_betweenx(z_centers, pred_p025, pred_p975,
                     color="#88aaff", alpha=0.35, label="95% MC dropout")
    ax.fill_betweenx(z_centers, pred_p05, pred_p95,
                     color="#2166ac", alpha=0.45, label="90% MC dropout")
    ax.plot(pred_med, z_centers, color="#08306b", lw=1.6,
            label="NN median prediction")
    ax.plot(sonic_vp, sonic_tvd, color="#b2182b", lw=0.5,
            label="DT-EDIT sonic")
    ax.invert_yaxis()
    ax.set_xlabel("Vp (km/s)")
    ax.set_ylabel("depth below sea surface (m)")
    ax.set_title(f"NN MC-dropout baseline (MLP, supervised on synthetic) "
                 f"- F-15A\n90% sonic-inside = "
                 f"{100 * n_in90 / n_ok:.1f}%, "
                 f"95% sonic-inside = {100 * n_in95 / n_ok:.1f}%",
                 fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xlim(1.3, 5.8)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130)
    plt.close(fig)
    print(f"figure: {OUT_FIG}")

    # Snapshot
    np.savez_compressed(
        OUT_NPZ,
        preds=preds,
        z_centers_m=z_centers,
        bin_thick_m=np.array([grid[1] - grid[0]]),
        p025=pred_p025, p05=pred_p05, p50=pred_med,
        p95=pred_p95, p975=pred_p975,
        val_losses=np.array(val_losses, dtype=np.float32),
    )

    cert = {
        "label": "volve_f15a_nn_mlp_mcdropout",
        "method": "MLP supervised regression on synthetic eikonal data, "
                  "MC dropout for uncertainty",
        "settings": {
            "n_train": int(N_TRAIN), "n_val": int(N_VAL),
            "n_epochs": int(N_EPOCHS), "batch_size": int(BATCH_SIZE),
            "lr": float(LR), "dropout_p": float(DROPOUT_P),
            "mc_dropout_runs": int(MC_DROPOUT_RUNS),
            "hidden_layers": [512, 256, 128],
            "n_pick": int(n_pick), "n_bins": int(n_bins),
            "vp_envelope_kms": [float(VP_MIN_KMS), float(VP_MAX_KMS)],
            "prior_smooth_corr_m": float(SMOOTH_CORR_M),
        },
        "val_loss_final": float(val_losses[-1]),
        "sonic_calibration": {
            "sonic_bin_coverage": f"{n_ok}/{n_bins}",
            "inside_90pct_band": int(n_in90),
            "inside_95pct_band": int(n_in95),
            "inside_90pct_pct": float(100 * n_in90 / n_ok),
            "inside_95pct_pct": float(100 * n_in95 / n_ok),
        },
        "elapsed_seconds": float(elapsed),
    }
    Path(OUT_CERT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_CERT).write_text(json.dumps(cert, indent=2))
    print(f"certificate: {OUT_CERT}")


if __name__ == "__main__":
    main()
