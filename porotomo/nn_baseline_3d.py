"""porotomo/nn_baseline_3d.py - deep-learning baseline for the PoroTomo 3D
comparison: a pointwise MLP with MC-dropout uncertainty, trained on
synthetic eikonal data drawn from the SAME prior class as the ensemble.

3D port of volve/nn_baseline.py (which scored 7-9% sonic-inside on Volve).
Architecture choice is deliberately the "honest straightforward thing a
practitioner would try", not the state of the art - the comparison is
about what the uncertainty REPRESENTATION delivers under matched physics
and matched training distribution, not about NN engineering.

DESIGN.
  - Target: Vp at a query cell. Input features are aggregations of the
    observed picks around that cell:
        [z_elev, depth_below_surface,
         mean apparent slowness of nearby rays (proximity-weighted),
         p10/p90 apparent slowness, mean offset, mean time,
         log total proximity weight (data density)]
    "Nearby" = the straight source->receiver chord passes within
    PROX_R_M of the cell centre. The chord-proximity WEIGHTS depend only
    on geometry, so they are precomputed ONCE as a sparse (cells x picks)
    matrix; per synthetic model only the times change and features are
    sparse matrix-vector products.
  - Training data: N_MODELS random models from _smooth_random_vp_3d (the
    ensemble's own prior class) forwarded through the SAME eikonal C
    kernel; supervised pairs (features(cell), vp(cell)) on illuminated
    cells.
  - Uncertainty: MC-dropout (p=0.2) at inference, N_DROPOUT passes ->
    per-cell predictive interval; the passes are also pushed through the
    eikonal forward as "sampled models" for the stage-2 holdout test, so
    all three methods face identical tests.

Run:  uv run python -m porotomo.nn_baseline_3d           (~15 min)
Outputs: porotomo_nn_cert.json, porotomo/data/nn_models.npz
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix

from porotomo.inversion3d import prepare, forward_3d, Config3D, \
    _smooth_random_vp_3d
from porotomo.decompose_3d import load_ensemble, illumination, \
    ILLUM_MIN_PATH_M

PROX_R_M = 120.0
N_MODELS = 240
N_DROPOUT = 30          # matched to the other methods' member counts
SEED = 20260610


def chord_weights(grid, ds, cell_idx: np.ndarray) -> coo_matrix:
    """Sparse (n_cells_queried x n_picks) proximity weights: for each pick,
    w = max(0, 1 - d/PROX_R_M) with d the distance from the cell centre to
    the straight source->receiver chord."""
    cz, cy, cx = np.unravel_index(cell_idx, (grid.nz, grid.ny, grid.nx))
    centers = np.column_stack([cz, cy, cx]).astype(float)   # cell coords
    rows_l, cols_l, vals_l = [], [], []
    col0 = 0
    r_cells = PROX_R_M / grid.cell_m
    for i in range(len(ds.src_pts)):
        a = ds.src_pts[i]                       # (3,)
        B = ds.recv_pts[i]                      # (m, 3)
        ab = B - a                              # (m, 3)
        ab2 = (ab ** 2).sum(axis=1)             # (m,)
        # distance from each center to each chord, batched over chords
        for j0 in range(0, len(B), 64):
            abj = ab[j0:j0 + 64]
            ab2j = ab2[j0:j0 + 64]
            ac = centers[:, None, :] - a[None, None, :]      # (nc, 1, 3)
            tpar = (ac * abj[None, :, :]).sum(axis=2) / np.maximum(ab2j, 1e-9)
            tpar = np.clip(tpar, 0.0, 1.0)                   # (nc, mj)
            proj = a[None, None, :] + tpar[:, :, None] * abj[None, :, :]
            d = np.linalg.norm(centers[:, None, :] - proj, axis=2)
            w = np.maximum(0.0, 1.0 - d / r_cells)
            nz_r, nz_c = np.nonzero(w)
            rows_l.append(nz_r)
            cols_l.append(nz_c + col0 + j0)
            vals_l.append(w[nz_r, nz_c])
        col0 += len(B)
    return coo_matrix(
        (np.concatenate(vals_l),
         (np.concatenate(rows_l), np.concatenate(cols_l))),
        shape=(len(cell_idx), ds.n_picks)).tocsr()


def build_features(W, t_picks, offsets, depths, z_elevs):
    """Feature matrix from precomputed weights and one set of pick times."""
    app_slow = t_picks / np.maximum(offsets, 1.0)            # s/m
    wsum = np.asarray(W.sum(axis=1)).ravel() + 1e-9
    f_mean_slow = np.asarray(W @ app_slow).ravel() / wsum
    f_mean_off = np.asarray(W @ offsets).ravel() / wsum
    f_mean_t = np.asarray(W @ t_picks).ravel() / wsum
    f = np.column_stack([
        z_elevs, depths, f_mean_slow * 1000.0, f_mean_off / 1000.0,
        f_mean_t, np.log10(wsum),
    ])
    return f


class MLP(nn.Module):
    def __init__(self, n_in, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    cfg = Config3D()
    picks, grid, air, ds = prepare()
    members, air_e, grid_e, _ = load_ensemble()
    lit = (illumination(members, air, grid, ds) >= ILLUM_MIN_PATH_M) & (~air)
    lit_idx = np.flatnonzero(lit.ravel())
    print(f"lit cells: {len(lit_idx)}")

    # geometry-only precomputations
    offsets = np.concatenate([
        np.linalg.norm((ds.recv_pts[i] - ds.src_pts[i]) * grid.cell_m, axis=1)
        for i in range(len(ds.src_pts))])
    cz, cy, cx = np.unravel_index(lit_idx, (grid.nz, grid.ny, grid.nx))
    z_elevs = grid.cell_centers_elev()[cz]
    surf_k = np.argmax(~air, axis=0)
    depths = (cz - surf_k[cy, cx]) * grid.cell_m
    print("precomputing chord weights...")
    t0 = time.time()
    W = chord_weights(grid, ds, lit_idx)
    print(f"  W: {W.shape}, nnz {W.nnz} ({time.time()-t0:.0f} s)")

    # ---- synthetic training set ------------------------------------------
    feats_l, targs_l = [], []
    print(f"generating {N_MODELS} synthetic models...")
    t0 = time.time()
    for i in range(N_MODELS):
        vp = _smooth_random_vp_3d(rng, grid, cfg)
        vp[air] = cfg.vp_air_kms
        t_syn, _ = forward_3d(vp, ds, grid)
        t_syn = t_syn + rng.normal(0.0, cfg.noise_rms_s, len(t_syn))
        f = build_features(W, t_syn, offsets, depths, z_elevs)
        sub = rng.choice(len(lit_idx), size=min(400, len(lit_idx)),
                         replace=False)
        feats_l.append(f[sub])
        targs_l.append(vp.ravel()[lit_idx[sub]])
        if (i + 1) % 60 == 0:
            print(f"  {i+1}/{N_MODELS} ({time.time()-t0:.0f} s)")
    X = np.concatenate(feats_l).astype(np.float32)
    y = np.concatenate(targs_l).astype(np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xn = (X - mu) / sd
    print(f"training set: {X.shape}")

    # ---- train -------------------------------------------------------------
    model = MLP(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt = torch.from_numpy(Xn)
    yt = torch.from_numpy(y)
    n = len(yt)
    for epoch in range(40):
        perm = torch.randperm(n)
        tot = 0.0
        for j0 in range(0, n, 4096):
            idx = perm[j0:j0 + 4096]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}: train mse {tot/n:.4f}")

    # ---- MC-dropout inference on the REAL picks ----------------------------
    t_real = np.concatenate(ds.times)
    f_real = build_features(W, t_real, offsets, depths, z_elevs)
    Xr = torch.from_numpy(((f_real - mu) / sd).astype(np.float32))
    model.train()                                # keep dropout ON
    passes = np.stack([
        model(Xr).detach().numpy() for _ in range(N_DROPOUT)
    ])                                           # (N_DROPOUT, n_lit)
    vp_lo_cell = np.quantile(passes, 0.025, axis=0)
    vp_hi_cell = np.quantile(passes, 0.975, axis=0)

    # sampled models for the holdout test: trend background + NN cells
    nn_models = np.zeros((N_DROPOUT, grid.nz, grid.ny, grid.nx))
    base = np.median(members, axis=0)            # fill unlit with median
    for i in range(N_DROPOUT):
        vp_i = base.copy()
        vp_i.ravel()[lit_idx] = np.clip(passes[i], cfg.vp_min_kms,
                                        cfg.vp_max_kms)
        nn_models[i] = vp_i
    np.savez_compressed("porotomo/data/nn_models.npz", models=nn_models,
                        vp_lo=vp_lo_cell, vp_hi=vp_hi_cell, lit_idx=lit_idx)

    # ---- stage-2 holdout ----------------------------------------------------
    _p2, _g2, _a2, ds2 = prepare(stage=2)
    t_obs2 = np.concatenate(ds2.times)
    preds = np.empty((N_DROPOUT, len(t_obs2)))
    for i in range(N_DROPOUT):
        preds[i], _ = forward_3d(nn_models[i], ds2, grid)
    t_lo2, t_hi2 = preds.min(axis=0), preds.max(axis=0)
    t_med2 = np.median(preds, axis=0)
    cert = {
        "method": "pointwise MLP + MC-dropout, trained on synthetic "
                  "eikonal data from the ensemble's prior class",
        "n_dropout": N_DROPOUT, "n_models_train": N_MODELS,
        "prox_radius_m": PROX_R_M,
        "holdout": {
            "inside_raw": round(float(((t_obs2 >= t_lo2)
                                       & (t_obs2 <= t_hi2)).mean()), 4),
            "inside_with_pick_noise": round(float(
                ((t_obs2 >= t_lo2 - 0.036)
                 & (t_obs2 <= t_hi2 + 0.036)).mean()), 4),
            "rms_median_prediction_ms": round(float(
                np.sqrt(np.mean((t_obs2 - t_med2) ** 2))) * 1000, 1),
        },
        "median_interval_width_kms_lit": round(float(
            np.median(vp_hi_cell - vp_lo_cell)), 3),
    }
    with open("porotomo_nn_cert.json", "w") as fh:
        json.dump(cert, fh, indent=2)
    print(json.dumps(cert, indent=2))


if __name__ == "__main__":
    main()
