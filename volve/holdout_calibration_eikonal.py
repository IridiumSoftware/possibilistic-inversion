"""
volve/holdout_calibration_eikonal.py - PHASE 4 held-out calibration.

Same structure as holdout_calibration.py but uses the eikonal Gauss-Newton
ensemble. The headline question is the same: do held-out picks fall inside
the ensemble's predicted-pick-time interval at the rate the ensemble
claims?

Run:  uv run python -m volve.holdout_calibration_eikonal
"""

from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from volve.inversion_1d import (
    PickData, load_picks, depth_grid_for_picks,
)
from volve.inversion_eikonal import (
    vp_ensemble_eikonal, EikonalConfig,
    forward_eikonal_1d, grid_dimensions, pick_grid_coords,
)
from volve.holdout_calibration import _split, _subset


OUT_JSON = "volve/picks/phase4_holdout.json"
OUT_FIG = "volve_phase4_holdout.png"


def predict_held_out_eikonal(members_vp_kms, test, nz, nx, cell_m):
    """For each ensemble member, run an eikonal forward on the test
    picks; return predicted travel times of shape (n_test, n_members)."""
    ix_recv, iz_recv = pick_grid_coords(test, cell_m)
    n_test = test.n()
    n_mem = members_vp_kms.shape[0]
    out = np.zeros((n_test, n_mem), dtype=float)
    for k in range(n_mem):
        t_k, _ = forward_eikonal_1d(
            members_vp_kms[k], test, nz, nx, cell_m,
            ix_recv=ix_recv, iz_recv=iz_recv,
            compute_jacobian=False)
        out[:, k] = t_k
    return out


def main():
    picks = load_picks()
    grid = depth_grid_for_picks(picks)
    train_idx, test_idx = _split(picks)
    train = _subset(picks, train_idx)
    test = _subset(picks, test_idx)
    print(f"loaded {picks.n()} ok picks; split {len(train_idx)} train / "
          f"{len(test_idx)} test")

    cfg = EikonalConfig(n_members=30, n_gn_iters=3)
    members_vp, meta = vp_ensemble_eikonal(train, grid, cfg)
    print(f"ensemble (eikonal, train): {members_vp.shape[0]} feasible "
          f"Vp(z); median RMS = "
          f"{np.median(meta['members_rms_s']) * 1000:.1f} ms")

    nz, nx, cell_m = grid_dimensions(test, cell_m=cfg.cell_m)
    pred_t = predict_held_out_eikonal(members_vp, test, nz, nx, cell_m)

    pred_min = pred_t.min(axis=1)
    pred_max = pred_t.max(axis=1)
    pred_med = np.median(pred_t, axis=1)
    obs_t = test.times_s

    inside = (obs_t >= pred_min) & (obs_t <= pred_max)
    resid = obs_t - pred_med
    interval_width = pred_max - pred_min

    print()
    print("phase 4 held-out calibration:")
    print(f"  inside ensemble interval : {int(inside.sum())}/{len(obs_t)} "
          f"({100*inside.mean():.1f}%)")
    print(f"  residual (obs - ens_med) :")
    print(f"    mean   = {resid.mean()*1000:+.2f} ms")
    print(f"    median = {np.median(resid)*1000:+.2f} ms")
    print(f"    std    = {resid.std()*1000:.2f} ms")
    print(f"    RMS    = {np.sqrt((resid**2).mean())*1000:.2f} ms")
    print(f"  interval width:")
    print(f"    mean   = {interval_width.mean()*1000:.1f} ms")
    print(f"    median = {np.median(interval_width)*1000:.1f} ms")

    # Figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.5))
    ax[0].errorbar(obs_t * 1000, pred_med * 1000,
                   yerr=[(pred_med - pred_min) * 1000,
                         (pred_max - pred_med) * 1000],
                   fmt="o", ms=2, color="#2166ac", ecolor="#88aaff",
                   alpha=0.6, elinewidth=0.4)
    lims = [min(obs_t.min(), pred_med.min()) * 1000,
            max(obs_t.max(), pred_med.max()) * 1000]
    ax[0].plot(lims, lims, "k--", lw=0.6)
    ax[0].set_xlabel("observed pick time (ms)")
    ax[0].set_ylabel("ensemble predicted pick time (ms)")
    ax[0].set_title(f"phase 4 eikonal: predicted vs observed\n"
                    f"(inside={100*inside.mean():.1f}%)")
    ax[0].grid(alpha=0.3)

    ax[1].scatter(test.source_offsets_m, resid * 1000, s=4,
                  color="#2166ac", alpha=0.6)
    ax[1].axhline(0, color="0.5", lw=0.5)
    ax[1].set_xlabel("source-to-well offset (m)")
    ax[1].set_ylabel("residual obs - ens_med (ms)")
    ax[1].set_title("residual vs offset")
    ax[1].grid(alpha=0.3)

    order = np.argsort(test.receivers_xyz[:, 2])
    z = test.receivers_xyz[order, 2]
    ax[2].fill_between(z, pred_min[order] * 1000, pred_max[order] * 1000,
                       color="#88aaff", alpha=0.5,
                       label="ensemble interval")
    ax[2].plot(z, pred_med[order] * 1000, color="#08306b", lw=0.7,
               label="ensemble median")
    ax[2].plot(z, obs_t[order] * 1000, "ro", ms=2.0, alpha=0.5,
               label="held-out observed")
    ax[2].set_xlabel("receiver depth (m)")
    ax[2].set_ylabel("pick time (ms)")
    ax[2].set_title("coverage band along depth")
    ax[2].grid(alpha=0.3)
    ax[2].legend(fontsize=8, loc="upper left")
    fig.suptitle("Volve 15/9-F-15A phase 4 EIKONAL held-out calibration "
                 f"({len(test_idx)} held out)",
                 fontsize=11, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_FIG, dpi=130)
    plt.close(fig)
    print(f"\nfigure: {OUT_FIG}")

    cert = {
        "label": "volve_15_9_F_15A_eikonal_holdout",
        "split": {
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "train_frac": float(len(train_idx) / picks.n()),
        },
        "ensemble": {
            "n_members": int(members_vp.shape[0]),
            "n_gn_iters_per_member": int(cfg.n_gn_iters),
            "training_rms_median_ms":
                float(np.median(meta["members_rms_s"]) * 1000),
        },
        "calibration": {
            "n_inside_interval": int(inside.sum()),
            "n_total": int(len(obs_t)),
            "inside_pct": float(100 * inside.mean()),
            "residual_mean_ms": float(resid.mean() * 1000),
            "residual_median_ms": float(np.median(resid) * 1000),
            "residual_std_ms": float(resid.std() * 1000),
            "residual_rms_ms": float(np.sqrt((resid ** 2).mean()) * 1000),
            "interval_width_mean_ms":
                float(interval_width.mean() * 1000),
            "interval_width_median_ms":
                float(np.median(interval_width) * 1000),
        },
        "scope_notes": [
            "Eikonal forward gives true refracted first-arrival times; "
            "no straight-line modeling bias remains.",
            "Held-out picks are predicted by re-running the eikonal "
            "forward on each ensemble member's Vp(z) profile.",
            "Compare with phase 3 (straight-ray) holdout: phase 3 had "
            "0/243 inside, +122 ms median residual, 19 ms interval "
            "width - dramatically under-calibrated.",
        ],
    }
    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_JSON).write_text(json.dumps(cert, indent=2))
    print(f"certificate: {OUT_JSON}")


if __name__ == "__main__":
    main()
