"""
volve/decompose_2d_b.py - PHASE B 2D joint inversion with the four
Phase-A corrections folded in.

Changes from phase 5 / volve.decompose_2d:

  1. Ensemble size 30 (was 12) and GN iter 3 (was 2). Compromise from
     the triad's 80 + 3 ask; ~75 min on a laptop, vs ~3.5 h at 80.
  2. Looser prior: smoothness correlation 700 m (was 350 m), Vp envelope
     1.3 - 6.0 km/s (was 1.5 - 5.5). Tests if the +0.3 km/s forced-low
     bias survives a wider feasible model class.
  3. Anomaly baseline = WIRELINE-SONIC LINEAR TREND (was ensemble-mean
     trend). Phase A2 showed the ensemble-mean baseline was
     tautological; the sonic trend is independent of the ensemble.
  4. Sonic comparison along BORE TRAJECTORY (was wellhead w-column).
     Phase A3 showed the deviation mismatch contributed ~20 pp of the
     10% sonic-inside drop.

If Phase B recovers calibration to phase-4-comparable levels, the
methodology fully recovers and phase 5's numbers were instrumentation
artifacts. If Phase B still under-performs by a wide margin, ChatGPT's
structural-misspecification hypothesis (2D for 3D Earth) is confirmed.

Run:  uv run python -m volve.decompose_2d_b
"""

from pathlib import Path
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import posdec
from volve.inversion_eikonal_2d import (
    load_joint_picks, make_grid, vp_ensemble_2d, Config2D,
    forward_2d, _unique_sources, ProjectedPicks,
)
from volve.decompose_2d import (
    load_sonic_f15a, load_sonic_f11t2,
)
from volve.phase5a import sonic_along_bore


OUT_CERT = "volve/picks/phaseB_certificate.json"
OUT_FIG = "volve_phaseB_decomposition.png"
OUT_NPZ = "volve/picks/phaseB_snapshot.npz"


def decompose_against_sonic_baseline(members_vp, grid,
                                     sonic_tvd, sonic_vp,
                                     eps_kms=0.10):
    """Per-cell forced/measure-dependent classification against the
    WIRELINE-SONIC linear trend, not the ensemble mean. Phase-A2 fix."""
    z = grid.z_centers()
    sm = np.isfinite(sonic_tvd) & np.isfinite(sonic_vp)
    a, b = np.polyfit(sonic_tvd[sm], sonic_vp[sm], 1)
    trend_z = a * z + b
    trend_2d = np.broadcast_to(trend_z[:, None], (grid.nz, grid.nw))
    anom = members_vp - trend_2d[None, ...]
    a_min = anom.min(axis=0)
    a_max = anom.max(axis=0)
    cls = posdec.classify(a_min, a_max, eps=eps_kms)
    masks = posdec.three_masks(cls)
    return {
        "trend_z": trend_z,
        "anom_min": a_min,
        "anom_max": a_max,
        "cls": cls,
        "masks": masks,
    }


def holdout_joint(pp, grid, members_vp, train_mask):
    """Predict held-out pick times across the ensemble via eikonal
    forward. Returns calibration stats."""
    test_idx = np.where(~train_mask)[0]
    n_test = len(test_idx)
    n_mem = members_vp.shape[0]
    pp_test = ProjectedPicks(
        src_w=pp.src_w[test_idx], src_z=pp.src_z[test_idx],
        rec_w=pp.rec_w[test_idx], rec_z=pp.rec_z[test_idx],
        times=pp.times[test_idx], well_id=pp.well_id[test_idx],
        f15a_wellhead_w=pp.f15a_wellhead_w,
        f11t2_wellhead_w=pp.f11t2_wellhead_w,
    )
    src_cells_t, src_inv_t = _unique_sources(pp_test, grid)
    pred = np.zeros((n_test, n_mem), dtype=float)
    for k in range(n_mem):
        t_k, _ = forward_2d(members_vp[k], pp_test, grid,
                            src_cells_t, src_inv_t,
                            compute_jacobian=False)
        pred[:, k] = t_k
    pmin = pred.min(axis=1)
    pmax = pred.max(axis=1)
    pmed = np.median(pred, axis=1)
    obs = pp.times[test_idx]
    inside = (obs >= pmin) & (obs <= pmax)
    resid = obs - pmed
    return {
        "n_test": int(n_test),
        "inside_count": int(inside.sum()),
        "inside_pct": float(100 * inside.mean()),
        "residual_mean_ms": float(resid.mean() * 1000),
        "residual_median_ms": float(np.median(resid) * 1000),
        "residual_rms_ms": float(np.sqrt((resid ** 2).mean()) * 1000),
        "interval_width_median_ms":
            float(np.median(pmax - pmin) * 1000),
    }


def main():
    t0 = time.time()
    pp, frame = load_joint_picks()
    print(f"joint picks: {pp.n()}")
    grid = make_grid(pp, frame)
    print(f"grid: {grid.nw} x {grid.nz} cells of {grid.cell_m:.0f} m")

    # Train/test split (same seed as phase 5 for direct comparison)
    rng = np.random.default_rng(20260608)
    train_mask = rng.uniform(size=pp.n()) < 0.80
    pp_train = ProjectedPicks(
        src_w=pp.src_w[train_mask], src_z=pp.src_z[train_mask],
        rec_w=pp.rec_w[train_mask], rec_z=pp.rec_z[train_mask],
        times=pp.times[train_mask], well_id=pp.well_id[train_mask],
        f15a_wellhead_w=pp.f15a_wellhead_w,
        f11t2_wellhead_w=pp.f11t2_wellhead_w,
    )
    print(f"split: {int(train_mask.sum())} train / "
          f"{int((~train_mask).sum())} test")

    # Phase B config: looser prior, more members, more GN iters.
    cfg = Config2D(
        n_members=30,
        n_gn_iters=3,
        cell_m=40.0,
        noise_rms_s=0.030,
        vp_min_kms=1.3,                   # was 1.5
        vp_max_kms=6.0,                   # was 5.5
        smooth_correlation_m=700.0,       # was 350
        lam_lo=100.0,
        lam_hi=1e6,
        bisect_iters=12,                  # was 20; lambda converges fast
        bisect_tol=1.25,                  # was 1.20; allow a little slack
        seed=20260608,
    )
    print(f"running ensemble: {cfg.n_members} members, "
          f"{cfg.n_gn_iters} GN iters, "
          f"prior smoothness {cfg.smooth_correlation_m:.0f} m, "
          f"envelope {cfg.vp_min_kms}-{cfg.vp_max_kms} km/s")
    members_vp, meta = vp_ensemble_2d(pp_train, grid, cfg)
    elapsed_ens = time.time() - t0
    print(f"ensemble done in {elapsed_ens:.0f} s; "
          f"median RMS = "
          f"{np.median(meta['members_rms_s']) * 1000:.1f} ms")

    # Decomposition with wireline-sonic baseline.
    sonic_f15a = load_sonic_f15a()
    sonic_f11t2 = load_sonic_f11t2()
    sonic_all_tvd = np.concatenate([sonic_f15a[0], sonic_f11t2[0]])
    sonic_all_vp = np.concatenate([sonic_f15a[1], sonic_f11t2[1]])
    dec = decompose_against_sonic_baseline(
        members_vp, grid, sonic_all_tvd, sonic_all_vp)
    n_total = grid.nz * grid.nw
    n_fh = int((dec["cls"] == 2).sum())
    n_fl = int((dec["cls"] == -2).sum())
    n_fq = int((dec["cls"] == 0).sum())
    n_md = int((dec["cls"] == 1).sum())
    print()
    print("phase B decomposition (vs wireline-sonic linear trend, "
          "eps=0.10 km/s):")
    print(f"  forced-high       : {n_fh}/{n_total} "
          f"({100*n_fh/n_total:.1f}%)")
    print(f"  forced-low        : {n_fl}/{n_total} "
          f"({100*n_fl/n_total:.1f}%)")
    print(f"  forced-quiet      : {n_fq}/{n_total} "
          f"({100*n_fq/n_total:.1f}%)")
    print(f"  measure-dependent : {n_md}/{n_total} "
          f"({100*n_md/n_total:.1f}%)")

    # Snapshot for diagnostics
    snap_dict = {
        "members_vp": members_vp,
        "grid": grid,
        "cls": dec["cls"],
        "anom_min": dec["anom_min"],
        "anom_max": dec["anom_max"],
        "trend_z": dec["trend_z"],
        "train_mask": train_mask,
        "f15a_w": pp.f15a_wellhead_w,
        "f11t2_w": pp.f11t2_wellhead_w,
    }

    # Sonic validation along bore trajectories (phase A3 fix)
    val_f15a = sonic_along_bore(
        snap_dict, pp, "f15a", sonic_f15a[0], sonic_f15a[1])
    val_f11t2 = sonic_along_bore(
        snap_dict, pp, "f11t2", sonic_f11t2[0], sonic_f11t2[1])
    print()
    print("sonic-along-bore validation:")
    print(f"  F-15A : {val_f15a['inside_count']}/{val_f15a['n_points']} "
          f"({val_f15a['inside_pct']:.1f}%) inside")
    print(f"  F-11T2: {val_f11t2['inside_count']}/{val_f11t2['n_points']} "
          f"({val_f11t2['inside_pct']:.1f}%) inside")

    # Held-out calibration
    print()
    print("running holdout calibration...")
    holdout = holdout_joint(pp, grid, members_vp, train_mask)
    print(f"  inside ensemble interval : "
          f"{holdout['inside_count']}/{holdout['n_test']} "
          f"({holdout['inside_pct']:.1f}%)")
    print(f"  residual median   = {holdout['residual_median_ms']:+.1f} ms")
    print(f"  residual RMS      = {holdout['residual_rms_ms']:.1f} ms")
    print(f"  interval width    = "
          f"{holdout['interval_width_median_ms']:.0f} ms median")

    # Figure
    _plot(snap_dict, pp, val_f15a, val_f11t2, sonic_f15a, sonic_f11t2,
          dec, holdout, OUT_FIG)
    print(f"\nfigure: {OUT_FIG}")

    # Snapshot to npz (for any further phase B' diagnostics)
    np.savez_compressed(
        OUT_NPZ,
        members_vp_kms=members_vp,
        grid_cell_m=np.array([grid.cell_m]),
        grid_w0=np.array([grid.w0]),
        grid_z0=np.array([grid.z0]),
        grid_nw=np.array([grid.nw]),
        grid_nz=np.array([grid.nz]),
        cls=dec["cls"],
        anom_min=dec["anom_min"],
        anom_max=dec["anom_max"],
        trend_z=dec["trend_z"],
        train_mask=train_mask,
        f15a_wellhead_w=np.array([pp.f15a_wellhead_w]),
        f11t2_wellhead_w=np.array([pp.f11t2_wellhead_w]),
    )
    print(f"snapshot: {OUT_NPZ}")

    # Certificate
    cert = {
        "label": "volve_15_9_F_15A_F_11_T2_phaseB",
        "schema_version": "phaseB.1.0",
        "phaseA_corrections_applied": [
            "anomaly baseline = wireline-sonic linear trend "
            "(phase A2 fix)",
            "sonic validation along actual bore trajectory "
            "(phase A3 fix)",
            "n_members raised 12 -> 30 (toward triad's 80 ask)",
            "n_gn_iters raised 2 -> 3",
            "looser prior: smooth corr 350 -> 700 m, "
            "envelope 1.5-5.5 -> 1.3-6.0 km/s",
        ],
        "geometry": {
            "n_picks_joint": int(pp.n()),
            "grid_nz": int(grid.nz),
            "grid_nw": int(grid.nw),
            "grid_cell_m": float(grid.cell_m),
        },
        "ensemble": {
            "n_members": int(members_vp.shape[0]),
            "n_gn_iters_per_member": int(cfg.n_gn_iters),
            "noise_rms_target_s": float(cfg.noise_rms_s),
            "noise_rms_achieved_median_ms":
                float(np.median(meta["members_rms_s"]) * 1000),
            "elapsed_seconds": float(elapsed_ens),
            "smooth_correlation_m": float(cfg.smooth_correlation_m),
            "vp_envelope_kms":
                [float(cfg.vp_min_kms), float(cfg.vp_max_kms)],
        },
        "decomposition_vs_sonic_baseline": {
            "eps_kms": 0.10,
            "forced_high_pct": float(100 * n_fh / n_total),
            "forced_low_pct": float(100 * n_fl / n_total),
            "forced_quiet_pct": float(100 * n_fq / n_total),
            "measure_dependent_pct": float(100 * n_md / n_total),
        },
        "sonic_along_bore": {
            "f15a": {
                "n_points": val_f15a["n_points"],
                "inside_count": val_f15a["inside_count"],
                "inside_pct": val_f15a["inside_pct"],
            },
            "f11t2": {
                "n_points": val_f11t2["n_points"],
                "inside_count": val_f11t2["inside_count"],
                "inside_pct": val_f11t2["inside_pct"],
            },
        },
        "holdout_calibration": holdout,
    }
    Path(OUT_CERT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_CERT).write_text(json.dumps(cert, indent=2))
    print(f"certificate: {OUT_CERT}")
    print(f"\ntotal elapsed: {time.time() - t0:.0f} s")


def _plot(snap, pp, val_f15a, val_f11t2, sonic_f15a, sonic_f11t2,
          dec, holdout, out_path):
    fig, ax = plt.subplots(2, 2, figsize=(15, 11))

    grid = snap["grid"]
    members = snap["members_vp"]

    # Top-left: 2D Vp ensemble median + measure-dep contour
    ens_med = np.median(members, axis=0)
    w_axis = grid.w_centers()
    z_axis = grid.z_centers()
    im = ax[0, 0].imshow(ens_med, aspect="auto",
                         extent=[w_axis[0], w_axis[-1],
                                 z_axis[-1], z_axis[0]],
                         cmap="viridis", vmin=1.5, vmax=5.5)
    ax[0, 0].axvline(snap["f15a_w"], color="white", lw=0.8, ls="--")
    ax[0, 0].axvline(snap["f11t2_w"], color="white", lw=0.8, ls="--")
    ax[0, 0].set_xlabel("w (m)")
    ax[0, 0].set_ylabel("depth (m)")
    ax[0, 0].set_title("ensemble median Vp(w, z)")
    fig.colorbar(im, ax=ax[0, 0], label="Vp (km/s)")

    # Top-right: classification (wireline baseline)
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(["#2166ac", "#bbbbbb", "#fdb338", "#b2182b"])
    norm = BoundaryNorm([-2.5, -1, 0.5, 1.5, 2.5], cmap.N)
    ax[0, 1].imshow(dec["cls"], aspect="auto",
                    extent=[w_axis[0], w_axis[-1],
                            z_axis[-1], z_axis[0]],
                    cmap=cmap, norm=norm)
    ax[0, 1].axvline(snap["f15a_w"], color="white", lw=0.8, ls="--")
    ax[0, 1].axvline(snap["f11t2_w"], color="white", lw=0.8, ls="--")
    ax[0, 1].set_xlabel("w (m)")
    ax[0, 1].set_title("decomposition vs WIRELINE-SONIC baseline\n"
                       "(red high, blue low, orange MD, grey quiet)")

    # Bottom-left: F-15A sonic along bore
    z = val_f15a["bore_z"]
    ax[1, 0].fill_betweenx(z, val_f15a["ens_min"], val_f15a["ens_max"],
                           color="#88aaff", alpha=0.5,
                           label="ensemble at bore")
    ax[1, 0].plot(val_f15a["ens_med"], z, color="#08306b", lw=1.5,
                  label="ensemble median")
    ax[1, 0].plot(val_f15a["sonic_at_z_kms"], z, color="#b2182b",
                  lw=1.0, label="sonic @ bore z")
    ax[1, 0].invert_yaxis()
    ax[1, 0].set_xlabel("Vp (km/s)")
    ax[1, 0].set_ylabel("depth (m)")
    ax[1, 0].set_title(f"F-15A sonic along bore\n"
                       f"inside = {val_f15a['inside_pct']:.1f}%")
    ax[1, 0].grid(alpha=0.3); ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_xlim(1.3, 5.8)

    # Bottom-right: F-11T2 sonic along bore
    z = val_f11t2["bore_z"]
    ax[1, 1].fill_betweenx(z, val_f11t2["ens_min"], val_f11t2["ens_max"],
                           color="#88aaff", alpha=0.5,
                           label="ensemble at bore")
    ax[1, 1].plot(val_f11t2["ens_med"], z, color="#08306b", lw=1.5,
                  label="ensemble median")
    ax[1, 1].plot(val_f11t2["sonic_at_z_kms"], z, color="#b2182b",
                  lw=1.0, label="sonic @ bore z")
    ax[1, 1].invert_yaxis()
    ax[1, 1].set_xlabel("Vp (km/s)")
    ax[1, 1].set_ylabel("depth (m)")
    ax[1, 1].set_title(f"F-11T2 sonic along bore\n"
                       f"inside = {val_f11t2['inside_pct']:.1f}%")
    ax[1, 1].grid(alpha=0.3); ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_xlim(1.3, 5.8)

    fig.suptitle("Phase B - 2D joint with phase-A corrections "
                 f"(holdout inside = {holdout['inside_pct']:.1f}%)",
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
