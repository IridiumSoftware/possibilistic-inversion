"""
volve/decompose_2d.py - phase 5 2D decomposition + multi-well sonic
validation + joint held-out calibration.

Runs the joint F-15A + F-11 T2 ensemble inversion, then:
  - 2D forced/measure-dependent decomposition on the (w, z) Vp anomaly
    against a depth-trend baseline;
  - validates against EACH well's wireline sonic separately
    (DT-EDIT from F-15A's TZV LAS; DT from F-11 T2's petrophysical LAS);
  - 80/20 joint-pick held-out calibration.

Run:  uv run python -m volve.decompose_2d
"""

from pathlib import Path
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lasio

import posdec
from volve.inversion_eikonal_2d import (
    load_joint_picks, make_grid, vp_ensemble_2d, Config2D,
    forward_2d, _unique_sources, ProjectedPicks,
)


F15A_LAS = "volve/data/15_9-F-15 A/TZV_DEPTH_MD_COMPUTED_1.LAS"
F11T2_LAS = "volve/data/15_9-F-11 T2/05.PETROPHYSICAL INTERPRETATION/" \
            "WLC_PETRO_COMPUTED_INPUT_1.LAS"
OUT_CERT = "volve/picks/phase5_certificate.json"
OUT_FIG = "volve_phase5_decomposition.png"
OUT_NPZ = "volve/picks/phase5_snapshot.npz"


def _vp_from_dt(las, dt_curve, depth_curve="MD"):
    """Return (depth_m, vp_kms) from a LAS curve, filtered to finite +
    physical."""
    depth = np.asarray(las[depth_curve], dtype=float)
    dt = np.asarray(las[dt_curve], dtype=float)
    m = np.isfinite(depth) & np.isfinite(dt) & (dt > 0) & (dt < 500)
    return depth[m], 304.8 / dt[m]


def load_sonic_f15a():
    las = lasio.read(F15A_LAS)
    tvd = np.asarray(las["TVD"], dtype=float)
    dt = np.asarray(las["DT-EDIT"], dtype=float)
    m = np.isfinite(tvd) & np.isfinite(dt) & (dt > 0) & (dt < 500)
    return tvd[m], 304.8 / dt[m]


def load_sonic_f11t2():
    """F-11 T2 has no TVD column - the LAS depth is MD. Use the
    checkshot to convert MD -> TVD. For phase 5 we assume the well
    is near-vertical (TVD ~= MD) above ~3 km; this introduces some
    error but is consistent with the 2D inversion's assumption."""
    las = lasio.read(F11T2_LAS)
    md = np.asarray(las["DEPTH"], dtype=float)
    dt = np.asarray(las["DT"], dtype=float)
    m = np.isfinite(md) & np.isfinite(dt) & (dt > 0) & (dt < 500)
    return md[m], 304.8 / dt[m]


def validate_against_sonic_2d(members_vp, grid, w_well, sonic_depth,
                              sonic_vp):
    """At a given well's w position, extract per-bin Vp interval from
    ensemble and the sonic mean. Returns dicts of arrays per z-bin."""
    # Find the grid w column for this well
    w_cell = int(round(grid.w_to_cell(w_well)))
    w_cell = max(0, min(grid.nw - 1, w_cell))
    z_centers = grid.z_centers()
    half = 0.5 * grid.cell_m
    # Ensemble Vp at this w column: (n_members, nz)
    ens_col = members_vp[:, :, w_cell]
    ens_min = ens_col.min(axis=0)
    ens_max = ens_col.max(axis=0)
    ens_med = np.median(ens_col, axis=0)
    sonic_mean = np.full(grid.nz, np.nan)
    for j in range(grid.nz):
        sel = (sonic_depth >= z_centers[j] - half) & \
              (sonic_depth < z_centers[j] + half)
        if sel.any():
            sonic_mean[j] = float(np.mean(sonic_vp[sel]))
    inside = (sonic_mean >= ens_min) & (sonic_mean <= ens_max)
    signed_err = ens_med - sonic_mean
    return {
        "w_cell": w_cell,
        "z_centers": z_centers,
        "ens_min": ens_min,
        "ens_max": ens_max,
        "ens_med": ens_med,
        "sonic_mean": sonic_mean,
        "inside": inside,
        "signed_error": signed_err,
    }


def decompose_2d(members_vp, grid, eps_kms=0.10):
    """Per-cell forced/measure-dependent classification vs a Z-trend
    baseline (linear fit of ensemble mean vs depth, same trend for
    every w column)."""
    ens_mean_z = members_vp.mean(axis=(0, 2))
    z = grid.z_centers()
    a, b = np.polyfit(z, ens_mean_z, 1)
    trend_z = a * z + b
    # anom shape (n_members, nz, nw)
    anom = members_vp - trend_z[None, :, None]
    a_min = anom.min(axis=0)
    a_max = anom.max(axis=0)
    cls = posdec.classify(a_min, a_max, eps=eps_kms)
    masks = posdec.three_masks(cls)
    return {
        "anom_min": a_min,
        "anom_max": a_max,
        "cls": cls,
        "masks": masks,
        "trend_z": trend_z,
    }


def holdout_2d(pp: ProjectedPicks, grid, members_vp, train_mask):
    """Predict held-out pick times across the ensemble. Return inside
    fraction + residual stats."""
    src_cells, src_inverse = _unique_sources(pp, grid)
    test_mask = ~train_mask
    test_idx = np.where(test_mask)[0]
    n_test = len(test_idx)
    n_mem = members_vp.shape[0]
    pred = np.zeros((n_test, n_mem), dtype=float)
    # Subset pp to test traces, keeping the sources from the joint pool
    pp_test = ProjectedPicks(
        src_w=pp.src_w[test_idx], src_z=pp.src_z[test_idx],
        rec_w=pp.rec_w[test_idx], rec_z=pp.rec_z[test_idx],
        times=pp.times[test_idx], well_id=pp.well_id[test_idx],
        f15a_wellhead_w=pp.f15a_wellhead_w,
        f11t2_wellhead_w=pp.f11t2_wellhead_w,
    )
    src_cells_t, src_inverse_t = _unique_sources(pp_test, grid)
    for k in range(n_mem):
        t_k, _ = forward_2d(members_vp[k], pp_test, grid,
                            src_cells_t, src_inverse_t,
                            compute_jacobian=False)
        pred[:, k] = t_k
    p_min = pred.min(axis=1)
    p_max = pred.max(axis=1)
    p_med = np.median(pred, axis=1)
    obs = pp.times[test_idx]
    inside = (obs >= p_min) & (obs <= p_max)
    resid = obs - p_med
    return {
        "n_test": n_test,
        "inside_count": int(inside.sum()),
        "inside_pct": float(100 * inside.mean()),
        "residual_mean_ms": float(resid.mean() * 1000),
        "residual_median_ms": float(np.median(resid) * 1000),
        "residual_rms_ms": float(np.sqrt((resid ** 2).mean()) * 1000),
        "interval_width_median_ms":
            float(np.median(p_max - p_min) * 1000),
    }


def main():
    t0 = time.time()
    pp, frame = load_joint_picks()
    print(f"joint picks: {pp.n()}")
    grid = make_grid(pp, frame)
    print(f"grid: {grid.nw} x {grid.nz} cells of {grid.cell_m:.0f} m")

    # Train/test split BEFORE inversion (consistent split for the
    # ensemble + the holdout test).
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

    cfg = Config2D(n_members=12, n_gn_iters=2)
    print(f"running ensemble: {cfg.n_members} members, "
          f"{cfg.n_gn_iters} GN iters")
    members_vp, meta = vp_ensemble_2d(pp_train, grid, cfg)
    elapsed_ens = time.time() - t0
    print(f"ensemble done in {elapsed_ens:.0f} s; "
          f"median RMS = "
          f"{np.median(meta['members_rms_s']) * 1000:.1f} ms")

    dec = decompose_2d(members_vp, grid, eps_kms=0.10)
    n_total = grid.nz * grid.nw
    n_forced_hi = int(dec["masks"]["forced_high"].sum())
    n_forced_lo = int(dec["masks"]["forced_low"].sum())
    n_meas_dep = int(dec["masks"]["measure_dependent"].sum())
    n_forced_q = int(dec["masks"]["forced_quiet"].sum())
    print()
    print("phase 5 2D decomposition:")
    print(f"  forced-high       : {n_forced_hi}/{n_total} "
          f"({100*n_forced_hi/n_total:.1f}%)")
    print(f"  forced-low        : {n_forced_lo}/{n_total} "
          f"({100*n_forced_lo/n_total:.1f}%)")
    print(f"  forced-quiet      : {n_forced_q}/{n_total} "
          f"({100*n_forced_q/n_total:.1f}%)")
    print(f"  measure-dependent : {n_meas_dep}/{n_total} "
          f"({100*n_meas_dep/n_total:.1f}%)")

    # Per-well sonic validation
    f15a_sonic = load_sonic_f15a()
    f11t2_sonic = load_sonic_f11t2()
    val_f15a = validate_against_sonic_2d(
        members_vp, grid, pp.f15a_wellhead_w,
        f15a_sonic[0], f15a_sonic[1])
    val_f11t2 = validate_against_sonic_2d(
        members_vp, grid, pp.f11t2_wellhead_w,
        f11t2_sonic[0], f11t2_sonic[1])

    def _val_report(name, val):
        ok_bins = np.isfinite(val["sonic_mean"])
        in_count = int(val["inside"][ok_bins].sum())
        n_ok = int(ok_bins.sum())
        signed = val["signed_error"][ok_bins]
        print(f"\nsonic validation at {name}:")
        print(f"  sonic-bin coverage : {n_ok}/{grid.nz}")
        print(f"  inside ensemble    : {in_count}/{n_ok} "
              f"({100 * in_count / n_ok:.1f}%)")
        print(f"  signed error       : "
              f"median = {np.median(signed):+.3f} km/s, "
              f"mean = {signed.mean():+.3f} km/s")
        return {
            "well": name,
            "sonic_bin_coverage": f"{n_ok}/{grid.nz}",
            "sonic_inside_count": int(in_count),
            "sonic_inside_pct": float(100 * in_count / n_ok),
            "signed_error_median_kms": float(np.median(signed)),
            "signed_error_mean_kms": float(signed.mean()),
        }
    s_f15a = _val_report("F-15A (w=%.0f m)" % pp.f15a_wellhead_w, val_f15a)
    s_f11t2 = _val_report(
        "F-11T2 (w=%.0f m)" % pp.f11t2_wellhead_w, val_f11t2)

    # Held-out calibration
    print()
    print("running holdout calibration...")
    holdout = holdout_2d(pp, grid, members_vp, train_mask)
    print("phase 5 joint held-out calibration:")
    print(f"  inside ensemble interval : "
          f"{holdout['inside_count']}/{holdout['n_test']} "
          f"({holdout['inside_pct']:.1f}%)")
    print(f"  residual median   = {holdout['residual_median_ms']:+.1f} ms")
    print(f"  residual mean     = {holdout['residual_mean_ms']:+.1f} ms")
    print(f"  residual RMS      = {holdout['residual_rms_ms']:.1f} ms")
    print(f"  interval width    = "
          f"{holdout['interval_width_median_ms']:.0f} ms median")

    # Figure
    _plot(members_vp, grid, dec, val_f15a, val_f11t2,
          f15a_sonic, f11t2_sonic, pp, OUT_FIG)
    print(f"\nfigure: {OUT_FIG}")

    cert = {
        "label": "volve_15_9_F_15A_F_11_T2_joint_eikonal_2d",
        "schema_version": "phase5.1.0",
        "geometry": {
            "n_picks_joint": int(pp.n()),
            "n_picks_f15a": int((pp.well_id == "f15a").sum()),
            "n_picks_f11t2": int((pp.well_id == "f11t2").sum()),
            "wellhead_distance_m": float(pp.f11t2_wellhead_w),
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
        },
        "decomposition_2d": {
            "eps_kms": 0.10,
            "forced_high_pct": float(100 * n_forced_hi / n_total),
            "forced_low_pct": float(100 * n_forced_lo / n_total),
            "measure_dependent_pct": float(100 * n_meas_dep / n_total),
            "forced_quiet_pct": float(100 * n_forced_q / n_total),
        },
        "sonic_validation_per_well": [s_f15a, s_f11t2],
        "joint_holdout_calibration": holdout,
        "scope_notes": [
            "2D Vp(w, z) joint inversion of both wells' picks; eikonal "
            "forward via FMM on a 40 m cell grid; lateral-translation-"
            "symmetry trick of phase 4 is not available (Vp now varies "
            "laterally). Cost scaled with number of UNIQUE source "
            "grid cells, which is ~32 (sources cluster).",
            "Per-well sonic validation samples the ensemble Vp at the "
            "well's w-column on the grid; this assumes the well bore is "
            "vertical at that w. For deviated wells this approximation "
            "spreads sonic error.",
            "F-11 T2 sonic LAS uses MD; for now we equate MD with TVD "
            "(near-vertical assumption above ~3 km), which contributes "
            "to that well's sonic-inside rate.",
        ],
    }
    Path(OUT_CERT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_CERT).write_text(json.dumps(cert, indent=2))
    print(f"certificate: {OUT_CERT}")

    # Snapshot for phase-A diagnostics (baseline sweep, sonic
    # trajectory, illumination, ensemble diversity).
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
    print(f"snapshot:    {OUT_NPZ}")
    print(f"\ntotal elapsed: {time.time() - t0:.0f} s")


def _plot(members_vp, grid, dec, val_f15a, val_f11t2,
          f15a_sonic, f11t2_sonic, pp, out_path):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.5, 1.0],
                          width_ratios=[1.0, 1.0, 1.0, 1.0],
                          hspace=0.28, wspace=0.32)

    z = grid.z_centers()
    w = grid.w_centers()
    ens_med = np.median(members_vp, axis=0)

    # (a) Vp(w,z) median map
    ax = fig.add_subplot(gs[0, 0:2])
    im = ax.imshow(ens_med, aspect="auto",
                   extent=[w[0], w[-1], z[-1], z[0]],
                   cmap="viridis", vmin=1.5, vmax=5.5)
    ax.axvline(pp.f15a_wellhead_w, color="white", lw=1.0, ls="--")
    ax.axvline(pp.f11t2_wellhead_w, color="white", lw=1.0, ls="--")
    ax.text(pp.f15a_wellhead_w, 100, "F-15A", color="white", fontsize=9)
    ax.text(pp.f11t2_wellhead_w, 100, "F-11T2", color="white", fontsize=9)
    ax.set_xlabel("along-well-line w (m)")
    ax.set_ylabel("depth (m)")
    ax.set_title("ensemble median Vp(w, z)")
    fig.colorbar(im, ax=ax, label="Vp (km/s)")

    # (b) classification
    ax = fig.add_subplot(gs[0, 2:4])
    cls = dec["cls"]
    cls_img = np.zeros_like(cls, dtype=float)
    cls_img[cls == 2] = 2
    cls_img[cls == -2] = -2
    cls_img[cls == 1] = 1
    cls_img[cls == 0] = 0
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(["#2166ac", "#bbbbbb", "#fdb338", "#b2182b"])
    norm = BoundaryNorm([-2.5, -1, 0.5, 1.5, 2.5], cmap.N)
    ax.imshow(cls_img, aspect="auto",
              extent=[w[0], w[-1], z[-1], z[0]],
              cmap=cmap, norm=norm)
    ax.axvline(pp.f15a_wellhead_w, color="white", lw=1.0, ls="--")
    ax.axvline(pp.f11t2_wellhead_w, color="white", lw=1.0, ls="--")
    ax.set_xlabel("along-well-line w (m)")
    ax.set_title("decomposition (red high, blue low, "
                 "orange meas-dep, grey quiet)")

    # (c, d) sonic validation at each well
    for col, (name, val, sonic) in enumerate([
        ("F-15A", val_f15a, f15a_sonic),
        ("F-11T2", val_f11t2, f11t2_sonic),
    ]):
        ax = fig.add_subplot(gs[1, 2 * col:2 * col + 2])
        z_c = val["z_centers"]
        ax.fill_betweenx(z_c, val["ens_min"], val["ens_max"],
                         color="#88aaff", alpha=0.5,
                         label="ensemble Vp interval")
        ax.plot(val["ens_med"], z_c, color="#08306b", lw=1.5,
                label="ensemble median")
        ax.plot(sonic[1], sonic[0], color="#b2182b", lw=0.5,
                label=f"{name} sonic")
        ax.invert_yaxis()
        ax.set_xlabel("Vp (km/s)")
        ax.set_ylabel("depth (m)")
        ax.set_title(f"{name} sonic vs ensemble at its w-column")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xlim(1.3, 5.8)

    fig.suptitle("Volve phase 5 - 2D joint F-15A + F-11 T2 eikonal "
                 "decomposition + multi-well sonic validation",
                 fontsize=12, weight="bold")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
