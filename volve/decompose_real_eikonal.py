"""
volve/decompose_real_eikonal.py - PHASE 4 version of the real-data
decomposition + sonic validation. Same structure as decompose_real.py
but uses the EIKONAL Gauss-Newton ensemble instead of the straight-ray
LSQR ensemble. Direct head-to-head comparison.

Run:  uv run python -m volve.decompose_real_eikonal
"""

from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import posdec
from volve.inversion_1d import load_picks, depth_grid_for_picks, depth_centers
from volve.inversion_eikonal import vp_ensemble_eikonal, EikonalConfig
from volve.decompose_real import (
    load_sonic_vp_kms_vs_tvd, decompose_ensemble,
    validate_against_sonic, _linear_trend,
)


OUT_REPORT_JSON = "volve/picks/phase4_certificate.json"
OUT_FIG = "volve_phase4_decomposition.png"


def main():
    picks = load_picks()
    grid = depth_grid_for_picks(picks)
    centers = depth_centers(grid)
    print(f"loaded {picks.n()} ok picks")
    print(f"depth grid: {grid.size - 1} bins of "
          f"{grid[1] - grid[0]:.1f} m to {grid[-1]:.0f} m")

    cfg = EikonalConfig(n_members=30, n_gn_iters=3)
    members_vp, meta = vp_ensemble_eikonal(picks, grid, cfg)
    print(f"ensemble (eikonal): {members_vp.shape[0]} feasible Vp(z); "
          f"median RMS residual = "
          f"{np.median(meta['members_rms_s']) * 1000:.1f} ms")

    sonic_tvd, sonic_vp = load_sonic_vp_kms_vs_tvd()
    print(f"sonic: {sonic_vp.size} DT-EDIT samples, "
          f"TVD {sonic_tvd.min():.1f} -> {sonic_tvd.max():.1f} m")

    dec = decompose_ensemble(members_vp, centers, eps_kms=0.10)
    val = validate_against_sonic(members_vp, centers, sonic_tvd, sonic_vp)

    n_total = len(centers)
    n_forced_hi = int(dec["masks"]["forced_high"].sum())
    n_forced_lo = int(dec["masks"]["forced_low"].sum())
    n_meas_dep = int(dec["masks"]["measure_dependent"].sum())
    n_forced_q = int(dec["masks"]["forced_quiet"].sum())
    print()
    print("eikonal decomposition (vs depth-trend baseline, eps=0.10 km/s):")
    print(f"  forced-high       : {n_forced_hi}/{n_total} "
          f"({100*n_forced_hi/n_total:.1f}%)")
    print(f"  forced-low        : {n_forced_lo}/{n_total} "
          f"({100*n_forced_lo/n_total:.1f}%)")
    print(f"  forced-quiet      : {n_forced_q}/{n_total} "
          f"({100*n_forced_q/n_total:.1f}%)")
    print(f"  measure-dependent : {n_meas_dep}/{n_total} "
          f"({100*n_meas_dep/n_total:.1f}%)")

    sonic_bins = np.isfinite(val["sonic_mean_kms"])
    inside_count = int(val["inside"][sonic_bins].sum())
    sonic_bin_count = int(sonic_bins.sum())
    signed_finite = val["signed_error_kms"][sonic_bins]
    print()
    print("sonic calibration (bin-averaged):")
    print(f"  sonic-bin coverage : {sonic_bin_count}/{n_total}")
    print(f"  inside ensemble    : {inside_count}/{sonic_bin_count} "
          f"({100*inside_count/sonic_bin_count:.1f}%) "
          f"bin-mean sonic Vp inside ensemble interval")
    print(f"  signed error       : "
          f"median = {np.median(signed_finite):+.3f} km/s, "
          f"mean = {signed_finite.mean():+.3f} km/s, "
          f"|median| = {np.median(np.abs(signed_finite)):.3f}")

    _plot(members_vp, centers, dec, val, sonic_tvd, sonic_vp, OUT_FIG)
    print(f"\nfigure: {OUT_FIG}")

    cert = {
        "label": "volve_15_9_F_15A_eikonal_1D",
        "schema_version": "phase4.1.0",
        "geometry": {
            "n_picks_used": int(picks.n()),
            "depth_grid_n_bins": int(meta["n_bins"]),
            "depth_grid_bin_thick_m": float(meta["bin_thick_m"]),
            "fmm_grid": [int(meta["nz_grid"]), int(meta["nx_grid"])],
            "fmm_cell_m": float(meta["cell_m"]),
        },
        "ensemble": {
            "n_members": int(members_vp.shape[0]),
            "n_gn_iters_per_member": int(meta["cfg"].n_gn_iters),
            "noise_rms_target_s": float(meta["cfg"].noise_rms_s),
            "noise_rms_achieved_median_ms":
                float(np.median(meta["members_rms_s"]) * 1000),
            "vp_envelope_kms":
                [float(meta["cfg"].vp_min_kms),
                 float(meta["cfg"].vp_max_kms)],
        },
        "decomposition": {
            "eps_kms": 0.10,
            "forced_high_bins": int(n_forced_hi),
            "forced_low_bins": int(n_forced_lo),
            "forced_quiet_bins": int(n_forced_q),
            "measure_dependent_bins": int(n_meas_dep),
            "total_bins": int(n_total),
            "forced_high_pct": 100 * n_forced_hi / n_total,
            "forced_low_pct": 100 * n_forced_lo / n_total,
            "measure_dependent_pct": 100 * n_meas_dep / n_total,
        },
        "sonic_validation": {
            "sonic_source": "TZV_DEPTH_MD_COMPUTED_1.LAS / DT-EDIT",
            "sonic_bin_coverage": f"{sonic_bin_count}/{n_total}",
            "sonic_inside_ensemble_bins": int(inside_count),
            "sonic_inside_pct": float(100 * inside_count / sonic_bin_count),
            "signed_error_median_kms": float(np.median(signed_finite)),
            "signed_error_mean_kms": float(signed_finite.mean()),
            "abs_signed_error_median_kms":
                float(np.median(np.abs(signed_finite))),
        },
        "scope_notes": [
            "Eikonal forward via FMM on a 25 m cell grid; the lateral-"
            "translation-symmetry trick exploits Vp(z) being laterally "
            "uniform to need one FMM solve per ensemble member.",
            "Nonlinear Gauss-Newton inversion: 3 GN iterations per member "
            "with lambda bisected on the trial-step RMS each iteration.",
            "Compared to phase 3 (straight-ray, RMS median 174 ms, "
            "sonic-inside 11.1%, holdout 0%), the eikonal forward is "
            "expected to reduce the systematic bias from straight-line "
            "path lengths; see phase 4 holdout for the calibration result.",
        ],
    }
    Path(OUT_REPORT_JSON).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_REPORT_JSON).write_text(json.dumps(cert, indent=2))
    print(f"certificate: {OUT_REPORT_JSON}")


def _plot(members_vp, centers, dec, val, sonic_tvd, sonic_vp, out_path):
    fig, ax = plt.subplots(1, 3, figsize=(14, 9), sharey=True)
    for k in range(min(members_vp.shape[0], 80)):
        ax[0].plot(members_vp[k], centers, color="#88aaff",
                   lw=0.5, alpha=0.6)
    ax[0].plot(np.median(members_vp, axis=0), centers, color="#08306b",
               lw=2.0, label="ensemble median")
    ax[0].plot(sonic_vp, sonic_tvd, color="#b2182b", lw=0.6,
               label="DT-EDIT (wireline sonic)", alpha=0.85)
    ax[0].set_xlabel("Vp (km/s)")
    ax[0].set_ylabel("depth below sea surface (m)")
    ax[0].set_title("phase 4 eikonal: ensemble fan + sonic")
    ax[0].invert_yaxis(); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[0].set_xlim(1.3, 5.8)

    a_min, a_max = dec["a_min"], dec["a_max"]
    ax[1].fill_betweenx(centers, a_min, a_max, color="#88aaff",
                        alpha=0.5, label="ensemble anomaly interval")
    ax[1].plot(np.zeros_like(centers), centers, color="0.4", lw=0.5, ls="--")
    ax[1].set_xlabel("Vp anomaly vs depth trend (km/s)")
    ax[1].set_title("anomaly fan"); ax[1].invert_yaxis()
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8); ax[1].set_xlim(-1.0, 1.0)

    colors = {2: "#b2182b", -2: "#2166ac", 0: "#bbbbbb", 1: "#fdb338"}
    labels = {2: "forced-high", -2: "forced-low",
              0: "forced-quiet", 1: "measure-dependent"}
    cls = dec["cls"]
    width = (centers[1] - centers[0]) * 0.95
    for j, z in enumerate(centers):
        ax[2].barh(z, 1.0, height=width, color=colors[int(cls[j])],
                   align="center")
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors[k], label=labels[k]) for k in [2, -2, 0, 1]]
    ax[2].legend(handles=handles, fontsize=8, loc="lower right")
    ax[2].set_xlabel("class"); ax[2].set_xticks([])
    ax[2].set_title("decomposition"); ax[2].invert_yaxis()

    fig.suptitle("Volve 15/9-F-15A - phase 4 EIKONAL real-data decomposition",
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
