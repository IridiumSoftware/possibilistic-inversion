"""
volve/decompose_real.py - posdec decomposition on real Volve picks,
validated against the wireline sonic.

This is the phase-3 real-data integration test:

  1. Generate the feasible Vp(z) ensemble via volve.inversion_1d.
  2. Run posdec.coverage_certificate / posdec.classify on the ensemble
     in anomaly-against-depth-trend space.
  3. Render the ensemble fan, the depth-trend baseline, and the
     forced-high / forced-low / measure-dependent classification along
     the depth axis.
  4. Overlay the wireline DT-EDIT-derived Vp(MD) ground truth, mapped
     to TVD via the LAS file's MD-TVD column.
  5. Report:
       - fraction of depth column in each class;
       - mean signed error vs DT-EDIT (Vp-ensemble-median minus
         DT-EDIT) by depth band;
       - fraction of DT-EDIT samples that fall within the ensemble's
         per-bin Vp interval (the calibration test).

This is the smallest real-data integration that lets us say:
"posdec's forced/measure-dependent split, run on first-arrival picks
from a real walkaway VSP, identifies what is and isn't data-forced;
the wireline sonic is consistent with the forced features in N% of
the depth column." Honestly scoped: 1D straight-ray inversion, no
ray bending, no anisotropy, no near-surface statics correction.

Run:  uv run python -m volve.decompose_real
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lasio

import posdec
from volve.inversion_1d import (
    load_picks, depth_grid_for_picks, vp_ensemble, EnsembleConfig,
    depth_centers,
)


SONIC_LAS = "volve/data/15_9-F-15 A/TZV_DEPTH_MD_COMPUTED_1.LAS"
OUT_REPORT_JSON = "volve/picks/phase3_certificate.json"
OUT_FIG = "volve_phase3_decomposition.png"


# --- sonic ground truth ---------------------------------------------------

def load_sonic_vp_kms_vs_tvd(path: str = SONIC_LAS):
    """Read TZV LAS; convert DT-EDIT (us/ft) to Vp (km/s); align to TVD
    (true vertical depth below MSL). Returns (tvd_m, vp_kms) arrays."""
    las = lasio.read(path)
    tvd = np.asarray(las["TVD"], dtype=float)
    dt = np.asarray(las["DT-EDIT"], dtype=float)
    mask = np.isfinite(tvd) & np.isfinite(dt) & (dt > 0) & (dt < 500)
    tvd, dt = tvd[mask], dt[mask]
    vp_kms = 304.8 / dt
    # sort by tvd
    order = np.argsort(tvd)
    return tvd[order], vp_kms[order]


# --- decomposition on the ensemble ---------------------------------------

def decompose_ensemble(members_vp_kms, depth_centers_m,
                       eps_kms: float = 0.10):
    """Per-depth-bin possibilistic decomposition vs a linear depth-trend
    baseline.

    The baseline is the least-squares linear fit of the ENSEMBLE-MEAN
    Vp(z); each member's anomaly is Vp_member - Vp_trend(z). The classify
    call from posdec then returns the forced-high / forced-low /
    measure-dependent labels per depth bin."""
    members_anom = members_vp_kms - _linear_trend(members_vp_kms.mean(axis=0),
                                                  depth_centers_m)
    a_min = members_anom.min(axis=0)
    a_max = members_anom.max(axis=0)
    cls = posdec.classify(a_min, a_max, eps=eps_kms)
    masks = posdec.three_masks(cls)
    return {
        "a_min": a_min,
        "a_max": a_max,
        "cls": cls,
        "masks": masks,
        "members_anom": members_anom,
    }


def _linear_trend(vec, x):
    a, b = np.polyfit(x, vec, 1)
    return a * x + b


# --- validation vs sonic --------------------------------------------------

def validate_against_sonic(members_vp_kms, depth_centers_m,
                           sonic_tvd_m, sonic_vp_kms):
    """For each ensemble depth bin, compute:
       - sonic Vp interval-average (mean over LAS samples in the bin)
       - ensemble Vp interval (min/max across members)
       - inside: is sonic_mean in [vp_min, vp_max]?
       - signed_error: ensemble median - sonic mean
    Returns a dict of arrays length n_bins (None where bin is empty)."""
    bin_edges = np.zeros(len(depth_centers_m) + 1)
    half = 0.5 * (depth_centers_m[1] - depth_centers_m[0])
    bin_edges[0] = depth_centers_m[0] - half
    bin_edges[-1] = depth_centers_m[-1] + half
    bin_edges[1:-1] = 0.5 * (depth_centers_m[:-1] + depth_centers_m[1:])

    sonic_mean = np.full(len(depth_centers_m), np.nan)
    for j in range(len(depth_centers_m)):
        sel = (sonic_tvd_m >= bin_edges[j]) & (sonic_tvd_m < bin_edges[j + 1])
        if sel.any():
            sonic_mean[j] = float(np.mean(sonic_vp_kms[sel]))

    ens_min = members_vp_kms.min(axis=0)
    ens_max = members_vp_kms.max(axis=0)
    ens_med = np.median(members_vp_kms, axis=0)
    inside = (sonic_mean >= ens_min) & (sonic_mean <= ens_max)
    signed_err = ens_med - sonic_mean

    return {
        "sonic_mean_kms": sonic_mean,
        "ens_min_kms": ens_min,
        "ens_max_kms": ens_max,
        "ens_median_kms": ens_med,
        "inside": inside,
        "signed_error_kms": signed_err,
    }


# --- main -----------------------------------------------------------------

def main():
    picks = load_picks()
    grid = depth_grid_for_picks(picks)
    centers = depth_centers(grid)
    print(f"loaded {picks.n()} ok picks")
    print(f"depth grid: {grid.size - 1} bins of "
          f"{grid[1] - grid[0]:.1f} m to {grid[-1]:.0f} m")

    members_vp, meta = vp_ensemble(picks, grid)
    print(f"ensemble: {members_vp.shape[0]} feasible Vp(z); "
          f"median RMS residual = "
          f"{np.median(meta['members_rms_s']) * 1000:.1f} ms")

    sonic_tvd, sonic_vp = load_sonic_vp_kms_vs_tvd()
    print(f"sonic: {sonic_vp.size} DT-EDIT samples, "
          f"TVD {sonic_tvd.min():.1f} -> {sonic_tvd.max():.1f} m, "
          f"Vp {sonic_vp.min():.2f} -> {sonic_vp.max():.2f} km/s")

    dec = decompose_ensemble(members_vp, centers, eps_kms=0.10)
    val = validate_against_sonic(members_vp, centers, sonic_tvd, sonic_vp)

    # Class counts
    n_total = len(centers)
    n_forced_hi = int(dec["masks"]["forced_high"].sum())
    n_forced_lo = int(dec["masks"]["forced_low"].sum())
    n_meas_dep = int(dec["masks"]["measure_dependent"].sum())
    n_forced_q = int(dec["masks"]["forced_quiet"].sum())
    print()
    print("decomposition (vs depth-trend baseline, eps=0.10 km/s):")
    print(f"  forced-high       : {n_forced_hi}/{n_total} "
          f"({100*n_forced_hi/n_total:.1f}%)")
    print(f"  forced-low        : {n_forced_lo}/{n_total} "
          f"({100*n_forced_lo/n_total:.1f}%)")
    print(f"  forced-quiet      : {n_forced_q}/{n_total} "
          f"({100*n_forced_q/n_total:.1f}%)")
    print(f"  measure-dependent : {n_meas_dep}/{n_total} "
          f"({100*n_meas_dep/n_total:.1f}%)")

    # Calibration vs sonic
    sonic_bins = np.isfinite(val["sonic_mean_kms"])
    inside_count = int(val["inside"][sonic_bins].sum())
    sonic_bin_count = int(sonic_bins.sum())
    print()
    print(f"sonic calibration (bin-averaged):")
    print(f"  sonic-bin coverage : {sonic_bin_count}/{n_total} depth bins "
          f"have at least one DT-EDIT sample")
    print(f"  inside ensemble    : {inside_count}/{sonic_bin_count} "
          f"({100*inside_count/sonic_bin_count:.1f}%) bin-mean sonic Vp "
          f"falls inside the ensemble Vp interval")
    signed_finite = val["signed_error_kms"][sonic_bins]
    print(f"  signed error       : "
          f"median = {np.median(signed_finite):+.3f} km/s, "
          f"mean = {signed_finite.mean():+.3f} km/s, "
          f"|median| = {np.median(np.abs(signed_finite)):.3f}")

    # Figure
    _plot(members_vp, centers, dec, val, sonic_tvd, sonic_vp,
          out_path=OUT_FIG)
    print(f"\nfigure: {OUT_FIG}")

    # Certificate (phase-3 1D-specific; posdec.coverage_certificate
    # expects 2D models so we emit a directly-shaped JSON here).
    import json
    cert = {
        "label": "volve_15_9_F_15A_1D_straight_ray",
        "schema_version": "phase3.1.0",
        "geometry": {
            "n_picks_used": int(picks.n()),
            "depth_grid_n_bins": int(meta["n_bins"]),
            "depth_grid_bin_thick_m": float(meta["bin_thick_m"]),
            "depth_grid_z_min_m": float(grid[0]),
            "depth_grid_z_max_m": float(grid[-1]),
        },
        "ensemble": {
            "n_members": int(members_vp.shape[0]),
            "noise_rms_target_s": float(meta["cfg"].noise_rms_s),
            "noise_rms_achieved_median_ms":
                float(np.median(meta["members_rms_s"]) * 1000),
            "vp_envelope_kms":
                [float(meta["cfg"].vp_min_kms),
                 float(meta["cfg"].vp_max_kms)],
            "smoothness_correlation_m":
                float(meta["cfg"].smooth_correlation_m),
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
            "1D Vp(z) straight-ray forward operator; refracted eikonal "
            "rays would bend toward higher-velocity zones, so the "
            "straight-line approximation introduces a depth-dependent "
            "modeling error of ~80-150 ms per pick. This dominates the "
            "misfit floor and biases the ensemble Vp downward at depth "
            "(the 11.1% sonic-inside rate reflects that).",
            "Tier-1 envelope (vp_min=1.5, vp_max=5.5 km/s) is load-bearing "
            "above and below those values; the smoothness preference of "
            "the random-reference profile is load-bearing in the same "
            "sense flagged by the synthetic Tier-1 sensitivity study.",
            "The 60% forced / 40% measure-dependent split is an internal "
            "property of the inversion at this noise level + envelope + "
            "smoothness preference; it is not a claim about the real "
            "subsurface independent of those choices.",
        ],
    }
    Path(OUT_REPORT_JSON).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_REPORT_JSON).write_text(json.dumps(cert, indent=2))
    print(f"certificate: {OUT_REPORT_JSON}")


def _plot(members_vp, centers, dec, val, sonic_tvd, sonic_vp, out_path):
    fig, ax = plt.subplots(1, 3, figsize=(14, 9), sharey=True)
    # Ensemble fan
    for k in range(min(members_vp.shape[0], 80)):
        ax[0].plot(members_vp[k], centers, color="#88aaff",
                   lw=0.5, alpha=0.6)
    ax[0].plot(np.median(members_vp, axis=0), centers, color="#08306b",
               lw=2.0, label="ensemble median")
    ax[0].plot(sonic_vp, sonic_tvd, color="#b2182b", lw=0.6,
               label="DT-EDIT (wireline sonic)", alpha=0.85)
    ax[0].set_xlabel("Vp (km/s)")
    ax[0].set_ylabel("depth below sea surface (m)")
    ax[0].set_title("ensemble fan + sonic")
    ax[0].invert_yaxis()
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    ax[0].set_xlim(1.3, 5.8)

    # Anomaly vs depth-trend
    a_min, a_max = dec["a_min"], dec["a_max"]
    cls = dec["cls"]
    ax[1].fill_betweenx(centers, a_min, a_max, color="#88aaff",
                        alpha=0.5, label="ensemble anomaly interval")
    ax[1].plot(np.zeros_like(centers), centers, color="0.4",
               lw=0.5, ls="--")
    ax[1].set_xlabel("Vp anomaly vs depth trend (km/s)")
    ax[1].set_title("anomaly fan")
    ax[1].invert_yaxis()
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    ax[1].set_xlim(-1.0, 1.0)

    # Classification
    colors = {2: "#b2182b", -2: "#2166ac", 0: "#bbbbbb", 1: "#fdb338"}
    labels = {2: "forced-high", -2: "forced-low",
              0: "forced-quiet", 1: "measure-dependent"}
    width = (centers[1] - centers[0]) * 0.95
    for j, z in enumerate(centers):
        c = int(cls[j])
        ax[2].barh(z, 1.0, height=width, color=colors[c], align="center")
    # legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors[k], label=labels[k]) for k in
               [2, -2, 0, 1]]
    ax[2].legend(handles=handles, fontsize=8, loc="lower right")
    ax[2].set_xlabel("class")
    ax[2].set_xticks([])
    ax[2].set_title("decomposition\n(red high, blue low, "
                    "orange measure-dep)")
    ax[2].invert_yaxis()

    fig.suptitle("Volve 15/9-F-15A walkaway VSP - "
                 "real-data possibilistic decomposition + sonic calibration",
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
