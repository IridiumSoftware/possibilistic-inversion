"""
coverage_diagnostics.py - standalone coverage + diagnostics layer for the
possibilistic decomposition (ORSI propagation steps #2, #4, #6).

INPUT (caller's responsibility):
  * ensemble      : list[(NZ,NX) ndarrays]    each a feasible velocity model
  * bg            : (NZ,NX) ndarray            reference background
  * eps           : float                      sign deadband (km/s)
  * coverage_curve (optional) : dict from RWC-1   Ns, forced_sizes
  * false_forced_rate (optional) : float from RWC-2

OUTPUT:
  * coverage_certificate(...) : dict (JSON-serializable) - the production
    metadata that accompanies every forced/measure-dependent map. Replaces
    the post-hoc-disclaimer pattern with a first-class certificate.
  * plot_three_maps_and_width(...) : the standard reporting figure
    (forced-high / forced-low / measure-dependent masks + interval-width
    map + coverage curve).

DESIGN INTENT.
  * Forward-operator agnostic. This module solves no inversion; it takes
    an ensemble the caller has already certified feasible. The decomposition
    layer's transport invariance is what makes this module possible.
  * Standalone. No imports from synthetic_demo or eikonal modules; numpy +
    matplotlib + stdlib only. The Tier B "modular library" move (ORSI #5)
    just packages this file with an entry point.

Conventions in force:
  * Models indexed [NZ, NX]; per-cell anomaly = model - bg; units km/s.
  * Pairwise distance: RMS over cells (km/s).
  * Smoothness: RMS gradient magnitude (km/s per grid step).
  * Principal-direction diameter is computed by SVD of the centered ensemble
    matrix - the directions the sampler EXPLORED. The true data-null
    diameter of the forward operator G is operator-dependent and not
    computed here by design; the explored-direction diameter is a lower
    bound on the actual feasible-set diameter.

API stability: STABLE for the certificate dict's top-level keys. The plot
function is EXPERIMENTAL (figure layout may iterate).
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----- decomposition primitives (transport-invariant) -----------------------

def feasible_interval(ensemble, bg):
    """Per-cell [a_min, a_max] over the ensemble's anomaly (model - bg)."""
    anom = np.array([m - bg for m in ensemble])
    return anom.min(axis=0), anom.max(axis=0)


def interval_width(a_min, a_max):
    return a_max - a_min


def classify(a_min, a_max, eps):
    """Forced-sign decomposition:
        +2  forced-high          a_min > +eps
        -2  forced-low           a_max < -eps
         0  forced-quiet         interval within +/-eps
        +1  measure-dependent    interval straddles - sign not data-forced.
    Identical to synthetic_demo.classify but expressed here so this module
    has no dependency on the demos."""
    out = np.full(a_min.shape, 1, int)
    out[a_min > eps] = 2
    out[a_max < -eps] = -2
    out[(a_min >= -eps) & (a_max <= eps)] = 0
    return out


def three_masks(cls):
    return {
        "forced_high": cls == 2,
        "forced_low": cls == -2,
        "forced_quiet": cls == 0,
        "measure_dependent": cls == 1,
    }


# ----- ensemble diagnostics (ORSI #4) ---------------------------------------

def pairwise_distances(ensemble):
    """Symmetric matrix of RMS pairwise model distances (km/s).
    Diameter of the sampled feasible set = max off-diagonal entry."""
    arr = np.stack([m.ravel() for m in ensemble])
    diff = arr[:, None, :] - arr[None, :, :]
    return np.sqrt((diff ** 2).mean(axis=-1))


def feasible_set_diameter(ensemble):
    D = pairwise_distances(ensemble)
    return float(D.max())


def smoothness(model):
    """RMS gradient magnitude. Larger => rougher. Units: km/s per grid step."""
    gz, gx = np.gradient(model)
    return float(np.sqrt((gz ** 2 + gx ** 2).mean()))


def smoothness_stats(ensemble):
    vals = np.array([smoothness(m) for m in ensemble])
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def explored_directions(ensemble):
    """SVD of the centered ensemble matrix. Returns the singular spectrum,
    explained-variance ratios, and the max-min extent of the projections
    onto the top explored directions (RMS-normalized over cells, so
    directly comparable to feasible_set_diameter_kms_rms).

    Captures only directions the sampler ACTUALLY visited; basins the
    sampler missed contribute nothing. RWC-2 covers that gap."""
    arr = np.stack([m.ravel() for m in ensemble])
    centered = arr - arr.mean(axis=0, keepdims=True)
    U, s, _ = np.linalg.svd(centered, full_matrices=False)
    total = float((s ** 2).sum())
    evr = (s ** 2 / total).tolist() if total > 0 else [0.0] * len(s)
    # Projection of member i onto direction k is (U @ diag(s))[i, k]; its
    # units are km/s * sqrt(D), where D = #cells (because v_k is a unit
    # vector in R^D). Divide by sqrt(D) to get an RMS-per-cell extent
    # comparable to the pairwise-distance diameter.
    proj = U * s
    D_cells = arr.shape[1]
    extents = ((proj.max(axis=0) - proj.min(axis=0)) /
               np.sqrt(D_cells)).tolist() if D_cells > 0 else []
    return {
        "singular_values": s.tolist(),
        "explained_variance_ratio": evr,
        "principal_diameter_kms_rms":
            float(extents[0]) if extents else 0.0,
        "top_k_diameters_kms_rms": extents[:5],
    }


# ----- certificate (ORSI #2) ------------------------------------------------

def coverage_certificate(ensemble, bg, eps,
                         coverage_curve=None,
                         false_forced_rate=None,
                         label=None):
    """Build the certificate dict that accompanies every decomposition.
    A forced/measure-dependent map is only as trustworthy as this metadata."""
    a_min, a_max = feasible_interval(ensemble, bg)
    cls = classify(a_min, a_max, eps)
    masks = three_masks(cls)
    width = interval_width(a_min, a_max)
    smooth = smoothness_stats(ensemble)
    explored = explored_directions(ensemble)
    return {
        "label": label,
        "ensemble_size": int(len(ensemble)),
        "grid_shape": list(ensemble[0].shape),
        "eps_deadband_kms": float(eps),
        "decomposition": {
            "forced_high_cells": int(masks["forced_high"].sum()),
            "forced_low_cells": int(masks["forced_low"].sum()),
            "forced_quiet_cells": int(masks["forced_quiet"].sum()),
            "measure_dependent_cells": int(masks["measure_dependent"].sum()),
        },
        "interval_width_kms": {
            "mean": float(width.mean()),
            "median": float(np.median(width)),
            "max": float(width.max()),
        },
        "ensemble_diagnostics": {
            "feasible_set_diameter_kms_rms": feasible_set_diameter(ensemble),
            "principal_diameter_kms_rms":
                explored["principal_diameter_kms_rms"],
            "leading_singular_values": explored["singular_values"][:5],
            "leading_explained_variance_ratio":
                explored["explained_variance_ratio"][:5],
            "smoothness_kms_per_step": smooth,
        },
        "coverage": {
            "rwc1_curve": coverage_curve,
            "rwc1_stabilized": _stabilization_point(coverage_curve),
            "rwc2_false_forced_rate": false_forced_rate,
            "rwc2_status": _rwc2_status(false_forced_rate),
        },
        "schema_version": "1.0",
    }


def _stabilization_point(curve, plateau_tol=0.02, window=3):
    """N at which the forced set has converged.

    Prefer the curve's own `stabilization_N` field (RWC-1 owns the criterion:
    smallest N at which within-resolution false-forced first clears <=1 cell).
    Fall back to a forced-size plateau heuristic for arbitrary curves that
    don't supply that field: smallest N at which forced size stays within
    `plateau_tol` of its final value over `window` consecutive samples."""
    if not curve:
        return None
    if curve.get("stabilization_N") is not None:
        return int(curve["stabilization_N"])
    Ns = curve.get("Ns")
    sizes = curve.get("forced_sizes")
    if not Ns or not sizes or len(Ns) != len(sizes) or len(Ns) < window:
        return None
    final = sizes[-1]
    if final <= 0:
        return None
    rel = [abs(s - final) / final for s in sizes]
    for i in range(len(Ns) - window + 1):
        if all(r <= plateau_tol for r in rel[i:i + window]):
            return int(Ns[i])
    return None


def _rwc2_status(false_forced_rate):
    if false_forced_rate is None:
        return None
    if false_forced_rate <= 0.05:
        return "ok"        # under-5% of forced labels demonstrably wrong
    if false_forced_rate <= 0.15:
        return "warn"      # report-with-caveat band
    return "fail"          # too many false-forced; sampler under-covers


def write_certificate(cert, path):
    Path(path).write_text(json.dumps(cert, indent=2))
    return path


def read_json_if_present(path):
    """Tolerant loader for the RWC sidecars. Returns None if missing."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# ----- standard reporting figure (ORSI #6) ----------------------------------

def plot_three_maps_and_width(a_min, a_max, eps,
                              coverage_curve=None,
                              false_forced_rate=None,
                              out_path=None,
                              title=None):
    """Standard production figure: three forced-sign masks + interval-width
    map + coverage curve. The forced/measure-dependent split is a single
    artifact, not three separate plots; the width map shows the magnitude
    of measure-dependence everywhere; the coverage curve states how
    trustworthy the whole thing is."""
    cls = classify(a_min, a_max, eps)
    width = interval_width(a_min, a_max)
    forced_hi = cls == 2
    forced_lo = cls == -2
    meas_dep = cls == 1

    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0],
                          hspace=0.32, wspace=0.22)

    # Top row: three masks.
    for col, (mask, name, cmap) in enumerate([
        (forced_hi, "forced-high",        "Reds"),
        (forced_lo, "forced-low",         "Blues"),
        (meas_dep, "measure-dependent",   "Oranges"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(mask, cmap=cmap, origin="upper", vmin=0, vmax=1)
        ax.set_title(f"{name}  ({int(mask.sum())} cells)", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    # Bottom-left/middle: width map.
    ax_w = fig.add_subplot(gs[1, 0:2])
    im = ax_w.imshow(width, cmap="viridis", origin="upper")
    ax_w.set_title("feasible-interval width  (a_max - a_min, km/s)",
                   fontsize=11)
    ax_w.set_xticks([]); ax_w.set_yticks([])
    fig.colorbar(im, ax=ax_w, fraction=0.030, pad=0.02)

    # Bottom-right: coverage curve.
    ax_c = fig.add_subplot(gs[1, 2])
    if coverage_curve and coverage_curve.get("Ns") \
            and coverage_curve.get("forced_sizes"):
        ax_c.plot(coverage_curve["Ns"], coverage_curve["forced_sizes"],
                  "-o", color="#b2182b", lw=1.4, ms=3,
                  label="forced-set size")
        if coverage_curve.get("false_forced_res") is not None:
            ax_c.plot(coverage_curve["Ns"],
                      coverage_curve["false_forced_res"],
                      "-s", color="#444", lw=1.0, ms=2.5,
                      label="false-forced (resolved)")
        ax_c.set_xlabel("ensemble size N", fontsize=9)
        ax_c.set_ylabel("cell count", fontsize=9)
        ax_c.set_title("coverage curve (RWC-1)", fontsize=11)
        ax_c.legend(fontsize=8, loc="best", frameon=False)
        ax_c.grid(alpha=0.3)
    else:
        ax_c.text(0.5, 0.55,
                  "coverage curve\nnot supplied",
                  transform=ax_c.transAxes, ha="center", va="center",
                  color="#888", fontsize=10)
        ax_c.set_title("coverage curve (RWC-1)", fontsize=11)
        ax_c.set_xticks([]); ax_c.set_yticks([])

    if false_forced_rate is not None:
        ax_c.text(0.02, -0.18,
                  f"RWC-2 false-forced rate: "
                  f"{100 * false_forced_rate:.1f}%",
                  transform=ax_c.transAxes,
                  fontsize=9, color="#444",
                  verticalalignment="top")

    if title:
        fig.suptitle(title, fontsize=12)
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ----- CLI smoke check ------------------------------------------------------

def _smoke():
    """N=1 self-check: generate a toy ensemble + bg, run the full pipeline.
    Verifies the module imports cleanly, the certificate serializes to JSON,
    and the standard figure renders."""
    rng = np.random.default_rng(0)
    NZ, NX = 8, 10
    bg = np.full((NZ, NX), 5.0)
    members = []
    for _ in range(12):
        m = bg + 0.5 * rng.standard_normal((NZ, NX))
        m[1:3, 1:4] += 1.0          # forced-high region
        m[5:7, 6:9] -= 1.0          # forced-low region
        members.append(m)
    cert = coverage_certificate(
        members, bg, eps=0.1,
        coverage_curve={"Ns": [3, 6, 9, 12],
                        "forced_sizes": [5, 8, 9, 9],
                        "false_forced_res": [1, 0, 0, 0]},
        false_forced_rate=0.04,
        label="smoke",
    )
    write_certificate(cert, "coverage_diagnostics_smoke.json")
    a_min, a_max = feasible_interval(members, bg)
    plot_three_maps_and_width(
        a_min, a_max, eps=0.1,
        coverage_curve={"Ns": [3, 6, 9, 12],
                        "forced_sizes": [5, 8, 9, 9],
                        "false_forced_res": [1, 0, 0, 0]},
        false_forced_rate=0.04,
        out_path="coverage_diagnostics_smoke.png",
        title="coverage_diagnostics smoke",
    )
    print("SMOKE ensemble_size:", cert["ensemble_size"])
    print("SMOKE forced_high:", cert["decomposition"]["forced_high_cells"])
    print("SMOKE rwc1_stabilized:", cert["coverage"]["rwc1_stabilized"])
    print("SMOKE rwc2_status:", cert["coverage"]["rwc2_status"])
    print("SMOKE certificate:", "coverage_diagnostics_smoke.json")
    print("SMOKE figure:     ", "coverage_diagnostics_smoke.png")


if __name__ == "__main__":
    _smoke()
