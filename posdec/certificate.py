"""
posdec.certificate - coverage certificate (ORSI propagation #2).

The certificate is the production metadata that accompanies every forced /
measure-dependent map. It is a JSON-serializable dict: decomposition counts,
interval-width stats, ensemble diagnostics, and RWC-1 / RWC-2 coverage
status. Schema-versioned so downstream consumers can pin a version.

The coverage curve sidecar (RWC-1) and false-forced-rate sidecar (RWC-2)
are optional; the certificate degrades gracefully when they are absent.
"""

import json
from pathlib import Path

import numpy as np

from posdec.decomposition import (
    feasible_interval,
    interval_width,
    classify,
    three_masks,
)
from posdec.diagnostics import (
    feasible_set_diameter,
    smoothness_stats,
    explored_directions,
)


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
    Fall back to a forced-size plateau heuristic for arbitrary curves."""
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
        return "ok"
    if false_forced_rate <= 0.15:
        return "warn"
    return "fail"


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


