"""porotomo/sensitivity_3d.py - smoothness-class sensitivity of the
PoroTomo forced labels (ORSI Tier-1 protocol in 3D).

The smoothness class is DECLARED and load-bearing (it had to be added to
fix bound-to-bound streaking), so the labels it produces must be reported
with their sensitivity to it. Protocol (mirrors sensitivity_tier1.py):

  - Re-run the full 30-member stage-1 ensemble at perturbed class
    hyperparameters, one axis at a time:
        smooth_ratio          in {3, 10*, 30}      (* = base)
        smooth_correlation_m  in {150, 300*, 600}
  - For each variant: Jaccard overlap of its forced-high cell set
    (eps 0.25, illuminated-by-base cells only) with the base ensemble's.
  - CONTROL: the base ensemble split into two 15-member halves; the
    Jaccard between the two half-ensemble forced-high sets is the
    stability ceiling - no variant can be expected to beat it.

Reading: J(variant) close to the control ceiling = labels are robust to
the class dial; J(variant) << ceiling = the dial is doing the work and
the labels are class-artifacts (the Volve Tier-1 outcome for the
smoothness percentile, J=0.358 vs control 0.773).

Run:  uv run python -m porotomo.sensitivity_3d            # full (~25 min)
      uv run python -m porotomo.sensitivity_3d --smoke    # tiny configs
Outputs: porotomo_sensitivity.png, porotomo_sensitivity.json,
         variant ensembles cached at porotomo/data/ens_sens_*.npz
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from posdec.decomposition import feasible_interval, classify
from porotomo.inversion3d import prepare, vp_ensemble_3d, Config3D
from porotomo.decompose_3d import (load_ensemble, illumination,
                                   ILLUM_MIN_PATH_M)

EPS_KMS = 0.25
BASE_NPZ = "porotomo/data/ensemble_stage1.npz"

VARIANTS = [
    ("smooth_ratio_3", {"smooth_ratio": 3.0}),
    ("smooth_ratio_30", {"smooth_ratio": 30.0}),
    ("corr_150m", {"smooth_correlation_m": 150.0}),
    ("corr_600m", {"smooth_correlation_m": 600.0}),
]


def gauge_1d(members: np.ndarray, air: np.ndarray) -> np.ndarray:
    ground = ~air
    vp_1d = np.array([
        np.median(members[:, k][:, ground[k]]) if ground[k].any() else np.nan
        for k in range(members.shape[1])
    ])
    return np.broadcast_to(vp_1d[:, None, None],
                           members.shape[1:]).copy()


def forced_sets(members: np.ndarray, air: np.ndarray, lit: np.ndarray,
                eps: float = EPS_KMS):
    """(forced_high, forced_low, forced_quiet) boolean masks on lit cells,
    each ensemble decomposed against ITS OWN 1D gauge (the gauge is part
    of the method, so a variant re-derives it)."""
    a_min, a_max = feasible_interval(members, gauge_1d(members, air))
    cls = classify(a_min, a_max, eps)
    return {
        "high": (cls == 2) & lit,
        "low": (cls == -2) & lit,
        "quiet": (cls == 0) & lit,
    }


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def ensure_variant(name: str, overrides: dict, ds, grid, air,
                   n_members: int, n_workers: int) -> np.ndarray:
    path = f"porotomo/data/ens_sens_{name}.npz"
    if os.path.exists(path):
        return np.load(path)["members"]
    cfg = Config3D(n_members=n_members, **overrides)
    print(f"[sensitivity] running variant {name}: {overrides}")
    members, _meta = vp_ensemble_3d(ds, grid, air, cfg, n_workers=n_workers)
    np.savez_compressed(path, members=members)
    return members


def main() -> None:
    smoke = "--smoke" in sys.argv
    n_members = 4 if smoke else 30
    n_workers = min(10, max(1, (os.cpu_count() or 4) - 2))
    base_members, air, grid, _z = load_ensemble()
    picks, _g, _a, ds = prepare()
    lit = (illumination(base_members, air, grid, ds)
           >= ILLUM_MIN_PATH_M) & (~air)
    base = forced_sets(base_members, air, lit)
    n_base = base_members.shape[0]

    # control ceiling: two half-ensembles of the base run
    half_a = forced_sets(base_members[: n_base // 2], air, lit)
    half_b = forced_sets(base_members[n_base // 2:], air, lit)
    control = {k: jaccard(half_a[k], half_b[k]) for k in ("high", "quiet")}

    results: dict = {
        "eps_kms": EPS_KMS,
        "n_members": n_members,
        "base_counts": {k: int(v.sum()) for k, v in base.items()},
        "control_half_ensemble": {k: round(v, 3) for k, v in control.items()},
        "note": "control uses 15-member halves; half-ensembles have "
                "narrower intervals than the full 30, so the ceiling is "
                "conservative (true 30-vs-30 stability would be higher)",
        "variants": {},
    }
    for name, overrides in VARIANTS:
        members = ensure_variant(name, overrides, ds, grid, air,
                                 n_members, n_workers)
        fs = forced_sets(members, air, lit)
        results["variants"][name] = {
            "overrides": overrides,
            "counts": {k: int(v.sum()) for k, v in fs.items()},
            "jaccard_vs_base": {
                k: round(jaccard(fs[k], base[k]), 3)
                for k in ("high", "quiet")
            },
        }
        print(f"  {name}: J(high) = "
              f"{results['variants'][name]['jaccard_vs_base']['high']}, "
              f"J(quiet) = "
              f"{results['variants'][name]['jaccard_vs_base']['quiet']}")

    with open("porotomo_sensitivity.json", "w") as fh:
        json.dump(results, fh, indent=2)

    # figure: bar chart vs control ceiling
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, title in ((axes[0], "high", "forced-high"),
                           (axes[1], "quiet", "forced-quiet")):
        names = [n for n, _ in VARIANTS]
        vals = [results["variants"][n]["jaccard_vs_base"][key]
                for n in names]
        ax.bar(range(len(names)), vals, color="tab:blue")
        ax.axhline(control[key], color="tab:red", ls="--",
                   label=f"half-ensemble control ({control[key]:.2f})")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Jaccard vs base labels")
        ax.set_title(f"{title} (eps {EPS_KMS} km/s)")
        ax.legend()
    fig.suptitle("PoroTomo: smoothness-class sensitivity of forced labels")
    fig.tight_layout()
    fig.savefig("porotomo_sensitivity.png", dpi=130)
    print("wrote porotomo_sensitivity.png, porotomo_sensitivity.json")


if __name__ == "__main__":
    main()
