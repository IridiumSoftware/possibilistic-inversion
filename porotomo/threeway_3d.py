"""porotomo/threeway_3d.py - head-to-head of the three uncertainty
representations on PoroTomo, under matched physics, noise, prior class,
and matched tests (the 3D analogue of volve/threeway.py).

  possibilistic : 30-member feasible-set ensemble  (ensemble_stage1.npz)
  Bayesian      : 30 RTO posterior samples          (bayes_samples.npz)
  deep learning : 30 MC-dropout models              (nn_models.npz)

Tests (identical for all three):
  1. stage-2 holdout coverage  - read from the methods' own certs;
  2. published-model-inside    - fraction of illuminated cells where the
     Thurber 2017-11 Vp model (GDR 1124 ID10) falls inside the method's
     per-cell interval (possibilistic: min..max; Bayes: 2.5..97.5%;
     NN: MC-dropout 2.5..97.5%), strict and +/-0.25 km/s;
  3. interval geometry         - median per-cell width in lit cells.

Run:  uv run python -m porotomo.threeway_3d
Outputs: porotomo_threeway.png, porotomo_threeway.json
"""

from __future__ import annotations

import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from porotomo.decompose_3d import (load_ensemble, illumination,
                                   ILLUM_MIN_PATH_M)
from porotomo.inversion3d import prepare
from porotomo.published_model_check import load_meshed
from porotomo.loader import X0, Y0

THURBER_COL = "ID10_Vp_Thurber20171123"


def thurber_on_grid(grid):
    xyz, mv = load_meshed([THURBER_COL])
    v = mv[THURBER_COL] * 0.001
    iz = np.round(grid.elev_to_cell(xyz[:, 2])).astype(int)
    iy = np.round(grid.y_to_cell(xyz[:, 1])).astype(int)
    ix = np.round(grid.x_to_cell(xyz[:, 0])).astype(int)
    ok = ((iz >= 0) & (iz < grid.nz) & (iy >= 0) & (iy < grid.ny)
          & (ix >= 0) & (ix < grid.nx) & np.isfinite(v))
    flat = (iz * grid.ny + iy) * grid.nx + ix
    sums = np.bincount(flat[ok], weights=v[ok], minlength=grid.n_cells)
    cnts = np.bincount(flat[ok], minlength=grid.n_cells)
    out = np.full(grid.n_cells, np.nan)
    out[cnts > 0] = sums[cnts > 0] / cnts[cnts > 0]
    return out.reshape(grid.nz, grid.ny, grid.nx)


def main() -> None:
    members, air, grid, _z = load_ensemble()
    picks, _g, _a, ds = prepare()
    lit = (illumination(members, air, grid, ds) >= ILLUM_MIN_PATH_M) & (~air)
    thur = thurber_on_grid(grid)
    cmp_mask = lit & np.isfinite(thur)

    bayes = np.load("porotomo/data/bayes_samples.npz")["samples"]
    nn = np.load("porotomo/data/nn_models.npz")["models"]

    methods = {
        "possibilistic": (members.min(0), members.max(0)),
        "bayes_rto_95": (np.quantile(bayes, 0.025, axis=0),
                         np.quantile(bayes, 0.975, axis=0)),
        "nn_mcdropout_95": (np.quantile(nn, 0.025, axis=0),
                            np.quantile(nn, 0.975, axis=0)),
    }
    certs = {
        "possibilistic": json.load(open("porotomo_holdout_cert.json")),
        "bayes_rto_95": json.load(open("porotomo_bayes_cert.json")),
        "nn_mcdropout_95": json.load(open("porotomo_nn_cert.json")),
    }
    out: dict = {"published_model": THURBER_COL,
                 "n_cells_compared": int(cmp_mask.sum()), "methods": {}}
    for name, (lo, hi) in methods.items():
        inside = ((thur >= lo) & (thur <= hi))[cmp_mask]
        inside_tol = ((thur >= lo - 0.25) & (thur <= hi + 0.25))[cmp_mask]
        c = certs[name]
        hold = c["holdout"] if "holdout" in c else {
            "inside_raw": c["inside_raw"],
            "inside_with_pick_noise": c["inside_with_pick_noise"],
            "rms_median_prediction_ms": c["rms_median_prediction_ms"],
        }
        key_raw = [k for k in hold if k.startswith("inside_raw")][0]
        out["methods"][name] = {
            "published_inside_strict": round(float(inside.mean()), 4),
            "published_inside_tol_0p25": round(float(inside_tol.mean()), 4),
            "median_width_kms_lit": round(float(np.median((hi - lo)[lit])), 3),
            "holdout_inside_raw": hold[key_raw],
            "holdout_inside_with_noise": hold["inside_with_pick_noise"],
            "holdout_rms_ms": hold["rms_median_prediction_ms"],
        }
    with open("porotomo_threeway.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))

    # ---- summary figure -----------------------------------------------------
    labels = list(out["methods"].keys())
    metrics = [
        ("published_inside_strict", "Thurber inside (strict)"),
        ("published_inside_tol_0p25", "Thurber inside (+/-0.25)"),
        ("holdout_inside_raw", "holdout inside (raw)"),
        ("holdout_inside_with_noise", "holdout inside (+/-36 ms)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(metrics))
    width = 0.26
    colors = ["tab:blue", "tab:orange", "tab:green"]
    ax = axes[0]
    for i, name in enumerate(labels):
        vals = [out["methods"][name][m] for m, _ in metrics]
        ax.bar(x + (i - 1) * width, vals, width, label=name, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels([t for _, t in metrics], rotation=15, ha="right",
                       fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("coverage tests (higher = intervals bracket truth)")

    ax = axes[1]
    vals = [out["methods"][n]["median_width_kms_lit"] for n in labels]
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel("km/s")
    ax.set_title("median interval width (lit cells)")
    ax.tick_params(axis="x", labelsize=8)

    ax = axes[2]
    vals = [out["methods"][n]["holdout_rms_ms"] for n in labels]
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel("ms")
    ax.set_title("stage-2 holdout RMS (median model)")
    ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("PoroTomo 3D: possibilistic vs Bayesian (RTO) vs NN "
                 "(MC-dropout) under matched physics/noise/prior/tests")
    fig.tight_layout()
    fig.savefig("porotomo_threeway.png", dpi=130)
    print("wrote porotomo_threeway.png, porotomo_threeway.json")


if __name__ == "__main__":
    main()
