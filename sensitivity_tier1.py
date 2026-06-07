"""
sensitivity_tier1.py - Tier-1-bound sensitivity sweep (ORSI propagation #3).

Brian's ORSI evaluation flagged: "Smoothness is enforced upstream as a
Tier-1 admissibility condition; this moves a form of regularization into
the possibilistic layer." The shipped paper acknowledged the velocity
envelope and the smoothness preference but did not quantify how the
forced / measure-dependent split depends on either.

This script answers that quantitatively, using the cached 396-member
RWC-1 ensemble. For each Tier-1 bound sweep value:
  * filter the baseline ensemble (keep only members satisfying the bound)
  * recompute the per-cell forced-sign decomposition on the kept members
  * compare against the baseline decomposition: forced-set sizes, Jaccard
    indices on the three masks, fraction of cells reclassified.

A bound is ROBUST if tightening or relaxing it within a plausible range
leaves the decomposition essentially unchanged (Jaccard >= ~0.9, cells
reclassified <= a few percent). A bound is BINDING if changing it shifts
the decomposition materially - meaning the published forced/measure-
dependent split is conditional on that choice and the paper must say so.

Sweeps:
  * Velocity envelope - tighten VP_MIN upward and VP_MAX downward
    symmetrically, filter members whose Vp leaves the tightened envelope.
  * Smoothness admissibility bar - keep only members with smoothness <=
    a percentile threshold, sweeping the percentile from 100% (all kept)
    down to 25% (smoothest quarter only).

Output:
  * stdout table summarizing the sweep
  * sensitivity_tier1.json - structured sweep result
  * sensitivity_tier1.png - forced-set size + Jaccard vs sweep

Run:  uv run python sensitivity_tier1.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import posdec
import synthetic_demo as sd
from posdec.diagnostics import smoothness

ENSEMBLE_CACHE = "rwc1_ensemble.npy"
EPS = sd.EPS                  # km/s; same deadband as the eikonal demo
OUT_JSON = "sensitivity_tier1.json"
OUT_FIG = "sensitivity_tier1.png"


# --- helpers ---------------------------------------------------------------

def jaccard(a_mask, b_mask):
    """Jaccard index on two boolean masks. 1.0 if identical, 0.0 if
    disjoint, undefined (returned as 1.0 by convention) if both empty."""
    inter = np.logical_and(a_mask, b_mask).sum()
    union = np.logical_or(a_mask, b_mask).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def decompose(ensemble, bg):
    a_min, a_max = posdec.feasible_interval(ensemble, bg)
    cls = posdec.classify(a_min, a_max, EPS)
    return cls, posdec.three_masks(cls)


def compare(baseline_masks, masks, n_kept):
    return {
        "n_kept": int(n_kept),
        "forced_high_cells": int(masks["forced_high"].sum()),
        "forced_low_cells": int(masks["forced_low"].sum()),
        "measure_dependent_cells": int(masks["measure_dependent"].sum()),
        "jaccard_forced_high":
            jaccard(baseline_masks["forced_high"], masks["forced_high"]),
        "jaccard_forced_low":
            jaccard(baseline_masks["forced_low"], masks["forced_low"]),
        "jaccard_measure_dependent":
            jaccard(baseline_masks["measure_dependent"],
                    masks["measure_dependent"]),
    }


# --- sweeps ----------------------------------------------------------------

def sweep_velocity_floor(ensemble, bg, baseline_masks):
    """Tighten VP_MIN upward only; keep only members whose minimum Vp
    is at or above the tightened floor. (The ceiling at 9.0 km/s is
    saturated in this ensemble - every member touches it - so a
    ceiling sweep would collapse to zero members at the first tighten;
    that is reported separately as a saturation flag.)"""
    rows = []
    for floor in [sd.VP_MIN, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5]:
        kept = [m for m in ensemble if m.min() >= floor]
        if not kept:
            rows.append({"vp_floor": floor, "n_kept": 0, "note": "empty"})
            continue
        cls, masks = decompose(kept, bg)
        rows.append({"vp_floor": floor,
                     **compare(baseline_masks, masks, len(kept))})
    return rows


def ceiling_saturation(ensemble):
    """Fraction of members that touch the VP_MAX ceiling (or come within
    one numerical-tolerance band of it). A saturated ceiling means the
    bound is binding for the whole ensemble."""
    eps = 1e-6
    tol_kms = 0.05
    touches = sum(1 for m in ensemble if m.max() >= sd.VP_MAX - tol_kms - eps)
    return {"vp_max_kms": float(sd.VP_MAX),
            "tol_kms": tol_kms,
            "members_touching_ceiling": int(touches),
            "fraction": float(touches) / float(len(ensemble))}


def sweep_smoothness_vs_random(ensemble, bg, baseline_masks,
                               rng_seed=20260607):
    """Smoothness sub-selection within the already-smooth pool, vs. a
    same-size random sub-selection as the control. The Jaccard difference
    isolates the smoothness-specific effect from the generic
    'smaller subset -> narrower interval' artifact."""
    smooths = np.array([smoothness(m) for m in ensemble])
    rng = np.random.default_rng(rng_seed)
    rows = []
    for p in [100, 90, 75, 60, 50, 40, 25]:
        thresh = float(np.percentile(smooths, p))
        smooth_kept_idx = np.where(smooths <= thresh)[0]
        if len(smooth_kept_idx) == 0:
            rows.append({"percentile": p, "smoothness_thresh": thresh,
                         "n_kept": 0, "note": "empty"})
            continue
        smooth_kept = [ensemble[i] for i in smooth_kept_idx]
        _, smooth_masks = decompose(smooth_kept, bg)

        # Random control of the SAME size; averaged across 5 draws for
        # stability.
        n = len(smooth_kept_idx)
        n_trials = 5
        rj_hi, rj_lo, rj_md = [], [], []
        for _ in range(n_trials):
            rnd_idx = rng.choice(len(ensemble), size=n, replace=False)
            rnd_kept = [ensemble[i] for i in rnd_idx]
            _, rnd_masks = decompose(rnd_kept, bg)
            rj_hi.append(jaccard(baseline_masks["forced_high"],
                                 rnd_masks["forced_high"]))
            rj_lo.append(jaccard(baseline_masks["forced_low"],
                                 rnd_masks["forced_low"]))
            rj_md.append(jaccard(baseline_masks["measure_dependent"],
                                 rnd_masks["measure_dependent"]))
        rows.append({
            "percentile": p,
            "smoothness_thresh": thresh,
            **compare(baseline_masks, smooth_masks, n),
            "random_control": {
                "jaccard_forced_high_mean": float(np.mean(rj_hi)),
                "jaccard_forced_low_mean": float(np.mean(rj_lo)),
                "jaccard_measure_dependent_mean": float(np.mean(rj_md)),
            },
        })
    return rows


# --- reporting -------------------------------------------------------------

def print_table(rows, label, x_key, x_fmt, with_random=False):
    print(f"\n  {label}")
    header_cols = (f"  {'x':>10}   {'n_kept':>6}   "
                   f"{'F-hi':>4} {'F-lo':>4} {'MD':>5}   "
                   f"{'J(F-hi)':>7} {'J(F-lo)':>7} {'J(MD)':>7}")
    if with_random:
        header_cols += f"  | {'J_rnd(F-hi)':>11} {'J_rnd(F-lo)':>11}"
    print(header_cols)
    print("  " + "-" * (len(header_cols) - 2))
    for r in rows:
        if r.get("note") == "empty":
            print(f"  {x_fmt.format(r[x_key]):>10}   {0:6d}   "
                  f"(empty - bound too tight)")
            continue
        line = (f"  {x_fmt.format(r[x_key]):>10}   {r['n_kept']:6d}   "
                f"{r['forced_high_cells']:4d} "
                f"{r['forced_low_cells']:4d} "
                f"{r['measure_dependent_cells']:5d}   "
                f"{r['jaccard_forced_high']:7.3f} "
                f"{r['jaccard_forced_low']:7.3f} "
                f"{r['jaccard_measure_dependent']:7.3f}")
        if with_random and "random_control" in r:
            rc = r["random_control"]
            line += (f"  | {rc['jaccard_forced_high_mean']:11.3f} "
                     f"{rc['jaccard_forced_low_mean']:11.3f}")
        print(line)


def plot_sweeps(env_rows, smooth_rows, baseline_masks, out_path):
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))

    def usable(rows):
        return [r for r in rows if r.get("note") != "empty"]

    e = usable(env_rows)
    s = usable(smooth_rows)

    # Top row: velocity floor sweep.
    xs = [r["vp_floor"] for r in e]
    ax[0, 0].plot(xs, [r["forced_high_cells"] for r in e],
                  "-o", color="#b2182b", label="forced-high")
    ax[0, 0].plot(xs, [r["forced_low_cells"] for r in e],
                  "-o", color="#2166ac", label="forced-low")
    ax[0, 0].plot(xs, [r["measure_dependent_cells"] for r in e],
                  "-o", color="#e8a33d", label="measure-dependent")
    ax[0, 0].set_xlabel("VP_MIN floor (km/s)")
    ax[0, 0].set_ylabel("cell count")
    ax[0, 0].set_title("velocity floor: forced-set sizes")
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=0.3)

    ax[0, 1].plot(xs, [r["jaccard_forced_high"] for r in e],
                  "-o", color="#b2182b", label="J(forced-high)")
    ax[0, 1].plot(xs, [r["jaccard_forced_low"] for r in e],
                  "-o", color="#2166ac", label="J(forced-low)")
    ax[0, 1].plot(xs, [r["jaccard_measure_dependent"] for r in e],
                  "-o", color="#e8a33d", label="J(measure-dep)")
    ax[0, 1].axhline(0.9, color="0.5", lw=0.8, ls="--",
                     label="robustness band (J>=0.9)")
    ax[0, 1].set_xlabel("VP_MIN floor (km/s)")
    ax[0, 1].set_ylabel("Jaccard vs baseline")
    ax[0, 1].set_ylim(0.0, 1.05)
    ax[0, 1].set_title("velocity floor: agreement with baseline")
    ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=0.3)

    # Bottom row: smoothness sweep vs random control.
    xs = [r["percentile"] for r in s]
    ax[1, 0].plot(xs, [r["forced_high_cells"] for r in s],
                  "-o", color="#b2182b", label="forced-high")
    ax[1, 0].plot(xs, [r["forced_low_cells"] for r in s],
                  "-o", color="#2166ac", label="forced-low")
    ax[1, 0].plot(xs, [r["measure_dependent_cells"] for r in s],
                  "-o", color="#e8a33d", label="measure-dependent")
    ax[1, 0].set_xlabel("smoothness percentile kept (100 = all)")
    ax[1, 0].set_ylabel("cell count")
    ax[1, 0].set_title("smoothness bar: forced-set sizes")
    ax[1, 0].invert_xaxis()
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3)

    ax[1, 1].plot(xs, [r["jaccard_forced_high"] for r in s],
                  "-o", color="#b2182b", label="J(F-hi, smoothness-kept)")
    ax[1, 1].plot(xs, [r["jaccard_forced_low"] for r in s],
                  "-o", color="#2166ac", label="J(F-lo, smoothness-kept)")
    ax[1, 1].plot(xs,
                  [r["random_control"]["jaccard_forced_high_mean"] for r in s],
                  "--^", color="#d98c6a", alpha=0.85,
                  label="J(F-hi, random control)")
    ax[1, 1].plot(xs,
                  [r["random_control"]["jaccard_forced_low_mean"] for r in s],
                  "--^", color="#6ba0c8", alpha=0.85,
                  label="J(F-lo, random control)")
    ax[1, 1].axhline(0.9, color="0.5", lw=0.8, ls="--",
                     label="robustness band (J>=0.9)")
    ax[1, 1].set_xlabel("percentile kept (100 = all)")
    ax[1, 1].set_ylabel("Jaccard vs baseline")
    ax[1, 1].set_ylim(0.0, 1.05)
    ax[1, 1].set_title("smoothness-kept vs random subset (same N)")
    ax[1, 1].invert_xaxis()
    ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)

    fig.suptitle("Tier-1 sensitivity: velocity floor (top) + "
                 "smoothness bar with random control (bottom)",
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --- main ------------------------------------------------------------------

def main():
    cache = Path(ENSEMBLE_CACHE)
    if not cache.exists():
        raise SystemExit(
            f"missing {ENSEMBLE_CACHE} — run rwc1_forced_stability.py first.")
    ensemble = np.load(cache)
    members = [ensemble[k] for k in range(ensemble.shape[0])]
    bg = sd.depth_gradient_background(ensemble.mean(axis=0))

    print("Tier-1 sensitivity sweep")
    print("=" * 72)
    print(f"baseline ensemble: {len(members)} members, "
          f"grid {ensemble.shape[1]}x{ensemble.shape[2]}, "
          f"eps_deadband = {EPS} km/s")

    _, baseline_masks = decompose(members, bg)
    print(f"baseline: forced-high {int(baseline_masks['forced_high'].sum())}, "
          f"forced-low {int(baseline_masks['forced_low'].sum())}, "
          f"measure-dependent "
          f"{int(baseline_masks['measure_dependent'].sum())}")

    env_rows = sweep_velocity_floor(members, bg, baseline_masks)
    smooth_rows = sweep_smoothness_vs_random(members, bg, baseline_masks)
    saturation = ceiling_saturation(members)
    print(f"\nVP_MAX={saturation['vp_max_kms']:.1f} km/s ceiling: "
          f"{saturation['members_touching_ceiling']}/{len(members)} "
          f"({100 * saturation['fraction']:.0f}%) members touch within "
          f"{saturation['tol_kms']} km/s "
          f"-> ceiling is saturated; not separately sweepable.")

    print_table(env_rows,
                "Velocity floor sweep (VP_MIN_new in km/s)",
                "vp_floor", "{:.2f}")
    print_table(smooth_rows,
                "Smoothness percentile sweep + random control "
                "(p = max smoothness percentile kept)",
                "percentile", "{:d}",
                with_random=True)

    Path(OUT_JSON).write_text(json.dumps({
        "baseline_forced_high":
            int(baseline_masks["forced_high"].sum()),
        "baseline_forced_low":
            int(baseline_masks["forced_low"].sum()),
        "baseline_measure_dependent":
            int(baseline_masks["measure_dependent"].sum()),
        "velocity_floor_sweep": env_rows,
        "ceiling_saturation": saturation,
        "smoothness_sweep": smooth_rows,
        "eps_deadband_kms": EPS,
        "ensemble_size": len(members),
    }, indent=2))
    plot_sweeps(env_rows, smooth_rows, baseline_masks, OUT_FIG)
    print(f"\nJSON written:   {OUT_JSON}")
    print(f"Figure written: {OUT_FIG}")
    _verdict(env_rows, smooth_rows)


def _verdict(env_rows, smooth_rows):
    print("\nVerdict")
    def worst_jaccard(rows):
        worst = 1.0
        for r in rows:
            if r.get("note") == "empty":
                continue
            for k in ("jaccard_forced_high",
                      "jaccard_forced_low",
                      "jaccard_measure_dependent"):
                worst = min(worst, r[k])
        return worst

    def worst_random_control_jaccard(rows):
        worst = 1.0
        for r in rows:
            rc = r.get("random_control")
            if not rc:
                continue
            for k in ("jaccard_forced_high_mean",
                      "jaccard_forced_low_mean",
                      "jaccard_measure_dependent_mean"):
                worst = min(worst, rc[k])
        return worst

    we = worst_jaccard(env_rows)
    ws = worst_jaccard(smooth_rows)
    wr = worst_random_control_jaccard(smooth_rows)
    band = 0.9
    print(f"  velocity floor    : worst Jaccard = {we:.3f}  "
          f"=> {'ROBUST' if we >= band else 'BINDING'} "
          f"(threshold J >= {band})")
    print(f"  smoothness bar    : worst Jaccard = {ws:.3f}")
    print(f"  random control    : worst Jaccard = {wr:.3f}  "
          f"(same N, random subset)")
    if ws + 0.05 < wr:
        print("    => smoothness-kept Jaccard sits clearly below the "
              "random-control curve")
        print("       at the same N. The smoothness preference is "
              "load-bearing in the")
        print("       possibilistic layer (Brian's flag); the published "
              "split is")
        print("       conditional on the sampler's smoothness preference, "
              "not only on")
        print("       the smaller effective sample.")
    elif abs(ws - wr) <= 0.05:
        print("    => smoothness-kept Jaccard tracks the random-control "
              "curve. The")
        print("       drop in agreement at smaller N reflects generic "
              "sub-selection")
        print("       narrowing the feasible interval, NOT a specific "
              "smoothness")
        print("       preference effect.")
    else:
        print("    => smoothness-kept Jaccard sits ABOVE the random-control "
              "curve.")
        print("       The smoothness preference is non-binding within the "
              "swept range.")
    print("\n  A Tier-1 bound is BINDING if changing it within the swept "
          "range shifts the")
    print("  decomposition enough that the published forced/measure-"
          "dependent split is")
    print("  conditional on that choice and the paper must say so.")


if __name__ == "__main__":
    main()
