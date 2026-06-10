"""porotomo/timelapse.py - possibilistic TIME-LAPSE study across the four
PoroTomo acquisition stages.

The PoroTomo experiment deliberately modulated the plant's injection/
production between stages (March 2016, ~3 weeks). The time-lapse question
in possibilistic form: per cell, is the stage-A -> stage-B velocity CHANGE
forced (every feasible pair of models shows it), or measure-dependent?
Neither the deterministic (Parker) nor any Bayesian treatment of this
dataset has addressed forced time-lapse structure.

DESIGN - paired references. Each stage is inverted with the SAME member
seeds (same cfg.seed -> identical smooth random references). Member k of
stage B differs from member k of stage A only through the data, so the
paired difference d_k = vp_k^B - vp_k^A cancels the reference-induced
spread; the per-cell interval over k of d_k is the feasible-change
interval. Unpaired ensembles would overstate the change interval by the
full prior spread.

NULL CONTROL - stage 1 -> 2. The repeat scatter showed a +2 ms median
offset and the stages are closest in time; forced-change cells in the
1->2 pair estimate the false-positive rate of the procedure at each eps.

Classification per cell (illuminated in BOTH stages):
  forced-faster   min_k d_k > +eps_dt
  forced-slower   max_k d_k < -eps_dt
  forced-stable   interval within +/-eps_dt
  open            otherwise
with eps_dt swept over {0.05, 0.10, 0.20} km/s.

Run:  uv run python -m porotomo.timelapse            # full (~15 min)
      uv run python -m porotomo.timelapse --smoke
Outputs: porotomo_timelapse.png, porotomo_timelapse.json,
         stage ensembles cached at porotomo/data/ensemble_stage{2,3,4}.npz
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from porotomo.inversion3d import prepare, vp_ensemble_3d, Config3D
from porotomo.decompose_3d import (load_ensemble, illumination,
                                   ILLUM_MIN_PATH_M)

EPS_DT_SWEEP = (0.05, 0.10, 0.20)
EPS_DT_MAIN = 0.10
PAIRS = [(1, 2), (1, 3), (1, 4)]          # (1,2) = null control


def ensure_stage_ensemble(stage: int, n_members: int, n_workers: int,
                          suffix: str = ""):
    """Invert one stage with the BASE config (same seed => paired refs).
    Stage 1 reuses the existing base ensemble. Smoke runs use a suffix so
    they never poison the full-run cache."""
    path = f"porotomo/data/ensemble_stage{stage}{suffix}.npz"
    if os.path.exists(path):
        z = np.load(path)
        return z["members"], z["air"].astype(bool)
    picks, grid, air, ds = prepare(stage=stage)
    cfg = Config3D(n_members=n_members)
    print(f"[timelapse] inverting stage {stage} "
          f"({ds.n_picks} picks, {len(ds.src_pts)} sources)")
    members, meta = vp_ensemble_3d(ds, grid, air, cfg, n_workers=n_workers)
    np.savez_compressed(path, members=members, air=air,
                        members_rms_s=meta["members_rms_s"],
                        grid_cell_m=grid.cell_m, grid_x0=grid.x0,
                        grid_y0=grid.y0, grid_z_top_m=grid.z_top_m,
                        noise_rms_s=cfg.noise_rms_s, seed=cfg.seed)
    return members, air


def lit_mask_for_stage(members, air, grid, stage: int):
    _p, _g, _a, ds = prepare(stage=stage)
    return (illumination(members, air, grid, ds) >= ILLUM_MIN_PATH_M) \
        & (~air)


def main() -> None:
    smoke = "--smoke" in sys.argv
    n_workers = min(10, max(1, (os.cpu_count() or 4) - 2))
    base_members, air, grid, z1 = load_ensemble()
    n_members = 4 if smoke else base_members.shape[0]
    if smoke:
        base_members = base_members[:n_members]

    lit1 = lit_mask_for_stage(base_members, air, grid, 1)
    results: dict = {
        "design": "paired references (same member seeds per stage); "
                  "pair (1,2) is the null control",
        "eps_dt_sweep_kms": list(EPS_DT_SWEEP),
        "n_members": int(n_members),
        "pairs": {},
    }
    maps = {}
    for st_a, st_b in PAIRS:
        mem_b, air_b = ensure_stage_ensemble(
            st_b, n_members, n_workers, suffix="_smoke" if smoke else "")
        if smoke:
            mem_b = mem_b[:n_members]
        assert air_b.shape == air.shape
        lit_b = lit_mask_for_stage(mem_b, air, grid, st_b)
        both = lit1 & lit_b
        d = mem_b - base_members                  # paired differences
        d_lo, d_hi = d.min(axis=0), d.max(axis=0)
        entry: dict = {"n_cells_lit_both": int(both.sum()),
                       "median_paired_spread_kms": round(
                           float(np.median((d_hi - d_lo)[both])), 3),
                       "eps": {}}
        for eps in EPS_DT_SWEEP:
            faster = (d_lo > eps) & both
            slower = (d_hi < -eps) & both
            stable = (d_lo >= -eps) & (d_hi <= eps) & both
            entry["eps"][f"{eps:.2f}"] = {
                "forced_faster": int(faster.sum()),
                "forced_slower": int(slower.sum()),
                "forced_stable_frac": round(
                    float(stable.sum() / both.sum()), 4),
                "open_frac": round(float(
                    (both.sum() - faster.sum() - slower.sum()
                     - stable.sum()) / both.sum()), 4),
            }
        results["pairs"][f"{st_a}->{st_b}"] = entry
        cls = np.full(air.shape, np.nan)
        cls[both] = 0.0
        cls[(d_lo >= -EPS_DT_MAIN) & (d_hi <= EPS_DT_MAIN) & both] = 1.0
        cls[(d_lo > EPS_DT_MAIN) & both] = 2.0
        cls[(d_hi < -EPS_DT_MAIN) & both] = -2.0
        maps[(st_a, st_b)] = cls
        print(f"  {st_a}->{st_b}: spread {entry['median_paired_spread_kms']}"
              f" km/s; eps {EPS_DT_MAIN}: "
              f"{entry['eps'][f'{EPS_DT_MAIN:.2f}']}")

    with open("porotomo_timelapse.json", "w") as fh:
        json.dump(results, fh, indent=2)

    # figure: classification maps at two depths per pair
    k_slices = [3, 5]
    cmap = ListedColormap(["#2166ac", "#f7f7f7", "#d9d9d9", "#b2182b"])
    norm = BoundaryNorm([-2.5, -0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, axes = plt.subplots(len(PAIRS), len(k_slices),
                             figsize=(11, 4.6 * len(PAIRS)))
    extent = [grid.x0, grid.x0 + grid.nx * grid.cell_m,
              grid.y0, grid.y0 + grid.ny * grid.cell_m]
    zc = grid.cell_centers_elev()
    for i, (st_a, st_b) in enumerate(PAIRS):
        for j, k in enumerate(k_slices):
            ax = axes[i, j]
            ax.imshow(maps[(st_a, st_b)][k], origin="lower", extent=extent,
                      cmap=cmap, norm=norm)
            ax.set_title(f"stage {st_a}->{st_b} @ {zc[k]:.0f} m ASL "
                         f"(eps {EPS_DT_MAIN})", fontsize=10)
    fig.suptitle("PoroTomo possibilistic time-lapse: blue=forced-slower, "
                 "white=open, grey=forced-stable, red=forced-faster",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig("porotomo_timelapse.png", dpi=130)
    print("wrote porotomo_timelapse.png, porotomo_timelapse.json")


if __name__ == "__main__":
    main()
