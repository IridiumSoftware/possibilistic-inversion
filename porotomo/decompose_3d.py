"""porotomo/decompose_3d.py - possibilistic decomposition of the PoroTomo
3D feasible-set ensemble.

Carries the Volve phase-5 / phase-A lessons forward as DESIGN RULES:

  1. GAUGE DECLARED. The background is the laterally-uniform 1D profile
     vp_1d(z) = median over members and lateral ground cells. This is an
     interpretable physical null ("no lateral structure"); forced-high/low
     cells are lateral anomalies the data forces relative to it. It is NOT
     the full ensemble mean (which made forced-quiet tautological in Volve
     phase 5 - the baseline then tracked every member, killing the anomaly
     by construction). A 1D background cannot track lateral structure, so
     lateral labels are non-tautological; VERTICAL structure is absorbed
     into the gauge and is deliberately out of scope here.

  2. ILLUMINATION MASK. Cells without ray coverage are prior-dominated:
     their ensemble interval reflects the random references, not data.
     They are labelled UNILLUMINATED and excluded from the forced
     statistics (reported separately), rather than being allowed to pose
     as measure-dependent "findings".

  3. EPS SWEEP. The deadband is not a free dial to tune the headline:
     the certificate records label fractions at eps in {0.15, 0.25, 0.35}
     km/s; the figure uses the middle value.

Requires porotomo/data/ensemble_stage1.npz (porotomo.inversion3d --full).

Run:  uv run python -m porotomo.decompose_3d
Outputs: porotomo_decomposition.png, porotomo_decomposition_cert.json
"""

from __future__ import annotations

import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from posdec.decomposition import feasible_interval, classify
from porotomo.inversion3d import prepare, forward_3d, Grid3D

ENSEMBLE_NPZ = "porotomo/data/ensemble_stage1.npz"
EPS_SWEEP_KMS = (0.15, 0.25, 0.35)
EPS_MAIN_KMS = 0.25
# A cell is "illuminated" when the stage-1 rays of the ensemble-median
# model accumulate at least this much path length in it (metres). One
# 50 m cell crossed by ~3 rays clears it; isolated grazing hits do not.
ILLUM_MIN_PATH_M = 150.0


def load_ensemble():
    z = np.load(ENSEMBLE_NPZ)
    grid = Grid3D(cell_m=float(z["grid_cell_m"]), nz=z["members"].shape[1],
                  ny=z["members"].shape[2], nx=z["members"].shape[3],
                  x0=float(z["grid_x0"]), y0=float(z["grid_y0"]),
                  z_top_m=float(z["grid_z_top_m"]))
    return z["members"], z["air"].astype(bool), grid, z


def illumination(members: np.ndarray, air: np.ndarray, grid: Grid3D,
                 ds) -> np.ndarray:
    """Total ray path length (metres) per cell, traced through the
    ensemble-median model. The data-coverage map for rule 2."""
    vp_med = np.median(members, axis=0)
    vp_med[air] = 0.34
    _, J = forward_3d(vp_med, ds, grid, compute_jacobian=True)
    # J columns are d t / d slowness_SI = path metres per cell
    path_m = np.asarray(np.abs(J).sum(axis=0)).ravel()
    return path_m.reshape(grid.nz, grid.ny, grid.nx)


def main() -> None:
    members, air, grid, _z = load_ensemble()
    picks, grid2, air2, ds = prepare()
    assert grid2.n_cells == grid.n_cells, "grid drift between runs"

    # gauge: laterally-uniform 1D median profile over ground cells
    ground = ~air
    vp_1d = np.array([
        np.median(members[:, k][:, ground[k]]) if ground[k].any() else np.nan
        for k in range(grid.nz)
    ])
    bg = np.broadcast_to(vp_1d[None, :, None, None],
                         members.shape).copy()
    bg = bg[0]                                 # (nz, ny, nx)

    a_min, a_max = feasible_interval(members, bg)
    width = a_max - a_min

    illum = illumination(members, air, grid, ds)
    lit = (illum >= ILLUM_MIN_PATH_M) & ground

    cert: dict = {
        "dataset": "PoroTomo stage-1 nodal P picks (GDR 924)",
        "gauge": "laterally-uniform 1D median profile vp_1d(z); "
                 "lateral-anomaly labels only",
        "vp_1d_kms": [round(float(v), 3) for v in vp_1d],
        "illum_min_path_m": ILLUM_MIN_PATH_M,
        "n_ground_cells": int(ground.sum()),
        "n_illuminated": int(lit.sum()),
        "frac_illuminated_of_ground": round(float(lit.sum() / ground.sum()), 4),
        "eps_sweep": {},
    }
    for eps in EPS_SWEEP_KMS:
        cls = classify(a_min, a_max, eps)
        n = lit.sum()
        cert["eps_sweep"][f"{eps:.2f}"] = {
            "forced_high": round(float((cls[lit] == 2).sum() / n), 4),
            "forced_low": round(float((cls[lit] == -2).sum() / n), 4),
            "forced_quiet": round(float((cls[lit] == 0).sum() / n), 4),
            "measure_dependent": round(float((cls[lit] == 1).sum() / n), 4),
        }

    cls_main = classify(a_min, a_max, EPS_MAIN_KMS)
    # display code: -2 lo, 0 quiet, 1 md, 2 hi, 9 unilluminated/air
    disp = cls_main.astype(float)
    disp[~lit] = np.nan

    # median width among illuminated cells (decomposition diagnostics)
    cert["median_width_kms_illuminated"] = round(
        float(np.median(width[lit])), 3)
    cert["median_width_kms_unilluminated_ground"] = round(
        float(np.median(width[ground & ~lit])), 3)

    with open("porotomo_decomposition_cert.json", "w") as fh:
        json.dump(cert, fh, indent=2)
    print(json.dumps(cert["eps_sweep"], indent=2))
    print(f"illuminated {cert['n_illuminated']}/{cert['n_ground_cells']} "
          f"ground cells "
          f"({cert['frac_illuminated_of_ground']*100:.1f}%)")
    print(f"median width: {cert['median_width_kms_illuminated']} km/s lit, "
          f"{cert['median_width_kms_unilluminated_ground']} km/s unlit")

    # ---- figure: 4 depth slices x 3 rows -----------------------------------
    k_slices = [1, 3, 5, 8]
    k_slices = [k for k in k_slices if k < grid.nz]
    fig, axes = plt.subplots(3, len(k_slices), figsize=(4.2 * len(k_slices), 11))
    cmap_cls = ListedColormap(["#2166ac", "#cccccc", "#fddbc7", "#b2182b"])
    norm_cls = BoundaryNorm([-2.5, -1.5, 0.5, 1.5, 2.5], cmap_cls.N)
    extent = [grid.x0, grid.x0 + grid.nx * grid.cell_m,
              grid.y0, grid.y0 + grid.ny * grid.cell_m]
    zc = grid.cell_centers_elev()
    vp_med_all = np.median(members, axis=0)
    for j, k in enumerate(k_slices):
        ax = axes[0, j]
        v = np.where(ground[k], vp_med_all[k], np.nan)
        im = ax.imshow(v, origin="lower", extent=extent, cmap="viridis")
        ax.set_title(f"median Vp @ {zc[k]:.0f} m ASL")
        plt.colorbar(im, ax=ax, shrink=0.8, label="km/s")

        ax = axes[1, j]
        w = np.where(lit[k], width[k], np.nan)
        im = ax.imshow(w, origin="lower", extent=extent, cmap="magma")
        ax.set_title("interval width (lit cells)")
        plt.colorbar(im, ax=ax, shrink=0.8, label="km/s")

        ax = axes[2, j]
        d = np.where(lit[k], cls_main[k].astype(float), np.nan)
        im = ax.imshow(d, origin="lower", extent=extent,
                       cmap=cmap_cls, norm=norm_cls)
        ax.set_title(f"classification (eps {EPS_MAIN_KMS})")
    fig.suptitle(
        "PoroTomo stage-1: possibilistic decomposition vs 1D gauge "
        "(blue=forced-low, grey=quiet, pink=measure-dep, red=forced-high; "
        "white=unilluminated)",
        fontsize=11)
    fig.tight_layout()
    fig.savefig("porotomo_decomposition.png", dpi=130)
    print("wrote porotomo_decomposition.png, porotomo_decomposition_cert.json")


if __name__ == "__main__":
    main()
