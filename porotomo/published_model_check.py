"""porotomo/published_model_check.py - test the stage-1 feasible-set
ensemble against the INDEPENDENT published Vp models of the PoroTomo
final material-property grid (GDR 1124, MESHEDTOMO_20190108.csv).

The 3D analogue of the Volve sonic-inside test. The GDR 1124 grid carries
several Vp models built by different groups with different methods:

  ID05_Vp_Thurber_velfile20170727   body-wave travel-time tomography
  ID10_Vp_Thurber20171123           (Parker/Thurber line, three vintages)
  ID13_Vp_Thurber20180504
  ID14_Vp_Nayak20180513             body-wave tomography (Nayak)
  ID18_Vp_Nayak20180607
  ID01_Vp_MatzelSweepInterfNov2017  sweep interferometry (different physics)

For each: resample its 25 m points into our 50 m cells (mean of points in
a cell), then report the fraction of illuminated cells whose published
value falls inside our per-cell ensemble interval [vp_min, vp_max]
(optionally with a tolerance for the differing grids/datums).

Reading of the result: the Thurber vintages used (a superset of) the same
travel-time data, so they are method-independent but not data-independent;
Matzel sweep interferometry is closest to fully independent. High
inside-fractions across all of them say the feasible set brackets what
other methods recovered; a model falling outside in illuminated cells
localizes a real disagreement.

Requires porotomo/data/ensemble_stage1.npz and
porotomo/data/meshedtomo_20190108.csv (GDR 1124, CC-BY 4.0).

Run:  uv run python -m porotomo.published_model_check
Outputs: porotomo_published_check.png, porotomo_published_check_cert.json
"""

from __future__ import annotations

import csv
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from porotomo.decompose_3d import load_ensemble, illumination, ILLUM_MIN_PATH_M
from porotomo.inversion3d import prepare
from porotomo.loader import X0, Y0

MESHED_CSV = "porotomo/data/meshedtomo_20190108.csv"
VP_COLUMNS = [
    "ID05_Vp_Thurber_velfile20170727",
    "ID10_Vp_Thurber20171123",
    "ID13_Vp_Thurber20180504",
    "ID14_Vp_Nayak20180513",
    "ID18_Vp_Nayak20180607",
    "ID01_Vp_MatzelSweepInterfNov2017",
]
TOL_KMS = 0.0          # strict inside test; tolerance variants in the JSON


def load_meshed(columns: list[str]):
    """Stream the 25 m grid CSV; return (xyz local-frame, {col: values})."""
    xs, ys, zs = [], [], []
    vals: dict[str, list] = {c: [] for c in columns}
    with open(MESHED_CSV) as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            xs.append(float(row["E"]) - X0)
            ys.append(float(row["N"]) - Y0)
            zs.append(float(row["H"]))      # elevation m ASL
            for c in columns:
                v = row[c]
                vals[c].append(float(v) if v not in ("NaN", "") else np.nan)
    xyz = np.column_stack([xs, ys, zs])
    return xyz, {c: np.array(v) for c, v in vals.items()}


def main() -> None:
    members, air, grid, _z = load_ensemble()
    picks, _g, _a, ds = prepare()
    vp_lo = members.min(axis=0)
    vp_hi = members.max(axis=0)
    vp_med = np.median(members, axis=0)
    illum = illumination(members, air, grid, ds)
    lit = (illum >= ILLUM_MIN_PATH_M) & (~air)

    print("loading GDR 1124 meshed grid...")
    xyz, model_vals = load_meshed(VP_COLUMNS)

    # map 25 m points -> our cells
    iz = np.round(grid.elev_to_cell(xyz[:, 2])).astype(int)
    iy = np.round(grid.y_to_cell(xyz[:, 1])).astype(int)
    ix = np.round(grid.x_to_cell(xyz[:, 0])).astype(int)
    in_box = ((iz >= 0) & (iz < grid.nz) & (iy >= 0) & (iy < grid.ny)
              & (ix >= 0) & (ix < grid.nx))
    flat = (iz * grid.ny + iy) * grid.nx + ix

    cert: dict = {
        "dataset": "GDR 1124 MESHEDTOMO_20190108 (CC-BY 4.0, "
                   "DOI 10.15121/1501544)",
        "tol_kms": TOL_KMS,
        "illum_min_path_m": ILLUM_MIN_PATH_M,
        "models": {},
    }
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for j, col in enumerate(VP_COLUMNS):
        v = model_vals[col]
        ok = in_box & np.isfinite(v)
        # units sanity: Thurber/Nayak files may be m/s or km/s
        med_raw = float(np.nanmedian(v[ok])) if ok.any() else np.nan
        scale = 0.001 if med_raw > 100 else 1.0
        # cell-average the published model
        sums = np.bincount(flat[ok], weights=v[ok] * scale,
                           minlength=grid.n_cells)
        cnts = np.bincount(flat[ok], minlength=grid.n_cells)
        cell_val = np.full(grid.n_cells, np.nan)
        has = cnts > 0
        cell_val[has] = sums[has] / cnts[has]
        cell_val = cell_val.reshape(grid.nz, grid.ny, grid.nx)

        m = lit & np.isfinite(cell_val)
        n = int(m.sum())
        if n == 0:
            cert["models"][col] = {"n_cells_compared": 0}
            continue
        inside = ((cell_val >= vp_lo - TOL_KMS)
                  & (cell_val <= vp_hi + TOL_KMS))[m]
        inside_tol = ((cell_val >= vp_lo - 0.25)
                      & (cell_val <= vp_hi + 0.25))[m]
        corr = float(np.corrcoef(cell_val[m], vp_med[m])[0, 1])
        cert["models"][col] = {
            "n_cells_compared": n,
            "unit_scale_applied": scale,
            "inside_strict": round(float(inside.mean()), 4),
            "inside_tol_0p25": round(float(inside_tol.mean()), 4),
            "corr_with_median": round(corr, 3),
            "published_median_kms": round(float(np.median(cell_val[m])), 3),
            "ensemble_median_kms": round(float(np.median(vp_med[m])), 3),
        }
        ax = axes.ravel()[j]
        ax.scatter(cell_val[m], vp_med[m], s=2, alpha=0.2)
        lim = [0, 6]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"published (km/s)")
        ax.set_ylabel("ensemble median (km/s)")
        ax.set_title(f"{col.split('_', 1)[1]}\n"
                     f"inside {inside.mean()*100:.0f}% "
                     f"(n={n}, r={corr:.2f})", fontsize=9)
    with open("porotomo_published_check_cert.json", "w") as fh:
        json.dump(cert, fh, indent=2)
    print(json.dumps(cert["models"], indent=2))
    fig.suptitle("PoroTomo: published Vp models vs stage-1 feasible intervals "
                 "(illuminated cells)", fontsize=12)
    fig.tight_layout()
    fig.savefig("porotomo_published_check.png", dpi=130)
    print("wrote porotomo_published_check.png, "
          "porotomo_published_check_cert.json")


if __name__ == "__main__":
    main()
