"""porotomo/geology_check.py - does the forced-high body coincide with
mapped geology?

The GDR 1124 grid carries ID15_SilerLithology: the Siler & Faulds 3D
geologic model (USGS SIM 3469; 24 wells of cuttings/core + legacy seismic)
sampled on the same 25 m mesh. That model is built from drilling and
mapping, NOT from the travel-time data - so agreement between our
forced-high cells and specific mapped units is independent corroboration
that the forced label is tracking geology, not acquisition geometry.

Test: cross-tabulate the lithologic-unit distribution of forced-high cells
(eps 0.25, illuminated) against the distribution over ALL illuminated
cells; report the units that are over-represented (lift = P(unit | forced)
/ P(unit | lit)). Unit names from the GDR 490 descriptions table.

Run:  uv run python -m porotomo.geology_check
Outputs: porotomo_geology_check.json (+ stdout table)
"""

from __future__ import annotations

import csv
import json

import numpy as np

from posdec.decomposition import feasible_interval, classify
from porotomo.decompose_3d import (load_ensemble, illumination,
                                   ILLUM_MIN_PATH_M)
from porotomo.inversion3d import prepare
from porotomo.loader import X0, Y0
from porotomo.sensitivity_3d import gauge_1d

EPS_KMS = 0.25
MESHED_CSV = "porotomo/data/meshedtomo_20190108.csv"
DESC_CSV = "porotomo/data/well_lith_descriptions.csv"


def load_lith_grid(grid):
    """Mode lithology ID per model cell from the 25 m mesh."""
    from collections import Counter
    per_cell: dict[int, Counter] = {}
    with open(MESHED_CSV) as fh:
        for row in csv.DictReader(fh):
            v = row["ID15_SilerLithology"]
            if v in ("NaN", ""):
                continue
            x = float(row["E"]) - X0
            y = float(row["N"]) - Y0
            z = float(row["H"])
            iz = int(round(grid.elev_to_cell(z)))
            iy = int(round(grid.y_to_cell(y)))
            ix = int(round(grid.x_to_cell(x)))
            if not (0 <= iz < grid.nz and 0 <= iy < grid.ny
                    and 0 <= ix < grid.nx):
                continue
            flat = (iz * grid.ny + iy) * grid.nx + ix
            per_cell.setdefault(flat, Counter())[int(float(v))] += 1
    lith = np.full(grid.n_cells, -1, dtype=int)
    for flat, cnt in per_cell.items():
        lith[flat] = cnt.most_common(1)[0][0]
    return lith.reshape(grid.nz, grid.ny, grid.nx)


def load_unit_names() -> dict[int, str]:
    names = {}
    with open(DESC_CSV, encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            try:
                names[int(row["ID"])] = (row["Code"] or "?") + " - " + \
                    row["Name/Description"].strip()
            except (ValueError, KeyError):
                continue
    return names


def main() -> None:
    members, air, grid, _z = load_ensemble()
    picks, _g, _a, ds = prepare()
    lit = (illumination(members, air, grid, ds) >= ILLUM_MIN_PATH_M) & (~air)
    a_min, a_max = feasible_interval(members, gauge_1d(members, air))
    cls = classify(a_min, a_max, EPS_KMS)
    forced_hi = (cls == 2) & lit

    lith = load_lith_grid(grid)
    names = load_unit_names()
    has = lith >= 0
    base = lit & has
    fh_ = forced_hi & has
    print(f"lit cells with lithology: {base.sum()}, "
          f"forced-high with lithology: {fh_.sum()}")

    units = np.unique(lith[base])
    rows = []
    for u in units:
        p_base = float((lith[base] == u).mean())
        p_fh = float((lith[fh_] == u).mean()) if fh_.any() else 0.0
        if p_base < 0.005 and p_fh < 0.005:
            continue
        rows.append({
            "unit_id": int(u),
            "unit": names.get(int(u), "?"),
            "frac_of_lit": round(p_base, 4),
            "frac_of_forced_high": round(p_fh, 4),
            "lift": round(p_fh / p_base, 2) if p_base > 0 else None,
        })
    rows.sort(key=lambda r: -(r["lift"] or 0))
    out = {
        "eps_kms": EPS_KMS,
        "n_lit_with_lith": int(base.sum()),
        "n_forced_high_with_lith": int(fh_.sum()),
        "source": "ID15_SilerLithology on GDR 1124 mesh "
                  "(Siler & Faulds 3D geologic model, USGS SIM 3469); "
                  "unit names GDR 490",
        "units": rows,
    }
    with open("porotomo_geology_check.json", "w") as fh2:
        json.dump(out, fh2, indent=2)
    for r in rows:
        print(f"  {r['unit'][:60]:60s} lit {r['frac_of_lit']*100:5.1f}%  "
              f"forced-high {r['frac_of_forced_high']*100:5.1f}%  "
              f"lift {r['lift']}")
    print("wrote porotomo_geology_check.json")


if __name__ == "__main__":
    main()
