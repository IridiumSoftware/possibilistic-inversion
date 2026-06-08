# volve - real-data integration on the Equinor Volve VSP (15/9-F-15A)

This subpackage takes the shipped possibilistic-inversion methodology
(`posdec`) from synthetic-only into real-data territory. The target dataset
is the **Equinor Volve open release**, well **15/9-F-15A**, Vertical
Incidence VSP (acquired 5 Jan 2009 by READ Well Services for StatoilHydro),
paired with the well's wireline sonic log as independent Vp ground truth.

## Workflow phases

| Phase | Status | What |
|-------|--------|------|
| 1     | landed | Ingestion + real geometry decoded from SEG-Y headers + sonic ground-truth cross-checked. |
| 2     | open   | First-arrival picking on the 1248 raw VSP traces (Z component). |
| 3     | open   | `posdec` decomposition on the picks; validate forced/measure-dependent split against DT-EDIT; held-out arrival calibration. |
| 4     | open   | (optional) rj-MCMC + simple NN-tomography parallel comparators. |

## What landed in phase 1

**Real geometry (from `VSPNI_RAW_2.SEGY` headers):**

- **312 FieldRecords (shots) × 4 sensors per record = 1248 traces** on the
  Z component. Same shape on X (`VSPNI_RAW_3.SEGY`) and Y
  (`VSPNI_RAW_4.SEGY`).
- **145 unique source positions**, two clusters:
  - main bank at ~430-620 m offset from the wellhead (140 sources)
  - far-offset bank at ~1500-1600 m offset (5 sources)
- **224 unique receiver elevations** spanning 130.8 - 3134.7 m below
  ElevationScalar datum. Intra-array spacing ~15 m.
- **Wellhead at (434927, 6477976) m** in the survey's UTM-like frame.
- **Well is significantly deviated**: from wellhead at depth ~0 the well
  curves to ~1100 m horizontal offset by 3134 m depth.
- Z + horizontals: 5000 samples x 1 ms = 5 s record. Hydrophone monitor on
  `VSPNI_RAW_1.SEGY`: 1997 samples x 0.25 ms.

**Sonic ground truth (cross-checked):**

| Source | Coverage | Use |
|---|---|---|
| `TZV_DEPTH_MD_COMPUTED_1.LAS` (DT-EDIT) | MD 54.9 - 4070 m, 0.1524 m step | PRIMARY ground truth - full well |
| `WLC_PETRO_COMPUTED_INPUT_1.DLIS` (DT) | MD 2607 - 4070 m, 0.1524 m step | independent cross-check on reservoir |

On the overlap (MD 2607-4070 m), the two sources agree to median residual
+0.001 us/ft, std 3.36 us/ft over 9369 samples. Above MD 2607 m the LAS
DT-EDIT is the only source; the cross-check ratifies it on the deep half.

## Data layout

`volve/data/` is gitignored. After downloading both bundles, the tree is:

```
volve/data/
  15_9-F-15 A/
    VSPNI_RAW_1..4.SEGY                 # raw VSP: hydrophone + Z/X/Y geophones
    VSPNI_COMPUTED_1..30.SEGY           # READ's processed result (Schlumberger)
    VSPNI_COMPUTED_31.LAS               # processed VSP corridor stack
    TZV_DEPTH_MD_COMPUTED_1.LAS         # depth-domain Vp + sonic (DT-EDIT / DT-CKS)
    TZV_DEPTH_MD_CHECKSHOT_1.ASC        # checkshot time-depth pairs
    TZV_TIME_CHECKSHOT_1.ASC
    TZV_TIME_SYNSEIS_1..4.LAS           # synthetic seismograms
    VELOCITY_REPORT_1..4.ASC            # text reports (geometry, Q, sonic-cal, time-index)
    VSP_REPORT_1.PDF / 2.PDF            # survey + processing documentation
    *_INF_*.ASC                         # per-file metadata
  05.PETROPHYSICAL INTERPRETATION/
    WLC_PETRO_COMPUTED_INPUT_1.DLIS     # raw wireline logs (DT, RHOB, GR, NPHI, ...)
    WLC_PETRO_COMPUTED_OUTPUT_1.DLIS    # interpreted petrophysics
    PETROPHYSICAL_REPORT_1.PDF
    geomod09/                           # facies + perm
```

## Quick start (after data drop)

```bash
uv run python -m volve.geometry "volve/data/15_9-F-15 A/VSPNI_RAW_2.SEGY"
uv run python -m volve.smoke
```

From a Python session:

```python
from volve.geometry import load_geometry_from_segy, summarize, plot_geometry
geo = load_geometry_from_segy("volve/data/15_9-F-15 A/VSPNI_RAW_2.SEGY")
print(summarize(geo))

from volve.load_logs import load_well_log, summary
log = load_well_log("volve/data/15_9-F-15 A/TZV_DEPTH_MD_COMPUTED_1.LAS",
                    dt_curve="DT-EDIT", depth_curve="MD")
print(summary(log))
```

## Access (one-time, ~10 min)

Equinor moved Volve onto **Databricks Marketplace** since the 2018 release;
the older Azure-Blob + SAS-URL path our initial recon described is
deprecated. Current path:

1. `data.equinor.com` -> Databricks Marketplace (B2C / Microsoft signup).
2. Locate the Volve listing; the **inventory is exposed as .xlsx files**
   (a master `VOLVE_INVENTORY.xlsx` plus per-well sheets).
3. Per-well bundles download as ZIPs. For this integration:
   - `15_9-F-15 A.zip` (~140 MB) - VSP + sonic + checkshot + reports
   - `05.PETROPHYSICAL INTERPRETATION.zip` (~2 MB) - raw wireline DLIS
4. Unzip both into `volve/data/`.

The master inventory's "08. VSP & VELOCITY LOGS" row marks **N/A** for
the F-1 / F-1A / F-1B / F-1C wells; an earlier recon claimed walkaway VSP
was on 15/9-F-1A, but the inventory says otherwise. The two wells that
actually have VSP data are **15/9-F-15 A** (this bundle) and
**15/9-F-11 T2** (a secondary option, similar SEG-Y + checkshot).

## License

Equinor's license is more permissive than commonly cited - permits
commercial use of Adapted Material, requires attribution and
ShareAlike-style propagation of terms to derived data. Derived products
(picks, inverted models, figures) can be published with attribution.

## Conventions

- All distances in metres; all times in seconds.
- SEG-Y `CoordinateScalar` = -10 (divide xy by 10); `ElevationScalar` =
  -10000 (divide by 10000).
- DLIS DEPTH stored in `0.1 in`; multiply by 0.00254 m/unit.
- LAS DT in us/ft; Vp(km/s) = 304.8 / DT.
- Well datum: Kelly Bushing 54.9 m above MSL; sea bed at 91 m below
  surface; water velocity 1500 m/s.
