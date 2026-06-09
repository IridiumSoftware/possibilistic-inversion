# Volve real-data companion — possibilistic-inversion on the Equinor walkaway VSP

**Session arc:** synthetic-only methodology → real-data integration on Volve
15/9-F-15A + 15/9-F-11 T2 → triad witness pass + artifact-correction round
→ 2D structural-misspecification diagnosis → three-way head-to-head (posdec
vs MCMC vs NN) → §6 written into the note + license attribution.

**Owner:** Aaron Green. **Co-author (this run):** Claude Opus 4.7.

---

## §1 — Computational basis

### Data

All data from the **Equinor Volve open dataset** under the *HRS Terms and
conditions for licence to data — Volve* (CC BY 4.0-derived, no-sale + covers-
all-data modifications). See note §6.6.

| Bundle | Source | Size | Path in repo (gitignored) |
|---|---|---|---|
| 15/9-F-15A VSP_VELOCITY | Equinor Volve | 139 MB | `volve/data/15_9-F-15 A/` |
| 15/9-F-15A petrophysical | Equinor Volve | 2 MB | `volve/data/05.PETROPHYSICAL INTERPRETATION/` |
| 15/9-F-11 T2 VSP_VELOCITY | Equinor Volve | 411 MB | `volve/data/15_9-F-11 T2/08.VSP_VELOCITY/` |
| 15/9-F-11 T2 petrophysical | Equinor Volve | 6 MB | `volve/data/15_9-F-11 T2/05.PETROPHYSICAL INTERPRETATION/` |

Access: `data.equinor.com` Databricks Marketplace, B2C signup, SAS URL,
AZCopy or direct ZIP download. The 2018 Azure-Blob path our initial recon
described is deprecated.

### Files added (this session)

Ingestion + picker (phases 1–2):
- `volve/__init__.py`, `volve/geometry.py` — survey-deck decoder, SEG-Y
  header reader with INT32 sentinel filtering and `original_trace_idx`
  tracking
- `volve/load_vsp.py`, `volve/load_logs.py` — SEG-Y + LAS readers
- `volve/preprocess.py`, `volve/picker.py`, `volve/picker_qc.py` — bandpass
  + STA/LTA + AIC picker pipeline
- `volve/smoke.py`, `volve/README.md`

Inversion + decomposition (phases 3–5):
- `volve/inversion_1d.py` — 1D straight-ray feasibility ensemble
- `volve/inversion_eikonal.py` — 1D Eikonal GN ensemble, lateral-
  translation-symmetry trick (one FMM per Vp)
- `volve/decompose_real.py`, `volve/decompose_real_eikonal.py` — phase 3
  + phase 4 decompositions with sonic validation
- `volve/holdout_calibration.py`, `volve/holdout_calibration_eikonal.py` —
  held-out arrival calibration
- `volve/inversion_eikonal_2d.py`, `volve/decompose_2d.py` — phase 5 2D
  joint inversion + multi-well sonic + holdout

Triad-witness-pass response (phase A + B):
- `docs/phase5_triad_brief.md` — the brief sent to the witnesses
- `volve/phase5a.py` — four diagnostics on the phase 5 snapshot
- `volve/decompose_2d_b.py` — phase B re-run with the corrections

Three-way baselines:
- `volve/mcmc_baseline.py` — emcee on linearized 1D Vp(z)
- `volve/nn_baseline.py` — MLP + MC dropout, trained on synthetic eikonal
- `volve/threeway.py` — head-to-head plot + report

Picks (CSV, tracked):
- `volve/picks/picks_z.csv` — F-15A Z-component, 1248 rows
- `volve/picks/picks_z_f11t2.csv` — F-11 T2 Z-component, 832 rows

Snapshots (`.npz`, tracked, ~1 MB each):
- `volve/picks/phase5_snapshot.npz`
- `volve/picks/phaseB_snapshot.npz`
- `volve/picks/mcmc_samples.npz`
- `volve/picks/nn_predictions.npz`

Certificates (JSON, tracked, ~5 KB each):
- `volve/picks/phase3_certificate.json` and `phase3_holdout.json`
- `volve/picks/phase4_certificate.json` (via `posdec` writer)
- `volve/picks/phase5_certificate.json` (12-member ensemble-mean baseline)
- `volve/picks/phase5a_report.json` (post-witness diagnostics)
- `volve/picks/phaseB_certificate.json` (30-member, wireline baseline,
  bore-trajectory sonic)
- `volve/picks/mcmc_certificate.json`, `volve/picks/nn_certificate.json`
- `volve/picks/threeway_report.json`

Figures (PNG, tracked):
- `volve_phase3_decomposition.png`, `volve_phase3_holdout.png`
- `volve_phase4_decomposition.png`, `volve_phase4_holdout.png`
- `volve_phase5_decomposition.png` (the inflated narrative — kept for the record)
- `volve_phase5a_diagnostics.png` (4-panel post-witness diagnostics)
- `volve_phaseB_decomposition.png` (the corrected narrative)
- `volve_mcmc_baseline.png`, `volve_nn_baseline.png`
- `volve_threeway.png` (the head-to-head)

Note revisions:
- `possibilistic_tomography_note.md` — new §6 "Demonstration 3 — real-data"
  + §6.6 "Data attribution and licence" + renumber §6/7/8 → §7/8/9, Figure 8
  → Figure 11
- `possibilistic_tomography_note.tex` — mirrored
- `possibilistic_tomography_note.pdf` — rebuilt via tectonic (2.29 MB)
- `possibilistic_tomography_note.html` — rebuilt via pandoc

### Dependencies added

```toml
# pyproject.toml additions this session:
"segyio>=1.9.13",     # SEG-Y reader
"lasio>=0.31",        # LAS reader
"dlisio>=1.0.4",      # DLIS reader (F-15A petrophysical was DLIS)
"scipy>=1.14",        # signal processing + ndimage
"emcee>=3.1",         # MCMC comparator
"torch>=2.4",         # NN comparator
```

All hermetic, `cabal.project.freeze`-style pinned via `uv.lock`.

### Build / run commands

```bash
# Per-phase reproduction (after data download into volve/data/):
uv run python -m volve.picker            # phase 2: pick F-15A Z-component
uv run python -m volve.decompose_real_eikonal  # phase 4
uv run python -m volve.holdout_calibration_eikonal
uv run python -m volve.decompose_2d_b    # phase B re-run (~ 50 min)
uv run python -m volve.phase5a           # phase A diagnostics (~ 1 min)
uv run python -m volve.mcmc_baseline     # MCMC comparator (~ 1 min)
uv run python -m volve.nn_baseline       # NN comparator (~ 2 min)
uv run python -m volve.threeway          # head-to-head (~ 2 min)

# Build the note PDF:
tectonic possibilistic_tomography_note.tex
```

### Compute envelope

All on Aaron's laptop (CPU only). Phase 4 ~ 2 min, phase B ~ 50 min,
phase A ~ 1 min, MCMC ~ 1 min, NN training+inference ~ 2 min, three-way ~ 2 min.

---

## §2 — Results

### Phase 4 — 1D F-15A — the calibrated result

- Ensemble: 30 Eikonal Gauss–Newton members on 45 bins (70 m), envelope
  1.5–5.5 km/s, smoothness correlation 250 m
- Sonic-inside (DT-EDIT vs ensemble per-bin interval): **36/45 = 80.0%**
- Held-out arrival inside (228 of 243 picks predicted inside ensemble interval):
  **93.8%**
- Holdout median residual: **+58 ms** | RMS: **158 ms**
- Signed error median (ensemble median minus sonic): **+0.29 km/s**

Methodology calibrates at the rate it claims.

### Phase 5/A/B — 2D joint F-15A + F-11 T2 — structural misspecification

| Metric | Phase 5 first read | Phase A (post-processing) | Phase B (re-run) |
|---|---|---|---|
| Sonic-inside F-15A | 10.1% (wellhead column) | 32.4% (bore) | 56.9% |
| Sonic-inside F-11 T2 | 10.0% | 38.8% | 61.9% |
| Joint holdout-inside | 22.8% | — | 31.7% |
| Holdout interval width | 21 ms | — | 24 ms |
| Holdout median residual | +37 ms | — | +36 ms |
| Forced-low (vs sonic baseline) | (0.8% vs ens-mean) | 43.9% | 44.3% |
| Forced-quiet | 79.2% (ens-mean baseline) | 22.0% | 23.2% |
| Measure-dependent | 19.1% | 33.7% | 32.0% |

Phase B's looser prior (correlation 350→700 m, envelope 1.5-5.5→1.3-6.0 km/s)
did NOT widen the predicted-time interval or shrink the residual. The 44%
forced-low is stable under prior loosening — the binding constraint is the
2D parameterization, not the prior strength. Methodology correctly diagnoses
structural misspecification.

### Three-way head-to-head — F-15A 1D, matched physics

| Method | Sonic-inside | Holdout median RMS |
|---|---|---|
| posdec (30-member feasibility interval) | **80.0%** | 179 ms |
| MCMC emcee (linearized, 90% credible) | 55.6% | 179 ms |
| MCMC (95% credible) | 64.4% | (same) |
| NN MLP (MC dropout, 90% band) | 6.7% | 490 ms |
| NN (MC dropout, 95% band) | 8.9% | (same) |

Possibilistic = widest interval, highest sonic coverage (by design).
Bayesian credible = tighter (Gaussian likelihood imposes shape constraints
beyond feasibility); same central estimate. MC dropout = narrow band that
captures within-training-distribution epistemic uncertainty only, blind to
real-data distributional shift.

### Triad witness pass — interpretation artifacts caught

Phase 5's first read was "79% forced-quiet, 10% sonic-inside, prior
dominance." Triad witnesses (Venice/Grok/ChatGPT) converged on:

1. **Anomaly baseline tautology** (CONFIRMED). 79% forced-quiet collapses
   to 22% under a wireline-sonic baseline; the original number was
   gauge-sensitive.
2. **Sonic-vs-wellhead-column comparison error** (PARTIAL FIX). Bore-
   trajectory sampling raises sonic-inside from 10% to 32–39%
   (post-processing).
3. **Illumination–decomposition alignment** (PARTIAL). r(measure-dep,
   illumination) = +0.23, r(forced-quiet, illumination) = −0.23 —
   methodology IS weakly tracking data illumination.
4. **Ensemble diversity** (GENUINE, not collapse). Leading SV explains
   16.3% of variance; 10 effective dimensions out of 12 members.

Phase B (corrected re-run) confirmed ChatGPT's leading hypothesis:
**structural misspecification** — 2D Vp(w, z) is the binding constraint,
not the prior. The 44% forced-LOW is the methodology correctly reporting
that the 2D model family cannot represent the 3D Earth at this geometry.

### Note revision

- New §6 in `possibilistic_tomography_note.md` (and TeX, PDF, HTML rebuilt).
- §6.6 = data attribution + Volve HRS T&C licence reference + non-endorsement.
- Renumber §6/7/8 → §7/8/9.
- Old Figure 8 (Tier-1 sensitivity) → Figure 11 (TeX `\ref` auto-updates).
- §9 (was §8) honest-scope opening bullet updated: "Everything here is
  synthetic" → "The original demonstrations (§§4–5) are synthetic; §6
  promotes the same machinery to real data."

---

## §3 — Verification

### Picker (phase 2) — example-tested

- N=1 smoke on three representative traces (shallow / mid / deep): picks at
  634 / 718 / 1284 ms; deep pick matches energy-envelope expectation to
  within 2 ms.
- Full file run: 1215/1248 ok (97.4% yield), 33 no_trigger (the 5 far-offset
  sources, expected SNR drop).
- Wiggle plot on FR 153 (4 deep receivers at 2836–2853 m): picks land
  exactly at the first-break onset, monotone in depth, apparent velocity
  ~2.5 km/s consistent with overburden.

### Phase 4 calibration — held-out-test verified

- Train/test split: 80/20 random with seed 20260607.
- 228/243 (93.8%) of held-out pick times fall inside the ensemble's
  predicted-time interval; the possibilistic claim ("100% coverage forecast
  should cover the truth at its stated rate") survives.
- Sonic-inside 80% is the direct measurement of "per-depth ensemble interval
  covers wireline truth."

### Phase B 44% forced-LOW persistence — robustness-tested

- Looser smoothness correlation (350 → 700 m): unchanged
- Wider envelope (1.5-5.5 → 1.3-6.0 km/s): unchanged
- Wireline-sonic baseline (independent of ensemble): drives the 44% number
- Ensemble has 10 effective dimensions, pairwise distances 0.21–0.35 km/s
  (genuine diversity, not collapse)

### Three-way head-to-head — matched-physics-controlled

- All three methods: same Eikonal FMM forward, same prior (smooth random
  Vp(z), envelope 1.5-5.5, correlation 250 m), same data (1215 picks).
- Differences = uncertainty-representation choice, not method-quality
  ranking.

### Witness-pass corrections — adversarial

- External witnesses (Venice/Grok/ChatGPT) ran independently on the same
  brief.
- Their four concerns converged; each was tested as a Phase A diagnostic
  (cheap, post-processing-only) and reported with verdict.

---

## §4 — Spec impact and forward directions

This project doesn't carry an ENGINE_SPEC; the analog is "what gets folded
into the methodology note and the next paper."

**Folded into the note (`possibilistic_tomography_note.md/.tex/.pdf/.html`):**

- New §6 Demonstration 3 — real-data on Volve walkaway VSP
- §6.6 Data attribution and licence
- Renumbering §6/7/8 → §7/8/9, Figure 8 → Figure 11
- §9 (was §8) honest scope updated to reflect that §6 is real data, not
  synthetic

**Not folded but committed for the record:**

- Phase 3 (straight-ray) — kept as a foil for phase 4; the calibration
  failure under straight-ray was diagnostic of the forward operator, not
  the methodology.
- Phase 5 first-read result — kept in the commit history as the "inflated
  narrative" before the witness pass.
- `docs/phase5_triad_brief.md` — the brief that went to the witnesses.

**Open / forward directions:**

- **Phase C (full 3D inversion).** Brian's ORSI evaluation specifically
  recommended controlled-source or cross-well experiments with independent
  ground truth. 3D refraction tomography on Volve's surface seismic is
  available (the OBS surveys ST0202 and ST10010 per HRS T&C Exhibit 1
  folder 7) but is multi-week and needs a geophysics collaborator (Zagid
  is the right collaborator per §8). Phase C-lite (pull sonic logs from
  ~10 more Volve wells, validate the structural-misspecification claim
  against multiple ground-truth wells) is the tractable solo move; phase
  C-real waits for collaboration.
- **NN tomography state-of-the-art.** The MLP + MC dropout baseline is
  intentionally a "what does a stock NN inverter give you" comparator.
  A normalizing-flow posterior (Mosser, Devilee–Curtis lineage) or a
  PINN-eikonal (Smith/Waheed) would behave differently and is worth
  separate scope.
- **Zagid + Brian outreach.** With phase 4 calibrated and phase B
  diagnosing structural misspecification on the same data, the
  collaboration case (per the note's §8 "Open frontier") is now real-data-
  grounded, not synthetic-only.

**Process discipline that emerged:**

- **Triad witness pass on consequential interpretations** is now part of
  the workflow, not an extra. Two of two times triad witnesses caught
  interpretation drift (Crane/Martin figure pass; this one). When a
  result has consequential interpretation attached, run the witness pass
  before commit-and-ship.
- **Snapshot-then-diagnose** pattern: `vp_ensemble_2d` now writes
  `.npz` snapshots so re-running interpretations on the same ensemble
  doesn't require 50 minutes of re-inversion.
- **Wireline-sonic baseline as independent anomaly reference** — kills
  the ensemble-mean tautology. Use it whenever the anomaly definition
  affects the headline number.

---

## Pointers

- Repo: `github.com/IridiumSoftware/possibilistic-inversion`
- Commit range covering this arc:
  - `ac5af9b` Volve phase 1 update (geometry rewrite + DT cross-check)
  - `379ded5` phase 2 (picker)
  - `6582b81` phase 3 (straight-ray decomposition + holdout)
  - `f772e27` phase 4 (eikonal — calibration recovered)
  - `4163857` phase 5 (2D joint — inflated first read)
  - `048426b` phase A diagnostics
  - `2e564d7` phase B re-run
  - `bade991` three-way head-to-head
  - `b049e89` note §6 markdown draft
  - `6cc7bee` note §6.6 licence
  - (this commit) TeX mirror + companion doc
- Triad brief: `docs/phase5_triad_brief.md`
- Volve licence: `equinor.com/energy/data-sharing` → HRS T&C PDF
