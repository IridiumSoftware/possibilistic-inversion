# PoroTomo 3D possibilistic inversion — companion doc

Session: 2026-06-09. Full 3D real-data extension of the methodology on the
DOE PoroTomo dataset (Brady Hot Springs, NV) — the first genuinely 3D
acquisition geometry the method has been run on (Volve §6 was 1D/2D from
walkaway VSP).

## §1 — Computational basis

**Data (all CC-BY 4.0, ~17 MB total input, gitignored under
`porotomo/data/`, re-fetchable by URL):**

| File | Source | Content |
|---|---|---|
| `AIC_Stage{1..4}_Picks.txt` | GDR 924 (DOI 10.15121/1787666), `s3://nrel-pds-porotomo/Nodal/nodal_analysis/p_picks/` | published AIC auto-picks: 165,359 P arrivals, 732 source stacks (196 vibe points × 4 stages), 238 live nodes, per-pick SNR + RMSD |
| `nodal_metadata.csv` | GDR 826 (`Nodal_continuous_metadata (1).csv`) | 240 node positions (UTM 11N) + elevations |
| `vibroseis_timing_log.xlsx` | GDR 824 | source timing log (not needed: pick files embed source coords) |
| `meshedtomo_20190108.csv` | GDR 1124 (DOI 10.15121/1501544) | PoroTomo-team 25 m property grid incl. 6 published Vp model columns |

**Code (committed):**

- `porotomo/loader.py` — pick/station parsers. Local frame x=UTM_E−327000,
  y=UTM_N−4405000; station z from the geoid-height column.
- `porotomo/inspect_data.py` — phase-1 inventory (`porotomo_geometry.png`).
- `porotomo/c/eikonal3d.c` + `porotomo/eikonal3d.py` — 3D first-order FMM in
  dependency-free C (runtime dims, ctypes, compile-on-import) with analytic
  source-ball seeding (`ball_r=5`); vectorized batch ray back-tracer
  (steepest descent, distance-to-source termination) returning sparse
  Fréchet triplets. Self-test: homogeneous-analytic + kernel-length checks.
- `porotomo/inversion3d.py` — stage-1 feasible-set ensemble. 50 m grid
  13×50×47 (30,550 cells; 27,234 ground after topography masking — air
  cells fixed at 0.34 km/s, excluded from updates). Random-reference damped
  GN; sparse LSQR on the augmented system [J; λI; (r_s·λ)L] with L the
  ground-cell graph Laplacian (`smooth_ratio=10`); λ log-bisected to
  `noise_rms_s=0.060` with plateau break; members parallelized
  (ProcessPoolExecutor, spawned seeds, reproducible for fixed cfg).
  Output `porotomo/data/ensemble_stage1.npz` (30 members).
- `porotomo/decompose_3d.py`, `porotomo/holdout_calibration.py`,
  `porotomo/published_model_check.py` — consumers; see §2.

**Run:** `python -m porotomo.inversion3d --full` (~5 min, 10 workers),
then each consumer module. Figures/certs at repo root.

**Dataset split (design):** stage 1 inverted (41,971 picks after an
apparent-velocity 300–4000 m/s gross-outlier filter dropped 149); stage 2
never touched by inversion → held-out calibration; stages 3/4 reserved
(candidate time-lapse study — pumping was modulated between stages).

## §2 — Results

**Empirical noise (stage-1 vs stage-2 matched pairs, n=35,460):**
median|dt| 34 ms → robust σ_pick ≈ 36 ms (consistent with Parker et al.
2018's 31 ms); RMS 78 ms, p95 140 ms — a heavy auto-pick outlier tail.
Signed stage offsets are small (median +2 to +6 ms for stages 2/3/4 vs 1):
no per-stage timing systematic.

**Negative result that drove a methods fix.** The first full ensemble
(damping to reference only, no roughness penalty) produced *inverted width
ordering*: median interval width 4.8 km/s in illuminated cells vs 1.8 in
unilluminated — members streaked bound-to-bound along rays (99.3%
measure-dependent). Adding the declared smoothness class (the [.. ; μL]
block) fixed ordering (0.57 lit vs 1.50 unlit), removed bound-pinning
(0%), *improved* fit (53–63 ms vs 56–61), and cut member cost ~20×
(LSQR conditioning). This is the ORSI Tier-1 finding (smoothness prior is
load-bearing) reappearing as a hard failure mode in 3D, where the
cells-per-datum ratio makes the implicit class insufficient.

**Decomposition (gauge: laterally-uniform 1D median profile; illumination
≥150 m total ray path; 5,636/27,234 ground cells lit = 20.7%):**

| eps (km/s) | forced-high | forced-low | forced-quiet | measure-dep |
|---|---|---|---|---|
| 0.15 | 13.0% | 0.4% | 3.2% | 83.4% |
| 0.25 | 7.7% | 0.1% | 13.1% | 79.1% |
| 0.35 | 4.5% | 0.0% | 27.8% | 67.7% |

The forced-high volume is a contiguous NE-trending fast body at
~1150–1050 m ASL (50–150 m depth), consistent with the structure in the
median model and with the known geometry of the Brady fault zone.

**Stage-2 holdout (38,555 picks):** ensemble-median RMS 53.9 ms vs 53.1 ms
in-sample — no overfit; bias < +6 ms both stages. Raw inside-interval
26.8%; with ±36 ms pick-noise allowance 65.4%; median predictive interval
width 38 ms. Reading: unlike Volve phase 4 (93.8% raw, intervals ≫ noise),
here the intervals are *tight relative to pick noise*, so raw inside-rate
mostly measures pick noise, not model error; the noise-allowance number and
the in≈out RMS equality carry the calibration content. The remaining
shortfall vs the ~87% Gaussian expectation is the documented outlier tail.

**Published-model check (GDR 1124 grid, illuminated cells, n≈2,666):**

| model | method | inside strict | inside ±0.25 | r | median (pub/ours) |
|---|---|---|---|---|---|
| Thurber 2017-07 (ID05) | travel-time tomo | 61% | 95% | 0.89 | 2.34 / 2.36 |
| Thurber 2017-11 (ID10≡ID13) | travel-time tomo | 70% | 98% | 0.92 | 2.36 / 2.36 |
| Nayak 2018-05 (ID14) | body-wave tomo | 73% | 98% | 0.92 | 2.43 / 2.36 |
| Nayak 2018-06 (ID18) | body-wave tomo | 71% | 97% | 0.91 | 2.39 / 2.36 |
| Matzel (ID01) | sweep interferometry | 52% | 82% | 0.88 | 2.68 / 2.36 |

(ID10 and ID13 are byte-identical in the published grid.) The feasible set
brackets the independent travel-time/body-wave models at 95–98% within
0.25 km/s with matching medians; the interferometry model is systematically
~0.3 km/s faster — different physics, real and interpretable divergence,
not a calibration failure of either.

## §3 — Verification

- *3D FMM correctness:* example-tested — `python -m porotomo.eikonal3d`
  self-test vs homogeneous analytic (mean far-field error 2.8% with ball
  seeding, ray kernels 0.0–0.6% of geometric length; PASS gate at 5%).
  First-order diagonal anisotropy (max 4.6%) documented as accepted
  operator error, same convention as the 2D Volve operator.
- *Ensemble fit:* benchmarked in-run — all 30 members 53–63 ms vs the
  60 ms achievable-floor target; recorded in `ensemble_run.log` and the
  npz (`members_rms_s`).
- *Decomposition labels:* computational — `porotomo_decomposition_cert.json`
  records gauge, illumination threshold, eps sweep, and width medians.
  Labels are gauge-relative (lateral-anomaly claims only) by construction.
- *Holdout:* example-tested — `porotomo_holdout_cert.json`; in≈out RMS
  equality (53.1/53.9 ms) is the no-overfit witness. CAVEAT recorded:
  stage-2 picks share the AIC picker and its preliminary-inversion windows
  with stage 1, so picker systematics are common-mode and untested here.
- *Published-model agreement:* example-tested —
  `porotomo_published_check_cert.json`. The Thurber models are
  method-independent but partly data-overlapping (same experiment);
  Matzel is the closest to fully independent.
- *Verification status of the forced-high body as geology:* none beyond
  the above (no well-log ground truth at Brady in our data; the USGS/
  Siler-Faulds geologic model comparison is future work).

## §4 — Spec impact / forward directions

No spec-bearing claims in this repo's note yet — drafting a "Demonstration
4 (3D real data)" section is pending Aaron's review of these results.

Forward candidates, in rough order of value:
1. **Three-way bake-off on PoroTomo** (MCMC + NN baselines, as on Volve §6.5)
   — the linearized-forward MCMC transfers directly; the NN needs 3D
   synthetic training data.
2. **Time-lapse possibilistic study** (stages 3/4): which stage-to-stage
   velocity changes are *forced* — a question type neither deterministic
   nor Bayesian treatments of this dataset have addressed; pumping was
   deliberately modulated between stages.
3. **Smoothness-class sensitivity** (ORSI Tier-1 protocol): sweep
   `smooth_ratio` and `smooth_correlation_m`, report label stability
   (Jaccard) with a random-subset control.
4. Geologic-model comparison (Siler & Faulds USGS SIM 3469 faults vs the
   forced-high body geometry).
