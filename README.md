# possibilistic-inversion

A possibilistic treatment of uncertainty in tomographic inversion — applying
the two-layer discipline of closure-v5's `inverse_born_methodology.md`
(possibilistic = forced/forbidden by structure; probabilistic = measure over
what's permitted) to seismic-style travel-time tomography.

**Motivation.** Standard tomography is probabilistic: pick a damping, report
one model + a posterior covariance — all conditional on the regularization
choice. The possibilistic reading instead asks, of an ensemble of models that
each fit the data within noise: which features appear in *every* one
(**forced** — data-certain) and which only in *some* (**measure-dependent** —
artifacts of the regularization choice).

## Contents

| File | What |
|---|---|
| `possibilistic_tomography_note.md` | **The write-up** — the scientific communication: the possibilistic decomposition, both demonstrations, the open frontier. |
| `witness_pass.md` | The synthesis witness pass — adversarial external review of the method by three independent models; brief, returns, and synthesis collected as the pass proceeds. |
| `geophysical_invariants.md` | The stratified constraint set (Tier-1 hard / Tier-2 frame-dependent / Tier-3 observational). The possibilistic feasible set is bounded by the Tier-1 invariants. |
| `synthetic_demo.py` | Straight-ray demonstration: synthetic model → straight-ray data + noise → feasible-set ensemble → possibilistic decomposition → validation. |
| `eikonal.py` | The faithful nonlinear forward operator — Fast Marching Method Eikonal solver + ray-path Fréchet kernel. Standalone, self-tested. |
| `synthetic_demo_eikonal.py` | Eikonal demonstration: same model, the `eikonal.py` operator, a Levenberg–Marquardt inversion + smooth-perturbation feasible-set sampler, the same possibilistic decomposition. |
| `decomposition_exact.jl` | Exact-arithmetic (`Rational{BigInt}`) formalization of the decomposition layer in Julia — `classify` proven total/exclusive/exhaustive; the feasible-set monotonicity properties exact-verified. |
| `c/` | C port — dependency-free `.c`/`.h` modules (`eikonal`, `straightray`, `decomposition`, `linalg`, `inversion`) and a master `possibilistic_inversion.c` that pulls them in. Build: `cd c && make`. Runs the full pipeline: forward operator → noisy data → feasible-set sampler (Levenberg–Marquardt + smooth perturbations) → possibilistic decomposition → validation. |
| `make_figures.py` | The explanatory figures for the write-up (`fig_schematic.png`, `fig_ray_bending.png`). |
| `*.png` | Figures: the two demonstrations and the two explanatory figures. |
| `pyproject.toml`, `uv.lock` | Pinned environment (numpy, matplotlib). |

## Run

```
uv run python synthetic_demo.py          # straight-ray operator
uv run python synthetic_demo_eikonal.py  # Eikonal (FMM) operator, ~1 min
uv run python eikonal.py                 # operator self-test
```

## What the demos show

Synthetic model: one large high-V slab, one broad low-V zone, three
sub-resolution blobs. The possibilistic object is the per-cell **feasible
interval** [a_min, a_max] of the anomaly; "forced" is read off it as a sign
statement (forced-high / forced-low / forced-quiet / measure-dependent).

**Straight-ray operator** (`synthetic_demo.py`) — the linear case, the clean
validated result. The forced-sign cores recover the true features (~45% / 40%
recall of the genuine high/low anomalies; ~7 sign errors within the resolution
length out of ~300 forced cells, ~2%); the measure-dependent shell captures
~89% of the sub-resolution blobs; ~19% of the grid is sign-certain. Most of a
tomographic image is *not* forced — that is the methodology's point.

**Eikonal operator** (`synthetic_demo_eikonal.py`) — the faithful nonlinear
case (rays bend). Same possibilistic decomposition; the forced cores come out
~89% sign-correct within the resolution length (~71–81% strict, cell-exact);
the measure-dependent shell captures ~76% of the blobs. Reaching this needed
the nonlinear-inversion machinery the linear case did not — Levenberg–Marquardt
for the inversion, and a feasible-set sampler that perturbs with *smooth*
fields (the raw null space of a ray operator is full of unphysical
checkerboard modes a line-integral cannot see). The decomposition itself is
identical to the straight-ray case: it is forward-model-agnostic. These figures
are at the sampler's default operating point; an adversarial witness pass
(`witness_pass.md`) shows the forced set there is coverage-gated and
over-claims — see the note's §6–§8.

Methodological subtleties surfaced and fixed while building these — all
documented in the script headers: the ensemble must sample the feasible set's
diversity (or it certifies its own shared bias); it must be sampled where the
data is actually fit; "forced" is a sign statement off the interval, not a
hard threshold; precision is honest only at the resolution length; and
feasible models must be physically plausible (smooth), not merely data-fitting.

## Status

EXPERIMENTAL — a methodology demonstration, not a library. Both forward
operators (straight-ray, Eikonal) are implemented and the possibilistic
decomposition is validated against ground truth for each; the decomposition
layer is additionally formalized in exact `Rational{BigInt}` arithmetic
(`decomposition_exact.jl`), its core properties proven or exact-verified. The
write-up (`possibilistic_tomography_note.md`) is drafted. A dependency-free C
port (`c/`) runs the whole method end to end — forward operator, a
Levenberg–Marquardt inversion over a hand-rolled dense-Cholesky linear-algebra
module, the feasible-set sampler, and the decomposition — reproducing the
Python's forced-core accuracy (~89% within the resolution length). Next:
applying the decomposition to a real inversion ensemble (e.g. ZTM/TFM's
`runs=20` output).
