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
| `geophysical_invariants.md` | The stratified constraint set (Tier-1 hard / Tier-2 frame-dependent / Tier-3 observational). The possibilistic feasible set is bounded by the Tier-1 invariants. |
| `synthetic_demo.py` | Straight-ray demonstration: synthetic model → straight-ray data + noise → feasible-set ensemble → possibilistic decomposition → validation. |
| `eikonal.py` | The faithful nonlinear forward operator — Fast Marching Method Eikonal solver + ray-path Fréchet kernel. Standalone, self-tested. |
| `synthetic_demo_eikonal.py` | Eikonal demonstration: same model, the `eikonal.py` operator, a Levenberg–Marquardt inversion + smooth-perturbation feasible-set sampler, the same possibilistic decomposition. |
| `possibilistic_decomposition*.png` | Output figures (straight-ray, Eikonal). |
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
~89–93% sign-correct, 8 sign errors out of ~240 forced cells (~3%); the
measure-dependent shell captures ~76% of the blobs. Reaching this needed the
nonlinear-inversion machinery the linear case did not — Levenberg–Marquardt
for the inversion, and a feasible-set sampler that perturbs with *smooth*
fields (the raw null space of a ray operator is full of unphysical
checkerboard modes a line-integral cannot see). The decomposition itself is
identical to the straight-ray case: it is forward-model-agnostic.

Methodological subtleties surfaced and fixed while building these — all
documented in the script headers: the ensemble must sample the feasible set's
diversity (or it certifies its own shared bias); it must be sampled where the
data is actually fit; "forced" is a sign statement off the interval, not a
hard threshold; precision is honest only at the resolution length; and
feasible models must be physically plausible (smooth), not merely data-fitting.

## Status

EXPERIMENTAL — a methodology demonstration, not a library. Both the straight-ray
(linear) and Eikonal (nonlinear) operators are implemented, and the
possibilistic decomposition is validated against ground truth for each.
Natural next steps: a detailed write-up; and applying the decomposition to a
real inversion ensemble (e.g. ZTM/TFM's `runs=20` output).
