# Possibilistic Decomposition of a Tomographic Inversion

### Separating data-forced structure from regularization artifact

**Aaron Green** — draft prepared for Zagid Abatchev, 2026-05-17.

---

## What this is

A tomographic image is not a picture of the Earth. It is the output of a long
chain of lossy, selective steps — sparse rays, a simplified forward model, a
parameterization, a regularizer, an optimizer — and what survives that chain is
not the structure but a *filtered residue* of it. You and I have said as much
to each other already. The practical question that framing forces is the one
this note is about:

> **Given a finished inversion, which features did the data force, and which
> did the regularization invent?**

Standard practice does not answer that question. It picks a damping value,
returns one model, and attaches a posterior covariance — and every part of that
is conditional on the damping choice. Your own thesis says it plainly: "there
is no simple solution for regularization, and optimization of damping
conditions remains a highly parametrization and input dependent problem"
(ZTM/TFM, §11).

This note proposes a different reading of an inversion — a **possibilistic**
one — and demonstrates it end-to-end on synthetic travel-time tomography with
two forward operators, straight-ray and Eikonal. It is a methods communication
and a draft; the demonstrations are synthetic; nothing here is claimed as
proven. What I am claiming is that the reading is sound, that it is implemented
and validated against known ground truth, and that the place it stops working
cleanly is a real open problem worth our working on together.

---

## 1. The problem: a damping choice is not a fact

Tomographic inversion is ill-posed. Many models fit the data within noise. To
return *one* model you must regularize — damp, smooth, prefer a reference —
and the model you get is as much a property of that choice as of the data.

The honest consequence: a feature in a tomographic image belongs to one of
three classes, and they are not the same kind of thing.

| Class | Meaning |
|---|---|
| **Forced** | Present in *every* model consistent with the data and the hard physical bounds. Data-determined. Independent of the damping choice. |
| **Forbidden** | Present in *no* such model. |
| **Measure-dependent** | Present in *some* consistent models and absent in others. The data do not determine it — which sign or structure you see is set by the regularization. Such a feature may still be physically real: measure-dependence diagnoses *underdetermination*, not falsehood. |

A posterior covariance does not draw this line. It reports a spread *under one
prior and one damping*; it cannot tell you that a given feature would vanish
under a different, equally defensible choice. The damping problem of §11 is, in
this language, a symptom of reading an inversion through the wrong layer: it is
the search for the "right" measure-layer choice to extract a model, when the
features that *depend* on that choice are exactly the ones the data does not
determine.

---

## 2. The possibilistic frame

The distinction in §1 is the **possibilistic / probabilistic** split, taken
from the two-layer discipline of the Closure Forces Structure programme
(`inverse_born_methodology.md`) and carried over to inversion:

- The **possibilistic layer** asks what is *forced or forbidden* by the data
  and the hard physical constraints alone. Its answers are unconditional — they
  do not depend on a prior, a damping value, or a measure.
- The **probabilistic layer** asks how a measure distributes weight over what
  the possibilistic layer permits. Its answers are conditional on that measure.

A posterior covariance lives entirely in the probabilistic layer. The
possibilistic layer is the one that answers the §1 question, and it is the one
standard tomography leaves on the table.

The split has a natural geometry, and the next three figures build it up one
idea at a time.

**Figure 1 — the feasible set.** The data and the hard constraints do not pick
out a model; they pick out a *set* F of models — every model that fits within
noise and respects the bounds. Project F onto a feature's axis and the feature
falls into one of two classes. It is **forced** when the projection lies
entirely on one side of zero: every model consistent with the data agrees on
its sign, and no prior is needed to settle it. It is **measure-dependent** when
the projection straddles zero: the data leave the sign open, and any single
value reported there is fixed by the measure you add, not by the data.

![Figure 1. The data give a feasible set F of models, not a point. Projected onto a feature's axis, F either clears zero — the feature is forced, its sign holds for every model in F — or straddles zero — the feature is measure-dependent, its sign fixed by the measure, not the data. Both terms are defined on the figure.](fig_feasible_set.png)

*Figure 1. The data give a set, not a point. A feature is forced when F's
projection clears zero and measure-dependent when it straddles zero — the two
terms this note turns on, defined geometrically.*

**On the word "possibilistic."** I use it in a deliberately narrow sense — the
crisp, set-membership corner of possibility theory, not a graded possibility
distribution. F is an ordinary feasible set: a model is in it or it is not. A
feature is *forced* exactly when a fact holds for every model in F (necessity 1,
in possibility-theory terms) and *measure-dependent* when F neither forces nor
forbids it. No fuzzy membership is invoked; the only structure used is the set
F and the all-or-nothing quantifier over it.

**Figure 2 — two reports.** The same F can be turned into a reported answer two
ways. The *possibilistic report* uses exactly F: forced features pinned to a
sign, measure-dependent features returned as intervals — no more information
than F carries, and no less. The *Bayesian report* uses F plus a measure μ — it
places μ on F and reports its mean and credible interval. μ is information
beyond F, and two defensible choices of μ can disagree on a measure-dependent
feature's sign, so that choice needs a justification of its own. On a forced
feature the two reports agree; the divergence is confined to the
measure-dependent ones.

![Figure 2. The same feasible set F, turned into a reported answer two ways. The possibilistic report uses exactly F. The Bayesian report uses F plus a measure μ; two defensible μ agree on the forced feature A but disagree on the sign of the measure-dependent feature B.](fig_two_reports.png)

*Figure 2. Two reports from one feasible set. Possibilistic: use exactly F.
Bayesian: use F plus a measure μ — defensible μ agree where F forces the answer
and can conflict where it does not.*

**Figure 3 — the layers compose.** The two are not rivals run side by side;
they compose in sequence. The possibilistic layer runs first and bounds which
measures are admissible — a measure is admissible only if it is supported on F.
The Bayesian layer then runs inside that bound: sweep every admissible measure
and the spread of the answer, the *measure-uncertainty*, can never exceed F's
extent. On a forced feature that bound clears zero, so no admissible measure
can move the sign; on a measure-dependent feature it straddles zero and the
sign is genuinely open. Possibilism does not compete with Bayes — it brackets
it.

![Figure 3. The two layers in sequence. Step 1: the possibilistic layer fixes F and bounds the admissible measures — a measure is admissible only if supported on F. Step 2: the Bayesian sweep runs inside that bound, and F's extent caps the measure-uncertainty.](fig_bounded_uncertainty.png)

*Figure 3. The layers compose. Possibilism runs first and bounds the admissible
measures; the Bayesian sweep then stays inside that bound. The feasible
interval below is exactly that bound.*

The object that carries the possibilistic content is the **feasible
interval**. Take an ensemble of models that each fit the data within noise and
respect the hard bounds. For each cell, record the interval `[a_min, a_max]` of
the anomaly across the ensemble. That interval is F projected onto the cell's
axis — exactly the bound Figure 3 draws:

- `a_min > 0` everywhere in the ensemble → **forced-high** (no data-consistent
  model makes this cell non-positive);
- `a_max < 0` → **forced-low**;
- the interval straddles zero → **measure-dependent** (the sign itself is not
  data-forced);
- the interval sits inside a small band around zero → **forced-quiet**.

Figure 4 puts the classification on a concrete ensemble. The same ensemble,
read two ways: the probabilistic reading collapses it to a mean and an error
band; the possibilistic reading keeps the feasible interval and classifies it.
The big feature is forced; the flanks and the small bump are measure-dependent
— and the probabilistic band does not distinguish them.

![Figure 4. The same feasible ensemble read probabilistically (mean + error band) and possibilistically (feasible interval, classified into forced-high / forced-low / measure-dependent).](fig_schematic.png)

*Figure 4. The same feasible ensemble, read two ways. Left: one model, one
uncertainty band. Right: the feasible interval, classified.*

**On prior art — said straight.** This is not unprecedented, and pretending
otherwise would not survive five minutes of your scrutiny. Computing bounds on
what a model *can* be, rather than a single estimate, is the spirit of
Backus–Gilbert extremal inversion (1968); the under-determined directions are
the null space of resolution analysis; multi-model joint coupling has the
Gramian-constraint literature (Zhdanov, 2012 onward). And a transdimensional /
reversible-jump MCMC posterior ensemble (Bodin & Sambridge, 2009) already
*contains* this information — forced structure appears as high-consensus,
measure-dependent structure as sign-variable posterior mass. The epistemic
intuition is old. What I am putting forward is narrower: not a new inverse-theory
principle but a *reporting discipline* — the forced / measure-dependent split
made an explicit, first-class output; measure-dependence treated as the
operational diagnostic of *underdetermination*; and the whole run as a
transferable, documented procedure on top of arbitrary inversion machinery,
rather than the informal robustness check practitioners already perform
inconsistently. The contribution is the operationalization, not the distinction.

---

## 3. Method

**The feasible set.** A model is *feasible* if it (a) fits the data to within
the noise level and (b) satisfies the hard physical bounds. The bounds are not
left implicit: `geophysical_invariants.md` stratifies them into Tier-1
(frame-independent — positivity, first-arrival minimality, a generous velocity
envelope; used freely), Tier-2 (frame-dependent — typical mantle/crust values,
reference Earth models, petrophysical relations; soft priors only), and Tier-3
(the observational context). The possibilistic feasible set is bounded by
Tier-1 alone. Worth noting in passing: TFM currently clamps node velocities to
7.5–9.5 km/s and initializes from IASP91 — Tier-2 values used as hard limits.
That is the layer-confusion the stratification is meant to catch.

**Sampling it.** Generate many feasible models by inverting toward many random
reference models, each inversion taken to the point where its misfit equals the
noise level. The references diversify the data-null directions; the
data-resolved directions are common to all. Then read the feasible interval
and classify it (§2).

**The decomposition is forward-model-agnostic.** It operates on the ensemble of
models; it does not care how they were produced. That is the point of the two
demonstrations below: the *same* decomposition code runs on a linear straight-
ray operator and a nonlinear Eikonal operator.

---

## 4. Demonstration 1 — straight-ray operator (linear)

A synthetic velocity model: one large tilted high-velocity slab, one broad
low-velocity zone, three small sub-resolution blobs. Synthetic first-arrival
times with 1.2% noise, dense four-edge ray coverage. The feasible set is
sampled exactly per §3 (the linear operator admits an exact eigendecomposition,
so the feasible set is parametrized cleanly).

Figure 5 is the result. The forced-sign cores sit on the true features — panel
(e), the black forced-high contour lies inside the true slab, the white
forced-low contour inside the true low-velocity zone. The numbers:

- **Forced cores are correct.** ~7 sign errors within the resolution length out
  of ~300 forced cells (~2%). What the method certifies, the ground truth
  bears out.
- **The measure-dependent shell captures ~89% of the sub-resolution blob
  cells** — it correctly flags the detail the data cannot pin down.
- **The forced set is conservative**: ~19% of the grid is sign-certain. This is
  not a weakness. It is the method saying out loud what is true — most of a
  tomographic image is *not* forced.

![Figure 5. Straight-ray demonstration: ground truth, feasible-interval bounds, the forced-sign decomposition, the forced cores against true anomaly, and the measure-dependent shell against the blob centres.](possibilistic_decomposition.png)

*Figure 5. The straight-ray (linear) demonstration. Panel (d) is the
decomposition; panel (e) shows the forced cores landing on the true features.*

---

## 5. Demonstration 2 — Eikonal operator (nonlinear)

The straight-ray operator is exact only in a homogeneous medium. The faithful
operator is the Eikonal first-arrival solver — the operator class your FMM.cpp
implements. First-arrival rays bend: toward fast structure, away from slow
(Figure 6). `eikonal.py` provides it (Fast Marching Method solver + ray-path
Fréchet kernel), standalone and self-tested.

![Figure 6. First-arrival rays through the synthetic model: Eikonal rays (solid) bend toward the fast slab; the straight-ray approximation (dashed) ignores it.](fig_ray_bending.png)

*Figure 6. Why the forward operator matters. Solid: Eikonal first-arrival rays,
bending through the medium. Dashed: the straight-ray approximation.*

Because travel time is now a nonlinear functional of slowness, the one-shot
linear solve is replaced by an iterative Levenberg–Marquardt inversion that
recomputes the ray paths through the current model — the DGN + FMM structure of
your own pipeline. Figure 7 is the result, and the decomposition code is
*identical* to Demonstration 1:

- **Forced cores ~89% sign-correct within the resolution length**, but ~71–81%
  strict (cell-exact). The gap is resolution-length blur — a forced cell one or
  two cells off a true feature edge, not a sign mistake — but the strict figure
  is the honest co-headline; the within-resolution number alone overstates the
  precision.
- **The measure-dependent shell captures ~76% of the blobs.**

These figures are from the feasible-set sampler at its default operating point,
and at that operating point the forced set itself over-claims: an adversarial
stress test (§6 point 6; `witness_pass.md`, RWC-2) finds ~41% of forced cells
admit a feasible, equally-smooth, opposite-sign model. The decomposition layer
is sound and forward-model-agnostic; what is coverage-limited is the
feasible-set *sampling* that feeds it (§7).

The decomposition transferred across two genuinely different forward operators
without change. That is the load-bearing result of this note: the possibilistic
reading is a property of *inversions*, not of a particular operator.

![Figure 7. Eikonal demonstration: the same possibilistic decomposition, with the nonlinear FMM forward operator and a Levenberg-Marquardt inversion ensemble.](possibilistic_decomposition_eikonal.png)

*Figure 7. The Eikonal (nonlinear) demonstration — same decomposition, faithful
operator.*

---

## 6. What building it surfaced — the discipline the method needs

A method that always says "yes" is a gimmick. This one has real failure modes,
and finding them is how I know it has content. Each was surfaced by a synthetic
run failing honestly; each is now documented in the script headers.

1. **The ensemble must sample the feasible set's diversity, or it certifies its
   own bias.** An ensemble built by varying damping *strength* alone shares a
   bias in the data-null directions; the intersection then stamps that shared
   bias "forced." The fix is many random references.
2. **Sample where the data is actually fit** — at the misfit = noise level. An
   over-damped ensemble has had data-resolved structure regularized away.
3. **"Forced" is a sign statement off the interval, not a hard threshold.** A
   single anomaly-magnitude cutoff is brittle; the feasible interval is the
   honest object.
4. **Precision is honest only at the resolution length.** Every feasible model
   inherits the same forward-operator blur; the forced core is correct to
   within roughly one cell, not cell-exact.
5. **Feasible models must be physically plausible — smooth.** The raw null
   space of a ray (line-integral) operator is dominated by high-frequency
   checkerboard modes the operator cannot see. Those fit the data but are not
   admissible Earth models; perturbing freely in them injects unphysical
   speckle. A feasible model is data-fitting *and* smooth.
6. **The forced set is coverage-gated — and the gate is measurable.** A forced
   label asserts that *every* feasible model agrees; an ensemble that
   under-samples the feasible set will agree spuriously and over-claim. Two
   checks quantify this (`witness_pass.md`, RWC-1 and RWC-2): the forced set
   converges only above an ensemble-coverage threshold — here N ≈ 250 sampled
   models for a 40×40 grid — and at the default operating point an adversarial
   stress test finds ~41% of forced cells *false-forced* (a feasible,
   equally-smooth, opposite-sign model exists). A forced claim is honest only
   with its coverage-adequacy curve attached.

These are not patches. They are the methodology's anti-anchoring discipline —
the same one that, in the Closure programme, keeps a derivation from quietly
tuning itself to the answer it wants.

---

## 7. The open frontier — and where you come in

Here is the honest seam, and it is the reason this is a note to *you* and not
a finished claim.

The linear straight-ray case has a clean, exact answer: the operator is fixed,
`GᵀG` eigendecomposes once, and the feasible set is parametrized directly. The
nonlinear case does not. Sampling the feasible set of a nonlinear inverse
problem — getting genuine diversity in the data-null directions without
injecting artifacts — is itself a hard problem. Building Demonstration 2 walked
straight into it: a naive Levenberg–Marquardt ensemble shares an
inversion-dynamics bias; fixing that needs explicit, physically-constrained
feasible-set sampling. I have a working version, but I would not call the
nonlinear feasible-set sampler solved.

This is exactly the territory of transdimensional / reversible-jump MCMC
tomography (Bodin & Sambridge, 2009 onward) — methods built precisely to
produce a genuine posterior ensemble. And it is where ZTM already lives: your
`runs=20` top-level Monte Carlo *is* a feasible-set sampler. A coverage study on
the synthetic problem (`witness_pass.md`, RWC-1) now puts a number on the
question: the forced set stabilizes only near N ≈ 250 sampled models, and a
20-run ensemble sits at roughly three times the converged forced-set size — 20
runs is far short of coverage adequacy. So the open question is no longer
*whether* the sampler needs upgrading but *to what*: whether a transdimensional
sampler reaches the coverage the forced/measure-dependent split requires, on
real data, at tractable cost.

That question is not mine to answer from the outside. It needs your inversion
expertise, your code, and your data. My contribution is the possibilistic
reading and the formal scaffolding behind it; yours is the geophysics and the
instrument. Neither half is sufficient alone. That is the collaboration I am
proposing — not "adopt this method," but "here is a method, validated in the
linear case, and here is the concrete nonlinear problem your own pipeline is
already engaging. Let us solve that part together."

---

## 8. Honest scope

- Everything here is **synthetic**. The demonstrations validate the *method*
  against known ground truth; they are not a result about the real Earth.
- The Eikonal solver is **first-order FMM**. Its ~2% mean accuracy is fine for
  a forward-model-agnostic demonstration; your FMM-VFD hybrid is the
  production-grade version.
- Two dimensions, modest velocity contrasts, a single noise realization. None
  of the conclusions depend on scale, but none have been *tested* at scale.
- The forced/measure-dependent split is **operator-relative**: it reports what
  is forced *given a forward operator*. An operator with systematic error
  propagates that error into the forced set. This is not a defect of the
  method — it is the method correctly reflecting the operator — but it means
  the operator's fidelity is load-bearing.
- The **forced set is coverage-gated**. It is trustworthy only above an
  ensemble-coverage threshold (RWC-1: N ≈ 250 here); below it the sampler
  over-claims (RWC-2: ~41% of forced cells false-forced at the default
  operating point). The demonstrations are reported at that default point and
  so over-state the forced set. The decomposition layer is sound — what is
  coverage-limited is the feasible-set *sampling* that feeds it; a forced claim
  must travel with its coverage-adequacy curve.

---

## Pointers

- `geophysical_invariants.md` — the stratified constraint set.
- `synthetic_demo.py`, `synthetic_demo_eikonal.py` — the two demonstrations.
- `eikonal.py` — the standalone Eikonal forward operator (self-test:
  `uv run python eikonal.py`).
- `decomposition_exact.jl` — the decomposition layer formalized in exact
  `Rational{BigInt}` arithmetic (Julia); `classify` proven total, exclusive,
  and exhaustive, the feasible-interval properties exact-verified.
- `c/` — a dependency-free C port of the whole method: the forward operators,
  the Levenberg–Marquardt inversion over a hand-rolled dense-Cholesky module,
  the feasible-set sampler, and the decomposition, pulled together by
  `possibilistic_inversion.c`. Build and run: `cd c && make && ./pi`.
- `witness_pass.md` — the synthesis witness pass: an adversarial external review
  of the method by three independent models, and the two Required Witness Checks
  it carried, `rwc1_forced_stability.py` and `rwc2_disconnected_test.py`.
- `inverse_born_methodology.md` (Closure Forces Structure programme) — the
  source of the two-layer possibilistic / probabilistic discipline.
- Bodin, T. & Sambridge, M. (2009), *Seismic tomography with the reversible
  jump algorithm*, Geophys. J. Int. — the transdimensional-sampling reference.
- Abatchev, Z. (2019), ZTM/TFM, UCLA — the joint seismic+gravity inverter this
  note is in conversation with.
