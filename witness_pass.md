# Synthesis Witness Pass — possibilistic-inversion

**Opened 2026-05-17.**

A witness pass routes the claims a project cannot validate from inside to
independent external review. This one stress-tests the possibilistic-inversion
method before it is offered as a collaboration seam. Three independent language
models each take a role; their returns are collected here verbatim and
synthesized at the end. The repository is public, so each witness reads the
actual artifact — `possibilistic_tomography_note.md` and the code — not a
summary of it.

This document is committed at each stage. The git history *is* the record of
the pass: brief opened → each return logged → synthesis written.

## Roles

| Witness | Model | Role |
|---|---|---|
| A | Grok | **Edge-witness (Φ).** Red-team. The strongest available attack — the counterexample, the unstated assumption, the failure the synthetic demonstrations did not hit. Not a balanced review. |
| B | Gemini | **Synthesis.** Coherence of the whole; what is load-bearing versus over-claimed; whether the framing does real work or relabels known analysis. |
| C | ChatGPT | **Domain-probabilist witness.** Stands in for the skeptical Bayesian geophysicist. Whether a probabilist would accept the argument, and the strongest probabilist rebuttal to it. |

Claude metabolizes — prepares this brief, collects the returns, writes the
synthesis. It does not witness its own work.

## The artifact under witness

possibilistic-inversion applies the two-layer epistemic discipline of the
Closure Forces Structure programme's `inverse_born_methodology.md` to
uncertainty in travel-time tomography. Standard inversion is *probabilistic*:
pick a regularization, report one model plus a posterior covariance — all
conditional on the choice. The *possibilistic* reading instead decomposes an
ensemble of data-fitting models into the **forced** structure (a feature
present, with constant sign, in *every* feasible model — data-determined) and
the **measure-dependent** structure (a feature whose sign varies across the
feasible set — an artifact of the regularization choice). The method is
demonstrated on a linear (straight-ray) and a nonlinear (Eikonal) forward
operator, formalized in exact arithmetic (Julia), and ported to dependency-free
C. Full statement: `possibilistic_tomography_note.md`.

## Claim ledger

| ID | Claim | Current evidence |
|---|---|---|
| P-1 | Forced vs measure-dependent is a distinction the posterior covariance does not draw. | Argued; demonstrated synthetically. |
| P-2 | The decomposition is forward-model-agnostic — an identical `classify` layer serves the linear and the nonlinear operator. | Demonstrated on both. |
| P-3 | `classify` is total, mutually exclusive, and exhaustive. | Proven — exhaustive enumeration in exact `Rational{BigInt}` arithmetic (`decomposition_exact.jl`). |
| P-4 | Linear case: the forced-sign cores recover the true features; ~2% sign error within the resolution length. | Synthetic validation against known ground truth. |
| P-5 | Nonlinear case: the forced cores are ~89% sign-correct within a 2-cell resolution length (~71% strict per-cell). | Synthetic validation; the Python and the independent C port agree. |
| P-6 | The nonlinear feasible-set sampler is *not* solved — random references + smooth perturbations is a working heuristic, not a guaranteed cover of the feasible set. | Stated open (note §7). |
| P-7 | The contribution is narrower than, and downstream of, Backus–Gilbert extremal inversion, Zhdanov Gramian constraints, and Bodin–Sambridge transdimensional tomography: the novel part is the *disciplined reading* in which measure-dependence is the operational diagnostic of artifact. | Argued. |

## Witness checks

Each check is a question the project cannot settle from inside.

- **W-1 (Grok).** Attack P-6. Construct the scenario in which random-reference +
  smooth-perturbation sampling systematically misses a region of the feasible
  set and therefore stamps a measure-dependent feature as **forced** — a
  *false-forced*, the dangerous error, since forced is the trusted output. How
  fragile is the split to under-sampling?
- **W-2 (Grok).** Attack P-5. Strict per-cell sign accuracy is ~71%; the
  headline ~89% uses a 2-cell "within resolution" tolerance. Make the strongest
  case that resolution-length scoring is a moving goalpost, not an honest metric.
- **W-3 (Gemini).** Test P-1. Is "forced" operationally more than "outside the
  null space," and "measure-dependent" more than "in the null space"? If the
  possibilistic framing reduces to classical resolution / null-space analysis,
  say so. If it adds content, name precisely what.
- **W-4 (Gemini).** Coherence of the whole: invariants → feasible set →
  decomposition → two demonstrations → exact formalization → C port. Where is it
  over-claimed? Identify the single weakest sentence in the note, verbatim.
- **W-5 (ChatGPT).** Steelman the probabilist. A Bayesian holds that a prior is
  information, not a defect — "the answer moves with the prior" is how inference
  is supposed to work. Make the strongest probabilist rebuttal to Figure 1 and
  to the claim that measure-dependence diagnoses artifact.
- **W-6 (ChatGPT).** Audit P-7. Is the prior-art positioning fair and the
  residual novelty real? Has "measure-dependence as the operational artifact
  diagnostic," run as a transferable methodology, been done — under any name —
  in the geophysical or inverse-problems literature?
- **W-7 (all three).** The §7 collaboration seam asks whether a 20-run Monte
  Carlo samples the feasible set well enough for the forced/measure-dependent
  split to be trustworthy on real data. Is that the right question, or is there
  a sharper one?

## Witness prompts

Copy-paste, one block per model.

### → Grok (edge-witness)

> You are the edge-witness in an adversarial review. The artifact is a public
> repository: https://github.com/IridiumSoftware/possibilistic-inversion —
> read `possibilistic_tomography_note.md` and, as needed, the code (`eikonal.py`,
> `synthetic_demo_eikonal.py`, `c/inversion.c`). Your job is the strongest
> available attack, not a balanced review.
>
> The method: an ensemble of data-fitting tomographic models is decomposed into
> *forced* structure (constant sign in every feasible model) and
> *measure-dependent* structure (sign varies across the ensemble). "Forced" is
> the trusted output.
>
> **W-1.** The nonlinear feasible-set sampler builds the ensemble from
> Levenberg–Marquardt inversions toward random reference models, then perturbs
> each with smooth fields kept within the noise band. Construct the scenario in
> which this systematically misses a region of the feasible set and so labels a
> genuinely measure-dependent feature as *forced* (a false-forced). How fragile
> is the split to under-sampling?
>
> **W-2.** Strict per-cell sign accuracy of the forced cores is ~71%; the
> headline ~89% applies a 2-cell "within the resolution length" tolerance. Make
> the strongest case that this scoring is a moving goalpost.
>
> Return, for each: the attack stated as sharply as possible, a concrete
> scenario or counterexample, and a severity — fatal / serious / manageable. End
> with the single thing that, if true, would most damage the method.

### → Gemini (synthesis)

> You are the synthesis witness in a structured review. The artifact is a public
> repository: https://github.com/IridiumSoftware/possibilistic-inversion — read
> `possibilistic_tomography_note.md`. Judge the whole, not the parts.
>
> The method decomposes an ensemble of data-fitting tomographic models into
> *forced* structure (constant sign in every feasible model) and
> *measure-dependent* structure (sign varies). It claims this is a distinction a
> Bayesian posterior covariance does not draw.
>
> **W-3.** Is "forced" operationally more than "outside the null space of the
> resolution operator," and "measure-dependent" more than "in the null space"?
> If the possibilistic framing reduces to classical resolution / null-space
> analysis, say so plainly. If it adds genuine content, name precisely what it
> adds.
>
> **W-4.** Assess coherence: invariants → feasible set → decomposition → linear
> and nonlinear demonstrations → exact-arithmetic formalization → C port. Where
> is the note over-claimed? Quote the single weakest sentence verbatim.
>
> Return: a verdict on whether the possibilistic framing is load-bearing or a
> relabeling, the weakest sentence quoted, and whether the parts cohere into one
> argument.

### → ChatGPT (domain-probabilist)

> You are the domain-probabilist witness in a structured review — stand in for a
> skeptical Bayesian geophysicist who inverts seismic data for a living. The
> artifact is a public repository:
> https://github.com/IridiumSoftware/possibilistic-inversion — read
> `possibilistic_tomography_note.md`, especially §2 and Figure 1.
>
> The method argues that because a Bayesian tomographic answer is conditional on
> the prior/regularization, one should additionally report which features are
> *forced* (hold across all data-fitting models, under every prior) versus
> *measure-dependent* (sign set by the prior).
>
> **W-5.** Steelman the probabilist. A prior is information, not a defect; "the
> answer moves with the prior" is how inference is supposed to work. Make the
> strongest probabilist rebuttal to Figure 1 and to the claim that
> measure-dependence diagnoses artifact.
>
> **W-6.** Is the prior-art positioning fair (Backus–Gilbert extremal inversion,
> Zhdanov Gramian constraints, Bodin–Sambridge transdimensional tomography), and
> is the residual novelty real? Has "measure-dependence as the operational
> diagnostic of artifact," run as a transferable methodology, been done — under
> any name — in the geophysical or inverse-problems literature?
>
> Return: the strongest probabilist rebuttal, stated so its author would accept
> it; and a prior-art verdict — novel / partly novel / already done.

## Returns

### Witness A — Grok

*Pending.*

### Witness B — Gemini

*Returned 2026-05-17.*

**Role:** Synthesis — Coherence of the whole; what is load-bearing versus over-claimed; whether the framing does real work or relabels known analysis.

#### 1. Coherence of the Whole

The two-layer epistemic framework proposed in `possibilistic_tomography_note.md` is structurally coherent if evaluated as a non-parametric, outer-loop diagnostic suite applied to multi-physics inversions (e.g., joint seismic-gravity systems).

The mathematical trajectory flows logically: continuous data misfits are bounded by an acceptable error tolerance $\epsilon$, defining an upper-bound data-admissible model space. The framework then sweeps across varying prior/regularization measures to partition model features into a binary classification: **Forced Features** (invariants that persist across all valid topologies regardless of the metric) and **Measure-Dependent Features** (geometrical structures whose sign or existence is conditional on the choice of the prior).

However, a core conceptual tension exists between the discrete, topological constraints derived from the triadic closure/graph-grammar framework and the continuous, non-linear operators of geophysical field theories. The coherence of the system relies entirely on the translation layer—specifically, how continuous fields are discretized into categorical "features" (e.g., via Voronoi cells or localized basis functions). If this discretization is unstable, the boundary between "forced" and "measure-dependent" features blurs, collapsing the operational utility of the binary classification.

#### 2. What is Load-Bearing vs. Over-Claimed

* **Load-Bearing:** The operational diagnostic core is robust. Isolating features that survive under arbitrary prior distributions provides a clean, transferable metric for structural validation. In joint inversions where cross-property relationships (e.g., mapping seismic velocity $V_p$ to density $\rho$) are highly uncertain, using a strict topological invariant (the "riverbed" syntax) to restrict cross-operator couplings does real work. It prevents the optimization from generating mathematically minimized but physically decoupled unstratified artifacts.
* **Over-Claimed:** The assertion that "measure-dependence diagnoses artifact" is an overstatement that misinterprets the role of priors in physical systems. In highly underdetermined, ill-posed inverse problems, certain real, physical boundaries (such as a sharp Moho discontinuity or a localized magmatic chamber) cannot be resolved by sparse data alone; they require the injection of sharp structural priors (e.g., Total Variation or Gramian constraints). Labeling a feature as an "artifact" simply because it requires a specific prior to manifest conflates a data-deficiency with a systemic error. Furthermore, framing the system as an autonomous "discovery engine" overpromises; it functions as a highly sophisticated topological linter and cross-channel consistency checker, not a generator of new physical realities.

#### 3. Framing Evaluation: Real Work vs. Relabeling Known Analysis

The framework sits at a crossroads where novel structural syntax meets deeply established inverse-problem methodologies.

* **Where it does real work:** The framework introduces value by establishing a formal, machine-verifiable grammar for multi-channel coupling. Rather than relying on empirical, case-by-case joint-inversion weights, it treats cross-operator consistency as an absolute boundary condition. This topological enforcement of structural conservation across distinct forward operators provides an objective, transferable pipeline for verifying complex, multi-layered ontologies.
* **Where it relabels known analysis:** The core mathematical objectives heavily re-map existing geophysical literature under new nomenclature:
* **Backus–Gilbert Extremal Inversion:** The search for "forced features" that hold across all data-fitting models is conceptually identical to the Backus–Gilbert approach, which calculates resolving kernels and strict bounds on linear functionals of the model space independent of a subjective prior.
* **Null-Space Exploration:** In classical linear and non-linear inverse theory, features that shift or vanish based on the choice of regularization ($L_1$ vs. $L_2$, smoothness penalties, or covariance scaling) are known as *null-space or near-null-space components*. Describing these as "measure-dependent features" is a direct relabeling of "null-space elements stabilized by the regularization choice."
* **Transdimensional Tomography (rj-MCMC):** The use of adaptive Voronoi parameterization to explore model space without fixing grid geometry is the standard domain of transdimensional MCMC (Bodin & Sambridge, 2009). In rj-MCMC, "forced features" naturally emerge as high-probability, high-consensus spatial structures across the posterior ensemble, while "measure-dependent features" show up as high-variance, low-consensus fluctuations driven by individual chain trajectories.

#### 4. Verdict

**Partly Novel.** The execution of a two-layer possibilistic filter to split model spaces into forced and measure-dependent categories is a highly effective, operationalized packaging of ensemble analysis. However, the foundational inverse-problem mechanics are deeply rooted in established null-space exploration and Backus–Gilbert style extremal bounding. The true residual novelty lies not in the underlying optimization math, but in the deployment of discrete topological closure conditions (the algebraic syntax) as a strict, automated arbiter for continuous multi-physics coupling.

> *[Metabolizer's note — recorded with the return, not part of it. This return imports framing from adjacent projects (triadic closure, graph-grammar, Voronoi parameterization, a "discovery engine," "riverbed syntax") that is **not** part of possibilistic-inversion — the method here is a plain per-grid-cell tomographic decomposition with no graph grammar and no triadic structure. That contamination fouls the W-4 coherence read (it assesses a system this is not) and the verbatim weakest-sentence ask went undelivered. The W-3 finding, however, lands clean and is not contaminated: forced ≈ Backus–Gilbert extremal bounds, measure-dependent ≈ regularization-stabilized near-null-space components, and rj-MCMC already yields high-consensus vs high-variance structure. That relabeling critique of P-1 and P-7 is carried to synthesis at full weight; the off-target framing is discounted.]*

### Witness C — ChatGPT

*Pending.*

## Synthesis

*Pending all three returns. Will record: which claims in the ledger survived
intact, which need downgrading and to what, which witness checks become Required
Witness Checks carried forward into the note, and any new open item the pass
surfaced.*
