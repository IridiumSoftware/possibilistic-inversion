# Geophysical Invariants Reference

**Compiled 2026-05-17 — for the possibilistic-inversion uncertainty study**

Companion to `physical_invariants.md` (closure-v5 `BUSINESS/`). This file is
the explicit constraint set for the **possibilistic layer** of the inversion-
uncertainty study: the bounds that decide which model features are *forced*,
which are *forbidden*, and which are merely *measure-dependent*.

The stratification follows `inverse_born_methodology.md` §10:

- **Tier 1 — frame-independent / hard.** True no-go bounds; hold regardless of
  temperature, composition, depth, or any inversion choice. Used freely as the
  feasible-set boundary.
- **Tier 2 — frame-dependent / convergence targets.** Plausible values that
  depend on local conditions. Used as soft priors only. **Using any Tier-2
  value as a hard constraint is the "anchoring" anti-pattern** — it would
  silently contaminate the forced/forbidden classification with composition
  assumptions.
- **Tier 3 — observational end.** Known regional facts; the fixed "observed"
  end for an inverse-obstruction argument ("given the data *and* these, what
  is forced?").

Scope: P-wave velocity field `Vp`, density field `ρ`, and Moho depth, for the
crust + upper mantle of a continental subduction setting (the Peruvian Andes /
Nazca slab — the ZTM / TFM study region).

---

## 1. Tier 1 — Frame-independent / hard constraints

These define the possibilistic feasible set. They are deliberately **generous**
— wide enough to hold for any plausible silicate Earth material in this depth
range, so that "forbidden" means genuinely forbidden, not merely unusual.

| Invariant | Hard bound | Basis |
|---|---|---|
| Velocity positivity | `Vp > 0`, slowness `u = 1/Vp > 0` | physical impossibility otherwise |
| Density positivity | `ρ > 0` | physical impossibility otherwise |
| First-arrival minimality | predicted first-arrival time = minimum over all paths (Fermat / Eikonal) | a model whose forward times violate this is self-inconsistent; ZTM's FMM already enforces it for the forward solve |
| Moho single-valuedness | one Moho depth per `(lat, lon)` | structural property of the o2 / 2-D NNI parameterization itself |
| Velocity envelope | `2 ≤ Vp ≤ 9` km/s | no plausible crustal-or-upper-mantle silicate material falls outside this, at any geotherm/composition in range |
| Density envelope | `2.2 ≤ ρ ≤ 3.7` g/cm³ | same — spans weathered crust to dense peridotite, composition-agnostic |

**Verification flag.** The two envelope rows are stated at a defensible,
generous level; before they are used in a published result they should be
checked against an authoritative reference (PREM / AK135 mineral-physics
ranges) and against Zagid's own judgment. The *positivity*, *minimality*, and
*single-valuedness* rows are unconditional.

---

## 2. Tier 2 — Frame-dependent / convergence targets

Plausible values. **Soft priors only — never hard constraints.** They depend on
temperature, pressure, composition, and depth, which are exactly what the
inversion is solving for.

| Quantity | Typical value | Why frame-dependent |
|---|---|---|
| Upper-mantle `Vp` | ~8.0–8.3 km/s | varies with geotherm, composition, depth |
| Crustal `Vp` | ~6.0–6.5 km/s | composition, porosity, geotherm |
| Crust–mantle density contrast | ~0.3–0.5 g/cm³ | composition-dependent |
| In-situ upper-mantle density | ~3.3–3.4 g/cm³ (PREM near top) | pressure/temperature-dependent |
| STP reference densities | peridotite ≈ 3.3, basalt ≈ 2.9, granite ≈ 2.7 g/cm³ | **frame-pinned**: STP fixes the frame; the *in-situ* value is P/T-shifted. Use to identify the composition family, not as a bound. |
| Reference 1-D Earth models | IASP91, AK135, PREM depth profiles | regional structure *deviates* from these — that deviation is the tomographic signal; as hard constraints they would forbid the very anomalies sought |
| `Vp`–`ρ` petrophysics | Birch's law (linear `ρ`–`Vp`); Gardner's relation (`ρ ∝ Vp^0.25`, sedimentary) | empirical, composition / mean-atomic-weight dependent. ZTM's single scalar `dr` is its instance. |
| Airy isostatic compensation factor | ~4–6 (crustal-root : topography) | the *existence* of long-wavelength isostatic compensation is robust; the specific Airy model and factor are not |

**More-robust soft kernel.** One Tier-2 item has a fairly robust *sign*: for a
**thermal** anomaly, `Vp` and `ρ` co-vary in the same direction (hot → slower
*and* less dense). The *direction* of `Vp`–`ρ` coupling for thermal origin is
much more robust than its magnitude; compositional anomalies can decouple it.
Treat the sign as a strong soft prior, the magnitude as a convergence target.

---

## 3. Tier 3 — Observational end (for inverse-obstruction)

Known regional facts. Not constraints on the model fields directly; rather, the
fixed "observed" end against which the inverse question is posed.

| Fact | Note |
|---|---|
| The Nazca plate subducts beneath Peru | the slab is a high-`Vp`, high-`ρ` feature; the study region's defining structure |
| Andean crust is thickened | Moho ~50–75 km regionally |
| Seismicity defines slab geometry | hypocenters cluster in/near the slab (Wadati–Benioff zone) — independent geometric constraint |

---

## 4. How ZTM / TFM currently uses these — the tier-confusion

This is itself a finding the possibilistic study makes explicit. In the
published `TFM.cpp`:

- the DGN step **clamps** mantle-node velocity to `7.5 ≤ Vp ≤ 9.5` km/s and the
  Moho to a depth band;
- initialization draws from IASP91 and the Tables 3–4 ranges
  (`vmantle` 7.7–8.3, `vcrust` 5.9–6.2, `drhomc` 0.3–0.45, …).

Those are **Tier-2 values used as Tier-1 hard clamps.** That is the layer-
confusion the inverse-Born methodology is built to catch. The possibilistic
study deliberately does **not** inherit it: it uses the Tier-1 envelopes (§1)
as the feasible-set boundary and demotes the Tier-2 ranges (§2) to priors.
A model feature that survives only because of a Tier-2 clamp will then surface
as **measure-dependent** rather than **forced** — which is a result of the
study, not a defect of it.

---

## 5. Status

- Tier 1 positivity / minimality / single-valuedness — unconditional.
- Tier 1 velocity & density envelopes — defensible and deliberately generous;
  flagged for verification against an authoritative reference and Zagid.
- Tier 2 — convergence targets per `inverse_born_methodology.md` §10; must not
  be promoted to hard constraints.
- Tier 3 — regional geological context; sourced from standard knowledge of the
  Peru subduction zone, to be cited properly when the study is written up.
