"""
synthetic_demo.py - Possibilistic decomposition of a tomographic inversion.

Demonstrates the possibilistic-uncertainty methodology (the two-layer
discipline of closure-v5's inverse_born_methodology.md) applied to
seismic-style travel-time tomography.

The possibilistic object is the per-cell FEASIBLE INTERVAL [a_min, a_max] -
the range of anomaly values consistent with the data across an ensemble of
models that each fit the data to the noise level. From the interval:

    forced-high        a_min > +EPS   every feasible model: a positive anomaly
    forced-low         a_max < -EPS   every feasible model: a negative anomaly
    forced-quiet       interval within +/-EPS : every model says ~nothing here
    measure-dependent  interval straddles : the SIGN itself is not data-forced

"Forced" is a statement about the data, threshold-free in sign: a cell is
forced-high iff no data-consistent model can make its anomaly non-positive.
A secondary, magnitude-level readout (a_min > DELTA) flags where the anomaly
is forced not just positive but large.

Pipeline:
  * a KNOWN synthetic velocity model (ground truth);
  * 2-D straight-ray first-arrival travel-time data + noise, dense four-edge
    ray coverage;
  * the FEASIBLE SET sampled by inverting toward many random reference models,
    each bisected in lambda to the minimally-damped model whose RMS data
    misfit equals the noise level;
  * the feasible interval and the forced-sign decomposition;
  * VALIDATION against the ground truth.

THREE METHODOLOGICAL POINTS, learned by iterating this script:

 1. WHY MANY RANDOM REFERENCES. The feasible interval is min/max over the
    feasible set. An ensemble that varies only damping STRENGTH shares a bias
    - all members pulled toward one reference - so the interval is artificially
    narrow and certifies that shared bias as "forced." Many random references
    sample the data-null directions, so "forced" means resolved-by-the-data.

 2. WHY THE MINIMALLY-DAMPED POINT. The feasible set is sampled where the data
    is actually fit (RMS misfit = noise level). Over-damped members have had
    data-resolved structure regularized away.

 3. WHY SIGN, NOT A HARD MAGNITUDE THRESHOLD. A hard 3-way label at a single
    anomaly threshold is brittle: a cell that is +0.45 in 38 models and +0.15
    in 2 gets dropped to "measure-dependent." The feasible interval is the
    honest object; the forced-SIGN classification reads off whether the
    existence of an anomaly is data-forced, independent of any threshold.

All three are the methodology's anti-anchoring discipline: do not let the
regularization choice, the operator's blur, or an arbitrary threshold
masquerade as a data-forced result.

Conventions in force:
  * Velocity v in km/s; slowness s = 1/v. Grid row-major [NZ, NX]; z downward;
    cell (z, x) -> flat index z*NX + x. Distances in km.
  * Forward model: straight-ray linear tomography. The decomposition is
    forward-model-agnostic; an Eikonal solver (ZTM's FMM) changes only G.
  * Regularization: identity (Tikhonov) damping toward a reference slowness.
  * "anomaly" = v minus a SINGLE fixed depth-gradient background (the gradient
    fit of the ensemble mean).
  * G^T G is eigendecomposed once; every (lambda, reference) inversion is then
    a vector operation.

Status: EXPERIMENTAL throughout - a methodology demonstration, not a library.

Run:  uv run python synthetic_demo.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

RNG = np.random.default_rng(20260517)

# --- Tier-1 hard bounds (geophysical_invariants.md sec.1) -------------------
VP_MIN, VP_MAX = 2.0, 9.0          # km/s - frame-independent velocity envelope

# --- Grid / thresholds -----------------------------------------------------
NZ, NX = 40, 40
N = NZ * NX
EPS = 0.04                          # km/s - sign deadband (numerical/noise floor)
DELTA = 0.20                        # km/s - "forced-large" magnitude readout
BLOB_SIGMA = 1.7                    # cells - fine-scale (sub-resolution) blobs

_ZZ = (np.arange(NZ)[:, None] + 0.5) * np.ones((NZ, NX))
_XX = (np.arange(NX)[None, :] + 0.5) * np.ones((NZ, NX))

# fine-scale blob centres + amplitudes (the measure-dependent target)
_BLOBS = [(8, 31, 0.36), (33, 9, 0.34), (15, 34, -0.32)]


def depth_gradient_background(v):
    """Best-fit background a + b*z; return the fitted field."""
    A = np.column_stack([np.ones(N), _ZZ.ravel()])
    coef, *_ = np.linalg.lstsq(A, v.ravel(), rcond=None)
    return (A @ coef).reshape(NZ, NX)


def ground_truth():
    """Depth gradient + one large tilted high-V slab (forced-high target) +
    one broad low-V zone (forced-low target) + three small Gaussian blobs
    (sub-resolution measure-dependent target)."""
    v = 5.5 + 2.0 * (_ZZ / NZ)                              # background
    v = v + 0.9 * (np.abs(_XX - (0.55 * _ZZ + 6.0)) < 4.0)  # high-V slab
    v = v - 0.85 * np.exp(-((_ZZ - 27) ** 2 + (_XX - 31) ** 2)
                          / (2 * 5.5 ** 2))                 # broad low-V zone
    for z0, x0, amp in _BLOBS:                              # fine-scale blobs
        v = v + amp * np.exp(-((_ZZ - z0) ** 2 + (_XX - x0) ** 2)
                             / (2 * BLOB_SIGMA ** 2))
    return v


def blob_mask():
    """Cells within ~1.5 sigma of a fine-scale blob centre."""
    m = np.zeros((NZ, NX), bool)
    for z0, x0, _ in _BLOBS:
        m |= ((_ZZ - z0) ** 2 + (_XX - x0) ** 2) < (1.5 * BLOB_SIGMA) ** 2
    return m


def dilate(mask):
    """8-neighbour dilation by one cell - the resolution-length tolerance.
    A forced-sign cell is a genuine error only if the truth contradicts it
    across the whole neighbourhood; a one-cell shift is the operator's blur."""
    m = mask.copy()
    m[1:, :] |= mask[:-1, :];    m[:-1, :] |= mask[1:, :]
    m[:, 1:] |= mask[:, :-1];    m[:, :-1] |= mask[:, 1:]
    m[1:, 1:] |= mask[:-1, :-1]; m[:-1, :-1] |= mask[1:, 1:]
    m[1:, :-1] |= mask[:-1, 1:]; m[:-1, 1:] |= mask[1:, :-1]
    return m


def build_G():
    """Straight-ray tomography operator with dense four-edge coverage:
    stations on all four edges, rays between every station pair not sharing
    an edge. G[i, c] = length of ray i in cell c, by fine sampling."""
    m = 13
    L = [(z, 0.0) for z in np.linspace(2, NZ - 2, m)]
    R = [(z, float(NX)) for z in np.linspace(2, NZ - 2, m)]
    T = [(0.0, x) for x in np.linspace(2, NX - 2, m)]
    B = [(float(NZ), x) for x in np.linspace(2, NX - 2, m)]
    edges = [L, R, T, B]
    rays = [(p, q) for a in range(4) for b in range(a + 1, 4)
            for p in edges[a] for q in edges[b]]
    G = np.zeros((len(rays), N))
    for i, ((z0, x0), (z1, x1)) in enumerate(rays):
        length = float(np.hypot(z1 - z0, x1 - x0))
        nstep = max(2, int(length / 0.2))
        t = (np.arange(nstep) + 0.5) / nstep
        iz = np.clip((z0 + t * (z1 - z0)).astype(int), 0, NZ - 1)
        ix = np.clip((x0 + t * (x1 - x0)).astype(int), 0, NX - 1)
        np.add.at(G[i], iz * NX + ix, length / nstep)
    return G


def random_reference():
    """A random, plausible, smooth slowness reference - a random depth
    gradient plus a smoothed random bump field, clipped to the Tier-1
    envelope. Inverting toward many of these samples the feasible set's
    data-null directions."""
    v = 4.8 + RNG.uniform(2.2, 3.6) * (_ZZ / NZ)
    bump = RNG.normal(0.0, 1.0, (NZ, NX))
    for _ in range(60):
        bump = 0.25 * (np.roll(bump, 1, 0) + np.roll(bump, -1, 0)
                       + np.roll(bump, 1, 1) + np.roll(bump, -1, 1))
    bump /= (np.abs(bump).max() + 1e-12)
    v = np.clip(v + 0.7 * bump, VP_MIN, VP_MAX)
    return (1.0 / v).ravel()


def run_ensemble(G, d, noise_sigma, n_ref=40):
    """Sample the feasible set. G^T G is eigendecomposed once; for each random
    reference, the damping lambda is bisected so the model's RMS data misfit
    equals the noise level - the minimally-damped data-consistent model for
    that reference."""
    GtG, Gtd = G.T @ G, G.T @ d
    evals, Q = np.linalg.eigh(GtG)                  # GtG = Q diag(evals) Q^T
    evals = np.maximum(evals, 0.0)
    Qt_Gtd = Q.T @ Gtd
    slo, shi = 1.0 / VP_MAX, 1.0 / VP_MIN

    def model_at(lam2, Qt_sref):
        s = Q @ ((Qt_Gtd + lam2 * Qt_sref) / (evals + lam2))
        return np.clip(s, slo, shi)

    def misfit(s):
        return float(np.sqrt(np.mean((G @ s - d) ** 2)))

    members = []
    for _ in range(n_ref):
        Qt_sref = Q.T @ random_reference()
        lo2, hi2 = 1e-8, 1e8                        # bisect lambda^2
        for _ in range(48):
            mid2 = np.sqrt(lo2 * hi2)
            if misfit(model_at(mid2, Qt_sref)) > noise_sigma:
                hi2 = mid2
            else:
                lo2 = mid2
        s = model_at(np.sqrt(lo2 * hi2), Qt_sref)
        members.append((1.0 / s).reshape(NZ, NX))
    return members


def feasible_interval(model_vs, bg):
    """The possibilistic object. Per cell, the feasible interval [a_min,a_max]
    of the anomaly (v - bg) over the ensemble - the set of anomaly values
    consistent with the data."""
    anom = np.array([v - bg for v in model_vs])
    return anom.min(axis=0), anom.max(axis=0)


def classify(a_min, a_max):
    """Forced-sign decomposition from the feasible interval:
        +2 forced-high       a_min > +EPS
        -2 forced-low        a_max < -EPS
         0 forced-quiet      interval within +/-EPS
        +1 measure-dependent interval straddles - sign not data-forced."""
    out = np.full((NZ, NX), 1, int)
    out[a_min > EPS] = 2
    out[a_max < -EPS] = -2
    out[(a_min >= -EPS) & (a_max <= EPS)] = 0
    return out


def main():
    v_true = ground_truth()
    G = build_G()
    d_clean = G @ (1.0 / v_true.ravel())
    noise_sigma = 0.012 * float(d_clean.mean())
    d_obs = d_clean + RNG.normal(0.0, noise_sigma, size=d_clean.shape)
    print(f"Geometry: {G.shape[0]} rays, {N} cells.")

    feasible = run_ensemble(G, d_obs, noise_sigma)
    print(f"Feasible set: {len(feasible)} models, one per random reference, "
          f"each bisected to RMS misfit = noise level.")

    bg = depth_gradient_background(np.mean(feasible, axis=0))
    a_min, a_max = feasible_interval(feasible, bg)
    cls = classify(a_min, a_max)
    a_true = v_true - bg

    forced_hi, forced_lo = cls == 2, cls == -2
    forced_quiet, meas_dep = cls == 0, cls == 1
    true_hi, true_lo = a_true > DELTA, a_true < -DELTA      # genuine features

    def pct(a, b):
        return 100.0 * int(np.sum(a)) / max(1, int(np.sum(b)))

    print("\n--- Possibilistic decomposition vs. ground truth ---")
    print(f"forced-high core : {int(forced_hi.sum()):4d} cells | "
          f"sign-correct {pct(forced_hi & (a_true > -EPS), forced_hi):5.1f}% "
          f"(not a true negative) | "
          f"forced-large (a_min>{DELTA}) {pct(a_min > DELTA, forced_hi):4.1f}% "
          f"| recall of true highs {pct(forced_hi & true_hi, true_hi):5.1f}%")
    print(f"forced-low  core : {int(forced_lo.sum()):4d} cells | "
          f"sign-correct {pct(forced_lo & (a_true < EPS), forced_lo):5.1f}% | "
          f"forced-large (a_max<-{DELTA}) {pct(a_max < -DELTA, forced_lo):4.1f}% "
          f"| recall of true lows {pct(forced_lo & true_lo, true_lo):5.1f}%")
    se_exact = int(np.sum(forced_hi & (a_true < -EPS))
                   + np.sum(forced_lo & (a_true > EPS)))
    se_res = int(np.sum(forced_hi & ~dilate(a_true > -EPS))
                 + np.sum(forced_lo & ~dilate(a_true < EPS)))
    print(f"sign errors      : {se_res} within resolution length, "
          f"{se_exact} cell-exact (forced to a sign the truth contradicts)")
    blobs = blob_mask()
    print(f"measure-dependent: {int(meas_dep.sum()):4d} cells | "
          f"covers {pct(meas_dep & blobs, blobs):5.1f}% of the fine-scale "
          f"blob cells")
    print(f"forced-quiet     : {int(forced_quiet.sum()):4d} cells "
          f"(every model: ~no anomaly)")
    slab_core = np.abs(_XX - (0.55 * _ZZ + 6.0)) < 2.0
    print(f"slab interior in forced-high: {pct(forced_hi & slab_core, slab_core):5.1f}%")
    fc = forced_hi | forced_lo | forced_quiet
    print(f"forced total     : {int(fc.sum())} cells "
          f"({100.0 * fc.sum() / N:.0f}% of grid is sign-or-quiet certain)")

    # ---- figure -----------------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    am = float(max(np.abs(a_true).max(), np.abs(a_min).max(),
                   np.abs(a_max).max()))
    div = dict(cmap="RdBu_r", vmin=-am, vmax=am, origin="upper")

    im = ax[0, 0].imshow(v_true, cmap="turbo", origin="upper")
    ax[0, 0].set_title("(a) ground truth velocity")
    fig.colorbar(im, ax=ax[0, 0], label="Vp (km/s)", fraction=0.046)

    im = ax[0, 1].imshow(a_min, **div)
    ax[0, 1].set_title("(b) feasible interval - lower bound a_min")
    fig.colorbar(im, ax=ax[0, 1], label="anomaly (km/s)", fraction=0.046)

    im = ax[0, 2].imshow(a_max, **div)
    ax[0, 2].set_title("(c) feasible interval - upper bound a_max")
    fig.colorbar(im, ax=ax[0, 2], label="anomaly (km/s)", fraction=0.046)

    cmap = ListedColormap(["#2166ac", "#d9d9d9", "#fdb338", "#b2182b"])
    norm = BoundaryNorm([-2.5, -1, 0.5, 1.5, 2.5], cmap.N)
    ax[1, 0].imshow(cls, cmap=cmap, norm=norm, origin="upper")
    ax[1, 0].set_title("(d) forced-sign decomposition")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cmap.colors]
    ax[1, 0].legend(handles, ["forced-low", "forced-quiet",
                              "measure-dependent", "forced-high"],
                    loc="upper center", ncol=2, fontsize=8,
                    bbox_to_anchor=(0.5, -0.04))

    ax[1, 1].imshow(a_true, **div)
    ax[1, 1].contour(forced_hi.astype(float), levels=[0.5], colors="k",
                     linewidths=1.6)
    ax[1, 1].contour(forced_lo.astype(float), levels=[0.5], colors="lime",
                     linewidths=1.6)
    ax[1, 1].set_title("(e) forced cores (black=high, green=low) on true anomaly")

    width = a_max - a_min
    im = ax[1, 2].imshow(width, cmap="YlOrBr", origin="upper")
    for z0, x0, _ in _BLOBS:
        ax[1, 2].plot(x0, z0, "x", color="navy", ms=12, mew=3)
    ax[1, 2].set_title("(f) feasible interval width vs blob centres (x)")
    fig.colorbar(im, ax=ax[1, 2], label="km/s", fraction=0.046)

    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Possibilistic decomposition of a tomographic inversion "
                 "ensemble - synthetic demonstration", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig("possibilistic_decomposition.png", dpi=130)
    print("\nFigure written: possibilistic_decomposition.png")

    # ---- standard reporting layer (coverage_diagnostics) ------------------
    # ORSI #2/#4/#6: every decomposition ships with a coverage certificate
    # and the standard 3-mask + interval-width + coverage-curve figure.
    import coverage_diagnostics as cd
    rwc1 = cd.read_json_if_present("rwc1_coverage_curve.json")
    rwc2 = cd.read_json_if_present("rwc2_certificate.json")
    cert = cd.coverage_certificate(
        feasible, bg, eps=EPS,
        coverage_curve=rwc1,
        false_forced_rate=(rwc2 or {}).get("false_forced_rate"),
        label="synthetic_demo (linear, straight-ray)",
    )
    cd.write_certificate(cert, "synthetic_demo_certificate.json")
    cd.plot_three_maps_and_width(
        a_min, a_max, eps=EPS,
        coverage_curve=rwc1,
        false_forced_rate=(rwc2 or {}).get("false_forced_rate"),
        out_path="synthetic_demo_report.png",
        title="Possibilistic decomposition - linear straight-ray "
              "(standard report)",
    )
    print(f"Certificate written: synthetic_demo_certificate.json "
          f"(ensemble {cert['ensemble_size']}, RWC-1 stabilized at N="
          f"{cert['coverage']['rwc1_stabilized']}, RWC-2 status="
          f"{cert['coverage']['rwc2_status']})")
    print("Standard report:     synthetic_demo_report.png")


if __name__ == "__main__":
    main()
