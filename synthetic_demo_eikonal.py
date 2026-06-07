"""
synthetic_demo_eikonal.py - Possibilistic decomposition with the faithful
nonlinear (Eikonal) forward operator.

The Eikonal counterpart of synthetic_demo.py. Same synthetic model, same grid,
same possibilistic decomposition - but the forward model is the Fast Marching
Method Eikonal solver of eikonal.py (rays bend through the medium), not the
straight-ray approximation. Because travel time is then a NONLINEAR functional
of slowness, the one-shot regularized least-squares of synthetic_demo.py is
replaced by an iterative Gauss-Newton inversion that recomputes the ray paths
(the Frechet kernel) through the current model each step - the operator class
ZTM/TFM's DGN + FMM uses.

The feasible set is sampled in two stages. (1) Levenberg-Marquardt inversions
from many random reference models give base models that fit the data and
differ in their references. (2) Each base model is then perturbed by SMOOTH
random fields, each backtracked in amplitude to keep the RMS misfit data-
consistent. Stage 2 is essential and is what an LM ensemble alone does not
provide: a nonlinear iterative inversion deposits common inversion-dynamics
structure in weakly-constrained cells, so an LM-only ensemble shares a bias
and the decomposition certifies it as "forced".

The perturbations must be SMOOTH. The raw null space of a ray (line-integral)
operator is dominated by high-frequency checkerboard modes - a ray averages
along its path, so an oscillating perturbation cancels and is data-invisible -
but those are not admissible Earth models; perturbing freely in the raw null
space injects unphysical speckle and swamps the decomposition. A feasible
model must be both data-fitting and physically plausible (smooth). The
possibilistic decomposition (feasible interval -> forced-sign classes) is the
shared code imported from synthetic_demo.py.

Running this alongside synthetic_demo.py shows the decomposition is
forward-model-agnostic: the straight-ray and Eikonal operators are different
physics, but the possibilistic machinery on top is identical.

Status: EXPERIMENTAL - a methodology demonstration, not a library.

Run:  uv run python synthetic_demo_eikonal.py   (~1-2 min: pure-Python FMM)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

import eikonal
from synthetic_demo import (NZ, NX, N, VP_MIN, VP_MAX, EPS, DELTA, _ZZ, _XX,
                            _BLOBS, ground_truth, blob_mask, dilate,
                            feasible_interval, classify, random_reference)


def make_stations():
    """Sources on the left+top edges, receivers on the right+bottom edges -
    every source-receiver pair crosses the grid. Integer edge cells (the FMM
    works on the grid)."""
    src = ([(int(round(z)), 0) for z in np.linspace(3, NZ - 4, 10)]
           + [(0, int(round(x))) for x in np.linspace(3, NX - 4, 10)])
    rec = ([(int(round(z)), NX - 1) for z in np.linspace(3, NZ - 4, 12)]
           + [(NZ - 1, int(round(x))) for x in np.linspace(3, NX - 4, 12)])
    return src, rec


def invert_gn(d_obs, sources, receivers, s_ref, noise_sigma, max_iter=18):
    """Levenberg-Marquardt inversion of the nonlinear Eikonal forward model.
    Started from reference slowness s_ref, early-stopped when the RMS data
    misfit reaches the noise level.

    Adaptive Marquardt damping mu: decreased after a step that lowers the
    misfit (toward fast Gauss-Newton), increased after one that does not
    (toward safe gradient descent) - so it self-tunes between the
    over-damped regime (which plateaus above the noise level) and the
    undamped regime (which oscillates). The damped step
        ds = D^-1 G^T (G D^-1 G^T + mu I)^-1 r,   D = diag(G^T G)
    is solved in DATA space (a small n_ray x n_ray system). Early stopping
    is the regularization; the reference fixes the data-null directions.
    Returns (velocity (NZ,NX), final RMS misfit)."""
    slo, shi = 1.0 / VP_MAX, 1.0 / VP_MIN
    s = np.clip(s_ref.copy(), slo, shi)                  # flat-N slowness
    t = eikonal.traveltimes(s.reshape(NZ, NX), sources, receivers)
    rms = float(np.sqrt(np.mean((d_obs - t) ** 2)))
    mu = None
    for _ in range(max_iter):
        if rms <= noise_sigma:                           # early stop
            break
        t, G = eikonal.forward(s.reshape(NZ, NX), sources, receivers)
        r = d_obs - t
        rms = float(np.sqrt(np.mean(r ** 2)))
        if rms <= noise_sigma:
            break
        dinv = 1.0 / (np.sum(G * G, axis=0) + 1e-12)     # D^-1 (flat-N)
        GDGt = (G * dinv) @ G.T                          # G D^-1 G^T (n_ray^2)
        eye = np.eye(len(r))
        if mu is None:
            mu = 0.1 * float(np.mean(np.diag(GDGt)) + 1e-12)
        improved = False
        for _ in range(9):                               # damping search
            y = np.linalg.solve(GDGt + mu * eye, r)
            s_try = np.clip(s + dinv * (G.T @ y), slo, shi)
            t_try = eikonal.traveltimes(s_try.reshape(NZ, NX),
                                        sources, receivers)
            rms_try = float(np.sqrt(np.mean((d_obs - t_try) ** 2)))
            if rms_try < rms:                            # accept, less damping
                s, rms = s_try, rms_try
                mu = max(mu * 0.4, 1e-9)
                improved = True
                break
            mu *= 3.0                                    # reject, more damping
        if not improved:
            break
    return (1.0 / s).reshape(NZ, NX), rms


def smooth_field(rng, n_smooth):
    """A smooth random field on the grid: white noise put through n_smooth
    4-neighbour averaging passes, normalized to unit RMS. Smooth = physically
    plausible - see the module docstring on why perturbations must be smooth."""
    f = rng.normal(0.0, 1.0, (NZ, NX))
    for _ in range(int(n_smooth)):
        f = 0.25 * (np.roll(f, 1, 0) + np.roll(f, -1, 0)
                    + np.roll(f, 1, 1) + np.roll(f, -1, 1))
    return f / (f.std() + 1e-12)


def feasible_set(d_obs, sources, receivers, noise_sigma,
                 n_base=10, n_pert=7, pert_rms=0.55):
    """Sample the feasible set in two stages (see module docstring):
      (1) LM inversions from n_base random references -> base models;
      (2) n_pert SMOOTH random perturbations of each base model, each
          backtracked in amplitude to keep the RMS misfit data-consistent
          (<= 1.3 x noise). pert_rms is the unbacktracked perturbation
          amplitude in km/s; the smoothing length is randomized per
          perturbation so they probe a range of scales.
    Returns (base_models, all_feasible_members)."""
    base = []
    for k in range(n_base):
        v, rms = invert_gn(d_obs, sources, receivers,
                           random_reference(), noise_sigma)
        if rms <= 1.3 * noise_sigma:
            base.append(v)
        print(f"  base inversion {k + 1:2d}/{n_base}: "
              f"misfit {rms / noise_sigma:.2f} x noise"
              f"{'' if rms <= 1.3 * noise_sigma else '  (rejected)'}")
    if not base:
        return [], []
    rng = np.random.default_rng(7)
    members = list(base)
    for v in base:
        for _ in range(n_pert):
            shape = pert_rms * smooth_field(rng, rng.integers(18, 45))
            for f in (1.0, 0.6, 0.35, 0.2, 0.1, 0.05):    # backtrack amplitude
                v_try = np.clip(v + f * shape, VP_MIN, VP_MAX)
                t = eikonal.traveltimes(1.0 / v_try, sources, receivers)
                if np.sqrt(np.mean((d_obs - t) ** 2)) <= 1.3 * noise_sigma:
                    members.append(v_try)
                    break
    return base, members


def main():
    v_true = ground_truth()
    sources, receivers = make_stations()
    print(f"Geometry: {len(sources)} sources, {len(receivers)} receivers, "
          f"{len(sources) * len(receivers)} ray pairs, {N} cells. "
          f"Eikonal (FMM) forward operator.")

    d_clean, _ = eikonal.forward(1.0 / v_true, sources, receivers)
    noise_sigma = 0.012 * float(d_clean.mean())
    rng = np.random.default_rng(20260517)
    d_obs = d_clean + rng.normal(0.0, noise_sigma, size=d_clean.shape)

    print("Sampling the feasible set (LM inversions, then smooth "
          "perturbation):")
    base, feasible = feasible_set(d_obs, sources, receivers, noise_sigma)
    print(f"Feasible set: {len(base)} base inversions + "
          f"{len(feasible) - len(base)} smooth perturbations "
          f"= {len(feasible)} data-consistent models.")
    if len(feasible) < 12:
        raise SystemExit("Too few feasible members.")

    bg = 5.5 + 2.0 * (_ZZ / NZ)              # the known synthetic background
    a_min, a_max = feasible_interval(feasible, bg)
    cls = classify(a_min, a_max)
    a_true = v_true - bg
    forced_hi, forced_lo = cls == 2, cls == -2
    forced_quiet, meas_dep = cls == 0, cls == 1
    true_hi, true_lo = a_true > DELTA, a_true < -DELTA

    def pct(a, b):
        return 100.0 * int(np.sum(a)) / max(1, int(np.sum(b)))

    print("\n--- Possibilistic decomposition vs. ground truth "
          "(Eikonal operator) ---")
    print(f"forced-high core : {int(forced_hi.sum()):4d} cells | "
          f"sign-correct {pct(forced_hi & (a_true > -EPS), forced_hi):5.1f}% "
          f"| recall of true highs {pct(forced_hi & true_hi, true_hi):5.1f}%")
    print(f"forced-low  core : {int(forced_lo.sum()):4d} cells | "
          f"sign-correct {pct(forced_lo & (a_true < EPS), forced_lo):5.1f}% "
          f"| recall of true lows {pct(forced_lo & true_lo, true_lo):5.1f}%")
    se_res = int(np.sum(forced_hi & ~dilate(a_true > -EPS))
                 + np.sum(forced_lo & ~dilate(a_true < EPS)))
    print(f"sign errors      : {se_res} within resolution length "
          f"(forced to a sign the truth contradicts)")
    blobs = blob_mask()
    print(f"measure-dependent: {int(meas_dep.sum()):4d} cells | "
          f"covers {pct(meas_dep & blobs, blobs):5.1f}% of the fine-scale "
          f"blob cells")
    print(f"forced-quiet     : {int(forced_quiet.sum()):4d} cells")
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
    ax[1, 0].set_title("(d) forced-sign decomposition (Eikonal)")
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

    im = ax[1, 2].imshow(a_max - a_min, cmap="YlOrBr", origin="upper")
    for z0, x0, _ in _BLOBS:
        ax[1, 2].plot(x0, z0, "x", color="navy", ms=12, mew=3)
    ax[1, 2].set_title("(f) feasible interval width vs blob centres (x)")
    fig.colorbar(im, ax=ax[1, 2], label="km/s", fraction=0.046)

    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Possibilistic decomposition - Eikonal (FMM) forward operator "
                 "- synthetic demonstration", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig("possibilistic_decomposition_eikonal.png", dpi=130)
    print("\nFigure written: possibilistic_decomposition_eikonal.png")

    # ---- standard reporting layer (posdec) --------------------------------
    # ORSI #2/#4/#6: ship coverage certificate + standard report alongside
    # the demo figure. RWC-1 curve + RWC-2 false-forced rate folded in if
    # their sidecar JSON files are present.
    import posdec
    rwc1 = posdec.read_json_if_present("rwc1_coverage_curve.json")
    rwc2 = posdec.read_json_if_present("rwc2_certificate.json")
    cert = posdec.coverage_certificate(
        feasible, bg, eps=EPS,
        coverage_curve=rwc1,
        false_forced_rate=(rwc2 or {}).get("false_forced_rate"),
        label="synthetic_demo_eikonal (FMM, nonlinear)",
    )
    posdec.write_certificate(cert, "synthetic_demo_eikonal_certificate.json")
    posdec.plot_three_maps_and_width(
        a_min, a_max, eps=EPS,
        coverage_curve=rwc1,
        false_forced_rate=(rwc2 or {}).get("false_forced_rate"),
        out_path="synthetic_demo_eikonal_report.png",
        title="Possibilistic decomposition - Eikonal (FMM) "
              "(standard report)",
    )
    print(f"Certificate written: synthetic_demo_eikonal_certificate.json "
          f"(ensemble {cert['ensemble_size']}, RWC-1 stabilized at N="
          f"{cert['coverage']['rwc1_stabilized']}, RWC-2 status="
          f"{cert['coverage']['rwc2_status']})")
    print("Standard report:     synthetic_demo_eikonal_report.png")


if __name__ == "__main__":
    main()
