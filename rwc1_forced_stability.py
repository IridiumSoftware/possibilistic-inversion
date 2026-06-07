"""
rwc1_forced_stability.py - Required Witness Check 1 (witness_pass.md).

Does the forced set converge as the feasible-set ensemble grows, and does the
false-forced rate fall and flatten?

The synthesis witness pass (Grok W-1, ChatGPT W-7) identified forced-set
stability under coverage expansion - not ensemble size per se - as the
load-bearing empirical question for the nonlinear method. A *false-forced*
cell (a forced label whose sign the truth contradicts) is the asymmetric,
dangerous error: a missed measure-dependent feature can be recovered later,
but a false-forced label manufactures unwarranted trust.

METHOD. Build one large feasible-set ensemble with the Eikonal sampler. Then,
for increasing subset size N, draw R random orderings of the ensemble, take
nested prefixes, decompose each prefix, and record three things, averaged over
the orderings:
  - forced-set size            : does it shrink and then plateau?
  - false-forced count         : strict (cell-exact) and within the resolution
                                 length; does it fall and flatten?
  - churn                      : cells that LEAVE the forced set when the
                                 prefix grows; does it fall toward zero?

VERDICT CRITERION. The decomposition is Monte-Carlo-stable at ensemble size N*
if, beyond N*, the forced-set size has plateaued, churn is small, and the
within-resolution false-forced count is flat.

HONEST SCOPE. This check tests *stability* of the forced set under expansion of
THIS sampler - it does not test *coverage completeness*. A feasible basin the
sampler never reaches is invisible here: every member would agree, the forced
set would look perfectly stable, and the false-forced would persist undetected.
Detecting that is RWC-2 (the disconnected-feasible-set stress test). RWC-1 and
RWC-2 are complementary by design; neither alone is sufficient.

Run:  uv run python rwc1_forced_stability.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import synthetic_demo as sd
import eikonal
from synthetic_demo import NZ, NX, N, EPS, ground_truth, dilate, \
    feasible_interval, classify, _ZZ
from synthetic_demo_eikonal import make_stations, feasible_set

ENSEMBLE_CACHE = "rwc1_ensemble.npy"   # local cache; gitignored, deterministic
N_BASE, N_PERT = 44, 8                 # target ~ 44 + 44*8 = 396 members
R_SHUFFLES = 16                        # random orderings averaged over
SEED = 20260517


def build_ensemble():
    """Generate (or load from cache) one large Eikonal feasible-set ensemble.
    The cache makes re-runs instant; the build is fully seeded, so a fresh
    build reproduces the cached ensemble exactly."""
    if os.path.exists(ENSEMBLE_CACHE):
        ens = np.load(ENSEMBLE_CACHE)
        print(f"Loaded cached ensemble: {ens.shape[0]} members.")
        return ens

    sd.RNG = np.random.default_rng(SEED)          # determinism: random_reference
    v_true = ground_truth()
    sources, receivers = make_stations()
    d_clean, _ = eikonal.forward(1.0 / v_true, sources, receivers)
    noise_sigma = 0.012 * float(d_clean.mean())
    d_obs = d_clean + np.random.default_rng(SEED).normal(
        0.0, noise_sigma, size=d_clean.shape)

    print(f"Building feasible ensemble (n_base={N_BASE}, n_pert={N_PERT}) — "
          "this is the cost, a few minutes of pure-Python FMM...")
    base, members = feasible_set(d_obs, sources, receivers, noise_sigma,
                                 n_base=N_BASE, n_pert=N_PERT)
    ens = np.array(members)
    np.save(ENSEMBLE_CACHE, ens)
    print(f"Feasible ensemble: {ens.shape[0]} members "
          f"({len(base)} base inversions + "
          f"{ens.shape[0] - len(base)} smooth perturbations).")
    return ens


def analyze(ens):
    """Convergence study: forced-set size, false-forced, and churn vs N."""
    v_true = ground_truth()
    bg = 5.5 + 2.0 * (_ZZ / NZ)
    a_true = v_true - bg
    M = ens.shape[0]

    sched = [n for n in (4, 6, 9, 13, 18, 25, 34, 46, 62, 84, 112, 145,
                          190, 250, 320)
             if n <= M]
    if not sched or sched[-1] != M:
        sched.append(M)

    f_size = {n: [] for n in sched}      # forced (high+low) cell count
    ff_strict = {n: [] for n in sched}   # false-forced, cell-exact
    ff_res = {n: [] for n in sched}      # false-forced, within resolution length
    churn = {n: [] for n in sched}       # forced cells lost since previous N

    rng = np.random.default_rng(SEED)
    for _ in range(R_SHUFFLES):
        order = rng.permutation(M)
        prev_forced = None
        for n in sched:
            a_min, a_max = feasible_interval(ens[order[:n]], bg)
            cls = classify(a_min, a_max)
            fhi, flo = cls == 2, cls == -2
            forced = fhi | flo
            f_size[n].append(int(forced.sum()))
            # sign-correct convention matches synthetic_demo_eikonal.py:
            # forced-high is wrong where a_true <= -EPS, forced-low where >= EPS.
            ff_strict[n].append(int(np.sum(fhi & (a_true <= -EPS))
                                    + np.sum(flo & (a_true >= EPS))))
            ff_res[n].append(int(np.sum(fhi & ~dilate(a_true > -EPS))
                                 + np.sum(flo & ~dilate(a_true < EPS))))
            if prev_forced is not None:
                churn[n].append(int(np.sum(prev_forced & ~forced)))
            prev_forced = forced

    return sched, f_size, ff_strict, ff_res, churn


def report_and_plot(sched, f_size, ff_strict, ff_res, churn):
    M = sched[-1]
    mean = lambda d, n: float(np.mean(d[n]))
    std = lambda d, n: float(np.std(d[n]))

    fs_m = np.array([mean(f_size, n) for n in sched])
    fs_s = np.array([std(f_size, n) for n in sched])
    ffs_m = np.array([mean(ff_strict, n) for n in sched])
    ffr_m = np.array([mean(ff_res, n) for n in sched])
    ffr_s = np.array([std(ff_res, n) for n in sched])
    ch_x = sched[1:]
    ch_m = np.array([mean(churn, n) for n in ch_x])

    print("\n  N    forced(hi+lo)   false-forced(strict / within-res)   churn")
    print("  " + "-" * 64)
    for i, n in enumerate(sched):
        ch = "" if i == 0 else f"{mean(churn, n):5.1f}"
        print(f"  {n:3d}    {fs_m[i]:7.1f}        "
              f"{ffs_m[i]:6.1f} / {ffr_m[i]:6.1f}              {ch:>5}")

    # verdict ---------------------------------------------------------------
    churn_end = ch_m[-1]
    ffr_end = ffr_m[-1]
    decr = -np.diff(fs_m)                        # per-step decrease, forced size
    decel = decr[-1] / max(decr[len(decr) // 2], 1e-9)
    clear = next((n for n, v in zip(sched, ffr_m) if v <= 1.0), None)
    ref = min(sched, key=lambda n: abs(n - 22))  # the ZTM runs=20 analogue
    ref_i = sched.index(ref)

    print("\n  Verdict")
    print(f"  false-forced, within resolution : {ffr_m[0]:.0f} cells (N={sched[0]})"
          f"  ->  {ffr_end:.1f} (N={M})")
    print(f"  false-forced, strict cell-exact : {ffs_m[0]:.0f}"
          f"  ->  {ffs_m[-1]:.1f}")
    if ffr_end <= 1.0:
        print("    => false-forced is ELIMINATED under coverage expansion —")
        print("       W-1 (the false-forced concern) is discharged for this")
        print("       synthetic problem with this sampler.")
    else:
        print("    => false-forced persists — W-1 stands at this coverage.")
    print(f"  forced-set size : {fs_m[0]:.0f}  ->  {fs_m[-1]:.0f} cells; "
          f"per-step decrease decelerating ({decr[0]:.0f} -> {decr[-1]:.1f}), "
          f"churn {churn_end:.1f} at N={M}")
    if decel < 0.35 and churn_end < 0.06 * fs_m[-1]:
        print(f"    => forced set is CONVERGING toward a stable core "
              f"(~{fs_m[-1]:.0f} cells).")
    else:
        print("    => forced set still contracting materially.")
    if clear:
        print(f"  Coverage threshold: within-resolution false-forced first "
              f"clears (<=1 cell) near N~{clear}.")
    print(f"  Reference: a ZTM-style runs=20 ensemble sits near N={ref} on this "
          f"curve — false-forced")
    print(f"    {ffr_m[ref_i]:.0f} within-res / {ffs_m[ref_i]:.0f} strict, "
          f"forced set {fs_m[ref_i] / fs_m[-1]:.1f}x its converged size. "
          f"20 runs is far")
    print("    short of coverage adequacy — the quantitative answer to §7.")
    print("  NOTE: this measures STABILITY under expansion of this sampler, not")
    print("  coverage COMPLETENESS — a feasible basin the sampler never reaches")
    print("  is invisible here. That is the target of RWC-2.")

    # figure ----------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.8))

    ax[0].fill_between(sched, fs_m - fs_s, fs_m + fs_s,
                       color="#9ecae1", alpha=0.7)
    ax[0].plot(sched, fs_m, "o-", color="#08519c", lw=2)
    ax[0].set_title("forced-set size vs ensemble size", fontsize=11)
    ax[0].set_ylabel("forced cells (high + low)")

    ax[1].fill_between(sched, ffr_m - ffr_s, ffr_m + ffr_s,
                       color="#f4c9b8", alpha=0.7)
    ax[1].plot(sched, ffr_m, "o-", color="#b2182b", lw=2,
               label="within resolution length")
    ax[1].plot(sched, ffs_m, "s--", color="#d98c6a", lw=1.4,
               label="strict (cell-exact)")
    ax[1].set_title("false-forced vs ensemble size", fontsize=11)
    ax[1].set_ylabel("false-forced cells")
    ax[1].legend(fontsize=8)

    ax[2].axhline(0, color="0.7", lw=0.8)
    ax[2].plot(ch_x, ch_m, "o-", color="#1a8a3a", lw=2)
    ax[2].set_title("churn — forced cells lost per growth step", fontsize=11)
    ax[2].set_ylabel("cells leaving the forced set")

    for a in ax:
        a.set_xscale("log")
        a.set_xlabel("ensemble size N (log)")
        a.grid(alpha=0.25)

    fig.suptitle("RWC-1 — forced-set stability under feasible-set coverage "
                 "expansion (Eikonal operator)", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("rwc1_forced_stability.png", dpi=130)
    plt.close(fig)
    print("\nFigure written: rwc1_forced_stability.png")

    # ---- sidecar JSON for coverage_diagnostics (ORSI #2 metadata) ---------
    # Consumed by coverage_diagnostics.coverage_certificate() to populate the
    # coverage curve in every standard report.
    import json
    from pathlib import Path
    sidecar = {
        "Ns": [int(n) for n in sched],
        "forced_sizes": [float(x) for x in fs_m],
        "false_forced_strict": [float(x) for x in ffs_m],
        "false_forced_res": [float(x) for x in ffr_m],
        "stabilization_N":
            int(clear) if clear is not None else None,
        "verdict_false_forced_eliminated": bool(ffr_end <= 1.0),
        "verdict_forced_set_converging":
            bool(decel < 0.35 and churn_end < 0.06 * fs_m[-1]),
    }
    Path("rwc1_coverage_curve.json").write_text(
        json.dumps(sidecar, indent=2))
    print("Sidecar written: rwc1_coverage_curve.json")


if __name__ == "__main__":
    print("RWC-1 — forced-set stability under coverage expansion")
    print("=" * 64)
    ensemble = build_ensemble()
    report_and_plot(*analyze(ensemble))
