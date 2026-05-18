"""
rwc2_disconnected_test.py - Required Witness Check 2 (witness_pass.md).

Grok's W-1: the standard sampler (Levenberg-Marquardt from random references +
smooth perturbations) can stamp a genuinely measure-dependent cell as *forced*
when the feasible set has basins the sampler does not reach. A false-forced is
the dangerous error: the method's trusted output is then a sampler artifact.

RWC-1 measured stability under expansion of the standard sampler; by
construction it cannot see a basin that sampler never reaches. RWC-2 closes
that gap by manufacturing the witnesses INDEPENDENTLY.

METHOD.
  1. Run the standard sampler. Record its forced set, and the roughness of its
     members - this fixes an admissibility bar (a witness must be no rougher
     than the roughest model the method itself accepted; the method admits
     only smooth feasible models - witness_pass.md / note s.6.5).
  2. The true model fits the data within noise - one feasible witness.
  3. Adversarial probes: for each target, run an LM inversion started from a
     reference with a large opposite-sign smooth bump, amplitude well beyond
     the standard random-reference distribution. Early-stopped LM keeps the
     reference's data-null component, so the adversarial model retains the
     opposite sign where the data permit it. Accept it as a witness only if it
     is BOTH data-feasible (misfit <= 1.3 x noise) AND admissible (roughness
     within the bar from step 1).
  4. A cell is PROVEN measure-dependent if, across the admissible witness pool,
     its anomaly straddles zero. This proof is independent of the standard
     sampler.
  5. Of the cells the standard sampler labelled forced, how many do the
     witnesses prove measure-dependent? Those are false-forced - the
     demonstrated form of W-1.

Run:  uv run python rwc2_disconnected_test.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

import synthetic_demo as sd
import eikonal
from synthetic_demo import NZ, NX, EPS, VP_MIN, VP_MAX, ground_truth, \
    feasible_interval, classify, _ZZ, _XX, _BLOBS
from synthetic_demo_eikonal import make_stations, invert_gn, feasible_set

SEED = 20260517
NOISE_FRAC = 0.012             # noise as a fraction of mean travel time
ADV_AMP = 2.0                  # adversarial bump amplitude (km/s)
ADV_SIG = 3.5                  # adversarial bump width (cells)


def roughness(v):
    """RMS of the discrete Laplacian - the smoothness / admissibility measure.
    The method admits only smooth feasible models; a witness must clear the
    same bar."""
    lap = (4.0 * v - np.roll(v, 1, 0) - np.roll(v, -1, 0)
           - np.roll(v, 1, 1) - np.roll(v, -1, 1))
    return float(np.sqrt(np.mean(lap ** 2)))


def rms_misfit(v, sources, receivers, d_obs):
    t = eikonal.traveltimes(1.0 / v, sources, receivers)
    return float(np.sqrt(np.mean((d_obs - t) ** 2)))


def main():
    sd.RNG = np.random.default_rng(SEED)        # determinism: random_reference
    rng = np.random.default_rng(SEED)

    v_true = ground_truth()
    bg = 5.5 + 2.0 * (_ZZ / NZ)
    a_true = v_true - bg
    sources, receivers = make_stations()
    d_clean, _ = eikonal.forward(1.0 / v_true, sources, receivers)
    noise_sigma = NOISE_FRAC * float(d_clean.mean())
    d_obs = d_clean + rng.normal(0.0, noise_sigma, size=d_clean.shape)
    tol = 1.3 * noise_sigma

    print("RWC-2 - disconnected-feasible-set / false-forced stress test")
    print("=" * 66)
    print(f"true model misfit "
          f"{rms_misfit(v_true, sources, receivers, d_obs) / noise_sigma:.2f}"
          f" x noise - feasible, and the first witness")

    # --- 1. the standard sampler, and the admissibility bar ----------------
    print("\nStandard sampler (random references + smooth perturbations, "
          "default n_base=10, n_pert=7):")
    base, members = feasible_set(d_obs, sources, receivers, noise_sigma)
    member_rough = np.array([roughness(m) for m in members])
    rough_bar = float(member_rough.max())
    a_min, a_max = feasible_interval(members, bg)
    cls = classify(a_min, a_max)
    forced = (cls == 2) | (cls == -2)
    std_md = cls == 1
    nF = int(forced.sum())
    print(f"  {len(members)} members; {nF} forced (high/low), "
          f"{int(std_md.sum())} measure-dependent.")
    print(f"  member roughness (Laplacian RMS): "
          f"{member_rough.min():.4f} - {rough_bar:.4f}  "
          f"-> admissibility bar = {rough_bar:.4f}")

    # --- 2-3. adversarial probes, filtered on misfit AND admissibility -----
    # Localized smooth Gaussian bumps on a grid of centres; each pushes its
    # neighbourhood against the truth's local sign. Localized + smooth so the
    # inverted model has a chance of staying admissible (a broad random field
    # at this amplitude inverts to a model far rougher than the bar).
    print("\nAdversarial probes (LM from localized opposite-sign references):")
    witnesses = [v_true]
    probes = []
    for z0 in (7, 16, 25, 33):
        for x0 in (7, 16, 25, 33):
            g = np.exp(-((_ZZ - z0) ** 2 + (_XX - x0) ** 2)
                       / (2 * ADV_SIG ** 2))
            local = float(np.sum(a_true * g) / np.sum(g))   # truth, locally
            sign = -1.0 if local >= 0.0 else 1.0            # oppose it
            probes.append((f"({z0:2d},{x0:2d})", sign * ADV_AMP * g))

    for name, bump in probes:
        v_ref = np.clip(bg + bump, VP_MIN, VP_MAX)
        v_adv, rms = invert_gn(d_obs, sources, receivers,
                               (1.0 / v_ref).ravel(), noise_sigma)
        rgh = roughness(v_adv)
        feasible = rms <= tol
        admissible = rgh <= rough_bar
        keep = feasible and admissible
        why = ("accepted" if keep else
               "REJECTED (misfit)" if not feasible else
               "REJECTED (too rough)")
        print(f"  {name:13s}: misfit {rms / noise_sigma:4.2f}x  "
              f"roughness {rgh:.4f}  {why}")
        if keep:
            witnesses.append(v_adv)

    # --- 4. cells PROVEN measure-dependent by the admissible witnesses -----
    W = np.array(witnesses) - bg
    witnessed_md = (W.min(axis=0) < -EPS) & (W.max(axis=0) > EPS)
    nW = int(witnessed_md.sum())
    print(f"\n{len(witnesses)} admissible feasible witnesses (true + "
          f"adversarial) prove {nW} cells")
    print("measure-dependent - their feasible interval straddles zero, "
          "independent of")
    print("the standard sampler.")

    # --- 5. the test: false-forced -----------------------------------------
    false_forced = witnessed_md & forced
    caught = witnessed_md & std_md
    quiet = witnessed_md & (cls == 0)
    ff = int(false_forced.sum())
    honest_forced = nF - ff

    print("\n  RWC-2 result")
    print(f"  the standard sampler labelled              {nF:4d} cells forced")
    print(f"  of those, proven measure-dependent         {ff:4d}  "
          f"<- FALSE-FORCED")
    print(f"  forced labels that survive the witnesses   {honest_forced:4d}")
    print(f"  (witnessed-MD cells the sampler also got right: {int(caught.sum())}"
          f"; forced-quiet misses: {int(quiet.sum())})")
    if ff:
        print(f"\n  => W-1 DEMONSTRATED. {100.0 * ff / max(1, nF):.0f}% of the "
              f"standard sampler's forced cells")
        print("     are false-forced — an admissible, equally-smooth, "
              "data-feasible model")
        print("     of the opposite sign exists. This is a LOWER BOUND: more")
        print("     adversarial witnesses would reveal more false-forced.")
        print(f"     At its default operating point at most ~{honest_forced} "
              f"of the {nF} forced")
        print("     labels survive — the rest is coverage artifact. The honest")
        print("     forced set shrinks further with coverage, toward RWC-1's")
        print("     ~64-cell converged core.")
    else:
        print("\n  => No false-forced among the witnessed cells: for every "
              "basin the")
        print("     adversarial probes reached, the sampler's coverage was "
              "adequate.")

    # --- figure ------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    am = float(np.abs(a_true).max())
    ax[0].imshow(a_true, cmap="RdBu_r", vmin=-am, vmax=am, origin="upper")
    for z0, x0, _ in _BLOBS:
        ax[0].plot(x0, z0, "x", color="k", ms=11, mew=2.5)
    ax[0].set_title("(a) true anomaly\n(x = adversarial blob targets)",
                    fontsize=10)

    cmap = ListedColormap(["#2166ac", "#d9d9d9", "#fdb338", "#b2182b"])
    ax[1].imshow(cls, cmap=cmap, norm=BoundaryNorm([-2.5, -1, 0.5, 1.5, 2.5],
                 cmap.N), origin="upper")
    ax[1].set_title(f"(b) standard sampler decomposition\n{nF} cells labelled "
                    "forced", fontsize=10)

    test = np.zeros((NZ, NX), int)            # 0 none 1 caught 2 quiet 3 false
    test[caught] = 1
    test[quiet] = 2
    test[false_forced] = 3
    tmap = ListedColormap(["#f2f2f2", "#3a9d4a", "#e8a33d", "#c1131b"])
    ax[2].imshow(test, cmap=tmap, norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5],
                 tmap.N), origin="upper")
    ax[2].set_title(f"(c) adversarial witnesses vs the sampler\n"
                    f"{ff} false-forced cells (red)", fontsize=10)

    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in tmap.colors]
    fig.legend(handles, ["not witnessed", "caught (measure-dependent)",
                         "forced-quiet miss", "FALSE-FORCED"],
               loc="lower center", ncol=4, fontsize=9)
    fig.suptitle("RWC-2 - false-forced stress test: admissible adversarial "
                 "witnesses vs the standard sampler", fontsize=12,
                 weight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig("rwc2_disconnected_test.png", dpi=130)
    plt.close(fig)
    print("\nFigure written: rwc2_disconnected_test.png")


if __name__ == "__main__":
    main()
