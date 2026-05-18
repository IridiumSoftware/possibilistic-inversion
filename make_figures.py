"""
make_figures.py - the explanatory figures for the write-up.

  fig_possibilism.png  - possibilistic vs probabilistic in model space: a
                         measure on the feasible set vs the shape of it
                         (the conceptual leap, for a probabilist reader);
  fig_schematic.png    - probabilistic vs possibilistic reading of an ensemble
                         (the central idea, as a 1-D cartoon);
  fig_ray_bending.png  - straight rays vs first-arrival Eikonal rays through
                         the synthetic model (why the forward operator matters).

The two demonstration figures (possibilistic_decomposition*.png) are produced
by synthetic_demo.py and synthetic_demo_eikonal.py.

Status: EXPERIMENTAL - figure generation for the methodology note.

Run:  uv run python make_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import eikonal
from synthetic_demo import ground_truth, NZ, NX, EPS


def fig_schematic():
    """A 1-D cartoon. An ensemble of models all fit the data; they agree
    where the data constrains them and spread where it does not. The
    probabilistic reading collapses that to one model + one error band; the
    possibilistic reading keeps the feasible interval and classifies it."""
    x = np.linspace(0.0, 1.0, 240)

    def g(c, w):
        return np.exp(-((x - c) ** 2) / (2.0 * w ** 2))

    # data-forced structure: a strong positive feature and a broad negative one
    forced = 0.85 * g(0.27, 0.060) - 0.62 * g(0.62, 0.085)
    truth = forced + 0.32 * g(0.86, 0.028)            # + a small, unresolved bump

    # spread is small where the data constrains, large where it does not
    constrained = np.clip(g(0.27, 0.11) + g(0.62, 0.14), 0.0, 1.0)
    spread = 0.55 * (1.0 - constrained)

    rng = np.random.default_rng(3)
    ens = []
    for _ in range(45):
        n = rng.normal(size=x.size)
        for _ in range(34):                           # smooth the perturbation
            n = np.convolve(n, [0.25, 0.5, 0.25], "same")
        ens.append(forced + spread * n / (n.std() + 1e-9))
    ens = np.array(ens)
    a_min, a_max = ens.min(0), ens.max(0)
    mean, std = ens.mean(0), ens.std(0)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)

    ax[0].axhline(0, color="0.6", lw=0.8)
    ax[0].fill_between(x, mean - std, mean + std, color="#9ecae1", alpha=0.8,
                       label="+/- 1 std")
    ax[0].plot(x, mean, color="#08519c", lw=2, label="ensemble mean")
    ax[0].set_title("Probabilistic reading\none model, one uncertainty band")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_ylabel("anomaly")

    fh = a_min > EPS
    fl = a_max < -EPS
    md = ~fh & ~fl
    ax[1].axhline(0, color="0.6", lw=0.8)
    ax[1].fill_between(x, a_min, a_max, where=fh, color="#b2182b", alpha=0.8,
                       label="forced-high")
    ax[1].fill_between(x, a_min, a_max, where=fl, color="#2166ac", alpha=0.8,
                       label="forced-low")
    ax[1].fill_between(x, a_min, a_max, where=md, color="#fdb338", alpha=0.85,
                       label="measure-dependent")
    ax[1].plot(x, truth, "k--", lw=1.6, label="true anomaly")
    ax[1].set_title("Possibilistic reading\nthe feasible interval, classified")
    ax[1].legend(loc="upper right", fontsize=8, ncol=2)

    for a in ax:
        a.set_xticks([]); a.set_xlabel("position")
    fig.suptitle("The same feasible ensemble, read two ways", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig("fig_schematic.png", dpi=130)
    plt.close(fig)
    print("fig_schematic.png")


def fig_possibilism():
    """The conceptual leap, drawn in model space - for a reader fluent in
    probabilism who is not yet sure possibilism adds anything.

    The data defines a feasible set F (every model in it fits within noise).
    Probabilistic inversion puts a *measure* on F - and its reported answer
    moves with that measure. The possibilistic reading instead looks at the
    *shape* of F: a feature is forced if F lies entirely on one side of zero
    (true under every measure), measure-dependent if F straddles zero (the
    measure, not the data, picks the sign). Each cell of a tomogram is one
    such feature; its feasible interval is F projected onto that cell's axis.
    """
    from matplotlib.patches import Ellipse

    # the feasible set F: a smooth lumpy closed blob in (A, B) model space
    th = np.linspace(0.0, 2.0 * np.pi, 400)
    r = (0.30 + 0.045 * np.cos(3 * th + 0.6) + 0.030 * np.cos(5 * th - 1.1)
              + 0.022 * np.cos(2 * th + 2.0))
    cx, cy = 0.60, 0.05
    fx = cx + r * np.cos(th) * 1.05
    fy = cy + r * np.sin(th) * 1.40

    eps = 0.06
    RED, AMB, BLU = "#b2182b", "#e8a33d", "#2166ac"

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 5.9))

    def base(a):
        a.axhline(0, color="0.5", lw=1.0, zorder=1)
        a.axvline(eps, color="0.8", lw=1.0, ls=":", zorder=1)
        a.text(eps + 0.012, 0.83, r"$\epsilon$", fontsize=9, color="0.55")
        a.set_xlim(-0.28, 1.08)
        a.set_ylim(-1.14, 0.92)
        a.set_xticks([]); a.set_yticks([])
        a.set_xlabel("amplitude of feature A", fontsize=9.5)
        for s in a.spines.values():
            s.set_visible(False)

    # ---- panel 1: the data gives a set ----------------------------------
    base(ax[0])
    ax[0].fill(fx, fy, color="#cfe3f0", ec="#3b7aa8", lw=1.7, zorder=2)
    ax[0].text(cx, cy, "F", ha="center", va="center", fontsize=18,
               style="italic", color="#1f4e6b", zorder=3)
    ax[0].set_ylabel("amplitude of feature B", fontsize=9.5)
    ax[0].set_title("1.  The data gives a SET", fontsize=11.5, weight="bold")
    ax[0].text(0.46, -0.82,
               "every model in F fits the data within\n"
               "noise — the data alone gives a region\n"
               "of model space, not a single point",
               ha="center", va="top", fontsize=9, color="0.25")

    # ---- panel 2: probabilistic - a measure on F ------------------------
    base(ax[1])
    ax[1].fill(fx, fy, color="#eceef0", ec="#9aa7ad", lw=1.5, zorder=2)
    p1, p2 = (0.65, 0.27), (0.56, -0.23)
    ax[1].add_patch(Ellipse(p1, 0.24, 0.17, angle=18,
                            color=RED, alpha=0.28, zorder=3))
    ax[1].add_patch(Ellipse(p2, 0.27, 0.16, angle=-12,
                            color=BLU, alpha=0.28, zorder=3))
    ax[1].plot(*p1, "o", color=RED, ms=8, zorder=4)
    ax[1].plot(*p2, "o", color=BLU, ms=8, zorder=4)
    ax[1].annotate("prior I (favours smooth)\nposterior:  B = +0.27",
                   xy=p1, xytext=(0.83, 0.68), fontsize=8.2, color=RED,
                   ha="center", va="center",
                   arrowprops=dict(arrowstyle="-", color=RED, lw=0.9))
    ax[1].annotate("prior II (favours blocky)\nposterior:  B = -0.23",
                   xy=p2, xytext=(0.28, -0.60), fontsize=8.2, color=BLU,
                   ha="center", va="center",
                   arrowprops=dict(arrowstyle="-", color=BLU, lw=0.9))
    ax[1].set_title("2.  PROBABILISTIC — put a MEASURE on F",
                    fontsize=11.5, weight="bold")
    ax[1].text(0.44, -0.84,
               "both priors fit the data; both are defensible — and\n"
               "they disagree on the sign of B.  Nothing inside a\n"
               "single Bayesian run flags that the conflict exists.",
               ha="center", va="top", fontsize=9, color="0.25")

    # ---- panel 3: possibilistic - the shape of F ------------------------
    base(ax[2])
    ax[2].fill(fx, fy, color="#cfe3f0", ec="#3b7aa8", lw=1.6, zorder=2)
    xmin, xmax = fx.min(), fx.max()
    ymin, ymax = fy.min(), fy.max()
    imn, imx = int(np.argmin(fx)), int(np.argmax(fx))
    jmn, jmx = int(np.argmin(fy)), int(np.argmax(fy))
    yb, xb = -0.72, -0.12                       # the projection bars

    for xx, yy in ((xmin, fy[imn]), (xmax, fy[imx])):
        ax[2].plot([xx, xx], [yb, yy], ls=":", color=RED, lw=0.9, zorder=3)
    for yy, xx in ((ymin, fx[jmn]), (ymax, fx[jmx])):
        ax[2].plot([xb, xx], [yy, yy], ls=":", color=AMB, lw=0.9, zorder=3)

    ax[2].plot([xmin, xmax], [yb, yb], color=RED, lw=6,
               solid_capstyle="butt", zorder=4)
    for xx in (xmin, xmax):
        ax[2].plot([xx, xx], [yb - 0.045, yb + 0.045], color=RED, lw=2.4,
                   zorder=4)
    ax[2].text((xmin + xmax) / 2, -0.85, "A:  FORCED",
               ha="center", va="top", fontsize=9, weight="bold", color=RED)

    ax[2].plot([xb, xb], [ymin, ymax], color=AMB, lw=6,
               solid_capstyle="butt", zorder=4)
    for yy in (ymin, ymax):
        ax[2].plot([xb - 0.035, xb + 0.035], [yy, yy], color=AMB, lw=2.4,
                   zorder=4)
    ax[2].plot(xb, 0.0, "o", color="0.5", ms=6, zorder=5)
    ax[2].text(-0.21, (ymin + ymax) / 2, "B:  MEASURE-DEPENDENT",
               ha="center", va="center", fontsize=8.4, weight="bold",
               color="#9a6a14", rotation=90)
    ax[2].set_title("3.  POSSIBILISTIC — read the SHAPE of F",
                    fontsize=11.5, weight="bold")
    ax[2].text(0.52, -0.97,
               "project F onto each feature's axis: A's interval clears the\n"
               "threshold (forced), B's straddles zero (measure-dependent)",
               ha="center", va="top", fontsize=9, color="0.25")

    fig.suptitle("Possibilistic vs probabilistic — the same feasible set, "
                 "two different questions", fontsize=13, weight="bold")
    fig.text(0.5, 0.075,
             "FORCED:  F lies entirely on one side of zero — the feature's "
             "sign holds under every measure (every prior).        "
             "MEASURE-DEPENDENT:  F straddles zero — the data leaves the sign "
             "open; the measure decides it.",
             ha="center", fontsize=8.6, color="0.15")
    fig.text(0.5, 0.035,
             "The probabilist reports one point of F and a spread, both "
             "conditional on the prior.  The possibilist reports the part of "
             "the answer no prior can move.",
             ha="center", fontsize=8.6, style="italic", color="0.35")
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))
    fig.savefig("fig_possibilism.png", dpi=130)
    plt.close(fig)
    print("fig_possibilism.png")


def _bilinear(T, z, x):
    nz, nx = T.shape
    z = min(max(z, 0.0), nz - 1.0); x = min(max(x, 0.0), nx - 1.0)
    z0, x0 = int(z), int(x)
    z1, x1 = min(z0 + 1, nz - 1), min(x0 + 1, nx - 1)
    fz, fx = z - z0, x - x0
    return ((1 - fz) * (1 - fx) * T[z0, x0] + (1 - fz) * fx * T[z0, x1]
            + fz * (1 - fx) * T[z1, x0] + fz * fx * T[z1, x1])


def fig_ray_bending():
    """First-arrival rays bend through the medium - toward fast structure,
    away from slow. The straight-ray operator ignores this; the Eikonal
    operator does not. Rays are traced as steepest descent on the FMM
    travel-time field."""
    v = ground_truth()
    T = eikonal.fmm(1.0 / v, (2, 2))
    src = (2, 2)
    recvs = [(NZ - 3, NX - 3), (NZ - 3, NX // 2 + 4), (NZ // 2, NX - 3),
             (NZ - 3, 9), (10, NX - 3)]

    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    im = ax.imshow(v, cmap="turbo", origin="upper")
    fig.colorbar(im, ax=ax, label="Vp (km/s)", fraction=0.046)
    ax.contour(T, levels=12, colors="white", linewidths=0.5, alpha=0.6)

    for recv in recvs:
        ax.plot([src[1], recv[1]], [src[0], recv[0]], "k--", lw=1.0)  # straight
        z, x = float(recv[0]), float(recv[1])
        path = [(x, z)]
        for _ in range(2000):
            here = _bilinear(T, z, x)
            gz = _bilinear(T, z + 0.5, x) - _bilinear(T, z - 0.5, x)
            gx = _bilinear(T, z, x + 0.5) - _bilinear(T, z, x - 0.5)
            gn = np.hypot(gz, gx)
            if here <= 0.4 or gn < 1e-12:
                break
            z -= 0.4 * gz / gn; x -= 0.4 * gx / gn
            path.append((x, z))
        px, pz = zip(*path)
        ax.plot(px, pz, "k-", lw=2.0)
    ax.plot(src[1], src[0], "w*", ms=16, mec="k")
    ax.plot([r[1] for r in recvs], [r[0] for r in recvs], "wo", ms=7, mec="k")
    ax.set_title("First-arrival rays: Eikonal (solid) vs straight (dashed)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig("fig_ray_bending.png", dpi=130)
    plt.close(fig)
    print("fig_ray_bending.png")


if __name__ == "__main__":
    fig_possibilism()
    fig_schematic()
    fig_ray_bending()
