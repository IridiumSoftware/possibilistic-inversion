"""
make_figures.py - the explanatory figures for the write-up.

  fig_feasible_set.png - the data gives a SET of models; within it a feature
                         is either FORCED or MEASURE-DEPENDENT - both terms
                         defined on the figure itself (the definitional one);
  fig_two_reports.png  - the same set F, two reports: possibilistic (uses
                         exactly F) vs Bayesian (uses F plus a measure mu);
  fig_bounded_uncertainty.png
                       - the two layers in sequence: possibilism bounds the
                         admissible measures, then the Bayesian sweep stays
                         inside that bound (possibilism brackets Bayes);
  fig_schematic.png    - probabilistic vs possibilistic reading of an ensemble
                         (the central idea, as a 1-D cartoon);
  fig_ray_bending.png  - straight rays vs first-arrival Eikonal rays through
                         the synthetic model (why the forward operator matters).

The two demonstration figures (possibilistic_decomposition*.png) are produced
by synthetic_demo.py and synthetic_demo_eikonal.py.

The conceptual figures (fig_feasible_set, fig_two_reports,
fig_bounded_uncertainty) replace the earlier single three-panel
fig_possibilism.png. It was split on reviewer feedback (H. Crane, R. Martin):
one figure was doing too much, and "forced" / "measure-dependent" were used
before being defined. Each figure now carries one idea and defines its own
terms; fig_bounded_uncertainty was added to show the two layers composing in
sequence rather than rivalling each other.

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


# shared palette: forced = red, measure-dependent = amber, the alternate
# Bayesian measure = blue, inert grey for a de-emphasised set.
RED, AMB, BLU, GREY = "#b2182b", "#e8a33d", "#2166ac", "#9aa7ad"


def _feasible_blob():
    """The feasible set F: a smooth lumpy closed blob in (A, B) model space.
    Geometry is chosen so F lies entirely right of A = 0 (feature A is
    forced) and straddles B = 0 (feature B is measure-dependent)."""
    th = np.linspace(0.0, 2.0 * np.pi, 400)
    r = (0.30 + 0.045 * np.cos(3 * th + 0.6) + 0.030 * np.cos(5 * th - 1.1)
              + 0.022 * np.cos(2 * th + 2.0))
    fx = 0.60 + r * np.cos(th) * 1.05
    fy = 0.05 + r * np.sin(th) * 1.40
    return fx, fy


def fig_feasible_set():
    """Figure 1 — the definitional figure. The data picks out a *set* F of
    models, not a point. Project F onto a feature's axis: if the projection
    clears zero the feature is FORCED; if it straddles zero it is
    MEASURE-DEPENDENT. Both terms are defined on the figure itself — this is
    the figure a probabilist reader meets first, and the one the reviewer
    feedback (Crane: "I don't understand what forced means") targets."""
    from matplotlib.path import Path

    fx, fy = _feasible_blob()
    xmin, xmax = fx.min(), fx.max()
    ymin, ymax = fy.min(), fy.max()

    fig = plt.figure(figsize=(13.0, 6.2))
    ax = fig.add_axes((0.045, 0.115, 0.525, 0.775))

    # zero reference lines — the thresholds the projections are read against
    ax.axvline(0.0, color="0.55", lw=1.1, zorder=1)
    ax.axhline(0.0, color="0.55", lw=1.1, zorder=1)
    ax.text(0.022, 0.585, "A = 0", fontsize=8, color="0.5")
    ax.text(0.86, 0.022, "B = 0", fontsize=8, color="0.5")

    # the feasible set, with a scatter of the models that make it up
    ax.fill(fx, fy, color="#cfe3f0", ec="#3b7aa8", lw=1.8, zorder=3)
    poly = Path(np.column_stack([fx, fy]))
    rng = np.random.default_rng(7)
    pts = []
    while len(pts) < 9:
        p = (rng.uniform(xmin, xmax), rng.uniform(ymin, ymax))
        if poly.contains_point(p):
            pts.append(p)
    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], "o", ms=5, color="#1f4e6b", alpha=0.75,
            zorder=4)
    ax.text(0.60, 0.05, "F", ha="center", va="center", fontsize=22,
            style="italic", color="#1f4e6b", zorder=5)

    ybar, xbar = -0.80, -0.14                    # the two projection bars

    # ---- A projection: clears zero -> FORCED ----------------------------
    for xx in (xmin, xmax):
        yy = fy[int(np.argmin(np.abs(fx - xx)))]
        ax.plot([xx, xx], [ybar, yy], ls=":", color=RED, lw=0.9, zorder=2)
    ax.plot([xmin, xmax], [ybar, ybar], color=RED, lw=7,
            solid_capstyle="butt", zorder=4)
    for xx in (xmin, xmax):
        ax.plot([xx, xx], [ybar - 0.05, ybar + 0.05], color=RED, lw=2.5,
                zorder=4)
    ax.plot([0.0, xmin], [ybar, ybar], ls=":", color="0.6", lw=0.9, zorder=2)
    ax.text((xmin + xmax) / 2, ybar - 0.10,
            "feature A — F projects to this interval; it clears zero",
            ha="center", va="top", fontsize=8.6, color=RED, weight="bold")

    # ---- B projection: straddles zero -> MEASURE-DEPENDENT --------------
    for yy in (ymin, ymax):
        xx = fx[int(np.argmin(np.abs(fy - yy)))]
        ax.plot([xbar, xx], [yy, yy], ls=":", color=AMB, lw=0.9, zorder=2)
    ax.plot([xbar, xbar], [ymin, ymax], color=AMB, lw=7,
            solid_capstyle="butt", zorder=4)
    for yy in (ymin, ymax):
        ax.plot([xbar - 0.045, xbar + 0.045], [yy, yy], color=AMB, lw=2.5,
                zorder=4)
    ax.plot(xbar, 0.0, "o", color="0.35", ms=7, zorder=5)
    ax.text(xbar + 0.055, 0.0, "crosses zero", fontsize=7.3, color="0.4",
            va="center")
    ax.text(xbar - 0.085, (ymin + ymax) / 2,
            "feature B — F projects to this interval; it straddles zero",
            ha="center", va="center", fontsize=8.6, color="#9a6a14",
            weight="bold", rotation=90)

    ax.set_xlim(-0.40, 1.06)
    ax.set_ylim(-1.04, 0.66)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("amplitude of feature A", fontsize=9.5)
    ax.set_ylabel("amplitude of feature B", fontsize=9.5)
    for s in ax.spines.values():
        s.set_visible(False)

    # ---- definition cards (the point of the figure) ---------------------
    fig.text(0.620, 0.840, "FORCED", fontsize=11.5, weight="bold", color=RED)
    fig.text(0.620, 0.805,
             "F's projection onto the feature's axis lies entirely on\n"
             "one side of zero.  Every model that fits the data agrees\n"
             "on this feature's sign.  F settles it — the answer needs\n"
             "no prior and no measure.   (Feature A, above.)",
             fontsize=9.3, va="top", ha="left", color="0.12",
             linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.7", fc="#fbe9e9", ec=RED,
                       lw=1.3))

    fig.text(0.620, 0.475, "MEASURE-DEPENDENT", fontsize=11.5, weight="bold",
             color="#9a6a14")
    fig.text(0.620, 0.440,
             "F's projection straddles zero.  The data leaves this\n"
             "feature's sign open.  Any single value reported here is\n"
             "fixed by the measure (prior) you add — not by the data.\n"
             "F neither forces it nor forbids it.   (Feature B, above.)",
             fontsize=9.3, va="top", ha="left", color="0.12",
             linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.7", fc="#fdf1df", ec=AMB,
                       lw=1.3))

    fig.text(0.620, 0.140,
             "Each dot is one model consistent with the data; F is the\n"
             "set of all of them.  Reading facts off F — what holds for\n"
             "every model in F, vs what F leaves open — is the\n"
             "possibilistic layer.",
             fontsize=8.8, va="top", ha="left", style="italic",
             color="0.35", linespacing=1.4)

    fig.suptitle("Figure 1.  The data gives a SET of models — and within "
                 "it, a feature is forced or measure-dependent",
                 fontsize=12.5, weight="bold", y=0.965)
    fig.savefig("fig_feasible_set.png", dpi=130)
    plt.close(fig)
    print("fig_feasible_set.png")


def fig_two_reports():
    """Figure 2 — the same feasible set F, two reports. The possibilistic
    report uses *exactly* F (no more, no less): forced features pinned,
    measure-dependent features returned as intervals. The Bayesian report
    uses F *plus* a measure mu; two defensible choices of mu can disagree on
    a measure-dependent feature, so the choice of mu needs its own
    justification. That last clause is reviewer R. Martin's point, drawn."""
    from matplotlib.patches import Ellipse

    fx, fy = _feasible_blob()
    xmin, xmax = fx.min(), fx.max()
    ymin, ymax = fy.min(), fy.max()

    fig, ax = plt.subplots(1, 2, figsize=(14.0, 6.1))

    def base(a):
        a.axvline(0.0, color="0.6", lw=1.0, zorder=1)
        a.axhline(0.0, color="0.6", lw=1.0, zorder=1)
        a.set_xlim(-0.44, 1.12)
        a.set_ylim(-1.02, 0.96)
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)

    # ---- left: the possibilistic report -- the answer read straight off F
    # F's shadow on each axis IS the report: A's shadow clears A = 0 (forced),
    # B's shadow straddles B = 0 (sign open). The shadows make the claims
    # visible -- a forced sign is a gap between F's shadow and the zero line.
    base(ax[0])
    ax[0].fill(fx, fy, color="#cfe3f0", ec="#3b7aa8", lw=1.6, zorder=3)
    ax[0].text(0.60, 0.05, "F", ha="center", va="center", fontsize=17,
               style="italic", color="#1f4e6b", zorder=4)
    ax[0].text(0.026, 0.64, "A = 0", fontsize=7.6, color="0.45")
    ax[0].text(0.99, 0.05, "B = 0", fontsize=7.6, color="0.45")

    ybar, xbar = -0.66, -0.22

    # A: F's shadow on the A-axis -- a clear gap from A = 0 -> forced +
    for xx in (xmin, xmax):
        yy = fy[int(np.argmin(np.abs(fx - xx)))]
        ax[0].plot([xx, xx], [ybar, yy], ls=":", color=RED, lw=0.9, zorder=2)
    ax[0].plot([xmin, xmax], [ybar, ybar], color=RED, lw=7,
               solid_capstyle="butt", zorder=4)
    for xx in (xmin, xmax):
        ax[0].plot([xx, xx], [ybar - 0.05, ybar + 0.05], color=RED, lw=2.4,
                   zorder=4)
    ax[0].plot([0.0, 0.0], [ybar - 0.06, ybar + 0.06], color="0.5", lw=1.5,
               zorder=2)
    ax[0].plot([0.0, xmin], [ybar, ybar], ls=(0, (2, 2)), color="0.55",
               lw=1.1, zorder=2)
    ax[0].text(xmin / 2, ybar + 0.13, "gap from\nA = 0", fontsize=7,
               color="0.45", ha="center", va="bottom", linespacing=1.2)
    ax[0].text((xmin + xmax) / 2, ybar - 0.14,
               "A's shadow on the A-axis clears A = 0\n→   A:  +, forced",
               ha="center", va="top", fontsize=8.4, weight="bold", color=RED,
               linespacing=1.5)

    # B: F's shadow on the B-axis -- straddles B = 0 -> sign open
    for yy in (ymin, ymax):
        xx = fx[int(np.argmin(np.abs(fy - yy)))]
        ax[0].plot([xbar, xx], [yy, yy], ls=":", color=AMB, lw=0.9, zorder=2)
    ax[0].plot([xbar, xbar], [ymin, ymax], color=AMB, lw=7,
               solid_capstyle="butt", zorder=4)
    for yy in (ymin, ymax):
        ax[0].plot([xbar - 0.05, xbar + 0.05], [yy, yy], color=AMB, lw=2.4,
                   zorder=4)
    ax[0].plot(xbar, 0.0, "o", color="0.35", ms=6, zorder=5)
    ax[0].text(xbar - 0.085, (ymin + ymax) / 2,
               "B's shadow straddles B = 0   →   B:  sign open",
               ha="center", va="center", fontsize=8.4, weight="bold",
               color="#9a6a14", rotation=90)
    ax[0].set_title("POSSIBILISTIC REPORT      uses exactly F",
                    fontsize=11, weight="bold", color="#1f4e6b", pad=14)

    # ---- right: the Bayesian report ------------------------------------
    base(ax[1])
    ax[1].fill(fx, fy, color="#eceef0", ec=GREY, lw=1.5, zorder=2)
    ax[1].text(0.60, 0.05, "F", ha="center", va="center", fontsize=17,
               style="italic", color="0.55", zorder=3)
    ax[1].text(0.026, 0.64, "A = 0", fontsize=7.6, color="0.45")
    ax[1].text(0.99, 0.05, "B = 0", fontsize=7.6, color="0.45")
    p1, p2 = (0.66, 0.30), (0.55, -0.24)
    ax[1].add_patch(Ellipse(p1, 0.27, 0.21, angle=18, color=RED,
                            alpha=0.24, zorder=3))
    ax[1].add_patch(Ellipse(p2, 0.31, 0.19, angle=-12, color=BLU,
                            alpha=0.24, zorder=3))
    ax[1].plot(*p1, "o", color=RED, ms=8, zorder=4)
    ax[1].plot(*p2, "o", color=BLU, ms=8, zorder=4)
    # both posteriors give A > 0 (the forced feature) — they agree there;
    # they split only on B (the measure-dependent feature).
    ax[1].annotate("measure μ₁\nposterior:  A = +0.66,  B = +0.30", xy=p1,
                   xytext=(0.80, 0.76), fontsize=8, color=RED,
                   ha="center", va="center",
                   arrowprops=dict(arrowstyle="-", color=RED, lw=0.9))
    ax[1].annotate("measure μ₂\nposterior:  A = +0.55,  B = −0.24", xy=p2,
                   xytext=(0.30, -0.76), fontsize=8, color=BLU,
                   ha="center", va="center",
                   arrowprops=dict(arrowstyle="-", color=BLU, lw=0.9))
    ax[1].set_title("BAYESIAN REPORT      uses F  +  a measure μ",
                    fontsize=11, weight="bold", color="0.3", pad=14)

    fig.suptitle("Figure 2.  The same feasible set F — two reports",
                 fontsize=12.5, weight="bold", y=0.97)
    fig.text(0.275, 0.130,
             "Returns the shape of F: forced features pinned to a sign,\n"
             "measure-dependent features returned as intervals.  Exactly\n"
             "the information in F — no more, and no less.",
             ha="center", fontsize=8.8, color="0.18", linespacing=1.5)
    fig.text(0.745, 0.130,
             "Places a measure μ on F; reports its mean and credible "
             "interval.\nBoth μ give A > 0 — there F forces the answer, no μ "
             "can move it.\nThey split on B's sign — that choice of μ needs a "
             "justification of its own.",
             ha="center", fontsize=8.8, color="0.18", linespacing=1.5)
    fig.text(0.5, 0.038,
             "Where F forces the answer, the two reports agree.  Where F "
             "leaves it open, the Bayesian number is whatever μ supplied "
             "— and the possibilistic report marks exactly those features.",
             ha="center", fontsize=8.8, style="italic", color="0.35")
    fig.tight_layout(rect=(0, 0.17, 1, 0.92))
    fig.savefig("fig_two_reports.png", dpi=130)
    plt.close(fig)
    print("fig_two_reports.png")


def fig_bounded_uncertainty():
    """Figure 3 — the two layers in sequence. The possibilistic treatment
    runs first and bounds which measures are admissible: a measure is
    admissible only if it is supported on F. The Bayesian treatment runs
    second, inside that bound — sweep every admissible measure and the
    spread of the answer (the measure-uncertainty) can never exceed F's
    extent. Possibilism does not compete with Bayes; it brackets it.

    Panel 2 draws the bound as F's own bounding extent (a red bar along F's
    A-extent, an amber bar along its B-extent) so the bound hugs F rather
    than flying off to the panel edge."""
    from matplotlib.patches import Ellipse, FancyArrowPatch
    from matplotlib.path import Path

    fx, fy = _feasible_blob()
    xmin, xmax = fx.min(), fx.max()
    ymin, ymax = fy.min(), fy.max()
    poly = Path(np.column_stack([fx, fy]))
    GRN = "#2e7d32"

    fig, ax = plt.subplots(1, 2, figsize=(14.6, 6.6))
    fig.subplots_adjust(left=0.045, right=0.965, top=0.82, bottom=0.205,
                        wspace=0.14)

    def base(a):
        a.axvline(0.0, color="0.62", lw=1.0, zorder=1)
        a.axhline(0.0, color="0.62", lw=1.0, zorder=1)
        a.set_xlim(-0.52, 1.32)
        a.set_ylim(-1.12, 0.96)
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)

    # ---- panel 1: possibilism bounds which measures are admissible -----
    base(ax[0])
    ax[0].fill(fx, fy, color="#cfe3f0", ec="#3b7aa8", lw=1.7, zorder=2)
    ax[0].text(0.44, -0.20, "F", ha="center", va="center", fontsize=18,
               style="italic", color="#1f4e6b", zorder=3)

    # an admissible measure -- supported entirely within F
    ax[0].add_patch(Ellipse((0.62, 0.15), 0.30, 0.20, angle=14,
                            facecolor=GRN, edgecolor="none", alpha=0.24,
                            zorder=4))
    ax[0].add_patch(Ellipse((0.62, 0.15), 0.30, 0.20, angle=14,
                            facecolor="none", edgecolor=GRN, lw=1.5,
                            zorder=4))
    ax[0].plot(0.62, 0.15, "o", color=GRN, ms=4.5, zorder=5)
    ax[0].annotate("admissible μ — supported\nentirely inside F   ✓",
                   xy=(0.75, 0.21), xytext=(0.16, 0.72), fontsize=8.3,
                   color=GRN, ha="center", va="center",
                   arrowprops=dict(arrowstyle="-", color=GRN, lw=0.9))

    # an inadmissible measure -- spills past F's boundary
    ax[0].add_patch(Ellipse((1.06, -0.05), 0.32, 0.24, angle=-20,
                            facecolor=RED, edgecolor="none", alpha=0.13,
                            zorder=4))
    ax[0].add_patch(Ellipse((1.06, -0.05), 0.32, 0.24, angle=-20,
                            facecolor="none", edgecolor=RED, lw=1.5,
                            ls=(0, (4, 2)), zorder=4))
    ax[0].annotate("inadmissible μ — puts weight\non models F rules out   ✗",
                   xy=(1.18, -0.13), xytext=(0.82, -0.80), fontsize=8.3,
                   color=RED, ha="center", va="center",
                   arrowprops=dict(arrowstyle="-", color=RED, lw=0.9))
    ax[0].set_title("STEP 1 — POSSIBILISTIC\nbound the admissible measures",
                    fontsize=10.5, weight="bold", color="#1f4e6b", pad=12)

    # ---- panel 2: the Bayesian sweep cannot escape F's extent ----------
    base(ax[1])
    ax[1].text(0.03, -0.52, "A = 0", fontsize=7.6, color="0.5")
    ax[1].text(1.0, 0.06, "B = 0", fontsize=7.6, color="0.5")
    ax[1].fill(fx, fy, color="#e6eff4", ec="#3b7aa8", lw=1.6, zorder=2)

    # the family of posteriors -- one per admissible measure, all inside F
    rng = np.random.default_rng(11)
    fam = []
    while len(fam) < 11:
        p = (rng.uniform(xmin, xmax), rng.uniform(ymin, ymax))
        if poly.contains_point(p):
            fam.append(p)
    fam = np.array(fam)
    ax[1].plot(fam[:, 0], fam[:, 1], "o", ms=5.5, color="#3a3a3a", zorder=6)
    ax[1].text(0.52, -0.24, "F", ha="center", va="center", fontsize=14,
               style="italic", color="#6b97b2", zorder=3)

    # F's A-extent -- the bound on A's measure-uncertainty, clears A = 0
    yb = ymin - 0.05
    ax[1].plot([xmin, xmax], [yb, yb], color=RED, lw=6,
               solid_capstyle="butt", zorder=4)
    for xx in (xmin, xmax):
        ax[1].plot([xx, xx], [yb - 0.05, yb + 0.05], color=RED, lw=2.3,
                   zorder=4)
    ax[1].plot([0.0, 0.0], [yb - 0.05, yb + 0.05], color="0.5", lw=1.5,
               zorder=2)
    ax[1].plot([0.0, xmin], [yb, yb], ls=(0, (2, 2)), color="0.55", lw=1.1,
               zorder=2)
    ax[1].annotate("A — F's extent bounds the measure-uncertainty;\n"
                   "it clears A = 0   →   sign forced +",
                   xy=(0.62, yb), xytext=(0.60, -0.84), fontsize=8.2,
                   weight="bold", color=RED, ha="center", va="top",
                   linespacing=1.5,
                   arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))

    # F's B-extent -- the bound on B's measure-uncertainty, straddles B = 0
    xb = xmin - 0.05
    ax[1].plot([xb, xb], [ymin, ymax], color=AMB, lw=6,
               solid_capstyle="butt", zorder=4)
    for yy in (ymin, ymax):
        ax[1].plot([xb - 0.05, xb + 0.05], [yy, yy], color=AMB, lw=2.3,
                   zorder=4)
    ax[1].plot(xb, 0.0, "o", color="0.35", ms=5.5, zorder=6)
    ax[1].annotate("B — F's extent bounds\nthe measure-uncertainty;\n"
                   "it straddles B = 0   →   open",
                   xy=(xb, 0.20), xytext=(0.02, 0.72), fontsize=8.2,
                   weight="bold", color="#9a6a14", ha="center", va="center",
                   linespacing=1.5,
                   arrowprops=dict(arrowstyle="-", color="#9a6a14", lw=0.8))
    ax[1].set_title("STEP 2 — BAYESIAN\nthe sweep stays inside the bound",
                    fontsize=10.5, weight="bold", color="0.3", pad=12)

    # ---- the "then" arrow: possibilism first, Bayes second -------------
    fig.add_artist(FancyArrowPatch((0.477, 0.50), (0.533, 0.50),
                                   transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=24,
                                   color="0.5", lw=2.4))
    fig.text(0.505, 0.575, "then", ha="center", va="center", fontsize=9.5,
             style="italic", color="0.45")

    fig.suptitle("Figure 3.  The layers compose — possibilism runs first "
                 "and bounds the Bayesian measure-uncertainty",
                 fontsize=12.5, weight="bold", y=0.965)
    fig.text(0.27, 0.115,
             "The data and the hard physical constraints fix F.  A measure "
             "is admissible\nonly if it is supported on F — possibilism "
             "bounds which priors are allowed.",
             ha="center", fontsize=8.7, color="0.18", linespacing=1.5)
    fig.text(0.74, 0.115,
             "Sweep every admissible μ; each gives a posterior, all of them "
             "inside F.\nThe spread of the answer — the measure-uncertainty "
             "— never exceeds F's extent.",
             ha="center", fontsize=8.7, color="0.18", linespacing=1.5)
    fig.text(0.5, 0.04,
             "Possibilism does not compete with Bayes — it runs first and "
             "brackets it.  The feasible interval of Figure 1 is exactly that "
             "bound: the most any admissible prior can move the answer.",
             ha="center", fontsize=8.7, style="italic", color="0.35")
    fig.savefig("fig_bounded_uncertainty.png", dpi=130)
    plt.close(fig)
    print("fig_bounded_uncertainty.png")


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
    fig_feasible_set()
    fig_two_reports()
    fig_bounded_uncertainty()
    fig_schematic()
    fig_ray_bending()
