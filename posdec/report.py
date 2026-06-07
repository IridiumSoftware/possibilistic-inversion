"""
posdec.report - the standard reporting figure (ORSI propagation #6).

Three forced-sign masks + interval-width map + coverage curve, on one
figure. The forced / measure-dependent split is a single artifact, not
three separate plots; the width map shows the magnitude of
measure-dependence everywhere; the coverage curve states how trustworthy
the whole thing is.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from posdec.decomposition import classify, interval_width


def plot_three_maps_and_width(a_min, a_max, eps,
                              coverage_curve=None,
                              false_forced_rate=None,
                              out_path=None,
                              title=None):
    cls = classify(a_min, a_max, eps)
    width = interval_width(a_min, a_max)
    forced_hi = cls == 2
    forced_lo = cls == -2
    meas_dep = cls == 1

    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0],
                          hspace=0.32, wspace=0.22)

    for col, (mask, name, cmap) in enumerate([
        (forced_hi, "forced-high",        "Reds"),
        (forced_lo, "forced-low",         "Blues"),
        (meas_dep, "measure-dependent",   "Oranges"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(mask, cmap=cmap, origin="upper", vmin=0, vmax=1)
        ax.set_title(f"{name}  ({int(mask.sum())} cells)", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    ax_w = fig.add_subplot(gs[1, 0:2])
    im = ax_w.imshow(width, cmap="viridis", origin="upper")
    ax_w.set_title("feasible-interval width  (a_max - a_min, km/s)",
                   fontsize=11)
    ax_w.set_xticks([]); ax_w.set_yticks([])
    fig.colorbar(im, ax=ax_w, fraction=0.030, pad=0.02)

    ax_c = fig.add_subplot(gs[1, 2])
    if coverage_curve and coverage_curve.get("Ns") \
            and coverage_curve.get("forced_sizes"):
        ax_c.plot(coverage_curve["Ns"], coverage_curve["forced_sizes"],
                  "-o", color="#b2182b", lw=1.4, ms=3,
                  label="forced-set size")
        if coverage_curve.get("false_forced_res") is not None:
            ax_c.plot(coverage_curve["Ns"],
                      coverage_curve["false_forced_res"],
                      "-s", color="#444", lw=1.0, ms=2.5,
                      label="false-forced (resolved)")
        ax_c.set_xlabel("ensemble size N", fontsize=9)
        ax_c.set_ylabel("cell count", fontsize=9)
        ax_c.set_title("coverage curve (RWC-1)", fontsize=11)
        ax_c.legend(fontsize=8, loc="best", frameon=False)
        ax_c.grid(alpha=0.3)
    else:
        ax_c.text(0.5, 0.55,
                  "coverage curve\nnot supplied",
                  transform=ax_c.transAxes, ha="center", va="center",
                  color="#888", fontsize=10)
        ax_c.set_title("coverage curve (RWC-1)", fontsize=11)
        ax_c.set_xticks([]); ax_c.set_yticks([])

    if false_forced_rate is not None:
        ax_c.text(0.02, -0.18,
                  f"RWC-2 false-forced rate: "
                  f"{100 * false_forced_rate:.1f}%",
                  transform=ax_c.transAxes,
                  fontsize=9, color="#444",
                  verticalalignment="top")

    if title:
        fig.suptitle(title, fontsize=12)
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
