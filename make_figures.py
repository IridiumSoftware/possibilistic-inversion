"""
make_figures.py - the explanatory figures for the write-up.

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
    fig_schematic()
    fig_ray_bending()
