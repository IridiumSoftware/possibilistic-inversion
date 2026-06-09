"""PoroTomo phase 1 — data inventory & coverage inspection.

Quantifies what the published pick tables actually give us before any
inversion: source/station counts, picks per stage, SNR/RMSD distributions,
travel-time vs offset sanity (apparent velocity), and geometry extents.

Run:  python -m porotomo.inspect_data
Outputs: porotomo_geometry.png (repo root), summary printed to stdout.
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from porotomo.loader import load_picks, load_stations


def main() -> None:
    stations = load_stations()
    picks = load_picks()

    print(f"stations: {len(stations)}")
    sx = np.array([v[0] for v in stations.values()])
    sy = np.array([v[1] for v in stations.values()])
    sz = np.array([v[2] for v in stations.values()])
    print(
        f"  x range {sx.min():.0f}..{sx.max():.0f} m, "
        f"y range {sy.min():.0f}..{sy.max():.0f} m, "
        f"z range {sz.min():.1f}..{sz.max():.1f} m ASL"
    )

    print(f"picks: {len(picks)} total")
    n_src = len(picks.sources)
    uniq_vp = len({(s.stage, s.vibe_point) for s in picks.sources})
    uniq_vp_all = len({s.vibe_point for s in picks.sources})
    print(f"  source lines: {n_src} ({uniq_vp} unique stage/vibe-point, "
          f"{uniq_vp_all} unique vibe-point IDs)")
    for stg in (1, 2, 3, 4):
        m = picks.stage == stg
        nodes = np.unique(picks.node[m])
        srcs = len({s.vibe_point for s in picks.sources if s.stage == stg})
        print(
            f"  stage {stg}: {m.sum():6d} picks, {srcs:3d} sources, "
            f"{len(nodes)} nodes (node {nodes.min()}..{nodes.max()})"
        )

    # node coverage
    counts = np.bincount(picks.node, minlength=max(stations) + 1)
    covered = (counts > 0).sum()
    print(f"  nodes with >=1 pick: {covered}; missing: "
          f"{sorted(set(stations) - set(np.flatnonzero(counts > 0).tolist()))[:10]}")

    # picks per source-line histogram summary
    per_src = []
    idx = 0
    for s in picks.sources:
        m = (picks.stage == s.stage) & (picks.vibe_point == s.vibe_point)
        per_src.append(m.sum())
    per_src = np.array(per_src)
    print(
        f"  picks per source line: min {per_src.min()}, "
        f"median {np.median(per_src):.0f}, max {per_src.max()}"
    )

    # travel-time sanity: offset vs time, apparent velocity
    node_xyz = np.array([stations.get(n, (np.nan,) * 3) for n in picks.node])
    dx = node_xyz[:, 0] - picks.src_x
    dy = node_xyz[:, 1] - picks.src_y
    dz = node_xyz[:, 2] - picks.src_z
    offset = np.sqrt(dx**2 + dy**2 + dz**2)
    ok = np.isfinite(offset) & (picks.time_s > 0)
    vapp = offset[ok] / picks.time_s[ok]
    print(
        f"  offsets: {np.nanmin(offset):.0f}..{np.nanmax(offset):.0f} m; "
        f"apparent velocity p5/p50/p95 = "
        f"{np.percentile(vapp, 5):.0f}/{np.percentile(vapp, 50):.0f}/"
        f"{np.percentile(vapp, 95):.0f} m/s"
    )
    print(
        f"  SNR p5/p50/p95 = {np.percentile(picks.snr, 5):.1f}/"
        f"{np.percentile(picks.snr, 50):.1f}/{np.percentile(picks.snr, 95):.1f}; "
        f"RMSD p50 = {np.percentile(picks.rmsd, 50):.3f}"
    )

    # figure: map + offset-time + SNR hist + picks-per-source
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    ax = axes[0, 0]
    srcs_xy = np.array([(s.utm_e - 327000.0, s.utm_n - 4405000.0) for s in picks.sources])
    ax.scatter(sx, sy, s=14, c="tab:blue", marker="^", label=f"nodes ({len(stations)})")
    ax.scatter(srcs_xy[:, 0], srcs_xy[:, 1], s=10, c="tab:red", marker="*",
               label=f"vibe points ({uniq_vp_all})")
    ax.set_aspect("equal")
    ax.set_xlabel("x = UTM_E − 327000 (m)")
    ax.set_ylabel("y = UTM_N − 4405000 (m)")
    ax.set_title("PoroTomo geometry (UTM 11N local frame)")
    ax.legend()

    ax = axes[0, 1]
    ax.scatter(offset[ok], picks.time_s[ok], s=1, alpha=0.15, c=picks.stage[ok],
               cmap="viridis")
    for v in (1000, 2000, 4000):
        d = np.array([0, np.nanmax(offset)])
        ax.plot(d, d / v, lw=0.8, ls="--", label=f"{v} m/s")
    ax.set_xlabel("source–node offset (m)")
    ax.set_ylabel("picked travel time (s)")
    ax.set_title("offset vs travel time (color = stage)")
    ax.legend()

    ax = axes[1, 0]
    ax.hist(picks.snr, bins=80)
    ax.set_xlabel("SNR")
    ax.set_title("pick SNR distribution")
    ax.set_yscale("log")

    ax = axes[1, 1]
    ax.hist(per_src, bins=40)
    ax.set_xlabel("picks per source line")
    ax.set_title(f"coverage per source (median {np.median(per_src):.0f}/236 live nodes)")

    fig.tight_layout()
    out = "porotomo_geometry.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
