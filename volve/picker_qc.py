"""
volve/picker_qc.py - QC plots for the first-arrival pick CSV.

Three panels:
  (a) Wiggle plot of one FieldRecord with picks overlaid (per-shot sanity)
  (b) Pick-time vs source-to-wellhead offset (moveout structure)
  (c) Pick-time vs receiver elevation, coloured by offset
       (the VSP travel-time-vs-depth curve - core ground truth for the
       phase-3 inversion)

Plus stdout summary.

Run:  uv run python -m volve.picker_qc
"""

import csv
from pathlib import Path
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import segyio

from volve.preprocess import bandpass


PICKS_CSV = "volve/picks/picks_z.csv"
SEGY = "volve/data/15_9-F-15 A/VSPNI_RAW_2.SEGY"


def _read_picks(path: str):
    """Return a list of dicts (one per trace) with parsed types."""
    rows = []
    with open(path, "r") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            for k in ("trace_idx", "field_record", "trace_number"):
                r[k] = int(r[k])
            for k in ("source_x", "source_y", "receiver_x", "receiver_y",
                      "receiver_elev_m", "source_offset_m",
                      "pick_time_s", "pick_quality"):
                r[k] = float(r[k]) if r[k] not in ("", "None") else None
            rows.append(r)
    return rows


def plot_wiggle_with_picks(rows, segy_path,
                           field_record: int,
                           out_path: str):
    fr_rows = [r for r in rows if r["field_record"] == field_record]
    if not fr_rows:
        raise ValueError(f"no rows for FieldRecord {field_record}")
    # Order by receiver elev
    fr_rows.sort(key=lambda r: r["receiver_elev_m"])

    idxs = [r["trace_idx"] for r in fr_rows]
    elev = [r["receiver_elev_m"] for r in fr_rows]
    picks = [r["pick_time_s"] for r in fr_rows]
    quals = [r["pick_quality"] for r in fr_rows]

    with segyio.open(segy_path, "r", ignore_geometry=True) as f:
        dt_s = float(f.samples[1] - f.samples[0]) / 1000.0
        n_samp = len(f.samples)
        t_ms = np.arange(n_samp) * dt_s * 1000
        traces = []
        for idx in idxs:
            t = f.trace[idx].astype(np.float32)
            traces.append(bandpass(t, dt_s))

    fig, ax = plt.subplots(figsize=(10, max(3, 0.7 * len(idxs))))
    yticks = []
    for i, (t, z, p) in enumerate(zip(traces, elev, picks)):
        norm = t / max(np.abs(t).max(), 1e-12) * 0.45
        ax.plot(t_ms, norm + i, color="k", lw=0.5)
        if p is not None:
            ax.plot(p * 1000, i, "rv", ms=9)
        yticks.append((i, f"z={z:.0f} m"))
    ax.set_yticks([y[0] for y in yticks])
    ax.set_yticklabels([y[1] for y in yticks], fontsize=8)
    ax.set_xlim(0, 1800)
    ax.set_xlabel("time (ms)")
    ax.set_title(f"FieldRecord {field_record}: traces + picks (red v)",
                 fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_summary_moveout(rows, out_path: str):
    ok = [r for r in rows if r["pick_time_s"] is not None]
    no = [r for r in rows if r["pick_time_s"] is None]
    offs = np.array([r["source_offset_m"] for r in ok])
    picks = np.array([r["pick_time_s"] for r in ok])
    elevs = np.array([r["receiver_elev_m"] for r in ok])
    quals = np.array([r["pick_quality"] for r in ok])

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))

    # Panel a: pick time vs receiver elevation, coloured by offset
    sc = ax[0, 0].scatter(picks, elevs, c=offs, s=8, cmap="viridis",
                          alpha=0.8)
    ax[0, 0].invert_yaxis()
    ax[0, 0].set_xlabel("pick time (s)")
    ax[0, 0].set_ylabel("receiver elevation below datum (m)")
    ax[0, 0].set_title("pick time vs receiver depth\n"
                       "(colour = source-to-well offset)")
    ax[0, 0].grid(alpha=0.3)
    plt.colorbar(sc, ax=ax[0, 0], label="offset (m)")

    # Panel b: pick time vs source offset, coloured by depth
    sc = ax[0, 1].scatter(offs, picks, c=elevs, s=8, cmap="plasma",
                          alpha=0.8)
    ax[0, 1].set_xlabel("source-to-well offset (m)")
    ax[0, 1].set_ylabel("pick time (s)")
    ax[0, 1].set_title("pick time vs offset\n(colour = receiver depth)")
    ax[0, 1].grid(alpha=0.3)
    plt.colorbar(sc, ax=ax[0, 1], label="receiver depth (m)")

    # Panel c: quality distribution
    ax[1, 0].hist(quals, bins=40, color="#2166ac")
    ax[1, 0].set_xlabel("STA/LTA at pick (quality)")
    ax[1, 0].set_ylabel("count")
    ax[1, 0].set_title(f"pick-quality distribution (N={len(ok)})")
    ax[1, 0].grid(alpha=0.3)

    # Panel d: text panel - yield + by-flag
    ax[1, 1].axis("off")
    flag_counts = {}
    for r in rows:
        flag = r["flag"]
        flag_counts[flag] = flag_counts.get(flag, 0) + 1
    msg = [
        f"pick yield: {len(ok)} / {len(rows)} = "
        f"{100 * len(ok) / len(rows):.1f}%",
        f"by flag: {flag_counts}",
        f"pick time (ok): "
        f"min={picks.min():.3f} median={np.median(picks):.3f} "
        f"max={picks.max():.3f} s",
        f"quality: "
        f"min={quals.min():.1f} median={np.median(quals):.1f} "
        f"max={quals.max():.1f}",
        f"source offsets: "
        f"min={offs.min():.0f} median={np.median(offs):.0f} "
        f"max={offs.max():.0f} m",
    ]
    ax[1, 1].text(0.0, 0.95, "\n".join(msg),
                  transform=ax[1, 1].transAxes,
                  fontsize=10, family="monospace",
                  verticalalignment="top")

    fig.suptitle("VSPNI_RAW_2 first-arrival picks - QC summary",
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main():
    rows = _read_picks(PICKS_CSV)
    print(f"loaded {len(rows)} pick rows from {PICKS_CSV}")
    # Pick 4 field records (early, mid 1, mid 2, late) for wiggle QC
    frs = sorted({r["field_record"] for r in rows})
    sel_frs = [frs[0], frs[len(frs) // 3], frs[2 * len(frs) // 3], frs[-1]]
    for fr in sel_frs:
        out = f"volve_picks_fr{fr:03d}.png"
        plot_wiggle_with_picks(rows, SEGY, fr, out)
        print(f"  wiggle plot: {out}")
    summary_path = "volve_picks_summary.png"
    plot_summary_moveout(rows, summary_path)
    print(f"summary plot: {summary_path}")


if __name__ == "__main__":
    main()
