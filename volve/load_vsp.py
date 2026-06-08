"""
volve/load_vsp.py - SEG-Y loader for the Volve walkaway VSP shot gathers.

Returns the four-axis tensor and the associated headers needed to build a
travel-time-tomography input from raw shot gathers.

INPUTS:
  - One or more SEG-Y files in `Well_logs/08.VSP_VELOCITY/` of the Volve
    release. The exact file naming is to be confirmed once the SAS pull
    finishes (the recon's path strings are reconstructed from secondary
    writeups, not verified against the live tree).

OUTPUTS (`load_walkaway`):
  - data : ndarray of shape (n_shots, n_receivers, n_components, n_samples)
           dtype float32; raw amplitudes (un-gained, un-filtered).
  - hdr  : dict with shot/receiver coordinates and sampling.
           shots     : (n_shots, 2)     ndarray of (x, z) in m
           receivers : (n_receivers, 2) ndarray of (x, z) in m
           dt_s      : float - sample interval in seconds
           t_axis    : ndarray of shape (n_samples,) - time axis in s

DESIGN INTENT.
  - The recon told us the walkaway is 151 shots x 467 receivers x 4-C x 2001
    samples. If the SEG-Y has fewer/more traces per shot, or a different
    component layout, this loader REPORTS the mismatch and shapes the
    output tensor from header inspection - it does NOT silently truncate
    or pad to the expected counts.
  - Multi-component is assumed to be interleaved per receiver (the standard
    multi-component VSP storage layout). If the actual ordering is
    [all hydrophones, then all geophone-x, ...] we'll re-shape after a
    header peek.
  - Coordinate frame conversion is deferred. SEG-Y stores SourceX/SourceY/
    GroupX/GroupY in survey units (often centimetres-of-metres via
    CoordinateScalar); this loader returns them in metres post-conversion.

CONVENTIONS:
  - All distances in metres; all times in seconds.
  - z positive downward.
  - SEG-Y's TraceField coordinate scalar applied as per the SEG-Y rev 1
    convention (positive = multiplier, negative = divisor).

API stability: EXPERIMENTAL until validated against a real Volve SEG-Y file.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path

import numpy as np

try:
    import segyio
except ImportError as e:
    raise ImportError(
        "segyio is required; add it to pyproject.toml dependencies "
        "and run `uv sync`."
    ) from e


SCALAR_FIELD = segyio.TraceField.SourceGroupScalar


def _decode_coord_scalar(scalar: int) -> float:
    """SEG-Y CoordinateScalar: positive = multiplier, negative = -divisor.
    0 means no scaling. Returns the numeric factor to multiply raw coords by."""
    if scalar == 0:
        return 1.0
    if scalar > 0:
        return float(scalar)
    return 1.0 / float(-scalar)


def _read_xy_m(f: "segyio.SegyFile", i: int) -> Tuple[float, float, float, float]:
    """Return (source_x, source_y, recv_x, recv_y) for trace i, in metres."""
    scalar = _decode_coord_scalar(f.header[i][SCALAR_FIELD])
    sx = f.header[i][segyio.TraceField.SourceX] * scalar
    sy = f.header[i][segyio.TraceField.SourceY] * scalar
    gx = f.header[i][segyio.TraceField.GroupX] * scalar
    gy = f.header[i][segyio.TraceField.GroupY] * scalar
    return float(sx), float(sy), float(gx), float(gy)


@dataclass
class WalkawayBlock:
    """One SEG-Y file's worth of traces, geometry-decoded, undifferentiated.
    Call `bin_to_shot_recv_tensor` to fold into the 4D (n_shot, n_recv, n_comp,
    n_samp) shape."""
    file: str
    n_traces: int
    n_samples: int
    dt_s: float
    sx_m: np.ndarray   # (n_traces,)
    sy_m: np.ndarray
    gx_m: np.ndarray
    gz_m: np.ndarray   # receiver z (depth) - SEG-Y stores it in GroupY for VSP, often
    samples: np.ndarray  # (n_traces, n_samples) raw amplitudes


def load_file(path: str) -> WalkawayBlock:
    """Read one SEG-Y file: traces + headers, no reshape. Returns a
    WalkawayBlock. Use `summarize_block` to see how to fold it."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with segyio.open(str(p), "r", ignore_geometry=True) as f:
        n = f.tracecount
        ns = len(f.samples)
        dt_s = float(f.samples[1] - f.samples[0]) / 1000.0 if ns > 1 else 0.0
        sx = np.empty(n); sy = np.empty(n); gx = np.empty(n); gy = np.empty(n)
        for i in range(n):
            sx[i], sy[i], gx[i], gy[i] = _read_xy_m(f, i)
        data = f.trace.raw[:].astype(np.float32)
    return WalkawayBlock(
        file=str(p),
        n_traces=n, n_samples=ns, dt_s=dt_s,
        sx_m=sx, sy_m=sy, gx_m=gx, gz_m=gy,
        samples=data,
    )


def summarize_block(blk: WalkawayBlock) -> str:
    """Human-readable summary for header sanity-checking."""
    uniq_src = np.unique(np.column_stack([blk.sx_m, blk.sy_m]), axis=0)
    uniq_rec = np.unique(np.column_stack([blk.gx_m, blk.gz_m]), axis=0)
    lines = [
        f"file:               {blk.file}",
        f"traces:             {blk.n_traces}",
        f"samples/trace:      {blk.n_samples}",
        f"dt (s):             {blk.dt_s:.6f}",
        f"unique source xy:   {len(uniq_src)}",
        f"unique receiver xy: {len(uniq_rec)}",
        f"source x range (m): {blk.sx_m.min():.1f} -> {blk.sx_m.max():.1f}",
        f"source y range (m): {blk.sy_m.min():.1f} -> {blk.sy_m.max():.1f}",
        f"recv   x range (m): {blk.gx_m.min():.1f} -> {blk.gx_m.max():.1f}",
        f"recv   z range (m): {blk.gz_m.min():.1f} -> {blk.gz_m.max():.1f}",
    ]
    return "\n".join(lines)


def bin_to_shot_recv_tensor(blk: WalkawayBlock,
                            n_components: int = 4) -> Tuple[np.ndarray, dict]:
    """Fold a flat-trace block into (n_shots, n_recv, n_comp, n_samp).

    Assumes traces are sorted as (shot, receiver, component) - the standard
    multi-component VSP layout. If your file is sorted differently, re-sort
    blk.samples first by examining the header keys. We DO NOT silently
    reshape if the count doesn't divide evenly; we error.
    """
    if blk.n_traces % n_components != 0:
        raise ValueError(
            f"trace count {blk.n_traces} not divisible by n_components "
            f"{n_components}; pre-sort the file or pass a different "
            f"n_components.")
    # Detect shot count by unique source positions; receivers per shot is
    # then n_traces / (n_shots * n_components).
    src_xy = np.column_stack([blk.sx_m, blk.sy_m])
    uniq_src, src_idx = np.unique(src_xy, axis=0, return_inverse=True)
    n_shots = len(uniq_src)
    per_shot = blk.n_traces // n_shots
    if per_shot % n_components != 0:
        raise ValueError(
            f"{per_shot} traces per shot not divisible by {n_components} "
            f"components.")
    n_recv = per_shot // n_components
    # Sort traces by (shot_idx, then keep original order within shot - the
    # component/receiver layout is per-shot-internal and assumed consistent
    # across shots).
    order = np.argsort(src_idx, kind="stable")
    arr = blk.samples[order].reshape(n_shots, n_recv, n_components,
                                      blk.n_samples)
    src_xy_sorted = uniq_src
    # Receiver xy: take the first shot's slice as the canonical receiver layout.
    first_shot_traces = order[:per_shot]
    rec_x = blk.gx_m[first_shot_traces][::n_components]
    rec_z = blk.gz_m[first_shot_traces][::n_components]
    hdr = {
        "shots": np.column_stack([src_xy_sorted[:, 0],
                                  np.zeros(n_shots)]),  # z=0 placeholder
        "receivers": np.column_stack([rec_x, rec_z]),
        "dt_s": blk.dt_s,
        "t_axis": np.arange(blk.n_samples) * blk.dt_s,
        "n_components": n_components,
    }
    return arr, hdr


def plot_one_shot_gather(arr: np.ndarray, hdr: dict, shot_idx: int = 0,
                         component: int = 0, out_path: str = "volve_gather.png",
                         decim_t: int = 4):
    """Quick wiggle/image plot of one shot's record. arr shape
    (n_shots, n_recv, n_comp, n_samp)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = arr[shot_idx, :, component, ::decim_t]
    t = hdr["t_axis"][::decim_t]
    rz = hdr["receivers"][:, 1]
    # imshow with depth on Y and time on X
    fig, ax = plt.subplots(figsize=(11, 6))
    vmax = float(np.percentile(np.abs(g), 99))
    ax.imshow(g, aspect="auto", cmap="gray",
              extent=[t[0], t[-1], rz[-1], rz[0]],
              vmin=-vmax, vmax=vmax)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("receiver depth (m)")
    ax.set_title(f"Volve walkaway shot {shot_idx}, "
                 f"component {component} - raw")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    print("volve.load_vsp - module loaded.")
    print("Usage:")
    print("  from volve.load_vsp import load_file, summarize_block, "
          "bin_to_shot_recv_tensor")
    print("  blk = load_file('Well_logs/08.VSP_VELOCITY/<file>.sgy')")
    print("  print(summarize_block(blk))")
    print("  arr, hdr = bin_to_shot_recv_tensor(blk, n_components=4)")
