"""3D eikonal forward operator: ctypes wrapper over porotomo/c/eikonal3d.c
plus a vectorised batch ray tracer for the Frechet (sensitivity) rows.

Why this split: the FMM inner loop is heap-driven and scalar — Python is
~500x too slow at the PoroTomo problem size (186 sources x 30 members x
GN iterations), so it lives in dependency-free C (compiled on first import,
.so gitignored). The ray back-tracing is embarrassingly parallel across
receivers of one source, so it stays in numpy.

CONVENTIONS (match eikonal3d.c):
  - Fields are C-order (nz, ny, nx); cell spacing is unity; slowness is in
    seconds per cell (slowness_SI * cell_m). All metric scaling is the
    caller's job.
  - Ray termination: by DISTANCE to the source point (< 1 cell), not the 2D
    eikonal.py heuristic `T <= step` — that heuristic silently truncates
    rays whose total travel time is below `step` seconds (near offsets),
    which is wrong for the s-per-cell unit choice used here.
  - Frechet rows are returned sparse (rows, cols, vals) in CELL-LENGTH
    units: d t / d slow_cell = path length in cells. Multiply vals by
    cell_m to get d t / d slowness_SI.

Self-test:  uv run python -m porotomo.eikonal3d
"""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

_C_DIR = os.path.join(os.path.dirname(__file__), "c")
_SRC = os.path.join(_C_DIR, "eikonal3d.c")
_SO = os.path.join(_C_DIR, "eikonal3d.so")

_lib = None


def _ensure_lib() -> ctypes.CDLL:
    """Compile the C kernel if missing or older than its source, then load."""
    global _lib
    if _lib is not None:
        return _lib
    if (not os.path.exists(_SO)
            or os.path.getmtime(_SO) < os.path.getmtime(_SRC)):
        subprocess.run(
            ["cc", "-O2", "-shared", "-fPIC", "-o", _SO, _SRC],
            check=True,
        )
    lib = ctypes.CDLL(_SO)
    lib.fmm3d.restype = ctypes.c_int
    lib.fmm3d.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ]
    _lib = lib
    return lib


def fmm3d(slow_cells: np.ndarray, src_cell: tuple[int, int, int],
          ball_r: int = 5) -> np.ndarray:
    """First-arrival travel-time field from one point source.

    slow_cells : (nz, ny, nx) slowness in s/cell, C-contiguous float64.
    src_cell   : (iz, iy, ix) integer source cell.
    ball_r     : analytic source-ball radius in cells (kills the first-order
                 source-singularity error; assumes near-homogeneity within
                 the ball).
    Returns T  : (nz, ny, nx) travel times, seconds."""
    lib = _ensure_lib()
    slow_cells = np.ascontiguousarray(slow_cells, dtype=np.float64)
    nz, ny, nx = slow_cells.shape
    T = np.empty_like(slow_cells)
    rc = lib.fmm3d(slow_cells, nz, ny, nx,
                   int(src_cell[0]), int(src_cell[1]), int(src_cell[2]),
                   int(ball_r), T)
    if rc != 0:
        raise MemoryError("fmm3d: allocation failure in C kernel")
    return T


# --- vectorised trilinear sampling -----------------------------------------

def sample3(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Trilinear sample of field T at real cell coordinates pts (m, 3) in
    (z, y, x) order, edge-clamped. Returns (m,)."""
    nz, ny, nx = T.shape
    z = np.clip(pts[:, 0], 0.0, nz - 1.0)
    y = np.clip(pts[:, 1], 0.0, ny - 1.0)
    x = np.clip(pts[:, 2], 0.0, nx - 1.0)
    z0 = z.astype(int); y0 = y.astype(int); x0 = x.astype(int)
    z1 = np.minimum(z0 + 1, nz - 1)
    y1 = np.minimum(y0 + 1, ny - 1)
    x1 = np.minimum(x0 + 1, nx - 1)
    fz = z - z0; fy = y - y0; fx = x - x0
    c000 = T[z0, y0, x0]; c001 = T[z0, y0, x1]
    c010 = T[z0, y1, x0]; c011 = T[z0, y1, x1]
    c100 = T[z1, y0, x0]; c101 = T[z1, y0, x1]
    c110 = T[z1, y1, x0]; c111 = T[z1, y1, x1]
    return ((1 - fz) * ((1 - fy) * ((1 - fx) * c000 + fx * c001)
                        + fy * ((1 - fx) * c010 + fx * c011))
            + fz * ((1 - fy) * ((1 - fx) * c100 + fx * c101)
                    + fy * ((1 - fx) * c110 + fx * c111)))


def trace_rays(
    T: np.ndarray,
    src_pt: np.ndarray,
    recv_pts: np.ndarray,
    step: float = 0.4,
    max_steps: int = 6000,
):
    """Back-trace all receivers of one source through travel-time field T.

    Steepest descent on T (rays run anti-parallel to grad T), all receivers
    stepping in lockstep. Termination: Euclidean distance to src_pt < 1 cell
    (the residual distance is credited to the source cell).

    T        : (nz, ny, nx) travel-time field from fmm3d.
    src_pt   : (3,) real source position, cell coords (z, y, x).
    recv_pts : (m, 3) real receiver positions, cell coords.
    Returns (rows, cols, vals): COO triplets of the m Frechet rows; rows are
    receiver indices 0..m-1, cols are flat cell indices, vals are path
    lengths in cells (duplicate (row, col) pairs are to be summed by the
    sparse constructor).
    """
    nz, ny, nx = T.shape
    m = recv_pts.shape[0]
    pos = recv_pts.astype(float).copy()
    active = np.ones(m, dtype=bool)
    rows_l: list[np.ndarray] = []
    cols_l: list[np.ndarray] = []
    vals_l: list[np.ndarray] = []
    ridx = np.arange(m)
    half = 0.5

    for _ in range(max_steps):
        if not active.any():
            break
        p = pos[active]
        dist = np.linalg.norm(p - src_pt[None, :], axis=1)
        arrived = dist < 1.0
        if arrived.any():
            # credit the remaining straight run-in to the source cell
            aidx = ridx[active][arrived]
            sz = min(max(int(round(src_pt[0])), 0), nz - 1)
            sy = min(max(int(round(src_pt[1])), 0), ny - 1)
            sx = min(max(int(round(src_pt[2])), 0), nx - 1)
            rows_l.append(aidx)
            cols_l.append(np.full(len(aidx), (sz * ny + sy) * nx + sx))
            vals_l.append(dist[arrived])
            sel = np.flatnonzero(active)
            active[sel[arrived]] = False
            p = pos[active]
            if p.shape[0] == 0:
                break

        # central-difference gradient at the active points
        gz = sample3(T, p + [half, 0, 0]) - sample3(T, p - [half, 0, 0])
        gy = sample3(T, p + [0, half, 0]) - sample3(T, p - [0, half, 0])
        gx = sample3(T, p + [0, 0, half]) - sample3(T, p - [0, 0, half])
        gnorm = np.sqrt(gz**2 + gy**2 + gx**2)
        stuck = gnorm < 1e-14
        gnorm[stuck] = 1.0  # avoid 0/0; stuck rays fall toward the source below
        # accumulate path length in the current cell
        iz = np.clip(np.round(p[:, 0]).astype(int), 0, nz - 1)
        iy = np.clip(np.round(p[:, 1]).astype(int), 0, ny - 1)
        ix = np.clip(np.round(p[:, 2]).astype(int), 0, nx - 1)
        aidx = ridx[active]
        rows_l.append(aidx)
        cols_l.append((iz * ny + iy) * nx + ix)
        vals_l.append(np.full(len(aidx), step))
        # step downhill on T; rays in a flat region head straight at the source
        dzs = np.where(stuck, src_pt[0] - p[:, 0], -gz / gnorm)
        dys = np.where(stuck, src_pt[1] - p[:, 1], -gy / gnorm)
        dxs = np.where(stuck, src_pt[2] - p[:, 2], -gx / gnorm)
        snorm = np.sqrt(dzs**2 + dys**2 + dxs**2)
        snorm[snorm < 1e-14] = 1.0
        pos[active, 0] = np.clip(p[:, 0] + step * dzs / snorm, 0, nz - 1)
        pos[active, 1] = np.clip(p[:, 1] + step * dys / snorm, 0, ny - 1)
        pos[active, 2] = np.clip(p[:, 2] + step * dxs / snorm, 0, nx - 1)

    rows = np.concatenate(rows_l) if rows_l else np.empty(0, int)
    cols = np.concatenate(cols_l) if cols_l else np.empty(0, int)
    vals = np.concatenate(vals_l) if vals_l else np.empty(0, float)
    return rows, cols, vals


if __name__ == "__main__":
    # Self-test 1: homogeneous medium vs analytic spherical wavefront.
    nz, ny, nx = 24, 40, 40
    s0 = 0.03                                  # s per cell
    slow = np.full((nz, ny, nx), s0)
    src = (2, 20, 20)
    T = fmm3d(slow, src)
    zz, yy, xx = np.mgrid[0:nz, 0:ny, 0:nx]
    T_exact = s0 * np.sqrt(
        (zz - src[0]) ** 2.0 + (yy - src[1]) ** 2.0 + (xx - src[2]) ** 2.0
    )
    far = T_exact > 5 * s0
    rel = np.abs(T - T_exact)[far] / T_exact[far]
    print(f"FMM3D vs analytic (homogeneous, {nz}x{ny}x{nx}):")
    print(f"  far-field mean relative error {rel.mean() * 100:.2f}%")
    print(f"  far-field max  relative error {rel.max() * 100:.2f}%")

    # Self-test 2: ray kernel length vs geometric distance.
    src_pt = np.array(src, dtype=float)
    recvs = np.array([[20.0, 35.0, 5.0], [21.0, 8.0, 33.0], [3.0, 22.0, 21.0]])
    rows, cols, vals = trace_rays(T, src_pt, recvs)
    import numpy as _np
    for r in range(len(recvs)):
        ksum = vals[rows == r].sum()
        geom = float(_np.linalg.norm(recvs[r] - src_pt))
        print(f"  ray {r}: kernel sum {ksum:.2f} vs geometric {geom:.2f} "
              f"({abs(ksum - geom) / geom * 100:.1f}% off)")

    # Self-test 3: predicted time from kernel vs sampled T (consistency).
    t_sampled = sample3(T, recvs)
    for r in range(len(recvs)):
        t_kernel = (vals[rows == r] * s0).sum()
        print(f"  ray {r}: t(kernel) {t_kernel:.4f} vs T(recv) "
              f"{t_sampled[r]:.4f}")

    # timing at the PoroTomo problem size
    import time
    nz2, ny2, nx2 = 12, 44, 40
    slow2 = np.full((nz2, ny2, nx2), 0.03)
    t0 = time.time()
    n_rep = 50
    for _ in range(n_rep):
        fmm3d(slow2, (0, 22, 20))
    dt = (time.time() - t0) / n_rep
    print(f"timing: {dt * 1000:.2f} ms per FMM solve on "
          f"{nz2}x{ny2}x{nx2} ({nz2 * ny2 * nx2} cells)")

    ok = rel.mean() < 0.05
    print("SELF-TEST:", "PASS" if ok else "CHECK - errors above tolerance")
