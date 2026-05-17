"""
eikonal.py - 2-D first-arrival Eikonal forward operator (Fast Marching Method).

The straight-ray operator of synthetic_demo.py assumes rays are straight -
exact only for a homogeneous medium. This module provides the faithful,
nonlinear forward model used by real first-arrival tomography (and the operator
class ZTM/TFM's FMM.cpp implements):

  * fmm(slow, src)      -> the first-arrival travel-time field T from a point
                           source, by the Fast Marching Method
                           (Sethian, "A fast marching level set method...",
                           PNAS 1996);
  * ray_path(T, recv)   -> the first-arrival ray from receiver back to source,
                           as a per-cell path-length field - the Frechet
                           kernel  d t_recv / d slow  that linearizes the
                           inversion (Fermat: the kernel is the ray itself);
  * forward(slow, srcs, recvs) -> travel times + the full Frechet matrix for
                           every source-receiver pair, at the given slowness.

Travel time is thereby a NONLINEAR functional of the slowness field; an
inversion built on this operator is correspondingly iterative (Gauss-Newton),
recomputing the ray paths through the current model each step - which is why
synthetic_demo_eikonal.py (next increment) replaces the one-shot linear solve
of synthetic_demo.py with a GN loop.

Note on fidelity: this is plain first-order FMM. ZTM's FMM.cpp uses an
FMM + Vidale-finite-difference hybrid to cut the first-order scheme's small
diagonal error; the operator class is the same, and the possibilistic
decomposition is forward-model-agnostic, so first-order FMM is sufficient here.

Conventions: grid [nz, nx], unit cell spacing h = 1; slowness s = 1/v at cell
centres; T defined at cell centres; sources/receivers given as integer cells.

Status: EXPERIMENTAL - a methodology-demonstration component.

Self-test:  uv run python eikonal.py
"""

import heapq
import numpy as np


def _update(T, frozen, s, z, x, nz, nx):
    """First-order Eikonal update at cell (z, x): with the smaller frozen
    upwind neighbour time on each axis (tz, tx) and slowness s, solve
    (T - tz)^2 + (T - tx)^2 = s^2  (cell spacing h = 1). Falls back to the
    one-sided update T = min(tz, tx) + s when the two-sided root is
    non-causal."""
    tz = np.inf
    if z > 0 and frozen[z - 1, x]:
        tz = min(tz, T[z - 1, x])
    if z < nz - 1 and frozen[z + 1, x]:
        tz = min(tz, T[z + 1, x])
    tx = np.inf
    if x > 0 and frozen[z, x - 1]:
        tx = min(tx, T[z, x - 1])
    if x < nx - 1 and frozen[z, x + 1]:
        tx = min(tx, T[z, x + 1])
    if tz == np.inf:
        return tx + s
    if tx == np.inf:
        return tz + s
    disc = 2.0 * s * s - (tz - tx) ** 2
    if disc >= 0.0:
        cand = 0.5 * (tz + tx + np.sqrt(disc))
        if cand >= max(tz, tx):                  # causal: root is upwind
            return cand
    return min(tz, tx) + s


def fmm(slow, src):
    """First-arrival travel-time field by the Fast Marching Method.

    slow : (nz, nx) slowness array.   src : (iz, ix) source cell.
    Returns T : (nz, nx) travel-time field (finite everywhere on a connected
    grid). O(N log N) - a binary heap of the narrow band."""
    nz, nx = slow.shape
    T = np.full((nz, nx), np.inf)
    frozen = np.zeros((nz, nx), bool)
    sz, sx = int(src[0]), int(src[1])
    T[sz, sx] = 0.0
    heap = [(0.0, sz, sx)]
    while heap:
        _, z, x = heapq.heappop(heap)
        if frozen[z, x]:
            continue
        frozen[z, x] = True                      # T[z, x] is now final
        for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cz, cx = z + dz, x + dx
            if 0 <= cz < nz and 0 <= cx < nx and not frozen[cz, cx]:
                cand = _update(T, frozen, slow[cz, cx], cz, cx, nz, nx)
                if cand < T[cz, cx]:
                    T[cz, cx] = cand
                    heapq.heappush(heap, (cand, cz, cx))
    return T


def _sample(T, z, x):
    """Bilinear sample of field T at real coordinate (z, x), edge-clamped."""
    nz, nx = T.shape
    z = min(max(z, 0.0), nz - 1.0)
    x = min(max(x, 0.0), nx - 1.0)
    z0, x0 = int(z), int(x)
    z1, x1 = min(z0 + 1, nz - 1), min(x0 + 1, nx - 1)
    fz, fx = z - z0, x - x0
    return ((1 - fz) * (1 - fx) * T[z0, x0] + (1 - fz) * fx * T[z0, x1]
            + fz * (1 - fx) * T[z1, x0] + fz * fx * T[z1, x1])


def ray_path(T, recv, step=0.4, max_steps=2000):
    """First-arrival ray from receiver back toward the source: steepest
    descent on the travel-time field T (rays run anti-parallel to grad T).

    Returns a (nz, nx) field of per-cell path length - one row of the Frechet
    matrix  d t_recv / d slow  (Fermat's principle: first-arrival time is
    stationary in the path, so the sensitivity to slowness is the ray length
    in each cell)."""
    nz, nx = T.shape
    G = np.zeros((nz, nx))
    z, x = float(recv[0]), float(recv[1])
    for _ in range(max_steps):
        here = _sample(T, z, x)
        gz = _sample(T, z + 0.5, x) - _sample(T, z - 0.5, x)
        gx = _sample(T, z, x + 0.5) - _sample(T, z, x - 0.5)
        gnorm = np.hypot(gz, gx)
        iz = min(max(int(round(z)), 0), nz - 1)
        ix = min(max(int(round(x)), 0), nx - 1)
        if here <= step or gnorm < 1e-12:        # arrived at the source
            G[iz, ix] += max(here, 0.0)
            break
        G[iz, ix] += step
        z -= step * gz / gnorm                   # downhill on T
        x -= step * gx / gnorm
    return G


def traveltimes(slow, sources, receivers):
    """Travel times only, no Frechet matrix - one FMM solve per source.
    The cheap forward evaluation used by the inversion's damping search
    (Levenberg-Marquardt trial steps), where the Jacobian is not needed."""
    out = []
    for s in sources:
        T = fmm(slow, s)
        out.extend(T[int(r[0]), int(r[1])] for r in receivers)
    return np.array(out)


def forward(slow, sources, receivers):
    """Travel times and the Frechet matrix for every source-receiver pair.

    slow : (nz, nx) slowness.  sources, receivers : lists of (iz, ix) cells.
    Returns (t, G):
      t : travel-time vector, length len(sources) * len(receivers);
      G : (len(t), nz*nx) Frechet matrix  d t / d slow  at this slowness.
    One FMM solve per source yields times to all receivers at once."""
    rows_t, rows_G = [], []
    for s in sources:
        T = fmm(slow, s)
        for r in receivers:
            rows_t.append(T[int(r[0]), int(r[1])])
            rows_G.append(ray_path(T, r).ravel())
    return np.array(rows_t), np.array(rows_G)


if __name__ == "__main__":
    # Self-test against the analytic homogeneous-medium solution:
    # in a constant-slowness medium the first-arrival time is exactly
    # s0 * (Euclidean distance from the source), and the ray is straight.
    nz, nx = 60, 60
    s0 = 0.16                                    # v ~ 6.25 km/s
    slow = np.full((nz, nx), s0)
    src = (0, 0)
    T = fmm(slow, src)
    zz, xx = np.mgrid[0:nz, 0:nx]
    T_exact = s0 * np.hypot(zz - src[0], xx - src[1])
    rel = np.abs(T - T_exact) / (T_exact + 1e-9)
    far = T_exact > 5 * s0                       # exclude the source singularity
    print(f"FMM vs analytic (homogeneous medium, {nz}x{nx}):")
    print(f"  far-field mean relative error {rel[far].mean() * 100:.2f}%  "
          f"(the representative accuracy)")
    print(f"  far-field max  relative error {rel[far].max() * 100:.2f}%  "
          f"(first-order scheme's diagonal transient; ZTM's FMM-VFD hybrid "
          f"targets exactly this)")

    recv = (nz - 1, nx - 1)
    G = ray_path(T, recv)
    geom = float(np.hypot(recv[0] - src[0], recv[1] - src[1]))
    print(f"ray_path Frechet kernel:")
    print(f"  kernel sum {G.sum():.3f} vs geometric ray length {geom:.3f} "
          f"({abs(G.sum() - geom) / geom * 100:.2f}% off)")
    # the straight ray should hug the main diagonal
    on_diag = sum(G[i, j] for i in range(nz) for j in range(nx)
                  if abs(i - j) <= 1)
    print(f"  {on_diag / G.sum() * 100:.1f}% of the kernel lies within "
          f"1 cell of the source-receiver straight line")

    t, Gm = forward(slow, [(0, 0), (0, nx - 1)], [(nz - 1, 0), (nz - 1, nx - 1)])
    print(f"forward(): {len(t)} ray pairs, Frechet matrix {Gm.shape}")
    # PASS on the representative accuracy (mean) and the ray-kernel accuracy;
    # the first-order scheme's local diagonal transient is expected, not a bug.
    ok = rel[far].mean() < 0.03 and abs(G.sum() - geom) / geom < 0.06
    print("SELF-TEST:", "PASS" if ok else "CHECK - errors above tolerance")
