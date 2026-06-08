"""
volve/inversion_1d.py - 1D Vp(z) feasible-set inversion of the picked
first arrivals from VSPNI_RAW_2.

WHY 1D. The Volve walkaway VSP has 145 unique source positions clustered
mostly at ~570 m offset and 224 unique receiver depths along a single
(deviated) well. The lateral aperture is small relative to the depth
range. A 1D Vp(z) parameterization with straight-line ray paths through
horizontally-layered medium captures most of the data variance and is
the standard VSP travel-time inversion shape. A 2D treatment with the
deviated well's xy track is the natural next step.

FORWARD MODEL.
For each (source, receiver) pair (xs, ys, zs) -> (xr, yr, zr), assume
the seismic path is a straight line through a horizontally-layered
Vp(z) medium. The travel time is then:

    t = sum over depth-bins j of (path_length_in_bin_j / Vp_j)

The path length in bin j depends on the ray geometry. For a bin at
depth [z_top, z_bot] intersected by a straight line from source to
receiver, the path length in the bin is the line-element length over
that depth interval.

INVERSION.
G x = d, where:
  - G[i, j] is the path-length contribution of bin j to pick i,
    in metres;
  - x_j = 1 / Vp_j (slowness) for bin j, in s/m;
  - d_i is the observed travel time for pick i, in seconds.

Ensemble generation by "random reference, damped LSQR" (same pattern
as the synthetic eikonal demo): for each of N members, draw a random
reference slowness profile, solve a damped least-squares problem
toward that reference, bisecting lambda so the RMS residual equals
the noise level. Each member is a feasible Vp(z).

CONVENTIONS:
  - Vertical coordinate z = depth below sea surface (z=0 at sea
    surface, z positive downward).
  - Sources at z = SOURCE_Z_M (5 m below sea surface per the
    velocity report).
  - Receiver elevations from SEG-Y headers are in metres below the
    ElevationScalar datum (= MSL = sea surface here, per Volve
    convention).
  - All velocities in m/s; all distances in metres; all times in
    seconds.

API: pick_data() loads the CSV picks; build_G() constructs the path-
length matrix; vp_ensemble() runs the random-reference LSQR ensemble.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import csv

import numpy as np

from volve.geometry import SOURCE_DEPTH_M


# --- depth grid -----------------------------------------------------------

DEPTH_BIN_THICK_M = 70.0
DEPTH_MIN_M = 0.0
DEPTH_MAX_M = 3500.0


def depth_grid(thick_m: float = DEPTH_BIN_THICK_M,
               z_min: float = DEPTH_MIN_M,
               z_max: float = DEPTH_MAX_M) -> np.ndarray:
    """Edges of the 1D depth bins, in metres below sea surface.
    Returned array has shape (n_bins + 1,)."""
    n_bins = int(round((z_max - z_min) / thick_m))
    return z_min + np.arange(n_bins + 1) * thick_m


def depth_grid_for_picks(picks: "PickData",
                         thick_m: float = DEPTH_BIN_THICK_M,
                         z_min: float = DEPTH_MIN_M,
                         pad_m: float = 0.0) -> np.ndarray:
    """Auto-sized grid: max bin edge sits at most pad_m above the deepest
    receiver in the picks, so every bin has at least one ray crossing it."""
    z_max = float(picks.receivers_xyz[:, 2].max()) + pad_m
    return depth_grid(thick_m=thick_m, z_min=z_min, z_max=z_max)


def depth_centers(grid: np.ndarray) -> np.ndarray:
    return 0.5 * (grid[:-1] + grid[1:])


# --- picks loader (filter to ok) ------------------------------------------

@dataclass
class PickData:
    """Loaded picks, filtered to flag='ok'.
    Shape conventions:
      sources_xyz   : (n, 3) source coordinates
      receivers_xyz : (n, 3) receiver coordinates
      times_s       : (n,)   observed travel times
    """
    sources_xyz: np.ndarray
    receivers_xyz: np.ndarray
    times_s: np.ndarray
    quality: np.ndarray
    field_records: np.ndarray
    source_offsets_m: np.ndarray

    def n(self) -> int:
        return self.times_s.size


def load_picks(path: str = "volve/picks/picks_z.csv",
               flag_keep: str = "ok",
               source_z_m: float = SOURCE_DEPTH_M) -> PickData:
    """Load picks CSV; keep only `flag_keep` rows. Source z is taken as
    `source_z_m` below sea surface (positive downward)."""
    src, rec, t, q, fr, off = [], [], [], [], [], []
    with open(path, "r") as fh:
        for row in csv.DictReader(fh):
            if row["flag"] != flag_keep:
                continue
            src.append([float(row["source_x"]),
                        float(row["source_y"]),
                        source_z_m])
            rec.append([float(row["receiver_x"]),
                        float(row["receiver_y"]),
                        float(row["receiver_elev_m"])])
            t.append(float(row["pick_time_s"]))
            q.append(float(row["pick_quality"]))
            fr.append(int(row["field_record"]))
            off.append(float(row["source_offset_m"]))
    return PickData(
        sources_xyz=np.array(src, dtype=float),
        receivers_xyz=np.array(rec, dtype=float),
        times_s=np.array(t, dtype=float),
        quality=np.array(q, dtype=float),
        field_records=np.array(fr, dtype=int),
        source_offsets_m=np.array(off, dtype=float),
    )


# --- forward operator -----------------------------------------------------

def straight_ray_path_lengths(source_xyz: np.ndarray,
                              receiver_xyz: np.ndarray,
                              grid: np.ndarray) -> np.ndarray:
    """Path length of the straight-line ray from source to receiver
    inside each depth bin defined by `grid` (edges, length n_bins + 1).
    Returns an array of shape (n_bins,) in metres."""
    sx, sy, sz = source_xyz
    rx, ry, rz = receiver_xyz
    dx = rx - sx
    dy = ry - sy
    dz = rz - sz
    L = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    n_bins = grid.size - 1
    lengths = np.zeros(n_bins, dtype=float)
    if abs(dz) < 1e-9:
        # Horizontal ray: doesn't actually cross depth bins meaningfully;
        # all length stays in the bin containing z = sz.
        bin_idx = int(np.searchsorted(grid, sz) - 1)
        if 0 <= bin_idx < n_bins:
            lengths[bin_idx] = L
        return lengths
    # For each depth bin, find the [z1, z2] subinterval covered by the
    # ray, then scale by dL/dz = L / |dz|.
    z_top = grid[:-1]
    z_bot = grid[1:]
    # The ray's depth span between source and receiver is [zmin, zmax]:
    zmin = min(sz, rz)
    zmax = max(sz, rz)
    overlap_top = np.maximum(z_top, zmin)
    overlap_bot = np.minimum(z_bot, zmax)
    overlap = np.maximum(0.0, overlap_bot - overlap_top)
    lengths = overlap * (L / abs(dz))
    return lengths


def build_G(picks: PickData, grid: np.ndarray) -> np.ndarray:
    """Forward-operator matrix G of shape (n_picks, n_bins) in metres,
    using straight-line rays through 1D Vp(z)."""
    n_picks = picks.n()
    n_bins = grid.size - 1
    G = np.zeros((n_picks, n_bins), dtype=float)
    for i in range(n_picks):
        G[i, :] = straight_ray_path_lengths(
            picks.sources_xyz[i], picks.receivers_xyz[i], grid)
    return G


# --- ensemble generator ---------------------------------------------------

@dataclass
class EnsembleConfig:
    n_members: int = 80
    # Effective data uncertainty: picker uncertainty (~5 ms) PLUS
    # straight-ray modeling error from approximating refracted eikonal
    # rays as straight lines. On the Volve 15/9-F-15A geometry the
    # diagnostic sweep shows the lowest achievable RMS without the Vp
    # envelope clip kicking in is ~80 ms. Setting noise_rms_s below
    # that lands the bisection in the unphysical-Vp regime and clip-
    # damages the ensemble; setting it well above weakens the data
    # constraint. 0.080 s is the honest sweet spot.
    noise_rms_s: float = 0.080
    vp_min_kms: float = 1.5             # km/s, water column / soft sed
    vp_max_kms: float = 5.5             # km/s, max plausible at depth
    smooth_correlation_m: float = 250.0  # smoothness scale for ref profiles
    # Damping bracket. lam acts as a characteristic path length in the
    # damping term lam^2 * ||s - s_ref||^2; below ~50 the solution wanders
    # into the singular null-space (Vp goes wildly unphysical); above
    # ~1e5 the reference dominates. Bracket spans both extremes so
    # bisection can find any reasonable misfit.
    lam_lo: float = 50.0
    lam_hi: float = 1e6
    bisect_tol: float = 1.10            # accept when RMS within 10% of target
    bisect_iters: int = 50
    seed: int = 20260607


def _smooth_random_vp_kms(rng: np.random.Generator,
                          n_bins: int, cfg: EnsembleConfig,
                          bin_thick_m: float) -> np.ndarray:
    """A smooth random Vp(z) profile in km/s within the envelope, with a
    monotonic upward depth trend perturbed by smooth noise."""
    # depth trend: 1.7 -> 4.5 km/s baseline
    z_frac = np.linspace(0.0, 1.0, n_bins)
    trend = 1.7 + 2.8 * z_frac
    # Smooth noise: Gaussian-filtered white noise
    from scipy.ndimage import gaussian_filter1d
    sigma_bins = max(1.0, cfg.smooth_correlation_m / bin_thick_m)
    white = rng.standard_normal(n_bins)
    noise = gaussian_filter1d(white, sigma=sigma_bins) * 0.5
    vp = trend + noise
    return np.clip(vp, cfg.vp_min_kms, cfg.vp_max_kms)


def _damped_solve(G: np.ndarray, d: np.ndarray,
                  s_ref: np.ndarray, lam: float) -> np.ndarray:
    """Damped least-squares: minimize ||G s - d||^2 + lam^2 ||s - s_ref||^2.
    Returns s. We solve via normal equations (n_bins is small)."""
    n_bins = G.shape[1]
    A = G.T @ G + (lam ** 2) * np.eye(n_bins)
    b = G.T @ d + (lam ** 2) * s_ref
    return np.linalg.solve(A, b)


def _bisect_lambda(G: np.ndarray, d: np.ndarray, s_ref: np.ndarray,
                   target_rms: float, cfg: EnsembleConfig) -> Tuple[np.ndarray, float]:
    """Bisect lambda so that RMS data misfit lands at `target_rms`."""
    lo, hi = cfg.lam_lo, cfg.lam_hi
    for _ in range(cfg.bisect_iters):
        mid = np.sqrt(lo * hi)
        s = _damped_solve(G, d, s_ref, mid)
        rms = float(np.sqrt(np.mean((G @ s - d) ** 2)))
        if rms < target_rms:
            lo = mid
        else:
            hi = mid
        if abs(rms - target_rms) <= (cfg.bisect_tol - 1.0) * target_rms:
            return s, mid
    return s, mid


def vp_ensemble(picks: PickData,
                grid: np.ndarray,
                cfg: Optional[EnsembleConfig] = None) -> Tuple[np.ndarray, dict]:
    """Random-reference damped-LSQR ensemble. Returns:
       members_vp_kms : ndarray (n_members, n_bins) - Vp in km/s per member
       meta           : dict with G, d_pred per member, residual RMS, etc.
    """
    if cfg is None:
        cfg = EnsembleConfig()
    G = build_G(picks, grid)              # path lengths in m
    d = picks.times_s.copy()              # travel times in s
    rng = np.random.default_rng(cfg.seed)
    n_bins = G.shape[1]
    bin_thick = float(grid[1] - grid[0])
    members_vp = np.zeros((cfg.n_members, n_bins), dtype=float)
    members_rms = np.zeros(cfg.n_members, dtype=float)
    members_lam = np.zeros(cfg.n_members, dtype=float)
    for k in range(cfg.n_members):
        # Random reference Vp profile -> slowness (s/m, NOT km/s reciprocal)
        vp_ref_kms = _smooth_random_vp_kms(rng, n_bins, cfg, bin_thick)
        s_ref = 1.0 / (vp_ref_kms * 1000.0)
        s, lam = _bisect_lambda(G, d, s_ref, cfg.noise_rms_s, cfg)
        # Clip slowness so Vp stays in envelope
        s = np.clip(s, 1.0 / (cfg.vp_max_kms * 1000.0),
                       1.0 / (cfg.vp_min_kms * 1000.0))
        members_vp[k] = 1.0 / (s * 1000.0)
        members_rms[k] = float(np.sqrt(np.mean((G @ s - d) ** 2)))
        members_lam[k] = lam
    return members_vp, {
        "G": G,
        "d": d,
        "members_rms_s": members_rms,
        "members_lambda": members_lam,
        "n_bins": n_bins,
        "bin_thick_m": bin_thick,
        "depth_centers_m": depth_centers(grid),
        "cfg": cfg,
    }


if __name__ == "__main__":
    picks = load_picks()
    print(f"loaded {picks.n()} ok picks")
    grid = depth_grid_for_picks(picks)
    print(f"depth grid: {grid.size - 1} bins of {grid[1] - grid[0]:.1f} m "
          f"from {grid[0]:.0f} to {grid[-1]:.0f} m below sea surface "
          f"(receiver depth max = {picks.receivers_xyz[:, 2].max():.0f} m)")
    members_vp, meta = vp_ensemble(picks, grid)
    print(f"ensemble: {members_vp.shape[0]} feasible Vp(z) profiles, "
          f"RMS residual median = "
          f"{np.median(meta['members_rms_s']) * 1000:.2f} ms")
    print(f"Vp range across ensemble:")
    for i_bin in range(0, members_vp.shape[1], max(1, members_vp.shape[1] // 8)):
        zc = meta["depth_centers_m"][i_bin]
        vmin = members_vp[:, i_bin].min()
        vmax = members_vp[:, i_bin].max()
        print(f"  z = {zc:.0f} m   Vp = {vmin:.2f} .. {vmax:.2f} km/s   "
              f"(spread = {vmax - vmin:.2f})")
