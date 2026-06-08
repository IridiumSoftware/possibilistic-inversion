"""
volve/inversion_eikonal.py - 1D Vp(z) feasible-set inversion with the
EIKONAL forward operator. Phase 4 replacement for the straight-ray
phase 3 inversion.

WHY EIKONAL. Phase 3 used a straight-ray forward (path = chord from
source to receiver). The held-out calibration on real Volve picks
showed the resulting ensemble was over-confident by ~10x and biased
the predicted travel times ~120 ms early - because real first
arrivals bend toward higher-velocity zones (Snell's law / Fermat),
shortcutting through deeper-faster overburden. The straight chord
overestimates path length and so under-estimates Vp.

A true first-arrival forward model is the eikonal equation
    |grad T|^2 = 1 / Vp^2,
solved by FMM. This module wraps eikonal.fmm (the project's existing
2D FMM implementation) for a 1D-Vp(z) parameterization that exploits
LATERAL TRANSLATION SYMMETRY: when Vp depends only on z, the
travel-time field T(x, z) from a surface source depends only on
HORIZONTAL OFFSET |x - x_source| and depth. So ONE FMM solve per Vp
model suffices; per-pick travel times are sampled from that single
field at (|x_r - x_s|, z_r). The cost per Vp model is ~1 FMM solve
plus 1 ray-trace per pick (for the Jacobian) instead of 145 FMM
solves.

NONLINEAR GAUSS-NEWTON. Travel time is a nonlinear functional of Vp,
so the inversion is iterative:
  1. start with reference Vp_ref(z), compute slowness s = 1/Vp;
  2. FMM at s -> T field;
  3. ray-trace from each pick's (offset, depth) back to source ->
     per-cell path-length kernel; sum across x to get the
     per-depth-layer path length (the Jacobian row);
  4. damped LSQR step: ds = (J^T J + lam^2 I)^{-1}
                         (J^T (t_obs - t_pred) - lam^2 (s - s_ref));
  5. update s, repeat.

For each ensemble member we draw a smooth random Vp_ref(z) and run
this GN loop, with lambda bisected so the final RMS residual lands
near a realistic noise floor.

UNITS / CONVENTIONS:
  - Vertical z: depth below sea surface in metres.
  - Horizontal x: distance from a local origin in metres.
  - Source at (x = 0, z = source_z_m); pick offsets are mapped into
    this local frame via |source_x - receiver_x| / dx grid.
  - Cell size dx = dz = CELL_SIZE_M (default 25 m).
  - Slowness s in s/m; travel time T in seconds.

Run:  uv run python -m volve.inversion_eikonal
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d

import eikonal as _eik
from volve.inversion_1d import (
    PickData, load_picks, depth_grid_for_picks,
    EnsembleConfig,
)


CELL_SIZE_M = 25.0           # grid cell spacing (square cells)
SOURCE_GRID_X_CELL = 0       # source always at i_x = 0 in local frame
SOURCE_GRID_Z_M = 5.0        # source z (5 m below sea surface)


# --- grid setup -----------------------------------------------------------

def grid_dimensions(picks: PickData,
                    cell_m: float = CELL_SIZE_M,
                    z_pad_m: float = 100.0,
                    x_pad_m: float = 100.0) -> Tuple[int, int, float]:
    """Determine (nz, nx, cell_m) covering all source-receiver geometries.

    nz : enough cells to reach the deepest receiver + pad
    nx : enough cells to reach the largest source-receiver horizontal
         offset + pad
    """
    z_max = float(picks.receivers_xyz[:, 2].max()) + z_pad_m
    nz = int(np.ceil(z_max / cell_m)) + 1
    src_xy = picks.sources_xyz[:, :2]
    rec_xy = picks.receivers_xyz[:, :2]
    dxy = rec_xy - src_xy
    offsets = np.sqrt(dxy[:, 0] ** 2 + dxy[:, 1] ** 2)
    x_max = float(offsets.max()) + x_pad_m
    nx = int(np.ceil(x_max / cell_m)) + 1
    return nz, nx, cell_m


def pick_grid_coords(picks: PickData, cell_m: float = CELL_SIZE_M):
    """For each pick, the grid coords of source (always (0, source_z))
    and receiver (|offset|, recv_z) in the local source-centred frame.
    Returns float arrays of shape (n_picks,) for ix_recv and iz_recv."""
    dxy = picks.receivers_xyz[:, :2] - picks.sources_xyz[:, :2]
    offsets_m = np.sqrt(dxy[:, 0] ** 2 + dxy[:, 1] ** 2)
    ix_recv = offsets_m / cell_m
    iz_recv = picks.receivers_xyz[:, 2] / cell_m
    return ix_recv, iz_recv


# --- forward + Jacobian for one Vp(z) ------------------------------------

def forward_eikonal_1d(vp_kms: np.ndarray,
                       picks: PickData,
                       nz: int, nx: int,
                       cell_m: float = CELL_SIZE_M,
                       ix_recv: Optional[np.ndarray] = None,
                       iz_recv: Optional[np.ndarray] = None,
                       compute_jacobian: bool = False):
    """Eikonal travel-time forward for a laterally-uniform Vp(z) model.

    Returns (t_pred_s, J_depth) where:
      t_pred_s : (n_picks,) predicted travel times in seconds
      J_depth  : (n_picks, nz_depth_bins) per-depth-layer path-length
                 Jacobian if compute_jacobian, else None.

    We do ONE FMM solve at the source (always at x = 0 in the local
    frame); each receiver samples the resulting T field at its
    (|x_r - x_s|, z_r) coordinate."""
    if ix_recv is None or iz_recv is None:
        ix_recv, iz_recv = pick_grid_coords(picks, cell_m)
    # Build slowness array, replicate Vp(z) along x.
    # vp_kms has one entry per depth BIN; we need slowness at each grid CELL.
    # Assume vp_kms is indexed by a depth grid of step `bin_thick_m`; we
    # interpolate to the FMM cell grid.
    n_depth_bins = vp_kms.size
    z_centers_m = (np.arange(nz) + 0.5) * cell_m
    # depth_centers of vp grid: same spacing if EnsembleConfig.bin_thick_m
    # was set up to match. Here we accept whatever spacing came in via
    # depth_grid_for_picks: 70 m bins.
    bin_thick_m = (cell_m * nz) / n_depth_bins if n_depth_bins else cell_m
    # Instead: be explicit - caller passes a grid via depth_grid_for_picks,
    # so n_depth_bins -> per-bin Vp; we map by floor(z_cell / bin_thick).
    # Actually keep things simple: interpolate vp_kms (one value per
    # 70 m bin) onto z_centers_m (25 m grid) via piecewise-constant
    # extension by bin index.
    bin_idx = np.minimum((z_centers_m / bin_thick_m).astype(int),
                         n_depth_bins - 1)
    vp_cell_kms = vp_kms[bin_idx]                       # (nz,)
    slow_cell = (1.0 / (vp_cell_kms * 1000.0)) * cell_m  # s per unit-cell
    slow_2d = np.broadcast_to(slow_cell[:, None], (nz, nx)).copy()
    src_iz = int(round(SOURCE_GRID_Z_M / cell_m))
    src_ix = SOURCE_GRID_X_CELL
    T = _eik.fmm(slow_2d, (src_iz, src_ix))
    # Sample at receivers (bilinear).
    t_pred = np.array([_eik._sample(T, iz, ix)
                       for iz, ix in zip(iz_recv, ix_recv)])

    if not compute_jacobian:
        return t_pred, None

    # Jacobian: ray-trace from each receiver back to source. Per-cell
    # path length; sum over x to collapse to per-depth-layer
    # contribution. Map z-cells to vp-bins via bin_idx.
    n_picks = len(ix_recv)
    J_depth = np.zeros((n_picks, n_depth_bins), dtype=float)
    for i, (iz, ix) in enumerate(zip(iz_recv, ix_recv)):
        path = _eik.ray_path(T, (iz, ix))
        # path is (nz, nx); sum across x to get path length per z cell;
        # then bin by depth-layer index. Path units are CELL UNITS (h=1);
        # multiply by cell_m to get metres (the d t / d (s_cell) Jacobian
        # is the path length in metres / cell_m, since slow_cell is in
        # (s/m) * cell_m, the Jacobian w.r.t. slow_cell is path*cell_m).
        # We want d t / d Vp_bin instead: via slowness chain rule
        # d t / d (1/V_bin) = sum over cells in bin of path_cell_metres.
        path_per_z = path.sum(axis=1) * cell_m         # metres per z-row
        for j, b in enumerate(bin_idx):
            J_depth[i, b] += path_per_z[j]
    return t_pred, J_depth


# --- nonlinear GN ensemble member ----------------------------------------

@dataclass
class EikonalConfig:
    n_members: int = 30
    n_gn_iters: int = 4
    cell_m: float = CELL_SIZE_M
    noise_rms_s: float = 0.020          # eikonal expected misfit floor
    vp_min_kms: float = 1.5
    vp_max_kms: float = 5.5
    smooth_correlation_m: float = 250.0
    lam_lo: float = 50.0
    lam_hi: float = 1e6
    bisect_iters: int = 30
    bisect_tol: float = 1.15
    seed: int = 20260607
    verbose: bool = False


def _smooth_random_vp_kms(rng, n_bins, cfg, bin_thick_m):
    z_frac = np.linspace(0.0, 1.0, n_bins)
    trend = 1.7 + 2.8 * z_frac
    sigma = max(1.0, cfg.smooth_correlation_m / bin_thick_m)
    noise = gaussian_filter1d(rng.standard_normal(n_bins), sigma=sigma) * 0.5
    return np.clip(trend + noise, cfg.vp_min_kms, cfg.vp_max_kms)


def _gn_solve(vp_ref_kms, picks, grid, nz, nx, cell_m, cfg,
              ix_recv, iz_recv):
    """Damped Gauss-Newton iterations toward s_ref = 1/(vp_ref*1000).
    Returns (vp_kms_final, rms_s_history)."""
    vp = vp_ref_kms.copy()
    s_ref = 1.0 / (vp_ref_kms * 1000.0)
    s = s_ref.copy()
    d_obs = picks.times_s
    rms_history = []
    for it in range(cfg.n_gn_iters):
        vp = 1.0 / (s * 1000.0)
        t_pred, J = forward_eikonal_1d(
            vp, picks, nz, nx, cell_m,
            ix_recv=ix_recv, iz_recv=iz_recv,
            compute_jacobian=True)
        resid = d_obs - t_pred
        rms = float(np.sqrt(np.mean(resid ** 2)))
        rms_history.append(rms)
        if cfg.verbose:
            print(f"    GN iter {it}: rms={rms*1000:.1f} ms")
        # Bisect lambda so the trial step lands near noise rms.
        lo, hi = cfg.lam_lo, cfg.lam_hi
        s_trial = s
        for _bi in range(cfg.bisect_iters):
            lam = np.sqrt(lo * hi)
            A = J.T @ J + (lam ** 2) * np.eye(J.shape[1])
            b = J.T @ resid - (lam ** 2) * (s - s_ref)
            ds = np.linalg.solve(A, b)
            s_try = np.clip(
                s + ds,
                1.0 / (cfg.vp_max_kms * 1000.0),
                1.0 / (cfg.vp_min_kms * 1000.0),
            )
            t_try, _ = forward_eikonal_1d(
                1.0 / (s_try * 1000.0), picks, nz, nx, cell_m,
                ix_recv=ix_recv, iz_recv=iz_recv,
                compute_jacobian=False)
            rms_try = float(np.sqrt(np.mean((d_obs - t_try) ** 2)))
            if rms_try < cfg.noise_rms_s:
                lo = lam
            else:
                hi = lam
            if abs(rms_try - cfg.noise_rms_s) <= (cfg.bisect_tol - 1.0) * cfg.noise_rms_s:
                s_trial = s_try
                break
            s_trial = s_try
        s = s_trial
    vp_final = 1.0 / (s * 1000.0)
    return vp_final, rms_history


def vp_ensemble_eikonal(picks: PickData,
                        grid: np.ndarray,
                        cfg: Optional[EikonalConfig] = None):
    """Random-reference Gauss-Newton eikonal ensemble. Returns
    (members_vp_kms, meta) shaped like volve.inversion_1d.vp_ensemble."""
    if cfg is None:
        cfg = EikonalConfig()
    nz, nx, cell_m = grid_dimensions(picks, cell_m=cfg.cell_m)
    ix_recv, iz_recv = pick_grid_coords(picks, cell_m)
    n_bins = grid.size - 1
    bin_thick = float(grid[1] - grid[0])
    rng = np.random.default_rng(cfg.seed)
    members = np.zeros((cfg.n_members, n_bins), dtype=float)
    rms_finals = np.zeros(cfg.n_members, dtype=float)
    for k in range(cfg.n_members):
        vp_ref = _smooth_random_vp_kms(rng, n_bins, cfg, bin_thick)
        if cfg.verbose:
            print(f"  member {k+1}/{cfg.n_members}")
        vp_final, rms_hist = _gn_solve(
            vp_ref, picks, grid, nz, nx, cell_m, cfg, ix_recv, iz_recv)
        members[k] = vp_final
        rms_finals[k] = rms_hist[-1]
    return members, {
        "members_rms_s": rms_finals,
        "n_bins": n_bins,
        "bin_thick_m": bin_thick,
        "nz_grid": nz,
        "nx_grid": nx,
        "cell_m": cell_m,
        "cfg": cfg,
    }


if __name__ == "__main__":
    picks = load_picks()
    grid = depth_grid_for_picks(picks)
    nz, nx, cell_m = grid_dimensions(picks)
    print(f"picks: {picks.n()} ok")
    print(f"grid: {nz} x {nx} cells of {cell_m:.1f} m  "
          f"(z to {nz * cell_m:.0f} m, x to {nx * cell_m:.0f} m)")
    print(f"depth bins: {grid.size - 1} of "
          f"{grid[1] - grid[0]:.1f} m to {grid[-1]:.0f} m")
    print()
    print("N=1 smoke: one ensemble member, verbose")
    cfg = EikonalConfig(n_members=1, n_gn_iters=3, verbose=True)
    members, meta = vp_ensemble_eikonal(picks, grid, cfg)
    print(f"\nfinal Vp(z):")
    for i_bin in range(0, members.shape[1], max(1, members.shape[1] // 8)):
        zc = (i_bin + 0.5) * meta["bin_thick_m"]
        print(f"  z = {zc:.0f} m   Vp = {members[0, i_bin]:.2f} km/s")
