"""
volve/inversion_eikonal_2d.py - 2D Vp(w, z) joint eikonal inversion of
the F-15A + F-11 T2 VSP picks.

Phase 5 replacement for the phase 4 1D-Vp(z) inversion. Two wells'
picks combined into one inverse problem, parameterized on a 2D grid
in the (along-well-line, depth) plane.

GEOMETRY.
The two wellheads are 1061 m apart in the survey's UTM-like frame.
Take the line between them as the "w-axis" (along-well-line
coordinate); depth z is vertical-below-sea-surface (z > 0).
Each pick's (source_xy, receiver_xy) is projected onto this line:
  w = (xy - origin) . unit_vector
The perpendicular offset (cross-line distance) is ignored - the
inversion assumes Vp varies in (w, z) only, no out-of-plane
structure.

FORWARD.
1 FMM solve per UNIQUE source position - same as phase 4 - but
the lateral-translation-symmetry trick is GONE because Vp(w, z) is
not laterally uniform. Per ensemble member, the forward cost is
(n_unique_sources * FMM_solve) + (n_picks * ray-trace_for_Jacobian).
On the joint F-15A + F-11 T2 set: ~300 unique sources, ~2000 picks,
~5 minutes per Vp model evaluation.

ENSEMBLE.
Same random-reference Gauss-Newton pattern as phase 4: smooth random
Vp(w, z), N GN iterations, lambda bisected on trial-step RMS.

UNITS / CONVENTIONS:
  - w: along-line coordinate from origin (F-15A wellhead) toward
    F-11 T2 wellhead, in metres.
  - z: depth below sea surface, in metres (positive down).
  - Vp in km/s; slowness in s/m; T in s.

Run:  uv run python -m volve.inversion_eikonal_2d
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import csv

import numpy as np
from scipy.ndimage import gaussian_filter1d, gaussian_filter

import eikonal as _eik
from volve.inversion_1d import PickData, load_picks


# --- well-line frame -----------------------------------------------------

@dataclass
class WellLineFrame:
    """The 2D inversion plane: a line from F-15A wellhead toward
    F-11 T2 wellhead, plus a depth axis.

    project(xy) returns the w-coordinate (signed distance along the
    line from origin).
    """
    origin_xy: np.ndarray         # (2,) F-15A wellhead xy
    unit_xy: np.ndarray           # (2,) unit vector toward F-11 T2
    w_pad_m: float = 500.0        # extend the model past each wellhead

    def project(self, xy: np.ndarray) -> np.ndarray:
        """Project (n, 2) xy positions onto the well-line.
        Returns w-coordinates (n,) in metres."""
        delta = xy - self.origin_xy
        return delta @ self.unit_xy


def well_line_frame(picks_f15a: PickData,
                    picks_f11t2: PickData) -> WellLineFrame:
    """Build the line from F-15A median receiver to F-11 T2 median
    receiver (using receiver xy as a wellhead proxy)."""
    f15a = np.array([np.median(picks_f15a.receivers_xyz[:, 0]),
                     np.median(picks_f15a.receivers_xyz[:, 1])])
    f11t2 = np.array([np.median(picks_f11t2.receivers_xyz[:, 0]),
                      np.median(picks_f11t2.receivers_xyz[:, 1])])
    delta = f11t2 - f15a
    length = float(np.linalg.norm(delta))
    return WellLineFrame(origin_xy=f15a, unit_xy=delta / length)


# --- pick projection ------------------------------------------------------

@dataclass
class ProjectedPicks:
    """Picks in the (w, z) plane, ready for 2D inversion."""
    src_w: np.ndarray     # (n,) source w-coordinate
    src_z: np.ndarray     # (n,) source depth
    rec_w: np.ndarray     # (n,) receiver w
    rec_z: np.ndarray     # (n,) receiver depth
    times: np.ndarray     # (n,) observed travel times
    well_id: np.ndarray   # (n,) "f15a" or "f11t2"
    f15a_wellhead_w: float
    f11t2_wellhead_w: float

    def n(self) -> int:
        return self.times.size


def project_picks(picks: PickData, frame: WellLineFrame,
                  well_id: str) -> ProjectedPicks:
    src_w = frame.project(picks.sources_xyz[:, :2])
    rec_w = frame.project(picks.receivers_xyz[:, :2])
    return ProjectedPicks(
        src_w=src_w,
        src_z=picks.sources_xyz[:, 2],
        rec_w=rec_w,
        rec_z=picks.receivers_xyz[:, 2],
        times=picks.times_s,
        well_id=np.array([well_id] * picks.n()),
        f15a_wellhead_w=0.0,
        f11t2_wellhead_w=float(np.linalg.norm(
            frame.unit_xy * 0)),  # set below
    )


def join_projected(*pps: ProjectedPicks) -> ProjectedPicks:
    return ProjectedPicks(
        src_w=np.concatenate([p.src_w for p in pps]),
        src_z=np.concatenate([p.src_z for p in pps]),
        rec_w=np.concatenate([p.rec_w for p in pps]),
        rec_z=np.concatenate([p.rec_z for p in pps]),
        times=np.concatenate([p.times for p in pps]),
        well_id=np.concatenate([p.well_id for p in pps]),
        f15a_wellhead_w=pps[0].f15a_wellhead_w,
        f11t2_wellhead_w=pps[0].f11t2_wellhead_w,
    )


# --- 2D grid --------------------------------------------------------------

@dataclass
class Grid2D:
    cell_m: float
    nw: int
    nz: int
    w0: float            # left edge in w
    z0: float            # top edge in z

    def w_centers(self):
        return self.w0 + (np.arange(self.nw) + 0.5) * self.cell_m

    def z_centers(self):
        return self.z0 + (np.arange(self.nz) + 0.5) * self.cell_m

    def w_to_cell(self, w):
        return (w - self.w0) / self.cell_m

    def z_to_cell(self, z):
        return (z - self.z0) / self.cell_m


def make_grid(pp: ProjectedPicks, frame: WellLineFrame,
              cell_m: float = 40.0, w_pad_m: float = 300.0,
              z_pad_m: float = 100.0) -> Grid2D:
    w_min = float(min(pp.src_w.min(), pp.rec_w.min())) - w_pad_m
    w_max = float(max(pp.src_w.max(), pp.rec_w.max())) + w_pad_m
    z_min = 0.0
    z_max = float(pp.rec_z.max()) + z_pad_m
    nw = int(np.ceil((w_max - w_min) / cell_m))
    nz = int(np.ceil((z_max - z_min) / cell_m))
    return Grid2D(cell_m=cell_m, nw=nw, nz=nz, w0=w_min, z0=z_min)


# --- forward + Jacobian ---------------------------------------------------

def _unique_sources(pp: ProjectedPicks, grid: Grid2D):
    """Return (uniq_idx, inverse) for unique source grid cells. Each
    pick is mapped to one uniq_idx for its source."""
    sw_cell = np.round(grid.w_to_cell(pp.src_w)).astype(int)
    sz_cell = np.round(grid.z_to_cell(pp.src_z)).astype(int)
    keys = sz_cell * grid.nw + sw_cell
    uniq_keys, inverse = np.unique(keys, return_inverse=True)
    src_cells = np.column_stack([
        uniq_keys // grid.nw,    # iz
        uniq_keys % grid.nw,     # ix
    ])
    return src_cells, inverse


def forward_2d(vp_kms: np.ndarray, pp: ProjectedPicks, grid: Grid2D,
               src_cells: np.ndarray, src_inverse: np.ndarray,
               compute_jacobian: bool = False):
    """One FMM per unique source. Return (t_pred, J) where J is
    (n_picks, nz * nw) flattened if compute_jacobian, else None."""
    slow = (1.0 / (vp_kms * 1000.0)) * grid.cell_m         # (nz, nw) in s/cell
    n_pick = pp.n()
    n_unique = src_cells.shape[0]
    T_fields = np.empty((n_unique, grid.nz, grid.nw), dtype=float)
    for k in range(n_unique):
        T_fields[k] = _eik.fmm(slow, (int(src_cells[k, 0]),
                                       int(src_cells[k, 1])))
    # Bilinear-sample at each receiver
    iz_rec = grid.z_to_cell(pp.rec_z)
    iw_rec = grid.w_to_cell(pp.rec_w)
    t_pred = np.empty(n_pick, dtype=float)
    for i in range(n_pick):
        k = src_inverse[i]
        t_pred[i] = _eik._sample(T_fields[k], iz_rec[i], iw_rec[i])

    if not compute_jacobian:
        return t_pred, None
    # ray-trace for each pick: Jacobian row = per-cell path length
    # (flattened). Sum to grid.nz * grid.nw entries; pp picks use only
    # depth-z, lateral-w sensitivity; the eikonal.ray_path gives a
    # per-cell field that already encodes this.
    n_cells = grid.nz * grid.nw
    J = np.zeros((n_pick, n_cells), dtype=float)
    for i in range(n_pick):
        k = src_inverse[i]
        path = _eik.ray_path(T_fields[k], (iz_rec[i], iw_rec[i]))
        # Path is in cell units; multiply by cell_m to get metres.
        # The Jacobian d t / d slow_cell = path metres (since slow_cell
        # is in s/cell = (s/m) * cell_m, d t / d slow_cell = path_cells).
        # We want d t / d slow_si = path_cells * cell_m.
        J[i] = (path * grid.cell_m).ravel()
    return t_pred, J


# --- ensemble inversion --------------------------------------------------

@dataclass
class Config2D:
    n_members: int = 15
    n_gn_iters: int = 3
    cell_m: float = 40.0
    noise_rms_s: float = 0.030
    vp_min_kms: float = 1.5
    vp_max_kms: float = 5.5
    smooth_correlation_m: float = 350.0
    lam_lo: float = 100.0
    lam_hi: float = 1e6
    bisect_iters: int = 20
    bisect_tol: float = 1.20
    seed: int = 20260608
    verbose: bool = False


def _smooth_random_vp_2d(rng, grid: Grid2D, cfg: Config2D) -> np.ndarray:
    z_frac = np.linspace(0.0, 1.0, grid.nz)
    trend = 1.7 + 2.8 * z_frac
    trend_2d = np.broadcast_to(trend[:, None], (grid.nz, grid.nw)).copy()
    white = rng.standard_normal((grid.nz, grid.nw))
    sigma = cfg.smooth_correlation_m / cfg.cell_m
    noise = gaussian_filter(white, sigma=sigma) * 0.5
    return np.clip(trend_2d + noise, cfg.vp_min_kms, cfg.vp_max_kms)


def vp_ensemble_2d(pp: ProjectedPicks, grid: Grid2D,
                   cfg: Optional[Config2D] = None):
    if cfg is None:
        cfg = Config2D()
    src_cells, src_inverse = _unique_sources(pp, grid)
    n_pick = pp.n()
    n_cells = grid.nz * grid.nw
    rng = np.random.default_rng(cfg.seed)
    members = np.zeros((cfg.n_members, grid.nz, grid.nw), dtype=float)
    rms_finals = np.zeros(cfg.n_members, dtype=float)
    print(f"  unique sources: {src_cells.shape[0]}; "
          f"grid: {grid.nz} x {grid.nw}; picks: {n_pick}")
    for k in range(cfg.n_members):
        vp_ref = _smooth_random_vp_2d(rng, grid, cfg)
        s_ref = 1.0 / (vp_ref * 1000.0)
        s = s_ref.copy()
        for it in range(cfg.n_gn_iters):
            vp = 1.0 / (s * 1000.0)
            t_pred, J = forward_2d(vp, pp, grid, src_cells, src_inverse,
                                   compute_jacobian=True)
            resid = pp.times - t_pred
            rms = float(np.sqrt(np.mean(resid ** 2)))
            if cfg.verbose:
                print(f"    member {k+1} GN iter {it}: "
                      f"rms = {rms * 1000:.1f} ms")
            # Bisect lambda using forward-only trial steps.
            lo, hi = cfg.lam_lo, cfg.lam_hi
            s_flat = s.ravel()
            s_ref_flat = s_ref.ravel()
            JtJ = J.T @ J
            Jtr = J.T @ resid
            for _bi in range(cfg.bisect_iters):
                lam = np.sqrt(lo * hi)
                A = JtJ + (lam ** 2) * np.eye(n_cells)
                b = Jtr - (lam ** 2) * (s_flat - s_ref_flat)
                ds = np.linalg.solve(A, b)
                s_try_flat = np.clip(
                    s_flat + ds,
                    1.0 / (cfg.vp_max_kms * 1000.0),
                    1.0 / (cfg.vp_min_kms * 1000.0),
                )
                s_try_2d = s_try_flat.reshape(grid.nz, grid.nw)
                vp_try = 1.0 / (s_try_2d * 1000.0)
                t_try, _ = forward_2d(vp_try, pp, grid,
                                      src_cells, src_inverse,
                                      compute_jacobian=False)
                rms_try = float(np.sqrt(np.mean((pp.times - t_try) ** 2)))
                if rms_try < cfg.noise_rms_s:
                    lo = lam
                else:
                    hi = lam
                if abs(rms_try - cfg.noise_rms_s) <= \
                        (cfg.bisect_tol - 1.0) * cfg.noise_rms_s:
                    break
            s = s_try_2d
        members[k] = 1.0 / (s * 1000.0)
        # Final RMS
        t_pred, _ = forward_2d(members[k], pp, grid,
                               src_cells, src_inverse,
                               compute_jacobian=False)
        rms_finals[k] = float(np.sqrt(np.mean((pp.times - t_pred) ** 2)))
        if not cfg.verbose:
            print(f"    member {k+1}/{cfg.n_members}: "
                  f"rms = {rms_finals[k] * 1000:.1f} ms")
    return members, {
        "members_rms_s": rms_finals,
        "grid": grid,
        "src_cells": src_cells,
        "src_inverse": src_inverse,
        "cfg": cfg,
    }


# --- joint pick loader ----------------------------------------------------

def load_joint_picks() -> Tuple[ProjectedPicks, WellLineFrame]:
    picks_f15a = load_picks("volve/picks/picks_z.csv")
    picks_f11t2 = load_picks("volve/picks/picks_z_f11t2.csv")
    frame = well_line_frame(picks_f15a, picks_f11t2)
    pp_f15a = project_picks(picks_f15a, frame, "f15a")
    pp_f11t2 = project_picks(picks_f11t2, frame, "f11t2")
    # F-11 T2 well's projected w-coord = line length
    line_len = float(np.linalg.norm(frame.unit_xy * 0))  # not used; we want length
    line_len = float(np.linalg.norm(
        np.array([np.median(picks_f11t2.receivers_xyz[:, 0]),
                  np.median(picks_f11t2.receivers_xyz[:, 1])])
        - frame.origin_xy))
    joint = join_projected(pp_f15a, pp_f11t2)
    joint.f11t2_wellhead_w = line_len
    return joint, frame


if __name__ == "__main__":
    pp, frame = load_joint_picks()
    print(f"joint picks: {pp.n()} "
          f"(F-15A {(pp.well_id == 'f15a').sum()}, "
          f"F-11T2 {(pp.well_id == 'f11t2').sum()})")
    print(f"F-15A wellhead at w = {pp.f15a_wellhead_w:.1f} m")
    print(f"F-11T2 wellhead at w = {pp.f11t2_wellhead_w:.1f} m")
    print(f"source w: {pp.src_w.min():.1f} -> {pp.src_w.max():.1f}")
    print(f"recv w  : {pp.rec_w.min():.1f} -> {pp.rec_w.max():.1f}")
    print(f"recv z  : {pp.rec_z.min():.1f} -> {pp.rec_z.max():.1f}")
    grid = make_grid(pp, frame)
    print(f"grid: {grid.nw} x {grid.nz} cells of "
          f"{grid.cell_m:.0f} m (w0={grid.w0:.0f}, z0={grid.z0:.0f})")
    print()
    print("N=1 smoke: 1 ensemble member, 2 GN iters, verbose")
    cfg = Config2D(n_members=1, n_gn_iters=2, verbose=True)
    members, meta = vp_ensemble_2d(pp, grid, cfg)
    print(f"final member RMS: {meta['members_rms_s'][0] * 1000:.1f} ms")
