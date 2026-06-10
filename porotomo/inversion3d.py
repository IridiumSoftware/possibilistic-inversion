"""porotomo/inversion3d.py - 3D Vp feasible-set ensemble inversion of the
PoroTomo stage-1 P-wave picks.

3D extension of the Volve phase-4/5 machinery (volve/inversion_eikonal*.py):
random-reference damped Gauss-Newton members, lambda bisected so each member
fits the picks to the noise floor; the ensemble's per-cell [min, max] is the
feasible interval that posdec decomposes.

DATASET DESIGN.
  - INVERT stage 1 only (~42k picks, 186 sources).
  - Stage 2 is never touched by the inversion: its sources re-occupy the
    same vibe points (within a few metres), so it is a genuine held-out
    calibration set (porotomo/holdout_calibration.py).
  - Stages 3/4 are reserved (candidate time-lapse study; the field's pumping
    was modulated between stages, so stage differences may be signal).
  - Empirical noise floor: stage-1-vs-2 repeat scatter on matched
    (vibe_point, node) pairs, computed in the smoke test; cfg.noise_rms_s
    defaults to Parker et al. (2018)'s reported 31 ms misfit and should be
    checked against the repeat number.

GEOMETRY / UNITS / CONVENTIONS:
  - Local frame from porotomo.loader: x = UTM_E - 327000, y = UTM_N - 4405000
    (metres, UTM 11N); elevation z_m in metres ASL, positive UP.
  - Grid index k runs DOWN: cell-centre elevation = z_top_m - (k + 0.5)*cell.
    Fields are (nz, ny, nx) C-order, matching eikonal3d.
  - Slowness passed to fmm3d is s per CELL (slowness_SI * cell_m); the
    Jacobian is converted to d t / d slowness_SI (vals * cell_m) so the
    model vector is SI slowness (s/m). Vp in km/s at the API surface.
  - TOPOGRAPHY: surface elevation is interpolated from station + source
    elevations (linear griddata, nearest fill). Cells whose centre lies
    above the surface are AIR: Vp fixed at 0.34 km/s, excluded from the
    inversion update (columns dropped), never counted in the decomposition.
    Sources/receivers are clamped to lie at or below the surface cell.

PERFORMANCE assumptions: grid ~ 13 x 45 x 42 = 25k cells fits everything in
RAM; Jacobian is scipy.sparse CSR (~3M nonzeros); the augmented damped
system is solved by LSQR. Single-threaded; one full member = a few minutes;
the 30-member ensemble is a background run, not interactive.

Run:  uv run python -m porotomo.inversion3d            # N=1 smoke
      uv run python -m porotomo.inversion3d --full     # full ensemble -> npz
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, asdict

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix, csr_matrix, eye as sp_eye, vstack as sp_vstack
from scipy.sparse.linalg import lsqr

from porotomo.loader import load_picks, load_stations
from porotomo import eikonal3d as eik3


# --- grid -------------------------------------------------------------------

@dataclass
class Grid3D:
    cell_m: float
    nz: int
    ny: int
    nx: int
    x0: float            # west edge, local-frame metres
    y0: float            # south edge
    z_top_m: float       # top-of-model elevation, m ASL

    def x_to_cell(self, x):
        return (x - self.x0) / self.cell_m - 0.5

    def y_to_cell(self, y):
        return (y - self.y0) / self.cell_m - 0.5

    def elev_to_cell(self, z_m):
        """Elevation (m ASL, positive up) -> z grid coordinate (down)."""
        return (self.z_top_m - z_m) / self.cell_m - 0.5

    def cell_centers_elev(self):
        return self.z_top_m - (np.arange(self.nz) + 0.5) * self.cell_m

    @property
    def n_cells(self):
        return self.nz * self.ny * self.nx


def make_grid(stations: dict, picks, cell_m: float = 50.0,
              xy_pad_m: float = 250.0, depth_below_m: float = 500.0) -> Grid3D:
    sx = np.array([v[0] for v in stations.values()])
    sy = np.array([v[1] for v in stations.values()])
    sz = np.array([v[2] for v in stations.values()])
    all_x = np.concatenate([sx, picks.src_x])
    all_y = np.concatenate([sy, picks.src_y])
    all_z = np.concatenate([sz, picks.src_z])
    x0 = all_x.min() - xy_pad_m
    y0 = all_y.min() - xy_pad_m
    nx = int(np.ceil((all_x.max() + xy_pad_m - x0) / cell_m))
    ny = int(np.ceil((all_y.max() + xy_pad_m - y0) / cell_m))
    z_top = all_z.max() + cell_m * 0.5
    z_bot = all_z.min() - depth_below_m
    nz = int(np.ceil((z_top - z_bot) / cell_m))
    return Grid3D(cell_m=cell_m, nz=nz, ny=ny, nx=nx, x0=x0, y0=y0,
                  z_top_m=z_top)


def surface_mask(grid: Grid3D, stations: dict, picks) -> np.ndarray:
    """Boolean (nz, ny, nx): True where the cell centre is ABOVE the
    interpolated ground surface (air)."""
    pts = np.array([[v[0], v[1]] for v in stations.values()]
                   + [[x, y] for x, y in zip(picks.src_x, picks.src_y)])
    elev = np.array([v[2] for v in stations.values()]
                    + list(picks.src_z))
    xc = grid.x0 + (np.arange(grid.nx) + 0.5) * grid.cell_m
    yc = grid.y0 + (np.arange(grid.ny) + 0.5) * grid.cell_m
    XX, YY = np.meshgrid(xc, yc)             # (ny, nx)
    surf = griddata(pts, elev, (XX, YY), method="linear")
    surf_nn = griddata(pts, elev, (XX, YY), method="nearest")
    surf = np.where(np.isnan(surf), surf_nn, surf)   # fill hull exterior
    zc = grid.cell_centers_elev()                    # (nz,)
    return zc[:, None, None] > surf[None, :, :]      # air above surface


# --- dataset ------------------------------------------------------------------

@dataclass
class Dataset3D:
    """Stage picks grouped by source, in grid coordinates."""
    src_cells: np.ndarray     # (n_src, 3) int source cells (z, y, x)
    src_pts: np.ndarray       # (n_src, 3) float source positions, cell coords
    recv_pts: list            # n_src arrays (m_i, 3): receiver cell coords
    times: list               # n_src arrays (m_i,): observed travel times, s
    vibe_point: np.ndarray    # (n_src,) vibe-point IDs
    node_ids: list            # n_src arrays (m_i,): node numbers (audit trail)

    @property
    def n_picks(self):
        return int(sum(len(t) for t in self.times))


def build_dataset(grid: Grid3D, stations: dict, picks, air: np.ndarray,
                  stage: int, vapp_range=(300.0, 4000.0)) -> Dataset3D:
    """Group one stage's picks by vibe point; drop gross outliers by
    apparent velocity (the AIC auto-picker has no physical bounds)."""
    m_st = picks.stage == stage
    node_xyz = {n: np.array(v) for n, v in stations.items()}

    # surface cell index per column, to clamp endpoints underground
    surf_k = np.argmax(~air, axis=0)         # first non-air cell per (y, x)

    def clamp_pt(x, y, z_m):
        pz = grid.elev_to_cell(z_m)
        py = grid.y_to_cell(y)
        px = grid.x_to_cell(x)
        iy = int(np.clip(round(py), 0, grid.ny - 1))
        ix = int(np.clip(round(px), 0, grid.nx - 1))
        pz = max(pz, float(surf_k[iy, ix]))  # at or below the surface cell
        return np.array([pz, py, px])

    src_cells, src_pts, recv_pts, times, vps, node_ids = [], [], [], [], [], []
    n_drop = 0
    for s in picks.sources:
        if s.stage != stage:
            continue
        sel = m_st & (picks.vibe_point == s.vibe_point)
        if not sel.any():
            continue
        nodes = picks.node[sel]
        t_obs = picks.time_s[sel]
        sp = clamp_pt(s.utm_e - 327000.0, s.utm_n - 4405000.0, s.elev_m)
        rp = np.array([clamp_pt(*node_xyz[n]) for n in nodes])
        # apparent-velocity outlier filter (metres / second)
        d_m = np.linalg.norm((rp - sp) * grid.cell_m, axis=1)
        vapp = d_m / np.maximum(t_obs, 1e-9)
        keep = (vapp >= vapp_range[0]) & (vapp <= vapp_range[1])
        n_drop += int((~keep).sum())
        if keep.sum() < 5:
            continue
        src_pts.append(sp)
        src_cells.append(np.clip(np.round(sp), 0,
                                 np.array(air.shape) - 1).astype(int))
        recv_pts.append(rp[keep])
        times.append(t_obs[keep])
        vps.append(s.vibe_point)
        node_ids.append(nodes[keep])
    print(f"  stage {stage}: {len(vps)} sources, "
          f"{sum(len(t) for t in times)} picks "
          f"({n_drop} dropped by vapp filter)")
    return Dataset3D(
        src_cells=np.array(src_cells), src_pts=np.array(src_pts),
        recv_pts=recv_pts, times=times, vibe_point=np.array(vps),
        node_ids=node_ids,
    )


# --- forward + Jacobian -------------------------------------------------------

def forward_3d(vp_kms: np.ndarray, ds: Dataset3D, grid: Grid3D,
               compute_jacobian: bool = False):
    """Predicted times for every pick (and optionally the sparse Jacobian
    d t / d slowness_SI). One FMM solve per source."""
    slow_cells = (1.0 / (vp_kms * 1000.0)) * grid.cell_m
    t_pred = []
    if compute_jacobian:
        rows_l, cols_l, vals_l = [], [], []
        row0 = 0
    for i in range(len(ds.src_pts)):
        T = eik3.fmm3d(slow_cells, tuple(ds.src_cells[i]))
        t_pred.append(eik3.sample3(T, ds.recv_pts[i]))
        if compute_jacobian:
            r, c, v = eik3.trace_rays(T, ds.src_pts[i], ds.recv_pts[i])
            rows_l.append(r + row0)
            cols_l.append(c)
            vals_l.append(v * grid.cell_m)   # cells -> d t / d slowness_SI
            row0 += len(ds.recv_pts[i])
    t_pred = np.concatenate(t_pred)
    if not compute_jacobian:
        return t_pred, None
    n_picks = ds.n_picks
    J = coo_matrix(
        (np.concatenate(vals_l),
         (np.concatenate(rows_l), np.concatenate(cols_l))),
        shape=(n_picks, grid.n_cells),
    ).tocsr()
    return t_pred, J


# --- ensemble -----------------------------------------------------------------

@dataclass
class Config3D:
    n_members: int = 30
    n_gn_iters: int = 4
    cell_m: float = 50.0
    # Achievable-floor target, not bulk pick noise: robust repeat-scatter
    # noise is ~36 ms/pick (median|dt| 34 ms stage-1-vs-2, /sqrt2, x1.4826)
    # and Parker et al. 2018 report 31 ms, but the auto-pick outlier tail
    # (p95 |dt| = 140 ms) plus first-order-FMM + 50 m-grid forward error
    # saturate the GN at ~56 ms. Fitting to 60 ms underfits the bulk noise
    # slightly -> wider feasible intervals -> conservative decomposition
    # (same convention as the Volve 1D phase, which used its achievable
    # 80 ms floor).
    noise_rms_s: float = 0.060
    vp_min_kms: float = 0.5
    vp_max_kms: float = 6.0
    vp_air_kms: float = 0.34
    smooth_correlation_m: float = 300.0
    trend_top_kms: float = 1.0
    trend_bot_kms: float = 3.5
    noise_sd_kms: float = 0.4
    lam_lo: float = 1.0
    lam_hi: float = 1e8
    bisect_iters: int = 12
    bisect_tol: float = 1.15
    lsqr_iters: int = 400
    seed: int = 20260609
    verbose: bool = False


def _smooth_random_vp_3d(rng, grid: Grid3D, cfg: Config3D) -> np.ndarray:
    z_frac = np.linspace(0.0, 1.0, grid.nz)
    trend = cfg.trend_top_kms + (cfg.trend_bot_kms - cfg.trend_top_kms) * z_frac
    vp = np.broadcast_to(trend[:, None, None],
                         (grid.nz, grid.ny, grid.nx)).copy()
    white = rng.standard_normal(vp.shape)
    sigma = cfg.smooth_correlation_m / cfg.cell_m
    noise = gaussian_filter(white, sigma=sigma)
    noise *= cfg.noise_sd_kms / max(noise.std(), 1e-12)
    return np.clip(vp + noise, cfg.vp_min_kms, cfg.vp_max_kms)


def _invert_member(args):
    """One ensemble member: smooth random reference, n_gn_iters damped GN
    steps with lambda bisected toward cfg.noise_rms_s. Module-level so it
    pickles for multiprocessing."""
    k, member_seed, ds, grid, air, cfg = args
    t_start = time.time()
    t_obs = np.concatenate(ds.times)
    ground = np.flatnonzero(~air.ravel())
    n_ground = len(ground)
    s_lo = 1.0 / (cfg.vp_max_kms * 1000.0)
    s_hi = 1.0 / (cfg.vp_min_kms * 1000.0)
    rng = np.random.default_rng(member_seed)
    vp_ref = _smooth_random_vp_3d(rng, grid, cfg)
    vp_ref[air] = cfg.vp_air_kms
    s_ref = (1.0 / (vp_ref * 1000.0)).ravel()[ground]
    s = s_ref.copy()

    def to_vp(s_ground):
        vp = np.full(grid.n_cells, 1.0 / (cfg.vp_air_kms * 1000.0))
        vp[ground] = s_ground
        return 1.0 / (vp.reshape(grid.nz, grid.ny, grid.nx) * 1000.0)

    s_try = s
    for it in range(cfg.n_gn_iters):
        t_pred, J = forward_3d(to_vp(s), ds, grid, compute_jacobian=True)
        resid = t_obs - t_pred
        rms = float(np.sqrt(np.mean(resid**2)))
        Jg = J[:, ground]
        lo, hi = cfg.lam_lo, cfg.lam_hi
        rms_prev = None
        for _bi in range(cfg.bisect_iters):
            lam = np.sqrt(lo * hi)
            A = sp_vstack([Jg, lam * sp_eye(n_ground, format="csr")],
                          format="csr")
            b = np.concatenate([resid, -lam * (s - s_ref)])
            ds_step = lsqr(A, b, iter_lim=cfg.lsqr_iters)[0]
            s_try = np.clip(s + ds_step, s_lo, s_hi)
            t_try, _ = forward_3d(to_vp(s_try), ds, grid)
            rms_try = float(np.sqrt(np.mean((t_obs - t_try)**2)))
            if cfg.verbose:
                print(f"      lam {lam:9.1f}: rms {rms_try*1000:6.1f} ms")
            if rms_try < cfg.noise_rms_s:
                lo = lam
            else:
                hi = lam
            if abs(rms_try - cfg.noise_rms_s) <= \
                    (cfg.bisect_tol - 1.0) * cfg.noise_rms_s:
                break
            # plateau: below some lambda the step is regularized by LSQR
            # truncation, not by lambda - further bisection is a no-op
            if rms_prev is not None and abs(rms_try - rms_prev) < 0.0005:
                break
            rms_prev = rms_try
        s = s_try
        if cfg.verbose:
            print(f"    member {k+1} GN iter {it}: start rms {rms*1000:.1f} ms")
    vp_final = to_vp(s)
    t_fin, _ = forward_3d(vp_final, ds, grid)
    rms_final = float(np.sqrt(np.mean((t_obs - t_fin)**2)))
    print(f"    member {k+1}: rms {rms_final*1000:.1f} ms "
          f"({time.time() - t_start:.0f} s)", flush=True)
    return k, vp_final, rms_final


def vp_ensemble_3d(ds: Dataset3D, grid: Grid3D, air: np.ndarray,
                   cfg: Config3D | None = None, n_workers: int = 1):
    """Random-reference damped-GN feasible-set ensemble.

    Members are independent given their seeds, so they parallelise across
    processes (n_workers > 1); per-member seeds are spawned from cfg.seed,
    so results are reproducible for a fixed cfg regardless of n_workers.
    Returns (members (n, nz, ny, nx), meta dict)."""
    if cfg is None:
        cfg = Config3D()
    ground = np.flatnonzero(~air.ravel())
    print(f"  grid {grid.nz}x{grid.ny}x{grid.nx} = {grid.n_cells} cells "
          f"({len(ground)} ground); picks {ds.n_picks}; "
          f"sources {len(ds.src_pts)}")
    seeds = np.random.SeedSequence(cfg.seed).spawn(cfg.n_members)
    jobs = [(k, seeds[k], ds, grid, air, cfg) for k in range(cfg.n_members)]
    members = np.zeros((cfg.n_members, grid.nz, grid.ny, grid.nx))
    rms_finals = np.zeros(cfg.n_members)
    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for k, vp, rms in pool.map(_invert_member, jobs):
                members[k] = vp
                rms_finals[k] = rms
    else:
        for job in jobs:
            k, vp, rms = _invert_member(job)
            members[k] = vp
            rms_finals[k] = rms
    return members, {
        "members_rms_s": rms_finals,
        "cfg": asdict(cfg),
    }


# --- repeat scatter (empirical noise floor) -----------------------------------

def stage_repeat_scatter(picks, st_a: int = 1, st_b: int = 2) -> dict:
    """Match picks on (vibe_point, node) across two stages; the scatter of
    the time differences is an empirical bound on pick noise (it also
    contains any real time-lapse signal between the stages, so it is an
    UPPER bound on instrument+picking noise)."""
    key_a = {}
    for i in np.flatnonzero(picks.stage == st_a):
        key_a[(picks.vibe_point[i], picks.node[i])] = picks.time_s[i]
    dt = []
    for i in np.flatnonzero(picks.stage == st_b):
        k = (picks.vibe_point[i], picks.node[i])
        if k in key_a:
            dt.append(picks.time_s[i] - key_a[k])
    dt = np.array(dt)
    return {
        "n_matched": len(dt),
        "median_abs_ms": float(np.median(np.abs(dt)) * 1000),
        "rms_ms": float(np.sqrt(np.mean(dt**2)) * 1000),
        "p95_abs_ms": float(np.percentile(np.abs(dt), 95) * 1000),
    }


# --- entry points ---------------------------------------------------------------

def prepare(stage: int = 1, cell_m: float = 50.0):
    stations = load_stations()
    picks = load_picks()
    grid = make_grid(stations, picks, cell_m=cell_m)
    air = surface_mask(grid, stations, picks)
    ds = build_dataset(grid, stations, picks, air, stage=stage)
    return picks, grid, air, ds


def main() -> None:
    full = "--full" in sys.argv
    picks, grid, air, ds = prepare()
    rep = stage_repeat_scatter(picks)
    print(f"  stage1/2 repeat scatter: {rep['n_matched']} matched pairs, "
          f"median|dt| {rep['median_abs_ms']:.1f} ms, "
          f"rms {rep['rms_ms']:.1f} ms, p95 {rep['p95_abs_ms']:.1f} ms")
    if full:
        import os
        cfg = Config3D()
        n_workers = min(10, max(1, (os.cpu_count() or 4) - 2))
        print(f"  running {cfg.n_members} members on {n_workers} workers")
        members, meta = vp_ensemble_3d(ds, grid, air, cfg,
                                       n_workers=n_workers)
        out = "porotomo/data/ensemble_stage1.npz"
        np.savez_compressed(
            out, members=members, air=air,
            members_rms_s=meta["members_rms_s"],
            grid_cell_m=grid.cell_m, grid_x0=grid.x0, grid_y0=grid.y0,
            grid_z_top_m=grid.z_top_m,
            noise_rms_s=cfg.noise_rms_s, seed=cfg.seed,
        )
        print(f"saved {out}")
    else:
        print("N=1 smoke: 1 member, 2 GN iters, verbose")
        cfg = Config3D(n_members=1, n_gn_iters=2, verbose=True)
        members, meta = vp_ensemble_3d(ds, grid, air, cfg)
        vp = members[0][~air]
        print(f"member vp range {vp.min():.2f}..{vp.max():.2f} km/s, "
              f"final rms {meta['members_rms_s'][0]*1000:.1f} ms")


if __name__ == "__main__":
    main()
