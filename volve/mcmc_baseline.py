"""
volve/mcmc_baseline.py - MCMC comparator on F-15A 1D Vp(z).

Same forward operator, same prior, same data as phase 4 / posdec
(volve.inversion_eikonal). Bayesian credible intervals via emcee
(affine-invariant ensemble MCMC). Apples-to-apples comparator for the
three-way bake-off in volve/threeway.py.

DESIGN.
  - Parameterize slowness s(z) in km/s^{-1} on the same 45 depth-bin grid
    used by phase 4.
  - Forward: ONE eikonal solve from a reference Vp_0(z) (the phase-4
    ensemble median), then linearize:
        t(s) ~ t_0 + G_eik (s - s_0)
    where G_eik is the eikonal Jacobian at the reference. The MCMC
    runs on this linearized forward so each likelihood eval is a
    matrix-vector product (microseconds), not a full FMM solve. The
    linearization is honest for small deviations from the reference;
    deviations are clipped to the Vp envelope so the linearization
    stays valid.
  - Log-likelihood: Gaussian with sigma = 25 ms (the modeling-error
    floor we measured in phase 3).
  - Log-prior: smooth Gaussian on log-slowness with correlation
    250 m (matches the phase 4 prior); envelope-clipped at
    Vp_min=1.5, Vp_max=5.5 km/s via -inf if out of range.
  - Sampler: 100 walkers, 3500 steps, 1000 burn-in.

Output:
  - per-bin posterior credible intervals (50%, 90%, 95%)
  - JSON certificate with calibration vs DT-EDIT sonic
  - npz snapshot of samples for the threeway figure

Run:  uv run python -m volve.mcmc_baseline
"""

from dataclasses import dataclass
from pathlib import Path
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import emcee
import lasio

from volve.inversion_1d import (
    PickData, load_picks, depth_grid_for_picks, depth_centers,
)
from volve.inversion_eikonal import (
    EikonalConfig, vp_ensemble_eikonal,
    forward_eikonal_1d, grid_dimensions, pick_grid_coords,
)


OUT_CERT = "volve/picks/mcmc_certificate.json"
OUT_NPZ = "volve/picks/mcmc_samples.npz"
OUT_FIG = "volve_mcmc_baseline.png"

# Phase-4-matched prior + likelihood settings
NOISE_SIGMA_S = 0.025
PRIOR_VP_MIN_KMS = 1.5
PRIOR_VP_MAX_KMS = 5.5
PRIOR_SMOOTH_CORR_M = 250.0

# MCMC settings
N_WALKERS = 100
N_STEPS = 3500
N_BURNIN = 1000


# --- linearized forward at reference --------------------------------------

def _build_linearized_forward(picks: PickData, grid):
    """Run phase-4 ensemble to get a sensible reference Vp_0(z); then
    compute eikonal forward + Jacobian at Vp_0 once. The linearized
    forward used by MCMC is t(s) ~ t_0 + J_eik (s - s_0)."""
    print("  building linearized forward at phase-4 reference...")
    cfg = EikonalConfig(n_members=12, n_gn_iters=3)   # mid-size ensemble for stable reference
    members, meta = vp_ensemble_eikonal(picks, grid, cfg)
    vp_0_kms = np.median(members, axis=0)
    s_0 = 1.0 / (vp_0_kms * 1000.0)
    nz, nx, cell_m = grid_dimensions(picks, cell_m=cfg.cell_m)
    ix_recv, iz_recv = pick_grid_coords(picks, cell_m)
    t_0, J = forward_eikonal_1d(
        vp_0_kms, picks, nz, nx, cell_m,
        ix_recv=ix_recv, iz_recv=iz_recv, compute_jacobian=True)
    return {
        "vp_0_kms": vp_0_kms,
        "s_0": s_0,
        "t_0": t_0,
        "J": J,                    # d t / d s, shape (n_picks, n_bins)
    }


# --- log-posterior --------------------------------------------------------

def _make_log_posterior(picks, grid, lin):
    z_centers = depth_centers(grid)
    bin_thick = float(grid[1] - grid[0])
    sigma_obs = NOISE_SIGMA_S
    d_obs = picks.times_s
    t_0 = lin["t_0"]
    J = lin["J"]
    s_0 = lin["s_0"]

    s_min = 1.0 / (PRIOR_VP_MAX_KMS * 1000.0)
    s_max = 1.0 / (PRIOR_VP_MIN_KMS * 1000.0)

    # Smoothness covariance prior on slowness (Gaussian process style)
    # Approximate with diagonal + lambda * D^T D where D is finite-difference
    n_bins = z_centers.size
    D = np.zeros((n_bins - 1, n_bins), dtype=float)
    for i in range(n_bins - 1):
        D[i, i] = -1.0
        D[i, i + 1] = +1.0
    # Smoothness penalty strength: corr_m / bin_thick -> bins per corr length
    smooth_lambda_inv = max(1.0, PRIOR_SMOOTH_CORR_M / bin_thick)
    # Reference slowness for the smoothness prior is the linearization point.

    def log_prior(s_kms_inv):
        if np.any(s_kms_inv < s_min) or np.any(s_kms_inv > s_max):
            return -np.inf
        # Smoothness penalty: -0.5 * lambda * ||D log(s)||^2
        log_s = np.log(s_kms_inv)
        diff = D @ log_s
        return -0.5 * smooth_lambda_inv * np.sum(diff ** 2)

    def log_likelihood(s_kms_inv):
        ds = s_kms_inv - s_0
        t_pred = t_0 + J @ ds
        resid = d_obs - t_pred
        return -0.5 * np.sum((resid / sigma_obs) ** 2)

    def log_posterior(s_kms_inv):
        lp = log_prior(s_kms_inv)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(s_kms_inv)

    return log_posterior


# --- sonic loader ---------------------------------------------------------

def _load_sonic_f15a():
    las = lasio.read("volve/data/15_9-F-15 A/TZV_DEPTH_MD_COMPUTED_1.LAS")
    tvd = np.asarray(las["TVD"], dtype=float)
    dt = np.asarray(las["DT-EDIT"], dtype=float)
    m = np.isfinite(tvd) & np.isfinite(dt) & (dt > 0) & (dt < 500)
    return tvd[m], 304.8 / dt[m]


# --- main -----------------------------------------------------------------

def main():
    t0 = time.time()
    picks = load_picks("volve/picks/picks_z.csv")
    grid = depth_grid_for_picks(picks)
    print(f"picks: {picks.n()} ok")
    print(f"grid: {grid.size - 1} bins of "
          f"{grid[1] - grid[0]:.1f} m to {grid[-1]:.0f} m")

    lin = _build_linearized_forward(picks, grid)
    log_post = _make_log_posterior(picks, grid, lin)
    n_bins = lin["s_0"].size
    print(f"linearization at reference Vp(z) RMS = "
          f"{float(np.sqrt(np.mean((picks.times_s - lin['t_0'])**2)))*1000:.1f} ms")

    # Initialize walkers near the linearization point
    rng = np.random.default_rng(20260608)
    p0 = lin["s_0"][None, :].repeat(N_WALKERS, axis=0)
    p0 = p0 * (1.0 + 0.02 * rng.standard_normal(p0.shape))
    p0 = np.clip(p0, 1.0 / (PRIOR_VP_MAX_KMS * 1000.0),
                       1.0 / (PRIOR_VP_MIN_KMS * 1000.0))

    print(f"running emcee: {N_WALKERS} walkers x {N_STEPS} steps "
          f"(burn-in {N_BURNIN})...")
    sampler = emcee.EnsembleSampler(N_WALKERS, n_bins, log_post)
    sampler.run_mcmc(p0, N_STEPS, progress=False)
    samples_s = sampler.get_chain(discard=N_BURNIN, flat=True)
    samples_vp = 1.0 / (samples_s * 1000.0)
    elapsed = time.time() - t0
    print(f"sampler done in {elapsed:.0f} s")
    print(f"posterior samples: {samples_vp.shape[0]} x {n_bins}")
    print(f"acceptance fraction (mean): "
          f"{sampler.acceptance_fraction.mean():.2f}")

    # Per-bin marginals
    pct = {
        "p025": np.percentile(samples_vp, 2.5, axis=0),
        "p05":  np.percentile(samples_vp, 5.0, axis=0),
        "p25":  np.percentile(samples_vp, 25.0, axis=0),
        "p50":  np.percentile(samples_vp, 50.0, axis=0),
        "p75":  np.percentile(samples_vp, 75.0, axis=0),
        "p95":  np.percentile(samples_vp, 95.0, axis=0),
        "p975": np.percentile(samples_vp, 97.5, axis=0),
    }

    # Sonic calibration: does the 90% / 95% credible interval cover sonic?
    sonic_tvd, sonic_vp = _load_sonic_f15a()
    z_centers = depth_centers(grid)
    half = 0.5 * float(grid[1] - grid[0])
    sonic_bin_mean = np.full(n_bins, np.nan)
    for j in range(n_bins):
        sel = (sonic_tvd >= z_centers[j] - half) & \
              (sonic_tvd < z_centers[j] + half)
        if sel.any():
            sonic_bin_mean[j] = float(np.mean(sonic_vp[sel]))

    ok_bins = np.isfinite(sonic_bin_mean)
    in90 = (sonic_bin_mean >= pct["p05"]) & (sonic_bin_mean <= pct["p95"])
    in95 = (sonic_bin_mean >= pct["p025"]) & (sonic_bin_mean <= pct["p975"])
    n_ok = int(ok_bins.sum())
    n_in90 = int(in90[ok_bins].sum())
    n_in95 = int(in95[ok_bins].sum())
    print()
    print("MCMC sonic calibration vs DT-EDIT:")
    print(f"  sonic-bin coverage : {n_ok}/{n_bins}")
    print(f"  inside 90% CI      : {n_in90}/{n_ok} "
          f"({100 * n_in90 / n_ok:.1f}%)")
    print(f"  inside 95% CI      : {n_in95}/{n_ok} "
          f"({100 * n_in95 / n_ok:.1f}%)")

    # Figure
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.fill_betweenx(z_centers, pct["p025"], pct["p975"],
                     color="#88aaff", alpha=0.35, label="95% credible")
    ax.fill_betweenx(z_centers, pct["p05"], pct["p95"],
                     color="#2166ac", alpha=0.45, label="90% credible")
    ax.plot(pct["p50"], z_centers, color="#08306b", lw=1.6,
            label="MCMC posterior median")
    ax.plot(sonic_vp, sonic_tvd, color="#b2182b", lw=0.5,
            label="DT-EDIT sonic")
    ax.invert_yaxis()
    ax.set_xlabel("Vp (km/s)")
    ax.set_ylabel("depth below sea surface (m)")
    ax.set_title(f"MCMC baseline (emcee, linearized forward) - "
                 f"F-15A\n90% CI sonic-inside = "
                 f"{100 * n_in90 / n_ok:.1f}%, "
                 f"95% CI sonic-inside = {100 * n_in95 / n_ok:.1f}%",
                 fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xlim(1.3, 5.8)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130)
    plt.close(fig)
    print(f"figure: {OUT_FIG}")

    # Snapshot
    np.savez_compressed(
        OUT_NPZ,
        samples_vp_kms=samples_vp.astype(np.float32),
        z_centers_m=z_centers,
        bin_thick_m=np.array([grid[1] - grid[0]]),
        p025=pct["p025"], p05=pct["p05"], p25=pct["p25"],
        p50=pct["p50"], p75=pct["p75"], p95=pct["p95"], p975=pct["p975"],
        vp_0_kms=lin["vp_0_kms"],
    )
    print(f"samples: {OUT_NPZ}")

    cert = {
        "label": "volve_f15a_mcmc_emcee",
        "method": "emcee EnsembleSampler on linearized 1D Vp(z)",
        "settings": {
            "n_walkers": N_WALKERS,
            "n_steps": N_STEPS,
            "n_burnin": N_BURNIN,
            "n_bins": int(n_bins),
            "noise_sigma_s": float(NOISE_SIGMA_S),
            "vp_envelope_kms":
                [float(PRIOR_VP_MIN_KMS), float(PRIOR_VP_MAX_KMS)],
            "prior_smooth_corr_m": float(PRIOR_SMOOTH_CORR_M),
        },
        "samples_size":
            f"{samples_vp.shape[0]} x {samples_vp.shape[1]}",
        "acceptance_fraction_mean":
            float(sampler.acceptance_fraction.mean()),
        "sonic_calibration": {
            "sonic_bin_coverage": f"{n_ok}/{n_bins}",
            "inside_90pct_ci": int(n_in90),
            "inside_95pct_ci": int(n_in95),
            "inside_90pct_pct": float(100 * n_in90 / n_ok),
            "inside_95pct_pct": float(100 * n_in95 / n_ok),
        },
        "elapsed_seconds": float(elapsed),
    }
    Path(OUT_CERT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_CERT).write_text(json.dumps(cert, indent=2))
    print(f"certificate: {OUT_CERT}")


if __name__ == "__main__":
    main()
