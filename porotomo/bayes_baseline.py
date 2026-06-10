"""porotomo/bayes_baseline.py - Bayesian baseline for the PoroTomo 3D
comparison, by randomize-then-optimize (RTO) sampling of the linearized
Gaussian posterior.

WHY NOT MCMC HERE. The Volve baseline used emcee on the linearized forward
(45-200 dims). The 3D model has 27,234 ground cells; affine-invariant
ensemble samplers need >= 2*dim walkers and degrade far below that scale.
For a LINEAR-Gaussian model, RTO draws EXACT posterior samples (Bardsley
et al. 2014): each sample solves the damped least-squares problem with all
residual blocks perturbed by standard normal noise. So this baseline is
"the Bayesian answer under matched physics, noise, and prior":

  forward   : eikonal J at the MAP point (same C kernel, linearized once -
              same status as the Volve MCMC's linearization);
  likelihood: t ~ N(J s, sigma^2 I), sigma = the ensemble's 60 ms target;
  prior     : s ~ N(m_trend, C) with C^-1 = (lam^2 I + (r_s lam)^2 L^T L)
              / sigma^2 - the SAME smoothness class as the possibilistic
              ensemble, with lam fixed by bisecting the MAP to the 60 ms
              floor (the prior mean m_trend is the ensemble's deterministic
              trend, with no per-member randomization).

Sample i:  min || (J ds - r)/sigma - e1 ||^2
             + || (lam/sigma)(s_MAP + ds - m) - e2 ||^2
             + || (r_s lam/sigma) L (s_MAP + ds - m) - e3 ||^2,
e* ~ N(0, I). Credible intervals from the sample quantiles then face the
same two tests as the possibilistic ensemble: stage-2 holdout coverage and
published-model-inside.

Run:  uv run python -m porotomo.bayes_baseline           (~10 min)
Outputs: porotomo_bayes_cert.json, porotomo/data/bayes_samples.npz
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy.sparse import eye as sp_eye, vstack as sp_vstack
from scipy.sparse.linalg import lsqr

from porotomo.inversion3d import (prepare, forward_3d, Config3D,
                                  ground_laplacian, _smooth_random_vp_3d)
from porotomo.decompose_3d import (load_ensemble, illumination,
                                   ILLUM_MIN_PATH_M)

N_SAMPLES = 30          # matched to the possibilistic ensemble size
N_MAP_ITERS = 4
SAMPLES_NPZ = "porotomo/data/bayes_samples.npz"


def trend_model(grid, air, cfg):
    """The deterministic prior mean: the trend with no random noise."""
    rng = np.random.default_rng(0)
    cfg0 = Config3D(**{**cfg.__dict__, "noise_sd_kms": 0.0})
    vp = _smooth_random_vp_3d(rng, grid, cfg0)
    vp[air] = cfg.vp_air_kms
    return vp


def main() -> None:
    cfg = Config3D()
    picks, grid, air, ds = prepare()
    t_obs = np.concatenate(ds.times)
    ground = np.flatnonzero(~air.ravel())
    n_ground = len(ground)
    L = ground_laplacian(grid, air)
    sigma = cfg.noise_rms_s
    s_lo = 1.0 / (cfg.vp_max_kms * 1000.0)
    s_hi = 1.0 / (cfg.vp_min_kms * 1000.0)

    vp_m = trend_model(grid, air, cfg)
    m = (1.0 / (vp_m * 1000.0)).ravel()[ground]

    def to_vp(s_ground):
        vp = np.full(grid.n_cells, 1.0 / (cfg.vp_air_kms * 1000.0))
        vp[ground] = s_ground
        return 1.0 / (vp.reshape(grid.nz, grid.ny, grid.nx) * 1000.0)

    # ---- MAP by the same GN + lambda bisection as the ensemble ----------
    s = m.copy()
    lam_star = None
    for it in range(N_MAP_ITERS):
        t_pred, J = forward_3d(to_vp(s), ds, grid, compute_jacobian=True)
        resid = t_obs - t_pred
        Jg = J[:, ground]
        lo, hi = cfg.lam_lo, cfg.lam_hi
        rms_prev, s_try = None, s
        for _bi in range(cfg.bisect_iters):
            lam = np.sqrt(lo * hi)
            mu = cfg.smooth_ratio * lam
            A = sp_vstack([Jg, lam * sp_eye(n_ground, format="csr"),
                           mu * L], format="csr")
            b = np.concatenate([resid, -lam * (s - m), -mu * (L @ (s - m))])
            ds_step = lsqr(A, b, iter_lim=cfg.lsqr_iters)[0]
            s_try = np.clip(s + ds_step, s_lo, s_hi)
            t_try, _ = forward_3d(to_vp(s_try), ds, grid)
            rms_try = float(np.sqrt(np.mean((t_obs - t_try) ** 2)))
            if rms_try < sigma:
                lo = lam
            else:
                hi = lam
            lam_star = lam
            if abs(rms_try - sigma) <= (cfg.bisect_tol - 1.0) * sigma:
                break
            if rms_prev is not None and abs(rms_try - rms_prev) < 0.0005:
                break
            rms_prev = rms_try
        s = s_try
        print(f"  MAP iter {it}: rms -> {rms_try*1000:.1f} ms "
              f"(lam {lam_star:.1f})")
    s_map = s

    # ---- RTO samples around the MAP --------------------------------------
    t_map, J = forward_3d(to_vp(s_map), ds, grid, compute_jacobian=True)
    resid = t_obs - t_map
    Jg = (J[:, ground]).tocsr() * (1.0 / sigma)
    lam = lam_star
    mu = cfg.smooth_ratio * lam
    B1 = (lam / sigma) * sp_eye(n_ground, format="csr")
    B2 = (mu / sigma) * L
    A = sp_vstack([Jg, B1, B2], format="csr")
    n_pick = len(t_obs)
    rng = np.random.default_rng(cfg.seed + 7)
    samples = np.zeros((N_SAMPLES, grid.nz, grid.ny, grid.nx))
    t0 = time.time()
    for i in range(N_SAMPLES):
        e1 = rng.standard_normal(n_pick)
        e2 = rng.standard_normal(n_ground)
        e3 = rng.standard_normal(L.shape[0])
        b = np.concatenate([
            resid / sigma + e1,
            -(lam / sigma) * (s_map - m) + e2,
            -(mu / sigma) * (L @ (s_map - m)) + e3,
        ])
        ds_i = lsqr(A, b, iter_lim=cfg.lsqr_iters)[0]
        samples[i] = to_vp(np.clip(s_map + ds_i, s_lo, s_hi))
        if (i + 1) % 10 == 0:
            print(f"  sample {i+1}/{N_SAMPLES} ({time.time()-t0:.0f} s)")
    np.savez_compressed(SAMPLES_NPZ, samples=samples,
                        lam=lam, sigma=sigma)

    # ---- the same two calibration tests as the possibilistic ensemble ----
    members, air_e, grid_e, _z = load_ensemble()
    lit = (illumination(members, air, grid, ds) >= ILLUM_MIN_PATH_M) & (~air)
    q_lo = np.quantile(samples, 0.025, axis=0)
    q_hi = np.quantile(samples, 0.975, axis=0)

    # stage-2 holdout
    _p2, _g2, _a2, ds2 = prepare(stage=2)
    t_obs2 = np.concatenate(ds2.times)
    preds = np.empty((N_SAMPLES, len(t_obs2)))
    for i in range(N_SAMPLES):
        preds[i], _ = forward_3d(samples[i], ds2, grid)
    t_lo2 = np.quantile(preds, 0.025, axis=0)
    t_hi2 = np.quantile(preds, 0.975, axis=0)
    t_med2 = np.median(preds, axis=0)
    inside_raw = float(((t_obs2 >= t_lo2) & (t_obs2 <= t_hi2)).mean())
    inside_n = float(((t_obs2 >= t_lo2 - 0.036)
                      & (t_obs2 <= t_hi2 + 0.036)).mean())
    rms2 = float(np.sqrt(np.mean((t_obs2 - t_med2) ** 2)))

    cert = {
        "method": "RTO exact sampling of the linearized-Gaussian posterior "
                  "(matched physics/noise/prior class)",
        "n_samples": N_SAMPLES,
        "lam": float(lam), "sigma_s": sigma,
        "holdout": {
            "inside_raw_95cred": round(inside_raw, 4),
            "inside_with_pick_noise": round(inside_n, 4),
            "rms_median_prediction_ms": round(rms2 * 1000, 1),
        },
        "median_cred_width_kms_lit": round(
            float(np.median((q_hi - q_lo)[lit])), 3),
    }
    with open("porotomo_bayes_cert.json", "w") as fh:
        json.dump(cert, fh, indent=2)
    print(json.dumps(cert, indent=2))


if __name__ == "__main__":
    main()
