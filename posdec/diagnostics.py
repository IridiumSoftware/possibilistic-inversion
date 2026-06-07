"""
posdec.diagnostics - ensemble diagnostics (ORSI propagation #4).

Per-ensemble measurements that go with the decomposition:
  * pairwise model distances (RMS over cells, km/s);
  * sampled feasible-set diameter (max pairwise distance);
  * smoothness statistics (RMS gradient magnitude, per-member);
  * explored-direction extents (SVD of centered ensemble; max-min projection
    along leading directions, RMS-normalized over cells).

These say something about what the sampler EXPLORED. They say nothing about
basins the sampler missed; RWC-2 covers that gap.
"""

import numpy as np


def pairwise_distances(ensemble):
    """Symmetric matrix of RMS pairwise model distances (km/s)."""
    arr = np.stack([m.ravel() for m in ensemble])
    diff = arr[:, None, :] - arr[None, :, :]
    return np.sqrt((diff ** 2).mean(axis=-1))


def feasible_set_diameter(ensemble):
    """Max pairwise RMS distance - a proxy for the diameter of the SAMPLED
    feasible set. Says nothing about basins the sampler missed."""
    D = pairwise_distances(ensemble)
    return float(D.max())


def smoothness(model):
    """RMS gradient magnitude. Larger => rougher. Units: km/s per grid step."""
    gz, gx = np.gradient(model)
    return float(np.sqrt((gz ** 2 + gx ** 2).mean()))


def smoothness_stats(ensemble):
    vals = np.array([smoothness(m) for m in ensemble])
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def explored_directions(ensemble):
    """SVD of the centered ensemble matrix. Returns the singular spectrum,
    explained-variance ratios, and the max-min extent of the projections
    onto the top explored directions (RMS-normalized over cells, so
    directly comparable to feasible_set_diameter)."""
    arr = np.stack([m.ravel() for m in ensemble])
    centered = arr - arr.mean(axis=0, keepdims=True)
    U, s, _ = np.linalg.svd(centered, full_matrices=False)
    total = float((s ** 2).sum())
    evr = (s ** 2 / total).tolist() if total > 0 else [0.0] * len(s)
    proj = U * s
    D_cells = arr.shape[1]
    extents = ((proj.max(axis=0) - proj.min(axis=0)) /
               np.sqrt(D_cells)).tolist() if D_cells > 0 else []
    return {
        "singular_values": s.tolist(),
        "explained_variance_ratio": evr,
        "principal_diameter_kms_rms":
            float(extents[0]) if extents else 0.0,
        "top_k_diameters_kms_rms": extents[:5],
    }
