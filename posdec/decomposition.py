"""
posdec.decomposition - the transport-invariant primitives.

These functions are the entire decomposition layer. They do not solve any
inversion; they take an ensemble of feasible velocity models and produce
the per-cell feasible interval and forced-sign classification.

Conventions:
  * Models indexed [NZ, NX]; per-cell anomaly = model - bg; units km/s.
  * Sign deadband eps (km/s) controls the forced-quiet band.
  * Classification:
        +2  forced-high          a_min > +eps
        -2  forced-low           a_max < -eps
         0  forced-quiet         interval within +/-eps
        +1  measure-dependent    interval straddles - sign not data-forced.
"""

import numpy as np


def feasible_interval(ensemble, bg):
    """Per-cell [a_min, a_max] over the ensemble's anomaly (model - bg)."""
    anom = np.array([m - bg for m in ensemble])
    return anom.min(axis=0), anom.max(axis=0)


def interval_width(a_min, a_max):
    return a_max - a_min


def classify(a_min, a_max, eps):
    """Forced-sign decomposition. See module docstring for the encoding."""
    out = np.full(a_min.shape, 1, int)
    out[a_min > eps] = 2
    out[a_max < -eps] = -2
    out[(a_min >= -eps) & (a_max <= eps)] = 0
    return out


def three_masks(cls):
    return {
        "forced_high": cls == 2,
        "forced_low": cls == -2,
        "forced_quiet": cls == 0,
        "measure_dependent": cls == 1,
    }
