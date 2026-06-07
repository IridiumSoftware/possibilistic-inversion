"""
posdec - the possibilistic-decomposition layer as a standalone library.

The decomposition is forward-operator-agnostic. Given any ensemble of
velocity models the caller asserts are feasible (each fits the data to
the noise level), posdec produces:

  * the per-cell feasible interval and forced-sign decomposition;
  * ensemble diagnostics (sampled-set diameter, principal-direction extent,
    smoothness statistics);
  * the coverage certificate (decomposition counts + diagnostics + RWC-1
    coverage curve + RWC-2 false-forced rate);
  * the standard reporting figure (3 masks + interval-width + coverage
    curve).

This package depends only on numpy + matplotlib + stdlib. It contains no
inversion code; the sampler is the caller's responsibility. The decomposition
layer's transport invariance under different forward operators is what
makes this clean separation possible.

ORSI propagation #5 (modular library) is satisfied here. The package is
the same code as the earlier coverage_diagnostics.py, refactored into a
package and given a CLI; coverage_diagnostics.py remains as a back-compat
shim.

CLI: `python -m posdec decompose ensemble.npy bg.npy --eps 0.04 --out-dir out/`
"""

from posdec.decomposition import (
    feasible_interval,
    interval_width,
    classify,
    three_masks,
)
from posdec.diagnostics import (
    pairwise_distances,
    feasible_set_diameter,
    smoothness,
    smoothness_stats,
    explored_directions,
)
from posdec.certificate import (
    coverage_certificate,
    write_certificate,
    read_json_if_present,
)
from posdec.report import plot_three_maps_and_width

__all__ = [
    "feasible_interval",
    "interval_width",
    "classify",
    "three_masks",
    "pairwise_distances",
    "feasible_set_diameter",
    "smoothness",
    "smoothness_stats",
    "explored_directions",
    "coverage_certificate",
    "write_certificate",
    "read_json_if_present",
    "plot_three_maps_and_width",
]

__version__ = "0.1.0"
