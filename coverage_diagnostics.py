"""
coverage_diagnostics.py - BACK-COMPAT SHIM. Use `import posdec` instead.

The standalone decomposition library now lives at `posdec/` (ORSI #5);
this module re-exports the public API for any caller still importing
coverage_diagnostics. New code should import posdec directly:

    from posdec import (
        coverage_certificate, write_certificate,
        plot_three_maps_and_width, read_json_if_present,
        feasible_interval, classify,
    )
"""

from posdec import (
    feasible_interval,
    interval_width,
    classify,
    three_masks,
    pairwise_distances,
    feasible_set_diameter,
    smoothness,
    smoothness_stats,
    explored_directions,
    coverage_certificate,
    write_certificate,
    read_json_if_present,
    plot_three_maps_and_width,
)

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
