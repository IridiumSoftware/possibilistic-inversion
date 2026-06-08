"""
volve - real-data integration for the possibilistic-inversion library on
the Equinor Volve walkaway VSP dataset (2018 open release).

Submodules:
  - geometry  : the known survey geometry (151 shots, 467 receivers, ...)
  - load_vsp  : SEG-Y reader for the walkaway shot gathers
  - load_logs : LAS reader for the well-log Vp/Vs/rho ground truth
  - smoke     : N=1 round-trip of synthetic-SEG-Y + synthetic-LAS through
                the loaders, run before real data arrives to verify the
                pipeline.

Phase 1 of the Volve workflow (see volve/README.md). Phase 2 (first-arrival
picking) and Phase 3 (possibilistic decomposition on the picked arrivals
+ validation vs sonic log) follow once the real SEG-Y + LAS arrive.
"""

from volve import geometry, load_vsp, load_logs

__all__ = ["geometry", "load_vsp", "load_logs"]
