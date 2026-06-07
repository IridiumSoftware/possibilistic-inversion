"""
posdec.cli - command-line entry to the decomposition library.

Subcommands:
  decompose   - ingest an ensemble + bg + eps, emit certificate + report
  smoke       - run a tiny self-test (used for CI / N=1 verification)

Examples:
  python -m posdec decompose ensemble.npy bg.npy --eps 0.04 \\
      --out-dir out/ --label "my-run" \\
      --rwc1 rwc1_coverage_curve.json --rwc2 rwc2_certificate.json
  python -m posdec smoke

Ensemble file (.npy):
  shape (K, NZ, NX) - K feasible velocity models, identical grid.
Background file (.npy):
  shape (NZ, NX) - the reference background velocity field.

Both files are loaded with np.load. The user is responsible for asserting
ensemble members are feasible (each fits the data to the noise level);
posdec computes the decomposition unconditionally and the caller's
admissibility filter must run upstream.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from posdec.decomposition import feasible_interval
from posdec.certificate import (
    coverage_certificate,
    write_certificate,
    read_json_if_present,
)
from posdec.report import plot_three_maps_and_width


def _cmd_decompose(args):
    ensemble = np.load(args.ensemble)
    bg = np.load(args.bg)
    if ensemble.ndim != 3 or ensemble.shape[1:] != bg.shape:
        sys.exit(f"shape mismatch: ensemble {ensemble.shape}, "
                 f"bg {bg.shape}; expected ensemble (K, NZ, NX) "
                 f"matching bg (NZ, NX).")
    members = [ensemble[k] for k in range(ensemble.shape[0])]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rwc1 = read_json_if_present(args.rwc1) if args.rwc1 else None
    rwc2 = read_json_if_present(args.rwc2) if args.rwc2 else None
    cert = coverage_certificate(
        members, bg, eps=args.eps,
        coverage_curve=rwc1,
        false_forced_rate=(rwc2 or {}).get("false_forced_rate"),
        label=args.label,
    )
    cert_path = out / "certificate.json"
    write_certificate(cert, cert_path)
    a_min, a_max = feasible_interval(members, bg)
    fig_path = out / "report.png"
    plot_three_maps_and_width(
        a_min, a_max, eps=args.eps,
        coverage_curve=rwc1,
        false_forced_rate=(rwc2 or {}).get("false_forced_rate"),
        out_path=fig_path,
        title=args.label or "posdec - standard report",
    )
    print(f"certificate: {cert_path}")
    print(f"report:      {fig_path}")
    print(f"  ensemble size           : {cert['ensemble_size']}")
    print(f"  forced-high cells       : "
          f"{cert['decomposition']['forced_high_cells']}")
    print(f"  forced-low cells        : "
          f"{cert['decomposition']['forced_low_cells']}")
    print(f"  measure-dependent cells : "
          f"{cert['decomposition']['measure_dependent_cells']}")
    print(f"  RWC-1 stabilized at N   : "
          f"{cert['coverage']['rwc1_stabilized']}")
    print(f"  RWC-2 false-forced rate : "
          f"{cert['coverage']['rwc2_false_forced_rate']}")
    print(f"  RWC-2 status            : {cert['coverage']['rwc2_status']}")


def _cmd_smoke(args):
    """Build a toy ensemble, run the full pipeline, verify outputs land."""
    rng = np.random.default_rng(0)
    NZ, NX = 8, 10
    bg = np.full((NZ, NX), 5.0)
    members = []
    for _ in range(12):
        m = bg + 0.5 * rng.standard_normal((NZ, NX))
        m[1:3, 1:4] += 1.0
        m[5:7, 6:9] -= 1.0
        members.append(m)
    cert = coverage_certificate(
        members, bg, eps=0.1,
        coverage_curve={"Ns": [3, 6, 9, 12],
                        "forced_sizes": [5, 8, 9, 9],
                        "false_forced_res": [1, 0, 0, 0],
                        "stabilization_N": 9},
        false_forced_rate=0.04,
        label="posdec smoke",
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_certificate(cert, out / "smoke_certificate.json")
    a_min, a_max = feasible_interval(members, bg)
    plot_three_maps_and_width(
        a_min, a_max, eps=0.1,
        coverage_curve={"Ns": [3, 6, 9, 12],
                        "forced_sizes": [5, 8, 9, 9],
                        "false_forced_res": [1, 0, 0, 0],
                        "stabilization_N": 9},
        false_forced_rate=0.04,
        out_path=out / "smoke_report.png",
        title="posdec smoke",
    )
    print("SMOKE OK")
    print(f"  forced-high cells       : "
          f"{cert['decomposition']['forced_high_cells']}")
    print(f"  rwc1_stabilized         : "
          f"{cert['coverage']['rwc1_stabilized']}")
    print(f"  rwc2_status             : {cert['coverage']['rwc2_status']}")
    print(f"  artifacts in            : {out}")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="posdec",
        description="Possibilistic decomposition - standalone library CLI.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("decompose",
                       help="Decompose an ensemble + bg into a certificate + "
                            "standard report.")
    d.add_argument("ensemble", help="ensemble .npy, shape (K, NZ, NX).")
    d.add_argument("bg", help="background .npy, shape (NZ, NX).")
    d.add_argument("--eps", type=float, required=True,
                   help="sign deadband, km/s (e.g. 0.04).")
    d.add_argument("--out-dir", default="posdec_out",
                   help="directory for certificate + report (created).")
    d.add_argument("--label", default=None,
                   help="run label written into the certificate.")
    d.add_argument("--rwc1", default=None,
                   help="optional rwc1_coverage_curve.json.")
    d.add_argument("--rwc2", default=None,
                   help="optional rwc2_certificate.json.")
    d.set_defaults(func=_cmd_decompose)

    s = sub.add_parser("smoke", help="run the package self-test.")
    s.add_argument("--out-dir", default="posdec_smoke",
                   help="directory for smoke artifacts.")
    s.set_defaults(func=_cmd_smoke)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
