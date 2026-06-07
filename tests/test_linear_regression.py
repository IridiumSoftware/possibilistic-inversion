"""
tests/test_linear_regression.py - linear-case regression test for posdec.

ORSI propagation #5: the standalone decomposition library ships with the
exact Rational{BigInt} Julia implementation as a regression target. This
test runs the same fixed ensemble through:

  1. posdec.classify in Float64                                  - Python
  2. hardcoded expected labels (hand-computed from exact rationals) - reference
  3. decomposition_exact.jl in Rational{BigInt}                   - Julia, optional

Asserts agreement on the per-cell forced-sign labels. The hand-computed
reference is what the test trusts; Julia is checked when available so
the cross-language path is exercised when a Julia runtime is on PATH.

The reference ensemble is the same hardcoded rational ensemble from
decomposition_exact.jl's `demo()` (eps = 1/5):

  M1: [ 3/4, -1/2,  1/10, -3/5,  0    ]
  M2: [ 7/8, -2/5, -1/4,  -7/10, 1/20 ]
  M3: [ 2/3, -3/4,  2/5,  -1/2,  -1/15]
  M4: [ 9/10,-1/3, -1/3,  -4/5,  1/12 ]

Per-cell expected labels (hand-derived from exact rationals + eps = 1/5):
  cell 1: a_min = 2/3 > 1/5                       -> forced-high       (+2)
  cell 2: a_max = -1/3 < -1/5                     -> forced-low        (-2)
  cell 3: a_min = -1/3 < -1/5, a_max = 2/5 > 1/5  -> measure-dependent (+1)
  cell 4: a_max = -1/2 < -1/5                     -> forced-low        (-2)
  cell 5: |a_min|, |a_max| both <= 1/5            -> forced-quiet      ( 0)

Run:  uv run python tests/test_linear_regression.py
"""

import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

# Re-root sys.path so we can import posdec when this file is invoked from
# anywhere (uv run python tests/..., pytest, manual python tests/...).
HERE = Path(__file__).resolve()
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

import posdec  # noqa: E402

# --- Reference ensemble (same as Julia demo) --------------------------------

ENSEMBLE_FRACS = [
    [Fraction(3, 4),  Fraction(-1, 2), Fraction(1, 10),  Fraction(-3, 5), Fraction(0)],
    [Fraction(7, 8),  Fraction(-2, 5), Fraction(-1, 4),  Fraction(-7, 10), Fraction(1, 20)],
    [Fraction(2, 3),  Fraction(-3, 4), Fraction(2, 5),   Fraction(-1, 2), Fraction(-1, 15)],
    [Fraction(9, 10), Fraction(-1, 3), Fraction(-1, 3),  Fraction(-4, 5), Fraction(1, 12)],
]
EPS_FRAC = Fraction(1, 5)
EXPECTED_LABELS = np.array([2, -2, 1, -2, 0])

# Label codes from posdec.classify:
#   +2 forced-high   -2 forced-low   0 forced-quiet   +1 measure-dependent
LABEL_NAME = {
    2: "ForcedHigh",
    -2: "ForcedLow",
    0: "ForcedQuiet",
    1: "MeasureDependent",
}


def _ensemble_as_float():
    """The hand-built rational ensemble, cast to Float64 1-D 'cells' to fit
    posdec's NZ x NX convention (here NZ=1, NX=5)."""
    rows = np.array(
        [[float(v) for v in row] for row in ENSEMBLE_FRACS],
        dtype=float,
    )
    members = [rows[k:k+1, :] for k in range(rows.shape[0])]
    bg = np.zeros_like(members[0])
    return members, bg


def check_posdec():
    members, bg = _ensemble_as_float()
    a_min, a_max = posdec.feasible_interval(members, bg)
    cls = posdec.classify(a_min, a_max, eps=float(EPS_FRAC))
    actual = cls.ravel()
    ok = bool(np.array_equal(actual, EXPECTED_LABELS))
    print("posdec (Float64) labels:    ", actual.tolist())
    print("expected (hand-computed):   ", EXPECTED_LABELS.tolist())
    print("posdec vs expected          :", "PASS" if ok else "FAIL")
    return ok


def _parse_julia_labels(stdout_text):
    """Extract `cell N: ... -> Label` lines from julia decomposition_exact.jl
    demo output, return list of label-code ints in cell order."""
    name_to_code = {v: k for k, v in LABEL_NAME.items()}
    labels = []
    for line in stdout_text.splitlines():
        if "->" not in line or "cell" not in line:
            continue
        tail = line.split("->", 1)[1].strip()
        token = tail.split()[0].strip()
        if token not in name_to_code:
            continue
        labels.append(name_to_code[token])
    return labels


def check_julia():
    julia = shutil.which("julia")
    if not julia:
        print("julia not on PATH - skipping cross-language check")
        return None
    script = REPO / "decomposition_exact.jl"
    proc = subprocess.run(
        [julia, "--project=.", str(script)],
        check=False, capture_output=True, text=True,
        cwd=str(REPO),
    )
    if proc.returncode != 0:
        print("julia run FAILED:\n", proc.stderr[-1500:])
        return False
    julia_labels = _parse_julia_labels(proc.stdout)
    ok = (julia_labels == EXPECTED_LABELS.tolist())
    print("julia (Rational{BigInt}):   ", julia_labels)
    print("julia vs expected           :", "PASS" if ok else "FAIL")
    return ok


def main():
    print("posdec linear-case regression test")
    print("=" * 60)
    ok_py = check_posdec()
    print()
    ok_jl = check_julia()
    print("\nSummary")
    print(f"  posdec (Python) : {'PASS' if ok_py else 'FAIL'}")
    if ok_jl is None:
        print( "  julia (exact)   : SKIPPED (julia not on PATH)")
        passing = ok_py
    else:
        print(f"  julia (exact)   : {'PASS' if ok_jl else 'FAIL'}")
        passing = ok_py and ok_jl
    sys.exit(0 if passing else 1)


if __name__ == "__main__":
    main()
