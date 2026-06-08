"""
volve/smoke.py - N=1 round-trip of the Volve ingestion pipeline on
synthetic data.

Builds a tiny synthetic SEG-Y file and a tiny synthetic LAS file in the
shape of (a scaled-down) Volve walkaway VSP + tie well, runs them through
the loaders, and verifies output shapes + units. The point is not to
validate the inversion - just to confirm the tooling works before the
real ~5 GB SEG-Y arrives.

Run:  uv run python -m volve.smoke
"""

from pathlib import Path
import tempfile

import numpy as np
import segyio
import lasio

from volve import geometry as G
from volve import load_vsp as V
from volve import load_logs as L


def _write_synthetic_segy(path: str,
                          n_shots: int = 4,
                          n_recv: int = 6,
                          n_comp: int = 4,
                          n_samples: int = 200,
                          dt_us: int = 8000) -> None:
    n_traces = n_shots * n_recv * n_comp
    spec = segyio.spec()
    spec.sorting = 1
    spec.format = 1
    spec.samples = [i * (dt_us / 1000.0) for i in range(n_samples)]  # ms
    spec.tracecount = n_traces
    rng = np.random.default_rng(0)
    with segyio.create(path, spec) as f:
        i = 0
        for s in range(n_shots):
            sx_m = 3700.0 + s * (15000.0 / max(1, n_shots - 1))
            sy_m = 0.0
            for r in range(n_recv):
                gx_m = 10000.0
                gz_m = 1000.0 + r * ((7990.0 - 1000.0) / max(1, n_recv - 1))
                for c in range(n_comp):
                    f.trace[i] = (
                        rng.standard_normal(n_samples).astype(np.float32))
                    f.header[i] = {
                        segyio.TraceField.SourceX: int(sx_m),
                        segyio.TraceField.SourceY: int(sy_m),
                        segyio.TraceField.GroupX:  int(gx_m),
                        segyio.TraceField.GroupY:  int(gz_m),
                        segyio.TraceField.SourceGroupScalar: 1,
                    }
                    i += 1
        f.bin = {segyio.BinField.Samples: n_samples,
                 segyio.BinField.Interval: dt_us}


def _write_synthetic_las(path: str) -> None:
    las = lasio.LASFile()
    las.well["WELL"].value = "15/9-F-1A-smoke"
    depth = np.linspace(0.0, 4000.0, 401)
    # plausible Vp profile increasing with depth: 1.5 -> 4.5 km/s
    vp = 1.5 + 3.0 * (depth / depth.max())
    # DT in us/ft from Vp(km/s) = 304.8 / DT
    dt = 304.8 / vp
    rho = 1.95 + 0.75 * (depth / depth.max())     # 1.95 -> 2.70 g/cm3
    las.append_curve("DEPT", depth, unit="m", descr="Depth")
    las.append_curve("DT",   dt,    unit="us/ft", descr="P-wave sonic")
    las.append_curve("RHOB", rho,   unit="g/cm3", descr="Bulk density")
    las.write(path, version=2.0)


def main() -> None:
    print("volve.smoke - N=1 round-trip of the ingestion pipeline")
    print("=" * 64)
    print("Geometry deck: see `python -m volve.geometry <real-SEG-Y>` for")
    print("the authoritative geometry. This smoke only exercises the LAS")
    print("and SEG-Y loaders on synthetic data with a placeholder layout.")

    with tempfile.TemporaryDirectory() as tmp:
        sgy = str(Path(tmp) / "_volve_smoke.sgy")
        las = str(Path(tmp) / "_volve_smoke.las")
        _write_synthetic_segy(sgy, n_shots=4, n_recv=6, n_comp=4,
                              n_samples=200, dt_us=8000)
        _write_synthetic_las(las)

        print("\nSEG-Y loader:")
        blk = V.load_file(sgy)
        print(V.summarize_block(blk))
        arr, hdr = V.bin_to_shot_recv_tensor(blk, n_components=4)
        assert arr.shape == (4, 6, 4, 200), \
            f"shape mismatch: {arr.shape}"
        assert hdr["dt_s"] == 0.008, hdr["dt_s"]
        print(f"  tensor shape: {arr.shape} (n_shots, n_recv, n_comp, n_samp)")
        print(f"  dt_s={hdr['dt_s']}  t_axis[-1]={hdr['t_axis'][-1]:.3f} s")
        assert hdr["receivers"].shape == (6, 2)
        assert hdr["shots"].shape == (4, 2)

        print("\nLAS loader:")
        log = L.load_well_log(las)
        print(L.summary(log))
        assert log.vp_kms.min() > 1.0 and log.vp_kms.max() < 6.0, \
            f"Vp out of range: {log.vp_kms.min()}-{log.vp_kms.max()}"
        assert log.rho_gcc is not None
        assert log.vs_kms is None  # smoke LAS has no DTS

    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
