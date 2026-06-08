"""
volve/load_logs.py - LAS loader for the Volve well logs.

Extracts the ground-truth Vp profile we will validate the possibilistic
inversion against. The walkaway tie well is 15/9-F-1A (or -F-1B; both have
DT and DTS coverage per the recon).

INPUT:
  - A LAS 2.0 file from Volve's `Well_logs/` tree (e.g. `15_9-F-1A.LAS`).

OUTPUT (`load_well_log`):
  - WellLog dataclass with:
      well_name     : str
      depth_m       : ndarray (n_depth,)
      vp_kms        : ndarray (n_depth,) - P-wave velocity in km/s
      vs_kms        : ndarray or None    - shear velocity in km/s (if DTS)
      rho_gcc       : ndarray or None    - density in g/cm^3 (if RHOB)

DT (compressional slowness) is the canonical curve; LAS DT is conventionally
in microseconds per foot. Vp (km/s) = 304.8 / DT(us/ft). The other curves
are optional.

A depth grid filter is applied: NaN-valued samples (logger gaps, casing
zones) are dropped from the returned arrays; the depth axis is the depths
that REMAIN after that filter.

CONVENTIONS:
  - Depth in metres (LAS DEPT is in metres for Volve releases - assert).
  - Vp in km/s for direct comparison to the inversion output.

API stability: STABLE.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import numpy as np

try:
    import lasio
except ImportError as e:
    raise ImportError(
        "lasio is required; add it to pyproject.toml dependencies "
        "and run `uv sync`."
    ) from e


@dataclass
class WellLog:
    well_name: str
    depth_m: np.ndarray
    vp_kms: np.ndarray
    vs_kms: Optional[np.ndarray] = None
    rho_gcc: Optional[np.ndarray] = None
    source_curves: dict = None    # original LAS curves found, for trace-back


def _dt_to_vp_kms(dt: np.ndarray, unit: str) -> np.ndarray:
    """Convert sonic slowness to P-wave velocity in km/s.
        DT us/ft -> Vp = 304.8 / DT km/s
        DT us/m  -> Vp = 1000 / DT km/s
    """
    u = unit.lower().strip()
    if u in ("us/ft", "usec/ft", "usft"):
        return 304.8 / dt
    if u in ("us/m", "usec/m", "usm"):
        return 1000.0 / dt
    raise ValueError(f"unknown DT unit: '{unit}'")


def load_well_log(path: str,
                  dt_curve: str = "DT",
                  dts_curve: str = "DTS",
                  rho_curve: str = "RHOB",
                  depth_curve: str = "DEPT") -> WellLog:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    las = lasio.read(str(p))
    curves = {c.mnemonic: c for c in las.curves}

    if depth_curve not in curves:
        raise KeyError(
            f"{depth_curve} not in LAS file; available: {list(curves)}")
    depth = np.asarray(las[depth_curve], dtype=float)
    depth_unit = curves[depth_curve].unit or ""
    if depth_unit.lower() not in ("m", "meter", ""):
        # Volve standard is metres; refuse to silently convert.
        raise ValueError(
            f"DEPT unit '{depth_unit}' not metres; rejecting silent unit "
            f"conversion.")

    if dt_curve not in curves:
        raise KeyError(
            f"{dt_curve} (sonic) not in LAS file; available: {list(curves)}")
    dt_unit = curves[dt_curve].unit or "us/ft"
    dt = np.asarray(las[dt_curve], dtype=float)
    vp = _dt_to_vp_kms(dt, dt_unit)

    vs = None
    if dts_curve in curves:
        dts_unit = curves[dts_curve].unit or "us/ft"
        dts = np.asarray(las[dts_curve], dtype=float)
        vs = _dt_to_vp_kms(dts, dts_unit)

    rho = None
    if rho_curve in curves:
        rho = np.asarray(las[rho_curve], dtype=float)

    # Mask out NaN sonic samples (logger gaps, casing, mudcake).
    finite = np.isfinite(vp) & np.isfinite(depth)
    depth = depth[finite]
    vp = vp[finite]
    if vs is not None:
        vs = vs[finite]
    if rho is not None:
        rho = rho[finite]

    well = las.well.get("WELL")
    well_name = str(well.value) if well else p.stem

    return WellLog(
        well_name=well_name,
        depth_m=depth,
        vp_kms=vp,
        vs_kms=vs,
        rho_gcc=rho,
        source_curves=dict(curves),
    )


def summary(log: WellLog) -> str:
    lines = [
        f"well:               {log.well_name}",
        f"depth samples:      {len(log.depth_m)}",
        f"depth range (m):    "
        f"{log.depth_m.min():.1f} -> {log.depth_m.max():.1f}",
        f"Vp range (km/s):    "
        f"{log.vp_kms.min():.2f} -> {log.vp_kms.max():.2f}",
        f"Vp mean (km/s):     {log.vp_kms.mean():.2f}",
    ]
    if log.vs_kms is not None:
        lines.append(
            f"Vs range (km/s):    "
            f"{np.nanmin(log.vs_kms):.2f} -> {np.nanmax(log.vs_kms):.2f}")
    if log.rho_gcc is not None:
        lines.append(
            f"rho range (g/cm3):  "
            f"{np.nanmin(log.rho_gcc):.2f} -> {np.nanmax(log.rho_gcc):.2f}")
    return "\n".join(lines)


def plot_log(log: WellLog, out_path: str = "volve_sonic.png"):
    """Sonic + (if present) density log on a depth track."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if log.rho_gcc is not None:
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 8), sharey=True)
        ax_vp, ax_rho = axes
    else:
        fig, ax_vp = plt.subplots(1, 1, figsize=(4.5, 8))
        ax_rho = None

    ax_vp.plot(log.vp_kms, log.depth_m, color="#b2182b", lw=0.7)
    ax_vp.invert_yaxis()
    ax_vp.set_xlabel("Vp (km/s)")
    ax_vp.set_ylabel("depth (m, MD)")
    ax_vp.set_title(f"{log.well_name} - Vp")
    ax_vp.grid(alpha=0.3)

    if ax_rho is not None:
        ax_rho.plot(log.rho_gcc, log.depth_m, color="#2166ac", lw=0.7)
        ax_rho.set_xlabel("rho (g/cm3)")
        ax_rho.set_title(f"{log.well_name} - rho")
        ax_rho.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    print("volve.load_logs - module loaded.")
    print("Usage:")
    print("  from volve.load_logs import load_well_log, summary, plot_log")
    print("  log = load_well_log('Well_logs/15_9-F-1A.LAS')")
    print("  print(summary(log))")
    print("  plot_log(log, out_path='volve_sonic.png')")
