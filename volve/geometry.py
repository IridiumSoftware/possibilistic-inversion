"""
volve/geometry.py - the Volve 15/9-F-15A VSP geometry deck.

The shipped values come from the real SEG-Y headers in
`volve/data/15_9-F-15 A/VSPNI_RAW_2.SEGY` (Z-component, 1248 traces).
This module supersedes an earlier placeholder deck (151 shots / 467 receivers
in an imagined walkaway layout) that was based on a secondary-source
reconstruction and turned out to be wrong for this dataset. The earlier
constants are gone.

SURVEY (READ Well Services, 5 Jan 2009 for StatoilHydro):
  - VSP Near-Incidence (VSPNI), but actually a SPARSE WALKAWAY -
    source-to-wellhead offsets 421-1601 m, not strict zero-offset.
  - 312 shot records (FieldRecord IDs 14..360) at 145 unique surface
    source positions; multiple field records per source position
    (replicate shots for stacking).
  - 4 sensors per recording, with ~15 m intra-array spacing along the
    well bore.
  - 224 unique receiver elevations covering ~130.8 to ~3134.7 m below
    datum, swept by repositioning the 4-sensor array.
  - 3-component geophone (Z, X, Y) split across VSPNI_RAW_2, _3, _4
    SEGY files; near-field hydrophone monitor on VSPNI_RAW_1.
  - Z and horizontals: 5000 samples x 1 ms = 5 s record.
  - Hydrophone monitor: 1997 samples x 0.25 ms = 0.499 s record.

CONVENTIONS:
  - All horizontal coordinates in metres, in the survey's UTM-like frame
    (X ~ 433-435 km, Y ~ 6477-6478.5 km). The SEG-Y CoordinateScalar
    field (= -10) is applied; values in headers are stored as integer
    tenths of a metre.
  - Receiver elevation in metres below SEG-Y elevation datum
    (ElevationScalar = -10000; values stored in 0.1 mm = ten-thousandths
    of a metre). Combined with the per-bundle datum (Kelly Bushing
    54.9 m above MSL per the velocity report), elevations decode to
    sensible measured-depth-below-KB values for the well bore.
  - Time in seconds; sample rate in seconds.

API stability: STABLE for `Geometry` dataclass + `load_geometry_from_segy`.
The recorded constants below are the values observed in THIS bundle's
VSPNI_RAW_2 SEGY; they are documentary, not load-bearing - the
authoritative geometry always comes from `load_geometry_from_segy`
on the real file.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import segyio
except ImportError as e:
    raise ImportError(
        "segyio is required; add it to pyproject.toml dependencies "
        "and run `uv sync`."
    ) from e


# --- Per-bundle datum + scale conventions (from velocity report INF) -------

KB_ABOVE_MSL_M = 54.9         # Kelly Bushing elevation above MSL
SOURCE_DEPTH_M = 5.0          # source below sea surface
SEA_BED_DEPTH_M = 91.0        # sea bed below sea surface
WATER_VELOCITY_KMS = 1.500    # km/s, water column

# Standard SEG-Y scale fields for this bundle, observed once.
COORD_SCALAR = -10            # multiply raw header xy by 1/10 to get metres
ELEV_SCALAR = -10000          # multiply raw elevations by 1/10000 to get metres


# --- Observed geometry summary (documentary; see load_geometry_from_segy) --

N_FIELD_RECORDS_OBS = 312
N_TRACES_PER_FR_OBS = 4
N_TRACES_TOTAL_OBS = N_FIELD_RECORDS_OBS * N_TRACES_PER_FR_OBS    # 1248
N_UNIQUE_SOURCE_XY_OBS = 145
N_UNIQUE_RECV_XY_OBS = 219
N_UNIQUE_RECV_ELEV_OBS = 224
RECV_ELEV_MIN_M = 130.815
RECV_ELEV_MAX_M = 3134.711
RECV_ELEV_STEP_M = 15.12      # intra-array spacing along the well

SOURCE_OFFSET_MIN_M = 421.35
SOURCE_OFFSET_MEDIAN_M = 529.79
SOURCE_OFFSET_MAX_M = 1600.61

WELLHEAD_XY_M_OBS = (434927.30, 6477975.60)   # median of group xy

# Files in the bundle, components:
VSPNI_FILES = {
    "Z":  "VSPNI_RAW_2.SEGY",    # 1248 traces, 5000 samples @ 1 ms
    "X":  "VSPNI_RAW_3.SEGY",    # 1248 traces, 5000 samples @ 1 ms
    "Y":  "VSPNI_RAW_4.SEGY",    # 1248 traces, 5000 samples @ 1 ms
    "Hyd": "VSPNI_RAW_1.SEGY",   # 312 traces, 1997 samples @ 0.25 ms - monitor
}


# --- Authoritative geometry, loaded from a real SEG-Y --------------------

@dataclass
class Geometry:
    """Full geometry of a VSP component file, decoded from headers.

    The traces in `field_records` align element-wise with `source_xy`,
    `receiver_xy`, `receiver_elev_m`. So row i is "FieldRecord
    field_records[i], source at source_xy[i], receiver at
    (receiver_xy[i], receiver_elev_m[i])." Use this layout to derive
    pick tables and per-trace metadata downstream.
    """
    n_traces: int
    n_samples: int
    sample_interval_s: float
    field_records: np.ndarray       # (n_traces,) int
    trace_numbers: np.ndarray       # (n_traces,) int
    source_xy: np.ndarray           # (n_traces, 2) float metres
    receiver_xy: np.ndarray         # (n_traces, 2) float metres
    receiver_elev_m: np.ndarray     # (n_traces,) float, m below ElevationScalar datum
    wellhead_xy_m: np.ndarray       # (2,) median of receiver_xy

    @property
    def record_length_s(self) -> float:
        return float(self.n_samples * self.sample_interval_s)

    def source_offsets_m(self) -> np.ndarray:
        """Per-trace horizontal source-to-wellhead offset."""
        wx, wy = self.wellhead_xy_m
        dx = self.source_xy[:, 0] - wx
        dy = self.source_xy[:, 1] - wy
        return np.sqrt(dx ** 2 + dy ** 2)

    def unique_source_xy(self) -> np.ndarray:
        return np.unique(self.source_xy, axis=0)

    def unique_receiver_elev_m(self) -> np.ndarray:
        return np.unique(np.round(self.receiver_elev_m, 4))


def _decode_coord_scalar(scalar: int) -> float:
    if scalar == 0:
        return 1.0
    if scalar > 0:
        return float(scalar)
    return 1.0 / float(-scalar)


def load_geometry_from_segy(path: str) -> Geometry:
    """Decode geometry from a single SEG-Y component file. This is the
    authoritative source for any downstream computation; the documentary
    constants above are observed values from one specific bundle and may
    NOT match other VSP files even within the same well."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with segyio.open(str(p), "r", ignore_geometry=True) as f:
        n = f.tracecount
        n_samp = len(f.samples)
        dt_s = float(f.samples[1] - f.samples[0]) / 1000.0 if n_samp > 1 else 0.0

        # Bulk read - per-trace header values
        coord_scale = _decode_coord_scalar(
            f.header[0][segyio.TraceField.SourceGroupScalar])
        elev_scale = _decode_coord_scalar(
            f.header[0][segyio.TraceField.ElevationScalar])

        fr = np.empty(n, dtype=int)
        tn = np.empty(n, dtype=int)
        sx = np.empty(n); sy = np.empty(n)
        gx = np.empty(n); gy = np.empty(n)
        re = np.empty(n)
        for i in range(n):
            h = f.header[i]
            fr[i] = h[segyio.TraceField.FieldRecord]
            tn[i] = h[segyio.TraceField.TraceNumber]
            sx[i] = h[segyio.TraceField.SourceX]
            sy[i] = h[segyio.TraceField.SourceY]
            gx[i] = h[segyio.TraceField.GroupX]
            gy[i] = h[segyio.TraceField.GroupY]
            re[i] = h[segyio.TraceField.ReceiverGroupElevation]

    src_xy = np.column_stack([sx, sy]) * coord_scale
    grp_xy = np.column_stack([gx, gy]) * coord_scale
    rec_e = re * elev_scale
    wellhead = np.array([float(np.median(grp_xy[:, 0])),
                         float(np.median(grp_xy[:, 1]))])
    return Geometry(
        n_traces=n,
        n_samples=n_samp,
        sample_interval_s=dt_s,
        field_records=fr,
        trace_numbers=tn,
        source_xy=src_xy,
        receiver_xy=grp_xy,
        receiver_elev_m=rec_e,
        wellhead_xy_m=wellhead,
    )


def summarize(geo: Geometry) -> str:
    offs = geo.source_offsets_m()
    elev = geo.unique_receiver_elev_m()
    src = geo.unique_source_xy()
    lines = [
        f"Volve VSP geometry (loaded from headers)",
        f"  n_traces:           {geo.n_traces}",
        f"  samples / trace:    {geo.n_samples}",
        f"  sample interval:    {geo.sample_interval_s * 1000:.3f} ms",
        f"  record length:      {geo.record_length_s:.3f} s",
        f"  field records:      {len(np.unique(geo.field_records))} unique "
        f"({geo.field_records.min()} .. {geo.field_records.max()})",
        f"  source positions:   {len(src)} unique xy",
        f"    source offsets:   {offs.min():.2f} / "
        f"{np.median(offs):.2f} / {offs.max():.2f} m (min/median/max)",
        f"  receiver elevations:{len(elev)} unique, "
        f"{elev.min():.3f} .. {elev.max():.3f} m below ElevationScalar datum",
        f"  wellhead xy (m):    "
        f"({geo.wellhead_xy_m[0]:.2f}, {geo.wellhead_xy_m[1]:.2f})",
    ]
    return "\n".join(lines)


# --- Sanity plot ----------------------------------------------------------

def plot_geometry(geo: Geometry,
                  out_path: str = "volve_geometry.png") -> str:
    """Three-panel survey diagram: map view + cross-section + offset
    histogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    src = geo.unique_source_xy()
    elev = geo.unique_receiver_elev_m()
    rxy = geo.receiver_xy
    wx, wy = geo.wellhead_xy_m

    fig = plt.figure(figsize=(13.5, 6.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 0.8], wspace=0.32)

    # Map view: sources + wellhead + receiver-xy track
    axm = fig.add_subplot(gs[0, 0])
    axm.scatter(src[:, 0], src[:, 1], marker="*", color="#b2182b", s=22,
                label=f"sources ({len(src)})")
    axm.scatter(rxy[:, 0], rxy[:, 1], marker="o", color="#2166ac", s=6,
                alpha=0.4, label=f"receiver xy ({len(geo.receiver_xy)} traces)")
    axm.scatter([wx], [wy], marker="P", color="black", s=110,
                label="wellhead (median)")
    axm.set_xlabel("X (m)")
    axm.set_ylabel("Y (m)")
    axm.set_title("map view")
    axm.set_aspect("equal")
    axm.grid(alpha=0.3)
    axm.legend(fontsize=8, loc="best")

    # Depth cross-section: sources at surface vs receivers in well
    axc = fig.add_subplot(gs[0, 1])
    off = geo.source_offsets_m()
    rec_off = np.sqrt((rxy[:, 0] - wx) ** 2 + (rxy[:, 1] - wy) ** 2)
    src_z = np.zeros_like(off)
    axc.scatter(off, src_z, marker="*", color="#b2182b", s=22, label="shots")
    axc.scatter(rec_off, geo.receiver_elev_m, marker="o", color="#2166ac",
                s=6, alpha=0.5, label="receivers")
    axc.invert_yaxis()
    axc.set_xlabel("horizontal offset from wellhead (m)")
    axc.set_ylabel("depth below datum (m)")
    axc.set_title("cross-section")
    axc.grid(alpha=0.3)
    axc.legend(fontsize=8)

    # Source-offset histogram
    axh = fig.add_subplot(gs[0, 2])
    axh.hist(np.unique(off), bins=20, color="#b2182b", alpha=0.85)
    axh.set_xlabel("source-to-wellhead offset (m)")
    axh.set_ylabel("unique source count")
    axh.set_title("offset distribution")
    axh.grid(alpha=0.3)

    fig.suptitle(f"15/9-F-15A VSPNI - real geometry from SEG-Y headers "
                 f"({geo.n_traces} traces)", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# --- CLI ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.argv.append("volve/data/15_9-F-15 A/VSPNI_RAW_2.SEGY")
    geo = load_geometry_from_segy(sys.argv[1])
    print(summarize(geo))
    out = plot_geometry(geo)
    print(f"\ngeometry plot: {out}")
