"""PoroTomo (Brady Hot Springs) data loaders: P-wave picks + nodal station geometry.

Data source (CC-BY 4.0):
  - Picks: GDR submission 924, s3://nrel-pds-porotomo/Nodal/nodal_analysis/p_picks/
    AIC_Stage{1..4}_Picks.txt  (DOI 10.15121/1787666)
  - Station coords: GDR submission 826, Nodal_continuous_metadata.csv
  - Source timing log: GDR submission 824, vibroseis_timing_log.xlsx

Pick-file structure (per README.txt):
  Source line (13 whitespace-separated fields):
    Year Month Day Hour Min Sec Lat Lon Depth_km UTM_E UTM_N Elev_m VibePoint
  followed by pick lines (5 fields):
    Stage Node Time_sec SNR RMSD
  Picks were auto-generated (AIC) on stacks of 3-6 sweeps inside a +/-0.1 s
  window around travel times predicted from a preliminary inversion; absent
  picks mean the AIC picker found nothing in the window at that node.

CONVENTIONS:
  - Coordinates: UTM zone 11N (meters), as given in both files. We work in a
    local frame (x = UTM_E - X0, y = UTM_N - Y0) to keep numbers small.
  - Elevation: source line carries Elev_m (m above sea level, positive up);
    station file carries ellipsoid and geoid heights. We use the *geoid*
    column to match the source Elev_m datum (both ~1240-1330 m at Brady).
  - Travel time: Time_sec is seconds after the source line's timestamp
    (stack reference = second P-mode sweep timing, per README).
  - VibePoint IDs repeat across stages at (almost) the same location
    ("no more than a few meters" per README); we keep (stage, vibe_point)
    as the source key and store per-stage coordinates.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Local-frame origin (round numbers near the SW corner of the array).
X0 = 327000.0  # UTM_E offset, m
Y0 = 4405000.0  # UTM_N offset, m


@dataclass
class Source:
    """One vibe-point stack (a source line in a pick file)."""

    stage: int
    vibe_point: int
    lat: float
    lon: float
    utm_e: float
    utm_n: float
    elev_m: float
    epoch: tuple  # (Y, M, D, h, m, s) — kept verbatim; only used for bookkeeping


@dataclass
class Picks:
    """All picks from one or more stage files, flattened to arrays."""

    # one row per pick
    stage: np.ndarray  # int
    vibe_point: np.ndarray  # int
    node: np.ndarray  # int (1-based station number)
    time_s: np.ndarray  # float, travel time
    snr: np.ndarray  # float
    rmsd: np.ndarray  # float
    # source coordinates broadcast per pick (local frame)
    src_x: np.ndarray
    src_y: np.ndarray
    src_z: np.ndarray  # elevation, m ASL (positive up)
    sources: list = field(default_factory=list)  # list[Source]

    def __len__(self) -> int:
        return len(self.time_s)


def load_stations(path: str | None = None) -> dict[int, tuple[float, float, float]]:
    """Station number -> (x, y, z) in the local frame; z = geoid height m ASL.

    The CSV uses CR-only line endings; text mode with universal newlines
    handles that transparently.
    """
    if path is None:
        path = os.path.join(DATA_DIR, "nodal_metadata.csv")
    stations: dict[int, tuple[float, float, float]] = {}
    with open(path, "r", newline=None) as fh:
        header = fh.readline()
        assert header.lstrip().startswith("Station"), "unexpected header"
        for line in fh:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7 or not parts[0]:
                continue
            tag = parts[0].lstrip("N")
            if not tag.isdigit():
                continue  # NPP01/NPP04 reference stations — not in the pick files
            num = int(tag)
            utm_e = float(parts[3])
            utm_n = float(parts[4])
            z_geoid = float(parts[6])  # height above (unknown) geoid ~ m ASL
            stations[num] = (utm_e - X0, utm_n - Y0, z_geoid)
    return stations


def load_picks(
    stages: tuple[int, ...] = (1, 2, 3, 4),
    data_dir: str | None = None,
    min_snr: float = 0.0,
) -> Picks:
    """Parse AIC_Stage{n}_Picks.txt files into a flat Picks structure."""
    if data_dir is None:
        data_dir = DATA_DIR
    rows: list[tuple] = []
    sources: list[Source] = []
    for stg in stages:
        path = os.path.join(data_dir, f"AIC_Stage{stg}_Picks.txt")
        cur: Source | None = None
        with open(path) as fh:
            for line in fh:
                parts = line.split()
                if not parts:
                    continue
                if len(parts) >= 13:  # source line
                    cur = Source(
                        stage=stg,
                        vibe_point=int(parts[12]),
                        lat=float(parts[6]),
                        lon=float(parts[7]),
                        utm_e=float(parts[9]),
                        utm_n=float(parts[10]),
                        elev_m=float(parts[11]),
                        epoch=tuple(parts[0:6]),
                    )
                    sources.append(cur)
                elif len(parts) == 5:  # pick line
                    assert cur is not None, "pick line before any source line"
                    snr = float(parts[3])
                    if snr < min_snr:
                        continue
                    rows.append(
                        (
                            int(parts[0]),
                            cur.vibe_point,
                            int(parts[1]),
                            float(parts[2]),
                            snr,
                            float(parts[4]),
                            cur.utm_e - X0,
                            cur.utm_n - Y0,
                            cur.elev_m,
                        )
                    )
                else:
                    raise ValueError(f"unparseable line in {path}: {line!r}")
    arr = np.array(rows, dtype=float)
    return Picks(
        stage=arr[:, 0].astype(int),
        vibe_point=arr[:, 1].astype(int),
        node=arr[:, 2].astype(int),
        time_s=arr[:, 3],
        snr=arr[:, 4],
        rmsd=arr[:, 5],
        src_x=arr[:, 6],
        src_y=arr[:, 7],
        src_z=arr[:, 8],
        sources=sources,
    )
