"""
volve/picker.py - first-arrival picker for VSP shot gathers.

The pipeline:
  1. Pre-condition each trace: bandpass 8-80 Hz (volve.preprocess).
  2. STA/LTA trigger to localize the first break to within ~20 ms.
  3. AIC refinement inside a tight window around the STA/LTA pick.
  4. Emit a pick table (one row per trace) with provenance fields from
     the SEG-Y headers and a per-pick quality score.

Returning a pick is optional - if no STA/LTA trigger exceeds threshold in
the search window, the pick is recorded as None and the trace is flagged
in the output as `flag = 'no_trigger'`. Downstream the integration test
filters those out; we do not silently fill in zeros.

CONVENTIONS:
  - All times in seconds; sample indices are integer offsets into trace.
  - Pick time is the LOCATION OF THE FIRST P-WAVE ENERGY ARRIVAL,
    measured from the SEG-Y trace start (t=0 = first sample).
  - Quality is the STA/LTA ratio AT the pick (higher = stronger trigger
    above noise).

API stability: STABLE for `pick_one`, `pick_file`, and the
`PickRecord` shape.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List
import csv

import numpy as np

try:
    import segyio
except ImportError as e:
    raise ImportError("segyio required; uv sync") from e

from volve.preprocess import bandpass


# --- STA/LTA --------------------------------------------------------------

def sta_lta(trace, sta_len, lta_len):
    """Classic STA/LTA ratio.

    `sta_len` and `lta_len` are sample counts (sta_len < lta_len). Returns
    an array of length n_samples whose value at sample i is the ratio of
    mean energy in trace[i-sta_len+1..i] to mean energy in
    trace[i-lta_len+1..i]. The first lta_len-1 samples are 1.0
    (insufficient context)."""
    n = len(trace)
    energy = (trace ** 2).astype(np.float64)
    cum = np.concatenate([[0.0], np.cumsum(energy)])
    sta = (cum[sta_len:] - cum[:-sta_len]) / sta_len      # len n-sta_len+1
    lta = (cum[lta_len:] - cum[:-lta_len]) / lta_len      # len n-lta_len+1
    out = np.ones(n, dtype=np.float64)
    n_ratio = n - lta_len + 1
    sta_aligned = sta[lta_len - sta_len: lta_len - sta_len + n_ratio]
    lta_aligned = lta[:n_ratio]
    out[lta_len - 1: lta_len - 1 + n_ratio] = \
        sta_aligned / np.maximum(lta_aligned, 1e-20)
    return out


def aic(trace):
    """Akaike Information Criterion picker (Maeda 1985).
    Returns AIC[i] = i log var(trace[:i]) + (N-i-1) log var(trace[i:]).
    The argmin (excluding endpoints) is the optimal change-point."""
    n = len(trace)
    aic_vals = np.full(n, np.inf)
    for i in range(1, n - 1):
        v1 = np.var(trace[:i + 1])
        v2 = np.var(trace[i + 1:])
        if v1 > 0 and v2 > 0:
            aic_vals[i] = i * np.log(v1) + (n - i - 1) * np.log(v2)
    return aic_vals


# --- Per-trace pick -------------------------------------------------------

@dataclass
class PickRecord:
    trace_idx: int
    field_record: int
    trace_number: int
    source_x: float
    source_y: float
    receiver_x: float
    receiver_y: float
    receiver_elev_m: float
    source_offset_m: float
    pick_time_s: Optional[float]
    pick_quality: Optional[float]
    flag: str                  # 'ok', 'no_trigger', 'low_quality'


def pick_one(trace, dt_s,
             sta_ms=10.0, lta_ms=100.0,
             threshold=8.0,
             search_window_s=(0.05, 2.5),
             refine_window_ms=20.0,
             pre_filter=True):
    """Pick first arrival on one trace.
    Returns (pick_time_s_or_None, quality, flag)."""
    if pre_filter:
        trace = bandpass(trace, dt_s)

    sta_n = max(2, int(round(sta_ms * 1e-3 / dt_s)))
    lta_n = max(sta_n * 4, int(round(lta_ms * 1e-3 / dt_s)))
    ratio = sta_lta(trace, sta_n, lta_n)

    lo_idx = max(0, int(round(search_window_s[0] / dt_s)))
    hi_idx = min(len(trace) - 1, int(round(search_window_s[1] / dt_s)))
    if hi_idx <= lo_idx:
        return None, None, "search_window_empty"

    win = ratio[lo_idx:hi_idx]
    above = np.where(win > threshold)[0]
    if len(above) == 0:
        return None, None, "no_trigger"

    trig_local = int(above[0])
    trig_idx = lo_idx + trig_local

    # AIC refine inside +/- refine_window_ms around the trigger.
    refine_n = max(8, int(round(refine_window_ms * 1e-3 / dt_s)))
    r_lo = max(0, trig_idx - refine_n)
    r_hi = min(len(trace), trig_idx + refine_n)
    aic_vals = aic(trace[r_lo:r_hi])
    refine_offset = int(np.argmin(aic_vals))
    pick_idx = r_lo + refine_offset
    pick_time_s = pick_idx * dt_s
    quality = float(ratio[trig_idx])

    flag = "ok" if quality >= threshold else "low_quality"
    return pick_time_s, quality, flag


# --- File-level pick ------------------------------------------------------

def pick_file(segy_path, geo, **picker_kwargs) -> List[PickRecord]:
    """Pick every trace in a SEG-Y file. `geo` is the matching Geometry
    instance from volve.geometry.load_geometry_from_segy()."""
    records: List[PickRecord] = []
    with segyio.open(str(segy_path), "r", ignore_geometry=True) as f:
        dt_s = float(f.samples[1] - f.samples[0]) / 1000.0
        offsets = geo.source_offsets_m()
        for i in range(f.tracecount):
            trace = f.trace[i].astype(np.float32)
            pick_t, quality, flag = pick_one(trace, dt_s, **picker_kwargs)
            records.append(PickRecord(
                trace_idx=i,
                field_record=int(geo.field_records[i]),
                trace_number=int(geo.trace_numbers[i]),
                source_x=float(geo.source_xy[i, 0]),
                source_y=float(geo.source_xy[i, 1]),
                receiver_x=float(geo.receiver_xy[i, 0]),
                receiver_y=float(geo.receiver_xy[i, 1]),
                receiver_elev_m=float(geo.receiver_elev_m[i]),
                source_offset_m=float(offsets[i]),
                pick_time_s=pick_t,
                pick_quality=quality,
                flag=flag,
            ))
    return records


def write_csv(records: List[PickRecord], path: str) -> str:
    fields = list(asdict(records[0]).keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))
    return path


def summarize(records: List[PickRecord]) -> str:
    n = len(records)
    flags = {}
    for r in records:
        flags[r.flag] = flags.get(r.flag, 0) + 1
    ok = [r for r in records if r.flag == "ok"]
    lines = [
        f"first-arrival pick summary",
        f"  total traces:      {n}",
        f"  by flag:           {flags}",
        f"  yield (ok):        {len(ok)} ({100 * len(ok) / n:.1f}%)",
    ]
    if ok:
        ts = np.array([r.pick_time_s for r in ok])
        qs = np.array([r.pick_quality for r in ok])
        lines += [
            f"  pick time (ok):    "
            f"min={ts.min():.3f} s, median={np.median(ts):.3f} s, "
            f"max={ts.max():.3f} s",
            f"  quality (ok):      "
            f"min={qs.min():.1f}, median={np.median(qs):.1f}, "
            f"max={qs.max():.1f}",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    from volve.geometry import load_geometry_from_segy

    p = sys.argv[1] if len(sys.argv) > 1 else \
        "volve/data/15_9-F-15 A/VSPNI_RAW_2.SEGY"
    geo = load_geometry_from_segy(p)
    records = pick_file(p, geo)
    print(summarize(records))
    out = "volve/picks/picks_z.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_csv(records, out)
    print(f"picks CSV: {out}")
