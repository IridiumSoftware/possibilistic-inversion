"""
volve/preprocess.py - signal-conditioning for VSP shot gathers.

A thin wrapper around scipy.signal that defines the conditioning pipeline
the picker depends on. Two stages:

  1. BANDPASS: 8-80 Hz, 4-pole Butterworth, zero-phase via filtfilt. The
     marine-source bubble pulse and high-frequency tube-wave noise both
     get suppressed; the direct-arrival energy band is preserved.
  2. (optional) AMPLITUDE NORMALIZATION: divide by the RMS of a noise
     window at the start of the trace. STA/LTA picking is amplitude-
     invariant in principle, but normalization stabilizes the trigger
     across orders-of-magnitude amplitude variation (deep receivers see
     much less direct-wave energy than shallow ones).

CONVENTIONS:
  - Input traces are 1-D float arrays, length n_samples.
  - dt_s is the sample interval in seconds (e.g. 0.001 for 1 ms).
  - All filter passbands declared in Hz.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass(trace, dt_s, f_low=8.0, f_high=80.0, order=4):
    """Zero-phase Butterworth band-pass filter."""
    fs = 1.0 / dt_s
    nyq = 0.5 * fs
    if f_high >= nyq:
        f_high = 0.99 * nyq
    sos = butter(order, [f_low, f_high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, trace).astype(np.float32)


def noise_rms(trace, n_samples_lead=50):
    """RMS of the leading `n_samples_lead` samples - the assumed
    pre-arrival noise floor."""
    lead = trace[:max(2, n_samples_lead)]
    return float(np.sqrt(np.mean(lead ** 2)))


def normalize_by_noise(trace, n_samples_lead=50, floor=1e-12):
    """Divide trace by its pre-arrival noise RMS."""
    rms = noise_rms(trace, n_samples_lead)
    return trace / max(rms, floor)
