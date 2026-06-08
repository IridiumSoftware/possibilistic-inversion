"""
volve/geometry.py - the Volve walkaway VSP geometry deck.

Encodes the known geometry of the Volve walkaway VSP (Equinor open dataset,
2018 release; recovered from the DiscoverVolve dataset map and corroborating
secondary writeups). Source values:

  - 151 surface shot points, spaced 100 m EW, from E=3700 m to E=18700 m
    (relative to a local Volve origin), source elevation 15 m below sea
    surface (i.e. 15 m below z=0).
  - 467 downhole 4-component receivers in well 15/9-F-1A (or -F-1B), at
    measured depths 1000-7990 m at 15 m spacing.
  - Sampling 8 ms, 2001 samples per trace = 16 s record length.

These are deterministic constants in the survey - this module is the
single source of truth so loaders and the inversion can share them.

CONVENTIONS:
  - All distances in METRES.
  - Coordinate frame: Volve-local. x is the EW shot direction (positive
    east); z is depth, positive downward; the well sits at x = X_WELL
    (a survey-defined easting offset).
  - Receiver depths are MEASURED DEPTH along the well bore. For 15/9-F-1A
    the well is near-vertical to ~3000 m and only modestly deviated below;
    TVD vs MD is small in the depth range of interest. A full deviation
    survey lives in 15_9-F-1A.dev.las (or equivalent) and should be applied
    once available.
  - First-arrival times are absolute travel times from shot to receiver,
    in milliseconds (or seconds; declare units at call site).

API stability: STABLE for constants below; the helpers may add fields.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


# --- survey constants -------------------------------------------------------

N_SHOTS = 151
SHOT_X_MIN_M = 3700.0
SHOT_X_MAX_M = 18700.0
SHOT_SPACING_M = 100.0
SOURCE_Z_M = -15.0          # 15 m below sea surface => z=-15 (downward+ convention)

N_RECEIVERS = 467
RECV_Z_MIN_M = 1000.0
RECV_Z_MAX_M = 7990.0
RECV_SPACING_M = 15.0
RECV_COMPONENTS = 4          # hydrophone + 3-C particle motion

# placeholder: precise well easting in the Volve-local frame. The dataset
# documentation gives a UTM/easting; this constant is updated once the
# header inspection on the downloaded SEG-Y confirms it.
X_WELL_M_PLACEHOLDER = 10000.0   # rough midpoint of the shot line; refine post-download

SAMPLE_INTERVAL_S = 0.008    # 8 ms
N_SAMPLES = 2001
RECORD_LENGTH_S = SAMPLE_INTERVAL_S * (N_SAMPLES - 1)


# --- coordinate arrays ------------------------------------------------------

def shot_coords() -> np.ndarray:
    """Shot easting + depth arrays. Shape (N_SHOTS, 2): columns (x, z) in m."""
    x = np.linspace(SHOT_X_MIN_M, SHOT_X_MAX_M, N_SHOTS)
    z = np.full_like(x, SOURCE_Z_M)
    return np.stack([x, z], axis=1)


def receiver_coords(x_well_m: float = X_WELL_M_PLACEHOLDER) -> np.ndarray:
    """Receiver coordinates assuming a vertical well at x = x_well_m.
    Shape (N_RECEIVERS, 2): columns (x, z) in m. Once a deviation survey
    is loaded, swap this for `receiver_coords_from_deviation(...)`."""
    z = np.linspace(RECV_Z_MIN_M, RECV_Z_MAX_M, N_RECEIVERS)
    x = np.full_like(z, x_well_m)
    return np.stack([x, z], axis=1)


# --- summary ----------------------------------------------------------------

@dataclass
class VolveSurvey:
    """One-stop survey description."""
    n_shots: int = N_SHOTS
    n_receivers: int = N_RECEIVERS
    record_length_s: float = RECORD_LENGTH_S
    sample_interval_s: float = SAMPLE_INTERVAL_S
    shots: np.ndarray = field(default_factory=shot_coords)
    receivers: np.ndarray = field(default_factory=receiver_coords)

    def n_picks(self) -> int:
        """Maximum picks if every shot-receiver pair is used."""
        return self.n_shots * self.n_receivers


def summary() -> str:
    s = VolveSurvey()
    lines = [
        "Volve walkaway VSP - geometry deck",
        f"  shots:     {s.n_shots} surface points, "
        f"x in [{SHOT_X_MIN_M:.0f}, {SHOT_X_MAX_M:.0f}] m "
        f"at {SHOT_SPACING_M:.0f} m, z = {SOURCE_Z_M:.1f} m",
        f"  receivers: {s.n_receivers} downhole, "
        f"z in [{RECV_Z_MIN_M:.0f}, {RECV_Z_MAX_M:.0f}] m "
        f"at {RECV_SPACING_M:.0f} m, {RECV_COMPONENTS}-component",
        f"  sampling:  {SAMPLE_INTERVAL_S * 1000:.0f} ms, "
        f"{N_SAMPLES} samples, record length {RECORD_LENGTH_S:.2f} s",
        f"  max picks: {s.n_picks()} (shot x receiver)",
        f"  well x (placeholder): {X_WELL_M_PLACEHOLDER:.0f} m "
        "(refine post-download from SEG-Y headers)",
    ]
    return "\n".join(lines)


# --- sanity plot ------------------------------------------------------------

def plot_geometry(out_path: str = "volve_geometry.png"):
    """Render the shot line + receiver string in cross-section."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    survey = VolveSurvey()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.scatter(survey.shots[:, 0], survey.shots[:, 1],
               marker="*", color="#b2182b", s=22, label="shots (151)",
               zorder=3)
    ax.scatter(survey.receivers[:, 0], survey.receivers[:, 1],
               marker="v", color="#2166ac", s=8, label="receivers (467)",
               zorder=3)
    ax.invert_yaxis()
    ax.set_xlabel("easting (m)")
    ax.set_ylabel("depth (m)  -  positive downward")
    ax.set_title("Volve walkaway VSP - geometry "
                 "(well x = placeholder, refine post-download)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    print(summary())
    out = plot_geometry()
    print(f"geometry plot: {out}")
