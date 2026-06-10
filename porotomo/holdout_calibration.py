"""porotomo/holdout_calibration.py - stage-2 held-out calibration of the
stage-1 feasible-set ensemble.

Stage 2 re-occupied the stage-1 vibe points (within a few metres) and was
never seen by the inversion, so it is a genuine held-out test of the
ensemble's predictive intervals - the PoroTomo analogue of the Volve
phase-4 held-out-arrival test (93.8% there).

Per stage-2 pick: predict travel times under every ensemble member (one
FMM per member per source); the ensemble interval is [min_k t_k, max_k t_k].
Reported:
  - raw inside-interval fraction;
  - inside fraction with a +/- pick-noise allowance (36 ms: the robust
    stage-repeat estimate, see inversion3d.Config3D notes) - the fair
    test, since the interval covers model uncertainty but the held-out
    pick itself carries its own noise;
  - RMS of the ensemble-median prediction vs observed.

CAVEAT recorded in the JSON: the stage-2 picks come from the same AIC
auto-picker seeded with windows from the SAME preliminary inversion as
stage 1, so picker-systematic errors are common-mode between inversion
and holdout - this test cannot catch them. (Same status as Volve, where
the picker was ours on both sides.)

Requires porotomo/data/ensemble_stage1.npz.

Run:  uv run python -m porotomo.holdout_calibration
Outputs: porotomo_holdout.png, porotomo_holdout_cert.json
"""

from __future__ import annotations

import json
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from porotomo.inversion3d import prepare, forward_3d
from porotomo.decompose_3d import load_ensemble

PICK_NOISE_S = 0.036


def main() -> None:
    members, air, grid, _z = load_ensemble()
    picks, grid2, air2, ds2 = prepare(stage=2)
    t_obs = np.concatenate(ds2.times)
    n_picks = len(t_obs)
    n_members = members.shape[0]

    print(f"stage-2 holdout: {n_picks} picks, {len(ds2.src_pts)} sources, "
          f"{n_members} members")
    preds = np.empty((n_members, n_picks))
    t0 = time.time()
    for k in range(n_members):
        preds[k], _ = forward_3d(members[k], ds2, grid)
        if (k + 1) % 10 == 0:
            print(f"  member {k+1}/{n_members} ({time.time()-t0:.0f} s)")
    t_lo = preds.min(axis=0)
    t_hi = preds.max(axis=0)
    t_med = np.median(preds, axis=0)

    inside_raw = (t_obs >= t_lo) & (t_obs <= t_hi)
    inside_noise = (t_obs >= t_lo - PICK_NOISE_S) & \
                   (t_obs <= t_hi + PICK_NOISE_S)
    resid_med = t_obs - t_med
    rms_med = float(np.sqrt(np.mean(resid_med**2)))

    cert = {
        "dataset": "PoroTomo stage-2 nodal P picks (GDR 924), held out",
        "n_picks": int(n_picks),
        "n_members": int(n_members),
        "inside_raw": round(float(inside_raw.mean()), 4),
        "inside_with_pick_noise": round(float(inside_noise.mean()), 4),
        "pick_noise_allowance_s": PICK_NOISE_S,
        "rms_median_prediction_ms": round(rms_med * 1000, 1),
        "median_interval_width_ms": round(
            float(np.median(t_hi - t_lo)) * 1000, 1),
        "caveat": "stage-2 picks share the AIC auto-picker and its "
                  "preliminary-inversion windows with stage 1; "
                  "picker-systematic errors are common-mode and not "
                  "tested here",
    }
    with open("porotomo_holdout_cert.json", "w") as fh:
        json.dump(cert, fh, indent=2)
    print(json.dumps(cert, indent=2))

    # ---- figure -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    ax.hist(resid_med * 1000, bins=120, range=(-300, 300))
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("stage-2 observed - ensemble-median predicted (ms)")
    ax.set_title(f"holdout residuals (RMS {rms_med*1000:.0f} ms)")

    ax = axes[1]
    # inside fraction vs offset
    d_m = np.concatenate([
        np.linalg.norm((ds2.recv_pts[i] - ds2.src_pts[i]) * grid.cell_m,
                       axis=1)
        for i in range(len(ds2.src_pts))
    ])
    bins = np.linspace(0, d_m.max(), 16)
    mid = 0.5 * (bins[1:] + bins[:-1])
    frac_raw = [inside_raw[(d_m >= bins[i]) & (d_m < bins[i+1])].mean()
                if ((d_m >= bins[i]) & (d_m < bins[i+1])).any() else np.nan
                for i in range(len(mid))]
    frac_n = [inside_noise[(d_m >= bins[i]) & (d_m < bins[i+1])].mean()
              if ((d_m >= bins[i]) & (d_m < bins[i+1])).any() else np.nan
              for i in range(len(mid))]
    ax.plot(mid, frac_raw, "o-", label="raw interval")
    ax.plot(mid, frac_n, "s-", label=f"+/-{PICK_NOISE_S*1000:.0f} ms allowance")
    ax.axhline(1.0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("source-node offset (m)")
    ax.set_ylabel("inside-interval fraction")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("coverage vs offset")

    ax = axes[2]
    ax.hist((t_hi - t_lo) * 1000, bins=80)
    ax.set_xlabel("ensemble interval width (ms)")
    ax.set_title(f"predictive interval widths "
                 f"(median {cert['median_interval_width_ms']:.0f} ms)")
    fig.tight_layout()
    fig.savefig("porotomo_holdout.png", dpi=130)
    print("wrote porotomo_holdout.png, porotomo_holdout_cert.json")


if __name__ == "__main__":
    main()
