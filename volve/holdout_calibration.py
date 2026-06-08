"""
volve/holdout_calibration.py - held-out arrival calibration test.

Train the random-reference Vp(z) ensemble on a random 80% of the picks;
predict the held-out 20% pick times across the ensemble; ask the
calibration question:

  Does the observed held-out pick time fall inside the ensemble's
  predicted-pick-time interval, at the rate the ensemble claims?

This is the central calibration check for a possibilistic forecast: a
correctly-shaped forecast covers the truth at its stated coverage rate.
Possibilistic 100%-coverage forecasts should land their interval
around the truth on every held-out point; if they don't, the ensemble
under-states uncertainty or the forward operator is mis-specified.

Output:
  - stdout summary
  - JSON in volve/picks/phase3_holdout.json
  - figure with three panels: (a) predicted-vs-observed scatter with
    error bars, (b) residual-vs-offset, (c) coverage band.

Run:  uv run python -m volve.holdout_calibration
"""

from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from volve.inversion_1d import (
    PickData, load_picks, depth_grid_for_picks,
    build_G, vp_ensemble, EnsembleConfig, _damped_solve, _bisect_lambda,
    _smooth_random_vp_kms,
)


OUT_JSON = "volve/picks/phase3_holdout.json"
OUT_FIG = "volve_phase3_holdout.png"


def _split(picks: PickData, train_frac: float = 0.80, seed: int = 20260607):
    rng = np.random.default_rng(seed)
    idx = np.arange(picks.n())
    rng.shuffle(idx)
    n_train = int(round(train_frac * picks.n()))
    train_idx = np.sort(idx[:n_train])
    test_idx = np.sort(idx[n_train:])
    return train_idx, test_idx


def _subset(picks: PickData, sel) -> PickData:
    return PickData(
        sources_xyz=picks.sources_xyz[sel],
        receivers_xyz=picks.receivers_xyz[sel],
        times_s=picks.times_s[sel],
        quality=picks.quality[sel],
        field_records=picks.field_records[sel],
        source_offsets_m=picks.source_offsets_m[sel],
    )


def predict_held_out(members_vp_kms, G_test):
    """Given (n_members, n_bins) ensemble of Vp(z) [km/s] and G_test
    forward matrix (n_test, n_bins), return (n_members, n_test) predicted
    travel times in seconds."""
    s_members = 1.0 / (members_vp_kms * 1000.0)        # (n_mem, n_bins)
    return G_test @ s_members.T                          # (n_test, n_mem)


def main():
    picks = load_picks()
    grid = depth_grid_for_picks(picks)
    train_idx, test_idx = _split(picks)
    train = _subset(picks, train_idx)
    test = _subset(picks, test_idx)
    print(f"loaded {picks.n()} ok picks; split {len(train_idx)} train / "
          f"{len(test_idx)} test")

    members_vp, meta = vp_ensemble(train, grid)
    print(f"ensemble (train): {members_vp.shape[0]} feasible Vp(z); "
          f"median RMS = "
          f"{np.median(meta['members_rms_s']) * 1000:.1f} ms")

    G_test = build_G(test, grid)
    pred_t = predict_held_out(members_vp, G_test)        # (n_test, n_mem)
    pred_min = pred_t.min(axis=1)
    pred_max = pred_t.max(axis=1)
    pred_med = np.median(pred_t, axis=1)
    obs_t = test.times_s

    inside = (obs_t >= pred_min) & (obs_t <= pred_max)
    resid = obs_t - pred_med
    interval_width = pred_max - pred_min

    print()
    print("held-out calibration:")
    print(f"  inside ensemble interval : {int(inside.sum())}/{len(obs_t)} "
          f"({100*inside.mean():.1f}%)")
    print(f"  residual (obs - ens_med) :")
    print(f"    mean   = {resid.mean()*1000:+.2f} ms")
    print(f"    median = {np.median(resid)*1000:+.2f} ms")
    print(f"    std    = {resid.std()*1000:.2f} ms")
    print(f"    RMS    = {np.sqrt((resid**2).mean())*1000:.2f} ms")
    print(f"  interval width:")
    print(f"    mean   = {interval_width.mean()*1000:.1f} ms")
    print(f"    median = {np.median(interval_width)*1000:.1f} ms")

    # Figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.5))

    # (a) predicted-vs-observed
    ax[0].errorbar(obs_t * 1000, pred_med * 1000,
                   yerr=[(pred_med - pred_min) * 1000,
                         (pred_max - pred_med) * 1000],
                   fmt="o", ms=2, color="#2166ac", ecolor="#88aaff",
                   alpha=0.6, elinewidth=0.4)
    lims = [min(obs_t.min(), pred_med.min()) * 1000,
            max(obs_t.max(), pred_med.max()) * 1000]
    ax[0].plot(lims, lims, "k--", lw=0.6)
    ax[0].set_xlabel("observed pick time (ms)")
    ax[0].set_ylabel("ensemble predicted pick time (ms)")
    ax[0].set_title(f"predicted vs observed\n"
                    f"(inside={100*inside.mean():.1f}%)")
    ax[0].grid(alpha=0.3)

    # (b) residual vs offset
    ax[1].scatter(test.source_offsets_m, resid * 1000, s=4,
                  color="#2166ac", alpha=0.6)
    ax[1].axhline(0, color="0.5", lw=0.5)
    ax[1].set_xlabel("source-to-well offset (m)")
    ax[1].set_ylabel("residual obs - ens_med (ms)")
    ax[1].set_title("residual vs offset")
    ax[1].grid(alpha=0.3)

    # (c) interval coverage band
    order = np.argsort(test.receivers_xyz[:, 2])
    z = test.receivers_xyz[order, 2]
    ax[2].fill_between(z, pred_min[order] * 1000, pred_max[order] * 1000,
                       color="#88aaff", alpha=0.5,
                       label="ensemble interval")
    ax[2].plot(z, pred_med[order] * 1000, color="#08306b", lw=0.7,
               label="ensemble median")
    ax[2].plot(z, obs_t[order] * 1000, "ro", ms=2.0, alpha=0.5,
               label="held-out observed")
    ax[2].set_xlabel("receiver depth (m)")
    ax[2].set_ylabel("pick time (ms)")
    ax[2].set_title("coverage band along depth")
    ax[2].grid(alpha=0.3)
    ax[2].legend(fontsize=8, loc="upper left")

    fig.suptitle("Volve 15/9-F-15A - held-out pick calibration "
                 f"({len(test_idx)} held out)",
                 fontsize=11, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_FIG, dpi=130)
    plt.close(fig)
    print(f"\nfigure: {OUT_FIG}")

    # JSON
    cert = {
        "label": "volve_15_9_F_15A_holdout_calibration",
        "split": {
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "train_frac": float(len(train_idx) / picks.n()),
        },
        "ensemble": {
            "n_members": int(members_vp.shape[0]),
            "training_rms_median_ms":
                float(np.median(meta["members_rms_s"]) * 1000),
        },
        "calibration": {
            "n_inside_interval": int(inside.sum()),
            "n_total": int(len(obs_t)),
            "inside_pct": float(100 * inside.mean()),
            "residual_mean_ms": float(resid.mean() * 1000),
            "residual_median_ms": float(np.median(resid) * 1000),
            "residual_std_ms": float(resid.std() * 1000),
            "residual_rms_ms": float(np.sqrt((resid ** 2).mean()) * 1000),
            "interval_width_mean_ms":
                float(interval_width.mean() * 1000),
            "interval_width_median_ms":
                float(np.median(interval_width) * 1000),
        },
        "scope_notes": [
            "Calibration is the right yardstick: a 100%-coverage "
            "possibilistic forecast should land its interval over the "
            "truth on every held-out pick. The actual inside-rate is the "
            "honest report.",
            "Inside-rate well below 100% indicates the ensemble UNDER-"
            "STATES uncertainty (or the forward operator is biased), and "
            "the forced/measure-dependent split should be read with the "
            "matching caveat.",
            "Train/test split is random with seed 20260607; the test "
            "set inherits the same straight-ray modeling-error from the "
            "training inversion.",
        ],
    }
    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_JSON).write_text(json.dumps(cert, indent=2))
    print(f"certificate: {OUT_JSON}")


if __name__ == "__main__":
    main()
