"""
volve/phase5a.py - Phase A diagnostics on the phase-5 snapshot.

After the triad witness pass on phase 5, four concerns surfaced:

  A2. The 79% forced-quiet might be tautological because the anomaly
      baseline is the ensemble mean (which is prior-dominated when
      illumination is sparse). -> baseline sensitivity sweep.

  A3. The per-well sonic comparison samples a fixed wellhead w-column,
      but both wells deviate ~1 km laterally. Sonic at depth z is
      sampled at the bore's actual xy, not the wellhead's. -> reproject
      sonic onto bore trajectories.

  A4. ChatGPT's critical experiment: compute a per-cell ray-hit /
      sensitivity map and check whether it correlates with the
      measure-dependent zone. If yes, methodology IS reporting data
      illumination correctly. If no, decomposition is reporting
      ensemble internals not data constraint.

  A5. Per-cell ensemble diversity: is the ensemble exploring a
      high-dimensional model space, or has it collapsed to a thin
      sliver of the prior?

These are all post-processing on the snapshot saved by
volve.decompose_2d (volve/picks/phase5_snapshot.npz). No re-inversion.

Run:  uv run python -m volve.phase5a
"""

from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lasio

import posdec
from volve.inversion_eikonal_2d import (
    Grid2D, load_joint_picks, _unique_sources,
)
from volve.decompose_2d import (
    load_sonic_f15a, load_sonic_f11t2,
)


SNAPSHOT = "volve/picks/phase5_snapshot.npz"
OUT_FIG = "volve_phase5a_diagnostics.png"
OUT_REPORT = "volve/picks/phase5a_report.json"


# --- snapshot loader ------------------------------------------------------

def load_snapshot():
    npz = np.load(SNAPSHOT)
    grid = Grid2D(
        cell_m=float(npz["grid_cell_m"][0]),
        nw=int(npz["grid_nw"][0]),
        nz=int(npz["grid_nz"][0]),
        w0=float(npz["grid_w0"][0]),
        z0=float(npz["grid_z0"][0]),
    )
    return {
        "members_vp": npz["members_vp_kms"],
        "grid": grid,
        "cls": npz["cls"],
        "anom_min": npz["anom_min"],
        "anom_max": npz["anom_max"],
        "trend_z": npz["trend_z"],
        "train_mask": npz["train_mask"],
        "f15a_w": float(npz["f15a_wellhead_w"][0]),
        "f11t2_w": float(npz["f11t2_wellhead_w"][0]),
    }


# --- A2. baseline sensitivity sweep --------------------------------------

def baseline_sweep(snap, sonic_f15a, sonic_f11t2, eps_kms=0.10):
    """Recompute forced/measure-dep fractions under multiple anomaly
    baselines. If 79% forced-quiet is gauge-sensitive, the original
    decomposition was reporting the BASELINE choice, not the data."""
    members = snap["members_vp"]                  # (M, nz, nw)
    grid = snap["grid"]
    z = grid.z_centers()
    n_total = grid.nz * grid.nw
    n_members = members.shape[0]

    # Baseline 1: original (ensemble-mean linear depth trend)
    ens_mean_z_orig = members.mean(axis=(0, 2))
    a, b = np.polyfit(z, ens_mean_z_orig, 1)
    trend_orig_z = a * z + b
    trend_orig_2d = np.broadcast_to(trend_orig_z[:, None],
                                    (grid.nz, grid.nw))

    # Baseline 2: wireline-sonic linear trend (independent of ensemble)
    sonic_tvd = np.concatenate([sonic_f15a[0], sonic_f11t2[0]])
    sonic_vp = np.concatenate([sonic_f15a[1], sonic_f11t2[1]])
    sm = np.isfinite(sonic_tvd) & np.isfinite(sonic_vp)
    a2, b2 = np.polyfit(sonic_tvd[sm], sonic_vp[sm], 1)
    trend_sonic_z = a2 * z + b2
    trend_sonic_2d = np.broadcast_to(trend_sonic_z[:, None],
                                     (grid.nz, grid.nw))

    # Baseline 3: fixed linear (1.7 + 2.8 * z_frac) - the inversion prior
    trend_prior_z = 1.7 + 2.8 * (z / z.max())
    trend_prior_2d = np.broadcast_to(trend_prior_z[:, None],
                                     (grid.nz, grid.nw))

    # Baseline 4: leave-one-member-out trend (use first member's depth profile)
    trend_m0_z = np.polyval(np.polyfit(z, members[0].mean(axis=1), 1), z)
    trend_m0_2d = np.broadcast_to(trend_m0_z[:, None], (grid.nz, grid.nw))

    baselines = [
        ("ensemble-mean (original)", trend_orig_2d, trend_orig_z),
        ("wireline-sonic", trend_sonic_2d, trend_sonic_z),
        ("fixed prior-shape", trend_prior_2d, trend_prior_z),
        ("leave-one-out member 0", trend_m0_2d, trend_m0_z),
    ]

    rows = []
    for name, t2d, t1d in baselines:
        anom = members - t2d[None, ...]
        a_min = anom.min(axis=0)
        a_max = anom.max(axis=0)
        cls = posdec.classify(a_min, a_max, eps=eps_kms)
        n_fh = int((cls == 2).sum())
        n_fl = int((cls == -2).sum())
        n_fq = int((cls == 0).sum())
        n_md = int((cls == 1).sum())
        rows.append({
            "baseline": name,
            "trend_z_kms": t1d,
            "forced_high_pct": 100 * n_fh / n_total,
            "forced_low_pct": 100 * n_fl / n_total,
            "forced_quiet_pct": 100 * n_fq / n_total,
            "measure_dep_pct": 100 * n_md / n_total,
            "cls": cls,
        })
    return rows


# --- A3. reproject sonic onto bore trajectories -------------------------

def sonic_along_bore(snap, pp, well_id: str, sonic_tvd, sonic_vp):
    """For each receiver in `pp` (well = well_id), determine its
    (w_bore, z_bore). Sort by z. At each receiver's (w, z), bilinear-
    sample the ensemble Vp and check if sonic Vp at that depth (mapped
    via the trajectory) falls inside ensemble's [vp_min, vp_max]."""
    members = snap["members_vp"]
    grid = snap["grid"]

    sel = pp.well_id == well_id
    rec_w = pp.rec_w[sel]
    rec_z = pp.rec_z[sel]
    # Group by depth: unique receiver depths along the bore
    uniq_z, uniq_idx = np.unique(np.round(rec_z, 2), return_index=True)
    bore_w = rec_w[uniq_idx]      # one w per unique receiver depth

    iw_cells = grid.w_to_cell(bore_w)
    iz_cells = grid.z_to_cell(uniq_z)
    # Bilinear sample per member
    n_pts = uniq_z.size
    M = members.shape[0]
    ens_at_bore = np.zeros((M, n_pts), dtype=float)
    for k in range(M):
        # Bilinear interp on members[k] at (iz_cell, iw_cell)
        m = members[k]
        for i in range(n_pts):
            iz = float(iz_cells[i])
            iw = float(iw_cells[i])
            iz0 = max(0, min(grid.nz - 2, int(iz)))
            iw0 = max(0, min(grid.nw - 2, int(iw)))
            fz = iz - iz0
            fw = iw - iw0
            ens_at_bore[k, i] = (
                (1 - fz) * (1 - fw) * m[iz0, iw0]
                + (1 - fz) * fw * m[iz0, iw0 + 1]
                + fz * (1 - fw) * m[iz0 + 1, iw0]
                + fz * fw * m[iz0 + 1, iw0 + 1]
            )
    vp_min = ens_at_bore.min(axis=0)
    vp_max = ens_at_bore.max(axis=0)
    vp_med = np.median(ens_at_bore, axis=0)

    # Interpolate sonic Vp to the same depths
    order = np.argsort(sonic_tvd)
    sonic_at_z = np.interp(uniq_z, sonic_tvd[order], sonic_vp[order])
    inside = (sonic_at_z >= vp_min) & (sonic_at_z <= vp_max)
    return {
        "n_points": int(n_pts),
        "bore_z": uniq_z,
        "bore_w": bore_w,
        "sonic_at_z_kms": sonic_at_z,
        "ens_min": vp_min,
        "ens_max": vp_max,
        "ens_med": vp_med,
        "inside": inside,
        "inside_count": int(inside.sum()),
        "inside_pct": float(100 * inside.mean()),
    }


# --- A4. illumination map -----------------------------------------------

def illumination_map(pp, grid):
    """Per-cell ray-hit count using straight-line rays from source to
    receiver. The hit count is a proxy for sensitivity; cells with many
    rays are data-illuminated, cells with few are prior-dominated."""
    nw, nz = grid.nw, grid.nz
    hits = np.zeros((nz, nw), dtype=float)
    sw_cell = grid.w_to_cell(pp.src_w)
    sz_cell = grid.z_to_cell(pp.src_z)
    rw_cell = grid.w_to_cell(pp.rec_w)
    rz_cell = grid.z_to_cell(pp.rec_z)
    n_steps = 200
    for i in range(pp.n()):
        # Sample n_steps along straight-ray
        ts = np.linspace(0.0, 1.0, n_steps)
        ws = (1 - ts) * sw_cell[i] + ts * rw_cell[i]
        zs = (1 - ts) * sz_cell[i] + ts * rz_cell[i]
        # Convert to integer indices
        wi = np.clip(np.round(ws).astype(int), 0, nw - 1)
        zi = np.clip(np.round(zs).astype(int), 0, nz - 1)
        for k in range(n_steps):
            hits[zi[k], wi[k]] += 1.0
    return hits


def alignment_check(cls, illumination):
    """Does measure-dependent correlate with high illumination?
    Returns: mean illumination by class + Pearson correlation.
    Methodology test (ChatGPT's critical experiment):
      - if measure-dep cells have HIGH illumination AND forced-quiet
        cells have LOW illumination -> methodology IS reporting data
        constraint correctly
      - if no relationship -> decomposition is reporting ensemble
        internals, not data."""
    flat_ill = illumination.ravel()
    flat_cls = cls.ravel()
    by_class = {}
    for c in [-2, 0, 1, 2]:
        sel = flat_cls == c
        if sel.any():
            by_class[int(c)] = {
                "n": int(sel.sum()),
                "mean_illumination": float(flat_ill[sel].mean()),
                "median_illumination": float(np.median(flat_ill[sel])),
                "max_illumination": float(flat_ill[sel].max()),
            }
    # Pearson: measure-dep vs illumination
    is_md = (flat_cls == 1).astype(float)
    if flat_ill.std() > 0 and is_md.std() > 0:
        pearson_md = float(np.corrcoef(flat_ill, is_md)[0, 1])
    else:
        pearson_md = float("nan")
    is_quiet = (flat_cls == 0).astype(float)
    if flat_ill.std() > 0 and is_quiet.std() > 0:
        pearson_quiet = float(np.corrcoef(flat_ill, is_quiet)[0, 1])
    else:
        pearson_quiet = float("nan")
    return {
        "by_class": by_class,
        "pearson_md_vs_illumination": pearson_md,
        "pearson_quiet_vs_illumination": pearson_quiet,
    }


# --- A5. ensemble diversity ---------------------------------------------

def ensemble_diversity(members):
    """Pairwise distance + SVD of centered ensemble (the same diagnostic
    posdec.diagnostics.explored_directions uses, here for the 2D
    ensemble). If the ensemble lives on a ~1D manifold (one dominant
    SV), it's not exploring much; if it has many comparable SVs, it
    is genuinely diverse."""
    arr = members.reshape(members.shape[0], -1)
    centered = arr - arr.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    total = (s ** 2).sum()
    evr = (s ** 2 / total).tolist() if total > 0 else []
    # Pairwise RMS distances
    diff = arr[:, None, :] - arr[None, :, :]
    D = np.sqrt((diff ** 2).mean(axis=-1))
    pairwise_off_diag = D[np.triu_indices(D.shape[0], k=1)]
    return {
        "singular_values": s.tolist(),
        "explained_variance_ratio": evr,
        "leading_sv_pct": float(100 * evr[0]) if evr else 0.0,
        "top5_sv_pct": float(100 * sum(evr[:5])) if evr else 0.0,
        "n_effective_dims_90":
            int(np.searchsorted(np.cumsum(evr), 0.90)) + 1 if evr else 0,
        "pairwise_dist_kms": {
            "min": float(pairwise_off_diag.min()),
            "mean": float(pairwise_off_diag.mean()),
            "max": float(pairwise_off_diag.max()),
        },
    }


# --- main ---------------------------------------------------------------

def main():
    snap = load_snapshot()
    pp, _frame = load_joint_picks()
    sonic_f15a = load_sonic_f15a()
    sonic_f11t2 = load_sonic_f11t2()

    grid = snap["grid"]
    members = snap["members_vp"]
    M = members.shape[0]
    print(f"snapshot: {M} members, grid {grid.nz} x {grid.nw}")
    print(f"joint picks: {pp.n()}")
    print()

    # --- A2 ---
    print("=" * 64)
    print("A2. Baseline sensitivity sweep (anomaly baseline tautology test)")
    print("=" * 64)
    bs_rows = baseline_sweep(snap, sonic_f15a, sonic_f11t2)
    print(f"  {'baseline':<28s}  {'F-hi':>6s}  {'F-lo':>6s}  "
          f"{'F-qt':>6s}  {'M-dep':>6s}")
    for r in bs_rows:
        print(f"  {r['baseline']:<28s}  "
              f"{r['forced_high_pct']:6.1f}  "
              f"{r['forced_low_pct']:6.1f}  "
              f"{r['forced_quiet_pct']:6.1f}  "
              f"{r['measure_dep_pct']:6.1f}")
    fq_range = (min(r['forced_quiet_pct'] for r in bs_rows),
                max(r['forced_quiet_pct'] for r in bs_rows))
    fq_spread = fq_range[1] - fq_range[0]
    print(f"\n  -> forced-quiet % ranges {fq_range[0]:.1f} to "
          f"{fq_range[1]:.1f}  (spread {fq_spread:.1f} pp)")
    a2_verdict = (
        "TAUTOLOGICAL" if fq_spread > 30
        else "STABLE" if fq_spread < 10
        else "WEAKLY DEPENDENT")
    print(f"  -> A2 verdict: {a2_verdict}")

    # --- A3 ---
    print()
    print("=" * 64)
    print("A3. Sonic-vs-ensemble along ACTUAL bore trajectories")
    print("=" * 64)
    a3_f15a = sonic_along_bore(snap, pp, "f15a",
                               sonic_f15a[0], sonic_f15a[1])
    a3_f11t2 = sonic_along_bore(snap, pp, "f11t2",
                                sonic_f11t2[0], sonic_f11t2[1])
    print(f"  F-15A : {a3_f15a['inside_count']}/{a3_f15a['n_points']} "
          f"({a3_f15a['inside_pct']:.1f}%) inside  "
          f"[was 10.1% at wellhead w-column]")
    print(f"  F-11T2: {a3_f11t2['inside_count']}/{a3_f11t2['n_points']} "
          f"({a3_f11t2['inside_pct']:.1f}%) inside  "
          f"[was 10.0% at wellhead w-column]")
    # Verdict
    if a3_f15a['inside_pct'] > 60 or a3_f11t2['inside_pct'] > 60:
        a3_verdict = "BORE-TRAJECTORY FIX RECOVERS SONIC COVERAGE"
    elif (a3_f15a['inside_pct'] > a3_f15a['n_points'] * 0
          and a3_f15a['inside_pct'] - 10 > 20):
        a3_verdict = "BORE-TRAJECTORY FIX IMPROVES BUT DOES NOT RESOLVE"
    else:
        a3_verdict = "BORE-TRAJECTORY FIX DOES NOT EXPLAIN THE DROP"
    print(f"  -> A3 verdict: {a3_verdict}")

    # --- A4 ---
    print()
    print("=" * 64)
    print("A4. Illumination vs measure-dependent alignment "
          "(ChatGPT critical experiment)")
    print("=" * 64)
    ill = illumination_map(pp, grid)
    align = alignment_check(snap["cls"], ill)
    print(f"  illumination range: {ill.min():.1f} -> {ill.max():.1f} "
          f"hits/cell  (median {np.median(ill):.1f})")
    print(f"  mean illumination by class:")
    for c, label in [(2, "forced-high"), (-2, "forced-low"),
                     (0, "forced-quiet"), (1, "measure-dep")]:
        bc = align["by_class"].get(c)
        if bc:
            print(f"    {label:<15s} (n={bc['n']:5d}): "
                  f"{bc['mean_illumination']:8.1f}")
    print(f"  Pearson r(illumination, is_measure_dependent) = "
          f"{align['pearson_md_vs_illumination']:+.3f}")
    print(f"  Pearson r(illumination, is_forced_quiet)      = "
          f"{align['pearson_quiet_vs_illumination']:+.3f}")
    if (align["pearson_md_vs_illumination"] > 0.30
            and align["pearson_quiet_vs_illumination"] < -0.30):
        a4_verdict = ("METHODOLOGY ALIGNED: measure-dep is data-illuminated, "
                      "forced-quiet is prior-dominated")
    elif abs(align["pearson_md_vs_illumination"]) < 0.10:
        a4_verdict = "DECOUPLED: decomposition not tracking data illumination"
    else:
        a4_verdict = (f"PARTIAL: r(md,ill)="
                      f"{align['pearson_md_vs_illumination']:+.2f}, "
                      f"r(quiet,ill)="
                      f"{align['pearson_quiet_vs_illumination']:+.2f}")
    print(f"  -> A4 verdict: {a4_verdict}")

    # --- A5 ---
    print()
    print("=" * 64)
    print("A5. Ensemble diversity")
    print("=" * 64)
    div = ensemble_diversity(members)
    print(f"  leading SV explains : {div['leading_sv_pct']:.1f}% of variance")
    print(f"  top 5 SVs explain    : {div['top5_sv_pct']:.1f}%")
    print(f"  effective dims (90%) : {div['n_effective_dims_90']}")
    print(f"  pairwise dist (km/s) : min {div['pairwise_dist_kms']['min']:.3f}"
          f"  mean {div['pairwise_dist_kms']['mean']:.3f}  "
          f"max {div['pairwise_dist_kms']['max']:.3f}")
    if div["leading_sv_pct"] > 70:
        a5_verdict = "ENSEMBLE COLLAPSED to ~1D manifold"
    elif div["n_effective_dims_90"] < 5:
        a5_verdict = f"LOW DIVERSITY ({div['n_effective_dims_90']} eff dims)"
    else:
        a5_verdict = (f"GENUINE DIVERSITY "
                      f"({div['n_effective_dims_90']} eff dims)")
    print(f"  -> A5 verdict: {a5_verdict}")

    # --- figure ---
    _plot(snap, pp, bs_rows, a3_f15a, a3_f11t2, ill, align, div, OUT_FIG)
    print(f"\nfigure: {OUT_FIG}")

    # --- report ---
    # NumPy-clean version of the data (no ndarrays)
    bs_rows_json = []
    for r in bs_rows:
        bs_rows_json.append({k: v for k, v in r.items()
                             if k not in ("trend_z_kms", "cls")})
    report = {
        "snapshot": SNAPSHOT,
        "A2_baseline_sweep": {
            "rows": bs_rows_json,
            "forced_quiet_pct_range": list(fq_range),
            "verdict": a2_verdict,
        },
        "A3_sonic_along_bore": {
            "f15a_inside_count": a3_f15a["inside_count"],
            "f15a_n_points": a3_f15a["n_points"],
            "f15a_inside_pct": a3_f15a["inside_pct"],
            "f11t2_inside_count": a3_f11t2["inside_count"],
            "f11t2_n_points": a3_f11t2["n_points"],
            "f11t2_inside_pct": a3_f11t2["inside_pct"],
            "verdict": a3_verdict,
        },
        "A4_illumination_alignment": {
            "by_class": align["by_class"],
            "pearson_md_vs_illumination":
                align["pearson_md_vs_illumination"],
            "pearson_quiet_vs_illumination":
                align["pearson_quiet_vs_illumination"],
            "verdict": a4_verdict,
        },
        "A5_ensemble_diversity": {
            "leading_sv_pct": div["leading_sv_pct"],
            "top5_sv_pct": div["top5_sv_pct"],
            "effective_dims_90": div["n_effective_dims_90"],
            "pairwise_distance_kms": div["pairwise_dist_kms"],
            "verdict": a5_verdict,
        },
    }
    Path(OUT_REPORT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_REPORT).write_text(json.dumps(report, indent=2))
    print(f"report: {OUT_REPORT}")


def _plot(snap, pp, bs_rows, a3_f15a, a3_f11t2, ill, align, div, out_path):
    fig, ax = plt.subplots(2, 2, figsize=(15, 11))

    # A2: baseline sweep
    ax_a2 = ax[0, 0]
    names = [r["baseline"] for r in bs_rows]
    fq = [r["forced_quiet_pct"] for r in bs_rows]
    md = [r["measure_dep_pct"] for r in bs_rows]
    fhfl = [r["forced_high_pct"] + r["forced_low_pct"] for r in bs_rows]
    x = np.arange(len(names))
    ax_a2.bar(x - 0.25, fq, 0.25, color="#bbbbbb", label="forced-quiet")
    ax_a2.bar(x, md, 0.25, color="#fdb338", label="measure-dep")
    ax_a2.bar(x + 0.25, fhfl, 0.25, color="#b2182b",
              label="forced-high+low")
    ax_a2.set_xticks(x)
    ax_a2.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax_a2.set_ylabel("% of cells")
    ax_a2.set_title("A2. baseline sensitivity sweep")
    ax_a2.legend(fontsize=8); ax_a2.grid(alpha=0.3)

    # A3: sonic along bore for F-15A
    ax_a3 = ax[0, 1]
    z = a3_f15a["bore_z"]
    ax_a3.fill_betweenx(z, a3_f15a["ens_min"], a3_f15a["ens_max"],
                        color="#88aaff", alpha=0.5,
                        label="F-15A ensemble (bore)")
    ax_a3.plot(a3_f15a["ens_med"], z, color="#08306b", lw=1.5,
               label="F-15A ens median")
    ax_a3.plot(a3_f15a["sonic_at_z_kms"], z, color="#b2182b", lw=1.0,
               label="F-15A sonic @ bore z")
    ax_a3.invert_yaxis()
    ax_a3.set_xlabel("Vp (km/s)")
    ax_a3.set_ylabel("depth (m)")
    ax_a3.set_title(f"A3. sonic vs ensemble along bore, F-15A\n"
                    f"inside = {a3_f15a['inside_pct']:.1f}% "
                    f"(was 10.1% at wellhead w)")
    ax_a3.grid(alpha=0.3); ax_a3.legend(fontsize=8)
    ax_a3.set_xlim(1.3, 5.8)

    # A4: illumination map + classification overlay
    ax_a4 = ax[1, 0]
    grid = snap["grid"]
    z_axis = grid.z_centers()
    w_axis = grid.w_centers()
    im = ax_a4.imshow(np.log10(np.maximum(ill, 1)), aspect="auto",
                      extent=[w_axis[0], w_axis[-1], z_axis[-1], z_axis[0]],
                      cmap="viridis")
    # overlay measure-dependent cells
    cls = snap["cls"]
    md_mask = (cls == 1).astype(float)
    ax_a4.contour(np.linspace(w_axis[0], w_axis[-1], grid.nw),
                  np.linspace(z_axis[0], z_axis[-1], grid.nz),
                  md_mask, levels=[0.5], colors="#fdb338",
                  linewidths=1.0)
    ax_a4.axvline(snap["f15a_w"], color="white", lw=0.8, ls="--")
    ax_a4.axvline(snap["f11t2_w"], color="white", lw=0.8, ls="--")
    ax_a4.set_xlabel("along-well-line w (m)")
    ax_a4.set_ylabel("depth (m)")
    ax_a4.set_title(f"A4. log10 illumination + measure-dep contour\n"
                    f"r(md, illumination) = "
                    f"{align['pearson_md_vs_illumination']:+.3f}")
    fig.colorbar(im, ax=ax_a4, label="log10(hits/cell)")

    # A5: ensemble diversity (singular spectrum)
    ax_a5 = ax[1, 1]
    sv = np.array(div["singular_values"])
    evr = np.array(div["explained_variance_ratio"])
    ax_a5.semilogy(np.arange(1, len(sv) + 1), sv, "-o",
                   color="#2166ac", ms=5)
    ax_a5.set_xlabel("singular-value index")
    ax_a5.set_ylabel("singular value")
    ax_a5.set_title(f"A5. ensemble singular spectrum\n"
                    f"leading SV = {div['leading_sv_pct']:.1f}%, "
                    f"effective dims (90%) = {div['n_effective_dims_90']}")
    ax_a5.grid(alpha=0.3, which="both")

    fig.suptitle("Phase A diagnostics on the phase 5 ensemble "
                 "(triad witness pass response)",
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
