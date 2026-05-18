/* possibilistic_inversion.c - master file of the C port.
 *
 * Pulls in the modules - decomposition, straightray, eikonal, linalg,
 * inversion - and runs the possibilistic-inversion method end to end:
 *
 *   synthetic model  ->  Eikonal forward operator  ->  noisy travel-time data
 *     ->  feasible-set sampler (LM inversions + smooth perturbations)
 *     ->  possibilistic decomposition  ->  validation against ground truth.
 *
 * This is the complete pipeline of synthetic_demo_eikonal.py in dependency-free
 * C. The ensemble decomposed below is a genuine feasible set - the output of
 * the inversion - not a hand-built demonstration ensemble.
 *
 * Dependency-free: C standard library only.
 * Build:  make        (or: cc -O2 -std=c11 *.c -o pi -lm)
 * Run:    ./pi
 */

#include <stdio.h>
#include <math.h>

#include "grid.h"
#include "decomposition.h"
#include "straightray.h"
#include "eikonal.h"
#include "inversion.h"

/* --- the synthetic model (mirrors ground_truth in synthetic_demo.py) ------- */

static double cell_z(int z) { return z + 0.5; }
static double cell_x(int x) { return x + 0.5; }

static double gauss(double zz, double xx, double z0, double x0, double sig)
{
    double d2 = (zz - z0) * (zz - z0) + (xx - x0) * (xx - x0);
    return exp(-d2 / (2.0 * sig * sig));
}

/* Synthetic velocity: depth gradient + a tilted high-V slab + a broad low-V
 * zone + three small sub-resolution blobs. */
static void ground_truth(double *v)
{
    const double blob[3][3] = {           /* z0, x0, amplitude */
        { 8.0, 31.0,  0.36},
        {33.0,  9.0,  0.34},
        {15.0, 34.0, -0.32},
    };
    for (int z = 0; z < NZ; z++) {
        for (int x = 0; x < NX; x++) {
            double zz = cell_z(z), xx = cell_x(x);
            double val = 5.5 + 2.0 * (zz / NZ);                  /* background */
            if (fabs(xx - (0.55 * zz + 6.0)) < 4.0) val += 0.9;  /* high-V slab */
            val -= 0.85 * gauss(zz, xx, 27.0, 31.0, 5.5);        /* low-V zone  */
            for (int b = 0; b < 3; b++)
                val += blob[b][2] * gauss(zz, xx, blob[b][0], blob[b][1], 1.7);
            v[IDX(z, x)] = val;
        }
    }
}

/* The planted depth-gradient background - the reference the anomaly is
 * measured against (geophysical_invariants.md Tier-2: a reference model). */
static void background(double *bg)
{
    for (int z = 0; z < NZ; z++)
        for (int x = 0; x < NX; x++)
            bg[IDX(z, x)] = 5.5 + 2.0 * (cell_z(z) / NZ);
}

/* True anomaly of the wanted sign within Chebyshev radius R of cell (z,x)?
 * A tomographic image is honest only at the resolution length: a forced cell
 * one or two cells off a true feature is resolution blur, not a sign error. */
static int sign_within(const double *v, const double *bg,
                       int z, int x, int want_pos, int R)
{
    for (int dz = -R; dz <= R; dz++) {
        int zz = z + dz;
        if (zz < 0 || zz >= NZ) continue;
        for (int dx = -R; dx <= R; dx++) {
            int xx = x + dx;
            if (xx < 0 || xx >= NX) continue;
            double a = v[IDX(zz, xx)] - bg[IDX(zz, xx)];
            if (want_pos ? (a > 0.0) : (a < 0.0)) return 1;
        }
    }
    return 0;
}

/* --- survey geometry: sources and receivers on the grid boundary ---------- */

static void build_geometry(int *src, int *rec)
{
    int k = 0;
    for (int i = 0; i < 10; i++) src[k++] = IDX(0, 2 + i * 4);  /* top edge  */
    for (int i = 0; i < 10; i++) src[k++] = IDX(2 + i * 4, 0);  /* left edge */
    k = 0;
    for (int i = 0; i < 12; i++) rec[k++] = IDX(NZ - 1, 2 + i * 3); /* bottom */
    for (int i = 0; i < 12; i++) rec[k++] = IDX(2 + i * 3, NX - 1); /* right  */
}

/* --- main ------------------------------------------------------------------ */

int main(void)
{
    static double v[NCELL], bg[NCELL], slow[NCELL], T[NCELL], kernel[NCELL];
    static double d_clean[NRAY], d_obs[NRAY];
    static double v_ens[MAX_MEMBERS * NCELL], anom[MAX_MEMBERS * NCELL];
    int src[N_SRC], rec[N_REC];
    Label labels[NCELL];

    inv_seed(20260517ULL);

    ground_truth(v);
    background(bg);
    for (int c = 0; c < NCELL; c++) slow[c] = 1.0 / v[c];
    build_geometry(src, rec);

    puts("======================================================================");
    puts("possibilistic-inversion - C port - end-to-end Eikonal pipeline");
    puts("======================================================================");
    printf("Grid %dx%d (%d cells). Synthetic model: slab + low-V zone + 3 blobs.\n",
           NZ, NX, NCELL);
    printf("Survey: %d sources, %d receivers -> %d travel-time data.\n",
           N_SRC, N_REC, NRAY);

    /* --- forward operators: straight-ray vs Eikonal ----------------------- */
    fmm(slow, 3, 3, T);
    int rz[3] = {36, 36, 20}, rx[3] = {36, 20, 36};
    puts("\nForward operators (source cell 3,3):");
    puts("  receiver     straight-ray   Eikonal(FMM)   difference");
    for (int r = 0; r < 3; r++) {
        double ts = straightray_time(slow, 3, 3, rz[r], rx[r]);
        double te = T[IDX(rz[r], rx[r])];
        printf("  (%2d,%2d)        %8.4f       %8.4f       %+6.2f%%\n",
               rz[r], rx[r], ts, te, 100.0 * (te - ts) / ts);
    }
    ray_path(T, 36, 36, kernel);
    double klen = 0.0;
    for (int c = 0; c < NCELL; c++) klen += kernel[c];
    printf("  Eikonal ray (36,36)->source: kernel path length %.3f.\n", klen);

    /* --- synthetic data: Eikonal forward + noise -------------------------- */
    traveltimes(slow, src, N_SRC, rec, N_REC, d_clean);
    double mean_t = 0.0;
    for (int k = 0; k < NRAY; k++) mean_t += d_clean[k];
    mean_t /= NRAY;
    double noise = 0.005 * mean_t;          /* 0.5% travel-time noise */
    for (int k = 0; k < NRAY; k++)
        d_obs[k] = d_clean[k] + noise * inv_normal();
    printf("\nSynthetic data: mean travel time %.3f, noise sigma %.4f (RMS target).\n",
           mean_t, noise);

    /* --- the inversion: feasible-set sampler ------------------------------ */
    puts("\nRunning the feasible-set sampler (LM inversions + smooth");
    puts("perturbations) - this is the work; expect tens of seconds...");
    int nmem = feasible_set(d_obs, src, rec, noise, v_ens);
    printf("Feasible set: %d data-consistent models assembled.\n", nmem);
    if (nmem < 2) {
        puts("ERROR: feasible set too small to decompose.");
        return 1;
    }

    /* member anomalies = velocity minus the planted background */
    for (int m = 0; m < nmem; m++)
        for (int c = 0; c < NCELL; c++)
            anom[m * NCELL + c] = v_ens[m * NCELL + c] - bg[c];

    /* --- the possibilistic decomposition ---------------------------------- */
    double eps = 0.04;
    decompose(anom, nmem, eps, labels);
    int count[4] = {0, 0, 0, 0};
    for (int c = 0; c < NCELL; c++) count[labels[c]]++;

    printf("\nPossibilistic decomposition (%d-member feasible set, eps = %.2f):\n",
           nmem, eps);
    for (int l = 0; l < 4; l++)
        printf("  %-18s %4d cells  (%4.1f%%)\n", label_name((Label)l),
               count[l], 100.0 * count[l] / NCELL);

    /* --- validation against ground truth ---------------------------------- */
    /* A forced-sign cell is scored two ways: strict (the true anomaly has
     * that sign in that exact cell) and within-resolution (a true anomaly of
     * that sign lies within RES_CELLS - the honest precision of the image). */
    const int RES_CELLS = 2;
    int fh = 0, fh_strict = 0, fh_res = 0;
    int fl = 0, fl_strict = 0, fl_res = 0;
    for (int c = 0; c < NCELL; c++) {
        int z = c / NX, x = c % NX;
        double a_true = v[c] - bg[c];
        if (labels[c] == FORCED_HIGH) {
            fh++;
            if (a_true > 0.0) fh_strict++;
            if (sign_within(v, bg, z, x, 1, RES_CELLS)) fh_res++;
        }
        if (labels[c] == FORCED_LOW) {
            fl++;
            if (a_true < 0.0) fl_strict++;
            if (sign_within(v, bg, z, x, 0, RES_CELLS)) fl_res++;
        }
    }
    puts("\nValidation against ground truth (sign of the true anomaly):");
    printf("              strict (exact cell)   within %d-cell resolution\n",
           RES_CELLS);
    if (fh) printf("  forced-high  %3d/%3d  (%5.1f%%)        %3d/%3d  (%5.1f%%)\n",
                   fh_strict, fh, 100.0 * fh_strict / fh,
                   fh_res, fh, 100.0 * fh_res / fh);
    if (fl) printf("  forced-low   %3d/%3d  (%5.1f%%)        %3d/%3d  (%5.1f%%)\n",
                   fl_strict, fl, 100.0 * fl_strict / fl,
                   fl_res, fl, 100.0 * fl_res / fl);
    printf("  the forced cores are data-certain; the %d measure-dependent\n",
           count[MEASURE_DEPENDENT]);
    puts("  cells are where the regularization choice, not the data, decides.");

    puts("\nPipeline complete: forward -> data -> inversion -> feasible set");
    puts("-> possibilistic decomposition. All modules exercised.");
    return 0;
}
