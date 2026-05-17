/* possibilistic_inversion.c - master file of the C port.
 *
 * Pulls in the modules - decomposition, straightray, eikonal - and exercises
 * them end to end: builds the synthetic velocity model, runs both forward
 * operators, traces a first-arrival ray, and runs the possibilistic
 * decomposition on a demonstration ensemble.
 *
 * Scope. This port covers the forward operators and the decomposition layer.
 * The inversion (Levenberg-Marquardt) and the feasible-set samplers are not
 * ported here - they need a small linear-algebra module (symmetric
 * eigendecomposition, a linear solve) and are the next C increment. So the
 * ensemble decomposed below is a *demonstration* ensemble - the true model
 * plus smooth perturbations - not the output of an inversion.
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

/* --- main ------------------------------------------------------------------ */

int main(void)
{
    static double v[NCELL], bg[NCELL], slow[NCELL], T[NCELL], kernel[NCELL];
    static double ens[6 * NCELL];
    Label labels[NCELL];

    ground_truth(v);
    background(bg);
    for (int c = 0; c < NCELL; c++) slow[c] = 1.0 / v[c];

    puts("======================================================================");
    puts("possibilistic-inversion - C port - module demonstration");
    puts("======================================================================");
    printf("Grid %dx%d (%d cells). Synthetic model: slab + low-V zone + 3 blobs.\n",
           NZ, NX, NCELL);

    /* --- forward operators: straight-ray vs Eikonal ----------------------- */
    int sz = 3, sx = 3;
    fmm(slow, sz, sx, T);
    int rz[3] = {36, 36, 20}, rx[3] = {36, 20, 36};
    puts("\nForward operators (source cell 3,3):");
    puts("  receiver     straight-ray   Eikonal(FMM)   difference");
    for (int r = 0; r < 3; r++) {
        double ts = straightray_time(slow, sz, sx, rz[r], rx[r]);
        double te = T[IDX(rz[r], rx[r])];
        printf("  (%2d,%2d)        %8.4f       %8.4f       %+6.2f%%\n",
               rz[r], rx[r], ts, te, 100.0 * (te - ts) / ts);
    }
    puts("  (the operators agree to ~1%: Eikonal is lower where ray bending");
    puts("   through fast structure helps, and the residual is first-order");
    puts("   FMM error - the operators are genuinely distinct, not identical)");

    /* --- a first-arrival ray (the Frechet kernel) ------------------------- */
    ray_path(T, 36, 36, kernel);
    double klen = 0.0;
    for (int c = 0; c < NCELL; c++) klen += kernel[c];
    double geom = sqrt((36.0 - sz) * (36.0 - sz) + (36.0 - sx) * (36.0 - sx));
    printf("\nRay (36,36)->source: kernel path length %.3f "
           "(straight-line distance %.3f).\n", klen, geom);

    /* --- decomposition on a demonstration ensemble ------------------------ */
    /* member 0: the true anomaly; members 1..5: true + a smooth perturbation. */
    const double pert[5][4] = {           /* z0, x0, sigma, amplitude (km/s) */
        {12.0, 14.0, 6.0,  0.35},
        {26.0, 22.0, 7.0, -0.30},
        {18.0, 30.0, 5.0,  0.32},
        {32.0, 12.0, 6.0, -0.28},
        { 9.0, 20.0, 5.5,  0.30},
    };
    for (int c = 0; c < NCELL; c++) ens[c] = v[c] - bg[c];     /* member 0 */
    for (int m = 1; m < 6; m++) {
        const double *p = pert[m - 1];
        for (int z = 0; z < NZ; z++)
            for (int x = 0; x < NX; x++) {
                double bump = p[3] * gauss(cell_z(z), cell_x(x), p[0], p[1], p[2]);
                ens[m * NCELL + IDX(z, x)] = (v[IDX(z, x)] + bump) - bg[IDX(z, x)];
            }
    }

    double eps = 0.04;
    decompose(ens, 6, eps, labels);
    int count[4] = {0, 0, 0, 0};
    for (int c = 0; c < NCELL; c++) count[labels[c]]++;

    printf("\nPossibilistic decomposition (6-member demonstration ensemble, "
           "eps = %.2f):\n", eps);
    for (int l = 0; l < 4; l++)
        printf("  %-18s %4d cells\n", label_name((Label)l), count[l]);

    puts("\nAll three modules exercised: straightray, eikonal, decomposition.");
    return 0;
}
