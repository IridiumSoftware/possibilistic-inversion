/* inversion.c - Levenberg-Marquardt inversion + feasible-set sampler.
 *
 * Port of the inversion layer of synthetic_demo_eikonal.py. Two methodological
 * points, both learned the hard way in the Python and carried over verbatim:
 *
 *  1. The LM step is taken in DATA space:
 *         ds = D^-1 G^T (G D^-1 G^T + mu I)^-1 r
 *     with D the diagonal G^T G preconditioner. The bracketed matrix is
 *     NRAY-square (a few hundred), not NCELL-square - so the only linear
 *     algebra needed is one small dense Cholesky solve per damping trial.
 *
 *  2. The feasible set is sampled with SMOOTH perturbation fields. The raw
 *     null space of a ray operator is full of checkerboard modes a line
 *     integral cannot see; perturbing with those would fill the ensemble with
 *     unphysical models and make the decomposition meaningless. Tier-2 of
 *     geophysical_invariants.md (smoothness) is enforced at sampling time.
 *
 * Dependency-free: C standard library + linalg.h.
 */

#include "inversion.h"
#include "eikonal.h"
#include "linalg.h"

#include <math.h>
#include <string.h>

/* --- deterministic RNG (xorshift128+) -------------------------------------
 * Small, dependency-free, reproducible. Quality is ample for synthetic
 * reference models and perturbation fields. */

static unsigned long long rng_s[2] = { 0x853c49e6748fea9bULL,
                                        0xda3e39cb94b95bdbULL };

void inv_seed(unsigned long long seed)
{
    rng_s[0] = seed ? seed : 1ULL;
    rng_s[1] = seed ^ 0x9e3779b97f4a7c15ULL;
    if (rng_s[1] == 0) rng_s[1] = 1ULL;
}

static unsigned long long rng_u64(void)
{
    unsigned long long x = rng_s[0], y = rng_s[1];
    rng_s[0] = y;
    x ^= x << 23;
    rng_s[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
    return rng_s[1] + y;
}

/* uniform double in [0, 1) */
static double rng_uniform(void)
{
    return (double)(rng_u64() >> 11) * (1.0 / 9007199254740992.0);
}

/* one standard-normal draw (Box-Muller) */
double inv_normal(void)
{
    double u1 = rng_uniform(), u2 = rng_uniform();
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

/* --- small helpers --------------------------------------------------------- */

static double clipd(double v, double lo, double hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

/* RMS misfit between two length-n vectors. */
static double rms_diff(const double *a, const double *b, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; i++) { double d = a[i] - b[i]; s += d * d; }
    return sqrt(s / n);
}

/* n passes of periodic 4-neighbour averaging - the smoothing operator shared
 * by random_reference and smooth_field (mirrors np.roll-based smoothing). */
static void smooth_passes(double *f, int npass)
{
    static double tmp[NCELL];
    for (int p = 0; p < npass; p++) {
        for (int z = 0; z < NZ; z++) {
            int zm = (z - 1 + NZ) % NZ, zp = (z + 1) % NZ;
            for (int x = 0; x < NX; x++) {
                int xm = (x - 1 + NX) % NX, xp = (x + 1) % NX;
                tmp[IDX(z, x)] = 0.25 * (f[IDX(zm, x)] + f[IDX(zp, x)]
                                       + f[IDX(z, xm)] + f[IDX(z, xp)]);
            }
        }
        memcpy(f, tmp, sizeof tmp);
    }
}

/* --- random reference model ------------------------------------------------
 * A random depth-gradient velocity plus a heavily-smoothed bump, clipped to
 * the Tier-1 envelope. Diverse starting models for the feasible-set sampler;
 * returned as slowness. (Mirrors random_reference in synthetic_demo.py.) */
static void random_reference(double *slow_out)
{
    static double bump[NCELL];
    double slope = 2.2 + rng_uniform() * (3.6 - 2.2);

    for (int c = 0; c < NCELL; c++) bump[c] = inv_normal();
    smooth_passes(bump, 60);
    double mx = 0.0;
    for (int c = 0; c < NCELL; c++) {
        double a = fabs(bump[c]);
        if (a > mx) mx = a;
    }
    if (mx < 1e-12) mx = 1e-12;

    for (int z = 0; z < NZ; z++) {
        for (int x = 0; x < NX; x++) {
            double v = 4.8 + slope * ((z + 0.5) / NZ)
                     + 0.7 * (bump[IDX(z, x)] / mx);
            v = clipd(v, VP_MIN, VP_MAX);
            slow_out[IDX(z, x)] = 1.0 / v;
        }
    }
}

/* --- smooth perturbation field --------------------------------------------
 * White noise smoothed npass times, normalized to unit RMS. The shape of a
 * physically-plausible feasible-set perturbation. (Mirrors smooth_field in
 * synthetic_demo_eikonal.py.) */
static void smooth_field(double *out, int npass)
{
    for (int c = 0; c < NCELL; c++) out[c] = inv_normal();
    smooth_passes(out, npass);

    double mean = 0.0;
    for (int c = 0; c < NCELL; c++) mean += out[c];
    mean /= NCELL;
    double var = 0.0;
    for (int c = 0; c < NCELL; c++) {
        double d = out[c] - mean;
        var += d * d;
    }
    double sd = sqrt(var / NCELL);
    if (sd < 1e-12) sd = 1e-12;
    for (int c = 0; c < NCELL; c++) out[c] = (out[c] - mean) / sd;
}

/* --- Levenberg-Marquardt inversion ---------------------------------------- */

#define LM_MAX_ITER  18
#define LM_DAMP_TRIES 9

double invert_lm(const double *d_obs, const int *src, const int *rec,
                 const double *slow_ref, double noise, double *v_out)
{
    /* All work arrays static: invert_lm is not reentrant, and NRAY*NCELL is
     * too large for the stack (~2.9 MB for G alone). */
    static double s[NCELL], s_try[NCELL], dinv[NCELL];
    static double G[NRAY * NCELL];
    static double t[NRAY], t_try[NRAY], r[NRAY], y[NRAY];
    static double GDGt[NRAY * NRAY], fac[NRAY * NRAY];

    const double slo = 1.0 / VP_MAX, shi = 1.0 / VP_MIN;

    for (int c = 0; c < NCELL; c++) s[c] = clipd(slow_ref[c], slo, shi);

    traveltimes(s, src, N_SRC, rec, N_REC, t);
    double rms = rms_diff(d_obs, t, NRAY);
    double mu = -1.0;                       /* set from diag(GDGt) on iter 0 */

    for (int iter = 0; iter < LM_MAX_ITER; iter++) {
        if (rms <= noise) break;

        forward(s, src, N_SRC, rec, N_REC, t, G);
        for (int k = 0; k < NRAY; k++) r[k] = d_obs[k] - t[k];
        rms = 0.0;
        for (int k = 0; k < NRAY; k++) rms += r[k] * r[k];
        rms = sqrt(rms / NRAY);
        if (rms <= noise) break;

        /* D = diag(G^T G); dinv its inverse (the LM preconditioner). */
        for (int c = 0; c < NCELL; c++) dinv[c] = 1e-12;
        for (int k = 0; k < NRAY; k++) {
            const double *Gk = &G[k * NCELL];
            for (int c = 0; c < NCELL; c++) dinv[c] += Gk[c] * Gk[c];
        }
        for (int c = 0; c < NCELL; c++) dinv[c] = 1.0 / dinv[c];

        /* GDGt = (G .* dinv) G^T - the NRAY-square data-space matrix. */
        for (int i = 0; i < NRAY; i++) {
            const double *Gi = &G[i * NCELL];
            for (int j = 0; j <= i; j++) {
                const double *Gj = &G[j * NCELL];
                double sum = 0.0;
                for (int c = 0; c < NCELL; c++)
                    sum += Gi[c] * dinv[c] * Gj[c];
                GDGt[i * NRAY + j] = sum;
                GDGt[j * NRAY + i] = sum;
            }
        }

        if (mu < 0.0) {
            double md = 0.0;
            for (int i = 0; i < NRAY; i++) md += GDGt[i * NRAY + i];
            mu = 0.1 * md / NRAY;
            if (mu < 1e-12) mu = 1e-12;
        }

        /* Adaptive damping: shrink mu on a step that reduces the misfit,
         * grow it on one that does not. */
        int improved = 0;
        for (int trial = 0; trial < LM_DAMP_TRIES; trial++) {
            memcpy(fac, GDGt, sizeof GDGt);
            for (int i = 0; i < NRAY; i++) fac[i * NRAY + i] += mu;
            if (chol_factor(fac, NRAY) != 0) { mu *= 3.0; continue; }
            chol_solve(fac, NRAY, r, y);

            /* ds = dinv .* (G^T y);  s_try = clip(s + ds). */
            for (int c = 0; c < NCELL; c++) {
                double gty = 0.0;
                for (int k = 0; k < NRAY; k++) gty += G[k * NCELL + c] * y[k];
                s_try[c] = clipd(s[c] + dinv[c] * gty, slo, shi);
            }
            traveltimes(s_try, src, N_SRC, rec, N_REC, t_try);
            double rms_try = rms_diff(d_obs, t_try, NRAY);

            if (rms_try < rms) {
                memcpy(s, s_try, sizeof s);
                rms = rms_try;
                mu *= 0.4;
                if (mu < 1e-9) mu = 1e-9;
                improved = 1;
                break;
            }
            mu *= 3.0;
        }
        if (!improved) break;               /* no descent direction found */
    }

    for (int c = 0; c < NCELL; c++) v_out[c] = 1.0 / s[c];
    return rms;
}

/* --- feasible-set sampler -------------------------------------------------- */

#define FS_N_BASE  10       /* LM inversions from random references          */
#define FS_N_PERT   7       /* smooth perturbations attempted per base model */
#define FS_PERT_RMS 0.55    /* perturbation amplitude (km/s, before backoff) */
#define FS_TOL      1.3     /* accept a model if its RMS <= FS_TOL * noise   */

int feasible_set(const double *d_obs, const int *src, const int *rec,
                 double noise, double *v_ens_out)
{
    static double slow_ref[NCELL], v_base[NCELL];
    static double shape[NCELL], v_try[NCELL], slow_try[NCELL], t[NRAY];
    static double base_models[FS_N_BASE * NCELL];

    /* backtracking factors: take the largest perturbation still feasible */
    static const double facs[6] = { 1.0, 0.6, 0.35, 0.2, 0.1, 0.05 };

    int count = 0, n_base = 0;

    /* base models: LM inversions from diverse random references */
    for (int b = 0; b < FS_N_BASE && count < MAX_MEMBERS; b++) {
        random_reference(slow_ref);
        double rms = invert_lm(d_obs, src, rec, slow_ref, noise, v_base);
        if (rms <= FS_TOL * noise) {
            memcpy(&v_ens_out[count * NCELL], v_base, NCELL * sizeof(double));
            memcpy(&base_models[n_base * NCELL], v_base, NCELL * sizeof(double));
            count++;
            n_base++;
        }
    }

    /* smooth perturbations around each accepted base model */
    for (int bi = 0; bi < n_base && count < MAX_MEMBERS; bi++) {
        const double *base = &base_models[bi * NCELL];
        for (int p = 0; p < FS_N_PERT && count < MAX_MEMBERS; p++) {
            int npass = 18 + (int)(rng_uniform() * 28.0);   /* in [18, 45] */
            smooth_field(shape, npass);
            for (int c = 0; c < NCELL; c++) shape[c] *= FS_PERT_RMS;

            for (int fi = 0; fi < 6; fi++) {
                for (int c = 0; c < NCELL; c++) {
                    double v = clipd(base[c] + facs[fi] * shape[c],
                                     VP_MIN, VP_MAX);
                    v_try[c] = v;
                    slow_try[c] = 1.0 / v;
                }
                traveltimes(slow_try, src, N_SRC, rec, N_REC, t);
                if (rms_diff(d_obs, t, NRAY) <= FS_TOL * noise) {
                    memcpy(&v_ens_out[count * NCELL], v_try,
                           NCELL * sizeof(double));
                    count++;
                    break;                  /* largest feasible step taken */
                }
            }
        }
    }

    return count;
}
