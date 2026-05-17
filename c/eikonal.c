/* eikonal.c - the Eikonal first-arrival forward operator (Fast Marching). */

#include "eikonal.h"
#include <math.h>

#define BIG 1.0e30

/* --- binary min-heap of (time, cell) --------------------------------------
 * The narrow band of the Fast Marching Method. Lazy deletion: a cell may be
 * pushed several times as neighbours freeze; stale entries are skipped on
 * pop. A cell is pushed at most once per neighbour relaxation (<= 4 NCELL
 * total), so 8 * NCELL capacity is safe. */

typedef struct { double t; int idx; } HeapItem;

static HeapItem heap_buf[8 * NCELL];
static int heap_n;

static void heap_clear(void) { heap_n = 0; }

static void heap_push(double t, int idx)
{
    int i = heap_n++;
    heap_buf[i].t = t;
    heap_buf[i].idx = idx;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (heap_buf[p].t <= heap_buf[i].t) break;
        HeapItem tmp = heap_buf[p]; heap_buf[p] = heap_buf[i]; heap_buf[i] = tmp;
        i = p;
    }
}

static HeapItem heap_pop(void)
{
    HeapItem top = heap_buf[0];
    heap_buf[0] = heap_buf[--heap_n];
    int i = 0;
    while (1) {
        int l = 2 * i + 1, r = 2 * i + 2, s = i;
        if (l < heap_n && heap_buf[l].t < heap_buf[s].t) s = l;
        if (r < heap_n && heap_buf[r].t < heap_buf[s].t) s = r;
        if (s == i) break;
        HeapItem tmp = heap_buf[s]; heap_buf[s] = heap_buf[i]; heap_buf[i] = tmp;
        i = s;
    }
    return top;
}

/* --- first-order Eikonal update -------------------------------------------
 * With the smaller frozen upwind neighbour time on each axis (tz, tx) and
 * slowness s, solve (T-tz)^2 + (T-tx)^2 = s^2 (cell spacing h = 1); fall back
 * to the one-sided T = min(tz,tx) + s when the two-sided root is non-causal. */

static double eik_update(const double *T, const char *frozen,
                         double s, int z, int x)
{
    double tz = BIG, tx = BIG;
    if (z > 0      && frozen[IDX(z-1, x)] && T[IDX(z-1, x)] < tz) tz = T[IDX(z-1, x)];
    if (z < NZ - 1 && frozen[IDX(z+1, x)] && T[IDX(z+1, x)] < tz) tz = T[IDX(z+1, x)];
    if (x > 0      && frozen[IDX(z, x-1)] && T[IDX(z, x-1)] < tx) tx = T[IDX(z, x-1)];
    if (x < NX - 1 && frozen[IDX(z, x+1)] && T[IDX(z, x+1)] < tx) tx = T[IDX(z, x+1)];

    if (tz >= BIG) return tx + s;
    if (tx >= BIG) return tz + s;

    double disc = 2.0 * s * s - (tz - tx) * (tz - tx);
    if (disc >= 0.0) {
        double cand = 0.5 * (tz + tx + sqrt(disc));
        double hi = tz > tx ? tz : tx;
        if (cand >= hi) return cand;            /* causal: root is upwind */
    }
    return (tz < tx ? tz : tx) + s;
}

void fmm(const double *slow, int sz, int sx, double *T)
{
    char frozen[NCELL];
    for (int i = 0; i < NCELL; i++) { T[i] = BIG; frozen[i] = 0; }

    heap_clear();
    int s0 = IDX(sz, sx);
    T[s0] = 0.0;
    heap_push(0.0, s0);

    static const int dz[4] = {1, -1, 0, 0};
    static const int dx[4] = {0, 0, 1, -1};

    while (heap_n > 0) {
        HeapItem it = heap_pop();
        int idx = it.idx;
        if (frozen[idx]) continue;              /* stale entry */
        frozen[idx] = 1;                        /* T[idx] is now final */
        int z = idx / NX, x = idx % NX;
        for (int d = 0; d < 4; d++) {
            int nz = z + dz[d], nx = x + dx[d];
            if (nz < 0 || nz >= NZ || nx < 0 || nx >= NX) continue;
            int n = IDX(nz, nx);
            if (frozen[n]) continue;
            double cand = eik_update(T, frozen, slow[n], nz, nx);
            if (cand < T[n]) { T[n] = cand; heap_push(cand, n); }
        }
    }
}

/* --- ray tracing ----------------------------------------------------------- */

static double sample(const double *T, double z, double x)
{
    if (z < 0.0) z = 0.0; else if (z > NZ - 1) z = NZ - 1;
    if (x < 0.0) x = 0.0; else if (x > NX - 1) x = NX - 1;
    int z0 = (int)z, x0 = (int)x;
    int z1 = z0 + 1 < NZ ? z0 + 1 : z0;
    int x1 = x0 + 1 < NX ? x0 + 1 : x0;
    double fz = z - z0, fx = x - x0;
    return (1 - fz) * (1 - fx) * T[IDX(z0, x0)]
         + (1 - fz) * fx       * T[IDX(z0, x1)]
         + fz       * (1 - fx) * T[IDX(z1, x0)]
         + fz       * fx       * T[IDX(z1, x1)];
}

void ray_path(const double *T, int rz, int rx, double *kernel)
{
    for (int i = 0; i < NCELL; i++) kernel[i] = 0.0;

    double z = rz, x = rx, step = 0.4;
    for (int it = 0; it < 2000; it++) {
        double here = sample(T, z, x);
        double gz = sample(T, z + 0.5, x) - sample(T, z - 0.5, x);
        double gx = sample(T, z, x + 0.5) - sample(T, z, x - 0.5);
        double gn = sqrt(gz * gz + gx * gx);
        int iz = (int)(z + 0.5), ix = (int)(x + 0.5);
        if (iz < 0) iz = 0; else if (iz > NZ - 1) iz = NZ - 1;
        if (ix < 0) ix = 0; else if (ix > NX - 1) ix = NX - 1;
        if (here <= step || gn < 1e-12) {        /* arrived at the source */
            kernel[IDX(iz, ix)] += here > 0.0 ? here : 0.0;
            break;
        }
        kernel[IDX(iz, ix)] += step;
        z -= step * gz / gn;                     /* downhill on T */
        x -= step * gx / gn;
    }
}
