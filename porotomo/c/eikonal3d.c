/* eikonal3d.c - 3D first-arrival eikonal forward operator (Fast Marching).
 *
 * 3D extension of c/eikonal.c for the PoroTomo inversion. Differences from
 * the 2D version:
 *   - runtime grid dimensions (nz, ny, nx) instead of compile-time NCELL,
 *     so one shared library serves any grid; buffers are malloc'd per call;
 *   - 6-neighbour first-order update solving
 *       sum_axes (T - t_axis)^2 = s^2
 *     over the causal subset of upwind axis times (3-, 2-, then 1-sided);
 *   - exported with a flat C ABI for ctypes (see porotomo/eikonal3d.py).
 *
 * CONVENTIONS:
 *   - T and slow are C-order arrays indexed idx = (z*ny + y)*nx + x.
 *   - slow is slowness in seconds PER CELL (s/m times cell_m): the grid
 *     spacing is unity here; all metric scaling lives in the caller.
 *   - Ray tracing / Frechet rows are NOT done here: the Python wrapper
 *     back-traces receivers through T in a vectorised batch, which is fast
 *     enough and keeps this file minimal.
 *
 * Dependency-free: C standard library only.
 *
 * Build (see porotomo/eikonal3d.py, which runs this automatically):
 *   cc -O2 -shared -fPIC -o eikonal3d.so eikonal3d.c
 */

#include <stdlib.h>
#include <math.h>

#define BIG 1.0e30

/* --- binary min-heap of (time, cell), lazy deletion ---------------------- */

typedef struct { double t; int idx; } HeapItem;

typedef struct {
    HeapItem *buf;
    int n;
} Heap;

static void heap_push(Heap *h, double t, int idx)
{
    int i = h->n++;
    h->buf[i].t = t;
    h->buf[i].idx = idx;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h->buf[p].t <= h->buf[i].t) break;
        HeapItem tmp = h->buf[p]; h->buf[p] = h->buf[i]; h->buf[i] = tmp;
        i = p;
    }
}

static HeapItem heap_pop(Heap *h)
{
    HeapItem top = h->buf[0];
    h->buf[0] = h->buf[--h->n];
    int i = 0;
    while (1) {
        int l = 2 * i + 1, r = 2 * i + 2, s = i;
        if (l < h->n && h->buf[l].t < h->buf[s].t) s = l;
        if (r < h->n && h->buf[r].t < h->buf[s].t) s = r;
        if (s == i) break;
        HeapItem tmp = h->buf[s]; h->buf[s] = h->buf[i]; h->buf[i] = tmp;
        i = s;
    }
    return top;
}

/* --- first-order 3D Eikonal update ---------------------------------------
 * Upwind axis times a[0..k-1] (the smaller frozen neighbour per axis, only
 * axes that have one), sorted ascending. Try the m-sided solution for
 * m = k..1:  m T^2 - 2 S1 T + (S2 - s^2) = 0,  S1 = sum a_i, S2 = sum a_i^2
 * over the first m; accept the + root if it is causal (>= a[m-1]). */

static double eik_update3(const double *T, const char *frozen, double s,
                          int z, int y, int x, int nz, int ny, int nx)
{
    double a[3];
    int k = 0;
    double t;

    t = BIG;
    if (z > 0      && frozen[((z-1)*ny + y)*nx + x] && T[((z-1)*ny + y)*nx + x] < t)
        t = T[((z-1)*ny + y)*nx + x];
    if (z < nz - 1 && frozen[((z+1)*ny + y)*nx + x] && T[((z+1)*ny + y)*nx + x] < t)
        t = T[((z+1)*ny + y)*nx + x];
    if (t < BIG) a[k++] = t;

    t = BIG;
    if (y > 0      && frozen[(z*ny + y-1)*nx + x] && T[(z*ny + y-1)*nx + x] < t)
        t = T[(z*ny + y-1)*nx + x];
    if (y < ny - 1 && frozen[(z*ny + y+1)*nx + x] && T[(z*ny + y+1)*nx + x] < t)
        t = T[(z*ny + y+1)*nx + x];
    if (t < BIG) a[k++] = t;

    t = BIG;
    if (x > 0      && frozen[(z*ny + y)*nx + x-1] && T[(z*ny + y)*nx + x-1] < t)
        t = T[(z*ny + y)*nx + x-1];
    if (x < nx - 1 && frozen[(z*ny + y)*nx + x+1] && T[(z*ny + y)*nx + x+1] < t)
        t = T[(z*ny + y)*nx + x+1];
    if (t < BIG) a[k++] = t;

    /* insertion sort, k <= 3 */
    if (k >= 2 && a[1] < a[0]) { t = a[0]; a[0] = a[1]; a[1] = t; }
    if (k == 3 && a[2] < a[1]) { t = a[1]; a[1] = a[2]; a[2] = t;
        if (a[1] < a[0]) { t = a[0]; a[0] = a[1]; a[1] = t; } }

    for (int m = k; m >= 2; m--) {
        double S1 = 0.0, S2 = 0.0;
        for (int i = 0; i < m; i++) { S1 += a[i]; S2 += a[i] * a[i]; }
        double disc = S1 * S1 - (double)m * (S2 - s * s);
        if (disc >= 0.0) {
            double cand = (S1 + sqrt(disc)) / (double)m;
            if (cand >= a[m - 1])            /* causal: root upwind of all used */
                return cand;
        }
    }
    return a[0] + s;                         /* one-sided fallback */
}

/* --- Fast Marching --------------------------------------------------------
 * First-arrival travel-time field from a point source.
 *   slow      : nz*ny*nx slowness (s per cell), C-order.
 *   sz,sy,sx  : source cell.
 *   ball_r    : source-ball radius in cells. The first-order update cannot
 *               represent the wavefront curvature near the point source, and
 *               that error propagates outward (~5% mean in 3D). Cells within
 *               ball_r of the source are therefore seeded analytically with
 *               T = dist * (s_src + s_cell)/2 before marching. 0 disables.
 *   T         : caller-allocated nz*ny*nx array, filled with travel times.
 * Returns 0 on success, 1 on allocation failure. */

int fmm3d(const double *slow, int nz, int ny, int nx,
          int sz, int sy, int sx, int ball_r, double *T)
{
    const long n = (long)nz * ny * nx;
    char *frozen = (char *)calloc((size_t)n, 1);
    /* each cell is pushed at most once per neighbour relaxation, plus the
     * source-ball seeds: <= 7n + ball volume */
    HeapItem *buf = (HeapItem *)malloc(sizeof(HeapItem) * (size_t)(8 * n + 8));
    if (!frozen || !buf) { free(frozen); free(buf); return 1; }
    Heap h = { buf, 0 };

    for (long i = 0; i < n; i++) T[i] = BIG;
    const int s_idx = (sz * ny + sy) * nx + sx;
    const double s_src = slow[s_idx];
    T[s_idx] = 0.0;
    heap_push(&h, 0.0, s_idx);

    /* analytic source-ball seeds; exact values are never overwritten because
     * relaxation only lowers T and first-order updates are >= the local
     * two-point analytic time */
    for (int z = sz - ball_r; z <= sz + ball_r; z++) {
        if (z < 0 || z >= nz) continue;
        for (int y = sy - ball_r; y <= sy + ball_r; y++) {
            if (y < 0 || y >= ny) continue;
            for (int x = sx - ball_r; x <= sx + ball_r; x++) {
                if (x < 0 || x >= nx) continue;
                const double dist = sqrt((double)((z - sz) * (z - sz)
                                                  + (y - sy) * (y - sy)
                                                  + (x - sx) * (x - sx)));
                if (dist > (double)ball_r || dist == 0.0) continue;
                const int idx = (z * ny + y) * nx + x;
                const double t0 = dist * 0.5 * (s_src + slow[idx]);
                if (t0 < T[idx]) {
                    T[idx] = t0;
                    heap_push(&h, t0, idx);
                }
            }
        }
    }

    static const int dz[6] = { 1, -1, 0, 0, 0, 0 };
    static const int dy[6] = { 0, 0, 1, -1, 0, 0 };
    static const int dx[6] = { 0, 0, 0, 0, 1, -1 };

    while (h.n > 0) {
        HeapItem it = heap_pop(&h);
        if (frozen[it.idx]) continue;        /* stale lazy-deleted entry */
        frozen[it.idx] = 1;
        const int z = it.idx / (ny * nx);
        const int rem = it.idx % (ny * nx);
        const int y = rem / nx;
        const int x = rem % nx;
        for (int j = 0; j < 6; j++) {
            const int cz = z + dz[j], cy = y + dy[j], cx = x + dx[j];
            if (cz < 0 || cz >= nz || cy < 0 || cy >= ny || cx < 0 || cx >= nx)
                continue;
            const int cidx = (cz * ny + cy) * nx + cx;
            if (frozen[cidx]) continue;
            const double cand = eik_update3(T, frozen, slow[cidx],
                                            cz, cy, cx, nz, ny, nx);
            if (cand < T[cidx]) {
                T[cidx] = cand;
                heap_push(&h, cand, cidx);
            }
        }
    }
    free(frozen);
    free(buf);
    return 0;
}
