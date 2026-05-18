/* eikonal.h - the Eikonal first-arrival forward operator.
 *
 * Mirrors eikonal.py: the faithful nonlinear forward model in which rays bend
 * through the medium - the operator class ZTM/TFM's FMM.cpp implements.
 *
 *   fmm        - first-arrival travel-time field by the Fast Marching Method
 *                (Sethian 1996); plain first-order scheme.
 *   ray_path   - the first-arrival ray as per-cell path length, by steepest
 *                descent on the travel-time field - the Frechet kernel
 *                d t_recv / d slow.
 *
 * Dependency-free: C standard library only (math.h).
 */
#ifndef EIKONAL_H
#define EIKONAL_H

#include "grid.h"

/* First-arrival travel-time field from a point source.
 *   slow : NCELL slowness field (1/v).
 *   sz,sx: source cell.
 *   T    : caller-allocated NCELL array, filled with travel times. */
void fmm(const double *slow, int sz, int sx, double *T);

/* First-arrival ray from receiver (rz,rx) back toward the source: steepest
 * descent on the travel-time field T. kernel (caller-allocated NCELL, zeroed
 * by this call) receives the per-cell path length - one row of the Frechet
 * matrix d t / d slow. */
void ray_path(const double *T, int rz, int rx, double *kernel);

/* All source-receiver travel times (mirrors traveltimes in eikonal.py).
 *   slow      : NCELL slowness field.
 *   src, n_src: source cells as flat indices (z*NX + x).
 *   rec, n_rec: receiver cells as flat indices.
 *   t_out     : caller-allocated n_src*n_rec array; row s*n_rec + r is the
 *               time from source s to receiver r. */
void traveltimes(const double *slow, const int *src, int n_src,
                 const int *rec, int n_rec, double *t_out);

/* Travel times plus the Frechet matrix (mirrors forward in eikonal.py).
 *   t_out : as traveltimes, length n_src*n_rec.
 *   G_out : caller-allocated (n_src*n_rec) x NCELL, row-major; row s*n_rec + r
 *           is the ray-path Frechet kernel for that source-receiver pair. */
void forward(const double *slow, const int *src, int n_src,
             const int *rec, int n_rec, double *t_out, double *G_out);

#endif /* EIKONAL_H */
