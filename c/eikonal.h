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

#endif /* EIKONAL_H */
