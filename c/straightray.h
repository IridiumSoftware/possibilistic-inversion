/* straightray.h - the straight-ray forward operator.
 *
 * Mirrors build_G of synthetic_demo.py: the straight-ray (line-integral)
 * tomography operator. A ray from a source to a receiver is the straight
 * segment between them; the operator row is the per-cell path length.
 *
 * Exact only for a homogeneous medium; the faithful (bending) operator is
 * eikonal.h. The two share the same downstream decomposition - that is the
 * forward-model-agnostic property the method rests on.
 */
#ifndef STRAIGHTRAY_H
#define STRAIGHTRAY_H

#include "grid.h"

/* Accumulate the straight ray from (z0,x0) to (z1,x1) as per-cell path
 * lengths into row (a caller-allocated NCELL array, zeroed by this call).
 * Coordinates are in cell units. */
void straight_ray(double z0, double x0, double z1, double x1, double *row);

/* Straight-ray first-arrival travel time through a slowness field:
 * the path integral sum_c row[c] * slow[c]. */
double straightray_time(const double *slow,
                        double z0, double x0, double z1, double x1);

#endif /* STRAIGHTRAY_H */
