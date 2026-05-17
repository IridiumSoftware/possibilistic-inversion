/* decomposition.h - the possibilistic decomposition layer.
 *
 * Mirrors the decomposition of synthetic_demo.py (feasible_interval, classify,
 * decompose). Given an ensemble of feasible models, the per-cell feasible
 * interval [a_min, a_max] and the forced-sign classification.
 *
 * This is the Float64 numerical port; the exact-arithmetic formalization of
 * the same layer (proven properties) is decomposition_exact.jl.
 */
#ifndef DECOMPOSITION_H
#define DECOMPOSITION_H

#include "grid.h"

typedef enum {
    FORCED_HIGH,        /* a_min >  eps  - every feasible model: positive    */
    FORCED_LOW,         /* a_max < -eps  - every feasible model: negative    */
    FORCED_QUIET,       /* interval within +/-eps                            */
    MEASURE_DEPENDENT   /* interval straddles - the sign is not data-forced  */
} Label;

/* Per-cell feasible interval over an ensemble.
 *   ens    : nmodel * NCELL, row-major (model m, cell c) at ens[m*NCELL + c]
 *   amin,
 *   amax   : caller-allocated NCELL arrays, filled with the interval bounds. */
void feasible_interval(const double *ens, int nmodel,
                       double *amin, double *amax);

/* Forced-sign classification of one cell from its feasible interval. */
Label classify(double amin, double amax, double eps);

/* Decompose an ensemble into per-cell forced-sign labels.
 *   labels : caller-allocated NCELL array. */
void decompose(const double *ens, int nmodel, double eps, Label *labels);

/* Human-readable name of a label. */
const char *label_name(Label l);

#endif /* DECOMPOSITION_H */
