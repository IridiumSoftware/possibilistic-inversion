/* decomposition.c - the possibilistic decomposition layer. */

#include "decomposition.h"

void feasible_interval(const double *ens, int nmodel,
                       double *amin, double *amax)
{
    for (int c = 0; c < NCELL; c++) {
        double lo = ens[c], hi = ens[c];          /* model 0 */
        for (int m = 1; m < nmodel; m++) {
            double v = ens[m * NCELL + c];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        amin[c] = lo;
        amax[c] = hi;
    }
}

Label classify(double amin, double amax, double eps)
{
    /* Mutually exclusive, exhaustive (proven in decomposition_exact.jl, T1). */
    if (amin > eps)                      return FORCED_HIGH;
    if (amax < -eps)                     return FORCED_LOW;
    if (amin >= -eps && amax <= eps)     return FORCED_QUIET;
    return MEASURE_DEPENDENT;
}

void decompose(const double *ens, int nmodel, double eps, Label *labels)
{
    double amin[NCELL], amax[NCELL];
    feasible_interval(ens, nmodel, amin, amax);
    for (int c = 0; c < NCELL; c++)
        labels[c] = classify(amin[c], amax[c], eps);
}

const char *label_name(Label l)
{
    switch (l) {
        case FORCED_HIGH:  return "forced-high";
        case FORCED_LOW:   return "forced-low";
        case FORCED_QUIET: return "forced-quiet";
        default:           return "measure-dependent";
    }
}
