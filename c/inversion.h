/* inversion.h - the nonlinear inversion and the feasible-set sampler.
 *
 * Mirrors synthetic_demo_eikonal.py: a Levenberg-Marquardt inversion of
 * Eikonal travel-time data, and a feasible-set sampler that builds an
 * ensemble of data-consistent models. This is the C port of the inversion
 * layer - the piece the forward-operator + decomposition increment left open.
 *
 * The feasible set is what the possibilistic decomposition operates on:
 * every member fits the data within noise, so a feature present in all of
 * them is data-forced and one present in only some is measure-dependent.
 *
 * Dependency-free: C standard library + linalg.h (dense Cholesky).
 */
#ifndef INVERSION_H
#define INVERSION_H

#include "grid.h"

/* Survey geometry for the Eikonal demonstration: sources and receivers on
 * the grid boundary. NRAY source-receiver pairs - the data vector length.
 * Matches synthetic_demo_eikonal.py (20 sources, 24 receivers). */
#define N_SRC 20
#define N_REC 24
#define NRAY  (N_SRC * N_REC)

/* Upper bound on the feasible-set ensemble size feasible_set assembles. */
#define MAX_MEMBERS 96

/* Seed the deterministic RNG. Same seed -> identical run (reproducibility). */
void inv_seed(unsigned long long seed);

/* One draw from the standard normal - exposed so the caller can build
 * noisy synthetic data from the same deterministic stream. */
double inv_normal(void);

/* Levenberg-Marquardt inversion of Eikonal travel-time data.
 *   d_obs    : NRAY observed travel times.
 *   src, rec : survey geometry as flat cell indices (N_SRC, N_REC long).
 *   slow_ref : NCELL starting slowness model.
 *   noise    : data noise level (RMS) - the misfit target.
 *   v_out    : caller-allocated NCELL, filled with the recovered velocity.
 * Returns the achieved data RMS misfit. */
double invert_lm(const double *d_obs, const int *src, const int *rec,
                 const double *slow_ref, double noise, double *v_out);

/* Build a feasible-set ensemble: LM inversions from diverse random reference
 * models, each accepted model then perturbed with smooth fields kept inside
 * the noise band. Diversity of references is what stops the ensemble from
 * certifying its own shared regularization bias.
 *   v_ens_out : caller-allocated MAX_MEMBERS*NCELL; row m holds member m as a
 *               velocity model (row-major, ens[m*NCELL + c]).
 * Returns the number of members written. */
int feasible_set(const double *d_obs, const int *src, const int *rec,
                 double noise, double *v_ens_out);

#endif /* INVERSION_H */
