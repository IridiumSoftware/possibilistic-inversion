/* linalg.h - minimal dense linear algebra for the inversion.
 *
 * The Levenberg-Marquardt step is a symmetric positive-definite solve
 * (G D^-1 G^T + mu I) y = r in data space - a few-hundred-square system.
 * Cholesky factorization handles it; that is all the linear algebra the
 * Eikonal inversion needs. Dependency-free.
 */
#ifndef LINALG_H
#define LINALG_H

/* In-place Cholesky factorization of a symmetric positive-definite matrix.
 * A is n*n row-major; on success its lower triangle holds L with A = L L^T.
 * Returns 0 on success, 1 if A is not positive-definite. */
int chol_factor(double *A, int n);

/* Solve A x = b given the factor L (lower triangle) from chol_factor. */
void chol_solve(const double *L, int n, const double *b, double *x);

#endif /* LINALG_H */
