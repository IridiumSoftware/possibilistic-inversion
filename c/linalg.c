/* linalg.c - dense Cholesky factorization and solve. */

#include "linalg.h"
#include <math.h>

/* Standard column-by-column Cholesky. A is overwritten: its lower triangle
 * becomes L with A = L L^T. The upper triangle is left as input garbage and
 * never read by chol_solve. Returns 1 if a non-positive pivot appears - i.e.
 * A was not positive-definite. */
int chol_factor(double *A, int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = A[i * n + j];
            for (int k = 0; k < j; k++)
                sum -= A[i * n + k] * A[j * n + k];
            if (i == j) {
                if (sum <= 0.0) return 1;
                A[i * n + j] = sqrt(sum);
            } else {
                A[i * n + j] = sum / A[j * n + j];
            }
        }
    }
    return 0;
}

/* Solve L L^T x = b: forward substitution L y = b, then back substitution
 * L^T x = y. x may alias neither L nor b. */
void chol_solve(const double *L, int n, const double *b, double *x)
{
    for (int i = 0; i < n; i++) {
        double sum = b[i];
        for (int k = 0; k < i; k++)
            sum -= L[i * n + k] * x[k];
        x[i] = sum / L[i * n + i];
    }
    for (int i = n - 1; i >= 0; i--) {
        double sum = x[i];
        for (int k = i + 1; k < n; k++)
            sum -= L[k * n + i] * x[k];
        x[i] = sum / L[i * n + i];
    }
}
