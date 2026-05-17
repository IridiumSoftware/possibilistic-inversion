/* straightray.c - the straight-ray forward operator. */

#include "straightray.h"
#include <math.h>

void straight_ray(double z0, double x0, double z1, double x1, double *row)
{
    for (int i = 0; i < NCELL; i++) row[i] = 0.0;

    double len = sqrt((z1 - z0) * (z1 - z0) + (x1 - x0) * (x1 - x0));
    int nstep = (int)(len / 0.2);
    if (nstep < 2) nstep = 2;
    double seg = len / nstep;

    for (int k = 0; k < nstep; k++) {
        double t = (k + 0.5) / nstep;             /* fine sampling of the segment */
        int iz = (int)(z0 + t * (z1 - z0));
        int ix = (int)(x0 + t * (x1 - x0));
        if (iz < 0) iz = 0; else if (iz > NZ - 1) iz = NZ - 1;
        if (ix < 0) ix = 0; else if (ix > NX - 1) ix = NX - 1;
        row[IDX(iz, ix)] += seg;
    }
}

double straightray_time(const double *slow,
                        double z0, double x0, double z1, double x1)
{
    double row[NCELL];
    straight_ray(z0, x0, z1, x1, row);
    double t = 0.0;
    for (int c = 0; c < NCELL; c++) t += row[c] * slow[c];
    return t;
}
