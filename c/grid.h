/* grid.h - shared grid geometry and Tier-1 bounds.
 *
 * Common definitions for the possibilistic-inversion C port. The grid is
 * row-major [NZ, NX]; cell (z, x) maps to flat index z*NX + x; z increases
 * downward; unit cell spacing h = 1. Velocities are km/s, slowness s = 1/v.
 *
 * Part of the C port of the possibilistic-inversion method - a small set of
 * .c/.h modules mirroring the Python (synthetic_demo.py, eikonal.py, the
 * decomposition layer), pulled together by possibilistic_inversion.c.
 */
#ifndef GRID_H
#define GRID_H

#define NZ    40
#define NX    40
#define NCELL (NZ * NX)

/* Tier-1 frame-independent velocity envelope (geophysical_invariants.md s.1). */
#define VP_MIN 2.0
#define VP_MAX 9.0

/* cell (z, x) -> flat index */
#define IDX(z, x) ((z) * NX + (x))

#endif /* GRID_H */
