# SPH Physics Integration Design

**Date:** 2026-02-08
**Status:** Implementation

## Goal

Wire the existing SPH modules (neighbor search, density, pressure, viscosity) into
`Simulation::step()` so particles behave as fluid instead of only falling under gravity.

## Changes

### 1. `crates/sim-core/src/simulation.rs`

- Add `grid: SpatialHashGrid<D>` and `kernel: CubicSplineKernel` as persistent fields.
- Add trait bound `where CubicSplineKernel: SmoothingKernel<D>` on the impl block containing `step()`.
- Rewrite `step()` to follow the standard SPH loop:
  1. Clear accelerations
  2. Build neighbor grid
  3. Compute densities (SPH summation)
  4. Compute pressures (Tait EOS)
  5. Compute pressure forces
  6. Compute viscosity forces
  7. Apply gravity
  8. Integrate (symplectic Euler)
  9. Enforce boundaries

### 2. `crates/sim-core/src/sph/density.rs`

- **Bug fix:** Remove explicit self-contribution (line 27). The spatial hash grid returns
  particle `i` in its own cell, so the neighbor loop already includes `W(0, h)`.
  The current code double-counts self-contribution.

### 3. `crates/common/src/config.rs`

Updated default parameters for SPH correctness:

| Parameter | Old | New | Reasoning |
|-----------|-----|-----|-----------|
| `smoothing_radius` | 0.1 | 0.026 | h = 1.3 * spacing (0.02) |
| `stiffness` | 1000.0 | 50000.0 | B = rho0 * c0^2 / gamma, moderate value |
| `viscosity` | 0.1 | 0.01 | Lower for fluid-like behavior |
| `time_step` | 0.001 | 0.0002 | CFL constraint: dt <= 0.2 * h / c0 |

### 4. `crates/sim-core/src/scene.rs`

- Update `dam_break()` and `double_emitter()` to use tuned configs (via updated defaults).

### 5. `assets/scenes/*.json`

- Mirror parameter changes in JSON scene files.

## Design Decisions

- **Grid cell_size = h** (not 2h): 3x3 adjacent cells at size h covers 3h > 2h kernel support. More efficient than 2h cells.
- **No distance filtering in grid**: Kernel naturally returns 0 for r > 2h. Avoids unnecessary sqrt.
- **Persistent grid field**: Avoids HashMap reallocation each step. `build()` clears and rebuilds in-place.
- **Kernel cutoff for filtering**: Rely on cubic spline returning 0 for q > 2 rather than explicit distance checks.
- **Gravity after SPH forces**: External force added alongside fluid forces. Order within accumulation is irrelevant.

## Known Limitations (deferred)

- No XSPH velocity smoothing
- No Shepard density filtering
- No CFL-based adaptive timestep
- No speed-of-sound parameter in config
- Neighbor search allocates Vec per query
- CubicSplineKernel only supports Dim2
- Symplectic Euler instead of Leapfrog
