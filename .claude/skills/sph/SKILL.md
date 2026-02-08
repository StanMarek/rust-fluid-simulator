---
name: sph
description: "**SPH Fluid Simulation Builder**: Create, debug, and optimize Smoothed Particle Hydrodynamics (SPH) simulations in Python. Use this skill whenever the user mentions SPH, smoothed particle hydrodynamics, particle-based fluid simulation, meshless methods, Lagrangian fluid solvers, dam break simulation, free-surface flows, WCSPH, ISPH, kernel interpolation for fluids, or wants to simulate fluids/solids/multiphase flows without a mesh. Also trigger when users ask about implementing fluid simulations from scratch, particle methods for CFD, or want to visualize particle-based physics. Even if the user just says 'simulate water' or 'fluid simulation' or 'particle physics simulation', this skill likely applies."
---

# SPH Simulation Builder

Build working Smoothed Particle Hydrodynamics simulations — from simple 2D dam breaks to multi-phase 3D flows with GPU acceleration.

## When to use this skill

This skill helps you write SPH code that actually works. SPH is deceptively simple in concept (particles carry fluid properties, interact through kernel functions) but full of subtle numerical traps. This skill encodes the practical knowledge that prevents you from spending hours debugging pressure explosions, particle clumping, or leaking boundaries.

Use it when the user wants to:

- Build an SPH simulation from scratch (any language, but Python examples provided)
- Debug an existing SPH simulation (pressure noise, instability, boundary issues)
- Choose the right SPH variant for their problem (WCSPH vs ISPH vs δ-SPH)
- Optimize performance (neighbor search, GPU acceleration)
- Implement specific SPH features (multi-phase, free surface, solid coupling)

## Architecture of an SPH simulation

Every SPH simulation follows this loop. Understanding this structure helps you organize code and find bugs:

```
┌─────────────────────────────────────────┐
│  1. NEIGHBOR SEARCH  (20-30% of cost)   │
│     Spatial hash → find particles       │
│     within kernel radius κh             │
├─────────────────────────────────────────┤
│  2. DENSITY                              │
│     ρ_i = Σ_j m_j W(r_ij, h)           │
│     or continuity: dρ/dt = -ρ ∇·v      │
├─────────────────────────────────────────┤
│  3. PRESSURE (from equation of state)    │
│     Tait: p = B[(ρ/ρ₀)^γ - 1]          │
│     or solve Poisson equation (ISPH)     │
├─────────────────────────────────────────┤
│  4. FORCES                               │
│     Pressure: -Σ m_j(p_i/ρ² + p_j/ρ²)∇W│
│     Viscosity: artificial + laminar      │
│     External: gravity, etc.              │
├─────────────────────────────────────────┤
│  5. TIME INTEGRATION                     │
│     Leapfrog or Velocity Verlet          │
│     Δt ≤ CFL × h / (c₀ + |v|_max)      │
├─────────────────────────────────────────┤
│  6. CORRECTIONS (optional but important) │
│     XSPH velocity smoothing             │
│     Shepard density filter               │
│     δ-SPH density diffusion             │
└─────────────────────────────────────────┘
```

## Quick-start defaults

For a first implementation, use these defaults — they produce a working simulation before you start tuning:

| Parameter | Value | Why |
|-----------|-------|-----|
| Kernel | Cubic spline | Simple, well-understood, good enough to start |
| Smoothing ratio η | 1.3 (h = 1.3 × Δx) | ~40 neighbors in 2D, good balance |
| EOS | Tait, γ=7 | Standard for water-like fluids |
| Sound speed c₀ | 10 × v_max | Keeps density fluctuation < 1% |
| Time integrator | Leapfrog | Symplectic, second-order, simple |
| CFL | 0.2 | Safe starting point for WCSPH |
| Artificial viscosity α | 0.03 | Moderate damping, tune from here |
| XSPH ε | 0.5 | Reduces particle disorder |
| Boundary | Ghost particles | Most accurate for walls |

## Mathematical foundations

For the full equation set with all kernel formulas and operator derivations, read `references/math_reference.md`. Key equations summarized here.

### Kernel approximation
Any field quantity f at position r is approximated by:
```
f(r) ≈ Σ_j (m_j / ρ_j) f_j W(r - r_j, h)
```
The kernel W must satisfy: normalization (∫W dr = 1), compact support (W=0 for |r|>κh), and symmetry.

### Gradient (symmetric form — prefer this)
```
∇f_i ≈ Σ_j m_j (f_i/ρ_i² + f_j/ρ_j²) ∇W_ij
```
This form conserves linear and angular momentum.

### Momentum equation
```
dv_i/dt = -Σ_j m_j (p_i/ρ_i² + p_j/ρ_j²) ∇W_ij + viscous + gravity
```

### Tait equation of state
```
p = B × [(ρ/ρ₀)^γ - 1]      where B = ρ₀ c₀² / γ
```
Set c₀ to ~10× the maximum expected velocity. Too low → large density oscillations. Too high → tiny time steps.

## Choosing your SPH variant

Read `references/sph_variants.md` for the full comparison. Decision tree:

```
What are you simulating?
│
├─ Free-surface flow (dam break, waves, sloshing)?
│  └─ WCSPH with δ-SPH density diffusion
│
├─ Smooth internal flow (pipe, cavity)?
│  └─ ISPH or WCSPH with Shepard correction
│
├─ Astrophysics (stars, galaxies)?
│  └─ WCSPH, adaptive h, Wendland C4+, octree search
│
├─ Solid mechanics (impact, fracture)?
│  └─ Total Lagrangian SPH with damage model
│
├─ Multi-phase (oil-water, bubbles)?
│  └─ Multi-phase WCSPH with interface handling
│
└─ Not sure / general purpose?
   └─ WCSPH + cubic spline + Tait EOS
```

## Implementation guide

### Step 1: Kernel functions

Start with cubic spline. For production, switch to Wendland C2 (avoids pairing instability). Full implementations with gradients are in `scripts/kernels.py`.

```python
def cubic_spline_2d(r, h):
    q = r / h
    sigma = 10.0 / (7.0 * np.pi * h**2)
    if q <= 1.0:
        return sigma * (1.0 - 1.5*q**2 + 0.75*q**3)
    elif q <= 2.0:
        return sigma * 0.25 * (2.0 - q)**3
    return 0.0
```

If particles clump in pairs → pairing instability → switch to Wendland kernel.

### Step 2: Neighbor search

Cell-linked list is the simplest O(N) approach. Implementation in `scripts/neighbor_search.py`.

Set `cell_size = κh` (κ = kernel support radius, typically 2). This ensures all neighbors fall within adjacent cells.

### Step 3: Core SPH loop

Compute density FIRST, then pressure, then forces. Never compute forces with stale density — this causes pressure explosions.

A complete working example is in `scripts/sph_2d_dambreak.py`.

### Step 4: Time integration

Leapfrog (symplectic, simple):
```python
v_half = v + 0.5 * dt * a
pos_new = pos + dt * v_half
# ... recompute accelerations with pos_new ...
v_new = v_half + 0.5 * dt * a_new
```

All three time step constraints must be satisfied:
```
dt_cfl   = CFL × h / (c₀ + v_max)
dt_visc  = 0.125 × h² / ν
dt_force = 0.25 × √(h / |a|_max)
dt = min(dt_cfl, dt_visc, dt_force)
```

### Step 5: Boundary handling

Ghost particles are the most accurate. For each wall: mirror fluid particles, set ghost velocity = -fluid velocity (no-slip), and include ghosts in SPH summations. Use at least 2-3 layers of boundary particles.

Read `references/boundary_methods.md` for detailed techniques including corners and curved surfaces.

## Debugging guide

SPH simulations fail in characteristic ways:

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Particles explode outward | c₀ too low → huge pressure | Increase c₀ to 10× v_max |
| Particles clump in pairs | Pairing instability | Switch to Wendland kernel |
| Pressure oscillates wildly | Density noise in WCSPH | Add δ-SPH diffusion (δ=0.1) |
| Particles leak through walls | Insufficient boundary layers | Add ghost layers, reduce dt |
| Energy grows without bound | Time step too large | Reduce CFL to 0.1 |
| Simulation too slow | O(N²) neighbor search | Use cell-linked list |
| Flow too viscous | α_visc too high | Reduce α toward 0.01 |
| Asymmetric flow | Particle arrangement | Use regular lattice, not random |
| Void regions forming | Tensile instability | Add artificial pressure or XSPH |

## Performance optimization

For >10K particles, priorities:

1. **Neighbor search** (biggest win): Cell-linked list → O(N)
2. **NumPy vectorization**: Replace Python loops with array ops → 10-50×
3. **Numba JIT**: `@numba.jit(nopython=True)` on hot loops → 10-50×
4. **GPU**: For >100K particles, use CuPy or CUDA → 50-200×

Read `references/performance.md` for GPU patterns and memory layout advice (SoA vs AoS).

## Reference files

| File | Contents | When to read |
|------|----------|-------------|
| `references/math_reference.md` | Complete equations, kernel formulas, all SPH operators | Building from scratch, debugging math |
| `references/sph_variants.md` | WCSPH vs ISPH vs δ-SPH, hybrid methods | Choosing approach |
| `references/boundary_methods.md` | Ghost particles, repulsive forces, corners | Boundary implementation |
| `references/performance.md` | GPU patterns, memory layout, vectorization | Speed optimization |
| `references/applications.md` | Domain-specific setups and parameter recipes | Specific simulation types |
| `references/multiphase.md` | Multi-phase SPH, interface, surface tension | Multiple fluids |
| `references/software_guide.md` | DualSPHysics, PySPH, SPHinXsys comparison | Using existing libraries |
| `references/validation.md` | Test cases, convergence studies, benchmarks | Verifying correctness |

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/sph_2d_dambreak.py` | Complete working 2D dam break |
| `scripts/kernels.py` | All kernel functions with gradients |
| `scripts/neighbor_search.py` | Cell-linked list implementation |

## Common parameter recipes

**Dam break** (free-surface, water-like):
Wendland C2, η=1.3, γ=7, c₀=10×√(gH), α=0.02, XSPH ε=0.5, CFL=0.2, ghost boundaries (3 layers), δ-SPH δ=0.1.

**Pipe/channel flow** (internal, viscous):
Cubic spline, η=1.2, physical viscosity, α=0.01, no XSPH, CFL=0.3, periodic BCs in flow direction.

**High-speed impact** (solid mechanics):
Wendland C4, η=1.5, Total Lagrangian, damage model, α=0.05, β=0.05, CFL=0.1.

**Astrophysics** (gravitational, compressible):
Wendland C4/C6, adaptive h, ideal gas EOS p=(γ-1)ρe, octree search, α=0.5-1.0, adaptive individual time stepping.
