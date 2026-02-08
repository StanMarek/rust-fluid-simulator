# SPH Variants Reference Guide

A comprehensive comparison of different Smoothed Particle Hydrodynamics formulations for fluid simulation.

---

## 1. WCSPH (Weakly Compressible SPH)

### Overview

WCSPH is the most straightforward SPH formulation, treating the fluid as slightly compressible and computing pressure directly from density using an equation of state (EOS).

### How It Works

Particles maintain artificial compressibility by solving continuity and momentum equations directly without enforcing incompressibility constraints:

**Continuity Equation:**
```
dρ_i/dt = Σ_j m_j (u_i - u_j) · ∇W_ij
```

**Momentum Equation (Pressure Force):**
```
du_i/dt = -Σ_j m_j (p_i/ρ_i² + p_j/ρ_j²) ∇W_ij
```

**Equation of State (Tait EOS):**
```
p = (ρ₀ c₀²/γ) * [(ρ/ρ₀)^γ - 1]
```

Where:
- `c₀` = sound speed in the fluid
- `γ` = 7 (typical for water, gas dynamics use γ=1.4)
- `ρ₀` = reference density

### Typical Parameters

| Parameter | Typical Value | Range |
|-----------|---------------|-------|
| Sound speed (c₀) | 10-20 × max velocity | Higher = stiffer |
| Reference density (ρ₀) | 1000 kg/m³ | Material dependent |
| EOS exponent (γ) | 7 | 1-7 |
| Max density deviation | ±1% | ±0.5-2% |

### CFL Condition

```
Δt < CFL × h / c₀
```

Where CFL ≈ 0.1-0.4 (lower values more stable, higher more efficient).

**Critical:** CFL must account for sound speed, not just velocity. High sound speeds require very small timesteps.

### Advantages

- **Simple to implement**: Direct calculation from EOS, no iterative solvers needed
- **Explicit time integration**: Standard RK2 or Leapfrog schemes work well
- **Good parallelization**: All particle updates are independent
- **Fast computation**: One pass through particle pairs per timestep
- **Handles free surfaces naturally**: No boundary tracking needed

### Disadvantages

- **Pressure noise**: Density fluctuations cause oscillating pressures
- **Small timesteps**: Sound speed requirement forces CFL ≪ 0.4
- **Artificial dissipation needed**: Requires artificial viscosity to suppress instabilities
- **Density errors accumulate**: Over long simulations, density field becomes noisy
- **Not truly incompressible**: Allows density variations (design choice)

### When to Use

- Real-time applications requiring speed (games, interactive simulations)
- Scenarios with large velocities relative to sound speed
- Free-surface flows without strict incompressibility requirements
- When fast prototyping is priority over accuracy
- GPU-friendly simulations

---

## 2. ISPH (Incompressible SPH)

### Overview

ISPH enforces strict incompressibility by treating pressure as a constraint. Uses a pressure Poisson equation solver to maintain constant density.

### Projection Method (Key Concept)

1. **Predict velocity** ignoring pressure:
   ```
   u*_i = u_i + Δt × (f_gravity + f_viscosity) / ρ_i
   ```

2. **Compute pressure source** from predicted velocity divergence:
   ```
   source_i = -(ρ₀ / Δt) × Σ_j m_j (u*_i - u*_j) · ∇W_ij
   ```

3. **Solve Poisson equation** for pressure:
   ```
   Σ_j m_j × (p_i - p_j) × ∇²W_ij / ρ_j = source_i
   ```

4. **Correct velocity** using pressure gradient:
   ```
   u_{i+1} = u*_i - (Δt / ρ_i) × Σ_j m_j × (p_i/ρ_i² + p_j/ρ_j²) × ∇W_ij
   ```

5. **Update density** from corrected velocity (no change in continuity, by design)

### Poisson Equation Details

The pressure Poisson system is sparse and symmetric, allowing use of:
- **Conjugate Gradient (CG)** solver: 20-50 iterations typical
- **Preconditioned CG**: Much faster convergence
- **Multigrid methods**: Best for large-scale problems
- **GPU sparse solvers**: CuSPARSE, cuSOLVER libraries

### Advantages

- **Accurate incompressibility**: Density stays nearly constant (< 0.1% variation)
- **Physical pressure**: Pressure is truly reactive to incompressibility constraints
- **Larger timesteps**: Not limited by sound speed (CFL ≈ 0.5-1.0)
- **Smooth velocity fields**: Fewer oscillations in velocity
- **Pressure quality**: Much smoother pressure field than WCSPH
- **Better for viscous flows**: Cleaner viscous force computation

### Disadvantages

- **Complex implementation**: Requires Poisson solver, matrix construction
- **Iterative solver overhead**: 20-100 iterations per timestep
- **Poor parallelization**: Poisson solver is sequential bottleneck
- **Memory intensive**: Sparse matrix storage for large systems
- **Convergence issues**: Solver may diverge with poor preconditioning
- **Free surface handling**: Requires special treatment, ghost particles often needed
- **Slower on CPU**: 5-20× slower than WCSPH in practice

### Implementation Complexity

**High** - requires:
- Sparse matrix construction and storage (CSR format)
- Linear solver with preconditioning
- Boundary condition handling in matrix
- Pressure field updates
- Pressure-to-force conversion

### When to Use

- Highly accurate simulations (research, offline rendering)
- Fluid-structure interaction with rigid body dynamics
- Water with tight incompressibility requirements
- Multi-phase flows (each phase has own Poisson equation)
- Scenarios where accuracy is worth the speed penalty

---

## 3. δ-SPH (Delta-SPH)

### Overview

δ-SPH adds a density diffusion term to the continuity equation to suppress pressure oscillations without requiring a full Poisson solver. Simple but effective.

### The Density Diffusion Mechanism

Instead of pure continuity:
```
dρ_i/dt = Σ_j m_j (u_i - u_j) · ∇W_ij
```

Add diffusion term:
```
dρ_i/dt = Σ_j m_j (u_i - u_j) · ∇W_ij + δ × c₀ × h × Σ_j m_j × (ρ_j - ρ_i)/ρ_i × W_ij
```

Or in vector form with kernel Laplacian:
```
dρ_i/dt = Σ_j m_j (u_i - u_j) · ∇W_ij + δ × c₀ × h × Σ_j m_j × (ρ_j - ρ_i) × ∇²W_ij / ρ_j
```

### How It Reduces Pressure Noise

1. **Density diffusion** smooths out local density peaks and valleys
2. **Laplacian term** acts like diffusion: areas with high density diffuse to neighbors
3. **Pressure variance** drops significantly: pressure noise ∝ density fluctuations
4. **No Poisson solver** needed: purely explicit, single-pass computation
5. **Effective smoothing** without solving expensive linear system

### Parameter Selection: δ

**Typical value: δ = 0.1**

| δ Value | Effect | When to Use |
|---------|--------|------------|
| δ = 0 | Pure continuity (standard WCSPH) | Baseline, no diffusion |
| δ = 0.01-0.05 | Minimal diffusion | Already smooth simulations |
| δ = 0.1 | Standard, balanced | Default recommendation |
| δ = 0.2-0.5 | Strong diffusion | Very noisy pressure field |
| δ = 1.0+ | Over-diffusion | Density becomes unrealistic |

**Formula for selecting δ:**
```
δ_optimal ≈ 0.1 × (average_pressure_error / reference_pressure)
```

### Formulation Alternatives

**Version 1: Density difference-based**
```
diffusion_term = δ × c₀ × h × Σ_j m_j × (ρ_j - ρ_i)/ρ_avg × ∇²W_ij
```

**Version 2: Density ratio-based**
```
diffusion_term = δ × c₀ × h × Σ_j m_j × (ρ_j/ρ_i - 1) × ∇²W_ij
```

### Advantages

- **Very simple**: Single line of code modification
- **Nearly free**: Minimal computational overhead vs WCSPH
- **Tunable**: δ parameter allows control of diffusion strength
- **Maintains speed**: Still explicit, CFL ≈ 0.1-0.3
- **Effective**: Eliminates 60-80% of pressure noise from pure WCSPH
- **Parallelizable**: Trivial to parallelize, no solver needed

### Disadvantages

- **Not truly incompressible**: Density still varies (±0.5-1%)
- **Artificial dissipation**: Not physically derived, damping mechanism unclear
- **Parameter tuning needed**: δ varies per scenario
- **Lagrangian vs Eulerian**: Some debate on physical justification
- **Not as smooth as ISPH**: Better than WCSPH but not projection-method quality

### When to Use

- Budget-conscious simulations needing better pressure
- Real-time applications (games, interactive VR)
- Multi-GPU/distributed simulations where Poisson is impractical
- Quick prototyping before committing to ISPH
- Hybrid approach: δ-SPH with low-cost stabilization

### Typical Implementation

```cpp
// Density diffusion term computation
float density_diffusion = 0.0f;
for each neighbor j:
    float rho_diff = (density[j] - density[i]) / density[i];
    float laplacian_w = compute_kernel_laplacian(r_ij, h);
    density_diffusion += mass[j] * rho_diff * laplacian_w;

density_diffusion *= delta * sound_speed * h;

// Update density
ddt_density[i] = continuity_term + density_diffusion;
```

---

## 4. PCISPH (Predictive-Corrective Incompressible SPH)

### Overview

PCISPH uses a predictive-corrective approach to enforce incompressibility iteratively within each timestep, avoiding explicit Poisson solver construction.

### Algorithm

1. **Predict intermediate particle positions** using non-pressure forces:
   ```
   x*_i = x_i + Δt × u_i + (Δt)² × (f_g + f_visc) / ρ_i
   u*_i = u_i + Δt × (f_g + f_visc) / ρ_i
   ```

2. **For k = 1 to max_iterations:**
   - Compute predicted density: ρ*_i from x*_i
   - Compute density error: Δρ_i = ρ*_i - ρ₀
   - Compute pressure correction:
     ```
     p_i^(k) = p_i^(k-1) + (β/Δt) × Δρ_i
     ```
     where β is a scaling factor (tuned for convergence)

   - Update predicted velocity:
     ```
     u*_i := u*_i - (Δt / ρ_i) × Σ_j m_j × (p_i^(k)/ρ_i² + p_j^(k)/ρ_j²) × ∇W_ij
     ```

   - Update predicted position:
     ```
     x*_i := x_i + Δt × u*_i
     ```

3. **Accept corrected state** once convergence criterion met (typical: max density error < 0.5%)

### Convergence Scaling Parameter β

```
β = ρ₀ × (c₀)² × Δt² / m_i
```

Where c₀ is used for stability (not physical sound speed). Empirically:
```
β ≈ 0.5 × h × ρ₀ × c₀² × Δt²
```

### Advantages

- **No explicit Poisson solver**: Iterations happen on particle level
- **Good incompressibility**: Density error typically < 0.1% after 2-3 iterations
- **Better parallelization than ISPH**: Each iteration is particle-local
- **Larger timesteps**: Similar to ISPH (CFL ≈ 0.5-1.0)
- **GPU-friendly**: No global matrix construction or sparse solver needed
- **Pressure is computed iteratively**: Natural from incompressibility constraint

### Disadvantages

- **Iterative per timestep**: 2-5 iterations typical, doubles computational cost
- **Parameter tuning**: β must be carefully chosen for stability
- **Convergence not guaranteed**: Poor β leads to divergence or oscillation
- **Still complex**: More involved than WCSPH or δ-SPH
- **Memory overhead**: Must store intermediate positions and pressures
- **Pressure oscillations**: Still present if iteration count insufficient

### When to Use

- GPU simulations (avoids Poisson solver bottleneck)
- Real-time with incompressibility requirement
- Scenarios requiring 2-3 correctness iterations
- Multi-GPU setups (local particle updates, minimal communication)

---

## 5. EICSPH (Explicit Incompressible Corrective SPH)

### Overview

EICSPH extends PCISPH with explicit pressure computation, eliminating iterative corrections and achieving incompressibility more directly.

### Algorithm

1. **Predict velocity** from non-pressure forces (standard):
   ```
   u*_i = u_i + Δt × (f_g + f_visc) / ρ_i
   ```

2. **Compute density divergence** (how much density changed):
   ```
   div_ρ_i = Σ_j m_j × (u*_i - u*_j) · ∇W_ij × ρ_i
   ```

3. **Compute pressure explicitly** from divergence:
   ```
   p_i = -(ρ₀ × c₀² / (divergence_factor)) × div_ρ_i
   ```

   Where `divergence_factor` relates to the SPH Laplacian operator norm.

4. **Apply pressure force** to correct velocity:
   ```
   u_{i+1} = u*_i - (Δt / ρ_i) × Σ_j m_j × (p_i/ρ_i² + p_j/ρ_j²) × ∇W_ij
   ```

5. **Update position**:
   ```
   x_{i+1} = x_i + Δt × u_{i+1}
   ```

### Advantages

- **Single-pass**: One computation pass per force and pressure
- **No iterations**: Faster than PCISPH
- **Incompressibility**: Density maintained at similar level to ISPH
- **Parallel**: Fully parallelizable, no global synchronization
- **Pressure field**: Computed explicitly without iterative refinement

### Disadvantages

- **Tuning pressure computation**: Divergence factor must be calibrated
- **Less robust than ISPH**: May still show density drift over long simulations
- **Parameter sensitivity**: Divergence factor varies by kernel choice
- **Academic novelty**: Less tested in production than PCISPH or ISPH

### When to Use

- When iterative cost of PCISPH is unacceptable
- Academic or research implementations
- Scenarios with very large timesteps (CFL near 1.0)

---

## 6. Artificial Compressibility SPH

### Overview

Uses dual-time stepping: introduces a pseudo-time dimension to decouple incompressibility from the physical time stepping. Ensures incompressibility at each physical timestep.

### Algorithm

For each physical timestep Δt:

1. **Inner pseudo-time loop** (τ iterations, τ ∈ [0, Δt_inner]):
   - Compute density from current positions
   - Compute pressure from EOS: p = p(ρ)
   - Update velocity:
     ```
     du/dτ = -(1/ρ) ∇p + f_external + f_viscous
     ```
   - Update position:
     ```
     dx/dτ = u
     ```

2. **Pseudo-time stepping** continues until:
   - Density converges to reference (< 0.1% error)
   - Typically 3-5 pseudo-steps per physical step
   - Use small Δt_inner (Δt_inner ≈ 0.1 × Δt_physical)

3. **Accept final state** after pseudo-time convergence

### Artificial Compressibility Parameter

```
Δt_inner = β × Δt_physical
```

Where β ≈ 0.1-0.3 (balance between accuracy and speed).

### Advantages

- **Decoupled dynamics**: Incompressibility constraint in pseudo-time
- **Smooth pressure field**: Pressure naturally smooths in pseudo-time iterations
- **Physical intuition**: Resembles artificial compressibility in finite difference methods
- **Stable**: Very robust to parameter variations

### Disadvantages

- **Computational cost**: Multiple pseudo-steps per physical step (3-5×)
- **Complex to implement**: Nested time loops with different step sizes
- **Not widely used**: Limited literature and established best practices
- **Pseudo-time tuning**: β requires calibration per scenario
- **Slower than single-pass methods**: Despite elegance, generally slower in practice

### When to Use

- Research implementations exploring time-splitting approaches
- Scenarios where pressure smoothness is paramount
- Hybrid simulations combining with other methods

---

## 7. Riemann-SPH

### Overview

Riemann-SPH applies Riemann solvers (from hyperbolic conservation law theory) at particle boundaries to better approximate fluxes between particles. Improves shock capturing and pressure/velocity discontinuity handling.

### Core Concept

Instead of symmetric kernel gradients:
```
Standard: f_i = Σ_j m_j × (f_i/ρ_i² + f_j/ρ_j²) × ∇W_ij
```

Use asymmetric Riemann fluxes:
```
Riemann: f_i = Σ_j m_j × F_Riemann(u_L=u_i, u_R=u_j) × ∇W_ij
```

Where F_Riemann solves the local Riemann problem between states at particle i and j.

### Riemann Problem at Particle Boundary

Given left state (particle i) and right state (particle j):
- Pressure: p_L, p_R
- Velocity: u_L, u_R
- Density: ρ_L, ρ_R

Riemann solver computes:
- **Contact discontinuity**: intermediate pressure p*, intermediate velocity u*
- **Wave speeds**: left and right signal speeds
- **Flux across interface**: computed from Riemann solution

### Common Riemann Solvers Used

1. **Exact Riemann Solver**: Solves gas dynamics exactly (expensive)
2. **Roe Solver**: Approximate, linearized, faster
3. **HLLC Solver**: Hybrid Low-Low-Contact, good for shocks
4. **Toro Solver**: Simplified, efficient for SPH

### Equations (HLLC Example)

```
Pressure flux:
F_p = (p_L + p_R) / 2 + ρ_avg × c_avg × (u_L - u_R) / 2

Velocity flux:
F_u = (u_L + u_R) / 2 - (p_R - p_L) / (ρ_avg × c_avg) / 2

Where:
  c_avg = (c_L + c_R) / 2  (average sound speed)
  ρ_avg = (ρ_L + ρ_R) / 2
```

### Advantages

- **Shock capturing**: Handles pressure/velocity jumps better than standard SPH
- **Better convergence**: Smoother capture of discontinuities
- **Physical consistency**: Derived from hyperbolic conservation laws
- **Improved stability**: Riemann-based fluxes are more stable across discontinuities
- **Low-Mach flows**: Works well from low-Mach to compressible regimes

### Disadvantages

- **Computational cost**: Riemann solver ≈ 5-10× more expensive per pair than simple kernel gradient
- **Implementation complexity**: Requires understanding of Riemann solvers
- **Limited parallelization benefit**: Riemann computation is bottleneck
- **Not mainstream**: Fewer libraries and tools available
- **Overkill for many scenarios**: Standard SPH sufficient for smooth flows
- **Research-level**: Most implementations academic, not production-tested

### When to Use

- Shock-dominated flows (explosions, detonations, supersonic jets)
- Multi-phase flows with strong pressure discontinuities
- Compressible flows with wide Mach number range
- Research simulations in compressible dynamics
- When shock capturing is critical (not smooth flow)

### Implementation Note

Riemann-SPH typically combines with:
- δ-SPH diffusion for smoothing
- WCSPH equation of state for closure
- Explicit time integration (RK2, Leapfrog)

---

## Comparison Table: SPH Variants

| Aspect | WCSPH | WCSPH+δ | ISPH | PCISPH | EICSPH | Artificial Comp. | Riemann-SPH |
|--------|-------|---------|------|--------|--------|------------------|------------|
| **Pressure Quality** | Noisy | Smooth | Very Smooth | Smooth | Smooth | Very Smooth | Very Smooth |
| **Density Error** | ±1-2% | ±0.5-1% | ±0.1% | ±0.1-0.5% | ±0.1-0.5% | ±0.1% | ±0.1-1% |
| **Timestep Size (CFL)** | 0.1-0.3 | 0.1-0.3 | 0.5-1.0 | 0.5-1.0 | 0.5-1.0 | 0.3-0.5 | 0.1-0.3 |
| **Implementation Complexity** | Very Low | Very Low | High | Medium-High | Medium | Medium-High | High |
| **Parallelization** | Excellent | Excellent | Poor (Poisson) | Good | Excellent | Good | Good |
| **Computational Cost** | 1× | 1.1× | 5-10× | 2-5× | 1.2-1.5× | 3-5× | 5-15× |
| **Free Surface Handling** | Native | Native | Complex | Natural | Natural | Moderate | Moderate |
| **Best For** | Real-time | Games, Interactive | Research, Accuracy | GPU Clusters | GPU Clusters | Pseudo-time Studies | Shock Flows |
| **Pressure Oscillations** | High | Low | Minimal | Minimal | Minimal | Minimal | Minimal |
| **Shock Capturing** | Poor | Poor | Moderate | Moderate | Moderate | Moderate | Excellent |
| **Maturity/Use** | ★★★★★ | ★★★★ | ★★★★ | ★★★ | ★★ | ★★ | ★★ |

---

## Density Computation Approaches

### Overview

The density field can be computed in different ways, each with trade-offs. This choice affects pressure stability independent of pressure computation method.

### 1. Density Summation (SPH Summation)

**Formula:**
```
ρ_i = Σ_j m_j × W_ij
```

**Properties:**
- Direct summation over neighbors
- Always conservative: Σ m_j × W_ij ≈ constant
- Kernel-dependent
- No time derivative involved

**Advantages:**
- **Positive-definite**: Density always positive (W > 0)
- **Conserves mass**: Total mass = Σ m_j (kernel independence)
- **Straightforward**: No time integration needed
- **Smoother**: Averaging effect reduces noise
- **Stable**: Doesn't accumulate errors over time

**Disadvantages:**
- **Decoupled from velocity**: Doesn't respect continuity equation directly
- **Pressure may not follow velocity**: Velocity divergence ignored
- **Slower density response**: Lags behind velocity changes
- **Compressibility artifact**: Can cause unphysical density-velocity coupling
- **Free surface issues**: Density underestimated near boundaries (kernel truncation)

**When to use:**
- Stable, long-running simulations
- When pressure smoothness is priority
- Incompressible flow simulations (ISPH, PCISPH)
- Multi-phase flows (consistent inter-phase treatment)

---

### 2. Continuity Equation (Eulerian Derivative)

**Formula:**
```
dρ_i/dt = -ρ_i × Σ_j m_j × (u_i - u_j) · ∇W_ij

or equivalently:

dρ_i/dt = Σ_j m_j × (u_i - u_j) · ∇W_ij  (alternative form)
```

**Properties:**
- Time-integrated: ρ_{i,n+1} = ρ_{i,n} + Δt × (dρ/dt)
- Coupled to velocity: directly responds to velocity divergence
- Physical continuity equation
- Non-conservative in standard SPH

**Advantages:**
- **Physical**: Derived from compressible Euler equations
- **Responsive**: Density immediately reflects velocity changes
- **Pressure coupling**: Direct link between density and velocity
- **Standard in WCSPH**: Widely tested and understood
- **Lower pressure noise**: Density tracks fluid motion directly

**Disadvantages:**
- **Error accumulation**: Time integration drift over long simulations
- **Positive definiteness**: Density can go negative with poor time stepping
- **Free surface issues**: Worse than summation at boundaries
- **Numerical diffusion**: Requires dissipation terms for stability
- **Initial transient**: Density oscillates before settling
- **Kernel-dependent errors**: Errors compound through ∇W computation

**When to use:**
- WCSPH-based simulations
- Real-time applications (short simulation times)
- Scenarios with strong velocity divergence (splash detection)
- When pressure response is important

**Time Discretization Options:**

```cpp
// Forward Euler (basic, less stable)
rho_new = rho + dt * drho_dt;

// RK2 (better stability)
rho_mid = rho + 0.5 * dt * drho_dt;
drho_dt_mid = compute_drho_dt(rho_mid, u);
rho_new = rho + dt * drho_dt_mid;

// Leapfrog (energy stable)
rho_half_old = rho_old - dt * drho_dt_old;
rho_new = rho_half_old + dt * drho_dt_current;
```

---

### 3. Hybrid Approach: Summation + Continuity

**Motivation:**
- Summation provides stability and mass conservation
- Continuity provides physical coupling and responsiveness
- Hybrid gets benefits of both

**Formula:**
```
ρ_i = λ × Σ_j m_j × W_ij + (1-λ) × ρ_{i,integrated}

where ρ_{i,integrated} = ρ_{i,n} + Δt × (dρ/dt)

Typical: λ = 0.5-0.9 (summation-dominant)
```

**Alternative: Filter-based Hybrid**
```
ρ_i^(summation) = Σ_j m_j × W_ij
ρ_i^(continuity) = ρ_{i,n} + Δt × (dρ/dt)
ρ_i = (ρ_i^(summation) + α × ρ_i^(continuity)) / (1 + α)

Where α = 0.1-0.5 (blend ratio)
```

**Advantages:**
- **Robustness**: Summation prevents density collapse/explosion
- **Responsiveness**: Continuity adds coupling to velocity
- **Stability**: Weighted mix more stable than pure continuity
- **Noise reduction**: Summation averaging smooths noise
- **Practical**: Well-established in industry (e.g., Weta Digital, Pixar)

**Disadvantages:**
- **Additional parameter**: λ requires tuning per scenario
- **Increased computation**: Two density computations per step
- **Conceptual mixing**: Not pure physics from any single formulation
- **Memory**: Store both summation and integrated density

**When to use:**
- **Production simulations**: Balance quality and stability
- **WCSPH with pressure requirements**: Want smooth pressure from summation + responsiveness
- **Large-scale simulations**: Long runtime stability important
- **Multi-phase flows**: Different λ per phase possible

**Typical Parameter Selection:**

| Scenario | λ (summation weight) | Use case |
|----------|---------------------|----------|
| λ = 1.0 | Pure summation | Stable, long-running, smooth pressure |
| λ = 0.9 | 90/10 hybrid | Industry standard, balanced |
| λ = 0.7 | 70/30 hybrid | More continuity coupling |
| λ = 0.5 | 50/50 equal blend | Equal weighting |
| λ = 0.0 | Pure continuity | Responsive, short-duration, noisy |

---

### Density Computation Comparison

| Aspect | Summation | Continuity | Hybrid |
|--------|-----------|-----------|--------|
| **Stability** | Excellent | Poor | Good |
| **Mass Conservation** | Perfect | Approximate | Good |
| **Pressure Smoothness** | High | Low | High |
| **Density Responsiveness** | Low | High | Medium |
| **Positive-Definiteness** | Guaranteed | No guarantee | High probability |
| **Computational Cost** | 1× | 1× | 2× |
| **Free Surface Handling** | Moderate | Poor | Better |
| **Shock Capturing** | Moderate | Better | Better |
| **Long-term Drift** | Minimal | Significant | Low |
| **Best For** | Accuracy, smooth fields | Responsiveness, short-term | Production use |

---

## Practical Selection Guide

### Choose WCSPH if:
- Real-time performance is critical
- Implementing SPH for the first time
- GPU acceleration is target
- Can tolerate pressure oscillations
- Free-surface flows preferred

### Choose WCSPH + δ-SPH if:
- WCSPH pressure too noisy
- Need minor improvement without solver
- Still want real-time speed
- Multi-GPU systems

### Choose ISPH if:
- Maximum accuracy required
- Incompressibility critical (< 0.1% density error)
- CPU-based (not GPU-bottlenecked by Poisson)
- Fluid-structure interaction important
- Offline rendering or research

### Choose PCISPH if:
- GPU simulation (no global Poisson)
- Need incompressibility (> 0.5% acceptable)
- Multiple GPUs with per-particle updates only
- Moderate accuracy requirements

### Choose EICSPH if:
- PCISPH iteration cost unacceptable
- Single-pass pressure desired
- Academic exploration
- Extreme timestep requirements

### Choose Artificial Compressibility if:
- Pseudo-time physics interesting
- Pressure smoothness paramount
- Research/novelty-seeking
- Multi-scale temporal phenomena

### Choose Riemann-SPH if:
- Shocks/detonations present
- Compressible flow multi-scale
- Pressure discontinuities important
- Accept higher computational cost

### Hybrid Summation/Continuity Density if:
- Production quality simulation
- Long-running (> 10,000 frames)
- Both accuracy and stability needed
- Industry standard preferred

---

## References & Further Reading

### Key Papers
- **WCSPH**: Monaghan (1994, 2005) - Smoothed particle hydrodynamics
- **δ-SPH**: Antuono et al. (2010) - Numerical diffusive terms in weakly compressible SPH
- **ISPH**: Shao & Lo (2003) - Incompressible SPH method for simulating Newtonian and non-Newtonian flows
- **PCISPH**: Solenthaler & Pauly (2009) - Predictive-corrective incompressible SPH
- **Riemann-SPH**: Cha & Whitworth (2003) - Implementations of the Riemann solver in smoothed particle hydrodynamics
- **Density Filtering**: Violeau & Issa (2007) - Numerical modelling of complex turbulent free-surface flows with the SPH method

### Implementation Resources
- **DualSPHysics**: Open-source WCSPH/δ-SPH (https://github.com/DualSPHysics/DualSPHysics)
- **GiD**: GiD preprocessing + SPH output (https://www.gidhome.com/)
- **FLIP/PIC-SPH Hybrid**: Zhu et al. (2015)

---

**Last Updated:** 2026-02-08
