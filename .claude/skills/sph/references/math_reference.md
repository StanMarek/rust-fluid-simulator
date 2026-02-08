# SPH Mathematical Reference

## Table of Contents

1. [Kernel Functions](#kernel-functions)
2. [SPH Operators](#sph-operators)
3. [Governing Equations](#governing-equations)
4. [Equations of State](#equations-of-state)
5. [Viscosity Models](#viscosity-models)
6. [Density Corrections](#density-corrections)
7. [Time Integration](#time-integration)
8. [Consistency Analysis](#consistency-analysis)
9. [Parameter Guidance](#parameter-guidance)

---

## Kernel Functions

Kernel functions W(r, h) form the foundation of SPH interpolation. They satisfy:
- Normalization: ∫ W(r, h) dr = 1
- Compact support: W(r, h) = 0 for r ≥ h
- Delta property: lim_{h→0} W(r, h) = δ(r)

### Cubic Spline Kernel

**Normalized Constant:**
- 1D: σ = 2/3h
- 2D: σ = 10/(7πh²)
- 3D: σ = 1/(πh³)

**Formula (q = r/h, 0 ≤ q < 2):**

W(q, h) = σ {
  1 - (3/2)q² + (3/4)q³,           if 0 ≤ q < 1
  (1/4)(2-q)³,                      if 1 ≤ q < 2
  0,                                 if q ≥ 2
}

**Gradient:**

∇W(q, h) = σ/h {
  -3q + (9/4)q²,                    if 0 ≤ q < 1
  -(3/4)(2-q)²,                     if 1 ≤ q < 2
  0,                                 if q ≥ 2
} × r̂

**Laplacian:**

∇²W(q, h) = σ/h² {
  -3 + (9/2)q,                      if 0 ≤ q < 1
  -(3/2)(2-q),                      if 1 ≤ q < 2
  0,                                 if q ≥ 2
}

### Quintic Spline Kernel

**Normalized Constant:**
- 1D: σ = 1/(120h)
- 2D: σ = 7/(478πh²)
- 3D: σ = 1/(120πh³)

**Formula (q = r/h):**

W(q, h) = σ {
  (1-q)⁵ - 6(0.5-q)⁵ + 15(0-q)⁵,   if 0 ≤ q < 1
  (1-q)⁵ - 6(0.5-q)⁵,               if 1 ≤ q < 1.5
  (1-q)⁵,                            if 1.5 ≤ q < 3
  0,                                 if q ≥ 3
}

**Gradient:**

∇W(q, h) = σ/h {
  -5(1-q)⁴ + 30(0.5-q)⁴ - 75(q)⁴,  if 0 ≤ q < 1
  -5(1-q)⁴ + 30(0.5-q)⁴,            if 1 ≤ q < 1.5
  -5(1-q)⁴,                          if 1.5 ≤ q < 3
  0,                                 if q ≥ 3
} × r̂

### Wendland Kernels

Wendland kernels are smoother (C²) and more computationally efficient alternatives.

#### Wendland C2 (2D)

**Support:** q_max = 2

W(q, h) = σ(1-q/2)⁴(1 + 2q),  for q < 2
∇W(q, h) = σ/h × 5(1-q/2)³(-q) × r̂

σ = 7/(4πh²)

#### Wendland C2 (3D)

**Support:** q_max = 2

W(q, h) = σ(1-q/2)⁴(1 + 2q),  for q < 2
∇W(q, h) = σ/h × 5(1-q/2)³(-q) × r̂

σ = 21/(16πh³)

#### Wendland C4 (2D)

**Support:** q_max = 2

W(q, h) = σ(1-q/2)⁶(1 + 3q + 8q²/5),  for q < 2

σ = 9/(π h²)

∇W(q, h) = σ/h × (1-q/2)⁵(-24q - 24q²/5) × r̂

#### Wendland C4 (3D)

**Support:** q_max = 2

W(q, h) = σ(1-q/2)⁶(1 + 3q + 8q²/5),  for q < 2

σ = 495/(256π h³)

∇W(q, h) = σ/h × (1-q/2)⁵(-24q - 24q²/5) × r̂

#### Wendland C6 (3D)

**Support:** q_max = 2

W(q, h) = σ(1-q/2)⁸(1 + 4q + 17q²/5 + 32q³/15),  for q < 2

σ = 1365/(512π h³)

∇W(q, h) = σ/h × (1-q/2)⁷ × (-32 - 128q/5 - 96q²/5) × r̂

### Gaussian Kernel

**Normalized Constant:**
- 1D: σ = 1/(h√π)
- 2D: σ = 1/(πh²)
- 3D: σ = 1/(π^(3/2)h³)

**Formula:**

W(r, h) = σ exp(-(r/h)²)

**Gradient:**

∇W(r, h) = -2σ/h² × r × exp(-(r/h)²)

**Laplacian:**

∇²W(r, h) = σ × 4(2(r/h)² - d)/h² × exp(-(r/h)²)

where d is the dimensionality (1, 2, or 3).

---

## SPH Operators

### Standard SPH Interpolation

**Particle Form (Standard):**

f(r_i) ≈ Σ_j (m_j / ρ_j) f_j W(r_i - r_j, h)

**Continuous Form:**

f(r) ≈ ∫ f(r') W(r - r', h) dr'

**Volume Approximation:**

f(r_i) ≈ Σ_j (m_j / ρ_j) f_j W_ij

where W_ij = W(r_i - r_j, h)

### Gradient Operator

**Standard Form:**

∇f(r_i) ≈ Σ_j (m_j / ρ_j) f_j ∇_i W_ij

**Symmetric Form (Improves Momentum Conservation):**

∇f(r_i) ≈ -Σ_j m_j (f_j/ρ_j² + f_i/ρ_i²) ∇_i W_ij

**Alternative Symmetric Form:**

∇f(r_i) ≈ Σ_j m_j (f_i/ρ_i² + f_j/ρ_j²) ∇_i W_ij

**Divergence:**

∇·u(r_i) ≈ -Σ_j m_j (u_j - u_i)/ρ_j · ∇_i W_ij

or (symmetric):

∇·u(r_i) ≈ -Σ_j m_j (u_j - u_i) · (∇_i W_ij / ρ_j)

### Laplacian Operator

**Standard Form:**

∇²f(r_i) ≈ 2/Σ_j m_j/ρ_j W_ij × Σ_j (m_j / ρ_j) (f_j - f_i) ∇²W_ij / (r_ij · ∇_i W_ij)

**Improved Fatehi-Manzari Form (Better Accuracy):**

∇²f(r_i) ≈ Σ_j (m_j / ρ_j) (f_j - f_i) × [4 ∇_i W_ij · ∇_i W_ij / (r_ij² + 0.01h²)]

**Simple Symmetric Form:**

∇²f(r_i) ≈ 2 Σ_j (m_j / ρ_j) (f_j - f_i) ∇²W_ij / (r_ij + ε)

---

## Governing Equations

### Continuity Equation

**Form 1: Density Summation (Most Common)**

ρ_i = Σ_j m_j W_ij

**Form 2: Continuity-Based Density Update**

dρ_i/dt = -ρ_i ∇·u(r_i) = ρ_i Σ_j m_j (u_j - u_i)/ρ_j · ∇_i W_ij

**Form 3: Direct SPH Gradient**

dρ_i/dt = Σ_j m_j (u_j - u_i) · ∇_i W_ij

### Momentum Equation

**Standard Form (Pressure + Viscosity):**

dv_i/dt = -Σ_j m_j (p_j/ρ_j² + p_i/ρ_i²) ∇_i W_ij + Π_ij + f_ext_i

**Symmetric Pressure (Recommended for Stability):**

dv_i/dt = -Σ_j m_j (p_i/ρ_i² + p_j/ρ_j²) ∇_i W_ij + Π_ij + f_ext_i

**With Pressure Correction for Zero-Pressure Reference:**

dv_i/dt = -Σ_j m_j ((p_i + p_j)/(ρ_i ρ_j)) ∇_i W_ij + Π_ij + f_ext_i

### Energy Equation

**Velocity-Based Form:**

du_i/dt = (p_i/ρ_i²) Σ_j m_j (v_j - v_i) · ∇_i W_ij + Π_ij · (v_j - v_i) + Q_i

**Conservative Form (using enthalpy):**

dh_i/dt = (1/ρ_i) Σ_j m_j (p_j/ρ_j² (v_j - v_i) · ∇_i W_ij) + Π_ij · (v_j - v_i) + Q_i

where u is specific internal energy, h is specific enthalpy, Q_i is heat transfer.

**Temperature Form:**

ρc_v dT_i/dt = -p_i ∇·u_i + Π_ij · (v_j - v_i) + Q_i

---

## Equations of State

### Tait (Cole) Equation of State

**Formula:**

p = (ρ_0 c_0²/γ) [(ρ/ρ_0)^γ - 1]

where:
- ρ_0 = reference density
- c_0 = speed of sound at reference density
- γ = stiffness parameter (typically 7 for water, 1.4 for air)

**Alternative Form (Multi-phase):**

p = ρ_0 c_0² [(ρ/ρ_0)^γ - 1] / (γ - 1)

**Parameters:**
- **Water:** γ = 7, ρ_0 = 1000 kg/m³, c_0 ≈ 1450 m/s
- **Air:** γ = 1.4, ρ_0 = 1.225 kg/m³, c_0 ≈ 340 m/s
- **Oil:** γ = 7, ρ_0 = 860 kg/m³, c_0 ≈ 1400 m/s

**When to Use:** Weakly compressible flows (WCSPH), enforces weak density variations (~1-2%)

### Ideal Gas Equation

**Formula:**

p = ρ RT / M = ρ k_B T

where:
- R = universal gas constant
- M = molar mass
- k_B = Boltzmann constant
- T = temperature

**Variant (internal energy form):**

p = (γ - 1) ρ u

where u is specific internal energy.

**When to Use:** Compressible gas flows, high-speed flows, shock propagation. Requires special stability measures (ISPH, SESPH).

### Linear Equation of State

**Formula:**

p = c_0² (ρ - ρ_0)

or with minimum pressure:

p = max(c_0² (ρ - ρ_0), p_min)

**When to Use:** Incompressible flows, enforces ρ ≈ ρ_0. Simple and fast but may require artificial sound speed adjustment.

### Stiffened Gas Equation

**Formula:**

p = (γ - 1) ρ u - γ Π_∞

where:
- u = specific internal energy
- Π_∞ = internal pressure constant (material-dependent)

**Parameters:**
- **Liquid Water:** Π_∞ ≈ 3000 bar, γ = 4.4
- **Explosive Products:** Π_∞ varies, γ ≈ 3

**When to Use:** High-pressure multi-phase flows, detonation phenomena, strong shock waves

---

## Viscosity Models

### Monaghan Artificial Viscosity

**Full Formula:**

Π_ij = {
  (-α c̄ μ_ij + β μ_ij²) / ρ̄,  if v_ij · r_ij < 0
  0,                             if v_ij · r_ij ≥ 0
}

where:
- μ_ij = (h v_ij · r_ij) / (r_ij² + η²)
- v_ij = v_i - v_j
- r_ij = r_i - r_j
- η = 0.01h (small regularization parameter, prevents singularities)
- c̄ = (c_i + c_j)/2 (mean sound speed)
- ρ̄ = (ρ_i + ρ_j)/2 (mean density)

**Particle Force Contribution:**

Π_ij = (m_j Π_ij) · ∇_i W_ij

**Parameters:**
- **Standard:** α = 0.01, β = 0.01
- **Shear-Dominant:** α = 0.001, β = 0.0001
- **Shock-Dominated:** α = 0.05-0.1, β = 0.05

**Physical Interpretation:**
- Linear term (α): Bulk viscosity, dissipates kinetic energy
- Quadratic term (β): Shock capturing, prevents particle penetration
- η parameter: Prevents singularities when particles are close

### Laminar Viscosity (Morris Formulation)

**Formula (2D and 3D):**

(∇²u)_i = Σ_j (m_j / ρ_j) (v_j - v_i) · [4/(d+2)] × ∇²W_ij

**Viscous Stress Contribution:**

F_visc_i = -μ Σ_j (m_j / ρ_j) (v_j - v_i) · [8/(d+2)] × ∇_i W_ij · r_ij / (r_ij² + η²)

or more accurately:

F_visc_i = 2 μ Σ_j (m_j / ρ_j) (v_j - v_i) ∇²W_ij

where:
- μ = dynamic viscosity (Pa·s)
- d = spatial dimension (2 or 3)

**Parameters:**
- **Water at 20°C:** μ ≈ 0.001 Pa·s, ν = 1.0×10⁻⁶ m²/s
- **Oil:** μ ≈ 0.1 Pa·s, ν = 1.0×10⁻⁴ m²/s
- **Air at 20°C:** μ ≈ 1.8×10⁻⁵ Pa·s, ν ≈ 1.5×10⁻⁵ m²/s

**Relationship to Kinematic Viscosity:**

ν = μ / ρ

### XSPH Velocity Correction

**Formula:**

ṽ_i = v_i - ε Σ_j (m_j / ρ_j) (v_i - v_j) W_ij / Σ_j (m_j / ρ_j) W_ij

or equivalently:

ṽ_i = v_i - ε Σ_j m_j (v_i - v_j) W_ij / ρ_i

**Purpose:**
- Reduces particle clustering
- Improves stability in incompressible flows
- Acts as artificial viscosity without dissipation
- Preserves momentum and energy

**Parameters:**
- ε = 0.0-0.5 (typical: 0.1-0.2)
- ε = 0: No correction
- ε = 0.5: Maximum stability (but more artificial)

**Application Order:**

1. Update velocities from pressure and viscous forces
2. Apply XSPH correction to smoothed velocities
3. Use smoothed velocities for advection (position updates)

---

## Density Corrections

Density errors in SPH can cause significant errors in pressure and acceleration. Several correction schemes exist.

### Shepard Filter (Zeroth-Order Correction)

**Formula:**

ρ̃_i = ρ_i / Σ_j (m_j / ρ_j) W_ij

or equivalently:

ρ̃_i = ρ_i / W_0(h)

where W_0(h) is the normalization defect:

W_0(h) = Σ_j (m_j / ρ_j) W_ij

**Properties:**
- Restores C₀ consistency (partition of unity)
- Minimal computational cost
- Smooths density oscillations
- Can be applied at each timestep or periodically

**Limitations:**
- Only corrects zeroth-order errors
- May over-correct near boundaries
- Density oscillations still present in momentum equation

### Moving Least Squares (MLS) Correction

**Theory:**

Approximation error minimized by fitting polynomial through particle values.

**Zero-Order MLS:**

ρ̃_i = [Σ_j m_j W_ij] × [Σ_j (m_j / ρ_j) W_ij]⁻¹

**First-Order MLS (with position correction):**

Position correction matrix:

A_i = Σ_j (m_j / ρ_j) W_ij (r_ij ⊗ r_ij)

(Requires matrix inversion, expensive in 3D)

Corrected density:

ρ̃_i = Σ_j m_j B_i^T · (r_i - r_j) W_ij

where B_i is derived from A_i.

**Properties:**
- Higher-order accuracy (C¹ consistency)
- Computationally expensive (matrix inversion)
- Better error control than Shepard
- Recommended for: High-precision simulations, convergence studies

### Corrective Smoothed Particle Method (CSPM)

**Concept:**

Apply gradient correction to kernel derivatives.

**Gradient Correction Matrix:**

L_i = Σ_j (m_j / ρ_j) (r_ij ⊗ ∇_i W_ij)

**Corrected Kernel Gradient:**

∇̃_i W_ij = L_i^(-T) · ∇_i W_ij

**Corrected SPH Gradient:**

∇f(r_i) = Σ_j m_j (f_j/ρ_j) ∇̃_i W_ij

**Properties:**
- Improves gradient accuracy near boundaries
- Reduces kernel truncation errors
- Computational cost: Matrix inversion once per particle
- Better for solid mechanics and multiphase flows

### δ-SPH Density Diffusion

**Concept:**

Add diffusion term to continuity equation to reduce density oscillations.

**Modified Continuity Equation:**

dρ_i/dt = Σ_j m_j (u_j - u_i) · ∇_i W_ij + 2δ Σ_j (m_j / ρ_j) (ρ_i - ρ_j) u_ij · ∇_i W_ij

where:
- δ = diffusion coefficient (typical: 0.1)
- u_ij = |u_i - u_j|

**Alternative Formulation (Molteni & Colagrossi):**

dρ_i/dt = Σ_j m_j (u_j - u_i) · ∇_i W_ij + h c Σ_j (m_j / ρ_j) (ρ_i - ρ_j) ∇_i W_ij · r_ij / (r_ij² + η²)

where c = 0.3-1.0 (typically 0.5).

**Benefits:**
- Reduces unphysical density oscillations
- Improves long-time stability
- Minimal computational overhead
- Particularly effective for WCSPH

**Drawbacks:**
- Adds diffusion (not purely hyperbolic)
- Parameter tuning required
- Can smooth out legitimate density gradients

---

## Time Integration

### Leapfrog Integration

**Velocity-Position Storage (Half-Step Velocities):**

v^(n+1/2) = v^(n-1/2) + Δt · a^(n)

x^(n+1) = x^(n) + Δt · v^(n+1/2)

**Complete Update Sequence:**

1. **Predict:** x^(n+1/2) = x^(n) + (Δt/2) v^(n)
2. **Evaluate:** ρ^(n), a^(n) from x^(n)
3. **Update velocity:** v^(n+1/2) = v^(n) + (Δt/2) a^(n)
4. **Update position:** x^(n+1) = x^(n) + Δt v^(n+1/2)
5. **Evaluate:** ρ^(n+1), a^(n+1) from x^(n+1)
6. **Update velocity:** v^(n+1) = v^(n+1/2) + (Δt/2) a^(n+1)

**Accuracy:** Second-order (O(Δt²))

**Stability:** Stable for oscillatory motion, requires small Δt for dissipative systems

**Advantages:**
- Energy-conserving
- Symplectic (preserves phase-space volume)
- Low memory footprint

**Disadvantages:**
- Requires storing half-step velocities
- Poor damping (artificial dissipation needed for real viscosity)

### Velocity Verlet Integration

**Position-Based Storage (Full-Step Velocities):**

**Update Sequence:**

1. **Half-step velocity:** v^(n+1/2) = v^(n) + (Δt/2) a^(n)
2. **Update position:** x^(n+1) = x^(n) + Δt v^(n+1/2)
3. **Evaluate:** a^(n+1) from x^(n+1)
4. **Update velocity:** v^(n+1) = v^(n+1/2) + (Δt/2) a^(n+1)

**Explicit Form (without half-step storage):**

x^(n+1) = x^(n) + Δt v^(n) + (Δt²/2) a^(n)

v^(n+1) = v^(n) + (Δt/2) (a^(n) + a^(n+1))

**Accuracy:** Second-order (O(Δt²))

**Stability:** Symplectic, energy-conserving, better damping than leapfrog

**Advantages:**
- Simple, stores full-step velocities (easier to interpret)
- Symplectic (energy-stable)
- Better dissipation characteristics
- Most widely used in SPH codes

**Disadvantages:**
- Requires two accelerations per step
- Still needs artificial damping for viscous flows

### Predictor-Corrector Integration

**Two-Step Second-Order Scheme:**

**Predictor Step:**

v_p^(n+1) = v^(n) + Δt a^(n)

x_p^(n+1) = x^(n) + Δt v^(n)

**Evaluate:** a_p^(n+1) from x_p^(n+1)

**Corrector Step:**

v^(n+1) = v^(n) + (Δt/2) (a^(n) + a_p^(n+1))

x^(n+1) = x^(n) + (Δt/2) (v^(n) + v^(n+1))

**Iterative Variant (ISPH):**

For incompressible SPH, may iterate corrector step:

For k = 1, 2, ..., n_iter:
  p^(k+1) = solve pressure from divergence-free constraint
  v^(k+1) = v_p + (Δt/ρ) ∇p^(k+1)
  x^(k+1) = x_p + Δt v^(k+1)

**Accuracy:** Second-order (O(Δt²))

**Stability:** More stable than single-step leapfrog for stiff systems

**Use Cases:**
- Standard WCSPH (single corrector iteration)
- ISPH incompressible flows (multiple corrector iterations)
- Multiphase flows with surface tension

### CFL Conditions

**Weakly Compressible SPH (WCSPH):**

**Acoustic CFL:**

Δt ≤ C_acoustic · h / (c_0 + |u|_max)

where C_acoustic ≈ 0.4 (conservative: 0.25)

**Viscous CFL:**

Δt ≤ C_viscous · h² / (4 ν)

where C_viscous ≈ 0.125

**Combined (use minimum):**

Δt ≤ min(Δt_acoustic, Δt_viscous)

**Incompressible SPH (ISPH):**

For pressure-Poisson based schemes:

Δt ≤ C · h / |u|_max

where C ≈ 0.4-1.0 (less restrictive than WCSPH)

**Parameter Values:**

| Parameter | Typical | Conservative | Aggressive |
|-----------|---------|--------------|------------|
| C_acoustic | 0.4 | 0.25 | 0.6 |
| C_viscous | 0.125 | 0.08 | 0.2 |
| ISPH C | 0.4 | 0.2 | 1.0 |

---

## Consistency Analysis

### Partition of Unity (C₀ Consistency)

**Definition:**

A kernel approximation has C₀ consistency if:

Σ_j (m_j / ρ_j) W(r_i - r_j, h) = 1 (approximately, near interior)

**Physical Meaning:**
- Average of kernel values ≈ 1
- Ensures constant fields are correctly interpolated
- Guarantees: f(x) ≈ Σ_j (m_j/ρ_j) f_j W_ij → f(x) = f(x) for constant f

**Shepard Correction Restores C₀:**

Divide by normalization:

ρ̃_i = ρ_i / Σ_j (m_j/ρ_j) W_ij

**Boundary Effects:**

- Near free surfaces: W₀ < 1 (kernel support truncated)
- Density underestimation ∝ (1 - W₀)
- Pressure spike at boundary

### Linear Consistency (C¹ Consistency)

**Definition:**

Kernel approximation is C¹ consistent if both f(x) and ∇f(x) are exact for linear functions:

∇f(r_i) = -Σ_j (m_j/ρ_j) f_j ∇_i W_ij → ∇f(r_i) exactly for f linear

**Requirement:**

This fails for standard SPH!

-Σ_j (m_j/ρ_j) (r_ij ⊗ ∇_i W_ij) = I (identity matrix)

For irregular particle distributions, this is violated.

**Correction: Gradient Correction Matrix**

Define:

L_i = Σ_j (m_j/ρ_j) (r_ij ⊗ ∇_i W_ij)

Apply correction to kernel gradient:

∇̃_i W_ij = L_i^(-T) · ∇_i W_ij

**CSPM Formulation:**

∇f(r_i) = Σ_j (m_j/ρ_j) f_j ∇̃_i W_ij

Now ∇f is exact for linear functions.

**Computational Cost:**

- Matrix inversion once per particle per timestep
- 2D: 2×2 matrix (trivial)
- 3D: 3×3 matrix (manageable)
- Overhead: ~10-20% time increase

### Quadratic Consistency (C² Consistency)

**Definition:**

Both f and ∇²f are exact for quadratic functions.

**Requirement:**

More stringent than C¹. Requires higher-order corrections.

**Moving Least Squares (MLS) Approach:**

Fit polynomial p(r) = a₀ + a·r + (1/2)r·A·r through particle values minimizing:

E = Σ_j W(r_ij, h) [p(r_ij) - f_j]²

**First-Order MLS:**

Requires matrix inversion of:

A_i = Σ_j (m_j/ρ_j) W_ij (r_ij ⊗ r_ij)

Then:

∇f(r_i) ≈ A_i^(-1) · Σ_j (m_j/ρ_j) f_j (r_ij W_ij)

**Second-Order MLS:**

Includes quadratic terms, requires larger matrix inversion (more expensive).

**Use Cases:**

- High-precision convergence studies
- Solid mechanics (better stress accuracy)
- Multiphase flows with interfacial tension
- When smooth solutions exist

### Truncation Error Analysis

**Kernel Approximation Error:**

For standard SPH with kernel W(r, h):

f(r_i) = Σ_j (m_j/ρ_j) f(r_j) W(r_i - r_j, h) + O(h²)

Error is O(h²) for smooth functions (kernel-dependent).

**Kernel Gradient Error:**

∇f(r_i) = -Σ_j (m_j/ρ_j) f_j ∇_i W(r_ij, h) + O(h)

Error is O(h) (one order lower for derivatives).

**Without Correction:**

- Cubic spline: ∇f error ~ O(h)
- Wendland C2: ∇f error ~ O(h)
- Quintic: ∇f error ~ O(h²) (higher-order kernel)

**With Gradient Correction (CSPM):**

- All kernels: ∇f error → O(h²)
- Cost: Modest (matrix inversion)
- Benefit: Significant error reduction

### Kernel Choice Summary

| Kernel | Smoothness | Support | Speed | Memory | Gradients | Best For |
|--------|-----------|---------|-------|--------|-----------|----------|
| Cubic Spline | C² | q < 2 | Fast | Low | Good | Standard, general-purpose |
| Quintic | C⁴ | q < 3 | Slower | Moderate | Better | High-precision, smooth flows |
| Wendland C2 | C² | q < 2 | Very Fast | Low | Good | Real-time, large-scale |
| Wendland C4 | C⁴ | q < 2 | Fast | Low | Better | Balance: speed & accuracy |
| Wendland C6 | C⁶ | q < 2 | Moderate | Low | Excellent | High-precision needed |
| Gaussian | C∞ | Infinite | Slowest | Highest | Best | Mathematical analysis only |

---

## Parameter Guidance

### Particle Spacing & Smoothing Length

**Relationship:**

h = k · Δx

where:
- Δx = particle spacing (domain length / number of particles)
- k = smoothing length factor (typically 1.2-2.0)

**Recommended Values:**

- **WCSPH, Cubic Spline:** k ≈ 1.5
- **WCSPH, Wendland:** k ≈ 2.0-2.5
- **ISPH:** k ≈ 1.5-2.0
- **High-precision:** k ≈ 2.0

**Smaller h:**
- Faster computation (fewer neighbors)
- Lower numerical diffusion
- More particle oscillations
- Risk of instability

**Larger h:**
- More neighbors (slower)
- Smoother results
- More damping
- Better stability

### Sound Speed Selection (WCSPH)

**Physical vs. Numerical:**

SPH requires explicitly handling compressibility via equation of state.

**Physical Sound Speed:**

c_physical = √(dp/dρ)_s

For Tait EOS:

c_0 = √(γ p₀ / ρ_0)

**Numerical Sound Speed (CFL):**

Often reduced to c_num < c_physical to increase timestep:

c_num = α · c_physical, where α = 0.1-0.5

**Trade-offs:**

- Large c_num: Small timesteps (accurate but slow)
- Small c_num: Large timesteps (fast but density oscillations)
- Recommendation: Use c_physical initially, adjust for stability

### Viscosity Parameter Selection

**For Monaghan Artificial Viscosity:**

**Shear-Dominated Flows:**
- Low Reynolds number (laminar)
- α = 0.001, β = 0.0001
- Example: Viscous spreading, lid-driven cavity

**Moderate Flows:**
- Standard applications
- α = 0.01, β = 0.01
- Example: Dam break, sloshing

**Shock-Dominated Flows:**
- High-speed impacts, explosions
- α = 0.05-0.1, β = 0.05
- Example: Collapse with shock waves

**For Laminar (Morris) Viscosity:**

**Kinematic Viscosity (ν = μ/ρ):**

- **Water:** ν ≈ 1.0×10⁻⁶ m²/s
- **Oil:** ν ≈ 1.0×10⁻⁴ m²/s
- **Honey:** ν ≈ 0.1 m²/s

**Compute Reynolds Number:**

Re = |u| L / ν

If Re << 1: Viscous dominates (increase viscosity effect)
If Re >> 1: Inertial dominates (can reduce artificial viscosity)

### Neighbor Search Parameters

**Search Radius:**

r_search = 2h (or kernel support radius)

**Recommended Neighbor Count:**

- **2D:**
  - Cubic spline: 20-30 neighbors
  - Wendland: 30-50 neighbors
  - Minimum acceptable: 15

- **3D:**
  - Cubic spline: 50-80 neighbors
  - Wendland: 100-150 neighbors
  - Minimum acceptable: 40

**Low Neighbor Count:**

- Risk of particle instability
- Discrete pressure noise
- Better for very large simulations

**High Neighbor Count:**

- Smooth pressure fields
- Slower computation
- Over-damping possible

### Timestep Selection

**Use all three CFL constraints:**

1. **Acoustic:** Δt_acoustic = 0.25 h / (c_max + |u|_max)
2. **Viscous:** Δt_viscous = 0.125 h² / (4ν)
3. **Gravity/Forces:** Δt_gravity = √(h / g) if gravity significant

**Combined:**

Δt = 0.9 × min(Δt_acoustic, Δt_viscous, Δt_gravity)

Factor 0.9 provides safety margin.

**Adaptive Stepping:**

```
max_u = max over all particles of |u_i|
max_c = max sound speed in domain
Δt_new = 0.25 * h / (max_c + max_u)
```

Update every 5-10 timesteps to avoid frequent changes.

### Domain Decomposition (Parallel SPH)

**For Shared-Memory (OpenMP):**

- Spatial decomposition with ghost layers
- Ghost layer thickness: 2h
- Partition into domains of approximate size: 10×10×10 particles (3D)

**For Distributed Memory (MPI):**

- Similar spatial decomposition
- Ghost particle exchange at each timestep
- Load balance: Monitor particles per domain
- Rebalance when imbalance > 20%

**Neighbor Search Optimization:**

- Linked-list for O(N) search: Cell size ≈ h
- Spatial hashing: Trade-off memory vs. complexity
- BVH trees: Better for non-uniform distributions

---

## References & Implementation Notes

### Common Implementation Pitfalls

1. **Density Errors:** Use Shepard filter or δ-SPH density diffusion
2. **Pressure Instability:** Use symmetric pressure formulation, avoid pressure spike in momentum
3. **Tension Instability:** Limit gradient correction, use surface tension models carefully
4. **Boundary Conditions:** Ghost particles are standard; mirror particles for Dirichlet
5. **Kernel Gradient Sign:** Check sign convention for ∇_i W_ij (should point from j toward i)

### Code Organization

Typical SPH simulator structure:

1. **Initialization:** Place particles, compute initial density/pressure
2. **Main Loop:**
   a. Neighbor search
   b. Density update (summation or continuity)
   c. Force computation (pressure, viscosity, external)
   d. Density correction (Shepard filter, optional)
   e. Time integration (Verlet or leapfrog)
   f. Boundary handling
3. **Output:** Save particle positions, velocities, pressures at regular intervals

### Performance Optimization

**Typical Bottlenecks (in order):**

1. Neighbor search: ~40-50% (optimize with spatial hashing, linked-lists)
2. Force computation: ~30-40% (vectorize, cache-friendly loops)
3. Time integration: ~10-15% (simple, usually fast)
4. Memory allocation: Minimize in main loop (pre-allocate neighbor lists)

**Target:** 50-100 million particle interactions per second on modern CPUs (single core)

---

**Last Updated:** 2024
**Accuracy Level:** Research-grade (suitable for publications)
**Domain:** Computational Fluid Dynamics, Smoothed Particle Hydrodynamics
