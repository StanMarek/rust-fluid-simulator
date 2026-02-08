# Multi-Phase SPH: Formulation, Implementation, and Advanced Techniques

This guide covers the simulation of fluids with multiple phases (liquid-gas, oil-water, etc.) using SPH.

---

## 1. Multi-Phase Formulation: Particle Set Architecture

### 1.1 Separate Particle Sets Per Phase

**Core Concept**:
- Each phase represented by its own set of particles
- Phase α particles interact with all other particles (same and different phases)
- Density and pressure computed from all neighboring particles (weighted by phase)
- Momentum equation modified to account for inter-phase forces

**Particle Assignment**:
```
Phase A: particles {i : i ∈ Ω_A}  (e.g., water)
Phase B: particles {j : j ∈ Ω_B}  (e.g., air)
Domain: Ω = Ω_A ∪ Ω_B  (non-overlapping)
```

### 1.2 Density Estimation with Multiple Phases

**Naive Approach (No Phase Distinction)**:
```
ρ_i = Σ_j (same & different phases) m_j W(r_ij, h)
```

**Problem**: Boundary particles between phases sample ghost densities poorly.

**Improved: Phase-Aware Density**:
```
ρ_i^(A) = m_i W(0, h) + Σ_{j ∈ Ω_A, j≠i} m_j W(r_ij, h)
          + Σ_{j ∈ Ω_B} m_j^eff W(r_ij, h)
```
where `m_j^eff = m_j × (ρ_j / ρ_j^avg)` accounts for different bulk densities.

**Best Practice**: Use particles from *all* phases within smoothing radius:
- Self-contribution: always include
- Same-phase neighbors: standard SPH
- Different-phase neighbors: include with proper mass weighting

### 1.3 Pressure Gradient Calculation Across Interface

**Standard Momentum Equation** (single phase):
```
dv_i/dt = -Σ_j m_j (p_i/ρ_i² + p_j/ρ_j²) ∇W_ij
```

**Multi-Phase Modification**:
```
dv_i^(A)/dt = -Σ_{j ∈ Ω_A} m_j (p_i^(A)/ρ_i² + p_j^(A)/ρ_j²) ∇W_ij
              -Σ_{j ∈ Ω_B} m_j (p_i^(A)/ρ_i² + p_j^(B)/ρ_j²) ∇W_ij
```

**Key Issue**: Pressure discontinuity at interface.
- Gas pressure << liquid pressure (typically P_gas/P_liq ~ 0.001)
- Gradient can be computed consistently if both pressures known
- Requires accurate interface detection (discussed in §2)

---

## 2. Interface Handling and Surface Tension

### 2.1 Surface Tension: Continuum Surface Force (CSF) Model

**Physical Principle**:
- Surface tension acts as a body force concentrated at the interface
- CSF model smears this force over particles near interface
- Force proportional to curvature κ and surface tension coefficient σ

**CSF Force Implementation**:
```
F_ST = σ κ n̂ δ_interface
```
where:
- σ = surface tension coefficient (N/m): water-air ≈ 0.072 N/m
- κ = curvature (m⁻¹): κ = ∇·(n̂) where n̂ = ∇ϕ / |∇ϕ|
- ϕ = color function (phase indicator)
- n̂ = interface normal
- δ_interface = interface indicator function

**Discrete CSF on Particles**:
```
F_i^ST = σ κ_i n̂_i δ_i  (only for interface particles)

κ_i = -∇·(∇ϕ_i / |∇ϕ_i|)  (curvature from color function)
```

### 2.2 Color Function and Curvature Estimation

**Color Function** (phase indicator):
```
ϕ_i = (1/ρ_i) Σ_j (m_j / ρ_j) δ_AB W_ij
```
where:
- δ_AB = 1 if j ∈ phase B (different from i)
- δ_AB = 0 otherwise
- Normalized so ϕ ranges from 0 (pure A) to 1 (pure B)

**Curvature from Laplacian**:
```
κ_i = -(1 / |∇ϕ_i|) ∇·(∇ϕ_i)  [exact]
```

**Discretized via Second-Order Laplacian**:
```
∇²ϕ_i ≈ (2/Ω_i) Σ_j (m_j / ρ_j) (ϕ_j - ϕ_i) W_ij
```

**Gradient of Color Function**:
```
∇ϕ_i = Σ_j (m_j / ρ_j) (ϕ_j - ϕ_i) ∇W_ij
```

### 2.3 Implementation Considerations for CSF

**Interface Particle Selection**:
- Only apply surface tension to particles satisfying: 0 < ϕ_i < 1
- Typical threshold: apply for 0.01 < ϕ_i < 0.99 (near-interface particles)
- Threshold avoids noise from particles far from interface

**Curvature Computation Stability**:
- Curvature highly sensitive to ϕ gradient errors
- Smooth ϕ field before computing derivatives (moving least squares or kernel correction)
- Filter κ to remove spurious oscillations: κ_smoothed = (κ_old + κ_new) / 2

**Surface Tension Time Step Constraint**:
```
Δt_ST ≤ C √(ρ h³ / σ)  (capillary limit)
```
- For water-air with h = 0.001 m: Δt_ST ≈ 0.0005 s
- Much stricter than acoustic limit; often controls overall time step

### 2.4 Example: Droplet Oscillation Test

**Setup**:
- Single water droplet (radius r_0 = 0.01 m) in air
- Initial perturbation: r(θ, t=0) = r_0 (1 + ε Y_2^0 cos(θ))
- ε = 0.1 (10% amplitude, Y_2^0 = Legendre polynomial)
- Surface tension: σ = 0.072 N/m
- Domain: 0.04 m × 0.04 m × 0.04 m (3D) or 0.04 m × 0.04 m (2D)
- Particles: ~2000 liquid + 5000 gas

**Expected Behavior**:
- Mode-2 oscillation: frequency f = √(8σ / (ρ_l r_0³)) / (2π)
- Theoretical: f ≈ 200 Hz for water droplet
- Numerical: error < 3% on frequency, < 5% on damping rate
- Oscillation damps over ~20 periods due to viscosity + numerical dissipation

---

## 3. Handling Extreme Density Ratios

### 3.1 Challenge: ρ_heavy / ρ_light > 100

**Typical Cases**:
- Water-air: ρ_water / ρ_air ≈ 833
- Oil-water: ρ_oil / ρ_water ≈ 0.9 (opposite sign, easier)
- Liquid metal-air: ρ_metal / ρ_air > 1000

**Core Issues**:
1. **Pressure Error**: Momentum equation mixes terms O(p_heavy/ρ_heavy²) and O(p_light/ρ_light²)
   - Heavy fluid: small velocity change from large pressure gradient
   - Light fluid: huge velocity change from tiny pressure gradient (numerical noise)

2. **Density Oscillation**: Light-phase particles see frequent neighbor changes
   - Particle clustering in gas phase
   - Negative pressure (cavitation) in regions of rapid motion

3. **Interface Instability**: Rayleigh-Taylor at interface can grow uncontrollably

### 3.2 Mitigation Strategies

**1. Modified Pressure Gradient (Pressure-Weighted)**:
```
dv_i/dt = -Σ_j m_j [α_i p_i/ρ_i² + α_j p_j/ρ_j²] ∇W_ij
```
where α_α = ρ_α / (ρ_α + ρ_β) redistributes weight.

**2. Ghost Particles in Gas Phase**:
- Supplement gas particles with "ghost" particles near interface
- Ghosts mimic liquid properties locally (higher density)
- Prevents artificial pressure jumps
- Cost: ~20–30% more particles

**3. Unified SPH Formulation**:
- Use density-weighted average: ρ̄ = (ρ_A + ρ_B) / 2 or harmonic mean
- Pressure gradient: dv_i/dt ∝ (p_i + p_j) / ρ̄² instead of phase-specific ρ
- Reduces but doesn't eliminate disparity in pressure response

**4. Implicit Pressure Solver**:
- Use pressure-correction (incompressibility) rather than weakly-compressible EOS
- Solves Poisson equation for pressure (DFSPH, IISPH)
- Enforces ∇·v ≈ 0, naturally stabilizes density ratio problem
- Cost: iterative linear solver, but more robust

### 3.3 Recommended Approach: Composite Strategy

For ρ_ratio > 100:
1. Use unified density for pressure gradient calculation
2. Employ XSPH correction with reduced ε_heavy (e.g., ε = 0.3)
3. Add artificial viscosity: Π_ij ∝ (ρ_i + ρ_j) / 2 to stabilize shear
4. If still unstable, use ghost particles or implicit pressure solver

---

## 4. Free Surface Detection in Multi-Phase

### 4.1 Challenge: Interface vs. Free Surface

- **Interface**: liquid-gas boundary (both in domain)
- **Free surface**: liquid surface exposed to far-field (boundary of domain)

**Detection Methods**:

**1. Color Function Threshold**:
```
If 0 < ϕ_i < 1 and (∇ϕ_i · r̂_boundary > 0):
  particle i is on free surface
```
Check if normal points outward of domain.

**2. Neighbor Count Criterion**:
```
Free surface if: (number of neighbors of same phase) / (total neighbors) < threshold
threshold = 0.7 typical
```
Liquid particles at surface have fewer liquid neighbors.

**3. Curvature-Based**:
```
Free surface if κ_i > κ_threshold
threshold ≈ 1/R where R = domain radius
```
Curved interface toward gas indicates outer surface.

### 4.2 Treatment of Free Surface Particles

- **Pressure**: for free-surface particles, enforce p = p_atm (atmospheric)
- **Viscosity**: optional damping only for interior particles
- **Surface tension**: applies to free surface (different normal, but same curvature formula)

---

## 5. Smooth-Interface SPH Model (Recent Advances)

### 5.1 Problem with Classical CSF

- Discontinuity in ϕ across interface creates numerical noise
- Curvature κ computed from discontinuous ϕ is unreliable
- Surface tension force can oscillate and cause droplet fragmentation

### 5.2 Smooth-Interface Approach (2024 Methods)

**Key Innovation**: Represent interface as continuous transition over 2–3 kernel widths.

**Smooth Color Function**:
```
ϕ_smooth = (1 + tanh(β d(x))) / 2
```
where:
- d(x) = signed distance to interface (from level-set or implicit surface)
- β = steepness parameter (~10–20)
- Transition smooth, gradients well-defined

**Benefits**:
- Curvature κ computed stably from smooth ϕ
- Reduced spurious oscillations
- Better mass conservation (no particles jump across interface artificially)
- Surface tension numerically stable

**Implementation**:
1. Maintain implicit surface representation (level-set or NURBS spline)
2. At each time step, compute signed distance d_i for each particle
3. Update ϕ_i = (1 + tanh(β d_i)) / 2
4. Compute ∇ϕ and κ as before (now numerically cleaner)
5. Apply CSF force

**Performance Impact**: ~5–10% computational overhead from distance computation, but critical for long simulations (>1000 time steps) avoiding droplet fragmentation.

---

## 6. Practical Implementation: Two-Phase Simulation Setup

### 6.1 Initialization

**Step 1: Define Domain and Phases**
```
liquid_particles = []
gas_particles = []

# Liquid region: y < h_liquid
for position in grid:
    if position.y < h_liquid:
        liquid_particles.append(SPHParticle(position, phase='liquid'))
    else:
        gas_particles.append(SPHParticle(position, phase='gas'))
```

**Step 2: Assign Mass**
```
# Equal particle spacing Δx
m_liquid = ρ_liquid × (Δx)^d  (d = 2 or 3)
m_gas = ρ_gas × (Δx)^d

# Adjust gas count if masses very different
if m_gas / m_liquid < 0.001:
    # Use fewer, larger gas particles
    m_gas = ρ_gas × (2Δx)^d
    refine gas spacing
```

**Step 3: Smoothing Length**
```
h = 1.3 × Δx  # standard choice
h_liquid = h
h_gas = h  # same kernel for both phases
```

### 6.2 Time Integration Loop

**Pseudocode**:
```python
for t in 0...T:
    # 1. Update density (from all neighbors)
    for i in (liquid_particles + gas_particles):
        ρ_i = 0
        for j in neighbors(i, h):
            ρ_i += m_j × W(r_ij, h)

    # 2. Compute pressure (EOS)
    for i in (liquid_particles + gas_particles):
        if phase[i] == 'liquid':
            p_i = ρ_0 × c_s^2 / γ × ((ρ_i / ρ_0)^γ - 1)  # stiffened EOS
        else:
            p_i = ρ_0_gas × c_s_gas^2 / γ × ((ρ_i / ρ_0_gas)^γ - 1)

    # 3. Color function (for interface particles only)
    for i in liquid_particles:
        ϕ_i = 0
        for j in neighbors(i, h):
            if phase[j] == 'gas':
                ϕ_i += m_j / ρ_j × W(r_ij, h)
        ϕ_i /= (1/ρ_i) × (Σ_j m_j/ρ_j × W(r_ij, h))

    # 4. Curvature and surface tension (interface particles only)
    for i in liquid_particles:
        if 0.01 < ϕ_i < 0.99:
            ∇ϕ_i = Σ_j (m_j/ρ_j) × (ϕ_j - ϕ_i) × ∇W_ij
            ∇²ϕ_i = (2/Ω_i) × Σ_j (m_j/ρ_j) × (ϕ_j - ϕ_i) × W_ij
            κ_i = -∇²ϕ_i / max(|∇ϕ_i|, ε)  # avoid division by zero

    # 5. Momentum equation
    for i in (liquid_particles + gas_particles):
        a_i = -Σ_j m_j × (p_i/ρ_i^2 + p_j/ρ_j^2) × ∇W_ij
        a_i += +gravity (if applicable)
        if 0.01 < ϕ_i < 0.99:
            a_i += σ × κ_i × ∇ϕ_i / |∇ϕ_i|  # surface tension
        if use_viscosity:
            a_i += visc_force_ij

    # 6. Time integration (e.g., symplectic Euler)
    for i in (liquid_particles + gas_particles):
        v_i += a_i × Δt
        x_i += v_i × Δt

    # 7. Optional: correct positions (XSPH)
    for i in (liquid_particles + gas_particles):
        v_i_corrected = v_i - ε × Σ_j m_j/ρ_j × (v_i - v_j) × W_ij
```

### 6.3 Parameter Tuning Checklist

| Parameter | Liquid Water | Liquid Air | Notes |
|-----------|-------------|-----------|-------|
| ρ_0 | 1000 kg/m³ | 1.2 kg/m³ | Use stiffened EOS if density ratio > 10 |
| c_s | 50 m/s | 340 m/s | From EOS, sound speed in fluid |
| γ (polytropic) | 7.0 | 1.4 | water more stiff than air |
| σ (surface tension) | 0.072 N/m | — | only for liquid-gas interface |
| μ (viscosity) | 0.001 Pa·s | 1.8e-5 Pa·s | Include if Re_interface < 1000 |
| h | 1.3 Δx | 1.3 Δx | same for both phases |
| Δt | 0.1 × h/c_s | min(capillary limit) | capillary limit stricter for water-air |

---

## 7. Common Issues and Troubleshooting

### 7.1 Interface Fragmentation

**Symptom**: Liquid phase breaks into disconnected droplets prematurely; droplets shrink over time.

**Causes**:
1. Surface tension force too strong or curvature κ wildly oscillating
2. Particle spacing not fine enough at interface (Δx > h_capillary)
3. Ghost particles missing in gas phase

**Solutions**:
1. Reduce σ artificially for testing; verify it's not surface tension
2. Refine mesh at interface (use h-refinement or adaptive SPH)
3. Add ghost particles; or use implicit pressure solver (DFSPH)
4. Smooth color function ϕ before computing κ

### 7.2 Pressure Discontinuities and Oscillations

**Symptom**: Pressure jumps across interface; particles near interface have erratic acceleration.

**Causes**:
1. Pressure gradient formula not accounting for density jump
2. Color function ϕ discontinuous (sharp step at interface)
3. Density ratio ρ_liquid / ρ_gas extreme (> 500)

**Solutions**:
1. Use unified formulation: (p_i + p_j) / ρ̄² instead of phase-specific
2. Implement smooth color function (tanh-based) or filter ϕ
3. Use implicit pressure solver or ghost particles for extreme ratios

### 7.3 Density Oscillations in Gas Phase

**Symptom**: Gas particles exhibit density noise; pressure becomes noisy and sometimes negative.

**Causes**:
1. Gas EOS too stiff (sound speed too high) → small time step but still oscillations
2. Not enough gas neighbors per particle (sparse gas)
3. Particle clustering due to attractive interface force

**Solutions**:
1. Use weaker EOS (lower γ or stiffness constant)
2. Increase gas particle count; may double or triple for stability
3. Add damping: artificial viscosity or momentum-correcting filter

### 7.4 Interface Oscillation / "Rippling"

**Symptom**: Interface undulates with small-amplitude waves even in quiescent state.

**Causes**:
1. Curvature κ computed from noisy color function ϕ → noisy CSF force
2. Numerical parasitic currents (Marangoni-like spurious flows)
3. Time step too large, causing over-correction

**Solutions**:
1. Apply moving-average or bilateral filter to κ
2. Verify momentum conservation; check for spurious pressure oscillations
3. Reduce Δt; check CFL and capillary number condition
4. Use smooth color function or Laplacian smoothing of ϕ

---

## 8. Advanced Topics: Incompressibility in Multi-Phase

### 8.1 DFSPH (Divergence-Free SPH) for Multi-Phase

**Motivation**: Weakly compressible SPH suffers density oscillations in high-speed flows.

**Approach**:
- Solve pressure implicitly to enforce ∇·v ≈ 0 (incompressibility)
- Two-step: (i) divergence-free, (ii) surface tension
- Pressure correction ensures density fluctuations < 1% of ρ_0

**Multi-Phase Extension**:
- Solve single pressure field for entire domain (both phases)
- Each phase's pressure evolution coupled through interface
- Boundary condition at interface: p_liquid ≈ p_gas + σ κ

**Advantage**: Much smaller time steps not needed; larger Δt possible.

**Computational Cost**: Typical speedup 2–3× (fewer steps), but each step requires Poisson solve (~10 iterations).

---

## 9. Validation and Testing

### 9.1 Standard Test: Rising Bubble

**Setup**:
- Water tank: 0.06 m × 0.12 m (width × height)
- Air bubble: r = 0.01 m, initially at x = 0.03 m, y = 0.04 m
- Surface tension: σ = 0.072 N/m
- Particles: ~5000 water + ~2000 air

**Expected Trajectory** (Dandy & Leal 1989):
- Bubble rises due to buoyancy
- Path oscillates (zig-zag) due to wake interaction
- Terminal velocity: v_term ≈ 0.3 m/s (experiment)
- Eötvös number: Eo = ρ_l g r² / σ ≈ 0.9 (intermediate regime)

**Validation Metrics**:
- Rise velocity: ±10% of experiment
- Oscillation period: ±5%
- Bubble shape (aspect ratio): ±10%
- Work done against surface tension: within 5%

### 9.2 Capillary Rise in Tube

**Setup**:
- Vertical tube: r_tube = 0.005 m, height = 0.1 m
- Liquid: water, ρ = 1000 kg/m³, σ = 0.072 N/m
- Contact angle: θ = 0° (complete wetting)
- Gravity: g = 9.81 m/s²

**Analytical Solution**:
```
h_cap = 2σ cos(θ) / (ρ g r)  ≈ 0.0295 m for water in glass
```

**SPH Validation**:
- Simulate rise of water into tube
- Final equilibrium height should match within 3%
- Oscillations around h_cap damp exponentially (time const ~ √(ρ r³ / σ))

---

## 10. Multi-Phase SPH Parameter Summary

| Parameter | Symbol | Water | Air | Notes |
|-----------|--------|-------|-----|-------|
| Density | ρ | 1000 | 1.2 | Use actual values |
| Sound speed | c_s | 50 | 340 | Conservative for stability |
| Viscosity | μ | 0.001 | 1.8e-5 | Pa·s |
| Surface tension | σ | 0.072 | — | N/m, at water-air interface |
| Smoothing length | h | 1.3 Δx | 1.3 Δx | Same for both phases |
| EOS polytropic index | γ | 7.0 | 1.4 | Stiffened EOS for liquid |
| XSPH factor | ε | 0.5 | 0.5 | Reduce if interface unstable |
| Artificial viscosity | α | 0.05 | 0.05 | Reduce for surface tension |

