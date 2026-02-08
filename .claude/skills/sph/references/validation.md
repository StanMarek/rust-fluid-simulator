# SPH Validation and Verification: Standards and Best Practices

Comprehensive guide to validating SPH implementations with standard test cases, convergence analysis, and validation metrics.

---

## 1. Fundamental Validation: Hydrostatic Pressure

**Purpose**: Most basic test; validates kernel computation and density summation.

### 1.1 Setup

**Geometry**:
- Single column of fluid: width W = 0.1 m, height H = 1.0 m
- Domain: W × H × W (3D) or W × H (2D)
- Particles: uniform spacing Δx = 0.01 m (100 particles vertically)
- No gravity initially; apply later

**Boundary Conditions**:
- Fixed walls (rigid boundary particles)
- Initial density: ρ_0 = 1000 kg/m³ everywhere
- Initial velocity: v = 0
- EOS: weakly-compressible, c_s = 50 m/s

### 1.2 Expected Solution

**Without Gravity**:
- Pressure everywhere: p = p_atm (constant)
- Density everywhere: ρ = ρ_0
- No acceleration: dv/dt = 0

**With Gravity (Earth)**: g = 9.81 m/s²
- Pressure varies linearly with depth:
  ```
  p(y) = p_atm + ρ_0 g (H - y)
  ```
- At bottom (y = 0): p_max = p_atm + 1000 × 9.81 × 1.0 ≈ 9.81 kPa (gauge)
- Density slightly increased at bottom due to compressibility:
  ```
  ρ(y) = ρ_0 [1 + p(y) / (c_s² ρ_0)]^(1/γ)  ≈ 1.0005 × ρ_0 (weak effect)
  ```
- Particles remain stationary (no net acceleration)

### 1.3 Verification

**Metric 1: Pressure Profile**
- Compute p(y) at each particle
- Fit to linear model: p = a₀ + a₁ × y
- Check: a₁ ≈ -ρ_0 g, |a₁ + ρ_0 g| / (ρ_0 g) < 0.5% for well-tuned h

**Metric 2: Particle Stability**
- Compute max acceleration: a_max = max(|dv/dt|) over all particles
- Expect: a_max < 0.1 m/s² (small numerical noise)
- Better codes: a_max < 0.01 m/s²

**Metric 3: Density Fluctuation**
- Compute ρ_std = std(ρ(t)) at equilibrium
- Expect: ρ_std / ρ_0 < 0.2% for quiescent fluid

**Typical Results** (well-tuned SPH):
- Pressure error vs. analytical: < 1%
- Max acceleration: 0.001–0.01 m/s²
- Particles remain within box after 1000 time steps

---

## 2. Standard Test Case: Dam Break

**Most important and widely-used validation test in SPH.**

### 2.1 Geometry and Setup

**Standard Configuration** (Colagrossi & Landrini 2003):
```
Reservoir:
  Width (x): W = 0.146 m
  Height (y): H = 0.146 m
  Depth (z): D = 0.4 m (3D) or plane strain (2D)

Channel:
  Length (x): L = 2.448 m
  Height (y): H_c = 0.6 m
  Depth (z): D = 0.4 m

Gate:
  Initial position: x_gate = 0.146 m (right edge of reservoir)
  Removal: instantaneous at t = 0
```

**Particle Setup**:
- Spacing: Δx = 0.002 m (≈ 2 mm)
- Reservoir particles: ~(73 × 73) in 2D ≈ 5329 particles
- Channel particles: ~(1224 × 3 / 0.002) ≈ 1836 particles
- Total: ~7000 particles (2D) or ~200k (3D full)
- Smoothing length: h = 1.3 Δx = 0.0026 m

**Material Properties**:
- Water density: ρ_0 = 1000 kg/m³
- Sound speed: c_s = 50 m/s (conservative for water)
- Artificial viscosity: α = 0.05, β = 0 (Monaghan form)
- XSPH correction: ε = 0.5
- Gravity: g = 9.81 m/s² (downward)

**Time Step**:
- CFL condition: Δt = 0.1 × h / c_s ≈ 0.52 ms
- Total simulation time: T = 0.6–1.0 s

### 2.2 Expected Results

**Wave Front Position** (most important metric):

| Time (s) | Analytical (m) | SPH Expected (m) | Tolerance (%) |
|----------|---|---|---|
| 0.1 | 0.350 | 0.352 | ±3 |
| 0.2 | 0.716 | 0.715 | ±2 |
| 0.3 | 1.056 | 1.050 | ±2 |
| 0.4 | 1.281 | 1.275 | ±3 |

**Front Velocity**:
```
v_front(t) ≈ √(2 g H) ≈ 1.695 m/s (initial, from dam break theory)
Decelerates as water spreads; v → 0.5–0.6 m/s by t=0.6 s
```

**Rayleigh-Taylor Instability**:
- Wave front exhibits finger-like structures (particle-scale ripples)
- Expected: wavy interface with amplitude ~h, wavelength ~3h
- **Acceptable**: perturbations < 2 mm (0.1% of dam height)
- **Problem**: perturbations > 5 mm indicates h too small or noise

**Pressure Field**:
- Shock front: pressure spike up to ~20 kPa (peak amplitude)
- Behind front: pressure gradually decreases
- Hydrostatic equilibrium far behind front
- Negative pressure (cavitation): should not occur; if it does, c_s too small

**Energy**:
- Initial: E_0 = potential energy = ρ g V H/2 ≈ 52.5 J (2D unit depth)
- Kinetic + potential energy vs. time should plateau at ~70–80% of E_0 (rest dissipated by viscosity + numerics)

### 2.3 Convergence: h-Refinement

**Study**: Vary Δx (and h proportionally) while keeping L/Δx constant.

**Parameters**:
```
Case A: Δx = 0.002 m (baseline, h = 0.0026 m)
Case B: Δx = 0.001 m (h = 0.0013 m)
Case C: Δx = 0.005 m (h = 0.0065 m)
```

**Metric**: Front position x_f(t = 0.4 s)

Expected Convergence:
```
Case A: x_f = 1.275 m (error ≈ -0.5% vs. analytical 1.281 m)
Case B: x_f = 1.278 m (error ≈ -0.2%)
Case C: x_f = 1.260 m (error ≈ -1.6%)
```

**Conclusion**: Error decreases with finer mesh → code converging with correct order.

---

## 3. Cavity Flows and Shear-Driven Test Cases

### 3.1 Lid-Driven Cavity

**Geometry**:
- Square cavity: 0.1 m × 0.1 m
- Viscous Newtonian fluid: ν = 0.01 m²/s (or 0.001 for lower Re)
- Particle spacing: Δx = 0.005 m (400 particles)

**Boundary Conditions**:
- Top wall (y = 0.1 m): moving with velocity v_lid = 1.0 m/s in x-direction
- Other three walls: no-slip (v = 0)
- No gravity; closed domain (no free surface)

**Physics**:
- Reynolds number: Re = ρ v_lid L / μ = 1.0 × 1.0 × 0.1 / 0.01 = 10
- Low Re: primary vortex centered near (0.5, 0.7)
- Secondary vortices in bottom corners (weak for Re = 10)

### 3.2 Expected Results (Re = 10)

**Velocity Profile**:
- Along vertical centerline (x = 0.05 m): velocity u(y) increases from 0 (bottom) to 1.0 m/s (top)
- Analytical solution available from FEM literature
- SPH should match to within ±5% (smoothing over particle width)

**Primary Vortex Center**: (x_c, y_c) ≈ (0.05, 0.068) for Re = 10
- SPH prediction: within ±0.005 m (typical)

**Pressure**: p = 0 (atmospheric) since closed domain and no gravity

### 3.3 Validation Against Analytical/Numerical

**Comparison with Ghia et al. (1982)** (benchmark FEM solution):
- Extract u-velocity along centerline: u_center = u(y, x=0.05)
- Plot vs. Ghia's results (published table)
- Root-mean-square error: RMSE = √(Σ(u_SPH - u_Ghia)²/N)
- Acceptable: RMSE < 0.05 m/s for Re = 10

---

## 4. Poiseuille Flow (Channel Flow)

**Canonical test for viscous flow validation.**

### 4.1 Setup

**Geometry**:
- Channel: length L = 1.0 m, height H = 0.1 m, depth (3D) W = 0.1 m
- Particles: Δx = 0.01 m (~10 × 100 particles)

**Boundary Conditions**:
- Pressure-driven flow: inlet (x=0) p = p_in, outlet (x=L) p = p_out
- Pressure difference: Δp = p_in - p_out = 1000 Pa
- Top/bottom walls: no-slip (v = 0)
- Side walls (3D): periodic or no-slip

**Fluid Properties**:
- Density: ρ = 1000 kg/m³
- Viscosity: μ = 0.1 Pa·s → ν = 0.0001 m²/s
- No gravity, incompressible-like (constant ρ)

### 4.2 Analytical Solution

**Fully-Developed Parabolic Profile**:
```
u(y) = (Δp / (2 μ L)) × (H y - y²)  [maximum at y = H/2]
u_max = Δp H² / (8 μ L) = 1000 × 0.01² / (8 × 0.1 × 1.0) = 1.25 m/s
```

**Mean Velocity**: u_mean = u_max / 3 ≈ 0.417 m/s

**Volumetric Flow Rate** (per unit depth): Q = u_mean × H = 0.0417 m²/s

### 4.3 SPH Validation

**Metric 1: Velocity Profile**
- Extract u(y) at several x-positions (x = 0.2, 0.5, 0.8 m)
- At x = 0.5 m (fully developed), should match parabolic profile
- Error: |u_SPH(y) - u_analytical(y)| / u_max < 3% (point-wise)

**Metric 2: Pressure Gradient**
- Compute dp/dx from pressure field
- Should be approximately constant: |dp/dx| ≈ Δp/L = 1000 Pa/m
- Check: |dp/dx_SPH - 1000| / 1000 < 5%

**Metric 3: Flow Rate Conservation**
- Integrate Q = ∫ u dA across section
- Check at different x-positions: should be constant (no loss/gain)
- Error: |Q(x) - Q_analytical| / Q_analytical < 2%

---

## 5. Taylor-Green Vortex (Temporal Decay)

**Excellent test for inviscid code or code with controlled viscosity.**

### 5.1 Setup

**Periodic Domain**: [0, 1] × [0, 1] (2D) or [0, 1]³ (3D), periodic BC all sides

**Initial Velocity Field** (2D):
```
u(x, y) = U_0 sin(2π x) cos(2π y)
v(x, y) = -U_0 cos(2π x) sin(2π y)
w(x, y) = 0
```
- U_0 = 1.0 m/s
- Velocity magnitude: |v| = U_0 √(sin²(2πx)cos²(2πy) + cos²(2πx)sin²(2πy))
- **Divergence-free**: ∂u/∂x + ∂v/∂y = 0 ✓

**Initial Pressure**: p = p_atm + (ρ U_0²/16) [cos(4πx) + cos(4πy)]
(derived from Euler equation)

### 5.2 Analytical Decay (No Viscosity)

**Inviscid Euler**: velocity field should NOT decay
- For SPH with low artificial viscosity: kinetic energy should remain ~constant
- Any decay is purely numerical dissipation

**With Viscosity** (ν > 0):
```
u(x, y, t) = U_0 exp(-8π² ν t) sin(2πx) cos(2πy)
Kinetic energy: E(t) = E_0 exp(-16π² ν t)
```

### 5.3 Validation Metrics

**Metric 1: Energy Decay**
- Compute total kinetic energy: E(t) = (1/2) ρ Σ_i |v_i|² V_i
- Plot E(t) vs. t on log scale
- Should be straight line: ln(E) = ln(E_0) - 16π² ν t
- Slope = -16π² ν gives effective viscosity (≈ true ν if artificial viscosity tuned)

**Metric 2: Enstrophy Decay**
- Enstrophy: Z = (1/2) ∫ |∇×v|² dA = (1/2) ρ Σ_i ω_i² V_i
- For TGV: ω_z = 2π U_0 cos(2πx) sin(2πy)
- Analytical decay: Z(t) = Z_0 exp(-32π² ν t) (faster than energy)

**Metric 3: L∞ Error on Velocity**
- Max point-wise error: e_∞ = max(|u_SPH - u_analytical|)
- For periodic domain without slip, should remain < 0.05 × U_0 even at t = 10/f (f = frequency)

---

## 6. Convergence Studies: h-Refinement and p-Refinement

### 6.1 h-Refinement (Mesh Refinement)

**Procedure**:
1. Select test case (e.g., Poiseuille flow, dam break)
2. Define coarse resolution: h₁ = baseline
3. Run three cases: h₁, h₂ = h₁/2, h₃ = h₁/4 (halving each time)
4. Select error metric (e.g., L∞ norm of velocity)
5. Compute error: E_i = ||u_SPH - u_analytical||_∞

**Results Table** (example from Poiseuille):

| Case | h (m) | Particles | E_∞ (m/s) | Ratio | Convergence Rate |
|------|-------|-----------|-----------|-------|---|
| 1 | 0.01 | 1k | 0.0320 | — | — |
| 2 | 0.005 | 8k | 0.0089 | 3.60 | ~1.85 ≈ 2 |
| 3 | 0.0025 | 64k | 0.0022 | 4.05 | ~2.0 ≈ 2 |

**Interpretation**:
- Ratio ≈ 2 for each halving of h → 2nd-order convergence (expected for SPH with cubic kernel)
- Plot log(E) vs. log(h): slope ≈ 2 confirms O(h²) convergence

### 6.2 p-Refinement (Kernel Order)

Compare different kernel functions (same h):

| Kernel | Order | Taylor Series Error |
|--------|-------|---|
| Cubic spline | 3 (quintic) | O(h⁴) or O(h⁵) |
| Quintic spline | 5 | O(h⁶) |
| Wendland C⁴ | 4 | O(h⁴) |

**Example**:
```
Fixed h = 0.005 m

Cubic kernel: E = 0.0089 m/s
Quintic kernel: E = 0.0052 m/s  (improvement: 41%)
Wendland C⁴: E = 0.0061 m/s    (improvement: 31%)
```

**Decision**: Higher-order kernels give better accuracy at same h, but cost more (more neighbors, denser computation).

---

## 7. Energy Conservation Monitoring

### 7.1 Total Energy Balance

**Definition**:
```
E_total(t) = E_kinetic + E_potential + E_internal
           = (1/2) Σ_i m_i |v_i|² + Σ_i m_i g y_i + Σ_i u_i m_i

where u_i = internal energy per unit mass (= 0 for incompressible; ≠ 0 for compressible)
```

### 7.2 Expected Behavior

**Inviscid, Conservative Scheme**:
- E_total should remain constant (< 1% variation)
- If decaying: indicates numerical dissipation

**Viscous Flow**:
- E_kinetic + E_potential should decay
- Rate of decay: dE/dt = -D (dissipation rate)
- D = ∫ σ : ∇v dV (internal viscous work)
- For Poiseuille: D = μ U_0² / (4 H) [per unit depth]

### 7.3 Energy Monitoring Example: Dam Break

**Timeline**:
- t = 0: E_total = ρ g V H/2 ≈ 52.5 J (potential, relative to final surface)
- t = 0.4 s: E_kinetic ≈ 40 J, E_potential ≈ 4 J → total ≈ 44 J (84% of initial)
- Dissipated by viscosity + artificial damping: ~8% over first pulse

**Check**: Plot E_kinetic(t) + E_potential(t)
- Should decrease monotonically (if viscous) or oscillate with decay (if compressible)
- Spike in E_potential near t = 0.15 s (wave breaks, maximum height)
- Final E_total / E_initial should be ≥ 0.7 (not too much dissipation)

---

## 8. Comparison with Analytical Solutions

### 8.1 Available Analytical Solutions in SPH

| Problem | Analytical Form | Domain | Remarks |
|---------|---|---|---|
| Hydrostatic pressure | p(y) = p_0 + ρgy | Any | Exact verification |
| Poiseuille flow | u(y) parabolic | Channel | Fully developed only |
| Couette flow | u(y) linear | Gap | Shear-driven; simple |
| Stokes flow (Re<<1) | Low Re asymptotics | Around sphere, etc. | Creeping flow regime |
| Sound wave | u, p sinusoidal | Unbounded | Linear acoustics |
| Taylor-Green vortex | u = U₀ sin... exp(-νt) | Periodic domain | Decay rate measures ν |

### 8.2 Error Norms

**L₁ Error** (mean absolute error):
```
E₁ = (1/N) Σᵢ |u_SPH(xᵢ) - u_analytical(xᵢ)|
```

**L₂ Error** (RMS error):
```
E₂ = √((1/N) Σᵢ |u_SPH(xᵢ) - u_analytical(xᵢ)|²)
```

**L∞ Error** (maximum error):
```
E∞ = max |u_SPH(xᵢ) - u_analytical(xᵢ)|
```

**Typically**: E∞ ≥ E₂ ≥ E₁; for smooth analytical solutions, E₂ ≈ 0.1–0.5 × E∞

---

## 9. Comparison with Experimental Data

### 9.1 Wave Tank Experiments

**Dam Break Experiment** (Lauber & Hager 1998):
- Measured free surface profile at several x-positions and times using high-speed camera
- Published data: x_f(t), y(x, t) as curves and point measurements
- Available at: https://www.asce.org/ (ASCE Digital Library)

**SPH Validation Protocol**:
1. Run SPH dam break with same geometry as experiment
2. Extract free surface (particles at interface)
3. Fit surface to curve (e.g., spline)
4. Compare x_f(t) with experimental data
5. Quantify error: Δx_f = x_f,SPH - x_f,exp
6. Accept if: |Δx_f| / H_dam < 3% for t ≤ 0.4 s

### 9.2 Published Experimental Datasets

| Problem | Data Source | Key Output |
|---------|---|---|
| Dam break | Colagrossi & Landrini (2003) | Wave front position, water depth |
| Sloshing | Akyildiz & Ünal (2006) | Surface elevation at wall |
| Wave impact | Oumeraci et al. (HYDRALAB) | Impact pressure on structure |
| Droplet collision | Ashgriz & Poo (1990) | Coalescence dynamics, masses |
| Blood flow (in vitro) | Papaioannou et al. | WSS (wall shear stress), separation zones |

---

## 10. Common Validation Pitfalls

### 10.1 Insufficient Domain Size

**Problem**: Particles feel artificial boundaries too strongly.

**Example**: Dam break in narrow channel (width = H)
- Side walls reflect waves, create spurious pressure
- Wave front position corrupted by wall effects

**Solution**: Ensure domain width >> dam height; typically W ≥ 2H.

### 10.2 Non-Uniform Initial Particle Distribution

**Problem**: Irregular particle placement causes density noise.

**Symptom**: Pressure oscillates even in hydrostatic case.

**Solution**: Always use perfectly uniform grid for initialization:
```python
# Good
for i in range(n_x):
    for j in range(n_y):
        x = i * dx
        y = j * dx
        particles.append((x, y))

# Bad (avoid)
x = random.uniform(0, L)
y = random.uniform(0, H)
particles.append((x, y))
```

### 10.3 Insufficient Time Step Accuracy

**Problem**: Δt chosen conservatively; accumulates time-integration error.

**Example**: Δt = 0.1 × h/c_s (CFL 0.1) is safe but may be over-conservative.
- Test with Δt = 0.5 × h/c_s if CFL ≤ 0.4 is maintained dynamically
- Measure relative error change; if < 2% improvement, reduce Δt

### 10.4 Wrong Kernel Normalization

**Problem**: Kernel integral ≠ 1 due to boundary effects.

**Symptom**: Particle density ≠ ρ_0 even in interior; all particles see error.

**Solution**: Use kernel correction (e.g., CSPM):
```
W_corrected(r) = W(r) / Σⱼ (mⱼ/ρⱼ) W(rᵢⱼ)
```

### 10.5 Inconsistent Mass Weighting in Multi-Phase

**Problem**: Two-phase density computed without accounting for ρ differences.

**Symptom**: Interface particles have anomalously high/low density.

**Solution**: Always include mass-per-density weighting:
```
ρᵢ = Σⱼ (mⱼ/ρⱼ)⁻¹ W(rᵢⱼ)  [WRONG]
ρᵢ = Σⱼ mⱼ W(rᵢⱼ)            [CORRECT]
```

### 10.6 Overlooking Time-Dependent Phenomena

**Problem**: Running simulation too short; missing transient effects.

**Example**: Sloshing tank
- First few cycles: initial transients (energy redistribution)
- Need ~3–5 periods to reach quasi-steady oscillation
- Measure only from period 3 onward

---

## 11. Validation Checklist for New SPH Implementation

### Minimal Checklist (< 1 week of testing)

- [ ] Hydrostatic pressure test: density, pressure, no acceleration
- [ ] Dam break: wave front position at t = 0.4 s within ±5%
- [ ] Poiseuille flow: velocity profile matches analytical within ±5%
- [ ] Energy conservation: total energy varies < 10% over simulation
- [ ] Convergence study (h-refinement): 2nd-order convergence observed

### Comprehensive Checklist (1–4 weeks)

- [ ] All minimal checklist items
- [ ] Lid-driven cavity: vortex center, streamlines match literature
- [ ] Multi-phase rising bubble: trajectory, terminal velocity agree
- [ ] Surface tension test: droplet oscillation frequency within ±3%
- [ ] Convergence in p (kernel order): quintic smoother than cubic
- [ ] CFL sensitivity: robust for CFL ∈ [0.1, 0.4]
- [ ] Comparison with published experimental data
- [ ] Negative pressure (cavitation) never occurs in hydrostatic case
- [ ] Artificial viscosity tuning: document optimal α for each problem class

### Expert Validation (4+ weeks)

- [ ] All comprehensive checklist items
- [ ] FSI benchmark (if applicable): structure deformation matches FEM
- [ ] Astrophysical: core fragmentation matches adaptive mesh codes
- [ ] Biomedical (if applicable): hemodynamics vs. in vitro data
- [ ] Publication review: results match or exceed peer-reviewed validation studies

---

## 12. Documenting Validation Results

### Example Report Structure

```
VALIDATION REPORT: Custom SPH Code v1.0

1. Implementation Details
   - Kernel: cubic spline
   - EOS: weakly compressible, γ = 7
   - Pressure gradient: symmetric formulation
   - Viscosity: Laminar (Cleary & Monaghan), α = 0.05

2. Validation Tests

   2.1 Hydrostatic Pressure
       [Graph: p(y) numerical vs. analytical]
       Result: L∞ error = 0.8%, PASS

   2.2 Dam Break (Standard Geometry)
       [Graph: x_f(t) numerical vs. Colagrossi & Landrini 2003]
       Result: Error at t=0.4s = 1.2%, PASS

   2.3 Poiseuille Flow
       [Graph: u(y) velocity profile]
       Result: L∞ error = 2.1%, PASS

   2.4 Energy Conservation (Dam Break)
       [Graph: E(t) vs time]
       Result: Dissipation 8% over 0.6s, acceptable

   2.5 h-Refinement Convergence
       [Table and log-log plot]
       Result: O(h²) convergence confirmed, PASS

3. Conclusion
   Code validated for 2D free-surface flows in range 10k-100k particles.
   Recommended: use h ≥ 1.2Δx, c_s ≥ 50 m/s for standard water problems.
```

