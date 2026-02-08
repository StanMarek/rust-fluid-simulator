# SPH Applications: Domain-Specific Setups and Validation

This guide covers best-practice setups for SPH simulations across different engineering and scientific domains.

---

## 1. Free-Surface Flows

### 1.1 Dam Break (Standard Benchmark)

**Description**: Water initially confined by a dam suddenly released. This is the canonical SPH validation test.

**Standard Geometry (Colagrossi & Landrini 2003)**:
- Reservoir width: W = 0.146 m, height: H = 0.146 m
- Channel length: L = 2.448 m, height: H_c = 0.6 m
- Particle spacing: Δx = 0.002 m (73 × 73 particles in reservoir)
- Initial density: ρ = 1000 kg/m³
- Smoothing length: h = 1.3 × Δx

**Expected Results**:
- Wave front reaches 1.33 m at t = 0.4 s (analytical: 1.281 m)
- Maximum displacement error: < 5% for well-tuned h, sound speed
- Rayleigh instability (particle finger-like protrusions): acceptable if h ≥ 1.2Δx
- Pressure remains near hydrostatic away from shock

**Key Tuning Parameters**:
- Sound speed: c = 50 m/s (wave celerity √(gH) ≈ 1.2 m/s, so Ma ≈ 0.024)
- Time step: Δt ≤ 0.1 × h / c_s ≈ 0.26 ms
- XSPH correction factor: ε = 0.5

### 1.2 Wave Propagation

**Setup**: Sinusoidal wave on shallow water surface propagates over flat bed.

**Parameters**:
- Wave amplitude: A = 0.01 m
- Wave length: λ = 0.5 m
- Water depth: H = 0.1 m
- Domain length: 2λ (periodic BC or absorbing)
- Particle spacing: Δx = λ / 50 = 0.01 m

**Expected Results**:
- Phase speed: c = √(gH) ≈ 0.99 m/s
- Dispersion relation preserved: ω² ≈ gk for shallow water
- Wave shape maintained for 2-3 periods before numerical dampening

**Validation Metric**: Track wave crest position vs. analytical solution; error should remain < 2% for 3 periods.

### 1.3 Sloshing (Confined Water with Oscillating Boundary)

**Setup**: Rectangular tank with water, subjected to horizontal sinusoidal motion.

**Parameters**:
- Tank: 1.0 m × 0.5 m (width × height)
- Initial fill: 60% of height
- Excitation: x(t) = A sin(ωt), A = 0.05 m, ω = 2π rad/s (T = 1 s)
- Particle spacing: Δx = 0.01 m
- Particles: ~5000 water + boundary particles

**Expected Results**:
- Free surface remains smooth (no fragmentation)
- Water depth oscillates with amplitude ~0.08–0.12 m (amplification at resonance)
- Energy dissipation from viscous damping, numerical dampening
- Pressure field oscillates in phase with surface elevation

**Tuning**: Artificial viscosity α = 0.05 for sloshing (< 0.05 causes instability, > 0.1 over-dampens).

---

## 2. Astrophysics

### 2.1 Star Formation and Gravitational Collapse

**Physics**: Dominant force is gravity (no pressure gradient in early collapse); SPH excellent for tracking fragmentation.

**Typical Setup**:
- Uniform density sphere (Bonnor-Ebert type), optionally with rotation
- Mass: 1 M_sun to 10⁶ M_sun depending on target
- Particle count: 10⁴–10⁶ particles for test runs, 10⁷–10⁸ for production
- Smoothing: h = 2–4 × initial neighbor distance
- Gravity: direct N-body or hierarchical tree (crucial for performance)

**Key Physics**:
- Isothermal EOS: P = c_s² ρ (c_s = 0.1–1.0 km/s typical)
- Fragmentation during collapse: monitoring core formation and mass distribution
- Angular momentum conservation: critical test for tree gravity codes

**Expected Results**:
- Collapse timescale: t_ff ~ 1/√(Gρ) matches free-fall time
- Core masses follow power-law distribution in fragmenting case
- Specific angular momentum of fragments conserved to ~1%

**Recommended Codes**: GADGET-3, SWIFT

### 2.2 Galaxy Simulations and Cosmological SPH

**Setup**: Dark matter + gas SPH in expanding universe with N-body gravity.

**Parameters**:
- Particle count: ~10⁶–10⁸ (gas + dark matter combined)
- Box size: 10–100 Mpc/h
- Redshift range: z = 127 (early) to z = 0 (present)
- Cooling physics: optionally include star formation, feedback
- Time step: adaptive, typically Δt ~ 0.1–0.5 Myr

**Validation**:
- Power spectrum of matter distribution vs. CMB predictions
- Galaxy stellar mass function matches observations
- Star formation history (SFR vs. redshift)

**Recommended Codes**:
- GADGET-3 (industry standard)
- SWIFT (newer, better task-based parallelism)
- GIZMO (alternative gravity/hydro solver)

---

## 3. Solid Mechanics and Fracture

### 3.1 Total Lagrangian SPH Formulation

**Key Difference from Eulerian**:
- Track material coordinates: particles represent fixed material points
- Deformation gradient F tracked explicitly
- Strain tensor computed as ε = (F^T F - I) / 2
- Stress derived from strain via constitutive model

**Setup Example: Uniaxial Tension**:
- Bar: 0.1 m × 0.01 m × 0.01 m
- Material: aluminum (E = 70 GPa, ν = 0.3, ρ = 2700 kg/m³)
- Particle spacing: Δx = 0.001 m (~10³ particles)
- Applied strain rate: ε̇ = 1 s⁻¹
- Boundary: fixed left end, prescribed displacement right end

**Expected Results**:
- Stress-strain follows elastic law σ = E ε until yield
- Localization (necking) forms at ~εp = 0.15 for ductile material
- Failure modes match physical observations (ductile vs. brittle)

### 3.2 Damage Models and Fracture

**Continuum Damage Mechanics**:
- Damage variable D ∈ [0, 1] reduces effective stress: σ_eff = (1 − D) σ
- Damage evolution: dD/dt = f(σ_eq, ε_p) (function of equivalent stress, plastic strain)
- Critical damage: D_crit = 0.9–0.99; beyond which element (particle) fails

**Crack Propagation**:
- Use gradient damage (smeared crack) or explicit crack tracking
- Stress concentration at crack tip naturally emerges in SPH (no mesh locking)
- G_I (Mode I stress intensity) validation: compare with Griffith formula

**Typical Test: Single Edge Notch Specimen**:
- Dimensions: 0.05 m × 0.1 m, notch depth a = 0.025 m
- K_I = Y σ √(πa), compare numerical vs. analytical
- Error tolerance: ±3% on stress intensity factor

### 3.3 Impact Dynamics

**Setup**: Projectile impacting plate at high velocity.

**Parameters**:
- Projectile: 10 mm diameter steel sphere, v = 100–500 m/s
- Plate: 50 mm × 50 mm × 5 mm steel
- Particle spacing: Δx = 0.5 mm (~10⁵ particles total)
- Material model: elastoplastic with strain-rate hardening

**Validation**:
- Crater depth and diameter vs. ballistic scaling laws
- Energy partitioning: kinetic → elastic + plastic + heat
- Spall ejecta velocity (if applicable)

---

## 4. Geotechnical Engineering

### 4.1 Landslides and Debris Flows

**Physics**:
- Dry or saturated granular material with internal friction
- Large deformations (Lagrangian tracking essential)
- Mohr-Coulomb plasticity: τ_max = c + σ_n tan(φ)
  - c = cohesion (0 for dry sand)
  - φ = friction angle (25°–40° typical)

**Typical Setup: Slope Failure**:
- Slope angle: β = 35°
- Material column: H = 5 m, L = 10 m
- Particles: ~5000
- Friction angle: φ = 30°
- Cohesion: c = 0 (dry) or 5 kPa (saturated)

**Expected Behavior**:
- Run-out distance: L_runout ≈ H / tan(β − φ) for frictionless limit
- Actual run-out: 20–50% farther due to strain-rate effects
- Deposit angle: φ_dep ≈ 3–5° steeper than friction angle

**Validation**:
- Compare run-out distance with analytical prediction and flume experiments
- Check energy dissipation: (KE + PE)_initial → plastic work + friction
- Pressure profile should vary linearly with depth in quiescent regions

### 4.2 Pore Pressure Effects

For saturated soil, couple SPH with pore-pressure field:
- Two-phase formulation: solid skeleton + fluid in pores
- Effective stress: σ' = σ − u (u = pore pressure)
- Mohr-Coulomb criterion in terms of σ'
- Consolidation effects if drainage is slow

**Setup Parameter Example**:
- Permeability: k = 10⁻⁴ m/s (sandy soil)
- Consolidation time: t_c ~ L² / c_v ≈ 100 s (L = 5 m, c_v = k/(γ_w m_v))
- Simulation duration: if t << t_c, assume undrained (u = constant); else model drainage

---

## 5. Biomedical: Blood Flow and Cell Mechanics

### 5.1 Hemodynamics (Blood Flow in Vessels)

**Setup**: Blood (modeled as Newtonian fluid) flowing through vessel.

**Parameters**:
- Vessel: 5 mm diameter artery, 5 cm length
- Blood viscosity: μ = 3.5 mPa·s
- Density: ρ = 1055 kg/m³
- Inlet flow: pulsatile (cardiac cycle, mean Re ≈ 500–1000)
- Particle spacing: Δx = 0.05 mm (~10⁴ particles)

**Key Physics**:
- Shear stress at wall: τ_w = μ (∂v_x/∂n) affects endothelial cells
- Turbulence at high Re: model with sub-grid viscosity (SPS)
- Blood cells (RBC, WBC) can be modeled as suspended particles

**Validation**:
- Wall shear stress oscillation magnitude and phase matches imaging data
- Mean velocity profile matches Hagen-Poiseuille in straight sections
- Pressure drop matches ΔP = 32 μ L v_mean / d² (Poiseuille)

### 5.2 Cell Mechanics and Deformation

**Setup**: Single cell in flow or subjected to stress.

**Model**:
- Cell membrane: elastic shell (Young's modulus E_m ≈ 10⁻⁶ N/m)
- Interior: viscoelastic fluid (μ_in ≈ 1 Pa·s)
- Cytoskeleton: reinforcing network

**Parameters**:
- Cell diameter: ~10 μm
- Particle spacing: Δx = 0.1–0.5 μm
- Applied stress: capillary flow at Re_cell < 1 (creeping flow)

**Expected**:
- Cell elongates: deformation index D = (L − W)/(L + W) increases with shear
- Tank-treading or tumbling motion in simple shear
- Transition occurs at critical capillary number Ca = μ γ̇ a / E (a = cell radius)

---

## 6. Industrial Applications

### 6.1 Metal Casting

**Physics**: Molten metal (~1500 K) flowing into mold cavity; solidification on walls.

**Setup**:
- Liquid steel: ρ = 7850 kg/m³, μ ≈ 5 mPa·s (high temperature)
- Mold: complex geometry with runners, gates, risers
- Domain particles: ~10⁴–10⁵
- Solidification: latent heat L ≈ 260 kJ/kg, Tm = 1811 K

**Simulation Aspects**:
- Mold-filling (Eulerian SPH) with free-surface tracking
- Thermal coupling: energy transport, phase change
- Shrinkage cavitation: volume change at solidification
- Validation: pour time, filling pattern vs. experimental casting X-rays

### 6.2 Additive Manufacturing (Metal Powder Fusion)

**Physics**: Laser melts metal powder, pool solidifies as laser moves.

**Setup**:
- Powder bed: 50 μm particles in loose random packing
- Laser: 200 W, scan speed 1 m/s, spot size 100 μm
- Domain: ~10×10 mm footprint, 1 mm depth
- Material: aluminum or titanium alloy

**Challenges**:
- Two-phase coupling: solid + liquid metal
- Marangoni convection (surface tension varies with T) dominates flow
- Powder particles not yet molten must track independently
- Phase change, latent heat

**Validation**:
- Melt pool geometry (width, depth) vs. high-speed imaging
- Solidification structure (grain size, orientation)

### 6.3 Coating and Thin Film Deposition

**Spray/Flow Coating**:
- Liquid polymer or paint sprayed onto substrate
- Must wet, spread, and harden
- Contact angle hysteresis, capillary flow

**Key Parameters**:
- Liquid viscosity: 10–100 mPa·s
- Surface tension: σ ≈ 0.02–0.03 N/m
- Substrate wettability: contact angle θ_eq = 20°–90°
- Film thickness: 10–100 μm

**Simulation Output**:
- Final coating thickness uniformity
- Drip/sag (if viscosity insufficient)
- Void fraction from trapped air bubbles

---

## 7. Ocean Engineering: Wave-Structure Interaction

### 7.1 Wave Impact on Fixed Structure (Breakwater)

**Setup**:
- Regular or irregular waves incident on vertical wall or caisson
- Wave: H = 0.5–2 m, T = 4–10 s
- Structure: concrete block, 2 m height above waterline
- Particles: ~10⁴–10⁵ for moderate scale
- Domain: wave propagation + structure + splash zone

**Measured Quantities**:
- Impact pressure on structure: P(t) vs. theoretical max ρ g H
- Run-up height: maximum water elevation on wall
- Force time series: F(t) integrated from pressure
- Overtopping flow rate (if water passes over structure)

**Validation**:
- Impact pressure peak ±10% of Miche formula (shallow water)
- Run-up height within ±15% of empirical Battjes-Groenewold correlation
- Compare forces with model tests (ISVA or similar tank data)

### 7.2 Ship Hydrodynamics and Wave-Body Interaction

**Setup**:
- Ship hull: 50–200 m length, various breadth-to-depth ratios
- Forward speed: Fn = V / √(gL) = 0.1–0.3 (Froude number)
- Free surface: must move and deform
- Particles: 10⁵–10⁷ depending on refinement

**Physics**:
- Boundary layer: Re = V L / ν > 10⁷ (turbulent)
- Wave making: transverse waves behind ship, kelvin-angle 19.47°
- Trim and sinkage: ship attitude changes due to hydrodynamic forces

**Validation**:
- Wave pattern matches Kelvin-wake angle and wave lengths
- Drag coefficient C_D = D / (½ ρ V² S) matches ITTC (International Towing Tank Conference)
- Trim and sinkage vs. model test data
- Pitch and heave response in wave conditions

---

## Summary Table: Domain Selection and Parameters

| Domain | SPH Variant | Typical N | Δx (m) | h/Δx | c_s (m/s) | Validation Test |
|--------|------------|----------|--------|------|-----------|-----------------|
| Free surface | Eulerian | 10⁴–10⁶ | 0.001–0.01 | 1.2–1.5 | 10–50 | Dam break |
| Astrophysics | Lagrangian | 10⁴–10⁸ | 0.01–1 pc | 2–4 | varies | Core formation |
| Solid mech. | Total Lagr. | 10³–10⁶ | 0.1–1 mm | 1.2–1.5 | — | Tensile test, KI |
| Geotechnical | Total Lagr. | 10³–10⁵ | 0.01–0.1 m | 1.3–1.5 | — | Slope run-out |
| Biomedical | Eulerian | 10⁴–10⁶ | 1–10 μm | 1.2–1.5 | 0.01 | Poiseuille, cell deform |
| Metal casting | Eulerian | 10⁴–10⁵ | 0.1–1 mm | 1.3–1.5 | 100–200 | Pour dynamics |
| Ocean eng. | Eulerian | 10⁴–10⁷ | 0.01–0.5 m | 1.2–1.5 | 10–30 | Wave run-up |

