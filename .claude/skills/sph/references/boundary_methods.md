# SPH Boundary Handling Methods

A comprehensive reference for implementing boundary conditions in Smoothed Particle Hydrodynamics (SPH) simulations.

## 1. Ghost (Mirror) Particles

Ghost particles (also called mirror particles) are the most widely used boundary handling method in SPH. They create a virtual layer of particles that mirror the fluid properties across a boundary to enforce boundary conditions.

### 1.1 Generation Algorithm

Ghost particles are created by reflecting fluid particles across the boundary surface.

**Basic Algorithm:**
```pseudocode
function generateGhostParticles(fluidParticles, boundary, h):
    ghostParticles = []

    for each fluidParticle in fluidParticles:
        // Check if particle is near boundary
        distanceToBoundary = computeSignedDistanceToBoundary(fluidParticle.position, boundary)

        if distanceToBoundary < h then
            // Create ghost particle by reflection
            normalVector = computeOutwardNormal(fluidParticle.position, boundary)
            mirrorDistance = 2.0 * abs(distanceToBoundary)
            ghostPosition = fluidParticle.position - normalVector * mirrorDistance

            ghostParticle = createParticle()
            ghostParticle.position = ghostPosition
            ghostParticle.velocity = computeGhostVelocity(fluidParticle, normalVector, boundaryCondition)
            ghostParticle.density = fluidParticle.density
            ghostParticle.pressure = fluidParticle.pressure
            ghostParticle.type = GHOST

            ghostParticles.append(ghostParticle)

    return ghostParticles

function computeSignedDistance(particlePos, boundary):
    // Returns negative if inside domain, positive if outside
    // Distance to nearest point on boundary surface
    closestPoint = projectPointOntoBoundary(particlePos, boundary)
    distance = magnitude(particlePos - closestPoint)

    if isPointInsideDomain(particlePos, boundary) then
        return -distance
    else
        return distance
```

### 1.2 Property Assignment for No-Slip Walls

No-slip walls require the fluid velocity to match the wall velocity at the boundary.

**Velocity Assignment (No-Slip):**
```pseudocode
function assignNoSlipVelocity(fluidParticle, ghostParticle, wallVelocity, normalVector):
    // Mirror velocity component normal to wall, flip tangential
    // Result: ghost particle velocity mirrors the reflected motion

    fluidVelocityNormal = dot(fluidParticle.velocity, normalVector) * normalVector
    fluidVelocityTangent = fluidParticle.velocity - fluidVelocityNormal

    // No-slip: reflect normal component, keep tangential matching wall
    ghostVelocityNormal = -fluidVelocityNormal + 2.0 * wallVelocity
    ghostVelocityTangent = -fluidVelocityTangent + 2.0 * projectOntoTangentPlane(wallVelocity, normalVector)

    ghostParticle.velocity = ghostVelocityNormal + ghostVelocityTangent

    // Alternative simpler form (stationary wall):
    // ghostParticle.velocity = -fluidParticle.velocity
```

**Pressure Assignment (No-Slip):**
```pseudocode
function assignNoPressure(ghostParticle, fluidParticle):
    // Option 1: Mirror pressure (maintains pressure jump at wall)
    ghostParticle.pressure = fluidParticle.pressure

    // Option 2: Zero pressure (for free-slip variants)
    // ghostParticle.pressure = 0.0

    // Option 3: Computed from energy balance
    // ghostParticle.pressure = fluidParticle.pressure + 0.5 * density * velocity_normal^2
```

### 1.3 Free-Slip Wall Properties

Free-slip boundaries allow tangential motion while preventing penetration.

**Velocity Assignment (Free-Slip):**
```pseudocode
function assignFreeSlipVelocity(fluidParticle, ghostParticle, wallVelocity, normalVector):
    // Reflect only the normal component
    // Keep tangential component unchanged (or match wall tangential velocity)

    fluidVelocityNormal = dot(fluidParticle.velocity, normalVector) * normalVector
    fluidVelocityTangent = fluidParticle.velocity - fluidVelocityNormal

    ghostVelocityNormal = -fluidVelocityNormal + wallVelocity_normal
    ghostVelocityTangent = fluidVelocityTangent  // Unchanged, no friction

    ghostParticle.velocity = ghostVelocityNormal + ghostVelocityTangent
```

### 1.4 Multi-Layer Ghost Particles

A single layer of ghost particles can result in pressure spikes and particle leakage. Multiple layers improve accuracy and stability.

**Multi-Layer Generation:**
```pseudocode
function generateMultiLayerGhostParticles(fluidParticles, boundary, h, numLayers):
    allGhostParticles = []

    // Track which fluid particles have been used
    processedParticles = set()

    for layer = 1 to numLayers:
        ghostParticles = []

        for each fluidParticle in fluidParticles:
            if fluidParticle in processedParticles then
                continue

            distanceToBoundary = computeSignedDistanceToBoundary(fluidParticle.position, boundary)

            if distanceToBoundary < (layer) * h then
                normalVector = computeOutwardNormal(fluidParticle.position, boundary)

                // Create ghost particle at distance 2*layer*h from boundary
                mirrorDistance = 2.0 * layer * abs(distanceToBoundary)
                ghostPosition = fluidParticle.position - normalVector * mirrorDistance

                ghostParticle = createParticle()
                ghostParticle.position = ghostPosition
                ghostParticle.velocity = computeGhostVelocity(fluidParticle, normalVector, BC)
                ghostParticle.density = fluidParticle.density
                ghostParticle.pressure = fluidParticle.pressure
                ghostParticle.layer = layer
                ghostParticle.type = GHOST

                ghostParticles.append(ghostParticle)

                if layer == numLayers then
                    processedParticles.insert(fluidParticle)

        allGhostParticles.extend(ghostParticles)

    return allGhostParticles
```

### 1.5 Required Number of Layers

**Recommended Guidelines:**
- **Minimum: 2-3 layers** for most applications
  - 1 layer: Fast but inaccurate, high pressure noise
  - 2 layers: Good balance between accuracy and cost
  - 3 layers: Better continuity, smoother pressure field
  - 4+ layers: Diminishing returns, primarily for high-precision simulations

**Selection Criteria:**
- Use 2 layers for real-time applications with smooth boundaries
- Use 3 layers for high-pressure accuracy requirements
- Use 3-4 layers for strong impact/shock simulations
- Use more layers for very small h (higher spatial resolution)

**Performance:** Each layer adds ~25-40% computational cost but significantly reduces pressure spikes.

### 1.6 Implementation Pseudocode (Complete)

```pseudocode
class GhostParticleManager:

    function initialize(kernel, h, nu):
        this.kernel = kernel
        this.h = h
        this.nu = nu  // viscosity
        this.ghostParticles = []

    function updateGhosts(fluidParticles, boundaries, wallVelocities):
        this.ghostParticles.clear()

        for each boundary in boundaries:
            ghosts = this.generateLayeredGhosts(fluidParticles, boundary,
                                               wallVelocities[boundary])
            this.ghostParticles.extend(ghosts)

    function generateLayeredGhosts(fluidParticles, boundary, wallVelocity):
        ghosts = []

        // Layer 1: Primary ghost layer
        for each particle p in fluidParticles:
            dist = signedDistanceToBoundary(p.position, boundary)
            if dist >= -this.h and dist < 0:
                ghost = mirrorParticle(p, boundary, wallVelocity, 1)
                ghost.damping = 1.0  // Full strength
                ghosts.append(ghost)

        // Layer 2: Secondary ghost layer
        for each particle p in fluidParticles:
            dist = signedDistanceToBoundary(p.position, boundary)
            if dist >= -2*this.h and dist < -this.h:
                ghost = mirrorParticle(p, boundary, wallVelocity, 2)
                ghost.damping = 0.8  // Reduced strength
                ghosts.append(ghost)

        // Layer 3: Tertiary ghost layer (optional)
        for each particle p in fluidParticles:
            dist = signedDistanceToBoundary(p.position, boundary)
            if dist >= -3*this.h and dist < -2*this.h:
                ghost = mirrorParticle(p, boundary, wallVelocity, 3)
                ghost.damping = 0.5  // Further reduced
                ghosts.append(ghost)

        return ghosts

    function mirrorParticle(fluidParticle, boundary, wallVelocity, layer):
        normal = outwardNormal(fluidParticle.position, boundary)
        dist = signedDistanceToBoundary(fluidParticle.position, boundary)

        ghost = ParticleData()
        ghost.position = fluidParticle.position - normal * (2.0 * layer * abs(dist))
        ghost.velocity = fluidParticle.velocity - 2.0 * dot(fluidParticle.velocity - wallVelocity, normal) * normal
        ghost.density = fluidParticle.density
        ghost.pressure = fluidParticle.pressure
        ghost.mass = fluidParticle.mass
        ghost.type = BOUNDARY_PARTICLE
        ghost.isGhost = true
        ghost.layer = layer

        return ghost

    function getParticlesForForceCalculation():
        // Return all particles (fluid + ghosts) for SPH force calculations
        return this.ghostParticles
```

---

## 2. Repulsive Force (Lennard-Jones Style)

Repulsive forces provide an alternative to ghost particles by pushing fluid particles away from walls.

### 2.1 Lennard-Jones Formula

**Basic Formulation:**
```
f_wall(r) = (d₀/r)^n - (d₀/r)^m  · C_rep
```

where:
- `r`: distance from particle to wall
- `d₀`: equilibrium distance (typically 1.2 * h)
- `n`: repulsive exponent (typically 4)
- `m`: attractive exponent (typically 2)
- `C_rep`: repulsion coefficient

**Common Simplified Form:**
```
f_wall(r) = α * (d₀ - r) / r   for r < d₀
f_wall(r) = 0                  for r ≥ d₀
```

**Monaghan-Style Boundary Repulsion:**
```
f_wall = -C_rep * W(r, h) * n̂

where:
- W(r, h): kernel function value
- n̂: outward normal from wall
- C_rep: coefficient (typically 0.01 to 0.1 * c² * ρ * h)
```

### 2.2 Parameter Selection

**Equilibrium Distance (d₀):**
```pseudocode
d₀ = (1.0 to 1.3) * h

// Larger d₀: earlier repulsion, prevents penetration better
// Smaller d₀: particles can get closer to wall, more contact
// Typical: d₀ = 1.2 * h
```

**Repulsion Coefficient (α):**
```pseudocode
// For incompressible fluid with sound speed c:
α = 0.01 * c² * ρ * h

// Adjust based on velocity:
α = 0.1 * ρ * U² * h  // U is characteristic velocity

// Conservative (prevent penetration):
α_min = 0.05 * c² * ρ * h

// Aggressive (very stiff wall):
α_max = 0.5 * c² * ρ * h

// Default recommendation:
α = 0.1 * c² * ρ * h
```

**Algorithm:**
```pseudocode
function computeWallRepulsionForce(particle, boundaries):
    force = Vector3(0, 0, 0)

    for each boundary in boundaries:
        // Find closest point on boundary
        closestPoint = projectParticleOntoBoundary(particle.position, boundary)
        distance = magnitude(particle.position - closestPoint)
        normal = normalize(particle.position - closestPoint)

        if distance < d₀:
            // Penetration depth
            penetration = d₀ - distance

            // Repulsive force magnitude
            forceMagnitude = α * penetration / distance

            // Velocity component normal to wall
            velocityNormal = dot(particle.velocity, normal)

            // Damping term (prevents particle overshooting)
            damping = 0.5 * particle.mass * velocityNormal

            force += (forceMagnitude - damping) * normal

    return force
```

### 2.3 Pros and Cons

**Advantages of Repulsive Forces:**
- No extra particles needed (lower memory)
- Simpler to implement
- Works naturally with time stepping
- Can be easily made position and velocity dependent
- Good for complex boundaries (no discretization needed)
- No multi-layer complexity

**Disadvantages of Repulsive Forces:**
- Pressure field undefined at wall (no wall pressure available)
- Cannot enforce no-slip conditions directly
- Particles can still penetrate if coefficient too low
- Force becomes singular as r → 0
- Less accurate for viscous interactions
- Cannot model porous or moving boundaries easily
- Adds artificial stiffness (requires smaller timestep)

**Comparison Summary:**
- Ghost particles: Better accuracy, natural pressure distribution, higher cost
- Repulsive forces: Simpler, lower memory, less accurate, artificial forces

---

## 3. Dummy Particles

Dummy particles are static boundary particles that participate in the SPH computation with interpolated or fixed properties.

### 3.1 Definition and Differences from Ghost Particles

**Ghost Particles:**
- Created from fluid particles (dynamic)
- Properties mirror fluid neighbors
- Updated every timestep
- Enforce boundary conditions through mirroring

**Dummy Particles:**
- Pre-placed on boundary surface (static)
- Properties computed once or interpolated from nearby fluid
- Fixed positions throughout simulation
- Participate as normal SPH particles

### 3.2 Generation Algorithm

```pseudocode
function generateDummyParticles(boundary, h, targetDensity):
    dummyParticles = []

    // Surface discretization with spacing ~ h
    particleSpacing = h / 1.0  // Adjust for coverage

    for each point on boundary surface:
        dummyParticle = ParticleData()
        dummyParticle.position = point
        dummyParticle.mass = targetDensity * particleSpacing^3
        dummyParticle.density = targetDensity
        dummyParticle.pressure = 0.0  // Initially zero
        dummyParticle.velocity = boundaryVelocity  // Wall velocity
        dummyParticle.type = BOUNDARY_PARTICLE
        dummyParticle.isDummy = true
        dummyParticle.isFixed = true

        dummyParticles.append(dummyParticle)

    return dummyParticles
```

### 3.3 Property Assignment

```pseudocode
function initializeDummyParticleProperties(dummyParticles, fluidParticles, kernel, h):

    for each dummy in dummyParticles:
        neighbors = findNeighbors(dummy.position, fluidParticles, 2*h)

        if neighbors.empty():
            continue

        // Interpolate properties from neighbors
        weightedDensity = 0.0
        weightedPressure = 0.0
        totalWeight = 0.0

        for each neighbor in neighbors:
            weight = kernel(distance(dummy.position, neighbor.position), h)

            weightedDensity += weight * neighbor.density
            weightedPressure += weight * neighbor.pressure
            totalWeight += weight

        if totalWeight > 0:
            dummy.density = weightedDensity / totalWeight
            dummy.pressure = weightedPressure / totalWeight

        // Adjust mass for new density
        dummy.mass = dummy.density * particleVolume
```

### 3.4 Interaction Strategy

**Include in Density Calculation:**
```pseudocode
function computeDensity(particle, allParticles):
    rho = 0.0

    // Sum over fluid AND dummy particles
    for each neighbor in allParticles:
        if distance(particle, neighbor) < 2*h:
            rho += neighbor.mass * kernel(distance, h)

    return rho
```

**Include in Force Calculation:**
```pseudocode
function computePressureForce(particle, allParticles):
    force = Vector3(0, 0, 0)

    for each neighbor in allParticles:
        if neighbor.type == BOUNDARY_PARTICLE:
            // Special treatment for dummy particles
            // Often use reduced contribution or zeroed pressure from dummies

            factor = 1.0
            if neighbor.isDummy:
                factor = 0.5  // Reduce influence
        else:
            factor = 1.0

        dist = distance(particle.position, neighbor.position)
        if dist < 2*h:
            kernelGrad = kernelGradient(dist, h)
            force -= neighbor.mass * factor * (particle.pressure / particle.density^2
                     + neighbor.pressure / neighbor.density^2) * kernelGrad

    return force
```

---

## 4. Dynamic Boundary Conditions (DBC)

Dynamic Boundary Conditions, as implemented in DualSPHysics, treat boundary particles as full participants in SPH equations with computed densities and pressures.

### 4.1 Overview

DBC is more sophisticated than dummy particles:
- Boundary particles compute density from all neighbors
- Pressure is derived from equation of state (not pre-assigned)
- Particles move with boundary velocity (fixed or prescribed)
- Natural handling of pressure forces
- Superior accuracy at fluid-boundary interface

### 4.2 Boundary Particle Characteristics

```pseudocode
class BoundaryParticle:
    position: Vector3          // Fixed or moving with boundary
    velocity: Vector3          // Prescribed (wall velocity)
    mass: float               // Computed from density and volume

    density: float            // Computed from neighbors
    pressure: float           // From equation of state: P = (ρ/ρ₀)^γ - 1

    acceleration: Vector3     // Boundary acceleration (if moving)
    normal: Vector3           // Outward normal (for asymmetric kernels)

    type: BOUNDARY           // Particle classification
    isFixed: bool            // True: position fixed; False: prescribed motion
```

### 4.3 Density Computation

```pseudocode
function computeBoundaryDensity(boundaryParticle, allNeighbors, kernel, h):
    density = 0.0

    // Standard SPH summation (includes fluid and boundary neighbors)
    for each neighbor in allNeighbors:
        if distance(boundaryParticle, neighbor) < 2*h:
            dist = distance(boundaryParticle.position, neighbor.position)
            density += neighbor.mass * kernel(dist, h)

    // Alternative: Asymmetric kernel (one-sided)
    // Only sum over particles on fluid side
    // density += neighbor.mass * kernelAsymmetric(dist, h, normal)

    return max(density, ρ₀)  // Clamp to reference density minimum
```

### 4.4 Pressure Computation and Equation of State

```pseudocode
function computeBoundaryPressure(boundaryParticle, referenceDensity, c, gamma):
    // Tait equation of state (standard in SPH)
    // P = (c² * ρ₀ / γ) * [(ρ/ρ₀)^γ - 1]

    densityRatio = boundaryParticle.density / referenceDensity

    // Pressure component
    pressure = (c * c * referenceDensity / gamma) * (pow(densityRatio, gamma) - 1.0)

    // Option: Add static pressure term
    staticPressure = referenceDensity * 9.81 * height  // Hydrostatic

    return pressure + staticPressure
```

### 4.5 Complete DBC Algorithm

```pseudocode
class DualSPHysicsStyleDBC:

    function initializeBoundaryParticles(boundary, fluidDensity, h):
        particles = []
        spacing = h / 1.0

        // Generate particles on boundary surface
        for each point on boundary:
            p = BoundaryParticle()
            p.position = point
            p.velocity = boundaryVelocity(point)  // Prescribed
            p.type = BOUNDARY
            p.isFixed = not boundary.isMoving
            p.mass = fluidDensity * spacing^3
            p.density = fluidDensity
            p.pressure = 0.0

            particles.append(p)

        return particles

    function updateBoundary(fluidParticles, boundaryParticles, kernel, h, c, gamma, rho0):

        // Build neighbor list (fluid + boundary)
        allParticles = fluidParticles + boundaryParticles
        neighborLists = buildNeighborList(allParticles, 2*h)

        // Step 1: Compute density for boundary particles
        for each boundaryParticle in boundaryParticles:
            neighbors = neighborLists[boundaryParticle]
            boundaryParticle.density = computeBoundaryDensity(boundaryParticle, neighbors, kernel, h)

        // Step 2: Compute pressure from EOS
        for each boundaryParticle in boundaryParticles:
            boundaryParticle.pressure = computeBoundaryPressure(
                boundaryParticle, rho0, c, gamma)

        // Step 3: Boundary particles contribute to fluid density
        // (already included in neighbor summation)

        // Step 4: Pressure forces on boundary particles (for feedback)
        for each boundaryParticle in boundaryParticles:
            if not boundaryParticle.isFixed:
                neighbors = neighborLists[boundaryParticle]
                force = computePressureForce(boundaryParticle, neighbors)
                // Use force to compute reaction or motion

    function computeFluidForces(fluidParticle, allParticles, kernel, h):
        // Fluid particles compute pressure and viscous forces
        // including contributions from boundary particles

        neighbors = findNeighbors(fluidParticle, allParticles, 2*h)

        pressureForce = 0.0
        viscousForce = 0.0

        for each neighbor in neighbors:
            dist = distance(fluidParticle.position, neighbor.position)

            // Pressure force (same for fluid and boundary)
            pressureTerm = (fluidParticle.pressure / fluidParticle.density^2 +
                           neighbor.pressure / neighbor.density^2)

            kernelGrad = kernelGradient(dist, h)
            pressureForce += -neighbor.mass * pressureTerm * kernelGrad

            // Viscous force
            if neighbor.type == BOUNDARY:
                // Boundary velocity (no-slip)
                relativeVelocity = fluidParticle.velocity - neighbor.velocity
            else:
                // Fluid velocity
                relativeVelocity = fluidParticle.velocity - neighbor.velocity

            viscousForce += 2.0 * viscosity * fluidParticle.mass * neighbor.mass /
                           (fluidParticle.density * neighbor.density) *
                           dot(relativeVelocity, kernelGrad) * kernelGrad

        return pressureForce + viscousForce
```

### 4.6 Advantages of DBC

- **Accurate pressure distribution:** Boundary pressures computed from EOS
- **Natural no-slip enforcement:** Via velocity difference in SPH summation
- **Better stability:** Pressure gradients smoother
- **Feedback forces:** Can compute reaction forces from boundary
- **Correct physics:** Respects conservation laws
- **Handles moving boundaries:** Naturally integrates with prescribed motion

---

## 5. Corner Handling

Corners present a fundamental challenge in SPH boundary methods: the outward normal is undefined at sharp corners, making ghost particle reflection ambiguous.

### 5.1 The Corner Problem

**Issue Description:**
```
At a sharp corner, multiple valid normal directions exist:

    Fluid region:  |
                   |
                   +------ ← Which normal?

            Normal options: (0,1), (1,0), or (1,1)/√2 blend
```

**Consequences:**
- Inconsistent ghost particle placement
- Pressure spikes at corners
- Particle clustering or gaps near corners
- Spurious velocities
- Poor energy conservation

### 5.2 Solution 1: Rounded Corners

The simplest solution: smooth sharp corners with circular/elliptical arcs.

```pseudocode
function roundCorners(boundary, cornerRadius):
    // Replace sharp corners with circular arcs
    // Radius typically: cornerRadius = 1.5 * h to 2.0 * h

    newBoundary = []

    for i in 0 to boundary.vertices.length:
        v_prev = boundary.vertices[i-1]
        v_curr = boundary.vertices[i]
        v_next = boundary.vertices[i+1]

        // Direction vectors
        dir1 = normalize(v_curr - v_prev)
        dir2 = normalize(v_next - v_curr)

        // Angle at corner
        angle = acos(dot(dir1, dir2))

        if angle < π - 0.01:  // Sharp corner
            // Blend radius: smaller for very sharp angles
            radius = cornerRadius * sin(angle/2)

            // Find points before/after corner for arc
            p1 = v_curr - dir1 * radius
            p2 = v_curr + dir2 * radius

            // Create circular arc from p1 to p2
            arcPoints = createCircularArc(p1, p2, radius, numSubdivisions=10)
            newBoundary.extend(arcPoints)
        else:
            newBoundary.append(v_curr)

    return newBoundary
```

**Pros:**
- Simple to implement
- Smooth pressure field
- No special corner logic needed

**Cons:**
- Changes actual geometry
- May not be appropriate for all applications
- Requires tuning corner radius

### 5.3 Solution 2: Special Corner Treatment

Treat corners explicitly with special logic.

```pseudocode
function generateGhostParticleAtCorner(fluidParticle, corner, boundary1, boundary2, h):
    // Two adjacent boundary faces meet at corner

    normal1 = outwardNormal(boundary1)
    normal2 = outwardNormal(boundary2)

    // Multiple options for corner normal:

    // Option A: Bisector approach (most common)
    cornerNormal = normalize(normal1 + normal2)

    // Option B: Weighted by local curvature
    curvature1 = computeCurvature(boundary1)
    curvature2 = computeCurvature(boundary2)
    weight1 = 1.0 / (1.0 + curvature1)
    weight2 = 1.0 / (1.0 + curvature2)
    cornerNormal = normalize(weight1 * normal1 + weight2 * normal2)

    // Option C: Min-angle approach (smoother gradient)
    angle = acos(dot(normal1, normal2))
    if angle > π/2:  // Obtuse corner
        cornerNormal = normal1  // Use one of the normals
    else:
        cornerNormal = normalize(normal1 + normal2)

    // Mirror particle along blended normal
    distToBoundary = distance(fluidParticle.position, corner)
    ghostPosition = fluidParticle.position - cornerNormal * (2.0 * distToBoundary)

    ghost = createGhostParticle()
    ghost.position = ghostPosition
    ghost.velocity = mirrorVelocity(fluidParticle.velocity, cornerNormal)
    ghost.density = fluidParticle.density
    ghost.pressure = fluidParticle.pressure

    return ghost
```

### 5.4 Solution 3: Multi-Normal Approach

Use contributions from multiple boundary normals to smooth out corner effects.

```pseudocode
function generateMultiNormalGhost(fluidParticle, boundaries, h):
    // Particle influenced by multiple nearby boundary segments

    ghost = createGhostParticle()
    ghost.position = fluidParticle.position  // Start at fluid position
    ghost.velocity = fluidParticle.velocity

    nearbyNormals = []
    nearbyDistances = []

    // Find all boundary segments within 2h
    for each boundarySegment in boundaries:
        dist = distance(fluidParticle, boundarySegment)

        if dist < 2*h:
            normal = outwardNormal(boundarySegment)
            nearbyNormals.append(normal)
            nearbyDistances.append(dist)

    if nearbyNormals.empty():
        return null  // Not near boundary

    // Compute weighted average normal
    totalWeight = 0.0
    blendedNormal = Vector3(0, 0, 0)

    for i in 0 to nearbyNormals.length:
        // Weight by inverse distance
        weight = 1.0 / (nearbyDistances[i] + ε)
        blendedNormal += weight * nearbyNormals[i]
        totalWeight += weight

    blendedNormal = normalize(blendedNormal / totalWeight)

    // Create ghost with blended normal
    distToBoundary = nearbyDistances[0]  // Closest boundary
    ghost.position = fluidParticle.position - blendedNormal * (2.0 * distToBoundary)
    ghost.velocity = mirrorVelocity(fluidParticle.velocity, blendedNormal)

    return ghost
```

### 5.5 Comparison and Recommendations

| Approach | Implementation | Accuracy | Cost | Best For |
|----------|---------------|----------|------|----------|
| Rounded Corners | Very Simple | Good | Low | Simple geometries, aesthetic applications |
| Bisector Normal | Simple | Fair-Good | Very Low | Engineering, quick prototyping |
| Multi-Normal Blend | Moderate | Very Good | Low | Complex corners, high accuracy needed |
| Explicit Treatment | Complex | Excellent | Medium | Production simulations |

**Recommendation:** Start with rounded corners or bisector approach. For higher accuracy, use multi-normal blending. Reserve explicit corner tables for very complex geometries.

---

## 6. Curved Boundaries

Curved boundaries (cylinders, spheres, arbitrary surfaces) require careful ghost particle placement to maintain accuracy.

### 6.1 Mirror Particle Placement on Curved Surfaces

**Basic Algorithm:**
```pseudocode
function mirrorParticleOnCurvedSurface(fluidParticle, curvedBoundary, h):
    // Step 1: Find closest point on curved surface
    closestPoint = projectParticleOntoCurvedSurface(fluidParticle.position, curvedBoundary)
    distanceToBoundary = distance(fluidParticle.position, closestPoint)

    // Step 2: Compute outward normal at closest point
    normal = computeOutwardNormal(closestPoint, curvedBoundary)

    // Step 3: Mirror across surface
    ghostPosition = closestPoint - normal * distanceToBoundary

    // Step 4: Additional correction for curvature
    curvature = computeLocalCurvature(closestPoint, curvedBoundary)
    if curvature != 0:
        // Adjust mirror position for non-zero curvature
        curveCorrection = curvature * distanceToBoundary^2 / 2.0
        ghostPosition -= normal * curveCorrection

    ghost = ParticleData()
    ghost.position = ghostPosition
    ghost.velocity = mirrorVelocity(fluidParticle.velocity, normal)
    ghost.density = fluidParticle.density
    ghost.pressure = fluidParticle.pressure

    return ghost
```

### 6.2 Spherical Boundaries

For spherical boundaries (common in particle confinement):

```pseudocode
function mirrorOnSphere(fluidParticle, sphereCenter, sphereRadius, h):
    // Vector from sphere center to particle
    radialVector = fluidParticle.position - sphereCenter
    distance = magnitude(radialVector)
    normal = radialVector / distance  // Outward normal

    if distance >= sphereRadius:
        return null  // Outside sphere

    // Mirror along radial direction
    distanceToBoundary = sphereRadius - distance
    ghostPosition = sphereCenter + normal * (2.0 * sphereRadius - distance)

    // Curvature correction (for spheres, highly important)
    // Gaussian curvature = 1/R²
    curvature = 1.0 / sphereRadius
    curveCorrection = 0.5 * curvature * distanceToBoundary^2
    ghostPosition -= normal * curveCorrection

    ghost = ParticleData()
    ghost.position = ghostPosition
    ghost.velocity = mirrorVelocity(fluidParticle.velocity, normal)
    ghost.density = fluidParticle.density
    ghost.pressure = fluidParticle.pressure

    return ghost
```

### 6.3 Cylindrical Boundaries

```pseudocode
function mirrorOnCylinder(fluidParticle, cylinderAxis, cylinderCenter, radius, h):
    // Project particle onto cylinder axis
    axisVector = cylinderAxis / magnitude(cylinderAxis)
    toParticle = fluidParticle.position - cylinderCenter
    projectionLength = dot(toParticle, axisVector)

    // Closest point on cylinder axis
    closestAxisPoint = cylinderCenter + projectionLength * axisVector

    // Radial vector (perpendicular to axis)
    radialVector = fluidParticle.position - closestAxisPoint
    radialDistance = magnitude(radialVector)
    normalRadial = radialVector / radialDistance

    if radialDistance >= radius:
        return null

    // Mirror along radial direction
    distanceToBoundary = radius - radialDistance
    ghostPosition = closestAxisPoint + normalRadial * (2.0 * radius - radialDistance)

    // Curvature correction (cylindrical curvature = 1/R)
    curvature = 1.0 / radius
    curveCorrection = 0.5 * curvature * distanceToBoundary^2
    ghostPosition -= normalRadial * curveCorrection

    ghost = ParticleData()
    ghost.position = ghostPosition
    ghost.velocity = mirrorVelocity(fluidParticle.velocity, normalRadial)
    ghost.density = fluidParticle.density
    ghost.pressure = fluidParticle.pressure

    return ghost
```

### 6.4 Curvature Corrections

**Why Curvature Matters:**

For surfaces with mean curvature κ, particles closer to the surface experience kernel gradients that differ from flat surface expectations. This is especially critical for:
- Strong confinement (small radius)
- High spatial resolution (small h)
- Dynamic simulations with pressure gradients

**Curvature-Aware Ghost Position:**
```pseudocode
function correctGhostPositionForCurvature(ghostPosition, closestSurfacePoint,
                                          normal, curvature, h):
    // Taylor expansion correction
    // Ghost position = base reflection + curvature correction

    distanceFromSurface = distance(ghostPosition, closestSurfacePoint)

    // First-order curvature correction
    correction = -0.5 * curvature * distanceFromSurface^2 * normal

    // Second-order correction (optional, for very curved surfaces)
    if abs(curvature) > 1.0 / (2.0 * h):  // Highly curved
        // Add principal curvature effects
        principal_curvature1 = curvature
        principal_curvature2 = 0.0  // Orthogonal curvature

        correction += (principal_curvature1 + principal_curvature2) / 8.0 *
                     distanceFromSurface^3 * normal

    return ghostPosition + correction
```

### 6.5 Implementation Tips

**Efficiency Considerations:**
```pseudocode
class CurvedBoundaryManager:

    function precomputeSurfaceData(curvedBoundary):
        // Build spatial index for quick closest-point queries
        this.spatialIndex = buildKDTree(curvedBoundary.points, 3)

        // Precompute normals and curvatures at key points
        for each point on boundary:
            point.normal = computeOutwardNormal(point)
            point.curvature = computePrincipalCurvatures(point)

    function generateGhostsEfficient(fluidParticles, h):
        ghosts = []

        for each particle p in fluidParticles:
            // Quick lookup in spatial index
            nearestPoint = this.spatialIndex.findNearest(p.position)
            dist = distance(p.position, nearestPoint)

            if dist < h:  // Potentially needs ghost
                ghost = this.mirrorWithCurvatureCorrection(p, nearestPoint, h)
                ghosts.append(ghost)

        return ghosts
```

---

## 7. Inlet/Outlet Boundaries

Inlet/outlet boundaries are non-reflective conditions for open domain simulations. They allow fluid to enter and exit without acoustic reflections.

### 7.1 Buffer Zone Approach

Buffer zones dampen waves before they reach open boundaries to prevent reflections.

```pseudocode
function createBufferZones(domain, h):
    // Create damping regions near inlet/outlet

    bufferWidth = 4.0 * h  // Thickness of damping zone

    inletBuffer = Region()
    inletBuffer.position = domain.inlet
    inletBuffer.width = bufferWidth

    outletBuffer = Region()
    outletBuffer.position = domain.outlet
    outletBuffer.width = bufferWidth

    return [inletBuffer, outletBuffer]

function applyBufferDamping(particles, buffers, c, h):
    for each particle p in particles:
        for each buffer in buffers:
            if particle.position in buffer:
                // Distance from buffer edge (0 at boundary, bufferWidth at buffer back)
                distanceFromBoundary = computeDistanceInBuffer(p.position, buffer)

                // Exponential damping profile
                dampingFactor = exp(-(4.0 * (bufferWidth - distanceFromBoundary)^2) / bufferWidth^2)

                // Apply damping to particle velocity
                p.velocity *= (1.0 - 0.5 * dampingFactor)

                // Alternatively: reduce mass/momentum
                // p.velocity *= (1.0 - 0.3 * dampingFactor)

function inletBufferTreatment(particles, inletBuffer, targetVelocity, h):
    for each particle p in particles:
        if p.position in inletBuffer:
            distanceFromBoundary = computeDistanceInBuffer(p.position, inletBuffer)

            // Blend: inlet target velocity → internal velocity
            blendFactor = distanceFromBoundary / inletBuffer.width

            // Ramped velocity prescription
            p.velocity = (1.0 - blendFactor) * targetVelocity + blendFactor * p.velocity
```

### 7.2 Open Boundary Conditions

Open boundaries minimize reflections by reducing acoustic coupling.

```pseudocode
function applyOpenBoundaryCondition(particles, openBoundary, c, h):
    // Non-reflecting boundary condition
    // Based on characteristic variables (Riemann invariants)

    for each particle p near openBoundary:
        // Distance to boundary
        distance = distance(p.position, openBoundary)

        if distance < 2*h:
            // Outward normal
            normal = outwardNormal(p.position, openBoundary)

            // Velocity component normal to boundary
            velocityNormal = dot(p.velocity, normal)

            // Sound speed
            c_local = c

            // Non-reflecting condition: allow outflow freely
            // Prevent inflow with characteristics-based adjustment

            if velocityNormal < 0:  // Trying to flow inward
                // Modify velocity to prevent inflow
                // Characteristic variable: u - 2c/(gamma-1)
                p.velocity -= velocityNormal * normal

            // Alternative: Free outflow (completely non-reflecting)
            // Just leave velocity as is

function periodicBoundaryCondition(particles, domainSize):
    // Particle exiting one side enters opposite side

    for each particle p in particles:
        for dimension in [X, Y, Z]:
            if p.position[dimension] < 0:
                p.position[dimension] += domainSize[dimension]

            if p.position[dimension] > domainSize[dimension]:
                p.position[dimension] -= domainSize[dimension]
```

### 7.3 Wave Beach (Sponge Layer)

For long-wave simulations, a wave beach with progressive damping prevents reflections.

```pseudocode
class WaveBeach:

    function initialize(startPosition, endPosition, beachWidth, c, h):
        this.startPos = startPosition
        this.endPos = endPosition
        this.width = beachWidth
        this.c = c
        this.h = h

    function computeDampingCoefficient(position):
        // Distance into beach (0 at start, 1 at end)
        parameter = (distance(position, this.startPos) - 0) / this.width
        parameter = clamp(parameter, 0, 1)

        // Polynomial damping profile (quartic for smoothness)
        dampingCoeff = 0.5 * (1.0 - cos(π * parameter))

        return dampingCoeff

    function applyBeachDamping(particles, beachWidth, c, dt):
        for each particle p in particles:
            if this.isInBeach(p.position):
                dampingCoeff = this.computeDampingCoefficient(p.position)

                // Stokes-type damping with relaxation time
                relaxationTime = 1.0 / (dampingCoeff * c / this.h)
                decayFactor = exp(-dt / relaxationTime)

                p.velocity *= decayFactor
                p.acceleration *= decayFactor
```

### 7.4 Inlet Generation

```pseudocode
function generateInletParticles(inletBoundary, flowVelocity, density, spacing):
    particles = []

    // Discretize inlet surface
    for each point on inletBoundary:
        p = ParticleData()
        p.position = point
        p.velocity = flowVelocity
        p.density = density
        p.mass = density * spacing^3
        p.type = FLUID
        p.isInlet = true

        particles.append(p)

    // Particles generated at inlet each timestep
    return particles
```

---

## 8. Free Surface Detection and Treatment

Free surfaces are interfaces between fluid and air where special care is needed.

### 8.1 Neighborhood Deficit Method

The simplest detection: count particles in neighborhood and compare to interior reference.

```pseudocode
function detectFreeSurfaceNeighborhoodDeficit(particle, neighbors, h, refCount):
    // Reference particle count in interior (fully surrounded)
    // refCount ≈ constant, ~50-100 for typical kernels and spacing

    neighborCount = len(neighbors)

    // Deficit ratio
    deficitRatio = neighborCount / refCount

    // Threshold-based detection
    if deficitRatio < 0.65:  // Less than 65% of reference
        return true  // Particle is at free surface
    else:
        return false
```

**Implementation:**
```pseudocode
function markFreeSurfaceParticles(fluidParticles, h):
    // Compute reference neighbor count from interior particles
    refCount = 0
    numInteriorSamples = 0

    for each particle p in fluidParticles:
        neighbors = findNeighbors(p, fluidParticles, 2*h)

        // Is this particle far from any boundary?
        if p.distanceToBoundary > 3*h:
            refCount += len(neighbors)
            numInteriorSamples += 1

    refCount = refCount / numInteriorSamples  // Average

    // Now mark free surface particles
    for each particle p in fluidParticles:
        neighbors = findNeighbors(p, fluidParticles, 2*h)
        neighborCount = len(neighbors)

        if neighborCount < 0.65 * refCount:
            p.isFreeSurface = true
        else:
            p.isFreeSurface = false
```

### 8.2 Divergence-Based Detection

Free surfaces have low divergence (particles not strongly convergent).

```pseudocode
function detectFreeSurfaceDivergence(particle, neighbors, kernel, h):
    // Divergence of velocity field
    // div(v) = ∑_j (v_j - v_i) · ∇W_ij

    divergence = 0.0

    for each neighbor j:
        if distance(particle, neighbor) < 2*h:
            velocityDifference = neighbor.velocity - particle.velocity
            kernelGrad = kernelGradient(particle.position - neighbor.position, h)

            divergence += (particle.mass / neighbor.density) *
                         dot(velocityDifference, kernelGrad)

    // At free surface: low divergence (fluid moving away)
    // At interior: high negative divergence (fluid compressing)

    // Threshold depends on flow speed
    divThreshold = -0.1 * soundSpeed / h

    if divergence > divThreshold:
        return true  // Free surface
    else:
        return false
```

### 8.3 Curvature-Based Detection

Curvature detection uses surface normals and their variation.

```pseudocode
function detectFreeSurfaceFromCurvature(particle, neighbors, kernel, h):
    // Compute surface normal estimate
    // Uses particle positions and densities

    colorField = 0.0
    colorGradient = Vector3(0, 0, 0)
    colorLaplacian = 0.0

    // CSF (Color-Signed Function) method
    for each neighbor j:
        if distance(particle, neighbor) < 2*h:
            dist = distance(particle.position, neighbor.position)
            kernelValue = kernel(dist, h)
            kernelGrad = kernelGradient(dist, h)

            // Color field (normalized density)
            colorField += (neighbor.mass / neighbor.density) * kernelValue

            // Gradient (surface normal direction)
            colorGradient += (neighbor.mass / neighbor.density) *
                            (neighbor.density - particle.density) * kernelGrad

            // Laplacian (curvature indicator)
            colorLaplacian += (neighbor.mass / neighbor.density) *
                             (neighbor.density - particle.density) *
                             kernelLaplacian(dist, h)

    // Curvature: proportional to Laplacian of color field
    curvatureMagnitude = abs(colorLaplacian)

    // Free surface has high curvature gradient
    // Interior has low (smooth field)

    if curvatureMagnitude > 0.01 * soundSpeed / h:
        return true
    else:
        return false
```

### 8.4 Free Surface Treatment

Once detected, special treatment is applied:

```pseudocode
function treatFreeSurfaceParticle(particle, kernel, h, c, gamma, rho0):
    if not particle.isFreeSurface:
        return

    // Treatment 1: Zero pressure at free surface
    particle.pressure = 0.0
    // This enforces stress-free boundary condition

    // Treatment 2: Special density smoothing
    neighbors = findNeighbors(particle, allParticles, 2*h)
    smoothedDensity = 0.0

    for each neighbor in neighbors:
        dist = distance(particle, neighbor)
        weight = kernel(dist, h) / kernel(0, h)
        smoothedDensity += weight * neighbor.density

    particle.density = max(smoothedDensity / len(neighbors), rho0)

    // Treatment 3: Reduced acceleration (particles at surface less constrained)
    particle.surfaceAccelerationFactor = 0.9  // Slightly relaxed

    // Treatment 4: Surface tension (if included)
    if simulation.includeSurfaceTension:
        surfaceTensionForce = computeSurfaceTensionForce(particle, neighbors)
        particle.acceleration += surfaceTensionForce / particle.mass

function computeSurfaceTensionForce(particle, neighbors, sigma, h):
    // Surface tension model (Brackbill et al.)

    // Curvature from color field
    colorField = 0.0
    colorGradient = Vector3(0, 0, 0)
    colorLaplacian = 0.0

    for each neighbor in neighbors:
        if distance(particle, neighbor) < 2*h:
            dist = distance(particle.position, neighbor.position)
            kGrad = kernelGradient(dist, h)
            kLaplacian = kernelLaplacian(dist, h)

            colorField += (neighbor.mass / neighbor.density) * kernel(dist, h)
            colorGradient += (neighbor.mass / neighbor.density) * kGrad
            colorLaplacian += (neighbor.mass / neighbor.density) * kLaplacian

    if magnitude(colorGradient) < 1e-10:
        return Vector3(0, 0, 0)

    // Surface normal
    normalVector = colorGradient / magnitude(colorGradient)

    // Curvature
    curvature = -colorLaplacian / magnitude(colorGradient)

    // Surface tension force
    surfaceTensionForce = sigma * curvature * normalVector

    return surfaceTensionForce
```

---

## 9. Implementation Recommendations

### 9.1 Which Method to Start With

**Recommended progression:**

1. **Prototype Phase: Ghost Particles with 2 Layers**
   - Fastest to implement
   - Reliable for basic applications
   - Easy to debug
   - Sufficient for most engineering problems

2. **If Problems Occur: Add 3rd Layer or Rounded Corners**
   - Pressure spikes → add 3rd layer
   - Corner issues → round corners or use bisector normals
   - Particle leakage → increase h or damping

3. **Production Quality: Dynamic Boundary Conditions (DBC)**
   - Superior accuracy
   - Better pressure continuity
   - Worth effort for publication/critical applications
   - Use after validating with ghost particles

4. **Advanced: Hybrid Approaches**
   - DBC for walls, ghost particles for free surface
   - Dummy particles for moving boundaries
   - Repulsive forces for secondary confinement

### 9.2 Common Bugs and Solutions

**Problem 1: Particle Leakage Through Walls**

```pseudocode
// Symptom: Particles escape domain despite boundary conditions

// Solutions (in order of effectiveness):
1. // Increase ghost particle layers
   numLayers = 3 or 4  // instead of 2

2. // Reduce time step
   dt = 0.5 * dt_previous

3. // Increase repulsive force coefficient (if using repulsive forces)
   alpha = 2.0 * alpha_previous

4. // Smooth particle positions at boundary
   if particle.isNearBoundary:
       particle.position = projectToBoundary(particle.position)

5. // Add explicit penetration check
   function checkAndRepelParticles(particles, boundaries):
       for each particle in particles:
           if particle.position outside boundaries:
               // Find boundary and push back
               closestBoundary = findClosestBoundary(particle)
               penetrationDepth = computePenetrationDepth(particle, closestBoundary)
               normal = outwardNormal(particle, closestBoundary)
               particle.position += normal * (penetrationDepth + 0.01*h)
```

**Problem 2: Pressure Spikes at Walls**

```pseudocode
// Symptom: Spurious pressure oscillations near boundaries

// Solutions:
1. // Use more ghost particle layers
   numLayers = 3 or 4

2. // Apply pressure smoothing at boundaries
   for each particle p near boundary:
       neighbors = findNeighbors(p, particles, 2*h)
       smoothedPressure = 0.0
       for each neighbor:
           if neighbor.isBoundary or neighbor.isNearBoundary:
               smoothedPressure += neighbor.pressure
       p.pressure = 0.7 * p.pressure + 0.3 * smoothedPressure

3. // Increase kernel support radius locally
   h_local = 1.2 * h  // near boundaries only

4. // Use asymmetric kernels
   // Only sum contributions from fluid side

5. // Switch to Dynamic Boundary Conditions (DBC)
```

**Problem 3: High Computational Cost**

```pseudocode
// Symptom: Simulation very slow due to boundary handling

// Solutions:
1. // Reduce number of ghost layers
   numLayers = 2  // minimum, but watch pressure

2. // Use spatial partitioning for ghost generation
   spatialGrid = buildUniformGrid(domain, 2*h)

   // Only check particles in neighboring grid cells
   for each boundary segment:
       relevantCells = spatialGrid.getCellsNear(segment)
       fluidParticlesToCheck = spatialGrid.getParticlesInCells(relevantCells)

3. // Switch to dummy particles (static, no regeneration)
   // Cost: one-time generation, then very fast

4. // Use repulsive forces instead of ghost particles
   // No extra particles, just add force computation

5. // Parallelize ghost particle generation
   // Each boundary segment independent
```

**Problem 4: Oscillatory Flow at Boundaries**

```pseudocode
// Symptom: Unphysical oscillations near walls

// Solutions:
1. // Increase viscosity (artificial)
   nu_effective = nu + 0.1 * c * h

2. // Reduce pressure gradient amplification
   // Use lower-order kernel (e.g., quadratic instead of quintic)

3. // Apply velocity smoothing at boundaries
   for each particle near boundary:
       smoothedVelocity = averageVelocityOfNeighbors(particle)
       particle.velocity = 0.8 * particle.velocity + 0.2 * smoothedVelocity

4. // Increase coupling between fluid and boundary particles
   // Reduce ghost particle spacing, use 4+ layers
```

### 9.3 Comparison Table

| Method | Accuracy | Memory | Speed | No-Slip | Moving Boundaries | Learning Curve | Best For |
|--------|----------|--------|-------|---------|-------------------|-----------------|----------|
| **Ghost Particles (2L)** | Good | +30% | Fast | Excellent | Good | Very Easy | Quick prototyping, simple geometries |
| **Ghost Particles (3L)** | Very Good | +50% | Good | Excellent | Good | Very Easy | Standard engineering problems |
| **Repulsive Force** | Fair | No extra | Fast | Fair | Good | Easy | Real-time, complex geometries |
| **Dummy Particles** | Fair | +10% | Very Fast | Fair | Poor | Easy | Static boundaries, low accuracy needs |
| **Dynamic BC (DBC)** | Excellent | +30% | Good | Excellent | Excellent | Hard | Production, high accuracy, moving boundaries |
| **Hybrid (Ghost + DBC)** | Excellent | +40% | Good | Excellent | Excellent | Hard | Complex simulations with mixed boundaries |

---

## 10. Advanced Topics

### 10.1 Energy Conservation at Boundaries

```pseudocode
function correctEnergyDissipation(particle, wallParticle, dt):
    // Boundary interaction should conserve energy
    // (in elastic collisions with rigid walls)

    // Kinetic energy before interaction
    KE_before = 0.5 * particle.mass * magnitude(particle.velocity)^2

    // Velocity component normal to wall
    normal = outwardNormal(wallParticle.position, boundary)
    v_normal = dot(particle.velocity, normal)
    v_tangent = particle.velocity - v_normal * normal

    // Elastic reflection (tangential preserved, normal reversed)
    v_normal_new = -v_normal * (1.0 - damping)  // damping ∈ [0,1]
    v_tangent_new = v_tangent * (1.0 - friction)

    particle.velocity_new = v_normal_new * normal + v_tangent_new

    // Check energy conservation
    KE_after = 0.5 * particle.mass * magnitude(particle.velocity_new)^2
    energyLoss = (KE_before - KE_after) / KE_before

    // Log warnings for large energy loss
    if energyLoss > 0.1 and not particle.isColliding:
        log("Warning: High energy loss at boundary: {}", energyLoss)
```

### 10.2 Anisotropic Kernels for Boundaries

Standard isotropic kernels may not be ideal near boundaries. Anisotropic kernels provide better pressure accuracy.

```pseudocode
function anisotropicKernelNearBoundary(dist, h, normal, isNearBoundary):
    if not isNearBoundary:
        return standardKernel(dist, h)

    // One-sided kernel: enhanced in normal direction
    // Reduces spurious pressure oscillations

    // Decompose distance into normal and tangential components
    dist_normal = abs(dot(normalize(dist), normal))
    dist_tangent = sqrt(magnitude(dist)^2 - dist_normal^2)

    // Different smoothing length in each direction
    h_normal = 1.2 * h   // Wider in normal direction (toward fluid)
    h_tangent = 0.8 * h  // Narrower in tangent direction

    // Compute anisotropic kernel value
    kernel_value = gaussianKernel(dist_normal / h_normal) *
                  gaussianKernel(dist_tangent / h_tangent)

    return kernel_value
```

### 10.3 Implicit Integration at Boundaries

For stiff wall interactions, implicit time integration provides better stability.

```pseudocode
function implicitBoundaryUpdate(particles, wallParticles, dt, nu):
    // Solve: (I + dt²ν∇²) u_new = u_old + dt * f_external

    for each particle near boundary:
        // Build implicit system for this particle
        // Involves particle and nearby wall particles

        // Simplified: use semi-implicit approach
        u_fluid = particle.velocity_old
        u_wall = wallParticle.velocity

        // Relative velocity at boundary
        u_rel = u_fluid - u_wall

        // Implicit viscous coupling
        lambda = dt * nu / (spacing^2)
        u_new = u_old / (1.0 + lambda) + lambda * u_wall / (1.0 + lambda)

        particle.velocity = u_new
```

---

## 11. Summary and Best Practices

### Quick Start Checklist

- [ ] Start with **ghost particles (2 layers)**
- [ ] Use **rounded corners** or **bisector normals** for corner treatment
- [ ] Set ghost particle spacing equal to fluid particle spacing
- [ ] Set ghost layer thickness = h (2 layers within 2h of boundary)
- [ ] Verify: no pressure spikes in first test
- [ ] Verify: no particle leakage after 1000 timesteps
- [ ] If issues: add 3rd layer and retest

### Parameter Guidelines

```pseudocode
// Standard configuration
h = domain_size / (50 to 100)        // Kernel support
nu = 0.001 * c * h                   // Kinematic viscosity
dt = 0.25 * h / c                    // CFL-based timestep
numGhostLayers = 2 or 3              // Ghost particles

// Curved boundaries
cornerRadius = 1.5 * h               // For corner rounding
curvatureCorrection = true           // Enable for R < 10*h

// Free surface
deficitThreshold = 0.65              // Neighborhood deficit
freeSurfaceZone = 2*h                // Treatment region
```

### Testing Recommendations

Before using boundary method in production:

1. **Hydrostatic Test:** Still fluid in closed domain, pressure = ρgh
2. **Poiseuille Flow:** Channel flow, compare to analytical profile
3. **Impact Test:** Particle hits wall, check energy loss (should be <5% for elastic)
4. **Corner Test:** Particle near corner, check stability
5. **Free Surface Test:** Dam break, observe interface quality

---

## References and Further Reading

Key aspects of each boundary method are well-documented in the SPH literature. For implementation details beyond this reference, consult:

- Standard SPH textbooks and review articles
- DualSPHysics documentation for Dynamic Boundary Conditions
- Journal articles on specific methods of interest
- Published open-source SPH codes for reference implementations

---

**End of Boundary Methods Reference**

*Last Updated: 2026*
*Target Audience: SPH Simulation Developers*
