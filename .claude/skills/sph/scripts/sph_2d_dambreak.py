"""
2D Dam Break Simulation using Weakly Compressible SPH (WCSPH)

This is a complete, working SPH simulation demonstrating:
- Tait equation of state for pressure
- Cubic spline kernel
- Monaghan artificial viscosity
- XSPH velocity correction
- Leapfrog time integration
- Adaptive time stepping (CFL condition)
- Cell-linked list neighbor search
- Boundary handling with static wall particles

Simulation setup:
- Water column: 1m wide × 2m tall
- Tank: 4m wide × 3m tall with 3-layer wall particles
- Initial particle spacing: configurable (default 0.02m)

Run with:
    python sph_2d_dambreak.py --dx 0.02 --t_end 2.0 --output_dir ./output
"""

import numpy as np
import argparse
import os
from typing import Tuple, List
import sys

# Import SPH components
from kernels import get_kernel
from neighbor_search import CellLinkedList


# ============================================================================
# SPH Simulation Parameters
# ============================================================================

class SPHParams:
    """SPH simulation parameters"""

    def __init__(self, dx: float = 0.02):
        self.dx = dx  # Particle spacing
        self.h = 1.3 * dx  # Smoothing length
        self.g = 9.81  # Gravity (m/s²)
        self.rho0 = 1000.0  # Reference density (kg/m³)
        self.gamma = 7.0  # Tait EOS parameter
        self.H = 2.0  # Reference height for c0 calculation (m)
        self.c0 = 10.0 * np.sqrt(self.g * self.H)  # Speed of sound
        self.B = self.rho0 * self.c0**2 / self.gamma  # Tait EOS constant
        self.alpha = 0.02  # Artificial viscosity coefficient
        self.eps_xsph = 0.5  # XSPH correction coefficient
        self.CFL = 0.2  # CFL coefficient for time stepping
        self.min_dt = 1e-4  # Minimum time step
        self.max_dt = 0.01  # Maximum time step


# ============================================================================
# Particle System Initialization
# ============================================================================

def create_dam_break_particles(
    dx: float, h: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create initial particle configuration for dam break.

    Creates:
    1. Water particles: 1m wide × 2m tall column at origin
    2. Boundary particles: 3 layers of wall particles (bottom and side walls)

    Args:
        dx: Particle spacing
        h: Smoothing length (not used for creation, for reference)

    Returns:
        positions: (n_particles, 2) array of positions
        velocities: (n_particles, 2) array of velocities
        particle_types: (n_particles,) array of type (0=fluid, 1=boundary)
    """
    particles = []
    types = []

    # ========== Water particles ==========
    # Create water column: 1m wide × 2m tall
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 2.0

    x_water = np.arange(x_min + dx/2, x_max, dx)
    y_water = np.arange(y_min + dx/2, y_max, dx)

    for x in x_water:
        for y in y_water:
            particles.append([x, y])
            types.append(0)  # Fluid particle

    # ========== Boundary particles ==========
    # Bottom wall: 4m long, 3 layers
    x_wall_bottom = np.arange(0.0, 4.0, dx)
    for layer in range(3):
        y_layer = -(layer + 1) * dx
        for x in x_wall_bottom:
            particles.append([x, y_layer])
            types.append(1)  # Boundary particle

    # Left wall: 3m tall, 3 layers (excluding bottom corner)
    y_wall_left = np.arange(0.0, 3.0, dx)
    for layer in range(3):
        x_layer = -(layer + 1) * dx
        for y in y_wall_left:
            particles.append([x_layer, y])
            types.append(1)

    # Right wall: 3m tall, 3 layers (excluding bottom corner)
    for layer in range(3):
        x_layer = 4.0 + layer * dx
        for y in y_wall_left:
            particles.append([x_layer, y])
            types.append(1)

    # Convert to arrays
    positions = np.array(particles, dtype=float)
    velocities = np.zeros((len(particles), 2), dtype=float)
    particle_types = np.array(types, dtype=int)

    return positions, velocities, particle_types


# ============================================================================
# SPH Force Calculations
# ============================================================================

def compute_density(
    positions: np.ndarray,
    velocities: np.ndarray,
    particle_types: np.ndarray,
    mass: float,
    kernel_fn,
    cll: CellLinkedList,
    h: float,
    support_radius: float,
) -> np.ndarray:
    """
    Compute density for all particles using SPH.

    ρ_i = Σ_j m_j W(r_ij, h)

    Args:
        positions: Particle positions (n, 2)
        velocities: Particle velocities (n, 2)
        particle_types: Particle types (n,)
        mass: Particle mass
        kernel_fn: Kernel function
        cll: Cell-linked list
        h: Smoothing length
        support_radius: Kernel support radius multiplier

    Returns:
        Density array (n,)
    """
    n_particles = len(positions)
    densities = np.zeros(n_particles, dtype=float)

    search_radius = support_radius * h

    for i in range(n_particles):
        # Find neighbors
        neighbors = cll.find_neighbors(i, positions, search_radius)

        # Self-contribution
        densities[i] = mass * kernel_fn(0.0, h)

        # Neighbor contributions
        for j in neighbors:
            r_ij = np.linalg.norm(positions[j] - positions[i])
            densities[i] += mass * kernel_fn(r_ij, h)

    return densities


def compute_pressure(densities: np.ndarray, params: SPHParams) -> np.ndarray:
    """
    Compute pressure using Tait equation of state.

    P = B * [(ρ/ρ0)^γ - 1]

    Args:
        densities: Particle densities
        params: SPH parameters

    Returns:
        Pressure array
    """
    return params.B * ((densities / params.rho0)**params.gamma - 1.0)


def compute_pressure_force(
    positions: np.ndarray,
    densities: np.ndarray,
    pressures: np.ndarray,
    particle_types: np.ndarray,
    mass: float,
    kernel_fn,
    kernel_grad_fn,
    cll: CellLinkedList,
    h: float,
    support_radius: float,
) -> np.ndarray:
    """
    Compute pressure force using SPH.

    f_pressure_i = -Σ_j m_j (P_i/ρ_i² + P_j/ρ_j²) ∇W(r_ij, h)

    Args:
        positions: Particle positions (n, 2)
        densities: Particle densities (n,)
        pressures: Particle pressures (n,)
        particle_types: Particle types (n,)
        mass: Particle mass
        kernel_fn: Kernel function
        kernel_grad_fn: Kernel gradient function
        cll: Cell-linked list
        h: Smoothing length
        support_radius: Kernel support radius multiplier

    Returns:
        Pressure force array (n, 2)
    """
    n_particles = len(positions)
    forces = np.zeros((n_particles, 2), dtype=float)

    search_radius = support_radius * h

    for i in range(n_particles):
        neighbors = cll.find_neighbors(i, positions, search_radius)

        for j in neighbors:
            r_ij_vec = positions[j] - positions[i]
            r_ij = np.linalg.norm(r_ij_vec)

            if r_ij < 1e-10:
                continue

            # Prevent pressure from negative density
            rho_i = max(densities[i], 1e-6)
            rho_j = max(densities[j], 1e-6)

            # Pressure term
            pressure_term = (
                pressures[i] / (rho_i**2) + pressures[j] / (rho_j**2)
            )

            # Kernel gradient
            grad_w = kernel_grad_fn(r_ij_vec[np.newaxis, :], np.array([r_ij]), h)[0]

            forces[i] -= mass * pressure_term * grad_w

    return forces


def compute_viscosity_force(
    positions: np.ndarray,
    velocities: np.ndarray,
    densities: np.ndarray,
    particle_types: np.ndarray,
    mass: float,
    kernel_fn,
    kernel_grad_fn,
    cll: CellLinkedList,
    h: float,
    support_radius: float,
    alpha: float,
    c0: float,
) -> np.ndarray:
    """
    Compute artificial viscosity force (Monaghan).

    f_visc_i = -Σ_j m_j Π_ij ∇W(r_ij, h)

    Where Π_ij = {
        -α*c0*h/(ρ_ij) * u_ij·r_ij / |r_ij|²  if u_ij·r_ij < 0
        0                                      otherwise
    }

    Args:
        positions: Particle positions (n, 2)
        velocities: Particle velocities (n, 2)
        densities: Particle densities (n,)
        particle_types: Particle types (n,)
        mass: Particle mass
        kernel_fn: Kernel function
        kernel_grad_fn: Kernel gradient function
        cll: Cell-linked list
        h: Smoothing length
        support_radius: Kernel support radius multiplier
        alpha: Artificial viscosity coefficient
        c0: Speed of sound

    Returns:
        Viscosity force array (n, 2)
    """
    n_particles = len(positions)
    forces = np.zeros((n_particles, 2), dtype=float)

    search_radius = support_radius * h

    for i in range(n_particles):
        neighbors = cll.find_neighbors(i, positions, search_radius)

        for j in neighbors:
            r_ij_vec = positions[j] - positions[i]
            r_ij = np.linalg.norm(r_ij_vec)

            if r_ij < 1e-10:
                continue

            u_ij = velocities[i] - velocities[j]
            u_ij_dot_r_ij = np.dot(u_ij, r_ij_vec)

            # Only apply if particles approaching
            if u_ij_dot_r_ij < 0:
                rho_ij = 0.5 * (densities[i] + densities[j])
                pi_ij = -alpha * c0 * h / rho_ij * u_ij_dot_r_ij / (r_ij**2)

                grad_w = kernel_grad_fn(
                    r_ij_vec[np.newaxis, :], np.array([r_ij]), h
                )[0]

                forces[i] -= mass * pi_ij * grad_w

    return forces


def compute_gravitational_force(
    densities: np.ndarray, particle_types: np.ndarray, params: SPHParams
) -> np.ndarray:
    """
    Compute gravitational force.

    Args:
        densities: Particle densities
        particle_types: Particle types (0=fluid, 1=boundary)
        params: SPH parameters

    Returns:
        Gravitational force array (n, 2)
    """
    n_particles = len(densities)
    forces = np.zeros((n_particles, 2), dtype=float)

    # Gravity only on fluid particles
    for i in range(n_particles):
        if particle_types[i] == 0:  # Fluid
            forces[i, 1] = -params.g

    return forces


def apply_xsph_correction(
    positions: np.ndarray,
    velocities: np.ndarray,
    densities: np.ndarray,
    particle_types: np.ndarray,
    mass: float,
    kernel_fn,
    cll: CellLinkedList,
    h: float,
    support_radius: float,
    eps: float,
) -> np.ndarray:
    """
    Apply XSPH velocity correction.

    v'_i = v_i + ε Σ_j (m_j / ρ_j) (v_j - v_i) W(r_ij, h)

    Args:
        positions: Particle positions (n, 2)
        velocities: Particle velocities (n, 2)
        densities: Particle densities (n,)
        particle_types: Particle types (n,)
        mass: Particle mass
        kernel_fn: Kernel function
        cll: Cell-linked list
        h: Smoothing length
        support_radius: Kernel support radius multiplier
        eps: XSPH coefficient

    Returns:
        Corrected velocities (n, 2)
    """
    n_particles = len(positions)
    velocities_corrected = velocities.copy()

    search_radius = support_radius * h

    for i in range(n_particles):
        neighbors = cll.find_neighbors(i, positions, search_radius)

        correction = np.zeros(2, dtype=float)
        for j in neighbors:
            r_ij = np.linalg.norm(positions[j] - positions[i])
            rho_j = max(densities[j], 1e-6)
            w_ij = kernel_fn(r_ij, h)
            correction += (mass / rho_j) * (velocities[j] - velocities[i]) * w_ij

        velocities_corrected[i] += eps * correction

    return velocities_corrected


# ============================================================================
# Time Integration
# ============================================================================

def compute_time_step(
    velocities: np.ndarray,
    accelerations: np.ndarray,
    densities: np.ndarray,
    pressures: np.ndarray,
    params: SPHParams,
) -> float:
    """
    Compute adaptive time step using CFL condition.

    dt = CFL * h / max(c_s, |u| + sqrt(|∇P|/ρ))

    Args:
        velocities: Particle velocities (n, 2)
        accelerations: Particle accelerations (n, 2)
        densities: Particle densities (n,)
        pressures: Particle pressures (n,)
        params: SPH parameters

    Returns:
        Time step
    """
    # Speed of particles
    speeds = np.linalg.norm(velocities, axis=1)
    max_speed = np.max(speeds) + 1e-10

    # Sound speed
    c_s = params.c0

    # Maximum signal speed
    signal_speed = c_s + max_speed

    # CFL time step
    dt = params.CFL * params.h / signal_speed

    # Clamp to bounds
    dt = np.clip(dt, params.min_dt, params.max_dt)

    return dt


# ============================================================================
# Main Simulation Loop
# ============================================================================

def run_simulation(
    dx: float = 0.02,
    t_end: float = 2.0,
    output_dir: str = "./output",
    output_interval: int = 10,
) -> None:
    """
    Run the 2D dam break simulation.

    Args:
        dx: Particle spacing
        t_end: Final simulation time
        output_dir: Directory for output files
        output_interval: Save every N steps
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize parameters
    params = SPHParams(dx=dx)

    # Create particles
    print("Creating particles...")
    positions, velocities, particle_types = create_dam_break_particles(dx, params.h)
    n_particles = len(positions)
    mass = params.rho0 * (dx**2)  # Particle mass

    print(f"  Total particles: {n_particles}")
    print(f"    Fluid: {np.sum(particle_types == 0)}")
    print(f"    Boundary: {np.sum(particle_types == 1)}")
    print(f"  Particle mass: {mass:.6e} kg")
    print(f"  Smoothing length h: {params.h:.6f} m")
    print(f"  Speed of sound c0: {params.c0:.2f} m/s")

    # Get kernel
    kernel_fn, kernel_grad_fn, support_radius = get_kernel("cubic_spline", dim=2)

    # Initialize simulation state
    densities = np.full(n_particles, params.rho0, dtype=float)
    pressures = np.zeros(n_particles, dtype=float)
    accelerations = np.zeros((n_particles, 2), dtype=float)

    # Simulation loop
    t = 0.0
    step = 0
    output_count = 0

    print("\nStarting simulation...")
    print("=" * 70)

    while t < t_end:
        # Build neighbor list
        cll = CellLinkedList(
            np.array([0.0, 0.0]) - 2.0 * params.h,
            np.array([4.0, 3.0]) + 2.0 * params.h,
            cell_size=2.0 * params.h,
        )
        cll.build(positions)

        # Compute densities
        densities = compute_density(
            positions,
            velocities,
            particle_types,
            mass,
            kernel_fn,
            cll,
            params.h,
            support_radius,
        )

        # Compute pressures
        pressures = compute_pressure(densities, params)

        # Compute forces
        f_pressure = compute_pressure_force(
            positions,
            densities,
            pressures,
            particle_types,
            mass,
            kernel_fn,
            kernel_grad_fn,
            cll,
            params.h,
            support_radius,
        )

        f_viscosity = compute_viscosity_force(
            positions,
            velocities,
            densities,
            particle_types,
            mass,
            kernel_fn,
            kernel_grad_fn,
            cll,
            params.h,
            support_radius,
            params.alpha,
            params.c0,
        )

        f_gravity = compute_gravitational_force(densities, particle_types, params)

        # Total acceleration (only for fluid particles)
        accelerations = np.zeros((n_particles, 2), dtype=float)
        for i in range(n_particles):
            if particle_types[i] == 0:  # Fluid
                rho_i = max(densities[i], 1e-6)
                accelerations[i] = (f_pressure[i] + f_viscosity[i]) / rho_i + f_gravity[i]

        # Compute time step
        dt = compute_time_step(velocities, accelerations, densities, pressures, params)

        # Leapfrog integration
        # Update velocities
        velocities += accelerations * dt

        # Update positions
        positions += velocities * dt

        # Apply XSPH correction
        velocities = apply_xsph_correction(
            positions,
            velocities,
            densities,
            particle_types,
            mass,
            kernel_fn,
            cll,
            params.h,
            support_radius,
            params.eps_xsph,
        )

        # Enforce boundary conditions (particles shouldn't cross walls)
        positions = np.clip(positions, 0.0, 4.0)  # x bounds
        positions[:, 1] = np.maximum(positions[:, 1], 0.0)  # y lower bound

        # Save output
        if step % output_interval == 0:
            output_file = os.path.join(output_dir, f"particles_{output_count:05d}.npz")
            np.savez(
                output_file,
                positions=positions,
                velocities=velocities,
                densities=densities,
                pressures=pressures,
                particle_types=particle_types,
                time=t,
            )
            output_count += 1

        # Print progress
        if step % 10 == 0:
            max_vel = np.max(np.linalg.norm(velocities, axis=1))
            print(
                f"Step {step:6d} | t={t:8.4f}s | dt={dt:.6f}s | "
                f"Particles={n_particles} | max(u)={max_vel:8.3f} m/s"
            )

        t += dt
        step += 1

    print("=" * 70)
    print(f"Simulation completed: {step} steps, t={t:.4f}s")
    print(f"Output saved to {output_dir}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="2D Dam Break SPH Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sph_2d_dambreak.py
  python sph_2d_dambreak.py --dx 0.01 --t_end 3.0
  python sph_2d_dambreak.py --dx 0.02 --output_dir /tmp/dambreak
        """,
    )

    parser.add_argument(
        "--dx",
        type=float,
        default=0.02,
        help="Particle spacing (m) [default: 0.02]",
    )

    parser.add_argument(
        "--t_end",
        type=float,
        default=2.0,
        help="Final simulation time (s) [default: 2.0]",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory [default: ./output]",
    )

    parser.add_argument(
        "--output_interval",
        type=int,
        default=10,
        help="Save every N steps [default: 10]",
    )

    args = parser.parse_args()

    try:
        run_simulation(
            dx=args.dx,
            t_end=args.t_end,
            output_dir=args.output_dir,
            output_interval=args.output_interval,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
