# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Build all crates (check only)
cargo check

# Run desktop app (always use --release for real performance)
cargo run -p desktop --release

# Run benchmarks
cargo run -p benchmark --release -- --particles 50000 --steps 1000

# Build for web (WASM)
cd apps/web && wasm-pack build --target web --out-dir pkg

# Check a specific crate
cargo check -p sim-core
cargo check -p renderer
```

There are no tests yet. Debug builds are 10-50x slower for SPH — always use `--release` for performance testing.

## Architecture

Cargo workspace implementing a real-time 2D SPH (Smoothed Particle Hydrodynamics) fluid simulator with GPU acceleration. SPH physics fully integrated with Rayon-parallelized neighbor search, density, pressure, viscosity, and boundary enforcement. Interactive tools (emit, drag, erase, obstacle) and UI panels are functional.

### Crate Dependency Graph

```
apps/desktop, apps/web
    └── ui (egui panels, app orchestration)
            ├── renderer (wgpu particle visualization)
            │     └── common
            ├── sim-core (CPU physics)
            │     └── common
            └── common (Dimension trait, SimConfig, events)

sim-gpu (GPU compute, mirrors sim-core) → common
tools/benchmark → sim-core, common
```

### Dimensional Generics

All simulation code is generic over `D: Dimension` (defined in `common/dimension.rs`). `Dim2` uses `nalgebra::Vector2<f32>`, `Dim3` uses `nalgebra::Vector3<f32>`. The `Dimension` trait provides vector operations (zero, magnitude, dot, normalize, clamp, component access, from_slice). Currently only `Dim2` is actively used.

### Particle Storage (SoA)

`sim-core/particle/storage.rs` — Struct of Arrays layout: separate `Vec`s for positions, velocities, accelerations, densities, pressures, masses. Uses `swap_remove` for O(1) particle deletion. This layout is designed for cache-friendly access and direct GPU buffer mapping.

### Simulation Loop (`sim-core/simulation.rs`)

`Simulation<D>` holds `ParticleStorage<D>`, `SpatialHashGrid<D>`, `SimConfig`, `Vec<Obstacle<D>>`, time, step count. The `step()` method runs the full SPH pipeline each frame, parallelized with Rayon:

1. Clear accelerations
2. Build spatial hash grid (`neighbor/grid.rs` — `SpatialHashGrid<D>`, cell_size = 2h, zero-allocation neighbor iterator)
3. Compute densities (`sph/density.rs` — SPH kernel summation)
4. Compute pressures (`sph/pressure.rs` — Tait equation, WCSPH, γ=7)
5. Compute pressure gradient forces (`sph/pressure.rs`)
6. Compute viscosity forces (`sph/viscosity.rs` — artificial viscosity via Laplacian kernel)
7. Apply gravity (`solver.rs`)
8. Symplectic Euler integration (`solver.rs`)
9. Boundary enforcement (`boundary.rs` — clamp + velocity reflection with damping)
10. Obstacle enforcement (`boundary.rs` — SDF-based collision with velocity reflection)

Supporting modules:
- `sph/kernel.rs` — `SmoothingKernel` trait + `CubicSplineKernel` (w, grad_w, laplacian_w for `Dim2`)
- `neighbor/grid.rs` — flat sorted-array spatial hash with large primes, 9-cell 2D query
- `obstacle.rs` — `Obstacle<D>` enum (Circle, Box) with SDF and normal computation

### Rendering

`renderer/particle_renderer.rs` — Instanced quad rendering via wgpu. Each particle is a `ParticleInstance` (world_pos, color, radius) rendered as a circle via WGSL fragment shader. Instance buffer auto-grows (power-of-two). Camera is 2D with pan/zoom (`renderer/camera.rs`).

Color maps: viridis, plasma, coolwarm, water — in `renderer/color_map.rs`.

### GPU Backend

`sim-gpu/` — Scaffolded but not implemented. `GpuContext` (device/queue setup) works. `ParticleBuffers` is a placeholder. WGSL compute shaders exist as files but pipeline implementations are stubs. Intended to mirror sim-core's SPH loop on GPU.

### UI

`ui/` uses egui/eframe with a custom theme (`ui/theme.rs`). Single left sidebar contains: custom-painted transport controls (play/pause/step/reset with working speed multiplier), 2x2 tool grid (emit/drag/erase/obstacle), collapsible property panels (physics, gravity, timestep, boundary), and scene/display options (presets + JSON file loading via `rfd`). Bottom status bar shows play state, particle count, FPS, sim time, step count. Center viewport renders particles via wgpu with domain boundary corners, obstacle shapes (semi-transparent), and tool cursor visualization. `InteractionState` handles tool selection, mouse→world coordinate conversion, and drag-based emit/erase/force/obstacle application.

### Scene System

Scenes defined as `SceneDescription` (name, config, emitters) in `sim-core/scene.rs`. Emitters spawn particles in rectangular grids. Built-in presets: `dam_break()`, `double_emitter()`. JSON scene files in `assets/scenes/`. Scenes can be loaded from JSON files via native file dialog.

## Key Conventions

- **Math**: `nalgebra` for simulation vectors, `glam` for rendering. `bytemuck` for GPU buffer data.
- **Configs**: `SimConfig` is `Serialize`/`Deserialize` via serde. Defaults: h=0.026, ρ₀=1000, k=50000, μ=0.01, dt=0.0002, gravity=[0,-9.81].
- **Sim↔UI communication**: `SimEvent` enum (Play/Pause/Step/Reset/Spawn/ApplyForce/Erase/UpdateConfig) and `SimStatus` struct in `common/events.rs`.
- **SPH parameter relationships**: smoothing radius `h` and particle spacing should satisfy `h ≈ 1.2 * spacing`. Speed of sound must be ≥ 10x max fluid velocity for WCSPH stability. Fixed timestep with accumulator pattern decouples sim and render rates.

## Implementation Status (per SPEC.md phases)

- **Phase 1** (Foundation): Complete — particles fall, bounce, render
- **Phase 2** (SPH Physics): Complete — full SPH pipeline integrated into simulation loop, Rayon-parallelized
- **Phase 3** (Interactivity): Complete — emit/drag/erase/obstacle tools functional, properties panel with live sliders, scene presets + JSON file loading, timeline with working step button and speed multiplier, velocity-based color mapping
- **Phase 4** (GPU): Scaffolded only (context works, pipelines are stubs, shaders orphaned)
- **Phase 5** (Web): Entry point exists, not tested
- **Phase 6** (Polish): Not started (surface tension stub, export trait defined)
