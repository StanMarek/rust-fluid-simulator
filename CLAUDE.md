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

Cargo workspace implementing a real-time 2D SPH (Smoothed Particle Hydrodynamics) fluid simulator with GPU acceleration. Currently in early phases (gravity-only solver works; SPH kernels and forces are implemented but not yet wired into the main simulation loop).

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

`Simulation<D>` holds `ParticleStorage<D>`, `SimConfig`, time, step count. Current `step()` does: clear accelerations → apply gravity → symplectic Euler integration → boundary enforcement (clamp + velocity reflection).

SPH modules exist but are **not yet called** from `step()`:
- `sph/density.rs` — density summation via kernel contributions
- `sph/pressure.rs` — Tait equation (WCSPH, γ=7) + pressure gradient forces
- `sph/viscosity.rs` — artificial viscosity forces
- `sph/kernel.rs` — `SmoothingKernel` trait + `CubicSplineKernel` (implemented for `Dim2` only)
- `neighbor/grid.rs` — `SpatialHashGrid<D>` using spatial hashing with large primes, cell size = h

### Rendering

`renderer/particle_renderer.rs` — Instanced quad rendering via wgpu. Each particle is a `ParticleInstance` (world_pos, color, radius) rendered as a circle via WGSL fragment shader. Instance buffer auto-grows (power-of-two). Camera is 2D with pan/zoom (`renderer/camera.rs`).

Color maps: viridis, plasma, coolwarm, water — in `renderer/color_map.rs`.

### GPU Backend

`sim-gpu/` — Scaffolded but not implemented. `GpuContext` (device/queue setup) works. `ParticleBuffers` is a placeholder. WGSL compute shaders exist as files but pipeline implementations are stubs. Intended to mirror sim-core's SPH loop on GPU.

### UI

`ui/` uses egui/eframe. Panels: toolbar (emit/drag/erase/obstacle tools), properties (sim parameter sliders), scene (preset loading), viewport (wgpu surface), timeline (play/pause/step). `InteractionState` handles tool selection and mouse→world coordinate conversion.

### Scene System

Scenes defined as `SceneDescription` (name, config, emitters) in `sim-core/scene.rs`. Emitters spawn particles in rectangular grids. Built-in presets: `dam_break()`, `double_emitter()`. JSON scene files in `assets/scenes/`.

## Key Conventions

- **Math**: `nalgebra` for simulation vectors, `glam` for rendering. `bytemuck` for GPU buffer data.
- **Configs**: `SimConfig` is `Serialize`/`Deserialize` via serde. Defaults: h=0.1, ρ₀=1000, k=1000, μ=0.1, dt=0.001, gravity=[0,-9.81].
- **Sim↔UI communication**: `SimEvent` enum (Play/Pause/Step/Reset/Spawn/ApplyForce/Erase/UpdateConfig) and `SimStatus` struct in `common/events.rs`.
- **SPH parameter relationships**: smoothing radius `h` and particle spacing should satisfy `h ≈ 1.2 * spacing`. Speed of sound must be ≥ 10x max fluid velocity for WCSPH stability. Fixed timestep with accumulator pattern decouples sim and render rates.

## Implementation Status (per SPEC.md phases)

- **Phase 1** (Foundation): Complete — particles fall, bounce, render
- **Phase 2** (SPH Physics): Modules written but not integrated into simulation loop
- **Phase 3** (Interactivity): Partially scaffolded (tools, panels exist as structures)
- **Phase 4** (GPU): Scaffolded only (context works, pipelines are stubs)
- **Phase 5** (Web): Entry point exists, not tested
- **Phase 6** (Polish): Not started (surface tension stub, export trait defined)
