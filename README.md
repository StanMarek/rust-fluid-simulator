# Fluid Simulator

Real-time 2D SPH (Smoothed Particle Hydrodynamics) fluid simulation built in Rust with dual CPU/GPU backends, interactive tools, and a custom egui interface.

## Quick Start

```bash
# Desktop (native) — always use --release for real performance
cargo run -p desktop --release

# Benchmarks
cargo run -p benchmark --release -- --particles 50000 --steps 1000

# Build for web (WASM/WebGPU)
cd apps/web && wasm-pack build --target web --out-dir pkg
```

> Debug builds are 10-50x slower for SPH — always use `--release` for performance testing.

## Features

### Physics
- Full SPH pipeline: density estimation, pressure (Tait EOS / WCSPH, γ=7), artificial viscosity, gravity
- Cubic spline smoothing kernel with gradient and Laplacian
- Spatial hash grid for O(n) neighbor search (cell size = 2h, 9-cell query)
- Symplectic Euler integration with boundary and obstacle enforcement
- SDF-based obstacle collisions (circles and boxes) with velocity reflection

### Backends
- **CPU** — Rayon-parallelized across all pipeline stages
- **GPU** — wgpu compute shaders (5 dispatches per step: clear/gravity → density → pressure → forces → integrate)
- Live CPU/GPU switching at runtime via UI toggle

### Interaction
- **Emit** — Click/drag to spawn particles
- **Drag** — Apply forces to nearby particles
- **Erase** — Remove particles under cursor
- **Obstacle** — Place circle/box obstacles in the domain

### UI
- Single left sidebar with transport controls (play/pause/step/reset, speed multiplier)
- 2x2 custom-painted tool grid
- Collapsible property panels (physics parameters, gravity, timestep, boundary damping)
- Scene presets and JSON scene file loading via native file dialog
- CPU/GPU backend toggle
- Velocity-based color mapping (water, viridis, plasma, coolwarm)
- Bottom status bar: play state, backend, particle count, FPS, sim time, step count

## Architecture

Cargo workspace with the following crates:

```
apps/desktop, apps/web
    └── ui (egui panels, app orchestration, backend switching)
            ├── renderer (wgpu instanced particle visualization)
            │     └── common
            ├── sim-core (CPU SPH physics, Rayon-parallelized)
            │     └── common
            ├── sim-gpu (GPU compute shaders via wgpu)
            │     ├── sim-core
            │     └── common
            └── common (Dimension trait, SimConfig, events)

tools/benchmark → sim-core, common
```

| Crate | Description |
|-------|-------------|
| **common** | Shared types: `Dimension` trait (Dim2/Dim3 via nalgebra), `SimConfig`, `SimEvent`/`SimStatus` |
| **sim-core** | CPU simulation: particle storage (SoA), spatial hash grid, SPH kernels, solver, boundary/obstacle enforcement |
| **sim-gpu** | GPU compute backend: 5 WGSL shaders mirroring the CPU pipeline, particle/grid GPU buffers with staging readback |
| **renderer** | wgpu instanced quad rendering, 2D camera with pan/zoom, color maps |
| **ui** | egui/eframe app: `SimulationBackend` abstraction, toolbar/properties/scene/viewport/status panels, interaction state |
| **desktop** | Native entry point (eframe) |
| **web** | WASM/WebGPU entry point |
| **benchmark** | Headless performance benchmarks |

## Scene System

Scenes are defined as JSON files with simulation config and particle emitters. Built-in presets: **Dam Break**, **Double Emitter**. Custom scenes can be loaded via the file dialog.

```json
{
    "name": "Dam Break",
    "config": {
        "smoothing_radius": 0.026,
        "rest_density": 1000.0,
        "stiffness": 50000.0,
        "viscosity": 0.01,
        "gravity": [0.0, -9.81],
        "time_step": 0.0002,
        "domain_min": [0.0, 0.0],
        "domain_max": [1.0, 1.0],
        "boundary_damping": 0.3
    },
    "emitters": [
        {
            "center": [0.25, 0.5],
            "half_size": [0.2, 0.4],
            "spacing": 0.02,
            "initial_velocity": [0.0, 0.0]
        }
    ]
}
```

Scene files are located in `assets/scenes/`.

## Key Dependencies

| Library | Usage |
|---------|-------|
| nalgebra | Simulation vectors (Vector2/Vector3) |
| glam | Rendering math |
| wgpu | GPU compute shaders and particle rendering |
| egui / eframe | UI framework |
| rayon | CPU parallelization |
| bytemuck | GPU buffer data casting |
| serde / serde_json | Config and scene serialization |

## Implementation Status

- **Phase 1** (Foundation) — Complete: particles fall, bounce, render
- **Phase 2** (SPH Physics) — Complete: full pipeline with Rayon parallelization
- **Phase 3** (Interactivity) — Complete: tools, properties panel, scene presets, color mapping
- **Phase 4** (GPU Compute) — Complete: full SPH pipeline on GPU with live CPU/GPU toggle
- **Phase 5** (Web) — Entry point exists, not yet tested
- **Phase 6** (Polish) — Not started
