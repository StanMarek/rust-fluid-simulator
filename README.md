# Fluid Simulator

Real-time 2D SPH (Smoothed Particle Hydrodynamics) fluid simulation with CPU and GPU backends.

## Quick Start

```bash
# Desktop (native)
cargo run -p desktop --release

# Benchmarks
cargo run -p benchmark --release -- --particles 50000 --steps 1000
```

## Features

- Full SPH pipeline: density, pressure (Tait/WCSPH), viscosity, gravity, boundary and obstacle enforcement
- CPU backend with Rayon parallelization, GPU backend with wgpu compute shaders
- Live CPU/GPU backend switching via UI toggle
- Interactive tools: emit, drag, erase, obstacle placement
- Scene presets (dam break, double emitter) and JSON scene file loading
- Velocity-based color mapping (water, viridis, plasma, coolwarm)

## Architecture

Cargo workspace with the following crates:

- **common** — Shared types, dimension traits, configuration
- **sim-core** — CPU-based SPH simulation (Rayon-parallelized)
- **sim-gpu** — GPU compute backend via wgpu (5 WGSL compute shaders)
- **renderer** — wgpu-based instanced particle visualization
- **ui** — egui interface panels, backend abstraction, app orchestration
- **desktop** — Native desktop entry point
- **web** — WASM/WebGPU entry point
- **benchmark** — Headless performance benchmarks
