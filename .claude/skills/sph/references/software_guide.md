# SPH Software Guide: Comparison and Selection

Comprehensive comparison of open-source and commercial SPH simulation frameworks.

---

## 1. DualSPHysics

**Overview**: GPU-accelerated SPH code developed by CIMNE (Universitat Politècnica de Catalunya). Widely used in coastal engineering, industrial fluid dynamics.

**Features**:
- GPU support: CUDA (NVIDIA) and OpenCL (portable)
- Multi-phase simulation (liquid-gas, liquid-liquid)
- Boundary conditions: fixed walls, moving objects, symmetry planes
- Post-processing: built-in tools for energy analysis, wave generation
- Free-surface tracking via color function
- Two formulations: weakly-compressible (WCSPH) and incompressible (ISPH)
- Couple with FEA solver (ANSYS, RADIOSS) for FSI

**Strengths**:
- Mature codebase (active development 2010–present)
- Excellent documentation and tutorials
- Robust free-surface handling
- Fast GPU execution (10–100× CPU speedup)
- Pre/post-processing tools (ParaView integration)
- Active user community

**Weaknesses**:
- Complex installation (dependencies: VTK, HDF5, CUDA/OpenCL)
- Limited extensibility for custom physics (not easy to add new equations)
- Requires NVIDIA GPU for production runs (OpenCL slower)
- Less suitable for academic research requiring algorithm experimentation

**Typical Use Cases**:
- Coastal engineering (wave-structure interaction, tsunami propagation)
- Dam break simulations
- Multiphase flows (oil spillage, industrial separation)
- Ship hydrodynamics

**Getting Started**:
```bash
# Clone from GitHub
git clone https://github.com/DualSPHysics/DualSPHysics.git
cd DualSPHysics

# Compile (requires CUDA toolkit 11.0+, GCC)
cd src/cpp
make GPU_COMPUTE=60  # Compile for GPU compute capability 6.0+

# Run example
cd ../../examples/xml/01_DamBreak
../../bin/DualSPHysics -h  # Check help
../../bin/DualSPHysics DamBreak_Example -gpu
```

**Performance**: ~50–200 million particles per GPU (RTX 3090) per second, depending on kernel and precision.

**Documentation**: https://dualsphysics.github.io/

---

## 2. PySPH

**Overview**: Pure-Python SPH framework for rapid prototyping and research. Emphasizes equation system specification and code clarity.

**Features**:
- Particle filters and helper equations (pressure boundary conditions, etc.)
- Custom equation specification via `Equation` subclass
- SPH variants: WCSPH, ISPH, EDAC (energy-stable)
- OpenMP parallelization (multi-core CPU)
- Pure Python code; easy to modify and experiment
- Includes examples: dam break, cavity, FSI, rotating fluids

**Architecture**:
```python
# Define equations as composable objects
equations = [
    # Group 1: Continuity (density evolution)
    Group(equations=[
        ContinuityEquation(dest='fluid', sources=['fluid']),
    ]),
    # Group 2: Momentum
    Group(equations=[
        MomentumEquation(dest='fluid', sources=['fluid']),
        GravityForce(dest='fluid'),
    ]),
]
```

**Strengths**:
- Minimal setup boilerplate; equations transparent and readable
- Rapid algorithm prototyping (hours vs. days in C++)
- Excellent for educational purposes
- Comprehensive example gallery
- Active development and responsive maintainers

**Weaknesses**:
- Slow execution: 1000–5000 particles/s on modern CPU (vs. 10M+/s on GPU)
- No GPU support (planned but not mature)
- Limited for production-scale simulations (>100k particles impractical)
- Requires Python 3.6+, dependency on NumPy, mpi4py

**Typical Use Cases**:
- Academic research: testing new algorithms
- Undergraduate/graduate education
- Prototyping before moving to production code
- Diagnostic/verification simulations (smaller problem sizes)

**Getting Started**:
```bash
# Install from PyPI
pip install pysph

# Or clone and install editable
git clone https://github.com/pyamsoft/pysph.git
cd pysph
pip install -e .

# Run example
python -c "
from pysph.examples.incompressible import TaylorGreenVortex
app = TaylorGreenVortex()
app.run()
"
```

**Documentation**: https://pysph.readthedocs.io/

**Code Example** (custom equation):
```python
from pysph.base.cython_generator import CythonGenerator
from pysph.sph.equation import Equation

class CustomViscosity(Equation):
    '''Custom viscous force with user-defined formula'''
    def __init__(self, dest, sources, nu=0.001):
        self.nu = nu
        super(CustomViscosity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au):
        d_au[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho, d_rho,
             d_u, d_v, s_u, s_v, d_au, d_av, XIJ, RIJ, R2IJ, WIJ):
        # Compute viscous force (your formula here)
        fac = 2.0 * self.nu / (d_rho[d_idx] + s_rho[s_idx])
        # ... update d_au, d_av
```

---

## 3. SPHinXsys

**Overview**: Multi-physics SPH framework targeting FSI (fluid-structure interaction), biomedical simulation, and coupled mechanics.

**Features**:
- Multi-physics: fluids (weakly-compressible), structures (solid mechanics), FSI
- Implicit pressure solver (incompressible flow)
- Built-in FSI solver (structure + fluid coupled)
- Hemodynamics (blood flow in vessels)
- Viscoelasticity and material models for solids
- Parallel: OpenMP + MPI

**Strengths**:
- Purpose-built for FSI problems (rare among SPH codes)
- Biomedical capability (hemodynamics, cell mechanics)
- Clean C++ design with template-based extensibility
- Good documentation for target applications
- Active research group (Shanghai Jiao Tong University)

**Weaknesses**:
- Steeper learning curve than PySPH
- Limited to CPU parallelization (no GPU)
- Smaller user community than DualSPHysics
- Niche focus (less suitable for pure hydrodynamics)

**Typical Use Cases**:
- Fluid-structure interaction (deformable structures in fluid)
- Biomedical: arterial flow, aneurysm rupture risk
- Material deformation with phase changes
- Coupled multi-field phenomena

**Getting Started**:
```bash
git clone https://github.com/Xiangyu-Hu/SPHinXsys.git
cd SPHinXsys
mkdir build && cd build
cmake .. -DBUILD_SHARED_LIBS=ON
make -j8

# Run example (e.g., 2D flow over cylinder)
cd ../examples/2d_examples/02_flow_around_cylinder
# Executable compiled in build/ directory
```

**Documentation**: https://sphinxsys.github.io/

---

## 4. SPlisHSPlasH

**Overview**: Research-oriented SPH framework emphasizing pressure solvers and algorithmic variants. Popular in computer graphics and animation research.

**Features**:
- Multiple pressure solvers: WCSPH, DFSPH (divergence-free), IISPH (implicit incompressible)
- Surface reconstruction and rendering
- Viscosity models: artificial, viscous boundary layer, SPH Laminar
- Neighborhood search: grid, compact hashing, SPH compact
- Fluid-solid coupling (simplified)
- Many equation variants in same codebase for comparison

**Strengths**:
- Research-focused: easy to swap pressure solvers and kernels
- Excellent choice for method comparison
- Good visual output tools (for graphics/animation)
- Modular design (switch solver with one parameter)
- Active development (contributors from multiple institutions)

**Weaknesses**:
- CPU-only (no GPU support)
- Less mature industrial documentation
- Smaller ecosystem (fewer tools than DualSPHysics)
- Limited to hydrodynamics (less suitable for solid mechanics)

**Typical Use Cases**:
- Research: comparing pressure solvers (WCSPH vs. DFSPH vs. IISPH)
- Animation and graphics
- Algorithm development and validation
- Prototyping before moving to production code

**Getting Started**:
```bash
git clone https://github.com/InteractiveComputerGraphics/SPlisHSPlasH.git
cd SPlisHSPlasH
mkdir build && cd build
cmake .. -DUSE_DOUBLE_PRECISION=ON
make -j8

# Configuration: modify config/config.json
# Run: ./SPlisHSPlasH ../examples/dam_break.json
```

**Documentation**: https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/wiki

---

## 5. GPUSPH

**Overview**: GPU-only SPH simulator optimized for NVIDIA GPUs using CUDA.

**Features**:
- Pure CUDA implementation (no CPU fallback)
- Periodic and open boundary conditions
- Multi-GPU support (MPI)
- Variety of test cases (dam break, lid-driven cavity, etc.)
- Vehicle crash simulation (SPH for fluid deformation)

**Strengths**:
- High GPU utilization; very fast for large particle counts
- Straightforward CUDA code; good for GPU algorithm development
- Supports multi-GPU clusters via MPI

**Weaknesses**:
- GPU required (no CPU version)
- Documentation sparse; less user-friendly
- Smaller community; less maintained
- Limited extensibility

**Typical Use Cases**:
- High-performance production simulations (millions of particles)
- GPU algorithm research
- Crash and impact dynamics

**Documentation**: Limited; primarily examples and comments in code. GitHub: https://github.com/GPUSPH/gpusph

---

## 6. OpenFPM (Open Framework for Particle Method)

**Overview**: Distributed-memory particle method framework targeting heterogeneous architectures (CPUs, GPUs, multi-node clusters).

**Features**:
- Abstract particle data structure; automatic memory management
- Distributed computing via MPI
- GPU acceleration via CUDA (transparent to user code)
- Generic neighbor search on distributed data
- Suitable for any particle-based method (SPH, DEM, Molecular Dynamics)

**Strengths**:
- Scalability: tested on thousands of nodes
- GPU transparent: same code runs on CPU, GPU, clusters
- General-purpose: useful beyond just SPH
- Excellent for large-scale simulations (> 10⁹ particles)

**Weaknesses**:
- Steep learning curve (complex abstraction layer)
- Limited pre-built SPH implementations (requires custom code)
- Sparse documentation for SPH users
- Designed for HPC; overkill for small simulations

**Typical Use Cases**:
- Massive parallel simulations (supercomputer scale)
- Multi-GPU clusters
- Custom particle methods beyond standard SPH

**Documentation**: https://github.com/mosaic-group/openfpm

---

## 7. SWIFT (SPH With INTEGRAL Treatment of Forces)

**Overview**: Modern astrophysical SPH code developed for cosmological simulations.

**Features**:
- Adaptive smoothing length: h varies per particle
- Task-based parallelism (excellent GPU scaling)
- Cosmological initial conditions and solver
- Multi-physics: gravity, cooling, star formation, feedback
- Produced data for Illustris TNG simulations (most cited cosmology papers)

**Strengths**:
- State-of-the-art parallel efficiency
- Purpose-built for cosmological SPH
- Production-proven (TeraScale simulations)
- Active development and support

**Weaknesses**:
- Astrophysics-only focus; not suitable for terrestrial engineering
- Requires supercomputer for significant problems
- Complex setup and parameter files

**Typical Use Cases**:
- Cosmological simulations
- Galaxy cluster evolution
- Astrophysical research

**Documentation**: https://swift.strathclyde.ac.uk/

---

## 8. Commercial Options

### 8.1 LS-DYNA (SPH Module)

**Overview**: General-purpose FEA/FEM code with SPH module for fluid, impact, and coupled problems.

**SPH Capabilities**:
- Hydrodynamics (water, explosives)
- Impact and penetration
- Coupled SPH-FEM (structures + fluid)
- Adaptive refinement

**Pros**:
- Integrated with large FEA ecosystem
- Industrial support and validation
- Multi-physics coupling robust

**Cons**:
- Expensive licensing
- SPH module less mature than structural analysis
- Overhead for pure fluid problems

**Cost**: $5k–50k/year depending on license type.

### 8.2 Altair radiOSS

**Overview**: Multi-physics solver with SPH for fluid-structure interaction.

**Capabilities**:
- Multiphase flows
- Coupled analysis (fluids + structures)
- GUI pre-processor

**Pros**:
- Integrated environment
- Good for industrial coupled problems

**Cons**:
- Expensive
- Less specialized in SPH than pure codes

**Cost**: $10k–50k/year.

---

## 9. Feature Comparison Matrix

| Feature | DualSPHysics | PySPH | SPHinXsys | SPlisHSPlasH | GPUSPH | OpenFPM | SWIFT |
|---------|-------------|-------|-----------|-------------|--------|---------|-------|
| **Free/Open** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **GPU Support** | ✓ (CUDA) | ✗ | ✗ | ✗ | ✓ (CUDA) | ✓ (CUDA) | ✓ (CUDA) |
| **Multi-GPU** | ✓ | Limited | ✗ | ✗ | ✓ (MPI) | ✓ (MPI) | ✓ (MPI) |
| **Multi-Phase** | ✓ | Limited | ✓ | Limited | ✗ | Generic | ✗ |
| **FSI** | Basic | ✗ | ✓✓ | Limited | ✗ | ✗ | ✗ |
| **Solid Mech.** | ✗ | ✗ | ✓✓ | ✗ | ✗ | ✗ | ✗ |
| **Biomedical** | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| **Astrophysics** | ✗ | Limited | ✗ | ✗ | ✗ | ✗ | ✓✓ |
| **DFSPH Solver** | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |
| **IISPH Solver** | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| **Performance** | 50M p/s | 1k p/s | 10M p/s | 10M p/s | 100M p/s | 50M p/s | 100M p/s |
| **Documentation** | ✓✓ | ✓✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| **User Community** | ✓✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ |
| **Ease of Use** | ✓ | ✓✓ | ✗ | ✓ | ✗ | ✗ | ✗ |

Legend: ✓✓ = excellent, ✓ = good, ✗ = not available, Limited = partial support

---

## 10. Decision Guide: Choosing an SPH Code

### 10.1 Flowchart

```
Start: Choose SPH code
│
├─ Research goal?
│  ├─ "Test new algorithms"  → PySPH or SPlisHSPlasH
│  ├─ "Compare solvers"      → SPlisHSPlasH
│  └─ "Production runs"      → Continue below
│
├─ Problem domain?
│  ├─ "Coastal/hydro"        → DualSPHysics
│  ├─ "Biomedicine/FSI"      → SPHinXsys
│  ├─ "Astrophysics"         → SWIFT
│  ├─ "Graphics/animation"   → SPlisHSPlasH
│  └─ "Ultra-large scale"    → OpenFPM + SWIFT
│
├─ Available hardware?
│  ├─ "GPU (NVIDIA)"         → DualSPHysics or GPUSPH
│  ├─ "Multi-node cluster"   → SWIFT or OpenFPM
│  └─ "Single CPU machine"   → PySPH or SPlisHSPlasH
│
└─ Performance target?
   ├─ "> 10M particles"      → GPU-accelerated (DualSPHysics)
   ├─ "10k–1M particles"     → CPU: SPlisHSPlasH, GPU: DualSPHysics
   └─ "< 10k particles"      → PySPH (ease of use)
```

### 10.2 Quick Selection Table

| Use Case | Recommended | Alternative | Rationale |
|----------|------------|-------------|-----------|
| **Academic research: new algorithm** | PySPH | SPlisHSPlasH | Fast iteration; readable code |
| **Coastal/marine engineering** | DualSPHysics | SPlisHSPlasH | Domain expertise; multi-phase |
| **Fluid-structure interaction** | SPHinXsys | DualSPHysics | FSI solver integrated |
| **Biomedical (hemodynamics)** | SPHinXsys | PySPH | Domain-specific tools |
| **Cosmological simulation** | SWIFT | GADGET-3 | Modern task-based parallelism |
| **High-performance (>10M p)** | DualSPHysics | GPUSPH | Mature; industrial |
| **Pressure solver comparison** | SPlisHSPlasH | PySPH | Multiple solvers; quick swap |
| **Teaching SPH concepts** | PySPH | — | Clearest code; minimal overhead |
| **Massive parallel (HPC)** | OpenFPM + SWIFT | — | Scalability proven |

---

## 11. Installation Summary

### 11.1 Quick Install Commands

**DualSPHysics**:
```bash
git clone https://github.com/DualSPHysics/DualSPHysics.git
cd DualSPHysics/src/cpp
make GPU_COMPUTE=60  # Requires CUDA 11+
```

**PySPH**:
```bash
pip install pysph
```

**SPHinXsys**:
```bash
git clone https://github.com/Xiangyu-Hu/SPHinXsys.git
cd SPHinXsys && mkdir build && cd build
cmake .. && make -j8
```

**SPlisHSPlasH**:
```bash
git clone https://github.com/InteractiveComputerGraphics/SPlisHSPlasH.git
cd SPlisHSPlasH && mkdir build && cd build
cmake .. && make -j8
```

---

## 12. Transition Path: From Research to Production

### Typical Workflow

```
Stage 1: Proof of Concept
  Code: PySPH
  Goal: Verify algorithm, validate against analytical solution
  Particle count: 1k–10k
  Time: 1–4 weeks

  ↓ Algorithm confirmed

Stage 2: Validation & Benchmarking
  Code: SPlisHSPlasH or PySPH
  Goal: Compare solvers (WCSPH vs DFSPH), tune parameters
  Particle count: 10k–100k
  Time: 2–8 weeks

  ↓ Parameters optimized

Stage 3: Production Simulation
  Code: DualSPHysics (if GPU available) or SPlisHSPlasH (CPU)
  Goal: High-resolution simulation, industrial validation
  Particle count: 100k–10M
  Time: varies (days to weeks per run)

  ↓ Results validated

Stage 4: Deployment (if needed)
  Integration into workflow; potential custom C++ optimization
```

