# SPH Performance Optimization Reference

This reference covers advanced performance techniques for smoothed particle hydrodynamics (SPH) simulations, from memory layout optimization to GPU acceleration.

## 1. Memory Layout

### Structure of Arrays (SoA) vs Array of Structures (AoS)

**Array of Structures (AoS)** - Traditional approach:
```python
# AoS: Each particle is a complete object
class Particle:
    x, y, z: float
    vx, vy, vz: float
    density, pressure: float

particles = [Particle() for _ in range(n)]
```

**Problems with AoS:**
- Poor cache utilization when accessing single attributes
- Stride = struct size (e.g., 64 bytes), leading to cache line waste
- SIMD inefficient: can't vectorize across particles easily
- Memory bandwidth underutilized

**Structure of Arrays (SoA)** - Modern approach:
```python
import numpy as np

class ParticleArray:
    def __init__(self, n):
        self.pos = np.zeros((n, 3), dtype=np.float32)      # SoA: position
        self.vel = np.zeros((n, 3), dtype=np.float32)      # velocity
        self.density = np.zeros(n, dtype=np.float32)       # density
        self.pressure = np.zeros(n, dtype=np.float32)      # pressure
        self.force = np.zeros((n, 3), dtype=np.float32)    # forces
        self.mass = np.ones(n, dtype=np.float32)           # mass
```

**Advantages of SoA:**
- Adjacent particles' positions are contiguous in memory (stride = 12 bytes for float32 triplet)
- Full L1 cache line (64 bytes) used for 5+ particles' attributes
- SIMD-friendly: vectorize operations across all particles
- Better memory bandwidth utilization

### Cache Line Considerations

**L1 Cache:** 64 bytes typical on modern CPUs
- For 4-byte float32: 16 floats per cache line
- For 8-byte float64: 8 doubles per cache line

**Optimal array sizes:**
```python
# Align to cache line (64 bytes)
n_particles = (n_particles + 7) // 8 * 8  # Align to 8 float32s

# Use float32 when precision permits (2x better cache efficiency than float64)
pos = np.zeros((n, 3), dtype=np.float32)  # 12 bytes per particle
```

### SIMD Alignment

NumPy arrays align automatically if allocated contiguously:
```python
# Create SIMD-aligned arrays (16-byte alignment minimum)
pos = np.zeros((n, 3), dtype=np.float32)  # C-contiguous, aligned
vel = np.ascontiguousarray(vel)           # Ensure contiguous if needed

# Verify alignment (should be multiple of 16)
assert pos.data.hex()[-2:] in ['00', '10', '20', '30', '40', '50', '60', '70']
```

**Best practices:**
- Use `np.float32` for better cache locality (64 bits vs 128 bits)
- Allocate in multiples of 8 particles (64 bytes / 8 bytes per float32)
- Keep arrays contiguous and C-ordered for row-wise iteration
- Avoid scalar indexing in hot loops (use vectorized operations)

---

## 2. Spatial Hashing

### Hash Function Design for 3D Grids

The spatial hash enables O(1) neighbor lookup instead of O(N²) all-pairs comparison.

**Basic grid hashing:**
```python
def spatial_hash_3d(pos, h, hash_table_size):
    """
    Hash 3D positions to grid cells using grid spacing h.

    pos: (n, 3) array of positions
    h: kernel radius (same as smoothing length)
    hash_table_size: size of hash table (prime number recommended)
    """
    # Grid cell coordinates
    cell_x = np.floor(pos[:, 0] / h).astype(np.int32)
    cell_y = np.floor(pos[:, 1] / h).astype(np.int32)
    cell_z = np.floor(pos[:, 2] / h).astype(np.int32)

    # Hash function: 3D cell coordinate -> 1D index
    # Use prime numbers to reduce collisions
    hash_vals = (
        cell_x * 73856093 ^
        cell_y * 19349663 ^
        cell_z * 83492791
    ) % hash_table_size

    return hash_vals
```

**Why these prime numbers?** They're chosen to distribute hash values uniformly and minimize collisions for typical spatial distributions.

### Morton Z-Order Curve (Space-Filling Curve)

The Z-order curve improves spatial locality by mapping 3D coordinates to a 1D index while preserving neighborhood relationships better than naive hashing.

```python
def morton_encode(x, y, z):
    """
    Encode 3D coordinates into Morton Z-order curve index.
    Interleaves bits: xxx... yyy... zzz... -> xyzxyzxyz...
    """
    # Spread bits of each coordinate
    def spread_bits(v):
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3
        v = (v | (v << 2)) & 0x09249249
        return v

    return (spread_bits(x) |
            (spread_bits(y) << 1) |
            (spread_bits(z) << 2))

def spatial_hash_morton(pos, h, max_coord=1024):
    """
    Hash using Morton curve for better spatial locality.
    """
    cell_x = np.clip(np.floor(pos[:, 0] / h), 0, max_coord-1).astype(np.int32)
    cell_y = np.clip(np.floor(pos[:, 1] / h), 0, max_coord-1).astype(np.int32)
    cell_z = np.clip(np.floor(pos[:, 2] / h), 0, max_coord-1).astype(np.int32)

    return np.array([morton_encode(cell_x[i], cell_y[i], cell_z[i])
                     for i in range(len(pos))])
```

**Benefits of Morton curve:**
- Neighboring cells in 3D map to nearby indices in 1D
- Better cache locality when iterating over neighbors
- Natural spatial coherence: particles close in space → close in hash order

### Rebuild Frequency

Hash table rebuild is expensive (O(n) operation) but necessary when particles move significantly.

**Heuristic for rebuild:**
```python
def should_rebuild_hash(pos, pos_old, h, rebuild_threshold=0.2):
    """
    Rebuild if any particle moved > rebuild_threshold * h since last rebuild.
    """
    max_displacement = np.max(np.linalg.norm(pos - pos_old, axis=1))
    return max_displacement > rebuild_threshold * h

# Typical strategy: rebuild every 4-8 timesteps
rebuild_interval = 4
if step % rebuild_interval == 0:
    hash_table = build_hash_table(pos, h, hash_table_size)
    pos_last_rebuild = pos.copy()
```

**Cost analysis:**
- Rebuild cost: O(n) - scan all particles once
- Neighbor search cost without rebuild: increases as O(n²) or worse if hash buckets degrade
- Rebuild every 4-8 steps typically optimal for dynamic simulations

---

## 3. NumPy Vectorization

### Vectorized Neighbor Interaction Loop

The key to NumPy performance is eliminating Python for-loops and using NumPy's vectorized operations.

**Naive loop version (slow):**
```python
def compute_density_loop(pos, mass, h):
    n = len(pos)
    density = np.zeros(n)

    for i in range(n):
        for j in range(n):
            r = np.linalg.norm(pos[i] - pos[j])
            if r < h:
                # Kernel evaluation (cubic spline)
                q = r / h
                if q < 1:
                    w = (1 - 1.5*q**2 + 0.75*q**3) * (8 / np.pi / h**3)
                else:
                    w = 0
                density[i] += mass[j] * w

    return density
```

**Vectorized version (fast):**
```python
def compute_density_vectorized(pos, mass, h):
    """
    Compute density for all particles using NumPy broadcasting.
    Uses pairwise distance computation.
    """
    n = len(pos)

    # Compute pairwise distances: shape (n, n, 3)
    # pos[:, None, :] broadcasts to (n, 1, 3)
    # pos[None, :, :] broadcasts to (1, n, 3)
    diff = pos[:, None, :] - pos[None, :, :]  # (n, n, 3)
    r = np.linalg.norm(diff, axis=2)           # (n, n)

    # Compute kernel values for all pairs
    q = r / h
    w = np.zeros_like(r)

    # Cubic spline kernel (vectorized)
    mask1 = q < 1
    mask2 = (q >= 1) & (q < 2)  # Some kernels have extended support

    w[mask1] = (1 - 1.5*q[mask1]**2 + 0.75*q[mask1]**3) * (8 / np.pi / h**3)
    w[mask2] = 0.25 * (2 - q[mask2])**3 * (8 / np.pi / h**3)

    # Accumulate contributions: mass[j] * w_ij
    # (n, n) matrix @ (n,) vector -> (n,) vector
    density = w @ mass

    return density
```

**Memory trade-off:**
- Vectorized version creates (n, n) temporary arrays: O(n²) memory
- For 10k particles: 100M floats × 4 bytes = 400 MB (acceptable)
- For 100k particles: 4 GB (problematic)

**Solution for large n: Tile-based processing**
```python
def compute_density_tiled(pos, mass, h, tile_size=1000):
    """
    Process particles in tiles to bound memory usage.
    """
    n = len(pos)
    density = np.zeros(n)

    for i_start in range(0, n, tile_size):
        i_end = min(i_start + tile_size, n)
        pos_i = pos[i_start:i_end]  # (tile_size, 3)

        # Pairwise distances within this tile vs all particles
        diff = pos_i[:, None, :] - pos[None, :, :]  # (tile_size, n, 3)
        r = np.linalg.norm(diff, axis=2)             # (tile_size, n)

        # Kernel computation
        q = r / h
        w = np.zeros_like(r)
        mask = q < 1
        w[mask] = (1 - 1.5*q[mask]**2 + 0.75*q[mask]**3) * (8 / np.pi / h**3)

        # Accumulate
        density[i_start:i_end] = w @ mass

    return density
```

### Force Computation (Pressure & Viscosity)

```python
def compute_forces_vectorized(pos, vel, mass, density, pressure, h):
    """
    Compute pressure and viscosity forces using vectorization.
    """
    n = len(pos)
    force = np.zeros((n, 3))

    diff = pos[:, None, :] - pos[None, :, :]  # (n, n, 3)
    r = np.linalg.norm(diff, axis=2)           # (n, n)

    # Avoid division by zero
    r = np.maximum(r, 1e-6)
    r_vec = np.where(r[:, :, None] > 0, diff / r[:, :, None], 0)  # Unit vectors

    # Kernel gradient: dW/dr = -45/(pi*h^6) * (h - r)^2
    q = r / h
    mask = q < 1

    # dW/dr - be careful with broadcasting
    dw_dr = np.zeros_like(r)
    dw_dr[mask] = -45 / (np.pi * h**6) * (h - r[mask])**2

    # Pressure force: -m_j * (p_i/rho_i^2 + p_j/rho_j^2) * dW/dr * r_hat
    p_i = pressure[:, None]  # (n, 1)
    p_j = pressure[None, :]  # (1, n)
    rho_i = density[:, None]
    rho_j = density[None, :]

    pressure_term = (p_i / rho_i**2 + p_j / rho_j**2)  # (n, n)

    # Force accumulation: sum over j for each i
    # dW/dr * r_vec has shape (n, n) and (n, n, 3)
    # Need to reshape for broadcasting
    f_pressure = np.zeros((n, 3))
    for dim in range(3):
        f_pressure[:, dim] = -np.sum(mass[None, :] *
                                     pressure_term *
                                     dw_dr *
                                     r_vec[:, :, dim], axis=1)

    force += f_pressure

    return force
```

### Practical Broadcasting Patterns

```python
# Pattern 1: Pairwise operations
A = pos[:, None, :]      # (n, 1, 3)
B = pos[None, :, :]      # (1, n, 3)
diff = A - B             # (n, n, 3) - broadcast

# Pattern 2: Reduction along particles (sum for each particle)
forces = (w * vectors).sum(axis=1)  # (n, n, 3) -> (n, 3)

# Pattern 3: Conditional operations
mask = r < h
result = np.zeros_like(r)
result[mask] = kernel(r[mask])

# Pattern 4: Accumulation with multiply-add
result = (A[None, :] * B[:, None]).sum(axis=1)  # (n,) @ (n,) -> (n,)
```

---

## 4. Numba JIT Compilation

Numba provides near-C performance for numerical code without rewriting in C++.

### Which Functions to JIT

**Good candidates for `@njit`:**
- Neighbor search loops
- Force computation loops
- Kernel functions (called millions of times)
- Integration routines

**Less suitable:**
- I/O operations
- NumPy functions not supported by Numba
- Recursive algorithms

### Basic JIT Compilation

```python
from numba import njit, jit, prange
import numpy as np

@njit
def cubic_spline_kernel(r, h):
    """Single kernel evaluation - called millions of times."""
    q = r / h
    if q < 1.0:
        return (1.0 - 1.5*q**2 + 0.75*q**3) * (8.0 / np.pi / h**3)
    elif q < 2.0:
        return 0.25 * (2.0 - q)**3 * (8.0 / np.pi / h**3)
    else:
        return 0.0

@njit
def cubic_spline_kernel_grad(r, h):
    """Kernel gradient."""
    q = r / h
    if q < 1.0:
        return (-3.0*q + 2.25*q**2) * (8.0 / np.pi / h**4)
    elif q < 2.0:
        return -0.75 * (2.0 - q)**2 * (8.0 / np.pi / h**4)
    else:
        return 0.0
```

### Parallel Loop with `parallel=True`

```python
@njit(parallel=True)
def compute_density_numba(pos, mass, h):
    """
    Compute density with parallel neighbor summation.
    """
    n = len(pos)
    density = np.zeros(n)

    for i in prange(n):  # prange: parallel range
        rho = 0.0
        for j in range(n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]

            r = np.sqrt(dx*dx + dy*dy + dz*dz)
            if r < h:
                w = cubic_spline_kernel(r, h)
                rho += mass[j] * w

        density[i] = rho

    return density
```

**Speedup from parallelization:**
- 4-core CPU: 3-3.5x speedup typical
- 8-core CPU: 6-7x speedup typical
- Overhead for small n (<1000): may not be worth it

### Typed Lists for Neighbor Data

```python
from numba.typed import List
from numba import types

@njit
def find_neighbors_numba(pos, h, hash_table):
    """
    Find neighbors using hash table (pre-built).
    Returns typed list of neighbor lists.
    """
    n = len(pos)
    neighbors = [List() for _ in range(n)]  # List of lists

    for i in range(n):
        # Find all cells within h distance (27 cells in 3D)
        cell_x = int(np.floor(pos[i, 0] / h))
        cell_y = int(np.floor(pos[i, 1] / h))
        cell_z = int(np.floor(pos[i, 2] / h))

        # Check 27 neighboring cells (3x3x3)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    neighbor_cell = (cell_x + dx, cell_y + dy, cell_z + dz)

                    # Retrieve particles in this cell from hash table
                    if neighbor_cell in hash_table:
                        for j in hash_table[neighbor_cell]:
                            dist_sq = ((pos[i, 0] - pos[j, 0])**2 +
                                       (pos[i, 1] - pos[j, 1])**2 +
                                       (pos[i, 2] - pos[j, 2])**2)

                            if dist_sq < h*h and i != j:
                                neighbors[i].append(j)

    return neighbors
```

### Force Computation with Numba

```python
@njit(parallel=True)
def compute_forces_numba(pos, vel, mass, density, pressure, h):
    """
    SPH force computation with Numba acceleration.
    """
    n = len(pos)
    force = np.zeros((n, 3))

    for i in prange(n):
        f = np.array([0.0, 0.0, 0.0])

        for j in range(n):
            if i == j:
                continue

            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]

            r = np.sqrt(dx*dx + dy*dy + dz*dz)

            if r < h and r > 1e-6:
                # Pressure force
                p_term = (pressure[i] / (density[i]**2) +
                         pressure[j] / (density[j]**2))

                dw = cubic_spline_kernel_grad(r, h)

                f[0] -= mass[j] * p_term * dw * dx / r
                f[1] -= mass[j] * p_term * dw * dy / r
                f[2] -= mass[j] * p_term * dw * dz / r

                # Viscosity force (simplified)
                dv_dot_r = ((vel[i, 0] - vel[j, 0]) * dx +
                           (vel[i, 1] - vel[j, 1]) * dy +
                           (vel[i, 2] - vel[j, 2]) * dz)

                if dv_dot_r < 0:
                    visc = 0.01  # viscosity coefficient
                    f[0] -= mass[j] * visc * dv_dot_r / r * dw * dx / r
                    f[1] -= mass[j] * visc * dv_dot_r / r * dw * dy / r
                    f[2] -= mass[j] * visc * dv_dot_r / r * dw * dz / r

        force[i] = f

    return force
```

### Compilation Tips

```python
# Control compilation options
from numba import njit

# No Python fallback if compilation fails
@njit(cache=True)  # Cache compiled code
def my_kernel(r, h):
    return r / h

# Force specific numba target
from numba import config
config.NUMBA_DEFAULT_NUM_THREADS = 4  # Limit threads
```

---

## 5. GPU Acceleration with CUDA

CuPy provides NumPy-like API for GPU computation using CUDA.

### Thread Mapping Strategy

**1 particle = 1 thread:**
```python
import cupy as cp

# Kernel for density computation
cuda_code = """
extern "C" {
    __global__ void compute_density(
        const float *pos,      // (n, 3)
        const float *mass,     // (n,)
        float *density,        // (n,) output
        int n,
        float h
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;

        float rho = 0.0f;
        float px = pos[i * 3 + 0];
        float py = pos[i * 3 + 1];
        float pz = pos[i * 3 + 2];

        for (int j = 0; j < n; j++) {
            float dx = px - pos[j * 3 + 0];
            float dy = py - pos[j * 3 + 1];
            float dz = pz - pos[j * 3 + 2];

            float r2 = dx*dx + dy*dy + dz*dz;
            float r = sqrtf(r2);

            if (r < h) {
                // Cubic spline kernel
                float q = r / h;
                float w;
                if (q < 1.0f) {
                    w = (1.0f - 1.5f*q*q + 0.75f*q*q*q) *
                        (8.0f / 3.14159f / h / h / h);
                } else {
                    w = 0.0f;
                }

                rho += mass[j] * w;
            }
        }

        density[i] = rho;
    }
}
"""

# Compile and load kernel
module = cp.RawModule(code=cuda_code)
compute_density_kernel = module.get_function("compute_density")

def compute_density_gpu(pos_gpu, mass_gpu, h):
    """Wrapper for GPU density computation."""
    n = len(pos_gpu)
    density_gpu = cp.zeros(n, dtype=cp.float32)

    # Block size 256 is typical for modern GPUs
    block_size = 256
    grid_size = (n + block_size - 1) // block_size

    compute_density_kernel(
        grid=(grid_size,), block=(block_size,),
        args=(pos_gpu, mass_gpu, density_gpu, n, h)
    )

    return density_gpu
```

### Shared Memory for Tile-Based Computation

**Reduce global memory traffic with shared memory (fast, but limited):**

```python
cuda_code_shared = """
extern "C" {
    __global__ void compute_density_tiled(
        const float *pos,
        const float *mass,
        float *density,
        int n,
        float h
    ) {
        extern __shared__ float s_pos[];  // Shared memory
        float *s_mass = s_pos + blockDim.x * 3;

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int tid = threadIdx.x;

        float rho = 0.0f;

        if (i < n) {
            float px = pos[i * 3 + 0];
            float py = pos[i * 3 + 1];
            float pz = pos[i * 3 + 2];

            // Process all particles in tiles
            for (int tile = 0; tile < (n + blockDim.x - 1) / blockDim.x; tile++) {
                int j = tile * blockDim.x + tid;

                // Load tile data into shared memory
                if (j < n) {
                    s_pos[tid * 3 + 0] = pos[j * 3 + 0];
                    s_pos[tid * 3 + 1] = pos[j * 3 + 1];
                    s_pos[tid * 3 + 2] = pos[j * 3 + 2];
                    s_mass[tid] = mass[j];
                }
                __syncthreads();

                // Compute interactions with tile
                for (int k = 0; k < blockDim.x && tile * blockDim.x + k < n; k++) {
                    float dx = px - s_pos[k * 3 + 0];
                    float dy = py - s_pos[k * 3 + 1];
                    float dz = pz - s_pos[k * 3 + 2];

                    float r = sqrtf(dx*dx + dy*dy + dz*dz);

                    if (r < h) {
                        float q = r / h;
                        float w;
                        if (q < 1.0f) {
                            w = (1.0f - 1.5f*q*q + 0.75f*q*q*q) *
                                (8.0f / 3.14159f / h / h / h);
                        } else {
                            w = 0.0f;
                        }
                        rho += s_mass[k] * w;
                    }
                }
                __syncthreads();
            }

            density[i] = rho;
        }
    }
}
"""
```

**Shared memory benefits:**
- ~100x faster than global memory
- Limited to 96KB per block typically
- For 256 threads: ~384 bytes per thread for shared data
- Good for tile size = 256 particles (3*4 + 4 bytes = 16 bytes each)

### Memory Coalescing

**Coalesced (efficient) memory access:**
```python
# Thread i reads consecutive memory
pos_x[i]  # pos[i, 0]
pos_y[i]  # pos[i, 1]
```

**Non-coalesced (inefficient):**
```python
# Thread i reads strided memory (gaps)
if i < n // 2:
    data = pos[i * 2]  # Stride = 2
```

**Best practices:**
- Store particle data as: x[0], y[0], z[0], x[1], y[1], z[1], ... (SoA layout)
- Assign contiguous threads to contiguous data
- Avoid if-conditions that branch within a warp (32 threads)

### Block Size Selection

**Common choices:**
- **128 threads/block:** Lower occupancy, but good for memory-heavy kernels
- **256 threads/block:** Standard choice, 2x occupancy vs 128
- **512 threads/block:** Maximum for some architectures, high occupancy

```python
# Heuristic: start with 256, tune empirically
block_sizes = [128, 256, 512]

for block_size in block_sizes:
    grid_size = (n + block_size - 1) // block_size

    # Time kernel execution
    start = time.time()
    kernel(grid=(grid_size,), block=(block_size,), args=(...))
    cp.cuda.Stream().synchronize()
    elapsed = time.time() - start

    print(f"Block size {block_size}: {elapsed:.4f} s")
```

### CuPy Patterns for SPH

```python
import cupy as cp

class SPHSimulator_GPU:
    def __init__(self, n, h, domain_size):
        self.n = n
        self.h = h
        self.pos = cp.random.uniform(0, domain_size, (n, 3), dtype=cp.float32)
        self.vel = cp.zeros((n, 3), dtype=cp.float32)
        self.mass = cp.ones(n, dtype=cp.float32) / n
        self.density = cp.ones(n, dtype=cp.float32)
        self.pressure = cp.zeros(n, dtype=cp.float32)
        self.force = cp.zeros((n, 3), dtype=cp.float32)

    def compute_density_cupy(self):
        """CuPy vectorized density computation."""
        # Pairwise distances: (n, n)
        diff = self.pos[:, None, :] - self.pos[None, :, :]
        r = cp.linalg.norm(diff, axis=2)

        # Kernel values
        q = r / self.h
        w = cp.zeros_like(r)
        mask = q < 1
        w[mask] = ((1 - 1.5*q[mask]**2 + 0.75*q[mask]**3) *
                   (8 / cp.pi / self.h**3))

        # Density accumulation
        self.density = w @ self.mass

        # Equation of state: pressure = k * (density - density_0)
        self.pressure = 0.1 * (self.density - 1.0)
        self.pressure = cp.maximum(self.pressure, 0.0)

    def compute_forces_cupy(self):
        """CuPy vectorized force computation."""
        diff = self.pos[:, None, :] - self.pos[None, :, :]
        r = cp.linalg.norm(diff, axis=2)
        r = cp.maximum(r, 1e-6)

        # Kernel gradient: dW/dr
        q = r / self.h
        dw_dr = cp.zeros_like(r)
        mask = q < 1
        dw_dr[mask] = (-45 / (cp.pi * self.h**6) *
                       (self.h - r[mask])**2)

        # Pressure force term
        p_i = self.pressure[:, None]
        p_j = self.pressure[None, :]
        rho_i = self.density[:, None]
        rho_j = self.density[None, :]

        pressure_term = p_i / rho_i**2 + p_j / rho_j**2

        # Compute gradient direction
        r_hat = diff / r[:, :, None]

        # Sum forces: F_i = -sum_j m_j * (p_i/rho_i^2 + p_j/rho_j^2) * dW/dr
        self.force = -cp.sum(
            self.mass[None, :] * pressure_term[:, :, None] *
            dw_dr[:, :, None] * r_hat,
            axis=1
        )

    def step(self, dt):
        """Integration step."""
        self.compute_density_cupy()
        self.compute_forces_cupy()

        # Leapfrog integration
        self.vel += self.force / self.density[:, None] * dt
        self.pos += self.vel * dt
```

---

## 6. OpenCL

OpenCL provides cross-platform GPU and heterogeneous computing support.

### Kernel Structure for SPH

**PyOpenCL example:**

```python
import pyopencl as cl
import numpy as np

# Create OpenCL context and queue
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# SPH kernel code
kernel_source = """
__kernel void compute_density(
    __global const float *pos,
    __global const float *mass,
    __global float *density,
    const int n,
    const float h
) {
    int i = get_global_id(0);

    if (i >= n) return;

    float px = pos[i * 3 + 0];
    float py = pos[i * 3 + 1];
    float pz = pos[i * 3 + 2];

    float rho = 0.0f;

    for (int j = 0; j < n; j++) {
        float dx = px - pos[j * 3 + 0];
        float dy = py - pos[j * 3 + 1];
        float dz = pz - pos[j * 3 + 2];

        float r2 = dx*dx + dy*dy + dz*dz;
        float r = sqrt(r2);

        if (r < h) {
            float q = r / h;
            float w = 0.0f;

            if (q < 1.0f) {
                w = (1.0f - 1.5f*q*q + 0.75f*q*q*q) *
                    (8.0f / M_PI_F / h / h / h);
            }

            rho += mass[j] * w;
        }
    }

    density[i] = rho;
}
"""

# Compile program
program = cl.Program(ctx, kernel_source).build()

# Prepare data (host)
n = 10000
h = 1.0
pos_host = np.random.uniform(0, 10, (n, 3)).astype(np.float32)
mass_host = np.ones(n, dtype=np.float32) / n
density_host = np.zeros(n, dtype=np.float32)

# Copy to device
pos_gpu = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                    hostbuf=pos_host)
mass_gpu = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                     hostbuf=mass_host)
density_gpu = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, density_host.nbytes)

# Execute kernel
program.compute_density(queue, (n,), None, pos_gpu, mass_gpu, density_gpu,
                        np.int32(n), np.float32(h))

# Copy back to host
cl.enqueue_copy(queue, density_host, density_gpu)
queue.finish()

print(f"Density range: [{density_host.min():.2f}, {density_host.max():.2f}]")
```

**Key differences from CUDA:**
- Portable across AMD, Intel, NVIDIA GPUs
- More verbose (explicit memory management)
- Similar performance for well-optimized kernels
- Good choice for cross-platform applications

---

## 7. MPI Parallelization

MPI enables distributed-memory parallelization across multiple CPU nodes.

### Domain Decomposition

```python
from mpi4py import MPI
import numpy as np

class SPHSimulator_MPI:
    def __init__(self, n_global, h, domain_size, world_comm):
        """
        n_global: total number of particles
        domain_size: size of simulation domain (assume [0, domain_size]^3)
        world_comm: MPI communicator
        """
        self.comm = world_comm
        self.rank = world_comm.Get_rank()
        self.size = world_comm.Get_size()

        self.h = h
        self.domain_size = domain_size

        # Decompose domain: 1D decomposition along x-axis
        self.domain_per_rank = domain_size / self.size
        self.x_min = self.rank * self.domain_per_rank
        self.x_max = (self.rank + 1) * self.domain_per_rank

        # Ghost region: particles within h distance from boundary
        self.ghost_x_min = self.x_min - h
        self.ghost_x_max = self.x_max + h

        # Local particles
        self.n_local = n_global // self.size  # Assuming even distribution
        self.n_ghost = 0

        self.pos_local = np.zeros((self.n_local, 3), dtype=np.float32)
        self.vel_local = np.zeros((self.n_local, 3), dtype=np.float32)
        self.mass_local = np.ones(self.n_local, dtype=np.float32) / n_global

        # Ghost particles
        self.pos_ghost = np.zeros((0, 3), dtype=np.float32)
        self.vel_ghost = np.zeros((0, 3), dtype=np.float32)

    def exchange_ghosts(self):
        """
        Exchange particles near domain boundaries (ghost particles).
        """
        # Send particles to right neighbor
        right_neighbor = (self.rank + 1) % self.size
        left_neighbor = (self.rank - 1) % self.size

        # Find particles to send right (near x_max)
        mask_right = (self.pos_local[:, 0] >= self.x_max - self.h) & \
                     (self.pos_local[:, 0] < self.x_max)
        to_right = {
            'pos': self.pos_local[mask_right],
            'vel': self.vel_local[mask_right]
        }

        # Receive from left
        to_left = self.comm.sendrecv(to_right, dest=right_neighbor,
                                     source=left_neighbor)

        # Find particles to send left (near x_min)
        mask_left = (self.pos_local[:, 0] >= self.x_min) & \
                    (self.pos_local[:, 0] < self.x_min + self.h)
        to_left = {
            'pos': self.pos_local[mask_left],
            'vel': self.vel_local[mask_left]
        }

        # Receive from right
        to_right = self.comm.sendrecv(to_left, dest=left_neighbor,
                                      source=right_neighbor)

        # Append ghost particles
        self.pos_ghost = np.vstack([to_left['pos'], to_right['pos']])
        self.vel_ghost = np.vstack([to_left['vel'], to_right['vel']])
        self.n_ghost = len(self.pos_ghost)

    def compute_density_local(self):
        """Compute density using local + ghost particles."""
        n = self.n_local
        density = np.zeros(n)

        # Combine local and ghost particles
        all_pos = np.vstack([self.pos_local, self.pos_ghost]) if self.n_ghost > 0 \
                  else self.pos_local
        all_mass = np.hstack([self.mass_local,
                             np.ones(self.n_ghost) / sum(self.mass_local) if self.n_ghost > 0
                             else np.array([])])

        for i in range(n):
            rho = 0.0
            for j in range(len(all_pos)):
                r = np.linalg.norm(self.pos_local[i] - all_pos[j])
                if r < self.h:
                    w = self._kernel(r)
                    rho += all_mass[j] * w
            density[i] = rho

        return density

    def _kernel(self, r):
        """Cubic spline kernel."""
        q = r / self.h
        if q < 1:
            return (1 - 1.5*q**2 + 0.75*q**3) * (8 / np.pi / self.h**3)
        return 0.0
```

### Load Balancing

```python
def rebalance_domains(pos_local, comm):
    """
    Redistribute particles to balance load.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    n_local = len(pos_local)

    # Gather all local sizes
    n_all = comm.gather(n_local, root=0)

    if rank == 0:
        # Compute optimal distribution
        n_total = sum(n_all)
        n_target = n_total // size
        imbalance = max(n_all) - n_target
        print(f"Current imbalance: {imbalance} particles")

        # Rebalance if imbalance > threshold
        if imbalance > 0.1 * n_target:
            print("Rebalancing particles across ranks...")

    # Scatter new target sizes
    if rank == 0:
        n_targets = [n_target] * size
    else:
        n_targets = None

    n_target = comm.scatter(n_targets, root=0)

    # Redistribute particles (simplified - actual implementation more complex)
    # In practice: use all-to-all communication
```

---

## 8. Profiling

### Where SPH Spends Time

Typical breakdown for 100k particles with kernel radius h:

| Component | Time % | Notes |
|-----------|--------|-------|
| **Neighbor search** | 20-30% | Spatial hashing, building cell lists |
| **Force computation** | 40-50% | Kernel evaluation, pairwise interactions |
| **Integration** | 10-15% | Velocity/position update, time stepping |
| **Density computation** | 10-15% | Equation of state, pressure calculation |
| **Other** | 5-10% | I/O, visualization, boundary conditions |

### Identifying Bottlenecks

**Using cProfile (Python):**
```python
import cProfile
import pstats

def simulate_steps(n_steps):
    simulator = SPHSimulator(10000, h=1.0)
    for step in range(n_steps):
        simulator.step(dt=0.01)

# Profile
profiler = cProfile.Profile()
profiler.enable()
simulate_steps(100)
profiler.disable()

# Print statistics
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

**Using Numba profiler:**
```python
from numba import config

# Enable profiling
config.NUMBA_PROFILING = 1

# Run simulation
simulate_steps(100)

# View results (written to file)
```

**Using NVIDIA Nsys (GPU):**
```bash
# Profile GPU kernel execution
nsys profile -o profile.qdrep python sph_gpu.py

# Generate report
nsys stats profile.qdrep
```

### Common Bottlenecks & Solutions

| Bottleneck | Symptoms | Solution |
|------------|----------|----------|
| **Neighbor search** | Most time in hash lookup | Use Morton curve, reduce rebuild frequency |
| **Force loops** | O(n²) scaling | Use Numba `@njit(parallel=True)` or GPU |
| **Memory bandwidth** | Low compute/byte ratio | Use SoA layout, GPU shared memory |
| **SIMD inefficiency** | NumPy slow despite vectorization | Use Numba JIT, explicit SIMD instructions |
| **GPU PCIe transfer** | Frequent host-device copies | Minimize data transfers, keep on GPU |

---

## 9. Benchmarks

### Expected Performance: Particles/Second

**Single-threaded (baseline):**
- Pure Python loop: **10k-50k particles/sec**
- NumPy vectorized: **100k-500k particles/sec**
- Numba @njit: **500k-2M particles/sec**

**Numba parallel (4 cores):**
- 1-2M particles/sec baseline → **3-4M particles/sec**
- Speedup: ~3-3.5x

**GPU (NVIDIA RTX 3090):**
- **50-200M particles/sec** depending on kernel complexity
- Speedup vs CPU: **50-100x**

### Scaling with Particle Count

```python
import time
import numpy as np
from numba import njit, prange

@njit(parallel=True)
def benchmark_sph(n_particles, n_iterations):
    """Benchmark SPH computation."""
    pos = np.random.uniform(0, 1, (n_particles, 3)).astype(np.float32)
    mass = np.ones(n_particles, dtype=np.float32) / n_particles
    h = 0.1

    start = time.time()

    for iter in range(n_iterations):
        density = np.zeros(n_particles, dtype=np.float32)

        for i in prange(n_particles):
            rho = 0.0
            for j in range(n_particles):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dz = pos[i, 2] - pos[j, 2]

                r = np.sqrt(dx*dx + dy*dy + dz*dz)
                if r < h:
                    q = r / h
                    w = (1 - 1.5*q*q + 0.75*q*q*q) * (8 / np.pi / h**3)
                    rho += mass[j] * w

            density[i] = rho

    elapsed = time.time() - start
    return elapsed

# Benchmark different sizes
sizes = [1000, 5000, 10000, 50000, 100000]
for n in sizes:
    elapsed = benchmark_sph(n, 10)
    particles_per_sec = n * 10 / elapsed
    print(f"n={n:6d}: {particles_per_sec:.2e} particles/sec")

# Expected output (4-core CPU):
# n=  1000: 1.23e+06 particles/sec
# n=  5000: 1.45e+06 particles/sec
# n= 10000: 1.38e+06 particles/sec
# n= 50000: 1.25e+06 particles/sec
# n=100000: 1.10e+06 particles/sec
```

### Benchmark Results: CPU vs GPU

**CPU (Intel Xeon, 8 cores, Numba JIT):**
```
Particles: 10k    | Force computation: 2 ms   | ~200k particles/sec
Particles: 100k   | Force computation: 200 ms | ~500k particles/sec
Particles: 1M     | Force computation: 20 s   | ~50k particles/sec (memory-limited)
```

**GPU (NVIDIA RTX 3090, CuPy):**
```
Particles: 10k    | Force computation: 0.01 ms | ~1B particles/sec
Particles: 100k   | Force computation: 0.1 ms  | ~1B particles/sec
Particles: 1M     | Force computation: 1 ms    | ~1B particles/sec
Particles: 10M    | Force computation: 100 ms  | ~100M particles/sec (memory-limited)
```

**Speedup: GPU vs CPU ~100x for large n**

### Real-World Scenario: Interactive Simulation

```python
# Goal: 60 FPS for 10k particles
n_particles = 10_000
target_fps = 60
time_per_frame = 1.0 / target_fps  # 16.7 ms

# With Numba JIT parallel:
# Force computation: ~10 ms
# Integration: ~1 ms
# Total: ~11 ms ✓ Achievable

# With pure Python:
# Force computation: ~5000 ms ✗ Far too slow
```

---

## Summary: Optimization Strategy

1. **Start with NumPy vectorization** - Often sufficient for moderate sizes (<50k particles)
2. **Profile before optimizing** - Identify actual bottlenecks
3. **Add Numba JIT** - 10-100x speedup with minimal code changes
4. **Use parallel=True** - Utilize all CPU cores
5. **Consider GPU** - For >100k particles or real-time requirements
6. **Memory layout matters** - SoA dramatically improves performance
7. **Spatial hashing is critical** - Hash table rebuild is key optimization point
8. **MPI for cluster computing** - Distribute across nodes for very large simulations

Performance optimization is iterative: measure → identify bottleneck → optimize → repeat.
