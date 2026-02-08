"""
Smoothing Kernel Functions for SPH Simulations

This module implements various kernel functions commonly used in SPH simulations.
Each kernel is provided in 2D and 3D versions with their gradient functions.

Kernel functions:
- Cubic Spline (standard, widely used)
- Quintic Spline (smoother, more computationally expensive)
- Wendland C2 (compactly supported, lower computational cost)
- Wendland C4 (smoother than C2)
- Gaussian (smooth but infinite support in theory)

Each kernel has:
- Kernel value function W(r, h)
- Gradient function ∇W(r_vec, h)
- Support radius (cutoff distance)
- Proper normalization constants
"""

import numpy as np
from typing import Callable, Tuple


# ============================================================================
# Cubic Spline Kernel
# ============================================================================

def cubic_spline_2d(r: np.ndarray, h: float) -> np.ndarray:
    """
    Cubic spline kernel in 2D.

    W(r, h) = σ_2d * {
        (2/3 - r²/h² + r³/(2h³))        if 0 ≤ r < h
        (1/6)(2 - r/h)³                 if h ≤ r < 2h
        0                               if r ≥ 2h
    }

    Normalization constant: σ_2d = 10/(7πh²)

    Args:
        r: Distance(s) between particles (scalar or array)
        h: Smoothing length

    Returns:
        Kernel value(s)
    """
    r = np.asarray(r)
    h = float(h)
    sigma = 10.0 / (7.0 * np.pi * h**2)

    result = np.zeros_like(r, dtype=float)

    # Region 1: 0 ≤ r < h
    mask1 = (r >= 0) & (r < h)
    q = r[mask1] / h
    result[mask1] = sigma * (2.0/3.0 - q**2 + 0.5*q**3)

    # Region 2: h ≤ r < 2h
    mask2 = (r >= h) & (r < 2*h)
    q = r[mask2] / h
    result[mask2] = sigma * (1.0/6.0) * (2.0 - q)**3

    return result


def cubic_spline_3d(r: np.ndarray, h: float) -> np.ndarray:
    """
    Cubic spline kernel in 3D.

    W(r, h) = σ_3d * {
        (2/3 - r²/h² + r³/(2h³))        if 0 ≤ r < h
        (1/6)(2 - r/h)³                 if h ≤ r < 2h
        0                               if r ≥ 2h
    }

    Normalization constant: σ_3d = 1/(πh³)

    Args:
        r: Distance(s) between particles
        h: Smoothing length

    Returns:
        Kernel value(s)
    """
    r = np.asarray(r)
    h = float(h)
    sigma = 1.0 / (np.pi * h**3)

    result = np.zeros_like(r, dtype=float)

    # Region 1: 0 ≤ r < h
    mask1 = (r >= 0) & (r < h)
    q = r[mask1] / h
    result[mask1] = sigma * (2.0/3.0 - q**2 + 0.5*q**3)

    # Region 2: h ≤ r < 2h
    mask2 = (r >= h) & (r < 2*h)
    q = r[mask2] / h
    result[mask2] = sigma * (1.0/6.0) * (2.0 - q)**3

    return result


def cubic_spline_gradient_2d(r_vec: np.ndarray, r: np.ndarray, h: float) -> np.ndarray:
    """
    Gradient of cubic spline kernel in 2D.

    ∇W(r) = dW/dr * r_vec/r

    dW/dr = σ_2d * {
        (-2r/h² + 3r²/(2h³))            if 0 ≤ r < h
        (-1/2)(2 - r/h)² / h            if h ≤ r < 2h
        0                               if r ≥ 2h
    }

    Args:
        r_vec: Vector from particle i to j (shape: (N, 2))
        r: Distance(s) between particles (shape: (N,))
        h: Smoothing length

    Returns:
        Kernel gradient vectors (shape: (N, 2))
    """
    r_vec = np.asarray(r_vec)
    r = np.asarray(r)
    h = float(h)
    sigma = 10.0 / (7.0 * np.pi * h**2)

    # Handle scalar vs array
    if r.ndim == 0:
        r = r[np.newaxis]
        r_vec = r_vec[np.newaxis, :]
        scalar_input = True
    else:
        scalar_input = False

    n_particles = r.shape[0]
    grad_result = np.zeros((n_particles, 2), dtype=float)

    # Avoid division by zero
    safe_r = np.where(r > 1e-10, r, 1e-10)

    # Region 1: 0 ≤ r < h
    mask1 = (r > 0) & (r < h)
    q = r[mask1] / h
    dw_dr = sigma * (-2.0*q/h + 3.0*q**2/(2.0*h))
    grad_result[mask1] = dw_dr[:, np.newaxis] * r_vec[mask1] / safe_r[mask1, np.newaxis]

    # Region 2: h ≤ r < 2h
    mask2 = (r >= h) & (r < 2*h)
    q = r[mask2] / h
    dw_dr = -sigma * 0.5 * (2.0 - q)**2 / h
    grad_result[mask2] = dw_dr[:, np.newaxis] * r_vec[mask2] / safe_r[mask2, np.newaxis]

    if scalar_input:
        grad_result = grad_result[0]

    return grad_result


def cubic_spline_gradient_3d(r_vec: np.ndarray, r: np.ndarray, h: float) -> np.ndarray:
    """
    Gradient of cubic spline kernel in 3D.

    ∇W(r) = dW/dr * r_vec/r

    Args:
        r_vec: Vector from particle i to j (shape: (N, 3))
        r: Distance(s) between particles (shape: (N,))
        h: Smoothing length

    Returns:
        Kernel gradient vectors (shape: (N, 3))
    """
    r_vec = np.asarray(r_vec)
    r = np.asarray(r)
    h = float(h)
    sigma = 1.0 / (np.pi * h**3)

    # Handle scalar vs array
    if r.ndim == 0:
        r = r[np.newaxis]
        r_vec = r_vec[np.newaxis, :]
        scalar_input = True
    else:
        scalar_input = False

    n_particles = r.shape[0]
    grad_result = np.zeros((n_particles, 3), dtype=float)

    # Avoid division by zero
    safe_r = np.where(r > 1e-10, r, 1e-10)

    # Region 1: 0 ≤ r < h
    mask1 = (r > 0) & (r < h)
    q = r[mask1] / h
    dw_dr = sigma * (-2.0*q/h + 3.0*q**2/(2.0*h))
    grad_result[mask1] = dw_dr[:, np.newaxis] * r_vec[mask1] / safe_r[mask1, np.newaxis]

    # Region 2: h ≤ r < 2h
    mask2 = (r >= h) & (r < 2*h)
    q = r[mask2] / h
    dw_dr = -sigma * 0.5 * (2.0 - q)**2 / h
    grad_result[mask2] = dw_dr[:, np.newaxis] * r_vec[mask2] / safe_r[mask2, np.newaxis]

    if scalar_input:
        grad_result = grad_result[0]

    return grad_result


# ============================================================================
# Quintic Spline Kernel (Wendland C4 alternative)
# ============================================================================

def quintic_spline_2d(r: np.ndarray, h: float) -> np.ndarray:
    """
    Quintic spline kernel in 2D.

    W(r, h) = σ_2d * (1 - r/(2h))⁴ * (1 + 2r/h)  for 0 ≤ r ≤ 2h

    Normalization constant: σ_2d = 7/(478πh²)

    Args:
        r: Distance(s) between particles
        h: Smoothing length

    Returns:
        Kernel value(s)
    """
    r = np.asarray(r)
    h = float(h)
    sigma = 7.0 / (478.0 * np.pi * h**2)

    result = np.zeros_like(r, dtype=float)

    mask = (r >= 0) & (r <= 2*h)
    q = r[mask] / h
    result[mask] = sigma * (1.0 - q/2.0)**4 * (1.0 + 2.0*q)

    return result


def quintic_spline_3d(r: np.ndarray, h: float) -> np.ndarray:
    """
    Quintic spline kernel in 3D.

    W(r, h) = σ_3d * (1 - r/(2h))⁴ * (1 + 2r/h)  for 0 ≤ r ≤ 2h

    Normalization constant: σ_3d = 1/(120πh³)

    Args:
        r: Distance(s) between particles
        h: Smoothing length

    Returns:
        Kernel value(s)
    """
    r = np.asarray(r)
    h = float(h)
    sigma = 1.0 / (120.0 * np.pi * h**3)

    result = np.zeros_like(r, dtype=float)

    mask = (r >= 0) & (r <= 2*h)
    q = r[mask] / h
    result[mask] = sigma * (1.0 - q/2.0)**4 * (1.0 + 2.0*q)

    return result


# ============================================================================
# Wendland C2 Kernel
# ============================================================================

def wendland_c2_2d(r: np.ndarray, h: float) -> np.ndarray:
    """
    Wendland C2 kernel in 2D (compactly supported).

    W(r, h) = σ_2d * (1 - r/(2h))³ * (1 + 3r/(2h))  for 0 ≤ r ≤ 2h

    Normalization constant: σ_2d = 7/(πh²) * 1/64

    Args:
        r: Distance(s) between particles
        h: Smoothing length

    Returns:
        Kernel value(s)
    """
    r = np.asarray(r)
    h = float(h)
    sigma = 7.0 / (64.0 * np.pi * h**2)

    result = np.zeros_like(r, dtype=float)

    mask = (r >= 0) & (r <= 2*h)
    q = r[mask] / h
    result[mask] = sigma * (1.0 - q/2.0)**3 * (1.0 + 1.5*q)

    return result


def wendland_c2_3d(r: np.ndarray, h: float) -> np.ndarray:
    """
    Wendland C2 kernel in 3D (compactly supported).

    W(r, h) = σ_3d * (1 - r/(2h))³ * (1 + 3r/(2h))  for 0 ≤ r ≤ 2h

    Normalization constant: σ_3d = 21/(2πh³) * 1/16

    Args:
        r: Distance(s) between particles
        h: Smoothing length

    Returns:
        Kernel value(s)
    """
    r = np.asarray(r)
    h = float(h)
    sigma = 21.0 / (16.0 * 2.0 * np.pi * h**3)

    result = np.zeros_like(r, dtype=float)

    mask = (r >= 0) & (r <= 2*h)
    q = r[mask] / h
    result[mask] = sigma * (1.0 - q/2.0)**3 * (1.0 + 1.5*q)

    return result


def wendland_c2_gradient_2d(r_vec: np.ndarray, r: np.ndarray, h: float) -> np.ndarray:
    """
    Gradient of Wendland C2 kernel in 2D.

    dW/dr = σ_2d * [-(3/2)(1 - q/2)² * (1 + 3q/2) - 3(1 - q/2)³ / 2h] / h

    Args:
        r_vec: Vector from particle i to j (shape: (N, 2))
        r: Distance(s) between particles (shape: (N,))
        h: Smoothing length

    Returns:
        Kernel gradient vectors (shape: (N, 2))
    """
    r_vec = np.asarray(r_vec)
    r = np.asarray(r)
    h = float(h)
    sigma = 7.0 / (64.0 * np.pi * h**2)

    # Handle scalar vs array
    if r.ndim == 0:
        r = r[np.newaxis]
        r_vec = r_vec[np.newaxis, :]
        scalar_input = True
    else:
        scalar_input = False

    n_particles = r.shape[0]
    grad_result = np.zeros((n_particles, 2), dtype=float)

    # Avoid division by zero
    safe_r = np.where(r > 1e-10, r, 1e-10)

    mask = (r > 0) & (r <= 2*h)
    q = r[mask] / h
    dw_dr = sigma * (-1.5*(1.0 - q/2.0)**2 * (1.0 + 1.5*q) - 1.5*(1.0 - q/2.0)**3) / h
    grad_result[mask] = dw_dr[:, np.newaxis] * r_vec[mask] / safe_r[mask, np.newaxis]

    if scalar_input:
        grad_result = grad_result[0]

    return grad_result


def wendland_c2_gradient_3d(r_vec: np.ndarray, r: np.ndarray, h: float) -> np.ndarray:
    """
    Gradient of Wendland C2 kernel in 3D.

    Args:
        r_vec: Vector from particle i to j (shape: (N, 3))
        r: Distance(s) between particles (shape: (N,))
        h: Smoothing length

    Returns:
        Kernel gradient vectors (shape: (N, 3))
    """
    r_vec = np.asarray(r_vec)
    r = np.asarray(r)
    h = float(h)
    sigma = 21.0 / (16.0 * 2.0 * np.pi * h**3)

    # Handle scalar vs array
    if r.ndim == 0:
        r = r[np.newaxis]
        r_vec = r_vec[np.newaxis, :]
        scalar_input = True
    else:
        scalar_input = False

    n_particles = r.shape[0]
    grad_result = np.zeros((n_particles, 3), dtype=float)

    # Avoid division by zero
    safe_r = np.where(r > 1e-10, r, 1e-10)

    mask = (r > 0) & (r <= 2*h)
    q = r[mask] / h
    dw_dr = sigma * (-1.5*(1.0 - q/2.0)**2 * (1.0 + 1.5*q) - 1.5*(1.0 - q/2.0)**3) / h
    grad_result[mask] = dw_dr[:, np.newaxis] * r_vec[mask] / safe_r[mask, np.newaxis]

    if scalar_input:
        grad_result = grad_result[0]

    return grad_result


# ============================================================================
# Wendland C4 Kernel
# ============================================================================

def wendland_c4_2d(r: np.ndarray, h: float) -> np.ndarray:
    """
    Wendland C4 kernel in 2D (smoother than C2).

    W(r, h) = σ_2d * (1 - r/(2h))⁵ * (1 + 5r/(2h) + 8r²/(4h²))

    Normalization constant: σ_2d = 9/(πh²) * 1/128

    Args:
        r: Distance(s) between particles
        h: Smoothing length

    Returns:
        Kernel value(s)
    """
    r = np.asarray(r)
    h = float(h)
    sigma = 9.0 / (128.0 * np.pi * h**2)

    result = np.zeros_like(r, dtype=float)

    mask = (r >= 0) & (r <= 2*h)
    q = r[mask] / h
    result[mask] = sigma * (1.0 - q/2.0)**5 * (1.0 + 2.5*q + 2.0*q**2)

    return result


def wendland_c4_3d(r: np.ndarray, h: float) -> np.ndarray:
    """
    Wendland C4 kernel in 3D (smoother than C2).

    W(r, h) = σ_3d * (1 - r/(2h))⁵ * (1 + 5r/(2h) + 8r²/(4h²))

    Normalization constant: σ_3d = 495/(32πh³) * 1/128

    Args:
        r: Distance(s) between particles
        h: Smoothing length

    Returns:
        Kernel value(s)
    """
    r = np.asarray(r)
    h = float(h)
    sigma = 495.0 / (32.0 * 128.0 * np.pi * h**3)

    result = np.zeros_like(r, dtype=float)

    mask = (r >= 0) & (r <= 2*h)
    q = r[mask] / h
    result[mask] = sigma * (1.0 - q/2.0)**5 * (1.0 + 2.5*q + 2.0*q**2)

    return result


def wendland_c4_gradient_2d(r_vec: np.ndarray, r: np.ndarray, h: float) -> np.ndarray:
    """
    Gradient of Wendland C4 kernel in 2D.

    Args:
        r_vec: Vector from particle i to j (shape: (N, 2))
        r: Distance(s) between particles (shape: (N,))
        h: Smoothing length

    Returns:
        Kernel gradient vectors (shape: (N, 2))
    """
    r_vec = np.asarray(r_vec)
    r = np.asarray(r)
    h = float(h)
    sigma = 9.0 / (128.0 * np.pi * h**2)

    # Handle scalar vs array
    if r.ndim == 0:
        r = r[np.newaxis]
        r_vec = r_vec[np.newaxis, :]
        scalar_input = True
    else:
        scalar_input = False

    n_particles = r.shape[0]
    grad_result = np.zeros((n_particles, 2), dtype=float)

    # Avoid division by zero
    safe_r = np.where(r > 1e-10, r, 1e-10)

    mask = (r > 0) & (r <= 2*h)
    q = r[mask] / h
    # dW/dr = d/dr[ sigma * (1 - q/2)^5 * (1 + 2.5*q + 2*q^2) ]
    dw_dr = sigma * (
        -2.5*(1.0 - q/2.0)**4 * (1.0 + 2.5*q + 2.0*q**2) / h +
        (1.0 - q/2.0)**5 * (2.5 + 4.0*q) / h
    )
    grad_result[mask] = dw_dr[:, np.newaxis] * r_vec[mask] / safe_r[mask, np.newaxis]

    if scalar_input:
        grad_result = grad_result[0]

    return grad_result


def wendland_c4_gradient_3d(r_vec: np.ndarray, r: np.ndarray, h: float) -> np.ndarray:
    """
    Gradient of Wendland C4 kernel in 3D.

    Args:
        r_vec: Vector from particle i to j (shape: (N, 3))
        r: Distance(s) between particles (shape: (N,))
        h: Smoothing length

    Returns:
        Kernel gradient vectors (shape: (N, 3))
    """
    r_vec = np.asarray(r_vec)
    r = np.asarray(r)
    h = float(h)
    sigma = 495.0 / (32.0 * 128.0 * np.pi * h**3)

    # Handle scalar vs array
    if r.ndim == 0:
        r = r[np.newaxis]
        r_vec = r_vec[np.newaxis, :]
        scalar_input = True
    else:
        scalar_input = False

    n_particles = r.shape[0]
    grad_result = np.zeros((n_particles, 3), dtype=float)

    # Avoid division by zero
    safe_r = np.where(r > 1e-10, r, 1e-10)

    mask = (r > 0) & (r <= 2*h)
    q = r[mask] / h
    dw_dr = sigma * (
        -2.5*(1.0 - q/2.0)**4 * (1.0 + 2.5*q + 2.0*q**2) / h +
        (1.0 - q/2.0)**5 * (2.5 + 4.0*q) / h
    )
    grad_result[mask] = dw_dr[:, np.newaxis] * r_vec[mask] / safe_r[mask, np.newaxis]

    if scalar_input:
        grad_result = grad_result[0]

    return grad_result


# ============================================================================
# Gaussian Kernel (infinite support in theory, truncated in practice)
# ============================================================================

def gaussian_2d(r: np.ndarray, h: float) -> np.ndarray:
    """
    Gaussian kernel in 2D.

    W(r, h) = σ_2d * exp(-(r/h)²)

    Normalization constant: σ_2d = 1/(πh²)

    Support radius: typically 3h (99.7% of mass)

    Args:
        r: Distance(s) between particles
        h: Smoothing length

    Returns:
        Kernel value(s)
    """
    r = np.asarray(r)
    h = float(h)
    sigma = 1.0 / (np.pi * h**2)

    q = r / h
    return sigma * np.exp(-q**2)


def gaussian_3d(r: np.ndarray, h: float) -> np.ndarray:
    """
    Gaussian kernel in 3D.

    W(r, h) = σ_3d * exp(-(r/h)²)

    Normalization constant: σ_3d = 1/(πh³)^(3/2)

    Support radius: typically 3h (99.7% of mass)

    Args:
        r: Distance(s) between particles
        h: Smoothing length

    Returns:
        Kernel value(s)
    """
    r = np.asarray(r)
    h = float(h)
    sigma = 1.0 / (np.pi**1.5 * h**3)

    q = r / h
    return sigma * np.exp(-q**2)


# ============================================================================
# Kernel Factory Function
# ============================================================================

def get_kernel(name: str, dim: int) -> Tuple[Callable, Callable, float]:
    """
    Factory function to retrieve kernel functions.

    Args:
        name: Kernel name ('cubic_spline', 'quintic_spline', 'wendland_c2',
              'wendland_c4', 'gaussian')
        dim: Dimension (2 or 3)

    Returns:
        Tuple of (kernel_func, gradient_func, support_radius)

    Example:
        kernel_fn, grad_fn, h_support = get_kernel('cubic_spline', dim=2)
        W = kernel_fn(r, h)
        grad_W = grad_fn(r_vec, r, h)
    """
    kernels = {
        'cubic_spline': {
            2: (cubic_spline_2d, cubic_spline_gradient_2d, 2.0),
            3: (cubic_spline_3d, cubic_spline_gradient_3d, 2.0),
        },
        'quintic_spline': {
            2: (quintic_spline_2d, None, 2.0),
            3: (quintic_spline_3d, None, 2.0),
        },
        'wendland_c2': {
            2: (wendland_c2_2d, wendland_c2_gradient_2d, 2.0),
            3: (wendland_c2_3d, wendland_c2_gradient_3d, 2.0),
        },
        'wendland_c4': {
            2: (wendland_c4_2d, wendland_c4_gradient_2d, 2.0),
            3: (wendland_c4_3d, wendland_c4_gradient_3d, 2.0),
        },
        'gaussian': {
            2: (gaussian_2d, None, 3.0),  # Truncated at 3h
            3: (gaussian_3d, None, 3.0),
        },
    }

    if name not in kernels:
        raise ValueError(f"Unknown kernel: {name}. Available: {list(kernels.keys())}")

    if dim not in kernels[name]:
        raise ValueError(f"Kernel {name} not available in {dim}D")

    return kernels[name][dim]


if __name__ == "__main__":
    """Test kernel functions"""
    import matplotlib.pyplot as plt

    h = 0.1
    r = np.linspace(0, 0.3, 100)

    # Test cubic spline
    W_cubic_2d = cubic_spline_2d(r, h)
    W_cubic_3d = cubic_spline_3d(r, h)

    # Test Wendland C2
    W_wendland_c2_2d = wendland_c2_2d(r, h)
    W_wendland_c2_3d = wendland_c2_3d(r, h)

    print("Kernel functions tested successfully")
    print(f"Cubic spline 2D at r=0: {cubic_spline_2d(0, h):.6f}")
    print(f"Cubic spline 3D at r=0: {cubic_spline_3d(0, h):.6f}")
    print(f"Wendland C2 2D at r=0: {wendland_c2_2d(0, h):.6f}")
    print(f"Wendland C2 3D at r=0: {wendland_c2_3d(0, h):.6f}")
