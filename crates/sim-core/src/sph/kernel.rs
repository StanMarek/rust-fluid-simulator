use common::Dimension;

/// Precomputed kernel parameters to avoid redundant calculation per call.
#[derive(Debug, Clone, Copy)]
pub struct KernelParams {
    /// Smoothing radius.
    pub h: f32,
    /// Precomputed normalization constant: 10 / (7 * pi * h^2) for 2D cubic spline.
    pub sigma: f32,
    /// Precomputed h^2.
    pub h_sq: f32,
    /// Precomputed kernel support radius squared: (2h)^2.
    pub support_sq: f32,
    /// Inverse of h: 1.0 / h.
    pub h_inv: f32,
}

impl KernelParams {
    /// Construct kernel parameters for a given smoothing radius (2D cubic spline).
    pub fn new(h: f32) -> Self {
        Self {
            h,
            sigma: 10.0 / (7.0 * std::f32::consts::PI * h * h),
            h_sq: h * h,
            support_sq: 4.0 * h * h,
            h_inv: 1.0 / h,
        }
    }
}

/// Trait for SPH smoothing kernels.
pub trait SmoothingKernel<D: Dimension> {
    /// Kernel value W(r, h).
    fn w(&self, r: f32, params: &KernelParams) -> f32;
    /// Kernel gradient grad_W(r_vec, |r|, h).
    fn grad_w(&self, r_vec: D::Vector, r: f32, params: &KernelParams) -> D::Vector;
    /// Kernel laplacian laplacian_W(r, h).
    fn laplacian_w(&self, r: f32, params: &KernelParams) -> f32;
}

/// Cubic spline kernel (M4), the default for SPH.
pub struct CubicSplineKernel;

impl SmoothingKernel<common::Dim2> for CubicSplineKernel {
    #[inline]
    fn w(&self, r: f32, params: &KernelParams) -> f32 {
        let q = r * params.h_inv;

        if q <= 1.0 {
            params.sigma * (1.0 - 1.5 * q * q + 0.75 * q * q * q)
        } else if q <= 2.0 {
            let t = 2.0 - q;
            params.sigma * 0.25 * t * t * t
        } else {
            0.0
        }
    }

    #[inline]
    fn grad_w(
        &self,
        r_vec: nalgebra::Vector2<f32>,
        r: f32,
        params: &KernelParams,
    ) -> nalgebra::Vector2<f32> {
        if r < 1e-10 {
            return nalgebra::Vector2::zeros();
        }

        let q = r * params.h_inv;
        let grad_q = r_vec * (1.0 / (r * params.h));

        let dw_dq = if q <= 1.0 {
            -3.0 * q + 2.25 * q * q
        } else if q <= 2.0 {
            let t = 2.0 - q;
            -0.75 * t * t
        } else {
            return nalgebra::Vector2::zeros();
        };

        grad_q * (params.sigma * dw_dq)
    }

    #[inline]
    fn laplacian_w(&self, r: f32, params: &KernelParams) -> f32 {
        let q = r * params.h_inv;

        let d2w_dq2 = if q <= 1.0 {
            -3.0 + 4.5 * q
        } else if q <= 2.0 {
            1.5 * (2.0 - q)
        } else {
            return 0.0;
        };

        // Simplified laplacian for 2D.
        params.sigma * d2w_dq2 / params.h_sq
    }
}
