use common::Dimension;

/// Trait for SPH smoothing kernels.
pub trait SmoothingKernel<D: Dimension> {
    /// Kernel value W(r, h).
    fn w(&self, r: f32, h: f32) -> f32;
    /// Kernel gradient ∇W(r_vec, |r|, h).
    fn grad_w(&self, r_vec: D::Vector, r: f32, h: f32) -> D::Vector;
    /// Kernel laplacian ∇²W(r, h) — used for viscosity.
    fn laplacian_w(&self, r: f32, h: f32) -> f32;
}

/// Cubic spline kernel (M4), the default for SPH.
pub struct CubicSplineKernel;

impl SmoothingKernel<common::Dim2> for CubicSplineKernel {
    fn w(&self, r: f32, h: f32) -> f32 {
        let q = r / h;
        let sigma = 10.0 / (7.0 * std::f32::consts::PI * h * h);

        if q < 0.0 {
            0.0
        } else if q <= 1.0 {
            sigma * (1.0 - 1.5 * q * q + 0.75 * q * q * q)
        } else if q <= 2.0 {
            let t = 2.0 - q;
            sigma * 0.25 * t * t * t
        } else {
            0.0
        }
    }

    fn grad_w(&self, r_vec: nalgebra::Vector2<f32>, r: f32, h: f32) -> nalgebra::Vector2<f32> {
        if r < 1e-10 {
            return nalgebra::Vector2::zeros();
        }

        let q = r / h;
        let sigma = 10.0 / (7.0 * std::f32::consts::PI * h * h);
        let grad_q = r_vec / (r * h);

        let dw_dq = if q <= 1.0 {
            -3.0 * q + 2.25 * q * q
        } else if q <= 2.0 {
            let t = 2.0 - q;
            -0.75 * t * t
        } else {
            return nalgebra::Vector2::zeros();
        };

        grad_q * (sigma * dw_dq)
    }

    fn laplacian_w(&self, r: f32, h: f32) -> f32 {
        let q = r / h;
        let sigma = 10.0 / (7.0 * std::f32::consts::PI * h * h);

        let d2w_dq2 = if q <= 1.0 {
            -3.0 + 4.5 * q
        } else if q <= 2.0 {
            1.5 * (2.0 - q)
        } else {
            return 0.0;
        };

        // Simplified laplacian for 2D
        sigma * d2w_dq2 / (h * h)
    }
}
