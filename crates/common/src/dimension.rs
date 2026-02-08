use nalgebra::{Vector2, Vector3};
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

/// Trait abstracting over 2D and 3D dimensions.
/// All simulation code is generic over `D: Dimension` so that
/// adding 3D support does not require a rewrite.
pub trait Dimension: 'static + Send + Sync + Clone + Debug {
    type Vector: Copy
        + Default
        + Send
        + Sync
        + Debug
        + Add<Output = Self::Vector>
        + Sub<Output = Self::Vector>
        + Mul<f32, Output = Self::Vector>
        + AddAssign
        + SubAssign
        + MulAssign<f32>
        + PartialEq;

    const DIM: usize;

    /// Create a zero vector.
    fn zero() -> Self::Vector;

    /// Compute the squared magnitude of a vector.
    fn magnitude_sq(v: &Self::Vector) -> f32;

    /// Compute the magnitude of a vector.
    fn magnitude(v: &Self::Vector) -> f32 {
        Self::magnitude_sq(v).sqrt()
    }

    /// Normalize a vector. Returns zero vector if magnitude is zero.
    fn normalize(v: &Self::Vector) -> Self::Vector;

    /// Dot product of two vectors.
    fn dot(a: &Self::Vector, b: &Self::Vector) -> f32;

    /// Component-wise clamp each element between min and max vectors.
    fn clamp(v: &Self::Vector, min: &Self::Vector, max: &Self::Vector) -> Self::Vector;

    /// Get the i-th component of a vector.
    fn component(v: &Self::Vector, i: usize) -> f32;

    /// Set the i-th component of a vector.
    fn set_component(v: &mut Self::Vector, i: usize, val: f32);

    /// Create a vector from an array of floats (length DIM, extras ignored).
    fn from_slice(s: &[f32]) -> Self::Vector;
}

#[derive(Clone, Debug)]
pub struct Dim2;

#[derive(Clone, Debug)]
pub struct Dim3;

impl Dimension for Dim2 {
    type Vector = Vector2<f32>;
    const DIM: usize = 2;

    fn zero() -> Self::Vector {
        Vector2::zeros()
    }

    fn magnitude_sq(v: &Self::Vector) -> f32 {
        v.norm_squared()
    }

    fn normalize(v: &Self::Vector) -> Self::Vector {
        let m = v.norm();
        if m < 1e-10 {
            Self::zero()
        } else {
            v / m
        }
    }

    fn dot(a: &Self::Vector, b: &Self::Vector) -> f32 {
        a.dot(b)
    }

    fn clamp(v: &Self::Vector, min: &Self::Vector, max: &Self::Vector) -> Self::Vector {
        Vector2::new(v.x.clamp(min.x, max.x), v.y.clamp(min.y, max.y))
    }

    fn component(v: &Self::Vector, i: usize) -> f32 {
        v[i]
    }

    fn set_component(v: &mut Self::Vector, i: usize, val: f32) {
        v[i] = val;
    }

    fn from_slice(s: &[f32]) -> Self::Vector {
        Vector2::new(s[0], s[1])
    }
}

impl Dimension for Dim3 {
    type Vector = Vector3<f32>;
    const DIM: usize = 3;

    fn zero() -> Self::Vector {
        Vector3::zeros()
    }

    fn magnitude_sq(v: &Self::Vector) -> f32 {
        v.norm_squared()
    }

    fn normalize(v: &Self::Vector) -> Self::Vector {
        let m = v.norm();
        if m < 1e-10 {
            Self::zero()
        } else {
            v / m
        }
    }

    fn dot(a: &Self::Vector, b: &Self::Vector) -> f32 {
        a.dot(b)
    }

    fn clamp(v: &Self::Vector, min: &Self::Vector, max: &Self::Vector) -> Self::Vector {
        Vector3::new(
            v.x.clamp(min.x, max.x),
            v.y.clamp(min.y, max.y),
            v.z.clamp(min.z, max.z),
        )
    }

    fn component(v: &Self::Vector, i: usize) -> f32 {
        v[i]
    }

    fn set_component(v: &mut Self::Vector, i: usize, val: f32) {
        v[i] = val;
    }

    fn from_slice(s: &[f32]) -> Self::Vector {
        Vector3::new(s[0], s[1], s[2])
    }
}
