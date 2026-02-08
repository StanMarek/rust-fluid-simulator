use common::Dimension;

/// Compute the distance between two vectors.
pub fn distance<D: Dimension>(a: &D::Vector, b: &D::Vector) -> f32 {
    let diff = *a - *b;
    D::magnitude(&diff)
}

/// Compute the squared distance between two vectors.
pub fn distance_sq<D: Dimension>(a: &D::Vector, b: &D::Vector) -> f32 {
    let diff = *a - *b;
    D::magnitude_sq(&diff)
}

/// Scale a vector by a scalar.
pub fn scale<D: Dimension>(v: &D::Vector, s: f32) -> D::Vector {
    *v * s
}

/// Linear interpolation between two vectors.
pub fn lerp<D: Dimension>(a: &D::Vector, b: &D::Vector, t: f32) -> D::Vector {
    *a * (1.0 - t) + *b * t
}
