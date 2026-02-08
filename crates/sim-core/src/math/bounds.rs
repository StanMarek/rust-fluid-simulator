use common::Dimension;

/// Axis-Aligned Bounding Box generic over dimension.
#[derive(Debug, Clone)]
pub struct AABB<D: Dimension> {
    pub min: D::Vector,
    pub max: D::Vector,
}

impl<D: Dimension> AABB<D> {
    pub fn new(min: D::Vector, max: D::Vector) -> Self {
        Self { min, max }
    }

    /// Check if a point is inside the AABB.
    pub fn contains(&self, point: &D::Vector) -> bool {
        for i in 0..D::DIM {
            let c = D::component(point, i);
            if c < D::component(&self.min, i) || c > D::component(&self.max, i) {
                return false;
            }
        }
        true
    }

    /// Clamp a point to lie within the AABB.
    pub fn clamp(&self, point: &D::Vector) -> D::Vector {
        D::clamp(point, &self.min, &self.max)
    }

    /// Get the size of the AABB along each dimension.
    pub fn size(&self) -> D::Vector {
        self.max - self.min
    }
}
