use common::Dimension;

/// A static obstacle that particles collide against.
pub enum Obstacle<D: Dimension> {
    Circle {
        center: D::Vector,
        radius: f32,
    },
    Box {
        min: D::Vector,
        max: D::Vector,
    },
}

impl<D: Dimension> Obstacle<D> {
    /// Signed distance from a point to the obstacle surface.
    /// Negative values mean the point is inside the obstacle.
    pub fn sdf(&self, point: &D::Vector) -> f32 {
        match self {
            Obstacle::Circle { center, radius } => {
                let diff = *point - *center;
                D::magnitude(&diff) - radius
            }
            Obstacle::Box { min, max } => {
                // For an axis-aligned box, SDF is the distance to the nearest surface.
                // Negative inside, positive outside.
                let mut max_neg = f32::NEG_INFINITY;
                for d in 0..D::DIM {
                    let p = D::component(point, d);
                    let lo = D::component(min, d);
                    let hi = D::component(max, d);
                    // Distance to each face (negative if inside)
                    let d_lo = lo - p; // negative if p > lo (inside)
                    let d_hi = p - hi; // negative if p < hi (inside)
                    let face_dist = d_lo.max(d_hi);
                    max_neg = max_neg.max(face_dist);
                }
                max_neg
            }
        }
    }

    /// Compute the outward surface normal at the nearest surface point.
    pub fn normal(&self, point: &D::Vector) -> D::Vector {
        match self {
            Obstacle::Circle { center, .. } => {
                let diff = *point - *center;
                D::normalize(&diff)
            }
            Obstacle::Box { min, max } => {
                // Normal points toward the nearest face.
                let mut best_d = 0;
                let mut best_dist = f32::NEG_INFINITY;
                let mut best_sign = 1.0_f32;
                for d in 0..D::DIM {
                    let p = D::component(point, d);
                    let lo = D::component(min, d);
                    let hi = D::component(max, d);
                    let d_lo = lo - p;
                    let d_hi = p - hi;
                    if d_lo > best_dist {
                        best_dist = d_lo;
                        best_d = d;
                        best_sign = -1.0;
                    }
                    if d_hi > best_dist {
                        best_dist = d_hi;
                        best_d = d;
                        best_sign = 1.0;
                    }
                }
                let mut n = D::zero();
                D::set_component(&mut n, best_d, best_sign);
                n
            }
        }
    }
}
