use common::Dimension;

use super::grid::SpatialHashGrid;

/// High-level neighbor query interface.
/// Provides filtered neighbor queries (e.g., within exact radius).
pub struct NeighborQuery;

impl NeighborQuery {
    /// Query neighbors within exact radius, filtering by distance.
    pub fn query_within_radius<D: Dimension>(
        grid: &SpatialHashGrid<D>,
        positions: &[D::Vector],
        query_pos: &D::Vector,
        radius: f32,
    ) -> Vec<usize> {
        let radius_sq = radius * radius;

        grid.query_neighbors_iter(query_pos)
            .filter(|&idx| {
                let diff = positions[idx] - *query_pos;
                D::magnitude_sq(&diff) <= radius_sq
            })
            .collect()
    }
}
