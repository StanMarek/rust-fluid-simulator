use std::collections::HashMap;

use common::Dimension;

/// Uniform spatial hash grid for neighbor search.
/// Cell size equals the smoothing radius h.
pub struct SpatialHashGrid<D: Dimension> {
    /// Map from cell hash to list of particle indices.
    cells: HashMap<u64, Vec<usize>>,
    /// Cell size (= smoothing radius h).
    cell_size: f32,
    /// Table size for hash modulo.
    table_size: u64,
    _phantom: std::marker::PhantomData<D>,
}

impl<D: Dimension> SpatialHashGrid<D> {
    pub fn new(cell_size: f32) -> Self {
        Self {
            cells: HashMap::new(),
            cell_size,
            table_size: 262144, // 2^18, a reasonable default
            _phantom: std::marker::PhantomData,
        }
    }

    /// Clear the grid and re-insert all particles.
    pub fn build(&mut self, positions: &[D::Vector]) {
        self.cells.clear();

        for (idx, pos) in positions.iter().enumerate() {
            let hash = self.hash_position(pos);
            self.cells.entry(hash).or_default().push(idx);
        }
    }

    /// Update cell size (e.g., if smoothing radius changes).
    pub fn set_cell_size(&mut self, cell_size: f32) {
        self.cell_size = cell_size;
    }

    /// Query all neighbor particle indices within radius of a position.
    /// Checks the current cell and all adjacent cells.
    pub fn query_neighbors(&self, pos: &D::Vector, _radius: f32) -> Vec<usize> {
        let mut neighbors = Vec::new();
        let cell_coords = self.cell_coords(pos);

        // Check all adjacent cells (3^DIM cells)
        self.visit_adjacent_cells(&cell_coords, &mut |hash| {
            if let Some(indices) = self.cells.get(&hash) {
                neighbors.extend_from_slice(indices);
            }
        });

        neighbors
    }

    /// Compute integer cell coordinates for a position.
    fn cell_coords(&self, pos: &D::Vector) -> Vec<i64> {
        let mut coords = Vec::with_capacity(D::DIM);
        for i in 0..D::DIM {
            let c = D::component(pos, i);
            coords.push((c / self.cell_size).floor() as i64);
        }
        coords
    }

    /// Hash cell coordinates to a table index.
    fn hash_coords(&self, coords: &[i64]) -> u64 {
        // Spatial hash using large primes
        let primes = [73856093u64, 19349663u64, 83492791u64];
        let mut hash: u64 = 0;
        for (i, &c) in coords.iter().enumerate() {
            hash ^= (c as u64).wrapping_mul(primes[i % primes.len()]);
        }
        hash % self.table_size
    }

    /// Hash a position directly.
    fn hash_position(&self, pos: &D::Vector) -> u64 {
        let coords = self.cell_coords(pos);
        self.hash_coords(&coords)
    }

    /// Visit all adjacent cells (including self) and call the callback with each hash.
    fn visit_adjacent_cells(&self, center: &[i64], callback: &mut dyn FnMut(u64)) {
        match D::DIM {
            2 => {
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        let coords = vec![center[0] + dx, center[1] + dy];
                        callback(self.hash_coords(&coords));
                    }
                }
            }
            3 => {
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            let coords = vec![center[0] + dx, center[1] + dy, center[2] + dz];
                            callback(self.hash_coords(&coords));
                        }
                    }
                }
            }
            _ => {
                // Fallback: only check self cell
                callback(self.hash_coords(center));
            }
        }
    }
}
