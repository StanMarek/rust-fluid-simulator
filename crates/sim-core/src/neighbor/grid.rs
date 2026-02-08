use std::marker::PhantomData;

use common::Dimension;

/// Flat sorted-array spatial hash grid for allocation-free neighbor search.
/// Particles are sorted by cell hash, and an offset table maps each hash
/// to a contiguous slice within the sorted array.
pub struct SpatialHashGrid<D: Dimension> {
    /// Sorted (hash, particle_index) entries.
    entries: Vec<(u32, u32)>,
    /// Offset table: `offsets[hash] = (start, count)` into `entries`.
    offsets: Vec<(u32, u32)>,
    /// Cell size (= 2 * smoothing_radius for full kernel support).
    cell_size: f32,
    /// Number of hash buckets.
    table_size: u32,
    _marker: PhantomData<D>,
}

/// Large primes for spatial hashing.
const HASH_PRIME_X: u32 = 73856093;
const HASH_PRIME_Y: u32 = 19349663;
/// Future: used for 3D spatial hashing.
#[allow(dead_code)]
const HASH_PRIME_Z: u32 = 83492791;

impl<D: Dimension> SpatialHashGrid<D> {
    /// Create a new spatial hash grid with the given cell size.
    pub fn new(cell_size: f32) -> Self {
        let table_size: u32 = 262144; // 2^18
        Self {
            entries: Vec::new(),
            offsets: vec![(0, 0); table_size as usize],
            cell_size,
            table_size,
            _marker: PhantomData,
        }
    }

    /// Update cell size (e.g., if smoothing radius changes).
    pub fn set_cell_size(&mut self, cell_size: f32) {
        self.cell_size = cell_size;
    }

    /// Clear the grid and re-insert all particles using counting sort.
    pub fn build(&mut self, positions: &[D::Vector]) {
        let n = positions.len();
        let ts = self.table_size as usize;

        // 1. Resize entries buffer (reuse allocation)
        self.entries.resize(n, (0, 0));

        // 2. Count particles per hash bucket
        for slot in self.offsets.iter_mut() {
            *slot = (0, 0);
        }
        for pos in positions.iter() {
            let h = self.hash_position(pos);
            self.offsets[h as usize].1 += 1;
        }

        // 3. Prefix sum to compute start offsets
        let mut running = 0u32;
        for i in 0..ts {
            let count = self.offsets[i].1;
            self.offsets[i].0 = running;
            self.offsets[i].1 = 0; // Reset count, will use as insertion cursor
            running += count;
        }

        // 4. Place particles into sorted positions using offset cursors
        for (idx, pos) in positions.iter().enumerate() {
            let h = self.hash_position(pos);
            let slot = &mut self.offsets[h as usize];
            let dest = slot.0 + slot.1;
            self.entries[dest as usize] = (h, idx as u32);
            slot.1 += 1;
        }
    }

    /// Iterate all neighbor candidate particle indices for a given position.
    /// Visits the 9 adjacent cells (2D) or 27 cells (3D) and yields indices
    /// from each cell's slice in the sorted entry array. Zero heap allocations.
    #[inline]
    pub fn query_neighbors_iter(&self, pos: &D::Vector) -> NeighborIter<'_> {
        let cx = (D::component(pos, 0) / self.cell_size).floor() as i32;
        let cy = (D::component(pos, 1) / self.cell_size).floor() as i32;

        // Precompute all 9 cell hashes
        let mut cell_hashes = [0u32; 9];
        let mut idx = 0;
        for dx in -1i32..=1 {
            for dy in -1i32..=1 {
                cell_hashes[idx] = self.hash_coords_2d(cx + dx, cy + dy);
                idx += 1;
            }
        }

        NeighborIter {
            entries: &self.entries,
            offsets: &self.offsets,
            cell_hashes,
            cell_count: 9,
            current_cell: 0,
            current_pos: 0,
            current_end: 0,
        }
    }

    /// Hash 2D integer cell coordinates to a table index.
    #[inline]
    fn hash_coords_2d(&self, cx: i32, cy: i32) -> u32 {
        let hx = (cx as u32).wrapping_mul(HASH_PRIME_X);
        let hy = (cy as u32).wrapping_mul(HASH_PRIME_Y);
        (hx ^ hy) % self.table_size
    }

    /// Export grid data for GPU upload.
    /// Returns (sorted_indices, cell_starts, cell_counts) as flat arrays.
    pub fn export_for_gpu(&self) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let sorted_indices: Vec<u32> = self.entries.iter().map(|(_, idx)| *idx).collect();
        let cell_starts: Vec<u32> = self.offsets.iter().map(|(start, _)| *start).collect();
        let cell_counts: Vec<u32> = self.offsets.iter().map(|(_, count)| *count).collect();
        (sorted_indices, cell_starts, cell_counts)
    }

    /// Get the table size (number of hash buckets).
    pub fn table_size(&self) -> u32 {
        self.table_size
    }

    /// Hash a position directly.
    #[inline]
    fn hash_position(&self, pos: &D::Vector) -> u32 {
        let cx = (D::component(pos, 0) / self.cell_size).floor() as i32;
        let cy = (D::component(pos, 1) / self.cell_size).floor() as i32;
        self.hash_coords_2d(cx, cy)
    }
}

/// Zero-allocation iterator over neighbor particle indices.
pub struct NeighborIter<'a> {
    entries: &'a [(u32, u32)],
    offsets: &'a [(u32, u32)],
    cell_hashes: [u32; 9],
    cell_count: usize,
    current_cell: usize,
    current_pos: u32,
    current_end: u32,
}

impl<'a> Iterator for NeighborIter<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        loop {
            // Yield from current cell slice
            if self.current_pos < self.current_end {
                let entry = self.entries[self.current_pos as usize];
                self.current_pos += 1;
                return Some(entry.1 as usize);
            }

            // Advance to next cell
            if self.current_cell >= self.cell_count {
                return None;
            }

            let hash = self.cell_hashes[self.current_cell] as usize;
            self.current_cell += 1;

            let (start, count) = self.offsets[hash];
            self.current_pos = start;
            self.current_end = start + count;
        }
    }
}
