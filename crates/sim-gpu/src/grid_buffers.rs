/// GPU storage buffers for the spatial hash grid.
pub struct GridBuffers {
    pub sorted_indices: wgpu::Buffer,
    pub cell_start: wgpu::Buffer,
    pub cell_count: wgpu::Buffer,
    particle_capacity: u32,
    table_size: u32,
}

impl GridBuffers {
    /// Create grid buffers with given particle capacity and table size.
    pub fn new(device: &wgpu::Device, particle_capacity: u32, table_size: u32) -> Self {
        let particle_cap = particle_capacity.max(1).next_power_of_two();
        Self {
            sorted_indices: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("grid_sorted_indices"),
                size: (particle_cap as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            cell_start: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("grid_cell_start"),
                size: (table_size as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            cell_count: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("grid_cell_count"),
                size: (table_size as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            particle_capacity: particle_cap,
            table_size,
        }
    }

    /// Ensure capacity for particle count.
    pub fn ensure_capacity(&mut self, device: &wgpu::Device, count: u32, table_size: u32) {
        if count <= self.particle_capacity && table_size <= self.table_size {
            return;
        }
        *self = Self::new(device, count, table_size);
    }

    /// Upload grid data from CPU grid export.
    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sorted_indices: &[u32],
        cell_starts: &[u32],
        cell_counts: &[u32],
    ) {
        let particle_count = sorted_indices.len() as u32;
        let table_size = cell_starts.len() as u32;
        self.ensure_capacity(device, particle_count, table_size);

        if !sorted_indices.is_empty() {
            queue.write_buffer(
                &self.sorted_indices,
                0,
                bytemuck::cast_slice(sorted_indices),
            );
        }
        queue.write_buffer(&self.cell_start, 0, bytemuck::cast_slice(cell_starts));
        queue.write_buffer(&self.cell_count, 0, bytemuck::cast_slice(cell_counts));
    }
}
