use common::Dim2;
use sim_core::particle::ParticleStorage;

/// GPU buffer set for particle simulation data (SoA layout).
pub struct ParticleBuffers {
    pub positions: wgpu::Buffer,
    pub velocities: wgpu::Buffer,
    pub accelerations: wgpu::Buffer,
    pub densities: wgpu::Buffer,
    pub pressures: wgpu::Buffer,
    pub masses: wgpu::Buffer,
    pub capacity: u32,
}

impl ParticleBuffers {
    /// Create GPU buffers with the given initial capacity (in particles).
    pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
        let cap = capacity.max(1).next_power_of_two();
        Self {
            positions: Self::create_vec2_buffer(device, cap, "positions", true),
            velocities: Self::create_vec2_buffer(device, cap, "velocities", true),
            accelerations: Self::create_vec2_buffer(device, cap, "accelerations", false),
            densities: Self::create_f32_buffer(device, cap, "densities"),
            pressures: Self::create_f32_buffer(device, cap, "pressures"),
            masses: Self::create_f32_buffer(device, cap, "masses"),
            capacity: cap,
        }
    }

    fn create_vec2_buffer(
        device: &wgpu::Device,
        capacity: u32,
        label: &str,
        copy_src: bool,
    ) -> wgpu::Buffer {
        let mut usage =
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        if copy_src {
            usage |= wgpu::BufferUsages::COPY_SRC;
        }
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (capacity as u64) * 8, // 2 * f32 = 8 bytes
            usage,
            mapped_at_creation: false,
        })
    }

    fn create_f32_buffer(device: &wgpu::Device, capacity: u32, label: &str) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (capacity as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Ensure buffers can hold at least `count` particles. Recreates with power-of-two growth.
    pub fn ensure_capacity(&mut self, device: &wgpu::Device, count: u32) {
        if count <= self.capacity {
            return;
        }
        let new_cap = count.next_power_of_two();
        *self = Self::new(device, new_cap);
    }

    /// Upload particle data from CPU storage to GPU buffers.
    pub fn upload_from_storage(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        storage: &ParticleStorage<Dim2>,
    ) {
        let count = storage.len() as u32;
        self.ensure_capacity(device, count);

        // Convert nalgebra Vector2 -> [f32; 2] staging arrays
        let positions: Vec<[f32; 2]> = storage.positions.iter().map(|v| [v.x, v.y]).collect();
        queue.write_buffer(&self.positions, 0, bytemuck::cast_slice(&positions));

        let velocities: Vec<[f32; 2]> = storage.velocities.iter().map(|v| [v.x, v.y]).collect();
        queue.write_buffer(&self.velocities, 0, bytemuck::cast_slice(&velocities));

        let accelerations: Vec<[f32; 2]> =
            storage.accelerations.iter().map(|v| [v.x, v.y]).collect();
        queue.write_buffer(
            &self.accelerations,
            0,
            bytemuck::cast_slice(&accelerations),
        );

        queue.write_buffer(&self.densities, 0, bytemuck::cast_slice(&storage.densities));
        queue.write_buffer(&self.pressures, 0, bytemuck::cast_slice(&storage.pressures));
        queue.write_buffer(&self.masses, 0, bytemuck::cast_slice(&storage.masses));
    }

    /// Download positions from GPU to CPU.
    pub fn download_positions(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        count: u32,
    ) -> Vec<[f32; 2]> {
        Self::download_vec2_buffer(device, queue, &self.positions, count)
    }

    /// Download velocities from GPU to CPU.
    pub fn download_velocities(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        count: u32,
    ) -> Vec<[f32; 2]> {
        Self::download_vec2_buffer(device, queue, &self.velocities, count)
    }

    fn download_vec2_buffer(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: &wgpu::Buffer,
        count: u32,
    ) -> Vec<[f32; 2]> {
        let byte_size = (count as u64) * 8;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("download_staging"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("download_encoder"),
        });
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, byte_size);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<[f32; 2]> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }
}
