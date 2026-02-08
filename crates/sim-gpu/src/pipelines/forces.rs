use crate::shader_loader;

const FORCES_WGSL: &str = include_str!("../shaders/forces.wgsl");

pub struct ForcesPipeline {
    pipeline: wgpu::ComputePipeline,
    particle_bind_group_layout: wgpu::BindGroupLayout,
    grid_bind_group_layout: wgpu::BindGroupLayout,
}

impl ForcesPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader =
            shader_loader::load_shader_with_common(device, "forces", FORCES_WGSL);

        // Group 0: particle data + params (7 bindings)
        let particle_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("forces_particle_bgl"),
                entries: &[
                    // binding 0: params (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 1: positions (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 2: velocities (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 3: densities (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 4: pressures (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 5: masses (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 6: accelerations (rw)
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Group 1: grid data (reuse same layout as density)
        let grid_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("forces_grid_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("forces_layout"),
            bind_group_layouts: &[&particle_bind_group_layout, &grid_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("forces_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            particle_bind_group_layout,
            grid_bind_group_layout,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_particle_bind_group(
        &self,
        device: &wgpu::Device,
        params: &wgpu::Buffer,
        positions: &wgpu::Buffer,
        velocities: &wgpu::Buffer,
        densities: &wgpu::Buffer,
        pressures: &wgpu::Buffer,
        masses: &wgpu::Buffer,
        accelerations: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("forces_particle_bg"),
            layout: &self.particle_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: positions.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: velocities.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: densities.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pressures.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: masses.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: accelerations.as_entire_binding() },
            ],
        })
    }

    pub fn create_grid_bind_group(
        &self,
        device: &wgpu::Device,
        sorted_indices: &wgpu::Buffer,
        cell_start: &wgpu::Buffer,
        cell_count: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("forces_grid_bg"),
            layout: &self.grid_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sorted_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cell_start.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cell_count.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch<'a>(
        &'a self,
        pass: &mut wgpu::ComputePass<'a>,
        particle_bg: &'a wgpu::BindGroup,
        grid_bg: &'a wgpu::BindGroup,
        particle_count: u32,
    ) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, particle_bg, &[]);
        pass.set_bind_group(1, grid_bg, &[]);
        pass.dispatch_workgroups(particle_count.div_ceil(64), 1, 1);
    }
}
