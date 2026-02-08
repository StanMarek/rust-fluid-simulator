use crate::shader_loader;

const DENSITY_WGSL: &str = include_str!("../shaders/density.wgsl");

pub struct DensityPipeline {
    pipeline: wgpu::ComputePipeline,
    particle_bind_group_layout: wgpu::BindGroupLayout,
    grid_bind_group_layout: wgpu::BindGroupLayout,
}

impl DensityPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader =
            shader_loader::load_shader_with_common(device, "density", DENSITY_WGSL);

        // Group 0: particle data + params
        let particle_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("density_particle_bgl"),
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
                    // binding 2: masses (read)
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
                    // binding 3: densities (rw)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
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

        // Group 1: grid data
        let grid_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("density_grid_bgl"),
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
            label: Some("density_layout"),
            bind_group_layouts: &[&particle_bind_group_layout, &grid_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("density_pipeline"),
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

    pub fn create_particle_bind_group(
        &self,
        device: &wgpu::Device,
        params: &wgpu::Buffer,
        positions: &wgpu::Buffer,
        masses: &wgpu::Buffer,
        densities: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("density_particle_bg"),
            layout: &self.particle_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: positions.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: masses.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: densities.as_entire_binding() },
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
            label: Some("density_grid_bg"),
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
        pass.dispatch_workgroups((particle_count + 63) / 64, 1, 1);
    }
}
