/// WGSL shader loading with shared struct prepending.
/// Since WGSL has no #include, we concatenate shared definitions at load time.
const SIM_PARAMS_WGSL: &str = include_str!("shaders/sim_params.wgsl");
const GRID_COMMON_WGSL: &str = include_str!("shaders/grid_common.wgsl");

/// Load a shader with shared sim_params and grid_common prepended.
pub fn load_shader_with_common(
    device: &wgpu::Device,
    label: &str,
    main_source: &str,
) -> wgpu::ShaderModule {
    let full = format!("{}\n{}\n{}", SIM_PARAMS_WGSL, GRID_COMMON_WGSL, main_source);
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(full.into()),
    })
}

/// Load a shader with only sim_params prepended (no grid functions needed).
pub fn load_shader_with_params(
    device: &wgpu::Device,
    label: &str,
    main_source: &str,
) -> wgpu::ShaderModule {
    let full = format!("{}\n{}", SIM_PARAMS_WGSL, main_source);
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(full.into()),
    })
}
