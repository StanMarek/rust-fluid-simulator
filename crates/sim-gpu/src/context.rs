/// GPU context: wgpu device and queue setup.
/// Shared between sim-gpu compute and renderer.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
    /// Create a new GPU context, requesting a device with compute capabilities.
    pub async fn new(instance: &wgpu::Instance, surface: Option<&wgpu::Surface<'_>>) -> Option<Self> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: surface,
                force_fallback_adapter: false,
            })
            .await?;

        let adapter_info = adapter.get_info();
        log::info!("GPU adapter: {:?}", adapter_info);

        let (device, queue) = match adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Fluid Sim GPU"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
        {
            Ok(pair) => pair,
            Err(e) => {
                log::warn!("Failed to request GPU device: {}", e);
                return None;
            }
        };

        Some(Self {
            device,
            queue,
            adapter_info,
        })
    }
}
