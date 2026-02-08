use eframe::egui;
use eframe::NativeOptions;
use ui::FluidSimApp;

fn main() -> eframe::Result<()> {
    env_logger::init();
    log::info!("Starting Fluid Simulator (desktop)");

    let options = NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Fluid Simulator")
            .with_inner_size([1280.0, 720.0])
            .with_min_inner_size([640.0, 480.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Fluid Simulator",
        options,
        Box::new(|cc| Ok(Box::new(FluidSimApp::new(cc)))),
    )
}
