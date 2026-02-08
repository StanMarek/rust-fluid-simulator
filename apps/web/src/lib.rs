// WASM entry point for the fluid simulator.
// Build with: cd apps/web && wasm-pack build --target web

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), wasm_bindgen::JsValue> {
    eframe::WebLogger::init(log::LevelFilter::Debug).ok();

    let web_options = eframe::WebOptions::default();

    eframe::WebRunner::new()
        .start(
            canvas_id,
            web_options,
            Box::new(|cc| Ok(Box::new(ui::FluidSimApp::new(cc)))),
        )
        .await?;

    Ok(())
}
