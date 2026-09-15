//! teeter — entry point.
//!
//! Creates a winit event loop, constructs the app (Vulkan renderer + audio +
//! input), requests an immediate redraw, and spins the loop.

mod app;
mod audio;
mod engine;
mod fivecell;
mod input;
mod vulkan;

use winit::event_loop::{ControlFlow, EventLoop};

use app::App;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // --- CLI args: boot straight into a look. -------------------------------
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut shader = None;
    let mut preset = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--shader" => shader = it.next().and_then(|s| s.parse().ok()),
            "--preset" => preset = it.next().and_then(|s| s.parse().ok()),
            _ => {}
        }
    }

    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    app.boot_shader = shader;
    app.boot_preset = preset;
    event_loop
        .run_app(&mut app)
        .expect("event loop exited with error");
}
