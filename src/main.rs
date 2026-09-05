//! teeter — entry point.
//!
//! Creates a winit event loop, constructs the app (Vulkan renderer + audio +
//! input), requests an immediate redraw, and spins the loop.

mod app;
mod audio;
mod engine;
mod input;
mod vulkan;

use winit::event_loop::{ControlFlow, EventLoop};

use app::App;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop
        .run_app(&mut app)
        .expect("event loop exited with error");
}
