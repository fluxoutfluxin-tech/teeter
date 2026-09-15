//! Vulkan renderer.
//!
//! A feedback-loop visualizer ("teeter"):
//!   1. Draw a fullscreen triangle with `warp.frag`. It samples the previous
//!      frame stored in a `feedback` texture, applies the MilkDrop-style
//!      audio-reactive UV warp + hue rotation, and writes to the swapchain
//!      image (what you see on screen).
//!   2. Blit the swapchain image back into the `feedback` texture, so the
//!      next frame wraps the previous one again.
//!
//! One graphics pipeline + a device-local feedback texture + one UBO + one
//! blit = a tight, simple loop.
//!
//! The Win32 surface is created manually from the winit raw window handle
//! (the window is always a native Win32 window on the ROG Ally), so we do
//! not depend on `ash-window`.

use std::ffi::CString;
use std::sync::Arc;

use ash::vk;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use winit::window::Window;

use crate::audio::AudioFrame;
use crate::engine::EngineState;
use crate::fivecell::{self, DSPUniforms};
use crate::input::InputState;

// SPIR-V blobs (produced by the Vulkan SDK glslc, committed at build time).
const VERT_SPV: &[u8] = include_bytes!("../../shaders/spv/composite.vert.spv");
const WARP_SPV: &[u8] = include_bytes!("../../shaders/spv/warp.frag.spv");
const WARP2_SPV: &[u8] = include_bytes!("../../shaders/spv/warp2.frag.spv");
const WARP3_SPV: &[u8] = include_bytes!("../../shaders/spv/warp3.frag.spv");
const WARP4_SPV: &[u8] = include_bytes!("../../shaders/spv/warp4.frag.spv");
const WARP5_SPV: &[u8] = include_bytes!("../../shaders/spv/warp5.frag.spv");
const WARP6_SPV: &[u8] = include_bytes!("../../shaders/spv/warp6.frag.spv");
const WARP7_SPV: &[u8] = include_bytes!("../../shaders/spv/warp7.frag.spv");
const WARP8_SPV: &[u8] = include_bytes!("../../shaders/spv/warp8.frag.spv");
const WARP9_SPV: &[u8] = include_bytes!("../../shaders/spv/warp9.frag.spv");
const WARP10_SPV: &[u8] = include_bytes!("../../shaders/spv/warp10.frag.spv");
const WARP11_SPV: &[u8] = include_bytes!("../../shaders/spv/warp11.frag.spv");
const WARP12_SPV: &[u8] = include_bytes!("../../shaders/spv/warp12.frag.spv");
const WARP13_SPV: &[u8] = include_bytes!("../../shaders/spv/warp13.frag.spv");
const WARP14_SPV: &[u8] = include_bytes!("../../shaders/spv/warp14.frag.spv");
const WARP15_SPV: &[u8] = include_bytes!("../../shaders/spv/warp15.frag.spv");
const WARP16_SPV: &[u8] = include_bytes!("../../shaders/spv/warp16.frag.spv");
const WARP17_SPV: &[u8] = include_bytes!("../../shaders/spv/warp17.frag.spv");
const WARP18_SPV: &[u8] = include_bytes!("../../shaders/spv/warp18.frag.spv");
const WARP19_SPV: &[u8] = include_bytes!("../../shaders/spv/warp19.frag.spv");
const WARP20_SPV: &[u8] = include_bytes!("../../shaders/spv/warp20.frag.spv");

// 5-cell overlay shaders (own pipeline with a separate DSPUniforms UBO).
const CELL_VERT_SPV: &[u8] = include_bytes!("../../shaders/spv/5cell_projection.vert.spv");
const CELL_FRAG_SPV: &[u8] = include_bytes!("../../shaders/spv/5cell_render.frag.spv");

/// Display metadata for the numbered warp shaders, indexed same as the
/// pipeline array (warp.frag = 0 .. warp20.frag = 19). The first 13 classic
/// shaders predate named presets, so they return `None` and the UI falls back
/// to a plain "warp N" tag; the newer high-polygon geometric warps get names
/// and a visual genre for the title bar.
pub fn warp_meta(idx: usize) -> Option<(&'static str, &'static str)> {
    match idx {
        13 => Some(("kaleido mandala", "sacred geometry")),
        14 => Some(("hex lattice", "geometric")),
        15 => Some(("crystal pillars", "crystalline")),
        16 => Some(("interference moiré", "optical")),
        17 => Some(("star mandala", "sacred geometry")),
        18 => Some(("infinite staircase", "surreal")),
        19 => Some(("lowpoly shards", "chromatic")),
        _ => None,
    }
}

/// Number of frames we keep in flight (command buffers + sync objects).
/// This is deliberately independent of the swapchain image count so that
/// recreating the swapchain (windowed <-> fullscreen) never leaves the
/// per-frame arrays at a stale size.
const MAX_FRAMES_IN_FLIGHT: usize = 2;

/// Must match the `FrameUniforms` block in the shaders (13 floats = 52 bytes)
/// plus the extended audio shape features appended AFTER `aspect` so old warps
/// that only declare the first 13 floats stay layout-compatible.
#[repr(C)]
#[derive(Clone, Copy)]
struct UniformData {
    i_time: f32,
    bass: f32,
    mid: f32,
    treble: f32,
    beat: f32,
    zoom: f32,
    warp_x: f32,
    warp_y: f32,
    rotate: f32,
    dissolve: f32,
    palette: f32,
    preset_seed: f32,
    aspect: f32,
    // Extended audio shape descriptors (new warps read these; old ones don't).
    sub_bass: f32,
    centroid: f32,
    crest: f32,
    flux: f32,
    rolloff: f32,
}

fn cstring(s: &str) -> CString {
    CString::new(s).expect("CString")
}

pub struct Renderer {
    _entry: ash::Entry,
    _instance: ash::Instance,
    surface: vk::SurfaceKHR,
    khr_surface: ash::extensions::khr::Surface,
    physical_device: vk::PhysicalDevice,

    swapchain_khr: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_views: Vec<vk::ImageView>,
    swapchain_fbs: Vec<vk::Framebuffer>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,

    device: ash::Device,
    graphics_queue: vk::Queue,

    // feedback texture ("previous frame")
    feedback_image: vk::Image,
    feedback_memory: vk::DeviceMemory,
    feedback_view: vk::ImageView,
    feedback_sampler: vk::Sampler,
    // frozen pre-switch copy for the warp crossfade (texOld, binding 2)
    feedback_old_image: vk::Image,
    feedback_old_memory: vk::DeviceMemory,
    feedback_old_view: vk::ImageView,
    /// Crossfade dissolve override: 0 normally, spiked to 1.0 on a warp switch
    /// and decayed each frame so switching melts instead of cutting.
    crossfade: f32,
    /// Copy the current feedback frame into `feedback_old` on the next render
    /// (set when the warp changes, so texOld captures the pre-switch look).
    freeze_pending: bool,

    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    /// One pipeline per selectable warp shader.
    pipelines: Vec<vk::Pipeline>,
    shader_index: usize,
    desc_set: vk::DescriptorSet,
    uniform_memory: vk::DeviceMemory,
    uniform_size: vk::DeviceSize,

    // 5-cell wireframe overlay (own pipeline, descriptor set + buffers).
    fivecell_pipeline: vk::Pipeline,
    fivecell_pipeline_layout: vk::PipelineLayout,
    fivecell_desc_set: vk::DescriptorSet,
    fivecell_vertex_buffer: vk::Buffer,
    fivecell_vertex_memory: vk::DeviceMemory,
    fivecell_index_buffer: vk::Buffer,
    fivecell_index_memory: vk::DeviceMemory,
    fivecell_uniform_buffer: vk::Buffer,
    fivecell_uniform_memory: vk::DeviceMemory,
    fivecell_uniform_size: vk::DeviceSize,
    fivecell_active: bool,

    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available: Vec<vk::Semaphore>,
    render_finished: Vec<vk::Semaphore>,
    in_flight: Vec<vk::Fence>,
    frame_index: usize,
}

fn first_error(r: impl std::fmt::Display) -> String {
    r.to_string()
}

impl Renderer {
    pub fn new(window: Arc<Window>) -> Result<Self, String> {
        let entry = unsafe { ash::Entry::load() }.map_err(first_error)?;

        // --- Instance -----------------------------------------------------
        let app_name = cstring("teeter");
        let engine_name = cstring("teeter");
        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .engine_name(&engine_name)
            .api_version(vk::API_VERSION_1_2);
        let display = window
            .display_handle()
            .map_err(|e| format!("display handle: {e}"))?;
        let win_handle = window
            .window_handle()
            .map_err(|e| format!("window handle: {e}"))?;
        let instance_extensions = [
            vk::KhrSurfaceFn::name().as_ptr(),
            vk::KhrWin32SurfaceFn::name().as_ptr(),
        ];
        let instance_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&instance_extensions);
        let instance = unsafe { entry.create_instance(&instance_create_info, None) }
            .map_err(|e| format!("create_instance: {e}"))?;

        // --- Surface (Win32, created manually from the winit raw handles) ---
        let (hinstance, hwnd) = match (display.as_raw(), win_handle.as_raw()) {
            (RawDisplayHandle::Windows(_), RawWindowHandle::Win32(win)) => (
                win.hinstance
                    .ok_or_else(|| "missing Win32 hinstance".to_string())?
                    .get() as *const std::ffi::c_void,
                win.hwnd.get() as *const std::ffi::c_void,
            ),
            _ => return Err("teeter requires a Win32 window".to_string()),
        };
        let surface_info = vk::Win32SurfaceCreateInfoKHR::builder()
            .hinstance(hinstance)
            .hwnd(hwnd);
        let win32_surface =
            ash::extensions::khr::Win32Surface::new(&entry, &instance);
        let surface = unsafe { win32_surface.create_win32_surface(&surface_info, None) }
            .map_err(|e| format!("create_win32_surface: {e}"))?;
        let khr_surface = ash::extensions::khr::Surface::new(&entry, &instance);

        // --- Physical device + queue family --------------------------------
        let physical_device = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| format!("enumerate: {e}"))?
            .first()
            .copied()
            .ok_or_else(|| "no physical device".to_string())?;

        let (queue_family_index, _) = unsafe {
            find_queue_family(&instance, &khr_surface, physical_device, surface)?
        };

        let queue_priorities = [1.0f32];
        let queue_create_infos = [vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities)
            .build()];

        let device_extensions: [*const std::os::raw::c_char; 1] =
            [vk::KhrSwapchainFn::name().as_ptr()];
        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions);
        let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }
            .map_err(|e| format!("create_device: {e}"))?;

        let graphics_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // --- Swapchain -----------------------------------------------------
        let (extent, format) = unsafe {
            let cp = khr_surface
                .get_physical_device_surface_capabilities(physical_device, surface)
                .map_err(first_error)?;
            let formats = khr_surface
                .get_physical_device_surface_formats(physical_device, surface)
                .map_err(first_error)?;
            let fmt = formats
                .iter()
                .find(|f| f.format == vk::Format::B8G8R8A8_SRGB)
                .map(|f| f.format)
                .unwrap_or(formats[0].format);
            let ext = if cp.current_extent.width != u32::MAX {
                cp.current_extent
            } else {
                let s = window.inner_size();
                vk::Extent2D { width: s.width.max(1), height: s.height.max(1) }
            };
            (ext, fmt)
        };

        let swapchain_khr = ash::extensions::khr::Swapchain::new(&instance, &device);
        let (swapchain, swapchain_images) = unsafe {
            let present_modes = khr_surface
                .get_physical_device_surface_present_modes(physical_device, surface)
                .map_err(first_error)?;
            let present_mode = if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
                vk::PresentModeKHR::MAILBOX
            } else {
                vk::PresentModeKHR::FIFO
            };
            let caps = khr_surface
                .get_physical_device_surface_capabilities(physical_device, surface)
                .map_err(first_error)?;
            let mut count = caps.min_image_count.saturating_add(1);
            if caps.max_image_count > 0 && count > caps.max_image_count {
                count = caps.max_image_count;
            }
            let create = vk::SwapchainCreateInfoKHR::builder()
                .surface(surface)
                .min_image_count(count)
                .image_format(format)
                .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(caps.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .old_swapchain(vk::SwapchainKHR::null());
            let sc = swapchain_khr.create_swapchain(&create, None).map_err(|e| format!("swapchain: {e}"))?;
            let images = swapchain_khr.get_swapchain_images(sc).map_err(first_error)?;
            (sc, images)
        };

        // --- Image views + render pass -------------------------------------
        let swapchain_views = unsafe {
            (0..swapchain_images.len())
                .map(|i| {
                    create_image_view(&device, swapchain_images[i], format)
                })
                .collect::<Result<Vec<_>, String>>()?
        };

        let render_pass = unsafe { create_render_pass(&device, format)? };
        let swapchain_fbs = unsafe {
            (0..swapchain_views.len())
                .map(|i| create_framebuffer(&device, &render_pass, swapchain_views[i], extent))
                .collect::<Result<Vec<_>, String>>()?
        };

        // --- Feedback texture ----------------------------------------------
        let (feedback_image, feedback_memory, feedback_view) = unsafe {
            create_image(
                &device,
                instance.get_physical_device_memory_properties(physical_device),
                extent,
                format,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            )?
        };
        let feedback_sampler = unsafe { create_sampler(&device)? };
        // Frozen pre-switch copy for the warp crossfade (sampled as texOld).
        let (feedback_old_image, feedback_old_memory, feedback_old_view) = unsafe {
            create_image(
                &device,
                instance.get_physical_device_memory_properties(physical_device),
                extent,
                format,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            )?
        };

        // --- Pipeline (single: warp.frag -> swapchain) ----------------------
        let desc_set_layout = unsafe {
            create_desc_set_layout(
                &device,
                ubo_binding(),
                feedback_binding(),
                tex_old_binding(),
            )?
        };
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::builder().set_layouts(&[desc_set_layout]),
                    None,
                )
                .map_err(|e| format!("pipeline layout: {e}"))?
        };
        // The 5-cell overlay uses its own UBO-only layout (DSPUniforms block,
        // different layout from the warp FrameUniforms).
        let fivecell_desc_set_layout = unsafe {
            create_desc_set_layout_ubo_only(&device, ubo_binding())?
        };
        let fivecell_pipeline_layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::builder()
                        .set_layouts(&[fivecell_desc_set_layout]),
                    None,
                )
                .map_err(|e| format!("5-cell pipeline layout: {e}"))?
        };
        // Build one pipeline per selectable warp shader (same layout/render
        // pass, different fragment shader). They share the descriptor layout,
        // so the desc_set stays valid across all of them.
        let mut pipelines = Vec::with_capacity(20);
        for frag in [
            WARP_SPV, WARP2_SPV, WARP3_SPV, WARP4_SPV, WARP5_SPV, WARP6_SPV, WARP7_SPV,
            WARP8_SPV, WARP9_SPV, WARP10_SPV, WARP11_SPV, WARP12_SPV, WARP13_SPV,
            WARP14_SPV, WARP15_SPV, WARP16_SPV, WARP17_SPV, WARP18_SPV, WARP19_SPV, WARP20_SPV,
        ] {
            pipelines.push(unsafe { create_pipeline(&device, &pipeline_layout, &render_pass, format, frag)? });
        }
        // 5-cell wireframe pipeline (additive lines drawn over the feedback
        // pass; own geometry + UBO layout).
        let fivecell_pipeline = unsafe {
            create_fivecell_pipeline(
                &device,
                &fivecell_pipeline_layout,
                &render_pass,
                format,
                CELL_VERT_SPV,
                CELL_FRAG_SPV,
            )?
        };

        // --- Uniform buffer -------------------------------------------------
        let uniform_size = std::mem::size_of::<UniformData>() as vk::DeviceSize;
        let (uniform_buffer, uniform_memory) = unsafe {
            create_buffer(
                &device,
                instance.get_physical_device_memory_properties(physical_device),
                uniform_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?
        };

        // --- 5-cell overlay buffers (static geometry + DSP UBO) ------------
        let vertex_data: Vec<u8> = fivecell::vertices()
            .iter()
            .flat_map(|v| v.iter())
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let index_data: Vec<u8> = fivecell::edge_indices()
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();
        let (fivecell_vertex_buffer, fivecell_vertex_memory) = unsafe {
            create_buffer(
                &device,
                instance.get_physical_device_memory_properties(physical_device),
                vertex_data.len() as vk::DeviceSize,
                vk::BufferUsageFlags::VERTEX_BUFFER,
            )?
        };
        let (fivecell_index_buffer, fivecell_index_memory) = unsafe {
            create_buffer(
                &device,
                instance.get_physical_device_memory_properties(physical_device),
                index_data.len() as vk::DeviceSize,
                vk::BufferUsageFlags::INDEX_BUFFER,
            )?
        };
        let fivecell_uniform_size = fivecell::UNIFORM_SIZE as vk::DeviceSize;
        let (fivecell_uniform_buffer, fivecell_uniform_memory) = unsafe {
            create_buffer(
                &device,
                instance.get_physical_device_memory_properties(physical_device),
                fivecell_uniform_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?
        };
        // Fill static vertex + index data once (the geometry never changes).
        unsafe {
            let vptr = device
                .map_memory(
                    fivecell_vertex_memory,
                    0,
                    vertex_data.len() as vk::DeviceSize,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("map 5-cell vertex buffer");
            std::ptr::copy_nonoverlapping(vertex_data.as_ptr(), vptr as *mut u8, vertex_data.len());
            device.unmap_memory(fivecell_vertex_memory);

            let iptr = device
                .map_memory(
                    fivecell_index_memory,
                    0,
                    index_data.len() as vk::DeviceSize,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("map 5-cell index buffer");
            std::ptr::copy_nonoverlapping(index_data.as_ptr(), iptr as *mut u8, index_data.len());
            device.unmap_memory(fivecell_index_memory);
        }

        // --- Descriptor pool + sets (warp set + 5-cell set) -----------------
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 2,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 2,
            },
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(2);
        let desc_pool = unsafe {
            device
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| format!("desc pool: {e}"))?
        };
        let desc_set_layouts = [desc_set_layout, fivecell_desc_set_layout];
        let desc_sets = unsafe {
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(desc_pool)
                        .set_layouts(&desc_set_layouts),
                )
                .map_err(first_error)?
        };
        let desc_set = desc_sets[0];
        let fivecell_desc_set = desc_sets[1];

        let ubo_info = vk::DescriptorBufferInfo::builder()
            .buffer(uniform_buffer)
            .offset(0)
            .range(uniform_size)
            .build();
        let img_info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(feedback_view)
            .sampler(feedback_sampler)
            .build();
        let old_img_info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(feedback_old_view)
            .sampler(feedback_sampler)
            .build();
        let c_ubo_info = vk::DescriptorBufferInfo::builder()
            .buffer(fivecell_uniform_buffer)
            .offset(0)
            .range(fivecell_uniform_size)
            .build();
        let ubo_infos = [ubo_info];
        let img_infos = [img_info];
        let old_img_infos = [old_img_info];
        let c_ubo_infos = [c_ubo_info];
        let writes = [
            vk::WriteDescriptorSet::builder()
                .dst_set(desc_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&ubo_infos)
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(desc_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&img_infos)
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(desc_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&old_img_infos)
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(fivecell_desc_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&c_ubo_infos)
                .build(),
        ];
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }

        // --- Command pool/buffers + sync ------------------------------------
        let command_pool = unsafe {
            device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::builder()
                        .queue_family_index(queue_family_index)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(|e| format!("cmd pool: {e}"))?
        };
        let command_buffers = unsafe {
            device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::builder()
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32),
                )
                .map_err(first_error)?
        };

        let mut image_available = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut render_finished = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut in_flight = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            image_available.push(
                unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                    .map_err(first_error)?,
            );
            render_finished.push(
                unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                    .map_err(first_error)?,
            );
            in_flight.push(
                unsafe {
                    device.create_fence(
                        &vk::FenceCreateInfo::builder()
                            .flags(vk::FenceCreateFlags::SIGNALED),
                        None,
                    )
                }
                .map_err(first_error)?,
            );
        }

        // Seed the feedback texture (one-shot clear -> SHADER_READ_ONLY) so the
        // first frame samples valid, defined content instead of garbage.
        unsafe { seed_feedback(&device, graphics_queue, command_pool, feedback_image, extent)? };
        // Seed the frozen frame copy too (texOld starts defined, not garbage).
        unsafe { seed_feedback(&device, graphics_queue, command_pool, feedback_old_image, extent)? };

        Ok(Renderer {
            _entry: entry,
            _instance: instance,
            surface,
            khr_surface,
            physical_device,
            swapchain_khr,
            swapchain,
            swapchain_images,
            swapchain_views,
            swapchain_fbs,
            swapchain_format: format,
            swapchain_extent: extent,
            device,
            graphics_queue,
            feedback_image,
            feedback_memory,
            feedback_view,
            feedback_sampler,
            feedback_old_image,
            feedback_old_memory,
            feedback_old_view,
            crossfade: 0.0,
            freeze_pending: false,
            render_pass,
            pipeline_layout,
            pipelines,
            shader_index: 0,
            desc_set,
            uniform_memory,
            uniform_size,
            fivecell_pipeline,
            fivecell_pipeline_layout,
            fivecell_desc_set,
            fivecell_vertex_buffer,
            fivecell_vertex_memory,
            fivecell_index_buffer,
            fivecell_index_memory,
            fivecell_uniform_buffer,
            fivecell_uniform_memory,
            fivecell_uniform_size,
            fivecell_active: false,
            command_pool,
            command_buffers,
            image_available,
            render_finished,
            in_flight,
            frame_index: 0,
        })
    }

    pub fn recreate_swapchain(&mut self, window: &Window) -> Result<(), String> {
        // Rebuild the swapchain, views, framebuffers and the feedback texture
        // at the window's current size (handles windowed -> fullscreen, and
        // any resize). Pipelines, UBO, desc_pool, cmd pool and sync objects
        // are extent-independent and stay as-is. The descriptor set is
        // re-pointed to the fresh feedback view.
        let size = window.inner_size();
        let extent = vk::Extent2D { width: size.width.max(1), height: size.height.max(1) };
        if extent == self.swapchain_extent {
            return Ok(());
        }

        let device = &self.device;
        let surface_ext = &self.khr_surface;
        let surface = self.surface;
        let physical_device = self.physical_device;
        let format = self.swapchain_format;

        unsafe {
            device.device_wait_idle().map_err(first_error)?;

            // Tear down swapchain-dependent resources.
            for fb in self.swapchain_fbs.drain(..) {
                device.destroy_framebuffer(fb, None);
            }
            for v in self.swapchain_views.drain(..) {
                device.destroy_image_view(v, None);
            }
            device.destroy_image(self.feedback_image, None);
            device.free_memory(self.feedback_memory, None);
            device.destroy_image_view(self.feedback_view, None);
            device.destroy_image(self.feedback_old_image, None);
            device.free_memory(self.feedback_old_memory, None);
            device.destroy_image_view(self.feedback_old_view, None);
            self.swapchain_khr.destroy_swapchain(self.swapchain, None);
        }

        // --- New swapchain ---------------------------------------------------
        let (swapchain, images) = unsafe {
            let caps = surface_ext
                .get_physical_device_surface_capabilities(physical_device, surface)
                .map_err(first_error)?;
            let present_modes = surface_ext
                .get_physical_device_surface_present_modes(physical_device, surface)
                .map_err(first_error)?;
            let present_mode = if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
                vk::PresentModeKHR::MAILBOX
            } else {
                vk::PresentModeKHR::FIFO
            };
            let mut count = caps.min_image_count.saturating_add(1);
            if caps.max_image_count > 0 && count > caps.max_image_count {
                count = caps.max_image_count;
            }
            // Reuse the prior surface supported extent where possible.
            let ext = if caps.current_extent.width != u32::MAX {
                caps.current_extent
            } else {
                extent
            };
            let create = vk::SwapchainCreateInfoKHR::builder()
                .surface(surface)
                .min_image_count(count)
                .image_format(format)
                .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                .image_extent(ext)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(caps.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .old_swapchain(vk::SwapchainKHR::null());
            let sc = self
                .swapchain_khr
                .create_swapchain(&create, None)
                .map_err(|e| format!("recreate swapchain: {e}"))?;
            let imgs = self.swapchain_khr.get_swapchain_images(sc).map_err(first_error)?;
            (sc, imgs)
        };

        let views = unsafe {
            (0..images.len())
                .map(|i| create_image_view(&device, images[i], format))
                .collect::<Result<Vec<_>, String>>()?
        };
        let fbs = unsafe {
            (0..views.len())
                .map(|i| create_framebuffer(&device, &self.render_pass, views[i], extent))
                .collect::<Result<Vec<_>, String>>()?
        };

        // --- New feedback texture + re-point descriptor ---------------------
        let (feedback_image, feedback_memory, feedback_view) = unsafe {
            create_image(
                &device,
                self.instance_memory_props(),
                extent,
                format,
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )?
        };

        let (feedback_old_image, feedback_old_memory, feedback_old_view) = unsafe {
            create_image(
                &device,
                self.instance_memory_props(),
                extent,
                format,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            )?
        };

        // Seed the fresh feedback texture (clear -> SHADER_READ_ONLY) so the
        // resized loop starts from defined content rather than garbage.
        unsafe { seed_feedback(&device, self.graphics_queue, self.command_pool, feedback_image, extent)? };
        unsafe { seed_feedback(&device, self.graphics_queue, self.command_pool, feedback_old_image, extent)? };

        unsafe {
            let img_info = vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(feedback_view)
                .sampler(self.feedback_sampler)
                .build();
            let old_img_info = vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(feedback_old_view)
                .sampler(self.feedback_sampler)
                .build();
            let img_infos = [img_info];
            let old_img_infos = [old_img_info];
            let writes = [
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.desc_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&img_infos)
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.desc_set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&old_img_infos)
                    .build(),
            ];
            device.update_descriptor_sets(&writes, &[]);
        }

        self.swapchain = swapchain;
        self.swapchain_images = images;
        self.swapchain_views = views;
        self.swapchain_fbs = fbs;
        self.swapchain_extent = extent;
        self.feedback_image = feedback_image;
        self.feedback_memory = feedback_memory;
        self.feedback_view = feedback_view;
        self.feedback_old_image = feedback_old_image;
        self.feedback_old_memory = feedback_old_memory;
        self.feedback_old_view = feedback_old_view;
        self.frame_index = 0;

        log::info!("teeter: swapchain resized to {}x{}", extent.width, extent.height);
        Ok(())
    }

    /// Helper to fetch the physical device memory properties (for image/buffer
    /// allocation during recreate).
    fn instance_memory_props(&self) -> vk::PhysicalDeviceMemoryProperties {
        unsafe { self._instance.get_physical_device_memory_properties(self.physical_device) }
    }

    /// Cycle to the next warp shader.
    pub fn next_shader(&mut self) {
        self.shader_index = (self.shader_index + 1) % self.pipelines.len();
        self.crossfade = 1.0;
        self.freeze_pending = true;
        log::info!("teeter: shader {} of {}", self.shader_index + 1, self.pipelines.len());
    }

    /// Toggle the 5-cell wireframe overlay (West / X button).
    pub fn toggle_fivecell(&mut self) {
        self.fivecell_active = !self.fivecell_active;
        log::info!("teeter: 5-cell overlay -> {}", self.fivecell_active);
    }

    /// Is the 5-cell overlay currently active?
    pub fn fivecell_active(&self) -> bool {
        self.fivecell_active
    }

    /// Active warp shader index.
    pub fn shader_index(&self) -> usize {
        self.shader_index
    }

    /// Number of selectable warp shaders (for display purposes).
    pub fn shader_count(&self) -> usize {
        self.pipelines.len()
    }

    /// Display metadata for the active warp: (name, genre). Falls back to a
    /// generic "warp N" tag for the classic shaders that have no name yet.
    pub fn warp_meta(&self) -> Option<(&'static str, &'static str)> {
        warp_meta(self.shader_index)
    }

    /// Jump to a specific shader index (used by --shader CLI arg).
    pub fn set_shader_index(&mut self, idx: usize) {
        if !self.pipelines.is_empty() {
            self.shader_index = idx % self.pipelines.len();
        }
    }

    pub fn render(&mut self, engine: &EngineState, input: &InputState, audio: &AudioFrame) {
        let device = &self.device;
        if self.swapchain_views.is_empty() {
            return;
        }
        // Cycle frames in flight on a FIXED count (independent of the swapchain
        // image count) so recreation never runs these arrays out of bounds.
        self.frame_index = (self.frame_index + 1) % MAX_FRAMES_IN_FLIGHT;
        let i = self.frame_index;

        let fence = self.in_flight[i];
        unsafe {
            let _ = device.wait_for_fences(&[fence], true, u64::MAX);
            let _ = device.reset_fences(&[fence]);
        }

        let acquire_sem = self.image_available[i];
        let complete_sem = self.render_finished[i];
        let (image_index, _) = unsafe {
            self.swapchain_khr
                .acquire_next_image(self.swapchain, u64::MAX, acquire_sem, vk::Fence::null())
        }
        .unwrap_or((0, false));
        if image_index as usize >= self.swapchain_fbs.len() {
            return; // swapchain changed under us (out-of-date); next frame retries
        }

        // Upload UBO.
        let mut data = UniformData::from((engine, input, audio));
        data.aspect = self.swapchain_extent.width as f32
            / self.swapchain_extent.height.max(1) as f32;
        // Crossfade: a warp switch spikes `dissolve`, which the shaders treat
        // as the mix weight against the frozen pre-switch frame (texOld). The
        // spike decays each frame so switching melts rather than cuts.
        data.dissolve = data.dissolve.max(self.crossfade);
        self.crossfade *= 0.92;
        unsafe {
            let ptr = device
                .map_memory(self.uniform_memory, 0, self.uniform_size, vk::MemoryMapFlags::empty())
                .expect("map uniform memory");
            std::ptr::copy_nonoverlapping(
                (&data as *const UniformData) as *const u8,
                ptr as *mut u8,
                std::mem::size_of::<UniformData>(),
            );
            device.unmap_memory(self.uniform_memory);
        }

        let cb = self.command_buffers[i];
        unsafe {
            device
                .begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default())
                .expect("begin cb");
        }

        // On a warp switch, freeze the current pre-switch frame into the
        // texOld copy BEFORE the new shader draws, so the crossfade has
        // something to melt out of.
        if self.freeze_pending {
            unsafe {
                let src_old = self.feedback_image;
                let dst_old = self.feedback_old_image;
                transition_image(device, cb, src_old, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
                transition_image(device, cb, dst_old, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
                let freeze_region = vk::ImageBlit {
                    src_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    src_offsets: [
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: self.swapchain_extent.width as i32,
                            y: self.swapchain_extent.height as i32,
                            z: 1,
                        },
                    ],
                    dst_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    dst_offsets: [
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: self.swapchain_extent.width as i32,
                            y: self.swapchain_extent.height as i32,
                            z: 1,
                        },
                    ],
                };
                device.cmd_blit_image(
                    cb,
                    src_old,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    dst_old,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[freeze_region],
                    vk::Filter::LINEAR,
                );
                transition_image(device, cb, src_old, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
                transition_image(device, cb, dst_old, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            }
            self.freeze_pending = false;
        }

        // Render pass: warp.frag samples feedback texture -> swapchain image.
        let clear = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };
        let area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.swapchain_extent,
        };
        let clear_values = [clear];
        let pass = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .framebuffer(self.swapchain_fbs[image_index as usize])
            .render_area(area)
            .clear_values(&clear_values);
        unsafe {
            device.cmd_begin_render_pass(cb, &pass, vk::SubpassContents::INLINE);
            // The pipeline uses DYNAMIC viewport/scissor — must be set before
            // draw, otherwise the triangle clips to a zero region (black).
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.swapchain_extent.width as f32,
                height: self.swapchain_extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            let viewports = [viewport];
            device.cmd_set_viewport(cb, 0, &viewports);
            let scissors = [area];
            device.cmd_set_scissor(cb, 0, &scissors);
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipelines[self.shader_index]);
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.desc_set],
                &[],
            );
            device.cmd_draw(cb, 3, 1, 0, 0);

            // --- 5-cell wireframe overlay (additive, over the feedback) ----
            if self.fivecell_active {
                let aspect = self.swapchain_extent.width as f32
                    / self.swapchain_extent.height.max(1) as f32;
                let proj = glam::Mat4::perspective_rh_gl(
                    std::f32::consts::FRAC_PI_4,
                    aspect,
                    0.1,
                    100.0,
                );
                let view = glam::Mat4::look_at_rh(
                    glam::Vec3::new(0.0, 0.0, 3.5),
                    glam::Vec3::ZERO,
                    glam::Vec3::new(0.0, 1.0, 0.0),
                );
                let mvp = proj * view;

                // Upload the DSP UBO for this frame (audio-driven rotations).
                let cu = DSPUniforms::from_audio(mvp, audio, data.i_time);
                let cptr = device
                    .map_memory(
                        self.fivecell_uniform_memory,
                        0,
                        self.fivecell_uniform_size,
                        vk::MemoryMapFlags::empty(),
                    )
                    .expect("map 5-cell uniform memory");
                std::ptr::copy_nonoverlapping(
                    (&cu as *const DSPUniforms) as *const u8,
                    cptr as *mut u8,
                    std::mem::size_of::<DSPUniforms>(),
                );
                device.unmap_memory(self.fivecell_uniform_memory);

                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.fivecell_pipeline);
                device.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.fivecell_pipeline_layout,
                    0,
                    &[self.fivecell_desc_set],
                    &[],
                );
                let vbs = [self.fivecell_vertex_buffer];
                let offsets = [0u64];
                device.cmd_bind_vertex_buffers(cb, 0, &vbs, &offsets);
                device.cmd_bind_index_buffer(cb, self.fivecell_index_buffer, 0, vk::IndexType::UINT16);
                device.cmd_draw_indexed(cb, 20, 1, 0, 0, 0);
            }

            device.cmd_end_render_pass(cb);

            // Blit the freshly rendered swapchain image into the feedback tex.
            // Transition feedback tex to TRANSFER_DST_OPTIMAL, blit, then back
            // to SHADER_READ_ONLY for the next frame's sampling.
            let dst = self.feedback_image;
            transition_image(device, cb, dst, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            let blit_region = vk::ImageBlit {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: self.swapchain_extent.width as i32,
                        y: self.swapchain_extent.height as i32,
                        z: 1,
                    },
                ],
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                dst_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: self.swapchain_extent.width as i32,
                        y: self.swapchain_extent.height as i32,
                        z: 1,
                    },
                ],
            };
            let src = self.swapchain_images[image_index as usize];
            // src must be in TRANSFER_SRC_OPTIMAL.
            transition_image(device, cb, src, vk::ImageLayout::PRESENT_SRC_KHR, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
            device.cmd_blit_image(
                cb,
                src,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit_region],
                vk::Filter::LINEAR,
            );
            transition_image(device, cb, dst, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            transition_image(device, cb, src, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR);

            device.end_command_buffer(cb).expect("end cb");
        }

        // Submit + present.
        let wait = [acquire_sem];
        let wait_stage = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal = [complete_sem];
        let cbs = [cb];
        let submit = vk::SubmitInfo::builder()
            .wait_semaphores(&wait)
            .wait_dst_stage_mask(&wait_stage)
            .command_buffers(&cbs)
            .signal_semaphores(&signal)
            .build();
        let swapchains = [self.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&signal)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        unsafe {
            device
                .queue_submit(self.graphics_queue, &[submit], fence)
                .expect("queue submit");
            let _ = self.swapchain_khr.queue_present(self.graphics_queue, &present_info);
        }
    }
}

impl From<(&EngineState, &InputState, &AudioFrame)> for UniformData {
    fn from((e, i, a): (&EngineState, &InputState, &AudioFrame)) -> Self {
        // Consume the engine's *blended* motion values (genre-driven auto
        // motion mixed with whatever the player is actually doing) so the
        // seeds / decay / manual-weight logic in EngineState actually reach the
        // shader instead of being bypassed by raw input values.
        let u = e.uniforms(i, a);
        Self {
            i_time: u.i_time,
            bass: u.bass,
            mid: u.mid,
            treble: u.treble,
            beat: u.beat,
            zoom: u.zoom,
            warp_x: u.warp_x,
            warp_y: u.warp_y,
            rotate: u.rotate,
            dissolve: u.dissolve,
            palette: u.palette,
            preset_seed: u.preset_seed,
            aspect: 1.0,
            sub_bass: u.sub_bass,
            centroid: u.centroid,
            crest: u.crest,
            flux: u.flux,
            rolloff: u.rolloff,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers

const UB_BINDING: u32 = 0;
const TEX_BINDING: u32 = 1;
const TEX_OLD_BINDING: u32 = 2;

fn ubo_binding() -> vk::DescriptorSetLayoutBinding {
    vk::DescriptorSetLayoutBinding::builder()
        .binding(UB_BINDING)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .build()
}

fn feedback_binding() -> vk::DescriptorSetLayoutBinding {
    vk::DescriptorSetLayoutBinding::builder()
        .binding(TEX_BINDING)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .build()
}

fn tex_old_binding() -> vk::DescriptorSetLayoutBinding {
    vk::DescriptorSetLayoutBinding::builder()
        .binding(TEX_OLD_BINDING)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .build()
}

unsafe fn find_queue_family(
    instance: &ash::Instance,
    surface_ext: &ash::extensions::khr::Surface,
    phy: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> Result<(u32, u32), String> {
    let props = instance.get_physical_device_queue_family_properties(phy);
    for (i, q) in props.iter().enumerate() {
        let present = surface_ext
            .get_physical_device_surface_support(phy, i as u32, surface)
            .unwrap_or(false);
        if q.queue_flags.contains(vk::QueueFlags::GRAPHICS) && present {
            return Ok((i as u32, i as u32));
        }
    }
    Err("no suitable queue family".to_string())
}

unsafe fn create_image_view(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
) -> Result<vk::ImageView, String> {
    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    let info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(range);
    device
        .create_image_view(&info, None)
        .map_err(|e| format!("image view: {e}"))
}

unsafe fn create_render_pass(
    device: &ash::Device,
    format: vk::Format,
) -> Result<vk::RenderPass, String> {
    let attachment = vk::AttachmentDescription::builder()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .build();
    let attachments = [attachment];
    let ref0 = vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    };
    let color_attachments = [ref0];
    let subpass = vk::SubpassDescription::builder().color_attachments(&color_attachments).build();
    let subpasses = [subpass];
    let info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachments)
        .subpasses(&subpasses);
    device
        .create_render_pass(&info, None)
        .map_err(|e| format!("render pass: {e}"))
}

unsafe fn create_framebuffer(
    device: &ash::Device,
    render_pass: &vk::RenderPass,
    view: vk::ImageView,
    extent: vk::Extent2D,
) -> Result<vk::Framebuffer, String> {
    let attach_views = [view];
    let info = vk::FramebufferCreateInfo::builder()
        .render_pass(*render_pass)
        .attachments(&attach_views)
        .width(extent.width.max(1))
        .height(extent.height.max(1))
        .layers(1);
    device
        .create_framebuffer(&info, None)
        .map_err(|e| format!("framebuffer: {e}"))
}

unsafe fn create_image(
    device: &ash::Device,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    extent: vk::Extent2D,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView), String> {
    let info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: extent.width.max(1),
            height: extent.height.max(1),
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = device
        .create_image(&info, None)
        .map_err(|e| format!("create image: {e}"))?;
    let reqs = device.get_image_memory_requirements(image);
    let mem_type = find_memory_type(mem_props, reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .ok_or_else(|| "no device-local memory".to_string())?;
    let alloc = vk::MemoryAllocateInfo::builder()
        .allocation_size(reqs.size)
        .memory_type_index(mem_type);
    let memory = device
        .allocate_memory(&alloc, None)
        .map_err(|e| format!("allocate image mem: {e}"))?;
    device
        .bind_image_memory(image, memory, 0)
        .map_err(|e| format!("bind image mem: {e}"))?;
    let view = create_image_view(device, image, format)?;
    Ok((image, memory, view))
}

unsafe fn create_buffer(
    device: &ash::Device,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory), String> {
    let info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = device
        .create_buffer(&info, None)
        .map_err(|e| format!("create buffer: {e}"))?;
    let reqs = device.get_buffer_memory_requirements(buffer);
    let mem_type = find_memory_type(
        mem_props,
        reqs.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .ok_or_else(|| "no host-visible memory".to_string())?;
    let alloc = vk::MemoryAllocateInfo::builder()
        .allocation_size(reqs.size)
        .memory_type_index(mem_type);
    let memory = device
        .allocate_memory(&alloc, None)
        .map_err(|e| format!("allocate buffer mem: {e}"))?;
    device
        .bind_buffer_memory(buffer, memory, 0)
        .map_err(|e| format!("bind buffer mem: {e}"))?;
    Ok((buffer, memory))
}

fn find_memory_type(
    props: vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    props.memory_types.iter().enumerate().find_map(|(i, m)| {
        if (type_bits & (1 << i)) != 0 && m.property_flags.contains(flags) {
            Some(i as u32)
        } else {
            None
        }
    })
}

unsafe fn create_sampler(device: &ash::Device) -> Result<vk::Sampler, String> {
    let info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .mip_lod_bias(0.0)
        .min_lod(0.0)
        .max_lod(0.0);
    device
        .create_sampler(&info, None)
        .map_err(|e| format!("sampler: {e}"))
}

unsafe fn create_desc_set_layout(
    device: &ash::Device,
    ubo: vk::DescriptorSetLayoutBinding,
    tex: vk::DescriptorSetLayoutBinding,
    tex_old: vk::DescriptorSetLayoutBinding,
) -> Result<vk::DescriptorSetLayout, String> {
    let bindings = [ubo, tex, tex_old];
    let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
    device
        .create_descriptor_set_layout(&info, None)
        .map_err(|e| format!("desc set layout: {e}"))
}

unsafe fn create_desc_set_layout_ubo_only(
    device: &ash::Device,
    ubo: vk::DescriptorSetLayoutBinding,
) -> Result<vk::DescriptorSetLayout, String> {
    let bindings = [ubo];
    let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
    device
        .create_descriptor_set_layout(&info, None)
        .map_err(|e| format!("5-cell desc set layout: {e}"))
}

unsafe fn create_pipeline(
    device: &ash::Device,
    pipeline_layout: &vk::PipelineLayout,
    render_pass: &vk::RenderPass,
    _format: vk::Format,
    frag_spv: &[u8],
) -> Result<vk::Pipeline, String> {
    let vert_code = to_u32s(VERT_SPV);
    let vert_module = device
        .create_shader_module(
            &vk::ShaderModuleCreateInfo::builder().code(&vert_code),
            None,
        )
        .map_err(|e| format!("vert module: {e}"))?;
    let frag_code = to_u32s(frag_spv);
    let frag_module = device
        .create_shader_module(
            &vk::ShaderModuleCreateInfo::builder().code(&frag_code),
            None,
        )
        .map_err(|e| format!("frag module: {e}"))?;

    let entry = cstring("main");
    let stages = [
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(&entry)
            .build(),
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(&entry)
            .build(),
    ];

    let vertex_input = vk::PipelineVertexInputStateCreateInfo::builder().build();
    let input_asm = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .build();

    // Dynamic viewport/scissor so resize stays trivial.
    let viewport = vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: 1920.0,
        height: 1080.0,
        min_depth: 0.0,
        max_depth: 1.0,
    };
    let scissor = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D { width: 1920, height: 1080 },
    };
    let viewports = [viewport];
    let scissors = [scissor];
    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(&viewports)
        .scissors(&scissors)
        .build();
    let raster = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false)
        .line_width(1.0)
        .build();
    let msaa = vk::PipelineMultisampleStateCreateInfo::builder()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
        .build();
    let blend_attach = vk::PipelineColorBlendAttachmentState::builder()
        .blend_enable(false)
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        )
        .build();
    let blend_attachments = [blend_attach];
    let blend = vk::PipelineColorBlendStateCreateInfo::builder()
        .attachments(&blend_attachments)
        .build();
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
        .dynamic_states(&dynamic_states)
        .build();
    let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_asm)
        .viewport_state(&viewport_state)
        .rasterization_state(&raster)
        .multisample_state(&msaa)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic_state)
        .layout(*pipeline_layout)
        .render_pass(*render_pass)
        .subpass(0)
        .build();

    let pipeline = device
        .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        .map_err(|(_, e)| format!("pipeline: {e}"))?[0];

    device.destroy_shader_module(vert_module, None);
    device.destroy_shader_module(frag_module, None);
    Ok(pipeline)
}

unsafe fn create_fivecell_pipeline(
    device: &ash::Device,
    pipeline_layout: &vk::PipelineLayout,
    render_pass: &vk::RenderPass,
    _format: vk::Format,
    vert_spv: &[u8],
    frag_spv: &[u8],
) -> Result<vk::Pipeline, String> {
    let vert_code = to_u32s(vert_spv);
    let vert_module = device
        .create_shader_module(
            &vk::ShaderModuleCreateInfo::builder().code(&vert_code),
            None,
        )
        .map_err(|e| format!("5-cell vert module: {e}"))?;
    let frag_code = to_u32s(frag_spv);
    let frag_module = device
        .create_shader_module(
            &vk::ShaderModuleCreateInfo::builder().code(&frag_code),
            None,
        )
        .map_err(|e| format!("5-cell frag module: {e}"))?;

    let entry = cstring("main");
    let stages = [
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(&entry)
            .build(),
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(&entry)
            .build(),
    ];

    // Per-vertex 4D position (vec4, 16-byte stride).
    let binding = vk::VertexInputBindingDescription::builder()
        .binding(0)
        .stride(16)
        .input_rate(vk::VertexInputRate::VERTEX)
        .build();
    let attribute = vk::VertexInputAttributeDescription::builder()
        .binding(0)
        .location(0)
        .format(vk::Format::R32G32B32A32_SFLOAT)
        .offset(0)
        .build();
    let bindings = [binding];
    let attributes = [attribute];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attributes)
        .build();
    let input_asm = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::LINE_LIST)
        .build();

    // Dynamic viewport/scissor (set per frame, matches the warp pipeline).
    let viewport = vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: 1920.0,
        height: 1080.0,
        min_depth: 0.0,
        max_depth: 1.0,
    };
    let scissor = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D { width: 1920, height: 1080 },
    };
    let viewports = [viewport];
    let scissors = [scissor];
    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(&viewports)
        .scissors(&scissors)
        .build();
    let raster = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false)
        .line_width(1.0)
        .build();
    let msaa = vk::PipelineMultisampleStateCreateInfo::builder()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
        .build();
    // Additive blend: the wireframe glows over the feedback background.
    let blend_attach = vk::PipelineColorBlendAttachmentState::builder()
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ONE)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD)
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        )
        .build();
    let blend_attachments = [blend_attach];
    let blend = vk::PipelineColorBlendStateCreateInfo::builder()
        .attachments(&blend_attachments)
        .build();
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
        .dynamic_states(&dynamic_states)
        .build();
    let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_asm)
        .viewport_state(&viewport_state)
        .rasterization_state(&raster)
        .multisample_state(&msaa)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic_state)
        .layout(*pipeline_layout)
        .render_pass(*render_pass)
        .subpass(0)
        .build();

    let pipeline = device
        .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        .map_err(|(_, e)| format!("5-cell pipeline: {e}"))?[0];

    device.destroy_shader_module(vert_module, None);
    device.destroy_shader_module(frag_module, None);
    Ok(pipeline)
}

unsafe fn transition_image(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    image: vk::Image,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
) {
    let (src_mask, dst_mask) = match (old, new) {
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => {
            (vk::AccessFlags::NONE, vk::AccessFlags::TRANSFER_WRITE)
        }
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => {
            (vk::AccessFlags::NONE, vk::AccessFlags::SHADER_READ)
        }
        (vk::ImageLayout::PRESENT_SRC_KHR, vk::ImageLayout::TRANSFER_SRC_OPTIMAL) => {
            (vk::AccessFlags::MEMORY_READ, vk::AccessFlags::TRANSFER_READ)
        }
        (vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR) => {
            (vk::AccessFlags::TRANSFER_READ, vk::AccessFlags::MEMORY_READ)
        }
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => {
            (vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::SHADER_READ)
        }
        (vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => {
            (vk::AccessFlags::SHADER_READ, vk::AccessFlags::TRANSFER_WRITE)
        }
        (vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL) => {
            (vk::AccessFlags::SHADER_READ, vk::AccessFlags::TRANSFER_READ)
        }
        (vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => {
            (vk::AccessFlags::TRANSFER_READ, vk::AccessFlags::SHADER_READ)
        }
        _ => (vk::AccessFlags::NONE, vk::AccessFlags::NONE),
    };
    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(old)
        .new_layout(new)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(src_mask)
        .dst_access_mask(dst_mask);
    device.cmd_pipeline_barrier(
        cb,
        vk::PipelineStageFlags::ALL_COMMANDS,
        vk::PipelineStageFlags::ALL_COMMANDS,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &[*barrier],
    );
}

fn to_u32s(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Fill the feedback texture with a flat mid-gray so the very first frame
/// already has valid, well-defined content (instead of UNDEFINED garbage) and
/// leaves it in `SHADER_READ_ONLY_OPTIMAL`, ready to be sampled as `texPrev`.
///
/// Runs on a one-shot command buffer and blocks until the copies finish.
unsafe fn seed_feedback(
    device: &ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    image: vk::Image,
    extent: vk::Extent2D,
) -> Result<(), String> {
    let allocated = device
        .allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::builder()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
        .map_err(first_error)?;
    let cb = allocated[0];

    device
        .begin_command_buffer(cb, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT).build())
        .map_err(first_error)?;

    transition_image(device, cb, image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
    let clear = vk::ClearColorValue {
        float32: [0.10, 0.12, 0.14, 1.0],
    };
    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    let ranges = [range];
    let _ = extent;
    device.cmd_clear_color_image(cb, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &clear, &ranges);
    transition_image(device, cb, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    device.end_command_buffer(cb).map_err(first_error)?;

    let fence = device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(first_error)?;
    let cbs = [cb];
    let submit = vk::SubmitInfo::builder().command_buffers(&cbs).build();
    device
        .queue_submit(queue, &[submit], fence)
        .map_err(first_error)?;
    device
        .wait_for_fences(&[fence], true, u64::MAX)
        .map_err(first_error)?;
    device.destroy_fence(fence, None);
    device.free_command_buffers(pool, &cbs);
    Ok(())
}
