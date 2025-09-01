// src/gpu_burn_to_bevy.rs

use std::{borrow::Cow, marker::PhantomData, num::NonZeroU64};

use bevy::{
    asset::{load_internal_asset, uuid_handle},
    ecs::{
        query::QueryState,
        world::FromWorld,
    },
    prelude::*,
    render::{
        graph::CameraDriverLabel,
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraph, RenderGraphContext, RenderLabel},
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
        Render, RenderApp, RenderSystems,
    },
};
use burn_autodiff::Autodiff;
use burn_fusion::client::FusionClient;
use burn::tensor::{backend::Backend as BurnBackend, Tensor};
use burn_wgpu::{
    CubeBackend,
    FloatElement,
    IntElement,
    Wgpu as BurnWgpu,
    WgpuResource,
    WgpuRuntime,
};
use wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

// from your bridge
use crate::{BindingDirection, ExtractedGpuHandle};

// log target for easy filtering: RUST_LOG=bevy_burn::gpu_burn_to_bevy=info
const LOG: &str = "bevy_burn::gpu_burn_to_bevy";

// ---------- shader handle (internal asset) ----------
pub const PACK_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("d522e455-de07-468c-982d-678cf46ac89a");

// ---------- helpers / resources ----------

#[inline]
fn padded_bytes_per_row(width: u32, bpp: u32) -> u32 {
    let unpadded = width * bpp;
    let align = COPY_BYTES_PER_ROW_ALIGNMENT; // 256
    ((unpadded + align - 1) / align) * align
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    width: u32,
    height: u32,
    padded_words_per_row: u32,
    _pad: u32,
}

#[derive(Resource)]
struct PackPipeline {
    layout: BindGroupLayout,
    pipeline_id: CachedComputePipelineId,
}

// pub trait AsWgpuRes<B: BurnBackend> {
//     fn get(tensor: &Tensor<B, 3>) -> Option<WgpuResource>;
// }

// impl<F, I> AsWgpuRes<Autodiff<BurnWgpu<F, I>>> for ()
// where
//     F: FloatElement,
//     I: IntElement,
// {
//     fn get(t: &Tensor<Autodiff<BurnWgpu<F, I>>, 3>) -> Option<WgpuResource> {
//         let inner = t.clone().inner();
//         as_wgpu_from_wgpu(&inner)
//     }
// }

// impl<F, I> AsWgpuRes<BurnWgpu<F, I>> for ()
// where
//     F: FloatElement,
//     I: IntElement,
// {
//     fn get(t: &Tensor<BurnWgpu<F, I>, 3>) -> Option<WgpuResource> {
//         as_wgpu_from_wgpu(t)
//     }
// }

// fn as_wgpu_from_wgpu<F, I>(t: &Tensor<BurnWgpu<F, I>, 3>) -> Option<WgpuResource>
// where
//     F: FloatElement,
//     I: IntElement,
// {
//     use burn::tensor::TensorPrimitive;

//     // Turn the high-level tensor into its primitive (for Wgpu this is a FusionTensor).
//     let prim = t.clone().into_primitive();
//     let TensorPrimitive::Float(fusion_tensor) = prim else {
//         info!(target: LOG, "burn_tensor_as_wgpu_resource: non-float tensor; skipping");
//         return None;
//     };

//     // Resolve Fusion → Cube primitive for the *concrete* Cube backend we run (WgpuRuntime).
//     // We assume the default bool element (u32) for the Wgpu alias.
//     let cube = fusion_tensor
//         .client
//         .resolve_tensor_float::<CubeBackend<WgpuRuntime, F, I, u32>>(fusion_tensor);

//     // We require a linear backing buffer (no overlaps/holes).
//     if !cube.is_contiguous_buffer() {
//         info!(target: LOG, "burn_tensor_as_wgpu_resource: non-contiguous buffer; skipping");
//         return None;
//     }

//     // Grab the storage handle and fetch the wgpu-side resource view.
//     let h = cube.as_handle_ref(); // gives access to { handle.memory, ... }
//     let mut storage = cube.client.storage(); // WgpuStorage
//     let res = storage.get(&h.handle.memory); // -> WgpuResource { buffer, offset, size }

//     Some(res)
// }

// ----- public entry point (keeps your original signature) -----

fn burn_tensor_as_wgpu_resource<B: BurnBackend>(_tensor: &Tensor<B, 3>) -> Option<WgpuResource>
// where
//     (): AsWgpuRes<B>,
{
    info!(
        target: LOG,
        "burn_tensor_as_wgpu_resource: unavailable on this Burn version; \
         public API doesn't expose backend storage (skipping GPU→GPU fast path)"
    );
    None
    // <() as AsWgpuRes<B>>::get(tensor)
}

// ---------- compute node (uses query state like your radix example) ----------

pub struct BurnCopyNode<B: BurnBackend> {
    handles_q: QueryState<&'static ExtractedGpuHandle<B>>,
    initialized: bool,
    _phantom: PhantomData<B>,
}

impl<B: BurnBackend> FromWorld for BurnCopyNode<B>
// where
//     (): AsWgpuRes<B>,
{
    fn from_world(world: &mut World) -> Self {
        info!(target: LOG, "node: created (FromWorld)");
        Self {
            handles_q: world.query(),
            initialized: false,
            _phantom: PhantomData,
        }
    }
}

impl<B: BurnBackend> Node for BurnCopyNode<B>
// where
//     (): AsWgpuRes<B>,
{
    fn update(&mut self, world: &mut World) {
        // called every render frame before run()
        if !self.initialized {
            let cache = world.resource::<PipelineCache>();
            let pipe = world.resource::<PackPipeline>();
            match cache.get_compute_pipeline_state(pipe.pipeline_id) {
                CachedPipelineState::Ok(_) => {
                    self.initialized = true;
                    info!(target: LOG, "node.update: pipeline ready -> initialized=true");
                }
                CachedPipelineState::Queued => {
                    info!(target: LOG, "node.update: pipeline still queued");
                    return;
                }
                CachedPipelineState::Err(err) => {
                    error!(target: LOG, "node.update: pipeline error: {err:?}");
                    return;
                }
                CachedPipelineState::Creating(_) => {
                    info!(target: LOG, "node.update: pipeline not found yet");
                    return;
                }
            }
        }

        self.handles_q.update_archetypes(world);
        debug!(target: LOG, "node.update: query archetypes updated");
    }

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_ctx: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        if !self.initialized {
            info!(target: LOG, "node.run: not initialized; skipping");
            return Ok(());
        }

        let render_device = world.resource::<RenderDevice>();
        let render_queue = world.resource::<RenderQueue>();
        let images = world.resource::<RenderAssets<GpuImage>>();
        let pipe = world.resource::<PackPipeline>();
        let cache = world.resource::<PipelineCache>();

        let Some(pipeline) = cache.get_compute_pipeline(pipe.pipeline_id) else {
            info!(target: LOG, "node.run: pipeline not fetched from cache; skipping");
            return Ok(());
        };

        let mut seen = 0usize;
        let mut processed = 0usize;
        let bpp = 4;

        for h in self.handles_q.iter_manual(world) {
            seen += 1;

            if h.direction != BindingDirection::BurnToBevy {
                debug!(target: LOG, "node.run: entity skipped (direction != BurnToBevy)");
                continue;
            }
            if !h.upload {
                debug!(target: LOG, "node.run: entity skipped (upload=false)");
                continue;
            }

            let Some(gpu_image) = images.get(&h.image) else {
                debug!(target: LOG, "node.run: no GpuImage for handle; skipping");
                continue;
            };

            let extent = Extent3d {
                width: gpu_image.size.width,
                height: gpu_image.size.height,
                depth_or_array_layers: 1,
            };

            let Some(src) = burn_tensor_as_wgpu_resource(&h.tensor) else {
                info!(target: LOG, "node.run: missing gpu resource for burn tensor (stub returns None); skipping");
                continue;
            };

            // expect width*height*4 floats (rgba32f)
            let expected_words = (extent.width as u64) * (extent.height as u64) * 4;
            if src.size() / 4 != expected_words {
                warn!(
                    target: LOG,
                    "node.run: size mismatch (src_words={}, expected={}); skipping",
                    src.size() / 4,
                    expected_words
                );
                continue;
            }

            let padded_row = padded_bytes_per_row(extent.width, bpp);
            let total_bytes = (padded_row as u64) * (extent.height as u64);

            info!(
                target: LOG,
                "node.run: dispatch copy for {}x{} (bytes_per_row={}, total_bytes={})",
                extent.width,
                extent.height,
                padded_row,
                total_bytes
            );

            // scratch gpu buffer for packed rgba8 (with row padding)
            let scratch = render_device.create_buffer(&BufferDescriptor {
                label: Some("bevy_burn.pack.scratch"),
                size: total_bytes,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            // tiny uniform with layout params
            let params = Params {
                width: extent.width,
                height: extent.height,
                padded_words_per_row: padded_row / 4,
                _pad: 0,
            };

            let params_buf = render_device.create_buffer(&BufferDescriptor {
                label: Some("bevy_burn.pack.params"),
                size: std::mem::size_of::<Params>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            render_queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

            // bind group
            let src_binding = BufferBinding {
                buffer: src.buffer(),
                offset: src.offset(),
                size: NonZeroU64::new(src.size()),
            };
            let dst_binding = BufferBinding {
                buffer: &scratch,
                offset: 0,
                size: NonZeroU64::new(total_bytes),
            };

            let bg = render_device.create_bind_group(
                Some("bevy_burn.pack.bg"),
                &pipe.layout,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer(src_binding),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Buffer(dst_binding),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &params_buf,
                            offset: 0,
                            size: NonZeroU64::new(std::mem::size_of::<Params>() as u64),
                        }),
                    },
                ],
            );

            // record compute + copy
            let enc = render_ctx.command_encoder();

            enc.clear_buffer(&scratch, 0, None);

            {
                let mut cpass = enc.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("bevy_burn.pack.cpass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(pipeline);
                cpass.set_bind_group(0, &bg, &[]);
                let wg_x = (extent.width + 15) / 16;
                let wg_y = (extent.height + 15) / 16;
                info!(target: LOG, "node.run: dispatch_workgroups({}, {}, 1)", wg_x, wg_y);
                cpass.dispatch_workgroups(wg_x, wg_y, 1);
            }

            enc.copy_buffer_to_texture(
                TexelCopyBufferInfo {
                    buffer: &scratch,
                    layout: TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_row),
                        rows_per_image: Some(extent.height),
                    },
                },
                TexelCopyTextureInfo {
                    texture: &gpu_image.texture,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                extent,
            );

            info!(target: LOG, "node.run: copy_buffer_to_texture recorded");
            // we don't flip h.upload=false here (immutable world); do that in your queue/extract flow
            processed += 1;
        }

        info!(
            target: LOG,
            "node.run: finished (seen={}, processed={})",
            seen,
            processed
        );

        Ok(())
    }
}

// ---------- plugin ----------

#[derive(Clone, Debug, Eq, Hash, PartialEq, RenderLabel)]
struct CopyNodeLabel;

pub struct GpuBurnToBevyPlugin<B: BurnBackend>
// where
//     (): AsWgpuRes<B>,
{
    _phantom: PhantomData<B>,
}

impl<B: BurnBackend> Default for GpuBurnToBevyPlugin<B>
// where
//     (): AsWgpuRes<B>,
{
    fn default() -> Self {
        Self { _phantom: PhantomData }
    }
}

impl<B: BurnBackend + 'static> Plugin for GpuBurnToBevyPlugin<B>
// where
//     (): AsWgpuRes<B>,
{
    fn build(&self, app: &mut App) {
        info!(target: LOG, "plugin.build: loading internal shader asset");
        load_internal_asset!(
            app,
            PACK_SHADER_HANDLE,
            "pack_rgba_to_bytes.wgsl",
            Shader::from_wgsl
        );

        let render_app = app.sub_app_mut(RenderApp);

        info!(target: LOG, "plugin.build: registering prepare system");
        render_app.add_systems(
            Render,
            init_pack_pipeline.in_set(RenderSystems::Prepare),
        );

        info!(target: LOG, "plugin.build: adding render graph node + edge");
        let node = BurnCopyNode::<B>::from_world(render_app.world_mut());
        let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
        graph.add_node(CopyNodeLabel, node);
        graph.add_node_edge(CopyNodeLabel, CameraDriverLabel);
    }
}


fn init_pack_pipeline(
    mut commands: Commands,
    rd: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    maybe_existing: Option<Res<PackPipeline>>,
) {
    if maybe_existing.is_some() {
        return;
    }
    info!(target: LOG, "prepare: init_pack_pipeline (create layout + queue pipeline)");

    let layout = rd.create_bind_group_layout(
        Some("bevy_burn.pack.bgl"),
        &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    );

    let pipeline_id = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Cow::from("bevy_burn.pack.pipeline").into(),
        layout: vec![layout.clone()],
        push_constant_ranges: vec![],
        shader: PACK_SHADER_HANDLE,
        entry_point: Some(Cow::Borrowed("main")),
        zero_initialize_workgroup_memory: false,
        shader_defs: vec![],
    });

    info!(target: LOG, "prepare: queued compute pipeline id = {:?}", pipeline_id);

    commands.insert_resource(PackPipeline { layout, pipeline_id });
    info!(target: LOG, "prepare: PackPipeline resource inserted");
}
