// src/gpu_burn_to_bevy.rs

use std::borrow::Cow;
use std::marker::PhantomData;

use bevy::{
    prelude::*,
    asset::{load_internal_asset, uuid_handle},
    ecs::{query::QueryState, world::FromWorld},
    render::{
        graph::CameraDriverLabel,
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraph, RenderGraphContext, RenderLabel},
        render_resource::*,
        renderer::{RenderContext, RenderDevice},
        texture::GpuImage,
        Render,
        RenderApp,
        RenderSystems,
    },
};
use burn::tensor::{backend::Backend as BurnBackend, Tensor, TensorPrimitive};
use burn_fusion::client::FusionClient;
use burn_wgpu::{CubeBackend, FloatElement, IntElement, Wgpu as BurnWgpu, WgpuRuntime};

// from your bridge
use crate::{BindingDirection, BurnDevice, ExtractedGpuHandle};

// log target for easy filtering: RUST_LOG=bevy_burn::gpu_burn_to_bevy=info
const LOG: &str = "bevy_burn::gpu_burn_to_bevy";



#[derive(Component)]
pub struct CopyBindGroup {
    pub bg: wgpu::BindGroup,
    pub workgroups: [u32; 3],
}


pub trait BurnBevyPrepare<B: BurnBackend> {
    fn prepare_bind_group(
        tensor: &Tensor<B, 3>,
        burn_device: &BurnDevice,
        render_device: &RenderDevice,
        layout: &BindGroupLayout,
        texture: &wgpu::Texture,
        extent: Extent3d,
    ) -> Option<CopyBindGroup>;
}



impl<F, I> BurnBevyPrepare<BurnWgpu<F, I>> for ()
where
    F: FloatElement,
    I: IntElement,
{
    fn prepare_bind_group(
        tensor: &Tensor<BurnWgpu<F, I>, 3>,
        burn_device: &BurnDevice,
        render_device: &RenderDevice,
        layout: &BindGroupLayout,
        texture: &wgpu::Texture,
        extent: Extent3d,
    ) -> Option<CopyBindGroup> {
        let [h, w, c] = tensor.dims();
        if c != 4 {
            warn!(target: LOG, "expected f32 c==4 (rgba32f), got c={c}");
            return None;
        }

        let prim_fusion = tensor
            .clone()
            .to_device(burn_device)
            .into_primitive()
            .tensor();
        let fusion_client = prim_fusion.client.clone();
        let base = fusion_client
            .resolve_tensor_float::<CubeBackend<WgpuRuntime, F, I, u32>>(prim_fusion);
        let base_img: Tensor<CubeBackend<WgpuRuntime, F, I, u32>, 3> =
            Tensor::from_primitive(TensorPrimitive::Float(base));
        let prim2 = base_img.into_primitive().tensor();
        let client = &prim2.client;
        let res = client.get_resource(prim2.handle.clone().binding());
        client.flush();

        // src buffer must be 256B-aligned for binding as storage
        let src_off = res.resource().offset();
        if src_off & 0xFF != 0 {
            warn!(target: LOG, "tensor storage offset {} is not 256-aligned; cannot bind.", src_off);
            return None;
        }

        let src_binding = wgpu::BufferBinding {
            buffer: res.resource().buffer(),
            offset: src_off,
            size: std::num::NonZero::new(res.resource().size()).into(),
        };

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bg = render_device.wgpu_device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("buffer-rgba32f bg"),
            layout: layout.value(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(src_binding),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        });

        let copy_w = Ord::min(w as u32, extent.width);
        let copy_h = Ord::min(h as u32, extent.height);
        let gx = (copy_w + 15) / 16;
        let gy = (copy_h + 15) / 16;

        Some(CopyBindGroup {
            bg,
            workgroups: [gx, gy, 1],
        })
    }
}



#[derive(Resource)]
struct Rgba32fPipe {
    bgl: BindGroupLayout,
    id: CachedComputePipelineId,
}

impl FromWorld for Rgba32fPipe {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let bgl = device.create_bind_group_layout(
            "buffer-rgba32f bgl",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    binding_types::storage_buffer_read_only_sized(false, None),
                    binding_types::texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                ),
            ),
        );

        let id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("buffer-rgba32f pipe".into()),
            layout: vec![bgl.clone()],
            shader: COPY_SHADER_HANDLE,
            shader_defs: vec![],
            entry_point: Cow::from("main").into(),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: true,
        });

        Rgba32fPipe { bgl, id }
    }
}


pub struct BurnCopyNode<B: BurnBackend> {
    bg_q: QueryState<&'static CopyBindGroup>,
    _phantom: PhantomData<B>,
}

impl<B: BurnBackend> FromWorld for BurnCopyNode<B> {
    fn from_world(world: &mut World) -> Self {
        Self { bg_q: world.query(), _phantom: PhantomData }
    }
}


impl<B: BurnBackend> Node for BurnCopyNode<B>
{
    fn update(&mut self, world: &mut World) {
        self.bg_q.update_archetypes(world);
        debug!(target: LOG, "node.update: query archetypes updated");
    }

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_ctx: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let mut seen = 0usize;

        let cache = world.resource::<PipelineCache>();
        let pipe = world.resource::<Rgba32fPipe>();
        if let Some(p) = cache.get_compute_pipeline(pipe.id) {
            for bg in self.bg_q.iter_manual(world) {
                seen += 1;

                let mut pass = render_ctx
                    .command_encoder()
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("buffer-rgba32f write"),
                        ..Default::default()
                    });
                pass.set_pipeline(p);
                pass.set_bind_group(0, &bg.bg, &[]);
                pass.dispatch_workgroups(bg.workgroups[0], bg.workgroups[1], bg.workgroups[2]);
            }
        } else {
            debug!(target: LOG, "node.run: pipeline not ready yet");
        }

        debug!(
            target: LOG,
            "node.run: finished (seen={})",
            seen
        );

        Ok(())
    }
}


#[derive(Clone, Debug, Eq, Hash, PartialEq, RenderLabel)]
struct CopyNodeLabel;

pub struct GpuBurnToBevyPlugin<B: BurnBackend> {
    _phantom: PhantomData<B>,
}

impl<B: BurnBackend> Default for GpuBurnToBevyPlugin<B> {
    fn default() -> Self {
        Self { _phantom: PhantomData }
    }
}

const COPY_SHADER_HANDLE: Handle<Shader> = uuid_handle!("4477f827-8df7-4da1-906e-1f8e5ff64935");

impl<B> Plugin for GpuBurnToBevyPlugin<B>
where
    B: BurnBackend + 'static,
    (): BurnBevyPrepare<B>,
{
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            COPY_SHADER_HANDLE,
            "buffer_to_rgba32f.wgsl",
            Shader::from_wgsl
        );

        let render_app = app.sub_app_mut(RenderApp);

        render_app.init_resource::<Rgba32fPipe>();

        render_app.add_systems(
            Render,
            queue_copy_bind_groups::<B>.in_set(RenderSystems::Queue),
        );

        let node = BurnCopyNode::<B>::from_world(render_app.world_mut());
        let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
        graph.add_node(CopyNodeLabel, node);
        graph.add_node_edge(CopyNodeLabel, CameraDriverLabel);
    }
}


/// build per-entity bind groups from burn tensors (queue stage)
#[allow(clippy::type_complexity)]
fn queue_copy_bind_groups<B: BurnBackend>(
    mut commands: Commands,
    burn_device: Res<BurnDevice>,
    render_device: Res<RenderDevice>,
    pipe: Res<Rgba32fPipe>,
    images: Res<RenderAssets<GpuImage>>,
    q_handles: Query<(Entity, &ExtractedGpuHandle<B>)>,
) where
    (): BurnBevyPrepare<B>,
{
    for (entity, h) in q_handles.iter() {
        // only handle burn → bevy, and only when requested
        if h.direction != BindingDirection::BurnToBevy || !h.upload {
            continue;
        }

        let Some(gpu_image) = images.get(&h.image) else {
            debug!(target: LOG, "queue: no GpuImage for handle; skipping");
            continue;
        };

        let extent = Extent3d {
            width: gpu_image.size.width,
            height: gpu_image.size.height,
            depth_or_array_layers: 1,
        };

        // produce a bind group targeting the current texture
        if let Some(bg) = <() as BurnBevyPrepare<B>>::prepare_bind_group(
            &h.tensor,
            &burn_device,
            &render_device,
            &pipe.bgl,
            &gpu_image.texture,
            extent,
        ) {
            commands.entity(entity).insert(bg);
            trace!(target: LOG, "queue: bind group prepared for entity {:?}", entity);
        } else {
            // optional: remove any stale component if preparation failed this frame
            // commands.entity(entity).remove::<CopyBindGroup>();
            debug!(target: LOG, "queue: preparation failed (incompatible tensor/offset)");
        }
    }
}
