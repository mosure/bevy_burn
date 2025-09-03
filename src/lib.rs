#![recursion_limit = "256"]

use std::marker::PhantomData;

use bevy::{
    prelude::*,
    asset::Handle,
    render::{
        render_asset::{RenderAssetUsages, RenderAssets},
        render_resource::*,
        renderer::{
            RenderAdapter, RenderAdapterInfo, RenderDevice, RenderInstance, RenderQueue,
        },
        texture::GpuImage,
        Extract, ExtractSchedule,
        sync_world::{RenderEntity, SyncToRenderWorld},
        Render, RenderApp, RenderSystems,
    },
    utils::WgpuWrapper,
};
use burn::{
    backend::wgpu::{
        RuntimeOptions as BurnRuntimeOptions, WgpuDevice as BurnWgpuDevice,
        WgpuSetup as BurnWgpuSetup, init_device as init_burn_device,
    },
    prelude::Backend,
    tensor::{Tensor, Int},
};

pub mod gpu_burn_to_bevy;
use gpu_burn_to_bevy::{
    BurnBevyPrepare,
    GpuBurnToBevyPlugin,
};



#[derive(Resource, Deref, DerefMut, Clone, Debug, Hash, PartialEq, Eq)]
pub struct BurnDevice(BurnWgpuDevice);


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingDirection {
    BurnToBevy,
    BevyToBurn,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferKind {
    Cpu,
    Gpu,
}

#[derive(Component, Clone)]
pub struct BevyBurnHandle<B: Backend> {
    pub bevy_image: Handle<Image>,
    pub tensor: Tensor<B, 3>,
    pub upload: bool,
    pub direction: BindingDirection,
    pub xfer: TransferKind,
}

impl<B: Backend> Default for BevyBurnHandle<B> {
    fn default() -> Self {
        Self {
            bevy_image: Handle::default(),
            tensor: Tensor::<B, 3>::zeros([0, 0, 0], &Default::default()),
            upload: true,
            direction: BindingDirection::BurnToBevy,
            xfer: TransferKind::Cpu,
        }
    }
}

#[derive(Default)]
pub struct BevyBurnBridgePlugin<B: Backend> {
    pub cpu_only: bool,
    pub _marker: PhantomData<B>,
}

impl<B: Backend> Plugin for BevyBurnBridgePlugin<B>
where
    B: Backend + 'static,
    (): BurnBevyPrepare<B>,
{
    fn build(&self, app: &mut App) {
        // cpu path in main world
        app.add_systems(Update, (bevy_to_burn_update::<B>, burn_to_bevy_update::<B>));

        if self.cpu_only {
            return;
        }

        app.add_systems(First, ensure_sync_to_render_world::<B>);
    }

    fn finish(&self, app: &mut App) {
        if self.cpu_only {
            return;
        }

        let render_app = app
            .get_sub_app_mut(RenderApp)
            .expect("Failed to setup Burn plugin: RenderApp not found");

        let burn_device = {
            let bevy_adapter = render_app.world().resource::<RenderAdapter>();
            let wgpu_adapter = unwrap_wgpu_wrapper(&bevy_adapter.0);

            let bevy_device = render_app.world().resource::<RenderDevice>();
            let wgpu_device = bevy_device.wgpu_device().clone();

            let bevy_instance = render_app.world().resource::<RenderInstance>();
            let wgpu_instance = unwrap_wgpu_wrapper(&bevy_instance.0);

            let bevy_queue = render_app.world().resource::<RenderQueue>();
            let wgpu_queue = unwrap_wgpu_wrapper(&bevy_queue.0);

            let render_adapter_info = render_app.world().resource::<RenderAdapterInfo>();
            let wgpu_backend = render_adapter_info.backend;

            let wgpu_setup = BurnWgpuSetup {
                adapter: wgpu_adapter,
                device: wgpu_device,
                instance: wgpu_instance,
                queue: wgpu_queue,
                backend: wgpu_backend,
            };

            let runtime_options = BurnRuntimeOptions::default();
            let burn_device = init_burn_device(wgpu_setup, runtime_options);

            render_app
                .add_systems(ExtractSchedule, extract_gpu_handles::<B>)
                .add_systems(
                    Render,
                    gpu_bevy_to_burn::<B>.in_set(RenderSystems::Queue),
                );

            burn_device
        };

        render_app.insert_resource(BurnDevice(burn_device.clone()));
        app.insert_resource(BurnDevice(burn_device));

        app.add_plugins(GpuBurnToBevyPlugin::<B>::default());
    }
}


/// make sure entities we care about are synced into the render world
fn ensure_sync_to_render_world<B: Backend>(
    mut commands: Commands,
    q: Query<(Entity, Option<&SyncToRenderWorld>), With<BevyBurnHandle<B>>>,
) {
    for (e, synced) in &q {
        if synced.is_none() {
            commands.entity(e).insert(SyncToRenderWorld);
        }
    }
}


fn unwrap_wgpu_wrapper<T: Clone>(wrapper: &WgpuWrapper<T>) -> T {
    <WgpuWrapper<T> as Clone>::clone(wrapper).into_inner()
}


// ---------- cpu path ----------

fn bevy_to_burn_update<B: Backend>(
    images: Res<Assets<Image>>,
    mut q: Query<&mut BevyBurnHandle<B>>,
) {
    for mut handle in &mut q {
        if handle.direction != BindingDirection::BevyToBurn || handle.xfer != TransferKind::Cpu {
            continue;
        }

        if handle.upload {
            let Some(img) = images.get(&handle.bevy_image) else { continue };
            let size = img.size();
            let (width, height) = (size.x as usize, size.y as usize);
            let Some(raw) = &img.data else { continue };
            if raw.len() != width * height * 4 {
                continue;
            }

            let device = handle.tensor.device();

            let raw_tensor = Tensor::<B, 1, Int>::from_data(&raw[..], &device);
            let float_tensor = raw_tensor.float();
            let normalised = float_tensor.div_scalar(255.0);
            let new_tensor = normalised.reshape([height, width, 4]);

            handle.tensor = new_tensor;
            handle.upload = false;
        }
    }
}

fn burn_to_bevy_update<B: Backend>(
    mut images: ResMut<Assets<Image>>,
    mut q: Query<&mut BevyBurnHandle<B>>,
) {
    for mut handle in &mut q {
        if handle.direction != BindingDirection::BurnToBevy || handle.xfer != TransferKind::Cpu {
            continue;
        }

        if handle.upload {
            let data = handle.tensor.to_data();
            let Ok(floats) = data.to_vec::<f32>() else { continue };

            let mut bytes = Vec::with_capacity(floats.len());
            for &f in floats.iter() {
                let v = f.clamp(0.0, 1.0) * 255.0;
                bytes.push(v.round() as u8);
            }

            if let Some(img) = images.get_mut(&handle.bevy_image) {
                if img.height() != handle.tensor.shape().dims[0] as u32
                    || img.width() != handle.tensor.shape().dims[1] as u32
                {
                    info!(
                        "resizing image from {}x{} to {}x{}",
                        img.width(),
                        img.height(),
                        handle.tensor.shape().dims[1],
                        handle.tensor.shape().dims[0]
                    );

                    img.resize(Extent3d {
                        height: handle.tensor.shape().dims[0] as u32,
                        width: handle.tensor.shape().dims[1] as u32,
                        depth_or_array_layers: 1,
                    });
                }

                match img.data {
                    Some(ref mut d) => {
                        if d.len() == bytes.len() {
                            d.copy_from_slice(&bytes);
                        } else {
                            *d = bytes;
                        }
                    }
                    None => img.data = Some(bytes),
                }
            } else {
                let img = Image::new_fill(
                    Extent3d {
                        width: handle.tensor.shape().dims[1] as u32,
                        height: handle.tensor.shape().dims[0] as u32,
                        depth_or_array_layers: 1,
                    },
                    TextureDimension::D2,
                    &bytes,
                    TextureFormat::Rgba8UnormSrgb,
                    RenderAssetUsages::default(),
                );
                handle.bevy_image = images.add(img);
            }

            handle.upload = false;
        }
    }
}

// ---------- gpu path (render world) ----------

#[derive(Component, Clone, Debug)]
struct ExtractedGpuHandle<B: Backend> {
    image: Handle<Image>,
    tensor: Tensor<B, 3>,
    direction: BindingDirection,
    upload: bool,
}

fn extract_gpu_handles<B: Backend>(
    mut commands: Commands,
    q: Extract<Query<(RenderEntity, &BevyBurnHandle<B>)>>,
) {
    let mut seen = 0usize;

    for (render_entity, h) in &q {
        seen += 1;
        if h.xfer != TransferKind::Gpu {
            continue;
        }

        commands.entity(render_entity).insert(ExtractedGpuHandle::<B> {
            image: h.bevy_image.clone(),
            tensor: h.tensor.clone(),
            direction: h.direction,
            upload: h.upload,
        });

        seen += 1;
    }

    debug!(
        target: "bevy_burn::extract",
        "extract_gpu_handles: seen={}",
        seen
    );
}



#[inline]
fn padded_bytes_per_row(width: u32, bytes_per_pixel: u32) -> u32 {
    // wgpu COPY_BYTES_PER_ROW_ALIGNMENT is 256
    const ALIGN: u32 = 256;
    let row = width * bytes_per_pixel;
    ((row + ALIGN - 1) / ALIGN) * ALIGN
}

/// bevy image -> burn (gpu-side). schedules copy + readback; blocks to map.
fn gpu_bevy_to_burn<B: Backend>(
    // burn_device: Res<BurnDevice>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    images: Res<RenderAssets<GpuImage>>,
    mut q: Query<&mut ExtractedGpuHandle<B>>,
) {
    for mut h in &mut q {
        if h.direction != BindingDirection::BevyToBurn || !h.upload {
            continue;
        }
        let Some(gpu_image) = images.get(&h.image) else { continue };

        let bpp = 4u32; // RGBA8
        let extent = Extent3d {
            width: gpu_image.size.width,
            height: gpu_image.size.height,
            depth_or_array_layers: 1,
        };
        let row_bytes = extent.width * bpp;
        let padded_row = padded_bytes_per_row(extent.width, bpp);
        let total = (padded_row as u64) * (extent.height as u64);

        // staging buffer
        let staging = render_device.create_buffer(&BufferDescriptor {
            label: Some("bevy_burn.gpu_t2b.staging"),
            size: total,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // copy texture -> buffer (wgpu v25)
        let mut enc =
            render_device.create_command_encoder(&CommandEncoderDescriptor { label: Some("bevy_burn.copy_tex_to_buf") });
        enc.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &gpu_image.texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &staging,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(extent.height),
                },
            },
            extent,
        );
        render_queue.submit([enc.finish()]);

        // map & normalize to tensor
        staging.slice(..).map_async(MapMode::Read, |_| {});
        let _ = render_device.wgpu_device().poll(PollType::Wait);

        let view = staging.slice(..).get_mapped_range();
        // strip row padding
        let mut compact =
            Vec::with_capacity((row_bytes as usize) * (extent.height as usize));
        for y in 0..extent.height as usize {
            let src_off = y * padded_row as usize;
            compact.extend_from_slice(&view[src_off..src_off + row_bytes as usize]);
        }

        let device = h.tensor.device();
        let raw_tensor = Tensor::<B, 1, Int>::from_data(&compact[..], &device);
        let float_tensor = raw_tensor.float();
        let normalised = float_tensor.div_scalar(255.0);
        h.tensor = normalised.reshape([extent.height as usize, extent.width as usize, 4]);

        staging.unmap();
        h.upload = false; // render-world flag
    }
}

// ---------- tests ----------

#[cfg(test)]
mod cpu_tests {
    use super::*;
    use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
    use burn_wgpu::Wgpu;

    type BurnBackend = Wgpu<f32, i32>;

    fn default_app() -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.insert_resource(Assets::<Image>::default());
        app.add_plugins(BevyBurnBridgePlugin::<BurnBackend>::default());
        app
    }

    #[test]
    fn bevy_to_burn_cpu_1x1() {
        let mut app = default_app();

        let pixel = [255, 128, 0, 255];
        let img = Image::new_fill(
            Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            TextureDimension::D2,
            &pixel,
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::default(),
        );
        let handle = {
            let mut images = app.world_mut().resource_mut::<Assets<Image>>();
            images.add(img)
        };

        let tensor = Tensor::<BurnBackend, 3>::zeros([1, 1, 4], &Default::default());
        let entity = app
            .world_mut()
            .spawn(BevyBurnHandle {
                bevy_image: handle.clone(),
                tensor,
                upload: true,
                direction: BindingDirection::BevyToBurn,
                xfer: TransferKind::Cpu,
            })
            .id();

        app.update();
        let comp = app.world().get::<BevyBurnHandle<BurnBackend>>(entity).unwrap();
        let data = comp.tensor.to_data();
        let floats: Vec<f32> = data.to_vec::<f32>().unwrap();

        let max_err = pixel
            .iter()
            .enumerate()
            .map(|(i, &x)| (x as f32 / 255.0 - floats[i]).abs())
            .fold(0.0, f32::max);

        assert!(max_err < 0.0001, "max error: {}", max_err);
    }

    #[test]
    fn burn_to_bevy_cpu_1x1() {
        let mut app = default_app();

        let img = Image::new_fill(
            Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            TextureDimension::D2,
            &[0, 0, 0, 0],
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::default(),
        );
        let handle = {
            let mut images = app.world_mut().resource_mut::<Assets<Image>>();
            images.add(img)
        };

        let tensor =
            Tensor::<BurnBackend, 3>::from_data([[[0.0f32, 0.5, 1.0, 1.0]]], &Default::default());
        app.world_mut().spawn(BevyBurnHandle {
            bevy_image: handle.clone(),
            tensor,
            upload: true,
            direction: BindingDirection::BurnToBevy,
            xfer: TransferKind::Cpu,
        });

        app.update();
        let images = app.world().resource::<Assets<Image>>();
        let updated = images.get(&handle).unwrap();
        assert_eq!(updated.data.as_deref().unwrap(), &[0, 128, 255, 255]);
    }
}


// #[cfg(test)]
// mod gpu_tests {
//     use super::*;
//     use bevy::prelude::*;
//     use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
//     use burn_wgpu::Wgpu;

//     type BurnBackend = Wgpu<f32, i32>;

//     fn render_app() -> App {
//         let mut app = App::new();
//         // Per request: use the full DefaultPlugins stack for GPU tests.
//         app.add_plugins(DefaultPlugins);
//         // Our bridge
//         app.add_plugins(BevyBurnBridgePlugin::<BurnBackend>::default());
//         app
//     }

//     #[test]
//     #[ignore = "gpu-dependent; run with `cargo test -- --ignored`"]
//     fn burn_to_bevy_gpu_smoke() {
//         let mut app = render_app();

//         let size = Extent3d { width: 1, height: 1, depth_or_array_layers: 1 };
//         let mut img = Image::new_fill(
//             size,
//             TextureDimension::D2,
//             &[0, 0, 0, 255],
//             TextureFormat::Rgba8UnormSrgb,
//             // Ensure render-world residency so the pipeline prepares GpuImage.
//             RenderAssetUsages::RENDER_WORLD,
//         );
//         // Allow our copy passes.
//         img.texture_descriptor.usage |=
//             TextureUsages::COPY_SRC | TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;

//         let handle = {
//             let mut images = app.world_mut().resource_mut::<Assets<Image>>();
//             images.add(img)
//         };

//         let tensor =
//             Tensor::<BurnBackend, 3>::from_data([[[0.0f32, 0.5, 1.0, 1.0]]], &Default::default());

//         app.world_mut().spawn(BevyBurnHandle {
//             bevy_image: handle.clone(),
//             tensor,
//             upload: true,
//             direction: BindingDirection::BurnToBevy,
//             xfer: TransferKind::Gpu,
//         });

//         // advance frames to let extract/prepare/queue run
//         for _ in 0..20 {
//             app.update();
//         }

//         // verify render-world upload completed
//         let render_world = app.get_sub_app_mut(RenderApp).unwrap().world_mut();
//         let mut q = render_world.query::<&ExtractedGpuHandle<BurnBackend>>();
//         let ok = q.iter(render_world).any(|h| {
//             h.direction == BindingDirection::BurnToBevy && h.image == handle && !h.upload
//         });
//         assert!(ok, "gpu burn->bevy copy didn't complete");
//     }

//     #[test]
//     #[ignore = "gpu-dependent; run with `cargo test -- --ignored`"]
//     fn bevy_to_burn_gpu_1x1() {
//         let mut app = render_app();

//         let pixel = [255u8, 128, 0, 255];
//         let size = Extent3d { width: 1, height: 1, depth_or_array_layers: 1 };
//         let mut img = Image::new_fill(
//             size,
//             TextureDimension::D2,
//             &pixel,
//             TextureFormat::Rgba8UnormSrgb,
//             RenderAssetUsages::RENDER_WORLD,
//         );
//         img.texture_descriptor.usage |=
//             TextureUsages::COPY_SRC | TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;

//         let handle = {
//             let mut images = app.world_mut().resource_mut::<Assets<Image>>();
//             images.add(img)
//         };

//         let tensor = Tensor::<BurnBackend, 3>::zeros([1, 1, 4], &Default::default());

//         app.world_mut().spawn(BevyBurnHandle {
//             bevy_image: handle.clone(),
//             tensor,
//             upload: true,
//             direction: BindingDirection::BevyToBurn,
//             xfer: TransferKind::Gpu,
//         });

//         for _ in 0..24 {
//             app.update();
//         }

//         // read back tensor produced in render world and check values
//         let render_world = app.get_sub_app_mut(RenderApp).unwrap().world_mut();
//         let mut q = render_world.query::<&ExtractedGpuHandle<BurnBackend>>();
//         let mut found = false;
//         for h in q.iter(render_world) {
//             if h.direction != BindingDirection::BevyToBurn || h.image != handle || h.upload {
//                 continue;
//             }
//             let data = h.tensor.to_data();
//             let floats: Vec<f32> = data.to_vec::<f32>().unwrap();
//             let expected = [
//                 pixel[0] as f32 / 255.0,
//                 pixel[1] as f32 / 255.0,
//                 pixel[2] as f32 / 255.0,
//                 pixel[3] as f32 / 255.0,
//             ];
//             let max_err =
//                 (0..4).map(|i| (floats[i] - expected[i]).abs()).fold(0.0, f32::max);
//             assert!(max_err < 0.02, "max error: {}", max_err);
//             found = true;
//             break;
//         }
//         assert!(found, "no extracted gpu handle finished upload");
//     }
// }
