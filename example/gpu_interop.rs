use bevy::prelude::*;
use bevy::render::{
    render_asset::RenderAssetUsages,
    render_resource::*,
};
use burn_core::tensor::Int;
use burn_wgpu::Wgpu as BurnWgpu;
use bevy_burn::*;

type BB = BurnWgpu<f32, i32>;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(BevyBurnBridgePlugin::<BB>::default())
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut cmds: Commands,
    mut images: ResMut<Assets<Image>>,
) {
    // make a 256x256 gpu-only image
    let size = Extent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 1,
    };
    let mut img = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0; 16],
        TextureFormat::Rgba32Float,
        RenderAssetUsages::RENDER_WORLD, // no main-world CPU copy
    );
    img.texture_descriptor.usage |= TextureUsages::COPY_SRC
        | TextureUsages::COPY_DST
        | TextureUsages::TEXTURE_BINDING
        | TextureUsages::STORAGE_BINDING;
    let handle = images.add(img);

    // create a burn tensor gradient
    let h = size.height as usize;
    let w = size.width as usize;
    let dev = Default::default();
    let xs = burn_core::tensor::Tensor::<BB, 1, Int>::arange(0..(w * h * 4) as i64, &dev)
        .float()
        .div_scalar((w * h * 4) as f32);
    let rgba = xs.reshape([h, w, 4]);

    // attach gpu binding (render-world copy)
    cmds.spawn((
        ImageNode {
            image: handle.clone(),
            ..default()
        },
        BevyBurnHandle::<BB> {
            bevy_image: handle,
            tensor: rgba,
            upload: true,
            direction: BindingDirection::BurnToBevy,
            xfer: TransferKind::Gpu,
        },
    ));

    cmds.spawn(Camera2d);
}
