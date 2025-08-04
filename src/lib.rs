#![recursion_limit = "256"]

use std::marker::PhantomData;

use bevy::{
    prelude::*,
    asset::Handle,
    render::{
        render_asset::RenderAssetUsages,
        render_resource::*,
    },
};
use burn_core::tensor::{backend::Backend, Int, Tensor};


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingDirection {
    BurnToBevy,
    BevyToBurn,
}

#[derive(Component, Clone)]
pub struct BevyBurnHandle<B: Backend> {
    pub bevy_image: Handle<Image>,
    pub tensor: Tensor<B, 3>,
    pub upload: bool,
    pub direction: BindingDirection,
}

impl<B: Backend> Default for BevyBurnHandle<B> {
    fn default() -> Self {
        Self {
            bevy_image: Handle::default(),
            tensor: Tensor::<B, 3>::zeros([0, 0, 0], &Default::default()),
            upload: true,
            direction: BindingDirection::BurnToBevy,
        }
    }
}


#[derive(Default)]
pub struct BevyBurnBridgePlugin<B: Backend> {
    _marker: PhantomData<B>,
}

impl<B: Backend> Plugin for BevyBurnBridgePlugin<B> {
    fn build(&self, app: &mut App) {
        // TODO: if bevy multi-world is supported, move to burn world (or prioritize render world copies)
        app.add_systems(
            Update,
            (
                bevy_to_burn_update::<B>,
                burn_to_bevy_update::<B>,
            ),
        );

        // TODO: support GPU <-> GPU specialization
    }
}


fn bevy_to_burn_update<B: Backend>(
    images: Res<Assets<Image>>,
    mut q: Query<&mut BevyBurnHandle<B>>,
) {
    for mut handle in &mut q {
        if handle.direction != BindingDirection::BevyToBurn {
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
        if handle.direction != BindingDirection::BurnToBevy {
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
                if img.height() != handle.tensor.shape().dims[0] as u32 ||
                   img.width() != handle.tensor.shape().dims[1] as u32
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



#[cfg(test)]
mod tests {
    use super::*;
    use bevy::render::{
        render_asset::RenderAssetUsages,
        render_resource::{Extent3d, TextureDimension, TextureFormat},
    };
    use burn_autodiff::Autodiff;
    use burn_wgpu::Wgpu;

    type BurnBackend = Autodiff<Wgpu<f32, i32>>;

    fn default_app() -> App {
        let mut app = App::new();

        app.add_plugins(MinimalPlugins);
        app.insert_resource(Assets::<Image>::default());
        app.add_plugins(BevyBurnBridgePlugin::<BurnBackend>::default());

        app
    }


    #[test]
    fn test_bevy_to_burn() {
        let mut app = default_app();

        let pixel = [255, 128, 0, 255];
        let img = Image::new_fill(
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            &pixel,
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::default(),
        );
        let handle = {
            let mut images = app.world_mut().resource_mut::<Assets<Image>>();
            images.add(img)
        };

        let tensor = Tensor::<BurnBackend, 3>::zeros(
            [1, 1, 4],
            &Default::default(),
        );
        let entity = app
            .world_mut()
            .spawn(BevyBurnHandle {
                bevy_image: handle.clone(),
                tensor,
                upload: true,
                direction: BindingDirection::BevyToBurn,
            })
            .id();

        app.update();
        let comp = app
            .world()
            .get::<BevyBurnHandle<BurnBackend>>(entity)
            .unwrap();
        let data = comp.tensor.to_data();
        let floats: Vec<f32> = data.to_vec::<f32>().unwrap();

        let max_err = pixel.iter()
            .enumerate()
            .map(|(i, &x)| (x as f32 / 255.0 - floats[i]).abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        assert!(max_err < 0.0001, "max error: {}", max_err);
    }

    #[test]
    fn test_burn_to_bevy() {
        let mut app = default_app();

        let img = Image::new_fill(
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            &[0, 0, 0, 0],
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::default(),
        );
        let handle = {
            let mut images = app.world_mut().resource_mut::<Assets<Image>>();
            images.add(img)
        };

        let tensor = Tensor::<BurnBackend, 3>::from_data(
            [[[0.0f32, 0.5, 1.0, 1.0]]],
            &Default::default(),
        );
        app
            .world_mut()
            .spawn(BevyBurnHandle {
                bevy_image: handle.clone(),
                tensor,
                upload: true,
                direction: BindingDirection::BurnToBevy,
            });

        app.update();
        let images = app.world().resource::<Assets<Image>>();
        let updated = images.get(&handle).unwrap();
        assert_eq!(updated.data.as_deref().unwrap(), &[0, 128, 255, 255]);
    }
}
