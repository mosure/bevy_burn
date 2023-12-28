use std::io::ErrorKind;

use bevy::{
    prelude::*,
    asset::{
        AssetLoader,
        AsyncReadExt,
        io::Reader,
        LoadContext,
    },
    render::RenderApp,
    utils::BoxedFuture,
};

use burn::{
    config::Config,
    module::Module,
    tensor::backend::Backend,
};


#[derive(
    Asset,
    Debug,
    Default,
    Reflect,
)]
pub struct BurnModel;


#[derive(Default)]
pub struct ModelLoader;

impl AssetLoader for ModelLoader {
    type Asset = ();
    type Settings = ();
    type Error = std::io::Error;

    fn load<'a>(
        &'a self,
        reader: &'a mut Reader,
        _settings: &'a Self::Settings,
        load_context: &'a mut LoadContext,
    ) -> BoxedFuture<'a, Result<Self::Asset, Self::Error>> {

        Box::pin(async move {
            let mut bytes = Vec::new();
            reader.read_to_end(&mut bytes).await?;

            match load_context.path().extension() {
                Some(ext) if ext == "onnx" => {
                    Ok(())
                },
                _ => Err(std::io::Error::new(ErrorKind::Other, "only .onnx supported")),
            }
        })
    }

    fn extensions(&self) -> &[&str] {
        &["onnx"]
    }
}


#[derive(Default)]
pub struct BurnPlugin;

impl Plugin for BurnPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<BurnModel>();
        app.init_asset::<BurnModel>();
        app.register_asset_reflect::<BurnModel>();

        app.init_asset_loader::<ModelLoader>();

        // if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {

        // }
    }
}

