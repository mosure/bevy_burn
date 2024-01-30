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
pub struct BurnModel {
    // TODO: burn model asset handle & asset derived buffers
}


#[derive(Default)]
pub struct BurnPlugin;

impl Plugin for BurnPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<BurnModel>();
        app.init_asset::<BurnModel>();
        app.register_asset_reflect::<BurnModel>();
    }
}
