# bevy_burn


[![test](https://github.com/mosure/bevy_burn/workflows/test/badge.svg)](https://github.com/Mosure/bevy_burn/actions?query=workflow%3Atest)
[![GitHub License](https://img.shields.io/github/license/mosure/bevy_burn)](https://raw.githubusercontent.com/mosure/bevy_burn/main/LICENSE)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/mosure/bevy_burn)](https://github.com/mosure/bevy_burn)
[![GitHub Releases](https://img.shields.io/github/v/release/mosure/bevy_burn?include_prereleases&sort=semver)](https://github.com/mosure/bevy_burn/releases)
[![GitHub Issues](https://img.shields.io/github/issues/mosure/bevy_burn)](https://github.com/mosure/bevy_burn/issues)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/mosure/bevy_burn.svg)](http://isitmaintained.com/project/mosure/bevy_burn)
[![crates.io](https://img.shields.io/crates/v/bevy_burn.svg)](https://crates.io/crates/bevy_burn)

bevy burn async compute nodes. write compute shaders in burn with wgpu input and output buffers shared with bevy's render pipeline.


## usage

```rust
use bevy::prelude::*;
use bevy_burn::{
    BurnInference,
    BurnModel,
    BurnPlugin,
};


fn main() {
    App::build()
        .add_plugins(DefaultPlugins)
        .add_plugin(BurnPlugin)
        .add_system(Startup, setup)
        .add_system(Update, burn_inference)
        .run();
}

fn setup(
    mut commands: Commands,
    burn_models: Res<Assets<BurnModel>>,
) {
    let model = burn_models.load("model.onnx");
    let input = SomeInput::default();

    commands.spawn().insert(input).insert(model);
}

fn burn_inference(
    mut commands: Commands,
    burn_inference: Res<BurnInference>,
    burn_models: Res<Assets<BurnModel>>,
    input_data: Query<(
        Entity,
        &SomeInput,
        &Handle<BurnModel>,
        Without<BurnOutput>,
    )>,
) {
    for (entity, model_handle, input, _) in input_data.iter() {
        if let Some(model) = burn_models.get(model_handle) {
            let output = model.inference(input).unwrap();

            commands.entity(entity).insert(output);
        }
    }
}
```
