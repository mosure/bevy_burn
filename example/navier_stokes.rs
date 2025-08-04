#![recursion_limit = "256"]

use bevy::{
    prelude::*,
    color::palettes::css::GOLD,
    diagnostic::{
        DiagnosticsStore,
        FrameTimeDiagnosticsPlugin,
    },
};
use bevy_burn::{
    BevyBurnBridgePlugin,
    BevyBurnHandle,
    BindingDirection,
};
use burn_autodiff::Autodiff;
use burn_core::{
    tensor::{
        backend::Backend,
        ElementConversion,
        Int,
        module::conv2d,
        ops::ConvOptions,
        Tensor,
    },
};
use burn_wgpu::Wgpu;


type BurnBackend = Autodiff<Wgpu<f32, i32>>;


// TODO: convert to resource/live edit
const CFL: f32 = 0.6;
const MAX_DT: f32 = 0.02;
const MAX_STEPS: usize = 4;
const MAX_VEL: f32 = 1_000.0;
const PRESSURE_ITERS: usize = 40;
const SIZE: u32 = 512;
const SOURCE_MAG: f32 = 0.000001;
const SOURCE_POS: (usize, usize) = (SIZE as usize / 2, SIZE as usize / 2);
const SOURCE_SIGMA: f32 = 3.0;
const VISCOSITY: f32 = 1e-4;
const DIFF_ITERS: usize = 10;



fn velocity_to_rgba<B: Backend>(v: &Tensor<B, 4>) -> Tensor<B, 3> {
    // --------------------------------------------------------------------
    // take batch‑0 and split components → `[h, w]`
    // --------------------------------------------------------------------
    let vx = v.clone()
        .slice_dim(0, 0..1)
        .slice_dim(1, 0..1)
        .squeeze_dims::<2>(&[0, 1]);

    let vy = v.clone()
        .slice_dim(0, 0..1)
        .slice_dim(1, 1..2)
        .squeeze_dims::<2>(&[0, 1]);

    // --------------------------------------------------------------------
    // magnitude (‖v‖) → grayscale in [0, 1]
    // --------------------------------------------------------------------
    let mag = (vx.powf_scalar(2.) + vy.powf_scalar(2.)).sqrt();
    let max_val = mag.clone().max().add_scalar(1e-8f32).into_scalar();
    let mag = mag.div_scalar(max_val);

    // --------------------------------------------------------------------
    // replicate magnitude → rgb, add constant alpha
    // --------------------------------------------------------------------
    let r = mag.clone().unsqueeze_dim::<3>(2);
    let g = mag.clone().unsqueeze_dim::<3>(2);
    let b = mag.unsqueeze_dim::<3>(2);
    let a = Tensor::<B, 3>::ones(r.dims(), &Default::default());

    // `[h, w, 4]`
    Tensor::cat(vec![r, g, b, a], 2)
}



fn velocity_point_source<B: Backend>(grid_like: &Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, _, h, w] = grid_like.dims();

    let xs = Tensor::<B, 1, Int>::arange(0..w as i64, &Default::default())
        .float()
        .reshape([1, 1, 1, w]);
    let ys = Tensor::<B, 1, Int>::arange(0..h as i64, &Default::default())
        .float()
        .reshape([1, 1, h, 1]);

    let dx2 = (xs.sub_scalar(SOURCE_POS.0 as f32)).powf_scalar(2.);
    let dy2 = (ys.sub_scalar(SOURCE_POS.1 as f32)).powf_scalar(2.);

    let gaussian = (dx2 + dy2)
        .div_scalar(-2.0 * SOURCE_SIGMA * SOURCE_SIGMA)
        .exp()
        .mul_scalar(SOURCE_MAG);            // [1,1,h,w]

    // replicate over the batch dimension
    let gaussian = gaussian.repeat(&[b, 1, 1, 1]); // [b,1,h,w]

    let zero = Tensor::<B, 4>::zeros([b, 1, h, w], &Default::default());
    Tensor::cat(vec![zero, gaussian], 1)           // [b,2,h,w]
}


/// * `v`   – velocity field `[b, 2, h, w]`
/// * `nu`  – kinematic viscosity
/// * `dt`  – time step
fn navier_stokes<B: Backend>(
    v: Tensor<B, 4>,
    nu: f32,
    dt: f32,
) -> Tensor<B, 4> {
    if dt <= 0.0 {
        return v;
    }

    info!(
        "field min: {}, max: {}",
        v.clone().min().into_scalar(),
        v.clone().max().into_scalar(),
    );

    let v = v.clamp(-MAX_VEL, MAX_VEL);

    // 1) advect
    let v1 = advect::<B>(wrap(v), dt);

    // 2) implicit viscosity
    let v2 = diffuse::<B>(v1, nu, dt);

    // 3) projection (unchanged)
    let u = v2.clone().slice_dim(1, 0..1);
    let wv = v2.slice_dim(1, 1..2);
    let dx = diff_kernel::<B>(true);
    let dy = diff_kernel::<B>(false);

    let div = d(&u, &dx) + d(&wv, &dy);
    let mut p = Tensor::<B, 4>::zeros(div.dims(), &Default::default());
    let rhs = div * (1.0 / dt);
    for _ in 0..PRESSURE_ITERS {
        p = (d(&p, &neighbor_kernel::<B>()) + rhs.clone()) * 0.25;
    }
    let u_corr = u - d(&p, &dx) * dt;
    let w_corr = wv - d(&p, &dy) * dt;
    Tensor::cat(vec![u_corr, w_corr], 1)
}

/// apply 2‑d convolution with “same” padding
fn d<B: Backend>(x: &Tensor<B, 4>, k: &Tensor<B, 4>) -> Tensor<B, 4> {
    let ch = x.dims()[1];
    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], ch);
    conv2d::<B>(x.clone(), k.clone(), None, opts)
}

/// central‑difference kernel: x‑axis if `horz`, else y‑axis
fn diff_kernel<B: Backend>(horz: bool) -> Tensor<B, 4> {
    let k = if horz {
        // [-½, 0, ½] along x
        vec![0.0, 0.0, 0.0, -0.5, 0.0, 0.5, 0.0, 0.0, 0.0]
    } else {
        // [-½, 0, ½] along y
        vec![0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0]
    };
    Tensor::<B, 1>::from_floats(k.as_slice(), &Default::default())
        .reshape([1, 1, 3, 3])
}

/// five‑point laplacian kernel
fn lap_kernel<B: Backend>() -> Tensor<B, 4> {
    Tensor::<B, 1>::from_floats(
        [0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0],
        &Default::default(),
    ).reshape([1, 1, 3, 3])
}


fn neighbor_kernel<B: Backend>() -> Tensor<B, 4> {
    Tensor::<B, 1>::from_floats(
        [
            0.0, 1.0, 0.0,
            1.0, 0.0, 1.0,
            0.0, 1.0, 0.0,
         ],
        &Default::default(),
    ).reshape([1, 1, 3, 3])
}


fn advect<B: Backend>(v: Tensor<B, 4>, dt: f32) -> Tensor<B, 4> {
    let [_, _, h, w] = v.dims();

    // build coord grids
    let xs = Tensor::<B, 1, Int>::arange(0..w as i64, &Default::default())
        .float().reshape([1, 1, 1, w]);
    let ys = Tensor::<B, 1, Int>::arange(0..h as i64, &Default::default())
        .float().reshape([1, 1, h, 1]);

    // velocities
    let u = v.clone().slice_dim(1, 0..1);   // [b,1,h,w]
    let wv = v.clone().slice_dim(1, 1..2);          // [b,1,h,w]

    // back‑trace
    let x_back = xs - u * dt;               // [b,1,h,w]
    let y_back = ys - wv * dt;              // [b,1,h,w]

    // bring coords into [0,w) × [0,h) with wrap
    let x_back = (x_back + w as f32).remainder_scalar(w as f32);
    let y_back = (y_back + h as f32).remainder_scalar(h as f32);

    let mut xi = x_back
        .round()
        .int()
        .clamp(0, (w - 1) as i64);    // [b,1,h,w]
    let mut yi = y_back
        .round()
        .int()
        .clamp(0, (h - 1) as i64);    // [b,1,h,w]

    /* replicate indices along the channel dimension so their shape
       matches v = [b,c,h,w]                                           */
    let c = v.dims()[1];
    xi = xi.repeat(&[1, c, 1, 1]);     // [b,c,h,w]
    yi = yi.repeat(&[1, c, 1, 1]);     // [b,c,h,w]

    /* gather: first x, then y                                         */
    v.gather(3, xi)                      // gather along width
     .gather(2, yi)                      // gather along height
}

 fn diffuse<B: Backend>(v: Tensor<B, 4>, nu: f32, dt: f32) -> Tensor<B, 4> {
    if nu == 0.0 { return v }
    let a = nu * dt;
    let mut x = v.clone();
    for _ in 0..DIFF_ITERS {
        let lap = d(&x, &lap_kernel::<B>());
        x = (v.clone() + lap * a) / (1.0 + 4.0 * a);
    }
    x
}

fn wrap<B: Backend>(v: Tensor<B, 4>) -> Tensor<B, 4> {  // x,y periodic boundaries
     let [_, _, h, w] = v.dims();
     v.roll(&[h as i32 / 2, w as i32 / 2], &[2, 3])
}


#[derive(Resource)]
struct NavierStokesState<B: Backend> {
    velocity: Tensor::<B, 4>,
}

impl Default for NavierStokesState<BurnBackend> {
    fn default() -> Self {
        // let velocity = Tensor::<BurnBackend, 4>::random(
        //     [1, 2, SIZE as usize, SIZE as usize],
        //     Distribution::Uniform(0.0, 1.0),
        //     &Default::default(),
        // );
        let velocity = Tensor::<BurnBackend, 4>::zeros(
            [1, 2, SIZE as usize, SIZE as usize],
            &Default::default(),
        );
        NavierStokesState { velocity }
    }
}


fn setup<B: Backend>(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut ns: ResMut<NavierStokesState<B>>,
) {
    ns.velocity = ns.velocity.clone() + velocity_point_source::<B>(&ns.velocity);
    let rgba = velocity_to_rgba(&ns.velocity);

    let bevy_image = images.add(Image::default());
    commands.spawn(BevyBurnHandle {
        bevy_image: bevy_image.clone(),
        tensor: rgba,
        upload: true,
        direction: BindingDirection::BurnToBevy,
    });


    commands.spawn(Camera2d::default());

    commands.spawn((
        Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            ..default()
        },
        ImageNode::new(bevy_image),
    ));
}

fn update_tensor<B: Backend>(
    time: Res<Time>,
    mut handles: Query<&mut BevyBurnHandle<B>>,
    mut ns: ResMut<NavierStokesState<B>>,
) {
    for mut handle in handles.iter_mut() {
        let frame_dt  = time.delta_secs();
        let vmax = ns.velocity.clone().abs().max().into_scalar().elem::<f32>().max(1e-3);
        let safe_dt = (CFL / vmax).min(MAX_DT);

        let steps = ((frame_dt / safe_dt).ceil() as usize).clamp(1, MAX_STEPS);
        let sub_dt = frame_dt / steps as f32;

        for _ in 0..steps {
            // ns.velocity = ns.velocity.clone() + velocity_point_source::<B>(&ns.velocity) * sub_dt;
            ns.velocity = navier_stokes(ns.velocity.clone(), VISCOSITY, sub_dt);
        }

        let rgba = velocity_to_rgba(&ns.velocity);

        handle.tensor = rgba;
        handle.upload = true;
    }
}


fn fps_display_setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
) {
    commands.spawn((
        Text("fps: ".to_string()),
        TextFont {
            font: asset_server.load("fonts/Caveat-Bold.ttf"),
            font_size: 60.0,
            ..Default::default()
        },
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(5.0),
            left: Val::Px(15.0),
            ..default()
        },
        ZIndex(2),
    )).with_child((
        FpsText,
        TextColor(Color::Srgba(GOLD)),
        TextFont {
            font: asset_server.load("fonts/Caveat-Bold.ttf"),
            font_size: 60.0,
            ..Default::default()
        },
        TextSpan::default(),
    ));
}

#[derive(Component)]
struct FpsText;

fn fps_update_system(
    diagnostics: Res<DiagnosticsStore>,
    mut query: Query<&mut TextSpan, With<FpsText>>,
) {
    for mut text in &mut query {
        if let Some(fps) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) {
            if let Some(value) = fps.smoothed() {
                **text = format!("{value:.2}");
            }
        }
    }
}


fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            FrameTimeDiagnosticsPlugin::default(),
            BevyBurnBridgePlugin::<BurnBackend>::default(),
        ))
        .init_resource::<NavierStokesState::<BurnBackend>>()
        .add_systems(
            Startup,
            (
                fps_display_setup,
                setup::<BurnBackend>,
            )
        )
        .add_systems(
            Update,
            (
                fps_update_system,
                update_tensor::<BurnBackend>,
            )
        )
        .run();
}
