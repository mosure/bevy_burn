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
        // module::conv2d,
        // ops::ConvOptions,
        Tensor,
    },
};
use burn_wgpu::Wgpu;


type BurnBackend = Autodiff<Wgpu<f32, i32>>;


// TODO: convert to resource and add world inspector
const CFL: f32 = 0.3;
const MAX_DT: f32 = 0.01;
const MAX_STEPS: usize = 2;
const MAX_VEL: f32 = 1000.0;
const PRESSURE_ITERS: usize = 30;
const SIZE: u32 = 512;
const SOURCE_MAG: f32 = 0.05;
const SOURCE_POS: (usize, usize) = (SIZE as usize / 2, SIZE as usize / 2);
const SOURCE_SIGMA: f32 = 3.0;
const VISCOSITY: f32 = 1e-3;
const DIFF_ITERS: usize = 10;
const OMEGA: f32 = 1.7;



fn velocity_to_rgba<B: Backend>(v: &Tensor<B, 4>) -> Tensor<B, 3> {
    let vx = v.clone()
        .slice_dim(0, 0..1)
        .slice_dim(1, 0..1)
        .squeeze_dims::<2>(&[0, 1]);

    let vy = v.clone()
        .slice_dim(0, 0..1)
        .slice_dim(1, 1..2)
        .squeeze_dims::<2>(&[0, 1]);

    let mag = (vx.powf_scalar(2.) + vy.powf_scalar(2.)).sqrt();
    let max_val = mag.clone().max().add_scalar(1e-8f32).into_scalar();
    let mag = mag.div_scalar(max_val);

    let r = mag.clone().unsqueeze_dim::<3>(2);
    let g = mag.clone().unsqueeze_dim::<3>(2);
    let b = mag.unsqueeze_dim::<3>(2);
    let a = Tensor::<B, 3>::ones(r.dims(), &Default::default());

    Tensor::cat(vec![r, g, b, a], 2)
}



fn vortex_source<B: Backend>(grid_like: &Tensor<B,4>) -> Tensor<B,4> {
    let [b, _, h, w] = grid_like.dims();
    let xs = Tensor::<B,1,Int>::arange(0..w as i64,&Default::default())
               .float().reshape([1,1,1,w]);
    let ys = Tensor::<B,1,Int>::arange(0..h as i64,&Default::default())
               .float().reshape([1,1,h,1]);

    let cx = SOURCE_POS.0 as f32;
    let cy = SOURCE_POS.1 as f32;
    let dx = xs - cx;
    let dy = ys - cy;

    let r2 = dx.clone().powf_scalar(2.) + dy.clone().powf_scalar(2.);
    let g = (-r2 / (2.0*SOURCE_SIGMA*SOURCE_SIGMA)).exp().mul_scalar(SOURCE_MAG);

    let u = -dy * g.clone();
    let w = dx * g;

    let u = u.repeat(&[b,1,1,1]);
    let w = w.repeat(&[b,1,1,1]);
    Tensor::cat(vec![u, w], 1)
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
        .mul_scalar(SOURCE_MAG);

    let gaussian = gaussian.repeat(&[b, 1, 1, 1]);

    let zero = Tensor::<B, 4>::zeros([b, 1, h, w], &Default::default());
    Tensor::cat(vec![zero, gaussian], 1)
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

    let v1 = advect::<B>(v, dt);
    let v2 = diffuse::<B>(v1, nu, dt);

    let u = v2.clone().slice_dim(1, 0..1);
    let wv = v2.slice_dim(1, 1..2);

    let div = divergence(&u, &wv);
    let mut p = Tensor::<B, 4>::zeros(div.dims(), &Default::default());
    let rhs = div.clone() / dt;

    // TODO: move to static allocation
    let [_, _, h, w] = p.dims();
    let mask_red = colour_mask::<B>(h, w,  true ).repeat(&[1,1,1,1]);
    let mask_black = colour_mask::<B>(h, w, false).repeat(&[1,1,1,1]);

    for _ in 0..PRESSURE_ITERS {
        // let neigh = neigh_sum(&p);
        // p = (neigh + rhs.clone()) * 0.25;

        let sum_nb = neighbor_sum_rb(&p, true);
        let p_new  = (sum_nb - rhs.clone()) * 0.25;
        p = p.clone() + (p_new - p) * mask_red.clone() * OMEGA;

        let sum_nb = neighbor_sum_rb(&p, false);
        let p_new  = (sum_nb - rhs.clone()) * 0.25;
        p = p.clone() + (p_new - p) * mask_black.clone() * OMEGA;
    }
    let (dp_dx, dp_dy) = grad_p(&p);
    let u_corr = u - dp_dx * dt;
    let w_corr = wv - dp_dy * dt;
    Tensor::cat(vec![u_corr, w_corr], 1).clamp(-MAX_VEL, MAX_VEL)
}


fn d_dx<B: Backend>(f: &Tensor<B,4>) -> Tensor<B,4> {
    (
        f.clone().roll(&[ 1], &[3]) -
        f.clone().roll(&[-1], &[3])
    ) * 0.5
}

fn d_dy<B: Backend>(f: &Tensor<B,4>) -> Tensor<B,4> {
    (
        f.clone().roll(&[ 1], &[2]) -
        f.clone().roll(&[-1], &[2])
    ) * 0.5
}

fn divergence<B: Backend>(u: &Tensor<B,4>, v: &Tensor<B,4>) -> Tensor<B,4> {
    d_dx(u) + d_dy(v)
}

fn grad_p<B: Backend>(p: &Tensor<B,4>) -> (Tensor<B,4>, Tensor<B,4>) {
    (d_dx(p), d_dy(p))
}

fn neigh_sum<B: Backend>(v: &Tensor<B, 4>) -> Tensor<B, 4> {
    v.clone().roll(&[ 1], &[2]) +
        v.clone().roll(&[-1], &[2]) +
        v.clone().roll(&[ 1], &[3]) +
        v.clone().roll(&[-1], &[3])
}

fn colour_mask<B: Backend>(h: usize, w: usize, red: bool) -> Tensor<B, 4> {
    let xs = Tensor::<B, 1, Int>::arange(0..w as i64, &Default::default())
        .float()
        .reshape([1, 1, 1, w]);
    let ys = Tensor::<B, 1, Int>::arange(0..h as i64, &Default::default())
        .float()
        .reshape([1, 1, h, 1]);

    let parity = (xs + ys).remainder_scalar(2.0f32);

    parity
        .equal_elem(if red { 0.0f32 } else { 1.0f32 })
        .float()
}

fn neighbor_sum_rb<B: Backend>(v: &Tensor<B, 4>, red: bool) -> Tensor<B, 4> {
    let [b, _, h, w] = v.dims();
    let mask = colour_mask::<B>(h, w, red).repeat(&[b, 1, 1, 1]);

    neigh_sum(v) * mask
}


fn advect<B: Backend>(v: Tensor<B,4>, dt: f32) -> Tensor<B,4> {
    /* first pass: semi‑lagrangian (existing bilinear) */
    let v1 = advect_linear::<B>(v.clone(), dt);

    /* second pass: forward step to estimate error */
    let v_back = advect_linear::<B>(v1.clone(), -dt);

    /* error estimate and correction */
    let v2 = v1 + (v - v_back) * 0.5;

    /* one last clamp to avoid new overshoots */
    v2.clamp(-MAX_VEL, MAX_VEL)
}


fn advect_linear<B: Backend>(v: Tensor<B, 4>, dt: f32) -> Tensor<B, 4> {
    let [_, _, h, w] = v.dims();

    let xs = Tensor::<B, 1, Int>::arange(0..w as i64, &Default::default())
        .float().reshape([1, 1, 1, w]);
    let ys = Tensor::<B, 1, Int>::arange(0..h as i64, &Default::default())
        .float().reshape([1, 1, h, 1]);

    let u = v.clone().slice_dim(1, 0..1);
    let wv = v.clone().slice_dim(1, 1..2);

    let x_back = xs - u * dt;
    let y_back = ys - wv * dt;

    let x_back = (x_back + w as f32).remainder_scalar(w as f32);
    let y_back = (y_back + h as f32).remainder_scalar(h as f32);

    let x0 = x_back.clone().floor();
    let y0 = y_back.clone().floor();
    let x1 = (x0.clone() + 1.0).remainder_scalar(w as f32);
    let y1 = (y0.clone() + 1.0).remainder_scalar(h as f32);

    let wx = x_back - x0.clone();
    let wy = y_back - y0.clone();

    let gather = |
        xx: Tensor<B,4>,
        yy: Tensor<B,4>,
    | {
        let mut xi = xx.int();
        let mut yi = yy.int();
        let v = v.clone();
        let c = v.dims()[1];
        xi = xi.repeat(&[1, c, 1, 1]);
        yi = yi.repeat(&[1, c, 1, 1]);
        v.gather(3, xi).gather(2, yi)
    };

    let v00 = gather(x0.clone(), y0.clone());
    let v10 = gather(x1.clone(), y0);
    let v01 = gather(x0, y1.clone());
    let v11 = gather(x1, y1);

    let wx = wx.repeat(&[1, v.dims()[1], 1, 1]);
    let wy = wy.repeat(&[1, v.dims()[1], 1, 1]);
    let v0 = v00 * (1.0 - wx.clone()) + v10 * wx.clone();
    let v1 = v01 * (1.0 - wx.clone()) + v11 * wx;
    v0 * (1.0 - wy.clone()) + v1 * wy
}

 fn diffuse<B: Backend>(v: Tensor<B, 4>, nu: f32, dt: f32) -> Tensor<B, 4> {
    if nu == 0.0 { return v }
    let a = nu * dt;
    let mut x = v.clone();
    for _ in 0..DIFF_ITERS {
        let s = neigh_sum(&x);
        x = (v.clone() + s * a) / (1.0 + 4.0 * a);
    }
    x
}


#[derive(Resource)]
struct NavierStokesState<B: Backend> {
    velocity: Tensor::<B, 4>,
}

impl Default for NavierStokesState<BurnBackend> {
    fn default() -> Self {
        let velocity = Tensor::<BurnBackend, 4>::random(
            [1, 2, SIZE as usize, SIZE as usize],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &Default::default(),
        );
        // let velocity = Tensor::<BurnBackend, 4>::zeros(
        //     [1, 2, SIZE as usize, SIZE as usize],
        //     &Default::default(),
        // );
        NavierStokesState { velocity }
    }
}


fn setup<B: Backend>(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut ns: ResMut<NavierStokesState<B>>,
) {
    // ns.velocity = ns.velocity.clone() + velocity_point_source::<B>(&ns.velocity);
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

        // ns.velocity = ns.velocity.clone() + velocity_point_source::<B>(&ns.velocity) * frame_dt;
        // ns.velocity = ns.velocity.clone() + vortex_source::<B>(&ns.velocity) * frame_dt;

        for _ in 0..steps {
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
