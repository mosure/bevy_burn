// pack_rgba_to_bytes.wgsl
struct Params {
    width: u32,
    height: u32,
    padded_words_per_row: u32, // padded bytes per row / 4
    _pad: u32,
};

@group(0) @binding(0) var<storage, read>  src : array<f32>;  // length = width*height*4
@group(0) @binding(1) var<storage, read_write> dst : array<u32>; // packed RGBA8 -> 1 u32 per pixel
@group(0) @binding(2) var<uniform> params : Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let w = params.width;
    let x = gid.x;
    let y = gid.y;

    let base = (y * w + x) * 4u;
    // clamp to [0,1] then pack to 4x8 unorm in one u32
    let rgba = vec4<f32>(
        clamp(src[base + 0u], 0.0, 1.0),
        clamp(src[base + 1u], 0.0, 1.0),
        clamp(src[base + 2u], 0.0, 1.0),
        clamp(src[base + 3u], 0.0, 1.0)
    );
    let word = pack4x8unorm(rgba);

    // one u32 per pixel, pad the row by skipping to the padded index
    let dst_index = y * params.padded_words_per_row + x;
    dst[dst_index] = word;
}
