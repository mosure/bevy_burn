@group(0) @binding(0)
var<storage, read> src: array<vec4<f32>>;

@group(0) @binding(1)
var dst: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(dst);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let idx = gid.y * dims.x + gid.x;
  textureStore(dst, vec2<i32>(i32(gid.x), i32(gid.y)), src[idx]);
}
