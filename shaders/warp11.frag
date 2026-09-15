#version 450

// "Cut-glass kaleidoscope" â€” a 6-fold mirrored radial kaleidoscope overlaid with
// a coarse cell-grid quantization, so motion shatters into rotating shards.
//
// Same descriptor contract as warp.frag (set 0 binding 0 = UBO, binding 1 =
// texPrev), so it plugs into the existing pipeline with no renderer changes.
//
// Look: the plane is folded into 6 mirrored radial sectors, then each sector is
// remapped into a low-res cell grid whose outputs were drawn from a sheared
// offset of the previous frame. The result is a faceted, jewel-like bloom that
// rotates and pulses with the bands. Distinct from warp4 (4-fold, smooth
// wedges) â€” this one has 6 shards AND a blocky tile quantization.

#include "teeter_common.h"

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform FrameUniforms {
    float iTime;
    float bass;
    float mid;
    float treble;
    float beat;
    float zoom;
    float warpX;
    float warpY;
    float rotate;
    float dissolve;
    float palette;
    float presetSeed;
    float aspect;
} u;

layout(set = 0, binding = 1) uniform sampler2D texPrev;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    vec2 p = uv - 0.5;
    p.x *= u.aspect;

    // 6-fold radial mirror (kaleidoscope): fold angle into [0, pi/3].
    const float PI = 3.14159265;
    float seg = PI / 3.0;
    float a = atan(p.y, p.x);
    float r = length(p);
    a = mod(a, seg);
    a = abs(a - seg * 0.5);

    // Rotate + drive the fold angle with rotate knob and time.
    a += u.rotate * 0.5 + u.iTime * (0.10 + 0.20 * u.mid);

    // Back to cartesian in the folded sector, then zoom the shards outward.
    vec2 folded = vec2(cos(a), sin(a)) * r;
    folded *= 1.0 + 0.32 * u.zoom + 0.30 * u.bass;

    // Cell-grid quantize: snap to a low-res tile so each wedge reads as shards.
    float cell = 0.035 + 0.05 * u.treble;
    vec2 quant = floor(folded / cell) * cell;
    // Slight per-cell shear so the tiles slope like cut glass.
    vec2 shear = quant + vec2(0.0, sin(quant.x * 60.0 + u.presetSeed * 20.0) * cell * 0.5);

    // Feedback contraction toward center.
    vec2 tex = shear + 0.5;
    tex = 0.5 + (tex - 0.5) * 0.985;

    vec4 col = texture(texPrev, tex);

    // Hue wash: full spin cycling, pushed warm on bass.
    float hue = u.palette + u.bass * 0.42 + u.iTime * 0.05 + u.presetSeed * 0.5;
    float h = hue * 6.2831;
    float s = 1.0;
    float c = s * cos(h);
    float sm = s * sin(h);
    mat3 hueRot = mat3(
        vec3(0.299 + 0.701*c + 0.168*sm, 0.587 - 0.587*c + 0.330*sm, 0.114 - 0.114*c - 0.497*sm),
        vec3(0.299 - 0.299*c - 0.328*sm, 0.587 + 0.413*c + 0.035*sm, 0.114 - 0.114*c + 0.292*sm),
        vec3(0.299 - 0.300*c + 1.250*sm, 0.587 - 0.588*c - 1.050*sm, 0.114 + 0.886*c - 0.203*sm)
    );
    vec3 shifted = hueRot * col.rgb;
    float luma = dot(shifted, vec3(0.299, 0.587, 0.114));
    shifted = mix(vec3(luma), shifted, 0.74);

    // Beat flash (soft).
    float flash = u.beat * 0.22;
    shifted += flash * (0.5 + 0.5 * sin(vec3(0.0, 1.0, 2.0) + u.presetSeed * 10.0));

    shifted = max(shifted, col.rgb * 0.02 + vec3(0.004));

    // Bright faceted core glows on bass/beat.
    float d = length(p);
    shifted += exp(-d * 20.0 + 1.8) * (0.06 + 0.20 * u.bass + 0.10 * u.beat) * vec3(0.7, 0.9, 1.0);

    // Filmic finish: preserves detail, tames highlights (no flat white).
    float energy = clamp(0.84*u.bass + 0.24*u.mid + 0.24*u.treble + 0.66*u.beat, 0.0, 1.0);
    shifted = teeter_colorize(shifted, energy);
    shifted = teeter_crossfade(shifted, tex, u.dissolve);
    outColor = vec4(shifted, 1.0);
}
