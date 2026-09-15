#version 450

// "Teeter variant" â€” an alternate organic feedback warp.
//
// Same descriptor contract as warp.frag (set 0: UBO binding 0, texPrev
// binding 1), so it plugs into the existing pipeline/descriptor layout with no
// renderer changes to the bindings. Different personality: a layered domain
// warp with a soft radial fold, gentler auto-drift, and a warm color wash.
//
// Still a single-pass feedback shader: samples texPrev (previous frame),
// warps, writes to the swapchain; the CPU blits it back into texPrev.

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

    // Gentle base rotation (slower drift than the default warp).
    float ang = u.rotate * 0.5 + u.presetSeed * 6.2831 * 0.15
              + 0.4 * sin(u.iTime * 0.09 + u.presetSeed * 31.0);
    float ca = cos(ang);
    float sa = sin(ang);
    vec2 rp = vec2(ca * p.x - sa * p.y, sa * p.x + ca * p.y);

    // Zoom toward center.
    rp *= 1.0 + 0.30 * u.zoom + 0.28 * u.bass;

    // Layered organic domain warp: two sine octaves at different frequencies
    // fold into each other, giving a flowing "liquid cloth" feel.
    vec2 wob = vec2(
        0.10 * sin(rp.y * 9.0 + u.iTime * 1.4) * u.treble,
        0.10 * sin(rp.x * 13.0 - u.iTime * 1.1) * u.mid
    );
    wob += 0.05 * vec2(
        sin(rp.x * 23.0 + u.iTime * 2.6) * u.mid + u.warpX,
        cos(rp.y * 21.0 - u.iTime * 2.3) * u.treble + u.warpY
    );
    wob.x /= u.aspect;
    rp += wob;

    // Sample previous frame + inward contraction.
    vec2 tex = rp + 0.5;
    tex = 0.5 + (tex - 0.5) * 0.975;

    vec4 col = texture(texPrev, tex);

    // Warm hue wash (shifts toward amber/rose with bass) + continuous cycling.
    // `+ iTime * 0.05` rotates the color wheel over time so colors keep moving.
    float hue = u.palette + u.bass * 0.30 + u.presetSeed * 0.5 + u.iTime * 0.05;
    float h = hue * 6.2831;
    // Full-strength rotation for clear cycling, then desaturate to keep it soft.
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
    shifted = mix(vec3(luma), shifted, 0.70);

    // Beat flash (soft).
    float flash = u.beat * 0.22;
    shifted += flash * (0.5 + 0.5 * sin(vec3(0.0, 1.0, 2.0) + u.presetSeed * 10.0));

    shifted = max(shifted, col.rgb * 0.02 + vec3(0.004));

    // Soft radial glow driven by bass.
    float d = length(rp);
    shifted += exp(-d * 18.0 + 1.5) * (0.06 + 0.14 * u.bass) * vec3(0.9, 0.6, 0.4);

    // Filmic finish: preserves detail, tames highlights (no flat white).
    float energy = clamp(0.84*u.bass + 0.24*u.mid + 0.24*u.treble + 0.66*u.beat, 0.0, 1.0);
    shifted = teeter_colorize(shifted, energy);
    shifted = teeter_crossfade(shifted, tex, u.dissolve);
    outColor = vec4(shifted, 1.0);
}
