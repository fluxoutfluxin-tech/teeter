#version 450

// "Ripple pool" â€” concentric circular ripples radiating from a bass-pulsed
// epicentre, with mid-driven interference rings.
//
// Same descriptor contract as warp.frag (set 0 binding 0 = UBO, binding 1 =
// texPrev), so it plugs into the existing pipeline with no renderer changes.
//
// Look: the plane is remapped through concentric rings whose radius ripples
// with sine waves, so the previous frame congeals into a pond of expanding
// rings. Bass pulses the ring spacing (a fresh plunk on the beat), mid adds
// angular ripples around the circumference, and the centre glows like a new
// drop landing. Distinct from warp8 (a polar tube you fly down) â€” here the
// rings are the star and they bloom outward, not march forward.

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

void main() {
    vec2 p = uv - 0.5;
    p.x *= u.aspect;

    float r = length(p);
    float a = atan(p.y, p.x);

    // Ring spacing pulses with bass (a fresh "plunk" expanding outward).
    float spacing = 0.09 + 0.10 * u.bass * (0.5 + 0.5 * sin(u.iTime * 2.0 + u.presetSeed * 40.0));
    float rings = r / spacing;

    // Ripple the rings: radial sine crests + angular ripples on mid/treble.
    float wave = sin(rings * 2.0 * 3.14159265 * 2.0 + u.rotate * 3.0 - u.iTime * 1.5)
                * (0.5 + u.mid);
    wave += sin(r * 60.0 - u.iTime * 3.0 + sin(a * 6.0 + u.iTime) * 2.0) * 0.05 * u.treble;

    // Distorted polar radius: rings expand + wobble with the waves.
    float rd = (r + 0.012 * wave) * (1.0 + 0.28 * u.zoom);

    // Slow angular twist so the pool slowly swivels.
    float ad = a + u.iTime * 0.05;

    vec2 rp = vec2(cos(ad), sin(ad)) * rd;
    rp.x /= u.aspect;
    vec2 tex = rp + 0.5;
    tex = 0.5 + (tex - 0.5) * 0.985;

    vec4 col = texture(texPrev, tex);

    // Hue wash: full spin cycling, pulled cool on treble.
    float hue = u.palette + u.treble * 0.30 + u.iTime * 0.05 + u.presetSeed * 0.5;
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
    shifted = mix(vec3(luma), shifted, 0.72);

    // Beat flash + bright "new drop" core.
    shifted += u.beat * 0.20 * (0.5 + 0.5 * sin(vec3(0.0, 1.0, 2.0) + u.presetSeed * 10.0));
    shifted += exp(-r * 40.0 + 2.0) * (0.08 + 0.30 * u.beat) * vec3(0.5, 0.8, 1.0);

    shifted = max(shifted, col.rgb * 0.02 + vec3(0.004));

    // Filmic finish: preserves detail, tames highlights (no flat white).
    float energy = clamp(0.84*u.bass + 0.24*u.mid + 0.24*u.treble + 0.66*u.beat, 0.0, 1.0);
    shifted = teeter_colorize(shifted, energy);
    shifted = teeter_crossfade(shifted, tex, u.dissolve);
    outColor = vec4(shifted, 1.0);
}
