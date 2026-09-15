#version 450

// "Tunnel" â€” a polar ring tube you fly through.
//
// Same descriptor contract as warp.frag (set 0: UBO binding 0, texPrev
// binding 1, aspect in the uniform block), so it plugs into the existing
// single-pass pipeline with no renderer binding changes.
//
// Look: the plane is folded into polar coordinates and divided into concentric
// rings. The rings march outward as you advance (bass pushes you deeper and
// faster, zoom dials it by hand) so it reads as flying down a glowing tube.
// Treble wrinkles the ring edges; a slow angular twist bends the tube into a
// spiral. Hue is driven by ring position + bass, and a bright "tunnel mouth"
// glows at the center. Feedback contraction, warp of the previous frame, and
// the teeter_colorize finish are the same as every other preset.

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

    // Polar coordinates around the tunnel axis.
    float r = length(p);
    float a = atan(p.y, p.x);

    // Forward motion: rings march toward the center. Bass deepens the run and
    // speeds it up; zoom gives manual control.
    float speed = u.iTime * (0.6 + 0.9 * u.bass) + u.presetSeed * 40.0;
    // Ring coordinate: bass spreads the rings toward the viewer as you advance.
    float rings = r * (14.0 + 14.0 * u.bass) - speed;
    // Treble ripples the edges of each ring.
    rings += 0.6 * sin(rings * 3.0 + u.iTime * 2.0) * u.treble;

    // Ring band (bright in, dark out) that pulses with the beat.
    float band = 0.5 + 0.5 * cos(rings * 6.2831);
    band = mix(band, 0.5 + 0.5 * cos(rings * 6.2831 + u.beat * 1.5), 0.6);

    // Twist the tube: theta turns as it advances plus a stable per-preset lean
    // and manual knob. Gives the tube a gentle spiral bore.
    float th = a + u.rotate * 0.5 + u.presetSeed * 6.2831 * 0.25
             + 0.3 * rings + u.iTime * 0.15;

    // Sample radius: mild feedback contraction + audio zoom toward center.
    float rr = r * 0.985 * (1.0 + 0.30 * u.zoom + 0.10 * u.bass);

    vec2 rp = vec2(cos(th), sin(th)) * rr;
    rp.x /= u.aspect; // keep sampling isotropic in UV space
    vec2 tex = rp + 0.5;
    tex = 0.5 + (tex - 0.5) * 0.985;

    vec4 col = texture(texPrev, tex);

    // Hue wash: rings and bass push the wheel, palette adds a manual shift,
    // and iTime keeps it cycling. Left at full strength then desaturated.
    float hue = u.palette + u.bass * 0.4 + rings * 0.05 + u.iTime * 0.05;
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
    shifted = mix(vec3(luma), shifted, 0.75);

    // Beat flash (soft) so hits pop the whole tube.
    float flash = u.beat * 0.22;
    shifted += flash * (0.5 + 0.5 * sin(vec3(0.0, 1.0, 2.0) + u.presetSeed * 10.0));

    shifted = max(shifted, col.rgb * 0.02 + vec3(0.004));

    // Bright tunnel-mouth glow at center, fatter and brighter on the bass.
    float d = length(rp);
    shifted += exp(-d * 16.0 + 2.0) * (0.08 + 0.20 * u.bass) * vec3(0.5, 0.7, 1.0);
    shifted += band * 0.05 * u.mid * vec3(0.8, 0.9, 1.1);

    // Filmic finish: preserves detail, tames highlights (no flat white).
    float energy = clamp(0.84*u.bass + 0.24*u.mid + 0.24*u.treble + 0.66*u.beat, 0.0, 1.0);
    shifted = teeter_colorize(shifted, energy);
    shifted = teeter_crossfade(shifted, tex, u.dissolve);
    outColor = vec4(shifted, 1.0);
}
