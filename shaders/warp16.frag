#version 450

// warp16 â€” crystal pillars (crystalline).
//
// A hall of vertical "crystal" pillars with refraction fed from the feedback
// frame: bass skews, flux shears, crest ignites the tips. Low rolloff leaves
// the columns darkened, high rolloff frosts the whole cathedral.

#include "teeter_common.h"

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform FrameUniforms {
    float iTime;          // seconds since start
    float bass;           // low band energy   0..1
    float mid;            // mid band energy   0..1
    float treble;         // high band energy  0..1
    float beat;           // beat pulse        0..1
    float zoom;           // right-stick zoom  -1..1
    float warpX;          // left-stick warpX  -1..1
    float warpY;          // left-stick warpY  -1..1
    float rotate;         // touch / stick rotation -1..1
    float dissolve;       // warp-switch crossfade weight 0..1
    float palette;        // palette shift 0..1
    float presetSeed;     // stable random per preset
    float aspect;         // framebuffer width / height
    float subBass;        // sub-60Hz energy     0..1
    float cent;       // spectral centroid   0..1
    float crest;          // spectral crestfactor 0..1
    float flux;           // spectral flux (motion) 0..1
    float rolloff;        // high-end rolloff brightness 0..1
} u;

layout(set = 0, binding = 1) uniform sampler2D texPrev;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    vec2 p = uv - 0.5;
    p.x *= u.aspect;

    // Column grid density driven by centroid (brighter -> more, thinner).
    float n = 6.0 + floor(u.cent * 6.0);
    float colW = 1.0 / n;

    // Pillar x-index and intra-column phase.
    float gx = fract(p.x / colW);
    float xi = floor(p.x / colW);

    // Bass bends the columns; flux makes the bend wander over time.
    float bend = 0.20 * u.bass + 0.10 * u.flux * sin(u.iTime * 0.9 + xi * 7.0);
    float y = p.y + bend * (gx - 0.5) * 4.0;

    // Intra-column "shard" split; y-index into stacked sections.
    float shard = fract(y * 2.0 + 0.5);
    float yi = floor(y * 2.0);

    // Shear by flux (crystal facet twist).
    vec2 q = vec2(p.x + 0.08 * u.flux * shard - gx * 0.5, y);

    // Sample the frame with slight prismic refraction offset by energy.
    vec2 tex = fract(vec2(q.x / u.aspect, q.y) + 0.5);
    vec2 refr = 0.02 * (u.bass * vec2(cos(u.iTime * 2.1), sin(u.iTime * 1.7)) + vec2(u.warpX, u.warpY) * 0.5);
    refr.x /= u.aspect;
    vec3 col = texture(texPrev, tex + refr).rgb;

    // Pillar edge highlight (bright rims) + facet darkening.
    float edge = 1.0 - abs(gx - 0.5) * 6.0;
    edge = smoothstep(0.0, 1.0, edge);
    float facet = 0.5 + 0.5 * sin(shard * 6.2831 * 3.0 + xi * 1.3);

    // Crest ignites the column tips (high spectral peak -> more "crushing" peaks).
    float tip = pow(1.0 - shard, 2.0) * (0.15 + 0.5 * u.crest);
    col += edge * (0.04 + 0.35 * u.mid) * vec3(0.45, 0.65, 1.0);
    col += tip * vec3(1.0, 0.9, 0.7);
    col *= 0.55 + 0.5 * facet + 0.30 * u.beat;
    // Rolloff frost glow lifts everything slightly.
    col += 0.05 * u.rolloff;

    float h = u.palette * 6.2831 + u.presetSeed * 2.0 + 0.3 * sin(u.iTime * 0.2);
    float c = cos(h), sm = sin(h);
    mat3 hueRot = mat3(
        vec3(0.299 + 0.701*c + 0.168*sm, 0.587 - 0.587*c + 0.330*sm, 0.114 - 0.114*c - 0.497*sm),
        vec3(0.299 - 0.299*c - 0.328*sm, 0.587 + 0.413*c + 0.035*sm, 0.114 - 0.114*c + 0.292*sm),
        vec3(0.299 - 0.300*c + 1.250*sm, 0.587 - 0.588*c - 1.050*sm, 0.114 + 0.886*c - 0.203*sm)
    );
    vec3 shifted = hueRot * col.rgb;
    float luma = dot(shifted, vec3(0.299, 0.587, 0.114));
    shifted = mix(vec3(luma), shifted, 0.82);
    shifted = max(shifted, col.rgb * 0.02 + vec3(0.004));

    float energy = clamp(0.80*u.bass + 0.50*u.mid + 0.55*u.treble + 0.55*u.beat, 0.0, 1.0);
    shifted = teeter_colorize(shifted, energy);
    shifted = teeter_crossfade(shifted, tex, u.dissolve);
    outColor = vec4(shifted, 1.0);
}