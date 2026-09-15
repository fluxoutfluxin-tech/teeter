#version 450

// warp17 â€” interference moirÃ© (optical).
//
// Two counter-rotating line gratings over the feedback frame; their moirÃ©
// bands shimmer where minima/maxima beat against each other. Centroid controls
// the grating pitch, flux drives the counter-rotation speed, beat pops spokes.

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

void main() {
    vec2 p = uv - 0.5;
    p.x *= u.aspect;

    float ang = u.rotate * 0.8 + u.presetSeed * 6.2831;
    float ca = cos(ang), sa = sin(ang);
    p = mat2(ca, -sa, sa, ca) * p;
    p *= 1.0 + 0.4 * u.zoom;

    // Grating pitch: centroid thins the lattice (higher pitch).
    float pitch = 3.0 + 14.0 * (1.0 - u.cent) * 0.5 + 8.0 * (1.0 - u.cent) * 0.5;
    pitch = max(pitch, 2.0);
    float f = u.iTime * (0.05 + 0.5 * u.flux); // counter-rotation speed

    // Two gratings rotating opposite directions.
    vec2 a1 = mat2(cos(f), -sin(f), sin(f), cos(f)) * p;
    vec2 a2 = mat2(cos(-f), -sin(-f), sin(-f), cos(-f)) * p;

    float g1 = sin(6.2831 * pitch * a1.x * 0.5);
    float g2 = sin(6.2831 * pitch * a2.y * 0.5);
    float moire = g1 * g2;

    // Spokes pop out on the beat; sub-bass adds a slow bass-moire pulse.
    float spokes = cos(atan(p.y, p.x) * 6.0 + u.iTime * 1.2);
    moire += 0.5 * u.beat * spokes;

    // Rebuild the sampling point from the grating phase so the frame gets
    // "warped" by the interference field.
    vec2 tex = fract(vec2(p.x / u.aspect, p.y) + 0.5);
    tex += 0.05 * moire * vec2(1.0, -1.0) * (1.0 + 1.5 * abs(g1 - g2));
    vec3 col = texture(texPrev, tex).rgb;

    // MoirÃ© bands assert as brightness with crest/mid; keep a base light via rolloff.
    col += abs(moire) * (0.06 + 0.30 * u.mid + 0.4 * u.crest) * vec3(0.6, 0.75, 1.0);
    col += 0.035 * u.rolloff;
    col *= 0.8 + 0.4 * u.bass;

    float h = u.palette * 6.2831 + u.presetSeed * 1.3;
    float c = cos(h), sm = sin(h);
    mat3 hueRot = mat3(
        vec3(0.299 + 0.701*c + 0.168*sm, 0.587 - 0.587*c + 0.330*sm, 0.114 - 0.114*c - 0.497*sm),
        vec3(0.299 - 0.299*c - 0.328*sm, 0.587 + 0.413*c + 0.035*sm, 0.114 - 0.114*c + 0.292*sm),
        vec3(0.299 - 0.300*c + 1.250*sm, 0.587 - 0.588*c - 1.050*sm, 0.114 + 0.886*c - 0.203*sm)
    );
    vec3 shifted = hueRot * col.rgb;
    float luma = dot(shifted, vec3(0.299, 0.587, 0.114));
    shifted = mix(vec3(luma), shifted, 0.80);
    shifted = max(shifted, col.rgb * 0.02 + vec3(0.004));

    float energy = clamp(0.75*u.bass + 0.55*u.mid + 0.60*u.treble + 0.60*u.beat, 0.0, 1.0);
    shifted = teeter_colorize(shifted, energy);
    shifted = teeter_crossfade(shifted, tex, u.dissolve);
    outColor = vec4(shifted, 1.0);
}