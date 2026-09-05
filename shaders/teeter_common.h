// Shared tone-mapping + finish helpers for the teeter warp shaders.
//
// teeter_colorize: apply a filmic curve that preserves midtone contrast
// (so detail survives) while rolling off highlights so brights never clip to
// flat white, then keep chroma so colors read as colour instead of washed-out
// gray. Called at the end of every warp shader:
//   shifted = teeter_colorize(shifted, energy);
//
// `energy` is a 0..1 audio-loudness term (bass/mid/treble/beat) that lifts the
// whole frame so the lighting follows the music, and stops it fading out to
// black between hits.

// Filmic rolloff: slope ~1 near black so the feedback loop doesn't slowly
// drain to black, rolls highlights off so they never clip to flat white.
vec3 teeter_filmic(vec3 x) {
    x = max(x, vec3(0.0));
    return x / (1.0 + max(x - 0.60, vec3(0.0)) * 0.90);
}

// Contrast pivoted LOW (0.22): lifts dim/mid content so it reads clearly and
// keeps detail, without crushing it toward black each frame (the cause of the
// slow fade-out). True blacks stay black.
vec3 teeter_contrast(vec3 x) {
    const float k = 1.30;
    return clamp((x - 0.22) * k + 0.22, 0.0, 1.0);
}

vec3 teeter_colorize(vec3 shifted, float energy) {
    // Dim BASE exposure (kept constant), then rolloff + contrast for detail.
    // This base is what the audio overlays get screen-blended on top of.
    shifted = teeter_filmic(shifted * 0.55);
    shifted = teeter_contrast(shifted);

    float en = clamp(energy, 0.0, 1.0);
    float luma = dot(shifted, vec3(0.299, 0.587, 0.114));

    // Bass-driven BRIGHTNESS overlay, SCREEN-blended: additive light that
    // ramps with the music. Gated by luminance so black stays black.
    vec3 brightness = en * vec3(0.54, 0.66, 0.84) * luma;
    shifted = 1.0 - (1.0 - shifted) * (1.0 - brightness);

    // Colour tint overlay, also SCREEN-blended (kept subtler).
    luma = dot(shifted, vec3(0.299, 0.587, 0.114));
    vec3 tint = en * vec3(0.10, 0.14, 0.20) * luma;
    shifted = 1.0 - (1.0 - shifted) * (1.0 - tint);

    // Re-saturate slightly so the compression doesn't leave it gray.
    shifted = mix(vec3(luma), shifted, 0.90);
    return clamp(shifted, 0.0, 1.0);
}
