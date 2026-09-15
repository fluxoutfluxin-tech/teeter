# teeter build script (run on the Windows side / ROG Ally)
#
# Compiles GLSL shaders to SPIR-V using the Vulkan SDK shader toolchain,
# then builds the Rust binary with cargo.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File build.ps1
#   powershell -ExecutionPolicy Bypass -File build.ps1 --release
#   powershell -ExecutionPolicy Bypass -File build.ps1 --shaders-only

param(
    [switch]$Release,
    [switch]$ShadersOnly
)

$ErrorActionPreference = "Stop"

# --- Locate the Vulkan SDK & shader compiler --------------------------------
$vulkan = $env:VULKAN_SDK
if (-not $vulkan) {
    # Fall back to the standard SDK install path used by ASUS ROG Ally setups.
    $vulkan = "C:\VulkanSDK\1.4.357.0"
}
$glslc = Join-Path $vulkan "Bin\glslc.exe"
if (-not (Test-Path $glslc)) {
    Write-Error "glslc not found at: $glslc. Install the Vulkan SDK or set VULKAN_SDK."
    exit 1
}
Write-Host "[teeter] using glslc: $glslc"

# --- Compile shaders -> SPIR-V --------------------------------------------
$shaderDir  = Join-Path $PSScriptRoot "shaders"
$spvDir     = Join-Path $shaderDir "spv"
New-Item -ItemType Directory -Force -Path $spvDir | Out-Null

function Compile-Shader($src, $dst, $stage) {
    $srcPath = Join-Path $shaderDir $src
    $dstPath = Join-Path $spvDir $dst
    & $glslc "-fshader-stage=$stage" -o $dstPath $srcPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to compile $src"
        exit 1
    }
    Write-Host "[teeter] OK  $src -> $dst"
}

Compile-Shader "composite.vert" "composite.vert.spv" "vertex"
Compile-Shader "warp.frag"      "warp.frag.spv"      "fragment"
Compile-Shader "warp2.frag"     "warp2.frag.spv"     "fragment"
Compile-Shader "warp3.frag"     "warp3.frag.spv"     "fragment"
Compile-Shader "warp4.frag"     "warp4.frag.spv"     "fragment"
Compile-Shader "warp5.frag"     "warp5.frag.spv"     "fragment"
Compile-Shader "warp6.frag"     "warp6.frag.spv"     "fragment"
Compile-Shader "warp7.frag"     "warp7.frag.spv"     "fragment"
Compile-Shader "warp8.frag"     "warp8.frag.spv"     "fragment"
Compile-Shader "warp9.frag"     "warp9.frag.spv"     "fragment"
Compile-Shader "warp10.frag"    "warp10.frag.spv"    "fragment"
Compile-Shader "warp11.frag"    "warp11.frag.spv"    "fragment"
Compile-Shader "warp12.frag"    "warp12.frag.spv"    "fragment"
Compile-Shader "warp13.frag"    "warp13.frag.spv"    "fragment"
Compile-Shader "warp14.frag"    "warp14.frag.spv"    "fragment"
Compile-Shader "warp15.frag"    "warp15.frag.spv"    "fragment"
Compile-Shader "warp16.frag"    "warp16.frag.spv"    "fragment"
Compile-Shader "warp17.frag"    "warp17.frag.spv"    "fragment"
Compile-Shader "warp18.frag"    "warp18.frag.spv"    "fragment"
Compile-Shader "warp19.frag"    "warp19.frag.spv"    "fragment"
Compile-Shader "warp20.frag"    "warp20.frag.spv"    "fragment"
Compile-Shader "feedback.frag"  "feedback.frag.spv"  "fragment"
Compile-Shader "5cell_projection.vert" "5cell_projection.vert.spv" "vertex"
Compile-Shader "5cell_render.frag"     "5cell_render.frag.spv"     "fragment"

if ($ShadersOnly) {
    Write-Host "[teeter] shaders compiled. done."
    exit 0
}

# --- Build the Rust binary --------------------------------------------------
$profileArg = if ($Release) { "--release" } else { "" }
Write-Host "[teeter] cargo build $profileArg ..."
Push-Location $PSScriptRoot
try {
    cargo build $profileArg
    if ($LASTEXITCODE -ne 0) {
        Write-Error "cargo build failed"
        exit 1
    }
    $target = if ($Release) { "release" } else { "debug" }
    $bin = Join-Path $PSScriptRoot "target\$target\teeter.exe"
    Write-Host "[teeter] build complete: $bin"
} finally {
    Pop-Location
}
