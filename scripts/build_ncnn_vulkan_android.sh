#!/usr/bin/env bash
# Build ncnn for Android arm64-v8a WITH Vulkan.
#
# The official prebuilt (ncnn-20260526-android) ships NCNN_VULKAN=OFF, so it
# cannot exercise ncnn's GPU path at all -- comparing its fp32 CPU result
# against MNN-OpenCL / ORT-NNAPI would understate ncnn. This rebuilds it.
#
# NCNN_SIMPLEVK=ON means ncnn brings its own minimal Vulkan header and
# dlopen()s libvulkan.so at runtime, so no Vulkan SDK is needed on the host.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="${ANDROID_SDK_ROOT:-$HOME/.local/share/mise/installs/android-sdk/22.0}"
NDK="$SDK/ndk/27.0.12077973"
SRC="$ROOT/third_party/ncnn-src"
BUILD="$ROOT/third_party/build-ncnn-android"
# No cmake on PATH in this environment -- use the one the Android SDK ships.
CMAKE="$SDK/cmake/3.22.1/bin/cmake"
[ -x "$CMAKE" ] || CMAKE="$(command -v cmake)"

if [ ! -d "$SRC/.git" ]; then
  echo "==> cloning ncnn"
  rm -rf "$SRC"
  git clone --depth 1 https://ghfast.top/https://github.com/Tencent/ncnn.git "$SRC"
fi

mkdir -p "$BUILD"
echo "==> configuring"
"$CMAKE" -S "$SRC" -B "$BUILD" \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 \
  -DANDROID_ARM_NEON=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DNCNN_VULKAN=ON \
  -DNCNN_SIMPLEVK=ON \
  -DNCNN_OPENMP=ON \
  -DNCNN_THREADS=ON \
  -DNCNN_BUILD_TOOLS=OFF \
  -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_BUILD_BENCHMARK=OFF \
  -DNCNN_BUILD_TESTS=OFF \
  -DNCNN_SHARED_LIB=OFF

echo "==> building"
"$CMAKE" --build "$BUILD" --target ncnn -j"$(sysctl -n hw.ncpu)"

echo "==> staging"
OUT="$ROOT/third_party/stage/ncnn-vulkan"
mkdir -p "$OUT/lib/arm64-v8a"
cp -R "$SRC/src" "$OUT/include-src" 2>/dev/null || true
mkdir -p "$OUT/include"
cp -R "$SRC/src/." "$OUT/include/" 2>/dev/null || true
rm -rf "$OUT/include/CMakeLists.txt"
cp "$BUILD/src/libncnn.a" "$OUT/lib/arm64-v8a/libncnn.a"
# generated headers (ncnn_export.h, layer_shader_type.h, ...) live in the build tree
find "$BUILD/src" -name "*.h" -exec cp {} "$OUT/include/" \;

echo "==> done: $OUT/lib/arm64-v8a/libncnn.a"
ls -la "$OUT/lib/arm64-v8a/"
