cmake_minimum_required(VERSION 3.22.1)
project(ppocr_bench LANGUAGES C CXX)

if(NOT DEFINED THIRD_PARTY_DIR)
    message(FATAL_ERROR "THIRD_PARTY_DIR must point at third_party/stage")
endif()

set(ABI_DIR ${CMAKE_ANDROID_ARCH_ABI})

# ---- lw.PPOCR.C: pure-C runtime, no deps ----
add_library(lw STATIC IMPORTED)
set_target_properties(lw PROPERTIES
        IMPORTED_LOCATION ${THIRD_PARTY_DIR}/lw/lib/${ABI_DIR}/liblw_ppocr_c_static.a
        INTERFACE_INCLUDE_DIRECTORIES ${THIRD_PARTY_DIR}/lw/include)

# ---- ONNX Runtime 1.29.0: shared + C API headers from the official AAR ----
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION ${THIRD_PARTY_DIR}/ort/lib/${ABI_DIR}/libonnxruntime.so
        INTERFACE_INCLUDE_DIRECTORIES ${THIRD_PARTY_DIR}/ort/include)

# ---- MNN 3.6.1 ----
add_library(MNN SHARED IMPORTED)
set_target_properties(MNN PROPERTIES
        IMPORTED_LOCATION ${THIRD_PARTY_DIR}/mnn/lib/${ABI_DIR}/libMNN.so
        INTERFACE_INCLUDE_DIRECTORIES ${THIRD_PARTY_DIR}/mnn/include)

# ---- ncnn: rebuilt from source with Vulkan ----
# The official prebuilt is NCNN_VULKAN=OFF, which would make ncnn the only
# engine unable to reach the GPU and turn "strongest config" into "fastest CPU
# path" for it. stage/ncnn-vulkan comes from
# scripts/build_ncnn_vulkan_android.sh (NCNN_VULKAN=ON, NCNN_SIMPLEVK=ON, so
# libvulkan.so is dlopen()ed at runtime and needs no link dependency).
add_library(ncnn STATIC IMPORTED)
set_target_properties(ncnn PROPERTIES
        IMPORTED_LOCATION ${THIRD_PARTY_DIR}/ncnn-vulkan/lib/${ABI_DIR}/libncnn.a
        INTERFACE_INCLUDE_DIRECTORIES ${THIRD_PARTY_DIR}/ncnn-vulkan/include)

# ncnn compiles GLSL -> SPIR-V at runtime and therefore drags in glslang even
# though every standard layer already ships precompiled SPIR-V.
add_library(ncnn_glslang STATIC IMPORTED)
set_target_properties(ncnn_glslang PROPERTIES
        IMPORTED_LOCATION ${THIRD_PARTY_DIR}/ncnn-vulkan/lib/${ABI_DIR}/libglslang.a)
add_library(ncnn_spirv STATIC IMPORTED)
set_target_properties(ncnn_spirv PROPERTIES
        IMPORTED_LOCATION ${THIRD_PARTY_DIR}/ncnn-vulkan/lib/${ABI_DIR}/libSPIRV.a)

add_library(ppocr_bench SHARED
        bench_common.cpp
        det_pipeline.cpp
        backend_lw.cpp
        backend_ort.cpp
        backend_mnn.cpp
        backend_ncnn.cpp
        jni_bridge.cpp)

target_compile_features(ppocr_bench PRIVATE cxx_std_17)
# ORT's CXX wrapper and the MNN headers throw/catch, so exceptions and RTTI stay on.
target_compile_options(ppocr_bench PRIVATE -O3 -fvisibility=hidden -funroll-loops)

# ncnn's prebuilt static library is compiled with OpenMP (it references
# __kmpc_* symbols), so the LLVM OpenMP runtime from the NDK must come along.
# Linking libomp.a statically keeps the APK free of an extra .so.
get_filename_component(CLANG_BIN_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
get_filename_component(TOOLCHAIN_ROOT ${CLANG_BIN_DIR} DIRECTORY)
# NOTE: the NDK layout uses LLVM-style ABI names (aarch64), not Android ones
# (arm64-v8a), so CMAKE_ANDROID_ARCH_ABI cannot be dropped into the path.
file(GLOB OMP_A_ALL ${TOOLCHAIN_ROOT}/lib/clang/*/lib/linux/*/libomp.a)
foreach(candidate ${OMP_A_ALL})
    if(candidate MATCHES "aarch64")
        set(OMP_LIB ${candidate})
    endif()
endforeach()
if(NOT OMP_LIB)
    message(FATAL_ERROR "NDK libomp.a for aarch64 not found under ${TOOLCHAIN_ROOT}/lib/clang")
endif()
message(STATUS "libomp: ${OMP_LIB}")

# glslang and SPIRV are mutually dependent archives, so glslang appears twice
# to give the single-pass linker a second chance at resolving the cycle.
target_link_libraries(ppocr_bench PRIVATE lw onnxruntime MNN ncnn
        ncnn_glslang ncnn_spirv ncnn_glslang ${OMP_LIB}
        log android jnigraphics)
