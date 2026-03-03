# CPU Backend
# Always available, uses SIMD and OpenMP for optimization
#
# Optimization Headers (header-only, included where needed):
#   kernels/reduction_optimized.hpp    - Pre-computed strides, SIMD horizontal reductions
#   kernels/float16_simd.hpp           - F16C SIMD intrinsics for Float16 operations
#   kernels/buffer_pool.hpp            - Thread-local buffer pool for temp allocations
#   kernels/fused_conv_bn_relu.hpp     - Fused Conv2d + BatchNorm + ReLU kernel
#   kernels/activations_backward_simd.hpp - SIMD backward passes for activations
#   kernels/gemm_packing_simd.hpp      - SIMD-accelerated GEMM packing
#   kernels/fused_linear_optimized.hpp - Optimized fused linear layer operations
#   kernels/fused_attention.hpp        - Fused attention operations for transformers
#   kernels/gemm_optimized.hpp         - High-performance GEMM micro-kernels
#   kernels/simd_fast_math.hpp         - Fast transcendental approximations

# Option for portable binary builds (multi-ISA kernel compilation)
option(TENZOR_PORTABLE "Build portable binaries with runtime SIMD dispatch instead of -march=native" OFF)

set(CPU_BACKEND_SOURCES
    cpu_backend.cpp
    cpu_caching_allocator.cpp
    cpu_kernel_registry.cpp
    simd.cpp
    kernels/math.cpp
    kernels/math_simd.cpp
    kernels/math_simd_avx512.cpp
    kernels/reduction.cpp
    kernels/transform.cpp
    kernels/indexing.cpp
    kernels/activations.cpp
    kernels/activations_simd.cpp
    kernels/activations_simd_avx512.cpp
    kernels/batchnorm.cpp
    kernels/creation.cpp
    kernels/conv2d.cpp
    kernels/conv3d.cpp
    kernels/fused_ops.cpp
    kernels/simd_dispatch.cpp
    kernels/pooling.cpp
    kernels/nn_kernels.cpp
    kernels/rnn_kernels.cpp
    kernels/vision.cpp
    kernels/advanced.cpp
    kernels/flash_attention.cpp
)

# Create CPU backend shared library
add_library(tenzor_backend_cpu SHARED ${CPU_BACKEND_SOURCES})

# Apply Tenzor-specific warning and optimization flags
target_compile_options(tenzor_backend_cpu PRIVATE ${TENZOR_WARNING_FLAGS})

# Set target properties
set_target_properties(tenzor_backend_cpu PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    OUTPUT_NAME "tenzor_backend_cpu"
    PREFIX ""  # Remove 'lib' prefix on Unix
)

# Include directories
target_include_directories(tenzor_backend_cpu PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Link against core library
target_link_libraries(tenzor_backend_cpu PRIVATE
    tenzor_core
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(tenzor_backend_cpu PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "CPU Backend: OpenMP enabled")
endif()

# Platform-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(TENZOR_PORTABLE)
        # ============================================================
        # Portable build: compile at baseline x86-64 level, use
        # function-level __attribute__((target(...))) for multi-ISA
        # dispatch. The runtime SIMD detection system
        # (runtime_simd.hpp / SIMDKernelTable) selects the best
        # kernel at startup.
        # ============================================================
        message(STATUS "CPU Backend: PORTABLE mode - baseline x86-64 with runtime SIMD dispatch")

        target_compile_options(tenzor_backend_cpu PRIVATE
            -fno-math-errno
            -fno-trapping-math
            -fno-finite-math-only  # Preserves NaN/Inf handling for IEEE 754 compliance
            -Wno-unused-parameter
        )

        target_compile_definitions(tenzor_backend_cpu PRIVATE TENZOR_PORTABLE)

        # In portable mode, SIMD kernel files that use intrinsics must use
        # __attribute__((target("avx2"))) etc. on each function. The existing
        # simd_dispatch.cpp in src/backend/ already does this. The CPU-backend
        # SIMD files (math_simd.cpp, activations_simd.cpp) use #ifdef guards
        # so they fall back to scalar when compiled without -march=native.
        #
        # For files that need specific ISA to compile at all (e.g., they use
        # intrinsics at file scope), we set per-file compile flags:
        set_source_files_properties(
            kernels/math_simd.cpp
            kernels/activations_simd.cpp
            kernels/simd_dispatch.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c"
        )

        # AVX-512 implementations are in separate translation units so they
        # can be compiled with -mavx512f even in portable mode. Runtime
        # dispatch (CPUInfo::has_avx512()) selects them only on capable hardware.
        set_source_files_properties(
            kernels/math_simd_avx512.cpp
            kernels/activations_simd_avx512.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx2;-mfma;-mf16c"
        )

        # F16C and FMA are available in portable mode via per-function attributes
        # or per-file flags above, so we define these for the SIMD kernel files
        set_source_files_properties(
            kernels/math_simd.cpp
            kernels/activations_simd.cpp
            kernels/simd_dispatch.cpp
            kernels/math_simd_avx512.cpp
            kernels/activations_simd_avx512.cpp
            PROPERTIES COMPILE_DEFINITIONS "TENZOR_HAS_F16C;TENZOR_HAS_FMA"
        )

    else()
        # ============================================================
        # Native build (default): use -march=native for maximum
        # performance on the build host. Not portable to CPUs with
        # fewer features than the build machine.
        # ============================================================
        message(STATUS "CPU Backend: NATIVE mode - using -march=native")

        # Enable SIMD optimizations with IEEE 754 compliance preserved
        target_compile_options(tenzor_backend_cpu PRIVATE
            -march=native
            -mtune=native
            -fno-math-errno
            -fno-trapping-math
            -fno-finite-math-only  # Preserves NaN/Inf handling for IEEE 754 compliance
            -Wno-unused-parameter
        )

        # AVX-512 translation units always get -mavx512f so that even on
        # non-AVX-512 build hosts, the binary contains real AVX-512 code
        # available for runtime dispatch on capable hardware.
        set_source_files_properties(
            kernels/math_simd_avx512.cpp
            kernels/activations_simd_avx512.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx512f"
        )

        # Check for specific SIMD support
        # With -march=native, the compiler will emit the correct instructions
        # We detect CPU capabilities at compile time via compiler predefined macros
        message(STATUS "CPU Backend: Using -march=native for automatic SIMD detection")

        # Check for F16C support (required for Float16 SIMD operations)
        include(CheckCXXSourceCompiles)
        set(CMAKE_REQUIRED_FLAGS "-march=native")
        check_cxx_source_compiles("
            #include <immintrin.h>
            int main() {
                __m128i packed = _mm_setzero_si128();
                __m256 unpacked = _mm256_cvtph_ps(packed);
                (void)unpacked;
                return 0;
            }
        " TENZOR_HAS_F16C)

        if(TENZOR_HAS_F16C)
            target_compile_definitions(tenzor_backend_cpu PRIVATE TENZOR_HAS_F16C)
            message(STATUS "CPU Backend: F16C (Float16 conversion) support enabled")
        else()
            message(STATUS "CPU Backend: F16C not available, Float16 ops will use scalar fallback")
        endif()

        # Check for FMA support
        check_cxx_source_compiles("
            #include <immintrin.h>
            int main() {
                __m256 a = _mm256_setzero_ps();
                __m256 b = _mm256_setzero_ps();
                __m256 c = _mm256_fmadd_ps(a, b, a);
                (void)c;
                return 0;
            }
        " TENZOR_HAS_FMA)

        if(TENZOR_HAS_FMA)
            target_compile_definitions(tenzor_backend_cpu PRIVATE TENZOR_HAS_FMA)
            message(STATUS "CPU Backend: FMA (Fused Multiply-Add) support enabled")
        endif()

        # Check for AVX-512 support
        check_cxx_source_compiles("
            #include <immintrin.h>
            int main() {
                __m512 a = _mm512_setzero_ps();
                __m512 b = _mm512_add_ps(a, a);
                (void)b;
                return 0;
            }
        " TENZOR_HAS_AVX512)

        if(TENZOR_HAS_AVX512)
            target_compile_definitions(tenzor_backend_cpu PRIVATE TENZOR_HAS_AVX512)
            message(STATUS "CPU Backend: AVX-512 support enabled")
        else()
            message(STATUS "CPU Backend: Using AVX2 (AVX-512 not available)")
        endif()
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    if(TENZOR_PORTABLE)
        message(STATUS "CPU Backend: PORTABLE mode (MSVC) - baseline with runtime dispatch")
        target_compile_options(tenzor_backend_cpu PRIVATE
            /fp:fast
        )
        target_compile_definitions(tenzor_backend_cpu PRIVATE TENZOR_PORTABLE)
    else()
        target_compile_options(tenzor_backend_cpu PRIVATE
            /arch:AVX2
            /fp:fast
        )
    endif()
endif()

# Optional: Intel MKL for high-performance GEMM
# MKL provides highly optimized SGEMM/DGEMM that can be 5-10x faster
# IMPORTANT: MKL is found FIRST to ensure we use GNU threading (not Intel OpenMP)
# to avoid conflicts with GCC's libgomp. If we let find_package(BLAS) find MKL first,
# it uses intel_thread which links libiomp5, causing dual OpenMP runtime issues.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(${CMAKE_SOURCE_DIR}/cmake/FindMKL.cmake OPTIONAL RESULT_VARIABLE MKL_MODULE_FOUND)
if(MKL_MODULE_FOUND AND MKL_FOUND)
    target_link_libraries(tenzor_backend_cpu PRIVATE ${MKL_LIBRARIES})
    target_include_directories(tenzor_backend_cpu PRIVATE ${MKL_INCLUDE_DIRS})
    # MKL_ILP64 is required when linking against libmkl_intel_ilp64 (64-bit integers)
    target_compile_definitions(tenzor_backend_cpu PRIVATE TENZOR_USE_MKL TENZOR_HAS_BLAS MKL_ILP64)
    message(STATUS "CPU Backend: Intel MKL enabled for optimized GEMM")
else()
    # Fallback: Link against generic BLAS library (OpenBLAS, etc.)
    find_package(BLAS QUIET)
    if(BLAS_FOUND)
        target_link_libraries(tenzor_backend_cpu PRIVATE ${BLAS_LIBRARIES})
        target_compile_definitions(tenzor_backend_cpu PRIVATE TENZOR_HAS_BLAS)
        message(STATUS "CPU Backend: BLAS support enabled")
    else()
        message(STATUS "CPU Backend: BLAS not found, using built-in matrix operations")
    endif()
endif()

# Optional: Intel oneDNN for optimized DNN primitives (Conv, BatchNorm, etc.)
include(${CMAKE_SOURCE_DIR}/cmake/FindOneDNN.cmake OPTIONAL RESULT_VARIABLE ONEDNN_MODULE_FOUND)
if(ONEDNN_MODULE_FOUND AND ONEDNN_FOUND)
    target_link_libraries(tenzor_backend_cpu PRIVATE ${ONEDNN_LIBRARIES})
    target_include_directories(tenzor_backend_cpu PRIVATE ${ONEDNN_INCLUDE_DIRS})
    target_compile_definitions(tenzor_backend_cpu PRIVATE TENZOR_USE_ONEDNN)
    message(STATUS "CPU Backend: Intel oneDNN enabled for optimized DNN primitives")
endif()

# Installation
install(TARGETS tenzor_backend_cpu
    LIBRARY DESTINATION lib/tenzor/backends
    RUNTIME DESTINATION bin
)

message(STATUS "CPU Backend: Configured successfully")
