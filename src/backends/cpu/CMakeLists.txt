# CPU Backend
# Always available, uses SIMD and OpenMP for optimization
#
# Optimization Headers (header-only, included where needed):
#   kernels/float16_simd.hpp           - F16C SIMD intrinsics for Float16 operations
#   kernels/buffer_pool.hpp            - Thread-local buffer pool for temp allocations
#   kernels/fused_conv_bn_relu.hpp     - Fused Conv2d + BatchNorm + ReLU kernel
#   kernels/gemm_optimized.hpp         - High-performance GEMM micro-kernels
#   kernels/simd_fast_math.hpp         - Fast transcendental approximations
# (Audit item F.1: removed reduction_optimized.hpp,
#  activations_backward_simd.hpp, gemm_packing_simd.hpp,
#  fused_linear_optimized.hpp, quantized_matmul.hpp, transpose_simd.hpp
#  — all six were orphan experiments with zero non-self includers.)

# Option for portable binary builds (multi-ISA kernel compilation)
option(TENZOR_PORTABLE "Build portable binaries with runtime SIMD dispatch instead of -march=native" OFF)

set(CPU_BACKEND_SOURCES
    cpu_backend.cpp
    cpu_caching_allocator.cpp
    cpu_thread_config.cpp
    cpu_kernel_registry.cpp
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
    kernels/batchnorm_avx512.cpp
    kernels/creation.cpp
    kernels/conv2d.cpp
    kernels/conv3d.cpp
    kernels/depthwise_conv1d.cpp
    kernels/depthwise_conv3d.cpp
    kernels/fused_ops.cpp
    kernels/simd_dispatch.cpp
    kernels/simd_kernels_avx512_f64.cpp
    kernels/pooling.cpp
    kernels/pooling_avx512.cpp
    kernels/nn_kernels.cpp
    kernels/rnn_kernels.cpp
    kernels/nested_kernels.cpp
    kernels/vision.cpp
    kernels/grid_sample.cpp
    kernels/advanced.cpp
    kernels/flash_attention.cpp
    kernels/fft.cpp
    kernels/ctc.cpp
    kernels/onednn_cache.cpp
)

# Create CPU backend shared library
add_library(tenzor_backend_cpu SHARED ${CPU_BACKEND_SOURCES})

# Apply Tenzor-specific warning and optimization flags
target_compile_options(tenzor_backend_cpu PRIVATE ${TENZOR_WARNING_FLAGS})
target_compile_definitions(tenzor_backend_cpu PRIVATE TENZOR_INTERNAL)

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
else()
    message(WARNING "CPU Backend: OpenMP not found — parallel operations will use a single thread")
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
        # dispatch (tenzor::backend::get_simd_features().avx512f) selects them
        # only on capable hardware.
        set_source_files_properties(
            kernels/math_simd_avx512.cpp
            kernels/activations_simd_avx512.cpp
            kernels/batchnorm_avx512.cpp
            kernels/pooling_avx512.cpp
            kernels/simd_kernels_avx512_f64.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512dq;-mavx2;-mfma;-mf16c"
        )

        # F16C and FMA are available in portable mode via per-function attributes
        # or per-file flags above, so we define these for the SIMD kernel files
        set_source_files_properties(
            kernels/math_simd.cpp
            kernels/activations_simd.cpp
            kernels/simd_dispatch.cpp
            kernels/math_simd_avx512.cpp
            kernels/activations_simd_avx512.cpp
            kernels/simd_kernels_avx512_f64.cpp
            PROPERTIES COMPILE_DEFINITIONS "TENZOR_HAS_F16C;TENZOR_HAS_FMA"
        )

        # Fix 3 (revised — SIGILL hardening): AVX-512 capability check in
        # PORTABLE mode.
        #
        # ROOT-CAUSE FIX: previously this set TENZOR_HAS_AVX512 *target-wide*
        # in portable mode. That was a real SIGILL-class bug: TENZOR_HAS_AVX512
        # is used by the BASELINE translation units (math.cpp, activations.cpp,
        # creation.cpp, simd_elementwise.hpp) as a *codegen gate* — each of them
        # already does `#if defined(__AVX512F__) #define TENZOR_HAS_AVX512 1`,
        # i.e. the macro is meant to mirror the *current TU's* compile target.
        # Those baseline TUs are compiled at -m64 (no -mavx512f) in portable
        # mode, so __AVX512F__ is unset and they correctly fall back to scalar.
        # Forcing -DTENZOR_HAS_AVX512 target-wide overrode that local logic and
        # made the `#ifdef TENZOR_HAS_AVX512` blocks emit _mm512_* intrinsics in
        # a baseline TU → AVX-512 instructions a non-AVX512 CPU cannot execute
        # → SIGILL at runtime (and a hard "target specific option mismatch"
        # compile error under GCC, which is how this was caught).
        #
        # The dedicated AVX-512 TUs (math_simd_avx512.cpp and friends) receive
        # -mavx512f per-file above, so they define TENZOR_HAS_AVX512 *locally*
        # via their own __AVX512F__ block and are selected at runtime by the
        # SIMD dispatcher (runtime_simd.hpp / SIMDKernelTable) only on capable
        # hardware. They therefore need NO target-wide define.
        #
        # We still probe the compiler so the build log accurately reports
        # whether AVX-512 codegen is available; runtime dispatch gates actual
        # execution on the running CPU's features. We deliberately do NOT add a
        # target-wide TENZOR_HAS_AVX512 define.
        include(CheckCXXSourceCompiles)
        set(CMAKE_REQUIRED_FLAGS "-mavx512f")
        check_cxx_source_compiles("
            #include <immintrin.h>
            int main() {
                __m512 a = _mm512_setzero_ps();
                __m512 b = _mm512_add_ps(a, a);
                (void)b;
                return 0;
            }
        " TENZOR_HAS_AVX512_PORTABLE)
        unset(CMAKE_REQUIRED_FLAGS)

        if(TENZOR_HAS_AVX512_PORTABLE)
            message(STATUS "CPU Backend: AVX-512 codegen available in PORTABLE mode (runtime-dispatched via the dedicated *_avx512 TUs; NOT defined target-wide so baseline TUs stay scalar and cannot SIGILL)")
        else()
            message(STATUS "CPU Backend: PORTABLE mode - compiler does not support -mavx512f, AVX-512 kernels disabled")
        endif()

    else()
        # ============================================================
        # Native build (default): use -march=native for maximum
        # performance on the build host. Not portable to CPUs with
        # fewer features than the build machine.
        # ============================================================
        message(STATUS "CPU Backend: NATIVE mode - using -march=native")
        message(WARNING
            "CPU Backend: NATIVE build (-march=native) is host-pinned and NOT "
            "redistributable: a binary built on an AVX-512 host will SIGILL on "
            "an AVX2-only deploy target. Configure with -DTENZOR_PORTABLE=ON for "
            "a portable, runtime-dispatched build.")

        # Enable SIMD optimizations with IEEE 754 compliance preserved
        target_compile_options(tenzor_backend_cpu PRIVATE
            -march=native
            -mtune=native
            -fno-math-errno
            -fno-trapping-math
            -fno-finite-math-only  # Preserves NaN/Inf handling for IEEE 754 compliance
        )

        # AVX-512 translation units get -mavx512f so that even on non-AVX-512
        # build hosts, the binary contains real AVX-512 code available for
        # runtime dispatch on capable hardware. This is gated on the COMPILER
        # actually accepting -mavx512f (probed below) rather than applied
        # unconditionally: an old toolchain that does not understand the flag
        # would otherwise fail to compile those TUs. The runtime dispatcher
        # (CPUID/XCR0) still decides whether the AVX-512 code path executes.
        include(CheckCXXSourceCompiles)
        set(CMAKE_REQUIRED_FLAGS "-mavx512f")
        check_cxx_source_compiles("
            #include <immintrin.h>
            int main() {
                __m512 a = _mm512_setzero_ps();
                __m512 b = _mm512_add_ps(a, a);
                (void)b;
                return 0;
            }
        " TENZOR_HAS_AVX512)
        unset(CMAKE_REQUIRED_FLAGS)

        if(TENZOR_HAS_AVX512)
            set_source_files_properties(
                kernels/math_simd_avx512.cpp
                kernels/activations_simd_avx512.cpp
                kernels/batchnorm_avx512.cpp
                kernels/pooling_avx512.cpp
                kernels/simd_kernels_avx512_f64.cpp
                PROPERTIES COMPILE_OPTIONS "-mavx512f"
            )
        endif()

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

        # AVX-512 support was probed above (TENZOR_HAS_AVX512, via the -mavx512f
        # compiler-capability check that also gates the per-file flag). Publish
        # the compile definition target-wide so #ifdef TENZOR_HAS_AVX512 in
        # headers / non-AVX512 TUs agrees with the per-file codegen.
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
        # /fp:precise (not /fp:fast) to preserve IEEE-754 NaN/Inf handling,
        # mirroring the GCC/Clang -fno-finite-math-only policy. /fp:fast assumes
        # finite math and may optimize away isnan/isinf checks and NaN propagation
        # (e.g. softmax -inf masking), silently diverging from the other compilers.
        target_compile_options(tenzor_backend_cpu PRIVATE
            /fp:precise
        )
        target_compile_definitions(tenzor_backend_cpu PRIVATE TENZOR_PORTABLE)

        # MSVC AVX2 files need /arch:AVX2 even in portable mode
        set_source_files_properties(
            kernels/math_simd.cpp
            kernels/activations_simd.cpp
            kernels/simd_dispatch.cpp
            PROPERTIES COMPILE_OPTIONS "/arch:AVX2"
        )
        set_source_files_properties(
            kernels/math_simd.cpp
            kernels/activations_simd.cpp
            kernels/simd_dispatch.cpp
            PROPERTIES COMPILE_DEFINITIONS "TENZOR_HAS_F16C;TENZOR_HAS_FMA"
        )

        # AVX-512 files need /arch:AVX512 (MSVC 2019 16.9+)
        set_source_files_properties(
            kernels/math_simd_avx512.cpp
            kernels/activations_simd_avx512.cpp
            kernels/batchnorm_avx512.cpp
            kernels/pooling_avx512.cpp
            kernels/simd_kernels_avx512_f64.cpp
            PROPERTIES COMPILE_OPTIONS "/arch:AVX512"
        )

        # Fix 3 (MSVC portable): mirror native-branch AVX-512 capability check
        # so TENZOR_HAS_AVX512 is visible to non-AVX512 TUs in portable mode.
        include(CheckCXXSourceCompiles)
        set(CMAKE_REQUIRED_FLAGS "/arch:AVX512")
        check_cxx_source_compiles("
            #include <immintrin.h>
            int main() {
                __m512 a = _mm512_setzero_ps();
                __m512 b = _mm512_add_ps(a, a);
                (void)b;
                return 0;
            }
        " TENZOR_HAS_AVX512_PORTABLE_MSVC)
        unset(CMAKE_REQUIRED_FLAGS)

        if(TENZOR_HAS_AVX512_PORTABLE_MSVC)
            target_compile_definitions(tenzor_backend_cpu PRIVATE TENZOR_HAS_AVX512)
            message(STATUS "CPU Backend: AVX-512 codegen available in PORTABLE mode (MSVC, runtime-dispatched)")
        else()
            message(STATUS "CPU Backend: PORTABLE mode (MSVC) - /arch:AVX512 not supported, AVX-512 kernels disabled")
        endif()
    else()
        message(STATUS "CPU Backend: NATIVE mode (MSVC) - using /arch:AVX2")
        # /fp:precise (not /fp:fast) to preserve IEEE-754 NaN/Inf handling,
        # mirroring the GCC/Clang -fno-finite-math-only policy (see portable branch).
        target_compile_options(tenzor_backend_cpu PRIVATE
            /arch:AVX2
            /fp:precise
        )

        # Check for F16C support (available with AVX2 on MSVC)
        include(CheckCXXSourceCompiles)
        set(CMAKE_REQUIRED_FLAGS "/arch:AVX2")
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
        endif()

        # FMA is always available with AVX2 on MSVC
        target_compile_definitions(tenzor_backend_cpu PRIVATE TENZOR_HAS_FMA)
        message(STATUS "CPU Backend: FMA (Fused Multiply-Add) support enabled")

        # Check for AVX-512 support
        set(CMAKE_REQUIRED_FLAGS "/arch:AVX512")
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

            # AVX-512 translation units
            set_source_files_properties(
                kernels/math_simd_avx512.cpp
                kernels/activations_simd_avx512.cpp
                kernels/batchnorm_avx512.cpp
                kernels/pooling_avx512.cpp
                kernels/simd_kernels_avx512_f64.cpp
                PROPERTIES COMPILE_OPTIONS "/arch:AVX512"
            )
        else()
            message(STATUS "CPU Backend: Using AVX2 (AVX-512 not available)")
        endif()
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
    # Do NOT define MKL_ILP64: FindMKL links the LP64 interface (mkl_intel_lp64,
    # 32-bit MKL_INT). Defining MKL_ILP64 makes the MKL headers' MKL_INT 64-bit,
    # mismatching the linked LP64 library (only by-value cblas_* args survive the
    # ABI mismatch; any pointer/array MKL_INT would be corrupted).
    target_compile_definitions(tenzor_backend_cpu PRIVATE TENZOR_USE_MKL TENZOR_HAS_BLAS)
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
    # oneDNN's pre-built libdnnl.so links libtbb.so.12 (release). During library
    # init, libtbb.so does dlsym("scalable_free") to find tbbmalloc. If tbbmalloc
    # isn't loaded yet, the function pointer stays NULL, causing a segfault in
    # __TBB_InitOnce's destructor during process exit.
    #
    # In Debug builds this is worse: dnnl's FindTBB selects libtbb_debug.so which
    # loads libtbbmalloc_debug.so, but libtbb.so (release, from libdnnl.so) inits
    # before either release tbbmalloc variant is available.
    #
    # Fix: link libtbbmalloc.so BEFORE libdnnl.so so the dynamic linker loads it
    # first, making scalable_free available when libtbb.so initializes.
    # oneDNN links libtbb.so.12 which does dlsym("scalable_free") during its
    # static init to find tbbmalloc. We must ensure libtbbmalloc.so.2 (which
    # provides scalable_free) is loaded and initialized BEFORE libtbb.so.12.
    # Directly linking both libraries before dnnl puts them in the NEEDED list
    # ahead of dnnl's transitive dependencies.
    # Search for TBB libs via env-controlled HINTS only. The previous version
    # had a hard-coded "/opt/intel/oneapi/2025.3/lib" path that pinned the
    # build to a specific oneAPI point release and broke on every bump.
    # Users / CI should set TBBROOT or ONEAPI_ROOT (oneAPI's setvars.sh does
    # both). System TBB (e.g. apt's libtbb-dev) is picked up via the default
    # search path.
    find_library(_tbb_release NAMES tbb
        HINTS "${TBBROOT}/lib" "$ENV{TBBROOT}/lib"
              "${ONEAPI_ROOT}/lib" "$ENV{ONEAPI_ROOT}/lib"
              "${ONEAPI_ROOT}/tbb/latest/lib" "$ENV{ONEAPI_ROOT}/tbb/latest/lib")
    find_library(_tbbmalloc_release NAMES tbbmalloc
        HINTS "${TBBROOT}/lib" "$ENV{TBBROOT}/lib"
              "${ONEAPI_ROOT}/lib" "$ENV{ONEAPI_ROOT}/lib"
              "${ONEAPI_ROOT}/tbb/latest/lib" "$ENV{ONEAPI_ROOT}/tbb/latest/lib")
    if(_tbbmalloc_release AND _tbb_release)
        # Link tbbmalloc FIRST, then tbb — ensures tbbmalloc inits before tbb
        target_link_libraries(tenzor_backend_cpu PRIVATE
            ${_tbbmalloc_release} ${_tbb_release})
        message(STATUS "CPU Backend: Linked release TBB libs for init order fix")
    else()
        # Fix 4: oneDNN is enabled, but TBB libs were not located. oneDNN's
        # libdnnl.so has libtbb.so.12 in its NEEDED list and the load-order
        # workaround above relies on linking libtbbmalloc.so.2 ahead of
        # libdnnl. Silently skipping would yield a binary that segfaults at
        # process exit (or worse, mid-run on Debug builds where libtbb_debug
        # is selected). Fail the configure so the user installs TBB or
        # sources oneAPI's setvars.sh first.
        message(FATAL_ERROR
            "CPU Backend: Intel oneDNN is enabled but TBB (libtbb / libtbbmalloc) "
            "was not found. Install libtbb-dev (Debian/Ubuntu) or set TBBROOT / "
            "ONEAPI_ROOT to a oneAPI tree before re-running cmake. To disable "
            "oneDNN entirely, configure with -DTENZOR_USE_ONEDNN=OFF (or remove "
            "the oneDNN install from CMAKE_PREFIX_PATH).")
    endif()

    target_link_libraries(tenzor_backend_cpu PRIVATE ${ONEDNN_LIBRARIES})
    target_include_directories(tenzor_backend_cpu PRIVATE ${ONEDNN_INCLUDE_DIRS})
    target_compile_definitions(tenzor_backend_cpu PRIVATE TENZOR_USE_ONEDNN)
    message(STATUS "CPU Backend: Intel oneDNN enabled for optimized DNN primitives")
endif()

# Installation (FHS layout only; wheel installs are handled centrally in the
# root CMakeLists.txt SKBUILD branch)
if(NOT SKBUILD)
    install(TARGETS tenzor_backend_cpu
        LIBRARY DESTINATION lib/tenzor/backends
        RUNTIME DESTINATION bin
    )
endif()

message(STATUS "CPU Backend: Configured successfully")
