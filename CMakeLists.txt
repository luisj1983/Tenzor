cmake_minimum_required(VERSION 3.25)

# Default to Release build if no build type specified (single-config generators only)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Pre-project configuration for ROCm/HIP
# Must happen before project() because HIP language must be enabled at project() time
# Locate ROCm root: prefer ROCM_PATH env, then find hipcc, then fallback to /opt/rocm
set(_ROCM_AVAILABLE FALSE)
if(DEFINED ENV{ROCM_PATH})
    set(_ROCM_ROOT "$ENV{ROCM_PATH}")
else()
    # ROCm 6.x+/7.x split the toolchain into a "core/" subdirectory, so hipcc
    # may be at /opt/rocm/core/bin/hipcc rather than the legacy /opt/rocm/bin.
    find_program(_HIPCC_EXE hipcc PATHS /opt/rocm/bin /opt/rocm/core/bin NO_DEFAULT_PATH)
    if(_HIPCC_EXE)
        get_filename_component(_ROCM_ROOT "${_HIPCC_EXE}" DIRECTORY)   # .../core/bin or .../bin
        get_filename_component(_ROCM_ROOT "${_ROCM_ROOT}" DIRECTORY)   # .../core   or /opt/rocm
        # Normalise the new "core/" split layout back to the ROCm top-level so
        # ${_ROCM_ROOT}/core/lib and ${_ROCM_ROOT}/include resolve correctly.
        if(_ROCM_ROOT MATCHES "/core$")
            get_filename_component(_ROCM_ROOT "${_ROCM_ROOT}" DIRECTORY)
        endif()
    else()
        set(_ROCM_ROOT "/opt/rocm")
    endif()
endif()
if(EXISTS "${_ROCM_ROOT}/bin/hipcc" OR EXISTS "${_ROCM_ROOT}/core/bin/hipcc")
    set(_ROCM_AVAILABLE TRUE)
endif()


# Set TENZOR_BUILD_ROCM before project() so the HIP language can be enabled.
# Defaults ON (GPU backends are on by default); pass -DTENZOR_BUILD_ROCM=OFF to
# opt out. The previous `if(NOT DEFINED ...)` block was dead code — option()
# above already defines the variable, so the guard was never taken and only
# contradicted the ON default.
option(TENZOR_BUILD_ROCM "Build ROCm backend" ON)

# Set up HIP architectures if building ROCm backend
if(TENZOR_BUILD_ROCM AND _ROCM_AVAILABLE)
    if(NOT CMAKE_HIP_ARCHITECTURES)
        # Try to auto-detect GPU architecture. The enumerator moved with the
        # ROCm 6.x+/7.x "core/" split (now /opt/rocm/core/bin), so resolve the
        # actual path rather than assuming ${_ROCM_ROOT}/bin.
        find_program(ROCM_AGENT_ENUMERATOR rocm_agent_enumerator
            PATHS ${_ROCM_ROOT}/bin ${_ROCM_ROOT}/core/bin NO_DEFAULT_PATH)
        execute_process(
            COMMAND ${ROCM_AGENT_ENUMERATOR}
            OUTPUT_VARIABLE ROCM_AGENTS
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE ROCM_ENUM_RESULT
            ERROR_VARIABLE ROCM_ENUM_ERROR
        )
        if(NOT ROCM_ENUM_RESULT EQUAL 0)
            message(WARNING "ROCm agent enumeration failed (exit code ${ROCM_ENUM_RESULT}): ${ROCM_ENUM_ERROR}")
            message(WARNING "Set CMAKE_HIP_ARCHITECTURES manually for best performance.")
        endif()

        # Extract first gfx architecture (skip CPU)
        string(REGEX MATCH "gfx[0-9a-z]+" DETECTED_ARCH "${ROCM_AGENTS}")

        if(DETECTED_ARCH)
            set(CMAKE_HIP_ARCHITECTURES "${DETECTED_ARCH}" CACHE STRING "HIP GPU architectures" FORCE)
            message(STATUS "Auto-detected GPU architecture: ${DETECTED_ARCH}")
        else()
            # Fallback to common architectures
            set(CMAKE_HIP_ARCHITECTURES "gfx906;gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1150" CACHE STRING "HIP GPU architectures" FORCE)
            message(STATUS "Could not detect GPU, using common architectures: gfx906;gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1150")
        endif()
    endif()
endif()

# Determine languages to enable
set(TENZOR_LANGUAGES CXX)
if(TENZOR_BUILD_ROCM AND _ROCM_AVAILABLE)
    list(APPEND TENZOR_LANGUAGES HIP)
endif()

project(tenzor VERSION 0.1.0 LANGUAGES ${TENZOR_LANGUAGES})

# tenzor_core is a SHARED library that statically links FetchContent'd
# dependencies (spdlog when no system package is found, etc.). CMake compiles
# SHARED library targets' own sources with -fPIC automatically, but that does
# NOT propagate to a STATIC dependency built without it -- on toolchains that
# don't default to -fPIC for static libs (stock Ubuntu/Debian GCC, unlike some
# distros that patch their spec files), linking such a static lib into
# tenzor_core.so fails at link time with "relocation R_X86_64_TPOFF32 ...
# can not be used when making a shared object; recompile with -fPIC". Setting
# this globally before any add_subdirectory()/FetchContent_MakeAvailable()
# ensures every target configured after this point -- including fetched
# static deps -- is compiled PIC.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Use ccache if available for faster rebuilds
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "ccache found: ${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(TENZOR_BUILD_CUDA "Build CUDA backend" ON)
# Note: TENZOR_BUILD_ROCM is set before project() to enable HIP language
option(TENZOR_BUILD_ONEAPI "Build OneAPI backend" ON)
option(TENZOR_BUILD_VULKAN "Build Vulkan backend" ON)
if(APPLE)
    option(TENZOR_BUILD_MPS "Build Apple Metal/MPS backend" ON)
endif()
option(TENZOR_BUILD_PYTHON "Build Python bindings" ON)
option(TENZOR_BUILD_TESTS "Build tests" ON)
option(TENZOR_BUILD_BENCHMARKS "Build benchmarks" ON)
# Expand the default multi-dtype parity matrix to include BFloat16. BF16 is
# the dominant training dtype on modern GPUs (CUDA TF32/H100, ROCm MI300,
# OneAPI PVC, Vulkan Cooperative Matrix), so the BF16 column of the parity
# matrix must not be ~5% of the F32 column by default. Audit-5 Y.29.
#
# Default policy:
#   * Release / RelWithDebInfo / MinSizeRel — ON (full parity matrix for CI).
#   * Debug                                 — OFF (dev iteration speed; the
#                                                  Debug build is for local
#                                                  edit-compile-debug loops,
#                                                  not the CI matrix).
#
# Tests that need BF16 unconditionally still use ALL_FLOAT_DTYPES regardless
# of this flag. See TESTING.md ("BFloat16 parity coverage") for the policy
# rationale.
set(_tenzor_bf16_default ON)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(_tenzor_bf16_default OFF)
endif()
option(TENZOR_TEST_BFLOAT16
       "Include BFloat16 in the default multi-dtype parity matrix"
       ${_tenzor_bf16_default})
unset(_tenzor_bf16_default)
if(TENZOR_TEST_BFLOAT16)
    add_compile_definitions(TENZOR_TEST_BFLOAT16)
endif()
option(TENZOR_FUZZ "Build libFuzzer fuzz targets (requires Clang)" OFF)
option(TENZOR_BUILD_EXAMPLES "Build examples" ON)
option(TENZOR_BUILD_MODEL_HUB "Build model hub (requires CURL and OpenSSL)" ON)
option(TENZOR_BUILD_SERVING "Build inference serving (requires gRPC for full support)" OFF)
option(TENZOR_BUILD_LITE "Build lightweight mobile/embedded inference runtime" OFF)
# PyTorch interop (python/torch_interop.cpp) is OFF by default: it #includes
# <torch/torch.h> and would otherwise make libtorch a hard dependency of the
# core Python module. When ON, we locate Torch and compile the interop TU into
# the module, defining TENZOR_HAS_TORCH so the `torch_interop` submodule is
# registered. When OFF, importing `tenzor.torch_interop` raises a clear
# ImportError pointing at this flag (see python/tenzor/__init__.py).
option(TENZOR_BUILD_TORCH_INTEROP "Build optional PyTorch interop (requires libtorch)" OFF)
option(TENZOR_REQUIRE_MKL "Require Intel MKL (FATAL_ERROR if not found)" OFF)
option(TENZOR_ENABLE_PROFILING "Enable NVTX/ROCTX profiling annotations" OFF)

# MLIR/StableHLO/IREE JIT pipeline (Phase 13)
#
# Auto-detect a usable IREE distribution so a plain `cmake -B build` on a host
# that HAS IREE enables the MLIR/IREE JIT tests BY DEFAULT — otherwise a default
# build compiles zero cross-backend IREE coverage (the numeric IREE tests live
# behind TENZOR_USE_MLIR_JIT). Detection keys off the same locations the enabled
# block below searches — an explicit IREE dir env var, the project-local
# third_party/iree_dist, or an iree-compile on PATH — and NEVER triggers the
# heavy from-source FetchContent build. This only sets the option DEFAULT: an
# explicit -DTENZOR_USE_MLIR_JIT=ON/OFF still wins, existing build dirs keep
# their cached value, and a host without IREE defaults to OFF (no IREE deps, no
# build break).
set(_tenzor_iree_marker "lib/cmake/IREE/IREECompilerConfig.cmake")
set(_tenzor_iree_autodetect OFF)
if(DEFINED ENV{IREE_DIR} AND EXISTS "$ENV{IREE_DIR}/${_tenzor_iree_marker}")
    set(_tenzor_iree_autodetect ON)
elseif(DEFINED ENV{TENZOR_IREE_DIR} AND EXISTS "$ENV{TENZOR_IREE_DIR}/${_tenzor_iree_marker}")
    set(_tenzor_iree_autodetect ON)
elseif(EXISTS "${PROJECT_SOURCE_DIR}/third_party/iree_dist/${_tenzor_iree_marker}")
    set(_tenzor_iree_autodetect ON)
else()
    find_program(_tenzor_iree_compile NAMES iree-compile)
    if(_tenzor_iree_compile)
        set(_tenzor_iree_autodetect ON)
    endif()
endif()
if(_tenzor_iree_autodetect)
    message(STATUS "Tenzor: IREE distribution detected — TENZOR_USE_MLIR_JIT "
                   "defaults to ON (override with -DTENZOR_USE_MLIR_JIT=OFF)")
endif()

option(TENZOR_USE_MLIR_JIT "Enable MLIR/StableHLO JIT (Phase 13)" ${_tenzor_iree_autodetect})
option(TENZOR_USE_INPROCESS_IREE "Link IREE compiler in-process (Phase 3C)" OFF)

# Pinned version — IREE distribution bundles MLIR + StableHLO. The 2026-05-19
# spec amendment (docs/superpowers/specs/2026-05-19-tz-jit-stablehlo-amendment-text-mlir.md)
# folds MLIR/StableHLO into a single IREE dependency surface; Tenzor emits MLIR
# as text and consumes IREE Compiler + Runtime C APIs only.
set(TENZOR_IREE_VERSION       3.0   CACHE STRING "Required IREE version (informational)")

if(TENZOR_USE_MLIR_JIT)
    message(STATUS "Tenzor: MLIR JIT enabled (text-based, IREE-driven per 2026-05-19 amendment)")
    message(STATUS "  IREE ${TENZOR_IREE_VERSION}")

    # IREE Compiler (embedding API) and Runtime — single dependency surface per
    # the 2026-05-19 spec amendment. MLIR/StableHLO come bundled inside IREE.
    #
    # Discovery (D + B hybrid):
    #   1. $IREE_DIR / $TENZOR_IREE_DIR env var (explicit override)
    #   2. third_party/iree_dist/lib/cmake/IREE inside the project tree
    #      (gitignored; lives outside source control. See docs/jit-mlir-setup.md
    #      for what should live here.)
    #   3. System-installed IREE via find_package's default search
    #   4. FetchContent source build (heavy: ~30-60 min cold, ~10 GB build dir
    #      because IREE bundles LLVM/StableHLO). Triggered only when 1-3 fail.
    set(_iree_local_dist "${PROJECT_SOURCE_DIR}/third_party/iree_dist")
    if(DEFINED ENV{IREE_DIR})
        list(APPEND CMAKE_PREFIX_PATH "$ENV{IREE_DIR}")
    elseif(DEFINED ENV{TENZOR_IREE_DIR})
        list(APPEND CMAKE_PREFIX_PATH "$ENV{TENZOR_IREE_DIR}")
    elseif(EXISTS "${_iree_local_dist}/lib/cmake/IREE/IREECompilerConfig.cmake")
        list(APPEND CMAKE_PREFIX_PATH "${_iree_local_dist}/lib/cmake/IREE")
        message(STATUS "  Using project-local IREE at: ${_iree_local_dist}")
    endif()

    # Try find_package first; if that fails, FetchContent a source build.
    # FIND_PACKAGE_ARGS isn't supported on some CMake variants so probe
    # explicitly: try the find_package, fall through to fetch on failure.
    find_package(IREECompiler CONFIG QUIET)
    find_package(IREERuntime  CONFIG QUIET)
    if(NOT IREECompiler_FOUND OR NOT IREERuntime_FOUND)
        message(STATUS "  IREE not found locally — falling back to FetchContent")
        message(WARNING
            "  Building IREE from source. First build is ~30-60 min and "
            "produces ~10 GB of intermediate artifacts (IREE bundles its own "
            "LLVM + StableHLO). Subsequent builds are incremental. To skip "
            "this, install IREE binaries to third_party/iree_dist/ or point "
            "TENZOR_IREE_DIR at an existing install. See docs/jit-mlir-setup.md.")
        include(FetchContent)
        FetchContent_Declare(iree_src
            GIT_REPOSITORY https://github.com/iree-org/iree.git
            # Pinned to the immutable commit of tag iree-3.11.0 (supply-chain
            # hardening: a retag upstream must not change the ~10 GB JIT build).
            GIT_TAG ca60cabdcd1615499706a6a2c951b41f541e969b
            GIT_SUBMODULES_RECURSE TRUE)
        set(IREE_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
        set(IREE_BUILD_SAMPLES         OFF CACHE BOOL "" FORCE)
        set(IREE_BUILD_TESTS           OFF CACHE BOOL "" FORCE)
        set(IREE_BUILD_DOCS            OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(iree_src)
        # Re-find_package after the source build installs configs into
        # ${iree_src_BINARY_DIR}/lib/cmake/IREE.
        list(APPEND CMAKE_PREFIX_PATH "${iree_src_BINARY_DIR}")
        find_package(IREECompiler REQUIRED CONFIG)
        find_package(IREERuntime  REQUIRED CONFIG)
    endif()

    # Verify iree-compile binary too — needed by tests that prefer subprocess for
    # debug-dumping AND by the GPU-target subprocess fallback in iree_paths.cpp
    # when the linked libIREECompiler lacks a target the user requested.
    #
    # Discovery priority: explicit $TENZOR_IREE_COMPILE → pip-installed full-GPU
    # venvs (have cuda/rocm out of the box) → CMake-found dist bin/ → $PATH. The
    # pip wheel is preferred because the lite iree-dist commonly ships without
    # cuda/rocm backends while the official `iree-base-compiler` wheel always
    # has them.
    find_program(IREE_COMPILE_EXECUTABLE iree-compile
        HINTS
            $ENV{TENZOR_IREE_COMPILE}
            ${PROJECT_SOURCE_DIR}/.venv-iree/bin
            ${IREECompiler_DIR}/../../../bin
        REQUIRED)
    message(STATUS "  iree-compile: ${IREE_COMPILE_EXECUTABLE}")

    # Probe iree-compile for its registered HAL target backends at configure
    # time. Tests that target cuda/rocm can read these to make a hard FAIL vs
    # SKIP decision: hardware present + binary missing target = fail loudly.
    execute_process(
        COMMAND "${IREE_COMPILE_EXECUTABLE}" --iree-hal-list-target-backends
        OUTPUT_VARIABLE _iree_hal_targets_raw
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "  iree-compile HAL targets:")
    string(REPLACE "\n" ";" _iree_hal_targets_lines "${_iree_hal_targets_raw}")
    set(IREE_COMPILE_HAL_TARGETS "")
    foreach(_line IN LISTS _iree_hal_targets_lines)
        string(STRIP "${_line}" _line_s)
        if(NOT _line_s OR _line_s MATCHES "^Registered" OR _line_s MATCHES "^#")
            continue()
        endif()
        list(APPEND IREE_COMPILE_HAL_TARGETS "${_line_s}")
        message(STATUS "    - ${_line_s}")
    endforeach()
    set(IREE_COMPILE_HAL_TARGETS "${IREE_COMPILE_HAL_TARGETS}"
        CACHE INTERNAL "HAL target backends supported by IREE_COMPILE_EXECUTABLE")

    # iree-run-module discovery follows the same priority. Cached so tests can
    # bake it into argv via -DTENZOR_IREE_RUN_MODULE_PATH=...
    find_program(IREE_RUN_MODULE_EXECUTABLE iree-run-module
        HINTS
            $ENV{TENZOR_IREE_RUN_MODULE}
            ${PROJECT_SOURCE_DIR}/.venv-iree/bin
            ${IREECompiler_DIR}/../../../bin
        REQUIRED)
    message(STATUS "  iree-run-module: ${IREE_RUN_MODULE_EXECUTABLE}")

    # ROCm runtime library discovery for the IREE HIP HAL driver.
    #
    # The IREE runtime dlopens libamdhip64.so + dependents (libhsa-runtime64,
    # libamd_comgr, librocm-core, librocprofiler-register, libroctracer64,
    # libhiprtc, libnuma) when an `hip` device is requested. Tenzor's Path C.2
    # (see docs/superpowers/plans/2026-05-19-tz-jit-mlir-phase1a.md) decouples
    # the JIT-test rocm gating from the Tenzor ROCm BACKEND being loaded — what
    # matters at runtime is that the IREE runtime can dlopen a working set of
    # ROCm libs.
    #
    # Discovery priority:
    #   1. $TENZOR_ROCM_RUNTIME_LIB_DIR  — explicit override
    #   2. ${TENZOR_ROCM_RUNTIME_LIB_DIR} — CMake-level override
    #   3. /opt/rocm/lib                 — host ROCm install
    #
    # The discovery probes one canonical .so (libamdhip64.so) and rejects
    # zero-byte files outright.
    set(_tenzor_rocm_runtime_lib_dir "")
    set(_rocm_candidate_dirs "")
    if(DEFINED ENV{TENZOR_ROCM_RUNTIME_LIB_DIR})
        list(APPEND _rocm_candidate_dirs "$ENV{TENZOR_ROCM_RUNTIME_LIB_DIR}")
    endif()
    if(TENZOR_ROCM_RUNTIME_LIB_DIR)
        list(APPEND _rocm_candidate_dirs "${TENZOR_ROCM_RUNTIME_LIB_DIR}")
    endif()
    list(APPEND _rocm_candidate_dirs "/opt/rocm/lib")
    foreach(_d IN LISTS _rocm_candidate_dirs)
        if(EXISTS "${_d}/libamdhip64.so")
            file(SIZE "${_d}/libamdhip64.so" _hip_so_size)
            # Follow one level of symlinks to detect truncated targets.
            get_filename_component(_real_hip_so "${_d}/libamdhip64.so" REALPATH)
            if(EXISTS "${_real_hip_so}")
                file(SIZE "${_real_hip_so}" _hip_so_size)
            endif()
            if(_hip_so_size GREATER 1024)
                set(_tenzor_rocm_runtime_lib_dir "${_d}")
                break()
            endif()
        endif()
    endforeach()
    if(_tenzor_rocm_runtime_lib_dir)
        message(STATUS "  ROCm runtime libs (for IREE HIP HAL): "
                "${_tenzor_rocm_runtime_lib_dir}")
        set(TENZOR_ROCM_RUNTIME_LIB_DIR "${_tenzor_rocm_runtime_lib_dir}"
            CACHE PATH "ROCm runtime libs for IREE HIP HAL dlopen." FORCE)
    else()
        message(STATUS "  ROCm runtime libs: none found "
                "(rocm IREE-target tests will skip via "
                "iree_can_initialize_default_device(\"hip\"))")
        set(TENZOR_ROCM_RUNTIME_LIB_DIR ""
            CACHE PATH "ROCm runtime libs for IREE HIP HAL dlopen." FORCE)
    endif()

    # IREE's iree-compile rocm target requires an explicit `--iree-rocm-target
    # =<chip>` flag (gfx1150/gfx942/...); there is no default. Mirror the
    # `CMAKE_HIP_ARCHITECTURES` value (auto-detected by the top-of-file ROCm
    # block via rocm_agent_enumerator) into a separate cache var so the
    # iree_compile.cpp subprocess invocation picks it up via the
    # TENZOR_DEFAULT_ROCM_TARGET compile-time define. If no chip is
    # detected, default to gfx1150 (the lee@archlinux box's Radeon 890M).
    set(_tenzor_default_rocm_target "")
    if(CMAKE_HIP_ARCHITECTURES)
        list(GET CMAKE_HIP_ARCHITECTURES 0 _tenzor_default_rocm_target)
    endif()
    if(NOT _tenzor_default_rocm_target)
        set(_tenzor_default_rocm_target "gfx1150")
    endif()
    set(TENZOR_DEFAULT_ROCM_TARGET "${_tenzor_default_rocm_target}"
        CACHE STRING "Default --iree-rocm-target chip for iree-compile." FORCE)
    message(STATUS "  Default IREE rocm target chip: ${TENZOR_DEFAULT_ROCM_TARGET}")

    # JIT-R043 (R2-01 follow-up): extends the TENZOR_DEFAULT_ROCM_TARGET block
    # above — IREE's iree-compile cuda target requires an explicit
    # `--iree-cuda-target=<smXX>` flag, just like rocm requires
    # `--iree-rocm-target=<gfxXXX>`. Only ROCm had a build-time default macro
    # wired up; CUDA fell all the way through to iree_compile.cpp's hardcoded
    # "sm_80" literal even though this repo already auto-detects the build
    # machine's real compute capability for the EAGER CUDA backend
    # (src/backends/cuda/CMakeLists.txt) — that detection just never got
    # threaded into a JIT-side default. Query nvidia-smi directly here
    # (independent of src/backends/cuda/CMakeLists.txt's
    # CMAKE_CUDA_ARCHITECTURES, which on CMake 3.24+ is the literal string
    # "native" — not a usable IREE target string) so this block works
    # regardless of CMake version or subdirectory processing order. Detection
    # failure (no nvidia-smi, no GPU present at build time, parse failure)
    # leaves TENZOR_DEFAULT_CUDA_TARGET undefined entirely — strictly
    # additive, never worse than the pre-existing "sm_80" runtime fallback in
    # iree_compile.cpp.
    #
    # R2-01: this block used to be wrapped in its own `if(TENZOR_BUILD_CUDA)`
    # AFTER this scope's endif(), so — unlike the ROCm block immediately
    # above, which the "Path C.2" design note explicitly decouples from
    # TENZOR_BUILD_ROCM — configuring -DTENZOR_BUILD_CUDA=OFF
    # -DTENZOR_USE_MLIR_JIT=ON (e.g. a ROCm-only or CPU-only CI leg that still
    # wants to sanity-check the CUDA IREE target string) left
    # TENZOR_DEFAULT_CUDA_TARGET undefined and iree_compile.cpp silently fell
    # back to a hardcoded sm_80 instead of querying the actual GPU, despite
    # the "JIT-R043 mirrors the ROCm block above" comment claiming parity.
    # Moved inside this if(TENZOR_USE_MLIR_JIT) scope and un-gated from
    # TENZOR_BUILD_CUDA to actually match the ROCm block's decoupling —
    # nvidia-smi probing already works independently of whether the Tenzor
    # CUDA backend itself is being built.
    set(_tenzor_default_cuda_target "")
    find_program(_TENZOR_NVIDIA_SMI nvidia-smi)
    if(_TENZOR_NVIDIA_SMI)
        execute_process(
            COMMAND ${_TENZOR_NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader,nounits
            OUTPUT_VARIABLE _TENZOR_CUDA_COMPUTE_CAPS
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _TENZOR_NVIDIA_SMI_RESULT
        )
        if(_TENZOR_NVIDIA_SMI_RESULT EQUAL 0 AND _TENZOR_CUDA_COMPUTE_CAPS)
            # Take the FIRST reported GPU's compute_cap (e.g. "8.6"); IREE
            # wants "sm_86" (no dot, sm_ prefix) -- matches iree_compile.cpp's
            # existing "sm_80"-style fallback literal format.
            string(REPLACE "\n" ";" _TENZOR_CUDA_CAP_LIST "${_TENZOR_CUDA_COMPUTE_CAPS}")
            list(GET _TENZOR_CUDA_CAP_LIST 0 _TENZOR_CUDA_FIRST_CAP)
            string(STRIP "${_TENZOR_CUDA_FIRST_CAP}" _TENZOR_CUDA_FIRST_CAP)
            string(REPLACE "." "" _TENZOR_CUDA_ARCH_NUM "${_TENZOR_CUDA_FIRST_CAP}")
            if(_TENZOR_CUDA_ARCH_NUM MATCHES "^[0-9]+$")
                set(_tenzor_default_cuda_target "sm_${_TENZOR_CUDA_ARCH_NUM}")
            endif()
        endif()
    endif()
    if(_tenzor_default_cuda_target)
        set(TENZOR_DEFAULT_CUDA_TARGET "${_tenzor_default_cuda_target}"
            CACHE STRING "Default --iree-cuda-target chip for iree-compile." FORCE)
        message(STATUS "  Default IREE cuda target chip: ${TENZOR_DEFAULT_CUDA_TARGET}")
    else()
        message(STATUS "  Default IREE cuda target chip: none detected "
                "(iree_compile.cpp falls back to its hardcoded sm_80 default)")
    endif()
endif()

# Link-Time Optimization
option(TENZOR_ENABLE_LTO "Enable Link-Time Optimization" OFF)
if(TENZOR_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT lto_supported OUTPUT lto_error)
    if(lto_supported)
        message(STATUS "LTO enabled")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "LTO requested but not supported: ${lto_error}")
    endif()
endif()

# Coverage build type
option(TENZOR_ENABLE_COVERAGE "Enable coverage reporting" OFF)

# Sanitizer options
option(TENZOR_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(TENZOR_ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(TENZOR_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

if(TENZOR_ENABLE_ASAN AND TENZOR_ENABLE_TSAN)
    message(FATAL_ERROR "AddressSanitizer and ThreadSanitizer are mutually exclusive")
endif()

# Static analysis
option(TENZOR_ENABLE_CLANG_TIDY "Enable clang-tidy static analysis" OFF)

# Export compile_commands.json for IDE integration and clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Tenzor-specific compiler flags (stored in variables, applied per-target to avoid
# polluting consumers via add_subdirectory or FetchContent)
option(TENZOR_WERROR "Treat warnings as errors (for CI)" OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(TENZOR_WARNING_FLAGS -Wall -Wextra -pedantic)
    if(TENZOR_WERROR)
        list(APPEND TENZOR_WARNING_FLAGS -Werror)
    endif()

    # Use -march=native for maximum performance on the build machine.
    # Disabled by default for portable binaries. Enable for local dev builds.
    option(TENZOR_NATIVE_ARCH "Use -march=native for host-specific optimizations" OFF)
    if(TENZOR_NATIVE_ARCH)
        add_compile_options(-march=native)
    endif()

    # Enable coverage if requested (must be global — applies to all compiled code)
    if(TENZOR_ENABLE_COVERAGE)
        message(STATUS "Coverage reporting enabled")
        add_compile_options(--coverage -fprofile-arcs -ftest-coverage)
        add_link_options(--coverage)
    endif()

    # Sanitizers (must be global — applies to all compiled code)
    if(TENZOR_ENABLE_ASAN)
        message(STATUS "AddressSanitizer enabled")
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address)
    endif()

    if(TENZOR_ENABLE_TSAN)
        message(STATUS "ThreadSanitizer enabled")
        add_compile_options(-fsanitize=thread)
        add_link_options(-fsanitize=thread)
    endif()

    if(TENZOR_ENABLE_UBSAN)
        message(STATUS "UndefinedBehaviorSanitizer enabled")
        add_compile_options(-fsanitize=undefined -fno-omit-frame-pointer)
        add_link_options(-fsanitize=undefined)
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(TENZOR_WARNING_FLAGS /W4 /arch:AVX2)
    if(TENZOR_WERROR)
        list(APPEND TENZOR_WARNING_FLAGS /WX)
    endif()
endif()

# clang-tidy integration
if(TENZOR_ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
    if(CLANG_TIDY_EXE)
        message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
    else()
        message(WARNING "clang-tidy requested but not found")
    endif()
endif()

# Output directories and RPATH defaults.
#
# These are imposed ONLY for a standalone (top-level) Tenzor build. When Tenzor
# is consumed via add_subdirectory / FetchContent, the parent project owns
# artifact placement and RPATH policy — forcing them here would fight the
# consumer and, worse, pre-fix the source-tree output dir wrote compiled
# artifacts back into ${PROJECT_SOURCE_DIR} (read-only under FetchContent's
# _deps). A consumer that wants Tenzor's binaries collected in one place sets
# CMAKE_{RUNTIME,LIBRARY,ARCHIVE}_OUTPUT_DIRECTORY itself.
if(PROJECT_IS_TOP_LEVEL)
    # All binaries, backend .so and archives land in <build>/bin
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

    # For multi-configuration generators (Visual Studio, Xcode)
    foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${CONFIG} CONFIG_UPPER)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${PROJECT_BINARY_DIR}/bin)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${PROJECT_BINARY_DIR}/bin)
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${PROJECT_BINARY_DIR}/bin)
    endforeach()

    # RPATH so executables find sibling shared libraries. $ORIGIN resolves to
    # <build>/bin for any in-tree binary (all outputs land there above), so a
    # single relative entry suffices and avoids leaking an absolute build path
    # into every binary's RPATH. Per-target INSTALL_RPATH is set in the install()
    # section below; these only govern the standalone build/install of Tenzor.
    set(CMAKE_SKIP_BUILD_RPATH FALSE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    set(CMAKE_BUILD_RPATH "$ORIGIN")
endif()

# Find required packages
find_package(Threads REQUIRED)
find_package(OpenMP)

# D.1: spdlog for structured logging across distributed / serving / io.
# Prefer the system package; FetchContent as fallback so offline builds
# still work if spdlog isn't installed (FetchContent is also already used
# elsewhere in this build for cudnn_frontend).
find_package(spdlog QUIET CONFIG)
if(NOT spdlog_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        # Pinned to the immutable commit of tag v1.14.1 (a retag upstream must
        # not silently change build inputs — supply-chain hardening).
        GIT_TAG        27cb4c76708608465c413f6d0e6b8d99a4d84302
    )
    set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
    set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(spdlog)
endif()

# D.1: fmt — the PUBLIC logging header (utils/logging.hpp) inlines
# ::fmt::format, so any consumer using the TENZOR_LOG* macros instantiates fmt
# symbols. With a system (compiled) fmt that requires an explicit link, fmt is
# therefore part of Tenzor's public interface and must be propagated (spdlog
# itself stays PRIVATE behind utils/log.cpp). System spdlog's CONFIG pulls fmt
# in transitively; ensure the target explicitly. When spdlog is FetchContent'd
# with bundled header-only fmt, no separate fmt::fmt target exists and none is
# needed (the inline format is fully header-only) — the link below is guarded.
if(NOT TARGET fmt::fmt)
    find_package(fmt QUIET CONFIG)
endif()
if(TARGET fmt::fmt)
    set(TENZOR_FMT_PUBLIC TRUE)
else()
    set(TENZOR_FMT_PUBLIC FALSE)
endif()

# Include CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# G.12: vendored nlohmann/json (single header, header-only, no link deps).
# Exposed as an INTERFACE library so the serving module (and any future
# consumer such as model hub / config IO) can `#include <nlohmann/json.hpp>`.
add_library(tenzor_nlohmann_json INTERFACE)
target_include_directories(tenzor_nlohmann_json INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party>
)

# Add subdirectories
add_subdirectory(src)

# Python bindings
if(TENZOR_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development)
    find_package(pybind11 CONFIG)

    if(pybind11_FOUND AND Python_FOUND)
        pybind11_add_module(tenzor_python
            python/bindings.cpp
            python/numpy_interop.cpp
            # Incremental split of the ~10k-line bindings.cpp monolith
            # (P3.4). New submodule TUs live under python/bindings/.
            python/bindings/bindings_core.cpp
            python/bindings/bindings_linalg.cpp
            python/bindings/bindings_fft.cpp
            python/bindings/bindings_vision_detection.cpp
            python/bindings/bindings_distributed.cpp
            python/bindings/bindings_jit.cpp
            python/bindings/bindings_quantization.cpp
            python/bindings/bindings_optim.cpp
            python/bindings/bindings_autograd.cpp
            python/bindings/bindings_nn.cpp
            python/bindings/bindings_nested.cpp
            python/bindings/bindings_lite.cpp
        )

        target_link_libraries(tenzor_python PRIVATE tenzor_core tenzor_backend_cpu)

        # Optional PyTorch interop. Compiled into the module only when
        # TENZOR_BUILD_TORCH_INTEROP=ON and libtorch is discoverable. Keeps
        # libtorch out of the default CPU wheel; bindings.cpp gates the
        # `torch_interop` submodule on TENZOR_HAS_TORCH.
        if(TENZOR_BUILD_TORCH_INTEROP)
            find_package(Torch REQUIRED)
            target_sources(tenzor_python PRIVATE python/torch_interop.cpp)
            target_link_libraries(tenzor_python PRIVATE "${TORCH_LIBRARIES}")
            target_include_directories(tenzor_python PRIVATE ${TORCH_INCLUDE_DIRS})
            target_compile_definitions(tenzor_python PRIVATE TENZOR_HAS_TORCH)
            message(STATUS "  PyTorch interop:      ON (libtorch found)")
        endif()

        if(TENZOR_USE_MLIR_JIT)
            # bindings/bindings_jit.cpp uses TENZOR_HAS_MLIR_JIT-guarded code
            # (cache_stats / reset_cache_stats) so propagate the define.
            target_compile_definitions(tenzor_python PRIVATE TENZOR_HAS_MLIR_JIT)
            if(TARGET tenzor_jit_mlir)
                target_link_libraries(tenzor_python PRIVATE tenzor_jit_mlir)
            endif()
        endif()
        if(TARGET tenzor_backend_cuda)
            target_link_libraries(tenzor_python PRIVATE tenzor_backend_cuda)
        endif()
        if(TARGET tenzor_backend_rocm)
            target_link_libraries(tenzor_python PRIVATE tenzor_backend_rocm)
        endif()
        if(TARGET tenzor_backend_oneapi)
            target_link_libraries(tenzor_python PRIVATE tenzor_backend_oneapi)
        endif()
        if(TARGET tenzor_backend_vulkan)
            target_link_libraries(tenzor_python PRIVATE tenzor_backend_vulkan)
        endif()
        target_compile_options(tenzor_python PRIVATE ${TENZOR_WARNING_FLAGS})


        set_target_properties(tenzor_python PROPERTIES
            OUTPUT_NAME "tenzor_core"
            LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/python/tenzor"
        )

        # Copy Python package files incrementally: only re-copy when source
        # .py/.pyi files change (uses add_custom_command with DEPENDS for
        # proper dependency tracking instead of always-stale custom target).
        set(_PY_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/__init__.py
            ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/nn.py
            ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/functional.py
            ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/autograd.py
            ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/data.py
            ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/overrides.py
            ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/jit.py
            # Audit-8 II.12: typed-exception re-export namespace.
            ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/exceptions.py
            # audit-10 OO.12: composable function transforms (torch.func parity).
            ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/func.py
        )
        # Subpackages (directories with their own __init__.py) — copied via
        # CMake's directory copy so that all files inside them land in the
        # build tree without each having to be enumerated here.
        file(GLOB _PY_SPECIAL_SOURCES CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/special/*.py)
        file(GLOB _PY_DISTRIBUTIONS_SOURCES CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/distributions/*.py)
        file(GLOB _PY_STUBS CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/*.pyi)
        list(APPEND _PY_STUBS
            ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/py.typed)
        set(_PY_ALL_INPUTS ${_PY_SOURCES} ${_PY_STUBS})

        # Build list of expected outputs in the build directory
        set(_PY_OUTPUTS "")
        foreach(_src ${_PY_ALL_INPUTS})
            get_filename_component(_fname ${_src} NAME)
            list(APPEND _PY_OUTPUTS ${PROJECT_BINARY_DIR}/python/tenzor/${_fname})
        endforeach()
        foreach(_src ${_PY_SPECIAL_SOURCES})
            get_filename_component(_fname ${_src} NAME)
            list(APPEND _PY_OUTPUTS
                ${PROJECT_BINARY_DIR}/python/tenzor/special/${_fname})
        endforeach()
        foreach(_src ${_PY_DISTRIBUTIONS_SOURCES})
            get_filename_component(_fname ${_src} NAME)
            list(APPEND _PY_OUTPUTS
                ${PROJECT_BINARY_DIR}/python/tenzor/distributions/${_fname})
        endforeach()

        add_custom_command(
            OUTPUT ${_PY_OUTPUTS}
            COMMAND ${CMAKE_COMMAND} -E make_directory
                ${PROJECT_BINARY_DIR}/python/tenzor/special
            COMMAND ${CMAKE_COMMAND} -E make_directory
                ${PROJECT_BINARY_DIR}/python/tenzor/distributions
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_PY_SOURCES}
                ${PROJECT_BINARY_DIR}/python/tenzor/
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_PY_STUBS}
                ${PROJECT_BINARY_DIR}/python/tenzor/
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${_PY_SPECIAL_SOURCES}
                ${PROJECT_BINARY_DIR}/python/tenzor/special/
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${_PY_DISTRIBUTIONS_SOURCES}
                ${PROJECT_BINARY_DIR}/python/tenzor/distributions/
            DEPENDS ${_PY_ALL_INPUTS}
                ${_PY_SPECIAL_SOURCES} ${_PY_DISTRIBUTIONS_SOURCES}
            COMMENT "Copying Python files and type stubs to build directory"
        )
        add_custom_target(copy_python_files DEPENDS ${_PY_OUTPUTS})
        add_dependencies(tenzor_python copy_python_files)
    else()
        message(WARNING "pybind11 or Python not found, Python bindings will not be built")
    endif()
endif()

# Tests
if(TENZOR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)

    # Phase 10 of the test-coverage campaign — TESTING.md contract lint.
    # Runs `tools/lint_test_contract.py` and uses the baseline file to ratchet
    # down pre-existing violations. Invoke via:
    #     ctest -L lint
    find_package(Python COMPONENTS Interpreter QUIET)
    if(Python_FOUND)
        add_test(
            NAME lint_test_contract
            COMMAND ${Python_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tools/lint_test_contract.py
                --tests-dir ${PROJECT_SOURCE_DIR}/tests
                --baseline ${PROJECT_SOURCE_DIR}/tools/lint_test_contract_baseline.txt
        )
        set_tests_properties(lint_test_contract PROPERTIES LABELS "lint")

        # audit-3 T.4: wire the four standalone audit scripts into ctest so
        # they execute under `ctest -L lint`. Each script exits non-zero on
        # drift; AuditMarkersScan reports lingering TODO/FIXME/"for now" etc.
        add_test(
            NAME AuditMarkersScan
            COMMAND ${Python_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tools/scan_audit_markers.py
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        )
        set_tests_properties(AuditMarkersScan PROPERTIES LABELS "lint")

        add_test(
            NAME PyiDrift
            COMMAND ${Python_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tools/check_pyi_drift.py
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        )
        set_tests_properties(PyiDrift PROPERTIES LABELS "lint")

        add_test(
            NAME ParamGroupSerialization
            COMMAND ${Python_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tools/check_param_group_serialization.py
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        )
        set_tests_properties(ParamGroupSerialization PROPERTIES LABELS "lint")

        add_test(
            NAME TensorboardWireFormat
            COMMAND ${Python_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tools/check_tensorboard_wire_format.py
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        )
        set_tests_properties(TensorboardWireFormat PROPERTIES LABELS "lint")

        # audit-3 T.14: ratchet for multi-dtype companion coverage. Errors on
        # any new test_*.cpp that lacks a sibling test_*_multidtype.cpp and is
        # not on the known-intentional allowlist in
        # tests/backend_parity/MULTIDTYPE_COVERAGE_GAPS.md.
        add_test(
            NAME MultidtypeCoverage
            COMMAND ${Python_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tools/check_multidtype_coverage.py
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        )
        set_tests_properties(MultidtypeCoverage PROPERTIES LABELS "lint")

        # Guard against re-introducing CMAKE_SOURCE_DIR/CMAKE_BINARY_DIR in
        # Tenzor's own cmake files — those break add_subdirectory/FetchContent
        # consumption (they resolve to the consumer's tree). Must use PROJECT_*.
        add_test(
            NAME CMakeProjectDirLint
            COMMAND ${Python_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tools/check_cmake_project_dir.py
                --root ${PROJECT_SOURCE_DIR}
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        )
        set_tests_properties(CMakeProjectDirLint PROPERTIES LABELS "lint")
    endif()
endif()

# Benchmarks
if(TENZOR_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Fuzzing targets
if(TENZOR_FUZZ)
    add_subdirectory(fuzz)
endif()

# Examples
if(TENZOR_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tools
option(TENZOR_BUILD_TOOLS "Build developer tools" ON)
if(TENZOR_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

# Documentation
find_package(Doxygen)
if(DOXYGEN_FOUND)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
    message(STATUS "  Doxygen:              Found (run 'make docs' to generate)")
else()
    message(STATUS "  Doxygen:              Not found (documentation will not be available)")
endif()

# Installation
#
# Two install layouts:
#  - SKBUILD (wheel build via scikit-build-core / pip): everything lands inside
#    the wheel's tenzor/ package directory. All shared libraries go to
#    tenzor/lib/ so the backend discovery in src/core/init.cpp (dladdr on
#    libtenzor_core.so, then look for tenzor_backend_*.so next to it) works
#    inside a pip-installed wheel without TENZOR_BACKEND_DIR.
#  - Standard (cmake --install): FHS layout for C++ consumers
#    (find_package(Tenzor) / pkg-config).
if(SKBUILD)
    set_target_properties(tenzor_core PROPERTIES INSTALL_RPATH "$ORIGIN")
    install(TARGETS tenzor_core LIBRARY DESTINATION lib COMPONENT python)

    foreach(backend cpu cuda rocm oneapi vulkan mps)
        if(TARGET tenzor_backend_${backend})
            get_target_property(_imported tenzor_backend_${backend} IMPORTED)
            if(NOT _imported)
                set_target_properties(tenzor_backend_${backend}
                    PROPERTIES INSTALL_RPATH "$ORIGIN")
                install(TARGETS tenzor_backend_${backend}
                    LIBRARY DESTINATION lib COMPONENT python)
            else()
                # Prebuilt (imported) backend, e.g. the externally-compiled
                # oneAPI backend: ship the binary as-is when present.
                get_target_property(_loc tenzor_backend_${backend} IMPORTED_LOCATION)
                if(_loc AND EXISTS "${_loc}")
                    install(FILES "${_loc}" DESTINATION lib COMPONENT python)
                endif()
            endif()
        endif()
    endforeach()

    if(TARGET tenzor_python)
        # Extension module sits at the package root (site-packages/tenzor/);
        # $ORIGIN/lib resolves libtenzor_core.so and the directly-linked
        # backend libraries at import time.
        set_target_properties(tenzor_python PROPERTIES INSTALL_RPATH "$ORIGIN/lib")
        install(TARGETS tenzor_python LIBRARY DESTINATION . COMPONENT python)
    endif()
else()
# Relocatable install: $ORIGIN-relative RPATHs instead of the configure-time
# absolute prefix (CMAKE_INSTALL_RPATH_USE_LINK_PATH still appends external
# link dirs such as oneAPI/MKL).
set_target_properties(tenzor_core PROPERTIES INSTALL_RPATH "$ORIGIN")
install(TARGETS tenzor_core
    EXPORT TenzorTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include/tenzor
    INCLUDES DESTINATION include
)

install(DIRECTORY include/tenzor DESTINATION include)

# Install backend shared libraries (only non-imported targets that were built).
# Fix 2: include backends in TenzorTargets so downstream find_package(Tenzor)
# resolves Tenzor::tenzor_backend_cpu etc. via the exported config.
foreach(backend cpu cuda rocm oneapi vulkan mps)
    if(TARGET tenzor_backend_${backend})
        get_target_property(_imported tenzor_backend_${backend} IMPORTED)
        if(NOT _imported)
            # lib/tenzor/backends -> lib (libtenzor_core.so)
            set_target_properties(tenzor_backend_${backend}
                PROPERTIES INSTALL_RPATH "$ORIGIN/../..")
            install(TARGETS tenzor_backend_${backend}
                EXPORT TenzorTargets
                LIBRARY DESTINATION lib/tenzor/backends
                ARCHIVE DESTINATION lib/tenzor/backends
                RUNTIME DESTINATION bin)
        endif()
    endif()
endforeach()

# Export targets
install(EXPORT TenzorTargets
    FILE TenzorTargets.cmake
    NAMESPACE Tenzor::
    DESTINATION lib/cmake/Tenzor
)

# Package configuration
include(CMakePackageConfigHelpers)

# Report backend availability based on whether the backend was ACTUALLY built
# (its target exists), not merely requested via TENZOR_BUILD_*. A backend can be
# auto-disabled at configure time (e.g. CUDA when no compiler is found), in which
# case the request flag is ON but no library was produced.
foreach(_b CUDA ROCM ONEAPI VULKAN MPS)
    string(TOLOWER ${_b} _bl)
    if(TARGET tenzor_backend_${_bl})
        set(TENZOR_${_b}_BUILT ON)
    else()
        set(TENZOR_${_b}_BUILT OFF)
    endif()
endforeach()

# Reflect optional-dependency availability into the installed config so it
# only find_dependency()s what was actually built against.
if(OpenMP_CXX_FOUND)
    set(TENZOR_OPENMP_FOUND TRUE)
else()
    set(TENZOR_OPENMP_FOUND FALSE)
endif()

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TenzorConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/TenzorConfig.cmake"
    INSTALL_DESTINATION lib/cmake/Tenzor
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/TenzorConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/TenzorConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/TenzorConfigVersion.cmake"
    DESTINATION lib/cmake/Tenzor
)

# pkg-config support
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tenzor.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/tenzor.pc"
    @ONLY
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/tenzor.pc"
    DESTINATION lib/pkgconfig
)

# Python module installation (manual cmake --install).
# Destinations must stay relative: absolute ${CMAKE_INSTALL_PREFIX} paths break
# staged installs (DESTDIR) and wheel staging.
if(TENZOR_BUILD_PYTHON AND TARGET tenzor_python)
    # Install the compiled extension module.
    # lib/python/tenzor -> lib (libtenzor_core.so) and lib/tenzor/backends
    # (directly-linked tenzor_backend_*.so).
    set_target_properties(tenzor_python PROPERTIES
        INSTALL_RPATH "$ORIGIN/../..;$ORIGIN/../../tenzor/backends")
    install(TARGETS tenzor_python
        LIBRARY DESTINATION lib/python/tenzor
        COMPONENT python
    )
    # Install Python source files, type stubs and the PEP 561 marker
    install(DIRECTORY python/tenzor/
        DESTINATION lib/python/tenzor
        COMPONENT python
        FILES_MATCHING
            PATTERN "*.py"
            PATTERN "*.pyi"
            PATTERN "py.typed"
    )
endif()
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "Tenzor Configuration Summary")
message(STATUS "============================")
message(STATUS "  Version:              ${PROJECT_VERSION}")
message(STATUS "  Build type:           ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ compiler:         ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "")
message(STATUS "Build Options:")
message(STATUS "  CUDA backend:         ${TENZOR_BUILD_CUDA}")
message(STATUS "  ROCm backend:         ${TENZOR_BUILD_ROCM}")
message(STATUS "  OneAPI backend:       ${TENZOR_BUILD_ONEAPI}")
message(STATUS "  Vulkan backend:       ${TENZOR_BUILD_VULKAN}")
message(STATUS "  Python bindings:      ${TENZOR_BUILD_PYTHON}")
message(STATUS "  Tests:                ${TENZOR_BUILD_TESTS}")
message(STATUS "  Benchmarks:           ${TENZOR_BUILD_BENCHMARKS}")
message(STATUS "  Examples:             ${TENZOR_BUILD_EXAMPLES}")
message(STATUS "")
if(OpenMP_CXX_FOUND)
    message(STATUS "  OpenMP:               Enabled")
else()
    message(STATUS "  OpenMP:               Disabled")
endif()
if(CCACHE_PROGRAM)
    message(STATUS "  ccache:               ${CCACHE_PROGRAM}")
else()
    message(STATUS "  ccache:               Not found")
endif()
if(TENZOR_ENABLE_LTO AND lto_supported)
    message(STATUS "  LTO:                  Enabled")
else()
    message(STATUS "  LTO:                  Disabled")
endif()
message(STATUS "")
