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
    find_program(_HIPCC_EXE hipcc PATHS /opt/rocm/bin NO_DEFAULT_PATH)
    if(_HIPCC_EXE)
        get_filename_component(_ROCM_ROOT "${_HIPCC_EXE}" DIRECTORY)
        get_filename_component(_ROCM_ROOT "${_ROCM_ROOT}" DIRECTORY)
    else()
        set(_ROCM_ROOT "/opt/rocm")
    endif()
endif()
if(EXISTS "${_ROCM_ROOT}/bin/hipcc")
    set(_ROCM_AVAILABLE TRUE)
endif()


# Set TENZOR_BUILD_ROCM before project() - use cache to preserve user preference
# Default to ON only if ROCm is available on the system
if(NOT DEFINED TENZOR_BUILD_ROCM)
    option(TENZOR_BUILD_ROCM "Build ROCm backend" ON)
endif()

# Set up HIP architectures if building ROCm backend
if(TENZOR_BUILD_ROCM AND _ROCM_AVAILABLE)
    if(NOT CMAKE_HIP_ARCHITECTURES)
        # Try to auto-detect GPU architecture
        execute_process(
            COMMAND ${_ROCM_ROOT}/bin/rocm_agent_enumerator
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
option(TENZOR_BUILD_BENCHMARKS "Build benchmarks" OFF)
# Expand the default multi-dtype parity matrix to include BFloat16. Off by
# default because it doubles the existing FLOAT_DTYPES (F32/F64/F16) test
# matrix — the full suite already takes 15+ hours and a 4-dtype default
# would be disproportionate. Tests that need BF16 unconditionally use
# ALL_FLOAT_DTYPES regardless of this flag.
option(TENZOR_TEST_BFLOAT16 "Include BFloat16 in the default multi-dtype parity matrix" OFF)
if(TENZOR_TEST_BFLOAT16)
    add_compile_definitions(TENZOR_TEST_BFLOAT16)
endif()
option(TENZOR_FUZZ "Build libFuzzer fuzz targets (requires Clang)" OFF)
option(TENZOR_BUILD_EXAMPLES "Build examples" ON)
option(TENZOR_BUILD_MODEL_HUB "Build model hub (requires CURL and OpenSSL)" ON)
option(TENZOR_BUILD_SERVING "Build inference serving (requires gRPC for full support)" OFF)
option(TENZOR_BUILD_LITE "Build lightweight mobile/embedded inference runtime" OFF)
option(TENZOR_REQUIRE_MKL "Require Intel MKL (FATAL_ERROR if not found)" OFF)
option(TENZOR_ENABLE_PROFILING "Enable NVTX/ROCTX profiling annotations" OFF)

# MLIR/StableHLO/IREE JIT pipeline (Phase 13)
option(TENZOR_USE_MLIR_JIT "Enable MLIR/StableHLO JIT (Phase 13)" OFF)
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
    list(APPEND CMAKE_PREFIX_PATH $ENV{IREE_DIR})
    find_package(IREECompiler REQUIRED CONFIG)
    find_package(IREERuntime REQUIRED CONFIG)

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
            /home/lee/.venvs/tenzor-jit/bin
            /home/lee/venv-iree/bin
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
            /home/lee/.venvs/tenzor-jit/bin
            /home/lee/venv-iree/bin
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

# Set output directories - all binaries and libraries in <project>/bin
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# For multi-configuration generators (Visual Studio, Xcode)
foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${CONFIG} CONFIG_UPPER)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_SOURCE_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_SOURCE_DIR}/bin)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_SOURCE_DIR}/bin)
endforeach()

# Set RPATH so executables can find shared libraries in same directory
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Use $ORIGIN to find libraries in same directory as executable
set(CMAKE_BUILD_RPATH "$ORIGIN:${CMAKE_SOURCE_DIR}/bin")

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
        GIT_TAG        v1.14.1
    )
    set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
    set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(spdlog)
endif()

# Include CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

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
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/tenzor"
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
        )
        # Subpackages (directories with their own __init__.py) — copied via
        # CMake's directory copy so that all files inside them land in the
        # build tree without each having to be enumerated here.
        file(GLOB _PY_SPECIAL_SOURCES CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/special/*.py)
        file(GLOB _PY_DISTRIBUTIONS_SOURCES CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/distributions/*.py)
        set(_PY_STUBS
            ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/py.typed
            ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/__init__.pyi
            ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/nn.pyi
            ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/functional.pyi
            ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/optim.pyi
            ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/torch_interop.pyi
        )
        set(_PY_ALL_INPUTS ${_PY_SOURCES} ${_PY_STUBS})

        # Build list of expected outputs in the build directory
        set(_PY_OUTPUTS "")
        foreach(_src ${_PY_ALL_INPUTS})
            get_filename_component(_fname ${_src} NAME)
            list(APPEND _PY_OUTPUTS ${CMAKE_BINARY_DIR}/python/tenzor/${_fname})
        endforeach()
        foreach(_src ${_PY_SPECIAL_SOURCES})
            get_filename_component(_fname ${_src} NAME)
            list(APPEND _PY_OUTPUTS
                ${CMAKE_BINARY_DIR}/python/tenzor/special/${_fname})
        endforeach()
        foreach(_src ${_PY_DISTRIBUTIONS_SOURCES})
            get_filename_component(_fname ${_src} NAME)
            list(APPEND _PY_OUTPUTS
                ${CMAKE_BINARY_DIR}/python/tenzor/distributions/${_fname})
        endforeach()

        add_custom_command(
            OUTPUT ${_PY_OUTPUTS}
            COMMAND ${CMAKE_COMMAND} -E make_directory
                ${CMAKE_BINARY_DIR}/python/tenzor/special
            COMMAND ${CMAKE_COMMAND} -E make_directory
                ${CMAKE_BINARY_DIR}/python/tenzor/distributions
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_PY_SOURCES}
                ${CMAKE_BINARY_DIR}/python/tenzor/
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_PY_STUBS}
                ${CMAKE_BINARY_DIR}/python/tenzor/
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${_PY_SPECIAL_SOURCES}
                ${CMAKE_BINARY_DIR}/python/tenzor/special/
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${_PY_DISTRIBUTIONS_SOURCES}
                ${CMAKE_BINARY_DIR}/python/tenzor/distributions/
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
                ${CMAKE_SOURCE_DIR}/tools/lint_test_contract.py
                --tests-dir ${CMAKE_SOURCE_DIR}/tests
                --baseline ${CMAKE_SOURCE_DIR}/tools/lint_test_contract_baseline.txt
        )
        set_tests_properties(lint_test_contract PROPERTIES LABELS "lint")
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
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
    message(STATUS "  Doxygen:              Found (run 'make docs' to generate)")
else()
    message(STATUS "  Doxygen:              Not found (documentation will not be available)")
endif()

# Installation
install(TARGETS tenzor_core
    EXPORT TenzorTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include/tenzor
    INCLUDES DESTINATION include
)

install(DIRECTORY include/tenzor DESTINATION include)

# Install backend shared libraries (only non-imported targets that were built)
foreach(backend cpu cuda rocm oneapi vulkan)
    if(TARGET tenzor_backend_${backend})
        get_target_property(_imported tenzor_backend_${backend} IMPORTED)
        if(NOT _imported)
            install(TARGETS tenzor_backend_${backend}
                LIBRARY DESTINATION lib/tenzor/backends)
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

# Python module installation
if(TENZOR_BUILD_PYTHON AND TARGET tenzor_python)
    # Install the compiled extension module
    install(TARGETS tenzor_python
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python/tenzor
        COMPONENT python
    )
    # Install Python source files
    install(DIRECTORY python/tenzor/
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python/tenzor
        COMPONENT python
        FILES_MATCHING
            PATTERN "*.py"
            PATTERN "*.pyi"
    )
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
