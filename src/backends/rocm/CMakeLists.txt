# ROCm Backend
# Requires ROCm 5.0 or higher for AMD GPUs
# Mirrors CUDA backend structure for consistency
#
# Note: This backend invokes hipcc directly via add_custom_command() instead
# of using CMake's native HIP language support (set_source_files_properties(
# ... LANGUAGE HIP) + add_library()). On Windows, CMake's Windows-Clang.cmake
# platform module hard-blocks mixing MSVC (cl.exe) and any Clang-family
# compiler (hipcc's clang frontend included) as co-enabled languages within
# one project() -- an unconditional CMake policy with no flag-level
# workaround. HIP is therefore never added to the top-level project()'s
# LANGUAGES on Windows (see top-level CMakeLists.txt); this file treats
# hipcc.exe purely as an external tool, exactly like the OneAPI backend
# treats icpx for SYCL (src/backends/oneapi/CMakeLists.txt is the reference
# pattern this file mirrors).

# _ROCM_ROOT / _ROCM_AVAILABLE are computed at the top of the top-level
# CMakeLists.txt (before project()) and already handle both Linux (/opt/rocm)
# and the Windows HIP SDK installer's HIP_PATH env var. src/backends/CMakeLists.txt
# only add_subdirectory()s here when _ROCM_AVAILABLE is already TRUE.

# hipcc.exe mishandles quoting when it re-invokes clang.exe internally for
# any path containing spaces (reproduced directly: "C:/Program Files/AMD/
# ROCm/7.2/include" arrives at clang.exe split into two unquoted argv
# tokens). _ROCM_ROOT itself is already normalised to its 8.3 short path on
# Windows for this reason (see top-level CMakeLists.txt); this helper covers
# OTHER include dirs pulled from ROCm CONFIG packages below (hipcub/rocprim/
# rocthrust), whose *-config.cmake files may re-derive/canonicalise their own
# install prefix back to the long form regardless of how they were found.
function(tenzor_win_short_path path_var)
    if(WIN32)
        execute_process(
            COMMAND cmd /c for %A in ("${${path_var}}") do @echo %~sA
            OUTPUT_VARIABLE _short
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(_short)
            file(TO_CMAKE_PATH "${_short}" _short)
            set(${path_var} "${_short}" PARENT_SCOPE)
        endif()
    endif()
endfunction()

# Ensure HIP_PLATFORM is set — some ROCm packages disable hipconfig,
# preventing hip-config.cmake from auto-detecting the platform.
if(NOT DEFINED HIP_PLATFORM AND NOT DEFINED ENV{HIP_PLATFORM})
    set(HIP_PLATFORM "amd" CACHE STRING "HIP platform (amd or nvidia)")
endif()

# Find ROCm/HIP packages
find_package(hip QUIET CONFIG PATHS ${_ROCM_ROOT})
find_package(rocblas QUIET CONFIG PATHS ${_ROCM_ROOT})
find_package(rocsolver QUIET CONFIG PATHS ${_ROCM_ROOT})
find_package(hiprand QUIET CONFIG PATHS ${_ROCM_ROOT})
find_package(rocfft QUIET CONFIG PATHS ${_ROCM_ROOT})
find_package(rocsparse QUIET CONFIG PATHS ${_ROCM_ROOT})
find_package(hipcub QUIET CONFIG PATHS ${_ROCM_ROOT})
find_package(rocprim QUIET CONFIG PATHS ${_ROCM_ROOT})
find_package(rocthrust QUIET CONFIG PATHS ${_ROCM_ROOT})

if(NOT hip_FOUND)
    message(WARNING "ROCm/HIP not found. ROCm backend will not be built.")
    message(WARNING "Install ROCm from: https://rocm.docs.amd.com/")
    message(WARNING "Make sure ROCm is installed in /opt/rocm (Linux) or the Windows HIP SDK is installed and HIP_PATH is set")
    set(TENZOR_BUILD_ROCM OFF CACHE BOOL "Build ROCm backend" FORCE)
    return()
endif()

# Find hipcc — the actual compiler invoked by every custom command below.
if(WIN32)
    find_program(HIPCC_COMPILER NAMES hipcc.exe hipcc.bat PATHS ${_ROCM_ROOT}/bin NO_DEFAULT_PATH)
else()
    find_program(HIPCC_COMPILER NAMES hipcc PATHS ${_ROCM_ROOT}/bin ${_ROCM_ROOT}/core/bin NO_DEFAULT_PATH)
endif()
if(NOT HIPCC_COMPILER)
    message(WARNING "hipcc not found under ${_ROCM_ROOT} — ROCm backend will not be built.")
    set(TENZOR_BUILD_ROCM OFF CACHE BOOL "Build ROCm backend" FORCE)
    return()
endif()

# Get HIP version
if(DEFINED HIP_VERSION)
    message(STATUS "ROCm Backend: Found HIP ${HIP_VERSION} (hipcc: ${HIPCC_COMPILER})")
else()
    message(STATUS "ROCm Backend: Found HIP (hipcc: ${HIPCC_COMPILER})")
endif()

# Find MIOpen (ROCm's DNN library - equivalent to cuDNN) early: kernel
# sources (pooling, batchnorm, lstm, gru) need to know about it via
# USE_MIOPEN at HIP-kernel compile time, not just at final-link time.
find_package(MIOpen QUIET CONFIG PATHS ${_ROCM_ROOT})
if(MIOpen_FOUND)
    message(STATUS "ROCm Backend: MIOpen support enabled (conv2d, pooling, batchnorm, LSTM, GRU)")
else()
    message(STATUS "ROCm Backend: MIOpen not found, using built-in HIP kernels")
endif()

# =========================================================================
# Host-compiler (pure C++, no HIP) flag translation
# =========================================================================
# rocm_backend.cpp / rocm_kernel_registry.cpp / hip_graph_impl.cpp /
# nccl_backend.cpp / nccl_process_group.cpp are pure C++ (no __global__/HIP
# device code) and are compiled with the PROJECT's normal C++ compiler
# (${CMAKE_CXX_COMPILER}: cl.exe on Windows, g++/clang++ on Linux) rather
# than hipcc, avoiding HIP flag pollution. MSVC needs different flag
# spellings than GCC/Clang for a few of these.
string(TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_UPPER)
# Separate multi-value flag strings into proper lists for VERBATIM custom commands
separate_arguments(ROCM_CXX_FLAGS_LIST NATIVE_COMMAND "${CMAKE_CXX_FLAGS}")
separate_arguments(ROCM_CXX_FLAGS_BUILD_LIST NATIVE_COMMAND "${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}}")
if(MSVC)
    set(ROCM_HOST_STD20 -std:c++20)
    # This MSVC toolset only recognizes "c++latest" (not the literal value
    # "c++23") as a -std: argument -- matches every other MSVC compile in
    # this project, all of which already use -std:c++latest.
    set(ROCM_HOST_STD23 -std:c++latest)
    set(ROCM_HOST_PIC "")               # meaningless for Windows PE targets
    set(ROCM_HOST_WARN "")              # -Wall/-Wextra/-pedantic have no MSVC equivalent worth chasing here
    set(ROCM_HOST_OPENMP /openmp:llvm)  # matches the project-wide MSVC OpenMP choice (see top-level CMakeLists.txt)
    # cl.exe defaults to the static CRT (/MT) unless told otherwise; without
    # this these objects end up MT_StaticRelease while the hipcc-compiled
    # kernel objects (built with -fms-runtime-lib=dll, i.e. /MD) are
    # MD_DynamicRelease, and lld-link hard-fails on the mismatch
    # (/failifmismatch: mismatch detected for 'RuntimeLibrary') when both
    # land in the same tenzor_backend_rocm.dll. Explicit rather than relying
    # on CMAKE_CXX_FLAGS_RELEASE already containing it.
    set(ROCM_HOST_CRT -MD)
else()
    set(ROCM_HOST_STD20 -std=c++20)
    set(ROCM_HOST_STD23 -std=c++23)
    set(ROCM_HOST_PIC -fPIC)
    set(ROCM_HOST_WARN -Wall -Wextra -pedantic)
    set(ROCM_HOST_OPENMP -fopenmp)
    set(ROCM_HOST_CRT "")
endif()

# rocm_backend.cpp / rocm_kernel_registry.cpp / hip_graph_impl.cpp /
# nccl_backend.cpp / nccl_process_group.cpp transitively include
# tenzor/utils/logging.hpp -> <spdlog/spdlog.h>. Unlike a normal CMake
# target, these custom commands don't get spdlog's include dir via
# target_link_libraries(... spdlog::spdlog), so a FetchContent'd spdlog (no
# system package found) fails with "spdlog/spdlog.h: No such file or
# directory". No FMT_CONSTEVAL workaround needed here (unlike the hipcc-
# compiled kernels below) -- these are compiled by cl.exe, whose consteval
# handling has never needed that fix anywhere else in this project.
set(ROCM_HOST_SPDLOG_FLAGS "")
if(TARGET spdlog::spdlog)
    get_target_property(_ROCM_HOST_SPDLOG_INCLUDE_DIRS spdlog::spdlog INTERFACE_INCLUDE_DIRECTORIES)
    if(_ROCM_HOST_SPDLOG_INCLUDE_DIRS)
        foreach(_rocm_host_spdlog_dir ${_ROCM_HOST_SPDLOG_INCLUDE_DIRS})
            if(_rocm_host_spdlog_dir)
                list(APPEND ROCM_HOST_SPDLOG_FLAGS -I${_rocm_host_spdlog_dir})
            endif()
        endforeach()
    endif()
    get_target_property(_ROCM_HOST_SPDLOG_COMPILE_DEFS spdlog::spdlog INTERFACE_COMPILE_DEFINITIONS)
    if(_ROCM_HOST_SPDLOG_COMPILE_DEFS)
        foreach(_rocm_host_spdlog_def ${_ROCM_HOST_SPDLOG_COMPILE_DEFS})
            if(_rocm_host_spdlog_def)
                list(APPEND ROCM_HOST_SPDLOG_FLAGS -D${_rocm_host_spdlog_def})
            endif()
        endforeach()
    endif()
endif()

# =========================================================================
# Function to compile a HIP (.hip.cpp) kernel source to an object file
# =========================================================================
function(tenzor_compile_hip SOURCE_FILE OBJ_VAR)
    cmake_parse_arguments(ARG "" "" "EXTRA_FLAGS" ${ARGN})

    get_filename_component(OBJ_NAME ${SOURCE_FILE} NAME_WE)
    set(OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/${OBJ_NAME}.o")

    if(IS_ABSOLUTE ${SOURCE_FILE})
        set(SRC_FULL ${SOURCE_FILE})
    else()
        set(SRC_FULL ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE})
    endif()

    add_custom_command(
        OUTPUT ${OBJ_FILE}
        COMMAND ${HIPCC_COMPILER}
            -c ${SRC_FULL}
            -o ${OBJ_FILE}
            ${ROCM_HIP_COMMON_FLAGS}
            ${ARG_EXTRA_FLAGS}
        DEPENDS ${SRC_FULL}
        COMMENT "Compiling HIP: ${SOURCE_FILE}"
        VERBATIM
    )

    set(${OBJ_VAR} ${OBJ_FILE} PARENT_SCOPE)
endfunction()

# =========================================================================
# Common HIP compile flags
# =========================================================================
# hipcc does not fully support C++23 — C++20 is the highest supported standard
set(ROCM_HIP_COMMON_FLAGS
    -std=c++20
    -D__HIP_PLATFORM_AMD__
    -D__HIP_PLATFORM_HCC__
    -DTENZOR_ROCM_BACKEND
    -DTENZOR_INTERNAL
    # Keep IEEE Inf/NaN semantics for the WHOLE kernels build so any kernel
    # using isinf()/isnan()/-inf sentinels is correct by default. -ffast-math
    # otherwise implies -ffinite-math-only, which folds those to constants and
    # silently produces NaN — previously each affected file had to opt out by
    # hand (a fragile, easy-to-forget allow-list). The per-file -fno-fast-math
    # overrides below stay for their OTHER reasons (fp-contract / approximate
    # rsqrt / a compiler ICE), which this flag does not address.
    -ffast-math
    -fno-finite-math-only
    # -munsafe-fp-atomics lets the compiler emit native hardware FP atomicAdd
    # (global_atomic_add_f32 etc.) instead of a CAS loop. This is correct and
    # used by reduction / scatter-add / grad-accumulation kernels on every
    # target Tenzor supports (CDNA gfx9xx and RDNA2/3 gfx10xx/gfx11xx all
    # implement hardware FP atomics). The "unsafe" name refers to the IEEE
    # rounding/denormal caveats of the hardware op, NOT to dropping updates:
    # the atomic is still a true read-modify-write, so concurrent accumulation
    # is not lost. ROCm documents that on a (hypothetical) target lacking the
    # hw atomic the compiler falls back to a CAS loop, which is also atomic.
    -munsafe-fp-atomics
    -I${PROJECT_SOURCE_DIR}/include
    -I${_ROCM_ROOT}/include
)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND ROCM_HIP_COMMON_FLAGS -g -ggdb)
else()
    list(APPEND ROCM_HIP_COMMON_FLAGS -O3)
endif()
if(NOT WIN32)
    list(APPEND ROCM_HIP_COMMON_FLAGS -fPIC)
else()
    # M_PI etc. are gated behind _USE_MATH_DEFINES in the Windows UCRT
    # <cmath>/<math.h> regardless of which compiler front-end (MSVC, icpx,
    # hipcc) is used to reach it — matches the same fix already applied
    # project-wide for the MSVC-compiled targets.
    list(APPEND ROCM_HIP_COMMON_FLAGS -D_USE_MATH_DEFINES)
    # hipcc's clang defaults to the static CRT (/MT); tenzor_core.dll (which
    # this backend links against) uses the dynamic CRT (/MD, CMake's default
    # for MSVC). Mixing the two within one process is a hard link error
    # (LNK2038 RuntimeLibrary mismatch) -- match it explicitly.
    list(APPEND ROCM_HIP_COMMON_FLAGS -fms-runtime-lib=dll)
endif()

# GPU targets to build for. CMAKE_HIP_ARCHITECTURES is set by the top-level
# CMakeLists.txt's pre-project() ROCm block (auto-detected via
# rocm_agent_enumerator, falling back to a broad common-architecture list).
foreach(_rocm_arch ${CMAKE_HIP_ARCHITECTURES})
    list(APPEND ROCM_HIP_COMMON_FLAGS --offload-arch=${_rocm_arch})
endforeach()

# include/tenzor/utils/log.hpp (pulled in transitively by logging.hpp, which
# some kernel sources include) #includes <spdlog/spdlog.h> directly. Unlike a
# normal CMake target (which gets this via target_link_libraries(...
# spdlog::spdlog)), these custom commands invoke hipcc directly, so spdlog's
# include dir/defines need to be added explicitly or a FetchContent'd spdlog
# (no system package found) fails with "spdlog/spdlog.h: No such file or
# directory". See src/backends/oneapi/CMakeLists.txt for the identical fix
# (including the FMT_CONSTEVAL workaround below) applied to icpx, which is
# also a Clang-family front-end and hits the exact same fmt/consteval issue.
if(TARGET spdlog::spdlog)
    get_target_property(_ROCM_SPDLOG_INCLUDE_DIRS spdlog::spdlog INTERFACE_INCLUDE_DIRECTORIES)
    if(_ROCM_SPDLOG_INCLUDE_DIRS)
        foreach(_rocm_spdlog_dir ${_ROCM_SPDLOG_INCLUDE_DIRS})
            if(_rocm_spdlog_dir)
                list(APPEND ROCM_HIP_COMMON_FLAGS -I${_rocm_spdlog_dir})
            endif()
        endforeach()
    endif()
    get_target_property(_ROCM_SPDLOG_COMPILE_DEFS spdlog::spdlog INTERFACE_COMPILE_DEFINITIONS)
    if(_ROCM_SPDLOG_COMPILE_DEFS)
        foreach(_rocm_spdlog_def ${_ROCM_SPDLOG_COMPILE_DEFS})
            if(_rocm_spdlog_def)
                list(APPEND ROCM_HIP_COMMON_FLAGS -D${_rocm_spdlog_def})
            endif()
        endforeach()
    endif()
    # hipcc's bundled Clang has the same C++23 consteval strictness as icpx;
    # spdlog's *bundled* fmt copy (fmt/bundled/*.h) is always header-only
    # regardless of SPDLOG_COMPILED_LIB, so this is needed even with that
    # define set above.
    list(APPEND ROCM_HIP_COMMON_FLAGS -DFMT_CONSTEVAL=)
endif()

if(hiprand_FOUND)
    list(APPEND ROCM_HIP_COMMON_FLAGS -DTENZOR_HAS_HIPRAND)
endif()
if(rocfft_FOUND)
    list(APPEND ROCM_HIP_COMMON_FLAGS -DTENZOR_HAS_ROCFFT)
endif()
if(rocsolver_FOUND)
    list(APPEND ROCM_HIP_COMMON_FLAGS -DTENZOR_HAS_ROCSOLVER)
endif()
if(rocsparse_FOUND)
    list(APPEND ROCM_HIP_COMMON_FLAGS -DTENZOR_HAS_ROCSPARSE)
endif()
if(MIOpen_FOUND)
    list(APPEND ROCM_HIP_COMMON_FLAGS -DUSE_MIOPEN -I${MIOpen_INCLUDE_DIRS})
endif()

# Each library's headers live in their own <root>/include/<name>/ subdir
# (e.g. hiprand_kernel.h is at include/hiprand/hiprand_kernel.h, not
# directly under include/), which the general -I${_ROCM_ROOT}/include above
# doesn't reach. hipcub/rocprim/rocthrust are additionally header-only (no
# .lib/.a to link at all) -- their include dirs, pulled the same way as
# spdlog above, are the ONLY way these get found. Needed for e.g.
# hiprand_kernel.h (device-side RNG, math.hip.cpp) and hipcub::DeviceRadixSort
# (sort/cumsum/cumprod/unique kernels).
foreach(_rocm_header_only_tgt hip::hipcub roc::rocprim roc::rocthrust hip::hiprand roc::rocblas roc::rocsolver roc::rocfft roc::rocsparse)
    if(TARGET ${_rocm_header_only_tgt})
        get_target_property(_rocm_hdr_dirs ${_rocm_header_only_tgt} INTERFACE_INCLUDE_DIRECTORIES)
        if(_rocm_hdr_dirs)
            foreach(_rocm_hdr_dir ${_rocm_hdr_dirs})
                if(_rocm_hdr_dir)
                    tenzor_win_short_path(_rocm_hdr_dir)
                    list(APPEND ROCM_HIP_COMMON_FLAGS -I${_rocm_hdr_dir})
                endif()
            endforeach()
        endif()
    endif()
endforeach()
if(hipcub_FOUND)
    message(STATUS "ROCm Backend: hipcub support enabled")
endif()
if(rocprim_FOUND)
    message(STATUS "ROCm Backend: rocprim support enabled")
endif()
if(rocthrust_FOUND)
    message(STATUS "ROCm Backend: rocthrust support enabled")
endif()

# =========================================================================
# HIP kernel sources
# =========================================================================
set(ROCM_HIP_SOURCES
    ../../backend/rocm_caching_allocator.hip.cpp
    kernels/math.hip.cpp
    kernels/activations.hip.cpp
    kernels/batchnorm.hip.cpp
    kernels/comparison.hip.cpp
    kernels/conv2d.hip.cpp
    kernels/conv3d.hip.cpp
    kernels/pooling.hip.cpp
    kernels/indexing.hip.cpp
    kernels/transform.hip.cpp
    kernels/fused_ops.hip.cpp
    kernels/flash_attention_f64.hip.cpp
    kernels/reduction.hip.cpp
    kernels/matmul.hip.cpp
    kernels/normalization.hip.cpp
    kernels/nn_ops.hip.cpp
    kernels/lstm.hip.cpp
    kernels/gru.hip.cpp
    kernels/nms.hip.cpp
    kernels/roi_align.hip.cpp
    kernels/vision.hip.cpp
    kernels/grid_sample.hip.cpp
    kernels/sampling.hip.cpp
    kernels/stft.hip.cpp
    kernels/sort.hip.cpp
    kernels/quantization.hip.cpp
    kernels/sparse.hip.cpp
    kernels/nested.hip.cpp
    kernels/fft.hip.cpp
    kernels/linalg.hip.cpp
    kernels/ctc_loss.hip.cpp
)

# Files needing -fno-fast-math, each for its own reason (kept exactly as
# these were previously documented per-file via set_source_files_properties):
#  - sort.hip.cpp, nms.hip.cpp: workaround for a ROCm clang-22 ICE (segfault
#    in correlated-propagation pass) on rocprim::radix_sort_single_helper at
#    -O3 -ffast-math (hipcub::DeviceRadixSort), observed on gfx1150.
#  - normalization.hip.cpp: -ffp-contract=fast and the approximate rsqrt
#    implied by -ffast-math break Float64 LayerNorm/BatchNorm parity
#    (observed 1e-7 abs diff vs CPU/CUDA/Vulkan/OneAPI). The perf cost of
#    strict IEEE math here is negligible (not on the critical training path).
#  - ctc_loss.hip.cpp (AA.10): relies on IEEE -inf semantics for its log-add
#    sentinel (NEG_INF), isfinite() for the zero_infinity branch, and
#    log1p/exp at the extremes of the log-sum-exp reduction. -ffast-math
#    implies -ffinite-math-only, letting the compiler treat isfinite()/-inf
#    comparisons as always-true and eliminate the guarding branches —
#    producing NaN forward outputs for any T > ~25.
#  - math.hip.cpp (JIT-F050): implements element-wise division, whose IEEE
#    contract the test suite checks (x/0 -> +-Inf for x!=0, 0/0 -> NaN).
#    -ffast-math implies -ffinite-math-only, which lets the compiler assume
#    no op produces Inf/NaN and eliminate the explicit zero-divisor branch in
#    ieee_div() (same mechanism as ctc_loss above). Compiled with full
#    -fno-fast-math (a superset of -fno-finite-math-only) rather than just
#    the latter, because fast-math here also made eager ROCm exp/log/sqrt/
#    rsqrt/div use ~1e-3-relative approximate transcendentals and
#    -ffp-contract=fast, diverging from the same op run through the JIT and
#    from the CPU/CUDA reference (cross-backend parity failure).
#  - activations.hip.cpp (softmax exp), fused_ops.hip.cpp (RMSNorm rsqrt):
#    feed attention/transformer/HRM forward; -ffast-math's ~1e-3-relative
#    approximate transcendentals diverge ~3e-3 from the precise CPU/CUDA
#    reference (cross-backend parity failure).
set(ROCM_NO_FAST_MATH_SOURCES
    kernels/sort.hip.cpp
    kernels/nms.hip.cpp
    kernels/normalization.hip.cpp
    kernels/ctc_loss.hip.cpp
    kernels/math.hip.cpp
    kernels/activations.hip.cpp
    kernels/fused_ops.hip.cpp
)

set(ALL_OBJECTS "")

foreach(SRC ${ROCM_HIP_SOURCES})
    set(_rocm_extra "")
    if(SRC IN_LIST ROCM_NO_FAST_MATH_SOURCES)
        set(_rocm_extra EXTRA_FLAGS -fno-fast-math)
    endif()
    tenzor_compile_hip(${SRC} OBJ_FILE ${_rocm_extra})
    list(APPEND ALL_OBJECTS ${OBJ_FILE})
endforeach()

# =========================================================================
# Manually compile rocm_backend.cpp (pure C++, no HIP flags)
# CMake's HIP language support (when enabled, i.e. on Linux) would otherwise
# add HIP flags to all CXX compilation in this directory.
# =========================================================================
set(ROCM_BACKEND_OBJ "${CMAKE_CURRENT_BINARY_DIR}/rocm_backend.cpp.o")
if(MSVC)
    set(_rocm_backend_out /Fo${ROCM_BACKEND_OBJ})
else()
    set(_rocm_backend_out -o ${ROCM_BACKEND_OBJ})
endif()
add_custom_command(
    OUTPUT ${ROCM_BACKEND_OBJ}
    COMMAND ${CMAKE_CXX_COMPILER}
        -c ${CMAKE_CURRENT_SOURCE_DIR}/rocm_backend.cpp
        ${_rocm_backend_out}
        ${ROCM_HOST_STD20} ${ROCM_HOST_PIC} ${ROCM_HOST_CRT}
        ${ROCM_CXX_FLAGS_LIST} ${ROCM_CXX_FLAGS_BUILD_LIST}
        ${ROCM_HOST_WARN}
        ${ROCM_HOST_OPENMP}
        ${ROCM_HOST_SPDLOG_FLAGS}
        -I${PROJECT_SOURCE_DIR}/include
        -I${_ROCM_ROOT}/include
        -DTENZOR_HAS_MIOPEN
        -DTENZOR_HAS_ROCBLAS
        -DTENZOR_ROCM_BACKEND
        -DUSE_PROF_API=1
        -D__HIP_PLATFORM_AMD__=1
        -Dtenzor_backend_rocm_EXPORTS
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/rocm_backend.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rocm_backend.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rocm_error.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hip_buffer.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hip_graph.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rocm_arch_detect.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/miopen_guards.hpp
    COMMENT "Compiling rocm_backend.cpp (pure C++, no HIP flags)"
    VERBATIM
)
list(APPEND ALL_OBJECTS ${ROCM_BACKEND_OBJ})

# Manually compile rocm_kernel_registry.cpp (also pure C++, no HIP flags)
set(ROCM_REGISTRY_EXTRA_DEFS "")
if(rocfft_FOUND)
    list(APPEND ROCM_REGISTRY_EXTRA_DEFS -DTENZOR_HAS_ROCFFT)
endif()
if(rocsolver_FOUND)
    list(APPEND ROCM_REGISTRY_EXTRA_DEFS -DTENZOR_HAS_ROCSOLVER)
endif()

set(ROCM_KERNEL_REGISTRY_OBJ "${CMAKE_CURRENT_BINARY_DIR}/rocm_kernel_registry.cpp.o")
if(MSVC)
    set(_rocm_registry_out /Fo${ROCM_KERNEL_REGISTRY_OBJ})
else()
    set(_rocm_registry_out -o ${ROCM_KERNEL_REGISTRY_OBJ})
endif()
add_custom_command(
    OUTPUT ${ROCM_KERNEL_REGISTRY_OBJ}
    COMMAND ${CMAKE_CXX_COMPILER}
        -c ${CMAKE_CURRENT_SOURCE_DIR}/rocm_kernel_registry.cpp
        ${_rocm_registry_out}
        ${ROCM_HOST_STD20} ${ROCM_HOST_PIC} ${ROCM_HOST_CRT}
        ${ROCM_CXX_FLAGS_LIST} ${ROCM_CXX_FLAGS_BUILD_LIST}
        ${ROCM_HOST_WARN}
        ${ROCM_HOST_OPENMP}
        ${ROCM_HOST_SPDLOG_FLAGS}
        -I${PROJECT_SOURCE_DIR}/include
        -I${_ROCM_ROOT}/include
        -DTENZOR_HAS_MIOPEN
        -DTENZOR_HAS_ROCBLAS
        -DTENZOR_ROCM_BACKEND
        -DUSE_PROF_API=1
        -D__HIP_PLATFORM_AMD__=1
        -Dtenzor_backend_rocm_EXPORTS
        ${ROCM_REGISTRY_EXTRA_DEFS}
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/rocm_kernel_registry.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rocm_backend.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rocm_error.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hip_buffer.hpp
    COMMENT "Compiling rocm_kernel_registry.cpp (pure C++, no HIP flags)"
    VERBATIM
)
list(APPEND ALL_OBJECTS ${ROCM_KERNEL_REGISTRY_OBJ})

# Manually compile hip_graph_impl.cpp (pure C++ HIP host API — CUDAGraph-interface
# impl backed by HIP graph, plus the backend-registered factory).
set(ROCM_HIP_GRAPH_IMPL_OBJ "${CMAKE_CURRENT_BINARY_DIR}/hip_graph_impl.cpp.o")
if(MSVC)
    set(_rocm_graph_out /Fo${ROCM_HIP_GRAPH_IMPL_OBJ})
else()
    set(_rocm_graph_out -o ${ROCM_HIP_GRAPH_IMPL_OBJ})
endif()
add_custom_command(
    OUTPUT ${ROCM_HIP_GRAPH_IMPL_OBJ}
    COMMAND ${CMAKE_CXX_COMPILER}
        -c ${CMAKE_CURRENT_SOURCE_DIR}/hip_graph_impl.cpp
        ${_rocm_graph_out}
        ${ROCM_HOST_STD20} ${ROCM_HOST_PIC} ${ROCM_HOST_CRT}
        ${ROCM_CXX_FLAGS_LIST} ${ROCM_CXX_FLAGS_BUILD_LIST}
        ${ROCM_HOST_WARN}
        ${ROCM_HOST_OPENMP}
        ${ROCM_HOST_SPDLOG_FLAGS}
        -I${PROJECT_SOURCE_DIR}/include
        -I${_ROCM_ROOT}/include
        -DTENZOR_HAS_MIOPEN
        -DTENZOR_HAS_ROCBLAS
        -DTENZOR_ROCM_BACKEND
        -DUSE_PROF_API=1
        -D__HIP_PLATFORM_AMD__=1
        -Dtenzor_backend_rocm_EXPORTS
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/hip_graph_impl.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hip_graph.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rocm_stream.hpp
    COMMENT "Compiling hip_graph_impl.cpp (pure C++, no HIP flags)"
    VERBATIM
)
list(APPEND ALL_OBJECTS ${ROCM_HIP_GRAPH_IMPL_OBJ})

# RCCL collective library — owns the ROCm twin of the CUDA backend's NCCL
# wiring (src/backends/cuda/CMakeLists.txt). nccl_backend.cpp/
# nccl_process_group.cpp are shared, source-compatible across NVIDIA
# NCCL / AMD RCCL (same nccl* ABI — see nccl_backend.cpp's own comment); this
# backend DSO owns the RCCL twin (hipStream_t) exactly as that comment always
# said it would, but never actually wired up. Compiled as separate objects
# (not folded into rocm_backend.cpp.o) so TENZOR_HAS_NCCL/TENZOR_USE_ROCM can
# be scoped to just these two TUs, matching the CUDA side's per-file
# COMPILE_DEFINITIONS scoping — rocm_backend.cpp itself does not need either
# define: NCCLBackend's stub ncclComm_t/ncclUniqueId typedefs (used when
# TENZOR_HAS_NCCL is undefined) are byte-for-byte layout-identical to the
# real RCCL/NCCL typedefs, so `new NCCLBackend()` from an unguarded TU is
# ABI-safe either way — the same trick the CUDA backend already relies on.
# NOTE: RCCL is Linux-only (AMD does not ship it as part of the Windows HIP
# SDK); rccl_FOUND is simply always FALSE on Windows and this degrades to the
# existing no-op-stub path below, same as when RCCL is absent on Linux.
option(TENZOR_REQUIRE_RCCL "Require RCCL library (fail if not found)" OFF)
find_package(rccl QUIET CONFIG PATHS ${_ROCM_ROOT})
set(ROCM_NCCL_DEFS
    -DTENZOR_ROCM_BACKEND
    -DUSE_PROF_API=1
    -D__HIP_PLATFORM_AMD__=1
    -Dtenzor_backend_rocm_EXPORTS
)
if(rccl_FOUND)
    list(APPEND ROCM_NCCL_DEFS -DTENZOR_HAS_NCCL -DTENZOR_USE_ROCM)
    # Exposed so tests/distributed/CMakeLists.txt can tell whether
    # tenzor_backend_rocm actually has NCCLProcessGroup compiled in (its
    # declaration in process_group.hpp is itself gated on #ifdef
    # TENZOR_HAS_NCCL, so a test referencing the class needs to know this
    # too, not just link the DSO).
    set(TENZOR_ROCM_RCCL_AVAILABLE ON CACHE INTERNAL "ROCm backend built with real RCCL")
    message(STATUS "ROCm Backend: RCCL support enabled")
else()
    set(TENZOR_ROCM_RCCL_AVAILABLE OFF CACHE INTERNAL "ROCm backend built with real RCCL")
    if(TENZOR_REQUIRE_RCCL)
        message(FATAL_ERROR "RCCL library required (-DTENZOR_REQUIRE_RCCL=ON) but not found. "
                "Set CMAKE_PREFIX_PATH or install RCCL.")
    else()
        message(STATUS "ROCm Backend: RCCL not found — collective backend compiled as no-op stub. "
                "Set -DTENZOR_REQUIRE_RCCL=ON to make this an error.")
    endif()
endif()

set(ROCM_NCCL_BACKEND_OBJ "${CMAKE_CURRENT_BINARY_DIR}/nccl_backend.cpp.o")
if(MSVC)
    set(_rocm_nccl_backend_out /Fo${ROCM_NCCL_BACKEND_OBJ})
else()
    set(_rocm_nccl_backend_out -o ${ROCM_NCCL_BACKEND_OBJ})
endif()
add_custom_command(
    OUTPUT ${ROCM_NCCL_BACKEND_OBJ}
    COMMAND ${CMAKE_CXX_COMPILER}
        -c ${PROJECT_SOURCE_DIR}/src/distributed/nccl_backend.cpp
        ${_rocm_nccl_backend_out}
        ${ROCM_HOST_STD23} ${ROCM_HOST_PIC} ${ROCM_HOST_CRT}
        ${ROCM_CXX_FLAGS_LIST} ${ROCM_CXX_FLAGS_BUILD_LIST}
        ${ROCM_HOST_WARN}
        ${ROCM_HOST_OPENMP}
        ${ROCM_HOST_SPDLOG_FLAGS}
        -I${PROJECT_SOURCE_DIR}/include
        -I${_ROCM_ROOT}/include
        ${ROCM_NCCL_DEFS}
    DEPENDS
        ${PROJECT_SOURCE_DIR}/src/distributed/nccl_backend.cpp
        ${PROJECT_SOURCE_DIR}/include/tenzor/distributed/nccl_backend.hpp
        ${PROJECT_SOURCE_DIR}/include/tenzor/distributed/distributed.hpp
    COMMENT "Compiling nccl_backend.cpp (pure C++, RCCL collective backend)"
    VERBATIM
)
list(APPEND ALL_OBJECTS ${ROCM_NCCL_BACKEND_OBJ})

set(ROCM_NCCL_PG_OBJ "${CMAKE_CURRENT_BINARY_DIR}/nccl_process_group.cpp.o")
if(MSVC)
    set(_rocm_nccl_pg_out /Fo${ROCM_NCCL_PG_OBJ})
else()
    set(_rocm_nccl_pg_out -o ${ROCM_NCCL_PG_OBJ})
endif()
add_custom_command(
    OUTPUT ${ROCM_NCCL_PG_OBJ}
    COMMAND ${CMAKE_CXX_COMPILER}
        -c ${PROJECT_SOURCE_DIR}/src/distributed/nccl_process_group.cpp
        ${_rocm_nccl_pg_out}
        ${ROCM_HOST_STD23} ${ROCM_HOST_PIC} ${ROCM_HOST_CRT}
        ${ROCM_CXX_FLAGS_LIST} ${ROCM_CXX_FLAGS_BUILD_LIST}
        ${ROCM_HOST_WARN}
        ${ROCM_HOST_OPENMP}
        ${ROCM_HOST_SPDLOG_FLAGS}
        -I${PROJECT_SOURCE_DIR}/include
        -I${_ROCM_ROOT}/include
        ${ROCM_NCCL_DEFS}
    DEPENDS
        ${PROJECT_SOURCE_DIR}/src/distributed/nccl_process_group.cpp
        ${PROJECT_SOURCE_DIR}/include/tenzor/distributed/process_group.hpp
        ${PROJECT_SOURCE_DIR}/include/tenzor/distributed/nccl_backend.hpp
    COMMENT "Compiling nccl_process_group.cpp (pure C++, RCCL collective backend)"
    VERBATIM
)
list(APPEND ALL_OBJECTS ${ROCM_NCCL_PG_OBJ})

# =========================================================================
# Link into shared library
# =========================================================================
set(ROCM_LIB_DIR "${_ROCM_ROOT}/lib")

set(ROCM_LINK_LIBS "")
if(rocblas_FOUND)
    list(APPEND ROCM_LINK_LIBS -lrocblas)
    message(STATUS "ROCm Backend: rocBLAS support enabled")
else()
    message(STATUS "ROCm Backend: rocBLAS not found")
endif()
if(hiprand_FOUND)
    list(APPEND ROCM_LINK_LIBS -lhiprand)
    message(STATUS "ROCm Backend: hipRAND support enabled")
else()
    message(STATUS "ROCm Backend: hipRAND not found")
endif()
if(rocfft_FOUND)
    list(APPEND ROCM_LINK_LIBS -lrocfft)
    message(STATUS "ROCm Backend: rocFFT support enabled")
else()
    message(STATUS "ROCm Backend: rocFFT not found")
endif()
if(rocsolver_FOUND)
    list(APPEND ROCM_LINK_LIBS -lrocsolver -lrocblas)
    message(STATUS "ROCm Backend: rocSOLVER support enabled")
else()
    message(STATUS "ROCm Backend: rocSOLVER not found")
endif()
if(rocsparse_FOUND)
    list(APPEND ROCM_LINK_LIBS -lrocsparse)
    message(STATUS "ROCm Backend: rocSPARSE support enabled")
else()
    message(STATUS "ROCm Backend: rocSPARSE not found")
endif()
if(rccl_FOUND)
    list(APPEND ROCM_LINK_LIBS -lrccl)
    message(STATUS "ROCm Backend: RCCL support linked into tenzor_backend_rocm")
endif()
if(MIOpen_FOUND)
    list(APPEND ROCM_LINK_LIBS -lMIOpen)
endif()

if(WIN32)
    set(ROCM_DLL_OUTPUT "${PROJECT_BINARY_DIR}/bin/tenzor_backend_rocm.dll")
    # hipcc's Windows link (via lld-link) emits this import lib automatically
    # even with no explicit /IMPLIB: flag; list it as a second OUTPUT so
    # ninja tracks it as a real build product of this rule (needed for
    # downstream link-time dependents, e.g. the python bindings module, via
    # IMPORTED_IMPLIB below).
    set(ROCM_IMPLIB "${PROJECT_BINARY_DIR}/bin/tenzor_backend_rocm.lib")
    add_custom_command(
        OUTPUT ${ROCM_DLL_OUTPUT} ${ROCM_IMPLIB}
        COMMAND ${HIPCC_COMPILER}
            -shared
            ${ALL_OBJECTS}
            -o ${ROCM_DLL_OUTPUT}
            -L${PROJECT_BINARY_DIR}/bin -ltenzor_core
            -L${ROCM_LIB_DIR} -lamdhip64
            ${ROCM_LINK_LIBS}
        DEPENDS ${ALL_OBJECTS}
        COMMENT "Linking ROCm backend shared library"
        VERBATIM
    )
else()
    set(ROCM_DLL_OUTPUT "${PROJECT_BINARY_DIR}/bin/tenzor_backend_rocm.so")
    add_custom_command(
        OUTPUT ${ROCM_DLL_OUTPUT}
        COMMAND ${HIPCC_COMPILER}
            -shared -fPIC
            ${ALL_OBJECTS}
            -o ${ROCM_DLL_OUTPUT}
            -Wl,-soname,tenzor_backend_rocm.so
            -L${PROJECT_BINARY_DIR}/bin -ltenzor_core
            # Build-tree rpath (find libtenzor_core during local runs) plus
            # relocatable $ORIGIN rpaths so the INSTALLED .so finds
            # libtenzor_core relative to itself: $ORIGIN for the wheel
            # (co-located) layout and $ORIGIN/../.. for the FHS layout
            # (lib/tenzor/backends -> lib).
            -Wl,-rpath,${PROJECT_BINARY_DIR}/bin
            -Wl,-rpath,$ORIGIN
            -Wl,-rpath,$ORIGIN/../..
            -L${ROCM_LIB_DIR} -Wl,-rpath,${ROCM_LIB_DIR} -lamdhip64
            ${ROCM_LINK_LIBS}
        DEPENDS ${ALL_OBJECTS}
        COMMENT "Linking ROCm backend shared library"
        VERBATIM
    )
endif()

# Custom target for building the shared library
add_custom_target(tenzor_backend_rocm_build ALL
    DEPENDS ${ROCM_DLL_OUTPUT}
)

# Ensure core library is built first
add_dependencies(tenzor_backend_rocm_build tenzor_core)

# Create an imported library target for downstream use
add_library(tenzor_backend_rocm SHARED IMPORTED GLOBAL)
set_target_properties(tenzor_backend_rocm PROPERTIES
    IMPORTED_LOCATION ${ROCM_DLL_OUTPUT}
    INTERFACE_INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/include"
)
if(WIN32)
    # Required for any target that link-time-depends on this IMPORTED SHARED
    # target on Windows (e.g. the python bindings module).
    set_target_properties(tenzor_backend_rocm PROPERTIES
        IMPORTED_IMPLIB ${ROCM_IMPLIB}
    )
endif()
add_dependencies(tenzor_backend_rocm tenzor_backend_rocm_build)

if(rocblas_FOUND)
    set_property(TARGET tenzor_backend_rocm APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS TENZOR_HAS_ROCBLAS)
endif()
if(hiprand_FOUND)
    set_property(TARGET tenzor_backend_rocm APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS TENZOR_HAS_HIPRAND)
endif()
if(rocfft_FOUND)
    set_property(TARGET tenzor_backend_rocm APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS TENZOR_HAS_ROCFFT)
endif()
if(rocsolver_FOUND)
    set_property(TARGET tenzor_backend_rocm APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS TENZOR_HAS_ROCSOLVER)
endif()
if(rocsparse_FOUND)
    set_property(TARGET tenzor_backend_rocm APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS TENZOR_HAS_ROCSPARSE)
endif()
if(MIOpen_FOUND)
    set_property(TARGET tenzor_backend_rocm APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS TENZOR_HAS_MIOPEN)
endif()

# Installation rules (FHS layout only; wheel installs are handled centrally in
# the root CMakeLists.txt SKBUILD branch)
if(NOT SKBUILD)
    install(FILES ${ROCM_DLL_OUTPUT}
        DESTINATION lib/tenzor/backends
    )
endif()

# Install header files for ROCm backend if they exist
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/rocm_fp16.hpp")
    install(FILES rocm_fp16.hpp
        DESTINATION include/tenzor/backends/rocm
    )
endif()

# Summary message
message(STATUS "ROCm Backend: Configured successfully")
if(DEFINED HIP_VERSION)
    message(STATUS "  HIP Version: ${HIP_VERSION}")
endif()
message(STATUS "  HIP Architectures: ${CMAKE_HIP_ARCHITECTURES}")
message(STATUS "  Compiler: ${HIPCC_COMPILER}")
message(STATUS "  Note: GEMM operations use rocBLAS library for optimal performance")
message(STATUS "  NOTE: To compile for your specific GPU, set CMAKE_HIP_ARCHITECTURES")
message(STATUS "        Example: cmake -DCMAKE_HIP_ARCHITECTURES=\"gfx906;gfx908\"")
