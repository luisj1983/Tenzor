# OneAPI Backend - Intel CPU/GPU via SYCL
# Uses icpx from Intel oneAPI Base Toolkit
#
# Note: We use add_custom_command() instead of add_library() because CMake
# does not natively support icpx's -fsycl compilation mode. The icpx compiler
# requires -fsycl flags at both compile and link time, and CMake's CXX compiler
# detection doesn't handle SYCL target triples. Using custom commands gives us
# full control over the compilation pipeline.

# Find Intel icpx compiler
find_program(ICPX_COMPILER NAMES icpx)
if(NOT ICPX_COMPILER)
    message(WARNING "OneAPI backend requires icpx from Intel oneAPI Base Toolkit")
    message(WARNING "Install from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html")
    # Returning here alone leaves TENZOR_BUILD_ONEAPI (and thus the
    # tenzor_backend_oneapi TARGET) in a state the rest of the build can't see:
    # test CMakeLists.txt files gate on `if(TENZOR_BUILD_ONEAPI)` without also
    # checking `TARGET tenzor_backend_oneapi`, so find_package(...REQUIRED)/link
    # calls inside those blocks would still run and abort the ENTIRE configure
    # (see FINDING 26). FORCE the cache var off so every later
    # if(TENZOR_BUILD_ONEAPI) check — including in tests/ — degrades gracefully,
    # mirroring the CUDA/ROCm probe-and-FORCE-OFF pattern in
    # src/backends/CMakeLists.txt.
    set(TENZOR_BUILD_ONEAPI OFF CACHE BOOL "Build OneAPI backend" FORCE)
    return()
endif()

message(STATUS "OneAPI Backend: Found icpx at ${ICPX_COMPILER}")

set(SYCL_COMPILER ${ICPX_COMPILER})
set(SYCL_TARGET_ARCH "spir64")

# Derive SYCL paths from compiler location
get_filename_component(SYCL_BIN_DIR ${SYCL_COMPILER} DIRECTORY)
get_filename_component(SYCL_COMPILER_DIR ${SYCL_BIN_DIR} DIRECTORY)
set(SYCL_INCLUDE_DIR "${SYCL_COMPILER_DIR}/include")
set(SYCL_LIBRARY_DIR "${SYCL_COMPILER_DIR}/lib")

# Verify SYCL library exists
if(NOT EXISTS "${SYCL_LIBRARY_DIR}/libsycl.so")
    if(EXISTS "${SYCL_COMPILER_DIR}/lib64/libsycl.so")
        set(SYCL_LIBRARY_DIR "${SYCL_COMPILER_DIR}/lib64")
    else()
        message(WARNING "SYCL library not found at ${SYCL_LIBRARY_DIR}/libsycl.so")
    endif()
endif()

message(STATUS "OneAPI Backend: SYCL include: ${SYCL_INCLUDE_DIR}")
message(STATUS "OneAPI Backend: SYCL library: ${SYCL_LIBRARY_DIR}")

find_package(MKL QUIET)

# Check for oneMKL SYCL interface
set(ONEMKL_SYCL_FOUND FALSE)
if(MKL_FOUND)
    find_path(ONEMKL_SYCL_INCLUDE_DIR
        NAMES oneapi/mkl.hpp
        HINTS ${MKL_INCLUDE} ${SYCL_COMPILER_DIR}/include
        PATH_SUFFIXES include
    )
    if(ONEMKL_SYCL_INCLUDE_DIR)
        set(ONEMKL_SYCL_FOUND TRUE)
        message(STATUS "OneAPI Backend: oneMKL SYCL interface found at ${ONEMKL_SYCL_INCLUDE_DIR}")
    else()
        message(STATUS "OneAPI Backend: oneMKL SYCL interface not found - using native SYCL kernels")
    endif()
endif()

# Opt-in override to build the native (non-oneMKL) SYCL linalg kernels even when
# oneMKL is installed. Lets the arbitrary-N global-memory fallback be built and
# tested on machines that have oneMKL, and supports MKL-free deployments.
option(TENZOR_ONEAPI_FORCE_NATIVE_LINALG
       "Force the native SYCL linalg fallback (ignore oneMKL) for the OneAPI backend" OFF)
if(TENZOR_ONEAPI_FORCE_NATIVE_LINALG)
    set(ONEMKL_SYCL_FOUND FALSE)
    message(STATUS "OneAPI Backend: TENZOR_ONEAPI_FORCE_NATIVE_LINALG=ON - using native SYCL linalg")
endif()

# Check for oneDPL (Intel DPC++ Library)
set(ONEDPL_FOUND FALSE)
find_path(ONEDPL_INCLUDE_DIR
    NAMES oneapi/dpl/algorithm
    HINTS ${SYCL_COMPILER_DIR}/include
    PATH_SUFFIXES include
)
if(ONEDPL_INCLUDE_DIR)
    set(ONEDPL_FOUND TRUE)
    message(STATUS "OneAPI Backend: oneDPL found at ${ONEDPL_INCLUDE_DIR}")
else()
    message(STATUS "OneAPI Backend: oneDPL not found - using on-device SYCL bitonic sort fallback")
endif()

# Check for oneDNN (Intel oneAPI Deep Neural Network Library). When present it
# provides the optimized convolution_forward/backward primitives used by
# conv2d.cpp / conv3d.cpp (guarded by TENZOR_HAS_ONEDNN). When absent, the
# always-compiled im2col/im3col SYCL fallback is used instead — both paths cover
# the full Float32/Float64/Float16/BFloat16 dtype set.
set(ONEDNN_FOUND FALSE)
find_path(ONEDNN_INCLUDE_DIR
    NAMES oneapi/dnnl/dnnl_sycl.hpp
    HINTS ${SYCL_COMPILER_DIR}/include ${SYCL_INCLUDE_DIR}
    PATH_SUFFIXES include
)
find_library(ONEDNN_LIBRARY
    NAMES dnnl
    HINTS ${SYCL_COMPILER_DIR}/lib ${SYCL_COMPILER_DIR}/lib64 ${SYCL_LIBRARY_DIR}
)
option(TENZOR_ONEAPI_DISABLE_ONEDNN
       "Force the native im2col/im3col SYCL conv fallback (ignore oneDNN) for the OneAPI backend" OFF)
if(ONEDNN_INCLUDE_DIR AND ONEDNN_LIBRARY AND NOT TENZOR_ONEAPI_DISABLE_ONEDNN)
    set(ONEDNN_FOUND TRUE)
    get_filename_component(ONEDNN_LIB_DIR ${ONEDNN_LIBRARY} DIRECTORY)
    message(STATUS "OneAPI Backend: oneDNN found (include=${ONEDNN_INCLUDE_DIR}, lib=${ONEDNN_LIBRARY})")
else()
    message(STATUS "OneAPI Backend: oneDNN not found/disabled - using im2col/im3col SYCL conv fallback")
endif()

# =========================================================================
# Common SYCL compile/link flags
# =========================================================================
# icpx supports C++23 — using it for std::expected, std::ranges, and other C++23 features
set(SYCL_COMMON_FLAGS
    -std=c++23 -fPIC
    -fsycl -fsycl-targets=${SYCL_TARGET_ARCH}
    -O3 -DNDEBUG
    -I${PROJECT_SOURCE_DIR}/include
)

if(ONEDPL_FOUND)
    list(APPEND SYCL_COMMON_FLAGS -DTENZOR_HAS_ONEDPL)
endif()

if(ONEDNN_FOUND)
    list(APPEND SYCL_COMMON_FLAGS -DTENZOR_HAS_ONEDNN -I${ONEDNN_INCLUDE_DIR})
endif()

set(ONEMKL_COMPILE_FLAGS "")
if(ONEMKL_SYCL_FOUND)
    list(APPEND ONEMKL_COMPILE_FLAGS -DTENZOR_HAS_ONEMKL -I${ONEMKL_SYCL_INCLUDE_DIR})

    # Check if oneMKL exposes `geev` (non-symmetric eigendecomposition) in its
    # SYCL LAPACK interface. CMake's `check_cxx_source_compiles` uses the host
    # C++ compiler with `-fsycl`, which tolerates unresolved oneMKL symbols at
    # compile time and gives FALSE POSITIVES. Probe directly with icpx +
    # `-fsycl -c` instead — that's the only reliable test.
    #
    # As of oneMKL 2025.3, the SYCL interface does NOT expose geev (the oneMath
    # spec lists only symmetric eigenvalue routines: syevd / heevd / sygvd /
    # hegvd). When this probe fails, `OpId::LinalgEig` is served by the always-
    # compiled native SYCL Francis double-shift QR kernel in
    # src/backends/oneapi/kernels/linalg.cpp (`linalg_eig_qr_kernel`).
    # The probe source lives in-tree at src/backends/oneapi/geev_probe/ so it
    # can be inspected, version-controlled, and edited without touching CMake.
    set(_GEEV_PROBE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/geev_probe/geev_probe.cpp")
    set(_GEEV_PROBE_DIR "${CMAKE_CURRENT_BINARY_DIR}/geev_probe")
    file(MAKE_DIRECTORY "${_GEEV_PROBE_DIR}")
    execute_process(
        COMMAND ${SYCL_COMPILER} -fsycl -c
                -o ${_GEEV_PROBE_DIR}/geev_probe.o
                -I${ONEMKL_SYCL_INCLUDE_DIR}
                -DTENZOR_HAS_ONEMKL
                ${_GEEV_PROBE_SRC}
        RESULT_VARIABLE _GEEV_PROBE_RC
        OUTPUT_QUIET ERROR_QUIET
    )
    if(_GEEV_PROBE_RC EQUAL 0)
        set(TENZOR_ONEMKL_HAS_GEEV TRUE CACHE INTERNAL "oneMKL SYCL geev available")
        list(APPEND ONEMKL_COMPILE_FLAGS -DTENZOR_HAS_ONEMKL_GEEV)
        message(STATUS "oneMKL: geev (non-symmetric eigendecomposition) detected -- "
                       "used as fast path for OpId::LinalgEig")
    else()
        set(TENZOR_ONEMKL_HAS_GEEV FALSE CACHE INTERNAL "oneMKL SYCL geev unavailable")
        message(STATUS "oneMKL: geev not exposed in SYCL interface (expected) -- "
                       "OneAPI LinalgEig uses native SYCL Francis QR kernel.")
    endif()
endif()

# =========================================================================
# Function to compile a SYCL source file to an object file
# =========================================================================
function(tenzor_compile_sycl SOURCE_FILE OBJ_VAR)
    cmake_parse_arguments(ARG "" "" "EXTRA_FLAGS" ${ARGN})

    get_filename_component(OBJ_NAME ${SOURCE_FILE} NAME_WE)
    # Disambiguate backend glue vs kernel objects
    if(ARG_EXTRA_FLAGS)
        set(OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/${OBJ_NAME}_backend.o")
    else()
        set(OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/${OBJ_NAME}.o")
    endif()

    # Resolve source path (may be relative or absolute)
    if(IS_ABSOLUTE ${SOURCE_FILE})
        set(SRC_FULL ${SOURCE_FILE})
    else()
        set(SRC_FULL ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE})
    endif()

    add_custom_command(
        OUTPUT ${OBJ_FILE}
        COMMAND ${SYCL_COMPILER}
            -c ${SRC_FULL}
            -o ${OBJ_FILE}
            ${SYCL_COMMON_FLAGS}
            ${ONEMKL_COMPILE_FLAGS}
            ${ARG_EXTRA_FLAGS}
        DEPENDS ${SRC_FULL}
        COMMENT "Compiling SYCL: ${SOURCE_FILE}"
        VERBATIM
    )

    set(${OBJ_VAR} ${OBJ_FILE} PARENT_SCOPE)
endfunction()

# =========================================================================
# SYCL kernel sources
# =========================================================================
set(ONEAPI_SYCL_SOURCES
    kernels/math.cpp
    kernels/special_math.cpp
    kernels/grid_sample.cpp
    kernels/sampling.cpp
    kernels/philox_dropout.cpp   # F13-followup: device-side Philox dropout
    kernels/stft.cpp
    kernels/activations.cpp
    kernels/reduction.cpp
    kernels/transform.cpp
    kernels/creation.cpp
    kernels/batchnorm.cpp
    kernels/conv2d.cpp
    kernels/conv3d.cpp
    kernels/pooling.cpp
    kernels/indexing.cpp
    kernels/advanced_index.cpp
    kernels/embedding.cpp
    kernels/comparison.cpp
    kernels/im2col.cpp
    kernels/expand.cpp
    kernels/statistical.cpp
    kernels/fused_ops.cpp
    kernels/flash_attention_f64.cpp
    kernels/lstm.cpp
    kernels/gru.cpp
    kernels/vision.cpp
    kernels/quantization.cpp
    kernels/utilities.cpp
    kernels/rnn.cpp
    kernels/linalg.cpp
    kernels/fft.cpp
    kernels/sparse.cpp
    kernels/nested_attention.cpp
    kernels/ctc_loss.cpp
)

set(ALL_OBJECTS "")

foreach(SRC ${ONEAPI_SYCL_SOURCES})
    set(_fp_extra "")
    if(SRC STREQUAL "kernels/activations.cpp" OR SRC STREQUAL "kernels/fused_ops.cpp")
        # softmax exp / RMSNorm rsqrt feed attention/transformer/HRM forward.
        # icpx defaults to -fp-model=fast (~1e-3 approximate transcendentals),
        # which diverges ~3e-3 from the precise CPU/CUDA reference (cross-backend
        # parity failure). Compile these with precise FP so they match.
        set(_fp_extra EXTRA_FLAGS -fp-model=precise)
    elseif(SRC STREQUAL "kernels/ctc_loss.cpp")
        # Mirrors ROCm's ctc_loss.hip.cpp -fno-fast-math exemption (see
        # src/backends/rocm/CMakeLists.txt, AA.10): ctc_loss relies on IEEE
        # -inf semantics for its log-add sentinel (NEG_INF), isfinite() for
        # the zero_infinity branch, and log1p/exp at the extremes of the
        # log-sum-exp reduction. icpx's default -fp-model=fast permits the
        # same finite-math-only assumptions as -ffast-math, which lets the
        # compiler treat isfinite()/-inf comparisons as always-true and
        # eliminate the branches guarding them — producing NaN forward
        # outputs for any T > ~25. -fp-model=precise disables that.
        set(_fp_extra EXTRA_FLAGS -fp-model=precise)
    endif()
    tenzor_compile_sycl(${SRC} OBJ_FILE ${_fp_extra})
    list(APPEND ALL_OBJECTS ${OBJ_FILE})
endforeach()

# =========================================================================
# Backend glue sources (need TENZOR_ONEAPI_BACKEND define)
# =========================================================================
set(BACKEND_GLUE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/oneapi_backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/oneapi_kernel_registry.cpp
    ${PROJECT_SOURCE_DIR}/src/backend/oneapi_caching_allocator.cpp
)

foreach(SRC ${BACKEND_GLUE_SOURCES})
    tenzor_compile_sycl(${SRC} OBJ_FILE EXTRA_FLAGS -DTENZOR_ONEAPI_BACKEND)
    list(APPEND ALL_OBJECTS ${OBJ_FILE})
endforeach()

# =========================================================================
# Link into shared library
# =========================================================================
set(ONEMKL_LINK_FLAGS "")
if(ONEMKL_SYCL_FOUND)
    get_filename_component(ONEMKL_LIB_DIR "${ONEMKL_SYCL_INCLUDE_DIR}/../lib" ABSOLUTE)
    # Use GNU threading layer instead of Intel threading to avoid cleanup symbol issues
    # Link order: mkl_sycl -> mkl_intel_ilp64 -> mkl_gnu_thread -> mkl_core -> gomp
    set(ONEMKL_LINK_FLAGS
        -L${ONEMKL_LIB_DIR} -Wl,-rpath,${ONEMKL_LIB_DIR}
        -Wl,--no-as-needed -lmkl_sycl -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -Wl,--as-needed
    )
endif()

set(ONEDNN_LINK_FLAGS "")
if(ONEDNN_FOUND)
    set(ONEDNN_LINK_FLAGS -L${ONEDNN_LIB_DIR} -Wl,-rpath,${ONEDNN_LIB_DIR} -ldnnl)
endif()

set(SYCL_LINK_FLAGS -L${SYCL_LIBRARY_DIR} -lsycl -Wl,-rpath,${SYCL_LIBRARY_DIR})

set(ONEAPI_SO_OUTPUT "${PROJECT_BINARY_DIR}/bin/tenzor_backend_oneapi.so")
add_custom_command(
    OUTPUT ${ONEAPI_SO_OUTPUT}
    COMMAND ${SYCL_COMPILER}
        -shared -fPIC
        -fsycl -fsycl-targets=${SYCL_TARGET_ARCH}
        ${ALL_OBJECTS}
        -o ${ONEAPI_SO_OUTPUT}
        -Wl,-soname,tenzor_backend_oneapi.so
        -L${PROJECT_BINARY_DIR}/bin -ltenzor_core
        # Build-tree rpath (find libtenzor_core during local runs) plus
        # relocatable $ORIGIN rpaths so the INSTALLED .so finds libtenzor_core
        # relative to itself: $ORIGIN for the wheel (co-located) layout and
        # $ORIGIN/../.. for the FHS layout (lib/tenzor/backends -> lib).
        -Wl,-rpath,${PROJECT_BINARY_DIR}/bin
        -Wl,-rpath,$ORIGIN
        -Wl,-rpath,$ORIGIN/../..
        ${SYCL_LINK_FLAGS}
        ${ONEMKL_LINK_FLAGS}
        ${ONEDNN_LINK_FLAGS}
    DEPENDS ${ALL_OBJECTS}
    COMMENT "Linking oneAPI backend shared library"
    VERBATIM
)

# Custom target for building the shared library
add_custom_target(tenzor_backend_oneapi_build ALL
    DEPENDS ${ONEAPI_SO_OUTPUT}
)

# Ensure core library is built first
add_dependencies(tenzor_backend_oneapi_build tenzor_core)

# Create an imported library target for downstream use
add_library(tenzor_backend_oneapi SHARED IMPORTED GLOBAL)
set_target_properties(tenzor_backend_oneapi PROPERTIES
    IMPORTED_LOCATION ${ONEAPI_SO_OUTPUT}
    INTERFACE_INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/include"
)
add_dependencies(tenzor_backend_oneapi tenzor_backend_oneapi_build)

# Installation (FHS layout only; wheel installs are handled centrally in the
# root CMakeLists.txt SKBUILD branch)
if(NOT SKBUILD)
    install(FILES ${ONEAPI_SO_OUTPUT}
        DESTINATION lib/tenzor/backends
    )
endif()

message(STATUS "OneAPI Backend: Configured successfully")
message(STATUS "  Compiler: ${SYCL_COMPILER}")
message(STATUS "  Target: ${SYCL_TARGET_ARCH} (Intel CPU/GPU)")
if(ONEMKL_SYCL_FOUND)
    message(STATUS "  oneMKL SYCL: Enabled")
else()
    message(STATUS "  oneMKL SYCL: Disabled")
endif()
if(ONEDPL_FOUND)
    message(STATUS "  oneDPL: Enabled")
else()
    message(STATUS "  oneDPL: Disabled")
endif()
if(ONEDNN_FOUND)
    message(STATUS "  oneDNN conv primitives: Enabled")
else()
    message(STATUS "  oneDNN conv primitives: Disabled (im2col/im3col SYCL fallback)")
endif()
