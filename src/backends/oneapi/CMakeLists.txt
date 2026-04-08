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
    message(STATUS "OneAPI Backend: oneDPL not found - using host-side sort fallback")
endif()

# =========================================================================
# Common SYCL compile/link flags
# =========================================================================
# icpx supports C++23 — using it for std::expected, std::ranges, and other C++23 features
set(SYCL_COMMON_FLAGS
    -std=c++23 -fPIC
    -fsycl -fsycl-targets=${SYCL_TARGET_ARCH}
    -O3 -DNDEBUG
    -I${CMAKE_SOURCE_DIR}/include
)

if(ONEDPL_FOUND)
    list(APPEND SYCL_COMMON_FLAGS -DTENZOR_HAS_ONEDPL)
endif()

set(ONEMKL_COMPILE_FLAGS "")
if(ONEMKL_SYCL_FOUND)
    list(APPEND ONEMKL_COMPILE_FLAGS -DTENZOR_HAS_ONEMKL -I${ONEMKL_SYCL_INCLUDE_DIR})

    # Check if oneMKL exposes geev (non-symmetric eigendecomposition)
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES ${ONEMKL_SYCL_INCLUDE_DIR})
    set(CMAKE_REQUIRED_FLAGS "-fsycl")
    set(CMAKE_REQUIRED_DEFINITIONS -DTENZOR_HAS_ONEMKL)
    check_cxx_source_compiles("
        #include <oneapi/mkl.hpp>
        #include <sycl/sycl.hpp>
        int main() {
            sycl::queue q;
            (void)::oneapi::mkl::lapack::geev_scratchpad_size<float>(
                q, ::oneapi::mkl::compvl::vec, ::oneapi::mkl::compvr::novec, 1, 1);
            return 0;
        }
    " TENZOR_ONEMKL_HAS_GEEV)
    unset(CMAKE_REQUIRED_INCLUDES)
    unset(CMAKE_REQUIRED_FLAGS)
    unset(CMAKE_REQUIRED_DEFINITIONS)
    if(TENZOR_ONEMKL_HAS_GEEV)
        list(APPEND ONEMKL_COMPILE_FLAGS -DTENZOR_HAS_ONEMKL_GEEV)
        message(STATUS "oneMKL: geev (non-symmetric eigendecomposition) detected")
    else()
        message(STATUS "oneMKL: geev not available — LinalgEig will use stub")
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
    kernels/embedding.cpp
    kernels/comparison.cpp
    kernels/im2col.cpp
    kernels/expand.cpp
    kernels/statistical.cpp
    kernels/fused_ops.cpp
    kernels/lstm.cpp
    kernels/gru.cpp
    kernels/vision.cpp
    kernels/quantization.cpp
    kernels/utilities.cpp
    kernels/rnn.cpp
    kernels/linalg.cpp
    kernels/fft.cpp
    kernels/sparse.cpp
)

set(ALL_OBJECTS "")

foreach(SRC ${ONEAPI_SYCL_SOURCES})
    tenzor_compile_sycl(${SRC} OBJ_FILE)
    list(APPEND ALL_OBJECTS ${OBJ_FILE})
endforeach()

# =========================================================================
# Backend glue sources (need TENZOR_ONEAPI_BACKEND define)
# =========================================================================
set(BACKEND_GLUE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/oneapi_backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/oneapi_kernel_registry.cpp
    ${CMAKE_SOURCE_DIR}/src/backend/oneapi_caching_allocator.cpp
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

set(SYCL_LINK_FLAGS -L${SYCL_LIBRARY_DIR} -lsycl -Wl,-rpath,${SYCL_LIBRARY_DIR})

set(ONEAPI_SO_OUTPUT "${CMAKE_BINARY_DIR}/bin/tenzor_backend_oneapi.so")
add_custom_command(
    OUTPUT ${ONEAPI_SO_OUTPUT}
    COMMAND ${SYCL_COMPILER}
        -shared -fPIC
        -fsycl -fsycl-targets=${SYCL_TARGET_ARCH}
        ${ALL_OBJECTS}
        -o ${ONEAPI_SO_OUTPUT}
        -Wl,-soname,tenzor_backend_oneapi.so
        -L${CMAKE_BINARY_DIR}/bin -ltenzor_core
        -Wl,-rpath,${CMAKE_BINARY_DIR}/bin
        ${SYCL_LINK_FLAGS}
        ${ONEMKL_LINK_FLAGS}
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
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/include"
)
add_dependencies(tenzor_backend_oneapi tenzor_backend_oneapi_build)

# Installation
install(FILES ${ONEAPI_SO_OUTPUT}
    DESTINATION lib/tenzor/backends
)

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
