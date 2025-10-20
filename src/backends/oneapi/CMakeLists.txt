# OneAPI Backend
# Requires Intel oneAPI Base Toolkit for Intel GPUs and CPUs
# Also supports NVIDIA GPUs via CUDA backend plugin

# Find Intel oneAPI (use IntelSYCL for oneAPI 2025.0+, fallback to IntelDPCPP for older versions)
find_package(IntelSYCL QUIET)
if(NOT IntelSYCL_FOUND)
    find_package(IntelDPCPP QUIET)
    if(NOT IntelDPCPP_FOUND)
        message(WARNING "Intel oneAPI SYCL compiler not found. OneAPI backend will not be built.")
        message(WARNING "Install from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html")
        return()
    endif()
    set(SYCL_TARGET "IntelDPCPP::DPCPP")
    message(STATUS "OneAPI Backend: Found Intel DPCPP (legacy)")
else()
    set(SYCL_TARGET "IntelSYCL::SYCL_CXX")
    message(STATUS "OneAPI Backend: Found Intel SYCL")
endif()

find_package(MKL QUIET)

set(ONEAPI_BACKEND_SOURCES
    oneapi_backend.cpp
    kernels/math.cpp
    kernels/activations.cpp
    kernels/reduction.cpp
    kernels/transform.cpp
    kernels/batchnorm.cpp
    kernels/conv2d.cpp
    kernels/pooling.cpp
    kernels/indexing.cpp
)

# Separate SYCL kernel sources from C++ backend source
# Similar to ROCm's 2-stage approach
set(ONEAPI_SYCL_SOURCES
    kernels/math.cpp
    kernels/activations.cpp
    kernels/reduction.cpp
    kernels/transform.cpp
    kernels/batchnorm.cpp
    kernels/conv2d.cpp
    kernels/pooling.cpp
    kernels/indexing.cpp
)

set(ONEAPI_BACKEND_SOURCE
    oneapi_backend.cpp
)

# Stage 1: Compile SYCL kernels into static library using icpx
# This is built with icpx via custom commands
set(ICPX_COMPILER "${INTEL_SYCL_COMPILER}")

# Build each SYCL kernel file individually
set(SYCL_OBJECTS "")
foreach(SRC ${ONEAPI_SYCL_SOURCES})
    get_filename_component(OBJ_NAME ${SRC} NAME_WE)
    set(OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/${OBJ_NAME}.o")
    list(APPEND SYCL_OBJECTS ${OBJ_FILE})

    add_custom_command(
        OUTPUT ${OBJ_FILE}
        COMMAND ${ICPX_COMPILER}
            -c ${CMAKE_CURRENT_SOURCE_DIR}/${SRC}
            -o ${OBJ_FILE}
            -std=c++23 -fPIC
            -fsycl -fsycl-targets=${SYCL_TARGET_ARCH}
            -O3 -ffast-math
            -I${CMAKE_SOURCE_DIR}/include
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SRC}
        COMMENT "Compiling SYCL kernel: ${SRC}"
        VERBATIM
    )
endforeach()

# Create static library from SYCL objects
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libtenzor_oneapi_kernels.a
    COMMAND ${CMAKE_AR} rcs ${CMAKE_CURRENT_BINARY_DIR}/libtenzor_oneapi_kernels.a ${SYCL_OBJECTS}
    DEPENDS ${SYCL_OBJECTS}
    COMMENT "Creating SYCL kernels static library"
    VERBATIM
)

# Stage 2: Manually compile oneapi_backend.cpp with g++ (no SYCL flags)
set(ONEAPI_BACKEND_OBJ "${CMAKE_CURRENT_BINARY_DIR}/oneapi_backend.cpp.o")
add_custom_command(
    OUTPUT ${ONEAPI_BACKEND_OBJ}
    COMMAND ${CMAKE_CXX_COMPILER}
        -c ${CMAKE_CURRENT_SOURCE_DIR}/${ONEAPI_BACKEND_SOURCE}
        -o ${ONEAPI_BACKEND_OBJ}
        -std=c++23 -fPIC
        -O3 -DNDEBUG
        -Wall -Wextra -pedantic
        -march=native
        -Wno-unused-parameter -Wno-unused-variable
        -fopenmp
        -I${CMAKE_SOURCE_DIR}/include
        -I${SYCL_INCLUDE_DIR}
        -DTENZOR_ONEAPI_BACKEND
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${ONEAPI_BACKEND_SOURCE}
    COMMENT "Compiling oneapi_backend.cpp (pure C++, no SYCL flags)"
    VERBATIM
)

# Final link step: Create shared library using icpx
# This links the backend object with SYCL kernels library
set(ONEAPI_SO_OUTPUT "${CMAKE_BINARY_DIR}/../bin/tenzor_backend_oneapi.so")

# Build MKL link flags if available
set(MKL_LINK_FLAGS "")
if(MKL_FOUND)
    set(MKL_LINK_FLAGS "-L/opt/intel/oneapi/2025.2/lib -lmkl_sycl -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5")
endif()

add_custom_command(
    OUTPUT ${ONEAPI_SO_OUTPUT}
    COMMAND ${ICPX_COMPILER}
        -shared
        -fsycl -fsycl-targets=${SYCL_TARGET_ARCH}
        -o ${ONEAPI_SO_OUTPUT}
        ${ONEAPI_BACKEND_OBJ}
        ${CMAKE_CURRENT_BINARY_DIR}/libtenzor_oneapi_kernels.a
        -L${CMAKE_BINARY_DIR}/../bin -ltenzor_core
        -L${SYCL_LIBRARY_DIR} -lsycl
        ${MKL_LINK_FLAGS}
    DEPENDS ${ONEAPI_BACKEND_OBJ} ${CMAKE_CURRENT_BINARY_DIR}/libtenzor_oneapi_kernels.a tenzor_core
    COMMENT "Linking OneAPI backend shared library with icpx"
    VERBATIM
)

# Create custom target for the shared library
add_custom_target(tenzor_backend_oneapi ALL
    DEPENDS ${ONEAPI_SO_OUTPUT}
)

# Note: MKL and oneDNN linking is handled in the custom link command above
if(MKL_FOUND)
    message(STATUS "OneAPI Backend: oneMKL support enabled")
endif()

# Find oneDNN (Intel's DNN library)
find_package(dnnl QUIET)
if(dnnl_FOUND)
    message(STATUS "OneAPI Backend: oneDNN support enabled")
else()
    message(STATUS "OneAPI Backend: oneDNN not found, using built-in kernels")
endif()

# Detect SYCL target architecture
# For NVIDIA GPUs via CUDA backend: nvidia_gpu_sm_XX (XX = compute capability)
# For Intel GPUs: spir64_gen or intel_gpu_XX
# For CPU: spir64_x86_64
if(NOT DEFINED SYCL_TARGET_ARCH)
    # Auto-detect from CUDA if available
    if(CMAKE_CUDA_COMPILER AND CMAKE_CUDA_ARCHITECTURES)
        # Use first CUDA architecture
        list(GET CMAKE_CUDA_ARCHITECTURES 0 CUDA_ARCH)
        set(SYCL_TARGET_ARCH "nvidia_gpu_sm_${CUDA_ARCH}")
        message(STATUS "OneAPI Backend: Auto-detected NVIDIA GPU target: ${SYCL_TARGET_ARCH}")
    else()
        # Default to NVIDIA sm_75 (Turing) or Intel CPU
        set(SYCL_TARGET_ARCH "nvidia_gpu_sm_75")
        message(STATUS "OneAPI Backend: Using default target: ${SYCL_TARGET_ARCH}")
        message(STATUS "  Set SYCL_TARGET_ARCH to override (e.g., nvidia_gpu_sm_80 for Ampere)")
    endif()
endif()

# Device selection (CPU, GPU, FPGA)
# Users can specify with -DTENZOR_ONEAPI_DEVICES="cpu;gpu"
if(NOT DEFINED TENZOR_ONEAPI_DEVICES)
    set(TENZOR_ONEAPI_DEVICES "cpu;gpu")
endif()

# Note: Device definitions would be added to compile flags in the custom commands above
# For now, we support both CPU and GPU by default via the SYCL runtime

# Installation
install(FILES ${ONEAPI_SO_OUTPUT}
    DESTINATION lib/tenzor/backends
)

message(STATUS "OneAPI Backend: Configured successfully")
message(STATUS "  SYCL Target: ${SYCL_TARGET_ARCH}")
message(STATUS "  Device Support: ${TENZOR_ONEAPI_DEVICES}")
if(MKL_FOUND)
    message(STATUS "  oneMKL: Enabled")
endif()
if(dnnl_FOUND)
    message(STATUS "  oneDNN: Enabled")
endif()
