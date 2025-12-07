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
    kernels/creation.cpp
    kernels/batchnorm.cpp
    kernels/conv2d.cpp
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
)

# Separate SYCL kernel sources from C++ backend source
# Similar to ROCm's 2-stage approach
set(ONEAPI_SYCL_SOURCES
    kernels/math.cpp
    kernels/activations.cpp
    kernels/reduction.cpp
    kernels/transform.cpp
    kernels/creation.cpp
    kernels/batchnorm.cpp
    kernels/conv2d.cpp
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
)

set(ONEAPI_BACKEND_SOURCE
    oneapi_backend.cpp
)

# Stage 1: Compile SYCL kernels into static library using icpx
# This is built with icpx via custom commands
#set(ICPX_COMPILER "${INTEL_SYCL_COMPILER}")
find_program(ICPX_COMPILER NAMES icpx REQUIRED)

# Detect SYCL target architecture BEFORE building kernels
# Target CPU for better compatibility
if(NOT DEFINED SYCL_TARGET_ARCH)
    # Use spir64 for CPU and Intel GPUs (better compatibility)
    set(SYCL_TARGET_ARCH "spir64")
    message(STATUS "OneAPI Backend: Target ${SYCL_TARGET_ARCH} (CPU/Intel GPU)")
endif()

# Build each SYCL kernel file individually
set(SYCL_OBJECTS "")
foreach(SRC ${ONEAPI_SYCL_SOURCES})
    get_filename_component(OBJ_NAME ${SRC} NAME_WE)
    set(OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/${OBJ_NAME}.o")
    list(APPEND SYCL_OBJECTS ${OBJ_FILE})

    # Prepare MKL compile flags
    set(MKL_COMPILE_FLAGS "")
    if(MKL_FOUND)
        set(MKL_COMPILE_FLAGS "-DTENZOR_HAS_ONEMKL" "-I/opt/intel/oneapi/2025.2/include")
    endif()

    # Note: Coverage is NOT enabled for OneAPI SYCL kernels
    # icpx doesn't support coverage instrumentation for SYCL/spir64 targets
    set(COVERAGE_COMPILE_FLAGS "")

    add_custom_command(
        OUTPUT ${OBJ_FILE}
        COMMAND ${ICPX_COMPILER}
            -c ${CMAKE_CURRENT_SOURCE_DIR}/${SRC}
            -o ${OBJ_FILE}
            -std=c++23 -fPIC
            -fsycl -fsycl-targets=${SYCL_TARGET_ARCH}
            -O3
            -I${CMAKE_SOURCE_DIR}/include
            ${MKL_COMPILE_FLAGS}
            ${COVERAGE_COMPILE_FLAGS}
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

# Stage 2: Create shared library using proper CMake target
# This allows us to use target_link_options for SYCL flags

# Override CMAKE link command to use icpx instead of g++ for shared library creation
# This is necessary because SYCL linking requires icpx
set(CMAKE_CXX_CREATE_SHARED_LIBRARY
    "${ICPX_COMPILER} <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>"
)

# Create the shared library target
add_library(tenzor_backend_oneapi SHARED
    ${ONEAPI_BACKEND_SOURCE}
)

# Add dependency on SYCL kernels static library
add_custom_target(tenzor_oneapi_kernels_lib
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtenzor_oneapi_kernels.a
)
add_dependencies(tenzor_backend_oneapi tenzor_oneapi_kernels_lib)

# Set output location
set_target_properties(tenzor_backend_oneapi PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin
    OUTPUT_NAME tenzor_backend_oneapi
    PREFIX ""
)

# Compile options for backend source
target_compile_options(tenzor_backend_oneapi PRIVATE
    -std=c++23
    -O3 -DNDEBUG
    -Wall -Wextra -pedantic
    -march=native
    -Wno-unused-parameter -Wno-unused-variable
    -fopenmp
    -DTENZOR_ONEAPI_BACKEND
)

# Note: Coverage is NOT enabled for OneAPI backend
# icpx doesn't support coverage instrumentation for SYCL/spir64 targets
# This prevents undefined symbol errors (__gcov_merge_add) when loading the backend

# Add MKL definitions if available
if(MKL_FOUND)
    target_compile_definitions(tenzor_backend_oneapi PRIVATE TENZOR_HAS_ONEMKL)
    target_include_directories(tenzor_backend_oneapi PRIVATE /opt/intel/oneapi/2025.2/include)
endif()

# Include directories
target_include_directories(tenzor_backend_oneapi PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${SYCL_INCLUDE_DIR}
)

# Link with SYCL kernels static library
target_link_libraries(tenzor_backend_oneapi PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/libtenzor_oneapi_kernels.a
    tenzor_core
    ${SYCL_LIBRARY_DIR}/libsycl.so
)

# Link with gcov if coverage is enabled (tenzor_core was built with coverage)
if(TENZOR_ENABLE_COVERAGE)
    target_link_libraries(tenzor_backend_oneapi PRIVATE gcov)
endif()

# Add SYCL link options (THIS IS WHERE THE MAGIC HAPPENS)
target_link_options(tenzor_backend_oneapi PRIVATE
    -fsycl
    -fsycl-targets=${SYCL_TARGET_ARCH}
)

# Note: Coverage linking is NOT enabled for OneAPI backend
# icpx doesn't support coverage for SYCL targets, which would cause
# undefined symbol errors (__gcov_merge_add, etc.) when loading the backend

# Add MKL link options if available
if(MKL_FOUND)
    target_link_options(tenzor_backend_oneapi PRIVATE
        -L/opt/intel/oneapi/2025.2/lib
        -lmkl_sycl -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5
    )
endif()

# Find oneDNN (Intel's DNN library)
find_package(dnnl QUIET)
if(dnnl_FOUND)
    message(STATUS "OneAPI Backend: oneDNN support enabled")
else()
    message(STATUS "OneAPI Backend: oneDNN not found, using built-in kernels")
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
