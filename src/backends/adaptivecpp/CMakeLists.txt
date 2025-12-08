# AdaptiveCpp Backend (formerly hipSYCL)
# Supports multiple backends: CUDA, HIP/ROCm, OpenMP (CPU), Level Zero (Intel GPUs)
# https://github.com/AdaptiveCpp/AdaptiveCpp

# Find AdaptiveCpp compiler
find_program(ACPP_COMPILER NAMES acpp syclcc hipsycl QUIET)

if(NOT ACPP_COMPILER)
    message(WARNING "AdaptiveCpp compiler not found. AdaptiveCpp backend will not be built.")
    message(WARNING "Install from: https://github.com/AdaptiveCpp/AdaptiveCpp")
    return()
endif()

message(STATUS "AdaptiveCpp Backend: Found compiler at ${ACPP_COMPILER}")

# Detect available targets
# Common targets: cuda:sm_XX, hip:gfxXXX, omp (OpenMP CPU), generic (SPIR-V)
if(NOT DEFINED ACPP_TARGETS)
    # AdaptiveCpp requires specific target format for multi-backend compilation
    # For SSCP (generic single-source) compilation, use "generic" target
    # For specific backends, use only ONE backend type per compilation unit
    #
    # Valid formats:
    #   - "generic" : SSCP compilation (portable, JIT-compiled at runtime)
    #   - "omp"     : OpenMP CPU backend only
    #   - "cuda:sm_XX" : CUDA only (e.g., cuda:sm_75)
    #   - "hip:gfxXXX" : HIP only (e.g., hip:gfx90c)
    #
    # Note: Mixing backend types (cuda + omp + hip) requires SSCP "generic" target
    # or separate compilation units for each backend.

    # Check clang version and CUDA compatibility
    # Clang 19 has incompatibility with CUDA 13.0+ (missing texture_fetch_functions.h)
    execute_process(
        COMMAND ${ACPP_COMPILER} --version
        OUTPUT_VARIABLE ACPP_VERSION_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    string(REGEX MATCH "clang version ([0-9]+)" CLANG_VERSION_MATCH "${ACPP_VERSION_OUTPUT}")
    if(CMAKE_MATCH_1)
        set(ACPP_CLANG_MAJOR ${CMAKE_MATCH_1})
    else()
        set(ACPP_CLANG_MAJOR 0)
    endif()

    # Check system CUDA version
    find_program(NVCC_COMPILER nvcc QUIET)
    set(CUDA_MAJOR 0)
    if(NVCC_COMPILER)
        execute_process(
            COMMAND ${NVCC_COMPILER} --version
            OUTPUT_VARIABLE NVCC_VERSION_OUTPUT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        string(REGEX MATCH "release ([0-9]+)\\.([0-9]+)" CUDA_VERSION_MATCH "${NVCC_VERSION_OUTPUT}")
        if(CMAKE_MATCH_1)
            set(CUDA_MAJOR ${CMAKE_MATCH_1})
        endif()
    endif()

    # Check for CUDA 12.6 installation (workaround for Clang 19 + CUDA 13 incompatibility)
    set(ACPP_CUDA_PATH "")
    if(EXISTS "/opt/cuda-12.6/bin/nvcc")
        set(ACPP_CUDA_PATH "/opt/cuda-12.6")
        message(STATUS "AdaptiveCpp Backend: Found CUDA 12.6 at ${ACPP_CUDA_PATH}")
    endif()

    # Default to OpenMP for best compatibility
    set(ACPP_TARGETS "omp")

    # Auto-detect CUDA if available and compatible
    if(NVCC_COMPILER AND NOT ACPP_USE_GENERIC)
        # Check for clang/CUDA incompatibility
        # Clang 19+ has issues with CUDA 13.0+ due to removed texture_fetch_functions.h
        if(ACPP_CLANG_MAJOR GREATER_EQUAL 19 AND CUDA_MAJOR GREATER_EQUAL 13)
            if(ACPP_CUDA_PATH)
                # Use CUDA 12.6 as workaround
                message(STATUS "AdaptiveCpp Backend: Clang ${ACPP_CLANG_MAJOR} incompatible with system CUDA ${CUDA_MAJOR}")
                message(STATUS "AdaptiveCpp Backend: Using CUDA 12.6 from ${ACPP_CUDA_PATH}")
                # Try to detect GPU architecture
                execute_process(
                    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
                    OUTPUT_VARIABLE GPU_COMPUTE_CAP
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET
                    RESULT_VARIABLE NVIDIA_SMI_RESULT
                )
                if(NVIDIA_SMI_RESULT EQUAL 0 AND GPU_COMPUTE_CAP)
                    string(REPLACE "." "" GPU_SM "${GPU_COMPUTE_CAP}")
                    string(REGEX REPLACE "\n.*" "" GPU_SM "${GPU_SM}")  # Take first GPU
                    set(ACPP_TARGETS "cuda:sm_${GPU_SM}")
                    message(STATUS "AdaptiveCpp Backend: Auto-detected CUDA sm_${GPU_SM}")
                endif()
            else()
                message(STATUS "AdaptiveCpp Backend: Clang ${ACPP_CLANG_MAJOR} incompatible with CUDA ${CUDA_MAJOR}")
                message(STATUS "AdaptiveCpp Backend: Using OpenMP target (install CUDA 12.x to /opt/cuda-12.6 for CUDA support)")
            endif()
        else()
            # Try to detect GPU architecture
            execute_process(
                COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
                OUTPUT_VARIABLE GPU_COMPUTE_CAP
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                RESULT_VARIABLE NVIDIA_SMI_RESULT
            )
            if(NVIDIA_SMI_RESULT EQUAL 0 AND GPU_COMPUTE_CAP)
                string(REPLACE "." "" GPU_SM "${GPU_COMPUTE_CAP}")
                string(REGEX REPLACE "\n.*" "" GPU_SM "${GPU_SM}")  # Take first GPU
                # Use CUDA-specific target for best performance
                set(ACPP_TARGETS "cuda:sm_${GPU_SM}")
                message(STATUS "AdaptiveCpp Backend: Auto-detected CUDA sm_${GPU_SM}")
            endif()
        endif()
    endif()
endif()

message(STATUS "AdaptiveCpp Backend: Target architectures: ${ACPP_TARGETS}")

# Convert target list to acpp format
string(REPLACE ";" "," ACPP_TARGETS_STR "${ACPP_TARGETS}")

# Source files
set(ADAPTIVECPP_SYCL_SOURCES
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

set(ADAPTIVECPP_BACKEND_SOURCE
    adaptivecpp_backend.cpp
)

# Set CUDA path flag if using alternate CUDA installation
set(ACPP_CUDA_PATH_FLAG "")
if(ACPP_CUDA_PATH AND ACPP_TARGETS MATCHES "cuda")
    set(ACPP_CUDA_PATH_FLAG "--cuda-path=${ACPP_CUDA_PATH}")
endif()

# Build SYCL kernel objects
set(SYCL_OBJECTS "")
foreach(SRC ${ADAPTIVECPP_SYCL_SOURCES})
    get_filename_component(OBJ_NAME ${SRC} NAME_WE)
    set(OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/${OBJ_NAME}.o")
    list(APPEND SYCL_OBJECTS ${OBJ_FILE})

    # Coverage flags (AdaptiveCpp supports coverage for omp target)
    set(COVERAGE_COMPILE_FLAGS "")
    if(TENZOR_ENABLE_COVERAGE AND ACPP_TARGETS MATCHES "omp")
        set(COVERAGE_COMPILE_FLAGS "--coverage")
    endif()

    add_custom_command(
        OUTPUT ${OBJ_FILE}
        COMMAND ${ACPP_COMPILER}
            -c ${CMAKE_CURRENT_SOURCE_DIR}/${SRC}
            -o ${OBJ_FILE}
            -std=c++23 -fPIC
            --acpp-targets=${ACPP_TARGETS_STR}
            ${ACPP_CUDA_PATH_FLAG}
            -O3
            -I${CMAKE_SOURCE_DIR}/include
            -DTENZOR_ADAPTIVECPP_BACKEND
            ${COVERAGE_COMPILE_FLAGS}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SRC}
        COMMENT "Compiling AdaptiveCpp kernel: ${SRC}"
        VERBATIM
    )
endforeach()

# Create static library from SYCL objects
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libtenzor_adaptivecpp_kernels.a
    COMMAND ${CMAKE_AR} rcs ${CMAKE_CURRENT_BINARY_DIR}/libtenzor_adaptivecpp_kernels.a ${SYCL_OBJECTS}
    DEPENDS ${SYCL_OBJECTS}
    COMMENT "Creating AdaptiveCpp kernels static library"
    VERBATIM
)

# Compile backend source with acpp to ensure SYCL namespace compatibility
set(BACKEND_OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/adaptivecpp_backend.o")

# Coverage flags for backend
set(BACKEND_COVERAGE_FLAGS "")
if(TENZOR_ENABLE_COVERAGE AND ACPP_TARGETS MATCHES "omp")
    set(BACKEND_COVERAGE_FLAGS "--coverage")
endif()

add_custom_command(
    OUTPUT ${BACKEND_OBJ_FILE}
    COMMAND ${ACPP_COMPILER}
        -c ${CMAKE_CURRENT_SOURCE_DIR}/adaptivecpp_backend.cpp
        -o ${BACKEND_OBJ_FILE}
        -std=c++23 -fPIC
        --acpp-targets=${ACPP_TARGETS_STR}
        ${ACPP_CUDA_PATH_FLAG}
        -O3
        -I${CMAKE_SOURCE_DIR}/include
        -DTENZOR_ADAPTIVECPP_BACKEND
        ${BACKEND_COVERAGE_FLAGS}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/adaptivecpp_backend.cpp
    COMMENT "Compiling AdaptiveCpp backend source with acpp"
    VERBATIM
)

# Create custom target for backend object
add_custom_target(tenzor_adaptivecpp_backend_obj
    DEPENDS ${BACKEND_OBJ_FILE}
)

# Add dependency on kernels
add_custom_target(tenzor_adaptivecpp_kernels_lib
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtenzor_adaptivecpp_kernels.a
)

# Create the shared library using acpp linker
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/../bin/tenzor_backend_adaptivecpp.so
    COMMAND ${ACPP_COMPILER}
        -shared -fPIC
        -o ${CMAKE_BINARY_DIR}/../bin/tenzor_backend_adaptivecpp.so
        ${BACKEND_OBJ_FILE}
        ${CMAKE_CURRENT_BINARY_DIR}/libtenzor_adaptivecpp_kernels.a
        --acpp-targets=${ACPP_TARGETS_STR}
        ${ACPP_CUDA_PATH_FLAG}
        -L${CMAKE_BINARY_DIR}/../bin -ltenzor_core
        -Wl,-rpath,'$$ORIGIN'
        ${BACKEND_COVERAGE_FLAGS}
    DEPENDS ${BACKEND_OBJ_FILE} ${CMAKE_CURRENT_BINARY_DIR}/libtenzor_adaptivecpp_kernels.a tenzor_core
    COMMENT "Linking AdaptiveCpp backend shared library"
    VERBATIM
)

# Create the main target
add_custom_target(tenzor_backend_adaptivecpp ALL
    DEPENDS ${CMAKE_BINARY_DIR}/../bin/tenzor_backend_adaptivecpp.so
)
add_dependencies(tenzor_backend_adaptivecpp tenzor_adaptivecpp_backend_obj tenzor_adaptivecpp_kernels_lib tenzor_core)

# Optional library support - these are handled in the link command above
# For now, we'll detect and report but not add to link (custom targets don't support target_link_libraries)
find_package(dnnl QUIET)
if(dnnl_FOUND)
    message(STATUS "AdaptiveCpp Backend: oneDNN available (manual linking may be needed)")
endif()

if(ACPP_TARGETS MATCHES "cuda")
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "AdaptiveCpp Backend: cuBLAS available")
    endif()
endif()

if(ACPP_TARGETS MATCHES "hip")
    find_package(rocblas QUIET)
    if(rocblas_FOUND)
        message(STATUS "AdaptiveCpp Backend: rocBLAS available")
    endif()
endif()

# Installation
install(FILES ${CMAKE_BINARY_DIR}/../bin/tenzor_backend_adaptivecpp.so
    DESTINATION lib/tenzor/backends
    OPTIONAL
)

message(STATUS "AdaptiveCpp Backend: Configured successfully")
message(STATUS "  Targets: ${ACPP_TARGETS}")
if(dnnl_FOUND)
    message(STATUS "  oneDNN: Enabled")
endif()
