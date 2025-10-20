# CUDA Backend
# Requires CUDA Toolkit 12.0 or higher

# Enable CUDA language
enable_language(CUDA)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA Toolkit not found. Please install CUDA 12.0 or higher.")
endif()

message(STATUS "CUDA Backend: Found CUDA ${CUDAToolkit_VERSION}")

set(CUDA_BACKEND_SOURCES
    cuda_backend.cpp
    ../../backend/caching_allocator.cpp
    kernels/math.cu
    kernels/matmul.cu
    kernels/reduction.cu
    kernels/activations.cu
    kernels/transform.cu
    kernels/batchnorm.cu
    kernels/conv2d.cu
    kernels/lstm.cu
    kernels/quantization/quantized_linear.cu
    kernels/quantization/quantized_conv2d.cu
    kernels/nms.cu
    kernels/roi_align.cu
    cudnn_ops.cu
)

# Create CUDA backend shared library
add_library(tenzor_backend_cuda SHARED ${CUDA_BACKEND_SOURCES})

# Set CUDA properties
set_target_properties(tenzor_backend_cuda PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    OUTPUT_NAME "tenzor_backend_cuda"
    PREFIX ""  # Remove 'lib' prefix on Unix
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)

# Include directories
target_include_directories(tenzor_backend_cuda PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link against core library and CUDA libraries
target_link_libraries(tenzor_backend_cuda PRIVATE
    tenzor_core
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::curand
)

# Link cuBLAS if available
if(TARGET CUDA::cublas)
    target_link_libraries(tenzor_backend_cuda PRIVATE CUDA::cublas)
    target_compile_definitions(tenzor_backend_cuda PRIVATE TENZOR_HAS_CUBLAS)
    message(STATUS "CUDA Backend: cuBLAS support enabled")
endif()

# Link cuDNN if available
# First try to find cuDNN using find_package
find_package(CUDNN QUIET)

# If not found, try manual detection
if(NOT CUDNN_FOUND)
    # Try to find cuDNN in CUDA Toolkit directories
    find_path(CUDNN_INCLUDE_DIR cudnn.h
        HINTS
        ${CUDAToolkit_INCLUDE_DIRS}
        /usr/local/cuda/include
        /usr/include
        $ENV{CUDA_PATH}/include
    )

    find_library(CUDNN_LIBRARY cudnn
        HINTS
        ${CUDAToolkit_LIBRARY_DIR}
        /usr/local/cuda/lib64
        /usr/lib/x86_64-linux-gnu
        $ENV{CUDA_PATH}/lib64
    )

    if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
        set(CUDNN_FOUND TRUE)
        set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_DIR})
        set(CUDNN_LIBRARIES ${CUDNN_LIBRARY})

        # Try to extract version
        if(EXISTS "${CUDNN_INCLUDE_DIR}/cudnn_version.h")
            file(READ "${CUDNN_INCLUDE_DIR}/cudnn_version.h" CUDNN_VERSION_CONTENT)
            string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)" _ "${CUDNN_VERSION_CONTENT}")
            set(CUDNN_MAJOR_VERSION ${CMAKE_MATCH_1})
            string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)" _ "${CUDNN_VERSION_CONTENT}")
            set(CUDNN_MINOR_VERSION ${CMAKE_MATCH_1})
            string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)" _ "${CUDNN_VERSION_CONTENT}")
            set(CUDNN_PATCH_VERSION ${CMAKE_MATCH_1})
            set(CUDNN_VERSION "${CUDNN_MAJOR_VERSION}.${CUDNN_MINOR_VERSION}.${CUDNN_PATCH_VERSION}")
        endif()
    endif()
endif()

if(CUDNN_FOUND)
    target_include_directories(tenzor_backend_cuda PRIVATE ${CUDNN_INCLUDE_DIRS})
    target_link_libraries(tenzor_backend_cuda PRIVATE ${CUDNN_LIBRARIES})
    target_compile_definitions(tenzor_backend_cuda PRIVATE TENZOR_HAS_CUDNN)
    message(STATUS "CUDA Backend: cuDNN support enabled (${CUDNN_VERSION})")
    message(STATUS "  cuDNN Include: ${CUDNN_INCLUDE_DIRS}")
    message(STATUS "  cuDNN Library: ${CUDNN_LIBRARIES}")
else()
    message(STATUS "CUDA Backend: cuDNN not found, using built-in convolution kernels")
endif()

# CUDA architecture flags
# Detect GPU architecture or use common ones
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Set common GPU architectures
    # 70 = Volta (V100), 75 = Turing (RTX 20xx), 80 = Ampere (A100, RTX 30xx),
    # 86 = Ampere (RTX 3090), 89 = Ada Lovelace (RTX 40xx), 90 = Hopper (H100)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
    message(STATUS "CUDA Backend: Building for architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "CUDA Backend: Using user-specified architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# CUDA compiler options
target_compile_options(tenzor_backend_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        --use_fast_math
        --generate-line-info
    >
)

# Set compute capability definitions
foreach(ARCH ${CMAKE_CUDA_ARCHITECTURES})
    target_compile_definitions(tenzor_backend_cuda PRIVATE
        CUDA_ARCH_${ARCH}
    )
endforeach()

# Enable Tensor Core support for architectures >= 70 (Volta+)
set(HAS_TENSOR_CORES FALSE)
foreach(ARCH ${CMAKE_CUDA_ARCHITECTURES})
    if(ARCH GREATER_EQUAL 70)
        set(HAS_TENSOR_CORES TRUE)
        break()
    endif()
endforeach()

if(HAS_TENSOR_CORES)
    target_compile_definitions(tenzor_backend_cuda PRIVATE TENZOR_HAS_TENSOR_CORES)
    message(STATUS "CUDA Backend: Tensor Core support enabled (compute capability >= 7.0)")
endif()

# Optimization flags
if(CMAKE_BUILD_TYPE MATCHES "Release|RelWithDebInfo")
    target_compile_options(tenzor_backend_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )
endif()

# Debug info for debugging builds
if(CMAKE_BUILD_TYPE MATCHES "Debug")
    target_compile_options(tenzor_backend_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G>  # Generate debug info for cuda-gdb
    )
endif()

# Installation
install(TARGETS tenzor_backend_cuda
    LIBRARY DESTINATION lib/tenzor/backends
    RUNTIME DESTINATION bin
)

message(STATUS "CUDA Backend: Configured successfully")
message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
if(HAS_TENSOR_CORES)
    message(STATUS "  Tensor Cores: Enabled")
endif()
if(TARGET CUDA::cublas)
    message(STATUS "  cuBLAS: Enabled")
else()
    message(STATUS "  cuBLAS: Not found")
endif()
if(TARGET CUDA::curand)
    message(STATUS "  cuRAND: Enabled")
endif()
if(CUDNN_FOUND)
    message(STATUS "  cuDNN: ${CUDNN_VERSION}")
endif()
