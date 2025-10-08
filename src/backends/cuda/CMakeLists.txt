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
    kernels/math.cu
    kernels/reduction.cu
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
    CUDA_STANDARD 17
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
)

# Link cuBLAS if available
if(TARGET CUDA::cublas)
    target_link_libraries(tenzor_backend_cuda PRIVATE CUDA::cublas)
    target_compile_definitions(tenzor_backend_cuda PRIVATE TENZOR_HAS_CUBLAS)
    message(STATUS "CUDA Backend: cuBLAS support enabled")
endif()

# Link cuDNN if available
find_package(CUDNN QUIET)
if(CUDNN_FOUND)
    target_include_directories(tenzor_backend_cuda PRIVATE ${CUDNN_INCLUDE_DIRS})
    target_link_libraries(tenzor_backend_cuda PRIVATE ${CUDNN_LIBRARIES})
    target_compile_definitions(tenzor_backend_cuda PRIVATE TENZOR_HAS_CUDNN)
    message(STATUS "CUDA Backend: cuDNN support enabled")
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
if(TARGET CUDA::cublas)
    message(STATUS "  cuBLAS: Enabled")
else()
    message(STATUS "  cuBLAS: Not found")
endif()
if(CUDNN_FOUND)
    message(STATUS "  cuDNN: ${CUDNN_VERSION}")
endif()
