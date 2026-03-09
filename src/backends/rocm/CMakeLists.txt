# ROCm Backend
# Requires ROCm 5.0 or higher for AMD GPUs
# Mirrors CUDA backend structure for consistency

# HIP language should already be enabled at project level

# Remove HIP-specific flags from CMAKE_CXX_FLAGS for this directory
# CMake's HIP language support pollutes CXX_FLAGS with HIP flags
string(REGEX REPLACE "-x hip" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REGEX REPLACE "--offload-arch=[^ ]+" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Find ROCm/HIP packages
find_package(hip QUIET CONFIG PATHS /opt/rocm)
find_package(rocblas QUIET CONFIG PATHS /opt/rocm)
find_package(rocsolver QUIET CONFIG PATHS /opt/rocm)
find_package(hiprand QUIET CONFIG PATHS /opt/rocm)
find_package(rocfft QUIET CONFIG PATHS /opt/rocm)
find_package(hipcub QUIET CONFIG PATHS /opt/rocm)
find_package(rocprim QUIET CONFIG PATHS /opt/rocm)
find_package(rocthrust QUIET CONFIG PATHS /opt/rocm)

if(NOT hip_FOUND)
    message(WARNING "ROCm/HIP not found. ROCm backend will not be built.")
    message(WARNING "Install ROCm from: https://rocm.docs.amd.com/")
    message(WARNING "Make sure ROCm is installed in /opt/rocm or set ROCM_PATH environment variable")
    return()
endif()

# Get HIP version
if(DEFINED HIP_VERSION)
    message(STATUS "ROCm Backend: Found HIP ${HIP_VERSION}")
else()
    message(STATUS "ROCm Backend: Found HIP")
endif()

# Separate C++ sources and HIP sources to avoid CMake flag conflicts
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
    kernels/reduction.hip.cpp
    kernels/matmul.hip.cpp
    kernels/normalization.hip.cpp
    kernels/nn_ops.hip.cpp
    kernels/lstm.hip.cpp
    kernels/gru.hip.cpp
    kernels/nms.hip.cpp
    kernels/roi_align.hip.cpp
    kernels/vision.hip.cpp
    kernels/sort.hip.cpp
)

# Conditionally add linalg source if rocSOLVER is found
if(rocsolver_FOUND)
    list(APPEND ROCM_HIP_SOURCES kernels/linalg.hip.cpp)
endif()

# Conditionally add FFT source if rocFFT is found
if(rocfft_FOUND)
    list(APPEND ROCM_HIP_SOURCES kernels/fft.hip.cpp)
endif()

# Mark .hip.cpp files to be compiled as HIP language
set_source_files_properties(
    ${ROCM_HIP_SOURCES}
    PROPERTIES LANGUAGE HIP
)

# Create HIP kernels library (static, internal use only)
add_library(tenzor_rocm_kernels STATIC ${ROCM_HIP_SOURCES})

# Set properties for HIP kernels library
# Note: HIP_SEPARABLE_COMPILATION OFF to avoid RDC linking issues
set_target_properties(tenzor_rocm_kernels PROPERTIES
    HIP_SEPARABLE_COMPILATION OFF
    # hipcc does not fully support C++23 — C++20 is the highest supported standard
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    LINKER_LANGUAGE HIP
)

# Include directories for HIP kernels
target_include_directories(tenzor_rocm_kernels PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${HIP_INCLUDE_DIRS}
    ${ROCM_PATH}/include
    /opt/rocm/include
)

# HIP-specific compile options
# Note: Removed -fgpu-rdc to avoid relocatable device code linking issues
target_compile_options(tenzor_rocm_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
    $<$<COMPILE_LANGUAGE:HIP>:-munsafe-fp-atomics>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:HIP>>:-O3>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:HIP>>:-g -ggdb>
)

# HIP-specific definitions
target_compile_definitions(tenzor_rocm_kernels PRIVATE
    __HIP_PLATFORM_AMD__
    __HIP_PLATFORM_HCC__
    TENZOR_ROCM_BACKEND
    TENZOR_INTERNAL
)

# Add hipRAND support to kernels library if available
if(hiprand_FOUND)
    target_compile_definitions(tenzor_rocm_kernels PRIVATE TENZOR_HAS_HIPRAND)
    target_link_libraries(tenzor_rocm_kernels PRIVATE hip::hiprand)
endif()

# Add rocFFT support to kernels library if available
if(rocfft_FOUND)
    target_compile_definitions(tenzor_rocm_kernels PRIVATE TENZOR_HAS_ROCFFT)
    target_link_libraries(tenzor_rocm_kernels PRIVATE roc::rocfft)
endif()

# Add rocSOLVER support to kernels library if available
if(rocsolver_FOUND)
    target_compile_definitions(tenzor_rocm_kernels PRIVATE TENZOR_HAS_ROCSOLVER)
    target_link_libraries(tenzor_rocm_kernels PRIVATE roc::rocsolver roc::rocblas)
endif()

# Add hipcub support (needed for sort, cumsum, cumprod, unique)
if(hipcub_FOUND)
    target_link_libraries(tenzor_rocm_kernels PRIVATE hip::hipcub)
    message(STATUS "ROCm Backend: hipcub support enabled")
endif()

# Add rocprim support (dependency of hipcub)
if(rocprim_FOUND)
    target_link_libraries(tenzor_rocm_kernels PRIVATE roc::rocprim)
    message(STATUS "ROCm Backend: rocprim support enabled")
endif()

# Add rocthrust support (needed for sort, unique)
if(rocthrust_FOUND)
    target_link_libraries(tenzor_rocm_kernels PRIVATE roc::rocthrust)
    message(STATUS "ROCm Backend: rocthrust support enabled")
endif()

# Manually compile rocm_backend.cpp to avoid HIP flag pollution
# CMake's HIP language support adds HIP flags to all CXX compilation in this directory
set(ROCM_BACKEND_OBJ "${CMAKE_CURRENT_BINARY_DIR}/rocm_backend.cpp.o")
add_custom_command(
    OUTPUT ${ROCM_BACKEND_OBJ}
    COMMAND ${CMAKE_CXX_COMPILER}
        -c ${CMAKE_CURRENT_SOURCE_DIR}/rocm_backend.cpp
        -o ${ROCM_BACKEND_OBJ}
        -std=c++20 -fPIC
        -O3 -DNDEBUG
        -Wall -Wextra -pedantic
        -march=native
        -Wno-unused-parameter -Wno-unused-variable
        -fopenmp
        -I${CMAKE_SOURCE_DIR}/include
        -isystem /opt/rocm/include
        -DTENZOR_HAS_MIOPEN
        -DTENZOR_HAS_ROCBLAS
        -DTENZOR_ROCM_BACKEND
        -DUSE_PROF_API=1
        -D__HIP_PLATFORM_AMD__=1
        -Dtenzor_backend_rocm_EXPORTS
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/rocm_backend.cpp
    COMMENT "Compiling rocm_backend.cpp (pure C++, no HIP flags)"
    VERBATIM
)

# Manually compile rocm_kernel_registry.cpp (also pure C++, no HIP flags)
# Build conditional define flags for the kernel registry
set(ROCM_REGISTRY_EXTRA_DEFS "")
if(rocfft_FOUND)
    list(APPEND ROCM_REGISTRY_EXTRA_DEFS -DTENZOR_HAS_ROCFFT)
endif()
if(rocsolver_FOUND)
    list(APPEND ROCM_REGISTRY_EXTRA_DEFS -DTENZOR_HAS_ROCSOLVER)
endif()

set(ROCM_KERNEL_REGISTRY_OBJ "${CMAKE_CURRENT_BINARY_DIR}/rocm_kernel_registry.cpp.o")
add_custom_command(
    OUTPUT ${ROCM_KERNEL_REGISTRY_OBJ}
    COMMAND ${CMAKE_CXX_COMPILER}
        -c ${CMAKE_CURRENT_SOURCE_DIR}/rocm_kernel_registry.cpp
        -o ${ROCM_KERNEL_REGISTRY_OBJ}
        -std=c++20 -fPIC
        -O3 -DNDEBUG
        -Wall -Wextra -pedantic
        -march=native
        -Wno-unused-parameter -Wno-unused-variable
        -fopenmp
        -I${CMAKE_SOURCE_DIR}/include
        -isystem /opt/rocm/include
        -DTENZOR_HAS_MIOPEN
        -DTENZOR_HAS_ROCBLAS
        -DTENZOR_ROCM_BACKEND
        -DUSE_PROF_API=1
        -D__HIP_PLATFORM_AMD__=1
        -Dtenzor_backend_rocm_EXPORTS
        ${ROCM_REGISTRY_EXTRA_DEFS}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/rocm_kernel_registry.cpp
    COMMENT "Compiling rocm_kernel_registry.cpp (pure C++, no HIP flags)"
    VERBATIM
)

# Create ROCm backend shared library using the manually compiled objects
add_library(tenzor_backend_rocm SHARED ${ROCM_BACKEND_OBJ} ${ROCM_KERNEL_REGISTRY_OBJ})

# Set target properties
set_target_properties(tenzor_backend_rocm PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    OUTPUT_NAME "tenzor_backend_rocm"
    PREFIX ""  # Remove 'lib' prefix on Unix
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    LINKER_LANGUAGE HIP
)

# Include directories
target_include_directories(tenzor_backend_rocm PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${HIP_INCLUDE_DIRS}
    ${ROCM_PATH}/include
    /opt/rocm/include
)

# Link against core library, HIP kernels, and HIP runtime
target_link_libraries(tenzor_backend_rocm PRIVATE
    tenzor_core
    tenzor_rocm_kernels
    hip::device
    hip::host
    $<TARGET_NAME_IF_EXISTS:hip::amdhip64>
)

# Link rocBLAS if available (equivalent to cuBLAS)
if(rocblas_FOUND)
    target_link_libraries(tenzor_backend_rocm PRIVATE roc::rocblas)
    target_compile_definitions(tenzor_backend_rocm PRIVATE TENZOR_HAS_ROCBLAS)
    message(STATUS "ROCm Backend: rocBLAS support enabled")
else()
    message(STATUS "ROCm Backend: rocBLAS not found")
endif()

# Link hipRAND if available (equivalent to cuRAND)
if(hiprand_FOUND)
    target_link_libraries(tenzor_backend_rocm PRIVATE hip::hiprand)
    target_compile_definitions(tenzor_backend_rocm PRIVATE TENZOR_HAS_HIPRAND)
    message(STATUS "ROCm Backend: hipRAND support enabled")
else()
    message(STATUS "ROCm Backend: hipRAND not found")
endif()

# Link rocFFT if available (equivalent to cuFFT)
if(rocfft_FOUND)
    target_link_libraries(tenzor_backend_rocm PRIVATE roc::rocfft)
    target_compile_definitions(tenzor_backend_rocm PRIVATE TENZOR_HAS_ROCFFT)
    message(STATUS "ROCm Backend: rocFFT support enabled")
else()
    message(STATUS "ROCm Backend: rocFFT not found")
endif()

# Link rocSOLVER if available (equivalent to cuSOLVER)
if(rocsolver_FOUND)
    target_link_libraries(tenzor_backend_rocm PRIVATE roc::rocsolver roc::rocblas)
    target_compile_definitions(tenzor_backend_rocm PRIVATE TENZOR_HAS_ROCSOLVER)
    message(STATUS "ROCm Backend: rocSOLVER support enabled")
else()
    message(STATUS "ROCm Backend: rocSOLVER not found")
endif()

# Find MIOpen (ROCm's DNN library - equivalent to cuDNN)
find_package(MIOpen QUIET CONFIG PATHS /opt/rocm)
if(MIOpen_FOUND)
    target_include_directories(tenzor_backend_rocm PRIVATE ${MIOpen_INCLUDE_DIRS})
    target_link_libraries(tenzor_backend_rocm PRIVATE MIOpen)
    target_compile_definitions(tenzor_backend_rocm PRIVATE TENZOR_HAS_MIOPEN)
    message(STATUS "ROCm Backend: MIOpen support enabled")
else()
    message(STATUS "ROCm Backend: MIOpen not found, using built-in convolution kernels")
endif()

# C++ backend definitions (HIP definitions are on the kernels library)
target_compile_definitions(tenzor_backend_rocm PRIVATE
    TENZOR_ROCM_BACKEND
    TENZOR_INTERNAL
)

# Override compile options to force C++ compilation (override any -x hip flags)
# This is necessary because CMake's HIP language support adds HIP flags globally
set_source_files_properties(rocm_backend.cpp PROPERTIES
    COMPILE_FLAGS "-x c++"
)

# Installation rules
install(TARGETS tenzor_backend_rocm
    LIBRARY DESTINATION lib/tenzor/backends
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib/tenzor/backends
)

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
message(STATUS "  Sources:")
message(STATUS "    - rocm_caching_allocator.hip.cpp (HIP-optimized memory allocator)")
message(STATUS "  Kernel Sources:")
message(STATUS "    - math.hip.cpp (element-wise operations, broadcasting)")
message(STATUS "    - activations.hip.cpp (ReLU, sigmoid, tanh, GELU, etc.)")
message(STATUS "    - batchnorm.hip.cpp (BatchNorm, LayerNorm, GroupNorm)")
message(STATUS "    - conv2d.hip.cpp (convolution with im2col/col2im)")
message(STATUS "    - pooling.hip.cpp (MaxPool, AvgPool, adaptive)")
message(STATUS "    - indexing.hip.cpp (gather, scatter, masked ops)")
message(STATUS "    - transform.hip.cpp (reshape, transpose, permute, cat)")
message(STATUS "    - fused_ops.hip.cpp (fused operations for optimization)")
message(STATUS "    - reduction.hip.cpp (sum, mean, max, min reductions)")
message(STATUS "    - matmul.hip.cpp (matrix multiplication with rocBLAS)")
message(STATUS "    - lstm.hip.cpp (LSTM cell forward/backward)")
message(STATUS "    - gru.hip.cpp (GRU cell forward/backward)")
message(STATUS "    - nms.hip.cpp (Non-Maximum Suppression for object detection)")
message(STATUS "    - roi_align.hip.cpp (ROI Align for region-based detection)")
message(STATUS "  Note: GEMM operations use rocBLAS library for optimal performance")
if(rocblas_FOUND)
    message(STATUS "  rocBLAS: Enabled")
else()
    message(STATUS "  rocBLAS: Not found")
endif()
if(hiprand_FOUND)
    message(STATUS "  hipRAND: Enabled")
else()
    message(STATUS "  hipRAND: Not found")
endif()
if(MIOpen_FOUND)
    if(DEFINED MIOpen_VERSION)
        message(STATUS "  MIOpen: ${MIOpen_VERSION}")
    else()
        message(STATUS "  MIOpen: Enabled")
    endif()
else()
    message(STATUS "  MIOpen: Not found")
endif()
if(rocsolver_FOUND)
    message(STATUS "  rocSOLVER: Enabled")
else()
    message(STATUS "  rocSOLVER: Not found")
endif()
if(rocfft_FOUND)
    message(STATUS "  rocFFT: Enabled")
else()
    message(STATUS "  rocFFT: Not found")
endif()
if(hipcub_FOUND)
    message(STATUS "  hipcub: Enabled")
else()
    message(STATUS "  hipcub: Not found")
endif()
if(rocthrust_FOUND)
    message(STATUS "  rocthrust: Enabled")
else()
    message(STATUS "  rocthrust: Not found")
endif()
message(STATUS "  NOTE: To compile for your specific GPU, set CMAKE_HIP_ARCHITECTURES")
message(STATUS "        Example: cmake -DCMAKE_HIP_ARCHITECTURES=\"gfx906;gfx908\"")
