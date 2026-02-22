# Vulkan backend for Tenzor
cmake_minimum_required(VERSION 3.15)

# Find Vulkan SDK
find_package(Vulkan REQUIRED)

if(NOT Vulkan_FOUND)
    message(WARNING "Vulkan SDK not found. Vulkan backend will not be built.")
    return()
endif()

message(STATUS "Building Vulkan backend")
message(STATUS "Vulkan Include Dir: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "Vulkan Library: ${Vulkan_LIBRARIES}")

# Find glslc shader compiler (part of Vulkan SDK)
find_program(GLSLC glslc HINTS
    ${Vulkan_INCLUDE_DIRS}/../bin
    $ENV{VULKAN_SDK}/bin
    /usr/bin
    /usr/local/bin
)

if(NOT GLSLC)
    message(WARNING "glslc shader compiler not found. Shaders will not be compiled.")
    message(WARNING "Please install Vulkan SDK or set VULKAN_SDK environment variable.")
else()
    message(STATUS "Found glslc: ${GLSLC}")
endif()

# Compile GLSL shaders to SPIR-V
set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kernels)
set(SHADER_OUTPUT_DIR ${CMAKE_BINARY_DIR}/shaders/vulkan)

file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

# List of shaders
set(SHADERS
    # Basic operations
    math
    math_f64
    math_f16
    math_i8
    math_i32
    math_broadcast
    math_broadcast_f64
    math_broadcast_f16
    math_broadcast_i8
    math_broadcast_bool
    comparison
    comparison_bool
    comparison_f64
    comparison_i32
    comparison_i64
    matmul
    matmul_bt
    matmul_i32
    bmm
    bmm_f64
    bmm_strided
    bmm_strided_f64
    activations
    activations_f64
    reduction
    reduction_f64
    reduction_i32
    conv2d
    pooling
    batchnorm
    transform
    indexing
    # Math functions
    trigonometric
    trigonometric_f64
    hyperbolic
    hyperbolic_f64
    # Vision operations
    im2col
    col2im
    # Advanced pooling
    pooling_backward
    pooling_forward_with_indices
    adaptive_pooling
    adaptive_pooling_f64
    adaptive_pooling_f16
    adaptive_avg_pool2d_backward
    adaptive_avg_pool2d_backward_f64
    adaptive_avg_pool2d_backward_f16
    # Normalization
    batchnorm_backward
    layer_norm
    group_norm
    # Softmax and losses
    softmax
    log_softmax
    cross_entropy
    # Advanced reductions
    argmax_argmin
    argmax_argmin_i32
    variance_std
    variance_std_f64
    prod_reduction
    prod_reduction_i32
    boolean_reduction
    boolean_reduction_f64
    # Indexing operations
    embedding
    embedding_f16
    embedding_f64
    gather
    gather_f64
    gather_f16
    scatter
    scatter_f64
    scatter_f16
    index_select
    # Shape operations (NEW - 9 critical operations)
    fill
    transpose
    permute
    # Backward activation functions
    activations_backward
    activations_backward_f64
    activations_backward_f16
    softmax_backward
    softmax_backward_f64
    softmax_backward_f16
    log_softmax_backward
    log_softmax_backward_f64
    swish_backward
    swish_backward_f64
    # Conv2d backward operations
    conv2d_backward_input
    conv2d_backward_weight
    conv2d_backward_bias
    conv2d_backward_input_f64
    conv2d_backward_weight_f64
    conv2d_backward_bias_f64
    # Tensor manipulation operations
    expand
    expand_f64
    expand_f16
    cat
    cat_f64
    cat_f16
    clamp
    clamp_f64
    clamp_f16
    # BatchNorm2d operations
    batchnorm2d_forward
    batchnorm2d_backward
    batchnorm2d_mean_var
    batchnorm2d_mean_var_f64
    # Pooling operations (new)
    avg_pool2d
    max_pool2d
    avg_pool2d_backward
    max_pool2d_backward
    # Final 3 operations for 100% coverage
    conv2d_forward
    conv2d_forward_f64
    conv2d_forward_f16
    conv_transpose2d_forward
    full
    full_f64
    full_f16
    full_i8
    ones
    ones_f64
    ones_f16
    ones_i8
    ones_uint8
    ones_i64
    full_uint8
    full_i64
    # UInt8 and Int64 math operations
    math_uint8
    math_broadcast_uint8
    math_i64
    math_broadcast_i64
    # Float16 transformer operations
    matmul_f16
    softmax_f16
    layer_norm_f16
    activations_f16
    reduction_f16
    # Float64 transformer operations
    matmul_f64
    matmul_f64_bt
    matmul_f64_optimized
    softmax_f64
    layer_norm_f64
    log_softmax_f64
    # Vision transformer operations
    gather_relative_position_bias
    gather_relative_position_bias_f64
    gather_relative_position_bias_f16
    # GPU-optimized operations (replacing CPU fallbacks)
    strided_copy
    strided_copy_f64
    strided_copy_f16
    strided_copy_u8
    random
    random_f64
    random_f16
    max_pool2d_backward_indices
    max_pool2d_backward_indices_f64
    max_pool2d_backward_indices_f16
    batchnorm_update_stats
    batchnorm_update_stats_f64
    batchnorm_update_stats_f16
    # Masked select support (prefix sum)
    masked_select_count
    masked_select_gather
    prefix_sum
    # Float64 pooling operations
    max_pool2d_f64
    avg_pool2d_f64
    avg_pool2d_backward_f64
    # Float64 batchnorm operations
    batchnorm2d_forward_f64
    batchnorm2d_backward_f64
    # Interpolation operations
    bilinear_interpolate
    bilinear_interpolate_f64
    bilinear_interpolate_f16
    nearest_interpolate
    nearest_interpolate_f64
    nearest_interpolate_f16
    # F16 math function shaders
    trigonometric_f16
    hyperbolic_f16
    comparison_f16
    # F16 pooling shaders
    max_pool2d_f16
    avg_pool2d_f16
    avg_pool2d_backward_f16
    # F16 batchnorm shaders
    batchnorm2d_forward_f16
    batchnorm2d_backward_f16
    batchnorm2d_mean_var_f16
    # Multi-dtype argmax/argmin
    argmax_argmin_f16
    argmax_argmin_f64
    # Multi-dtype index_select
    index_select_f16
    index_select_f64
    index_select_i64
    # Multi-dtype conv_transpose2d
    conv_transpose2d_forward_f64
    conv_transpose2d_forward_f16
    # ROI Align operations
    roi_align
    roi_align_f64
    roi_align_f16
    roi_align_backward
    roi_align_backward_f64
    roi_align_backward_f16
    # GPU sorting
    bitonic_sort
    # LayerNorm backward
    layer_norm_backward
    layer_norm_backward_f64
    layer_norm_backward_f16
    # GroupNorm backward
    group_norm_backward
    group_norm_backward_f64
    group_norm_backward_f16
    # Embedding backward
    embedding_backward
    embedding_backward_f64
    embedding_backward_f16
    # RMSNorm forward and backward
    rms_norm
    rms_norm_f64
    rms_norm_f16
    rms_norm_backward
    rms_norm_backward_f64
    rms_norm_backward_f16
    # Phase 3: Nonzero, OneHot, BoxIoU, dtype variants
    nonzero_count
    nonzero_gather
    one_hot
    box_iou
    masked_select_gather_f64
    masked_select_gather_f16
    bitonic_sort_f64
    bitonic_sort_i32
    cast_int64_to_int32
    # Creation operations
    arange
    arange_f64
    linspace
    linspace_f64
    eye
    eye_f64
    # Phase 4: Fused optimizer kernels
    fused_rmsprop_step
    fused_adadelta_step
    fused_adagrad_step
)

# Compile each shader
if(GLSLC)
    foreach(SHADER ${SHADERS})
        set(SHADER_SOURCE ${SHADER_DIR}/${SHADER}.comp)
        set(SHADER_OUTPUT ${SHADER_OUTPUT_DIR}/${SHADER}.spv)

        add_custom_command(
            OUTPUT ${SHADER_OUTPUT}
            COMMAND ${GLSLC} -fshader-stage=compute ${SHADER_SOURCE} -o ${SHADER_OUTPUT}
            DEPENDS ${SHADER_SOURCE}
            COMMENT "Compiling shader ${SHADER}.comp to SPIR-V"
        )

        list(APPEND SHADER_OUTPUTS ${SHADER_OUTPUT})
    endforeach()

    # Custom target for shader compilation
    add_custom_target(vulkan_shaders ALL DEPENDS ${SHADER_OUTPUTS})

    # Generate embedded shaders header and implementation
    set(EMBEDDED_SHADERS_HEADER ${CMAKE_CURRENT_BINARY_DIR}/embedded_shaders.hpp)
    set(EMBEDDED_SHADERS_IMPL ${CMAKE_CURRENT_BINARY_DIR}/embedded_shaders.cpp)

    add_custom_command(
        OUTPUT ${EMBEDDED_SHADERS_HEADER} ${EMBEDDED_SHADERS_IMPL}
        COMMAND ${CMAKE_COMMAND} -E env python3 ${CMAKE_SOURCE_DIR}/scripts/embed_shaders.py
                ${SHADER_OUTPUT_DIR} ${EMBEDDED_SHADERS_HEADER}
        DEPENDS ${SHADER_OUTPUTS} ${CMAKE_SOURCE_DIR}/scripts/embed_shaders.py
        COMMENT "Embedding Vulkan shaders into C++ source"
    )

    # Custom target for embedded shaders
    add_custom_target(vulkan_embedded_shaders ALL
        DEPENDS ${EMBEDDED_SHADERS_HEADER} ${EMBEDDED_SHADERS_IMPL}
    )
endif()

# Vulkan backend library
if(GLSLC)
    add_library(tenzor_backend_vulkan SHARED
        vulkan_backend.cpp
        vulkan_kernel_registry.cpp
        vulkan_backend.hpp
        vulkan_utils.hpp
        ${CMAKE_SOURCE_DIR}/src/backend/vulkan_caching_allocator.cpp
        ${EMBEDDED_SHADERS_IMPL}
    )
else()
    add_library(tenzor_backend_vulkan SHARED
        vulkan_backend.cpp
        vulkan_kernel_registry.cpp
        vulkan_backend.hpp
        vulkan_utils.hpp
        ${CMAKE_SOURCE_DIR}/src/backend/vulkan_caching_allocator.cpp
    )
endif()

# Set target properties
set_target_properties(tenzor_backend_vulkan PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    OUTPUT_NAME "tenzor_backend_vulkan"
    PREFIX ""  # Remove 'lib' prefix on Unix
)

target_include_directories(tenzor_backend_vulkan PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}  # For embedded_shaders.hpp
)

target_link_libraries(tenzor_backend_vulkan PRIVATE
    ${Vulkan_LIBRARIES}
    tenzor_core
)

# C++20 standard
target_compile_features(tenzor_backend_vulkan PRIVATE cxx_std_20)

# Set shader path for runtime
target_compile_definitions(tenzor_backend_vulkan PRIVATE
    TENZOR_VULKAN_SHADER_PATH="${SHADER_OUTPUT_DIR}/"
)

# Platform-specific settings
if(WIN32)
    target_compile_definitions(tenzor_backend_vulkan PRIVATE
        VK_USE_PLATFORM_WIN32_KHR
    )
elseif(APPLE)
    target_compile_definitions(tenzor_backend_vulkan PRIVATE
        VK_USE_PLATFORM_MACOS_MVK
        VK_USE_PLATFORM_METAL_EXT
    )
else() # Linux
    target_compile_definitions(tenzor_backend_vulkan PRIVATE
        VK_USE_PLATFORM_XLIB_KHR
        VK_USE_PLATFORM_XCB_KHR
    )
endif()

# Optimization flags
if(CMAKE_BUILD_TYPE MATCHES "Release")
    target_compile_options(tenzor_backend_vulkan PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native>
        $<$<CXX_COMPILER_ID:MSVC>:/O2>
    )
endif()

# Warning flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tenzor_backend_vulkan PRIVATE
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
    )
endif()

# Dependency on shader compilation and embedding
if(GLSLC)
    add_dependencies(tenzor_backend_vulkan vulkan_shaders vulkan_embedded_shaders)
endif()

# Installation
install(TARGETS tenzor_backend_vulkan
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install shaders
if(GLSLC)
    install(DIRECTORY ${SHADER_OUTPUT_DIR}/
        DESTINATION share/tenzor/shaders/vulkan
        FILES_MATCHING PATTERN "*.spv"
    )
endif()

# Print configuration summary
message(STATUS "Vulkan Backend Configuration:")
message(STATUS "  - Vulkan Version: ${Vulkan_VERSION}")
message(STATUS "  - Shader Compiler: ${GLSLC}")
message(STATUS "  - Shader Output Dir: ${SHADER_OUTPUT_DIR}")
message(STATUS "  - Platform: ${CMAKE_SYSTEM_NAME}")
