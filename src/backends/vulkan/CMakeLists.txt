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
    math_broadcast
    math_broadcast_f64
    math_broadcast_f16
    math_broadcast_i8
    comparison
    comparison_bool
    comparison_f64
    comparison_i32
    comparison_i64
    matmul
    activations
    activations_f64
    reduction
    reduction_f64
    conv2d
    pooling
    batchnorm
    transform
    indexing
    # Math functions
    trigonometric
    hyperbolic
    # Vision operations
    im2col
    col2im
    # Advanced pooling
    pooling_backward
    pooling_forward_with_indices
    adaptive_pooling
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
    variance_std
    prod_reduction
    boolean_reduction
    # Indexing operations
    embedding
    gather
    scatter
    scatter_f64
    index_select
    # Shape operations (NEW - 9 critical operations)
    fill
    transpose
    permute
    # Backward activation functions
    activations_backward
    softmax_backward
    log_softmax_backward
    swish_backward
    # Conv2d backward operations
    conv2d_backward_input
    conv2d_backward_weight
    conv2d_backward_bias
    # Tensor manipulation operations
    expand
    expand_f64
    cat
    clamp
    clamp_f64
    # BatchNorm2d operations
    batchnorm2d_forward
    batchnorm2d_backward
    batchnorm2d_mean_var
    # Pooling operations (new)
    avg_pool2d
    max_pool2d
    avg_pool2d_backward
    max_pool2d_backward
    # Final 3 operations for 100% coverage
    conv2d_forward
    conv2d_forward_f64
    conv2d_forward_f16
    full
    full_f64
    full_f16
    full_i8
    ones
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
    softmax_f64
    layer_norm_f64
    log_softmax_f64
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
        vulkan_backend.hpp
        vulkan_utils.hpp
        ${EMBEDDED_SHADERS_IMPL}
    )
else()
    add_library(tenzor_backend_vulkan SHARED
        vulkan_backend.cpp
        vulkan_backend.hpp
        vulkan_utils.hpp
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
