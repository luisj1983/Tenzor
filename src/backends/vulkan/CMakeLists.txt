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
    matmul
    activations
    reduction
    conv2d
    pooling
    batchnorm
    transform
    indexing
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
    index_select
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
endif()

# Vulkan backend library
add_library(tenzor_backend_vulkan SHARED
    vulkan_backend.cpp
    vulkan_backend.hpp
    vulkan_utils.hpp
)

target_include_directories(tenzor_backend_vulkan PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
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

# Dependency on shader compilation
if(GLSLC)
    add_dependencies(tenzor_backend_vulkan vulkan_shaders)
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
