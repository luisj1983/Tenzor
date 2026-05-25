# Vulkan backend for Tenzor
cmake_minimum_required(VERSION 3.15)

# Find Vulkan SDK
find_package(Vulkan)

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

# Auto-discover all .comp shader files (prevents missing-shader bugs)
file(GLOB SHADER_FILES "${SHADER_DIR}/*.comp")
set(SHADERS "")
foreach(SHADER_FILE ${SHADER_FILES})
    get_filename_component(SHADER_NAME ${SHADER_FILE} NAME_WE)
    # Exclude shaders that need SPIR-V 1.3 (handled separately below).
    # HH.9: 'conv2d' is the dead pre-im2col path with a mismatched
    # 18-uint push-constant struct — moved to kernels/_disabled/, but
    # guard the embed list so accidental re-adds don't trap getPipeline.
    if(NOT SHADER_NAME STREQUAL "reduction_subgroup" AND
       NOT SHADER_NAME STREQUAL "reduction_tree" AND
       NOT SHADER_NAME STREQUAL "conv2d")
        list(APPEND SHADERS ${SHADER_NAME})
    endif()
endforeach()
list(SORT SHADERS)

# Shaders requiring SPIR-V 1.3 (subgroup operations, Vulkan 1.1+)
set(SHADERS_SPIRV13
    reduction_subgroup
    reduction_tree
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

    # Compile SPIR-V 1.3 shaders (needed for subgroup operations)
    foreach(SHADER ${SHADERS_SPIRV13})
        set(SHADER_SOURCE ${SHADER_DIR}/${SHADER}.comp)
        set(SHADER_OUTPUT ${SHADER_OUTPUT_DIR}/${SHADER}.spv)

        add_custom_command(
            OUTPUT ${SHADER_OUTPUT}
            COMMAND ${GLSLC} -fshader-stage=compute --target-env=vulkan1.1 ${SHADER_SOURCE} -o ${SHADER_OUTPUT}
            DEPENDS ${SHADER_SOURCE}
            COMMENT "Compiling shader ${SHADER}.comp to SPIR-V 1.3"
        )

        list(APPEND SHADER_OUTPUTS ${SHADER_OUTPUT})
    endforeach()

    # Custom target for shader compilation
    add_custom_target(vulkan_shaders ALL DEPENDS ${SHADER_OUTPUTS})

    # Generate embedded shaders header and implementation
    set(EMBEDDED_SHADERS_HEADER ${CMAKE_CURRENT_BINARY_DIR}/embedded_shaders.hpp)
    set(EMBEDDED_SHADERS_IMPL ${CMAKE_CURRENT_BINARY_DIR}/embedded_shaders.cpp)

    if(NOT Python_EXECUTABLE)
        find_package(Python COMPONENTS Interpreter QUIET)
        if(NOT Python_EXECUTABLE)
            message(FATAL_ERROR "Python interpreter required for Vulkan shader embedding but not found")
        endif()
    endif()

    add_custom_command(
        OUTPUT ${EMBEDDED_SHADERS_HEADER} ${EMBEDDED_SHADERS_IMPL}
        COMMAND ${CMAKE_COMMAND} -E env ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/embed_shaders.py
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
        vulkan_memory.cpp
        vulkan_commands.cpp
        vulkan_dispatch.cpp
        vulkan_ops_math.cpp
        vulkan_ops_pooling.cpp
        vulkan_ops_norm.cpp
        vulkan_ops_vision.cpp
        vulkan_ops_activation.cpp
        vulkan_ops_reduction.cpp
        vulkan_ops_indexing.cpp
        vulkan_ops_shape.cpp
        vulkan_ops_memory.cpp
        vulkan_ops_conv.cpp
        vulkan_ops_linear.cpp
        vulkan_ops_rnn.cpp
        vulkan_ops_sort.cpp
        vulkan_ops_misc.cpp
        vulkan_ops_complex.cpp
        vulkan_ops_fft.cpp
        vulkan_ops_linalg.cpp
        vulkan_ops_embed.cpp
        vulkan_ops_fused.cpp
        vulkan_ops_special_math.cpp
        vulkan_ops_grid_sample.cpp
        vulkan_ops_advanced_index.cpp
        vulkan_ops_sampling.cpp
        vulkan_ops_stft.cpp
        vulkan_kernel_registry.cpp
        vulkan_backend.hpp
        vulkan_helpers.hpp
        vulkan_utils.hpp
        ${CMAKE_SOURCE_DIR}/src/backend/vulkan_caching_allocator.cpp
        ${EMBEDDED_SHADERS_IMPL}
    )
else()
    add_library(tenzor_backend_vulkan SHARED
        vulkan_backend.cpp
        vulkan_memory.cpp
        vulkan_commands.cpp
        vulkan_dispatch.cpp
        vulkan_ops_math.cpp
        vulkan_ops_pooling.cpp
        vulkan_ops_norm.cpp
        vulkan_ops_vision.cpp
        vulkan_ops_activation.cpp
        vulkan_ops_reduction.cpp
        vulkan_ops_indexing.cpp
        vulkan_ops_shape.cpp
        vulkan_ops_memory.cpp
        vulkan_ops_conv.cpp
        vulkan_ops_linear.cpp
        vulkan_ops_rnn.cpp
        vulkan_ops_sort.cpp
        vulkan_ops_misc.cpp
        vulkan_ops_complex.cpp
        vulkan_ops_fft.cpp
        vulkan_ops_linalg.cpp
        vulkan_ops_embed.cpp
        vulkan_ops_fused.cpp
        vulkan_ops_special_math.cpp
        vulkan_ops_grid_sample.cpp
        vulkan_ops_advanced_index.cpp
        vulkan_ops_sampling.cpp
        vulkan_ops_stft.cpp
        vulkan_kernel_registry.cpp
        vulkan_backend.hpp
        vulkan_helpers.hpp
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

# Optional: Vulkan Memory Allocator (VMA) for better memory management
option(TENZOR_USE_VMA "Use Vulkan Memory Allocator (VMA) for Vulkan backend" OFF)
if(TENZOR_USE_VMA)
    include(FetchContent)
    FetchContent_Declare(
        VulkanMemoryAllocator
        GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
        GIT_TAG v3.1.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(VulkanMemoryAllocator)
    target_link_libraries(tenzor_backend_vulkan PRIVATE GPUOpen::VulkanMemoryAllocator)
    target_compile_definitions(tenzor_backend_vulkan PRIVATE TENZOR_HAS_VMA)
    message(STATUS "  VMA integration: ON")
else()
    message(STATUS "  VMA integration: OFF (using custom caching allocator)")
endif()

# C++20 standard
target_compile_features(tenzor_backend_vulkan PRIVATE cxx_std_20)

# Set shader path for runtime
target_compile_definitions(tenzor_backend_vulkan PRIVATE
    TENZOR_VULKAN_SHADER_PATH="${SHADER_OUTPUT_DIR}/"
    TENZOR_INTERNAL
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
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O3>
        $<$<CXX_COMPILER_ID:MSVC>:/O2>
    )
    if(TENZOR_NATIVE_ARCH)
        target_compile_options(tenzor_backend_vulkan PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-march=native>
        )
    endif()
endif()

# Warning flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tenzor_backend_vulkan PRIVATE
        -Wall -Wextra -Wpedantic
    )
endif()

# Dependency on shader compilation and embedding
if(GLSLC)
    add_dependencies(tenzor_backend_vulkan vulkan_shaders vulkan_embedded_shaders)
endif()

# Installation
install(TARGETS tenzor_backend_vulkan
    LIBRARY DESTINATION lib/tenzor/backends
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
