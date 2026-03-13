# WebGPU Backend for Tenzor
cmake_minimum_required(VERSION 3.15)

# Option to enable/disable WebGPU backend
option(TENZOR_ENABLE_WEBGPU "Enable WebGPU backend support" ON)

if(NOT TENZOR_ENABLE_WEBGPU)
    message(STATUS "WebGPU backend disabled")
    return()
endif()

message(STATUS "Configuring WebGPU backend...")

# ============================================================================
# Find WebGPU Implementation
# ============================================================================

# Try to find Dawn (Google's WebGPU implementation)
find_package(Dawn QUIET)

# If Dawn not found, try wgpu-native
if(NOT Dawn_FOUND)
    find_package(wgpu-native QUIET)
endif()

# If neither found, try pkg-config
if(NOT Dawn_FOUND AND NOT wgpu-native_FOUND)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(WEBGPU QUIET webgpu)
    endif()
endif()

# Check if we found any WebGPU implementation
set(WEBGPU_FOUND FALSE)

if(Dawn_FOUND)
    message(STATUS "Found Dawn WebGPU implementation")
    set(WEBGPU_FOUND TRUE)
    set(WEBGPU_LIBRARIES dawn::webgpu)
    set(WEBGPU_INCLUDE_DIRS ${Dawn_INCLUDE_DIRS})
elseif(wgpu-native_FOUND)
    message(STATUS "Found wgpu-native WebGPU implementation")
    set(WEBGPU_FOUND TRUE)
    set(WEBGPU_LIBRARIES wgpu-native::wgpu-native)
    set(WEBGPU_INCLUDE_DIRS ${wgpu-native_INCLUDE_DIRS})
elseif(WEBGPU_FOUND)
    message(STATUS "Found WebGPU via pkg-config")
    set(WEBGPU_LIBRARIES ${WEBGPU_LIBRARIES})
    set(WEBGPU_INCLUDE_DIRS ${WEBGPU_INCLUDE_DIRS})
else()
    message(WARNING "WebGPU implementation not found. WebGPU backend will not be built.")
    message(WARNING "Install Dawn or wgpu-native to enable WebGPU support:")
    message(WARNING "  - Dawn: https://dawn.googlesource.com/dawn")
    message(WARNING "  - wgpu-native: https://github.com/gfx-rs/wgpu-native")
    return()
endif()

# ============================================================================
# WebGPU Backend Library
# ============================================================================

# Collect source files
set(WEBGPU_BACKEND_SOURCES
    webgpu_backend.cpp
    webgpu_kernel_registry.cpp
)

set(WEBGPU_BACKEND_HEADERS
    webgpu_backend.hpp
    webgpu_utils.hpp
)

# Collect WGSL shader files
set(WEBGPU_SHADERS
    kernels/matmul.wgsl
    kernels/conv2d.wgsl
    kernels/pooling.wgsl
    kernels/batchnorm.wgsl
    kernels/activations.wgsl
    kernels/reduction.wgsl
    kernels/transform.wgsl
    kernels/math.wgsl
    kernels/indexing.wgsl
)

# Create library
add_library(tenzor_webgpu_backend STATIC
    ${WEBGPU_BACKEND_SOURCES}
    ${WEBGPU_BACKEND_HEADERS}
)

# Include directories
target_include_directories(tenzor_webgpu_backend
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${WEBGPU_INCLUDE_DIRS}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

# Link WebGPU library
target_link_libraries(tenzor_webgpu_backend
    PUBLIC
        ${WEBGPU_LIBRARIES}
)

# Compiler features
target_compile_features(tenzor_webgpu_backend
    PUBLIC
        cxx_std_23
)

# Compiler options
if(MSVC)
    target_compile_options(tenzor_webgpu_backend PRIVATE
        /W4
        /WX-  # Don't treat warnings as errors for external deps
    )
else()
    target_compile_options(tenzor_webgpu_backend PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
    )
endif()

# Define preprocessor macros
target_compile_definitions(tenzor_webgpu_backend
    PUBLIC
        TENZOR_WEBGPU_BACKEND_ENABLED
)

# Platform-specific settings
if(EMSCRIPTEN)
    target_compile_definitions(tenzor_webgpu_backend
        PUBLIC
            __EMSCRIPTEN__
    )
    target_link_options(tenzor_webgpu_backend
        PUBLIC
            -sUSE_WEBGPU=1
    )
    message(STATUS "WebGPU backend configured for Emscripten/WASM")
endif()

# ============================================================================
# Install Shader Files
# ============================================================================

# Install WGSL shaders to a known location
install(FILES ${WEBGPU_SHADERS}
    DESTINATION share/tenzor/webgpu/kernels
    COMPONENT shaders
)

# Define shader installation directory as compile definition
target_compile_definitions(tenzor_webgpu_backend
    PRIVATE
        TENZOR_WEBGPU_SHADER_DIR="${CMAKE_INSTALL_PREFIX}/share/tenzor/webgpu/kernels"
)

# For build tree, copy shaders to build directory
add_custom_target(copy_webgpu_shaders ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_BINARY_DIR}/share/tenzor/webgpu/kernels
    COMMENT "Creating WebGPU shader directory"
)

foreach(shader ${WEBGPU_SHADERS})
    add_custom_command(TARGET copy_webgpu_shaders POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/${shader}
            ${CMAKE_BINARY_DIR}/share/tenzor/webgpu/kernels/
        COMMENT "Copying ${shader}"
    )
endforeach()

add_dependencies(tenzor_webgpu_backend copy_webgpu_shaders)

# ============================================================================
# Testing
# ============================================================================

if(TENZOR_BUILD_TESTS)
    # WebGPU backend tests
    add_executable(test_webgpu_backend
        tests/test_webgpu_backend.cpp
    )

    target_link_libraries(test_webgpu_backend
        PRIVATE
            tenzor_webgpu_backend
            gtest_main
    )

    add_test(NAME webgpu_backend
        COMMAND test_webgpu_backend
    )

    # Set test properties
    set_tests_properties(webgpu_backend PROPERTIES
        TIMEOUT 60
        LABELS "backend;webgpu"
    )
endif()

# ============================================================================
# Examples
# ============================================================================

if(TENZOR_BUILD_EXAMPLES)
    # WebGPU backend example
    add_executable(webgpu_example
        examples/webgpu_example.cpp
    )

    target_link_libraries(webgpu_example
        PRIVATE
            tenzor_webgpu_backend
    )

    install(TARGETS webgpu_example
        RUNTIME DESTINATION bin/examples
        COMPONENT examples
    )
endif()

# ============================================================================
# Installation
# ============================================================================

install(TARGETS tenzor_webgpu_backend
    EXPORT TenzorTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
    COMPONENT runtime
)

install(FILES ${WEBGPU_BACKEND_HEADERS}
    DESTINATION include/tenzor/backends/webgpu
    COMPONENT development
)

# ============================================================================
# Package Configuration
# ============================================================================

# Export configuration for find_package
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/TenzorWebGPUConfigVersion.cmake"
    VERSION 1.0.0
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/TenzorWebGPUConfigVersion.cmake"
    DESTINATION lib/cmake/Tenzor
    COMPONENT development
)

# ============================================================================
# Summary
# ============================================================================

message(STATUS "WebGPU Backend Configuration:")
message(STATUS "  WebGPU Implementation: ${WEBGPU_LIBRARIES}")
message(STATUS "  Include Directories: ${WEBGPU_INCLUDE_DIRS}")
message(STATUS "  Shaders: ${CMAKE_INSTALL_PREFIX}/share/tenzor/webgpu/kernels")
message(STATUS "  Emscripten Build: ${EMSCRIPTEN}")
message(STATUS "WebGPU backend configured successfully")
