# Metal Backend for Tenzor
# Only builds on macOS

cmake_minimum_required(VERSION 3.16)

# Only build on macOS
if(NOT APPLE)
    message(STATUS "Metal backend skipped (not on macOS)")
    return()
endif()

# Enable Objective-C++
enable_language(OBJCXX)

# Find Metal framework
find_library(METAL_LIBRARY Metal REQUIRED)
find_library(METALKIT_LIBRARY MetalKit REQUIRED)
find_library(MPS_LIBRARY MetalPerformanceShaders REQUIRED)
find_library(FOUNDATION_LIBRARY Foundation REQUIRED)

message(STATUS "Found Metal: ${METAL_LIBRARY}")
message(STATUS "Found MetalPerformanceShaders: ${MPS_LIBRARY}")

# Metal shader sources
set(METAL_SHADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/matmul.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/conv2d.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/pooling.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/batchnorm.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/activations.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/reduction.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/transform.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/math.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/indexing.metal
)

# Compile Metal shaders to .air (intermediate representation)
set(METAL_AIR_FILES "")
foreach(SHADER ${METAL_SHADERS})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    set(AIR_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.air)

    add_custom_command(
        OUTPUT ${AIR_FILE}
        COMMAND xcrun -sdk macosx metal -c ${SHADER} -o ${AIR_FILE}
        DEPENDS ${SHADER}
        COMMENT "Compiling Metal shader: ${SHADER_NAME}"
        VERBATIM
    )

    list(APPEND METAL_AIR_FILES ${AIR_FILE})
endforeach()

# Link .air files into .metallib
set(METALLIB_FILE ${CMAKE_CURRENT_BINARY_DIR}/default.metallib)
add_custom_command(
    OUTPUT ${METALLIB_FILE}
    COMMAND xcrun -sdk macosx metallib ${METAL_AIR_FILES} -o ${METALLIB_FILE}
    DEPENDS ${METAL_AIR_FILES}
    COMMENT "Linking Metal library: default.metallib"
    VERBATIM
)

# Create custom target for Metal shaders
add_custom_target(metal_shaders ALL
    DEPENDS ${METALLIB_FILE}
    COMMENT "Building Metal shaders"
)

# Metal backend source files
set(METAL_BACKEND_SOURCES
    metal_backend.mm
)

set(METAL_BACKEND_HEADERS
    metal_backend.hpp
    metal_utils.hpp
)

# Create Metal backend library
add_library(tenzor_metal STATIC
    ${METAL_BACKEND_SOURCES}
    ${METAL_BACKEND_HEADERS}
)

# Add dependency on shader compilation
add_dependencies(tenzor_metal metal_shaders)

# Set C++17 standard
target_compile_features(tenzor_metal PUBLIC cxx_std_17)

# Compiler flags for Objective-C++
target_compile_options(tenzor_metal PRIVATE
    -fobjc-arc  # Enable ARC
    -fno-objc-exceptions  # Disable Objective-C exceptions for performance
)

# Optimization flags for Apple Silicon
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    target_compile_options(tenzor_metal PRIVATE
        -mcpu=apple-m1  # Optimize for Apple Silicon
        -mtune=apple-m1
    )
    message(STATUS "Optimizing Metal backend for Apple Silicon")
endif()

# Include directories
target_include_directories(tenzor_metal PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

# Link Metal frameworks
target_link_libraries(tenzor_metal PUBLIC
    ${METAL_LIBRARY}
    ${METALKIT_LIBRARY}
    ${MPS_LIBRARY}
    ${FOUNDATION_LIBRARY}
)

# Install metallib file
install(FILES ${METALLIB_FILE}
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT runtime
)

# Install headers
install(FILES ${METAL_BACKEND_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tenzor/backends/metal
    COMPONENT development
)

# Install library
install(TARGETS tenzor_metal
    EXPORT TenzorTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Define preprocessor macros
target_compile_definitions(tenzor_metal PUBLIC
    TENZOR_METAL_BACKEND
    TENZOR_USE_METAL
)

# Metal backend configuration
set(TENZOR_METAL_ENABLED ON PARENT_SCOPE)

# Print configuration summary
message(STATUS "Metal Backend Configuration:")
message(STATUS "  Metal Library: ${METAL_LIBRARY}")
message(STATUS "  MPS Library: ${MPS_LIBRARY}")
message(STATUS "  Shader Library: ${METALLIB_FILE}")
message(STATUS "  Number of shaders: ${CMAKE_LIST_LENGTH:METAL_SHADERS}")

# Optional: Add tests for Metal backend
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# Optional: Generate Metal shader documentation
option(GENERATE_METAL_DOCS "Generate documentation for Metal shaders" OFF)
if(GENERATE_METAL_DOCS)
    find_program(METAL_DOC_TOOL metaldoc)
    if(METAL_DOC_TOOL)
        add_custom_target(metal_docs
            COMMAND ${METAL_DOC_TOOL} ${METAL_SHADERS}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Generating Metal shader documentation"
        )
    endif()
endif()

# Verify Metal support at runtime
add_custom_command(TARGET tenzor_metal POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Metal backend built successfully"
    COMMAND ${CMAKE_COMMAND} -E echo "Metallib location: ${METALLIB_FILE}"
    VERBATIM
)

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON PARENT_SCOPE)
