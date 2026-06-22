# =============================================================================
# MPS (Apple Metal) Backend
# =============================================================================
# Requires macOS 13+ with Metal framework
# Uses Objective-C++ (.mm) for Metal API and .metal compute shaders

find_library(METAL_FRAMEWORK Metal)
find_library(MPS_FRAMEWORK MetalPerformanceShaders)
find_library(MPSG_FRAMEWORK MetalPerformanceShadersGraph)
find_library(FOUNDATION_FRAMEWORK Foundation)

if(NOT METAL_FRAMEWORK OR NOT MPS_FRAMEWORK)
    message(WARNING "Metal/MPS frameworks not found, skipping MPS backend")
    return()
endif()

# Collect Objective-C++ sources (including kernels subdirectory).
# CONFIGURE_DEPENDS so CMake re-globs when sources are added/removed; without it
# a newly added .mm/.cpp is silently dropped until a manual reconfigure.
file(GLOB MPS_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cpp
)

# Compile Metal shaders to metallib
file(GLOB METAL_SHADERS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.metal)

if(METAL_SHADERS)
    set(METALLIB_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tenzor_kernels.metallib)

    add_custom_command(
        OUTPUT ${METALLIB_OUTPUT}
        COMMAND xcrun -sdk macosx metal
            -std=metal3.0
            ${METAL_SHADERS}
            -o ${METALLIB_OUTPUT}
        DEPENDS ${METAL_SHADERS}
        COMMENT "Compiling Metal shaders"
    )

    add_custom_target(mps_shaders DEPENDS ${METALLIB_OUTPUT})
endif()

add_library(tenzor_backend_mps SHARED ${MPS_SOURCES})

target_include_directories(tenzor_backend_mps PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(tenzor_backend_mps PRIVATE
    tenzor_core
    ${METAL_FRAMEWORK}
    ${MPS_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

if(MPSG_FRAMEWORK)
    target_link_libraries(tenzor_backend_mps PRIVATE ${MPSG_FRAMEWORK})
    target_compile_definitions(tenzor_backend_mps PRIVATE TENZOR_HAS_MPSGRAPH)
endif()

target_compile_definitions(tenzor_backend_mps PRIVATE
    TENZOR_MPS_AVAILABLE
    TENZOR_INTERNAL
)

# Enable ARC for Objective-C++
target_compile_options(tenzor_backend_mps PRIVATE -fobjc-arc)

set_target_properties(tenzor_backend_mps PROPERTIES
    CXX_STANDARD 20
    OUTPUT_NAME tenzor_backend_mps
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)

if(TARGET mps_shaders)
    add_dependencies(tenzor_backend_mps mps_shaders)
endif()

if(NOT SKBUILD)
install(TARGETS tenzor_backend_mps
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
endif()
