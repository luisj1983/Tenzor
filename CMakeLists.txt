cmake_minimum_required(VERSION 3.25)

# Set default HIP architecture before project() if building ROCm backend
# Auto-detect GPU architecture if not specified
if(TENZOR_BUILD_ROCM OR NOT DEFINED TENZOR_BUILD_ROCM)
    if(NOT CMAKE_HIP_ARCHITECTURES)
        # Try to auto-detect GPU architecture
        execute_process(
            COMMAND /opt/rocm/bin/rocm_agent_enumerator
            OUTPUT_VARIABLE ROCM_AGENTS
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )

        # Extract first non-gfx agent (skip CPU)
        string(REGEX MATCH "gfx[0-9a-z]+" DETECTED_ARCH "${ROCM_AGENTS}")

        if(DETECTED_ARCH)
            set(CMAKE_HIP_ARCHITECTURES "${DETECTED_ARCH}" CACHE STRING "HIP GPU architectures" FORCE)
            message(STATUS "Auto-detected GPU architecture: ${DETECTED_ARCH}")
        else()
            # Fallback to common architecture
            set(CMAKE_HIP_ARCHITECTURES "gfx900;gfx90c;gfx906;gfx908" CACHE STRING "HIP GPU architectures" FORCE)
            message(STATUS "Could not detect GPU, using common architectures: gfx900;gfx90c;gfx906;gfx908")
        endif()
    endif()
endif()

# Determine languages to enable
set(TENZOR_LANGUAGES CXX)
if(TENZOR_BUILD_ROCM OR NOT DEFINED TENZOR_BUILD_ROCM)
    list(APPEND TENZOR_LANGUAGES HIP)
endif()

project(tenzor VERSION 1.0.0 LANGUAGES ${TENZOR_LANGUAGES})

# C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(TENZOR_BUILD_CUDA "Build CUDA backend" ON)
option(TENZOR_BUILD_ROCM "Build ROCm backend" OFF) #do not enable, causes system crashes
option(TENZOR_BUILD_ONEAPI "Build OneAPI backend" ON)
option(TENZOR_BUILD_PYTHON "Build Python bindings" ON)
option(TENZOR_BUILD_TESTS "Build tests" ON)
option(TENZOR_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(TENZOR_BUILD_EXAMPLES "Build examples" ON)

# Global compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -pedantic -march=native)
    # Disable specific warnings for stub implementations
    add_compile_options(-Wno-unused-parameter -Wno-unused-variable)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/W4 /arch:AVX2)
endif()

# Set output directories - all binaries and libraries in bin/
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# For multi-configuration generators (Visual Studio, Xcode)
foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${CONFIG} CONFIG_UPPER)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_SOURCE_DIR}/bin)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_SOURCE_DIR}/bin)
endforeach()

# Set RPATH so executables can find shared libraries in same directory
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Use $ORIGIN to find libraries in same directory as executable
set(CMAKE_BUILD_RPATH "$ORIGIN:${CMAKE_BINARY_DIR}/bin")

# Find required packages
find_package(Threads REQUIRED)
find_package(OpenMP)

# Include CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Add subdirectories
add_subdirectory(src)

# Python bindings
if(TENZOR_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development)
    find_package(pybind11 CONFIG)

    if(pybind11_FOUND AND Python_FOUND)
        pybind11_add_module(tenzor_python
            python/bindings.cpp
            python/numpy_interop.cpp
        )

        target_link_libraries(tenzor_python PRIVATE tenzor_core)

        set_target_properties(tenzor_python PROPERTIES
            OUTPUT_NAME "tenzor_core"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/tenzor"
        )

        # Copy Python package files
        file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/__init__.py
             DESTINATION ${CMAKE_BINARY_DIR}/python/tenzor)
    else()
        message(WARNING "pybind11 or Python not found, Python bindings will not be built")
    endif()
endif()

# Tests
if(TENZOR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
if(TENZOR_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Examples
if(TENZOR_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Documentation
find_package(Doxygen)
if(DOXYGEN_FOUND)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
    message(STATUS "  Doxygen:              Found (run 'make docs' to generate)")
else()
    message(STATUS "  Doxygen:              Not found (documentation will not be available)")
endif()

# Installation
install(TARGETS tenzor_core
    EXPORT TenzorTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/tenzor DESTINATION include)

# Export targets
install(EXPORT TenzorTargets
    FILE TenzorTargets.cmake
    NAMESPACE Tenzor::
    DESTINATION lib/cmake/Tenzor
)

# Package configuration
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TenzorConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/TenzorConfig.cmake"
    INSTALL_DESTINATION lib/cmake/Tenzor
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/TenzorConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/TenzorConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/TenzorConfigVersion.cmake"
    DESTINATION lib/cmake/Tenzor
)

# Print configuration summary
message(STATUS "")
message(STATUS "Tenzor Configuration Summary")
message(STATUS "============================")
message(STATUS "  Version:              ${PROJECT_VERSION}")
message(STATUS "  Build type:           ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ compiler:         ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "")
message(STATUS "Build Options:")
message(STATUS "  CUDA backend:         ${TENZOR_BUILD_CUDA}")
message(STATUS "  ROCm backend:         ${TENZOR_BUILD_ROCM}")
message(STATUS "  OneAPI backend:       ${TENZOR_BUILD_ONEAPI}")
message(STATUS "  Python bindings:      ${TENZOR_BUILD_PYTHON}")
message(STATUS "  Tests:                ${TENZOR_BUILD_TESTS}")
message(STATUS "  Benchmarks:           ${TENZOR_BUILD_BENCHMARKS}")
message(STATUS "  Examples:             ${TENZOR_BUILD_EXAMPLES}")
message(STATUS "")
if(OpenMP_CXX_FOUND)
    message(STATUS "  OpenMP:               Enabled")
else()
    message(STATUS "  OpenMP:               Disabled")
endif()
message(STATUS "")
