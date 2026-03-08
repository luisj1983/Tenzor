cmake_minimum_required(VERSION 3.25)

# Default to Release build if no build type specified (single-config generators only)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Pre-project configuration for ROCm/HIP
# Must happen before project() because HIP language must be enabled at project() time
# Check if ROCm is available on the system first
set(_ROCM_AVAILABLE FALSE)
if(EXISTS "/opt/rocm/bin/hipcc")
    set(_ROCM_AVAILABLE TRUE)
endif()


# Set TENZOR_BUILD_ROCM before project() - use cache to preserve user preference
# Default to ON only if ROCm is available on the system
if(NOT DEFINED TENZOR_BUILD_ROCM)
    option(TENZOR_BUILD_ROCM "Build ROCm backend" OFF)
endif()

# Set up HIP architectures if building ROCm backend
if(TENZOR_BUILD_ROCM AND _ROCM_AVAILABLE)
    if(NOT CMAKE_HIP_ARCHITECTURES)
        # Try to auto-detect GPU architecture
        execute_process(
            COMMAND /opt/rocm/bin/rocm_agent_enumerator
            OUTPUT_VARIABLE ROCM_AGENTS
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )

        # Extract first gfx architecture (skip CPU)
        string(REGEX MATCH "gfx[0-9a-z]+" DETECTED_ARCH "${ROCM_AGENTS}")

        if(DETECTED_ARCH)
            set(CMAKE_HIP_ARCHITECTURES "${DETECTED_ARCH}" CACHE STRING "HIP GPU architectures" FORCE)
            message(STATUS "Auto-detected GPU architecture: ${DETECTED_ARCH}")
        else()
            # Fallback to common architectures
            set(CMAKE_HIP_ARCHITECTURES "gfx1150;gfx900;gfx90c;gfx906;gfx908" CACHE STRING "HIP GPU architectures" FORCE)
            message(STATUS "Could not detect GPU, using common architectures: gfx900;gfx90c;gfx906;gfx908")
        endif()
    endif()
endif()

# Determine languages to enable
set(TENZOR_LANGUAGES CXX)
if(TENZOR_BUILD_ROCM AND _ROCM_AVAILABLE)
    list(APPEND TENZOR_LANGUAGES HIP)
endif()

project(tenzor VERSION 1.0.0 LANGUAGES ${TENZOR_LANGUAGES})

# Use ccache if available for faster rebuilds
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "ccache found: ${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(TENZOR_BUILD_CUDA "Build CUDA backend" ON)
# Note: TENZOR_BUILD_ROCM is set before project() to enable HIP language
option(TENZOR_BUILD_ONEAPI "Build OneAPI backend" ON)
option(TENZOR_BUILD_VULKAN "Build Vulkan backend" ON)
option(TENZOR_BUILD_METAL "Build Metal backend (macOS/iOS only)" OFF)
option(TENZOR_BUILD_WEBGPU "Build WebGPU backend (WASM/browser)" OFF)
option(TENZOR_BUILD_PYTHON "Build Python bindings" ON)
option(TENZOR_BUILD_TESTS "Build tests" ON)
option(TENZOR_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(TENZOR_BUILD_EXAMPLES "Build examples" ON)
option(TENZOR_BUILD_MODEL_HUB "Build model hub (requires CURL and OpenSSL)" ON)
option(TENZOR_REQUIRE_MKL "Require Intel MKL (FATAL_ERROR if not found)" OFF)

# Link-Time Optimization
option(TENZOR_ENABLE_LTO "Enable Link-Time Optimization" OFF)
if(TENZOR_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT lto_supported OUTPUT lto_error)
    if(lto_supported)
        message(STATUS "LTO enabled")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "LTO requested but not supported: ${lto_error}")
    endif()
endif()

# Coverage build type
option(TENZOR_ENABLE_COVERAGE "Enable coverage reporting" OFF)

# Sanitizer options
option(TENZOR_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(TENZOR_ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(TENZOR_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

if(TENZOR_ENABLE_ASAN AND TENZOR_ENABLE_TSAN)
    message(FATAL_ERROR "AddressSanitizer and ThreadSanitizer are mutually exclusive")
endif()

# Static analysis
option(TENZOR_ENABLE_CLANG_TIDY "Enable clang-tidy static analysis" OFF)

# Export compile_commands.json for IDE integration and clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Tenzor-specific compiler flags (stored in variables, applied per-target to avoid
# polluting consumers via add_subdirectory or FetchContent)
option(TENZOR_WERROR "Treat warnings as errors (for CI)" OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(TENZOR_WARNING_FLAGS -Wall -Wextra -pedantic)
    if(TENZOR_WERROR)
        list(APPEND TENZOR_WARNING_FLAGS -Werror)
    endif()

    # Use -march=native for maximum performance on the build machine.
    # Disabled by default for portable binaries. Enable for local dev builds.
    option(TENZOR_NATIVE_ARCH "Use -march=native for host-specific optimizations" OFF)
    if(TENZOR_NATIVE_ARCH)
        list(APPEND TENZOR_WARNING_FLAGS -march=native)
    endif()

    # Enable coverage if requested (must be global — applies to all compiled code)
    if(TENZOR_ENABLE_COVERAGE)
        message(STATUS "Coverage reporting enabled")
        add_compile_options(--coverage -fprofile-arcs -ftest-coverage)
        add_link_options(--coverage)
    endif()

    # Sanitizers (must be global — applies to all compiled code)
    if(TENZOR_ENABLE_ASAN)
        message(STATUS "AddressSanitizer enabled")
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address)
    endif()

    if(TENZOR_ENABLE_TSAN)
        message(STATUS "ThreadSanitizer enabled")
        add_compile_options(-fsanitize=thread)
        add_link_options(-fsanitize=thread)
    endif()

    if(TENZOR_ENABLE_UBSAN)
        message(STATUS "UndefinedBehaviorSanitizer enabled")
        add_compile_options(-fsanitize=undefined -fno-omit-frame-pointer)
        add_link_options(-fsanitize=undefined)
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(TENZOR_WARNING_FLAGS /W4 /arch:AVX2)
    if(TENZOR_WERROR)
        list(APPEND TENZOR_WARNING_FLAGS /WX)
    endif()
endif()

# clang-tidy integration
if(TENZOR_ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
    if(CLANG_TIDY_EXE)
        message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
    else()
        message(WARNING "clang-tidy requested but not found")
    endif()
endif()

# Set output directories - all binaries and libraries in bin/
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# For multi-configuration generators (Visual Studio, Xcode)
foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${CONFIG} CONFIG_UPPER)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_BINARY_DIR}/bin)
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

        target_link_libraries(tenzor_python PRIVATE tenzor_core tenzor_backend_cpu)
        target_compile_options(tenzor_python PRIVATE ${TENZOR_WARNING_FLAGS})

        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(tenzor_python PRIVATE -Wno-unused-parameter)
        endif()

        set_target_properties(tenzor_python PROPERTIES
            OUTPUT_NAME "tenzor_core"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/tenzor"
        )

        # Copy Python package files at build time (not just configure time)
        # so that edits to .py files are picked up by subsequent builds.
        add_custom_target(copy_python_files ALL
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/__init__.py
                ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/nn.py
                ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/functional.py
                ${CMAKE_BINARY_DIR}/python/tenzor/
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/py.typed
                ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/__init__.pyi
                ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/nn.pyi
                ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/functional.pyi
                ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/optim.pyi
                ${CMAKE_CURRENT_SOURCE_DIR}/python/tenzor/torch_interop.pyi
                ${CMAKE_BINARY_DIR}/python/tenzor/
            COMMENT "Copying Python files and type stubs to build directory"
        )
        add_dependencies(tenzor_python copy_python_files)
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
    PUBLIC_HEADER DESTINATION include/tenzor
    INCLUDES DESTINATION include
)

install(DIRECTORY include/tenzor DESTINATION include)

# Install backend shared libraries (only non-imported targets that were built)
foreach(backend cpu cuda rocm oneapi vulkan metal webgpu)
    if(TARGET tenzor_backend_${backend})
        get_target_property(_imported tenzor_backend_${backend} IMPORTED)
        if(NOT _imported)
            install(TARGETS tenzor_backend_${backend}
                LIBRARY DESTINATION lib/tenzor/backends)
        endif()
    endif()
endforeach()

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

# pkg-config support
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tenzor.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/tenzor.pc"
    @ONLY
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/tenzor.pc"
    DESTINATION lib/pkgconfig
)

# Python module installation
if(TENZOR_BUILD_PYTHON AND TARGET tenzor_python)
    # Install the compiled extension module
    install(TARGETS tenzor_python
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python/tenzor
        COMPONENT python
    )
    # Install Python source files
    install(DIRECTORY python/tenzor/
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python/tenzor
        COMPONENT python
        FILES_MATCHING
            PATTERN "*.py"
            PATTERN "*.pyi"
    )
endif()

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
message(STATUS "  Vulkan backend:       ${TENZOR_BUILD_VULKAN}")
message(STATUS "  Metal backend:        ${TENZOR_BUILD_METAL}")
message(STATUS "  WebGPU backend:       ${TENZOR_BUILD_WEBGPU}")
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
if(CCACHE_PROGRAM)
    message(STATUS "  ccache:               ${CCACHE_PROGRAM}")
else()
    message(STATUS "  ccache:               Not found")
endif()
if(TENZOR_ENABLE_LTO AND lto_supported)
    message(STATUS "  LTO:                  Enabled")
else()
    message(STATUS "  LTO:                  Disabled")
endif()
message(STATUS "")
