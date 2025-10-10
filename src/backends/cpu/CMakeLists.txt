# CPU Backend
# Always available, uses SIMD and OpenMP for optimization

set(CPU_BACKEND_SOURCES
    cpu_backend.cpp
    kernels/math.cpp
    kernels/reduction.cpp
    kernels/transform.cpp
    kernels/activations.cpp
)

# Create CPU backend shared library
add_library(tenzor_backend_cpu SHARED ${CPU_BACKEND_SOURCES})

# Set target properties
set_target_properties(tenzor_backend_cpu PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    OUTPUT_NAME "tenzor_backend_cpu"
    PREFIX ""  # Remove 'lib' prefix on Unix
)

# Include directories
target_include_directories(tenzor_backend_cpu PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Link against core library
target_link_libraries(tenzor_backend_cpu PRIVATE
    tenzor_core
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(tenzor_backend_cpu PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "CPU Backend: OpenMP enabled")
endif()

# Platform-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Enable SIMD optimizations
    target_compile_options(tenzor_backend_cpu PRIVATE
        -march=native
        -mtune=native
        -ffast-math
    )

    # Check for specific SIMD support
    # With -march=native, the compiler will emit the correct instructions
    # We detect CPU capabilities at compile time via compiler predefined macros
    message(STATUS "CPU Backend: Using -march=native for automatic SIMD detection")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(tenzor_backend_cpu PRIVATE
        /arch:AVX2
        /fp:fast
    )
endif()

# Optional: Link against optimized BLAS library
find_package(BLAS QUIET)
if(BLAS_FOUND)
    target_link_libraries(tenzor_backend_cpu PRIVATE ${BLAS_LIBRARIES})
    target_compile_definitions(tenzor_backend_cpu PRIVATE TENZOR_HAS_BLAS)
    message(STATUS "CPU Backend: BLAS support enabled")
else()
    message(STATUS "CPU Backend: BLAS not found, using built-in matrix operations")
endif()

# Installation
install(TARGETS tenzor_backend_cpu
    LIBRARY DESTINATION lib/tenzor/backends
    RUNTIME DESTINATION bin
)

message(STATUS "CPU Backend: Configured successfully")
