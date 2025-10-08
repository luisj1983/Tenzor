# ROCm Backend
# Requires ROCm 5.0 or higher for AMD GPUs

# Find ROCm/HIP
find_package(hip QUIET)
find_package(rocblas QUIET)

if(NOT hip_FOUND)
    message(WARNING "ROCm/HIP not found. ROCm backend will not be built.")
    message(WARNING "Install ROCm from: https://rocm.docs.amd.com/")
    return()
endif()

message(STATUS "ROCm Backend: Found HIP")

set(ROCM_BACKEND_SOURCES
    rocm_backend.cpp
    kernels/math.cpp
)

# Create ROCm backend shared library
add_library(tenzor_backend_rocm SHARED ${ROCM_BACKEND_SOURCES})

# Set target properties
set_target_properties(tenzor_backend_rocm PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    OUTPUT_NAME "tenzor_backend_rocm"
    PREFIX ""  # Remove 'lib' prefix on Unix
)

# Include directories
target_include_directories(tenzor_backend_rocm PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Link against core library and HIP
target_link_libraries(tenzor_backend_rocm PRIVATE
    tenzor_core
    hip::host
)

# Link rocBLAS if available
if(rocblas_FOUND)
    target_link_libraries(tenzor_backend_rocm PRIVATE roc::rocblas)
    target_compile_definitions(tenzor_backend_rocm PRIVATE TENZOR_HAS_ROCBLAS)
    message(STATUS "ROCm Backend: rocBLAS support enabled")
endif()

# Find MIOpen (ROCm's DNN library)
find_package(MIOpen QUIET)
if(MIOpen_FOUND)
    target_link_libraries(tenzor_backend_rocm PRIVATE MIOpen)
    target_compile_definitions(tenzor_backend_rocm PRIVATE TENZOR_HAS_MIOPEN)
    message(STATUS "ROCm Backend: MIOpen support enabled")
else()
    message(STATUS "ROCm Backend: MIOpen not found, using built-in kernels")
endif()

# HIP architecture flags
# Common AMD GPU architectures:
# gfx900 = Vega (MI25), gfx906 = Vega 7nm (MI50/MI60), gfx908 = CDNA (MI100)
# gfx90a = CDNA2 (MI200 series), gfx940 = CDNA3 (MI300)
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "gfx900;gfx906;gfx908;gfx90a;gfx940")
    message(STATUS "ROCm Backend: Building for architectures: ${CMAKE_HIP_ARCHITECTURES}")
else()
    message(STATUS "ROCm Backend: Using user-specified architectures: ${CMAKE_HIP_ARCHITECTURES}")
endif()

# Set HIP architectures
set_target_properties(tenzor_backend_rocm PROPERTIES
    HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
)

# Compiler options
target_compile_options(tenzor_backend_rocm PRIVATE
    -ffast-math
)

# Installation
install(TARGETS tenzor_backend_rocm
    LIBRARY DESTINATION lib/tenzor/backends
    RUNTIME DESTINATION bin
)

message(STATUS "ROCm Backend: Configured successfully")
message(STATUS "  HIP Architectures: ${CMAKE_HIP_ARCHITECTURES}")
if(rocblas_FOUND)
    message(STATUS "  rocBLAS: Enabled")
endif()
if(MIOpen_FOUND)
    message(STATUS "  MIOpen: Enabled")
endif()
