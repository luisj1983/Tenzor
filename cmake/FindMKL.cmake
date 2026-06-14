# FindMKL.cmake
# Find Intel Math Kernel Library (MKL)
#
# This module defines:
#   MKL_FOUND        - True if MKL was found
#   MKL_INCLUDE_DIRS - Include directories for MKL
#   MKL_LIBRARIES    - Libraries to link against
#   MKL_VERSION      - Version string (if available)

# Force GNU threading to avoid conflicts with GCC's OpenMP (libgomp)
# Intel MKL's default intel_thread links against Intel OpenMP (libiomp5),
# which causes undefined behavior when mixed with libgomp
set(MKL_THREADING gnu_thread CACHE STRING "MKL threading layer")

# Use the LP64 interface (32-bit MKL_INT) consistently in BOTH the CMake-config
# and manual-fallback paths. The call sites pass plain `int`/`MKL_INT` to cblas_*,
# so an ILP64 (64-bit MKL_INT) interface would misread GEMM dimensions/strides.
# This must be set before find_package(MKL) so MKL::MKL picks the LP64 variant
# rather than its default.
set(MKL_INTERFACE_FULL intel_lp64 CACHE STRING "MKL integer interface (LP64 = 32-bit MKL_INT)")

# First try to find via CMake config (Intel oneAPI)
find_package(MKL CONFIG QUIET)

if(MKL_FOUND)
    set(MKL_LIBRARIES MKL::MKL)
    message(STATUS "Found MKL via CMake config: ${MKL_ROOT}")
    message(STATUS "  Threading: ${MKL_THREADING}")
    return()
endif()

# Fallback: Manual search using MKLROOT environment variable
set(MKL_SEARCH_PATHS
    $ENV{MKLROOT}
    /opt/intel/mkl
    /opt/intel/oneapi/mkl/latest
    /usr/local/mkl
    /usr
)

# Find include directory
find_path(MKL_INCLUDE_DIR
    NAMES mkl.h mkl_cblas.h
    PATHS ${MKL_SEARCH_PATHS}
    PATH_SUFFIXES include
)

# Determine library suffix based on architecture
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(MKL_LIB_DIR_SUFFIX "lib/intel64" "lib")
else()
    set(MKL_LIB_DIR_SUFFIX "lib/ia32" "lib")
endif()

# Find core library
find_library(MKL_CORE_LIBRARY
    NAMES mkl_core
    PATHS ${MKL_SEARCH_PATHS}
    PATH_SUFFIXES ${MKL_LIB_DIR_SUFFIX}
)

# Find interface library (LP64 = 32-bit MKL_INT, matching the config path and
# the plain-int cblas_* call sites; ILP64 would silently misread dimensions).
find_library(MKL_INTERFACE_LIBRARY
    NAMES mkl_intel_lp64
    PATHS ${MKL_SEARCH_PATHS}
    PATH_SUFFIXES ${MKL_LIB_DIR_SUFFIX}
)

# Find threading library (GNU or Intel)
# Prefer GNU threading for GCC compatibility
find_library(MKL_THREAD_LIBRARY
    NAMES mkl_gnu_thread mkl_intel_thread mkl_sequential
    PATHS ${MKL_SEARCH_PATHS}
    PATH_SUFFIXES ${MKL_LIB_DIR_SUFFIX}
)

# Handle results
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKL
    REQUIRED_VARS
        MKL_CORE_LIBRARY
        MKL_INTERFACE_LIBRARY
        MKL_THREAD_LIBRARY
        MKL_INCLUDE_DIR
)

if(MKL_FOUND)
    set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})

    # Build library list with proper linking order
    # MKL requires specific link order: interface -> threading -> core
    set(MKL_LIBRARIES
        ${MKL_INTERFACE_LIBRARY}
        ${MKL_THREAD_LIBRARY}
        ${MKL_CORE_LIBRARY}
    )

    # Add required system libraries
    find_package(Threads REQUIRED)
    list(APPEND MKL_LIBRARIES Threads::Threads)
    list(APPEND MKL_LIBRARIES m dl)

    # For GNU threading, we need OpenMP
    if(MKL_THREAD_LIBRARY MATCHES "gnu_thread")
        find_package(OpenMP QUIET)
        if(OpenMP_CXX_FOUND)
            list(APPEND MKL_LIBRARIES OpenMP::OpenMP_CXX)
        else()
            # Fallback to gomp
            list(APPEND MKL_LIBRARIES gomp)
        endif()
    endif()

    # Create imported target
    if(NOT TARGET MKL::MKL)
        add_library(MKL::MKL INTERFACE IMPORTED)
        set_target_properties(MKL::MKL PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES "${MKL_LIBRARIES}"
        )
    endif()

    message(STATUS "Found MKL:")
    message(STATUS "  Include: ${MKL_INCLUDE_DIR}")
    message(STATUS "  Core:    ${MKL_CORE_LIBRARY}")
    message(STATUS "  Interface: ${MKL_INTERFACE_LIBRARY}")
    message(STATUS "  Threading: ${MKL_THREAD_LIBRARY}")
endif()

mark_as_advanced(
    MKL_INCLUDE_DIR
    MKL_CORE_LIBRARY
    MKL_INTERFACE_LIBRARY
    MKL_THREAD_LIBRARY
)
