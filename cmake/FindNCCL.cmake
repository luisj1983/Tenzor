# FindNCCL.cmake
# Find NVIDIA Collective Communications Library (NCCL)
#
# This module defines:
#   NCCL_FOUND          - True if NCCL was found
#   NCCL_INCLUDE_DIRS   - Include directories for NCCL
#   NCCL_LIBRARIES      - Libraries to link against
#   NCCL_VERSION        - Version string (e.g. "2.18.3")
#   NCCL_MAJOR_VERSION  - Major version number
#   NCCL_MINOR_VERSION  - Minor version number
#   NCCL_PATCH_VERSION  - Patch version number
#
# The following environment variables are searched:
#   NCCL_ROOT / NCCL_DIR / NCCL_HOME - Root directory of NCCL installation
#   CUDA_HOME / CUDA_TOOLKIT_ROOT_DIR - CUDA toolkit directory (NCCL often bundled)
#
# Imported target:
#   NCCL::NCCL - Interface target with include dirs and libraries

# Build search paths from environment variables
set(_NCCL_SEARCH_PATHS)

if(DEFINED ENV{NCCL_ROOT})
    list(APPEND _NCCL_SEARCH_PATHS "$ENV{NCCL_ROOT}")
endif()
if(DEFINED ENV{NCCL_DIR})
    list(APPEND _NCCL_SEARCH_PATHS "$ENV{NCCL_DIR}")
endif()
if(DEFINED ENV{NCCL_HOME})
    list(APPEND _NCCL_SEARCH_PATHS "$ENV{NCCL_HOME}")
endif()
if(DEFINED ENV{CUDA_HOME})
    list(APPEND _NCCL_SEARCH_PATHS "$ENV{CUDA_HOME}")
endif()
if(DEFINED ENV{CUDA_TOOLKIT_ROOT_DIR})
    list(APPEND _NCCL_SEARCH_PATHS "$ENV{CUDA_TOOLKIT_ROOT_DIR}")
endif()

# Also check CUDAToolkit_ROOT if available from find_package(CUDAToolkit)
if(CUDAToolkit_ROOT)
    list(APPEND _NCCL_SEARCH_PATHS "${CUDAToolkit_ROOT}")
endif()

# Standard system paths
list(APPEND _NCCL_SEARCH_PATHS
    /usr
    /usr/local
    /usr/local/cuda
    /opt/nccl
)

# Find include directory
find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    PATHS ${_NCCL_SEARCH_PATHS}
    PATH_SUFFIXES include
)

# Find library
find_library(NCCL_LIBRARY
    NAMES nccl
    PATHS ${_NCCL_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
)

# Extract version from nccl.h
if(NCCL_INCLUDE_DIR AND EXISTS "${NCCL_INCLUDE_DIR}/nccl.h")
    # NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH are defined as macros in nccl.h
    file(READ "${NCCL_INCLUDE_DIR}/nccl.h" _NCCL_HEADER_CONTENTS)

    string(REGEX MATCH "#define[ \t]+NCCL_MAJOR[ \t]+([0-9]+)" _NCCL_MAJOR_MATCH "${_NCCL_HEADER_CONTENTS}")
    if(_NCCL_MAJOR_MATCH)
        set(NCCL_MAJOR_VERSION "${CMAKE_MATCH_1}")
    endif()

    string(REGEX MATCH "#define[ \t]+NCCL_MINOR[ \t]+([0-9]+)" _NCCL_MINOR_MATCH "${_NCCL_HEADER_CONTENTS}")
    if(_NCCL_MINOR_MATCH)
        set(NCCL_MINOR_VERSION "${CMAKE_MATCH_1}")
    endif()

    string(REGEX MATCH "#define[ \t]+NCCL_PATCH[ \t]+([0-9]+)" _NCCL_PATCH_MATCH "${_NCCL_HEADER_CONTENTS}")
    if(_NCCL_PATCH_MATCH)
        set(NCCL_PATCH_VERSION "${CMAKE_MATCH_1}")
    endif()

    if(NCCL_MAJOR_VERSION AND NCCL_MINOR_VERSION AND NCCL_PATCH_VERSION)
        set(NCCL_VERSION "${NCCL_MAJOR_VERSION}.${NCCL_MINOR_VERSION}.${NCCL_PATCH_VERSION}")
    elseif(NCCL_MAJOR_VERSION AND NCCL_MINOR_VERSION)
        set(NCCL_VERSION "${NCCL_MAJOR_VERSION}.${NCCL_MINOR_VERSION}")
    endif()
endif()

# Handle REQUIRED and QUIET arguments, set NCCL_FOUND
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL
    REQUIRED_VARS
        NCCL_LIBRARY
        NCCL_INCLUDE_DIR
    VERSION_VAR
        NCCL_VERSION
)

if(NCCL_FOUND)
    set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})
    set(NCCL_LIBRARIES ${NCCL_LIBRARY})

    # Create imported target
    if(NOT TARGET NCCL::NCCL)
        add_library(NCCL::NCCL UNKNOWN IMPORTED)
        set_target_properties(NCCL::NCCL PROPERTIES
            IMPORTED_LOCATION "${NCCL_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
        )
    endif()

    message(STATUS "Found NCCL: ${NCCL_LIBRARY}")
    message(STATUS "  Version:  ${NCCL_VERSION}")
    message(STATUS "  Include:  ${NCCL_INCLUDE_DIR}")
endif()

mark_as_advanced(
    NCCL_INCLUDE_DIR
    NCCL_LIBRARY
)
