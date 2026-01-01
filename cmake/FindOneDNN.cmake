# FindOneDNN.cmake
# Find Intel oneDNN (oneAPI Deep Neural Network Library)
#
# This module defines:
#   ONEDNN_FOUND        - True if oneDNN was found
#   ONEDNN_INCLUDE_DIRS - Include directories for oneDNN
#   ONEDNN_LIBRARIES    - Libraries to link against
#   ONEDNN_VERSION      - Version string

# First try to find via CMake config (preferred)
find_package(dnnl CONFIG QUIET)

if(dnnl_FOUND)
    set(ONEDNN_FOUND TRUE)
    set(ONEDNN_LIBRARIES DNNL::dnnl)
    get_target_property(ONEDNN_INCLUDE_DIRS DNNL::dnnl INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "Found oneDNN via CMake config: ${dnnl_DIR}")
    if(DEFINED dnnl_VERSION)
        set(ONEDNN_VERSION ${dnnl_VERSION})
        message(STATUS "oneDNN version: ${ONEDNN_VERSION}")
    endif()
    return()
endif()

# Fallback: Manual search
set(ONEDNN_SEARCH_PATHS
    /usr
    /usr/local
    /opt/intel/onednn
    /opt/intel/oneapi/dnnl/latest
    $ENV{DNNL_ROOT}
    $ENV{ONEDNN_ROOT}
)

# Find include directory
find_path(ONEDNN_INCLUDE_DIR
    NAMES dnnl.hpp oneapi/dnnl/dnnl.hpp
    PATHS ${ONEDNN_SEARCH_PATHS}
    PATH_SUFFIXES include
)

# Find library
find_library(ONEDNN_LIBRARY
    NAMES dnnl mkldnn
    PATHS ${ONEDNN_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64
)

# Handle results
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OneDNN
    REQUIRED_VARS ONEDNN_LIBRARY ONEDNN_INCLUDE_DIR
)

if(ONEDNN_FOUND)
    set(ONEDNN_LIBRARIES ${ONEDNN_LIBRARY})
    set(ONEDNN_INCLUDE_DIRS ${ONEDNN_INCLUDE_DIR})

    # Create imported target
    if(NOT TARGET OneDNN::dnnl)
        add_library(OneDNN::dnnl UNKNOWN IMPORTED)
        set_target_properties(OneDNN::dnnl PROPERTIES
            IMPORTED_LOCATION "${ONEDNN_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ONEDNN_INCLUDE_DIR}"
        )
    endif()

    message(STATUS "Found oneDNN: ${ONEDNN_LIBRARY}")
endif()

mark_as_advanced(ONEDNN_INCLUDE_DIR ONEDNN_LIBRARY)
