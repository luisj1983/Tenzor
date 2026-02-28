# OneAPI Backend - Intel CPU/GPU via SYCL
# Uses icpx from Intel oneAPI Base Toolkit

# Find Intel icpx compiler
find_program(ICPX_COMPILER NAMES icpx)
if(NOT ICPX_COMPILER)
    message(WARNING "OneAPI backend requires icpx from Intel oneAPI Base Toolkit")
    message(WARNING "Install from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html")
    return()
endif()

message(STATUS "OneAPI Backend: Found icpx at ${ICPX_COMPILER}")

set(SYCL_COMPILER ${ICPX_COMPILER})
set(SYCL_TARGET_ARCH "spir64")

# Derive SYCL paths from compiler location
get_filename_component(SYCL_BIN_DIR ${SYCL_COMPILER} DIRECTORY)
get_filename_component(SYCL_COMPILER_DIR ${SYCL_BIN_DIR} DIRECTORY)
set(SYCL_INCLUDE_DIR "${SYCL_COMPILER_DIR}/include")
set(SYCL_LIBRARY_DIR "${SYCL_COMPILER_DIR}/lib")

# Verify SYCL library exists
if(NOT EXISTS "${SYCL_LIBRARY_DIR}/libsycl.so")
    if(EXISTS "${SYCL_COMPILER_DIR}/lib64/libsycl.so")
        set(SYCL_LIBRARY_DIR "${SYCL_COMPILER_DIR}/lib64")
    else()
        message(WARNING "SYCL library not found at ${SYCL_LIBRARY_DIR}/libsycl.so")
    endif()
endif()

message(STATUS "OneAPI Backend: SYCL include: ${SYCL_INCLUDE_DIR}")
message(STATUS "OneAPI Backend: SYCL library: ${SYCL_LIBRARY_DIR}")

find_package(MKL QUIET)

# Check for oneMKL SYCL interface
set(ONEMKL_SYCL_FOUND FALSE)
if(MKL_FOUND)
    find_path(ONEMKL_SYCL_INCLUDE_DIR
        NAMES oneapi/mkl.hpp
        HINTS ${MKL_INCLUDE} ${SYCL_COMPILER_DIR}/include
        PATH_SUFFIXES include
    )
    if(ONEMKL_SYCL_INCLUDE_DIR)
        set(ONEMKL_SYCL_FOUND TRUE)
        message(STATUS "OneAPI Backend: oneMKL SYCL interface found at ${ONEMKL_SYCL_INCLUDE_DIR}")
    else()
        message(STATUS "OneAPI Backend: oneMKL SYCL interface not found - using native SYCL kernels")
    endif()
endif()

# SYCL kernel sources
set(ONEAPI_SYCL_SOURCES
    kernels/math.cpp
    kernels/activations.cpp
    kernels/reduction.cpp
    kernels/transform.cpp
    kernels/creation.cpp
    kernels/batchnorm.cpp
    kernels/conv2d.cpp
    kernels/pooling.cpp
    kernels/indexing.cpp
    kernels/embedding.cpp
    kernels/comparison.cpp
    kernels/im2col.cpp
    kernels/expand.cpp
    kernels/statistical.cpp
    kernels/fused_ops.cpp
    kernels/lstm.cpp
    kernels/gru.cpp
    kernels/vision.cpp
    kernels/quantization.cpp
    kernels/utilities.cpp
)

# Prepare compile flags
set(ONEMKL_COMPILE_FLAGS "")
if(ONEMKL_SYCL_FOUND)
    set(ONEMKL_COMPILE_FLAGS "-DTENZOR_HAS_ONEMKL" "-I${ONEMKL_SYCL_INCLUDE_DIR}")
endif()

# Collect all object files
set(ALL_OBJECTS "")

# Build SYCL kernel sources
foreach(SRC ${ONEAPI_SYCL_SOURCES})
    get_filename_component(OBJ_NAME ${SRC} NAME_WE)
    set(OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/${OBJ_NAME}.o")
    list(APPEND ALL_OBJECTS ${OBJ_FILE})

    add_custom_command(
        OUTPUT ${OBJ_FILE}
        COMMAND ${SYCL_COMPILER}
            -c ${CMAKE_CURRENT_SOURCE_DIR}/${SRC}
            -o ${OBJ_FILE}
            -std=c++23 -fPIC
            -fsycl -fsycl-targets=${SYCL_TARGET_ARCH}
            -O3 -DNDEBUG
            -I${CMAKE_SOURCE_DIR}/include
            ${ONEMKL_COMPILE_FLAGS}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SRC}
        COMMENT "Compiling SYCL kernel: ${SRC}"
        VERBATIM
    )
endforeach()

# Build backend glue sources
set(BACKEND_GLUE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/oneapi_backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/oneapi_kernel_registry.cpp
    ${CMAKE_SOURCE_DIR}/src/backend/oneapi_caching_allocator.cpp
)

foreach(SRC ${BACKEND_GLUE_SOURCES})
    get_filename_component(OBJ_NAME ${SRC} NAME_WE)
    set(OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/${OBJ_NAME}_backend.o")
    list(APPEND ALL_OBJECTS ${OBJ_FILE})

    add_custom_command(
        OUTPUT ${OBJ_FILE}
        COMMAND ${SYCL_COMPILER}
            -c ${SRC}
            -o ${OBJ_FILE}
            -std=c++23 -fPIC
            -fsycl -fsycl-targets=${SYCL_TARGET_ARCH}
            -O3 -DNDEBUG
            -DTENZOR_ONEAPI_BACKEND
            -I${CMAKE_SOURCE_DIR}/include
            ${ONEMKL_COMPILE_FLAGS}
        DEPENDS ${SRC}
        COMMENT "Compiling backend source: ${OBJ_NAME}"
        VERBATIM
    )
endforeach()

# Prepare link flags
set(ONEMKL_LINK_FLAGS "")
if(ONEMKL_SYCL_FOUND)
    get_filename_component(ONEMKL_LIB_DIR "${ONEMKL_SYCL_INCLUDE_DIR}/../lib" ABSOLUTE)
    # Use GNU threading layer instead of Intel threading to avoid cleanup symbol issues
    # Link order: mkl_sycl -> mkl_intel_ilp64 -> mkl_gnu_thread -> mkl_core -> gomp
    set(ONEMKL_LINK_FLAGS "-L${ONEMKL_LIB_DIR}" "-Wl,-rpath,${ONEMKL_LIB_DIR}" "-Wl,--no-as-needed" "-lmkl_sycl" "-lmkl_intel_ilp64" "-lmkl_gnu_thread" "-lmkl_core" "-lgomp" "-Wl,--as-needed")
endif()

# SYCL library linking
set(SYCL_LINK_FLAGS "-L${SYCL_LIBRARY_DIR}" "-lsycl" "-Wl,-rpath,${SYCL_LIBRARY_DIR}")

# Create shared library
set(ONEAPI_SO_OUTPUT "${CMAKE_BINARY_DIR}/bin/tenzor_backend_oneapi.so")
add_custom_command(
    OUTPUT ${ONEAPI_SO_OUTPUT}
    COMMAND ${SYCL_COMPILER}
        -shared -fPIC
        -fsycl -fsycl-targets=${SYCL_TARGET_ARCH}
        ${ALL_OBJECTS}
        -o ${ONEAPI_SO_OUTPUT}
        -Wl,-soname,tenzor_backend_oneapi.so
        -L${CMAKE_BINARY_DIR}/bin -ltenzor_core
        -Wl,-rpath,${CMAKE_BINARY_DIR}/bin
        ${SYCL_LINK_FLAGS}
        ${ONEMKL_LINK_FLAGS}
    DEPENDS ${ALL_OBJECTS}
    COMMENT "Linking oneAPI backend shared library"
    VERBATIM
)

# Custom target for building the shared library
add_custom_target(tenzor_backend_oneapi_build ALL
    DEPENDS ${ONEAPI_SO_OUTPUT}
)

# Ensure core library is built first
add_dependencies(tenzor_backend_oneapi_build tenzor_core)

# Create an imported library target
add_library(tenzor_backend_oneapi SHARED IMPORTED GLOBAL)
set_target_properties(tenzor_backend_oneapi PROPERTIES
    IMPORTED_LOCATION ${ONEAPI_SO_OUTPUT}
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/include"
)
add_dependencies(tenzor_backend_oneapi tenzor_backend_oneapi_build)

# Installation
install(FILES ${ONEAPI_SO_OUTPUT}
    DESTINATION lib/tenzor/backends
)

message(STATUS "OneAPI Backend: Configured successfully")
message(STATUS "  Compiler: ${SYCL_COMPILER}")
message(STATUS "  Target: ${SYCL_TARGET_ARCH} (Intel CPU/GPU)")
if(ONEMKL_SYCL_FOUND)
    message(STATUS "  oneMKL SYCL: Enabled")
else()
    message(STATUS "  oneMKL SYCL: Disabled")
endif()
