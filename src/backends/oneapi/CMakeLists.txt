# OneAPI Backend
# Requires Intel oneAPI Base Toolkit for Intel GPUs and CPUs

# Find Intel oneAPI
find_package(IntelDPCPP QUIET)
find_package(MKL QUIET)

if(NOT IntelDPCPP_FOUND)
    message(WARNING "Intel oneAPI DPCPP compiler not found. OneAPI backend will not be built.")
    message(WARNING "Install from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html")
    return()
endif()

message(STATUS "OneAPI Backend: Found Intel DPCPP")

set(ONEAPI_BACKEND_SOURCES
    oneapi_backend.cpp
    kernels/math.cpp
)

# Create OneAPI backend shared library
add_library(tenzor_backend_oneapi SHARED ${ONEAPI_BACKEND_SOURCES})

# Set target properties
set_target_properties(tenzor_backend_oneapi PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    OUTPUT_NAME "tenzor_backend_oneapi"
    PREFIX ""  # Remove 'lib' prefix on Unix
)

# Include directories
target_include_directories(tenzor_backend_oneapi PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Link against core library and oneAPI SYCL
target_link_libraries(tenzor_backend_oneapi PRIVATE
    tenzor_core
    IntelDPCPP::DPCPP
)

# Link oneMKL if available
if(MKL_FOUND)
    target_link_libraries(tenzor_backend_oneapi PRIVATE MKL::MKL)
    target_compile_definitions(tenzor_backend_oneapi PRIVATE TENZOR_HAS_ONEMKL)
    message(STATUS "OneAPI Backend: oneMKL support enabled")
endif()

# Find oneDNN (Intel's DNN library)
find_package(dnnl QUIET)
if(dnnl_FOUND)
    target_link_libraries(tenzor_backend_oneapi PRIVATE dnnl)
    target_compile_definitions(tenzor_backend_oneapi PRIVATE TENZOR_HAS_ONEDNN)
    message(STATUS "OneAPI Backend: oneDNN support enabled")
else()
    message(STATUS "OneAPI Backend: oneDNN not found, using built-in kernels")
endif()

# SYCL compiler options
target_compile_options(tenzor_backend_oneapi PRIVATE
    -fsycl
    -ffast-math
)

# Link options
target_link_options(tenzor_backend_oneapi PRIVATE
    -fsycl
)

# Device selection (CPU, GPU, FPGA)
# Users can specify with -DTENZOR_ONEAPI_DEVICES="cpu;gpu"
if(NOT DEFINED TENZOR_ONEAPI_DEVICES)
    set(TENZOR_ONEAPI_DEVICES "cpu;gpu")
endif()

foreach(DEVICE ${TENZOR_ONEAPI_DEVICES})
    target_compile_definitions(tenzor_backend_oneapi PRIVATE
        TENZOR_ONEAPI_${DEVICE}
    )
endforeach()

# Ahead-of-time compilation for specific devices
# Uncomment and modify for specific Intel GPU models
# target_compile_options(tenzor_backend_oneapi PRIVATE
#     -fsycl-targets=spir64_gen
#     -Xs "-device gen12lp"  # For Intel Iris Xe
# )

# Installation
install(TARGETS tenzor_backend_oneapi
    LIBRARY DESTINATION lib/tenzor/backends
    RUNTIME DESTINATION bin
)

message(STATUS "OneAPI Backend: Configured successfully")
message(STATUS "  Target Devices: ${TENZOR_ONEAPI_DEVICES}")
if(MKL_FOUND)
    message(STATUS "  oneMKL: Enabled")
endif()
if(dnnl_FOUND)
    message(STATUS "  oneDNN: Enabled")
endif()
