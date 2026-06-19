# Backends directory - build each backend as a separate shared library

# Always build CPU backend (default)
add_subdirectory(cpu)

# Optional GPU backends
if(TENZOR_BUILD_CUDA)
    # Probe for a CUDA compiler BEFORE entering the subdirectory. The CUDA
    # backend's CMakeLists does enable_language(CUDA) + find_package(CUDAToolkit
    # REQUIRED), both of which abort the ENTIRE configure on a machine without
    # CUDA. Since the GPU backends default ON, that broke every source build that
    # lacks CUDA. Degrade gracefully like any optional backend: if no CUDA
    # compiler is found, warn and auto-disable instead of failing the configure.
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        add_subdirectory(cuda)
    else()
        message(WARNING
            "TENZOR_BUILD_CUDA is ON but no CUDA compiler was found; disabling "
            "the CUDA backend. Install the CUDA Toolkit or pass "
            "-DTENZOR_BUILD_CUDA=OFF to silence this warning.")
        set(TENZOR_BUILD_CUDA OFF CACHE BOOL "Build CUDA backend" FORCE)
    endif()
endif()

if(TENZOR_BUILD_ROCM)
    add_subdirectory(rocm)
endif()

if(TENZOR_BUILD_ONEAPI)
    add_subdirectory(oneapi)
endif()

if(TENZOR_BUILD_VULKAN)
    add_subdirectory(vulkan)
endif()

if(TENZOR_BUILD_MPS AND APPLE)
    add_subdirectory(mps)
endif()

