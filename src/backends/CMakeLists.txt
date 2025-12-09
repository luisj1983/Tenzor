# Backends directory - build each backend as a separate shared library

# Always build CPU backend (default)
add_subdirectory(cpu)

# Optional GPU backends
if(TENZOR_BUILD_CUDA)
    add_subdirectory(cuda)
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

if(TENZOR_BUILD_METAL)
    add_subdirectory(metal)
endif()

if(TENZOR_BUILD_WEBGPU)
    add_subdirectory(webgpu)
endif()
