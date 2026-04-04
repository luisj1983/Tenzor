# =============================================================================
# tests/backends/CMakeLists.txt — Backend-specific kernel and threading tests
# =============================================================================

# CUDA-specific tests
if(TENZOR_BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)

    # CUDA kernel tests
    add_executable(test_cuda_kernels test_cuda_kernels.cpp)
    target_link_libraries(test_cuda_kernels PRIVATE
        tenzor_core GTest::gtest_main CUDA::cudart)
    target_include_directories(test_cuda_kernels PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    set_target_properties(test_cuda_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # FP16/BF16 kernel tests
    add_executable(test_fp16_kernels test_fp16_kernels.cpp)
    target_link_libraries(test_fp16_kernels PRIVATE
        tenzor_core tenzor_backend_cuda GTest::gtest_main CUDA::cudart)
    target_include_directories(test_fp16_kernels PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    set_target_properties(test_fp16_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # Test registration
    tenzor_discover_tests(test_cuda_kernels DISCOVERY_TIMEOUT 60)
    tenzor_discover_tests(test_fp16_kernels DISCOVERY_TIMEOUT 60)
endif()

# ROCm-specific tests
if(TENZOR_BUILD_ROCM)
    find_package(hip REQUIRED)

    # ROCm backend tests
    add_executable(test_rocm_backend test_rocm_backend.cpp)
    target_link_libraries(test_rocm_backend PRIVATE tenzor_core GTest::gtest_main)

    # ROCm kernel tests (comprehensive)
    add_executable(test_rocm_kernels test_rocm_kernels.cpp)
    target_link_libraries(test_rocm_kernels PRIVATE tenzor_core GTest::gtest_main hip::host)
    target_include_directories(test_rocm_kernels PRIVATE ${ROCM_PATH}/include)

    # ROCm FFT tests
    add_executable(test_rocm_fft test_rocm_fft.cpp)
    target_link_libraries(test_rocm_fft PRIVATE tenzor_core GTest::gtest_main)

    # ROCm detection kernel tests
    add_executable(test_rocm_detection_kernels test_rocm_detection_kernels.cpp)
    target_link_libraries(test_rocm_detection_kernels PRIVATE tenzor_core GTest::gtest_main hip::host)
    target_include_directories(test_rocm_detection_kernels PRIVATE ${ROCM_PATH}/include)

    # ROCm reduction tests
    add_executable(test_rocm_reduction test_rocm_reduction.cpp)
    target_link_libraries(test_rocm_reduction PRIVATE tenzor_core GTest::gtest_main)

    # ROCm RNN kernel tests
    add_executable(test_rocm_rnn_kernels test_rocm_rnn_kernels.cpp)
    target_link_libraries(test_rocm_rnn_kernels PRIVATE tenzor_core GTest::gtest_main hip::host)
    target_include_directories(test_rocm_rnn_kernels PRIVATE ${ROCM_PATH}/include)

    # ROCm initialization can be slow, use 60 second timeout
    tenzor_discover_tests(test_rocm_backend DISCOVERY_TIMEOUT 60)
    tenzor_discover_tests(test_rocm_kernels DISCOVERY_TIMEOUT 60)
    tenzor_discover_tests(test_rocm_fft DISCOVERY_TIMEOUT 60)
    tenzor_discover_tests(test_rocm_detection_kernels DISCOVERY_TIMEOUT 60)
    tenzor_discover_tests(test_rocm_reduction DISCOVERY_TIMEOUT 60)
    tenzor_discover_tests(test_rocm_rnn_kernels DISCOVERY_TIMEOUT 60)
endif()

# Sparse GPU tests (works with any GPU backend: CUDA or ROCm)
if(TENZOR_BUILD_CUDA OR TENZOR_BUILD_ROCM)
    add_executable(test_sparse_gpu test_sparse_gpu.cpp)
    target_link_libraries(test_sparse_gpu PRIVATE tenzor_core GTest::gtest_main)
    tenzor_discover_tests(test_sparse_gpu DISCOVERY_TIMEOUT 60)
endif()

# Vulkan-specific tests
if(TENZOR_BUILD_VULKAN)
    add_executable(test_vulkan_threading test_vulkan_threading.cpp)
    target_link_libraries(test_vulkan_threading PRIVATE tenzor_core GTest::gtest_main)
    tenzor_discover_tests(test_vulkan_threading DISCOVERY_TIMEOUT 60)

    find_package(Vulkan REQUIRED)

    add_executable(test_vulkan_allocator_merge test_vulkan_allocator_merge.cpp)
    target_link_libraries(test_vulkan_allocator_merge PRIVATE
        tenzor_core tenzor_backend_vulkan ${Vulkan_LIBRARIES} GTest::gtest_main)
    target_include_directories(test_vulkan_allocator_merge PRIVATE ${Vulkan_INCLUDE_DIRS})
    tenzor_discover_tests(test_vulkan_allocator_merge DISCOVERY_TIMEOUT 60)
endif()

# Event API tests (backend-agnostic, skips unavailable backends)
add_executable(test_event_api test_event_api.cpp)
target_link_libraries(test_event_api PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_event_api DISCOVERY_TIMEOUT 60)

# MPS training tests (skips if MPS not available)
add_executable(test_mps_training test_mps_training.cpp)
target_link_libraries(test_mps_training PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_mps_training DISCOVERY_TIMEOUT 60)
