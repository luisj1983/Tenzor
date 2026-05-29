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

# ARM NEON kernel tests (compiles on all platforms; tests skip on non-ARM)
add_executable(test_neon_kernels test_neon_kernels.cpp)
target_link_libraries(test_neon_kernels PRIVATE tenzor_core GTest::gtest_main)
target_include_directories(test_neon_kernels PRIVATE ${PROJECT_SOURCE_DIR})
tenzor_discover_tests(test_neon_kernels LABELS "backends" DISCOVERY_TIMEOUT 60)

# Event API tests (backend-agnostic, skips unavailable backends)
add_executable(test_event_api test_event_api.cpp)
target_link_libraries(test_event_api PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_event_api DISCOVERY_TIMEOUT 60)

# MPS training tests (skips if MPS not available)
add_executable(test_mps_training test_mps_training.cpp)
target_link_libraries(test_mps_training PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_mps_training DISCOVERY_TIMEOUT 60)

# Backend kernel operations (14 tests using BackendTest fixture)
add_executable(test_backend_kernel_ops test_backend_kernel_ops.cpp)
target_link_libraries(test_backend_kernel_ops PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_backend_kernel_ops LABELS "backends" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Signed-INT8 regression for the CPU quantized_linear kernel (release-prep S3).
# Verifies the AVX2 / AVX-512 paths use signed widening + madd_epi16 rather
# than the (broken) unsigned _mm256_maddubs_epi16 / _mm512_dpbusd_epi32 path.
add_executable(test_quantized_linear_signed test_quantized_linear_signed.cpp)
target_link_libraries(test_quantized_linear_signed PRIVATE
    tenzor_core tenzor_backend_cpu GTest::gtest_main)
set_target_properties(test_quantized_linear_signed PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
tenzor_discover_tests(test_quantized_linear_signed
    LABELS "backends" DISCOVERY_TIMEOUT 60 BACKENDS cpu)

# Signed + asymmetric zero-point correctness for the CPU quantized_conv2d
# kernels (release-prep A3/A4): signed-safe dot_int8, full (q-zp) expansion,
# and input_zp padding in im2col. Ground-truth reference dequantizes by
# definition so it is independent of the kernel's internal decomposition.
add_executable(test_quantized_conv2d_signed_zp test_quantized_conv2d_signed_zp.cpp)
target_link_libraries(test_quantized_conv2d_signed_zp PRIVATE
    tenzor_core tenzor_backend_cpu GTest::gtest_main)
set_target_properties(test_quantized_conv2d_signed_zp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
tenzor_discover_tests(test_quantized_conv2d_signed_zp
    LABELS "backends" DISCOVERY_TIMEOUT 60 BACKENDS cpu)

# Consolidated CPU correctness regressions from the release audit (isin native
# bit-width, index_select non-contiguous, norm overflow scaling, ...).
add_executable(test_release_audit_cpu test_release_audit_cpu.cpp)
target_link_libraries(test_release_audit_cpu PRIVATE
    tenzor_core tenzor_backend_cpu GTest::gtest_main)
set_target_properties(test_release_audit_cpu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
tenzor_discover_tests(test_release_audit_cpu
    LABELS "backends" DISCOVERY_TIMEOUT 60 BACKENDS cpu)

# S13 dtype-preservation regressions: sampler/loss/FFT kernels that used to
# silently change output dtype away from input dtype.
add_executable(test_dtype_preservation test_dtype_preservation.cpp)
target_link_libraries(test_dtype_preservation PRIVATE
    tenzor_core tenzor_backend_cpu GTest::gtest_main)
set_target_properties(test_dtype_preservation PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
tenzor_discover_tests(test_dtype_preservation
    LABELS "backends" DISCOVERY_TIMEOUT 60 BACKENDS cpu)

# S17 CPU thread / SIMD unification regression: deprecates legacy CPUInfo,
# unifies OMP thread defaults, exercises CPUBackend::event_elapsed_ms wall-clock.
add_executable(test_thread_simd_unification test_thread_simd_unification.cpp)
target_link_libraries(test_thread_simd_unification PRIVATE
    tenzor_core tenzor_backend_cpu GTest::gtest_main)
set_target_properties(test_thread_simd_unification PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
tenzor_discover_tests(test_thread_simd_unification
    LABELS "backends" DISCOVERY_TIMEOUT 60 BACKENDS cpu)

# S26 OCL ICD workaround regression: verifies init.cpp's AMD-OCL probe
# workaround glob-discovers libintelocl.so, honours TENZOR_OCL_ICD_PATH, and
# fails silently when no oneAPI install is present.
add_executable(test_init_ocl_workaround test_init_ocl_workaround.cpp)
target_link_libraries(test_init_ocl_workaround PRIVATE
    tenzor_core GTest::gtest_main)
set_target_properties(test_init_ocl_workaround PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
tenzor_discover_tests(test_init_ocl_workaround
    LABELS "backends" DISCOVERY_TIMEOUT 60)

# S14 indexing/reduction dtype gap fills: index_select/gather/scatter were
# missing Int16/UInt16/UInt32/UInt64/Complex64/Complex128, and sum_kernel
# was missing Int8/UInt8/Int16/UInt16/UInt32/UInt64/Bool. Includes the
# Int8 200x127 overflow regression for the public sum() promotion path.
add_executable(test_indexing_reduction_dtypes test_indexing_reduction_dtypes.cpp)
target_link_libraries(test_indexing_reduction_dtypes PRIVATE
    tenzor_core tenzor_backend_cpu GTest::gtest_main)
set_target_properties(test_indexing_reduction_dtypes PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
tenzor_discover_tests(test_indexing_reduction_dtypes
    LABELS "backends" DISCOVERY_TIMEOUT 60 BACKENDS cpu)
