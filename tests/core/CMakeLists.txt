# =============================================================================
# tests/core/CMakeLists.txt — Core CUDA memory/transfer tests
# =============================================================================

if(TENZOR_BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)

    # Memory Manager tests (ZeRO Phase 1)
    add_executable(test_memory_manager test_memory_manager.cpp)
    target_link_libraries(test_memory_manager PRIVATE tenzor_core GTest::gtest_main CUDA::cudart)
    target_include_directories(test_memory_manager PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(test_memory_manager PRIVATE TENZOR_USE_CUDA)

    # Transfer Engine tests (ZeRO Phase 1)
    add_executable(test_transfer_engine test_transfer_engine.cpp)
    target_link_libraries(test_transfer_engine PRIVATE tenzor_core GTest::gtest_main CUDA::cudart)
    target_include_directories(test_transfer_engine PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(test_transfer_engine PRIVATE TENZOR_USE_CUDA)

    # Pinned Allocator tests (ZeRO Phase 1)
    add_executable(test_pinned_allocator test_pinned_allocator.cpp)
    target_link_libraries(test_pinned_allocator PRIVATE tenzor_core GTest::gtest_main CUDA::cudart)
    target_include_directories(test_pinned_allocator PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(test_pinned_allocator PRIVATE TENZOR_USE_CUDA)

    # Transfer Bandwidth Benchmark (ZeRO Phase 1)
    add_executable(test_transfer_benchmark test_transfer_benchmark.cpp)
    target_link_libraries(test_transfer_benchmark PRIVATE tenzor_core GTest::gtest_main CUDA::cudart)
    target_include_directories(test_transfer_benchmark PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(test_transfer_benchmark PRIVATE TENZOR_USE_CUDA)

    # Phase 2: Offload Engine tests
    add_executable(test_offload_engine test_offload_engine.cpp)
    target_link_libraries(test_offload_engine PRIVATE tenzor_core GTest::gtest_main CUDA::cudart)
    target_include_directories(test_offload_engine PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(test_offload_engine PRIVATE TENZOR_USE_CUDA)

    # Phase 2: Offload Engine Diagnostic tests
    add_executable(test_offload_engine_diagnostic test_offload_engine_diagnostic.cpp)
    target_link_libraries(test_offload_engine_diagnostic PRIVATE tenzor_core GTest::gtest_main CUDA::cudart)
    target_include_directories(test_offload_engine_diagnostic PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(test_offload_engine_diagnostic PRIVATE TENZOR_USE_CUDA)

    # Phase 2: Offload Engine Stress tests (larger than GPU memory)
    add_executable(test_offload_engine_stress test_offload_engine_stress.cpp)
    target_link_libraries(test_offload_engine_stress PRIVATE tenzor_core GTest::gtest_main CUDA::cudart)
    target_include_directories(test_offload_engine_stress PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(test_offload_engine_stress PRIVATE TENZOR_USE_CUDA)

    # Phase 2: Parameter Offloading API tests
    add_executable(test_parameter_offload ${CMAKE_CURRENT_SOURCE_DIR}/../nn/test_offload.cpp)
    target_link_libraries(test_parameter_offload PRIVATE tenzor_core GTest::gtest_main CUDA::cudart)
    target_include_directories(test_parameter_offload PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(test_parameter_offload PRIVATE TENZOR_USE_CUDA)

    # CUDA training tests
    add_executable(test_cuda_training ${CMAKE_CURRENT_SOURCE_DIR}/../integration/test_cuda_training.cpp)
    target_link_libraries(test_cuda_training PRIVATE tenzor_core GTest::gtest_main CUDA::cudart)
    target_include_directories(test_cuda_training PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    set_target_properties(test_cuda_training PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # ============================================================================
    # Test Registration
    # ============================================================================
    tenzor_discover_tests(test_memory_manager LABELS "core" DISCOVERY_TIMEOUT 60)
    tenzor_discover_tests(test_pinned_allocator LABELS "core" DISCOVERY_TIMEOUT 60)
    tenzor_discover_tests(test_transfer_engine LABELS "core" DISCOVERY_TIMEOUT 60)
    tenzor_discover_tests(test_transfer_benchmark LABELS "core" DISCOVERY_TIMEOUT 60)
    tenzor_discover_tests(test_offload_engine LABELS "core" DISCOVERY_TIMEOUT 60)
    tenzor_discover_tests(test_parameter_offload LABELS "core" DISCOVERY_TIMEOUT 60)
    tenzor_discover_tests(test_cuda_training LABELS "core" DISCOVERY_TIMEOUT 60)
endif()

# =============================================================================
# CPU-only core tests (new ops)
# =============================================================================
tenzor_add_test(NAME test_stft SOURCE test_stft.cpp LABELS core)
tenzor_add_test(NAME test_cdist SOURCE test_cdist.cpp LABELS core)
tenzor_add_test(NAME test_multinomial SOURCE test_multinomial.cpp LABELS core)
tenzor_add_test(NAME test_bernoulli SOURCE test_bernoulli.cpp LABELS core)
tenzor_add_test(NAME test_histogram SOURCE test_histogram.cpp LABELS core)
tenzor_add_test(NAME test_bucketize SOURCE test_bucketize.cpp LABELS core)
tenzor_add_test(NAME test_memory_profiler SOURCE test_memory_profiler.cpp LABELS core)
tenzor_add_test(NAME test_dimname SOURCE test_dimname.cpp LABELS core)
tenzor_add_test(NAME test_sparse_bsr SOURCE test_sparse_bsr.cpp LABELS core)
tenzor_add_test(NAME test_quantized_dtype SOURCE test_quantized_dtype.cpp LABELS core)
tenzor_add_test(NAME test_view_tracking SOURCE test_view_tracking.cpp LABELS core)
tenzor_add_test(NAME test_dtype_promotion SOURCE test_dtype_promotion.cpp LABELS core)
tenzor_add_test(NAME test_alias_detection SOURCE test_alias_detection.cpp LABELS core)
tenzor_add_test(NAME test_from_blob_overflow SOURCE test_from_blob_overflow.cpp LABELS core)
tenzor_add_test(NAME test_dtype_capability_queries SOURCE test_dtype_capability_queries.cpp LABELS core)
tenzor_add_test(NAME test_mutation_version SOURCE test_mutation_version.cpp LABELS core)
tenzor_add_test(NAME test_dispatch_interceptor_stack SOURCE test_dispatch_interceptor_stack.cpp LABELS core)
tenzor_add_test(NAME test_dispatch_coverage SOURCE test_dispatch_coverage.cpp LABELS core)
tenzor_add_test(NAME test_shape_info_snapshot SOURCE test_shape_info_snapshot.cpp LABELS core)
# test_shape_info_snapshot exercises TensorImpl internal mutators
# (mutable_shape / mutable_strides / set_offset) to verify that the
# snapshot cache is invalidated on in-place mutation, so it needs
# the TENZOR_INTERNAL define that unlocks those accessors.
target_compile_definitions(test_shape_info_snapshot PRIVATE TENZOR_INTERNAL)
tenzor_add_test(NAME test_dlpack_interop SOURCE test_dlpack_interop.cpp LABELS core)
tenzor_add_test(NAME test_inplace_version_detection SOURCE test_inplace_version_detection.cpp LABELS core)
tenzor_add_test(NAME test_laplace_distribution SOURCE test_laplace_distribution.cpp LABELS core)
tenzor_add_test(NAME test_distributions_advanced SOURCE test_distributions_advanced.cpp LABELS core)
tenzor_add_test(NAME test_distributions_gap_fill SOURCE test_distributions_gap_fill.cpp LABELS core)
tenzor_add_test(NAME test_sddmm SOURCE test_sddmm.cpp LABELS core)
tenzor_add_test(NAME test_vulkan_flash_attention_parity SOURCE test_vulkan_flash_attention_parity.cpp LABELS core)
tenzor_add_test(NAME test_new_features SOURCE test_new_features.cpp LABELS core)

# =============================================================================
# Multi-backend multi-dtype core tests (Batch A)
# =============================================================================
add_executable(test_cdist_multidtype test_cdist_multidtype.cpp)
target_link_libraries(test_cdist_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_cdist_multidtype LABELS "core" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_stft_multidtype test_stft_multidtype.cpp)
target_link_libraries(test_stft_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_stft_multidtype LABELS "core" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_histogram_multidtype test_histogram_multidtype.cpp)
target_link_libraries(test_histogram_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_histogram_multidtype LABELS "core" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_multinomial_multidtype test_multinomial_multidtype.cpp)
target_link_libraries(test_multinomial_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_multinomial_multidtype LABELS "core" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_bernoulli_multidtype test_bernoulli_multidtype.cpp)
target_link_libraries(test_bernoulli_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_bernoulli_multidtype LABELS "core" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_bucketize_multidtype test_bucketize_multidtype.cpp)
target_link_libraries(test_bucketize_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_bucketize_multidtype LABELS "core" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_sddmm_multidtype test_sddmm_multidtype.cpp)
target_link_libraries(test_sddmm_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_sddmm_multidtype LABELS "core" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_laplace_distribution_multidtype test_laplace_distribution_multidtype.cpp)
target_link_libraries(test_laplace_distribution_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_laplace_distribution_multidtype LABELS "core" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Phase 4.2 Batch B: dtype/type system
add_executable(test_dtype_promotion_multidtype test_dtype_promotion_multidtype.cpp)
target_link_libraries(test_dtype_promotion_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_dtype_promotion_multidtype LABELS "core" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_dtype_capability_queries_multidtype test_dtype_capability_queries_multidtype.cpp)
target_link_libraries(test_dtype_capability_queries_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_dtype_capability_queries_multidtype LABELS "core" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_quantized_dtype_multidtype test_quantized_dtype_multidtype.cpp)
target_link_libraries(test_quantized_dtype_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_quantized_dtype_multidtype LABELS "core" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_dimname_multidtype test_dimname_multidtype.cpp)
target_link_libraries(test_dimname_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_dimname_multidtype LABELS "core" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_dlpack_interop_multidtype test_dlpack_interop_multidtype.cpp)
target_link_libraries(test_dlpack_interop_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_dlpack_interop_multidtype LABELS "core" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_from_blob_overflow_multidtype test_from_blob_overflow_multidtype.cpp)
target_link_libraries(test_from_blob_overflow_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_from_blob_overflow_multidtype LABELS "core" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_distributions_advanced_multidtype test_distributions_advanced_multidtype.cpp)
target_link_libraries(test_distributions_advanced_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_distributions_advanced_multidtype LABELS "core" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_new_features_multidtype test_new_features_multidtype.cpp)
target_link_libraries(test_new_features_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_new_features_multidtype LABELS "core" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Phase 4.2 Batch C: mutation/views/dispatch
add_executable(test_view_tracking_multidtype test_view_tracking_multidtype.cpp)
target_link_libraries(test_view_tracking_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_view_tracking_multidtype LABELS "core" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_alias_detection_multidtype test_alias_detection_multidtype.cpp)
target_link_libraries(test_alias_detection_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_alias_detection_multidtype LABELS "core" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_mutation_version_multidtype test_mutation_version_multidtype.cpp)
target_link_libraries(test_mutation_version_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_mutation_version_multidtype LABELS "core" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_inplace_version_detection_multidtype test_inplace_version_detection_multidtype.cpp)
target_link_libraries(test_inplace_version_detection_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_inplace_version_detection_multidtype LABELS "core" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_dispatch_interceptor_stack_multidtype test_dispatch_interceptor_stack_multidtype.cpp)
target_link_libraries(test_dispatch_interceptor_stack_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_dispatch_interceptor_stack_multidtype LABELS "core" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_dispatch_coverage_multidtype test_dispatch_coverage_multidtype.cpp)
target_link_libraries(test_dispatch_coverage_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_dispatch_coverage_multidtype LABELS "core" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_shape_info_snapshot_multidtype test_shape_info_snapshot_multidtype.cpp)
target_link_libraries(test_shape_info_snapshot_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_shape_info_snapshot_multidtype LABELS "core" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_memory_profiler_multidtype test_memory_profiler_multidtype.cpp)
target_link_libraries(test_memory_profiler_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_memory_profiler_multidtype LABELS "core" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_sparse_bsr_multidtype test_sparse_bsr_multidtype.cpp)
target_link_libraries(test_sparse_bsr_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_sparse_bsr_multidtype LABELS "core" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
