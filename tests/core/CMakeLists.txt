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
    tenzor_discover_tests(test_memory_manager DISCOVERY_TIMEOUT 30)
    tenzor_discover_tests(test_pinned_allocator DISCOVERY_TIMEOUT 30)
    tenzor_discover_tests(test_transfer_engine DISCOVERY_TIMEOUT 30)
    tenzor_discover_tests(test_transfer_benchmark DISCOVERY_TIMEOUT 30)
    tenzor_discover_tests(test_offload_engine DISCOVERY_TIMEOUT 30)
    tenzor_discover_tests(test_parameter_offload DISCOVERY_TIMEOUT 30)
    tenzor_discover_tests(test_cuda_training DISCOVERY_TIMEOUT 30)
endif()
