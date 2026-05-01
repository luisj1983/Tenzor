# Integration Tests CMakeLists.txt
# Comprehensive end-to-end tests for complete workflows

# Training loops integration tests
add_executable(test_training_loops
    test_training_loops.cpp
)

target_link_libraries(test_training_loops PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_training_loops DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT} LABELS "integration")

# Model persistence integration tests
add_executable(test_model_persistence
    test_model_persistence.cpp
)

target_link_libraries(test_model_persistence PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_model_persistence DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT} LABELS "integration")

# Cross-backend integration tests
add_executable(test_cross_backend
    test_cross_backend.cpp
)

target_link_libraries(test_cross_backend PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_cross_backend DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT} LABELS "integration")

# Attention contract dispatch-table audit (M9 — enforces every backend
# registers all required attention OpIds; see docs/internals/attention-contract.md).
add_executable(test_attention_contract
    test_attention_contract.cpp
)

target_link_libraries(test_attention_contract PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_attention_contract DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT} LABELS "integration")

# Attention autograd grad-flow tests (M9 — catches the audit's C1-C6 dead
# training path / severed grad_fn class of bugs).
add_executable(test_attention_autograd
    test_attention_autograd.cpp
)

target_link_libraries(test_attention_autograd PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_attention_autograd DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT} LABELS "integration")

# Data pipeline integration tests
add_executable(test_data_pipeline
    test_data_pipeline.cpp
)

target_link_libraries(test_data_pipeline PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_data_pipeline DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT} LABELS "integration")

# Model zoo integration tests
add_executable(test_model_zoo
    test_model_zoo.cpp
)

target_link_libraries(test_model_zoo PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_model_zoo DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT} LABELS "integration")

# Optimization integration tests
add_executable(test_optimization
    test_optimization.cpp
)

target_link_libraries(test_optimization PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_optimization DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT} LABELS "integration")

# DataParallel multi-GPU integration tests
add_executable(test_data_parallel_integration
    test_data_parallel.cpp
)

target_link_libraries(test_data_parallel_integration PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_data_parallel_integration DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT} LABELS "integration")

# Distributed training tests
add_executable(test_distributed
    test_distributed.cpp
)

target_link_libraries(test_distributed PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Propagate TENZOR_HAS_MPI and link MPI::MPI_CXX so the MPIBackend-specific
# test block in test_distributed.cpp compiles. The macro is PRIVATE on
# tenzor_core, so test binaries have to opt in explicitly.
if(TENZOR_BUILD_MPI)
    find_package(MPI QUIET COMPONENTS CXX)
    if(MPI_CXX_FOUND)
        target_compile_definitions(test_distributed PRIVATE TENZOR_HAS_MPI)
        target_link_libraries(test_distributed PRIVATE MPI::MPI_CXX)
    endif()
endif()

# Distributed tests require special setup (multi-process environment)
# Tests will automatically skip if RANK/WORLD_SIZE environment variables are not set
gtest_discover_tests(test_distributed DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT} LABELS "integration")

# Multi-GPU integration tests (CUDA required)
if(TENZOR_BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_executable(test_multi_gpu
        test_multi_gpu.cpp
    )

    target_link_libraries(test_multi_gpu PRIVATE
        tenzor_core
        GTest::gtest_main
        CUDA::cudart
    )

    target_include_directories(test_multi_gpu PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_compile_definitions(test_multi_gpu PRIVATE
        TENZOR_USE_CUDA
    )

    # Multi-GPU tests capped at maximum test timeout
    gtest_discover_tests(test_multi_gpu DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT} LABELS "integration")
endif()

# ============================================================================
# Multi-backend integration tests (Phase 4)
# ============================================================================

add_executable(test_data_pipeline_multidtype test_data_pipeline_multidtype.cpp)
target_link_libraries(test_data_pipeline_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_data_pipeline_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT} LABELS "integration")

add_executable(test_model_persistence_multidtype test_model_persistence_multidtype.cpp)
target_link_libraries(test_model_persistence_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_model_persistence_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT} LABELS "integration")

add_executable(test_model_zoo_multidtype test_model_zoo_multidtype.cpp)
target_link_libraries(test_model_zoo_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_model_zoo_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT} LABELS "integration")

add_executable(test_optimization_multidtype test_optimization_multidtype.cpp)
target_link_libraries(test_optimization_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_optimization_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT} LABELS "integration")

# Note: Existing integration tests
# - test_nn.cpp
# - test_training.cpp
# - test_cuda_training.cpp (if CUDA enabled)

# Model backend parity tests (end-to-end forward pass parity for small models)
add_executable(test_model_backend_parity test_model_backend_parity.cpp)
target_link_libraries(test_model_backend_parity PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_model_backend_parity DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT} LABELS "integration")
