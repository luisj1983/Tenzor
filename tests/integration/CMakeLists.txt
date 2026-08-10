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

tenzor_discover_tests(test_training_loops DISCOVERY_TIMEOUT 60 LABELS "integration" BACKENDS multi)

# Model persistence integration tests
add_executable(test_model_persistence
    test_model_persistence.cpp
)

target_link_libraries(test_model_persistence PRIVATE
    tenzor_core
    GTest::gtest_main
)

tenzor_discover_tests(test_model_persistence DISCOVERY_TIMEOUT 60 LABELS "integration" BACKENDS cpu)

# Cross-backend integration tests
add_executable(test_cross_backend
    test_cross_backend.cpp
)

target_link_libraries(test_cross_backend PRIVATE
    tenzor_core
    GTest::gtest_main
)

tenzor_discover_tests(test_cross_backend DISCOVERY_TIMEOUT 60 LABELS "integration" BACKENDS multi)

# Attention contract dispatch-table audit (M9 — enforces every backend
# registers all required attention OpIds; see docs/internals/attention-contract.md).
add_executable(test_attention_contract
    test_attention_contract.cpp
)

target_link_libraries(test_attention_contract PRIVATE
    tenzor_core
    GTest::gtest_main
)

tenzor_discover_tests(test_attention_contract DISCOVERY_TIMEOUT 60 LABELS "integration" BACKENDS multi)

# Attention autograd grad-flow tests (M9 — catches the audit's C1-C6 dead
# training path / severed grad_fn class of bugs).
add_executable(test_attention_autograd
    test_attention_autograd.cpp
)

target_link_libraries(test_attention_autograd PRIVATE
    tenzor_core
    GTest::gtest_main
)

tenzor_discover_tests(test_attention_autograd DISCOVERY_TIMEOUT 60 LABELS "integration" BACKENDS multi)

# Phase 13b — Per-backend Philox seed-determinism TEST_P harness.
add_executable(test_attention_philox_multibackend
    test_attention_philox_multibackend.cpp
)

target_link_libraries(test_attention_philox_multibackend PRIVATE
    tenzor_core
    GTest::gtest_main
)

tenzor_discover_tests(test_attention_philox_multibackend DISCOVERY_TIMEOUT 60 LABELS "integration" BACKENDS multi)

# Cross-backend FlashAttention/FusedAttention parity sweep (M9 — verifies
# every enabled backend matches the CPU reference within F32 tolerance
# across {causal, GQA, contiguous/permuted} axes per the contract).
add_executable(test_attention_parity
    test_attention_parity.cpp
)

target_link_libraries(test_attention_parity PRIVATE
    tenzor_core
    GTest::gtest_main
)

tenzor_discover_tests(test_attention_parity DISCOVERY_TIMEOUT 60 LABELS "integration" BACKENDS multi)

# Data pipeline integration tests
add_executable(test_data_pipeline
    test_data_pipeline.cpp
)

target_link_libraries(test_data_pipeline PRIVATE
    tenzor_core
    GTest::gtest_main
)

tenzor_discover_tests(test_data_pipeline DISCOVERY_TIMEOUT 60 LABELS "integration" BACKENDS cpu)

# Model zoo integration tests
add_executable(test_model_zoo
    test_model_zoo.cpp
)

target_link_libraries(test_model_zoo PRIVATE
    tenzor_core
    GTest::gtest_main
)

tenzor_discover_tests(test_model_zoo DISCOVERY_TIMEOUT 60 LABELS "integration" BACKENDS multi)

# Optimization integration tests
add_executable(test_optimization
    test_optimization.cpp
)

target_link_libraries(test_optimization PRIVATE
    tenzor_core
    GTest::gtest_main
)

tenzor_discover_tests(test_optimization DISCOVERY_TIMEOUT 60 LABELS "integration" BACKENDS multi)

# DataParallel multi-GPU integration tests
add_executable(test_data_parallel_integration
    test_data_parallel.cpp
)

target_link_libraries(test_data_parallel_integration PRIVATE
    tenzor_core
    GTest::gtest_main
)

tenzor_discover_tests(test_data_parallel_integration DISCOVERY_TIMEOUT 60 LABELS "integration" BACKENDS cpu)

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
tenzor_discover_tests(test_distributed DISCOVERY_TIMEOUT 60 LABELS "integration" BACKENDS cpu)

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
    tenzor_discover_tests(test_multi_gpu DISCOVERY_TIMEOUT 60 LABELS "integration" BACKENDS cuda)
endif()

# ============================================================================
# Multi-backend integration tests (Phase 4)
# ============================================================================

add_executable(test_data_pipeline_multidtype test_data_pipeline_multidtype.cpp)
target_link_libraries(test_data_pipeline_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_data_pipeline_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT} LABELS "integration" BACKENDS multi)

add_executable(test_model_persistence_multidtype test_model_persistence_multidtype.cpp)
target_link_libraries(test_model_persistence_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_model_persistence_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT} LABELS "integration" BACKENDS multi)

add_executable(test_model_zoo_multidtype test_model_zoo_multidtype.cpp)
target_link_libraries(test_model_zoo_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_model_zoo_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT} LABELS "integration" BACKENDS multi)

add_executable(test_optimization_multidtype test_optimization_multidtype.cpp)
target_link_libraries(test_optimization_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_optimization_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT} LABELS "integration" BACKENDS multi)

add_executable(test_training_loops_multidtype test_training_loops_multidtype.cpp)
target_link_libraries(test_training_loops_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_training_loops_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT} LABELS "integration" BACKENDS multi)

# Note: Existing integration tests
# - test_nn.cpp
# - test_training.cpp
# - test_cuda_training.cpp (if CUDA enabled)

# Model backend parity tests (end-to-end forward pass parity for small models)
add_executable(test_model_backend_parity test_model_backend_parity.cpp)
target_link_libraries(test_model_backend_parity PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_model_backend_parity DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT} LABELS "integration")

# Program export round-trip (Phase 5.1)
add_executable(test_program_export test_program_export.cpp)
target_link_libraries(test_program_export PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_program_export DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT} LABELS "integration")

# Inf-G: audit-style quantization end-to-end tests (8 behaviour probes).
add_executable(test_quantization_e2e test_quantization_e2e.cpp)
target_link_libraries(test_quantization_e2e PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_quantization_e2e DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT} LABELS "integration;quantization" BACKENDS multi)

# G.13: eager-equivalence parity across lazy / JIT / lite execution paths.
# First cut of the audit's "every model bitwise-identical across paths" target —
# exercises representative modules (MLP, Conv stack) against lazy materialisation,
# CPU JIT (CompiledModule + jit::compile), and the lite runtime. GPU JIT
# variants and broader lite coverage roll in once Phase C.3 lands more op
# coverage in the lite exporter.
add_executable(test_eager_parity test_eager_parity.cpp)
target_link_libraries(test_eager_parity PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_eager_parity DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT} LABELS "integration;parity" BACKENDS multi)
