cmake_minimum_required(VERSION 3.25)

# Find Google Test
find_package(GTest)

if(NOT GTest_FOUND)
    message(STATUS "Google Test not found, downloading...")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
endif()

# Include GoogleTest module for gtest_discover_tests
include(GoogleTest)

# Maximum test execution timeout (25 minutes)
set(TENZOR_TEST_TIMEOUT 1500)

# When ON, tests enumerate all devices per backend (e.g. cuda:0, cuda:1)
# instead of only testing device index 0.
option(TENZOR_TEST_ALL_DEVICES "Test all devices per backend" off)
if(TENZOR_TEST_ALL_DEVICES)
    add_compile_definitions(TENZOR_TEST_ALL_DEVICES)
endif()

# When ON, every ctest run propagates TENZOR_ONEAPI_ALLOW_CPU=1, which lets
# the OneAPI backend register Intel/AMD CPU SYCL devices (it skips them by
# default per the project policy that "OneAPI is the GPU backend"). Set this
# on machines where the only available SYCL runtime is the OpenCL CPU
# runtime — without it, every parametrized OneAPI test fails its SetUp with
# "Backend dispatch table not ready" because no device gets registered.
option(TENZOR_TEST_ONEAPI_ALLOW_CPU
       "Allow OneAPI tests to bind to a SYCL CPU device (debug/CI parity runs)" off)

# Custom function that wraps gtest_discover_tests
# Supports optional LABELS parameter for CTest label-based filtering
function(tenzor_discover_tests target)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "DISCOVERY_TIMEOUT;DISCOVERY_MODE" "PROPERTIES;LABELS")
    set(TIMEOUT ${ARG_DISCOVERY_TIMEOUT})
    if(NOT TIMEOUT)
        set(TIMEOUT 60)
    endif()
    set(PROPS ${ARG_PROPERTIES})
    set(MODE ${ARG_DISCOVERY_MODE})
    if(NOT MODE)
        set(MODE "POST_BUILD")
    endif()

    # Inject TIMEOUT into properties if not already specified
    list(FIND PROPS "TIMEOUT" _timeout_idx)
    if(_timeout_idx EQUAL -1)
        list(APPEND PROPS TIMEOUT ${TENZOR_TEST_TIMEOUT})
    endif()

    # Inject LABELS if provided.
    #
    # Multi-label note: to attach more than one label, escape the
    # semicolon at the call site (e.g. `LABELS "nn\;smoke"`). A bare
    # `LABELS "nn;smoke"` is split into a CMake list before reaching
    # gtest_discover_tests, which then writes `LABELS nn smoke` into
    # the generated tests.cmake — and CTest only picks up the first.
    # The escaped form `nn\;smoke` survives as a single literal string
    # all the way through to set_tests_properties, which then parses
    # the embedded `;` as a real list separator.
    if(ARG_LABELS)
        list(APPEND PROPS LABELS "${ARG_LABELS}")
    endif()

    # Inject TENZOR_ONEAPI_ALLOW_CPU=1 if the project was configured with
    # TENZOR_TEST_ONEAPI_ALLOW_CPU=ON. Honors any user-supplied ENVIRONMENT
    # entry by appending instead of overwriting.
    if(TENZOR_TEST_ONEAPI_ALLOW_CPU)
        list(FIND PROPS "ENVIRONMENT" _env_idx)
        if(_env_idx EQUAL -1)
            list(APPEND PROPS ENVIRONMENT "TENZOR_ONEAPI_ALLOW_CPU=1")
        else()
            math(EXPR _env_val_idx "${_env_idx} + 1")
            list(GET PROPS ${_env_val_idx} _existing_env)
            list(REMOVE_AT PROPS ${_env_val_idx})
            list(INSERT PROPS ${_env_val_idx} "${_existing_env};TENZOR_ONEAPI_ALLOW_CPU=1")
        endif()
    endif()

    # Now call gtest_discover_tests
    gtest_discover_tests(${target}
        DISCOVERY_TIMEOUT ${TIMEOUT}
        DISCOVERY_MODE ${MODE}
        PROPERTIES ${PROPS})
endfunction()

# Helper macro: create a test executable, link GTest, and register with CTest.
# Usage: tenzor_add_test(NAME test_foo SOURCE unit/test_foo.cpp [LIBS extra_lib] [LABELS unit])
macro(tenzor_add_test)
    cmake_parse_arguments(ARG "" "NAME;SOURCE" "LIBS;LABELS" ${ARGN})
    add_executable(${ARG_NAME} ${ARG_SOURCE})
    target_link_libraries(${ARG_NAME} PRIVATE tenzor_core GTest::gtest_main ${ARG_LIBS})
    if(ARG_LABELS)
        tenzor_discover_tests(${ARG_NAME} DISCOVERY_TIMEOUT 60 LABELS "${ARG_LABELS}")
    else()
        tenzor_discover_tests(${ARG_NAME} DISCOVERY_TIMEOUT 60)
    endif()
endmacro()

# ============================================================================
# Subdirectories
# ============================================================================
add_subdirectory(unit)
add_subdirectory(nn)
add_subdirectory(autograd)
add_subdirectory(ops)
add_subdirectory(backends)
add_subdirectory(backend)
add_subdirectory(core)
add_subdirectory(jit)
add_subdirectory(distributed)
add_subdirectory(serving)
add_subdirectory(lite)
add_subdirectory(integration)
add_subdirectory(backend_parity)
add_subdirectory(utils)
add_subdirectory(examples)
add_subdirectory(nested)
add_subdirectory(sparse)

# ============================================================================
# Combined unit test executable
# ============================================================================
add_executable(tenzor_unit_tests
    unit/test_tensor.cpp
    unit/test_device.cpp
    unit/test_ops.cpp
    unit/test_ops_additional.cpp
    unit/test_special_math.cpp
    unit/test_matrix_construction.cpp
    unit/test_ddp_unused_params.cpp
    unit/test_autograd.cpp
    test_autograd_transform.cpp
    unit/test_cpu_kernels.cpp
    unit/test_broadcasting.cpp
    unit/test_transforms.cpp
    unit/test_linear.cpp
    unit/test_losses.cpp
    unit/test_optimizers.cpp
    unit/test_callbacks.cpp
    unit/test_graph_optimizer.cpp
    unit/test_simd_dispatch.cpp
    unit/test_simd.cpp
    unit/test_oneapi_backend_loading.cpp
    unit/test_oneapi_operations.cpp
)

target_link_libraries(tenzor_unit_tests PRIVATE
    tenzor_core
    tenzor_backend_cpu
    GTest::gtest_main
)
foreach(backend cuda rocm oneapi vulkan)
    if(TARGET tenzor_backend_${backend})
        target_link_libraries(tenzor_unit_tests PRIVATE tenzor_backend_${backend})
    endif()
endforeach()

# gtest_discover_tests enables individual test case discovery for better sharding
# (Previously used add_test due to static destruction crash — now resolved)
tenzor_discover_tests(tenzor_unit_tests DISCOVERY_TIMEOUT 120 LABELS "unit")

# SIMD Benchmark tests (separate executable for benchmarking)
add_executable(tenzor_simd_benchmark
    unit/benchmark_simd.cpp
)

target_link_libraries(tenzor_simd_benchmark PRIVATE
    tenzor_core
    tenzor_backend_cpu
    GTest::gtest_main
)

# ============================================================================
# Root-level test executables (source files in tests/ root)
# ============================================================================

# CUDA scalar debug test (separate executable for quick debugging)
add_executable(test_cuda_scalar_debug test_cuda_scalar_debug.cpp)
target_link_libraries(test_cuda_scalar_debug PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_cuda_scalar_debug DISCOVERY_TIMEOUT 60)





# Legacy integration tests (kept for compatibility).
# integration/test_nn.cpp moved out into the dedicated test_nn_smoke
# executable below (audit 2026-05-02 I.1) so its small-MLP cases can be
# tagged for the smoke set without dragging in the heavier tests in
# this bundle.
add_executable(tenzor_integration_tests
    integration/test_training.cpp
    integration/test_debug_mlp.cpp
)

target_link_libraries(tenzor_integration_tests PRIVATE
    tenzor_core
    GTest::gtest_main
)

tenzor_discover_tests(tenzor_integration_tests DISCOVERY_TIMEOUT 60)

# Audit 2026-05-02 I.1: extract just the smallest-MLP integration cases
# from test_nn.cpp into a separate executable so they can be tagged for
# the smoke set (`ctest -L smoke`) without dragging in the heavier
# test_training.cpp + test_debug_mlp.cpp content of the parent bundle.
add_executable(test_nn_smoke
    integration/test_nn.cpp
)
target_link_libraries(test_nn_smoke PRIVATE
    tenzor_core
    GTest::gtest_main
)
tenzor_discover_tests(test_nn_smoke
    LABELS "integration\;smoke"
    DISCOVERY_TIMEOUT 60
)

# Conv1d tests
add_executable(test_conv1d test_conv1d.cpp)
target_link_libraries(test_conv1d PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_conv1d DISCOVERY_TIMEOUT 60)

# ConvTranspose2d tests
add_executable(test_convtranspose2d test_convtranspose2d.cpp)
target_link_libraries(test_convtranspose2d PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_convtranspose2d DISCOVERY_TIMEOUT 60)

# Minimal training test (debugging NaN issue)
add_executable(test_minimal_training test_minimal_training.cpp)
target_link_libraries(test_minimal_training PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_minimal_training DISCOVERY_TIMEOUT 60)

# Split operation tests
add_executable(test_split_operation test_split_operation.cpp)
target_link_libraries(test_split_operation PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_split_operation DISCOVERY_TIMEOUT 60)

# DType conversion tests
add_executable(test_dtype_conversion test_dtype_conversion.cpp)
target_link_libraries(test_dtype_conversion PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_dtype_conversion DISCOVERY_TIMEOUT 60)

# BMM autograd tests
add_executable(test_bmm_autograd test_bmm_autograd.cpp)
target_link_libraries(test_bmm_autograd PRIVATE tenzor_core GTest::gtest)
tenzor_discover_tests(test_bmm_autograd DISCOVERY_TIMEOUT 60)

# Linear reshape integration tests
add_executable(test_linear_reshape_integration test_linear_reshape_integration.cpp)
target_link_libraries(test_linear_reshape_integration PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_linear_reshape_integration DISCOVERY_TIMEOUT 60)

# Creation operations tests
add_executable(test_creation_ops test_creation_ops.cpp)
target_link_libraries(test_creation_ops PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_creation_ops DISCOVERY_TIMEOUT 60)

# Creation operations multi-dtype tests
add_executable(test_creation_ops_multidtype test_creation_ops_multidtype.cpp)
target_link_libraries(test_creation_ops_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_creation_ops_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Comparison operators tests
add_executable(test_comparison_operators test_comparison_operators.cpp)
target_link_libraries(test_comparison_operators PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_comparison_operators DISCOVERY_TIMEOUT 60)

# Checkpoint leaf fix test
add_executable(test_checkpoint_leaf_fix test_checkpoint_leaf_fix.cpp)
target_link_libraries(test_checkpoint_leaf_fix tenzor_core)
target_include_directories(test_checkpoint_leaf_fix PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME CheckpointLeafFix COMMAND test_checkpoint_leaf_fix)
set_tests_properties(CheckpointLeafFix PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT})

# Phase 9-followup #29: tolerance audit. Advisory check (not strict by
# default — fixing the 261 existing unjustified loose tolerances is its own
# workstream). Adds a CTest entry so devs can spot-run it; --strict in CI
# turns each unjustified tolerance into a build failure.
add_test(NAME ToleranceAudit
    COMMAND ${CMAKE_COMMAND} -E env
            python3 ${PROJECT_SOURCE_DIR}/scripts/audit_tolerances.py
                --root ${PROJECT_SOURCE_DIR}/tests)
set_tests_properties(ToleranceAudit PROPERTIES TIMEOUT 60 LABELS "audit")

# Manual KLDiv debug test
add_executable(test_kldiv_manual test_kldiv_manual.cpp)
target_link_libraries(test_kldiv_manual PRIVATE tenzor_core GTest::gtest_main)
add_test(NAME KLDivManual.StepByStepBackward COMMAND test_kldiv_manual --gtest_filter=KLDivManual.StepByStepBackward)
add_test(NAME KLDivManual.SimpleSubtract COMMAND test_kldiv_manual --gtest_filter=KLDivManual.SimpleSubtract)
add_test(NAME KLDivManual.SimpleMultiply COMMAND test_kldiv_manual --gtest_filter=KLDivManual.SimpleMultiply)
set_tests_properties(KLDivManual.StepByStepBackward KLDivManual.SimpleSubtract KLDivManual.SimpleMultiply
    PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT})

# ONNX import tests
add_executable(test_onnx_import test_onnx_import.cpp)
target_link_libraries(test_onnx_import PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_onnx_import DISCOVERY_TIMEOUT 60)

# Quantization tests
add_executable(test_quantization test_quantization.cpp)
target_link_libraries(test_quantization PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_quantization DISCOVERY_TIMEOUT 60)

# ============================================================================
# Phase 11 Tests - Additional Backend Support
# ============================================================================
add_executable(test_phase11_backends test_phase11_backends.cpp)
target_link_libraries(test_phase11_backends PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_phase11_backends DISCOVERY_TIMEOUT 60)

# CIoU Loss Tests
add_executable(test_ciou_loss test_ciou_loss.cpp)
target_link_libraries(test_ciou_loss PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_ciou_loss DISCOVERY_TIMEOUT 60)

# Backend Parity Test - Verify slice operations work correctly on all backends
add_executable(test_slice_backend_parity test_slice_backend_parity.cpp)
target_link_libraries(test_slice_backend_parity PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_slice_backend_parity DISCOVERY_TIMEOUT 60)

# Slice Debug Test - Debug slice offset and data_ptr behavior
add_executable(test_slice_debug test_slice_debug.cpp)
target_link_libraries(test_slice_debug PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_slice_debug DISCOVERY_TIMEOUT 60)

# Contiguous Fix Test - Verify .contiguous() fixes slice data access
add_executable(test_contiguous_fix test_contiguous_fix.cpp)
target_link_libraries(test_contiguous_fix PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_contiguous_fix DISCOVERY_TIMEOUT 60)

# Tensor Lifetime Test - Debug tensor data pointer stability
add_executable(test_tensor_lifetime test_tensor_lifetime.cpp)
target_link_libraries(test_tensor_lifetime PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_tensor_lifetime DISCOVERY_TIMEOUT 60)

# Backend Parity Example - Shows how to write multi-backend tests
add_executable(test_backend_parity_example test_backend_parity_example.cpp)
target_link_libraries(test_backend_parity_example PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_backend_parity_example DISCOVERY_TIMEOUT 60)

# Tensor indexing operator test
add_executable(test_indexing_operator test_indexing_operator.cpp)
target_link_libraries(test_indexing_operator PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_indexing_operator DISCOVERY_TIMEOUT 60)

# ============================================================================
# Phase 12 Tests - Enhanced Test Coverage and Validation
# ============================================================================

# Quantization conversion tests
add_executable(test_quantization_conversion test_quantization_conversion.cpp)
target_link_libraries(test_quantization_conversion PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_quantization_conversion DISCOVERY_TIMEOUT 60)

# Mask R-CNN loss tests
add_executable(test_mask_rcnn_losses test_mask_rcnn_losses.cpp)
target_link_libraries(test_mask_rcnn_losses PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_mask_rcnn_losses DISCOVERY_TIMEOUT 60)

# Vulkan complete operations tests
add_executable(test_vulkan_complete_ops test_vulkan_complete_ops.cpp)
target_link_libraries(test_vulkan_complete_ops PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_vulkan_complete_ops DISCOVERY_TIMEOUT 60)

# Repeat and Tile tests
add_executable(test_repeat_tile test_repeat_tile.cpp)
target_link_libraries(test_repeat_tile PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_repeat_tile DISCOVERY_TIMEOUT 60)

# ============================================================================
# Multi-backend root-level tests (Phase 4)
# ============================================================================

# Repeat/tile multi-backend
add_executable(test_repeat_tile_multidtype test_repeat_tile_multidtype.cpp)
target_link_libraries(test_repeat_tile_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_repeat_tile_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Comparison operators multi-backend
add_executable(test_comparison_operators_multidtype test_comparison_operators_multidtype.cpp)
target_link_libraries(test_comparison_operators_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_comparison_operators_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Statistics operations (cov, corrcoef) multi-backend
add_executable(test_statistics_multidtype test_statistics_multidtype.cpp)
target_link_libraries(test_statistics_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_statistics_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Indexing operator multi-backend
add_executable(test_indexing_operator_multidtype test_indexing_operator_multidtype.cpp)
target_link_libraries(test_indexing_operator_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_indexing_operator_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# BMM autograd multi-backend
add_executable(test_bmm_autograd_multidtype test_bmm_autograd_multidtype.cpp)
target_link_libraries(test_bmm_autograd_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_bmm_autograd_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# DType conversion multi-backend
add_executable(test_dtype_conversion_multidtype test_dtype_conversion_multidtype.cpp)
target_link_libraries(test_dtype_conversion_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_dtype_conversion_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Split operation multi-backend
add_executable(test_split_operation_multidtype test_split_operation_multidtype.cpp)
target_link_libraries(test_split_operation_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_split_operation_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Conv1d multi-backend
add_executable(test_conv1d_multidtype test_conv1d_multidtype.cpp)
target_link_libraries(test_conv1d_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_conv1d_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# CholeskySolve multi-backend
add_executable(test_cholesky_solve_multidtype test_cholesky_solve_multidtype.cpp)
target_link_libraries(test_cholesky_solve_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_cholesky_solve_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# DeformableConv2d multi-backend
add_executable(test_deformable_conv2d_multidtype test_deformable_conv2d_multidtype.cpp)
target_link_libraries(test_deformable_conv2d_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_deformable_conv2d_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Einsum dispatch multi-backend
add_executable(test_einsum_dispatch_multidtype test_einsum_dispatch_multidtype.cpp)
target_link_libraries(test_einsum_dispatch_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_einsum_dispatch_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# ConvTranspose2d multi-backend
add_executable(test_convtranspose2d_multidtype test_convtranspose2d_multidtype.cpp)
target_link_libraries(test_convtranspose2d_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_convtranspose2d_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# KL divergence multi-backend
add_executable(test_kldiv_manual_multidtype test_kldiv_manual_multidtype.cpp)
target_link_libraries(test_kldiv_manual_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_kldiv_manual_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# ============================================================================
# Phase 4: New tests for previously untested components
# ============================================================================

# Missing loss functions (MarginRanking, Poisson, Cosine, Triplet, etc.)
add_executable(test_losses_missing_multidtype test_losses_missing_multidtype.cpp)
target_link_libraries(test_losses_missing_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_losses_missing_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# RNN cells (RNNCell, GRUCell)
add_executable(test_rnn_cells_multidtype test_rnn_cells_multidtype.cpp)
target_link_libraries(test_rnn_cells_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_rnn_cells_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# RNN utilities (pack_padded_sequence, pad_packed_sequence)
add_executable(test_rnn_utils_multidtype test_rnn_utils_multidtype.cpp)
target_link_libraries(test_rnn_utils_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_rnn_utils_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# KV Cache
add_executable(test_kv_cache_multidtype test_kv_cache_multidtype.cpp)
target_link_libraries(test_kv_cache_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_kv_cache_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Gradient accumulation
add_executable(test_grad_accumulation test_grad_accumulation.cpp)
target_link_libraries(test_grad_accumulation PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_grad_accumulation DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# ROI operations (ROIAlign)
add_executable(test_roi_ops_multidtype test_roi_ops_multidtype.cpp)
target_link_libraries(test_roi_ops_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_roi_ops_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# LOBPCG eigenvalue solver multi-backend
add_executable(test_lobpcg_multidtype test_lobpcg_multidtype.cpp)
target_link_libraries(test_lobpcg_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_lobpcg_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# DCT / IDCT multi-backend
add_executable(test_dct_multidtype test_dct_multidtype.cpp)
target_link_libraries(test_dct_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_dct_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_mfcc_multidtype test_mfcc_multidtype.cpp)
target_link_libraries(test_mfcc_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_mfcc_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# LazyTensor deferred-execution tests (Phase 5.2)
add_executable(test_lazy_tensor test_lazy_tensor.cpp)
target_link_libraries(test_lazy_tensor PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_lazy_tensor DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# LazyTensor multi-backend × multi-dtype tests (audit 2026-05-02 D.2)
add_executable(test_lazy_tensor_multidtype test_lazy_tensor_multidtype.cpp)
target_link_libraries(test_lazy_tensor_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_lazy_tensor_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# LazyTensor: materialise -> Variable -> backward parity (audit 2026-05-02 D.3)
add_executable(test_lazy_tensor_backward test_lazy_tensor_backward.cpp)
target_link_libraries(test_lazy_tensor_backward PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_lazy_tensor_backward DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Vulkan tensor creation test (standalone, non-GTest)
add_executable(vulkan_tensor_test vulkan_tensor_test.cpp)
target_link_libraries(vulkan_tensor_test PRIVATE tenzor_core)

# Vulkan diagnostic test
add_executable(vulkan_diagnostic vulkan_diagnostic.cpp)
target_link_libraries(vulkan_diagnostic PRIVATE tenzor_core)

# Vulkan addition debug test
add_executable(vulkan_add_debug vulkan_add_debug.cpp)
target_link_libraries(vulkan_add_debug PRIVATE tenzor_core)

# Debug DataLoader test
add_executable(debug_dataloader debug_dataloader.cpp)
target_link_libraries(debug_dataloader PRIVATE tenzor_core)

# Minimal Swin debug test - Compare Float32 vs Float16 backward pass
add_executable(minimal_swin_debug minimal_swin_debug.cpp)
target_link_libraries(minimal_swin_debug PRIVATE tenzor_core GTest::gtest_main)

# Check Float32 gradient validity
add_executable(check_float32_gradients check_float32_gradients.cpp)
target_link_libraries(check_float32_gradients PRIVATE tenzor_core)

# Phase 1.5: Metrics extended tests
add_executable(test_metrics_extended unit/test_metrics_extended.cpp)
target_link_libraries(test_metrics_extended PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_metrics_extended LABELS "unit" DISCOVERY_TIMEOUT 60)

# ============================================================================
# CUDA root-level tests
# ============================================================================
if(TENZOR_BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)

    # CUDA concatenation tests
    add_executable(test_cuda_cat test_cuda_cat.cpp)
    target_link_libraries(test_cuda_cat PRIVATE tenzor_core CUDA::cudart)
    target_include_directories(test_cuda_cat PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    set_target_properties(test_cuda_cat PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME CUDAConcatenation COMMAND test_cuda_cat)
    set_tests_properties(CUDAConcatenation PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT})
endif()

# test_jit.cpp orphan removed: rotted APIs (Tensor::full/zeros/ones static methods, DeviceType enum no longer exist)

# ============================================================================
# Benchmark executables
# ============================================================================

# Benchmark suite
add_executable(benchmark_suite benchmarks/benchmark_suite.cpp)
target_link_libraries(benchmark_suite PRIVATE tenzor_core tenzor_backend_cpu)

# Cross-backend benchmark suite
add_executable(benchmark_backends benchmarks/benchmark_backends.cpp)
target_link_libraries(benchmark_backends PRIVATE tenzor_core)

# ==============================================================================
# Layer-test grad-flow guard (lint test, no compilation)
#
# Fails CI if any tests/nn/layers/test_*.cpp file calls .backward() without
# at least one of: EXPECT_GRAD_FLOWS macro, numerical_gradient finite-
# difference helper, an explicit max-abs > 0 check, or a per-element
# EXPECT_NEAR pattern. Catches regressions in the silently-zeroed-gradient
# bug class at PR-time rather than after the bug ships.
# ==============================================================================
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND)
    add_test(
        NAME LayerTestGradFlowGuard
        COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_SOURCE_DIR}/tools/check_layer_tests.py"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
    set_tests_properties(LayerTestGradFlowGuard PROPERTIES
        LABELS "lint;layer_tests"
        TIMEOUT 30
    )
endif()

# ==============================================================================
# Python Tests (registered as CTest tests)
# ==============================================================================
if(TENZOR_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter)
    if(Python_FOUND)
        set(PYTHON_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/python")
        set(PYTHON_TEST_ENV "PYTHONPATH=${CMAKE_BINARY_DIR}/python")

        # Python tests in this list also get the "smoke" label so the
        # fast-feedback `ctest -L smoke` set covers basic Python-binding
        # surface (audit 2026-05-02 I.1). Each must be a small,
        # fast-running test that exercises a load-bearing path.
        set(PYTHON_SMOKE_TESTS
            test_loss_simple
            test_tensor_ops
            test_autograd
        )

        file(GLOB PYTHON_TEST_FILES "${PYTHON_TEST_DIR}/test_*.py")
        foreach(test_file ${PYTHON_TEST_FILES})
            get_filename_component(test_name ${test_file} NAME_WE)
            add_test(
                NAME "python_${test_name}"
                COMMAND ${Python_EXECUTABLE} ${test_file}
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            )
            list(FIND PYTHON_SMOKE_TESTS ${test_name} _smoke_idx)
            if(_smoke_idx GREATER -1)
                set(_py_labels "python;smoke")
            else()
                set(_py_labels "python")
            endif()
            set_tests_properties("python_${test_name}" PROPERTIES
                LABELS "${_py_labels}"
                TIMEOUT ${TENZOR_TEST_TIMEOUT}
                ENVIRONMENT "${PYTHON_TEST_ENV}"
            )
        endforeach()
    endif()
endif()
