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

# Custom function that wraps gtest_discover_tests
function(tenzor_discover_tests target)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "DISCOVERY_TIMEOUT;DISCOVERY_MODE" "PROPERTIES")
    set(TIMEOUT ${ARG_DISCOVERY_TIMEOUT})
    if(NOT TIMEOUT)
        set(TIMEOUT 30)
    endif()
    set(PROPS ${ARG_PROPERTIES})
    set(MODE ${ARG_DISCOVERY_MODE})
    if(NOT MODE)
        set(MODE "POST_BUILD")
    endif()

    # Now call gtest_discover_tests
    gtest_discover_tests(${target}
        DISCOVERY_TIMEOUT ${TIMEOUT}
        DISCOVERY_MODE ${MODE}
        PROPERTIES ${PROPS})
endfunction()

# Helper macro: create a test executable, link GTest, and register with CTest.
# Usage: tenzor_add_test(NAME test_foo SOURCE unit/test_foo.cpp [LIBS extra_lib])
macro(tenzor_add_test)
    cmake_parse_arguments(ARG "" "NAME;SOURCE" "LIBS" ${ARGN})
    add_executable(${ARG_NAME} ${ARG_SOURCE})
    target_link_libraries(${ARG_NAME} PRIVATE tenzor_core GTest::gtest_main ${ARG_LIBS})
    tenzor_discover_tests(${ARG_NAME} DISCOVERY_TIMEOUT 30)
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
add_subdirectory(integration)
add_subdirectory(backend_parity)
add_subdirectory(utils)

# ============================================================================
# Combined unit test executable
# ============================================================================
add_executable(tenzor_unit_tests
    unit/test_tensor.cpp
    unit/test_device.cpp
    unit/test_ops.cpp
    unit/test_ops_additional.cpp
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

# tenzor_unit_tests crashes during --gtest_list_tests (static destruction in linked
# backends), so use add_test instead of gtest_discover_tests
add_test(NAME tenzor_unit_tests COMMAND tenzor_unit_tests)
set_tests_properties(tenzor_unit_tests PROPERTIES TIMEOUT 120)

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
tenzor_discover_tests(test_cuda_scalar_debug DISCOVERY_TIMEOUT 30)

# Legacy integration tests (kept for compatibility)
add_executable(tenzor_integration_tests
    integration/test_nn.cpp
    integration/test_training.cpp
    integration/test_debug_mlp.cpp
)

target_link_libraries(tenzor_integration_tests PRIVATE
    tenzor_core
    GTest::gtest_main
)

tenzor_discover_tests(tenzor_integration_tests DISCOVERY_TIMEOUT 30)

# Conv1d tests
add_executable(test_conv1d test_conv1d.cpp)
target_link_libraries(test_conv1d PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_conv1d DISCOVERY_TIMEOUT 30)

# ConvTranspose2d tests
add_executable(test_convtranspose2d test_convtranspose2d.cpp)
target_link_libraries(test_convtranspose2d PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_convtranspose2d DISCOVERY_TIMEOUT 30)

# Minimal training test (debugging NaN issue)
add_executable(test_minimal_training test_minimal_training.cpp)
target_link_libraries(test_minimal_training PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_minimal_training DISCOVERY_TIMEOUT 30)

# Minimal Float16 tests (debugging Float16 corruption)
add_executable(minimal_float16_test minimal_float16_test.cpp)
target_link_libraries(minimal_float16_test PRIVATE tenzor_core)

add_executable(minimal_float16_variable_test minimal_float16_variable_test.cpp)
target_link_libraries(minimal_float16_variable_test PRIVATE tenzor_core)

add_executable(minimal_float16_embedding_test minimal_float16_embedding_test.cpp)
target_link_libraries(minimal_float16_embedding_test PRIVATE tenzor_core)

add_executable(minimal_float16_attention_test minimal_float16_attention_test.cpp)
target_link_libraries(minimal_float16_attention_test PRIVATE tenzor_core)

# Split operation tests
add_executable(test_split_operation test_split_operation.cpp)
target_link_libraries(test_split_operation PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_split_operation DISCOVERY_TIMEOUT 30)

# DType conversion tests
add_executable(test_dtype_conversion test_dtype_conversion.cpp)
target_link_libraries(test_dtype_conversion PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_dtype_conversion DISCOVERY_TIMEOUT 30)

# BMM autograd tests
add_executable(test_bmm_autograd test_bmm_autograd.cpp)
target_link_libraries(test_bmm_autograd PRIVATE tenzor_core GTest::gtest)
tenzor_discover_tests(test_bmm_autograd DISCOVERY_TIMEOUT 30)

# Linear reshape integration tests
add_executable(test_linear_reshape_integration test_linear_reshape_integration.cpp)
target_link_libraries(test_linear_reshape_integration PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_linear_reshape_integration DISCOVERY_TIMEOUT 30)

# Creation operations tests
add_executable(test_creation_ops test_creation_ops.cpp)
target_link_libraries(test_creation_ops PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_creation_ops DISCOVERY_TIMEOUT 30)

# Creation operations multi-dtype tests
add_executable(test_creation_ops_multidtype test_creation_ops_multidtype.cpp)
target_link_libraries(test_creation_ops_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_creation_ops_multidtype DISCOVERY_TIMEOUT 30)

# Comparison operators tests
add_executable(test_comparison_operators test_comparison_operators.cpp)
target_link_libraries(test_comparison_operators PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_comparison_operators DISCOVERY_TIMEOUT 30)

# Checkpoint leaf fix test
add_executable(test_checkpoint_leaf_fix test_checkpoint_leaf_fix.cpp)
target_link_libraries(test_checkpoint_leaf_fix tenzor_core)
target_include_directories(test_checkpoint_leaf_fix PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME CheckpointLeafFix COMMAND test_checkpoint_leaf_fix)

# Manual KLDiv debug test
add_executable(test_kldiv_manual test_kldiv_manual.cpp)
target_link_libraries(test_kldiv_manual PRIVATE tenzor_core GTest::gtest_main)
add_test(NAME KLDivManual.StepByStepBackward COMMAND test_kldiv_manual --gtest_filter=KLDivManual.StepByStepBackward)
add_test(NAME KLDivManual.SimpleSubtract COMMAND test_kldiv_manual --gtest_filter=KLDivManual.SimpleSubtract)
add_test(NAME KLDivManual.SimpleMultiply COMMAND test_kldiv_manual --gtest_filter=KLDivManual.SimpleMultiply)

# ONNX import tests
add_executable(test_onnx_import test_onnx_import.cpp)
target_link_libraries(test_onnx_import PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_onnx_import DISCOVERY_TIMEOUT 30)

# Quantization tests
add_executable(test_quantization test_quantization.cpp)
target_link_libraries(test_quantization PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_quantization DISCOVERY_TIMEOUT 30)

# ============================================================================
# Phase 11 Tests - Additional Backend Support
# ============================================================================
add_executable(test_phase11_backends test_phase11_backends.cpp)
target_link_libraries(test_phase11_backends PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_phase11_backends DISCOVERY_TIMEOUT 30)

# CIoU Loss Tests
add_executable(test_ciou_loss test_ciou_loss.cpp)
target_link_libraries(test_ciou_loss PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_ciou_loss DISCOVERY_TIMEOUT 30)

# Backend Parity Test - Verify slice operations work correctly on all backends
add_executable(test_slice_backend_parity test_slice_backend_parity.cpp)
target_link_libraries(test_slice_backend_parity PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_slice_backend_parity DISCOVERY_TIMEOUT 30)

# Slice Debug Test - Debug slice offset and data_ptr behavior
add_executable(test_slice_debug test_slice_debug.cpp)
target_link_libraries(test_slice_debug PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_slice_debug DISCOVERY_TIMEOUT 30)

# Contiguous Fix Test - Verify .contiguous() fixes slice data access
add_executable(test_contiguous_fix test_contiguous_fix.cpp)
target_link_libraries(test_contiguous_fix PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_contiguous_fix DISCOVERY_TIMEOUT 30)

# Tensor Lifetime Test - Debug tensor data pointer stability
add_executable(test_tensor_lifetime test_tensor_lifetime.cpp)
target_link_libraries(test_tensor_lifetime PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_tensor_lifetime DISCOVERY_TIMEOUT 30)

# Backend Parity Example - Shows how to write multi-backend tests
add_executable(test_backend_parity_example test_backend_parity_example.cpp)
target_link_libraries(test_backend_parity_example PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_backend_parity_example DISCOVERY_TIMEOUT 30)

# Tensor indexing operator test
add_executable(test_indexing_operator test_indexing_operator.cpp)
target_link_libraries(test_indexing_operator PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_indexing_operator DISCOVERY_TIMEOUT 30)

# ============================================================================
# Phase 12 Tests - Enhanced Test Coverage and Validation
# ============================================================================

# Quantization conversion tests
add_executable(test_quantization_conversion test_quantization_conversion.cpp)
target_link_libraries(test_quantization_conversion PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_quantization_conversion DISCOVERY_TIMEOUT 30)

# Mask R-CNN loss tests
add_executable(test_mask_rcnn_losses test_mask_rcnn_losses.cpp)
target_link_libraries(test_mask_rcnn_losses PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_mask_rcnn_losses DISCOVERY_TIMEOUT 30)

# Vulkan complete operations tests
add_executable(test_vulkan_complete_ops test_vulkan_complete_ops.cpp)
target_link_libraries(test_vulkan_complete_ops PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_vulkan_complete_ops DISCOVERY_TIMEOUT 30)

# Repeat and Tile tests
add_executable(test_repeat_tile test_repeat_tile.cpp)
target_link_libraries(test_repeat_tile PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_repeat_tile DISCOVERY_TIMEOUT 30)

# Vulkan tensor creation test (standalone, non-GTest)
add_executable(vulkan_tensor_test vulkan_tensor_test.cpp)
target_link_libraries(vulkan_tensor_test PRIVATE tenzor_core)

# Vulkan diagnostic test
add_executable(vulkan_diagnostic vulkan_diagnostic.cpp)
target_link_libraries(vulkan_diagnostic PRIVATE tenzor_core)

# Vulkan addition debug test
add_executable(vulkan_add_debug vulkan_add_debug.cpp)
target_link_libraries(vulkan_add_debug PRIVATE tenzor_core)

# Debug vulkan hang with command batching
add_executable(debug_vulkan_hang debug_vulkan_hang.cpp)
target_link_libraries(debug_vulkan_hang PRIVATE tenzor_core)

# Debug DataLoader test
add_executable(debug_dataloader debug_dataloader.cpp)
target_link_libraries(debug_dataloader PRIVATE tenzor_core)

# Minimal Float16 Linear backward test
add_executable(minimal_float16_linear_backward_test minimal_float16_linear_backward_test.cpp)
target_link_libraries(minimal_float16_linear_backward_test PRIVATE tenzor_core)

# Minimal Float16 Embedding backward test
add_executable(minimal_float16_embedding_backward_test minimal_float16_embedding_backward_test.cpp)
target_link_libraries(minimal_float16_embedding_backward_test PRIVATE tenzor_core)

# Minimal Float16 LayerNorm backward test
add_executable(minimal_float16_layernorm_backward_test minimal_float16_layernorm_backward_test.cpp)
target_link_libraries(minimal_float16_layernorm_backward_test PRIVATE tenzor_core)

# Minimal Float16 Attention backward test
add_executable(minimal_float16_attention_backward_test minimal_float16_attention_backward_test.cpp)
target_link_libraries(minimal_float16_attention_backward_test PRIVATE tenzor_core)

# Minimal Float16 BMM backward test
add_executable(minimal_float16_bmm_backward_test minimal_float16_bmm_backward_test.cpp)
target_link_libraries(minimal_float16_bmm_backward_test PRIVATE tenzor_core)

# Minimal Float16 Softmax backward test
add_executable(minimal_float16_softmax_backward_test minimal_float16_softmax_backward_test.cpp)
target_link_libraries(minimal_float16_softmax_backward_test PRIVATE tenzor_core)

# Minimal Float16 Reshape backward test
add_executable(minimal_float16_reshape_backward_test minimal_float16_reshape_backward_test.cpp)
target_link_libraries(minimal_float16_reshape_backward_test PRIVATE tenzor_core)

# Minimal Float16 Permute backward test
add_executable(minimal_float16_permute_backward_test minimal_float16_permute_backward_test.cpp)
target_link_libraries(minimal_float16_permute_backward_test PRIVATE tenzor_core)

# Minimal Float16 Scaled Dot-Product Attention test
add_executable(minimal_float16_scaled_dot_product_attention_test minimal_float16_scaled_dot_product_attention_test.cpp)
target_link_libraries(minimal_float16_scaled_dot_product_attention_test PRIVATE tenzor_core)

# Minimal Float16 Linear + Reshape + Permute test
add_executable(minimal_float16_linear_reshape_permute_test minimal_float16_linear_reshape_permute_test.cpp)
target_link_libraries(minimal_float16_linear_reshape_permute_test PRIVATE tenzor_core)

# Minimal Float16 Linear test (2x768)
add_executable(minimal_float16_linear_test minimal_float16_linear_test.cpp)
target_link_libraries(minimal_float16_linear_test PRIVATE tenzor_core GTest::gtest_main)

# Minimal Float16 Matmul test (256x128x768) - Debug ALBERT issue
add_executable(minimal_float16_matmul_256x128x768 minimal_float16_matmul_256x128x768.cpp)
target_link_libraries(minimal_float16_matmul_256x128x768 PRIVATE tenzor_core GTest::gtest_main)

add_executable(minimal_float16_mean_backward_test minimal_float16_mean_backward_test.cpp)
target_link_libraries(minimal_float16_mean_backward_test PRIVATE tenzor_core GTest::gtest_main)

# Minimal Float16 debug test - Compare Float16 vs Float32 with zero inputs
add_executable(minimal_float16_debug minimal_float16_debug.cpp)
target_link_libraries(minimal_float16_debug PRIVATE tenzor_core GTest::gtest_main)

# Minimal Float16 Conv2d test - Test patch embedding with zero inputs
add_executable(minimal_float16_conv2d_test minimal_float16_conv2d_test.cpp)
target_link_libraries(minimal_float16_conv2d_test PRIVATE tenzor_core GTest::gtest_main)

# Minimal Swin debug test - Compare Float32 vs Float16 backward pass
add_executable(minimal_swin_debug minimal_swin_debug.cpp)
target_link_libraries(minimal_swin_debug PRIVATE tenzor_core GTest::gtest_main)

# Check Float32 gradient validity
add_executable(check_float32_gradients check_float32_gradients.cpp)
target_link_libraries(check_float32_gradients PRIVATE tenzor_core)

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
endif()

# ============================================================================
# Benchmark executables
# ============================================================================

# Benchmark suite
add_executable(benchmark_suite benchmarks/benchmark_suite.cpp)
target_link_libraries(benchmark_suite PRIVATE tenzor_core tenzor_backend_cpu)

# Cross-backend benchmark suite
add_executable(benchmark_backends benchmarks/benchmark_backends.cpp)
target_link_libraries(benchmark_backends PRIVATE tenzor_core)

# ============================================================================
# Debug tests commented out - tmp files not tracked in git
# ============================================================================

# add_executable(test_batchnorm_precision_debug
#     /tmp/test_batchnorm_precision.cpp
# )
#
# target_link_libraries(test_batchnorm_precision_debug PRIVATE
#     tenzor_core
# )

# add_executable(test_variance_debug
#     /tmp/test_variance_debug.cpp
# )
#
# target_link_libraries(test_variance_debug PRIVATE
#     tenzor_core
# )

# add_executable(test_multichannel
#     /tmp/test_multichannel.cpp
# )
#
# target_link_libraries(test_multichannel PRIVATE
#     tenzor_core
# )

# add_executable(test_linear_backward_debug
#     tmp/debug_linear_backward.cpp
# )
#
# target_link_libraries(test_linear_backward_debug PRIVATE
#     tenzor_core
# )

# add_executable(debug_attention_flow
#     tmp/debug_attention_flow.cpp
# )
#
# target_link_libraries(debug_attention_flow PRIVATE
#     tenzor_core
# )

# add_executable(debug_module_grad
#     tmp/debug_module_grad.cpp
# )
#
# target_link_libraries(debug_module_grad PRIVATE
#     tenzor_core
# )

# add_executable(debug_pointers
#     tmp/debug_pointers.cpp
# )
#
# target_link_libraries(debug_pointers PRIVATE
#     tenzor_core
# )

# add_executable(debug_full_backward
#     tmp/debug_full_backward.cpp
# )
#
# target_link_libraries(debug_full_backward PRIVATE
#     tenzor_core
# )

# add_executable(debug_multi_linear ../tests/tmp/debug_multi_linear.cpp)
# target_link_libraries(debug_multi_linear PRIVATE tenzor_core)
# add_executable(debug_attention_copy tmp/debug_attention_copy.cpp)
# target_link_libraries(debug_attention_copy PRIVATE tenzor_core)
# add_executable(debug_grad_fn_refs tmp/debug_grad_fn_refs.cpp)
# target_link_libraries(debug_grad_fn_refs PRIVATE tenzor_core)
# add_executable(debug_transformer_grad tmp/debug_transformer_grad.cpp)
# target_link_libraries(debug_transformer_grad PRIVATE tenzor_core)
# add_executable(debug_encoder_only tmp/debug_encoder_only.cpp)
# target_link_libraries(debug_encoder_only PRIVATE tenzor_core)
# add_executable(debug_simple_residual tmp/debug_simple_residual.cpp)
# target_link_libraries(debug_simple_residual PRIVATE tenzor_core)
# add_executable(debug_addition_only tmp/debug_addition_only.cpp)
# target_link_libraries(debug_addition_only PRIVATE tenzor_core)
# add_executable(debug_layernorm_grad tmp/debug_layernorm_grad.cpp)
# target_link_libraries(debug_layernorm_grad PRIVATE tenzor_core)
# add_executable(debug_encoder_attn_only tmp/debug_encoder_attn_only.cpp)
# target_link_libraries(debug_encoder_attn_only PRIVATE tenzor_core)
# add_executable(debug_attn_residual_norm tmp/debug_attn_residual_norm.cpp)
# target_link_libraries(debug_attn_residual_norm PRIVATE tenzor_core)
# add_executable(debug_add_norm_only tmp/debug_add_norm_only.cpp)
# target_link_libraries(debug_add_norm_only PRIVATE tenzor_core)
