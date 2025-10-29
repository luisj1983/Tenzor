cmake_minimum_required(VERSION 3.25)

# Find Google Test
find_package(GTest)

if(NOT GTest_FOUND)
    message(STATUS "Google Test not found, downloading...")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )
    FetchContent_MakeAvailable(googletest)
endif()

# Unit tests
add_executable(tenzor_unit_tests
    unit/test_tensor.cpp
    unit/test_device.cpp
    unit/test_ops.cpp
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
)

target_link_libraries(tenzor_unit_tests PRIVATE
    tenzor_core
    GTest::gtest_main
)

# SIMD Benchmark tests (separate executable for benchmarking)
add_executable(tenzor_simd_benchmark
    unit/benchmark_simd.cpp
)

target_link_libraries(tenzor_simd_benchmark PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Legacy integration tests (kept for compatibility)
add_executable(tenzor_integration_tests
    integration/test_nn.cpp
    integration/test_training.cpp
)

target_link_libraries(tenzor_integration_tests PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Comprehensive integration tests subdirectory
add_subdirectory(integration)

# Dropout tests
add_executable(test_dropout
    nn/layers/test_dropout.cpp
)

target_link_libraries(test_dropout PRIVATE
    tenzor_core
    GTest::gtest_main
)

# BatchNorm2d tests
add_executable(test_batchnorm2d
    nn/layers/test_batchnorm2d.cpp
)

target_link_libraries(test_batchnorm2d PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Conv2d tests
add_executable(test_conv2d
    nn/layers/test_conv2d.cpp
)

target_link_libraries(test_conv2d PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Pooling tests
add_executable(test_pooling
    nn/layers/test_pooling.cpp
)

target_link_libraries(test_pooling PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Conv1d tests
add_executable(test_conv1d
    test_conv1d.cpp
)

target_link_libraries(test_conv1d PRIVATE
    tenzor_core
    GTest::gtest_main
)

# ConvTranspose2d tests (temporarily disabled due to compilation errors)
# add_executable(test_convtranspose2d
#     test_convtranspose2d.cpp
# )

# target_link_libraries(test_convtranspose2d PRIVATE
#     tenzor_core
#     GTest::gtest_main
# )

# Scheduler tests
add_executable(test_schedulers
    nn/optim/test_schedulers.cpp
)

target_link_libraries(test_schedulers PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Normalization tests
add_executable(test_normalization
    nn/layers/test_normalization.cpp
)

target_link_libraries(test_normalization PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Serialization tests
add_executable(test_serialization
    nn/test_serialization.cpp
)

target_link_libraries(test_serialization PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Embedding tests
add_executable(test_embedding
    unit/test_embedding.cpp
)

target_link_libraries(test_embedding PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Extended optimizers tests
add_executable(test_optimizers_extended
    unit/test_optimizers_extended.cpp
)

target_link_libraries(test_optimizers_extended PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Advanced schedulers tests
add_executable(test_schedulers_advanced
    unit/test_schedulers_advanced.cpp
)

target_link_libraries(test_schedulers_advanced PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Advanced losses tests
add_executable(test_losses_advanced
    unit/test_losses_advanced.cpp
)

target_link_libraries(test_losses_advanced PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Offload tests (ZeRO Stage 2 Parameter Offloading)
add_executable(test_offload
    nn/test_offload.cpp
)

target_link_libraries(test_offload PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Note: Distributed tests for Phase 3 (collective operations) are in
# integration/test_distributed.cpp and built via integration/CMakeLists.txt
# as the 'test_distributed' target.
#
# The nn/test_distributed.cpp file tests Phase 4+ features (DistributedDataParallel,
# GradientBucket) that are not yet implemented and will be added in future phases.

# RNN tests
add_executable(test_rnn
    unit/test_rnn.cpp
)

target_link_libraries(test_rnn PRIVATE
    tenzor_core
    GTest::gtest_main
)

# LSTM tests
add_executable(test_lstm
    unit/test_lstm.cpp
)

target_link_libraries(test_lstm PRIVATE
    tenzor_core
    GTest::gtest_main
)

# GRU tests
add_executable(test_gru
    unit/test_gru.cpp
)

target_link_libraries(test_gru PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Attention tests
add_executable(test_attention
    unit/test_attention.cpp
)

target_link_libraries(test_attention PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Transformer tests
add_executable(test_transformer
    unit/test_transformer.cpp
)

target_link_libraries(test_transformer PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Chunk tests
add_executable(test_chunk
    unit/test_chunk.cpp
)

target_link_libraries(test_chunk PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Split operation tests
add_executable(test_split_operation
    test_split_operation.cpp
)

target_link_libraries(test_split_operation PRIVATE
    tenzor_core
    GTest::gtest_main
)

# GradScaler (AMP) tests
add_executable(test_grad_scaler
    unit/test_grad_scaler.cpp
)

target_link_libraries(test_grad_scaler PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Autocast tests
add_executable(test_autocast
    unit/test_autocast.cpp
)

target_link_libraries(test_autocast PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Mixed precision training tests
add_executable(test_mixed_precision
    unit/test_mixed_precision.cpp
)

target_link_libraries(test_mixed_precision PRIVATE
    tenzor_core
    GTest::gtest_main
)

# DataLoader tests
add_executable(test_dataloader
    unit/test_dataloader.cpp
)

target_link_libraries(test_dataloader PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Float16/BFloat16 tests
add_executable(test_fp16
    unit/test_fp16.cpp
)

target_link_libraries(test_fp16 PRIVATE
    tenzor_core
    GTest::gtest_main
)

# DType conversion tests
add_executable(test_dtype_conversion
    test_dtype_conversion.cpp
)

target_link_libraries(test_dtype_conversion PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Gradient checkpoint tests
add_executable(test_gradient_checkpoint
    unit/test_gradient_checkpoint.cpp
)

target_link_libraries(test_gradient_checkpoint PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Model checkpoint tests
add_executable(test_model_checkpoint
    unit/test_model_checkpoint.cpp
)

target_link_libraries(test_model_checkpoint PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Comprehensive checkpoint tests
add_executable(test_checkpoint
    unit/test_checkpoint.cpp
)

target_link_libraries(test_checkpoint PRIVATE
    tenzor_core
    GTest::gtest_main
)

# BMM autograd tests
add_executable(test_bmm_autograd
    test_bmm_autograd.cpp
)

target_link_libraries(test_bmm_autograd PRIVATE
    tenzor_core
    GTest::gtest
)

# Linear reshape integration tests
add_executable(test_linear_reshape_integration
    test_linear_reshape_integration.cpp
)

target_link_libraries(test_linear_reshape_integration PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Fused operations tests
add_executable(test_fused_ops
    unit/test_fused_ops.cpp
)

target_link_libraries(test_fused_ops PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Backend operations parameterized tests (runs on ALL backends)
add_executable(test_backend_ops_parameterized
    unit/test_backend_ops_parameterized.cpp
)

target_link_libraries(test_backend_ops_parameterized PRIVATE
    tenzor_core
    GTest::gtest_main
)

# CachingAllocator tests (universal - uses mock backend)
add_executable(test_caching_allocator
    unit/test_caching_allocator.cpp
)

target_link_libraries(test_caching_allocator PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_caching_allocator DISCOVERY_TIMEOUT 30)

# CUDA-specific tests
if(TENZOR_BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)

    # Memory Manager tests (ZeRO Phase 1)
    add_executable(test_memory_manager
        core/test_memory_manager.cpp
    )

    target_link_libraries(test_memory_manager PRIVATE
        tenzor_core
        GTest::gtest_main
        CUDA::cudart
    )

    target_include_directories(test_memory_manager PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_compile_definitions(test_memory_manager PRIVATE
        TENZOR_CUDA_ENABLED
    )

    # Transfer Engine tests (ZeRO Phase 1)
    add_executable(test_transfer_engine
        core/test_transfer_engine.cpp
    )

    target_link_libraries(test_transfer_engine PRIVATE
        tenzor_core
        GTest::gtest_main
        CUDA::cudart
    )

    target_include_directories(test_transfer_engine PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_compile_definitions(test_transfer_engine PRIVATE
        TENZOR_CUDA_ENABLED
    )

    # Pinned Allocator tests (ZeRO Phase 1)
    add_executable(test_pinned_allocator
        core/test_pinned_allocator.cpp
    )

    target_link_libraries(test_pinned_allocator PRIVATE
        tenzor_core
        GTest::gtest_main
        CUDA::cudart
    )

    target_include_directories(test_pinned_allocator PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_compile_definitions(test_pinned_allocator PRIVATE
        TENZOR_CUDA_ENABLED
    )

    # Transfer Bandwidth Benchmark (ZeRO Phase 1)
    add_executable(test_transfer_benchmark
        core/test_transfer_benchmark.cpp
    )

    target_link_libraries(test_transfer_benchmark PRIVATE
        tenzor_core
        GTest::gtest_main
        CUDA::cudart
    )

    target_include_directories(test_transfer_benchmark PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_compile_definitions(test_transfer_benchmark PRIVATE
        TENZOR_CUDA_ENABLED
    )

    # Phase 2: Offload Engine tests
    add_executable(test_offload_engine
        core/test_offload_engine.cpp
    )

    target_link_libraries(test_offload_engine PRIVATE
        tenzor_core
        GTest::gtest_main
        CUDA::cudart
    )

    target_include_directories(test_offload_engine PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_compile_definitions(test_offload_engine PRIVATE
        TENZOR_CUDA_ENABLED
    )

    # Phase 2: Parameter Offloading API tests
    add_executable(test_parameter_offload
        nn/test_offload.cpp
    )

    target_link_libraries(test_parameter_offload PRIVATE
        tenzor_core
        GTest::gtest_main
        CUDA::cudart
    )

    target_include_directories(test_parameter_offload PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_compile_definitions(test_parameter_offload PRIVATE
        TENZOR_CUDA_ENABLED
    )

    # DataParallel tests (requires CUDA)
    add_executable(test_data_parallel
        unit/test_data_parallel.cpp
    )

    target_link_libraries(test_data_parallel PRIVATE
        tenzor_core
        tenzor_backend_cuda
        GTest::gtest_main
        CUDA::cudart
    )

    target_include_directories(test_data_parallel PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    # DataParallel single-GPU tests (comprehensive)
    add_executable(test_data_parallel_single_gpu
        unit/test_data_parallel_single_gpu.cpp
    )

    target_link_libraries(test_data_parallel_single_gpu PRIVATE
        tenzor_core
        tenzor_backend_cuda
        GTest::gtest_main
        CUDA::cudart
    )

    target_include_directories(test_data_parallel_single_gpu PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_compile_definitions(test_data_parallel_single_gpu PRIVATE
        TENZOR_USE_CUDA
    )

    # CUDA kernel tests
    add_executable(test_cuda_kernels
        backends/test_cuda_kernels.cpp
    )

    target_link_libraries(test_cuda_kernels PRIVATE
        tenzor_core
        GTest::gtest_main
        CUDA::cudart
    )

    target_include_directories(test_cuda_kernels PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    set_target_properties(test_cuda_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    # CUDA training tests
    add_executable(test_cuda_training
        integration/test_cuda_training.cpp
    )

    target_link_libraries(test_cuda_training PRIVATE
        tenzor_core
        GTest::gtest_main
        CUDA::cudart
    )

    target_include_directories(test_cuda_training PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    set_target_properties(test_cuda_training PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    # CUDA concatenation tests
    add_executable(test_cuda_cat
        test_cuda_cat.cpp
    )

    target_link_libraries(test_cuda_cat PRIVATE
        tenzor_core
        CUDA::cudart
    )

    target_include_directories(test_cuda_cat PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    set_target_properties(test_cuda_cat PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    # FP16/BF16 kernel tests
    add_executable(test_fp16_kernels
        backends/test_fp16_kernels.cpp
    )

    target_link_libraries(test_fp16_kernels PRIVATE
        tenzor_core
        tenzor_backend_cuda
        GTest::gtest_main
        CUDA::cudart
    )

    target_include_directories(test_fp16_kernels PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    set_target_properties(test_fp16_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# Debug test for BatchNorm precision (commented out - tmp files not tracked)
# add_executable(test_batchnorm_precision_debug
#     /tmp/test_batchnorm_precision.cpp
# )
#
# target_link_libraries(test_batchnorm_precision_debug PRIVATE
#     tenzor_core
# )

# Debug test for variance computation (commented out - tmp files not tracked)
# add_executable(test_variance_debug
#     /tmp/test_variance_debug.cpp
# )
#
# target_link_libraries(test_variance_debug PRIVATE
#     tenzor_core
# )

# Debug test for multi-channel BatchNorm (commented out - tmp files not tracked)
# add_executable(test_multichannel
#     /tmp/test_multichannel.cpp
# )
#
# target_link_libraries(test_multichannel PRIVATE
#     tenzor_core
# )

# Debug test for Linear backward (commented out - tmp files not tracked)
# add_executable(test_linear_backward_debug
#     tmp/debug_linear_backward.cpp
# )
#
# target_link_libraries(test_linear_backward_debug PRIVATE
#     tenzor_core
# )

# Debug test for Attention flow (commented out - tmp files not tracked)
# add_executable(debug_attention_flow
#     tmp/debug_attention_flow.cpp
# )
#
# target_link_libraries(debug_attention_flow PRIVATE
#     tenzor_core
# )

# Debug test for module gradients (commented out - tmp files not tracked)
# add_executable(debug_module_grad
#     tmp/debug_module_grad.cpp
# )
#
# target_link_libraries(debug_module_grad PRIVATE
#     tenzor_core
# )

# Debug test for pointer addresses (commented out - tmp files not tracked)
# add_executable(debug_pointers
#     tmp/debug_pointers.cpp
# )
#
# target_link_libraries(debug_pointers PRIVATE
#     tenzor_core
# )

# Debug test for full backward pass (commented out - tmp files not tracked)
# add_executable(debug_full_backward
#     tmp/debug_full_backward.cpp
# )
#
# target_link_libraries(debug_full_backward PRIVATE
#     tenzor_core
# )

# Register tests with CTest
# Note: Using longer discovery timeout (30 seconds) because ROCm backend initialization can be slow
include(GoogleTest)
gtest_discover_tests(tenzor_unit_tests DISCOVERY_TIMEOUT 30)
gtest_discover_tests(tenzor_integration_tests DISCOVERY_TIMEOUT 30)
gtest_discover_tests(test_dropout DISCOVERY_TIMEOUT 30)
gtest_discover_tests(test_batchnorm2d DISCOVERY_TIMEOUT 30)
gtest_discover_tests(test_conv2d DISCOVERY_TIMEOUT 30)
gtest_discover_tests(test_pooling DISCOVERY_TIMEOUT 30)
gtest_discover_tests(test_conv1d DISCOVERY_TIMEOUT 30)
# gtest_discover_tests(test_convtranspose2d DISCOVERY_TIMEOUT 30)
gtest_discover_tests(test_schedulers DISCOVERY_TIMEOUT 30)
gtest_discover_tests(test_normalization DISCOVERY_TIMEOUT 30)
gtest_discover_tests(test_serialization DISCOVERY_TIMEOUT 30)
gtest_discover_tests(test_embedding DISCOVERY_TIMEOUT 30)
gtest_discover_tests(test_optimizers_extended DISCOVERY_TIMEOUT 30)
gtest_discover_tests(test_schedulers_advanced DISCOVERY_TIMEOUT 30)
gtest_discover_tests(test_losses_advanced DISCOVERY_TIMEOUT 30)
gtest_discover_tests(test_rnn DISCOVERY_TIMEOUT 30)
gtest_discover_tests(test_lstm DISCOVERY_TIMEOUT 30)
gtest_discover_tests(test_gru DISCOVERY_TIMEOUT 30)
gtest_discover_tests(test_attention DISCOVERY_TIMEOUT 30)
gtest_discover_tests(test_transformer DISCOVERY_TIMEOUT 30)
gtest_discover_tests(test_chunk DISCOVERY_TIMEOUT 30)
gtest_discover_tests(test_split_operation DISCOVERY_TIMEOUT 30)
gtest_discover_tests(test_bmm_autograd DISCOVERY_TIMEOUT 30)
gtest_discover_tests(test_grad_scaler DISCOVERY_TIMEOUT 30)
gtest_discover_tests(test_autocast DISCOVERY_TIMEOUT 30)
gtest_discover_tests(test_dataloader DISCOVERY_TIMEOUT 30)
gtest_discover_tests(test_fp16 DISCOVERY_TIMEOUT 30)
gtest_discover_tests(test_dtype_conversion DISCOVERY_TIMEOUT 30)
gtest_discover_tests(test_gradient_checkpoint DISCOVERY_TIMEOUT 30)
gtest_discover_tests(test_model_checkpoint DISCOVERY_TIMEOUT 30)
gtest_discover_tests(test_checkpoint DISCOVERY_TIMEOUT 30)

gtest_discover_tests(test_fused_ops DISCOVERY_TIMEOUT 30)
gtest_discover_tests(test_backend_ops_parameterized DISCOVERY_TIMEOUT 30)

# SIMD operations tests
add_executable(test_simd_ops
    unit/test_simd_ops.cpp
)

target_link_libraries(test_simd_ops PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_simd_ops DISCOVERY_TIMEOUT 30)

# Comparison operators tests
add_executable(test_comparison_operators
    test_comparison_operators.cpp
)

target_link_libraries(test_comparison_operators PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_comparison_operators DISCOVERY_TIMEOUT 30)

# Benchmark suite
add_executable(benchmark_suite
    benchmarks/benchmark_suite.cpp
)

target_link_libraries(benchmark_suite PRIVATE
    tenzor_core
)

# Debug DataLoader test
add_executable(debug_dataloader
    debug_dataloader.cpp
)

target_link_libraries(debug_dataloader PRIVATE
    tenzor_core
)

if(TENZOR_BUILD_CUDA)
    # CUDA tests with thread-safe handle management - can run in parallel
    gtest_discover_tests(test_memory_manager DISCOVERY_TIMEOUT 30)
    gtest_discover_tests(test_pinned_allocator DISCOVERY_TIMEOUT 30)
    gtest_discover_tests(test_transfer_engine DISCOVERY_TIMEOUT 30)
    gtest_discover_tests(test_transfer_benchmark DISCOVERY_TIMEOUT 30)
    gtest_discover_tests(test_offload_engine DISCOVERY_TIMEOUT 30)
    gtest_discover_tests(test_parameter_offload DISCOVERY_TIMEOUT 30)
    gtest_discover_tests(test_cuda_kernels DISCOVERY_TIMEOUT 30)
    gtest_discover_tests(test_cuda_training DISCOVERY_TIMEOUT 30)
    gtest_discover_tests(test_data_parallel DISCOVERY_TIMEOUT 30)
    gtest_discover_tests(test_data_parallel_single_gpu DISCOVERY_TIMEOUT 30)
    gtest_discover_tests(test_fp16_kernels DISCOVERY_TIMEOUT 30)
endif()

# OneAPI-specific tests
if(TENZOR_BUILD_ONEAPI)
    find_package(IntelSYCL QUIET)

    # OneAPI backend tests (comprehensive)
    add_executable(test_oneapi_backend
        backend/test_oneapi_backend.cpp
    )

    target_link_libraries(test_oneapi_backend PRIVATE
        tenzor_core
        GTest::gtest_main
    )

    # OneAPI device initialization can be slow, use 60 second timeout
    gtest_discover_tests(test_oneapi_backend DISCOVERY_TIMEOUT 60)
endif()

# ROCm-specific tests
if(TENZOR_BUILD_ROCM)
    find_package(hip REQUIRED)

    # ROCm backend tests
    add_executable(test_rocm_backend
        backends/test_rocm_backend.cpp
    )

    target_link_libraries(test_rocm_backend PRIVATE
        tenzor_core
        GTest::gtest_main
    )

    # ROCm kernel tests (comprehensive)
    add_executable(test_rocm_kernels
        backends/test_rocm_kernels.cpp
    )

    target_link_libraries(test_rocm_kernels PRIVATE
        tenzor_core
        GTest::gtest_main
        hip::host
    )

    target_include_directories(test_rocm_kernels PRIVATE
        ${ROCM_PATH}/include
    )

    # ROCm initialization can be slow, use 60 second timeout
    gtest_discover_tests(test_rocm_backend DISCOVERY_TIMEOUT 60)
    gtest_discover_tests(test_rocm_kernels DISCOVERY_TIMEOUT 60)
endif()
# Debug tests commented out - tmp files not tracked in git
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
add_executable(test_checkpoint_leaf_fix test_checkpoint_leaf_fix.cpp)
target_link_libraries(test_checkpoint_leaf_fix tenzor_core)
target_include_directories(test_checkpoint_leaf_fix PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Manual KLDiv debug test
add_executable(test_kldiv_manual test_kldiv_manual.cpp)
target_link_libraries(test_kldiv_manual PRIVATE tenzor_core GTest::gtest_main)
add_test(NAME KLDivManual.StepByStepBackward COMMAND test_kldiv_manual --gtest_filter=KLDivManual.StepByStepBackward)
add_test(NAME KLDivManual.SimpleSubtract COMMAND test_kldiv_manual --gtest_filter=KLDivManual.SimpleSubtract)
add_test(NAME KLDivManual.SimpleMultiply COMMAND test_kldiv_manual --gtest_filter=KLDivManual.SimpleMultiply)

# ONNX export tests
add_executable(test_onnx_export
    unit/test_onnx_export.cpp
)

target_link_libraries(test_onnx_export PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_onnx_export DISCOVERY_TIMEOUT 30)

# ONNX import tests
add_executable(test_onnx_import
    test_onnx_import.cpp
)

target_link_libraries(test_onnx_import PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_onnx_import DISCOVERY_TIMEOUT 30)

# Gradient checking tests
add_executable(test_gradcheck
    autograd/test_gradcheck.cpp
)

target_link_libraries(test_gradcheck PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_gradcheck DISCOVERY_TIMEOUT 30)

# ModelHub tests
add_executable(test_model_hub
    unit/test_model_hub.cpp
)

target_link_libraries(test_model_hub PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_model_hub DISCOVERY_TIMEOUT 30)

# BERT model tests
add_executable(test_bert
    unit/test_bert.cpp
)

target_link_libraries(test_bert PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_bert DISCOVERY_TIMEOUT 30)

# Classic CNN models tests
add_executable(test_classic_models
    unit/test_classic_models.cpp
)

target_link_libraries(test_classic_models PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_classic_models DISCOVERY_TIMEOUT 30)

# GPT model tests
add_executable(test_gpt
    unit/test_gpt.cpp
)

target_link_libraries(test_gpt PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_gpt DISCOVERY_TIMEOUT 30)

# ResNet model tests
add_executable(test_resnet
    unit/test_resnet.cpp
)

target_link_libraries(test_resnet PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_resnet DISCOVERY_TIMEOUT 30)

# RoBERTa model tests
add_executable(test_roberta
    unit/test_roberta.cpp
)

target_link_libraries(test_roberta PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_roberta DISCOVERY_TIMEOUT 30)

# ELECTRA model tests
add_executable(test_electra
    unit/test_electra.cpp
)

target_link_libraries(test_electra PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_electra DISCOVERY_TIMEOUT 30)

# ============================================================================
# Phase 9 Model Tests - Vision Models
# ============================================================================

# EfficientNet tests (B0-B7)
add_executable(test_efficientnet
    unit/test_efficientnet.cpp
)

target_link_libraries(test_efficientnet PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_efficientnet DISCOVERY_TIMEOUT 30)

# Vision Transformer tests (Base/Large/Huge)
add_executable(test_vit
    unit/test_vit.cpp
)

target_link_libraries(test_vit PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_vit DISCOVERY_TIMEOUT 30 PROPERTIES TIMEOUT 600)

# Swin Transformer tests (DISABLED - needs API fixes)
# add_executable(test_swin_transformer
#     unit/test_swin_transformer.cpp
# )

# target_link_libraries(test_swin_transformer PRIVATE
#     tenzor_core
#     GTest::gtest_main
# )

# gtest_discover_tests(test_swin_transformer DISCOVERY_TIMEOUT 30)

# ConvNeXt tests
add_executable(test_convnext
    unit/test_convnext.cpp
)

target_link_libraries(test_convnext PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_convnext DISCOVERY_TIMEOUT 30)

# MobileNet V2/V3 tests
add_executable(test_mobilenet_v2_v3
    unit/test_mobilenet_v2_v3.cpp
)

target_link_libraries(test_mobilenet_v2_v3 PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_mobilenet_v2_v3 DISCOVERY_TIMEOUT 30)

# Vision Components tests (PatchEmbedding, SE, MBConv, etc.) (DISABLED - depends on Swin)
# add_executable(test_vision_components
#     unit/test_vision_components.cpp
# )

# target_link_libraries(test_vision_components PRIVATE
#     tenzor_core
#     GTest::gtest_main
# )

# gtest_discover_tests(test_vision_components DISCOVERY_TIMEOUT 30)

# ============================================================================
# Phase 9 Model Tests - NLP Models
# ============================================================================

# RoBERTa and ELECTRA tests
add_executable(test_roberta_electra
    unit/test_roberta_electra.cpp
)

target_link_libraries(test_roberta_electra PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_roberta_electra DISCOVERY_TIMEOUT 30)

# ALBERT and T5 tests
add_executable(test_albert_t5
    unit/test_albert_t5.cpp
)

target_link_libraries(test_albert_t5 PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_albert_t5 DISCOVERY_TIMEOUT 30 PROPERTIES TIMEOUT 600)

# ============================================================================
# Phase 9 Model Tests - Detection and Segmentation Models
# ============================================================================

# Faster R-CNN tests
add_executable(test_faster_rcnn
    unit/test_faster_rcnn.cpp
)

target_link_libraries(test_faster_rcnn PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_faster_rcnn DISCOVERY_TIMEOUT 30 PROPERTIES TIMEOUT 1200)

# YOLO tests (v3 and v5)
add_executable(test_yolo
    unit/test_yolo.cpp
)

target_link_libraries(test_yolo PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_yolo DISCOVERY_TIMEOUT 30)

# Mask R-CNN tests
add_executable(test_mask_rcnn
    unit/test_mask_rcnn.cpp
)

target_link_libraries(test_mask_rcnn PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_mask_rcnn DISCOVERY_TIMEOUT 30 PROPERTIES TIMEOUT 1200)

# U-Net tests
add_executable(test_unet
    unit/test_unet.cpp
)

target_link_libraries(test_unet PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_unet DISCOVERY_TIMEOUT 30 PROPERTIES TIMEOUT 1200)

# DeepLab v3+ tests
add_executable(test_deeplabv3plus
    unit/test_deeplabv3plus.cpp
)

target_link_libraries(test_deeplabv3plus PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_deeplabv3plus DISCOVERY_TIMEOUT 30 PROPERTIES TIMEOUT 1200)

# Detection Operations tests (ROI, NMS, etc.)
add_executable(test_detection_ops
    unit/test_detection_ops.cpp
)

target_link_libraries(test_detection_ops PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_detection_ops DISCOVERY_TIMEOUT 30)

# ============================================================================
# Phase 10 Tests - Model Export, Optimization, and Compression
# ============================================================================

# JIT (Just-In-Time) compilation tests
add_executable(test_jit
    unit/test_jit.cpp
)

target_link_libraries(test_jit PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_jit DISCOVERY_TIMEOUT 30)

# Pruning tests
add_executable(test_pruning
    unit/test_pruning.cpp
)

target_link_libraries(test_pruning PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_pruning DISCOVERY_TIMEOUT 30)

# Knowledge distillation tests
add_executable(test_distillation
    unit/test_distillation.cpp
)

target_link_libraries(test_distillation PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_distillation DISCOVERY_TIMEOUT 30)

# Quantization tests
add_executable(test_quantization
    test_quantization.cpp
)

target_link_libraries(test_quantization PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_quantization DISCOVERY_TIMEOUT 30)

# ============================================================================
# Phase 11 Tests - Additional Backend Support
# ============================================================================
# Tests for OneAPI, Vulkan, Metal, and WebGPU backends
# ROCm tests intentionally EXCLUDED per user request (system crashes)

add_executable(test_phase11_backends
    test_phase11_backends.cpp
)

target_link_libraries(test_phase11_backends PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_phase11_backends DISCOVERY_TIMEOUT 30)

# ============================================================================
# CIoU Loss Tests - Complete IoU implementation for YOLO
# ============================================================================

add_executable(test_ciou_loss
    test_ciou_loss.cpp
)

target_link_libraries(test_ciou_loss PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_ciou_loss DISCOVERY_TIMEOUT 30)

# Backend Parity Test - Verify slice operations work correctly on all backends
add_executable(test_slice_backend_parity
    test_slice_backend_parity.cpp
)

target_link_libraries(test_slice_backend_parity PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_slice_backend_parity DISCOVERY_TIMEOUT 30)

# Slice Debug Test - Debug slice offset and data_ptr behavior
add_executable(test_slice_debug
    test_slice_debug.cpp
)

target_link_libraries(test_slice_debug PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_slice_debug DISCOVERY_TIMEOUT 30)

# Contiguous Fix Test - Verify .contiguous() fixes slice data access
add_executable(test_contiguous_fix
    test_contiguous_fix.cpp
)

target_link_libraries(test_contiguous_fix PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_contiguous_fix DISCOVERY_TIMEOUT 30)

# Tensor Lifetime Test - Debug tensor data pointer stability
add_executable(test_tensor_lifetime
    test_tensor_lifetime.cpp
)

target_link_libraries(test_tensor_lifetime PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_tensor_lifetime DISCOVERY_TIMEOUT 30)

# Backend Parity Example - Shows how to write multi-backend tests
add_executable(test_backend_parity_example
    test_backend_parity_example.cpp
)

target_link_libraries(test_backend_parity_example PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_backend_parity_example DISCOVERY_TIMEOUT 30)

# Tensor indexing operator test
add_executable(test_indexing_operator
    test_indexing_operator.cpp
)

target_link_libraries(test_indexing_operator PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_indexing_operator DISCOVERY_TIMEOUT 30)

# ============================================================================
# Phase 12 Tests - Enhanced Test Coverage and Validation
# ============================================================================

# Quantization conversion tests
add_executable(test_quantization_conversion
    test_quantization_conversion.cpp
)

target_link_libraries(test_quantization_conversion PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_quantization_conversion DISCOVERY_TIMEOUT 30)

# Mask R-CNN loss tests
add_executable(test_mask_rcnn_losses
    test_mask_rcnn_losses.cpp
)

target_link_libraries(test_mask_rcnn_losses PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_mask_rcnn_losses DISCOVERY_TIMEOUT 30)

# Vulkan complete operations tests
add_executable(test_vulkan_complete_ops
    test_vulkan_complete_ops.cpp
)

target_link_libraries(test_vulkan_complete_ops PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_vulkan_complete_ops DISCOVERY_TIMEOUT 30)

# Async operations tests
add_executable(test_async_ops
    unit/test_async_ops.cpp
)

target_link_libraries(test_async_ops PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_async_ops DISCOVERY_TIMEOUT 30)

# Repeat and Tile tests
add_executable(test_repeat_tile
    test_repeat_tile.cpp
)

target_link_libraries(test_repeat_tile PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_repeat_tile DISCOVERY_TIMEOUT 30)

# Utility module tests
add_subdirectory(utils)
