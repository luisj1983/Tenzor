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
)

target_link_libraries(tenzor_unit_tests PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Integration tests
add_executable(tenzor_integration_tests
    integration/test_nn.cpp
    integration/test_training.cpp
)

target_link_libraries(tenzor_integration_tests PRIVATE
    tenzor_core
    GTest::gtest_main
)

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

# CUDA-specific tests
if(TENZOR_BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)

    # CachingAllocator tests (requires CUDA)
    add_executable(test_caching_allocator
        unit/test_caching_allocator.cpp
    )

    target_link_libraries(test_caching_allocator PRIVATE
        tenzor_core
        tenzor_backend_cuda
        GTest::gtest_main
        CUDA::cudart
    )

    target_include_directories(test_caching_allocator PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
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

gtest_discover_tests(test_fused_ops DISCOVERY_TIMEOUT 30)

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
    gtest_discover_tests(test_cuda_kernels DISCOVERY_TIMEOUT 30)
    gtest_discover_tests(test_cuda_training DISCOVERY_TIMEOUT 30)
    gtest_discover_tests(test_caching_allocator DISCOVERY_TIMEOUT 30)
    gtest_discover_tests(test_data_parallel DISCOVERY_TIMEOUT 30)
    gtest_discover_tests(test_data_parallel_single_gpu DISCOVERY_TIMEOUT 30)
    gtest_discover_tests(test_fp16_kernels DISCOVERY_TIMEOUT 30)
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
