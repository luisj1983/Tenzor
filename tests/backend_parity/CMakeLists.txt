# Backend Parity Tests CMakeLists.txt
# Comprehensive test suite for backend parity across CPU, CUDA, OneAPI, Vulkan

cmake_minimum_required(VERSION 3.18)

# Find required packages
find_package(GTest REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GTEST_INCLUDE_DIRS}
)

# Helper function to create a parity test executable
function(add_parity_test TEST_NAME SOURCE_FILE)
    add_executable(${TEST_NAME} ${SOURCE_FILE})

    target_link_libraries(${TEST_NAME}
        tenzor_core
        GTest::gtest
        GTest::gtest_main
        ${CMAKE_THREAD_LIBS_INIT}
    )

    # Add test to CTest
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

    # Set test properties
    set_tests_properties(${TEST_NAME} PROPERTIES
        TIMEOUT ${TENZOR_TEST_TIMEOUT}  # 10 minute timeout for all tests
        LABELS "backend_parity"
    )

    # Install test
    install(TARGETS ${TEST_NAME}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/tests/backend_parity
    )
endfunction()

# ==============================================================================
# Backend Parity Test Executables
# ==============================================================================

# Operation parity tests (40+ math and reduction operations)
add_parity_test(test_operation_parity test_operation_parity.cpp)

# Neural network operation parity tests (30+ NN operations)
add_parity_test(test_nn_parity test_nn_parity.cpp)

# Gradient parity tests with numerical gradient checking
add_parity_test(test_gradient_parity test_gradient_parity.cpp)

# Data type parity tests (Float32, Float64, Int32, Int64)
add_parity_test(test_dtype_parity test_dtype_parity.cpp)

# Stress tests for large tensors and deep computation graphs
add_parity_test(test_backend_stress test_backend_stress.cpp)

# Numerical stability tests for edge cases
add_parity_test(test_numerical_stability test_numerical_stability.cpp)

# Performance regression baseline tests
add_parity_test(test_performance_regression test_performance_regression.cpp)

# Vulkan backend parity tests (Float64 shaders, Float16 buffer rounding, slice views)
add_parity_test(test_vulkan_parity test_vulkan_parity.cpp)

# Missing ops parity tests (9 CUDA-exclusive ops now available on all backends)
add_parity_test(test_missing_ops_parity test_missing_ops_parity.cpp)

# RNN parity tests (LSTM, GRU, Sparse, Quantization)
add_parity_test(test_rnn_parity test_rnn_parity.cpp)

# FFT parity tests
add_parity_test(test_fft_parity test_fft_parity.cpp)

# Sparse parity tests
add_parity_test(test_sparse_parity test_sparse_parity.cpp)

# Quantization parity tests
add_parity_test(test_quantization_parity test_quantization_parity.cpp)

# Nested tensor parity tests (nested_softmax, layer_norm, attention fwd+bwd)
add_parity_test(test_nested_parity test_nested_parity.cpp)

# Vision helpers and fused ops parity (Phase 3.1)
add_parity_test(test_vision_fused_parity test_vision_fused_parity.cpp)

# Sampling parity (Phase 3.8) — uses statistical-moment matching, not RNG parity
add_parity_test(test_sampling_parity test_sampling_parity.cpp)

# Autograd transforms parity (Phase 4.1) — jvp, vjp, jacobian, hessian, vmap
add_parity_test(test_grad_transform_parity test_grad_transform_parity.cpp)

# AMP / GradScaler parity (Phase 4.2)
add_parity_test(test_amp_parity test_amp_parity.cpp)

# Custom op cross-backend parity (Phase 4.3)
add_parity_test(test_custom_op_parity test_custom_op_parity.cpp)

# Phase 5: ALiBi, RoPE, window ops parity
add_parity_test(test_phase5_parity test_phase5_parity.cpp)

# Phase 5.1: distributions parity (log_prob / entropy / mean / variance)
add_parity_test(test_distributions_parity test_distributions_parity.cpp)

# Phase 5.3: FP8 parity (hardware-dependent; graceful skip via try/catch)
add_parity_test(test_fp8_parity test_fp8_parity.cpp)

# Phase 5.4/5.5: MoE and HRM parity
add_parity_test(test_moe_hrm_parity test_moe_hrm_parity.cpp)

# Phase 3.6: extended linalg parity (pinv, lstsq, matrix_exp)
add_parity_test(test_linalg_extended_parity test_linalg_extended_parity.cpp)

# Phase 4.4: backward-through-JIT parity (DISABLED_ pending upstream fix)
add_parity_test(test_jit_autograd_parity test_jit_autograd_parity.cpp)

# Extended math parity tests (log2, log10, exp2, erf, erfc, rsqrt, etc.)
add_parity_test(test_extended_math_parity test_extended_math_parity.cpp)

# Trigonometric parity tests (asin, acos, sinh, cosh, asinh, acosh, atanh, etc.)
add_parity_test(test_trig_parity test_trig_parity.cpp)

# Shape operations parity tests (reshape, transpose, permute, stack, cat, etc.)
add_parity_test(test_shape_ops_parity test_shape_ops_parity.cpp)

# Comparison parity tests (eq, ne, lt, le, gt, ge + broadcasting)
add_parity_test(test_comparison_parity test_comparison_parity.cpp)

# Logical parity tests (and, or, not, xor + combined expressions)
add_parity_test(test_logical_parity test_logical_parity.cpp)

# Indexing parity tests (index_select, gather, scatter, masked_select, where, etc.)
add_parity_test(test_indexing_parity test_indexing_parity.cpp)

# Manipulation parity tests (triu, tril, diag, trace, flip, roll)
add_parity_test(test_manipulation_parity test_manipulation_parity.cpp)

# Cumulative/sorting parity tests (cumsum, cumprod, topk, sort, cummax, cummin)
add_parity_test(test_cumulative_parity test_cumulative_parity.cpp)

# Complex number parity tests (conj, real, imag, angle, polar, complex arithmetic)
add_parity_test(test_complex_parity test_complex_parity.cpp)

# Special math parity tests (gamma, lgamma, digamma, bessel, sinc, erfinv, etc.)
add_parity_test(test_special_math_parity test_special_math_parity.cpp)

# Linear algebra parity tests (det, inv, solve, svd, qr, eigh, cholesky, lu, etc.)
add_parity_test(test_linalg_parity test_linalg_parity.cpp)

# NN Convolution parity tests (Conv1d, Conv3d, ConvTranspose, DeformableConv, groups, depthwise)
add_parity_test(test_nn_conv_parity test_nn_conv_parity.cpp)

# NN Pooling parity tests (MaxPool1d/3d, AvgPool1d/3d, Adaptive, LP, Fractional)
add_parity_test(test_nn_pooling_parity test_nn_pooling_parity.cpp)

# NN RNN parity tests (LSTMCell, GRUCell, RNNCell, LSTM, GRU, RNN)
add_parity_test(test_nn_rnn_parity test_nn_rnn_parity.cpp)

# NN Linear/Embedding parity tests (Linear, Bilinear, EmbeddingBag, LazyLinear)
add_parity_test(test_nn_linear_emb_parity test_nn_linear_emb_parity.cpp)

# NN Normalization parity tests (InstanceNorm, RMSNorm, LocalResponseNorm, SyncBatchNorm)
add_parity_test(test_nn_norm_parity test_nn_norm_parity.cpp)

# NN Activation parity tests (SELU, Mish, Softplus, Hardswish, PReLU, etc.)
add_parity_test(test_nn_activation_parity test_nn_activation_parity.cpp)

# NN Loss parity tests (HuberLoss, NLLLoss, KLDivLoss, TripletMarginLoss, etc.)
add_parity_test(test_nn_loss_parity test_nn_loss_parity.cpp)

# NN Transformer parity tests (TransformerEncoder/Decoder, MultiheadAttention, etc.)
add_parity_test(test_nn_transformer_parity test_nn_transformer_parity.cpp)

# Gradient arithmetic parity tests (Sub, Div, Pow, Sqrt, Exp, Log, etc.)
add_parity_test(test_grad_arithmetic_parity test_grad_arithmetic_parity.cpp)

# Gradient activation parity tests (LeakyReLU, ELU, SELU, Mish, Softmax, etc.)
add_parity_test(test_grad_activation_parity test_grad_activation_parity.cpp)

# Gradient reduction parity tests (Sum, Mean, Var, Std, Norm, LogSumExp, etc.)
add_parity_test(test_grad_reduction_parity test_grad_reduction_parity.cpp)

# Gradient linear algebra parity tests (MatMul, Bmm, Linear, Det, etc.)
add_parity_test(test_grad_linalg_parity test_grad_linalg_parity.cpp)

# Gradient NN layer parity tests (Conv, Pool, BatchNorm, LayerNorm, Embedding, RNN)
add_parity_test(test_grad_nn_parity test_grad_nn_parity.cpp)

# Gradient shape/indexing parity tests (Reshape, Transpose, Gather, Cat, Split)
add_parity_test(test_grad_shape_parity test_grad_shape_parity.cpp)

# Numerical gradcheck parity tests (finite difference verification on each backend)
add_parity_test(test_gradcheck_parity test_gradcheck_parity.cpp)

# Kernel completeness tests (Phase 4A — required ops registered on each backend)
add_parity_test(test_kernel_completeness test_kernel_completeness.cpp)

# Registration report (Phase 4B — informational coverage matrix)
add_parity_test(test_registration_report test_registration_report.cpp)

# ==============================================================================
# Test Groups
# ==============================================================================

# Quick parity tests (basic operations only)
add_custom_target(test_parity_quick
    COMMAND ${CMAKE_CTEST_COMMAND} -L "backend_parity" -R "test_operation_parity|test_dtype_parity" --output-on-failure
    DEPENDS test_operation_parity test_dtype_parity
    COMMENT "Running quick backend parity tests..."
)

# Full parity test suite
add_custom_target(test_parity_full
    COMMAND ${CMAKE_CTEST_COMMAND} -L "backend_parity" --output-on-failure
    DEPENDS
        test_operation_parity
        test_nn_parity
        test_gradient_parity
        test_dtype_parity
        test_backend_stress
        test_numerical_stability
        test_performance_regression
        test_vulkan_parity
        test_extended_math_parity
        test_trig_parity
        test_shape_ops_parity
        test_comparison_parity
        test_logical_parity
        test_indexing_parity
        test_manipulation_parity
        test_cumulative_parity
        test_complex_parity
        test_special_math_parity
        test_linalg_parity
        test_nn_conv_parity
        test_nn_pooling_parity
        test_nn_rnn_parity
        test_nn_linear_emb_parity
        test_nn_norm_parity
        test_nn_activation_parity
        test_nn_loss_parity
        test_nn_transformer_parity
        test_grad_arithmetic_parity
        test_grad_activation_parity
        test_grad_reduction_parity
        test_grad_linalg_parity
        test_grad_nn_parity
        test_grad_shape_parity
        test_gradcheck_parity
        test_kernel_completeness
        test_registration_report
    COMMENT "Running full backend parity test suite..."
)

# Stress tests only
add_custom_target(test_parity_stress
    COMMAND ${CMAKE_CTEST_COMMAND} -L "backend_parity" -R "test_backend_stress|test_numerical_stability" --output-on-failure
    DEPENDS test_backend_stress test_numerical_stability
    COMMENT "Running backend stress tests..."
)

# Performance tests only
add_custom_target(test_parity_performance
    COMMAND ${CMAKE_CTEST_COMMAND} -L "backend_parity" -R "test_performance_regression" --output-on-failure
    DEPENDS test_performance_regression
    COMMENT "Running backend performance regression tests..."
)

# Gradient tests only
add_custom_target(test_parity_gradients
    COMMAND ${CMAKE_CTEST_COMMAND} -L "backend_parity" -R "test_grad.*_parity" --output-on-failure
    DEPENDS test_gradient_parity test_grad_arithmetic_parity test_grad_activation_parity
            test_grad_reduction_parity test_grad_linalg_parity test_grad_nn_parity
            test_grad_shape_parity test_gradcheck_parity
    COMMENT "Running gradient parity tests..."
)

# ==============================================================================
# Test Documentation
# ==============================================================================

# Print test information
message(STATUS "Backend Parity Tests Configuration:")
message(STATUS "  - Operation parity tests: test_operation_parity")
message(STATUS "  - NN operation parity tests: test_nn_parity")
message(STATUS "  - Gradient parity tests: test_gradient_parity")
message(STATUS "  - DType parity tests: test_dtype_parity")
message(STATUS "  - Stress tests: test_backend_stress")
message(STATUS "  - Numerical stability tests: test_numerical_stability")
message(STATUS "  - Performance regression tests: test_performance_regression")
message(STATUS "  - Vulkan parity tests: test_vulkan_parity")
message(STATUS "")
message(STATUS "Run test groups with:")
message(STATUS "  - make test_parity_quick      # Quick tests only")
message(STATUS "  - make test_parity_full       # Full test suite")
message(STATUS "  - make test_parity_stress     # Stress tests only")
message(STATUS "  - make test_parity_performance # Performance tests only")
message(STATUS "  - make test_parity_gradients  # Gradient tests only")

# ==============================================================================
# Install utilities header
# ==============================================================================

install(FILES parity_test_utils.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tenzor/tests
)
