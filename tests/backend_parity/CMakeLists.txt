# Backend Parity Tests CMakeLists.txt
# Comprehensive test suite for backend parity across CPU, CUDA, OneAPI, Vulkan

cmake_minimum_required(VERSION 3.18)

# Find required packages
find_package(GTest REQUIRED)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GTEST_INCLUDE_DIRS}
)

# Path to the on-disk golden-tensor directory. Baked into each parity test
# binary at compile time; overridable at runtime via $TENZOR_GOLDEN_DIR.
set(TENZOR_GOLDEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/golden" CACHE PATH
    "Directory containing recorded parity goldens")

# Helper function to create a parity test executable.
#
# Each *_parity binary is a TEST_P harness parametrized over the available
# backends (see e.g. INSTANTIATE_TEST_SUITE_P(..., AllBackends, …) in the
# .cpp files). We discover individual TEST_P cases via tenzor_discover_tests
# with BACKENDS multi so that:
#   - `ctest -R cuda` finds the parametrized "cuda" cases by name match,
#   - `ctest -L cuda` finds them by label match,
#   - `ctest -L backend_parity` keeps working.
function(add_parity_test TEST_NAME SOURCE_FILE)
    # Optional 3rd+ positional args (or LABELS keyword) -> extra labels.
    # Used for the smoke-tagged subset (test_kernel_completeness, etc.).
    cmake_parse_arguments(PT "" "" "LABELS" ${ARGN})
    set(_labels "backend_parity")
    foreach(_extra IN LISTS PT_LABELS)
        set(_labels "${_labels}\;${_extra}")
    endforeach()

    add_executable(${TEST_NAME} ${SOURCE_FILE})

    target_link_libraries(${TEST_NAME}
        tenzor_core
        GTest::gtest
        GTest::gtest_main
        ${CMAKE_THREAD_LIBS_INIT}
    )

    target_compile_definitions(${TEST_NAME} PRIVATE
        TENZOR_TEST_GOLDEN_DIR="${TENZOR_GOLDEN_DIR}"
    )

    # Discover per-case rather than one CTest entry per binary so that
    # the parametrized backend value lands in the test name and `-R`
    # filtering works. PRE_TEST mode keeps discovery in step with builds
    # when only the source changes.
    tenzor_discover_tests(${TEST_NAME}
        DISCOVERY_TIMEOUT 60
        DISCOVERY_MODE PRE_TEST
        LABELS "${_labels}"
        BACKENDS multi)

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

# Extended operation parity tests (indexing, shape, cumulative, advanced math)
add_parity_test(test_operation_parity_extended test_operation_parity_extended.cpp)

# Multi-dtype parity tests (Float16, BFloat16, Float64 across key ops)
add_parity_test(test_operation_parity_multidtype test_operation_parity_multidtype.cpp)

# Stride parity tests — exercises each op with non-contiguous inputs to catch
# kernels that read strides from shape instead of the tensor's actual strides.
add_parity_test(test_stride_parity test_stride_parity.cpp)

# C3: stride-variant parity for STRUCTURED ops (Conv, Pool, Norm, MatMul).
# Companion to test_stride_parity (which covers elementwise math).
add_parity_test(test_stride_structured_parity test_stride_structured_parity.cpp)

# BFloat16 parity tests — exercises each op with BF16 inputs to catch
# dispatchers that silently fall through to the F32 shader and corrupt data.
add_parity_test(test_bf16_parity test_bf16_parity.cpp)

# Neural network operation parity tests (30+ NN operations)
add_parity_test(test_nn_parity test_nn_parity.cpp)

# Gradient parity tests with numerical gradient checking
add_parity_test(test_gradient_parity test_gradient_parity.cpp)

# Phase 6 — cross-backend backward parity for the highest-impact ops, using
# the test_gradient_parity helper. Catches backward kernels that diverge
# between backends (which the many forward-only parity files cannot detect).
add_parity_test(test_gradient_parity_critical test_gradient_parity_critical.cpp)

# Data type parity tests (Float32, Float64, Int32, Int64)
add_parity_test(test_dtype_parity test_dtype_parity.cpp)

# Stress tests for large tensors and deep computation graphs
add_parity_test(test_backend_stress test_backend_stress.cpp)

# Numerical stability tests for edge cases
add_parity_test(test_numerical_stability test_numerical_stability.cpp)

# Performance regression baseline tests
add_parity_test(test_performance_regression test_performance_regression.cpp)

# Missing ops parity tests (9 CUDA-exclusive ops now available on all backends)
add_parity_test(test_missing_ops_parity test_missing_ops_parity.cpp)

# RNN parity tests (LSTM, GRU, Sparse, Quantization)
add_parity_test(test_rnn_parity test_rnn_parity.cpp)

# FFT parity tests
add_parity_test(test_fft_parity test_fft_parity.cpp)

# Sparse parity tests
add_parity_test(test_sparse_parity test_sparse_parity.cpp)

# Sparse parity with realistic density patterns (random, banded, block-diag)
add_parity_test(test_sparse_real_patterns test_sparse_real_patterns.cpp)

# Quantization parity tests
add_parity_test(test_quantization_parity test_quantization_parity.cpp)

# Inf-G (deferred → landed): multi-backend INT8 e2e quantization parity.
# Each backend test gracefully GTEST_SKIPs when the device is unavailable.
# CC.16: BACKENDS multi so the `cpu_` ctest prefix does not get prepended to
# tests that exercise CUDA/ROCm/Vulkan/OneAPI — the test names themselves
# carry the device under test, and tagging "multi" attaches every backend
# label so `ctest -L cuda` etc. still finds these.
add_executable(test_quantization_e2e_parity test_quantization_e2e_parity.cpp)
target_link_libraries(test_quantization_e2e_parity PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_quantization_e2e_parity LABELS "backend_parity;quantization" BACKENDS multi DISCOVERY_TIMEOUT 60)

# Real-int8 quantized-kernel parity (QuantizedLinear across backends)
add_parity_test(test_quantized_kernel_parity test_quantized_kernel_parity.cpp)

# Quantized conv2d with batch*out_channels > 65535 — guards the CUDA gridDim.z
# overflow fix (cap grid_z at 65535 + grid-stride over flattened (b, oc)).
add_parity_test(test_quantized_conv2d_large_batch test_quantized_conv2d_large_batch.cpp)

# Nested tensor parity tests (nested_softmax, layer_norm, attention fwd+bwd)
add_parity_test(test_nested_parity test_nested_parity.cpp)

# Nested layer norm with NON-power-of-two inner dim — guards the nested LN
# tree-reduction fix (a power-of-two-only reduction drops the per-row tail).
add_parity_test(test_nested_layernorm_parity test_nested_layernorm_parity.cpp)

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

# Cross-backend (non-CPU-reference) parity — compares all available pairs
add_parity_test(test_cross_backend_pairs test_cross_backend_pairs.cpp)

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

# Kernel completeness tests (Phase 4A — required ops registered on each backend).
# Smoke-tagged because <2s and catches dispatch-table regressions instantly.
add_parity_test(test_kernel_completeness test_kernel_completeness.cpp LABELS smoke)

# Registration report (Phase 4B — informational coverage matrix).
# Smoke-tagged for the same reason as test_kernel_completeness.
add_parity_test(test_registration_report test_registration_report.cpp LABELS smoke)

# Audit 2026-05-02 I.3: skip-policy meta-test.
add_parity_test(test_skip_policy test_skip_policy.cpp)

# ==============================================================================
# Audit-driven additions (2026-05-02 test-coverage plan, Workstream C)
# ==============================================================================

# In-place ops parity (covers AddInplace..GeluInplace; was 0% covered by name).
# Smoke-tagged because this fast (~16s) cross-backend test runs in the smoke set.
add_parity_test(test_inplace_ops_parity test_inplace_ops_parity.cpp LABELS smoke)

# Advanced (NumPy-style fancy) indexing parity
add_parity_test(test_advanced_index_parity test_advanced_index_parity.cpp)

# Bitwise integer ops parity (BitwiseAnd/Or/Xor/Not/LeftShift/RightShift × Int8-Int64)
add_parity_test(test_bitwise_parity test_bitwise_parity.cpp)

# Signal-processing tail (IDCT, MelScale, MFCC) — extends the existing
# STFT/ISTFT/DCT coverage in test_special_math_parity.
add_parity_test(test_signal_processing_full_parity test_signal_processing_full_parity.cpp)

# Scatter family parity (SelectScatter, SliceScatter, DiagonalScatter,
# MaskedScatter, TakeAlongDim) — none had dedicated parity coverage before.
add_parity_test(test_index_scatter_parity test_index_scatter_parity.cpp)

# DeformableConv2d forward + 3 backward variants — backward had zero parity
# refs even though kernels are registered everywhere.
add_parity_test(test_deformable_conv2d_backward_parity test_deformable_conv2d_backward_parity.cpp)

# EmbeddingBag forward + backward parity — backward had zero parity refs.
add_parity_test(test_embedding_bag_backward_parity test_embedding_bag_backward_parity.cpp)

# Phase-5 PyTorch-parity math (Deg2Rad, Rad2Deg, Logit, Signbit, FloatPower,
# Xlog1py, Ldexp, IsReal, IsPosInf, IsNegInf, Frexp).
add_parity_test(test_phase5_math_parity test_phase5_math_parity.cpp)

# Numerically-stable math (LogAddExp, LogAddExp2, XLogY, CosineSimilarity, Renorm).
add_parity_test(test_stable_math_parity test_stable_math_parity.cpp)

# NaN-aware stats + quantiles + Cov/Corrcoef.
add_parity_test(test_nanstats_parity test_nanstats_parity.cpp)

# Audit E1: asymmetric (per-axis) Conv2d stride/padding/dilation.
add_parity_test(test_conv2d_asymmetric_parity test_conv2d_asymmetric_parity.cpp)

# Wave C: FlexAttention with user-registered score_mod (ID >= 3) on Vulkan + ROCm.
add_parity_test(test_flex_attention_user_mod_parity test_flex_attention_user_mod_parity.cpp)

# Wave D: Vulkan FlashAttention tiled shader native causal masking (lifts stale gate).
add_parity_test(test_vulkan_flash_attn_causal_parity test_vulkan_flash_attn_causal_parity.cpp)

# JIT-R158/JIT-R159: ROCm tiled FlashAttention kernel seq_len_q/seq_len_k mixup
# and causal-offset tile-skip fix, for seq_len_q != seq_len_k (KV-cache) calls.
add_parity_test(test_rocm_flash_attn_kvcache_parity test_rocm_flash_attn_kvcache_parity.cpp)

# JIT-R160/JIT-R161/JIT-R162: cross-backend GQA (H_kv < H_q) consistency for a
# direct OpId::FlashAttention/FusedAttention dispatch.
add_parity_test(test_flash_attention_gqa_parity test_flash_attention_gqa_parity.cpp)

# JIT-R171: complex sign() must propagate NaN (not silently return 0+0i) on
# every backend, not just CPU.
add_parity_test(test_complex_sign_nan_parity test_complex_sign_nan_parity.cpp)

# JIT-R192: native (non-MLIR) comparison-predicate kernels must agree with
# CPU on NaN-comparison semantics across every backend.
add_parity_test(test_logical_predicates_nan_parity test_logical_predicates_nan_parity.cpp)

# Wave E: CUDA fused kernels with native F16/BF16 (F32 accumulator inside kernel).
add_parity_test(test_cuda_fused_dtype_parity test_cuda_fused_dtype_parity.cpp)

# Wave F2: OneAPI im2col native F16/BF16.
add_parity_test(test_oneapi_im2col_dtype_parity test_oneapi_im2col_dtype_parity.cpp)

# Wave G: CPU sparse SpMV/SpMM native F16/BF16/Int32/Int64 (Acc=F32 for half).
add_parity_test(test_cpu_sparse_dtype_native test_cpu_sparse_dtype_native.cpp)

# Wave H4: ROCm InterpolateBackward native nearest-mode scatter.
add_parity_test(test_rocm_interpolate_nearest_backward test_rocm_interpolate_nearest_backward.cpp)

# Wave Inf-A: CPU FFT non-MKL DFT fallback (covers any signal length).
add_parity_test(test_cpu_fft_fallback test_cpu_fft_fallback.cpp)

# Wave Inf-B: distribution entropy/mean/variance closed-form additions.
add_parity_test(test_distributions_closed_forms test_distributions_closed_forms.cpp)

# Stream S18: CPU DepthwiseConv1d / DepthwiseConv3d native fast paths.
# CPU-only fixture (TEST_F not TEST_P) — the new OpIds are registered on the
# CPU backend in this stream; other backends still have throw-stubs. We use
# a bespoke target rather than add_parity_test so test names aren't prefixed
# with backend labels (the test fixture itself pins Device::cpu()).
add_executable(test_depthwise_conv test_depthwise_conv.cpp)
target_link_libraries(test_depthwise_conv PRIVATE
    tenzor_core
    GTest::gtest
    GTest::gtest_main
    ${CMAKE_THREAD_LIBS_INIT}
)
tenzor_discover_tests(test_depthwise_conv
    DISCOVERY_TIMEOUT 60
    LABELS "backend_parity"
    BACKENDS cpu)

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
        test_stride_parity
        test_bf16_parity
        test_nn_parity
        test_gradient_parity
        test_dtype_parity
        test_backend_stress
        test_numerical_stability
        test_performance_regression
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
