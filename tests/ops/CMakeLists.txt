# =============================================================================
# tests/ops/CMakeLists.txt — Operation-level tests
# =============================================================================

# Advanced operations multi-dtype tests
add_executable(test_advanced_ops_multidtype test_advanced_ops_multidtype.cpp)
target_link_libraries(test_advanced_ops_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Shape operations multi-dtype tests
add_executable(test_shape_ops_multidtype test_shape_ops_multidtype.cpp)
target_link_libraries(test_shape_ops_multidtype PRIVATE tenzor_core GTest::gtest_main)

# ============================================================================
# Test Registration
# ============================================================================
# FP8 operations tests
add_executable(test_fp8_ops test_fp8_ops.cpp)
target_link_libraries(test_fp8_ops PRIVATE tenzor_core GTest::gtest_main)

# LU decomposition tests
add_executable(test_linalg_lu test_linalg_lu.cpp)
target_link_libraries(test_linalg_lu PRIVATE tenzor_core GTest::gtest_main)

# linalg::norm string-ord variants ("1", "-1", "2", "-2", "inf", "-inf",
# "nuc") + vector_norm(ord=0) exact L0 count. Release-prep S6.
add_executable(test_linalg_norm_ords test_linalg_norm_ords.cpp)
target_link_libraries(test_linalg_norm_ords PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_linalg_norm_ords LABELS "ops" DISCOVERY_TIMEOUT 60)

# Edge case tests (NaN, Inf, empty tensors, size-1 tensors, dtype precision)
add_executable(test_ops_edge_cases test_edge_cases.cpp)
target_link_libraries(test_ops_edge_cases PRIVATE tenzor_core GTest::gtest_main)

tenzor_discover_tests(test_advanced_ops_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_shape_ops_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_fp8_ops LABELS "ops" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_linalg_lu LABELS "ops" DISCOVERY_TIMEOUT 60)
# Numerical gradient verification for core ops
add_executable(test_numerical_gradients test_numerical_gradients.cpp)
target_link_libraries(test_numerical_gradients PRIVATE tenzor_core GTest::gtest_main)

tenzor_discover_tests(test_ops_edge_cases LABELS "ops" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_numerical_gradients LABELS "ops" DISCOVERY_TIMEOUT 60)

# ============================================================================
# Extended math multi-backend tests (Log2, Log10, Log1p, Exp2, Expm1, Erf, etc.)
# ============================================================================
add_executable(test_extended_math_multidtype test_extended_math_multidtype.cpp)
target_link_libraries(test_extended_math_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_extended_math_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Special math functions (Gamma, Bessel, Zeta, etc.)
add_executable(test_special_math_multidtype test_special_math_multidtype.cpp)
target_link_libraries(test_special_math_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_special_math_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# 1D pooling operations
add_executable(test_pooling1d_multidtype test_pooling1d_multidtype.cpp)
target_link_libraries(test_pooling1d_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_pooling1d_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# 3D pooling operations
add_executable(test_pooling3d_multidtype test_pooling3d_multidtype.cpp)
target_link_libraries(test_pooling3d_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_pooling3d_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# MaxUnpool2d/3d (fwd + backward with synthetic indices)
add_executable(test_maxunpool_multidtype test_maxunpool_multidtype.cpp)
target_link_libraries(test_maxunpool_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_maxunpool_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# FractionalMaxPool2d/3d (fwd + backward)
add_executable(test_fractional_maxpool_multidtype test_fractional_maxpool_multidtype.cpp)
target_link_libraries(test_fractional_maxpool_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_fractional_maxpool_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# In-place arithmetic (add_, sub_, mul_, div_) across all backends and dtypes
add_executable(test_inplace_arithmetic_multidtype test_inplace_arithmetic_multidtype.cpp)
target_link_libraries(test_inplace_arithmetic_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_inplace_arithmetic_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Signal processing (STFT, ISTFT, CDist)
add_executable(test_signal_processing_multidtype test_signal_processing_multidtype.cpp)
target_link_libraries(test_signal_processing_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_signal_processing_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Complex number operations (Conj, Real, Imag, Angle, Polar)
add_executable(test_complex_ops_multidtype test_complex_ops_multidtype.cpp)
target_link_libraries(test_complex_ops_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_complex_ops_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Search and sampling (SearchSorted, GumbelSoftmax)
add_executable(test_search_sampling_multidtype test_search_sampling_multidtype.cpp)
target_link_libraries(test_search_sampling_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_search_sampling_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# New ops (NormalSample, ExponentialSample, ndtr, trapezoid, gradient, pairwise_distance, segment_reduce, etc.)
add_executable(test_new_ops test_new_ops.cpp)
target_link_libraries(test_new_ops PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_new_ops LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Convolution gaps (Conv1d, ConvTranspose1d/2d/3d)
add_executable(test_conv_gaps_multidtype test_conv_gaps_multidtype.cpp)
target_link_libraries(test_conv_gaps_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_conv_gaps_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Window functions (Hann, Hamming, Blackman, Bartlett, Kaiser)
add_executable(test_windows_multidtype test_windows_multidtype.cpp)
target_link_libraries(test_windows_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_windows_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Edge cases multi-backend (NaN, Inf, empty tensors, size-1 tensors)
add_executable(test_edge_cases_ops_multidtype test_edge_cases_multidtype.cpp)
target_link_libraries(test_edge_cases_ops_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_edge_cases_ops_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Numerical gradients multi-backend
add_executable(test_numerical_gradients_multidtype test_numerical_gradients_multidtype.cpp)
target_link_libraries(test_numerical_gradients_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_numerical_gradients_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# LU decomposition multi-backend multi-dtype
add_executable(test_linalg_lu_multidtype test_linalg_lu_multidtype.cpp)
target_link_libraries(test_linalg_lu_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_linalg_lu_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# FP8 operations multi-backend
add_executable(test_fp8_ops_multidtype test_fp8_ops_multidtype.cpp)
target_link_libraries(test_fp8_ops_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_fp8_ops_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Conv3d multi-backend
add_executable(test_conv3d_multidtype test_conv3d_multidtype.cpp)
target_link_libraries(test_conv3d_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_conv3d_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Activation backward multi-backend
add_executable(test_activation_backward_multidtype test_activation_backward_multidtype.cpp)
target_link_libraries(test_activation_backward_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_activation_backward_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Fused ops dispatch multi-backend
add_executable(test_fused_ops_dispatch_multidtype test_fused_ops_dispatch_multidtype.cpp)
target_link_libraries(test_fused_ops_dispatch_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_fused_ops_dispatch_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# EmbeddingBag multi-backend
add_executable(test_embedding_bag_multidtype test_embedding_bag_multidtype.cpp)
target_link_libraries(test_embedding_bag_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_embedding_bag_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# BatchNorm dispatch multi-backend
add_executable(test_batchnorm_dispatch_multidtype test_batchnorm_dispatch_multidtype.cpp)
target_link_libraries(test_batchnorm_dispatch_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_batchnorm_dispatch_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# RNN cells dispatch multi-backend
add_executable(test_rnn_cells_dispatch_multidtype test_rnn_cells_dispatch_multidtype.cpp)
target_link_libraries(test_rnn_cells_dispatch_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_rnn_cells_dispatch_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Previously untested ops: inplace activations, LogSumExp, GridSample, AffineGrid
add_executable(test_untested_ops_multidtype test_untested_ops_multidtype.cpp)
target_link_libraries(test_untested_ops_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_untested_ops_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# OpIds flagged by scripts/audit_op_coverage.py as having zero test references
# (Frac, Heaviside, NanToNum, Nextafter, Gcd/Lcm, Addcmul/Addcdiv, Rad2Deg,
# Signbit, LogAddExp2, XLogY, Xlog1py, Entr, IsNan/IsInf/IsFinite, MaskedScatter,
# TrilIndices/TriuIndices, Ldexp, Frexp).
add_executable(test_missing_ops_multidtype test_missing_ops_multidtype.cpp)
target_link_libraries(test_missing_ops_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_missing_ops_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Linalg OpIds flagged as untested: Det, SLogDet, Inv, Cholesky, QR, SVD, Eigh,
# LU, Solve, SolveTriangular, CholeskySolve, CholeskyInverse, MatrixExp, Pinv,
# MatrixNorm, VectorNorm, VecDot, Pdist, Geqrf, Ormqr, HouseholderProduct,
# MatrixRank, LDLFactor, LDLSolve.
add_executable(test_linalg_missing_multidtype test_linalg_missing_multidtype.cpp)
target_link_libraries(test_linalg_missing_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_linalg_missing_multidtype LABELS "ops" DISCOVERY_TIMEOUT 120 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# 1D/3D pool + Conv3d + ConvTranspose + MaxUnpool + DeformableConv2d OpIds.
add_executable(test_pool_conv_missing_multidtype test_pool_conv_missing_multidtype.cpp)
target_link_libraries(test_pool_conv_missing_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_pool_conv_missing_multidtype LABELS "ops" DISCOVERY_TIMEOUT 120 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Stats / FFT / indexing / diag / scatter / misc OpIds.
add_executable(test_stats_misc_missing_multidtype test_stats_misc_missing_multidtype.cpp)
target_link_libraries(test_stats_misc_missing_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_stats_misc_missing_multidtype LABELS "ops" DISCOVERY_TIMEOUT 120 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Final 29 untested OpIds (BatchNorm internals, ComplexTensor, EmbeddingBag,
# GeluInplace, Sparse softmax / SpGEMM, Nested, Fused optimizer/graph, etc.).
add_executable(test_final_untested_multidtype test_final_untested_multidtype.cpp)
target_link_libraries(test_final_untested_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_final_untested_multidtype LABELS "ops" DISCOVERY_TIMEOUT 120 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Previously untested OpIds: StridedFill, EmbeddingWithBoundsCheck, HasInfNan, Cast
add_executable(test_dispatch_ops_multidtype test_dispatch_ops_multidtype.cpp)
target_link_libraries(test_dispatch_ops_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_dispatch_ops_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# ScatterReduce (sum, prod, mean, amax, amin)
add_executable(test_scatter_reduce test_scatter_reduce.cpp)
target_link_libraries(test_scatter_reduce PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_scatter_reduce LABELS "ops" DISCOVERY_TIMEOUT 60)

add_executable(test_scatter_oob_contract test_scatter_oob_contract.cpp)
target_link_libraries(test_scatter_oob_contract PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_scatter_oob_contract LABELS "ops" DISCOVERY_TIMEOUT 60)

add_executable(test_argmax_tie_parity test_argmax_tie_parity.cpp)
target_link_libraries(test_argmax_tie_parity PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_argmax_tie_parity LABELS "ops" DISCOVERY_TIMEOUT 60)

# Logcumsumexp, bincount, index_reduce
add_executable(test_logcumsumexp_bincount_index_reduce test_logcumsumexp_bincount_index_reduce.cpp)
target_link_libraries(test_logcumsumexp_bincount_index_reduce PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_logcumsumexp_bincount_index_reduce LABELS "ops" DISCOVERY_TIMEOUT 60)

# Histogramdd (multi-dimensional histogram)
add_executable(test_histogramdd test_histogramdd.cpp)
target_link_libraries(test_histogramdd PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_histogramdd LABELS "ops" DISCOVERY_TIMEOUT 60)

# New transform ops (moveaxis, narrow_copy, column_stack, row_stack, broadcast_tensors)
add_executable(test_transform_new test_transform_new.cpp)
target_link_libraries(test_transform_new PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_transform_new LABELS "ops" DISCOVERY_TIMEOUT 60)

# ============================================================================
# Multi-backend multi-dtype conversions for ops tests
# ============================================================================

add_executable(test_histogramdd_multidtype test_histogramdd_multidtype.cpp)
target_link_libraries(test_histogramdd_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_histogramdd_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

add_executable(test_logcumsumexp_bincount_index_reduce_multidtype test_logcumsumexp_bincount_index_reduce_multidtype.cpp)
target_link_libraries(test_logcumsumexp_bincount_index_reduce_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_logcumsumexp_bincount_index_reduce_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

add_executable(test_scatter_reduce_multidtype test_scatter_reduce_multidtype.cpp)
target_link_libraries(test_scatter_reduce_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_scatter_reduce_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

add_executable(test_transform_new_multidtype test_transform_new_multidtype.cpp)
target_link_libraries(test_transform_new_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_transform_new_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Griffin-Lim phase reconstruction (Fast Griffin-Lim algorithm). Release-prep S7.
add_executable(test_griffin_lim test_griffin_lim.cpp)
target_link_libraries(test_griffin_lim PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_griffin_lim LABELS "ops" DISCOVERY_TIMEOUT 60 BACKENDS cpu)
