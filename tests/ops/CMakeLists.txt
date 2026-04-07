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
tenzor_discover_tests(test_extended_math_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Special math functions (Gamma, Bessel, Zeta, etc.)
add_executable(test_special_math_multidtype test_special_math_multidtype.cpp)
target_link_libraries(test_special_math_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_special_math_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# 1D pooling operations
add_executable(test_pooling1d_multidtype test_pooling1d_multidtype.cpp)
target_link_libraries(test_pooling1d_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_pooling1d_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# 3D pooling operations
add_executable(test_pooling3d_multidtype test_pooling3d_multidtype.cpp)
target_link_libraries(test_pooling3d_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_pooling3d_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Signal processing (STFT, ISTFT, CDist)
add_executable(test_signal_processing_multidtype test_signal_processing_multidtype.cpp)
target_link_libraries(test_signal_processing_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_signal_processing_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Complex number operations (Conj, Real, Imag, Angle, Polar)
add_executable(test_complex_ops_multidtype test_complex_ops_multidtype.cpp)
target_link_libraries(test_complex_ops_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_complex_ops_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Search and sampling (SearchSorted, GumbelSoftmax)
add_executable(test_search_sampling_multidtype test_search_sampling_multidtype.cpp)
target_link_libraries(test_search_sampling_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_search_sampling_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Convolution gaps (Conv1d, ConvTranspose1d/2d/3d)
add_executable(test_conv_gaps_multidtype test_conv_gaps_multidtype.cpp)
target_link_libraries(test_conv_gaps_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_conv_gaps_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Window functions (Hann, Hamming, Blackman, Bartlett, Kaiser)
add_executable(test_windows_multidtype test_windows_multidtype.cpp)
target_link_libraries(test_windows_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_windows_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Edge cases multi-backend (NaN, Inf, empty tensors, size-1 tensors)
add_executable(test_edge_cases_ops_multidtype test_edge_cases_multidtype.cpp)
target_link_libraries(test_edge_cases_ops_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_edge_cases_ops_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Numerical gradients multi-backend
add_executable(test_numerical_gradients_multidtype test_numerical_gradients_multidtype.cpp)
target_link_libraries(test_numerical_gradients_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_numerical_gradients_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# LU decomposition multi-backend multi-dtype
add_executable(test_linalg_lu_multidtype test_linalg_lu_multidtype.cpp)
target_link_libraries(test_linalg_lu_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_linalg_lu_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# FP8 operations multi-backend
add_executable(test_fp8_ops_multidtype test_fp8_ops_multidtype.cpp)
target_link_libraries(test_fp8_ops_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_fp8_ops_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Conv3d multi-backend
add_executable(test_conv3d_multidtype test_conv3d_multidtype.cpp)
target_link_libraries(test_conv3d_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_conv3d_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Activation backward multi-backend
add_executable(test_activation_backward_multidtype test_activation_backward_multidtype.cpp)
target_link_libraries(test_activation_backward_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_activation_backward_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Fused ops dispatch multi-backend
add_executable(test_fused_ops_dispatch_multidtype test_fused_ops_dispatch_multidtype.cpp)
target_link_libraries(test_fused_ops_dispatch_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_fused_ops_dispatch_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# EmbeddingBag multi-backend
add_executable(test_embedding_bag_multidtype test_embedding_bag_multidtype.cpp)
target_link_libraries(test_embedding_bag_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_embedding_bag_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# BatchNorm dispatch multi-backend
add_executable(test_batchnorm_dispatch_multidtype test_batchnorm_dispatch_multidtype.cpp)
target_link_libraries(test_batchnorm_dispatch_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_batchnorm_dispatch_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# RNN cells dispatch multi-backend
add_executable(test_rnn_cells_dispatch_multidtype test_rnn_cells_dispatch_multidtype.cpp)
target_link_libraries(test_rnn_cells_dispatch_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_rnn_cells_dispatch_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Previously untested ops: inplace activations, LogSumExp, GridSample, AffineGrid
add_executable(test_untested_ops_multidtype test_untested_ops_multidtype.cpp)
target_link_libraries(test_untested_ops_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_untested_ops_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
