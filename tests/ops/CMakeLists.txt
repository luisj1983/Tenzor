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

tenzor_discover_tests(test_advanced_ops_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_shape_ops_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60)
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
tenzor_discover_tests(test_extended_math_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60)

# Special math functions (Gamma, Bessel, Zeta, etc.)
add_executable(test_special_math_multidtype test_special_math_multidtype.cpp)
target_link_libraries(test_special_math_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_special_math_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60)

# 1D pooling operations
add_executable(test_pooling1d_multidtype test_pooling1d_multidtype.cpp)
target_link_libraries(test_pooling1d_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_pooling1d_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60)

# 3D pooling operations
add_executable(test_pooling3d_multidtype test_pooling3d_multidtype.cpp)
target_link_libraries(test_pooling3d_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_pooling3d_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60)

# Signal processing (STFT, ISTFT, CDist)
add_executable(test_signal_processing_multidtype test_signal_processing_multidtype.cpp)
target_link_libraries(test_signal_processing_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_signal_processing_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60)

# Complex number operations (Conj, Real, Imag, Angle, Polar)
add_executable(test_complex_ops_multidtype test_complex_ops_multidtype.cpp)
target_link_libraries(test_complex_ops_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_complex_ops_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60)

# Search and sampling (SearchSorted, GumbelSoftmax)
add_executable(test_search_sampling_multidtype test_search_sampling_multidtype.cpp)
target_link_libraries(test_search_sampling_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_search_sampling_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60)

# Convolution gaps (Conv1d, ConvTranspose1d/2d/3d)
add_executable(test_conv_gaps_multidtype test_conv_gaps_multidtype.cpp)
target_link_libraries(test_conv_gaps_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_conv_gaps_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60)

# Window functions (Hann, Hamming, Blackman, Bartlett, Kaiser)
add_executable(test_windows_multidtype test_windows_multidtype.cpp)
target_link_libraries(test_windows_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_windows_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60)
