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

tenzor_discover_tests(test_advanced_ops_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_shape_ops_multidtype LABELS "ops" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_fp8_ops LABELS "ops" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_linalg_lu LABELS "ops" DISCOVERY_TIMEOUT 60)
