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
tenzor_discover_tests(test_advanced_ops_multidtype DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_shape_ops_multidtype DISCOVERY_TIMEOUT 60)
