# =============================================================================
# tests/autograd/CMakeLists.txt — Autograd-specific tests
# =============================================================================

# Autograd edge cases
add_executable(test_autograd_edge_cases test_autograd_edge_cases.cpp)
target_link_libraries(test_autograd_edge_cases PRIVATE tenzor_core GTest::gtest_main)

# Higher order gradients
add_executable(test_higher_order_gradients test_higher_order_gradients.cpp)
target_link_libraries(test_higher_order_gradients PRIVATE tenzor_core GTest::gtest_main)

# Gradient checking tests
add_executable(test_gradcheck test_gradcheck.cpp)
target_link_libraries(test_gradcheck PRIVATE tenzor_core GTest::gtest_main)

# Extended gradcheck tests (multi-backend)
add_executable(test_gradcheck_extended test_gradcheck_extended.cpp)
target_link_libraries(test_gradcheck_extended PRIVATE tenzor_core GTest::gtest_main)

# Additional autograd tests for increased coverage
add_executable(test_autograd_additional test_autograd_additional.cpp)
target_link_libraries(test_autograd_additional PRIVATE tenzor_core GTest::gtest_main)

# Autograd Hooks Tests
add_executable(test_hooks test_hooks.cpp)
target_link_libraries(test_hooks PRIVATE tenzor_core GTest::gtest_main)

# In-Place + Autograd Version Counter Tests
add_executable(test_inplace_autograd test_inplace_autograd.cpp)
target_link_libraries(test_inplace_autograd PRIVATE tenzor_core GTest::gtest_main)

# ============================================================================
# Test Registration
# ============================================================================
tenzor_discover_tests(test_autograd_edge_cases DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_higher_order_gradients DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_gradcheck DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_gradcheck_extended DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_autograd_additional DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_hooks DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_inplace_autograd DISCOVERY_TIMEOUT 60)
