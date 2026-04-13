# =============================================================================
# tests/nested/CMakeLists.txt -- Nested tensor tests
# =============================================================================

# Core nested tensor tests (construction, ops, padding roundtrip)
add_executable(test_nested_tensor test_nested_tensor.cpp)
target_link_libraries(test_nested_tensor PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_nested_tensor LABELS "nested" DISCOVERY_TIMEOUT 60)

# Autograd tests for nested operations (softmax, attention, reduction gradients)
add_executable(test_nested_autograd test_nested_autograd.cpp)
target_link_libraries(test_nested_autograd PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_nested_autograd LABELS "nested" DISCOVERY_TIMEOUT 60)
