# =============================================================================
# tests/examples/CMakeLists.txt — Example tests demonstrating fixture patterns
# =============================================================================

# Multi-parameter test example (8 tests demonstrating BackendDTypeParam usage)
add_executable(test_multi_param_example test_multi_param_example.cpp)
target_link_libraries(test_multi_param_example PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_multi_param_example LABELS "examples" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
