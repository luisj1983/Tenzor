# =============================================================================
# tests/lite/CMakeLists.txt — Lite runtime tests
# =============================================================================

add_executable(test_lite_ops test_lite_ops.cpp)
target_link_libraries(test_lite_ops PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_lite_ops LABELS "lite" DISCOVERY_TIMEOUT 60)
