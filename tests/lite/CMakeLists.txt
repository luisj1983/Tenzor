# =============================================================================
# tests/lite/CMakeLists.txt — Lite runtime tests
# =============================================================================

add_executable(test_lite_ops test_lite_ops.cpp)
target_link_libraries(test_lite_ops PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_lite_ops LABELS "lite" DISCOVERY_TIMEOUT 60)

# LiteGraph and model format tests — re-enabled after lite_graph.cpp and
# model_format.cpp implementations were added to src/lite/.
add_executable(test_lite_graph test_lite_graph.cpp)
target_link_libraries(test_lite_graph PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_lite_graph LABELS "lite" DISCOVERY_TIMEOUT 60)

add_executable(test_model_format test_model_format.cpp)
target_link_libraries(test_model_format PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_model_format LABELS "lite" DISCOVERY_TIMEOUT 60)

add_executable(test_lite_runtime test_lite_runtime.cpp)
target_link_libraries(test_lite_runtime PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_lite_runtime LABELS "lite" DISCOVERY_TIMEOUT 60)

add_executable(test_exporter test_exporter.cpp)
target_link_libraries(test_exporter PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_exporter LABELS "lite" DISCOVERY_TIMEOUT 60)
