# =============================================================================
# tests/serving/CMakeLists.txt — Inference serving tests
# =============================================================================

add_executable(test_model_repository test_model_repository.cpp)
target_link_libraries(test_model_repository PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_model_repository LABELS "serving" DISCOVERY_TIMEOUT 60)
