# =============================================================================
# tests/serving/CMakeLists.txt — Inference serving tests
# =============================================================================

add_executable(test_model_repository test_model_repository.cpp)
target_link_libraries(test_model_repository PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_model_repository LABELS "serving" DISCOVERY_TIMEOUT 60)

add_executable(test_rate_limiter test_rate_limiter.cpp)
target_link_libraries(test_rate_limiter PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_rate_limiter LABELS "serving" DISCOVERY_TIMEOUT 60)

add_executable(test_traffic_router test_traffic_router.cpp)
target_link_libraries(test_traffic_router PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_traffic_router LABELS "serving" DISCOVERY_TIMEOUT 60)

add_executable(test_auth test_auth.cpp)
target_link_libraries(test_auth PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_auth LABELS "serving" DISCOVERY_TIMEOUT 60)

add_executable(test_dynamic_batcher test_dynamic_batcher.cpp)
target_link_libraries(test_dynamic_batcher PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_dynamic_batcher LABELS "serving" DISCOVERY_TIMEOUT 60)

add_executable(test_metrics test_metrics.cpp)
target_link_libraries(test_metrics PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_metrics LABELS "serving" DISCOVERY_TIMEOUT 60)
