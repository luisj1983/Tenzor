# =============================================================================
# tests/distributed/CMakeLists.txt — Distributed training tests
# =============================================================================

add_executable(test_rpc_agent test_rpc_agent.cpp)
target_link_libraries(test_rpc_agent PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_rpc_agent LABELS "distributed" DISCOVERY_TIMEOUT 60)

add_executable(test_rref test_rref.cpp)
target_link_libraries(test_rref PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_rref LABELS "distributed" DISCOVERY_TIMEOUT 60)

add_executable(test_rendezvous test_rendezvous.cpp)
target_link_libraries(test_rendezvous PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_rendezvous LABELS "distributed" DISCOVERY_TIMEOUT 60)

add_executable(test_health_monitor test_health_monitor.cpp)
target_link_libraries(test_health_monitor PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_health_monitor LABELS "distributed" DISCOVERY_TIMEOUT 60)

add_executable(test_elastic_trainer test_elastic_trainer.cpp)
target_link_libraries(test_elastic_trainer PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_elastic_trainer LABELS "distributed" DISCOVERY_TIMEOUT 60)
