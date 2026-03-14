# Utility module tests

# Logging tests
add_executable(test_utils_logging test_logging.cpp)
target_link_libraries(test_utils_logging PRIVATE
    tenzor_core
    GTest::gtest
    GTest::gtest_main
)
set_target_properties(test_utils_logging PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)
gtest_discover_tests(test_utils_logging DISCOVERY_TIMEOUT 60)

# Config tests
add_executable(test_utils_config test_config.cpp)
target_link_libraries(test_utils_config PRIVATE
    tenzor_core
    GTest::gtest
    GTest::gtest_main
)
set_target_properties(test_utils_config PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)
gtest_discover_tests(test_utils_config DISCOVERY_TIMEOUT 60)

# Benchmark tests
add_executable(test_utils_benchmark test_benchmark.cpp)
target_link_libraries(test_utils_benchmark PRIVATE
    tenzor_core
    GTest::gtest
    GTest::gtest_main
)
set_target_properties(test_utils_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)
gtest_discover_tests(test_utils_benchmark DISCOVERY_TIMEOUT 60)

# TensorBoard tests
add_executable(test_utils_tensorboard test_tensorboard.cpp)
target_link_libraries(test_utils_tensorboard PRIVATE
    tenzor_core
    GTest::gtest
    GTest::gtest_main
)
set_target_properties(test_utils_tensorboard PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)
gtest_discover_tests(test_utils_tensorboard DISCOVERY_TIMEOUT 60)
