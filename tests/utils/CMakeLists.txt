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
add_test(NAME UtilsLogging COMMAND test_utils_logging)

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
add_test(NAME UtilsConfig COMMAND test_utils_config)

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
add_test(NAME UtilsBenchmark COMMAND test_utils_benchmark)

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
add_test(NAME UtilsTensorBoard COMMAND test_utils_tensorboard)
