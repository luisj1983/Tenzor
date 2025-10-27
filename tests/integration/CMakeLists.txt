# Integration Tests CMakeLists.txt
# Comprehensive end-to-end tests for complete workflows

# Training loops integration tests
add_executable(test_training_loops
    test_training_loops.cpp
)

target_link_libraries(test_training_loops PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_training_loops DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 600)

# Model persistence integration tests
add_executable(test_model_persistence
    test_model_persistence.cpp
)

target_link_libraries(test_model_persistence PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_model_persistence DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 600)

# Cross-backend integration tests
add_executable(test_cross_backend
    test_cross_backend.cpp
)

target_link_libraries(test_cross_backend PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_cross_backend DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 600)

# Data pipeline integration tests
add_executable(test_data_pipeline
    test_data_pipeline.cpp
)

target_link_libraries(test_data_pipeline PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_data_pipeline DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 600)

# Model zoo integration tests
add_executable(test_model_zoo
    test_model_zoo.cpp
)

target_link_libraries(test_model_zoo PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_model_zoo DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 600)

# Optimization integration tests
add_executable(test_optimization
    test_optimization.cpp
)

target_link_libraries(test_optimization PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_optimization DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 600)

# DataParallel multi-GPU integration tests
add_executable(test_data_parallel_integration
    test_data_parallel.cpp
)

target_link_libraries(test_data_parallel_integration PRIVATE
    tenzor_core
    GTest::gtest_main
)

gtest_discover_tests(test_data_parallel_integration DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 600)

# Distributed training tests
add_executable(test_distributed
    test_distributed.cpp
)

target_link_libraries(test_distributed PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Distributed tests require special setup (multi-process environment)
# Tests will automatically skip if RANK/WORLD_SIZE environment variables are not set
gtest_discover_tests(test_distributed DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 600)

# Note: Existing integration tests
# - test_nn.cpp
# - test_training.cpp
# - test_cuda_training.cpp (if CUDA enabled)
