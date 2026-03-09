# =============================================================================
# tests/nn/CMakeLists.txt — Neural network layer and optimizer tests
# =============================================================================

# Dropout tests
add_executable(test_dropout layers/test_dropout.cpp)
target_link_libraries(test_dropout PRIVATE tenzor_core GTest::gtest_main)

# BatchNorm2d tests
add_executable(test_batchnorm2d layers/test_batchnorm2d.cpp)
target_link_libraries(test_batchnorm2d PRIVATE tenzor_core GTest::gtest_main)

# Conv2d tests
add_executable(test_conv2d layers/test_conv2d.cpp)
target_link_libraries(test_conv2d PRIVATE tenzor_core GTest::gtest_main)

# Conv2d multi-dtype tests
add_executable(test_conv2d_multidtype layers/test_conv2d_multidtype.cpp)
target_link_libraries(test_conv2d_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Pooling tests
add_executable(test_pooling layers/test_pooling.cpp)
target_link_libraries(test_pooling PRIVATE tenzor_core GTest::gtest_main)

# Scheduler tests
add_executable(test_schedulers optim/test_schedulers.cpp)
target_link_libraries(test_schedulers PRIVATE tenzor_core GTest::gtest_main)

# ZeRO Stage 1 Optimizer tests (Phase 4)
add_executable(test_zero_stage1 optim/test_zero_stage1.cpp)
target_link_libraries(test_zero_stage1 PRIVATE tenzor_core GTest::gtest_main)

# ZeRO Stage 1 Integration tests
add_executable(test_zero_stage1_integration optim/test_zero_stage1_integration.cpp)
target_link_libraries(test_zero_stage1_integration PRIVATE tenzor_core GTest::gtest_main)

# ZeRO Stage 1 Distributed tests
add_executable(test_zero_stage1_distributed optim/test_zero_stage1_distributed.cpp)
target_link_libraries(test_zero_stage1_distributed PRIVATE tenzor_core GTest::gtest_main)

# Gradient Utilities tests (ZeRO Stage 2)
add_executable(test_gradient_utils optim/test_gradient_utils.cpp)
target_link_libraries(test_gradient_utils PRIVATE tenzor_core GTest::gtest_main)

# ZeRO Stage 2 Optimizer tests (Phase 5 - Gradient Partitioning)
add_executable(test_zero_stage2 optim/test_zero_stage2.cpp)
target_link_libraries(test_zero_stage2 PRIVATE tenzor_core GTest::gtest_main)

# ZeRO Stage 2 Integration tests
add_executable(test_zero_stage2_integration optim/test_zero_stage2_integration.cpp)
target_link_libraries(test_zero_stage2_integration PRIVATE tenzor_core GTest::gtest_main)

# ZeRO Stage 3 Optimizer tests (Phase 6 - Parameter Partitioning)
add_executable(test_zero_stage3 optim/test_zero_stage3.cpp)
target_link_libraries(test_zero_stage3 PRIVATE tenzor_core GTest::gtest_main)

# ZeRO Stage 3 Integration tests (Phase 6 - Parameter Partitioning)
add_executable(test_zero_stage3_integration optim/test_zero_stage3_integration.cpp)
target_link_libraries(test_zero_stage3_integration PRIVATE tenzor_core GTest::gtest_main)

# ZeRO Profiling tests (Phase 7 - Performance Profiling)
add_executable(test_zero_profiling optim/test_zero_profiling.cpp)
target_link_libraries(test_zero_profiling PRIVATE tenzor_core GTest::gtest_main)

# Normalization tests
add_executable(test_normalization layers/test_normalization.cpp)
target_link_libraries(test_normalization PRIVATE tenzor_core GTest::gtest_main)

# Serialization tests
add_executable(test_serialization test_serialization.cpp)
target_link_libraries(test_serialization PRIVATE tenzor_core GTest::gtest_main)

# Offload tests (ZeRO Stage 2 Parameter Offloading)
add_executable(test_offload test_offload.cpp)
target_link_libraries(test_offload PRIVATE tenzor_core GTest::gtest_main)

# NN Layer Multi-DType Tests
add_executable(test_pooling_multidtype layers/test_pooling_multidtype.cpp)
target_link_libraries(test_pooling_multidtype PRIVATE tenzor_core GTest::gtest_main)

add_executable(test_dropout_multidtype layers/test_dropout_multidtype.cpp)
target_link_libraries(test_dropout_multidtype PRIVATE tenzor_core GTest::gtest_main)

add_executable(test_batchnorm2d_multidtype layers/test_batchnorm2d_multidtype.cpp)
target_link_libraries(test_batchnorm2d_multidtype PRIVATE tenzor_core GTest::gtest_main)

add_executable(test_normalization_multidtype layers/test_normalization_multidtype.cpp)
target_link_libraries(test_normalization_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Flatten layer multi-dtype tests
add_executable(test_flatten_multidtype layers/test_flatten_multidtype.cpp)
target_link_libraries(test_flatten_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Segmentation operations multi-dtype tests
add_executable(test_segmentation_multidtype layers/test_segmentation_multidtype.cpp)
target_link_libraries(test_segmentation_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Hierarchical Reasoning Model (HRM) Tests
add_executable(test_hrm layers/test_hrm.cpp)
target_link_libraries(test_hrm PRIVATE tenzor_core GTest::gtest_main)

# ============================================================================
# Test Registration
# ============================================================================
tenzor_discover_tests(test_dropout DISCOVERY_TIMEOUT 30)
tenzor_discover_tests(test_batchnorm2d DISCOVERY_TIMEOUT 30)
tenzor_discover_tests(test_conv2d DISCOVERY_TIMEOUT 30)
tenzor_discover_tests(test_conv2d_multidtype DISCOVERY_TIMEOUT 30)
tenzor_discover_tests(test_pooling DISCOVERY_TIMEOUT 30)
tenzor_discover_tests(test_schedulers DISCOVERY_TIMEOUT 30)
tenzor_discover_tests(test_zero_stage1 DISCOVERY_TIMEOUT 30)
tenzor_discover_tests(test_zero_stage1_integration DISCOVERY_TIMEOUT 30)
tenzor_discover_tests(test_zero_stage1_distributed DISCOVERY_TIMEOUT 30)
tenzor_discover_tests(test_gradient_utils DISCOVERY_TIMEOUT 30)
tenzor_discover_tests(test_zero_stage2 DISCOVERY_TIMEOUT 30)
tenzor_discover_tests(test_zero_stage2_integration DISCOVERY_TIMEOUT 30)
tenzor_discover_tests(test_zero_stage3 DISCOVERY_TIMEOUT 30)
tenzor_discover_tests(test_zero_stage3_integration DISCOVERY_TIMEOUT 30)
tenzor_discover_tests(test_zero_profiling DISCOVERY_TIMEOUT 30)
tenzor_discover_tests(test_normalization DISCOVERY_TIMEOUT 30)
tenzor_discover_tests(test_serialization DISCOVERY_TIMEOUT 30)
tenzor_discover_tests(test_offload DISCOVERY_TIMEOUT 30)
tenzor_discover_tests(test_pooling_multidtype DISCOVERY_TIMEOUT 30)
tenzor_discover_tests(test_dropout_multidtype DISCOVERY_TIMEOUT 30)
tenzor_discover_tests(test_batchnorm2d_multidtype DISCOVERY_TIMEOUT 30)
tenzor_discover_tests(test_normalization_multidtype DISCOVERY_TIMEOUT 30)
tenzor_discover_tests(test_flatten_multidtype DISCOVERY_TIMEOUT 30)
tenzor_discover_tests(test_segmentation_multidtype DISCOVERY_TIMEOUT 30)
tenzor_discover_tests(test_hrm DISCOVERY_TIMEOUT 60)
