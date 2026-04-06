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

# ChannelShuffle tests
add_executable(test_channel_shuffle layers/test_channel_shuffle.cpp)
target_link_libraries(test_channel_shuffle PRIVATE tenzor_core GTest::gtest_main)

# Quantized inference tests
add_executable(test_quantized_inference test_quantized_inference.cpp)
target_link_libraries(test_quantized_inference PRIVATE tenzor_core GTest::gtest_main)

# Segmentation operations multi-dtype tests
add_executable(test_segmentation_multidtype layers/test_segmentation_multidtype.cpp)
target_link_libraries(test_segmentation_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Hierarchical Reasoning Model (HRM) Tests
add_executable(test_hrm layers/test_hrm.cpp)
target_link_libraries(test_hrm PRIVATE tenzor_core GTest::gtest_main)

# ============================================================================
# Test Registration
# ============================================================================
tenzor_discover_tests(test_dropout LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_batchnorm2d LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_conv2d LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_conv2d_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_pooling LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_schedulers LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_zero_stage1 LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_zero_stage1_integration LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_zero_stage1_distributed LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_gradient_utils LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_zero_stage2 LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_zero_stage2_integration LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_zero_stage3 LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_zero_stage3_integration LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_zero_profiling LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_normalization LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_serialization LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_offload LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_pooling_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_dropout_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_batchnorm2d_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_normalization_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_flatten_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_channel_shuffle LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_quantized_inference LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_segmentation_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60)
# PyTorch loader tests
add_executable(test_pytorch_loader test_pytorch_loader.cpp)
target_link_libraries(test_pytorch_loader PRIVATE tenzor_core GTest::gtest_main)

# FSDP tests
add_executable(test_fsdp test_fsdp.cpp)
target_link_libraries(test_fsdp PRIVATE tenzor_core GTest::gtest_main)

# Tensor Parallelism tests
add_executable(test_tensor_parallel test_tensor_parallel.cpp)
target_link_libraries(test_tensor_parallel PRIVATE tenzor_core GTest::gtest_main)

# Functional API tests
add_executable(test_functional test_functional.cpp)
target_link_libraries(test_functional PRIVATE tenzor_core GTest::gtest_main)

# SafeTensors format tests
add_executable(test_safetensors test_safetensors.cpp)
target_link_libraries(test_safetensors PRIVATE tenzor_core GTest::gtest_main)

tenzor_discover_tests(test_hrm LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_pytorch_loader LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_fsdp LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_tensor_parallel LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_functional LABELS "nn" DISCOVERY_TIMEOUT 120)
tenzor_discover_tests(test_safetensors LABELS "nn" DISCOVERY_TIMEOUT 120)

# Parametrize module tests
add_executable(test_parametrize test_parametrize.cpp)
target_link_libraries(test_parametrize PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_parametrize LABELS "nn" DISCOVERY_TIMEOUT 60)

# ============================================================================
# New module tests (Phase 2)
# ============================================================================

# Distributions multi-backend tests
add_executable(test_distributions_multidtype test_distributions_multidtype.cpp)
target_link_libraries(test_distributions_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_distributions_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60)

# Weight initialization multi-backend tests
add_executable(test_init_multidtype test_init_multidtype.cpp)
target_link_libraries(test_init_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_init_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60)

# Gradient clipping multi-backend tests
add_executable(test_clip_grad_multidtype test_clip_grad_multidtype.cpp)
target_link_libraries(test_clip_grad_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_clip_grad_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60)

# Spectral normalization multi-backend tests
add_executable(test_spectral_norm_multidtype test_spectral_norm_multidtype.cpp)
target_link_libraries(test_spectral_norm_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_spectral_norm_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60)

# Weight normalization multi-backend tests
add_executable(test_weight_norm_multidtype test_weight_norm_multidtype.cpp)
target_link_libraries(test_weight_norm_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_weight_norm_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60)
