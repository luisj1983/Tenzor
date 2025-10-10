cmake_minimum_required(VERSION 3.25)

# Find Google Test
find_package(GTest)

if(NOT GTest_FOUND)
    message(STATUS "Google Test not found, downloading...")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )
    FetchContent_MakeAvailable(googletest)
endif()

# Unit tests
add_executable(tenzor_unit_tests
    unit/test_tensor.cpp
    unit/test_device.cpp
    unit/test_ops.cpp
    unit/test_autograd.cpp
    unit/test_cpu_kernels.cpp
    unit/test_broadcasting.cpp
    unit/test_transforms.cpp
    unit/test_linear.cpp
    unit/test_losses.cpp
    unit/test_optimizers.cpp
)

target_link_libraries(tenzor_unit_tests PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Integration tests
add_executable(tenzor_integration_tests
    integration/test_nn.cpp
    integration/test_training.cpp
)

target_link_libraries(tenzor_integration_tests PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Dropout tests
add_executable(test_dropout
    nn/layers/test_dropout.cpp
)

target_link_libraries(test_dropout PRIVATE
    tenzor_core
    GTest::gtest_main
)

# BatchNorm2d tests
add_executable(test_batchnorm2d
    nn/layers/test_batchnorm2d.cpp
)

target_link_libraries(test_batchnorm2d PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Conv2d tests
add_executable(test_conv2d
    nn/layers/test_conv2d.cpp
)

target_link_libraries(test_conv2d PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Pooling tests
add_executable(test_pooling
    nn/layers/test_pooling.cpp
)

target_link_libraries(test_pooling PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Scheduler tests
add_executable(test_schedulers
    nn/optim/test_schedulers.cpp
)

target_link_libraries(test_schedulers PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Normalization tests
add_executable(test_normalization
    nn/layers/test_normalization.cpp
)

target_link_libraries(test_normalization PRIVATE
    tenzor_core
    GTest::gtest_main
)

# Serialization tests
add_executable(test_serialization
    nn/test_serialization.cpp
)

target_link_libraries(test_serialization PRIVATE
    tenzor_core
    GTest::gtest_main
)

# CUDA kernel tests
if(TENZOR_BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_executable(test_cuda_kernels
        backends/test_cuda_kernels.cpp
    )

    target_link_libraries(test_cuda_kernels PRIVATE
        tenzor_core
        GTest::gtest_main
        CUDA::cudart
    )

    set_target_properties(test_cuda_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    # CUDA training tests
    add_executable(test_cuda_training
        integration/test_cuda_training.cpp
    )

    target_link_libraries(test_cuda_training PRIVATE
        tenzor_core
        GTest::gtest_main
        CUDA::cudart
    )

    set_target_properties(test_cuda_training PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# Debug test for BatchNorm precision
add_executable(test_batchnorm_precision_debug
    /tmp/test_batchnorm_precision.cpp
)

target_link_libraries(test_batchnorm_precision_debug PRIVATE
    tenzor_core
)

# Debug test for variance computation
add_executable(test_variance_debug
    /tmp/test_variance_debug.cpp
)

target_link_libraries(test_variance_debug PRIVATE
    tenzor_core
)

# Debug test for multi-channel BatchNorm
add_executable(test_multichannel
    /tmp/test_multichannel.cpp
)

target_link_libraries(test_multichannel PRIVATE
    tenzor_core
)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(tenzor_unit_tests)
gtest_discover_tests(tenzor_integration_tests)
gtest_discover_tests(test_dropout)
gtest_discover_tests(test_batchnorm2d)
gtest_discover_tests(test_conv2d)
gtest_discover_tests(test_pooling)
gtest_discover_tests(test_schedulers)
gtest_discover_tests(test_normalization)
gtest_discover_tests(test_serialization)

if(TENZOR_BUILD_CUDA)
    gtest_discover_tests(test_cuda_kernels)
    gtest_discover_tests(test_cuda_training)
endif()
