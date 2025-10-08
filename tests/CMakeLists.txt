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

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(tenzor_unit_tests)
gtest_discover_tests(tenzor_integration_tests)
