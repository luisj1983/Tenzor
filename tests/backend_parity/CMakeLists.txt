# Backend Parity Tests CMakeLists.txt
# Comprehensive test suite for backend parity across CPU, CUDA, OneAPI, Vulkan

cmake_minimum_required(VERSION 3.18)

# Find required packages
find_package(GTest REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GTEST_INCLUDE_DIRS}
)

# Helper function to create a parity test executable
function(add_parity_test TEST_NAME SOURCE_FILE)
    add_executable(${TEST_NAME} ${SOURCE_FILE})

    target_link_libraries(${TEST_NAME}
        tenzor
        GTest::gtest
        GTest::gtest_main
        ${CMAKE_THREAD_LIBS_INIT}
    )

    # Add test to CTest
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

    # Set test properties
    set_tests_properties(${TEST_NAME} PROPERTIES
        TIMEOUT 600  # 10 minute timeout for stress tests
        LABELS "backend_parity"
    )

    # Install test
    install(TARGETS ${TEST_NAME}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/tests/backend_parity
    )
endfunction()

# ==============================================================================
# Backend Parity Test Executables
# ==============================================================================

# Operation parity tests (40+ math and reduction operations)
add_parity_test(test_operation_parity test_operation_parity.cpp)

# Neural network operation parity tests (30+ NN operations)
add_parity_test(test_nn_parity test_nn_parity.cpp)

# Gradient parity tests with numerical gradient checking
add_parity_test(test_gradient_parity test_gradient_parity.cpp)

# Data type parity tests (Float32, Float64, Int32, Int64)
add_parity_test(test_dtype_parity test_dtype_parity.cpp)

# Stress tests for large tensors and deep computation graphs
add_parity_test(test_backend_stress test_backend_stress.cpp)

# Numerical stability tests for edge cases
add_parity_test(test_numerical_stability test_numerical_stability.cpp)

# Performance regression baseline tests
add_parity_test(test_performance_regression test_performance_regression.cpp)

# ==============================================================================
# Test Groups
# ==============================================================================

# Quick parity tests (basic operations only)
add_custom_target(test_parity_quick
    COMMAND ${CMAKE_CTEST_COMMAND} -L "backend_parity" -R "test_operation_parity|test_dtype_parity" --output-on-failure
    DEPENDS test_operation_parity test_dtype_parity
    COMMENT "Running quick backend parity tests..."
)

# Full parity test suite
add_custom_target(test_parity_full
    COMMAND ${CMAKE_CTEST_COMMAND} -L "backend_parity" --output-on-failure
    DEPENDS
        test_operation_parity
        test_nn_parity
        test_gradient_parity
        test_dtype_parity
        test_backend_stress
        test_numerical_stability
        test_performance_regression
    COMMENT "Running full backend parity test suite..."
)

# Stress tests only
add_custom_target(test_parity_stress
    COMMAND ${CMAKE_CTEST_COMMAND} -L "backend_parity" -R "test_backend_stress|test_numerical_stability" --output-on-failure
    DEPENDS test_backend_stress test_numerical_stability
    COMMENT "Running backend stress tests..."
)

# Performance tests only
add_custom_target(test_parity_performance
    COMMAND ${CMAKE_CTEST_COMMAND} -L "backend_parity" -R "test_performance_regression" --output-on-failure
    DEPENDS test_performance_regression
    COMMENT "Running backend performance regression tests..."
)

# Gradient tests only
add_custom_target(test_parity_gradients
    COMMAND ${CMAKE_CTEST_COMMAND} -L "backend_parity" -R "test_gradient_parity" --output-on-failure
    DEPENDS test_gradient_parity
    COMMENT "Running gradient parity tests..."
)

# ==============================================================================
# Test Documentation
# ==============================================================================

# Print test information
message(STATUS "Backend Parity Tests Configuration:")
message(STATUS "  - Operation parity tests: test_operation_parity")
message(STATUS "  - NN operation parity tests: test_nn_parity")
message(STATUS "  - Gradient parity tests: test_gradient_parity")
message(STATUS "  - DType parity tests: test_dtype_parity")
message(STATUS "  - Stress tests: test_backend_stress")
message(STATUS "  - Numerical stability tests: test_numerical_stability")
message(STATUS "  - Performance regression tests: test_performance_regression")
message(STATUS "")
message(STATUS "Run test groups with:")
message(STATUS "  - make test_parity_quick      # Quick tests only")
message(STATUS "  - make test_parity_full       # Full test suite")
message(STATUS "  - make test_parity_stress     # Stress tests only")
message(STATUS "  - make test_parity_performance # Performance tests only")
message(STATUS "  - make test_parity_gradients  # Gradient tests only")

# ==============================================================================
# Install utilities header
# ==============================================================================

install(FILES parity_test_utils.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tenzor/tests
)
