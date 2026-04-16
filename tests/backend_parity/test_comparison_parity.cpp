/**
 * @file test_comparison_parity.cpp
 * @brief Comparison operation parity tests across backends
 *
 * Tests 6 comparison operators (eq, ne, lt, le, gt, ge) plus broadcasting
 * variants to ensure all backends produce identical Bool results.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <tenzor/tenzor.hpp>
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Comparison Operations Parity Tests
// ============================================================================

TEST(ComparisonParity, Eq) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] == inputs[1];
    }, {a, b}, 0.0f, 0.0f, "Eq");
}

TEST(ComparisonParity, Ne) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] != inputs[1];
    }, {a, b}, 0.0f, 0.0f, "Ne");
}

TEST(ComparisonParity, Lt) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] < inputs[1];
    }, {a, b}, 0.0f, 0.0f, "Lt");
}

TEST(ComparisonParity, Le) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] <= inputs[1];
    }, {a, b}, 0.0f, 0.0f, "Le");
}

TEST(ComparisonParity, Gt) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] > inputs[1];
    }, {a, b}, 0.0f, 0.0f, "Gt");
}

TEST(ComparisonParity, Ge) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] >= inputs[1];
    }, {a, b}, 0.0f, 0.0f, "Ge");
}

// ============================================================================
// Broadcast Comparison Tests
// ============================================================================

TEST(ComparisonParity, Eq_Broadcast) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({1, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] == inputs[1];
    }, {a, b}, 0.0f, 0.0f, "Eq_Broadcast");
}

TEST(ComparisonParity, Lt_Broadcast) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({1, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] < inputs[1];
    }, {a, b}, 0.0f, 0.0f, "Lt_Broadcast");
}



int main(int argc, char** argv) {
    try {
        tenzor::initialize();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
    }
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    try {
        tenzor::finalize();
    } catch (...) {}
    return result;
}
