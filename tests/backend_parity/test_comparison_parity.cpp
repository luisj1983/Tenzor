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
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;


class ComparisonParity : public BackendTest {};
// ============================================================================
// Comparison Operations Parity Tests
// ============================================================================

TEST_P(ComparisonParity, Eq) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0] == inputs[1];
    }, {a, b}, device, 0.0f, 0.0f, "Eq");
}

TEST_P(ComparisonParity, Ne) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0] != inputs[1];
    }, {a, b}, device, 0.0f, 0.0f, "Ne");
}

TEST_P(ComparisonParity, Lt) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0] < inputs[1];
    }, {a, b}, device, 0.0f, 0.0f, "Lt");
}

TEST_P(ComparisonParity, Le) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0] <= inputs[1];
    }, {a, b}, device, 0.0f, 0.0f, "Le");
}

TEST_P(ComparisonParity, Gt) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0] > inputs[1];
    }, {a, b}, device, 0.0f, 0.0f, "Gt");
}

TEST_P(ComparisonParity, Ge) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0] >= inputs[1];
    }, {a, b}, device, 0.0f, 0.0f, "Ge");
}

// ============================================================================
// Broadcast Comparison Tests
// ============================================================================

TEST_P(ComparisonParity, Eq_Broadcast) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({1, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0] == inputs[1];
    }, {a, b}, device, 0.0f, 0.0f, "Eq_Broadcast");
}

TEST_P(ComparisonParity, Lt_Broadcast) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({1, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0] < inputs[1];
    }, {a, b}, device, 0.0f, 0.0f, "Lt_Broadcast");
}

INSTANTIATE_BACKEND_TESTS(ComparisonParity);




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
