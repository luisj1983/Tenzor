/**
 * @file test_manipulation_parity.cpp
 * @brief Tensor manipulation operation parity tests across backends
 *
 * Tests triu, tril, diag, trace, flip, and roll operations. Most manipulation
 * ops are exact (rtol=0, atol=0) since they rearrange or mask data without
 * arithmetic. Trace involves summation so uses a small tolerance.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <tenzor/tenzor.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;


class ManipulationParity : public BackendTest {};
// ============================================================================
// Tensor Manipulation Operations Parity Tests (8 tests)
// ============================================================================

TEST_P(ManipulationParity, Triu) {

    auto input = generate_uniform_tensor({16, 16}, -1.0f, 1.0f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return triu(inputs[0]);
    }, {input}, device, 0, 0, "Triu");
}

TEST_P(ManipulationParity, Triu_Diagonal) {

    auto input = generate_uniform_tensor({16, 16}, -1.0f, 1.0f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return triu(inputs[0], 1);
    }, {input}, device, 0, 0, "Triu_Diagonal");
}

TEST_P(ManipulationParity, Tril) {

    auto input = generate_uniform_tensor({16, 16}, -1.0f, 1.0f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return tril(inputs[0]);
    }, {input}, device, 0, 0, "Tril");
}

TEST_P(ManipulationParity, Diag_Extract) {

    auto input = generate_uniform_tensor({16, 16}, -1.0f, 1.0f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return diag(inputs[0]);
    }, {input}, device, 0, 0, "Diag_Extract");
}

TEST_P(ManipulationParity, Diag_Construct) {

    auto input = generate_uniform_tensor({16}, -1.0f, 1.0f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return diag(inputs[0]);
    }, {input}, device, 0, 0, "Diag_Construct");
}

TEST_P(ManipulationParity, Trace) {

    auto input = generate_uniform_tensor({16, 16}, -1.0f, 1.0f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return trace(inputs[0]);
    }, {input}, device, 1e-5f, 1e-7f, "Trace");
}

TEST_P(ManipulationParity, Flip) {

    auto input = generate_uniform_tensor({8, 16}, -1.0f, 1.0f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return flip(inputs[0], std::vector<int64_t>{0, 1});
    }, {input}, device, 0, 0, "Flip");
}

TEST_P(ManipulationParity, Roll) {

    auto input = generate_uniform_tensor({32, 32}, -1.0f, 1.0f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return roll(inputs[0], static_cast<int64_t>(5), static_cast<int64_t>(0));
    }, {input}, device, 0, 0, "Roll");
}

INSTANTIATE_BACKEND_TESTS(ManipulationParity);




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
