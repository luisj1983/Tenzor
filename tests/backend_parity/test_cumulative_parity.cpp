/**
 * @file test_cumulative_parity.cpp
 * @brief Cumulative and sorting operation parity tests across backends
 *
 * Tests cumsum, cumprod, topk, sort, cummax, and cummin operations to ensure
 * all backends (CPU, CUDA, ROCm, Vulkan, OneAPI) produce identical results.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <tenzor/tenzor.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;


class CumulativeParity : public BackendTest {};
// Wrappers to disambiguate Tensor-level ops from autograd Variable overloads
namespace {
Tensor tensor_cumsum(const Tensor& input, int64_t dim) {
    return tenzor::cumsum(Variable(input, false), dim).tensor();
}
Tensor tensor_cumprod(const Tensor& input, int64_t dim) {
    return tenzor::cumprod(Variable(input, false), dim).tensor();
}
} // namespace

// ============================================================================
// Cumulative Sum
// ============================================================================

TEST_P(CumulativeParity, CumSum_Dim0) {

    auto a = generate_test_tensor({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return tensor_cumsum(inputs[0], 0);
    }, {a}, device, 1e-4f, 1e-5f, "CumSum_Dim0");
}

TEST_P(CumulativeParity, CumSum_Dim1) {

    auto a = generate_test_tensor({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return tensor_cumsum(inputs[0], 1);
    }, {a}, device, 1e-4f, 1e-5f, "CumSum_Dim1");
}

// ============================================================================
// Cumulative Product
// ============================================================================

TEST_P(CumulativeParity, CumProd_Dim0) {

    // Use uniform [0.5, 1.5] to avoid underflow in cumulative product
    auto a = generate_uniform_tensor({16, 16}, 0.5f, 1.5f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return tensor_cumprod(inputs[0], 0);
    }, {a}, device, 1e-3f, 1e-4f, "CumProd_Dim0");
}

TEST_P(CumulativeParity, CumProd_Dim1) {

    auto a = generate_uniform_tensor({16, 16}, 0.5f, 1.5f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return tensor_cumprod(inputs[0], 1);
    }, {a}, device, 1e-3f, 1e-4f, "CumProd_Dim1");
}

// ============================================================================
// TopK
// ============================================================================

TEST_P(CumulativeParity, TopK_k5) {

    auto a = generate_test_tensor({32, 32}, DType::Float32, Device::cpu());

    // Compare values only — indices may differ for tied elements
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        auto [values, indices] = topk(Variable(inputs[0], false), 5, 1, true, true);
        return values.tensor();
    }, {a}, device, 1e-5f, 1e-8f, "TopK_k5");
}

TEST_P(CumulativeParity, TopK_k1) {

    auto a = generate_test_tensor({32, 32}, DType::Float32, Device::cpu());

    // Compare values only — indices may differ for tied elements
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        auto [values, indices] = topk(Variable(inputs[0], false), 1, 1, true, true);
        return values.tensor();
    }, {a}, device, 1e-5f, 1e-8f, "TopK_k1");
}

// ============================================================================
// Sort
// ============================================================================

TEST_P(CumulativeParity, Sort_Ascending) {

    auto a = generate_test_tensor({32, 32}, DType::Float32, Device::cpu());

    // Compare values only — indices may differ for tied elements
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        auto [values, indices] = sort(Variable(inputs[0], false), 1, false);
        return values.tensor();
    }, {a}, device, 0.0f, 0.0f, "Sort_Ascending");
}

TEST_P(CumulativeParity, Sort_Descending) {

    auto a = generate_test_tensor({32, 32}, DType::Float32, Device::cpu());

    // Compare values only — indices may differ for tied elements
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        auto [values, indices] = sort(Variable(inputs[0], false), 1, true);
        return values.tensor();
    }, {a}, device, 0.0f, 0.0f, "Sort_Descending");
}

// ============================================================================
// Cumulative Max / Min
// ============================================================================

TEST_P(CumulativeParity, CumMax) {

    auto a = generate_test_tensor({16, 16}, DType::Float32, Device::cpu());

    // Compare values only — indices may differ for tied elements
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        auto [vals, idxs] = cummax(inputs[0], 1);
        return vals;
    }, {a}, device, 0.0f, 0.0f, "CumMax");
}

TEST_P(CumulativeParity, CumMin) {

    auto a = generate_test_tensor({16, 16}, DType::Float32, Device::cpu());

    // Compare values only — indices may differ for tied elements
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        auto [vals, idxs] = cummin(inputs[0], 1);
        return vals;
    }, {a}, device, 0.0f, 0.0f, "CumMin");
}

INSTANTIATE_BACKEND_TESTS(CumulativeParity);




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
