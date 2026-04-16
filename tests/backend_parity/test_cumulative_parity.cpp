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
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

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

TEST(CumulativeParity, CumSum_Dim0) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_test_tensor({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tensor_cumsum(inputs[0], 0);
    }, {a}, 1e-4f, 1e-5f, "CumSum_Dim0");
}

TEST(CumulativeParity, CumSum_Dim1) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_test_tensor({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tensor_cumsum(inputs[0], 1);
    }, {a}, 1e-4f, 1e-5f, "CumSum_Dim1");
}

// ============================================================================
// Cumulative Product
// ============================================================================

TEST(CumulativeParity, CumProd_Dim0) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // Use uniform [0.5, 1.5] to avoid underflow in cumulative product
    auto a = generate_uniform_tensor({16, 16}, 0.5f, 1.5f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tensor_cumprod(inputs[0], 0);
    }, {a}, 1e-3f, 1e-4f, "CumProd_Dim0");
}

TEST(CumulativeParity, CumProd_Dim1) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({16, 16}, 0.5f, 1.5f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tensor_cumprod(inputs[0], 1);
    }, {a}, 1e-3f, 1e-4f, "CumProd_Dim1");
}

// ============================================================================
// TopK
// ============================================================================

TEST(CumulativeParity, TopK_k5) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_test_tensor({32, 32}, DType::Float32, Device::cpu());

    // Compare values only — indices may differ for tied elements
    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto [values, indices] = topk(Variable(inputs[0], false), 5, 1, true, true);
        return values.tensor();
    }, {a}, 1e-5f, 1e-8f, "TopK_k5");
}

TEST(CumulativeParity, TopK_k1) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_test_tensor({32, 32}, DType::Float32, Device::cpu());

    // Compare values only — indices may differ for tied elements
    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto [values, indices] = topk(Variable(inputs[0], false), 1, 1, true, true);
        return values.tensor();
    }, {a}, 1e-5f, 1e-8f, "TopK_k1");
}

// ============================================================================
// Sort
// ============================================================================

TEST(CumulativeParity, Sort_Ascending) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_test_tensor({32, 32}, DType::Float32, Device::cpu());

    // Compare values only — indices may differ for tied elements
    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto [values, indices] = sort(Variable(inputs[0], false), 1, false);
        return values.tensor();
    }, {a}, 0.0f, 0.0f, "Sort_Ascending");
}

TEST(CumulativeParity, Sort_Descending) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_test_tensor({32, 32}, DType::Float32, Device::cpu());

    // Compare values only — indices may differ for tied elements
    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto [values, indices] = sort(Variable(inputs[0], false), 1, true);
        return values.tensor();
    }, {a}, 0.0f, 0.0f, "Sort_Descending");
}

// ============================================================================
// Cumulative Max / Min
// ============================================================================

TEST(CumulativeParity, CumMax) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_test_tensor({16, 16}, DType::Float32, Device::cpu());

    // Compare values only — indices may differ for tied elements
    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto [vals, idxs] = cummax(inputs[0], 1);
        return vals;
    }, {a}, 0.0f, 0.0f, "CumMax");
}

TEST(CumulativeParity, CumMin) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_test_tensor({16, 16}, DType::Float32, Device::cpu());

    // Compare values only — indices may differ for tied elements
    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto [vals, idxs] = cummin(inputs[0], 1);
        return vals;
    }, {a}, 0.0f, 0.0f, "CumMin");
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
