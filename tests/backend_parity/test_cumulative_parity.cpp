/**
 * @file test_cumulative_parity.cpp
 * @brief Cumulative and sorting operation parity tests across backends
 *
 * Tests cumsum, cumprod, topk, sort, cummax, and cummin operations to ensure
 * all backends (CPU, CUDA, ROCm, Vulkan, OneAPI) produce identical results.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
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

// Phase 6-followup #27: gradient parity for cumulative reductions.
// Backward of cumsum is reverse-cumsum; easy to mis-implement per backend.
TEST_P(CumulativeParity, CumSum_Dim0_GradientParity) {
    auto a = randn({4, 6}, DType::Float32, Device::cpu());
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable {
            return cumsum(in[0], 0);
        },
        {a}, {}, 1e-5f, 1e-7f, 1e-4f, 1e-5f, {}, "CumSum_Dim0_Grad");
}

// Regression guard for #40: ensure cumsum(dim=1) of ones produces
// [1,2,3,4,5,6] on every backend (was broken on Vulkan/ROCm because their
// Flip kernel registrations read AttrKey::Dim instead of AttrKey::Dims and
// the cumsum backward path uses flip+cumsum+flip).
TEST_P(CumulativeParity, CumSumDim1_OnesRegression) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();
    auto a_cpu = ones({2, 6}, DType::Float32, Device::cpu());
    auto cs_cpu = tensor_cumsum(a_cpu, 1);
    // Expect [[1,2,3,4,5,6], [1,2,3,4,5,6]]
    for (size_t i = 1; i < backends.size(); ++i) {
        auto a_dev = a_cpu.to(backends[i]);
        auto cs_dev = tensor_cumsum(a_dev, 1);
        backends[i].synchronize();
        auto cs_dev_cpu = cs_dev.to(Device::cpu()).contiguous();
        auto cs_cpu_c = cs_cpu.contiguous();
        double max_abs = 0.0;
        for (int64_t k = 0; k < cs_cpu_c.numel(); ++k) {
            double diff = std::abs(cs_cpu_c.data<float>()[k]
                                   - cs_dev_cpu.data<float>()[k]);
            if (diff > max_abs) max_abs = diff;
        }
        std::ostringstream debug;
        debug << " dev[0..5]=";
        for (int k = 0; k < 6; ++k) {
            debug << cs_dev_cpu.data<float>()[k] << ",";
        }
        EXPECT_LT(max_abs, 1e-5) << "cumsum(dim=1, ones) on " << backend_name(backends[i])
            << " differs from CPU max_abs=" << max_abs
            << debug.str();
    }
}

// Regression guard for #40: ensure flip(dim=1) reverses correctly on every
// backend. Was broken on Vulkan/ROCm because their OpId::Flip dispatched
// via AttrKey::Dim (singular int, default 0) instead of AttrKey::Dims
// (plural string).
TEST_P(CumulativeParity, FlipDim1_Regression) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();
    auto a_cpu = randn({4, 6}, DType::Float32, Device::cpu());
    auto flipped_cpu = flip(a_cpu, std::vector<int64_t>{1});
    for (size_t i = 1; i < backends.size(); ++i) {
        auto a_dev = a_cpu.to(backends[i]);
        auto flipped_dev = flip(a_dev, std::vector<int64_t>{1});
        backends[i].synchronize();
        auto flipped_dev_cpu = flipped_dev.to(Device::cpu()).contiguous();
        auto flipped_cpu_c = flipped_cpu.contiguous();
        double max_abs = 0.0;
        for (int64_t k = 0; k < flipped_cpu_c.numel(); ++k) {
            double diff = std::abs(flipped_cpu_c.data<float>()[k]
                                   - flipped_dev_cpu.data<float>()[k]);
            if (diff > max_abs) max_abs = diff;
        }
        EXPECT_LT(max_abs, 1e-5) << "flip(dim=1) on " << backend_name(backends[i])
            << " differs from CPU max_abs=" << max_abs;
    }
}

// CumSum_Dim1 backward gradient differs by ~5.0 across all backends — the
// backward (reverse-cumsum) likely doesn't honor dim=1 properly. Filed as
// followup #40 — root cause traced to GPU flip kernel returning input.
TEST_P(CumulativeParity, CumSum_Dim1_GradientParity) {
    auto a = randn({4, 6}, DType::Float32, Device::cpu());
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable {
            return cumsum(in[0], 1);
        },
        {a}, {}, 1e-5f, 1e-7f, 1e-4f, 1e-5f, {}, "CumSum_Dim1_Grad");
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
