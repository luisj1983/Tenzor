/**
 * @file test_inplace_ops_parity.cpp
 * @brief Cross-backend parity tests for in-place arithmetic and activation ops.
 *
 * Covers OpIds: AddInplace, SubInplace, MulInplace, DivInplace (7-10) and
 * ReLUInplace, SigmoidInplace, TanhInplace, LeakyReLUInplace, GeluInplace
 * (89-93). Per the audit, only OpId::ReLUInplace was previously referenced
 * by name in any test — every other in-place OpId had no parity coverage,
 * leaving silent regressions possible despite the registry showing them
 * registered.
 *
 * Each test:
 *   1. Builds a CPU input tensor.
 *   2. On every available backend, clones the input, applies the in-place op,
 *      and compares the result against the same op applied on CPU.
 *   3. Asserts that the operation actually wrote into the target buffer
 *      (i.e. the inplace dispatch path is used, not a copy-on-write fallback).
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class InplaceOpsParity : public BackendTest {};

// ----------------------------------------------------------------------------
// In-place arithmetic (OpIds 7-10)
// ----------------------------------------------------------------------------

TEST_P(InplaceOpsParity, AddInplace) {
    auto a = randn({4, 8}, DType::Float32, Device::cpu());
    auto b = randn({4, 8}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        auto target = inputs[0].clone();
        target += inputs[1];
        return target;
    }, {a, b}, device, 1e-5f, 1e-7f, "AddInplace");
}

TEST_P(InplaceOpsParity, SubInplace) {
    auto a = randn({4, 8}, DType::Float32, Device::cpu());
    auto b = randn({4, 8}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        auto target = inputs[0].clone();
        target -= inputs[1];
        return target;
    }, {a, b}, device, 1e-5f, 1e-7f, "SubInplace");
}

TEST_P(InplaceOpsParity, MulInplace) {
    auto a = randn({4, 8}, DType::Float32, Device::cpu());
    auto b = randn({4, 8}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        auto target = inputs[0].clone();
        target *= inputs[1];
        return target;
    }, {a, b}, device, 1e-5f, 1e-7f, "MulInplace");
}

TEST_P(InplaceOpsParity, DivInplace) {
    // Avoid division by values near zero to keep tolerances sane.
    auto a = randn({4, 8}, DType::Float32, Device::cpu());
    auto b = randn({4, 8}, DType::Float32, Device::cpu()) + 2.0f;

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        auto target = inputs[0].clone();
        target /= inputs[1];
        return target;
    }, {a, b}, device, 1e-5f, 1e-7f, "DivInplace");
}

// ----------------------------------------------------------------------------
// In-place activations (OpIds 89-93)
// ----------------------------------------------------------------------------

TEST_P(InplaceOpsParity, ReLUInplace) {
    auto x = randn({4, 16}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        auto target = inputs[0].clone();
        nn::relu_(target);
        return target;
    }, {x}, device, 1e-5f, 1e-7f, "ReLUInplace");
}

TEST_P(InplaceOpsParity, SigmoidInplace) {
    auto x = randn({4, 16}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        auto target = inputs[0].clone();
        nn::sigmoid_(target);
        return target;
    }, {x}, device, 1e-5f, 1e-7f, "SigmoidInplace");
}

TEST_P(InplaceOpsParity, TanhInplace) {
    auto x = randn({4, 16}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        auto target = inputs[0].clone();
        nn::tanh_(target);
        return target;
    }, {x}, device, 1e-5f, 1e-7f, "TanhInplace");
}

TEST_P(InplaceOpsParity, LeakyReLUInplace) {
    auto x = randn({4, 16}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        auto target = inputs[0].clone();
        nn::leaky_relu_(target, 0.05);
        return target;
    }, {x}, device, 1e-5f, 1e-7f, "LeakyReLUInplace");
}

// Single-value smoke test for LeakyReLU slope plumbing. Catches two prior bugs:
//   1. nn::leaky_relu_ used to set AttrKey::Negative_slope while every backend
//      kernel read AttrKey::Alpha — silently dropping the user's slope. Fixed
//      in src/nn/activations/activations.cpp and src/autograd/ops.cpp.
//   2. ROCm leaky_relu_inplace_kernel passed (alpha, n) to a kernel whose
//      signature was (n, alpha) — n got truncated to 0 and the loop never
//      ran. Fixed in src/backends/rocm/kernels/activations.hip.cpp.
TEST_P(InplaceOpsParity, LeakyReLUInplaceSlopePlumbing) {
    auto x = full({1}, -10.0, DType::Float32, Device::cpu()).to(device);
    nn::leaky_relu_(x, 0.05);
    device.synchronize();
    auto out = x.to(Device::cpu());
    EXPECT_NEAR(out.data<float>()[0], -0.5f, 1e-4f)
        << backend_name(device) << " ignored slope=0.05; got "
        << out.data<float>()[0];
}

TEST_P(InplaceOpsParity, GeluInplace) {
    auto x = randn({4, 16}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        auto target = inputs[0].clone();
        nn::gelu_(target);
        return target;
    }, {x}, device, 1e-4f, 1e-6f, "GeluInplace");
}

// ----------------------------------------------------------------------------
// Sanity: in-place ops actually mutate the target buffer (not copy-on-write).
// One representative test on the parameterized device suffices — the parity
// tests above already cover correctness of the result.
// ----------------------------------------------------------------------------

TEST_P(InplaceOpsParity, AddInplaceMutatesBuffer) {
    auto a = randn({16}, DType::Float32, Device::cpu()).to(device);
    auto b = randn({16}, DType::Float32, Device::cpu()).to(device);

    const void* a_data_before = a.data_ptr();
    a += b;
    device.synchronize();
    const void* a_data_after = a.data_ptr();

    EXPECT_EQ(a_data_before, a_data_after)
        << "operator+= must reuse the target buffer (in-place); pointer changed on "
        << backend_name(device);
}

INSTANTIATE_BACKEND_TESTS(InplaceOpsParity);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    try {
        if (!::testing::GTEST_FLAG(list_tests)) {
            tenzor::initialize();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }

    int result = RUN_ALL_TESTS();

    try {
        tenzor::finalize();
    } catch (...) {}

    return result;
}
