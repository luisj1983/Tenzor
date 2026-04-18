/**
 * @file test_custom_op_parity.cpp
 * @brief Backend parity for user-registered custom ops (Phase 4.3).
 *
 * Custom ops are registered per device via register_custom_op(name, device,
 * kernel). A multi-backend custom op registers a kernel for each available
 * Device::Type. The parity guarantee we test is: dispatching the same named
 * custom op with the same input tensor on different backends produces the
 * same result when all kernels compute the same function.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/custom_op.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;


class CustomOpParity : public BackendTest {};
// Register the same `f(x) = 2x + 1` custom op on each available backend, then
// dispatch it on tensors living on each backend and compare results.
TEST_P(CustomOpParity, Affine_MultiBackend) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // Kernel body is identical per backend — it just uses tenzor ops which
    // dispatch to the right backend at call time.
    auto kernel = [](std::span<const Tensor> inputs,
                     const OpAttributes&) -> Tensor {
        return inputs[0] * 2.0f + 1.0f;
    };

    // Register on every available backend under a unique name.
    CustomOpId op_id(OpId::Add);  // placeholder; reassigned below
    bool registered_any = false;
    for (const auto& dev : backends) {
        try {
            op_id = register_custom_op("parity::affine2x_plus_1",
                                        dev.type, kernel);
            registered_any = true;
        } catch (const std::exception& e) {
            ADD_FAILURE() << "register_custom_op failed on "
                      << backend_name(dev) << ": " << e.what() << std::endl;
        }
    }
    if (!registered_any) GTEST_SKIP() << "No backends accepted the kernel";

    auto input_cpu = randn({4, 8}, DType::Float32, Device::cpu());
    // dispatch_custom_op has a Variable overload and a Tensor overload.
    // Wrap with Variables here so we can use the Variable path.
    auto ref_var = dispatch_custom_op(op_id, {Variable(input_cpu, false)});
    Tensor ref = ref_var.tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto out = dispatch_custom_op(op_id,
                {Variable(input_cpu.to(backends[i]), false)});
            backends[i].synchronize();
            SCOPED_TRACE(std::string("CustomOp on ") + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, out.tensor().to(Device::cpu()),
                                 1e-5f, 1e-7f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "CustomOp dispatch failed on "
                      << backend_name(backends[i]) << ": " << e.what()
                      << std::endl;
        }
    }
}

// Custom op with backward — verify both forward and gradient match.
TEST_P(CustomOpParity, Squared_Backward_MultiBackend) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto forward = [](std::span<const Tensor> inputs,
                      const OpAttributes&) -> Tensor {
        return inputs[0] * inputs[0];
    };
    auto backward = [](std::span<const Tensor> saved,
                        std::span<const Tensor> grads) -> std::vector<Tensor> {
        return {saved[0] * grads[0] * 2.0f};
    };

    CustomOpId op_id(OpId::Add);
    bool registered_any = false;
    for (const auto& dev : backends) {
        try {
            op_id = register_custom_op_with_backward(
                "parity::squared_with_bwd", dev.type, forward, backward);
            registered_any = true;
        } catch (const std::exception& e) {
            ADD_FAILURE() << "register_custom_op_with_backward failed on "
                      << backend_name(dev) << ": " << e.what() << std::endl;
        }
    }
    // Tag inline so scripts/count_skips.py classifies without needing the
    // SKIP_WITH_REASON macro header.
    if (!registered_any) GTEST_SKIP() <<
        "[SkipReason::KernelNotImplemented] Custom op failed to register on any backend";

    auto input_cpu = randn({8}, DType::Float32, Device::cpu());

    // Reference
    auto x_cpu = Variable(input_cpu.clone(), true);
    auto y_cpu = dispatch_custom_op(op_id, {x_cpu});
    sum(y_cpu).backward();
    ASSERT_TRUE(x_cpu.has_grad());
    Tensor ref_out = y_cpu.tensor();
    Tensor ref_grad = x_cpu.grad().value();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto x_dev = Variable(input_cpu.to(backends[i]), true);
            auto y_dev = dispatch_custom_op(op_id, {x_dev});
            sum(y_dev).backward();
            backends[i].synchronize();
            SCOPED_TRACE(std::string("CustomOp+bwd on ")
                         + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref_out, y_dev.tensor().to(Device::cpu()),
                                 1e-5f, 1e-7f);
            ASSERT_TRUE(x_dev.has_grad());
            EXPECT_TENSORS_CLOSE(ref_grad,
                                 x_dev.grad().value().to(Device::cpu()),
                                 1e-4f, 1e-6f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "CustomOp+bwd dispatch failed on "
                      << backend_name(backends[i]) << ": " << e.what()
                      << std::endl;
        }
    }
}

INSTANTIATE_BACKEND_TESTS(CustomOpParity);


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
    try { tenzor::finalize(); } catch (...) {}
    return result;
}
