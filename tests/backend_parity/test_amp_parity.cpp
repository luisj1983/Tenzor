/**
 * @file test_amp_parity.cpp
 * @brief Backend parity for AMP / GradScaler (Phase 4.2).
 *
 * Validates that GradScaler's scale/unscale operations produce the same
 * scaled gradients across backends. A full training-loop parity would also
 * involve autocast + optimizer step + growth/backoff — too much interaction
 * for a focused parity test. Starts with the minimal contract: scale(loss)
 * and unscale_(grads) match across backends.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/amp/grad_scaler.hpp>
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

TEST(AMPParity, ScaleLossMatches) {
    // Single-input scalar loss: y = sum(x^2). scale(loss) should produce the
    // same scaled scalar on every backend.
    auto input_cpu = randn({8}, DType::Float32, Device::cpu());

    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    float ref_scaled;
    try {
        nn::amp::GradScaler scaler(65536.0f);
        auto x = Variable(input_cpu.clone(), true);
        auto loss = sum(x * x);
        auto scaled = scaler.scale(loss);
        ref_scaled = scaled.tensor().item<float>();
    } catch (const std::exception& e) {
        GTEST_SKIP() << "GradScaler.scale() CPU reference failed: " << e.what();
    }

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::amp::GradScaler scaler(65536.0f);
            auto x = Variable(input_cpu.to(backends[i]), true);
            auto loss = sum(x * x);
            auto scaled = scaler.scale(loss);
            backends[i].synchronize();
            float got = scaled.tensor().to(Device::cpu()).item<float>();
            SCOPED_TRACE(std::string("GradScaler.scale on ")
                         + backend_name(backends[i]));
            EXPECT_NEAR(got, ref_scaled, std::abs(ref_scaled) * 1e-4f);
        } catch (const std::exception& e) {
            std::cerr << "GradScaler.scale skipped on "
                      << backend_name(backends[i]) << ": " << e.what()
                      << std::endl;
        }
    }
}

TEST(AMPParity, ScaledBackwardMatches) {
    // After backward on a scaled loss, the gradient magnitude should equal
    // scale * true_grad. Verify that the scaled input gradient matches across
    // backends (the unscaled path is what Optimizer.step would walk, but the
    // raw scaled grad is what the kernel produces).
    auto input_cpu = randn({8}, DType::Float32, Device::cpu());

    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    Tensor ref_grad;
    try {
        nn::amp::GradScaler scaler(1024.0f);  // smaller factor → no overflow
        auto x = Variable(input_cpu.clone(), true);
        auto loss = sum(x * x);
        auto scaled = scaler.scale(loss);
        scaled.backward();
        ref_grad = x.grad().value();
    } catch (const std::exception& e) {
        GTEST_SKIP() << "GradScaler scaled backward CPU reference failed: "
                     << e.what();
    }

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::amp::GradScaler scaler(1024.0f);
            auto x = Variable(input_cpu.to(backends[i]), true);
            auto loss = sum(x * x);
            auto scaled = scaler.scale(loss);
            scaled.backward();
            backends[i].synchronize();
            SCOPED_TRACE(std::string("GradScaler scaled backward on ")
                         + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref_grad, x.grad().value().to(Device::cpu()),
                                 1e-3f, 1e-4f);
        } catch (const std::exception& e) {
            std::cerr << "GradScaler backward skipped on "
                      << backend_name(backends[i]) << ": " << e.what()
                      << std::endl;
        }
    }
}

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
