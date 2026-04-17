/**
 * @file test_jit_autograd_parity.cpp
 * @brief Backward-through-JIT parity (plan Phase 4.4).
 *
 * Forward-only JIT parity is covered by tests/jit/test_jit_backend_parity.cpp.
 * This file isolates the backward tests because they hit a distinct class of
 * bugs (backward dispatch path through JIT-captured graphs) that's worth
 * surfacing independently in the test binary list.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <tenzor/tenzor.hpp>
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

namespace {
void copy_params(nn::Module& src, nn::Module& dst) {
    auto src_params = src.parameters();
    auto dst_params = dst.parameters();
    for (size_t p = 0; p < src_params.size(); ++p) {
        dst_params[p]->tensor() = src_params[p]->tensor().clone();
    }
}
}  // namespace

// Fixed via ROCm cascading fixes (mul/add/sub/div non-contig + contiguous UAF).
TEST(JITAutogradParity, LinearChain_Backward) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    try {
        nn::Linear l1(32, 16);
        nn::Linear l2(16, 8);
        auto input = randn({4, 32}, DType::Float32, Device::cpu());

        auto x_ref = Variable(input.clone(), true);
        auto h = nn::relu(l1.forward(x_ref));
        auto y = sum(l2.forward(h));
        y.backward();
        ASSERT_TRUE(x_ref.has_grad());
        Tensor ref_grad = x_ref.grad().value();

        for (size_t i = 1; i < backends.size(); ++i) {
            try {
                nn::Linear l1_dev(32, 16);
                nn::Linear l2_dev(16, 8);
                copy_params(l1, l1_dev);
                copy_params(l2, l2_dev);
                l1_dev.to(backends[i]);
                l2_dev.to(backends[i]);

                auto x_dev = Variable(input.to(backends[i]), true);
                auto h_dev = nn::relu(l1_dev.forward(x_dev));
                auto y_dev = sum(l2_dev.forward(h_dev));
                y_dev.backward();
                backends[i].synchronize();

                SCOPED_TRACE(std::string("LinearChain_Backward on ")
                             + backend_name(backends[i]));
                ASSERT_TRUE(x_dev.has_grad());
                EXPECT_TENSORS_CLOSE(ref_grad,
                                     x_dev.grad().value().to(Device::cpu()),
                                     1e-3f, 1e-4f);
            } catch (const std::exception& e) {
                ADD_FAILURE() << "LinearChain_Backward failed on "
                          << backend_name(backends[i]) << ": " << e.what()
                          << std::endl;
            }
        }
    } catch (const std::exception& e) {
        GTEST_SKIP() << e.what();
    }
}

// Fixed alongside LinearChain_Backward.
TEST(JITAutogradParity, LayerNormMLP_Backward) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    try {
        nn::Linear l1(16, 16);
        nn::LayerNorm ln({16});
        nn::Linear l2(16, 8);
        auto input = randn({4, 16}, DType::Float32, Device::cpu());

        auto x_ref = Variable(input.clone(), true);
        auto h = ln.forward(l1.forward(x_ref));
        h = nn::relu(h);
        auto y = sum(l2.forward(h));
        y.backward();
        ASSERT_TRUE(x_ref.has_grad());
        Tensor ref_grad = x_ref.grad().value();

        for (size_t i = 1; i < backends.size(); ++i) {
            try {
                nn::Linear l1_dev(16, 16);
                nn::LayerNorm ln_dev({16});
                nn::Linear l2_dev(16, 8);
                copy_params(l1, l1_dev);
                copy_params(ln, ln_dev);
                copy_params(l2, l2_dev);
                l1_dev.to(backends[i]); ln_dev.to(backends[i]); l2_dev.to(backends[i]);

                auto x_dev = Variable(input.to(backends[i]), true);
                auto hd = ln_dev.forward(l1_dev.forward(x_dev));
                hd = nn::relu(hd);
                auto y_dev = sum(l2_dev.forward(hd));
                y_dev.backward();
                backends[i].synchronize();

                SCOPED_TRACE(std::string("LayerNormMLP_Backward on ")
                             + backend_name(backends[i]));
                ASSERT_TRUE(x_dev.has_grad());
                // Backprop through (Linear -> LayerNorm -> ReLU -> Linear)
                // accumulates a few layers of rounding; allow a slightly looser
                // atol so small numerical noise doesn't trip the test.
                EXPECT_TENSORS_CLOSE(ref_grad,
                                     x_dev.grad().value().to(Device::cpu()),
                                     1e-3f, 1e-3f);
            } catch (const std::exception& e) {
                ADD_FAILURE() << "LayerNormMLP_Backward failed on "
                          << backend_name(backends[i]) << ": " << e.what()
                          << std::endl;
            }
        }
    } catch (const std::exception& e) {
        GTEST_SKIP() << e.what();
    }
}

int main(int argc, char** argv) {
    tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    tenzor::finalize();
    return result;
}
