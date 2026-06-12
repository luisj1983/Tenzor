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
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;


class JITAutogradParity : public BackendTest {};
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
TEST_P(JITAutogradParity, LinearChain_Backward) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("jit autograd parity");

    // Audit: previously the whole body (CPU reference + per-backend loop) was
    // wrapped in try{...}catch(...){GTEST_SKIP(e.what())}, swallowing a broken
    // CPU reference or any structural failure as a clean skip. The inner loop
    // already reports per-backend failures via ADD_FAILURE; let everything else
    // propagate.
    {
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
                // Chained Linear backward accumulates rounding through
                // two MatMuls + ReLU mask. CUDA cuBLAS SGEMM (even with
                // TF32 disabled) picks different kernels from MKL's
                // cblas_sgemm, so the last two–three mantissa bits of
                // the accumulated gradient can differ. Measured deltas
                // fall in the 1.3e-4..1.6e-4 range on RTX 5070 even in
                // full FP32 mode.
                EXPECT_TENSORS_CLOSE(ref_grad,
                                     x_dev.grad().value().to(Device::cpu()),
                                     1e-3f, 5e-4f);
            } catch (const std::exception& e) {
                ADD_FAILURE() << "LinearChain_Backward failed on "
                          << backend_name(backends[i]) << ": " << e.what()
                          << std::endl;
            }
        }
    }
}

// Fixed alongside LinearChain_Backward.
TEST_P(JITAutogradParity, LayerNormMLP_Backward) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("jit autograd parity");

    // Audit: previously the whole body (CPU reference + per-backend loop) was
    // wrapped in try{...}catch(...){GTEST_SKIP(e.what())}, swallowing a broken
    // CPU reference or any structural failure as a clean skip. The inner loop
    // already reports per-backend failures via ADD_FAILURE; let everything else
    // propagate.
    {
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
    }
}

INSTANTIATE_BACKEND_TESTS(JITAutogradParity);


int main(int argc, char** argv) {
    // Force full IEEE 754 FP32 on CUDA matmul (disable TF32 tensor cores)
    // so CPU↔CUDA parity for chained Linear backward is measurable at the
    // 1e-3/1e-4 tolerance the test uses. Without this the test is
    // intermittently flaky: RTX 5070 in TF32 mode drops enough mantissa
    // bits that accumulated gradients exceed the atol.
    setenv("TENZOR_DISABLE_TF32", "1", /*overwrite=*/1);
    tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    tenzor::finalize();
    return result;
}
