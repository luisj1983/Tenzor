/**
 * @file test_jit_mlp_example.cpp
 * @brief Regression test for the JIT-compiled MLP training/inference example.
 *
 * R2-02: this is the first example wired into the "examples-as-tests"
 * regression suite (tests/examples/) that exercises the JIT subsystem
 * end-to-end. Every other showcase/training example only ever runs eager
 * autograd; a cross-backend or trace/compile divergence in a real (if
 * small) model could exist with zero regression coverage from this suite.
 * This test runs the actual example's training loop, asserting BOTH:
 *   1. Loss decreases over training (the standard convergence guard every
 *      other example in this suite already applies), and
 *   2. The JIT-compiled inference forward pass matches eager execution on
 *      the trained model to tight tolerance (the JIT-specific guard).
 *
 * JIT-R122: the original version of this test hardcoded Device::cpu(), and
 * the runner's default CompileConfig (backend="nvrtc") has no native codegen
 * for CPU tensors (JIT-033) -- it silently runs the traced graph through the
 * Graph interpreter, not real compiled kernels. So this test, even 100%
 * passing, never exercised actual JIT-compiled-kernel execution at all, and
 * CUDA/ROCm got zero regression protection despite nvrtc genuinely
 * dispatching to NVRTC/HIPRTC codegen for those devices. Parametrized over
 * every available backend (mirroring test_jit_backend_parity.cpp's
 * LinearChain_JitVsEagerAllBackends and test_multi_param_example.cpp's
 * sibling convention in this same directory), and now asserts
 * num_cached() > 0 so a silent full-eager fallback can't masquerade as a
 * pass.
 */

#include <gtest/gtest.h>

#include <tenzor/tenzor.hpp>

#include "../../examples/cpp/training/jit_mlp_training_runner.hpp"
#include "../backend_parity/parity_test_utils.hpp"

class JitMlpExampleRegression : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        tenzor::initialize();
    }
};

TEST_F(JitMlpExampleRegression, TrainingConvergesAndJitMatchesEagerAllBackends) {
    const auto backends = ::tenzor::testing::get_available_backends();
    ASSERT_FALSE(backends.empty()) << "no backends available at all (not even CPU)";

    for (const auto& device : backends) {
        SCOPED_TRACE(::tenzor::testing::backend_name(device));

        double initial = -1.0;
        double final_ = -1.0;
        double jit_diff = -1.0;
        std::size_t num_cached = 0;
        int rc = tenzor::examples::jit_mlp::run_jit_mlp_training(
            /*epochs=*/200,
            &initial,
            &final_,
            &jit_diff,
            device,
            /*verbose=*/false,
            &num_cached);

        ASSERT_EQ(rc, 0)
            << "JIT MLP training/compile runner returned a non-zero exit "
               "code on " << ::tenzor::testing::backend_name(device)
            << " (see stderr above for the JIT failure)";
        ASSERT_GE(initial, 0.0) << "initial loss not captured (epoch 0 missed?)";
        EXPECT_LT(final_, initial)
            << "MLP training made no progress on "
            << ::tenzor::testing::backend_name(device)
            << " — a regression in Linear/ReLU backward or the manual SGD "
               "update. initial=" << initial << ", final=" << final_;

        ASSERT_GE(jit_diff, 0.0) << "JIT-vs-eager diff not captured";
        EXPECT_LT(jit_diff, 1e-3)
            << "JIT-compiled inference diverged from eager execution by "
            << jit_diff << " on " << ::tenzor::testing::backend_name(device)
            << " — a regression in JIT tracing, codegen, or cross-backend "
               "dispatch for this model's Linear/ReLU/Linear forward pass.";

        // JIT-R122: proves the compiled path (trace -> compile -> cache) was
        // actually taken on THIS backend, not a silent full-eager fallback
        // that would trivially make jit_diff == 0 and pass vacuously.
        EXPECT_GT(num_cached, 0u)
            << "JIT silently fell back to eager on "
            << ::tenzor::testing::backend_name(device)
            << " (num_cached == 0) — the parity check above has no teeth "
               "for this backend.";
    }
}
