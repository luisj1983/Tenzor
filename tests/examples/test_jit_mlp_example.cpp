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
 */

#include <gtest/gtest.h>

#include <tenzor/tenzor.hpp>

#include "../../examples/cpp/training/jit_mlp_training_runner.hpp"

class JitMlpExampleRegression : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        tenzor::initialize();
    }
};

TEST_F(JitMlpExampleRegression, TrainingConvergesAndJitMatchesEager) {
    double initial = -1.0;
    double final_ = -1.0;
    double jit_diff = -1.0;
    int rc = tenzor::examples::jit_mlp::run_jit_mlp_training(
        /*epochs=*/200,
        &initial,
        &final_,
        &jit_diff,
        tenzor::Device::cpu(),
        /*verbose=*/false);

    ASSERT_EQ(rc, 0) << "JIT MLP training/compile runner returned a non-zero "
                         "exit code (see stderr above for the JIT failure)";
    ASSERT_GE(initial, 0.0) << "initial loss not captured (epoch 0 missed?)";
    EXPECT_LT(final_, initial)
        << "MLP training made no progress — a regression in Linear/ReLU "
           "backward or the manual SGD update. initial=" << initial
        << ", final=" << final_;

    ASSERT_GE(jit_diff, 0.0) << "JIT-vs-eager diff not captured";
    EXPECT_LT(jit_diff, 1e-3)
        << "JIT-compiled inference diverged from eager execution by "
        << jit_diff << " — a regression in JIT tracing, codegen, or "
           "cross-backend dispatch for this model's Linear/ReLU/Linear "
           "forward pass.";
}
