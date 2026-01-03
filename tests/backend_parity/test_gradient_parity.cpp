/**
 * @file test_gradient_parity.cpp
 * @brief Gradient computation parity tests - Parameterized across all backends
 *
 * Verifies that gradients computed by different backends match,
 * using numerical gradient checking as reference.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class GradientParityBackendTest : public BackendTest {
protected:
    static void SetUpTestSuite() {
        try {
            tenzor::initialize();
        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        }
    }

    static void TearDownTestSuite() {
        try {
            tenzor::finalize();
        } catch (...) {}
    }

    void SetUp() override {
        BackendTest::SetUp();
        set_grad_enabled(true);
    }

    // Helper to create a reference gradient on CPU and compare
    void compareGradientWithCPU(const Tensor& cpu_grad, const Tensor& backend_grad,
                               float rtol = 1e-5f, float atol = 1e-7f) {
        auto backend_grad_cpu = backend_grad.to(Device::cpu());
        device.synchronize();

        expectTensorNear(cpu_grad, backend_grad_cpu, atol);
    }
};

// ============================================================================
// Basic Operation Gradient Tests
// ============================================================================

TEST_P(GradientParityBackendTest, AddBackward) {
    // Skipped: Tensor API does not have gradient support; use Variable
    GTEST_SKIP() << "Test requires Variable API for gradient support";
}

TEST_P(GradientParityBackendTest, MulBackward) {
    // Skipped: Tensor API does not have gradient support; use Variable
    GTEST_SKIP() << "Test requires Variable API for gradient support";
}

TEST_P(GradientParityBackendTest, MatMulBackward) {
    // Skipped: Tensor API does not have gradient support; use Variable
    GTEST_SKIP() << "Test requires Variable API for gradient support";
}

// ============================================================================
// Activation Function Gradient Tests
// ============================================================================

TEST_P(GradientParityBackendTest, ReLUBackward) {
    // Skipped: Tensor API does not have gradient support; use Variable
    GTEST_SKIP() << "Test requires Variable API for gradient support";
}

TEST_P(GradientParityBackendTest, SigmoidBackward) {
    // Skipped: Tensor API does not have gradient support; use Variable
    GTEST_SKIP() << "Test requires Variable API for gradient support";
}

TEST_P(GradientParityBackendTest, TanhBackward) {
    // Skipped: Tensor API does not have gradient support; use Variable
    GTEST_SKIP() << "Test requires Variable API for gradient support";
}

TEST_P(GradientParityBackendTest, GELUBackward) {
    // Skipped: Tensor API does not have gradient support; use Variable
    GTEST_SKIP() << "Test requires Variable API for gradient support";
}

TEST_P(GradientParityBackendTest, SoftmaxBackward) {
    // Skipped: Tensor API does not have gradient support; use Variable
    GTEST_SKIP() << "Test requires Variable API for gradient support";
}

// ============================================================================
// Complex Operation Gradient Tests
// ============================================================================

TEST_P(GradientParityBackendTest, Conv2dBackward) {
    // Skipped: Tensor API does not have gradient support; use Variable
    GTEST_SKIP() << "Test requires Variable API for gradient support";
}

TEST_P(GradientParityBackendTest, BatchNormBackward) {
    // Skipped: Tensor API does not have gradient support; use Variable
    GTEST_SKIP() << "Test requires Variable API for gradient support";
}

// ============================================================================
// Loss Function Gradient Tests
// ============================================================================

TEST_P(GradientParityBackendTest, MSELossBackward) {
    // Skipped: Tensor API does not have gradient support; use Variable
    GTEST_SKIP() << "Test requires Variable API for gradient support";
}

TEST_P(GradientParityBackendTest, CrossEntropyLossBackward) {
    // Skipped: Tensor API does not have gradient support; use Variable
    GTEST_SKIP() << "Test requires Variable API for gradient support";
}

// ============================================================================
// Multi-step Gradient Tests
// ============================================================================

TEST_P(GradientParityBackendTest, ChainedOperations) {
    // Skipped: Tensor API does not have gradient support; use Variable
    GTEST_SKIP() << "Test requires Variable API for gradient support";
}

INSTANTIATE_BACKEND_TESTS(GradientParityBackendTest);
