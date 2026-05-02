/**
 * @file test_inplace_autograd.cpp
 * @brief Tests for in-place operations interaction with autograd version counters
 *
 * Verifies:
 * - In-place ops (fill_, zero_, operator+=, operator*=) bump version counters
 * - save_for_backward correctly records tensor version at save time
 * - In-place modification after save_for_backward is detected during backward
 * - Version counter propagation through views and clones
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/autograd/function.hpp>
#include <tenzor/autograd/ops.hpp>
#include <tenzor/autograd/engine.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/creation.hpp>
#include "../grad_flow_helpers.hpp"

using namespace tenzor;

// ============================================================================
// Test Fixture
// ============================================================================

class InplaceAutogradTest : public ::testing::Test {
protected:
    static bool initialized;
    void SetUp() override {
        if (!initialized) {
            tenzor::initialize();
            initialized = true;
        }
    }
};

bool InplaceAutogradTest::initialized = false;

// ============================================================================
// 1. Version Counter Basics
// ============================================================================

TEST_F(InplaceAutogradTest, NewTensorVersionIsZero) {
    auto t = ones({3, 3}, DType::Float32, Device::cpu());
    EXPECT_EQ(t.version(), 0u);
}

TEST_F(InplaceAutogradTest, FillBumpsVersion) {
    auto t = zeros({4, 4}, DType::Float32, Device::cpu());
    uint64_t v0 = t.version();

    t.fill_(1.0);
    uint64_t v1 = t.version();
    EXPECT_GT(v1, v0) << "fill_() should bump version counter";

    t.fill_(2.0);
    uint64_t v2 = t.version();
    EXPECT_GT(v2, v1) << "Second fill_() should bump version again";
}

TEST_F(InplaceAutogradTest, ZeroBumpsVersion) {
    auto t = ones({3, 3}, DType::Float32, Device::cpu());
    uint64_t v0 = t.version();

    t.zero_();
    uint64_t v1 = t.version();
    EXPECT_GT(v1, v0) << "zero_() should bump version counter";
}

TEST_F(InplaceAutogradTest, OperatorPlusEqualBumpsVersion) {
    auto t = ones({3, 3}, DType::Float32, Device::cpu());
    auto other = ones({3, 3}, DType::Float32, Device::cpu());
    uint64_t v0 = t.version();

    t += other;
    uint64_t v1 = t.version();
    EXPECT_GT(v1, v0) << "operator+= should bump version counter";
}

TEST_F(InplaceAutogradTest, OperatorTimesEqualBumpsVersion) {
    auto t = ones({3, 3}, DType::Float32, Device::cpu());
    auto other = ones({3, 3}, DType::Float32, Device::cpu()) * 2.0f;
    uint64_t v0 = t.version();

    t *= other;
    uint64_t v1 = t.version();
    EXPECT_GT(v1, v0) << "operator*= should bump version counter";
}

TEST_F(InplaceAutogradTest, OperatorMinusEqualBumpsVersion) {
    auto t = ones({3, 3}, DType::Float32, Device::cpu());
    auto other = ones({3, 3}, DType::Float32, Device::cpu());
    uint64_t v0 = t.version();

    t -= other;
    uint64_t v1 = t.version();
    EXPECT_GT(v1, v0) << "operator-= should bump version counter";
}

TEST_F(InplaceAutogradTest, OperatorDivEqualBumpsVersion) {
    auto t = ones({3, 3}, DType::Float32, Device::cpu());
    auto other = ones({3, 3}, DType::Float32, Device::cpu()) * 2.0f;
    uint64_t v0 = t.version();

    t /= other;
    uint64_t v1 = t.version();
    EXPECT_GT(v1, v0) << "operator/= should bump version counter";
}

TEST_F(InplaceAutogradTest, MultipleFillsIncrementSequentially) {
    auto t = zeros({2, 2}, DType::Float32, Device::cpu());
    uint64_t v0 = t.version();

    t.fill_(1.0);
    uint64_t v1 = t.version();

    t.fill_(2.0);
    uint64_t v2 = t.version();

    t.fill_(3.0);
    uint64_t v3 = t.version();

    EXPECT_LT(v0, v1);
    EXPECT_LT(v1, v2);
    EXPECT_LT(v2, v3);
    // Each fill_ should increment by exactly 1
    EXPECT_EQ(v1, v0 + 1);
    EXPECT_EQ(v2, v1 + 1);
    EXPECT_EQ(v3, v2 + 1);
}

// ============================================================================
// 2. Version Counter with Clone
// ============================================================================

TEST_F(InplaceAutogradTest, CloneGetsIndependentVersion) {
    auto t = ones({3, 3}, DType::Float32, Device::cpu());
    t.fill_(5.0);  // bump version to 1
    uint64_t v_before = t.version();

    auto c = t.clone();
    // Modifying clone should not affect original's version
    c.fill_(10.0);

    EXPECT_EQ(t.version(), v_before)
        << "Modifying clone should not affect original's version counter";
}

// ============================================================================
// 3. In-Place Ops Rejected on Tensors with requires_grad
// ============================================================================

TEST_F(InplaceAutogradTest, InplaceAddRejectsRequiresGrad) {
    auto t = ones({3, 3}, DType::Float32, Device::cpu());
    t.set_requires_grad(true);
    auto other = ones({3, 3}, DType::Float32, Device::cpu());

    // The free-function add_() should throw on tensors that require grad
    EXPECT_THROW(add_(t, other), std::runtime_error);
}

TEST_F(InplaceAutogradTest, InplaceMulRejectsRequiresGrad) {
    auto t = ones({3, 3}, DType::Float32, Device::cpu());
    t.set_requires_grad(true);
    auto other = ones({3, 3}, DType::Float32, Device::cpu());

    EXPECT_THROW(mul_(t, other), std::runtime_error);
}

// ============================================================================
// 4. Version Mismatch Detection with save_for_backward
// ============================================================================

TEST_F(InplaceAutogradTest, InplaceAfterForwardDetectedOnBackward) {
    // Build a computation graph: y = x * x
    // Then modify x in-place before calling backward.
    // The engine validates saved tensor versions before backward(),
    // so in-place modification should be detected and throw.

    auto x = Variable(ones({3, 3}, DType::Float32, Device::cpu()), /*requires_grad=*/true);
    auto y = x * x;  // Forward: MulBackward saves x via save_for_backward()

    // Modify the underlying tensor in-place via fill_
    x.tensor().fill_(999.0);

    auto loss = tenzor::sum(y);
    EXPECT_THROW(loss.backward(), std::runtime_error)
        << "backward() should detect in-place modification of saved tensor";
}

TEST_F(InplaceAutogradTest, InplaceAfterForwardNoSavedTensorsAdd) {
    // AddBackward does not save input tensors (only needs shapes for
    // broadcasting reduction), so in-place modification of inputs
    // does not trigger a version mismatch. backward() should succeed.
    auto x = Variable(ones({2, 2}, DType::Float32, Device::cpu()), /*requires_grad=*/true);
    auto y = x + x;

    // Modify underlying tensor in-place
    x.tensor().zero_();

    auto loss = tenzor::sum(y);
    EXPECT_NO_THROW(loss.backward())
        << "AddBackward has no saved tensors, so in-place modification is undetected";

    // Even though the input was modified in place, backward should still
    // populate a non-zero gradient — AddBackward's gradient is `grad_output`
    // for each addend, independent of the input value, so the post-zero_()
    // input still produces a real gradient. Verifies backward actually ran.
    EXPECT_GRAD_FLOWS(x);
}

TEST_F(InplaceAutogradTest, NoModificationNoThrow) {
    // When no in-place modification happens, backward should succeed
    auto x = Variable(ones({3, 3}, DType::Float32, Device::cpu()), /*requires_grad=*/true);
    auto y = x * x;
    auto loss = tenzor::sum(y);

    EXPECT_NO_THROW(loss.backward())
        << "backward() should succeed when no in-place modification occurred";

    EXPECT_GRAD_FLOWS(x);
}

// ============================================================================
// 5. Version Counter with Multiple Operations
// ============================================================================

TEST_F(InplaceAutogradTest, MixedInplaceOpsBumpVersionCumulatively) {
    auto t = ones({4, 4}, DType::Float32, Device::cpu());
    auto other = ones({4, 4}, DType::Float32, Device::cpu());
    uint64_t v0 = t.version();

    t.fill_(2.0);  // v0 + 1
    t += other;     // v0 + 2
    t *= other;     // v0 + 3
    t.zero_();      // v0 + 4

    EXPECT_EQ(t.version(), v0 + 4)
        << "Four in-place ops should bump version by 4";
}

TEST_F(InplaceAutogradTest, VersionCounterOnDifferentDtypes) {
    // Verify version counter works across different dtypes
    auto t_f32 = ones({2, 2}, DType::Float32, Device::cpu());
    auto t_f64 = ones({2, 2}, DType::Float64, Device::cpu());

    EXPECT_EQ(t_f32.version(), 0u);
    EXPECT_EQ(t_f64.version(), 0u);

    t_f32.fill_(5.0);
    t_f64.fill_(5.0);

    EXPECT_EQ(t_f32.version(), 1u);
    EXPECT_EQ(t_f64.version(), 1u);
}

// ============================================================================
// 6. Autograd Variable Backward Without Modification
// ============================================================================

TEST_F(InplaceAutogradTest, SimpleGradientComputationWorks) {
    // Sanity: y = 2*x, dy/dx = 2
    auto x = Variable(ones({2, 2}, DType::Float32, Device::cpu()), /*requires_grad=*/true);
    auto y = x + x;  // 2*x
    auto loss = tenzor::sum(y);
    loss.backward();

    EXPECT_TRUE(x.has_grad());
    auto grad = x.grad().value();
    auto* data = grad.data<float>();
    for (int64_t i = 0; i < grad.numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 2.0f)
            << "Gradient of sum(x+x) w.r.t. x should be 2 at index " << i;
    }
}

TEST_F(InplaceAutogradTest, SquaredGradientComputationWorks) {
    // y = x^2, dy/dx = 2x
    auto x_data = ones({2, 2}, DType::Float32, Device::cpu()) * 3.0f;
    auto x = Variable(x_data, /*requires_grad=*/true);
    auto y = x * x;
    auto loss = tenzor::sum(y);
    loss.backward();

    EXPECT_TRUE(x.has_grad());
    auto grad = x.grad().value();
    auto* data = grad.data<float>();
    for (int64_t i = 0; i < grad.numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 6.0f)
            << "Gradient of sum(x*x) w.r.t. x (x=3) should be 6 at index " << i;
    }
}

// ============================================================================
// Version Tracking for save_for_backward (Phase 1B)
// ============================================================================

TEST_F(InplaceAutogradTest, MulBackwardDetectsInplaceModification) {
    // y = a * b; in-place modify a; then backward → should detect stale saved tensor
    auto a_data = ones({2, 2}, DType::Float32, Device::cpu()) * 2.0f;
    auto b_data = ones({2, 2}, DType::Float32, Device::cpu()) * 3.0f;
    auto a = Variable(a_data, /*requires_grad=*/true);
    auto b = Variable(b_data, /*requires_grad=*/true);
    auto y = a * b;  // MulBackward saves a and b

    // In-place modify a's underlying tensor after it was saved
    a_data.fill_(99.0f);

    // backward should detect the version mismatch and throw
    EXPECT_THROW(tenzor::sum(y).backward(), std::runtime_error)
        << "MulBackward should detect in-place modification of saved tensor";
}

TEST_F(InplaceAutogradTest, DivBackwardDetectsInplaceModification) {
    auto a_data = ones({2, 2}, DType::Float32, Device::cpu()) * 6.0f;
    auto b_data = ones({2, 2}, DType::Float32, Device::cpu()) * 3.0f;
    auto a = Variable(a_data, /*requires_grad=*/true);
    auto b = Variable(b_data, /*requires_grad=*/true);
    auto y = a / b;  // DivBackward saves a and b

    b_data.fill_(99.0f);

    EXPECT_THROW(tenzor::sum(y).backward(), std::runtime_error)
        << "DivBackward should detect in-place modification of saved tensor";
}

TEST_F(InplaceAutogradTest, MatMulBackwardDetectsInplaceModification) {
    auto a_data = ones({2, 3}, DType::Float32, Device::cpu());
    auto b_data = ones({3, 2}, DType::Float32, Device::cpu());
    auto a = Variable(a_data, /*requires_grad=*/true);
    auto b = Variable(b_data, /*requires_grad=*/true);
    auto y = matmul(a, b);  // MatMulBackward saves a and b

    a_data.fill_(99.0f);

    EXPECT_THROW(tenzor::sum(y).backward(), std::runtime_error)
        << "MatMulBackward should detect in-place modification of saved tensor";
}

TEST_F(InplaceAutogradTest, AddBackwardWithVersionTracking) {
    // AddBackward in variable.cpp operator+ now uses save_for_backward
    // Verify normal backward still works (no false positive)
    auto a = Variable(ones({2, 2}, DType::Float32, Device::cpu()) * 2.0f, true);
    auto b = Variable(ones({2, 2}, DType::Float32, Device::cpu()) * 3.0f, true);
    auto y = a + b;
    auto loss = tenzor::sum(y);

    // No in-place modification → backward should succeed
    EXPECT_NO_THROW(loss.backward());
    EXPECT_TRUE(a.has_grad());
}
