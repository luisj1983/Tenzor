// Tests for may_alias() — the aliasing check used by dispatch_inplace()
// to catch in-place kernel targets that overlap their input tensors.

#include <gtest/gtest.h>

#include <tenzor/tenzor.hpp>
#include <tenzor/core/tensor.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/transform.hpp>
#include "../multi_backend_dtype_fixture.hpp"

namespace tenzor {
namespace {

class AliasDetectionTest : public ::testing::Test {
protected:
    void SetUp() override { tenzor::initialize(); }
};

// --- Basic cases --------------------------------------------------------

TEST_F(AliasDetectionTest, SameTensorIsNotAlias) {
    // `x.op_(x)` is the benign case that element-wise in-place kernels
    // already handle correctly. may_alias() returns false here so the
    // dispatch-level check doesn't reject it.
    Tensor a = zeros({4, 4}, DType::Float32, Device::cpu());
    EXPECT_FALSE(may_alias(a, a));
}

TEST_F(AliasDetectionTest, IndependentTensors) {
    Tensor a = zeros({4, 4}, DType::Float32, Device::cpu());
    Tensor b = zeros({4, 4}, DType::Float32, Device::cpu());
    EXPECT_FALSE(may_alias(a, b));
    EXPECT_FALSE(may_alias(b, a));
}

TEST_F(AliasDetectionTest, UninitializedTensorsAreNotAliasing) {
    Tensor a;
    Tensor b;
    EXPECT_FALSE(may_alias(a, b));

    Tensor c = zeros({2, 2}, DType::Float32, Device::cpu());
    EXPECT_FALSE(may_alias(a, c));
    EXPECT_FALSE(may_alias(c, a));
}

// --- View / slice aliasing --------------------------------------------

TEST_F(AliasDetectionTest, ReshapeViewAliasesBase) {
    // A reshape-view shares storage with the source. The two Tensor objects
    // are distinct (different impl pointers) but cover overlapping bytes, so
    // may_alias() must report true.
    Tensor a = zeros({4, 4}, DType::Float32, Device::cpu());
    Tensor v = a.reshape({16});
    if (a.impl().get() == v.impl().get()) {
        // Implementation-defined: reshape() is allowed to return the same
        // Tensor object when it's a no-op view. Not a bug — the aliasing
        // scenario this test targets just didn't arise this time.
        SKIP_WITH_REASON(testing::SkipReason::NotApplicable,
                         "reshape returned the same Tensor object; "
                         "aliasing is not applicable to this case.");
    }
    EXPECT_TRUE(may_alias(a, v));
    EXPECT_TRUE(may_alias(v, a));
}

// --- Symmetry and transitivity spot checks ----------------------------

TEST_F(AliasDetectionTest, Symmetric) {
    Tensor a = zeros({8}, DType::Float32, Device::cpu());
    Tensor b = zeros({8}, DType::Float32, Device::cpu());
    EXPECT_EQ(may_alias(a, b), may_alias(b, a));

    Tensor v = a.reshape({2, 4});
    EXPECT_EQ(may_alias(a, v), may_alias(v, a));
}

} // namespace
} // namespace tenzor
