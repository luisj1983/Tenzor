// Phase 5.5 end-to-end test: verify that the autograd version-counter
// detection fires when a saved tensor is mutated via the internal
// TensorImpl mutation API (mutable_shape / mutable_strides / set_offset).
//
// The chain is:
//   1. Phase 1.3 made those mutators bump version_counter_.
//   2. Function::save_for_backward() captures t.version() at forward.
//   3. Function::validate_saved_tensors() compares them at backward.
//
// This test stitches the three together: do a forward that saves a
// tensor, mutate its metadata, call backward, expect a clear error.

#define TENZOR_INTERNAL 1

#include <gtest/gtest.h>

#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/core/tensor.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>

namespace tenzor {
namespace {

class InplaceVersionDetectionTest : public ::testing::Test {
protected:
    void SetUp() override { tenzor::initialize(); }
};

TEST_F(InplaceVersionDetectionTest, MutatingSavedTensorShapeIsDetected) {
    // Build a tiny autograd graph: y = x * x. The Mul op saves x for
    // backward (to compute dy/dx = 2x).
    auto x_tensor = ones({3, 4}, DType::Float32, Device::cpu());
    Variable x(x_tensor, /*requires_grad=*/true);

    Variable y = x * x;  // Saves x

    // Sanity: version is fresh at this point.
    auto v_before = x.tensor().version();

    // Mutate the underlying TensorImpl via the internal API. This is the
    // kind of action backend kernels perform when constructing views,
    // but doing it *here* on a tensor that autograd is watching should
    // bump the version counter and make the backward pass detect the
    // modification.
    {
        auto mutable_tensor = x.tensor();
        // Any mutator bumps the version; set_offset(0) is a safe no-op
        // value that still triggers the bump because our Phase 1.3 fix
        // made the mutator unconditional.
        mutable_tensor.set_offset(0);
    }

    auto v_after = x.tensor().version();
    EXPECT_GT(v_after, v_before)
        << "Internal mutator must bump the version counter (Phase 1.3 fix)";

    // Backward must now detect the version mismatch and throw.
    auto grad = ones({3, 4}, DType::Float32, Device::cpu());
    EXPECT_THROW(y.backward(grad), std::runtime_error)
        << "Autograd should throw when a saved tensor's version counter "
           "does not match — this verifies Phase 5.5 wiring.";
}

TEST_F(InplaceVersionDetectionTest, UnmodifiedSavedTensorBackwardSucceeds) {
    // Control: no mutation, backward should complete normally.
    auto x_tensor = ones({3, 4}, DType::Float32, Device::cpu());
    Variable x(x_tensor, /*requires_grad=*/true);
    Variable y = x * x;

    auto grad = ones({3, 4}, DType::Float32, Device::cpu());
    EXPECT_NO_THROW(y.backward(grad));
}

} // namespace
} // namespace tenzor
