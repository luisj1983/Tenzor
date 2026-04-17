/**
 * @file test_inplace_version_detection_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for autograd in-place version detection
 */

#define TENZOR_INTERNAL 1

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/core/tensor.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class InplaceVersionDetectionMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(InplaceVersionDetectionMultiDTypeTest, MutatingSavedTensorShapeIsDetected) {
    auto x_tensor = ones({3, 4}, dtype(), device());
    Variable x(x_tensor, /*requires_grad=*/true);

    Variable y = x * x;  // Saves x

    auto v_before = x.tensor().version();

    {
        auto mutable_tensor = x.tensor();
        mutable_tensor.set_offset(0);
    }

    auto v_after = x.tensor().version();
    EXPECT_GT(v_after, v_before)
        << "Internal mutator must bump the version counter";

    auto grad = ones({3, 4}, dtype(), device());
    EXPECT_THROW(y.backward(grad), std::runtime_error)
        << "Autograd should throw when a saved tensor's version counter "
           "does not match";
}

TEST_P(InplaceVersionDetectionMultiDTypeTest, UnmodifiedSavedTensorBackwardSucceeds) {
    auto x_tensor = ones({3, 4}, dtype(), device());
    Variable x(x_tensor, /*requires_grad=*/true);
    Variable y = x * x;

    auto grad = ones({3, 4}, dtype(), device());
    EXPECT_NO_THROW(y.backward(grad));
}

TEST_P(InplaceVersionDetectionMultiDTypeTest, VersionBumpDetectedAfterSetOffset) {
    auto t = zeros({3, 4}, dtype(), device());
    auto v0 = t.version();
    t.set_offset(0);
    EXPECT_EQ(t.version(), v0 + 1);
}

TEST_P(InplaceVersionDetectionMultiDTypeTest, VersionBumpDetectedAfterMutableShape) {
    auto t = zeros({3, 4}, dtype(), device());
    auto v0 = t.version();
    (void)t.mutable_shape();
    EXPECT_EQ(t.version(), v0 + 1);
}

TEST_P(InplaceVersionDetectionMultiDTypeTest, VersionBumpDetectedAfterMutableStrides) {
    auto t = zeros({3, 4}, dtype(), device());
    auto v0 = t.version();
    (void)t.mutable_strides();
    EXPECT_EQ(t.version(), v0 + 1);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(InplaceVersionDetectionMultiDTypeTest);
