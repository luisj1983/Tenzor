/**
 * @file test_identity_multidtype.cpp
 * @brief Multi-backend, multi-dtype tests for nn::Identity.
 *
 * Identity is a no-op pass-through; verify the output is bit-equal to the
 * input and that gradient flows through unchanged.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/identity.hpp>
#include "../../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class IdentityMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(IdentityMultiDTypeTest, ForwardEqualsInput) {
    nn::Identity id;
    Variable input = createInput({3, 4}, /*requires_grad=*/false);
    auto output = id.forward(input);
    expectShape(output.tensor(), {3, 4});
    expectDType(output.tensor());
    // Identity must return the SAME tensor data.
    auto a = input.tensor().to(Device::cpu()).contiguous();
    auto b = output.tensor().to(Device::cpu()).contiguous();
    ASSERT_EQ(a.numel(), b.numel());
    expectTensorNear(b, a);
}

TEST_P(IdentityMultiDTypeTest, BackwardPassThrough) {
    nn::Identity id;
    Variable input = createInput({2, 5}, /*requires_grad=*/true);
    auto output = id.forward(input);
    sum(output).backward();
    ASSERT_TRUE(input.has_grad()) << device().to_string();
    EXPECT_EQ(input.grad()->numel(), input.tensor().numel());
    // Gradient of sum w.r.t. each input element is 1; verify by checking
    // that all elements of grad are equal to 1.0.
    auto g = input.grad()->to(Device::cpu()).to(DType::Float32).contiguous();
    for (int64_t i = 0; i < g.numel(); ++i) {
        EXPECT_NEAR(g.data<float>()[i], 1.0f, 1e-3f);
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(IdentityMultiDTypeTest);
