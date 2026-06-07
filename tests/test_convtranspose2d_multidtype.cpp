/**
 * @file test_convtranspose2d_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for ConvTranspose2d layer
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/conv.hpp>
#include "multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class ConvTranspose2dMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(ConvTranspose2dMultiDTypeTest, UpsampleShape) {
    // stride=2 upsamples spatially
    nn::ConvTranspose2d deconv(4, 8, 4, 2, 1);
    auto input_cpu = tenzor::randn({1, 4, 8, 8}, DType::Float32, Device::cpu());
    auto ref = deconv.forward(Variable(input_cpu, false));

    convert_model(deconv);
    auto input = Variable(input_cpu.to(dtype_).to(device_), false);
    auto output = deconv.forward(input);
    expectShape(output.tensor(), {1, 8, 16, 16});
    expectDevice(output.tensor());
    expectDType(output.tensor());
    expectTensorNear(output.tensor(), ref.tensor(), std::max(atol_, 5e-2f));
}

TEST_P(ConvTranspose2dMultiDTypeTest, IdentityStride) {
    nn::ConvTranspose2d deconv(3, 6, 3, 1, 1);
    auto input_cpu = tenzor::randn({1, 3, 8, 8}, DType::Float32, Device::cpu());
    auto ref = deconv.forward(Variable(input_cpu, false));

    convert_model(deconv);
    auto input = Variable(input_cpu.to(dtype_).to(device_), false);
    auto output = deconv.forward(input);
    expectShape(output.tensor(), {1, 6, 8, 8});
    expectTensorNear(output.tensor(), ref.tensor(), std::max(atol_, 5e-2f));
}

TEST_P(ConvTranspose2dMultiDTypeTest, BatchDim) {
    nn::ConvTranspose2d deconv(4, 8, 3, 2, 1);
    auto input_cpu = tenzor::randn({4, 4, 6, 6}, DType::Float32, Device::cpu());
    auto ref = deconv.forward(Variable(input_cpu, false));

    convert_model(deconv);
    auto input = Variable(input_cpu.to(dtype_).to(device_), false);
    auto output = deconv.forward(input);
    EXPECT_EQ(output.tensor().shape()[0], 4);
    EXPECT_EQ(output.tensor().shape()[1], 8);
    expectTensorNear(output.tensor(), ref.tensor(), std::max(atol_, 5e-2f));
}

TEST_P(ConvTranspose2dMultiDTypeTest, BackwardProducesGradients) {
    // Reseed before each construction so reference and device modules get
    // byte-identical weights (the fixture seeds once per test; two sequential
    // constructions would otherwise draw different init weights).
    tenzor::manual_seed(44);
    nn::ConvTranspose2d deconv_ref(3, 6, 4, 2, 1);
    auto input_cpu = tenzor::randn({1, 3, 4, 4}, DType::Float32, Device::cpu());
    auto in_ref = Variable(input_cpu, /*requires_grad=*/true);
    auto out_ref = deconv_ref.forward(in_ref);
    tenzor::sum(out_ref).backward();

    tenzor::manual_seed(44);
    nn::ConvTranspose2d deconv(3, 6, 4, 2, 1);
    convert_model(deconv);
    auto input = Variable(input_cpu.to(dtype_).to(device_), /*requires_grad=*/true);
    auto output = deconv.forward(input);
    auto loss = tenzor::sum(output);
    loss.backward();

    ASSERT_TRUE(input.has_grad()) << "input gradient must be populated";
    auto g = input.grad().value();
    EXPECT_EQ(g.shape()[0], 1);
    EXPECT_EQ(g.shape()[1], 3);
    expectDevice(g);
    expectTensorNear(output.tensor(), out_ref.tensor(), std::max(atol_, 5e-2f));
    expectTensorNear(g, in_ref.grad().value(), std::max(atol_, 5e-2f));

    for (const auto& [name, p] : deconv.named_parameters()) {
        ASSERT_TRUE(p->has_grad()) << "parameter " << name << " missing grad";
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ConvTranspose2dMultiDTypeTest);
