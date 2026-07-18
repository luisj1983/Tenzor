/**
 * @file test_conv1d_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for Conv1d layer
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/conv.hpp>
#include "multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class Conv1dMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(Conv1dMultiDTypeTest, BasicForwardShape) {
    auto conv = nn::Conv1d(2, 3, 3, 1, 0, 1, 1, false);
    // CPU reference forward (conv is on CPU before convert_model)
    auto input_cpu = tenzor::randn({1, 2, 5}, DType::Float32, Device::cpu());
    auto ref = conv.forward(Variable(input_cpu, false));

    convert_model(conv);
    auto input = Variable(input_cpu.to(dtype_).to(device_), false);
    auto output = conv.forward(input);
    expectShape(output.tensor(), {1, 3, 3});
    expectDevice(output.tensor());
    expectDType(output.tensor());
    expectTensorNear(output.tensor(), ref.tensor(), std::max(atol_, 5e-2f));
}

TEST_P(Conv1dMultiDTypeTest, WithBias) {
    auto conv = nn::Conv1d(4, 8, 3, 1, 1, 1, 1, true);
    auto input_cpu = tenzor::randn({2, 4, 10}, DType::Float32, Device::cpu());
    auto ref = conv.forward(Variable(input_cpu, false));

    convert_model(conv);
    auto input = Variable(input_cpu.to(dtype_).to(device_), false);
    auto output = conv.forward(input);
    expectShape(output.tensor(), {2, 8, 10});
    expectTensorNear(output.tensor(), ref.tensor(), std::max(atol_, 5e-2f));
}

TEST_P(Conv1dMultiDTypeTest, WithStridePadding) {
    auto conv = nn::Conv1d(3, 6, 5, 2, 2);
    auto input_cpu = tenzor::randn({1, 3, 16}, DType::Float32, Device::cpu());
    auto ref = conv.forward(Variable(input_cpu, false));

    convert_model(conv);
    auto input = Variable(input_cpu.to(dtype_).to(device_), false);
    auto output = conv.forward(input);
    // L_out = floor((16 + 2*2 - 5) / 2) + 1 = floor(15/2) + 1 = 8
    expectShape(output.tensor(), {1, 6, 8});
    expectTensorNear(output.tensor(), ref.tensor(), std::max(atol_, 5e-2f));
}

TEST_P(Conv1dMultiDTypeTest, MultipleKernelSizes) {
    for (int64_t k : {1, 3, 5, 7}) {
        auto conv = nn::Conv1d(2, 4, k, 1, k / 2);
        auto input_cpu = tenzor::randn({1, 2, 16}, DType::Float32, Device::cpu());
        auto ref = conv.forward(Variable(input_cpu, false));

        convert_model(conv);
        auto input = Variable(input_cpu.to(dtype_).to(device_), false);
        auto output = conv.forward(input);
        EXPECT_EQ(output.tensor().shape()[0], 1);
        EXPECT_EQ(output.tensor().shape()[1], 4);
        expectTensorNear(output.tensor(), ref.tensor(), std::max(atol_, 5e-2f));
    }
}

// Backward — exercises Conv1d's autograd path, which dispatches
// OpId::Conv1dBackwardInput/Weight/Bias directly on 3-D operands (no
// unsqueeze-to-4D detour through Conv2dBackward*; see AUTOGRAD-R056). Every
// backend's Conv1dBackwardInput/Weight/Bias kernel is exercised here for
// real, including the weight/bias gradient VALUES (not just has_grad()),
// which is what actually catches a backend-specific wiring bug in these
// OpIds now that nn::Conv1d dispatches them in production. The
// optimizer-style sum().backward() pattern populates .grad() on the input
// tensor and the conv's parameters.
TEST_P(Conv1dMultiDTypeTest, BackwardProducesGradients) {
    // CPU reference: same conv config, same input, run on CPU.
    // Reseed before each construction so reference and device modules get
    // byte-identical weights (the fixture seeds once per test, so constructing
    // two modules sequentially would otherwise draw different init weights and
    // the forward/grad comparison would fail even for cpu/Float32).
    tenzor::manual_seed(42);
    auto conv_ref = nn::Conv1d(2, 4, 3, 1, 1, 1, 1, /*bias=*/true);
    auto input_cpu = tenzor::randn({1, 2, 8}, DType::Float32, Device::cpu());
    auto in_ref = Variable(input_cpu, /*requires_grad=*/true);
    auto out_ref = conv_ref.forward(in_ref);
    tenzor::sum(out_ref).backward();

    tenzor::manual_seed(42);
    auto conv = nn::Conv1d(2, 4, 3, 1, 1, 1, 1, /*bias=*/true);
    convert_model(conv);
    auto input = Variable(input_cpu.to(dtype_).to(device_), /*requires_grad=*/true);
    auto output = conv.forward(input);
    auto loss = tenzor::sum(output);
    loss.backward();

    ASSERT_TRUE(input.has_grad()) << "input gradient must be populated";
    auto g = input.grad().value();
    EXPECT_EQ(g.shape()[0], 1);
    EXPECT_EQ(g.shape()[1], 2);
    EXPECT_EQ(g.shape()[2], 8);
    expectDevice(g);

    // Forward output equals CPU reference (weights identical via per-test seed)
    expectTensorNear(output.tensor(), out_ref.tensor(), std::max(atol_, 5e-2f));
    // Input gradient matches CPU reference (exercises Conv1dBackwardInput)
    expectTensorNear(g, in_ref.grad().value(), std::max(atol_, 5e-2f));

    // Weight and bias gradients must both be populated AND numerically match
    // the CPU reference — exercises Conv1dBackwardWeight/Bias directly.
    // named_parameters() is insertion-ordered ("weight" then "bias" per the
    // Conv1d ctor), so conv/conv_ref line up positionally.
    auto ref_params = conv_ref.named_parameters();
    auto params = conv.named_parameters();
    ASSERT_EQ(params.size(), ref_params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        const auto& [name, p] = params[i];
        const auto& [ref_name, ref_p] = ref_params[i];
        ASSERT_EQ(name, ref_name);
        ASSERT_TRUE(p->has_grad()) << "parameter " << name << " missing grad";
        ASSERT_TRUE(ref_p->has_grad()) << "reference parameter " << name << " missing grad";
        SCOPED_TRACE("parameter " + name);
        expectTensorNear(p->grad().value(), ref_p->grad().value(),
                         std::max(atol_, 5e-2f));
    }
}

TEST_P(Conv1dMultiDTypeTest, BackwardWithStridePadding) {
    tenzor::manual_seed(43);
    auto conv_ref = nn::Conv1d(3, 6, 5, 2, 2);
    auto input_cpu = tenzor::randn({2, 3, 12}, DType::Float32, Device::cpu());
    auto in_ref = Variable(input_cpu, /*requires_grad=*/true);
    auto out_ref = conv_ref.forward(in_ref);
    tenzor::sum(out_ref).backward();

    tenzor::manual_seed(43);
    auto conv = nn::Conv1d(3, 6, 5, 2, 2);
    convert_model(conv);
    auto input = Variable(input_cpu.to(dtype_).to(device_), /*requires_grad=*/true);
    auto output = conv.forward(input);
    auto loss = tenzor::sum(output);
    loss.backward();

    ASSERT_TRUE(input.has_grad());
    EXPECT_EQ(input.grad().value().shape()[0], 2);
    EXPECT_EQ(input.grad().value().shape()[1], 3);
    EXPECT_EQ(input.grad().value().shape()[2], 12);
    expectTensorNear(output.tensor(), out_ref.tensor(), std::max(atol_, 5e-2f));
    expectTensorNear(input.grad().value(), in_ref.grad().value(),
                     std::max(atol_, 5e-2f));

    // Weight gradient (Conv1dBackwardWeight) with real stride/padding — this
    // is exactly the shape/padding combination that previously required the
    // nn-layer to manually pad + unsqueeze to 4D before dispatch.
    auto ref_params = conv_ref.named_parameters();
    auto params = conv.named_parameters();
    ASSERT_EQ(params.size(), ref_params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        const auto& [name, p] = params[i];
        const auto& [ref_name, ref_p] = ref_params[i];
        ASSERT_EQ(name, ref_name);
        ASSERT_TRUE(p->has_grad()) << "parameter " << name << " missing grad";
        SCOPED_TRACE("parameter " + name);
        expectTensorNear(p->grad().value(), ref_p->grad().value(),
                         std::max(atol_, 5e-2f));
    }
}

// Grouped convolution backward — Groups is threaded through
// OpId::Conv1dBackwardInput/Weight via AttrKey::Groups. groups=2 (with
// in_channels=4, out_channels=8) keeps forward on the "native 1-D conv"
// path (groups_ != in_channels_, so the CC.5 depthwise fast-path below is
// not eligible) and specifically stresses the multi-input-channel-per-group
// case of the native backward OpIds.
TEST_P(Conv1dMultiDTypeTest, BackwardGrouped) {
    tenzor::manual_seed(44);
    auto conv_ref = nn::Conv1d(4, 8, 3, 1, 1, 1, /*groups=*/2, /*bias=*/true);
    auto input_cpu = tenzor::randn({2, 4, 10}, DType::Float32, Device::cpu());
    auto in_ref = Variable(input_cpu, /*requires_grad=*/true);
    auto out_ref = conv_ref.forward(in_ref);
    tenzor::sum(out_ref).backward();

    tenzor::manual_seed(44);
    auto conv = nn::Conv1d(4, 8, 3, 1, 1, 1, /*groups=*/2, /*bias=*/true);
    convert_model(conv);
    auto input = Variable(input_cpu.to(dtype_).to(device_), /*requires_grad=*/true);
    auto output = conv.forward(input);
    auto loss = tenzor::sum(output);
    loss.backward();

    ASSERT_TRUE(input.has_grad());
    auto g = input.grad().value();
    EXPECT_EQ(g.shape()[0], 2);
    EXPECT_EQ(g.shape()[1], 4);
    EXPECT_EQ(g.shape()[2], 10);
    expectTensorNear(output.tensor(), out_ref.tensor(), std::max(atol_, 5e-2f));
    expectTensorNear(g, in_ref.grad().value(), std::max(atol_, 5e-2f));

    auto ref_params = conv_ref.named_parameters();
    auto params = conv.named_parameters();
    ASSERT_EQ(params.size(), ref_params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        const auto& [name, p] = params[i];
        const auto& [ref_name, ref_p] = ref_params[i];
        ASSERT_EQ(name, ref_name);
        ASSERT_TRUE(p->has_grad()) << "parameter " << name << " missing grad";
        SCOPED_TRACE("parameter " + name);
        expectTensorNear(p->grad().value(), ref_p->grad().value(),
                         std::max(atol_, 5e-2f));
    }
}

// Depthwise (groups == in_channels == out_channels, multiplier 1) backward —
// covers the CC.5 forward fast-path's interaction with the now-native
// backward: forward takes the OpId::DepthwiseConv1d fast path (when
// registered for the backend) while backward always goes through
// Conv1dBackward -> OpId::Conv1dBackwardInput/Weight/Bias regardless of
// which forward path produced the output. NOTE: does not use a channel
// multiplier != 1 (e.g. out_channels = 2 * in_channels with groups =
// in_channels) — cpu::depthwise_conv1d_kernel (src/backends/cpu/kernels/
// depthwise_conv1d.cpp) hard-assumes out_channels == in_channels == groups
// and silently produces a wrong-shaped/wrong-valued output otherwise; that
// is a pre-existing bug in the OpId::DepthwiseConv1d forward fast path
// (unrelated to OpId::Conv1dBackwardInput/Weight/Bias) and out of scope here.
TEST_P(Conv1dMultiDTypeTest, BackwardDepthwise) {
    tenzor::manual_seed(45);
    auto conv_ref = nn::Conv1d(4, 4, 3, 1, 1, 1, /*groups=*/4, /*bias=*/true);
    auto input_cpu = tenzor::randn({2, 4, 10}, DType::Float32, Device::cpu());
    auto in_ref = Variable(input_cpu, /*requires_grad=*/true);
    auto out_ref = conv_ref.forward(in_ref);
    tenzor::sum(out_ref).backward();

    tenzor::manual_seed(45);
    auto conv = nn::Conv1d(4, 4, 3, 1, 1, 1, /*groups=*/4, /*bias=*/true);
    convert_model(conv);
    auto input = Variable(input_cpu.to(dtype_).to(device_), /*requires_grad=*/true);
    auto output = conv.forward(input);
    auto loss = tenzor::sum(output);
    loss.backward();

    ASSERT_TRUE(input.has_grad());
    auto g = input.grad().value();
    EXPECT_EQ(g.shape()[0], 2);
    EXPECT_EQ(g.shape()[1], 4);
    EXPECT_EQ(g.shape()[2], 10);
    expectTensorNear(output.tensor(), out_ref.tensor(), std::max(atol_, 5e-2f));
    expectTensorNear(g, in_ref.grad().value(), std::max(atol_, 5e-2f));

    auto ref_params = conv_ref.named_parameters();
    auto params = conv.named_parameters();
    ASSERT_EQ(params.size(), ref_params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        const auto& [name, p] = params[i];
        const auto& [ref_name, ref_p] = ref_params[i];
        ASSERT_EQ(name, ref_name);
        ASSERT_TRUE(p->has_grad()) << "parameter " << name << " missing grad";
        SCOPED_TRACE("parameter " + name);
        expectTensorNear(p->grad().value(), ref_p->grad().value(),
                         std::max(atol_, 5e-2f));
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(Conv1dMultiDTypeTest);
