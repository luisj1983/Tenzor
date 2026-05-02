/**
 * @file test_deformable_conv2d_backward_parity.cpp
 * @brief Cross-backend parity tests for DeformableConv2d forward + backward.
 *
 * Covers OpIds 184-187: DeformableConv2dForward, DeformableConv2dBackwardInput,
 * DeformableConv2dBackwardWeight, DeformableConv2dBackwardBias. Per the audit
 * (2026-05-02), the three backward OpIds had zero parity-test references
 * before this file even though they were registered on every non-MPS backend.
 *
 * Strategy: build the layer on CPU, copy weights to each backend, run a
 * forward pass and a fixed-grad backward pass, then compare:
 *   - forward output (parity test_operation_parity_single)
 *   - input.grad, weight.grad, bias.grad after backward (manual diff)
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/conv.hpp>
#include <tenzor/autograd/ops.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class DeformableConv2dParity : public BackendTest {};

namespace {

// Returns the deformable-conv2d output tensor for the given inputs and
// fixed layer hyperparameters. Used by the parity helper which invokes
// the lambda on CPU and the target device with the inputs already on
// each device.
Tensor deformable_conv2d_forward_lambda(const std::vector<Tensor>& inputs) {
    nn::DeformableConv2d layer(/*in_channels=*/4, /*out_channels=*/8,
                               /*kernel_size=*/3, /*stride=*/1,
                               /*padding=*/1, /*dilation=*/1,
                               /*groups=*/1, /*offset_groups=*/1,
                               /*bias=*/true);
    // Overwrite the layer's randomly-initialised parameters with the test's
    // fixed weight (inputs[3]) and bias (inputs[4]) so CPU and target
    // arithmetic agree exactly. get_parameter(name) returns the shared
    // ptr that the Module stores internally, so writing through it
    // updates the layer in place.
    *layer.get_parameter("weight") = Variable(inputs[3], true);
    *layer.get_parameter("bias")   = Variable(inputs[4], true);

    auto out = layer.forward(Variable(inputs[0], false),  // input
                             Variable(inputs[1], false),  // offset
                             Variable(inputs[2], false)); // mask
    return out.tensor();
}

}  // namespace

// ----------------------------------------------------------------------------
// Forward parity
// ----------------------------------------------------------------------------

TEST_P(DeformableConv2dParity, Forward_NoMask) {
    const int64_t N = 2, C_in = 4, C_out = 8, H = 8, W = 8, K = 3;
    const int64_t offset_channels = 2 * K * K;  // dy, dx per kernel pos

    auto x      = randn({N, C_in,  H, W}, DType::Float32, Device::cpu());
    auto offset = randn({N, offset_channels, H, W}, DType::Float32, Device::cpu()) * 0.1f;
    auto mask   = zeros({0},                   DType::Float32, Device::cpu());  // no mask
    auto weight = randn({C_out, C_in, K, K},  DType::Float32, Device::cpu());
    auto bias   = randn({C_out},               DType::Float32, Device::cpu());

    std::vector<Tensor> inputs;
    inputs.push_back(x);
    inputs.push_back(offset);
    inputs.push_back(mask);
    inputs.push_back(weight);
    inputs.push_back(bias);

    test_operation_parity_single(deformable_conv2d_forward_lambda,
                                 inputs, device, 1e-4f, 1e-5f,
                                 "DeformableConv2d_Forward_NoMask");
}

TEST_P(DeformableConv2dParity, Forward_WithMask) {
    const int64_t N = 1, C_in = 4, C_out = 8, H = 6, W = 6, K = 3;
    const int64_t offset_channels = 2 * K * K;
    const int64_t mask_channels   = K * K;

    auto x      = randn({N, C_in,  H, W}, DType::Float32, Device::cpu());
    auto offset = randn({N, offset_channels, H, W}, DType::Float32, Device::cpu()) * 0.1f;
    // Modulation mask values in (0, 1) — sigmoid of small randn.
    auto mask   = sigmoid(randn({N, mask_channels, H, W}, DType::Float32, Device::cpu()));
    auto weight = randn({C_out, C_in, K, K}, DType::Float32, Device::cpu());
    auto bias   = randn({C_out},              DType::Float32, Device::cpu());

    std::vector<Tensor> inputs;
    inputs.push_back(x);
    inputs.push_back(offset);
    inputs.push_back(mask);
    inputs.push_back(weight);
    inputs.push_back(bias);

    test_operation_parity_single(deformable_conv2d_forward_lambda,
                                 inputs, device, 1e-4f, 1e-5f,
                                 "DeformableConv2d_Forward_WithMask");
}

// ----------------------------------------------------------------------------
// Backward parity — exercises DeformableConv2dBackwardInput / Weight / Bias
// ----------------------------------------------------------------------------

TEST_P(DeformableConv2dParity, Backward_GradientsMatchCPU) {
    const int64_t N = 2, C_in = 4, C_out = 8, H = 6, W = 6, K = 3;
    const int64_t offset_channels = 2 * K * K;

    // Fixed CPU inputs / parameters / grad-out so CPU and target see the
    // same arithmetic.
    auto x_cpu      = randn({N, C_in,  H, W}, DType::Float32, Device::cpu());
    auto offset_cpu = randn({N, offset_channels, H, W}, DType::Float32, Device::cpu()) * 0.1f;
    auto mask_cpu   = zeros({0}, DType::Float32, Device::cpu());
    auto weight_cpu = randn({C_out, C_in, K, K}, DType::Float32, Device::cpu());
    auto bias_cpu   = randn({C_out}, DType::Float32, Device::cpu());

    auto run_one = [&](Device target,
                       Tensor& out_grad_input,
                       Tensor& out_grad_weight,
                       Tensor& out_grad_bias) {
        nn::DeformableConv2d layer(C_in, C_out, K, 1, 1, 1, 1, 1, true);
        *layer.get_parameter("weight") = Variable(weight_cpu.to(target), true);
        *layer.get_parameter("bias")   = Variable(bias_cpu.to(target),   true);
        Variable x_v(x_cpu.to(target), true);
        Variable o_v(offset_cpu.to(target), false);
        Variable m_v(mask_cpu.to(target), false);

        auto y = layer.forward(x_v, o_v, m_v);
        // Use a deterministic gradient (sum === ones) so backward parity
        // is purely a function of the kernel implementations.
        auto loss = sum(y);
        loss.backward();

        target.synchronize();
        out_grad_input  = (*x_v.grad()).to(Device::cpu());
        out_grad_weight = (*layer.get_parameter("weight")->grad()).to(Device::cpu());
        out_grad_bias   = (*layer.get_parameter("bias")->grad()).to(Device::cpu());
    };

    Tensor cpu_grad_input, cpu_grad_weight, cpu_grad_bias;
    run_one(Device::cpu(), cpu_grad_input, cpu_grad_weight, cpu_grad_bias);

    if (device.type == Device::Type::CPU) return;

    Tensor dev_grad_input, dev_grad_weight, dev_grad_bias;
    run_one(device, dev_grad_input, dev_grad_weight, dev_grad_bias);

    EXPECT_LT(max_abs_diff(cpu_grad_input, dev_grad_input), 1e-3f)
        << "grad_input mismatch on " << backend_name(device);
    EXPECT_LT(max_abs_diff(cpu_grad_weight, dev_grad_weight), 1e-3f)
        << "grad_weight mismatch on " << backend_name(device);
    EXPECT_LT(max_abs_diff(cpu_grad_bias, dev_grad_bias), 1e-3f)
        << "grad_bias mismatch on " << backend_name(device);
}

INSTANTIATE_BACKEND_TESTS(DeformableConv2dParity);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    try {
        if (!::testing::GTEST_FLAG(list_tests)) {
            tenzor::initialize();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }

    int result = RUN_ALL_TESTS();

    try {
        tenzor::finalize();
    } catch (...) {}

    return result;
}
