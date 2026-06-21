#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <cmath>
#include "../../grad_flow_helpers.hpp"  // W.26: EXPECT_GRAD_FLOWS
#include "../../backend_test_fixture.hpp"

using namespace tenzor;

// One fixture per suite (BackendTest provides the per-test `device` member).
class AdaptiveAvgPool2dTest : public ::tenzor::testing::BackendTest {};
class AvgPool2dTest : public ::tenzor::testing::BackendTest {};
class MaxPool2dTest : public ::tenzor::testing::BackendTest {};
class PoolingTest : public ::tenzor::testing::BackendTest {};

// Helper function to check if two tensors are close.
// Tensors may live on any backend device; pull both to CPU before reading.
bool tensors_close(const Tensor& a, const Tensor& b, float rtol = 1e-5f, float atol = 1e-7f) {
    auto a_shape = a.shape();
    auto b_shape = b.shape();
    if (a_shape.size() != b_shape.size() ||
        !std::equal(a_shape.begin(), a_shape.end(), b_shape.begin())) {
        return false;
    }

    auto a_cpu = a.cpu();
    auto b_cpu = b.cpu();
    const float* a_data = a_cpu.data<float>();
    const float* b_data = b_cpu.data<float>();
    size_t numel = a_cpu.numel();

    for (size_t i = 0; i < numel; ++i) {
        float diff = std::abs(a_data[i] - b_data[i]);
        float threshold = atol + rtol * std::abs(b_data[i]);
        if (diff > threshold) {
            return false;
        }
    }
    return true;
}

// Helper function to compute numerical gradient.
// `input` may live on a backend device; perturbation is applied on a CPU copy
// and pushed back to the device for each forward evaluation.
Tensor numerical_gradient(std::function<Variable(Variable&)> func,
                         Variable& input, const tenzor::Device& device, float eps = 1e-4f) {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    auto grad = zeros(shape_vec);  // host-side accumulator
    float* grad_data = grad.data<float>();

    // Keep a CPU mirror of the input so we can perturb individual elements.
    auto input_cpu = input.tensor().cpu();
    float* input_data = input_cpu.data<float>();
    size_t numel = input_cpu.numel();

    for (size_t i = 0; i < numel; ++i) {
        float original = input_data[i];

        // f(x + eps)
        input_data[i] = original + eps;
        input = Variable(input_cpu.to(device), true);
        auto out_plus = func(input);
        float loss_plus = sum(out_plus.tensor()).cpu().data<float>()[0];

        // f(x - eps)
        input_data[i] = original - eps;
        input = Variable(input_cpu.to(device), true);
        auto out_minus = func(input);
        float loss_minus = sum(out_minus.tensor()).cpu().data<float>()[0];

        // Restore original
        input_data[i] = original;

        // Central difference
        grad_data[i] = (loss_plus - loss_minus) / (2.0f * eps);
    }

    // Restore the input Variable to its unperturbed state on the device.
    input = Variable(input_cpu.to(device), true);

    return grad;
}

// ============================
// MaxPool2d Basic Shape Tests
// ============================

TEST_P(MaxPool2dTest, ForwardShapeBasic) {
    auto pool = nn::MaxPool2d(2, 2, 0);
    pool.to(device);
    auto input = Variable(randn({2, 3, 32, 32}, DType::Float32, device), true);
    auto output = pool.forward(input);

    // Output size = (32 - 2) / 2 + 1 = 16
    EXPECT_EQ(output.shape()[0], 2);   // batch
    EXPECT_EQ(output.shape()[1], 3);   // channels
    EXPECT_EQ(output.shape()[2], 16);  // height
    EXPECT_EQ(output.shape()[3], 16);  // width
}

TEST_P(MaxPool2dTest, ForwardShapeSingleBatch) {
    auto pool = nn::MaxPool2d(2);
    pool.to(device);
    auto input = Variable(randn({1, 1, 28, 28}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 1);
    EXPECT_EQ(output.shape()[2], 14);  // 28 / 2
    EXPECT_EQ(output.shape()[3], 14);
}

TEST_P(MaxPool2dTest, ForwardShapeMultiBatch) {
    auto pool = nn::MaxPool2d(3, 3, 0);
    pool.to(device);
    auto input = Variable(randn({32, 16, 64, 64}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[0], 32);
    EXPECT_EQ(output.shape()[1], 16);
    EXPECT_EQ(output.shape()[2], 21);  // (64 - 3) / 3 + 1
    EXPECT_EQ(output.shape()[3], 21);
}

TEST_P(MaxPool2dTest, KernelSize2) {
    auto pool = nn::MaxPool2d(2);
    pool.to(device);
    auto input = Variable(randn({4, 8, 32, 32}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 16);
    EXPECT_EQ(output.shape()[3], 16);
}

TEST_P(MaxPool2dTest, KernelSize3) {
    auto pool = nn::MaxPool2d(3);
    pool.to(device);
    auto input = Variable(randn({2, 3, 33, 33}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 11);  // 33 / 3
    EXPECT_EQ(output.shape()[3], 11);
}

TEST_P(MaxPool2dTest, KernelSize4) {
    auto pool = nn::MaxPool2d(4);
    pool.to(device);
    auto input = Variable(randn({1, 1, 32, 32}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 8);  // 32 / 4
    EXPECT_EQ(output.shape()[3], 8);
}

// ============================
// MaxPool2d Stride Tests
// ============================

TEST_P(MaxPool2dTest, StrideDefault) {
    auto pool = nn::MaxPool2d(3);  // stride defaults to kernel_size
    pool.to(device);
    auto input = Variable(randn({2, 4, 33, 33}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 11);  // (33 - 3) / 3 + 1
    EXPECT_EQ(output.shape()[3], 11);
}

TEST_P(MaxPool2dTest, Stride1) {
    auto pool = nn::MaxPool2d(2, 1);  // kernel=2, stride=1
    pool.to(device);
    auto input = Variable(randn({2, 3, 32, 32}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 31);  // (32 - 2) / 1 + 1
    EXPECT_EQ(output.shape()[3], 31);
}

TEST_P(MaxPool2dTest, Stride2) {
    auto pool = nn::MaxPool2d(3, 2);  // kernel=3, stride=2
    pool.to(device);
    auto input = Variable(randn({2, 3, 32, 32}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 15);  // (32 - 3) / 2 + 1
    EXPECT_EQ(output.shape()[3], 15);
}

// ============================
// MaxPool2d Padding Tests
// ============================

TEST_P(MaxPool2dTest, Padding0) {
    auto pool = nn::MaxPool2d(2, 2, 0);
    pool.to(device);
    auto input = Variable(randn({2, 3, 32, 32}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 16);
    EXPECT_EQ(output.shape()[3], 16);
}

TEST_P(MaxPool2dTest, Padding1) {
    auto pool = nn::MaxPool2d(3, 2, 1);
    pool.to(device);
    auto input = Variable(randn({2, 3, 32, 32}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 16);  // (32 + 2*1 - 3) / 2 + 1
    EXPECT_EQ(output.shape()[3], 16);
}

// ============================
// MaxPool2d Functional Tests
// ============================

TEST_P(MaxPool2dTest, MaxValueSelection) {
    // Test that max value is correctly selected
    auto pool = nn::MaxPool2d(2, 2, 0);
    pool.to(device);

    // Create a simple input where we know the max values (host write -> device)
    auto input_tensor = zeros({1, 1, 4, 4});
    float* data = input_tensor.data<float>();

    // Set specific values to verify max selection
    data[0] = 1.0f;  data[1] = 2.0f;  data[2] = 3.0f;  data[3] = 4.0f;
    data[4] = 5.0f;  data[5] = 6.0f;  data[6] = 7.0f;  data[7] = 8.0f;
    data[8] = 9.0f;  data[9] = 10.0f; data[10] = 11.0f; data[11] = 12.0f;
    data[12] = 13.0f; data[13] = 14.0f; data[14] = 15.0f; data[15] = 16.0f;

    auto input = Variable(input_tensor.to(device), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 2);
    EXPECT_EQ(output.shape()[3], 2);

    auto out_cpu = output.tensor().cpu();
    const float* out_data = out_cpu.data<float>();
    EXPECT_FLOAT_EQ(out_data[0], 6.0f);   // max of [1,2,5,6]
    EXPECT_FLOAT_EQ(out_data[1], 8.0f);   // max of [3,4,7,8]
    EXPECT_FLOAT_EQ(out_data[2], 14.0f);  // max of [9,10,13,14]
    EXPECT_FLOAT_EQ(out_data[3], 16.0f);  // max of [11,12,15,16]
}

TEST_P(MaxPool2dTest, RequiresGrad) {
    auto pool = nn::MaxPool2d(2);
    pool.to(device);
    auto input = Variable(randn({2, 3, 16, 16}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_TRUE(output.requires_grad());
}

TEST_P(MaxPool2dTest, NoGradWhenInputNoGrad) {
    auto pool = nn::MaxPool2d(2);
    pool.to(device);
    auto input = Variable(randn({2, 3, 16, 16}, DType::Float32, device), false);
    auto output = pool.forward(input);

    EXPECT_FALSE(output.requires_grad());
}

// ============================
// MaxPool2d Gradient Tests
// ============================

TEST_P(MaxPool2dTest, BackwardPassExecutes) {
    auto pool = nn::MaxPool2d(2);
    pool.to(device);
    auto input = Variable(randn({2, 3, 16, 16}, DType::Float32, device), true);
    auto output = pool.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = ones(out_shape_vec, DType::Float32, device);

    EXPECT_NO_THROW({
        output.backward(grad_output);
    });
    EXPECT_GRAD_FLOWS(input);  // W.26
}

TEST_P(MaxPool2dTest, GradientNonZero) {
    auto pool = nn::MaxPool2d(2);
    pool.to(device);
    auto input = Variable(randn({2, 3, 16, 16}, DType::Float32, device), true);
    auto output = pool.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    output.backward(ones(out_shape_vec, DType::Float32, device));

    EXPECT_TRUE(input.grad().has_value());

    auto grad_cpu = input.grad()->cpu();
    auto grad_data = grad_cpu.data<float>();
    bool has_nonzero = false;
    size_t numel = static_cast<size_t>(grad_cpu.numel());
    for (size_t i = 0; i < numel; ++i) {
        if (std::abs(grad_data[i]) > 1e-6f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST_P(MaxPool2dTest, GradientCheckSmall) {
    auto pool = nn::MaxPool2d(2);
    pool.to(device);
    auto input = Variable(randn({1, 1, 4, 4}, DType::Float32, device), true);

    // loss = sum(pool(x)) so d_loss/d_out is identically 1; the numerical path
    // only reads .tensor(), so the severed-grad_fn rewrap is harmless there.
    auto loss_fn = [&pool](Variable& inp) -> Variable {
        auto out = pool.forward(inp);
        auto loss_tensor = sum(out.tensor());
        return Variable(loss_tensor, true);
    };

    auto numerical_grad = numerical_gradient(loss_fn, input, device, 1e-3f);

    // Analytic path: backprop through the INTACT Variable graph (no .tensor()
    // rewrap). Seed with ones because d(sum(out))/d(out) == 1 everywhere.
    auto output = pool.forward(input);
    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    output.backward(ones(out_shape_vec, DType::Float32, device));

    ASSERT_TRUE(input.grad().has_value())
        << "MaxPool2d backward produced no gradient for input";
    EXPECT_TRUE(tensors_close(*input.grad(), numerical_grad, 1e-3f, 1e-3f));
}

// ============================
// AvgPool2d Basic Shape Tests
// ============================

TEST_P(AvgPool2dTest, ForwardShapeBasic) {
    auto pool = nn::AvgPool2d(2, 2, 0);
    pool.to(device);
    auto input = Variable(randn({2, 3, 32, 32}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 16);
    EXPECT_EQ(output.shape()[3], 16);
}

TEST_P(AvgPool2dTest, ForwardShapeSingleBatch) {
    auto pool = nn::AvgPool2d(3);
    pool.to(device);
    auto input = Variable(randn({1, 1, 27, 27}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 1);
    EXPECT_EQ(output.shape()[2], 9);  // 27 / 3
    EXPECT_EQ(output.shape()[3], 9);
}

TEST_P(AvgPool2dTest, ForwardShapeMultiBatch) {
    auto pool = nn::AvgPool2d(2, 2, 0);
    pool.to(device);
    auto input = Variable(randn({16, 32, 56, 56}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[0], 16);
    EXPECT_EQ(output.shape()[1], 32);
    EXPECT_EQ(output.shape()[2], 28);
    EXPECT_EQ(output.shape()[3], 28);
}

TEST_P(AvgPool2dTest, KernelSize2) {
    auto pool = nn::AvgPool2d(2);
    pool.to(device);
    auto input = Variable(randn({4, 8, 32, 32}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 16);
    EXPECT_EQ(output.shape()[3], 16);
}

TEST_P(AvgPool2dTest, KernelSize3) {
    auto pool = nn::AvgPool2d(3);
    pool.to(device);
    auto input = Variable(randn({2, 3, 30, 30}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 10);
    EXPECT_EQ(output.shape()[3], 10);
}

// ============================
// AvgPool2d Stride Tests
// ============================

TEST_P(AvgPool2dTest, StrideDefault) {
    auto pool = nn::AvgPool2d(4);
    pool.to(device);
    auto input = Variable(randn({2, 4, 32, 32}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 8);
    EXPECT_EQ(output.shape()[3], 8);
}

TEST_P(AvgPool2dTest, Stride1) {
    auto pool = nn::AvgPool2d(2, 1);
    pool.to(device);
    auto input = Variable(randn({2, 3, 32, 32}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 31);  // (32 - 2) / 1 + 1
    EXPECT_EQ(output.shape()[3], 31);
}

TEST_P(AvgPool2dTest, Stride2) {
    auto pool = nn::AvgPool2d(3, 2);
    pool.to(device);
    auto input = Variable(randn({2, 3, 32, 32}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 15);
    EXPECT_EQ(output.shape()[3], 15);
}

// ============================
// AvgPool2d Functional Tests
// ============================

TEST_P(AvgPool2dTest, AverageValueComputation) {
    auto pool = nn::AvgPool2d(2, 2, 0);
    pool.to(device);

    auto input_tensor = zeros({1, 1, 4, 4});
    float* data = input_tensor.data<float>();

    // Set known values
    for (int i = 0; i < 16; ++i) {
        data[i] = static_cast<float>(i + 1);
    }

    auto input = Variable(input_tensor.to(device), true);
    auto output = pool.forward(input);

    auto out_cpu = output.tensor().cpu();
    const float* out_data = out_cpu.data<float>();
    EXPECT_FLOAT_EQ(out_data[0], 3.5f);   // avg of [1,2,5,6]
    EXPECT_FLOAT_EQ(out_data[1], 5.5f);   // avg of [3,4,7,8]
    EXPECT_FLOAT_EQ(out_data[2], 11.5f);  // avg of [9,10,13,14]
    EXPECT_FLOAT_EQ(out_data[3], 13.5f);  // avg of [11,12,15,16]
}

TEST_P(AvgPool2dTest, RequiresGrad) {
    auto pool = nn::AvgPool2d(2);
    pool.to(device);
    auto input = Variable(randn({2, 3, 16, 16}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_TRUE(output.requires_grad());
}

TEST_P(AvgPool2dTest, NoGradWhenInputNoGrad) {
    auto pool = nn::AvgPool2d(2);
    pool.to(device);
    auto input = Variable(randn({2, 3, 16, 16}, DType::Float32, device), false);
    auto output = pool.forward(input);

    EXPECT_FALSE(output.requires_grad());
}

// ============================
// AvgPool2d Gradient Tests
// ============================

TEST_P(AvgPool2dTest, BackwardPassExecutes) {
    auto pool = nn::AvgPool2d(2);
    pool.to(device);
    auto input = Variable(randn({2, 3, 16, 16}, DType::Float32, device), true);
    auto output = pool.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = ones(out_shape_vec, DType::Float32, device);

    EXPECT_NO_THROW({
        output.backward(grad_output);
    });
    EXPECT_GRAD_FLOWS(input);  // W.26
}

TEST_P(AvgPool2dTest, GradientNonZero) {
    auto pool = nn::AvgPool2d(2);
    pool.to(device);
    auto input = Variable(randn({2, 3, 16, 16}, DType::Float32, device), true);
    auto output = pool.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    output.backward(ones(out_shape_vec, DType::Float32, device));

    EXPECT_TRUE(input.grad().has_value());

    auto grad_cpu = input.grad()->cpu();
    auto grad_data = grad_cpu.data<float>();
    bool has_nonzero = false;
    size_t numel = static_cast<size_t>(grad_cpu.numel());
    for (size_t i = 0; i < numel; ++i) {
        if (std::abs(grad_data[i]) > 1e-6f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST_P(AvgPool2dTest, GradientCheckSmall) {
    auto pool = nn::AvgPool2d(2);
    pool.to(device);
    auto input = Variable(randn({1, 1, 4, 4}, DType::Float32, device), true);

    auto loss_fn = [&pool](Variable& inp) -> Variable {
        auto out = pool.forward(inp);
        auto loss_tensor = sum(out.tensor());
        return Variable(loss_tensor, true);
    };

    auto numerical_grad = numerical_gradient(loss_fn, input, device, 1e-3f);

    // Analytic path: backprop through the INTACT Variable graph (no .tensor()
    // rewrap). Seed with ones because d(sum(out))/d(out) == 1 everywhere.
    auto output = pool.forward(input);
    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    output.backward(ones(out_shape_vec, DType::Float32, device));

    ASSERT_TRUE(input.grad().has_value())
        << "AvgPool2d backward produced no gradient for input";
    EXPECT_TRUE(tensors_close(*input.grad(), numerical_grad, 1e-3f, 1e-3f));
}

// ============================
// AdaptiveAvgPool2d Tests
// ============================

TEST_P(AdaptiveAvgPool2dTest, ForwardShapeBasic) {
    auto pool = nn::AdaptiveAvgPool2d(7, 7);
    pool.to(device);
    auto input = Variable(randn({2, 3, 32, 32}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 7);
    EXPECT_EQ(output.shape()[3], 7);
}

TEST_P(AdaptiveAvgPool2dTest, ForwardShapeSquare) {
    auto pool = nn::AdaptiveAvgPool2d(1);  // Global average pooling
    pool.to(device);
    auto input = Variable(randn({4, 512, 7, 7}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 512);
    EXPECT_EQ(output.shape()[2], 1);
    EXPECT_EQ(output.shape()[3], 1);
}

TEST_P(AdaptiveAvgPool2dTest, ForwardShapeRectangular) {
    auto pool = nn::AdaptiveAvgPool2d(5, 3);
    pool.to(device);
    auto input = Variable(randn({2, 16, 28, 28}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 16);
    EXPECT_EQ(output.shape()[2], 5);
    EXPECT_EQ(output.shape()[3], 3);
}

TEST_P(AdaptiveAvgPool2dTest, GlobalAveragePooling) {
    auto pool = nn::AdaptiveAvgPool2d(1, 1);
    pool.to(device);
    auto input = Variable(randn({2, 64, 14, 14}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 1);
    EXPECT_EQ(output.shape()[3], 1);
}

TEST_P(AdaptiveAvgPool2dTest, OutputLargerThanInput) {
    // Output can be larger than input (upsampling)
    auto pool = nn::AdaptiveAvgPool2d(8, 8);
    pool.to(device);
    auto input = Variable(randn({1, 3, 4, 4}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 8);
    EXPECT_EQ(output.shape()[3], 8);
}

TEST_P(AdaptiveAvgPool2dTest, RequiresGrad) {
    auto pool = nn::AdaptiveAvgPool2d(7);
    pool.to(device);
    auto input = Variable(randn({2, 3, 32, 32}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_TRUE(output.requires_grad());
}

TEST_P(AdaptiveAvgPool2dTest, NoGradWhenInputNoGrad) {
    auto pool = nn::AdaptiveAvgPool2d(7);
    pool.to(device);
    auto input = Variable(randn({2, 3, 32, 32}, DType::Float32, device), false);
    auto output = pool.forward(input);

    EXPECT_FALSE(output.requires_grad());
}

TEST_P(AdaptiveAvgPool2dTest, BackwardPassExecutes) {
    auto pool = nn::AdaptiveAvgPool2d(7);
    pool.to(device);
    auto input = Variable(randn({2, 3, 32, 32}, DType::Float32, device), true);
    auto output = pool.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = ones(out_shape_vec, DType::Float32, device);

    EXPECT_NO_THROW({
        output.backward(grad_output);
    });
    EXPECT_GRAD_FLOWS(input);  // W.26
}

TEST_P(AdaptiveAvgPool2dTest, GradientNonZero) {
    auto pool = nn::AdaptiveAvgPool2d(4);
    pool.to(device);
    auto input = Variable(randn({2, 3, 16, 16}, DType::Float32, device), true);
    auto output = pool.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    output.backward(ones(out_shape_vec, DType::Float32, device));

    EXPECT_TRUE(input.grad().has_value());

    auto grad_cpu = input.grad()->cpu();
    auto grad_data = grad_cpu.data<float>();
    bool has_nonzero = false;
    size_t numel = static_cast<size_t>(grad_cpu.numel());
    for (size_t i = 0; i < numel; ++i) {
        if (std::abs(grad_data[i]) > 1e-6f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST_P(AdaptiveAvgPool2dTest, GradientCheckSmall) {
    auto pool = nn::AdaptiveAvgPool2d(2);
    pool.to(device);
    auto input = Variable(randn({1, 1, 4, 4}, DType::Float32, device), true);

    auto loss_fn = [&pool](Variable& inp) -> Variable {
        auto out = pool.forward(inp);
        auto loss_tensor = sum(out.tensor());
        return Variable(loss_tensor, true);
    };

    auto numerical_grad = numerical_gradient(loss_fn, input, device, 1e-3f);

    // Analytic path: backprop through the INTACT Variable graph (no .tensor()
    // rewrap). Seed with ones because d(sum(out))/d(out) == 1 everywhere.
    auto output = pool.forward(input);
    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    output.backward(ones(out_shape_vec, DType::Float32, device));

    ASSERT_TRUE(input.grad().has_value())
        << "AdaptiveAvgPool2d backward produced no gradient for input";
    EXPECT_TRUE(tensors_close(*input.grad(), numerical_grad, 1e-3f, 1e-3f));
}

// ============================
// Edge Cases
// ============================

TEST_P(PoolingTest, MaxPoolEdgeCase1x1) {
    auto pool = nn::MaxPool2d(1);
    pool.to(device);
    auto input = Variable(randn({2, 3, 16, 16}, DType::Float32, device), true);
    auto output = pool.forward(input);

    // 1x1 pooling should not change dimensions
    EXPECT_EQ(output.shape()[2], 16);
    EXPECT_EQ(output.shape()[3], 16);
}

TEST_P(PoolingTest, AvgPoolEdgeCase1x1) {
    auto pool = nn::AvgPool2d(1);
    pool.to(device);
    auto input = Variable(randn({2, 3, 16, 16}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 16);
    EXPECT_EQ(output.shape()[3], 16);
}

TEST_P(PoolingTest, AdaptivePoolSameSize) {
    // Adaptive pool to same size as input
    auto pool = nn::AdaptiveAvgPool2d(16, 16);
    pool.to(device);
    auto input = Variable(randn({2, 3, 16, 16}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 16);
    EXPECT_EQ(output.shape()[3], 16);
}

TEST_P(PoolingTest, LargeBatch) {
    auto pool = nn::MaxPool2d(2);
    pool.to(device);
    auto input = Variable(randn({128, 3, 32, 32}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[0], 128);
}

TEST_P(PoolingTest, ManyChannels) {
    auto pool = nn::AvgPool2d(2);
    pool.to(device);
    auto input = Variable(randn({2, 512, 7, 7}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[1], 512);
}

// ============================
// Error Handling
// ============================

TEST_P(PoolingTest, InvalidInputDimensions) {
    auto pool = nn::MaxPool2d(2);
    pool.to(device);
    auto input_3d = Variable(randn({2, 3, 16}, DType::Float32, device), true);

    EXPECT_THROW({
        pool.forward(input_3d);
    }, std::invalid_argument);
}

// ============================
// Integration Tests
// ============================

TEST_P(PoolingTest, SequentialPooling) {
    // Test multiple pooling layers in sequence
    auto pool1 = nn::MaxPool2d(2);
    pool1.to(device);
    auto pool2 = nn::MaxPool2d(2);
    pool2.to(device);

    auto input = Variable(randn({2, 3, 64, 64}, DType::Float32, device), true);
    auto x = pool1.forward(input);
    auto output = pool2.forward(x);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 16);  // 64 -> 32 -> 16
    EXPECT_EQ(output.shape()[3], 16);
}

TEST_P(PoolingTest, MixedPoolingTypes) {
    auto max_pool = nn::MaxPool2d(2);
    max_pool.to(device);
    auto avg_pool = nn::AvgPool2d(2);
    avg_pool.to(device);

    auto input = Variable(randn({2, 3, 32, 32}, DType::Float32, device), true);
    auto x1 = max_pool.forward(input);
    auto x2 = avg_pool.forward(input);

    // Both should have same output shape
    auto shape1 = x1.shape();
    auto shape2 = x2.shape();
    EXPECT_TRUE(std::equal(shape1.begin(), shape1.end(), shape2.begin()));
}

TEST_P(PoolingTest, PoolingWithConvolution) {
    // Common pattern: Conv -> Pool
    auto conv = nn::Conv2d(3, 16, 3, 1, 1);
    conv.to(device);
    auto pool = nn::MaxPool2d(2);
    pool.to(device);

    auto input = Variable(randn({2, 3, 32, 32}, DType::Float32, device), true);
    auto x = conv.forward(input);
    auto output = pool.forward(x);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 16);
    EXPECT_EQ(output.shape()[2], 16);  // 32 -> 16 after pooling
    EXPECT_EQ(output.shape()[3], 16);
}

TEST_P(PoolingTest, ResNetStyleBottleneck) {
    // Test ResNet-style block with pooling
    auto conv1 = nn::Conv2d(64, 128, 3, 1, 1);
    conv1.to(device);
    auto pool = nn::MaxPool2d(2);
    pool.to(device);
    auto conv2 = nn::Conv2d(128, 128, 3, 1, 1);
    conv2.to(device);

    auto input = Variable(randn({2, 64, 56, 56}, DType::Float32, device), true);
    auto x = conv1.forward(input);
    x = pool.forward(x);
    auto output = conv2.forward(x);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 128);
    EXPECT_EQ(output.shape()[2], 28);  // 56 / 2
    EXPECT_EQ(output.shape()[3], 28);
}

INSTANTIATE_BACKEND_TESTS(MaxPool2dTest);
INSTANTIATE_BACKEND_TESTS(AvgPool2dTest);
INSTANTIATE_BACKEND_TESTS(AdaptiveAvgPool2dTest);
INSTANTIATE_BACKEND_TESTS(PoolingTest);
