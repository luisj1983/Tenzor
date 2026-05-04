#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <cmath>

using namespace tenzor;

// Global test environment that initializes Tenzor before tests
class PoolingTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

// Register the environment
static ::testing::Environment* const pooling_env =
    ::testing::AddGlobalTestEnvironment(new PoolingTestEnvironment);

// Helper function to check if two tensors are close
bool tensors_close(const Tensor& a, const Tensor& b, float rtol = 1e-5f, float atol = 1e-7f) {
    auto a_shape = a.shape();
    auto b_shape = b.shape();
    if (a_shape.size() != b_shape.size() ||
        !std::equal(a_shape.begin(), a_shape.end(), b_shape.begin())) {
        return false;
    }

    const float* a_data = a.data<float>();
    const float* b_data = b.data<float>();
    size_t numel = a.numel();

    for (size_t i = 0; i < numel; ++i) {
        float diff = std::abs(a_data[i] - b_data[i]);
        float threshold = atol + rtol * std::abs(b_data[i]);
        if (diff > threshold) {
            return false;
        }
    }
    return true;
}

// Helper function to compute numerical gradient
Tensor numerical_gradient(std::function<Variable(Variable&)> func,
                         Variable& input, float eps = 1e-4f) {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    auto grad = zeros(shape_vec);
    float* grad_data = grad.data<float>();
    float* input_data = input.tensor().data<float>();
    size_t numel = input.tensor().numel();

    for (size_t i = 0; i < numel; ++i) {
        float original = input_data[i];

        // f(x + eps)
        input_data[i] = original + eps;
        auto out_plus = func(input);
        float loss_plus = sum(out_plus.tensor()).data<float>()[0];

        // f(x - eps)
        input_data[i] = original - eps;
        auto out_minus = func(input);
        float loss_minus = sum(out_minus.tensor()).data<float>()[0];

        // Restore original
        input_data[i] = original;

        // Central difference
        grad_data[i] = (loss_plus - loss_minus) / (2.0f * eps);
    }

    return grad;
}

// ============================
// MaxPool2d Basic Shape Tests
// ============================

TEST(MaxPool2dTest, ForwardShapeBasic) {
    auto pool = nn::MaxPool2d(2, 2, 0);
    auto input = Variable(randn({2, 3, 32, 32}), true);
    auto output = pool.forward(input);

    // Output size = (32 - 2) / 2 + 1 = 16
    EXPECT_EQ(output.shape()[0], 2);   // batch
    EXPECT_EQ(output.shape()[1], 3);   // channels
    EXPECT_EQ(output.shape()[2], 16);  // height
    EXPECT_EQ(output.shape()[3], 16);  // width
}

TEST(MaxPool2dTest, ForwardShapeSingleBatch) {
    auto pool = nn::MaxPool2d(2);
    auto input = Variable(randn({1, 1, 28, 28}), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 1);
    EXPECT_EQ(output.shape()[2], 14);  // 28 / 2
    EXPECT_EQ(output.shape()[3], 14);
}

TEST(MaxPool2dTest, ForwardShapeMultiBatch) {
    auto pool = nn::MaxPool2d(3, 3, 0);
    auto input = Variable(randn({32, 16, 64, 64}), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[0], 32);
    EXPECT_EQ(output.shape()[1], 16);
    EXPECT_EQ(output.shape()[2], 21);  // (64 - 3) / 3 + 1
    EXPECT_EQ(output.shape()[3], 21);
}

TEST(MaxPool2dTest, KernelSize2) {
    auto pool = nn::MaxPool2d(2);
    auto input = Variable(randn({4, 8, 32, 32}), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 16);
    EXPECT_EQ(output.shape()[3], 16);
}

TEST(MaxPool2dTest, KernelSize3) {
    auto pool = nn::MaxPool2d(3);
    auto input = Variable(randn({2, 3, 33, 33}), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 11);  // 33 / 3
    EXPECT_EQ(output.shape()[3], 11);
}

TEST(MaxPool2dTest, KernelSize4) {
    auto pool = nn::MaxPool2d(4);
    auto input = Variable(randn({1, 1, 32, 32}), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 8);  // 32 / 4
    EXPECT_EQ(output.shape()[3], 8);
}

// ============================
// MaxPool2d Stride Tests
// ============================

TEST(MaxPool2dTest, StrideDefault) {
    auto pool = nn::MaxPool2d(3);  // stride defaults to kernel_size
    auto input = Variable(randn({2, 4, 33, 33}), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 11);  // (33 - 3) / 3 + 1
    EXPECT_EQ(output.shape()[3], 11);
}

TEST(MaxPool2dTest, Stride1) {
    auto pool = nn::MaxPool2d(2, 1);  // kernel=2, stride=1
    auto input = Variable(randn({2, 3, 32, 32}), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 31);  // (32 - 2) / 1 + 1
    EXPECT_EQ(output.shape()[3], 31);
}

TEST(MaxPool2dTest, Stride2) {
    auto pool = nn::MaxPool2d(3, 2);  // kernel=3, stride=2
    auto input = Variable(randn({2, 3, 32, 32}), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 15);  // (32 - 3) / 2 + 1
    EXPECT_EQ(output.shape()[3], 15);
}

// ============================
// MaxPool2d Padding Tests
// ============================

TEST(MaxPool2dTest, Padding0) {
    auto pool = nn::MaxPool2d(2, 2, 0);
    auto input = Variable(randn({2, 3, 32, 32}), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 16);
    EXPECT_EQ(output.shape()[3], 16);
}

TEST(MaxPool2dTest, Padding1) {
    auto pool = nn::MaxPool2d(3, 2, 1);
    auto input = Variable(randn({2, 3, 32, 32}), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 16);  // (32 + 2*1 - 3) / 2 + 1
    EXPECT_EQ(output.shape()[3], 16);
}

// ============================
// MaxPool2d Functional Tests
// ============================

TEST(MaxPool2dTest, MaxValueSelection) {
    // Test that max value is correctly selected
    auto pool = nn::MaxPool2d(2, 2, 0);

    // Create a simple input where we know the max values
    auto input_tensor = zeros({1, 1, 4, 4});
    float* data = input_tensor.data<float>();

    // Set specific values to verify max selection
    data[0] = 1.0f;  data[1] = 2.0f;  data[2] = 3.0f;  data[3] = 4.0f;
    data[4] = 5.0f;  data[5] = 6.0f;  data[6] = 7.0f;  data[7] = 8.0f;
    data[8] = 9.0f;  data[9] = 10.0f; data[10] = 11.0f; data[11] = 12.0f;
    data[12] = 13.0f; data[13] = 14.0f; data[14] = 15.0f; data[15] = 16.0f;

    auto input = Variable(input_tensor, true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 2);
    EXPECT_EQ(output.shape()[3], 2);

    const float* out_data = output.tensor().data<float>();
    EXPECT_FLOAT_EQ(out_data[0], 6.0f);   // max of [1,2,5,6]
    EXPECT_FLOAT_EQ(out_data[1], 8.0f);   // max of [3,4,7,8]
    EXPECT_FLOAT_EQ(out_data[2], 14.0f);  // max of [9,10,13,14]
    EXPECT_FLOAT_EQ(out_data[3], 16.0f);  // max of [11,12,15,16]
}

TEST(MaxPool2dTest, RequiresGrad) {
    auto pool = nn::MaxPool2d(2);
    auto input = Variable(randn({2, 3, 16, 16}), true);
    auto output = pool.forward(input);

    EXPECT_TRUE(output.requires_grad());
}

TEST(MaxPool2dTest, NoGradWhenInputNoGrad) {
    auto pool = nn::MaxPool2d(2);
    auto input = Variable(randn({2, 3, 16, 16}), false);
    auto output = pool.forward(input);

    EXPECT_FALSE(output.requires_grad());
}

// ============================
// MaxPool2d Gradient Tests
// ============================

TEST(MaxPool2dTest, BackwardPassExecutes) {
    auto pool = nn::MaxPool2d(2);
    auto input = Variable(randn({2, 3, 16, 16}), true);
    auto output = pool.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = ones(out_shape_vec);

    EXPECT_NO_THROW({
        output.backward(grad_output);
    });
}

TEST(MaxPool2dTest, GradientNonZero) {
    auto pool = nn::MaxPool2d(2);
    auto input = Variable(randn({2, 3, 16, 16}), true);
    auto output = pool.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    output.backward(ones(out_shape_vec));

    EXPECT_TRUE(input.grad().has_value());

    auto grad_data = input.grad()->data<float>();
    bool has_nonzero = false;
    size_t numel = static_cast<size_t>(input.grad()->numel());
    for (size_t i = 0; i < numel; ++i) {
        if (std::abs(grad_data[i]) > 1e-6f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST(MaxPool2dTest, GradientCheckSmall) {
    auto pool = nn::MaxPool2d(2);
    auto input = Variable(randn({1, 1, 4, 4}), true);

    auto loss_fn = [&pool](Variable& inp) -> Variable {
        auto out = pool.forward(inp);
        auto loss_tensor = sum(out.tensor());
        return Variable(loss_tensor, true);
    };

    auto numerical_grad = numerical_gradient(loss_fn, input, 1e-3f);
    auto loss = loss_fn(input);
    loss.backward();

    if (input.grad().has_value()) {
        EXPECT_TRUE(tensors_close(*input.grad(), numerical_grad, 1e-3f, 1e-3f));
    }
}

// ============================
// AvgPool2d Basic Shape Tests
// ============================

TEST(AvgPool2dTest, ForwardShapeBasic) {
    auto pool = nn::AvgPool2d(2, 2, 0);
    auto input = Variable(randn({2, 3, 32, 32}), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 16);
    EXPECT_EQ(output.shape()[3], 16);
}

TEST(AvgPool2dTest, ForwardShapeSingleBatch) {
    auto pool = nn::AvgPool2d(3);
    auto input = Variable(randn({1, 1, 27, 27}), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 1);
    EXPECT_EQ(output.shape()[2], 9);  // 27 / 3
    EXPECT_EQ(output.shape()[3], 9);
}

TEST(AvgPool2dTest, ForwardShapeMultiBatch) {
    auto pool = nn::AvgPool2d(2, 2, 0);
    auto input = Variable(randn({16, 32, 56, 56}), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[0], 16);
    EXPECT_EQ(output.shape()[1], 32);
    EXPECT_EQ(output.shape()[2], 28);
    EXPECT_EQ(output.shape()[3], 28);
}

TEST(AvgPool2dTest, KernelSize2) {
    auto pool = nn::AvgPool2d(2);
    auto input = Variable(randn({4, 8, 32, 32}), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 16);
    EXPECT_EQ(output.shape()[3], 16);
}

TEST(AvgPool2dTest, KernelSize3) {
    auto pool = nn::AvgPool2d(3);
    auto input = Variable(randn({2, 3, 30, 30}), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 10);
    EXPECT_EQ(output.shape()[3], 10);
}

// ============================
// AvgPool2d Stride Tests
// ============================

TEST(AvgPool2dTest, StrideDefault) {
    auto pool = nn::AvgPool2d(4);
    auto input = Variable(randn({2, 4, 32, 32}), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 8);
    EXPECT_EQ(output.shape()[3], 8);
}

TEST(AvgPool2dTest, Stride1) {
    auto pool = nn::AvgPool2d(2, 1);
    auto input = Variable(randn({2, 3, 32, 32}), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 31);  // (32 - 2) / 1 + 1
    EXPECT_EQ(output.shape()[3], 31);
}

TEST(AvgPool2dTest, Stride2) {
    auto pool = nn::AvgPool2d(3, 2);
    auto input = Variable(randn({2, 3, 32, 32}), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 15);
    EXPECT_EQ(output.shape()[3], 15);
}

// ============================
// AvgPool2d Functional Tests
// ============================

TEST(AvgPool2dTest, AverageValueComputation) {
    auto pool = nn::AvgPool2d(2, 2, 0);

    auto input_tensor = zeros({1, 1, 4, 4});
    float* data = input_tensor.data<float>();

    // Set known values
    for (int i = 0; i < 16; ++i) {
        data[i] = static_cast<float>(i + 1);
    }

    auto input = Variable(input_tensor, true);
    auto output = pool.forward(input);

    const float* out_data = output.tensor().data<float>();
    EXPECT_FLOAT_EQ(out_data[0], 3.5f);   // avg of [1,2,5,6]
    EXPECT_FLOAT_EQ(out_data[1], 5.5f);   // avg of [3,4,7,8]
    EXPECT_FLOAT_EQ(out_data[2], 11.5f);  // avg of [9,10,13,14]
    EXPECT_FLOAT_EQ(out_data[3], 13.5f);  // avg of [11,12,15,16]
}

TEST(AvgPool2dTest, RequiresGrad) {
    auto pool = nn::AvgPool2d(2);
    auto input = Variable(randn({2, 3, 16, 16}), true);
    auto output = pool.forward(input);

    EXPECT_TRUE(output.requires_grad());
}

TEST(AvgPool2dTest, NoGradWhenInputNoGrad) {
    auto pool = nn::AvgPool2d(2);
    auto input = Variable(randn({2, 3, 16, 16}), false);
    auto output = pool.forward(input);

    EXPECT_FALSE(output.requires_grad());
}

// ============================
// AvgPool2d Gradient Tests
// ============================

TEST(AvgPool2dTest, BackwardPassExecutes) {
    auto pool = nn::AvgPool2d(2);
    auto input = Variable(randn({2, 3, 16, 16}), true);
    auto output = pool.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = ones(out_shape_vec);

    EXPECT_NO_THROW({
        output.backward(grad_output);
    });
}

TEST(AvgPool2dTest, GradientNonZero) {
    auto pool = nn::AvgPool2d(2);
    auto input = Variable(randn({2, 3, 16, 16}), true);
    auto output = pool.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    output.backward(ones(out_shape_vec));

    EXPECT_TRUE(input.grad().has_value());

    auto grad_data = input.grad()->data<float>();
    bool has_nonzero = false;
    size_t numel = static_cast<size_t>(input.grad()->numel());
    for (size_t i = 0; i < numel; ++i) {
        if (std::abs(grad_data[i]) > 1e-6f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST(AvgPool2dTest, GradientCheckSmall) {
    auto pool = nn::AvgPool2d(2);
    auto input = Variable(randn({1, 1, 4, 4}), true);

    auto loss_fn = [&pool](Variable& inp) -> Variable {
        auto out = pool.forward(inp);
        auto loss_tensor = sum(out.tensor());
        return Variable(loss_tensor, true);
    };

    auto numerical_grad = numerical_gradient(loss_fn, input, 1e-3f);
    auto loss = loss_fn(input);
    loss.backward();

    if (input.grad().has_value()) {
        EXPECT_TRUE(tensors_close(*input.grad(), numerical_grad, 1e-3f, 1e-3f));
    }
}

// ============================
// AdaptiveAvgPool2d Tests
// ============================

TEST(AdaptiveAvgPool2dTest, ForwardShapeBasic) {
    auto pool = nn::AdaptiveAvgPool2d(7, 7);
    auto input = Variable(randn({2, 3, 32, 32}), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 7);
    EXPECT_EQ(output.shape()[3], 7);
}

TEST(AdaptiveAvgPool2dTest, ForwardShapeSquare) {
    auto pool = nn::AdaptiveAvgPool2d(1);  // Global average pooling
    auto input = Variable(randn({4, 512, 7, 7}), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 512);
    EXPECT_EQ(output.shape()[2], 1);
    EXPECT_EQ(output.shape()[3], 1);
}

TEST(AdaptiveAvgPool2dTest, ForwardShapeRectangular) {
    auto pool = nn::AdaptiveAvgPool2d(5, 3);
    auto input = Variable(randn({2, 16, 28, 28}), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 16);
    EXPECT_EQ(output.shape()[2], 5);
    EXPECT_EQ(output.shape()[3], 3);
}

TEST(AdaptiveAvgPool2dTest, GlobalAveragePooling) {
    auto pool = nn::AdaptiveAvgPool2d(1, 1);
    auto input = Variable(randn({2, 64, 14, 14}), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 1);
    EXPECT_EQ(output.shape()[3], 1);
}

TEST(AdaptiveAvgPool2dTest, OutputLargerThanInput) {
    // Output can be larger than input (upsampling)
    auto pool = nn::AdaptiveAvgPool2d(8, 8);
    auto input = Variable(randn({1, 3, 4, 4}), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 8);
    EXPECT_EQ(output.shape()[3], 8);
}

TEST(AdaptiveAvgPool2dTest, RequiresGrad) {
    auto pool = nn::AdaptiveAvgPool2d(7);
    auto input = Variable(randn({2, 3, 32, 32}), true);
    auto output = pool.forward(input);

    EXPECT_TRUE(output.requires_grad());
}

TEST(AdaptiveAvgPool2dTest, NoGradWhenInputNoGrad) {
    auto pool = nn::AdaptiveAvgPool2d(7);
    auto input = Variable(randn({2, 3, 32, 32}), false);
    auto output = pool.forward(input);

    EXPECT_FALSE(output.requires_grad());
}

TEST(AdaptiveAvgPool2dTest, BackwardPassExecutes) {
    auto pool = nn::AdaptiveAvgPool2d(7);
    auto input = Variable(randn({2, 3, 32, 32}), true);
    auto output = pool.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = ones(out_shape_vec);

    EXPECT_NO_THROW({
        output.backward(grad_output);
    });
}

TEST(AdaptiveAvgPool2dTest, GradientNonZero) {
    auto pool = nn::AdaptiveAvgPool2d(4);
    auto input = Variable(randn({2, 3, 16, 16}), true);
    auto output = pool.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    output.backward(ones(out_shape_vec));

    EXPECT_TRUE(input.grad().has_value());

    auto grad_data = input.grad()->data<float>();
    bool has_nonzero = false;
    size_t numel = static_cast<size_t>(input.grad()->numel());
    for (size_t i = 0; i < numel; ++i) {
        if (std::abs(grad_data[i]) > 1e-6f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST(AdaptiveAvgPool2dTest, GradientCheckSmall) {
    auto pool = nn::AdaptiveAvgPool2d(2);
    auto input = Variable(randn({1, 1, 4, 4}), true);

    auto loss_fn = [&pool](Variable& inp) -> Variable {
        auto out = pool.forward(inp);
        auto loss_tensor = sum(out.tensor());
        return Variable(loss_tensor, true);
    };

    auto numerical_grad = numerical_gradient(loss_fn, input, 1e-3f);
    auto loss = loss_fn(input);
    loss.backward();

    if (input.grad().has_value()) {
        EXPECT_TRUE(tensors_close(*input.grad(), numerical_grad, 1e-3f, 1e-3f));
    }
}

// ============================
// Edge Cases
// ============================

TEST(PoolingTest, MaxPoolEdgeCase1x1) {
    auto pool = nn::MaxPool2d(1);
    auto input = Variable(randn({2, 3, 16, 16}), true);
    auto output = pool.forward(input);

    // 1x1 pooling should not change dimensions
    EXPECT_EQ(output.shape()[2], 16);
    EXPECT_EQ(output.shape()[3], 16);
}

TEST(PoolingTest, AvgPoolEdgeCase1x1) {
    auto pool = nn::AvgPool2d(1);
    auto input = Variable(randn({2, 3, 16, 16}), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 16);
    EXPECT_EQ(output.shape()[3], 16);
}

TEST(PoolingTest, AdaptivePoolSameSize) {
    // Adaptive pool to same size as input
    auto pool = nn::AdaptiveAvgPool2d(16, 16);
    auto input = Variable(randn({2, 3, 16, 16}), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 16);
    EXPECT_EQ(output.shape()[3], 16);
}

TEST(PoolingTest, LargeBatch) {
    auto pool = nn::MaxPool2d(2);
    auto input = Variable(randn({128, 3, 32, 32}), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[0], 128);
}

TEST(PoolingTest, ManyChannels) {
    auto pool = nn::AvgPool2d(2);
    auto input = Variable(randn({2, 512, 7, 7}), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[1], 512);
}

// ============================
// Error Handling
// ============================

TEST(PoolingTest, InvalidInputDimensions) {
    auto pool = nn::MaxPool2d(2);
    auto input_3d = Variable(randn({2, 3, 16}), true);

    EXPECT_THROW({
        pool.forward(input_3d);
    }, std::invalid_argument);
}

// ============================
// Integration Tests
// ============================

TEST(PoolingTest, SequentialPooling) {
    // Test multiple pooling layers in sequence
    auto pool1 = nn::MaxPool2d(2);
    auto pool2 = nn::MaxPool2d(2);

    auto input = Variable(randn({2, 3, 64, 64}), true);
    auto x = pool1.forward(input);
    auto output = pool2.forward(x);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 16);  // 64 -> 32 -> 16
    EXPECT_EQ(output.shape()[3], 16);
}

TEST(PoolingTest, MixedPoolingTypes) {
    auto max_pool = nn::MaxPool2d(2);
    auto avg_pool = nn::AvgPool2d(2);

    auto input = Variable(randn({2, 3, 32, 32}), true);
    auto x1 = max_pool.forward(input);
    auto x2 = avg_pool.forward(input);

    // Both should have same output shape
    auto shape1 = x1.shape();
    auto shape2 = x2.shape();
    EXPECT_TRUE(std::equal(shape1.begin(), shape1.end(), shape2.begin()));
}

TEST(PoolingTest, PoolingWithConvolution) {
    // Common pattern: Conv -> Pool
    auto conv = nn::Conv2d(3, 16, 3, 1, 1);
    auto pool = nn::MaxPool2d(2);

    auto input = Variable(randn({2, 3, 32, 32}), true);
    auto x = conv.forward(input);
    auto output = pool.forward(x);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 16);
    EXPECT_EQ(output.shape()[2], 16);  // 32 -> 16 after pooling
    EXPECT_EQ(output.shape()[3], 16);
}

TEST(PoolingTest, ResNetStyleBottleneck) {
    // Test ResNet-style block with pooling
    auto conv1 = nn::Conv2d(64, 128, 3, 1, 1);
    auto pool = nn::MaxPool2d(2);
    auto conv2 = nn::Conv2d(128, 128, 3, 1, 1);

    auto input = Variable(randn({2, 64, 56, 56}), true);
    auto x = conv1.forward(input);
    x = pool.forward(x);
    auto output = conv2.forward(x);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 128);
    EXPECT_EQ(output.shape()[2], 28);  // 56 / 2
    EXPECT_EQ(output.shape()[3], 28);
}
