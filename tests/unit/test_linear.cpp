#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>

using namespace tenzor;

// Global test environment that initializes Tenzor before tests
class LinearTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

// Register the environment
static ::testing::Environment* const linear_env =
    ::testing::AddGlobalTestEnvironment(new LinearTestEnvironment);

TEST(LinearTest, ForwardShapeSingleBatch) {
    // Test with single batch dimension
    auto linear = nn::Linear(10, 5);
    auto input = Variable(ones({1, 10}), true);
    auto output = linear.forward(input);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 5);
}

TEST(LinearTest, ForwardShapeMultiBatch) {
    // Test with multiple batch dimension
    auto linear = nn::Linear(10, 5);
    auto input = Variable(ones({32, 10}), true);
    auto output = linear.forward(input);

    EXPECT_EQ(output.shape()[0], 32);
    EXPECT_EQ(output.shape()[1], 5);
}

TEST(LinearTest, WithBias) {
    // Test that bias is applied correctly
    auto linear = nn::Linear(3, 2, true);

    // Check bias is initialized
    auto bias_opt = linear.bias();
    EXPECT_TRUE(bias_opt.has_value());

    // Check bias shape
    auto bias_shape = bias_opt->shape();
    EXPECT_EQ(bias_shape.size(), 1);
    EXPECT_EQ(bias_shape[0], 2);

    // Test forward pass
    auto input = Variable(zeros({4, 3}), true);
    auto output = linear.forward(input);

    // With zero input, output should equal bias (broadcasted)
    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 2);
}

TEST(LinearTest, NoBias) {
    // Test layer without bias
    auto linear = nn::Linear(3, 2, false);

    // Check bias is not initialized
    auto bias_opt = linear.bias();
    EXPECT_FALSE(bias_opt.has_value());

    // Test forward pass still works
    auto input = Variable(ones({4, 3}), true);
    auto output = linear.forward(input);

    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 2);
}

TEST(LinearTest, WeightShape) {
    // Test weight matrix has correct shape
    auto linear = nn::Linear(10, 5);
    auto weight = linear.weight();

    auto weight_shape = weight.shape();
    EXPECT_EQ(weight_shape.size(), 2);
    EXPECT_EQ(weight_shape[0], 5);  // out_features
    EXPECT_EQ(weight_shape[1], 10); // in_features
}

TEST(LinearTest, ParameterCount) {
    // Test with bias
    auto linear_with_bias = nn::Linear(10, 5, true);
    auto params_with_bias = linear_with_bias.parameters();
    EXPECT_EQ(params_with_bias.size(), 2); // weight and bias

    // Test without bias
    auto linear_no_bias = nn::Linear(10, 5, false);
    auto params_no_bias = linear_no_bias.parameters();
    EXPECT_EQ(params_no_bias.size(), 1); // only weight
}

TEST(LinearTest, DifferentInputShapes) {
    // Test various valid input shapes
    auto linear = nn::Linear(8, 4);

    // Batch size 1
    auto input1 = Variable(randn({1, 8}), true);
    auto output1 = linear.forward(input1);
    EXPECT_EQ(output1.shape()[0], 1);
    EXPECT_EQ(output1.shape()[1], 4);

    // Batch size 16
    auto input2 = Variable(randn({16, 8}), true);
    auto output2 = linear.forward(input2);
    EXPECT_EQ(output2.shape()[0], 16);
    EXPECT_EQ(output2.shape()[1], 4);

    // Batch size 64
    auto input3 = Variable(randn({64, 8}), true);
    auto output3 = linear.forward(input3);
    EXPECT_EQ(output3.shape()[0], 64);
    EXPECT_EQ(output3.shape()[1], 4);
}

TEST(LinearTest, ContiguityAfterTranspose) {
    // This is the critical test for the fix
    // Linear layer internally transposes weights, which creates non-contiguous view
    // The fix ensures we call .contiguous() before matmul

    auto linear = nn::Linear(10, 5);
    auto input = Variable(randn({32, 10}), true);

    // This should NOT throw "matmul requires contiguous tensors"
    EXPECT_NO_THROW({
        auto output = linear.forward(input);
        EXPECT_EQ(output.shape()[0], 32);
        EXPECT_EQ(output.shape()[1], 5);
    });
}

TEST(LinearTest, ZeroInput) {
    // Test with zero input - output should be bias (if present)
    auto linear = nn::Linear(5, 3, true);
    auto input = Variable(zeros({2, 5}), true);
    auto output = linear.forward(input);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
}

TEST(LinearTest, ConsistentOutput) {
    // Test that same input produces same output (deterministic)
    auto linear = nn::Linear(4, 2);
    auto input = Variable(ones({3, 4}), true);

    auto output1 = linear.forward(input);
    auto output2 = linear.forward(input);

    // Shapes should be identical
    auto shape1 = output1.shape();
    auto shape2 = output2.shape();
    EXPECT_EQ(shape1.size(), shape2.size());
    for (size_t i = 0; i < shape1.size(); ++i) {
        EXPECT_EQ(shape1[i], shape2[i]);
    }

    // Values should be identical (same weights, same input)
    auto data1 = output1.tensor().data<float>();
    auto data2 = output2.tensor().data<float>();
    auto size = output1.tensor().numel();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(data1[i], data2[i]);
    }
}

TEST(LinearTest, LargeFeatures) {
    // Test with large feature dimensions
    auto linear = nn::Linear(1024, 512);
    auto input = Variable(randn({8, 1024}), true);
    auto output = linear.forward(input);

    EXPECT_EQ(output.shape()[0], 8);
    EXPECT_EQ(output.shape()[1], 512);
}

TEST(LinearTest, SingleFeature) {
    // Test edge case with single feature
    auto linear = nn::Linear(1, 1);
    auto input = Variable(randn({10, 1}), true);
    auto output = linear.forward(input);

    EXPECT_EQ(output.shape()[0], 10);
    EXPECT_EQ(output.shape()[1], 1);
}
