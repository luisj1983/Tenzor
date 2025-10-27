#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class LinearTest : public BackendTest {};

TEST_P(LinearTest, ForwardShapeSingleBatch) {
    // Test with single batch dimension
    auto linear = nn::Linear(10, 5);
    auto input = Variable(ones({1, 10}, DType::Float32, device), true);
    auto output = linear.forward(input);

    EXPECT_EQ(output.shape()[0], 1) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string();
}

TEST_P(LinearTest, ForwardShapeMultiBatch) {
    // Test with multiple batch dimension
    auto linear = nn::Linear(10, 5);
    auto input = Variable(ones({32, 10}, DType::Float32, device), true);
    auto output = linear.forward(input);

    EXPECT_EQ(output.shape()[0], 32) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string();
}

TEST_P(LinearTest, WithBias) {
    // Test that bias is applied correctly
    auto linear = nn::Linear(3, 2, true);

    // Check bias is initialized
    EXPECT_TRUE(linear.has_bias()) << "Failed on " << device.to_string();
    auto bias_ptr = linear.bias();
    EXPECT_NE(bias_ptr, nullptr) << "Failed on " << device.to_string();

    // Check bias shape
    auto bias_shape = bias_ptr->shape();
    EXPECT_EQ(bias_shape.size(), 1) << "Failed on " << device.to_string();
    EXPECT_EQ(bias_shape[0], 2) << "Failed on " << device.to_string();

    // Test forward pass
    auto input = Variable(zeros({4, 3}, DType::Float32, device), true);
    auto output = linear.forward(input);

    // With zero input, output should equal bias (broadcasted)
    EXPECT_EQ(output.shape()[0], 4) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 2) << "Failed on " << device.to_string();
}

TEST_P(LinearTest, NoBias) {
    // Test layer without bias
    auto linear = nn::Linear(3, 2, false);

    // Check bias is not initialized
    EXPECT_FALSE(linear.has_bias()) << "Failed on " << device.to_string();
    auto bias_ptr = linear.bias();
    EXPECT_EQ(bias_ptr, nullptr) << "Failed on " << device.to_string();

    // Test forward pass still works
    auto input = Variable(ones({4, 3}, DType::Float32, device), true);
    auto output = linear.forward(input);

    EXPECT_EQ(output.shape()[0], 4) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 2) << "Failed on " << device.to_string();
}

TEST_P(LinearTest, WeightShape) {
    // Test weight matrix has correct shape
    auto linear = nn::Linear(10, 5);
    auto weight = linear.weight();

    auto weight_shape = weight->shape();
    EXPECT_EQ(weight_shape.size(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(weight_shape[0], 5) << "Failed on " << device.to_string();  // out_features
    EXPECT_EQ(weight_shape[1], 10) << "Failed on " << device.to_string(); // in_features
}

TEST_P(LinearTest, ParameterCount) {
    // Test with bias
    auto linear_with_bias = nn::Linear(10, 5, true);
    auto params_with_bias = linear_with_bias.parameters();
    EXPECT_EQ(params_with_bias.size(), 2) << "Failed on " << device.to_string(); // weight and bias

    // Test without bias
    auto linear_no_bias = nn::Linear(10, 5, false);
    auto params_no_bias = linear_no_bias.parameters();
    EXPECT_EQ(params_no_bias.size(), 1) << "Failed on " << device.to_string(); // only weight
}

TEST_P(LinearTest, DifferentInputShapes) {
    // Test various valid input shapes
    auto linear = nn::Linear(8, 4);

    // Batch size 1
    auto input1 = Variable(randn({1, 8}, DType::Float32, device), true);
    auto output1 = linear.forward(input1);
    EXPECT_EQ(output1.shape()[0], 1) << "Failed on " << device.to_string();
    EXPECT_EQ(output1.shape()[1], 4) << "Failed on " << device.to_string();

    // Batch size 16
    auto input2 = Variable(randn({16, 8}, DType::Float32, device), true);
    auto output2 = linear.forward(input2);
    EXPECT_EQ(output2.shape()[0], 16) << "Failed on " << device.to_string();
    EXPECT_EQ(output2.shape()[1], 4) << "Failed on " << device.to_string();

    // Batch size 64
    auto input3 = Variable(randn({64, 8}, DType::Float32, device), true);
    auto output3 = linear.forward(input3);
    EXPECT_EQ(output3.shape()[0], 64) << "Failed on " << device.to_string();
    EXPECT_EQ(output3.shape()[1], 4) << "Failed on " << device.to_string();
}

TEST_P(LinearTest, ContiguityAfterTranspose) {
    // This is the critical test for the fix
    // Linear layer internally transposes weights, which creates non-contiguous view
    // The fix ensures we call .contiguous() before matmul

    auto linear = nn::Linear(10, 5);
    auto input = Variable(randn({32, 10}, DType::Float32, device), true);

    // This should NOT throw "matmul requires contiguous tensors"
    EXPECT_NO_THROW({
        auto output = linear.forward(input);
        EXPECT_EQ(output.shape()[0], 32) << "Failed on " << device.to_string();
        EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string();
    }) << "Failed on " << device.to_string();
}

TEST_P(LinearTest, ZeroInput) {
    // Test with zero input - output should be bias (if present)
    auto linear = nn::Linear(5, 3, true);
    auto input = Variable(zeros({2, 5}, DType::Float32, device), true);
    auto output = linear.forward(input);

    EXPECT_EQ(output.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 3) << "Failed on " << device.to_string();
}

TEST_P(LinearTest, ConsistentOutput) {
    // Test that same input produces same output (deterministic)
    auto linear = nn::Linear(4, 2);
    auto input = Variable(ones({3, 4}, DType::Float32, device), true);

    auto output1 = linear.forward(input);
    auto output2 = linear.forward(input);

    // Shapes should be identical
    auto shape1 = output1.shape();
    auto shape2 = output2.shape();
    EXPECT_EQ(shape1.size(), shape2.size()) << "Failed on " << device.to_string();
    for (size_t i = 0; i < shape1.size(); ++i) {
        EXPECT_EQ(shape1[i], shape2[i]) << "Failed on " << device.to_string();
    }

    // Values should be identical (same weights, same input)
    auto data1 = output1.tensor().to(Device::cpu()).data<float>();
    auto data2 = output2.tensor().to(Device::cpu()).data<float>();
    auto size = output1.tensor().numel();

    for (size_t i = 0; i < static_cast<size_t>(size); ++i) {
        EXPECT_FLOAT_EQ(data1[i], data2[i]) << "Failed on " << device.to_string();
    }
}

TEST_P(LinearTest, LargeFeatures) {
    // Test with large feature dimensions
    auto linear = nn::Linear(1024, 512);
    auto input = Variable(randn({8, 1024}, DType::Float32, device), true);
    auto output = linear.forward(input);

    EXPECT_EQ(output.shape()[0], 8) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 512) << "Failed on " << device.to_string();
}

TEST_P(LinearTest, SingleFeature) {
    // Test edge case with single feature
    auto linear = nn::Linear(1, 1);
    auto input = Variable(randn({10, 1}, DType::Float32, device), true);
    auto output = linear.forward(input);

    EXPECT_EQ(output.shape()[0], 10) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 1) << "Failed on " << device.to_string();
}

INSTANTIATE_BACKEND_TESTS(LinearTest);
