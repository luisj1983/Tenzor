#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>

using namespace tenzor;
using namespace tenzor::testing;

// Global test environment that initializes Tenzor before tests
class TenzorTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

// Register the environment
static ::testing::Environment* const tenzor_env =
    ::testing::AddGlobalTestEnvironment(new TenzorTestEnvironment);

//==============================================================================
// Neural Network Tests - Backend Parameterized
//==============================================================================

class NNTest : public BackendTest {};

TEST_P(NNTest, LinearLayer) {
    auto layer = std::make_shared<nn::Linear>(10, 5);
    layer->to(device);

    auto input = Variable(randn({32, 10}, DType::Float32, device), true);

    auto output = layer->forward(input);

    EXPECT_EQ(output.shape()[0], 32);
    EXPECT_EQ(output.shape()[1], 5);
    EXPECT_EQ(output.device().type, device.type);
}

TEST_P(NNTest, Sequential) {
    auto model = nn::Sequential(
        std::make_shared<nn::Linear>(10, 20),
        std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Linear>(20, 5)
    );
    model.to(device);

    auto input = Variable(randn({16, 10}, DType::Float32, device), true);
    auto output = model.forward(input);

    EXPECT_EQ(output.shape()[0], 16);
    EXPECT_EQ(output.shape()[1], 5);
    EXPECT_EQ(output.device().type, device.type);
}

TEST_P(NNTest, Conv2dLayer) {
    auto conv = std::make_shared<nn::Conv2d>(3, 16, 3, 1, 1);
    conv->to(device);

    auto input = Variable(randn({4, 3, 32, 32}, DType::Float32, device), true);
    auto output = conv->forward(input);

    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 16);
    EXPECT_EQ(output.shape()[2], 32);
    EXPECT_EQ(output.shape()[3], 32);
    EXPECT_EQ(output.device().type, device.type);
}

TEST_P(NNTest, BatchNorm2d) {
    auto bn = std::make_shared<nn::BatchNorm2d>(16);
    bn->to(device);
    bn->eval();  // Use eval mode for deterministic behavior

    auto input = Variable(randn({4, 16, 32, 32}, DType::Float32, device), true);
    auto output = bn->forward(input);

    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 16);
    EXPECT_EQ(output.shape()[2], 32);
    EXPECT_EQ(output.shape()[3], 32);
    EXPECT_EQ(output.device().type, device.type);
}

TEST_P(NNTest, ReLUActivation) {
    auto relu = std::make_shared<nn::ReLU>();

    auto input_data = randn({10, 20}, DType::Float32, device);
    auto input = Variable(input_data, true);
    auto output = relu->forward(input);

    EXPECT_EQ(output.shape()[0], 10);
    EXPECT_EQ(output.shape()[1], 20);
    EXPECT_EQ(output.device().type, device.type);

    // Verify ReLU property: all values >= 0
    auto output_cpu = output.tensor().to(Device::cpu());
    auto* data = output_cpu.data<float>();
    for (int64_t i = 0; i < output_cpu.numel(); ++i) {
        EXPECT_GE(data[i], 0.0f);
    }
}

TEST_P(NNTest, DropoutInTrainMode) {
    auto dropout = std::make_shared<nn::Dropout>(0.5);
    dropout->train();

    auto input = Variable(ones({100, 100}, DType::Float32, device), true);
    auto output = dropout->forward(input);

    EXPECT_EQ(output.shape()[0], 100);
    EXPECT_EQ(output.shape()[1], 100);
    EXPECT_EQ(output.device().type, device.type);
}

TEST_P(NNTest, DropoutInEvalMode) {
    auto dropout = std::make_shared<nn::Dropout>(0.5);
    dropout->eval();

    auto input_data = ones({100, 100}, DType::Float32, device);
    auto input = Variable(input_data, true);
    auto output = dropout->forward(input);

    EXPECT_EQ(output.shape()[0], 100);
    EXPECT_EQ(output.shape()[1], 100);
    EXPECT_EQ(output.device().type, device.type);

    // In eval mode, output should equal input
    expectTensorNear(output.tensor(), input_data, 1e-6f);
}

TEST_P(NNTest, MaxPool2d) {
    auto pool = std::make_shared<nn::MaxPool2d>(2, 2);

    auto input = Variable(randn({4, 16, 32, 32}, DType::Float32, device), true);
    auto output = pool->forward(input);

    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 16);
    EXPECT_EQ(output.shape()[2], 16);
    EXPECT_EQ(output.shape()[3], 16);
    EXPECT_EQ(output.device().type, device.type);
}

TEST_P(NNTest, Flatten) {
    auto flatten = std::make_shared<nn::Flatten>(1);

    auto input = Variable(randn({8, 3, 32, 32}, DType::Float32, device), true);
    auto output = flatten->forward(input);

    EXPECT_EQ(output.shape()[0], 8);
    EXPECT_EQ(output.shape()[1], 3 * 32 * 32);
    EXPECT_EQ(output.device().type, device.type);
}

INSTANTIATE_BACKEND_TESTS(NNTest);
