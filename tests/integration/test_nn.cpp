#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>

using namespace tenzor;

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

TEST(NNTest, LinearLayer) {
    auto layer = std::make_shared<nn::Linear>(10, 5);
    auto input = Variable(randn({32, 10}), true);

    auto output = layer->forward(input);

    EXPECT_EQ(output.shape()[0], 32);
    EXPECT_EQ(output.shape()[1], 5);
}

TEST(NNTest, Sequential) {
    auto model = nn::Sequential(
        std::make_shared<nn::Linear>(10, 20),
        std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Linear>(20, 5)
    );

    auto input = Variable(randn({16, 10}), true);
    auto output = model.forward(input);

    EXPECT_EQ(output.shape()[0], 16);
    EXPECT_EQ(output.shape()[1], 5);
}
