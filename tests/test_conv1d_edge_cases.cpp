#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <stdexcept>

using namespace tenzor;

// Test environment that initializes Tenzor
class Conv1dEdgeCaseTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const edge_case_env =
    ::testing::AddGlobalTestEnvironment(new Conv1dEdgeCaseTestEnvironment);

class Conv1dEdgeCaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::srand(42);
    }
};

// Test edge case: kernel larger than input (should throw meaningful error, not bad_alloc)
TEST_F(Conv1dEdgeCaseTest, KernelLargerThanInput) {
    auto input_tensor = randn({1, 2, 3});  // length=3
    auto input = Variable(input_tensor, false);

    // kernel_size=5 > input length=3, should throw invalid_argument with helpful message
    auto conv = nn::Conv1d(2, 4, 5, 1, 0, 1, 1, false);

    EXPECT_THROW({
        try {
            auto output = conv.forward(input);
        } catch (const std::invalid_argument& e) {
            // Verify it's the right error with helpful message
            std::string msg = e.what();
            EXPECT_TRUE(msg.find("output length is non-positive") != std::string::npos ||
                       msg.find("Invalid Conv1d configuration") != std::string::npos);
            throw;  // Re-throw for EXPECT_THROW
        }
    }, std::invalid_argument);
}

// Test edge case: large dilation causing negative output size
TEST_F(Conv1dEdgeCaseTest, LargeDilation) {
    auto input_tensor = randn({1, 2, 5});  // length=5
    auto input = Variable(input_tensor, false);

    // dilation=3, kernel=3 -> effective kernel = 3 + (3-1)*3 = 9 > input=5
    auto conv = nn::Conv1d(2, 4, 3, 1, 0, 3, 1, false);

    EXPECT_THROW({
        try {
            auto output = conv.forward(input);
        } catch (const std::invalid_argument& e) {
            std::string msg = e.what();
            EXPECT_TRUE(msg.find("output length is non-positive") != std::string::npos ||
                       msg.find("Invalid Conv1d configuration") != std::string::npos);
            throw;
        }
    }, std::invalid_argument);
}

// Test valid edge case: minimal valid configuration
TEST_F(Conv1dEdgeCaseTest, MinimalValidConfig) {
    auto input_tensor = randn({1, 2, 5});  // length=5
    auto input = Variable(input_tensor, false);

    // kernel=5, length=5 -> output length=1 (valid)
    auto conv = nn::Conv1d(2, 4, 5, 1, 0, 1, 1, false);

    EXPECT_NO_THROW({
        auto output = conv.forward(input);
        auto shape = output.shape();
        EXPECT_EQ(shape[2], 1);  // output length should be 1
    });
}

// Test valid case with padding that makes large kernel work
TEST_F(Conv1dEdgeCaseTest, LargeKernelWithPadding) {
    auto input_tensor = randn({1, 2, 3});  // length=3
    auto input = Variable(input_tensor, false);

    // kernel=5, length=3, padding=2 -> padded length=7 -> output length=3 (valid)
    auto conv = nn::Conv1d(2, 4, 5, 1, 2, 1, 1, false);

    EXPECT_NO_THROW({
        auto output = conv.forward(input);
        auto shape = output.shape();
        EXPECT_EQ(shape[2], 3);  // output length should be 3
    });
}

// Test Conv2d edge case
TEST_F(Conv1dEdgeCaseTest, Conv2dEdgeCase) {
    auto input_tensor = randn({1, 2, 3, 3});  // 3x3 input
    auto input = Variable(input_tensor, false);

    // 5x5 kernel on 3x3 input with no padding should fail
    auto conv = nn::Conv2d(2, 4, 5, 1, 0, 1, 1, false);

    EXPECT_THROW({
        try {
            auto output = conv.forward(input);
        } catch (const std::invalid_argument& e) {
            std::string msg = e.what();
            EXPECT_TRUE(msg.find("output dimensions are non-positive") != std::string::npos ||
                       msg.find("Invalid Conv2d configuration") != std::string::npos);
            throw;
        }
    }, std::invalid_argument);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
