#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/layers/attention.hpp"
#include "tenzor/nn/layers/normalization.hpp"
#include <iostream>

using namespace tenzor;
using namespace tenzor::nn;

class TransformerMinimalTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        tenzor::initialize();
    }
};

// Test just layer norm
TEST_F(TransformerMinimalTest, LayerNormBackward) {
    std::cout << "\n=== Testing LayerNorm backward ===" << std::endl;

    LayerNorm norm({128});
    Variable input(randn({2, 5, 128}), true);

    Variable output = norm.forward(input);
    Variable loss = mean(output);

    std::cout << "Calling backward..." << std::endl;
    try {
        loss.backward();
        std::cout << "SUCCESS: LayerNorm backward passed!" << std::endl;
        EXPECT_TRUE(input.has_grad());
    } catch (const std::exception& e) {
        std::cout << "FAILED: " << e.what() << std::endl;
        EXPECT_TRUE(false);
    }
}

// Test multi-head attention
TEST_F(TransformerMinimalTest, MultiheadAttentionBackward) {
    std::cout << "\n=== Testing MultiheadAttention backward ===" << std::endl;

    MultiheadAttention attn(128, 4);
    Variable query(randn({2, 5, 128}), true);
    Variable key(randn({2, 5, 128}), true);
    Variable value(randn({2, 5, 128}), true);

    Variable output = attn.forward(query, key, value);
    Variable loss = mean(output);

    std::cout << "Calling backward..." << std::endl;
    try {
        loss.backward();
        std::cout << "SUCCESS: MultiheadAttention backward passed!" << std::endl;
        EXPECT_TRUE(query.has_grad());
    } catch (const std::exception& e) {
        std::cout << "FAILED: " << e.what() << std::endl;
        EXPECT_TRUE(false);
    }
}
