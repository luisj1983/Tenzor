/**
 * @file test_onednn_format_any.cpp
 * @brief Task 6.6: format_tag::any for oneDNN pooling and linear primitives
 *
 * Tests that the oneDNN pooling and linear paths produce correct results
 * after format_tag::any is used in the primitive descriptor (allowing
 * oneDNN to pick its preferred blocked layout, with reorder if needed).
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/nn/layers/pooling.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/core/tensor.hpp>
#include <tenzor/core/dtype.hpp>
#include <tenzor/ops/creation.hpp>
#include <cmath>
#include <vector>

using namespace tenzor;

struct OneDNNFormatEnv : ::testing::Environment {
    void SetUp() override { tenzor::initialize(); }
};

static ::testing::Environment* const fmt_env =
    ::testing::AddGlobalTestEnvironment(new OneDNNFormatEnv);

// ===========================================================================
// AvgPool2d — verify correctness under format_tag::any
// ===========================================================================

TEST(OneDNNFormatAny, AvgPool2dKnownInput) {
    // 1×1×4×4, avg pool 2×2 stride=2 → 1×1×2×2
    auto t = ops::zeros({1, 1, 4, 4}, DType::Float32, Device::cpu());
    float* p = t.data<float>();
    for (int i = 0; i < 16; ++i) p[i] = static_cast<float>(i);

    Variable x(t, /*requires_grad=*/false);
    nn::AvgPool2d pool(/*kernel=*/2, /*stride=*/2);
    auto y = pool.forward(x);

    auto out = y.tensor().to(DType::Float32);
    const float* op = out.data<float>();

    // Expected avg of each 2×2 block:
    // top-left:   (0+1+4+5)/4   = 2.5
    // top-right:  (2+3+6+7)/4   = 4.5
    // bot-left:   (8+9+12+13)/4 = 10.5
    // bot-right:  (10+11+14+15)/4= 12.5
    EXPECT_NEAR(op[0], 2.5f,  0.1f) << "top-left";
    EXPECT_NEAR(op[1], 4.5f,  0.1f) << "top-right";
    EXPECT_NEAR(op[2], 10.5f, 0.1f) << "bot-left";
    EXPECT_NEAR(op[3], 12.5f, 0.1f) << "bot-right";
}

TEST(OneDNNFormatAny, AvgPool2dUniformInput) {
    // Uniform input → output must equal input value
    const float val = 3.14f;
    auto t = ops::full({2, 4, 8, 8}, val, DType::Float32, Device::cpu());

    Variable x(t, /*requires_grad=*/false);
    nn::AvgPool2d pool(2, 2, 0);
    auto y = pool.forward(x);

    auto out = y.tensor().to(DType::Float32);
    const float* op = out.data<float>();
    int64_t n = out.numel();
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(op[i], val, 0.01f) << "element " << i;
        if (HasFatalFailure()) FAIL();
    }
}

TEST(OneDNNFormatAny, AvgPool2dLargerBatch) {
    // Batch=4, C=16, H=W=32 → 2×2 pool, stride=2 → H=W=16
    // Fill with ones → output should all be 1.0f
    auto t = ops::ones({4, 16, 32, 32}, DType::Float32, Device::cpu());
    Variable x(t, /*requires_grad=*/false);
    nn::AvgPool2d pool(2, 2, 0);
    auto y = pool.forward(x);

    auto out = y.tensor().to(DType::Float32);
    const float* op = out.data<float>();
    int64_t n = out.numel();
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(op[i], 1.0f, 0.01f);
        if (HasFatalFailure()) FAIL();
    }
}

// ===========================================================================
// Linear — verify correctness under format_tag::any
// ===========================================================================

TEST(OneDNNFormatAny, LinearForwardSmall) {
    // input 2×4, weight 3×4, bias 3 → output 2×3
    auto input_t  = ops::zeros({2, 4},  DType::Float32, Device::cpu());
    auto weight_t = ops::zeros({3, 4},  DType::Float32, Device::cpu());
    auto bias_t   = ops::zeros({3},     DType::Float32, Device::cpu());

    // input: row0=[1,0,0,0] row1=[0,1,0,0]
    float* ip = input_t.data<float>();
    ip[0] = 1.0f; ip[5] = 1.0f;

    // weight: [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
    float* wp = weight_t.data<float>();
    for (int i = 0; i < 12; ++i) wp[i] = static_cast<float>(i + 1);

    // bias: zeros → no offset

    // nn::Linear stores weight as (out_features, in_features)
    // Forward: output = input * weight^T + bias
    // row0: [1*1+0*2+0*3+0*4, 1*5+...+0*8, 1*9+...+0*12] = [1, 5, 9]
    // row1: [0*1+1*2+0*3+0*4, 0*5+1*6+..., 0*9+1*10+...] = [2, 6, 10]

    Variable x(input_t, false);
    nn::Linear linear(4, 3, /*bias=*/true);
    // Override weight and bias with our known values
    *linear.weight() = Variable(weight_t, false);
    if (linear.has_bias()) *linear.bias() = Variable(bias_t, false);

    auto y = linear.forward(x);
    auto out = y.tensor().to(DType::Float32);
    const float* op = out.data<float>();

    EXPECT_NEAR(op[0], 1.0f,  0.1f) << "out[0,0]";
    EXPECT_NEAR(op[1], 5.0f,  0.1f) << "out[0,1]";
    EXPECT_NEAR(op[2], 9.0f,  0.1f) << "out[0,2]";
    EXPECT_NEAR(op[3], 2.0f,  0.1f) << "out[1,0]";
    EXPECT_NEAR(op[4], 6.0f,  0.1f) << "out[1,1]";
    EXPECT_NEAR(op[5], 10.0f, 0.1f) << "out[1,2]";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
