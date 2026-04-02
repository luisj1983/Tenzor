/**
 * @file test_vmap.cpp
 * @brief Tests for vectorized map (vmap) transform
 */

#include <gtest/gtest.h>
#include "tenzor/autograd/vmap.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/tenzor.hpp"
#include <cmath>

using namespace tenzor;

class VmapTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        tenzor::initialize();
    }

    void SetUp() override {
        set_grad_enabled(true);
    }
};

TEST_F(VmapTest, VmapIdentity) {
    // vmap(identity) should return the same tensor
    auto data = tenzor::ones({4, 3}, DType::Float32, Device::cpu());
    Variable input(data, false);

    auto f = [](const Variable& x) -> Variable {
        return x;
    };

    auto result = vmap(f, input, 0);
    auto result_shape = result.tensor().shape();
    EXPECT_EQ(result_shape[0], 4);
    EXPECT_EQ(result_shape[1], 3);

    // Values should match
    auto diff = tenzor::abs(tenzor::sub(result.tensor(), data));
    float max_diff = *tenzor::max(diff).data<float>();
    EXPECT_LT(max_diff, 1e-6f);
}

TEST_F(VmapTest, VmapSquare) {
    // vmap(x -> x*x) should give element-wise squaring along batch
    auto data = tenzor::zeros({3, 2}, DType::Float32, Device::cpu());
    float* dp = data.data<float>();
    dp[0] = 1.0f; dp[1] = 2.0f;
    dp[2] = 3.0f; dp[3] = 4.0f;
    dp[4] = 5.0f; dp[5] = 6.0f;

    Variable input(data, false);

    auto f = [](const Variable& x) -> Variable {
        return Variable(tenzor::mul(x.tensor(), x.tensor()), false);
    };

    auto result = vmap(f, input, 0);

    // Compare with manual loop
    std::vector<Tensor> manual_results;
    for (int64_t i = 0; i < 3; ++i) {
        auto slice = tenzor::select(data, 0, i);
        manual_results.push_back(tenzor::mul(slice, slice));
    }
    auto expected = tenzor::stack(std::span<const Tensor>(manual_results), 0);

    auto diff = tenzor::abs(tenzor::sub(result.tensor(), expected));
    float max_diff = *tenzor::max(diff).data<float>();
    EXPECT_LT(max_diff, 1e-6f);
}

TEST_F(VmapTest, VmapSum) {
    // vmap(x -> sum(x)) reduces each row
    auto data = tenzor::ones({5, 4}, DType::Float32, Device::cpu());
    Variable input(data, false);

    auto f = [](const Variable& x) -> Variable {
        return Variable(tenzor::sum(x.tensor()), false);
    };

    auto result = vmap(f, input, 0);

    // Each row sum should be 4.0
    auto result_shape = result.tensor().shape();
    EXPECT_EQ(result_shape[0], 5);

    float* rp = result.tensor().data<float>();
    for (int64_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(rp[i], 4.0f, 1e-5f);
    }
}

TEST_F(VmapTest, VmapMatchesManualLoop) {
    // More complex function: f(x) = exp(x) + x^2
    auto data = tenzor::zeros({4, 3}, DType::Float32, Device::cpu());
    float* dp = data.data<float>();
    for (int i = 0; i < 12; ++i) {
        dp[i] = static_cast<float>(i) * 0.1f;
    }

    Variable input(data, false);

    auto f = [](const Variable& x) -> Variable {
        auto t = x.tensor();
        return Variable(tenzor::add(tenzor::exp(t), tenzor::mul(t, t)), false);
    };

    auto vmapped = vmap(f, input, 0);

    // Compare with manual loop
    std::vector<Tensor> manual_results;
    for (int64_t i = 0; i < 4; ++i) {
        auto slice = tenzor::select(data, 0, i);
        manual_results.push_back(tenzor::add(tenzor::exp(slice), tenzor::mul(slice, slice)));
    }
    auto expected = tenzor::stack(std::span<const Tensor>(manual_results), 0);

    auto diff = tenzor::abs(tenzor::sub(vmapped.tensor(), expected));
    float max_diff = *tenzor::max(diff).data<float>();
    EXPECT_LT(max_diff, 1e-5f);
}
