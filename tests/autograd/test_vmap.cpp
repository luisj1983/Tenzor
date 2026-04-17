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
#include "../backend_test_fixture.hpp"
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

// ============================================================================
// Backend-parameterized variants (plan 4.1)
//
// The legacy TEST_F tests above stay CPU-only. This fixture runs the same
// core vmap semantics on every available backend, skipping unsupported ones.
// ============================================================================

class VmapBackendTest : public tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        BackendTest::SetUp();
        set_grad_enabled(true);
    }
};

TEST_P(VmapBackendTest, VmapSquare_CrossBackend) {
    auto data = tenzor::ones({4, 3}, DType::Float32, device);
    Variable input(data, false);

    auto f = [](const Variable& x) -> Variable {
        auto t = x.tensor();
        return Variable(tenzor::mul(t, t), false);
    };

    auto result = vmap(f, input, 0);
    device.synchronize();
    auto result_shape = result.tensor().shape();
    EXPECT_EQ(result_shape[0], 4);
    EXPECT_EQ(result_shape[1], 3);
    // ones squared = ones
    auto result_cpu = result.tensor().to(Device::cpu());
    float max_diff = *tenzor::max(
        tenzor::abs(tenzor::sub(result_cpu, tenzor::ones({4, 3},
                                DType::Float32, Device::cpu())))).data<float>();
    EXPECT_LT(max_diff, 1e-5f);
}

// ============================================================================
// Phase 7 expansion: nested vmap
// ============================================================================

// vmap over two axes should produce the same result as two nested vmaps,
// and a pointwise op ought to come back with both leading dims preserved
// intact. Catches regressions where the outer vmap collapses an inner
// vmap's axis.
TEST_F(VmapTest, NestedVmapPreservesBothDims) {
    auto data = tenzor::ones({3, 4, 5}, DType::Float32, Device::cpu());
    // Fill with a recognisable pattern so we can verify per-slice behaviour.
    auto* ptr = data.data<float>();
    for (int64_t i = 0; i < data.numel(); ++i) ptr[i] = static_cast<float>(i);
    Variable input(data, false);

    // Inner lambda squares the per-slice variable; outer vmap batches over
    // the outermost dim, inner vmap batches over the next dim down.
    auto inner_square = [](const Variable& x) -> Variable {
        auto t = x.tensor();
        return Variable(tenzor::mul(t, t), false);
    };
    auto outer = [&inner_square](const Variable& x) -> Variable {
        return vmap(inner_square, x, 0);
    };

    auto result = vmap(outer, input, 0);
    auto result_shape = result.tensor().shape();
    ASSERT_EQ(result_shape.size(), 3u);
    EXPECT_EQ(result_shape[0], 3);
    EXPECT_EQ(result_shape[1], 4);
    EXPECT_EQ(result_shape[2], 5);

    // Spot-check a couple of known elements.
    auto cpu = result.tensor().to(Device::cpu());
    auto* out = cpu.data<float>();
    EXPECT_FLOAT_EQ(out[0], 0.0f);           // 0² = 0
    EXPECT_FLOAT_EQ(out[cpu.numel() - 1],
                    static_cast<float>((data.numel() - 1) * (data.numel() - 1)));
}

INSTANTIATE_BACKEND_TESTS(VmapBackendTest);
