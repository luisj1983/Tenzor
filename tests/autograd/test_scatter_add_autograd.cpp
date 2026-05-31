/**
 * @file test_scatter_add_autograd.cpp
 * @brief Tests for Variable-level scatter_add and higher-order gradient support
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/autograd/ops.hpp>
#include <tenzor/autograd/engine.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/indexing.hpp>
#include <cmath>
#include "../grad_flow_helpers.hpp"
#include "../backend_test_fixture.hpp"

using namespace tenzor;

class ScatterAddAutogradTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

// ============================================================================
// Basic scatter_add forward through Variable
// ============================================================================

TEST_P(ScatterAddAutogradTest, ForwardCorrectness) {
    auto input_t = zeros({3, 5}, DType::Float32, device);
    auto src_t = ones({3, 2}, DType::Float32, device);
    auto index_cpu = Tensor({3, 2}, DType::Int64, Device::cpu());
    auto* idx = index_cpu.data<int64_t>();
    // scatter src into columns 0,1 / 2,3 / 4,0
    idx[0] = 0; idx[1] = 1;
    idx[2] = 2; idx[3] = 3;
    idx[4] = 4; idx[5] = 0;
    auto index_t = index_cpu.to(device);

    Variable input(input_t, false);
    Variable src(src_t, false);

    auto result = scatter_add(input, 1, index_t, src);
    auto result_t = result.tensor().cpu();

    // Row 0: [1, 1, 0, 0, 0]
    EXPECT_FLOAT_EQ(result_t.data<float>()[0], 1.0f);
    EXPECT_FLOAT_EQ(result_t.data<float>()[1], 1.0f);
    EXPECT_FLOAT_EQ(result_t.data<float>()[2], 0.0f);

    // Row 1: [0, 0, 1, 1, 0]
    EXPECT_FLOAT_EQ(result_t.data<float>()[7], 1.0f);
    EXPECT_FLOAT_EQ(result_t.data<float>()[8], 1.0f);

    // Row 2: [1, 0, 0, 0, 1]
    EXPECT_FLOAT_EQ(result_t.data<float>()[10], 1.0f);
    EXPECT_FLOAT_EQ(result_t.data<float>()[14], 1.0f);
}

// ============================================================================
// First-order gradient of scatter_add
// ============================================================================

TEST_P(ScatterAddAutogradTest, FirstOrderGradient) {
    auto input_t = zeros({2, 4}, DType::Float32, device);
    auto src_cpu = Tensor({2, 2}, DType::Float32, Device::cpu());
    src_cpu.data<float>()[0] = 2.0f;
    src_cpu.data<float>()[1] = 3.0f;
    src_cpu.data<float>()[2] = 4.0f;
    src_cpu.data<float>()[3] = 5.0f;
    auto src_t = src_cpu.to(device);

    auto index_cpu = Tensor({2, 2}, DType::Int64, Device::cpu());
    auto* idx = index_cpu.data<int64_t>();
    idx[0] = 0; idx[1] = 2;
    idx[2] = 1; idx[3] = 3;
    auto index_t = index_cpu.to(device);

    Variable input(input_t, true);
    Variable src(src_t, true);

    auto result = scatter_add(input, 1, index_t, src);
    // result = [[2, 0, 3, 0], [0, 4, 0, 5]]

    // Sum to get scalar for backward
    auto loss = sum(result);
    loss.backward();

    // grad_input should be all ones (identity gradient)
    EXPECT_GRAD_FLOWS(input);
    auto grad_input = input.grad().value().cpu();
    for (int64_t i = 0; i < grad_input.numel(); ++i) {
        EXPECT_FLOAT_EQ(grad_input.data<float>()[i], 1.0f);
    }

    // grad_src should be all ones (gather from all-ones grad_output)
    EXPECT_GRAD_FLOWS(src);
    auto grad_src = src.grad().value().cpu();
    for (int64_t i = 0; i < grad_src.numel(); ++i) {
        EXPECT_FLOAT_EQ(grad_src.data<float>()[i], 1.0f);
    }
}

// ============================================================================
// scatter_add with scaled source (non-trivial gradient)
// ============================================================================

TEST_P(ScatterAddAutogradTest, ScaledSourceGradient) {
    auto input_t = ones({1, 3}, DType::Float32, device);
    auto src_cpu = Tensor({1, 2}, DType::Float32, Device::cpu());
    src_cpu.data<float>()[0] = 2.0f;
    src_cpu.data<float>()[1] = 3.0f;
    auto src_t = src_cpu.to(device);

    auto index_cpu = Tensor({1, 2}, DType::Int64, Device::cpu());
    index_cpu.data<int64_t>()[0] = 0;
    index_cpu.data<int64_t>()[1] = 2;
    auto index_t = index_cpu.to(device);

    Variable src(src_t, true);
    Variable input(input_t, true);

    // Scale src before scatter_add
    auto scaled_src = src * 2.0f;
    auto result = scatter_add(input, 1, index_t, scaled_src);
    // result = [[1+4, 1, 1+6]] = [[5, 1, 7]]

    auto loss = sum(result);  // = 13
    loss.backward();

    // grad through scatter_add: grad_src_scaled = gather(ones, dim=1, index) = [1, 1]
    // grad through mul: grad_src = grad_src_scaled * 2.0 = [2, 2]
    EXPECT_GRAD_FLOWS(src);
    auto grad_src = src.grad().value().cpu();
    EXPECT_FLOAT_EQ(grad_src.data<float>()[0], 2.0f);
    EXPECT_FLOAT_EQ(grad_src.data<float>()[1], 2.0f);
}

// ============================================================================
// No-grad path
// ============================================================================

TEST_P(ScatterAddAutogradTest, NoGradPath) {
    auto input_t = zeros({2, 3}, DType::Float32, device);
    auto src_t = ones({2, 1}, DType::Float32, device);
    auto index_cpu = Tensor({2, 1}, DType::Int64, Device::cpu());
    index_cpu.data<int64_t>()[0] = 1;
    index_cpu.data<int64_t>()[1] = 0;
    auto index_t = index_cpu.to(device);

    Variable input(input_t, false);
    Variable src(src_t, false);

    auto result = scatter_add(input, 1, index_t, src);
    EXPECT_FALSE(result.requires_grad());
    EXPECT_EQ(result.grad_fn(), nullptr);
}

// ============================================================================
// Higher-order gradient: gather -> backward uses scatter_add -> backward uses gather
// ============================================================================

TEST_P(ScatterAddAutogradTest, GatherHigherOrderGrad) {
    // gather backward uses scatter_add, scatter_add backward uses gather
    // This tests the full chain with create_graph=true
    auto input_cpu = Tensor({2, 3}, DType::Float32, Device::cpu());
    auto* data = input_cpu.data<float>();
    data[0] = 1.0f; data[1] = 2.0f; data[2] = 3.0f;
    data[3] = 4.0f; data[4] = 5.0f; data[5] = 6.0f;
    auto input_t = input_cpu.to(device);

    auto index_cpu = Tensor({2, 2}, DType::Int64, Device::cpu());
    auto* idx = index_cpu.data<int64_t>();
    idx[0] = 0; idx[1] = 2;
    idx[2] = 1; idx[3] = 0;
    auto index_t = index_cpu.to(device);

    Variable x(input_t, true);

    // Forward: gathered = x.gather(1, index) = [[1, 3], [5, 4]]
    auto gathered = gather(x, 1, index_t);
    auto loss = sum(gathered * gathered);  // sum of squares

    // First backward with create_graph=true for higher-order
    auto& engine = backward_engine();
    engine.execute(loss, {}, /*retain_graph=*/true, /*create_graph=*/true);

    // x.grad should exist and be correct
    ASSERT_TRUE(x.grad().has_value());
    auto first_grad = x.grad().value().cpu();

    // The gradient of sum(gather(x, idx)^2) w.r.t. x:
    // d/dx_ij = 2 * x_ij * (number of times (i,j) appears in index)
    // Row 0: idx selects cols 0,2 -> grad[0][0] = 2*1=2, grad[0][1]=0, grad[0][2]=2*3=6
    // Row 1: idx selects cols 1,0 -> grad[1][0] = 2*4=8, grad[1][1]=2*5=10, grad[1][2]=0
    EXPECT_FLOAT_EQ(first_grad.data<float>()[0], 2.0f);
    EXPECT_FLOAT_EQ(first_grad.data<float>()[1], 0.0f);
    EXPECT_FLOAT_EQ(first_grad.data<float>()[2], 6.0f);
    EXPECT_FLOAT_EQ(first_grad.data<float>()[3], 8.0f);
    EXPECT_FLOAT_EQ(first_grad.data<float>()[4], 10.0f);
    EXPECT_FLOAT_EQ(first_grad.data<float>()[5], 0.0f);
}

INSTANTIATE_BACKEND_TESTS(ScatterAddAutogradTest);
