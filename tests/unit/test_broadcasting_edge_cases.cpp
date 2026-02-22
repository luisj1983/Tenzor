/**
 * @file test_broadcasting_edge_cases.cpp
 * @brief Tests for broadcasting edge cases
 *
 * Covers:
 * - (1,1,1,N) vs (N,) broadcasting
 * - Scalar broadcasting
 * - Backward gradient reduction through broadcast
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include <cmath>

using namespace tenzor;

class BroadcastingEdgeCaseTest : public ::testing::Test {
protected:
    static bool initialized;
    void SetUp() override {
        if (!initialized) {
            tenzor::initialize();
            initialized = true;
        }
    }
};

bool BroadcastingEdgeCaseTest::initialized = false;

// ============================================================================
// 1. High-Rank to Low-Rank Broadcasting
// ============================================================================

TEST_F(BroadcastingEdgeCaseTest, Broadcast_1_1_1_N_Plus_N) {
    // (1, 1, 1, 4) + (4,) should broadcast to (1, 1, 1, 4)
    auto a = ones({1, 1, 1, 4}, DType::Float32, Device::cpu());
    auto* a_data = a.data<float>();
    a_data[0] = 1.0f; a_data[1] = 2.0f; a_data[2] = 3.0f; a_data[3] = 4.0f;

    auto b = ones({4}, DType::Float32, Device::cpu());
    auto* b_data = b.data<float>();
    b_data[0] = 10.0f; b_data[1] = 20.0f; b_data[2] = 30.0f; b_data[3] = 40.0f;

    auto c = add(a, b);

    EXPECT_EQ(c.ndim(), 4);
    EXPECT_EQ(c.shape()[0], 1);
    EXPECT_EQ(c.shape()[1], 1);
    EXPECT_EQ(c.shape()[2], 1);
    EXPECT_EQ(c.shape()[3], 4);

    auto* c_data = c.to(Device::cpu()).data<float>();
    EXPECT_FLOAT_EQ(c_data[0], 11.0f);
    EXPECT_FLOAT_EQ(c_data[1], 22.0f);
    EXPECT_FLOAT_EQ(c_data[2], 33.0f);
    EXPECT_FLOAT_EQ(c_data[3], 44.0f);
}

TEST_F(BroadcastingEdgeCaseTest, Broadcast_N_Plus_1_1_1_N) {
    // (4,) + (1, 1, 1, 4) should broadcast to (1, 1, 1, 4)
    auto a = ones({4}, DType::Float32, Device::cpu());
    auto* a_data = a.data<float>();
    a_data[0] = 10.0f; a_data[1] = 20.0f; a_data[2] = 30.0f; a_data[3] = 40.0f;

    auto b = ones({1, 1, 1, 4}, DType::Float32, Device::cpu());
    auto* b_data = b.data<float>();
    b_data[0] = 1.0f; b_data[1] = 2.0f; b_data[2] = 3.0f; b_data[3] = 4.0f;

    auto c = add(a, b);

    EXPECT_EQ(c.shape()[3], 4);

    auto c_contig = c.contiguous().to(Device::cpu());
    auto* c_data = c_contig.data<float>();
    EXPECT_FLOAT_EQ(c_data[0], 11.0f);
    EXPECT_FLOAT_EQ(c_data[1], 22.0f);
    EXPECT_FLOAT_EQ(c_data[2], 33.0f);
    EXPECT_FLOAT_EQ(c_data[3], 44.0f);
}

// ============================================================================
// 2. Scalar Broadcasting
// ============================================================================

TEST_F(BroadcastingEdgeCaseTest, ScalarAddToMatrix) {
    // (1,) + (2, 3) should broadcast to (2, 3)
    auto scalar = ones({1}, DType::Float32, Device::cpu());
    scalar.data<float>()[0] = 5.0f;

    auto matrix = zeros({2, 3}, DType::Float32, Device::cpu());
    auto* m_data = matrix.data<float>();
    for (int i = 0; i < 6; ++i) {
        m_data[i] = static_cast<float>(i + 1);
    }

    auto result = add(matrix, scalar);
    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.shape()[1], 3);

    auto* r_data = result.to(Device::cpu()).data<float>();
    EXPECT_FLOAT_EQ(r_data[0], 6.0f);   // 1 + 5
    EXPECT_FLOAT_EQ(r_data[1], 7.0f);   // 2 + 5
    EXPECT_FLOAT_EQ(r_data[2], 8.0f);   // 3 + 5
    EXPECT_FLOAT_EQ(r_data[3], 9.0f);   // 4 + 5
    EXPECT_FLOAT_EQ(r_data[4], 10.0f);  // 5 + 5
    EXPECT_FLOAT_EQ(r_data[5], 11.0f);  // 6 + 5
}

TEST_F(BroadcastingEdgeCaseTest, ScalarMulToMatrix) {
    auto scalar = ones({1}, DType::Float32, Device::cpu());
    scalar.data<float>()[0] = 3.0f;

    auto matrix = ones({2, 4}, DType::Float32, Device::cpu());
    auto* m_data = matrix.data<float>();
    for (int i = 0; i < 8; ++i) {
        m_data[i] = static_cast<float>(i + 1);
    }

    auto result = mul(matrix, scalar);
    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.shape()[1], 4);

    auto* r_data = result.to(Device::cpu()).data<float>();
    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(r_data[i], static_cast<float>((i + 1) * 3));
    }
}

TEST_F(BroadcastingEdgeCaseTest, ScalarAddWithOperator) {
    // Test scalar add via the add(Tensor, double) overload
    auto t = ones({3, 3}, DType::Float32, Device::cpu());
    auto result = add(t, 2.0);
    auto* data = result.to(Device::cpu()).data<float>();
    for (int i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(data[i], 3.0f);
    }
}

// ============================================================================
// 3. Multi-Dimensional Broadcasting Patterns
// ============================================================================

TEST_F(BroadcastingEdgeCaseTest, Broadcast_N1_Plus_1M) {
    // (3, 1) + (1, 4) should broadcast to (3, 4)
    auto a = zeros({3, 1}, DType::Float32, Device::cpu());
    auto* a_data = a.data<float>();
    a_data[0] = 1.0f; a_data[1] = 2.0f; a_data[2] = 3.0f;

    auto b = zeros({1, 4}, DType::Float32, Device::cpu());
    auto* b_data = b.data<float>();
    b_data[0] = 10.0f; b_data[1] = 20.0f; b_data[2] = 30.0f; b_data[3] = 40.0f;

    auto c = add(a, b);
    EXPECT_EQ(c.shape()[0], 3);
    EXPECT_EQ(c.shape()[1], 4);

    auto* c_data = c.to(Device::cpu()).data<float>();
    // Row 0: 1 + [10, 20, 30, 40] = [11, 21, 31, 41]
    EXPECT_FLOAT_EQ(c_data[0], 11.0f);
    EXPECT_FLOAT_EQ(c_data[1], 21.0f);
    EXPECT_FLOAT_EQ(c_data[2], 31.0f);
    EXPECT_FLOAT_EQ(c_data[3], 41.0f);
    // Row 1: 2 + [10, 20, 30, 40] = [12, 22, 32, 42]
    EXPECT_FLOAT_EQ(c_data[4], 12.0f);
    EXPECT_FLOAT_EQ(c_data[5], 22.0f);
    EXPECT_FLOAT_EQ(c_data[6], 32.0f);
    EXPECT_FLOAT_EQ(c_data[7], 42.0f);
    // Row 2: 3 + [10, 20, 30, 40] = [13, 23, 33, 43]
    EXPECT_FLOAT_EQ(c_data[8], 13.0f);
    EXPECT_FLOAT_EQ(c_data[9], 23.0f);
    EXPECT_FLOAT_EQ(c_data[10], 33.0f);
    EXPECT_FLOAT_EQ(c_data[11], 43.0f);
}

TEST_F(BroadcastingEdgeCaseTest, Broadcast_Batch_Plus_NonBatch) {
    // (2, 3, 4) + (3, 4) -> (2, 3, 4)
    auto a = ones({2, 3, 4}, DType::Float32, Device::cpu());
    auto b = ones({3, 4}, DType::Float32, Device::cpu());
    auto* b_data = b.data<float>();
    for (int i = 0; i < 12; ++i) {
        b_data[i] = static_cast<float>(i);
    }

    auto c = add(a, b);
    EXPECT_EQ(c.ndim(), 3);
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 3);
    EXPECT_EQ(c.shape()[2], 4);

    auto* c_data = c.to(Device::cpu()).data<float>();
    // Both batches should get same broadcast result
    for (int batch = 0; batch < 2; ++batch) {
        for (int i = 0; i < 12; ++i) {
            EXPECT_FLOAT_EQ(c_data[batch * 12 + i], 1.0f + static_cast<float>(i));
        }
    }
}

// ============================================================================
// 4. Backward Gradient Reduction Through Broadcast
// ============================================================================

TEST_F(BroadcastingEdgeCaseTest, BackwardGradientReductionBroadcastAdd) {
    // x: (3, 4), b: (1, 4) (bias), y = x + b, loss = sum(y)
    // Gradient of b should be (3, 4) summed to (1, 4), i.e., each element = 3
    auto x_data = ones({3, 4}, DType::Float32, Device::cpu());
    auto b_data = ones({1, 4}, DType::Float32, Device::cpu());

    Variable x(x_data, true);
    Variable b(b_data, true);

    auto y = x + b;  // broadcasts (1,4) to (3,4)
    auto loss = tenzor::sum(y);
    loss.backward();

    // x gradient: all ones (gradient flows straight through add)
    ASSERT_TRUE(x.has_grad());
    auto x_grad = x.grad()->to(Device::cpu());
    auto* x_grad_data = x_grad.data<float>();
    for (int64_t i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(x_grad_data[i], 1.0f);
    }

    // b gradient: should be reduced from (3,4) to (1,4), each = 3.0
    ASSERT_TRUE(b.has_grad());
    auto b_grad = b.grad()->to(Device::cpu());
    auto* b_grad_data = b_grad.data<float>();
    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(b_grad_data[i], 3.0f);
    }
}

TEST_F(BroadcastingEdgeCaseTest, BackwardGradientReductionBroadcastMul) {
    // x: (2, 3), s: (1,) scalar, y = x * s, loss = sum(y)
    // dL/ds should be sum of x
    auto x_data = ones({2, 3}, DType::Float32, Device::cpu());
    auto* xp = x_data.data<float>();
    for (int i = 0; i < 6; ++i) {
        xp[i] = static_cast<float>(i + 1);  // [1,2,3,4,5,6]
    }

    auto s_data = full({1}, 2.0f, DType::Float32, Device::cpu());

    Variable x(x_data, true);
    Variable s(s_data, true);

    auto y = x * s;
    auto loss = tenzor::sum(y);
    loss.backward();

    // dL/dx = s = 2.0 for all elements
    ASSERT_TRUE(x.has_grad());
    auto x_grad = x.grad()->to(Device::cpu());
    auto* xg = x_grad.data<float>();
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(xg[i], 2.0f);
    }

    // dL/ds = sum(x) = 1+2+3+4+5+6 = 21
    ASSERT_TRUE(s.has_grad());
    auto s_grad = s.grad()->to(Device::cpu());
    EXPECT_FLOAT_EQ(s_grad.data<float>()[0], 21.0f);
}

// ============================================================================
// 5. Broadcasting Shape Correctness
// ============================================================================

TEST_F(BroadcastingEdgeCaseTest, Broadcast_1x1_Plus_MxN) {
    // (1, 1) + (3, 4) -> (3, 4)
    auto a = full({1, 1}, 10.0f, DType::Float32, Device::cpu());
    auto b = ones({3, 4}, DType::Float32, Device::cpu());
    auto c = add(a, b);
    EXPECT_EQ(c.shape()[0], 3);
    EXPECT_EQ(c.shape()[1], 4);

    auto* c_data = c.to(Device::cpu()).data<float>();
    for (int i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(c_data[i], 11.0f);
    }
}

TEST_F(BroadcastingEdgeCaseTest, Broadcast_1_Plus_1_1_1) {
    // (1,) + (1, 1, 1) -> (1, 1, 1)
    auto a = full({1}, 5.0f, DType::Float32, Device::cpu());
    auto b = full({1, 1, 1}, 3.0f, DType::Float32, Device::cpu());
    auto c = add(a, b);
    EXPECT_EQ(c.numel(), 1);
    auto c_contig = c.contiguous().to(Device::cpu());
    EXPECT_FLOAT_EQ(c_contig.data<float>()[0], 8.0f);
}
