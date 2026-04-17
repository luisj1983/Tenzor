/**
 * @file test_matrix_construction_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for matrix construction ops:
 *        kron, block_diag, vander, combinations
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include <tenzor/ops/advanced.hpp>
#include <tenzor/ops/creation.hpp>
#include <cmath>
#include <cstring>

using namespace tenzor;
using namespace tenzor::testing;

class MatrixConstructionMultiDTypeTest : public MultiBackendDTypeTest {};

// Helper: build tensor from float data, then cast to test dtype/device
static Tensor make_tensor(const float* data, std::vector<int64_t> shape,
                          Device dev, DType dt) {
    auto t = Tensor::from_blob(const_cast<float*>(data), shape,
                               DType::Float32, Device::cpu()).clone();
    return t.to(dt).to(dev);
}

TEST_P(MatrixConstructionMultiDTypeTest, KronIdentity) {
    auto I2 = eye(2, std::nullopt, dtype(), device());
    auto result = kron(I2, I2);
    auto I4 = eye(4, std::nullopt, DType::Float32, Device::cpu());

    auto r_cpu = result.to(Device::cpu()).to(DType::Float32);
    auto* r = r_cpu.data<float>();
    auto* e = I4.data<float>();
    for (int i = 0; i < 16; ++i) {
        EXPECT_NEAR(r[i], e[i], atol()) << "kron identity mismatch at " << i;
    }
}

TEST_P(MatrixConstructionMultiDTypeTest, KronScalar) {
    float a_data[] = {2.0f};
    float b_data[] = {3.0f, 4.0f};
    auto a = make_tensor(a_data, {1, 1}, device(), dtype());
    auto b = make_tensor(b_data, {1, 2}, device(), dtype());
    auto result = kron(a, b);

    auto r_cpu = result.to(Device::cpu()).to(DType::Float32);
    EXPECT_EQ(r_cpu.shape()[0], 1);
    EXPECT_EQ(r_cpu.shape()[1], 2);
    auto* r = r_cpu.data<float>();
    EXPECT_NEAR(r[0], 6.0f, atol());
    EXPECT_NEAR(r[1], 8.0f, atol());
}

TEST_P(MatrixConstructionMultiDTypeTest, BlockDiag) {
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {5.0f, 6.0f};
    auto a = make_tensor(a_data, {2, 2}, device(), dtype());
    auto b = make_tensor(b_data, {1, 2}, device(), dtype());
    std::vector<Tensor> tensors = {a, b};
    auto result = block_diag(tensors);

    auto r_cpu = result.to(Device::cpu()).to(DType::Float32);
    EXPECT_EQ(r_cpu.shape()[0], 3);
    EXPECT_EQ(r_cpu.shape()[1], 4);
    auto* r = r_cpu.data<float>();
    EXPECT_NEAR(r[0], 1.0f, atol());
    EXPECT_NEAR(r[1], 2.0f, atol());
    EXPECT_NEAR(r[2], 0.0f, atol());
    EXPECT_NEAR(r[10], 5.0f, atol());
    EXPECT_NEAR(r[11], 6.0f, atol());
}

TEST_P(MatrixConstructionMultiDTypeTest, Vander) {
    float x_data[] = {1.0f, 2.0f, 3.0f};
    auto x = make_tensor(x_data, {3}, device(), dtype());
    auto result = vander(x, 3, true);

    auto r_cpu = result.to(Device::cpu()).to(DType::Float32);
    EXPECT_EQ(r_cpu.shape()[0], 3);
    EXPECT_EQ(r_cpu.shape()[1], 3);
    auto* r = r_cpu.data<float>();
    // Row 0: [1, 1, 1], Row 1: [1, 2, 4], Row 2: [1, 3, 9]
    EXPECT_NEAR(r[0], 1.0f, atol());
    EXPECT_NEAR(r[4], 2.0f, atol());
    EXPECT_NEAR(r[5], 4.0f, atol());
    EXPECT_NEAR(r[8], 9.0f, atol());
}

TEST_P(MatrixConstructionMultiDTypeTest, Combinations) {
    float x_data[] = {10.0f, 20.0f, 30.0f};
    auto x = make_tensor(x_data, {3}, device(), dtype());
    auto result = combinations(x, 2);

    auto r_cpu = result.to(Device::cpu()).to(DType::Float32);
    EXPECT_EQ(r_cpu.shape()[0], 3);
    EXPECT_EQ(r_cpu.shape()[1], 2);
    auto* r = r_cpu.data<float>();
    EXPECT_NEAR(r[0], 10.0f, atol());
    EXPECT_NEAR(r[1], 20.0f, atol());
    EXPECT_NEAR(r[4], 20.0f, atol());
    EXPECT_NEAR(r[5], 30.0f, atol());
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(MatrixConstructionMultiDTypeTest);
