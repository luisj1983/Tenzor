#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "tenzor/ops/advanced.hpp"
#include "tenzor/ops/creation.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

class MatrixConstructionTest : public BackendTest {};

// Helper
static auto make_tensor(std::vector<float> vals, std::vector<int64_t> shape, Device dev) -> Tensor {
    auto t = Tensor::from_blob(vals.data(), shape, DType::Float32, Device::cpu());
    auto out = t.clone();
    if (dev.type != Device::Type::CPU) out = out.to(dev);
    return out;
}

static auto make_1d(std::vector<float> vals, Device dev) -> Tensor {
    return make_tensor(vals, {static_cast<int64_t>(vals.size())}, dev);
}

TEST_P(MatrixConstructionTest, Kron_Identity) {
    // kron(I2, I2) = I4
    auto I2 = tenzor::eye(2, std::nullopt, DType::Float32, device);
    auto result = tenzor::kron(I2, I2).to(Device::cpu());
    auto I4 = tenzor::eye(4, std::nullopt, DType::Float32, Device::cpu());
    auto* r = result.data<float>();
    auto* e = I4.data<float>();
    for (int i = 0; i < 16; ++i) {
        EXPECT_NEAR(r[i], e[i], 1e-5) << "kron identity mismatch at " << i;
    }
}

TEST_P(MatrixConstructionTest, Kron_Scalar) {
    // kron([[2]], [[3,4]]) = [[6,8]]
    auto a = make_tensor({2.0f}, {1, 1}, device);
    auto b = make_tensor({3.0f, 4.0f}, {1, 2}, device);
    auto result = tenzor::kron(a, b).to(Device::cpu());
    EXPECT_EQ(result.shape()[0], 1);
    EXPECT_EQ(result.shape()[1], 2);
    auto* r = result.data<float>();
    EXPECT_NEAR(r[0], 6.0f, 1e-5);
    EXPECT_NEAR(r[1], 8.0f, 1e-5);
}

TEST_P(MatrixConstructionTest, BlockDiag) {
    auto a = make_tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, device);
    auto b = make_tensor({5.0f, 6.0f}, {1, 2}, device);
    std::vector<Tensor> tensors = {a, b};
    auto result = tenzor::block_diag(tensors).to(Device::cpu());

    // Should be 3x4
    EXPECT_EQ(result.shape()[0], 3);
    EXPECT_EQ(result.shape()[1], 4);
    auto* r = result.data<float>();
    // Row 0: [1, 2, 0, 0]
    EXPECT_NEAR(r[0], 1.0f, 1e-5);
    EXPECT_NEAR(r[1], 2.0f, 1e-5);
    EXPECT_NEAR(r[2], 0.0f, 1e-5);
    EXPECT_NEAR(r[3], 0.0f, 1e-5);
    // Row 2: [0, 0, 5, 6]
    EXPECT_NEAR(r[8], 0.0f, 1e-5);
    EXPECT_NEAR(r[9], 0.0f, 1e-5);
    EXPECT_NEAR(r[10], 5.0f, 1e-5);
    EXPECT_NEAR(r[11], 6.0f, 1e-5);
}

TEST_P(MatrixConstructionTest, Vander) {
    auto x = make_1d({1.0f, 2.0f, 3.0f}, device);
    auto result = tenzor::vander(x, 3, true).to(Device::cpu());
    // increasing=true: [[1, 1, 1], [1, 2, 4], [1, 3, 9]]
    EXPECT_EQ(result.shape()[0], 3);
    EXPECT_EQ(result.shape()[1], 3);
    auto* r = result.data<float>();
    EXPECT_NEAR(r[0], 1.0f, 1e-5);  // 1^0
    EXPECT_NEAR(r[1], 1.0f, 1e-5);  // 1^1
    EXPECT_NEAR(r[2], 1.0f, 1e-5);  // 1^2
    EXPECT_NEAR(r[3], 1.0f, 1e-5);  // 2^0
    EXPECT_NEAR(r[4], 2.0f, 1e-5);  // 2^1
    EXPECT_NEAR(r[5], 4.0f, 1e-5);  // 2^2
    EXPECT_NEAR(r[6], 1.0f, 1e-5);  // 3^0
    EXPECT_NEAR(r[7], 3.0f, 1e-5);  // 3^1
    EXPECT_NEAR(r[8], 9.0f, 1e-4);  // 3^2
}

TEST_P(MatrixConstructionTest, Combinations) {
    auto x = make_1d({10.0f, 20.0f, 30.0f}, device);
    auto result = tenzor::combinations(x, 2).to(Device::cpu());
    // C(3,2) = 3 combinations: (10,20), (10,30), (20,30)
    EXPECT_EQ(result.shape()[0], 3);
    EXPECT_EQ(result.shape()[1], 2);
    auto* r = result.data<float>();
    EXPECT_NEAR(r[0], 10.0f, 1e-5);
    EXPECT_NEAR(r[1], 20.0f, 1e-5);
    EXPECT_NEAR(r[2], 10.0f, 1e-5);
    EXPECT_NEAR(r[3], 30.0f, 1e-5);
    EXPECT_NEAR(r[4], 20.0f, 1e-5);
    EXPECT_NEAR(r[5], 30.0f, 1e-5);
}

INSTANTIATE_BACKEND_TESTS(MatrixConstructionTest);
