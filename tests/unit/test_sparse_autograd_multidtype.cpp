/**
 * @file test_sparse_autograd_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for sparse autograd operations
 */

#include "../multi_backend_dtype_fixture.hpp"
#include "tenzor/sparse/sparse_tensor.hpp"
#include "tenzor/sparse/sparse_ops.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/nn/loss/losses.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class SparseAutogradMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();
        // Skip if sparse ops are not available on this backend
        try {
            auto indices = Tensor({2, int64_t(1)}, DType::Int64, Device::cpu());
            auto values = Tensor({int64_t(1)}, DType::Float32, Device::cpu());
            indices.data<int64_t>()[0] = 0;
            indices.data<int64_t>()[1] = 0;
            values.data<float>()[0] = 1.0f;
            auto sp = SparseTensor::sparse_coo(indices, values, {2, 2});
            sp.to_dense();
        } catch (...) {
            GTEST_SKIP() << "Sparse ops not available on " << backend_name();
        }
    }

    // Helper: create a 3x4 CSR sparse matrix
    SparseTensor make_test_sparse() {
        auto crow = Tensor({int64_t(4)}, DType::Int64, Device::cpu());
        auto col = Tensor({int64_t(4)}, DType::Int64, Device::cpu());
        auto vals_f32 = Tensor({int64_t(4)}, DType::Float32, Device::cpu());

        auto* cp = crow.data<int64_t>();
        cp[0] = 0; cp[1] = 1; cp[2] = 3; cp[3] = 4;

        auto* colp = col.data<int64_t>();
        colp[0] = 0; colp[1] = 1; colp[2] = 2; colp[3] = 3;

        auto* vp = vals_f32.data<float>();
        vp[0] = 1.0f; vp[1] = 2.0f; vp[2] = 3.0f; vp[3] = 4.0f;

        auto vals = vals_f32.to(dtype());
        return SparseTensor::sparse_csr(crow, col, vals, {3, 4});
    }
};

TEST_P(SparseAutogradMultiDTypeTest, SpMMBackwardGradExists) {
    auto sparse = make_test_sparse();
    auto dense_tensor = randn({4, 2}, DType::Float32, device()).to(dtype());
    Variable dense(dense_tensor, true);

    auto result = spmm(sparse, dense);
    EXPECT_EQ(result.tensor().shape()[0], 3);
    EXPECT_EQ(result.tensor().shape()[1], 2);

    auto target = Variable(zeros({3, 2}, dtype(), device()), false);
    nn::MSELoss loss_fn;
    auto loss = loss_fn(result, target);
    loss.backward();

    EXPECT_TRUE(dense.grad().has_value());
    if (dense.grad().has_value()) {
        auto& grad_tensor = dense.grad().value();
        EXPECT_EQ(grad_tensor.shape()[0], 4);
        EXPECT_EQ(grad_tensor.shape()[1], 2);
    }
}

TEST_P(SparseAutogradMultiDTypeTest, SpMVBackwardGradExists) {
    auto sparse = make_test_sparse();
    auto vec_tensor = randn({4}, DType::Float32, device()).to(dtype());
    Variable vec(vec_tensor, true);

    auto result = spmv(sparse, vec);
    EXPECT_EQ(result.tensor().shape()[0], 3);

    auto target = Variable(zeros({3}, dtype(), device()), false);
    nn::MSELoss loss_fn;
    auto loss = loss_fn(result, target);
    loss.backward();

    EXPECT_TRUE(vec.grad().has_value());
    if (vec.grad().has_value()) {
        EXPECT_EQ(vec.grad().value().shape()[0], 4);
    }
}

TEST_P(SparseAutogradMultiDTypeTest, SparseAddBackwardGradExists) {
    auto sparse = make_test_sparse();
    auto dense_tensor = randn({3, 4}, DType::Float32, device()).to(dtype());
    Variable dense(dense_tensor, true);

    auto result = sparse_add(sparse, dense);
    EXPECT_EQ(result.tensor().shape()[0], 3);
    EXPECT_EQ(result.tensor().shape()[1], 4);

    auto target = Variable(zeros({3, 4}, dtype(), device()), false);
    nn::MSELoss loss_fn;
    auto loss = loss_fn(result, target);
    loss.backward();

    EXPECT_TRUE(dense.grad().has_value());
    if (dense.grad().has_value()) {
        EXPECT_EQ(dense.grad().value().shape()[0], 3);
        EXPECT_EQ(dense.grad().value().shape()[1], 4);
    }
}

TEST_P(SparseAutogradMultiDTypeTest, SpMMNoGradWhenNotRequired) {
    auto sparse = make_test_sparse();
    auto dense_tensor = randn({4, 2}, DType::Float32, device()).to(dtype());
    Variable dense(dense_tensor, false);

    auto result = spmm(sparse, dense);
    EXPECT_FALSE(result.requires_grad());
    EXPECT_EQ(result.grad_fn(), nullptr);
}

TEST_P(SparseAutogradMultiDTypeTest, SpMMOutputShape) {
    auto sparse = make_test_sparse();
    auto dense_tensor = randn({4, 5}, DType::Float32, device()).to(dtype());
    Variable dense(dense_tensor, false);

    auto result = spmm(sparse, dense);
    EXPECT_EQ(result.tensor().shape()[0], 3);
    EXPECT_EQ(result.tensor().shape()[1], 5);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(SparseAutogradMultiDTypeTest);
