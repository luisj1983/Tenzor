#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/sparse/sparse_tensor.hpp"
#include "tenzor/sparse/sparse_ops.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/nn/loss/losses.hpp"
#include "../grad_flow_helpers.hpp"
#include "../backend_test_fixture.hpp"

using namespace tenzor;

class SparseAutograd : public ::tenzor::testing::BackendTest {};

// Helper: create a simple 3x4 CSR sparse matrix on the given device.
// [[1, 0, 0, 0],
//  [0, 2, 3, 0],
//  [0, 0, 0, 4]]
// CSR components are built on CPU (host writes), then moved to `device`.
static auto make_test_sparse(const tenzor::Device& device) -> SparseTensor {
    auto crow = Tensor({int64_t(4)}, DType::Int64, Device::cpu());
    auto col = Tensor({int64_t(4)}, DType::Int64, Device::cpu());
    auto vals = Tensor({int64_t(4)}, DType::Float32, Device::cpu());

    auto* cp = crow.data<int64_t>();
    cp[0] = 0; cp[1] = 1; cp[2] = 3; cp[3] = 4;

    auto* colp = col.data<int64_t>();
    colp[0] = 0; colp[1] = 1; colp[2] = 2; colp[3] = 3;

    auto* vp = vals.data<float>();
    vp[0] = 1.0f; vp[1] = 2.0f; vp[2] = 3.0f; vp[3] = 4.0f;

    return SparseTensor::sparse_csr(crow.to(device), col.to(device),
                                    vals.to(device), {3, 4});
}

// ============================================================================
// SpMM Backward Tests
// ============================================================================

TEST_P(SparseAutograd, SpMMBackwardGradExists) {
    auto sparse = make_test_sparse(device);  // 3x4
    auto dense_tensor = randn({4, 2}, DType::Float32, device);  // 4x2
    Variable dense(dense_tensor, true);

    // Forward: Y = S @ D -> shape 3x2
    auto result = spmm(sparse, dense);
    EXPECT_EQ(result.tensor().shape()[0], 3);
    EXPECT_EQ(result.tensor().shape()[1], 2);

    // Create scalar loss for backward
    auto target = Variable(zeros({3, 2}, DType::Float32, device), false);
    nn::MSELoss loss_fn;
    auto loss = loss_fn(result, target);
    loss.backward();

    // Dense input should have gradient
    EXPECT_GRAD_FLOWS(dense);
    if (dense.grad().has_value()) {
        auto grad_tensor = dense.grad().value();
        EXPECT_EQ(grad_tensor.shape()[0], 4);
        EXPECT_EQ(grad_tensor.shape()[1], 2);
    }
}

TEST_P(SparseAutograd, SpMVBackwardGradExists) {
    auto sparse = make_test_sparse(device);  // 3x4
    auto vec_tensor = randn({4}, DType::Float32, device);
    Variable vec(vec_tensor, true);

    // Forward: y = S @ v -> shape 3
    auto result = spmv(sparse, vec);
    EXPECT_EQ(result.tensor().shape()[0], 3);

    // Create scalar loss
    auto target = Variable(zeros({3}, DType::Float32, device), false);
    nn::MSELoss loss_fn;
    auto loss = loss_fn(result, target);
    loss.backward();

    EXPECT_GRAD_FLOWS(vec);
    if (vec.grad().has_value()) {
        auto grad_tensor = vec.grad().value();
        EXPECT_EQ(grad_tensor.shape()[0], 4);
    }
}

TEST_P(SparseAutograd, SparseAddBackwardGradExists) {
    auto sparse = make_test_sparse(device);  // 3x4
    auto dense_tensor = randn({3, 4}, DType::Float32, device);
    Variable dense(dense_tensor, true);

    // Forward: Y = S + D -> shape 3x4
    auto result = sparse_add(sparse, dense);
    EXPECT_EQ(result.tensor().shape()[0], 3);
    EXPECT_EQ(result.tensor().shape()[1], 4);

    // Create scalar loss
    auto target = Variable(zeros({3, 4}, DType::Float32, device), false);
    nn::MSELoss loss_fn;
    auto loss = loss_fn(result, target);
    loss.backward();

    EXPECT_GRAD_FLOWS(dense);
    if (dense.grad().has_value()) {
        auto grad_tensor = dense.grad().value();
        EXPECT_EQ(grad_tensor.shape()[0], 3);
        EXPECT_EQ(grad_tensor.shape()[1], 4);
    }
}

TEST_P(SparseAutograd, SpMMNoGradWhenNotRequired) {
    auto sparse = make_test_sparse(device);
    auto dense_tensor = randn({4, 2}, DType::Float32, device);
    Variable dense(dense_tensor, false);  // requires_grad = false

    auto result = spmm(sparse, dense);
    EXPECT_FALSE(result.requires_grad());
    EXPECT_EQ(result.grad_fn(), nullptr);
}

INSTANTIATE_BACKEND_TESTS(SparseAutograd);
