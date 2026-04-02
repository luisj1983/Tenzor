/**
 * @file test_sparse_gpu.cpp
 * @brief GPU tests for sparse tensor operations.
 *
 * Tests SpMM, SpMV, sparse-dense roundtrip, and SparseAdd on GPU devices.
 * Automatically skips if no GPU (CUDA or ROCm) is available.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/sparse/sparse_tensor.hpp>
#include <tenzor/sparse/sparse_ops.hpp>
#include <cmath>

using namespace tenzor;

class SparseGPUTest : public ::testing::Test {
protected:
    void SetUp() override {
        initialize();
        // Try CUDA first, then ROCm
        try {
            device_ = Device::cuda(0);
            tenzor::zeros({2}, DType::Float32, device_);
        } catch (...) {
            try {
                device_ = Device::rocm(0);
                tenzor::zeros({2}, DType::Float32, device_);
            } catch (...) {
                GTEST_SKIP() << "No GPU device available";
            }
        }
    }
    Device device_ = Device::cpu();
};

// Helper: create a small sparse matrix in COO format on CPU, then transfer
static SparseTensor make_test_sparse(Device device) {
    // 4x4 sparse matrix with 5 non-zeros:
    //  [1 0 2 0]
    //  [0 3 0 0]
    //  [0 0 4 0]
    //  [5 0 0 6]
    auto indices = Tensor({2, 6}, DType::Int64, Device::cpu());
    auto* idx = indices.data<int64_t>();
    // row indices
    idx[0] = 0; idx[1] = 0; idx[2] = 1; idx[3] = 2; idx[4] = 3; idx[5] = 3;
    // col indices
    idx[6] = 0; idx[7] = 2; idx[8] = 1; idx[9] = 2; idx[10] = 0; idx[11] = 3;

    auto values = Tensor({6}, DType::Float32, Device::cpu());
    auto* v = values.data<float>();
    v[0] = 1.0f; v[1] = 2.0f; v[2] = 3.0f; v[3] = 4.0f; v[4] = 5.0f; v[5] = 6.0f;

    auto sparse = SparseTensor::sparse_coo(indices, values, {4, 4});
    return sparse.to(device);
}

// Helper: check two tensors are close (after moving to CPU)
static void expect_tensors_close(const Tensor& a, const Tensor& b, float atol = 1e-5f) {
    auto a_cpu = a.to(Device::cpu()).contiguous();
    auto b_cpu = b.to(Device::cpu()).contiguous();

    ASSERT_EQ(a_cpu.numel(), b_cpu.numel()) << "Tensor size mismatch";

    const float* a_data = a_cpu.data<float>();
    const float* b_data = b_cpu.data<float>();
    for (int64_t i = 0; i < a_cpu.numel(); ++i) {
        EXPECT_NEAR(a_data[i], b_data[i], atol)
            << "Mismatch at index " << i;
    }
}

// ============================================================================
// SpMM: Sparse @ Dense matrix multiplication
// ============================================================================

TEST_F(SparseGPUTest, SpMM_GPUvsCPU) {
    auto sparse_cpu = make_test_sparse(Device::cpu());
    auto sparse_gpu = make_test_sparse(device_);

    // Dense matrix 4x3
    auto dense_cpu = ones({4, 3}, DType::Float32, Device::cpu());
    auto* d = dense_cpu.data<float>();
    for (int64_t i = 0; i < 12; ++i) {
        d[i] = static_cast<float>(i + 1);
    }
    auto dense_gpu = dense_cpu.to(device_);

    auto result_cpu = sparse::spmm(sparse_cpu, dense_cpu);
    auto result_gpu = sparse::spmm(sparse_gpu, dense_gpu);

    expect_tensors_close(result_gpu, result_cpu);
}

// ============================================================================
// SpMV: Sparse @ Dense vector multiplication
// ============================================================================

TEST_F(SparseGPUTest, SpMV_GPUvsCPU) {
    auto sparse_cpu = make_test_sparse(Device::cpu());
    auto sparse_gpu = make_test_sparse(device_);

    auto vec_cpu = ones({4}, DType::Float32, Device::cpu());
    auto* v = vec_cpu.data<float>();
    v[0] = 1.0f; v[1] = 2.0f; v[2] = 3.0f; v[3] = 4.0f;
    auto vec_gpu = vec_cpu.to(device_);

    auto result_cpu = sparse::spmv(sparse_cpu, vec_cpu);
    auto result_gpu = sparse::spmv(sparse_gpu, vec_gpu);

    expect_tensors_close(result_gpu, result_cpu);
}

// ============================================================================
// SparseToDense / DenseToSparse roundtrip
// ============================================================================

TEST_F(SparseGPUTest, ToDenseRoundtrip) {
    auto sparse_gpu = make_test_sparse(device_);

    // Sparse -> Dense
    auto dense_gpu = sparse_gpu.to_dense();

    // Dense -> Sparse -> Dense (roundtrip)
    auto sparse_rt = SparseTensor::from_dense(dense_gpu);
    auto dense_rt = sparse_rt.to_dense();

    expect_tensors_close(dense_gpu, dense_rt);
}

// ============================================================================
// SparseAdd: Sparse + Dense
// ============================================================================

TEST_F(SparseGPUTest, SparseAdd_GPUvsCPU) {
    auto sparse_cpu = make_test_sparse(Device::cpu());
    auto sparse_gpu = make_test_sparse(device_);

    auto dense_cpu = ones({4, 4}, DType::Float32, Device::cpu());
    auto dense_gpu = dense_cpu.to(device_);

    auto result_cpu = sparse::add(sparse_cpu, dense_cpu);
    auto result_gpu = sparse::add(sparse_gpu, dense_gpu);

    expect_tensors_close(result_gpu, result_cpu);
}
