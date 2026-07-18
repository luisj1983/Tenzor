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
#include "../backend_parity/parity_test_utils.hpp"
#include <cmath>

using namespace tenzor;

// Parametrized over backend name so one fixture exercises every GPU
// backend the build actually supports. Each parametrization probes the
// backend at setup time and skips itself if the backend isn't
// available (e.g. ROCm on a CUDA-only box, or vice versa), so a single
// test binary covers both CUDA and ROCm without duplicating code.
class SparseGPUTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        initialize();
        const auto& name = GetParam();
        // Select the device for this parametrization. A bad param name is a
        // test-wiring bug and must FAIL (kept outside any catch so it isn't
        // swallowed into a skip).
        if (name == "cuda") {
            device_ = Device::cuda(0);
        } else if (name == "rocm") {
            device_ = Device::rocm(0);
        } else if (name == "oneapi") {
            device_ = Device::oneapi(0);
        } else if (name == "vulkan") {
            device_ = Device::vulkan(0);
        } else {
            FAIL() << "Unknown backend: " << name;
        }
        // Deterministic availability precondition: skip only when the backend
        // is genuinely absent. Past this point any sparse-kernel exception is
        // a real bug and must propagate, not be reclassified as "not available".
        if (!tenzor::testing::is_backend_name_available(name)) {
            GTEST_SKIP() << "Backend " << name << " not available";
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

TEST_P(SparseGPUTest, SpMM_GPUvsCPU) {
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

TEST_P(SparseGPUTest, SpMV_GPUvsCPU) {
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

TEST_P(SparseGPUTest, ToDenseRoundtrip) {
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

TEST_P(SparseGPUTest, SparseAdd_GPUvsCPU) {
    auto sparse_cpu = make_test_sparse(Device::cpu());
    auto sparse_gpu = make_test_sparse(device_);

    auto dense_cpu = ones({4, 4}, DType::Float32, Device::cpu());
    auto dense_gpu = dense_cpu.to(device_);

    auto result_cpu = sparse::add(sparse_cpu, dense_cpu);
    auto result_gpu = sparse::add(sparse_gpu, dense_gpu);

    expect_tensors_close(result_gpu, result_cpu);
}

// ============================================================================
// SpGEMM — sparse × sparse → sparse (cuSPARSE / rocSPARSE path)
// ============================================================================
//
// Skipped if running on a non-cuSPARSE build: only CUDA with TENZOR_HAS_CUSPARSE
// (the default on a CUDA install) currently routes through the GPU kernel.
// The ROCm path lands in the next plan item.

TEST_P(SparseGPUTest, SpGEMM_GPUvsCPU) {
    if (device_.type != Device::Type::CUDA && device_.type != Device::Type::ROCm) {
        GTEST_SKIP() << "SpGEMM GPU path currently only wired for CUDA/ROCm";
    }
    auto a_cpu = make_test_sparse(Device::cpu());
    auto b_cpu = make_test_sparse(Device::cpu());
    auto a_gpu = a_cpu.to(device_);
    auto b_gpu = b_cpu.to(device_);

    auto c_cpu = sparse::spgemm(a_cpu, b_cpu);
    auto c_gpu = sparse::spgemm(a_gpu, b_gpu);

    // Compare densified results — SpGEMM on CPU returns a different ordering
    // of nonzeros than cuSPARSE in general, so densify before comparing.
    // SparseTensor::to_dense() is currently CPU-only, so bring c_gpu to
    // CPU first; the SpGEMM itself ran on-device via cusparseSpGEMM.
    auto dense_cpu = c_cpu.to_dense();
    auto dense_gpu = c_gpu.to(Device::cpu()).to_dense();
    expect_tensors_close(dense_gpu, dense_cpu);
}

TEST_P(SparseGPUTest, SpGEMM_LargerRandom_GPUvsCPU) {
    if (device_.type != Device::Type::CUDA && device_.type != Device::Type::ROCm) {
        GTEST_SKIP() << "SpGEMM GPU path currently only wired for CUDA/ROCm";
    }
    // 128 × 128 random-ish structure: diagonal + a few off-diagonal entries,
    // built manually so the test is deterministic.
    const int64_t N = 128;
    std::vector<int64_t> rows_v, cols_v;
    std::vector<float> vals_v;
    for (int64_t i = 0; i < N; ++i) {
        rows_v.push_back(i); cols_v.push_back(i);           vals_v.push_back(2.0f);
        if (i + 1 < N) {
            rows_v.push_back(i); cols_v.push_back(i + 1);   vals_v.push_back(-1.0f);
        }
        if (i >= 1) {
            rows_v.push_back(i); cols_v.push_back(i - 1);   vals_v.push_back(-0.5f);
        }
    }
    const int64_t nnz = static_cast<int64_t>(rows_v.size());
    auto indices = Tensor({2, nnz}, DType::Int64, Device::cpu());
    auto values  = Tensor({nnz},    DType::Float32, Device::cpu());
    auto* idx = indices.data<int64_t>();
    for (int64_t k = 0; k < nnz; ++k) {
        idx[k]       = rows_v[static_cast<size_t>(k)];
        idx[k + nnz] = cols_v[static_cast<size_t>(k)];
    }
    auto* v = values.data<float>();
    for (int64_t k = 0; k < nnz; ++k) v[k] = vals_v[static_cast<size_t>(k)];

    auto a_cpu = SparseTensor::sparse_coo(indices, values, {N, N});
    auto a_gpu = a_cpu.to(device_);

    // c = a * a — the classic spgemm test, product has roughly 5x nnz.
    auto c_cpu = sparse::spgemm(a_cpu, a_cpu);
    auto c_gpu = sparse::spgemm(a_gpu, a_gpu);

    auto dense_cpu = c_cpu.to_dense();
    auto dense_gpu = c_gpu.to(Device::cpu()).to_dense();
    expect_tensors_close(dense_gpu, dense_cpu, /*atol=*/1e-4f);
}

// ============================================================================
// Sparse triangular solve — GPU SpSV path (CUDA/ROCm/Vulkan/OneAPI)
// ============================================================================
//
// Builds a lower triangular 4×4 matrix and solves L*x = b against the
// CPU path. On CUDA this exercises cusparseSpSV_analysis + cusparseSpSV_solve
// end-to-end; on ROCm/Vulkan/OneAPI it exercises those backends' own
// SparseTrsv/SparseTrsm kernels (hand-rolled sequential substitution on
// Vulkan/ROCm, oneMKL/CPU-fallback on OneAPI).

static SparseTensor make_lower_triangular(Device device) {
    // L =
    //  [ 2 0 0 0 ]
    //  [ 1 3 0 0 ]
    //  [ 0 2 4 0 ]
    //  [ 1 0 3 5 ]
    // 8 non-zeros. COO layout: indices [2, nnz], row 0 = row-indices,
    // row 1 = col-indices. Row-major storage puts row indices first.
    auto indices = Tensor({2, 8}, DType::Int64, Device::cpu());
    auto* idx = indices.data<int64_t>();
    //        (0,0) (1,0) (1,1) (2,1) (2,2) (3,0) (3,2) (3,3)
    idx[0] = 0; idx[1] = 1; idx[2] = 1; idx[3] = 2; idx[4] = 2; idx[5] = 3; idx[6] = 3; idx[7] = 3;
    idx[8] = 0; idx[9] = 0; idx[10] = 1; idx[11] = 1; idx[12] = 2; idx[13] = 0; idx[14] = 2; idx[15] = 3;

    auto values = Tensor({8}, DType::Float32, Device::cpu());
    auto* v = values.data<float>();
    v[0] = 2.0f;  // (0,0)
    v[1] = 1.0f;  // (1,0)
    v[2] = 3.0f;  // (1,1)
    v[3] = 2.0f;  // (2,1)
    v[4] = 4.0f;  // (2,2)
    v[5] = 1.0f;  // (3,0)
    v[6] = 3.0f;  // (3,2)
    v[7] = 5.0f;  // (3,3)

    auto sparse = SparseTensor::sparse_coo(indices, values, {4, 4});
    return sparse.to(device);
}

TEST_P(SparseGPUTest, SparseTrsv_GPUvsCPU) {
    if (device_.type != Device::Type::CUDA && device_.type != Device::Type::ROCm &&
        device_.type != Device::Type::Vulkan && device_.type != Device::Type::OneAPI) {
        GTEST_SKIP() << "Sparse TRSV GPU path currently only wired for "
                        "CUDA/ROCm/Vulkan/OneAPI";
    }
    auto L_cpu = make_lower_triangular(Device::cpu());
    auto L_gpu = make_lower_triangular(device_);

    auto b_cpu = Tensor({4}, DType::Float32, Device::cpu());
    auto* bp = b_cpu.data<float>();
    bp[0] = 2.0f; bp[1] = 4.0f; bp[2] = 10.0f; bp[3] = 17.0f;
    auto b_gpu = b_cpu.to(device_);

    auto x_cpu = sparse::sparse_triangular_solve(L_cpu, b_cpu, /*upper=*/false);
    auto x_gpu = sparse::sparse_triangular_solve(L_gpu, b_gpu, /*upper=*/false);
    expect_tensors_close(x_gpu, x_cpu, /*atol=*/1e-5f);
}

TEST_P(SparseGPUTest, SparseTrsm_GPUvsCPU) {
    if (device_.type != Device::Type::CUDA && device_.type != Device::Type::ROCm &&
        device_.type != Device::Type::Vulkan && device_.type != Device::Type::OneAPI) {
        GTEST_SKIP() << "Sparse TRSM GPU path currently only wired for "
                        "CUDA/ROCm/Vulkan/OneAPI";
    }
    auto L_cpu = make_lower_triangular(Device::cpu());
    auto L_gpu = make_lower_triangular(device_);

    // 4×2 dense RHS
    auto B_cpu = Tensor({4, 2}, DType::Float32, Device::cpu());
    auto* bp = B_cpu.data<float>();
    bp[0] = 2.0f;  bp[1] = 4.0f;
    bp[2] = 4.0f;  bp[3] = 9.0f;
    bp[4] = 10.0f; bp[5] = 18.0f;
    bp[6] = 17.0f; bp[7] = 35.0f;
    auto B_gpu = B_cpu.to(device_);

    auto X_cpu = sparse::sparse_triangular_solve(L_cpu, B_cpu, /*upper=*/false);
    auto X_gpu = sparse::sparse_triangular_solve(L_gpu, B_gpu, /*upper=*/false);
    expect_tensors_close(X_gpu, X_cpu, /*atol=*/1e-5f);
}


// ============================================================================
// Backend instantiations
// ============================================================================
INSTANTIATE_TEST_SUITE_P(
    CUDA, SparseGPUTest,
    ::testing::Values(std::string("cuda")),
    [](const ::testing::TestParamInfo<std::string>& info) { return info.param; });

INSTANTIATE_TEST_SUITE_P(
    ROCm, SparseGPUTest,
    ::testing::Values(std::string("rocm")),
    [](const ::testing::TestParamInfo<std::string>& info) { return info.param; });

// Vulkan and OneAPI now have genuine SparseTrsv/SparseTrsm kernels (see
// vulkan_ops_linalg.cpp and oneapi_kernel_registry.cpp), and SpMM/SpMV/
// SparseAdd/ToDenseRoundtrip already worked on both — but this suite never
// instantiated either parametrization, so none of it actually ran on those
// backends despite SetUp() above already knowing how to select their
// devices. SpGEMM_GPUvsCPU/SpGEMM_LargerRandom_GPUvsCPU still correctly
// self-skip on these two (their own CUDA/ROCm-only guard is unrelated to
// this gap and is left as-is).
INSTANTIATE_TEST_SUITE_P(
    OneAPI, SparseGPUTest,
    ::testing::Values(std::string("oneapi")),
    [](const ::testing::TestParamInfo<std::string>& info) { return info.param; });

INSTANTIATE_TEST_SUITE_P(
    Vulkan, SparseGPUTest,
    ::testing::Values(std::string("vulkan")),
    [](const ::testing::TestParamInfo<std::string>& info) { return info.param; });
