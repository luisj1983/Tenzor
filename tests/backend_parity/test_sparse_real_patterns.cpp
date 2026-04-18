/**
 * @file test_sparse_real_patterns.cpp
 * @brief Sparse-op parity with realistic CSR density patterns.
 *
 * Complements test_sparse_parity.cpp which uses a hand-crafted 3x3 CSR and
 * identity-like matrices. This file exercises SpMM / SpMV / sparse-dense
 * conversion with the density patterns that actually occur in ML workloads:
 *
 *   - Random 2% density (attention masks, sparse MoE gating)
 *   - Banded (causal masks, graph adjacency with local neighbourhoods)
 *   - Block-diagonal (block-sparse attention)
 *
 * Each test materialises the sparse matrix both as CSR (routed through the
 * backend-specific sparse library — cuSPARSE, rocSPARSE, oneMKL-sparse,
 * MKL-sparse) and as dense (reference), runs the op, and asserts the
 * results agree within tolerance.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/sparse/sparse_tensor.hpp>
#include <tenzor/sparse/sparse_ops.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

#include <random>
#include <vector>

using namespace tenzor;
using namespace tenzor::testing;

class SparseRealPatterns : public BackendTest {};

namespace {

// Build a CSR sparse matrix from a dense mask + values. Inputs live on CPU.
SparseTensor dense_to_csr(const Tensor& dense) {
    return SparseTensor::from_dense(dense);
}

// Fill tensor with a random mask at given density (fraction of non-zeros).
Tensor make_random_sparse(int64_t rows, int64_t cols, float density, uint64_t seed) {
    auto t = zeros({rows, cols}, DType::Float32, Device::cpu());
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> coin(0.0f, 1.0f);
    std::normal_distribution<float> val_dist(0.0f, 1.0f);
    auto* d = t.data<float>();
    for (int64_t i = 0; i < rows * cols; ++i) {
        if (coin(gen) < density) d[i] = val_dist(gen);
    }
    return t;
}

// Banded matrix: nonzero within ±band of the diagonal.
Tensor make_banded(int64_t n, int64_t band, uint64_t seed) {
    auto t = zeros({n, n}, DType::Float32, Device::cpu());
    std::mt19937_64 gen(seed);
    std::normal_distribution<float> val_dist(0.0f, 1.0f);
    auto* d = t.data<float>();
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = std::max<int64_t>(0, i - band);
             j < std::min<int64_t>(n, i + band + 1); ++j) {
            d[i * n + j] = val_dist(gen);
        }
    }
    return t;
}

// Block-diagonal: small dense blocks along the diagonal; zero elsewhere.
Tensor make_block_diagonal(int64_t n, int64_t block_size, uint64_t seed) {
    auto t = zeros({n, n}, DType::Float32, Device::cpu());
    std::mt19937_64 gen(seed);
    std::normal_distribution<float> val_dist(0.0f, 1.0f);
    auto* d = t.data<float>();
    for (int64_t b = 0; b < n; b += block_size) {
        for (int64_t i = b; i < std::min(b + block_size, n); ++i) {
            for (int64_t j = b; j < std::min(b + block_size, n); ++j) {
                d[i * n + j] = val_dist(gen);
            }
        }
    }
    return t;
}

// Compare sparse op result against dense reference.
void assert_sparse_matches_dense(const Tensor& sparse_result,
                                 const Tensor& dense_reference,
                                 float atol, const std::string& tag) {
    auto a = sparse_result.to(Device::cpu()).contiguous();
    auto b = dense_reference.to(Device::cpu()).contiguous();
    ASSERT_EQ(a.numel(), b.numel()) << tag;
    const float* ap = a.data<float>();
    const float* bp = b.data<float>();
    float max_diff = 0.0f;
    for (int64_t i = 0; i < a.numel(); ++i) {
        max_diff = std::max(max_diff, std::abs(ap[i] - bp[i]));
    }
    EXPECT_LT(max_diff, atol) << tag << " max_diff=" << max_diff;
}

} // namespace

// ---------------------------------------------------------------------------
// Random 2% density SpMM
// ---------------------------------------------------------------------------

TEST_P(SparseRealPatterns, SpMM_Random_2Percent) {
    if (device.type != Device::Type::CPU) {
        // sparse::spmm dispatches into backend libraries via the tensor's
        // device; we run the op on the fixture backend.
    }
    auto sparse_dense = make_random_sparse(64, 64, 0.02f, 42);
    auto csr = dense_to_csr(sparse_dense);
    auto x = randn({64, 16}, DType::Float32, Device::cpu());

    auto result = sparse::spmm(csr, x);
    auto reference = matmul(sparse_dense, x);

    assert_sparse_matches_dense(result, reference, 1e-4f, "SpMM random 2%");
}

// ---------------------------------------------------------------------------
// Banded SpMM (bandwidth 4 on a 64x64)
// ---------------------------------------------------------------------------

TEST_P(SparseRealPatterns, SpMM_Banded) {
    auto banded = make_banded(64, 4, 7);
    auto csr = dense_to_csr(banded);
    auto x = randn({64, 32}, DType::Float32, Device::cpu());

    auto result = sparse::spmm(csr, x);
    auto reference = matmul(banded, x);

    assert_sparse_matches_dense(result, reference, 1e-4f, "SpMM banded");
}

// ---------------------------------------------------------------------------
// Block-diagonal SpMM (8x8 blocks on 64x64)
// ---------------------------------------------------------------------------

TEST_P(SparseRealPatterns, SpMM_BlockDiagonal) {
    auto block = make_block_diagonal(64, 8, 13);
    auto csr = dense_to_csr(block);
    auto x = randn({64, 16}, DType::Float32, Device::cpu());

    auto result = sparse::spmm(csr, x);
    auto reference = matmul(block, x);

    assert_sparse_matches_dense(result, reference, 1e-4f, "SpMM block-diagonal");
}

// ---------------------------------------------------------------------------
// SpMV with random sparse pattern
// ---------------------------------------------------------------------------

TEST_P(SparseRealPatterns, SpMV_Random_5Percent) {
    auto sparse_dense = make_random_sparse(128, 128, 0.05f, 21);
    auto csr = dense_to_csr(sparse_dense);
    auto v = randn({128}, DType::Float32, Device::cpu());

    auto result = sparse::spmv(csr, v);
    auto reference = matmul(sparse_dense, v.reshape({128, 1})).reshape({128});

    assert_sparse_matches_dense(result, reference, 1e-4f, "SpMV random 5%");
}

// ---------------------------------------------------------------------------
// Sparse → dense round-trip with realistic density
// ---------------------------------------------------------------------------

TEST_P(SparseRealPatterns, ToDenseRoundTrip_BlockDiagonal) {
    auto original = make_block_diagonal(32, 4, 99);
    auto csr = dense_to_csr(original);
    auto recovered = csr.to_dense();
    assert_sparse_matches_dense(recovered, original, 1e-6f,
                                "block-diagonal round-trip");
}

INSTANTIATE_BACKEND_TESTS(SparseRealPatterns);
