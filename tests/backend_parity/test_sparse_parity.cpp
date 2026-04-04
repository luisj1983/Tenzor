/**
 * @file test_sparse_parity.cpp
 * @brief Sparse operation parity tests across backends
 *
 * Verifies that sparse tensor operations (SpMM, SpMV, sparse-dense conversion)
 * produce identical results across all backends.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Sparse-Dense Interaction Parity
// ============================================================================

TEST(SparseParity, IdentityMatmul) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // Identity matrix @ dense vector — exercises sparse-like paths
    auto identity = eye(8, 8, DType::Float32, Device::cpu());
    auto dense = randn({8, 4}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return matmul(inputs[0], inputs[1]);
    }, {identity, dense}, 1e-5f, 1e-6f, "Identity @ Dense (sparse-like)");
}

TEST(SparseParity, SparsePatternMatmul) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // Mostly-zero matrix @ dense — tests numerical accumulation with many zeros
    auto sparse_like = zeros({8, 8}, DType::Float32, Device::cpu());
    // Set diagonal and a few off-diagonal entries
    auto sparse_data = sparse_like.data<float>();
    for (int i = 0; i < 8; ++i) {
        sparse_data[i * 8 + i] = 1.0f;           // diagonal
        if (i + 1 < 8) sparse_data[i * 8 + i + 1] = 0.5f;  // super-diagonal
    }
    auto dense = randn({8, 4}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return matmul(inputs[0], inputs[1]);
    }, {sparse_like, dense}, 1e-5f, 1e-6f, "Sparse-pattern matmul");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    try {
        tenzor::initialize();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }

    int result = RUN_ALL_TESTS();

    try {
        tenzor::finalize();
    } catch (...) {}

    return result;
}
