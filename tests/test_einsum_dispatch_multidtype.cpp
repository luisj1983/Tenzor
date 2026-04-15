/**
 * @file test_einsum_dispatch_multidtype.cpp
 * @brief Multi-backend tests for einsum OpId dispatch
 *
 * Verifies that the einsum dispatch path produces correct results
 * for common einsum patterns across all available backends.
 */

#include "backend_test_fixture.hpp"
#include "tenzor/tenzor.hpp"
#include "tenzor/ops/advanced.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <gtest/gtest.h>

using namespace tenzor;
using namespace tenzor::testing;

class EinsumDispatchTest : public BackendTest {};

// ij,jk->ik  (matrix multiply)
TEST_P(EinsumDispatchTest, MatMul) {
    auto a = randn({4, 3}, DType::Float32, device);
    auto b = randn({3, 5}, DType::Float32, device);

    auto result = einsum("ij,jk->ik", std::vector<Tensor>{a, b});
    auto expected = matmul(a, b);

    expectTensorNear(result, expected, 1e-4f);
}

// bij,bjk->bik  (batch matmul)
TEST_P(EinsumDispatchTest, BatchMatMul) {
    auto a = randn({2, 4, 3}, DType::Float32, device);
    auto b = randn({2, 3, 5}, DType::Float32, device);

    auto result = einsum("bij,bjk->bik", std::vector<Tensor>{a, b});
    auto expected = bmm(a, b);

    expectTensorNear(result, expected, 1e-4f);
}

// i,i->  (dot product)
TEST_P(EinsumDispatchTest, DotProduct) {
    auto a = randn({8}, DType::Float32, device);
    auto b = randn({8}, DType::Float32, device);

    auto result = einsum("i,i->", std::vector<Tensor>{a, b});
    auto expected = dot(a, b);

    expectTensorNear(result, expected, 1e-4f);
}

// i,j->ij  (outer product)
TEST_P(EinsumDispatchTest, OuterProduct) {
    auto a = randn({4}, DType::Float32, device);
    auto b = randn({5}, DType::Float32, device);

    auto result = einsum("i,j->ij", std::vector<Tensor>{a, b});

    // Manual outer product: reshape + matmul
    auto a_col = reshape(a, {4, 1});
    auto b_row = reshape(b, {1, 5});
    auto expected = matmul(a_col, b_row);

    expectTensorNear(result, expected, 1e-4f);
}

// ii->  (trace)
TEST_P(EinsumDispatchTest, Trace) {
    auto a = randn({4, 4}, DType::Float32, device);

    auto result = einsum("ii->", std::vector<Tensor>{a});
    auto expected = trace(a);

    expectTensorNear(result, expected, 1e-4f);
}

// ii->i  (diagonal)
TEST_P(EinsumDispatchTest, Diagonal) {
    auto a = randn({4, 4}, DType::Float32, device);

    auto result = einsum("ii->i", std::vector<Tensor>{a});
    auto expected = diag(a);

    expectTensorNear(result, expected, 1e-4f);
}

// ij->ji  (transpose via general path)
TEST_P(EinsumDispatchTest, Transpose) {
    auto a = randn({3, 5}, DType::Float32, device);

    auto result = einsum("ij->ji", std::vector<Tensor>{a});
    auto expected = transpose(a, 0, 1);

    expectTensorNear(result, expected, 1e-5f);
}

// Verify dispatch path matches composed path
TEST_P(EinsumDispatchTest, DispatchMatchesComposed) {
    auto a = randn({3, 4}, DType::Float32, device);
    auto b = randn({4, 5}, DType::Float32, device);

    std::string eq = "ij,jk->ik";
    std::vector<Tensor> tensors = {a, b};

    // Call dispatch-based einsum
    auto dispatched = einsum(eq, tensors);

    // Call composed path directly
    auto composed = einsum_composed(eq, tensors);

    expectTensorNear(dispatched, composed, 1e-5f);
}

INSTANTIATE_BACKEND_TESTS(EinsumDispatchTest);
