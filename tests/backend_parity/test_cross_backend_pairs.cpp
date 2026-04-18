/**
 * @file test_cross_backend_pairs.cpp
 * @brief Cross-backend parity (non-CPU-reference) for the highest-traffic ops.
 *
 * The default `test_operation_parity()` helper compares every backend to CPU.
 * That misses bugs where CPU itself is wrong: every other backend then looks
 * "wrong" against the broken CPU reference, but a CUDA-vs-Vulkan check would
 * have caught the real divergence. This file uses
 * `test_operation_parity_cross_backend()` to compare all available pairs.
 *
 * Coverage is intentionally narrow — the most commonly used Tensor-level ops
 * where CPU regressions would have the largest blast radius. Run with:
 *   ctest -R "CrossBackendPairs" -j1 --output-on-failure
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "parity_test_utils.hpp"
#include "parity_tolerances.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ----------------------------------------------------------------------------
// Element-wise math
// ----------------------------------------------------------------------------

TEST(CrossBackendPairs, Add) {
    auto a = randn({64, 64}, DType::Float32, Device::cpu());
    auto b = randn({64, 64}, DType::Float32, Device::cpu());
    test_operation_parity_cross_backend(
        [](const std::vector<Tensor>& ins) { return ins[0] + ins[1]; },
        {a, b}, {}, parity::MATH_RTOL, parity::MATH_ATOL, "add");
}

TEST(CrossBackendPairs, Mul) {
    auto a = randn({64, 64}, DType::Float32, Device::cpu());
    auto b = randn({64, 64}, DType::Float32, Device::cpu());
    test_operation_parity_cross_backend(
        [](const std::vector<Tensor>& ins) { return ins[0] * ins[1]; },
        {a, b}, {}, parity::MATH_RTOL, parity::MATH_ATOL, "mul");
}

TEST(CrossBackendPairs, Exp) {
    auto a = randn({64, 64}, DType::Float32, Device::cpu()) * 0.5f;
    test_operation_parity_cross_backend(
        [](const std::vector<Tensor>& ins) { return exp(ins[0]); },
        {a}, {}, parity::TRANSCENDENTAL_RTOL, parity::TRANSCENDENTAL_ATOL, "exp");
}

TEST(CrossBackendPairs, Log) {
    // Strictly positive inputs to avoid -inf.
    auto a = exp(randn({64, 64}, DType::Float32, Device::cpu()) * 0.5f);
    test_operation_parity_cross_backend(
        [](const std::vector<Tensor>& ins) { return log(ins[0]); },
        {a}, {}, parity::TRANSCENDENTAL_RTOL, parity::TRANSCENDENTAL_ATOL, "log");
}

// ----------------------------------------------------------------------------
// MatMul
// ----------------------------------------------------------------------------

TEST(CrossBackendPairs, MatMulSmall) {
    auto a = randn({16, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 8}, DType::Float32, Device::cpu());
    test_operation_parity_cross_backend(
        [](const std::vector<Tensor>& ins) { return matmul(ins[0], ins[1]); },
        {a, b}, {}, parity::MATMUL_RTOL, parity::MATMUL_ATOL, "matmul_small");
}

TEST(CrossBackendPairs, MatMulMedium) {
    auto a = randn({64, 128}, DType::Float32, Device::cpu());
    auto b = randn({128, 64}, DType::Float32, Device::cpu());
    test_operation_parity_cross_backend(
        [](const std::vector<Tensor>& ins) { return matmul(ins[0], ins[1]); },
        {a, b}, {}, parity::MATMUL_RTOL, parity::MATMUL_ATOL, "matmul_medium");
}

// ----------------------------------------------------------------------------
// Reductions
// ----------------------------------------------------------------------------

TEST(CrossBackendPairs, Sum) {
    auto a = randn({32, 64}, DType::Float32, Device::cpu());
    test_operation_parity_cross_backend(
        [](const std::vector<Tensor>& ins) { return sum(ins[0]); },
        {a}, {}, parity::REDUCTION_RTOL, parity::REDUCTION_ATOL, "sum");
}

TEST(CrossBackendPairs, MeanAxis) {
    auto a = randn({32, 64}, DType::Float32, Device::cpu());
    test_operation_parity_cross_backend(
        [](const std::vector<Tensor>& ins) { return mean(ins[0], 1); },
        {a}, {}, parity::REDUCTION_RTOL, parity::REDUCTION_ATOL, "mean_axis1");
}

TEST(CrossBackendPairs, SumAxis) {
    auto a = randn({32, 64}, DType::Float32, Device::cpu());
    test_operation_parity_cross_backend(
        [](const std::vector<Tensor>& ins) { return sum(ins[0], 0); },
        {a}, {}, parity::REDUCTION_RTOL, parity::REDUCTION_ATOL, "sum_axis0");
}

// ----------------------------------------------------------------------------
// Indexing / shape
// ----------------------------------------------------------------------------

TEST(CrossBackendPairs, Cat) {
    auto a = randn({4, 8}, DType::Float32, Device::cpu());
    auto b = randn({4, 8}, DType::Float32, Device::cpu());
    test_operation_parity_cross_backend(
        [](const std::vector<Tensor>& ins) { return cat({ins[0], ins[1]}, 0); },
        {a, b}, {}, parity::MATH_RTOL, parity::MATH_ATOL, "cat");
}

TEST(CrossBackendPairs, Slice) {
    auto a = randn({16, 16}, DType::Float32, Device::cpu());
    test_operation_parity_cross_backend(
        [](const std::vector<Tensor>& ins) { return ins[0].slice(0, 4, 12); },
        {a}, {}, parity::MATH_RTOL, parity::MATH_ATOL, "slice");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Force full IEEE 754 FP32 on CUDA matmul so CPU↔CUDA parity is measurable.
    setenv("TENZOR_DISABLE_TF32", "1", /*overwrite=*/1);

    try {
        if (!::testing::GTEST_FLAG(list_tests)) {
            tenzor::initialize();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }

    int result = RUN_ALL_TESTS();

    try { tenzor::finalize(); } catch (...) {}
    return result;
}
