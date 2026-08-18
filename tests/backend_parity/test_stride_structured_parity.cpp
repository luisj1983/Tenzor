/**
 * @file test_stride_structured_parity.cpp
 * @brief C3: backend-parity for Conv / Pool / Norm with non-contiguous inputs.
 *
 * test_stride_parity.cpp covers elementwise math with stride variants. This
 * file extends the same contract to structured ops (convolution, pooling,
 * normalization) — categories the audit flagged as most likely to harbor
 * stride-ignoring kernel bugs because they do multi-dimensional indexing
 * with manually-computed offsets.
 *
 * Failure here indicates a real stride-handling bug in the corresponding
 * kernel on the reported backend (not a test-design issue). The project's
 * recorded bug pattern: "CUDA/OneAPI/Vulkan all diverge identically from
 * CPU+ROCm when they compute strides from shape instead of reading the
 * tensor's actual .strides()".
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/functional.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"
#include "parity_tolerances.hpp"

using namespace tenzor;
using namespace tenzor::testing;

namespace {

// Build a Float32 input of the requested shape as a NON-CONTIGUOUS view of
// a larger tensor. Slicing the H dimension past the beginning gives a
// stride[0] that does NOT equal the packed size, which is what trips
// stride-from-shape bugs.
Tensor make_non_contig_nchw(int64_t N, int64_t C, int64_t H, int64_t W) {
    // Allocate 2x in H so the slice start != 0.
    auto big = randn({N, C, 2 * H, W}, DType::Float32, Device::cpu());
    return big.slice(/*dim=*/2, /*start=*/H, /*end=*/2 * H);
}

// Same idea for 2D (N, F): pad F and slice to produce a non-packed stride[0].
Tensor make_non_contig_2d(int64_t N, int64_t F) {
    auto big = randn({N, 2 * F}, DType::Float32, Device::cpu());
    return big.slice(/*dim=*/1, /*start=*/F, /*end=*/2 * F);
}

}  // namespace

class StrideStructuredParity : public BackendTest {};

// ---------------------------------------------------------------------------
// Conv2d with a non-contiguous input. The weight stays contiguous — only the
// input exercises the stride path, which is where the bug pattern lives.
// ---------------------------------------------------------------------------
TEST_P(StrideStructuredParity, Conv2d_NonContigInput) {
    auto input = make_non_contig_nchw(2, 3, 8, 8);
    auto weight = randn({4, 3, 3, 3}, DType::Float32, Device::cpu());
    ASSERT_FALSE(input.is_contiguous())
        << "precondition: the sliced input should be non-contiguous";

    test_operation_parity_single(
        [](const std::vector<Tensor>& inputs) {
            Variable in(inputs[0], false);
            Variable w(inputs[1], false);
            return tenzor::nn::functional::conv2d(in, w, std::nullopt,
                                                  {1, 1}, {1, 1}).tensor();
        },
        {input, weight}, device, parity::CONV_RTOL, parity::CONV_ATOL, "Conv2d non-contig input");
}

// ---------------------------------------------------------------------------
// MaxPool2d non-contiguous input. Pooling reads per-window maxima — a
// stride-from-shape bug shows up as reading the wrong elements and picking
// a different max.
// ---------------------------------------------------------------------------
TEST_P(StrideStructuredParity, MaxPool2d_NonContigInput) {
    auto input = make_non_contig_nchw(2, 3, 8, 8);
    ASSERT_FALSE(input.is_contiguous());

    test_operation_parity_single(
        [](const std::vector<Tensor>& inputs) {
            Variable in(inputs[0], false);
            return tenzor::nn::functional::max_pool2d(in, {2, 2}, {2, 2}).tensor();
        },
        {input}, device, 1e-5f, 1e-6f, "MaxPool2d non-contig input");
}

TEST_P(StrideStructuredParity, AvgPool2d_NonContigInput) {
    auto input = make_non_contig_nchw(2, 3, 8, 8);
    ASSERT_FALSE(input.is_contiguous());

    test_operation_parity_single(
        [](const std::vector<Tensor>& inputs) {
            Variable in(inputs[0], false);
            return tenzor::nn::functional::avg_pool2d(in, {2, 2}, {2, 2},
                                                      {0, 0}).tensor();
        },
        {input}, device, 1e-5f, 1e-6f, "AvgPool2d non-contig input");
}

// ---------------------------------------------------------------------------
// LayerNorm on a non-contiguous 2D input. The reduction is along the last
// (feature) dim; a bug reading stride[0] from shape would shuffle which
// rows are normalized together.
// ---------------------------------------------------------------------------
TEST_P(StrideStructuredParity, LayerNorm_NonContigInput) {
    auto input = make_non_contig_2d(4, 16);
    ASSERT_FALSE(input.is_contiguous());

    test_operation_parity_single(
        [](const std::vector<Tensor>& inputs) {
            Variable in(inputs[0], false);
            return tenzor::nn::functional::layer_norm(
                in, std::vector<int64_t>{16}).tensor();
        },
        {input}, device, 1e-4f, 1e-5f, "LayerNorm non-contig input");
}

// ---------------------------------------------------------------------------
// MatMul with a non-contiguous left operand. This is the "transposed matmul"
// shape without calling .contiguous() afterwards — a classic stride test.
// ---------------------------------------------------------------------------
TEST_P(StrideStructuredParity, MatMul_TransposedLHS) {
    // Construct a (32, 16) tensor that is a transposed view of (16, 32).
    auto src = randn({16, 32}, DType::Float32, Device::cpu());
    auto lhs = src.transpose(0, 1);   // (32, 16), non-contig
    auto rhs = randn({16, 8}, DType::Float32, Device::cpu());
    ASSERT_FALSE(lhs.is_contiguous());

    test_operation_parity_single(
        [](const std::vector<Tensor>& inputs) {
            return matmul(inputs[0], inputs[1]);
        },
        {lhs, rhs}, device, parity::MATMUL_RTOL, parity::MATMUL_ATOL, "MatMul transposed LHS");
}

INSTANTIATE_BACKEND_TESTS(StrideStructuredParity);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    // Force full IEEE 754 FP32 on CUDA matmul so MatMul parity is measurable.
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
