/**
 * @file test_stride_parity.cpp
 * @brief Backend-parity tests with non-contiguous inputs.
 *
 * Motivation: kernels that compute strides from shape (instead of reading the
 * tensor's actual `.strides()`) silently produce wrong results for inputs that
 * are views into non-contiguous memory. The classic pattern is that CPU and
 * ROCm handle strides correctly while CUDA / OneAPI / Vulkan diverge
 * identically — the existing parity tests all use contiguous inputs and miss
 * this entire class of bugs.
 *
 * This test file feeds a standard suite of ops a set of non-contiguous input
 * variants (transpose, permute, narrow-slice, unsqueeze/squeeze), transfers
 * them to every enabled backend, and asserts the result matches CPU.
 *
 * A failure here does NOT indicate a bug in this test — it indicates a
 * stride-handling bug in the corresponding kernel on the reported backend.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/functional.hpp>
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

namespace {

// A named transformation applied to a CPU tensor to produce a non-contiguous
// view. Applied identically to all inputs in a test.
struct StrideVariant {
    const char* name;
    std::function<Tensor(const Tensor&)> apply;
};

// Variants chosen to stress every kind of stride non-uniformity a kernel might
// encounter: transposed (stride swap), permuted (arbitrary axis reorder),
// narrowed (non-whole slice), unsqueezed-squeezed (stride 0 then collapsed).
static const std::vector<StrideVariant> kStrideVariants = {
    {"contig",             [](const Tensor& t) { return t; }},
    {"transpose",          [](const Tensor& t) { return t.transpose(-1, -2); }},
    {"permute_01",         [](const Tensor& t) { return t.permute({1, 0}); }},
    {"narrow_slice",       [](const Tensor& t) { return t.slice(0, 0, t.size(0) / 2); }},
    {"unsqueeze_squeeze",  [](const Tensor& t) { return t.unsqueeze(0).squeeze(0); }},
};

// Tolerances are looser than contiguous parity because non-contiguous kernels
// may take different reduction orders; 1e-4/1e-6 matches nn_parity conventions.
constexpr float kStrideRtol = 1e-4f;
constexpr float kStrideAtol = 1e-6f;

} // namespace

// ============================================================================
// Helper: run parity for a single op × all variants using (square) inputs.
// Square shapes let transpose/permute produce a shape the binary op still
// broadcasts against itself.
// ============================================================================

template <typename Op>
static void check_all_variants_binary(Op op, const std::string& op_name) {
    for (const auto& variant : kStrideVariants) {
        SCOPED_TRACE(std::string(op_name) + "/" + variant.name);
        auto a = randn({16, 16}, DType::Float32, Device::cpu());
        auto b = randn({16, 16}, DType::Float32, Device::cpu());
        auto a_v = variant.apply(a);
        auto b_v = variant.apply(b);

        // Require both inputs to have the same shape after the variant.
        ASSERT_EQ(a_v.shape().size(), b_v.shape().size());

        test_operation_parity(op, {a_v, b_v}, kStrideRtol, kStrideAtol,
                              std::string(op_name) + "_" + variant.name);
    }
}

template <typename Op>
static void check_all_variants_unary(Op op, const std::string& op_name) {
    for (const auto& variant : kStrideVariants) {
        SCOPED_TRACE(std::string(op_name) + "/" + variant.name);
        // Use strictly positive inputs for ops that require it (log, sqrt).
        // abs() on randn is safe for all numerics tested here.
        auto a = tenzor::abs(randn({16, 16}, DType::Float32, Device::cpu())) + 0.5f;
        auto a_v = variant.apply(a);

        test_operation_parity(op, {a_v}, kStrideRtol, kStrideAtol,
                              std::string(op_name) + "_" + variant.name);
    }
}

// ============================================================================
// Binary elementwise ops — most likely to expose broadcast + stride bugs.
// ============================================================================

TEST(StrideParity, Add) {
    check_all_variants_binary(
        [](const std::vector<Tensor>& in) { return in[0] + in[1]; },
        "Add");
}

TEST(StrideParity, Sub) {
    check_all_variants_binary(
        [](const std::vector<Tensor>& in) { return in[0] - in[1]; },
        "Sub");
}

TEST(StrideParity, Mul) {
    check_all_variants_binary(
        [](const std::vector<Tensor>& in) { return in[0] * in[1]; },
        "Mul");
}

TEST(StrideParity, Div) {
    // Use inputs that avoid div-by-zero by scaling away from 0.
    for (const auto& variant : kStrideVariants) {
        SCOPED_TRACE(std::string("Div/") + variant.name);
        auto a = randn({16, 16}, DType::Float32, Device::cpu());
        auto b_raw = randn({16, 16}, DType::Float32, Device::cpu());
        // Shift away from zero
        auto b = b_raw + 2.0f;
        auto a_v = variant.apply(a);
        auto b_v = variant.apply(b);

        test_operation_parity(
            [](const std::vector<Tensor>& in) { return in[0] / in[1]; },
            {a_v, b_v}, kStrideRtol, kStrideAtol,
            std::string("Div_") + variant.name);
    }
}

// ============================================================================
// Unary elementwise ops.
// ============================================================================

TEST(StrideParity, Neg) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return in[0] * -1.0f; },
        "Neg");
}

TEST(StrideParity, Abs) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::abs(in[0]); },
        "Abs");
}

TEST(StrideParity, Sqrt) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::sqrt(in[0]); },
        "Sqrt");
}

TEST(StrideParity, Exp) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::exp(in[0]); },
        "Exp");
}

TEST(StrideParity, Log) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::log(in[0]); },
        "Log");
}

// ----- Element-wise math (rest of the floor from required_ops.hpp) ----------

TEST(StrideParity, Sign) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::sign(in[0]); },
        "Sign");
}

TEST(StrideParity, Reciprocal) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::reciprocal(in[0]); },
        "Reciprocal");
}

TEST(StrideParity, Floor) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::floor(in[0]); },
        "Floor");
}

TEST(StrideParity, Ceil) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::ceil(in[0]); },
        "Ceil");
}

TEST(StrideParity, Round) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::round(in[0]); },
        "Round");
}

// ----- Trigonometric --------------------------------------------------------

TEST(StrideParity, Sin) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::sin(in[0]); },
        "Sin");
}

TEST(StrideParity, Cos) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::cos(in[0]); },
        "Cos");
}

TEST(StrideParity, Tan) {
    // tan diverges near ±π/2 + kπ. check_all_variants_unary feeds values
    // in roughly [0.5, 4], which occasionally lands near π/2 ≈ 1.5708 and
    // produces huge elementwise differences from ordinary float rounding
    // alone. Restrict to [-1, 1] so every backend's tan is well-conditioned.
    for (const auto& variant : kStrideVariants) {
        SCOPED_TRACE(std::string("Tan/") + variant.name);
        auto a = tenzor::clamp(randn({16, 16}, DType::Float32, Device::cpu()),
                               -1.0f, 1.0f);
        auto a_v = variant.apply(a);
        test_operation_parity(
            [](const std::vector<Tensor>& in) { return tenzor::tan(in[0]); },
            {a_v}, kStrideRtol, kStrideAtol,
            std::string("Tan_") + variant.name);
    }
}

TEST(StrideParity, Tanh) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::tanh(in[0]); },
        "Tanh");
}

TEST(StrideParity, Sinh) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::sinh(in[0]); },
        "Sinh");
}

TEST(StrideParity, Cosh) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::cosh(in[0]); },
        "Cosh");
}

// ----- Extended math --------------------------------------------------------

TEST(StrideParity, Log2) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::log2(in[0]); },
        "Log2");
}

TEST(StrideParity, Log10) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::log10(in[0]); },
        "Log10");
}

TEST(StrideParity, Log1p) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::log1p(in[0]); },
        "Log1p");
}

TEST(StrideParity, Exp2) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::exp2(in[0]); },
        "Exp2");
}

TEST(StrideParity, Expm1) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::expm1(in[0]); },
        "Expm1");
}

TEST(StrideParity, Erf) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::erf(in[0]); },
        "Erf");
}

TEST(StrideParity, Erfc) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::erfc(in[0]); },
        "Erfc");
}

// ----- Activations (forward) -----------------------------------------------

TEST(StrideParity, ReLU) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) {
            return tenzor::nn::functional::relu(Variable(in[0], false)).tensor();
        },
        "ReLU");
}

TEST(StrideParity, Sigmoid) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) {
            return tenzor::nn::functional::sigmoid(Variable(in[0], false)).tensor();
        },
        "Sigmoid");
}

TEST(StrideParity, GeLU) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) {
            return tenzor::nn::functional::gelu(Variable(in[0], false)).tensor();
        },
        "GeLU");
}

TEST(StrideParity, Softmax) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) {
            return tenzor::nn::functional::softmax(Variable(in[0], false), /*dim=*/-1).tensor();
        },
        "Softmax");
}

// ----- Additional reductions ------------------------------------------------

TEST(StrideParity, Prod) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::prod(in[0]); },
        "Prod");
}

// NOTE: ArgMax / ArgMin return Int64 index tensors. The generic
// test_operation_parity helper uses float tolerances and does not compare
// integer tensors correctly (see tests/backend_parity/parity_test_utils.hpp
// tensors_close). Adding these here reports a false positive failure with
// max_abs_diff == 0. They should be covered by a dedicated integer-aware
// parity test (not yet written).

// ----- Binary elementwise (extended) ---------------------------------------

TEST(StrideParity, PowScalar) {
    // pow(Tensor, float) — unary with a fixed exponent.
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::pow(in[0], 2.5f); },
        "PowScalar");
}

// ============================================================================
// Reductions — these easily misread strides since they index non-contiguously.
// ============================================================================

TEST(StrideParity, Sum) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::sum(in[0]); },
        "Sum");
}

TEST(StrideParity, Mean) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::mean(in[0]); },
        "Mean");
}

TEST(StrideParity, Max) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::max(in[0]); },
        "Max");
}

TEST(StrideParity, Min) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::min(in[0]); },
        "Min");
}

// ============================================================================
// MatMul with transposed operands — the single most common non-contiguous
// use-case in real ML workloads (attention Q @ K^T).
// ============================================================================

TEST(StrideParity, MatMulTransposedB) {
    auto a = randn({16, 32}, DType::Float32, Device::cpu());
    auto b = randn({16, 32}, DType::Float32, Device::cpu());  // transpose makes this (32, 16)

    auto b_t = b.transpose(-1, -2);  // non-contiguous

    test_operation_parity(
        [](const std::vector<Tensor>& in) { return tenzor::matmul(in[0], in[1]); },
        {a, b_t}, kStrideRtol, kStrideAtol, "MatMulTransposedB");
}

TEST(StrideParity, MatMulTransposedA) {
    auto a = randn({32, 16}, DType::Float32, Device::cpu());
    auto b = randn({32, 64}, DType::Float32, Device::cpu());

    auto a_t = a.transpose(-1, -2);  // now (16, 32) non-contiguous

    test_operation_parity(
        [](const std::vector<Tensor>& in) { return tenzor::matmul(in[0], in[1]); },
        {a_t, b}, kStrideRtol, kStrideAtol, "MatMulTransposedA");
}

TEST(StrideParity, MatMulBothTransposed) {
    auto a = randn({32, 16}, DType::Float32, Device::cpu());  // becomes (16, 32) transposed
    auto b = randn({64, 32}, DType::Float32, Device::cpu());  // becomes (32, 64) transposed

    test_operation_parity(
        [](const std::vector<Tensor>& in) {
            return tenzor::matmul(in[0].transpose(-1, -2), in[1].transpose(-1, -2));
        },
        {a, b}, kStrideRtol, kStrideAtol, "MatMulBothTransposed");
}

// ============================================================================
// Phase 5 expansion — axis-reductions, axis-softmax, broadcast-binary,
// indexing, normalization, and sorted/top-k ops on non-contiguous inputs.
// These are the ops most likely to harbor stride-from-shape bugs that the
// audit memory flagged (CUDA/OneAPI/Vulkan diverging together from CPU+ROCm).
// ============================================================================

TEST(StrideParity, SumDim) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::sum(in[0], 0, false); },
        "SumDim0");
}

TEST(StrideParity, MeanDim) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::mean(in[0], 1, false); },
        "MeanDim1");
}

TEST(StrideParity, MaxDim) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::max(in[0], 0, false); },
        "MaxDim0");
}

TEST(StrideParity, MinDim) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::min(in[0], 1, false); },
        "MinDim1");
}

TEST(StrideParity, ProdDim) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::prod(in[0], 0, false); },
        "ProdDim0");
}

// Single-pass Welford Vulkan kernel (welford_variance*.comp) closed the
// precision gap from issue #30 — these tests were previously disabled.
TEST(StrideParity, StdDim) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::std(in[0], 0, false, false); },
        "StdDim0");
}

TEST(StrideParity, VarDim) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) { return tenzor::var(in[0], 0, false, false); },
        "VarDim0");
}

TEST(StrideParity, SoftmaxDim0) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) {
            return tenzor::nn::functional::softmax(Variable(in[0], false), /*dim=*/0).tensor();
        },
        "SoftmaxDim0");
}

TEST(StrideParity, LogSoftmaxDim0) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) {
            return tenzor::nn::functional::log_softmax(Variable(in[0], false), /*dim=*/0).tensor();
        },
        "LogSoftmaxDim0");
}

TEST(StrideParity, CumsumDim0) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) {
            return tenzor::cumsum(Variable(in[0], false), 0).tensor();
        },
        "CumsumDim0");
}

TEST(StrideParity, CumsumDim1) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) {
            return tenzor::cumsum(Variable(in[0], false), 1).tensor();
        },
        "CumsumDim1");
}

// LayerNorm with the default-derived normalized_shape from the last dim.
// Uses no learnable affine (weight/bias both null) so it's a pure stride
// test of the normalization kernel.
TEST(StrideParity, LayerNorm) {
    check_all_variants_unary(
        [](const std::vector<Tensor>& in) {
            std::vector<int64_t> norm_shape = {in[0].shape().back()};
            auto v = tenzor::nn::functional::layer_norm(
                Variable(in[0], false), norm_shape,
                std::nullopt, std::nullopt, /*eps=*/1e-5);
            return v.tensor();
        },
        "LayerNorm");
}

// Broadcast-binary on non-square shapes — exercises stride-aware broadcast.
TEST(StrideParity, AddBroadcast_NonSquare) {
    // Shape (2, 4, 6) + (4, 1) — broadcast along dim 0 and dim 2 of the
    // first operand. Apply transposes to both to cover the non-contiguous
    // case.
    auto a = randn({2, 4, 6}, DType::Float32, Device::cpu());
    auto b = randn({4, 1}, DType::Float32, Device::cpu());
    test_operation_parity(
        [](const std::vector<Tensor>& in) { return tenzor::add(in[0], in[1]); },
        {a, b}, kStrideRtol, kStrideAtol, "AddBroadcast_NonSquare");

    // Now with a transposed (a.transpose(0,2)) — strides become non-trivial.
    auto a_t = a.transpose(0, 2);  // (6, 4, 2) view
    test_operation_parity(
        [](const std::vector<Tensor>& in) { return tenzor::add(in[0], in[1]); },
        {a_t, b}, kStrideRtol, kStrideAtol, "AddBroadcast_NonSquare_Transposed");
}

// Top-k along the last dim — kernel must walk strided rows correctly.
TEST(StrideParity, TopK_LastDim) {
    auto a = randn({16, 32}, DType::Float32, Device::cpu());
    test_operation_parity(
        [](const std::vector<Tensor>& in) {
            // Tensor-level topk returns std::pair<Tensor, Tensor> per
            // ops/advanced.hpp. The parity helper compares only the first
            // returned tensor (values), which is what we want — indices are
            // an integer side output not covered by the float comparator.
            auto [v, idx] = tenzor::topk(Variable(in[0], false), /*k=*/4,
                                         /*dim=*/-1, /*largest=*/true,
                                         /*sorted=*/true);
            return v.tensor();
        },
        {a}, kStrideRtol, kStrideAtol, "TopK_LastDim_Contiguous");

    auto a_t = a.transpose(-1, -2);  // (32, 16) non-contiguous
    test_operation_parity(
        [](const std::vector<Tensor>& in) {
            auto [v, idx] = tenzor::topk(Variable(in[0], false), /*k=*/4,
                                         /*dim=*/-1, /*largest=*/true,
                                         /*sorted=*/true);
            return v.tensor();
        },
        {a_t}, kStrideRtol, kStrideAtol, "TopK_LastDim_Transposed");
}

// Custom main so tenzor::initialize() runs before any test. Mirrors the
// pattern in test_operation_parity.cpp — gtest_main would run tests before
// initialize, and the first .to(device) throws "Backend not available".
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Match the other parity tests: disable TF32 so CPU↔CUDA Float32 matmul
    // is bit-reproducible within FP32 tolerance.
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

    try {
        tenzor::finalize();
    } catch (...) {
        // best-effort cleanup
    }
    return result;
}
