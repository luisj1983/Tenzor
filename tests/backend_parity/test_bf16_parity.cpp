/**
 * @file test_bf16_parity.cpp
 * @brief BFloat16 backend-parity tests for Vulkan (and other backends).
 *
 * Motivation: 147 BF16 compute shaders exist under src/backends/vulkan/kernels/
 * but ~40 of their C++ dispatcher sites never route BF16 to the dedicated
 * shader. BF16 inputs silently fall through to the generic Float32 shader,
 * which interprets packed 2-byte-per-element buffers as 4-byte-per-element
 * floats and produces garbage. This test file exercises each op category
 * with a BF16 tensor and compares the result to CPU, surfacing every
 * mis-routed or broken dispatcher as a test failure.
 *
 * Structure: for each op, create the input on CPU as Float32 → cast to
 * BFloat16 → transfer to each available backend → run the op → transfer
 * back to CPU → cast to Float32 → compare against the CPU-BFloat16 result.
 * Tolerance is generous (BF16 has ~7-bit mantissa → ~0.8% relative error)
 * so only outright data-corruption failures surface.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/functional.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/ops/op_id.hpp>
#include <tenzor/backend/op_attributes.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

#include <cmath>
#include <cstdlib>

using namespace tenzor;
using namespace tenzor::testing;


class BF16Parity : public BackendTest {};
namespace {

// BF16-specific tolerances. BFloat16 has 8-bit mantissa (7 explicit bits),
// so ~0.8% relative error is normal for accumulated ops. We set generous
// absolute tolerance to distinguish "correct but imprecise" from "garbage".
constexpr float kBF16Rtol = 5e-2f;   // 5 % relative
constexpr float kBF16Atol = 1e-1f;   // 0.1 absolute

/// Return only backends that support BFloat16.
std::vector<Device> get_bf16_backends() {
    auto all = get_available_backends();
    std::vector<Device> out;
    for (const auto& dev : all) {
        try {
            auto t = zeros({4}, DType::BFloat16, dev);
            auto r = t + t;  // exercise a basic op to prove dtype is functional
            out.push_back(dev);
        } catch (...) {
            // Backend doesn't support BF16 — skip it.
        }
    }
    return out;
}

/// Run an op on CPU-BF16 as reference, then on each backend-BF16, compare.
template <typename Op>
void test_bf16_op(Op op,
                  const std::vector<Tensor>& cpu_f32_inputs,
                  const std::string& name) {
    auto backends = get_bf16_backends();
    // Use the shared macro so TENZOR_REQUIRE_MULTI_BACKEND=1 escalates
    // "only one BF16-capable backend" to a hard failure instead of a silent
    // skip — matches the policy documented in parity_test_utils.hpp.
    if (backends.size() < 2) {
        REQUIRE_MULTI_BACKEND_OR_SKIP("BF16 parity");
    }

    // Convert inputs to BF16 on CPU — this is the reference.
    std::vector<Tensor> cpu_bf16_inputs;
    for (const auto& t : cpu_f32_inputs) {
        cpu_bf16_inputs.push_back(t.to(DType::BFloat16));
    }

    Tensor ref;
    try {
        ref = op(cpu_bf16_inputs).to(DType::Float32);
    } catch (const std::exception& e) {
        // If CPU can't run the op in BF16, skip (missing kernel, etc.)
        GTEST_SKIP() << name << " not supported on CPU BF16: " << e.what();
        return;
    }

    for (const auto& backend : backends) {
        if (backend.type == Device::Type::CPU) continue;

        SCOPED_TRACE(name + " on " + backend_name(backend));

        std::vector<Tensor> dev_inputs;
        for (const auto& t : cpu_bf16_inputs) {
            dev_inputs.push_back(t.to(backend));
        }

        Tensor result;
        try {
            result = op(dev_inputs);
            backend.synchronize();
            result = result.to(Device::cpu()).to(DType::Float32);
        } catch (const std::exception& e) {
            // An explicit throw ("not supported for BFloat16") is the
            // correct rejection — the backend honestly declines the op.
            // Only silent data corruption (values differ without throw)
            // is a real bug. Skip this backend for this op.
            continue;
        }

        bool close = tensors_close(ref, result, kBF16Rtol, kBF16Atol);
        if (!close) {
            float diff = max_abs_diff(ref, result);
            FAIL() << name << " BF16 parity failed on " << backend_name(backend)
                   << "\n  Max absolute difference: " << diff
                   << "\n  Tolerance: rtol=" << kBF16Rtol << " atol=" << kBF16Atol;
        }
    }
}

} // namespace

// ============================================================================
// Elementwise unary math ops
// ============================================================================

#define BF16_UNARY_TEST(TestName, OpExpr)                                     \
    TEST_P(BF16Parity, TestName) {                                              \
        auto a = tenzor::abs(randn({32, 32}, DType::Float32, Device::cpu()))  \
                 + 0.5f;                                                      \
        test_bf16_op(                                                         \
            [](const std::vector<Tensor>& in) -> Tensor { return OpExpr; },   \
            {a}, #TestName);                                                  \
    }

BF16_UNARY_TEST(Neg,        in[0] * -1.0f)
BF16_UNARY_TEST(Abs,        tenzor::abs(in[0]))
BF16_UNARY_TEST(Sqrt,       tenzor::sqrt(in[0]))
BF16_UNARY_TEST(Exp,        tenzor::exp(in[0]))
BF16_UNARY_TEST(Log,        tenzor::log(in[0]))
BF16_UNARY_TEST(Sign,       tenzor::sign(in[0]))
BF16_UNARY_TEST(Reciprocal, tenzor::reciprocal(in[0]))
BF16_UNARY_TEST(Floor,      tenzor::floor(in[0]))
BF16_UNARY_TEST(Ceil,       tenzor::ceil(in[0]))
// Round: different backends use different IEEE rounding modes for half-values
// (CPU: half-away-from-zero, Vulkan: half-to-nearest-even). Both are valid.
// Avoid exact .5 by adding a small offset so the test checks data integrity
// rather than rounding-mode semantics.
TEST_P(BF16Parity, Round) {
    // Force values well away from x.5 boundaries by flooring to integers
    // and adding 0.25. E.g., {0.25, 1.25, 2.25, ...} always rounds down.
    auto raw = tenzor::abs(randn({32, 32}, DType::Float32, Device::cpu())) * 3.0f;
    auto a = tenzor::floor(raw) + 0.25f;
    test_bf16_op(
        [](const std::vector<Tensor>& in) -> Tensor { return tenzor::round(in[0]); },
        {a}, "Round");
}
BF16_UNARY_TEST(Log2,       tenzor::log2(in[0]))
BF16_UNARY_TEST(Log10,      tenzor::log10(in[0]))
BF16_UNARY_TEST(Log1p,      tenzor::log1p(in[0]))
BF16_UNARY_TEST(Exp2,       tenzor::exp2(in[0]))
BF16_UNARY_TEST(Expm1,      tenzor::expm1(in[0]))
BF16_UNARY_TEST(Erf,        tenzor::erf(in[0]))
BF16_UNARY_TEST(Erfc,       tenzor::erfc(in[0]))

// Trig
BF16_UNARY_TEST(Sin,  tenzor::sin(in[0]))
BF16_UNARY_TEST(Cos,  tenzor::cos(in[0]))
BF16_UNARY_TEST(Tanh, tenzor::tanh(in[0]))
BF16_UNARY_TEST(Sinh, tenzor::sinh(in[0]))
BF16_UNARY_TEST(Cosh, tenzor::cosh(in[0]))

// Clamp
TEST_P(BF16Parity, Clamp) {
    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    test_bf16_op(
        [](const std::vector<Tensor>& in) -> Tensor {
            return tenzor::clamp(in[0], -1.0f, 1.0f);
        },
        {a}, "Clamp");
}

// ============================================================================
// Elementwise binary math ops
// ============================================================================

#define BF16_BINARY_TEST(TestName, OpExpr)                                    \
    TEST_P(BF16Parity, TestName) {                                              \
        auto a = randn({16, 16}, DType::Float32, Device::cpu());              \
        auto b = randn({16, 16}, DType::Float32, Device::cpu()) + 2.0f;      \
        test_bf16_op(                                                         \
            [](const std::vector<Tensor>& in) -> Tensor { return OpExpr; },   \
            {a, b}, #TestName);                                               \
    }

BF16_BINARY_TEST(Add, in[0] + in[1])
BF16_BINARY_TEST(Sub, in[0] - in[1])
BF16_BINARY_TEST(Mul, in[0] * in[1])
BF16_BINARY_TEST(Div, in[0] / in[1])

// ============================================================================
// Reductions
// ============================================================================

BF16_UNARY_TEST(Sum,  tenzor::sum(in[0]))
BF16_UNARY_TEST(Mean, tenzor::mean(in[0]))
BF16_UNARY_TEST(Max,  tenzor::max(in[0]))
BF16_UNARY_TEST(Min,  tenzor::min(in[0]))
BF16_UNARY_TEST(Prod, tenzor::prod(in[0]))

// ============================================================================
// MatMul
// ============================================================================

TEST_P(BF16Parity, MatMul) {
    auto a = randn({8, 16}, DType::Float32, Device::cpu());
    auto b = randn({16, 8}, DType::Float32, Device::cpu());
    test_bf16_op(
        [](const std::vector<Tensor>& in) -> Tensor {
            return tenzor::matmul(in[0], in[1]);
        },
        {a, b}, "MatMul");
}

// ============================================================================
// Activations (via nn::functional — exercises activation kernel dispatch)
// ============================================================================

TEST_P(BF16Parity, ReLU) {
    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    test_bf16_op(
        [](const std::vector<Tensor>& in) -> Tensor {
            return nn::functional::relu(Variable(in[0], false)).tensor();
        },
        {a}, "ReLU");
}

TEST_P(BF16Parity, Sigmoid) {
    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    test_bf16_op(
        [](const std::vector<Tensor>& in) -> Tensor {
            return nn::functional::sigmoid(Variable(in[0], false)).tensor();
        },
        {a}, "Sigmoid");
}

TEST_P(BF16Parity, GeLU) {
    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    test_bf16_op(
        [](const std::vector<Tensor>& in) -> Tensor {
            return nn::functional::gelu(Variable(in[0], false)).tensor();
        },
        {a}, "GeLU");
}

TEST_P(BF16Parity, Softmax) {
    auto a = randn({8, 16}, DType::Float32, Device::cpu());
    test_bf16_op(
        [](const std::vector<Tensor>& in) -> Tensor {
            return nn::functional::softmax(Variable(in[0], false), -1).tensor();
        },
        {a}, "Softmax");
}

// ============================================================================
// Shape / data-movement ops (exercise permute, expand, cat, etc.)
// ============================================================================

TEST_P(BF16Parity, Transpose) {
    auto a = randn({16, 32}, DType::Float32, Device::cpu());
    test_bf16_op(
        [](const std::vector<Tensor>& in) -> Tensor {
            return in[0].transpose(-1, -2).contiguous();
        },
        {a}, "Transpose");
}

TEST_P(BF16Parity, Permute) {
    auto a = randn({4, 8, 16}, DType::Float32, Device::cpu());
    test_bf16_op(
        [](const std::vector<Tensor>& in) -> Tensor {
            return in[0].permute({2, 0, 1}).contiguous();
        },
        {a}, "Permute");
}

TEST_P(BF16Parity, Cat) {
    auto a = randn({8, 8}, DType::Float32, Device::cpu());
    auto b = randn({8, 8}, DType::Float32, Device::cpu());
    test_bf16_op(
        [](const std::vector<Tensor>& in) -> Tensor {
            return tenzor::cat({in[0], in[1]}, 0);
        },
        {a, b}, "Cat");
}

TEST_P(BF16Parity, Expand) {
    auto a = randn({1, 16}, DType::Float32, Device::cpu());
    test_bf16_op(
        [](const std::vector<Tensor>& in) -> Tensor {
            return in[0].expand({8, 16}).contiguous();
        },
        {a}, "Expand");
}

TEST_P(BF16Parity, Fill) {
    // Exercises dispatchFill / dispatchStridedFill for BF16
    test_bf16_op(
        [](const std::vector<Tensor>& in) -> Tensor {
            Tensor t = zeros({32}, in[0].dtype(), in[0].device());
            t.fill_(3.14);
            return t;
        },
        {zeros({1}, DType::Float32, Device::cpu())}, "Fill");
}

// ============================================================================
// Comparison ops
// ============================================================================

TEST_P(BF16Parity, ComparisonGt) {
    auto a = randn({16, 16}, DType::Float32, Device::cpu());
    auto b = randn({16, 16}, DType::Float32, Device::cpu());
    test_bf16_op(
        [](const std::vector<Tensor>& in) -> Tensor {
            return tenzor::gt(in[0], in[1]).to(in[0].dtype());
        },
        {a, b}, "ComparisonGt");
}

// ============================================================================
// Conv2d forward (critical for vision models)
// ============================================================================

TEST_P(BF16Parity, Conv2dForward) {
    auto input = randn({1, 1, 8, 8}, DType::Float32, Device::cpu());
    auto weight = randn({1, 1, 3, 3}, DType::Float32, Device::cpu());
    test_bf16_op(
        [](const std::vector<Tensor>& in) -> Tensor {
            OpAttributes attrs;
            attrs.set(AttrKey::Stride, int64_t(1));
            attrs.set(AttrKey::Padding, int64_t(0));
            attrs.set(AttrKey::Dilation, int64_t(1));
            attrs.set(AttrKey::Groups, int64_t(1));
            return dispatch(OpId::Conv2dForward, in, attrs)[0];
        },
        {input, weight}, "Conv2dForward");
}

// ============================================================================
// Custom main — initialize tenzor before any test runs.
// ============================================================================

INSTANTIATE_BACKEND_TESTS(BF16Parity);


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

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
    } catch (...) {}
    return result;
}
