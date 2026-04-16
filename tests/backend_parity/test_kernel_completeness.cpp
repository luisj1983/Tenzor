/**
 * @file test_kernel_completeness.cpp
 * @brief Verify that all required operations have registered kernels on each backend.
 *
 * Phase 4A: For each compute backend (CPU, CUDA, ROCm, Vulkan, OneAPI), checks that
 * a curated set of "required" OpIds are present in the dispatch table.  Specialized,
 * fused, backward-only, in-place, and creation ops are excluded from the required set.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/op_id.hpp>
#include <tenzor/backend/dispatch_table.hpp>
#include "parity_test_utils.hpp"
#include <vector>
#include <string>
#include <sstream>

using namespace tenzor;
using namespace tenzor::testing;

namespace {

// ---------------------------------------------------------------------------
// Helper: curated list of ops every compute backend must support
// ---------------------------------------------------------------------------
std::vector<OpId> get_required_ops() {
    return {
        // Arithmetic (0-6)
        OpId::Add, OpId::Sub, OpId::Mul, OpId::Div,
        OpId::MatMul, OpId::Bmm, OpId::Dot,

        // Reductions (11-20)
        OpId::Sum, OpId::Mean, OpId::Max, OpId::Min,
        OpId::ArgMax, OpId::ArgMin, OpId::Prod,
        OpId::Var, OpId::Std, OpId::Norm,

        // Element-wise math (30-42)
        OpId::Sqrt, OpId::Neg, OpId::Abs, OpId::Sign,
        OpId::Log, OpId::Exp, OpId::Pow, OpId::Clamp,
        OpId::Reciprocal, OpId::Floor, OpId::Ceil, OpId::Round,

        // Trigonometric (50-61)
        OpId::Sin, OpId::Cos, OpId::Tan,
        OpId::Asin, OpId::Acos, OpId::Atan,
        OpId::Sinh, OpId::Cosh, OpId::Tanh,
        OpId::Asinh, OpId::Acosh, OpId::Atanh,

        // Activations — forward only (65-94)
        OpId::ReLU, OpId::Sigmoid, OpId::TanhActivation,
        OpId::Gelu, OpId::Swish, OpId::LeakyReLU,
        OpId::Elu, OpId::Selu, OpId::Mish,
        OpId::Softplus, OpId::Softmax, OpId::LogSoftmax,
        OpId::LogSigmoid,

        // Shape/View (100-114)
        OpId::Reshape, OpId::Transpose, OpId::Permute,
        OpId::Squeeze, OpId::Unsqueeze, OpId::Flatten,
        OpId::Contiguous, OpId::Clone, OpId::Fill,
        OpId::Repeat, OpId::Tile, OpId::Expand,
        OpId::Stack, OpId::Split, OpId::Chunk,

        // Indexing (120-130)
        OpId::IndexSelect, OpId::Gather, OpId::Scatter,
        OpId::MaskedSelect, OpId::MaskedFill, OpId::Where,
        OpId::Slice, OpId::Cat, OpId::Take, OpId::Put,
        OpId::Nonzero,

        // Comparison (140-145)
        OpId::Eq, OpId::Ne, OpId::Lt, OpId::Le, OpId::Gt, OpId::Ge,

        // Conv2d forward (170)
        OpId::Conv2dForward,

        // Pooling forward (190-196)
        OpId::MaxPool2dForward, OpId::AvgPool2dForward,
        OpId::AdaptiveAvgPool2d, OpId::AdaptiveMaxPool2d,

        // Embedding (260)
        OpId::Embedding,

        // Linear (270)
        OpId::Linear,

        // Cast (316)
        OpId::Cast,

        // Extended math (320-335)
        OpId::Log2, OpId::Log10, OpId::Log1p,
        OpId::Exp2, OpId::Expm1,
        OpId::Erf, OpId::Erfc,
        OpId::Atan2, OpId::Fmod, OpId::Remainder,
        OpId::Hypot, OpId::Copysign,

        // Manipulation (340-345)
        OpId::Triu, OpId::Tril, OpId::Diag,
        OpId::Trace, OpId::Flip, OpId::Roll,

        // Logical (350-353)
        OpId::LogicalAnd, OpId::LogicalOr,
        OpId::LogicalNot, OpId::LogicalXor,

        // Complex (440-444)
        OpId::Conj, OpId::Real, OpId::Imag,
        OpId::Angle, OpId::Polar,
    };
}

// ---------------------------------------------------------------------------
// Helper: join a vector of strings with ", "
// ---------------------------------------------------------------------------
std::string join(const std::vector<std::string>& items) {
    std::ostringstream oss;
    for (size_t i = 0; i < items.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << items[i];
    }
    return oss.str();
}

// ---------------------------------------------------------------------------
// Generic checker used by each per-backend TEST()
// ---------------------------------------------------------------------------
void check_backend_completeness(Device::Type device_type, const char* backend_name) {
    tenzor::initialize();

    const auto& table = DispatchTableRegistry::get_table_const(device_type);
    const auto required = get_required_ops();

    std::vector<std::string> missing;
    for (auto op : required) {
        if (!table.has_kernel(op)) {
            missing.emplace_back(std::string(op_id_to_name(op)));
        }
    }

    EXPECT_TRUE(missing.empty())
        << backend_name << " backend is missing " << missing.size()
        << " of " << required.size() << " required kernels:\n  "
        << join(missing);
}

}  // namespace

// ===========================================================================
// Per-backend completeness tests
// ===========================================================================

TEST(KernelCompleteness, CPU) {
    check_backend_completeness(Device::Type::CPU, "CPU");
}

TEST(KernelCompleteness, CUDA) {
    if (!has_cuda()) GTEST_SKIP() << "CUDA backend not available";
    check_backend_completeness(Device::Type::CUDA, "CUDA");
}

TEST(KernelCompleteness, ROCm) {
    if (!has_rocm()) GTEST_SKIP() << "ROCm backend not available";
    check_backend_completeness(Device::Type::ROCm, "ROCm");
}

TEST(KernelCompleteness, Vulkan) {
    if (!has_vulkan()) GTEST_SKIP() << "Vulkan backend not available";
    check_backend_completeness(Device::Type::Vulkan, "Vulkan");
}

TEST(KernelCompleteness, OneAPI) {
    if (!has_oneapi()) GTEST_SKIP() << "OneAPI backend not available";
    check_backend_completeness(Device::Type::OneAPI, "OneAPI");
}
