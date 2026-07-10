// JIT-F013 regression — the MLIR backend must NOT drop fused epilogues.
//
// optimize_for_inference (FuseConvReluPass / FuseLinearReluPass /
// FuseMatMulAddPass / FuseLayerNormActivationPass) folds a following
// ReLU / bias-add / activation into the producer node and deletes the
// separate node, encoding it as a fused_relu / fused_bias / fused_activation
// marker. The interpreter/NVRTC path applies those markers; the StableHLO
// lowerer previously ignored them, so Conv+ReLU, Linear+ReLU and MatMul+bias
// compiled through IREE ran with the epilogue MISSING on every target —
// silently wrong, no error, no eager fallback. These tests compile each fused
// pattern on every available IREE target with fusion enabled (the default) and
// assert the JIT output matches eager. A dropped epilogue fails the tolerance.

#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/jit/compile.hpp"
#include "tenzor/nn/functional.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/tenzor.hpp"

#include <gtest/gtest.h>

#include "mlir_target_util.hpp"

#include <string>

namespace {

namespace F = ::tenzor::nn::functional;

// Compile `fn` on every available IREE target with fusion ENABLED and assert the
// output matches the eager reference. Mirrors test_op_linalg's helper.
inline void run_over_targets(const std::string& name,
                             ::tenzor::jit::CompiledFunction::FnType fn,
                             const ::tenzor::Variable& x, float tol) {
    namespace mt = ::tenzor::testing::mlir;
    auto eager_cpu = fn(x).tensor().to(::tenzor::Device::cpu())
                          .to(::tenzor::DType::Float32);
    const auto targets = mt::available_iree_targets();
    ASSERT_FALSE(targets.empty()) << "no IREE target available";
    for (const auto& target : targets) {
        ::tenzor::jit::CompileConfig cfg;
        cfg.backend = "mlir";
        cfg.target  = target;
        cfg.enable_fusion = true;  // exercise the fused-epilogue path
        auto compiled = ::tenzor::jit::CompiledFunction(fn, cfg);
        mt::reset_jit_stats();
        auto jitted = compiled(x);
        mt::assert_jit_used(name, target);
        auto jit_cpu = jitted.tensor().to(::tenzor::Device::cpu())
                             .to(::tenzor::DType::Float32);
        auto diff = ::tenzor::max(::tenzor::abs(eager_cpu - jit_cpu))
                        .template item<float>();
        EXPECT_LT(diff, tol) << "op=" << name << " target=" << target
                             << " diff=" << diff
                             << " (fused epilogue likely dropped)";
    }
}

auto ensure_core_init() -> void {
    static const bool inited = []() {
        ::tenzor::initialize();
        return true;
    }();
    (void)inited;
}

}  // namespace

// Linear + ReLU: FuseLinearReluPass marks fused_relu and deletes the ReLU node.
TEST(FusedEpilogue, LinearReluMatchesEager) {
    ensure_core_init();
    auto W = ::tenzor::randn({6, 8}, ::tenzor::DType::Float32);
    auto b = ::tenzor::randn({6}, ::tenzor::DType::Float32);
    ::tenzor::Variable wv(W, false), bv(b, false);
    auto fn = [wv, bv](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        return ::tenzor::nn::relu(::tenzor::linear(x, wv, bv));
    };
    auto raw = ::tenzor::randn({4, 8}, ::tenzor::DType::Float32);
    ::tenzor::Variable x(raw, false);
    run_over_targets("linear_relu", fn, x, 1e-3F);
}

// MatMul + bias: FuseMatMulAddPass folds the Add into the MatMul as a 3rd input.
TEST(FusedEpilogue, MatMulBiasMatchesEager) {
    ensure_core_init();
    auto W = ::tenzor::randn({8, 6}, ::tenzor::DType::Float32);
    auto b = ::tenzor::randn({6}, ::tenzor::DType::Float32);
    ::tenzor::Variable wv(W, false), bv(b, false);
    auto fn = [wv, bv](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        return ::tenzor::matmul(x, wv) + bv;
    };
    auto raw = ::tenzor::randn({4, 8}, ::tenzor::DType::Float32);
    ::tenzor::Variable x(raw, false);
    run_over_targets("matmul_bias", fn, x, 1e-3F);
}

// MatMul + bias + ReLU: both fused_bias and fused_relu on the same node.
TEST(FusedEpilogue, MatMulBiasReluMatchesEager) {
    ensure_core_init();
    auto W = ::tenzor::randn({8, 6}, ::tenzor::DType::Float32);
    auto b = ::tenzor::randn({6}, ::tenzor::DType::Float32);
    ::tenzor::Variable wv(W, false), bv(b, false);
    auto fn = [wv, bv](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        return ::tenzor::nn::relu(::tenzor::matmul(x, wv) + bv);
    };
    auto raw = ::tenzor::randn({4, 8}, ::tenzor::DType::Float32);
    ::tenzor::Variable x(raw, false);
    run_over_targets("matmul_bias_relu", fn, x, 1e-3F);
}

// Conv2d + ReLU: FuseConvReluPass marks fused_relu and deletes the ReLU node.
TEST(FusedEpilogue, Conv2dReluMatchesEager) {
    ensure_core_init();
    auto W = ::tenzor::randn({4, 3, 3, 3}, ::tenzor::DType::Float32);
    ::tenzor::Variable wv(W, false);
    auto fn = [wv](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        return ::tenzor::nn::relu(
            F::conv2d(x, wv, std::nullopt, {1, 1}, {1, 1}));
    };
    auto raw = ::tenzor::randn({2, 3, 8, 8}, ::tenzor::DType::Float32);
    ::tenzor::Variable x(raw, false);
    run_over_targets("conv2d_relu", fn, x, 2e-3F);
}
