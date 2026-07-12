// Phase 13 / Group C.3 — Reduction ops via the MLIR backend.
//
// Covered ops:
//   Sum     → stablehlo.reduce {add} + optional keepdim reshape
//   Mean    → reduce-sum then divide by reduced element count
//   Max     → stablehlo.reduce {maximum} with -inf init
//   Softmax → numerically-stable decomposition along one dim

#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/jit/compile.hpp"
#include "tenzor/jit/mlir/iree_compile.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/tenzor.hpp"

#include <gtest/gtest.h>

#include "mlir_target_util.hpp"

#include <functional>
#include <string>
#include <vector>

namespace {

auto ensure_core_init() -> void {
    static const bool inited = []() {
        ::tenzor::initialize();
        return true;
    }();
    (void)inited;
}

using FnT = ::tenzor::jit::CompiledFunction::FnType;

void check_matches_eager(const std::string& name, FnT fn,
                         std::vector<int64_t> shape = {4, 8},
                         float tol = 1e-5F,
                         ::tenzor::DType dt = ::tenzor::DType::Float32) {
    ensure_core_init();
    namespace mt = ::tenzor::testing::mlir;
    auto raw = ::tenzor::randn(shape, dt);
    auto x = ::tenzor::Variable(raw, /*requires_grad=*/false);
    auto eager = fn(x);
    auto eager_cpu = eager.tensor().to(::tenzor::Device::cpu());

    // Fan out over every available IREE target so the reduction lowerings are
    // numerically validated on the GPU targets too, not just llvm-cpu (JIT-F028).
    const auto targets = mt::available_iree_targets();
    ASSERT_FALSE(targets.empty()) << "no IREE target available";
    for (const auto& target : targets) {
        ::tenzor::jit::CompileConfig cfg;
        cfg.backend = "mlir";
        cfg.target  = target;
        auto compiled = ::tenzor::jit::CompiledFunction(fn, cfg);
        mt::reset_jit_stats();
        auto jitted = compiled(x);
        mt::assert_jit_used(name, target);
        auto jitted_cpu = jitted.tensor().to(::tenzor::Device::cpu());
        auto diff = ::tenzor::max(::tenzor::abs(
            eager_cpu.to(::tenzor::DType::Float32) -
            jitted_cpu.to(::tenzor::DType::Float32))).template item<float>();
        EXPECT_LT(diff, tol) << "op=" << name << " target=" << target
                             << " diff=" << diff;
    }
}

}  // namespace

TEST(OpReductions, SumAll) {
    // Reduce over all dims (no "dim" attr).
    check_matches_eager("sum_all", [](const ::tenzor::Variable& x) {
        return ::tenzor::sum(x);
    });
}

TEST(OpReductions, SumDim) {
    check_matches_eager("sum_dim1", [](const ::tenzor::Variable& x) {
        return ::tenzor::sum(x, /*dim=*/1, /*keepdim=*/false);
    });
}

TEST(OpReductions, SumDimKeepdim) {
    check_matches_eager("sum_dim1_keep", [](const ::tenzor::Variable& x) {
        return ::tenzor::sum(x, /*dim=*/1, /*keepdim=*/true);
    });
}

TEST(OpReductions, MeanDim) {
    check_matches_eager("mean_dim1", [](const ::tenzor::Variable& x) {
        return ::tenzor::mean(x, /*dim=*/1, /*keepdim=*/false);
    });
}

TEST(OpReductions, MaxDim) {
    check_matches_eager("max_dim1", [](const ::tenzor::Variable& x) {
        return ::tenzor::max(x, /*dim=*/1, /*keepdim=*/false);
    });
}

TEST(OpReductions, SoftmaxLastDim) {
    // 5e-5 tolerance: softmax involves exp/divide which accumulate
    // slightly more error than a single elementwise op.
    check_matches_eager("softmax_last", [](const ::tenzor::Variable& x) {
        return ::tenzor::softmax(x, /*dim=*/-1);
    }, {4, 8}, 5e-5F);
}

// findings.txt JIT-R119: these two tests are the primary regression coverage
// for F16-widens-to-F32-then-reduces correctness, yet were hardcoded to
// llvm-cpu despite this same file's check_matches_eager helper already
// fanning out over every available IREE target (JIT-F028) -- the multi-
// target infra existed right here and simply wasn't applied to the two
// tests that matter most for this exact class of risk. Fan out the same way.
TEST(OpReductions, SumFloat16WidensAccumulator) {
    // Sum a long row of 1.0 values in Float16. A half-precision accumulator
    // stops incrementing near ~2048 (1.0 + small rounds away), so an in-half
    // reduction diverges badly from the true 4096; the eager kernel and the
    // (now widened) JIT both accumulate in Float32, so they must agree.
    ensure_core_init();
    namespace mt = ::tenzor::testing::mlir;
    auto fn = [](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        return ::tenzor::sum(x, /*dim=*/1, /*keepdim=*/false);
    };
    auto raw = ::tenzor::ones({1, 4096}, ::tenzor::DType::Float16);
    auto x   = ::tenzor::Variable(raw, /*requires_grad=*/false);
    auto eager  = fn(x).tensor();

    const auto targets = mt::available_iree_targets();
    ASSERT_FALSE(targets.empty()) << "no IREE target available";
    for (const auto& target : targets) {
        ::tenzor::jit::CompileConfig cfg;
        cfg.backend = "mlir";
        cfg.target  = target;
        auto compiled = ::tenzor::jit::CompiledFunction(fn, cfg);
        ::tenzor::jit::mlir_jit::reset_cache_stats();
        auto jitted = compiled(x);
        EXPECT_GE(::tenzor::jit::mlir_jit::cache_stats().misses, 1u)
            << "op did not run through IREE (silent eager fallback; target="
            << target << ")";
        auto diff = ::tenzor::max(::tenzor::abs(
            eager.to(::tenzor::DType::Float32) -
            jitted.tensor().to(::tenzor::Device::cpu()).to(::tenzor::DType::Float32)))
                        .template item<float>();
        EXPECT_LT(diff, 1.0F) << "F16 sum JIT vs eager diff=" << diff
                              << " target=" << target;
    }
}

TEST(OpReductions, MeanFloat16WidensAccumulator) {
    ensure_core_init();
    namespace mt = ::tenzor::testing::mlir;
    auto fn = [](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        return ::tenzor::mean(x, /*dim=*/1, /*keepdim=*/false);
    };
    auto raw = ::tenzor::ones({1, 4096}, ::tenzor::DType::Float16);
    auto x   = ::tenzor::Variable(raw, /*requires_grad=*/false);
    auto eager  = fn(x).tensor();

    const auto targets = mt::available_iree_targets();
    ASSERT_FALSE(targets.empty()) << "no IREE target available";
    for (const auto& target : targets) {
        ::tenzor::jit::CompileConfig cfg;
        cfg.backend = "mlir";
        cfg.target  = target;
        auto compiled = ::tenzor::jit::CompiledFunction(fn, cfg);
        ::tenzor::jit::mlir_jit::reset_cache_stats();
        auto jitted = compiled(x);
        EXPECT_GE(::tenzor::jit::mlir_jit::cache_stats().misses, 1u)
            << "op did not run through IREE (silent eager fallback; target="
            << target << ")";
        auto diff = ::tenzor::max(::tenzor::abs(
            eager.to(::tenzor::DType::Float32) -
            jitted.tensor().to(::tenzor::Device::cpu()).to(::tenzor::DType::Float32)))
                        .template item<float>();
        EXPECT_LT(diff, 1e-2F) << "F16 mean JIT vs eager diff=" << diff
                               << " target=" << target;
    }
}
