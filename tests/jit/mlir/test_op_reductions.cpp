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
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/tenzor.hpp"

#include <gtest/gtest.h>

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
    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target  = "llvm-cpu";
    auto compiled = ::tenzor::jit::CompiledFunction(fn, cfg);

    auto raw = ::tenzor::randn(shape, dt);
    auto x = ::tenzor::Variable(raw, /*requires_grad=*/false);

    auto eager  = fn(x);
    auto jitted = compiled(x);

    auto diff = ::tenzor::max(::tenzor::abs(
        eager.tensor() - jitted.tensor())).template item<float>();
    EXPECT_LT(diff, tol) << "op=" << name << " diff=" << diff;
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
