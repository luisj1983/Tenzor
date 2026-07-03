// Phase 13 / Group C.1 — `tz.where` end-to-end via the MLIR backend.
//
// Pipeline under test:
//   `Variable x` -> tracer -> @tz.jit (mlir/llvm-cpu)
//     -> stablehlo.select via handle_where
//     -> iree-compile -> IreeInvoker
//     -> output must match eager `tenzor::where`.
//
// The condition tensor is built inline from a Bool-typed constant.
// `tenzor::full(shape, 1.0F, DType::Bool)` dispatches OpId::Full (which
// is intentionally unmapped in the tracer so the value becomes a frozen
// graph constant — exactly the semantic we need for a non-trainable
// mask). The Variable input `x` flows through the traced Where as the
// `on_true` operand; a Bool 1-mask makes the test verify that select
// returns `on_true` for every element and `on_false` for none.

#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/jit/compile.hpp"
#include "tenzor/jit/mlir/iree_compile.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/tenzor.hpp"

#include <gtest/gtest.h>

#include <vector>

namespace {

auto ensure_core_init() -> void {
    static const bool inited = []() {
        ::tenzor::initialize();
        return true;
    }();
    (void)inited;
}

}  // namespace

TEST(OpWhere, WhereSelectsOnTrueWhenMaskIsAllTrue) {
    ensure_core_init();

    auto fn = [](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        auto cond = ::tenzor::full({16}, 1.0F, ::tenzor::DType::Bool);
        auto other = ::tenzor::full({16}, -7.0F, ::tenzor::DType::Float32);
        ::tenzor::Variable cond_v(cond, /*requires_grad=*/false);
        ::tenzor::Variable other_v(other, /*requires_grad=*/false);
        return ::tenzor::where(cond_v, x, other_v);
    };

    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target  = "llvm-cpu";
    auto compiled = ::tenzor::jit::CompiledFunction(fn, cfg);

    auto x = ::tenzor::Variable(
        ::tenzor::full({16}, 3.0F, ::tenzor::DType::Float32),
        /*requires_grad=*/false);

    auto eager = fn(x);
    ::tenzor::jit::mlir_jit::reset_cache_stats();
    auto jit   = compiled(x);
    EXPECT_GE(::tenzor::jit::mlir_jit::cache_stats().misses, 1u)
        << "op did not run through IREE (silent eager fallback; llvm-cpu)";

    auto eager_cpu = eager.tensor().to(::tenzor::Device::cpu());
    auto jit_cpu   = jit.tensor().to(::tenzor::Device::cpu());
    auto diff = ::tenzor::max(::tenzor::abs(eager_cpu - jit_cpu))
                    .template item<float>();
    EXPECT_LT(diff, 1e-5F);
}

TEST(OpWhere, WhereSelectsOnFalseWhenMaskIsAllFalse) {
    ensure_core_init();

    auto fn = [](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        auto cond = ::tenzor::full({16}, 0.0F, ::tenzor::DType::Bool);
        auto other = ::tenzor::full({16}, -7.0F, ::tenzor::DType::Float32);
        ::tenzor::Variable cond_v(cond, /*requires_grad=*/false);
        ::tenzor::Variable other_v(other, /*requires_grad=*/false);
        return ::tenzor::where(cond_v, x, other_v);
    };

    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target  = "llvm-cpu";
    auto compiled = ::tenzor::jit::CompiledFunction(fn, cfg);

    auto x = ::tenzor::Variable(
        ::tenzor::full({16}, 3.0F, ::tenzor::DType::Float32),
        /*requires_grad=*/false);

    auto eager = fn(x);
    ::tenzor::jit::mlir_jit::reset_cache_stats();
    auto jit   = compiled(x);
    EXPECT_GE(::tenzor::jit::mlir_jit::cache_stats().misses, 1u)
        << "op did not run through IREE (silent eager fallback; llvm-cpu)";

    auto eager_cpu = eager.tensor().to(::tenzor::Device::cpu());
    auto jit_cpu   = jit.tensor().to(::tenzor::Device::cpu());
    auto diff = ::tenzor::max(::tenzor::abs(eager_cpu - jit_cpu))
                    .template item<float>();
    EXPECT_LT(diff, 1e-5F);
}
