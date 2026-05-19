// Phase 13 / Group C.1 — Elementwise ops via the MLIR backend.
//
// Each test compiles a small lambda with backend="mlir", target="llvm-cpu"
// and asserts the JIT output matches eager within a tight tolerance.
//
// Covered ops (lowered to direct StableHLO primitives):
//   Sub   → stablehlo.subtract
//   Mul   → stablehlo.multiply
//   Div   → stablehlo.divide
//   Neg   → stablehlo.negate
//   Abs   → stablehlo.abs
//   Exp   → stablehlo.exponential
//   Log   → stablehlo.log
//   Sqrt  → stablehlo.sqrt
//   Pow   → stablehlo.power
//   Clamp → stablehlo.clamp
//
// Note: Where (stablehlo.select) needs a bool predicate and is covered
// separately when an autograd `where` is wired.

#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/jit/compile.hpp"
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
                         std::vector<int64_t> shape = {16},
                         ::tenzor::DType dt = ::tenzor::DType::Float32,
                         float tol = 1e-5F,
                         bool positive_inputs = false) {
    ensure_core_init();

    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target  = "llvm-cpu";
    auto compiled = ::tenzor::jit::CompiledFunction(fn, cfg);

    auto raw = ::tenzor::randn(shape, dt);
    if (positive_inputs) {
        // For log/sqrt/pow: shift to ≥ 0.1 by abs + 0.1.
        raw = ::tenzor::abs(raw) + 0.1F;
    }
    auto x = ::tenzor::Variable(raw, /*requires_grad=*/false);

    auto eager  = fn(x);
    auto jitted = compiled(x);

    auto diff = ::tenzor::max(::tenzor::abs(
        eager.tensor() - jitted.tensor())).template item<float>();
    EXPECT_LT(diff, tol) << "op=" << name << " diff=" << diff;
}

}  // namespace

TEST(OpElementwise, Sub) {
    check_matches_eager("sub", [](const ::tenzor::Variable& x) {
        return x - x;
    });
}

TEST(OpElementwise, Mul) {
    check_matches_eager("mul", [](const ::tenzor::Variable& x) {
        return x * x;
    });
}

TEST(OpElementwise, Div) {
    // x / (x + 2) — avoid div-by-zero. IREE's reciprocal-then-multiply
    // codegen for divide accumulates slightly more error than eager's
    // direct division, so we widen the F32 tolerance from the default
    // 1e-5 to 1e-4 (still tight relative to the unit-magnitude inputs).
    check_matches_eager("div", [](const ::tenzor::Variable& x) {
        auto denom = x + 2.0F;
        return x / denom;
    }, {16}, ::tenzor::DType::Float32, 1e-4F);
}

TEST(OpElementwise, Neg) {
    check_matches_eager("neg", [](const ::tenzor::Variable& x) {
        return ::tenzor::neg(x);
    });
}

TEST(OpElementwise, Abs) {
    check_matches_eager("abs", [](const ::tenzor::Variable& x) {
        return ::tenzor::abs(x);
    });
}

TEST(OpElementwise, Exp) {
    // Scale magnitude down so exp doesn't overflow F32.
    check_matches_eager("exp", [](const ::tenzor::Variable& x) {
        auto y = x * 0.1F;
        return ::tenzor::exp(y);
    }, {16}, ::tenzor::DType::Float32, 1e-4F);
}

TEST(OpElementwise, Log) {
    check_matches_eager("log", [](const ::tenzor::Variable& x) {
        return ::tenzor::log(x);
    }, {16}, ::tenzor::DType::Float32, 1e-4F, /*positive_inputs=*/true);
}

TEST(OpElementwise, Sqrt) {
    check_matches_eager("sqrt", [](const ::tenzor::Variable& x) {
        return ::tenzor::sqrt(x);
    }, {16}, ::tenzor::DType::Float32, 1e-5F, /*positive_inputs=*/true);
}

TEST(OpElementwise, Pow) {
    check_matches_eager("pow", [](const ::tenzor::Variable& x) {
        return ::tenzor::pow(x, 2.0F);
    }, {16}, ::tenzor::DType::Float32, 1e-4F, /*positive_inputs=*/true);
}

TEST(OpElementwise, Clamp) {
    check_matches_eager("clamp", [](const ::tenzor::Variable& x) {
        return ::tenzor::clamp(x, -0.5F, 0.5F);
    });
}
