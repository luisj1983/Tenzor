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
#include "mlir_target_util.hpp"

#include <functional>
#include <string>
#include <vector>

namespace {

using FnT = ::tenzor::jit::CompiledFunction::FnType;

// Compile `fn` through the MLIR backend for EVERY IREE target usable on this
// host (llvm-cpu always; cuda / rocm / vulkan-spirv when present) and assert
// each target's JIT output matches the eager reference. Inputs live on CPU;
// IREE marshals them onto the target device, so this exercises the GPU IREE
// code paths without needing the corresponding Tenzor backend loaded.
void check_matches_eager(const std::string& name, FnT fn,
                         std::vector<int64_t> shape = {16},
                         ::tenzor::DType dt = ::tenzor::DType::Float32,
                         float tol = 1e-5F,
                         bool positive_inputs = false) {
    namespace mt = ::tenzor::testing::mlir;
    mt::ensure_core_init();

    auto raw = ::tenzor::randn(shape, dt);
    if (positive_inputs) {
        // For log/sqrt/pow: shift to ≥ 0.1 by abs + 0.1.
        raw = ::tenzor::abs(raw) + 0.1F;
    }
    auto x = ::tenzor::Variable(raw, /*requires_grad=*/false);
    auto eager = fn(x);
    auto eager_cpu = eager.tensor().to(::tenzor::Device::cpu());

    const auto targets = mt::available_iree_targets();
    ASSERT_FALSE(targets.empty()) << "no IREE target available (expected >=llvm-cpu)";
    for (const auto& target : targets) {
        ::tenzor::jit::CompileConfig cfg;
        cfg.backend = "mlir";
        cfg.target  = target;
        auto compiled = ::tenzor::jit::CompiledFunction(fn, cfg);

        mt::reset_jit_stats();
        auto jitted = compiled(x);
        mt::assert_jit_used(name, target);
        auto jitted_cpu = jitted.tensor().to(::tenzor::Device::cpu());
        // Compare in Float32 so the diff is not re-rounded in a low-precision
        // dtype and item<float>() is valid for F16/BF16 outputs.
        auto diff = ::tenzor::max(::tenzor::abs(
            eager_cpu.to(::tenzor::DType::Float32) -
            jitted_cpu.to(::tenzor::DType::Float32))).template item<float>();
        EXPECT_LT(diff, tol)
            << "op=" << name << " target=" << target << " diff=" << diff;
    }
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

// JIT-F061: pow with a NEGATIVE integer exponent and a negative base must be
// finite on GPU HAL targets (lowered as repeated-multiply + reciprocal), not
// NaN from stablehlo.power's exp(y*log(x)). Compared to eager on every target.
TEST(OpElementwise, PowNegativeIntExponent) {
    check_matches_eager("pow_neg2", [](const ::tenzor::Variable& x) {
        // x^-2 = 1/x^2 — finite for negative x (eager std::pow gives finite).
        return ::tenzor::pow(x, -2.0);
    }, {16}, ::tenzor::DType::Float32, 5e-3F);
}

// JIT-F037/F039: F16 transcendental must be widened to F32 in the lowering
// (exercised end-to-end on every available IREE target, incl. GPU).
TEST(OpElementwise, ExpF16) {
    check_matches_eager("exp_f16", [](const ::tenzor::Variable& x) {
        auto y = x * 0.1F;
        return ::tenzor::exp(y);
    }, {16}, ::tenzor::DType::Float16, 5e-3F);
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

// =====================================================================
// End-to-end @tz.jit test for the ResidualAdd marker path. The tracer
// only records OpType::Add (the eager `+` is an Add); the
// FuseResidualAddPass in src/jit/compiler.cpp recognises the
// `x + sublayer(x)` topology and tags the Add node with
// `residual=true`. The MLIR lowering in handle_binary uses the same
// stablehlo.add emit for both Add and ResidualAdd OpTypes, so the
// numerical output is identical; the assertion that matters is the
// pipeline (trace -> pass -> lower -> iree-compile -> invoke) does not
// throw on the residual-shape topology — any mismatch in the
// pattern matcher's value_feeds_into traversal or downstream lowering
// would surface here.
// =====================================================================

#include "tenzor/nn/functional.hpp"

TEST(OpElementwise, ResidualAddEndToEnd) {
    check_matches_eager("residual_add", [](const ::tenzor::Variable& x) {
        // ReLU is in FuseResidualAddPass::is_sublayer_op's allow-list,
        // so the trace `Add(x, ReLU(x))` lights up the residual
        // pattern. After the pass, the Add node carries
        // residual=true; lowering still emits stablehlo.add.
        auto sub = ::tenzor::nn::functional::relu(x);
        return x + sub;
    });
}
