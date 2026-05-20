// Phase 13 / Group C.4 — Linalg ops via the MLIR backend.
//
// Covered ops:
//   MatMul → stablehlo.dot_general
//   Bmm    → stablehlo.dot_general with batching_dims=[0]
//   Linear → stablehlo.dot_general + optional add(bias)

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

}  // namespace

TEST(OpLinalg, MatMul2D) {
    ensure_core_init();
    // Closed-over W captured by the lambda. randn() must run OUTSIDE the
    // lambda — otherwise it fires on every trace pass and graph-breaks
    // because OpId::Randn isn't in the OpId→OpType mapping.
    auto W = ::tenzor::randn({8, 6}, ::tenzor::DType::Float32);
    ::tenzor::Variable wv(W, false);

    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir"; cfg.target = "llvm-cpu";

    auto fn = [wv](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        return ::tenzor::matmul(x, wv);
    };
    auto compiled = ::tenzor::jit::CompiledFunction(fn, cfg);

    auto raw = ::tenzor::randn({4, 8}, ::tenzor::DType::Float32);
    ::tenzor::Variable x(raw, false);
    auto eager  = fn(x);
    auto jitted = compiled(x);
    auto diff = ::tenzor::max(::tenzor::abs(
        eager.tensor() - jitted.tensor())).template item<float>();
    EXPECT_LT(diff, 1e-3F) << "matmul diff=" << diff;
}

TEST(OpLinalg, Bmm3D) {
    ensure_core_init();
    auto W = ::tenzor::randn({3, 5, 4}, ::tenzor::DType::Float32);
    ::tenzor::Variable wv(W, false);

    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir"; cfg.target = "llvm-cpu";

    auto fn = [wv](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        return ::tenzor::bmm(x, wv);
    };
    auto compiled = ::tenzor::jit::CompiledFunction(fn, cfg);

    auto raw = ::tenzor::randn({3, 4, 5}, ::tenzor::DType::Float32);
    ::tenzor::Variable x(raw, false);
    auto eager  = fn(x);
    auto jitted = compiled(x);
    auto diff = ::tenzor::max(::tenzor::abs(
        eager.tensor() - jitted.tensor())).template item<float>();
    EXPECT_LT(diff, 1e-3F) << "bmm diff=" << diff;
}

TEST(OpLinalg, LinearWithBias) {
    ensure_core_init();
    auto W = ::tenzor::randn({6, 8}, ::tenzor::DType::Float32);
    auto b = ::tenzor::randn({6}, ::tenzor::DType::Float32);
    ::tenzor::Variable wv(W, false), bv(b, false);

    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir"; cfg.target = "llvm-cpu";

    auto fn = [wv, bv](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        return ::tenzor::linear(x, wv, bv);
    };
    auto compiled = ::tenzor::jit::CompiledFunction(fn, cfg);

    auto raw = ::tenzor::randn({4, 8}, ::tenzor::DType::Float32);
    ::tenzor::Variable x(raw, false);
    auto eager  = fn(x);
    auto jitted = compiled(x);
    auto diff = ::tenzor::max(::tenzor::abs(
        eager.tensor() - jitted.tensor())).template item<float>();
    EXPECT_LT(diff, 1e-3F) << "linear diff=" << diff;
}

// ---------------------------------------------------------------------------
// Audit item A.8 — unequal-rank MatMul must broadcast the smaller operand.
//
// PyTorch / NumPy MatMul semantics: with lhs of rank R and rhs of rank 2,
// the lhs's leading R-2 dims are batch dims; rhs is broadcast across them.
// Previously the MLIR lowering set empty batch lists when ranks differed,
// producing a stablehlo.dot_general with wrong batch semantics.
// ---------------------------------------------------------------------------
TEST(OpLinalg, MatMulUnequalRankBatchLhs) {
    ensure_core_init();
    auto W = ::tenzor::randn({8, 6}, ::tenzor::DType::Float32);
    ::tenzor::Variable wv(W, false);

    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir"; cfg.target = "llvm-cpu";

    auto fn = [wv](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        // lhs: (B=2, M=4, K=8), rhs: (K=8, N=6) ⇒ out (B=2, M=4, N=6)
        return ::tenzor::matmul(x, wv);
    };
    auto compiled = ::tenzor::jit::CompiledFunction(fn, cfg);

    auto raw = ::tenzor::randn({2, 4, 8}, ::tenzor::DType::Float32);
    ::tenzor::Variable x(raw, false);
    auto eager  = fn(x);
    auto jitted = compiled(x);
    auto diff = ::tenzor::max(::tenzor::abs(
        eager.tensor() - jitted.tensor())).template item<float>();
    EXPECT_LT(diff, 1e-3F) << "matmul rank-3 lhs @ rank-2 rhs diff=" << diff;
}

TEST(OpLinalg, MatMulUnequalRankBatchRhs) {
    ensure_core_init();
    // lhs of rank 2, rhs of rank 3 ⇒ rhs is broadcast over the batch dim
    // of the OUTPUT (PyTorch matmul: result shape is (B, M, N)).
    auto W = ::tenzor::randn({3, 8, 6}, ::tenzor::DType::Float32);
    ::tenzor::Variable wv(W, false);

    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir"; cfg.target = "llvm-cpu";

    auto fn = [wv](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        return ::tenzor::matmul(x, wv);
    };
    auto compiled = ::tenzor::jit::CompiledFunction(fn, cfg);

    auto raw = ::tenzor::randn({4, 8}, ::tenzor::DType::Float32);
    ::tenzor::Variable x(raw, false);
    auto eager  = fn(x);
    auto jitted = compiled(x);
    auto diff = ::tenzor::max(::tenzor::abs(
        eager.tensor() - jitted.tensor())).template item<float>();
    EXPECT_LT(diff, 1e-3F) << "matmul rank-2 lhs @ rank-3 rhs diff=" << diff;
}
