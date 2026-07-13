// JIT-R133 — LogicalAnd/Or/Not and the comparison predicates (Eq/Ne/Lt/Le/
// Gt/Ge) have real StableHLO lowerings (src/jit/mlir/lowering.cpp's OpType
// switch, via handle_binary/handle_unary for the logical ops and
// handle_compare for the comparisons) but had zero numeric test coverage on
// any backend before this file.
//
// LogicalAnd/Eq/Ne/Lt/Le/Gt/Ge have Variable-level wrappers (autograd/
// ops.hpp) reachable through the normal tracer. LogicalOr/LogicalNot have NO
// Variable-level wrapper anywhere in the codebase (only a Tensor-level free
// function and an OpId dispatch registration), so they are exercised via a
// raw dispatch(OpId::LogicalOr/LogicalNot, ...) call inside the traced
// closure instead — CompiledFunction's tracing interception operates at the
// dispatch level (tracing_interceptor.cpp's opid_to_optype already maps
// both to their OpType), the same technique test_jit_mlir_numeric_parity.
// cpp's PowNegativeBaseRuntimeTensorExponent test uses to reach a
// comparably wrapper-less op.

#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/jit/compile.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/tenzor.hpp"

#include <gtest/gtest.h>

#include "mlir_target_util.hpp"

#include <vector>

namespace {

auto ensure_core_init() -> void {
    static const bool inited = []() {
        ::tenzor::initialize();
        return true;
    }();
    (void)inited;
}

using ::tenzor::DType;
using ::tenzor::Tensor;
using ::tenzor::Variable;

// Fan out over every available IREE target; compares fn(x) (eager) against
// the compiled result. Bool outputs are widened to Float32 before diffing
// so the same tight-tolerance numeric check applies uniformly to both the
// logical ops (Bool in, Bool out) and the comparison predicates (Float32
// in, Bool out).
void check_matches_eager(const char* name,
                         ::tenzor::jit::CompiledFunction::FnType fn,
                         const Variable& x) {
    namespace mt = ::tenzor::testing::mlir;
    mt::ensure_core_init();
    auto eager = fn(x);
    auto eager_cpu = eager.tensor().to(::tenzor::Device::cpu()).to(DType::Float32);

    const auto targets = mt::available_iree_targets();
    ASSERT_FALSE(targets.empty()) << "no IREE target available";
    for (const auto& target : targets) {
        ::tenzor::jit::CompileConfig cfg;
        cfg.backend = "mlir";
        cfg.target  = target;
        ::tenzor::jit::CompiledFunction compiled(fn, cfg);
        mt::reset_jit_stats();
        auto jit = compiled(x);
        mt::assert_jit_used(name, target);
        auto jit_cpu = jit.tensor().to(::tenzor::Device::cpu()).to(DType::Float32);
        ASSERT_EQ(eager_cpu.numel(), jit_cpu.numel())
            << "op=" << name << " target=" << target;
        auto diff = ::tenzor::max(::tenzor::abs(eager_cpu - jit_cpu)).item<float>();
        EXPECT_LT(diff, 1e-5F)
            << "op=" << name << " target=" << target << " diff=" << diff;
    }
}

// x = [1, 2, 3, 2]; each fn compares against the constant 2.0 -> exercises
// equal/less/greater branches for every comparison op in one shot.
auto make_cmp_input() -> Variable {
    Tensor x_t({4}, DType::Float32, ::tenzor::Device::cpu());
    auto* p = x_t.data<float>();
    p[0] = 1.0F; p[1] = 2.0F; p[2] = 3.0F; p[3] = 2.0F;
    return Variable(x_t, /*requires_grad=*/false);
}

// x = [T, T, F, F]; each fn pairs it against the constant [T, F, T, F] ->
// exercises all 4 truth-table combinations for AND/OR/NOT.
auto make_bool_input() -> Variable {
    Tensor x_t({4}, DType::Bool, ::tenzor::Device::cpu());
    auto* p = x_t.data<bool>();
    p[0] = true; p[1] = true; p[2] = false; p[3] = false;
    return Variable(x_t, /*requires_grad=*/false);
}

auto bool_const_4() -> Tensor {
    Tensor y_t({4}, DType::Bool, ::tenzor::Device::cpu());
    auto* p = y_t.data<bool>();
    p[0] = true; p[1] = false; p[2] = true; p[3] = false;
    return y_t;
}

}  // namespace

TEST(OpLogicalPredicates, EqMatchesEager) {
    ensure_core_init();
    auto fn = [](const Variable& x) -> Variable {
        auto y = ::tenzor::full({4}, 2.0F, DType::Float32);
        return ::tenzor::eq(x, Variable(y, false));
    };
    check_matches_eager("eq", fn, make_cmp_input());
}

TEST(OpLogicalPredicates, NeMatchesEager) {
    ensure_core_init();
    auto fn = [](const Variable& x) -> Variable {
        auto y = ::tenzor::full({4}, 2.0F, DType::Float32);
        return ::tenzor::ne(x, Variable(y, false));
    };
    check_matches_eager("ne", fn, make_cmp_input());
}

TEST(OpLogicalPredicates, LtMatchesEager) {
    ensure_core_init();
    auto fn = [](const Variable& x) -> Variable {
        auto y = ::tenzor::full({4}, 2.0F, DType::Float32);
        return ::tenzor::lt(x, Variable(y, false));
    };
    check_matches_eager("lt", fn, make_cmp_input());
}

TEST(OpLogicalPredicates, LeMatchesEager) {
    ensure_core_init();
    auto fn = [](const Variable& x) -> Variable {
        auto y = ::tenzor::full({4}, 2.0F, DType::Float32);
        return ::tenzor::le(x, Variable(y, false));
    };
    check_matches_eager("le", fn, make_cmp_input());
}

TEST(OpLogicalPredicates, GtMatchesEager) {
    ensure_core_init();
    auto fn = [](const Variable& x) -> Variable {
        auto y = ::tenzor::full({4}, 2.0F, DType::Float32);
        return ::tenzor::gt(x, Variable(y, false));
    };
    check_matches_eager("gt", fn, make_cmp_input());
}

TEST(OpLogicalPredicates, GeMatchesEager) {
    ensure_core_init();
    auto fn = [](const Variable& x) -> Variable {
        auto y = ::tenzor::full({4}, 2.0F, DType::Float32);
        return ::tenzor::ge(x, Variable(y, false));
    };
    check_matches_eager("ge", fn, make_cmp_input());
}

TEST(OpLogicalPredicates, LogicalAndMatchesEager) {
    ensure_core_init();
    auto fn = [](const Variable& x) -> Variable {
        return ::tenzor::logical_and(x, Variable(bool_const_4(), false));
    };
    check_matches_eager("logical_and", fn, make_bool_input());
}

TEST(OpLogicalPredicates, LogicalOrMatchesEager) {
    ensure_core_init();
    auto fn = [](const Variable& x) -> Variable {
        std::vector<Tensor> inputs = {x.tensor(), bool_const_4()};
        auto result = ::tenzor::dispatch(::tenzor::OpId::LogicalOr, inputs,
                                         ::tenzor::OpAttributes{});
        return Variable(result[0], false);
    };
    check_matches_eager("logical_or", fn, make_bool_input());
}

TEST(OpLogicalPredicates, LogicalNotMatchesEager) {
    ensure_core_init();
    auto fn = [](const Variable& x) -> Variable {
        std::vector<Tensor> inputs = {x.tensor()};
        auto result = ::tenzor::dispatch(::tenzor::OpId::LogicalNot, inputs,
                                         ::tenzor::OpAttributes{});
        return Variable(result[0], false);
    };
    check_matches_eager("logical_not", fn, make_bool_input());
}
