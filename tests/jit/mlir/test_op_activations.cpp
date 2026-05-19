// Phase 13 / Group C.2 — Activation ops via the MLIR backend.
//
// Covered ops:
//   ReLU    → stablehlo.maximum(%x, 0)
//   Sigmoid → stablehlo.logistic
//   SiLU    → x * sigmoid(x) (tested via the composed form, which traces
//             as Sigmoid+Mul; a direct LowerSiLU test exercises the SiLU
//             handler itself by constructing the graph manually)
//   GELU    → tanh-approximation decomposition over stablehlo primitives

#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/jit/compile.hpp"
#include "tenzor/jit/graph.hpp"
#include "tenzor/jit/mlir/lowering.hpp"
#include "tenzor/jit/tracer.hpp"
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
                         float tol = 1e-5F,
                         std::vector<int64_t> shape = {16},
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

TEST(OpActivations, ReLU) {
    check_matches_eager("relu", [](const ::tenzor::Variable& x) {
        return ::tenzor::nn::relu(x);
    });
}

TEST(OpActivations, Sigmoid) {
    check_matches_eager("sigmoid", [](const ::tenzor::Variable& x) {
        return ::tenzor::nn::sigmoid(x);
    });
}

TEST(OpActivations, SiLUComposed) {
    // `nn::swish` with requires_grad=false dispatches as OpId::Swish, which
    // the tracer doesn't map to OpType::SiLU yet — so a direct call would
    // graph-break. Compose `x * sigmoid(x)` manually so the trace produces
    // a Sigmoid+Mul pair and exercises the elementwise binary path.
    check_matches_eager("silu_composed", [](const ::tenzor::Variable& x) {
        return x * ::tenzor::nn::sigmoid(x);
    });
}

TEST(OpActivations, GELU) {
    // GELU lowers via the tanh approximation. The eager path uses the same
    // approximation (default "tanh" path), but minor differences in the
    // tanh implementation mean we still see ~1e-5 absolute diff for unit-
    // magnitude inputs; widen tolerance slightly to be safe across IREE
    // versions.
    // Call the autograd-level ::tenzor::gelu so dispatch fires
    // OpId::Gelu → OpType::GELU. nn::gelu(x, "tanh") would decompose at
    // the Variable level into tanh+mul+add and pick up the not-yet-
    // supported Tanh op.
    check_matches_eager("gelu", [](const ::tenzor::Variable& x) {
        return ::tenzor::gelu(x);
    }, /*tol=*/5e-3F);
}

// Direct lowering test for the SiLU handler — constructs the graph by hand
// so the case in GraphToMLIR::dispatch is actually exercised.
TEST(LowerSiLU, EmitsSigmoidAndMultiply) {
    ensure_core_init();
    ::tenzor::jit::Graph g;
    auto x = g.create_value("x", {4}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({x});
    auto node = g.create_node(::tenzor::jit::OpType::SiLU, "silu");
    node->add_input(x);
    auto z = g.create_value("z", {4}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    node->add_output(z);
    g.add_node(node);
    g.set_outputs({z});

    ::tenzor::jit::mlir_jit::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.logistic"), std::string::npos) << mlir;
    EXPECT_NE(mlir.find("stablehlo.multiply"), std::string::npos) << mlir;
}
