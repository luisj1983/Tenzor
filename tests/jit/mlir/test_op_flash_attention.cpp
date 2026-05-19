// Phase 13 / Group D.1.1 — FlashAttention lowers to a plugin call.
//
// Per the 2026-05-19 amendment / Path A, the Tenzor dialect op
// FlashAttention is now emitted as a `call @tenzor_plugin.flash_attention(...)`
// against a `func.func private @tenzor_plugin.flash_attention(...) -> ...`
// declaration in the same module. The scalar attrs (causal, scale) travel
// as i32 args (causal as 0/1, scale as the bit pattern of the f32). At
// runtime the in-process IreeInvoker registers a VM native module that
// exports these symbols and routes them to the existing tensor kernels
// (see src/jit/mlir/iree_customcalls.cpp).
//
// This test validates the emitted MLIR *text* only — the runtime side is
// exercised by the end-to-end plugin-path callback tests.

#include "tenzor/jit/graph.hpp"
#include "tenzor/jit/mlir/lowering.hpp"
#include "tenzor/jit/tracer.hpp"
#include "tenzor/tenzor.hpp"

#include <gtest/gtest.h>

#include <string>

namespace tzj = ::tenzor::jit;
namespace tzm = ::tenzor::jit::mlir_jit;

namespace {

auto ensure_core_init() -> void {
    static const bool inited = []() {
        ::tenzor::initialize();
        return true;
    }();
    (void)inited;
}

}  // namespace

TEST(OpFlashAttention, EmitsPluginCallText) {
    ensure_core_init();
    // Q, K, V: (B=2, H=4, S=16, D=64)
    tzj::Graph g;
    const std::vector<int64_t> shape{2, 4, 16, 64};
    auto q = g.create_value("q", shape, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    auto k = g.create_value("k", shape, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    auto v = g.create_value("v", shape, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({q, k, v});

    auto node = g.create_node(tzj::OpType::FlashAttention);
    node->add_input(q);
    node->add_input(k);
    node->add_input(v);
    auto out = g.create_value("o", shape, ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    node->add_output(out);
    node->set_bool_attr("causal", true);
    node->set_attr("scale", 0.125f);
    g.add_node(node);
    g.set_outputs({out});

    tzm::GraphToMLIR lowerer;
    const std::string mlir = lowerer.lower(g);

    // External declaration appears alongside @main.
    EXPECT_NE(mlir.find("func.func private @tenzor_plugin.flash_attention"),
              std::string::npos) << mlir;
    // The call site invokes the plugin function (not a stablehlo.custom_call).
    EXPECT_NE(mlir.find("call @tenzor_plugin.flash_attention"),
              std::string::npos) << mlir;
    // The 3 tensor operands carrying (B,H,S,D)=(2,4,16,64) must appear in
    // the call's operand-type list.
    EXPECT_NE(mlir.find("tensor<2x4x16x64xf32>"), std::string::npos) << mlir;
    // i32 scalars for (scale_bits, causal) must appear in the type list.
    EXPECT_NE(mlir.find("i32, i32"), std::string::npos) << mlir;
}

TEST(OpFlashAttention, NonCausalDefaultEmitsCausalZero) {
    ensure_core_init();
    tzj::Graph g;
    const std::vector<int64_t> shape{1, 2, 8, 32};
    auto q = g.create_value("q", shape, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    auto k = g.create_value("k", shape, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    auto v = g.create_value("v", shape, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({q, k, v});
    auto node = g.create_node(tzj::OpType::FlashAttention);
    node->add_input(q);
    node->add_input(k);
    node->add_input(v);
    auto out = g.create_value("o", shape, ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    node->add_output(out);
    // No causal / scale attrs set: defaults must be causal=false (0),
    // scale defaults to 1/sqrt(D).
    g.add_node(node);
    g.set_outputs({out});

    tzm::GraphToMLIR lowerer;
    const std::string mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("call @tenzor_plugin.flash_attention"),
              std::string::npos) << mlir;
    // The causal=false scalar argument is the i32 constant `0` — two arith.
    // constants live in the body (scale_bits + causal); the second one
    // (causal) ends in ` 0 : i32`.
    EXPECT_NE(mlir.find("arith.constant 0 : i32"), std::string::npos) << mlir;
}
