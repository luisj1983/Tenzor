// Phase 13 / Group D.1.1 — FlashAttention lowers to stablehlo.custom_call
// @tenzor_flash_attention.
//
// The Tenzor dialect op FlashAttention is emitted as a single
// `stablehlo.custom_call @tenzor_flash_attention(%q, %k, %v) {backend_config
// = "causal=<b>,scale=<f>"}` line. iree-compile rejects an unregistered
// custom_call target (see test_iree_customcall_smoke.cpp), so this test
// validates the emitted MLIR *text* only — the runtime side is exercised by
// the IR-only smoke test in Group A.9. The expand-to-stablehlo decomposition
// for deploy targets without the Tenzor plugin lives in Group D.1.3.

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

TEST(OpFlashAttention, EmitsCustomCallText) {
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

    EXPECT_NE(mlir.find("stablehlo.custom_call @tenzor_flash_attention"),
              std::string::npos) << mlir;
    EXPECT_NE(mlir.find("causal=true"), std::string::npos) << mlir;
    EXPECT_NE(mlir.find("scale="),      std::string::npos) << mlir;
    // 3 operands carrying the (B,H,S,D)=(2,4,16,64) shape must appear as
    // tensor<2x4x16x64xf32> in the call's operand-type list.
    EXPECT_NE(mlir.find("tensor<2x4x16x64xf32>"), std::string::npos) << mlir;
}

TEST(OpFlashAttention, NonCausalDefaultEmitsCausalFalse) {
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
    // No causal / scale attrs set: defaults must be causal=false, scale=0.
    g.add_node(node);
    g.set_outputs({out});

    tzm::GraphToMLIR lowerer;
    const std::string mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("@tenzor_flash_attention"), std::string::npos) << mlir;
    EXPECT_NE(mlir.find("causal=false"), std::string::npos) << mlir;
}
