// Phase 13 / Group D.3.1 — RoPE (Rotary Position Embedding) lowers to
// stablehlo.custom_call @tenzor_rope_apply.
//
// Three operands: input tensor x and two precomputed sinusoid tables
// (cos, sin). The integer attr `offset` carries the starting sequence
// position so that KV-cache continuations resume at the right phase.

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

TEST(OpRoPE, EmitsCustomCallText) {
    ensure_core_init();
    // x: (B=2, H=4, S=16, D=64), tables: (S=16, D=64)
    tzj::Graph g;
    const std::vector<int64_t> x_shape  {2, 4, 16, 64};
    const std::vector<int64_t> tab_shape{16, 64};
    auto x   = g.create_value("x",   x_shape,   ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    auto cos = g.create_value("cos", tab_shape, ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    auto sin = g.create_value("sin", tab_shape, ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    g.set_inputs({x, cos, sin});

    auto node = g.create_node(tzj::OpType::RoPE);
    node->add_input(x);
    node->add_input(cos);
    node->add_input(sin);
    auto out = g.create_value("y", x_shape, ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    node->add_output(out);
    node->set_int_attr("offset", 64);
    g.add_node(node);
    g.set_outputs({out});

    tzm::GraphToMLIR lowerer;
    const std::string mlir = lowerer.lower(g);

    EXPECT_NE(mlir.find("stablehlo.custom_call @tenzor_rope_apply"),
              std::string::npos) << mlir;
    EXPECT_NE(mlir.find("offset=64"),               std::string::npos) << mlir;
    EXPECT_NE(mlir.find("tensor<2x4x16x64xf32>"),   std::string::npos) << mlir;
    EXPECT_NE(mlir.find("tensor<16x64xf32>"),       std::string::npos) << mlir;
}

TEST(OpRoPE, NoOffsetDefaultsToZero) {
    ensure_core_init();
    tzj::Graph g;
    const std::vector<int64_t> x_shape  {1, 2, 8, 32};
    const std::vector<int64_t> tab_shape{8, 32};
    auto x   = g.create_value("x",   x_shape,   ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    auto cos = g.create_value("cos", tab_shape, ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    auto sin = g.create_value("sin", tab_shape, ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    g.set_inputs({x, cos, sin});
    auto node = g.create_node(tzj::OpType::RoPE);
    node->add_input(x);
    node->add_input(cos);
    node->add_input(sin);
    auto out = g.create_value("y", x_shape, ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    node->add_output(out);
    g.add_node(node);
    g.set_outputs({out});
    tzm::GraphToMLIR lowerer;
    const std::string mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("@tenzor_rope_apply"), std::string::npos) << mlir;
    EXPECT_NE(mlir.find("offset=0"),           std::string::npos) << mlir;
}
