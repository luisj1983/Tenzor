// Phase 13 / Group D.2.1 — GQA (Grouped-Query Attention) lowers to a
// plugin call.
//
// Differs from FlashAttention in that K and V may have fewer heads than Q
// (the kernel broadcasts KV heads to Q heads). The lowering itself is
// shape-agnostic: it just records each operand's actual shape so the
// callback (and the expand-to-stablehlo pass in D.2.3) can do the
// broadcast.

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

TEST(OpGQA, EmitsCustomCallText) {
    ensure_core_init();
    // GQA: 8 query heads, 2 KV heads (group size 4).
    // Q: (B=2, H_q=8, S=16, D=64); K/V: (B=2, H_kv=2, S=16, D=64).
    tzj::Graph g;
    const std::vector<int64_t> q_shape {2, 8, 16, 64};
    const std::vector<int64_t> kv_shape{2, 2, 16, 64};
    auto q = g.create_value("q", q_shape,  ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    auto k = g.create_value("k", kv_shape, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    auto v = g.create_value("v", kv_shape, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({q, k, v});

    auto node = g.create_node(tzj::OpType::GQA);
    node->add_input(q);
    node->add_input(k);
    node->add_input(v);
    auto out = g.create_value("o", q_shape, ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    node->add_output(out);
    node->set_bool_attr("causal", true);
    node->set_attr("scale", 0.125f);
    g.add_node(node);
    g.set_outputs({out});

    tzm::GraphToMLIR lowerer;
    const std::string mlir = lowerer.lower(g);

    EXPECT_NE(mlir.find("func.func private @tenzor_plugin.gqa"),
              std::string::npos) << mlir;
    EXPECT_NE(mlir.find("call @tenzor_plugin.gqa"),
              std::string::npos) << mlir;
    // Distinct Q vs KV head dims preserved in the call's operand types.
    EXPECT_NE(mlir.find("tensor<2x8x16x64xf32>"), std::string::npos) << mlir;
    EXPECT_NE(mlir.find("tensor<2x2x16x64xf32>"), std::string::npos) << mlir;
}

TEST(OpGQA, MQAOneKVHeadEmitsValid) {
    ensure_core_init();
    // Multi-query attention is the GQA edge case with H_kv=1.
    tzj::Graph g;
    const std::vector<int64_t> q_shape {1, 8, 32, 64};
    const std::vector<int64_t> kv_shape{1, 1, 32, 64};
    auto q = g.create_value("q", q_shape,  ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    auto k = g.create_value("k", kv_shape, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    auto v = g.create_value("v", kv_shape, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({q, k, v});
    auto node = g.create_node(tzj::OpType::GQA);
    node->add_input(q);
    node->add_input(k);
    node->add_input(v);
    auto out = g.create_value("o", q_shape, ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    node->add_output(out);
    g.add_node(node);
    g.set_outputs({out});

    tzm::GraphToMLIR lowerer;
    const std::string mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("call @tenzor_plugin.gqa"), std::string::npos) << mlir;
    EXPECT_NE(mlir.find("tensor<1x1x32x64xf32>"),   std::string::npos) << mlir;
}
