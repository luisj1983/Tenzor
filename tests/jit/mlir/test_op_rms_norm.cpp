// Phase 13 / Group D.4.1 — RMSNorm lowers to stablehlo.custom_call
// @tenzor_plugin.rms_norm.
//
// Two operands (x, weight) and a single float attribute `eps`. The
// backend_config string carries `eps=<f>` in scientific notation so the
// callback / expand pass can round-trip it without precision loss.

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

TEST(OpRMSNorm, EmitsCustomCallText) {
    ensure_core_init();
    // x: (B=2, S=16, D=768), weight: (D=768,)
    tzj::Graph g;
    const std::vector<int64_t> x_shape{2, 16, 768};
    const std::vector<int64_t> w_shape{768};
    auto x = g.create_value("x", x_shape, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    auto w = g.create_value("w", w_shape, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({x, w});

    auto node = g.create_node(tzj::OpType::RMSNorm);
    node->add_input(x);
    node->add_input(w);
    auto out = g.create_value("y", x_shape, ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    node->add_output(out);
    node->set_attr("eps", 1e-6f);
    g.add_node(node);
    g.set_outputs({out});

    tzm::GraphToMLIR lowerer;
    const std::string mlir = lowerer.lower(g);

    EXPECT_NE(mlir.find("call @tenzor_plugin.rms_norm"),
              std::string::npos) << mlir;
    // Eps travels as an i32 bit-pattern of f32, not a backend_config string.
    EXPECT_NE(mlir.find("i32"),                   std::string::npos) << mlir;
    EXPECT_NE(mlir.find("tensor<2x16x768xf32>"),  std::string::npos) << mlir;
    EXPECT_NE(mlir.find("tensor<768xf32>"),       std::string::npos) << mlir;
}

TEST(OpRMSNorm, NoWeightInputAllowed) {
    ensure_core_init();
    // RMSNorm without a weight tensor (the eager layer can be constructed
    // with affine=false equivalent).
    tzj::Graph g;
    const std::vector<int64_t> x_shape{1, 8, 64};
    auto x = g.create_value("x", x_shape, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({x});
    auto node = g.create_node(tzj::OpType::RMSNorm);
    node->add_input(x);
    auto out = g.create_value("y", x_shape, ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    node->add_output(out);
    node->set_attr("eps", 1e-5f);
    g.add_node(node);
    g.set_outputs({out});
    tzm::GraphToMLIR lowerer;
    const std::string mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("@tenzor_plugin.rms_norm"),       std::string::npos) << mlir;
    EXPECT_NE(mlir.find("tensor<1x8x64xf32>"),     std::string::npos) << mlir;
}
