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

#include <cstring>
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
    // Eps always travels as a bit-exact i64 bit-pattern of the full double
    // value (not a backend_config string, and not truncated to f32/i32 even
    // when the tensor dtype itself is f32) so a Float64 RMSNorm graph never
    // silently loses precision through this plugin path.
    EXPECT_NE(mlir.find(" : i64"),                std::string::npos) << mlir;
    EXPECT_EQ(mlir.find(" : i32"),                std::string::npos) << mlir;
    EXPECT_NE(mlir.find("tensor<2x16x768xf32>"),  std::string::npos) << mlir;
    EXPECT_NE(mlir.find("tensor<768xf32>"),       std::string::npos) << mlir;
}

TEST(OpRMSNorm, EpsSurvivesFullDoublePrecisionBitExact) {
    ensure_core_init();
    // Regression test for the eps-narrowed-to-float32 bug: pick a value with
    // more significant digits than float32 can represent, and confirm the
    // emitted "arith.constant <bits> : i64" carries the *exact* double bit
    // pattern of `eps`, not the bit pattern of `static_cast<float>(eps)`
    // widened back to i64.
    const double eps = 0.100000001234567;
    tzj::Graph g;
    const std::vector<int64_t> x_shape{1, 1, 4};
    auto x = g.create_value("x", x_shape, ::tenzor::DType::Float64,
                            ::tenzor::Device::cpu());
    g.set_inputs({x});
    auto node = g.create_node(tzj::OpType::RMSNorm);
    node->add_input(x);
    auto out = g.create_value("y", x_shape, ::tenzor::DType::Float64,
                              ::tenzor::Device::cpu());
    node->add_output(out);
    node->set_attr("eps", eps);
    g.add_node(node);
    g.set_outputs({out});

    tzm::GraphToMLIR lowerer;
    const std::string mlir = lowerer.lower(g);

    const auto pos = mlir.find("arith.constant ");
    ASSERT_NE(pos, std::string::npos) << mlir;
    const auto num_start = pos + std::string("arith.constant ").size();
    const auto num_end = mlir.find(' ', num_start);
    ASSERT_NE(num_end, std::string::npos) << mlir;
    const int64_t bits = std::stoll(mlir.substr(num_start, num_end - num_start));

    double decoded;
    std::memcpy(&decoded, &bits, sizeof(decoded));
    EXPECT_EQ(decoded, eps) << "eps must survive the i64 bit-cast round trip "
                                "bit-exact, got " << decoded << " vs " << eps;
    EXPECT_NE(decoded, static_cast<double>(static_cast<float>(eps)))
        << "test's eps value must actually distinguish float32 from float64 "
           "precision, else this test can't detect the regression";
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
