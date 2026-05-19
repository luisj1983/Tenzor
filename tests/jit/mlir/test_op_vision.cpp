// Phase 13 / Group C.8 — Vision ops via the MLIR backend.
//
// Covered ops:
//   Conv2d              → stablehlo.convolution (NCHW, OIHW)
//   MaxPool2d           → stablehlo.reduce_window {maximum}
//   AvgPool2d           → stablehlo.reduce_window {add} / window_area
//   AdaptiveAvgPool2d   → reduce_window with computed kernel/stride
//   Dropout             → identity (inference)
//   ResidualAdd         → stablehlo.add (covered by elementwise tests)
//   Padding             → stablehlo.pad
//   Interpolate         → reshape + broadcast_in_dim + reshape (nearest,
//                         integer upsample only in MVP-1)

#include "tenzor/jit/graph.hpp"
#include "tenzor/jit/mlir/lowering.hpp"
#include "tenzor/jit/tracer.hpp"
#include "tenzor/tenzor.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
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

void assert_iree_compile_accepts(const std::string& mlir_text) {
    auto tmp = std::filesystem::temp_directory_path() /
               ("tenzor_op_vision_" + std::to_string(::getpid()) + "_" +
                std::to_string(reinterpret_cast<std::uintptr_t>(&mlir_text)));
    std::filesystem::create_directories(tmp);
    auto in_path  = tmp / "mod.mlir";
    auto out_path = tmp / "mod.vmfb";
    { std::ofstream of(in_path); of << mlir_text; }
    std::string iree_compile = "iree-compile";
    if (const char* env = std::getenv("TENZOR_IREE_COMPILE"); env && *env) {
        iree_compile = env;
    }
    std::string cmd = iree_compile +
        " --iree-hal-target-backends=llvm-cpu \"" + in_path.string() +
        "\" -o \"" + out_path.string() + "\" 2>&1";
    const int rc = std::system(cmd.c_str());
    EXPECT_EQ(rc, 0) << "iree-compile rejected emitted MLIR:\n" << mlir_text;
    std::filesystem::remove_all(tmp);
}

}  // namespace

TEST(OpVision, Conv2dEmitsAndParses) {
    ensure_core_init();
    // x: (1, 3, 8, 8); w: (4, 3, 3, 3); out: (1, 4, 6, 6) with stride=1,
    // no padding.
    tzj::Graph g;
    auto x = g.create_value("x", {1, 3, 8, 8}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    auto w = g.create_value("w", {4, 3, 3, 3}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({x, w});
    auto node = g.create_node(tzj::OpType::Conv2d);
    node->add_input(x);
    node->add_input(w);
    auto z = g.create_value("z", {1, 4, 6, 6}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    node->add_output(z);
    node->set_int_attr("stride_h", 1);
    node->set_int_attr("stride_w", 1);
    node->set_int_attr("padding_h", 0);
    node->set_int_attr("padding_w", 0);
    node->set_int_attr("dilation_h", 1);
    node->set_int_attr("dilation_w", 1);
    node->set_int_attr("groups", 1);
    g.add_node(node);
    g.set_outputs({z});
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.convolution"), std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir);
}

TEST(OpVision, MaxPool2dEmitsAndParses) {
    ensure_core_init();
    // x: (1, 3, 8, 8); k=2, s=2 → (1, 3, 4, 4)
    tzj::Graph g;
    auto x = g.create_value("x", {1, 3, 8, 8}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({x});
    auto node = g.create_node(tzj::OpType::MaxPool2d);
    node->add_input(x);
    auto z = g.create_value("z", {1, 3, 4, 4}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    node->add_output(z);
    node->set_int_attr("kernel_h", 2);
    node->set_int_attr("kernel_w", 2);
    node->set_int_attr("stride_h", 2);
    node->set_int_attr("stride_w", 2);
    g.add_node(node);
    g.set_outputs({z});
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.reduce_window"), std::string::npos) << mlir;
    EXPECT_NE(mlir.find("stablehlo.maximum"), std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir);
}

TEST(OpVision, AvgPool2dEmitsAndParses) {
    ensure_core_init();
    tzj::Graph g;
    auto x = g.create_value("x", {1, 3, 8, 8}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({x});
    auto node = g.create_node(tzj::OpType::AvgPool2d);
    node->add_input(x);
    auto z = g.create_value("z", {1, 3, 4, 4}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    node->add_output(z);
    node->set_int_attr("kernel_h", 2);
    node->set_int_attr("kernel_w", 2);
    node->set_int_attr("stride_h", 2);
    node->set_int_attr("stride_w", 2);
    g.add_node(node);
    g.set_outputs({z});
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.reduce_window"), std::string::npos) << mlir;
    EXPECT_NE(mlir.find("stablehlo.divide"), std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir);
}

TEST(OpVision, AdaptiveAvgPool2dEmitsAndParses) {
    ensure_core_init();
    // (1, 3, 6, 6) → (1, 3, 2, 2)
    tzj::Graph g;
    auto x = g.create_value("x", {1, 3, 6, 6}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({x});
    auto node = g.create_node(tzj::OpType::AdaptiveAvgPool2d);
    node->add_input(x);
    auto z = g.create_value("z", {1, 3, 2, 2}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    node->add_output(z);
    g.add_node(node);
    g.set_outputs({z});
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.reduce_window"), std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir);
}

TEST(OpVision, DropoutIsIdentity) {
    ensure_core_init();
    tzj::Graph g;
    auto x = g.create_value("x", {4}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({x});
    auto node = g.create_node(tzj::OpType::Dropout);
    node->add_input(x);
    auto z = g.create_value("z", {4}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    node->add_output(z);
    g.add_node(node);
    g.set_outputs({z});
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    // Inference dropout is an identity — no op inserted, just return %arg0.
    EXPECT_NE(mlir.find("return %arg0"), std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir);
}

TEST(OpVision, PaddingEmitsAndParses) {
    ensure_core_init();
    // (4,) → (8,) with pad_left=2, pad_right=2.
    tzj::Graph g;
    auto x = g.create_value("x", {4}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({x});
    auto node = g.create_node(tzj::OpType::Padding);
    node->add_input(x);
    auto z = g.create_value("z", {8}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    node->add_output(z);
    node->set_vec_attr("padding", {2, 2});
    g.add_node(node);
    g.set_outputs({z});
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.pad"), std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir);
}

TEST(OpVision, InterpolateUpsample2x) {
    ensure_core_init();
    // (1, 3, 4, 4) → (1, 3, 8, 8) nearest, factor 2.
    tzj::Graph g;
    auto x = g.create_value("x", {1, 3, 4, 4}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({x});
    auto node = g.create_node(tzj::OpType::Interpolate);
    node->add_input(x);
    auto z = g.create_value("z", {1, 3, 8, 8}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    node->add_output(z);
    g.add_node(node);
    g.set_outputs({z});
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.broadcast_in_dim"), std::string::npos)
        << mlir;
    EXPECT_NE(mlir.find("stablehlo.reshape"), std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir);
}
