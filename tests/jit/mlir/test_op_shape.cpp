// Phase 13 / Group C.5 — Shape ops via the MLIR backend.
//
// Covered ops:
//   Reshape   → stablehlo.reshape
//   Permute   → stablehlo.transpose with perm
//   Transpose → stablehlo.transpose (swap two dims)
//   Slice     → stablehlo.slice
//   Cat       → stablehlo.concatenate
//   Stack     → reshape + concatenate
//   Squeeze   → stablehlo.reshape
//   Unsqueeze → stablehlo.reshape
//   Flatten   → stablehlo.reshape
//   Broadcast → stablehlo.broadcast_in_dim
//
// Most of these shape ops don't dispatch through OpId on CPU (Tensor::
// permute / view / squeeze / unsqueeze rewrite strides directly), so the
// tracer can't observe them. We verify the LOWERING handlers by building
// graphs manually, then asserting the emitted MLIR is well-formed and
// parses through iree-compile.

#include "tenzor/jit/graph.hpp"
#include "tenzor/jit/mlir/iree_compile.hpp"
#include "tenzor/jit/mlir/lowering.hpp"
#include "tenzor/jit/tracer.hpp"
#include "tenzor/tenzor.hpp"

#include <gtest/gtest.h>

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

/// Build a single-node graph with the given OpType, attrs, and shapes.
auto build_unary_shape_graph(tzj::OpType op,
                             const std::vector<int64_t>& in_shape,
                             const std::vector<int64_t>& out_shape,
                             const std::vector<std::pair<std::string, int64_t>>& int_attrs = {},
                             const std::vector<std::pair<std::string, std::vector<int64_t>>>& vec_attrs = {})
    -> tzj::Graph {
    tzj::Graph g;
    auto x = g.create_value("x", in_shape, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({x});
    auto node = g.create_node(op);
    node->add_input(x);
    auto z = g.create_value("z", out_shape, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    node->add_output(z);
    for (const auto& [k, v] : int_attrs) node->set_int_attr(k, v);
    for (const auto& [k, v] : vec_attrs) node->set_vec_attr(k, v);
    g.add_node(node);
    g.set_outputs({z});
    return g;
}

/// Build a binary (cat-style) graph with N inputs of the same shape and
/// an output along `dim`.
auto build_cat_graph(tzj::OpType op, int n_inputs,
                     const std::vector<int64_t>& per_input_shape,
                     const std::vector<int64_t>& out_shape, int64_t dim)
    -> tzj::Graph {
    tzj::Graph g;
    std::vector<std::shared_ptr<tzj::Value>> inputs;
    for (int i = 0; i < n_inputs; ++i) {
        auto v = g.create_value("x" + std::to_string(i), per_input_shape,
                                ::tenzor::DType::Float32,
                                ::tenzor::Device::cpu());
        inputs.push_back(v);
    }
    g.set_inputs(inputs);
    auto node = g.create_node(op);
    for (auto& v : inputs) node->add_input(v);
    auto z = g.create_value("z", out_shape, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    node->add_output(z);
    node->set_int_attr("dim", dim);
    g.add_node(node);
    g.set_outputs({z});
    return g;
}

/// Round-trip the emitted MLIR through iree-compile to assert syntactic
/// validity beyond the smoke string-match test.
void assert_iree_compile_accepts(const std::string& mlir_text) {
    auto tmp = std::filesystem::temp_directory_path() /
               ("tenzor_op_shape_" + std::to_string(::getpid()) + "_" +
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

TEST(OpShape, ReshapeEmitsAndParses) {
    ensure_core_init();
    auto g = build_unary_shape_graph(tzj::OpType::Reshape, {2, 6}, {3, 4});
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.reshape"), std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir);
}

TEST(OpShape, PermuteEmitsAndParses) {
    ensure_core_init();
    // Permute (3,4,5) → (5,3,4) with explicit dims [2, 0, 1].
    auto g = build_unary_shape_graph(tzj::OpType::Permute, {3, 4, 5},
                                     {5, 3, 4}, {},
                                     {{"dims", {2, 0, 1}}});
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.transpose"), std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir);
}

TEST(OpShape, TransposeEmitsAndParses) {
    ensure_core_init();
    auto g = build_unary_shape_graph(tzj::OpType::Transpose, {3, 4}, {4, 3});
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.transpose"), std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir);
}

TEST(OpShape, SliceEmitsAndParses) {
    ensure_core_init();
    // Slice (8,) → (3,) on dim 0, start=2, end=5.
    auto g = build_unary_shape_graph(tzj::OpType::Slice, {8}, {3},
                                     {{"dim", 0}, {"start", 2}, {"end", 5},
                                      {"step", 1}});
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.slice"), std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir);
}

TEST(OpShape, CatEmitsAndParses) {
    ensure_core_init();
    auto g = build_cat_graph(tzj::OpType::Cat, /*n_inputs=*/3,
                             /*per_input=*/{4, 2}, /*out=*/{4, 6}, /*dim=*/1);
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.concatenate"), std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir);
}

TEST(OpShape, StackEmitsAndParses) {
    ensure_core_init();
    auto g = build_cat_graph(tzj::OpType::Stack, /*n_inputs=*/3,
                             /*per_input=*/{4, 5}, /*out=*/{3, 4, 5},
                             /*dim=*/0);
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.reshape"), std::string::npos) << mlir;
    EXPECT_NE(mlir.find("stablehlo.concatenate"), std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir);
}

TEST(OpShape, SqueezeEmitsAndParses) {
    ensure_core_init();
    auto g = build_unary_shape_graph(tzj::OpType::Squeeze, {1, 4, 1, 5},
                                     {4, 5});
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.reshape"), std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir);
}

TEST(OpShape, UnsqueezeEmitsAndParses) {
    ensure_core_init();
    auto g = build_unary_shape_graph(tzj::OpType::Unsqueeze, {4, 5},
                                     {4, 1, 5});
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.reshape"), std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir);
}

TEST(OpShape, FlattenEmitsAndParses) {
    ensure_core_init();
    auto g = build_unary_shape_graph(tzj::OpType::Flatten, {2, 3, 4}, {2, 12});
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.reshape"), std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir);
}

TEST(OpShape, BroadcastEmitsAndParses) {
    ensure_core_init();
    // Broadcast (4,) → (3, 4) right-aligned.
    auto g = build_unary_shape_graph(tzj::OpType::Broadcast, {4}, {3, 4});
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.broadcast_in_dim"), std::string::npos)
        << mlir;
    assert_iree_compile_accepts(mlir);
}
