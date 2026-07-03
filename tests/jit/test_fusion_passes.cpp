/**
 * @file test_fusion_passes.cpp
 * @brief Per-pass JIT fusion tests (Phase 5.5).
 *
 * `test_jit_compiler.cpp` already covers FuseConvReluPass, FuseLinearReluPass,
 * and FuseConvBatchNormPass with the 3-test pattern (NoPattern / WithPattern
 * / PassName). This file extends per-pass coverage to the remaining passes
 * declared in `compiler.hpp` so a regression in any single pass surfaces as
 * a focused failure rather than as a generic compile-output mismatch.
 *
 * Coverage added:
 *   - FuseConvBatchNormReluPass
 *   - FuseLayerNormActivationPass
 *   - FuseAttentionPass
 *   - FuseFFNPass
 *   - ExtendedFusionPass (PassName + Run-on-empty smoke)
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/jit/compiler.hpp>
#include <tenzor/jit/graph.hpp>
#include <tenzor/jit/pattern_matcher.hpp>
#include <tenzor/jit/tracer.hpp>
#include <memory>

using namespace tenzor;
using namespace tenzor::jit;

namespace {
struct InitTenzorOnce : public ::testing::Environment {
    void SetUp() override { tenzor::initialize(); }
};
::testing::Environment* const _g_init =
    ::testing::AddGlobalTestEnvironment(new InitTenzorOnce);
}  // namespace

class FusionPassesTest : public ::testing::Test {
protected:
    Device device_ = Device::cpu();

    auto create_simple_graph(OpType op) -> Graph {
        Graph graph;
        auto input = graph.create_value("input", {2, 3}, DType::Float32, device_);
        auto output = graph.create_value("output", {2, 3}, DType::Float32, device_);
        auto node = graph.create_node(op, "node");
        node->add_input(input);
        node->add_output(output);
        output->set_node(node);
        graph.add_node(node);
        graph.set_inputs({input});
        graph.set_outputs({output});
        return graph;
    }

    // Helper: append a node with a single input/output pair to `prev`.
    auto add_node(Graph& g, std::shared_ptr<Value> prev, OpType op,
                  const std::vector<int64_t>& shape, const std::string& name)
        -> std::shared_ptr<Value> {
        auto out = g.create_value(name + "_out", shape, DType::Float32, device_);
        auto node = g.create_node(op, name);
        node->add_input(prev);
        node->add_output(out);
        out->set_node(node);
        g.add_node(node);
        return out;
    }
};

// ============================================================================
// FuseConvBatchNormReluPass
// ============================================================================

TEST_F(FusionPassesTest, FuseConvBatchNormReLU_PassName) {
    FuseConvBatchNormReluPass pass;
    EXPECT_EQ(pass.name(), "FuseConvBatchNormReLU");
}

TEST_F(FusionPassesTest, FuseConvBatchNormReLU_NoPattern) {
    // Graph with only Add — no triplet to fuse, pass should be a no-op.
    auto graph = create_simple_graph(OpType::Add);
    FuseConvBatchNormReluPass pass;
    bool changed = pass.run(graph);
    EXPECT_FALSE(changed);
}

TEST_F(FusionPassesTest, FuseConvBatchNormReLU_WithPattern) {
    Graph graph;
    auto input = graph.create_value("input", {1, 3, 8, 8}, DType::Float32, device_);
    auto conv_v = add_node(graph, input,    OpType::Conv2d,      {1, 16, 8, 8}, "conv");
    auto bn_v   = add_node(graph, conv_v,   OpType::BatchNorm2d, {1, 16, 8, 8}, "bn");
    auto relu_v = add_node(graph, bn_v,     OpType::ReLU,        {1, 16, 8, 8}, "relu");
    graph.set_inputs({input});
    graph.set_outputs({relu_v});

    // fuse_triple folds the BN running-stats into the conv weights, so it bails
    // (no fusion) unless the conv weight and BN gamma/running_var tensors are
    // present. The value-only nodes above have none — which is exactly why the
    // old EXPECT_LE passed vacuously (the pass never actually fired). Attach
    // real params so the fold can happen and the assertions have teeth.
    conv_v->node()->set_tensor_attr(
        "weight", tenzor::randn({16, 3, 3, 3}, DType::Float32, device_));
    auto bn = bn_v->node();
    bn->set_tensor_attr("weight",       tenzor::ones({16}, DType::Float32, device_));
    bn->set_tensor_attr("bias",         tenzor::zeros({16}, DType::Float32, device_));
    bn->set_tensor_attr("running_mean", tenzor::zeros({16}, DType::Float32, device_));
    bn->set_tensor_attr("running_var",  tenzor::ones({16}, DType::Float32, device_));
    bn->set_attr("eps", 1e-5F);

    int orig_nodes = graph.num_nodes();
    FuseConvBatchNormReluPass pass;
    // The pass MUST fire on the canonical Conv->BN->ReLU triplet and reduce the
    // node count (the three collapse into one fused op). The old EXPECT_LE was
    // vacuous — it also holds when the pass does nothing, so a broken pass that
    // silently stopped matching would still pass the test.
    bool changed = pass.run(graph);
    EXPECT_TRUE(changed) << "FuseConvBatchNormReLU did not fire on Conv->BN->ReLU";
    EXPECT_LT(graph.num_nodes(), orig_nodes)
        << "fusion did not reduce the node count (expected the triplet to "
           "collapse into a single fused op)";
}

// ============================================================================
// FuseLayerNormActivationPass
// ============================================================================

TEST_F(FusionPassesTest, FuseLayerNormActivation_PassName) {
    FuseLayerNormActivationPass pass;
    EXPECT_EQ(pass.name(), "FuseLayerNormActivation");
}

TEST_F(FusionPassesTest, FuseLayerNormActivation_NoPattern) {
    auto graph = create_simple_graph(OpType::Add);
    FuseLayerNormActivationPass pass;
    bool changed = pass.run(graph);
    EXPECT_FALSE(changed);
}

TEST_F(FusionPassesTest, FuseLayerNormActivation_WithPattern) {
    Graph graph;
    auto input = graph.create_value("input", {2, 16}, DType::Float32, device_);
    auto ln_v  = add_node(graph, input, OpType::LayerNorm, {2, 16}, "ln");
    auto gelu_v = add_node(graph, ln_v, OpType::GELU,      {2, 16}, "gelu");
    graph.set_inputs({input});
    graph.set_outputs({gelu_v});

    int orig_nodes = graph.num_nodes();
    FuseLayerNormActivationPass pass;
    bool changed = pass.run(graph);
    EXPECT_TRUE(changed) << "FuseLayerNormActivation did not fire on LayerNorm->GELU";
    EXPECT_LT(graph.num_nodes(), orig_nodes)
        << "fusion did not reduce the node count";
}

// ============================================================================
// FuseAttentionPass
// ============================================================================

TEST_F(FusionPassesTest, FuseAttention_PassName) {
    FuseAttentionPass pass;
    EXPECT_EQ(pass.name(), "FuseAttention");
}

TEST_F(FusionPassesTest, FuseAttention_NoPattern) {
    auto graph = create_simple_graph(OpType::Add);
    FuseAttentionPass pass;
    bool changed = pass.run(graph);
    EXPECT_FALSE(changed);
}

// (No WithPattern test: building an attention pattern is multi-input and
// the exact recognized graph shape is implementation-detail.  PassName
// + NoPattern are the regression-prevention surface here; broader
// coverage lives in test_jit_compile.cpp's end-to-end compile path.)

// ============================================================================
// FuseFFNPass
// ============================================================================

TEST_F(FusionPassesTest, FuseFFN_PassName) {
    FuseFFNPass pass;
    EXPECT_EQ(pass.name(), "FuseFFN");
}

TEST_F(FusionPassesTest, FuseFFN_NoPattern) {
    auto graph = create_simple_graph(OpType::Add);
    FuseFFNPass pass;
    bool changed = pass.run(graph);
    EXPECT_FALSE(changed);
}

TEST_F(FusionPassesTest, FuseFFN_WithPattern) {
    // Linear -> GELU -> Linear is the canonical FFN block.
    Graph graph;
    auto input = graph.create_value("input", {2, 16}, DType::Float32, device_);
    auto l1 = add_node(graph, input, OpType::Linear, {2, 64}, "l1");
    auto act = add_node(graph, l1,    OpType::GELU,   {2, 64}, "act");
    auto l2 = add_node(graph, act,   OpType::Linear, {2, 16}, "l2");
    graph.set_inputs({input});
    graph.set_outputs({l2});

    int orig_nodes = graph.num_nodes();
    FuseFFNPass pass;
    bool changed = pass.run(graph);
    EXPECT_TRUE(changed) << "FuseFFN did not fire on Linear->GELU->Linear";
    EXPECT_LT(graph.num_nodes(), orig_nodes)
        << "fusion did not reduce the node count";
}

// ============================================================================
// ExtendedFusionPass
// ============================================================================

TEST_F(FusionPassesTest, ExtendedFusion_PassName) {
    ExtendedFusionPass pass;
    EXPECT_EQ(pass.name(), "ExtendedFusion");
}

TEST_F(FusionPassesTest, ExtendedFusion_RunsOnEmptyGraph) {
    Graph graph;
    ExtendedFusionPass pass;
    // An empty graph is a no-op for any optimization pass.
    bool changed = pass.run(graph);
    EXPECT_FALSE(changed);
    EXPECT_EQ(graph.num_nodes(), 0);
}

TEST_F(FusionPassesTest, ExtendedFusion_RunsOnSimpleGraph) {
    auto graph = create_simple_graph(OpType::Add);
    int orig_nodes = graph.num_nodes();
    ExtendedFusionPass pass;
    pass.run(graph);
    // Pass may or may not modify; node count never grows.
    EXPECT_LE(graph.num_nodes(), orig_nodes);
}

// ============================================================================
// GemmEpilogue matcher — residual-vs-bias discrimination (H1 regression)
//
// The GemmEpilogue matcher may absorb an Add after a bias-less MatMul/Linear as
// a per-column bias. It MUST only do so when the Add's other operand is a real
// [cols] per-column bias — a full [rows,cols] residual would be broadcast from
// its first row by the native kernel (bias[idx%cols]). We build the graph
// directly (bypassing the pass ordering that hides this via normal tracing) so
// the matcher itself is exercised: the [rows,cols] case must NOT fuse the Add
// (teeth: it would without the rank-1 guard), the [cols] case still must.
// ============================================================================

namespace {
// Build: (bias-less) Linear(x[4,16], w[8,16]) -> lin_out[4,8]; Add(lin_out,
// other[other_shape]) -> add_out[4,8]. Returns the graph and the Add node.
auto build_linear_add(Graph& g, const Device& dev,
                      const std::vector<int64_t>& other_shape)
    -> std::shared_ptr<Node> {
    auto x   = g.create_value("x",   {4, 16}, DType::Float32, dev);
    auto w   = g.create_value("w",   {8, 16}, DType::Float32, dev);
    auto lin = g.create_node(OpType::Linear, "linear");
    auto lin_out = g.create_value("lin_out", {4, 8}, DType::Float32, dev);
    lin->add_input(x);
    lin->add_input(w);                 // bias-less: exactly 2 inputs
    lin->add_output(lin_out);
    lin_out->set_node(lin);
    g.add_node(lin);

    auto other = g.create_value("other", other_shape, DType::Float32, dev);
    auto add = g.create_node(OpType::Add, "add");
    auto add_out = g.create_value("add_out", {4, 8}, DType::Float32, dev);
    add->add_input(lin_out);
    add->add_input(other);
    add->add_output(add_out);
    add_out->set_node(add);
    g.add_node(add);

    g.set_inputs({x});
    g.set_outputs({add_out});
    return add;
}

auto add_is_in_gemm_epilogue(const std::vector<FusionMatch>& matches,
                             const Node* add) -> bool {
    for (const auto& m : matches) {
        if (m.kind != FusionKind::GemmEpilogue) continue;
        for (const auto& n : m.nodes) {
            if (n.get() == add) return true;
        }
    }
    return false;
}
}  // namespace

TEST_F(FusionPassesTest, GemmEpilogue_ResidualNotFusedAsBias) {
    Graph graph;
    auto add = build_linear_add(graph, device_, {4, 8});  // [rows, cols] residual
    PatternMatcher matcher;
    auto matches = matcher.find_all(graph);
    EXPECT_FALSE(add_is_in_gemm_epilogue(matches, add.get()))
        << "a [rows,cols] residual Add was wrongly absorbed as a per-column "
           "GemmEpilogue bias";
}

TEST_F(FusionPassesTest, GemmEpilogue_PerColumnBiasStillFuses) {
    Graph graph;
    auto add = build_linear_add(graph, device_, {8});  // [cols] per-column bias
    PatternMatcher matcher;
    auto matches = matcher.find_all(graph);
    EXPECT_TRUE(add_is_in_gemm_epilogue(matches, add.get()))
        << "a genuine [cols] per-column bias must still fuse into GemmEpilogue "
           "(guard over-rejected a real bias)";
}
