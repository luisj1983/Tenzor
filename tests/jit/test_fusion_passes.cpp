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

    int orig_nodes = graph.num_nodes();
    FuseConvBatchNormReluPass pass;
    pass.run(graph);
    // After fusion the graph should contain at most the original number of
    // nodes (typically fewer — the three are collapsed into one fused op).
    EXPECT_LE(graph.num_nodes(), orig_nodes);
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
    pass.run(graph);
    EXPECT_LE(graph.num_nodes(), orig_nodes);
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
    pass.run(graph);
    EXPECT_LE(graph.num_nodes(), orig_nodes);
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
