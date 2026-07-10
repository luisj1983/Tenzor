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
    // fuse_triple resolves BN/Conv parameters from their real graph INPUTS
    // (kernel-canonical order: Conv2d is (x, weight, [bias]); affine
    // BatchNorm2d is (x, gamma, beta, mean, var)), never from node attrs --
    // see JIT-R001. Build real Constant-backed inputs so the fold has real
    // work to do and the assertions have teeth.
    Graph graph;
    auto input = graph.create_value("input", {1, 3, 8, 8}, DType::Float32, device_);

    // Bare Values registered in graph.constants(), mirroring how the real
    // tracer represents a captured parameter (no producer node -- see
    // Graph::set_constant's doc comment).
    auto make_const = [&](const std::string& name, const std::vector<int64_t>& shape,
                           double fill) {
        auto val = graph.create_value(name, shape, DType::Float32, device_);
        graph.set_constant(name, tenzor::full(shape, fill, DType::Float32, device_));
        return val;
    };

    auto conv_w_v = graph.create_value("conv_w", {16, 3, 3, 3}, DType::Float32, device_);
    graph.set_constant("conv_w", tenzor::randn({16, 3, 3, 3}, DType::Float32, device_));

    auto conv_node = graph.create_node(OpType::Conv2d, "conv");
    conv_node->add_input(input);
    conv_node->add_input(conv_w_v);
    auto conv_v = graph.create_value("conv_out", {1, 16, 8, 8}, DType::Float32, device_);
    conv_v->set_node(conv_node);
    conv_node->add_output(conv_v);
    graph.add_node(conv_node);

    auto gamma = make_const("gamma", {16}, 1.0);
    auto beta  = make_const("beta",  {16}, 0.0);
    auto mean  = make_const("mean",  {16}, 0.0);
    auto var   = make_const("var",   {16}, 1.0);
    auto bn_node = graph.create_node(OpType::BatchNorm2d, "bn");
    bn_node->add_input(conv_v);
    bn_node->add_input(gamma);
    bn_node->add_input(beta);
    bn_node->add_input(mean);
    bn_node->add_input(var);
    bn_node->set_attr("eps", 1e-5F);
    auto bn_v = graph.create_value("bn_out", {1, 16, 8, 8}, DType::Float32, device_);
    bn_v->set_node(bn_node);
    bn_node->add_output(bn_v);
    graph.add_node(bn_node);

    auto relu_v = add_node(graph, bn_v, OpType::ReLU, {1, 16, 8, 8}, "relu");
    graph.set_inputs({input});
    graph.set_outputs({relu_v});

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

    // The fused Conv2d node must carry over BN-folded weights (not the
    // original conv_w constant) and be marked fused_bn/fused_relu so replay
    // (Graph::execute_node) applies the folded weight and the ReLU.
    ASSERT_EQ(graph.outputs().size(), 1u);
    auto fused_node = graph.outputs()[0]->node();
    ASSERT_NE(fused_node, nullptr);
    EXPECT_EQ(fused_node->op_type(), OpType::Conv2d);
    EXPECT_TRUE(fused_node->get_bool_attr("fused_bn"));
    EXPECT_TRUE(fused_node->get_bool_attr("fused_relu"));
    ASSERT_GE(fused_node->inputs().size(), 2u);
    EXPECT_NE(fused_node->inputs()[1]->id(), conv_w_v->id())
        << "fused conv weight input must be the BN-folded constant, not the "
           "original unfused weight";
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
    // Linear -> GELU -> Linear is the canonical FFN block. fuse_triple
    // resolves weight/bias from each Linear's own graph INPUTS (a traced
    // Linear stores weight at inputs()[1] and an optional bias at
    // inputs()[2], never as node attrs -- see JIT-R001), so a Linear with
    // only a bare data input (the old version of this test) has nothing to
    // fold and must NOT fuse. Build real weight/bias Constant inputs so the
    // pass has real work to do and the assertions have teeth.
    Graph graph;
    auto input = graph.create_value("input", {2, 16}, DType::Float32, device_);

    // Bare Values registered in graph.constants(), mirroring how the real
    // tracer represents a captured parameter (no producer node -- see
    // Graph::set_constant's doc comment). Using OpType::Constant NODES here
    // instead would wedge extra nodes between l1/act/l2 in graph.nodes(),
    // breaking FuseFFNPass::run()'s positional Linear/Act/Linear match --
    // exactly the layout a real trace never produces.
    auto make_const = [&](const std::string& name, const std::vector<int64_t>& shape) {
        auto val = graph.create_value(name, shape, DType::Float32, device_);
        graph.set_constant(name, tenzor::randn(shape, DType::Float32, device_));
        return val;
    };

    auto w1 = make_const("w1", {64, 16});
    auto b1 = make_const("b1", {64});
    auto l1_node = graph.create_node(OpType::Linear, "l1");
    l1_node->add_input(input);
    l1_node->add_input(w1);
    l1_node->add_input(b1);
    auto l1_out = graph.create_value("l1_out", {2, 64}, DType::Float32, device_);
    l1_out->set_node(l1_node);
    l1_node->add_output(l1_out);
    graph.add_node(l1_node);

    auto act = add_node(graph, l1_out, OpType::GELU, {2, 64}, "act");

    auto w2 = make_const("w2", {16, 64});
    auto b2 = make_const("b2", {16});
    auto l2_node = graph.create_node(OpType::Linear, "l2");
    l2_node->add_input(act);
    l2_node->add_input(w2);
    l2_node->add_input(b2);
    auto l2_out = graph.create_value("l2_out", {2, 16}, DType::Float32, device_);
    l2_out->set_node(l2_node);
    l2_node->add_output(l2_out);
    graph.add_node(l2_node);

    graph.set_inputs({input});
    graph.set_outputs({l2_out});

    int orig_nodes = graph.num_nodes();
    FuseFFNPass pass;
    bool changed = pass.run(graph);
    EXPECT_TRUE(changed) << "FuseFFN did not fire on Linear->GELU->Linear";
    EXPECT_LT(graph.num_nodes(), orig_nodes)
        << "fusion did not reduce the node count";

    // The fused node must carry over the SAME weight values the original
    // Linears used (JIT-R001 regression: previously these were silently
    // dropped -- get_tensor_attr("weight") always returned an empty Tensor
    // -- and the fused node executed matmul against a 0-element operand).
    ASSERT_EQ(graph.outputs().size(), 1u);
    auto ffn_node = graph.outputs()[0]->node();
    ASSERT_NE(ffn_node, nullptr);
    EXPECT_EQ(ffn_node->op_type(), OpType::FusedFFN);
    ASSERT_GE(ffn_node->inputs().size(), 3u);
    EXPECT_EQ(ffn_node->inputs()[1]->id(), w1->id());
    EXPECT_EQ(ffn_node->inputs()[2]->id(), w2->id());
    EXPECT_TRUE(ffn_node->get_bool_attr("has_bias1"));
    EXPECT_TRUE(ffn_node->get_bool_attr("has_bias2"));
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

// ============================================================================
// Matcher guards: non-last-axis softmax/norm must NOT fuse (the fused kernels
// only support the last axis and would crash / read wrong strided elements with
// no fallback — F034/F035), and a plain std/variance subgraph must NOT be
// mistaken for a full LayerNorm (F009). Positive controls confirm the guards do
// not over-reject legitimate last-axis softmax / full LayerNorm.
// ============================================================================
namespace {
auto has_kind(const std::vector<FusionMatch>& ms, FusionKind k) -> bool {
    for (const auto& m : ms) {
        if (m.kind == k) return true;
    }
    return false;
}

// Numerically-stable softmax over `dim`:
//   Max[dim] -> Sub(x,max) -> Exp -> Sum[dim] -> Div(exp,sum).
auto build_softmax(Graph& g, const Device& dev,
                   const std::vector<int64_t>& shape, int64_t dim) -> void {
    const int64_t nd = dim < 0 ? dim + static_cast<int64_t>(shape.size()) : dim;
    std::vector<int64_t> red = shape;
    red[nd] = 1;
    auto x = g.create_value("x", shape, DType::Float32, dev);
    auto mx = g.create_node(OpType::Max, "max");
    auto mx_o = g.create_value("max_o", red, DType::Float32, dev);
    mx->add_input(x); mx->add_output(mx_o); mx_o->set_node(mx);
    mx->set_int_attr("dim", dim); mx->set_bool_attr("keepdim", true);
    g.add_node(mx);
    auto sub = g.create_node(OpType::Sub, "sub");
    auto sub_o = g.create_value("sub_o", shape, DType::Float32, dev);
    sub->add_input(x); sub->add_input(mx_o); sub->add_output(sub_o);
    sub_o->set_node(sub); g.add_node(sub);
    auto ex = g.create_node(OpType::Exp, "exp");
    auto ex_o = g.create_value("exp_o", shape, DType::Float32, dev);
    ex->add_input(sub_o); ex->add_output(ex_o); ex_o->set_node(ex); g.add_node(ex);
    auto sm = g.create_node(OpType::Sum, "sum");
    auto sm_o = g.create_value("sum_o", red, DType::Float32, dev);
    sm->add_input(ex_o); sm->add_output(sm_o); sm_o->set_node(sm);
    sm->set_int_attr("dim", dim); sm->set_bool_attr("keepdim", true);
    g.add_node(sm);
    auto dv = g.create_node(OpType::Div, "div");
    auto dv_o = g.create_value("div_o", shape, DType::Float32, dev);
    dv->add_input(ex_o); dv->add_input(sm_o); dv->add_output(dv_o);
    dv_o->set_node(dv); g.add_node(dv);
    g.set_inputs({x}); g.set_outputs({dv_o});
}

// LayerNorm-shaped chain over the last axis. If `full_layernorm`, append the
// normalizing Div (x-mean)/std; otherwise stop at Sqrt — a plain std/variance
// subgraph that is NOT a LayerNorm. The eps Constant is emitted first so the
// compute chain stays positionally contiguous for the matcher's scan.
auto build_layernorm_like(Graph& g, const Device& dev, bool full_layernorm)
    -> void {
    const std::vector<int64_t> shape = {2, 3, 4};
    const std::vector<int64_t> red = {2, 3, 1};
    auto x = g.create_value("x", shape, DType::Float32, dev);
    auto epsc = g.create_node(OpType::Constant, "eps");
    auto epsc_o = g.create_value("epsv", {1}, DType::Float32, dev);
    epsc->add_output(epsc_o); epsc_o->set_node(epsc);
    epsc->set_tensor_attr("value", ::tenzor::full({1}, 1e-5F, DType::Float32));
    g.add_node(epsc);
    auto m1 = g.create_node(OpType::Mean, "mean1");
    auto m1_o = g.create_value("m1", red, DType::Float32, dev);
    m1->add_input(x); m1->add_output(m1_o); m1_o->set_node(m1);
    m1->set_int_attr("dim", -1); m1->set_bool_attr("keepdim", true);
    g.add_node(m1);
    auto sub = g.create_node(OpType::Sub, "sub");
    auto sub_o = g.create_value("sub", shape, DType::Float32, dev);
    sub->add_input(x); sub->add_input(m1_o); sub->add_output(sub_o);
    sub_o->set_node(sub); g.add_node(sub);
    auto pw = g.create_node(OpType::Pow, "pow");
    auto pw_o = g.create_value("pow", shape, DType::Float32, dev);
    pw->add_input(sub_o); pw->add_output(pw_o); pw_o->set_node(pw);
    pw->set_attr("exponent", 2.0); g.add_node(pw);
    auto m2 = g.create_node(OpType::Mean, "mean2");
    auto m2_o = g.create_value("m2", red, DType::Float32, dev);
    m2->add_input(pw_o); m2->add_output(m2_o); m2_o->set_node(m2);
    m2->set_int_attr("dim", -1); m2->set_bool_attr("keepdim", true);
    g.add_node(m2);
    auto add = g.create_node(OpType::Add, "addeps");
    auto add_o = g.create_value("addeps", red, DType::Float32, dev);
    add->add_input(m2_o); add->add_input(epsc_o); add->add_output(add_o);
    add_o->set_node(add); g.add_node(add);
    auto sq = g.create_node(OpType::Sqrt, "sqrt");
    auto sq_o = g.create_value("sqrt", red, DType::Float32, dev);
    sq->add_input(add_o); sq->add_output(sq_o); sq_o->set_node(sq); g.add_node(sq);
    if (!full_layernorm) {
        g.set_inputs({x}); g.set_outputs({sq_o});
        return;
    }
    auto dv = g.create_node(OpType::Div, "div");
    auto dv_o = g.create_value("div", shape, DType::Float32, dev);
    dv->add_input(sub_o); dv->add_input(sq_o); dv->add_output(dv_o);
    dv_o->set_node(dv); g.add_node(dv);
    g.set_inputs({x}); g.set_outputs({dv_o});
}
}  // namespace

TEST_F(FusionPassesTest, Softmax_NonLastAxisNotFused) {
    Graph graph;
    build_softmax(graph, device_, {2, 3, 4}, /*dim=*/1);  // middle axis
    PatternMatcher matcher;
    auto matches = matcher.find_all(graph);
    EXPECT_FALSE(has_kind(matches, FusionKind::Softmax))
        << "a non-last-axis softmax must NOT fuse — the fused kernel only "
           "supports the last dim and hard-crashes with no fallback (F034)";
}

TEST_F(FusionPassesTest, Softmax_LastAxisStillFused) {
    Graph graph;
    build_softmax(graph, device_, {2, 3, 4}, /*dim=*/-1);  // last axis
    PatternMatcher matcher;
    auto matches = matcher.find_all(graph);
    EXPECT_TRUE(has_kind(matches, FusionKind::Softmax))
        << "a last-axis softmax must still fuse (guard over-rejected)";
}

TEST_F(FusionPassesTest, LayerNorm_PlainStdNotFused) {
    Graph graph;
    build_layernorm_like(graph, device_, /*full_layernorm=*/false);
    PatternMatcher matcher;
    auto matches = matcher.find_all(graph);
    EXPECT_FALSE(has_kind(matches, FusionKind::LayerNorm))
        << "a plain std/variance subgraph (no normalizing Div) must NOT fuse as "
           "a full LayerNorm (F009)";
}

TEST_F(FusionPassesTest, LayerNorm_FullStillFused) {
    Graph graph;
    build_layernorm_like(graph, device_, /*full_layernorm=*/true);
    PatternMatcher matcher;
    auto matches = matcher.find_all(graph);
    EXPECT_TRUE(has_kind(matches, FusionKind::LayerNorm))
        << "a full LayerNorm (with normalizing Div) must still fuse (guard "
           "over-rejected)";
}
