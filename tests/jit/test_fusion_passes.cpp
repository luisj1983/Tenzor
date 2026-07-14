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

namespace {
// Builds MatMul(Q,K) -> Mul(scale) -> [optional Add(mask)] -> Softmax ->
// MatMul(_,V), matching FuseAttentionPass::run's exact pattern, and returns
// the final output Value. Used by the JIT-R041 regression tests below.
auto build_attention_pattern(Graph& graph, Device device, bool with_mask)
    -> std::shared_ptr<Value> {
    auto q = graph.create_value("q", {1, 4, 8}, DType::Float32, device);
    auto k = graph.create_value("k", {1, 4, 8}, DType::Float32, device);
    auto v = graph.create_value("v", {1, 4, 8}, DType::Float32, device);

    auto qk_node = graph.create_node(OpType::MatMul, "qk_matmul");
    qk_node->add_input(q);
    qk_node->add_input(k);
    auto qk_out = graph.create_value("qk_out", {1, 4, 4}, DType::Float32, device);
    qk_out->set_node(qk_node);
    qk_node->add_output(qk_out);
    graph.add_node(qk_node);

    auto scale_const_node = graph.create_node(OpType::Constant, "scale_const");
    scale_const_node->set_tensor_attr(
        "value", tenzor::full({1}, 0.125, DType::Float32, device));
    auto scale_const_val = graph.create_value("scale_const_val", {1}, DType::Float32, device);
    scale_const_val->set_node(scale_const_node);
    scale_const_node->add_output(scale_const_val);
    graph.add_node(scale_const_node);

    auto scale_node = graph.create_node(OpType::Mul, "scale");
    scale_node->add_input(qk_out);
    scale_node->add_input(scale_const_val);
    auto scale_out = graph.create_value("scale_out", {1, 4, 4}, DType::Float32, device);
    scale_out->set_node(scale_node);
    scale_node->add_output(scale_out);
    graph.add_node(scale_node);

    std::shared_ptr<Value> softmax_input = scale_out;
    if (with_mask) {
        auto mask = graph.create_value("mask", {1, 4, 4}, DType::Float32, device);
        auto mask_add_node = graph.create_node(OpType::Add, "mask_add");
        mask_add_node->add_input(scale_out);
        mask_add_node->add_input(mask);
        auto mask_out = graph.create_value("mask_out", {1, 4, 4}, DType::Float32, device);
        mask_out->set_node(mask_add_node);
        mask_add_node->add_output(mask_out);
        graph.add_node(mask_add_node);
        softmax_input = mask_out;
    }

    auto softmax_node = graph.create_node(OpType::Softmax, "softmax");
    softmax_node->add_input(softmax_input);
    auto softmax_out = graph.create_value("softmax_out", {1, 4, 4}, DType::Float32, device);
    softmax_out->set_node(softmax_node);
    softmax_node->add_output(softmax_out);
    graph.add_node(softmax_node);

    auto av_node = graph.create_node(OpType::MatMul, "av_matmul");
    av_node->add_input(softmax_out);
    av_node->add_input(v);
    auto av_out = graph.create_value("av_out", {1, 4, 8}, DType::Float32, device);
    av_out->set_node(av_node);
    av_node->add_output(av_out);
    graph.add_node(av_node);

    graph.set_inputs({q, k, v});
    graph.set_outputs({av_out});
    return av_out;
}
}  // namespace

TEST_F(FusionPassesTest, FuseAttention_WithoutMask_Fuses) {
    // Baseline: an unmasked attention pattern still fuses into a single
    // 3-input FlashAttention node (sanity check that JIT-R041's added
    // early-return for the masked case didn't also break the plain path).
    Graph graph;
    build_attention_pattern(graph, device_, /*with_mask=*/false);

    int orig_nodes = graph.num_nodes();
    FuseAttentionPass pass;
    bool changed = pass.run(graph);
    EXPECT_TRUE(changed) << "FuseAttention did not fire on an unmasked "
                             "MatMul->Mul->Softmax->MatMul pattern";
    EXPECT_LT(graph.num_nodes(), orig_nodes);

    ASSERT_EQ(graph.outputs().size(), 1u);
    auto fused_node = graph.outputs()[0]->node();
    ASSERT_NE(fused_node, nullptr);
    EXPECT_EQ(fused_node->op_type(), OpType::FlashAttention);
    EXPECT_EQ(fused_node->inputs().size(), 3u)
        << "unmasked fusion should produce exactly [Q, K, V]";
    EXPECT_FALSE(fused_node->get_bool_attr("has_mask"));
}

TEST_F(FusionPassesTest, FuseAttention_WithMask_DoesNotFuse) {
    // JIT-R041 regression: every lowering handler for OpType::FlashAttention
    // (iree_customcalls.cpp, lowering.cpp's custom-call AND expand paths —
    // the expand path is the LIVE default) hard-rejects anything but exactly
    // 3 inputs. Before the fix, this masked pattern got fused into a 4-input
    // has_mask=true FlashAttention node that no lowering path could consume.
    // The pass must now refuse to fuse a masked pattern at all, leaving the
    // fully-correct (if unfused/unaccelerated) MatMul->Mul->Add->Softmax->
    // MatMul sequence intact.
    Graph graph;
    build_attention_pattern(graph, device_, /*with_mask=*/true);

    int orig_nodes = graph.num_nodes();
    FuseAttentionPass pass;
    bool changed = pass.run(graph);
    EXPECT_FALSE(changed)
        << "FuseAttentionPass fused a MASKED attention pattern into an "
           "unlowerable 4-input FlashAttention node";
    EXPECT_EQ(graph.num_nodes(), orig_nodes)
        << "graph should be untouched when fusion is refused";

    // No node anywhere in the graph should be the unlowerable 4-input form.
    for (const auto& node : graph.nodes()) {
        if (node->op_type() == OpType::FlashAttention) {
            EXPECT_LE(node->inputs().size(), 3u)
                << "found a FlashAttention node with >3 inputs — exactly "
                   "the unlowerable shape JIT-R041 must prevent";
        }
    }
}

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
// SmallMLP matcher — hidden<=0 must NOT fuse (R1-01 regression).
//
// match_small_mlp's hidden-dim bound check used to only reject
// `hidden > max_mlp_hidden_`, not `hidden <= 0`. ExtendedFusionPass::run
// inherits whatever hidden value comes through, and extended_codegen.cpp
// uses it directly for CUDA/ROCm dynamic shared-memory sizing and a kernel
// loop bound: hidden==0 silently drops the hidden-layer computation (wrong
// result), a negative value cast to unsigned requests a huge allocation
// (kernel launch failure) -- either way, a GPU-only divergence from the
// (never-fused) CPU eager path for a degenerate/corrupted model. A positive
// hidden dim must still fuse correctly (guard must not over-reject).
// ============================================================================
namespace {
// Build: Linear1(x[4,16], w1[hidden,16], b1[hidden]) -> GELU ->
// Linear2(w2[8,hidden], b2[8]). Returns the Linear1 node.
auto build_small_mlp(Graph& g, const Device& dev, int64_t hidden)
    -> std::shared_ptr<Node> {
    auto x  = g.create_value("x",  {4, 16}, DType::Float32, dev);
    auto w1 = g.create_value("w1", {hidden, 16}, DType::Float32, dev);
    auto b1 = g.create_value("b1", {hidden}, DType::Float32, dev);
    auto lin1 = g.create_node(OpType::Linear, "linear1");
    auto lin1_out = g.create_value("lin1_out", {4, hidden}, DType::Float32, dev);
    lin1->add_input(x); lin1->add_input(w1); lin1->add_input(b1);
    lin1->add_output(lin1_out); lin1_out->set_node(lin1);
    g.add_node(lin1);

    auto act_node = g.create_node(OpType::GELU, "act");
    auto act = g.create_value("act_out", {4, hidden}, DType::Float32, dev);
    act_node->add_input(lin1_out); act_node->add_output(act);
    act->set_node(act_node); g.add_node(act_node);

    auto w2 = g.create_value("w2", {8, hidden}, DType::Float32, dev);
    auto b2 = g.create_value("b2", {8}, DType::Float32, dev);
    auto lin2 = g.create_node(OpType::Linear, "linear2");
    auto lin2_out = g.create_value("lin2_out", {4, 8}, DType::Float32, dev);
    lin2->add_input(act); lin2->add_input(w2); lin2->add_input(b2);
    lin2->add_output(lin2_out); lin2_out->set_node(lin2);
    g.add_node(lin2);

    g.set_inputs({x});
    g.set_outputs({lin2_out});
    return lin1;
}
}  // namespace

TEST_F(FusionPassesTest, SmallMLP_ZeroHiddenDimNotFused) {
    Graph graph;
    auto lin1 = build_small_mlp(graph, device_, /*hidden=*/0);
    PatternMatcher matcher;
    auto matches = matcher.find_all(graph);
    for (const auto& m : matches) {
        EXPECT_FALSE(m.kind == FusionKind::SmallMLP &&
                     !m.nodes.empty() && m.nodes[0].get() == lin1.get())
            << "a Linear->GELU->Linear MLP with hidden==0 was wrongly fused "
               "into SmallMLP";
    }
}

TEST_F(FusionPassesTest, SmallMLP_NegativeHiddenDimNotFused) {
    Graph graph;
    auto lin1 = build_small_mlp(graph, device_, /*hidden=*/-1);
    PatternMatcher matcher;
    auto matches = matcher.find_all(graph);
    for (const auto& m : matches) {
        EXPECT_FALSE(m.kind == FusionKind::SmallMLP &&
                     !m.nodes.empty() && m.nodes[0].get() == lin1.get())
            << "a Linear->GELU->Linear MLP with hidden==-1 was wrongly fused "
               "into SmallMLP";
    }
}

TEST_F(FusionPassesTest, SmallMLP_PositiveHiddenDimStillFuses) {
    Graph graph;
    auto lin1 = build_small_mlp(graph, device_, /*hidden=*/32);
    PatternMatcher matcher;
    auto matches = matcher.find_all(graph);
    bool found = false;
    for (const auto& m : matches) {
        if (m.kind == FusionKind::SmallMLP && !m.nodes.empty() &&
            m.nodes[0].get() == lin1.get()) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found)
        << "a genuine Linear->GELU->Linear MLP with a valid hidden dim must "
           "still fuse into SmallMLP (guard over-rejected a real match)";
}

// ============================================================================
// ReductionChain matcher — must only match pre/post element-wise steps
// extended_codegen's generate_reduction can actually lower (R1-09 regression).
//
// generate_reduction (and compiler.cpp's ExtendedFusionPass::to_elem, which
// independently re-derives the same allowlist before committing a fusion) can
// only lower a genuine UNARY op or a same-operand self-square Mul(x,x) — a
// binary op with a DISTINCT second operand has no second data stream in the
// fused kernel's single-input contract. match_reduction_chain used to accept
// any is_elementwise() op (including arbitrary binary ops) and any reduction
// op type regardless of its axis attribute, relying entirely on compiler.cpp
// staying in sync to catch what it structurally shouldn't have matched.
// ============================================================================
namespace {
// Build: Mul(x,x)[optional self-square] -> Sum(dim=-1) -> Sqrt[optional unary
// post-op]. `pre_distinct` swaps the pre-op for Add(x, y) with a genuine
// distinct second operand; `reduction_has_dim` controls whether the Sum node
// carries the required single-axis "dim" attribute.
auto build_reduction_chain(Graph& g, const Device& dev, bool pre_distinct,
                          bool reduction_has_dim) -> std::shared_ptr<Node> {
    const std::vector<int64_t> shape{4, 8};
    const std::vector<int64_t> red_shape{4, 1};
    auto x = g.create_value("x", shape, DType::Float32, dev);

    std::shared_ptr<Node> pre_node;
    std::shared_ptr<Value> pre_out;
    if (pre_distinct) {
        auto y = g.create_value("y", shape, DType::Float32, dev);
        pre_node = g.create_node(OpType::Add, "pre_add");
        pre_out = g.create_value("pre_out", shape, DType::Float32, dev);
        pre_node->add_input(x); pre_node->add_input(y);
    } else {
        pre_node = g.create_node(OpType::Mul, "pre_selfsquare");
        pre_out = g.create_value("pre_out", shape, DType::Float32, dev);
        pre_node->add_input(x); pre_node->add_input(x);  // genuine self-square
    }
    pre_node->add_output(pre_out); pre_out->set_node(pre_node); g.add_node(pre_node);

    auto sum_node = g.create_node(OpType::Sum, "sum");
    auto sum_out = g.create_value("sum_out", red_shape, DType::Float32, dev);
    sum_node->add_input(pre_out); sum_node->add_output(sum_out);
    sum_out->set_node(sum_node);
    if (reduction_has_dim) sum_node->set_int_attr("dim", -1);
    g.add_node(sum_node);

    auto post_node = g.create_node(OpType::Sqrt, "post_sqrt");
    auto post_out = g.create_value("post_out", red_shape, DType::Float32, dev);
    post_node->add_input(sum_out); post_node->add_output(post_out);
    post_out->set_node(post_node); g.add_node(post_node);

    g.set_inputs(pre_distinct ? std::vector<std::shared_ptr<Value>>{x, pre_node->inputs()[1]}
                              : std::vector<std::shared_ptr<Value>>{x});
    g.set_outputs({post_out});
    return pre_node;
}
}  // namespace

TEST_F(FusionPassesTest, ReductionChain_DistinctOperandPreOpNotFused) {
    Graph graph;
    auto pre_node = build_reduction_chain(graph, device_, /*pre_distinct=*/true,
                                          /*reduction_has_dim=*/true);
    PatternMatcher matcher;
    auto matches = matcher.find_all(graph);
    for (const auto& m : matches) {
        EXPECT_FALSE(m.kind == FusionKind::Reduction && !m.nodes.empty() &&
                     m.nodes[0].get() == pre_node.get())
            << "Add(x,y) with a distinct second operand was wrongly fused as "
               "a Reduction pre-op — generate_reduction cannot lower this";
    }
}

TEST_F(FusionPassesTest, ReductionChain_NoAxisAttrNotFused) {
    Graph graph;
    auto pre_node = build_reduction_chain(graph, device_, /*pre_distinct=*/false,
                                          /*reduction_has_dim=*/false);
    PatternMatcher matcher;
    auto matches = matcher.find_all(graph);
    for (const auto& m : matches) {
        EXPECT_FALSE(m.kind == FusionKind::Reduction && !m.nodes.empty() &&
                     m.nodes[0].get() == pre_node.get())
            << "a Sum with no \"dim\" attribute (full/unknown-axis reduction) "
               "was wrongly fused as a Reduction — the fused kernel reduces "
               "exactly one explicit axis";
    }
}

TEST_F(FusionPassesTest, ReductionChain_SelfSquareWithAxisStillFuses) {
    Graph graph;
    auto pre_node = build_reduction_chain(graph, device_, /*pre_distinct=*/false,
                                          /*reduction_has_dim=*/true);
    PatternMatcher matcher;
    auto matches = matcher.find_all(graph);
    bool found = false;
    for (const auto& m : matches) {
        if (m.kind == FusionKind::Reduction && !m.nodes.empty() &&
            m.nodes[0].get() == pre_node.get()) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found)
        << "a genuine self-square Mul(x,x) -> Sum(dim) -> Sqrt chain must "
           "still fuse into Reduction (guard over-rejected a real match)";
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

// JIT-R166 regression fixture: builds Mean -> Sub -> Pow -> Mean -> Add(eps)
// -> Div(sub_o, add_o), i.e. dividing the centered input by (variance + eps)
// DIRECTLY with the Sqrt step omitted entirely -- not a real LayerNorm (which
// divides by the standard deviation sqrt(variance + eps)). This satisfies the
// old "Div present" guard alone; the fix requires a Sqrt/Rsqrt to also be
// present in the matched chain.
auto build_layernorm_no_sqrt(Graph& g, const Device& dev) -> void {
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
    // No Sqrt node -- divide the centered input by (variance + eps) directly.
    auto dv = g.create_node(OpType::Div, "div");
    auto dv_o = g.create_value("div", shape, DType::Float32, dev);
    dv->add_input(sub_o); dv->add_input(add_o); dv->add_output(dv_o);
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

TEST_F(FusionPassesTest, LayerNorm_DivWithoutSqrtNotFused) {
    Graph graph;
    build_layernorm_no_sqrt(graph, device_);
    PatternMatcher matcher;
    auto matches = matcher.find_all(graph);
    EXPECT_FALSE(has_kind(matches, FusionKind::LayerNorm))
        << "a Div present with no Sqrt/Rsqrt anywhere in the chain (dividing "
           "by variance+eps directly, not by the standard deviation) must "
           "NOT fuse as LayerNorm (JIT-R166) -- the old guard only checked "
           "for Div presence, not that the divisor was actually a sqrt";
}

// ============================================================================
// QuantizationPass (JIT-R039)
// ============================================================================

namespace {
// Builds a standalone Conv2d node [x, weight] with the given weight kernel
// shape and (equal H/W) stride/padding/dilation attrs, wired as the graph's
// sole node/output. device_ must support OpId::QuantizedConv2d (true for CPU
// in this codebase) for QuantizationPass::run to even attempt quantizing it.
auto build_conv2d_node(Graph& graph, Device device,
                       int64_t kernel_h, int64_t kernel_w,
                       int64_t stride_h, int64_t stride_w)
    -> std::shared_ptr<Node> {
    auto x = graph.create_value("x", {1, 3, 16, 16}, DType::Float32, device);
    auto w = graph.create_value("w", {8, 3, kernel_h, kernel_w}, DType::Float32, device);
    auto conv_node = graph.create_node(OpType::Conv2d, "conv");
    conv_node->add_input(x);
    conv_node->add_input(w);
    conv_node->set_vec_attr("stride", {stride_h, stride_w});
    conv_node->set_vec_attr("padding", {0, 0});
    conv_node->set_vec_attr("dilation", {1, 1});
    auto out = graph.create_value("conv_out", {1, 8, 8, 8}, DType::Float32, device);
    out->set_node(conv_node);
    conv_node->add_output(out);
    graph.add_node(conv_node);
    graph.set_inputs({x});
    graph.set_outputs({out});
    return conv_node;
}
}  // namespace

TEST_F(FusionPassesTest, Quantization_SquareSymmetricConv2d_Quantizes) {
    // Baseline: a square-kernel, symmetric-stride Conv2d IS eligible for
    // quantization (sanity check that JIT-R039's added rejection didn't
    // also break the plain, safe-to-quantize case).
    Graph graph;
    auto conv = build_conv2d_node(graph, device_, /*kh=*/3, /*kw=*/3,
                                  /*sh=*/2, /*sw=*/2);
    QuantizationPass pass;
    bool changed = pass.run(graph);
    EXPECT_TRUE(changed) << "a square-kernel, symmetric-stride Conv2d should "
                             "have been retagged QuantizedConv2d";
    EXPECT_EQ(conv->op_type(), OpType::QuantizedConv2d);
}

TEST_F(FusionPassesTest, Quantization_NonSquareKernelConv2d_NotQuantized) {
    // JIT-R039 regression: eager's QuantizedConv2d::from_float explicitly
    // REJECTS non-square kernels (the quantized ctor stores a single scalar
    // kernel_size and the interpreter's quantized_conv2d_dynamic reads only
    // weight.shape()[2], silently dropping the W-axis value). The JIT
    // retag must not do what that eager rejection exists to prevent.
    Graph graph;
    auto conv = build_conv2d_node(graph, device_, /*kh=*/3, /*kw=*/5,
                                  /*sh=*/1, /*sw=*/1);
    QuantizationPass pass;
    bool changed = pass.run(graph);
    EXPECT_FALSE(changed) << "a non-square-kernel (3x5) Conv2d must NOT be "
                              "retagged QuantizedConv2d";
    EXPECT_EQ(conv->op_type(), OpType::Conv2d);
}

TEST_F(FusionPassesTest, Quantization_AsymmetricStrideConv2d_NotQuantized) {
    // Same rationale as above, for asymmetric stride (square kernel here so
    // ONLY the stride asymmetry is under test).
    Graph graph;
    auto conv = build_conv2d_node(graph, device_, /*kh=*/3, /*kw=*/3,
                                  /*sh=*/2, /*sw=*/1);
    QuantizationPass pass;
    bool changed = pass.run(graph);
    EXPECT_FALSE(changed) << "an asymmetric-stride (2,1) Conv2d must NOT be "
                              "retagged QuantizedConv2d";
    EXPECT_EQ(conv->op_type(), OpType::Conv2d);
}

// ============================================================================
// LayoutOptimizationPass (JIT-R040)
// ============================================================================

TEST_F(FusionPassesTest, LayoutOptimization_ChannelsLastGraphOutput_GetsContiguousRestore) {
    // JIT-R040 regression: Phase 4 of LayoutOptimizationPass only inserted a
    // contiguous-restoring LayoutConvert at NODE-to-NODE format boundaries —
    // a channels-last node whose output is a graph OUTPUT directly (no other
    // node consumer, e.g. a bare conv-terminated subgraph) never got one,
    // leaving the graph's output physically channels-last while every
    // consumer (buffer protocol, raw data_ptr() reads) expects contiguous.
    // The pass is gated to CUDA/ROCm only (memory_format is real hardware
    // layout there); tag the graph's input with that device — this is pure
    // graph metadata, no physical GPU tensor is touched by this pass.
    Graph graph;
    Device cuda_dev = Device::cuda(0);
    auto x = graph.create_value("x", {1, 3, 8, 8}, DType::Float32, cuda_dev);
    auto w = graph.create_value("w", {4, 3, 3, 3}, DType::Float32, cuda_dev);
    auto conv_node = graph.create_node(OpType::Conv2d, "conv");
    conv_node->add_input(x);
    conv_node->add_input(w);
    auto conv_out = graph.create_value("conv_out", {1, 4, 8, 8}, DType::Float32, cuda_dev);
    conv_out->set_node(conv_node);
    conv_node->add_output(conv_out);
    graph.add_node(conv_node);
    graph.set_inputs({x});
    graph.set_outputs({conv_out});  // conv output IS a graph output directly

    LayoutOptimizationPass pass;
    bool changed = pass.run(graph);
    EXPECT_TRUE(changed);

    ASSERT_EQ(graph.outputs().size(), 1u);
    auto final_producer = graph.outputs()[0]->node();
    ASSERT_NE(final_producer, nullptr);
    EXPECT_EQ(final_producer->op_type(), OpType::LayoutConvert)
        << "a channels-last Conv2d feeding a graph output directly must get "
           "a trailing contiguous-restoring LayoutConvert node";
    EXPECT_EQ(final_producer->get_int_attr("target_format"), 0)
        << "the trailing LayoutConvert must restore Contiguous (0), not "
           "leave the output in ChannelsLast (1)";
}
