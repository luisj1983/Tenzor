/**
 * @file test_jit_correctness_fixes.cpp
 * @brief Phase P0 — JIT correctness fixes (4 items from the fourth-pass
 *        audit). Each test would fail against the pre-fix code.
 *
 *  1. CSE dedups stateful Dropout/BatchNorm nodes — should not.
 *  2. Conv+BatchNorm fusion runs in training mode — should refuse.
 *  3. GraphReader silently drops missing input values — should throw.
 *  4. Node::replace_input leaves stale uses_ entry — should clean up.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/jit/compiler.hpp>
#include <tenzor/jit/graph.hpp>
#include <tenzor/jit/serialization.hpp>
#include <tenzor/jit/tracer.hpp>
#include <tenzor/ops/creation.hpp>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace tenzor { void initialize(); }

namespace {
class JitFixesEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
[[maybe_unused]] auto* g_env =
    ::testing::AddGlobalTestEnvironment(new JitFixesEnv);
}  // namespace

using namespace tenzor;
using namespace tenzor::jit;

namespace {

// Build a minimal graph: input → Dropout → output. Repeat the Dropout
// to give CSE a potential dedup candidate.
auto build_two_dropouts_graph() -> std::shared_ptr<Graph> {
    auto g = std::make_shared<Graph>();
    auto in = g->create_value("x", {4, 4}, DType::Float32, Device::cpu());
    g->set_inputs({in});

    // First Dropout
    auto d1 = g->create_node(OpType::Dropout, "drop1");
    d1->add_input(in);
    auto y1 = g->create_value("y1", {4, 4}, DType::Float32, Device::cpu());
    d1->add_output(y1);
    y1->set_node(d1);
    g->add_node(d1);

    // Second Dropout — same input.
    auto d2 = g->create_node(OpType::Dropout, "drop2");
    d2->add_input(in);
    auto y2 = g->create_value("y2", {4, 4}, DType::Float32, Device::cpu());
    d2->add_output(y2);
    y2->set_node(d2);
    g->add_node(d2);

    g->set_outputs({y1, y2});
    return g;
}

}  // namespace

// =========================================================================
// Fix 1: CSE must not merge stateful ops
// =========================================================================
TEST(JitCorrectnessFixes, CSEDoesNotDedupDropout) {
    auto g = build_two_dropouts_graph();
    ASSERT_EQ(g->nodes().size(), 2u);

    CommonSubexpressionEliminationPass pass;
    pass.run(*g);

    // Pre-fix: CSE would consider the two Dropouts equivalent (same op,
    // same inputs, same attributes) and merge them — collapsing two
    // independent random masks into one.
    EXPECT_EQ(g->nodes().size(), 2u)
        << "CSE incorrectly merged two stateful Dropout nodes";
    EXPECT_NE(g->outputs()[0]->id(), g->outputs()[1]->id())
        << "CSE replaced one Dropout output with the other";
}

TEST(JitCorrectnessFixes, CSEStillDedupsPureOps) {
    // Regression guard: pure ops (Add, Mul, etc.) still get deduplicated.
    auto g = std::make_shared<Graph>();
    auto x = g->create_value("x", {4}, DType::Float32, Device::cpu());
    auto y = g->create_value("y", {4}, DType::Float32, Device::cpu());
    g->set_inputs({x, y});

    auto a1 = g->create_node(OpType::Add, "add1");
    a1->add_input(x); a1->add_input(y);
    auto v1 = g->create_value("v1", {4}, DType::Float32, Device::cpu());
    a1->add_output(v1); v1->set_node(a1); g->add_node(a1);

    auto a2 = g->create_node(OpType::Add, "add2");
    a2->add_input(x); a2->add_input(y);
    auto v2 = g->create_value("v2", {4}, DType::Float32, Device::cpu());
    a2->add_output(v2); v2->set_node(a2); g->add_node(a2);

    g->set_outputs({v1, v2});
    ASSERT_EQ(g->nodes().size(), 2u);

    CommonSubexpressionEliminationPass pass;
    pass.run(*g);

    // After CSE, one Add should be eliminated.
    EXPECT_LE(g->nodes().size(), 1u)
        << "CSE failed to dedup two identical pure Add nodes "
           "(regression: stateful guard over-applied)";
}

// =========================================================================
// Fix 2: Conv+BN fusion refuses to fuse a BN marked as training
// =========================================================================
TEST(JitCorrectnessFixes, ConvBnFusionSkipsTrainingBN) {
    auto g = std::make_shared<Graph>();
    auto x = g->create_value("x", {1, 3, 8, 8}, DType::Float32, Device::cpu());
    g->set_inputs({x});

    // Conv2d node with weight tensor attached.
    auto conv = g->create_node(OpType::Conv2d, "conv");
    conv->add_input(x);
    conv->set_tensor_attr("weight",
        tenzor::randn({4, 3, 3, 3}, DType::Float32, Device::cpu()));
    auto conv_out = g->create_value("cv", {1, 4, 6, 6}, DType::Float32, Device::cpu());
    conv->add_output(conv_out); conv_out->set_node(conv); g->add_node(conv);

    // BatchNorm2d marked training=true.
    auto bn = g->create_node(OpType::BatchNorm2d, "bn");
    bn->add_input(conv_out);
    bn->set_tensor_attr("weight", tenzor::ones({4}, DType::Float32, Device::cpu()));
    bn->set_tensor_attr("bias", tenzor::zeros({4}, DType::Float32, Device::cpu()));
    bn->set_tensor_attr("running_mean", tenzor::zeros({4}, DType::Float32, Device::cpu()));
    bn->set_tensor_attr("running_var", tenzor::ones({4}, DType::Float32, Device::cpu()));
    bn->set_attr("eps", 1e-5f);
    bn->set_bool_attr("training", true);  // <-- the new gate
    auto bn_out = g->create_value("bn_out", {1, 4, 6, 6}, DType::Float32, Device::cpu());
    bn->add_output(bn_out); bn_out->set_node(bn); g->add_node(bn);

    g->set_outputs({bn_out});
    ASSERT_EQ(g->nodes().size(), 2u);

    FuseConvBatchNormPass pass;
    pass.run(*g);

    EXPECT_EQ(g->nodes().size(), 2u)
        << "Conv+BN fusion ran on a training-mode BatchNorm — would silently "
           "diverge from eager (uses running stats instead of batch stats)";
}

// 5th-audit sibling-bug A3: the same training-mode guard must also apply to
// the Conv+BN+ReLU TRIPLE fusion. Pre-fix, fuse_triple ran on training-mode
// BatchNorms and silently folded running_mean/running_var into the conv
// weights — diverging from eager which uses live batch stats.
TEST(JitCorrectnessFixes, ConvBnReluTripleFusionSkipsTrainingBN) {
    auto g = std::make_shared<Graph>();
    auto x = g->create_value("x", {1, 3, 8, 8}, DType::Float32, Device::cpu());
    g->set_inputs({x});

    auto conv = g->create_node(OpType::Conv2d, "conv");
    conv->add_input(x);
    conv->set_tensor_attr("weight",
        tenzor::randn({4, 3, 3, 3}, DType::Float32, Device::cpu()));
    auto conv_out = g->create_value("cv", {1, 4, 6, 6}, DType::Float32, Device::cpu());
    conv->add_output(conv_out); conv_out->set_node(conv); g->add_node(conv);

    auto bn = g->create_node(OpType::BatchNorm2d, "bn");
    bn->add_input(conv_out);
    bn->set_tensor_attr("weight", tenzor::ones({4}, DType::Float32, Device::cpu()));
    bn->set_tensor_attr("bias", tenzor::zeros({4}, DType::Float32, Device::cpu()));
    bn->set_tensor_attr("running_mean", tenzor::zeros({4}, DType::Float32, Device::cpu()));
    bn->set_tensor_attr("running_var", tenzor::ones({4}, DType::Float32, Device::cpu()));
    bn->set_attr("eps", 1e-5f);
    bn->set_bool_attr("training", true);  // the new gate (mirrors pair-fusion)
    auto bn_out = g->create_value("bn_out", {1, 4, 6, 6}, DType::Float32, Device::cpu());
    bn->add_output(bn_out); bn_out->set_node(bn); g->add_node(bn);

    auto relu = g->create_node(OpType::ReLU, "relu");
    relu->add_input(bn_out);
    auto relu_out = g->create_value("relu_out", {1, 4, 6, 6}, DType::Float32, Device::cpu());
    relu->add_output(relu_out); relu_out->set_node(relu); g->add_node(relu);

    g->set_outputs({relu_out});
    ASSERT_EQ(g->nodes().size(), 3u);

    FuseConvBatchNormReluPass pass;
    pass.run(*g);

    EXPECT_EQ(g->nodes().size(), 3u)
        << "Conv+BN+ReLU triple-fusion ran on a training-mode BatchNorm — "
           "would silently diverge from eager (uses running stats instead of "
           "batch stats)";
}

TEST(JitCorrectnessFixes, ConvBnReluTripleFusionStillRunsInEvalMode) {
    // Regression guard: triple fusion still runs when BN is eval-mode.
    auto g = std::make_shared<Graph>();
    auto x = g->create_value("x", {1, 3, 8, 8}, DType::Float32, Device::cpu());
    g->set_inputs({x});

    auto conv = g->create_node(OpType::Conv2d, "conv");
    conv->add_input(x);
    conv->set_tensor_attr("weight",
        tenzor::randn({4, 3, 3, 3}, DType::Float32, Device::cpu()));
    conv->set_tensor_attr("bias", tenzor::zeros({4}, DType::Float32, Device::cpu()));
    auto conv_out = g->create_value("cv", {1, 4, 6, 6}, DType::Float32, Device::cpu());
    conv->add_output(conv_out); conv_out->set_node(conv); g->add_node(conv);

    auto bn = g->create_node(OpType::BatchNorm2d, "bn");
    bn->add_input(conv_out);
    bn->set_tensor_attr("weight", tenzor::ones({4}, DType::Float32, Device::cpu()));
    bn->set_tensor_attr("bias", tenzor::zeros({4}, DType::Float32, Device::cpu()));
    bn->set_tensor_attr("running_mean", tenzor::zeros({4}, DType::Float32, Device::cpu()));
    bn->set_tensor_attr("running_var", tenzor::ones({4}, DType::Float32, Device::cpu()));
    bn->set_attr("eps", 1e-5f);
    // training defaults to false → eval mode → fusion allowed.
    auto bn_out = g->create_value("bn_out", {1, 4, 6, 6}, DType::Float32, Device::cpu());
    bn->add_output(bn_out); bn_out->set_node(bn); g->add_node(bn);

    auto relu = g->create_node(OpType::ReLU, "relu");
    relu->add_input(bn_out);
    auto relu_out = g->create_value("relu_out", {1, 4, 6, 6}, DType::Float32, Device::cpu());
    relu->add_output(relu_out); relu_out->set_node(relu); g->add_node(relu);

    g->set_outputs({relu_out});
    FuseConvBatchNormReluPass pass;
    const bool modified = pass.run(*g);
    EXPECT_TRUE(modified) << "Triple fusion should run in eval mode";
}

TEST(JitCorrectnessFixes, ConvBnFusionStillRunsInEvalMode) {
    // Regression guard: same graph but BN is in eval mode (training=false
    // / absent) — fusion SHOULD run.
    auto g = std::make_shared<Graph>();
    auto x = g->create_value("x", {1, 3, 8, 8}, DType::Float32, Device::cpu());
    g->set_inputs({x});

    auto conv = g->create_node(OpType::Conv2d, "conv");
    conv->add_input(x);
    conv->set_tensor_attr("weight",
        tenzor::randn({4, 3, 3, 3}, DType::Float32, Device::cpu()));
    auto conv_out = g->create_value("cv", {1, 4, 6, 6}, DType::Float32, Device::cpu());
    conv->add_output(conv_out); conv_out->set_node(conv); g->add_node(conv);

    auto bn = g->create_node(OpType::BatchNorm2d, "bn");
    bn->add_input(conv_out);
    bn->set_tensor_attr("weight", tenzor::ones({4}, DType::Float32, Device::cpu()));
    bn->set_tensor_attr("bias", tenzor::zeros({4}, DType::Float32, Device::cpu()));
    bn->set_tensor_attr("running_mean", tenzor::zeros({4}, DType::Float32, Device::cpu()));
    bn->set_tensor_attr("running_var", tenzor::ones({4}, DType::Float32, Device::cpu()));
    bn->set_attr("eps", 1e-5f);
    // No training attr set — defaults to false → eval mode → fusion allowed.
    auto bn_out = g->create_value("bn_out", {1, 4, 6, 6}, DType::Float32, Device::cpu());
    bn->add_output(bn_out); bn_out->set_node(bn); g->add_node(bn);

    g->set_outputs({bn_out});
    FuseConvBatchNormPass pass;
    const bool modified = pass.run(*g);
    EXPECT_TRUE(modified) << "Conv+BN fusion should run in eval mode";
}

// =========================================================================
// Fix 3: GraphReader throws on missing input value
// =========================================================================
TEST(JitCorrectnessFixes, GraphReaderThrowsOnMissingInputValue) {
    // Build a valid graph, serialise it, then surgically corrupt the file
    // so a node input references a value ID that isn't in the values
    // section. Reload — should throw cleanly, not silently load.
    namespace fs = std::filesystem;
    const std::string path = (fs::temp_directory_path() /
                              ("jit_corrupted_" + std::to_string(::getpid()) + ".tzjt")).string();

    // Use a distinctive ASCII pattern in the value ID so we can find its
    // bytes in the serialised file reliably. Strings are length-prefixed
    // (u64 size, then chars).
    constexpr const char* kInputId = "QQQqqqQQQ_input";
    constexpr const char* kOutputId = "RRRrrrRRR_output";

    {
        auto g = std::make_shared<Graph>();
        auto x = g->create_value(kInputId, {4}, DType::Float32, Device::cpu());
        g->set_inputs({x});
        auto neg = g->create_node(OpType::Neg, "neg");
        neg->add_input(x);
        auto y = g->create_value(kOutputId, {4}, DType::Float32, Device::cpu());
        neg->add_output(y); y->set_node(neg); g->add_node(neg);
        g->set_outputs({y});
        GraphWriter w(path);
        w.write(*g);
    }

    // Corrupt the SECOND occurrence of kInputId (the Neg node's input ref)
    // to point at a value ID that doesn't exist. First occurrence is in
    // the values section declaration — leaving that intact means the
    // values table still has kInputId, but the node references a different
    // (non-existent) ID. That's exactly the corruption the fix guards
    // against.
    {
        std::ifstream in(path, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());
        in.close();
        const std::string needle = kInputId;
        size_t first_pos = std::string::npos;
        size_t pos = 0;
        while ((pos = std::string(bytes.begin(), bytes.end()).find(needle, pos))
               != std::string::npos) {
            if (first_pos == std::string::npos) {
                first_pos = pos;
                pos += needle.size();
                continue;
            }
            // Second occurrence — flip one character to break the lookup.
            bytes[pos] = 'Z';
            break;
        }
        std::ofstream out(path, std::ios::binary);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    EXPECT_THROW({
        GraphReader r(path);
        auto g = r.read();
        (void)g;
    }, std::runtime_error)
        << "GraphReader silently accepted a corrupted file where a node "
           "input value ID was not in the values section";

    std::filesystem::remove(path);
}

// =========================================================================
// Fix 4: Node::replace_input updates uses_ on BOTH old and new values
// =========================================================================
TEST(JitCorrectnessFixes, ReplaceInputCleansOldValueUses) {
    auto g = std::make_shared<Graph>();
    auto a = g->create_value("a", {4}, DType::Float32, Device::cpu());
    auto b = g->create_value("b", {4}, DType::Float32, Device::cpu());
    g->set_inputs({a, b});

    auto neg = g->create_node(OpType::Neg, "neg");
    neg->add_input(a);
    auto out = g->create_value("out", {4}, DType::Float32, Device::cpu());
    neg->add_output(out); out->set_node(neg); g->add_node(neg);
    g->set_outputs({out});

    // Now neg's input is `a`. Replace with `b`.
    EXPECT_EQ(a->uses().size(), 1u) << "a should have one consumer (neg)";
    EXPECT_EQ(b->uses().size(), 0u);

    neg->replace_input(0, b);

    // Phase P0 / JIT correctness fix: a's uses_ should no longer mention
    // neg; b's uses_ should now include it. Pre-fix, a's uses_ still
    // contained neg — DCE walking uses_ would conclude a is still live.
    EXPECT_EQ(a->uses().size(), 0u)
        << "Node::replace_input failed to remove this consumer from the "
           "OLD value's uses_ list — DCE and reachability passes see a "
           "stale entry and make wrong decisions";
    EXPECT_EQ(b->uses().size(), 1u)
        << "Node::replace_input did not add the consumer to the new value";
}

// -------------------------------------------------------------------------
// JIT review fix H1: OpType::Stack must be differentiable when the graph is
// replayed in grad_mode (training-through-JIT). Pre-fix, the Stack case wrapped
// its result as Variable(result, /*requires_grad=*/false), severing the
// autograd chain to the stacked inputs — so backward produced NO input grads,
// while the neighbouring Cat case (unsqueeze+cat) worked. Analytic check:
// d/da sum(stack([a, b], 0)) == ones_like(a).
// -------------------------------------------------------------------------
TEST(JitReviewFixes, StackGradientsFlowInGradMode) {
    auto g = std::make_shared<Graph>();
    auto a = g->create_value("a", {3, 4}, DType::Float32, Device::cpu());
    auto b = g->create_value("b", {3, 4}, DType::Float32, Device::cpu());
    g->set_inputs({a, b});

    auto st = g->create_node(OpType::Stack, "stack");
    st->add_input(a);
    st->add_input(b);
    st->set_int_attr("dim", 0);
    auto out = g->create_value("s", {2, 3, 4}, DType::Float32, Device::cpu());
    st->add_output(out);
    out->set_node(st);
    g->add_node(st);
    g->set_outputs({out});

    Variable va(randn({3, 4}, DType::Float32, Device::cpu()), /*requires_grad=*/true);
    Variable vb(randn({3, 4}, DType::Float32, Device::cpu()), /*requires_grad=*/true);

    auto results = g->forward({va, vb}, /*grad_mode=*/true);
    ASSERT_EQ(results.size(), 1u);
    ASSERT_TRUE(results[0].requires_grad())
        << "Stack replay severed grad tracking (requires_grad=false output)";

    Variable loss = tenzor::sum(results[0]);
    loss.backward();

    ASSERT_TRUE(va.has_grad()) << "no gradient reached stacked input a";
    ASSERT_TRUE(vb.has_grad()) << "no gradient reached stacked input b";
    // Each grad element is 1 (sum over the stacked tensor); 3*4 = 12.
    Tensor ga = va.grad().value().to(DType::Float32).to(Device::cpu());
    Tensor gb = vb.grad().value().to(DType::Float32).to(Device::cpu());
    float sa = tenzor::sum(Variable(ga, false)).tensor().item<float>();
    float sb = tenzor::sum(Variable(gb, false)).tensor().item<float>();
    EXPECT_NEAR(sa, 12.0f, 1e-4f);
    EXPECT_NEAR(sb, 12.0f, 1e-4f);
}
