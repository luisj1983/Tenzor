/**
 * @file test_dynamic_shapes.cpp
 * @brief Tests for JIT dynamic shape support via SymbolicShapeEnvironment
 */

#include <gtest/gtest.h>
#include <filesystem>
#include <memory>
#include <string>
#include <tenzor/jit/symbolic_shape.hpp>
#include <tenzor/jit/graph.hpp>
#include <tenzor/jit/compiler.hpp>
#include <tenzor/jit/tracer.hpp>
#include <tenzor/jit/symbolic_shape_inference.hpp>
#include <tenzor/core/dtype.hpp>
#include <tenzor/core/device.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/ops/creation.hpp>
#include "../backend_parity/parity_test_utils.hpp"

using namespace tenzor::jit;
using namespace tenzor::testing;

namespace {
std::shared_ptr<Value> mk_val(std::string id, std::vector<int64_t> shape) {
    return std::make_shared<Value>(std::move(id), std::move(shape),
                                   tenzor::DType::Float32, tenzor::Device::cpu());
}

// JIT-R015's regression test is the first test in this file to actually
// dispatch real tensor ops (randn/Linear::forward via CompiledModule::trace)
// rather than pure Graph/SymbolicShape unit-level construction, so it needs
// the backend dispatch tables initialized (mirrors test_jit_trace_ops.cpp's
// identical pattern).
class DynamicShapesEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
[[maybe_unused]] auto* g_dynamic_shapes_env =
    ::testing::AddGlobalTestEnvironment(new DynamicShapesEnv);
}  // namespace

// JIT-F021: a permutation whose length differs from the input rank must yield
// no confident inference (empty), not a wrong-rank shape.
TEST(SymbolicShapeInferenceFixes, PermuteWrongLengthReturnsEmpty) {
    auto x = mk_val("x", {2, 3, 4});
    auto node = std::make_shared<Node>(OpType::Permute);
    node->add_input(x);
    node->add_output(mk_val("y", {}));
    node->set_vec_attr("dims", {0, 2});  // length 2 != rank 3
    SymbolicShapeInference infer;
    EXPECT_TRUE(infer.infer(node.get()).empty());
    node->set_vec_attr("dims", {0, 2, 1});  // valid full-rank permutation
    auto out = infer.infer(node.get());
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].rank(), 3u);
}

// JIT-F020: a dimensioned Max produces values + indices; both outputs must
// receive the reduced symbolic shape.
TEST(SymbolicShapeInferenceFixes, MaxWithDimEmitsTwoOutputShapes) {
    auto x = mk_val("x", {2, 3, 4});
    auto node = std::make_shared<Node>(OpType::Max);
    node->add_input(x);
    node->add_output(mk_val("vals", {}));
    node->add_output(mk_val("idx", {}));
    node->set_int_attr("dim", 1);
    node->set_bool_attr("keepdim", false);
    SymbolicShapeInference infer;
    auto out = infer.infer(node.get());
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].rank(), 2u);
    EXPECT_EQ(out[1].rank(), 2u);
}

// JIT-R021: a duplicate index in Squeeze's "dims" attr (e.g. {0,0}) must not
// erase two elements. shape [1,1,5] with dims={0,0} (dedups to just {0}) must
// squeeze ONLY dim 0 -> [1,5] (rank 2). The pre-fix code sorted descending but
// never deduped: erasing index 0 twice on an already-shrunk vector landed the
// second erase on originally-index-1 (also size-1), silently also squeezing
// it -> wrong-rank [5] (rank 1).
TEST(SymbolicShapeInferenceFixes, SqueezeDupDimsDoesNotOverErase) {
    auto x = mk_val("x", {1, 1, 5});
    auto node = std::make_shared<Node>(OpType::Squeeze);
    node->add_input(x);
    node->add_output(mk_val("y", {}));
    node->set_vec_attr("dims", {0, 0});
    SymbolicShapeInference infer;
    auto out = infer.infer(node.get());
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].rank(), 2u);
    EXPECT_TRUE(out[0][0].is_concrete());
    EXPECT_EQ(out[0][0].value(), 1);
    EXPECT_TRUE(out[0][1].is_concrete());
    EXPECT_EQ(out[0][1].value(), 5);
}

// JIT-F022: a genuinely fixed reshape (no -1, trailing product differs from the
// input's) on a dynamic input must resolve to the concrete target, not a
// symbolized leading dim.
TEST(SymbolicShapeInferenceFixes, FixedReshapeOnDynamicInputStaysConcrete) {
    auto x = mk_val("x", {0, 10});
    x->set_symbolic_shape(SymbolicShape(
        {SymbolicDim::symbolic("B"), SymbolicDim::concrete(10)}));
    auto node = std::make_shared<Node>(OpType::Reshape);
    node->add_input(x);
    node->add_output(mk_val("y", {}));
    node->set_vec_attr("shape", {5, 4});  // fixed; trailing 4 != input trailing 10
    SymbolicShapeInference infer;
    auto out = infer.infer(node.get());
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].rank(), 2u);
    ASSERT_TRUE(out[0][0].is_concrete());
    EXPECT_EQ(out[0][0].value(), 5);
    ASSERT_TRUE(out[0][1].is_concrete());
    EXPECT_EQ(out[0][1].value(), 4);
}

// Companion: a batch-preserving reshape (trailing product matches) still
// symbolizes the leading dim.
TEST(SymbolicShapeInferenceFixes, BatchReshapeOnDynamicInputSymbolizesLeading) {
    auto x = mk_val("x", {0, 2, 5});
    x->set_symbolic_shape(SymbolicShape({SymbolicDim::symbolic("B"),
                                         SymbolicDim::concrete(2),
                                         SymbolicDim::concrete(5)}));
    auto node = std::make_shared<Node>(OpType::Reshape);
    node->add_input(x);
    node->add_output(mk_val("y", {}));
    node->set_vec_attr("shape", {0, 10});  // placeholder leading, trailing 10 == 2*5
    // Use a concrete leading in the target to mimic x.reshape(x.size(0), 10):
    node->set_vec_attr("shape", {32, 10});
    SymbolicShapeInference infer;
    auto out = infer.infer(node.get());
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].rank(), 2u);
    EXPECT_TRUE(out[0][0].is_symbolic());   // leading dim tracks batch
    ASSERT_TRUE(out[0][1].is_concrete());
    EXPECT_EQ(out[0][1].value(), 10);
}

// JIT-F040 — shape-preserving / rank-changing ops previously fell to
// default -> {} in SymbolicShapeInference::infer(), freezing the dynamic batch
// dim to its trace-time size. They must now propagate the symbolic dim.
TEST(SymbolicShapeInferenceFixes, CastPreservesDynamicDim) {
    auto x = mk_val("x", {0, 10});
    x->set_symbolic_shape(SymbolicShape(
        {SymbolicDim::symbolic("B"), SymbolicDim::concrete(10)}));
    auto node = std::make_shared<Node>(OpType::Cast);
    node->add_input(x);
    node->add_output(mk_val("y", {}));
    SymbolicShapeInference infer;
    auto out = infer.infer(node.get());
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].rank(), 2u);
    EXPECT_TRUE(out[0][0].is_symbolic());
    ASSERT_TRUE(out[0][1].is_concrete());
    EXPECT_EQ(out[0][1].value(), 10);
}

// JIT-R010 — SymbolicShapeInference::infer() was missing rules for ~20
// OpTypes that Graph::infer_symbolic_types() already handled, silently
// freezing dynamic dims for graphs containing them (same JIT-F040 pattern
// as CastPreservesDynamicDim above). Cover a representative cross-section.

TEST(SymbolicShapeInferenceFixes, FmodPreservesDynamicDim) {
    auto x = mk_val("x", {0, 10});
    x->set_symbolic_shape(SymbolicShape(
        {SymbolicDim::symbolic("B"), SymbolicDim::concrete(10)}));
    auto y = mk_val("y", {0, 10});
    y->set_symbolic_shape(SymbolicShape(
        {SymbolicDim::symbolic("B"), SymbolicDim::concrete(10)}));
    auto node = std::make_shared<Node>(OpType::Fmod);
    node->add_input(x);
    node->add_input(y);
    node->add_output(mk_val("z", {}));
    SymbolicShapeInference infer;
    auto out = infer.infer(node.get());
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].rank(), 2u);
    EXPECT_TRUE(out[0][0].is_symbolic());
}

TEST(SymbolicShapeInferenceFixes, RoundSiLURoPERollScatterPreserveDynamicDim) {
    for (auto op : {OpType::Round, OpType::SiLU, OpType::RoPE, OpType::Roll,
                    OpType::Scatter}) {
        auto x = mk_val("x", {0, 10});
        x->set_symbolic_shape(SymbolicShape(
            {SymbolicDim::symbolic("B"), SymbolicDim::concrete(10)}));
        auto node = std::make_shared<Node>(op);
        node->add_input(x);
        node->add_output(mk_val("y", {}));
        SymbolicShapeInference infer;
        auto out = infer.infer(node.get());
        ASSERT_EQ(out.size(), 1u) << "op=" << static_cast<int>(op);
        ASSERT_EQ(out[0].rank(), 2u) << "op=" << static_cast<int>(op);
        EXPECT_TRUE(out[0][0].is_symbolic()) << "op=" << static_cast<int>(op);
    }
}

TEST(SymbolicShapeInferenceFixes, MaxPool2dComputesPooledSpatialDims) {
    // Concrete 4D input [1,3,8,8], kernel=2 stride=2 -> [1,3,4,4].
    auto x = mk_val("x", {1, 3, 8, 8});
    auto node = std::make_shared<Node>(OpType::MaxPool2d);
    node->add_input(x);
    node->add_output(mk_val("y", {}));
    node->set_vec_attr("kernel_size", {2, 2});
    node->set_vec_attr("stride", {2, 2});
    SymbolicShapeInference infer;
    auto out = infer.infer(node.get());
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].rank(), 4u);
    EXPECT_EQ(out[0][0].value(), 1);
    EXPECT_EQ(out[0][1].value(), 3);
    EXPECT_EQ(out[0][2].value(), 4);
    EXPECT_EQ(out[0][3].value(), 4);
}

TEST(SymbolicShapeInferenceFixes, AdaptiveAvgPool2dUsesOutputSizeAttr) {
    auto x = mk_val("x", {1, 3, 17, 23});
    auto node = std::make_shared<Node>(OpType::AdaptiveAvgPool2d);
    node->add_input(x);
    node->add_output(mk_val("y", {}));
    node->set_vec_attr("output_size", {7, 7});
    SymbolicShapeInference infer;
    auto out = infer.infer(node.get());
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].rank(), 4u);
    EXPECT_EQ(out[0][2].value(), 7);
    EXPECT_EQ(out[0][3].value(), 7);
}

TEST(SymbolicShapeInferenceFixes, ConvTransposeComputesUpsampledSpatialDims) {
    // Input [1,4,4,4], weight [4,8,3,3] (in=4,out=8,k=3), stride=2, padding=1,
    // output_padding=1 -> H_out = (4-1)*2 - 2*1 + 1*(3-1) + 1 + 1 = 8.
    auto x = mk_val("x", {1, 4, 4, 4});
    auto w = mk_val("w", {4, 8, 3, 3});
    auto node = std::make_shared<Node>(OpType::ConvTranspose);
    node->add_input(x);
    node->add_input(w);
    node->add_output(mk_val("y", {}));
    node->set_vec_attr("stride", {2, 2});
    node->set_vec_attr("padding", {1, 1});
    node->set_vec_attr("output_padding", {1, 1});
    SymbolicShapeInference infer;
    auto out = infer.infer(node.get());
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].rank(), 4u);
    EXPECT_EQ(out[0][0].value(), 1);   // batch
    EXPECT_EQ(out[0][1].value(), 8);   // out_channels = w.shape[1]
    EXPECT_EQ(out[0][2].value(), 8);   // H_out
    EXPECT_EQ(out[0][3].value(), 8);   // W_out
}

TEST(SymbolicShapeInferenceFixes, QuantizedLinearSharesLinearShapeRule) {
    auto x = mk_val("x", {0, 16});
    x->set_symbolic_shape(SymbolicShape(
        {SymbolicDim::symbolic("B"), SymbolicDim::concrete(16)}));
    auto w = mk_val("w", {32, 16});  // [out_features, in_features]
    auto node = std::make_shared<Node>(OpType::QuantizedLinear);
    node->add_input(x);
    node->add_input(w);
    node->add_output(mk_val("y", {}));
    SymbolicShapeInference infer;
    auto out = infer.infer(node.get());
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].rank(), 2u);
    EXPECT_TRUE(out[0][0].is_symbolic());
    EXPECT_EQ(out[0][1].value(), 32);
}

TEST(SymbolicShapeInferenceFixes, EmbeddingAppendsEmbeddingDim) {
    auto weight = mk_val("w", {1000, 64});  // [num_embeddings, embedding_dim]
    auto idx = mk_val("idx", {0, 5});
    idx->set_symbolic_shape(SymbolicShape(
        {SymbolicDim::symbolic("B"), SymbolicDim::concrete(5)}));
    auto node = std::make_shared<Node>(OpType::Embedding);
    node->add_input(weight);
    node->add_input(idx);
    node->add_output(mk_val("y", {}));
    SymbolicShapeInference infer;
    auto out = infer.infer(node.get());
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].rank(), 3u);
    EXPECT_TRUE(out[0][0].is_symbolic());
    EXPECT_EQ(out[0][1].value(), 5);
    EXPECT_EQ(out[0][2].value(), 64);
}

TEST(SymbolicShapeInferenceFixes, EigvalshDropsLastDim) {
    auto x = mk_val("x", {4, 4});
    auto node = std::make_shared<Node>(OpType::Eigvalsh);
    node->add_input(x);
    node->add_output(mk_val("y", {}));
    SymbolicShapeInference infer;
    auto out = infer.infer(node.get());
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].rank(), 1u);
    EXPECT_EQ(out[0][0].value(), 4);
}

TEST(SymbolicShapeInferenceFixes, FusedFFNUsesSecondLinearWeight) {
    auto x = mk_val("x", {0, 16});
    x->set_symbolic_shape(SymbolicShape(
        {SymbolicDim::symbolic("B"), SymbolicDim::concrete(16)}));
    auto w1 = mk_val("w1", {64, 16});
    auto w2 = mk_val("w2", {16, 64});  // second linear: out_features=16
    auto node = std::make_shared<Node>(OpType::FusedFFN);
    node->add_input(x);
    node->add_input(w1);
    node->add_input(w2);
    node->add_output(mk_val("y", {}));
    SymbolicShapeInference infer;
    auto out = infer.infer(node.get());
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].rank(), 2u);
    EXPECT_TRUE(out[0][0].is_symbolic());
    EXPECT_EQ(out[0][1].value(), 16);
}

TEST(SymbolicShapeInferenceFixes, FlattenPreservesDynamicBatch) {
    // conv->Flatten->Linear: Flatten(start_dim=1) over [B,3,4] -> [B, 12] with B
    // still symbolic (12 = 3*4 concrete product).
    auto x = mk_val("x", {0, 3, 4});
    x->set_symbolic_shape(SymbolicShape({SymbolicDim::symbolic("B"),
                                         SymbolicDim::concrete(3),
                                         SymbolicDim::concrete(4)}));
    auto node = std::make_shared<Node>(OpType::Flatten);
    node->add_input(x);
    node->add_output(mk_val("y", {}));
    node->set_int_attr("start_dim", 1);
    node->set_int_attr("end_dim", -1);
    SymbolicShapeInference infer;
    auto out = infer.infer(node.get());
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].rank(), 2u);
    EXPECT_TRUE(out[0][0].is_symbolic());
    ASSERT_TRUE(out[0][1].is_concrete());
    EXPECT_EQ(out[0][1].value(), 12);
}

// F031: a module made dynamic via mark_dynamic_dims must reload DYNAMIC, not as a
// static graph frozen at the trace-time shape. dynamic_dims_ is serialized into
// graph metadata and re-applied on load.
TEST(SymbolicShapeInferenceFixes, DynamicModuleRoundTripsAsDynamic) {
    auto g = std::make_shared<Graph>();
    auto x = g->create_value("x", {4, 8}, tenzor::DType::Float32,
                             tenzor::Device::cpu());
    auto y = g->create_value("y", {4, 8}, tenzor::DType::Float32,
                             tenzor::Device::cpu());
    auto relu = g->create_node(OpType::ReLU, "relu");
    relu->add_input(x);
    relu->add_output(y);
    y->set_node(relu);
    g->add_node(relu);
    g->set_inputs({x});
    g->set_outputs({y});

    auto module = std::make_shared<CompiledModule>(g);
    std::vector<CompiledModule::DynamicDimSpec> specs = {{0, 0, "B"}};
    module->mark_dynamic_dims(specs);
    ASSERT_TRUE(module->has_dynamic_shapes());

    const auto path = std::filesystem::temp_directory_path() /
                      ("tenzor_f031_" + std::to_string(::getpid()) + ".tzg");
    module->save(path.string());
    auto loaded = CompiledModule::load(path.string());
    std::filesystem::remove(path);

    ASSERT_NE(loaded, nullptr);
    EXPECT_TRUE(loaded->has_dynamic_shapes())
        << "dynamic module reloaded as STATIC — dynamic_dims not serialized (F031)";
}

// R3-03: ShapeGuardInsertionPass used to bake every input dimension's
// trace-time concrete size into the guard's "expected_shape" attribute
// unconditionally, with no check of Value::has_symbolic_dims() --
// mark_dynamic_dims() had zero effect on what the guard actually checked at
// runtime (it only touches the separate symbolic_shape_ field). For a
// TRACED CompiledModule (source_module_ set) this was masked: forward()'s
// own shape-key comparison force-retraces on any shape change before the
// frozen guard could ever fire. But a LOADED module (this test's `loaded`,
// built directly from a Graph/deserialized -- source_module_ is null) has
// no such preemptive retrace, so the frozen guard was the only check that
// ran, and it threw "...ShapeGuard...but no source module is available to
// retrace with...Cannot safely continue" on the very first differently-
// shaped input -- defeating the entire point of a dynamic-dim module
// surviving a save()/load() round-trip (F031, tested above). This
// combines mark_dynamic_dims() with optimize_for_inference() (which is
// what actually runs ShapeGuardInsertionPass) and calls forward() on the
// LOADED module with a different batch size, which DynamicModuleRoundTrips
// AsDynamic above does not do.
TEST(SymbolicShapeInferenceFixes, LoadedDynamicModuleAcceptsDifferentBatchSize) {
    auto g = std::make_shared<Graph>();
    auto x = g->create_value("x", {4, 8}, tenzor::DType::Float32,
                             tenzor::Device::cpu());
    auto y = g->create_value("y", {4, 8}, tenzor::DType::Float32,
                             tenzor::Device::cpu());
    auto relu = g->create_node(OpType::ReLU, "relu");
    relu->add_input(x);
    relu->add_output(y);
    y->set_node(relu);
    g->add_node(relu);
    g->set_inputs({x});
    g->set_outputs({y});

    auto module = std::make_shared<CompiledModule>(g);
    module->mark_dynamic_dims({{0, 0, "B"}});
    // Runs the default pass pipeline, including ShapeGuardInsertionPass --
    // the pass under test. Without mark_dynamic_dims() having been called
    // first (order matters: SymbolicTracePass must annotate the graph's
    // input Values with a symbolic dim BEFORE this pass reads them), the
    // guard would bake batch=4 as concrete regardless of this fix.
    module->optimize_for_inference();

    const auto path = std::filesystem::temp_directory_path() /
                      ("tenzor_r303_" + std::to_string(::getpid()) + ".tzg");
    module->save(path.string());
    auto loaded = CompiledModule::load(path.string());
    std::filesystem::remove(path);
    ASSERT_NE(loaded, nullptr);
    ASSERT_TRUE(loaded->has_dynamic_shapes());

    // Same batch size (4) as trace time: must always have worked, even
    // before the fix -- included as a sanity baseline.
    auto raw4 = tenzor::randn({4, 8}, tenzor::DType::Float32, tenzor::Device::cpu());
    auto input4 = tenzor::Variable(raw4, /*requires_grad=*/false);
    ASSERT_NO_THROW({ (void)loaded->forward(input4); });

    // Different batch size (7): before the fix, this threw instead of
    // running, on every backend (the bug is in shape-guard bookkeeping, not
    // any backend kernel).
    auto raw7 = tenzor::randn({7, 8}, tenzor::DType::Float32, tenzor::Device::cpu());
    auto input7 = tenzor::Variable(raw7, /*requires_grad=*/false);
    tenzor::Variable out7;
    ASSERT_NO_THROW({ out7 = loaded->forward(input7); })
        << "loaded dynamic-dim module must accept a batch size other than "
           "the one it was traced/saved with, without throwing";

    auto out_shape = out7.tensor().shape();
    ASSERT_EQ(out_shape.size(), 2u);
    EXPECT_EQ(out_shape[0], 7);
    EXPECT_EQ(out_shape[1], 8);

    auto expected = tenzor::nn::relu(input7);
    auto diff = tenzor::max(tenzor::abs(expected.tensor() - out7.tensor()))
                    .item<float>();
    EXPECT_LT(diff, 1e-6f)
        << "loaded module's output at the new batch size must match eager "
           "relu, not just avoid throwing";
}

// JIT-R015 regression: a device/dtype-mismatch retrace (CompiledModule::
// forward's shape/device/dtype cache miss path) must propagate the
// mark_dynamic_dims() configuration and add_metadata() key/values onto the
// retraced replacement graph. Before this fix, dynamic_dims_/metadata_ are
// members of the ORIGINAL CompiledModule (not the Graph), so a retrace
// silently dropped them: has_dynamic_shapes() would go from true to
// (functionally) false, degrading a caller's "reuse one dynamic-shape
// graph" configuration into "retrace on every subsequent shape change" —
// confirmed perf-only (compute_shape_key() already retraces
// unconditionally on any shape change regardless of dynamic_dims_
// survival), never a wrong-output bug, but a real regression nonetheless.
TEST(SymbolicShapeInferenceFixes, RetraceOnDeviceMismatchPreservesDynamicDimsAndMetadata) {
    auto lin = std::make_shared<tenzor::nn::Linear>(4, 4);
    auto cpu_input = tenzor::Variable(
        tenzor::randn({2, 4}, tenzor::DType::Float32, tenzor::Device::cpu()), false);

    auto module = CompiledModule::trace(lin, cpu_input);
    module->mark_dynamic_dims({{0, 0, "batch"}});
    module->add_metadata("model_name", "r015_regression");
    ASSERT_TRUE(module->has_dynamic_shapes());

    // Baseline call on the trace device — no retrace yet.
    ASSERT_NO_THROW({ (void)module->forward(cpu_input); });
    ASSERT_TRUE(module->has_dynamic_shapes());

    for (const auto& dev : get_available_backends()) {
        if (dev.type == tenzor::Device::Type::CPU) continue;  // already the trace device

        auto other_device_input = tenzor::Variable(
            tenzor::randn({3, 4}, tenzor::DType::Float32, dev), false);

        // This call's device differs from the trace device -> forces the
        // device-mismatch retrace path in CompiledModule::forward.
        ASSERT_NO_THROW({ (void)module->forward(other_device_input); })
            << "retrace threw on " << backend_name(dev);

        EXPECT_TRUE(module->has_dynamic_shapes())
            << "mark_dynamic_dims() config was dropped by a device-mismatch "
               "retrace on " << backend_name(dev);
        EXPECT_EQ(module->get_metadata("model_name"), "r015_regression")
            << "add_metadata() was dropped by a device-mismatch retrace on "
            << backend_name(dev);

        // Prove has_dynamic_shapes() isn't just a stale flag: a THIRD call
        // with yet another batch size, on the NEW device, must still work
        // without throwing (the dynamic batch dim genuinely still functions
        // on the retraced graph).
        auto third_input = tenzor::Variable(
            tenzor::randn({5, 4}, tenzor::DType::Float32, dev), false);
        tenzor::Variable out;
        ASSERT_NO_THROW({ out = module->forward(third_input); })
            << "a further batch-size change on " << backend_name(dev)
            << " threw after the dynamic-dims config should have survived "
               "the earlier retrace";
        EXPECT_EQ(out.tensor().shape()[0], 5);
        break;  // one non-CPU backend is enough to exercise the retrace path
    }
}

// JIT-R020 regression: Graph::partition_at() must re-classify a boundary
// input backed by the PARENT graph's constants_/param_leaves_/
// buffer_leaves_ onto the sub-graph the same way, instead of leaving it as
// a generic boundary "sub_input" — a caller reconstructing that wiring
// manually (e.g. snapshotting a parameter's current value once) would
// otherwise reintroduce the exact frozen-value bug JIT-R005 fixed, at every
// partition boundary. partition_at() currently has no callers in this
// codebase (the obvious future use is pipeline-parallel/multi-device
// partitioning), so this exercises the Graph-level API directly rather
// than through any higher-level entry point.
TEST(SymbolicShapeInferenceFixes, PartitionAtForwardsConstantsParametersAndBuffers) {
    auto g = std::make_shared<Graph>();

    auto x = g->create_value("x", {2, 4}, tenzor::DType::Float32, tenzor::Device::cpu());
    auto w = g->create_value("w", {2, 4}, tenzor::DType::Float32, tenzor::Device::cpu());
    auto buf = g->create_value("buf", {2, 4}, tenzor::DType::Float32, tenzor::Device::cpu());
    auto b = g->create_value("b", {2, 4}, tenzor::DType::Float32, tenzor::Device::cpu());

    // Node A (partition 0): mid = x + w  (w is a param leaf)
    auto mid = g->create_value("mid", {2, 4}, tenzor::DType::Float32, tenzor::Device::cpu());
    auto add_a = g->create_node(OpType::Add, "add_a");
    add_a->add_input(x);
    add_a->add_input(w);
    add_a->add_output(mid);
    mid->set_node(add_a);
    g->add_node(add_a);

    // Marker node at index 1 — excluded by the break, splitting the graph
    // into partition0={add_a} and partition1={add_b1, add_b2}.
    auto mid2 = g->create_value("mid2", {2, 4}, tenzor::DType::Float32, tenzor::Device::cpu());
    auto marker = g->create_node(OpType::ReLU, "marker");
    marker->add_input(mid);
    marker->add_output(mid2);
    mid2->set_node(marker);
    g->add_node(marker);

    // Node B (partition 1): y = (mid + b) + buf. "mid" is a genuine cross-
    // partition boundary (produced by partition 0); "b" is a plain
    // constant; "buf" is a buffer leaf.
    auto tmp = g->create_value("tmp", {2, 4}, tenzor::DType::Float32, tenzor::Device::cpu());
    auto add_b1 = g->create_node(OpType::Add, "add_b1");
    add_b1->add_input(mid);
    add_b1->add_input(b);
    add_b1->add_output(tmp);
    tmp->set_node(add_b1);
    g->add_node(add_b1);

    auto y = g->create_value("y", {2, 4}, tenzor::DType::Float32, tenzor::Device::cpu());
    auto add_b2 = g->create_node(OpType::Add, "add_b2");
    add_b2->add_input(tmp);
    add_b2->add_input(buf);
    add_b2->add_output(y);
    y->set_node(add_b2);
    g->add_node(add_b2);

    g->set_inputs({x});
    g->set_outputs({y});

    // Wire "w" as a param leaf, "buf" as a buffer leaf, "b" as a plain
    // constant — exactly what end_trace() does for a captured module
    // parameter / buffer / other constant.
    auto w_var = std::make_shared<tenzor::Variable>(
        tenzor::randn({2, 4}, tenzor::DType::Float32, tenzor::Device::cpu()), true);
    g->set_parameters({w_var});
    g->add_param_leaf("w", 0);

    auto buf_var = std::make_shared<tenzor::Variable>(
        tenzor::randn({2, 4}, tenzor::DType::Float32, tenzor::Device::cpu()), false);
    g->set_buffers({buf_var});
    g->add_buffer_leaf("buf", 0);

    g->set_constant("b", tenzor::randn({2, 4}, tenzor::DType::Float32, tenzor::Device::cpu()));

    auto partitions = g->partition_at({1});
    ASSERT_EQ(partitions.size(), 2u);

    EXPECT_TRUE(partitions[0]->param_leaves().count("w") > 0)
        << "partition 0's boundary input 'w' was not re-classified as a "
           "param leaf";
    EXPECT_EQ(partitions[0]->parameters().size(), 1u);

    EXPECT_TRUE(partitions[1]->constants().count("b") > 0)
        << "partition 1's boundary input 'b' was not re-classified as a "
           "constant";
    EXPECT_TRUE(partitions[1]->buffer_leaves().count("buf") > 0)
        << "partition 1's boundary input 'buf' was not re-classified as a "
           "buffer leaf";
    EXPECT_EQ(partitions[1]->buffers().size(), 1u);

    // "mid" is a genuine cross-partition boundary — must remain a plain
    // sub-graph INPUT, not a constant/param/buffer leaf.
    bool mid_is_input = false;
    for (const auto& inp : partitions[1]->inputs()) {
        if (inp->id() == "mid") { mid_is_input = true; break; }
    }
    EXPECT_TRUE(mid_is_input);
    EXPECT_FALSE(partitions[1]->constants().count("mid") > 0);
    EXPECT_FALSE(partitions[1]->param_leaves().count("mid") > 0);
    EXPECT_FALSE(partitions[1]->buffer_leaves().count("mid") > 0);
}

class DynamicShapesTest : public ::testing::Test {};

// JIT-R018 regression: Graph::infer_symbolic_types()'s Interpolate case must
// leave the output's trace-time symbolic shape intact for the scale_factor
// form (no "output_size" attr) — NOT overwrite it with the unscaled input
// shape. Pre-fix, the push_back sat outside the `!size.empty()` guard, so an
// Interpolate node with no output_size attr always corrupted its output
// shape to match the input's.
TEST(GraphInferSymbolicTypes, InterpolateScaleFactorFormPreservesOutputShape) {
    Graph g;
    auto x = g.create_value("x", {1, 3, 4, 4}, tenzor::DType::Float32,
                            tenzor::Device::cpu());
    x->set_symbolic_shape(SymbolicShape({
        SymbolicDim::concrete(1), SymbolicDim::concrete(3),
        SymbolicDim::concrete(4), SymbolicDim::concrete(4)}));
    g.set_inputs({x});

    auto node = g.create_node(OpType::Interpolate, "interp");
    node->add_input(x);
    // scale_factor form: deliberately no "output_size" attr set.

    auto out = g.create_value("y", {1, 3, 8, 8}, tenzor::DType::Float32,
                              tenzor::Device::cpu());
    // Trace-time-correct output symbolic shape (2x upsample), the value
    // infer_symbolic_types() must NOT clobber.
    out->set_symbolic_shape(SymbolicShape({
        SymbolicDim::concrete(1), SymbolicDim::concrete(3),
        SymbolicDim::concrete(8), SymbolicDim::concrete(8)}));
    out->set_node(node);
    node->add_output(out);
    g.add_node(node);
    g.set_outputs({out});

    g.infer_symbolic_types();

    ASSERT_TRUE(out->has_symbolic_shape());
    ASSERT_EQ(out->symbolic_shape().rank(), 4u);
    EXPECT_EQ(out->symbolic_shape()[2].value(), 8)
        << "Interpolate (scale_factor form) corrupted its output shape to "
           "the unscaled input shape instead of leaving the trace-time "
           "shape intact";
    EXPECT_EQ(out->symbolic_shape()[3].value(), 8);
}

TEST_F(DynamicShapesTest, SymbolicDimConcrete) {
    auto dim = SymbolicDim::concrete(64);
    EXPECT_TRUE(dim.is_concrete());
    EXPECT_EQ(dim.value(), 64);
}

TEST_F(DynamicShapesTest, SymbolicDimSymbolic) {
    auto dim = SymbolicDim::symbolic("batch");
    EXPECT_FALSE(dim.is_concrete());
    EXPECT_EQ(dim.name(), "batch");
}

TEST_F(DynamicShapesTest, EnvironmentBindAndResolve) {
    SymbolicShapeEnvironment env;
    env.bind("batch", 32);
    env.bind("seq_len", 128);

    EXPECT_TRUE(env.is_bound("batch"));
    EXPECT_TRUE(env.is_bound("seq_len"));
    EXPECT_FALSE(env.is_bound("hidden"));

    EXPECT_EQ(env.get("batch"), 32);
    EXPECT_EQ(env.get("seq_len"), 128);
}

TEST_F(DynamicShapesTest, EnvironmentResolveConcretePassthrough) {
    SymbolicShapeEnvironment env;
    auto dim = SymbolicDim::concrete(64);
    EXPECT_EQ(env.resolve(dim), 64);
}

TEST_F(DynamicShapesTest, EnvironmentResolveSymbolic) {
    SymbolicShapeEnvironment env;
    env.bind("batch", 32);

    auto dim = SymbolicDim::symbolic("batch");
    EXPECT_EQ(env.resolve(dim), 32);
}

TEST_F(DynamicShapesTest, EnvironmentResolveUnboundThrows) {
    SymbolicShapeEnvironment env;
    auto dim = SymbolicDim::symbolic("unbound");
    EXPECT_THROW(env.resolve(dim), std::runtime_error);
}

TEST_F(DynamicShapesTest, EnvironmentResolveShape) {
    SymbolicShapeEnvironment env;
    env.bind("batch", 16);

    SymbolicShape shape = {
        SymbolicDim::symbolic("batch"),
        SymbolicDim::concrete(3),
        SymbolicDim::concrete(224),
        SymbolicDim::concrete(224)
    };

    auto concrete = env.resolve(shape);
    EXPECT_EQ(concrete.size(), 4);
    EXPECT_EQ(concrete[0], 16);
    EXPECT_EQ(concrete[1], 3);
    EXPECT_EQ(concrete[2], 224);
    EXPECT_EQ(concrete[3], 224);
}

TEST_F(DynamicShapesTest, EnvironmentCanResolve) {
    SymbolicShapeEnvironment env;
    env.bind("batch", 32);

    SymbolicShape resolvable = {
        SymbolicDim::symbolic("batch"),
        SymbolicDim::concrete(64)
    };
    EXPECT_TRUE(env.can_resolve(resolvable));

    SymbolicShape unresolvable = {
        SymbolicDim::symbolic("batch"),
        SymbolicDim::symbolic("missing")
    };
    EXPECT_FALSE(env.can_resolve(unresolvable));
}

TEST_F(DynamicShapesTest, EnvironmentUnbind) {
    SymbolicShapeEnvironment env;
    env.bind("batch", 32);
    EXPECT_TRUE(env.is_bound("batch"));

    env.unbind("batch");
    EXPECT_FALSE(env.is_bound("batch"));
}

TEST_F(DynamicShapesTest, EnvironmentClear) {
    SymbolicShapeEnvironment env;
    env.bind("a", 1);
    env.bind("b", 2);
    EXPECT_EQ(env.size(), 2);

    env.clear();
    EXPECT_EQ(env.size(), 0);
}

TEST_F(DynamicShapesTest, RebindOverwrites) {
    SymbolicShapeEnvironment env;
    env.bind("batch", 32);
    EXPECT_EQ(env.get("batch"), 32);

    env.bind("batch", 64);
    EXPECT_EQ(env.get("batch"), 64);
}

// ============================================================================
// SymbolicExpr AST tests
// ============================================================================

TEST_F(DynamicShapesTest, ExprCreation) {
    auto batch = SymbolicDim::symbolic("batch");
    auto result = batch + SymbolicDim::concrete(32);
    EXPECT_FALSE(result.is_concrete());
    EXPECT_TRUE(result.is_symbolic());
    EXPECT_TRUE(result.is_expr());
    EXPECT_FALSE(result.is_named_symbol());
}

TEST_F(DynamicShapesTest, ExprToString) {
    auto batch = SymbolicDim::symbolic("batch");
    auto result = batch + SymbolicDim::concrete(32);
    EXPECT_EQ(result.to_string(), "(batch + 32)");

    auto nested = (batch + SymbolicDim::concrete(4)) - SymbolicDim::concrete(3);
    EXPECT_EQ(nested.to_string(), "((batch + 4) - 3)");
}

TEST_F(DynamicShapesTest, ExprEquality) {
    auto batch = SymbolicDim::symbolic("batch");
    auto a = batch + SymbolicDim::concrete(32);
    auto b = batch + SymbolicDim::concrete(32);
    auto c = batch + SymbolicDim::concrete(64);
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST_F(DynamicShapesTest, ConstantFolding) {
    auto a = SymbolicDim::concrete(3);
    auto b = SymbolicDim::concrete(4);
    auto result = a + b;
    EXPECT_TRUE(result.is_concrete());
    EXPECT_EQ(result.value(), 7);

    auto product = a * b;
    EXPECT_TRUE(product.is_concrete());
    EXPECT_EQ(product.value(), 12);
}

TEST_F(DynamicShapesTest, IdentityAdd) {
    auto batch = SymbolicDim::symbolic("batch");
    auto result = batch + SymbolicDim::concrete(0);
    // Should simplify to just 'batch', not an expression
    EXPECT_TRUE(result.is_named_symbol());
    EXPECT_EQ(result.name(), "batch");

    auto result2 = SymbolicDim::concrete(0) + batch;
    EXPECT_TRUE(result2.is_named_symbol());
    EXPECT_EQ(result2.name(), "batch");
}

TEST_F(DynamicShapesTest, IdentityMul) {
    auto batch = SymbolicDim::symbolic("batch");
    auto result = batch * SymbolicDim::concrete(1);
    EXPECT_TRUE(result.is_named_symbol());
    EXPECT_EQ(result.name(), "batch");

    auto result2 = SymbolicDim::concrete(1) * batch;
    EXPECT_TRUE(result2.is_named_symbol());
}

TEST_F(DynamicShapesTest, ZeroMul) {
    auto batch = SymbolicDim::symbolic("batch");
    auto result = batch * SymbolicDim::concrete(0);
    EXPECT_TRUE(result.is_concrete());
    EXPECT_EQ(result.value(), 0);
}

TEST_F(DynamicShapesTest, ExprResolve) {
    SymbolicShapeEnvironment env;
    env.bind("batch", 32);

    auto batch = SymbolicDim::symbolic("batch");
    auto expr = batch + SymbolicDim::concrete(10);
    EXPECT_EQ(env.resolve(expr), 42);
}

TEST_F(DynamicShapesTest, NestedExprResolve) {
    SymbolicShapeEnvironment env;
    env.bind("H", 224);

    // Simulate conv2d output: (H + 2*pad - kernel) / stride + 1
    // With pad=1, kernel=3, stride=1: (224 + 2 - 3) / 1 + 1 = 224
    auto H = SymbolicDim::symbolic("H");
    auto pad2 = SymbolicDim::concrete(2);
    auto kernel = SymbolicDim::concrete(3);
    auto stride = SymbolicDim::concrete(1);
    auto one = SymbolicDim::concrete(1);

    auto H_out = (H + pad2 - kernel) / stride + one;
    EXPECT_TRUE(H_out.is_expr());
    EXPECT_EQ(env.resolve(H_out), 224);

    // Different H value
    env.bind("H", 112);
    EXPECT_EQ(env.resolve(H_out), 112);
}

TEST_F(DynamicShapesTest, CanResolveExpr) {
    SymbolicShapeEnvironment env;
    env.bind("batch", 32);

    auto batch = SymbolicDim::symbolic("batch");
    auto expr = batch + SymbolicDim::concrete(10);
    EXPECT_TRUE(env.can_resolve_dim(expr));

    // Expression with unbound symbol
    auto seq = SymbolicDim::symbolic("seq_len");
    auto expr2 = batch * seq;
    EXPECT_FALSE(env.can_resolve_dim(expr2));

    // Bind the missing symbol
    env.bind("seq_len", 128);
    EXPECT_TRUE(env.can_resolve_dim(expr2));
    EXPECT_EQ(env.resolve(expr2), 32 * 128);
}

TEST_F(DynamicShapesTest, ExprNameThrows) {
    auto batch = SymbolicDim::symbolic("batch");
    auto expr = batch + SymbolicDim::concrete(32);
    EXPECT_THROW((void)expr.name(), std::runtime_error);
}

TEST_F(DynamicShapesTest, BackwardCompatStringConstruction) {
    // Old-style construction should still work identically
    SymbolicDim dim(std::string("batch"));
    EXPECT_TRUE(dim.is_symbolic());
    EXPECT_TRUE(dim.is_named_symbol());
    EXPECT_FALSE(dim.is_expr());
    EXPECT_EQ(dim.name(), "batch");
    EXPECT_EQ(dim.to_string(), "batch");
}
