/**
 * @file test_jit_lowlevel.cpp
 * @brief Tests for the low-level JIT Graph and Tracer APIs
 *
 * Recreated after the original orphan test (tests/test_jit.cpp, 70 tests)
 * was deleted in test suite cleanup. The original had rotted APIs (Tensor::full
 * static methods, DeviceType enum) — this rewrite uses the current public API.
 *
 * Complements tests/unit/test_jit.cpp which tests the high-level tracing/
 * compilation/serialization layer (most of which is currently #if 0'd out
 * pending JIT API completion).
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/jit/graph.hpp"
#include "tenzor/jit/tracer.hpp"

using namespace tenzor;
using namespace tenzor::jit;

class JITGraphTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }

    Graph graph_;
};

class JITTracerTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }
};

// ============================================================================
// Graph: Value creation and lookup
// ============================================================================

TEST_F(JITGraphTest, EmptyGraphHasNoNodesOrValues) {
    EXPECT_TRUE(graph_.nodes().empty());
    EXPECT_TRUE(graph_.inputs().empty());
    EXPECT_TRUE(graph_.outputs().empty());
}

TEST_F(JITGraphTest, CreateValueAndLookup) {
    auto val = graph_.create_value("v1", {2, 3}, DType::Float32, Device::cpu());
    ASSERT_NE(val, nullptr);
    auto fetched = graph_.get_value("v1");
    EXPECT_EQ(val, fetched);
}

TEST_F(JITGraphTest, GetNonexistentValueReturnsNull) {
    EXPECT_EQ(graph_.get_value("nonexistent"), nullptr);
}

TEST_F(JITGraphTest, MultipleValuesAreDistinct) {
    auto a = graph_.create_value("a", {2, 3}, DType::Float32, Device::cpu());
    auto b = graph_.create_value("b", {2, 3}, DType::Float32, Device::cpu());
    EXPECT_NE(a, b);
    EXPECT_EQ(graph_.get_value("a"), a);
    EXPECT_EQ(graph_.get_value("b"), b);
}

// ============================================================================
// Graph: Node creation and wiring
// ============================================================================

TEST_F(JITGraphTest, CreateNodeWithName) {
    auto node = graph_.create_node(OpType::MatMul, "matmul1");
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->op_type(), OpType::MatMul);
    EXPECT_EQ(node->name(), "matmul1");
}

TEST_F(JITGraphTest, CreateNodeAutoName) {
    auto node = graph_.create_node(OpType::Add);
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->op_type(), OpType::Add);
    // Auto-generated name should not be empty
    EXPECT_FALSE(node->name().empty());
}

TEST_F(JITGraphTest, AddNodeWithInputsAndOutputs) {
    auto in1 = graph_.create_value("in1", {4, 8}, DType::Float32, Device::cpu());
    auto in2 = graph_.create_value("in2", {8, 4}, DType::Float32, Device::cpu());
    auto out = graph_.create_value("out", {4, 4}, DType::Float32, Device::cpu());

    auto node = graph_.create_node(OpType::MatMul, "mm");
    node->add_input(in1);
    node->add_input(in2);
    node->add_output(out);
    graph_.add_node(node);

    EXPECT_EQ(node->inputs().size(), 2u);
    EXPECT_EQ(node->outputs().size(), 1u);
    EXPECT_EQ(node->inputs()[0], in1);
    EXPECT_EQ(node->inputs()[1], in2);
    EXPECT_EQ(node->outputs()[0], out);
    EXPECT_EQ(graph_.nodes().size(), 1u);
}

TEST_F(JITGraphTest, ReplaceNodeInput) {
    auto in1 = graph_.create_value("in1", {4}, DType::Float32, Device::cpu());
    auto in2 = graph_.create_value("in2", {4}, DType::Float32, Device::cpu());
    auto in3 = graph_.create_value("in3", {4}, DType::Float32, Device::cpu());

    auto node = graph_.create_node(OpType::Add, "add");
    node->add_input(in1);
    node->add_input(in2);
    graph_.add_node(node);

    node->replace_input(1, in3);
    EXPECT_EQ(node->inputs()[0], in1);
    EXPECT_EQ(node->inputs()[1], in3);
}

// ============================================================================
// Graph: Node attributes
// ============================================================================

TEST_F(JITGraphTest, NodeIntAttribute) {
    auto node = graph_.create_node(OpType::Conv2d, "conv");
    node->set_int_attr("stride", 2);
    node->set_int_attr("padding", 1);
    EXPECT_EQ(node->get_int_attr("stride"), 2);
    EXPECT_EQ(node->get_int_attr("padding"), 1);
}

TEST_F(JITGraphTest, NodeFloatAttribute) {
    auto node = graph_.create_node(OpType::Dropout, "dropout");
    node->set_attr("p", 0.5f);
    EXPECT_FLOAT_EQ(node->get_attr("p"), 0.5f);
}

TEST_F(JITGraphTest, NodeBoolAttribute) {
    auto node = graph_.create_node(OpType::BatchNorm2d, "bn");
    node->set_bool_attr("training", true);
    EXPECT_TRUE(node->get_bool_attr("training"));
}

TEST_F(JITGraphTest, NodeVectorAttribute) {
    auto node = graph_.create_node(OpType::Reshape, "reshape");
    node->set_vec_attr("shape", {2, 3, 4});
    auto vec = node->get_vec_attr("shape");
    ASSERT_EQ(vec.size(), 3u);
    EXPECT_EQ(vec[0], 2);
    EXPECT_EQ(vec[1], 3);
    EXPECT_EQ(vec[2], 4);
}

TEST_F(JITGraphTest, HasAttrReturnsCorrectly) {
    auto node = graph_.create_node(OpType::Linear, "fc");
    node->set_int_attr("out_features", 64);
    EXPECT_TRUE(node->has_attr("out_features"));
    EXPECT_FALSE(node->has_attr("nonexistent"));
}

// ============================================================================
// Graph: Inputs/outputs
// ============================================================================

TEST_F(JITGraphTest, SetInputsAndOutputs) {
    auto i1 = graph_.create_value("input", {1, 3, 224, 224}, DType::Float32, Device::cpu());
    auto o1 = graph_.create_value("output", {1, 1000}, DType::Float32, Device::cpu());
    graph_.set_inputs({i1});
    graph_.set_outputs({o1});
    EXPECT_EQ(graph_.inputs().size(), 1u);
    EXPECT_EQ(graph_.outputs().size(), 1u);
    EXPECT_EQ(graph_.inputs()[0], i1);
    EXPECT_EQ(graph_.outputs()[0], o1);
}

// ============================================================================
// Graph: Topological sort
// ============================================================================

TEST_F(JITGraphTest, TopologicalSortLinearChain) {
    // Build: in -> add -> mul -> out
    auto in_val = graph_.create_value("in", {4}, DType::Float32, Device::cpu());
    auto mid_val = graph_.create_value("mid", {4}, DType::Float32, Device::cpu());
    auto out_val = graph_.create_value("out", {4}, DType::Float32, Device::cpu());

    auto add = graph_.create_node(OpType::Add, "add");
    add->add_input(in_val);
    add->add_output(mid_val);
    graph_.add_node(add);

    auto mul = graph_.create_node(OpType::Mul, "mul");
    mul->add_input(mid_val);
    mul->add_output(out_val);
    graph_.add_node(mul);

    EXPECT_NO_THROW(graph_.topological_sort());
    EXPECT_EQ(graph_.nodes().size(), 2u);
}

// ============================================================================
// Graph: Remove node
// ============================================================================

TEST_F(JITGraphTest, RemoveNodeReducesNodeCount) {
    auto v = graph_.create_value("v", {4}, DType::Float32, Device::cpu());
    auto node = graph_.create_node(OpType::ReLU, "relu");
    node->add_input(v);
    graph_.add_node(node);
    EXPECT_EQ(graph_.nodes().size(), 1u);

    graph_.remove_node(node);
    EXPECT_EQ(graph_.nodes().size(), 0u);
}

// ============================================================================
// Tracer: Basic operations
// ============================================================================

TEST_F(JITTracerTest, TracerInstanceAccessible) {
    auto& tracer = Tracer::get_instance();
    // Should not be tracing initially
    EXPECT_FALSE(tracer.is_tracing());
}

TEST_F(JITTracerTest, StartAndEndTraceLifecycle) {
    auto& tracer = Tracer::get_instance();
    tracer.clear();
    EXPECT_FALSE(tracer.is_tracing());

    tracer.start_trace();
    EXPECT_TRUE(tracer.is_tracing());

    // End trace with no recorded ops should produce a valid (empty) graph
    auto graph = tracer.end_trace({}, {});
    EXPECT_NE(graph, nullptr);
    EXPECT_FALSE(tracer.is_tracing());
}

TEST_F(JITTracerTest, ClearResetsRecordedOpsAndStopsTracing) {
    auto& tracer = Tracer::get_instance();
    // clear() is the tracer's full reset — it purges recorded state AND
    // stops tracing, so back-to-back TracingGuards don't bleed state.
    tracer.start_trace();
    EXPECT_TRUE(tracer.is_tracing());
    tracer.clear();
    EXPECT_FALSE(tracer.is_tracing());
}

TEST_F(JITTracerTest, RegisterTensorReturnsId) {
    auto& tracer = Tracer::get_instance();
    tracer.clear();
    tracer.start_trace();

    auto t = tenzor::randn({4}, DType::Float32, Device::cpu());
    auto id = tracer.register_tensor(t);
    EXPECT_FALSE(id.empty());

    tracer.end_trace({}, {});
}
