/**
 * @file test_fusion_optimizer.cpp
 * @brief Tests for graph-level kernel fusion optimization
 *
 * Recreated after the original orphan was deleted in test suite cleanup.
 * Verifies that FusionOptimizer can build a FusionGraph, detect fusion
 * patterns, and apply graph rewrites.
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/ops/fusion_optimizer.hpp"

using namespace tenzor;
using namespace tenzor::ops;

class FusionGraphTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }

    FusionGraph graph_;
};

class FusionOptimizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }

    FusionOptimizer optimizer_;
    FusionGraph graph_;
};

// ============================================================================
// FusionGraph: basic graph construction
// ============================================================================

TEST_F(FusionGraphTest, EmptyGraphHasZeroNodes) {
    EXPECT_EQ(graph_.size(), 0u);
    EXPECT_FALSE(graph_.has_cycle());
}

TEST_F(FusionGraphTest, AddSingleNode) {
    auto id = graph_.add_node(OpType::MatMul, "matmul1");
    EXPECT_EQ(graph_.size(), 1u);
    const auto& node = graph_.get_node(id);
    EXPECT_EQ(node.op_type, OpType::MatMul);
    EXPECT_EQ(node.op_name, "matmul1");
}

TEST_F(FusionGraphTest, AddNodeWithInputs) {
    auto a = graph_.add_node(OpType::MatMul, "matmul");
    auto b = graph_.add_node(OpType::Add, "add", {a});
    EXPECT_EQ(graph_.size(), 2u);
    auto inputs = graph_.get_inputs(b);
    ASSERT_EQ(inputs.size(), 1u);
    EXPECT_EQ(inputs[0], a);
}

TEST_F(FusionGraphTest, AddEdgeAndQueryOutputs) {
    auto a = graph_.add_node(OpType::MatMul, "a");
    auto b = graph_.add_node(OpType::ReLU, "b");
    graph_.add_edge(a, b);
    auto outputs = graph_.get_outputs(a);
    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(outputs[0], b);
}

TEST_F(FusionGraphTest, NodeAttributesAreStored) {
    std::unordered_map<std::string, std::string> attrs = {
        {"kernel_size", "3"},
        {"stride", "1"}
    };
    auto id = graph_.add_node(OpType::Conv2d, "conv1", {}, attrs);
    const auto& node = graph_.get_node(id);
    EXPECT_EQ(node.attributes.at("kernel_size"), "3");
    EXPECT_EQ(node.attributes.at("stride"), "1");
}

TEST_F(FusionGraphTest, TopologicalSortLinearChain) {
    auto a = graph_.add_node(OpType::MatMul, "a");
    auto b = graph_.add_node(OpType::ReLU, "b", {a});
    auto c = graph_.add_node(OpType::Add, "c", {b});
    auto sorted = graph_.topological_sort();
    ASSERT_EQ(sorted.size(), 3u);
    // Each node must come before its consumers
    auto pos_a = std::find(sorted.begin(), sorted.end(), a);
    auto pos_b = std::find(sorted.begin(), sorted.end(), b);
    auto pos_c = std::find(sorted.begin(), sorted.end(), c);
    EXPECT_LT(pos_a, pos_b);
    EXPECT_LT(pos_b, pos_c);
}

TEST_F(FusionGraphTest, NoCycleInDAG) {
    auto a = graph_.add_node(OpType::MatMul, "a");
    auto b = graph_.add_node(OpType::ReLU, "b", {a});
    graph_.add_node(OpType::Add, "c", {b});
    EXPECT_FALSE(graph_.has_cycle());
}

TEST_F(FusionGraphTest, GetNodeOutOfRangeThrows) {
    EXPECT_THROW(graph_.get_node(99999), std::out_of_range);
}

TEST_F(FusionGraphTest, ClearResetsGraph) {
    graph_.add_node(OpType::MatMul, "a");
    graph_.add_node(OpType::ReLU, "b");
    EXPECT_EQ(graph_.size(), 2u);
    graph_.clear();
    EXPECT_EQ(graph_.size(), 0u);
}

TEST_F(FusionGraphTest, ToDotProducesNonEmptyString) {
    graph_.add_node(OpType::Linear, "fc1");
    graph_.add_node(OpType::ReLU, "act1");
    auto dot = graph_.to_dot();
    EXPECT_FALSE(dot.empty());
}

// ============================================================================
// FusionOptimizer: pattern management
// ============================================================================

TEST_F(FusionOptimizerTest, AddPatternSucceedsForSupportedNames) {
    EXPECT_TRUE(optimizer_.add_pattern("linear_relu"));
    EXPECT_TRUE(optimizer_.add_pattern("conv_bn_relu"));
    EXPECT_TRUE(optimizer_.add_pattern("matmul_add"));
}

TEST_F(FusionOptimizerTest, AddPatternFailsForUnknownName) {
    EXPECT_FALSE(optimizer_.add_pattern("nonexistent_pattern_xyz"));
}

TEST_F(FusionOptimizerTest, RemovePatternWorks) {
    optimizer_.add_pattern("linear_relu");
    EXPECT_TRUE(optimizer_.remove_pattern("linear_relu"));
}

TEST_F(FusionOptimizerTest, IsPatternSupported) {
    EXPECT_TRUE(optimizer_.is_pattern_supported("linear_relu"));
    EXPECT_TRUE(optimizer_.is_pattern_supported("conv_bn_relu"));
    EXPECT_FALSE(optimizer_.is_pattern_supported("does_not_exist"));
}

TEST_F(FusionOptimizerTest, GetSupportedPatternsNonEmpty) {
    auto patterns = optimizer_.get_supported_patterns();
    EXPECT_FALSE(patterns.empty());
}

// ============================================================================
// FusionOptimizer: optimization
// ============================================================================

TEST_F(FusionOptimizerTest, OptimizeEmptyGraphReturnsEmpty) {
    optimizer_.add_pattern("linear_relu");
    auto optimized = optimizer_.optimize(graph_);
    EXPECT_EQ(optimized.size(), 0u);
}

TEST_F(FusionOptimizerTest, OptimizeLinearReLUPattern) {
    optimizer_.add_pattern("linear_relu");
    auto linear = graph_.add_node(OpType::Linear, "fc");
    graph_.add_node(OpType::ReLU, "relu", {linear});

    auto optimized = optimizer_.optimize(graph_);
    const auto& stats = optimizer_.get_statistics();

    // The optimizer should have observed the Linear+ReLU pattern.
    EXPECT_GE(stats.num_nodes_original, 2u);
    // Optimized graph should have fewer or equal nodes
    EXPECT_LE(optimized.size(), graph_.size());
}

TEST_F(FusionOptimizerTest, StatisticsResetWorks) {
    optimizer_.add_pattern("linear_relu");
    auto linear = graph_.add_node(OpType::Linear, "fc");
    graph_.add_node(OpType::ReLU, "relu", {linear});
    optimizer_.optimize(graph_);
    optimizer_.reset_statistics();
    const auto& stats = optimizer_.get_statistics();
    EXPECT_EQ(stats.num_fusions, 0u);
}

TEST_F(FusionOptimizerTest, AggressiveModeToggle) {
    EXPECT_NO_THROW(optimizer_.set_aggressive_mode(true));
    EXPECT_NO_THROW(optimizer_.set_aggressive_mode(false));
}

// ============================================================================
// Helper functions
// ============================================================================

TEST(FusionUtilsTest, StringToOpTypeKnownValues) {
    EXPECT_EQ(string_to_op_type("matmul"), OpType::MatMul);
    EXPECT_EQ(string_to_op_type("relu"), OpType::ReLU);
    EXPECT_EQ(string_to_op_type("conv2d"), OpType::Conv2d);
}

TEST(FusionUtilsTest, OpTypeToStringRoundTrip) {
    EXPECT_EQ(op_type_to_string(OpType::MatMul), "matmul");
    EXPECT_EQ(op_type_to_string(OpType::ReLU), "relu");
}
