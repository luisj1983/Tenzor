/**
 * @file test_fusion_optimizer_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for graph-level kernel fusion optimization
 *
 * Note: FusionOptimizer operates on graph representations, not tensors directly.
 * These tests verify that the graph optimization infrastructure works correctly
 * across different backends and dtypes by testing the graph construction,
 * pattern matching, and optimization pipeline.
 */

#include "../multi_backend_dtype_fixture.hpp"
#include "tenzor/ops/fusion_optimizer.hpp"

using namespace tenzor;
using namespace tenzor::ops;
using namespace tenzor::testing;

// audit-3 T.1 rationale — FusionOptimizer/FusionGraph operate on the symbolic
// computation graph (OpType + dependency edges), not on tensor values. Adding
// expectFiniteNonZero would be vacuous: the graph contains no tensor outputs
// to validate. Shape/structural assertions are the natural correctness bar.
class FusionOptimizerMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    FusionOptimizer optimizer_;
    FusionGraph graph_;
};

TEST_P(FusionOptimizerMultiDTypeTest, EmptyGraphHasZeroNodes) {
    EXPECT_EQ(graph_.size(), 0u);
    EXPECT_FALSE(graph_.has_cycle());
}

TEST_P(FusionOptimizerMultiDTypeTest, AddSingleNode) {
    auto id = graph_.add_node(OpType::MatMul, "matmul1");
    EXPECT_EQ(graph_.size(), 1u);
    const auto& node = graph_.get_node(id);
    EXPECT_EQ(node.op_type, OpType::MatMul);
}

TEST_P(FusionOptimizerMultiDTypeTest, TopologicalSortLinearChain) {
    auto a = graph_.add_node(OpType::MatMul, "a");
    auto b = graph_.add_node(OpType::ReLU, "b", {a});
    auto c = graph_.add_node(OpType::Add, "c", {b});
    auto sorted = graph_.topological_sort();
    ASSERT_EQ(sorted.size(), 3u);
    auto pos_a = std::find(sorted.begin(), sorted.end(), a);
    auto pos_b = std::find(sorted.begin(), sorted.end(), b);
    auto pos_c = std::find(sorted.begin(), sorted.end(), c);
    EXPECT_LT(pos_a, pos_b);
    EXPECT_LT(pos_b, pos_c);
}

TEST_P(FusionOptimizerMultiDTypeTest, OptimizeLinearReLUPattern) {
    optimizer_.add_pattern("linear_relu");
    auto linear = graph_.add_node(OpType::Linear, "fc");
    graph_.add_node(OpType::ReLU, "relu", {linear});

    auto optimized = optimizer_.optimize(graph_);
    const auto& stats = optimizer_.get_statistics();

    EXPECT_GE(stats.num_nodes_original, 2u);
    EXPECT_LE(optimized.size(), graph_.size());
}

TEST_P(FusionOptimizerMultiDTypeTest, AddPatternSucceeds) {
    EXPECT_TRUE(optimizer_.add_pattern("linear_relu"));
    EXPECT_TRUE(optimizer_.add_pattern("conv_bn_relu"));
    EXPECT_TRUE(optimizer_.add_pattern("matmul_add"));
}

TEST_P(FusionOptimizerMultiDTypeTest, AddPatternFailsForUnknown) {
    EXPECT_FALSE(optimizer_.add_pattern("nonexistent_pattern_xyz"));
}

TEST_P(FusionOptimizerMultiDTypeTest, StatisticsResetWorks) {
    optimizer_.add_pattern("linear_relu");
    auto linear = graph_.add_node(OpType::Linear, "fc");
    graph_.add_node(OpType::ReLU, "relu", {linear});
    optimizer_.optimize(graph_);
    optimizer_.reset_statistics();
    const auto& stats = optimizer_.get_statistics();
    EXPECT_EQ(stats.num_fusions, 0u);
}

TEST_P(FusionOptimizerMultiDTypeTest, ClearResetsGraph) {
    graph_.add_node(OpType::MatMul, "a");
    graph_.add_node(OpType::ReLU, "b");
    EXPECT_EQ(graph_.size(), 2u);
    graph_.clear();
    EXPECT_EQ(graph_.size(), 0u);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(FusionOptimizerMultiDTypeTest);
