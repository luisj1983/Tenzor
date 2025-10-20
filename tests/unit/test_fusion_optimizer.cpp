/**
 * @file test_fusion_optimizer.cpp
 * @brief Comprehensive unit tests for graph-level kernel fusion optimization
 */

#include <gtest/gtest.h>
#include "tenzor/ops/fusion_optimizer.hpp"
#include "tenzor/ops/creation.hpp"
#include <vector>
#include <memory>

using namespace tenzor;
using namespace tenzor::ops;

// ==============================================================================
// Test Fixtures
// ==============================================================================

class FusionGraphTest : public ::testing::Test {
protected:
    FusionGraph graph;

    void SetUp() override {
        graph = FusionGraph();
    }
};

class FusionOptimizerTest : public ::testing::Test {
protected:
    FusionOptimizer optimizer;

    void SetUp() override {
        optimizer = FusionOptimizer(false);  // Non-aggressive mode
    }
};

// ==============================================================================
// FusionGraph Tests
// ==============================================================================

TEST_F(FusionGraphTest, AddNode) {
    auto id = graph.add_node(OpType::Linear, "linear1");
    EXPECT_EQ(id, 0);
    EXPECT_EQ(graph.size(), 1);

    auto id2 = graph.add_node(OpType::ReLU, "relu1");
    EXPECT_EQ(id2, 1);
    EXPECT_EQ(graph.size(), 2);
}

TEST_F(FusionGraphTest, AddEdge) {
    auto id1 = graph.add_node(OpType::Linear, "linear1");
    auto id2 = graph.add_node(OpType::ReLU, "relu1");

    graph.add_edge(id1, id2);

    auto outputs = graph.get_outputs(id1);
    ASSERT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs[0], id2);

    auto inputs = graph.get_inputs(id2);
    ASSERT_EQ(inputs.size(), 1);
    EXPECT_EQ(inputs[0], id1);
}

TEST_F(FusionGraphTest, GetNode) {
    auto id = graph.add_node(OpType::MatMul, "matmul1");
    const auto& node = graph.get_node(id);

    EXPECT_EQ(node.id, id);
    EXPECT_EQ(node.op_type, OpType::MatMul);
    EXPECT_EQ(node.op_name, "matmul1");
    EXPECT_FALSE(node.is_fused);
}

TEST_F(FusionGraphTest, TopologicalSort) {
    // Create simple chain: linear -> relu -> add
    auto id1 = graph.add_node(OpType::Linear, "linear1");
    auto id2 = graph.add_node(OpType::ReLU, "relu1", {id1});
    auto id3 = graph.add_node(OpType::Add, "add1", {id2});

    auto sorted = graph.topological_sort();

    // Should be in reverse topological order (from output to input)
    ASSERT_EQ(sorted.size(), 3);
    // Note: topological sort returns nodes in reverse order
}

TEST_F(FusionGraphTest, NoCycle) {
    // Create acyclic graph
    auto id1 = graph.add_node(OpType::Linear, "linear1");
    auto id2 = graph.add_node(OpType::ReLU, "relu1", {id1});

    EXPECT_FALSE(graph.has_cycle());
}

TEST_F(FusionGraphTest, Clear) {
    graph.add_node(OpType::Linear, "linear1");
    graph.add_node(OpType::ReLU, "relu1");

    EXPECT_EQ(graph.size(), 2);

    graph.clear();

    EXPECT_EQ(graph.size(), 0);
}

TEST_F(FusionGraphTest, ToDot) {
    auto id1 = graph.add_node(OpType::Linear, "linear1");
    auto id2 = graph.add_node(OpType::ReLU, "relu1", {id1});

    auto dot = graph.to_dot();

    EXPECT_TRUE(dot.find("digraph") != std::string::npos);
    EXPECT_TRUE(dot.find("linear1") != std::string::npos);
    EXPECT_TRUE(dot.find("relu1") != std::string::npos);
}

// ==============================================================================
// FusionPattern Tests
// ==============================================================================

TEST(FusionPatternTest, LinearReLUPattern) {
    FusionPattern pattern("linear_relu");
    EXPECT_EQ(pattern.get_name(), "linear_relu");
    EXPECT_GT(pattern.get_speedup(), 1.0f);
}

TEST(FusionPatternTest, ConvBnReLUPattern) {
    FusionPattern pattern("conv_bn_relu");
    EXPECT_EQ(pattern.get_name(), "conv_bn_relu");
    EXPECT_GT(pattern.get_speedup(), 1.5f);
}

TEST(FusionPatternTest, MatchLinearReLU) {
    FusionGraph graph;
    auto id1 = graph.add_node(OpType::Linear, "linear1");
    auto id2 = graph.add_node(OpType::ReLU, "relu1", {id1});

    FusionPattern pattern("linear_relu");
    auto match = pattern.match(graph, id1);

    EXPECT_TRUE(match);
    EXPECT_EQ(match.matched_nodes.size(), 2);
    EXPECT_EQ(match.matched_nodes[0], id1);
    EXPECT_EQ(match.matched_nodes[1], id2);
    EXPECT_EQ(match.pattern_name, "linear_relu");
}

TEST(FusionPatternTest, NoMatchWrongOrder) {
    FusionGraph graph;
    auto id1 = graph.add_node(OpType::ReLU, "relu1");
    auto id2 = graph.add_node(OpType::Linear, "linear1", {id1});

    FusionPattern pattern("linear_relu");
    auto match = pattern.match(graph, id1);

    EXPECT_FALSE(match);  // Wrong order
}

TEST(FusionPatternTest, MatchConvBnReLU) {
    FusionGraph graph;
    auto id1 = graph.add_node(OpType::Conv2d, "conv1");
    auto id2 = graph.add_node(OpType::BatchNorm2d, "bn1", {id1});
    auto id3 = graph.add_node(OpType::ReLU, "relu1", {id2});

    FusionPattern pattern("conv_bn_relu");
    auto match = pattern.match(graph, id1);

    EXPECT_TRUE(match);
    EXPECT_EQ(match.matched_nodes.size(), 3);
    EXPECT_EQ(match.pattern_name, "conv_bn_relu");
}

TEST(FusionPatternTest, MatchMatMulAdd) {
    FusionGraph graph;
    auto id1 = graph.add_node(OpType::MatMul, "matmul1");
    auto id2 = graph.add_node(OpType::Add, "add1", {id1});

    FusionPattern pattern("matmul_add");
    auto match = pattern.match(graph, id1);

    EXPECT_TRUE(match);
    EXPECT_EQ(match.matched_nodes.size(), 2);
    EXPECT_GT(match.confidence, 0.8f);
}

TEST(FusionPatternTest, MatchElementwiseChain) {
    FusionGraph graph;
    auto id1 = graph.add_node(OpType::Add, "add1");
    auto id2 = graph.add_node(OpType::Mul, "mul1", {id1});
    auto id3 = graph.add_node(OpType::ReLU, "relu1", {id2});

    FusionPattern pattern("elementwise_chain");
    auto match = pattern.match(graph, id1);

    EXPECT_TRUE(match);
    EXPECT_GE(match.matched_nodes.size(), 3);
}

// ==============================================================================
// FusionOptimizer Tests
// ==============================================================================

TEST_F(FusionOptimizerTest, AddPattern) {
    EXPECT_TRUE(optimizer.add_pattern("linear_relu"));
    EXPECT_FALSE(optimizer.add_pattern("linear_relu"));  // Already added
    EXPECT_FALSE(optimizer.add_pattern("invalid_pattern"));
}

TEST_F(FusionOptimizerTest, RemovePattern) {
    optimizer.add_pattern("linear_relu");
    EXPECT_TRUE(optimizer.remove_pattern("linear_relu"));
    EXPECT_FALSE(optimizer.remove_pattern("linear_relu"));  // Already removed
}

TEST_F(FusionOptimizerTest, AddAllPatterns) {
    EXPECT_TRUE(optimizer.add_pattern("all"));

    auto patterns = optimizer.get_supported_patterns();
    for (const auto& pattern : patterns) {
        EXPECT_TRUE(optimizer.is_pattern_supported(pattern));
    }
}

TEST_F(FusionOptimizerTest, OptimizeLinearReLU) {
    optimizer.add_pattern("linear_relu");

    // Create graph with Linear -> ReLU pattern
    FusionGraph graph;
    auto id1 = graph.add_node(OpType::Linear, "linear1");
    auto id2 = graph.add_node(OpType::ReLU, "relu1", {id1});

    auto optimized = optimizer.optimize(graph);

    // Should have one fused node instead of two
    EXPECT_LT(optimized.size(), graph.size());

    const auto& stats = optimizer.get_statistics();
    EXPECT_EQ(stats.num_nodes_original, 2);
    EXPECT_EQ(stats.num_fusions, 1);
    EXPECT_GT(stats.expected_speedup, 1.0f);
    EXPECT_GT(stats.num_kernel_launches_saved, 0);
}

TEST_F(FusionOptimizerTest, OptimizeMultiplePatterns) {
    optimizer.add_pattern("linear_relu");
    optimizer.add_pattern("matmul_add");

    // Create graph with multiple fusion opportunities
    FusionGraph graph;
    auto id1 = graph.add_node(OpType::Linear, "linear1");
    auto id2 = graph.add_node(OpType::ReLU, "relu1", {id1});
    auto id3 = graph.add_node(OpType::MatMul, "matmul1", {id2});
    auto id4 = graph.add_node(OpType::Add, "add1", {id3});

    auto optimized = optimizer.optimize(graph);

    const auto& stats = optimizer.get_statistics();
    EXPECT_EQ(stats.num_fusions, 2);  // Two patterns should be fused
    EXPECT_EQ(stats.pattern_counts["linear_relu"], 1);
    EXPECT_EQ(stats.pattern_counts["matmul_add"], 1);
}

TEST_F(FusionOptimizerTest, NoOptimizationOpportunities) {
    optimizer.add_pattern("linear_relu");

    // Create graph with no fusion patterns
    FusionGraph graph;
    auto id1 = graph.add_node(OpType::Conv2d, "conv1");
    auto id2 = graph.add_node(OpType::BatchNorm2d, "bn1", {id1});

    auto optimized = optimizer.optimize(graph);

    const auto& stats = optimizer.get_statistics();
    EXPECT_EQ(stats.num_fusions, 0);
    EXPECT_EQ(optimized.size(), graph.size());
}

TEST_F(FusionOptimizerTest, Statistics) {
    optimizer.add_pattern("linear_relu");

    FusionGraph graph;
    graph.add_node(OpType::Linear, "linear1");
    graph.add_node(OpType::ReLU, "relu1", {0});

    optimizer.optimize(graph);

    const auto& stats = optimizer.get_statistics();
    EXPECT_GT(stats.num_nodes_original, 0);
    EXPECT_GT(stats.num_nodes_optimized, 0);
    EXPECT_GE(stats.num_kernel_launches_saved, 1);
    EXPECT_GT(stats.memory_bandwidth_reduction, 0.0f);
}

TEST_F(FusionOptimizerTest, ResetStatistics) {
    optimizer.add_pattern("linear_relu");

    FusionGraph graph;
    graph.add_node(OpType::Linear, "linear1");
    graph.add_node(OpType::ReLU, "relu1", {0});

    optimizer.optimize(graph);

    EXPECT_GT(optimizer.get_statistics().num_fusions, 0);

    optimizer.reset_statistics();

    const auto& stats = optimizer.get_statistics();
    EXPECT_EQ(stats.num_fusions, 0);
    EXPECT_EQ(stats.num_kernel_launches_saved, 0);
}

TEST_F(FusionOptimizerTest, AggressiveMode) {
    optimizer.set_aggressive_mode(true);
    optimizer.add_pattern("elementwise_chain");

    // Aggressive mode might fuse more aggressively
    FusionGraph graph;
    for (int i = 0; i < 6; ++i) {
        graph.add_node(OpType::Add, "add" + std::to_string(i), i > 0 ? std::vector<size_t>{size_t(i-1)} : std::vector<size_t>{});
    }

    auto optimized = optimizer.optimize(graph);

    const auto& stats = optimizer.get_statistics();
    EXPECT_GT(stats.num_fusions, 0);
}

TEST_F(FusionOptimizerTest, BackendSelection) {
    optimizer.set_backend("cuda", true);
    optimizer.set_backend("rocm", false);
    optimizer.set_backend("cpu", true);

    // Backend settings shouldn't affect pattern detection
    optimizer.add_pattern("linear_relu");

    FusionGraph graph;
    graph.add_node(OpType::Linear, "linear1");
    graph.add_node(OpType::ReLU, "relu1", {0});

    auto optimized = optimizer.optimize(graph);

    EXPECT_GT(optimizer.get_statistics().num_fusions, 0);
}

TEST_F(FusionOptimizerTest, SupportedPatterns) {
    auto patterns = optimizer.get_supported_patterns();

    EXPECT_FALSE(patterns.empty());
    EXPECT_TRUE(optimizer.is_pattern_supported("linear_relu"));
    EXPECT_TRUE(optimizer.is_pattern_supported("conv_bn_relu"));
    EXPECT_TRUE(optimizer.is_pattern_supported("matmul_add"));
    EXPECT_TRUE(optimizer.is_pattern_supported("elementwise_chain"));
    EXPECT_TRUE(optimizer.is_pattern_supported("attention"));
}

// ==============================================================================
// Helper Function Tests
// ==============================================================================

TEST(HelperFunctionTest, StringToOpType) {
    EXPECT_EQ(string_to_op_type("matmul"), OpType::MatMul);
    EXPECT_EQ(string_to_op_type("linear"), OpType::Linear);
    EXPECT_EQ(string_to_op_type("conv2d"), OpType::Conv2d);
    EXPECT_EQ(string_to_op_type("relu"), OpType::ReLU);
    EXPECT_EQ(string_to_op_type("gelu"), OpType::GELU);
    EXPECT_EQ(string_to_op_type("batchnorm2d"), OpType::BatchNorm2d);
    EXPECT_EQ(string_to_op_type("add"), OpType::Add);
    EXPECT_EQ(string_to_op_type("unknown_op"), OpType::Unknown);
}

TEST(HelperFunctionTest, OpTypeToString) {
    EXPECT_EQ(op_type_to_string(OpType::MatMul), "matmul");
    EXPECT_EQ(op_type_to_string(OpType::Linear), "linear");
    EXPECT_EQ(op_type_to_string(OpType::Conv2d), "conv2d");
    EXPECT_EQ(op_type_to_string(OpType::ReLU), "relu");
    EXPECT_EQ(op_type_to_string(OpType::GELU), "gelu");
    EXPECT_EQ(op_type_to_string(OpType::BatchNorm2d), "batchnorm2d");
    EXPECT_EQ(op_type_to_string(OpType::Add), "add");
    EXPECT_EQ(op_type_to_string(OpType::Unknown), "unknown");
}

TEST(HelperFunctionTest, RoundTripConversion) {
    std::vector<std::string> op_names = {
        "matmul", "linear", "conv2d", "relu", "gelu",
        "batchnorm2d", "layernorm", "add", "mul", "softmax"
    };

    for (const auto& name : op_names) {
        auto op_type = string_to_op_type(name);
        auto converted_back = op_type_to_string(op_type);
        EXPECT_EQ(name, converted_back);
    }
}

// ==============================================================================
// Complex Fusion Scenarios
// ==============================================================================

TEST(ComplexFusionTest, NestedPatterns) {
    FusionOptimizer optimizer;
    optimizer.add_pattern("all");

    // Create complex graph with nested patterns
    FusionGraph graph;
    auto id1 = graph.add_node(OpType::Linear, "linear1");
    auto id2 = graph.add_node(OpType::ReLU, "relu1", {id1});
    auto id3 = graph.add_node(OpType::MatMul, "matmul1", {id2});
    auto id4 = graph.add_node(OpType::Add, "add1", {id3});
    auto id5 = graph.add_node(OpType::ReLU, "relu2", {id4});

    auto optimized = optimizer.optimize(graph);

    const auto& stats = optimizer.get_statistics();
    EXPECT_GT(stats.num_fusions, 0);
    EXPECT_LT(optimized.size(), graph.size());
}

TEST(ComplexFusionTest, MultipleDisjointPatterns) {
    FusionOptimizer optimizer;
    optimizer.add_pattern("all");

    // Create graph with multiple independent patterns
    FusionGraph graph;

    // Pattern 1: Linear -> ReLU
    auto a1 = graph.add_node(OpType::Linear, "linear1");
    auto a2 = graph.add_node(OpType::ReLU, "relu1", {a1});

    // Pattern 2: MatMul -> Add (independent)
    auto b1 = graph.add_node(OpType::MatMul, "matmul1");
    auto b2 = graph.add_node(OpType::Add, "add1", {b1});

    // Pattern 3: Conv -> BN -> ReLU (independent)
    auto c1 = graph.add_node(OpType::Conv2d, "conv1");
    auto c2 = graph.add_node(OpType::BatchNorm2d, "bn1", {c1});
    auto c3 = graph.add_node(OpType::ReLU, "relu2", {c2});

    auto optimized = optimizer.optimize(graph);

    const auto& stats = optimizer.get_statistics();
    EXPECT_EQ(stats.num_fusions, 3);  // All three patterns should be detected
}

TEST(ComplexFusionTest, LongElementwiseChain) {
    FusionOptimizer optimizer;
    optimizer.add_pattern("elementwise_chain");

    // Create long chain of element-wise operations
    FusionGraph graph;
    size_t prev = graph.add_node(OpType::Add, "add0");

    for (int i = 1; i < 10; ++i) {
        OpType type = (i % 3 == 0) ? OpType::Add :
                     (i % 3 == 1) ? OpType::Mul : OpType::ReLU;
        prev = graph.add_node(type, "op" + std::to_string(i), {prev});
    }

    auto optimized = optimizer.optimize(graph);

    const auto& stats = optimizer.get_statistics();
    EXPECT_GT(stats.num_fusions, 0);
}

// ==============================================================================
// Error Handling Tests
// ==============================================================================

TEST(ErrorHandlingTest, InvalidNodeAccess) {
    FusionGraph graph;
    EXPECT_THROW(graph.get_node(999), std::out_of_range);
}

TEST(ErrorHandlingTest, EmptyGraphOptimization) {
    FusionOptimizer optimizer;
    optimizer.add_pattern("linear_relu");

    FusionGraph graph;  // Empty graph
    auto optimized = optimizer.optimize(graph);

    EXPECT_EQ(optimized.size(), 0);
    EXPECT_EQ(optimizer.get_statistics().num_fusions, 0);
}

// ==============================================================================
// Performance Characteristics Tests
// ==============================================================================

TEST(PerformanceTest, SpeedupEstimation) {
    FusionOptimizer optimizer;
    optimizer.add_pattern("linear_relu");
    optimizer.add_pattern("conv_bn_relu");

    // Linear + ReLU should have lower speedup than Conv + BN + ReLU
    FusionGraph graph1;
    graph1.add_node(OpType::Linear, "linear1");
    graph1.add_node(OpType::ReLU, "relu1", {0});

    FusionGraph graph2;
    graph2.add_node(OpType::Conv2d, "conv1");
    graph2.add_node(OpType::BatchNorm2d, "bn1", {0});
    graph2.add_node(OpType::ReLU, "relu1", {1});

    optimizer.optimize(graph1);
    float speedup1 = optimizer.get_statistics().expected_speedup;

    optimizer.reset_statistics();

    optimizer.optimize(graph2);
    float speedup2 = optimizer.get_statistics().expected_speedup;

    EXPECT_GT(speedup2, speedup1);  // More complex fusion should have higher speedup
}

TEST(PerformanceTest, MemoryBandwidthReduction) {
    FusionOptimizer optimizer;
    optimizer.add_pattern("all");

    FusionGraph graph;
    for (int i = 0; i < 5; ++i) {
        graph.add_node(OpType::Add, "add" + std::to_string(i),
                      i > 0 ? std::vector<size_t>{size_t(i-1)} : std::vector<size_t>{});
    }

    optimizer.optimize(graph);

    const auto& stats = optimizer.get_statistics();
    EXPECT_GT(stats.memory_bandwidth_reduction, 0.0f);
    EXPECT_LE(stats.memory_bandwidth_reduction, 1.0f);  // Should be a ratio
}

// ==============================================================================
// Main
// ==============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
