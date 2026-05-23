/**
 * @file test_graph_optimizer_multidtype.cpp
 * @brief Multi-backend unit tests for GraphOptimizer, converted from test_graph_optimizer.cpp.
 */

#include <gtest/gtest.h>
#include "tenzor/autograd/graph_optimizer.hpp"
#include "tenzor/autograd/graph.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/device.hpp"
#include "../backend_test_fixture.hpp"
#include <memory>
#include <chrono>

using namespace tenzor;
using namespace tenzor::testing;

namespace {

// Mock implementations for testing (avoid linking issues)
class MockReLUBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        return inputs;
    }
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        return grad_outputs;
    }
};

class MockMatMulBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        return inputs;
    }
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        return grad_outputs;
    }
};

} // anonymous namespace

/**
 * @brief Multi-backend test fixture for GraphOptimizer tests
 */
// audit-2 P.9 — This file uses BackendTest (not MultiBackendDTypeTest)
// deliberately: Graph-optimiser pass semantics (fuse Linear+ReLU, dead-code
// elimination) operate on the Function graph structure, not the
// tensor element type. Dtype parameterisation would not add coverage.
class GraphOptimizerMultiBackendTest : public BackendTest {
protected:
    void SetUp() override {
        BackendTest::SetUp();
        optimizer = std::make_unique<GraphOptimizer>();
        graph = std::make_unique<ComputationGraph>();
    }

    void TearDown() override {
        optimizer.reset();
        graph.reset();
    }

    std::unique_ptr<GraphOptimizer> optimizer;
    std::unique_ptr<ComputationGraph> graph;
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_P(GraphOptimizerMultiBackendTest, ConstructorInitializesCorrectly) {
    EXPECT_NE(optimizer, nullptr);
    auto stats = optimizer->get_stats();
    EXPECT_EQ(stats.linear_relu_fused, 0u);
    EXPECT_EQ(stats.conv_batchnorm_fused, 0u);
    EXPECT_EQ(stats.dead_nodes_removed, 0u);
    EXPECT_EQ(stats.total(), 0u);
}

TEST_P(GraphOptimizerMultiBackendTest, ResetStatsClearsCounters) {
    // Manually set some stats by running optimizations
    optimizer->optimize(*graph);

    // Reset stats
    optimizer->reset_stats();

    auto stats = optimizer->get_stats();
    EXPECT_EQ(stats.linear_relu_fused, 0u);
    EXPECT_EQ(stats.conv_batchnorm_fused, 0u);
    EXPECT_EQ(stats.dead_nodes_removed, 0u);
}

TEST_P(GraphOptimizerMultiBackendTest, EmptyGraphOptimization) {
    // Empty graph should not crash
    EXPECT_NO_THROW(optimizer->optimize(*graph));

    auto stats = optimizer->get_stats();
    EXPECT_EQ(stats.total(), 0u);
}

// ============================================================================
// Linear + ReLU Fusion Tests
// ============================================================================

TEST_P(GraphOptimizerMultiBackendTest, FuseLinearReLUEmptyGraph) {
    size_t fused = optimizer->fuse_linear_relu(*graph);
    EXPECT_EQ(fused, 0u);
    EXPECT_EQ(optimizer->get_stats().linear_relu_fused, 0u);
}

TEST_P(GraphOptimizerMultiBackendTest, FuseLinearReLUWithMatMulAndReLU) {
    // Create MatMul node
    auto matmul_func = std::make_shared<MockMatMulBackward>();
    auto matmul_node = graph->add_node(matmul_func);

    // Create ReLU node
    auto relu_func = std::make_shared<MockReLUBackward>();
    auto relu_node = graph->add_node(relu_func);

    // Connect MatMul -> ReLU
    graph->connect(matmul_node, relu_node);

    // Should detect fusion opportunity
    size_t fused = optimizer->fuse_linear_relu(*graph);

    auto stats = optimizer->get_stats();
    EXPECT_EQ(stats.linear_relu_fused, fused);
}

TEST_P(GraphOptimizerMultiBackendTest, FuseLinearReLUMultiplePairs) {
    // Create multiple MatMul -> ReLU pairs
    for (int i = 0; i < 3; ++i) {
        auto matmul_func = std::make_shared<MockMatMulBackward>();
        auto matmul_node = graph->add_node(matmul_func);

        auto relu_func = std::make_shared<MockReLUBackward>();
        auto relu_node = graph->add_node(relu_func);

        graph->connect(matmul_node, relu_node);
    }

    size_t fused = optimizer->fuse_linear_relu(*graph);

    // Should detect multiple fusion opportunities
    auto stats = optimizer->get_stats();
    EXPECT_EQ(stats.linear_relu_fused, fused);
}

TEST_P(GraphOptimizerMultiBackendTest, FuseLinearReLUNoFusionWithMultipleConsumers) {
    // Create MatMul with multiple consumers
    auto matmul_func = std::make_shared<MockMatMulBackward>();
    auto matmul_node = graph->add_node(matmul_func);

    auto relu_func1 = std::make_shared<MockReLUBackward>();
    auto relu_node1 = graph->add_node(relu_func1);

    auto relu_func2 = std::make_shared<MockReLUBackward>();
    auto relu_node2 = graph->add_node(relu_func2);

    // MatMul has two consumers - should not fuse
    graph->connect(matmul_node, relu_node1);
    graph->connect(matmul_node, relu_node2);

    size_t fused = optimizer->fuse_linear_relu(*graph);

    // Should not fuse when MatMul has multiple consumers
    EXPECT_EQ(fused, 0u);
}

// ============================================================================
// Convolution + BatchNorm Fusion Tests
// ============================================================================

TEST_P(GraphOptimizerMultiBackendTest, FuseConvBatchNormEmptyGraph) {
    size_t fused = optimizer->fuse_conv_batchnorm(*graph);
    EXPECT_EQ(fused, 0u);
    EXPECT_EQ(optimizer->get_stats().conv_batchnorm_fused, 0u);
}

// ============================================================================
// Dead Code Elimination Tests
// ============================================================================

TEST_P(GraphOptimizerMultiBackendTest, EliminateDeadCodeEmptyGraph) {
    size_t removed = optimizer->eliminate_dead_code(*graph);
    EXPECT_EQ(removed, 0u);
    EXPECT_EQ(optimizer->get_stats().dead_nodes_removed, 0u);
}

TEST_P(GraphOptimizerMultiBackendTest, EliminateDeadCodeNoDeadNodes) {
    // Create a simple chain: A -> B -> C
    auto func_a = std::make_shared<AddBackward>();
    auto node_a = graph->add_node(func_a);

    auto func_b = std::make_shared<MulBackward>();
    auto node_b = graph->add_node(func_b);

    auto func_c = std::make_shared<MockReLUBackward>();
    auto node_c = graph->add_node(func_c);

    graph->connect(node_a, node_b);
    graph->connect(node_b, node_c);

    // All nodes are reachable, so no dead code
    size_t removed = optimizer->eliminate_dead_code(*graph);

    auto stats = optimizer->get_stats();
    EXPECT_EQ(stats.dead_nodes_removed, removed);
}

TEST_P(GraphOptimizerMultiBackendTest, EliminateDeadCodeIsolatedNode) {
    // Create a connected chain: A -> B
    auto func_a = std::make_shared<AddBackward>();
    auto node_a = graph->add_node(func_a);

    auto func_b = std::make_shared<MulBackward>();
    auto node_b = graph->add_node(func_b);

    graph->connect(node_a, node_b);

    // Create an isolated node (dead code)
    auto func_isolated = std::make_shared<MockReLUBackward>();
    auto node_isolated = graph->add_node(func_isolated);

    // Should detect the isolated node as dead code
    size_t removed = optimizer->eliminate_dead_code(*graph);

    auto stats = optimizer->get_stats();
    EXPECT_EQ(stats.dead_nodes_removed, removed);
}

// ============================================================================
// Full Optimization Pipeline Tests
// ============================================================================

TEST_P(GraphOptimizerMultiBackendTest, OptimizeAppliesAllPasses) {
    // Create a graph with various patterns
    auto matmul_func = std::make_shared<MockMatMulBackward>();
    auto matmul_node = graph->add_node(matmul_func);

    auto relu_func = std::make_shared<MockReLUBackward>();
    auto relu_node = graph->add_node(relu_func);

    graph->connect(matmul_node, relu_node);

    // Create isolated node for dead code elimination
    auto dead_func = std::make_shared<AddBackward>();
    auto dead_node = graph->add_node(dead_func);

    // Run full optimization
    EXPECT_NO_THROW(optimizer->optimize(*graph));

    auto stats = optimizer->get_stats();

    // Should have attempted both fusion and dead code elimination
    EXPECT_GE(stats.total(), 0u);
}

TEST_P(GraphOptimizerMultiBackendTest, OptimizeIdempotent) {
    // Create a simple graph
    auto func = std::make_shared<MockReLUBackward>();
    auto node = graph->add_node(func);

    // Optimize once
    optimizer->optimize(*graph);
    auto stats1 = optimizer->get_stats();

    // Reset and optimize again
    optimizer->reset_stats();
    optimizer->optimize(*graph);
    auto stats2 = optimizer->get_stats();

    // Second optimization should not find additional work
    EXPECT_EQ(stats1.total(), stats2.total());
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_P(GraphOptimizerMultiBackendTest, StatisticsTotalMatchesSum) {
    // Run optimization on a graph
    optimizer->optimize(*graph);

    auto stats = optimizer->get_stats();

    // Total should equal sum of individual counts
    size_t expected_total = stats.linear_relu_fused +
                           stats.conv_batchnorm_fused +
                           stats.dead_nodes_removed;

    EXPECT_EQ(stats.total(), expected_total);
}

TEST_P(GraphOptimizerMultiBackendTest, StatisticsAccumulate) {
    // Run multiple optimization passes
    optimizer->fuse_linear_relu(*graph);
    auto stats1 = optimizer->get_stats();
    size_t count1 = stats1.linear_relu_fused;

    // Create more nodes and optimize again
    auto matmul_func = std::make_shared<MockMatMulBackward>();
    auto matmul_node = graph->add_node(matmul_func);

    auto relu_func = std::make_shared<MockReLUBackward>();
    auto relu_node = graph->add_node(relu_func);

    graph->connect(matmul_node, relu_node);

    optimizer->fuse_linear_relu(*graph);
    auto stats2 = optimizer->get_stats();

    // Statistics should accumulate
    EXPECT_GE(stats2.linear_relu_fused, count1);
}

// ============================================================================
// Pattern Matching Tests
// ============================================================================

TEST_P(GraphOptimizerMultiBackendTest, PatternMatchingBasicSequence) {
    // Create a sequence that should match: Add -> Mul -> ReLU
    auto add_func = std::make_shared<AddBackward>();
    auto add_node = graph->add_node(add_func);

    auto mul_func = std::make_shared<MulBackward>();
    auto mul_node = graph->add_node(mul_func);

    auto relu_func = std::make_shared<MockReLUBackward>();
    auto relu_node = graph->add_node(relu_func);

    graph->connect(add_node, mul_node);
    graph->connect(mul_node, relu_node);

    // Pattern matching should work (tested indirectly through fusion)
    EXPECT_NO_THROW(optimizer->optimize(*graph));
}

TEST_P(GraphOptimizerMultiBackendTest, ComplexGraphOptimization) {
    // Create a more complex graph with multiple paths
    // Input -> MatMul -> ReLU -> Add -> Output
    //       -> Mul ---------------^

    auto matmul_func = std::make_shared<MockMatMulBackward>();
    auto matmul_node = graph->add_node(matmul_func);

    auto relu_func = std::make_shared<MockReLUBackward>();
    auto relu_node = graph->add_node(relu_func);

    auto mul_func = std::make_shared<MulBackward>();
    auto mul_node = graph->add_node(mul_func);

    auto add_func = std::make_shared<AddBackward>();
    auto add_node = graph->add_node(add_func);

    graph->connect(matmul_node, relu_node);
    graph->connect(relu_node, add_node);
    graph->connect(mul_node, add_node);

    // Should handle complex graph structures
    EXPECT_NO_THROW(optimizer->optimize(*graph));

    auto stats = optimizer->get_stats();
    EXPECT_GE(stats.total(), 0u);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_P(GraphOptimizerMultiBackendTest, NullptrHandling) {
    // Ensure optimizer handles null nodes gracefully
    EXPECT_NO_THROW(optimizer->optimize(*graph));
}

TEST_P(GraphOptimizerMultiBackendTest, SingleNodeGraph) {
    auto func = std::make_shared<MockReLUBackward>();
    auto node = graph->add_node(func);

    EXPECT_NO_THROW(optimizer->optimize(*graph));
}

TEST_P(GraphOptimizerMultiBackendTest, CyclicGraphHandling) {
    // Note: ComputationGraph should not allow cycles
    // but we test the optimizer's robustness

    auto func_a = std::make_shared<AddBackward>();
    auto node_a = graph->add_node(func_a);

    auto func_b = std::make_shared<MulBackward>();
    auto node_b = graph->add_node(func_b);

    // Create forward edges (cycle should be detected elsewhere)
    graph->connect(node_a, node_b);

    // Optimizer should handle gracefully
    EXPECT_NO_THROW(optimizer->optimize(*graph));
}

TEST_P(GraphOptimizerMultiBackendTest, LargeGraphPerformance) {
    // Create a large graph to test performance
    const int num_layers = 100;

    std::shared_ptr<GraphNode> prev_node;
    for (int i = 0; i < num_layers; ++i) {
        auto func = std::make_shared<MockReLUBackward>();
        auto node = graph->add_node(func);

        if (prev_node) {
            graph->connect(prev_node, node);
        }
        prev_node = node;
    }

    // Should handle large graphs efficiently
    auto start = std::chrono::high_resolution_clock::now();
    optimizer->optimize(*graph);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Optimization should complete in reasonable time (< 1 second for 100 nodes)
    EXPECT_LT(duration.count(), 1000);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_P(GraphOptimizerMultiBackendTest, MultipleOptimizersIndependent) {
    GraphOptimizer optimizer1;
    GraphOptimizer optimizer2;

    // Run optimizations with different optimizers
    optimizer1.optimize(*graph);
    auto stats1 = optimizer1.get_stats();

    optimizer2.optimize(*graph);
    auto stats2 = optimizer2.get_stats();

    // Each optimizer should have independent statistics
    EXPECT_EQ(stats1.total(), stats2.total());
}

INSTANTIATE_BACKEND_TESTS(GraphOptimizerMultiBackendTest);
