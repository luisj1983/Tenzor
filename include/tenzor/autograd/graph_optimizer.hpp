/**
 * @file graph_optimizer.hpp
 * @brief Computation graph optimization for automatic differentiation
 *
 * Provides graph-level optimizations such as operation fusion, dead code
 * elimination, and pattern-based transformations to improve performance
 * while preserving gradient computation correctness.
 */

#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include "graph.hpp"
#include "function.hpp"
#include "variable.hpp"

namespace tenzor {

/**
 * @brief Pattern descriptor for graph optimization.
 *
 * Describes a sequence of operations that can be matched and fused
 * in the computation graph. Used by the pattern matching system to
 * identify fusion opportunities.
 */
struct OpPattern {
    std::vector<std::string> op_sequence;  ///< Sequence of operation type names
    std::string fused_op_name;             ///< Name of the fused operation

    /**
     * @brief Construct an operation pattern.
     * @param ops Sequence of operation names to match
     * @param fused Name of the fused operation
     */
    OpPattern(std::vector<std::string> ops, std::string fused)
        : op_sequence(std::move(ops)), fused_op_name(std::move(fused)) {}
};

/**
 * @brief Optimization statistics.
 *
 * Tracks the number of optimizations applied during a graph optimization pass.
 * Used for reporting and debugging.
 */
struct OptimizationStats {
    size_t linear_relu_fused{0};      ///< Number of linear+relu fusions
    size_t conv_batchnorm_fused{0};   ///< Number of conv+batchnorm fusions
    size_t conv_relu_fused{0};        ///< Number of conv+relu fusions
    size_t batchnorm_relu_fused{0};   ///< Number of batchnorm+relu fusions
    size_t linear_gelu_fused{0};      ///< Number of linear+gelu fusions
    size_t conv_bn_relu_fused{0};     ///< Number of conv+bn+relu fusions
    size_t transpose_pairs_eliminated{0}; ///< Number of transpose pairs eliminated
    size_t reshape_chains_collapsed{0};   ///< Number of reshape chains collapsed
    size_t dead_nodes_removed{0};     ///< Number of dead code nodes removed
    size_t total_optimizations{0};    ///< Total optimizations applied

    auto total() const -> size_t {
        return linear_relu_fused + conv_batchnorm_fused + conv_relu_fused +
               batchnorm_relu_fused + linear_gelu_fused + conv_bn_relu_fused +
               transpose_pairs_eliminated + reshape_chains_collapsed +
               dead_nodes_removed;
    }
};

/**
 * @brief Computation graph optimizer for performance improvements.
 *
 * The GraphOptimizer performs pattern-based transformations on the computation
 * graph to improve performance while preserving correctness of gradient computation.
 *
 * Key optimizations:
 * - **Operation Fusion**: Combines adjacent operations (e.g., Linear+ReLU, Conv+BatchNorm)
 *   into single fused operations to reduce memory traffic and kernel launch overhead
 * - **Dead Code Elimination**: Removes nodes that don't contribute to the final output
 *   using reachability analysis
 * - **Pattern Matching**: Generic pattern matching framework for identifying fusion
 *   opportunities
 *
 * The optimizer maintains gradient computation correctness by ensuring:
 * - Fused operations implement correct backward pass
 * - Graph topology is preserved for gradient flow
 * - All reachable nodes from outputs are retained
 *
 * @code
 * ComputationGraph graph;
 * // ... build graph with operations ...
 *
 * GraphOptimizer optimizer;
 * optimizer.optimize(graph);  // Apply all optimizations
 *
 * // Or apply specific optimizations
 * size_t fused = optimizer.fuse_linear_relu(graph);
 * size_t removed = optimizer.eliminate_dead_code(graph);
 *
 * // Get statistics
 * auto stats = optimizer.get_stats();
 * std::cout << "Total optimizations: " << stats.total() << "\n";
 * @endcode
 *
 * @note All optimizations preserve the semantics of gradient computation
 * @see ComputationGraph for graph representation
 * @see Function for operation definitions
 */
class GraphOptimizer {
public:
    /**
     * @brief Default constructor.
     */
    GraphOptimizer() = default;

    /**
     * @brief Apply all optimization passes to the computation graph.
     *
     * Runs all optimization passes in sequence:
     * 1. Operation fusion (Linear+ReLU, Conv+BatchNorm)
     * 2. Dead code elimination
     *
     * The order is chosen to maximize optimization opportunities.
     * For example, fusion may create new dead code that can be eliminated.
     *
     * @param graph Computation graph to optimize
     *
     * @code
     * ComputationGraph graph;
     * // ... build graph ...
     * GraphOptimizer optimizer;
     * optimizer.optimize(graph);  // Apply all optimizations
     * @endcode
     *
     * @note This is the recommended way to optimize a graph
     */
    auto optimize(ComputationGraph& graph) -> void;

    /**
     * @brief Optimize the computation graph rooted at a Variable.
     *
     * Convenience method that builds a ComputationGraph from the Variable's
     * grad_fn chain, optimizes it, and returns the optimization stats.
     * This is the recommended entry point for users.
     *
     * @param root Variable whose computation graph to optimize
     * @return Optimization statistics
     */
    auto optimize_variable(Variable& root) -> OptimizationStats;

    /**
     * @brief Fuse Linear+ReLU operation sequences.
     *
     * Identifies patterns of Linear (MatMul) followed by ReLU activation
     * and replaces them with a single fused operation. This reduces:
     * - Memory traffic (no intermediate tensor storage)
     * - Kernel launch overhead
     * - Memory allocations
     *
     * Pattern matched:
     * @code
     * x -> MatMul -> y -> ReLU -> z
     * Becomes:
     * x -> FusedLinearReLU -> z
     * @endcode
     *
     * The fused operation computes both forward and backward passes correctly,
     * maintaining gradient computation semantics.
     *
     * @param graph Computation graph to optimize
     * @return Number of Linear+ReLU fusions performed
     *
     * @code
     * ComputationGraph graph;
     * GraphOptimizer optimizer;
     * size_t fused = optimizer.fuse_linear_relu(graph);
     * std::cout << "Fused " << fused << " Linear+ReLU pairs\n";
     * @endcode
     *
     * @note Only adjacent operations are fused (no intermediate consumers)
     * @note Gradients are computed correctly through the fused operation
     */
    auto fuse_linear_relu(ComputationGraph& graph) -> size_t;

    /**
     * @brief Fuse Convolution+BatchNorm operation sequences.
     *
     * Identifies patterns of Convolution followed by BatchNorm and replaces
     * them with a single fused operation. During inference, BatchNorm parameters
     * can be folded into convolution weights, eliminating the BatchNorm entirely.
     *
     * Pattern matched:
     * @code
     * x -> Conv2d -> y -> BatchNorm -> z
     * Becomes:
     * x -> FusedConvBatchNorm -> z
     * @endcode
     *
     * Benefits:
     * - Reduced memory traffic
     * - Fewer kernel launches
     * - During inference: parameters can be folded (single conv operation)
     *
     * @param graph Computation graph to optimize
     * @return Number of Conv+BatchNorm fusions performed
     *
     * @code
     * ComputationGraph graph;
     * GraphOptimizer optimizer;
     * size_t fused = optimizer.fuse_conv_batchnorm(graph);
     * std::cout << "Fused " << fused << " Conv+BatchNorm pairs\n";
     * @endcode
     *
     * @note Maintains correct gradient flow during training
     * @note Can enable further optimizations during inference
     */
    auto fuse_conv_batchnorm(ComputationGraph& graph) -> size_t;

    /** @brief Fuse Conv + ReLU into FusedConv2dReLU. */
    auto fuse_conv_relu(ComputationGraph& graph) -> size_t;

    /** @brief Fuse BatchNorm + ReLU into FusedBatchNormReLU. */
    auto fuse_batchnorm_relu(ComputationGraph& graph) -> size_t;

    /** @brief Fuse Linear + GELU into a single fused op. */
    auto fuse_linear_gelu(ComputationGraph& graph) -> size_t;

    /** @brief Fuse Conv + BatchNorm + ReLU into a single fused op. */
    auto fuse_conv_batchnorm_relu(ComputationGraph& graph) -> size_t;

    /** @brief Eliminate redundant transpose pairs (A,B followed by B,A). */
    auto eliminate_transpose_pairs(ComputationGraph& graph) -> size_t;

    /** @brief Collapse consecutive reshape operations into one. */
    auto collapse_reshape_chains(ComputationGraph& graph) -> size_t;

    /**
     * @brief Eliminate dead code (unreachable nodes) from the graph.
     *
     * Removes nodes that don't contribute to any output using reachability
     * analysis. A node is dead if:
     * - It has no outgoing edges (no consumers)
     * - It is not reachable from any output node
     *
     * Algorithm:
     * 1. Identify all output nodes (nodes with consumers)
     * 2. Perform reverse reachability analysis from outputs
     * 3. Remove nodes not reachable from any output
     *
     * Benefits:
     * - Reduced memory usage
     * - Fewer operations executed
     * - Faster backward pass
     *
     * @param graph Computation graph to optimize
     * @return Number of dead nodes removed
     *
     * @code
     * ComputationGraph graph;
     * // ... build graph with some unused operations ...
     * GraphOptimizer optimizer;
     * size_t removed = optimizer.eliminate_dead_code(graph);
     * std::cout << "Removed " << removed << " dead nodes\n";
     * @endcode
     *
     * @note Preserves all nodes reachable from outputs
     * @note Safe to call multiple times (idempotent after first call)
     */
    auto eliminate_dead_code(ComputationGraph& graph) -> size_t;

    /**
     * @brief Get optimization statistics.
     *
     * Returns statistics about optimizations applied since the optimizer
     * was created or last reset.
     *
     * @return Optimization statistics
     *
     * @code
     * GraphOptimizer optimizer;
     * optimizer.optimize(graph);
     * auto stats = optimizer.get_stats();
     * std::cout << "Linear+ReLU fused: " << stats.linear_relu_fused << "\n";
     * std::cout << "Conv+BatchNorm fused: " << stats.conv_batchnorm_fused << "\n";
     * std::cout << "Dead nodes removed: " << stats.dead_nodes_removed << "\n";
     * @endcode
     */
    auto get_stats() const -> const OptimizationStats& {
        return stats_;
    }

    /**
     * @brief Reset optimization statistics.
     *
     * Clears all optimization counters to zero. Useful when optimizing
     * multiple graphs with the same optimizer instance.
     *
     * @code
     * GraphOptimizer optimizer;
     * optimizer.optimize(graph1);
     * optimizer.reset_stats();
     * optimizer.optimize(graph2);  // Fresh stats for second graph
     * @endcode
     */
    auto reset_stats() -> void {
        stats_ = OptimizationStats{};
    }

private:
    /**
     * @brief Match a pattern starting from a given node.
     *
     * Attempts to match the specified pattern starting from the given node.
     * Returns the sequence of matched nodes if successful, empty vector otherwise.
     *
     * @param start Starting node for pattern matching
     * @param pattern Pattern to match
     * @param graph Graph being analyzed
     * @return Vector of matched nodes (empty if no match)
     */
    auto match_pattern(std::shared_ptr<GraphNode> start,
                      const OpPattern& pattern,
                      const ComputationGraph& graph) const
        -> std::vector<std::shared_ptr<GraphNode>>;

    /**
     * @brief Check if a node has a specific operation type.
     *
     * Uses RTTI to determine the concrete type of the function in the node.
     *
     * @param node Node to check
     * @param op_type Expected operation type name
     * @return true if node contains the specified operation type
     */
    auto is_operation_type(std::shared_ptr<GraphNode> node,
                          const std::string& op_type) const -> bool;

    /**
     * @brief Replace a sequence of nodes with a fused operation.
     *
     * Replaces a matched pattern of nodes with a single fused node.
     * Updates all graph edges to maintain correct topology.
     *
     * @param nodes Nodes to replace
     * @param fused_node Replacement fused node
     * @param graph Graph being modified
     */
    auto replace_nodes(const std::vector<std::shared_ptr<GraphNode>>& nodes,
                      std::shared_ptr<GraphNode> fused_node,
                      ComputationGraph& graph) -> void;

    /**
     * @brief Perform reachability analysis from output nodes.
     *
     * Computes the set of all nodes reachable from nodes that have consumers.
     * Uses reverse graph traversal (backward from outputs).
     *
     * @param graph Graph to analyze
     * @return Set of reachable node pointers
     */
    auto compute_reachable_nodes(const ComputationGraph& graph) const
        -> std::unordered_set<GraphNode*>;

    /**
     * @brief Get all nodes in the graph.
     *
     * Extracts all graph nodes for iteration. Since the graph stores nodes
     * in a map, this provides a convenient way to iterate over all nodes.
     *
     * @param graph Graph to extract nodes from
     * @return Vector of all graph nodes
     */
    auto get_all_nodes(const ComputationGraph& graph) const
        -> std::vector<std::shared_ptr<GraphNode>>;

    /**
     * @brief Check if node can be fused (no intermediate consumers).
     *
     * A node can be fused with its successor if no other nodes consume
     * its output (i.e., it has exactly one consumer which is the fusion target).
     *
     * @param node Node to check
     * @return true if node has no intermediate consumers
     */
    auto has_single_consumer(std::shared_ptr<GraphNode> node) const -> bool;

    OptimizationStats stats_;  ///< Optimization statistics
};

} // namespace tenzor
