/**
 * @file graph.hpp
 * @brief Computation graph structure for automatic differentiation
 *
 * Defines the graph representation used to track operations and
 * dependencies for backpropagation in automatic differentiation.
 */

#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "function.hpp"

namespace tenzor {

// Forward declaration for friend
class GraphOptimizer;

/**
 * @brief Node in the computation graph.
 *
 * Represents a single operation (Function) in the computation graph
 * with connections to subsequent operations. Used during backpropagation
 * to traverse the graph in topological order.
 *
 * @see ComputationGraph for graph management
 * @see Function for operation definitions
 */
struct GraphNode {
    std::shared_ptr<Function> function;                  ///< Gradient function for this node
    std::vector<std::weak_ptr<GraphNode>> next_nodes;    ///< Subsequent nodes in forward pass
    int ref_count{0};                                    ///< Reference count for memory management
};

/**
 * @brief Computation graph for automatic differentiation.
 *
 * Manages the directed acyclic graph (DAG) of operations performed
 * during the forward pass. This graph is used during backpropagation
 * to compute gradients in the correct order.
 *
 * Key features:
 * - Dynamic graph construction during forward pass
 * - Topological sorting for backward pass
 * - Efficient memory management with reference counting
 * - Graph statistics and analysis
 *
 * The graph is typically managed automatically by Variable and
 * BackwardEngine rather than being used directly.
 *
 * @code
 * ComputationGraph graph;
 *
 * // Add operations to graph
 * auto node1 = graph.add_node(add_func);
 * auto node2 = graph.add_node(mul_func);
 *
 * // Connect operations
 * graph.connect(node1, node2);
 *
 * // Get execution order for backward pass
 * auto sorted = graph.topological_sort(node2);
 *
 * // Clean up
 * graph.clear();
 * @endcode
 *
 * @see BackwardEngine for gradient execution
 * @see Function for graph node operations
 */
class ComputationGraph {
public:
    /**
     * @brief Default constructor.
     */
    ComputationGraph() = default;

    // Friend declaration to allow GraphOptimizer to access private nodes
    friend class GraphOptimizer;

    /**
     * @brief Add a function node to the graph.
     *
     * Creates a new graph node for the given function. If a node
     * already exists for this function, returns the existing node.
     *
     * @param func Gradient function to add
     * @return Shared pointer to graph node
     *
     * @code
     * auto add_func = std::make_shared<AddBackward>();
     * auto node = graph.add_node(add_func);
     * @endcode
     */
    auto add_node(std::shared_ptr<Function> func) -> std::shared_ptr<GraphNode>;

    /**
     * @brief Connect two nodes in the graph.
     *
     * Creates a directed edge from one node to another, representing
     * the flow of gradients during backpropagation. The 'from' node
     * depends on the output of the 'to' node.
     *
     * @param from Source node (earlier in forward pass)
     * @param to Destination node (later in forward pass)
     *
     * @code
     * graph.connect(input_node, output_node);
     * // Gradients flow from output_node back to input_node
     * @endcode
     */
    auto connect(std::shared_ptr<GraphNode> from,
                std::shared_ptr<GraphNode> to) -> void;

    /**
     * @brief Perform topological sort from root node.
     *
     * Computes a topologically sorted order of nodes from the root
     * backward to all dependencies. This order ensures that gradients
     * are computed correctly during backpropagation.
     *
     * Uses depth-first search with cycle detection to traverse the graph.
     *
     * @param root Root node to start sorting from
     * @return Vector of nodes in reverse topological order
     *
     * @throws std::runtime_error if graph contains cycles
     *
     * @code
     * auto sorted_nodes = graph.topological_sort(output_node);
     * // Nodes are now in order for backward pass execution
     * @endcode
     */
    auto topological_sort(std::shared_ptr<GraphNode> root)
        -> std::vector<std::shared_ptr<GraphNode>>;

    /**
     * @brief Clear all nodes and edges from graph.
     *
     * Removes all nodes and resets edge count. Call this to free
     * memory after a backward pass is complete.
     */
    auto clear() -> void;

    /**
     * @brief Get number of nodes in graph.
     *
     * @return Count of function nodes in the graph
     */
    auto node_count() const -> size_t;

    /**
     * @brief Get number of edges in graph.
     *
     * @return Count of connections between nodes
     */
    auto edge_count() const -> size_t;

private:
    std::unordered_map<Function*, std::shared_ptr<GraphNode>> nodes_;  ///< Map of functions to nodes
    size_t edge_count_{0};  ///< Total number of edges in graph
};

} // namespace tenzor
