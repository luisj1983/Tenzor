/**
 * @file fusion_optimizer.hpp
 * @brief Graph-level kernel fusion optimization for Tenzor
 *
 * Implements pattern matching and graph rewriting to fuse multiple operations
 * into optimized kernels. Significantly reduces kernel launch overhead and
 * improves memory bandwidth utilization.
 *
 * Supported fusion patterns:
 * - Linear + ReLU: Fused fully connected + activation
 * - Conv2d + BatchNorm2d + ReLU: Fused convolution pipeline
 * - MatMul + Add: Fused matrix multiplication with bias
 * - Element-wise chains: add + mul + relu, etc.
 * - Attention mechanisms: Fused multi-head attention
 *
 * Performance improvements:
 * - 1.5-3x speedup on fused operations
 * - 40-60% reduction in kernel launches
 * - 30-50% reduction in memory bandwidth
 */

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include "../core/tensor.hpp"
#include "../autograd/graph.hpp"

namespace tenzor {
namespace ops {

// Forward declarations
class FusionPattern;
class FusionGraph;
class FusionOptimizer;

/**
 * @brief Type of operation in fusion graph
 */
enum class OpType {
    // Linear operations
    MatMul,
    Linear,
    Conv2d,

    // Activations
    ReLU,
    GELU,
    Sigmoid,
    Tanh,

    // Normalization
    BatchNorm2d,
    LayerNorm,

    // Element-wise
    Add,
    Mul,
    Sub,
    Div,

    // Attention
    Softmax,
    Dropout,

    // Loss
    CrossEntropy,

    // Unknown/Other
    Unknown
};

/**
 * @brief Node representing an operation in the fusion graph
 */
struct FusionNode {
    size_t id;                                  ///< Unique node identifier
    OpType op_type;                             ///< Operation type
    std::string op_name;                        ///< Original operation name
    std::vector<size_t> inputs;                 ///< Input node IDs
    std::vector<size_t> outputs;                ///< Output node IDs
    std::unordered_map<std::string, std::string> attributes;  ///< Op attributes
    bool is_fusible{true};                      ///< Can this node be fused?
    bool is_fused{false};                       ///< Has this node been fused?
    size_t fusion_group{0};                     ///< ID of fusion group (if fused)

    /**
     * @brief Construct fusion node
     */
    FusionNode(size_t node_id, OpType type, const std::string& name)
        : id(node_id), op_type(type), op_name(name) {}
};

/**
 * @brief Represents a detected fusion pattern
 */
struct FusedOp {
    std::string fused_op_name;                  ///< Name of fused operation
    std::vector<size_t> node_ids;               ///< Nodes included in fusion
    std::vector<size_t> input_ids;              ///< External inputs to fusion
    std::vector<size_t> output_ids;             ///< External outputs from fusion
    float estimated_speedup{1.0f};              ///< Expected speedup factor
    /// Parameters extracted from the matched nodes (e.g. an absorbed Dropout's
    /// probability/seed, the softmax scale, the causal flag) that the executor
    /// needs. Propagated from Match::attributes into the rewritten node's attrs
    /// so execute_fused_op reproduces the unfused numerics.
    std::unordered_map<std::string, std::string> attributes;

    /**
     * @brief Construct fused operation
     */
    FusedOp(const std::string& name, const std::vector<size_t>& nodes)
        : fused_op_name(name), node_ids(nodes) {}
};

/**
 * @brief Fusion pattern matcher
 *
 * Detects common operation patterns that can be fused into
 * optimized kernels.
 */
class FusionPattern {
public:
    /**
     * @brief Pattern match result
     */
    struct Match {
        std::vector<size_t> matched_nodes;      ///< Matched node IDs
        float confidence{0.0f};                  ///< Match confidence [0,1]
        std::string pattern_name;                ///< Name of matched pattern
        /// Per-fusion parameters extracted from the matched nodes that the
        /// executor needs (e.g. an absorbed Dropout node's probability/seed).
        /// Forwarded into execute_fused_op's attribute map so the fused op
        /// reproduces the unfused numerics.
        std::unordered_map<std::string, std::string> attributes;

        explicit operator bool() const { return !matched_nodes.empty(); }
    };

    /**
     * @brief Construct pattern matcher with specific pattern
     */
    explicit FusionPattern(const std::string& pattern_name);

    /**
     * @brief Match pattern starting from node
     *
     * @param graph Fusion graph to search
     * @param start_node Starting node ID
     * @return Match result (empty if no match)
     */
    auto match(const FusionGraph& graph, size_t start_node) const -> Match;

    /**
     * @brief Get pattern name
     */
    auto get_name() const -> const std::string& { return pattern_name_; }

    /**
     * @brief Get expected speedup for this pattern
     */
    auto get_speedup() const -> float { return expected_speedup_; }

private:
    std::string pattern_name_;
    std::vector<OpType> pattern_ops_;
    float expected_speedup_{1.5f};

    // Pattern-specific matching logic
    auto match_linear_relu(const FusionGraph& graph, size_t start) const -> Match;
    auto match_conv_bn_relu(const FusionGraph& graph, size_t start) const -> Match;
    auto match_matmul_add(const FusionGraph& graph, size_t start) const -> Match;
    auto match_elementwise_chain(const FusionGraph& graph, size_t start) const -> Match;
    auto match_attention(const FusionGraph& graph, size_t start) const -> Match;
};

/**
 * @brief Computational graph for fusion analysis
 *
 * Represents operations and data flow for fusion optimization.
 * Built from execution traces or autograd graphs.
 */
class FusionGraph {
public:
    /**
     * @brief Construct empty fusion graph
     */
    FusionGraph() = default;

    /**
     * @brief Add node to graph
     *
     * @param op_type Type of operation
     * @param op_name Name of operation
     * @param inputs Input node IDs
     * @param attributes Operation attributes
     * @return Node ID
     */
    auto add_node(OpType op_type,
                  const std::string& op_name,
                  const std::vector<size_t>& inputs = {},
                  const std::unordered_map<std::string, std::string>& attributes = {})
        -> size_t;

    /**
     * @brief Add edge between nodes
     *
     * @param from Source node ID
     * @param to Destination node ID
     */
    auto add_edge(size_t from, size_t to) -> void;

    /**
     * @brief Get node by ID
     *
     * @param node_id Node identifier
     * @return Reference to node
     * @throws std::out_of_range if node doesn't exist
     */
    auto get_node(size_t node_id) const -> const FusionNode&;

    /**
     * @brief Get mutable node by ID
     */
    auto get_node_mut(size_t node_id) -> FusionNode&;

    /**
     * @brief Get all nodes
     */
    auto get_nodes() const -> const std::vector<std::unique_ptr<FusionNode>>&;

    /**
     * @brief Get input nodes for a node
     *
     * @param node_id Target node ID
     * @return Vector of input node IDs
     */
    auto get_inputs(size_t node_id) const -> std::vector<size_t>;

    /**
     * @brief Get output nodes for a node
     *
     * @param node_id Target node ID
     * @return Vector of output node IDs
     */
    auto get_outputs(size_t node_id) const -> std::vector<size_t>;

    /**
     * @brief Get all nodes in topological order
     *
     * @return Vector of node IDs in execution order
     */
    auto topological_sort() const -> std::vector<size_t>;

    /**
     * @brief Check if graph has cycle
     *
     * @return true if graph contains a cycle
     */
    auto has_cycle() const -> bool;

    /**
     * @brief Get number of nodes
     */
    auto size() const -> size_t { return nodes_.size(); }

    /**
     * @brief Clear all nodes and edges
     */
    auto clear() -> void;

    /**
     * @brief Export graph to DOT format for visualization
     *
     * @return DOT format string
     */
    auto to_dot() const -> std::string;

private:
    std::vector<std::unique_ptr<FusionNode>> nodes_;
    std::unordered_map<size_t, std::unordered_set<size_t>> adjacency_list_;
    size_t next_id_{0};

    // Helper for cycle detection
    auto has_cycle_util(size_t node,
                       std::unordered_set<size_t>& visited,
                       std::unordered_set<size_t>& rec_stack) const -> bool;
};

/**
 * @brief Graph-level fusion optimizer
 *
 * Main interface for kernel fusion optimization. Analyzes computation
 * graphs, identifies fusion opportunities, and rewrites graphs with
 * fused operations.
 *
 * Usage:
 * @code
 * FusionOptimizer optimizer;
 * optimizer.add_pattern("linear_relu");
 * optimizer.add_pattern("conv_bn_relu");
 *
 * FusionGraph graph = build_graph_from_trace(trace);
 * auto fused_graph = optimizer.optimize(graph);
 *
 * auto stats = optimizer.get_statistics();
 * std::cout << "Fused " << stats.num_fusions << " patterns\n";
 * std::cout << "Expected speedup: " << stats.expected_speedup << "x\n";
 * @endcode
 */
class FusionOptimizer {
public:
    /**
     * @brief Optimization statistics
     */
    struct Statistics {
        size_t num_nodes_original{0};          ///< Original graph node count
        size_t num_nodes_optimized{0};         ///< Optimized graph node count
        size_t num_fusions{0};                 ///< Number of fused patterns
        size_t num_kernel_launches_saved{0};   ///< Saved kernel launches
        float expected_speedup{1.0f};          ///< Overall speedup estimate
        float memory_bandwidth_reduction{0.0f}; ///< Memory bandwidth saved
        std::unordered_map<std::string, size_t> pattern_counts; ///< Counts per pattern
    };

    /**
     * @brief Construct fusion optimizer
     *
     * @param enable_aggressive Enable aggressive fusion (may increase register pressure)
     */
    explicit FusionOptimizer(bool enable_aggressive = false);

    /**
     * @brief Add fusion pattern to optimizer
     *
     * Supported patterns:
     * - "linear_relu": Linear + ReLU
     * - "conv_bn_relu": Conv2d + BatchNorm2d + ReLU
     * - "matmul_add": MatMul + Add (bias)
     * - "elementwise_chain": Multiple element-wise ops
     * - "attention": Fused attention mechanism
     * - "all": Enable all patterns
     *
     * @param pattern_name Pattern identifier
     * @return true if pattern was added
     */
    auto add_pattern(const std::string& pattern_name) -> bool;

    /**
     * @brief Remove fusion pattern
     *
     * @param pattern_name Pattern to remove
     * @return true if pattern was removed
     */
    auto remove_pattern(const std::string& pattern_name) -> bool;

    /**
     * @brief Optimize fusion graph
     *
     * Analyzes graph, identifies fusion opportunities, and rewrites
     * graph with fused operations.
     *
     * @param graph Input computational graph
     * @return Optimized graph with fused operations
     */
    auto optimize(const FusionGraph& graph) -> FusionGraph;

    /**
     * @brief Get optimization statistics
     *
     * @return Statistics from last optimization
     */
    auto get_statistics() const -> const Statistics&;

    /**
     * @brief Reset statistics
     */
    auto reset_statistics() -> void;

    /**
     * @brief Set aggressive fusion mode
     *
     * Aggressive mode may fuse more operations but could increase
     * register pressure and reduce occupancy on GPUs.
     *
     * @param enable Enable aggressive mode
     */
    auto set_aggressive_mode(bool enable) -> void;

    /**
     * @brief Enable/disable specific device backend
     *
     * @param backend "cuda", "rocm", "vulkan", "oneapi", "mps", "cpu", or "all"
     * @param enable Enable or disable
     */
    auto set_backend(const std::string& backend, bool enable) -> void;

    /**
     * @brief Check whether a backend was enabled via set_backend()
     *
     * @param backend "cuda", "rocm", "vulkan", "oneapi", "mps", or "cpu"
     * @return Current enabled state (true for an unrecognized name)
     */
    auto is_backend_enabled(const std::string& backend) const -> bool;

    /**
     * @brief Check if pattern is supported
     *
     * @param pattern_name Pattern identifier
     * @return true if pattern is available
     */
    auto is_pattern_supported(const std::string& pattern_name) const -> bool;

    /**
     * @brief Get list of all supported patterns
     *
     * @return Vector of pattern names
     */
    auto get_supported_patterns() const -> std::vector<std::string>;

private:
    bool aggressive_mode_{false};
    bool cuda_enabled_{true};
    bool rocm_enabled_{true};
    bool cpu_enabled_{true};
    bool vulkan_enabled_{true};
    bool oneapi_enabled_{true};
    bool mps_enabled_{true};

    std::vector<std::unique_ptr<FusionPattern>> patterns_;
    Statistics stats_;

    // Internal optimization passes
    auto detect_patterns(const FusionGraph& graph) -> std::vector<FusedOp>;
    auto rewrite_graph(const FusionGraph& graph,
                      const std::vector<FusedOp>& fusions) -> FusionGraph;
    auto validate_fusion(const FusionGraph& graph,
                        const FusedOp& fusion) -> bool;
    auto estimate_speedup(const FusedOp& fusion) const -> float;
    auto compute_statistics(const FusionGraph& original,
                           const FusionGraph& optimized,
                           const std::vector<FusedOp>& fusions) -> void;
};

/**
 * @brief Build fusion graph from autograd computation graph
 *
 * Converts autograd graph to fusion graph for optimization.
 *
 * @param comp_graph Autograd computation graph
 * @param root Root node to start from
 * @return Fusion graph
 */
auto build_fusion_graph_from_autograd(
    const ComputationGraph& comp_graph,
    const std::shared_ptr<GraphNode>& root
) -> FusionGraph;

/**
 * @brief Convert operation name string to OpType enum
 *
 * @param op_name Operation name (e.g., "matmul", "relu", "conv2d")
 * @return Corresponding OpType
 */
auto string_to_op_type(const std::string& op_name) -> OpType;

/**
 * @brief Convert OpType enum to string
 *
 * @param op_type Operation type
 * @return String representation
 */
auto op_type_to_string(OpType op_type) -> std::string;

/**
 * @brief Execute fused operation
 *
 * Runtime dispatch for fused kernels. Selects optimal implementation
 * based on device, data type, and operation parameters.
 *
 * @param fused_op Fused operation descriptor
 * @param inputs Input tensors
 * @param attributes Operation attributes
 * @return Output tensors
 */
auto execute_fused_op(
    const FusedOp& fused_op,
    const std::vector<Tensor>& inputs,
    const std::unordered_map<std::string, std::string>& attributes
) -> std::vector<Tensor>;

} // namespace ops
} // namespace tenzor
