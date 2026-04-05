/**
 * @file pattern_matcher.hpp
 * @brief Graph pattern matching for extended kernel fusion
 *
 * Identifies fusible subgraph patterns beyond simple pairwise fusion.
 * Supports: reduction chains, GEMM epilogues, softmax, LayerNorm,
 * RMSNorm, and small MLP sequences.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include "codegen.hpp"   // FusionKind defined here
#include "graph.hpp"
#include "tracer.hpp"

namespace tenzor {
namespace jit {

/**
 * @brief A matched fusion pattern with its constituent nodes.
 */
struct FusionMatch {
    FusionKind kind;
    std::vector<std::shared_ptr<Node>> nodes;    ///< Nodes in topological order
    std::vector<std::shared_ptr<Value>> inputs;   ///< External inputs to the pattern
    std::vector<std::shared_ptr<Value>> outputs;  ///< External outputs from the pattern
    int64_t estimated_elements{0};                ///< Element count for cost model
    std::string signature;                        ///< Unique key for kernel caching

    /// Compute a cache-friendly signature from kind + node types + dtypes.
    auto compute_signature() -> std::string;
};

// ============================================================================
// Pattern matcher
// ============================================================================

/**
 * @brief Identifies fusible subgraph patterns in a JIT IR graph.
 *
 * Scans the graph for known multi-node patterns that can be replaced
 * with a single fused kernel. Patterns are checked in priority order
 * (most specific first) to avoid overlapping matches.
 *
 * Usage:
 * @code
 * PatternMatcher matcher;
 * auto matches = matcher.find_all(graph);
 * for (auto& match : matches) {
 *     if (cost_model.should_fuse({match.nodes.size(), match.estimated_elements, ...})) {
 *         // Replace nodes with fused kernel node
 *     }
 * }
 * @endcode
 */
class PatternMatcher {
public:
    /**
     * @brief Find all fusible patterns in the graph.
     *
     * Patterns are returned in topological order. Nodes that appear
     * in one match are excluded from subsequent matches.
     *
     * @param graph Graph to search
     * @return Vector of non-overlapping fusion matches
     */
    auto find_all(const Graph& graph) -> std::vector<FusionMatch>;

    /**
     * @brief Set maximum hidden dimension for SmallMLP patterns.
     *
     * @param max_dim Maximum hidden dimension (default: 4096)
     */
    auto set_max_mlp_hidden_dim(int64_t max_dim) -> void { max_mlp_hidden_ = max_dim; }

private:
    int64_t max_mlp_hidden_{4096};

    // Individual pattern detectors.
    // Each returns a match if the pattern starts at the given node, or nullopt.
    auto match_softmax(const Graph& graph, size_t start_idx,
                       const std::unordered_set<Node*>& used) -> std::optional<FusionMatch>;
    auto match_layer_norm(const Graph& graph, size_t start_idx,
                          const std::unordered_set<Node*>& used) -> std::optional<FusionMatch>;
    auto match_rms_norm(const Graph& graph, size_t start_idx,
                        const std::unordered_set<Node*>& used) -> std::optional<FusionMatch>;
    auto match_gemm_epilogue(const Graph& graph, size_t start_idx,
                             const std::unordered_set<Node*>& used) -> std::optional<FusionMatch>;
    auto match_small_mlp(const Graph& graph, size_t start_idx,
                         const std::unordered_set<Node*>& used) -> std::optional<FusionMatch>;
    auto match_reduction_chain(const Graph& graph, size_t start_idx,
                               const std::unordered_set<Node*>& used) -> std::optional<FusionMatch>;

    /// Check if a node's single output is consumed only by nodes in the candidate set.
    static auto has_single_use(const std::shared_ptr<Node>& node) -> bool;

    /// Check if a node is an element-wise unary or binary op.
    static auto is_elementwise(OpType op) -> bool;

    /// Check if a node is a reduction op.
    static auto is_reduction(OpType op) -> bool;

    /// Check if a node is an activation op.
    static auto is_activation(OpType op) -> bool;

    /// Collect external inputs for a set of matched nodes.
    static auto collect_external_inputs(
        const std::vector<std::shared_ptr<Node>>& nodes)
        -> std::vector<std::shared_ptr<Value>>;

    /// Collect external outputs for a set of matched nodes.
    static auto collect_external_outputs(
        const std::vector<std::shared_ptr<Node>>& nodes)
        -> std::vector<std::shared_ptr<Value>>;
};

} // namespace jit
} // namespace tenzor
