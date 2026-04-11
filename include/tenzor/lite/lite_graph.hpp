/**
 * @file lite_graph.hpp
 * @brief Static execution graph for the lite runtime
 *
 * LiteGraph represents an optimized, topologically-sorted sequence of
 * operations ready for inference execution. Nodes reference tensors by
 * integer ID into a flat tensor table managed by the runtime.
 */

#pragma once

#include "runtime.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace tenzor::lite {

// ============================================================================
// Operator type enum — covers the ops supported by the lite runtime
// ============================================================================

enum class LiteOpType : uint16_t {
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    Gemm,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    GELU,
    Conv2d,
    MaxPool2d,
    AvgPool2d,
    AdaptiveAvgPool2d,
    BatchNorm2d,
    LayerNorm,
    Reshape,
    Transpose,
    Flatten,
    Concat,
    QuantizedMatMul,
    QuantizedConv2d,
};

// ============================================================================
// Small-buffer attribute storage (no heap allocation for common cases)
// ============================================================================

struct LiteAttributes {
    float f[4]{};
    int64_t i[4]{};
};

// ============================================================================
// Graph node — one fused or atomic operation
// ============================================================================

struct LiteNode {
    LiteOpType op;
    std::vector<int16_t> input_ids;
    std::vector<int16_t> output_ids;
    LiteAttributes attrs;
};

// ============================================================================
// LiteGraph — the full execution plan
// ============================================================================

class LiteGraph {
public:
    /** Execute the graph with the given external inputs. */
    auto execute(const std::vector<LiteTensor>& inputs) -> std::vector<LiteTensor>;

    /** Append a node to the execution plan. */
    auto add_node(LiteNode node) -> void;

    /** Return the number of nodes in the graph. */
    auto num_nodes() const -> size_t;

    /** Read-only access to the node list (for serialization). */
    auto nodes() const -> const std::vector<LiteNode>& { return nodes_; }

private:
    std::vector<LiteNode> nodes_;
};

}  // namespace tenzor::lite
