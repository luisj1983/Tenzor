/**
 * @file symbolic_shape_inference.hpp
 * @brief Standalone symbolic shape inference engine for JIT graph nodes
 *
 * Provides a reusable SymbolicShapeInference class that can infer output
 * symbolic shapes for individual nodes. This decouples shape inference logic
 * from Graph::infer_symbolic_types() so it can be used by optimization passes,
 * graph transformations, and validation without requiring a full graph traversal.
 */

#pragma once

#include "symbolic_shape.hpp"
#include "graph.hpp"

#include <optional>
#include <vector>

namespace tenzor {
namespace jit {

/**
 * @brief Symbolic shape inference engine for JIT IR nodes.
 *
 * Infers output symbolic shapes for a node based on its operation type,
 * attributes, and input symbolic shapes. Supports elementwise broadcasting,
 * matrix operations, convolutions, reductions, shape transformations, and
 * linear layers.
 *
 * @code
 * SymbolicShapeInference inference;
 * auto output_shapes = inference.infer(node);
 * for (size_t i = 0; i < output_shapes.size(); ++i) {
 *     node->outputs()[i]->set_symbolic_shape(output_shapes[i]);
 * }
 * @endcode
 */
class SymbolicShapeInference {
public:
    /**
     * @brief Infer output shapes for a node given its input symbolic shapes.
     *
     * Dispatches to the appropriate inference method based on the node's
     * OpType. Input symbolic shapes are gathered from the node's input
     * values. If an input has no symbolic shape set, a concrete shape
     * is derived from its integer shape.
     *
     * @param node Node to infer shapes for
     * @return Vector of symbolic shapes, one per output
     */
    auto infer(const Node* node) -> std::vector<SymbolicShape>;

private:
    /// Infer output shape for binary/unary elementwise ops via broadcasting.
    auto infer_elementwise(const Node* node) -> std::vector<SymbolicShape>;

    /// Infer output shape for MatMul: (..., M, K) x (..., K, N) -> (..., M, N).
    auto infer_matmul(const Node* node) -> std::vector<SymbolicShape>;

    /// Infer output shape for Conv2d using the standard convolution formula.
    auto infer_conv2d(const Node* node) -> std::vector<SymbolicShape>;

    /// Infer output shape for Reshape, handling -1 wildcard dimension.
    auto infer_reshape(const Node* node) -> std::vector<SymbolicShape>;

    /// Infer output shape for reductions (Sum, Mean, Max, Min).
    auto infer_reduction(const Node* node) -> std::vector<SymbolicShape>;

    /// Infer output shape for Transpose by swapping two dimensions.
    auto infer_transpose(const Node* node) -> std::vector<SymbolicShape>;

    /// Infer output shape for Linear: (*, in_features) -> (*, out_features).
    auto infer_linear(const Node* node) -> std::vector<SymbolicShape>;

    /// Gather symbolic shapes from a node's inputs, falling back to concrete shapes.
    auto gather_input_shapes(const Node* node) -> std::vector<SymbolicShape>;
};

} // namespace jit
} // namespace tenzor
