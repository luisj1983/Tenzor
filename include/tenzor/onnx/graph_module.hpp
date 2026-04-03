/**
 * @file graph_module.hpp
 * @brief DAG-aware module for ONNX model execution
 *
 * Unlike nn::Sequential which only supports linear chains, GraphModule
 * can represent and execute arbitrary directed acyclic graphs (DAGs).
 * This supports residual connections, skip connections, and multi-input
 * multi-output architectures common in ONNX models.
 */

#pragma once

#include "tenzor/nn/module.hpp"
#include "tenzor/autograd/variable.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <memory>

namespace tenzor {
namespace onnx {

/**
 * @brief Describes a single operation in the computation graph.
 */
struct GraphOp {
    std::string name;                  ///< Unique node name
    std::string op_type;               ///< Operation type (e.g., "Add", "Conv")
    std::vector<std::string> inputs;   ///< Input value names
    std::vector<std::string> outputs;  ///< Output value names

    /// For stateless layers (Conv, Linear, etc.) — the nn::Module to call
    std::shared_ptr<nn::Module> module;

    /// For stateful/functional ops (Add, Reshape, etc.) — a lambda that
    /// computes outputs from inputs
    std::function<std::vector<Tensor>(const std::vector<Tensor>&)> compute_fn;
};

/**
 * @brief DAG-aware module that executes operations in topological order.
 *
 * Stores a list of GraphOps and their dependencies (via named values).
 * On forward(), executes ops in registered order (assumed topological),
 * maintaining a value map for intermediate results.
 *
 * Supports:
 * - Skip connections (values used by multiple later ops)
 * - Multiple graph inputs and outputs
 * - Mixed module-based and functional operations
 */
class GraphModule : public nn::Module {
public:
    GraphModule() = default;

    /**
     * @brief Add a graph operation.
     * Operations must be added in topological order.
     */
    void add_op(GraphOp op);

    /**
     * @brief Register a constant value (e.g., initialized weights).
     */
    void add_constant(const std::string& name, Tensor value);

    /**
     * @brief Set the graph input name(s).
     * These are populated from the forward() input before execution.
     */
    void set_input_names(std::vector<std::string> names);

    /**
     * @brief Set the graph output name(s).
     * These are collected after execution as the forward() result.
     */
    void set_output_names(std::vector<std::string> names);

    /**
     * @brief Execute the graph.
     * Input variable's tensor is bound to the first input name.
     * Returns the first output value.
     */
    auto forward_impl(const Variable& input) -> Variable override;

    auto name() const -> std::string { return "GraphModule"; }

    auto num_ops() const -> size_t { return ops_.size(); }

private:
    std::vector<GraphOp> ops_;
    std::unordered_map<std::string, Tensor> constants_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

} // namespace onnx
} // namespace tenzor
