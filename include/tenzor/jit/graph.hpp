/**
 * @file graph.hpp
 * @brief Intermediate representation (IR) graph for JIT compilation
 *
 * Provides a graph-based IR for representing neural network computations.
 * The graph consists of nodes (operations) and edges (data flow).
 * Supports graph construction, manipulation, optimization, and execution.
 */

#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include "../core/tensor.hpp"
#include "../autograd/variable.hpp"
#include "symbolic_shape.hpp"

namespace tenzor {
namespace jit {

// Forward declarations
class Node;
class Graph;
class Value;
enum class OpType;

/**
 * @brief Represents a value (tensor) in the IR graph.
 *
 * Values flow along edges between nodes. Each value has a unique ID,
 * type information, and a producing node.
 */
class Value {
public:
    /**
     * @brief Construct value.
     *
     * @param id Unique identifier
     * @param shape Tensor shape
     * @param dtype Data type
     * @param device Device location
     */
    Value(std::string id, std::vector<int64_t> shape, DType dtype, Device device)
        : id_(std::move(id)), shape_(std::move(shape)), dtype_(dtype), device_(device) {}

    /**
     * @brief Get value ID.
     *
     * @return Unique identifier
     */
    auto id() const -> const std::string& { return id_; }

    /**
     * @brief Get shape.
     *
     * @return Tensor dimensions
     */
    auto shape() const -> const std::vector<int64_t>& { return shape_; }

    /**
     * @brief Set shape (for type inference).
     *
     * @param new_shape New dimensions
     */
    auto set_shape(std::vector<int64_t> new_shape) -> void { shape_ = std::move(new_shape); }

    /**
     * @brief Get data type.
     *
     * @return DType
     */
    auto dtype() const -> DType { return dtype_; }

    /**
     * @brief Set data type (for dtype optimization pass).
     *
     * @param d New data type
     */
    auto set_dtype(DType d) -> void { dtype_ = d; }

    /**
     * @brief Get device.
     *
     * @return Device reference
     */
    auto device() const -> const Device& { return device_; }

    /**
     * @brief Get producing node.
     *
     * @return Node that produces this value (nullptr for inputs)
     */
    auto node() const -> std::shared_ptr<Node> { return node_.lock(); }

    /**
     * @brief Set producing node.
     *
     * @param n Node that produces this value
     */
    auto set_node(std::shared_ptr<Node> n) -> void { node_ = n; }

    /**
     * @brief Get all consuming nodes.
     *
     * @return Nodes that use this value as input
     */
    auto uses() const -> const std::vector<std::weak_ptr<Node>>& { return uses_; }

    /**
     * @brief Add a consuming node.
     *
     * @param n Node that uses this value
     */
    auto add_use(std::shared_ptr<Node> n) -> void { uses_.push_back(n); }

    /**
     * @brief Remove all uses (for dead code elimination).
     */
    auto clear_uses() -> void { uses_.clear(); }

    /**
     * @brief Remove a specific consuming node from the uses list.
     *
     * @param n Node to remove from uses
     */
    auto remove_use(const std::shared_ptr<Node>& n) -> void {
        uses_.erase(
            std::remove_if(uses_.begin(), uses_.end(),
                [&n](const std::weak_ptr<Node>& wp) {
                    auto sp = wp.lock();
                    return !sp || sp == n;
                }),
            uses_.end());
    }

    /**
     * @brief Get buffer ID for memory planning.
     *
     * @return Buffer pool index (SIZE_MAX if unassigned)
     */
    auto buffer_id() const -> size_t { return buffer_id_; }

    /**
     * @brief Set buffer ID for memory planning.
     *
     * @param id Buffer pool index
     */
    auto set_buffer_id(size_t id) -> void { buffer_id_ = id; }

    /**
     * @brief Get buffer offset for memory planning.
     *
     * @return Byte offset within the assigned buffer
     */
    auto buffer_offset() const -> size_t { return buffer_offset_; }

    /**
     * @brief Set buffer offset for memory planning.
     *
     * @param offset Byte offset within the assigned buffer
     */
    auto set_buffer_offset(size_t offset) -> void { buffer_offset_ = offset; }

    /**
     * @brief Check if this value has a memory plan assignment.
     *
     * @return true if buffer_id has been set
     */
    auto has_memory_plan() const -> bool { return buffer_id_ != SIZE_MAX; }

    // ========================================================================
    // Symbolic shape support
    // ========================================================================

    /**
     * @brief Get the symbolic shape.
     *
     * Returns the symbolic shape associated with this value. If no symbolic
     * shape has been set, returns an empty SymbolicShape.
     *
     * @return Symbolic shape
     */
    auto symbolic_shape() const -> const SymbolicShape& { return symbolic_shape_; }

    /**
     * @brief Set the symbolic shape.
     *
     * @param shape New symbolic shape
     */
    auto set_symbolic_shape(SymbolicShape shape) -> void {
        symbolic_shape_ = std::move(shape);
    }

    /**
     * @brief Check if this value has any symbolic dimensions.
     *
     * Returns true if a symbolic shape has been set and it contains
     * at least one symbolic (non-concrete) dimension.
     *
     * @return true if symbolic dimensions exist
     */
    auto has_symbolic_dims() const -> bool {
        return !symbolic_shape_.empty() && symbolic_shape_.has_symbolic_dims();
    }

    /**
     * @brief Check if a symbolic shape has been assigned.
     *
     * @return true if symbolic shape is non-empty
     */
    auto has_symbolic_shape() const -> bool {
        return !symbolic_shape_.empty();
    }

private:
    std::string id_;                              ///< Unique identifier
    std::vector<int64_t> shape_;                  ///< Tensor shape
    DType dtype_;                                 ///< Data type
    Device device_;                               ///< Device location
    std::weak_ptr<Node> node_;                    ///< Producing node
    std::vector<std::weak_ptr<Node>> uses_;       ///< Consuming nodes
    size_t buffer_id_{SIZE_MAX};                  ///< Memory plan buffer index
    size_t buffer_offset_{0};                     ///< Memory plan byte offset
    SymbolicShape symbolic_shape_;                ///< Symbolic shape (may be empty)
};

/**
 * @brief Represents an operation node in the IR graph.
 *
 * Each node corresponds to a single operation (e.g., Conv2d, Add, ReLU).
 * Nodes have typed inputs and outputs, and may have attributes
 * (e.g., kernel size, stride).
 */
class Node : public std::enable_shared_from_this<Node> {
public:
    /**
     * @brief Construct node.
     *
     * @param op_type Type of operation
     * @param name Node name (for debugging)
     */
    Node(OpType op_type, std::string name = "")
        : op_type_(op_type), name_(std::move(name)) {}

    /**
     * @brief Get operation type.
     *
     * @return OpType enum value
     */
    auto op_type() const -> OpType { return op_type_; }

    /**
     * @brief Get node name.
     *
     * @return Name string
     */
    auto name() const -> const std::string& { return name_; }

    /**
     * @brief Set node name.
     *
     * @param n New name
     */
    auto set_name(std::string n) -> void { name_ = std::move(n); }

    /**
     * @brief Get input values.
     *
     * @return Vector of input value pointers
     */
    auto inputs() const -> const std::vector<std::shared_ptr<Value>>& { return inputs_; }

    /**
     * @brief Get output values.
     *
     * @return Vector of output value pointers
     */
    auto outputs() const -> const std::vector<std::shared_ptr<Value>>& { return outputs_; }

    /**
     * @brief Add input value.
     *
     * @param val Input value
     */
    auto add_input(std::shared_ptr<Value> val) -> void;

    /**
     * @brief Add output value.
     *
     * @param val Output value
     */
    auto add_output(std::shared_ptr<Value> val) -> void;

    /**
     * @brief Replace input at index.
     *
     * @param idx Input index
     * @param val New input value
     */
    auto replace_input(size_t idx, std::shared_ptr<Value> val) -> void;

    /**
     * @brief Set float attribute.
     *
     * @param name Attribute name
     * @param value Float value
     */
    auto set_attr(const std::string& name, float value) -> void {
        attrs_[name] = value;
    }

    /**
     * @brief Set int attribute.
     *
     * @param name Attribute name
     * @param value Int value
     */
    auto set_int_attr(const std::string& name, int64_t value) -> void {
        int_attrs_[name] = value;
    }

    /**
     * @brief Set vector attribute.
     *
     * @param name Attribute name
     * @param value Vector value
     */
    auto set_vec_attr(const std::string& name, std::vector<int64_t> value) -> void {
        vec_attrs_[name] = std::move(value);
    }

    /**
     * @brief Set boolean attribute.
     *
     * @param name Attribute name
     * @param value Bool value
     */
    auto set_bool_attr(const std::string& name, bool value) -> void {
        bool_attrs_[name] = value;
    }

    /**
     * @brief Set tensor attribute (for constants/weights).
     *
     * @param name Attribute name
     * @param value Tensor value
     */
    auto set_tensor_attr(const std::string& name, Tensor value) -> void {
        tensor_attrs_[name] = std::move(value);
    }

    /**
     * @brief Get float attribute.
     *
     * @param name Attribute name
     * @return Attribute value (0.0 if not found)
     */
    auto get_attr(const std::string& name) const -> float {
        auto it = attrs_.find(name);
        return it != attrs_.end() ? it->second : 0.0f;
    }

    /**
     * @brief Get int attribute.
     *
     * @param name Attribute name
     * @return Attribute value (0 if not found)
     */
    auto get_int_attr(const std::string& name) const -> int64_t {
        auto it = int_attrs_.find(name);
        return it != int_attrs_.end() ? it->second : 0;
    }

    /**
     * @brief Get vector attribute.
     *
     * @param name Attribute name
     * @return Attribute value (empty if not found)
     */
    auto get_vec_attr(const std::string& name) const -> std::vector<int64_t> {
        auto it = vec_attrs_.find(name);
        return it != vec_attrs_.end() ? it->second : std::vector<int64_t>{};
    }

    /**
     * @brief Get boolean attribute.
     *
     * @param name Attribute name
     * @return Attribute value (false if not found)
     */
    auto get_bool_attr(const std::string& name) const -> bool {
        auto it = bool_attrs_.find(name);
        return it != bool_attrs_.end() ? it->second : false;
    }

    /**
     * @brief Get tensor attribute.
     *
     * @param name Attribute name
     * @return Attribute tensor (may be empty)
     */
    auto get_tensor_attr(const std::string& name) const -> const Tensor& {
        static Tensor empty;
        auto it = tensor_attrs_.find(name);
        return it != tensor_attrs_.end() ? it->second : empty;
    }

    /**
     * @brief Check if attribute exists.
     *
     * @param name Attribute name
     * @return true if attribute is set
     */
    auto has_attr(const std::string& name) const -> bool {
        return attrs_.count(name) > 0 || int_attrs_.count(name) > 0 ||
               vec_attrs_.count(name) > 0 || bool_attrs_.count(name) > 0 ||
               tensor_attrs_.count(name) > 0;
    }

    /**
     * @brief Check if a specific *typed* attribute is present.
     *
     * The MLIR lowering needs to distinguish between "kernel_size set as
     * a scalar int" (the pooling-layer record path) and "kernel_size set
     * as a [H, W] vector" (the dispatch-interceptor copy_hw_pair path).
     * `has_attr` doesn't disambiguate; these per-map probes do.
     */
    auto has_int_attr(const std::string& name) const -> bool {
        return int_attrs_.count(name) > 0;
    }
    auto has_vec_attr(const std::string& name) const -> bool {
        return vec_attrs_.count(name) > 0;
    }
    auto has_bool_attr(const std::string& name) const -> bool {
        return bool_attrs_.count(name) > 0;
    }
    auto has_float_attr(const std::string& name) const -> bool {
        return attrs_.count(name) > 0;
    }
    auto has_tensor_attr(const std::string& name) const -> bool {
        return tensor_attrs_.count(name) > 0;
    }

    // ========================================================================
    // Subgraph support (for control flow: If, Loop)
    // ========================================================================

    /**
     * @brief Set the "then" branch subgraph (for If nodes).
     */
    auto set_then_branch(std::shared_ptr<Graph> g) -> void { then_branch_ = std::move(g); }

    /**
     * @brief Set the "else" branch subgraph (for If nodes).
     */
    auto set_else_branch(std::shared_ptr<Graph> g) -> void { else_branch_ = std::move(g); }

    /**
     * @brief Set the loop body subgraph (for Loop nodes).
     */
    auto set_body(std::shared_ptr<Graph> g) -> void { body_ = std::move(g); }

    /**
     * @brief Get the "then" branch subgraph.
     */
    auto then_branch() const -> const std::shared_ptr<Graph>& { return then_branch_; }

    /**
     * @brief Get the "else" branch subgraph.
     */
    auto else_branch() const -> const std::shared_ptr<Graph>& { return else_branch_; }

    /**
     * @brief Get the loop body subgraph.
     */
    auto body() const -> const std::shared_ptr<Graph>& { return body_; }

    /**
     * @brief Get all attribute maps (for optimization passes).
     *
     * @return References to attribute storage
     */
    auto get_all_attrs() -> std::tuple<
        std::unordered_map<std::string, float>&,
        std::unordered_map<std::string, int64_t>&,
        std::unordered_map<std::string, std::vector<int64_t>>&,
        std::unordered_map<std::string, bool>&,
        std::unordered_map<std::string, Tensor>&> {
        return {attrs_, int_attrs_, vec_attrs_, bool_attrs_, tensor_attrs_};
    }

private:
    OpType op_type_;                                        ///< Operation type
    std::string name_;                                      ///< Node name
    std::vector<std::shared_ptr<Value>> inputs_;            ///< Input values
    std::vector<std::shared_ptr<Value>> outputs_;           ///< Output values
    std::unordered_map<std::string, float> attrs_;          ///< Float attributes
    std::unordered_map<std::string, int64_t> int_attrs_;    ///< Int attributes
    std::unordered_map<std::string, std::vector<int64_t>> vec_attrs_;  ///< Vector attributes
    std::unordered_map<std::string, bool> bool_attrs_;      ///< Boolean attributes
    std::unordered_map<std::string, Tensor> tensor_attrs_;  ///< Tensor constants
    std::shared_ptr<Graph> then_branch_;                    ///< If: then branch
    std::shared_ptr<Graph> else_branch_;                    ///< If: else branch
    std::shared_ptr<Graph> body_;                           ///< Loop: body graph
};

/**
 * @brief IR graph representing a complete computation.
 *
 * The graph contains nodes (operations) connected by values (tensors).
 * It supports:
 * - Graph construction from traced operations
 * - Type inference and shape propagation
 * - Optimization passes (fusion, DCE, CSE)
 * - Execution with runtime inputs
 * - Serialization to disk
 *
 * @code
 * auto graph = trace(model, dummy_input);
 * graph->optimize();  // Run optimization passes
 * graph->save("model.pt");
 * auto loaded = Graph::load("model.pt");
 * Variable output = loaded->forward({input});
 * @endcode
 */
class Graph {
public:
    /**
     * @brief Construct empty graph.
     */
    Graph() = default;

    /**
     * @brief Create node in graph.
     *
     * @param op_type Operation type
     * @param name Node name (auto-generated if empty)
     * @return New node
     */
    auto create_node(OpType op_type, const std::string& name = "") -> std::shared_ptr<Node>;

    /**
     * @brief Create value in graph.
     *
     * @param id Value ID
     * @param shape Tensor shape
     * @param dtype Data type
     * @param device Device location
     * @return New value
     */
    auto create_value(const std::string& id, std::vector<int64_t> shape,
                      DType dtype, Device device) -> std::shared_ptr<Value>;

    /**
     * @brief Get value by ID.
     *
     * @param id Value identifier
     * @return Value pointer (nullptr if not found)
     */
    auto get_value(const std::string& id) const -> std::shared_ptr<Value>;

    /**
     * @brief Get all nodes in topological order.
     *
     * @return Vector of nodes
     */
    auto nodes() const -> const std::vector<std::shared_ptr<Node>>& { return nodes_; }

    /**
     * @brief Get input values.
     *
     * @return Vector of graph inputs
     */
    auto inputs() const -> const std::vector<std::shared_ptr<Value>>& { return inputs_; }

    /**
     * @brief Get output values.
     *
     * @return Vector of graph outputs
     */
    auto outputs() const -> const std::vector<std::shared_ptr<Value>>& { return outputs_; }

    /**
     * @brief Set graph inputs.
     *
     * @param ins Input values
     */
    auto set_inputs(std::vector<std::shared_ptr<Value>> ins) -> void {
        inputs_ = std::move(ins);
    }

    /**
     * @brief Set graph outputs.
     *
     * @param outs Output values
     */
    auto set_outputs(std::vector<std::shared_ptr<Value>> outs) -> void {
        outputs_ = std::move(outs);
    }

    /**
     * @brief Add node to graph.
     *
     * @param node Node to add
     */
    auto add_node(std::shared_ptr<Node> node) -> void;

    /**
     * @brief Remove node from graph.
     *
     * Removes the node from the nodes list and removes the node from
     * each input value's uses list (without clearing other uses).
     *
     * @param node Node to remove
     */
    auto remove_node(std::shared_ptr<Node> node) -> void;

    /**
     * @brief Replace all uses of one value with another.
     *
     * For every node that has old_id in its inputs, replaces it with
     * the value for new_id. Updates the uses lists accordingly.
     *
     * @param old_id ID of value to replace
     * @param new_id ID of replacement value
     */
    auto replace_value(const std::string& old_id, const std::string& new_id) -> void;

    /**
     * @brief Replace a node with another node.
     *
     * Wires all consumers of old_node's outputs to use new_node's
     * outputs instead. The old_node is then removed from the graph.
     *
     * @param old_node Node to replace
     * @param new_node Replacement node
     */
    auto replace_node(std::shared_ptr<Node> old_node, std::shared_ptr<Node> new_node) -> void;

    /**
     * @brief Replace a node with an existing value (identity bypass).
     *
     * Redirects all consumers of the node's first output to use the
     * given value instead. Used for simplifications like x+0 -> x.
     *
     * @param node Node to bypass
     * @param value_id ID of the value to use instead
     */
    auto replace_node_with_value(std::shared_ptr<Node> node, const std::string& value_id) -> void;

    /**
     * @brief Perform topological sort of nodes.
     *
     * Orders nodes such that dependencies come before dependents.
     */
    auto topological_sort() -> void;

    /**
     * @brief Run type inference pass.
     *
     * Propagates shape and dtype information through the graph.
     */
    auto infer_types() -> void;

    /**
     * @brief Set symbolic shape for a graph input.
     *
     * Marks specific dimensions of an input value as symbolic. This is used
     * to indicate that certain dimensions (e.g., batch size, sequence length)
     * are dynamic and may vary between executions.
     *
     * @param input_idx Index of the input value (0-based)
     * @param shape Symbolic shape to assign
     * @throws std::out_of_range if input_idx is out of bounds
     *
     * @code
     * // Mark first input as having dynamic batch dimension
     * SymbolicShape shape = {
     *     SymbolicDim::symbolic("batch"),
     *     SymbolicDim::concrete(3),
     *     SymbolicDim::concrete(224),
     *     SymbolicDim::concrete(224)
     * };
     * graph->set_symbolic_input_shape(0, shape);
     * graph->infer_symbolic_types();
     * @endcode
     */
    auto set_symbolic_input_shape(size_t input_idx, SymbolicShape shape) -> void;

    /**
     * @brief Run symbolic type inference pass.
     *
     * Propagates symbolic shape information through the graph, using
     * the same operation-specific rules as infer_types() but operating
     * on SymbolicShape instead of concrete shapes. Should be called
     * after set_symbolic_input_shape() to propagate symbolic dimensions
     * through the entire graph.
     */
    auto infer_symbolic_types() -> void;

    /**
     * @brief Bind symbolic dimensions to concrete values using an environment.
     *
     * Walks all values in the graph and resolves any symbolic dimensions
     * to concrete values using the provided environment. After binding,
     * the graph can be executed with the concrete shapes.
     *
     * @param env Environment mapping symbolic names to concrete values
     * @throws std::runtime_error if any symbolic dimension is not bound
     */
    auto bind_symbolic_shapes(const SymbolicShapeEnvironment& env) -> void;

    /**
     * @brief Get the symbolic input shapes (if set).
     *
     * @return Vector of symbolic shapes for each input (empty if no symbolic shapes set)
     */
    auto symbolic_input_shapes() const -> std::vector<SymbolicShape>;

    /**
     * @brief Execute graph with runtime inputs.
     *
     * @param runtime_inputs Input tensors/variables
     * @return Output tensors/variables
     */
    auto forward(const std::vector<Variable>& runtime_inputs) -> std::vector<Variable>;

    /**
     * @brief Save graph to file.
     *
     * @param path Output file path
     */
    auto save(const std::string& path) const -> void;

    /**
     * @brief Load graph from file.
     *
     * @param path Input file path
     * @return Loaded graph
     */
    static auto load(const std::string& path) -> std::shared_ptr<Graph>;

    /**
     * @brief Print graph structure (for debugging).
     *
     * @return String representation
     */
    auto to_string() const -> std::string;

    /**
     * @brief Get number of nodes.
     *
     * @return Node count
     */
    auto num_nodes() const -> size_t { return nodes_.size(); }

    /**
     * @brief Get number of values.
     *
     * @return Value count
     */
    auto num_values() const -> size_t { return values_.size(); }

    /**
     * @brief Partition graph at specified node indices.
     *
     * Splits the graph into sub-graphs at the given break points.
     * Each sub-graph is a contiguous sequence of nodes that can be
     * compiled independently.
     *
     * @param break_indices Node indices where breaks occur
     * @return Vector of sub-graphs (one per partition)
     */
    auto partition_at(const std::vector<size_t>& break_indices)
        -> std::vector<std::shared_ptr<Graph>>;

    /**
     * @brief Find all nodes matching a given operation type name.
     *
     * Searches the graph for nodes whose OpType string representation
     * matches the given type name.
     *
     * @param type_name Operation type name (e.g., "Conv2d", "ReLU")
     * @return Vector of matching nodes
     */
    auto find_nodes_by_type(const std::string& type_name) const -> std::vector<std::shared_ptr<Node>>;

    /**
     * @brief Record a captured parameter / constant tensor.
     *
     * Called by the tracer during end_trace() for every tensor that
     * appears as an op input but is not a graph input. These tensors
     * (typically nn::Module parameters) are pre-populated into the
     * runtime value_map during Graph::forward.
     */
    auto set_constant(const std::string& value_id, const Tensor& tensor) -> void {
        constants_[value_id] = tensor;
    }

    /// Read captured constants map.
    auto constants() const -> const std::unordered_map<std::string, Tensor>& {
        return constants_;
    }

private:
    std::vector<std::shared_ptr<Node>> nodes_;              ///< All nodes (topologically sorted)
    std::unordered_map<std::string, std::shared_ptr<Value>> values_;  ///< ID -> Value map
    std::vector<std::shared_ptr<Value>> inputs_;            ///< Graph inputs
    std::vector<std::shared_ptr<Value>> outputs_;           ///< Graph outputs
    /// Captured parameter / constant tensors, keyed by Value ID. These
    /// are Tensors that were read by traced ops but were not graph
    /// inputs (e.g. Linear's weight/bias). During Graph::forward they
    /// are pre-populated into the runtime value_map before any node
    /// executes.
    std::unordered_map<std::string, Tensor> constants_;
    int64_t next_node_id_{0};                               ///< Node ID counter
    mutable bool needs_retrace_{false};                        ///< Set when ShapeGuard detects mismatch

public:
    /// Check if a shape guard triggered a retrace request
    auto needs_retrace() const -> bool { return needs_retrace_; }
    /// Reset the retrace flag
    auto reset_retrace() -> void { needs_retrace_ = false; }

private:
    /**
     * @brief Execute a single node.
     *
     * @param node Node to execute
     * @param value_map Runtime value map
     */
    auto execute_node(const std::shared_ptr<Node>& node,
                      std::unordered_map<std::string, Variable>& value_map) const -> void;

    friend class Compiler;  // Allow compiler to access internals
};

} // namespace jit
} // namespace tenzor
