/**
 * @file lazy_tensor.hpp
 * @brief LazyTensor: deferred execution with symbolic shape tracking
 *
 * LazyTensor records operations as a computation graph without executing them.
 * Actual computation is triggered on demand (materialize(), item(), data_ptr()).
 * This enables graph-level optimizations, symbolic shape analysis, and
 * backend-specific fusion before execution.
 *
 * @code
 * auto a = LazyTensor::from_tensor(some_tensor);
 * auto b = LazyTensor::from_tensor(other_tensor);
 * auto c = lazy::add(a, b);      // Records graph node, no computation
 * auto d = lazy::matmul(c, a);   // Records another node
 * Tensor result = d.materialize(); // Executes the full graph
 * @endcode
 */

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include "../core/tensor.hpp"
#include "../ops/op_id.hpp"
#include "../backend/op_attributes.hpp"

namespace tenzor {

// OpAttributes (`using OpAttributes = NewOpAttributes`) is provided by the
// included backend/op_attributes.hpp above.

namespace lazy {

// Forward declarations
class LazyGraph;
class LazyNode;

/**
 * @brief A deferred tensor that records operations instead of executing them.
 *
 * LazyTensor is a thin handle to a node in a computation graph. Operations
 * on LazyTensors build up the graph; actual execution is deferred until
 * materialize() is called. After materialization, the result is cached.
 */
class LazyTensor {
public:
    /// Create a LazyTensor wrapping an already-materialized tensor (graph input).
    static auto from_tensor(const Tensor& t) -> LazyTensor;

    /// Create an uninitialized LazyTensor with known shape/dtype/device (placeholder).
    static auto placeholder(std::vector<int64_t> shape, DType dtype, Device device,
                            const std::string& name = "") -> LazyTensor;

    /// Create a LazyTensor from a graph node (internal use).
    LazyTensor(std::shared_ptr<LazyNode> node, std::shared_ptr<LazyGraph> graph);

    /// Default/copy/move constructors
    LazyTensor() = default;
    LazyTensor(const LazyTensor&) = default;
    LazyTensor(LazyTensor&&) noexcept = default;
    LazyTensor& operator=(const LazyTensor&) = default;
    LazyTensor& operator=(LazyTensor&&) noexcept = default;

    /// Force execution of the computation graph and return the concrete tensor.
    /// Caches the result — subsequent calls return the cached value.
    auto materialize() const -> Tensor;

    /// Check if the tensor has been materialized.
    auto is_materialized() const -> bool;

    /// Get the shape (may be symbolic/estimated before materialization).
    auto shape() const -> std::vector<int64_t>;

    /// Get the number of dimensions.
    auto ndim() const -> int64_t;

    /// Get dtype.
    auto dtype() const -> DType;

    /// Get device.
    auto device() const -> Device;

    /// Get the underlying node.
    auto node() const -> std::shared_ptr<LazyNode>;

    /// Get the graph this tensor belongs to.
    auto graph() const -> std::shared_ptr<LazyGraph>;

private:
    std::shared_ptr<LazyNode> node_;
    std::shared_ptr<LazyGraph> graph_;
};

/**
 * @brief A node in the lazy computation graph.
 */
class LazyNode {
public:
    /// Input node (wraps an existing tensor)
    LazyNode(Tensor data);

    /// Placeholder node (shape/dtype/device known, no data yet)
    LazyNode(std::vector<int64_t> shape, DType dtype, Device device, std::string name);

    /// Computation node (op + inputs)
    LazyNode(OpId op, std::vector<std::shared_ptr<LazyNode>> inputs,
             std::vector<int64_t> output_shape, DType output_dtype, Device output_device);

    /// Computation node with op-specific attributes (preferred — attributes carry
    /// dim/shape/keepdim values required by the generic backend dispatch path).
    LazyNode(OpId op, std::vector<std::shared_ptr<LazyNode>> inputs,
             std::vector<int64_t> output_shape, DType output_dtype, Device output_device,
             OpAttributes attrs);

    auto op() const -> std::optional<OpId> { return op_; }
    auto inputs() const -> const std::vector<std::shared_ptr<LazyNode>>& { return inputs_; }
    auto attrs() const -> const OpAttributes& { return attrs_; }
    auto shape() const -> const std::vector<int64_t>& { return shape_; }
    auto dtype() const -> DType { return dtype_; }
    auto device() const -> Device { return device_; }
    auto name() const -> const std::string& { return name_; }

    /// Check if this node has a cached materialized result.
    auto is_materialized() const -> bool;

    /// Get the cached result (only valid if is_materialized()).
    auto materialized() const -> const Tensor&;

    /// Set the materialized result.
    void set_materialized(Tensor t);

    /// Check if this is an input node (wraps existing data).
    auto is_input() const -> bool { return is_input_; }

private:
    std::optional<OpId> op_;
    std::vector<std::shared_ptr<LazyNode>> inputs_;
    OpAttributes attrs_;
    std::vector<int64_t> shape_;
    DType dtype_;
    Device device_;
    std::string name_;
    bool is_input_ = false;

    mutable std::mutex materialize_mutex_;
    mutable std::optional<Tensor> materialized_;
};

/**
 * @brief The lazy computation graph — owns all nodes and handles execution.
 */
class LazyGraph {
public:
    LazyGraph() = default;

    /// Add an input node (wraps existing tensor).
    auto add_input(const Tensor& t) -> std::shared_ptr<LazyNode>;

    /// Add a placeholder node.
    auto add_placeholder(std::vector<int64_t> shape, DType dtype, Device device,
                         const std::string& name = "") -> std::shared_ptr<LazyNode>;

    /// Add a computation node (no attributes).
    auto add_node(OpId op, std::vector<std::shared_ptr<LazyNode>> inputs,
                  std::vector<int64_t> output_shape, DType output_dtype,
                  Device output_device) -> std::shared_ptr<LazyNode>;

    /// Add a computation node with op-specific attributes.
    auto add_node(OpId op, std::vector<std::shared_ptr<LazyNode>> inputs,
                  std::vector<int64_t> output_shape, DType output_dtype,
                  Device output_device, OpAttributes attrs) -> std::shared_ptr<LazyNode>;

    /// Attach a node that was created by another LazyGraph (used by merge_graphs).
    void adopt_node(std::shared_ptr<LazyNode> node);

    /// Execute the graph to materialize a specific node.
    auto execute(const std::shared_ptr<LazyNode>& target) -> Tensor;

    /// Run every unrealised node in topological order. Result is bit-identical
    /// to invoking the same ops eagerly (modulo non-determinism in the kernels
    /// themselves).
    void flush();

    /// Get all nodes in topological order.
    auto nodes() const -> const std::vector<std::shared_ptr<LazyNode>>& { return nodes_; }

    /// Get number of nodes.
    auto size() const -> size_t { return nodes_.size(); }

    /// Build a new LazyGraph whose nodes are the union of `a`'s and `b`'s nodes,
    /// preserving edges. Used to combine two independent lazy subgraphs into a
    /// single materialisation context.
    static auto merge_graphs(const std::shared_ptr<LazyGraph>& a,
                             const std::shared_ptr<LazyGraph>& b)
        -> std::shared_ptr<LazyGraph>;

private:
    std::vector<std::shared_ptr<LazyNode>> nodes_;

    /// Execute a single node given materialized inputs.
    auto execute_node(const std::shared_ptr<LazyNode>& node) -> Tensor;
};

// ============================================================================
// Lazy operations — record graph nodes, return LazyTensors
// ============================================================================

auto add(const LazyTensor& a, const LazyTensor& b) -> LazyTensor;
auto sub(const LazyTensor& a, const LazyTensor& b) -> LazyTensor;
auto mul(const LazyTensor& a, const LazyTensor& b) -> LazyTensor;
auto div(const LazyTensor& a, const LazyTensor& b) -> LazyTensor;
auto matmul(const LazyTensor& a, const LazyTensor& b) -> LazyTensor;
auto neg(const LazyTensor& a) -> LazyTensor;
auto relu(const LazyTensor& a) -> LazyTensor;
auto sigmoid(const LazyTensor& a) -> LazyTensor;
auto tanh(const LazyTensor& a) -> LazyTensor;
auto exp(const LazyTensor& a) -> LazyTensor;
auto log(const LazyTensor& a) -> LazyTensor;
auto sqrt(const LazyTensor& a) -> LazyTensor;
auto transpose(const LazyTensor& a, int64_t dim0, int64_t dim1) -> LazyTensor;
auto reshape(const LazyTensor& a, std::vector<int64_t> shape) -> LazyTensor;
auto sum(const LazyTensor& a, std::optional<int64_t> dim = std::nullopt) -> LazyTensor;
auto mean(const LazyTensor& a, std::optional<int64_t> dim = std::nullopt) -> LazyTensor;

} // namespace lazy
} // namespace tenzor
