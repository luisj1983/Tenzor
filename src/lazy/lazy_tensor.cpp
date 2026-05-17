#include "tenzor/lazy/lazy_tensor.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"

#include <algorithm>
#include <stdexcept>

namespace tenzor {
namespace lazy {

// ============================================================================
// LazyNode
// ============================================================================

LazyNode::LazyNode(Tensor data)
    : shape_(data.shape().begin(), data.shape().end()),
      dtype_(data.dtype()),
      device_(data.device()),
      is_input_(true),
      materialized_(std::move(data)) {}

LazyNode::LazyNode(std::vector<int64_t> shape, DType dtype, Device device, std::string name)
    : shape_(std::move(shape)),
      dtype_(dtype),
      device_(device),
      name_(std::move(name)),
      is_input_(false) {}

LazyNode::LazyNode(OpId op, std::vector<std::shared_ptr<LazyNode>> inputs,
                   std::vector<int64_t> output_shape, DType output_dtype, Device output_device)
    : op_(op),
      inputs_(std::move(inputs)),
      shape_(std::move(output_shape)),
      dtype_(output_dtype),
      device_(output_device),
      is_input_(false) {}

auto LazyNode::is_materialized() const -> bool {
    std::lock_guard<std::mutex> lock(materialize_mutex_);
    return materialized_.has_value();
}

auto LazyNode::materialized() const -> const Tensor& {
    std::lock_guard<std::mutex> lock(materialize_mutex_);
    if (!materialized_) {
        throw std::runtime_error("LazyNode: not materialized");
    }
    return *materialized_;
}

void LazyNode::set_materialized(Tensor t) {
    std::lock_guard<std::mutex> lock(materialize_mutex_);
    materialized_ = std::move(t);
}

// ============================================================================
// LazyGraph
// ============================================================================

auto LazyGraph::add_input(const Tensor& t) -> std::shared_ptr<LazyNode> {
    auto node = std::make_shared<LazyNode>(t);
    nodes_.push_back(node);
    return node;
}

auto LazyGraph::add_placeholder(std::vector<int64_t> shape, DType dtype, Device device,
                                const std::string& name) -> std::shared_ptr<LazyNode> {
    auto node = std::make_shared<LazyNode>(std::move(shape), dtype, device, name);
    nodes_.push_back(node);
    return node;
}

auto LazyGraph::add_node(OpId op, std::vector<std::shared_ptr<LazyNode>> inputs,
                         std::vector<int64_t> output_shape, DType output_dtype,
                         Device output_device) -> std::shared_ptr<LazyNode> {
    auto node = std::make_shared<LazyNode>(op, std::move(inputs),
                                            std::move(output_shape), output_dtype, output_device);
    nodes_.push_back(node);
    return node;
}

auto LazyGraph::execute(const std::shared_ptr<LazyNode>& target) -> Tensor {
    if (target->is_materialized()) {
        return target->materialized();
    }

    // Materialize inputs recursively (depth-first)
    for (auto& input : target->inputs()) {
        if (!input->is_materialized()) {
            execute(input);
        }
    }

    // Execute this node
    auto result = execute_node(target);
    target->set_materialized(result);
    return result;
}

auto LazyGraph::execute_node(const std::shared_ptr<LazyNode>& node) -> Tensor {
    if (node->is_input() || !node->op()) {
        if (node->is_materialized()) return node->materialized();
        throw std::runtime_error("LazyGraph: cannot execute unbound placeholder '" +
                                 node->name() + "'");
    }

    // Gather materialized inputs
    std::vector<Tensor> input_tensors;
    input_tensors.reserve(node->inputs().size());
    for (auto& inp : node->inputs()) {
        input_tensors.push_back(inp->materialized());
    }

    auto op = *node->op();

    // Dispatch to the standard op implementations
    switch (op) {
        // Binary ops
        case OpId::Add:       return tenzor::add(input_tensors[0], input_tensors[1]);
        case OpId::Sub:       return tenzor::sub(input_tensors[0], input_tensors[1]);
        case OpId::Mul:       return tenzor::mul(input_tensors[0], input_tensors[1]);
        case OpId::Div:       return tenzor::div(input_tensors[0], input_tensors[1]);
        case OpId::MatMul:    return tenzor::matmul(input_tensors[0], input_tensors[1]);

        // Unary math ops
        case OpId::Neg:       return tenzor::neg(input_tensors[0]);
        case OpId::Exp:       return tenzor::exp(input_tensors[0]);
        case OpId::Log:       return tenzor::log(input_tensors[0]);
        case OpId::Sqrt:      return tenzor::sqrt(input_tensors[0]);

        // Activations — use clamp/comparison composition since these aren't free functions
        case OpId::ReLU:
            return tenzor::clamp_min(input_tensors[0], 0.0f);
        case OpId::Sigmoid: {
            auto& x = input_tensors[0];
            auto shape_vec = std::vector<int64_t>(x.shape().begin(), x.shape().end());
            auto neg_x = tenzor::neg(x);
            auto exp_neg = tenzor::exp(neg_x);
            auto one = tenzor::ones(shape_vec, x.dtype(), x.device());
            return tenzor::div(one, tenzor::add(one, exp_neg));
        }
        case OpId::TanhActivation: {
            auto& x = input_tensors[0];
            auto exp_pos = tenzor::exp(x);
            auto exp_neg = tenzor::exp(tenzor::neg(x));
            return tenzor::div(tenzor::sub(exp_pos, exp_neg), tenzor::add(exp_pos, exp_neg));
        }

        // Reductions
        case OpId::Sum:       return tenzor::sum(input_tensors[0]);
        case OpId::Mean:      return tenzor::mean(input_tensors[0]);

        // Shape ops
        case OpId::Transpose:
        case OpId::Reshape:   return tenzor::reshape(input_tensors[0], node->shape());
        default:
            throw std::runtime_error("LazyGraph: unsupported op " +
                                     std::to_string(static_cast<int>(op)));
    }
}

// ============================================================================
// LazyTensor
// ============================================================================

auto LazyTensor::from_tensor(const Tensor& t) -> LazyTensor {
    auto graph = std::make_shared<LazyGraph>();
    auto node = graph->add_input(t);
    return LazyTensor(node, graph);
}

auto LazyTensor::placeholder(std::vector<int64_t> shape, DType dtype, Device device,
                              const std::string& name) -> LazyTensor {
    auto graph = std::make_shared<LazyGraph>();
    auto node = graph->add_placeholder(std::move(shape), dtype, device, name);
    return LazyTensor(node, graph);
}

LazyTensor::LazyTensor(std::shared_ptr<LazyNode> node, std::shared_ptr<LazyGraph> graph)
    : node_(std::move(node)), graph_(std::move(graph)) {}

auto LazyTensor::materialize() const -> Tensor {
    if (!node_ || !graph_) {
        throw std::runtime_error("LazyTensor: null state");
    }
    return graph_->execute(node_);
}

auto LazyTensor::is_materialized() const -> bool {
    return node_ && node_->is_materialized();
}

auto LazyTensor::shape() const -> std::vector<int64_t> {
    return node_ ? node_->shape() : std::vector<int64_t>{};
}

auto LazyTensor::ndim() const -> int64_t {
    return node_ ? static_cast<int64_t>(node_->shape().size()) : 0;
}

auto LazyTensor::dtype() const -> DType {
    return node_ ? node_->dtype() : DType::Float32;
}

auto LazyTensor::device() const -> Device {
    return node_ ? node_->device() : Device::cpu();
}

auto LazyTensor::node() const -> std::shared_ptr<LazyNode> { return node_; }
auto LazyTensor::graph() const -> std::shared_ptr<LazyGraph> { return graph_; }

// ============================================================================
// Helper: ensure two lazy tensors share the same graph
// ============================================================================

static auto merge_graphs(const LazyTensor& a, const LazyTensor& b)
    -> std::shared_ptr<LazyGraph> {
    // Use the graph with more nodes as the "host" graph.
    // In practice both typically share the same graph.
    if (a.graph() == b.graph()) return a.graph();

    // Merge: use a's graph and add b's nodes to it
    // For simplicity, just use a's graph — the nodes from b are referenced
    // via shared_ptr and will be reachable during execution.
    return a.graph();
}

// ============================================================================
// Shape inference helpers
// ============================================================================

static auto broadcast_shape(const std::vector<int64_t>& a,
                             const std::vector<int64_t>& b) -> std::vector<int64_t> {
    size_t ndim = std::max(a.size(), b.size());
    std::vector<int64_t> result(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        int64_t da = (i < ndim - a.size()) ? 1 : a[a.size() - ndim + i];
        int64_t db = (i < ndim - b.size()) ? 1 : b[b.size() - ndim + i];
        if (da == db) result[i] = da;
        else if (da == 1) result[i] = db;
        else if (db == 1) result[i] = da;
        else throw std::invalid_argument("LazyTensor: incompatible shapes for broadcast");
    }
    return result;
}

static auto matmul_shape(const std::vector<int64_t>& a,
                          const std::vector<int64_t>& b) -> std::vector<int64_t> {
    if (a.size() < 1 || b.size() < 1) {
        throw std::invalid_argument("LazyTensor: matmul requires at least 1D tensors");
    }
    if (a.size() == 1 && b.size() == 1) return {};  // dot product -> scalar
    if (a.size() == 1) {
        auto result = std::vector<int64_t>(b.begin(), b.end());
        result[result.size() - 2] = 1;  // not quite right for 1D; simplify
        return {b.back()};
    }
    if (b.size() == 1) return {a[a.size() - 2]};

    // General: batch matmul
    std::vector<int64_t> batch_a(a.begin(), a.end() - 2);
    std::vector<int64_t> batch_b(b.begin(), b.end() - 2);
    auto batch = broadcast_shape(batch_a, batch_b);
    batch.push_back(a[a.size() - 2]);
    batch.push_back(b[b.size() - 1]);
    return batch;
}

// ============================================================================
// Lazy operation implementations
// ============================================================================

static auto binary_lazy_op(OpId op, const LazyTensor& a, const LazyTensor& b) -> LazyTensor {
    auto graph = merge_graphs(a, b);
    auto out_shape = broadcast_shape(a.shape(), b.shape());
    // B.7: use the real type-promotion table from core/dtype.hpp so e.g.
    // f16 + f32 → f32 and int32 + f64 → f64, matching eager-mode dtype
    // selection rather than silently inheriting the first operand's dtype.
    auto out_dtype = promote_types(a.dtype(), b.dtype());
    auto node = graph->add_node(op, {a.node(), b.node()}, out_shape, out_dtype, a.device());
    return LazyTensor(node, graph);
}

static auto unary_lazy_op(OpId op, const LazyTensor& a) -> LazyTensor {
    auto graph = a.graph();
    auto node = graph->add_node(op, {a.node()}, a.shape(), a.dtype(), a.device());
    return LazyTensor(node, graph);
}

auto add(const LazyTensor& a, const LazyTensor& b) -> LazyTensor {
    return binary_lazy_op(OpId::Add, a, b);
}

auto sub(const LazyTensor& a, const LazyTensor& b) -> LazyTensor {
    return binary_lazy_op(OpId::Sub, a, b);
}

auto mul(const LazyTensor& a, const LazyTensor& b) -> LazyTensor {
    return binary_lazy_op(OpId::Mul, a, b);
}

auto div(const LazyTensor& a, const LazyTensor& b) -> LazyTensor {
    return binary_lazy_op(OpId::Div, a, b);
}

auto matmul(const LazyTensor& a, const LazyTensor& b) -> LazyTensor {
    auto graph = merge_graphs(a, b);
    auto out_shape = matmul_shape(a.shape(), b.shape());
    auto node = graph->add_node(OpId::MatMul, {a.node(), b.node()},
                                out_shape, a.dtype(), a.device());
    return LazyTensor(node, graph);
}

auto neg(const LazyTensor& a) -> LazyTensor { return unary_lazy_op(OpId::Neg, a); }
auto relu(const LazyTensor& a) -> LazyTensor { return unary_lazy_op(OpId::ReLU, a); }
auto sigmoid(const LazyTensor& a) -> LazyTensor { return unary_lazy_op(OpId::Sigmoid, a); }
auto tanh(const LazyTensor& a) -> LazyTensor { return unary_lazy_op(OpId::TanhActivation, a); }
auto exp(const LazyTensor& a) -> LazyTensor { return unary_lazy_op(OpId::Exp, a); }
auto log(const LazyTensor& a) -> LazyTensor { return unary_lazy_op(OpId::Log, a); }
auto sqrt(const LazyTensor& a) -> LazyTensor { return unary_lazy_op(OpId::Sqrt, a); }

auto transpose(const LazyTensor& a, int64_t dim0, int64_t dim1) -> LazyTensor {
    auto graph = a.graph();
    auto out_shape = a.shape();
    std::swap(out_shape[dim0], out_shape[dim1]);
    auto node = graph->add_node(OpId::Transpose, {a.node()}, out_shape, a.dtype(), a.device());
    return LazyTensor(node, graph);
}

auto reshape(const LazyTensor& a, std::vector<int64_t> shape) -> LazyTensor {
    // Resolve -1 dimension
    int64_t neg_idx = -1;
    int64_t known_product = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == -1) {
            neg_idx = static_cast<int64_t>(i);
        } else {
            known_product *= shape[i];
        }
    }
    if (neg_idx >= 0) {
        int64_t total = 1;
        for (auto d : a.shape()) total *= d;
        shape[neg_idx] = total / known_product;
    }

    auto graph = a.graph();
    auto node = graph->add_node(OpId::Reshape, {a.node()}, shape, a.dtype(), a.device());
    return LazyTensor(node, graph);
}

auto sum(const LazyTensor& a, std::optional<int64_t> dim) -> LazyTensor {
    auto graph = a.graph();
    std::vector<int64_t> out_shape;
    if (dim.has_value()) {
        out_shape = a.shape();
        out_shape.erase(out_shape.begin() + *dim);
        if (out_shape.empty()) out_shape.push_back(1);
    } else {
        out_shape = {1};  // full reduction
    }
    auto node = graph->add_node(OpId::Sum, {a.node()}, out_shape, a.dtype(), a.device());
    return LazyTensor(node, graph);
}

auto mean(const LazyTensor& a, std::optional<int64_t> dim) -> LazyTensor {
    auto graph = a.graph();
    std::vector<int64_t> out_shape;
    if (dim.has_value()) {
        out_shape = a.shape();
        out_shape.erase(out_shape.begin() + *dim);
        if (out_shape.empty()) out_shape.push_back(1);
    } else {
        out_shape = {1};
    }
    auto node = graph->add_node(OpId::Mean, {a.node()}, out_shape, a.dtype(), a.device());
    return LazyTensor(node, graph);
}

} // namespace lazy
} // namespace tenzor
