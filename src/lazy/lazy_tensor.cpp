#include "tenzor/lazy/lazy_tensor.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

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

LazyNode::LazyNode(OpId op, std::vector<std::shared_ptr<LazyNode>> inputs,
                   std::vector<int64_t> output_shape, DType output_dtype, Device output_device,
                   OpAttributes attrs)
    : op_(op),
      inputs_(std::move(inputs)),
      attrs_(std::move(attrs)),
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

auto LazyGraph::add_node(OpId op, std::vector<std::shared_ptr<LazyNode>> inputs,
                         std::vector<int64_t> output_shape, DType output_dtype,
                         Device output_device, OpAttributes attrs)
    -> std::shared_ptr<LazyNode> {
    auto node = std::make_shared<LazyNode>(op, std::move(inputs),
                                            std::move(output_shape), output_dtype, output_device,
                                            std::move(attrs));
    nodes_.push_back(node);
    return node;
}

void LazyGraph::adopt_node(std::shared_ptr<LazyNode> node) {
    nodes_.push_back(std::move(node));
}

auto LazyGraph::merge_graphs(const std::shared_ptr<LazyGraph>& a,
                              const std::shared_ptr<LazyGraph>& b)
    -> std::shared_ptr<LazyGraph> {
    if (a == b) return a;
    auto merged = std::make_shared<LazyGraph>();
    std::unordered_set<LazyNode*> seen;
    seen.reserve(a->nodes_.size() + b->nodes_.size());

    auto absorb = [&](const std::shared_ptr<LazyGraph>& g) {
        for (const auto& n : g->nodes_) {
            if (seen.insert(n.get()).second) {
                // Edges are preserved automatically: each LazyNode keeps
                // shared_ptr references to its input nodes, so we simply add
                // every node once to the merged graph and the topology
                // survives intact.
                merged->nodes_.push_back(n);
            }
        }
    };
    absorb(a);
    absorb(b);
    return merged;
}

void LazyGraph::flush() {
    // Materialise every node in topological order. Because add_node /
    // add_input append in dependency order (an input was added before any
    // node that references it), iterating nodes_ in insertion order is a
    // valid topological order. Each node's inputs are guaranteed to already
    // be materialised by the time we reach it.
    for (auto& node : nodes_) {
        if (node->is_materialized()) continue;
        if (node->is_input() || !node->op()) {
            // Unbound placeholder — surface a typed error rather than
            // silently producing garbage.
            if (!node->is_materialized()) {
                throw std::runtime_error("LazyGraph::flush: cannot execute unbound placeholder '" +
                                         node->name() + "'");
            }
            continue;
        }
        auto result = execute_node(node);
        node->set_materialized(std::move(result));
    }
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

    // Gather materialized inputs.
    std::vector<Tensor> input_tensors;
    input_tensors.reserve(node->inputs().size());
    for (auto& inp : node->inputs()) {
        input_tensors.push_back(inp->materialized());
    }

    // Route through the standard backend dispatch table so that the lazy
    // executor mirrors eager dispatch exactly — same kernel, same device,
    // no CPU fallback. The OpId + OpAttributes pair stored on the node is
    // already in the form the dispatch table expects.
    auto op = *node->op();
    try {
        auto outputs = tenzor::dispatch(op, input_tensors, node->attrs());
        if (outputs.empty()) {
            throw std::runtime_error(
                "LazyGraph::execute_node: dispatch returned no outputs for op " +
                std::to_string(static_cast<int>(op)));
        }
        // LazyTensor models single-output ops; multi-output kernels would need
        // a different node type. Take the first output (matches the eager
        // single-output API for every op currently emitted by lazy::*).
        return outputs[0];
    } catch (const std::exception& e) {
        // Surface a typed error explaining why this op cannot be lazily
        // dispatched, rather than falling back to a slower / wrong-device
        // path. No CPU fallback.
        std::ostringstream oss;
        oss << "LazyGraph::execute_node: failed to dispatch op "
            << static_cast<int>(op) << " on device "
            << node->device().to_string() << ": " << e.what();
        throw std::runtime_error(oss.str());
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
    // Defer to LazyGraph::merge_graphs which builds a proper union with
    // edges preserved. If both tensors already share a graph this is a
    // no-op fast path.
    return LazyGraph::merge_graphs(a.graph(), b.graph());
}

// Helper to format an int-list attribute value the way get_int_list expects.
static auto int_list_to_string(const std::vector<int64_t>& v) -> std::string {
    std::ostringstream oss;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i > 0) oss << ',';
        oss << v[i];
    }
    return oss.str();
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
    if (a.empty() || b.empty()) {
        throw std::invalid_argument("LazyTensor: matmul requires at least 1D tensors");
    }
    if (a.size() == 1 && b.size() == 1) return {};  // dot product -> scalar

    // 1-D × N-D (N >= 2): treat `a` as if it were (1, K), broadcast batch
    // dims, then drop the prepended 1. Result shape = batch_b ++ [b.back()].
    if (a.size() == 1) {
        std::vector<int64_t> result(b.begin(), b.end() - 2);
        result.push_back(b.back());
        return result;
    }
    // N-D × 1-D (N >= 2): treat `b` as if it were (K, 1), broadcast batch
    // dims, then drop the trailing 1. Result shape = batch_a ++ [a[-2]].
    if (b.size() == 1) {
        std::vector<int64_t> result(a.begin(), a.end() - 2);
        result.push_back(a[a.size() - 2]);
        return result;
    }

    // General: batch matmul with broadcasting on the leading dims.
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
    auto out_dtype = promote_types(a.dtype(), b.dtype());
    auto node = graph->add_node(OpId::MatMul, {a.node(), b.node()},
                                out_shape, out_dtype, a.device());
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
    auto in_shape = a.shape();
    int64_t ndim = static_cast<int64_t>(in_shape.size());
    int64_t d0 = dim0 < 0 ? dim0 + ndim : dim0;
    int64_t d1 = dim1 < 0 ? dim1 + ndim : dim1;
    if (d0 < 0 || d0 >= ndim || d1 < 0 || d1 >= ndim) {
        throw std::out_of_range("LazyTensor::transpose: dim out of range");
    }
    auto out_shape = in_shape;
    std::swap(out_shape[d0], out_shape[d1]);

    OpAttributes attrs;
    attrs.set(AttrKey::Dim0, d0);
    attrs.set(AttrKey::Dim1, d1);
    auto node = graph->add_node(OpId::Transpose, {a.node()}, out_shape,
                                a.dtype(), a.device(), std::move(attrs));
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

    OpAttributes attrs;
    attrs.set(AttrKey::Shape, int_list_to_string(shape));

    auto graph = a.graph();
    auto node = graph->add_node(OpId::Reshape, {a.node()}, shape,
                                a.dtype(), a.device(), std::move(attrs));
    return LazyTensor(node, graph);
}

auto sum(const LazyTensor& a, std::optional<int64_t> dim) -> LazyTensor {
    auto graph = a.graph();
    std::vector<int64_t> out_shape;
    OpAttributes attrs;
    if (dim.has_value()) {
        out_shape = a.shape();
        int64_t ndim = static_cast<int64_t>(out_shape.size());
        int64_t d = *dim < 0 ? *dim + ndim : *dim;
        if (d < 0 || d >= ndim) {
            throw std::out_of_range("LazyTensor::sum: dim out of range");
        }
        out_shape.erase(out_shape.begin() + d);
        if (out_shape.empty()) out_shape.push_back(1);
        attrs.set(AttrKey::Dim, d);
    } else {
        out_shape = {1};  // full reduction — Sum kernel sentinel is the
                          // default LLONG_MIN, no AttrKey::Dim needed.
    }
    attrs.set(AttrKey::Keepdim, false);
    auto node = graph->add_node(OpId::Sum, {a.node()}, out_shape,
                                a.dtype(), a.device(), std::move(attrs));
    return LazyTensor(node, graph);
}

auto mean(const LazyTensor& a, std::optional<int64_t> dim) -> LazyTensor {
    auto graph = a.graph();
    std::vector<int64_t> out_shape;
    OpAttributes attrs;
    if (dim.has_value()) {
        out_shape = a.shape();
        int64_t ndim = static_cast<int64_t>(out_shape.size());
        int64_t d = *dim < 0 ? *dim + ndim : *dim;
        if (d < 0 || d >= ndim) {
            throw std::out_of_range("LazyTensor::mean: dim out of range");
        }
        out_shape.erase(out_shape.begin() + d);
        if (out_shape.empty()) out_shape.push_back(1);
        attrs.set(AttrKey::Dim, d);
    } else {
        out_shape = {1};
    }
    attrs.set(AttrKey::Keepdim, false);
    auto node = graph->add_node(OpId::Mean, {a.node()}, out_shape,
                                a.dtype(), a.device(), std::move(attrs));
    return LazyTensor(node, graph);
}

} // namespace lazy
} // namespace tenzor
