#include "tenzor/lazy/lazy_tensor.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"

#include <algorithm>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace tenzor {
namespace lazy {

namespace {

// Eager reduction dtype promotion (mirrors src/ops/reduction.cpp). These MUST
// stay in lock-step with the eager paths so lazy recording, lazy materialization
// and eager dispatch all agree on the output dtype.
auto is_small_int_dtype(DType dt) -> bool {
    return dt == DType::Int8 || dt == DType::UInt8 || dt == DType::Int16 ||
           dt == DType::Int32 || dt == DType::UInt16 || dt == DType::UInt32 ||
           dt == DType::Bool;
}

auto is_integer_dtype(DType dt) -> bool {
    return dt == DType::Int8 || dt == DType::UInt8 || dt == DType::Int16 ||
           dt == DType::Int32 || dt == DType::Int64 || dt == DType::Bool;
}

// sum() promotes small integer types to Int64 to prevent overflow.
auto sum_out_dtype(DType in) -> DType {
    return is_small_int_dtype(in) ? DType::Int64 : in;
}

// mean() promotes integer/bool inputs to Float32 (floating-point result).
auto mean_out_dtype(DType in) -> DType {
    return is_integer_dtype(in) ? DType::Float32 : in;
}

}  // namespace

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

auto LazyNode::materialized() const -> Tensor {
    // Return BY VALUE while the lock is held. A Tensor is a cheap refcounted
    // handle, so the copy here merely bumps the storage refcount — but it must
    // happen under materialize_mutex_, otherwise a concurrent set_materialized()
    // (materialized_ = std::move(t)) could destroy the previously stored Tensor
    // and free its backing storage out from under a reference the caller is
    // still reading. Returning const Tensor& released the lock at function exit
    // and exposed exactly that use-after-free under the cross-graph concurrent
    // materialization scenario (shared nodes reachable from two graphs).
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

auto LazyNode::materialize_with(const std::function<Tensor()>& compute) -> Tensor {
    // Per-node compute-once primitive. The node's own materialize_mutex_ is the
    // unit of execution serialisation, which gives three guarantees that a
    // graph-wide lock could not:
    //   1. At-most-once execution even when the SAME LazyNode is reachable from
    //      two distinct LazyGraph objects (the post-merge_graphs case): both
    //      graphs lock THIS node's mutex, not their own per-graph mutex, so the
    //      shared node is computed exactly once and the cached result wins
    //      deterministically (critical for dropout/random kernels).
    //   2. The result is copied out (refcount bump) while the lock is still
    //      held, so it can never alias storage a concurrent writer frees.
    //   3. Independent (disjoint) nodes lock different mutexes and therefore
    //      compute concurrently — no graph-wide serialisation of unrelated work.
    //
    // compute() is invoked at most once per node for its lifetime. It must not
    // re-enter materialize_with()/materialized() on THIS node (it does not: the
    // executor materialises every input before a node's compute runs, so during
    // compute() only OTHER nodes' mutexes are taken).
    std::lock_guard<std::mutex> lock(materialize_mutex_);
    if (!materialized_) {
        materialized_ = compute();
    }
    return *materialized_;
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

    // Pick the larger graph as the base and append the smaller graph's nodes
    // into it (in place). Appending never invalidates existing nodes or their
    // cached results — each LazyNode holds shared_ptr references to its inputs,
    // so the topology survives intact — and it turns the common chaining
    // pattern (N binary ops, each new leaf in its own from_tensor graph) from
    // O(N^2) total node copies (a fresh union allocated and fully repopulated
    // at every step) into O(N): each step appends only the new leaf's nodes to
    // the already-large base. If the smaller graph's nodes are already all
    // present in the base (e.g. both operands derive from the same graph), the
    // append loop simply skips them and returns the base unchanged.
    auto& base = (a->nodes_.size() >= b->nodes_.size()) ? a : b;
    const auto& other = (a->nodes_.size() >= b->nodes_.size()) ? b : a;

    std::unordered_set<LazyNode*> seen;
    seen.reserve(base->nodes_.size() + other->nodes_.size());
    for (const auto& n : base->nodes_) {
        seen.insert(n.get());
    }
    for (const auto& n : other->nodes_) {
        if (seen.insert(n.get()).second) {
            base->nodes_.push_back(n);
        }
    }
    return base;
}

void LazyGraph::flush() {
    // Materialise every node in topological order. Because add_node /
    // add_input append in dependency order (an input was added before any
    // node that references it), iterating nodes_ in insertion order is a
    // valid topological order. Each node's inputs are guaranteed to already
    // be materialised by the time we reach it.
    //
    // Synchronisation is per-node (LazyNode::materialize_with) rather than
    // graph-wide: each node is computed at most once even under concurrent
    // materialize()/flush() on this OR any other graph that shares the node
    // (the post-merge_graphs case), while disjoint nodes may compute
    // concurrently.
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
        node->materialize_with([this, &node] { return execute_node(node); });
    }
}

auto LazyGraph::execute(const std::shared_ptr<LazyNode>& target) -> Tensor {
    // At-most-once execution is enforced per node (LazyNode::materialize_with),
    // NOT by a graph-wide lock. The per-node compute lock lives on the LazyNode
    // itself, so it is shared across every LazyGraph that references the node —
    // which is exactly what makes the post-merge_graphs case correct: after
    // merge_graphs() appends operand B's nodes into base A in place, the same
    // LazyNode objects are reachable from two distinct LazyGraph objects. A
    // graph-wide execute_mutex_ would let A->execute(sharedNode) and an older
    // B->execute(...) lock DIFFERENT mutexes and both run the shared node's
    // kernel (duplicate execution, last-writer-wins caching for dropout/random).
    // Serialising on the node's own mutex closes that cross-graph race while
    // still allowing disjoint nodes to compute concurrently.
    if (target->is_materialized()) {
        return target->materialized();
    }

    // Iterative post-order traversal with an explicit worklist. A previous
    // implementation recursed into each input before executing the node, so
    // recursion depth equalled graph depth (thousands of nodes for a long
    // sequential network or unary-op chain) and could overflow the native
    // stack. The explicit stack below bounds native stack usage to O(1)
    // regardless of graph depth while preserving the "materialize only the
    // transitive inputs of `target`" semantics (unlike flush(), which would
    // materialize every node in the graph).
    //
    // Each stack frame is visited twice: the first visit pushes the node back
    // with `expanded=true` after pushing its not-yet-materialized inputs; the
    // second visit (after all inputs are materialized) executes the node.
    // Each frame carries the owning shared_ptr (so the node stays alive and
    // execute_node receives a valid handle) plus an `expanded` flag. A node's
    // inputs() already returns shared_ptr handles, so no lookup is needed.
    std::vector<std::pair<std::shared_ptr<LazyNode>, bool>> stack;
    stack.emplace_back(target, false);
    std::unordered_set<LazyNode*> in_progress;

    while (!stack.empty()) {
        auto [node, expanded] = std::move(stack.back());
        stack.pop_back();

        if (node->is_materialized()) continue;

        if (!expanded) {
            // Re-push this node to be executed after its inputs.
            stack.emplace_back(node, true);
            in_progress.insert(node.get());
            for (auto& input : node->inputs()) {
                if (!input->is_materialized() &&
                    in_progress.find(input.get()) == in_progress.end()) {
                    stack.emplace_back(input, false);
                }
            }
        } else {
            // All inputs are materialized at this point. materialize_with()
            // computes the node at most once under the node's own lock (even if
            // another thread, possibly via a different graph that shares this
            // node, is racing to materialize it) and copies the result out
            // while the lock is held.
            node->materialize_with([this, &node] { return execute_node(node); });
            in_progress.erase(node.get());
        }
    }

    return target->materialized();
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

    // Mirror eager reduction promotion: eager sum()/mean() promote the input
    // dtype before dispatch (sum: small-int -> Int64, mean: integer/bool ->
    // Float32). execute_node dispatches OpId::Sum/Mean directly, so without this
    // the lazy result would silently disagree with eager for integer inputs and
    // diverge from the (now-promoted) recorded node dtype.
    if ((op == OpId::Sum || op == OpId::Mean) && !input_tensors.empty()) {
        const DType in = input_tensors[0].dtype();
        const DType promoted =
            (op == OpId::Sum) ? sum_out_dtype(in) : mean_out_dtype(in);
        if (promoted != in) {
            input_tensors[0] = input_tensors[0].to(promoted);
        }
    }

    // Mirror eager matmul exactly: lazy::matmul records output dtype via
    // promote_types() and shape via matmul_shape() (which supports broadcasted
    // batch dims), so the graph accepts mixed-dtype and 3D/4D batched inputs.
    // The raw cpu::matmul_kernel does NOT do dtype promotion or bmm
    // decomposition — it throws on dtype mismatch and ndim != 2. Route through
    // tenzor::matmul (which promotes inputs and dispatches 3D+ through bmm) so
    // materialization matches the recorded node and the eager result.
    if (op == OpId::MatMul && input_tensors.size() == 2) {
        return tenzor::matmul(input_tensors[0], input_tensors[1]);
    }

    // Mirror eager reshape exactly. Eager Tensor::reshape (tensor.cpp) forces a
    // contiguous copy of a non-contiguous input before dispatching OpId::Reshape,
    // because the raw reshape kernel merely reinterprets the SAME storage with a
    // new shape and assumes a contiguous, row-major layout. execute_node would
    // otherwise dispatch OpId::Reshape on the raw materialized input — so a lazy
    // reshape of a non-contiguous tensor (e.g. the output of transpose()) would
    // reinterpret a strided buffer as contiguous, silently corrupting data or
    // reading out of bounds. Route through tenzor::reshape (which contiguifies)
    // so materialization matches the eager result. Use the node's resolved
    // output shape (already has -1 inferred and validated).
    if (op == OpId::Reshape && input_tensors.size() == 1) {
        return tenzor::reshape(input_tensors[0], node->shape());
    }

    // Mirror eager elementwise binary ops exactly. lazy::add/sub/mul/div record
    // the output dtype via promote_types() (binary_lazy_op), but raw backend
    // dispatch (e.g. cpu::add_kernel) performs NO dtype promotion and rejects
    // non-contiguous inputs (validate_elementwise throws on dtype mismatch and
    // on non-contiguous tensors). The eager helpers tenzor::add/sub/mul/div
    // promote operands and contiguify before dispatch, so route through them to
    // make materialization match both the recorded node dtype and the eager
    // result for mixed-dtype (e.g. f16 + f32) and transpose-then-elementwise
    // (non-contiguous) patterns.
    if (input_tensors.size() == 2) {
        switch (op) {
            case OpId::Add: return tenzor::add(input_tensors[0], input_tensors[1]);
            case OpId::Sub: return tenzor::sub(input_tensors[0], input_tensors[1]);
            case OpId::Mul: return tenzor::mul(input_tensors[0], input_tensors[1]);
            case OpId::Div: return tenzor::div(input_tensors[0], input_tensors[1]);
            default: break;
        }
    }

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
    // Validate device at graph-build time so a mismatch is a clear error here
    // rather than a confusing failure deep inside execute_node's dispatch.
    if (a.device() != b.device()) {
        throw std::runtime_error(
            "LazyTensor binary op: operands are on different devices (" +
            a.device().to_string() + " vs " + b.device().to_string() + ")");
    }
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
    if (a.device() != b.device()) {
        throw std::runtime_error(
            "LazyTensor matmul: operands are on different devices (" +
            a.device().to_string() + " vs " + b.device().to_string() + ")");
    }
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
        if (known_product == 0) {
            throw std::invalid_argument(
                "LazyTensor::reshape: cannot infer -1 dimension when the "
                "product of the other requested dimensions is zero");
        }
        if (total % known_product != 0) {
            throw std::invalid_argument(
                "LazyTensor::reshape: requested shape is not compatible with "
                "the number of elements (" + std::to_string(total) +
                " is not divisible by " + std::to_string(known_product) + ")");
        }
        shape[neg_idx] = total / known_product;
    } else {
        // No -1 to infer: the requested shape must have exactly the same number
        // of elements as the input. Eager Tensor::reshape enforces this (and
        // throws) before dispatch; the lazy path previously skipped it, so a
        // mismatched shape (e.g. {4,4} over a 15-element {3,5} input) was
        // recorded verbatim. At materialization cpu::reshape_kernel performs no
        // validation — it reinterprets the SAME storage with the new shape — so
        // the resulting Tensor advertises more elements than it owns, producing
        // out-of-bounds reads / heap corruption on any downstream access.
        int64_t total = 1;
        for (auto d : a.shape()) total *= d;
        int64_t requested = 1;
        for (auto d : shape) requested *= d;
        if (requested != total) {
            auto shape_str = [](const std::vector<int64_t>& s) {
                std::string out = "[";
                for (size_t i = 0; i < s.size(); ++i) {
                    out += std::to_string(s[i]);
                    if (i + 1 < s.size()) out += ", ";
                }
                out += "]";
                return out;
            };
            throw std::invalid_argument(
                "LazyTensor::reshape: shape incompatible with number of "
                "elements: trying to reshape " + shape_str(a.shape()) +
                " (numel=" + std::to_string(total) + ") to " + shape_str(shape) +
                " (total=" + std::to_string(requested) + ")");
        }
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
        // keepdim=false: removing the last/only axis yields a 0-D scalar {},
        // matching compute_reduction_shape. No {1} fallback (that would record
        // ndim 1 while materialize() yields ndim 0).
        attrs.set(AttrKey::Dim, d);
    } else {
        out_shape = {};  // full reduction -> 0-D scalar (Sum kernel sentinel is
                         // the default LLONG_MIN, no AttrKey::Dim needed).
    }
    attrs.set(AttrKey::Keepdim, false);
    // Record the promoted output dtype (small-int -> Int64) so LazyTensor::dtype()
    // and any downstream promote_types() match eager sum(); execute_node performs
    // the matching input promotion at materialization.
    auto node = graph->add_node(OpId::Sum, {a.node()}, out_shape,
                                sum_out_dtype(a.dtype()), a.device(),
                                std::move(attrs));
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
        // keepdim=false: removing the last/only axis yields a 0-D scalar {},
        // matching compute_reduction_shape. No {1} fallback.
        attrs.set(AttrKey::Dim, d);
    } else {
        out_shape = {};  // full reduction -> 0-D scalar
    }
    attrs.set(AttrKey::Keepdim, false);
    // Record the promoted output dtype (integer/bool -> Float32) so it matches
    // eager mean(); execute_node performs the matching input promotion.
    auto node = graph->add_node(OpId::Mean, {a.node()}, out_shape,
                                mean_out_dtype(a.dtype()), a.device(),
                                std::move(attrs));
    return LazyTensor(node, graph);
}

} // namespace lazy
} // namespace tenzor
