/**
 * @file graph.cpp
 * @brief Implementation of IR graph construction and manipulation
 */

#include "../../include/tenzor/jit/graph.hpp"
#include "../../include/tenzor/jit/tracer.hpp"
#include "../../include/tenzor/jit/serialization.hpp"
#include "../../include/tenzor/ops/math.hpp"
#include "../../include/tenzor/ops/reduction.hpp"
#include "../../include/tenzor/ops/transform.hpp"
#include "../../include/tenzor/ops/indexing.hpp"
#include "../../include/tenzor/ops/vision.hpp"
#include "../../include/tenzor/ops/linalg.hpp"
#include "../../include/tenzor/autograd/ops.hpp"
#include "../../include/tenzor/nn/functional.hpp"
#include "../../include/tenzor/backend/fast_dispatch.hpp"
#include "../../include/tenzor/core/shape.hpp"
#include <algorithm>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <sstream>
#include <stdexcept>

namespace tenzor {
namespace jit {

namespace {

auto product_dims(const std::vector<int64_t>& dims) -> int64_t {
    return std::accumulate(dims.begin(), dims.end(), int64_t{1},
                           [](int64_t a, int64_t b) { return a * b; });
}

auto normalize_dim_for_rank(int64_t dim, size_t rank, bool allow_end = false) -> int64_t {
    const auto limit = static_cast<int64_t>(rank) + (allow_end ? 1 : 0);
    if (dim < 0) dim += limit;
    if (dim < 0 || dim >= limit) {
        throw std::out_of_range("JIT graph dimension out of range");
    }
    return dim;
}

// Normalize a multi-axis reduction "dims" list against a tensor rank: convert
// negatives, drop out-of-range entries, sort ascending and deduplicate. Mirrors
// resolve_reduce_dims (mlir/lowering.cpp) and SymbolicShapeInference::
// infer_reduction (symbolic_shape_inference.cpp) so the native executor and
// shape inference agree with the MLIR lowering path.
auto normalize_reduce_dims(const std::vector<int64_t>& dims, int64_t rank)
    -> std::vector<int64_t> {
    std::vector<int64_t> norm;
    norm.reserve(dims.size());
    for (auto d : dims) {
        if (d < 0) d += rank;
        if (d >= 0 && d < rank) norm.push_back(d);
    }
    std::sort(norm.begin(), norm.end());
    norm.erase(std::unique(norm.begin(), norm.end()), norm.end());
    return norm;
}

auto attr_vec_or_scalar(const Node& node, const char* name, size_t rank,
                        int64_t default_value) -> std::vector<int64_t> {
    auto values = node.get_vec_attr(name);
    if (!values.empty()) {
        if (values.size() == 1 && rank > 1) {
            values.resize(rank, values[0]);
        }
        return values;
    }
    if (node.has_int_attr(name)) {
        return std::vector<int64_t>(rank, node.get_int_attr(name));
    }
    return std::vector<int64_t>(rank, default_value);
}

auto conv_transpose_output_shape(const Node& node,
                                 const std::vector<std::vector<int64_t>>& input_shapes)
        -> std::vector<int64_t> {
    if (input_shapes.size() < 2 || input_shapes[0].size() < 3 ||
        input_shapes[1].size() < 3) {
        return {};
    }

    const auto spatial_rank = input_shapes[0].size() - 2;
    if (input_shapes[1].size() != spatial_rank + 2) {
        return {};
    }

    const int64_t groups = node.has_attr("groups") ? node.get_int_attr("groups") : 1;
    auto stride = attr_vec_or_scalar(node, "stride", spatial_rank, 1);
    auto padding = attr_vec_or_scalar(node, "padding", spatial_rank, 0);
    auto dilation = attr_vec_or_scalar(node, "dilation", spatial_rank, 1);
    auto output_padding = attr_vec_or_scalar(node, "output_padding", spatial_rank, 0);

    std::vector<int64_t> out;
    out.reserve(input_shapes[0].size());
    out.push_back(input_shapes[0][0]);
    out.push_back(input_shapes[1][1] * groups);
    for (size_t i = 0; i < spatial_rank; ++i) {
        const auto in_size = input_shapes[0][i + 2];
        const auto kernel = input_shapes[1][i + 2];
        out.push_back((in_size - 1) * stride[i] - 2 * padding[i] +
                      dilation[i] * (kernel - 1) + output_padding[i] + 1);
    }
    return out;
}

auto mode_attr_to_string(const Node& node) -> std::string {
    if (!node.has_int_attr("mode")) return "bilinear";
    switch (node.get_int_attr("mode")) {
        case 0: return "nearest";
        case 1: return "bilinear";
        case 2: return "bicubic";
        case 3: return "trilinear";
        default:
            throw std::runtime_error("JIT graph contains unsupported interpolate mode");
    }
}

auto set_conv_attrs_from_node(const Node& node, OpAttributes& attrs, size_t spatial_rank) -> void {
    const auto stride = attr_vec_or_scalar(node, "stride", spatial_rank, 1);
    const auto padding = attr_vec_or_scalar(node, "padding", spatial_rank, 0);
    const auto dilation = attr_vec_or_scalar(node, "dilation", spatial_rank, 1);
    const auto output_padding = attr_vec_or_scalar(node, "output_padding", spatial_rank, 0);

    attrs.set(AttrKey::Stride, stride[0]);
    attrs.set(AttrKey::Padding, padding[0]);
    attrs.set(AttrKey::Dilation, dilation[0]);
    attrs.set(AttrKey::OutputPadding, output_padding[0]);
    if (spatial_rank >= 2) {
        attrs.set(AttrKey::StrideH, stride[spatial_rank - 2]);
        attrs.set(AttrKey::StrideW, stride[spatial_rank - 1]);
        attrs.set(AttrKey::PaddingH, padding[spatial_rank - 2]);
        attrs.set(AttrKey::PaddingW, padding[spatial_rank - 1]);
        attrs.set(AttrKey::DilationH, dilation[spatial_rank - 2]);
        attrs.set(AttrKey::DilationW, dilation[spatial_rank - 1]);
        attrs.set(AttrKey::OutputPaddingH, output_padding[spatial_rank - 2]);
        attrs.set(AttrKey::OutputPaddingW, output_padding[spatial_rank - 1]);
    }
    if (spatial_rank == 3) {
        attrs.set(AttrKey::StrideD, stride[0]);
        attrs.set(AttrKey::PaddingD, padding[0]);
        attrs.set(AttrKey::DilationD, dilation[0]);
        attrs.set(AttrKey::OutputPaddingD, output_padding[0]);
    }
    attrs.set(AttrKey::Groups,
              node.has_attr("groups") ? node.get_int_attr("groups") : int64_t{1});
}

} // namespace

// ============================================================================
// Node implementation
// ============================================================================

auto Node::add_input(std::shared_ptr<Value> val) -> void {
    inputs_.push_back(val);
    val->add_use(shared_from_this());
}

auto Node::add_output(std::shared_ptr<Value> val) -> void {
    // 5th-audit sibling-bug A4: defensively ensure the Value's owning-node
    // pointer is set. Idempotent when callers (e.g. Graph::create_value)
    // already called set_node(this). Closes the latent contract violation
    // where a future caller forgets the wiring step and the resulting
    // Value->node back-edge silently dangles.
    if (val) {
        val->set_node(shared_from_this());
    }
    outputs_.push_back(val);
}

auto Node::replace_input(size_t idx, std::shared_ptr<Value> val) -> void {
    if (idx >= inputs_.size()) {
        throw std::runtime_error("Input index out of bounds");
    }
    // Disconnect this node from the OLD value's uses list before wiring up
    // the new value. Without this, downstream passes (DCE, replace_value,
    // dead-value detection) walk a stale uses_ list and either fail to
    // remove genuinely-dead nodes, or treat live nodes as dead.
    // (Phase P0 / JIT correctness fix.)
    if (auto old = inputs_[idx]) {
        old->remove_use(shared_from_this());
    }
    inputs_[idx] = val;
    val->add_use(shared_from_this());
}

// ============================================================================
// Graph implementation
// ============================================================================

auto Graph::create_node(OpType op_type, const std::string& name) -> std::shared_ptr<Node> {
    std::string node_name = name;
    if (node_name.empty()) {
        node_name = op_type_to_string(op_type) + "_" + std::to_string(next_node_id_++);
    }
    return std::make_shared<Node>(op_type, node_name);
}

auto Graph::create_value(const std::string& id, std::vector<int64_t> shape,
                         DType dtype, Device device) -> std::shared_ptr<Value> {
    auto value = std::make_shared<Value>(id, std::move(shape), dtype, device);
    values_[id] = value;
    return value;
}

auto Graph::get_value(const std::string& id) const -> std::shared_ptr<Value> {
    auto it = values_.find(id);
    return it != values_.end() ? it->second : nullptr;
}

auto Graph::add_node(std::shared_ptr<Node> node) -> void {
    nodes_.push_back(node);
}

auto Graph::remove_node(std::shared_ptr<Node> node) -> void {
    // Remove from nodes list
    auto it = std::find(nodes_.begin(), nodes_.end(), node);
    if (it != nodes_.end()) {
        nodes_.erase(it);
    }

    // Remove this specific node from each input value's uses list
    // (do NOT clear all uses, as other nodes may still reference the value)
    for (auto& input : node->inputs()) {
        input->remove_use(node);
    }
}

auto Graph::replace_value(const std::string& old_id, const std::string& new_id) -> void {
    auto old_val = get_value(old_id);
    auto new_val = get_value(new_id);
    if (!old_val || !new_val) return;

    // For every node that uses old_val as input, replace with new_val
    // Collect uses first since we'll be modifying the list
    std::vector<std::shared_ptr<Node>> consumers;
    for (const auto& weak_use : old_val->uses()) {
        if (auto use = weak_use.lock()) {
            consumers.push_back(use);
        }
    }

    for (auto& consumer : consumers) {
        for (size_t i = 0; i < consumer->inputs().size(); ++i) {
            if (consumer->inputs()[i]->id() == old_id) {
                consumer->replace_input(i, new_val);
            }
        }
    }

    // Clear old value's uses since they've all been redirected
    old_val->clear_uses();

    // Also update graph outputs if they reference the old value
    for (auto& output : outputs_) {
        if (output->id() == old_id) {
            output = new_val;
        }
    }
}

auto Graph::replace_node(std::shared_ptr<Node> old_node, std::shared_ptr<Node> new_node) -> void {
    // Wire all consumers of old_node's outputs to use new_node's outputs
    auto& old_outputs = old_node->outputs();
    auto& new_outputs = new_node->outputs();

    for (size_t i = 0; i < old_outputs.size() && i < new_outputs.size(); ++i) {
        replace_value(old_outputs[i]->id(), new_outputs[i]->id());
    }

    // Remove the old node from the graph
    remove_node(old_node);
}

auto Graph::replace_node_with_value(std::shared_ptr<Node> node, const std::string& value_id) -> void {
    // Redirect all consumers of the node's outputs to use the given value
    for (const auto& output : node->outputs()) {
        replace_value(output->id(), value_id);
    }

    // Remove the node from the graph
    remove_node(node);
}

auto Graph::topological_sort() -> void {
    // Build adjacency list and in-degree map
    std::unordered_map<Node*, std::vector<Node*>> adjacency;
    std::unordered_map<Node*, int> in_degree;

    // Initialize
    for (const auto& node : nodes_) {
        in_degree[node.get()] = 0;
        adjacency[node.get()] = {};
    }

    // Build graph edges
    for (const auto& node : nodes_) {
        for (const auto& input : node->inputs()) {
            auto producer = input->node();
            if (producer) {
                adjacency[producer.get()].push_back(node.get());
                in_degree[node.get()]++;
            }
        }
    }

    // Map raw Node* -> shared_ptr once so the Kahn loop recovers the owning
    // pointer in O(1) instead of an O(N) scan per popped node (was O(N^2)).
    std::unordered_map<Node*, std::shared_ptr<Node>> node_ptrs;
    node_ptrs.reserve(nodes_.size());
    for (const auto& n : nodes_) {
        node_ptrs.emplace(n.get(), n);
    }

    // Kahn's algorithm for topological sort
    std::queue<Node*> zero_in_degree;
    for (const auto& node : nodes_) {
        if (in_degree[node.get()] == 0) {
            zero_in_degree.push(node.get());
        }
    }

    std::vector<std::shared_ptr<Node>> sorted;
    while (!zero_in_degree.empty()) {
        auto node = zero_in_degree.front();
        zero_in_degree.pop();

        // Recover the shared_ptr for this node (O(1)).
        if (auto it = node_ptrs.find(node); it != node_ptrs.end()) {
            sorted.push_back(it->second);
        }

        // Reduce in-degree for successors
        for (auto successor : adjacency[node]) {
            in_degree[successor]--;
            if (in_degree[successor] == 0) {
                zero_in_degree.push(successor);
            }
        }
    }

    if (sorted.size() != nodes_.size()) {
        throw std::runtime_error("Graph contains cycles");
    }

    nodes_ = std::move(sorted);
}

auto Graph::infer_types() -> void {
    // Type inference pass - propagate shapes through operations.
    // Inference reads each node's producers' shapes, so the node list must be in
    // dependency order; enforce it rather than relying on insertion order.
    topological_sort();
    for (const auto& node : nodes_) {
        // Get input shapes
        std::vector<std::vector<int64_t>> input_shapes;
        for (const auto& input : node->inputs()) {
            input_shapes.push_back(input->shape());
        }

        // Infer output shapes based on operation type
        std::vector<std::vector<int64_t>> output_shapes;

        switch (node->op_type()) {
            // ================================================================
            // Binary ops: use proper broadcasting
            // ================================================================
            case OpType::Add:
            case OpType::Sub:
            case OpType::Mul:
            case OpType::Div:
                if (input_shapes.size() >= 2) {
                    output_shapes.push_back(
                        broadcast_shapes(input_shapes[0], input_shapes[1]));
                }
                break;

            // ================================================================
            // Matrix operations
            // ================================================================
            case OpType::MatMul:
                if (input_shapes.size() >= 2 && !input_shapes[0].empty() &&
                    !input_shapes[1].empty()) {
                    auto a = input_shapes[0];
                    auto b = input_shapes[1];
                    // torch.matmul 1-D promotion: a 1-D `a` is treated as a row
                    // (prepend 1), a 1-D `b` as a column (append 1); the added
                    // dims are stripped from the result. The old code required
                    // both operands rank>=2 and silently produced no shape for
                    // vector/dot-product cases.
                    const bool a_vec = a.size() == 1;
                    const bool b_vec = b.size() == 1;
                    if (a_vec) a.insert(a.begin(), 1);
                    if (b_vec) b.push_back(1);
                    if (a.size() >= 2 && b.size() >= 2) {
                        const int64_t M = a[a.size() - 2];
                        const int64_t N = b[b.size() - 1];
                        std::span<const int64_t> a_batch(a.data(), a.size() - 2);
                        std::span<const int64_t> b_batch(b.data(), b.size() - 2);
                        std::vector<int64_t> out_shape =
                            broadcast_shapes(a_batch, b_batch);
                        out_shape.push_back(M);
                        out_shape.push_back(N);
                        if (b_vec && !out_shape.empty()) out_shape.pop_back();
                        if (a_vec && !out_shape.empty()) {
                            out_shape.erase(out_shape.end() - (b_vec ? 1 : 2));
                        }
                        output_shapes.push_back(std::move(out_shape));
                    }
                }
                break;

            case OpType::Bmm:
                // Batch matmul: (B, M, K) @ (B, K, N) -> (B, M, N)
                if (input_shapes.size() >= 2 && input_shapes[0].size() == 3 && input_shapes[1].size() == 3) {
                    output_shapes.push_back({input_shapes[0][0], input_shapes[0][1], input_shapes[1][2]});
                }
                break;

            // ================================================================
            // Unary element-wise ops: preserve shape
            // ================================================================
            case OpType::ReLU:
            case OpType::Sigmoid:
            case OpType::Tanh:
            case OpType::Exp:
            case OpType::Log:
            case OpType::Sqrt:
            case OpType::Pow:
            case OpType::Abs:
            case OpType::Neg:
            case OpType::Clamp:
            case OpType::Dropout:
            case OpType::SiLU:
            case OpType::RMSNorm:
            case OpType::RoPE:
                if (!input_shapes.empty()) {
                    output_shapes.push_back(input_shapes[0]);
                }
                break;

            // ================================================================
            // Softmax / LogSoftmax: preserve shape
            // ================================================================
            case OpType::Softmax:
            case OpType::LogSoftmax:
                if (!input_shapes.empty()) {
                    output_shapes.push_back(input_shapes[0]);
                }
                break;

            // ================================================================
            // Shape operations
            // ================================================================
            case OpType::Reshape:
                if (node->has_attr("shape")) {
                    auto target = node->get_vec_attr("shape");
                    // Resolve a single -1 "infer this dim" placeholder against
                    // the input's element count. Leaving -1 in the shape makes
                    // every backend consume a negative buffer extent.
                    if (!input_shapes.empty()) {
                        int64_t in_numel = 1;
                        for (int64_t d : input_shapes[0]) in_numel *= d;
                        int64_t known = 1;
                        int infer_idx = -1;
                        bool multiple_infer = false;
                        for (size_t i = 0; i < target.size(); ++i) {
                            if (target[i] == -1) {
                                if (infer_idx >= 0) { multiple_infer = true; break; }
                                infer_idx = static_cast<int>(i);
                            } else {
                                known *= target[i];
                            }
                        }
                        if (!multiple_infer && infer_idx >= 0 && known > 0) {
                            target[infer_idx] = in_numel / known;
                        }
                    }
                    output_shapes.push_back(std::move(target));
                }
                break;

            case OpType::Transpose:
                if (!input_shapes.empty()) {
                    auto shape = input_shapes[0];
                    int64_t rank = static_cast<int64_t>(shape.size());
                    int64_t dim0 = node->get_int_attr("dim0");
                    int64_t dim1 = node->get_int_attr("dim1");
                    // Normalize negative dims before indexing (PyTorch-legal,
                    // e.g. transpose(-1,-2)). A bare `dim < size` signed check
                    // passes for negatives and then indexes the vector out of
                    // bounds.
                    if (dim0 < 0) dim0 += rank;
                    if (dim1 < 0) dim1 += rank;
                    if (dim0 >= 0 && dim0 < rank && dim1 >= 0 && dim1 < rank) {
                        std::swap(shape[dim0], shape[dim1]);
                    }
                    output_shapes.push_back(shape);
                }
                break;

            case OpType::Permute:
                if (!input_shapes.empty() && node->has_attr("dims")) {
                    auto dims = node->get_vec_attr("dims");
                    auto& in_shape = input_shapes[0];
                    int64_t rank = static_cast<int64_t>(in_shape.size());
                    std::vector<int64_t> out_shape(dims.size());
                    // Normalize negative dims (permute(0,-1,1) is legal) and
                    // require every index in range; a bare `dim < size` signed
                    // check passed negatives through to an out-of-bounds read
                    // and left out-of-range dims as a silent zero extent.
                    bool valid = true;
                    for (size_t i = 0; i < dims.size(); ++i) {
                        int64_t d = dims[i];
                        if (d < 0) d += rank;
                        if (d >= 0 && d < rank) {
                            out_shape[i] = in_shape[d];
                        } else {
                            valid = false;
                            break;
                        }
                    }
                    if (valid) output_shapes.push_back(std::move(out_shape));
                }
                break;

            case OpType::Squeeze:
                if (!input_shapes.empty()) {
                    auto shape = input_shapes[0];
                    int64_t rank = static_cast<int64_t>(shape.size());
                    if (node->has_attr("dims")) {
                        // Multi-dim squeeze: remove each listed size-1 dim.
                        auto dims = node->get_vec_attr("dims");
                        std::vector<bool> drop(shape.size(), false);
                        for (int64_t d : dims) {
                            if (d < 0) d += rank;
                            if (d >= 0 && d < rank && shape[d] == 1) drop[d] = true;
                        }
                        std::vector<int64_t> out;
                        for (size_t i = 0; i < shape.size(); ++i)
                            if (!drop[i]) out.push_back(shape[i]);
                        shape = std::move(out);
                    } else if (node->has_attr("dim")) {
                        // Single-dim squeeze. Normalize negative dim before
                        // indexing (the old `dim < size` check passed negatives
                        // through to an out-of-bounds read).
                        int64_t dim = node->get_int_attr("dim");
                        if (dim < 0) dim += rank;
                        if (dim >= 0 && dim < rank && shape[dim] == 1) {
                            shape.erase(shape.begin() + dim);
                        }
                    } else {
                        // No dim argument: torch squeeze() removes ALL size-1
                        // dims. The old code defaulted to dim=0 and dropped only
                        // the leading dim.
                        std::vector<int64_t> out;
                        for (int64_t d : shape) if (d != 1) out.push_back(d);
                        shape = std::move(out);
                    }
                    output_shapes.push_back(shape);
                }
                break;

            case OpType::Unsqueeze:
                if (!input_shapes.empty()) {
                    auto shape = input_shapes[0];
                    int64_t dim = node->get_int_attr("dim");
                    if (dim < 0) dim += static_cast<int64_t>(shape.size()) + 1;
                    if (dim >= 0 && dim <= static_cast<int64_t>(shape.size())) {  // guard lower bound too: a very-negative dim stays negative after normalization
                        shape.insert(shape.begin() + dim, 1);
                    }
                    output_shapes.push_back(shape);
                }
                break;

            case OpType::Flatten:
                if (!input_shapes.empty()) {
                    auto& in_shape = input_shapes[0];
                    int64_t start_dim = node->has_attr("start_dim") ? node->get_int_attr("start_dim") : 0;
                    int64_t end_dim = node->has_attr("end_dim") ? node->get_int_attr("end_dim") : -1;
                    if (start_dim < 0) start_dim += static_cast<int64_t>(in_shape.size());
                    if (end_dim < 0) end_dim += static_cast<int64_t>(in_shape.size());
                    std::vector<int64_t> out_shape;
                    int64_t flat = 1;
                    for (int64_t i = 0; i < static_cast<int64_t>(in_shape.size()); ++i) {
                        if (i < start_dim || i > end_dim) {
                            out_shape.push_back(in_shape[i]);
                        } else {
                            flat *= in_shape[i];
                            if (i == end_dim) {
                                out_shape.push_back(flat);
                            }
                        }
                    }
                    output_shapes.push_back(out_shape);
                }
                break;

            // ================================================================
            // Reductions
            // ================================================================
            case OpType::Sum:
            case OpType::Mean:
            case OpType::Max:
            case OpType::Min:
                if (!input_shapes.empty()) {
                    auto shape = input_shapes[0];
                    bool keepdim = node->get_bool_attr("keepdim");
                    if (node->has_int_attr("dim")) {
                        int64_t dim = node->get_int_attr("dim");
                        if (dim < 0) dim += static_cast<int64_t>(shape.size());
                        if (dim >= 0 && dim < static_cast<int64_t>(shape.size())) {
                            if (keepdim) {
                                shape[dim] = 1;
                            } else {
                                shape.erase(shape.begin() + dim);
                            }
                        }
                    } else if (node->has_vec_attr("dims")) {
                        // Multi-axis reduction (tracing interceptor emits
                        // "dims" with no scalar "dim"). Set each axis to 1
                        // (keepdim) or erase highest-to-lowest.
                        auto norm = normalize_reduce_dims(
                            node->get_vec_attr("dims"),
                            static_cast<int64_t>(shape.size()));
                        if (keepdim) {
                            for (auto d : norm) shape[d] = 1;
                        } else {
                            for (auto it = norm.rbegin(); it != norm.rend(); ++it) {
                                shape.erase(shape.begin() + *it);
                            }
                        }
                    } else {
                        shape = {};  // Reduce all: scalar output
                    }
                    output_shapes.push_back(shape);
                }
                break;

            // ================================================================
            // Convolution
            // ================================================================
            case OpType::Conv2d:
                if (!input_shapes.empty()) {
                    auto& shape = input_shapes[0];  // [N, C, H, W]
                    // Prefer reading from the weight tensor's shape
                    // (input 1: [out_channels, in_channels/groups,
                    // kernel_h, kernel_w]). The dispatch-level Conv2d
                    // doesn't push out_channels/kernel_* as attrs, so
                    // the legacy attr path is now a fallback for
                    // hand-built graphs.
                    int64_t out_channels = node->get_int_attr("out_channels");
                    int64_t kernel_h = node->get_int_attr("kernel_h");
                    int64_t kernel_w = node->get_int_attr("kernel_w");
                    if (input_shapes.size() >= 2 &&
                        input_shapes[1].size() == 4) {
                        out_channels = input_shapes[1][0];
                        kernel_h     = input_shapes[1][2];
                        kernel_w     = input_shapes[1][3];
                    }
                    // Honor both paired (stride_h/stride_w from the
                    // dispatch interceptor's hw-pair copy) and the
                    // pair-as-vec form ("stride", "padding", "dilation"
                    // = {h, w}) that the interceptor populates from
                    // AttrKey::Padding/Stride/Dilation scalars.
                    auto pair_from = [&](const char* h_key, const char* w_key,
                                          const char* vec_key,
                                          int64_t default_v)
                            -> std::pair<int64_t, int64_t> {
                        if (node->has_attr(h_key) && node->has_attr(w_key)) {
                            return {node->get_int_attr(h_key),
                                    node->get_int_attr(w_key)};
                        }
                        if (node->has_attr(vec_key)) {
                            auto v = node->get_vec_attr(vec_key);
                            if (v.size() == 2) return {v[0], v[1]};
                            if (v.size() == 1) return {v[0], v[0]};
                        }
                        return {default_v, default_v};
                    };
                    auto [stride_h, stride_w]   = pair_from(
                        "stride_h",  "stride_w",  "stride",   1);
                    auto [padding_h, padding_w] = pair_from(
                        "padding_h", "padding_w", "padding",  0);
                    auto [dilation_h, dilation_w] = pair_from(
                        "dilation_h", "dilation_w", "dilation", 1);

                    if (shape.size() == 4) {
                        int64_t H_out = (shape[2] + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
                        int64_t W_out = (shape[3] + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
                        output_shapes.push_back({shape[0], out_channels, H_out, W_out});
                    }
                }
                break;

            case OpType::ConvTranspose:
                if (auto shape = conv_transpose_output_shape(*node, input_shapes);
                    !shape.empty()) {
                    output_shapes.push_back(std::move(shape));
                }
                break;

            // ================================================================
            // Pooling
            // ================================================================
            case OpType::MaxPool2d:
            case OpType::AvgPool2d:
                if (!input_shapes.empty() && input_shapes[0].size() == 4) {
                    auto& shape = input_shapes[0];  // [N, C, H, W]
                    // kernel_size/stride/padding/dilation may be traced either as
                    // a scalar (square config) or as a [h, w] vec (rectangular);
                    // the old code only read the scalar and produced a 0 extent
                    // for rectangular pools. Also honor dilation and ceil_mode.
                    auto read_pair = [&](const char* name, int64_t dflt,
                                         bool have) -> std::pair<int64_t,int64_t> {
                        if (node->has_vec_attr(name)) {
                            auto v = node->get_vec_attr(name);
                            if (v.size() >= 2) return {v[0], v[1]};
                            if (v.size() == 1) return {v[0], v[0]};
                        }
                        if (node->has_int_attr(name)) {
                            int64_t s = node->get_int_attr(name);
                            return {s, s};
                        }
                        return {dflt, dflt};
                        (void)have;
                    };
                    auto [kh, kw] = read_pair("kernel_size", 1, true);
                    bool has_stride = node->has_int_attr("stride") || node->has_vec_attr("stride");
                    auto [sh, sw] = has_stride ? read_pair("stride", 1, true)
                                               : std::pair<int64_t,int64_t>{kh, kw};
                    auto [ph, pw] = read_pair("padding", 0, true);
                    auto [dh, dw] = read_pair("dilation", 1, true);
                    bool ceil_mode = node->has_bool_attr("ceil_mode") &&
                                     node->get_bool_attr("ceil_mode");
                    auto out_dim = [&](int64_t in, int64_t k, int64_t s,
                                       int64_t p, int64_t d) -> int64_t {
                        int64_t num = in + 2 * p - d * (k - 1) - 1;
                        int64_t o = ceil_mode ? (num + s - 1) / s + 1 : num / s + 1;
                        // ceil_mode: the last pooling window must start inside the
                        // (padded) input, else drop it (matches PyTorch).
                        if (ceil_mode && (o - 1) * s >= in + p) --o;
                        return o;
                    };
                    int64_t H_out = out_dim(shape[2], kh, sh, ph, dh);
                    int64_t W_out = out_dim(shape[3], kw, sw, pw, dw);
                    output_shapes.push_back({shape[0], shape[1], H_out, W_out});
                }
                break;

            case OpType::AdaptiveAvgPool2d:
                if (!input_shapes.empty() && input_shapes[0].size() == 4) {
                    auto output_size = node->get_vec_attr("output_size");
                    if (output_size.size() >= 2) {
                        output_shapes.push_back({input_shapes[0][0], input_shapes[0][1],
                                                 output_size[0], output_size[1]});
                    }
                }
                break;

            // ================================================================
            // Normalization: preserve input shape
            // ================================================================
            case OpType::BatchNorm2d:
            case OpType::LayerNorm:
                if (!input_shapes.empty()) {
                    output_shapes.push_back(input_shapes[0]);
                }
                break;

            // ================================================================
            // Linear: (*, in_features) -> (*, out_features)
            // ================================================================
            case OpType::Linear:
                if (input_shapes.size() >= 2) {
                    auto out_shape = input_shapes[0];
                    // Weight shape is [out_features, in_features]. Guard against a
                    // rank-0 activation: .back() on an empty shape is UB.
                    if (!input_shapes[1].empty() && !out_shape.empty()) {
                        out_shape.back() = input_shapes[1][0];
                    }
                    output_shapes.push_back(out_shape);
                }
                break;

            // ================================================================
            // Embedding: (*, ) -> (*, embedding_dim)
            // ================================================================
            case OpType::Embedding:
                if (input_shapes.size() >= 2) {
                    // input_shapes[0] = weight [num_embeddings, embedding_dim]
                    // input_shapes[1] = indices [*]
                    auto out_shape = input_shapes[1];
                    if (input_shapes[0].size() >= 2) {
                        out_shape.push_back(input_shapes[0][1]);
                    }
                    output_shapes.push_back(out_shape);
                }
                break;

            // ================================================================
            // Indexing
            // ================================================================
            case OpType::Slice:
                if (!input_shapes.empty()) {
                    auto shape = input_shapes[0];
                    int64_t rank = static_cast<int64_t>(shape.size());
                    int64_t dim = node->get_int_attr("dim");
                    int64_t start = node->get_int_attr("start");
                    int64_t end = node->get_int_attr("end");
                    int64_t step = node->has_attr("step") ? node->get_int_attr("step") : 1;
                    // Normalize negative dim before indexing (PyTorch-legal,
                    // e.g. dim=-1). A bare signed `dim < size` check passes for
                    // negatives and then indexes shape[dim] out of bounds.
                    if (dim < 0) dim += rank;
                    if (dim >= 0 && dim < rank) {
                        const int64_t dim_size = shape[dim];
                        // Normalize negative indices and clamp to [0, dim_size]
                        // to mirror Tensor::slice's runtime semantics, so the
                        // inferred dim never goes oversized or negative.
                        if (start < 0) start += dim_size;
                        if (end   < 0) end   += dim_size;
                        if (start < 0) start = 0;
                        if (start > dim_size) start = dim_size;
                        if (end   < start) end = start;
                        if (end   > dim_size) end = dim_size;
                        if (step <= 0) step = 1;
                        int64_t len = (end - start + step - 1) / step;
                        shape[dim] = len;
                    }
                    output_shapes.push_back(shape);
                }
                break;

            case OpType::Cat:
                if (!input_shapes.empty()) {
                    int64_t dim = node->has_attr("dim") ? node->get_int_attr("dim") : 0;
                    auto out_shape = input_shapes[0];
                    int64_t rank = static_cast<int64_t>(out_shape.size());
                    // Normalize negative concat dim (torch.cat(..., dim=-1) is
                    // common); a raw negative dim passes the signed `< size`
                    // check and then indexes out_shape[dim] out of bounds.
                    if (dim < 0) dim += rank;
                    if (dim >= 0 && dim < rank) {
                        int64_t total = 0;
                        for (auto& s : input_shapes) {
                            if (dim < static_cast<int64_t>(s.size())) {
                                total += s[dim];
                            }
                        }
                        out_shape[dim] = total;
                    }
                    output_shapes.push_back(out_shape);
                }
                break;

            case OpType::Stack:
                if (!input_shapes.empty()) {
                    auto out_shape = input_shapes[0];
                    int64_t dim = node->has_attr("dim") ? node->get_int_attr("dim") : 0;
                    dim = normalize_dim_for_rank(dim, out_shape.size(), true);
                    out_shape.insert(out_shape.begin() + dim,
                                     static_cast<int64_t>(input_shapes.size()));
                    output_shapes.push_back(std::move(out_shape));
                }
                break;

            case OpType::Broadcast:
                if (!input_shapes.empty()) {
                    auto target = node->get_vec_attr("shape");
                    output_shapes.push_back(target.empty() ? input_shapes[0] : std::move(target));
                }
                break;

            case OpType::Where:
                if (input_shapes.size() >= 3) {
                    // torch.where broadcasts all three operands, including the
                    // condition. When the condition is broadcast-larger than the
                    // value tensors, dropping it would under-size the output.
                    auto val_shape =
                        broadcast_shapes(input_shapes[1], input_shapes[2]);
                    output_shapes.push_back(
                        broadcast_shapes(input_shapes[0], val_shape));
                }
                break;

            case OpType::IndexSelect:
                if (input_shapes.size() >= 2) {
                    auto out_shape = input_shapes[0];
                    int64_t dim = node->has_attr("dim") ? node->get_int_attr("dim") : 0;
                    dim = normalize_dim_for_rank(dim, out_shape.size());
                    out_shape[dim] = product_dims(input_shapes[1]);
                    output_shapes.push_back(std::move(out_shape));
                }
                break;

            case OpType::Interpolate:
                if (!input_shapes.empty()) {
                    auto size = node->get_vec_attr("output_size");
                    if (!size.empty()) {
                        auto out_shape = input_shapes[0];
                        if (out_shape.size() >= size.size() + 2) {
                            std::copy(size.begin(), size.end(),
                                      out_shape.end() -
                                          static_cast<std::ptrdiff_t>(size.size()));
                        }
                        output_shapes.push_back(std::move(out_shape));
                    }
                    // else: scale_factor form (not captured as output_size) —
                    // leave the trace-time output shape intact instead of
                    // emitting the un-scaled input shape.
                }
                break;

            case OpType::Padding:
            case OpType::GQA:
                if (!input_shapes.empty()) {
                    output_shapes.push_back(input_shapes[0]);
                }
                break;

            // ================================================================
            // GELU: preserve shape (element-wise activation)
            // ================================================================
            case OpType::GELU:
                if (!input_shapes.empty()) {
                    output_shapes.push_back(input_shapes[0]);
                }
                break;

            // ================================================================
            // Linear algebra operations
            // ================================================================
            case OpType::Det:
                // (..., N, N) -> (...)
                if (!input_shapes.empty() && input_shapes[0].size() >= 2) {
                    auto shape = input_shapes[0];
                    shape.pop_back();
                    shape.pop_back();
                    output_shapes.push_back(shape);
                }
                break;

            case OpType::Inv:
            case OpType::Cholesky:
                // (..., N, N) -> (..., N, N)
                if (!input_shapes.empty()) {
                    output_shapes.push_back(input_shapes[0]);
                }
                break;

            case OpType::Solve:
                // A: (..., N, N), B: (..., N, K) -> (..., N, K)
                if (input_shapes.size() >= 2) {
                    output_shapes.push_back(input_shapes[1]);
                }
                break;

            case OpType::Svd:
                // (..., M, N) -> U: (..., M, M), S: (..., min(M,N)), Vt: (..., N, N)
                // (or reduced forms if full_matrices=false)
                if (!input_shapes.empty() && input_shapes[0].size() >= 2) {
                    auto& s = input_shapes[0];
                    int64_t M = s[s.size() - 2];
                    int64_t N = s[s.size() - 1];
                    int64_t K = std::min(M, N);
                    bool full = node->has_attr("full_matrices") ?
                        node->get_bool_attr("full_matrices") : true;
                    auto batch = std::vector<int64_t>(s.begin(), s.end() - 2);
                    auto u_shape = batch;
                    u_shape.push_back(M);
                    u_shape.push_back(full ? M : K);
                    auto s_shape = batch;
                    s_shape.push_back(K);
                    auto vt_shape = batch;
                    vt_shape.push_back(full ? N : K);
                    vt_shape.push_back(N);
                    output_shapes.push_back(u_shape);
                    output_shapes.push_back(s_shape);
                    output_shapes.push_back(vt_shape);
                }
                break;

            case OpType::Qr:
                // (..., M, N) -> Q: (..., M, K), R: (..., K, N) where K=min(M,N)
                if (!input_shapes.empty() && input_shapes[0].size() >= 2) {
                    auto& s = input_shapes[0];
                    int64_t M = s[s.size() - 2];
                    int64_t N = s[s.size() - 1];
                    int64_t K = std::min(M, N);
                    auto batch = std::vector<int64_t>(s.begin(), s.end() - 2);
                    auto q_shape = batch;
                    q_shape.push_back(M);
                    q_shape.push_back(K);
                    auto r_shape = batch;
                    r_shape.push_back(K);
                    r_shape.push_back(N);
                    output_shapes.push_back(q_shape);
                    output_shapes.push_back(r_shape);
                }
                break;

            case OpType::Eigh:
                // (..., N, N) -> eigenvalues: (..., N), eigenvectors: (..., N, N)
                if (!input_shapes.empty() && input_shapes[0].size() >= 2) {
                    auto& s = input_shapes[0];
                    int64_t N = s[s.size() - 1];
                    auto batch = std::vector<int64_t>(s.begin(), s.end() - 2);
                    auto w_shape = batch;
                    w_shape.push_back(N);
                    output_shapes.push_back(w_shape);
                    output_shapes.push_back(input_shapes[0]);  // eigenvectors same shape
                }
                break;

            case OpType::Eigvalsh:
                // (..., N, N) -> (..., N)
                if (!input_shapes.empty() && input_shapes[0].size() >= 2) {
                    auto shape = input_shapes[0];
                    shape.pop_back();  // remove last dim
                    output_shapes.push_back(shape);
                }
                break;

            case OpType::Norm:
                // Produces scalar output
                output_shapes.push_back({});
                break;

            case OpType::Slogdet:
                // (..., N, N) -> sign: (...), logabsdet: (...)
                if (!input_shapes.empty() && input_shapes[0].size() >= 2) {
                    auto shape = input_shapes[0];
                    shape.pop_back();
                    shape.pop_back();
                    output_shapes.push_back(shape);  // sign
                    output_shapes.push_back(shape);  // logabsdet
                }
                break;

            // ================================================================
            // Constants and I/O markers
            // ================================================================
            case OpType::Constant:
                if (node->has_attr("value")) {
                    auto& t = node->get_tensor_attr("value");
                    output_shapes.push_back(std::vector<int64_t>(t.shape().begin(), t.shape().end()));
                }
                break;

            case OpType::Input:
            case OpType::Output:
                if (!input_shapes.empty()) {
                    output_shapes.push_back(input_shapes[0]);
                }
                break;

            // ================================================================
            // Control flow: use subgraph output shapes
            // ================================================================
            case OpType::If:
                // Output shapes come from then_branch outputs
                if (node->then_branch()) {
                    for (const auto& out : node->then_branch()->outputs()) {
                        output_shapes.push_back(out->shape());
                    }
                }
                break;

            case OpType::Loop:
                // Loop outputs are the final loop-carried values (same shape as inputs)
                // Inputs: [max_iter, condition, carried_0, carried_1, ...]
                // Outputs: [carried_0, carried_1, ...]
                for (size_t i = 2; i < input_shapes.size(); ++i) {
                    output_shapes.push_back(input_shapes[i]);
                }
                break;

            // ================================================================
            // Fused operations
            // ================================================================
            case OpType::FlashAttention:
                // Q: (B, H, S, D), K: (B, H, S, D), V: (B, H, S, D) -> (B, H, S, D)
                if (!input_shapes.empty()) {
                    output_shapes.push_back(input_shapes[0]);
                }
                break;

            case OpType::FusedFFN:
                // input: (*, in) -> (*, out) -- output shape depends on second linear weight
                if (input_shapes.size() >= 3) {
                    // inputs[0]=input, inputs[1]=weight1, inputs[2]=weight2
                    auto out_shape = input_shapes[0];
                    if (!input_shapes[2].empty()) {
                        out_shape.back() = input_shapes[2][0];
                    }
                    output_shapes.push_back(out_shape);
                } else if (!input_shapes.empty()) {
                    output_shapes.push_back(input_shapes[0]);
                }
                break;

            case OpType::ResidualAdd:
                // x + sublayer(x) -> same shape as input
                if (!input_shapes.empty()) {
                    output_shapes.push_back(input_shapes[0]);
                }
                break;

            // ================================================================
            // Guard / memory management pseudo-ops (no shape transformation)
            // ================================================================
            case OpType::ShapeGuard:
            case OpType::GuardNode:
                if (!input_shapes.empty()) {
                    output_shapes.push_back(input_shapes[0]);
                }
                break;

            case OpType::SwapOut:
            case OpType::SwapIn:
                if (!input_shapes.empty()) {
                    output_shapes.push_back(input_shapes[0]);
                }
                break;

            case OpType::LayoutConvert:
            case OpType::Cast:
                // Shape-preserving: output has same shape as input
                if (!input_shapes.empty()) {
                    output_shapes.push_back(input_shapes[0]);
                }
                break;

            case OpType::QuantizedLinear:
            case OpType::QuantizedConv2d:
            case OpType::Dequantize:
            case OpType::Quantize:
            case OpType::SparseMatMul:
            case OpType::DenseToSparse:
                if (!input_shapes.empty()) {
                    output_shapes.push_back(input_shapes[0]);
                }
                break;
        }

        // Update output shapes
        for (size_t i = 0; i < node->outputs().size() && i < output_shapes.size(); ++i) {
            node->outputs()[i]->set_shape(output_shapes[i]);
        }
    }
}

// ============================================================================
// Symbolic shape support
// ============================================================================

auto Graph::set_symbolic_input_shape(size_t input_idx, SymbolicShape shape) -> void {
    if (input_idx >= inputs_.size()) {
        throw std::out_of_range(
            "Input index " + std::to_string(input_idx) + " out of range (graph has " +
            std::to_string(inputs_.size()) + " inputs)");
    }
    inputs_[input_idx]->set_symbolic_shape(std::move(shape));
}

auto Graph::infer_symbolic_types() -> void {
    // Symbolic type inference pass - propagate symbolic shapes through operations.
    // For each node, we gather the symbolic shapes of all inputs and compute
    // the symbolic shape of the outputs using the same rules as infer_types().
    // Requires dependency order (a node's producers must precede it).
    topological_sort();
    for (const auto& node : nodes_) {
        // Gather input symbolic shapes. If an input has no symbolic shape set,
        // derive one from its concrete shape.
        std::vector<SymbolicShape> input_sym_shapes;
        for (const auto& input : node->inputs()) {
            if (input->has_symbolic_shape()) {
                input_sym_shapes.push_back(input->symbolic_shape());
            } else {
                input_sym_shapes.push_back(SymbolicShape::from_concrete(input->shape()));
            }
        }

        // Infer output symbolic shapes based on operation type
        std::vector<SymbolicShape> output_sym_shapes;

        switch (node->op_type()) {
            // ================================================================
            // Binary ops: symbolic broadcasting
            // ================================================================
            case OpType::Add:
            case OpType::Sub:
            case OpType::Mul:
            case OpType::Div:
                if (input_sym_shapes.size() >= 2) {
                    output_sym_shapes.push_back(
                        broadcast_symbolic_shapes(input_sym_shapes[0], input_sym_shapes[1]));
                }
                break;

            // ================================================================
            // Matrix operations
            // ================================================================
            case OpType::MatMul:
                if (input_sym_shapes.size() >= 2 &&
                    input_sym_shapes[0].rank() >= 2 &&
                    input_sym_shapes[1].rank() >= 2) {
                    const auto& a = input_sym_shapes[0];
                    const auto& b = input_sym_shapes[1];
                    const SymbolicDim M = a[a.rank() - 2];
                    const SymbolicDim N = b[b.rank() - 1];
                    // Broadcast the leading batch dims (everything but the last
                    // two) of both operands, mirroring torch.matmul and the
                    // concrete infer_types path. The naive "input[0] with last
                    // dim replaced" form is wrong when input[1] has more batch
                    // dims (e.g. (M,K) @ (B,K,N) -> (B,M,N)).
                    SymbolicShape a_batch(std::vector<SymbolicDim>(
                        a.dims().begin(), a.dims().end() - 2));
                    SymbolicShape b_batch(std::vector<SymbolicDim>(
                        b.dims().begin(), b.dims().end() - 2));
                    SymbolicShape out_shape =
                        broadcast_symbolic_shapes(a_batch, b_batch);
                    out_shape.push_back(M);
                    out_shape.push_back(N);
                    output_sym_shapes.push_back(std::move(out_shape));
                }
                break;

            case OpType::Bmm:
                // Batch matmul: (B, M, K) @ (B, K, N) -> (B, M, N)
                if (input_sym_shapes.size() >= 2 &&
                    input_sym_shapes[0].rank() == 3 &&
                    input_sym_shapes[1].rank() == 3) {
                    output_sym_shapes.push_back(SymbolicShape({
                        input_sym_shapes[0][0],
                        input_sym_shapes[0][1],
                        input_sym_shapes[1][2]
                    }));
                }
                break;

            // ================================================================
            // Unary element-wise ops: preserve shape
            // ================================================================
            case OpType::ReLU:
            case OpType::Sigmoid:
            case OpType::Tanh:
            case OpType::Exp:
            case OpType::Log:
            case OpType::Sqrt:
            case OpType::Pow:
            case OpType::Abs:
            case OpType::Neg:
            case OpType::Clamp:
            case OpType::Dropout:
            case OpType::SiLU:
            case OpType::RMSNorm:
            case OpType::RoPE:
                if (!input_sym_shapes.empty()) {
                    output_sym_shapes.push_back(input_sym_shapes[0]);
                }
                break;

            // ================================================================
            // Softmax / LogSoftmax: preserve shape
            // ================================================================
            case OpType::Softmax:
            case OpType::LogSoftmax:
                if (!input_sym_shapes.empty()) {
                    output_sym_shapes.push_back(input_sym_shapes[0]);
                }
                break;

            // ================================================================
            // Shape operations
            // ================================================================
            case OpType::Reshape:
                if (node->has_attr("shape")) {
                    auto target = node->get_vec_attr("shape");
                    // Resolve a single -1 "infer" placeholder when the input is
                    // fully concrete; otherwise the symbolic shape would carry a
                    // concrete dim of -1 (worse than a real symbol).
                    int infer_idx = -1;
                    int64_t known = 1;
                    bool multi = false;
                    for (size_t i = 0; i < target.size(); ++i) {
                        if (target[i] == -1) {
                            if (infer_idx >= 0) { multi = true; break; }
                            infer_idx = static_cast<int>(i);
                        } else {
                            known *= target[i];
                        }
                    }
                    if (infer_idx >= 0 && !multi && known > 0 &&
                        !input_sym_shapes.empty()) {
                        int64_t numel = 1;
                        bool concrete = true;
                        for (size_t i = 0; i < input_sym_shapes[0].rank(); ++i) {
                            const auto& d = input_sym_shapes[0][i];
                            if (d.is_concrete()) numel *= d.value();
                            else { concrete = false; break; }
                        }
                        if (concrete) target[infer_idx] = numel / known;
                    }
                    output_sym_shapes.push_back(
                        SymbolicShape::from_concrete(target));
                }
                break;

            case OpType::Transpose:
                if (!input_sym_shapes.empty()) {
                    auto sym_shape = input_sym_shapes[0];
                    int64_t rank = static_cast<int64_t>(sym_shape.rank());
                    int64_t dim0 = node->get_int_attr("dim0");
                    int64_t dim1 = node->get_int_attr("dim1");
                    // Normalize negative dims before indexing (mirrors the
                    // concrete infer_types Transpose path). The common
                    // transpose(-1,-2) otherwise passes the signed `< rank`
                    // check and indexes out of bounds.
                    if (dim0 < 0) dim0 += rank;
                    if (dim1 < 0) dim1 += rank;
                    if (dim0 >= 0 && dim0 < rank && dim1 >= 0 && dim1 < rank) {
                        std::swap(sym_shape[static_cast<size_t>(dim0)],
                                  sym_shape[static_cast<size_t>(dim1)]);
                    }
                    output_sym_shapes.push_back(std::move(sym_shape));
                }
                break;

            case OpType::Permute:
                if (!input_sym_shapes.empty() && node->has_attr("dims")) {
                    auto dims = node->get_vec_attr("dims");
                    auto& in_shape = input_sym_shapes[0];
                    int64_t rank = static_cast<int64_t>(in_shape.rank());
                    std::vector<SymbolicDim> out_dims(dims.size());
                    // Mirror the concrete path: normalize negative dims and
                    // require every index in range (a bare `< rank` signed check
                    // let negatives index out of bounds and left out-of-range
                    // axes as a default-constructed dim).
                    bool valid = true;
                    for (size_t i = 0; i < dims.size(); ++i) {
                        int64_t d = dims[i];
                        if (d < 0) d += rank;
                        if (d >= 0 && d < rank) {
                            out_dims[i] = in_shape[static_cast<size_t>(d)];
                        } else {
                            valid = false;
                            break;
                        }
                    }
                    if (valid) {
                        output_sym_shapes.push_back(
                            SymbolicShape(std::move(out_dims)));
                    }
                }
                break;

            case OpType::Squeeze:
                if (!input_sym_shapes.empty()) {
                    auto sym_shape = input_sym_shapes[0];
                    int64_t dim = node->get_int_attr("dim");
                    // Normalize a negative dim (e.g. squeeze(-1)) before indexing;
                    // the concrete infer_types path does this too. Without it, the
                    // cast to size_t below turns -1 into SIZE_MAX -> OOB read.
                    if (dim < 0) {
                        dim += static_cast<int64_t>(sym_shape.rank());
                    }
                    if (dim >= 0 && dim < static_cast<int64_t>(sym_shape.rank())) {
                        auto& d = sym_shape[static_cast<size_t>(dim)];
                        // Only squeeze if concrete and equal to 1
                        if (d.is_concrete() && d.value() == 1) {
                            sym_shape.erase(static_cast<size_t>(dim));
                        }
                    }
                    output_sym_shapes.push_back(std::move(sym_shape));
                }
                break;

            case OpType::Unsqueeze:
                if (!input_sym_shapes.empty()) {
                    auto sym_shape = input_sym_shapes[0];
                    int64_t dim = node->get_int_attr("dim");
                    if (dim < 0) dim += static_cast<int64_t>(sym_shape.rank()) + 1;
                    if (dim >= 0 && dim <= static_cast<int64_t>(sym_shape.rank())) {  // guard lower bound (see infer_types Unsqueeze)
                        sym_shape.insert(static_cast<size_t>(dim), SymbolicDim::concrete(1));
                    }
                    output_sym_shapes.push_back(std::move(sym_shape));
                }
                break;

            case OpType::Flatten:
                if (!input_sym_shapes.empty()) {
                    auto& in_shape = input_sym_shapes[0];
                    int64_t start_dim = node->has_attr("start_dim") ? node->get_int_attr("start_dim") : 0;
                    int64_t end_dim = node->has_attr("end_dim") ? node->get_int_attr("end_dim") : -1;
                    if (start_dim < 0) start_dim += static_cast<int64_t>(in_shape.rank());
                    if (end_dim < 0) end_dim += static_cast<int64_t>(in_shape.rank());

                    std::vector<SymbolicDim> out_dims;
                    SymbolicDim flat = SymbolicDim::concrete(1);
                    for (int64_t i = 0; i < static_cast<int64_t>(in_shape.rank()); ++i) {
                        if (i < start_dim || i > end_dim) {
                            out_dims.push_back(in_shape[static_cast<size_t>(i)]);
                        } else {
                            flat = flat * in_shape[static_cast<size_t>(i)];
                            if (i == end_dim) {
                                out_dims.push_back(flat);
                            }
                        }
                    }
                    output_sym_shapes.push_back(SymbolicShape(std::move(out_dims)));
                }
                break;

            // ================================================================
            // Reductions
            // ================================================================
            case OpType::Sum:
            case OpType::Mean:
            case OpType::Max:
            case OpType::Min:
                if (!input_sym_shapes.empty()) {
                    auto sym_shape = input_sym_shapes[0];
                    bool keepdim = node->get_bool_attr("keepdim");
                    if (node->has_int_attr("dim")) {
                        int64_t dim = node->get_int_attr("dim");
                        if (dim < 0) dim += static_cast<int64_t>(sym_shape.rank());
                        if (dim >= 0 && dim < static_cast<int64_t>(sym_shape.rank())) {
                            if (keepdim) {
                                sym_shape[static_cast<size_t>(dim)] = SymbolicDim::concrete(1);
                            } else {
                                sym_shape.erase(static_cast<size_t>(dim));
                            }
                        }
                    } else if (node->has_vec_attr("dims")) {
                        // Multi-axis reduction (tracing interceptor emits
                        // "dims" with no scalar "dim").
                        auto norm = normalize_reduce_dims(
                            node->get_vec_attr("dims"),
                            static_cast<int64_t>(sym_shape.rank()));
                        if (keepdim) {
                            for (auto d : norm) {
                                sym_shape[static_cast<size_t>(d)] = SymbolicDim::concrete(1);
                            }
                        } else {
                            for (auto it = norm.rbegin(); it != norm.rend(); ++it) {
                                sym_shape.erase(static_cast<size_t>(*it));
                            }
                        }
                    } else {
                        sym_shape = SymbolicShape();  // Reduce all: scalar output
                    }
                    output_sym_shapes.push_back(std::move(sym_shape));
                }
                break;

            // ================================================================
            // Convolution
            // ================================================================
            case OpType::Conv2d:
                if (!input_sym_shapes.empty() && input_sym_shapes[0].rank() == 4) {
                    auto& in_shape = input_sym_shapes[0];
                    int64_t out_channels = node->get_int_attr("out_channels");
                    int64_t kernel_h = node->get_int_attr("kernel_h");
                    int64_t kernel_w = node->get_int_attr("kernel_w");
                    // Prefer the weight tensor's shape (input 1:
                    // [out_channels, in_channels/groups, kernel_h, kernel_w]),
                    // mirroring the concrete infer_types Conv2d path, since the
                    // dispatch interceptor does not push out_channels/kernel_*
                    // as attrs for traced graphs.
                    if (input_sym_shapes.size() >= 2 &&
                        input_sym_shapes[1].rank() == 4) {
                        const auto& w = input_sym_shapes[1];
                        if (w[0].is_concrete()) out_channels = w[0].value();
                        if (w[2].is_concrete()) kernel_h = w[2].value();
                        if (w[3].is_concrete()) kernel_w = w[3].value();
                    }
                    // Honor both the paired per-axis form (stride_h/stride_w
                    // from the dispatch interceptor's hw-pair copy) and the
                    // pair-as-vec form ("stride"/"padding"/"dilation" = {h, w})
                    // that the interceptor populates from AttrKey scalars. A
                    // bare has_attr("stride_h") check silently defaults traced
                    // graphs (which only carry the vec form) to 1/0/1, diverging
                    // from the concrete infer_types result.
                    auto pair_from = [&](const char* h_key, const char* w_key,
                                          const char* vec_key,
                                          int64_t default_v)
                            -> std::pair<int64_t, int64_t> {
                        if (node->has_attr(h_key) && node->has_attr(w_key)) {
                            return {node->get_int_attr(h_key),
                                    node->get_int_attr(w_key)};
                        }
                        if (node->has_attr(vec_key)) {
                            auto v = node->get_vec_attr(vec_key);
                            if (v.size() == 2) return {v[0], v[1]};
                            if (v.size() == 1) return {v[0], v[0]};
                        }
                        return {default_v, default_v};
                    };
                    auto [stride_h, stride_w]     = pair_from(
                        "stride_h",   "stride_w",   "stride",   1);
                    auto [padding_h, padding_w]   = pair_from(
                        "padding_h",  "padding_w",  "padding",  0);
                    auto [dilation_h, dilation_w] = pair_from(
                        "dilation_h", "dilation_w", "dilation", 1);

                    // Batch dim (N) is symbolic, spatial dims may be symbolic too
                    SymbolicDim N_dim = in_shape[0];
                    SymbolicDim H_dim = in_shape[2];
                    SymbolicDim W_dim = in_shape[3];

                    // H_out = (H + 2*padding - dilation*(kernel-1) - 1) / stride + 1
                    SymbolicDim padding_h_dim = SymbolicDim::concrete(2 * padding_h);
                    SymbolicDim kernel_term_h = SymbolicDim::concrete(dilation_h * (kernel_h - 1) + 1);
                    SymbolicDim stride_h_dim = SymbolicDim::concrete(stride_h);
                    SymbolicDim H_out = (H_dim + padding_h_dim - kernel_term_h) / stride_h_dim
                                        + SymbolicDim::concrete(1);

                    SymbolicDim padding_w_dim = SymbolicDim::concrete(2 * padding_w);
                    SymbolicDim kernel_term_w = SymbolicDim::concrete(dilation_w * (kernel_w - 1) + 1);
                    SymbolicDim stride_w_dim = SymbolicDim::concrete(stride_w);
                    SymbolicDim W_out = (W_dim + padding_w_dim - kernel_term_w) / stride_w_dim
                                        + SymbolicDim::concrete(1);

                    output_sym_shapes.push_back(SymbolicShape({
                        N_dim,
                        SymbolicDim::concrete(out_channels),
                        H_out,
                        W_out
                    }));
                }
                break;

            case OpType::ConvTranspose:
                if (input_sym_shapes.size() >= 2 &&
                    input_sym_shapes[0].rank() >= 3 &&
                    input_sym_shapes[1].rank() == input_sym_shapes[0].rank()) {
                    const auto spatial_rank = input_sym_shapes[0].rank() - 2;
                    const int64_t groups = node->has_attr("groups") ?
                        node->get_int_attr("groups") : 1;
                    auto stride = attr_vec_or_scalar(*node, "stride", spatial_rank, 1);
                    auto padding = attr_vec_or_scalar(*node, "padding", spatial_rank, 0);
                    auto dilation = attr_vec_or_scalar(*node, "dilation", spatial_rank, 1);
                    auto output_padding =
                        attr_vec_or_scalar(*node, "output_padding", spatial_rank, 0);
                    std::vector<SymbolicDim> dims;
                    dims.reserve(input_sym_shapes[0].rank());
                    dims.push_back(input_sym_shapes[0][0]);
                    dims.push_back(input_sym_shapes[1][1] * SymbolicDim::concrete(groups));
                    for (size_t i = 0; i < spatial_rank; ++i) {
                        auto in_size = input_sym_shapes[0][i + 2];
                        auto kernel = input_sym_shapes[1][i + 2];
                        dims.push_back((in_size - SymbolicDim::concrete(1)) *
                                       SymbolicDim::concrete(stride[i]) -
                                       SymbolicDim::concrete(2 * padding[i]) +
                                       SymbolicDim::concrete(dilation[i]) *
                                       (kernel - SymbolicDim::concrete(1)) +
                                       SymbolicDim::concrete(output_padding[i] + 1));
                    }
                    output_sym_shapes.push_back(SymbolicShape(std::move(dims)));
                }
                break;

            // ================================================================
            // Pooling
            // ================================================================
            case OpType::MaxPool2d:
            case OpType::AvgPool2d:
                if (!input_sym_shapes.empty() && input_sym_shapes[0].rank() == 4) {
                    auto& in_shape = input_sym_shapes[0];
                    int64_t kernel = node->get_int_attr("kernel_size");
                    int64_t stride = node->has_attr("stride") ? node->get_int_attr("stride") : kernel;
                    int64_t padding = node->has_attr("padding") ? node->get_int_attr("padding") : 0;

                    SymbolicDim H_dim = in_shape[2];
                    SymbolicDim W_dim = in_shape[3];
                    SymbolicDim pad2 = SymbolicDim::concrete(2 * padding);
                    SymbolicDim k = SymbolicDim::concrete(kernel);
                    SymbolicDim s = SymbolicDim::concrete(stride);
                    SymbolicDim one = SymbolicDim::concrete(1);

                    SymbolicDim H_out = (H_dim + pad2 - k) / s + one;
                    SymbolicDim W_out = (W_dim + pad2 - k) / s + one;

                    output_sym_shapes.push_back(SymbolicShape({
                        in_shape[0], in_shape[1], H_out, W_out
                    }));
                }
                break;

            case OpType::AdaptiveAvgPool2d:
                if (!input_sym_shapes.empty() && input_sym_shapes[0].rank() == 4) {
                    auto output_size = node->get_vec_attr("output_size");
                    if (output_size.size() >= 2) {
                        output_sym_shapes.push_back(SymbolicShape({
                            input_sym_shapes[0][0],
                            input_sym_shapes[0][1],
                            SymbolicDim::concrete(output_size[0]),
                            SymbolicDim::concrete(output_size[1])
                        }));
                    }
                }
                break;

            // ================================================================
            // Normalization: preserve input shape
            // ================================================================
            case OpType::BatchNorm2d:
            case OpType::LayerNorm:
                if (!input_sym_shapes.empty()) {
                    output_sym_shapes.push_back(input_sym_shapes[0]);
                }
                break;

            // ================================================================
            // Linear: (*, in_features) -> (*, out_features)
            // ================================================================
            case OpType::Linear:
                if (input_sym_shapes.size() >= 2) {
                    auto out_shape = input_sym_shapes[0];
                    // Weight shape is [out_features, in_features]
                    if (input_sym_shapes[1].rank() > 0) {
                        out_shape[out_shape.rank() - 1] = input_sym_shapes[1][0];
                    }
                    output_sym_shapes.push_back(std::move(out_shape));
                }
                break;

            // ================================================================
            // Embedding: (*, ) -> (*, embedding_dim)
            // ================================================================
            case OpType::Embedding:
                if (input_sym_shapes.size() >= 2) {
                    // input_sym_shapes[0] = weight [num_embeddings, embedding_dim]
                    // input_sym_shapes[1] = indices [*]
                    auto out_shape = input_sym_shapes[1];
                    if (input_sym_shapes[0].rank() >= 2) {
                        out_shape.push_back(input_sym_shapes[0][1]);
                    }
                    output_sym_shapes.push_back(std::move(out_shape));
                }
                break;

            // ================================================================
            // Indexing
            // ================================================================
            case OpType::Slice:
                if (!input_sym_shapes.empty()) {
                    auto sym_shape = input_sym_shapes[0];
                    int64_t rank = static_cast<int64_t>(sym_shape.rank());
                    int64_t dim = node->get_int_attr("dim");
                    int64_t start = node->get_int_attr("start");
                    int64_t end = node->get_int_attr("end");
                    int64_t step = node->has_attr("step") ? node->get_int_attr("step") : 1;
                    // Normalize negative dim before indexing (mirrors the
                    // concrete infer_types Slice path); a raw negative dim
                    // promoted to size_t indexes the shape out of bounds.
                    if (dim < 0) dim += rank;
                    if (dim >= 0 && dim < rank) {
                        if (step <= 0) step = 1;
                        const SymbolicDim& cur = sym_shape[static_cast<size_t>(dim)];
                        if (cur.is_concrete()) {
                            const int64_t dim_size = cur.value();
                            // Normalize negative indices and clamp to
                            // [0, dim_size] to mirror Tensor::slice.
                            if (start < 0) start += dim_size;
                            if (end   < 0) end   += dim_size;
                            if (start < 0) start = 0;
                            if (start > dim_size) start = dim_size;
                            if (end   < start) end = start;
                            if (end   > dim_size) end = dim_size;
                            int64_t len = (end - start + step - 1) / step;
                            sym_shape[static_cast<size_t>(dim)] =
                                SymbolicDim::concrete(len);
                        } else if (start >= 0 && end >= start) {
                            // Dynamic dim but both endpoints are non-negative,
                            // so the sliced length is a concrete value.
                            int64_t len = (end - start + step - 1) / step;
                            sym_shape[static_cast<size_t>(dim)] =
                                SymbolicDim::concrete(len);
                        }
                        // Otherwise the length depends on the unknown dim
                        // size (negative index against a dynamic dim); leave
                        // the dim symbolic rather than emit a garbage length.
                    }
                    output_sym_shapes.push_back(std::move(sym_shape));
                }
                break;

            case OpType::Cat:
                if (!input_sym_shapes.empty()) {
                    int64_t dim = node->has_attr("dim") ? node->get_int_attr("dim") : 0;
                    auto out_shape = input_sym_shapes[0];
                    int64_t rank = static_cast<int64_t>(out_shape.rank());
                    // Normalize negative concat dim (mirrors the concrete path);
                    // a raw negative dim promoted to size_t indexes the shape
                    // out of bounds.
                    if (dim < 0) dim += rank;
                    if (dim >= 0 && dim < rank) {
                        // Sum all dimensions along the cat axis
                        SymbolicDim total = SymbolicDim::concrete(0);
                        for (auto& s : input_sym_shapes) {
                            if (dim < static_cast<int64_t>(s.rank())) {
                                total = total + s[static_cast<size_t>(dim)];
                            }
                        }
                        out_shape[static_cast<size_t>(dim)] = total;
                    }
                    output_sym_shapes.push_back(std::move(out_shape));
	                }
	                break;

	            case OpType::Stack:
	                if (!input_sym_shapes.empty()) {
	                    auto sym_shape = input_sym_shapes[0];
	                    int64_t dim = node->has_attr("dim") ? node->get_int_attr("dim") : 0;
	                    dim = normalize_dim_for_rank(dim, sym_shape.rank(), true);
	                    sym_shape.insert(static_cast<size_t>(dim),
	                                     SymbolicDim::concrete(
	                                         static_cast<int64_t>(input_sym_shapes.size())));
	                    output_sym_shapes.push_back(std::move(sym_shape));
	                }
	                break;

	            case OpType::Broadcast:
	                if (!input_sym_shapes.empty()) {
	                    auto target = node->get_vec_attr("shape");
	                    output_sym_shapes.push_back(
	                        target.empty() ? input_sym_shapes[0]
	                                       : SymbolicShape::from_concrete(target));
	                }
	                break;

	            case OpType::Where:
	                if (input_sym_shapes.size() >= 3) {
	                    // Broadcast all three operands (condition included),
	                    // mirroring torch.where and the concrete infer_types path.
	                    auto val_shape = broadcast_symbolic_shapes(
	                        input_sym_shapes[1], input_sym_shapes[2]);
	                    output_sym_shapes.push_back(
	                        broadcast_symbolic_shapes(input_sym_shapes[0],
	                                                  val_shape));
	                }
	                break;

	            case OpType::IndexSelect:
	                if (input_sym_shapes.size() >= 2) {
	                    auto out_shape = input_sym_shapes[0];
	                    int64_t dim = node->has_attr("dim") ? node->get_int_attr("dim") : 0;
	                    dim = normalize_dim_for_rank(dim, out_shape.rank());
	                    SymbolicDim selected = SymbolicDim::concrete(1);
	                    for (size_t i = 0; i < input_sym_shapes[1].rank(); ++i) {
	                        selected = selected * input_sym_shapes[1][i];
	                    }
	                    out_shape[static_cast<size_t>(dim)] = selected;
	                    output_sym_shapes.push_back(std::move(out_shape));
	                }
	                break;

	            case OpType::Interpolate:
	                if (!input_sym_shapes.empty()) {
	                    auto out_shape = input_sym_shapes[0];
	                    auto size = node->get_vec_attr("output_size");
	                    if (!size.empty() && out_shape.rank() >= size.size() + 2) {
	                        const auto offset = out_shape.rank() - size.size();
	                        for (size_t i = 0; i < size.size(); ++i) {
	                            out_shape[offset + i] = SymbolicDim::concrete(size[i]);
	                        }
	                    }
	                    output_sym_shapes.push_back(std::move(out_shape));
	                }
	                break;

	            case OpType::Padding:
	            case OpType::GQA:
	                if (!input_sym_shapes.empty()) {
	                    output_sym_shapes.push_back(input_sym_shapes[0]);
	                }
	                break;

	            // ================================================================
	            // GELU: preserve shape (element-wise activation)
	            // ================================================================
            case OpType::GELU:
                if (!input_sym_shapes.empty()) {
                    output_sym_shapes.push_back(input_sym_shapes[0]);
                }
                break;

            // ================================================================
            // Linear algebra operations
            // ================================================================
            case OpType::Det:
                // (..., N, N) -> (...)
                if (!input_sym_shapes.empty() && input_sym_shapes[0].rank() >= 2) {
                    auto sym_shape = input_sym_shapes[0];
                    sym_shape.erase(sym_shape.rank() - 1);
                    sym_shape.erase(sym_shape.rank() - 1);
                    output_sym_shapes.push_back(std::move(sym_shape));
                }
                break;

            case OpType::Inv:
            case OpType::Cholesky:
                // (..., N, N) -> (..., N, N)
                if (!input_sym_shapes.empty()) {
                    output_sym_shapes.push_back(input_sym_shapes[0]);
                }
                break;

            case OpType::Solve:
                // A: (..., N, N), B: (..., N, K) -> (..., N, K)
                if (input_sym_shapes.size() >= 2) {
                    output_sym_shapes.push_back(input_sym_shapes[1]);
                }
                break;

            case OpType::Svd:
                // (..., M, N) -> U: (..., M, M|K), S: (..., K), Vt: (..., K|N, N)
                // with K = min(M, N); reduced forms when full_matrices=false.
                if (!input_sym_shapes.empty() && input_sym_shapes[0].rank() >= 2) {
                    auto& s = input_sym_shapes[0];
                    auto M = s[s.rank() - 2];
                    auto N_dim = s[s.rank() - 1];
                    // K = min(M, N). Concrete when both dims are concrete; the
                    // old code hardcoded M, giving wrong S/U/Vt for non-square.
                    auto min_dim = [](const SymbolicDim& a,
                                      const SymbolicDim& b) -> SymbolicDim {
                        if (a.is_concrete() && b.is_concrete())
                            return SymbolicDim(std::min(a.value(), b.value()));
                        return a.is_concrete() ? a : b;  // best-effort if dynamic
                    };
                    SymbolicDim K = min_dim(M, N_dim);
                    bool full = node->has_attr("full_matrices")
                                    ? node->get_bool_attr("full_matrices")
                                    : true;
                    std::vector<SymbolicDim> batch_dims;
                    for (size_t d = 0; d + 2 < s.rank(); ++d) {
                        batch_dims.push_back(s[d]);
                    }
                    auto u_dims = batch_dims;
                    u_dims.push_back(M);
                    u_dims.push_back(full ? M : K);
                    output_sym_shapes.push_back(SymbolicShape(std::move(u_dims)));
                    auto s_dims = batch_dims;
                    s_dims.push_back(K);
                    output_sym_shapes.push_back(SymbolicShape(std::move(s_dims)));
                    auto vt_dims = batch_dims;
                    vt_dims.push_back(full ? N_dim : K);
                    vt_dims.push_back(N_dim);
                    output_sym_shapes.push_back(SymbolicShape(std::move(vt_dims)));
                }
                break;

            case OpType::Qr:
                // (..., M, N) -> Q: (..., M, K), R: (..., K, N), K = min(M, N)
                if (!input_sym_shapes.empty() && input_sym_shapes[0].rank() >= 2) {
                    auto& s = input_sym_shapes[0];
                    auto M = s[s.rank() - 2];
                    auto N_dim = s[s.rank() - 1];
                    auto min_dim = [](const SymbolicDim& a,
                                      const SymbolicDim& b) -> SymbolicDim {
                        if (a.is_concrete() && b.is_concrete())
                            return SymbolicDim(std::min(a.value(), b.value()));
                        return a.is_concrete() ? a : b;
                    };
                    SymbolicDim K = min_dim(M, N_dim);
                    std::vector<SymbolicDim> batch_dims;
                    for (size_t d = 0; d + 2 < s.rank(); ++d) {
                        batch_dims.push_back(s[d]);
                    }
                    auto q_dims = batch_dims;
                    q_dims.push_back(M);
                    q_dims.push_back(K);
                    output_sym_shapes.push_back(SymbolicShape(std::move(q_dims)));
                    auto r_dims = batch_dims;
                    r_dims.push_back(K);
                    r_dims.push_back(N_dim);
                    output_sym_shapes.push_back(SymbolicShape(std::move(r_dims)));
                }
                break;

            case OpType::Eigh:
                // (..., N, N) -> eigenvalues: (..., N), eigenvectors: (..., N, N)
                if (!input_sym_shapes.empty() && input_sym_shapes[0].rank() >= 2) {
                    auto& s = input_sym_shapes[0];
                    auto N_dim = s[s.rank() - 1];
                    std::vector<SymbolicDim> batch_dims;
                    for (size_t d = 0; d + 2 < s.rank(); ++d) {
                        batch_dims.push_back(s[d]);
                    }
                    auto w_dims = batch_dims;
                    w_dims.push_back(N_dim);
                    output_sym_shapes.push_back(SymbolicShape(std::move(w_dims)));
                    output_sym_shapes.push_back(input_sym_shapes[0]);
                }
                break;

            case OpType::Eigvalsh:
                // (..., N, N) -> (..., N)
                if (!input_sym_shapes.empty() && input_sym_shapes[0].rank() >= 2) {
                    auto sym_shape = input_sym_shapes[0];
                    sym_shape.erase(sym_shape.rank() - 1);
                    output_sym_shapes.push_back(std::move(sym_shape));
                }
                break;

            case OpType::Norm:
                // Scalar output
                output_sym_shapes.push_back(SymbolicShape());
                break;

            case OpType::Slogdet:
                // (..., N, N) -> sign: (...), logabsdet: (...)
                if (!input_sym_shapes.empty() && input_sym_shapes[0].rank() >= 2) {
                    auto sym_shape = input_sym_shapes[0];
                    sym_shape.erase(sym_shape.rank() - 1);
                    sym_shape.erase(sym_shape.rank() - 1);
                    output_sym_shapes.push_back(sym_shape);  // sign
                    output_sym_shapes.push_back(sym_shape);  // logabsdet
                }
                break;

            // ================================================================
            // Constants and I/O markers
            // ================================================================
            case OpType::Constant:
                if (node->has_attr("value")) {
                    auto& t = node->get_tensor_attr("value");
                    output_sym_shapes.push_back(
                        SymbolicShape::from_concrete(
                            std::vector<int64_t>(t.shape().begin(), t.shape().end())));
                }
                break;

            case OpType::Input:
            case OpType::Output:
                if (!input_sym_shapes.empty()) {
                    output_sym_shapes.push_back(input_sym_shapes[0]);
                }
                break;

            // ================================================================
            // Control flow: propagate from subgraph outputs
            // ================================================================
            case OpType::If:
                if (node->then_branch()) {
                    for (const auto& out : node->then_branch()->outputs()) {
                        output_sym_shapes.push_back(out->symbolic_shape());
                    }
                }
                break;

            case OpType::Loop:
                // Loop-carried values preserve shapes
                for (size_t i = 2; i < input_sym_shapes.size(); ++i) {
                    output_sym_shapes.push_back(input_sym_shapes[i]);
                }
                break;

            // ================================================================
            // Fused operations
            // ================================================================
            case OpType::FlashAttention:
                // Q: (B, H, S, D) -> same shape as Q
                if (!input_sym_shapes.empty()) {
                    output_sym_shapes.push_back(input_sym_shapes[0]);
                }
                break;

            case OpType::FusedFFN:
                // input: (*, in) -> (*, out) via second linear weight
                if (input_sym_shapes.size() >= 3) {
                    auto out_shape = input_sym_shapes[0];
                    if (input_sym_shapes[2].rank() > 0) {
                        out_shape[out_shape.rank() - 1] = input_sym_shapes[2][0];
                    }
                    output_sym_shapes.push_back(std::move(out_shape));
                } else if (!input_sym_shapes.empty()) {
                    output_sym_shapes.push_back(input_sym_shapes[0]);
                }
                break;

            case OpType::ResidualAdd:
                // x + sublayer(x) -> same shape
                if (!input_sym_shapes.empty()) {
                    output_sym_shapes.push_back(input_sym_shapes[0]);
                }
                break;

            // ================================================================
            // Guard / memory management pseudo-ops (no shape transformation)
            // ================================================================
            case OpType::ShapeGuard:
            case OpType::GuardNode:
                if (!input_sym_shapes.empty()) {
                    output_sym_shapes.push_back(input_sym_shapes[0]);
                }
                break;

            case OpType::SwapOut:
            case OpType::SwapIn:
                if (!input_sym_shapes.empty()) {
                    output_sym_shapes.push_back(input_sym_shapes[0]);
                }
                break;

            case OpType::LayoutConvert:
            case OpType::Cast:
                // Shape-preserving: output has same shape as input
                if (!input_sym_shapes.empty()) {
                    output_sym_shapes.push_back(input_sym_shapes[0]);
                }
                break;

            case OpType::QuantizedLinear:
            case OpType::QuantizedConv2d:
            case OpType::Dequantize:
            case OpType::Quantize:
            case OpType::SparseMatMul:
            case OpType::DenseToSparse:
                if (!input_sym_shapes.empty()) {
                    output_sym_shapes.push_back(input_sym_shapes[0]);
                }
                break;
        }

        // Update output symbolic shapes
        for (size_t i = 0; i < node->outputs().size() && i < output_sym_shapes.size(); ++i) {
            node->outputs()[i]->set_symbolic_shape(output_sym_shapes[i]);
        }
    }
}

auto Graph::forward(const std::vector<Variable>& runtime_inputs) -> std::vector<Variable> {
    // Map values to runtime variables
    std::unordered_map<std::string, Variable> value_map;

    // Initialize with inputs
    if (runtime_inputs.size() != inputs_.size()) {
        throw std::runtime_error("Input count mismatch");
    }

    for (size_t i = 0; i < inputs_.size(); ++i) {
        value_map[inputs_[i]->id()] = runtime_inputs[i];
    }

    // Phase 6.4: pre-populate with captured constants (module
    // parameters, etc.) so ops that reference them don't fail with
    // "Input value not available". Constants are wrapped in Variables
    // with requires_grad=false — the graph is in inference mode.
    for (const auto& [id, tensor] : constants_) {
        value_map.emplace(id, Variable(tensor, /*requires_grad=*/false));
    }

    // Execute nodes in topological order
    for (const auto& node : nodes_) {
        execute_node(node, value_map);
    }

    // Gather outputs
    std::vector<Variable> results;
    for (const auto& output : outputs_) {
        auto it = value_map.find(output->id());
        if (it != value_map.end()) {
            results.push_back(it->second);
        } else {
            throw std::runtime_error("Output value not computed: " + output->id());
        }
    }

    return results;
}

auto Graph::execute_node(const std::shared_ptr<Node>& node,
                         std::unordered_map<std::string, Variable>& value_map) const -> void {
    // Get input variables
    std::vector<Variable> input_vars;
    for (const auto& input : node->inputs()) {
        auto it = value_map.find(input->id());
        if (it != value_map.end()) {
            input_vars.push_back(it->second);
        } else {
            throw std::runtime_error("Input value not available: " + input->id());
        }
    }

    // Execute operation based on type
    std::vector<Variable> outputs;

    switch (node->op_type()) {
        // ====================================================================
        // Arithmetic operations
        // ====================================================================
        case OpType::Add:
            if (input_vars.size() >= 2) {
                outputs.push_back(input_vars[0] + input_vars[1]);
            }
            break;

        case OpType::Sub:
            if (input_vars.size() >= 2) {
                outputs.push_back(input_vars[0] - input_vars[1]);
            }
            break;

        case OpType::Mul:
            if (input_vars.size() >= 2) {
                outputs.push_back(input_vars[0] * input_vars[1]);
            }
            break;

        case OpType::Div:
            if (input_vars.size() >= 2) {
                outputs.push_back(input_vars[0] / input_vars[1]);
            }
            break;

        // ====================================================================
        // Matrix operations
        // ====================================================================
        case OpType::MatMul:
            if (input_vars.size() >= 2) {
                Variable mm_out = tenzor::matmul(input_vars[0], input_vars[1]);
                // optimize_for_inference (FuseMatMulAddPass / activation fusion)
                // may fold a following bias-add and/or ReLU into this node and
                // delete the separate nodes. Honor the markers so the fused ops
                // are still applied; otherwise the bias/activation are dropped.
                if (node->get_bool_attr("fused_bias")) {
                    if (input_vars.size() >= 3) {
                        mm_out = mm_out + input_vars[2];
                    } else if (node->has_tensor_attr("fused_bias")) {
                        mm_out = mm_out +
                            Variable(node->get_tensor_attr("fused_bias"), false);
                    }
                }
                if (node->get_bool_attr("fused_relu")) {
                    mm_out = nn::relu(mm_out);
                }
                outputs.push_back(mm_out);
            }
            break;

        case OpType::Bmm:
            if (input_vars.size() >= 2) {
                outputs.push_back(tenzor::bmm(input_vars[0], input_vars[1]));
            }
            break;

        // ====================================================================
        // Activations
        // ====================================================================
        case OpType::ReLU:
            if (!input_vars.empty()) {
                outputs.push_back(nn::relu(input_vars[0]));
            }
            break;

        case OpType::Sigmoid:
            if (!input_vars.empty()) {
                outputs.push_back(nn::sigmoid(input_vars[0]));
            }
            break;

        case OpType::Tanh:
            if (!input_vars.empty()) {
                outputs.push_back(nn::tanh(input_vars[0]));
            }
            break;

        case OpType::Softmax:
            if (!input_vars.empty()) {
                int64_t dim = node->has_attr("dim") ? node->get_int_attr("dim") : -1;
                outputs.push_back(tenzor::softmax(input_vars[0], dim));
            }
            break;

        case OpType::LogSoftmax:
            if (!input_vars.empty()) {
                int64_t dim = node->has_attr("dim") ? node->get_int_attr("dim") : -1;
                outputs.push_back(tenzor::log_softmax(input_vars[0], dim));
            }
            break;

        // ====================================================================
        // Pooling (dispatch to backend kernels)
        // ====================================================================
        case OpType::MaxPool2d:
            if (!input_vars.empty()) {
                OpAttributes pool_attrs;
                const int64_t kh = node->get_int_attr("kernel_size");
                const int64_t sh = node->has_attr("stride") ?
                    node->get_int_attr("stride") : kh;
                const int64_t ph = node->has_attr("padding") ?
                    node->get_int_attr("padding") : static_cast<int64_t>(0);
                pool_attrs.set(AttrKey::KernelSize, kh);
                pool_attrs.set(AttrKey::Stride, sh);
                pool_attrs.set(AttrKey::Padding, ph);
                // Per-axis W attrs so non-square pools replay correctly; fall
                // back to the H value (square) for legacy traces without them.
                pool_attrs.set(AttrKey::KernelSizeW,
                    node->has_attr("kernel_size_w") ? node->get_int_attr("kernel_size_w") : kh);
                pool_attrs.set(AttrKey::StrideW,
                    node->has_attr("stride_w") ? node->get_int_attr("stride_w") : sh);
                pool_attrs.set(AttrKey::PaddingW,
                    node->has_attr("padding_w") ? node->get_int_attr("padding_w") : ph);
                std::vector<Tensor> inputs = {input_vars[0].tensor()};
                auto result = dispatch(OpId::MaxPool2dForward, inputs, pool_attrs);
                if (!result.empty()) {
                    outputs.push_back(Variable(result[0], false));
                }
            }
            break;

        case OpType::AvgPool2d:
            if (!input_vars.empty()) {
                OpAttributes pool_attrs;
                const int64_t kh = node->get_int_attr("kernel_size");
                const int64_t sh = node->has_attr("stride") ?
                    node->get_int_attr("stride") : kh;
                const int64_t ph = node->has_attr("padding") ?
                    node->get_int_attr("padding") : static_cast<int64_t>(0);
                pool_attrs.set(AttrKey::KernelSize, kh);
                pool_attrs.set(AttrKey::Stride, sh);
                pool_attrs.set(AttrKey::Padding, ph);
                pool_attrs.set(AttrKey::KernelSizeW,
                    node->has_attr("kernel_size_w") ? node->get_int_attr("kernel_size_w") : kh);
                pool_attrs.set(AttrKey::StrideW,
                    node->has_attr("stride_w") ? node->get_int_attr("stride_w") : sh);
                pool_attrs.set(AttrKey::PaddingW,
                    node->has_attr("padding_w") ? node->get_int_attr("padding_w") : ph);
                std::vector<Tensor> inputs = {input_vars[0].tensor()};
                auto result = dispatch(OpId::AvgPool2dForward, inputs, pool_attrs);
                if (!result.empty()) {
                    outputs.push_back(Variable(result[0], false));
                }
            }
            break;

        case OpType::AdaptiveAvgPool2d:
            if (!input_vars.empty()) {
                OpAttributes pool_attrs;
                auto output_size = node->get_vec_attr("output_size");
                if (output_size.size() >= 2) {
                    pool_attrs.set(AttrKey::OutputSizeH, output_size[0]);
                    pool_attrs.set(AttrKey::OutputSizeW, output_size[1]);
                }
                std::vector<Tensor> inputs = {input_vars[0].tensor()};
                auto result = dispatch(OpId::AdaptiveAvgPool2d, inputs, pool_attrs);
                if (!result.empty()) {
                    outputs.push_back(Variable(result[0], false));
                }
            }
            break;

        // ====================================================================
        // Convolution (dispatch to backend kernels)
        // ====================================================================
        case OpType::Conv2d:
            if (input_vars.size() >= 2) {
                // Read per-axis stride/padding/dilation honoring all forms the
                // tracer can emit: explicit per-axis ints ("stride_h"/"stride_w"
                // from the dispatch interceptor's hw-pair copy), the
                // pair-as-vec ("stride"/"padding"/"dilation" = {h, w}) the
                // interceptor populates from AttrKey::PaddingH/PaddingW etc.,
                // and the scalar int fallback ("stride") for square configs.
                // Audit item G.13: previously this only read the (never-set)
                // scalar int "stride_h"/"padding_h"/"dilation", so JIT-traced
                // Conv2d silently ran with stride=1,padding=0,dilation=1
                // regardless of what the eager module set.
                // Resolve a {h, w} pair from any of the three forms the
                // tracer emits: explicit per-axis ints, a 2-element vec, or
                // a scalar int (square config). has_attr() returns true if
                // the key lives in either int_attrs_ or vec_attrs_, so probe
                // both maps explicitly.
                auto pair_from = [&](const char* h_key, const char* w_key,
                                      const char* shared_key,
                                      int64_t default_v)
                        -> std::pair<int64_t, int64_t> {
                    if (node->has_attr(h_key) && node->has_attr(w_key)) {
                        return {node->get_int_attr(h_key),
                                node->get_int_attr(w_key)};
                    }
                    auto v = node->get_vec_attr(shared_key);
                    if (v.size() == 2) return {v[0], v[1]};
                    if (v.size() == 1) return {v[0], v[0]};
                    if (node->has_attr(shared_key)) {
                        int64_t s = node->get_int_attr(shared_key);
                        return {s, s};
                    }
                    return {default_v, default_v};
                };
                auto [stride_h, stride_w]     = pair_from(
                    "stride_h",   "stride_w",   "stride",   1);
                auto [padding_h, padding_w]   = pair_from(
                    "padding_h",  "padding_w",  "padding",  0);
                auto [dilation_h, dilation_w] = pair_from(
                    "dilation_h", "dilation_w", "dilation", 1);
                int64_t groups = node->has_attr("groups") ?
                    node->get_int_attr("groups") : static_cast<int64_t>(1);

                OpAttributes conv_attrs;
                // Per-axis keys are what the backend kernels actually read
                // (see cpu_kernel_registry.cpp Conv2dForward). Set the scalar
                // forms too so kernels that only consult them still work.
                conv_attrs.set(AttrKey::StrideH,   stride_h);
                conv_attrs.set(AttrKey::StrideW,   stride_w);
                conv_attrs.set(AttrKey::PaddingH,  padding_h);
                conv_attrs.set(AttrKey::PaddingW,  padding_w);
                conv_attrs.set(AttrKey::DilationH, dilation_h);
                conv_attrs.set(AttrKey::DilationW, dilation_w);
                conv_attrs.set(AttrKey::Stride,    stride_h);
                conv_attrs.set(AttrKey::Padding,   padding_h);
                conv_attrs.set(AttrKey::Dilation,  dilation_h);
                conv_attrs.set(AttrKey::Groups,    groups);

                std::vector<Tensor> inputs = {input_vars[0].tensor(), input_vars[1].tensor()};
                if (input_vars.size() >= 3) {
                    inputs.push_back(input_vars[2].tensor());
                }
                auto result = dispatch(OpId::Conv2dForward, inputs, conv_attrs);
                if (!result.empty()) {
                    Variable conv_out(result[0], false);
                    // optimize_for_inference (FuseConvReLU / FuseConvBNReLU) may
                    // fold a following ReLU into this node and delete the ReLU
                    // node; apply it here so the activation is not dropped.
                    if (node->get_bool_attr("fused_relu")) {
                        conv_out = nn::relu(conv_out);
                    }
                    outputs.push_back(conv_out);
                }
            }
            break;

        case OpType::ConvTranspose:
            if (input_vars.size() >= 2) {
                const auto rank = input_vars[0].tensor().shape().size();
                if (rank < 3 || rank > 5) {
                    throw std::runtime_error("ConvTranspose expects 3D, 4D, or 5D input");
                }
                const auto spatial_rank = rank - 2;
                OpAttributes attrs;
                set_conv_attrs_from_node(*node, attrs, spatial_rank);
                std::vector<Tensor> inputs = {input_vars[0].tensor(), input_vars[1].tensor()};
                if (input_vars.size() >= 3) {
                    inputs.push_back(input_vars[2].tensor());
                }
                const OpId op = spatial_rank == 1 ? OpId::ConvTranspose1dForward
                              : spatial_rank == 2 ? OpId::ConvTranspose2dForward
                                                  : OpId::ConvTranspose3dForward;
                auto result = dispatch(op, inputs, attrs);
                if (!result.empty()) {
                    outputs.push_back(Variable(result[0], false));
                }
            }
            break;

        // ====================================================================
        // Normalization (dispatch to backend kernels)
        // ====================================================================
        case OpType::BatchNorm2d:
            if (!input_vars.empty()) {
                OpAttributes bn_attrs;
                bn_attrs.set(AttrKey::Eps, node->has_attr("eps") ?
                    static_cast<double>(node->get_attr("eps")) : 1e-5);
                bn_attrs.set(AttrKey::Momentum, 0.1);
                bn_attrs.set(AttrKey::Training, false);
                std::vector<Tensor> inputs;
                for (auto& iv : input_vars) {
                    inputs.push_back(iv.tensor());
                }
                auto result = dispatch(OpId::BatchNorm2dForward, inputs, bn_attrs);
                if (!result.empty()) {
                    outputs.push_back(Variable(result[0], false));
                }
            }
            break;

        case OpType::LayerNorm:
            if (!input_vars.empty()) {
                OpAttributes ln_attrs;
                ln_attrs.set(AttrKey::Eps, node->has_attr("eps") ?
                    static_cast<double>(node->get_attr("eps")) : 1e-5);
                auto normalized_shape = node->get_vec_attr("normalized_shape");
                std::string ns_str;
                for (size_t i = 0; i < normalized_shape.size(); ++i) {
                    if (i > 0) ns_str += ',';
                    ns_str += std::to_string(normalized_shape[i]);
                }
                ln_attrs.set(AttrKey::NormalizedShape, ns_str);
                std::vector<Tensor> inputs;
                for (auto& iv : input_vars) {
                    inputs.push_back(iv.tensor());
                }
                auto result = dispatch(OpId::LayerNorm, inputs, ln_attrs);
                if (!result.empty()) {
                    outputs.push_back(Variable(result[0], false));
                }
            }
            break;

        // ====================================================================
        // Shape operations
        // ====================================================================
        case OpType::Reshape:
            if (!input_vars.empty()) {
                auto shape = node->get_vec_attr("shape");
                outputs.push_back(input_vars[0].reshape(shape));
            }
            break;

        case OpType::Transpose:
            if (!input_vars.empty()) {
                int64_t dim0 = node->get_int_attr("dim0");
                int64_t dim1 = node->get_int_attr("dim1");
                outputs.push_back(input_vars[0].transpose(dim0, dim1));
            }
            break;

        case OpType::Permute:
            if (!input_vars.empty()) {
                auto dims = node->get_vec_attr("dims");
                outputs.push_back(tenzor::permute(input_vars[0], dims));
            }
            break;

        case OpType::Squeeze:
            if (!input_vars.empty()) {
                int64_t dim = node->get_int_attr("dim");
                outputs.push_back(tenzor::squeeze(input_vars[0], dim));
            }
            break;

        case OpType::Unsqueeze:
            if (!input_vars.empty()) {
                int64_t dim = node->get_int_attr("dim");
                auto result = input_vars[0].tensor().unsqueeze(dim);
                outputs.push_back(Variable(result, false));
            }
            break;

        case OpType::Flatten:
            if (!input_vars.empty()) {
                int64_t start_dim = node->has_attr("start_dim") ? node->get_int_attr("start_dim") : 0;
                int64_t end_dim = node->has_attr("end_dim") ? node->get_int_attr("end_dim") : -1;
                auto result = input_vars[0].tensor().flatten(start_dim, end_dim);
                outputs.push_back(Variable(result, false));
            }
            break;

        case OpType::Stack:
            if (!input_vars.empty()) {
                int64_t dim = node->has_attr("dim") ? node->get_int_attr("dim") : 0;
                std::vector<Tensor> tensors;
                tensors.reserve(input_vars.size());
                for (const auto& v : input_vars) {
                    tensors.push_back(v.tensor());
                }
                auto result = tenzor::stack(std::span<const Tensor>(tensors), dim);
                outputs.push_back(Variable(result, false));
            }
            break;

        case OpType::Broadcast:
            if (!input_vars.empty()) {
                auto shape = node->get_vec_attr("shape");
                if (shape.empty()) {
                    outputs.push_back(input_vars[0]);
                } else {
                    outputs.push_back(tenzor::expand(input_vars[0], shape));
                }
            }
            break;

        // ====================================================================
        // Reductions
        // ====================================================================
        case OpType::Sum:
            if (!input_vars.empty()) {
                bool keepdim = node->has_attr("keepdim") ? node->get_bool_attr("keepdim") : false;
                if (node->has_int_attr("dim")) {
                    int64_t dim = node->get_int_attr("dim");
                    outputs.push_back(tenzor::sum(input_vars[0], dim, keepdim));
                } else if (node->has_vec_attr("dims")) {
                    // Multi-axis reduction: reduce sequentially. Mirror
                    // resolve_reduce_dims (lowering.cpp) / infer_reduction
                    // (symbolic_shape_inference.cpp). Reduce highest axis first
                    // (non-keepdim) so earlier reductions don't shift the
                    // indices still to remove.
                    auto norm = normalize_reduce_dims(
                        node->get_vec_attr("dims"),
                        static_cast<int64_t>(input_vars[0].tensor().shape().size()));
                    Variable acc = input_vars[0];
                    for (auto it = norm.rbegin(); it != norm.rend(); ++it) {
                        acc = tenzor::sum(acc, *it, keepdim);
                    }
                    outputs.push_back(acc);
                } else {
                    outputs.push_back(tenzor::sum(input_vars[0]));
                }
            }
            break;

        case OpType::Mean:
            if (!input_vars.empty()) {
                bool keepdim = node->has_attr("keepdim") ? node->get_bool_attr("keepdim") : false;
                if (node->has_int_attr("dim")) {
                    int64_t dim = node->get_int_attr("dim");
                    outputs.push_back(tenzor::mean(input_vars[0], dim, keepdim));
                } else if (node->has_vec_attr("dims")) {
                    auto norm = normalize_reduce_dims(
                        node->get_vec_attr("dims"),
                        static_cast<int64_t>(input_vars[0].tensor().shape().size()));
                    Variable acc = input_vars[0];
                    for (auto it = norm.rbegin(); it != norm.rend(); ++it) {
                        acc = tenzor::mean(acc, *it, keepdim);
                    }
                    outputs.push_back(acc);
                } else {
                    outputs.push_back(tenzor::mean(input_vars[0]));
                }
            }
            break;

        case OpType::Max:
            if (!input_vars.empty()) {
                bool keepdim = node->has_attr("keepdim") ? node->get_bool_attr("keepdim") : false;
                if (node->has_int_attr("dim")) {
                    int64_t dim = node->get_int_attr("dim");
                    outputs.push_back(tenzor::max(input_vars[0], dim, keepdim));
                } else if (node->has_vec_attr("dims")) {
                    auto norm = normalize_reduce_dims(
                        node->get_vec_attr("dims"),
                        static_cast<int64_t>(input_vars[0].tensor().shape().size()));
                    Variable acc = input_vars[0];
                    for (auto it = norm.rbegin(); it != norm.rend(); ++it) {
                        acc = tenzor::max(acc, *it, keepdim);
                    }
                    outputs.push_back(acc);
                } else {
                    outputs.push_back(tenzor::max(input_vars[0]));
                }
            }
            break;

        case OpType::Min:
            if (!input_vars.empty()) {
                // No autograd min — use raw tensor op and wrap
                bool keepdim = node->has_attr("keepdim") ? node->get_bool_attr("keepdim") : false;
                if (node->has_int_attr("dim")) {
                    int64_t dim = node->get_int_attr("dim");
                    auto result = tenzor::min(input_vars[0].tensor(), dim, keepdim);
                    outputs.push_back(Variable(result, false));
                } else if (node->has_vec_attr("dims")) {
                    auto norm = normalize_reduce_dims(
                        node->get_vec_attr("dims"),
                        static_cast<int64_t>(input_vars[0].tensor().shape().size()));
                    Tensor acc = input_vars[0].tensor();
                    for (auto it = norm.rbegin(); it != norm.rend(); ++it) {
                        acc = tenzor::min(acc, *it, keepdim);
                    }
                    outputs.push_back(Variable(acc, false));
                } else {
                    auto result = tenzor::min(input_vars[0].tensor());
                    outputs.push_back(Variable(result, false));
                }
            }
            break;

        // ====================================================================
        // Element-wise math
        // ====================================================================
        case OpType::Exp:
            if (!input_vars.empty()) {
                outputs.push_back(tenzor::exp(input_vars[0]));
            }
            break;

        case OpType::Log:
            if (!input_vars.empty()) {
                outputs.push_back(tenzor::log(input_vars[0]));
            }
            break;

        case OpType::Sqrt:
            if (!input_vars.empty()) {
                auto result = tenzor::sqrt(input_vars[0].tensor());
                outputs.push_back(Variable(result, false));
            }
            break;

        case OpType::Pow:
            if (input_vars.size() >= 2) {
                // Binary form: the exponent is a second (Constant) SCALAR input.
                // The ConstantFolding/StrengthReduction passes model Pow this
                // way, so reading only the "exponent" attr would use a stale/
                // default value once the exponent has been moved to an input.
                auto exp_cpu = input_vars[1].tensor().to(DType::Float32).to(Device::cpu());
                float exponent = exp_cpu.data<float>()[0];
                auto result = tenzor::pow(input_vars[0].tensor(), exponent);
                outputs.push_back(Variable(result, false));
            } else if (!input_vars.empty()) {
                float exponent = node->get_attr("exponent");
                auto result = tenzor::pow(input_vars[0].tensor(), exponent);
                outputs.push_back(Variable(result, false));
            }
            break;

        case OpType::Abs:
            if (!input_vars.empty()) {
                outputs.push_back(tenzor::abs(input_vars[0]));
            }
            break;

        case OpType::Neg:
            if (!input_vars.empty()) {
                outputs.push_back(tenzor::neg(input_vars[0]));
            }
            break;

        case OpType::Clamp:
            if (!input_vars.empty()) {
                double min_val = static_cast<double>(static_cast<float>(node->get_attr("min")));
                double max_val = static_cast<double>(static_cast<float>(node->get_attr("max")));
                outputs.push_back(tenzor::clamp(input_vars[0], min_val, max_val));
            }
            break;

        // ====================================================================
        // Indexing
        // ====================================================================
        case OpType::Slice:
            if (!input_vars.empty()) {
                int64_t dim = node->get_int_attr("dim");
                int64_t start = node->get_int_attr("start");
                int64_t end = node->get_int_attr("end");
                int64_t step = node->has_attr("step") ? node->get_int_attr("step") : 1;
                outputs.push_back(tenzor::slice(input_vars[0], dim, start, end, step));
            }
            break;

        case OpType::Cat:
            if (!input_vars.empty()) {
                int64_t dim = node->has_attr("dim") ? node->get_int_attr("dim") : 0;
                outputs.push_back(tenzor::cat(input_vars, dim));
            }
            break;

        case OpType::Where:
            if (input_vars.size() >= 3) {
                outputs.push_back(tenzor::where(input_vars[0], input_vars[1], input_vars[2]));
            }
            break;

        case OpType::IndexSelect:
            if (input_vars.size() >= 2) {
                int64_t dim = node->has_attr("dim") ? node->get_int_attr("dim") : 0;
                outputs.push_back(tenzor::index_select(input_vars[0], dim, input_vars[1].tensor()));
            }
            break;

        case OpType::Interpolate:
            if (!input_vars.empty()) {
                auto size = node->get_vec_attr("output_size");
                if (size.empty()) {
                    throw std::runtime_error("Interpolate JIT node is missing output_size");
                }
                std::string size_str;
                for (size_t i = 0; i < size.size(); ++i) {
                    if (i > 0) size_str += ',';
                    size_str += std::to_string(size[i]);
                }
                OpAttributes attrs;
                attrs.set(AttrKey::OutputSize, std::string_view(size_str));
                attrs.set(AttrKey::Mode, mode_attr_to_string(*node));
                attrs.set(AttrKey::AlignCorners,
                          node->has_attr("align_corners") &&
                          node->get_bool_attr("align_corners"));
                if (size.size() == 1) {
                    attrs.set(AttrKey::OutputSizeW, size[0]);
                } else if (size.size() == 2) {
                    attrs.set(AttrKey::OutputSizeH, size[0]);
                    attrs.set(AttrKey::OutputSizeW, size[1]);
                } else if (size.size() == 3) {
                    attrs.set(AttrKey::OutputSizeD, size[0]);
                    attrs.set(AttrKey::OutputSizeH, size[1]);
                    attrs.set(AttrKey::OutputSizeW, size[2]);
                }
                std::vector<Tensor> inputs = {input_vars[0].tensor()};
                auto result = dispatch(OpId::Interpolate, inputs, attrs);
                if (!result.empty()) {
                    outputs.push_back(Variable(result[0], false));
                }
            }
            break;

        // ====================================================================
        // Other
        // ====================================================================
        case OpType::Dropout:
            // In JIT execution, dropout is typically in eval mode (no-op)
            if (!input_vars.empty()) {
                outputs.push_back(input_vars[0]);
            }
            break;

        case OpType::Linear:
            if (input_vars.size() >= 2) {
                Variable lin_out = (input_vars.size() >= 3)
                    ? tenzor::linear(input_vars[0], input_vars[1], input_vars[2])
                    // Linear without bias: y = x @ W^T
                    : input_vars[0].matmul(input_vars[1].transpose(0, 1));
                // optimize_for_inference (FuseLinearReLU) may fold a following
                // ReLU into this node and delete the ReLU node; apply it here.
                if (node->get_bool_attr("fused_relu")) {
                    lin_out = nn::relu(lin_out);
                }
                outputs.push_back(lin_out);
            }
            break;

        case OpType::Embedding:
            if (input_vars.size() >= 2) {
                // input_vars[0] = weight table, input_vars[1] = indices
                std::vector<Tensor> emb_inputs = {input_vars[0].tensor(), input_vars[1].tensor()};
                auto result = dispatch(OpId::Embedding, emb_inputs, {});
                if (!result.empty()) {
                    outputs.push_back(Variable(result[0], false));
                }
            }
            break;

        case OpType::GELU:
            if (!input_vars.empty()) {
                outputs.push_back(nn::gelu(input_vars[0]));
            }
            break;

        case OpType::SiLU:
            if (!input_vars.empty()) {
                outputs.push_back(input_vars[0] * nn::sigmoid(input_vars[0]));
            }
            break;

        case OpType::RMSNorm:
            if (!input_vars.empty()) {
                OpAttributes attrs;
                attrs.set(AttrKey::Eps, node->has_attr("eps") ?
                    static_cast<double>(node->get_attr("eps")) : 1e-5);
                std::vector<Tensor> inputs;
                inputs.reserve(input_vars.size());
                for (const auto& v : input_vars) {
                    inputs.push_back(v.tensor());
                }
                auto result = dispatch(OpId::RMSNorm, inputs, attrs);
                if (!result.empty()) {
                    outputs.push_back(Variable(result[0], false));
                }
            }
            break;

        // ====================================================================
        // Linear algebra operations
        // ====================================================================
        case OpType::GQA:
        case OpType::RoPE:
        case OpType::Padding:
            throw std::runtime_error(
                "JIT interpreter does not execute compiler-dialect op " +
                op_type_to_string(node->op_type()) +
                "; lower this graph through the MLIR/IREE path");

        case OpType::Det:
            if (!input_vars.empty()) {
                auto result = tenzor::linalg::det(input_vars[0].tensor());
                outputs.push_back(Variable(result, false));
            }
            break;

        case OpType::Inv:
            if (!input_vars.empty()) {
                auto result = tenzor::linalg::inv(input_vars[0].tensor());
                outputs.push_back(Variable(result, false));
            }
            break;

        case OpType::Solve:
            if (input_vars.size() >= 2) {
                auto result = tenzor::linalg::solve(input_vars[0].tensor(),
                                                     input_vars[1].tensor());
                outputs.push_back(Variable(result, false));
            }
            break;

        case OpType::Cholesky:
            if (!input_vars.empty()) {
                bool upper = node->get_bool_attr("upper");
                auto result = tenzor::linalg::cholesky(input_vars[0].tensor(), upper);
                outputs.push_back(Variable(result, false));
            }
            break;

        case OpType::Svd:
            if (!input_vars.empty()) {
                bool full_matrices = node->has_attr("full_matrices") ?
                    node->get_bool_attr("full_matrices") : true;
                auto [U, S, Vt] = tenzor::linalg::svd(input_vars[0].tensor(), full_matrices);
                outputs.push_back(Variable(U, false));
                outputs.push_back(Variable(S, false));
                outputs.push_back(Variable(Vt, false));
            }
            break;

        case OpType::Qr:
            if (!input_vars.empty()) {
                auto [Q, R] = tenzor::linalg::qr(input_vars[0].tensor());
                outputs.push_back(Variable(Q, false));
                outputs.push_back(Variable(R, false));
            }
            break;

        case OpType::Eigh:
            if (!input_vars.empty()) {
                auto [W, V] = tenzor::linalg::eigh(input_vars[0].tensor());
                outputs.push_back(Variable(W, false));
                outputs.push_back(Variable(V, false));
            }
            break;

        case OpType::Eigvalsh:
            if (!input_vars.empty()) {
                auto result = tenzor::linalg::eigvalsh(input_vars[0].tensor());
                outputs.push_back(Variable(result, false));
            }
            break;

        case OpType::Norm:
            if (!input_vars.empty()) {
                std::string ord = "fro";
                // Node stores norm order as a string attribute — check int_attrs and bool_attrs
                // for backward compatibility, but the primary storage is a float attr named "ord"
                // with special values, or we use a string convention in the node name.
                // Use a simple mapping: ord_type int attr (0=fro, 1=1-norm, 2=2-norm, 3=inf, 4=nuc)
                if (node->has_attr("ord_type")) {
                    int64_t ord_type = node->get_int_attr("ord_type");
                    switch (ord_type) {
                        case 0: ord = "fro"; break;
                        case 1: ord = "1"; break;
                        case 2: ord = "2"; break;
                        case 3: ord = "inf"; break;
                        case 4: ord = "nuc"; break;
                        default: ord = "fro"; break;
                    }
                }
                auto result = tenzor::linalg::norm(input_vars[0].tensor(), ord);
                outputs.push_back(Variable(result, false));
            }
            break;

        case OpType::Slogdet:
            if (!input_vars.empty()) {
                auto [sign, logabsdet] = tenzor::linalg::slogdet(input_vars[0].tensor());
                outputs.push_back(Variable(sign, false));
                outputs.push_back(Variable(logabsdet, false));
            }
            break;

        // ====================================================================
        // Constants and I/O markers
        // ====================================================================
        case OpType::Constant:
            if (node->has_attr("value")) {
                Tensor t = node->get_tensor_attr("value");
                outputs.push_back(Variable(t, false));
            }
            break;

        case OpType::Input:
        case OpType::Output:
            // These are graph markers, not executable ops.
            // Input values are pre-populated; Output values are gathered after.
            if (!input_vars.empty()) {
                outputs.push_back(input_vars[0]);
            }
            break;

        // ====================================================================
        // Shape guard — runtime check, triggers retrace on mismatch
        // ====================================================================
        case OpType::ShapeGuard: {
            // Input: tensor to check
            // Attribute: expected_shape (vec<int64_t>)
            if (!input_vars.empty()) {
                auto actual_shape = input_vars[0].tensor().shape();
                auto expected = node->get_vec_attr("expected_shape");
                bool match = (static_cast<int64_t>(actual_shape.size()) == static_cast<int64_t>(expected.size()));
                if (match) {
                    for (size_t i = 0; i < expected.size(); ++i) {
                        if (actual_shape[i] != expected[i]) {
                            match = false;
                            break;
                        }
                    }
                }
                if (!match) {
                    needs_retrace_ = true;
                }
                // Pass through input unchanged
                outputs.push_back(input_vars[0]);
            }
            break;
        }

        // ====================================================================
        // Control flow
        // ====================================================================
        case OpType::If: {
            // Inputs: [condition, then_inputs...]
            // condition is a scalar bool tensor
            if (!input_vars.empty() && node->then_branch()) {
                bool cond = input_vars[0].tensor().to(DType::Float32).item<float>() != 0.0f;
                auto& branch = cond ? node->then_branch() : node->else_branch();
                if (branch) {
                    // Pass remaining inputs to the chosen branch
                    std::vector<Variable> branch_inputs(input_vars.begin() + 1, input_vars.end());
                    auto branch_outputs = branch->forward(branch_inputs);
                    for (auto& out : branch_outputs) {
                        outputs.push_back(std::move(out));
                    }
                } else if (cond) {
                    // then_branch must exist (checked above), else_branch optional
                    auto branch_inputs = std::vector<Variable>(input_vars.begin() + 1, input_vars.end());
                    auto branch_outputs = node->then_branch()->forward(branch_inputs);
                    for (auto& out : branch_outputs) {
                        outputs.push_back(std::move(out));
                    }
                }
            }
            break;
        }

        // ====================================================================
        // Guard node — data-dependent branch guard, triggers retrace
        // ====================================================================
        case OpType::GuardNode: {
            if (!input_vars.empty()) {
                // GuardNode checks a boolean condition; if false, triggers retrace
                bool guard_val = input_vars[0].tensor().to(DType::Float32).item<float>() != 0.0f;
                bool expected = node->get_bool_attr("expected_value");
                if (guard_val != expected) {
                    needs_retrace_ = true;
                }
                // Pass through remaining inputs unchanged
                for (size_t i = 1; i < input_vars.size(); ++i) {
                    outputs.push_back(input_vars[i]);
                }
                if (input_vars.size() == 1) {
                    outputs.push_back(input_vars[0]);
                }
            }
            break;
        }

        // ====================================================================
        // Fused operations — dispatch to fused kernel implementations
        // ====================================================================
        case OpType::FlashAttention: {
            // Fused multi-head attention: softmax(scale * Q*K^T [+ mask]) * V
            // Inputs: [Q, K, V] or [Q, K, V, mask]
            if (input_vars.size() >= 3) {
                auto& Q = input_vars[0];
                auto& K = input_vars[1];
                auto& V = input_vars[2];

                // FuseAttentionPass matches MatMul -> Mul(scale) -> [Add(mask)]
                // -> Softmax -> MatMul and accepts ANY scalar constant as the
                // scale (temperature / learned), storing it as the "scale"
                // attr. Honour that recorded value rather than recomputing
                // 1/sqrt(d_k) — otherwise a traced model using a non-1/sqrt(d_k)
                // scale silently diverges from eager.
                float scale;
                if (node->has_float_attr("scale")) {
                    scale = node->get_attr("scale");
                } else {
                    auto d_k = static_cast<float>(K.tensor().shape().back());
                    scale = 1.0f / std::sqrt(d_k);
                }

                auto scores = tenzor::matmul(Q, K.transpose(-2, -1));
                // Match the score dtype (Q/K may be Float64/half); a hardcoded
                // Float32 scalar would force/round the scaled scores to Float32,
                // diverging from eager for non-Float32 traces.
                auto scaled = scores * Variable(
                    tenzor::full({1}, scale, scores.tensor().dtype()), false);

                // Add the additive mask (4th input) before softmax when the
                // matched pattern attached one (FuseAttentionPass appends it as
                // input[3] and sets has_mask). Either signal suffices; the 4th
                // input is authoritative.
                if (input_vars.size() >= 4) {
                    scaled = scaled + input_vars[3];
                }

                auto attn = nn::softmax(scaled, -1);
                outputs.push_back(tenzor::matmul(attn, V));
            }
            break;
        }

        case OpType::FusedFFN: {
            // Fused feed-forward: Linear -> Activation -> Linear
            // Inputs may arrive either as [x, w1, b1, w2, b2] (all graph inputs)
            // or as [x] with w1/b1/w2/b2 carried as tensor attrs (the form
            // FuseFFNPass::fuse_triple emits — it transfers only linear1's data
            // input and stores the weights as attrs). Support both so the fused
            // node is actually executable.
            if (!input_vars.empty()) {
                Variable x = input_vars[0];
                Variable w1, b1, w2, b2;
                bool have_b1 = false, have_b2 = false;

                if (input_vars.size() >= 5) {
                    w1 = input_vars[1];
                    b1 = input_vars[2]; have_b1 = true;
                    w2 = input_vars[3];
                    b2 = input_vars[4]; have_b2 = true;
                } else {
                    // Pull weights/biases from the node's tensor attrs.
                    w1 = Variable(node->get_tensor_attr("weight1"), false);
                    w2 = Variable(node->get_tensor_attr("weight2"), false);
                    if (node->has_tensor_attr("bias1") &&
                        node->get_tensor_attr("bias1").numel() > 0) {
                        b1 = Variable(node->get_tensor_attr("bias1"), false);
                        have_b1 = true;
                    }
                    if (node->has_tensor_attr("bias2") &&
                        node->get_tensor_attr("bias2").numel() > 0) {
                        b2 = Variable(node->get_tensor_attr("bias2"), false);
                        have_b2 = true;
                    }
                }

                auto h = tenzor::matmul(x, w1.transpose(-2, -1));
                if (have_b1) h = h + b1;

                // activation_type: 1 = ReLU, 2 = GELU (default). Set by
                // FuseFFNPass; default to GELU when absent.
                int64_t act = node->has_attr("activation_type")
                                  ? node->get_int_attr("activation_type")
                                  : 2;
                h = (act == 1) ? nn::relu(h) : nn::gelu(h);

                auto out = tenzor::matmul(h, w2.transpose(-2, -1));
                if (have_b2) out = out + b2;
                outputs.push_back(out);
            }
            break;
        }

        case OpType::ResidualAdd: {
            // Residual connection: x + sublayer(x)
            // Inputs: [original, sublayer_output]
            if (input_vars.size() >= 2) {
                outputs.push_back(input_vars[0] + input_vars[1]);
            }
            break;
        }

        // ====================================================================
        // Memory management pseudo-ops (no-ops during normal execution)
        // ====================================================================
        case OpType::SwapOut:
        case OpType::SwapIn: {
            // These are hints for memory management; pass through during execution
            if (!input_vars.empty()) {
                outputs.push_back(input_vars[0]);
            }
            break;
        }

        case OpType::Loop: {
            // ONNX-style loop semantics:
            // Inputs: [max_iterations, condition, carried_0, carried_1, ...]
            // Body graph inputs: [iteration_num, condition, carried_0, carried_1, ...]
            // Body graph outputs: [condition, carried_0, carried_1, ...]
            // Loop outputs: [final_carried_0, final_carried_1, ...]
            if (input_vars.size() >= 2 && node->body()) {
                int64_t max_iter = static_cast<int64_t>(input_vars[0].tensor().to(DType::Float32).item<float>());
                bool cond = input_vars[1].tensor().to(DType::Float32).item<float>() != 0.0f;

                // Initialize loop-carried values
                std::vector<Variable> carried(input_vars.begin() + 2, input_vars.end());

                for (int64_t i = 0; i < max_iter && cond; ++i) {
                    // Build body inputs: [iter, cond, carried...]
                    std::vector<Variable> body_inputs;
                    body_inputs.push_back(Variable(
                        tenzor::full({1}, static_cast<float>(i), DType::Float32), false));
                    body_inputs.push_back(Variable(
                        tenzor::full({1}, cond ? 1.0f : 0.0f, DType::Float32), false));
                    for (auto& c : carried) {
                        body_inputs.push_back(c);
                    }

                    auto body_outputs = node->body()->forward(body_inputs);

                    // body_outputs[0] = new condition, rest = updated carried values
                    if (!body_outputs.empty()) {
                        cond = body_outputs[0].tensor().to(DType::Float32).item<float>() != 0.0f;
                        carried.clear();
                        for (size_t j = 1; j < body_outputs.size(); ++j) {
                            carried.push_back(std::move(body_outputs[j]));
                        }
                    } else {
                        break;
                    }
                }

                // Loop outputs are the final carried values
                for (auto& c : carried) {
                    outputs.push_back(std::move(c));
                }
            }
            break;
        }

        case OpType::LayoutConvert: {
            // Convert memory format
            if (!input_vars.empty()) {
                int64_t fmt = node->get_int_attr("target_format");
                auto mem_fmt = (fmt == 1) ? MemoryFormat::ChannelsLast
                             : (fmt == 2) ? MemoryFormat::ChannelsLast3d
                             : MemoryFormat::Contiguous;
                outputs.push_back(Variable(input_vars[0].tensor().to(mem_fmt), false));
            }
            break;
        }

        case OpType::Cast: {
            // Convert dtype
            if (!input_vars.empty()) {
                int64_t target_dtype_int = node->get_int_attr("target_dtype");
                auto target_dtype = static_cast<DType>(target_dtype_int);
                outputs.push_back(Variable(input_vars[0].tensor().to(target_dtype), false));
            }
            break;
        }

        case OpType::QuantizedLinear:
        case OpType::QuantizedConv2d:
        case OpType::Dequantize:
        case OpType::Quantize:
        case OpType::SparseMatMul:
        case OpType::DenseToSparse:
            throw std::runtime_error(
                "JIT interpreter does not execute specialized op " +
                op_type_to_string(node->op_type()) +
                "; lower this graph through a backend compiler/codegen path");
    }

    // Store outputs in value map
    for (size_t i = 0; i < node->outputs().size() && i < outputs.size(); ++i) {
        value_map[node->outputs()[i]->id()] = outputs[i];
    }
}

auto Graph::find_nodes_by_type(const std::string& type_name) const -> std::vector<std::shared_ptr<Node>> {
    std::vector<std::shared_ptr<Node>> result;
    for (const auto& node : nodes_) {
        if (op_type_to_string(node->op_type()) == type_name) {
            result.push_back(node);
        }
    }
    return result;
}

auto Graph::save(const std::string& path) const -> void {
    save_graph(*this, path);
}

auto Graph::load(const std::string& path) -> std::shared_ptr<Graph> {
    return load_graph(path);
}

auto Graph::to_string() const -> std::string {
    std::ostringstream oss;
    oss << "Graph {\n";
    oss << "  Inputs: " << inputs_.size() << "\n";
    oss << "  Outputs: " << outputs_.size() << "\n";
    oss << "  Nodes: " << nodes_.size() << "\n";
    oss << "  Values: " << values_.size() << "\n\n";

    oss << "  Nodes:\n";
    for (const auto& node : nodes_) {
        oss << "    " << node->name() << " [" << op_type_to_string(node->op_type()) << "]\n";
        oss << "      Inputs: ";
        for (const auto& input : node->inputs()) {
            oss << input->id();
            if (input->has_symbolic_shape()) {
                oss << input->symbolic_shape().to_string();
            }
            oss << " ";
        }
        oss << "\n      Outputs: ";
        for (const auto& output : node->outputs()) {
            oss << output->id();
            if (output->has_symbolic_shape()) {
                oss << output->symbolic_shape().to_string();
            }
            oss << " ";
        }
        oss << "\n";
    }

    oss << "}\n";
    return oss.str();
}

auto Graph::bind_symbolic_shapes(const SymbolicShapeEnvironment& env) -> void {
    // Resolve symbolic dimensions on all values in the graph
    for (auto& [id, value] : values_) {
        if (value && value->has_symbolic_shape()) {
            auto& sym_shape = value->symbolic_shape();
            if (env.can_resolve(sym_shape)) {
                auto concrete = env.resolve(sym_shape);
                value->set_shape(concrete);
            }
        }
    }
    // Also resolve input shapes
    for (auto& input : inputs_) {
        if (input && input->has_symbolic_shape()) {
            auto& sym_shape = input->symbolic_shape();
            if (env.can_resolve(sym_shape)) {
                auto concrete = env.resolve(sym_shape);
                input->set_shape(concrete);
            }
        }
    }
}

auto Graph::symbolic_input_shapes() const -> std::vector<SymbolicShape> {
    std::vector<SymbolicShape> result;
    result.reserve(inputs_.size());
    for (const auto& input : inputs_) {
        if (input && input->has_symbolic_shape()) {
            result.push_back(input->symbolic_shape());
        } else {
            result.emplace_back();  // Empty shape for non-symbolic inputs
        }
    }
    return result;
}

auto Graph::partition_at(const std::vector<size_t>& break_indices)
    -> std::vector<std::shared_ptr<Graph>> {

    if (break_indices.empty()) {
        // No breaks - return empty (caller should just use the original graph)
        return {};
    }

    std::vector<std::shared_ptr<Graph>> partitions;
    auto& all_nodes = nodes();
    size_t total = all_nodes.size();

    // Create sorted, deduplicated break points
    std::vector<size_t> breaks(break_indices.begin(), break_indices.end());
    std::sort(breaks.begin(), breaks.end());
    breaks.erase(std::unique(breaks.begin(), breaks.end()), breaks.end());

    // Build partition ranges: [start, end) excluding break nodes
    std::vector<std::pair<size_t, size_t>> ranges;
    size_t start = 0;
    for (size_t bp : breaks) {
        if (bp > start) {
            ranges.push_back({start, bp});
        }
        // Skip the break node itself
        start = bp + 1;
    }
    if (start < total) {
        ranges.push_back({start, total});
    }

    // Map every value id to the partition that produces it. Values not produced
    // by any node are external (graph inputs / constants).
    std::unordered_map<std::string, size_t> value_to_partition;

    // Values that are outputs of the overall graph must remain outputs of
    // whichever partition produces them.
    std::unordered_set<std::string> graph_output_ids;
    for (const auto& out : outputs_) {
        if (out) graph_output_ids.insert(out->id());
    }

    // First pass: which partition produces each value.
    for (size_t p = 0; p < ranges.size(); ++p) {
        auto [range_start, range_end] = ranges[p];
        for (size_t i = range_start; i < range_end; ++i) {
            for (const auto& output : all_nodes[i]->outputs()) {
                value_to_partition[output->id()] = p;
            }
        }
    }

    // Second pass: build each sub-graph and wire its boundary inputs/outputs.
    for (size_t p = 0; p < ranges.size(); ++p) {
        auto sub_graph = std::make_shared<Graph>();
        auto [range_start, range_end] = ranges[p];

        std::unordered_set<std::string> local_values;   // produced in this partition
        std::unordered_set<std::string> input_ids_seen;
        std::vector<std::shared_ptr<Value>> sub_inputs;
        std::vector<std::shared_ptr<Value>> sub_outputs;
        std::unordered_set<std::string> output_ids_seen;

        // Boundary inputs: a value consumed by a node in this partition that is
        // produced by an earlier partition or is external to the graph.
        for (size_t i = range_start; i < range_end; ++i) {
            const auto& node = all_nodes[i];
            sub_graph->add_node(node);

            for (const auto& input : node->inputs()) {
                const auto& id = input->id();
                if (local_values.count(id)) {
                    continue;  // produced inside this partition already
                }
                auto prod = value_to_partition.find(id);
                if (prod == value_to_partition.end() || prod->second != p) {
                    // Produced by another partition, or a graph input/constant.
                    if (input_ids_seen.insert(id).second) {
                        sub_inputs.push_back(input);
                    }
                }
            }

            for (const auto& output : node->outputs()) {
                local_values.insert(output->id());
            }
        }

        // Boundary outputs: a value produced here that is consumed by a later
        // partition, or that is an output of the overall graph.
        for (size_t i = range_start; i < range_end; ++i) {
            for (const auto& output : all_nodes[i]->outputs()) {
                const auto& id = output->id();
                bool is_graph_output = graph_output_ids.count(id) != 0;
                bool consumed_later = false;
                if (!is_graph_output) {
                    for (size_t q = p + 1; q < ranges.size() && !consumed_later; ++q) {
                        auto [qs, qe] = ranges[q];
                        for (size_t j = qs; j < qe; ++j) {
                            for (const auto& in : all_nodes[j]->inputs()) {
                                if (in->id() == id) { consumed_later = true; break; }
                            }
                            if (consumed_later) break;
                        }
                    }
                }
                if ((is_graph_output || consumed_later) &&
                    output_ids_seen.insert(id).second) {
                    sub_outputs.push_back(output);
                }
            }
        }

        sub_graph->set_inputs(std::move(sub_inputs));
        sub_graph->set_outputs(std::move(sub_outputs));
        partitions.push_back(sub_graph);
    }

    return partitions;
}

} // namespace jit
} // namespace tenzor
