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
#include "../../include/tenzor/ops/linalg.hpp"
#include "../../include/tenzor/autograd/ops.hpp"
#include "../../include/tenzor/nn/functional.hpp"
#include "../../include/tenzor/backend/fast_dispatch.hpp"
#include "../../include/tenzor/core/shape.hpp"
#include <algorithm>
#include <queue>
#include <sstream>
#include <stdexcept>

namespace tenzor {
namespace jit {

// ============================================================================
// Node implementation
// ============================================================================

auto Node::add_input(std::shared_ptr<Value> val) -> void {
    inputs_.push_back(val);
    val->add_use(shared_from_this());
}

auto Node::add_output(std::shared_ptr<Value> val) -> void {
    outputs_.push_back(val);
}

auto Node::replace_input(size_t idx, std::shared_ptr<Value> val) -> void {
    if (idx >= inputs_.size()) {
        throw std::runtime_error("Input index out of bounds");
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

        // Find the shared_ptr for this node
        for (const auto& n : nodes_) {
            if (n.get() == node) {
                sorted.push_back(n);
                break;
            }
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
    // Type inference pass - propagate shapes through operations
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
                if (input_shapes.size() >= 2 && input_shapes[0].size() >= 2 && input_shapes[1].size() >= 2) {
                    auto N = input_shapes[1][input_shapes[1].size() - 1];
                    std::vector<int64_t> out_shape = input_shapes[0];
                    out_shape[out_shape.size() - 1] = N;
                    output_shapes.push_back(out_shape);
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
                    output_shapes.push_back(node->get_vec_attr("shape"));
                }
                break;

            case OpType::Transpose:
                if (!input_shapes.empty()) {
                    auto shape = input_shapes[0];
                    int64_t dim0 = node->get_int_attr("dim0");
                    int64_t dim1 = node->get_int_attr("dim1");
                    if (dim0 < static_cast<int64_t>(shape.size()) && dim1 < static_cast<int64_t>(shape.size())) {
                        std::swap(shape[dim0], shape[dim1]);
                    }
                    output_shapes.push_back(shape);
                }
                break;

            case OpType::Permute:
                if (!input_shapes.empty() && node->has_attr("dims")) {
                    auto dims = node->get_vec_attr("dims");
                    auto& in_shape = input_shapes[0];
                    std::vector<int64_t> out_shape(dims.size());
                    for (size_t i = 0; i < dims.size(); ++i) {
                        if (dims[i] < static_cast<int64_t>(in_shape.size())) {
                            out_shape[i] = in_shape[dims[i]];
                        }
                    }
                    output_shapes.push_back(out_shape);
                }
                break;

            case OpType::Squeeze:
                if (!input_shapes.empty()) {
                    auto shape = input_shapes[0];
                    int64_t dim = node->get_int_attr("dim");
                    if (dim < static_cast<int64_t>(shape.size()) && shape[dim] == 1) {
                        shape.erase(shape.begin() + dim);
                    }
                    output_shapes.push_back(shape);
                }
                break;

            case OpType::Unsqueeze:
                if (!input_shapes.empty()) {
                    auto shape = input_shapes[0];
                    int64_t dim = node->get_int_attr("dim");
                    if (dim < 0) dim += static_cast<int64_t>(shape.size()) + 1;
                    if (dim <= static_cast<int64_t>(shape.size())) {
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
                    if (node->has_attr("dim")) {
                        int64_t dim = node->get_int_attr("dim");
                        bool keepdim = node->get_bool_attr("keepdim");
                        if (dim < 0) dim += static_cast<int64_t>(shape.size());
                        if (dim >= 0 && dim < static_cast<int64_t>(shape.size())) {
                            if (keepdim) {
                                shape[dim] = 1;
                            } else {
                                shape.erase(shape.begin() + dim);
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
                    int64_t out_channels = node->get_int_attr("out_channels");
                    int64_t kernel_h = node->get_int_attr("kernel_h");
                    int64_t kernel_w = node->get_int_attr("kernel_w");
                    int64_t stride_h = node->has_attr("stride_h") ? node->get_int_attr("stride_h") : 1;
                    int64_t stride_w = node->has_attr("stride_w") ? node->get_int_attr("stride_w") : 1;
                    int64_t padding_h = node->has_attr("padding_h") ? node->get_int_attr("padding_h") : 0;
                    int64_t padding_w = node->has_attr("padding_w") ? node->get_int_attr("padding_w") : 0;
                    int64_t dilation_h = node->has_attr("dilation_h") ? node->get_int_attr("dilation_h") : 1;
                    int64_t dilation_w = node->has_attr("dilation_w") ? node->get_int_attr("dilation_w") : 1;

                    if (shape.size() == 4) {
                        int64_t H_out = (shape[2] + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
                        int64_t W_out = (shape[3] + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
                        output_shapes.push_back({shape[0], out_channels, H_out, W_out});
                    }
                }
                break;

            // ================================================================
            // Pooling
            // ================================================================
            case OpType::MaxPool2d:
            case OpType::AvgPool2d:
                if (!input_shapes.empty() && input_shapes[0].size() == 4) {
                    auto& shape = input_shapes[0];  // [N, C, H, W]
                    int64_t kernel = node->get_int_attr("kernel_size");
                    int64_t stride = node->has_attr("stride") ? node->get_int_attr("stride") : kernel;
                    int64_t padding = node->has_attr("padding") ? node->get_int_attr("padding") : 0;
                    int64_t H_out = (shape[2] + 2 * padding - kernel) / stride + 1;
                    int64_t W_out = (shape[3] + 2 * padding - kernel) / stride + 1;
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
                    // Weight shape is [out_features, in_features]
                    if (!input_shapes[1].empty()) {
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
                    int64_t dim = node->get_int_attr("dim");
                    int64_t start = node->get_int_attr("start");
                    int64_t end = node->get_int_attr("end");
                    int64_t step = node->has_attr("step") ? node->get_int_attr("step") : 1;
                    if (dim < static_cast<int64_t>(shape.size())) {
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
                    if (dim < static_cast<int64_t>(out_shape.size())) {
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
                    // Output shape: same as input[0] except last dim = input[1]'s last dim
                    SymbolicShape out_shape(input_sym_shapes[0].dims());
                    out_shape[out_shape.rank() - 1] =
                        input_sym_shapes[1][input_sym_shapes[1].rank() - 1];
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
                    // Reshape target is concrete from the vec_attr
                    output_sym_shapes.push_back(
                        SymbolicShape::from_concrete(node->get_vec_attr("shape")));
                }
                break;

            case OpType::Transpose:
                if (!input_sym_shapes.empty()) {
                    auto sym_shape = input_sym_shapes[0];
                    int64_t dim0 = node->get_int_attr("dim0");
                    int64_t dim1 = node->get_int_attr("dim1");
                    if (dim0 < static_cast<int64_t>(sym_shape.rank()) &&
                        dim1 < static_cast<int64_t>(sym_shape.rank())) {
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
                    std::vector<SymbolicDim> out_dims(dims.size());
                    for (size_t i = 0; i < dims.size(); ++i) {
                        if (dims[i] < static_cast<int64_t>(in_shape.rank())) {
                            out_dims[i] = in_shape[static_cast<size_t>(dims[i])];
                        }
                    }
                    output_sym_shapes.push_back(SymbolicShape(std::move(out_dims)));
                }
                break;

            case OpType::Squeeze:
                if (!input_sym_shapes.empty()) {
                    auto sym_shape = input_sym_shapes[0];
                    int64_t dim = node->get_int_attr("dim");
                    if (dim < static_cast<int64_t>(sym_shape.rank())) {
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
                    if (dim <= static_cast<int64_t>(sym_shape.rank())) {
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
                    if (node->has_attr("dim")) {
                        int64_t dim = node->get_int_attr("dim");
                        bool keepdim = node->get_bool_attr("keepdim");
                        if (dim < 0) dim += static_cast<int64_t>(sym_shape.rank());
                        if (dim >= 0 && dim < static_cast<int64_t>(sym_shape.rank())) {
                            if (keepdim) {
                                sym_shape[static_cast<size_t>(dim)] = SymbolicDim::concrete(1);
                            } else {
                                sym_shape.erase(static_cast<size_t>(dim));
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
                    int64_t stride_h = node->has_attr("stride_h") ? node->get_int_attr("stride_h") : 1;
                    int64_t stride_w = node->has_attr("stride_w") ? node->get_int_attr("stride_w") : 1;
                    int64_t padding_h = node->has_attr("padding_h") ? node->get_int_attr("padding_h") : 0;
                    int64_t padding_w = node->has_attr("padding_w") ? node->get_int_attr("padding_w") : 0;
                    int64_t dilation_h = node->has_attr("dilation_h") ? node->get_int_attr("dilation_h") : 1;
                    int64_t dilation_w = node->has_attr("dilation_w") ? node->get_int_attr("dilation_w") : 1;

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
                    int64_t dim = node->get_int_attr("dim");
                    int64_t start = node->get_int_attr("start");
                    int64_t end = node->get_int_attr("end");
                    int64_t step = node->has_attr("step") ? node->get_int_attr("step") : 1;
                    if (dim < static_cast<int64_t>(sym_shape.rank())) {
                        int64_t len = (end - start + step - 1) / step;
                        sym_shape[static_cast<size_t>(dim)] = SymbolicDim::concrete(len);
                    }
                    output_sym_shapes.push_back(std::move(sym_shape));
                }
                break;

            case OpType::Cat:
                if (!input_sym_shapes.empty()) {
                    int64_t dim = node->has_attr("dim") ? node->get_int_attr("dim") : 0;
                    auto out_shape = input_sym_shapes[0];
                    if (dim < static_cast<int64_t>(out_shape.rank())) {
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
                // (..., M, N) -> U, S, Vt (3 outputs)
                if (!input_sym_shapes.empty() && input_sym_shapes[0].rank() >= 2) {
                    auto& s = input_sym_shapes[0];
                    auto M = s[s.rank() - 2];
                    auto N_dim = s[s.rank() - 1];
                    // For symbolic shapes, use the input dims directly
                    // Batch dims
                    std::vector<SymbolicDim> batch_dims;
                    for (size_t d = 0; d + 2 < s.rank(); ++d) {
                        batch_dims.push_back(s[d]);
                    }
                    // U shape: batch + [M, M] (full) or [M, min(M,N)]
                    auto u_dims = batch_dims;
                    u_dims.push_back(M);
                    u_dims.push_back(M);  // Assume full matrices for symbolic
                    output_sym_shapes.push_back(SymbolicShape(std::move(u_dims)));
                    // S shape: batch + [min(M,N)] - use M as approximation for symbolic
                    auto s_dims = batch_dims;
                    s_dims.push_back(M);  // Approximation
                    output_sym_shapes.push_back(SymbolicShape(std::move(s_dims)));
                    // Vt shape: batch + [N, N]
                    auto vt_dims = batch_dims;
                    vt_dims.push_back(N_dim);
                    vt_dims.push_back(N_dim);
                    output_sym_shapes.push_back(SymbolicShape(std::move(vt_dims)));
                }
                break;

            case OpType::Qr:
                // (..., M, N) -> Q: (..., M, K), R: (..., K, N)
                if (!input_sym_shapes.empty() && input_sym_shapes[0].rank() >= 2) {
                    auto& s = input_sym_shapes[0];
                    auto M = s[s.rank() - 2];
                    auto N_dim = s[s.rank() - 1];
                    std::vector<SymbolicDim> batch_dims;
                    for (size_t d = 0; d + 2 < s.rank(); ++d) {
                        batch_dims.push_back(s[d]);
                    }
                    // Q: batch + [M, K] where K=min(M,N) - use M as approx
                    auto q_dims = batch_dims;
                    q_dims.push_back(M);
                    q_dims.push_back(M);
                    output_sym_shapes.push_back(SymbolicShape(std::move(q_dims)));
                    // R: batch + [K, N]
                    auto r_dims = batch_dims;
                    r_dims.push_back(M);
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
                outputs.push_back(tenzor::matmul(input_vars[0], input_vars[1]));
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
                pool_attrs.set(AttrKey::KernelSize, node->get_int_attr("kernel_size"));
                pool_attrs.set(AttrKey::Stride, node->has_attr("stride") ?
                    node->get_int_attr("stride") : node->get_int_attr("kernel_size"));
                pool_attrs.set(AttrKey::Padding, node->has_attr("padding") ?
                    node->get_int_attr("padding") : static_cast<int64_t>(0));
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
                pool_attrs.set(AttrKey::KernelSize, node->get_int_attr("kernel_size"));
                pool_attrs.set(AttrKey::Stride, node->has_attr("stride") ?
                    node->get_int_attr("stride") : node->get_int_attr("kernel_size"));
                pool_attrs.set(AttrKey::Padding, node->has_attr("padding") ?
                    node->get_int_attr("padding") : static_cast<int64_t>(0));
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
                OpAttributes conv_attrs;
                conv_attrs.set(AttrKey::Stride, node->has_attr("stride_h") ?
                    node->get_int_attr("stride_h") : static_cast<int64_t>(1));
                conv_attrs.set(AttrKey::Padding, node->has_attr("padding_h") ?
                    node->get_int_attr("padding_h") : static_cast<int64_t>(0));
                conv_attrs.set(AttrKey::Dilation, node->has_attr("dilation") ?
                    node->get_int_attr("dilation") : static_cast<int64_t>(1));
                conv_attrs.set(AttrKey::Groups, node->has_attr("groups") ?
                    node->get_int_attr("groups") : static_cast<int64_t>(1));
                std::vector<Tensor> inputs = {input_vars[0].tensor(), input_vars[1].tensor()};
                if (input_vars.size() >= 3) {
                    inputs.push_back(input_vars[2].tensor());
                }
                auto result = dispatch(OpId::Conv2dForward, inputs, conv_attrs);
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

        // ====================================================================
        // Reductions
        // ====================================================================
        case OpType::Sum:
            if (!input_vars.empty()) {
                if (node->has_attr("dim")) {
                    int64_t dim = node->get_int_attr("dim");
                    bool keepdim = node->has_attr("keepdim") ? node->get_bool_attr("keepdim") : false;
                    outputs.push_back(tenzor::sum(input_vars[0], dim, keepdim));
                } else {
                    outputs.push_back(tenzor::sum(input_vars[0]));
                }
            }
            break;

        case OpType::Mean:
            if (!input_vars.empty()) {
                if (node->has_attr("dim")) {
                    int64_t dim = node->get_int_attr("dim");
                    bool keepdim = node->has_attr("keepdim") ? node->get_bool_attr("keepdim") : false;
                    outputs.push_back(tenzor::mean(input_vars[0], dim, keepdim));
                } else {
                    outputs.push_back(tenzor::mean(input_vars[0]));
                }
            }
            break;

        case OpType::Max:
            if (!input_vars.empty()) {
                if (node->has_attr("dim")) {
                    int64_t dim = node->get_int_attr("dim");
                    bool keepdim = node->has_attr("keepdim") ? node->get_bool_attr("keepdim") : false;
                    outputs.push_back(tenzor::max(input_vars[0], dim, keepdim));
                } else {
                    outputs.push_back(tenzor::max(input_vars[0]));
                }
            }
            break;

        case OpType::Min:
            if (!input_vars.empty()) {
                // No autograd min — use raw tensor op and wrap
                if (node->has_attr("dim")) {
                    int64_t dim = node->get_int_attr("dim");
                    bool keepdim = node->has_attr("keepdim") ? node->get_bool_attr("keepdim") : false;
                    auto result = tenzor::min(input_vars[0].tensor(), dim, keepdim);
                    outputs.push_back(Variable(result, false));
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
            if (!input_vars.empty()) {
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
                float min_val = node->get_attr("min");
                float max_val = node->get_attr("max");
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
                if (input_vars.size() >= 3) {
                    outputs.push_back(tenzor::linear(input_vars[0], input_vars[1], input_vars[2]));
                } else {
                    // Linear without bias: y = x @ W^T
                    outputs.push_back(input_vars[0].matmul(input_vars[1].transpose(0, 1)));
                }
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

        // ====================================================================
        // Linear algebra operations
        // ====================================================================
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
                bool cond = input_vars[0].tensor().template item<float>() != 0.0f;
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

        case OpType::Loop: {
            // ONNX-style loop semantics:
            // Inputs: [max_iterations, condition, carried_0, carried_1, ...]
            // Body graph inputs: [iteration_num, condition, carried_0, carried_1, ...]
            // Body graph outputs: [condition, carried_0, carried_1, ...]
            // Loop outputs: [final_carried_0, final_carried_1, ...]
            if (input_vars.size() >= 2 && node->body()) {
                int64_t max_iter = static_cast<int64_t>(input_vars[0].tensor().template item<float>());
                bool cond = input_vars[1].tensor().template item<float>() != 0.0f;

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
                        cond = body_outputs[0].tensor().template item<float>() != 0.0f;
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

} // namespace jit
} // namespace tenzor
