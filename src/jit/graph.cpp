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

    // Clear uses from input values
    for (auto& input : node->inputs()) {
        input->clear_uses();
    }
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
            case OpType::Add:
            case OpType::Sub:
            case OpType::Mul:
            case OpType::Div:
                // Binary ops: output shape is broadcast result
                if (input_shapes.size() >= 2) {
                    // For simplicity, assume same shape (proper broadcasting would be more complex)
                    output_shapes.push_back(input_shapes[0]);
                }
                break;

            case OpType::MatMul:
                // Matrix multiplication: (M, K) @ (K, N) -> (M, N)
                if (input_shapes.size() >= 2 && input_shapes[0].size() >= 2 && input_shapes[1].size() >= 2) {
                    auto M = input_shapes[0][input_shapes[0].size() - 2];
                    auto N = input_shapes[1][input_shapes[1].size() - 1];
                    std::vector<int64_t> out_shape = input_shapes[0];
                    out_shape[out_shape.size() - 1] = N;
                    output_shapes.push_back(out_shape);
                }
                break;

            case OpType::ReLU:
            case OpType::Sigmoid:
            case OpType::Tanh:
            case OpType::Exp:
            case OpType::Log:
            case OpType::Sqrt:
            case OpType::Abs:
            case OpType::Neg:
                // Unary ops: preserve input shape
                if (!input_shapes.empty()) {
                    output_shapes.push_back(input_shapes[0]);
                }
                break;

            case OpType::Reshape:
                // Reshape: use target shape from attributes
                if (node->has_attr("shape")) {
                    output_shapes.push_back(node->get_vec_attr("shape"));
                }
                break;

            case OpType::Transpose:
                // Transpose: swap two dimensions
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

            case OpType::Sum:
            case OpType::Mean:
            case OpType::Max:
            case OpType::Min:
                // Reductions: remove dimension or keep with size 1
                if (!input_shapes.empty()) {
                    auto shape = input_shapes[0];
                    if (node->has_attr("dim")) {
                        int64_t dim = node->get_int_attr("dim");
                        bool keepdim = node->get_bool_attr("keepdim");
                        if (dim < static_cast<int64_t>(shape.size())) {
                            if (keepdim) {
                                shape[dim] = 1;
                            } else {
                                shape.erase(shape.begin() + dim);
                            }
                        }
                    } else {
                        // Reduce all: scalar output
                        shape = {};
                    }
                    output_shapes.push_back(shape);
                }
                break;

            case OpType::Conv2d:
                // Convolution: complex shape calculation
                if (!input_shapes.empty()) {
                    auto shape = input_shapes[0];  // [N, C, H, W]
                    int64_t out_channels = node->get_int_attr("out_channels");
                    int64_t kernel_h = node->get_int_attr("kernel_h");
                    int64_t kernel_w = node->get_int_attr("kernel_w");
                    int64_t stride_h = node->get_int_attr("stride_h");
                    int64_t stride_w = node->get_int_attr("stride_w");
                    int64_t padding_h = node->get_int_attr("padding_h");
                    int64_t padding_w = node->get_int_attr("padding_w");

                    if (shape.size() == 4) {
                        int64_t H_out = (shape[2] + 2 * padding_h - kernel_h) / stride_h + 1;
                        int64_t W_out = (shape[3] + 2 * padding_w - kernel_w) / stride_w + 1;
                        output_shapes.push_back({shape[0], out_channels, H_out, W_out});
                    }
                }
                break;

            default:
                // For unknown ops, try to preserve input shape
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

        case OpType::MatMul:
            if (input_vars.size() >= 2) {
                outputs.push_back(input_vars[0].matmul(input_vars[1]));
            }
            break;

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

        case OpType::Constant:
            // Constants are stored as tensor attributes
            if (node->has_attr("value")) {
                Tensor t = node->get_tensor_attr("value");
                outputs.push_back(Variable(t, false));
            }
            break;

        default:
            // For unimplemented ops, just pass through first input
            if (!input_vars.empty()) {
                outputs.push_back(input_vars[0]);
            }
            break;
    }

    // Store outputs in value map
    for (size_t i = 0; i < node->outputs().size() && i < outputs.size(); ++i) {
        value_map[node->outputs()[i]->id()] = outputs[i];
    }
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
            oss << input->id() << " ";
        }
        oss << "\n      Outputs: ";
        for (const auto& output : node->outputs()) {
            oss << output->id() << " ";
        }
        oss << "\n";
    }

    oss << "}\n";
    return oss.str();
}

} // namespace jit
} // namespace tenzor
