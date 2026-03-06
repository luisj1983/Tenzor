/**
 * @file tracer.cpp
 * @brief Implementation of JIT operation tracing
 */

#include "../../include/tenzor/jit/tracer.hpp"
#include "../../include/tenzor/jit/compiler.hpp"
#include <sstream>
#include <iomanip>
#include <stdexcept>

namespace tenzor {
namespace jit {

// ============================================================================
// OpType conversion functions
// ============================================================================

auto op_type_to_string(OpType type) -> std::string {
    switch (type) {
        case OpType::Add: return "Add";
        case OpType::Sub: return "Sub";
        case OpType::Mul: return "Mul";
        case OpType::Div: return "Div";
        case OpType::MatMul: return "MatMul";
        case OpType::Bmm: return "Bmm";
        case OpType::ReLU: return "ReLU";
        case OpType::Sigmoid: return "Sigmoid";
        case OpType::Tanh: return "Tanh";
        case OpType::Softmax: return "Softmax";
        case OpType::LogSoftmax: return "LogSoftmax";
        case OpType::MaxPool2d: return "MaxPool2d";
        case OpType::AvgPool2d: return "AvgPool2d";
        case OpType::AdaptiveAvgPool2d: return "AdaptiveAvgPool2d";
        case OpType::Conv2d: return "Conv2d";
        case OpType::BatchNorm2d: return "BatchNorm2d";
        case OpType::LayerNorm: return "LayerNorm";
        case OpType::Reshape: return "Reshape";
        case OpType::Transpose: return "Transpose";
        case OpType::Permute: return "Permute";
        case OpType::Squeeze: return "Squeeze";
        case OpType::Unsqueeze: return "Unsqueeze";
        case OpType::Flatten: return "Flatten";
        case OpType::Sum: return "Sum";
        case OpType::Mean: return "Mean";
        case OpType::Max: return "Max";
        case OpType::Min: return "Min";
        case OpType::Exp: return "Exp";
        case OpType::Log: return "Log";
        case OpType::Sqrt: return "Sqrt";
        case OpType::Pow: return "Pow";
        case OpType::Abs: return "Abs";
        case OpType::Neg: return "Neg";
        case OpType::Clamp: return "Clamp";
        case OpType::Slice: return "Slice";
        case OpType::Cat: return "Cat";
        case OpType::Dropout: return "Dropout";
        case OpType::Linear: return "Linear";
        case OpType::Embedding: return "Embedding";
        case OpType::GELU: return "GELU";
        case OpType::Det: return "Det";
        case OpType::Inv: return "Inv";
        case OpType::Solve: return "Solve";
        case OpType::Cholesky: return "Cholesky";
        case OpType::Svd: return "Svd";
        case OpType::Qr: return "Qr";
        case OpType::Eigh: return "Eigh";
        case OpType::Eigvalsh: return "Eigvalsh";
        case OpType::Norm: return "Norm";
        case OpType::Slogdet: return "Slogdet";
        case OpType::Constant: return "Constant";
        case OpType::Input: return "Input";
        case OpType::Output: return "Output";
        default: return "Unknown";
    }
}

auto string_to_op_type(const std::string& str) -> OpType {
    static const std::unordered_map<std::string, OpType> string_to_type = {
        {"Add", OpType::Add},
        {"Sub", OpType::Sub},
        {"Mul", OpType::Mul},
        {"Div", OpType::Div},
        {"MatMul", OpType::MatMul},
        {"Bmm", OpType::Bmm},
        {"ReLU", OpType::ReLU},
        {"Sigmoid", OpType::Sigmoid},
        {"Tanh", OpType::Tanh},
        {"Softmax", OpType::Softmax},
        {"LogSoftmax", OpType::LogSoftmax},
        {"MaxPool2d", OpType::MaxPool2d},
        {"AvgPool2d", OpType::AvgPool2d},
        {"AdaptiveAvgPool2d", OpType::AdaptiveAvgPool2d},
        {"Conv2d", OpType::Conv2d},
        {"BatchNorm2d", OpType::BatchNorm2d},
        {"LayerNorm", OpType::LayerNorm},
        {"Reshape", OpType::Reshape},
        {"Transpose", OpType::Transpose},
        {"Permute", OpType::Permute},
        {"Squeeze", OpType::Squeeze},
        {"Unsqueeze", OpType::Unsqueeze},
        {"Flatten", OpType::Flatten},
        {"Sum", OpType::Sum},
        {"Mean", OpType::Mean},
        {"Max", OpType::Max},
        {"Min", OpType::Min},
        {"Exp", OpType::Exp},
        {"Log", OpType::Log},
        {"Sqrt", OpType::Sqrt},
        {"Pow", OpType::Pow},
        {"Abs", OpType::Abs},
        {"Neg", OpType::Neg},
        {"Clamp", OpType::Clamp},
        {"Slice", OpType::Slice},
        {"Cat", OpType::Cat},
        {"Dropout", OpType::Dropout},
        {"Linear", OpType::Linear},
        {"Embedding", OpType::Embedding},
        {"GELU", OpType::GELU},
        {"Det", OpType::Det},
        {"Inv", OpType::Inv},
        {"Solve", OpType::Solve},
        {"Cholesky", OpType::Cholesky},
        {"Svd", OpType::Svd},
        {"Qr", OpType::Qr},
        {"Eigh", OpType::Eigh},
        {"Eigvalsh", OpType::Eigvalsh},
        {"Norm", OpType::Norm},
        {"Slogdet", OpType::Slogdet},
        {"Constant", OpType::Constant},
        {"Input", OpType::Input},
        {"Output", OpType::Output}
    };

    auto it = string_to_type.find(str);
    if (it == string_to_type.end()) {
        throw std::runtime_error("Unknown operation type: " + str);
    }
    return it->second;
}

// ============================================================================
// Tracer implementation
// ============================================================================

auto Tracer::start_trace() -> void {
    clear();
    tracing_ = true;
}

auto Tracer::end_trace(const std::vector<Variable>& inputs,
                       const std::vector<Variable>& outputs) -> std::shared_ptr<Graph> {
    tracing_ = false;

    // Create graph
    auto graph = std::make_shared<Graph>();

    // Map tensor IDs to graph values
    std::unordered_map<std::string, std::shared_ptr<Value>> value_map;

    // Create input values
    std::vector<std::shared_ptr<Value>> graph_inputs;
    for (const auto& input : inputs) {
        auto tensor_id = register_tensor(input);
        const auto& info = tensor_info_[tensor_id];
        auto value = graph->create_value(tensor_id, info.shape, info.dtype, info.device);
        value_map[tensor_id] = value;
        graph_inputs.push_back(value);
    }
    graph->set_inputs(graph_inputs);

    // Process recorded operations
    for (const auto& op : ops_) {
        // Create node
        auto node = graph->create_node(op.type);

        // Add inputs
        for (const auto& input_id : op.inputs) {
            auto it = value_map.find(input_id);
            if (it != value_map.end()) {
                node->add_input(it->second);
            } else {
                // Create value for unknown input (likely a constant or parameter)
                const auto& info = tensor_info_[input_id];
                auto value = graph->create_value(input_id, info.shape, info.dtype, info.device);
                value_map[input_id] = value;
                node->add_input(value);
            }
        }

        // Add outputs
        for (const auto& output_id : op.outputs) {
            const auto& info = tensor_info_[output_id];
            auto value = graph->create_value(output_id, info.shape, info.dtype, info.device);
            value->set_node(node);
            node->add_output(value);
            value_map[output_id] = value;
        }

        // Copy attributes
        for (const auto& [name, val] : op.attrs) {
            node->set_attr(name, val);
        }
        for (const auto& [name, val] : op.int_attrs) {
            node->set_int_attr(name, val);
        }
        for (const auto& [name, val] : op.vec_attrs) {
            node->set_vec_attr(name, val);
        }
        for (const auto& [name, val] : op.bool_attrs) {
            node->set_bool_attr(name, val);
        }
        for (const auto& [name, val] : op.tensor_attrs) {
            node->set_tensor_attr(name, val);
        }

        graph->add_node(node);
    }

    // Set outputs
    std::vector<std::shared_ptr<Value>> graph_outputs;
    for (const auto& output : outputs) {
        auto tensor_id = register_tensor(output);
        auto it = value_map.find(tensor_id);
        if (it != value_map.end()) {
            graph_outputs.push_back(it->second);
        }
    }
    graph->set_outputs(graph_outputs);

    // Topological sort
    graph->topological_sort();

    // Type inference
    graph->infer_types();

    return graph;
}

auto Tracer::record_op(TracedOp op) -> void {
    if (tracing_) {
        ops_.push_back(std::move(op));
    }
}

auto Tracer::register_tensor(const Tensor& tensor) -> std::string {
    void* ptr = const_cast<void*>(tensor.data_ptr());

    auto it = tensor_id_map_.find(ptr);
    if (it != tensor_id_map_.end()) {
        return it->second;
    }

    auto id = generate_tensor_id();
    tensor_id_map_[ptr] = id;

    std::vector<int64_t> shape;
    for (auto s : tensor.shape()) {
        shape.push_back(s);
    }

    tensor_info_[id] = TensorInfo(shape, tensor.dtype(), tensor.device());
    return id;
}

auto Tracer::register_tensor(const Variable& var) -> std::string {
    return register_tensor(var.tensor());
}

auto Tracer::get_tensor_info(const std::string& tensor_id) const -> const TensorInfo& {
    auto it = tensor_info_.find(tensor_id);
    if (it == tensor_info_.end()) {
        throw std::runtime_error("Tensor ID not found: " + tensor_id);
    }
    return it->second;
}

auto Tracer::clear() -> void {
    ops_.clear();
    tensor_info_.clear();
    tensor_id_map_.clear();
    next_tensor_id_ = 0;
}

auto Tracer::get_instance() -> Tracer& {
    thread_local Tracer instance;
    return instance;
}

auto Tracer::generate_tensor_id() -> std::string {
    std::ostringstream oss;
    oss << "t" << next_tensor_id_++;
    return oss.str();
}

// ============================================================================
// TracingGuard implementation
// ============================================================================

TracingGuard::TracingGuard() : tracer_(Tracer::get_instance()) {
    tracer_.start_trace();
}

TracingGuard::~TracingGuard() {
    if (tracer_.is_tracing()) {
        tracer_.clear();
    }
}

auto TracingGuard::get_graph(const std::vector<Variable>& inputs,
                              const std::vector<Variable>& outputs) -> std::shared_ptr<Graph> {
    return tracer_.end_trace(inputs, outputs);
}

// ============================================================================
// Trace functions
// ============================================================================

auto trace(std::shared_ptr<nn::Module> module,
           const Variable& dummy_input) -> std::shared_ptr<Graph> {
    if (!module) {
        throw std::runtime_error("Cannot trace null module");
    }

    // Set module to eval mode
    module->eval();

    // Start tracing
    TracingGuard guard;

    // Run forward pass
    Variable output = module->forward(dummy_input);

    // End tracing and get graph
    return guard.get_graph({dummy_input}, {output});
}

auto trace(std::function<std::vector<Variable>(const std::vector<Variable>&)> func,
           const std::vector<Variable>& inputs) -> std::shared_ptr<Graph> {
    if (!func) {
        throw std::runtime_error("Cannot trace null function");
    }

    // Start tracing
    TracingGuard guard;

    // Run function
    auto outputs = func(inputs);

    // End tracing and get graph
    return guard.get_graph(inputs, outputs);
}

auto trace(std::shared_ptr<nn::Module> module,
           const Tensor& dummy_input) -> std::shared_ptr<CompiledModule> {
    if (!module) {
        throw std::runtime_error("Cannot trace null module");
    }

    Variable input_var(dummy_input, false);
    auto graph = trace(module, input_var);
    return std::make_shared<CompiledModule>(graph);
}

} // namespace jit
} // namespace tenzor
