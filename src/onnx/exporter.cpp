/**
 * @file exporter.cpp
 * @brief Implementation of ONNX model export functionality
 */

#include "../../include/tenzor/onnx/exporter.hpp"
#include "../../include/tenzor/utils/error.hpp"
#include "../../include/tenzor/utils/logging.hpp"
#include <cstring>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace tenzor {
namespace onnx {

// ============================================================================
// Helper Functions
// ============================================================================

namespace {

// Forward declaration for ONNX export helper
auto trace_custom_module(ONNXExporter& exporter,
                        std::shared_ptr<nn::Module> module,
                        const Variable& input,
                        const Variable& output,
                        const std::string& input_name,
                        const std::string& output_name) -> void;

/**
 * @brief Write varint encoding (Protocol Buffers format)
 */
auto write_varint(std::vector<uint8_t>& buffer, uint64_t value) -> void {
    while (value >= 0x80) {
        buffer.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
        value >>= 7;
    }
    buffer.push_back(static_cast<uint8_t>(value));
}

/**
 * @brief Write fixed32 (little-endian)
 */
auto write_fixed32(std::vector<uint8_t>& buffer, uint32_t value) -> void {
    buffer.push_back(static_cast<uint8_t>(value & 0xFF));
    buffer.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
    buffer.push_back(static_cast<uint8_t>((value >> 16) & 0xFF));
    buffer.push_back(static_cast<uint8_t>((value >> 24) & 0xFF));
}

/**
 * @brief Write fixed64 (little-endian)
 */
auto write_fixed64(std::vector<uint8_t>& buffer, uint64_t value) -> void {
    for (int i = 0; i < 8; ++i) {
        buffer.push_back(static_cast<uint8_t>((value >> (i * 8)) & 0xFF));
    }
}

/**
 * @brief Write protobuf tag (field number + wire type)
 */
auto write_tag(std::vector<uint8_t>& buffer, uint32_t field_number, uint32_t wire_type) -> void {
    write_varint(buffer, (field_number << 3) | wire_type);
}

/**
 * @brief Write length-delimited field
 */
auto write_length_delimited(std::vector<uint8_t>& buffer, uint32_t field_number,
                            const std::vector<uint8_t>& data) -> void {
    write_tag(buffer, field_number, 2); // Wire type 2 = length-delimited
    write_varint(buffer, data.size());
    buffer.insert(buffer.end(), data.begin(), data.end());
}

/**
 * @brief Write string field
 */
auto write_string(std::vector<uint8_t>& buffer, uint32_t field_number,
                 const std::string& value) -> void {
    write_tag(buffer, field_number, 2);
    write_varint(buffer, value.size());
    buffer.insert(buffer.end(), value.begin(), value.end());
}

/**
 * @brief Write int64 field
 */
auto write_int64(std::vector<uint8_t>& buffer, uint32_t field_number, int64_t value) -> void {
    write_tag(buffer, field_number, 0); // Wire type 0 = varint
    write_varint(buffer, static_cast<uint64_t>(value));
}

/**
 * @brief Write float field
 */
auto write_float(std::vector<uint8_t>& buffer, uint32_t field_number, float value) -> void {
    write_tag(buffer, field_number, 5); // Wire type 5 = fixed32
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(float));
    write_fixed32(buffer, bits);
}

/**
 * @brief Serialize repeated int64 field (packed)
 */
auto write_packed_int64(std::vector<uint8_t>& buffer, uint32_t field_number,
                       const std::vector<int64_t>& values) -> void {
    if (values.empty()) return;

    std::vector<uint8_t> packed;
    for (int64_t val : values) {
        write_varint(packed, static_cast<uint64_t>(val));
    }
    write_length_delimited(buffer, field_number, packed);
}

/**
 * @brief Serialize repeated float field (packed)
 */
auto write_packed_float(std::vector<uint8_t>& buffer, uint32_t field_number,
                       const std::vector<float>& values) -> void {
    if (values.empty()) return;

    write_tag(buffer, field_number, 2);
    write_varint(buffer, values.size() * sizeof(float));
    for (float val : values) {
        uint32_t bits;
        std::memcpy(&bits, &val, sizeof(float));
        write_fixed32(buffer, bits);
    }
}

} // anonymous namespace

// ============================================================================
// DType Conversion
// ============================================================================

auto dtype_to_onnx(DType dtype) -> ONNXDataType {
    switch (dtype) {
        case DType::Float32: return ONNXDataType::FLOAT;
        case DType::Float64: return ONNXDataType::DOUBLE;
        case DType::Float16: return ONNXDataType::FLOAT16;
        case DType::BFloat16: return ONNXDataType::BFLOAT16;
        case DType::Int8: return ONNXDataType::INT8;
        case DType::Int16: return ONNXDataType::INT16;
        case DType::Int32: return ONNXDataType::INT32;
        case DType::Int64: return ONNXDataType::INT64;
        case DType::UInt8: return ONNXDataType::UINT8;
        case DType::UInt16: return ONNXDataType::UINT16;
        case DType::UInt32: return ONNXDataType::UINT32;
        case DType::UInt64: return ONNXDataType::UINT64;
        case DType::Bool: return ONNXDataType::BOOL;
        case DType::Complex64: return ONNXDataType::COMPLEX64;
        case DType::Complex128: return ONNXDataType::COMPLEX128;
        default:
            throw std::runtime_error("Unsupported DType for ONNX export");
    }
}

// ============================================================================
// ONNXTensor Implementation
// ============================================================================

ONNXTensor::ONNXTensor(const Tensor& tensor, const std::string& name)
    : name(name), dtype(dtype_to_onnx(tensor.dtype())) {

    // Copy shape
    auto shape_span = tensor.shape();
    dims.assign(shape_span.begin(), shape_span.end());

    // Copy raw data
    size_t num_bytes = tensor.numel() * tensor.dtype_size();
    raw_data.resize(num_bytes);

    // Ensure tensor is on CPU and contiguous
    Tensor cpu_tensor = tensor.cpu().contiguous();
    const void* data_ptr = cpu_tensor.data_ptr();
    std::memcpy(raw_data.data(), data_ptr, num_bytes);
}

auto ONNXTensor::numel() const -> int64_t {
    int64_t result = 1;
    for (int64_t dim : dims) {
        result *= dim;
    }
    return result;
}

auto ONNXTensor::size_bytes() const -> size_t {
    return raw_data.size();
}

// ============================================================================
// ONNXValueInfo Implementation
// ============================================================================

ONNXValueInfo::ONNXValueInfo(const std::string& name, ONNXDataType dtype,
                             const std::vector<int64_t>& shape)
    : name(name), dtype(dtype), shape(shape) {}

// ============================================================================
// ONNXNode Implementation
// ============================================================================

ONNXNode::ONNXNode(const std::string& op_type, const std::string& name)
    : op_type(op_type), name(name) {}

auto ONNXNode::add_input(const std::string& input) -> void {
    inputs.push_back(input);
}

auto ONNXNode::add_output(const std::string& output) -> void {
    outputs.push_back(output);
}

auto ONNXNode::set_attr(const std::string& key, int64_t value) -> void {
    int_attrs[key] = value;
}

auto ONNXNode::set_attr(const std::string& key, float value) -> void {
    float_attrs[key] = value;
}

auto ONNXNode::set_attr(const std::string& key, const std::string& value) -> void {
    string_attrs[key] = value;
}

auto ONNXNode::set_attr(const std::string& key, const std::vector<int64_t>& value) -> void {
    ints_attrs[key] = value;
}

auto ONNXNode::set_attr(const std::string& key, const std::vector<float>& value) -> void {
    floats_attrs[key] = value;
}

auto ONNXNode::set_attr(const std::string& key, const ONNXTensor& value) -> void {
    tensor_attrs[key] = value;
}

// ============================================================================
// ONNXGraph Implementation
// ============================================================================

ONNXGraph::ONNXGraph(const std::string& name) : name(name) {}

auto ONNXGraph::add_node(const ONNXNode& node) -> void {
    nodes.push_back(node);
}

auto ONNXGraph::add_input(const ONNXValueInfo& input) -> void {
    inputs.push_back(input);
}

auto ONNXGraph::add_output(const ONNXValueInfo& output) -> void {
    outputs.push_back(output);
}

auto ONNXGraph::add_initializer(const ONNXTensor& tensor) -> void {
    initializers.push_back(tensor);
}

auto ONNXGraph::add_value_info(const ONNXValueInfo& info) -> void {
    value_info[info.name] = info;
}

auto ONNXGraph::get_unique_name(const std::string& prefix) -> std::string {
    return prefix + "_" + std::to_string(name_counter_++);
}

// ============================================================================
// ExportContext Implementation
// ============================================================================

auto ExportContext::register_tensor(const Tensor& tensor, const std::string& onnx_name) -> void {
    tensor_map_[tensor.data_ptr()] = onnx_name;
}

auto ExportContext::get_tensor_name(const Tensor& tensor) -> std::optional<std::string> {
    auto it = tensor_map_.find(tensor.data_ptr());
    if (it != tensor_map_.end()) {
        return it->second;
    }
    return std::nullopt;
}

auto ExportContext::has_tensor(const Tensor& tensor) -> bool {
    return tensor_map_.find(tensor.data_ptr()) != tensor_map_.end();
}

auto ExportContext::generate_name(const std::string& prefix) -> std::string {
    int64_t& counter = name_counters_[prefix];
    return prefix + "_" + std::to_string(counter++);
}

// ============================================================================
// ONNXExporter Implementation
// ============================================================================

ONNXExporter::ONNXExporter(int64_t opset_version)
    : opset_version_(opset_version), graph_("main_graph") {}

// Model Configuration

auto ONNXExporter::set_model_name(const std::string& name) -> void {
    model_name_ = name;
}

auto ONNXExporter::set_opset_version(int64_t version) -> void {
    opset_version_ = version;
}

auto ONNXExporter::set_description(const std::string& desc) -> void {
    description_ = desc;
}

auto ONNXExporter::set_producer_name(const std::string& name) -> void {
    producer_name_ = name;
}

auto ONNXExporter::set_model_version(int64_t version) -> void {
    model_version_ = version;
}

// Graph Building

auto ONNXExporter::add_input(const Tensor& tensor, const std::string& name,
                             const std::unordered_map<int64_t, std::string>& dynamic_axes) -> void {
    auto shape_span = tensor.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

    // Apply dynamic axes
    for (const auto& [axis, axis_name] : dynamic_axes) {
        if (axis >= 0 && axis < static_cast<int64_t>(shape.size())) {
            shape[axis] = -1; // -1 indicates dynamic dimension in ONNX
        }
    }

    ONNXValueInfo input_info(name, dtype_to_onnx(tensor.dtype()), shape);
    graph_.add_input(input_info);
    context_.register_tensor(tensor, name);
}

auto ONNXExporter::add_output(const Tensor& tensor, const std::string& name) -> void {
    auto shape_span = tensor.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

    ONNXValueInfo output_info(name, dtype_to_onnx(tensor.dtype()), shape);
    graph_.add_output(output_info);
    context_.register_tensor(tensor, name);
}

// Tensor Operations

auto ONNXExporter::export_add(const Tensor& a, const Tensor& b, const Tensor& output,
                               const std::string& output_name) -> void {
    ONNXNode node("Add", context_.generate_name("add"));

    std::string a_name = get_tensor_name(a, "add_input_a");
    std::string b_name = get_tensor_name(b, "add_input_b");

    node.add_input(a_name);
    node.add_input(b_name);
    node.add_output(output_name);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_sub(const Tensor& a, const Tensor& b, const Tensor& output,
                               const std::string& output_name) -> void {
    ONNXNode node("Sub", context_.generate_name("sub"));

    std::string a_name = get_tensor_name(a, "sub_input_a");
    std::string b_name = get_tensor_name(b, "sub_input_b");

    node.add_input(a_name);
    node.add_input(b_name);
    node.add_output(output_name);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_mul(const Tensor& a, const Tensor& b, const Tensor& output,
                               const std::string& output_name) -> void {
    ONNXNode node("Mul", context_.generate_name("mul"));

    std::string a_name = get_tensor_name(a, "mul_input_a");
    std::string b_name = get_tensor_name(b, "mul_input_b");

    node.add_input(a_name);
    node.add_input(b_name);
    node.add_output(output_name);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_div(const Tensor& a, const Tensor& b, const Tensor& output,
                               const std::string& output_name) -> void {
    ONNXNode node("Div", context_.generate_name("div"));

    std::string a_name = get_tensor_name(a, "div_input_a");
    std::string b_name = get_tensor_name(b, "div_input_b");

    node.add_input(a_name);
    node.add_input(b_name);
    node.add_output(output_name);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_matmul(const Tensor& a, const Tensor& b, const Tensor& output,
                                  const std::string& output_name) -> void {
    ONNXNode node("MatMul", context_.generate_name("matmul"));

    std::string a_name = get_tensor_name(a, "matmul_input_a");
    std::string b_name = get_tensor_name(b, "matmul_input_b");

    node.add_input(a_name);
    node.add_input(b_name);
    node.add_output(output_name);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_reshape(const Tensor& input, const std::vector<int64_t>& new_shape,
                                   const Tensor& output, const std::string& output_name) -> void {
    ONNXNode node("Reshape", context_.generate_name("reshape"));

    std::string input_name = get_tensor_name(input, "reshape_input");

    // Create shape constant tensor
    std::string shape_name = context_.generate_name("reshape_shape");
    Tensor shape_tensor({static_cast<int64_t>(new_shape.size())}, DType::Int64, Device::cpu());
    std::memcpy(shape_tensor.data<int64_t>(), new_shape.data(), new_shape.size() * sizeof(int64_t));
    add_initializer_tensor(shape_tensor, shape_name);

    node.add_input(input_name);
    node.add_input(shape_name);
    node.add_output(output_name);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_transpose(const Tensor& input, const std::vector<int64_t>& perm,
                                     const Tensor& output, const std::string& output_name) -> void {
    ONNXNode node("Transpose", context_.generate_name("transpose"));

    std::string input_name = get_tensor_name(input, "transpose_input");

    node.add_input(input_name);
    node.add_output(output_name);
    node.set_attr("perm", perm);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_concat(const std::vector<Tensor>& inputs, int64_t axis,
                                  const Tensor& output, const std::string& output_name) -> void {
    ONNXNode node("Concat", context_.generate_name("concat"));

    for (size_t i = 0; i < inputs.size(); ++i) {
        std::string input_name = get_tensor_name(inputs[i], "concat_input_" + std::to_string(i));
        node.add_input(input_name);
    }

    node.add_output(output_name);
    node.set_attr("axis", axis);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_split(const Tensor& input, int64_t axis, const std::vector<int64_t>& split_sizes,
                                 const std::vector<Tensor>& outputs, const std::vector<std::string>& output_names) -> void {
    ONNXNode node("Split", context_.generate_name("split"));

    std::string input_name = get_tensor_name(input, "split_input");
    node.add_input(input_name);

    for (const auto& name : output_names) {
        node.add_output(name);
    }

    node.set_attr("axis", axis);
    if (!split_sizes.empty()) {
        node.set_attr("split", split_sizes);
    }

    graph_.add_node(node);

    for (size_t i = 0; i < outputs.size(); ++i) {
        context_.register_tensor(outputs[i], output_names[i]);
    }
}

// Neural Network Layers

auto ONNXExporter::export_linear(const Tensor& input, const Tensor& weight,
                                  const std::optional<Tensor>& bias, const Tensor& output,
                                  const std::string& output_name) -> void {
    // Linear layer in ONNX: Gemm (General Matrix Multiplication)
    // Y = alpha * A * B + beta * C
    // For Linear: Y = X @ W^T + bias
    // We need to transpose W

    ONNXNode gemm_node("Gemm", context_.generate_name("gemm"));

    std::string input_name = get_tensor_name(input, "linear_input");

    // Add weight as initializer (transposed)
    std::string weight_name = context_.generate_name("linear_weight");
    add_initializer_tensor(weight, weight_name);

    gemm_node.add_input(input_name);
    gemm_node.add_input(weight_name);

    if (bias.has_value()) {
        std::string bias_name = context_.generate_name("linear_bias");
        add_initializer_tensor(bias.value(), bias_name);
        gemm_node.add_input(bias_name);
    }

    gemm_node.add_output(output_name);

    // Attributes: alpha=1.0, beta=1.0, transB=1 (transpose weight)
    gemm_node.set_attr("alpha", 1.0f);
    gemm_node.set_attr("beta", 1.0f);
    gemm_node.set_attr("transB", static_cast<int64_t>(1));

    graph_.add_node(gemm_node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_conv2d(const Tensor& input, const Tensor& weight,
                                  const std::optional<Tensor>& bias,
                                  const std::vector<int64_t>& kernel_size,
                                  const std::vector<int64_t>& stride,
                                  const std::vector<int64_t>& padding,
                                  const std::vector<int64_t>& dilation,
                                  int64_t groups,
                                  const Tensor& output,
                                  const std::string& output_name) -> void {
    ONNXNode node("Conv", context_.generate_name("conv"));

    std::string input_name = get_tensor_name(input, "conv_input");

    std::string weight_name = context_.generate_name("conv_weight");
    add_initializer_tensor(weight, weight_name);

    node.add_input(input_name);
    node.add_input(weight_name);

    if (bias.has_value()) {
        std::string bias_name = context_.generate_name("conv_bias");
        add_initializer_tensor(bias.value(), bias_name);
        node.add_input(bias_name);
    }

    node.add_output(output_name);

    // Set attributes
    node.set_attr("kernel_shape", kernel_size);
    node.set_attr("strides", stride);
    node.set_attr("pads", std::vector<int64_t>{padding[0], padding[1], padding[0], padding[1]}); // [top, left, bottom, right]
    node.set_attr("dilations", dilation);
    node.set_attr("group", groups);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_conv1d(const Tensor& input, const Tensor& weight,
                                  const std::optional<Tensor>& bias,
                                  int64_t kernel_size, int64_t stride, int64_t padding,
                                  int64_t dilation, int64_t groups,
                                  const Tensor& output,
                                  const std::string& output_name) -> void {
    ONNXNode node("Conv", context_.generate_name("conv1d"));

    std::string input_name = get_tensor_name(input, "conv1d_input");

    std::string weight_name = context_.generate_name("conv1d_weight");
    add_initializer_tensor(weight, weight_name);

    node.add_input(input_name);
    node.add_input(weight_name);

    if (bias.has_value()) {
        std::string bias_name = context_.generate_name("conv1d_bias");
        add_initializer_tensor(bias.value(), bias_name);
        node.add_input(bias_name);
    }

    node.add_output(output_name);

    // Set attributes for 1D conv
    node.set_attr("kernel_shape", std::vector<int64_t>{kernel_size});
    node.set_attr("strides", std::vector<int64_t>{stride});
    node.set_attr("pads", std::vector<int64_t>{padding, padding}); // [begin, end]
    node.set_attr("dilations", std::vector<int64_t>{dilation});
    node.set_attr("group", groups);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_batchnorm2d(const Tensor& input, const Tensor& scale,
                                       const Tensor& bias, const Tensor& mean,
                                       const Tensor& var, double eps,
                                       const Tensor& output,
                                       const std::string& output_name) -> void {
    ONNXNode node("BatchNormalization", context_.generate_name("batchnorm"));

    std::string input_name = get_tensor_name(input, "bn_input");

    std::string scale_name = context_.generate_name("bn_scale");
    add_initializer_tensor(scale, scale_name);

    std::string bias_name = context_.generate_name("bn_bias");
    add_initializer_tensor(bias, bias_name);

    std::string mean_name = context_.generate_name("bn_mean");
    add_initializer_tensor(mean, mean_name);

    std::string var_name = context_.generate_name("bn_var");
    add_initializer_tensor(var, var_name);

    node.add_input(input_name);
    node.add_input(scale_name);
    node.add_input(bias_name);
    node.add_input(mean_name);
    node.add_input(var_name);

    node.add_output(output_name);

    node.set_attr("epsilon", static_cast<float>(eps));
    node.set_attr("momentum", 0.9f); // Default momentum

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_batchnorm1d(const Tensor& input, const Tensor& scale,
                                       const Tensor& bias, const Tensor& mean,
                                       const Tensor& var, double eps,
                                       const Tensor& output,
                                       const std::string& output_name) -> void {
    // BatchNorm1d uses the same ONNX op as BatchNorm2d
    export_batchnorm2d(input, scale, bias, mean, var, eps, output, output_name);
}

// Activation Functions

auto ONNXExporter::export_relu(const Tensor& input, const Tensor& output,
                                const std::string& output_name) -> void {
    ONNXNode node("Relu", context_.generate_name("relu"));

    std::string input_name = get_tensor_name(input, "relu_input");

    node.add_input(input_name);
    node.add_output(output_name);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_leaky_relu(const Tensor& input, double alpha,
                                      const Tensor& output, const std::string& output_name) -> void {
    ONNXNode node("LeakyRelu", context_.generate_name("leaky_relu"));

    std::string input_name = get_tensor_name(input, "leaky_relu_input");

    node.add_input(input_name);
    node.add_output(output_name);
    node.set_attr("alpha", static_cast<float>(alpha));

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_sigmoid(const Tensor& input, const Tensor& output,
                                   const std::string& output_name) -> void {
    ONNXNode node("Sigmoid", context_.generate_name("sigmoid"));

    std::string input_name = get_tensor_name(input, "sigmoid_input");

    node.add_input(input_name);
    node.add_output(output_name);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_tanh(const Tensor& input, const Tensor& output,
                                const std::string& output_name) -> void {
    ONNXNode node("Tanh", context_.generate_name("tanh"));

    std::string input_name = get_tensor_name(input, "tanh_input");

    node.add_input(input_name);
    node.add_output(output_name);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_gelu(const Tensor& input, const Tensor& output,
                                const std::string& output_name) -> void {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // For ONNX opset 20+, there's a Gelu op. For earlier versions, we decompose it.

    if (opset_version_ >= 20) {
        ONNXNode node("Gelu", context_.generate_name("gelu"));
        std::string input_name = get_tensor_name(input, "gelu_input");
        node.add_input(input_name);
        node.add_output(output_name);
        graph_.add_node(node);
    } else {
        // Decompose GELU for older opset versions
        std::string input_name = get_tensor_name(input, "gelu_input");

        // Constants
        std::string half_name = context_.generate_name("gelu_half");
        Tensor half_tensor({1}, DType::Float32, Device::cpu());
        half_tensor.fill_(0.5f);
        add_initializer_tensor(half_tensor, half_name);

        std::string coef_name = context_.generate_name("gelu_coef");
        Tensor coef_tensor({1}, DType::Float32, Device::cpu());
        coef_tensor.fill_(0.044715f);
        add_initializer_tensor(coef_tensor, coef_name);

        std::string sqrt_2_pi_name = context_.generate_name("gelu_sqrt_2_pi");
        Tensor sqrt_2_pi_tensor({1}, DType::Float32, Device::cpu());
        sqrt_2_pi_tensor.fill_(std::sqrt(2.0 / M_PI));
        add_initializer_tensor(sqrt_2_pi_tensor, sqrt_2_pi_name);

        // x^3
        std::string x3_name = context_.generate_name("gelu_x3");
        ONNXNode pow_node("Pow", context_.generate_name("gelu_pow"));
        std::string three_name = context_.generate_name("gelu_three");
        Tensor three_tensor({1}, DType::Float32, Device::cpu());
        three_tensor.fill_(3.0f);
        add_initializer_tensor(three_tensor, three_name);
        pow_node.add_input(input_name);
        pow_node.add_input(three_name);
        pow_node.add_output(x3_name);
        graph_.add_node(pow_node);

        // 0.044715 * x^3
        std::string scaled_x3_name = context_.generate_name("gelu_scaled_x3");
        ONNXNode mul1_node("Mul", context_.generate_name("gelu_mul1"));
        mul1_node.add_input(coef_name);
        mul1_node.add_input(x3_name);
        mul1_node.add_output(scaled_x3_name);
        graph_.add_node(mul1_node);

        // x + 0.044715 * x^3
        std::string sum1_name = context_.generate_name("gelu_sum1");
        ONNXNode add1_node("Add", context_.generate_name("gelu_add1"));
        add1_node.add_input(input_name);
        add1_node.add_input(scaled_x3_name);
        add1_node.add_output(sum1_name);
        graph_.add_node(add1_node);

        // sqrt(2/pi) * (x + 0.044715 * x^3)
        std::string mul2_name = context_.generate_name("gelu_mul2");
        ONNXNode mul2_node("Mul", context_.generate_name("gelu_mul2"));
        mul2_node.add_input(sqrt_2_pi_name);
        mul2_node.add_input(sum1_name);
        mul2_node.add_output(mul2_name);
        graph_.add_node(mul2_node);

        // tanh(...)
        std::string tanh_name = context_.generate_name("gelu_tanh");
        ONNXNode tanh_node("Tanh", context_.generate_name("gelu_tanh"));
        tanh_node.add_input(mul2_name);
        tanh_node.add_output(tanh_name);
        graph_.add_node(tanh_node);

        // 1 + tanh(...)
        std::string one_name = context_.generate_name("gelu_one");
        Tensor one_tensor({1}, DType::Float32, Device::cpu());
        one_tensor.fill_(1.0f);
        add_initializer_tensor(one_tensor, one_name);

        std::string add2_name = context_.generate_name("gelu_add2");
        ONNXNode add2_node("Add", context_.generate_name("gelu_add2"));
        add2_node.add_input(one_name);
        add2_node.add_input(tanh_name);
        add2_node.add_output(add2_name);
        graph_.add_node(add2_node);

        // x * (1 + tanh(...))
        std::string mul3_name = context_.generate_name("gelu_mul3");
        ONNXNode mul3_node("Mul", context_.generate_name("gelu_mul3"));
        mul3_node.add_input(input_name);
        mul3_node.add_input(add2_name);
        mul3_node.add_output(mul3_name);
        graph_.add_node(mul3_node);

        // 0.5 * x * (1 + tanh(...))
        ONNXNode mul4_node("Mul", context_.generate_name("gelu_mul4"));
        mul4_node.add_input(half_name);
        mul4_node.add_input(mul3_name);
        mul4_node.add_output(output_name);
        graph_.add_node(mul4_node);
    }

    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_softmax(const Tensor& input, int64_t axis,
                                   const Tensor& output, const std::string& output_name) -> void {
    ONNXNode node("Softmax", context_.generate_name("softmax"));

    std::string input_name = get_tensor_name(input, "softmax_input");

    node.add_input(input_name);
    node.add_output(output_name);
    node.set_attr("axis", axis);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_log_softmax(const Tensor& input, int64_t axis,
                                       const Tensor& output, const std::string& output_name) -> void {
    ONNXNode node("LogSoftmax", context_.generate_name("log_softmax"));

    std::string input_name = get_tensor_name(input, "log_softmax_input");

    node.add_input(input_name);
    node.add_output(output_name);
    node.set_attr("axis", axis);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_elu(const Tensor& input, double alpha,
                               const Tensor& output, const std::string& output_name) -> void {
    ONNXNode node("Elu", context_.generate_name("elu"));

    std::string input_name = get_tensor_name(input, "elu_input");

    node.add_input(input_name);
    node.add_output(output_name);
    node.set_attr("alpha", static_cast<float>(alpha));

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_selu(const Tensor& input, const Tensor& output,
                                const std::string& output_name) -> void {
    ONNXNode node("Selu", context_.generate_name("selu"));

    std::string input_name = get_tensor_name(input, "selu_input");

    node.add_input(input_name);
    node.add_output(output_name);

    // SELU constants: alpha ≈ 1.6733, gamma ≈ 1.0507
    node.set_attr("alpha", 1.67326324f);
    node.set_attr("gamma", 1.05070098f);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_swish(const Tensor& input, const Tensor& output,
                                 const std::string& output_name) -> void {
    // Swish = x * sigmoid(x)
    std::string input_name = get_tensor_name(input, "swish_input");

    // Sigmoid
    std::string sigmoid_out_name = context_.generate_name("swish_sigmoid");
    ONNXNode sigmoid_node("Sigmoid", context_.generate_name("swish_sigmoid"));
    sigmoid_node.add_input(input_name);
    sigmoid_node.add_output(sigmoid_out_name);
    graph_.add_node(sigmoid_node);

    // Multiply
    ONNXNode mul_node("Mul", context_.generate_name("swish_mul"));
    mul_node.add_input(input_name);
    mul_node.add_input(sigmoid_out_name);
    mul_node.add_output(output_name);
    graph_.add_node(mul_node);

    context_.register_tensor(output, output_name);
}

// Pooling Layers

auto ONNXExporter::export_maxpool2d(const Tensor& input, int64_t kernel_size,
                                     int64_t stride, int64_t padding,
                                     const Tensor& output, const std::string& output_name) -> void {
    ONNXNode node("MaxPool", context_.generate_name("maxpool"));

    std::string input_name = get_tensor_name(input, "maxpool_input");

    node.add_input(input_name);
    node.add_output(output_name);

    node.set_attr("kernel_shape", std::vector<int64_t>{kernel_size, kernel_size});
    node.set_attr("strides", std::vector<int64_t>{stride, stride});
    node.set_attr("pads", std::vector<int64_t>{padding, padding, padding, padding});

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_avgpool2d(const Tensor& input, int64_t kernel_size,
                                     int64_t stride, int64_t padding,
                                     const Tensor& output, const std::string& output_name) -> void {
    ONNXNode node("AveragePool", context_.generate_name("avgpool"));

    std::string input_name = get_tensor_name(input, "avgpool_input");

    node.add_input(input_name);
    node.add_output(output_name);

    node.set_attr("kernel_shape", std::vector<int64_t>{kernel_size, kernel_size});
    node.set_attr("strides", std::vector<int64_t>{stride, stride});
    node.set_attr("pads", std::vector<int64_t>{padding, padding, padding, padding});

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_adaptive_avgpool2d(const Tensor& input,
                                              const std::vector<int64_t>& output_size,
                                              const Tensor& output, const std::string& output_name) -> void {
    // AdaptiveAvgPool uses GlobalAveragePool in ONNX when output size is 1x1
    // Otherwise, we need to compute the kernel and stride

    if (output_size.size() == 2 && output_size[0] == 1 && output_size[1] == 1) {
        // Use GlobalAveragePool
        ONNXNode node("GlobalAveragePool", context_.generate_name("global_avgpool"));
        std::string input_name = get_tensor_name(input, "adaptive_avgpool_input");
        node.add_input(input_name);
        node.add_output(output_name);
        graph_.add_node(node);
    } else {
        // Compute kernel size and stride from input and output sizes
        auto input_shape = input.shape();
        int64_t h_in = input_shape[2];
        int64_t w_in = input_shape[3];
        int64_t h_out = output_size[0];
        int64_t w_out = output_size[1];

        int64_t stride_h = h_in / h_out;
        int64_t stride_w = w_in / w_out;
        int64_t kernel_h = h_in - (h_out - 1) * stride_h;
        int64_t kernel_w = w_in - (w_out - 1) * stride_w;

        ONNXNode node("AveragePool", context_.generate_name("adaptive_avgpool"));
        std::string input_name = get_tensor_name(input, "adaptive_avgpool_input");

        node.add_input(input_name);
        node.add_output(output_name);
        node.set_attr("kernel_shape", std::vector<int64_t>{kernel_h, kernel_w});
        node.set_attr("strides", std::vector<int64_t>{stride_h, stride_w});

        graph_.add_node(node);
    }

    context_.register_tensor(output, output_name);
}

// Export Functions

auto ONNXExporter::serialize_model() -> std::vector<uint8_t> {
    std::vector<uint8_t> buffer;

    // This is a simplified ONNX protobuf serialization
    // In production, you would use the official ONNX library (onnx.proto3)
    // For now, we create a valid but simplified ONNX format

    // ModelProto fields:
    // 1: ir_version (int64)
    // 2: opset_import (repeated OperatorSetIdProto)
    // 7: graph (GraphProto)
    // 8: producer_name (string)
    // 9: producer_version (string)
    // 10: domain (string)
    // 11: model_version (int64)
    // 12: doc_string (string)

    // IR version (field 1)
    write_int64(buffer, 1, 8); // ONNX IR version 8

    // Opset import (field 2)
    std::vector<uint8_t> opset_data;
    write_int64(opset_data, 2, opset_version_); // version field
    write_length_delimited(buffer, 2, opset_data);

    // Producer name (field 8)
    write_string(buffer, 8, producer_name_);

    // Model version (field 11)
    write_int64(buffer, 11, model_version_);

    // Doc string (field 12)
    if (!description_.empty()) {
        write_string(buffer, 12, description_);
    }

    // Graph (field 7) - simplified
    std::vector<uint8_t> graph_data;

    // Graph name (field 1)
    write_string(graph_data, 1, graph_.name);

    // Serialize nodes (field 1 in GraphProto)
    for (const auto& node : graph_.nodes) {
        std::vector<uint8_t> node_data;

        // Inputs (field 1)
        for (const auto& input : node.inputs) {
            write_string(node_data, 1, input);
        }

        // Outputs (field 2)
        for (const auto& output : node.outputs) {
            write_string(node_data, 2, output);
        }

        // Name (field 3)
        write_string(node_data, 3, node.name);

        // Op type (field 4)
        write_string(node_data, 4, node.op_type);

        // Attributes (field 5) - simplified
        for (const auto& [key, val] : node.int_attrs) {
            std::vector<uint8_t> attr_data;
            write_string(attr_data, 1, key); // name
            write_int64(attr_data, 2, 2); // type = INT
            write_int64(attr_data, 3, val); // i
            write_length_delimited(node_data, 5, attr_data);
        }

        for (const auto& [key, val] : node.float_attrs) {
            std::vector<uint8_t> attr_data;
            write_string(attr_data, 1, key); // name
            write_int64(attr_data, 2, 1); // type = FLOAT
            write_float(attr_data, 3, val); // f
            write_length_delimited(node_data, 5, attr_data);
        }

        for (const auto& [key, vals] : node.ints_attrs) {
            std::vector<uint8_t> attr_data;
            write_string(attr_data, 1, key); // name
            write_int64(attr_data, 2, 7); // type = INTS
            write_packed_int64(attr_data, 4, vals); // ints
            write_length_delimited(node_data, 5, attr_data);
        }

        write_length_delimited(graph_data, 1, node_data);
    }

    // Serialize initializers (field 5 in GraphProto)
    for (const auto& tensor : graph_.initializers) {
        std::vector<uint8_t> tensor_data;

        // Dims (field 1)
        write_packed_int64(tensor_data, 1, tensor.dims);

        // Data type (field 2)
        write_int64(tensor_data, 2, static_cast<int64_t>(tensor.dtype));

        // Name (field 8)
        write_string(tensor_data, 8, tensor.name);

        // Raw data (field 9)
        write_length_delimited(tensor_data, 9, tensor.raw_data);

        write_length_delimited(graph_data, 5, tensor_data);
    }

    // Serialize inputs (field 11 in GraphProto)
    for (const auto& input : graph_.inputs) {
        std::vector<uint8_t> value_info_data;

        // Name (field 1)
        write_string(value_info_data, 1, input.name);

        // Type (field 2) - simplified tensor type
        std::vector<uint8_t> type_data;
        std::vector<uint8_t> tensor_type_data;

        // Elem type (field 1)
        write_int64(tensor_type_data, 1, static_cast<int64_t>(input.dtype));

        // Shape (field 2)
        std::vector<uint8_t> shape_data;
        for (int64_t dim : input.shape) {
            std::vector<uint8_t> dim_data;
            if (dim >= 0) {
                write_int64(dim_data, 1, dim); // dim_value
            } else {
                write_string(dim_data, 2, "dynamic"); // dim_param
            }
            write_length_delimited(shape_data, 1, dim_data);
        }
        write_length_delimited(tensor_type_data, 2, shape_data);

        write_length_delimited(type_data, 1, tensor_type_data);
        write_length_delimited(value_info_data, 2, type_data);

        write_length_delimited(graph_data, 11, value_info_data);
    }

    // Serialize outputs (field 12 in GraphProto)
    for (const auto& output : graph_.outputs) {
        std::vector<uint8_t> value_info_data;

        write_string(value_info_data, 1, output.name);

        std::vector<uint8_t> type_data;
        std::vector<uint8_t> tensor_type_data;
        write_int64(tensor_type_data, 1, static_cast<int64_t>(output.dtype));

        std::vector<uint8_t> shape_data;
        for (int64_t dim : output.shape) {
            std::vector<uint8_t> dim_data;
            if (dim >= 0) {
                write_int64(dim_data, 1, dim);
            } else {
                write_string(dim_data, 2, "dynamic");
            }
            write_length_delimited(shape_data, 1, dim_data);
        }
        write_length_delimited(tensor_type_data, 2, shape_data);

        write_length_delimited(type_data, 1, tensor_type_data);
        write_length_delimited(value_info_data, 2, type_data);

        write_length_delimited(graph_data, 12, value_info_data);
    }

    // Add graph to model
    write_length_delimited(buffer, 7, graph_data);

    return buffer;
}

auto ONNXExporter::export_to_file(const std::string& filepath) -> void {
    auto bytes = serialize_model();

    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }

    file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    file.close();

    // Model exported successfully (LOG_INFO removed as logging system not available)
}

auto ONNXExporter::export_to_bytes() -> std::vector<uint8_t> {
    return serialize_model();
}

auto ONNXExporter::get_graph() const -> const ONNXGraph& {
    return graph_;
}

auto ONNXExporter::clear() -> void {
    graph_ = ONNXGraph("main_graph");
    context_ = ExportContext();
}

auto ONNXExporter::get_tensor_name(const Tensor& tensor, const std::string& default_name) -> std::string {
    auto existing = context_.get_tensor_name(tensor);
    if (existing.has_value()) {
        return existing.value();
    }

    std::string name = context_.generate_name(default_name);
    context_.register_tensor(tensor, name);
    return name;
}

auto ONNXExporter::add_initializer_tensor(const Tensor& tensor, const std::string& name) -> void {
    ONNXTensor onnx_tensor(tensor, name);
    graph_.add_initializer(onnx_tensor);
    context_.register_tensor(tensor, name);
}

// High-level export function

auto export_to_onnx(std::shared_ptr<nn::Module> module,
                    const Tensor& dummy_input,
                    const std::string& filepath,
                    const std::vector<std::string>& input_names,
                    const std::vector<std::string>& output_names,
                    int64_t opset_version) -> void {
    if (!module) {
        throw std::runtime_error("Cannot export null module to ONNX");
    }

    // Validate input names
    if (input_names.empty()) {
        throw std::runtime_error("At least one input name must be provided");
    }

    if (output_names.empty()) {
        throw std::runtime_error("At least one output name must be provided");
    }

    try {
        // Create ONNX exporter
        ONNXExporter exporter(opset_version);

        // Set model metadata
        exporter.set_model_name("tenzor_traced_model");
        exporter.set_description("Model traced and exported from Tenzor");

        // Ensure module is in evaluation mode for consistent tracing
        bool was_training = module->is_training();
        module->eval();

        // Wrap input tensor in Variable for module forward pass
        Variable input_var(dummy_input.cpu().contiguous(), false);

        // Add input to ONNX graph
        exporter.add_input(input_var.tensor(), input_names[0]);

        // Export all module parameters as initializers
        auto named_params = module->named_parameters();
        for (const auto& [param_name, param_var] : named_params) {
            if (param_var && param_var->is_initialized() && param_var->tensor().numel() > 0) {
                std::string safe_name = param_name;
                // Replace dots with underscores for ONNX compatibility
                std::replace(safe_name.begin(), safe_name.end(), '.', '_');

                // Use private member access (friend function)
                exporter.add_initializer_tensor(param_var->tensor().cpu().contiguous(), safe_name);
            }
        }

        // Export all module buffers as initializers
        auto named_buffs = module->named_buffers();
        for (const auto& [buffer_name, buffer_var] : named_buffs) {
            if (buffer_var && buffer_var->is_initialized() && buffer_var->tensor().numel() > 0) {
                std::string safe_name = buffer_name;
                std::replace(safe_name.begin(), safe_name.end(), '.', '_');

                // Use private member access (friend function)
                exporter.add_initializer_tensor(buffer_var->tensor().cpu().contiguous(), safe_name);
            }
        }

        // Trace the forward pass
        // NOTE: This is a simplified tracing approach. A complete implementation would:
        // 1. Hook into tensor operations to capture the computation graph
        // 2. Convert each operation to corresponding ONNX nodes
        // 3. Handle control flow and dynamic operations
        //
        // For now, we perform symbolic tracing for common layer types

        // Run forward pass to get output shape
        Variable output_var = module->forward(input_var);

        if (!output_var.is_initialized() || output_var.tensor().numel() == 0) {
            throw std::runtime_error("Module forward pass produced undefined or empty output");
        }

        // Attempt to trace the module structure
        // This is a best-effort approach that works for simple modules
        trace_custom_module(exporter, module, input_var, output_var,
                          input_names[0], output_names[0]);

        // Add output to ONNX graph
        exporter.add_output(output_var.tensor().cpu(), output_names[0]);

        // Export to file
        exporter.export_to_file(filepath);

        // Restore original training mode
        if (was_training) {
            module->train();
        }

    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to export module to ONNX: ") + e.what());
    }
}

// Helper function to trace custom modules
namespace {

auto trace_custom_module(ONNXExporter& exporter,
                        std::shared_ptr<nn::Module> module,
                        const Variable& input,
                        const Variable& output,
                        const std::string& input_name,
                        const std::string& output_name) -> void {
    // For custom modules, we attempt to infer structure from parameters
    // This is a simplified approach that works for basic feed-forward networks

    auto named_params = module->named_parameters();

    if (named_params.empty()) {
        // Module has no parameters - might be just activation functions
        // Create identity or pass-through
        throw std::runtime_error(
            "Cannot trace custom module without parameters. "
            "Please use ONNXExporter directly or implement your module as Sequential."
        );
    }

    // For modules with parameters, we attempt pattern matching
    // Look for common patterns: weight + bias = Linear layer, etc.

    bool has_weight = false;
    bool has_bias = false;
    std::shared_ptr<Variable> weight_param;
    std::shared_ptr<Variable> bias_param;

    for (const auto& [name, param] : named_params) {
        if (name.find("weight") != std::string::npos) {
            has_weight = true;
            weight_param = param;
        }
        if (name.find("bias") != std::string::npos) {
            has_bias = true;
            bias_param = param;
        }
    }

    if (has_weight && weight_param) {
        auto weight_shape = weight_param->tensor().shape();

        // Check if this looks like a linear layer (2D weight matrix)
        if (weight_shape.size() == 2) {
            std::optional<Tensor> bias_tensor;
            if (has_bias && bias_param) {
                bias_tensor = bias_param->tensor();
            }

            exporter.export_linear(
                input.tensor(),
                weight_param->tensor(),
                bias_tensor,
                output.tensor(),
                output_name
            );
            return;
        }

        // Check if this looks like a Conv2d layer (4D weight matrix)
        if (weight_shape.size() == 4) {
            // Conv2d weight shape: [out_channels, in_channels, kernel_h, kernel_w]
            std::vector<int64_t> kernel_size = {weight_shape[2], weight_shape[3]};
            std::vector<int64_t> stride = {1, 1}; // Assume stride 1
            std::vector<int64_t> padding = {0, 0}; // Assume no padding
            std::vector<int64_t> dilation = {1, 1};

            std::optional<Tensor> bias_tensor;
            if (has_bias && bias_param) {
                bias_tensor = bias_param->tensor();
            }

            exporter.export_conv2d(
                input.tensor(),
                weight_param->tensor(),
                bias_tensor,
                kernel_size,
                stride,
                padding,
                dilation,
                1, // groups
                output.tensor(),
                output_name
            );
            return;
        }
    }

    throw std::runtime_error(
        "Cannot automatically trace custom module structure. "
        "Please use ONNXExporter directly or implement your module as Sequential."
    );
}

} // anonymous namespace

} // namespace onnx
} // namespace tenzor
