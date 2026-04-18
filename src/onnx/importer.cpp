/**
 * @file importer.cpp
 * @brief Implementation of ONNX model import functionality
 *
 * As of the P3 pass, this importer uses generated protobuf bindings from
 * proto/onnx.proto (via onnx.pb.h) instead of a hand-rolled wire-format
 * decoder. The hand-rolled version used non-canonical field numbers and
 * could not interoperate with onnxruntime / netron; this path targets the
 * canonical ONNX schema, so Tenzor-emitted and externally-produced models
 * are bidirectionally compatible.
 */

#include "../../include/tenzor/onnx/importer.hpp"
#include "../../include/tenzor/utils/error.hpp"
#include "../../include/tenzor/nn/layers/linear.hpp"
#include "../../include/tenzor/nn/layers/conv.hpp"
#include "../../include/tenzor/nn/layers/batchnorm.hpp"
#include "../../include/tenzor/nn/layers/normalization.hpp"
#include "../../include/tenzor/nn/layers/pooling.hpp"
#include "../../include/tenzor/nn/layers/flatten.hpp"
#include "../../include/tenzor/nn/activations/activations.hpp"
#include "../../include/tenzor/ops/math.hpp"
#include "../../include/tenzor/ops/transform.hpp"
#include "../../include/tenzor/ops/creation.hpp"
#include "../../include/tenzor/ops/reduction.hpp"
#include "../../include/tenzor/ops/indexing.hpp"
#include "../../include/tenzor/ops/vision.hpp"
#include <cstring>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <fstream>

#ifdef TENZOR_HAS_ONNX_PROTOBUF
#include "onnx.pb.h"
#endif

namespace tenzor {
namespace onnx {

// ============================================================================
// Protobuf → Internal IR conversion
// ============================================================================

namespace {

#ifdef TENZOR_HAS_ONNX_PROTOBUF

// Canonical ONNX AttributeType enum values (from proto/onnx.proto). Kept
// as plain integers here so the compile-time cost of including the full
// generated header in every translation unit is avoided.
constexpr int32_t kAttrTypeFloat  = 1;
constexpr int32_t kAttrTypeInt    = 2;
constexpr int32_t kAttrTypeString = 3;
constexpr int32_t kAttrTypeTensor = 4;
constexpr int32_t kAttrTypeFloats = 6;
constexpr int32_t kAttrTypeInts   = 7;

auto proto_to_ir_tensor(const tenzor_onnx::TensorProto& t) -> ONNXTensorData {
    ONNXTensorData tensor;
    tensor.name = t.name();
    tensor.dtype = static_cast<ONNXDataType>(t.data_type());
    tensor.shape.assign(t.dims().begin(), t.dims().end());

    if (!t.raw_data().empty()) {
        const std::string& raw = t.raw_data();
        tensor.raw_data.assign(raw.begin(), raw.end());
    } else if (!t.float_data().empty()) {
        // Producer used typed data instead of raw_data. Pack into raw_data
        // for the existing ONNXTensorData::to_tensor path (little-endian).
        tensor.raw_data.resize(static_cast<size_t>(t.float_data_size()) * sizeof(float));
        std::memcpy(tensor.raw_data.data(), t.float_data().data(),
                    tensor.raw_data.size());
    } else if (!t.int32_data().empty()) {
        tensor.raw_data.resize(static_cast<size_t>(t.int32_data_size()) * sizeof(int32_t));
        std::memcpy(tensor.raw_data.data(), t.int32_data().data(),
                    tensor.raw_data.size());
    } else if (!t.int64_data().empty()) {
        tensor.raw_data.resize(static_cast<size_t>(t.int64_data_size()) * sizeof(int64_t));
        std::memcpy(tensor.raw_data.data(), t.int64_data().data(),
                    tensor.raw_data.size());
    } else if (!t.double_data().empty()) {
        tensor.raw_data.resize(static_cast<size_t>(t.double_data_size()) * sizeof(double));
        std::memcpy(tensor.raw_data.data(), t.double_data().data(),
                    tensor.raw_data.size());
    }
    return tensor;
}

auto proto_to_ir_attribute(const tenzor_onnx::AttributeProto& a) -> ONNXAttribute {
    ONNXAttribute attr;
    attr.name = a.name();
    const int32_t type = static_cast<int32_t>(a.type());

    // Canonical ONNX marks the payload via `type`. We honour that, but also
    // fall back to whichever typed field is populated for producers that
    // leave `type` at UNDEFINED.
    if (type == kAttrTypeInt || (a.i() != 0 && type == 0)) {
        attr.i = static_cast<int64_t>(a.i());
    }
    if (type == kAttrTypeFloat || (a.f() != 0.0f && type == 0)) {
        attr.f = a.f();
    }
    if (type == kAttrTypeString || (!a.s().empty() && type == 0)) {
        attr.s = a.s();
    }
    if (type == kAttrTypeTensor && a.has_t()) {
        attr.tensor = proto_to_ir_tensor(a.t());
    }
    if (type == kAttrTypeInts || (!a.ints().empty() && type == 0)) {
        std::vector<int64_t> ints(a.ints().begin(), a.ints().end());
        attr.ints = std::move(ints);
    }
    if (type == kAttrTypeFloats || (!a.floats().empty() && type == 0)) {
        std::vector<float> floats(a.floats().begin(), a.floats().end());
        attr.floats = std::move(floats);
    }
    return attr;
}

auto proto_to_ir_node(const tenzor_onnx::NodeProto& n) -> ONNXImportNode {
    ONNXImportNode node;
    node.op_type = n.op_type();
    node.name    = n.name();
    node.inputs.assign(n.input().begin(), n.input().end());
    node.outputs.assign(n.output().begin(), n.output().end());
    for (const auto& a : n.attribute()) {
        auto attr = proto_to_ir_attribute(a);
        node.attributes[attr.name] = std::move(attr);
    }
    return node;
}

auto proto_to_ir_value_info(const tenzor_onnx::ValueInfoProto& vi) -> ONNXImportValueInfo {
    ONNXImportValueInfo info;
    info.name = vi.name();
    if (vi.has_type()) {
        const auto& type = vi.type();
        // Only tensor_type is supported (see comment on TypeProto in
        // proto/onnx.proto). Other branches silently fall through.
        if (type.has_tensor_type()) {
            const auto& tt = type.tensor_type();
            info.dtype = static_cast<ONNXDataType>(tt.elem_type());
            if (tt.has_shape()) {
                for (const auto& d : tt.shape().dim()) {
                    // Unknown / dynamic dimensions are signalled as -1.
                    info.shape.push_back(
                        d.dim_value() != 0 ? static_cast<int64_t>(d.dim_value()) : -1);
                }
            }
        }
    }
    return info;
}

auto proto_to_ir_graph(const tenzor_onnx::GraphProto& g) -> ONNXGraphData {
    ONNXGraphData graph;
    graph.name = g.name();
    for (const auto& n : g.node()) {
        graph.nodes.push_back(proto_to_ir_node(n));
    }
    for (const auto& init : g.initializer()) {
        auto tensor = proto_to_ir_tensor(init);
        graph.initializers[tensor.name] = std::move(tensor);
    }
    for (const auto& i : g.input()) {
        graph.inputs.push_back(proto_to_ir_value_info(i));
    }
    for (const auto& o : g.output()) {
        graph.outputs.push_back(proto_to_ir_value_info(o));
    }
    for (const auto& vi : g.value_info()) {
        auto info = proto_to_ir_value_info(vi);
        graph.value_info[info.name] = std::move(info);
    }
    return graph;
}

#endif // TENZOR_HAS_ONNX_PROTOBUF

auto shape_to_string(const std::vector<int64_t>& shape) -> std::string {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

} // anonymous namespace

// ============================================================================
// DType Conversion
// ============================================================================

auto onnx_to_dtype(ONNXDataType onnx_dtype) -> DType {
    switch (onnx_dtype) {
        case ONNXDataType::FLOAT: return DType::Float32;
        case ONNXDataType::DOUBLE: return DType::Float64;
        case ONNXDataType::FLOAT16: return DType::Float16;
        case ONNXDataType::BFLOAT16: return DType::BFloat16;
        case ONNXDataType::INT8: return DType::Int8;
        case ONNXDataType::INT16: return DType::Int16;
        case ONNXDataType::INT32: return DType::Int32;
        case ONNXDataType::INT64: return DType::Int64;
        case ONNXDataType::UINT8: return DType::UInt8;
        case ONNXDataType::BOOL: return DType::Bool;
        default:
            throw std::runtime_error("Unsupported ONNX data type for import");
    }
}

// ============================================================================
// ONNXTensorData Implementation
// ============================================================================

auto ONNXTensorData::to_tensor(Device device) const -> Tensor {
    auto tenzor_dtype = onnx_to_dtype(dtype);
    Tensor tensor(shape, tenzor_dtype, device);

    // Copy raw data
    if (!raw_data.empty()) {
        size_t expected_bytes = numel() * tensor.dtype_size();
        if (raw_data.size() != expected_bytes) {
            throw std::runtime_error(
                "ONNX tensor '" + name + "' has mismatched data size. Expected " +
                std::to_string(expected_bytes) + " bytes, got " + std::to_string(raw_data.size())
            );
        }

        // For CPU tensors, direct memcpy
        if (device == Device::cpu()) {
            std::memcpy(tensor.data_ptr(), raw_data.data(), raw_data.size());
        } else {
            // For GPU tensors, create temporary CPU tensor and copy
            Tensor cpu_tensor(shape, tenzor_dtype, Device::cpu());
            std::memcpy(cpu_tensor.data_ptr(), raw_data.data(), raw_data.size());
            tensor = cpu_tensor.to(device);
        }
    }

    return tensor;
}

auto ONNXTensorData::numel() const -> int64_t {
    int64_t result = 1;
    for (int64_t dim : shape) {
        result *= dim;
    }
    return result;
}

// ============================================================================
// ONNXAttribute Implementation
// ============================================================================

auto ONNXAttribute::get_int(int64_t default_val) const -> int64_t {
    return i.value_or(default_val);
}

auto ONNXAttribute::get_float(float default_val) const -> float {
    return f.value_or(default_val);
}

auto ONNXAttribute::get_string(const std::string& default_val) const -> std::string {
    return s.value_or(default_val);
}

auto ONNXAttribute::get_ints(const std::vector<int64_t>& default_val) const -> std::vector<int64_t> {
    return ints.value_or(default_val);
}

auto ONNXAttribute::get_floats(const std::vector<float>& default_val) const -> std::vector<float> {
    return floats.value_or(default_val);
}

// ============================================================================
// ONNXImportNode Implementation
// ============================================================================

auto ONNXImportNode::get_attr(const std::string& name) const -> std::optional<ONNXAttribute> {
    auto it = attributes.find(name);
    if (it != attributes.end()) {
        return it->second;
    }
    return std::nullopt;
}

// ============================================================================
// ONNXImportContext Implementation
// ============================================================================

auto ONNXImportContext::register_value(const std::string& name, const Tensor& tensor) -> void {
    values_[name] = tensor;
}

auto ONNXImportContext::get_value(const std::string& name) const -> std::optional<Tensor> {
    auto it = values_.find(name);
    if (it != values_.end()) {
        return it->second;
    }
    return std::nullopt;
}

auto ONNXImportContext::has_value(const std::string& name) const -> bool {
    return values_.find(name) != values_.end();
}

auto ONNXImportContext::register_module(const std::string& name, std::shared_ptr<nn::Module> module) -> void {
    modules_[name] = module;
}

auto ONNXImportContext::get_modules() const -> const std::unordered_map<std::string, std::shared_ptr<nn::Module>>& {
    return modules_;
}

auto ONNXImportContext::set_device(Device device) -> void {
    device_ = device;
}

auto ONNXImportContext::get_device() const -> Device {
    return device_;
}

// ============================================================================
// ONNXImporter Implementation
// ============================================================================

ONNXImporter::ONNXImporter(bool verbose)
    : verbose_(verbose) {}

auto ONNXImporter::set_verbose(bool verbose) -> void {
    verbose_ = verbose;
}

auto ONNXImporter::set_device(Device device) -> void {
    device_ = device;
    context_.set_device(device);
}

auto ONNXImporter::import_from_file(const std::string& filepath) -> std::shared_ptr<nn::Module> {
    log("Loading ONNX model from file: " + filepath);

    // Read file
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open ONNX file: " + filepath);
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> bytes(file_size);
    file.read(reinterpret_cast<char*>(bytes.data()), file_size);
    file.close();

    log("Read " + std::to_string(file_size) + " bytes from file");

    return import_from_bytes(bytes);
}

auto ONNXImporter::import_from_bytes(const std::vector<uint8_t>& bytes) -> std::shared_ptr<nn::Module> {
    log("Parsing ONNX model");

    // Parse model
    model_data_ = parse_model(bytes);

    log("Validating ONNX model");
    validate_model(model_data_);

    log("Converting ONNX graph to Tenzor module");
    auto module = convert_graph(model_data_.graph);

    log("Model import completed successfully");
    return module;
}

auto ONNXImporter::get_model_data() const -> const ONNXModelData& {
    return model_data_;
}

auto ONNXImporter::parse_model(const std::vector<uint8_t>& bytes) -> ONNXModelData {
#ifdef TENZOR_HAS_ONNX_PROTOBUF
    tenzor_onnx::ModelProto proto;
    if (!proto.ParseFromArray(bytes.data(), static_cast<int>(bytes.size()))) {
        throw std::runtime_error(
            "ONNXImporter::parse_model: failed to deserialize ModelProto — "
            "input is not a valid ONNX/protobuf message");
    }

    ONNXModelData model;
    model.ir_version     = static_cast<int64_t>(proto.ir_version());
    model.model_version  = static_cast<int64_t>(proto.model_version());
    model.producer_name  = proto.producer_name();
    model.doc_string     = proto.doc_string();

    // opset_version: pick the first (and usually only) ai.onnx opset entry.
    // Multi-domain opsets are collapsed to a single int, matching the
    // pre-migration behaviour.
    for (const auto& opset : proto.opset_import()) {
        if (opset.domain().empty() || opset.domain() == "ai.onnx") {
            model.opset_version = static_cast<int64_t>(opset.version());
            break;
        }
    }

    if (proto.has_graph()) {
        model.graph = proto_to_ir_graph(proto.graph());
    }
    return model;
#else
    (void)bytes;
    throw std::runtime_error(
        "ONNXImporter::parse_model: built without protobuf support. "
        "Reconfigure with libprotobuf-dev installed and rebuild tenzor_core.");
#endif
}

auto ONNXImporter::validate_model(const ONNXModelData& model) -> void {
    if (model.ir_version < 3) {
        throw std::runtime_error("Unsupported ONNX IR version: " + std::to_string(model.ir_version));
    }

    if (model.opset_version < 13) {
        log("Warning: ONNX opset version " + std::to_string(model.opset_version) +
            " is older than recommended version 13. Some operators may not be supported.");
    }

    if (model.graph.nodes.empty()) {
        throw std::runtime_error("ONNX model has no operations");
    }

    log("Model validation: IR version=" + std::to_string(model.ir_version) +
        ", opset=" + std::to_string(model.opset_version) +
        ", nodes=" + std::to_string(model.graph.nodes.size()) +
        ", initializers=" + std::to_string(model.graph.initializers.size()));
}

auto ONNXImporter::convert_graph(const ONNXGraphData& graph) -> std::shared_ptr<nn::Module> {
    // Load initializers (weights, biases)
    load_initializers(graph);

    // Create sequential container for model
    auto model = std::make_shared<nn::Sequential>();

    // Convert each node
    for (const auto& node : graph.nodes) {
        log("Converting node: " + node.op_type + " (" + node.name + ")");

        try {
            auto module = convert_node(node);
            if (module.has_value()) {
                model->add_module(module.value());
                context_.register_module(node.name, module.value());
            }
        } catch (const std::exception& e) {
            throw std::runtime_error(
                "Failed to convert ONNX node '" + node.name +
                "' (op_type=" + node.op_type + "): " + e.what()
            );
        }
    }

    return model;
}

auto ONNXImporter::load_initializers(const ONNXGraphData& graph) -> void {
    log("Loading " + std::to_string(graph.initializers.size()) + " initializer tensors");

    for (const auto& [name, tensor_data] : graph.initializers) {
        auto tensor = tensor_data.to_tensor(device_);
        context_.register_value(name, tensor);
        std::vector<int64_t> shape_vec(tensor.shape().begin(), tensor.shape().end());
        log("  Loaded: " + name + " " + shape_to_string(shape_vec));
    }
}

auto ONNXImporter::convert_node(const ONNXImportNode& node) -> std::optional<std::shared_ptr<nn::Module>> {
    // Tensor operations (in-place, return nullopt as they modify context)
    if (node.op_type == "Add") {
        convert_add(node);
        return std::nullopt;
    } else if (node.op_type == "Sub") {
        convert_sub(node);
        return std::nullopt;
    } else if (node.op_type == "Mul") {
        convert_mul(node);
        return std::nullopt;
    } else if (node.op_type == "Div") {
        convert_div(node);
        return std::nullopt;
    } else if (node.op_type == "MatMul") {
        convert_matmul(node);
        return std::nullopt;
    } else if (node.op_type == "Reshape") {
        convert_reshape(node);
        return std::nullopt;
    } else if (node.op_type == "Transpose") {
        convert_transpose(node);
        return std::nullopt;
    } else if (node.op_type == "Concat") {
        convert_concat(node);
        return std::nullopt;
    } else if (node.op_type == "Split") {
        convert_split(node);
        return std::nullopt;
    } else if (node.op_type == "Flatten") {
        convert_flatten(node);
        return std::nullopt;
    } else if (node.op_type == "Squeeze") {
        convert_squeeze(node);
        return std::nullopt;
    } else if (node.op_type == "Unsqueeze") {
        convert_unsqueeze(node);
        return std::nullopt;
    } else if (node.op_type == "Slice") {
        convert_slice(node);
        return std::nullopt;
    } else if (node.op_type == "Pad") {
        convert_pad(node);
        return std::nullopt;
    } else if (node.op_type == "Gather") {
        convert_gather(node);
        return std::nullopt;
    } else if (node.op_type == "Clip") {
        convert_clip(node);
        return std::nullopt;
    } else if (node.op_type == "Cast") {
        convert_cast(node);
        return std::nullopt;
    } else if (node.op_type == "Dropout") {
        convert_dropout(node);
        return std::nullopt;
    } else if (node.op_type == "Resize") {
        convert_resize(node);
        return std::nullopt;
    } else if (node.op_type == "ReduceSum") {
        convert_reduce_sum(node);
        return std::nullopt;
    } else if (node.op_type == "ReduceMean") {
        convert_reduce_mean(node);
        return std::nullopt;
    } else if (node.op_type == "ReduceMax") {
        convert_reduce_max(node);
        return std::nullopt;
    } else if (node.op_type == "Shape") {
        convert_shape(node);
        return std::nullopt;
    } else if (node.op_type == "ConstantOfShape") {
        convert_constant_of_shape(node);
        return std::nullopt;
    } else if (node.op_type == "Where") {
        convert_where(node);
        return std::nullopt;
    } else if (node.op_type == "Expand") {
        convert_expand(node);
        return std::nullopt;
    } else if (node.op_type == "Pow") {
        convert_pow(node);
        return std::nullopt;
    } else if (node.op_type == "Sqrt") {
        convert_sqrt(node);
        return std::nullopt;
    } else if (node.op_type == "Neg") {
        convert_neg(node);
        return std::nullopt;
    } else if (node.op_type == "Exp") {
        convert_exp(node);
        return std::nullopt;
    } else if (node.op_type == "Log") {
        convert_log(node);
        return std::nullopt;
    }

    // Neural network layers (return module)
    else if (node.op_type == "Gemm") {
        return convert_gemm(node);
    } else if (node.op_type == "Conv") {
        return convert_conv(node);
    } else if (node.op_type == "BatchNormalization") {
        return convert_batch_normalization(node);
    } else if (node.op_type == "LayerNormalization") {
        return convert_layer_normalization(node);
    }

    // Activation functions (in-place)
    else if (node.op_type == "Relu") {
        convert_relu(node);
        return std::nullopt;
    } else if (node.op_type == "LeakyRelu") {
        convert_leaky_relu(node);
        return std::nullopt;
    } else if (node.op_type == "Sigmoid") {
        convert_sigmoid(node);
        return std::nullopt;
    } else if (node.op_type == "Tanh") {
        convert_tanh(node);
        return std::nullopt;
    } else if (node.op_type == "Gelu") {
        convert_gelu(node);
        return std::nullopt;
    } else if (node.op_type == "Softmax") {
        convert_softmax(node);
        return std::nullopt;
    } else if (node.op_type == "LogSoftmax") {
        convert_log_softmax(node);
        return std::nullopt;
    } else if (node.op_type == "Elu") {
        convert_elu(node);
        return std::nullopt;
    } else if (node.op_type == "Selu") {
        convert_selu(node);
        return std::nullopt;
    }

    // Pooling layers (return module)
    else if (node.op_type == "MaxPool") {
        return convert_maxpool(node);
    } else if (node.op_type == "AveragePool") {
        return convert_avgpool(node);
    } else if (node.op_type == "GlobalAveragePool") {
        return convert_global_avgpool(node);
    }

    // Quantization (QDQ)
    else if (node.op_type == "QuantizeLinear") {
        convert_quantize_linear(node);
        return std::nullopt;
    } else if (node.op_type == "DequantizeLinear") {
        convert_dequantize_linear(node);
        return std::nullopt;
    }

    else {
        throw std::runtime_error("Unsupported ONNX operator: " + node.op_type);
    }
}

// ============================================================================
// Tensor Operations
// ============================================================================

auto ONNXImporter::convert_add(const ONNXImportNode& node) -> void {
    auto a = get_input(node.inputs[0]);
    auto b = get_input(node.inputs[1]);
    auto result = add(a, b);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_sub(const ONNXImportNode& node) -> void {
    auto a = get_input(node.inputs[0]);
    auto b = get_input(node.inputs[1]);
    auto result = sub(a, b);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_mul(const ONNXImportNode& node) -> void {
    auto a = get_input(node.inputs[0]);
    auto b = get_input(node.inputs[1]);
    auto result = mul(a, b);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_div(const ONNXImportNode& node) -> void {
    auto a = get_input(node.inputs[0]);
    auto b = get_input(node.inputs[1]);
    auto result = div(a, b);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_matmul(const ONNXImportNode& node) -> void {
    auto a = get_input(node.inputs[0]);
    auto b = get_input(node.inputs[1]);
    auto result = matmul(a, b);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_gemm(const ONNXImportNode& node) -> std::shared_ptr<nn::Module> {
    // ONNX Gemm: Y = alpha * A @ B^T + beta * C
    // Tenzor Linear: Y = X @ W^T + b
    //
    // Tenzor exports Linear(in_features, out_features) as Gemm with
    // alpha=beta=1, transB=1, weight stored as (out, in) on the wire (the
    // native Tenzor layout). ONNX Gemm with transB=1 takes B of shape
    // (N, K) and transposes it to (K, N) before the multiply; for our
    // case N=out, K=in, so X[M, in] @ B^T[in, out] → [M, out]. The wire
    // shape therefore matches Tenzor's (out, in) directly.
    //
    // For transB=0 the stored layout is (in, out) and we must transpose
    // to get Tenzor's (out, in) convention.

    auto weight = get_input(node.inputs[1]);
    std::optional<Tensor> bias;
    if (node.inputs.size() > 2) {
        bias = get_input(node.inputs[2]);
    }

    float alpha    = node.get_attr("alpha").value_or(ONNXAttribute{}).get_float(1.0f);
    float beta     = node.get_attr("beta").value_or(ONNXAttribute{}).get_float(1.0f);
    int64_t transB = node.get_attr("transB").value_or(ONNXAttribute{}).get_int(0);

    if (alpha != 1.0f || beta != 1.0f) {
        throw std::runtime_error("ONNX Gemm with alpha!=1 or beta!=1 not supported");
    }

    const auto weight_shape = weight.shape();
    if (weight_shape.size() != 2) {
        throw std::runtime_error(
            "ONNX Gemm: weight must be 2D, got ndim=" +
            std::to_string(weight_shape.size()));
    }

    int64_t out_features;
    int64_t in_features;
    Tensor tenzor_weight;
    if (transB == 1) {
        out_features = weight_shape[0];
        in_features  = weight_shape[1];
        tenzor_weight = weight;
    } else {
        in_features  = weight_shape[0];
        out_features = weight_shape[1];
        tenzor_weight = weight.transpose(0, 1);
    }

    auto linear = std::make_shared<nn::Linear>(in_features, out_features, bias.has_value());
    linear->weight()->tensor() = tenzor_weight;
    if (bias.has_value()) {
        linear->bias()->tensor() = bias.value();
    }
    return linear;
}

auto ONNXImporter::convert_reshape(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    auto shape = get_input(node.inputs[1]); // Shape tensor

    // Extract shape values
    std::vector<int64_t> new_shape;
    const int64_t* shape_data = shape.data<int64_t>();
    for (int64_t i = 0; i < shape.numel(); ++i) {
        new_shape.push_back(shape_data[i]);
    }

    auto result = input.reshape(new_shape);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_transpose(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    auto perm_attr = node.get_attr("perm");

    if (!perm_attr.has_value() || !perm_attr->ints.has_value()) {
        throw std::runtime_error("Transpose node missing 'perm' attribute");
    }

    auto perm = perm_attr->ints.value();
    auto result = input.permute(perm);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_concat(const ONNXImportNode& node) -> void {
    std::vector<Tensor> inputs;
    for (const auto& input_name : node.inputs) {
        inputs.push_back(get_input(input_name));
    }

    auto axis_attr = node.get_attr("axis");
    int64_t axis = axis_attr.has_value() ? axis_attr->get_int(0) : 0;

    auto result = cat(inputs, axis);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_split(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    auto axis_attr = node.get_attr("axis");
    int64_t axis = axis_attr.has_value() ? axis_attr->get_int(0) : 0;

    auto split_attr = node.get_attr("split");
    std::vector<int64_t> split_sizes;
    if (split_attr.has_value() && split_attr->ints.has_value()) {
        split_sizes = split_attr->ints.value();
    } else {
        // Equal splits
        int64_t num_outputs = node.outputs.size();
        int64_t size = input.shape()[axis] / num_outputs;
        split_sizes.assign(num_outputs, size);
    }

    // Manually split the tensor using slice
    std::vector<Tensor> results;
    int64_t offset = 0;
    for (int64_t size : split_sizes) {
        results.push_back(input.slice(axis, offset, offset + size));
        offset += size;
    }

    for (size_t i = 0; i < results.size(); ++i) {
        register_output(node.outputs[i], results[i]);
    }
}

auto ONNXImporter::convert_flatten(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    auto axis_attr = node.get_attr("axis");
    int64_t axis = axis_attr.has_value() ? axis_attr->get_int(1) : 1;

    auto result = input.flatten(axis);
    register_output(node.outputs[0], result);
}

// ============================================================================
// Neural Network Layers
// ============================================================================

auto ONNXImporter::convert_conv(const ONNXImportNode& node) -> std::shared_ptr<nn::Module> {
    auto weight = get_input(node.inputs[1]);
    std::optional<Tensor> bias;
    if (node.inputs.size() > 2) {
        bias = get_input(node.inputs[2]);
    }

    // Get convolution attributes
    auto kernel_shape = node.get_attr("kernel_shape")->get_ints();
    auto strides = node.get_attr("strides").value_or(ONNXAttribute{}).get_ints({1, 1});
    auto pads = node.get_attr("pads").value_or(ONNXAttribute{}).get_ints({0, 0, 0, 0});
    auto dilations = node.get_attr("dilations").value_or(ONNXAttribute{}).get_ints({1, 1});
    int64_t groups = node.get_attr("group").value_or(ONNXAttribute{}).get_int(1);

    // Weight shape: [out_channels, in_channels/groups, kernel_h, kernel_w]
    auto weight_shape = weight.shape();
    int64_t out_channels = weight_shape[0];
    int64_t in_channels = weight_shape[1] * groups;

    // Check if Conv1d or Conv2d
    if (kernel_shape.size() == 1) {
        // Conv1d
        auto conv = std::make_shared<nn::Conv1d>(
            in_channels, out_channels, kernel_shape[0],
            strides[0], pads[0], dilations[0], groups, bias.has_value()
        );

        // Load pretrained weights from ONNX
        auto params = conv->named_parameters();
        for (auto& [name, param] : params) {
            if (name == "weight") {
                param->tensor() = weight;
            } else if (name == "bias" && bias.has_value()) {
                param->tensor() = bias.value();
            }
        }

        return conv;
    } else if (kernel_shape.size() == 2) {
        // Conv2d
        // Convert padding format from [top, left, bottom, right] to [h, w]
        int64_t pad_h = pads[0];

        // Assume square kernel if both dims are same, otherwise use first dimension
        int64_t kernel_size = kernel_shape[0]; // Assuming square kernels
        int64_t stride = strides[0];  // Assuming same stride in both dims
        int64_t padding = pad_h;      // Assuming same padding
        int64_t dilation = dilations[0]; // Assuming same dilation

        auto conv = std::make_shared<nn::Conv2d>(
            in_channels, out_channels,
            kernel_size, stride, padding, dilation,
            groups, bias.has_value()
        );

        // Load pretrained weights from ONNX
        auto params = conv->named_parameters();
        for (auto& [name, param] : params) {
            if (name == "weight") {
                param->tensor() = weight;
            } else if (name == "bias" && bias.has_value()) {
                param->tensor() = bias.value();
            }
        }

        return conv;
    } else {
        throw std::runtime_error("Unsupported convolution dimension: " + std::to_string(kernel_shape.size()));
    }
}

auto ONNXImporter::convert_layer_normalization(const ONNXImportNode& node) -> std::shared_ptr<nn::Module> {
    // ONNX LayerNormalization inputs: X, Scale, Bias (bias optional per spec;
    // we only support the bias-present form the exporter emits).
    auto scale = get_input(node.inputs[1]);
    auto bias  = get_input(node.inputs[2]);
    float eps  = node.get_attr("epsilon").value_or(ONNXAttribute{}).get_float(1e-5f);

    // Scale shape is the normalized_shape. Our LayerNorm constructor takes
    // that as a vector of ints.
    std::vector<int64_t> normalized_shape(scale.shape().begin(), scale.shape().end());
    auto ln = std::make_shared<nn::LayerNorm>(normalized_shape, static_cast<double>(eps),
                                              /*elementwise_affine=*/true);

    auto params = ln->named_parameters();
    for (auto& [name, param] : params) {
        if (name == "weight") param->tensor() = scale;
        else if (name == "bias") param->tensor() = bias;
    }
    return ln;
}

auto ONNXImporter::convert_batch_normalization(const ONNXImportNode& node) -> std::shared_ptr<nn::Module> {
    auto scale = get_input(node.inputs[1]);  // gamma
    auto bias = get_input(node.inputs[2]);   // beta
    auto mean = get_input(node.inputs[3]);   // running mean
    auto var = get_input(node.inputs[4]);    // running variance

    float eps = node.get_attr("epsilon").value_or(ONNXAttribute{}).get_float(1e-5f);

    int64_t num_features = scale.shape()[0];

    // Determine if BatchNorm1d or BatchNorm2d based on input shape
    // For now, assume BatchNorm2d (most common in CNNs)
    auto bn = std::make_shared<nn::BatchNorm2d>(num_features, eps);

    // Load pretrained parameters from ONNX
    auto params = bn->named_parameters();
    for (auto& [name, param] : params) {
        if (name == "weight") {
            param->tensor() = scale;
        } else if (name == "bias") {
            param->tensor() = bias;
        }
    }

    // Load running statistics from ONNX buffers
    auto buffers = bn->named_buffers();
    for (auto& [name, buffer] : buffers) {
        if (name == "running_mean") {
            buffer->tensor() = mean;
        } else if (name == "running_var") {
            buffer->tensor() = var;
        }
    }

    return bn;
}

// ============================================================================
// Activation Functions
// ============================================================================

auto ONNXImporter::convert_relu(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    Variable var_input(input);
    auto result = nn::relu(var_input).tensor();
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_leaky_relu(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    float alpha = node.get_attr("alpha").value_or(ONNXAttribute{}).get_float(0.01f);
    Variable var_input(input);
    auto result = nn::leaky_relu(var_input, alpha).tensor();
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_sigmoid(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    Variable var_input(input);
    auto result = nn::sigmoid(var_input).tensor();
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_tanh(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    Variable var_input(input);
    auto result = nn::tanh(var_input).tensor();
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_gelu(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    Variable var_input(input);
    auto result = nn::gelu(var_input).tensor();
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_softmax(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    int64_t axis = node.get_attr("axis").value_or(ONNXAttribute{}).get_int(-1);
    Variable var_input(input);
    auto result = nn::softmax(var_input, axis).tensor();
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_log_softmax(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    int64_t axis = node.get_attr("axis").value_or(ONNXAttribute{}).get_int(-1);
    Variable var_input(input);
    auto result = nn::log_softmax(var_input, axis).tensor();
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_elu(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    float alpha = node.get_attr("alpha").value_or(ONNXAttribute{}).get_float(1.0f);
    Variable var_input(input);
    auto result = nn::elu(var_input, alpha).tensor();
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_selu(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    Variable var_input(input);
    auto result = nn::selu(var_input).tensor();
    register_output(node.outputs[0], result);
}

// ============================================================================
// Pooling Layers
// ============================================================================

auto ONNXImporter::convert_maxpool(const ONNXImportNode& node) -> std::shared_ptr<nn::Module> {
    auto kernel_shape = node.get_attr("kernel_shape")->get_ints();
    auto strides = node.get_attr("strides").value_or(ONNXAttribute{}).get_ints({1, 1});
    auto pads = node.get_attr("pads").value_or(ONNXAttribute{}).get_ints({0, 0, 0, 0});

    if (kernel_shape.size() == 2) {
        // MaxPool2d - assuming square kernels
        int64_t kernel_size = kernel_shape[0];
        int64_t stride = strides[0];
        int64_t padding = pads[0];

        auto pool = std::make_shared<nn::MaxPool2d>(
            kernel_size, stride, padding
        );

        return pool;
    } else {
        throw std::runtime_error("Unsupported MaxPool dimension: " + std::to_string(kernel_shape.size()));
    }
}

auto ONNXImporter::convert_avgpool(const ONNXImportNode& node) -> std::shared_ptr<nn::Module> {
    auto kernel_shape = node.get_attr("kernel_shape")->get_ints();
    auto strides = node.get_attr("strides").value_or(ONNXAttribute{}).get_ints({1, 1});
    auto pads = node.get_attr("pads").value_or(ONNXAttribute{}).get_ints({0, 0, 0, 0});

    if (kernel_shape.size() == 2) {
        // AvgPool2d - assuming square kernels
        int64_t kernel_size = kernel_shape[0];
        int64_t stride = strides[0];
        int64_t padding = pads[0];

        auto pool = std::make_shared<nn::AvgPool2d>(
            kernel_size, stride, padding
        );

        return pool;
    } else {
        throw std::runtime_error("Unsupported AvgPool dimension: " + std::to_string(kernel_shape.size()));
    }
}

auto ONNXImporter::convert_global_avgpool([[maybe_unused]] const ONNXImportNode& node) -> std::shared_ptr<nn::Module> {
    // GlobalAveragePool is AdaptiveAvgPool with output_size=(1, 1)
    auto pool = std::make_shared<nn::AdaptiveAvgPool2d>(1, 1);
    return pool;
}

// ============================================================================
// New Shape/Tensor Operations (Phase 6 expansion)
// ============================================================================

auto ONNXImporter::convert_squeeze(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    auto axes_attr = node.get_attr("axes");

    if (axes_attr.has_value() && axes_attr->ints.has_value()) {
        // Squeeze specified axes (reverse order to keep indices valid)
        auto axes = axes_attr->ints.value();
        std::sort(axes.begin(), axes.end(), std::greater<>());
        for (int64_t axis : axes) {
            input = input.squeeze(axis);
        }
    } else if (node.inputs.size() > 1) {
        // ONNX opset 13+: axes as second input tensor
        auto axes_tensor = get_input(node.inputs[1]);
        const int64_t* axes_data = axes_tensor.data<int64_t>();
        std::vector<int64_t> axes(axes_data, axes_data + axes_tensor.numel());
        std::sort(axes.begin(), axes.end(), std::greater<>());
        for (int64_t axis : axes) {
            input = input.squeeze(axis);
        }
    } else {
        input = input.squeeze();
    }
    register_output(node.outputs[0], input);
}

auto ONNXImporter::convert_unsqueeze(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);

    std::vector<int64_t> axes;
    if (node.inputs.size() > 1) {
        // ONNX opset 13+: axes as second input tensor
        auto axes_tensor = get_input(node.inputs[1]);
        const int64_t* axes_data = axes_tensor.data<int64_t>();
        axes.assign(axes_data, axes_data + axes_tensor.numel());
    } else {
        auto axes_attr = node.get_attr("axes");
        if (axes_attr.has_value() && axes_attr->ints.has_value()) {
            axes = axes_attr->ints.value();
        }
    }

    // Sort ascending so we insert dims in order
    std::sort(axes.begin(), axes.end());
    for (int64_t axis : axes) {
        input = input.unsqueeze(axis);
    }
    register_output(node.outputs[0], input);
}

auto ONNXImporter::convert_slice(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);

    // ONNX Slice: inputs are [data, starts, ends, axes(optional), steps(optional)]
    auto starts_t = get_input(node.inputs[1]);
    auto ends_t = get_input(node.inputs[2]);

    const int64_t* starts = starts_t.data<int64_t>();
    const int64_t* ends = ends_t.data<int64_t>();
    int64_t num_slices = starts_t.numel();

    std::vector<int64_t> axes_vec;
    std::vector<int64_t> steps_vec;

    if (node.inputs.size() > 3) {
        auto axes_t = get_input(node.inputs[3]);
        const int64_t* axes_data = axes_t.data<int64_t>();
        axes_vec.assign(axes_data, axes_data + axes_t.numel());
    } else {
        for (int64_t i = 0; i < num_slices; ++i) axes_vec.push_back(i);
    }

    if (node.inputs.size() > 4) {
        auto steps_t = get_input(node.inputs[4]);
        const int64_t* steps_data = steps_t.data<int64_t>();
        steps_vec.assign(steps_data, steps_data + steps_t.numel());
    } else {
        steps_vec.assign(num_slices, 1);
    }

    auto result = input;
    for (int64_t i = 0; i < num_slices; ++i) {
        int64_t dim = axes_vec[i];
        int64_t start = starts[i];
        int64_t end = ends[i];
        int64_t step = steps_vec[i];

        // Handle negative indices
        int64_t dim_size = result.shape()[dim < 0 ? dim + result.ndim() : dim];
        if (start < 0) start += dim_size;
        if (end < 0) end += dim_size;
        // Clamp
        start = std::max(int64_t(0), std::min(start, dim_size));
        end = std::max(int64_t(0), std::min(end, dim_size));

        result = result.slice(dim, start, end, step);
    }
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_pad(const ONNXImportNode& node) -> void {
    // Pad is complex — for now, support constant mode with zero padding
    // by creating a larger tensor and copying data in
    auto input = get_input(node.inputs[0]);

    std::vector<int64_t> pads_vec;
    if (node.inputs.size() > 1) {
        auto pads_t = get_input(node.inputs[1]);
        const int64_t* pads_data = pads_t.data<int64_t>();
        pads_vec.assign(pads_data, pads_data + pads_t.numel());
    } else {
        auto pads_attr = node.get_attr("pads");
        if (pads_attr.has_value() && pads_attr->ints.has_value()) {
            pads_vec = pads_attr->ints.value();
        }
    }

    float value = 0.0f;
    if (node.inputs.size() > 2) {
        auto value_t = get_input(node.inputs[2]);
        if (value_t.numel() > 0) {
            value = *static_cast<const float*>(value_t.to(DType::Float32).data_ptr());
        }
    }

    // ONNX pads format: [begin_d0, begin_d1, ..., end_d0, end_d1, ...]
    int64_t ndim = input.ndim();
    std::vector<int64_t> new_shape;
    for (int64_t d = 0; d < ndim; ++d) {
        new_shape.push_back(input.shape()[d] + pads_vec[d] + pads_vec[d + ndim]);
    }

    // For zero-padding, create output and add input at the correct offset
    // Use scatter-style approach: create zero tensor, then add input at offset
    auto result = tenzor::full(new_shape, value, input.dtype(), input.device());

    // Build a padded version by adding the input to the right region
    // For each dimension, compute the start offset from pad_begin
    // Use index_put or manual approach via contiguous iteration
    // Simple approach: iterate and use slice to overwrite
    Tensor view = result;
    for (int64_t d = 0; d < ndim; ++d) {
        int64_t begin = pads_vec[d];
        int64_t end = begin + input.shape()[d];
        view = view.slice(d, begin, end);
    }
    // view and input have the same shape — do element-wise zero + input
    view.fill_(0.0f);
    view += input;

    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_gather(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    auto indices = get_input(node.inputs[1]);
    int64_t axis = node.get_attr("axis").value_or(ONNXAttribute{}).get_int(0);

    auto result = tenzor::gather(input, axis, indices);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_clip(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);

    float min_val = -std::numeric_limits<float>::infinity();
    float max_val = std::numeric_limits<float>::infinity();

    if (node.inputs.size() > 1 && !node.inputs[1].empty()) {
        auto min_t = get_input(node.inputs[1]);
        if (min_t.numel() > 0) {
            min_val = *static_cast<const float*>(min_t.to(DType::Float32).data_ptr());
        }
    }
    if (node.inputs.size() > 2 && !node.inputs[2].empty()) {
        auto max_t = get_input(node.inputs[2]);
        if (max_t.numel() > 0) {
            max_val = *static_cast<const float*>(max_t.to(DType::Float32).data_ptr());
        }
    }

    auto result = tenzor::clamp(input, min_val, max_val);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_cast(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    int64_t to_type = node.get_attr("to").value_or(ONNXAttribute{}).get_int(1);

    // ONNX tensor type enum to DType
    DType dtype;
    switch (to_type) {
        case 1:  dtype = DType::Float32; break;
        case 2:  dtype = DType::UInt8; break;
        case 3:  dtype = DType::Int8; break;
        case 5:  dtype = DType::Int16; break;
        case 6:  dtype = DType::Int32; break;
        case 7:  dtype = DType::Int64; break;
        case 9:  dtype = DType::Bool; break;
        case 10: dtype = DType::Float16; break;
        case 11: dtype = DType::Float64; break;
        case 16: dtype = DType::BFloat16; break;
        default:
            throw std::runtime_error("Unsupported ONNX Cast target type: " + std::to_string(to_type));
    }

    register_output(node.outputs[0], input.to(dtype));
}

auto ONNXImporter::convert_dropout(const ONNXImportNode& node) -> void {
    // In inference mode, dropout is identity
    auto input = get_input(node.inputs[0]);
    register_output(node.outputs[0], input);
    // If there's a mask output, register an empty tensor
    if (node.outputs.size() > 1) {
        register_output(node.outputs[1], Tensor());
    }
}

auto ONNXImporter::convert_resize(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);

    // ONNX Resize: inputs are [X, roi, scales, sizes]
    // Try sizes first (input[3]), then scales (input[2])
    std::vector<int64_t> output_size;

    if (node.inputs.size() > 3 && !node.inputs[3].empty()) {
        auto sizes_t = get_input(node.inputs[3]);
        const int64_t* sizes_data = sizes_t.data<int64_t>();
        // sizes includes batch and channel dims — take only spatial
        int64_t ndim = sizes_t.numel();
        for (int64_t i = 2; i < ndim; ++i) {
            output_size.push_back(sizes_data[i]);
        }
    } else if (node.inputs.size() > 2 && !node.inputs[2].empty()) {
        auto scales_t = get_input(node.inputs[2]);
        auto scales_f32 = scales_t.to(DType::Float32);
        const float* scales = scales_f32.data<float>();
        // scales includes batch and channel dims
        for (int64_t i = 2; i < scales_t.numel(); ++i) {
            output_size.push_back(static_cast<int64_t>(input.shape()[i] * scales[i]));
        }
    } else {
        throw std::runtime_error("ONNX Resize: no sizes or scales provided");
    }

    auto mode_attr = node.get_attr("mode");
    std::string mode = "nearest";
    if (mode_attr.has_value() && mode_attr->s.has_value()) {
        mode = mode_attr->s.value();
        if (mode == "linear") mode = "bilinear";
        if (mode == "cubic") mode = "bicubic";
    }

    auto result = tenzor::ops::interpolate(input, output_size, mode, false);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_reduce_sum(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    bool keepdims = node.get_attr("keepdims").value_or(ONNXAttribute{}).get_int(1) != 0;

    std::optional<int64_t> dim;
    auto axes_attr = node.get_attr("axes");
    if (axes_attr.has_value() && axes_attr->ints.has_value() && !axes_attr->ints->empty()) {
        dim = axes_attr->ints->at(0);
    } else if (node.inputs.size() > 1) {
        auto axes_t = get_input(node.inputs[1]);
        if (axes_t.numel() > 0) {
            dim = axes_t.data<int64_t>()[0];
        }
    }

    auto result = tenzor::sum(input, dim, keepdims);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_reduce_mean(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    bool keepdims = node.get_attr("keepdims").value_or(ONNXAttribute{}).get_int(1) != 0;

    std::optional<int64_t> dim;
    auto axes_attr = node.get_attr("axes");
    if (axes_attr.has_value() && axes_attr->ints.has_value() && !axes_attr->ints->empty()) {
        dim = axes_attr->ints->at(0);
    } else if (node.inputs.size() > 1) {
        auto axes_t = get_input(node.inputs[1]);
        if (axes_t.numel() > 0) {
            dim = axes_t.data<int64_t>()[0];
        }
    }

    auto result = tenzor::mean(input, dim, keepdims);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_reduce_max(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    bool keepdims = node.get_attr("keepdims").value_or(ONNXAttribute{}).get_int(1) != 0;

    std::optional<int64_t> dim;
    auto axes_attr = node.get_attr("axes");
    if (axes_attr.has_value() && axes_attr->ints.has_value() && !axes_attr->ints->empty()) {
        dim = axes_attr->ints->at(0);
    } else if (node.inputs.size() > 1) {
        auto axes_t = get_input(node.inputs[1]);
        if (axes_t.numel() > 0) {
            dim = axes_t.data<int64_t>()[0];
        }
    }

    auto result = tenzor::max(input, dim, keepdims);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_shape(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    auto shape = input.shape();

    // Create a 1D Int64 tensor containing the shape
    Tensor shape_tensor({static_cast<int64_t>(shape.size())}, DType::Int64, Device::cpu());
    int64_t* data = shape_tensor.data<int64_t>();
    for (size_t i = 0; i < shape.size(); ++i) {
        data[i] = shape[i];
    }

    register_output(node.outputs[0], shape_tensor);
}

auto ONNXImporter::convert_constant_of_shape(const ONNXImportNode& node) -> void {
    auto shape_tensor = get_input(node.inputs[0]);
    const int64_t* shape_data = shape_tensor.data<int64_t>();
    std::vector<int64_t> shape(shape_data, shape_data + shape_tensor.numel());

    float value = 0.0f;
    auto value_attr = node.get_attr("value");
    if (value_attr.has_value() && value_attr->tensor.has_value()) {
        // value is a tensor attribute — extract scalar
        auto val_tensor = value_attr->tensor->to_tensor();
        if (val_tensor.numel() > 0) {
            value = *static_cast<const float*>(val_tensor.to(DType::Float32).data_ptr());
        }
    }

    auto result = tenzor::full(shape, value, DType::Float32, Device::cpu());
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_where(const ONNXImportNode& node) -> void {
    auto condition = get_input(node.inputs[0]);
    auto x = get_input(node.inputs[1]);
    auto y = get_input(node.inputs[2]);

    auto result = tenzor::where(condition, x, y);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_expand(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    auto shape_tensor = get_input(node.inputs[1]);
    const int64_t* shape_data = shape_tensor.data<int64_t>();
    std::vector<int64_t> shape(shape_data, shape_data + shape_tensor.numel());

    auto result = tenzor::expand(input, shape);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_pow(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    auto exponent = get_input(node.inputs[1]);

    // If exponent is scalar, use the scalar pow
    if (exponent.numel() == 1) {
        float exp_val = *static_cast<const float*>(exponent.to(DType::Float32).data_ptr());
        auto result = tenzor::pow(input, exp_val);
        register_output(node.outputs[0], result);
    } else {
        // Element-wise pow not directly supported — fall back to exp(log(x) * y)
        auto result = tenzor::exp(tenzor::mul(tenzor::log(input), exponent));
        register_output(node.outputs[0], result);
    }
}

auto ONNXImporter::convert_sqrt(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    register_output(node.outputs[0], tenzor::sqrt(input));
}

auto ONNXImporter::convert_neg(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    register_output(node.outputs[0], tenzor::neg(input));
}

auto ONNXImporter::convert_exp(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    register_output(node.outputs[0], tenzor::exp(input));
}

auto ONNXImporter::convert_log(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    register_output(node.outputs[0], tenzor::log(input));
}

// ============================================================================
// Helper Functions
// ============================================================================

auto ONNXImporter::get_input(const std::string& name) -> Tensor {
    auto value = context_.get_value(name);
    if (!value.has_value()) {
        throw std::runtime_error("Input tensor not found: " + name);
    }
    return value.value();
}

auto ONNXImporter::register_output(const std::string& name, const Tensor& tensor) -> void {
    context_.register_value(name, tensor);
}

auto ONNXImporter::log(const std::string& message) -> void {
    if (verbose_) {
        std::cout << "[ONNX Importer] " << message << std::endl;
    }
}

// ============================================================================
// Quantization (QDQ) Operations
// ============================================================================

auto ONNXImporter::convert_quantize_linear(const ONNXImportNode& node) -> void {
    // QuantizeLinear: y = saturate(round(x / y_scale) + y_zero_point)
    // Inputs: x, y_scale, y_zero_point (optional)
    auto x = get_input(node.inputs[0]);
    auto y_scale = get_input(node.inputs[1]);

    // Compute quantized = round(x / scale) + zero_point
    auto scaled = x / y_scale;

    // Round to nearest even
    auto rounded = round(scaled);

    if (node.inputs.size() > 2 && !node.inputs[2].empty()) {
        auto y_zero_point = get_input(node.inputs[2]);
        rounded = rounded + y_zero_point.to(rounded.dtype());
    }

    // Clamp to INT8 range (most common case)
    auto result = clamp(rounded, -128.0f, 127.0f).to(DType::Int8);

    register_output(node.outputs[0], result);
    log("Converted QuantizeLinear: " + node.outputs[0]);
}

auto ONNXImporter::convert_dequantize_linear(const ONNXImportNode& node) -> void {
    // DequantizeLinear: y = (x - x_zero_point) * x_scale
    // Inputs: x, x_scale, x_zero_point (optional)
    auto x = get_input(node.inputs[0]);
    auto x_scale = get_input(node.inputs[1]);

    // Convert quantized input to float
    auto x_float = x.to(DType::Float32);

    if (node.inputs.size() > 2 && !node.inputs[2].empty()) {
        auto x_zero_point = get_input(node.inputs[2]);
        x_float = x_float - x_zero_point.to(DType::Float32);
    }

    auto result = x_float * x_scale;

    register_output(node.outputs[0], result);
    log("Converted DequantizeLinear: " + node.outputs[0]);
}

// ============================================================================
// High-level Import Function
// ============================================================================

auto import_onnx(const std::string& filepath, bool verbose) -> std::shared_ptr<nn::Module> {
    ONNXImporter importer(verbose);
    return importer.import_from_file(filepath);
}

} // namespace onnx
} // namespace tenzor
