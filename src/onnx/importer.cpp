/**
 * @file importer.cpp
 * @brief Implementation of ONNX model import functionality
 */

#include "../../include/tenzor/onnx/importer.hpp"
#include "../../include/tenzor/utils/error.hpp"
#include "../../include/tenzor/nn/layers/linear.hpp"
#include "../../include/tenzor/nn/layers/conv.hpp"
#include "../../include/tenzor/nn/layers/batchnorm.hpp"
#include "../../include/tenzor/nn/layers/pooling.hpp"
#include "../../include/tenzor/nn/layers/flatten.hpp"
#include "../../include/tenzor/nn/activations/activations.hpp"
#include "../../include/tenzor/ops/math.hpp"
#include "../../include/tenzor/ops/transform.hpp"
#include <cstring>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <fstream>

namespace tenzor {
namespace onnx {

// ============================================================================
// Helper Functions for Protobuf Parsing
// ============================================================================

namespace {

/**
 * @brief Read varint from buffer
 */
auto read_varint(const uint8_t*& ptr, const uint8_t* end) -> uint64_t {
    uint64_t value = 0;
    int shift = 0;
    while (ptr < end) {
        uint8_t byte = *ptr++;
        value |= static_cast<uint64_t>(byte & 0x7F) << shift;
        if ((byte & 0x80) == 0) {
            return value;
        }
        shift += 7;
    }
    throw std::runtime_error("Incomplete varint in protobuf");
}

/**
 * @brief Read fixed32 from buffer
 */
auto read_fixed32(const uint8_t*& ptr, const uint8_t* end) -> uint32_t {
    if (ptr + 4 > end) {
        throw std::runtime_error("Incomplete fixed32 in protobuf");
    }
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
        value |= static_cast<uint32_t>(*ptr++) << (i * 8);
    }
    return value;
}

/**
 * @brief Read fixed64 from buffer
 */
auto read_fixed64(const uint8_t*& ptr, const uint8_t* end) -> uint64_t {
    if (ptr + 8 > end) {
        throw std::runtime_error("Incomplete fixed64 in protobuf");
    }
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(*ptr++) << (i * 8);
    }
    return value;
}

/**
 * @brief Read length-delimited field
 */
auto read_length_delimited(const uint8_t*& ptr, const uint8_t* end) -> std::vector<uint8_t> {
    uint64_t length = read_varint(ptr, end);
    if (ptr + length > end) {
        throw std::runtime_error("Incomplete length-delimited field in protobuf");
    }
    std::vector<uint8_t> data(ptr, ptr + length);
    ptr += length;
    return data;
}

/**
 * @brief Read string field
 */
auto read_string(const uint8_t*& ptr, const uint8_t* end) -> std::string {
    auto data = read_length_delimited(ptr, end);
    return std::string(data.begin(), data.end());
}

/**
 * @brief Parse ONNX TensorProto
 */
auto parse_tensor(const std::vector<uint8_t>& data) -> ONNXTensorData {
    ONNXTensorData tensor;
    const uint8_t* ptr = data.data();
    const uint8_t* end = ptr + data.size();

    while (ptr < end) {
        uint64_t tag = read_varint(ptr, end);
        uint32_t field_number = tag >> 3;
        uint32_t wire_type = tag & 0x7;

        switch (field_number) {
            case 1: { // dims (repeated int64)
                if (wire_type == 2) { // packed
                    auto dims_data = read_length_delimited(ptr, end);
                    const uint8_t* dims_ptr = dims_data.data();
                    const uint8_t* dims_end = dims_ptr + dims_data.size();
                    while (dims_ptr < dims_end) {
                        tensor.shape.push_back(static_cast<int64_t>(read_varint(dims_ptr, dims_end)));
                    }
                } else { // individual varints
                    tensor.shape.push_back(static_cast<int64_t>(read_varint(ptr, end)));
                }
                break;
            }
            case 2: { // data_type (int32)
                tensor.dtype = static_cast<ONNXDataType>(read_varint(ptr, end));
                break;
            }
            case 8: { // name (string)
                tensor.name = read_string(ptr, end);
                break;
            }
            case 9: { // raw_data (bytes)
                tensor.raw_data = read_length_delimited(ptr, end);
                break;
            }
            default:
                // Skip unknown fields
                if (wire_type == 0) {
                    read_varint(ptr, end);
                } else if (wire_type == 1) {
                    read_fixed64(ptr, end);
                } else if (wire_type == 2) {
                    read_length_delimited(ptr, end);
                } else if (wire_type == 5) {
                    read_fixed32(ptr, end);
                }
                break;
        }
    }

    return tensor;
}

/**
 * @brief Parse ONNX AttributeProto
 */
auto parse_attribute(const std::vector<uint8_t>& data) -> ONNXAttribute {
    ONNXAttribute attr;
    const uint8_t* ptr = data.data();
    const uint8_t* end = ptr + data.size();

    int32_t attr_type = 0;

    while (ptr < end) {
        uint64_t tag = read_varint(ptr, end);
        uint32_t field_number = tag >> 3;
        uint32_t wire_type = tag & 0x7;

        switch (field_number) {
            case 1: { // name (string)
                attr.name = read_string(ptr, end);
                break;
            }
            case 2: { // type (int32)
                attr_type = static_cast<int32_t>(read_varint(ptr, end));
                break;
            }
            case 3: { // i (int64)
                attr.i = static_cast<int64_t>(read_varint(ptr, end));
                break;
            }
            case 4: { // f (float)
                uint32_t bits = read_fixed32(ptr, end);
                float value;
                std::memcpy(&value, &bits, sizeof(float));
                attr.f = value;
                break;
            }
            case 5: { // s (bytes/string)
                attr.s = read_string(ptr, end);
                break;
            }
            case 6: { // t (TensorProto)
                auto tensor_data = read_length_delimited(ptr, end);
                attr.tensor = parse_tensor(tensor_data);
                break;
            }
            case 7: { // ints (repeated int64)
                if (wire_type == 2) { // packed
                    auto ints_data = read_length_delimited(ptr, end);
                    const uint8_t* ints_ptr = ints_data.data();
                    const uint8_t* ints_end = ints_ptr + ints_data.size();
                    std::vector<int64_t> ints;
                    while (ints_ptr < ints_end) {
                        ints.push_back(static_cast<int64_t>(read_varint(ints_ptr, ints_end)));
                    }
                    attr.ints = ints;
                }
                break;
            }
            case 8: { // floats (repeated float)
                if (wire_type == 2) { // packed
                    auto floats_data = read_length_delimited(ptr, end);
                    const uint8_t* floats_ptr = floats_data.data();
                    std::vector<float> floats;
                    while (floats_ptr < floats_data.data() + floats_data.size()) {
                        uint32_t bits = read_fixed32(floats_ptr, floats_data.data() + floats_data.size());
                        float value;
                        std::memcpy(&value, &bits, sizeof(float));
                        floats.push_back(value);
                    }
                    attr.floats = floats;
                }
                break;
            }
            default:
                // Skip unknown fields
                if (wire_type == 0) {
                    read_varint(ptr, end);
                } else if (wire_type == 1) {
                    read_fixed64(ptr, end);
                } else if (wire_type == 2) {
                    read_length_delimited(ptr, end);
                } else if (wire_type == 5) {
                    read_fixed32(ptr, end);
                }
                break;
        }
    }

    return attr;
}

/**
 * @brief Parse ONNX NodeProto
 */
auto parse_node(const std::vector<uint8_t>& data) -> ONNXNode {
    ONNXNode node;
    const uint8_t* ptr = data.data();
    const uint8_t* end = ptr + data.size();

    while (ptr < end) {
        uint64_t tag = read_varint(ptr, end);
        uint32_t field_number = tag >> 3;
        uint32_t wire_type = tag & 0x7;

        switch (field_number) {
            case 1: { // input (repeated string)
                node.inputs.push_back(read_string(ptr, end));
                break;
            }
            case 2: { // output (repeated string)
                node.outputs.push_back(read_string(ptr, end));
                break;
            }
            case 3: { // name (string)
                node.name = read_string(ptr, end);
                break;
            }
            case 4: { // op_type (string)
                node.op_type = read_string(ptr, end);
                break;
            }
            case 5: { // attribute (repeated AttributeProto)
                auto attr_data = read_length_delimited(ptr, end);
                auto attr = parse_attribute(attr_data);
                node.attributes[attr.name] = attr;
                break;
            }
            default:
                if (wire_type == 0) {
                    read_varint(ptr, end);
                } else if (wire_type == 1) {
                    read_fixed64(ptr, end);
                } else if (wire_type == 2) {
                    read_length_delimited(ptr, end);
                } else if (wire_type == 5) {
                    read_fixed32(ptr, end);
                }
                break;
        }
    }

    return node;
}

/**
 * @brief Parse ONNX ValueInfoProto
 */
auto parse_value_info(const std::vector<uint8_t>& data) -> ONNXValueInfo {
    ONNXValueInfo value_info;
    const uint8_t* ptr = data.data();
    const uint8_t* end = ptr + data.size();

    while (ptr < end) {
        uint64_t tag = read_varint(ptr, end);
        uint32_t field_number = tag >> 3;
        uint32_t wire_type = tag & 0x7;

        switch (field_number) {
            case 1: { // name (string)
                value_info.name = read_string(ptr, end);
                break;
            }
            case 2: { // type (TypeProto) - simplified parsing
                auto type_data = read_length_delimited(ptr, end);
                const uint8_t* type_ptr = type_data.data();
                const uint8_t* type_end = type_ptr + type_data.size();

                while (type_ptr < type_end) {
                    uint64_t type_tag = read_varint(type_ptr, type_end);
                    uint32_t type_field = type_tag >> 3;
                    uint32_t type_wire = type_tag & 0x7;

                    if (type_field == 1 && type_wire == 2) { // tensor_type
                        auto tensor_type_data = read_length_delimited(type_ptr, type_end);
                        const uint8_t* tt_ptr = tensor_type_data.data();
                        const uint8_t* tt_end = tt_ptr + tensor_type_data.size();

                        while (tt_ptr < tt_end) {
                            uint64_t tt_tag = read_varint(tt_ptr, tt_end);
                            uint32_t tt_field = tt_tag >> 3;
                            uint32_t tt_wire = tt_tag & 0x7;

                            if (tt_field == 1 && tt_wire == 0) { // elem_type
                                value_info.dtype = static_cast<ONNXDataType>(read_varint(tt_ptr, tt_end));
                            } else if (tt_field == 2 && tt_wire == 2) { // shape
                                auto shape_data = read_length_delimited(tt_ptr, tt_end);
                                const uint8_t* shape_ptr = shape_data.data();
                                const uint8_t* shape_end = shape_ptr + shape_data.size();

                                while (shape_ptr < shape_end) {
                                    uint64_t shape_tag = read_varint(shape_ptr, shape_end);
                                    uint32_t shape_field = shape_tag >> 3;
                                    uint32_t shape_wire = shape_tag & 0x7;

                                    if (shape_field == 1 && shape_wire == 2) { // dim
                                        auto dim_data = read_length_delimited(shape_ptr, shape_end);
                                        const uint8_t* dim_ptr = dim_data.data();
                                        const uint8_t* dim_end = dim_ptr + dim_data.size();

                                        int64_t dim_value = -1;
                                        while (dim_ptr < dim_end) {
                                            uint64_t dim_tag = read_varint(dim_ptr, dim_end);
                                            uint32_t dim_field = dim_tag >> 3;
                                            if (dim_field == 1) { // dim_value
                                                dim_value = static_cast<int64_t>(read_varint(dim_ptr, dim_end));
                                            } else {
                                                // Skip dim_param (dynamic dimension)
                                                if ((dim_tag & 0x7) == 2) {
                                                    read_length_delimited(dim_ptr, dim_end);
                                                } else if ((dim_tag & 0x7) == 0) {
                                                    read_varint(dim_ptr, dim_end);
                                                }
                                            }
                                        }
                                        value_info.shape.push_back(dim_value);
                                    } else {
                                        // Skip unknown fields
                                        if (shape_wire == 0) {
                                            read_varint(shape_ptr, shape_end);
                                        } else if (shape_wire == 2) {
                                            read_length_delimited(shape_ptr, shape_end);
                                        }
                                    }
                                }
                            } else {
                                // Skip unknown fields
                                if (tt_wire == 0) {
                                    read_varint(tt_ptr, tt_end);
                                } else if (tt_wire == 2) {
                                    read_length_delimited(tt_ptr, tt_end);
                                }
                            }
                        }
                    } else {
                        // Skip unknown type fields
                        if (type_wire == 0) {
                            read_varint(type_ptr, type_end);
                        } else if (type_wire == 2) {
                            read_length_delimited(type_ptr, type_end);
                        }
                    }
                }
                break;
            }
            default:
                if (wire_type == 0) {
                    read_varint(ptr, end);
                } else if (wire_type == 2) {
                    read_length_delimited(ptr, end);
                }
                break;
        }
    }

    return value_info;
}

/**
 * @brief Parse ONNX GraphProto
 */
auto parse_graph(const std::vector<uint8_t>& data) -> ONNXGraphData {
    ONNXGraphData graph;
    const uint8_t* ptr = data.data();
    const uint8_t* end = ptr + data.size();

    while (ptr < end) {
        uint64_t tag = read_varint(ptr, end);
        uint32_t field_number = tag >> 3;
        uint32_t wire_type = tag & 0x7;

        switch (field_number) {
            case 1: { // node (repeated NodeProto)
                auto node_data = read_length_delimited(ptr, end);
                graph.nodes.push_back(parse_node(node_data));
                break;
            }
            case 2: { // name (string)
                graph.name = read_string(ptr, end);
                break;
            }
            case 5: { // initializer (repeated TensorProto)
                auto tensor_data = read_length_delimited(ptr, end);
                auto tensor = parse_tensor(tensor_data);
                graph.initializers[tensor.name] = tensor;
                break;
            }
            case 11: { // input (repeated ValueInfoProto)
                auto input_data = read_length_delimited(ptr, end);
                graph.inputs.push_back(parse_value_info(input_data));
                break;
            }
            case 12: { // output (repeated ValueInfoProto)
                auto output_data = read_length_delimited(ptr, end);
                graph.outputs.push_back(parse_value_info(output_data));
                break;
            }
            case 13: { // value_info (repeated ValueInfoProto)
                auto vi_data = read_length_delimited(ptr, end);
                auto vi = parse_value_info(vi_data);
                graph.value_info[vi.name] = vi;
                break;
            }
            default:
                if (wire_type == 0) {
                    read_varint(ptr, end);
                } else if (wire_type == 1) {
                    read_fixed64(ptr, end);
                } else if (wire_type == 2) {
                    read_length_delimited(ptr, end);
                } else if (wire_type == 5) {
                    read_fixed32(ptr, end);
                }
                break;
        }
    }

    return graph;
}

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
// ONNXNode Implementation
// ============================================================================

auto ONNXNode::get_attr(const std::string& name) const -> std::optional<ONNXAttribute> {
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
    ONNXModelData model;
    const uint8_t* ptr = bytes.data();
    const uint8_t* end = ptr + bytes.size();

    while (ptr < end) {
        uint64_t tag = read_varint(ptr, end);
        uint32_t field_number = tag >> 3;
        uint32_t wire_type = tag & 0x7;

        switch (field_number) {
            case 1: { // ir_version (int64)
                model.ir_version = static_cast<int64_t>(read_varint(ptr, end));
                break;
            }
            case 2: { // opset_import (repeated OperatorSetIdProto)
                auto opset_data = read_length_delimited(ptr, end);
                const uint8_t* opset_ptr = opset_data.data();
                const uint8_t* opset_end = opset_ptr + opset_data.size();

                while (opset_ptr < opset_end) {
                    uint64_t opset_tag = read_varint(opset_ptr, opset_end);
                    uint32_t opset_field = opset_tag >> 3;
                    if (opset_field == 2) { // version
                        model.opset_version = static_cast<int64_t>(read_varint(opset_ptr, opset_end));
                    } else {
                        // Skip other fields
                        if ((opset_tag & 0x7) == 0) {
                            read_varint(opset_ptr, opset_end);
                        } else if ((opset_tag & 0x7) == 2) {
                            read_length_delimited(opset_ptr, opset_end);
                        }
                    }
                }
                break;
            }
            case 7: { // graph (GraphProto)
                auto graph_data = read_length_delimited(ptr, end);
                model.graph = parse_graph(graph_data);
                break;
            }
            case 8: { // producer_name (string)
                model.producer_name = read_string(ptr, end);
                break;
            }
            case 11: { // model_version (int64)
                model.model_version = static_cast<int64_t>(read_varint(ptr, end));
                break;
            }
            case 12: { // doc_string (string)
                model.doc_string = read_string(ptr, end);
                break;
            }
            default:
                // Skip unknown fields
                if (wire_type == 0) {
                    read_varint(ptr, end);
                } else if (wire_type == 1) {
                    read_fixed64(ptr, end);
                } else if (wire_type == 2) {
                    read_length_delimited(ptr, end);
                } else if (wire_type == 5) {
                    read_fixed32(ptr, end);
                }
                break;
        }
    }

    return model;
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

auto ONNXImporter::convert_node(const ONNXNode& node) -> std::optional<std::shared_ptr<nn::Module>> {
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
    }

    // Neural network layers (return module)
    else if (node.op_type == "Gemm") {
        return convert_gemm(node);
    } else if (node.op_type == "Conv") {
        return convert_conv(node);
    } else if (node.op_type == "BatchNormalization") {
        return convert_batch_normalization(node);
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

    else {
        throw std::runtime_error("Unsupported ONNX operator: " + node.op_type);
    }
}

// ============================================================================
// Tensor Operations
// ============================================================================

auto ONNXImporter::convert_add(const ONNXNode& node) -> void {
    auto a = get_input(node.inputs[0]);
    auto b = get_input(node.inputs[1]);
    auto result = add(a, b);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_sub(const ONNXNode& node) -> void {
    auto a = get_input(node.inputs[0]);
    auto b = get_input(node.inputs[1]);
    auto result = sub(a, b);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_mul(const ONNXNode& node) -> void {
    auto a = get_input(node.inputs[0]);
    auto b = get_input(node.inputs[1]);
    auto result = mul(a, b);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_div(const ONNXNode& node) -> void {
    auto a = get_input(node.inputs[0]);
    auto b = get_input(node.inputs[1]);
    auto result = div(a, b);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_matmul(const ONNXNode& node) -> void {
    auto a = get_input(node.inputs[0]);
    auto b = get_input(node.inputs[1]);
    auto result = matmul(a, b);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_gemm(const ONNXNode& node) -> std::shared_ptr<nn::Module> {
    // ONNX Gemm: Y = alpha * A @ B^T + beta * C
    // Tenzor Linear: Y = X @ W^T + b
    // For standard Linear layer: alpha=1, beta=1, transB=1

    auto weight = get_input(node.inputs[1]); // Weight matrix
    std::optional<Tensor> bias;
    if (node.inputs.size() > 2) {
        bias = get_input(node.inputs[2]);
    }

    // Get attributes
    float alpha = node.get_attr("alpha").value_or(ONNXAttribute{}).get_float(1.0f);
    float beta = node.get_attr("beta").value_or(ONNXAttribute{}).get_float(1.0f);
    int64_t transB = node.get_attr("transB").value_or(ONNXAttribute{}).get_int(0);

    // Validate standard Linear layer configuration
    if (alpha != 1.0f || beta != 1.0f) {
        throw std::runtime_error("ONNX Gemm with alpha!=1 or beta!=1 not supported");
    }

    auto weight_shape = weight.shape();
    int64_t in_features = weight_shape[1];
    int64_t out_features = weight_shape[0];

    if (transB == 1) {
        // Weight is already transposed in ONNX (matches Tenzor Linear)
        std::swap(in_features, out_features);
    }

    // Create Linear layer
    auto linear = std::make_shared<nn::Linear>(in_features, out_features, bias.has_value());

    // Set weights
    linear->weight()->tensor() = transB == 1 ? weight : weight.transpose(0, 1);
    if (bias.has_value()) {
        linear->bias()->tensor() = bias.value();
    }

    return linear;
}

auto ONNXImporter::convert_reshape(const ONNXNode& node) -> void {
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

auto ONNXImporter::convert_transpose(const ONNXNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    auto perm_attr = node.get_attr("perm");

    if (!perm_attr.has_value() || !perm_attr->ints.has_value()) {
        throw std::runtime_error("Transpose node missing 'perm' attribute");
    }

    auto perm = perm_attr->ints.value();
    auto result = input.permute(perm);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_concat(const ONNXNode& node) -> void {
    std::vector<Tensor> inputs;
    for (const auto& input_name : node.inputs) {
        inputs.push_back(get_input(input_name));
    }

    auto axis_attr = node.get_attr("axis");
    int64_t axis = axis_attr.has_value() ? axis_attr->get_int(0) : 0;

    auto result = cat(inputs, axis);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_split(const ONNXNode& node) -> void {
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

auto ONNXImporter::convert_flatten(const ONNXNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    auto axis_attr = node.get_attr("axis");
    int64_t axis = axis_attr.has_value() ? axis_attr->get_int(1) : 1;

    auto result = input.flatten(axis);
    register_output(node.outputs[0], result);
}

// ============================================================================
// Neural Network Layers
// ============================================================================

auto ONNXImporter::convert_conv(const ONNXNode& node) -> std::shared_ptr<nn::Module> {
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
        int64_t pad_w = pads[1];

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

auto ONNXImporter::convert_batch_normalization(const ONNXNode& node) -> std::shared_ptr<nn::Module> {
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

auto ONNXImporter::convert_relu(const ONNXNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    Variable var_input(input);
    auto result = nn::relu(var_input).tensor();
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_leaky_relu(const ONNXNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    float alpha = node.get_attr("alpha").value_or(ONNXAttribute{}).get_float(0.01f);
    Variable var_input(input);
    auto result = nn::leaky_relu(var_input, alpha).tensor();
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_sigmoid(const ONNXNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    Variable var_input(input);
    auto result = nn::sigmoid(var_input).tensor();
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_tanh(const ONNXNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    Variable var_input(input);
    auto result = nn::tanh(var_input).tensor();
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_gelu(const ONNXNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    Variable var_input(input);
    auto result = nn::gelu(var_input).tensor();
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_softmax(const ONNXNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    int64_t axis = node.get_attr("axis").value_or(ONNXAttribute{}).get_int(-1);
    Variable var_input(input);
    auto result = nn::softmax(var_input, axis).tensor();
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_log_softmax(const ONNXNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    int64_t axis = node.get_attr("axis").value_or(ONNXAttribute{}).get_int(-1);
    Variable var_input(input);
    auto result = nn::log_softmax(var_input, axis).tensor();
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_elu(const ONNXNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    float alpha = node.get_attr("alpha").value_or(ONNXAttribute{}).get_float(1.0f);
    Variable var_input(input);
    auto result = nn::elu(var_input, alpha).tensor();
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_selu(const ONNXNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    Variable var_input(input);
    auto result = nn::selu(var_input).tensor();
    register_output(node.outputs[0], result);
}

// ============================================================================
// Pooling Layers
// ============================================================================

auto ONNXImporter::convert_maxpool(const ONNXNode& node) -> std::shared_ptr<nn::Module> {
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

auto ONNXImporter::convert_avgpool(const ONNXNode& node) -> std::shared_ptr<nn::Module> {
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

auto ONNXImporter::convert_global_avgpool(const ONNXNode& node) -> std::shared_ptr<nn::Module> {
    // GlobalAveragePool is AdaptiveAvgPool with output_size=(1, 1)
    auto pool = std::make_shared<nn::AdaptiveAvgPool2d>(1, 1);
    return pool;
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
// High-level Import Function
// ============================================================================

auto import_onnx(const std::string& filepath, bool verbose) -> std::shared_ptr<nn::Module> {
    ONNXImporter importer(verbose);
    return importer.import_from_file(filepath);
}

} // namespace onnx
} // namespace tenzor
