/**
 * @file importer.hpp
 * @brief ONNX model import functionality
 */

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include "../core/tensor.hpp"
#include "../nn/module.hpp"

namespace tenzor {
namespace onnx {

enum class ONNXDataType {
    UNDEFINED = 0, FLOAT = 1, UINT8 = 2, INT8 = 3, UINT16 = 4,
    INT16 = 5, INT32 = 6, INT64 = 7, STRING = 8, BOOL = 9,
    FLOAT16 = 10, DOUBLE = 11, UINT32 = 12, UINT64 = 13,
    COMPLEX64 = 14, COMPLEX128 = 15, BFLOAT16 = 16
};

auto onnx_dtype_to_tenzor(ONNXDataType onnx_type) -> DType;

struct ONNXTensorData {
    std::string name;
    ONNXDataType dtype;
    std::vector<int64_t> shape;
    std::vector<uint8_t> raw_data;
    auto to_tensor(Device device = Device::cpu()) const -> Tensor;
    auto numel() const -> int64_t;
};

struct ONNXAttribute {
    std::string name;
    std::optional<int64_t> i;
    std::optional<float> f;
    std::optional<std::string> s;
    std::optional<ONNXTensorData> tensor;
    std::optional<std::vector<int64_t>> ints;
    std::optional<std::vector<float>> floats;

    auto get_int(int64_t default_val = 0) const -> int64_t;
    auto get_float(float default_val = 0.0f) const -> float;
    auto get_string(const std::string& default_val = "") const -> std::string;
    auto get_ints(const std::vector<int64_t>& default_val = {}) const -> std::vector<int64_t>;
    auto get_floats(const std::vector<float>& default_val = {}) const -> std::vector<float>;
};

struct ONNXNode {
    std::string op_type;
    std::string name;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::unordered_map<std::string, ONNXAttribute> attributes;

    auto get_attr(const std::string& name) const -> std::optional<ONNXAttribute>;
};

struct ONNXValueInfo {
    std::string name;
    ONNXDataType dtype;
    std::vector<int64_t> shape;
};

struct ONNXGraphData {
    std::string name;
    std::vector<ONNXNode> nodes;
    std::vector<ONNXValueInfo> inputs;
    std::vector<ONNXValueInfo> outputs;
    std::unordered_map<std::string, ONNXTensorData> initializers;
    std::unordered_map<std::string, ONNXValueInfo> value_info;
};

struct ONNXModelData {
    int64_t ir_version = 0;
    int64_t opset_version = 0;
    int64_t model_version = 0;
    std::string producer_name;
    std::string doc_string;
    ONNXGraphData graph;
};

class ONNXImportContext {
public:
    auto register_value(const std::string& name, const Tensor& tensor) -> void;
    auto get_value(const std::string& name) const -> std::optional<Tensor>;
    auto has_value(const std::string& name) const -> bool;
    auto register_module(const std::string& name, std::shared_ptr<nn::Module> module) -> void;
    auto get_modules() const -> const std::unordered_map<std::string, std::shared_ptr<nn::Module>>&;
    auto set_device(Device device) -> void;
    auto get_device() const -> Device;

private:
    std::unordered_map<std::string, Tensor> values_;
    std::unordered_map<std::string, std::shared_ptr<nn::Module>> modules_;
    Device device_ = Device::cpu();
};

class ONNXImporter {
public:
    explicit ONNXImporter(bool verbose = false);
    auto import_from_file(const std::string& filepath) -> std::shared_ptr<nn::Module>;
    auto import_from_bytes(const std::vector<uint8_t>& bytes) -> std::shared_ptr<nn::Module>;
    auto get_model_data() const -> const ONNXModelData&;
    auto set_verbose(bool verbose) -> void;
    auto set_device(Device device) -> void;

private:
    auto parse_model(const std::vector<uint8_t>& data) -> ONNXModelData;
    auto validate_model(const ONNXModelData& model) -> void;
    auto convert_graph(const ONNXGraphData& graph) -> std::shared_ptr<nn::Module>;
    auto load_initializers(const ONNXGraphData& graph) -> void;
    auto convert_node(const ONNXNode& node) -> std::optional<std::shared_ptr<nn::Module>>;

    // Tensor operations
    auto convert_add(const ONNXNode& node) -> void;
    auto convert_sub(const ONNXNode& node) -> void;
    auto convert_mul(const ONNXNode& node) -> void;
    auto convert_div(const ONNXNode& node) -> void;
    auto convert_matmul(const ONNXNode& node) -> void;
    auto convert_gemm(const ONNXNode& node) -> std::shared_ptr<nn::Module>;

    // Shape operations
    auto convert_reshape(const ONNXNode& node) -> void;
    auto convert_transpose(const ONNXNode& node) -> void;
    auto convert_concat(const ONNXNode& node) -> void;
    auto convert_split(const ONNXNode& node) -> void;
    auto convert_flatten(const ONNXNode& node) -> void;

    // Layer conversion
    auto convert_conv(const ONNXNode& node) -> std::shared_ptr<nn::Module>;
    auto convert_batch_normalization(const ONNXNode& node) -> std::shared_ptr<nn::Module>;

    // Activation functions
    auto convert_relu(const ONNXNode& node) -> void;
    auto convert_leaky_relu(const ONNXNode& node) -> void;
    auto convert_sigmoid(const ONNXNode& node) -> void;
    auto convert_tanh(const ONNXNode& node) -> void;
    auto convert_gelu(const ONNXNode& node) -> void;
    auto convert_softmax(const ONNXNode& node) -> void;
    auto convert_log_softmax(const ONNXNode& node) -> void;
    auto convert_elu(const ONNXNode& node) -> void;
    auto convert_selu(const ONNXNode& node) -> void;

    // Pooling layers
    auto convert_maxpool(const ONNXNode& node) -> std::shared_ptr<nn::Module>;
    auto convert_avgpool(const ONNXNode& node) -> std::shared_ptr<nn::Module>;
    auto convert_global_avgpool(const ONNXNode& node) -> std::shared_ptr<nn::Module>;

    // Helper functions
    auto get_input(const std::string& name) -> Tensor;
    auto register_output(const std::string& name, const Tensor& tensor) -> void;
    auto log(const std::string& message) -> void;

    bool verbose_ = false;
    Device device_ = Device::cpu();
    ONNXModelData model_data_;
    ONNXImportContext context_;
};

auto import_onnx(const std::string& filepath, bool verbose = false) -> std::shared_ptr<nn::Module>;

} // namespace onnx
} // namespace tenzor
