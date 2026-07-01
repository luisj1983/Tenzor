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
#include "types.hpp"

namespace tenzor {
namespace onnx {

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
    // Wave Inf-C4: ONNX RNN/LSTM/GRU `activations` is a STRINGS list (one
    // entry per direction × cell). Other attributes that take string lists
    // also use this field.
    std::optional<std::vector<std::string>> strings;

    auto get_int(int64_t default_val = 0) const -> int64_t;
    auto get_float(float default_val = 0.0f) const -> float;
    auto get_string(const std::string& default_val = "") const -> std::string;
    auto get_ints(const std::vector<int64_t>& default_val = {}) const -> std::vector<int64_t>;
    auto get_floats(const std::vector<float>& default_val = {}) const -> std::vector<float>;
    auto get_strings(const std::vector<std::string>& default_val = {}) const -> std::vector<std::string>;
};

struct ONNXImportNode {
    std::string op_type;
    std::string name;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::unordered_map<std::string, ONNXAttribute> attributes;

    auto get_attr(const std::string& name) const -> std::optional<ONNXAttribute>;
};

struct ONNXImportValueInfo {
    std::string name;
    ONNXDataType dtype;
    std::vector<int64_t> shape;
};

struct ONNXGraphData {
    std::string name;
    std::vector<ONNXImportNode> nodes;
    std::vector<ONNXImportValueInfo> inputs;
    std::vector<ONNXImportValueInfo> outputs;
    std::unordered_map<std::string, ONNXTensorData> initializers;
    std::unordered_map<std::string, ONNXImportValueInfo> value_info;
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
    // Convert a pre-parsed graph into a Module. Used by tests that hand-build
    // an ONNX graph to cover codepaths no exporter emits (e.g., asymmetric
    // begin/end pads that require a ConstantPad2d prefix).
    auto convert_graph(const ONNXGraphData& graph) -> std::shared_ptr<nn::Module>;
    auto get_model_data() const -> const ONNXModelData&;
    auto set_verbose(bool verbose) -> void;
    auto set_device(Device device) -> void;

private:
    auto parse_model(const std::vector<uint8_t>& data) -> ONNXModelData;
    auto validate_model(const ONNXModelData& model) -> void;
    auto load_initializers(const ONNXGraphData& graph) -> void;
    auto convert_node(const ONNXImportNode& node) -> std::optional<std::shared_ptr<nn::Module>>;

    // Tensor operations
    auto convert_add(const ONNXImportNode& node) -> void;
    auto convert_sub(const ONNXImportNode& node) -> void;
    auto convert_mul(const ONNXImportNode& node) -> void;
    auto convert_div(const ONNXImportNode& node) -> void;
    auto convert_matmul(const ONNXImportNode& node) -> void;
    auto convert_gemm(const ONNXImportNode& node) -> std::shared_ptr<nn::Module>;

    // Shape operations
    auto convert_reshape(const ONNXImportNode& node) -> void;
    auto convert_transpose(const ONNXImportNode& node) -> void;
    auto convert_concat(const ONNXImportNode& node) -> void;
    auto convert_split(const ONNXImportNode& node) -> void;
    auto convert_flatten(const ONNXImportNode& node) -> void;
    auto convert_squeeze(const ONNXImportNode& node) -> void;
    auto convert_unsqueeze(const ONNXImportNode& node) -> void;
    auto convert_slice(const ONNXImportNode& node) -> void;
    auto convert_pad(const ONNXImportNode& node) -> void;
    auto convert_gather(const ONNXImportNode& node) -> void;
    auto convert_gather_elements(const ONNXImportNode& node) -> void;
    auto convert_clip(const ONNXImportNode& node) -> void;
    auto convert_cast(const ONNXImportNode& node) -> void;
    auto convert_dropout(const ONNXImportNode& node) -> void;
    auto convert_resize(const ONNXImportNode& node) -> void;
    auto convert_reduce_sum(const ONNXImportNode& node) -> void;
    auto convert_reduce_mean(const ONNXImportNode& node) -> void;
    auto convert_reduce_max(const ONNXImportNode& node) -> void;
    auto convert_reduce_min(const ONNXImportNode& node) -> void;
    auto convert_reduce_prod(const ONNXImportNode& node) -> void;
    auto convert_reduce_l1(const ONNXImportNode& node) -> void;
    auto convert_reduce_l2(const ONNXImportNode& node) -> void;
    auto convert_shape(const ONNXImportNode& node) -> void;
    auto convert_constant_of_shape(const ONNXImportNode& node) -> void;
    auto convert_where(const ONNXImportNode& node) -> void;
    auto convert_expand(const ONNXImportNode& node) -> void;
    auto convert_pow(const ONNXImportNode& node) -> void;
    auto convert_sqrt(const ONNXImportNode& node) -> void;
    auto convert_neg(const ONNXImportNode& node) -> void;
    auto convert_exp(const ONNXImportNode& node) -> void;
    auto convert_log(const ONNXImportNode& node) -> void;

    // Audit I6: additional op converters wiring through existing tensor ops.
    auto convert_argmax(const ONNXImportNode& node) -> void;
    auto convert_argmin(const ONNXImportNode& node) -> void;
    auto convert_topk(const ONNXImportNode& node) -> void;
    auto convert_tile(const ONNXImportNode& node) -> void;
    auto convert_range(const ONNXImportNode& node) -> void;
    auto convert_non_zero(const ONNXImportNode& node) -> void;
    auto convert_round(const ONNXImportNode& node) -> void;
    auto convert_einsum(const ONNXImportNode& node) -> void;
    auto convert_trilu(const ONNXImportNode& node) -> void;

    // Layer conversion
    auto convert_conv(const ONNXImportNode& node) -> std::shared_ptr<nn::Module>;
    auto convert_conv_transpose(const ONNXImportNode& node) -> std::shared_ptr<nn::Module>;
    auto convert_batch_normalization(const ONNXImportNode& node) -> std::shared_ptr<nn::Module>;
    auto convert_layer_normalization(const ONNXImportNode& node) -> std::shared_ptr<nn::Module>;
    // I6-followup: InstanceNormalization / GroupNormalization op converters.
    auto convert_instance_normalization(const ONNXImportNode& node) -> std::shared_ptr<nn::Module>;
    auto convert_group_normalization(const ONNXImportNode& node) -> std::shared_ptr<nn::Module>;
    // I6-followup: LSTM/GRU/RNN op converters. Each routes through the existing
    // nn::LSTM/GRU/RNN modules with ONNX weight ordering remapped to the
    // tenzor convention.
    auto convert_lstm(const ONNXImportNode& node) -> std::shared_ptr<nn::Module>;
    auto convert_gru(const ONNXImportNode& node)  -> std::shared_ptr<nn::Module>;
    auto convert_rnn(const ONNXImportNode& node)  -> std::shared_ptr<nn::Module>;

    // Activation functions
    // Activations map to nn::Module equivalents so they are appended to the
    // imported Sequential and run (in order) at forward() time. Returning a
    // module — rather than eager-executing on placeholder activations — is what
    // lets a module op (Linear/Conv) feed an activation correctly.
    auto convert_relu(const ONNXImportNode& node) -> std::shared_ptr<nn::Module>;
    auto convert_leaky_relu(const ONNXImportNode& node) -> std::shared_ptr<nn::Module>;
    auto convert_sigmoid(const ONNXImportNode& node) -> std::shared_ptr<nn::Module>;
    auto convert_tanh(const ONNXImportNode& node) -> std::shared_ptr<nn::Module>;
    auto convert_gelu(const ONNXImportNode& node) -> std::shared_ptr<nn::Module>;
    auto convert_softmax(const ONNXImportNode& node) -> std::shared_ptr<nn::Module>;
    auto convert_log_softmax(const ONNXImportNode& node) -> std::shared_ptr<nn::Module>;
    auto convert_elu(const ONNXImportNode& node) -> std::shared_ptr<nn::Module>;
    auto convert_selu(const ONNXImportNode& node) -> std::shared_ptr<nn::Module>;

    // Pooling layers
    auto convert_maxpool(const ONNXImportNode& node) -> std::shared_ptr<nn::Module>;
    auto convert_avgpool(const ONNXImportNode& node) -> std::shared_ptr<nn::Module>;
    auto convert_global_avgpool(const ONNXImportNode& node) -> std::shared_ptr<nn::Module>;

    // Quantization (QDQ)
    auto convert_quantize_linear(const ONNXImportNode& node) -> void;
    auto convert_dequantize_linear(const ONNXImportNode& node) -> void;

    // Helper functions
    auto get_input(const std::string& name) -> Tensor;
    // Read a control/shape input (Slice/Pad/Resize starts/ends/axes/sizes/scales)
    // as a host (CPU) tensor. When the input is a graph initializer the value is
    // decoded straight from the host proto bytes, so the constant is never
    // uploaded to the GPU (and there is no device->host copy). Falls back to a
    // host copy of a computed value only for the rare non-initializer case.
    auto get_host_input(const std::string& name) -> Tensor;
    auto register_output(const std::string& name, const Tensor& tensor) -> void;
    auto log(const std::string& message) -> void;

    bool verbose_ = false;
    Device device_ = Device::cpu();
    // Pointer to the current graph's host-side initializer map (owned by the
    // ONNXGraphData being converted; valid for the duration of convert_graph).
    // Lets shape/control inputs be read from host bytes without uploading the
    // constant to the compute device.
    const std::unordered_map<std::string, ONNXTensorData>* initializers_ptr_ = nullptr;
    ONNXModelData model_data_;
    ONNXImportContext context_;

    // 6th-audit Fix #1: directory containing the .onnx file, used as the
    // base path when resolving `external_data` location entries. Set by
    // `import_from_file`; left empty for `import_from_bytes` (in which case
    // any EXTERNAL initializer throws — there's no anchor for the sidecar).
    std::string external_data_dir_;
    // 7th-audit Fix #3: single-shot flag — set by `import_from_file` after
    // it populates `external_data_dir_`, cleared by `import_from_bytes`
    // on entry. Prevents the dir from leaking across separate calls.
    bool called_from_file_path_anchor_set_ = false;
};

auto import_onnx(const std::string& filepath, bool verbose = false) -> std::shared_ptr<nn::Module>;

} // namespace onnx
} // namespace tenzor
