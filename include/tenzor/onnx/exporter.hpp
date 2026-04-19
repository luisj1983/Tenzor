/**
 * @file exporter.hpp
 * @brief ONNX model export functionality for Tenzor
 *
 * Provides comprehensive ONNX export capabilities for Tenzor models, supporting
 * ONNX opset 13+ with full operator mapping for tensor operations, neural network
 * layers, activations, and pooling operations.
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <optional>
#include <functional>
#include <fstream>
#include "../core/tensor.hpp"
#include "../nn/module.hpp"
#include "../ops/op_id.hpp"
#include "../jit/tracer.hpp"
#include "types.hpp"

namespace tenzor {

// Forward declarations for quantization types
namespace nn::quantization { class QuantizedLinear; }

namespace onnx {

// Forward declarations
class ONNXGraph;
class ONNXExportNode;
class ONNXExportValueInfo;
class ONNXTensor;

/**
 * @brief ONNX tensor representation
 *
 * Represents a tensor in ONNX format with shape, data type, and raw data.
 */
class ONNXTensor {
public:
    std::string name;                    ///< Tensor name
    ONNXDataType dtype;                  ///< Data type
    std::vector<int64_t> dims;           ///< Tensor dimensions
    std::vector<uint8_t> raw_data;       ///< Raw tensor data

    ONNXTensor() = default;

    /**
     * @brief Create ONNX tensor from Tenzor tensor
     *
     * @param tensor Source Tenzor tensor
     * @param name Tensor name in ONNX graph
     */
    explicit ONNXTensor(const Tensor& tensor, const std::string& name);

    /**
     * @brief Get total number of elements
     */
    auto numel() const -> int64_t;

    /**
     * @brief Get size in bytes
     */
    auto size_bytes() const -> size_t;
};

/**
 * @brief ONNX value info (input/output/intermediate value metadata) for export
 */
class ONNXExportValueInfo {
public:
    std::string name;                    ///< Value name
    ONNXDataType dtype;                  ///< Data type
    std::vector<int64_t> shape;          ///< Shape (-1 for dynamic dimensions)

    /**
     * @brief Symbolic names for dynamic dimensions.
     *
     * Maps dimension index to a symbolic dim_param string (e.g., "batch_size",
     * "seq_len"). When a dimension has a dim_param, the ONNX model emits
     * the symbolic name instead of a fixed integer, enabling dynamic shapes
     * at runtime.
     */
    std::unordered_map<int64_t, std::string> dim_params;

    ONNXExportValueInfo() = default;
    ONNXExportValueInfo(const std::string& name, ONNXDataType dtype,
                  const std::vector<int64_t>& shape);
};

/**
 * @brief ONNX graph node (operation) for export
 */
class ONNXExportNode {
public:
    std::string op_type;                                    ///< ONNX operator type
    std::string name;                                       ///< Node name
    std::vector<std::string> inputs;                        ///< Input value names
    std::vector<std::string> outputs;                       ///< Output value names
    std::unordered_map<std::string, int64_t> int_attrs;     ///< Integer attributes
    std::unordered_map<std::string, float> float_attrs;     ///< Float attributes
    std::unordered_map<std::string, std::string> string_attrs; ///< String attributes
    std::unordered_map<std::string, std::vector<int64_t>> ints_attrs; ///< Integer array attributes
    std::unordered_map<std::string, std::vector<float>> floats_attrs; ///< Float array attributes
    std::unordered_map<std::string, ONNXTensor> tensor_attrs; ///< Tensor attributes

    ONNXExportNode(const std::string& op_type, const std::string& name);

    /**
     * @brief Add input value
     */
    auto add_input(const std::string& input) -> void;

    /**
     * @brief Add output value
     */
    auto add_output(const std::string& output) -> void;

    /**
     * @brief Set integer attribute
     */
    auto set_attr(const std::string& key, int64_t value) -> void;

    /**
     * @brief Set float attribute
     */
    auto set_attr(const std::string& key, float value) -> void;

    /**
     * @brief Set string attribute
     */
    auto set_attr(const std::string& key, const std::string& value) -> void;

    /**
     * @brief Set integer array attribute
     */
    auto set_attr(const std::string& key, const std::vector<int64_t>& value) -> void;

    /**
     * @brief Set float array attribute
     */
    auto set_attr(const std::string& key, const std::vector<float>& value) -> void;

    /**
     * @brief Set tensor attribute
     */
    auto set_attr(const std::string& key, const ONNXTensor& value) -> void;
};

/**
 * @brief ONNX computational graph
 */
class ONNXGraph {
public:
    std::string name;                           ///< Graph name
    std::vector<ONNXExportNode> nodes;                ///< Graph nodes (operations)
    std::vector<ONNXExportValueInfo> inputs;          ///< Graph inputs
    std::vector<ONNXExportValueInfo> outputs;         ///< Graph outputs
    std::vector<ONNXTensor> initializers;       ///< Constant tensors (weights, biases)
    std::unordered_map<std::string, ONNXExportValueInfo> value_info; ///< Intermediate values

    explicit ONNXGraph(const std::string& name = "graph");

    /**
     * @brief Add node to graph
     */
    auto add_node(const ONNXExportNode& node) -> void;

    /**
     * @brief Add input to graph
     */
    auto add_input(const ONNXExportValueInfo& input) -> void;

    /**
     * @brief Add output to graph
     */
    auto add_output(const ONNXExportValueInfo& output) -> void;

    /**
     * @brief Add initializer tensor
     */
    auto add_initializer(const ONNXTensor& tensor) -> void;

    /**
     * @brief Register intermediate value info
     */
    auto add_value_info(const ONNXExportValueInfo& info) -> void;

    /**
     * @brief Get unique name for intermediate value
     */
    auto get_unique_name(const std::string& prefix) -> std::string;

private:
    int64_t name_counter_{0}; ///< Counter for unique name generation
};

/**
 * @brief Export context for tracking tensor mappings during export
 */
class ExportContext {
public:
    /**
     * @brief Register tensor mapping from Tenzor to ONNX name
     */
    auto register_tensor(const Tensor& tensor, const std::string& onnx_name) -> void;

    /**
     * @brief Get ONNX name for a tensor
     */
    auto get_tensor_name(const Tensor& tensor) -> std::optional<std::string>;

    /**
     * @brief Check if tensor is already registered
     */
    auto has_tensor(const Tensor& tensor) -> bool;

    /**
     * @brief Generate unique name
     */
    auto generate_name(const std::string& prefix) -> std::string;

private:
    std::unordered_map<const void*, std::string> tensor_map_; ///< Tensor pointer to ONNX name
    std::unordered_map<std::string, int64_t> name_counters_;  ///< Prefix to counter mapping
};

/**
 * @brief ONNX exporter for Tenzor models
 *
 * Provides complete ONNX export functionality with support for:
 * - Tensor operations (add, sub, mul, div, matmul, etc.)
 * - Neural network layers (Linear, Conv2d, BatchNorm2d, etc.)
 * - Activation functions (ReLU, Sigmoid, Tanh, GELU, Softmax, etc.)
 * - Pooling layers (MaxPool2d, AvgPool2d)
 * - Model metadata and graph optimization
 *
 * Supports ONNX opset version 13 and higher.
 *
 * @code
 * // Export a simple model
 * ONNXExporter exporter;
 *
 * // Set model metadata
 * exporter.set_model_name("my_model");
 * exporter.set_opset_version(13);
 *
 * // Add input
 * Tensor input({1, 3, 224, 224}, DType::Float32, Device::cpu());
 * exporter.add_input(input, "input");
 *
 * // Export model operations...
 *
 * // Save to file
 * exporter.export_to_file("model.onnx");
 * @endcode
 */
class ONNXExporter {
public:
    /**
     * @brief Construct ONNX exporter
     *
     * @param opset_version ONNX opset version (default: 13)
     */
    explicit ONNXExporter(int64_t opset_version = 13);

    ~ONNXExporter() = default;

    // ============================================================================
    // Static Convenience API
    // ============================================================================

    /**
     * @brief Export model to ONNX file (static convenience method).
     *
     * Traces the model with example inputs and converts the computation graph
     * to ONNX format. Handles parameter serialization, graph construction,
     * and binary output.
     *
     * @param model Module to export
     * @param example_inputs Example input tensors for shape tracing
     * @param output_path Path for the output .onnx file
     * @param opset_version ONNX opset version (default: 13)
     *
     * @code
     * auto model = std::make_shared<MyModel>();
     * std::vector<Tensor> inputs = {Tensor({1, 3, 224, 224}, DType::Float32, Device::cpu())};
     * ONNXExporter::export_model(*model, inputs, "model.onnx");
     * @endcode
     */
    static void export_model(nn::Module& model,
                             std::vector<Tensor> example_inputs,
                             const std::string& output_path,
                             int opset_version = 13);

    /**
     * @brief Map Tenzor OpId to ONNX operator type string.
     *
     * Provides a direct mapping from Tenzor's internal operation identifiers
     * to the corresponding ONNX operator names.
     *
     * @param op Tenzor operation identifier
     * @return ONNX operator type string (e.g., "Add", "MatMul", "Relu")
     * @throws std::runtime_error if the OpId has no ONNX mapping
     */
    static auto op_to_onnx(OpId op) -> std::string;

    /**
     * @brief Map Tenzor DType to ONNX data type integer.
     *
     * Returns the ONNX TensorProto.DataType integer value for a given
     * Tenzor DType, as defined in the ONNX specification.
     *
     * @param dtype Tenzor data type
     * @return ONNX data type integer (e.g., Float32 -> 1, Float64 -> 11)
     */
    static auto dtype_to_onnx_int(DType dtype) -> int;

    // ============================================================================
    // Model Configuration
    // ============================================================================

    /**
     * @brief Set model name
     */
    auto set_model_name(const std::string& name) -> void;

    /**
     * @brief Set ONNX opset version
     */
    auto set_opset_version(int64_t version) -> void;

    /**
     * @brief Set model description
     */
    auto set_description(const std::string& desc) -> void;

    /**
     * @brief Set producer name (default: "Tenzor")
     */
    auto set_producer_name(const std::string& name) -> void;

    /**
     * @brief Set model version
     */
    auto set_model_version(int64_t version) -> void;

    // ============================================================================
    // Graph Building
    // ============================================================================

    /**
     * @brief Add model input
     *
     * @param tensor Input tensor (shape template)
     * @param name Input name
     * @param dynamic_axes Map of dimension index to axis name for dynamic shapes
     */
    auto add_input(const Tensor& tensor, const std::string& name,
                   const std::unordered_map<int64_t, std::string>& dynamic_axes = {}) -> void;

    /**
     * @brief Add model output
     *
     * @param tensor Output tensor
     * @param name Output name
     * @param dynamic_axes Map of dimension index to axis name for dynamic shapes
     */
    auto add_output(const Tensor& tensor, const std::string& name,
                    const std::unordered_map<int64_t, std::string>& dynamic_axes = {}) -> void;

    // ============================================================================
    // Tensor Operations
    // ============================================================================

    /**
     * @brief Export element-wise add operation
     */
    auto export_add(const Tensor& a, const Tensor& b, const Tensor& output,
                    const std::string& output_name) -> void;

    /**
     * @brief Export element-wise subtract operation
     */
    auto export_sub(const Tensor& a, const Tensor& b, const Tensor& output,
                    const std::string& output_name) -> void;

    /**
     * @brief Export element-wise multiply operation
     */
    auto export_mul(const Tensor& a, const Tensor& b, const Tensor& output,
                    const std::string& output_name) -> void;

    /**
     * @brief Export element-wise divide operation
     */
    auto export_div(const Tensor& a, const Tensor& b, const Tensor& output,
                    const std::string& output_name) -> void;

    /**
     * @brief Export matrix multiplication
     */
    auto export_matmul(const Tensor& a, const Tensor& b, const Tensor& output,
                       const std::string& output_name) -> void;

    /**
     * @brief Export reshape operation
     */
    auto export_reshape(const Tensor& input, const std::vector<int64_t>& new_shape,
                        const Tensor& output, const std::string& output_name) -> void;

    /**
     * @brief Export transpose operation
     */
    auto export_transpose(const Tensor& input, const std::vector<int64_t>& perm,
                          const Tensor& output, const std::string& output_name) -> void;

    /**
     * @brief Export concat operation
     */
    auto export_concat(const std::vector<Tensor>& inputs, int64_t axis,
                       const Tensor& output, const std::string& output_name) -> void;

    /**
     * @brief Export split operation
     */
    auto export_split(const Tensor& input, int64_t axis, const std::vector<int64_t>& split_sizes,
                      const std::vector<Tensor>& outputs, const std::vector<std::string>& output_names) -> void;

    // ============================================================================
    // Neural Network Layers
    // ============================================================================

    /**
     * @brief Export Linear (fully connected) layer
     *
     * @param input Input tensor
     * @param weight Weight matrix
     * @param bias Optional bias vector
     * @param output Output tensor
     * @param output_name Output name in ONNX graph
     */
    auto export_linear(const Tensor& input, const Tensor& weight,
                       const std::optional<Tensor>& bias, const Tensor& output,
                       const std::string& output_name) -> void;

    /**
     * @brief Export Conv2d layer
     */
    auto export_conv2d(const Tensor& input, const Tensor& weight,
                       const std::optional<Tensor>& bias,
                       const std::vector<int64_t>& kernel_size,
                       const std::vector<int64_t>& stride,
                       const std::vector<int64_t>& padding,
                       const std::vector<int64_t>& dilation,
                       int64_t groups,
                       const Tensor& output,
                       const std::string& output_name) -> void;

    /**
     * @brief Export Conv1d layer
     */
    auto export_conv1d(const Tensor& input, const Tensor& weight,
                       const std::optional<Tensor>& bias,
                       int64_t kernel_size, int64_t stride, int64_t padding,
                       int64_t dilation, int64_t groups,
                       const Tensor& output,
                       const std::string& output_name) -> void;

    /**
     * @brief Export Conv3d layer. ONNX uses the same "Conv" op type; the
     *        spatial rank is inferred from the length of kernel_shape.
     */
    auto export_conv3d(const Tensor& input, const Tensor& weight,
                       const std::optional<Tensor>& bias,
                       int64_t kernel_size, int64_t stride, int64_t padding,
                       int64_t dilation, int64_t groups,
                       const Tensor& output,
                       const std::string& output_name) -> void;

    /**
     * @brief Export ConvTranspose{1,2,3}d layer. ONNX op type is
     *        "ConvTranspose". `spatial_rank` is 1, 2, or 3.
     */
    auto export_conv_transpose(const Tensor& input, const Tensor& weight,
                               const std::optional<Tensor>& bias,
                               int64_t spatial_rank,
                               int64_t kernel_size, int64_t stride, int64_t padding,
                               int64_t output_padding, int64_t dilation, int64_t groups,
                               const Tensor& output,
                               const std::string& output_name) -> void;

    /**
     * @brief Export BatchNorm2d layer
     */
    auto export_batchnorm2d(const Tensor& input, const Tensor& scale,
                            const Tensor& bias, const Tensor& mean,
                            const Tensor& var, double eps,
                            const Tensor& output,
                            const std::string& output_name) -> void;

    /**
     * @brief Export BatchNorm1d layer
     */
    auto export_batchnorm1d(const Tensor& input, const Tensor& scale,
                            const Tensor& bias, const Tensor& mean,
                            const Tensor& var, double eps,
                            const Tensor& output,
                            const std::string& output_name) -> void;

    /**
     * @brief Export a LayerNorm layer as ONNX LayerNormalization (opset 17+).
     *
     * Added as part of the tracer fix for custom modules with 1D weight+bias
     * and no running-stats buffers. Older opsets decompose via a ReduceMean
     * + sub + pow + mean + sqrt sequence; that's handled by the generic
     * LayerNorm OpId path, which callers who need pre-17 support should use.
     *
     * @param input Input tensor
     * @param scale Gamma (weight) of shape (C,)
     * @param bias Beta (bias) of shape (C,)
     * @param axis Axis to normalize over (default: -1)
     * @param eps Epsilon for numerical stability
     * @param output Output tensor (same shape as input)
     * @param output_name ONNX-side name for the output value
     */
    auto export_layernorm(const Tensor& input, const Tensor& scale,
                          const Tensor& bias, int64_t axis, double eps,
                          const Tensor& output,
                          const std::string& output_name) -> void;

    /**
     * @brief Export GroupNorm layer
     *
     * For opset >= 18, exports as ONNX GroupNormalization.
     * For earlier opsets, decomposes into Reshape + InstanceNormalization + Reshape
     * to achieve the same semantics.
     *
     * @param input Input tensor of shape (N, C, *)
     * @param weight Scale (gamma) tensor of shape (C)
     * @param bias Shift (beta) tensor of shape (C)
     * @param num_groups Number of groups to divide channels into
     * @param eps Epsilon for numerical stability
     * @param output Output tensor (same shape as input)
     * @param output_name Output name in ONNX graph
     */
    auto export_groupnorm(const Tensor& input, const Tensor& weight,
                          const Tensor& bias, int64_t num_groups, double eps,
                          const Tensor& output,
                          const std::string& output_name) -> void;

    /**
     * @brief Export LSTM layer
     *
     * Exports an LSTM as the ONNX LSTM operator. Handles the weight layout
     * transform from Tenzor's (i, o, f, g) gate order to ONNX's required
     * (i, o, f, c) gate order, and packs input-to-hidden and hidden-to-hidden
     * weights into the ONNX format [num_directions, 4*hidden_size, input_size].
     *
     * @param input Input tensor of shape (seq_len, batch, input_size)
     * @param weight_ih Input-to-hidden weight of shape (4*hidden_size, input_size)
     * @param weight_hh Hidden-to-hidden weight of shape (4*hidden_size, hidden_size)
     * @param bias_ih Optional input-to-hidden bias of shape (4*hidden_size)
     * @param bias_hh Optional hidden-to-hidden bias of shape (4*hidden_size)
     * @param hidden_size Hidden state size
     * @param num_layers Number of stacked LSTM layers (only layer 0 exported here)
     * @param bidirectional Whether the LSTM is bidirectional
     * @param output Output tensor
     * @param output_name Output name in ONNX graph
     */
    auto export_lstm(const Tensor& input,
                     const Tensor& weight_ih, const Tensor& weight_hh,
                     const std::optional<Tensor>& bias_ih,
                     const std::optional<Tensor>& bias_hh,
                     int64_t hidden_size, int64_t num_layers,
                     bool bidirectional,
                     const Tensor& output,
                     const std::string& output_name) -> void;

    /**
     * @brief Export GRU layer
     *
     * Exports a GRU as the ONNX GRU operator. Handles the weight layout
     * transform from Tenzor's (r, z, n) gate order to ONNX's required
     * (z, r, h) gate order, and packs weights into the ONNX format
     * [num_directions, 3*hidden_size, input_size].
     *
     * @param input Input tensor of shape (seq_len, batch, input_size)
     * @param weight_ih Input-to-hidden weight of shape (3*hidden_size, input_size)
     * @param weight_hh Hidden-to-hidden weight of shape (3*hidden_size, hidden_size)
     * @param bias_ih Optional input-to-hidden bias of shape (3*hidden_size)
     * @param bias_hh Optional hidden-to-hidden bias of shape (3*hidden_size)
     * @param hidden_size Hidden state size
     * @param num_layers Number of stacked GRU layers (only layer 0 exported here)
     * @param bidirectional Whether the GRU is bidirectional
     * @param output Output tensor
     * @param output_name Output name in ONNX graph
     */
    auto export_gru(const Tensor& input,
                    const Tensor& weight_ih, const Tensor& weight_hh,
                    const std::optional<Tensor>& bias_ih,
                    const std::optional<Tensor>& bias_hh,
                    int64_t hidden_size, int64_t num_layers,
                    bool bidirectional,
                    const Tensor& output,
                    const std::string& output_name) -> void;

    /**
     * @brief Export MultiHeadAttention layer
     *
     * Decomposes multi-head attention into ONNX primitives:
     *   1. Q/K/V linear projections (MatMul + Add)
     *   2. Reshape + Transpose into multi-head layout
     *   3. Scaled dot-product attention (MatMul + Div + Softmax + MatMul)
     *   4. Reshape + output projection (MatMul + Add)
     *
     * @param query Query input tensor of shape (N, L, E) or (L, N, E)
     * @param key Key input tensor of shape (N, S, E) or (S, N, E)
     * @param value Value input tensor of shape (N, S, E) or (S, N, E)
     * @param q_proj_weight Query projection weight (E, E)
     * @param k_proj_weight Key projection weight (E, kdim)
     * @param v_proj_weight Value projection weight (E, vdim)
     * @param out_proj_weight Output projection weight (E, E)
     * @param q_proj_bias Optional query projection bias (E)
     * @param k_proj_bias Optional key projection bias (E)
     * @param v_proj_bias Optional value projection bias (E)
     * @param out_proj_bias Optional output projection bias (E)
     * @param num_heads Number of attention heads
     * @param output Output tensor
     * @param output_name Output name in ONNX graph
     */
    auto export_multihead_attention(
        const Tensor& query, const Tensor& key, const Tensor& value,
        const Tensor& q_proj_weight, const Tensor& k_proj_weight,
        const Tensor& v_proj_weight, const Tensor& out_proj_weight,
        const std::optional<Tensor>& q_proj_bias,
        const std::optional<Tensor>& k_proj_bias,
        const std::optional<Tensor>& v_proj_bias,
        const std::optional<Tensor>& out_proj_bias,
        int64_t num_heads,
        const Tensor& output,
        const std::string& output_name) -> void;

    // ============================================================================
    // Activation Functions
    // ============================================================================

    /**
     * @brief Export ReLU activation
     */
    auto export_relu(const Tensor& input, const Tensor& output,
                     const std::string& output_name) -> void;

    /**
     * @brief Export LeakyReLU activation
     */
    auto export_leaky_relu(const Tensor& input, double alpha,
                           const Tensor& output, const std::string& output_name) -> void;

    /**
     * @brief Export Sigmoid activation
     */
    auto export_sigmoid(const Tensor& input, const Tensor& output,
                        const std::string& output_name) -> void;

    /**
     * @brief Export Tanh activation
     */
    auto export_tanh(const Tensor& input, const Tensor& output,
                     const std::string& output_name) -> void;

    /**
     * @brief Export GELU activation
     */
    auto export_gelu(const Tensor& input, const Tensor& output,
                     const std::string& output_name) -> void;

    /**
     * @brief Export Softmax activation
     */
    auto export_softmax(const Tensor& input, int64_t axis,
                        const Tensor& output, const std::string& output_name) -> void;

    /**
     * @brief Export LogSoftmax activation
     */
    auto export_log_softmax(const Tensor& input, int64_t axis,
                            const Tensor& output, const std::string& output_name) -> void;

    /**
     * @brief Export ELU activation
     */
    auto export_elu(const Tensor& input, double alpha,
                    const Tensor& output, const std::string& output_name) -> void;

    /**
     * @brief Export SELU activation
     */
    auto export_selu(const Tensor& input, const Tensor& output,
                     const std::string& output_name) -> void;

    /**
     * @brief Export Swish/SiLU activation
     */
    auto export_swish(const Tensor& input, const Tensor& output,
                      const std::string& output_name) -> void;

    // ============================================================================
    // Pooling Layers
    // ============================================================================

    /**
     * @brief Export MaxPool2d layer
     */
    auto export_maxpool2d(const Tensor& input, int64_t kernel_size,
                          int64_t stride, int64_t padding,
                          const Tensor& output, const std::string& output_name) -> void;

    /**
     * @brief Export AvgPool2d layer
     */
    auto export_avgpool2d(const Tensor& input, int64_t kernel_size,
                          int64_t stride, int64_t padding,
                          const Tensor& output, const std::string& output_name) -> void;

    /**
     * @brief Export AdaptiveAvgPool2d layer
     */
    auto export_adaptive_avgpool2d(const Tensor& input,
                                   const std::vector<int64_t>& output_size,
                                   const Tensor& output, const std::string& output_name) -> void;

    // ============================================================================
    // Phase 13: Expanded Export Coverage
    // ============================================================================

    /**
     * @brief Export Cast (dtype conversion) operation
     */
    auto export_cast(const Tensor& input, DType target_dtype,
                     const Tensor& output, const std::string& output_name) -> void;

    /**
     * @brief Export Triu (upper triangular) operation via ONNX Trilu
     */
    auto export_triu(const Tensor& input, int64_t diagonal,
                     const Tensor& output, const std::string& output_name) -> void;

    /**
     * @brief Export Tril (lower triangular) operation via ONNX Trilu
     */
    auto export_tril(const Tensor& input, int64_t diagonal,
                     const Tensor& output, const std::string& output_name) -> void;

    /**
     * @brief Export LogicalAnd operation
     */
    auto export_logical_and(const Tensor& a, const Tensor& b,
                            const Tensor& output, const std::string& output_name) -> void;

    /**
     * @brief Export LogicalOr operation
     */
    auto export_logical_or(const Tensor& a, const Tensor& b,
                           const Tensor& output, const std::string& output_name) -> void;

    /**
     * @brief Export LogicalNot operation
     */
    auto export_logical_not(const Tensor& input,
                            const Tensor& output, const std::string& output_name) -> void;

    /**
     * @brief Export ScatterAdd via ONNX ScatterElements with reduction='add'
     */
    auto export_scatter_add(const Tensor& data, const Tensor& indices,
                            const Tensor& updates, int64_t axis,
                            const Tensor& output, const std::string& output_name) -> void;

    /**
     * @brief Export FFT via ONNX DFT (opset 17+)
     */
    auto export_fft(const Tensor& input, int64_t signal_ndim,
                    const Tensor& output, const std::string& output_name) -> void;

    /**
     * @brief Export IFFT via ONNX DFT with inverse=1 (opset 17+)
     */
    auto export_ifft(const Tensor& input, int64_t signal_ndim,
                     const Tensor& output, const std::string& output_name) -> void;

    /**
     * @brief Export RFFT via ONNX DFT with onesided=1 (opset 17+)
     */
    auto export_rfft(const Tensor& input, int64_t signal_ndim,
                     const Tensor& output, const std::string& output_name) -> void;

    /**
     * @brief Export CumSum via ONNX CumSum (opset 11+)
     */
    auto export_cumsum(const Tensor& input, int64_t axis,
                       const Tensor& output, const std::string& output_name) -> void;

    /**
     * @brief Export CumProd via Log + CumSum + Exp decomposition
     */
    auto export_cumprod(const Tensor& input, int64_t axis,
                        const Tensor& output, const std::string& output_name) -> void;

    /**
     * @brief Export Unfold via Slice + Reshape decomposition
     */
    auto export_unfold(const Tensor& input, int64_t dimension,
                       int64_t size, int64_t step,
                       const Tensor& output, const std::string& output_name) -> void;

    /**
     * @brief Export Fold via ONNX Col2Im (opset 18+)
     */
    auto export_fold(const Tensor& input,
                     const std::vector<int64_t>& output_size,
                     const std::vector<int64_t>& kernel_size,
                     const std::vector<int64_t>& dilation,
                     const std::vector<int64_t>& padding,
                     const std::vector<int64_t>& stride,
                     const Tensor& output, const std::string& output_name) -> void;

    /**
     * @brief Export SearchSorted as custom op in tenzor domain
     */
    auto export_search_sorted(const Tensor& sorted_sequence,
                              const Tensor& values, bool right,
                              const Tensor& output, const std::string& output_name) -> void;

    /**
     * @brief Export Roll via Slice + Concat decomposition
     */
    auto export_roll(const Tensor& input, int64_t shift, int64_t axis,
                     const Tensor& output, const std::string& output_name) -> void;

    /**
     * @brief Export EmbeddingBag via Gather + Reduce decomposition
     *
     * @param mode 0=sum, 1=mean, 2=max
     */
    auto export_embedding_bag(const Tensor& weight, const Tensor& indices,
                              const Tensor& offsets, int64_t mode,
                              const Tensor& output, const std::string& output_name) -> void;

    /**
     * @brief Export DepthwiseConv2d via ONNX Conv with group=in_channels
     */
    auto export_depthwise_conv2d(const Tensor& input, const Tensor& weight,
                                 const std::optional<Tensor>& bias,
                                 const std::vector<int64_t>& kernel_size,
                                 const std::vector<int64_t>& stride,
                                 const std::vector<int64_t>& padding,
                                 const std::vector<int64_t>& dilation,
                                 int64_t in_channels,
                                 const Tensor& output,
                                 const std::string& output_name) -> void;

    /**
     * @brief Export Log2 via Log(x) / Log(2) decomposition
     */
    auto export_log2(const Tensor& input, const Tensor& output,
                     const std::string& output_name) -> void;

    /**
     * @brief Export QuantizedConv2d via ONNX QLinearConv (opset 10+)
     */
    auto export_quantized_conv2d(
        const Tensor& input, const Tensor& input_scale, const Tensor& input_zp,
        const Tensor& weight, const Tensor& weight_scale, const Tensor& weight_zp,
        const std::optional<Tensor>& bias,
        const Tensor& output_scale, const Tensor& output_zp,
        const std::vector<int64_t>& kernel_size,
        const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding,
        const std::vector<int64_t>& dilation,
        int64_t groups,
        const Tensor& output,
        const std::string& output_name) -> void;

    // ============================================================================
    // Quantization (QDQ) Nodes
    // ============================================================================

    /**
     * @brief Export QuantizeLinear node (ONNX opset 10+).
     *
     * Produces: y = clamp(round(x / y_scale) + y_zero_point)
     *
     * @param input       Input floating-point tensor
     * @param scale       Quantization scale (scalar or per-channel)
     * @param zero_point  Quantization zero point (scalar or per-channel)
     * @param output_name Name for the quantized output
     * @param axis        Channel axis for per-channel quantization (-1 = per-tensor)
     */
    auto export_quantize_linear(const Tensor& input, const Tensor& scale,
                                 const Tensor& zero_point,
                                 const std::string& output_name,
                                 int64_t axis = -1) -> void;

    /**
     * @brief Export DequantizeLinear node (ONNX opset 10+).
     *
     * Produces: y = (x - x_zero_point) * x_scale
     *
     * @param input       Quantized input tensor (INT8/UINT8)
     * @param scale       Dequantization scale
     * @param zero_point  Dequantization zero point
     * @param output_name Name for the dequantized output
     * @param axis        Channel axis for per-channel dequantization (-1 = per-tensor)
     */
    auto export_dequantize_linear(const Tensor& input, const Tensor& scale,
                                   const Tensor& zero_point,
                                   const std::string& output_name,
                                   int64_t axis = -1) -> void;

    /**
     * @brief Export a QuantizedLinear layer as QDQ pattern.
     *
     * Emits: DequantizeLinear(weight) → Gemm → QuantizeLinear(output)
     * This is the standard QDQ representation for quantized linear layers.
     */
    auto export_quantized_linear(const nn::quantization::QuantizedLinear& layer,
                                  const Tensor& input,
                                  const std::string& output_name) -> void;

    // ============================================================================
    // JIT-based Module Export
    // ============================================================================

    /**
     * @brief Export a module to ONNX via JIT tracing.
     *
     * Traces the module's forward pass with the provided dummy input using
     * the JIT tracer, producing an IR graph. The graph is then converted
     * to ONNX format, mapping each JIT node to the corresponding ONNX
     * operator(s). Module parameters and buffers are exported as ONNX
     * initializers.
     *
     * This method is more general than `export_model()` because it traces
     * the actual computation rather than relying on structural pattern
     * matching. It correctly handles arbitrary module architectures
     * including custom forward logic, control-flow-free branches, and
     * non-standard layer compositions.
     *
     * @param module Module to export (set to eval mode internally)
     * @param dummy_input Example input tensor for shape/type tracing
     * @param output_path Path for the output .onnx file
     *
     * @code
     * auto model = std::make_shared<MyNetwork>();
     * Tensor input({1, 3, 224, 224}, DType::Float32, Device::cpu());
     * ONNXExporter exporter(13);
     * exporter.export_module(*model, input, "model.onnx");
     * @endcode
     */
    auto export_module(nn::Module& module, const Tensor& dummy_input,
                       const std::string& output_path) -> void;

    /**
     * @brief Convert a JIT IR graph to ONNX nodes in the current graph.
     *
     * Iterates over all nodes in the JIT graph and creates corresponding
     * ONNX export nodes. JIT value IDs are mapped to ONNX tensor names.
     * Constant tensor attributes (weights, biases, etc.) from JIT nodes
     * are added as ONNX initializers.
     *
     * This is the core conversion routine used by `export_module()` and
     * can also be called directly if you have a pre-built JIT graph.
     *
     * @param graph JIT IR graph to convert
     */
    auto convert_jit_graph_to_onnx(const jit::Graph& graph) -> void;

    // ============================================================================
    // Dynamic Shapes
    // ============================================================================

    /**
     * @brief Propagate dynamic shape information through the graph.
     *
     * Analyzes inputs and outputs for dimensions marked as dynamic (value -1
     * with a dim_param string) and propagates those symbolic dimension names
     * to intermediate value_info entries in the graph. This ensures that
     * ONNX runtime consumers can correctly handle variable-length dimensions.
     *
     * Should be called after all nodes, inputs, and outputs have been added
     * to the graph, and before export_to_file() or export_to_bytes().
     */
    auto propagate_dynamic_shapes() -> void;

    // ============================================================================
    // Validation
    // ============================================================================

    /**
     * @brief Result of ONNX graph validation.
     */
    struct ValidationResult {
        bool valid{true};                    ///< Overall validity
        std::vector<std::string> errors;     ///< Fatal errors preventing export
        std::vector<std::string> warnings;   ///< Non-fatal warnings
    };

    /**
     * @brief Validate the ONNX graph for correctness.
     *
     * Checks the following conditions:
     * - All node inputs reference a defined value (graph input, initializer, or
     *   another node's output)
     * - No duplicate names across inputs, outputs, initializers, and node outputs
     * - Attribute types are correct for each ONNX operator (int, float, ints, etc.)
     * - Opset compatibility: operators that require a higher opset version than
     *   configured are flagged as errors
     *
     * @return ValidationResult containing errors and warnings
     */
    auto validate() const -> ValidationResult;

    // ============================================================================
    // Export Functions
    // ============================================================================

    /**
     * @brief Export model to ONNX file
     *
     * @param filepath Output file path
     * @throws std::runtime_error if export fails
     */
    auto export_to_file(const std::string& filepath) -> void;

    /**
     * @brief Export model to ONNX bytes
     *
     * @return ONNX model as byte vector
     */
    auto export_to_bytes() -> std::vector<uint8_t>;

    /**
     * @brief Get the ONNX graph
     */
    auto get_graph() const -> const ONNXGraph&;

    /**
     * @brief Clear the exporter state
     */
    auto clear() -> void;

private:
    std::string model_name_{"tenzor_model"};        ///< Model name
    std::string producer_name_{"Tenzor"};           ///< Producer name
    std::string description_;                        ///< Model description
    int64_t opset_version_{13};                     ///< ONNX opset version
    int64_t model_version_{1};                      ///< Model version

    ONNXGraph graph_;                               ///< ONNX computational graph
    ExportContext context_;                         ///< Export context for tensor tracking

    /**
     * @brief Serialize ONNX model to protobuf bytes
     */
    auto serialize_model() -> std::vector<uint8_t>;

    /**
     * @brief Get or register tensor name in ONNX graph
     */
    auto get_tensor_name(const Tensor& tensor, const std::string& default_name) -> std::string;

    /**
     * @brief Add constant tensor to graph initializers
     */
    auto add_initializer_tensor(const Tensor& tensor, const std::string& name) -> void;

    /**
     * @brief Map a JIT OpType to an ONNX operator type string.
     *
     * @param op_type JIT operation type
     * @return ONNX operator string (e.g., "Add", "Conv", "Relu")
     */
    static auto jit_op_type_to_onnx(jit::OpType op_type) -> std::string;

    /**
     * @brief Convert a single JIT node to ONNX node(s).
     *
     * Handles attribute mapping, special-case ops (Linear -> Gemm,
     * Conv2d -> Conv, etc.), and constant tensor initializers.
     *
     * @param node JIT IR node
     * @param value_name_map Mapping from JIT value ID to ONNX tensor name
     */
    auto convert_jit_node_to_onnx(const std::shared_ptr<jit::Node>& node,
                                  std::unordered_map<std::string, std::string>& value_name_map) -> void;

    // Friend declaration for high-level export function
    friend auto export_to_onnx(std::shared_ptr<nn::Module> module,
                               const Tensor& dummy_input,
                               const std::string& filepath,
                               const std::vector<std::string>& input_names,
                               const std::vector<std::string>& output_names,
                               int64_t opset_version) -> void;
};

/**
 * @brief High-level function to export a Tenzor module to ONNX
 *
 * @param module Module to export
 * @param dummy_input Example input for shape inference
 * @param filepath Output ONNX file path
 * @param input_names Names for inputs
 * @param output_names Names for outputs
 * @param opset_version ONNX opset version
 *
 * @code
 * auto model = std::make_shared<MyModel>();
 * Tensor dummy = Tensor({1, 3, 224, 224}, DType::Float32, Device::cpu());
 * export_to_onnx(model, dummy, "model.onnx", {"input"}, {"output"});
 * @endcode
 */
auto export_to_onnx(std::shared_ptr<nn::Module> module,
                    const Tensor& dummy_input,
                    const std::string& filepath,
                    const std::vector<std::string>& input_names = {"input"},
                    const std::vector<std::string>& output_names = {"output"},
                    int64_t opset_version = 13) -> void;

} // namespace onnx
} // namespace tenzor
