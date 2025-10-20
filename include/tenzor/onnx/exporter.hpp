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

namespace tenzor {
namespace onnx {

// Forward declarations
class ONNXGraph;
class ONNXNode;
class ONNXValueInfo;
class ONNXTensor;

/**
 * @brief ONNX data type enumeration mapping to Tenzor DType
 */
enum class ONNXDataType : int32_t {
    UNDEFINED = 0,
    FLOAT = 1,      // float32
    UINT8 = 2,      // uint8
    INT8 = 3,       // int8
    UINT16 = 4,     // uint16
    INT16 = 5,      // int16
    INT32 = 6,      // int32
    INT64 = 7,      // int64
    STRING = 8,     // string
    BOOL = 9,       // bool
    FLOAT16 = 10,   // float16
    DOUBLE = 11,    // float64
    UINT32 = 12,    // uint32
    UINT64 = 13,    // uint64
    COMPLEX64 = 14, // complex64
    COMPLEX128 = 15,// complex128
    BFLOAT16 = 16   // bfloat16
};

/**
 * @brief Convert Tenzor DType to ONNX data type
 *
 * @param dtype Tenzor data type
 * @return Corresponding ONNX data type
 */
auto dtype_to_onnx(DType dtype) -> ONNXDataType;

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
 * @brief ONNX value info (input/output/intermediate value metadata)
 */
class ONNXValueInfo {
public:
    std::string name;                    ///< Value name
    ONNXDataType dtype;                  ///< Data type
    std::vector<int64_t> shape;          ///< Shape (-1 for dynamic dimensions)

    ONNXValueInfo() = default;
    ONNXValueInfo(const std::string& name, ONNXDataType dtype,
                  const std::vector<int64_t>& shape);
};

/**
 * @brief ONNX graph node (operation)
 */
class ONNXNode {
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

    ONNXNode(const std::string& op_type, const std::string& name);

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
    std::vector<ONNXNode> nodes;                ///< Graph nodes (operations)
    std::vector<ONNXValueInfo> inputs;          ///< Graph inputs
    std::vector<ONNXValueInfo> outputs;         ///< Graph outputs
    std::vector<ONNXTensor> initializers;       ///< Constant tensors (weights, biases)
    std::unordered_map<std::string, ONNXValueInfo> value_info; ///< Intermediate values

    explicit ONNXGraph(const std::string& name = "graph");

    /**
     * @brief Add node to graph
     */
    auto add_node(const ONNXNode& node) -> void;

    /**
     * @brief Add input to graph
     */
    auto add_input(const ONNXValueInfo& input) -> void;

    /**
     * @brief Add output to graph
     */
    auto add_output(const ONNXValueInfo& output) -> void;

    /**
     * @brief Add initializer tensor
     */
    auto add_initializer(const ONNXTensor& tensor) -> void;

    /**
     * @brief Register intermediate value info
     */
    auto add_value_info(const ONNXValueInfo& info) -> void;

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
     */
    auto add_output(const Tensor& tensor, const std::string& name) -> void;

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
