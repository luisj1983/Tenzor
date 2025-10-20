/**
 * @file test_onnx_export.cpp
 * @brief Comprehensive tests for ONNX export functionality
 */

#include <gtest/gtest.h>
#include "../../include/tenzor/tenzor.hpp"
#include "../../include/tenzor/onnx/exporter.hpp"
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

using namespace tenzor;
using namespace tenzor::onnx;

class ONNXExportTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const onnx_env =
    ::testing::AddGlobalTestEnvironment(new ONNXExportTestEnvironment);

class ONNXExportTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temporary directory for test outputs
        test_dir_ = fs::temp_directory_path() / "tenzor_onnx_tests";
        fs::create_directories(test_dir_);
    }

    void TearDown() override {
        // Clean up test files
        if (fs::exists(test_dir_)) {
            fs::remove_all(test_dir_);
        }
    }

    std::string get_test_path(const std::string& filename) {
        return (test_dir_ / filename).string();
    }

    // Helper to verify ONNX file was created and has content
    bool verify_onnx_file(const std::string& filepath) {
        if (!fs::exists(filepath)) {
            return false;
        }
        auto size = fs::file_size(filepath);
        return size > 0;
    }

    fs::path test_dir_;
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_F(ONNXExportTest, ExporterConstruction) {
    ONNXExporter exporter;
    EXPECT_NO_THROW(exporter.set_model_name("test_model"));
    EXPECT_NO_THROW(exporter.set_opset_version(13));
}

TEST_F(ONNXExportTest, DTypeConversion) {
    EXPECT_EQ(dtype_to_onnx(DType::Float32), ONNXDataType::FLOAT);
    EXPECT_EQ(dtype_to_onnx(DType::Float64), ONNXDataType::DOUBLE);
    EXPECT_EQ(dtype_to_onnx(DType::Float16), ONNXDataType::FLOAT16);
    EXPECT_EQ(dtype_to_onnx(DType::BFloat16), ONNXDataType::BFLOAT16);
    EXPECT_EQ(dtype_to_onnx(DType::Int8), ONNXDataType::INT8);
    EXPECT_EQ(dtype_to_onnx(DType::Int16), ONNXDataType::INT16);
    EXPECT_EQ(dtype_to_onnx(DType::Int32), ONNXDataType::INT32);
    EXPECT_EQ(dtype_to_onnx(DType::Int64), ONNXDataType::INT64);
    EXPECT_EQ(dtype_to_onnx(DType::UInt8), ONNXDataType::UINT8);
    EXPECT_EQ(dtype_to_onnx(DType::Bool), ONNXDataType::BOOL);
}

TEST_F(ONNXExportTest, TensorConversion) {
    Tensor t({2, 3}, DType::Float32, Device::cpu());
    t.fill_(1.5f);

    ONNXTensor onnx_t(t, "test_tensor");

    EXPECT_EQ(onnx_t.name, "test_tensor");
    EXPECT_EQ(onnx_t.dtype, ONNXDataType::FLOAT);
    ASSERT_EQ(onnx_t.dims.size(), 2);
    EXPECT_EQ(onnx_t.dims[0], 2);
    EXPECT_EQ(onnx_t.dims[1], 3);
    EXPECT_EQ(onnx_t.numel(), 6);
    EXPECT_EQ(onnx_t.size_bytes(), 24); // 6 * 4 bytes
}

TEST_F(ONNXExportTest, GraphConstruction) {
    ONNXGraph graph("test_graph");

    EXPECT_EQ(graph.name, "test_graph");
    EXPECT_TRUE(graph.nodes.empty());
    EXPECT_TRUE(graph.inputs.empty());
    EXPECT_TRUE(graph.outputs.empty());
}

TEST_F(ONNXExportTest, NodeConstruction) {
    ONNXNode node("Add", "add_node");

    EXPECT_EQ(node.op_type, "Add");
    EXPECT_EQ(node.name, "add_node");

    node.add_input("input1");
    node.add_input("input2");
    node.add_output("output");

    ASSERT_EQ(node.inputs.size(), 2);
    EXPECT_EQ(node.inputs[0], "input1");
    EXPECT_EQ(node.inputs[1], "input2");
    ASSERT_EQ(node.outputs.size(), 1);
    EXPECT_EQ(node.outputs[0], "output");
}

TEST_F(ONNXExportTest, NodeAttributes) {
    ONNXNode node("Conv", "conv_node");

    node.set_attr("stride", static_cast<int64_t>(2));
    node.set_attr("alpha", 0.01f);
    node.set_attr("mode", std::string("same"));
    node.set_attr("kernel_shape", std::vector<int64_t>{3, 3});

    EXPECT_EQ(node.int_attrs["stride"], 2);
    EXPECT_FLOAT_EQ(node.float_attrs["alpha"], 0.01f);
    EXPECT_EQ(node.string_attrs["mode"], "same");
    ASSERT_EQ(node.ints_attrs["kernel_shape"].size(), 2);
    EXPECT_EQ(node.ints_attrs["kernel_shape"][0], 3);
    EXPECT_EQ(node.ints_attrs["kernel_shape"][1], 3);
}

// ============================================================================
// Tensor Operation Export Tests
// ============================================================================

TEST_F(ONNXExportTest, ExportAdd) {
    ONNXExporter exporter;
    exporter.set_model_name("add_model");

    Tensor a({2, 3}, DType::Float32, Device::cpu());
    Tensor b({2, 3}, DType::Float32, Device::cpu());
    Tensor output({2, 3}, DType::Float32, Device::cpu());

    a.fill_(1.0f);
    b.fill_(2.0f);

    exporter.add_input(a, "input_a");
    exporter.add_input(b, "input_b");
    exporter.export_add(a, b, output, "add_output");
    exporter.add_output(output, "add_output");

    std::string filepath = get_test_path("add_model.onnx");
    EXPECT_NO_THROW(exporter.export_to_file(filepath));
    EXPECT_TRUE(verify_onnx_file(filepath));

    const auto& graph = exporter.get_graph();
    ASSERT_EQ(graph.nodes.size(), 1);
    EXPECT_EQ(graph.nodes[0].op_type, "Add");
}

TEST_F(ONNXExportTest, ExportSub) {
    ONNXExporter exporter;

    Tensor a({2, 3}, DType::Float32, Device::cpu());
    Tensor b({2, 3}, DType::Float32, Device::cpu());
    Tensor output({2, 3}, DType::Float32, Device::cpu());

    exporter.add_input(a, "input_a");
    exporter.add_input(b, "input_b");
    exporter.export_sub(a, b, output, "sub_output");
    exporter.add_output(output, "sub_output");

    const auto& graph = exporter.get_graph();
    ASSERT_EQ(graph.nodes.size(), 1);
    EXPECT_EQ(graph.nodes[0].op_type, "Sub");
}

TEST_F(ONNXExportTest, ExportMul) {
    ONNXExporter exporter;

    Tensor a({2, 3}, DType::Float32, Device::cpu());
    Tensor b({2, 3}, DType::Float32, Device::cpu());
    Tensor output({2, 3}, DType::Float32, Device::cpu());

    exporter.add_input(a, "input_a");
    exporter.add_input(b, "input_b");
    exporter.export_mul(a, b, output, "mul_output");
    exporter.add_output(output, "mul_output");

    const auto& graph = exporter.get_graph();
    ASSERT_EQ(graph.nodes.size(), 1);
    EXPECT_EQ(graph.nodes[0].op_type, "Mul");
}

TEST_F(ONNXExportTest, ExportDiv) {
    ONNXExporter exporter;

    Tensor a({2, 3}, DType::Float32, Device::cpu());
    Tensor b({2, 3}, DType::Float32, Device::cpu());
    Tensor output({2, 3}, DType::Float32, Device::cpu());

    exporter.add_input(a, "input_a");
    exporter.add_input(b, "input_b");
    exporter.export_div(a, b, output, "div_output");
    exporter.add_output(output, "div_output");

    const auto& graph = exporter.get_graph();
    ASSERT_EQ(graph.nodes.size(), 1);
    EXPECT_EQ(graph.nodes[0].op_type, "Div");
}

TEST_F(ONNXExportTest, ExportMatMul) {
    ONNXExporter exporter;

    Tensor a({2, 3}, DType::Float32, Device::cpu());
    Tensor b({3, 4}, DType::Float32, Device::cpu());
    Tensor output({2, 4}, DType::Float32, Device::cpu());

    exporter.add_input(a, "input_a");
    exporter.add_input(b, "input_b");
    exporter.export_matmul(a, b, output, "matmul_output");
    exporter.add_output(output, "matmul_output");

    const auto& graph = exporter.get_graph();
    ASSERT_EQ(graph.nodes.size(), 1);
    EXPECT_EQ(graph.nodes[0].op_type, "MatMul");
}

TEST_F(ONNXExportTest, ExportReshape) {
    ONNXExporter exporter;

    Tensor input({2, 3, 4}, DType::Float32, Device::cpu());
    Tensor output({6, 4}, DType::Float32, Device::cpu());

    exporter.add_input(input, "input");
    exporter.export_reshape(input, {6, 4}, output, "reshape_output");
    exporter.add_output(output, "reshape_output");

    const auto& graph = exporter.get_graph();
    ASSERT_EQ(graph.nodes.size(), 1);
    EXPECT_EQ(graph.nodes[0].op_type, "Reshape");
}

TEST_F(ONNXExportTest, ExportTranspose) {
    ONNXExporter exporter;

    Tensor input({2, 3, 4}, DType::Float32, Device::cpu());
    Tensor output({4, 3, 2}, DType::Float32, Device::cpu());

    exporter.add_input(input, "input");
    exporter.export_transpose(input, {2, 1, 0}, output, "transpose_output");
    exporter.add_output(output, "transpose_output");

    const auto& graph = exporter.get_graph();
    ASSERT_EQ(graph.nodes.size(), 1);
    EXPECT_EQ(graph.nodes[0].op_type, "Transpose");
    ASSERT_EQ(graph.nodes[0].ints_attrs.count("perm"), 1);
}

TEST_F(ONNXExportTest, ExportConcat) {
    ONNXExporter exporter;

    Tensor input1({2, 3}, DType::Float32, Device::cpu());
    Tensor input2({2, 3}, DType::Float32, Device::cpu());
    Tensor output({4, 3}, DType::Float32, Device::cpu());

    exporter.add_input(input1, "input1");
    exporter.add_input(input2, "input2");
    exporter.export_concat({input1, input2}, 0, output, "concat_output");
    exporter.add_output(output, "concat_output");

    const auto& graph = exporter.get_graph();
    ASSERT_EQ(graph.nodes.size(), 1);
    EXPECT_EQ(graph.nodes[0].op_type, "Concat");
    EXPECT_EQ(graph.nodes[0].int_attrs.at("axis"), 0);
}

// ============================================================================
// Neural Network Layer Export Tests
// ============================================================================

TEST_F(ONNXExportTest, ExportLinear) {
    ONNXExporter exporter;

    Tensor input({2, 10}, DType::Float32, Device::cpu());
    Tensor weight({5, 10}, DType::Float32, Device::cpu());
    Tensor bias({5}, DType::Float32, Device::cpu());
    Tensor output({2, 5}, DType::Float32, Device::cpu());

    weight.fill_(0.1f);
    bias.fill_(0.01f);

    exporter.add_input(input, "input");
    exporter.export_linear(input, weight, bias, output, "linear_output");
    exporter.add_output(output, "linear_output");

    std::string filepath = get_test_path("linear_model.onnx");
    EXPECT_NO_THROW(exporter.export_to_file(filepath));
    EXPECT_TRUE(verify_onnx_file(filepath));

    const auto& graph = exporter.get_graph();
    ASSERT_EQ(graph.nodes.size(), 1);
    EXPECT_EQ(graph.nodes[0].op_type, "Gemm");
    EXPECT_EQ(graph.initializers.size(), 2); // weight and bias
}

TEST_F(ONNXExportTest, ExportLinearNoBias) {
    ONNXExporter exporter;

    Tensor input({2, 10}, DType::Float32, Device::cpu());
    Tensor weight({5, 10}, DType::Float32, Device::cpu());
    Tensor output({2, 5}, DType::Float32, Device::cpu());

    exporter.add_input(input, "input");
    exporter.export_linear(input, weight, std::nullopt, output, "linear_output");
    exporter.add_output(output, "linear_output");

    const auto& graph = exporter.get_graph();
    ASSERT_EQ(graph.nodes.size(), 1);
    EXPECT_EQ(graph.nodes[0].op_type, "Gemm");
    EXPECT_EQ(graph.initializers.size(), 1); // weight only
}

TEST_F(ONNXExportTest, ExportConv2d) {
    ONNXExporter exporter;

    Tensor input({1, 3, 32, 32}, DType::Float32, Device::cpu());
    Tensor weight({64, 3, 3, 3}, DType::Float32, Device::cpu());
    Tensor bias({64}, DType::Float32, Device::cpu());
    Tensor output({1, 64, 32, 32}, DType::Float32, Device::cpu());

    exporter.add_input(input, "input");
    exporter.export_conv2d(input, weight, bias, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1,
                          output, "conv_output");
    exporter.add_output(output, "conv_output");

    std::string filepath = get_test_path("conv2d_model.onnx");
    EXPECT_NO_THROW(exporter.export_to_file(filepath));
    EXPECT_TRUE(verify_onnx_file(filepath));

    const auto& graph = exporter.get_graph();
    ASSERT_EQ(graph.nodes.size(), 1);
    EXPECT_EQ(graph.nodes[0].op_type, "Conv");
    ASSERT_TRUE(graph.nodes[0].ints_attrs.count("kernel_shape"));
    ASSERT_TRUE(graph.nodes[0].ints_attrs.count("strides"));
    ASSERT_TRUE(graph.nodes[0].ints_attrs.count("pads"));
}

TEST_F(ONNXExportTest, ExportConv1d) {
    ONNXExporter exporter;

    Tensor input({1, 16, 100}, DType::Float32, Device::cpu());
    Tensor weight({32, 16, 3}, DType::Float32, Device::cpu());
    Tensor bias({32}, DType::Float32, Device::cpu());
    Tensor output({1, 32, 100}, DType::Float32, Device::cpu());

    exporter.add_input(input, "input");
    exporter.export_conv1d(input, weight, bias, 3, 1, 1, 1, 1, output, "conv1d_output");
    exporter.add_output(output, "conv1d_output");

    const auto& graph = exporter.get_graph();
    ASSERT_EQ(graph.nodes.size(), 1);
    EXPECT_EQ(graph.nodes[0].op_type, "Conv");
}

TEST_F(ONNXExportTest, ExportBatchNorm2d) {
    ONNXExporter exporter;

    Tensor input({1, 64, 32, 32}, DType::Float32, Device::cpu());
    Tensor scale({64}, DType::Float32, Device::cpu());
    Tensor bias({64}, DType::Float32, Device::cpu());
    Tensor mean({64}, DType::Float32, Device::cpu());
    Tensor var({64}, DType::Float32, Device::cpu());
    Tensor output({1, 64, 32, 32}, DType::Float32, Device::cpu());

    scale.fill_(1.0f);
    bias.fill_(0.0f);
    mean.fill_(0.0f);
    var.fill_(1.0f);

    exporter.add_input(input, "input");
    exporter.export_batchnorm2d(input, scale, bias, mean, var, 1e-5, output, "bn_output");
    exporter.add_output(output, "bn_output");

    std::string filepath = get_test_path("batchnorm2d_model.onnx");
    EXPECT_NO_THROW(exporter.export_to_file(filepath));
    EXPECT_TRUE(verify_onnx_file(filepath));

    const auto& graph = exporter.get_graph();
    ASSERT_EQ(graph.nodes.size(), 1);
    EXPECT_EQ(graph.nodes[0].op_type, "BatchNormalization");
    EXPECT_EQ(graph.initializers.size(), 4); // scale, bias, mean, var
}

// ============================================================================
// Activation Function Export Tests
// ============================================================================

TEST_F(ONNXExportTest, ExportReLU) {
    ONNXExporter exporter;

    Tensor input({2, 10}, DType::Float32, Device::cpu());
    Tensor output({2, 10}, DType::Float32, Device::cpu());

    exporter.add_input(input, "input");
    exporter.export_relu(input, output, "relu_output");
    exporter.add_output(output, "relu_output");

    const auto& graph = exporter.get_graph();
    ASSERT_EQ(graph.nodes.size(), 1);
    EXPECT_EQ(graph.nodes[0].op_type, "Relu");
}

TEST_F(ONNXExportTest, ExportLeakyReLU) {
    ONNXExporter exporter;

    Tensor input({2, 10}, DType::Float32, Device::cpu());
    Tensor output({2, 10}, DType::Float32, Device::cpu());

    exporter.add_input(input, "input");
    exporter.export_leaky_relu(input, 0.01, output, "leaky_relu_output");
    exporter.add_output(output, "leaky_relu_output");

    const auto& graph = exporter.get_graph();
    ASSERT_EQ(graph.nodes.size(), 1);
    EXPECT_EQ(graph.nodes[0].op_type, "LeakyRelu");
    EXPECT_FLOAT_EQ(graph.nodes[0].float_attrs.at("alpha"), 0.01f);
}

TEST_F(ONNXExportTest, ExportSigmoid) {
    ONNXExporter exporter;

    Tensor input({2, 10}, DType::Float32, Device::cpu());
    Tensor output({2, 10}, DType::Float32, Device::cpu());

    exporter.add_input(input, "input");
    exporter.export_sigmoid(input, output, "sigmoid_output");
    exporter.add_output(output, "sigmoid_output");

    const auto& graph = exporter.get_graph();
    ASSERT_EQ(graph.nodes.size(), 1);
    EXPECT_EQ(graph.nodes[0].op_type, "Sigmoid");
}

TEST_F(ONNXExportTest, ExportTanh) {
    ONNXExporter exporter;

    Tensor input({2, 10}, DType::Float32, Device::cpu());
    Tensor output({2, 10}, DType::Float32, Device::cpu());

    exporter.add_input(input, "input");
    exporter.export_tanh(input, output, "tanh_output");
    exporter.add_output(output, "tanh_output");

    const auto& graph = exporter.get_graph();
    ASSERT_EQ(graph.nodes.size(), 1);
    EXPECT_EQ(graph.nodes[0].op_type, "Tanh");
}

TEST_F(ONNXExportTest, ExportGELU) {
    ONNXExporter exporter(13); // Opset 13 - will use decomposed version

    Tensor input({2, 10}, DType::Float32, Device::cpu());
    Tensor output({2, 10}, DType::Float32, Device::cpu());

    exporter.add_input(input, "input");
    exporter.export_gelu(input, output, "gelu_output");
    exporter.add_output(output, "gelu_output");

    const auto& graph = exporter.get_graph();
    // GELU is decomposed into multiple ops for opset < 20
    EXPECT_GT(graph.nodes.size(), 1);
}

TEST_F(ONNXExportTest, ExportSoftmax) {
    ONNXExporter exporter;

    Tensor input({2, 10}, DType::Float32, Device::cpu());
    Tensor output({2, 10}, DType::Float32, Device::cpu());

    exporter.add_input(input, "input");
    exporter.export_softmax(input, -1, output, "softmax_output");
    exporter.add_output(output, "softmax_output");

    const auto& graph = exporter.get_graph();
    ASSERT_EQ(graph.nodes.size(), 1);
    EXPECT_EQ(graph.nodes[0].op_type, "Softmax");
    EXPECT_EQ(graph.nodes[0].int_attrs.at("axis"), -1);
}

TEST_F(ONNXExportTest, ExportLogSoftmax) {
    ONNXExporter exporter;

    Tensor input({2, 10}, DType::Float32, Device::cpu());
    Tensor output({2, 10}, DType::Float32, Device::cpu());

    exporter.add_input(input, "input");
    exporter.export_log_softmax(input, -1, output, "log_softmax_output");
    exporter.add_output(output, "log_softmax_output");

    const auto& graph = exporter.get_graph();
    ASSERT_EQ(graph.nodes.size(), 1);
    EXPECT_EQ(graph.nodes[0].op_type, "LogSoftmax");
}

TEST_F(ONNXExportTest, ExportELU) {
    ONNXExporter exporter;

    Tensor input({2, 10}, DType::Float32, Device::cpu());
    Tensor output({2, 10}, DType::Float32, Device::cpu());

    exporter.add_input(input, "input");
    exporter.export_elu(input, 1.0, output, "elu_output");
    exporter.add_output(output, "elu_output");

    const auto& graph = exporter.get_graph();
    ASSERT_EQ(graph.nodes.size(), 1);
    EXPECT_EQ(graph.nodes[0].op_type, "Elu");
    EXPECT_FLOAT_EQ(graph.nodes[0].float_attrs.at("alpha"), 1.0f);
}

TEST_F(ONNXExportTest, ExportSELU) {
    ONNXExporter exporter;

    Tensor input({2, 10}, DType::Float32, Device::cpu());
    Tensor output({2, 10}, DType::Float32, Device::cpu());

    exporter.add_input(input, "input");
    exporter.export_selu(input, output, "selu_output");
    exporter.add_output(output, "selu_output");

    const auto& graph = exporter.get_graph();
    ASSERT_EQ(graph.nodes.size(), 1);
    EXPECT_EQ(graph.nodes[0].op_type, "Selu");
}

TEST_F(ONNXExportTest, ExportSwish) {
    ONNXExporter exporter;

    Tensor input({2, 10}, DType::Float32, Device::cpu());
    Tensor output({2, 10}, DType::Float32, Device::cpu());

    exporter.add_input(input, "input");
    exporter.export_swish(input, output, "swish_output");
    exporter.add_output(output, "swish_output");

    const auto& graph = exporter.get_graph();
    // Swish is decomposed into Sigmoid + Mul
    ASSERT_EQ(graph.nodes.size(), 2);
}

// ============================================================================
// Pooling Layer Export Tests
// ============================================================================

TEST_F(ONNXExportTest, ExportMaxPool2d) {
    ONNXExporter exporter;

    Tensor input({1, 64, 32, 32}, DType::Float32, Device::cpu());
    Tensor output({1, 64, 16, 16}, DType::Float32, Device::cpu());

    exporter.add_input(input, "input");
    exporter.export_maxpool2d(input, 2, 2, 0, output, "maxpool_output");
    exporter.add_output(output, "maxpool_output");

    std::string filepath = get_test_path("maxpool2d_model.onnx");
    EXPECT_NO_THROW(exporter.export_to_file(filepath));
    EXPECT_TRUE(verify_onnx_file(filepath));

    const auto& graph = exporter.get_graph();
    ASSERT_EQ(graph.nodes.size(), 1);
    EXPECT_EQ(graph.nodes[0].op_type, "MaxPool");
}

TEST_F(ONNXExportTest, ExportAvgPool2d) {
    ONNXExporter exporter;

    Tensor input({1, 64, 32, 32}, DType::Float32, Device::cpu());
    Tensor output({1, 64, 16, 16}, DType::Float32, Device::cpu());

    exporter.add_input(input, "input");
    exporter.export_avgpool2d(input, 2, 2, 0, output, "avgpool_output");
    exporter.add_output(output, "avgpool_output");

    std::string filepath = get_test_path("avgpool2d_model.onnx");
    EXPECT_NO_THROW(exporter.export_to_file(filepath));
    EXPECT_TRUE(verify_onnx_file(filepath));

    const auto& graph = exporter.get_graph();
    ASSERT_EQ(graph.nodes.size(), 1);
    EXPECT_EQ(graph.nodes[0].op_type, "AveragePool");
}

TEST_F(ONNXExportTest, ExportAdaptiveAvgPool2dGlobal) {
    ONNXExporter exporter;

    Tensor input({1, 512, 7, 7}, DType::Float32, Device::cpu());
    Tensor output({1, 512, 1, 1}, DType::Float32, Device::cpu());

    exporter.add_input(input, "input");
    exporter.export_adaptive_avgpool2d(input, {1, 1}, output, "adaptive_avgpool_output");
    exporter.add_output(output, "adaptive_avgpool_output");

    const auto& graph = exporter.get_graph();
    ASSERT_EQ(graph.nodes.size(), 1);
    EXPECT_EQ(graph.nodes[0].op_type, "GlobalAveragePool");
}

// ============================================================================
// Complex Model Export Tests
// ============================================================================

TEST_F(ONNXExportTest, ExportComplexModel) {
    ONNXExporter exporter;
    exporter.set_model_name("complex_model");
    exporter.set_opset_version(13);

    // Build a simple CNN-like model: Conv -> ReLU -> MaxPool -> Linear

    // Conv layer
    Tensor input({1, 3, 32, 32}, DType::Float32, Device::cpu());
    Tensor conv_weight({16, 3, 3, 3}, DType::Float32, Device::cpu());
    Tensor conv_bias({16}, DType::Float32, Device::cpu());
    Tensor conv_output({1, 16, 32, 32}, DType::Float32, Device::cpu());

    exporter.add_input(input, "input");
    exporter.export_conv2d(input, conv_weight, conv_bias, {3, 3}, {1, 1}, {1, 1},
                          {1, 1}, 1, conv_output, "conv_output");

    // ReLU
    Tensor relu_output({1, 16, 32, 32}, DType::Float32, Device::cpu());
    exporter.export_relu(conv_output, relu_output, "relu_output");

    // MaxPool
    Tensor pool_output({1, 16, 16, 16}, DType::Float32, Device::cpu());
    exporter.export_maxpool2d(relu_output, 2, 2, 0, pool_output, "pool_output");

    // Reshape for linear
    Tensor reshape_output({1, 4096}, DType::Float32, Device::cpu());
    exporter.export_reshape(pool_output, {1, 4096}, reshape_output, "reshape_output");

    // Linear
    Tensor linear_weight({10, 4096}, DType::Float32, Device::cpu());
    Tensor linear_bias({10}, DType::Float32, Device::cpu());
    Tensor linear_output({1, 10}, DType::Float32, Device::cpu());
    exporter.export_linear(reshape_output, linear_weight, linear_bias,
                          linear_output, "linear_output");

    // Softmax
    Tensor softmax_output({1, 10}, DType::Float32, Device::cpu());
    exporter.export_softmax(linear_output, -1, softmax_output, "output");

    exporter.add_output(softmax_output, "output");

    std::string filepath = get_test_path("complex_model.onnx");
    EXPECT_NO_THROW(exporter.export_to_file(filepath));
    EXPECT_TRUE(verify_onnx_file(filepath));

    const auto& graph = exporter.get_graph();
    EXPECT_EQ(graph.nodes.size(), 6); // Conv, ReLU, MaxPool, Reshape, Linear, Softmax
    EXPECT_EQ(graph.inputs.size(), 1);
    EXPECT_EQ(graph.outputs.size(), 1);
}

TEST_F(ONNXExportTest, ExportWithDynamicDimensions) {
    ONNXExporter exporter;

    Tensor input({1, 3, 224, 224}, DType::Float32, Device::cpu());

    // Add input with dynamic batch dimension
    std::unordered_map<int64_t, std::string> dynamic_axes;
    dynamic_axes[0] = "batch";

    exporter.add_input(input, "input", dynamic_axes);

    const auto& graph = exporter.get_graph();
    ASSERT_EQ(graph.inputs.size(), 1);
    ASSERT_EQ(graph.inputs[0].shape.size(), 4);
    EXPECT_EQ(graph.inputs[0].shape[0], -1); // Dynamic dimension
    EXPECT_EQ(graph.inputs[0].shape[1], 3);
    EXPECT_EQ(graph.inputs[0].shape[2], 224);
    EXPECT_EQ(graph.inputs[0].shape[3], 224);
}

TEST_F(ONNXExportTest, ExportMultipleDataTypes) {
    ONNXExporter exporter;

    // Float32
    Tensor float_input({2, 3}, DType::Float32, Device::cpu());
    exporter.add_input(float_input, "float_input");

    // Int64
    Tensor int_input({2, 3}, DType::Int64, Device::cpu());
    exporter.add_input(int_input, "int_input");

    const auto& graph = exporter.get_graph();
    ASSERT_EQ(graph.inputs.size(), 2);
    EXPECT_EQ(graph.inputs[0].dtype, ONNXDataType::FLOAT);
    EXPECT_EQ(graph.inputs[1].dtype, ONNXDataType::INT64);
}

TEST_F(ONNXExportTest, ExporterClear) {
    ONNXExporter exporter;

    Tensor input({2, 3}, DType::Float32, Device::cpu());
    exporter.add_input(input, "input");

    EXPECT_EQ(exporter.get_graph().inputs.size(), 1);

    exporter.clear();

    EXPECT_EQ(exporter.get_graph().inputs.size(), 0);
    EXPECT_EQ(exporter.get_graph().nodes.size(), 0);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(ONNXExportTest, ExportToInvalidPath) {
    ONNXExporter exporter;

    Tensor input({2, 3}, DType::Float32, Device::cpu());
    exporter.add_input(input, "input");
    exporter.add_output(input, "output");

    // Try to export to invalid path
    EXPECT_THROW(exporter.export_to_file("/invalid/path/model.onnx"), std::runtime_error);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
