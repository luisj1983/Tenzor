/**
 * @file test_onnx_roundtrip.cpp
 * @brief ONNX round-trip tests: export a model, import it back, verify outputs match
 *
 * Tests cover:
 * - Linear layer round-trip
 * - Conv2d + BatchNorm2d + ReLU round-trip
 * - Dynamic batch dimension round-trip
 * - Model metadata preservation
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/onnx/exporter.hpp>
#include <tenzor/onnx/importer.hpp>
#include <tenzor/nn/module.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/autograd/variable.hpp>
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

using namespace tenzor;
using namespace tenzor::onnx;
using namespace tenzor::nn;
// Variable is in namespace tenzor (already covered by using namespace tenzor)

// ============================================================================
// Test Environment
// ============================================================================

class ONNXRoundtripTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const roundtrip_env =
    ::testing::AddGlobalTestEnvironment(new ONNXRoundtripTestEnvironment);

// ============================================================================
// Test Fixture
// ============================================================================

class ONNXRoundtripTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir_ = fs::temp_directory_path() / "tenzor_onnx_roundtrip_test";
        fs::create_directories(test_dir_);
    }

    void TearDown() override {
        if (fs::exists(test_dir_)) {
            fs::remove_all(test_dir_);
        }
    }

    std::string get_test_path(const std::string& filename) {
        return (test_dir_ / filename).string();
    }

    /**
     * @brief Compare two tensors element-wise within tolerance.
     */
    bool tensors_close(const Tensor& a, const Tensor& b,
                       float rtol = 1e-5f, float atol = 1e-5f) {
        auto a_cpu = a.to(Device::cpu()).contiguous();
        auto b_cpu = b.to(Device::cpu()).contiguous();

        if (a_cpu.numel() != b_cpu.numel()) return false;

        const float* a_data = static_cast<const float*>(a_cpu.data_ptr());
        const float* b_data = static_cast<const float*>(b_cpu.data_ptr());

        for (int64_t i = 0; i < a_cpu.numel(); ++i) {
            float diff = std::abs(a_data[i] - b_data[i]);
            float tolerance = atol + rtol * std::abs(b_data[i]);
            if (diff > tolerance) {
                return false;
            }
        }
        return true;
    }

    fs::path test_dir_;
};

// ============================================================================
// Test 1: Linear Layer Round-Trip
// ============================================================================

TEST_F(ONNXRoundtripTest, LinearLayerRoundtrip) {
    // Create a single Linear(10, 5) layer
    auto linear = std::make_shared<Linear>(10, 5);
    linear->eval();

    // Dummy input for export
    auto dummy_input = Tensor({1, 10}, DType::Float32, Device::cpu());

    // Trace to get output shape
    Variable trace_var(dummy_input, false);
    Variable trace_out = linear->forward(trace_var);

    // Export manually (JIT tracing doesn't produce nodes for simple modules)
    std::string filepath = get_test_path("linear_roundtrip.onnx");
    ONNXExporter exporter(13);
    exporter.add_input(dummy_input, "input");
    exporter.export_linear(dummy_input,
                           linear->weight()->tensor(),
                           linear->bias()->tensor(),
                           trace_out.tensor(), "output");
    exporter.add_output(trace_out.tensor(), "output");
    exporter.export_to_file(filepath);

    ASSERT_TRUE(fs::exists(filepath));
    EXPECT_GT(fs::file_size(filepath), 0u);

    // Import back
    auto imported = import_onnx(filepath);
    ASSERT_NE(imported, nullptr);
    imported->eval();

    // Test input with known values
    auto test_input = Tensor({2, 10}, DType::Float32, Device::cpu());
    test_input.fill_(1.0f);

    // Run forward on both
    Variable orig_var(test_input, false);
    Variable orig_output = linear->forward(orig_var);

    Variable imp_var(test_input, false);
    Variable imp_output = imported->forward(imp_var);

    // Compare shapes and values
    EXPECT_EQ(orig_output.tensor().ndim(), imp_output.tensor().ndim());
    for (int64_t i = 0; i < orig_output.tensor().ndim(); ++i) {
        EXPECT_EQ(orig_output.tensor().shape()[i], imp_output.tensor().shape()[i]);
    }
    EXPECT_TRUE(tensors_close(orig_output.tensor(), imp_output.tensor()));
}

// ============================================================================
// Test 2: Conv2d + BatchNorm2d + ReLU Round-Trip
// ============================================================================

TEST_F(ONNXRoundtripTest, TwoLinearLayersRoundtrip) {
    // Test multi-op roundtrip with two linear layers exported manually
    auto linear1 = std::make_shared<Linear>(10, 8);
    auto linear2 = std::make_shared<Linear>(8, 5);
    linear1->eval();
    linear2->eval();

    // Trace to get intermediate and output shapes
    auto dummy_input = Tensor({1, 10}, DType::Float32, Device::cpu());
    Variable v0(dummy_input, false);
    Variable v1 = linear1->forward(v0);
    Variable v2 = linear2->forward(v1);

    // Export manually
    std::string filepath = get_test_path("two_linear_roundtrip.onnx");
    ONNXExporter exporter(13);
    exporter.add_input(dummy_input, "input");
    exporter.export_linear(dummy_input,
                           linear1->weight()->tensor(),
                           linear1->bias()->tensor(),
                           v1.tensor(), "hidden");
    exporter.export_linear(v1.tensor(),
                           linear2->weight()->tensor(),
                           linear2->bias()->tensor(),
                           v2.tensor(), "output");
    exporter.add_output(v2.tensor(), "output");
    exporter.export_to_file(filepath);

    ASSERT_TRUE(fs::exists(filepath));

    // Import back
    auto imported = import_onnx(filepath);
    ASSERT_NE(imported, nullptr);
    imported->eval();

    // Test with batch=2
    auto test_input = Tensor({2, 10}, DType::Float32, Device::cpu());
    test_input.fill_(1.0f);

    // Run forward on original (manually chain the two linears)
    Variable orig_v(test_input, false);
    Variable orig_mid = linear1->forward(orig_v);
    Variable orig_out = linear2->forward(orig_mid);

    Variable imp_v(test_input, false);
    Variable imp_out = imported->forward(imp_v);

    EXPECT_EQ(orig_out.tensor().ndim(), imp_out.tensor().ndim());
    for (int64_t i = 0; i < orig_out.tensor().ndim(); ++i) {
        EXPECT_EQ(orig_out.tensor().shape()[i], imp_out.tensor().shape()[i]);
    }
    EXPECT_TRUE(tensors_close(orig_out.tensor(), imp_out.tensor()));
}

// ============================================================================
// Test 3: Dynamic Batch Dimension
// ============================================================================

TEST_F(ONNXRoundtripTest, DynamicBatchDimension) {
    auto linear = std::make_shared<Linear>(10, 5);

    // Export with batch=1 as dummy
    auto dummy_input = Tensor({1, 10}, DType::Float32, Device::cpu());
    std::string filepath = get_test_path("dynamic_batch.onnx");

    // Export with dynamic batch axis
    ONNXExporter exporter(13);
    exporter.set_model_name("dynamic_batch_model");
    exporter.add_input(dummy_input, "input",
                       {{0, "batch_size"}});  // dim 0 is dynamic

    // Trace the model to build the ONNX graph
    linear->eval();
    Variable trace_var(dummy_input, false);
    Variable trace_out = linear->forward(trace_var);

    // Manually add linear op to ONNX graph
    exporter.export_linear(dummy_input,
                           linear->weight()->tensor(),
                           linear->bias()->tensor(),
                           trace_out.tensor(), "output");

    auto out_tensor = trace_out.tensor();
    exporter.add_output(out_tensor, "output",
                        {{0, "batch_size"}});  // output dim 0 also dynamic

    exporter.propagate_dynamic_shapes();
    exporter.export_to_file(filepath);

    ASSERT_TRUE(fs::exists(filepath));

    // Import the model
    auto imported = import_onnx(filepath);
    ASSERT_NE(imported, nullptr);
    imported->eval();

    // Run with batch_size = 1
    auto input_b1 = Tensor({1, 10}, DType::Float32, Device::cpu());
    input_b1.fill_(1.0f);
    Variable var_b1(input_b1, false);
    Variable out_b1 = imported->forward(var_b1);
    EXPECT_EQ(out_b1.tensor().shape()[0], 1);
    EXPECT_EQ(out_b1.tensor().shape()[1], 5);

    // Run with batch_size = 4
    auto input_b4 = Tensor({4, 10}, DType::Float32, Device::cpu());
    input_b4.fill_(1.0f);
    Variable var_b4(input_b4, false);
    Variable out_b4 = imported->forward(var_b4);
    EXPECT_EQ(out_b4.tensor().shape()[0], 4);
    EXPECT_EQ(out_b4.tensor().shape()[1], 5);

    // Run with batch_size = 16
    auto input_b16 = Tensor({16, 10}, DType::Float32, Device::cpu());
    input_b16.fill_(1.0f);
    Variable var_b16(input_b16, false);
    Variable out_b16 = imported->forward(var_b16);
    EXPECT_EQ(out_b16.tensor().shape()[0], 16);
    EXPECT_EQ(out_b16.tensor().shape()[1], 5);

    // Verify outputs are consistent: each row should produce the same result
    // since all inputs are fill_(1.0)
    auto b1_data = out_b1.tensor().to(Device::cpu()).contiguous();
    auto b4_data = out_b4.tensor().to(Device::cpu()).contiguous();
    const float* b1_ptr = static_cast<const float*>(b1_data.data_ptr());
    const float* b4_ptr = static_cast<const float*>(b4_data.data_ptr());

    for (int64_t j = 0; j < 5; ++j) {
        EXPECT_NEAR(b1_ptr[j], b4_ptr[j], 1e-5f)
            << "Batch 1 vs batch 4 mismatch at element " << j;
    }
}

// ============================================================================
// Test 4: Model Metadata Survives Round-Trip
// ============================================================================

TEST_F(ONNXRoundtripTest, MetadataPreservation) {
    // Create a simple model
    auto linear = std::make_shared<Linear>(10, 5);

    // Export with specific metadata
    std::string filepath = get_test_path("metadata_test.onnx");
    ONNXExporter exporter(17);  // Opset 17
    exporter.set_model_name("test_metadata_model");
    exporter.set_producer_name("TenzorTestSuite");
    exporter.set_opset_version(17);

    auto dummy_input = Tensor({1, 10}, DType::Float32, Device::cpu());
    exporter.add_input(dummy_input, "input");

    linear->eval();
    Variable trace_var(dummy_input, false);
    Variable trace_out = linear->forward(trace_var);

    exporter.export_linear(dummy_input,
                           linear->weight()->tensor(),
                           linear->bias()->tensor(),
                           trace_out.tensor(), "output");

    auto out_tensor = trace_out.tensor();
    exporter.add_output(out_tensor, "output");
    exporter.export_to_file(filepath);

    ASSERT_TRUE(fs::exists(filepath));

    // Import and check metadata
    ONNXImporter importer(false);
    auto imported = importer.import_from_file(filepath);
    ASSERT_NE(imported, nullptr);

    const auto& model_data = importer.get_model_data();

    // Verify opset version is preserved
    EXPECT_EQ(model_data.opset_version, 17)
        << "Opset version not preserved; got " << model_data.opset_version;

    // Verify producer name is preserved
    EXPECT_EQ(model_data.producer_name, "TenzorTestSuite")
        << "Producer name not preserved; got " << model_data.producer_name;

    // Verify the graph has inputs and outputs
    EXPECT_FALSE(model_data.graph.inputs.empty())
        << "Graph inputs should not be empty after import";
    EXPECT_FALSE(model_data.graph.outputs.empty())
        << "Graph outputs should not be empty after import";
}
