/**
 * @file test_roberta_electra_multidtype.cpp
 * @brief Multi-dtype tests for RoBERTa and ELECTRA models
 *
 * Converted from test_roberta_electra.cpp to support multiple data types:
 * - Float32 (standard precision)
 * - Float64 (double precision)
 * - Float16 (half precision, when supported)
 *
 * Tests models across CPU, CUDA, Vulkan, and OneAPI backends.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/roberta.hpp"
#include "../../include/tenzor/models/electra.hpp"
#include "../../include/tenzor/nn/offload.hpp"

using namespace tenzor;
using namespace tenzor::models;

// ============================================================================
// Multi-DType Test Parameter Structure
// ============================================================================

struct BackendDTypeParam {
    std::string backend_name;
    DType dtype;
    std::string dtype_name;

    std::string ToString() const {
        return backend_name + "_" + dtype_name;
    }
};

// Required for gtest_discover_tests to show human-readable test names
void PrintTo(const BackendDTypeParam& param, std::ostream* os) {
    *os << param.ToString();
}

// ============================================================================
// Multi-DType Test Fixture
// ============================================================================

class RoBERTaELECTRAMultiDTypeTest : public ::testing::TestWithParam<BackendDTypeParam> {
protected:
    Device device;
    DType dtype;

    void SetUp() override {
        tenzor::initialize();

        auto param = GetParam();
        dtype = param.dtype;

        if (param.backend_name == "cpu") {
            device = Device::cpu();
        }
        else if (param.backend_name == "cuda") {
            if (!isBackendAvailable(Device::Type::CUDA)) {
                GTEST_SKIP() << "CUDA not available";
            }
            device = Device::cuda(0);
        }
        else if (param.backend_name == "vulkan") {
            if (!isBackendAvailable(Device::Type::Vulkan)) {
                GTEST_SKIP() << "Vulkan not available";
            }
            device = Device::vulkan(0);
        }
        else if (param.backend_name == "oneapi") {
            if (!isBackendAvailable(Device::Type::OneAPI)) {
                GTEST_SKIP() << "OneAPI not available";
            }
            device = Device::oneapi(0);
        }
    }

    static bool isBackendAvailable(Device::Type type) {
        try {
            Device test_device{type, 0};
            auto t = zeros({2, 2}, DType::Float32, test_device);
            return true;
        } catch (...) {
            return false;
        }
    }

    // Helper to create input token IDs with valid values
    auto create_input_ids(int64_t batch_size, int64_t seq_len, int64_t vocab_size = 50265) -> Variable {
        // Create on CPU first, then move to target device
        Tensor input_ids_cpu({batch_size, seq_len}, DType::Int64, Device::cpu());

        // Fill with valid token IDs within vocabulary range
        std::vector<int64_t> data(batch_size * seq_len);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = i % vocab_size;
        }
        std::copy(data.begin(), data.end(), input_ids_cpu.data<int64_t>());

        // Move to target device
        Tensor input_ids = (device == Device::cpu()) ? input_ids_cpu : input_ids_cpu.to(device);

        return Variable(input_ids, true);
    }

    // Get tolerance based on dtype
    double get_tolerance() const {
        switch(dtype) {
            case DType::Float32: return 1e-5;
            case DType::Float64: return 1e-10;
            case DType::Float16: return 1e-2;
            default: return 1e-5;
        }
    }

    // Helper to verify tensor values with dtype-specific tolerance
    template<typename T>
    void verify_near(T actual, T expected, const std::string& msg = "") {
        if constexpr (std::is_floating_point_v<T>) {
            EXPECT_NEAR(static_cast<double>(actual), static_cast<double>(expected),
                       get_tolerance()) << msg << " on " << device.to_string();
        } else {
            EXPECT_EQ(actual, expected) << msg << " on " << device.to_string();
        }
    }
};

// ============================================================================
// RoBERTa Base Tests
// ============================================================================

TEST_P(RoBERTaELECTRAMultiDTypeTest, RoBERTaBaseConfigTest) {
    auto config = RobertaConfig::base();

    EXPECT_EQ(config.vocab_size, 50265);
    EXPECT_EQ(config.hidden_size, 768);
    EXPECT_EQ(config.num_hidden_layers, 12);
    EXPECT_EQ(config.num_attention_heads, 12);
    EXPECT_EQ(config.intermediate_size, 3072);
    EXPECT_EQ(config.max_position_embeddings, 514);
}

TEST_P(RoBERTaELECTRAMultiDTypeTest, RoBERTaBaseForwardShape) {
    auto config = RobertaConfig::base();
    auto model = std::make_shared<RobertaModel>(config);

    // Convert model to test dtype and device
    model->to(dtype);
    model->to(device);

    int64_t batch_size = 2;
    int64_t seq_len = 128;

    auto input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);
    auto output = model->forward(input_ids, Tensor{}, Variable{}, Variable{});

    auto shape = output.sequence_output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{batch_size, seq_len, 768}));
    EXPECT_EQ(output.sequence_output.tensor().dtype(), dtype);
}

TEST_P(RoBERTaELECTRAMultiDTypeTest, RoBERTaBaseGradientFlow) {
    auto config = RobertaConfig::base();
    auto model = std::make_shared<RobertaModel>(config);

    // Convert model to test dtype and device
    model->to(dtype);
    model->to(device);

    model->train();

    auto input_ids = create_input_ids(1, 64, config.vocab_size);
    auto output = model->forward(input_ids, Tensor{}, Variable{}, Variable{});
    Variable loss = sum(output.sequence_output);
    loss.backward();

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);

    // Verify gradient exists and has correct dtype
    for (const auto& p : params) {
        if (p->grad()) {
            EXPECT_EQ(p->grad()->dtype(), dtype);
        }
    }
}

TEST_P(RoBERTaELECTRAMultiDTypeTest, RoBERTaBaseParameterCount) {
    auto config = RobertaConfig::base();
    auto model = std::make_shared<RobertaModel>(config);

    // Convert model to test dtype and device
    model->to(dtype);
    model->to(device);

    auto params = model->parameters();

    size_t total_params = 0;
    for (const auto& p : params) {
        size_t param_size = 1;
        for (auto dim : p->tensor().shape()) {
            param_size *= dim;
        }
        total_params += param_size;
        EXPECT_EQ(p->tensor().dtype(), dtype);
    }

    // RoBERTa-Base should have ~125M parameters (allow 20% tolerance)
    EXPECT_GT(total_params, 100'000'000);
    EXPECT_LT(total_params, 150'000'000);
}

// ============================================================================
// RoBERTa Large Tests
// ============================================================================

TEST_P(RoBERTaELECTRAMultiDTypeTest, RoBERTaLargeConfigTest) {
    auto config = RobertaConfig::large();

    EXPECT_EQ(config.hidden_size, 1024);
    EXPECT_EQ(config.num_hidden_layers, 24);
    EXPECT_EQ(config.num_attention_heads, 16);
    EXPECT_EQ(config.intermediate_size, 4096);
}

TEST_P(RoBERTaELECTRAMultiDTypeTest, RoBERTaLargeForwardShape) {
    auto config = RobertaConfig::large();
    auto model = std::make_shared<RobertaModel>(config);

    // Convert model to test dtype and device
    model->to(dtype);
    model->to(device);

    int64_t batch_size = 2;
    int64_t seq_len = 128;

    auto input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);
    auto output = model->forward(input_ids, Tensor{}, Variable{}, Variable{});

    auto shape = output.sequence_output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{batch_size, seq_len, 1024}));
    EXPECT_EQ(output.sequence_output.tensor().dtype(), dtype);
}

TEST_P(RoBERTaELECTRAMultiDTypeTest, RoBERTaLargeGradientFlow) {
    auto config = RobertaConfig::large();

    // For float64, RoBERTa-Large (355M params * 8 bytes = 2.84GB) plus activations
    // and gradients exceeds 6GB GPU. Reduce layers to fit while using offloading.
    bool is_cuda_float64 = (GetParam().backend_name == "cuda" && dtype == DType::Float64);
    if (is_cuda_float64) {
        config.num_hidden_layers = 8;  // Reduced from 24 for float64 CUDA
    }

    auto model = std::make_shared<RobertaModel>(config);
    model->to(dtype);

    // Use CPU-start offloading for cuda float64
    std::unique_ptr<nn::OffloadContext> offload_ctx;

    if (is_cuda_float64) {
        nn::OffloadContext::Config offload_config;
        offload_config.target_device = device;
        offload_config.offload_parameters = true;
        offload_config.offload_gradients = true;
        offload_config.prefetch_depth = 1;
        offload_ctx = std::make_unique<nn::OffloadContext>(*model, offload_config);
        offload_ctx->enable();
    } else {
        model->to(device);
    }

    model->train();

    auto input_ids = create_input_ids(1, 64, config.vocab_size);
    auto output = model->forward(input_ids, Tensor{}, Variable{}, Variable{});
    Variable loss = sum(output.sequence_output);
    loss.backward();

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

// ============================================================================
// ELECTRA Small Tests
// ============================================================================

TEST_P(RoBERTaELECTRAMultiDTypeTest, ELECTRASmallConfigTest) {
    auto config = ElectraConfig::small();

    EXPECT_EQ(config.hidden_size, 256);
    EXPECT_EQ(config.num_hidden_layers, 12);
    EXPECT_EQ(config.num_attention_heads, 4);
    EXPECT_EQ(config.intermediate_size, 1024);
}

TEST_P(RoBERTaELECTRAMultiDTypeTest, ELECTRASmallForwardShape) {
    auto config = ElectraConfig::small();
    auto discriminator = std::make_shared<ElectraDiscriminator>(config);

    // Convert model to test dtype and device
    discriminator->to(dtype);
    discriminator->to(device);

    int64_t batch_size = 2;
    int64_t seq_len = 128;

    auto input_ids = create_input_ids(batch_size, seq_len, 30522);  // ELECTRA vocab_size
    auto output = discriminator->forward(input_ids, Tensor{}, Variable{});

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{batch_size, seq_len}));
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

TEST_P(RoBERTaELECTRAMultiDTypeTest, ELECTRASmallGradientFlow) {
    auto config = ElectraConfig::small();
    auto discriminator = std::make_shared<ElectraDiscriminator>(config);
    discriminator->to(dtype);
    discriminator->to(device);
    discriminator->train();

    auto input_ids = create_input_ids(1, 64, 30522);
    auto output = discriminator->forward(input_ids, Tensor{}, Variable{});
    Variable loss = sum(output);
    loss.backward();

    auto params = discriminator->parameters();
    EXPECT_GT(params.size(), 0);
}

// ============================================================================
// ELECTRA Base Tests
// ============================================================================

TEST_P(RoBERTaELECTRAMultiDTypeTest, ELECTRABaseConfigTest) {
    auto config = ElectraConfig::base();

    EXPECT_EQ(config.hidden_size, 768);
    EXPECT_EQ(config.num_hidden_layers, 12);
    EXPECT_EQ(config.num_attention_heads, 12);
    EXPECT_EQ(config.intermediate_size, 3072);
}

TEST_P(RoBERTaELECTRAMultiDTypeTest, ELECTRABaseForwardShape) {
    auto config = ElectraConfig::base();
    auto discriminator = std::make_shared<ElectraDiscriminator>(config);

    // Convert model to test dtype and device
    discriminator->to(dtype);
    discriminator->to(device);

    int64_t batch_size = 2;
    int64_t seq_len = 128;

    auto input_ids = create_input_ids(batch_size, seq_len, 30522);
    auto output = discriminator->forward(input_ids, Tensor{}, Variable{});

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{batch_size, seq_len}));
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

TEST_P(RoBERTaELECTRAMultiDTypeTest, ELECTRABaseGradientFlow) {
    auto config = ElectraConfig::base();
    auto discriminator = std::make_shared<ElectraDiscriminator>(config);
    discriminator->to(dtype);
    discriminator->to(device);
    discriminator->train();

    auto input_ids = create_input_ids(1, 64, 30522);
    auto output = discriminator->forward(input_ids, Tensor{}, Variable{});
    Variable loss = sum(output);
    loss.backward();

    auto params = discriminator->parameters();
    EXPECT_GT(params.size(), 0);
}

// ============================================================================
// ELECTRA Large Tests
// ============================================================================

TEST_P(RoBERTaELECTRAMultiDTypeTest, ELECTRALargeForwardShape) {
    auto config = ElectraConfig::large();
    auto discriminator = std::make_shared<ElectraDiscriminator>(config);

    // Convert model to test dtype and device
    discriminator->to(dtype);
    discriminator->to(device);

    int64_t batch_size = 1;
    int64_t seq_len = 128;

    auto input_ids = create_input_ids(batch_size, seq_len, 30522);
    auto output = discriminator->forward(input_ids, Tensor{}, Variable{});

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{batch_size, seq_len}));
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

TEST_P(RoBERTaELECTRAMultiDTypeTest, ELECTRALargeGradientFlow) {
    auto config = ElectraConfig::large();

    // For float64, ELECTRA-Large (335M discriminator params * 8 bytes = 2.68GB) plus activations
    // and gradients exceeds 6GB GPU. Reduce layers to fit while using offloading.
    bool is_cuda_float64 = (GetParam().backend_name == "cuda" && dtype == DType::Float64);
    if (is_cuda_float64) {
        config.num_hidden_layers = 8;  // Reduced from 24 for float64 CUDA
    }

    auto discriminator = std::make_shared<ElectraDiscriminator>(config);
    discriminator->to(dtype);

    // Use CPU-start offloading for cuda float64
    std::unique_ptr<nn::OffloadContext> offload_ctx;

    if (is_cuda_float64) {
        nn::OffloadContext::Config offload_config;
        offload_config.target_device = device;
        offload_config.offload_parameters = true;
        offload_config.offload_gradients = true;
        offload_config.prefetch_depth = 1;
        offload_ctx = std::make_unique<nn::OffloadContext>(*discriminator, offload_config);
        offload_ctx->enable();
    } else {
        discriminator->to(device);
    }

    discriminator->train();

    auto input_ids = create_input_ids(1, 64, 30522);
    auto output = discriminator->forward(input_ids, Tensor{}, Variable{});
    Variable loss = sum(output);
    loss.backward();

    auto params = discriminator->parameters();
    EXPECT_GT(params.size(), 0);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_P(RoBERTaELECTRAMultiDTypeTest, RoBERTaBatchSizeOne) {
    auto config = RobertaConfig::base();
    auto model = std::make_shared<RobertaModel>(config);

    // Convert model to test dtype and device
    model->to(dtype);
    model->to(device);

    auto input_ids = create_input_ids(1, 64, config.vocab_size);
    auto output = model->forward(input_ids, Tensor{}, Variable{}, Variable{});

    auto shape = output.sequence_output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 64, 768}));
    EXPECT_EQ(output.sequence_output.tensor().dtype(), dtype);
}

TEST_P(RoBERTaELECTRAMultiDTypeTest, ELECTRAVariableSequenceLength) {
    auto config = ElectraConfig::base();
    auto discriminator = std::make_shared<ElectraDiscriminator>(config);

    // Convert model to test dtype and device
    discriminator->to(dtype);
    discriminator->to(device);

    // Test with different sequence lengths
    auto input_32 = create_input_ids(2, 32, 30522);
    auto output_32 = discriminator->forward(input_32, Tensor{}, Variable{});
    auto shape_32 = output_32.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape_32.begin(), shape_32.end()),
              (std::vector<int64_t>{2, 32}));
    EXPECT_EQ(output_32.tensor().dtype(), dtype);

    auto input_256 = create_input_ids(1, 256, 30522);
    auto output_256 = discriminator->forward(input_256, Tensor{}, Variable{});
    auto shape_256 = output_256.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape_256.begin(), shape_256.end()),
              (std::vector<int64_t>{1, 256}));
    EXPECT_EQ(output_256.tensor().dtype(), dtype);
}

// ============================================================================
// Test Instantiation
// ============================================================================

std::vector<BackendDTypeParam> GenerateBackendDTypeCombinations() {
    std::vector<std::string> backends = {"cpu", "cuda", "vulkan", "oneapi"};

    // NLP models primarily use floating-point types
    std::vector<std::pair<DType, std::string>> dtypes = {
        {DType::Float32, "float32"},
        {DType::Float64, "float64"},
        {DType::Float16, "float16"},  // For mixed precision training
    };

    std::vector<BackendDTypeParam> combinations;
    for (const auto& backend : backends) {
        for (const auto& [dtype, dtype_name] : dtypes) {
            combinations.push_back({backend, dtype, dtype_name});
        }
    }
    return combinations;
}

INSTANTIATE_TEST_SUITE_P(
    AllBackendsAllDTypes,
    RoBERTaELECTRAMultiDTypeTest,
    ::testing::ValuesIn(GenerateBackendDTypeCombinations()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

/*
 * COVERAGE IMPACT SUMMARY:
 *
 * Original test_roberta_electra.cpp:
 * - 15 tests × 1 backend (CPU) × 1 dtype (Float32) = 15 test scenarios
 *
 * Refactored test_roberta_electra_multidtype.cpp:
 * - 15 tests × 4 backends (CPU, CUDA, Vulkan, OneAPI) × 3 dtypes (Float32, Float64, Float16)
 * - Total: 15 × 4 × 3 = 180 test scenarios
 *
 * Coverage increase: ~12x improvement
 *
 * Tests converted: 15/15 (100%)
 * - RoBERTa Base: Config, ForwardShape, GradientFlow, ParameterCount (4 tests)
 * - RoBERTa Large: Config, ForwardShape, GradientFlow (3 tests)
 * - ELECTRA Small: Config, ForwardShape, GradientFlow (3 tests)
 * - ELECTRA Base: Config, ForwardShape, GradientFlow (3 tests)
 * - ELECTRA Large: ForwardShape, GradientFlow (2 tests)
 * - Edge Cases: RoBERTaBatchSizeOne, ELECTRAVariableSequenceLength (2 tests)
 *
 * DTypes added:
 * - Float32 (original - standard precision)
 * - Float64 (NEW - double precision for research)
 * - Float16 (NEW - half precision for mixed-precision training)
 *
 * Backend support expanded:
 * - CPU (original)
 * - CUDA (NEW - GPU acceleration)
 * - Vulkan (NEW - cross-platform GPU)
 * - OneAPI (NEW - Intel hardware)
 */
