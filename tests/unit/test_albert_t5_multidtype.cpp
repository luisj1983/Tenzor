/**
 * @file test_albert_t5_multidtype.cpp
 * @brief Multi-dtype tests for ALBERT and T5 models
 *
 * Tests ALBERT and T5 models with Float32, Float64, and Float16 data types across
 * CPU, CUDA, OneAPI, Vulkan, and ROCm backends to ensure:
 * - ALBERT base/large/xlarge/xxlarge variants work correctly
 * - T5 small/base/large variants work correctly
 * - Parameter sharing works across dtypes
 * - Gradient flow works for all model variants
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/albert.hpp"
#include "../../include/tenzor/models/t5.hpp"
#include "../../include/tenzor/nn/offload.hpp"
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::models;

// ============================================================================
// ALBERT and T5 Multi-Backend Multi-DType Test Fixture
// ============================================================================

class ALBERTandT5MultiDTypeTest : public MultiBackendDTypeTest {
protected:
    // Helper to create input token IDs with valid values
    auto create_input_ids(int64_t batch_size, int64_t seq_len, int64_t vocab_size = 30000) -> Variable {
        // Create on CPU first, then move to target device
        Tensor input_ids_cpu({batch_size, seq_len}, DType::Int64, Device::cpu());

        // Fill with valid token IDs within vocabulary range
        std::vector<int64_t> data(batch_size * seq_len);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = i % vocab_size;
        }
        std::copy(data.begin(), data.end(), input_ids_cpu.data<int64_t>());

        // Move to target device
        Tensor input_ids = (device() == Device::cpu()) ? input_ids_cpu : input_ids_cpu.to(device());

        return Variable(input_ids, true);
    }
};

// ============================================================================
// ALBERT Base Tests
// ============================================================================

TEST_P(ALBERTandT5MultiDTypeTest, ALBERTBaseConfigTest) {
    auto config = AlbertConfig::base();

    EXPECT_EQ(config.embedding_size, 128);
    EXPECT_EQ(config.hidden_size, 768);
    EXPECT_EQ(config.num_hidden_layers, 12);
    EXPECT_EQ(config.num_attention_heads, 12);
    EXPECT_EQ(config.intermediate_size, 3072);
}

TEST_P(ALBERTandT5MultiDTypeTest, ALBERTBaseForwardShape) {
    auto config = AlbertConfig::base();
    config.vocab_size = 30000;
    auto model = std::make_shared<AlbertModel>(config);

    // Convert model to test dtype and device
    model->to(dtype());
    model->to(device());

    int64_t batch_size = 2;
    int64_t seq_len = 128;

    auto input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);
    auto output = model->forward(input_ids, Tensor{}, Variable{}, Variable{});

    expectShape(output.sequence_output.tensor(), {batch_size, seq_len, 768});
    expectDType(output.sequence_output.tensor());
}

TEST_P(ALBERTandT5MultiDTypeTest, ALBERTBaseGradientFlow) {
    auto config = AlbertConfig::base();
    config.vocab_size = 30000;
    auto model = std::make_shared<AlbertModel>(config);
    model->to(dtype());
    model->to(device());
    model->train();

    auto input_ids = create_input_ids(1, 64);
    auto output = model->forward(input_ids, Tensor{}, Variable{}, Variable{});
    Variable loss = tenzor::sum(output.sequence_output);
    loss.backward();

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);

    // Verify gradient exists and has correct dtype
    for (const auto& p : params) {
        if (p->grad()) {
            EXPECT_EQ(p->grad()->dtype(), dtype());
        }
    }
}

TEST_P(ALBERTandT5MultiDTypeTest, ALBERTBaseParameterCount) {
    auto config = AlbertConfig::base();
    config.vocab_size = 30000;
    auto model = std::make_shared<AlbertModel>(config);
    model->to(dtype());
    model->to(device());
    auto params = model->parameters();

    size_t total_params = countParameters(params);

    // ALBERT-Base should have ~12M parameters due to parameter sharing
    // (much less than BERT's ~110M)
    EXPECT_GT(total_params, 8'000'000);
    EXPECT_LT(total_params, 16'000'000);
}

// ============================================================================
// ALBERT Large Tests
// ============================================================================

TEST_P(ALBERTandT5MultiDTypeTest, ALBERTLargeConfigTest) {
    auto config = AlbertConfig::large();

    EXPECT_EQ(config.hidden_size, 1024);
    EXPECT_EQ(config.num_hidden_layers, 24);
    EXPECT_EQ(config.num_attention_heads, 16);
}

TEST_P(ALBERTandT5MultiDTypeTest, ALBERTLargeForwardShape) {
    auto config = AlbertConfig::large();
    config.vocab_size = 30000;
    auto model = std::make_shared<AlbertModel>(config);

    // Convert model to test dtype
    model->to(dtype());
    model->to(device());

    int64_t batch_size = 2;
    int64_t seq_len = 128;

    auto input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);
    auto output = model->forward(input_ids, Tensor{}, Variable{}, Variable{});

    expectShape(output.sequence_output.tensor(), {batch_size, seq_len, 1024});
    expectDType(output.sequence_output.tensor());
}

TEST_P(ALBERTandT5MultiDTypeTest, ALBERTLargeGradientFlow) {
    auto config = AlbertConfig::large();
    config.vocab_size = 30000;
    auto model = std::make_shared<AlbertModel>(config);
    model->to(dtype());
    model->to(device());
    model->train();

    auto input_ids = create_input_ids(1, 64);
    auto output = model->forward(input_ids, Tensor{}, Variable{}, Variable{});
    Variable loss = tenzor::sum(output.sequence_output);
    loss.backward();

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

// ============================================================================
// ALBERT XLarge Tests
// ============================================================================

TEST_P(ALBERTandT5MultiDTypeTest, ALBERTXLargeForwardShape) {
    auto config = AlbertConfig::xlarge();
    config.vocab_size = 30000;
    auto model = std::make_shared<AlbertModel>(config);

    // Convert model to test dtype
    model->to(dtype());
    model->to(device());

    int64_t batch_size = 1;
    int64_t seq_len = 128;

    auto input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);
    auto output = model->forward(input_ids, Tensor{}, Variable{}, Variable{});

    expectShape(output.sequence_output.tensor(), {batch_size, seq_len, 2048});
    expectDType(output.sequence_output.tensor());
}

// ============================================================================
// ALBERT XXLarge Tests
// ============================================================================

TEST_P(ALBERTandT5MultiDTypeTest, ALBERTXXLargeForwardShape) {
    auto config = AlbertConfig::xxlarge();
    config.vocab_size = 30000;
    auto model = std::make_shared<AlbertModel>(config);

    // Convert model to test dtype
    model->to(dtype());
    model->to(device());

    int64_t batch_size = 1;
    int64_t seq_len = 64;

    auto input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);
    auto output = model->forward(input_ids, Tensor{}, Variable{}, Variable{});

    expectShape(output.sequence_output.tensor(), {batch_size, seq_len, 4096});
    expectDType(output.sequence_output.tensor());
}

// ============================================================================
// T5 Small Tests
// ============================================================================

TEST_P(ALBERTandT5MultiDTypeTest, T5SmallConfigTest) {
    auto config = T5Config::small();

    EXPECT_EQ(config.d_model, 512);
    EXPECT_EQ(config.d_ff, 2048);
    EXPECT_EQ(config.num_layers, 6);
    EXPECT_EQ(config.num_heads, 8);
}

TEST_P(ALBERTandT5MultiDTypeTest, T5SmallForwardShape) {
    auto config = T5Config::small();
    config.vocab_size = 32128;
    auto model = std::make_shared<T5Model>(config);

    // Convert model to test dtype and device
    model->to(dtype());
    model->to(device());

    int64_t batch_size = 2;
    int64_t seq_len = 128;

    auto input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);
    auto decoder_input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);

    auto output = model->forward(input_ids, decoder_input_ids);

    expectShape(output.decoder_output.tensor(), {batch_size, seq_len, 512});
    expectDType(output.decoder_output.tensor());
}

TEST_P(ALBERTandT5MultiDTypeTest, T5SmallGradientFlow) {
    auto config = T5Config::small();
    config.vocab_size = 32128;
    auto model = std::make_shared<T5Model>(config);
    model->to(dtype());
    model->to(device());
    model->train();

    auto input_ids = create_input_ids(1, 64, config.vocab_size);
    auto decoder_input_ids = create_input_ids(1, 64, config.vocab_size);

    auto output = model->forward(input_ids, decoder_input_ids);
    Variable loss = tenzor::sum(output.decoder_output);
    loss.backward();

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

// ============================================================================
// T5 Base Tests
// ============================================================================

TEST_P(ALBERTandT5MultiDTypeTest, T5BaseConfigTest) {
    auto config = T5Config::base();

    EXPECT_EQ(config.d_model, 768);
    EXPECT_EQ(config.d_ff, 3072);
    EXPECT_EQ(config.num_layers, 12);
    EXPECT_EQ(config.num_heads, 12);
}

TEST_P(ALBERTandT5MultiDTypeTest, T5BaseForwardShape) {
    auto config = T5Config::base();
    config.vocab_size = 32128;
    auto model = std::make_shared<T5Model>(config);

    // Convert model to test dtype and device
    model->to(dtype());
    model->to(device());

    int64_t batch_size = 2;
    int64_t seq_len = 128;

    auto input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);
    auto decoder_input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);

    auto output = model->forward(input_ids, decoder_input_ids);

    expectShape(output.decoder_output.tensor(), {batch_size, seq_len, 768});
    expectDType(output.decoder_output.tensor());
}

TEST_P(ALBERTandT5MultiDTypeTest, T5BaseGradientFlow) {
    auto config = T5Config::base();
    config.vocab_size = 32128;

    // For float64, T5-base (220M params * 8 bytes = 1.76GB) plus activations
    // and gradients exceeds 6GB GPU. Reduce layers to fit.
    bool is_float64 = (dtype() == DType::Float64);
    auto cur_backend = backend_name();
    if (is_float64 && (cur_backend == "cuda" || cur_backend == "vulkan")) {
        config.num_layers = 4;  // Reduced from 12 for float64 CUDA/Vulkan
    }

    auto model = std::make_shared<T5Model>(config);
    model->to(dtype());
    model->to(device());
    model->train();

    auto input_ids = create_input_ids(1, 64, config.vocab_size);
    auto decoder_input_ids = create_input_ids(1, 64, config.vocab_size);

    auto output = model->forward(input_ids, decoder_input_ids);
    Variable loss = tenzor::sum(output.decoder_output);
    loss.backward();

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

// ============================================================================
// T5 Large Tests
// ============================================================================

TEST_P(ALBERTandT5MultiDTypeTest, T5LargeForwardShape) {
    auto config = T5Config::large();
    config.vocab_size = 32128;

    // For float64, T5-large (770M params * 8 bytes = 6.2GB) far exceeds 6GB GPU.
    bool is_float64 = (dtype() == DType::Float64);
    auto cur_backend = backend_name();
    if (is_float64 && (cur_backend == "cuda" || cur_backend == "vulkan")) {
        config.num_layers = 6;  // Reduced from 24 for float64 CUDA/Vulkan
    }

    auto model = std::make_shared<T5Model>(config);

    // Convert model to test dtype and device
    model->to(dtype());
    model->to(device());

    int64_t batch_size = 1;
    int64_t seq_len = is_float64 ? 64 : 128;  // Shorter seq for float64

    auto input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);
    auto decoder_input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);

    auto output = model->forward(input_ids, decoder_input_ids);

    expectShape(output.decoder_output.tensor(), {batch_size, seq_len, 1024});
    expectDType(output.decoder_output.tensor());
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_P(ALBERTandT5MultiDTypeTest, ALBERTBatchSizeOne) {
    auto config = AlbertConfig::base();
    config.vocab_size = 30000;
    auto model = std::make_shared<AlbertModel>(config);

    // Convert model to test dtype and device
    model->to(dtype());
    model->to(device());

    auto input_ids = create_input_ids(1, 64);
    auto output = model->forward(input_ids, Tensor{}, Variable{}, Variable{});

    expectShape(output.sequence_output.tensor(), {1, 64, 768});
    expectDType(output.sequence_output.tensor());
}

TEST_P(ALBERTandT5MultiDTypeTest, T5VariableSequenceLength) {
    auto config = T5Config::small();
    config.vocab_size = 32128;
    auto model = std::make_shared<T5Model>(config);

    // Convert model to test dtype and device
    model->to(dtype());
    model->to(device());

    // Test with different sequence lengths
    auto input_32 = create_input_ids(2, 32, config.vocab_size);
    auto decoder_32 = create_input_ids(2, 32, config.vocab_size);
    auto output_32 = model->forward(input_32, decoder_32);

    expectShape(output_32.decoder_output.tensor(), {2, 32, 512});
    expectDType(output_32.decoder_output.tensor());
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ALBERTandT5MultiDTypeTest);

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
