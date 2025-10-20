/**
 * @file test_albert_t5.cpp
 * @brief Comprehensive tests for ALBERT and T5 models
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/albert.hpp"
#include "../../include/tenzor/models/t5.hpp"

using namespace tenzor;
using namespace tenzor::models;

class ALBERTandT5Test : public ::testing::Test {
protected:
    void SetUp() override {
        device_ = Device::cpu();
    }
    Device device_;

    // Helper to create input token IDs with valid values
    auto create_input_ids(int64_t batch_size, int64_t seq_len, int64_t vocab_size = 30000) -> Variable {
        Tensor input_ids({batch_size, seq_len}, DType::Int64, device_);

        // Fill with valid token IDs within vocabulary range
        std::vector<int64_t> data(batch_size * seq_len);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = i % vocab_size;
        }
        std::copy(data.begin(), data.end(), input_ids.data<int64_t>());

        return Variable(input_ids, true);
    }
};

// ============================================================================
// ALBERT Base Tests
// ============================================================================

TEST_F(ALBERTandT5Test, ALBERTBaseConfigTest) {
    auto config = AlbertConfig::base();

    EXPECT_EQ(config.embedding_size, 128);
    EXPECT_EQ(config.hidden_size, 768);
    EXPECT_EQ(config.num_hidden_layers, 12);
    EXPECT_EQ(config.num_attention_heads, 12);
    EXPECT_EQ(config.intermediate_size, 3072);
}

TEST_F(ALBERTandT5Test, ALBERTBaseForwardShape) {
    auto config = AlbertConfig::base();
    config.vocab_size = 30000;
    auto model = std::make_shared<AlbertModel>(config);
    int64_t batch_size = 2;
    int64_t seq_len = 128;

    auto input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);
    auto output = model->forward(input_ids, Tensor{}, Variable{}, Variable{});

    auto shape = output.sequence_output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{batch_size, seq_len, 768}));
}

TEST_F(ALBERTandT5Test, ALBERTBaseGradientFlow) {
    auto config = AlbertConfig::base();
    config.vocab_size = 30000;
    auto model = std::make_shared<AlbertModel>(config);
    model->train();

    auto input_ids = create_input_ids(1, 64);
    auto output = model->forward(input_ids, Tensor{}, Variable{}, Variable{});
    Variable loss = tenzor::sum(output.sequence_output);
    loss.backward();

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_F(ALBERTandT5Test, ALBERTBaseParameterCount) {
    auto config = AlbertConfig::base();
    config.vocab_size = 30000;
    auto model = std::make_shared<AlbertModel>(config);
    auto params = model->parameters();

    size_t total_params = 0;
    for (const auto& p : params) {
        size_t param_size = 1;
        for (auto dim : p->tensor().shape()) {
            param_size *= dim;
        }
        total_params += param_size;
    }

    // ALBERT-Base should have ~12M parameters due to parameter sharing
    // (much less than BERT's ~110M)
    EXPECT_GT(total_params, 8'000'000);
    EXPECT_LT(total_params, 16'000'000);
}

// ============================================================================
// ALBERT Large Tests
// ============================================================================

TEST_F(ALBERTandT5Test, ALBERTLargeConfigTest) {
    auto config = AlbertConfig::large();

    EXPECT_EQ(config.hidden_size, 1024);
    EXPECT_EQ(config.num_hidden_layers, 24);
    EXPECT_EQ(config.num_attention_heads, 16);
}

TEST_F(ALBERTandT5Test, ALBERTLargeForwardShape) {
    auto config = AlbertConfig::large();
    config.vocab_size = 30000;
    auto model = std::make_shared<AlbertModel>(config);
    int64_t batch_size = 2;
    int64_t seq_len = 128;

    auto input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);
    auto output = model->forward(input_ids, Tensor{}, Variable{}, Variable{});

    auto shape = output.sequence_output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{batch_size, seq_len, 1024}));
}

TEST_F(ALBERTandT5Test, ALBERTLargeGradientFlow) {
    auto config = AlbertConfig::large();
    config.vocab_size = 30000;
    auto model = std::make_shared<AlbertModel>(config);
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

TEST_F(ALBERTandT5Test, ALBERTXLargeForwardShape) {
    auto config = AlbertConfig::xlarge();
    config.vocab_size = 30000;
    auto model = std::make_shared<AlbertModel>(config);
    int64_t batch_size = 1;
    int64_t seq_len = 128;

    auto input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);
    auto output = model->forward(input_ids, Tensor{}, Variable{}, Variable{});

    auto shape = output.sequence_output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{batch_size, seq_len, 2048}));
}

// ============================================================================
// ALBERT XXLarge Tests
// ============================================================================

TEST_F(ALBERTandT5Test, ALBERTXXLargeForwardShape) {
    auto config = AlbertConfig::xxlarge();
    config.vocab_size = 30000;
    auto model = std::make_shared<AlbertModel>(config);
    int64_t batch_size = 1;
    int64_t seq_len = 64;

    auto input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);
    auto output = model->forward(input_ids, Tensor{}, Variable{}, Variable{});

    auto shape = output.sequence_output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{batch_size, seq_len, 4096}));
}

// ============================================================================
// T5 Small Tests
// ============================================================================

TEST_F(ALBERTandT5Test, T5SmallConfigTest) {
    auto config = T5Config::small();

    EXPECT_EQ(config.d_model, 512);
    EXPECT_EQ(config.d_ff, 2048);
    EXPECT_EQ(config.num_layers, 6);
    EXPECT_EQ(config.num_heads, 8);
}

TEST_F(ALBERTandT5Test, T5SmallForwardShape) {
    auto config = T5Config::small();
    config.vocab_size = 32128;
    auto model = std::make_shared<T5Model>(config);
    int64_t batch_size = 2;
    int64_t seq_len = 128;

    auto input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);
    auto decoder_input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);

    auto output = model->forward(input_ids, decoder_input_ids);

    auto shape = output.decoder_output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{batch_size, seq_len, 512}));
}

TEST_F(ALBERTandT5Test, T5SmallGradientFlow) {
    auto config = T5Config::small();
    config.vocab_size = 32128;
    auto model = std::make_shared<T5Model>(config);
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

TEST_F(ALBERTandT5Test, T5BaseConfigTest) {
    auto config = T5Config::base();

    EXPECT_EQ(config.d_model, 768);
    EXPECT_EQ(config.d_ff, 3072);
    EXPECT_EQ(config.num_layers, 12);
    EXPECT_EQ(config.num_heads, 12);
}

TEST_F(ALBERTandT5Test, T5BaseForwardShape) {
    auto config = T5Config::base();
    config.vocab_size = 32128;
    auto model = std::make_shared<T5Model>(config);
    int64_t batch_size = 2;
    int64_t seq_len = 128;

    auto input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);
    auto decoder_input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);

    auto output = model->forward(input_ids, decoder_input_ids);

    auto shape = output.decoder_output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{batch_size, seq_len, 768}));
}

TEST_F(ALBERTandT5Test, T5BaseGradientFlow) {
    auto config = T5Config::base();
    config.vocab_size = 32128;
    auto model = std::make_shared<T5Model>(config);
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

TEST_F(ALBERTandT5Test, T5LargeForwardShape) {
    auto config = T5Config::large();
    config.vocab_size = 32128;
    auto model = std::make_shared<T5Model>(config);
    int64_t batch_size = 1;
    int64_t seq_len = 128;

    auto input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);
    auto decoder_input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);

    auto output = model->forward(input_ids, decoder_input_ids);

    auto shape = output.decoder_output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{batch_size, seq_len, 1024}));
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(ALBERTandT5Test, ALBERTBatchSizeOne) {
    auto config = AlbertConfig::base();
    config.vocab_size = 30000;
    auto model = std::make_shared<AlbertModel>(config);
    auto input_ids = create_input_ids(1, 64);
    auto output = model->forward(input_ids, Tensor{}, Variable{}, Variable{});

    auto shape = output.sequence_output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 64, 768}));
}

TEST_F(ALBERTandT5Test, T5VariableSequenceLength) {
    auto config = T5Config::small();
    config.vocab_size = 32128;
    auto model = std::make_shared<T5Model>(config);

    // Test with different sequence lengths
    auto input_32 = create_input_ids(2, 32, config.vocab_size);
    auto decoder_32 = create_input_ids(2, 32, config.vocab_size);
    auto output_32 = model->forward(input_32, decoder_32);
    auto shape_32 = output_32.decoder_output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape_32.begin(), shape_32.end()),
              (std::vector<int64_t>{2, 32, 512}));
}


// ============================================================================
// Main  
// ============================================================================

int main(int argc, char** argv) {
    tenzor::initialize();  // Initialize Tenzor library and backends
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
