/**
 * @file test_roberta_electra.cpp
 * @brief Comprehensive tests for RoBERTa and ELECTRA models
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../backend_test_fixture.hpp"
#include "../../include/tenzor/models/roberta.hpp"
#include "../../include/tenzor/models/electra.hpp"

using namespace tenzor;
using namespace tenzor::models;

class RoBERTaELECTRATest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }

    // Helper to create input token IDs with valid values
    auto create_input_ids(int64_t batch_size, int64_t seq_len, int64_t vocab_size = 50265) -> Variable {
        // Token/index inputs are built on CPU via host writes, then moved to device.
        Tensor input_ids({batch_size, seq_len}, DType::Int64, Device::cpu());

        // Fill with valid token IDs within vocabulary range
        std::vector<int64_t> data(batch_size * seq_len);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = i % vocab_size;
        }
        std::copy(data.begin(), data.end(), input_ids.data<int64_t>());

        return Variable(input_ids.to(device), true);
    }
};

// ============================================================================
// RoBERTa Base Tests
// ============================================================================

TEST_P(RoBERTaELECTRATest, RoBERTaBaseConfigTest) {
    auto config = RobertaConfig::base();

    EXPECT_EQ(config.vocab_size, 50265);
    EXPECT_EQ(config.hidden_size, 768);
    EXPECT_EQ(config.num_hidden_layers, 12);
    EXPECT_EQ(config.num_attention_heads, 12);
    EXPECT_EQ(config.intermediate_size, 3072);
    EXPECT_EQ(config.max_position_embeddings, 514);
}

TEST_P(RoBERTaELECTRATest, RoBERTaBaseForwardShape) {
    auto config = RobertaConfig::base();
    auto model = std::make_shared<RobertaModel>(config);
    model->to(device);
    int64_t batch_size = 2;
    int64_t seq_len = 128;

    auto input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);
    auto output = model->forward(input_ids, Tensor{}, Variable{}, Variable{});

    auto shape = output.sequence_output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{batch_size, seq_len, 768}));
}

TEST_P(RoBERTaELECTRATest, RoBERTaBaseGradientFlow) {
    auto config = RobertaConfig::base();
    auto model = std::make_shared<RobertaModel>(config);
    model->to(device);
    model->train();

    auto input_ids = create_input_ids(1, 64, config.vocab_size);
    auto output = model->forward(input_ids, Tensor{}, Variable{}, Variable{});
    Variable loss = sum(output.sequence_output);
    loss.backward();

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_P(RoBERTaELECTRATest, RoBERTaBaseParameterCount) {
    auto config = RobertaConfig::base();
    auto model = std::make_shared<RobertaModel>(config);
    model->to(device);
    auto params = model->parameters();

    size_t total_params = 0;
    for (const auto& p : params) {
        size_t param_size = 1;
        for (auto dim : p->tensor().shape()) {
            param_size *= dim;
        }
        total_params += param_size;
    }

    // RoBERTa-Base should have ~125M parameters (allow 20% tolerance)
    EXPECT_GT(total_params, 100'000'000);
    EXPECT_LT(total_params, 150'000'000);
}

// ============================================================================
// RoBERTa Large Tests
// ============================================================================

TEST_P(RoBERTaELECTRATest, RoBERTaLargeConfigTest) {
    auto config = RobertaConfig::large();

    EXPECT_EQ(config.hidden_size, 1024);
    EXPECT_EQ(config.num_hidden_layers, 24);
    EXPECT_EQ(config.num_attention_heads, 16);
    EXPECT_EQ(config.intermediate_size, 4096);
}

TEST_P(RoBERTaELECTRATest, RoBERTaLargeForwardShape) {
    auto config = RobertaConfig::large();
    auto model = std::make_shared<RobertaModel>(config);
    model->to(device);
    int64_t batch_size = 2;
    int64_t seq_len = 128;

    auto input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);
    auto output = model->forward(input_ids, Tensor{}, Variable{}, Variable{});

    auto shape = output.sequence_output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{batch_size, seq_len, 1024}));
}

TEST_P(RoBERTaELECTRATest, RoBERTaLargeGradientFlow) {
    auto config = RobertaConfig::large();
    auto model = std::make_shared<RobertaModel>(config);
    model->to(device);
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

TEST_P(RoBERTaELECTRATest, ELECTRASmallConfigTest) {
    auto config = ElectraConfig::small();

    EXPECT_EQ(config.hidden_size, 256);
    EXPECT_EQ(config.num_hidden_layers, 12);
    EXPECT_EQ(config.num_attention_heads, 4);
    EXPECT_EQ(config.intermediate_size, 1024);
}

TEST_P(RoBERTaELECTRATest, ELECTRASmallForwardShape) {
    auto config = ElectraConfig::small();
    auto discriminator = std::make_shared<ElectraDiscriminator>(config);
    discriminator->to(device);
    int64_t batch_size = 2;
    int64_t seq_len = 128;

    auto input_ids = create_input_ids(batch_size, seq_len, 30522);  // ELECTRA vocab_size
    auto output = discriminator->forward(input_ids, Tensor{}, Variable{});

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{batch_size, seq_len}));
}

TEST_P(RoBERTaELECTRATest, ELECTRASmallGradientFlow) {
    auto config = ElectraConfig::small();
    auto discriminator = std::make_shared<ElectraDiscriminator>(config);
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

TEST_P(RoBERTaELECTRATest, ELECTRABaseConfigTest) {
    auto config = ElectraConfig::base();

    EXPECT_EQ(config.hidden_size, 768);
    EXPECT_EQ(config.num_hidden_layers, 12);
    EXPECT_EQ(config.num_attention_heads, 12);
    EXPECT_EQ(config.intermediate_size, 3072);
}

TEST_P(RoBERTaELECTRATest, ELECTRABaseForwardShape) {
    auto config = ElectraConfig::base();
    auto discriminator = std::make_shared<ElectraDiscriminator>(config);
    discriminator->to(device);
    int64_t batch_size = 2;
    int64_t seq_len = 128;

    auto input_ids = create_input_ids(batch_size, seq_len, 30522);
    auto output = discriminator->forward(input_ids, Tensor{}, Variable{});

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{batch_size, seq_len}));
}

TEST_P(RoBERTaELECTRATest, ELECTRABaseGradientFlow) {
    auto config = ElectraConfig::base();
    auto discriminator = std::make_shared<ElectraDiscriminator>(config);
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

TEST_P(RoBERTaELECTRATest, ELECTRALargeForwardShape) {
    auto config = ElectraConfig::large();
    auto discriminator = std::make_shared<ElectraDiscriminator>(config);
    discriminator->to(device);
    int64_t batch_size = 1;
    int64_t seq_len = 128;

    auto input_ids = create_input_ids(batch_size, seq_len, 30522);
    auto output = discriminator->forward(input_ids, Tensor{}, Variable{});

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{batch_size, seq_len}));
}

TEST_P(RoBERTaELECTRATest, ELECTRALargeGradientFlow) {
    auto config = ElectraConfig::large();
    auto discriminator = std::make_shared<ElectraDiscriminator>(config);
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
// Edge Case Tests
// ============================================================================

TEST_P(RoBERTaELECTRATest, RoBERTaBatchSizeOne) {
    auto config = RobertaConfig::base();
    auto model = std::make_shared<RobertaModel>(config);
    model->to(device);
    auto input_ids = create_input_ids(1, 64, config.vocab_size);
    auto output = model->forward(input_ids, Tensor{}, Variable{}, Variable{});

    auto shape = output.sequence_output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 64, 768}));
}

TEST_P(RoBERTaELECTRATest, ELECTRAVariableSequenceLength) {
    auto config = ElectraConfig::base();
    auto discriminator = std::make_shared<ElectraDiscriminator>(config);
    discriminator->to(device);

    // Test with different sequence lengths
    auto input_32 = create_input_ids(2, 32, 30522);
    auto output_32 = discriminator->forward(input_32, Tensor{}, Variable{});
    auto shape_32 = output_32.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape_32.begin(), shape_32.end()),
              (std::vector<int64_t>{2, 32}));

    auto input_256 = create_input_ids(1, 256, 30522);
    auto output_256 = discriminator->forward(input_256, Tensor{}, Variable{});
    auto shape_256 = output_256.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape_256.begin(), shape_256.end()),
              (std::vector<int64_t>{1, 256}));
}

INSTANTIATE_BACKEND_TESTS(RoBERTaELECTRATest);
