/**
 * @file test_electra.cpp
 * @brief Unit tests for ELECTRA models
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"  // For initialize()
#include "tenzor/models/electra.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"

using namespace tenzor;
using namespace tenzor::models;

class ElectraTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set random seed for reproducibility
        // srand(42);
    }
};

// ============================================================================
// ElectraConfig Tests
// ============================================================================

TEST_F(ElectraTest, ConfigSmall) {
    auto config = ElectraConfig::small();

    EXPECT_EQ(config.vocab_size, 30522);
    EXPECT_EQ(config.hidden_size, 256);
    EXPECT_EQ(config.num_hidden_layers, 12);
    EXPECT_EQ(config.num_attention_heads, 4);
    EXPECT_EQ(config.generator_hidden_size, 128);
    EXPECT_EQ(config.generator_heads, 2);
}

TEST_F(ElectraTest, ConfigBase) {
    auto config = ElectraConfig::base();

    EXPECT_EQ(config.vocab_size, 30522);
    EXPECT_EQ(config.hidden_size, 768);
    EXPECT_EQ(config.num_hidden_layers, 12);
    EXPECT_EQ(config.num_attention_heads, 12);
    EXPECT_EQ(config.generator_hidden_size, 256);  // 1/3 of discriminator
    EXPECT_EQ(config.generator_heads, 4);
}

TEST_F(ElectraTest, ConfigLarge) {
    auto config = ElectraConfig::large();

    EXPECT_EQ(config.hidden_size, 1024);
    EXPECT_EQ(config.num_hidden_layers, 24);
    EXPECT_EQ(config.num_attention_heads, 16);
    EXPECT_EQ(config.generator_hidden_size, 256);  // 1/4 of discriminator
}

TEST_F(ElectraTest, ConfigToBertConfig) {
    auto electra_config = ElectraConfig::base();

    auto disc_config = electra_config.to_discriminator_config();
    EXPECT_EQ(disc_config.hidden_size, 768);
    EXPECT_EQ(disc_config.num_hidden_layers, 12);

    auto gen_config = electra_config.to_generator_config();
    EXPECT_EQ(gen_config.hidden_size, 256);
    EXPECT_EQ(gen_config.num_hidden_layers, 12);
}

// ============================================================================
// ElectraGenerator Tests
// ============================================================================

TEST_F(ElectraTest, GeneratorCreation) {
    auto config = ElectraConfig::base();
    EXPECT_NO_THROW({
        ElectraGenerator generator(config);
    });
}

TEST_F(ElectraTest, GeneratorForward) {
    auto config = ElectraConfig::base();
    config.num_hidden_layers = 2;
    config.generator_layers = 2;
    ElectraGenerator generator(config);

    int64_t batch_size = 2;
    int64_t seq_len = 10;

    Tensor input_tensor({batch_size, seq_len}, DType::Int64, Device::cpu());
    input_tensor.fill_(42);
    Variable input_ids(input_tensor, false);

    auto logits = generator.forward(input_ids, Tensor{}, Variable{});

    // Check output shape: [batch, seq_len, vocab_size]
    EXPECT_EQ(logits.shape()[0], batch_size);
    EXPECT_EQ(logits.shape()[1], seq_len);
    EXPECT_EQ(logits.shape()[2], config.vocab_size);
}

// ============================================================================
// ElectraDiscriminator Tests
// ============================================================================

TEST_F(ElectraTest, DiscriminatorCreation) {
    auto config = ElectraConfig::base();
    EXPECT_NO_THROW({
        ElectraDiscriminator discriminator(config);
    });
}

TEST_F(ElectraTest, DiscriminatorForward) {
    auto config = ElectraConfig::base();
    config.num_hidden_layers = 2;
    ElectraDiscriminator discriminator(config);

    int64_t batch_size = 2;
    int64_t seq_len = 10;

    Tensor input_tensor({batch_size, seq_len}, DType::Int64, Device::cpu());
    input_tensor.fill_(42);
    Variable input_ids(input_tensor, false);

    auto logits = discriminator.forward(input_ids, Tensor{}, Variable{});

    // Check output shape: [batch, seq_len] (binary classification per token)
    EXPECT_EQ(logits.shape()[0], batch_size);
    EXPECT_EQ(logits.shape()[1], seq_len);
}

// ============================================================================
// ElectraForPreTraining Tests
// ============================================================================

TEST_F(ElectraTest, PreTrainingCreation) {
    auto config = ElectraConfig::base();
    config.num_hidden_layers = 2;
    config.generator_layers = 2;

    EXPECT_NO_THROW({
        ElectraForPreTraining model(config);
    });
}

TEST_F(ElectraTest, PreTrainingForward) {
    auto config = ElectraConfig::base();
    config.num_hidden_layers = 2;
    config.generator_layers = 2;
    ElectraForPreTraining model(config);

    int64_t batch_size = 2;
    int64_t seq_len = 10;

    // Create input with some masked positions
    Tensor input_tensor({batch_size, seq_len}, DType::Int64, Device::cpu());
    input_tensor.fill_(42);
    // Set some positions to [MASK] token (103)
    auto input_data = input_tensor.data<int64_t>();
    input_data[3] = 103;  // Mask position 3 in first batch
    input_data[seq_len + 5] = 103;  // Mask position 5 in second batch
    Variable input_ids(input_tensor, false);

    // Create masked positions tensor
    Tensor masked_positions({batch_size, seq_len}, DType::Int64, Device::cpu());
    masked_positions.zero_();
    auto mask_data = masked_positions.data<int64_t>();
    mask_data[3] = 1;  // Position 3 is masked
    mask_data[seq_len + 5] = 1;  // Position 5 is masked

    // Create original tokens (before masking)
    Tensor original_tokens({batch_size, seq_len}, DType::Int64, Device::cpu());
    original_tokens.fill_(42);
    auto orig_data = original_tokens.data<int64_t>();
    orig_data[3] = 100;  // Original token at position 3
    orig_data[seq_len + 5] = 200;  // Original token at position 5

    // Forward pass
    auto outputs = model.forward(input_ids, masked_positions, original_tokens);

    // Check generator logits shape: [batch, seq_len, vocab_size]
    EXPECT_EQ(outputs.gen_logits.shape()[0], batch_size);
    EXPECT_EQ(outputs.gen_logits.shape()[1], seq_len);
    EXPECT_EQ(outputs.gen_logits.shape()[2], config.vocab_size);

    // Check discriminator logits shape: [batch, seq_len]
    EXPECT_EQ(outputs.disc_logits.shape()[0], batch_size);
    EXPECT_EQ(outputs.disc_logits.shape()[1], seq_len);

    // Check is_replaced tensor
    EXPECT_EQ(outputs.is_replaced.shape()[0], batch_size);
    EXPECT_EQ(outputs.is_replaced.shape()[1], seq_len);
}

TEST_F(ElectraTest, PreTrainingLossComputation) {
    auto config = ElectraConfig::base();
    config.num_hidden_layers = 1;
    config.generator_layers = 1;
    ElectraForPreTraining model(config);

    int64_t batch_size = 2;
    int64_t seq_len = 8;

    // Create input
    Tensor input_tensor({batch_size, seq_len}, DType::Int64, Device::cpu());
    input_tensor.fill_(42);
    Variable input_ids(input_tensor, false);

    // Create masked positions
    Tensor masked_positions({batch_size, seq_len}, DType::Int64, Device::cpu());
    masked_positions.zero_();
    auto mask_data = masked_positions.data<int64_t>();
    mask_data[0] = 1;
    mask_data[seq_len + 2] = 1;

    // Create original tokens
    Tensor original_tokens({batch_size, seq_len}, DType::Int64, Device::cpu());
    original_tokens.fill_(42);

    // Forward pass
    auto outputs = model.forward(input_ids, masked_positions, original_tokens);

    // Compute loss
    EXPECT_NO_THROW({
        auto loss = model.compute_loss(
            outputs.gen_logits,
            outputs.disc_logits,
            outputs.is_replaced,
            masked_positions,
            original_tokens
        );

        // Loss should be a scalar
        EXPECT_EQ(loss.shape().size(), 0);  // Scalar
    });
}

// G13 regression: MLM loss must be a real cross-entropy on masked positions,
// not `mean(log_probs)` which is ≈ -log(V) and provides no learning signal.
//
// The verification: (a) gen_loss is non-negative (CE always is), (b) it
// changes when the set of masked positions changes — proving the masking
// actually influences the loss instead of being silently dropped.
TEST_F(ElectraTest, MLMLossIsRealCrossEntropyAtMaskedPositions_G13) {
    auto config = ElectraConfig::base();
    config.num_hidden_layers = 1;
    config.generator_layers = 1;
    // Use only generator weight so disc_loss doesn't dominate the signal —
    // makes the loss-magnitude assertions easy to reason about.
    config.gen_loss_weight = 1.0f;
    config.disc_loss_weight = 0.0f;
    ElectraForPreTraining model(config);
    model.eval();

    const int64_t B = 2;
    const int64_t T = 8;

    Tensor input_tensor({B, T}, DType::Int64, Device::cpu());
    input_tensor.fill_(42);
    Variable input_ids(input_tensor, false);

    // Two different mask patterns covering different positions.
    Tensor mask_a({B, T}, DType::Int64, Device::cpu());
    Tensor mask_b({B, T}, DType::Int64, Device::cpu());
    mask_a.zero_(); mask_b.zero_();
    auto* ma = mask_a.data<int64_t>();
    auto* mb = mask_b.data<int64_t>();
    ma[0] = 1; ma[1] = 1;                // mask positions 0,1 of batch 0
    mb[T + 4] = 1; mb[T + 5] = 1;        // mask positions 4,5 of batch 1

    Tensor original_tokens({B, T}, DType::Int64, Device::cpu());
    original_tokens.fill_(42);

    auto outputs = model.forward(input_ids, mask_a, original_tokens);

    auto loss_a = model.compute_loss(outputs.gen_logits, outputs.disc_logits,
                                     outputs.is_replaced, mask_a, original_tokens);
    auto loss_b = model.compute_loss(outputs.gen_logits, outputs.disc_logits,
                                     outputs.is_replaced, mask_b, original_tokens);

    float va = loss_a.tensor().to(Device::cpu()).to(DType::Float32).item<float>();
    float vb = loss_b.tensor().to(Device::cpu()).to(DType::Float32).item<float>();

    EXPECT_TRUE(std::isfinite(va)) << "loss_a not finite: " << va;
    EXPECT_TRUE(std::isfinite(vb)) << "loss_b not finite: " << vb;
    EXPECT_GE(va, 0.0f) << "Real MLM cross-entropy must be non-negative, got " << va;
    EXPECT_GE(vb, 0.0f) << "Real MLM cross-entropy must be non-negative, got " << vb;

    // Different masks → different losses. With the old mean(log_probs) bug,
    // masked_positions was ignored entirely so va == vb to floating precision.
    EXPECT_GT(std::abs(va - vb), 1e-4f)
        << "Loss did not change when masked_positions changed — generator "
           "loss is probably still ignoring the mask. va=" << va << ", vb=" << vb;
}

// ============================================================================
// ElectraForSequenceClassification Tests
// ============================================================================

TEST_F(ElectraTest, SequenceClassificationCreation) {
    auto config = ElectraConfig::base();
    config.num_hidden_layers = 2;
    int64_t num_labels = 3;

    EXPECT_NO_THROW({
        ElectraForSequenceClassification model(config, num_labels);
    });
}

TEST_F(ElectraTest, SequenceClassificationForward) {
    auto config = ElectraConfig::base();
    config.num_hidden_layers = 2;
    int64_t num_labels = 3;
    ElectraForSequenceClassification model(config, num_labels);

    int64_t batch_size = 2;
    int64_t seq_len = 10;

    Tensor input_tensor({batch_size, seq_len}, DType::Int64, Device::cpu());
    input_tensor.fill_(42);
    Variable input_ids(input_tensor, false);

    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    // Check output shape: [batch_size, num_labels]
    EXPECT_EQ(logits.shape()[0], batch_size);
    EXPECT_EQ(logits.shape()[1], num_labels);
}

// ============================================================================
// ElectraForTokenClassification Tests
// ============================================================================

TEST_F(ElectraTest, TokenClassificationCreation) {
    auto config = ElectraConfig::base();
    config.num_hidden_layers = 2;
    int64_t num_labels = 9;

    EXPECT_NO_THROW({
        ElectraForTokenClassification model(config, num_labels);
    });
}

TEST_F(ElectraTest, TokenClassificationForward) {
    auto config = ElectraConfig::base();
    config.num_hidden_layers = 2;
    int64_t num_labels = 9;
    ElectraForTokenClassification model(config, num_labels);

    int64_t batch_size = 2;
    int64_t seq_len = 10;

    Tensor input_tensor({batch_size, seq_len}, DType::Int64, Device::cpu());
    input_tensor.fill_(42);
    Variable input_ids(input_tensor, false);

    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    // Check output shape: [batch_size, seq_len, num_labels]
    EXPECT_EQ(logits.shape()[0], batch_size);
    EXPECT_EQ(logits.shape()[1], seq_len);
    EXPECT_EQ(logits.shape()[2], num_labels);
}

// ============================================================================
// ElectraForQuestionAnswering Tests
// ============================================================================

TEST_F(ElectraTest, QuestionAnsweringCreation) {
    auto config = ElectraConfig::base();
    config.num_hidden_layers = 2;

    EXPECT_NO_THROW({
        ElectraForQuestionAnswering model(config);
    });
}

TEST_F(ElectraTest, QuestionAnsweringForward) {
    auto config = ElectraConfig::base();
    config.num_hidden_layers = 2;
    ElectraForQuestionAnswering model(config);

    int64_t batch_size = 2;
    int64_t seq_len = 10;

    Tensor input_tensor({batch_size, seq_len}, DType::Int64, Device::cpu());
    input_tensor.fill_(42);
    Variable input_ids(input_tensor, false);

    auto outputs = model.forward(input_ids, Tensor{}, Variable{});

    // Check output shapes
    EXPECT_EQ(outputs.start_logits.shape()[0], batch_size);
    EXPECT_EQ(outputs.start_logits.shape()[1], seq_len);

    EXPECT_EQ(outputs.end_logits.shape()[0], batch_size);
    EXPECT_EQ(outputs.end_logits.shape()[1], seq_len);
}

// ============================================================================
// Gradient Flow Tests
// ============================================================================

TEST_F(ElectraTest, GradientFlowSequenceClassification) {
    auto config = ElectraConfig::base();
    config.num_hidden_layers = 1;
    int64_t num_labels = 2;
    ElectraForSequenceClassification model(config, num_labels);

    int64_t batch_size = 2;
    int64_t seq_len = 8;

    Tensor input_tensor({batch_size, seq_len}, DType::Int64, Device::cpu());
    input_tensor.fill_(42);
    Variable input_ids(input_tensor, false);

    // Forward pass
    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    // Simple loss: sum of all outputs
    auto loss = sum(logits);

    // Backward pass
    EXPECT_NO_THROW({
        loss.backward();
    });
}

TEST_F(ElectraTest, GradientFlowPreTraining) {
    auto config = ElectraConfig::small();  // Use smaller config for speed
    config.num_hidden_layers = 1;
    config.generator_layers = 1;
    ElectraForPreTraining model(config);

    int64_t batch_size = 2;
    int64_t seq_len = 8;

    Tensor input_tensor({batch_size, seq_len}, DType::Int64, Device::cpu());
    input_tensor.fill_(42);
    Variable input_ids(input_tensor, false);

    Tensor masked_positions({batch_size, seq_len}, DType::Int64, Device::cpu());
    masked_positions.zero_();
    masked_positions.data<int64_t>()[0] = 1;

    Tensor original_tokens({batch_size, seq_len}, DType::Int64, Device::cpu());
    original_tokens.fill_(42);

    // Forward pass
    auto outputs = model.forward(input_ids, masked_positions, original_tokens);

    // Compute loss
    auto loss = model.compute_loss(
        outputs.gen_logits,
        outputs.disc_logits,
        outputs.is_replaced,
        masked_positions,
        original_tokens
    );

    // Backward pass
    EXPECT_NO_THROW({
        loss.backward();
    });
}

// ============================================================================
// Component Access Tests
// ============================================================================

TEST_F(ElectraTest, GetGeneratorDiscriminator) {
    auto config = ElectraConfig::base();
    config.num_hidden_layers = 2;
    config.generator_layers = 2;
    ElectraForPreTraining model(config);

    // Test access to components
    EXPECT_NO_THROW({
        auto generator = model.get_generator();
        EXPECT_NE(generator, nullptr);

        auto discriminator = model.get_discriminator();
        EXPECT_NE(discriminator, nullptr);
    });
}

TEST_F(ElectraTest, DiscriminatorBertModel) {
    auto config = ElectraConfig::base();
    config.num_hidden_layers = 2;
    ElectraDiscriminator discriminator(config);

    // Test access to underlying BERT model
    EXPECT_NO_THROW({
        auto bert = discriminator.get_bert_model();
        EXPECT_NE(bert, nullptr);
    });
}

// ============================================================================
// Sample Efficiency Test
// ============================================================================

TEST_F(ElectraTest, AllTokensUsedForTraining) {
    // ELECTRA trains on all tokens, not just masked ones (unlike BERT)
    auto config = ElectraConfig::small();
    config.num_hidden_layers = 1;
    config.generator_layers = 1;
    ElectraForPreTraining model(config);

    int64_t batch_size = 2;
    int64_t seq_len = 10;

    Tensor input_tensor({batch_size, seq_len}, DType::Int64, Device::cpu());
    input_tensor.fill_(42);
    Variable input_ids(input_tensor, false);

    // Only mask 2 tokens (20% of sequence)
    Tensor masked_positions({batch_size, seq_len}, DType::Int64, Device::cpu());
    masked_positions.zero_();
    auto mask_data = masked_positions.data<int64_t>();
    mask_data[0] = 1;
    mask_data[seq_len + 1] = 1;

    Tensor original_tokens({batch_size, seq_len}, DType::Int64, Device::cpu());
    original_tokens.fill_(42);

    auto outputs = model.forward(input_ids, masked_positions, original_tokens);

    // Discriminator should produce predictions for ALL tokens
    EXPECT_EQ(outputs.disc_logits.shape()[1], seq_len);

    // All tokens should have a prediction (real vs replaced)
    // This is the key difference from BERT which only trains on masked tokens
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST_F(ElectraTest, BenchmarkInference) {
    // Disabled by default - enable for performance testing
    auto config = ElectraConfig::base();
    ElectraForSequenceClassification model(config, 2);

    int64_t batch_size = 32;
    int64_t seq_len = 128;

    Tensor input_tensor({batch_size, seq_len}, DType::Int64, Device::cpu());
    input_tensor.fill_(42);
    Variable input_ids(input_tensor, false);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        model.forward(input_ids, Tensor{}, Variable{});
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    int num_iterations = 100;
    for (int i = 0; i < num_iterations; ++i) {
        model.forward(input_ids, Tensor{}, Variable{});
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Average inference time: " << duration.count() / num_iterations << " ms" << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    if (!::testing::GTEST_FLAG(list_tests)) {
        tenzor::initialize();  // Initialize Tenzor library and backends
    }
    return RUN_ALL_TESTS();
}
