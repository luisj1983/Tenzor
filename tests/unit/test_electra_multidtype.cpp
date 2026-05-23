/**
 * @file test_electra_multidtype.cpp
 * @brief Multi-dtype parameterized tests for ELECTRA models
 *
 * Tests ELECTRA model operations with different data types across
 * CPU, CUDA, OneAPI, Vulkan, and ROCm backends:
 * - Float32 (standard precision for NLP models - native format)
 * - Float64 (high precision - test conversion for scientific computing)
 * - Float16 (mixed precision training)
 *
 * ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately)
 * is an efficient pre-training method that uses a generator-discriminator setup.
 *
 * This test suite verifies:
 * 1. ELECTRA models work correctly with Float32 storage
 * 2. Outputs can be converted to other dtypes for computation
 * 3. Generator and discriminator components work together
 * 4. Replaced token detection functions correctly
 * 5. Different sequence lengths are handled properly
 * 6. Gradient flow works for all model variants
 * 7. Different model sizes (small, base, large) work correctly
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "tenzor/models/electra.hpp"
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::models;

// ============================================================================
// ELECTRA Multi-Backend Multi-DType Test Fixture
// ============================================================================

class ElectraMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    // Helper to convert output to requested dtype and verify
    Tensor convertAndVerifyOutput(const Variable& output, const std::vector<int64_t>& expected_shape) {
        auto output_tensor = output.tensor();

        // Verify shape
        auto shape = output_tensor.shape();
        EXPECT_EQ(shape.size(), expected_shape.size())
            << "Shape mismatch on " << device().to_string();
        for (size_t i = 0; i < shape.size(); ++i) {
            EXPECT_EQ(shape[i], expected_shape[i])
                << "Dimension " << i << " mismatch on " << device().to_string();
        }

        // Model output dtype should match the parameterized dtype (after model.to(dtype()) call)
        // For Float16, models may output Float32 for numerical stability
        if (dtype() == DType::Float16) {
            // Float16 models may output Float32 for stability, accept either
            EXPECT_TRUE(output_tensor.dtype() == DType::Float16 || output_tensor.dtype() == DType::Float32)
                << "ELECTRA Float16 output should be Float16 or Float32 on " << device().to_string();
        } else {
            EXPECT_EQ(output_tensor.dtype(), dtype())
                << "ELECTRA output dtype should match model dtype on " << device().to_string();
        }

        return output_tensor;
    }

    // Helper to create input token tensor
    Tensor createTokenInput(int64_t batch_size, int64_t seq_len, int64_t base_token = 42) {
        Tensor input_tensor({batch_size, seq_len}, DType::Int64, device());
        auto input_cpu = input_tensor.to(Device::cpu());
        auto input_data = input_cpu.data<int64_t>();
        for (int64_t i = 0; i < batch_size * seq_len; ++i) {
            input_data[i] = base_token + (i % 100);
        }

        if (device().type != Device::Type::CPU) {
            input_tensor = input_cpu.to(device());
        }
        return input_tensor;
    }
};

// ============================================================================
// ELECTRA Generator Tests
// ============================================================================

TEST_P(ElectraMultiDTypeTest, GeneratorForwardPass) {
    auto config = ElectraConfig::small();
    config.num_hidden_layers = 2;
    config.generator_layers = 2;
    ElectraGenerator generator(config);
    generator.to(device());
    if (dtype() != DType::Float32) generator.to(dtype());

    int64_t batch_size = 2;
    int64_t seq_len = 8;

    Tensor input_tensor = createTokenInput(batch_size, seq_len);
    Variable input_ids(input_tensor, false);
    auto logits = generator.forward(input_ids, Tensor{}, Variable{});

    // Verify output shape: [batch_size, seq_len, vocab_size]
    auto output = convertAndVerifyOutput(logits, {batch_size, seq_len, config.vocab_size});

    // Verify output is not all zeros or NaN
    auto output_cpu = output.to(Device::cpu());
    if (dtype() == DType::Float32) {
        auto data = output_cpu.data<float>();
        bool has_nonzero = false;
        for (int64_t i = 0; i < batch_size * seq_len * config.vocab_size; ++i) {
            EXPECT_FALSE(std::isnan(data[i])) << "NaN detected at index " << i;
            if (std::abs(data[i]) > 1e-6f) has_nonzero = true;
        }
        EXPECT_TRUE(has_nonzero) << "Output should contain non-zero values on " << device().to_string();
    } else if (dtype() == DType::Float64) {
        auto data = output_cpu.data<double>();
        bool has_nonzero = false;
        for (int64_t i = 0; i < batch_size * seq_len * config.vocab_size; ++i) {
            EXPECT_FALSE(std::isnan(data[i])) << "NaN detected at index " << i;
            if (std::abs(data[i]) > 1e-6) has_nonzero = true;
        }
        EXPECT_TRUE(has_nonzero) << "Output should contain non-zero values on " << device().to_string();
    }
}

TEST_P(ElectraMultiDTypeTest, GeneratorDifferentSequenceLengths) {
    auto config = ElectraConfig::small();
    config.num_hidden_layers = 1;
    config.generator_layers = 1;
    ElectraGenerator generator(config);
    generator.to(device());
    if (dtype() != DType::Float32) generator.to(dtype());

    // Test with different sequence lengths
    std::vector<int64_t> seq_lengths = {4, 8, 16};

    for (auto seq_len : seq_lengths) {
        int64_t batch_size = 2;

        Tensor input_tensor = createTokenInput(batch_size, seq_len);
        Variable input_ids(input_tensor, false);
        auto logits = generator.forward(input_ids, Tensor{}, Variable{});

        // Verify output shape adapts to sequence length
        convertAndVerifyOutput(logits, {batch_size, seq_len, config.vocab_size});
        expectFiniteNonZero(logits.tensor());
    }
}

// ============================================================================
// ELECTRA Discriminator Tests
// ============================================================================

TEST_P(ElectraMultiDTypeTest, DiscriminatorForwardPass) {
    auto config = ElectraConfig::small();
    config.num_hidden_layers = 2;
    ElectraDiscriminator discriminator(config);
    discriminator.to(device());
    if (dtype() != DType::Float32) discriminator.to(dtype());

    int64_t batch_size = 2;
    int64_t seq_len = 8;

    Tensor input_tensor = createTokenInput(batch_size, seq_len, 50);
    Variable input_ids(input_tensor, false);
    auto logits = discriminator.forward(input_ids, Tensor{}, Variable{});

    // Verify output shape: [batch_size, seq_len] (binary classification per token)
    auto output = convertAndVerifyOutput(logits, {batch_size, seq_len});

    // Verify outputs are reasonable (should be in logit range)
    auto output_cpu = output.to(Device::cpu());
    if (dtype() == DType::Float32) {
        auto data = output_cpu.data<float>();
        for (int64_t i = 0; i < batch_size * seq_len; ++i) {
            EXPECT_FALSE(std::isnan(data[i])) << "NaN at index " << i;
            EXPECT_GE(data[i], -20.0f) << "Logit too negative at " << i;
            EXPECT_LE(data[i], 20.0f) << "Logit too positive at " << i;
        }
    } else if (dtype() == DType::Float64) {
        auto data = output_cpu.data<double>();
        for (int64_t i = 0; i < batch_size * seq_len; ++i) {
            EXPECT_FALSE(std::isnan(data[i])) << "NaN at index " << i;
            EXPECT_GE(data[i], -20.0) << "Logit too negative at " << i;
            EXPECT_LE(data[i], 20.0) << "Logit too positive at " << i;
        }
    }
}

TEST_P(ElectraMultiDTypeTest, DiscriminatorDifferentSequenceLengths) {
    auto config = ElectraConfig::small();
    config.num_hidden_layers = 1;
    ElectraDiscriminator discriminator(config);
    discriminator.to(device());
    if (dtype() != DType::Float32) discriminator.to(dtype());

    std::vector<int64_t> seq_lengths = {4, 8, 12};

    for (auto seq_len : seq_lengths) {
        int64_t batch_size = 2;

        Tensor input_tensor = createTokenInput(batch_size, seq_len);
        Variable input_ids(input_tensor, false);
        auto logits = discriminator.forward(input_ids, Tensor{}, Variable{});

        // Output should have same sequence length
        convertAndVerifyOutput(logits, {batch_size, seq_len});
        expectFiniteNonZero(logits.tensor());
    }
}

// ============================================================================
// ELECTRA Pre-Training Tests (Generator + Discriminator)
// ============================================================================

TEST_P(ElectraMultiDTypeTest, PreTrainingReplacedTokenDetection) {
    auto config = ElectraConfig::small();
    config.num_hidden_layers = 1;
    config.generator_layers = 1;
    ElectraForPreTraining model(config);
    model.to(device());
    if (dtype() != DType::Float32) model.to(dtype());

    int64_t batch_size = 2;
    int64_t seq_len = 8;

    // Create input with masked positions
    Tensor input_tensor = createTokenInput(batch_size, seq_len);
    auto input_cpu = input_tensor.to(Device::cpu());
    auto input_data = input_cpu.data<int64_t>();
    input_data[3] = 103;  // [MASK] token at position 3
    input_data[seq_len + 5] = 103;  // [MASK] at position 5 in second batch

    if (device().type != Device::Type::CPU) {
        input_tensor = input_cpu.to(device());
    }

    // Create masked positions tensor
    Tensor masked_positions({batch_size, seq_len}, DType::Int64, device());
    auto mask_cpu = masked_positions.to(Device::cpu());
    mask_cpu.zero_();
    auto mask_data = mask_cpu.data<int64_t>();
    mask_data[3] = 1;
    mask_data[seq_len + 5] = 1;

    if (device().type != Device::Type::CPU) {
        masked_positions = mask_cpu.to(device());
    }

    // Create original tokens
    Tensor original_tokens({batch_size, seq_len}, DType::Int64, device());
    auto orig_cpu = original_tokens.to(Device::cpu());
    orig_cpu.fill_(42);
    auto orig_data = orig_cpu.data<int64_t>();
    orig_data[3] = 100;
    orig_data[seq_len + 5] = 200;

    if (device().type != Device::Type::CPU) {
        original_tokens = orig_cpu.to(device());
    }

    Variable input_ids(input_tensor, false);
    auto outputs = model.forward(input_ids, masked_positions, original_tokens);

    // Verify generator logits
    convertAndVerifyOutput(outputs.gen_logits, {batch_size, seq_len, config.vocab_size});

    // Verify discriminator logits
    convertAndVerifyOutput(outputs.disc_logits, {batch_size, seq_len});

    // Verify is_replaced tensor exists and has correct shape
    EXPECT_EQ(outputs.is_replaced.shape()[0], batch_size);
    EXPECT_EQ(outputs.is_replaced.shape()[1], seq_len);
    expectFiniteNonZero(outputs.gen_logits.tensor());
    expectFiniteNonZero(outputs.disc_logits.tensor());
}

TEST_P(ElectraMultiDTypeTest, PreTrainingAllTokensUsed) {
    // ELECTRA trains on all tokens, not just masked ones (unlike BERT)
    auto config = ElectraConfig::small();
    config.num_hidden_layers = 1;
    config.generator_layers = 1;
    ElectraForPreTraining model(config);
    model.to(device());
    if (dtype() != DType::Float32) model.to(dtype());

    int64_t batch_size = 2;
    int64_t seq_len = 10;

    Tensor input_tensor = createTokenInput(batch_size, seq_len);

    // Only mask 2 tokens (20% of sequence)
    Tensor masked_positions({batch_size, seq_len}, DType::Int64, device());
    auto mask_cpu = masked_positions.to(Device::cpu());
    mask_cpu.zero_();
    auto mask_data = mask_cpu.data<int64_t>();
    mask_data[0] = 1;
    mask_data[seq_len + 1] = 1;

    if (device().type != Device::Type::CPU) {
        masked_positions = mask_cpu.to(device());
    }

    Tensor original_tokens({batch_size, seq_len}, DType::Int64, device());
    auto orig_cpu = original_tokens.to(Device::cpu());
    orig_cpu.fill_(42);

    if (device().type != Device::Type::CPU) {
        original_tokens = orig_cpu.to(device());
    }

    Variable input_ids(input_tensor, false);
    auto outputs = model.forward(input_ids, masked_positions, original_tokens);

    // Discriminator should produce predictions for ALL tokens
    auto disc_output = convertAndVerifyOutput(outputs.disc_logits, {batch_size, seq_len});

    // Verify all positions have predictions (key ELECTRA feature)
    EXPECT_EQ(disc_output.shape()[1], seq_len)
        << "All tokens should have discriminator predictions on " << device().to_string();
    expectFiniteNonZero(outputs.gen_logits.tensor());
    expectFiniteNonZero(outputs.disc_logits.tensor());
}

TEST_P(ElectraMultiDTypeTest, PreTrainingLossComputation) {
    auto config = ElectraConfig::small();
    config.num_hidden_layers = 1;
    config.generator_layers = 1;
    ElectraForPreTraining model(config);
    model.to(device());
    if (dtype() != DType::Float32) model.to(dtype());

    int64_t batch_size = 2;
    int64_t seq_len = 8;

    Tensor input_tensor = createTokenInput(batch_size, seq_len);

    Tensor masked_positions({batch_size, seq_len}, DType::Int64, device());
    auto mask_cpu = masked_positions.to(Device::cpu());
    mask_cpu.zero_();
    auto mask_data = mask_cpu.data<int64_t>();
    mask_data[0] = 1;
    mask_data[seq_len + 2] = 1;

    if (device().type != Device::Type::CPU) {
        masked_positions = mask_cpu.to(device());
    }

    Tensor original_tokens({batch_size, seq_len}, DType::Int64, device());
    auto orig_cpu = original_tokens.to(Device::cpu());
    orig_cpu.fill_(42);

    if (device().type != Device::Type::CPU) {
        original_tokens = orig_cpu.to(device());
    }

    Variable input_ids(input_tensor, false);
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
        EXPECT_EQ(loss.shape().size(), 0) << "Loss should be scalar on " << device().to_string();

        // Verify loss is finite and positive
        auto loss_cpu = loss.tensor().to(Device::cpu());
        if (dtype() == DType::Float32) {
            float loss_val = loss_cpu.data<float>()[0];
            EXPECT_FALSE(std::isnan(loss_val)) << "Loss is NaN on " << device().to_string();
            EXPECT_FALSE(std::isinf(loss_val)) << "Loss is infinite on " << device().to_string();
            EXPECT_GT(loss_val, 0.0f) << "Loss should be positive on " << device().to_string();
        } else if (dtype() == DType::Float64) {
            auto converted = loss_cpu.to(dtype());
            double loss_val = converted.data<double>()[0];
            EXPECT_FALSE(std::isnan(loss_val)) << "Loss is NaN on " << device().to_string();
            EXPECT_FALSE(std::isinf(loss_val)) << "Loss is infinite on " << device().to_string();
            EXPECT_GT(loss_val, 0.0) << "Loss should be positive on " << device().to_string();
        }
    });
}

// ============================================================================
// ELECTRA Model Size Tests
// ============================================================================

TEST_P(ElectraMultiDTypeTest, SmallModelConfiguration) {
    auto config = ElectraConfig::small();

    EXPECT_EQ(config.hidden_size, 256);
    EXPECT_EQ(config.num_hidden_layers, 12);
    EXPECT_EQ(config.num_attention_heads, 4);
    EXPECT_EQ(config.generator_hidden_size, 128);

    // Test model creation
    config.num_hidden_layers = 1;
    config.generator_layers = 1;
    EXPECT_NO_THROW({
        ElectraForPreTraining model(config);
    });
}

TEST_P(ElectraMultiDTypeTest, BaseModelConfiguration) {
    auto config = ElectraConfig::base();

    EXPECT_EQ(config.hidden_size, 768);
    EXPECT_EQ(config.num_hidden_layers, 12);
    EXPECT_EQ(config.num_attention_heads, 12);
    EXPECT_EQ(config.generator_hidden_size, 256);

    // Test model creation with reduced layers
    config.num_hidden_layers = 2;
    config.generator_layers = 2;
    EXPECT_NO_THROW({
        ElectraForPreTraining model(config);
    });
}

TEST_P(ElectraMultiDTypeTest, LargeModelConfiguration) {
    auto config = ElectraConfig::large();

    EXPECT_EQ(config.hidden_size, 1024);
    EXPECT_EQ(config.num_hidden_layers, 24);
    EXPECT_EQ(config.num_attention_heads, 16);

    // Test config conversion
    auto disc_config = config.to_discriminator_config();
    EXPECT_EQ(disc_config.hidden_size, 1024);

    auto gen_config = config.to_generator_config();
    EXPECT_EQ(gen_config.hidden_size, 256);
}

// ============================================================================
// ELECTRA Sequence Classification Tests
// ============================================================================

TEST_P(ElectraMultiDTypeTest, SequenceClassificationForward) {
    auto config = ElectraConfig::small();
    config.num_hidden_layers = 2;
    int64_t num_labels = 3;
    ElectraForSequenceClassification model(config, num_labels);
    model.to(device());
    if (dtype() != DType::Float32) model.to(dtype());

    int64_t batch_size = 2;
    int64_t seq_len = 10;

    Tensor input_tensor = createTokenInput(batch_size, seq_len);
    Variable input_ids(input_tensor, false);
    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    // Verify output shape: [batch_size, num_labels]
    convertAndVerifyOutput(logits, {batch_size, num_labels});
    expectFiniteNonZero(logits.tensor());
}

TEST_P(ElectraMultiDTypeTest, SequenceClassificationGradientFlow) {
    auto config = ElectraConfig::small();
    config.num_hidden_layers = 1;
    int64_t num_labels = 2;
    ElectraForSequenceClassification model(config, num_labels);
    model.to(device());
    if (dtype() != DType::Float32) model.to(dtype());

    int64_t batch_size = 2;
    int64_t seq_len = 8;

    Tensor input_tensor = createTokenInput(batch_size, seq_len);
    Variable input_ids(input_tensor, false);
    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    // Compute simple loss
    auto loss = sum(logits);
    loss.backward();

    // Strengthened from EXPECT_NO_THROW: verify at least one model parameter
    // actually received a non-zero gradient. The prior check would pass even
    // if every gradient silently zeroed out (the documented failure mode of
    // raw-tensor-op autograd breaks in model code).
    float max_abs_grad = 0.0f;
    for (const auto& param : model.parameters()) {
        if (param->has_grad()) {
            auto g = max(abs(param->grad()->to(Device::cpu()).to(DType::Float32)));
            max_abs_grad = std::max(max_abs_grad, g.item<float>());
        }
    }
    EXPECT_GT(max_abs_grad, 0.0f)
        << "ELECTRA discriminator grads all-zero on " << device().to_string();
}

// ============================================================================
// ELECTRA Token Classification Tests
// ============================================================================

TEST_P(ElectraMultiDTypeTest, TokenClassificationForward) {
    auto config = ElectraConfig::small();
    config.num_hidden_layers = 2;
    int64_t num_labels = 9;  // NER typically has ~9 labels
    ElectraForTokenClassification model(config, num_labels);
    model.to(device());
    if (dtype() != DType::Float32) model.to(dtype());

    int64_t batch_size = 2;
    int64_t seq_len = 10;

    Tensor input_tensor = createTokenInput(batch_size, seq_len);
    Variable input_ids(input_tensor, false);
    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    // Verify output shape: [batch_size, seq_len, num_labels]
    convertAndVerifyOutput(logits, {batch_size, seq_len, num_labels});
    expectFiniteNonZero(logits.tensor());
}

// ============================================================================
// ELECTRA Question Answering Tests
// ============================================================================

TEST_P(ElectraMultiDTypeTest, QuestionAnsweringForward) {
    auto config = ElectraConfig::small();
    config.num_hidden_layers = 2;
    ElectraForQuestionAnswering model(config);
    model.to(device());
    if (dtype() != DType::Float32) model.to(dtype());

    int64_t batch_size = 2;
    int64_t seq_len = 10;

    Tensor input_tensor = createTokenInput(batch_size, seq_len);
    Variable input_ids(input_tensor, false);
    auto outputs = model.forward(input_ids, Tensor{}, Variable{});

    // Verify start and end logits
    convertAndVerifyOutput(outputs.start_logits, {batch_size, seq_len});
    convertAndVerifyOutput(outputs.end_logits, {batch_size, seq_len});
    expectFiniteNonZero(outputs.start_logits.tensor());
    expectFiniteNonZero(outputs.end_logits.tensor());
}

// ============================================================================
// DType Conversion Tests
// ============================================================================

TEST_P(ElectraMultiDTypeTest, DTypeConversionAccuracy) {
    if (dtype() == DType::Float32) {
        GTEST_SKIP() << "Skipping conversion test for Float32";
    }

    auto config = ElectraConfig::small();
    config.num_hidden_layers = 1;
    ElectraDiscriminator discriminator(config);
    discriminator.to(device());
    if (dtype() != DType::Float32) discriminator.to(dtype());

    int64_t batch_size = 2;
    int64_t seq_len = 8;

    Tensor input_tensor = createTokenInput(batch_size, seq_len);
    Variable input_ids(input_tensor, false);
    auto logits = discriminator.forward(input_ids, Tensor{}, Variable{});

    // Get output and convert to Float32 for comparison (model outputs in target dtype)
    auto output_cpu = logits.tensor().to(Device::cpu());
    auto output_f32 = output_cpu.to(DType::Float32);

    // Convert to target dtype and back
    auto output_converted = output_f32.to(dtype());
    auto output_back = output_converted.to(DType::Float32);

    // Verify conversion preserves values (within tolerance)
    auto data_original = output_f32.data<float>();
    auto data_converted = output_back.data<float>();

    double max_diff = 0.0;
    for (int64_t i = 0; i < batch_size * seq_len; ++i) {
        double diff = std::abs(static_cast<double>(data_original[i]) -
                              static_cast<double>(data_converted[i]));
        max_diff = std::max(max_diff, diff);
    }

    // Float64 should have very high precision
    // Float16 will have some precision loss
    double tolerance = (dtype() == DType::Float64) ? 1e-7 : 1e-2;
    EXPECT_LT(max_diff, tolerance)
        << "DType conversion lost too much precision on " << device().to_string();
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ElectraMultiDTypeTest);

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    if (!::testing::GTEST_FLAG(list_tests)) {
        tenzor::initialize();
    }
    return RUN_ALL_TESTS();
}
