/**
 * @file electra.cpp
 * @brief Implementation of ELECTRA model family
 */

#include "tenzor/models/electra.hpp"
#include "tenzor/models/hub.hpp"  // Audit H4
#include "tenzor/core/tensor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include <cstring>
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"  // Audit G13: tenzor::sum on Tensor (for mask count)
#include <random>
#include <cmath>

namespace tenzor {
namespace models {

// ============================================================================
// ElectraGenerator Implementation
// ============================================================================

ElectraGenerator::ElectraGenerator(const ElectraConfig& config)
    : config_(config) {
    // Create small BERT model for generator
    auto gen_config = config.to_generator_config();
    generator_ = std::make_shared<BertModel>(gen_config);

    // Language model head for token prediction
    lm_head_ = std::make_shared<nn::Linear>(
        config.generator_hidden_size, config.vocab_size);

    register_module("generator", generator_);
    register_module("lm_head", lm_head_);
}

auto ElectraGenerator::forward(const Variable& input_ids,
                                const Tensor& attention_mask,
                                const Variable& token_type_ids) -> Variable {
    // Call forward pre-hooks (enables CPU-start offloading)
    // NOTE: This is necessary because this multi-argument forward bypasses Module::forward()
    call_forward_pre_hooks();

    // Get generator outputs
    auto outputs = generator_->forward(input_ids, attention_mask, token_type_ids, Variable{});

    // Predict tokens for all positions
    auto logits = lm_head_->forward(outputs.sequence_output);

    // Call forward post-hooks (enables CPU-start offloading)
    call_forward_post_hooks();

    return logits;  // [batch, seq_len, vocab_size]
}

auto ElectraGenerator::forward_impl(const Variable& input) -> Variable {
    return forward(input, Tensor{}, Variable{});
}

// ============================================================================
// ElectraDiscriminator Implementation
// ============================================================================

ElectraDiscriminator::ElectraDiscriminator(const ElectraConfig& config)
    : config_(config) {
    // Create full BERT model for discriminator
    auto disc_config = config.to_discriminator_config();
    discriminator_ = std::make_shared<BertModel>(disc_config);

    // Binary classification head (real vs replaced)
    classifier_ = std::make_shared<nn::Linear>(config.hidden_size, 1);

    register_module("discriminator", discriminator_);
    register_module("classifier", classifier_);
}

auto ElectraDiscriminator::forward(const Variable& input_ids,
                                    const Tensor& attention_mask,
                                    const Variable& token_type_ids) -> Variable {
    // Call forward pre-hooks (enables CPU-start offloading)
    // NOTE: This is necessary because this multi-argument forward bypasses Module::forward()
    call_forward_pre_hooks();

    // Get discriminator outputs
    auto outputs = discriminator_->forward(input_ids, attention_mask, token_type_ids, Variable{});

    // Classify each token as real (0) or replaced (1)
    auto logits = classifier_->forward(outputs.sequence_output);

    // Squeeze last dimension: [batch, seq_len, 1] -> [batch, seq_len]
    auto shape = logits.shape();
    int64_t batch_size = shape[0];
    int64_t seq_len = shape[1];

    auto squeezed = tenzor::reshape(logits, {batch_size, seq_len});

    // Call forward post-hooks (enables CPU-start offloading)
    call_forward_post_hooks();

    return squeezed;
}

auto ElectraDiscriminator::forward_impl(const Variable& input) -> Variable {
    return forward(input, Tensor{}, Variable{});
}

// ============================================================================
// ElectraForPreTraining Implementation
// ============================================================================

ElectraForPreTraining::ElectraForPreTraining(const ElectraConfig& config)
    : config_(config) {
    // Create generator and discriminator
    generator_ = std::make_shared<ElectraGenerator>(config);
    discriminator_ = std::make_shared<ElectraDiscriminator>(config);

    register_module("generator", generator_);
    register_module("discriminator", discriminator_);
}

auto ElectraForPreTraining::sample_from_distribution(const float* probs, int64_t size) -> int64_t {
    // Sample token index from probability distribution
    // Using discrete distribution for sampling
    std::discrete_distribution<int64_t> dist(probs, probs + size);
    static std::mt19937 gen{42};  // Fixed seed for reproducibility
    return dist(gen);
}

auto ElectraForPreTraining::forward(const Variable& input_ids,
                                    const Tensor& masked_positions,
                                    const Tensor& original_tokens,
                                    const Tensor& attention_mask) -> ElectraPreTrainingOutput {
    // Call forward pre-hooks (enables CPU-start offloading)
    call_forward_pre_hooks();

    int64_t batch_size = input_ids.shape()[0];
    int64_t seq_len = input_ids.shape()[1];

    // Step 1: Generator predicts masked tokens
    auto gen_logits = generator_->forward(input_ids, attention_mask);  // [batch, seq_len, vocab]

    // Step 2: Sample tokens from generator predictions (at masked positions)
    auto gen_probs = nn::softmax(gen_logits, -1);

    // Create corrupted input by replacing masked tokens with generator samples
    Tensor generated_tokens = input_ids.tensor().clone();  // Copy original
    auto shape_vec = std::vector<int64_t>(input_ids.shape().begin(), input_ids.shape().end());

    // Use the same dtype as gen_probs for consistency
    auto dtype = gen_probs.tensor().dtype();
    Device target_device = input_ids.tensor().device();

    // Move tensors to CPU for data access
    Tensor masked_positions_cpu = masked_positions;
    if (masked_positions.device() != Device::cpu()) {
        masked_positions_cpu = masked_positions.to(Device::cpu());
    }
    Tensor original_tokens_cpu = original_tokens;
    if (original_tokens.device() != Device::cpu()) {
        original_tokens_cpu = original_tokens.to(Device::cpu());
    }
    Tensor generated_tokens_cpu = generated_tokens;
    if (generated_tokens.device() != Device::cpu()) {
        generated_tokens_cpu = generated_tokens.to(Device::cpu());
    }
    Tensor gen_probs_cpu = gen_probs.tensor();
    if (gen_probs.tensor().device() != Device::cpu()) {
        gen_probs_cpu = gen_probs.tensor().to(Device::cpu());
    }

    // Create is_replaced on CPU
    Tensor is_replaced_cpu(shape_vec, dtype, Device::cpu());
    is_replaced_cpu.zero_();

    const int64_t* mask_data = masked_positions_cpu.data<int64_t>();
    const int64_t* orig_data = original_tokens_cpu.data<int64_t>();
    int64_t* gen_data = generated_tokens_cpu.data<int64_t>();

    // Sample from generator for masked positions with dtype-specific handling
    // For sampling, we convert probabilities to float32 as it doesn't require high precision
    std::vector<float> float_probs(config_.vocab_size);

    if (dtype == DType::Float32) {
        float* repl_data = is_replaced_cpu.data<float>();
        auto probs_data = gen_probs_cpu.data<float>();

        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t s = 0; s < seq_len; ++s) {
                int64_t idx = b * seq_len + s;

                if (mask_data[idx] == 1) {  // This position was masked
                    const float* pos_probs = probs_data + idx * config_.vocab_size;
                    int64_t sampled_token = sample_from_distribution(pos_probs, config_.vocab_size);
                    gen_data[idx] = sampled_token;
                    repl_data[idx] = (sampled_token != orig_data[idx]) ? 1.0f : 0.0f;
                }
            }
        }
    } else if (dtype == DType::Float64) {
        double* repl_data = is_replaced_cpu.data<double>();
        auto probs_data = gen_probs_cpu.data<double>();

        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t s = 0; s < seq_len; ++s) {
                int64_t idx = b * seq_len + s;

                if (mask_data[idx] == 1) {
                    // Convert double probabilities to float for sampling
                    const double* pos_probs = probs_data + idx * config_.vocab_size;
                    for (int64_t i = 0; i < config_.vocab_size; ++i) {
                        float_probs[i] = static_cast<float>(pos_probs[i]);
                    }
                    int64_t sampled_token = sample_from_distribution(float_probs.data(), config_.vocab_size);
                    gen_data[idx] = sampled_token;
                    repl_data[idx] = (sampled_token != orig_data[idx]) ? 1.0 : 0.0;
                }
            }
        }
    } else if (dtype == DType::Float16) {
        Float16* repl_data = is_replaced_cpu.data<Float16>();
        auto probs_data = gen_probs_cpu.data<Float16>();
        Float16 zero_f16(static_cast<uint16_t>(0x0000));  // Float16 representation of 0.0
        Float16 one_f16(static_cast<uint16_t>(0x3C00));   // Float16 representation of 1.0

        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t s = 0; s < seq_len; ++s) {
                int64_t idx = b * seq_len + s;

                if (mask_data[idx] == 1) {
                    // Convert float16 probabilities to float for sampling
                    const Float16* pos_probs = probs_data + idx * config_.vocab_size;
                    for (int64_t i = 0; i < config_.vocab_size; ++i) {
                        // Use Float16's conversion operator to float
                        float_probs[i] = static_cast<float>(pos_probs[i]);
                    }
                    int64_t sampled_token = sample_from_distribution(float_probs.data(), config_.vocab_size);
                    gen_data[idx] = sampled_token;
                    repl_data[idx] = (sampled_token != orig_data[idx]) ? one_f16 : zero_f16;
                }
            }
        }
    }

    // Transfer back to original device
    Tensor is_replaced = (target_device == Device::cpu())
        ? is_replaced_cpu
        : is_replaced_cpu.to(target_device);

    // Update generated_tokens on the original device
    if (target_device != Device::cpu()) {
        generated_tokens = generated_tokens_cpu.to(target_device);
    } else {
        generated_tokens = generated_tokens_cpu;
    }

    // Step 3: Discriminator classifies all tokens as real/replaced
    Variable gen_input(generated_tokens, false);  // Don't need gradients for input tensor
    auto disc_logits = discriminator_->forward(gen_input, attention_mask);  // [batch, seq_len]

    // Call forward post-hooks (enables CPU-start offloading)
    call_forward_post_hooks();

    return ElectraPreTrainingOutput{gen_logits, disc_logits, is_replaced};
}

auto ElectraForPreTraining::forward_impl(const Variable& input) -> Variable {
    // Single-input Module contract: the full ELECTRA pre-training pipeline
    // (generator MLM -> token replacement -> discriminator RTD) requires
    // masked_positions and original_tokens, which the one-argument Module
    // interface cannot supply. We therefore run the discriminator directly on
    // the input ids and return its replaced-token-detection logits — the head
    // that is kept and used for downstream fine-tuning. Callers performing
    // pre-training use the multi-output forward(input_ids, masked_positions,
    // original_tokens) overload, which returns {gen_logits, disc_logits,
    // is_replaced}.
    return discriminator_->forward(input, Tensor{}, Variable{});
}

auto ElectraForPreTraining::load_pretrained(const std::string& path, bool strict) -> void {
    // Audit H4. See AlbertModel::load_pretrained.
    ModelHub::load_pretrained_weights(*this, path, strict);
}

auto ElectraForPreTraining::compute_loss(const Variable& gen_logits,
                                         const Variable& disc_logits,
                                         const Tensor& is_replaced,
                                         [[maybe_unused]] const Tensor& masked_positions,
                                         const Tensor& original_tokens) -> Variable {
    int64_t batch_size = gen_logits.shape()[0];
    int64_t seq_len = gen_logits.shape()[1];
    int64_t vocab_size = gen_logits.shape()[2];

    // Generator loss: MLM loss on masked tokens only
    // Compute cross-entropy loss for masked positions
    auto reshaped_logits = tenzor::reshape(gen_logits, {batch_size * seq_len, vocab_size});

    // Create labels tensor - move to CPU for data access, then transfer to device
    Tensor original_tokens_cpu = original_tokens;
    if (original_tokens.device() != Device::cpu()) {
        original_tokens_cpu = original_tokens.to(Device::cpu());
    }
    Tensor labels_cpu(std::vector<int64_t>{batch_size * seq_len}, DType::Int64, Device::cpu());
    std::copy_n(original_tokens_cpu.data<int64_t>(), batch_size * seq_len, labels_cpu.data<int64_t>());
    Tensor labels_flat = (original_tokens.device() == Device::cpu())
        ? labels_cpu
        : labels_cpu.to(original_tokens.device());
    Variable labels_var(labels_flat, false);

    // Audit G13: real MLM cross-entropy at masked positions only.
    //
    // Previous code computed `auto gen_loss = mean(log_probs);` over EVERY
    // (B*T, V) entry — that's just the average log-prob over the whole vocab
    // and is ≈ -log(V) ≈ constant, providing no learning signal for the
    // generator.
    //
    // Correct MLM loss: for each *masked* position p, compute
    //   loss_p = -log_softmax(logits_p)[original_token_p]
    // then average over the count of masked positions (not all positions).
    //
    // Implementation: gather log-probs at the true labels via
    // `autograd::gather(log_probs, dim=1, labels)` → per-token NLL,
    // multiply by the (B*T,) mask of masked positions, divide by the count
    // of masked positions (clamped ≥1 to avoid div-by-zero when no tokens
    // were masked in this batch).
    auto log_probs = nn::log_softmax(reshaped_logits, 1);  // (B*T, V)

    // Discriminator loss: Binary cross-entropy on ALL tokens
    // Convert is_replaced to Variable
    Variable is_replaced_var(is_replaced, false);

    // Compute binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
    // where p = sigmoid(disc_logits)
    auto sigmoid_logits = nn::sigmoid(disc_logits);
    auto log_prob_real = tenzor::log(sigmoid_logits);
    auto ones_var = Variable(ones_like(sigmoid_logits.tensor()), false);
    auto log_prob_fake = tenzor::log(ones_var - sigmoid_logits);

    auto ones_repl = Variable(ones_like(is_replaced), false);
    auto disc_loss = tenzor::neg(is_replaced_var * log_prob_fake +
                       (ones_repl - is_replaced_var) * log_prob_real);

    // Mean over all tokens
    disc_loss = tenzor::mean(disc_loss);

    // Audit G13: real MLM loss on masked positions only.
    //
    // 1. Gather log-prob at the true token for every position: (B*T, 1) → (B*T,).
    // 2. Negate → per-position NLL.
    // 3. Build a (B*T,) float mask from `masked_positions` (preferred) or
    //    `is_replaced` (fallback when caller passed an empty masked_positions
    //    — still a reasonable proxy: every replaced position must have been
    //    a masked position).
    // 4. Multiply NLL by mask, sum, divide by count of masked positions
    //    (clamped ≥1 to avoid 0/0 when nothing was masked).
    const int64_t flat_n = batch_size * seq_len;
    Tensor labels_2d = labels_flat.reshape({flat_n, 1});
    auto gathered = tenzor::gather(log_probs, /*dim=*/1, labels_2d);       // (B*T, 1)
    auto nll_per_pos = tenzor::neg(gathered.reshape({flat_n}));            // (B*T,)

    const DType float_dtype = gen_logits.dtype();
    Tensor mask_flat;
    if (masked_positions.is_valid() && masked_positions.numel() == flat_n) {
        mask_flat = masked_positions.reshape({flat_n}).to(float_dtype).to(gen_logits.device());
    } else if (masked_positions.is_valid() && masked_positions.numel() > 0) {
        // Shape mismatch — fall back to is_replaced rather than crash.
        mask_flat = is_replaced.reshape({flat_n}).to(float_dtype).to(gen_logits.device());
    } else {
        mask_flat = is_replaced.reshape({flat_n}).to(float_dtype).to(gen_logits.device());
    }
    Variable mask_var(mask_flat, false);

    auto masked_nll = nll_per_pos * mask_var;                              // (B*T,)
    Variable total_nll = tenzor::sum(masked_nll);                          // scalar

    // Count masked positions on CPU (one-time scalar reduction).
    Tensor mask_sum_cpu = tenzor::sum(mask_flat).to(Device::cpu()).to(DType::Float32);
    float num_masked = mask_sum_cpu.item<float>();
    double divisor = static_cast<double>(std::max(num_masked, 1.0f));

    auto gen_loss = total_nll * (1.0 / divisor);                           // scalar Variable

    // Combined loss (weighted sum). gen_loss is already the MLM cross-entropy
    // averaged over masked positions, so it's positive and ready to use — no
    // extra negation needed (unlike the previous mean-of-log-probs which was
    // negative and needed to be flipped).
    auto gen_component  = gen_loss * config_.gen_loss_weight;
    auto disc_component = disc_loss * config_.disc_loss_weight;
    auto total_loss = gen_component + disc_component;

    return total_loss;
}

// ============================================================================
// ElectraForSequenceClassification Implementation
// ============================================================================

ElectraForSequenceClassification::ElectraForSequenceClassification(
    const ElectraConfig& config, int64_t num_labels)
    : config_(config), num_labels_(num_labels) {
    // Use discriminator's BERT model
    auto bert_config = config.to_discriminator_config();
    bert_ = std::make_shared<BertModel>(bert_config);

    // Classification head
    dropout_ = std::make_shared<nn::Dropout>(config.hidden_dropout_prob);
    classifier_ = std::make_shared<nn::Linear>(config.hidden_size, num_labels);

    register_module("electra", bert_);
    register_module("dropout", dropout_);
    register_module("classifier", classifier_);
}

auto ElectraForSequenceClassification::forward(const Variable& input_ids,
                                                const Tensor& attention_mask,
                                                const Variable& token_type_ids) -> Variable {
    // Call forward pre-hooks (enables CPU-start offloading)
    call_forward_pre_hooks();

    // Get BERT outputs
    auto outputs = bert_->forward(input_ids, attention_mask, token_type_ids, Variable{});

    // Use pooled output
    auto pooled_output = outputs.pooled_output;

    // Apply dropout
    pooled_output = dropout_->forward(pooled_output);

    // Classify
    auto logits = classifier_->forward(pooled_output);

    // Call forward post-hooks (enables CPU-start offloading)
    call_forward_post_hooks();

    return logits;
}

auto ElectraForSequenceClassification::forward_impl(const Variable& input) -> Variable {
    return forward(input, Tensor{}, Variable{});
}

// ============================================================================
// ElectraForTokenClassification Implementation
// ============================================================================

ElectraForTokenClassification::ElectraForTokenClassification(
    const ElectraConfig& config, int64_t num_labels)
    : config_(config), num_labels_(num_labels) {
    // Use discriminator's BERT model
    auto bert_config = config.to_discriminator_config();
    bert_ = std::make_shared<BertModel>(bert_config);

    // Token classification head
    dropout_ = std::make_shared<nn::Dropout>(config.hidden_dropout_prob);
    classifier_ = std::make_shared<nn::Linear>(config.hidden_size, num_labels);

    register_module("electra", bert_);
    register_module("dropout", dropout_);
    register_module("classifier", classifier_);
}

auto ElectraForTokenClassification::forward(const Variable& input_ids,
                                             const Tensor& attention_mask,
                                             const Variable& token_type_ids) -> Variable {
    // Call forward pre-hooks (enables CPU-start offloading)
    call_forward_pre_hooks();

    // Get BERT outputs
    auto outputs = bert_->forward(input_ids, attention_mask, token_type_ids, Variable{});

    // Use sequence output (token-level representations)
    auto sequence_output = outputs.sequence_output;

    // Apply dropout
    sequence_output = dropout_->forward(sequence_output);

    // Classify each token
    auto logits = classifier_->forward(sequence_output);

    // Call forward post-hooks (enables CPU-start offloading)
    call_forward_post_hooks();

    return logits;
}

auto ElectraForTokenClassification::forward_impl(const Variable& input) -> Variable {
    return forward(input, Tensor{}, Variable{});
}

// ============================================================================
// ElectraForQuestionAnswering Implementation
// ============================================================================

ElectraForQuestionAnswering::ElectraForQuestionAnswering(const ElectraConfig& config)
    : config_(config) {
    // Use discriminator's BERT model
    auto bert_config = config.to_discriminator_config();
    bert_ = std::make_shared<BertModel>(bert_config);

    // QA output layer (predicts start and end positions)
    qa_outputs_ = std::make_shared<nn::Linear>(config.hidden_size, 2);

    register_module("electra", bert_);
    register_module("qa_outputs", qa_outputs_);
}

auto ElectraForQuestionAnswering::forward(const Variable& input_ids,
                                          const Tensor& attention_mask,
                                          const Variable& token_type_ids) -> ElectraQAOutput {
    // Call forward pre-hooks (enables CPU-start offloading)
    call_forward_pre_hooks();

    // Get BERT outputs
    auto outputs = bert_->forward(input_ids, attention_mask, token_type_ids, Variable{});

    // Use sequence output
    auto sequence_output = outputs.sequence_output;

    // Predict start and end logits
    auto logits = qa_outputs_->forward(sequence_output);

    // Split into start and end logits while preserving gradients
    // logits: [batch, seq_len, 2]
    auto shape = logits.shape();
    int64_t batch_size = shape[0];
    int64_t seq_len = shape[1];

    // Reshape to [batch * seq_len, 2]
    auto reshaped = tenzor::reshape(logits, {batch_size * seq_len, 2});

    // Create selection matrices to extract start and end logits
    // Use the same dtype as logits for consistency
    auto dtype = logits.tensor().dtype();
    Device target_device = logits.tensor().device();

    // Start logits: multiply by [1, 0] - create on CPU first, then transfer
    Tensor start_selector_cpu(std::vector<int64_t>{2, 1}, dtype, Device::cpu());
    start_selector_cpu.zero_();
    if (dtype == DType::Float32) {
        start_selector_cpu.data<float>()[0] = 1.0f;
    } else if (dtype == DType::Float64) {
        start_selector_cpu.data<double>()[0] = 1.0;
    } else if (dtype == DType::Float16) {
        start_selector_cpu.data<Float16>()[0] = Float16(static_cast<uint16_t>(0x3C00));  // Float16 representation of 1.0
    }
    Tensor start_selector = (target_device == Device::cpu())
        ? start_selector_cpu
        : start_selector_cpu.to(target_device);

    // End logits: multiply by [0, 1] - create on CPU first, then transfer
    Tensor end_selector_cpu(std::vector<int64_t>{2, 1}, dtype, Device::cpu());
    end_selector_cpu.zero_();
    if (dtype == DType::Float32) {
        end_selector_cpu.data<float>()[1] = 1.0f;
    } else if (dtype == DType::Float64) {
        end_selector_cpu.data<double>()[1] = 1.0;
    } else if (dtype == DType::Float16) {
        end_selector_cpu.data<Float16>()[1] = Float16(static_cast<uint16_t>(0x3C00));  // Float16 representation of 1.0
    }
    Tensor end_selector = (target_device == Device::cpu())
        ? end_selector_cpu
        : end_selector_cpu.to(target_device);

    // Use matmul to select: [batch*seq_len, 2] @ [2, 1] = [batch*seq_len, 1]
    Variable start_selector_var(start_selector, false);
    Variable end_selector_var(end_selector, false);

    auto start_flat = tenzor::matmul(reshaped, start_selector_var);  // [batch*seq_len, 1]
    auto end_flat = tenzor::matmul(reshaped, end_selector_var);      // [batch*seq_len, 1]

    // Reshape back to [batch, seq_len]
    auto start_logits = tenzor::reshape(start_flat, {batch_size, seq_len});
    auto end_logits = tenzor::reshape(end_flat, {batch_size, seq_len});

    // Call forward post-hooks (enables CPU-start offloading)
    call_forward_post_hooks();

    return ElectraQAOutput{start_logits, end_logits};
}

auto ElectraForQuestionAnswering::forward_impl(const Variable& input) -> Variable {
    auto outputs = forward(input, Tensor{}, Variable{});
    return outputs.start_logits;
}

} // namespace models
} // namespace tenzor
