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
#include <stdexcept>

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

    // ELECTRA ties the token (word) embeddings between the generator and the
    // discriminator (previously config.tie_embeddings was never honored). Build
    // ONE shared embedding table at the generator hidden size and inject it into
    // both towers. The discriminator uses a larger hidden size, so its embeddings
    // module projects the shared table up to its hidden size (see
    // BertEmbeddings::set_shared_word_embeddings). The shared table is registered
    // exactly once here so its parameters are not double-counted.
    if (config.tie_embeddings) {
        shared_embeddings_ = std::make_shared<nn::Embedding>(
            config.vocab_size, config.generator_hidden_size);
        register_module("shared_embeddings", shared_embeddings_);

        generator_->get_bert_model()->embeddings()->set_shared_word_embeddings(
            shared_embeddings_, config.generator_hidden_size);
        discriminator_->get_bert_model()->embeddings()->set_shared_word_embeddings(
            shared_embeddings_, config.generator_hidden_size);
    }
}

auto ElectraForPreTraining::sample_from_distribution(const float* probs, int64_t size) -> int64_t {
    // Sample token index from probability distribution using a per-instance,
    // seeded RNG. The mutex makes concurrent forwards on the same instance
    // well-defined; reproducibility now depends only on this model's own
    // sampling history rather than process-global state.
    std::discrete_distribution<int64_t> dist(probs, probs + size);
    std::lock_guard<std::mutex> lock(sample_rng_mutex_);
    return dist(sample_rng_);
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

    // Widen generator probabilities to Float32 once and run a single
    // dtype-generic sampling loop. This eliminates the previous per-dtype
    // duplication (Float32/Float64/Float16) and, critically, the missing
    // BFloat16 branch that silently left is_replaced all-zero and skipped the
    // generator sampling step for BF16 models. is_replaced is built in Float32
    // and cast back to the model dtype at the end, mirroring the widen-narrow
    // policy used elsewhere.
    if (gen_probs_cpu.dtype() != DType::Float32) {
        gen_probs_cpu = gen_probs_cpu.to(DType::Float32);
    }

    // Create is_replaced on CPU in Float32 for the generic loop.
    Tensor is_replaced_f32(shape_vec, DType::Float32, Device::cpu());
    is_replaced_f32.zero_();

    // Validate the caller-provided index tensors before reinterpreting their
    // buffers as int64 (mirrors compute_loss): a wrong-dtype or undersized tensor
    // would drive a heap out-of-bounds read in the loop below.
    if (masked_positions_cpu.dtype() != DType::Int64 ||
        original_tokens_cpu.dtype() != DType::Int64) {
        throw std::invalid_argument(
            "ElectraForPreTraining::forward: masked_positions and original_tokens must have dtype Int64");
    }
    if (masked_positions_cpu.numel() < batch_size * seq_len ||
        original_tokens_cpu.numel() < batch_size * seq_len) {
        throw std::invalid_argument(
            "ElectraForPreTraining::forward: masked_positions/original_tokens must have >= batch_size*seq_len elements");
    }

    const int64_t* mask_data = masked_positions_cpu.data<int64_t>();
    const int64_t* orig_data = original_tokens_cpu.data<int64_t>();
    int64_t* gen_data = generated_tokens_cpu.data<int64_t>();
    float* repl_data = is_replaced_f32.data<float>();
    const float* probs_data = gen_probs_cpu.data<const float>();

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

    // Cast is_replaced back to the model dtype.
    Tensor is_replaced_cpu = (dtype == DType::Float32)
        ? is_replaced_f32
        : is_replaced_f32.to(dtype);

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
                                         const Tensor& original_tokens,
                                         const Tensor& attention_mask) -> Variable {
    int64_t batch_size = gen_logits.shape()[0];
    int64_t seq_len = gen_logits.shape()[1];
    int64_t vocab_size = gen_logits.shape()[2];

    // Generator loss: MLM loss on masked tokens only
    // Compute cross-entropy loss for masked positions
    auto reshaped_logits = tenzor::reshape(gen_logits, {batch_size * seq_len, vocab_size});

    // Validate original_tokens before reinterpreting its buffer as int64.
    // data<int64_t>() does a raw reinterpret_cast of the storage, so a caller
    // passing a wrong-dtype or undersized tensor (the [batch, seq_len]/Int64
    // contract is documented in electra.hpp but not enforced here) would cause
    // a heap out-of-bounds read and silently corrupt labels. Enforce the
    // contract explicitly.
    if (!original_tokens.is_valid()) {
        throw std::invalid_argument(
            "ElectraForPreTraining::compute_loss: original_tokens is invalid/uninitialized");
    }
    if (original_tokens.dtype() != DType::Int64) {
        throw std::invalid_argument(
            "ElectraForPreTraining::compute_loss: original_tokens must have dtype Int64");
    }
    if (original_tokens.numel() < batch_size * seq_len) {
        throw std::invalid_argument(
            "ElectraForPreTraining::compute_loss: original_tokens.numel() (" +
            std::to_string(original_tokens.numel()) +
            ") is smaller than batch_size*seq_len (" +
            std::to_string(batch_size * seq_len) +
            ") derived from gen_logits");
    }

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
    // where p = sigmoid(disc_logits), y = is_replaced.
    //
    // Numerically stable formulation via log-sigmoid (avoids log(sigmoid(x))
    // and log(1 - sigmoid(x)) which overflow to log(0) = -inf for saturated
    // logits and poison the disc_loss_weight-scaled loss). Using the identities
    //   log(sigmoid(x))     = log_sigmoid(x)
    //   log(1 - sigmoid(x)) = log_sigmoid(-x)
    // both branches stay finite for any logit magnitude.
    auto log_prob_real = nn::log_sigmoid(disc_logits);             // log(p)
    auto log_prob_fake = nn::log_sigmoid(tenzor::neg(disc_logits)); // log(1 - p)

    auto ones_repl = Variable(ones_like(is_replaced), false);
    auto disc_loss_per_tok = tenzor::neg(is_replaced_var * log_prob_fake +
                       (ones_repl - is_replaced_var) * log_prob_real);

    // Discriminator RTD loss reduction.
    //
    // When a valid (B, T) attention_mask is supplied, average the per-token
    // disc loss over the *valid* (non-padding) tokens only. Padding positions
    // otherwise dominate the mean for short sequences in a padded batch,
    // shrinking the effective RTD signal toward log(2). Masked reduction:
    //   disc_loss = sum(per_tok * mask) / max(sum(mask), 1)
    // Empty/invalid mask → plain mean over all positions (backward compatible).
    // `disc_loss_per_tok` and `disc_logits` are [batch, seq_len]; reshape the
    // attention_mask to match so the elementwise multiply broadcasts correctly.
    Variable disc_loss;
    const int64_t disc_n = batch_size * seq_len;
    if (attention_mask.is_valid() && attention_mask.numel() == disc_n) {
        Tensor disc_mask = attention_mask.reshape({batch_size, seq_len})
                               .to(disc_logits.tensor().dtype())
                               .to(disc_logits.tensor().device());
        Variable disc_mask_var(disc_mask, false);

        auto masked_disc = disc_loss_per_tok * disc_mask_var;       // [batch, seq_len]
        Variable disc_sum = tenzor::sum(masked_disc);               // scalar

        // Sum the mask in Float32, not the model dtype. disc_mask is cast to
        // disc_logits.dtype() (line above) so for BF16/F16 models the count
        // would accumulate in half precision and saturate (BF16 cannot
        // represent integers > 256 exactly, F16 > 2048), shrinking the divisor
        // and inflating disc_loss. Widen to Float32 before the reduction.
        Tensor mask_count_cpu =
            tenzor::sum(disc_mask.to(DType::Float32)).to(Device::cpu()).to(DType::Float32);
        float valid_tokens = mask_count_cpu.item<float>();
        double disc_divisor = static_cast<double>(std::max(valid_tokens, 1.0f));
        disc_loss = disc_sum * (1.0 / disc_divisor);
    } else {
        // Mean over all tokens
        disc_loss = tenzor::mean(disc_loss_per_tok);
    }

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
    // mask_flat is cast to the model dtype (float_dtype) above; for BF16/F16
    // models summing it in that dtype saturates the count (BF16 > 256, F16 >
    // 2048), shrinking the divisor and inflating gen_loss. Widen to Float32
    // before the reduction so the count is exact.
    Tensor mask_sum_cpu =
        tenzor::sum(mask_flat.to(DType::Float32)).to(Device::cpu()).to(DType::Float32);
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

    // Split into start and end logits while preserving gradients.
    // logits: [batch, seq_len, 2]. Extract channel 0 (start) and channel 1
    // (end) via autograd-aware slice + squeeze on the last dim. This both
    // preserves the grad_fn chain and sidesteps the old selection-matrix path,
    // which only filled Float32/Float64/Float16 selectors and therefore
    // produced an all-zero selector (and all-zero logits) for BFloat16.
    auto start_logits = tenzor::squeeze(tenzor::slice(logits, 2, 0, 1), 2);  // [batch, seq_len]
    auto end_logits   = tenzor::squeeze(tenzor::slice(logits, 2, 1, 2), 2);  // [batch, seq_len]

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
