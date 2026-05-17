/**
 * @file bert.cpp
 * @brief Implementation of BERT model family
 */

#include "tenzor/models/bert.hpp"
#include "tenzor/models/hub.hpp"
#include "tenzor/io/torch_pickle.hpp"
#include "tenzor/nn/safetensors.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <filesystem>
#include <numeric>
#include <mutex>

namespace tenzor {
namespace models {

// ============================================================================
// BertEmbeddings Implementation
// ============================================================================

BertEmbeddings::BertEmbeddings(const BertConfig& config)
    : config_(config) {
    // Initialize embedding layers
    word_embeddings_ = std::make_shared<nn::Embedding>(
        config.vocab_size, config.hidden_size);
    position_embeddings_ = std::make_shared<nn::Embedding>(
        config.max_position_embeddings, config.hidden_size);
    token_type_embeddings_ = std::make_shared<nn::Embedding>(
        config.type_vocab_size, config.hidden_size);

    // Layer normalization and dropout
    layer_norm_ = std::make_shared<nn::LayerNorm>(
        std::vector<int64_t>{config.hidden_size}, config.layer_norm_eps);
    dropout_ = std::make_shared<nn::Dropout>(config.hidden_dropout_prob);

    // Register modules
    register_module("word_embeddings", word_embeddings_);
    register_module("position_embeddings", position_embeddings_);
    register_module("token_type_embeddings", token_type_embeddings_);
    register_module("layer_norm", layer_norm_);
    register_module("dropout", dropout_);
}

auto BertEmbeddings::create_position_ids(const Tensor& input_ids) -> Tensor {
    auto shape = input_ids.shape();
    int64_t batch_size = shape[0];
    int64_t seq_len = shape[1];
    auto target_device = input_ids.device();

    // Create position IDs on CPU first: [0, 1, 2, ..., seq_len-1]
    std::vector<int64_t> pos_data(seq_len);
    for (int64_t i = 0; i < seq_len; ++i) {
        pos_data[i] = i;
    }

    Tensor position_ids_cpu(std::vector<int64_t>{seq_len}, DType::Int64, Device::cpu());
    std::copy(pos_data.begin(), pos_data.end(), position_ids_cpu.data<int64_t>());

    // Expand to [batch_size, seq_len] on CPU
    position_ids_cpu = position_ids_cpu.unsqueeze(0);  // [1, seq_len]
    Tensor expanded_cpu(std::vector<int64_t>{batch_size, seq_len}, DType::Int64, Device::cpu());
    auto pos_data_ptr = position_ids_cpu.data<int64_t>();
    auto expanded_ptr = expanded_cpu.data<int64_t>();
    for (int64_t b = 0; b < batch_size; ++b) {
        std::copy(pos_data_ptr, pos_data_ptr + seq_len, expanded_ptr + b * seq_len);
    }

    // Move to target device
    return (target_device == Device::cpu()) ? expanded_cpu : expanded_cpu.to(target_device);
}

auto BertEmbeddings::forward(const Variable& input_ids,
                             const Variable& token_type_ids,
                             const Variable& position_ids) -> Variable {
    // Call forward pre-hooks (enables CPU-start offloading)
    // NOTE: This is necessary because this multi-argument forward bypasses Module::forward()
    call_forward_pre_hooks();

    auto shape = input_ids.tensor().shape();
    int64_t batch_size = shape[0];
    int64_t seq_len = shape[1];

    // Get word embeddings
    auto embeddings = word_embeddings_->forward(input_ids);

    // Get position embeddings
    Variable pos_ids = position_ids;
    if (!position_ids.is_initialized() || position_ids.tensor().numel() == 0) {
        // Create default position IDs [0, 1, 2, ..., seq_len-1]
        auto pos_tensor = create_position_ids(input_ids.tensor());
        pos_ids = Variable(pos_tensor, false);  // Indices don't need gradients
    }
    auto position_embeddings = position_embeddings_->forward(pos_ids);

    // Get token type embeddings
    Variable type_ids = token_type_ids;
    if (!token_type_ids.is_initialized() || token_type_ids.tensor().numel() == 0) {
        // Create default token type IDs (all zeros) on CPU then move to device
        auto target_device = input_ids.tensor().device();
        Tensor zeros_cpu(std::vector<int64_t>{batch_size, seq_len}, DType::Int64, Device::cpu());
        zeros_cpu.zero_();
        auto zeros_tensor = (target_device == Device::cpu()) ? zeros_cpu : zeros_cpu.to(target_device);
        type_ids = Variable(zeros_tensor, false);  // Indices don't need gradients
    }
    auto token_type_embeddings = token_type_embeddings_->forward(type_ids);

    // Combine embeddings
    embeddings = embeddings + position_embeddings + token_type_embeddings;

    // Apply layer normalization and dropout
    embeddings = layer_norm_->forward(embeddings);
    embeddings = dropout_->forward(embeddings);

    // Call forward post-hooks (enables CPU-start offloading)
    call_forward_post_hooks();

    return embeddings;
}

auto BertEmbeddings::forward_impl(const Variable& input) -> Variable {
    return forward(input, Variable{}, Variable{});
}

// ============================================================================
// BertEncoder Implementation
// ============================================================================

BertEncoder::BertEncoder(const BertConfig& config)
    : config_(config) {
    // Create a single encoder layer
    auto encoder_layer = std::make_shared<nn::TransformerEncoderLayer>(
        config.hidden_size,
        config.num_attention_heads,
        config.intermediate_size,
        config.attention_probs_dropout_prob,
        config.hidden_act,
        true  // batch_first
    );

    // Create final layer norm
    auto norm = std::make_shared<nn::LayerNorm>(
        std::vector<int64_t>{config.hidden_size}, config.layer_norm_eps);

    // Create encoder stack
    encoder_ = std::make_shared<nn::TransformerEncoder>(
        encoder_layer, config.num_hidden_layers, norm);

    register_module("encoder", encoder_);
}

auto BertEncoder::prepare_attention_mask(const Tensor& mask, int64_t seq_len, DType compute_dtype) -> Tensor {
    // L9 note: returning `Tensor{}` here is the deliberate "no mask"
    // sentinel pattern. Downstream `encoder_->forward(hidden_states, mask, ...)`
    // tests `.is_valid()` / `.numel() != 0` and skips mask application
    // when this sentinel is returned. Equivalent to `std::optional<Tensor>`
    // but matches the established API across the model zoo.
    if (!mask.is_valid() || mask.numel() == 0) {
        return Tensor{};
    }

    // Input mask is [batch, seq_len] with 1 for valid, 0 for padding
    // Convert to [batch, 1, 1, seq_len] float mask with 0.0 for attended, -1e9 for masked

    auto batch_size = mask.shape()[0];

    // Convert mask to the model's compute dtype to avoid dtype mismatches during backward
    auto float_mask = mask.to(compute_dtype);
    // (mask - 1.0) gives -1.0 for padding (was 0), 0.0 for valid (was 1)
    // Use dtype-appropriate large value: Float16 max is ~65504, so 1e9 would overflow
    float mask_fill = (compute_dtype == DType::Float16 || compute_dtype == DType::BFloat16)
                      ? 1e4f : 1e9f;
    auto attn_mask = (float_mask - 1.0) * mask_fill;

    // Reshape to [batch, 1, 1, seq_len] for broadcasting across heads and query positions
    attn_mask = attn_mask.reshape({batch_size, 1, 1, seq_len});

    return attn_mask;
}

auto BertEncoder::forward(const Variable& hidden_states,
                          const Tensor& attention_mask) -> Variable {
    // Call forward pre-hooks (enables CPU-start offloading)
    // NOTE: This is necessary because this multi-argument forward bypasses Module::forward()
    call_forward_pre_hooks();

    // Prepare attention mask if provided
    auto mask = prepare_attention_mask(attention_mask, hidden_states.tensor().shape()[1],
                                       hidden_states.tensor().dtype());

    // Pass through encoder layers
    auto output = encoder_->forward(hidden_states, mask, Tensor{});

    // Call forward post-hooks (enables CPU-start offloading)
    call_forward_post_hooks();

    return output;
}

auto BertEncoder::forward_impl(const Variable& input) -> Variable {
    return forward(input, Tensor{});
}

// ============================================================================
// BertPooler Implementation
// ============================================================================

BertPooler::BertPooler(const BertConfig& config) {
    // Linear transformation for pooling
    dense_ = std::make_shared<nn::Linear>(config.hidden_size, config.hidden_size);
    register_module("dense", dense_);
}

auto BertPooler::forward_impl(const Variable& hidden_states) -> Variable {
    // Extract [CLS] token (first token) representation
    // hidden_states: [batch, seq_len, hidden_size]
    // We need to extract the first token while preserving gradients

    auto shape = hidden_states.shape();
    int64_t batch_size = shape[0];
    int64_t seq_len = shape[1];
    int64_t hidden_size = shape[2];

    // Approach: Use reshape to reorganize then use autograd-aware operations
    // Reshape to [batch, seq_len, hidden_size] -> [batch * seq_len, hidden_size]
    auto reshaped = tenzor::reshape(hidden_states, {batch_size * seq_len, hidden_size});

    // Create a mask tensor for selecting first tokens: [batch * seq_len]
    // We want indices [0, seq_len, 2*seq_len, ..., (batch-1)*seq_len]
    // Instead of using indexing, we'll create a weighted sum where only first tokens have weight 1

    // Alternative: Manually build a selection matrix and use matmul
    // Create selection matrix [batch, batch * seq_len] where each row has 1 at position b*seq_len
    // Use the same dtype as hidden_states for consistency
    auto dtype = hidden_states.tensor().dtype();
    auto target_device = hidden_states.tensor().device();

    // Create on CPU for data filling, then transfer to device
    Tensor selection_matrix_cpu(std::vector<int64_t>{batch_size, batch_size * seq_len},
                                dtype, Device::cpu());
    selection_matrix_cpu.zero_();

    // Fill the selection matrix with appropriate dtype
    if (dtype == DType::Float32) {
        float* sel_data = selection_matrix_cpu.data<float>();
        for (int64_t b = 0; b < batch_size; ++b) {
            sel_data[b * (batch_size * seq_len) + b * seq_len] = 1.0f;
        }
    } else if (dtype == DType::Float64) {
        double* sel_data = selection_matrix_cpu.data<double>();
        for (int64_t b = 0; b < batch_size; ++b) {
            sel_data[b * (batch_size * seq_len) + b * seq_len] = 1.0;
        }
    } else if (dtype == DType::Float16) {
        // Float16 data needs special handling
        auto* sel_data = selection_matrix_cpu.data<Float16>();
        Float16 one_f16(1.0f);
        for (int64_t b = 0; b < batch_size; ++b) {
            sel_data[b * (batch_size * seq_len) + b * seq_len] = one_f16;
        }
    }

    // Transfer to target device if needed
    Tensor selection_matrix = (target_device == Device::cpu()) ?
                               selection_matrix_cpu : selection_matrix_cpu.to(target_device);

    // Now: selection_matrix @ reshaped gives us [batch, hidden_size] with first tokens
    Variable selection_var(selection_matrix, false);  // No grad needed for constant matrix
    auto cls_token = tenzor::matmul(selection_var, reshaped);  // [batch, hidden_size]

    // Apply linear transformation
    auto pooled = dense_->forward(cls_token);

    // Apply tanh activation
    pooled = nn::tanh(pooled);

    return pooled;
}

// ============================================================================
// BertModel Implementation
// ============================================================================

BertModel::BertModel(const BertConfig& config, bool add_pooling_layer)
    : config_(config) {
    // Initialize components
    embeddings_ = std::make_shared<BertEmbeddings>(config);
    encoder_ = std::make_shared<BertEncoder>(config);

    // Register modules in order (maintain consistent ordering)
    register_module("embeddings", embeddings_);
    register_module("encoder", encoder_);

    // Pooler is optional - only create and register if requested
    if (add_pooling_layer) {
        pooler_ = std::make_shared<BertPooler>(config);
        register_module("pooler", pooler_);
    }
}

auto BertModel::forward(const Variable& input_ids,
                       const Tensor& attention_mask,
                       const Variable& token_type_ids,
                       const Variable& position_ids) -> BertOutput {
    // Call forward pre-hooks (enables CPU-start offloading)
    // NOTE: This is necessary because this multi-argument forward bypasses Module::forward()
    call_forward_pre_hooks();

    // Get embeddings
    auto embedding_output = embeddings_->forward(input_ids, token_type_ids, position_ids);

    // Encode
    auto sequence_output = encoder_->forward(embedding_output, attention_mask);

    // Pool (only if pooler exists)
    Variable pooled_output;
    if (pooler_) {
        pooled_output = pooler_->forward(sequence_output);
    } else {
        // Create empty pooled output tensor to avoid uninitialized Variable
        auto seq_shape = sequence_output.tensor().shape();
        Tensor empty_pooled({seq_shape[0], seq_shape[2]}, sequence_output.tensor().dtype(), sequence_output.tensor().device());
        empty_pooled.zero_();
        pooled_output = Variable(empty_pooled, false);
    }

    // Call forward post-hooks (enables CPU-start offloading)
    call_forward_post_hooks();

    return BertOutput{sequence_output, pooled_output};
}

auto BertModel::forward_impl(const Variable& input) -> Variable {
    auto outputs = forward(input, Tensor{}, Variable{}, Variable{});
    return outputs.sequence_output;
}

// ============================================================================
// BertForSequenceClassification Implementation
// ============================================================================

BertForSequenceClassification::BertForSequenceClassification(
    const BertConfig& config, int64_t num_labels)
    : config_(config), num_labels_(num_labels) {
    // Base BERT model
    bert_ = std::make_shared<BertModel>(config);

    // Classification head
    dropout_ = std::make_shared<nn::Dropout>(config.hidden_dropout_prob);
    classifier_ = std::make_shared<nn::Linear>(config.hidden_size, num_labels);

    register_module("bert", bert_);
    register_module("dropout", dropout_);
    register_module("classifier", classifier_);
}

auto BertForSequenceClassification::forward(const Variable& input_ids,
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

auto BertForSequenceClassification::forward_impl(const Variable& input) -> Variable {
    return forward(input, Tensor{}, Variable{});
}

// ============================================================================
// BertForTokenClassification Implementation
// ============================================================================

BertForTokenClassification::BertForTokenClassification(
    const BertConfig& config, int64_t num_labels)
    : config_(config), num_labels_(num_labels) {
    // Base BERT model (no pooling needed for token classification)
    bert_ = std::make_shared<BertModel>(config, false);

    // Token classification head
    dropout_ = std::make_shared<nn::Dropout>(config.hidden_dropout_prob);
    classifier_ = std::make_shared<nn::Linear>(config.hidden_size, num_labels);

    register_module("bert", bert_);
    register_module("dropout", dropout_);
    register_module("classifier", classifier_);
}

auto BertForTokenClassification::forward(const Variable& input_ids,
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

auto BertForTokenClassification::forward_impl(const Variable& input) -> Variable {
    return forward(input, Tensor{}, Variable{});
}

// ============================================================================
// BertForQuestionAnswering Implementation
// ============================================================================

BertForQuestionAnswering::BertForQuestionAnswering(const BertConfig& config)
    : config_(config) {
    // Base BERT model (no pooling needed for question answering)
    bert_ = std::make_shared<BertModel>(config, false);

    // QA output layer (predicts start and end positions)
    qa_outputs_ = std::make_shared<nn::Linear>(config.hidden_size, 2);

    register_module("bert", bert_);
    register_module("qa_outputs", qa_outputs_);
}

auto BertForQuestionAnswering::forward(const Variable& input_ids,
                                       const Tensor& attention_mask,
                                       const Variable& token_type_ids) -> BertQAOutput {
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
    auto target_device = logits.tensor().device();

    // Start logits: multiply by [1, 0] - create on CPU first
    Tensor start_selector_cpu(std::vector<int64_t>{2, 1}, dtype, Device::cpu());
    start_selector_cpu.zero_();
    if (dtype == DType::Float32) {
        start_selector_cpu.data<float>()[0] = 1.0f;
    } else if (dtype == DType::Float64) {
        start_selector_cpu.data<double>()[0] = 1.0;
    } else if (dtype == DType::Float16) {
        start_selector_cpu.data<Float16>()[0] = Float16(1.0f);
    }
    Tensor start_selector = (target_device == Device::cpu()) ?
                            start_selector_cpu : start_selector_cpu.to(target_device);

    // End logits: multiply by [0, 1] - create on CPU first
    Tensor end_selector_cpu(std::vector<int64_t>{2, 1}, dtype, Device::cpu());
    end_selector_cpu.zero_();
    if (dtype == DType::Float32) {
        end_selector_cpu.data<float>()[1] = 1.0f;
    } else if (dtype == DType::Float64) {
        end_selector_cpu.data<double>()[1] = 1.0;
    } else if (dtype == DType::Float16) {
        end_selector_cpu.data<Float16>()[1] = Float16(1.0f);
    }
    Tensor end_selector = (target_device == Device::cpu()) ?
                          end_selector_cpu : end_selector_cpu.to(target_device);

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

    return BertQAOutput{start_logits, end_logits};
}

auto BertForQuestionAnswering::forward_impl(const Variable& input) -> Variable {
    auto outputs = forward(input, Tensor{}, Variable{});
    return outputs.start_logits;
}

// ============================================================================
// ModelHub Implementation
// ============================================================================

auto BertModelHub::load_pretrained_weights(nn::Module& model,
                                      const std::string& model_name,
                                      const std::string& cache_dir) -> void {
    // Download model if needed
    auto checkpoint_path = download_model(model_name, cache_dir);

    // Load PyTorch checkpoint
    auto checkpoint = load_pytorch_checkpoint(checkpoint_path);

    // Verify compatibility
    verify_checkpoint_compatibility(model, checkpoint);

    // Map and load state dict
    std::unordered_map<std::string, Tensor> mapped_state;
    for (const auto& [hf_name, tensor] : checkpoint) {
        auto tenzor_name = map_parameter_name(hf_name);
        mapped_state[tenzor_name] = tensor;
    }

    // Load into model
    model.load_state_dict(mapped_state);
}

// ----------------------------------------------------------------------------
// BERT pretrained-model registry.
// Registers each canonical HuggingFace BERT family entry exactly once with
// the central ModelHub. Subsequent calls go through ModelHub's libcurl
// downloader, safetensors-mirror dispatch, SHA verification, and cache.
// ----------------------------------------------------------------------------

namespace {

struct BertRegistry {
    std::once_flag flag;

    void register_once() {
        std::call_once(flag, []() {
            using MWI = ModelWeightInfo;
            // Field order: { name, url, sha256, size, description, safetensors_url }.
            const std::vector<MWI> entries = {
                MWI{ "bert-base-uncased",
                     "https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin",
                     /*sha256=*/"",
                     /*size=*/0,
                     /*description=*/"BERT-Base uncased (110M, English, 12L/768H/12A)",
                     /*safetensors_url=*/"https://huggingface.co/bert-base-uncased/resolve/main/model.safetensors" },
                MWI{ "bert-base-cased",
                     "https://huggingface.co/bert-base-cased/resolve/main/pytorch_model.bin",
                     "", 0, "BERT-Base cased (110M, English, 12L/768H/12A)",
                     "https://huggingface.co/bert-base-cased/resolve/main/model.safetensors" },
                MWI{ "bert-large-uncased",
                     "https://huggingface.co/bert-large-uncased/resolve/main/pytorch_model.bin",
                     "", 0, "BERT-Large uncased (340M, English, 24L/1024H/16A)",
                     "https://huggingface.co/bert-large-uncased/resolve/main/model.safetensors" },
                MWI{ "bert-large-cased",
                     "https://huggingface.co/bert-large-cased/resolve/main/pytorch_model.bin",
                     "", 0, "BERT-Large cased (340M, English, 24L/1024H/16A)",
                     "https://huggingface.co/bert-large-cased/resolve/main/model.safetensors" },
                MWI{ "bert-base-multilingual-cased",
                     "https://huggingface.co/bert-base-multilingual-cased/resolve/main/pytorch_model.bin",
                     "", 0, "BERT-Base multilingual cased (104 languages)",
                     "https://huggingface.co/bert-base-multilingual-cased/resolve/main/model.safetensors" },
            };
            ModelHub::register_models(entries);
        });
    }
};

BertRegistry& bert_registry() {
    static BertRegistry r;
    return r;
}

} // namespace

auto BertModelHub::download_model(const std::string& model_name,
                                  const std::string& cache_dir) -> std::string {
    bert_registry().register_once();

    // Honour the caller-supplied cache_dir if non-empty by setting it on the
    // ModelHub (the singleton config is mutable). Empty means "use default".
    if (!cache_dir.empty()) {
        auto cfg = ModelHub::get_config();
        cfg.cache_dir = cache_dir;
        ModelHub::set_config(cfg);
    }

    // Delegate to ModelHub. Prefer safetensors mirror over legacy .bin —
    // safer parse, no pickle. ModelHub returns the local cache path.
    return ModelHub::download_pretrained_safetensors(
        model_name, /*prefer_safetensors=*/true);
}

auto BertModelHub::map_parameter_name(const std::string& hf_name) -> std::string {
    // Map Hugging Face parameter names to Tenzor names
    std::string name = hf_name;

    // Replace common prefixes
    if (name.find("bert.") == 0) {
        name = name.substr(5);  // Remove "bert." prefix
    }

    // Map specific layers
    // HF: embeddings.word_embeddings.weight -> Tenzor: embeddings.word_embeddings.weight
    // HF: encoder.layer.0.attention.self.query.weight -> Tenzor: encoder.layers.0.self_attn.query.weight

    // Replace layer indexing
    size_t pos = name.find("encoder.layer.");
    if (pos != std::string::npos) {
        name.replace(pos, 14, "encoder.layers.");
    }

    // Replace attention sublayer names
    pos = name.find("attention.self.");
    if (pos != std::string::npos) {
        name.replace(pos, 15, "self_attn.");
    }

    // Replace attention output
    pos = name.find("attention.output.dense.");
    if (pos != std::string::npos) {
        name.replace(pos, 23, "self_attn.out_proj.");
    }

    // Replace FFN layers
    pos = name.find("intermediate.dense.");
    if (pos != std::string::npos) {
        name.replace(pos, 19, "linear1.");
    }

    pos = name.find("output.dense.");
    if (pos != std::string::npos) {
        name.replace(pos, 13, "linear2.");
    }

    // Replace LayerNorm
    pos = name.find("attention.output.LayerNorm.");
    if (pos != std::string::npos) {
        name.replace(pos, 27, "norm1.");
    }

    pos = name.find("output.LayerNorm.");
    if (pos != std::string::npos) {
        name.replace(pos, 17, "norm2.");
    }

    return name;
}

auto BertModelHub::load_pytorch_checkpoint(const std::string& checkpoint_path)
    -> std::unordered_map<std::string, Tensor> {
    if (!std::filesystem::exists(checkpoint_path)) {
        throw std::runtime_error(
            "BertModelHub::load_pytorch_checkpoint: file not found: " + checkpoint_path);
    }

    // Dispatch by extension to the right deserializer (safetensors / pickle).
    // For HuggingFace BERT, the canonical extensions are .safetensors and .bin.
    auto ends_with = [](const std::string& s, const std::string& suffix) {
        return s.size() >= suffix.size() &&
               s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    };

    if (ends_with(checkpoint_path, ".safetensors")) {
        // SafeTensors path: parsed in ModelHub's static method.
        // Delegate to ModelHub::load_pretrained_weights' deserializer via
        // a direct call into the SafeTensors serializer to keep symmetry.
        return nn::SafeTensorsSerializer::load(checkpoint_path);
    }

    // .bin / .pth / .pt — PyTorch pickle archive. tenzor::io::load_torch_pickle
    // implements a native C++ reader that parses the torch.save zip + pickle
    // protocol (no PyTorch runtime dependency).
    return tenzor::io::load_torch_pickle(checkpoint_path);
}

auto BertModelHub::verify_checkpoint_compatibility(
    const nn::Module& model,
    const std::unordered_map<std::string, Tensor>& checkpoint) -> void {
    // Get model parameters
    auto model_params = model.state_dict();

    // Check that all model parameters are in checkpoint
    for (const auto& [name, tensor] : model_params) {
        if (checkpoint.find(name) == checkpoint.end()) {
            throw std::runtime_error(
                "Model parameter '" + name + "' not found in checkpoint");
        }

        // Check shape compatibility
        const auto& checkpoint_tensor = checkpoint.at(name);
        auto model_shape = tensor.shape();
        auto checkpoint_shape = checkpoint_tensor.shape();

        if (model_shape.size() != checkpoint_shape.size()) {
            throw std::runtime_error("Shape mismatch for parameter '" + name + "'");
        }

        for (size_t i = 0; i < model_shape.size(); ++i) {
            if (model_shape[i] != checkpoint_shape[i]) {
                throw std::runtime_error("Shape mismatch for parameter '" + name + "'");
            }
        }
    }

    // Warn about unused checkpoint parameters
    for (const auto& [name, tensor] : checkpoint) {
        if (model_params.find(name) == model_params.end()) {
            // This is just a warning, not an error
            // Some checkpoints may have extra parameters
        }
    }
}

} // namespace models
} // namespace tenzor
