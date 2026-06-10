/**
 * @file t5.cpp
 * @brief Implementation of T5 model family
 */

#include "tenzor/models/t5.hpp"
#include "tenzor/models/hub.hpp"  // Audit H4
#include "tenzor/core/tensor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/checkpoint.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace tenzor {
namespace models {

// ============================================================================
// T5Attention Implementation
// ============================================================================

T5Attention::T5Attention(const T5Config& config, bool has_relative_attention_bias)
    : config_(config), has_relative_attention_bias_(has_relative_attention_bias) {

    int64_t inner_dim = config.num_heads * config.d_kv;

    // Q, K, V projections
    q_proj_ = std::make_shared<nn::Linear>(config.d_model, inner_dim, false);
    k_proj_ = std::make_shared<nn::Linear>(config.d_model, inner_dim, false);
    v_proj_ = std::make_shared<nn::Linear>(config.d_model, inner_dim, false);
    o_proj_ = std::make_shared<nn::Linear>(inner_dim, config.d_model, false);

    // Relative position bias (only in first layer of each stack)
    if (has_relative_attention_bias_) {
        relative_attention_bias_ = std::make_shared<nn::Embedding>(
            config.relative_attention_num_buckets,
            config.num_heads
        );
        register_module("relative_attention_bias", relative_attention_bias_);
    }

    register_module("q_proj", q_proj_);
    register_module("k_proj", k_proj_);
    register_module("v_proj", v_proj_);
    register_module("o_proj", o_proj_);
}

auto T5Attention::relative_position_bucket(int64_t relative_position,
                                            int64_t num_buckets,
                                            int64_t max_distance) -> int64_t {
    // T5 uses bidirectional relative attention buckets
    // Half the buckets are for negative positions, half for positive
    int64_t num_buckets_per_direction = num_buckets / 2;
    int64_t bucket = 0;

    // Determine if position is negative and get absolute value
    int64_t n = std::abs(relative_position);

    // Half buckets for exact positions (small distances)
    // Half for logarithmic buckets (large distances)
    int64_t max_exact = num_buckets_per_direction / 2;
    bool is_small = n < max_exact;

    if (is_small) {
        bucket = n;
    } else {
        // Logarithmic bucketing for larger distances
        int64_t num_exact_buckets = num_buckets_per_direction / 2;
        int64_t scale = (num_buckets_per_direction - num_exact_buckets);
        bucket = num_exact_buckets + static_cast<int64_t>(
            std::log(static_cast<double>(n) / max_exact) /
            std::log(static_cast<double>(max_distance) / max_exact) * scale
        );
        bucket = std::min(bucket, num_buckets_per_direction - 1);
    }

    // Add offset for negative positions (use second half of buckets)
    if (relative_position < 0) {
        bucket = num_buckets_per_direction + bucket;
    }

    return bucket;
}

auto T5Attention::compute_bias(int64_t query_length, int64_t key_length) -> Tensor {
    // Compute relative position bias matrix
    // Infer dtype and device from embedding weights to support multi-dtype and multi-device models
    DType dtype = relative_attention_bias_->weight().tensor().dtype();
    Device target_device = relative_attention_bias_->weight().tensor().device();

    // Create on CPU for data filling, then transfer to target device
    Tensor position_bias(std::vector<int64_t>{config_.num_heads, query_length, key_length},
                         dtype, Device::cpu());

    // Zero-initialize (dtype-aware since zero_() may not support all dtypes)
    if (dtype == DType::Float16) {
        auto* data = position_bias.data<Float16>();
        std::fill_n(data, position_bias.numel(), Float16(0.0f));
    } else if (dtype == DType::Float32) {
        auto* data = position_bias.data<float>();
        std::fill_n(data, position_bias.numel(), 0.0f);
    } else if (dtype == DType::Float64) {
        auto* data = position_bias.data<double>();
        std::fill_n(data, position_bias.numel(), 0.0);
    }

    // Compute bias for each (query_pos, key_pos) pair
    for (int64_t i = 0; i < query_length; ++i) {
        for (int64_t j = 0; j < key_length; ++j) {
            int64_t relative_position = i - j;
            int64_t bucket = relative_position_bucket(
                relative_position,
                config_.relative_attention_num_buckets,
                config_.relative_attention_max_distance
            );

            // Look up bias for this bucket
            // Create bucket tensor on the target device to match embedding weights
            Tensor bucket_tensor({1}, DType::Int64, Device::cpu());
            bucket_tensor.data<int64_t>()[0] = bucket;
            // Move to target device for embedding lookup
            bucket_tensor = bucket_tensor.to(target_device);
            auto bias_values = relative_attention_bias_->forward(
                Variable(bucket_tensor, false)
            ).tensor();
            // Move result back to CPU for data access
            bias_values = bias_values.to(Device::cpu());

            // Assign to all heads (dtype-generic)
            if (dtype == DType::Float16) {
                auto* bias_ptr = position_bias.data<Float16>();
                auto* values_ptr = bias_values.data<Float16>();
                for (int64_t h = 0; h < config_.num_heads; ++h) {
                    bias_ptr[h * query_length * key_length + i * key_length + j] = values_ptr[h];
                }
            } else if (dtype == DType::Float32) {
                auto* bias_ptr = position_bias.data<float>();
                auto* values_ptr = bias_values.data<float>();
                for (int64_t h = 0; h < config_.num_heads; ++h) {
                    bias_ptr[h * query_length * key_length + i * key_length + j] = values_ptr[h];
                }
            } else if (dtype == DType::Float64) {
                auto* bias_ptr = position_bias.data<double>();
                auto* values_ptr = bias_values.data<double>();
                for (int64_t h = 0; h < config_.num_heads; ++h) {
                    bias_ptr[h * query_length * key_length + i * key_length + j] = values_ptr[h];
                }
            }
        }
    }

    // Transfer to target device if needed
    if (target_device != Device::cpu()) {
        position_bias = position_bias.to(target_device);
    }

    return position_bias;
}

auto T5Attention::forward(const Variable& hidden_states,
                           const Variable& key_value_states,
                           const Tensor& attention_mask,
                           const Tensor& position_bias)
    -> std::tuple<Variable, Tensor> {
    // Call forward pre-hooks (enables CPU-start offloading)
    // NOTE: This is necessary because this multi-argument forward bypasses Module::forward()
    call_forward_pre_hooks();

    bool is_cross_attention = key_value_states.is_initialized();
    auto shape = hidden_states.shape();
    int64_t batch_size = shape[0];
    int64_t seq_len = shape[1];
    int64_t kv_seq_len = is_cross_attention ? key_value_states.shape()[1] : seq_len;

    // Compute Q, K, V
    auto q = q_proj_->forward(hidden_states);
    auto k = is_cross_attention ? k_proj_->forward(key_value_states)
                                 : k_proj_->forward(hidden_states);
    auto v = is_cross_attention ? v_proj_->forward(key_value_states)
                                 : v_proj_->forward(hidden_states);

    // Reshape for multi-head attention: [batch, seq_len, num_heads, d_kv]
    q = q.reshape(std::vector<int64_t>{batch_size, seq_len, config_.num_heads, config_.d_kv});
    k = k.reshape(std::vector<int64_t>{batch_size, kv_seq_len, config_.num_heads, config_.d_kv});
    v = v.reshape(std::vector<int64_t>{batch_size, kv_seq_len, config_.num_heads, config_.d_kv});

    // Transpose to [batch, num_heads, seq_len, d_kv]
    q = q.transpose(1, 2);
    k = k.transpose(1, 2);
    v = v.transpose(1, 2);

    // Reshape to 3D for bmm: [batch * num_heads, seq_len, d_kv]
    auto q_3d = q.reshape(std::vector<int64_t>{batch_size * config_.num_heads, seq_len, config_.d_kv});
    auto k_3d = k.reshape(std::vector<int64_t>{batch_size * config_.num_heads, kv_seq_len, config_.d_kv});
    auto v_3d = v.reshape(std::vector<int64_t>{batch_size * config_.num_heads, kv_seq_len, config_.d_kv});

    // Transpose key for attention: [batch*heads, kv_seq_len, d_kv] -> [batch*heads, d_kv, kv_seq_len]
    std::vector<int64_t> key_perm = {0, 2, 1};
    auto k_transposed = tenzor::permute(k_3d, key_perm);

    // Compute attention scores: Q @ K^T / sqrt(d_kv)
    // bmm: [batch*heads, seq_len, d_kv] @ [batch*heads, d_kv, kv_seq_len] -> [batch*heads, seq_len, kv_seq_len]
    auto scores = tenzor::bmm(q_3d, k_transposed);

    // Reshape scores back to 4D: [batch, num_heads, seq_len, kv_seq_len]
    scores = scores.reshape(std::vector<int64_t>{batch_size, config_.num_heads, seq_len, kv_seq_len});
    double scale = 1.0 / std::sqrt(static_cast<double>(config_.d_kv));
    scores = scores * scale;

    // Add relative position bias
    Tensor bias = position_bias;
    if ((!position_bias.is_valid() || !position_bias.numel()) && has_relative_attention_bias_) {
        bias = compute_bias(seq_len, kv_seq_len);
    }

    if (bias.is_valid() && bias.numel() > 0) {
        // Expand bias to batch dimension: [num_heads, q_len, kv_len] -> [batch, num_heads, q_len, kv_len]
        // Create on CPU for data filling, then transfer to target device
        Device bias_device = bias.device();
        Tensor bias_cpu = (bias_device == Device::cpu()) ? bias : bias.to(Device::cpu());

        Tensor expanded_bias_cpu({batch_size, config_.num_heads, seq_len, kv_seq_len},
                                  bias.dtype(), Device::cpu());
        int64_t bias_size = config_.num_heads * seq_len * kv_seq_len;

        // Dtype-generic bias expansion
        if (bias.dtype() == DType::Float16) {
            auto* bias_ptr = bias_cpu.data<Float16>();
            auto* expanded_ptr = expanded_bias_cpu.data<Float16>();
            for (int64_t b = 0; b < batch_size; ++b) {
                std::copy(bias_ptr, bias_ptr + bias_size, expanded_ptr + b * bias_size);
            }
        } else if (bias.dtype() == DType::Float32) {
            auto* bias_ptr = bias_cpu.data<float>();
            auto* expanded_ptr = expanded_bias_cpu.data<float>();
            for (int64_t b = 0; b < batch_size; ++b) {
                std::copy(bias_ptr, bias_ptr + bias_size, expanded_ptr + b * bias_size);
            }
        } else if (bias.dtype() == DType::Float64) {
            auto* bias_ptr = bias_cpu.data<double>();
            auto* expanded_ptr = expanded_bias_cpu.data<double>();
            for (int64_t b = 0; b < batch_size; ++b) {
                std::copy(bias_ptr, bias_ptr + bias_size, expanded_ptr + b * bias_size);
            }
        }

        // Transfer to target device if needed
        Tensor expanded_bias = (bias_device == Device::cpu()) ?
                                expanded_bias_cpu : expanded_bias_cpu.to(bias_device);
        scores = scores + Variable(expanded_bias, false);
    }

    // Apply attention mask if provided
    if (attention_mask.is_valid() && attention_mask.numel() > 0) {
        scores = scores + Variable(attention_mask, false);
    }

    // Softmax over key dimension
    auto attn_weights = nn::softmax(scores, -1);

    // Reshape attn_weights to 3D for bmm: [batch * num_heads, seq_len, kv_seq_len]
    auto attn_weights_3d = attn_weights.reshape(std::vector<int64_t>{batch_size * config_.num_heads, seq_len, kv_seq_len});

    // Apply attention to values
    // bmm: [batch*heads, seq_len, kv_seq_len] @ [batch*heads, kv_seq_len, d_kv] -> [batch*heads, seq_len, d_kv]
    auto context = tenzor::bmm(attn_weights_3d, v_3d);

    // Reshape context back to 4D: [batch, num_heads, seq_len, d_kv]
    context = context.reshape(std::vector<int64_t>{batch_size, config_.num_heads, seq_len, config_.d_kv});

    // Transpose back and reshape: [batch, num_heads, seq_len, d_kv] -> [batch, seq_len, d_model]
    context = context.transpose(1, 2);
    context = context.reshape(std::vector<int64_t>{batch_size, seq_len, config_.num_heads * config_.d_kv});

    // Output projection
    auto output = o_proj_->forward(context);

    // Call forward post-hooks (enables CPU-start offloading)
    call_forward_post_hooks();

    return {output, bias};
}

auto T5Attention::forward_impl(const Variable& input) -> Variable {
    auto [output, bias] = forward(input, Variable{}, Tensor{}, Tensor{});
    return output;
}

// ============================================================================
// T5LayerNorm Implementation
// ============================================================================

T5LayerNorm::T5LayerNorm(int64_t d_model, double eps) {
    layer_norm_ = std::make_shared<nn::LayerNorm>(
        std::vector<int64_t>{d_model}, eps);
    register_module("layer_norm", layer_norm_);
}

auto T5LayerNorm::forward_impl(const Variable& input) -> Variable {
    return layer_norm_->forward(input);
}

// ============================================================================
// T5DenseActDense Implementation
// ============================================================================

T5DenseActDense::T5DenseActDense(const T5Config& config)
    : config_(config) {
    wi_ = std::make_shared<nn::Linear>(config.d_model, config.d_ff, false);
    wo_ = std::make_shared<nn::Linear>(config.d_ff, config.d_model, false);
    dropout_ = std::make_shared<nn::Dropout>(config.dropout_rate);

    register_module("wi", wi_);
    register_module("wo", wo_);
    register_module("dropout", dropout_);
}

auto T5DenseActDense::forward_impl(const Variable& input) -> Variable {
    auto h = wi_->forward(input);

    // Apply activation (ReLU for standard T5)
    if (config_.dense_act_fn == "relu") {
        h = nn::relu(h);
    } else if (config_.dense_act_fn == "gelu") {
        h = nn::gelu(h);
    } else {
        h = nn::relu(h);  // Default to ReLU
    }

    h = dropout_->forward(h);
    h = wo_->forward(h);
    return h;
}

// ============================================================================
// T5Block Implementation
// ============================================================================

T5Block::T5Block(const T5Config& config, bool has_relative_attention_bias, bool is_decoder)
    : config_(config), is_decoder_(is_decoder) {

    // Self-attention
    layer_norm_1_ = std::make_shared<T5LayerNorm>(config.d_model, config.layer_norm_epsilon);
    self_attention_ = std::make_shared<T5Attention>(config, has_relative_attention_bias);

    // Cross-attention (decoder only)
    if (is_decoder_) {
        layer_norm_2_ = std::make_shared<T5LayerNorm>(config.d_model, config.layer_norm_epsilon);
        cross_attention_ = std::make_shared<T5Attention>(config, false);
        register_module("layer_norm_2", layer_norm_2_);
        register_module("cross_attention", cross_attention_);
    }

    // Feed-forward
    layer_norm_ffn_ = std::make_shared<T5LayerNorm>(config.d_model, config.layer_norm_epsilon);
    ffn_ = std::make_shared<T5DenseActDense>(config);
    dropout_ = std::make_shared<nn::Dropout>(config.dropout_rate);

    register_module("layer_norm_1", layer_norm_1_);
    register_module("self_attention", self_attention_);
    register_module("layer_norm_ffn", layer_norm_ffn_);
    register_module("ffn", ffn_);
    register_module("dropout", dropout_);
}

auto T5Block::forward(const Variable& hidden_states,
                       const Variable& encoder_hidden_states,
                       const Tensor& attention_mask,
                       const Tensor& encoder_attention_mask,
                       const Tensor& position_bias,
                       const Tensor& encoder_position_bias)
    -> std::tuple<Variable, Tensor, Tensor> {
    // Call forward pre-hooks (enables CPU-start offloading)
    // NOTE: This is necessary because this multi-argument forward bypasses Module::forward()
    call_forward_pre_hooks();

    Tensor self_attn_bias;
    Tensor cross_attn_bias;

    // Pre-norm: normalize BEFORE self-attention
    Variable residual = hidden_states;  // Explicitly mutable copy
    auto normed = layer_norm_1_->forward(hidden_states);
    auto [attn_output, self_attn_bias_out] = self_attention_->forward(
        normed, Variable{}, attention_mask, position_bias
    );
    self_attn_bias = self_attn_bias_out;
    attn_output = dropout_->forward(attn_output);
    Variable hidden_states_mut = residual + attn_output;

    // Cross-attention (decoder only)
    if (is_decoder_ && encoder_hidden_states.is_initialized()) {
        residual = hidden_states_mut;
        normed = layer_norm_2_->forward(hidden_states_mut);
        auto [cross_output, cross_bias_out] = cross_attention_->forward(
            normed, encoder_hidden_states, encoder_attention_mask, encoder_position_bias
        );
        cross_attn_bias = cross_bias_out;
        cross_output = dropout_->forward(cross_output);
        hidden_states_mut = residual + cross_output;
    }

    // Pre-norm: normalize BEFORE feed-forward
    residual = hidden_states_mut;
    normed = layer_norm_ffn_->forward(hidden_states_mut);
    auto ffn_output = ffn_->forward(normed);
    ffn_output = dropout_->forward(ffn_output);
    hidden_states_mut = residual + ffn_output;

    // Call forward post-hooks (enables CPU-start offloading)
    call_forward_post_hooks();

    return {hidden_states_mut, self_attn_bias, cross_attn_bias};
}

auto T5Block::forward_impl(const Variable& input) -> Variable {
    auto [output, self_bias, cross_bias] = forward(
        input, Variable{}, Tensor{}, Tensor{}, Tensor{}, Tensor{}
    );
    return output;
}

// ============================================================================
// T5Encoder Implementation
// ============================================================================

T5Encoder::T5Encoder(const T5Config& config, std::shared_ptr<nn::Embedding> shared_embeddings)
    : config_(config), shared_embeddings_(shared_embeddings) {

    // Create encoder blocks
    for (int64_t i = 0; i < config.num_layers; ++i) {
        bool has_bias = (i == 0);  // Only first layer computes relative position bias
        auto block = std::make_shared<T5Block>(config, has_bias, false);
        blocks_.push_back(block);
        register_module("block_" + std::to_string(i), block);
    }

    // Final layer norm
    final_layer_norm_ = std::make_shared<T5LayerNorm>(config.d_model, config.layer_norm_epsilon);
    dropout_ = std::make_shared<nn::Dropout>(config.dropout_rate);

    register_module("final_layer_norm", final_layer_norm_);
    register_module("dropout", dropout_);
}

auto T5Encoder::forward(const Variable& input_ids,
                         const Tensor& attention_mask) -> Variable {
    // Call forward pre-hooks (enables CPU-start offloading)
    // NOTE: This is necessary because this multi-argument forward bypasses Module::forward()
    call_forward_pre_hooks();

    // Get token embeddings (no position embeddings in T5)
    auto hidden_states = shared_embeddings_->forward(input_ids);
    hidden_states = dropout_->forward(hidden_states);

    // Pass through encoder blocks
    Tensor position_bias;  // Computed in first layer, reused in rest
    for (size_t i = 0; i < blocks_.size(); ++i) {
        auto [output, self_bias, cross_bias] = blocks_[i]->forward(
            hidden_states, Variable{}, attention_mask, Tensor{}, position_bias, Tensor{}
        );
        hidden_states = output;
        if (i == 0) {
            position_bias = self_bias;  // Save bias from first layer
        }
    }

    // Final layer norm
    hidden_states = final_layer_norm_->forward(hidden_states);
    hidden_states = dropout_->forward(hidden_states);

    // Call forward post-hooks (enables CPU-start offloading)
    call_forward_post_hooks();

    return hidden_states;
}

auto T5Encoder::forward_impl(const Variable& input) -> Variable {
    return forward(input, Tensor{});
}

// ============================================================================
// T5Decoder Implementation
// ============================================================================

T5Decoder::T5Decoder(const T5Config& config, std::shared_ptr<nn::Embedding> shared_embeddings)
    : config_(config), shared_embeddings_(shared_embeddings) {

    // Create decoder blocks
    for (int64_t i = 0; i < config.num_layers; ++i) {
        bool has_bias = (i == 0);  // Only first layer computes relative position bias
        auto block = std::make_shared<T5Block>(config, has_bias, true);
        blocks_.push_back(block);
        register_module("block_" + std::to_string(i), block);
    }

    // Final layer norm
    final_layer_norm_ = std::make_shared<T5LayerNorm>(config.d_model, config.layer_norm_epsilon);
    dropout_ = std::make_shared<nn::Dropout>(config.dropout_rate);

    register_module("final_layer_norm", final_layer_norm_);
    register_module("dropout", dropout_);
}

auto T5Decoder::create_causal_mask(int64_t seq_len, Device device) -> Tensor {
    // Create causal mask: upper triangular matrix with -inf
    // Infer dtype from embedding weights to support multi-dtype models
    DType dtype = shared_embeddings_->weight().tensor().dtype();

    // Create on CPU for data filling, then transfer to target device
    Tensor mask_cpu({seq_len, seq_len}, dtype, Device::cpu());

    // Dtype-generic mask filling
    if (dtype == DType::Float16) {
        auto* data = mask_cpu.data<Float16>();
        for (int64_t i = 0; i < seq_len; ++i) {
            for (int64_t j = 0; j < seq_len; ++j) {
                data[i * seq_len + j] = (j <= i) ? Float16(0.0f) : Float16(-1e9f);
            }
        }
    } else if (dtype == DType::Float32) {
        auto* data = mask_cpu.data<float>();
        for (int64_t i = 0; i < seq_len; ++i) {
            for (int64_t j = 0; j < seq_len; ++j) {
                data[i * seq_len + j] = (j <= i) ? 0.0f : -1e9f;
            }
        }
    } else if (dtype == DType::Float64) {
        auto* data = mask_cpu.data<double>();
        for (int64_t i = 0; i < seq_len; ++i) {
            for (int64_t j = 0; j < seq_len; ++j) {
                data[i * seq_len + j] = (j <= i) ? 0.0 : -1e9;
            }
        }
    }

    // Transfer to target device if needed
    return (device == Device::cpu()) ? mask_cpu : mask_cpu.to(device);
}

auto T5Decoder::forward(const Variable& decoder_input_ids,
                         const Variable& encoder_hidden_states,
                         const Tensor& decoder_attention_mask,
                         const Tensor& encoder_attention_mask) -> Variable {
    // Call forward pre-hooks (enables CPU-start offloading)
    // NOTE: This is necessary because this multi-argument forward bypasses Module::forward()
    call_forward_pre_hooks();

    // Get token embeddings
    auto hidden_states = shared_embeddings_->forward(decoder_input_ids);
    hidden_states = dropout_->forward(hidden_states);

    // Create causal mask for decoder self-attention
    int64_t seq_len = decoder_input_ids.shape()[1];
    auto causal_mask = create_causal_mask(seq_len, decoder_input_ids.tensor().device());

    // Audit G12: real combination of causal mask with the decoder's padding
    // mask. Previously this branch just re-assigned `causal_mask` (the
    // padding mask was silently dropped), so generation past the first PAD
    // token would attend to padded positions.
    //
    // `decoder_attention_mask` follows the HuggingFace convention: shape
    // (B, T), 1=keep, 0=mask. T5 attention adds the mask additively to the
    // scores, so we convert binary → {-1e9, 0} via `(mask - 1) * 1e9`, then
    // reshape to (B, 1, 1, T) and broadcast-add with the (T, T) causal mask
    // reshaped to (1, 1, T, T). Result shape (B, 1, T, T) broadcasts to
    // attention scores (B, num_heads, T, T).
    Tensor combined_mask = causal_mask;
    if (decoder_attention_mask.is_valid() && decoder_attention_mask.numel() > 0) {
        const int64_t B = decoder_attention_mask.shape()[0];
        const DType dtype = causal_mask.dtype();

        Tensor pad = decoder_attention_mask.to(dtype).to(causal_mask.device());
        Tensor pad_additive = (pad - 1.0) * 1e9;  // 1.0→0, 0.0→-1e9
        pad_additive = pad_additive.reshape({B, 1, 1, seq_len});

        Tensor causal_4d = causal_mask.reshape({1, 1, seq_len, seq_len});
        combined_mask = causal_4d + pad_additive;  // (B, 1, T, T) via broadcast
    }

    // Pass through decoder blocks
    Tensor position_bias;         // Self-attention bias
    Tensor encoder_position_bias; // Cross-attention bias
    for (size_t i = 0; i < blocks_.size(); ++i) {
        auto [output, self_bias, cross_bias] = blocks_[i]->forward(
            hidden_states, encoder_hidden_states, combined_mask,
            encoder_attention_mask, position_bias, encoder_position_bias
        );
        hidden_states = output;
        if (i == 0) {
            position_bias = self_bias;
            encoder_position_bias = cross_bias;
        }
    }

    // Final layer norm
    hidden_states = final_layer_norm_->forward(hidden_states);
    hidden_states = dropout_->forward(hidden_states);

    // Call forward post-hooks (enables CPU-start offloading)
    call_forward_post_hooks();

    return hidden_states;
}

auto T5Decoder::forward_impl([[maybe_unused]] const Variable& input) -> Variable {
    // Decoder requires encoder outputs - use forward_with_encoder() instead
    throw std::runtime_error("T5Decoder::forward(input) requires encoder_hidden_states");
}

// ============================================================================
// T5Model Implementation
// ============================================================================

T5Model::T5Model(const T5Config& config)
    : config_(config) {
    // Shared token embeddings (encoder and decoder share weights)
    shared_embeddings_ = std::make_shared<nn::Embedding>(
        config.vocab_size, config.d_model);

    encoder_ = std::make_shared<T5Encoder>(config, shared_embeddings_);
    decoder_ = std::make_shared<T5Decoder>(config, shared_embeddings_);

    register_module("shared", shared_embeddings_);
    register_module("encoder", encoder_);
    register_module("decoder", decoder_);
}

auto T5Model::forward(const Variable& input_ids,
                       const Variable& decoder_input_ids,
                       const Tensor& attention_mask,
                       const Tensor& decoder_attention_mask) -> T5Output {
    // Call forward pre-hooks (enables CPU-start offloading)
    // NOTE: This is necessary because this multi-argument forward bypasses Module::forward()
    call_forward_pre_hooks();

    // Encode
    auto encoder_output = encoder_->forward(input_ids, attention_mask);

    // Decode
    auto decoder_output = decoder_->forward(
        decoder_input_ids, encoder_output,
        decoder_attention_mask, attention_mask
    );

    // Call forward post-hooks (enables CPU-start offloading)
    call_forward_post_hooks();

    return T5Output{encoder_output, decoder_output};
}

auto T5Model::forward_impl(const Variable& input) -> Variable {
    // Encoder-only mode
    return encoder_->forward(input, Tensor{});
}

auto T5Model::load_pretrained(const std::string& path, bool strict) -> void {
    // Audit H4. See AlbertModel::load_pretrained.
    ModelHub::load_pretrained_weights(*this, path, strict);
}

// ============================================================================
// T5ForConditionalGeneration Implementation
// ============================================================================

T5ForConditionalGeneration::T5ForConditionalGeneration(const T5Config& config)
    : config_(config) {
    t5_ = std::make_shared<T5Model>(config);
    lm_head_ = std::make_shared<nn::Linear>(config.d_model, config.vocab_size, false);

    register_module("t5", t5_);
    register_module("lm_head", lm_head_);
}

auto T5ForConditionalGeneration::forward(const Variable& input_ids,
                                          const Variable& decoder_input_ids,
                                          const Tensor& attention_mask,
                                          const Tensor& decoder_attention_mask) -> Variable {
    // Call forward pre-hooks (enables CPU-start offloading)
    // NOTE: This is necessary because this multi-argument forward bypasses Module::forward()
    call_forward_pre_hooks();

    auto outputs = t5_->forward(input_ids, decoder_input_ids,
                                attention_mask, decoder_attention_mask);
    auto logits = lm_head_->forward(outputs.decoder_output);

    // Call forward post-hooks (enables CPU-start offloading)
    call_forward_post_hooks();

    return logits;
}

auto T5ForConditionalGeneration::forward_impl([[maybe_unused]] const Variable& input) -> Variable {
    throw std::runtime_error("T5ForConditionalGeneration::forward requires decoder_input_ids");
}

auto T5ForConditionalGeneration::load_pretrained(const std::string& path, bool strict) -> void {
    // Audit H4. See AlbertModel::load_pretrained.
    ModelHub::load_pretrained_weights(*this, path, strict);
}

auto T5ForConditionalGeneration::generate(const Variable& input_ids,
                                           int64_t max_length,
                                           double temperature,
                                           int64_t bos_token_id) -> Tensor {
    // Simple greedy generation (can be extended with beam search, sampling, etc.)
    auto batch_size = input_ids.shape()[0];
    auto device = input_ids.tensor().device();

    // Audit G15: use the model's configured decoder_start_token_id instead
    // of hard-coding 0 ("assume token 0"). Caller can override via the
    // `bos_token_id` argument (e.g. for tokenizers whose start token differs
    // from the saved config). HuggingFace T5 convention: T5 starts decoding
    // from `decoder_start_token_id` rather than a separate BOS.
    const int64_t start_token = (bos_token_id >= 0)
        ? bos_token_id
        : config_.decoder_start_token_id;

    // Allocate on CPU for the int64 fill, then move to target device.
    Tensor generated_cpu({batch_size, 1}, DType::Int64, Device::cpu());
    auto* gp = generated_cpu.data<int64_t>();
    for (int64_t b = 0; b < batch_size; ++b) gp[b] = start_token;
    Tensor generated = (device == Device::cpu()) ? generated_cpu : generated_cpu.to(device);

    for (int64_t i = 0; i < max_length; ++i) {
        Variable decoder_ids(generated, false);
        auto logits = forward(input_ids, decoder_ids, Tensor{}, Tensor{});

        // Get logits for last position
        auto last_logits_shape = logits.shape();
        int64_t seq_len = last_logits_shape[1];
        int64_t vocab_size = last_logits_shape[2];

        // Extract last token logits [batch, vocab_size]. The greedy decode below
        // reads logits / writes token ids through raw host pointers, so this
        // bookkeeping runs on the CPU (logits live on the compute device).
        Tensor logits_host = logits.tensor();
        if (logits_host.device().type != Device::Type::CPU) logits_host = logits_host.cpu();
        Tensor last_logits({batch_size, vocab_size}, logits_host.dtype(), Device::cpu());

        // Dtype-generic last logits extraction
        DType logits_dtype = logits_host.dtype();
        if (logits_dtype == DType::Float16) {
            auto* src = logits_host.data<Float16>();
            auto* dst = last_logits.data<Float16>();
            for (int64_t b = 0; b < batch_size; ++b) {
                int64_t offset = b * seq_len * vocab_size + (seq_len - 1) * vocab_size;
                std::copy(src + offset, src + offset + vocab_size, dst + b * vocab_size);
            }
        } else if (logits_dtype == DType::Float32) {
            auto* src = logits_host.data<float>();
            auto* dst = last_logits.data<float>();
            for (int64_t b = 0; b < batch_size; ++b) {
                int64_t offset = b * seq_len * vocab_size + (seq_len - 1) * vocab_size;
                std::copy(src + offset, src + offset + vocab_size, dst + b * vocab_size);
            }
        } else if (logits_dtype == DType::Float64) {
            auto* src = logits_host.data<double>();
            auto* dst = last_logits.data<double>();
            for (int64_t b = 0; b < batch_size; ++b) {
                int64_t offset = b * seq_len * vocab_size + (seq_len - 1) * vocab_size;
                std::copy(src + offset, src + offset + vocab_size, dst + b * vocab_size);
            }
        }

        // Apply temperature
        if (temperature != 1.0) {
            if (logits_dtype == DType::Float16) {
                auto* data = last_logits.data<Float16>();
                for (int64_t j = 0; j < batch_size * vocab_size; ++j) {
                    data[j] = Float16(float(data[j]) / static_cast<float>(temperature));
                }
            } else if (logits_dtype == DType::Float32) {
                auto* data = last_logits.data<float>();
                for (int64_t j = 0; j < batch_size * vocab_size; ++j) {
                    data[j] /= temperature;
                }
            } else if (logits_dtype == DType::Float64) {
                auto* data = last_logits.data<double>();
                for (int64_t j = 0; j < batch_size * vocab_size; ++j) {
                    data[j] /= temperature;
                }
            }
        }

        // Greedy: argmax (host-side over CPU last_logits)
        Tensor next_tokens({batch_size, 1}, DType::Int64, Device::cpu());
        auto* tokens_data = next_tokens.data<int64_t>();

        if (logits_dtype == DType::Float16) {
            auto* logits_data = last_logits.data<Float16>();
            for (int64_t b = 0; b < batch_size; ++b) {
                int64_t max_idx = 0;
                float max_val = float(logits_data[b * vocab_size]);
                for (int64_t v = 1; v < vocab_size; ++v) {
                    float val = float(logits_data[b * vocab_size + v]);
                    if (val > max_val) {
                        max_val = val;
                        max_idx = v;
                    }
                }
                tokens_data[b] = max_idx;
            }
        } else if (logits_dtype == DType::Float32) {
            auto* logits_data = last_logits.data<float>();
            for (int64_t b = 0; b < batch_size; ++b) {
                int64_t max_idx = 0;
                float max_val = logits_data[b * vocab_size];
                for (int64_t v = 1; v < vocab_size; ++v) {
                    if (logits_data[b * vocab_size + v] > max_val) {
                        max_val = logits_data[b * vocab_size + v];
                        max_idx = v;
                    }
                }
                tokens_data[b] = max_idx;
            }
        } else if (logits_dtype == DType::Float64) {
            auto* logits_data = last_logits.data<double>();
            for (int64_t b = 0; b < batch_size; ++b) {
                int64_t max_idx = 0;
                double max_val = logits_data[b * vocab_size];
                for (int64_t v = 1; v < vocab_size; ++v) {
                    if (logits_data[b * vocab_size + v] > max_val) {
                        max_val = logits_data[b * vocab_size + v];
                        max_idx = v;
                    }
                }
                tokens_data[b] = max_idx;
            }
        }

        // Append to generated. Build on CPU (host pointer loop) then move back
        // to the compute device for the next decoder forward.
        Tensor generated_host = (generated.device().type != Device::Type::CPU)
            ? generated.cpu() : generated;
        Tensor new_generated({batch_size, i + 2}, DType::Int64, Device::cpu());
        auto* gen_src = generated_host.data<int64_t>();
        auto* gen_dst = new_generated.data<int64_t>();
        auto* next_data = next_tokens.data<int64_t>();
        for (int64_t b = 0; b < batch_size; ++b) {
            std::copy(gen_src + b * (i + 1), gen_src + (b + 1) * (i + 1),
                     gen_dst + b * (i + 2));
            gen_dst[b * (i + 2) + i + 1] = next_data[b];
        }
        generated = (device == Device::cpu()) ? new_generated : new_generated.to(device);
    }

    return generated;
}

} // namespace models
} // namespace tenzor
