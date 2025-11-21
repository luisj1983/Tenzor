/**
 * @file transformer.cpp
 * @brief Implementation of Transformer encoder and decoder layers
 */

#include "tenzor/nn/layers/transformer.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/autograd/ops.hpp"
#include <cmath>
#include <stdexcept>
#include <numbers>

namespace tenzor {
namespace nn {

// ============================================================================
// PositionalEncoding Implementation
// ============================================================================

PositionalEncoding::PositionalEncoding(int64_t d_model, int64_t max_len, double dropout)
    : d_model_(d_model), max_len_(max_len) {

    // Create dropout layer
    if (dropout > 0.0) {
        dropout_ = std::make_shared<Dropout>(dropout);
        register_module("dropout", dropout_);
    }

    // Initialize positional encoding
    init_positional_encoding();

    // Register as buffer (non-trainable)
    Variable pe_var(pe_, false);
    register_buffer("pe", pe_var);
}

auto PositionalEncoding::init_positional_encoding() -> void {
    // Create positional encoding matrix: (max_len, d_model)
    pe_ = zeros({max_len_, d_model_}, DType::Float32, Device::cpu());
    auto* pe_data = pe_.data<float>();

    // Compute positional encodings
    for (int64_t pos = 0; pos < max_len_; ++pos) {
        for (int64_t i = 0; i < d_model_; ++i) {
            double angle = static_cast<double>(pos) /
                          std::pow(10000.0, static_cast<double>(2 * (i / 2)) / static_cast<double>(d_model_));

            if (i % 2 == 0) {
                // Even indices: sin
                pe_data[pos * d_model_ + i] = static_cast<float>(std::sin(angle));
            } else {
                // Odd indices: cos
                pe_data[pos * d_model_ + i] = static_cast<float>(std::cos(angle));
            }
        }
    }
}

auto PositionalEncoding::forward(const Variable& x) -> Variable {
    // x shape: (batch, seq_len, d_model) - always batch_first format as per documentation
    auto shape = x.shape();

    if (shape.size() != 3) {
        throw std::runtime_error(
            "PositionalEncoding expects 3D input tensor with shape (batch, seq_len, d_model), got " +
            std::to_string(shape.size()) + "D tensor");
    }

    int64_t seq_len = shape[1]; // seq_len is always at index 1 for batch_first format

    if (seq_len > max_len_) {
        throw std::runtime_error(
            "Sequence length " + std::to_string(seq_len) +
            " exceeds maximum length " + std::to_string(max_len_));
    }

    // Extract positional encodings for this sequence length
    // pe_shape for batch_first=true: (1, seq_len, d_model)
    std::vector<int64_t> pe_shape = {1, seq_len, d_model_};

    // Get input device and dtype
    Device input_device = x.tensor().device();
    DType input_dtype = x.tensor().dtype();

    // Slice and reshape positional encoding on CPU first
    // Use pe_'s current dtype (may have been converted via to(dtype))
    DType pe_dtype = pe_.dtype();
    Tensor pe_slice = reshape(pe_, {max_len_, d_model_});

    // Take first seq_len rows with proper dtype handling
    std::vector<int64_t> slice_shape = {seq_len, d_model_};
    Tensor pe_for_seq = zeros({seq_len, d_model_}, pe_dtype, Device::cpu());

    // Copy data using appropriate dtype
    if (pe_dtype == DType::Float32) {
        auto* pe_src = pe_.data<float>();
        auto* pe_dst = pe_for_seq.data<float>();
        for (int64_t i = 0; i < seq_len * d_model_; ++i) {
            pe_dst[i] = pe_src[i];
        }
    } else if (pe_dtype == DType::Float64) {
        auto* pe_src = pe_.data<double>();
        auto* pe_dst = pe_for_seq.data<double>();
        for (int64_t i = 0; i < seq_len * d_model_; ++i) {
            pe_dst[i] = pe_src[i];
        }
    } else if (pe_dtype == DType::Float16) {
        auto* pe_src = pe_.data<uint16_t>();
        auto* pe_dst = pe_for_seq.data<uint16_t>();
        for (int64_t i = 0; i < seq_len * d_model_; ++i) {
            pe_dst[i] = pe_src[i];
        }
    }

    // Now move to target device if needed
    if (input_device != Device::cpu()) {
        pe_for_seq = pe_for_seq.to(input_device);
    }

    // Reshape to match input
    pe_for_seq = reshape(pe_for_seq, pe_shape);

    // Broadcast positional encoding to match batch size
    std::vector<int64_t> broadcast_shape(shape.begin(), shape.end());
    Tensor pe_broadcast = expand(pe_for_seq, broadcast_shape);

    // Convert to input dtype if pe_ dtype differs from input dtype
    if (pe_dtype != input_dtype) {
        pe_broadcast = pe_broadcast.to(input_dtype);
    }

    // Add to input
    Variable pe_var(pe_broadcast, false);
    Variable result = x + pe_var;

    // Apply dropout if configured
    if (dropout_) {
        result = dropout_->forward(result);
    }

    return result;
}

// ============================================================================
// TransformerEncoderLayer Implementation
// ============================================================================

TransformerEncoderLayer::TransformerEncoderLayer(int64_t d_model,
                                                 int64_t nhead,
                                                 int64_t dim_feedforward,
                                                 double dropout,
                                                 const std::string& activation,
                                                 bool batch_first)
    : d_model_(d_model),
      nhead_(nhead),
      dim_feedforward_(dim_feedforward),
      dropout_(dropout),
      activation_(activation),
      batch_first_(batch_first) {

    // Validate activation
    if (activation_ != "relu" && activation_ != "gelu") {
        throw std::invalid_argument(
            "activation must be 'relu' or 'gelu', got: " + activation_);
    }

    // Create layers
    self_attn_ = std::make_shared<MultiheadAttention>(
        d_model_, nhead_, dropout_, true, false, false, 0, 0, batch_first_);

    linear1_ = std::make_shared<Linear>(d_model_, dim_feedforward_);
    linear2_ = std::make_shared<Linear>(dim_feedforward_, d_model_);

    norm1_ = std::make_shared<LayerNorm>(std::vector<int64_t>{d_model_});
    norm2_ = std::make_shared<LayerNorm>(std::vector<int64_t>{d_model_});

    dropout1_ = std::make_shared<Dropout>(dropout_);
    dropout2_ = std::make_shared<Dropout>(dropout_);
    dropout3_ = std::make_shared<Dropout>(dropout_);

    // Register modules
    register_module("self_attn", self_attn_);
    register_module("linear1", linear1_);
    register_module("linear2", linear2_);
    register_module("norm1", norm1_);
    register_module("norm2", norm2_);
    register_module("dropout1", dropout1_);
    register_module("dropout2", dropout2_);
    register_module("dropout3", dropout3_);
}

auto TransformerEncoderLayer::apply_activation(const Variable& x) const -> Variable {
    if (activation_ == "relu") {
        return relu(x);
    } else if (activation_ == "gelu") {
        return gelu(x);
    }
    return x;
}

auto TransformerEncoderLayer::forward(const Variable& src,
                                     const Tensor& src_mask,
                                     const Tensor& src_key_padding_mask) -> Variable {
    // Self-attention block
    auto [attn_output, _] = self_attn_->forward(src, src, src,
                                                 src_key_padding_mask,
                                                 src_mask,
                                                 false);

    // Dropout + residual + norm (use unique variable names to avoid overwriting)
    Variable residual1 = src + dropout1_->forward(attn_output);
    Variable x_norm1 = norm1_->forward(residual1);

    // Feed-forward block
    Variable ff1 = linear1_->forward(x_norm1);
    Variable ff_act = apply_activation(ff1);
    Variable ff_drop = dropout2_->forward(ff_act);
    Variable ff_output = linear2_->forward(ff_drop);

    // Dropout + residual + norm (use unique variable names)
    Variable residual2 = x_norm1 + dropout3_->forward(ff_output);
    Variable output = norm2_->forward(residual2);

    return output;
}

auto TransformerEncoderLayer::forward(const Variable& input) -> Variable {
    throw std::runtime_error(
        "TransformerEncoderLayer requires source sequence input. "
        "Use forward(src, src_mask, src_key_padding_mask) instead.");
}

// ============================================================================
// TransformerEncoder Implementation
// ============================================================================

TransformerEncoder::TransformerEncoder(
    std::shared_ptr<TransformerEncoderLayer> encoder_layer,
    int64_t num_layers,
    std::shared_ptr<LayerNorm> norm)
    : norm_(norm), num_layers_(num_layers) {

    // Create independent copies of the encoder layer for each position
    // Each layer must have its own parameters and internal state
    for (int64_t i = 0; i < num_layers_; ++i) {
        // Create a new instance for each layer to ensure proper state isolation
        auto layer_copy = std::make_shared<TransformerEncoderLayer>(
            encoder_layer->d_model_,
            encoder_layer->nhead_,
            encoder_layer->dim_feedforward_,
            encoder_layer->dropout_,
            encoder_layer->activation_,
            encoder_layer->batch_first_
        );
        layers_.push_back(layer_copy);
        register_module("layer_" + std::to_string(i), layer_copy);
    }

    if (norm_) {
        register_module("norm", norm_);
    }
}

auto TransformerEncoder::forward(const Variable& src,
                                const Tensor& mask,
                                const Tensor& src_key_padding_mask) -> Variable {
    // Pass through first encoder layer using original src (no copy)
    Variable output = layers_[0]->forward(src, mask, src_key_padding_mask);

    // Pass through remaining encoder layers
    for (size_t i = 1; i < layers_.size(); ++i) {
        output = layers_[i]->forward(output, mask, src_key_padding_mask);
    }

    // Apply final normalization if provided
    if (norm_) {
        output = norm_->forward(output);
    }

    return output;
}

auto TransformerEncoder::forward(const Variable& input) -> Variable {
    throw std::runtime_error(
        "TransformerEncoder requires source sequence input. "
        "Use forward(src, mask, src_key_padding_mask) instead.");
}

// ============================================================================
// TransformerDecoderLayer Implementation
// ============================================================================

TransformerDecoderLayer::TransformerDecoderLayer(int64_t d_model,
                                                 int64_t nhead,
                                                 int64_t dim_feedforward,
                                                 double dropout,
                                                 const std::string& activation,
                                                 bool batch_first)
    : d_model_(d_model),
      nhead_(nhead),
      dim_feedforward_(dim_feedforward),
      dropout_(dropout),
      activation_(activation),
      batch_first_(batch_first) {

    // Validate activation
    if (activation_ != "relu" && activation_ != "gelu") {
        throw std::invalid_argument(
            "activation must be 'relu' or 'gelu', got: " + activation_);
    }

    // Create layers
    self_attn_ = std::make_shared<MultiheadAttention>(
        d_model_, nhead_, dropout_, true, false, false, 0, 0, batch_first_);

    multihead_attn_ = std::make_shared<MultiheadAttention>(
        d_model_, nhead_, dropout_, true, false, false, 0, 0, batch_first_);

    linear1_ = std::make_shared<Linear>(d_model_, dim_feedforward_);
    linear2_ = std::make_shared<Linear>(dim_feedforward_, d_model_);

    norm1_ = std::make_shared<LayerNorm>(std::vector<int64_t>{d_model_});
    norm2_ = std::make_shared<LayerNorm>(std::vector<int64_t>{d_model_});
    norm3_ = std::make_shared<LayerNorm>(std::vector<int64_t>{d_model_});

    dropout1_ = std::make_shared<Dropout>(dropout_);
    dropout2_ = std::make_shared<Dropout>(dropout_);
    dropout3_ = std::make_shared<Dropout>(dropout_);
    dropout4_ = std::make_shared<Dropout>(dropout_);

    // Register modules
    register_module("self_attn", self_attn_);
    register_module("multihead_attn", multihead_attn_);
    register_module("linear1", linear1_);
    register_module("linear2", linear2_);
    register_module("norm1", norm1_);
    register_module("norm2", norm2_);
    register_module("norm3", norm3_);
    register_module("dropout1", dropout1_);
    register_module("dropout2", dropout2_);
    register_module("dropout3", dropout3_);
    register_module("dropout4", dropout4_);
}

auto TransformerDecoderLayer::apply_activation(const Variable& x) const -> Variable {
    if (activation_ == "relu") {
        return relu(x);
    } else if (activation_ == "gelu") {
        return gelu(x);
    }
    return x;
}

auto TransformerDecoderLayer::forward(const Variable& tgt,
                                     const Variable& memory,
                                     const Tensor& tgt_mask,
                                     const Tensor& memory_mask,
                                     const Tensor& tgt_key_padding_mask,
                                     const Tensor& memory_key_padding_mask) -> Variable {
    // Masked self-attention block
    auto [self_attn_output, _] = self_attn_->forward(tgt, tgt, tgt,
                                                      tgt_key_padding_mask,
                                                      tgt_mask,
                                                      false);

    // Dropout + residual + norm (use unique variable names)
    Variable residual1 = tgt + dropout1_->forward(self_attn_output);
    Variable x_norm1 = norm1_->forward(residual1);

    // Cross-attention block (attend to encoder memory)
    auto [cross_attn_output, __] = multihead_attn_->forward(x_norm1, memory, memory,
                                                             memory_key_padding_mask,
                                                             memory_mask,
                                                             false);

    // Dropout + residual + norm (use unique variable names)
    Variable residual2 = x_norm1 + dropout2_->forward(cross_attn_output);
    Variable x_norm2 = norm2_->forward(residual2);

    // Feed-forward block
    Variable ff1 = linear1_->forward(x_norm2);
    Variable ff_act = apply_activation(ff1);
    Variable ff_drop = dropout3_->forward(ff_act);
    Variable ff_output = linear2_->forward(ff_drop);

    // Dropout + residual + norm (use unique variable names)
    Variable residual3 = x_norm2 + dropout4_->forward(ff_output);
    Variable output = norm3_->forward(residual3);

    return output;
}

auto TransformerDecoderLayer::forward(const Variable& input) -> Variable {
    throw std::runtime_error(
        "TransformerDecoderLayer requires both target and memory inputs. "
        "Use forward(tgt, memory, ...) instead.");
}

// ============================================================================
// TransformerDecoder Implementation
// ============================================================================

TransformerDecoder::TransformerDecoder(
    std::shared_ptr<TransformerDecoderLayer> decoder_layer,
    int64_t num_layers,
    std::shared_ptr<LayerNorm> norm)
    : norm_(norm), num_layers_(num_layers) {

    // Create independent copies of the decoder layer for each position
    // Each layer must have its own parameters and internal state
    for (int64_t i = 0; i < num_layers_; ++i) {
        // Create a new instance for each layer to ensure proper state isolation
        auto layer_copy = std::make_shared<TransformerDecoderLayer>(
            decoder_layer->d_model_,
            decoder_layer->nhead_,
            decoder_layer->dim_feedforward_,
            decoder_layer->dropout_,
            decoder_layer->activation_,
            decoder_layer->batch_first_
        );
        layers_.push_back(layer_copy);
        register_module("layer_" + std::to_string(i), layer_copy);
    }

    if (norm_) {
        register_module("norm", norm_);
    }
}

auto TransformerDecoder::forward(const Variable& tgt,
                                const Variable& memory,
                                const Tensor& tgt_mask,
                                const Tensor& memory_mask,
                                const Tensor& tgt_key_padding_mask,
                                const Tensor& memory_key_padding_mask) -> Variable {
    // Pass through first decoder layer using original tgt (no copy)
    Variable output = layers_[0]->forward(tgt, memory, tgt_mask, memory_mask,
                                         tgt_key_padding_mask, memory_key_padding_mask);

    // Pass through remaining decoder layers
    for (size_t i = 1; i < layers_.size(); ++i) {
        output = layers_[i]->forward(output, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask);
    }

    // Apply final normalization if provided
    if (norm_) {
        output = norm_->forward(output);
    }

    return output;
}

auto TransformerDecoder::forward(const Variable& input) -> Variable {
    throw std::runtime_error(
        "TransformerDecoder requires both target and memory inputs. "
        "Use forward(tgt, memory, ...) instead.");
}

// ============================================================================
// Transformer Implementation
// ============================================================================

Transformer::Transformer(int64_t d_model,
                        int64_t nhead,
                        int64_t num_encoder_layers,
                        int64_t num_decoder_layers,
                        int64_t dim_feedforward,
                        double dropout,
                        const std::string& activation,
                        bool batch_first) {

    // Create encoder
    auto encoder_layer = std::make_shared<TransformerEncoderLayer>(
        d_model, nhead, dim_feedforward, dropout, activation, batch_first);

    auto encoder_norm = std::make_shared<LayerNorm>(std::vector<int64_t>{d_model});

    encoder_ = std::make_shared<TransformerEncoder>(
        encoder_layer, num_encoder_layers, encoder_norm);

    // Create decoder
    auto decoder_layer = std::make_shared<TransformerDecoderLayer>(
        d_model, nhead, dim_feedforward, dropout, activation, batch_first);

    auto decoder_norm = std::make_shared<LayerNorm>(std::vector<int64_t>{d_model});

    decoder_ = std::make_shared<TransformerDecoder>(
        decoder_layer, num_decoder_layers, decoder_norm);

    // Register modules
    register_module("encoder", encoder_);
    register_module("decoder", decoder_);
}

auto Transformer::forward(const Variable& src,
                         const Variable& tgt,
                         const Tensor& src_mask,
                         const Tensor& tgt_mask,
                         const Tensor& memory_mask,
                         const Tensor& src_key_padding_mask,
                         const Tensor& tgt_key_padding_mask,
                         const Tensor& memory_key_padding_mask) -> Variable {

    // Encode source sequence
    Variable memory = encoder_->forward(src, src_mask, src_key_padding_mask);

    // Decode target sequence
    Variable output = decoder_->forward(tgt, memory, tgt_mask, memory_mask,
                                       tgt_key_padding_mask, memory_key_padding_mask);

    return output;
}

auto Transformer::forward(const Variable& input) -> Variable {
    throw std::runtime_error(
        "Transformer requires both source and target inputs. "
        "Use forward(src, tgt, ...) instead.");
}

} // namespace nn
} // namespace tenzor
