/**
 * @file transformer.hpp
 * @brief Transformer encoder and decoder layers
 *
 * Implements the Transformer architecture from "Attention Is All You Need"
 * (Vaswani et al., 2017). Includes encoder layers, decoder layers, and
 * positional encoding.
 */

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "../module.hpp"
#include "attention.hpp"
#include "linear.hpp"
#include "dropout.hpp"
#include "normalization.hpp"

namespace tenzor {
namespace nn {

/**
 * @brief Positional Encoding for Transformer models.
 *
 * Adds positional information to input embeddings using sine and cosine
 * functions of different frequencies. This allows the model to make use
 * of sequence order information.
 *
 * Formula:
 * - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
 * - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
 *
 * Where:
 * - pos is the position in the sequence
 * - i is the dimension index
 * - d_model is the model dimension
 *
 * The positional encodings are precomputed and added to input embeddings.
 *
 * @code
 * PositionalEncoding pe(512, 5000, 0.1);
 * Variable embeddings(Tensor({batch, seq_len, 512}, DType::Float32, Device::cpu()), true);
 * Variable encoded = pe.forward(embeddings);
 * @endcode
 *
 * @see TransformerEncoderLayer for usage in transformer
 */
class PositionalEncoding : public Module {
public:
    /**
     * @brief Construct positional encoding layer.
     *
     * @param d_model Embedding dimension
     * @param max_len Maximum sequence length (default: 5000)
     * @param dropout Dropout probability (default: 0.0)
     *
     * @code
     * PositionalEncoding pe1(512);              // No dropout
     * PositionalEncoding pe2(768, 10000, 0.1);  // With dropout
     * @endcode
     */
    PositionalEncoding(int64_t d_model, int64_t max_len = 5000, double dropout = 0.0);

    /**
     * @brief Forward pass - add positional encoding to input.
     *
     * @param x Input tensor of shape (batch, seq_len, d_model) or (seq_len, batch, d_model)
     * @return Tensor with positional encoding added
     *
     * @throws std::runtime_error if seq_len exceeds max_len
     */
    auto forward_impl(const Variable& x) -> Variable override;

private:
    int64_t d_model_;        ///< Model dimension
    int64_t max_len_;        ///< Maximum sequence length
    Tensor pe_;              ///< Precomputed positional encodings (CPU, Float32)
    mutable Tensor pe_cached_;  ///< PE cached on target device/dtype (lazily populated)
    mutable std::mutex pe_mutex_;  ///< Guards pe_cached_ for thread safety
    std::shared_ptr<Dropout> dropout_;  ///< Dropout layer

    /**
     * @brief Initialize positional encoding matrix.
     */
    auto init_positional_encoding() -> void;
};

/**
 * @brief Transformer Encoder Layer.
 *
 * A single layer of the Transformer encoder, consisting of:
 * 1. Multi-head self-attention
 * 2. Add & Norm (residual connection + layer normalization)
 * 3. Feed-forward network (2-layer MLP)
 * 4. Add & Norm (residual connection + layer normalization)
 *
 * Architecture:
 * @code
 *   Input
 *     |
 *     v
 *   MultiheadAttention
 *     |
 *     v
 *   Dropout + Residual + LayerNorm
 *     |
 *     v
 *   FeedForward (Linear -> Activation -> Dropout -> Linear)
 *     |
 *     v
 *   Dropout + Residual + LayerNorm
 *     |
 *     v
 *   Output
 * @endcode
 *
 * @code
 * // Create encoder layer: 512 dims, 8 heads, 2048 FFN
 * TransformerEncoderLayer layer(512, 8, 2048, 0.1, "relu");
 *
 * Variable src(Tensor({batch, seq_len, 512}, DType::Float32, Device::cpu()), true);
 * Variable output = layer.forward(src);
 * @endcode
 */
class TransformerEncoderLayer : public Module {
public:
    // Bring base class forward into scope (avoid hiding by multi-param forward)
    using Module::forward;

    /**
     * @brief Construct transformer encoder layer.
     *
     * @param d_model Model dimension (embedding size)
     * @param nhead Number of attention heads
     * @param dim_feedforward Dimension of feedforward network (default: 2048)
     * @param dropout Dropout probability (default: 0.1)
     * @param activation Activation function: "relu" or "gelu" (default: "relu")
     * @param batch_first If true, input is (batch, seq, feature) else (seq, batch, feature)
     * @param norm_first If true, layer norm is applied before attention/FFN (Pre-LN);
     *                   if false, layer norm is applied after (Post-LN, the original
     *                   "Attention Is All You Need" architecture). Default: false.
     *
     * @throws std::invalid_argument if activation is not "relu" or "gelu"
     *
     * @code
     * TransformerEncoderLayer layer1(512, 8);                   // Default post-LN
     * TransformerEncoderLayer layer2(768, 12, 3072, 0.1, "gelu"); // BERT-like
     * TransformerEncoderLayer layer3(512, 8, 2048, 0.1, "relu", false, true); // Pre-LN
     * @endcode
     */
    TransformerEncoderLayer(int64_t d_model,
                           int64_t nhead,
                           int64_t dim_feedforward = 2048,
                           double dropout = 0.1,
                           const std::string& activation = "relu",
                           bool batch_first = false,
                           bool norm_first = false);

    /**
     * @brief Forward pass through encoder layer.
     *
     * @param src Source sequence (batch, seq_len, d_model) or (seq_len, batch, d_model)
     * @param src_mask Attention mask (seq_len, seq_len) - optional
     * @param src_key_padding_mask Padding mask (batch, seq_len) - optional
     * @return Encoded sequence (same shape as input)
     *
     * @code
     * // Without masks
     * Variable output = layer.forward(src);
     *
     * // With causal mask
     * Tensor mask = create_causal_mask(seq_len);
     * Variable output = layer.forward(src, mask);
     *
     * // With padding mask
     * Tensor pad_mask({batch, seq_len}, DType::Float32, Device::cpu());
     * Variable output = layer.forward(src, Tensor{}, pad_mask);
     * @endcode
     */
    auto forward(const Variable& src,
                const Tensor& src_mask = Tensor{},
                const Tensor& src_key_padding_mask = Tensor{}) -> Variable;

    /**
     * @brief Default forward (not meaningful for encoder layer, throws error).
     */
    auto forward_impl(const Variable& input) -> Variable override;

    // Allow TransformerEncoder to access configuration for layer cloning
    friend class TransformerEncoder;

private:
    int64_t d_model_;
    int64_t nhead_;
    int64_t dim_feedforward_;
    double dropout_;
    std::string activation_;
    bool batch_first_;
    bool norm_first_;  ///< If true, use Pre-LN (norm before sublayer); else Post-LN

    // Layers
    std::shared_ptr<MultiheadAttention> self_attn_;  ///< Self-attention
    std::shared_ptr<Linear> linear1_;                 ///< FFN first layer
    std::shared_ptr<Linear> linear2_;                 ///< FFN second layer
    std::shared_ptr<LayerNorm> norm1_;                ///< Layer norm after attention
    std::shared_ptr<LayerNorm> norm2_;                ///< Layer norm after FFN
    std::shared_ptr<Dropout> dropout1_;               ///< Dropout after attention
    std::shared_ptr<Dropout> dropout2_;               ///< Dropout in FFN
    std::shared_ptr<Dropout> dropout3_;               ///< Dropout after FFN

    /**
     * @brief Apply activation function.
     */
    auto apply_activation(const Variable& x) const -> Variable;
};

/**
 * @brief Transformer Encoder (stack of encoder layers).
 *
 * Stacks multiple TransformerEncoderLayer modules and optionally applies
 * a final layer normalization.
 *
 * @code
 * auto encoder_layer = std::make_shared<TransformerEncoderLayer>(512, 8);
 * TransformerEncoder encoder(encoder_layer, 6);  // 6 layers
 *
 * Variable src(Tensor({batch, seq_len, 512}, DType::Float32, Device::cpu()), true);
 * Variable output = encoder.forward(src);
 * @endcode
 */
class TransformerEncoder : public Module {
public:
    // Bring base class forward into scope (avoid hiding by multi-param forward)
    using Module::forward;

    /**
     * @brief Construct transformer encoder.
     *
     * @param encoder_layer Encoder layer to replicate
     * @param num_layers Number of encoder layers
     * @param norm Optional final layer normalization
     *
     * @code
     * auto layer = std::make_shared<TransformerEncoderLayer>(512, 8);
     * auto norm = std::make_shared<LayerNorm>(std::vector<int64_t>{512});
     * TransformerEncoder encoder(layer, 6, norm);
     * @endcode
     */
    TransformerEncoder(std::shared_ptr<TransformerEncoderLayer> encoder_layer,
                      int64_t num_layers,
                      std::shared_ptr<LayerNorm> norm = nullptr);

    /**
     * @brief Forward pass through all encoder layers.
     *
     * @param src Source sequence
     * @param mask Attention mask (optional)
     * @param src_key_padding_mask Padding mask (optional)
     * @return Encoded sequence
     */
    auto forward(const Variable& src,
                const Tensor& mask = Tensor{},
                const Tensor& src_key_padding_mask = Tensor{}) -> Variable;

    /**
     * @brief Default forward (not meaningful for encoder, throws error).
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Enable/disable gradient (activation) checkpointing.
     *
     * When enabled, each encoder layer's forward is wrapped in
     * autograd::checkpoint(): intermediate activations are not retained, and
     * the layer forward is recomputed during backward. This trades ~one extra
     * forward of compute for an O(num_layers) reduction in peak activation
     * memory — letting very deep encoders (e.g. ViT-Huge, 32 layers) train
     * within tight GPU memory. Gradients are identical (the checkpoint captures
     * and replays RNG state for dropout). Default: off.
     */
    auto set_gradient_checkpointing(bool enabled) -> void { gradient_checkpointing_ = enabled; }
    auto gradient_checkpointing() const -> bool { return gradient_checkpointing_; }

private:
    std::vector<std::shared_ptr<TransformerEncoderLayer>> layers_;
    std::shared_ptr<LayerNorm> norm_;
    int64_t num_layers_;
    bool gradient_checkpointing_{false};
};

/**
 * @brief Transformer Decoder Layer.
 *
 * A single layer of the Transformer decoder, consisting of:
 * 1. Masked multi-head self-attention
 * 2. Add & Norm
 * 3. Multi-head cross-attention (attend to encoder output)
 * 4. Add & Norm
 * 5. Feed-forward network
 * 6. Add & Norm
 *
 * Architecture:
 * @code
 *   Target Input          Encoder Memory
 *        |                      |
 *        v                      |
 *   Self-Attention              |
 *   (masked)                    |
 *        |                      |
 *        v                      |
 *   Add & Norm                  |
 *        |                      |
 *        +----> Cross-Attention <+
 *               |
 *               v
 *           Add & Norm
 *               |
 *               v
 *          FeedForward
 *               |
 *               v
 *           Add & Norm
 *               |
 *               v
 *             Output
 * @endcode
 *
 * @code
 * TransformerDecoderLayer layer(512, 8, 2048, 0.1);
 *
 * Variable tgt(Tensor({batch, tgt_len, 512}, DType::Float32, Device::cpu()), true);
 * Variable memory(Tensor({batch, src_len, 512}, DType::Float32, Device::cpu()), true);
 * Variable output = layer.forward(tgt, memory);
 * @endcode
 */
class TransformerDecoderLayer : public Module {
public:
    // Bring base class forward into scope (avoid hiding by multi-param forward)
    using Module::forward;

    /**
     * @brief Construct transformer decoder layer.
     *
     * @param d_model Model dimension
     * @param nhead Number of attention heads
     * @param dim_feedforward Dimension of feedforward network (default: 2048)
     * @param dropout Dropout probability (default: 0.1)
     * @param activation Activation function: "relu" or "gelu" (default: "relu")
     * @param batch_first If true, input is (batch, seq, feature)
     * @param norm_first If true, layer norm is applied before attention/FFN (Pre-LN);
     *                   if false, layer norm is applied after (Post-LN). Default: false.
     *
     * @code
     * TransformerDecoderLayer layer1(512, 8);
     * TransformerDecoderLayer layer2(768, 12, 3072, 0.1, "gelu", true);
     * TransformerDecoderLayer layer3(512, 8, 2048, 0.1, "relu", false, true); // Pre-LN
     * @endcode
     */
    TransformerDecoderLayer(int64_t d_model,
                           int64_t nhead,
                           int64_t dim_feedforward = 2048,
                           double dropout = 0.1,
                           const std::string& activation = "relu",
                           bool batch_first = false,
                           bool norm_first = false);

    /**
     * @brief Forward pass through decoder layer.
     *
     * @param tgt Target sequence
     * @param memory Encoder output (for cross-attention)
     * @param tgt_mask Target attention mask (causal mask for autoregressive)
     * @param memory_mask Memory attention mask
     * @param tgt_key_padding_mask Target padding mask
     * @param memory_key_padding_mask Memory padding mask
     * @return Decoded sequence
     *
     * @code
     * // Autoregressive decoding with causal mask
     * Tensor tgt_mask = create_causal_mask(tgt_len);
     * Variable output = layer.forward(tgt, memory, tgt_mask);
     * @endcode
     */
    auto forward(const Variable& tgt,
                const Variable& memory,
                const Tensor& tgt_mask = Tensor{},
                const Tensor& memory_mask = Tensor{},
                const Tensor& tgt_key_padding_mask = Tensor{},
                const Tensor& memory_key_padding_mask = Tensor{}) -> Variable;

    /**
     * @brief Default forward (not meaningful for decoder, throws error).
     */
    auto forward_impl(const Variable& input) -> Variable override;

    // Allow TransformerDecoder to access configuration for layer cloning
    friend class TransformerDecoder;

private:
    int64_t d_model_;
    int64_t nhead_;
    int64_t dim_feedforward_;
    double dropout_;
    std::string activation_;
    bool batch_first_;
    bool norm_first_;  ///< If true, use Pre-LN (norm before sublayer); else Post-LN

    // Layers
    std::shared_ptr<MultiheadAttention> self_attn_;       ///< Masked self-attention
    std::shared_ptr<MultiheadAttention> multihead_attn_;  ///< Cross-attention
    std::shared_ptr<Linear> linear1_;                     ///< FFN first layer
    std::shared_ptr<Linear> linear2_;                     ///< FFN second layer
    std::shared_ptr<LayerNorm> norm1_;                    ///< Norm after self-attn
    std::shared_ptr<LayerNorm> norm2_;                    ///< Norm after cross-attn
    std::shared_ptr<LayerNorm> norm3_;                    ///< Norm after FFN
    std::shared_ptr<Dropout> dropout1_;                   ///< Dropout after self-attn
    std::shared_ptr<Dropout> dropout2_;                   ///< Dropout after cross-attn
    std::shared_ptr<Dropout> dropout3_;                   ///< Dropout in FFN
    std::shared_ptr<Dropout> dropout4_;                   ///< Dropout after FFN

    /**
     * @brief Apply activation function.
     */
    auto apply_activation(const Variable& x) const -> Variable;
};

/**
 * @brief Transformer Decoder (stack of decoder layers).
 *
 * Stacks multiple TransformerDecoderLayer modules.
 *
 * @code
 * auto decoder_layer = std::make_shared<TransformerDecoderLayer>(512, 8);
 * TransformerDecoder decoder(decoder_layer, 6);
 *
 * Variable tgt(Tensor({batch, tgt_len, 512}, DType::Float32, Device::cpu()), true);
 * Variable memory(Tensor({batch, src_len, 512}, DType::Float32, Device::cpu()), true);
 * Variable output = decoder.forward(tgt, memory);
 * @endcode
 */
class TransformerDecoder : public Module {
public:
    // Bring base class forward into scope (avoid hiding by multi-param forward)
    using Module::forward;

    /**
     * @brief Construct transformer decoder.
     *
     * @param decoder_layer Decoder layer to replicate
     * @param num_layers Number of decoder layers
     * @param norm Optional final layer normalization
     */
    TransformerDecoder(std::shared_ptr<TransformerDecoderLayer> decoder_layer,
                      int64_t num_layers,
                      std::shared_ptr<LayerNorm> norm = nullptr);

    /**
     * @brief Forward pass through all decoder layers.
     *
     * @param tgt Target sequence
     * @param memory Encoder output
     * @param tgt_mask Target attention mask
     * @param memory_mask Memory attention mask
     * @param tgt_key_padding_mask Target padding mask
     * @param memory_key_padding_mask Memory padding mask
     * @return Decoded sequence
     */
    auto forward(const Variable& tgt,
                const Variable& memory,
                const Tensor& tgt_mask = Tensor{},
                const Tensor& memory_mask = Tensor{},
                const Tensor& tgt_key_padding_mask = Tensor{},
                const Tensor& memory_key_padding_mask = Tensor{}) -> Variable;

    /**
     * @brief Default forward (not meaningful for decoder, throws error).
     */
    auto forward_impl(const Variable& input) -> Variable override;

private:
    std::vector<std::shared_ptr<TransformerDecoderLayer>> layers_;
    std::shared_ptr<LayerNorm> norm_;
    int64_t num_layers_;
};

/**
 * @brief Complete Transformer model (encoder-decoder architecture).
 *
 * Combines encoder and decoder for sequence-to-sequence tasks.
 *
 * @code
 * Transformer model(512, 8, 6, 6, 2048, 0.1);
 *
 * Variable src(Tensor({batch, src_len, 512}, DType::Float32, Device::cpu()), true);
 * Variable tgt(Tensor({batch, tgt_len, 512}, DType::Float32, Device::cpu()), true);
 * Variable output = model.forward(src, tgt);
 * @endcode
 */
class Transformer : public Module {
public:
    // Bring base class forward into scope (avoid hiding by multi-param forward)
    using Module::forward;

    /**
     * @brief Construct complete transformer model.
     *
     * @param d_model Model dimension
     * @param nhead Number of attention heads
     * @param num_encoder_layers Number of encoder layers
     * @param num_decoder_layers Number of decoder layers
     * @param dim_feedforward FFN dimension (default: 2048)
     * @param dropout Dropout probability (default: 0.1)
     * @param activation Activation: "relu" or "gelu" (default: "relu")
     * @param batch_first Batch dimension first (default: false)
     * @param norm_first If true, use Pre-LN ordering in all layers (default: false)
     */
    Transformer(int64_t d_model,
               int64_t nhead,
               int64_t num_encoder_layers = 6,
               int64_t num_decoder_layers = 6,
               int64_t dim_feedforward = 2048,
               double dropout = 0.1,
               const std::string& activation = "relu",
               bool batch_first = false,
               bool norm_first = false);

    /**
     * @brief Forward pass through encoder and decoder.
     *
     * @param src Source sequence
     * @param tgt Target sequence
     * @param src_mask Source attention mask
     * @param tgt_mask Target attention mask
     * @param memory_mask Memory attention mask
     * @param src_key_padding_mask Source padding mask
     * @param tgt_key_padding_mask Target padding mask
     * @param memory_key_padding_mask Memory padding mask
     * @return Decoder output
     */
    auto forward(const Variable& src,
                const Variable& tgt,
                const Tensor& src_mask = Tensor{},
                const Tensor& tgt_mask = Tensor{},
                const Tensor& memory_mask = Tensor{},
                const Tensor& src_key_padding_mask = Tensor{},
                const Tensor& tgt_key_padding_mask = Tensor{},
                const Tensor& memory_key_padding_mask = Tensor{}) -> Variable;

    /**
     * @brief Default forward (not meaningful, throws error).
     */
    auto forward_impl(const Variable& input) -> Variable override;

private:
    std::shared_ptr<TransformerEncoder> encoder_;
    std::shared_ptr<TransformerDecoder> decoder_;
};

} // namespace nn
} // namespace tenzor
