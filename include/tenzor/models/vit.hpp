/**
 * @file vit.hpp
 * @brief Vision Transformer (ViT) model family
 *
 * Implements Vision Transformer architecture and its variants for computer vision tasks:
 * - Base ViT model for feature extraction
 * - Image classification
 * - Multiple architectural variants (Base, Large, Huge)
 *
 * Reference: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
 * (Dosovitskiy et al., 2020)
 */

#pragma once

#include <memory>
#include <string>
#include "../nn/module.hpp"
#include "../nn/layers/linear.hpp"
#include "../nn/layers/dropout.hpp"
#include "../nn/layers/normalization.hpp"
#include "../nn/layers/transformer.hpp"
#include "../nn/layers/conv.hpp"
#include "../core/tensor.hpp"
#include "../autograd/variable.hpp"

namespace tenzor {
namespace models {

/**
 * @brief Configuration for Vision Transformer models
 */
struct ViTConfig {
    // Architecture hyperparameters
    int64_t image_size = 224;                ///< Input image size (H = W)
    int64_t patch_size = 16;                 ///< Size of each patch (P x P)
    int64_t num_channels = 3;                ///< Number of input channels (RGB)
    int64_t hidden_size = 768;               ///< Hidden dimension (embedding dim)
    int64_t num_hidden_layers = 12;          ///< Number of transformer encoder layers
    int64_t num_attention_heads = 12;        ///< Number of attention heads
    int64_t intermediate_size = 3072;        ///< FFN intermediate dimension (4 * hidden_size)
    double hidden_dropout_prob = 0.0;        ///< Hidden layer dropout (standard ViT uses 0.0)
    double attention_probs_dropout_prob = 0.0; ///< Attention dropout (standard ViT uses 0.0)
    double layer_norm_eps = 1e-6;            ///< Layer norm epsilon
    std::string hidden_act = "gelu";         ///< Activation function (always GELU for ViT)
    bool qkv_bias = true;                    ///< Whether to add bias to Q, K, V projections

    /**
     * @brief Create ViT-Base/16 configuration
     *
     * Architecture: 12 layers, 768 hidden, 12 heads, 3072 FFN dim
     * Parameters: ~86M
     * Patch size: 16x16
     */
    static ViTConfig base_patch16(int64_t img_size = 224) {
        ViTConfig config;
        config.image_size = img_size;
        config.patch_size = 16;
        config.hidden_size = 768;
        config.num_hidden_layers = 12;
        config.num_attention_heads = 12;
        config.intermediate_size = 3072;
        return config;
    }

    /**
     * @brief Create ViT-Base/32 configuration
     *
     * Same as Base/16 but with larger patches (32x32)
     */
    static ViTConfig base_patch32(int64_t img_size = 224) {
        ViTConfig config = base_patch16(img_size);
        config.patch_size = 32;
        return config;
    }

    /**
     * @brief Create ViT-Large/16 configuration
     *
     * Architecture: 24 layers, 1024 hidden, 16 heads, 4096 FFN dim
     * Parameters: ~307M
     */
    static ViTConfig large_patch16(int64_t img_size = 224) {
        ViTConfig config;
        config.image_size = img_size;
        config.patch_size = 16;
        config.hidden_size = 1024;
        config.num_hidden_layers = 24;
        config.num_attention_heads = 16;
        config.intermediate_size = 4096;
        return config;
    }

    /**
     * @brief Create ViT-Large/32 configuration
     */
    static ViTConfig large_patch32(int64_t img_size = 224) {
        ViTConfig config = large_patch16(img_size);
        config.patch_size = 32;
        return config;
    }

    /**
     * @brief Create ViT-Huge/14 configuration
     *
     * Architecture: 32 layers, 1280 hidden, 16 heads, 5120 FFN dim
     * Parameters: ~632M
     */
    static ViTConfig huge_patch14(int64_t img_size = 224) {
        ViTConfig config;
        config.image_size = img_size;
        config.patch_size = 14;
        config.hidden_size = 1280;
        config.num_hidden_layers = 32;
        config.num_attention_heads = 16;
        config.intermediate_size = 5120;
        return config;
    }

    /**
     * @brief Create ViT-Huge/16 configuration
     */
    static ViTConfig huge_patch16(int64_t img_size = 224) {
        ViTConfig config = huge_patch14(img_size);
        config.patch_size = 16;
        return config;
    }

    /**
     * @brief Get number of patches (H/P * W/P)
     */
    int64_t num_patches() const {
        int64_t patches_h = image_size / patch_size;
        int64_t patches_w = image_size / patch_size;
        return patches_h * patches_w;
    }

    /**
     * @brief Get sequence length (num_patches + 1 for [CLS] token)
     */
    int64_t seq_length() const {
        return num_patches() + 1;
    }
};

/**
 * @brief Patch Embedding layer
 *
 * Converts input images into patch embeddings using a convolutional layer.
 *
 * Architecture:
 * @code
 *   Image [B, C, H, W]
 *     |
 *     v
 *   Conv2D (kernel=P, stride=P) -> [B, hidden_size, H/P, W/P]
 *     |
 *     v
 *   Flatten & Transpose -> [B, num_patches, hidden_size]
 * @endcode
 *
 * This is mathematically equivalent to:
 * 1. Dividing image into non-overlapping patches
 * 2. Flattening each patch
 * 3. Linear projection to hidden_size
 */
class PatchEmbedding : public nn::Module {
public:
    /**
     * @brief Construct patch embedding layer
     *
     * @param image_size Input image size (H = W)
     * @param patch_size Patch size (P x P)
     * @param num_channels Number of input channels
     * @param hidden_size Output embedding dimension
     */
    PatchEmbedding(int64_t image_size, int64_t patch_size,
                   int64_t num_channels, int64_t hidden_size);

    /**
     * @brief Forward pass through patch embedding
     *
     * @param x Input images [batch, channels, height, width]
     * @return Patch embeddings [batch, num_patches, hidden_size]
     */
    auto forward_impl(const Variable& x) -> Variable override;

    /**
     * @brief Get number of patches
     */
    int64_t num_patches() const { return num_patches_; }

private:
    int64_t image_size_;
    int64_t patch_size_;
    int64_t num_channels_;
    int64_t hidden_size_;
    int64_t num_patches_;

    std::shared_ptr<nn::Conv2d> projection_;  ///< Convolutional projection
};

/**
 * @brief Vision Transformer Embeddings
 *
 * Combines patch embeddings, [CLS] token, and position embeddings.
 *
 * Architecture:
 * @code
 *   Image -> PatchEmbedding -> [B, N, D]
 *     |
 *     v
 *   Prepend [CLS] token -> [B, N+1, D]
 *     |
 *     v
 *   Add Position Embeddings -> [B, N+1, D]
 *     |
 *     v
 *   Dropout -> Output
 * @endcode
 */
class ViTEmbeddings : public nn::Module {
public:
    /**
     * @brief Construct ViT embeddings layer
     *
     * @param config ViT configuration
     */
    explicit ViTEmbeddings(const ViTConfig& config);

    /**
     * @brief Forward pass through embeddings
     *
     * @param pixel_values Input images [batch, channels, height, width]
     * @return Embeddings [batch, seq_len, hidden_size]
     *         where seq_len = num_patches + 1 (for [CLS] token)
     */
    auto forward_impl(const Variable& pixel_values) -> Variable override;

private:
    ViTConfig config_;
    std::shared_ptr<PatchEmbedding> patch_embeddings_;
    Variable cls_token_;          ///< [CLS] token embedding [1, 1, hidden_size]
    Variable position_embeddings_; ///< Position embeddings [1, seq_len, hidden_size]
    std::shared_ptr<nn::Dropout> dropout_;

    /**
     * @brief Initialize learnable parameters
     */
    auto initialize_parameters() -> void;
};

/**
 * @brief Vision Transformer Encoder
 *
 * Stack of transformer encoder layers using pre-normalization (LayerNorm before attention/FFN).
 */
class ViTEncoder : public nn::Module {
public:
    /**
     * @brief Construct ViT encoder
     *
     * @param config ViT configuration
     */
    explicit ViTEncoder(const ViTConfig& config);

    /**
     * @brief Forward pass through encoder
     *
     * @param hidden_states Input embeddings [batch, seq_len, hidden_size]
     * @return Encoded sequence [batch, seq_len, hidden_size]
     */
    auto forward_impl(const Variable& hidden_states) -> Variable override;

    /// Enable/disable activation (gradient) checkpointing on the underlying
    /// transformer encoder layers. Off by default.
    auto set_gradient_checkpointing(bool enabled) -> void {
        if (encoder_) encoder_->set_gradient_checkpointing(enabled);
    }

private:
    ViTConfig config_;
    std::shared_ptr<nn::TransformerEncoder> encoder_;
};

/**
 * @brief Output structure for ViT models
 */
struct ViTOutput {
    Variable last_hidden_state;  ///< Token-level representations [batch, seq_len, hidden_size]
    Variable pooler_output;      ///< [CLS] token representation [batch, hidden_size]
};

/**
 * @brief Base Vision Transformer model
 *
 * The core ViT model that outputs both sequence-level and pooled representations.
 * Can be used for feature extraction or as a base for task-specific models.
 *
 * Example usage:
 * ```cpp
 * auto config = ViTConfig::base_patch16(224);
 * auto vit = ViT(config);
 *
 * // Input: [batch, 3, 224, 224]
 * Variable images(Tensor({batch, 3, 224, 224}, DType::Float32, Device::cpu()), true);
 * auto outputs = vit.forward(images);
 * auto features = outputs.last_hidden_state;  // [batch, 197, 768]
 * auto cls_features = outputs.pooler_output;  // [batch, 768]
 * ```
 */
class ViT : public nn::Module {
public:
    /**
     * @brief Construct Vision Transformer model
     *
     * @param config ViT configuration
     * @param add_pooling_layer Whether to include pooling layer (default: true)
     */
    explicit ViT(const ViTConfig& config, bool add_pooling_layer = true);

    /**
     * @brief Forward pass through ViT
     *
     * @param pixel_values Input images [batch, channels, height, width]
     * @return ViTOutput with last_hidden_state and pooler_output
     */
    auto forward_vit(const Variable& pixel_values) -> ViTOutput;

    // Module interface implementation - returns pooler output
    auto forward_impl(const Variable& input) -> Variable override {
        auto output = forward_vit(input);
        return output.pooler_output;
    }

    /**
     * @brief Get model configuration
     */
    auto config() const -> const ViTConfig& { return config_; }

    /// Enable/disable activation (gradient) checkpointing on the encoder.
    auto set_gradient_checkpointing(bool enabled) -> void {
        if (encoder_) encoder_->set_gradient_checkpointing(enabled);
    }

private:
    ViTConfig config_;
    bool add_pooling_layer_;
    std::shared_ptr<ViTEmbeddings> embeddings_;
    std::shared_ptr<ViTEncoder> encoder_;
    std::shared_ptr<nn::LayerNorm> layernorm_;  ///< Final layer norm
    std::shared_ptr<nn::Linear> pooler_;         ///< Optional pooler (linear + tanh)
};

/**
 * @brief Vision Transformer for image classification
 *
 * ViT model with a classification head on top of the [CLS] token.
 * Suitable for:
 * - Image classification (ImageNet, etc.)
 * - Fine-grained recognition
 * - Transfer learning on custom datasets
 *
 * Example usage:
 * ```cpp
 * auto config = ViTConfig::base_patch16(224);
 * auto classifier = ViTForImageClassification(config, 1000);  // ImageNet classes
 *
 * Variable images(Tensor({batch, 3, 224, 224}, DType::Float32, Device::cpu()), true);
 * Variable logits = classifier.forward(images);  // [batch, 1000]
 * ```
 */
class ViTForImageClassification : public nn::Module {
public:
    /**
     * @brief Construct ViT classifier
     *
     * @param config ViT configuration
     * @param num_labels Number of classes
     */
    ViTForImageClassification(const ViTConfig& config, int64_t num_labels);

    /**
     * @brief Forward pass
     *
     * @param pixel_values Input images [batch, channels, height, width]
     * @return Classification logits [batch, num_labels]
     */
    auto forward_impl(const Variable& pixel_values) -> Variable override;

    /// Enable/disable activation (gradient) checkpointing on the ViT encoder.
    /// Recomputes layer activations in backward to cut peak memory (lets
    /// ViT-Huge train within tight GPU memory); gradients are unchanged.
    auto set_gradient_checkpointing(bool enabled) -> void {
        if (vit_) vit_->set_gradient_checkpointing(enabled);
    }

private:
    ViTConfig config_;
    int64_t num_labels_;
    std::shared_ptr<ViT> vit_;
    std::shared_ptr<nn::Linear> classifier_;
};

// ============================================================================
// Factory functions for pretrained variants
// ============================================================================

/**
 * @brief Create ViT-Base/16 model
 *
 * @param num_classes Number of output classes (0 for feature extraction)
 * @param pretrained Whether to load pretrained weights
 * @param img_size Input image size (default: 224)
 * @return ViT model
 */
auto ViT_Base_Patch16(int64_t num_classes = 1000, bool pretrained = false,
                     int64_t img_size = 224) -> std::shared_ptr<ViTForImageClassification>;

/**
 * @brief Create ViT-Base/32 model
 */
auto ViT_Base_Patch32(int64_t num_classes = 1000, bool pretrained = false,
                     int64_t img_size = 224) -> std::shared_ptr<ViTForImageClassification>;

/**
 * @brief Create ViT-Large/16 model
 *
 * @param num_classes Number of output classes
 * @param pretrained Whether to load pretrained weights
 * @param img_size Input image size (default: 224)
 * @return ViT model
 */
auto ViT_Large_Patch16(int64_t num_classes = 1000, bool pretrained = false,
                      int64_t img_size = 224) -> std::shared_ptr<ViTForImageClassification>;

/**
 * @brief Create ViT-Large/32 model
 */
auto ViT_Large_Patch32(int64_t num_classes = 1000, bool pretrained = false,
                      int64_t img_size = 224) -> std::shared_ptr<ViTForImageClassification>;

/**
 * @brief Create ViT-Huge/14 model
 *
 * @param num_classes Number of output classes
 * @param pretrained Whether to load pretrained weights
 * @param img_size Input image size (default: 224)
 * @return ViT model
 */
auto ViT_Huge_Patch14(int64_t num_classes = 1000, bool pretrained = false,
                     int64_t img_size = 224) -> std::shared_ptr<ViTForImageClassification>;

/**
 * @brief Create ViT-Huge/16 model
 */
auto ViT_Huge_Patch16(int64_t num_classes = 1000, bool pretrained = false,
                     int64_t img_size = 224) -> std::shared_ptr<ViTForImageClassification>;

/**
 * @brief Model hub utilities for loading pretrained ViT models
 */
class ViTModelHub {
public:
    /**
     * @brief Download pretrained ViT checkpoint
     *
     * @param model_name Model name (e.g., "vit_base_patch16_224")
     * @return Path to downloaded checkpoint
     */
    static auto download_pretrained(const std::string& model_name) -> std::string;

    /**
     * @brief Load pretrained weights into ViT model
     *
     * @param model ViT model
     * @param checkpoint_path Path to checkpoint
     */
    static auto load_pretrained_weights(nn::Module& model,
                                       const std::string& checkpoint_path) -> void;
};

} // namespace models
} // namespace tenzor
