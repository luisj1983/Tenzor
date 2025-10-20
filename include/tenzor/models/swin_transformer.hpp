/**
 * @file swin_transformer.hpp
 * @brief Swin Transformer hierarchical vision transformer
 *
 * Implements the complete Swin Transformer family with shifted window
 * multi-head self-attention for efficient hierarchical feature learning.
 *
 * Reference: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
 *            (Liu et al., ICCV 2021 Best Paper)
 */

#pragma once

#include <memory>
#include <vector>
#include "../nn/module.hpp"
#include "../nn/layers/conv.hpp"
#include "../nn/layers/linear.hpp"
#include "../nn/layers/normalization.hpp"
#include "../nn/layers/dropout.hpp"
#include "../nn/layers/vision.hpp"
#include "../nn/activations/activations.hpp"

namespace tenzor {
namespace models {

/**
 * @brief MLP block with GELU activation for Swin Transformer.
 *
 * Two-layer feed-forward network with expansion ratio of 4.
 *
 * Architecture:
 * ```
 * x -> Linear(in_features, hidden_features) -> GELU -> Dropout ->
 *      Linear(hidden_features, out_features) -> Dropout -> output
 * ```
 */
class SwinMLP : public nn::Module {
public:
    /**
     * @brief Construct MLP block.
     *
     * @param in_features Number of input features
     * @param hidden_features Number of hidden features (default: 0, uses in_features*4)
     * @param out_features Number of output features (default: 0, uses in_features)
     * @param drop Dropout probability (default: 0.0)
     */
    SwinMLP(int64_t in_features,
            int64_t hidden_features = 0,
            int64_t out_features = 0,
            double drop = 0.0);

    auto forward(const Variable& input) -> Variable override;

private:
    std::shared_ptr<nn::Linear> fc1_;
    nn::GELU gelu_;
    std::shared_ptr<nn::Dropout> drop1_;
    std::shared_ptr<nn::Linear> fc2_;
    std::shared_ptr<nn::Dropout> drop2_;
};

/**
 * @brief Swin Transformer block with window or shifted window attention.
 *
 * A single Swin Transformer block consists of:
 * 1. Layer Normalization
 * 2. Window-based Multi-head Self-Attention (W-MSA or SW-MSA)
 * 3. Residual connection
 * 4. Layer Normalization
 * 5. MLP with GELU
 * 6. Residual connection
 *
 * Two consecutive blocks form a pair with W-MSA and SW-MSA.
 *
 * Architecture:
 * ```
 * x -> LN -> W-MSA/SW-MSA -> + -> LN -> MLP -> + -> output
 *  |_______________________↑  |______________↑
 * ```
 */
class SwinTransformerBlock : public nn::Module {
public:
    /**
     * @brief Construct Swin Transformer block.
     *
     * @param dim Number of input channels
     * @param input_resolution Input resolution (H, W)
     * @param num_heads Number of attention heads
     * @param window_size Window size (default: 7)
     * @param shift_size Shift size for SW-MSA (default: 0)
     * @param mlp_ratio MLP hidden dim expansion ratio (default: 4.0)
     * @param qkv_bias If true, add bias to QKV (default: true)
     * @param qk_scale Override QK scale (default: 0.0)
     * @param drop Dropout rate (default: 0.0)
     * @param attn_drop Attention dropout rate (default: 0.0)
     * @param drop_path Stochastic depth rate (default: 0.0)
     */
    SwinTransformerBlock(int64_t dim,
                        const std::pair<int64_t, int64_t>& input_resolution,
                        int64_t num_heads,
                        int64_t window_size = 7,
                        int64_t shift_size = 0,
                        double mlp_ratio = 4.0,
                        bool qkv_bias = true,
                        double qk_scale = 0.0,
                        double drop = 0.0,
                        double attn_drop = 0.0,
                        double drop_path = 0.0);

    auto forward(const Variable& input) -> Variable override;

private:
    int64_t dim_;
    std::pair<int64_t, int64_t> input_resolution_;  ///< (H, W)
    int64_t num_heads_;
    int64_t window_size_;
    int64_t shift_size_;
    double mlp_ratio_;

    // Attention mask for shifted window (cached)
    Tensor attn_mask_;

    // Layers
    std::shared_ptr<nn::LayerNorm> norm1_;
    std::shared_ptr<nn::WindowAttention> attn_;
    std::shared_ptr<nn::LayerNorm> norm2_;
    std::shared_ptr<SwinMLP> mlp_;
    std::shared_ptr<nn::Dropout> drop_path_;

    /**
     * @brief Compute and cache attention mask for shifted windows.
     */
    auto compute_attention_mask() -> void;
};

/**
 * @brief Patch merging layer for downsampling.
 *
 * Reduces resolution by 2x and doubles the number of channels.
 * Merges 2x2 neighboring patches and applies a linear layer.
 *
 * Architecture:
 * ```
 * Input: (B, H, W, C)
 *   ↓ Concatenate 2×2 neighbors
 * (B, H/2, W/2, 4C)
 *   ↓ LayerNorm + Linear
 * Output: (B, H/2, W/2, 2C)
 * ```
 */
class PatchMerging : public nn::Module {
public:
    /**
     * @brief Construct patch merging layer.
     *
     * @param input_resolution Input resolution (H, W)
     * @param dim Number of input channels
     */
    PatchMerging(const std::pair<int64_t, int64_t>& input_resolution,
                int64_t dim);

    auto forward(const Variable& input) -> Variable override;

private:
    std::pair<int64_t, int64_t> input_resolution_;
    int64_t dim_;
    std::shared_ptr<nn::LayerNorm> norm_;
    std::shared_ptr<nn::Linear> reduction_;
};

/**
 * @brief Basic Swin Transformer stage.
 *
 * A stage consists of multiple Swin Transformer blocks followed by
 * an optional patch merging layer for downsampling.
 *
 * Blocks are arranged in W-MSA, SW-MSA pairs:
 * - Block 0: W-MSA (shift_size=0)
 * - Block 1: SW-MSA (shift_size=window_size/2)
 * - Block 2: W-MSA (shift_size=0)
 * - ...
 */
class BasicLayer : public nn::Module {
public:
    /**
     * @brief Construct basic Swin Transformer stage.
     *
     * @param dim Number of input channels
     * @param input_resolution Input resolution (H, W)
     * @param depth Number of blocks in this stage
     * @param num_heads Number of attention heads
     * @param window_size Window size (default: 7)
     * @param mlp_ratio MLP hidden dim expansion ratio (default: 4.0)
     * @param qkv_bias If true, add bias to QKV (default: true)
     * @param qk_scale Override QK scale (default: 0.0)
     * @param drop Dropout rate (default: 0.0)
     * @param attn_drop Attention dropout rate (default: 0.0)
     * @param drop_path Stochastic depth rates (default: empty)
     * @param downsample Downsample layer at end of stage (default: nullptr)
     */
    BasicLayer(int64_t dim,
              const std::pair<int64_t, int64_t>& input_resolution,
              int64_t depth,
              int64_t num_heads,
              int64_t window_size = 7,
              double mlp_ratio = 4.0,
              bool qkv_bias = true,
              double qk_scale = 0.0,
              double drop = 0.0,
              double attn_drop = 0.0,
              const std::vector<double>& drop_path = {},
              std::shared_ptr<nn::Module> downsample = nullptr);

    auto forward(const Variable& input) -> Variable override;

    /**
     * @brief Get output resolution after this stage.
     */
    auto output_resolution() const -> std::pair<int64_t, int64_t> {
        return output_resolution_;
    }

private:
    int64_t dim_;
    std::pair<int64_t, int64_t> input_resolution_;
    std::pair<int64_t, int64_t> output_resolution_;
    int64_t depth_;

    std::vector<std::shared_ptr<SwinTransformerBlock>> blocks_;
    std::shared_ptr<nn::Module> downsample_;
};

/**
 * @brief Patch embedding layer for Swin Transformer.
 *
 * Splits image into non-overlapping 4x4 patches and embeds them.
 *
 * Architecture:
 * ```
 * Image (B, 3, 224, 224)
 *   ↓ Conv 4×4, stride=4
 * Patches (B, 96, 56, 56)
 *   ↓ Permute to (B, 56, 56, 96)
 *   ↓ LayerNorm
 * Output: (B, H/4, W/4, embed_dim)
 * ```
 */
class PatchEmbed : public nn::Module {
public:
    /**
     * @brief Construct patch embedding layer.
     *
     * @param img_size Input image size (default: 224)
     * @param patch_size Patch size (default: 4)
     * @param in_chans Number of input channels (default: 3)
     * @param embed_dim Embedding dimension (default: 96)
     * @param norm_layer If true, apply LayerNorm (default: false)
     */
    PatchEmbed(int64_t img_size = 224,
              int64_t patch_size = 4,
              int64_t in_chans = 3,
              int64_t embed_dim = 96,
              bool norm_layer = false);

    auto forward(const Variable& input) -> Variable override;

    /**
     * @brief Get patch grid size.
     */
    auto patches_resolution() const -> std::pair<int64_t, int64_t> {
        return patches_resolution_;
    }

    /**
     * @brief Get number of patches.
     */
    auto num_patches() const -> int64_t { return num_patches_; }

private:
    std::pair<int64_t, int64_t> img_size_;
    std::pair<int64_t, int64_t> patches_resolution_;
    int64_t num_patches_;
    int64_t embed_dim_;

    std::shared_ptr<nn::Conv2d> proj_;
    std::shared_ptr<nn::LayerNorm> norm_;
};

/**
 * @brief Swin Transformer for image classification.
 *
 * Hierarchical vision transformer using shifted window attention
 * for linear complexity with respect to image size.
 *
 * Architecture:
 * ```
 * Input (224×224×3)
 *   ↓ Patch Embed (4×4)
 * Stage 1: 56×56×C,    depth[0] blocks, heads[0]
 *   ↓ Patch Merge
 * Stage 2: 28×28×2C,   depth[1] blocks, heads[1]
 *   ↓ Patch Merge
 * Stage 3: 14×14×4C,   depth[2] blocks, heads[2]
 *   ↓ Patch Merge
 * Stage 4: 7×7×8C,     depth[3] blocks, heads[3]
 *   ↓ LayerNorm + Global Average Pool
 * Output: num_classes
 * ```
 *
 * Variants:
 * - Swin-Tiny:  C=96,  depths=[2,2,6,2],   heads=[3,6,12,24],  29M params
 * - Swin-Small: C=96,  depths=[2,2,18,2],  heads=[3,6,12,24],  50M params
 * - Swin-Base:  C=128, depths=[2,2,18,2],  heads=[4,8,16,32],  88M params
 * - Swin-Large: C=192, depths=[2,2,18,2],  heads=[6,12,24,48], 197M params
 *
 * Computational Complexity:
 * - Swin-T: 4.5 GFLOPs @ 224×224
 * - Swin-S: 8.7 GFLOPs @ 224×224
 * - Swin-B: 15.4 GFLOPs @ 224×224
 * - Swin-L: 34.5 GFLOPs @ 224×224
 *
 * @code
 * // Create Swin-Tiny model
 * auto model = swin_tiny(1000);
 * Variable image({1, 3, 224, 224});
 * Variable output = model->forward(image);  // Shape: {1, 1000}
 *
 * // Create Swin-Base model
 * auto model_base = swin_base(1000);
 * @endcode
 */
class SwinTransformer : public nn::Module {
public:
    /**
     * @brief Construct Swin Transformer.
     *
     * @param img_size Input image size (default: 224)
     * @param patch_size Patch size (default: 4)
     * @param in_chans Number of input channels (default: 3)
     * @param num_classes Number of classification classes (default: 1000)
     * @param embed_dim Embedding dimension (default: 96)
     * @param depths Number of blocks in each stage
     * @param num_heads Number of attention heads in each stage
     * @param window_size Window size (default: 7)
     * @param mlp_ratio MLP hidden dim expansion ratio (default: 4.0)
     * @param qkv_bias If true, add bias to QKV (default: true)
     * @param qk_scale Override QK scale (default: 0.0)
     * @param drop_rate Dropout rate (default: 0.0)
     * @param attn_drop_rate Attention dropout rate (default: 0.0)
     * @param drop_path_rate Stochastic depth rate (default: 0.1)
     * @param norm_layer If true, use LayerNorm in patch embed (default: true)
     */
    SwinTransformer(int64_t img_size = 224,
                   int64_t patch_size = 4,
                   int64_t in_chans = 3,
                   int64_t num_classes = 1000,
                   int64_t embed_dim = 96,
                   const std::vector<int64_t>& depths = {2, 2, 6, 2},
                   const std::vector<int64_t>& num_heads = {3, 6, 12, 24},
                   int64_t window_size = 7,
                   double mlp_ratio = 4.0,
                   bool qkv_bias = true,
                   double qk_scale = 0.0,
                   double drop_rate = 0.0,
                   double attn_drop_rate = 0.0,
                   double drop_path_rate = 0.1,
                   bool norm_layer = true);

    /**
     * @brief Forward pass through Swin Transformer.
     *
     * @param input Input image tensor of shape (N, 3, H, W)
     * @return Output logits of shape (N, num_classes)
     */
    auto forward(const Variable& input) -> Variable override;

    /**
     * @brief Extract hierarchical features from all stages.
     *
     * Useful for dense prediction tasks (detection, segmentation).
     *
     * @param input Input image tensor of shape (N, 3, H, W)
     * @return Vector of features from each stage:
     *         - Stage 1: (N, H/4, W/4, C)
     *         - Stage 2: (N, H/8, W/8, 2C)
     *         - Stage 3: (N, H/16, W/16, 4C)
     *         - Stage 4: (N, H/32, W/32, 8C)
     */
    auto forward_features(const Variable& input) -> std::vector<Variable>;

    /**
     * @brief Load pretrained weights.
     *
     * @param path Path to pretrained weights file
     * @throws std::runtime_error if file doesn't exist or format is invalid
     */
    auto load_pretrained(const std::string& path) -> void;

private:
    int64_t num_classes_;
    int64_t num_layers_;
    int64_t embed_dim_;
    int64_t num_features_;  ///< Final feature dimension

    // Network components
    std::shared_ptr<PatchEmbed> patch_embed_;
    std::shared_ptr<nn::Dropout> pos_drop_;
    std::vector<std::shared_ptr<BasicLayer>> layers_;
    std::shared_ptr<nn::LayerNorm> norm_;
    std::shared_ptr<nn::AdaptiveAvgPool2d> avgpool_;
    std::shared_ptr<nn::Linear> head_;
};

// ============================================================================
// Factory Functions for Standard Swin Transformer Variants
// ============================================================================

/**
 * @brief Create Swin-Tiny model.
 *
 * Architecture: embed_dim=96, depths=[2,2,6,2], heads=[3,6,12,24]
 * Parameters: ~29M
 * FLOPs: 4.5G @ 224×224
 * Top-1 Accuracy (ImageNet-1K): ~81.3%
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param img_size Input image size (default: 224)
 * @param pretrained Load pretrained ImageNet weights (default: false)
 * @return Shared pointer to Swin-Tiny model
 *
 * @code
 * auto model = swin_tiny(1000);
 * Variable output = model->forward(input);
 * @endcode
 */
auto swin_tiny(int64_t num_classes = 1000,
              int64_t img_size = 224,
              bool pretrained = false) -> std::shared_ptr<SwinTransformer>;

/**
 * @brief Create Swin-Small model.
 *
 * Architecture: embed_dim=96, depths=[2,2,18,2], heads=[3,6,12,24]
 * Parameters: ~50M
 * FLOPs: 8.7G @ 224×224
 * Top-1 Accuracy (ImageNet-1K): ~83.0%
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param img_size Input image size (default: 224)
 * @param pretrained Load pretrained ImageNet weights (default: false)
 * @return Shared pointer to Swin-Small model
 */
auto swin_small(int64_t num_classes = 1000,
               int64_t img_size = 224,
               bool pretrained = false) -> std::shared_ptr<SwinTransformer>;

/**
 * @brief Create Swin-Base model.
 *
 * Architecture: embed_dim=128, depths=[2,2,18,2], heads=[4,8,16,32]
 * Parameters: ~88M
 * FLOPs: 15.4G @ 224×224
 * Top-1 Accuracy (ImageNet-1K): ~83.5%
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param img_size Input image size (default: 224)
 * @param pretrained Load pretrained ImageNet weights (default: false)
 * @return Shared pointer to Swin-Base model
 */
auto swin_base(int64_t num_classes = 1000,
              int64_t img_size = 224,
              bool pretrained = false) -> std::shared_ptr<SwinTransformer>;

/**
 * @brief Create Swin-Large model.
 *
 * Architecture: embed_dim=192, depths=[2,2,18,2], heads=[6,12,24,48]
 * Parameters: ~197M
 * FLOPs: 34.5G @ 224×224
 * Top-1 Accuracy (ImageNet-1K): ~84.5%
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param img_size Input image size (default: 224)
 * @param pretrained Load pretrained ImageNet weights (default: false)
 * @return Shared pointer to Swin-Large model
 */
auto swin_large(int64_t num_classes = 1000,
               int64_t img_size = 224,
               bool pretrained = false) -> std::shared_ptr<SwinTransformer>;

} // namespace models
} // namespace tenzor
