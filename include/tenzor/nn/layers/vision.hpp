/**
 * @file vision.hpp
 * @brief Vision-specific layers for computer vision models
 *
 * Implements specialized layers for modern computer vision architectures,
 * including window-based attention for hierarchical vision transformers.
 */

#pragma once

#include <memory>
#include <optional>
#include "../module.hpp"
#include "linear.hpp"
#include "dropout.hpp"
#include "normalization.hpp"

namespace tenzor {
namespace nn {

/**
 * @brief Patch embedding layer for Vision Transformers.
 *
 * Splits input image into patches and projects them to embedding dimension.
 * This is the first layer in Vision Transformer (ViT) and Swin Transformer
 * architectures.
 *
 * Architecture:
 *   Input: (N, C, H, W)
 *   -> Extract patches of size (patch_size, patch_size)
 *   -> Flatten patches: (N, num_patches, C * patch_size^2)
 *   -> Linear projection: (N, num_patches, embed_dim)
 *
 * Where num_patches = (H / patch_size) * (W / patch_size)
 *
 * Implementation:
 *   Uses Conv2d with kernel_size=patch_size and stride=patch_size
 *   for efficient patch extraction. This is mathematically equivalent
 *   to unfold + linear but much faster.
 *
 * @code
 * // ViT-B/16 configuration: 3-channel RGB -> 768-dim embeddings, 16x16 patches
 * PatchEmbedding patch_embed(3, 768, 16);
 *
 * Variable img({batch, 3, 224, 224}, DType::Float32, Device::cpu(), true);
 * Variable patches = patch_embed.forward(img);  // {batch, 196, 768}
 * // where 196 = (224/16) * (224/16)
 * @endcode
 *
 * @see Conv2d for the underlying convolution
 */
class PatchEmbedding : public Module {
public:
    /**
     * @brief Construct patch embedding layer.
     *
     * @param in_channels Number of input channels (e.g., 3 for RGB)
     * @param embed_dim Embedding dimension (hidden size)
     * @param patch_size Size of each square patch
     * @param img_size Expected input image size (for validation, default: 224)
     *
     * @code
     * PatchEmbedding pe1(3, 768, 16);      // ViT-B/16
     * PatchEmbedding pe2(3, 768, 32);      // ViT-B/32
     * PatchEmbedding pe3(3, 1024, 16);     // ViT-L/16
     * @endcode
     */
    PatchEmbedding(int64_t in_channels,
                   int64_t embed_dim,
                   int64_t patch_size,
                   int64_t img_size = 224);

    /**
     * @brief Forward pass to convert image to patch embeddings.
     *
     * @param input Input image tensor of shape (N, C, H, W)
     * @return Patch embeddings of shape (N, num_patches, embed_dim)
     *
     * @throws std::runtime_error if input dimensions not divisible by patch_size
     */
    auto forward(const Variable& input) -> Variable override;

    /**
     * @brief Get number of patches produced.
     *
     * @return Number of patches: (img_size/patch_size)^2
     */
    auto num_patches() const -> int64_t { return num_patches_; }

    /**
     * @brief Get patch size.
     */
    auto patch_size() const -> int64_t { return patch_size_; }

    /**
     * @brief Get embedding dimension.
     */
    auto embed_dim() const -> int64_t { return embed_dim_; }

private:
    int64_t in_channels_;   ///< Number of input channels
    int64_t embed_dim_;     ///< Embedding dimension
    int64_t patch_size_;    ///< Patch size
    int64_t img_size_;      ///< Expected image size
    int64_t num_patches_;   ///< Number of patches

    std::shared_ptr<Module> proj_;  ///< Conv2d projection layer
};

/**
 * @brief Window-based Multi-Head Self-Attention for Swin Transformer.
 *
 * Implements efficient window-based attention mechanism that computes
 * self-attention within local windows, achieving linear complexity
 * with respect to image size.
 *
 * Key features:
 * - Local window attention (7x7 default)
 * - Relative position bias for improved localization
 * - Support for shifted windows (SW-MSA)
 * - Linear complexity O(M²·H·W) where M is window size
 *
 * Shape transformations:
 * - Input: (num_windows*B, window_size*window_size, C)
 * - Output: (num_windows*B, window_size*window_size, C)
 *
 * Where:
 * - B = batch size
 * - C = number of channels
 * - num_windows = (H/M) * (W/M)
 * - M = window_size
 *
 * Computational Complexity:
 * - Time: O(M²·N·C) where N = H·W (linear in image size!)
 * - Memory: O(M²·num_heads) for attention weights per window
 *
 * Reference: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
 *            (Liu et al., ICCV 2021)
 *
 * @code
 * // Create window attention with 7x7 windows, 96 channels, 3 heads
 * WindowAttention win_attn(96, 7, 3);
 *
 * // Apply to windowed input
 * Variable windowed_input({num_windows * batch, 49, 96});
 * Variable output = win_attn.forward(windowed_input);
 *
 * // With attention mask for shifted windows
 * Tensor attn_mask({num_windows, 49, 49});
 * Variable masked_output = win_attn.forward(windowed_input, attn_mask);
 * @endcode
 */
class WindowAttention : public Module {
public:
    /**
     * @brief Construct window attention layer.
     *
     * @param dim Number of input channels
     * @param window_size Window size (M x M). Default: 7
     * @param num_heads Number of attention heads
     * @param qkv_bias If true, add bias to QKV projections (default: true)
     * @param qk_scale Override default QK scale of head_dim**-0.5 (default: 0.0)
     * @param attn_drop Attention dropout probability (default: 0.0)
     * @param proj_drop Output projection dropout probability (default: 0.0)
     *
     * @throws std::invalid_argument if dim not divisible by num_heads
     *
     * @code
     * WindowAttention attn1(96, 7, 3);                      // Standard Swin-T
     * WindowAttention attn2(192, 7, 6, true, 0.0, 0.1);    // With dropout
     * @endcode
     */
    WindowAttention(int64_t dim,
                   int64_t window_size = 7,
                   int64_t num_heads = 3,
                   bool qkv_bias = true,
                   double qk_scale = 0.0,
                   double attn_drop = 0.0,
                   double proj_drop = 0.0);

    /**
     * @brief Forward pass through window attention.
     *
     * @param input Input tensor of shape (num_windows*B, N, C)
     *              where N = window_size * window_size
     * @param mask Optional attention mask of shape (num_windows, N, N)
     *             Positions with large negative values (-100) are masked
     *
     * @return Output tensor of shape (num_windows*B, N, C)
     *
     * @code
     * // Without mask (W-MSA)
     * Variable output = attn.forward(input);
     *
     * // With mask (SW-MSA for shifted windows)
     * Tensor shift_mask = create_shifted_window_mask(H, W, window_size, shift_size);
     * Variable output = attn.forward(input, shift_mask);
     * @endcode
     */
    auto forward(const Variable& input, const Tensor& mask = Tensor{}) -> Variable;

    /**
     * @brief Default forward for Module interface.
     */
    auto forward(const Variable& input) -> Variable override {
        return forward(input, Tensor{});
    }

    /**
     * @brief Get dimension.
     */
    auto dim() const -> int64_t { return dim_; }

    /**
     * @brief Get window size.
     */
    auto window_size() const -> int64_t { return window_size_; }

    /**
     * @brief Get number of attention heads.
     */
    auto num_heads() const -> int64_t { return num_heads_; }

private:
    int64_t dim_;              ///< Input/output channels
    int64_t window_size_;      ///< Window size (M)
    int64_t num_heads_;        ///< Number of attention heads
    int64_t head_dim_;         ///< Channels per head
    double scale_;             ///< Attention scale factor

    // Learnable relative position bias table
    // Shape: (2*window_size-1, 2*window_size-1, num_heads)
    std::shared_ptr<Variable> relative_position_bias_table_;

    // Pre-computed relative position index
    // Shape: (window_size*window_size, window_size*window_size)
    Tensor relative_position_index_;

    // Projection layers
    std::shared_ptr<Linear> qkv_;         ///< Combined QKV projection
    std::shared_ptr<Dropout> attn_drop_;  ///< Attention dropout
    std::shared_ptr<Linear> proj_;        ///< Output projection
    std::shared_ptr<Dropout> proj_drop_;  ///< Projection dropout

    /**
     * @brief Compute and cache relative position index.
     *
     * The relative position index maps each (query, key) pair to an
     * index in the relative position bias table.
     */
    auto compute_relative_position_index() -> void;

    /**
     * @brief Get relative position bias for current window.
     *
     * @return Bias tensor of shape (num_heads, N, N) where N = window_size²
     */
    auto get_relative_position_bias() const -> Tensor;
};

/**
 * @brief Helper function to partition tensor into windows.
 *
 * Splits the input feature map into non-overlapping M×M windows.
 *
 * @param input Input tensor of shape (B, H, W, C)
 * @param window_size Window size M
 * @return Windowed tensor of shape (B*num_windows, M*M, C)
 *         where num_windows = (H/M) * (W/M)
 *
 * @throws std::invalid_argument if H or W not divisible by window_size
 *
 * @code
 * Variable features({batch, 56, 56, 96});
 * Variable windows = window_partition(features, 7);
 * // Result: shape (batch*64, 49, 96) since 56/7=8, 8*8=64 windows
 * @endcode
 */
auto window_partition(const Variable& input, int64_t window_size) -> Variable;

/**
 * @brief Helper function to reverse window partition.
 *
 * Merges windows back into a feature map.
 *
 * @param windows Windowed tensor of shape (B*num_windows, M*M, C)
 * @param window_size Window size M
 * @param H Height of the feature map
 * @param W Width of the feature map
 * @return Feature map of shape (B, H, W, C)
 *
 * @code
 * Variable windows({batch*64, 49, 96});
 * Variable features = window_reverse(windows, 7, 56, 56);
 * // Result: shape (batch, 56, 56, 96)
 * @endcode
 */
auto window_reverse(const Variable& windows, int64_t window_size,
                    int64_t H, int64_t W) -> Variable;

/**
 * @brief Create attention mask for shifted window attention.
 *
 * Creates a mask that prevents attention between non-adjacent regions
 * after applying cyclic shift. This is used for SW-MSA in Swin Transformer.
 *
 * @param H Height of feature map
 * @param W Width of feature map
 * @param window_size Window size M
 * @param shift_size Shift size (typically M/2)
 * @param device Device to create mask on
 * @return Attention mask of shape (num_windows, M*M, M*M)
 *         Masked positions have value -100.0
 *
 * @code
 * Tensor mask = create_shifted_window_mask(56, 56, 7, 3);
 * // Result: shape (64, 49, 49) with -100.0 for masked positions
 * @endcode
 */
auto create_shifted_window_mask(int64_t H, int64_t W,
                                 int64_t window_size,
                                 int64_t shift_size,
                                 Device device = Device::cpu(),
                                 DType dtype = DType::Float32) -> Tensor;

} // namespace nn
} // namespace tenzor
