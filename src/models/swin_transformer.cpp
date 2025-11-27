/**
 * @file swin_transformer.cpp
 * @brief Implementation of Swin Transformer model
 */

#include "tenzor/models/swin_transformer.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/checkpoint.hpp"
#include "tenzor/nn/checkpoint.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include <stdexcept>
#include <cmath>

namespace tenzor::models {

using namespace tenzor::nn;

// ============================================================================
// SwinMLP Implementation
// ============================================================================

SwinMLP::SwinMLP(int64_t in_features,
                 int64_t hidden_features,
                 int64_t out_features,
                 double drop)
{
    hidden_features = (hidden_features > 0) ? hidden_features : in_features * 4;
    out_features = (out_features > 0) ? out_features : in_features;

    fc1_ = std::make_shared<Linear>(in_features, hidden_features);
    register_module("fc1", fc1_);

    drop1_ = std::make_shared<Dropout>(drop);
    register_module("drop1", drop1_);

    fc2_ = std::make_shared<Linear>(hidden_features, out_features);
    register_module("fc2", fc2_);

    drop2_ = std::make_shared<Dropout>(drop);
    register_module("drop2", drop2_);
}

auto SwinMLP::forward_impl(const Variable& input) -> Variable {
    auto x = fc1_->forward(input);
    x = gelu_.forward(x);
    x = drop1_->forward(x);
    x = fc2_->forward(x);
    x = drop2_->forward(x);
    return x;
}

// ============================================================================
// SwinTransformerBlock Implementation
// ============================================================================

SwinTransformerBlock::SwinTransformerBlock(
    int64_t dim,
    const std::pair<int64_t, int64_t>& input_resolution,
    int64_t num_heads,
    int64_t window_size,
    int64_t shift_size,
    double mlp_ratio,
    bool qkv_bias,
    double qk_scale,
    double drop,
    double attn_drop,
    double drop_path)
    : dim_(dim)
    , input_resolution_(input_resolution)
    , num_heads_(num_heads)
    , window_size_(window_size)
    , shift_size_(shift_size)
    , mlp_ratio_(mlp_ratio)
{
    if (input_resolution.first <= window_size || input_resolution.second <= window_size) {
        // No window partitioning needed if input smaller than window
        shift_size_ = 0;
        window_size_ = std::min(input_resolution.first, input_resolution.second);
    }

    norm1_ = std::make_shared<LayerNorm>(std::vector<int64_t>{dim});
    register_module("norm1", norm1_);

    attn_ = std::make_shared<WindowAttention>(
        dim, window_size_, num_heads, qkv_bias, qk_scale, attn_drop, drop);
    register_module("attn", attn_);

    drop_path_ = std::make_shared<Dropout>(drop_path);
    register_module("drop_path", drop_path_);

    norm2_ = std::make_shared<LayerNorm>(std::vector<int64_t>{dim});
    register_module("norm2", norm2_);

    int64_t mlp_hidden_dim = static_cast<int64_t>(dim * mlp_ratio);
    mlp_ = std::make_shared<SwinMLP>(dim, mlp_hidden_dim, dim, drop);
    register_module("mlp", mlp_);

    // Compute attention mask for shifted windows
    if (shift_size_ > 0) {
        compute_attention_mask();
    }
}

auto SwinTransformerBlock::compute_attention_mask() -> void {
    int64_t H = input_resolution_.first;
    int64_t W = input_resolution_.second;

    // Create attention mask for SW-MSA
    attn_mask_ = create_shifted_window_mask(H, W, window_size_, shift_size_);
}

auto SwinTransformerBlock::forward_impl(const Variable& input) -> Variable {
    auto shape = input.tensor().shape();
    if (shape.size() != 3) {
        throw std::runtime_error(
            "SwinTransformerBlock expects 3D input (B, L, C), got " +
            std::to_string(shape.size()) + "D");
    }
    int64_t B = shape[0];
    int64_t L = shape[1];  // H * W
    int64_t C = shape[2];

    int64_t H = input_resolution_.first;
    int64_t W = input_resolution_.second;

    if (L != H * W) {
        throw std::runtime_error(
            "Input feature length (" + std::to_string(L) +
            ") doesn't match resolution (" + std::to_string(H) + "x" + std::to_string(W) + ")");
    }

    auto shortcut = input;

    // Layer norm
    auto x = norm1_->forward(input);

    // Reshape to (B, H, W, C)
    x = x.reshape({B, H, W, C});

    // Cyclic shift for SW-MSA
    if (shift_size_ > 0) {
        // Roll operation: shift by (-shift_size, -shift_size)
        // Use autograd-aware roll to preserve gradient flow
        x = tenzor::roll(x, -shift_size_, 1);  // shift height
        x = tenzor::roll(x, -shift_size_, 2);  // shift width
    }

    // Partition windows
    auto x_windows = window_partition(x, window_size_);

    // Window attention - compute mask dynamically with correct device
    // Keep mask as Float32 for numerical stability
    Tensor mask;
    if (shift_size_ > 0) {
        auto input_device = input.tensor().device();
        mask = create_shifted_window_mask(H, W, window_size_, shift_size_, input_device, DType::Float32);
    }
    auto attn_windows = attn_->forward(x_windows, mask);

    // Merge windows
    x = window_reverse(attn_windows, window_size_, H, W);

    // Reverse cyclic shift
    if (shift_size_ > 0) {
        // Use autograd-aware roll to preserve gradient flow
        x = tenzor::roll(x, shift_size_, 1);  // reverse shift height
        x = tenzor::roll(x, shift_size_, 2);  // reverse shift width
    }

    // Reshape to (B, H*W, C)
    x = x.reshape({B, H * W, C});

    // First residual connection
    x = shortcut + drop_path_->forward(x);

    // MLP block with second residual connection
    x = x + drop_path_->forward(mlp_->forward(norm2_->forward(x)));

    return x;
}

// ============================================================================
// PatchMerging Implementation
// ============================================================================

PatchMerging::PatchMerging(const std::pair<int64_t, int64_t>& input_resolution,
                           int64_t dim)
    : input_resolution_(input_resolution)
    , dim_(dim)
{
    norm_ = std::make_shared<LayerNorm>(std::vector<int64_t>{4 * dim});
    register_module("norm", norm_);

    reduction_ = std::make_shared<Linear>(4 * dim, 2 * dim, false);
    register_module("reduction", reduction_);
}

auto PatchMerging::forward_impl(const Variable& input) -> Variable {
    auto shape = input.tensor().shape();
    if (shape.size() != 3) {
        throw std::runtime_error(
            "PatchMerging expects 3D input (B, L, C), got " +
            std::to_string(shape.size()) + "D");
    }
    int64_t B = shape[0];
    int64_t L = shape[1];
    int64_t C = shape[2];

    int64_t H = input_resolution_.first;
    int64_t W = input_resolution_.second;

    if (L != H * W) {
        throw std::runtime_error("Input feature length doesn't match resolution");
    }

    // Reshape to (B, H, W, C)
    auto x = input.reshape({B, H, W, C});

    // Concatenate 2x2 neighbors using gradient-aware operations
    // Extract 4 patches: [0::2, 0::2], [0::2, 1::2], [1::2, 0::2], [1::2, 1::2]
    // Use autograd::slice to maintain gradient chain
    auto x0 = slice(slice(x, 1, 0, H, 2), 2, 0, W, 2);  // Top-left
    auto x1 = slice(slice(x, 1, 1, H, 2), 2, 0, W, 2);  // Bottom-left
    auto x2 = slice(slice(x, 1, 0, H, 2), 2, 1, W, 2);  // Top-right
    auto x3 = slice(slice(x, 1, 1, H, 2), 2, 1, W, 2);  // Bottom-right

    // Concatenate along channel dimension using gradient-aware cat
    x = cat({x0, x1, x2, x3}, -1);

    // Reshape to (B, H/2 * W/2, 4*C)
    x = x.reshape({B, H / 2 * W / 2, 4 * C});

    // Layer norm
    x = norm_->forward(x);

    // Linear reduction: 4C -> 2C
    x = reduction_->forward(x);

    return x;
}

// ============================================================================
// BasicLayer Implementation
// ============================================================================

BasicLayer::BasicLayer(int64_t dim,
                      const std::pair<int64_t, int64_t>& input_resolution,
                      int64_t depth,
                      int64_t num_heads,
                      int64_t window_size,
                      double mlp_ratio,
                      bool qkv_bias,
                      double qk_scale,
                      double drop,
                      double attn_drop,
                      const std::vector<double>& drop_path,
                      std::shared_ptr<Module> downsample,
                      bool use_checkpoint)
    : dim_(dim)
    , input_resolution_(input_resolution)
    , depth_(depth)
    , use_checkpoint_(use_checkpoint)
    , downsample_(downsample)
{
    // Build Swin Transformer blocks
    for (int64_t i = 0; i < depth; ++i) {
        // Alternate between W-MSA (shift=0) and SW-MSA (shift=window_size/2)
        int64_t shift = (i % 2 == 0) ? 0 : window_size / 2;

        double dp_rate = drop_path.empty() ? 0.0 : drop_path[i];

        auto block = std::make_shared<SwinTransformerBlock>(
            dim,
            input_resolution,
            num_heads,
            window_size,
            shift,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop,
            attn_drop,
            dp_rate);

        blocks_.push_back(block);
        register_module("block" + std::to_string(i), block);
    }

    // Register downsample if provided
    if (downsample_) {
        register_module("downsample", downsample_);
        output_resolution_ = {input_resolution.first / 2, input_resolution.second / 2};
    } else {
        output_resolution_ = input_resolution;
    }
}

auto BasicLayer::forward_impl(const Variable& input) -> Variable {
    auto x = input;

    // Apply all blocks with optional gradient checkpointing
    for (auto& block : blocks_) {
        if (use_checkpoint_ && training_) {
            // Use gradient checkpointing: don't save activations during forward,
            // recompute them during backward. Saves ~50-80% memory.
            x = autograd::checkpoint(
                [&block](const Variable& in) -> Variable {
                    return block->forward(in);
                },
                x
            );
        } else {
            x = block->forward(x);
        }
    }

    // Apply downsampling if exists
    if (downsample_) {
        x = downsample_->forward(x);
    }

    return x;
}

// ============================================================================
// PatchEmbed Implementation
// ============================================================================

PatchEmbed::PatchEmbed(int64_t img_size,
                      int64_t patch_size,
                      int64_t in_chans,
                      int64_t embed_dim,
                      bool norm_layer)
    : img_size_({img_size, img_size})
    , embed_dim_(embed_dim)
{
    patches_resolution_ = {img_size / patch_size, img_size / patch_size};
    num_patches_ = patches_resolution_.first * patches_resolution_.second;

    // Use Conv2d for patch embedding
    proj_ = std::make_shared<Conv2d>(
        in_chans,
        embed_dim,
        patch_size,  // kernel_size
        patch_size,  // stride
        0,           // padding
        1,           // dilation
        1,           // groups
        true         // bias
    );
    register_module("proj", proj_);

    // Optional layer norm
    if (norm_layer) {
        norm_ = std::make_shared<LayerNorm>(std::vector<int64_t>{embed_dim});
        register_module("norm", norm_);
    }
}

auto PatchEmbed::forward_impl(const Variable& input) -> Variable {
    auto shape = input.tensor().shape();
    if (shape.size() != 4) {
        throw std::runtime_error(
            "PatchEmbed expects 4D input (B, C, H, W), got " +
            std::to_string(shape.size()) + "D");
    }
    int64_t B = shape[0];

    // Apply convolution: (B, C, H, W) -> (B, embed_dim, H/4, W/4)
    auto x = proj_->forward(input);

    // Flatten: (B, embed_dim, H', W') -> (B, embed_dim, H'*W')
    auto x_shape = x.tensor().shape();
    if (x_shape.size() < 4) {
        throw std::runtime_error(
            "Conv2d output has unexpected shape size: " + std::to_string(x_shape.size()) +
            " (expected 4D tensor)");
    }
    int64_t H_out = x_shape[2];
    int64_t W_out = x_shape[3];
    int64_t actual_num_patches = H_out * W_out;
    x = x.reshape({B, embed_dim_, actual_num_patches});

    // Transpose: (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
    x = x.transpose(1, 2);

    // Apply layer norm if exists
    if (norm_) {
        x = norm_->forward(x);
    }

    return x;
}

// ============================================================================
// SwinTransformer Implementation
// ============================================================================

SwinTransformer::SwinTransformer(int64_t img_size,
                                int64_t patch_size,
                                int64_t in_chans,
                                int64_t num_classes,
                                int64_t embed_dim,
                                const std::vector<int64_t>& depths,
                                const std::vector<int64_t>& num_heads,
                                int64_t window_size,
                                double mlp_ratio,
                                bool qkv_bias,
                                double qk_scale,
                                double drop_rate,
                                double attn_drop_rate,
                                double drop_path_rate,
                                bool norm_layer,
                                bool use_checkpoint)
    : num_classes_(num_classes)
    , num_layers_(depths.size())
    , embed_dim_(embed_dim)
    , num_features_(embed_dim * (1 << (num_layers_ - 1)))  // embed_dim * 2^(num_layers-1)
    , use_checkpoint_(use_checkpoint)
{
    // Patch embedding
    patch_embed_ = std::make_shared<PatchEmbed>(
        img_size, patch_size, in_chans, embed_dim, norm_layer);
    register_module("patch_embed", patch_embed_);

    auto patches_resolution = patch_embed_->patches_resolution();

    // Position embedding dropout
    pos_drop_ = std::make_shared<Dropout>(drop_rate);
    register_module("pos_drop", pos_drop_);

    // Stochastic depth decay rule
    std::vector<double> dpr;
    double total_depth = 0;
    for (auto d : depths) total_depth += d;

    double dpr_increment = drop_path_rate / (total_depth - 1);
    double current_dpr = 0.0;
    for (int64_t i = 0; i < total_depth; ++i) {
        dpr.push_back(current_dpr);
        current_dpr += dpr_increment;
    }

    // Build layers
    int64_t dpr_offset = 0;
    for (int64_t i_layer = 0; i_layer < num_layers_; ++i_layer) {
        int64_t layer_dim = embed_dim * (1 << i_layer);  // embed_dim * 2^i
        std::pair<int64_t, int64_t> layer_resolution = {
            patches_resolution.first >> i_layer,
            patches_resolution.second >> i_layer
        };

        // Create downsample layer for all stages except the last
        std::shared_ptr<Module> downsample = nullptr;
        if (i_layer < num_layers_ - 1) {
            downsample = std::make_shared<PatchMerging>(layer_resolution, layer_dim);
        }

        // Extract drop_path rates for this layer
        std::vector<double> layer_dpr(
            dpr.begin() + dpr_offset,
            dpr.begin() + dpr_offset + depths[i_layer]
        );

        auto layer = std::make_shared<BasicLayer>(
            layer_dim,
            layer_resolution,
            depths[i_layer],
            num_heads[i_layer],
            window_size,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop_rate,
            attn_drop_rate,
            layer_dpr,
            downsample,
            use_checkpoint_  // Pass checkpoint flag to each stage
        );

        layers_.push_back(layer);
        register_module("layer" + std::to_string(i_layer), layer);

        dpr_offset += depths[i_layer];
    }

    // Final layer norm
    norm_ = std::make_shared<LayerNorm>(std::vector<int64_t>{num_features_});
    register_module("norm", norm_);

    // Global average pooling
    avgpool_ = std::make_shared<AdaptiveAvgPool2d>(1);
    register_module("avgpool", avgpool_);

    // Classification head
    head_ = std::make_shared<Linear>(num_features_, num_classes);
    register_module("head", head_);
}

auto SwinTransformer::forward_impl(const Variable& input) -> Variable {
    // Patch embedding
    auto x = patch_embed_->forward(input);

    // Position dropout
    x = pos_drop_->forward(x);

    // Apply all stages
    for (auto& layer : layers_) {
        x = layer->forward(x);
    }

    // Final layer norm
    x = norm_->forward(x);

    // Global average pooling
    // x shape: (B, num_patches, C)
    // Need to transpose to (B, C, num_patches) for pooling
    auto shape = x.tensor().shape();
    if (shape.size() != 3) {
        throw std::runtime_error(
            "Expected 3D tensor after normalization (B, num_patches, C), got " +
            std::to_string(shape.size()) + "D");
    }
    int64_t B = shape[0];
    int64_t C = shape[2];

    // Get final resolution
    auto final_resolution = layers_.back()->output_resolution();
    int64_t H = final_resolution.first;
    int64_t W = final_resolution.second;

    // Reshape: (B, H*W, C) -> (B, H, W, C) -> (B, C, H, W)
    x = x.reshape({B, H, W, C});
    x = x.permute({0, 3, 1, 2});  // (B, C, H, W)

    // Global pooling
    x = avgpool_->forward(x);

    // Flatten: (B, C, 1, 1) -> (B, C)
    x = x.reshape({B, C});

    // Classification head
    x = head_->forward(x);

    return x;
}

auto SwinTransformer::forward_features(const Variable& input) -> std::vector<Variable> {
    std::vector<Variable> features;

    // Patch embedding
    auto x = patch_embed_->forward(input);
    x = pos_drop_->forward(x);

    // Extract features from each stage
    for (size_t i = 0; i < layers_.size(); ++i) {
        x = layers_[i]->forward(x);

        // Save features before downsampling
        auto resolution = layers_[i]->output_resolution();
        auto shape = x.tensor().shape();
        if (shape.size() != 3) {
            throw std::runtime_error(
                "Expected 3D tensor from layer (B, L, C), got " +
                std::to_string(shape.size()) + "D");
        }
        int64_t B = shape[0];
        int64_t C = shape[2];

        // Reshape to (B, H, W, C) for compatibility
        auto feat = x.reshape({B, resolution.first, resolution.second, C});
        features.push_back(feat);
    }

    return features;
}

auto SwinTransformer::load_pretrained(const std::string& path) -> void {
    // Load checkpoint from file
    nn::ModelCheckpoint checkpoint_manager;
    try {
        auto checkpoint = checkpoint_manager.load(path);

        // Validate checkpoint has model state
        if (checkpoint.model_state.empty()) {
            throw std::runtime_error("Checkpoint file contains no model state");
        }

        // Load state dictionary into the model
        load_state_dict(checkpoint.model_state);

    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load pretrained weights from '" + path + "': " + e.what());
    }
}

// ============================================================================
// Factory Functions
// ============================================================================

auto swin_tiny(int64_t num_classes, int64_t img_size, bool pretrained, bool use_checkpoint)
    -> std::shared_ptr<SwinTransformer>
{
    auto model = std::make_shared<SwinTransformer>(
        img_size,
        4,                          // patch_size
        3,                          // in_chans
        num_classes,
        96,                         // embed_dim
        std::vector<int64_t>{2, 2, 6, 2},    // depths
        std::vector<int64_t>{3, 6, 12, 24},  // num_heads
        7,                          // window_size
        4.0,                        // mlp_ratio
        true,                       // qkv_bias
        0.0,                        // qk_scale
        0.0,                        // drop_rate
        0.0,                        // attn_drop_rate
        0.1,                        // drop_path_rate
        true,                       // norm_layer
        use_checkpoint              // gradient checkpointing
    );

    if (pretrained) {
        throw std::runtime_error(
            "Pretrained Swin Transformer weights not available. "
            "To use pretrained weights:\n"
            "  1. Download weights from a pretrained source\n"
            "  2. Convert to Tenzor checkpoint format if needed\n"
            "  3. Load using: model->load_pretrained(\"path/to/weights.pt\")\n"
            "For training from scratch, set pretrained=false"
        );
    }

    return model;
}

auto swin_small(int64_t num_classes, int64_t img_size, bool pretrained, bool use_checkpoint)
    -> std::shared_ptr<SwinTransformer>
{
    auto model = std::make_shared<SwinTransformer>(
        img_size,
        4,                          // patch_size
        3,                          // in_chans
        num_classes,
        96,                         // embed_dim
        std::vector<int64_t>{2, 2, 18, 2},   // depths
        std::vector<int64_t>{3, 6, 12, 24},  // num_heads
        7,                          // window_size
        4.0,                        // mlp_ratio
        true,                       // qkv_bias
        0.0,                        // qk_scale
        0.0,                        // drop_rate
        0.0,                        // attn_drop_rate
        0.2,                        // drop_path_rate
        true,                       // norm_layer
        use_checkpoint              // gradient checkpointing
    );

    if (pretrained) {
        throw std::runtime_error("Pretrained weights not yet available");
    }

    return model;
}

auto swin_base(int64_t num_classes, int64_t img_size, bool pretrained, bool use_checkpoint)
    -> std::shared_ptr<SwinTransformer>
{
    auto model = std::make_shared<SwinTransformer>(
        img_size,
        4,                          // patch_size
        3,                          // in_chans
        num_classes,
        128,                        // embed_dim
        std::vector<int64_t>{2, 2, 18, 2},   // depths
        std::vector<int64_t>{4, 8, 16, 32},  // num_heads
        7,                          // window_size
        4.0,                        // mlp_ratio
        true,                       // qkv_bias
        0.0,                        // qk_scale
        0.0,                        // drop_rate
        0.0,                        // attn_drop_rate
        0.5,                        // drop_path_rate
        true,                       // norm_layer
        use_checkpoint              // gradient checkpointing
    );

    if (pretrained) {
        throw std::runtime_error("Pretrained weights not yet available");
    }

    return model;
}

auto swin_large(int64_t num_classes, int64_t img_size, bool pretrained, bool use_checkpoint)
    -> std::shared_ptr<SwinTransformer>
{
    auto model = std::make_shared<SwinTransformer>(
        img_size,
        4,                          // patch_size
        3,                          // in_chans
        num_classes,
        192,                        // embed_dim
        std::vector<int64_t>{2, 2, 18, 2},   // depths
        std::vector<int64_t>{6, 12, 24, 48}, // num_heads
        7,                          // window_size
        4.0,                        // mlp_ratio
        true,                       // qkv_bias
        0.0,                        // qk_scale
        0.0,                        // drop_rate
        0.0,                        // attn_drop_rate
        0.5,                        // drop_path_rate
        true,                       // norm_layer
        use_checkpoint              // gradient checkpointing
    );

    if (pretrained) {
        throw std::runtime_error("Pretrained weights not yet available");
    }

    return model;
}

} // namespace tenzor::models
