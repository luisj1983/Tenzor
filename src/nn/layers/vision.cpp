/**
 * @file vision.cpp
 * @brief Implementation of vision-specific neural network layers
 */

#include "tenzor/nn/layers/vision.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include <stdexcept>
#include <cmath>

namespace tenzor::nn {

// ============================================================================
// PatchEmbedding Implementation
// ============================================================================

PatchEmbedding::PatchEmbedding(int64_t in_channels,
                               int64_t embed_dim,
                               int64_t patch_size,
                               int64_t img_size)
    : in_channels_(in_channels)
    , embed_dim_(embed_dim)
    , patch_size_(patch_size)
    , img_size_(img_size)
    , num_patches_((img_size / patch_size) * (img_size / patch_size))
{
    if (img_size % patch_size != 0) {
        throw std::invalid_argument(
            "Image size (" + std::to_string(img_size) + ") must be divisible by patch size (" +
            std::to_string(patch_size) + ")");
    }

    // Use Conv2d with kernel=patch_size, stride=patch_size for efficient patch extraction
    // This is mathematically equivalent to unfold + linear but much faster
    proj_ = std::make_shared<Conv2d>(
        in_channels,
        embed_dim,
        patch_size,  // kernel_size
        patch_size,  // stride (same as kernel for non-overlapping patches)
        0,           // padding
        1,           // dilation
        1,           // groups
        true         // bias
    );

    register_module("proj", proj_);
}

auto PatchEmbedding::forward(const Variable& input) -> Variable {
    auto shape = input.tensor().shape();

    // Validate input shape
    if (shape.size() != 4) {
        throw std::runtime_error(
            "PatchEmbedding expects 4D input (N, C, H, W), got " +
            std::to_string(shape.size()) + "D");
    }

    int64_t batch = shape[0];
    int64_t channels = shape[1];
    int64_t height = shape[2];
    int64_t width = shape[3];

    if (channels != in_channels_) {
        throw std::runtime_error(
            "Input channels (" + std::to_string(channels) +
            ") doesn't match expected (" + std::to_string(in_channels_) + ")");
    }

    if (height % patch_size_ != 0 || width % patch_size_ != 0) {
        throw std::runtime_error(
            "Input dimensions (" + std::to_string(height) + "x" + std::to_string(width) +
            ") must be divisible by patch size (" + std::to_string(patch_size_) + ")");
    }

    // Apply convolution to extract patches
    // Input: (N, C, H, W)
    // Output: (N, embed_dim, H/patch_size, W/patch_size)
    auto x = proj_->forward(input);

    // Flatten spatial dimensions
    // (N, embed_dim, H', W') -> (N, embed_dim, H'*W')
    auto x_shape = x.tensor().shape();
    int64_t num_patches = x_shape[2] * x_shape[3];
    x = x.reshape({batch, embed_dim_, num_patches});

    // Transpose to (N, num_patches, embed_dim)
    x = x.transpose(1, 2);

    return x;
}

// ============================================================================
// WindowAttention Implementation
// ============================================================================

WindowAttention::WindowAttention(int64_t dim,
                                 int64_t window_size,
                                 int64_t num_heads,
                                 bool qkv_bias,
                                 double qk_scale,
                                 double attn_drop,
                                 double proj_drop)
    : dim_(dim)
    , window_size_(window_size)
    , num_heads_(num_heads)
    , head_dim_(dim / num_heads)
    , scale_(qk_scale > 0.0 ? qk_scale : 1.0 / std::sqrt(static_cast<double>(head_dim_)))
{
    if (dim % num_heads != 0) {
        throw std::invalid_argument(
            "dim (" + std::to_string(dim) + ") must be divisible by num_heads (" +
            std::to_string(num_heads) + ")");
    }

    // QKV projection (single matrix for efficiency)
    qkv_ = std::make_shared<Linear>(dim, dim * 3, qkv_bias);
    register_module("qkv", qkv_);

    // Attention dropout
    attn_drop_ = std::make_shared<Dropout>(attn_drop);
    register_module("attn_drop", attn_drop_);

    // Output projection
    proj_ = std::make_shared<Linear>(dim, dim, true);
    register_module("proj", proj_);

    // Projection dropout
    proj_drop_ = std::make_shared<Dropout>(proj_drop);
    register_module("proj_drop", proj_drop_);

    // Relative position bias table
    // Table size: (2*window_size-1, 2*window_size-1, num_heads)
    int64_t table_size = 2 * window_size - 1;
    auto bias_table = zeros({table_size, table_size, num_heads}, DType::Float32, Device::cpu());
    // Initialize with small random values
    // TODO: Use proper truncated normal initialization
    auto bias_data = bias_table.data<float>();
    for (int64_t i = 0; i < table_size * table_size * num_heads; ++i) {
        bias_data[i] = (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 0.02f;
    }
    relative_position_bias_table_ = std::make_shared<Variable>(bias_table, true);
    register_parameter("relative_position_bias_table", *relative_position_bias_table_);

    // Compute relative position index
    compute_relative_position_index();
}

auto WindowAttention::compute_relative_position_index() -> void {
    // Create coordinate grids
    std::vector<int64_t> coords;
    for (int64_t i = 0; i < window_size_; ++i) {
        for (int64_t j = 0; j < window_size_; ++j) {
            coords.push_back(i);
            coords.push_back(j);
        }
    }

    int64_t num_positions = window_size_ * window_size_;
    relative_position_index_ = zeros({num_positions, num_positions}, DType::Int64, Device::cpu());
    auto index_data = relative_position_index_.data<int64_t>();

    // Compute relative positions for each (query, key) pair
    for (int64_t i = 0; i < num_positions; ++i) {
        int64_t qi = coords[i * 2];
        int64_t qj = coords[i * 2 + 1];

        for (int64_t j = 0; j < num_positions; ++j) {
            int64_t ki = coords[j * 2];
            int64_t kj = coords[j * 2 + 1];

            // Relative position
            int64_t rel_i = qi - ki + window_size_ - 1;
            int64_t rel_j = qj - kj + window_size_ - 1;

            // Index into bias table
            index_data[i * num_positions + j] = rel_i * (2 * window_size_ - 1) + rel_j;
        }
    }
}

auto WindowAttention::get_relative_position_bias() const -> Tensor {
    // Gather biases using precomputed indices
    auto bias_table = relative_position_bias_table_->tensor();
    int64_t table_size = 2 * window_size_ - 1;
    int64_t num_positions = window_size_ * window_size_;

    // Flatten table for indexing: (table_size * table_size, num_heads)
    auto bias_flat = bias_table.reshape({table_size * table_size, num_heads_});

    // Gather using relative position indices
    auto bias = zeros({num_positions, num_positions, num_heads_}, DType::Float32, bias_table.device());
    auto bias_data = bias.data<float>();
    auto table_data = bias_flat.data<float>();
    auto index_data = relative_position_index_.data<int64_t>();

    for (int64_t i = 0; i < num_positions; ++i) {
        for (int64_t j = 0; j < num_positions; ++j) {
            int64_t idx = index_data[i * num_positions + j];
            for (int64_t h = 0; h < num_heads_; ++h) {
                bias_data[(i * num_positions + j) * num_heads_ + h] =
                    table_data[idx * num_heads_ + h];
            }
        }
    }

    // Permute to (num_heads, num_positions, num_positions)
    return bias.permute({2, 0, 1});
}

auto WindowAttention::forward(const Variable& input, const Tensor& mask) -> Variable {
    auto shape = input.tensor().shape();
    int64_t B = shape[0];      // num_windows * batch
    int64_t N = shape[1];      // window_size * window_size
    int64_t C = shape[2];      // dim

    if (C != dim_) {
        throw std::runtime_error(
            "Input dimension (" + std::to_string(C) + ") doesn't match expected (" +
            std::to_string(dim_) + ")");
    }

    // QKV projection: (B, N, C) -> (B, N, 3*C)
    auto qkv = qkv_->forward(input);

    // Reshape to (B, N, 3, num_heads, head_dim)
    qkv = qkv.reshape({B, N, 3, num_heads_, head_dim_});

    // Permute to (3, B, num_heads, N, head_dim)
    qkv = qkv.permute({2, 0, 3, 1, 4});

    // Split into Q, K, V
    auto qkv_data = qkv.tensor();
    auto q = qkv_data.slice(0, 0, 1).squeeze(0);  // (B, num_heads, N, head_dim)
    auto k = qkv_data.slice(0, 1, 2).squeeze(0);
    auto v = qkv_data.slice(0, 2, 3).squeeze(0);

    // Scale Q
    auto q_var = Variable(q * scale_, input.requires_grad());
    auto k_var = Variable(k, input.requires_grad());
    auto v_var = Variable(v, input.requires_grad());

    // Attention scores: Q @ K^T -> (B, num_heads, N, N)
    auto attn = q_var.matmul(k_var.transpose(-2, -1));

    // Add relative position bias
    auto bias = get_relative_position_bias();  // (num_heads, N, N)
    // Unsqueeze to (1, num_heads, N, N) for broadcasting
    bias = bias.unsqueeze(0);
    attn = attn + Variable(bias, false);

    // Apply attention mask if provided
    if (mask.numel() > 0) {
        // Mask shape: (num_windows, N, N)
        // Unsqueeze to (num_windows, 1, N, N) for broadcasting over heads
        auto mask_expanded = mask.unsqueeze(1);
        attn = attn + Variable(mask_expanded, false);
    }

    // Softmax
    attn = tenzor::softmax(attn, -1);

    // Dropout
    attn = attn_drop_->forward(attn);

    // Attention output: attn @ V -> (B, num_heads, N, head_dim)
    auto x = attn.matmul(v_var);

    // Transpose and reshape: (B, num_heads, N, head_dim) -> (B, N, C)
    x = x.transpose(1, 2).reshape({B, N, C});

    // Output projection
    x = proj_->forward(x);
    x = proj_drop_->forward(x);

    return x;
}

// ============================================================================
// Helper Functions
// ============================================================================

auto window_partition(const Variable& input, int64_t window_size) -> Variable {
    auto shape = input.tensor().shape();
    if (shape.size() != 4) {
        throw std::runtime_error("window_partition expects 4D input (B, H, W, C)");
    }

    int64_t B = shape[0];
    int64_t H = shape[1];
    int64_t W = shape[2];
    int64_t C = shape[3];

    if (H % window_size != 0 || W % window_size != 0) {
        throw std::invalid_argument(
            "Height (" + std::to_string(H) + ") and width (" + std::to_string(W) +
            ") must be divisible by window_size (" + std::to_string(window_size) + ")");
    }

    // Reshape: (B, H, W, C) -> (B, H/M, M, W/M, M, C)
    auto x = input.reshape({B, H / window_size, window_size, W / window_size, window_size, C});

    // Permute: -> (B, H/M, W/M, M, M, C)
    x = x.permute({0, 1, 3, 2, 4, 5});

    // Reshape: -> (B * H/M * W/M, M * M, C)
    int64_t num_windows = (H / window_size) * (W / window_size);
    x = x.reshape({B * num_windows, window_size * window_size, C});

    return x;
}

auto window_reverse(const Variable& windows, int64_t window_size,
                    int64_t H, int64_t W) -> Variable {
    auto shape = windows.tensor().shape();
    if (shape.size() != 3) {
        throw std::runtime_error("window_reverse expects 3D input (B*num_windows, M*M, C)");
    }

    int64_t C = shape[2];
    int64_t num_windows = (H / window_size) * (W / window_size);
    int64_t B = shape[0] / num_windows;

    // Reshape: (B*num_windows, M*M, C) -> (B, H/M, W/M, M, M, C)
    auto x = windows.reshape({B, H / window_size, W / window_size, window_size, window_size, C});

    // Permute: -> (B, H/M, M, W/M, M, C)
    x = x.permute({0, 1, 3, 2, 4, 5});

    // Reshape: -> (B, H, W, C)
    x = x.reshape({B, H, W, C});

    return x;
}

auto create_shifted_window_mask(int64_t H, int64_t W,
                                 int64_t window_size,
                                 int64_t shift_size,
                                 Device device) -> Tensor {
    // Create image mask to identify different regions after cyclic shift
    auto img_mask = zeros({1, H, W, 1}, DType::Float32, device);
    auto mask_data = img_mask.data<float>();

    // Partition into regions based on shift
    int64_t cnt = 0;
    for (int64_t h_start : {int64_t(0), H - window_size, H - shift_size}) {
        for (int64_t w_start : {int64_t(0), W - window_size, W - shift_size}) {
            for (int64_t h = h_start; h < std::min(h_start + window_size, H); ++h) {
                for (int64_t w = w_start; w < std::min(w_start + window_size, W); ++w) {
                    mask_data[h * W + w] = static_cast<float>(cnt);
                }
            }
            cnt++;
        }
    }

    // Partition into windows
    auto mask_windows = window_partition(Variable(img_mask, false), window_size);

    // Reshape: (num_windows, M*M, 1) -> (num_windows, M*M)
    mask_windows = mask_windows.squeeze(-1);

    // Create attention mask
    int64_t num_windows = (H / window_size) * (W / window_size);
    int64_t M = window_size * window_size;
    auto attn_mask = zeros({num_windows, M, M}, DType::Float32, device);
    auto attn_data = attn_mask.data<float>();
    auto window_data = mask_windows.tensor().data<float>();

    // Set mask values: -100 where regions differ, 0 where same
    for (int64_t w = 0; w < num_windows; ++w) {
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < M; ++j) {
                float val_i = window_data[w * M + i];
                float val_j = window_data[w * M + j];
                attn_data[w * M * M + i * M + j] = (val_i != val_j) ? -100.0f : 0.0f;
            }
        }
    }

    return attn_mask;
}

} // namespace tenzor::nn
