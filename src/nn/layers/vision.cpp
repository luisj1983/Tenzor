/**
 * @file vision.cpp
 * @brief Implementation of vision-specific neural network layers
 */

#include "tenzor/nn/layers/vision.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/nn/init.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/vision.hpp"
#include "tenzor/backend/registry.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/jit/tracer.hpp"
#include <stdexcept>
#include <cmath>
#include <random>
#include <iostream>

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

auto PatchEmbedding::forward_impl(const Variable& input) -> Variable {
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

    // Initialize with truncated normal distribution (mean=0, std=0.02, range=[-2*std, 2*std]).
    // Route through the library's seeded init utility so results are
    // reproducible under manual_seed instead of a non-deterministic
    // std::random_device.
    init::trunc_normal_(bias_table, /*mean=*/0.0, /*std=*/0.02,
                        /*a=*/-2.0 * 0.02, /*b=*/2.0 * 0.02);

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

auto WindowAttention::get_relative_position_bias() const -> Variable {
    // Gather the relative-position biases THROUGH autograd so gradients flow
    // back into the bias table and it actually trains. Previously this read
    // `.tensor()` and gathered via a raw OpId dispatch, returning a detached
    // Tensor; forward then did `attn + Variable(bias, false)`, severing the
    // table from the loss so relative_position_bias_table_ never updated.
    //
    // Gather through the parameters_-owned Variable (the leaf the optimizer
    // updates and that offload hooks repoint), NOT the standalone
    // relative_position_bias_table_ copy: register_parameter() stores a *copy*,
    // so the gradient must land on the map-owned Variable to reach the
    // optimizer.
    auto bias_table_it = parameters_.find("relative_position_bias_table");
    const std::shared_ptr<Variable>& table_param =
        (bias_table_it != parameters_.end()) ? bias_table_it->second
                                             : relative_position_bias_table_;
    const Variable& table = *table_param;

    int64_t table_size = 2 * window_size_ - 1;
    int64_t num_positions = window_size_ * window_size_;

    // Flatten the table to [table_size*table_size, num_heads] (autograd reshape).
    auto bias_flat = table.reshape({table_size * table_size, num_heads_});

    // Move the (non-differentiable) gather index to the table's device and
    // flatten to 1-D for index_select along dim 0.
    auto target_device = table.tensor().device();
    auto index_on_device = relative_position_index_.device().type == target_device.type
                           ? relative_position_index_
                           : relative_position_index_.to(target_device);
    auto index_flat = index_on_device.reshape({num_positions * num_positions});

    // Autograd index_select: backward scatter-adds grad into the table, so the
    // bias table receives gradient through this gather.
    auto gathered = tenzor::index_select(bias_flat, /*dim=*/0, index_flat);

    // [num_positions*num_positions, num_heads] -> [num_positions, num_positions, num_heads]
    auto bias = gathered.reshape({num_positions, num_positions, num_heads_});

    // Permute to (num_heads, num_positions, num_positions).
    return bias.permute({2, 0, 1});
}

auto WindowAttention::forward(const Variable& input, const Tensor& mask) -> Variable {
    // Call forward pre-hooks (enables offloading)
    // NOTE: This is necessary because this 2-argument forward bypasses Module::forward()
    call_forward_pre_hooks();

    auto shape = input.tensor().shape();

    // Validate input shape
    if (shape.size() != 3) {
        throw std::runtime_error(
            "WindowAttention expects 3D input (B, N, C), got " +
            std::to_string(shape.size()) + "D");
    }

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

    // Split into Q, K, V using autograd-aware slice to preserve gradient flow
    auto q = tenzor::slice(qkv, 0, 0, 1).squeeze(0);  // (B, num_heads, N, head_dim)
    auto k = tenzor::slice(qkv, 0, 1, 2).squeeze(0);
    auto v = tenzor::slice(qkv, 0, 2, 3).squeeze(0);

    // Scale Q and reshape for bmm: (B, num_heads, N, head_dim) -> (B*num_heads, N, head_dim)
    auto q_scaled = q * scale_;
    auto q_3d = q_scaled.reshape({B * num_heads_, N, head_dim_});
    auto k_3d = k.reshape({B * num_heads_, N, head_dim_});
    auto v_3d = v.reshape({B * num_heads_, N, head_dim_});

    // Attention scores: Q @ K^T -> (B*num_heads, N, N) using autograd bmm
    auto attn_3d = tenzor::bmm(q_3d, k_3d.transpose(-2, -1));

    // Reshape back to 4D: (B*num_heads, N, N) -> (B, num_heads, N, N)
    auto attn = attn_3d.reshape({B, num_heads_, N, N});

    // Add relative position bias (computed through autograd so the bias table
    // trains). bias is a Variable carrying grad back to
    // relative_position_bias_table_; adding it with the normal autograd add
    // keeps that gradient path intact.
    auto bias = get_relative_position_bias();  // Variable (num_heads, N, N)
    // JIT-R103: relative_position_bias_table_ stays wherever the module was
    // constructed (CPU by default) until moved, so get_relative_position_bias()
    // (which derives entirely from the table) can return a bias on a
    // different device than the actual input/attn — a lazily-placed module
    // (constructed on CPU, called with a GPU input) hit a device-mismatch
    // dispatch error. Reconcile with the autograd-aware to_device() so
    // gradients still flow back to the table (mirrors Linear::forward_impl's
    // established pattern).
    if (bias.tensor().device() != attn.tensor().device()) {
        bias = tenzor::to_device(bias, attn.tensor().device());
    }
    // Unsqueeze to (1, num_heads, N, N) for broadcasting (autograd free fn)
    bias = tenzor::unsqueeze(bias, 0);
    attn = attn + bias;

    // Apply attention mask if provided
    if (mask.is_valid() && mask.numel() > 0) {
        // JIT-R070: `mask` is a plain (non-Parameter) Tensor argument, so
        // end_trace() freezes it as an opaque trace-time constant rather
        // than a live-rebound leaf — proven empirically: even after fixing
        // the raw Tensor::unsqueeze() calls below to dispatch (a real,
        // necessary fix on its own), tracing once and replaying with a
        // DIFFERENT mask value (same shape — e.g. a caller re-deriving the
        // shifted-window mask per call instead of caching it) silently
        // replays the FIRST call's mask. Matches JIT-R083 (RoPE)/JIT-R090
        // (KVCache)'s established "fail loudly instead of producing wrong
        // numerics" fix: refuse to trace whenever a real mask is supplied.
        // The common, safe case (mask precomputed once per resolution and
        // reused — the standard Swin/ViT pattern) is unaffected since no
        // mask ever needs to vary within a single compiled graph's
        // lifetime there; only a caller that rebuilds/changes the mask
        // across calls to the SAME compiled function is protected here.
        if (::tenzor::jit::Tracer::get_instance().is_tracing()) {
            throw std::runtime_error(
                "WindowAttention::forward: cannot be JIT-traced with a "
                "non-null attention mask — the mask is baked into the "
                "compiled graph as a constant, so a later call with a "
                "different mask (e.g. a different resolution/shift) would "
                "silently replay the wrong mask. Call outside "
                "jit.compile()/jit.trace(), or omit the mask.");
        }
        // mask shape: (num_windows, N, N)
        // attn shape: (B, num_heads, N, N) where B = batch_size * num_windows
        int64_t num_windows = mask.shape()[0];
        int64_t batch_size = B / num_windows;

        // Reshape attn: (B, nH, N, N) -> (batch_size, num_windows, nH, N, N)
        attn = attn.reshape({batch_size, num_windows, num_heads_, N, N});

        // Reshape mask: (num_windows, N, N) -> (1, num_windows, 1, N, N) for broadcasting.
        // Still routed through the dispatched Variable-level overload (not
        // the raw Tensor::unsqueeze()) even though tracing now always
        // refuses above — this keeps the eager computation itself correct
        // and consistent with the rest of the codebase's tracer-visibility
        // conventions, and matters if is_value_identity_op-style analysis
        // is ever added for non-traced call sites.
        auto mask_expanded = tenzor::unsqueeze(
            tenzor::unsqueeze(Variable(mask, false), 0), 2);  // (1, num_windows, 1, N, N)
        attn = attn + mask_expanded;

        // Reshape back: (batch_size, num_windows, nH, N, N) -> (B, nH, N, N)
        attn = attn.reshape({B, num_heads_, N, N});
    }

    // Softmax
    attn = tenzor::softmax(attn, -1);

    // Dropout
    attn = attn_drop_->forward(attn);

    // Reshape for bmm: (B, num_heads, N, N) -> (B*num_heads, N, N)
    auto attn_3d_for_v = attn.reshape({B * num_heads_, N, N});

    // Attention output: attn @ V -> (B*num_heads, N, head_dim) using autograd bmm
    auto x_3d = tenzor::bmm(attn_3d_for_v, v_3d);

    // Reshape to 4D then transpose and reshape: (B*num_heads, N, head_dim) -> (B, num_heads, N, head_dim) -> (B, N, C)
    auto x = x_3d.reshape({B, num_heads_, N, head_dim_}).transpose(1, 2).reshape({B, N, C});

    // Output projection
    x = proj_->forward(x);
    x = proj_drop_->forward(x);

    // Call forward post-hooks (enables offloading)
    call_forward_post_hooks();

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
                                 Device device,
                                 DType dtype) -> Tensor {
    // Create image mask on CPU to identify different regions after cyclic shift
    // (direct memory access requires CPU tensors)
    auto img_mask = zeros({1, H, W, 1}, DType::Float32, Device::cpu());
    auto mask_data = img_mask.data<float>();

    // Partition into regions based on shift using per-cell classification.
    // This matches the standard Swin slice-based partition (and the CUDA
    // create_window_mask_kernel): each cell is labelled exactly once by its
    // (h_region, w_region) pair, where region 2 is the [size-shift, size)
    // slice, region 1 is the [size-window, size-shift) slice, and region 0 is
    // the leading [0, size-window) slice.
    for (int64_t h = 0; h < H; ++h) {
        int64_t h_region = 0;
        if (h >= H - shift_size) h_region = 2;
        else if (h >= H - window_size) h_region = 1;
        for (int64_t w = 0; w < W; ++w) {
            int64_t w_region = 0;
            if (w >= W - shift_size) w_region = 2;
            else if (w >= W - window_size) w_region = 1;
            mask_data[h * W + w] = static_cast<float>(h_region * 3 + w_region);
        }
    }

    // Partition into windows
    auto mask_windows = window_partition(Variable(img_mask, false), window_size);

    // Reshape: (num_windows, M*M, 1) -> (num_windows, M*M)
    mask_windows = mask_windows.squeeze(-1);

    // Create attention mask on CPU
    int64_t num_windows = (H / window_size) * (W / window_size);
    int64_t M = window_size * window_size;
    auto attn_mask = zeros({num_windows, M, M}, DType::Float32, Device::cpu());
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

    // Convert to target dtype and device
    return attn_mask.to(dtype).to(device);
}

// =============================================================================
// Unfold (im2col) module
// =============================================================================

class UnfoldBackward : public Function {
public:
    UnfoldBackward(std::vector<int64_t> input_shape,
                   int64_t kernel_size, int64_t dilation,
                   int64_t padding, int64_t stride)
        : input_shape_(std::move(input_shape)),
          kernel_size_(kernel_size), dilation_(dilation),
          padding_(padding), stride_(stride) {}

    auto forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> override {
        throw std::runtime_error("UnfoldBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        if (grad_outputs.size() != 1) {
            throw std::invalid_argument("UnfoldBackward expects 1 gradient output");
        }
        // Backward of unfold is fold
        std::vector<int64_t> output_size = {input_shape_[2], input_shape_[3]};
        return {tenzor::ops::fold(grad_outputs[0], output_size,
                                  kernel_size_, stride_, padding_, dilation_)};
    }

    // Unfold is a linear rearrangement (im2col) so its second derivative
    // is structurally zero — the passthrough stub produces the correct
    // higher-order gradient.
    TENZOR_HIGHER_ORDER_STRUCTURAL_ZERO_STUB()

private:
    std::vector<int64_t> input_shape_;
    int64_t kernel_size_;
    int64_t dilation_;
    int64_t padding_;
    int64_t stride_;
};

Unfold::Unfold(int64_t kernel_size, int64_t dilation, int64_t padding, int64_t stride)
    : kernel_size_(kernel_size), dilation_(dilation), padding_(padding), stride_(stride) {}

auto Unfold::forward_impl(const Variable& input) -> Variable {
    auto result_tensor = tenzor::ops::unfold(
        input.tensor(), kernel_size_, stride_, padding_, dilation_);

    Variable output(result_tensor, input.requires_grad());

    if (input.requires_grad()) {
        auto input_shape = input.tensor().shape();
        std::vector<int64_t> shape_vec(input_shape.begin(), input_shape.end());

        auto unfold_fn = std::make_shared<UnfoldBackward>(
            shape_vec, kernel_size_, dilation_, padding_, stride_);

        std::vector<Variable> input_vars = {input};
        unfold_fn->set_input_variables(input_vars);

        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) {
            next_funcs.push_back(input.grad_fn());
        }
        unfold_fn->set_next_functions(next_funcs);
        output.set_grad_fn(unfold_fn);
    }

    return output;
}

// =============================================================================
// Fold (col2im) module
// =============================================================================

class FoldBackward : public Function {
public:
    FoldBackward(int64_t kernel_size, int64_t dilation,
                 int64_t padding, int64_t stride)
        : kernel_size_(kernel_size), dilation_(dilation),
          padding_(padding), stride_(stride) {}

    auto forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> override {
        throw std::runtime_error("FoldBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        if (grad_outputs.size() != 1) {
            throw std::invalid_argument("FoldBackward expects 1 gradient output");
        }
        // Backward of fold is unfold
        return {tenzor::ops::unfold(grad_outputs[0],
                                    kernel_size_, stride_, padding_, dilation_)};
    }

    // Fold is a linear rearrangement (col2im) so its second derivative
    // is structurally zero.
    TENZOR_HIGHER_ORDER_STRUCTURAL_ZERO_STUB()

private:
    int64_t kernel_size_;
    int64_t dilation_;
    int64_t padding_;
    int64_t stride_;
};

Fold::Fold(std::vector<int64_t> output_size, int64_t kernel_size,
           int64_t dilation, int64_t padding, int64_t stride)
    : output_size_(std::move(output_size)), kernel_size_(kernel_size),
      dilation_(dilation), padding_(padding), stride_(stride) {}

auto Fold::forward_impl(const Variable& input) -> Variable {
    auto result_tensor = tenzor::ops::fold(
        input.tensor(), output_size_, kernel_size_, stride_, padding_, dilation_);

    Variable output(result_tensor, input.requires_grad());

    if (input.requires_grad()) {
        auto fold_fn = std::make_shared<FoldBackward>(
            kernel_size_, dilation_, padding_, stride_);

        std::vector<Variable> input_vars = {input};
        fold_fn->set_input_variables(input_vars);

        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) {
            next_funcs.push_back(input.grad_fn());
        }
        fold_fn->set_next_functions(next_funcs);
        output.set_grad_fn(fold_fn);
    }

    return output;
}

} // namespace tenzor::nn
