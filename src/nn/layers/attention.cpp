/**
 * @file attention.cpp
 * @brief Implementation of multi-head attention mechanisms
 */

#include "tenzor/nn/layers/attention.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/autograd/ops.hpp"
#include <cmath>
#include <stdexcept>
#include <limits>

// Include fused attention kernel for CPU optimization
#include "../../backends/cpu/kernels/fused_attention.hpp"

// Include dispatch for fused attention
#include "tenzor/backend/fast_dispatch.hpp"

namespace tenzor {
namespace nn {

// Namespace alias for autograd operations
namespace autograd = tenzor;

// ============================================================================
// MultiheadAttention Implementation
// ============================================================================

MultiheadAttention::MultiheadAttention(int64_t embed_dim,
                                     int64_t num_heads,
                                     double dropout,
                                     bool bias,
                                     bool add_bias_kv,
                                     bool add_zero_attn,
                                     int64_t kdim,
                                     int64_t vdim,
                                     bool batch_first)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      kdim_(kdim > 0 ? kdim : embed_dim),
      vdim_(vdim > 0 ? vdim : embed_dim),
      dropout_(dropout),
      batch_first_(batch_first),
      add_zero_attn_(add_zero_attn) {

    // Validate parameters
    if (embed_dim_ % num_heads_ != 0) {
        throw std::invalid_argument(
            "embed_dim must be divisible by num_heads. Got embed_dim=" +
            std::to_string(embed_dim_) + ", num_heads=" + std::to_string(num_heads_));
    }

    if (dropout_ < 0.0 || dropout_ > 1.0) {
        throw std::invalid_argument(
            "dropout probability must be in [0, 1]. Got " + std::to_string(dropout_));
    }

    head_dim_ = embed_dim_ / num_heads_;

    // Create projection layers
    q_proj_ = std::make_shared<Linear>(embed_dim_, embed_dim_, bias);
    k_proj_ = std::make_shared<Linear>(kdim_, embed_dim_, bias);
    v_proj_ = std::make_shared<Linear>(vdim_, embed_dim_, bias);
    out_proj_ = std::make_shared<Linear>(embed_dim_, embed_dim_, bias);

    // Create dropout layer
    dropout_layer_ = std::make_shared<Dropout>(dropout_);

    // Register submodules
    register_module("q_proj", q_proj_);
    register_module("k_proj", k_proj_);
    register_module("v_proj", v_proj_);
    register_module("out_proj", out_proj_);
    register_module("dropout", dropout_layer_);
}

auto MultiheadAttention::transpose_for_scores(const Variable& x) const -> Variable {
    // Input: (batch, seq_len, embed_dim)
    // Output: (batch, num_heads, seq_len, head_dim)

    auto shape = x.shape();
    int64_t batch_size = shape[0];
    int64_t seq_len = shape[1];

    // Reshape to (batch, seq_len, num_heads, head_dim)
    std::vector<int64_t> new_shape = {batch_size, seq_len, num_heads_, head_dim_};
    auto reshaped = autograd::reshape(x, new_shape);

    // Permute to (batch, num_heads, seq_len, head_dim)
    std::vector<int64_t> perm = {0, 2, 1, 3};
    auto result = autograd::permute(reshaped, perm);

    return result;
}

auto MultiheadAttention::merge_heads(const Variable& x) const -> Variable {
    // Input: (batch, num_heads, seq_len, head_dim)
    // Output: (batch, seq_len, embed_dim)

    auto shape = x.shape();
    int64_t batch_size = shape[0];
    int64_t seq_len = shape[2];

    // Permute to (batch, seq_len, num_heads, head_dim)
    std::vector<int64_t> perm = {0, 2, 1, 3};
    auto permuted = autograd::permute(x, perm);

    // Reshape to (batch, seq_len, embed_dim)
    std::vector<int64_t> new_shape = {batch_size, seq_len, embed_dim_};
    auto result = autograd::reshape(permuted, new_shape);

    return result;
}

auto MultiheadAttention::scaled_dot_product_attention(
    const Variable& query,
    const Variable& key,
    const Variable& value,
    const Tensor& attn_mask,
    double dropout_p) const -> std::pair<Variable, Variable> {

    // Query: (batch, num_heads, seq_len_q, head_dim)
    // Key: (batch, num_heads, seq_len_k, head_dim)
    // Value: (batch, num_heads, seq_len_k, head_dim)

    auto q_shape = query.shape();
    int64_t batch_size = q_shape[0];
    int64_t num_heads = q_shape[1];
    int64_t seq_len_q = q_shape[2];
    int64_t head_dim = q_shape[3];

    auto k_shape = key.shape();
    int64_t seq_len_k = k_shape[2];

    // Use fused CPU kernel for inference when conditions allow:
    // - CPU device, Float32, no mask, no dropout (or eval mode)
    bool can_use_fused = query.device().type == Device::Type::CPU &&
                         query.dtype() == DType::Float32 &&
                         attn_mask.shape().size() == 0 &&  // No attention mask
                         (dropout_p <= 0.0 || !is_training());  // No dropout needed

    if (can_use_fused && !is_training()) {
        // Fast path: Use fused SDPA kernel
        int64_t batch_heads = batch_size * num_heads;

        // Make tensors contiguous if needed (permute/transpose creates non-contiguous views)
        Tensor q_contig = query.tensor().is_contiguous() ? query.tensor() : query.tensor().contiguous();
        Tensor k_contig = key.tensor().is_contiguous() ? key.tensor() : key.tensor().contiguous();
        Tensor v_contig = value.tensor().is_contiguous() ? value.tensor() : value.tensor().contiguous();

        // Get raw data pointers
        const float* Q = q_contig.template data<float>();
        const float* K = k_contig.template data<float>();
        const float* V = v_contig.template data<float>();

        // Allocate output tensor
        std::vector<int64_t> out_shape = {batch_size, num_heads, seq_len_q, head_dim};
        Tensor output_tensor = empty(out_shape, DType::Float32, query.device());
        float* output = output_tensor.data<float>();

        // Call fused kernel
        cpu::attention::scaled_dot_product_attention(
            Q, K, V, output,
            batch_heads,
            seq_len_q, seq_len_k,
            head_dim, head_dim,  // d_k = d_v = head_dim
            false  // causal = false (can be extended later)
        );

        // Return result (no attention weights computed in fused path for efficiency)
        Variable attended(output_tensor, false);

        // Return empty attention weights in fused path (saves memory/compute)
        Tensor empty_weights;
        Variable attn_weights(empty_weights, false);

        return {attended, attn_weights};
    }

    // CUDA Fast path: Flash Attention kernel
    // NOTE: Currently disabled because cuBLAS BMM with Tensor Cores is faster
    // than our custom Flash Attention kernel on modern GPUs (Ampere+).
    // For Flash Attention to be faster, we need to use CUTLASS or wmma intrinsics
    // to leverage Tensor Cores for the Q@K and attn@V matmuls.
    // The cuBLAS path below achieves reasonable performance through:
    // 1. Tensor Core acceleration for matmuls
    // 2. Efficient batched operations
    // 3. Optimized softmax kernel
    bool can_use_cuda_fused = false;

    if (can_use_cuda_fused) {
        // Flash Attention: O(N) memory instead of O(N^2)
        float scale_f = 1.0f / std::sqrt(static_cast<float>(head_dim));

        // Reshape from (batch, num_heads, seq_len, head_dim) to (batch*num_heads, seq_len, head_dim)
        int64_t batch_heads = batch_size * num_heads;
        std::vector<int64_t> reshaped_shape = {batch_heads, seq_len_q, head_dim};
        std::vector<int64_t> reshaped_k_shape = {batch_heads, seq_len_k, head_dim};

        // Make tensors contiguous for fused kernel
        Tensor q_contig = query.tensor().is_contiguous() ? query.tensor() : query.tensor().contiguous();
        Tensor k_contig = key.tensor().is_contiguous() ? key.tensor() : key.tensor().contiguous();
        Tensor v_contig = value.tensor().is_contiguous() ? value.tensor() : value.tensor().contiguous();

        // Reshape to 3D for Flash Attention
        Tensor q_3d = reshape(q_contig, reshaped_shape);
        Tensor k_3d = reshape(k_contig, reshaped_k_shape);
        Tensor v_3d = reshape(v_contig, reshaped_k_shape);

        // Call Flash Attention via dispatch system
        OpAttributes attrs;
        attrs["scale"] = std::to_string(scale_f);
        std::vector<Tensor> fused_inputs = {q_3d, k_3d, v_3d};
        Tensor output_3d = dispatch<OpId::FusedAttention>(fused_inputs, attrs)[0];

        // Reshape back to (batch, num_heads, seq_len_q, head_dim)
        std::vector<int64_t> out_shape = {batch_size, num_heads, seq_len_q, head_dim};
        Tensor output_4d = reshape(output_3d, out_shape);

        Variable attended(output_4d, false);
        Tensor empty_weights;
        Variable attn_weights_empty(empty_weights, false);

        return {attended, attn_weights_empty};
    }

    // Standard path: Use cuBLAS bmm operations (fast for all cases)

    // Compute scaling factor
    double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

    // Reshape Q, K, V from (batch, num_heads, seq_len, head_dim) to (batch*num_heads, seq_len, head_dim)
    // This allows us to use bmm for efficient batched matrix multiplication
    std::vector<int64_t> reshaped_q_shape = {batch_size * num_heads, seq_len_q, head_dim};
    std::vector<int64_t> reshaped_k_shape = {batch_size * num_heads, seq_len_k, head_dim};
    std::vector<int64_t> reshaped_v_shape = {batch_size * num_heads, seq_len_k, head_dim};

    auto query_3d = autograd::reshape(query, reshaped_q_shape);
    auto key_3d = autograd::reshape(key, reshaped_k_shape);
    auto value_3d = autograd::reshape(value, reshaped_v_shape);

    // Transpose key: (batch*num_heads, seq_len_k, head_dim) -> (batch*num_heads, head_dim, seq_len_k)
    std::vector<int64_t> key_perm = {0, 2, 1};
    auto key_transposed = autograd::permute(key_3d, key_perm);

    // Compute attention scores: QK^T using batch matrix multiplication
    // (batch*num_heads, seq_len_q, head_dim) @ (batch*num_heads, head_dim, seq_len_k)
    // Result: (batch*num_heads, seq_len_q, seq_len_k)
    auto scores = autograd::bmm(query_3d, key_transposed);

    // Reshape scores back to (batch, num_heads, seq_len_q, seq_len_k)
    std::vector<int64_t> scores_4d_shape = {batch_size, num_heads, seq_len_q, seq_len_k};
    scores = autograd::reshape(scores, scores_4d_shape);

    // Scale scores
    Tensor scale_tensor = full({1}, static_cast<float>(scale), query.dtype(), query.device());
    Variable scale_var(scale_tensor, false);

    scores = scores * scale_var;

    // Apply attention mask if provided
    if (attn_mask.shape().size() > 0) {
        // Add mask (mask should have -inf for positions to mask out)
        Variable mask_var(attn_mask, false);
        scores = scores + mask_var;
    }

    // Apply softmax to get attention weights
    // Softmax over last dimension (seq_len_k)
    Variable attn_weights = autograd::softmax(scores, -1);

    // Apply dropout if in training mode and dropout > 0
    if (dropout_p > 0.0 && is_training()) {
        attn_weights = dropout_layer_->forward(attn_weights);
    }

    // Reshape attention weights from (batch, num_heads, seq_len_q, seq_len_k) to (batch*num_heads, seq_len_q, seq_len_k)
    auto attn_weights_3d = autograd::reshape(attn_weights, {batch_size * num_heads, seq_len_q, seq_len_k});

    // Compute weighted sum of values using batch matrix multiplication
    // (batch*num_heads, seq_len_q, seq_len_k) @ (batch*num_heads, seq_len_k, head_dim)
    // Result: (batch*num_heads, seq_len_q, head_dim)
    auto attended_3d = autograd::bmm(attn_weights_3d, value_3d);

    // Reshape back to (batch, num_heads, seq_len_q, head_dim)
    std::vector<int64_t> attended_4d_shape = {batch_size, num_heads, seq_len_q, head_dim};
    auto attended = autograd::reshape(attended_3d, attended_4d_shape);

    return {attended, attn_weights};
}

auto MultiheadAttention::forward(const Variable& query,
                                const Variable& key,
                                const Variable& value,
                                const Tensor& key_padding_mask,
                                const Tensor& attn_mask,
                                bool need_weights) -> std::pair<Variable, Variable> {
    // Call forward pre-hooks (enables CPU-start offloading)
    // NOTE: This is necessary because this multi-argument forward bypasses Module::forward()
    call_forward_pre_hooks();

    // Handle batch_first parameter - only transform if needed
    const Variable* q_ptr = &query;
    const Variable* k_ptr = &key;
    const Variable* v_ptr = &value;

    Variable q_permuted, k_permuted, v_permuted;
    if (!batch_first_) {
        // Convert from (seq, batch, embed) to (batch, seq, embed)
        q_permuted = autograd::permute(query, {1, 0, 2});
        k_permuted = autograd::permute(key, {1, 0, 2});
        v_permuted = autograd::permute(value, {1, 0, 2});
        q_ptr = &q_permuted;
        k_ptr = &k_permuted;
        v_ptr = &v_permuted;
    }

    auto q_shape = q_ptr->shape();
    int64_t batch_size = q_shape[0];
    int64_t seq_len_q = q_shape[1];

    auto k_shape = k_ptr->shape();
    int64_t seq_len_k = k_shape[1];

    // Project inputs - use original references, not copies
    Variable Q = q_proj_->forward(*q_ptr);
    Variable K = k_proj_->forward(*k_ptr);
    Variable V = v_proj_->forward(*v_ptr);

    // Reshape for multi-head attention
    Q = transpose_for_scores(Q);  // (batch, num_heads, seq_len_q, head_dim)
    K = transpose_for_scores(K);  // (batch, num_heads, seq_len_k, head_dim)
    V = transpose_for_scores(V);  // (batch, num_heads, seq_len_k, head_dim)

    // Prepare attention mask
    Tensor combined_mask;

    if (attn_mask.shape().size() > 0) {
        combined_mask = attn_mask;
    }

    // Add key padding mask if provided
    if (key_padding_mask.shape().size() > 0) {
        // key_padding_mask: (batch, seq_len_k)
        // Need to broadcast to (batch, 1, 1, seq_len_k) then to (batch, num_heads, seq_len_q, seq_len_k)
        std::vector<int64_t> mask_shape = {batch_size, 1, 1, seq_len_k};
        Tensor padding_mask = reshape(key_padding_mask, mask_shape);

        // Save original device and move to CPU for pointer-based computation
        Device original_device = padding_mask.device();
        Tensor padding_mask_cpu = (original_device == Device::cpu()) ? padding_mask : padding_mask.to(Device::cpu());

        // Create -inf tensor for masked positions on CPU
        Tensor neg_inf_tensor = zeros(std::vector<int64_t>(padding_mask_cpu.shape().begin(), padding_mask_cpu.shape().end()),
                                      padding_mask_cpu.dtype(), Device::cpu());
        auto* mask_data = padding_mask_cpu.data<float>();
        auto* neg_inf_data = neg_inf_tensor.data<float>();

        for (int64_t i = 0; i < padding_mask_cpu.numel(); ++i) {
            neg_inf_data[i] = mask_data[i] > 0.5f ? -std::numeric_limits<float>::infinity() : 0.0f;
        }

        // Move result back to original device if needed
        if (original_device != Device::cpu()) {
            neg_inf_tensor = neg_inf_tensor.to(original_device);
        }

        // Broadcast to full attention shape
        std::vector<int64_t> full_mask_shape = {batch_size, num_heads_, seq_len_q, seq_len_k};
        Tensor broadcasted_mask = expand(neg_inf_tensor, full_mask_shape);

        if (combined_mask.shape().size() > 0) {
            combined_mask = Tensor(add(combined_mask, broadcasted_mask));
        } else {
            combined_mask = broadcasted_mask;
        }
    }

    // Compute attention
    auto [attended_values, attn_weights] = scaled_dot_product_attention(Q, K, V, combined_mask, dropout_);

    // Merge heads
    Variable output = merge_heads(attended_values);

    // Final output projection
    output = out_proj_->forward(output);

    // Convert back to (seq, batch, embed) if needed
    if (!batch_first_) {
        output = autograd::permute(output, {1, 0, 2});
    }

    // Call forward post-hooks (enables CPU-start offloading)
    call_forward_post_hooks();

    // Return attention weights based on need_weights flag
    if (need_weights) {
        return {output, attn_weights};
    } else {
        // Create empty Variable with no shape when weights not needed
        // Use zeros with empty shape to ensure proper initialization
        Tensor empty_tensor = zeros({}, DType::Float32, output.device());
        Variable empty_var(empty_tensor, false);  // requires_grad=false
        return {output, empty_var};
    }
}

auto MultiheadAttention::forward_impl(const Variable& input) -> Variable {
    // Self-attention: Q = K = V = input
    auto [output, _] = forward(input, input, input, Tensor{}, Tensor{}, false);
    return output;
}

// ============================================================================
// Helper Functions
// ============================================================================

auto create_causal_mask(int64_t seq_len, Device device, DType dtype) -> Tensor {
    // Create upper triangular matrix with -inf above diagonal
    // Always create on CPU first as Float32 to avoid dereferencing device pointers
    Tensor mask = zeros({seq_len, seq_len}, DType::Float32, Device::cpu());
    auto* data = mask.data<float>();

    for (int64_t i = 0; i < seq_len; ++i) {
        for (int64_t j = 0; j < seq_len; ++j) {
            if (j > i) {
                data[i * seq_len + j] = -std::numeric_limits<float>::infinity();
            } else {
                data[i * seq_len + j] = 0.0f;
            }
        }
    }

    // Convert to target dtype if needed
    if (dtype != DType::Float32) {
        mask = mask.to(dtype);
    }

    // Move to target device if needed
    if (device != Device::cpu()) {
        return mask.to(device);
    }

    return mask;
}

} // namespace nn
} // namespace tenzor
