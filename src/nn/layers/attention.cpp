/**
 * @file attention.cpp
 * @brief Implementation of multi-head attention mechanisms
 */

#include "tenzor/nn/layers/attention.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include <cmath>
#include <stdexcept>
#include <limits>

// Include fused attention kernel for CPU optimization
#include "../../backends/cpu/kernels/fused_attention.hpp"

// Include dispatch for fused attention
#include "tenzor/backend/fast_dispatch.hpp"

// Include cuDNN SDPA for optimized CUDA attention
#include "tenzor/backend/fused_ops.hpp"

namespace tenzor {
namespace nn {

// Namespace alias for autograd operations
namespace autograd = tenzor;

// Autograd-aware dtype cast (gradient flows through with proper dtype conversion)
class AttentionTypeCastBackward : public Function {
public:
    DType original_dtype_ = DType::Float32;
    auto forward(std::vector<Variable>) -> std::vector<Variable> override {
        throw std::runtime_error("AttentionTypeCastBackward::forward should not be called");
    }
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& grad = grad_outputs[0];
        return {(grad.dtype() != original_dtype_) ? grad.to(original_dtype_) : grad};
    }
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        auto& grad = grad_outputs[0];
        if (grad.dtype() != original_dtype_) {
            return {Variable(grad.tensor().to(original_dtype_), grad.requires_grad())};
        }
        return {grad};
    }
};

static auto attention_cast(const Variable& input, DType target_dtype) -> Variable {
    if (input.dtype() == target_dtype) return input;
    auto converted = input.tensor().to(target_dtype);
    Variable result(converted, input.requires_grad());
    if (input.requires_grad() && is_grad_enabled()) {
        auto grad_fn = std::make_shared<AttentionTypeCastBackward>();
        grad_fn->original_dtype_ = input.dtype();
        std::vector<Variable> input_vars = {input};
        grad_fn->set_input_variables(input_vars);
        if (auto fn = input.grad_fn()) {
            grad_fn->set_next_functions({fn});
        }
        result.set_grad_fn(grad_fn);
    }
    return result;
}

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
                                     bool batch_first,
                                     bool is_causal)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      kdim_(kdim > 0 ? kdim : embed_dim),
      vdim_(vdim > 0 ? vdim : embed_dim),
      dropout_(dropout),
      batch_first_(batch_first),
      add_zero_attn_(add_zero_attn),
      is_causal_(is_causal) {

    // Validate parameters
    if (embed_dim_ <= 0) {
        throw std::invalid_argument(
            "embed_dim must be positive. Got embed_dim=" + std::to_string(embed_dim_));
    }
    if (num_heads_ <= 0) {
        throw std::invalid_argument(
            "num_heads must be positive. Got num_heads=" + std::to_string(num_heads_));
    }
    if (embed_dim_ % num_heads_ != 0) {
        throw std::invalid_argument(
            "embed_dim must be divisible by num_heads. Got embed_dim=" +
            std::to_string(embed_dim_) + ", num_heads=" + std::to_string(num_heads_));
    }

    if (dropout_ < 0.0 || dropout_ > 1.0) {
        throw std::invalid_argument(
            "dropout probability must be in [0, 1]. Got " + std::to_string(dropout_));
    }

    add_bias_kv_ = add_bias_kv;

    if (add_bias_kv) {
        // Initialize bias_k and bias_v as learnable parameters [1, 1, embed_dim]
        auto bk = tenzor::zeros({1, 1, embed_dim_}, DType::Float32);
        auto bv = tenzor::zeros({1, 1, embed_dim_}, DType::Float32);
        bias_k_ = Variable(bk, true);
        bias_v_ = Variable(bv, true);
        register_parameter("bias_k", bias_k_);
        register_parameter("bias_v", bias_v_);
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
    double dropout_p,
    bool need_weights,
    const Tensor& position_bias) const -> std::pair<Variable, Variable> {

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
    // - CPU device, Float32, no mask (or is_causal handles masking), no dropout (or eval mode)
    // - need_weights must be false (fused path doesn't compute attention weights)
    bool can_use_fused = !need_weights &&
                         query.device().type == Device::Type::CPU &&
                         query.dtype() == DType::Float32 &&
                         (is_causal_ || !attn_mask.is_valid() || attn_mask.shape().size() == 0) &&  // No explicit mask needed
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
            is_causal_
        );

        // Return result (no attention weights computed in fused path for efficiency)
        Variable attended(output_tensor, false);

        // Return zero-shaped weights to indicate fused path (no weights computed)
        Tensor empty_weights({0}, DType::Float32, output_tensor.device());
        Variable attn_weights(empty_weights, false);

        return {attended, attn_weights};
    }

    // CUDA Fast path: cuDNN SDPA with Flash Attention and Tensor Cores
    // Uses cuDNN Graph API for fused attention via dispatch - up to 2x faster than separate BMM ops
    // Requirements: cuDNN 9.0+, Ampere+ GPU, head_dim in {32, 64, 128, 256}, FP16 input
    // Note: Only enabled for FP16 inputs - FP32→FP16 conversion overhead makes BMM faster for FP32
    // Note: Only enabled in inference mode - cuDNN SDPA doesn't build autograd graph
    // need_weights must be false (cuDNN SDPA path doesn't compute attention weights)
    bool is_fp16 = query.dtype() == DType::Float16 || query.dtype() == DType::BFloat16;
    bool can_use_cudnn_sdpa = !need_weights &&
        is_fp16 &&
        (query.device().type == Device::Type::CUDA) &&
        (head_dim == 32 || head_dim == 64 || head_dim == 128 || head_dim == 256) &&
        !is_training();  // Only use cuDNN SDPA in inference mode

    if (can_use_cudnn_sdpa) {
        try {
            float scale_f = 1.0f / std::sqrt(static_cast<float>(head_dim));

            // Make tensors contiguous for cuDNN
            Tensor q_contig = query.tensor().is_contiguous() ? query.tensor() : query.tensor().contiguous();
            Tensor k_contig = key.tensor().is_contiguous() ? key.tensor() : key.tensor().contiguous();
            Tensor v_contig = value.tensor().is_contiguous() ? value.tensor() : value.tensor().contiguous();

            // Use dispatch to call cuDNN SDPA - pass 4D tensors directly
            OpAttributes attrs;
            attrs.set(AttrKey::Scale, static_cast<double>(scale_f));
            attrs.set(AttrKey::UseCudnnSdpa, true);
            std::vector<Tensor> fused_inputs = {q_contig, k_contig, v_contig};
            Tensor output = dispatch<OpId::FusedAttention>(fused_inputs, attrs)[0];

            Variable attended(output, false);
            Tensor empty_weights({0}, output.dtype(), output.device());
            Variable attn_weights_empty(empty_weights, false);

            return {attended, attn_weights_empty};
        } catch (const std::exception& e) {
            // Fall through to BMM path if cuDNN SDPA fails
            // This can happen if GPU doesn't support SDPA or other issues
        }
    }

    // CPU Flash Attention: O(N) memory via tiled online softmax with fused dropout
    // Uses OpId::FlashAttention for 4D [batch, num_heads, seq_len, head_dim] tensors
    // Conditions: CPU, Float32, head_dim <= 256, no explicit attention mask
    // When is_causal_ is true, the kernel handles causal masking internally (no explicit mask needed)
    // When is_causal_ is false, we still require no external mask for the flash path
    // Dropout is handled inside the fused kernel via Philox RNG (no training guard needed)
    // need_weights must be false (flash attention path doesn't compute attention weights)
    bool can_use_flash_attention = !need_weights &&
                                   query.device().type == Device::Type::CPU &&
                                   query.dtype() == DType::Float32 &&
                                   head_dim <= 256 &&
                                   (is_causal_ || !attn_mask.is_valid() || attn_mask.shape().size() == 0);

    if (can_use_flash_attention && !is_training()) {
        try {
            float scale_f = 1.0f / std::sqrt(static_cast<float>(head_dim));

            // Make tensors contiguous for the kernel
            Tensor q_contig = query.tensor().is_contiguous() ? query.tensor() : query.tensor().contiguous();
            Tensor k_contig = key.tensor().is_contiguous() ? key.tensor() : key.tensor().contiguous();
            Tensor v_contig = value.tensor().is_contiguous() ? value.tensor() : value.tensor().contiguous();

            // Call Flash Attention via dispatch system (4D tensors directly)
            // When is_causal_ is true, the kernel applies causal masking internally,
            // which is more efficient than building and applying an explicit mask tensor
            // Dropout is fused into the kernel using Philox counter-based RNG
            OpAttributes attrs;
            attrs.set(AttrKey::Scale, static_cast<double>(scale_f));
            attrs.set(AttrKey::Causal, is_causal_);
            attrs.set(AttrKey::DropoutP, static_cast<double>(dropout_p));
            attrs.set(AttrKey::IsTraining, is_training());
            std::vector<Tensor> flash_inputs = {q_contig, k_contig, v_contig};
            Tensor output = dispatch<OpId::FlashAttention>(flash_inputs, attrs)[0];

            Variable attended(output, false);
            Tensor empty_weights({0}, output.dtype(), output.device());
            Variable attn_weights_empty(empty_weights, false);

            return {attended, attn_weights_empty};
        } catch (const std::exception& e) {
            // Fall through to standard BMM path if flash attention fails
        }
    }

    // Standard path: Use cuBLAS bmm operations (fast for all cases)

    // For Float16/BFloat16, upcast Q, K, V to Float32 for the full attention computation
    // to prevent gradient overflow. This matches PyTorch's scaled_dot_product_attention behavior.
    DType orig_dtype = query.dtype();
    bool needs_attn_upcast = (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16);
    Variable query_compute = needs_attn_upcast ? attention_cast(query, DType::Float32) : query;
    Variable key_compute = needs_attn_upcast ? attention_cast(key, DType::Float32) : key;
    Variable value_compute = needs_attn_upcast ? attention_cast(value, DType::Float32) : value;

    // Compute scaling factor
    double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

    // Reshape Q, K, V from (batch, num_heads, seq_len, head_dim) to (batch*num_heads, seq_len, head_dim)
    // This allows us to use bmm for efficient batched matrix multiplication
    std::vector<int64_t> reshaped_q_shape = {batch_size * num_heads, seq_len_q, head_dim};
    std::vector<int64_t> reshaped_k_shape = {batch_size * num_heads, seq_len_k, head_dim};
    std::vector<int64_t> reshaped_v_shape = {batch_size * num_heads, seq_len_k, head_dim};

    auto query_3d = autograd::reshape(query_compute, reshaped_q_shape);
    auto key_3d = autograd::reshape(key_compute, reshaped_k_shape);
    auto value_3d = autograd::reshape(value_compute, reshaped_v_shape);

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

    // Scale scores (use Float32 for reduced-precision types)
    DType score_dtype = needs_attn_upcast ? DType::Float32 : orig_dtype;
    Tensor scale_tensor = full({1}, static_cast<float>(scale), score_dtype, query.device());
    Variable scale_var(scale_tensor, false);

    scores = scores * scale_var;

    // Apply relative position bias if provided
    // position_bias shape: (num_heads, seq_len_q, seq_len_k) or (1, num_heads, seq_len_q, seq_len_k)
    if (position_bias.is_valid() && position_bias.shape().size() > 0) {
        Variable bias_var(position_bias, false);
        scores = scores + bias_var;
    }

    // Apply causal mask if is_causal_ is set (masks future tokens)
    if (is_causal_) {
        // Create rectangular causal mask on target device using tensor ops
        // triu with diagonal=1 gives upper-triangular 1s above the main diagonal
        float neg_inf_val = -std::numeric_limits<float>::infinity();
        if (!needs_attn_upcast && (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16)) {
            neg_inf_val = -1e4f;
        }
        auto causal = triu(ones({seq_len_q, seq_len_k}, score_dtype, query.device()), 1);
        causal = causal * full({1}, neg_inf_val, score_dtype, query.device());
        Variable causal_var(causal, false);
        scores = scores + causal_var;
    }

    // Apply attention mask if provided
    if (attn_mask.is_valid() && attn_mask.shape().size() > 0) {
        // Validate mask shape is broadcastable to scores (batch, num_heads, seq_q, seq_k)
        auto mask_shape = attn_mask.shape();
        auto scores_shape = scores.shape();
        // Mask must be 2D (seq_q, seq_k), 3D (num_heads, seq_q, seq_k),
        // or 4D (batch, num_heads, seq_q, seq_k) and broadcastable
        if (mask_shape.size() > 4) {
            throw std::runtime_error(
                "Attention mask must be 2D, 3D, or 4D, got " +
                std::to_string(mask_shape.size()) + "D");
        }
        // Validate trailing dimensions match
        int64_t mask_ndim = static_cast<int64_t>(mask_shape.size());
        int64_t scores_ndim = static_cast<int64_t>(scores_shape.size());
        for (int64_t i = 1; i <= std::min(mask_ndim, scores_ndim); ++i) {
            int64_t mask_dim = mask_shape[mask_ndim - i];
            int64_t scores_dim = scores_shape[scores_ndim - i];
            if (mask_dim != 1 && mask_dim != scores_dim) {
                throw std::runtime_error(
                    "Attention mask shape is not broadcastable to scores shape. "
                    "Mask dim " + std::to_string(mask_ndim - i) + " is " +
                    std::to_string(mask_dim) + " but scores dim is " +
                    std::to_string(scores_dim));
            }
        }
        // Add mask (mask should have -inf for positions to mask out)
        Variable mask_var(attn_mask, false);
        scores = scores + mask_var;
    }

    // Apply softmax to get attention weights
    // Softmax over last dimension (seq_len_k)
    // For Float16/BFloat16, scores are already in Float32 from upcast above
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

    // Downcast attention output back to original dtype
    if (needs_attn_upcast) {
        attended = attention_cast(attended, orig_dtype);
        attn_weights = attention_cast(attn_weights, orig_dtype);
    }

    return {attended, attn_weights};
}

auto MultiheadAttention::forward(const Variable& query,
                                const Variable& key,
                                const Variable& value,
                                const Tensor& key_padding_mask,
                                const Tensor& attn_mask,
                                bool need_weights,
                                const Tensor& position_bias) -> std::pair<Variable, Variable> {
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
    if (q_shape.size() != 3) {
        throw std::invalid_argument("MultiheadAttention: query must be 3D [batch, seq, embed], got " +
            std::to_string(q_shape.size()) + "D");
    }
    auto k_shape = k_ptr->shape();
    if (k_shape.size() != 3 || v_ptr->shape().size() != 3) {
        throw std::invalid_argument("MultiheadAttention: key/value must be 3D");
    }
    if (q_shape[2] != embed_dim_) {
        throw std::invalid_argument("MultiheadAttention: query embed_dim (" +
            std::to_string(q_shape[2]) + ") != expected (" + std::to_string(embed_dim_) + ")");
    }
    int64_t batch_size = q_shape[0];
    int64_t seq_len_q = q_shape[1];
    int64_t seq_len_k = k_shape[1];

    // Ensure projection weights are on the same device as input
    auto weight_device = q_proj_->own_parameters()[0]->tensor().device();
    auto input_device = q_ptr->tensor().device();

    if (weight_device != input_device) {
        // Move projection layers to input device to preserve autograd chain
        q_proj_->to(input_device);
        k_proj_->to(input_device);
        v_proj_->to(input_device);
        out_proj_->to(input_device);
        if (add_bias_kv_) {
            bias_k_ = Variable(bias_k_.tensor().to(input_device), bias_k_.requires_grad());
            bias_v_ = Variable(bias_v_.tensor().to(input_device), bias_v_.requires_grad());
        }
    }

    const Variable& q_compat = *q_ptr;
    const Variable& k_compat = *k_ptr;
    const Variable& v_compat = *v_ptr;

    // Project inputs
    Variable Q = q_proj_->forward(q_compat);
    Variable K = k_proj_->forward(k_compat);
    Variable V = v_proj_->forward(v_compat);

    // add_bias_kv: concatenate bias_k/bias_v to key/value sequences
    if (add_bias_kv_) {
        // bias_k_ is [1, 1, embed_dim], expand to [batch_size, 1, embed_dim]
        auto bk = Variable(tenzor::expand(bias_k_.tensor(), {batch_size, 1, embed_dim_}), false);
        auto bv = Variable(tenzor::expand(bias_v_.tensor(), {batch_size, 1, embed_dim_}), false);
        K = tenzor::cat({K, bk}, 1);  // [batch, seq_k+1, embed]
        V = tenzor::cat({V, bv}, 1);  // [batch, seq_k+1, embed]
        seq_len_k += 1;
    }

    // add_zero_attn: append zero row to key/value sequences
    if (add_zero_attn_) {
        auto zero_row = Variable(tenzor::zeros({batch_size, 1, embed_dim_},
                                                K.tensor().dtype(), K.tensor().device()), false);
        K = tenzor::cat({K, zero_row}, 1);
        V = tenzor::cat({V, zero_row}, 1);
        seq_len_k += 1;
    }

    // Reshape for multi-head attention
    Q = transpose_for_scores(Q);  // (batch, num_heads, seq_len_q, head_dim)
    K = transpose_for_scores(K);  // (batch, num_heads, seq_len_k, head_dim)
    V = transpose_for_scores(V);  // (batch, num_heads, seq_len_k, head_dim)

    // Prepare attention mask
    Tensor combined_mask;

    if (attn_mask.is_valid() && attn_mask.shape().size() > 0) {
        combined_mask = attn_mask;
    }

    // Add key padding mask if provided
    if (key_padding_mask.is_valid() && key_padding_mask.shape().size() > 0) {
        // key_padding_mask: (batch, seq_len_k)
        // Need to broadcast to (batch, 1, 1, seq_len_k) then to (batch, num_heads, seq_len_q, seq_len_k)
        std::vector<int64_t> mask_shape = {batch_size, 1, 1, seq_len_k};
        Tensor padding_mask = reshape(key_padding_mask, mask_shape);

        // Create -inf tensor for masked positions using device-agnostic tensor ops
        auto pm_shape = std::vector<int64_t>(padding_mask.shape().begin(), padding_mask.shape().end());
        Tensor neg_inf_tensor = full(pm_shape, -std::numeric_limits<float>::infinity(),
                                     padding_mask.dtype(), padding_mask.device());
        Tensor zero_tensor = zeros(pm_shape, padding_mask.dtype(), padding_mask.device());
        Tensor threshold = full(pm_shape, 0.5f, padding_mask.dtype(), padding_mask.device());

        // where(mask > 0.5, -inf, 0.0) — runs on GPU if tensors are on GPU
        Tensor mask_gt = Tensor(gt(padding_mask, threshold));
        neg_inf_tensor = Tensor(where(mask_gt, neg_inf_tensor, zero_tensor));

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
    auto [attended_values, attn_weights] = scaled_dot_product_attention(Q, K, V, combined_mask, dropout_, need_weights, position_bias);

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
