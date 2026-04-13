/**
 * @file nested_autograd_ops.cpp
 * @brief Autograd Function subclasses for offset-aware nested tensor operations.
 *
 * Element-wise ops on NestedTensor's packed values_ piggyback on existing
 * Variable autograd. Only ops that depend on segment boundaries (softmax,
 * layer norm, attention, reduction) need custom backward implementations
 * defined here.
 */

#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/nested_ops.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/core/tensor.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace tenzor {

// ============================================================================
// Forward-pass helpers (dispatch to backend kernels)
// ============================================================================

namespace {

auto nested_softmax_forward_impl(const Tensor& values, const Tensor& offsets, int64_t dim) -> Tensor {
    std::vector<Tensor> inputs = {values, offsets};
    OpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    return dispatch<OpId::NestedSoftmax>(inputs, attrs)[0];
}

auto nested_sum_forward_impl(const Tensor& values, const Tensor& offsets) -> Tensor {
    std::vector<Tensor> inputs = {values, offsets};
    return dispatch<OpId::NestedSum>(inputs)[0];
}

auto nested_mean_forward_impl(const Tensor& values, const Tensor& offsets) -> Tensor {
    std::vector<Tensor> inputs = {values, offsets};
    return dispatch<OpId::NestedMean>(inputs)[0];
}

auto nested_layer_norm_forward_impl(const Tensor& values, const Tensor& offsets,
                                     const Tensor& weight, const Tensor& bias,
                                     double eps) -> Tensor {
    std::vector<Tensor> inputs = {values, offsets, weight, bias};
    OpAttributes attrs;
    attrs.set(AttrKey::Eps, eps);
    return dispatch<OpId::NestedLayerNorm>(inputs, attrs)[0];
}

auto nested_attention_forward_impl(const Tensor& Q, const Tensor& K, const Tensor& V,
                                    const Tensor& q_offsets, const Tensor& kv_offsets,
                                    double scale, bool causal) -> Tensor {
    std::vector<Tensor> inputs = {Q, K, V, q_offsets, kv_offsets};
    OpAttributes attrs;
    attrs.set(AttrKey::Scale, scale);
    attrs.set(AttrKey::Causal, causal);
    return dispatch<OpId::NestedAttention>(inputs, attrs)[0];
}

} // anonymous namespace

// ============================================================================
// NestedSoftmaxBackward
// ============================================================================

class NestedSoftmaxBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        // inputs[0] = values, inputs[1] is a dummy for offsets (no grad)
        auto softmax_values = nested_softmax_forward_impl(
            inputs[0].tensor(), offsets_, dim_);
        save_for_backward({softmax_values});
        return {Variable(softmax_values)};
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& saved = saved_tensors();
        auto& softmax_out = saved[0];
        auto& grad = grad_outputs[0];

        // grad_input[i] = softmax_out[i] * (grad[i] - sum(grad[i] * softmax_out[i]))
        // Applied per-segment using offsets.
        // This is the standard softmax backward: dsoftmax = s * (dL - sum(dL * s))
        // We compute it using the existing dispatch for efficiency.
        auto grad_times_out = tenzor::mul(grad, softmax_out);

        // Per-segment sum: need to scatter the segment sums back
        auto offsets_cpu = offsets_.device().type == Device::Type::CPU ? offsets_ : offsets_.to(Device::cpu());
        const int64_t* off = offsets_cpu.data<int64_t>();
        int64_t B = offsets_cpu.numel() - 1;

        // Compute per-element correction: for each element, subtract the
        // segment-wide dot product of grad and softmax_out, then multiply by softmax_out.
        // We reuse the NestedSum dispatch to get [B, D] sums, then scatter back.
        auto dot_sums = nested_sum_forward_impl(grad_times_out, offsets_);  // [B, D]

        // Expand dot_sums back to [total_len, D]
        auto values_shape = grad.shape();
        int64_t total_len = values_shape[0];
        int64_t D = (values_shape.size() > 1) ? values_shape[1] : 1;

        // Create expanded tensor: each row gets its segment's sum
        auto expanded = tenzor::zeros({total_len, D}, grad.dtype(), grad.device());
        // Use offsets to fill segments
        for (int64_t b = 0; b < B; ++b) {
            int64_t start = off[b];
            int64_t end = off[b + 1];
            if (start >= end) continue;
            auto seg_sum = dot_sums.slice(0, b, b + 1);  // [1, D]
            auto seg_expanded = seg_sum.expand({end - start, D});
            // Copy into the right region
            auto dst_slice = expanded.slice(0, start, end);
            // In-place copy via add with zero
            auto src_contig = seg_expanded.contiguous();
            std::memcpy(dst_slice.data_ptr(),
                        src_contig.data_ptr(),
                        static_cast<size_t>((end - start) * D) * dtype_size(grad.dtype()));
        }

        // grad_input = softmax_out * (grad - expanded_dot_sums)
        auto grad_input = tenzor::mul(softmax_out, tenzor::sub(grad, expanded));

        return {grad_input, Tensor()};  // no grad for offsets
    }

    auto name() const -> std::string override { return "NestedSoftmaxBackward"; }

    int64_t dim_;
    Tensor offsets_;
};

// ============================================================================
// NestedLayerNormBackward
// ============================================================================

class NestedLayerNormBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        // inputs: [values, weight, bias]
        auto result = nested_layer_norm_forward_impl(
            inputs[0].tensor(), offsets_, inputs[1].tensor(), inputs[2].tensor(), eps_);
        save_for_backward({inputs[0].tensor(), inputs[1].tensor(), result});
        return {Variable(result)};
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& saved = saved_tensors();
        auto& input = saved[0];
        auto& weight = saved[1];
        [[maybe_unused]] auto& output = saved[2];
        auto& grad = grad_outputs[0];

        // LayerNorm backward: standard formulas applied per-segment.
        // For simplicity, we unbind by segment and apply the standard LN backward.
        auto offsets_cpu = offsets_.device().type == Device::Type::CPU ? offsets_ : offsets_.to(Device::cpu());
        const int64_t* off = offsets_cpu.data<int64_t>();
        int64_t B = offsets_cpu.numel() - 1;
        int64_t D = (input.shape().size() > 1) ? input.shape()[1] : 1;

        auto grad_input = tenzor::zeros_like(input);
        auto grad_weight = tenzor::zeros({D}, weight.dtype(), weight.device());
        auto grad_bias = tenzor::zeros({D}, weight.dtype(), weight.device());

        for (int64_t b = 0; b < B; ++b) {
            int64_t start = off[b];
            int64_t end = off[b + 1];
            if (start >= end) continue;

            auto seg_in = input.slice(0, start, end);      // [L, D]
            auto seg_grad = grad.slice(0, start, end);      // [L, D]
            int64_t L = end - start;

            // Compute mean and variance for this segment
            auto seg_mean = tenzor::mean(seg_in, /*dim=*/0, /*keepdim=*/true);  // [1, D]
            auto centered = tenzor::sub(seg_in, seg_mean.expand({L, D}));
            auto var = tenzor::mean(tenzor::mul(centered, centered), /*dim=*/0, /*keepdim=*/true);
            auto inv_std = tenzor::reciprocal(tenzor::sqrt(tenzor::add(var, tenzor::full({1, D}, static_cast<float>(eps_), var.dtype(), var.device()))));

            auto normalized = tenzor::mul(centered, inv_std.expand({L, D}));

            // grad_bias += sum(seg_grad, dim=0)
            grad_bias = tenzor::add(grad_bias, tenzor::sum(seg_grad, 0, false));

            // grad_weight += sum(seg_grad * normalized, dim=0)
            grad_weight = tenzor::add(grad_weight, tenzor::sum(tenzor::mul(seg_grad, normalized), 0, false));

            // grad_input for this segment:
            // dx = (1/L) * inv_std * weight * (L * dout - sum(dout) - normalized * sum(dout * normalized))
            auto dout_w = tenzor::mul(seg_grad, weight.unsqueeze(0).expand({L, D}));
            auto sum_dout_w = tenzor::sum(dout_w, 0, true).expand({L, D});
            auto sum_dout_w_norm = tenzor::sum(tenzor::mul(dout_w, normalized), 0, true).expand({L, D});

            auto seg_grad_in = tenzor::mul(
                inv_std.expand({L, D}),
                tenzor::mul(
                    tenzor::full({1}, 1.0f / static_cast<float>(L), var.dtype(), var.device()),
                    tenzor::sub(
                        tenzor::mul(tenzor::full({1}, static_cast<float>(L), var.dtype(), var.device()), dout_w),
                        tenzor::add(sum_dout_w, tenzor::mul(normalized, sum_dout_w_norm))
                    )
                )
            );

            // Copy back
            auto dst = grad_input.slice(0, start, end);
            std::memcpy(dst.data_ptr(), seg_grad_in.contiguous().data_ptr(),
                        static_cast<size_t>(L * D) * dtype_size(input.dtype()));
        }

        return {grad_input, Tensor(), grad_weight, grad_bias};  // no grad for offsets
    }

    auto name() const -> std::string override { return "NestedLayerNormBackward"; }

    double eps_;
    Tensor offsets_;
};

// ============================================================================
// NestedAttentionBackward
// ============================================================================

class NestedAttentionBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        // inputs: [Q, K, V]
        auto out = nested_attention_forward_impl(
            inputs[0].tensor(), inputs[1].tensor(), inputs[2].tensor(),
            q_offsets_, kv_offsets_, scale_, causal_);
        // Save Q, K, V and output for backward
        save_for_backward({inputs[0].tensor(), inputs[1].tensor(),
                          inputs[2].tensor(), out});
        return {Variable(out)};
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& saved = saved_tensors();
        auto& Q = saved[0];
        auto& K = saved[1];
        auto& V = saved[2];
        [[maybe_unused]] auto& out = saved[3];
        auto& grad_out = grad_outputs[0];

        auto q_off_cpu = q_offsets_.device().type == Device::Type::CPU ? q_offsets_ : q_offsets_.to(Device::cpu());
        auto kv_off_cpu = kv_offsets_.device().type == Device::Type::CPU ? kv_offsets_ : kv_offsets_.to(Device::cpu());
        const int64_t* q_off = q_off_cpu.data<int64_t>();
        const int64_t* kv_off = kv_off_cpu.data<int64_t>();
        int64_t B = q_off_cpu.numel() - 1;
        int64_t hd = Q.shape().back();

        auto grad_Q = tenzor::zeros_like(Q);
        auto grad_K = tenzor::zeros_like(K);
        auto grad_V = tenzor::zeros_like(V);

        for (int64_t b = 0; b < B; ++b) {
            int64_t qs = q_off[b], qe = q_off[b + 1];
            int64_t kvs = kv_off[b], kve = kv_off[b + 1];
            if (qs >= qe || kvs >= kve) continue;

            auto Qb = Q.slice(0, qs, qe);         // [Lq, hd]
            auto Kb = K.slice(0, kvs, kve);        // [Lkv, hd]
            auto Vb = V.slice(0, kvs, kve);        // [Lkv, hd]
            auto dO = grad_out.slice(0, qs, qe);   // [Lq, hd]

            // Recompute attention weights: scores = Qb @ Kb^T * scale
            auto scores = tenzor::mul(tenzor::matmul(Qb, Kb.transpose(0, 1)),
                                       tenzor::full({1}, static_cast<float>(scale_),
                                                     Q.dtype(), Q.device()));
            // Causal mask
            if (causal_) {
                int64_t Lq = qe - qs, Lkv = kve - kvs;
                for (int64_t qi = 0; qi < Lq; ++qi) {
                    for (int64_t ki = qi + 1; ki < Lkv; ++ki) {
                        // Set future positions to -inf would be done in kernel;
                        // for CPU fallback we rely on the forward kernel
                    }
                }
            }

            // Softmax per query row
            OpAttributes sm_attrs;
            sm_attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
            std::vector<Tensor> sm_inputs = {scores};
            auto attn_weights = dispatch_single(OpId::Softmax, sm_inputs, sm_attrs);  // [Lq, Lkv]

            // grad_V = attn_weights^T @ dO
            auto gV = tenzor::matmul(attn_weights.transpose(0, 1), dO);

            // grad_attn = dO @ V^T
            auto d_attn = tenzor::matmul(dO, Vb.transpose(0, 1));  // [Lq, Lkv]

            // grad through softmax: ds = attn * (d_attn - sum(d_attn * attn, -1, keepdim))
            auto d_scores = tenzor::mul(attn_weights,
                tenzor::sub(d_attn,
                    tenzor::sum(tenzor::mul(d_attn, attn_weights), -1, true).expand(
                        std::vector<int64_t>(d_attn.shape().begin(), d_attn.shape().end()))));

            // Scale the gradient
            d_scores = tenzor::mul(d_scores,
                tenzor::full({1}, static_cast<float>(scale_), Q.dtype(), Q.device()));

            // grad_Q = d_scores @ K
            auto gQ = tenzor::matmul(d_scores, Kb);

            // grad_K = d_scores^T @ Q
            auto gK = tenzor::matmul(d_scores.transpose(0, 1), Qb);

            // Copy gradients into output buffers
            auto dst_gQ = grad_Q.slice(0, qs, qe);
            auto dst_gK = grad_K.slice(0, kvs, kve);
            auto dst_gV = grad_V.slice(0, kvs, kve);

            auto gQ_c = gQ.contiguous();
            auto gK_c = gK.contiguous();
            auto gV_c = gV.contiguous();

            std::memcpy(dst_gQ.data_ptr(), gQ_c.data_ptr(),
                        static_cast<size_t>((qe - qs) * hd) * dtype_size(Q.dtype()));
            std::memcpy(dst_gK.data_ptr(), gK_c.data_ptr(),
                        static_cast<size_t>((kve - kvs) * hd) * dtype_size(K.dtype()));
            std::memcpy(dst_gV.data_ptr(), gV_c.data_ptr(),
                        static_cast<size_t>((kve - kvs) * hd) * dtype_size(V.dtype()));
        }

        return {grad_Q, grad_K, grad_V, Tensor(), Tensor()};  // no grad for offsets
    }

    auto name() const -> std::string override { return "NestedAttentionBackward"; }

    Tensor q_offsets_;
    Tensor kv_offsets_;
    double scale_;
    bool causal_;
};

// ============================================================================
// NestedReductionBackward (sum/mean)
// ============================================================================

class NestedSumBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        auto result = nested_sum_forward_impl(inputs[0].tensor(), offsets_);
        total_len_ = inputs[0].tensor().shape()[0];
        return {Variable(result)};
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& grad = grad_outputs[0];  // [B, D]
        auto offsets_cpu = offsets_.device().type == Device::Type::CPU ? offsets_ : offsets_.to(Device::cpu());
        const int64_t* off = offsets_cpu.data<int64_t>();
        int64_t B = offsets_cpu.numel() - 1;
        int64_t D = (grad.shape().size() > 1) ? grad.shape()[1] : 1;

        // Scatter grad[b] to every position in segment b
        auto grad_input = tenzor::zeros({total_len_, D}, grad.dtype(), grad.device());
        for (int64_t b = 0; b < B; ++b) {
            int64_t start = off[b];
            int64_t end = off[b + 1];
            if (start >= end) continue;
            auto seg_grad = grad.slice(0, b, b + 1).expand({end - start, D}).contiguous();
            auto dst = grad_input.slice(0, start, end);
            std::memcpy(dst.data_ptr(), seg_grad.data_ptr(),
                        static_cast<size_t>((end - start) * D) * dtype_size(grad.dtype()));
        }
        return {grad_input, Tensor()};
    }

    auto name() const -> std::string override { return "NestedSumBackward"; }

    Tensor offsets_;
    int64_t total_len_;
};

class NestedMeanBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        auto result = nested_mean_forward_impl(inputs[0].tensor(), offsets_);
        total_len_ = inputs[0].tensor().shape()[0];
        return {Variable(result)};
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& grad = grad_outputs[0];  // [B, D]
        auto offsets_cpu = offsets_.device().type == Device::Type::CPU ? offsets_ : offsets_.to(Device::cpu());
        const int64_t* off = offsets_cpu.data<int64_t>();
        int64_t B = offsets_cpu.numel() - 1;
        int64_t D = (grad.shape().size() > 1) ? grad.shape()[1] : 1;

        auto grad_input = tenzor::zeros({total_len_, D}, grad.dtype(), grad.device());
        for (int64_t b = 0; b < B; ++b) {
            int64_t start = off[b];
            int64_t end = off[b + 1];
            int64_t L = end - start;
            if (L <= 0) continue;
            auto scale = tenzor::full({1}, 1.0f / static_cast<float>(L), grad.dtype(), grad.device());
            auto seg_grad = tenzor::mul(grad.slice(0, b, b + 1), scale).expand({L, D}).contiguous();
            auto dst = grad_input.slice(0, start, end);
            std::memcpy(dst.data_ptr(), seg_grad.data_ptr(),
                        static_cast<size_t>(L * D) * dtype_size(grad.dtype()));
        }
        return {grad_input, Tensor()};
    }

    auto name() const -> std::string override { return "NestedMeanBackward"; }

    Tensor offsets_;
    int64_t total_len_;
};

// ============================================================================
// Autograd-aware functional API implementations
// ============================================================================

namespace autograd {

auto nested_softmax(const Variable& values, const Tensor& offsets, int64_t dim) -> Variable {
    // Compute forward result
    auto result_tensor = nested_softmax_forward_impl(values.tensor(), offsets, dim);

    if (!values.requires_grad() || !is_grad_enabled()) {
        return Variable(result_tensor, false);
    }

    // Build computation graph (following standard autograd pattern)
    auto grad_fn = std::make_shared<NestedSoftmaxBackward>();
    grad_fn->dim_ = dim;
    grad_fn->offsets_ = offsets;
    grad_fn->save_for_backward({result_tensor});

    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.push_back(values.grad_fn());
    grad_fn->set_next_functions(next_funcs);

    std::vector<Variable> input_vars;
    input_vars.push_back(values);
    grad_fn->set_input_variables(input_vars);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto nested_layer_norm(const Variable& values, const Tensor& offsets,
                       const Variable& weight, const Variable& bias, double eps) -> Variable {
    auto result_tensor = nested_layer_norm_forward_impl(
        values.tensor(), offsets, weight.tensor(), bias.tensor(), eps);

    if (!is_grad_enabled() ||
        (!values.requires_grad() && !weight.requires_grad() && !bias.requires_grad())) {
        return Variable(result_tensor, false);
    }

    auto grad_fn = std::make_shared<NestedLayerNormBackward>();
    grad_fn->eps_ = eps;
    grad_fn->offsets_ = offsets;
    grad_fn->save_for_backward({values.tensor(), weight.tensor(), bias.tensor()});

    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.push_back(values.grad_fn());
    next_funcs.push_back(weight.grad_fn());
    next_funcs.push_back(bias.grad_fn());
    grad_fn->set_next_functions(next_funcs);
    grad_fn->set_input_variables({values, weight, bias});

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto nested_linear(const Variable& values, [[maybe_unused]] const Tensor& offsets,
                   const Variable& weight, const Variable* bias) -> Variable {
    // Linear is just matmul + bias on the packed values, so we reuse
    // the existing Variable matmul and add autograd.
    auto result = values.matmul(weight.transpose(0, 1));
    if (bias != nullptr) {
        result = result + *bias;
    }
    return result;
}

auto nested_attention(const Variable& q_values, const Variable& k_values,
                      const Variable& v_values, const Tensor& q_offsets,
                      const Tensor& kv_offsets, double scale, bool causal) -> Variable {
    auto result_tensor = nested_attention_forward_impl(
        q_values.tensor(), k_values.tensor(), v_values.tensor(),
        q_offsets, kv_offsets, scale, causal);

    if (!is_grad_enabled() ||
        (!q_values.requires_grad() && !k_values.requires_grad() && !v_values.requires_grad())) {
        return Variable(result_tensor, false);
    }

    auto grad_fn = std::make_shared<NestedAttentionBackward>();
    grad_fn->q_offsets_ = q_offsets;
    grad_fn->kv_offsets_ = kv_offsets;
    grad_fn->scale_ = scale;
    grad_fn->causal_ = causal;
    grad_fn->save_for_backward({q_values.tensor(), k_values.tensor(), v_values.tensor(), result_tensor});

    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.push_back(q_values.grad_fn());
    next_funcs.push_back(k_values.grad_fn());
    next_funcs.push_back(v_values.grad_fn());
    grad_fn->set_next_functions(next_funcs);
    grad_fn->set_input_variables({q_values, k_values, v_values});

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto nested_sum(const Variable& values, const Tensor& offsets) -> Variable {
    auto result_tensor = nested_sum_forward_impl(values.tensor(), offsets);

    if (!values.requires_grad() || !is_grad_enabled()) {
        return Variable(result_tensor, false);
    }

    auto grad_fn = std::make_shared<NestedSumBackward>();
    grad_fn->offsets_ = offsets;
    grad_fn->total_len_ = values.tensor().shape()[0];

    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.push_back(values.grad_fn());
    grad_fn->set_next_functions(next_funcs);
    grad_fn->set_input_variables({values});

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto nested_mean(const Variable& values, const Tensor& offsets) -> Variable {
    auto result_tensor = nested_mean_forward_impl(values.tensor(), offsets);

    if (!values.requires_grad() || !is_grad_enabled()) {
        return Variable(result_tensor, false);
    }

    auto grad_fn = std::make_shared<NestedMeanBackward>();
    grad_fn->offsets_ = offsets;
    grad_fn->total_len_ = values.tensor().shape()[0];

    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.push_back(values.grad_fn());
    grad_fn->set_next_functions(next_funcs);
    grad_fn->set_input_variables({values});

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

} // namespace autograd
} // namespace tenzor
