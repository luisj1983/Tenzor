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

// Offsets are CPU index metadata in many call sites, but the nested backend
// kernels read them inside device kernels and dispatch requires one device per
// op — align offsets to the values' device (no-op when already co-located).
static inline Tensor align_offsets(const Tensor& offsets, const Tensor& ref) {
    return (offsets.device().type == ref.device().type) ? offsets : offsets.to(ref.device());
}

auto nested_softmax_forward_impl(const Tensor& values, const Tensor& offsets, int64_t dim) -> Tensor {
    std::vector<Tensor> inputs = {values, align_offsets(offsets, values)};
    OpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    return dispatch<OpId::NestedSoftmax>(inputs, attrs)[0];
}

auto nested_sum_forward_impl(const Tensor& values, const Tensor& offsets) -> Tensor {
    std::vector<Tensor> inputs = {values, align_offsets(offsets, values)};
    return dispatch<OpId::NestedSum>(inputs)[0];
}

auto nested_mean_forward_impl(const Tensor& values, const Tensor& offsets) -> Tensor {
    std::vector<Tensor> inputs = {values, align_offsets(offsets, values)};
    return dispatch<OpId::NestedMean>(inputs)[0];
}

auto nested_layer_norm_forward_impl(const Tensor& values, const Tensor& offsets,
                                     const Tensor& weight, const Tensor& bias,
                                     double eps) -> Tensor {
    std::vector<Tensor> inputs = {values, align_offsets(offsets, values), weight, bias};
    OpAttributes attrs;
    attrs.set(AttrKey::Eps, eps);
    return dispatch<OpId::NestedLayerNorm>(inputs, attrs)[0];
}

auto nested_attention_forward_impl(const Tensor& Q, const Tensor& K, const Tensor& V,
                                    const Tensor& q_offsets, const Tensor& kv_offsets,
                                    double scale, bool causal) -> Tensor {
    // The backend NestedAttention kernel reads the offset arrays inside the
    // device kernel, so they must live on Q's device; dispatch also requires a
    // single device across inputs. Offsets commonly arrive as CPU index
    // metadata — align them to Q's device (no-op when already co-located).
    const Tensor q_off  = (q_offsets.device().type  == Q.device().type) ? q_offsets  : q_offsets.to(Q.device());
    const Tensor kv_off = (kv_offsets.device().type == Q.device().type) ? kv_offsets : kv_offsets.to(Q.device());
    std::vector<Tensor> inputs = {Q, K, V, q_off, kv_off};
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

        // Build the "expanded" [total_len, D] tensor by concatenating each
        // per-segment expanded row block. This is correct on every backend;
        // the previous implementation used std::memcpy on data pointers,
        // which is undefined behavior when the tensors live on GPU memory.
        std::vector<Tensor> parts;
        parts.reserve(B);
        for (int64_t b = 0; b < B; ++b) {
            int64_t start = off[b];
            int64_t end = off[b + 1];
            if (start >= end) continue;
            auto seg_sum = dot_sums.slice(0, b, b + 1);          // [1, D]
            auto seg_expanded = seg_sum.expand({end - start, D}); // view
            parts.push_back(seg_expanded.contiguous());
        }
        Tensor expanded;
        if (parts.empty()) {
            expanded = tenzor::zeros({total_len, D}, grad.dtype(), grad.device());
        } else if (parts.size() == 1 && parts[0].shape()[0] == total_len) {
            expanded = std::move(parts[0]);
        } else {
            expanded = tenzor::cat(std::span<const Tensor>(parts), 0);
        }
        (void)total_len;

        // grad_input = softmax_out * (grad - expanded_dot_sums)
        auto grad_input = tenzor::mul(softmax_out, tenzor::sub(grad, expanded));

        // nested_softmax's set_input_variables({values}) has a single slot.
        // Returning a trailing Tensor() for "offsets" is a mismatch (offsets
        // is a Tensor, not a Variable input). Return exactly one grad.
        return {grad_input};
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
        auto& input_orig = saved[0];
        auto& weight_orig = saved[1];
        [[maybe_unused]] auto& output = saved[2];
        auto& grad_orig = grad_outputs[0];

        // Widen Float16/BFloat16 to Float32 for the whole backward: the LN
        // backward sums seg_grad / seg_grad*normalized into grad_weight/grad_bias
        // across B segments, and a half-precision running accumulator suffers
        // low-precision-accumulator error growth. The per-segment (input-mean)
        // centering also loses subtractive cancellation in half precision.
        // PyTorch accumulates LN parameter grads in fp32; mirror StdBackward's
        // is_half pattern and narrow the results back at the end.
        const DType orig_dtype = input_orig.dtype();
        const bool is_half =
            (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16) ||
            (weight_orig.dtype() == DType::Float16 || weight_orig.dtype() == DType::BFloat16);
        const DType compute_dtype = is_half ? DType::Float32 : orig_dtype;

        const Tensor input  = is_half ? input_orig.to(DType::Float32)  : input_orig;
        const Tensor weight = is_half ? weight_orig.to(DType::Float32) : weight_orig;
        const Tensor grad   = is_half ? grad_orig.to(DType::Float32)   : grad_orig;

        // LayerNorm backward: standard formulas applied per-segment.
        // For simplicity, we unbind by segment and apply the standard LN backward.
        auto offsets_cpu = offsets_.device().type == Device::Type::CPU ? offsets_ : offsets_.to(Device::cpu());
        const int64_t* off = offsets_cpu.data<int64_t>();
        int64_t B = offsets_cpu.numel() - 1;
        int64_t D = (input.shape().size() > 1) ? input.shape()[1] : 1;

        int64_t total_len = (input.shape().size() > 0) ? input.shape()[0] : 0;
        auto grad_weight = tenzor::zeros({D}, compute_dtype, weight.device());
        auto grad_bias = tenzor::zeros({D}, compute_dtype, weight.device());

        // Collect per-segment grad_input rows to cat at the end. Using cat
        // instead of std::memcpy is backend-safe; the previous memcpy approach
        // was undefined behavior on GPU-resident tensors.
        std::vector<Tensor> grad_input_parts;
        grad_input_parts.reserve(B);

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
            auto inv_std = tenzor::reciprocal(tenzor::sqrt(tenzor::add(var, tenzor::full({1, D}, static_cast<double>(eps_), var.dtype(), var.device()))));

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
                    tenzor::full({1}, 1.0 / static_cast<double>(L), var.dtype(), var.device()),
                    tenzor::sub(
                        tenzor::mul(tenzor::full({1}, static_cast<double>(L), var.dtype(), var.device()), dout_w),
                        tenzor::add(sum_dout_w, tenzor::mul(normalized, sum_dout_w_norm))
                    )
                )
            );

            grad_input_parts.push_back(seg_grad_in.contiguous());
        }

        Tensor grad_input;
        if (grad_input_parts.empty()) {
            grad_input = tenzor::zeros_like(input);
        } else if (grad_input_parts.size() == 1 &&
                   grad_input_parts[0].shape()[0] == total_len) {
            grad_input = std::move(grad_input_parts[0]);
        } else {
            grad_input = tenzor::cat(std::span<const Tensor>(grad_input_parts), 0);
        }

        // Narrow the Float32-widened gradients back to the original dtype so the
        // returned grads match the dtype of the Variable inputs.
        if (is_half) {
            grad_input  = grad_input.to(orig_dtype);
            grad_weight = grad_weight.to(weight_orig.dtype());
            grad_bias   = grad_bias.to(weight_orig.dtype());
        }

        // NestedLayerNorm takes 3 Variable inputs (values, weight, bias);
        // offsets is a Tensor passed separately and has no grad slot, so
        // returning grad_input / grad_weight / grad_bias — three entries —
        // matches set_input_variables({values, weight, bias}). Previously a
        // fourth Tensor() was returned for "offsets", causing a mismatch
        // that downstream propagated as "Operation on uninitialized tensor".
        return {grad_input, grad_weight, grad_bias};
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
        auto& Q_orig = saved[0];
        auto& K_orig = saved[1];
        auto& V_orig = saved[2];
        [[maybe_unused]] auto& out = saved[3];
        auto& grad_out_orig = grad_outputs[0];

        // Widen Float16/BFloat16 to Float32 for the recompute: the softmax
        // normalization sum, the d_scores correction sum(d_attn*attn), and the
        // matmul accumulations all lose precision in half precision and risk
        // backend-parity divergence with backends that compute attention in
        // fp32 internally. Compute in Float32 and narrow gQ/gK/gV back.
        const DType orig_dtype = Q_orig.dtype();
        const bool is_half =
            (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16);
        const DType compute_dtype = is_half ? DType::Float32 : orig_dtype;

        const Tensor Q        = is_half ? Q_orig.to(DType::Float32)        : Q_orig;
        const Tensor K        = is_half ? K_orig.to(DType::Float32)        : K_orig;
        const Tensor V        = is_half ? V_orig.to(DType::Float32)        : V_orig;
        const Tensor grad_out = is_half ? grad_out_orig.to(DType::Float32) : grad_out_orig;

        auto q_off_cpu = q_offsets_.device().type == Device::Type::CPU ? q_offsets_ : q_offsets_.to(Device::cpu());
        auto kv_off_cpu = kv_offsets_.device().type == Device::Type::CPU ? kv_offsets_ : kv_offsets_.to(Device::cpu());
        const int64_t* q_off = q_off_cpu.data<int64_t>();
        const int64_t* kv_off = kv_off_cpu.data<int64_t>();
        int64_t B = q_off_cpu.numel() - 1;
        int64_t hd = Q.shape().back();

        int64_t total_q = Q.shape()[0];
        int64_t total_kv = K.shape()[0];

        std::vector<Tensor> gQ_parts, gK_parts, gV_parts;
        gQ_parts.reserve(B); gK_parts.reserve(B); gV_parts.reserve(B);

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
                                       tenzor::full({1}, static_cast<double>(scale_),
                                                     compute_dtype, Q.device()));
            // Causal mask: positions (qi, ki) with ki > qi must contribute
            // softmax weight 0 so the recomputed attention matches the
            // forward path.  Build an additive mask whose upper-triangular
            // (excluding diagonal) is -inf and add it to the scaled scores
            // BEFORE the softmax.  Without this, the backward softmax sees
            // attention weights renormalised over the full key dimension and
            // d_scores leaks gradient onto the masked positions (audit item
            // A.6).
            if (causal_) {
                const int64_t Lq = qe - qs;
                const int64_t Lkv = kve - kvs;
                const float neg_inf = -std::numeric_limits<float>::infinity();
                // Construct -inf above the diagonal, 0 elsewhere directly.
                // (`triu(ones, 1) * -inf` would propagate NaN below the
                // diagonal because IEEE 754 `0 * -inf = NaN`.)
                Tensor neg_inf_mat = tenzor::full(
                    {Lq, Lkv}, neg_inf, compute_dtype, Q.device());
                Tensor mask = tenzor::triu(neg_inf_mat, /*diagonal=*/1);
                scores = tenzor::add(scores, mask);
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
                tenzor::full({1}, static_cast<double>(scale_), compute_dtype, Q.device()));

            // grad_Q = d_scores @ K
            auto gQ = tenzor::matmul(d_scores, Kb);

            // grad_K = d_scores^T @ Q
            auto gK = tenzor::matmul(d_scores.transpose(0, 1), Qb);

            gQ_parts.push_back(gQ.contiguous());
            gK_parts.push_back(gK.contiguous());
            gV_parts.push_back(gV.contiguous());
        }

        auto assemble = [&](std::vector<Tensor>& parts, int64_t total, const Tensor& like)
            -> Tensor {
            if (parts.empty()) return tenzor::zeros_like(like);
            if (parts.size() == 1 && parts[0].shape()[0] == total)
                return std::move(parts[0]);
            return tenzor::cat(std::span<const Tensor>(parts), 0);
        };
        auto grad_Q = assemble(gQ_parts, total_q, Q);
        auto grad_K = assemble(gK_parts, total_kv, K);
        auto grad_V = assemble(gV_parts, total_kv, V);

        // Narrow the Float32-widened gradients back to the original dtype.
        if (is_half) {
            grad_Q = grad_Q.to(orig_dtype);
            grad_K = grad_K.to(orig_dtype);
            grad_V = grad_V.to(orig_dtype);
        }

        // NestedAttention takes 3 Variable inputs (Q, K, V). q_offsets and
        // kv_offsets are Tensors, not Variables, so they have no grad slots.
        return {grad_Q, grad_K, grad_V};
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

        // Scatter grad[b] to every position in segment b via concatenation
        // (device-safe; the previous std::memcpy approach was undefined
        // behavior on GPU-resident tensors).
        std::vector<Tensor> parts;
        parts.reserve(B);
        for (int64_t b = 0; b < B; ++b) {
            int64_t start = off[b];
            int64_t end = off[b + 1];
            if (start >= end) continue;
            parts.push_back(
                grad.slice(0, b, b + 1).expand({end - start, D}).contiguous());
        }
        Tensor grad_input;
        if (parts.empty()) {
            grad_input = tenzor::zeros({total_len_, D}, grad.dtype(), grad.device());
        } else if (parts.size() == 1 && parts[0].shape()[0] == total_len_) {
            grad_input = std::move(parts[0]);
        } else {
            grad_input = tenzor::cat(std::span<const Tensor>(parts), 0);
        }
        // Single Variable input (values); offsets is a Tensor with no grad.
        return {grad_input};
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

        std::vector<Tensor> parts;
        parts.reserve(B);
        for (int64_t b = 0; b < B; ++b) {
            int64_t start = off[b];
            int64_t end = off[b + 1];
            int64_t L = end - start;
            if (L <= 0) continue;
            auto scale = tenzor::full({1}, 1.0 / static_cast<double>(L), grad.dtype(), grad.device());
            parts.push_back(
                tenzor::mul(grad.slice(0, b, b + 1), scale).expand({L, D}).contiguous());
        }
        Tensor grad_input;
        if (parts.empty()) {
            grad_input = tenzor::zeros({total_len_, D}, grad.dtype(), grad.device());
        } else if (parts.size() == 1 && parts[0].shape()[0] == total_len_) {
            grad_input = std::move(parts[0]);
        } else {
            grad_input = tenzor::cat(std::span<const Tensor>(parts), 0);
        }
        // Single Variable input; offsets is a Tensor with no grad.
        return {grad_input};
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
