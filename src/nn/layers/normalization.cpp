#include "tenzor/nn/layers/normalization.hpp"
#include "tenzor/nn/utils/variable_cast.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/jit/tracer.hpp"
#include "tenzor/utils/autograd_wrap.hpp"

namespace {

/// Record an `RMSNorm` op into the active JIT trace, if any. The CPU and
/// SIMD fast paths in `RMSNorm::forward_impl` bypass the kernel
/// dispatcher entirely (they write straight into a pre-allocated output
/// buffer), so the tracing interceptor never sees the op. Calling this
/// from inside the forward gives the JIT lowering the (x, weight) ->
/// output edge it needs to emit the StableHLO `rms_norm` expansion.
inline auto jit_record_rms_norm(const ::tenzor::Tensor& x,
                                const ::tenzor::Tensor& weight,
                                const ::tenzor::Tensor& output,
                                double eps,
                                int64_t normalized_shape) -> void {
    auto& tracer = ::tenzor::jit::Tracer::get_instance();
    if (!tracer.is_tracing()) return;
    auto x_id = tracer.register_tensor(x);
    auto w_id = tracer.register_tensor(weight);
    auto o_id = tracer.register_new_tensor(output);
    ::tenzor::jit::TracedOp op(::tenzor::jit::OpType::RMSNorm,
                               {x_id, w_id}, {o_id});
    op.attrs["eps"] = static_cast<float>(eps);
    op.vec_attrs["normalized_shape"] = {normalized_shape};
    tracer.record_op(std::move(op));
}

// BB.16: V.24/Y.20 added per-call-site guards to refuse backend kernels
// that regress to returning only {output} when the layer-side backward
// needs results[1] / results[2]. The guard was duplicated at five sites
// (LayerNorm, GroupNorm, InstanceNorm2d, InstanceNorm1d, RMSNorm GPU)
// and missing at others (the RMSNorm CPU path returns {output, rrms}
// directly without dispatch, but any future norm forward dispatch
// will need the same check). Extract into one helper so adding a new
// site is a one-line call. min_required defaults to 3 — the
// {output, mean/sum-stat, rstd/rrms} contract shared by LayerNorm,
// GroupNorm, InstanceNorm; pass 2 for the {output, rrms} RMSNorm case.
inline auto check_norm_outputs(const std::vector<::tenzor::Tensor>& results,
                               const char* op_name,
                               std::size_t min_required = 3) -> void {
    if (results.size() < min_required) {
        throw std::runtime_error(
            std::string(op_name) +
            ": backend kernel returned " +
            std::to_string(results.size()) +
            " tensors; the contract requires at least " +
            std::to_string(min_required) +
            " ({output, mean/stat, rstd/rrms}). Fix the backend kernel "
            "registration.");
    }
}

}  // namespace
#include <cmath>
#include <memory>
#include <stdexcept>

// SIMD headers for optimized LayerNorm
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

// OpenMP for parallel execution
#ifdef _OPENMP
#include <omp.h>
#include <thread>
#include <fstream>
#include <string>
#endif

// Get optimal thread count for compute-bound operations
// Uses physical cores (not hyperthreaded) to avoid contention
static inline int get_optimal_threads() {
#ifdef _OPENMP
    static int optimal = []() {
        unsigned int logical_cores = std::thread::hardware_concurrency();
        unsigned int physical_cores = logical_cores;

        // Detect physical cores via Linux sysfs
        std::ifstream siblings("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list");
        if (siblings.good()) {
            std::string line;
            if (std::getline(siblings, line)) {
                int threads_per_core = 1;
                for (char c : line) {
                    if (c == ',') threads_per_core++;
                }
                if (threads_per_core > 1) {
                    physical_cores = logical_cores / threads_per_core;
                }
            }
        }

        int num_threads = std::max(1u, physical_cores);
        omp_set_num_threads(num_threads);
        return num_threads;
    }();
    return optimal;
#else
    return 1;
#endif
}

namespace tenzor::nn {

// Thread count for the memory-bound normalization kernels (LayerNorm / RMSNorm).
// These are bandwidth-limited: aggregate memory bandwidth scales with the number
// of cores engaging memory channels, so PyTorch threads them across all cores and
// so do we. Default to the full logical-core count; `TENZOR_NORM_THREADS` allows
// overriding for tuning experiments.
static inline int norm_kernel_threads() {
    static const int n = []() {
        if (const char* env = std::getenv("TENZOR_NORM_THREADS")) {
            int v = std::atoi(env);
            if (v > 0) return v;
        }
        return static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    }();
    return n;
}

// ============================================================================
// SIMD-Optimized Fused RMSNorm (simpler than LayerNorm - no mean computation)
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64)
static void fused_rms_norm_f32(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    float* __restrict__ rrms_out,
    int64_t batch_size,
    int64_t norm_size,
    float eps
) {
    const float inv_n = 1.0f / static_cast<float>(norm_size);

    // Memory-bound: aggregate bandwidth scales with cores (more memory channels
    // engaged), so use the shared norm thread pool like PyTorch rather than the
    // old hard cap of 4 — that cap left common transformer shapes ~2x slower.
    // Bound by the number of rows, and skip threading for tiny inputs where
    // fork-join overhead would dominate.
    const int64_t total_elements = batch_size * norm_size;
    const int final_threads = (total_elements > 65536 && batch_size > 1)
        ? std::max(1, std::min(norm_kernel_threads(), static_cast<int>(batch_size)))
        : 1;
    #pragma omp parallel for if(final_threads > 1) num_threads(final_threads)
    for (int64_t b = 0; b < batch_size; b++) {
        const float* in_ptr = input + b * norm_size;
        float* out_ptr = output + b * norm_size;

#ifdef __AVX512F__
        // AVX-512 path: 16 floats at a time
        // First pass: compute sum of squares
        __m512 sum_sq_vec = _mm512_setzero_ps();
        int64_t i = 0;

        for (; i + 16 <= norm_size; i += 16) {
            _mm_prefetch(reinterpret_cast<const char*>(in_ptr + i + 64), _MM_HINT_T0);
            __m512 x = _mm512_loadu_ps(in_ptr + i);
            sum_sq_vec = _mm512_fmadd_ps(x, x, sum_sq_vec);
        }

        float sum_sq = _mm512_reduce_add_ps(sum_sq_vec);

        // Handle remainder
        for (; i < norm_size; i++) {
            sum_sq += in_ptr[i] * in_ptr[i];
        }

        float rms = std::sqrt(sum_sq * inv_n + eps);
        float rrms = 1.0f / rms;
        rrms_out[b] = rrms;

        // Second pass: normalize and apply weight
        __m512 rrms_v = _mm512_set1_ps(rrms);
        i = 0;

        for (; i + 16 <= norm_size; i += 16) {
            _mm_prefetch(reinterpret_cast<const char*>(in_ptr + i + 64), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(weight + i + 64), _MM_HINT_T0);
            __m512 x = _mm512_loadu_ps(in_ptr + i);
            __m512 w = _mm512_loadu_ps(weight + i);
            __m512 result = _mm512_mul_ps(_mm512_mul_ps(x, rrms_v), w);
            _mm512_storeu_ps(out_ptr + i, result);
        }

        // Handle remainder
        for (; i < norm_size; i++) {
            out_ptr[i] = in_ptr[i] * rrms * weight[i];
        }

#elif defined(__AVX2__)
        // AVX2 path: 8 floats at a time
        __m256 sum_sq_vec = _mm256_setzero_ps();
        int64_t i = 0;

        // First pass: compute sum of squares
        for (; i + 8 <= norm_size; i += 8) {
            _mm_prefetch(reinterpret_cast<const char*>(in_ptr + i + 32), _MM_HINT_T0);
            __m256 x = _mm256_loadu_ps(in_ptr + i);
            sum_sq_vec = _mm256_fmadd_ps(x, x, sum_sq_vec);
        }

        // Horizontal sum for AVX2
        __m128 sum_lo = _mm256_castps256_ps128(sum_sq_vec);
        __m128 sum_hi = _mm256_extractf128_ps(sum_sq_vec, 1);
        __m128 sum128 = _mm_add_ps(sum_lo, sum_hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float sum_sq = _mm_cvtss_f32(sum128);

        // Handle remainder
        for (; i < norm_size; i++) {
            sum_sq += in_ptr[i] * in_ptr[i];
        }

        float rms = std::sqrt(sum_sq * inv_n + eps);
        float rrms = 1.0f / rms;
        rrms_out[b] = rrms;

        // Second pass: normalize and apply weight
        __m256 rrms_v = _mm256_set1_ps(rrms);
        i = 0;

        for (; i + 8 <= norm_size; i += 8) {
            _mm_prefetch(reinterpret_cast<const char*>(in_ptr + i + 32), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(weight + i + 32), _MM_HINT_T0);
            __m256 x = _mm256_loadu_ps(in_ptr + i);
            __m256 w = _mm256_loadu_ps(weight + i);
            __m256 result = _mm256_mul_ps(_mm256_mul_ps(x, rrms_v), w);
            _mm256_storeu_ps(out_ptr + i, result);
        }

        // Handle remainder
        for (; i < norm_size; i++) {
            out_ptr[i] = in_ptr[i] * rrms * weight[i];
        }

#else
        // Scalar fallback
        float sum_sq = 0.0f;
        for (int64_t i = 0; i < norm_size; i++) {
            sum_sq += in_ptr[i] * in_ptr[i];
        }

        float rms = std::sqrt(sum_sq * inv_n + eps);
        float rrms = 1.0f / rms;
        rrms_out[b] = rrms;

        for (int64_t i = 0; i < norm_size; i++) {
            out_ptr[i] = in_ptr[i] * rrms * weight[i];
        }
#endif
    }
}
#endif // x86_64

// ============================================================================
// LayerNorm Implementation
// ============================================================================

// LayerNorm autograd function
class LayerNormBackward : public Function {
public:
    LayerNormBackward(bool elementwise_affine, double eps,
                      int64_t normalized_size, std::vector<Tensor> tensors_to_save)
        : elementwise_affine_(elementwise_affine), eps_(eps),
          normalized_size_(normalized_size) {
        save_for_backward(std::move(tensors_to_save));
    }

    LayerNormBackward(bool elementwise_affine, double eps,
                      int64_t normalized_size,
                      std::vector<int64_t> normalized_shape,
                      std::vector<Tensor> tensors_to_save)
        : elementwise_affine_(elementwise_affine), eps_(eps),
          normalized_size_(normalized_size),
          normalized_shape_(std::move(normalized_shape)) {
        save_for_backward(std::move(tensors_to_save));
    }

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("LayerNormBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;

    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return false; }

    // A.4 multi-op JVP walker hooks ---------------------------------------
    auto op_id() const -> OpId override { return OpId::LayerNorm; }

    auto saved_attributes() const -> OpAttributes override {
        OpAttributes attrs;
        attrs.set(AttrKey::Eps, eps_);
        if (!normalized_shape_.empty()) {
            std::string s;
            for (size_t i = 0; i < normalized_shape_.size(); ++i) {
                if (i > 0) s += ",";
                s += std::to_string(normalized_shape_[i]);
            }
            attrs.set(AttrKey::NormalizedShape, std::string_view(s));
        }
        return attrs;
    }

    auto jvp_pack_inputs_for_walker(
            const std::vector<Tensor>& walker_primals,
            const std::vector<Tensor>& walker_tangents) const
        -> std::optional<std::pair<std::vector<Tensor>, std::vector<Tensor>>> override {
        if (walker_primals.empty()) return std::nullopt;
        if (num_saved_tensors() < 4) return std::nullopt;
        const auto& sav = const_cast<LayerNormBackward*>(this)->saved_tensors();
        const Tensor& gamma = sav[3];

        std::vector<int64_t> g_shape(gamma.shape().begin(), gamma.shape().end());
        Tensor beta  = tenzor::zeros(g_shape, gamma.dtype(), gamma.device());
        Tensor dgamma = tenzor::zeros(g_shape, gamma.dtype(), gamma.device());
        Tensor dbeta  = tenzor::zeros(g_shape, gamma.dtype(), gamma.device());

        std::vector<Tensor> primals  = { walker_primals[0],  gamma,  beta  };
        std::vector<Tensor> tangents = { walker_tangents[0], dgamma, dbeta };
        return std::make_pair(std::move(primals), std::move(tangents));
    }

    // LayerNormBackward saves {x, mean, rstd, gamma}. The forward
    // OpId::LayerNorm produces outputs {y, mean, rstd}, so saved[1] → out 1
    // (mean) and saved[2] → out 2 (rstd). saved[0] (x) and saved[3] (gamma)
    // are inputs, not outputs — they don't correspond to any forward output
    // slot and should be left at the default of 0 (which won't be hit in
    // practice since consumers don't read x/gamma from a LayerNorm node).
    auto jvp_saved_tensor_to_output_idx(std::size_t saved_idx) const
        -> std::size_t override {
        switch (saved_idx) {
            case 1: return 1;  // mean
            case 2: return 2;  // rstd
            default: return 0;
        }
    }

private:
    bool elementwise_affine_;
    double eps_;
    int64_t normalized_size_;
    std::vector<int64_t> normalized_shape_;
};

auto LayerNormBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    auto& grad_output_orig = grad_outputs[0];
    auto saved = saved_tensors();
    auto& input_orig = saved[0];
    auto& mean_orig = saved[1];
    auto& rstd_orig = saved[2];  // reciprocal std (1 / sqrt(var + eps))
    auto& weight_orig = saved[3];

    Device original_device = input_orig.device();
    DType original_dtype = grad_output_orig.dtype();

    // GPU fast path
    if (original_device.type != Device::Type::CPU) {
        auto go = grad_output_orig.contiguous();
        auto inp = input_orig.contiguous();
        auto mn = mean_orig.contiguous();
        auto rs = rstd_orig.contiguous();
        auto wt = weight_orig.contiguous();

        DType orig_dt = inp.dtype();
        bool needs_upcast = (orig_dt == DType::Float16 || orig_dt == DType::BFloat16);
        if (needs_upcast) {
            go = go.to(DType::Float32);
            inp = inp.to(DType::Float32);
            mn = mn.to(DType::Float32);
            rs = rs.to(DType::Float32);
            wt = wt.to(DType::Float32);
        } else {
            if (go.dtype() != inp.dtype()) go = go.to(inp.dtype());
            if (mn.dtype() != inp.dtype()) mn = mn.to(inp.dtype());
            if (rs.dtype() != inp.dtype()) rs = rs.to(inp.dtype());
            if (wt.dtype() != inp.dtype()) wt = wt.to(inp.dtype());
        }

        NewOpAttributes attrs;
        attrs.set(AttrKey::NormalizedShape, std::to_string(normalized_size_));
        std::vector<Tensor> inputs_vec = {go, inp, mn, rs, wt};
        auto results = dispatch<OpId::LayerNormBackward>(inputs_vec, attrs);

        if (needs_upcast && results.size() >= 1) {
            results[0] = results[0].to(orig_dt);
        }
        return results;
    }

    // CPU path
    auto grad_output = grad_output_orig.contiguous().to(DType::Float32);
    auto input = input_orig.contiguous().to(DType::Float32);
    auto mean = mean_orig.contiguous().to(DType::Float32);
    auto rstd = rstd_orig.contiguous().to(DType::Float32);
    auto weight = weight_orig.contiguous().to(DType::Float32);

    auto shape = input.shape();
    int64_t batch_size = 1;
    for (size_t i = 0; i < shape.size() - 1; i++) {
        batch_size *= shape[i];
    }

    int64_t N = normalized_size_;

    auto* input_data = input.data<float>();
    auto* mean_data = mean.data<float>();
    auto* rstd_data = rstd.data<float>();
    auto* grad_out_data = grad_output.data<float>();
    auto* weight_data = weight.data<float>();

    auto grad_input = zeros_like(input);
    auto grad_weight = zeros({N}, grad_output.dtype(), grad_output.device());
    auto grad_bias = zeros({N}, grad_output.dtype(), grad_output.device());

    auto* grad_in_data = grad_input.data<float>();
    auto* grad_weight_data = grad_weight.data<float>();
    auto* grad_bias_data = grad_bias.data<float>();

    for (int64_t b = 0; b < batch_size; b++) {
        float mu = mean_data[b];
        float inv_std = rstd_data[b];

        // NOTE: do NOT short-circuit on low variance. rstd is the saved
        // 1/sqrt(var+eps) and is therefore always finite (eps guards the
        // sqrt), so the analytic gradient
        //   (grad_out*w - mean_grad_out - x_hat*mean_grad_out_normalized)*rstd
        // is well-defined and generally non-zero even when var < eps (e.g.
        // constant or heavily down-scaled rows). Zeroing grad_input on such
        // rows produced a wrong gradient that diverged from both the GPU
        // kernels and the higher-order (create_graph=true) CPU path.

        // S.15: promote reduction accumulators to double to match the
        // double-precision variance computation above. Mixing float reductions
        // with a double mean/var gives 4–7% drift on F32 LayerNorm gradchecks.
        double sum_grad_out = 0.0;
        double sum_grad_out_normalized = 0.0;

        for (int64_t i = 0; i < N; i++) {
            int64_t idx = b * N + i;
            float x_normalized = (input_data[idx] - mu) * inv_std;
            float grad_out = grad_out_data[idx] * weight_data[i];

            sum_grad_out += static_cast<double>(grad_out);
            sum_grad_out_normalized += static_cast<double>(grad_out) *
                                       static_cast<double>(x_normalized);

            grad_weight_data[i] += grad_out_data[idx] * x_normalized;
            grad_bias_data[i] += grad_out_data[idx];
        }

        const double mean_grad_out = sum_grad_out / static_cast<double>(N);
        const double mean_grad_out_normalized =
            sum_grad_out_normalized / static_cast<double>(N);

        for (int64_t i = 0; i < N; i++) {
            int64_t idx = b * N + i;
            const double x_normalized =
                (static_cast<double>(input_data[idx]) - static_cast<double>(mu)) *
                static_cast<double>(inv_std);

            const double grad_out_w =
                static_cast<double>(grad_out_data[idx]) *
                static_cast<double>(weight_data[i]);
            const double gi = (grad_out_w - mean_grad_out -
                               x_normalized * mean_grad_out_normalized) *
                              static_cast<double>(inv_std);
            grad_in_data[idx] = static_cast<float>(gi);
        }
    }

    // grad_weight/grad_bias are accumulated flat as [N]. When normalized_shape is
    // multi-dimensional the weight/bias parameters are multi-dimensional too, so
    // reshape the grads to the parameter shape — otherwise gradient accumulation
    // shape-mismatches and crashes. No-op for the common flat / affine-off case.
    Tensor gw = grad_weight.contiguous();
    Tensor gb = grad_bias.contiguous();
    if (weight.ndim() > 1 && weight.numel() == N) {
        std::vector<int64_t> param_shape(weight.shape().begin(), weight.shape().end());
        gw = gw.reshape(param_shape);
        gb = gb.reshape(param_shape);
    }
    return {grad_input.contiguous().to(original_dtype),
            gw.to(original_dtype),
            gb.to(original_dtype)};
}

// V.24: slot-3 contract for LayerNormBackward's saved buffers.
//
// LayerNormBackward saves {x, mean, rstd, gamma?} where the trailing gamma
// slot is *only* populated when elementwise_affine_ is true.  Today the
// forward always saves a placeholder ones-tensor at slot 3 so the size is
// always 4, but that placeholder costs a tensor allocation per forward and
// will be removed in a future optimisation.  Reading saved[3] / sv[3]
// unconditionally would then be an out-of-bounds access.  All consumers
// must gate slot-3 reads on `elementwise_affine_` (also true of the
// JVP walker hook above, which already checks `num_saved_tensors() < 4`).
auto LayerNormBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto& grad_out = grad_outputs[0];

    Variable input_var, mean_var, rstd_var, weight_var;
    if (has_saved_variables()) {
        const auto& sv = saved_variables();
        input_var = sv[0];
        mean_var = sv[1];
        rstd_var = sv[2];
        // V.24: gate slot-3 read on elementwise_affine_ — the slot is only
        // contractually populated when affine is true.
        if (elementwise_affine_) {
            weight_var = sv[3];
        }
    } else {
        auto saved = saved_tensors();
        input_var = Variable(saved[0], false);
        mean_var = Variable(saved[1], false);
        rstd_var = Variable(saved[2], false);
        if (elementwise_affine_) {
            weight_var = Variable(saved[3], false);
        }
    }

    int64_t norm_size = normalized_size_;

    auto mean_bc = unsqueeze(mean_var, -1);
    auto rstd_bc = unsqueeze(rstd_var, -1);

    auto x_hat = (input_var - mean_bc) * rstd_bc;

    // S.18: when elementwise_affine_ is false, weight is a placeholder ones
    // tensor — skip multiplying by it and short-circuit weight/bias grads.
    Variable grad_x_hat;
    if (elementwise_affine_) {
        grad_x_hat = grad_out * weight_var;
    } else {
        grad_x_hat = grad_out;
    }

    auto mean_gxh = sum(grad_x_hat, -1, true) / static_cast<float>(norm_size);

    auto mean_gxh_xh = sum(grad_x_hat * x_hat, -1, true) / static_cast<float>(norm_size);

    auto grad_input = (grad_x_hat - mean_gxh - x_hat * mean_gxh_xh) * rstd_bc;

    if (!elementwise_affine_) {
        // No weight/bias parameters — the autograd framework expects a
        // single grad-input slot when affine is off.
        return {grad_input};
    }

    auto go_xhat = grad_out * x_hat;
    auto grad_weight_var = go_xhat;
    auto gw_shape = grad_weight_var.shape();
    for (int d = static_cast<int>(gw_shape.size()) - 2; d >= 0; --d) {
        grad_weight_var = sum(grad_weight_var, d, false);
    }

    auto grad_bias_var = grad_out;
    auto gb_shape = grad_bias_var.shape();
    for (int d = static_cast<int>(gb_shape.size()) - 2; d >= 0; --d) {
        grad_bias_var = sum(grad_bias_var, d, false);
    }

    return {grad_input, grad_weight_var, grad_bias_var};
}

// Factory exposed via normalization.hpp for use by F::layer_norm in
// functional.cpp. Phase 24-followup #38 fix.
namespace internal {
auto make_layer_norm_backward(bool elementwise_affine, double eps,
                              int64_t normalized_size,
                              std::vector<::tenzor::Tensor> tensors_to_save)
    -> std::shared_ptr<::tenzor::Function> {
    return std::make_shared<LayerNormBackward>(elementwise_affine, eps,
                                                normalized_size,
                                                std::move(tensors_to_save));
}

auto make_layer_norm_backward(bool elementwise_affine, double eps,
                              int64_t normalized_size,
                              std::vector<int64_t> normalized_shape,
                              std::vector<::tenzor::Tensor> tensors_to_save)
    -> std::shared_ptr<::tenzor::Function> {
    return std::make_shared<LayerNormBackward>(elementwise_affine, eps,
                                                normalized_size,
                                                std::move(normalized_shape),
                                                std::move(tensors_to_save));
}
}  // namespace internal

LayerNorm::LayerNorm(std::vector<int64_t> normalized_shape,
                     double eps,
                     bool elementwise_affine)
    : normalized_shape_(normalized_shape), eps_(eps),
      elementwise_affine_(elementwise_affine) {

    // Compute total number of features to normalize
    num_features_ = 1;
    for (auto dim : normalized_shape_) {
        num_features_ *= dim;
    }

    if (elementwise_affine) {
        weight_ = Variable(ones(normalized_shape_), true);
        bias_ = Variable(zeros(normalized_shape_), true);
        register_parameter("weight", weight_);
        register_parameter("bias", bias_);
        // Cache pointers to avoid hash map lookups in forward pass (~2-3ms savings)
        cached_weight_ = parameters_["weight"];
        cached_bias_ = parameters_["bias"];
    } else {
        weight_ = Variable(ones(normalized_shape_), false);
        bias_ = Variable(zeros(normalized_shape_), false);
    }

    reset_parameters();
}

auto LayerNorm::forward_impl(const Variable& input_orig) -> Variable {
    // Every kernel below — the SIMD CPU fast path, the GPU FusedLayerNorm
    // path, and the generic STANDARD path — reads with shape-derived
    // offsets and requires contiguous storage. Materialize a contiguous
    // copy once when the caller passed a non-contig view; downstream code
    // refers to `input` from here on (the variable name shadows
    // `input_orig` deliberately so the rest of the body is untouched).
    Variable input = input_orig;
    if (!input_orig.tensor().is_contiguous()) {
        tenzor::utils::wrap_preserving_grad(input, input_orig.tensor().contiguous());
    }

    auto shape = input.shape();

    // Verify that input shape matches normalized_shape at the end
    if (shape.size() < normalized_shape_.size()) {
        throw std::runtime_error("Input dimensions must be >= normalized_shape dimensions");
    }

    size_t norm_start = shape.size() - normalized_shape_.size();
    for (size_t i = 0; i < normalized_shape_.size(); i++) {
        if (shape[norm_start + i] != normalized_shape_[i]) {
            throw std::runtime_error("Input shape doesn't match normalized_shape");
        }
    }

    int64_t N = num_features_;

    // ============================================================================
    // FAST INFERENCE PATH: Skip all autograd overhead when gradients not needed
    // ============================================================================
    // Use cached pointers to avoid hash map lookups (~2-3ms savings per call)
    // NOTE: For inference (input doesn't require grad), use fast path regardless of
    // whether weights require grad - weight gradients only matter during training
    const bool needs_input_grad = is_grad_enabled() && input.requires_grad();

    // ============================================================================
    // INFERENCE FAST PATH: Use the registered FusedLayerNorm op on every
    // device including CPU (cpu::layer_norm_kernel_with_stats's Float32
    // branch is itself SIMD-optimized, matching the hand-rolled CPU-only
    // "ultra-fast" path this replaced — see JIT-R046). CUDA / ROCm / OneAPI
    // / Vulkan / CPU all register OpId::FusedLayerNorm, so a single dispatch
    // covers all of them, and — critically for JIT — dispatch() is what
    // makes this op visible to the tracer; the old CPU-only hand-rolled
    // pointer loop was invisible to it. Float32-only — narrower / wider
    // dtypes either widen on the kernel side or fall through to the
    // dispatch-based training-fast-path below.
    if (!needs_input_grad &&
        input.tensor().dtype() == DType::Float32) {
        // Backend FusedLayerNorm kernels assume contiguous input — non-
        // contig views (transpose / narrow) read with the wrong stride.
        Tensor x = input.tensor().is_contiguous() ? input.tensor()
                                                  : input.tensor().contiguous();
        Device dev = x.device();

        Tensor weight_dev = (elementwise_affine_ && cached_weight_) ? cached_weight_->tensor() : weight_.tensor();
        Tensor bias_dev = (elementwise_affine_ && cached_bias_) ? cached_bias_->tensor() : bias_.tensor();
        if (weight_dev.device() != dev) weight_dev = weight_dev.to(dev);
        if (bias_dev.device() != dev) bias_dev = bias_dev.to(dev);

        std::string norm_shape_str;
        for (size_t i = 0; i < normalized_shape_.size(); ++i) {
            if (i > 0) norm_shape_str += ",";
            norm_shape_str += std::to_string(normalized_shape_[i]);
        }
        NewOpAttributes attrs;
        attrs.set(AttrKey::NormalizedShape, std::string_view(norm_shape_str));
        attrs.set(AttrKey::Eps, static_cast<double>(eps_));

        std::vector<Tensor> inputs_vec = {x, weight_dev, bias_dev};
        auto results = dispatch<OpId::FusedLayerNorm>(inputs_vec, attrs);
        // BB.16: inference fast path reads results[0] — guard the read so
        // a regressing backend that returns an empty vector surfaces here.
        check_norm_outputs(results, "LayerNorm fused inference", /*min_required=*/1);
        return Variable(results[0], false);
    }

    // ============================================================================
    // DISPATCH PATH: dispatch through OpId::LayerNorm on every device
    // (including CPU — see JIT-R046) when autograd is enabled, or for any
    // dtype other than Float32 (which the INFERENCE FAST PATH above already
    // handles for the no-grad case), building a LayerNormBackward grad_fn
    // with the kernel-returned (mean, inv_std) tensors saved on the device.
    // cpu::layer_norm_kernel_with_stats covers Float32 (SIMD)/Float64/
    // Float16/BFloat16, matching what this path already required of every
    // other backend, so this now subsumes the old CPU-only hand-rolled
    // per-dtype "STANDARD PATH" pointer loops that followed this function
    // (dead code, removed) — those bypassed dispatch() entirely, making
    // CPU LayerNorm (in training, or in any non-Float32 dtype) invisible to
    // the JIT tracer. Backend kernels handle Float16/BFloat16 internally
    // (widening to Float32 for accumulators), and the existing
    // LayerNormBackward Function already upcasts Float16/BFloat16 saved
    // tensors before dispatching the backward. Covers F32/F64/F16/BF16.
    if (input.tensor().dtype() == DType::Float32 ||
        input.tensor().dtype() == DType::Float64 ||
        input.tensor().dtype() == DType::Float16 ||
        input.tensor().dtype() == DType::BFloat16) {
        const Tensor& x = input.tensor();
        Device dev = x.device();

        Tensor weight_dev = (elementwise_affine_ && cached_weight_)
            ? cached_weight_->tensor() : weight_.tensor();
        Tensor bias_dev = (elementwise_affine_ && cached_bias_)
            ? cached_bias_->tensor() : bias_.tensor();
        if (weight_dev.device() != dev) weight_dev = weight_dev.to(dev);
        if (bias_dev.device() != dev) bias_dev = bias_dev.to(dev);
        if (weight_dev.dtype() != x.dtype()) weight_dev = weight_dev.to(x.dtype());
        if (bias_dev.dtype() != x.dtype()) bias_dev = bias_dev.to(x.dtype());

        // OpId::LayerNorm is registered on every backend with the contract
        // (input, weight, bias) → (output, mean, inv_std). NormalizedShape
        // is a comma-separated string per existing convention.
        std::string norm_shape_str;
        for (size_t i = 0; i < normalized_shape_.size(); ++i) {
            if (i > 0) norm_shape_str += ",";
            norm_shape_str += std::to_string(normalized_shape_[i]);
        }
        NewOpAttributes attrs;
        attrs.set(AttrKey::NormalizedShape, std::string_view(norm_shape_str));
        attrs.set(AttrKey::Eps, static_cast<double>(eps_));

        // OpId::LayerNorm now returns the {output, mean, rstd} triple on every
        // backend (CPU, CUDA, ROCm, Vulkan, OneAPI). Previously this code
        // dispatched FusedLayerNorm as a workaround for ROCm/Vulkan returning
        // only {output}; those backends were refit to compute and return the
        // stats from their forward kernel directly. Contract-enforced below.
        std::vector<Tensor> inputs_vec = {x, weight_dev, bias_dev};
        auto results = dispatch<OpId::LayerNorm>(inputs_vec, attrs);
        check_norm_outputs(results, "LayerNorm");
        Tensor out_dev  = results[0];
        Tensor mean_dev = results[1];
        Tensor rstd_dev = results[2];

        // Wire up autograd if needed.
        const bool wt_needs_grad = elementwise_affine_ && cached_weight_ && cached_weight_->requires_grad();
        const bool bs_needs_grad = elementwise_affine_ && cached_bias_   && cached_bias_->requires_grad();
        if (needs_input_grad || wt_needs_grad || bs_needs_grad) {
            auto result = Variable(out_dev, true);

            std::vector<Tensor> tensors_to_save = {
                x,
                mean_dev,
                rstd_dev,
                weight_dev,
            };
            auto grad_fn = std::make_shared<LayerNormBackward>(
                elementwise_affine_, eps_, N, normalized_shape_,
                std::move(tensors_to_save)
            );

            std::vector<std::shared_ptr<Function>> next_funcs;
            if (auto fn = input.grad_fn()) next_funcs.push_back(fn);
            if (wt_needs_grad && cached_weight_->grad_fn()) next_funcs.push_back(cached_weight_->grad_fn());
            if (bs_needs_grad && cached_bias_->grad_fn())   next_funcs.push_back(cached_bias_->grad_fn());
            grad_fn->set_next_functions(std::move(next_funcs));

            std::vector<Variable> input_vars;
            if (input.requires_grad()) input_vars.push_back(input);
            if (wt_needs_grad) input_vars.push_back(*cached_weight_);
            if (bs_needs_grad) input_vars.push_back(*cached_bias_);
            grad_fn->set_input_variables(std::move(input_vars));

            result.set_grad_fn(grad_fn);
            return result;
        }
        return Variable(out_dev, false);
    }

    // Every valid LayerNorm dtype (Float32/Float64/Float16/BFloat16) is
    // handled by the DISPATCH PATH above; this exists only to fail loudly
    // for an actually-unsupported dtype (mirroring the pre-existing
    // hand-rolled STANDARD PATH's own final `else` branch, removed above as
    // dead code once the DISPATCH PATH started covering every device).
    throw std::runtime_error(
        "LayerNorm only supports Float16, Float32, BFloat16, and Float64 dtypes");

}

auto LayerNorm::reset_parameters() -> void {
    // Weight initialized to 1, bias to 0 (already done in constructor)
}

// ============================================================================
// GroupNorm Implementation
// ============================================================================

// GroupNorm autograd function
class GroupNormBackward : public Function {
public:
    GroupNormBackward(bool affine, double eps, int64_t num_groups,
                     int64_t num_channels, int64_t group_size,
                     std::vector<Tensor> tensors_to_save)
        : affine_(affine), eps_(eps), num_groups_(num_groups),
          num_channels_(num_channels), group_size_(group_size) {
        save_for_backward(std::move(tensors_to_save));
    }

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("GroupNormBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& grad_output_orig = grad_outputs[0];
        auto saved = saved_tensors();
        auto& input_orig = saved[0];
        auto& mean_orig = saved[1];
        auto& rstd_orig = saved[2];
        auto& weight_orig = saved[3];

        // Save original device and dtype for returning results
        auto original_device = grad_output_orig.device();
        auto original_dtype = grad_output_orig.dtype();

        // GPU fast path: dispatch to backend kernel (CUDA, Vulkan, ROCm, etc.)
        if (original_device.type != Device::Type::CPU) {
            auto go = grad_output_orig.contiguous();
            auto inp = input_orig.contiguous();
            auto mn = mean_orig.contiguous();
            auto rs = rstd_orig.contiguous();
            auto wt = weight_orig.contiguous();

            // Upcast Float16/Float64 to Float32 for backend kernels that
            // only support Float32 internally (ROCm GroupNorm, etc.)
            bool needs_cast = (original_dtype != DType::Float32);
            if (needs_cast) {
                go = go.to(DType::Float32);
                inp = inp.to(DType::Float32);
                mn = mn.to(DType::Float32);
                rs = rs.to(DType::Float32);
                wt = wt.to(DType::Float32);
            }

            OpAttributes attrs;
            attrs.set(AttrKey::NumGroups, num_groups_);
            // Canonical input order for all backend registries:
            //   [grad_output, input, mean, rstd, weight]
            // CPU and CUDA registries previously read this as
            // [grad_output, input, weight, mean, rstd] (legacy), silently
            // producing garbage xhat because rstd was used as weight and
            // weight was used as mean. Fixed in both backends.
            std::vector<Tensor> inputs_vec = {go, inp, mn, rs, wt};
            auto results = dispatch<OpId::GroupNormBackward>(inputs_vec, attrs);

            if (needs_cast) {
                for (auto& r : results) r = r.to(original_dtype);
            }
            return results;
        }

        // CPU path: keep Float64 inputs in double precision. The reduction below
        // is already done in double, so the previous unconditional .to(Float32)
        // on the stored data was the only place that discarded the extra F64
        // mantissa bits (contradicting the S.15 double-accumulation intent and
        // causing gradcheck drift). Float16/BFloat16 widen to Float32; Float32
        // stays Float32; Float64 stays Float64.
        const DType work_dtype =
            (original_dtype == DType::Float64) ? DType::Float64 : DType::Float32;

        auto grad_output = grad_output_orig.to(Device::cpu()).to(work_dtype).contiguous();
        auto input = input_orig.to(Device::cpu()).to(work_dtype).contiguous();
        auto mean = mean_orig.to(Device::cpu()).to(work_dtype).contiguous();
        auto rstd = rstd_orig.to(Device::cpu()).to(work_dtype).contiguous();
        auto weight = weight_orig.to(Device::cpu()).to(work_dtype).contiguous();

        auto shape = input.shape();
        int64_t N = shape[0];
        int64_t C = shape[1];
        int64_t H = shape[2];
        int64_t W = shape[3];
        int64_t spatial_size = H * W;

        auto grad_input = zeros_like(input);
        auto grad_weight = zeros({C}, work_dtype, Device::cpu());
        auto grad_bias = zeros({C}, work_dtype, Device::cpu());

        // S.15: all arithmetic is done in double; only the storage type T varies
        // (float for F32/half-widened, double for F64) so no precision is lost.
        auto compute = [&]<typename T>() {
            const T* input_data = input.data<T>();
            const T* mean_data = mean.data<T>();
            const T* rstd_data = rstd.data<T>();
            const T* grad_out_data = grad_output.data<T>();
            const T* weight_data = weight.data<T>();
            T* grad_in_data = grad_input.data<T>();
            T* grad_weight_data = grad_weight.data<T>();
            T* grad_bias_data = grad_bias.data<T>();

            for (int64_t n = 0; n < N; n++) {
                for (int64_t g = 0; g < num_groups_; g++) {
                    int64_t group_idx = n * num_groups_ + g;
                    double mu = static_cast<double>(mean_data[group_idx]);
                    double inv_std = static_cast<double>(rstd_data[group_idx]);

                    int64_t c_start = g * group_size_;
                    int64_t c_end = c_start + group_size_;

                    double sum_grad_out = 0.0;
                    double sum_grad_out_normalized = 0.0;
                    int64_t group_numel = group_size_ * spatial_size;

                    for (int64_t c = c_start; c < c_end; c++) {
                        double gw = 0.0, gb = 0.0;  // per-channel accumulators in double
                        for (int64_t h = 0; h < H; h++) {
                            for (int64_t w = 0; w < W; w++) {
                                int64_t idx = ((n * C + c) * H + h) * W + w;
                                double x_normalized =
                                    (static_cast<double>(input_data[idx]) - mu) * inv_std;
                                double grad_out = static_cast<double>(grad_out_data[idx]) *
                                                  static_cast<double>(weight_data[c]);
                                sum_grad_out += grad_out;
                                sum_grad_out_normalized += grad_out * x_normalized;
                                gw += static_cast<double>(grad_out_data[idx]) * x_normalized;
                                gb += static_cast<double>(grad_out_data[idx]);
                            }
                        }
                        grad_weight_data[c] += static_cast<T>(gw);
                        grad_bias_data[c] += static_cast<T>(gb);
                    }

                    double mean_grad_out = sum_grad_out / static_cast<double>(group_numel);
                    double mean_grad_out_normalized =
                        sum_grad_out_normalized / static_cast<double>(group_numel);

                    for (int64_t c = c_start; c < c_end; c++) {
                        for (int64_t h = 0; h < H; h++) {
                            for (int64_t w = 0; w < W; w++) {
                                int64_t idx = ((n * C + c) * H + h) * W + w;
                                double x_normalized =
                                    (static_cast<double>(input_data[idx]) - mu) * inv_std;
                                double grad_out = static_cast<double>(grad_out_data[idx]) *
                                                  static_cast<double>(weight_data[c]);
                                grad_in_data[idx] = static_cast<T>(
                                    (grad_out - mean_grad_out -
                                     x_normalized * mean_grad_out_normalized) * inv_std);
                            }
                        }
                    }
                }
            }
        };
        if (work_dtype == DType::Float64) compute.template operator()<double>();
        else compute.template operator()<float>();

        // Move results back to original device and dtype
        return {grad_input.to(original_dtype).to(original_device).contiguous(),
                grad_weight.to(original_dtype).to(original_device).contiguous(),
                grad_bias.to(original_dtype).to(original_device).contiguous()};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        // Full higher-order gradient support for GroupNorm.
        // Enables create_graph=true for second derivatives through GroupNorm.
        //
        // GroupNorm: reshape to groups, apply LayerNorm-style normalization per group.
        // saved: [0]=input [N,C,H,W], [1]=mean [N*G], [2]=rstd [N*G], [3]=weight [C]
        auto& grad_out = grad_outputs[0];

        Variable input_var, mean_var, rstd_var, weight_var;
        if (has_saved_variables()) {
            const auto& sv = saved_variables();
            input_var = sv[0];
            mean_var = sv[1];
            rstd_var = sv[2];
            weight_var = sv[3];
        } else {
            auto saved = saved_tensors();
            input_var = Variable(saved[0], false);
            mean_var = Variable(saved[1], false);
            rstd_var = Variable(saved[2], false);
            weight_var = Variable(saved[3], false);
        }

        auto shape = input_var.shape();
        int64_t N = shape[0];
        int64_t C = shape[1];
        int64_t H = shape[2];
        int64_t W = shape[3];
        int64_t G = num_groups_;
        int64_t cpg = group_size_;  // channels per group
        int64_t group_numel = cpg * H * W;

        // Reshape input and grad_output to [N, G, cpg*H*W] for per-group normalization
        auto input_r = reshape(input_var, {N, G, group_numel});
        auto grad_out_r = reshape(grad_out, {N, G, group_numel});

        // mean/rstd are [N*G], reshape to [N, G, 1] for broadcasting
        auto mean_bc = reshape(mean_var, {N, G, 1});
        auto rstd_bc = reshape(rstd_var, {N, G, 1});

        // x_hat = (input - mean) * rstd, per group
        auto x_hat = (input_r - mean_bc) * rstd_bc;

        // Weight broadcast: [C] -> [1, G, cpg, 1] -> reshape to [1, G, cpg*H*W]
        // We need weight per element in the group. Weight is per-channel, so
        // reshape to [G, cpg] then tile over spatial dims.
        // Reshape weight [C] -> [1, C, 1, 1] and apply to [N, C, H, W] shaped grad_out
        auto weight_bc = unsqueeze(unsqueeze(unsqueeze(weight_var, 0), 2), 3); // [1, C, 1, 1]
        auto grad_weighted = grad_out * weight_bc;  // [N, C, H, W]
        auto grad_x_hat = reshape(grad_weighted, {N, G, group_numel});

        // mean(grad_x_hat) over group elements (last dim)
        auto mean_gxh = sum(grad_x_hat, 2, true) / static_cast<float>(group_numel);

        // mean(grad_x_hat * x_hat) over group elements
        auto mean_gxh_xh = sum(grad_x_hat * x_hat, 2, true) / static_cast<float>(group_numel);

        // grad_input = (grad_x_hat - mean_gxh - x_hat * mean_gxh_xh) * rstd
        auto grad_input_r = (grad_x_hat - mean_gxh - x_hat * mean_gxh_xh) * rstd_bc;
        auto grad_input = reshape(grad_input_r, {N, C, H, W});

        // grad_weight = sum(grad_output * x_hat_full, dims=[0,2,3]) -> [C]
        // x_hat reshaped back to [N, C, H, W]
        auto x_hat_full = reshape(x_hat, {N, C, H, W});
        auto go_xhat = reshape(grad_out * x_hat_full, {N, C, H * W});
        auto grad_weight_var = sum(sum(go_xhat, 0, false), 1, false);  // [C]

        // grad_bias = sum(grad_output, dims=[0,2,3]) -> [C]
        auto go_flat = reshape(grad_out, {N, C, H * W});
        auto grad_bias_var = sum(sum(go_flat, 0, false), 1, false);  // [C]

        return {grad_input, grad_weight_var, grad_bias_var};
    }

    // Full Variable-level backward enables create_graph=true for second derivatives.
    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return false; }

private:
    bool affine_;
    double eps_;
    int64_t num_groups_;
    int64_t num_channels_;
    int64_t group_size_;
};

GroupNorm::GroupNorm(int64_t num_groups,
                     int64_t num_channels,
                     double eps,
                     bool affine)
    : num_groups_(num_groups), num_channels_(num_channels),
      eps_(eps), affine_(affine) {

    if (num_channels_ % num_groups_ != 0) {
        throw std::runtime_error("num_channels must be divisible by num_groups");
    }

    if (affine) {
        weight_ = Variable(ones({num_channels_}), true);
        bias_ = Variable(zeros({num_channels_}), true);
        register_parameter("weight", weight_);
        register_parameter("bias", bias_);
    } else {
        weight_ = Variable(ones({num_channels_}), false);
        bias_ = Variable(zeros({num_channels_}), false);
    }

    reset_parameters();
}

auto GroupNorm::forward_impl(const Variable& input) -> Variable {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::runtime_error("GroupNorm expects 4D input (got " +
                               std::to_string(shape.size()) + "D)");
    }

    int64_t C = shape[1];

    if (C != num_channels_) {
        throw std::runtime_error("Expected " + std::to_string(num_channels_) +
                               " channels, got " + std::to_string(C));
    }

    int64_t group_size = num_channels_ / num_groups_;

    // Save original device and dtype
    auto original_device = input.tensor().device();
    auto original_dtype = input.tensor().dtype();

    // Dispatch to the backend kernel on every device including CPU
    // (cpu::group_norm_kernel_with_stats — see JIT-R047). The old CPU-only
    // hand-rolled pointer-based path bypassed dispatch() entirely, making
    // CPU GroupNorm invisible to the JIT tracer; removed as dead code now
    // that this dispatch call covers every backend.
    // Float16: upcast to Float32 for precision in mean/variance computation.
    // GroupNorm statistics (mean, variance) accumulate over group_size * spatial_size
    // elements; Float16's limited mantissa causes significant rounding errors.
    DType compute_dtype = original_dtype;
    bool needs_upcast = (original_dtype == DType::Float16 || original_dtype == DType::BFloat16);
    if (needs_upcast) compute_dtype = DType::Float32;

    Tensor input_compute = needs_upcast ? input.tensor().to(DType::Float32) : input.tensor();

    Tensor weight_tensor = affine_
        ? parameters_["weight"]->tensor().to(compute_dtype).to(original_device)
        : ones({C}, compute_dtype, original_device);
    Tensor bias_tensor = affine_
        ? parameters_["bias"]->tensor().to(compute_dtype).to(original_device)
        : zeros({C}, compute_dtype, original_device);

    OpAttributes attrs;
    attrs.set(AttrKey::NumGroups, num_groups_);
    attrs.set(AttrKey::Eps, eps_);
    std::vector<Tensor> inputs_vec = {input_compute, weight_tensor, bias_tensor};
    auto results = dispatch<OpId::GroupNorm>(inputs_vec, attrs);
    // BB.16: shared helper (V.24/Y.20 contract guard). Reading
    // results[1] / results[2] without checking size would SEGFAULT in
    // this layer if a backend regresses to returning {output} only.
    check_norm_outputs(results, "GroupNorm");

    Tensor output = results[0];
    Tensor saved_mean = results[1];
    Tensor saved_rstd = results[2];

    // Downcast output back to original dtype if we upcast for computation
    if (needs_upcast) {
        output = output.to(original_dtype);
    }

    if (is_grad_enabled() && (input.requires_grad() || (affine_ && parameters_["weight"]->requires_grad()))) {
        auto result = Variable(output, true);

        std::vector<Tensor> tensors_to_save = {
            input.tensor(), saved_mean, saved_rstd, weight_tensor
        };

        auto grad_fn = std::make_shared<GroupNormBackward>(
            affine_, eps_, num_groups_, num_channels_, group_size,
            std::move(tensors_to_save)
        );

        result.set_grad_fn(grad_fn);

        std::vector<std::shared_ptr<Function>> next_funcs;
        if (auto input_grad_fn = input.grad_fn()) {
            next_funcs.push_back(input_grad_fn);
        }
        if (affine_) {
            auto weight_it = parameters_.find("weight");
            auto bias_it = parameters_.find("bias");
            if (weight_it != parameters_.end() && weight_it->second->requires_grad()) {
                if (auto weight_grad_fn = weight_it->second->grad_fn()) {
                    next_funcs.push_back(weight_grad_fn);
                }
            }
            if (bias_it != parameters_.end() && bias_it->second->requires_grad()) {
                if (auto bias_grad_fn = bias_it->second->grad_fn()) {
                    next_funcs.push_back(bias_grad_fn);
                }
            }
        }
        grad_fn->set_next_functions(next_funcs);

        std::vector<Variable> input_vars;
        if (input.requires_grad()) {
            input_vars.push_back(input);
        }
        if (affine_) {
            auto weight_it = parameters_.find("weight");
            auto bias_it = parameters_.find("bias");
            if (weight_it != parameters_.end() && weight_it->second->requires_grad()) {
                input_vars.push_back(*weight_it->second);
            }
            if (bias_it != parameters_.end() && bias_it->second->requires_grad()) {
                input_vars.push_back(*bias_it->second);
            }
        }
        grad_fn->set_input_variables(input_vars);

        return result;
    }

    return Variable(output, false);
}

auto GroupNorm::reset_parameters() -> void {
    // Weight initialized to 1, bias to 0 (already done in constructor)
}

// Factory exposed via normalization.hpp for use by F::group_norm in
// functional.cpp. The functional path otherwise wraps the dispatch result
// in a Variable with no grad_fn, silently zeroing input/weight/bias
// gradients on backward — same pattern as the F::layer_norm fix.
namespace internal {
auto make_group_norm_backward(bool affine, double eps,
                              int64_t num_groups, int64_t num_channels,
                              int64_t group_size,
                              std::vector<::tenzor::Tensor> tensors_to_save)
    -> std::shared_ptr<::tenzor::Function> {
    return std::make_shared<GroupNormBackward>(affine, eps, num_groups,
                                                num_channels, group_size,
                                                std::move(tensors_to_save));
}
}  // namespace internal

// ============================================================================
// InstanceNorm2d Implementation
// ============================================================================

// Wrapper backward for InstanceNorm1d: reshapes 3D grad_output to 4D,
// delegates to the 4D InstanceNorm backward kernel, reshapes grad_input back to 3D.
class InstanceNorm1dBackwardFn : public Function {
public:
    InstanceNorm1dBackwardFn(bool affine, double eps, int64_t num_features,
                             int64_t L, std::vector<Tensor> tensors_to_save)
        : affine_(affine), eps_(eps), num_features_(num_features), L_(L) {
        save_for_backward(std::move(tensors_to_save));
    }

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("InstanceNorm1dBackwardFn::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& grad_output_3d = grad_outputs[0];  // (N, C, L)
        auto saved = saved_tensors();
        auto& input_4d = saved[0];     // (N, C, L, 1)
        auto& mean = saved[1];         // (N, C)
        auto& rstd = saved[2];         // (N, C)
        auto& weight = saved[3];       // (C)

        auto original_device = grad_output_3d.device();
        auto original_dtype = grad_output_3d.dtype();

        auto shape = grad_output_3d.shape();
        int64_t N = shape[0], C = shape[1];

        // Reshape grad_output from 3D to 4D
        Tensor grad_output_4d = grad_output_3d.reshape({N, C, L_, 1});

        // The forward saves mean/rstd/weight at Float32 for a half input, so the
        // saved operands have mixed dtypes; the backward kernel branches on the
        // input dtype and would call weight.data<half>() on a float tensor.
        // Upcast every operand to one compute dtype (float for half), then narrow
        // the grads back below.
        const DType compute_dt =
            (original_dtype == DType::Float16 || original_dtype == DType::BFloat16)
                ? DType::Float32 : original_dtype;

        std::vector<Tensor> inputs_vec;
        if (original_device.type != Device::Type::CPU) {
            inputs_vec = {grad_output_4d.to(compute_dt).contiguous(), input_4d.to(compute_dt).contiguous(),
                         weight.to(compute_dt).contiguous(), mean.to(compute_dt).contiguous(), rstd.to(compute_dt).contiguous()};
        } else {
            inputs_vec = {grad_output_4d.to(Device::cpu()).to(compute_dt).contiguous(),
                         input_4d.to(Device::cpu()).to(compute_dt).contiguous(),
                         weight.to(Device::cpu()).to(compute_dt).contiguous(),
                         mean.to(Device::cpu()).to(compute_dt).contiguous(),
                         rstd.to(Device::cpu()).to(compute_dt).contiguous()};
        }
        auto results = dispatch<OpId::InstanceNormBackward>(inputs_vec);

        // Reshape grad_input from 4D back to 3D, then narrow all grads back to
        // the original dtype/device (results are in compute_dt after the upcast).
        Tensor grad_input = results[0].reshape({N, C, L_});
        return {grad_input.to(original_dtype).to(original_device).contiguous(),
                results[1].to(original_dtype).to(original_device).contiguous(),
                results[2].to(original_dtype).to(original_device).contiguous()};
    }

    auto name() const -> std::string override { return "InstanceNorm1dBackwardFn"; }
    auto supports_higher_order() const -> bool override { return false; }

private:
    bool affine_;
    double eps_;
    int64_t num_features_;
    int64_t L_;
};

// Wrapper backward for InstanceNorm3d: reshapes 5D grad_output to 4D,
// delegates to the 4D InstanceNorm backward, reshapes grad_input back to 5D.
class InstanceNorm3dBackwardFn : public Function {
public:
    InstanceNorm3dBackwardFn(bool affine, double eps, int64_t num_features,
                             int64_t D, int64_t H, int64_t W,
                             std::vector<Tensor> tensors_to_save)
        : affine_(affine), eps_(eps), num_features_(num_features),
          D_(D), H_(H), W_(W) {
        save_for_backward(std::move(tensors_to_save));
    }

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("InstanceNorm3dBackwardFn::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& grad_output_5d = grad_outputs[0];  // (N, C, D, H, W)
        auto saved = saved_tensors();
        auto& input_4d = saved[0];     // (N, C, D*H, W)
        auto& mean = saved[1];         // (N, C)
        auto& rstd = saved[2];         // (N, C)
        auto& weight = saved[3];       // (C)

        auto original_device = grad_output_5d.device();
        auto original_dtype = grad_output_5d.dtype();

        auto shape = grad_output_5d.shape();
        int64_t N = shape[0], C = shape[1];

        // Reshape grad_output from 5D to 4D
        Tensor grad_output_4d = grad_output_5d.reshape({N, C, D_ * H_, W_});

        std::vector<Tensor> inputs_vec;
        if (original_device.type != Device::Type::CPU) {
            inputs_vec = {grad_output_4d.contiguous(), input_4d.contiguous(),
                         weight.contiguous(), mean.contiguous(), rstd.contiguous()};
        } else {
            inputs_vec = {grad_output_4d.to(Device::cpu()).contiguous(),
                         input_4d.to(Device::cpu()).contiguous(),
                         weight.to(Device::cpu()).contiguous(),
                         mean.to(Device::cpu()).contiguous(),
                         rstd.to(Device::cpu()).contiguous()};
        }
        auto results = dispatch<OpId::InstanceNormBackward>(inputs_vec);

        // Reshape grad_input from 4D back to 5D
        Tensor grad_input = results[0].reshape({N, C, D_, H_, W_});
        if (original_device.type == Device::Type::CPU) {
            return {grad_input.to(original_dtype).to(original_device).contiguous(),
                    results[1].to(original_dtype).to(original_device).contiguous(),
                    results[2].to(original_dtype).to(original_device).contiguous()};
        }
        return {grad_input, results[1], results[2]};
    }

    auto name() const -> std::string override { return "InstanceNorm3dBackwardFn"; }
    auto supports_higher_order() const -> bool override { return false; }

private:
    bool affine_;
    double eps_;
    int64_t num_features_;
    int64_t D_, H_, W_;
};

// InstanceNorm autograd function
class InstanceNormBackwardFn : public Function {
public:
    InstanceNormBackwardFn(bool affine, double eps, int64_t num_features,
                           std::vector<Tensor> tensors_to_save)
        : affine_(affine), eps_(eps), num_features_(num_features) {
        save_for_backward(std::move(tensors_to_save));
    }

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("InstanceNormBackwardFn::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& grad_output_orig = grad_outputs[0];
        auto saved = saved_tensors();
        auto& input_orig = saved[0];      // [N, C, *]
        auto& mean_orig = saved[1];       // [N, C]
        auto& rstd_orig = saved[2];       // [N, C]
        auto& weight_orig = saved[3];     // [C]

        auto original_device = grad_output_orig.device();
        auto original_dtype = grad_output_orig.dtype();

        // The forward saves mean/rstd/weight at Float32 for a Float16/BFloat16
        // input (for stat precision), so the saved tensors have MIXED dtypes
        // (input/grad_output half; weight/stats float). The backend backward
        // kernel branches on the input dtype and reads every operand at that
        // dtype, so a half input + float weight makes it call weight.data<half>()
        // on a float tensor and throw. Upcast all operands to a single compute
        // dtype (float for half inputs), dispatch, then narrow the grads back to
        // the original dtype.
        const DType compute_dt =
            (original_dtype == DType::Float16 || original_dtype == DType::BFloat16)
                ? DType::Float32 : original_dtype;

        // GPU fast path
        if (original_device.type != Device::Type::CPU) {
            auto go = grad_output_orig.to(compute_dt).contiguous();
            auto inp = input_orig.to(compute_dt).contiguous();
            auto wt = weight_orig.to(compute_dt).contiguous();
            auto mn = mean_orig.to(compute_dt).contiguous();
            auto rs = rstd_orig.to(compute_dt).contiguous();

            // InstanceNormBackward: inputs [grad_output, input, weight, mean, rstd]
            std::vector<Tensor> inputs_vec = {go, inp, wt, mn, rs};
            auto results = dispatch<OpId::InstanceNormBackward>(inputs_vec);
            for (auto& r : results) {
                r = r.to(original_dtype).to(original_device).contiguous();
            }
            return results;
        }

        // CPU path: dispatch to kernel
        auto go = grad_output_orig.to(Device::cpu()).to(compute_dt).contiguous();
        auto inp = input_orig.to(Device::cpu()).to(compute_dt).contiguous();
        auto mn = mean_orig.to(Device::cpu()).to(compute_dt).contiguous();
        auto rs = rstd_orig.to(Device::cpu()).to(compute_dt).contiguous();
        auto wt = weight_orig.to(Device::cpu()).to(compute_dt).contiguous();

        // InstanceNormBackward: inputs [grad_output, input, weight, mean, rstd]
        std::vector<Tensor> inputs_vec = {go, inp, wt, mn, rs};
        auto results = dispatch<OpId::InstanceNormBackward>(inputs_vec);

        return {results[0].to(original_dtype).to(original_device).contiguous(),
                results[1].to(original_dtype).to(original_device).contiguous(),
                results[2].to(original_dtype).to(original_device).contiguous()};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        // Full higher-order gradient support for InstanceNorm.
        // Enables create_graph=true for second derivatives through InstanceNorm.
        //
        // InstanceNorm is GroupNorm with groups = channels (each channel normalized independently).
        // saved: [0]=input [N,C,*], [1]=mean [N,C], [2]=rstd [N,C], [3]=weight [C]
        auto& grad_out = grad_outputs[0];

        Variable input_var, mean_var, rstd_var, weight_var;
        if (has_saved_variables()) {
            const auto& sv = saved_variables();
            input_var = sv[0];
            mean_var = sv[1];
            rstd_var = sv[2];
            weight_var = sv[3];
        } else {
            auto saved = saved_tensors();
            input_var = Variable(saved[0], false);
            mean_var = Variable(saved[1], false);
            rstd_var = Variable(saved[2], false);
            weight_var = Variable(saved[3], false);
        }

        auto shape = input_var.shape();
        int64_t N = shape[0];
        int64_t C = shape[1];
        // Compute spatial size from remaining dimensions
        int64_t spatial_size = 1;
        for (size_t d = 2; d < shape.size(); ++d) {
            spatial_size *= shape[d];
        }

        // Reshape input and grad_output to [N, C, spatial_size] for per-instance normalization
        auto input_r = reshape(input_var, {N, C, spatial_size});
        auto grad_out_r = reshape(grad_out, {N, C, spatial_size});

        // mean/rstd are [N, C], unsqueeze to [N, C, 1] for broadcasting
        auto mean_bc = unsqueeze(mean_var, -1);   // [N, C, 1]
        auto rstd_bc = unsqueeze(rstd_var, -1);   // [N, C, 1]

        // x_hat = (input - mean) * rstd, per instance
        auto x_hat = (input_r - mean_bc) * rstd_bc;

        // Weight broadcast: [C] -> [1, C, 1]
        auto weight_bc = unsqueeze(unsqueeze(weight_var, 0), -1);  // [1, C, 1]
        auto grad_x_hat = grad_out_r * weight_bc;

        // mean(grad_x_hat) over spatial dim (last dim)
        auto mean_gxh = sum(grad_x_hat, 2, true) / static_cast<float>(spatial_size);

        // mean(grad_x_hat * x_hat) over spatial dim
        auto mean_gxh_xh = sum(grad_x_hat * x_hat, 2, true) / static_cast<float>(spatial_size);

        // grad_input = (grad_x_hat - mean_gxh - x_hat * mean_gxh_xh) * rstd
        auto grad_input_r = (grad_x_hat - mean_gxh - x_hat * mean_gxh_xh) * rstd_bc;
        std::vector<int64_t> orig_shape(shape.begin(), shape.end());
        auto grad_input = reshape(grad_input_r, orig_shape);

        // grad_weight = sum(grad_output * x_hat, dims=[0, spatial_dims]) -> [C]
        // sum over dim 0 (batch) and dim 2 (spatial)
        auto go_xhat = grad_out_r * x_hat;
        Variable grad_weight_var = sum(sum(go_xhat, 0, false), 1, false);  // [C]

        // grad_bias = sum(grad_output, dims=[0, spatial_dims]) -> [C]
        Variable grad_bias_var = sum(sum(grad_out_r, 0, false), 1, false);  // [C]

        // Narrow the grads back to the grad_output (= original input/param)
        // dtype. The saved mean/rstd/weight are stored at Float32 for a
        // Float16/BFloat16 input, so the Variable ops above promote grad_input
        // (and grad_weight/bias) to Float32; without this narrow, backward
        // returns a Float32 gradient for a Float16 tensor (dtype mismatch on
        // accumulation). Differentiable, so create_graph=true still works.
        // Mirrors the Tensor-level backward() which narrows to original_dtype.
        DType out_dtype = grad_out.tensor().dtype();
        if (grad_input.tensor().dtype() != out_dtype) {
            grad_input = tenzor::nn::variable_cast(grad_input, out_dtype);
        }
        if (grad_weight_var.tensor().dtype() != out_dtype) {
            grad_weight_var = tenzor::nn::variable_cast(grad_weight_var, out_dtype);
        }
        if (grad_bias_var.tensor().dtype() != out_dtype) {
            grad_bias_var = tenzor::nn::variable_cast(grad_bias_var, out_dtype);
        }

        std::vector<Variable> result;
        result.push_back(std::move(grad_input));
        result.push_back(std::move(grad_weight_var));
        result.push_back(std::move(grad_bias_var));
        return result;
    }

    // Full Variable-level backward enables create_graph=true for second derivatives.
    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return false; }

private:
    bool affine_;
    double eps_;
    int64_t num_features_;
};

InstanceNorm2d::InstanceNorm2d(int64_t num_features, double eps, bool affine)
    : num_features_(num_features), eps_(eps), affine_(affine) {

    if (affine) {
        weight_ = Variable(ones({num_features_}), true);
        bias_ = Variable(zeros({num_features_}), true);
        register_parameter("weight", weight_);
        register_parameter("bias", bias_);
    } else {
        weight_ = Variable(ones({num_features_}), false);
        bias_ = Variable(zeros({num_features_}), false);
    }

    reset_parameters();
}

auto InstanceNorm2d::forward_impl(const Variable& input) -> Variable {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::runtime_error("InstanceNorm2d expects 4D input (N, C, H, W), got " +
                               std::to_string(shape.size()) + "D");
    }

    int64_t C = shape[1];
    if (C != num_features_) {
        throw std::runtime_error("Expected " + std::to_string(num_features_) +
                               " channels, got " + std::to_string(C));
    }

    auto original_device = input.tensor().device();
    auto original_dtype = input.tensor().dtype();

    // audit-2026-06 Fix #6: Float16/BFloat16 upcast to Float32 for the
    // per-instance mean/variance (which accumulate over H*W elements; the
    // half-precision mantissa loses accuracy). Mirrors the GroupNorm GPU-path
    // widen→compute→narrow above. compute_dtype stays the original dtype for
    // Float32/Float64.
    const bool needs_upcast =
        (original_dtype == DType::Float16 || original_dtype == DType::BFloat16);
    const DType compute_dtype = needs_upcast ? DType::Float32 : original_dtype;
    Tensor input_compute = needs_upcast ? input.tensor().to(DType::Float32) : input.tensor();

    // Get weight and bias tensors
    Tensor weight_tensor = affine_
        ? parameters_["weight"]->tensor().to(compute_dtype).to(original_device)
        : ones({C}, compute_dtype, original_device);
    Tensor bias_tensor = affine_
        ? parameters_["bias"]->tensor().to(compute_dtype).to(original_device)
        : zeros({C}, compute_dtype, original_device);

    // Dispatch to backend kernel
    NewOpAttributes attrs;
    attrs.set(AttrKey::Eps, static_cast<double>(eps_));
    std::vector<Tensor> inputs_vec = {input_compute, weight_tensor, bias_tensor};
    auto results = dispatch<OpId::InstanceNorm>(inputs_vec, attrs);
    // BB.16: shared helper — InstanceNorm2d 2D path (already had the guard).
    check_norm_outputs(results, "InstanceNorm2d");

    Tensor output = results[0];
    Tensor saved_mean = results[1];   // [N, C]
    Tensor saved_rstd = results[2];   // [N, C]

    // Narrow the output back to the original (half) dtype.
    if (needs_upcast) {
        output = output.to(original_dtype);
    }

    if (is_grad_enabled() && (input.requires_grad() || (affine_ && parameters_["weight"]->requires_grad()))) {
        auto result = Variable(output, true);

        std::vector<Tensor> tensors_to_save = {
            input.tensor(), saved_mean, saved_rstd, weight_tensor
        };

        auto grad_fn = std::make_shared<InstanceNormBackwardFn>(
            affine_, eps_, num_features_, std::move(tensors_to_save));

        result.set_grad_fn(grad_fn);

        std::vector<std::shared_ptr<Function>> next_funcs;
        if (auto input_grad_fn = input.grad_fn()) {
            next_funcs.push_back(input_grad_fn);
        }
        if (affine_) {
            auto weight_it = parameters_.find("weight");
            auto bias_it = parameters_.find("bias");
            if (weight_it != parameters_.end() && weight_it->second->requires_grad()) {
                if (auto weight_grad_fn = weight_it->second->grad_fn()) {
                    next_funcs.push_back(weight_grad_fn);
                }
            }
            if (bias_it != parameters_.end() && bias_it->second->requires_grad()) {
                if (auto bias_grad_fn = bias_it->second->grad_fn()) {
                    next_funcs.push_back(bias_grad_fn);
                }
            }
        }
        grad_fn->set_next_functions(next_funcs);

        std::vector<Variable> input_vars;
        if (input.requires_grad()) {
            input_vars.push_back(input);
        }
        if (affine_) {
            if (auto it = parameters_.find("weight"); it != parameters_.end() && it->second->requires_grad()) {
                input_vars.push_back(*it->second);
            }
            // R.17: backward returns {grad_input, grad_weight, grad_bias};
            // the bias parameter must appear in input_vars (paired with the
            // grad_bias slot) or the bias gradient is silently dropped.
            if (auto it = parameters_.find("bias"); it != parameters_.end() && it->second->requires_grad()) {
                input_vars.push_back(*it->second);
            }
        }
        grad_fn->set_input_variables(input_vars);

        return result;
    }

    return Variable(output, false);
}

auto InstanceNorm2d::reset_parameters() -> void {
    // Weight initialized to 1, bias to 0 (already done in constructor)
}

// Factory exposed via normalization.hpp for use by F::instance_norm in
// functional.cpp. Wires up InstanceNormBackwardFn so the functional path
// preserves gradients for input/weight/bias.
namespace internal {
auto make_instance_norm_backward(bool affine, double eps,
                                 int64_t num_features,
                                 std::vector<::tenzor::Tensor> tensors_to_save)
    -> std::shared_ptr<::tenzor::Function> {
    return std::make_shared<InstanceNormBackwardFn>(affine, eps, num_features,
                                                     std::move(tensors_to_save));
}
}  // namespace internal

// ============================================================================
// InstanceNorm1d Implementation
// ============================================================================

InstanceNorm1d::InstanceNorm1d(int64_t num_features, double eps, bool affine)
    : num_features_(num_features), eps_(eps), affine_(affine) {

    if (affine) {
        weight_ = Variable(ones({num_features_}), true);
        bias_ = Variable(zeros({num_features_}), true);
        register_parameter("weight", weight_);
        register_parameter("bias", bias_);
    } else {
        weight_ = Variable(ones({num_features_}), false);
        bias_ = Variable(zeros({num_features_}), false);
    }

    reset_parameters();
}

auto InstanceNorm1d::forward_impl(const Variable& input) -> Variable {
    auto shape = input.shape();
    if (shape.size() != 3) {
        throw std::runtime_error("InstanceNorm1d expects 3D input (N, C, L), got " +
                               std::to_string(shape.size()) + "D");
    }

    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t L = shape[2];

    if (C != num_features_) {
        throw std::runtime_error("Expected " + std::to_string(num_features_) +
                               " channels, got " + std::to_string(C));
    }

    auto original_device = input.tensor().device();
    auto original_dtype = input.tensor().dtype();

    // Reshape (N, C, L) -> (N, C, L, 1) to reuse the 4D kernel
    Tensor input_4d = input.tensor().reshape({N, C, L, 1});

    // audit-2026-06 Fix #6: Float16/BFloat16 upcast to Float32 for the
    // per-instance mean/variance (accumulated over L elements). Mirrors the
    // GroupNorm GPU-path widen→compute→narrow; original-dtype 4D input is
    // still saved for backward below.
    const bool needs_upcast =
        (original_dtype == DType::Float16 || original_dtype == DType::BFloat16);
    const DType compute_dtype = needs_upcast ? DType::Float32 : original_dtype;
    Tensor input_4d_compute = needs_upcast ? input_4d.to(DType::Float32) : input_4d;

    Tensor weight_tensor = affine_
        ? parameters_["weight"]->tensor().to(compute_dtype).to(original_device)
        : ones({C}, compute_dtype, original_device);
    Tensor bias_tensor = affine_
        ? parameters_["bias"]->tensor().to(compute_dtype).to(original_device)
        : zeros({C}, compute_dtype, original_device);

    NewOpAttributes attrs;
    attrs.set(AttrKey::Eps, static_cast<double>(eps_));
    std::vector<Tensor> inputs_vec = {input_4d_compute, weight_tensor, bias_tensor};
    auto results = dispatch<OpId::InstanceNorm>(inputs_vec, attrs);
    // BB.16: shared helper — InstanceNorm1d wrapper path. The 1D wrapper
    // reshapes to 4D and reuses the InstanceNorm2d kernel registration;
    // every backend that satisfies the {output, mean, rstd} contract for
    // 2D also satisfies it here, but a regressing backend must surface
    // the contract violation in this wrapper, not segfault on results[1].
    check_norm_outputs(results, "InstanceNorm1d");

    // Reshape output back to (N, C, L)
    Tensor output = results[0].reshape({N, C, L});
    Tensor saved_mean = results[1];
    Tensor saved_rstd = results[2];

    // Narrow the output back to the original (half) dtype.
    if (needs_upcast) {
        output = output.to(original_dtype);
    }

    if (is_grad_enabled() && (input.requires_grad() || (affine_ && parameters_["weight"]->requires_grad()))) {
        auto result = Variable(output, true);

        // Save 4D input for backward (backward kernel expects 4D)
        std::vector<Tensor> tensors_to_save = {
            input_4d, saved_mean, saved_rstd, weight_tensor
        };

        auto grad_fn = std::make_shared<InstanceNorm1dBackwardFn>(
            affine_, eps_, num_features_, L, std::move(tensors_to_save));

        result.set_grad_fn(grad_fn);

        std::vector<std::shared_ptr<Function>> next_funcs;
        if (auto input_grad_fn = input.grad_fn()) {
            next_funcs.push_back(input_grad_fn);
        }
        if (affine_) {
            auto weight_it = parameters_.find("weight");
            auto bias_it = parameters_.find("bias");
            if (weight_it != parameters_.end() && weight_it->second->requires_grad()) {
                if (auto weight_grad_fn = weight_it->second->grad_fn()) {
                    next_funcs.push_back(weight_grad_fn);
                }
            }
            if (bias_it != parameters_.end() && bias_it->second->requires_grad()) {
                if (auto bias_grad_fn = bias_it->second->grad_fn()) {
                    next_funcs.push_back(bias_grad_fn);
                }
            }
        }
        grad_fn->set_next_functions(next_funcs);

        std::vector<Variable> input_vars;
        if (input.requires_grad()) {
            input_vars.push_back(input);
        }
        if (affine_) {
            if (auto it = parameters_.find("weight"); it != parameters_.end() && it->second->requires_grad()) {
                input_vars.push_back(*it->second);
            }
            // R.17: backward returns {grad_input, grad_weight, grad_bias};
            // the bias parameter must appear in input_vars or the bias
            // gradient is silently dropped.
            if (auto it = parameters_.find("bias"); it != parameters_.end() && it->second->requires_grad()) {
                input_vars.push_back(*it->second);
            }
        }
        grad_fn->set_input_variables(input_vars);

        return result;
    }

    return Variable(output, false);
}

auto InstanceNorm1d::reset_parameters() -> void {
    // Weight initialized to 1, bias to 0 (already done in constructor)
}

// ============================================================================
// InstanceNorm3d — reshape 5D to 4D, delegate to InstanceNorm2d
// ============================================================================

InstanceNorm3d::InstanceNorm3d(int64_t num_features, double eps, bool affine)
    : num_features_(num_features),
      // audit-7 FF.13: heap-allocate the delegate so register_module owns a
      // real shared_ptr (no no-op-deleter wrap of a stack member, which was
      // fragile under copy/move of the enclosing InstanceNorm3d).
      in2d_(std::make_shared<InstanceNorm2d>(num_features, eps, affine)) {
    register_module("in2d", in2d_);
}

auto InstanceNorm3d::forward_impl(const Variable& input) -> Variable {
    auto shape = input.shape();
    if (shape.size() != 5) {
        throw std::runtime_error("InstanceNorm3d expects 5D input (N,C,D,H,W), got " +
            std::to_string(shape.size()) + "D");
    }

    int64_t N = shape[0], C = shape[1], D = shape[2], H = shape[3], W = shape[4];

    // audit-7 FF.18: reshape [N,C,D,H,W] -> [N,C,D*H,W] (the prior shape)
    // dispatches to InstanceNorm2d, which saves mean/rstd at shape
    // [N,C,1,1] — visually a 2D-spatial stat shape, but the underlying
    // reduction is over D*H*W per (n, c) instance, identical to a 1D-flat
    // reduction.  We keep the 4D reshape (InstanceNorm1d does not exist as
    // a separate dispatch path in this layer file, and synthesising one
    // would introduce a redundant code path) but document the invariant:
    // mean[n, c, 0, 0] and rstd[n, c, 0, 0] are the mean/rstd over all
    // D*H*W elements of the (n, c) instance.  The 2D-spatial shape is
    // therefore *equivalent* to [N, C, 1] up to a trailing singleton; the
    // backward in InstanceNorm2d consumes the stat at the same shape it
    // was saved at, so consumers see no inconsistency.
    // Use tenzor::reshape (autograd-aware) to maintain computation graph.
    auto reshaped = tenzor::reshape(input, {N, C, D * H, W});
    auto result = in2d_->forward(reshaped);
    return tenzor::reshape(result, {N, C, D, H, W});
}

// ============================================================================
// RMSNorm Implementation
// ============================================================================

// RMSNorm autograd function
class RMSNormBackward : public Function {
public:
    RMSNormBackward(double eps, int64_t normalized_size, std::vector<Tensor> tensors_to_save)
        : eps_(eps), normalized_size_(normalized_size) {
        save_for_backward(std::move(tensors_to_save));
    }

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("RMSNormBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& grad_output_orig = grad_outputs[0];
        auto saved = saved_tensors();
        auto& input_orig = saved[0];
        auto& rrms_orig = saved[1];  // reciprocal root mean square (1 / sqrt(mean(x^2) + eps))
        auto& weight_orig = saved[2];

        // Save original device before transferring to CPU
        Device original_device = input_orig.device();
        DType original_dtype = grad_output_orig.dtype();

        // GPU fast path: dispatch to backend kernel (CUDA, Vulkan, ROCm, etc.)
        if (original_device.type != Device::Type::CPU) {
            auto go = grad_output_orig.contiguous();
            auto inp = input_orig.contiguous();
            auto rm = rrms_orig.contiguous();
            auto wt = weight_orig.contiguous();

            // Ensure all tensors share input's dtype — backend kernels
            // typically reinterpret pointers to a single dtype and would
            // throw "Type mismatch" if go/rm/wt differ from inp. Same root
            // cause as the RMSNorm forward fix.
            DType in_dt = inp.dtype();
            if (go.dtype() != in_dt) go = go.to(in_dt);
            if (rm.dtype() != in_dt) rm = rm.to(in_dt);
            if (wt.dtype() != in_dt) wt = wt.to(in_dt);

            NewOpAttributes attrs;
            attrs.set(AttrKey::NormalizedShape, std::to_string(normalized_size_));
            std::vector<Tensor> inputs_vec = {go, inp, rm, wt};
            auto results = dispatch<OpId::RMSNormBackward>(inputs_vec, attrs);
            return results;
        }

        // CPU path: keep Float64 in double precision. The previous unconditional
        // .to(Float32) plus the `float sum_grad_x` accumulator truncated F64
        // inputs and reduced in single precision (gradcheck drift). All
        // arithmetic below is in double; only the storage type T varies
        // (Float64 for F64 input, Float32 otherwise).
        const DType work_dtype =
            (original_dtype == DType::Float64) ? DType::Float64 : DType::Float32;
        auto grad_output = (grad_output_orig.device() == Device::cpu())
                          ? grad_output_orig.contiguous().to(work_dtype)
                          : grad_output_orig.contiguous().to(Device::cpu()).to(work_dtype);
        auto input = (input_orig.device() == Device::cpu())
                    ? input_orig.contiguous().to(work_dtype)
                    : input_orig.contiguous().to(Device::cpu()).to(work_dtype);
        auto rrms = (rrms_orig.device() == Device::cpu())
                   ? rrms_orig.contiguous().to(work_dtype)
                   : rrms_orig.contiguous().to(Device::cpu()).to(work_dtype);
        auto weight = (weight_orig.device() == Device::cpu())
                     ? weight_orig.contiguous().to(work_dtype)
                     : weight_orig.contiguous().to(Device::cpu()).to(work_dtype);

        auto shape = input.shape();
        int64_t batch_size = 1;
        for (size_t i = 0; i < shape.size() - 1; i++) {
            batch_size *= shape[i];
        }

        int64_t N = normalized_size_;

        // Allocate gradient tensors
        auto grad_input = zeros_like(input);
        auto grad_weight = zeros({N}, work_dtype, Device::cpu());

        // Compute gradients for each batch element
        // RMSNorm: y = x * rrms * weight, rrms = 1 / sqrt(mean(x^2) + eps)
        // d/dx[y] = weight*rrms - x*weight*rrms^3*mean(x)/N ; d/dweight[y] = x*rrms
        auto compute = [&]<typename T>() {
            const T* input_data = input.data<T>();
            const T* rrms_data = rrms.data<T>();
            const T* grad_out_data = grad_output.data<T>();
            const T* weight_data = weight.data<T>();
            T* grad_in_data = grad_input.data<T>();
            T* grad_weight_data = grad_weight.data<T>();

            for (int64_t b = 0; b < batch_size; b++) {
                double inv_rms = static_cast<double>(rrms_data[b]);
                double inv_rms_cubed = inv_rms * inv_rms * inv_rms;

                double sum_grad_x = 0.0;
                for (int64_t i = 0; i < N; i++) {
                    int64_t idx = b * N + i;
                    sum_grad_x += static_cast<double>(grad_out_data[idx]) *
                                  static_cast<double>(weight_data[i]) *
                                  static_cast<double>(input_data[idx]);
                }
                double correction = sum_grad_x * inv_rms_cubed / static_cast<double>(N);

                for (int64_t i = 0; i < N; i++) {
                    int64_t idx = b * N + i;
                    double x = static_cast<double>(input_data[idx]);
                    double normalized = x * inv_rms;
                    grad_weight_data[i] += static_cast<T>(
                        static_cast<double>(grad_out_data[idx]) * normalized);
                    grad_in_data[idx] = static_cast<T>(
                        static_cast<double>(grad_out_data[idx]) *
                            static_cast<double>(weight_data[i]) * inv_rms -
                        x * correction);
                }
            }
        };
        if (work_dtype == DType::Float64) compute.template operator()<double>();
        else compute.template operator()<float>();

        // Transfer gradients back to original device and dtype if needed
        Tensor grad_input_final = (original_device == Device::cpu())
                                 ? grad_input.contiguous().to(original_dtype)
                                 : grad_input.contiguous().to(original_device).to(original_dtype);
        Tensor grad_weight_final = (original_device == Device::cpu())
                                  ? grad_weight.contiguous().to(original_dtype)
                                  : grad_weight.contiguous().to(original_device).to(original_dtype);

        return {grad_input_final, grad_weight_final};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        // Full higher-order gradient support for RMSNorm.
        // Enables create_graph=true for second derivatives through RMSNorm.
        //
        // RMSNorm: y = x / rms * gamma, where rms = sqrt(mean(x^2) + eps)
        // saved: [0]=input, [1]=rrms (1/rms), [2]=weight (gamma)
        auto& grad_out = grad_outputs[0];

        Variable input_var, rrms_var, weight_var;
        if (has_saved_variables()) {
            const auto& sv = saved_variables();
            input_var = sv[0];
            rrms_var = sv[1];
            weight_var = sv[2];
        } else {
            auto saved = saved_tensors();
            input_var = Variable(saved[0], false);
            rrms_var = Variable(saved[1], false);
            weight_var = Variable(saved[2], false);
        }

        int64_t norm_size = normalized_size_;

        // rrms shape: input_shape[:-1], need to unsqueeze last dim for broadcasting
        auto rrms_bc = unsqueeze(rrms_var, -1);  // [..., 1]

        // normalized = x * rrms
        auto normalized = input_var * rrms_bc;

        // grad_x_hat = grad_output * weight
        auto grad_x_hat = grad_out * weight_var;

        // mean(grad_x_hat * normalized) over last dim
        auto mean_gxh_norm = sum(grad_x_hat * normalized, -1, true) / static_cast<float>(norm_size);

        // grad_input = (grad_x_hat - normalized * mean(grad_x_hat * normalized)) * rrms
        auto grad_input = (grad_x_hat - normalized * mean_gxh_norm) * rrms_bc;

        // grad_weight = sum(grad_output * normalized, dims except last)
        auto go_norm = grad_out * normalized;
        auto grad_weight_var = go_norm;
        auto gw_shape = grad_weight_var.shape();
        for (int d = static_cast<int>(gw_shape.size()) - 2; d >= 0; --d) {
            grad_weight_var = sum(grad_weight_var, d, false);
        }

        return {grad_input, grad_weight_var};
    }

    // Full Variable-level backward enables create_graph=true for second derivatives.
    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return false; }

private:
    double eps_;
    int64_t normalized_size_;
};

RMSNorm::RMSNorm(int64_t normalized_shape, double eps)
    : normalized_shape_(normalized_shape), eps_(eps) {

    weight_ = Variable(ones({normalized_shape_}), true);
    register_parameter("weight", weight_);
    // Cache pointer to avoid hash map lookups in forward pass (~2-3ms savings)
    cached_weight_ = parameters_["weight"];

    reset_parameters();
}

auto RMSNorm::forward_impl(const Variable& input_orig) -> Variable {
    // The CPU pointer/SIMD paths and the fused GPU kernel below all assume
    // contiguous row-major storage, indexing as input_data[b * N + i]. Mirror
    // LayerNorm::forward_impl and materialize a contiguous copy for non-contig
    // views (e.g. a transposed input), preserving the autograd graph. `input`
    // shadows `input_orig` so the rest of the body is untouched.
    Variable input = input_orig;
    if (!input_orig.tensor().is_contiguous()) {
        tenzor::utils::wrap_preserving_grad(input, input_orig.tensor().contiguous());
    }

    auto shape = input.shape();

    // Verify that input's last dimension matches normalized_shape
    if (shape.empty() || shape.back() != normalized_shape_) {
        throw std::runtime_error("Input's last dimension (" +
                               std::to_string(shape.empty() ? 0 : shape.back()) +
                               ") doesn't match normalized_shape (" +
                               std::to_string(normalized_shape_) + ")");
    }

    // Calculate batch size (all dimensions except the last)
    int64_t batch_size = 1;
    for (size_t i = 0; i < shape.size() - 1; i++) {
        batch_size *= shape[i];
    }

    int64_t N = normalized_shape_;

    // ============================================================================
    // FAST INFERENCE PATH: Skip all autograd overhead when gradients not needed
    // ============================================================================
    // NOTE: For inference (input doesn't require grad), use fast path regardless of
    // whether weights require grad - weight gradients only matter during training
    const bool needs_input_grad = is_grad_enabled() && input.requires_grad();
    const bool needs_grad = needs_input_grad ||
        (is_grad_enabled() && cached_weight_ && cached_weight_->requires_grad());

    // ============================================================================
    // CUDA FAST PATH: Use fused RMSNorm kernel via dispatch (single kernel launch!)
    // Use fast path when input doesn't need grad (inference) - even if weights require grad
    // ============================================================================
    if (!needs_input_grad && input.tensor().device().type == Device::Type::CUDA && input.tensor().dtype() == DType::Float32) {
        const Tensor& x = input.tensor();

        // Get weight tensor, ensure on CUDA
        Tensor weight_cuda = cached_weight_ ? cached_weight_->tensor() : weight_.tensor();
        if (weight_cuda.device().type != Device::Type::CUDA) {
            weight_cuda = weight_cuda.to(Device::cuda());
        }

        // Dispatch to fused CUDA kernel (single kernel launch for max performance)
        NewOpAttributes attrs;
        attrs.set(AttrKey::Eps, eps_);

        std::vector<Tensor> inputs_vec = {x, weight_cuda};
        auto results = dispatch<OpId::FusedRMSNorm>(inputs_vec, attrs);
        // BB.16: inference fast paths still read results[0]; require at
        // least the output tensor (full {output, rrms} contract is only
        // needed by the training path that wires up the backward).
        check_norm_outputs(results, "RMSNorm CUDA inference", /*min_required=*/1);
        jit_record_rms_norm(x, weight_cuda, results[0], eps_, normalized_shape_);
        return Variable(results[0], false);  // results[0] is output, [1] is rrms
    }

    // Vulkan fast path: dispatch to GPU RMSNorm shader, no autograd wrapper.
    // E.11: this branch is an inference-time optimization (skips Variable
    // wrapping) — training/grad-tracked calls fall through to the standard
    // path below which also uses Vulkan's FusedRMSNorm but plumbs the
    // result through RMSNormBackward for full autograd support.
    if (!needs_input_grad && input.tensor().device().type == Device::Type::Vulkan) {
        const Tensor& x = input.tensor();

        Tensor weight_vk = cached_weight_ ? cached_weight_->tensor() : weight_.tensor();
        if (weight_vk.device().type != Device::Type::Vulkan) {
            weight_vk = weight_vk.to(input.tensor().device());
        }

        NewOpAttributes attrs;
        attrs.set(AttrKey::Eps, eps_);

        std::vector<Tensor> inputs_vec = {x, weight_vk};
        auto results = dispatch<OpId::FusedRMSNorm>(inputs_vec, attrs);
        // BB.16: same as the CUDA inference fast path above — guard the
        // read of results[0] so a regressing backend surfaces here.
        check_norm_outputs(results, "RMSNorm Vulkan inference", /*min_required=*/1);
        jit_record_rms_norm(x, weight_vk, results[0], eps_, normalized_shape_);
        return Variable(results[0], false);
    }

    if (!needs_input_grad && input.tensor().device().type == Device::Type::CPU && input.tensor().dtype() == DType::Float32) {
        // Ultra-fast path: CPU Float32 inference with no gradient tracking
        const Tensor& input_tensor = input.tensor();
        const auto* input_data = input_tensor.data<float>();

        // Get weight pointer directly from cache (avoids hash map lookups)
        const Tensor& weight_tensor = cached_weight_ ? cached_weight_->tensor() : weight_.tensor();
        const auto* weight_data = weight_tensor.data<float>();

        // Allocate only output tensor
        auto output = Tensor::empty_uninitialized(
            {input_tensor.shape().begin(), input_tensor.shape().end()},
            DType::Float32, Device::cpu());
        auto* output_data = output.data<float>();

        // Scratch buffer for per-row rrms. The stack array (4KB at threshold
        // 1024) is reserved only inside the small-batch branch, so the heap
        // path for large batches no longer pays any stack cost. Threshold
        // matches the LayerNorm fast path.
        constexpr int64_t STACK_THRESHOLD = 1024;

        auto run_rms_norm = [&](float* rrms_scratch) {
#if defined(__x86_64__) || defined(_M_X64)
            fused_rms_norm_f32(
                input_data, weight_data, output_data, rrms_scratch,
                batch_size, N, static_cast<float>(eps_)
            );
#else
            const int nthreads = get_optimal_threads();
            #pragma omp parallel for num_threads(nthreads)
            for (int64_t b = 0; b < batch_size; b++) {
                float sum_sq = 0.0f;
                for (int64_t i = 0; i < N; i++) {
                    float val = input_data[b * N + i];
                    sum_sq += val * val;
                }
                float rrms = 1.0f / std::sqrt(sum_sq / N + static_cast<float>(eps_));
                rrms_scratch[b] = rrms;

                for (int64_t i = 0; i < N; i++) {
                    int64_t idx = b * N + i;
                    output_data[idx] = input_data[idx] * rrms * weight_data[i];
                }
            }
#endif
        };

        if (batch_size <= STACK_THRESHOLD) {
            float stack_rrms[STACK_THRESHOLD];
            run_rms_norm(stack_rrms);
        } else {
            std::unique_ptr<float[]> rrms_scratch(new float[batch_size]);
            run_rms_norm(rrms_scratch.get());
        }

        jit_record_rms_norm(input.tensor(), weight_tensor, output, eps_,
                            normalized_shape_);
        return Variable(output, false);
    }

    // ============================================================================
    // STANDARD PATH: Full autograd support
    // ============================================================================

    Device original_device = input.tensor().device();

    // Dispatch through OpId::FusedRMSNorm on every device including CPU
    // (cpu::fused_rms_norm_kernel-equivalent — registered in
    // cpu_kernel_registry.cpp). The old CPU-only hand-rolled pointer-based
    // "CPU path" that followed this block bypassed dispatch() entirely,
    // making CPU RMSNorm (in training, or in any non-Float32 dtype)
    // invisible to the JIT tracer — same bug class as JIT-R046/R047, fixed
    // here for consistency even though RMSNorm wasn't separately flagged.
    {
        const Tensor& x = input.tensor();

        Tensor weight_dev = cached_weight_ ? cached_weight_->tensor() : weight_.tensor();
        if (weight_dev.device() != original_device) {
            weight_dev = weight_dev.to(original_device);
        }
        // Ensure weight dtype matches input dtype — backends like ROCm's
        // fused_rms_norm_hip take the Float32 fast-path when input is
        // Float32 and then call weight.data<float>(), throwing if weight
        // is still Float16/BFloat16. Also matches CPU/CUDA expectations.
        if (weight_dev.dtype() != input.tensor().dtype()) {
            weight_dev = weight_dev.to(input.tensor().dtype());
        }

        NewOpAttributes attrs;
        attrs.set(AttrKey::Eps, eps_);

        std::vector<Tensor> inputs_vec = {x, weight_dev};
        auto results = dispatch<OpId::FusedRMSNorm>(inputs_vec, attrs);

        // BB.16: shared helper. RMSNorm only requires {output, rrms} (no
        // mean), so pass min_required=2. Without this guard, the backward
        // cannot reconstruct the 1/sqrt(mean(x^2)+eps) factor and will
        // silently miscompute or read past the saved-tensor vector.
        check_norm_outputs(results, "RMSNorm (FusedRMSNorm)", /*min_required=*/2);
        Tensor output = results[0];
        Tensor saved_rrms = results[1];
        jit_record_rms_norm(x, weight_dev, output, eps_, normalized_shape_);

        if (needs_grad) {
            auto result = Variable(output, true);

            std::vector<Tensor> tensors_to_save = {
                input.tensor(),
                saved_rrms,
                cached_weight_ ? cached_weight_->tensor() : weight_.tensor()
            };

            auto grad_fn = std::make_shared<RMSNormBackward>(
                eps_, N, std::move(tensors_to_save)
            );

            result.set_grad_fn(grad_fn);

            std::vector<std::shared_ptr<Function>> next_funcs;
            if (auto input_grad_fn = input.grad_fn()) {
                next_funcs.push_back(input_grad_fn);
            }
            if (cached_weight_ && cached_weight_->requires_grad()) {
                if (auto weight_grad_fn = cached_weight_->grad_fn()) {
                    next_funcs.push_back(weight_grad_fn);
                }
            }
            grad_fn->set_next_functions(next_funcs);

            std::vector<Variable> input_vars;
            if (input.requires_grad()) {
                input_vars.push_back(input);
            }
            if (cached_weight_ && cached_weight_->requires_grad()) {
                input_vars.push_back(*cached_weight_);
            }
            grad_fn->set_input_variables(input_vars);

            return result;
        }

        return Variable(output, false);
    }

    // Every valid RMSNorm dtype (Float32/Float64/Float16/BFloat16) is
    // handled by the dispatch block above; this exists only to fail loudly
    // for an actually-unsupported dtype (mirroring the pre-existing
    // hand-rolled CPU path's own final `else` branch, removed above as
    // dead code once the dispatch block started covering every device).
    throw std::runtime_error(
        "RMSNorm only supports Float16, BFloat16, Float32, and Float64 dtypes");
}

auto RMSNorm::reset_parameters() -> void {
    // Weight initialized to 1 (already done in constructor)
}

// ============================================================================
// LocalResponseNorm implementation
// ============================================================================

LocalResponseNorm::LocalResponseNorm(int64_t size, double alpha, double beta, double k)
    : size_(size), alpha_(alpha), beta_(beta), k_(k) {
    if (size <= 0) throw std::runtime_error("LocalResponseNorm: size must be positive");
}

auto LocalResponseNorm::forward_impl(const Variable& input) -> Variable {
    // LRN: y_i = x_i / (k + alpha/size * sum(x_j^2, local_window))^beta
    // where the window spans exactly `size` channels around channel i
    // (symmetric for odd size; asymmetric — one extra lower channel — for even).
    //
    // Implementation strategy: compute x^2, then accumulate the sliding
    // channel-window sum by iterating over offsets and using narrow + cat
    // to build shifted versions with zero-padding. For typical LRN window
    // sizes (3, 5, 7) this is only a few additions through the autograd graph.

    auto input_shape = input.shape();
    if (input_shape.size() < 3) {
        throw std::invalid_argument(
            "LocalResponseNorm expects at least 3D input [batch, channels, ...]");
    }

    int64_t C = input_shape[1];
    // F128: the cross-channel window must span EXACTLY `size` channels to match
    // PyTorch (F.local_response_norm, an avg_pool1d over squared channels with
    // kernel=size, front-pad=size/2, back-pad=(size-1)/2, count_include_pad).
    // For odd `size` the window is symmetric ([-size/2, +size/2]); for even
    // `size` it is asymmetric — one extra channel on the lower-index side — so
    // the total is `size`, not `size+1`. Using half=size/2 on both sides summed
    // size+1 channels for even sizes, diverging from PyTorch and the functional
    // path. window_lo = size/2 (front pad), window_hi = (size-1)/2 (back pad).
    int64_t window_lo = size_ / 2;
    int64_t window_hi = (size_ - 1) / 2;

    // Compute the channel-window sum at the Variable level so backward()
    // through LRN actually populates input.grad. The previous implementation
    // operated on raw Tensors ("LRN has no learnable parameters") which
    // severed the graph at the input edge — the test
    // tests/nn/layers/test_local_response_norm_multidtype.cpp asserts the
    // broken behaviour as a reminder; update that test once this lands.
    auto dev = input.tensor().device();
    auto dt = input.tensor().dtype();

    Variable x_sq = input * input;
    Variable sum_sq = x_sq;  // j=0 contribution

    for (int64_t j = -window_lo; j <= window_hi; ++j) {
        if (j == 0) continue;

        int64_t src_start = std::max(int64_t(0), j);
        int64_t slice_len = C - std::abs(j);
        if (slice_len <= 0) continue;

        Variable sliced = ::tenzor::slice(x_sq, 1, src_start, src_start + slice_len);

        auto pad_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
        pad_shape[1] = std::abs(j);
        Variable pad(tenzor::zeros(pad_shape, dt, dev), false);

        std::vector<Variable> parts;
        if (j > 0) {
            parts = {sliced, pad};
        } else {
            parts = {pad, sliced};
        }
        Variable shifted = ::tenzor::cat(parts, 1);

        sum_sq = sum_sq + shifted;
    }

    // divisor = (k + alpha/size * sum_sq) ^ beta  — all Variable-level
    float alpha_over_size = static_cast<float>(alpha_) / static_cast<float>(size_);
    auto pow_base = sum_sq * alpha_over_size + static_cast<float>(k_);
    auto divisor = ::tenzor::pow(pow_base, static_cast<float>(beta_));

    return input / divisor;
}

} // namespace tenzor::nn
