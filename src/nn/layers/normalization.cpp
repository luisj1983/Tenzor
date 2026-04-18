#include "tenzor/nn/layers/normalization.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include <cmath>
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

// ============================================================================
// SIMD-Optimized Fused LayerNorm (computes output + mean + rstd in single pass)
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64)
// Fused LayerNorm that computes output AND saves mean/rstd for backward pass
// Uses 2-pass algorithm: (1) sum + sum_sq, (2) normalize
// var = sum((x - mean)^2) / N (numerically stable two-pass algorithm)
static void fused_layer_norm_f32(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float* __restrict__ mean_out,
    float* __restrict__ rstd_out,
    int64_t batch_size,
    int64_t norm_size,
    float eps)
{
    const float inv_n = 1.0f / static_cast<float>(norm_size);

    // LayerNorm is memory-bound, so parallelization only helps for very large data
    // Single-threaded SIMD achieves ~50 GB/s, close to single-core memory bandwidth
    // Parallelization overhead is ~100-500us, so only use for >4M elements (~16MB)
    // Also limit thread count to avoid cache thrashing
    const int64_t total_elements = batch_size * norm_size;
    const bool use_parallel = total_elements > 4 * 1024 * 1024;  // 4M elements
    const int max_threads = use_parallel ? std::min(4, static_cast<int>(batch_size / 256)) : 1;

    #pragma omp parallel for if(use_parallel) num_threads(std::max(1, max_threads))
    for (int64_t b = 0; b < batch_size; b++) {
        const float* in_ptr = input + b * norm_size;
        float* out_ptr = output + b * norm_size;

#if defined(__AVX512F__)
        // AVX-512 path: 16 floats at a time, numerically stable two-pass algorithm
        // Pass 1: compute mean
        __m512 vsum = _mm512_setzero_ps();
        int64_t i = 0;

        for (; i + 16 <= norm_size; i += 16) {
            _mm_prefetch(reinterpret_cast<const char*>(in_ptr + i + 64), _MM_HINT_T0);
            __m512 x = _mm512_loadu_ps(in_ptr + i);
            vsum = _mm512_add_ps(vsum, x);
        }

        float sum = _mm512_reduce_add_ps(vsum);
        for (; i < norm_size; i++) {
            sum += in_ptr[i];
        }

        float mean = sum * inv_n;

        // Pass 2: compute variance = sum((x - mean)^2) / N
        __m512 vmean = _mm512_set1_ps(mean);
        __m512 vvar = _mm512_setzero_ps();
        i = 0;

        for (; i + 16 <= norm_size; i += 16) {
            __m512 x = _mm512_loadu_ps(in_ptr + i);
            __m512 diff = _mm512_sub_ps(x, vmean);
            vvar = _mm512_fmadd_ps(diff, diff, vvar);
        }

        float var = _mm512_reduce_add_ps(vvar);
        for (; i < norm_size; i++) {
            float diff = in_ptr[i] - mean;
            var += diff * diff;
        }
        var *= inv_n;
        float rstd = 1.0f / std::sqrt(var + eps);

        // Save for backward
        mean_out[b] = mean;
        rstd_out[b] = rstd;

        // Pass 3: normalize and apply affine transform
        __m512 vrstd = _mm512_set1_ps(rstd);
        i = 0;

        for (; i + 16 <= norm_size; i += 16) {
            _mm_prefetch(reinterpret_cast<const char*>(in_ptr + i + 64), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(weight + i + 64), _MM_HINT_T0);
            __m512 x = _mm512_loadu_ps(in_ptr + i);
            __m512 w = _mm512_loadu_ps(weight + i);
            __m512 bb = _mm512_loadu_ps(bias + i);
            __m512 norm = _mm512_mul_ps(_mm512_sub_ps(x, vmean), vrstd);
            __m512 result = _mm512_fmadd_ps(norm, w, bb);
            _mm512_storeu_ps(out_ptr + i, result);
        }

        // Handle remainder
        for (; i < norm_size; i++) {
            float norm_val = (in_ptr[i] - mean) * rstd;
            out_ptr[i] = norm_val * weight[i] + bias[i];
        }

#elif defined(__AVX2__)
        // AVX2 path: 8 floats at a time, numerically stable two-pass algorithm
        // Pass 1: compute mean
        __m256 vsum = _mm256_setzero_ps();
        int64_t i = 0;

        for (; i + 8 <= norm_size; i += 8) {
            _mm_prefetch(reinterpret_cast<const char*>(in_ptr + i + 32), _MM_HINT_T0);
            __m256 x = _mm256_loadu_ps(in_ptr + i);
            vsum = _mm256_add_ps(vsum, x);
        }

        // Horizontal sum for AVX2
        __m128 sum_lo = _mm256_castps256_ps128(vsum);
        __m128 sum_hi = _mm256_extractf128_ps(vsum, 1);
        __m128 sum128 = _mm_add_ps(sum_lo, sum_hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float sum = _mm_cvtss_f32(sum128);

        for (; i < norm_size; i++) {
            sum += in_ptr[i];
        }

        float mean = sum * inv_n;

        // Pass 2: compute variance = sum((x - mean)^2) / N
        __m256 vmean = _mm256_set1_ps(mean);
        __m256 vvar = _mm256_setzero_ps();
        i = 0;

        for (; i + 8 <= norm_size; i += 8) {
            __m256 x = _mm256_loadu_ps(in_ptr + i);
            __m256 diff = _mm256_sub_ps(x, vmean);
            vvar = _mm256_fmadd_ps(diff, diff, vvar);
        }

        __m128 var_lo = _mm256_castps256_ps128(vvar);
        __m128 var_hi = _mm256_extractf128_ps(vvar, 1);
        __m128 var128 = _mm_add_ps(var_lo, var_hi);
        var128 = _mm_hadd_ps(var128, var128);
        var128 = _mm_hadd_ps(var128, var128);
        float var = _mm_cvtss_f32(var128);

        for (; i < norm_size; i++) {
            float diff = in_ptr[i] - mean;
            var += diff * diff;
        }
        var *= inv_n;
        float rstd = 1.0f / std::sqrt(var + eps);

        // Save for backward
        mean_out[b] = mean;
        rstd_out[b] = rstd;

        // Pass 3: normalize and apply affine transform
        __m256 vrstd = _mm256_set1_ps(rstd);
        i = 0;

        for (; i + 8 <= norm_size; i += 8) {
            _mm_prefetch(reinterpret_cast<const char*>(in_ptr + i + 32), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(weight + i + 32), _MM_HINT_T0);
            __m256 x = _mm256_loadu_ps(in_ptr + i);
            __m256 w = _mm256_loadu_ps(weight + i);
            __m256 bb = _mm256_loadu_ps(bias + i);
            __m256 norm = _mm256_mul_ps(_mm256_sub_ps(x, vmean), vrstd);
            __m256 result = _mm256_fmadd_ps(norm, w, bb);
            _mm256_storeu_ps(out_ptr + i, result);
        }

        // Handle remainder
        for (; i < norm_size; i++) {
            float norm_val = (in_ptr[i] - mean) * rstd;
            out_ptr[i] = norm_val * weight[i] + bias[i];
        }

#else
        // Scalar fallback - numerically stable two-pass algorithm
        // Pass 1: compute mean
        float sum = 0.0f;
        for (int64_t i = 0; i < norm_size; i++) {
            sum += in_ptr[i];
        }
        float mean = sum * inv_n;

        // Pass 2: compute variance = sum((x - mean)^2) / N
        float var = 0.0f;
        for (int64_t i = 0; i < norm_size; i++) {
            float diff = in_ptr[i] - mean;
            var += diff * diff;
        }
        var *= inv_n;
        float rstd = 1.0f / std::sqrt(var + eps);

        mean_out[b] = mean;
        rstd_out[b] = rstd;

        // Pass 3: normalize and apply affine transform
        for (int64_t i = 0; i < norm_size; i++) {
            float norm_val = (in_ptr[i] - mean) * rstd;
            out_ptr[i] = norm_val * weight[i] + bias[i];
        }
#endif
    }

}

// ============================================================================
// SIMD-Optimized Fused RMSNorm (simpler than LayerNorm - no mean computation)
// ============================================================================

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
    const int nthreads = get_optimal_threads();

    // Use fewer threads for memory-bound operations to avoid synchronization overhead
    const int effective_threads = std::min({nthreads, static_cast<int>(batch_size / 128), 4});
    const int final_threads = std::max(1, effective_threads);
    #pragma omp parallel for num_threads(final_threads)
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

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("LayerNormBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& grad_output_orig = grad_outputs[0];
        auto saved = saved_tensors();
        auto& input_orig = saved[0];
        auto& mean_orig = saved[1];
        auto& rstd_orig = saved[2];  // reciprocal std (1 / sqrt(var + eps))
        auto& weight_orig = saved[3];

        // Save original device and dtype for dispatch routing
        Device original_device = input_orig.device();
        DType original_dtype = grad_output_orig.dtype();

        // GPU fast path: dispatch to backend kernel (CUDA, Vulkan, ROCm, etc.)
        if (original_device.type != Device::Type::CPU) {
            auto go = grad_output_orig.contiguous();
            auto inp = input_orig.contiguous();
            auto mn = mean_orig.contiguous();
            auto rs = rstd_orig.contiguous();
            auto wt = weight_orig.contiguous();

            // For Float16/BFloat16, upcast all tensors to Float32 for backward
            // computation to prevent gradient overflow. The CUDA mixed kernel
            // computes internally in Float32 but writes __half output, losing
            // values > 65504 to Inf. This matches CPU backward behavior.
            DType orig_dt = inp.dtype();
            bool needs_upcast = (orig_dt == DType::Float16 || orig_dt == DType::BFloat16);
            if (needs_upcast) {
                go = go.to(DType::Float32);
                inp = inp.to(DType::Float32);
                mn = mn.to(DType::Float32);
                rs = rs.to(DType::Float32);
                wt = wt.to(DType::Float32);
            } else {
                // GPU backward kernels read all tensors with the same dtype as input
                // (e.g., CUDA Float64 kernel reinterpret_casts to double*).
                // Mixed precision (e.g. Float32 weights with Float64 input) requires
                // converting all tensors to match input dtype before dispatch.
                if (go.dtype() != inp.dtype()) go = go.to(inp.dtype());
                if (mn.dtype() != inp.dtype()) mn = mn.to(inp.dtype());
                if (rs.dtype() != inp.dtype()) rs = rs.to(inp.dtype());
                if (wt.dtype() != inp.dtype()) wt = wt.to(inp.dtype());
            }

            NewOpAttributes attrs;
            attrs.set(AttrKey::NormalizedShape, std::to_string(normalized_size_));
            // Standard order: [grad_output, input, mean, inv_std, weight]
            std::vector<Tensor> inputs_vec = {go, inp, mn, rs, wt};
            auto results = dispatch<OpId::LayerNormBackward>(inputs_vec, attrs);

            // Convert grad_input back to original dtype but keep grad_weight
            // and grad_bias in Float32.  Accumulated parameter gradients can
            // exceed Float16 range (~65504), so downcasting them would produce
            // Inf.  This matches CPU behavior which always stores these in F32.
            if (needs_upcast && results.size() >= 1) {
                results[0] = results[0].to(orig_dt);  // grad_input
            }
            return results;
        }

        // CPU path: pointer-based access
        // Convert to Float32 for computation
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

        // Compute normalized input: (x - mean) * rstd
        auto* input_data = input.data<float>();
        auto* mean_data = mean.data<float>();
        auto* rstd_data = rstd.data<float>();
        auto* grad_out_data = grad_output.data<float>();
        auto* weight_data = weight.data<float>();

        // Allocate gradient tensors on same device as input
        // Use the contiguous tensor's device to ensure consistency
        auto grad_input = zeros_like(input);
        auto grad_weight = zeros({N}, grad_output.dtype(), grad_output.device());
        auto grad_bias = zeros({N}, grad_output.dtype(), grad_output.device());

        auto* grad_in_data = grad_input.data<float>();
        auto* grad_weight_data = grad_weight.data<float>();
        auto* grad_bias_data = grad_bias.data<float>();

        // Compute gradients for each batch element
        for (int64_t b = 0; b < batch_size; b++) {
            float mu = mean_data[b];
            float inv_std = rstd_data[b];

            // Check if variance is essentially zero (which causes numerical instability)
            // When all inputs are identical (variance ~0), gradients should be zero
            // because small changes to input don't change the normalized output
            float inv_n = 1.0f / static_cast<float>(N);
            float sum_sq = 0.0f;
            for (int64_t i = 0; i < N; i++) {
                sum_sq += input_data[b * N + i] * input_data[b * N + i];
            }
            float var = (sum_sq * inv_n) - (mu * mu);
            bool zero_variance = (var < static_cast<float>(eps_));

            if (zero_variance) {
                // With zero variance, input gradients should be zero
                for (int64_t i = 0; i < N; i++) {
                    grad_in_data[b * N + i] = 0.0f;
                }
                // Weight and bias gradients are still computed normally from output gradients
                for (int64_t i = 0; i < N; i++) {
                    int64_t idx = b * N + i;
                    float x_normalized = 0.0f;  // (input - mean) * large_inv_std, but input≈mean
                    grad_weight_data[i] += grad_out_data[idx] * x_normalized;  // Will be ~0
                    grad_bias_data[i] += grad_out_data[idx];
                }
                continue;
            }

            // Normal case: compute gradients with stable variance
            // Compute intermediate sums for this batch element
            float sum_grad_out = 0.0f;
            float sum_grad_out_normalized = 0.0f;

            for (int64_t i = 0; i < N; i++) {
                int64_t idx = b * N + i;
                float x_normalized = (input_data[idx] - mu) * inv_std;
                float grad_out = grad_out_data[idx] * weight_data[i];

                sum_grad_out += grad_out;
                sum_grad_out_normalized += grad_out * x_normalized;

                // Accumulate weight and bias gradients
                grad_weight_data[i] += grad_out_data[idx] * x_normalized;
                grad_bias_data[i] += grad_out_data[idx];
            }

            // Compute input gradients using the formula:
            // grad_input = (grad_out - mean(grad_out) - normalized * mean(grad_out * normalized)) * rstd * weight
            float mean_grad_out = sum_grad_out / N;
            float mean_grad_out_normalized = sum_grad_out_normalized / N;

            for (int64_t i = 0; i < N; i++) {
                int64_t idx = b * N + i;
                float x_normalized = (input_data[idx] - mu) * inv_std;

                grad_in_data[idx] = (grad_out_data[idx] * weight_data[i] - mean_grad_out -
                                    x_normalized * mean_grad_out_normalized) * inv_std;
            }
        }

        // Convert gradients back to original dtype (already on CPU)
        return {grad_input.contiguous().to(original_dtype),
                grad_weight.contiguous().to(original_dtype),
                grad_bias.contiguous().to(original_dtype)};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        // Higher-order gradient support for LayerNorm.
        // LayerNorm normalizes over the last dimension(s) of size normalized_size_.
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

        int64_t norm_size = normalized_size_;

        // Expand mean and rstd to match input shape (add trailing dim)
        // mean/rstd shape: input_shape[:-1], need to unsqueeze last dim
        auto mean_bc = unsqueeze(mean_var, -1);    // [..., 1]
        auto rstd_bc = unsqueeze(rstd_var, -1);    // [..., 1]

        // x_hat = (input - mean) * rstd
        auto x_hat = (input_var - mean_bc) * rstd_bc;

        // grad_x_hat = grad_output * weight
        auto grad_x_hat = grad_out * weight_var;

        // mean(grad_x_hat) over last dim
        auto mean_gxh = sum(grad_x_hat, -1, true) / static_cast<float>(norm_size);

        // mean(grad_x_hat * x_hat) over last dim
        auto mean_gxh_xh = sum(grad_x_hat * x_hat, -1, true) / static_cast<float>(norm_size);

        // grad_input = (grad_x_hat - mean_gxh - x_hat * mean_gxh_xh) * rstd
        auto grad_input = (grad_x_hat - mean_gxh - x_hat * mean_gxh_xh) * rstd_bc;

        // grad_weight = sum(grad_output * x_hat, dims except last)
        // Flatten batch dims and sum over all except last
        auto go_xhat = grad_out * x_hat;
        // Sum over all dims except the last (normalized dim)
        auto grad_weight_var = go_xhat;
        auto gw_shape = grad_weight_var.shape();
        for (int d = static_cast<int>(gw_shape.size()) - 2; d >= 0; --d) {
            grad_weight_var = sum(grad_weight_var, d, false);
        }

        // grad_bias = sum(grad_output, dims except last)
        auto grad_bias_var = grad_out;
        auto gb_shape = grad_bias_var.shape();
        for (int d = static_cast<int>(gb_shape.size()) - 2; d >= 0; --d) {
            grad_bias_var = sum(grad_bias_var, d, false);
        }

        return {grad_input, grad_weight_var, grad_bias_var};
    }

    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return false; }

private:
    bool elementwise_affine_;
    double eps_;
    int64_t normalized_size_;
};

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

auto LayerNorm::forward_impl(const Variable& input) -> Variable {
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

    // Calculate batch size (all dimensions before normalized dimensions)
    int64_t batch_size = 1;
    for (size_t i = 0; i < norm_start; i++) {
        batch_size *= shape[i];
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
    // CUDA FAST PATH: Use fused LayerNorm kernel via dispatch (single kernel launch!)
    // Use fast path when input doesn't need grad (inference) - even if weights require grad
    // ============================================================================
    if (!needs_input_grad && input.tensor().device().type == Device::Type::CUDA && input.tensor().dtype() == DType::Float32) {
        const Tensor& x = input.tensor();

        // Get weight/bias tensors, ensure on CUDA
        Tensor weight_cuda = (elementwise_affine_ && cached_weight_) ? cached_weight_->tensor() : weight_.tensor();
        Tensor bias_cuda = (elementwise_affine_ && cached_bias_) ? cached_bias_->tensor() : bias_.tensor();

        if (weight_cuda.device().type != Device::Type::CUDA) {
            weight_cuda = weight_cuda.to(Device::cuda());
        }
        if (bias_cuda.device().type != Device::Type::CUDA) {
            bias_cuda = bias_cuda.to(Device::cuda());
        }

        // Build normalized_shape as comma-separated string for attrs
        std::string norm_shape_str;
        for (size_t i = 0; i < normalized_shape_.size(); ++i) {
            if (i > 0) norm_shape_str += ",";
            norm_shape_str += std::to_string(normalized_shape_[i]);
        }

        // Dispatch to fused CUDA kernel (single kernel launch for max performance)
        NewOpAttributes attrs;
        attrs.set(AttrKey::NormalizedShape, std::string_view(norm_shape_str));
        attrs.set(AttrKey::Eps, static_cast<double>(eps_));

        std::vector<Tensor> inputs_vec = {x, weight_cuda, bias_cuda};
        auto results = dispatch<OpId::FusedLayerNorm>(inputs_vec, attrs);
        return Variable(results[0], false);  // results[0] is output, [1] is mean, [2] is inv_std
    }

    if (!needs_input_grad && input.tensor().device().type == Device::Type::CPU && input.tensor().dtype() == DType::Float32) {
        // Ultra-fast path: CPU Float32 inference with no gradient tracking
        const Tensor& input_tensor = input.tensor();
        const auto* input_data = input_tensor.data<float>();

        // Get weight/bias from cached pointers (avoids hash map lookups)
        const Tensor& weight_tensor = (elementwise_affine_ && cached_weight_) ? cached_weight_->tensor() : weight_.tensor();
        const Tensor& bias_tensor = (elementwise_affine_ && cached_bias_) ? cached_bias_->tensor() : bias_.tensor();
        const auto* weight_data = weight_tensor.data<float>();
        const auto* bias_data = bias_tensor.data<float>();

        // Allocate only output tensor (no statistics tensors needed)
        auto output = Tensor::empty_uninitialized(
            {input_tensor.shape().begin(), input_tensor.shape().end()},
            DType::Float32, Device::cpu());
        auto* output_data = output.data<float>();

        // Use thread-local scratch buffers for statistics to avoid allocation
        // For small batch sizes, use stack allocation
        constexpr int64_t STACK_THRESHOLD = 8192;
        float stack_mean[STACK_THRESHOLD];
        float stack_rstd[STACK_THRESHOLD];

        float* mean_scratch = (batch_size <= STACK_THRESHOLD) ? stack_mean : new float[batch_size];
        float* rstd_scratch = (batch_size <= STACK_THRESHOLD) ? stack_rstd : new float[batch_size];

#if defined(__x86_64__) || defined(_M_X64)
        fused_layer_norm_f32(
            input_data, weight_data, bias_data,
            output_data, mean_scratch, rstd_scratch,
            batch_size, N, static_cast<float>(eps_)
        );
#else
        const int nthreads = get_optimal_threads();
        #pragma omp parallel for num_threads(nthreads)
        for (int64_t b = 0; b < batch_size; b++) {
            float sum = 0.0f;
            for (int64_t i = 0; i < N; i++) {
                sum += input_data[b * N + i];
            }
            float mean = sum / N;
            mean_scratch[b] = mean;

            float sum_sq = 0.0f;
            for (int64_t i = 0; i < N; i++) {
                float diff = input_data[b * N + i] - mean;
                sum_sq += diff * diff;
            }
            float rs = 1.0f / std::sqrt(sum_sq / N + static_cast<float>(eps_));
            rstd_scratch[b] = rs;

            for (int64_t i = 0; i < N; i++) {
                int64_t idx = b * N + i;
                output_data[idx] = (input_data[idx] - mean) * rs * weight_data[i] + bias_data[i];
            }
        }
#endif

        // Clean up heap allocation if used
        if (batch_size > STACK_THRESHOLD) {
            delete[] mean_scratch;
            delete[] rstd_scratch;
        }

        return Variable(output, false);
    }

    // ============================================================================
    // STANDARD PATH: Full autograd support with device transfers
    // ============================================================================

    // Save original device and move input to CPU for pointer-based computation
    Device original_device = input.tensor().device();
    Tensor input_cpu = (original_device == Device::cpu()) ? input.tensor() : input.tensor().to(Device::cpu());

    // Get weight/bias from cached pointers (faster) or fallback to parameters_ for hooks
    // Note: cached pointers point to same shared_ptr as parameters_["weight"]
    Tensor weight_cpu = (elementwise_affine_ && cached_weight_) ? cached_weight_->tensor() : weight_.tensor();
    Tensor bias_cpu = (elementwise_affine_ && cached_bias_) ? cached_bias_->tensor() : bias_.tensor();
    if (elementwise_affine_) {
        if (weight_cpu.device() != Device::cpu()) {
            weight_cpu = weight_cpu.to(Device::cpu());
        }
        if (weight_cpu.dtype() != input_cpu.dtype()) {
            weight_cpu = weight_cpu.to(input_cpu.dtype());
        }
        if (bias_cpu.device() != Device::cpu()) {
            bias_cpu = bias_cpu.to(Device::cpu());
        }
        if (bias_cpu.dtype() != input_cpu.dtype()) {
            bias_cpu = bias_cpu.to(input_cpu.dtype());
        }
    }

    // Dtype-aware computation
    DType input_dtype = input_cpu.dtype();

    if (input_dtype == DType::BFloat16) {
        // BFloat16: convert to Float32, compute, convert back
        // Uses same approach as Float16 - all computation in Float32 for precision
        auto batch_mean = zeros({batch_size}, DType::Float32, Device::cpu());
        auto batch_var = zeros({batch_size}, DType::Float32, Device::cpu());
        auto* mean_data = batch_mean.data<float>();
        auto* var_data = batch_var.data<float>();

        auto* input_data = input_cpu.data<BFloat16>();

        for (int64_t b = 0; b < batch_size; b++) {
            double sum = 0.0;
            for (int64_t i = 0; i < N; i++) {
                sum += static_cast<float>(input_data[b * N + i]);
            }
            mean_data[b] = static_cast<float>(sum / N);
        }

        for (int64_t b = 0; b < batch_size; b++) {
            double sum_sq = 0.0;
            float mu = mean_data[b];
            for (int64_t i = 0; i < N; i++) {
                float diff = static_cast<float>(input_data[b * N + i]) - mu;
                sum_sq += diff * diff;
            }
            var_data[b] = static_cast<float>(sum_sq / N);
        }

        auto rstd = zeros({batch_size}, DType::Float32, Device::cpu());
        auto* rstd_data = rstd.data<float>();
        for (int64_t b = 0; b < batch_size; b++) {
            rstd_data[b] = 1.0f / std::sqrt(var_data[b] + static_cast<float>(eps_));
        }

        auto output_cpu = zeros_like(input_cpu);
        auto* output_data = output_cpu.data<BFloat16>();
        // Weight/bias are already converted to input dtype (BFloat16) above
        auto* weight_data = elementwise_affine_ ? weight_cpu.data<BFloat16>() : nullptr;
        auto* bias_data = elementwise_affine_ ? bias_cpu.data<BFloat16>() : nullptr;

        for (int64_t b = 0; b < batch_size; b++) {
            float mu = mean_data[b];
            float inv_std = rstd_data[b];

            for (int64_t i = 0; i < N; i++) {
                int64_t idx = b * N + i;
                float normalized = (static_cast<float>(input_data[idx]) - mu) * inv_std;

                if (elementwise_affine_) {
                    output_data[idx] = BFloat16(normalized * static_cast<float>(weight_data[i]) + static_cast<float>(bias_data[i]));
                } else {
                    output_data[idx] = BFloat16(normalized);
                }
            }
        }

        Tensor output = (original_device == Device::cpu()) ? output_cpu : output_cpu.to(original_device);

        if (is_grad_enabled() && (input.requires_grad() || (elementwise_affine_ && cached_weight_ && cached_weight_->requires_grad()))) {
            auto result = Variable(output, true);

            Tensor saved_mean = (original_device == Device::cpu()) ? batch_mean : batch_mean.to(original_device);
            Tensor saved_rstd = (original_device == Device::cpu()) ? rstd : rstd.to(original_device);

            std::vector<Tensor> tensors_to_save = {
                input.tensor(),
                saved_mean,
                saved_rstd,
                (elementwise_affine_ && cached_weight_) ? cached_weight_->tensor() : ones({N}, input.tensor().dtype(), input.tensor().device())
            };

            auto grad_fn = std::make_shared<LayerNormBackward>(
                elementwise_affine_, eps_, N, std::move(tensors_to_save)
            );

            result.set_grad_fn(grad_fn);

            std::vector<std::shared_ptr<Function>> next_funcs;
            if (auto input_grad_fn = input.grad_fn()) {
                next_funcs.push_back(input_grad_fn);
            }
            if (elementwise_affine_ && cached_weight_ && cached_weight_->grad_fn()) {
                next_funcs.push_back(cached_weight_->grad_fn());
            }
            grad_fn->set_next_functions(std::move(next_funcs));

            std::vector<Variable> input_vars;
            if (input.requires_grad()) {
                input_vars.push_back(input);
            }
            if (elementwise_affine_ && cached_weight_ && cached_weight_->requires_grad()) {
                input_vars.push_back(*cached_weight_);
            }
            if (elementwise_affine_ && cached_bias_ && cached_bias_->requires_grad()) {
                input_vars.push_back(*cached_bias_);
            }
            grad_fn->set_input_variables(input_vars);

            return result;
        }

        return Variable(output, false);

    } else if (input_dtype == DType::Float16) {
        // Float16 path: need batch_mean and batch_var for two-pass algorithm
        auto batch_mean = zeros({batch_size}, DType::Float32, Device::cpu());
        auto batch_var = zeros({batch_size}, DType::Float32, Device::cpu());
        auto* mean_data = batch_mean.data<float>();
        auto* var_data = batch_var.data<float>();

        auto* input_data = input_cpu.data<Float16>();

        // Compute mean for each batch element (use float accumulation)
        for (int64_t b = 0; b < batch_size; b++) {
            double sum = 0.0;
            for (int64_t i = 0; i < N; i++) {
                sum += static_cast<float>(input_data[b * N + i]);
            }
            mean_data[b] = static_cast<float>(sum / N);
        }

        // Compute variance for each batch element
        for (int64_t b = 0; b < batch_size; b++) {
            double sum_sq = 0.0;
            float mu = mean_data[b];
            for (int64_t i = 0; i < N; i++) {
                float diff = static_cast<float>(input_data[b * N + i]) - mu;
                sum_sq += diff * diff;
            }
            var_data[b] = static_cast<float>(sum_sq / N);
        }

        // Compute reciprocal std (1 / sqrt(var + eps))
        auto rstd = zeros({batch_size}, DType::Float32, Device::cpu());
        auto* rstd_data = rstd.data<float>();
        for (int64_t b = 0; b < batch_size; b++) {
            rstd_data[b] = 1.0f / std::sqrt(var_data[b] + static_cast<float>(eps_));
        }

        // Normalize: (x - mean) * rstd on CPU
        auto output_cpu = zeros_like(input_cpu);
        auto* output_data = output_cpu.data<Float16>();
        auto* weight_data = elementwise_affine_ ? weight_cpu.data<Float16>() : nullptr;
        auto* bias_data = elementwise_affine_ ? bias_cpu.data<Float16>() : nullptr;

        for (int64_t b = 0; b < batch_size; b++) {
            float mu = mean_data[b];
            float inv_std = rstd_data[b];

            for (int64_t i = 0; i < N; i++) {
                int64_t idx = b * N + i;
                float normalized = (static_cast<float>(input_data[idx]) - mu) * inv_std;

                if (elementwise_affine_) {
                    output_data[idx] = Float16(normalized * static_cast<float>(weight_data[i]) + static_cast<float>(bias_data[i]));
                } else {
                    output_data[idx] = Float16(normalized);
                }
            }
        }

        // Move output back to original device if needed
        Tensor output = (original_device == Device::cpu()) ? output_cpu : output_cpu.to(original_device);

        // Set up autograd if needed - check is_grad_enabled() first for fast inference path
        // Use cached pointers to avoid hash map lookups
        if (is_grad_enabled() && (input.requires_grad() || (elementwise_affine_ && cached_weight_ && cached_weight_->requires_grad()))) {
            auto result = Variable(output, true);

            // Prepare tensors to save for backward
            // Move statistics to original device so backward dispatch finds all tensors on same device
            Tensor saved_mean = (original_device == Device::cpu()) ? batch_mean : batch_mean.to(original_device);
            Tensor saved_rstd = (original_device == Device::cpu()) ? rstd : rstd.to(original_device);

            std::vector<Tensor> tensors_to_save = {
                input.tensor(),
                saved_mean,
                saved_rstd,
                (elementwise_affine_ && cached_weight_) ? cached_weight_->tensor() : ones({N}, input.tensor().dtype(), input.tensor().device())
            };

            auto grad_fn = std::make_shared<LayerNormBackward>(
                elementwise_affine_, eps_, N, std::move(tensors_to_save)
            );

            result.set_grad_fn(grad_fn);

            // Set next functions to chain backward pass
            std::vector<std::shared_ptr<Function>> next_funcs;
            // Only add grad_fn if it exists (not null)
            if (auto input_grad_fn = input.grad_fn()) {
                next_funcs.push_back(input_grad_fn);
            }
            if (elementwise_affine_ && cached_weight_ && cached_weight_->grad_fn()) {
                next_funcs.push_back(cached_weight_->grad_fn());
            }

            grad_fn->set_next_functions(std::move(next_funcs));

            // Track input variables for gradient accumulation using cached pointers
            std::vector<Variable> input_vars;
            if (input.requires_grad()) {
                input_vars.push_back(input);
            }
            if (elementwise_affine_ && cached_weight_ && cached_weight_->requires_grad()) {
                input_vars.push_back(*cached_weight_);
            }
            if (elementwise_affine_ && cached_bias_ && cached_bias_->requires_grad()) {
                input_vars.push_back(*cached_bias_);
            }
            grad_fn->set_input_variables(input_vars);

            return result;
        }

        return Variable(output, false);

    } else if (input_dtype == DType::Float32) {
        // Use fused SIMD-optimized LayerNorm that computes output + mean + rstd in single pass
        // This eliminates the double computation where forward computes stats,
        // then backward re-computes them again

        // Create tensors - use empty_uninitialized since fused kernel writes all elements
        auto output_cpu = Tensor::empty_uninitialized(
            {input_cpu.shape().begin(), input_cpu.shape().end()},
            DType::Float32, Device::cpu());
        auto batch_mean = Tensor::empty_uninitialized({batch_size}, DType::Float32, Device::cpu());
        auto rstd = Tensor::empty_uninitialized({batch_size}, DType::Float32, Device::cpu());

        // Get pointers
        const auto* input_data = input_cpu.data<float>();
        const auto* weight_data = weight_cpu.data<float>();
        const auto* bias_data = bias_cpu.data<float>();
        auto* output_data = output_cpu.data<float>();
        auto* mean_data = batch_mean.data<float>();
        auto* rstd_data = rstd.data<float>();

#if defined(__x86_64__) || defined(_M_X64)
        // Use fused SIMD implementation - computes output, mean, and rstd in one pass
        fused_layer_norm_f32(
            input_data,
            weight_data,
            bias_data,
            output_data,
            mean_data,
            rstd_data,
            batch_size,
            N,
            static_cast<float>(eps_)
        );
#else
        // Fallback for non-x86 platforms
        const int nthreads = get_optimal_threads();
        #pragma omp parallel for num_threads(nthreads)
        for (int64_t b = 0; b < batch_size; b++) {
            // Compute mean
            float sum = 0.0f;
            for (int64_t i = 0; i < N; i++) {
                sum += input_data[b * N + i];
            }
            float mean = sum / N;
            mean_data[b] = mean;

            // Compute variance
            float sum_sq = 0.0f;
            for (int64_t i = 0; i < N; i++) {
                float diff = input_data[b * N + i] - mean;
                sum_sq += diff * diff;
            }
            float var = sum_sq / N;
            float rs = 1.0f / std::sqrt(var + static_cast<float>(eps_));
            rstd_data[b] = rs;

            // Normalize with affine transform
            for (int64_t i = 0; i < N; i++) {
                int64_t idx = b * N + i;
                float norm = (input_data[idx] - mean) * rs;
                output_data[idx] = norm * weight_data[i] + bias_data[i];
            }
        }
#endif

        // Move output back to original device if needed
        Tensor output = (original_device == Device::cpu()) ? output_cpu : output_cpu.to(original_device);

        // Set up autograd if needed - check is_grad_enabled() first for fast inference path
        // Use cached pointers to avoid hash map lookups
        if (is_grad_enabled() && (input.requires_grad() || (elementwise_affine_ && cached_weight_ && cached_weight_->requires_grad()))) {
            auto result = Variable(output, true);

            // Prepare tensors to save for backward
            // Move statistics to original device so backward dispatch finds all tensors on same device
            Tensor saved_mean = (original_device == Device::cpu()) ? batch_mean : batch_mean.to(original_device);
            Tensor saved_rstd = (original_device == Device::cpu()) ? rstd : rstd.to(original_device);

            std::vector<Tensor> tensors_to_save = {
                input.tensor(),
                saved_mean,
                saved_rstd,
                (elementwise_affine_ && cached_weight_) ? cached_weight_->tensor() : ones({N}, input.tensor().dtype(), input.tensor().device())
            };

            auto grad_fn = std::make_shared<LayerNormBackward>(
                elementwise_affine_, eps_, N, std::move(tensors_to_save)
            );

            result.set_grad_fn(grad_fn);

            // Set next functions to chain backward pass using cached pointers
            std::vector<std::shared_ptr<Function>> next_funcs;
            if (auto input_grad_fn = input.grad_fn()) {
                next_funcs.push_back(input_grad_fn);
            }
            if (elementwise_affine_ && cached_weight_ && cached_weight_->requires_grad()) {
                if (auto weight_grad_fn = cached_weight_->grad_fn()) {
                    next_funcs.push_back(weight_grad_fn);
                }
            }
            if (elementwise_affine_ && cached_bias_ && cached_bias_->requires_grad()) {
                if (auto bias_grad_fn = cached_bias_->grad_fn()) {
                    next_funcs.push_back(bias_grad_fn);
                }
            }
            grad_fn->set_next_functions(next_funcs);

            // Track input variables using cached pointers
            std::vector<Variable> input_vars;
            if (input.requires_grad()) {
                input_vars.push_back(input);
            }
            if (elementwise_affine_ && cached_weight_ && cached_weight_->requires_grad()) {
                input_vars.push_back(*cached_weight_);
            }
            if (elementwise_affine_ && cached_bias_ && cached_bias_->requires_grad()) {
                input_vars.push_back(*cached_bias_);
            }
            grad_fn->set_input_variables(input_vars);

            return result;
        }

        return Variable(output, false);

    } else if (input_dtype == DType::Float64) {
        auto* input_data = input_cpu.data<double>();

        // Use double precision for mean and variance computation
        auto batch_mean_f64 = zeros({batch_size}, DType::Float64, Device::cpu());
        auto batch_var_f64 = zeros({batch_size}, DType::Float64, Device::cpu());
        auto* mean_data_f64 = batch_mean_f64.data<double>();
        auto* var_data_f64 = batch_var_f64.data<double>();

        // Compute mean for each batch element
        for (int64_t b = 0; b < batch_size; b++) {
            double sum = 0.0;
            for (int64_t i = 0; i < N; i++) {
                sum += input_data[b * N + i];
            }
            mean_data_f64[b] = sum / N;
        }

        // Compute variance for each batch element
        for (int64_t b = 0; b < batch_size; b++) {
            double sum_sq = 0.0;
            double mu = mean_data_f64[b];
            for (int64_t i = 0; i < N; i++) {
                double diff = input_data[b * N + i] - mu;
                sum_sq += diff * diff;
            }
            var_data_f64[b] = sum_sq / N;
        }

        // Compute reciprocal std (1 / sqrt(var + eps))
        auto rstd_f64 = zeros({batch_size}, DType::Float64, Device::cpu());
        auto* rstd_data_f64 = rstd_f64.data<double>();
        for (int64_t b = 0; b < batch_size; b++) {
            rstd_data_f64[b] = 1.0 / std::sqrt(var_data_f64[b] + eps_);
        }

        // Normalize: (x - mean) * rstd on CPU
        auto output_cpu = zeros_like(input_cpu);
        auto* output_data = output_cpu.data<double>();
        auto* weight_data = elementwise_affine_ ? weight_cpu.data<double>() : nullptr;
        auto* bias_data = elementwise_affine_ ? bias_cpu.data<double>() : nullptr;

        for (int64_t b = 0; b < batch_size; b++) {
            double mu = mean_data_f64[b];
            double inv_std = rstd_data_f64[b];

            for (int64_t i = 0; i < N; i++) {
                int64_t idx = b * N + i;
                double normalized = (input_data[idx] - mu) * inv_std;

                if (elementwise_affine_) {
                    output_data[idx] = normalized * weight_data[i] + bias_data[i];
                } else {
                    output_data[idx] = normalized;
                }
            }
        }

        // Move output back to original device if needed
        Tensor output = (original_device == Device::cpu()) ? output_cpu : output_cpu.to(original_device);

        // Set up autograd if needed - check is_grad_enabled() first for fast inference path
        if (is_grad_enabled() && (input.requires_grad() || (elementwise_affine_ && cached_weight_->requires_grad()))) {
            auto result = Variable(output, true);

            // Save mean and rstd for backward pass
            // GPU backward kernels expect stats to match input dtype (Float64).
            // CPU backward converts to Float32 itself, so save as Float32 to avoid waste.
            Tensor batch_mean_save, rstd_save;
            if (original_device.type == Device::Type::CPU) {
                // CPU backward converts to Float32 anyway, save as Float32
                batch_mean_save = zeros({batch_size}, DType::Float32, Device::cpu());
                rstd_save = zeros({batch_size}, DType::Float32, Device::cpu());
                auto* mean_save_data = batch_mean_save.data<float>();
                auto* rstd_save_data = rstd_save.data<float>();
                for (int64_t b = 0; b < batch_size; b++) {
                    mean_save_data[b] = static_cast<float>(mean_data_f64[b]);
                    rstd_save_data[b] = static_cast<float>(rstd_data_f64[b]);
                }
            } else {
                // GPU backends (CUDA, Vulkan, ROCm, etc.) read stats as-is in backward kernels
                batch_mean_save = zeros({batch_size}, DType::Float64, Device::cpu());
                rstd_save = zeros({batch_size}, DType::Float64, Device::cpu());
                auto* mean_save_data = batch_mean_save.data<double>();
                auto* rstd_save_data = rstd_save.data<double>();
                for (int64_t b = 0; b < batch_size; b++) {
                    mean_save_data[b] = mean_data_f64[b];
                    rstd_save_data[b] = rstd_data_f64[b];
                }
            }

            // Prepare tensors to save for backward
            // Move statistics to original device so backward dispatch finds all tensors on same device
            Tensor saved_mean = (original_device == Device::cpu()) ? batch_mean_save : batch_mean_save.to(original_device);
            Tensor saved_rstd = (original_device == Device::cpu()) ? rstd_save : rstd_save.to(original_device);

            std::vector<Tensor> tensors_to_save = {
                input.tensor(),
                saved_mean,
                saved_rstd,
                elementwise_affine_ ? cached_weight_->tensor() : ones({N}, input.tensor().dtype(), input.tensor().device())
            };

            auto grad_fn = std::make_shared<LayerNormBackward>(
                elementwise_affine_, eps_, N, std::move(tensors_to_save)
            );

            result.set_grad_fn(grad_fn);

            // Set next functions to chain backward pass
            std::vector<std::shared_ptr<Function>> next_funcs;
            if (auto input_grad_fn = input.grad_fn()) {
                next_funcs.push_back(input_grad_fn);
            }
            if (elementwise_affine_ && cached_weight_) {
                if (cached_weight_->requires_grad()) {
                    if (auto weight_grad_fn = cached_weight_->grad_fn()) {
                        next_funcs.push_back(weight_grad_fn);
                    }
                }
                if (cached_bias_ && cached_bias_->requires_grad()) {
                    if (auto bias_grad_fn = cached_bias_->grad_fn()) {
                        next_funcs.push_back(bias_grad_fn);
                    }
                }
            }
            grad_fn->set_next_functions(next_funcs);

            // Track input variables for gradient accumulation
            std::vector<Variable> input_vars;
            if (input.requires_grad()) {
                input_vars.push_back(input);
            }
            if (elementwise_affine_ && cached_weight_) {
                if (cached_weight_->requires_grad()) {
                    input_vars.push_back(*cached_weight_);
                }
                if (cached_bias_ && cached_bias_->requires_grad()) {
                    input_vars.push_back(*cached_bias_);
                }
            }
            grad_fn->set_input_variables(input_vars);

            return result;
        }

        return Variable(output, false);

    } else {
        throw std::runtime_error("LayerNorm only supports Float16, Float32, and Float64 dtypes");
    }

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
            std::vector<Tensor> inputs_vec = {go, inp, mn, rs, wt};
            auto results = dispatch<OpId::GroupNormBackward>(inputs_vec, attrs);

            if (needs_cast) {
                for (auto& r : results) r = r.to(original_dtype);
            }
            return results;
        }

        // CPU path: pointer-based access
        auto grad_output = grad_output_orig.to(Device::cpu()).to(DType::Float32).contiguous();
        auto input = input_orig.to(Device::cpu()).to(DType::Float32).contiguous();
        auto mean = mean_orig.to(Device::cpu()).to(DType::Float32).contiguous();
        auto rstd = rstd_orig.to(Device::cpu()).to(DType::Float32).contiguous();
        auto weight = weight_orig.to(Device::cpu()).to(DType::Float32).contiguous();

        auto shape = input.shape();
        int64_t N = shape[0];
        int64_t C = shape[1];
        int64_t H = shape[2];
        int64_t W = shape[3];
        int64_t spatial_size = H * W;

        auto* input_data = input.data<float>();
        auto* mean_data = mean.data<float>();
        auto* rstd_data = rstd.data<float>();
        auto* grad_out_data = grad_output.data<float>();
        auto* weight_data = weight.data<float>();

        auto grad_input = zeros_like(input);
        auto grad_weight = zeros({C}, DType::Float32, Device::cpu());
        auto grad_bias = zeros({C}, DType::Float32, Device::cpu());

        auto* grad_in_data = grad_input.data<float>();
        auto* grad_weight_data = grad_weight.data<float>();
        auto* grad_bias_data = grad_bias.data<float>();

        // Process each batch and group
        for (int64_t n = 0; n < N; n++) {
            for (int64_t g = 0; g < num_groups_; g++) {
                int64_t group_idx = n * num_groups_ + g;
                float mu = mean_data[group_idx];
                float inv_std = rstd_data[group_idx];

                int64_t c_start = g * group_size_;
                int64_t c_end = c_start + group_size_;

                // Compute intermediate sums for this group
                float sum_grad_out = 0.0f;
                float sum_grad_out_normalized = 0.0f;
                int64_t group_numel = group_size_ * spatial_size;

                for (int64_t c = c_start; c < c_end; c++) {
                    for (int64_t h = 0; h < H; h++) {
                        for (int64_t w = 0; w < W; w++) {
                            int64_t idx = ((n * C + c) * H + h) * W + w;
                            float x_normalized = (input_data[idx] - mu) * inv_std;
                            float grad_out = grad_out_data[idx] * weight_data[c];

                            sum_grad_out += grad_out;
                            sum_grad_out_normalized += grad_out * x_normalized;

                            // Accumulate weight and bias gradients
                            grad_weight_data[c] += grad_out_data[idx] * x_normalized;
                            grad_bias_data[c] += grad_out_data[idx];
                        }
                    }
                }

                // Compute input gradients
                float mean_grad_out = sum_grad_out / group_numel;
                float mean_grad_out_normalized = sum_grad_out_normalized / group_numel;

                for (int64_t c = c_start; c < c_end; c++) {
                    for (int64_t h = 0; h < H; h++) {
                        for (int64_t w = 0; w < W; w++) {
                            int64_t idx = ((n * C + c) * H + h) * W + w;
                            float x_normalized = (input_data[idx] - mu) * inv_std;
                            float grad_out = grad_out_data[idx] * weight_data[c];

                            grad_in_data[idx] = (grad_out - mean_grad_out -
                                                x_normalized * mean_grad_out_normalized) * inv_std;
                        }
                    }
                }
            }
        }

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

    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];
    int64_t spatial_size = H * W;

    if (C != num_channels_) {
        throw std::runtime_error("Expected " + std::to_string(num_channels_) +
                               " channels, got " + std::to_string(C));
    }

    int64_t group_size = num_channels_ / num_groups_;

    // Save original device and dtype
    auto original_device = input.tensor().device();
    auto original_dtype = input.tensor().dtype();

    // GPU fast path: dispatch to backend kernel (CUDA, Vulkan, ROCm, etc.)
    if (original_device.type != Device::Type::CPU) {
        // Float16: upcast to Float32 for precision in mean/variance computation.
        // GroupNorm statistics (mean, variance) accumulate over group_size * spatial_size
        // elements; Float16's limited mantissa causes significant rounding errors.
        DType compute_dtype = original_dtype;
        bool needs_upcast = (original_dtype == DType::Float16);
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

        Tensor output = results[0];
        Tensor saved_mean = results[1];
        Tensor saved_rstd = results[2];

        // Downcast output back to original dtype if we upcast for computation
        if (needs_upcast) {
            output = output.to(original_dtype);
        }

        if (input.requires_grad() || (affine_ && parameters_["weight"]->requires_grad())) {
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

    // CPU path: pointer-based computation
    Tensor input_tensor_cpu = input.tensor();
    if (original_dtype == DType::Float16 || original_dtype == DType::BFloat16 || original_dtype == DType::Float64) {
        input_tensor_cpu = input_tensor_cpu.to(DType::Float32);
    }
    Variable input_cpu = Variable(input_tensor_cpu, input.requires_grad());
    // Get weight/bias from parameters_ to respect offload hooks
    Tensor weight_tensor = affine_ ? parameters_["weight"]->tensor() : weight_.tensor();
    Tensor bias_tensor = affine_ ? parameters_["bias"]->tensor() : bias_.tensor();
    bool weight_requires_grad = affine_ ? parameters_["weight"]->requires_grad() : weight_.requires_grad();
    bool bias_requires_grad = affine_ ? parameters_["bias"]->requires_grad() : bias_.requires_grad();

    // Move to CPU and convert Float16/Float64 to Float32 for computation
    Tensor weight_tensor_cpu = weight_tensor;
    Tensor bias_tensor_cpu = bias_tensor;
    if (weight_tensor.device().type != Device::Type::CPU) {
        weight_tensor_cpu = weight_tensor_cpu.cpu();
        bias_tensor_cpu = bias_tensor_cpu.cpu();
    }
    if (weight_tensor_cpu.dtype() == DType::Float16 || weight_tensor_cpu.dtype() == DType::BFloat16 || weight_tensor_cpu.dtype() == DType::Float64) {
        weight_tensor_cpu = weight_tensor_cpu.to(DType::Float32);
        bias_tensor_cpu = bias_tensor_cpu.to(DType::Float32);
    }
    Variable weight_cpu = Variable(weight_tensor_cpu, weight_requires_grad);
    Variable bias_cpu = Variable(bias_tensor_cpu, bias_requires_grad);

    // Compute mean and variance for each group
    // Shape: [N, num_groups]
    auto group_mean = zeros({N * num_groups_});
    auto group_var = zeros({N * num_groups_});

    auto* input_data = input_cpu.tensor().data<float>();
    auto* mean_data = group_mean.data<float>();
    auto* var_data = group_var.data<float>();

    int64_t group_numel = group_size * spatial_size;

    // Compute mean for each group (using manual NCHW indexing)
    for (int64_t n = 0; n < N; n++) {
        for (int64_t g = 0; g < num_groups_; g++) {
            double sum = 0.0;
            int64_t c_start = g * group_size;
            int64_t c_end = c_start + group_size;

            for (int64_t c = c_start; c < c_end; c++) {
                for (int64_t h = 0; h < H; h++) {
                    for (int64_t w = 0; w < W; w++) {
                        int64_t idx = ((n * C + c) * H + h) * W + w;
                        sum += input_data[idx];
                    }
                }
            }

            mean_data[n * num_groups_ + g] = static_cast<float>(sum / group_numel);
        }
    }

    // Compute variance for each group
    for (int64_t n = 0; n < N; n++) {
        for (int64_t g = 0; g < num_groups_; g++) {
            double sum_sq = 0.0;
            float mu = mean_data[n * num_groups_ + g];
            int64_t c_start = g * group_size;
            int64_t c_end = c_start + group_size;

            for (int64_t c = c_start; c < c_end; c++) {
                for (int64_t h = 0; h < H; h++) {
                    for (int64_t w = 0; w < W; w++) {
                        int64_t idx = ((n * C + c) * H + h) * W + w;
                        float diff = input_data[idx] - mu;
                        sum_sq += diff * diff;
                    }
                }
            }

            var_data[n * num_groups_ + g] = static_cast<float>(sum_sq / group_numel);
        }
    }

    // Compute reciprocal std
    auto rstd = zeros({N * num_groups_});
    auto* rstd_data = rstd.data<float>();
    for (int64_t i = 0; i < N * num_groups_; i++) {
        rstd_data[i] = 1.0f / std::sqrt(var_data[i] + static_cast<float>(eps_));
    }

    // Normalize and apply affine transformation
    auto output = zeros_like(input_cpu.tensor());
    auto* output_data = output.data<float>();
    auto* weight_data = weight_cpu.tensor().data<float>();
    auto* bias_data = bias_cpu.tensor().data<float>();

    for (int64_t n = 0; n < N; n++) {
        for (int64_t g = 0; g < num_groups_; g++) {
            float mu = mean_data[n * num_groups_ + g];
            float inv_std = rstd_data[n * num_groups_ + g];
            int64_t c_start = g * group_size;
            int64_t c_end = c_start + group_size;

            for (int64_t c = c_start; c < c_end; c++) {
                for (int64_t h = 0; h < H; h++) {
                    for (int64_t w = 0; w < W; w++) {
                        int64_t idx = ((n * C + c) * H + h) * W + w;
                        float normalized = (input_data[idx] - mu) * inv_std;

                        if (affine_) {
                            output_data[idx] = normalized * weight_data[c] + bias_data[c];
                        } else {
                            output_data[idx] = normalized;
                        }
                    }
                }
            }
        }
    }

    // Convert back to original dtype and move to original device if needed
    Tensor output_final = output;
    if (original_dtype == DType::Float16 || original_dtype == DType::BFloat16 || original_dtype == DType::Float64) {
        output_final = output_final.to(original_dtype);
    }
    if (original_device.type != Device::Type::CPU) {
        output_final = output_final.to(original_device);
    }

    // Set up autograd if needed
    if (input.requires_grad() || (affine_ && parameters_["weight"]->requires_grad())) {
        auto result = Variable(output_final, true);

        // Move statistics to original device so backward dispatch finds all tensors on same device
        // For Float64 on Vulkan, stats must match input dtype since the f64 backward shader
        // declares double[] buffers. Convert Float32 stats to Float64 before uploading.
        Tensor stats_mean = group_mean;
        Tensor stats_rstd = rstd;
        if (original_dtype == DType::Float64 && original_device.type == Device::Type::Vulkan) {
            stats_mean = stats_mean.to(DType::Float64);
            stats_rstd = stats_rstd.to(DType::Float64);
        }
        Tensor saved_mean = (original_device.type == Device::Type::CPU) ? stats_mean : stats_mean.to(original_device);
        Tensor saved_rstd = (original_device.type == Device::Type::CPU) ? stats_rstd : stats_rstd.to(original_device);

        std::vector<Tensor> tensors_to_save = {
            input.tensor(),
            saved_mean,
            saved_rstd,
            affine_ ? parameters_["weight"]->tensor() : ones({C}, input.tensor().dtype(), input.tensor().device())
        };

        auto grad_fn = std::make_shared<GroupNormBackward>(
            affine_, eps_, num_groups_, num_channels_, group_size,
            std::move(tensors_to_save)
        );

        result.set_grad_fn(grad_fn);

        // Set next functions to chain backward pass
        std::vector<std::shared_ptr<Function>> next_funcs;
        // Only add grad_fn if it exists (not null)
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

        // Track input variables
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
    } else {
        return Variable(output_final, false);
    }
}

auto GroupNorm::reset_parameters() -> void {
    // Weight initialized to 1, bias to 0 (already done in constructor)
}

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

        // Reshape grad_input from 4D back to 3D
        Tensor grad_input = results[0].reshape({N, C, L_});
        if (original_device.type == Device::Type::CPU) {
            return {grad_input.to(original_dtype).to(original_device).contiguous(),
                    results[1].to(original_dtype).to(original_device).contiguous(),
                    results[2].to(original_dtype).to(original_device).contiguous()};
        }
        return {grad_input, results[1], results[2]};
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

        // GPU fast path
        if (original_device.type != Device::Type::CPU) {
            auto go = grad_output_orig.contiguous();
            auto inp = input_orig.contiguous();
            auto wt = weight_orig.contiguous();
            auto mn = mean_orig.contiguous();
            auto rs = rstd_orig.contiguous();

            // InstanceNormBackward: inputs [grad_output, input, weight, mean, rstd]
            std::vector<Tensor> inputs_vec = {go, inp, wt, mn, rs};
            auto results = dispatch<OpId::InstanceNormBackward>(inputs_vec);
            return results;
        }

        // CPU path: dispatch to kernel
        auto go = grad_output_orig.to(Device::cpu()).contiguous();
        auto inp = input_orig.to(Device::cpu()).contiguous();
        auto mn = mean_orig.to(Device::cpu()).contiguous();
        auto rs = rstd_orig.to(Device::cpu()).contiguous();
        auto wt = weight_orig.to(Device::cpu()).contiguous();

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

    // Get weight and bias tensors
    Tensor weight_tensor = affine_
        ? parameters_["weight"]->tensor().to(original_dtype).to(original_device)
        : ones({C}, original_dtype, original_device);
    Tensor bias_tensor = affine_
        ? parameters_["bias"]->tensor().to(original_dtype).to(original_device)
        : zeros({C}, original_dtype, original_device);

    // Dispatch to backend kernel
    NewOpAttributes attrs;
    attrs.set(AttrKey::Eps, static_cast<double>(eps_));
    std::vector<Tensor> inputs_vec = {input.tensor(), weight_tensor, bias_tensor};
    auto results = dispatch<OpId::InstanceNorm>(inputs_vec, attrs);

    Tensor output = results[0];
    Tensor saved_mean = results[1];   // [N, C]
    Tensor saved_rstd = results[2];   // [N, C]

    if (input.requires_grad() || (affine_ && parameters_["weight"]->requires_grad())) {
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
        }
        grad_fn->set_input_variables(input_vars);

        return result;
    }

    return Variable(output, false);
}

auto InstanceNorm2d::reset_parameters() -> void {
    // Weight initialized to 1, bias to 0 (already done in constructor)
}

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

    Tensor weight_tensor = affine_
        ? parameters_["weight"]->tensor().to(original_dtype).to(original_device)
        : ones({C}, original_dtype, original_device);
    Tensor bias_tensor = affine_
        ? parameters_["bias"]->tensor().to(original_dtype).to(original_device)
        : zeros({C}, original_dtype, original_device);

    NewOpAttributes attrs;
    attrs.set(AttrKey::Eps, static_cast<double>(eps_));
    std::vector<Tensor> inputs_vec = {input_4d, weight_tensor, bias_tensor};
    auto results = dispatch<OpId::InstanceNorm>(inputs_vec, attrs);

    // Reshape output back to (N, C, L)
    Tensor output = results[0].reshape({N, C, L});
    Tensor saved_mean = results[1];
    Tensor saved_rstd = results[2];

    if (input.requires_grad() || (affine_ && parameters_["weight"]->requires_grad())) {
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
      in2d_(num_features, eps, affine) {
    auto in2d_ptr = std::shared_ptr<InstanceNorm2d>(&in2d_, [](InstanceNorm2d*) {});
    register_module("in2d", in2d_ptr);
}

auto InstanceNorm3d::forward_impl(const Variable& input) -> Variable {
    auto shape = input.shape();
    if (shape.size() != 5) {
        throw std::runtime_error("InstanceNorm3d expects 5D input (N,C,D,H,W), got " +
            std::to_string(shape.size()) + "D");
    }

    int64_t N = shape[0], C = shape[1], D = shape[2], H = shape[3], W = shape[4];

    // Use tenzor::reshape (autograd-aware) to maintain computation graph
    auto reshaped = tenzor::reshape(input, {N, C, D * H, W});
    auto result = in2d_.forward(reshaped);
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

            NewOpAttributes attrs;
            attrs.set(AttrKey::NormalizedShape, std::to_string(normalized_size_));
            std::vector<Tensor> inputs_vec = {go, inp, rm, wt};
            auto results = dispatch<OpId::RMSNormBackward>(inputs_vec, attrs);
            return results;
        }

        // CPU path: pointer-based access
        auto grad_output = (grad_output_orig.device() == Device::cpu())
                          ? grad_output_orig.contiguous().to(DType::Float32)
                          : grad_output_orig.contiguous().to(Device::cpu()).to(DType::Float32);
        auto input = (input_orig.device() == Device::cpu())
                    ? input_orig.contiguous().to(DType::Float32)
                    : input_orig.contiguous().to(Device::cpu()).to(DType::Float32);
        auto rrms = (rrms_orig.device() == Device::cpu())
                   ? rrms_orig.contiguous().to(DType::Float32)
                   : rrms_orig.contiguous().to(Device::cpu()).to(DType::Float32);
        auto weight = (weight_orig.device() == Device::cpu())
                     ? weight_orig.contiguous().to(DType::Float32)
                     : weight_orig.contiguous().to(Device::cpu()).to(DType::Float32);

        auto shape = input.shape();
        int64_t batch_size = 1;
        for (size_t i = 0; i < shape.size() - 1; i++) {
            batch_size *= shape[i];
        }

        int64_t N = normalized_size_;

        auto* input_data = input.data<float>();
        auto* rrms_data = rrms.data<float>();
        auto* grad_out_data = grad_output.data<float>();
        auto* weight_data = weight.data<float>();

        // Allocate gradient tensors
        auto grad_input = zeros_like(input);
        auto grad_weight = zeros({N}, grad_output.dtype(), grad_output.device());

        auto* grad_in_data = grad_input.data<float>();
        auto* grad_weight_data = grad_weight.data<float>();

        // Compute gradients for each batch element
        // RMSNorm: y = x * rrms * weight
        // where rrms = 1 / sqrt(mean(x^2) + eps)
        //
        // d/dx[y] = weight * rrms - x * weight * rrms^3 * mean(x) / N
        // d/dweight[y] = x * rrms
        for (int64_t b = 0; b < batch_size; b++) {
            float inv_rms = rrms_data[b];
            float inv_rms_cubed = inv_rms * inv_rms * inv_rms;

            // Compute sum for gradient correction term
            float sum_grad_x = 0.0f;
            for (int64_t i = 0; i < N; i++) {
                int64_t idx = b * N + i;
                sum_grad_x += grad_out_data[idx] * weight_data[i] * input_data[idx];
            }
            float correction = sum_grad_x * inv_rms_cubed / N;

            // Compute input and weight gradients
            for (int64_t i = 0; i < N; i++) {
                int64_t idx = b * N + i;
                float x = input_data[idx];
                float normalized = x * inv_rms;

                // Weight gradient
                grad_weight_data[i] += grad_out_data[idx] * normalized;

                // Input gradient: d/dx = weight * rrms - x * correction
                grad_in_data[idx] = grad_out_data[idx] * weight_data[i] * inv_rms - x * correction;
            }
        }

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

auto RMSNorm::forward_impl(const Variable& input) -> Variable {
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
        return Variable(results[0], false);  // results[0] is output, [1] is rrms
    }

    // Vulkan fast path: dispatch to GPU RMSNorm shader (inference only)
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

        // Use stack allocation for rrms scratch buffer
        constexpr int64_t STACK_THRESHOLD = 8192;
        float stack_rrms[STACK_THRESHOLD];
        float* rrms_scratch = (batch_size <= STACK_THRESHOLD) ? stack_rrms : new float[batch_size];

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

        if (batch_size > STACK_THRESHOLD) {
            delete[] rrms_scratch;
        }

        return Variable(output, false);
    }

    // ============================================================================
    // STANDARD PATH: Full autograd support
    // ============================================================================

    Device original_device = input.tensor().device();

    // GPU training path: use fused kernel and set up autograd
    if (original_device.type != Device::Type::CPU) {
        const Tensor& x = input.tensor();

        Tensor weight_dev = cached_weight_ ? cached_weight_->tensor() : weight_.tensor();
        if (weight_dev.device() != original_device) {
            weight_dev = weight_dev.to(original_device);
        }

        NewOpAttributes attrs;
        attrs.set(AttrKey::Eps, eps_);

        std::vector<Tensor> inputs_vec = {x, weight_dev};
        auto results = dispatch<OpId::FusedRMSNorm>(inputs_vec, attrs);

        Tensor output = results[0];
        Tensor saved_rrms = results.size() > 1 ? results[1] : Tensor();

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

    // CPU path: pointer-based computation
    Tensor input_cpu = input.tensor();

    // Get weight from cached pointer (faster) or fallback to parameters_ for hooks
    Tensor weight_cpu = cached_weight_ ? cached_weight_->tensor() : parameters_["weight"]->tensor();
    if (weight_cpu.device() != Device::cpu()) {
        weight_cpu = weight_cpu.to(Device::cpu());
    }
    if (weight_cpu.dtype() != input_cpu.dtype()) {
        weight_cpu = weight_cpu.to(input_cpu.dtype());
    }

    DType input_dtype = input_cpu.dtype();

    if (input_dtype == DType::Float32) {
        auto* input_data = input_cpu.data<float>();
        auto* weight_data = weight_cpu.data<float>();

        // Allocate output and rrms tensors
        auto output_cpu = zeros_like(input_cpu);
        auto* output_data = output_cpu.data<float>();
        auto rrms = zeros({batch_size}, DType::Float32, Device::cpu());
        auto* rrms_data = rrms.data<float>();

#if defined(__x86_64__) || defined(_M_X64)
        // Use SIMD-optimized fused kernel
        fused_rms_norm_f32(
            input_data, weight_data, output_data, rrms_data,
            batch_size, N, static_cast<float>(eps_)
        );
#else
        // Scalar fallback
        for (int64_t b = 0; b < batch_size; b++) {
            double sum_sq = 0.0;
            for (int64_t i = 0; i < N; i++) {
                float val = input_data[b * N + i];
                sum_sq += val * val;
            }
            double rms = std::sqrt(sum_sq / N + eps_);
            rrms_data[b] = static_cast<float>(1.0 / rms);

            float inv_rms = rrms_data[b];
            for (int64_t i = 0; i < N; i++) {
                int64_t idx = b * N + i;
                output_data[idx] = input_data[idx] * inv_rms * weight_data[i];
            }
        }
#endif

        // Move output back to original device if needed
        Tensor output = (original_device == Device::cpu()) ? output_cpu : output_cpu.to(original_device);

        // Set up autograd if needed - check is_grad_enabled() first for fast inference path
        // Use cached pointer for faster requires_grad check
        if (is_grad_enabled() && (input.requires_grad() || (cached_weight_ && cached_weight_->requires_grad()))) {
            auto result = Variable(output, true);

            // Move statistics to original device so backward dispatch finds all tensors on same device
            Tensor saved_rrms = (original_device == Device::cpu()) ? rrms : rrms.to(original_device);

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

            // Track input variables for gradient accumulation
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

    } else if (input_dtype == DType::Float64) {
        auto* input_data = input_cpu.data<double>();

        // Compute root mean square for each batch element
        auto rrms = zeros({batch_size}, DType::Float64, Device::cpu());
        auto* rrms_data = rrms.data<double>();

        for (int64_t b = 0; b < batch_size; b++) {
            double sum_sq = 0.0;
            for (int64_t i = 0; i < N; i++) {
                double val = input_data[b * N + i];
                sum_sq += val * val;
            }
            double rms = std::sqrt(sum_sq / N + eps_);
            rrms_data[b] = 1.0 / rms;
        }

        // Normalize: x * rrms * weight
        auto output_cpu = zeros_like(input_cpu);
        auto* output_data = output_cpu.data<double>();
        auto* weight_data = weight_cpu.data<double>();

        for (int64_t b = 0; b < batch_size; b++) {
            double inv_rms = rrms_data[b];
            for (int64_t i = 0; i < N; i++) {
                int64_t idx = b * N + i;
                output_data[idx] = input_data[idx] * inv_rms * weight_data[i];
            }
        }

        // Move output back to original device if needed
        Tensor output = (original_device == Device::cpu()) ? output_cpu : output_cpu.to(original_device);

        // Set up autograd if needed - check is_grad_enabled() first for fast inference path
        // Use cached pointer for faster requires_grad check
        if (is_grad_enabled() && (input.requires_grad() || (cached_weight_ && cached_weight_->requires_grad()))) {
            auto result = Variable(output, true);

            // Save rrms for backward pass
            // For Vulkan backward, stats must match input dtype (Float64) since the
            // f64 shader reads them as double[]. CPU backward converts to Float32 itself.
            Tensor rrms_save;
            if (original_device.type == Device::Type::Vulkan) {
                // Keep as Float64 to match the f64 backward shader's buffer declarations
                rrms_save = rrms;
            } else {
                // CPU backward converts to Float32 anyway
                rrms_save = zeros({batch_size}, DType::Float32, Device::cpu());
                auto* rrms_save_data = rrms_save.data<float>();
                for (int64_t b = 0; b < batch_size; b++) {
                    rrms_save_data[b] = static_cast<float>(rrms_data[b]);
                }
            }

            // Move statistics to original device so backward dispatch finds all tensors on same device
            Tensor saved_rrms = (original_device == Device::cpu()) ? rrms_save : rrms_save.to(original_device);

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

            // Track input variables for gradient accumulation
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

    } else if (input_dtype == DType::Float16) {
        auto* input_data = input_cpu.data<Float16>();

        // Compute root mean square for each batch element (use float accumulation)
        auto rrms = zeros({batch_size}, DType::Float32, Device::cpu());
        auto* rrms_data = rrms.data<float>();

        for (int64_t b = 0; b < batch_size; b++) {
            double sum_sq = 0.0;
            for (int64_t i = 0; i < N; i++) {
                float val = static_cast<float>(input_data[b * N + i]);
                sum_sq += val * val;
            }
            double rms = std::sqrt(sum_sq / N + eps_);
            rrms_data[b] = static_cast<float>(1.0 / rms);
        }

        // Normalize: x * rrms * weight
        auto output_cpu = zeros_like(input_cpu);
        auto* output_data = output_cpu.data<Float16>();
        auto* weight_data = weight_cpu.data<Float16>();

        for (int64_t b = 0; b < batch_size; b++) {
            float inv_rms = rrms_data[b];
            for (int64_t i = 0; i < N; i++) {
                int64_t idx = b * N + i;
                output_data[idx] = Float16(static_cast<float>(input_data[idx]) * inv_rms * static_cast<float>(weight_data[i]));
            }
        }

        // Move output back to original device if needed
        Tensor output = (original_device == Device::cpu()) ? output_cpu : output_cpu.to(original_device);

        // Set up autograd if needed - check is_grad_enabled() first for fast inference path
        // Use cached pointer for faster requires_grad check
        if (is_grad_enabled() && (input.requires_grad() || (cached_weight_ && cached_weight_->requires_grad()))) {
            auto result = Variable(output, true);

            // Move statistics to original device so backward dispatch finds all tensors on same device
            Tensor saved_rrms = (original_device == Device::cpu()) ? rrms : rrms.to(original_device);

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

            // Track input variables for gradient accumulation
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

    } else {
        throw std::runtime_error("RMSNorm only supports Float16, Float32, and Float64 dtypes");
    }
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
    // where the window spans `size` channels centered on channel i.
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
    int64_t half = size_ / 2;

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

    for (int64_t j = -half; j <= half; ++j) {
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
