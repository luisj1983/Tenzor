/**
 * @file sampling.hip.cpp
 * @brief HIP/ROCm port of Bernoulli, Multinomial, Bucketize, Histogram, CDist.
 *
 * Mirrors the equivalent CUDA implementations in src/backends/cuda/kernels/advanced.cu.
 * Replaces the previous CPU-roundtrip fallbacks in rocm_kernel_registry.cpp.
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/creation.hpp"  // tenzor::get_global_seed (manual_seed reproducibility)
#include "tenzor/ops/transform.hpp"  // tenzor::expand / reshape for N-D cdist
#include "../hip_buffer.hpp"  // HipBuffer (RAII device scratch)
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tenzor {
namespace rocm {

// Forward declaration: defined in sort.hip.cpp.
auto topk_kernel(const Tensor& input, int64_t k, int64_t dim, bool largest,
                 bool sorted, hipStream_t stream) -> std::pair<Tensor, Tensor>;

// Forward declaration: defined in reduction.hip.cpp.
auto sum_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor;

#ifndef HIP_CHECK
#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(err)); \
    } \
} while(0)
#endif

// =========================================================================
// Counter-based per-element RNG (splitmix64)
// =========================================================================
//
// A single LCG step (state = a*state + c) produces a high-bit output that is an
// almost-affine function of the seeded `tid`, so adjacent threads get strongly
// correlated uniforms (visible striping) and consecutive draws from one stream
// (the Box-Muller u1,u2) are correlated enough to bias the transform. Instead,
// fully hash (seed, tid, counter) with the splitmix64 finalizer — the same mix
// the multinomial gumbel-keys kernel already uses — so each (element, draw) is
// an independent, well-distributed uniform. `counter` distinguishes successive
// draws for one element (e.g. Box-Muller u1 vs u2, rejection-sampling rounds).
__device__ __forceinline__ uint64_t splitmix64_hash(uint64_t seed, uint64_t tid,
                                                     uint64_t counter) {
    uint64_t state = seed
                   ^ (tid     * 0x9E3779B97F4A7C15ULL)
                   ^ (counter * 0xD1B54A32D192ED03ULL);
    state ^= state >> 30; state *= 0xBF58476D1CE4E5B9ULL;
    state ^= state >> 27; state *= 0x94D049BB133111EBULL;
    state ^= state >> 31;
    return state;
}

// Uniform in (0, 1) with 24-bit resolution (matches the gumbel-keys mapping).
__device__ __forceinline__ float splitmix64_uniform(uint64_t seed, uint64_t tid,
                                                     uint64_t counter) {
    uint64_t h = splitmix64_hash(seed, tid, counter);
    float u = static_cast<float>((h >> 40) + 1u) / 16777217.0f;  // (0, 1)
    return (u >= 1.0f) ? 0.99999994f : u;
}

// Stateful sequential draws for an element (rejection sampling), advancing the
// counter so each call is an independent uniform.
__device__ __forceinline__ float splitmix64_next_uniform(uint64_t seed, uint64_t tid,
                                                          uint64_t& counter) {
    return splitmix64_uniform(seed, tid, counter++);
}

// =========================================================================
// Bernoulli sampling
// =========================================================================

__global__ void bernoulli_kernel_impl(const float* probs, float* output,
                                       int64_t n, uint64_t seed) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    float u = splitmix64_uniform(seed, static_cast<uint64_t>(tid), 0);
    output[tid] = (u < probs[tid]) ? 1.0f : 0.0f;
}

auto bernoulli_kernel(const Tensor& probs, hipStream_t stream) -> Tensor {
    const DType orig = probs.dtype();
    auto input = probs.contiguous();
    if (input.dtype() != DType::Float32) input = input.to(DType::Float32);

    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, DType::Float32, input.device());
    int64_t n = input.numel();
    if (n == 0) return (orig == DType::Float32) ? result : result.to(orig);

    int threads = 256;
    int blocks_n = static_cast<int>((n + threads - 1) / threads);
    uint64_t seed = ::tenzor::get_global_seed();  // reproducible under manual_seed(); see Generator

    hipLaunchKernelGGL(bernoulli_kernel_impl,
        dim3(blocks_n), dim3(threads), 0, stream,
        input.data<float>(), result.data<float>(), n, seed);
    HIP_CHECK(hipGetLastError());
    // Preserve the input dtype (0/1 samples are exact in any float dtype).
    return (orig == DType::Float32) ? result : result.to(orig);
}

// =========================================================================
// Poisson sampling (Knuth algorithm with LCG PRNG)
// =========================================================================

// Uniform draw for the Poisson samplers, clamped away from 0 so it is safe to
// take a log() of it (both the Knuth product test and the Hormann PTRS
// acceptance test do so). Mirrors CUDA's poisson_lcg_uniform helper
// (advanced.cu), just re-based on this file's counter-hashed PRNG instead of
// an evolving LCG state.
__device__ __forceinline__ float poisson_next_uniform(uint64_t seed, uint64_t tid,
                                                       uint64_t& counter) {
    return fmaxf(splitmix64_next_uniform(seed, tid, counter), 1.0e-7f);
}

__global__ void poisson_kernel_impl(const float* rates, int64_t* output,
                                     int64_t n, uint64_t seed) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // Counter-based draws: each loop iteration is an independent uniform
    // hashed from (seed, tid, counter), so adjacent threads are decorrelated.
    uint64_t counter = 0;
    uint64_t utid = static_cast<uint64_t>(tid);

    float lambda = rates[tid];

    if (!(lambda > 0.0f)) {
        output[tid] = 0;
        return;
    }

    if (lambda < 12.0f) {
        // Knuth multiply-uniforms. expf(-lambda) is well above the float
        // underflow threshold for lambda < ~88, and accumulating in double
        // keeps the product representable across the (rare) long tails. Cap
        // iterations as a hard safety bound so the kernel always terminates.
        const double L = ::exp(-static_cast<double>(lambda));
        int64_t k = 0;
        double p = 1.0;
        const int64_t max_iter = 1 << 20;
        do {
            k++;
            p *= static_cast<double>(poisson_next_uniform(seed, utid, counter));
        } while (p > L && k < max_iter);
        output[tid] = k - 1;
        return;
    }

    // Transformed rejection (Hoermann PTRS) for moderate/large lambda, where
    // Knuth's expected iteration count grows linearly and expf(-lambda)
    // underflows to exactly 0 for lambda gtr ~104 (silently truncating the
    // Knuth loop after ~100-150 iterations regardless of true lambda, which
    // severely low-biases the samples). Ported faithfully from CUDA's
    // reference implementation (src/backends/cuda/kernels/advanced.cu).
    const double dlam = static_cast<double>(lambda);
    const double b = 0.931 + 2.53 * ::sqrt(dlam);
    const double a = -0.059 + 0.02483 * b;
    const double inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
    const double v_r = 0.9277 - 3.6224 / (b - 2.0);
    const double loglam = ::log(dlam);

    int64_t k = 0;
    for (int iter = 0; iter < 1024; ++iter) {
        double U = static_cast<double>(poisson_next_uniform(seed, utid, counter)) - 0.5;
        double V = static_cast<double>(poisson_next_uniform(seed, utid, counter));
        double us = 0.5 - fabs(U);
        double kd = ::floor((2.0 * a / us + b) * U + dlam + 0.43);
        if (us >= 0.07 && V <= v_r) {
            k = static_cast<int64_t>(kd);
            break;
        }
        if (kd < 0.0 || (us < 0.013 && V > us)) {
            continue;
        }
        // lgamma(kd+1) = log(kd!)
        double logV = ::log(V) + ::log(inv_alpha) - ::log(a / (us * us) + b);
        double rhs = -dlam + kd * loglam - ::lgamma(kd + 1.0);
        if (logV <= rhs) {
            k = static_cast<int64_t>(kd);
            break;
        }
    }
    output[tid] = k;
}

auto poisson_sample_kernel(const Tensor& rates, hipStream_t stream) -> Tensor {
    auto input = rates.contiguous();
    if (input.dtype() != DType::Float32) input = input.to(DType::Float32);

    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, DType::Int64, input.device());
    int64_t n = input.numel();
    if (n == 0) return result;

    int threads = 256;
    int blocks_n = static_cast<int>((n + threads - 1) / threads);
    uint64_t seed = ::tenzor::get_global_seed();  // reproducible under manual_seed(); see Generator

    hipLaunchKernelGGL(poisson_kernel_impl,
        dim3(blocks_n), dim3(threads), 0, stream,
        input.data<float>(), result.data<int64_t>(), n, seed);
    HIP_CHECK(hipGetLastError());
    return result;
}

// =========================================================================
// Normal sampling (Box-Muller transform)
// =========================================================================

__global__ void normal_sample_kernel_impl(const float* mean, const float* stddev,
                                           float* output, int64_t n, uint64_t seed) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // Two INDEPENDENT uniforms (distinct counters) for Box-Muller, instead of
    // two consecutive correlated LCG outputs which bias the transform.
    float u1 = fmaxf(splitmix64_uniform(seed, static_cast<uint64_t>(tid), 0), 1.0e-7f);
    float u2 = splitmix64_uniform(seed, static_cast<uint64_t>(tid), 1);

    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
    output[tid] = mean[tid] + stddev[tid] * z;
}

auto normal_sample_kernel(const Tensor& mean, const Tensor& stddev, hipStream_t stream) -> Tensor {
    const DType orig = mean.dtype();
    auto m = mean.contiguous();
    auto s = stddev.contiguous();
    if (m.dtype() != DType::Float32) m = m.to(DType::Float32);
    if (s.dtype() != DType::Float32) s = s.to(DType::Float32);

    std::vector<int64_t> shape(m.shape().begin(), m.shape().end());
    Tensor result(shape, DType::Float32, m.device());
    int64_t n = m.numel();
    if (n == 0) return (orig == DType::Float32) ? result : result.to(orig);

    int threads = 256;
    int blocks_n = static_cast<int>((n + threads - 1) / threads);
    uint64_t seed = ::tenzor::get_global_seed();  // reproducible under manual_seed(); see Generator

    hipLaunchKernelGGL(normal_sample_kernel_impl,
        dim3(blocks_n), dim3(threads), 0, stream,
        m.data<float>(), s.data<float>(), result.data<float>(), n, seed);
    HIP_CHECK(hipGetLastError());
    // Preserve the requested dtype (the public API expects normal(mean) to keep
    // mean's dtype; computing the Box-Muller transform in Float32 is fine).
    return (orig == DType::Float32) ? result : result.to(orig);
}

// =========================================================================
// Exponential sampling (inverse CDF method)
// =========================================================================

__global__ void exponential_sample_kernel_impl(const float* rate, float* output,
                                                int64_t n, uint64_t seed) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float u = splitmix64_uniform(seed, static_cast<uint64_t>(tid), 0);
    u = fminf(u, 1.0f - 1.0e-7f);

    output[tid] = -logf(1.0f - u) / rate[tid];
}

// The Exponential distribution is only defined for rate > 0 (rate==0 gives an
// undefined +Inf-mean distribution; rate<0 has no valid support at all).
// Every thread independently flags its own element into a single atomic int
// (no reduction-tree ordering to worry about, so NaN is always caught). The
// host reads that one scalar back and throws before ever dispatching the
// sampler, instead of silently emitting +Inf (rate==0) or a mathematically
// invalid negative sample (rate<0).
__global__ void exponential_validate_rate_kernel(const float* rate, int64_t n,
                                                  int* invalid_flag) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    if (!(rate[tid] > 0.0f)) {
        atomicExch(invalid_flag, 1);
    }
}

auto exponential_sample_kernel(const Tensor& rate, hipStream_t stream) -> Tensor {
    auto input = rate.contiguous();
    if (input.dtype() != DType::Float32) input = input.to(DType::Float32);

    int64_t n = input.numel();
    if (n > 0) {
        // O(1) host<->device sync regardless of tensor size: one small
        // scratch allocation, one kernel launch, one scalar readback — not an
        // elementwise CPU round-trip.
        HipBuffer flag_buf(sizeof(int));
        int* d_flag = flag_buf.as<int>();
        HIP_CHECK(hipMemsetAsync(d_flag, 0, sizeof(int), stream));
        int vthreads = 256;
        int vblocks = static_cast<int>((n + vthreads - 1) / vthreads);
        hipLaunchKernelGGL(exponential_validate_rate_kernel,
            dim3(vblocks), dim3(vthreads), 0, stream,
            input.data<float>(), n, d_flag);
        int h_flag = 0;
        HIP_CHECK(hipMemcpyAsync(&h_flag, d_flag, sizeof(int), hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        if (h_flag != 0) {
            throw std::invalid_argument(
                "exponential: rate must be > 0 (got a non-positive or NaN rate)");
        }
    }

    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, DType::Float32, input.device());
    if (n == 0) return result;

    int threads = 256;
    int blocks_n = static_cast<int>((n + threads - 1) / threads);
    uint64_t seed = ::tenzor::get_global_seed();  // reproducible under manual_seed(); see Generator

    hipLaunchKernelGGL(exponential_sample_kernel_impl,
        dim3(blocks_n), dim3(threads), 0, stream,
        input.data<float>(), result.data<float>(), n, seed);
    HIP_CHECK(hipGetLastError());
    return result;
}


// =========================================================================
// Gamma sampling (Marsaglia-Tsang 2000) — device-side, no host fallback.
// =========================================================================

// Counter-based draws for one element (seed, tid fixed; counter advances).
struct GammaRng {
    uint64_t seed;
    uint64_t tid;
    uint64_t counter;
};

__device__ __forceinline__ float gamma_next_uniform(GammaRng& rng) {
    return splitmix64_next_uniform(rng.seed, rng.tid, rng.counter);
}

__device__ __forceinline__ float gamma_next_normal(GammaRng& rng) {
    float u1 = fmaxf(gamma_next_uniform(rng), 1.0e-7f);
    float u2 = gamma_next_uniform(rng);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
}

__global__ void gamma_sample_kernel_impl(const float* alpha_in, const float* beta_in,
                                         float* output, int64_t n, uint64_t seed) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    GammaRng rng{seed, static_cast<uint64_t>(tid), 0};

    float alpha = alpha_in[tid];
    float beta  = beta_in[tid];
    if (!(alpha > 0.0f)) alpha = 1.1754944e-38f;  // FLT_MIN floor
    if (!(beta  > 0.0f)) beta  = 1.1754944e-38f;

    float boost = 1.0f;
    if (alpha < 1.0f) {
        float u0 = fmaxf(gamma_next_uniform(rng), 1.0e-7f);
        boost = powf(u0, 1.0f / alpha);
        alpha += 1.0f;
    }

    const float d = alpha - 1.0f / 3.0f;
    const float c = 1.0f / sqrtf(9.0f * d);
    float result;
    for (;;) {
        float x = gamma_next_normal(rng);
        float v = 1.0f + c * x;
        if (v <= 0.0f) continue;
        v = v * v * v;
        float u = fmaxf(gamma_next_uniform(rng), 1.0e-7f);
        float x2 = x * x;
        if (u < 1.0f - 0.0331f * x2 * x2 ||
            logf(u) < 0.5f * x2 + d * (1.0f - v + logf(v))) {
            result = d * v;
            break;
        }
    }
    output[tid] = boost * result / beta;
}

auto gamma_sample_kernel(const Tensor& concentration, const Tensor& rate,
                         hipStream_t stream) -> Tensor {
    // Preserve the caller's dtype: sampling runs in Float32 for RNG, but
    // gamma(Float64/BFloat16, ...) must return that dtype — matches the
    // widen-compute-narrow pattern normal_sample_kernel/bernoulli_kernel
    // already use in this file (was previously always Float32 regardless of
    // input dtype).
    const DType orig_dtype = concentration.dtype();
    auto a = concentration.contiguous();
    if (a.dtype() != DType::Float32) a = a.to(DType::Float32);
    auto b = rate.contiguous();
    if (b.dtype() != DType::Float32) b = b.to(DType::Float32);

    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, DType::Float32, a.device());
    int64_t n = a.numel();
    if (n == 0) return (orig_dtype == DType::Float32) ? result : result.to(orig_dtype);

    int threads = 256;
    int blocks_n = static_cast<int>((n + threads - 1) / threads);
    uint64_t seed = ::tenzor::get_global_seed();  // reproducible under manual_seed(); see Generator

    hipLaunchKernelGGL(gamma_sample_kernel_impl,
        dim3(blocks_n), dim3(threads), 0, stream,
        a.data<float>(), b.data<float>(), result.data<float>(), n, seed);
    HIP_CHECK(hipGetLastError());
    return (orig_dtype == DType::Float32) ? result : result.to(orig_dtype);
}

// =========================================================================
// Multinomial sampling
// =========================================================================

__global__ void multinomial_sample_kernel(const float* cdf, int64_t* output,
                                           int64_t num_categories,
                                           int64_t num_samples, float total,
                                           uint64_t seed) {
    int64_t sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= num_samples) return;
    float u = splitmix64_uniform(seed, static_cast<uint64_t>(sid), 0) * total;

    int64_t lo = 0, hi = num_categories - 1;
    while (lo < hi) {
        int64_t mid = (lo + hi) / 2;
        if (cdf[mid] <= u) lo = mid + 1;
        else hi = mid;
    }
    output[sid] = lo;
}

// Gumbel top-k: compute key_i = log(p_i) + (-log(-log(U_i))) per category.
// A descending sort selects num_samples distinct categories with the correct
// multinomial-without-replacement distribution.
__global__ void multinomial_gumbel_keys_kernel(
    const float* __restrict__ probs, float* __restrict__ keys,
    int64_t num_categories, uint64_t seed, int64_t batch_idx)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_categories) return;

    // Per-element PRNG stream. Use a simple splitmix-style mix to produce a
    // float in (0, 1); standard libraries aren't available device-side.
    uint64_t state = seed ^ (static_cast<uint64_t>(batch_idx) * 0x9E3779B97F4A7C15ULL)
                          ^ (static_cast<uint64_t>(tid)       * 0xBF58476D1CE4E5B9ULL);
    state ^= state >> 30; state *= 0xBF58476D1CE4E5B9ULL;
    state ^= state >> 27; state *= 0x94D049BB133111EBULL;
    state ^= state >> 31;
    // Map to float (0, 1). Avoid exactly 0.
    float u = static_cast<float>((state >> 40) + 1u) / 16777217.0f;
    if (u >= 1.0f) u = 0.99999994f;
    float g = -logf(-logf(u));   // Gumbel noise
    float p = probs[tid];
    float log_p = (p > 0.0f) ? logf(p) : -1e30f;
    keys[tid] = log_p + g;
}

auto multinomial_kernel(const Tensor& probs, int64_t num_samples,
                        bool replacement, hipStream_t stream) -> Tensor {
    auto input = probs.contiguous();
    if (input.dtype() != DType::Float32) input = input.to(DType::Float32);

    bool was_1d = (input.dim() == 1);
    if (was_1d) input = input.reshape({1, input.numel()});

    int64_t batch_size = input.shape()[0];
    int64_t num_categories = input.shape()[1];

    // Defensive guard: without replacement, topk(num_samples) would run out of
    // candidates and leave best_pos == -1, emitting -1 indices that later index
    // probability rows out of bounds. Fail loudly instead. (The op layer should
    // also validate, but this keeps the backend safe on its own.)
    if (!replacement && num_samples > num_categories) {
        throw std::invalid_argument(
            "multinomial: num_samples (" + std::to_string(num_samples) +
            ") cannot exceed the number of categories (" +
            std::to_string(num_categories) + ") when replacement=false");
    }

    // Every row must have a positive probability sum (matches CPU/CUDA, which
    // throw "sum of probabilities must be > 0"). Without this, all-zero rows
    // produce arbitrary/garbage indices on ROCm (a degenerate CDF for the
    // with-replacement path, +inf Gumbel keys for without-replacement) while
    // CPU errors. Previously this backend silently coerced a non-positive
    // per-row total to 1.0 and emitted garbage — a workaround, not a fix.
    {
        Tensor row_sums = sum_kernel(input, /*dim=*/1, /*keepdim=*/false, stream);
        std::vector<float> sums(static_cast<size_t>(batch_size));
        HIP_CHECK(hipMemcpyAsync(sums.data(), row_sums.data<float>(),
            batch_size * sizeof(float), hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        for (int64_t b = 0; b < batch_size; ++b) {
            if (!(sums[b] > 0.0f)) {
                throw std::runtime_error("multinomial: sum of probabilities must be > 0");
            }
        }
    }

    uint64_t seed = ::tenzor::get_global_seed();  // reproducible under manual_seed(); see Generator

    if (!replacement) {
        // Gumbel top-k trick: log(p_i) + -log(-log(U_i)), then sort descending.
        Tensor keys({batch_size, num_categories}, DType::Float32, input.device());
        int threads = 256;
        int blocks_cat = static_cast<int>((num_categories + threads - 1) / threads);
        for (int64_t b = 0; b < batch_size; ++b) {
            const float* prob_ptr = input.data<float>() + b * num_categories;
            float* key_ptr = keys.data<float>() + b * num_categories;
            hipLaunchKernelGGL(multinomial_gumbel_keys_kernel,
                dim3(blocks_cat), dim3(threads), 0, stream,
                prob_ptr, key_ptr, num_categories, seed, b);
        }
        HIP_CHECK(hipGetLastError());
        // keys is 2D (batch_size, num_categories); dim 1 is the categories axis.
        auto [values, indices] = topk_kernel(keys, num_samples, /*dim=*/1,
                                             /*largest=*/true, /*sorted=*/true, stream);
        Tensor result = indices;
        if (was_1d) result = result.reshape({num_samples});
        return result;
    }

    Tensor result({batch_size, num_samples}, DType::Int64, input.device());
    Tensor cdf_buf({batch_size, num_categories}, DType::Float32, input.device());

    // Build the CDF with a grid-wide inclusive scan (hipcub DeviceScan) so it is
    // correct for any num_categories. The previous single-block kernel capped
    // the block at 1024 threads, so for num_categories > 1024 only the first
    // 1024 entries were prefix-summed and the rest of the CDF was uninitialized,
    // corrupting `total` and the binary search.
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    {
        const float* first_in = input.data<float>();
        float* first_out = cdf_buf.data<float>();
        HIP_CHECK(hipcub::DeviceScan::InclusiveSum(
            d_temp_storage, temp_storage_bytes, first_in, first_out,
            static_cast<int>(num_categories), stream));
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));
    }

    for (int64_t b = 0; b < batch_size; ++b) {
        const float* prob_ptr = input.data<float>() + b * num_categories;
        float* cdf_ptr = cdf_buf.data<float>() + b * num_categories;
        int64_t* out_ptr = result.data<int64_t>() + b * num_samples;

        HIP_CHECK(hipcub::DeviceScan::InclusiveSum(
            d_temp_storage, temp_storage_bytes, prob_ptr, cdf_ptr,
            static_cast<int>(num_categories), stream));

        // Read total (CDF[last]) — single scalar metadata sync, not a CPU compute fallback
        float total = 0.0f;
        HIP_CHECK(hipMemcpyAsync(&total, cdf_ptr + num_categories - 1, sizeof(float),
                                  hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        // total > 0 is guaranteed by the up-front row-sum validation above.

        int threads = 256;
        int blocks = static_cast<int>((num_samples + threads - 1) / threads);
        hipLaunchKernelGGL(multinomial_sample_kernel,
            dim3(blocks), dim3(threads), 0, stream,
            cdf_ptr, out_ptr, num_categories, num_samples, total,
            seed + b * 1000003);
    }

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_temp_storage));

    if (was_1d) result = result.reshape({num_samples});
    return result;
}

// =========================================================================
// Bucketize
// =========================================================================

__global__ void bucketize_kernel_impl(const float* input, const float* boundaries,
                                       int64_t* output, int64_t n,
                                       int64_t num_boundaries, bool right) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    float val = input[tid];
    int64_t lo = 0, hi = num_boundaries;
    while (lo < hi) {
        int64_t mid = (lo + hi) / 2;
        bool cond = right ? (boundaries[mid] <= val) : (boundaries[mid] < val);
        if (cond) lo = mid + 1;
        else hi = mid;
    }
    output[tid] = lo;
}

auto bucketize_kernel(const Tensor& input, const Tensor& boundaries,
                      bool right, hipStream_t stream) -> Tensor {
    auto in_contig = input.contiguous();
    auto bound_contig = boundaries.contiguous();
    if (in_contig.dtype() != DType::Float32)    in_contig = in_contig.to(DType::Float32);
    if (bound_contig.dtype() != DType::Float32) bound_contig = bound_contig.to(DType::Float32);

    std::vector<int64_t> shape(in_contig.shape().begin(), in_contig.shape().end());
    Tensor result(shape, DType::Int64, in_contig.device());
    int64_t n = in_contig.numel();
    if (n == 0) return result;

    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    hipLaunchKernelGGL(bucketize_kernel_impl,
        dim3(blocks), dim3(threads), 0, stream,
        in_contig.data<float>(), bound_contig.data<float>(),
        result.data<int64_t>(), n, bound_contig.numel(), right);
    HIP_CHECK(hipGetLastError());
    return result;
}

// =========================================================================
// Histogram (with on-device min/max + bin-edge fill via CUB-like)
// =========================================================================

__global__ void histogram_kernel_impl(const float* input, int64_t* counts,
                                       int64_t n, int64_t num_bins,
                                       float min_val, float max_val, float bin_width) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    float val = input[tid];
    // Match CPU semantics: drop out-of-range samples.
    if (val < min_val || val > max_val) return;
    int64_t bin = static_cast<int64_t>((val - min_val) / bin_width);
    if (bin >= num_bins) bin = num_bins - 1;
    if (bin < 0) bin = 0;
    atomicAdd(reinterpret_cast<unsigned long long*>(&counts[bin]),
              static_cast<unsigned long long>(1));
}

__global__ void fill_bin_edges_kernel(float* edges, float min_val, float bin_width, int64_t num_edges) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_edges) return;
    edges[i] = min_val + static_cast<float>(i) * bin_width;
}

// On-device parallel reduction for min/max into a 2-element scratch buffer.
// Single block, simple shared-memory tree reduction. Acceptable since we only
// need this for the auto-range case (typically called once per histogram op).
__global__ void min_max_reduce_kernel(const float* input, float* min_max_out, int64_t n) {
    extern __shared__ float smem[];
    float* smin = smem;
    float* smax = smem + blockDim.x;

    int tid = threadIdx.x;
    float lmin = (tid < n) ? input[tid] : std::numeric_limits<float>::max();
    float lmax = (tid < n) ? input[tid] : std::numeric_limits<float>::lowest();
    for (int64_t i = tid + blockDim.x; i < n; i += blockDim.x) {
        float v = input[i];
        lmin = (v < lmin) ? v : lmin;
        lmax = (v > lmax) ? v : lmax;
    }
    smin[tid] = lmin;
    smax[tid] = lmax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float a = smin[tid], b = smin[tid + s]; smin[tid] = (a < b) ? a : b;
            float c = smax[tid], d = smax[tid + s]; smax[tid] = (c > d) ? c : d;
        }
        __syncthreads();
    }
    if (tid == 0) {
        min_max_out[0] = smin[0];
        min_max_out[1] = smax[0];
    }
}

auto histogram_kernel(const Tensor& input, int64_t bins,
                      double min_val, double max_val,
                      hipStream_t stream) -> std::pair<Tensor, Tensor> {
    auto in_contig = input.contiguous();
    if (in_contig.dtype() != DType::Float32) in_contig = in_contig.to(DType::Float32);
    int64_t n = in_contig.numel();

    if (min_val == 0.0 && max_val == 0.0 && n > 0) {
        // On-device min/max reduction → scratch → 2-float scalar readback
        float* d_min_max;
        HIP_CHECK(hipMalloc(&d_min_max, 2 * sizeof(float)));
        int block_size = 256;
        size_t smem_bytes = 2 * block_size * sizeof(float);
        hipLaunchKernelGGL(min_max_reduce_kernel,
            dim3(1), dim3(block_size), smem_bytes, stream,
            in_contig.data<float>(), d_min_max, n);
        float h_min_max[2];
        HIP_CHECK(hipMemcpyAsync(h_min_max, d_min_max, 2 * sizeof(float),
                                  hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipFree(d_min_max));
        min_val = h_min_max[0];
        max_val = h_min_max[1];
    }
    if (max_val <= min_val) max_val = min_val + 1.0;

    float bin_width = static_cast<float>((max_val - min_val) / bins);

    // Counts allocated zero-initialised by ROCm tensor factory (zeros call)
    Tensor counts({bins}, DType::Int64, in_contig.device());
    HIP_CHECK(hipMemsetAsync(counts.data_ptr(), 0,
                              static_cast<size_t>(bins) * sizeof(int64_t), stream));

    if (n > 0) {
        int threads = 256;
        int blocks_n = static_cast<int>((n + threads - 1) / threads);
        hipLaunchKernelGGL(histogram_kernel_impl,
            dim3(blocks_n), dim3(threads), 0, stream,
            in_contig.data<float>(), counts.data<int64_t>(),
            n, bins, static_cast<float>(min_val), static_cast<float>(max_val), bin_width);
    }

    Tensor edges({bins + 1}, DType::Float32, in_contig.device());
    {
        int64_t num_edges = bins + 1;
        int threads_e = 128;
        int blocks_e = static_cast<int>((num_edges + threads_e - 1) / threads_e);
        hipLaunchKernelGGL(fill_bin_edges_kernel,
            dim3(blocks_e), dim3(threads_e), 0, stream,
            edges.data<float>(), static_cast<float>(min_val), bin_width, num_edges);
    }
    HIP_CHECK(hipGetLastError());
    return {counts, edges};
}

// =========================================================================
// CDist (pairwise L2 distance)
// =========================================================================

template <typename T>
__global__ void cdist_l2_kernel_impl(const T* x1, const T* x2,
                                      T* output,
                                      int64_t B, int64_t P, int64_t R, int64_t M) {
    int64_t b = blockIdx.z;
    int64_t p = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || p >= P || r >= R) return;

    const T* a_b = x1 + b * P * M;
    const T* b_b = x2 + b * R * M;
    // F130: accumulate the sum of squared differences in double even for T=float
    // so Float32 cdist(p=2) matches the CPU reference (which widens each diff to
    // double). For T=double this is unchanged. Clamp to >=0 before sqrt like CPU.
    double sum = 0.0;
    for (int64_t m = 0; m < M; ++m) {
        double diff = static_cast<double>(a_b[p * M + m]) - static_cast<double>(b_b[r * M + m]);
        sum += diff * diff;
    }
    output[(b * P + p) * R + r] = static_cast<T>(sqrt(sum > 0.0 ? sum : 0.0));
}

template <typename T>
__global__ void cdist_l1_kernel_impl(const T* x1, const T* x2,
                                      T* output,
                                      int64_t B, int64_t P, int64_t R, int64_t M) {
    int64_t b = blockIdx.z;
    int64_t p = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || p >= P || r >= R) return;

    const T* a_b = x1 + b * P * M;
    const T* b_b = x2 + b * R * M;
    T sum = T(0);
    for (int64_t m = 0; m < M; ++m) {
        sum += fabs(a_b[p * M + m] - b_b[r * M + m]);
    }
    output[(b * P + p) * R + r] = sum;
}

template <typename T>
__global__ void cdist_lp_kernel_impl(const T* x1, const T* x2,
                                      T* output, T p,
                                      int64_t B, int64_t P, int64_t R, int64_t M) {
    int64_t b = blockIdx.z;
    int64_t p_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || p_idx >= P || r >= R) return;

    const T* a_b = x1 + b * P * M;
    const T* b_b = x2 + b * R * M;
    T sum = T(0);
    for (int64_t m = 0; m < M; ++m) {
        T diff = fabs(a_b[p_idx * M + m] - b_b[r * M + m]);
        sum += pow(diff, p);
    }
    output[(b * P + p_idx) * R + r] = pow(sum, T(1) / p);
}

auto cdist_kernel(const Tensor& x1, const Tensor& x2, double p,
                  hipStream_t stream) -> Tensor {
    // Record the original dtype so we can narrow the result back. Previously the
    // function silently returned Float32 regardless of input dtype, which broke
    // dtype parity for Float64/Float16 tests. Float64 now computes in a native
    // double accumulator instead of being down-converted to Float32.
    const DType orig_dtype = x1.dtype();
    const bool use_f64 = (orig_dtype == DType::Float64);
    const DType compute_dtype = use_f64 ? DType::Float64 : DType::Float32;

    auto a = x1.contiguous();
    auto b = x2.contiguous();
    if (a.dtype() != compute_dtype) a = a.to(compute_dtype);
    if (b.dtype() != compute_dtype) b = b.to(compute_dtype);

    // General N-D path with batch broadcasting (matches CPU cdist, mirrors the
    // CUDA path in src/backends/cuda/kernels/advanced.cu): the last two dims are
    // the matrix (P/R, M); all leading dims are the batch, which broadcasts when
    // one side is 1. Normalize to a matching-batch 3D problem, recurse into the
    // 3D path, then reshape the result back. (Previously this backend threw for
    // ndim>3 and required an exactly-matching 3D batch, diverging from CPU/CUDA
    // — AllBackends/CDistTest.ND4DMatchesCPU and BatchOneBroadcastMatchesCPU.)
    {
        const int64_t an = a.ndim(), bn = b.ndim();
        if (an < 2 || bn < 2) throw std::runtime_error("cdist: inputs must have >= 2 dims");
        const bool nd_general = an > 3 || bn > 3 || an != bn ||
            (an == 3 && bn == 3 && a.shape()[0] != b.shape()[0]);
        if (nd_general) {
            std::vector<int64_t> batchA(a.shape().begin(), a.shape().end() - 2);
            std::vector<int64_t> batchB(b.shape().begin(), b.shape().end() - 2);
            const size_t bl = std::max(batchA.size(), batchB.size());
            std::vector<int64_t> batch(bl);
            for (size_t i = 0; i < bl; ++i) {
                int64_t da = i < batchA.size() ? batchA[batchA.size() - 1 - i] : 1;
                int64_t db = i < batchB.size() ? batchB[batchB.size() - 1 - i] : 1;
                if (da != db && da != 1 && db != 1)
                    throw std::runtime_error("cdist: batch dimensions are not broadcastable");
                batch[bl - 1 - i] = (da == 1) ? db : da;
            }
            const int64_t P = a.shape()[an - 2], M = a.shape()[an - 1];
            const int64_t R = b.shape()[bn - 2];
            std::vector<int64_t> aT = batch; aT.push_back(P); aT.push_back(M);
            std::vector<int64_t> bT = batch; bT.push_back(R); bT.push_back(M);
            Tensor aE = ::tenzor::expand(a, aT).contiguous();
            Tensor bE = ::tenzor::expand(b, bT).contiguous();
            int64_t Bp = 1; for (int64_t d : batch) Bp *= d;
            Tensor a3 = ::tenzor::reshape(aE, {Bp, P, M});
            Tensor b3 = ::tenzor::reshape(bE, {Bp, R, M});
            Tensor r3 = cdist_kernel(a3, b3, p, stream);
            std::vector<int64_t> rShape = batch; rShape.push_back(P); rShape.push_back(R);
            Tensor r = ::tenzor::reshape(r3, rShape);
            return (orig_dtype == compute_dtype) ? r : r.to(orig_dtype);
        }
    }

    // Accept 2D (P, M) or 3D (B, P, M); 2D treated as B=1
    int64_t B, P, M, R;
    if (a.ndim() == 2 && b.ndim() == 2) {
        B = 1; P = a.shape()[0]; M = a.shape()[1]; R = b.shape()[0];
    } else if (a.ndim() == 3 && b.ndim() == 3) {
        B = a.shape()[0]; P = a.shape()[1]; M = a.shape()[2]; R = b.shape()[1];
        if (b.shape()[0] != B) throw std::runtime_error("cdist: batch dims must match");
    } else {
        throw std::runtime_error("cdist: inputs must be 2D or 3D");
    }

    std::vector<int64_t> result_shape = (a.ndim() == 2) ? std::vector<int64_t>{P, R}
                                                         : std::vector<int64_t>{B, P, R};
    Tensor result(result_shape, compute_dtype, a.device());
    if (B == 0 || P == 0 || R == 0) {
        return (orig_dtype == compute_dtype) ? result : result.to(orig_dtype);
    }

    dim3 threads(16, 16, 1);
    dim3 blocks(static_cast<unsigned>((R + 15) / 16),
                static_cast<unsigned>((P + 15) / 16),
                static_cast<unsigned>(B));
    if (use_f64) {
        if (p == 2.0) {
            hipLaunchKernelGGL(cdist_l2_kernel_impl<double>,
                blocks, threads, 0, stream,
                a.data<double>(), b.data<double>(), result.data<double>(), B, P, R, M);
        } else if (p == 1.0) {
            hipLaunchKernelGGL(cdist_l1_kernel_impl<double>,
                blocks, threads, 0, stream,
                a.data<double>(), b.data<double>(), result.data<double>(), B, P, R, M);
        } else {
            hipLaunchKernelGGL(cdist_lp_kernel_impl<double>,
                blocks, threads, 0, stream,
                a.data<double>(), b.data<double>(), result.data<double>(),
                p, B, P, R, M);
        }
    } else {
        if (p == 2.0) {
            hipLaunchKernelGGL(cdist_l2_kernel_impl<float>,
                blocks, threads, 0, stream,
                a.data<float>(), b.data<float>(), result.data<float>(), B, P, R, M);
        } else if (p == 1.0) {
            hipLaunchKernelGGL(cdist_l1_kernel_impl<float>,
                blocks, threads, 0, stream,
                a.data<float>(), b.data<float>(), result.data<float>(), B, P, R, M);
        } else {
            hipLaunchKernelGGL(cdist_lp_kernel_impl<float>,
                blocks, threads, 0, stream,
                a.data<float>(), b.data<float>(), result.data<float>(),
                static_cast<float>(p), B, P, R, M);
        }
    }
    HIP_CHECK(hipGetLastError());
    return (orig_dtype == compute_dtype) ? result : result.to(orig_dtype);
}

// ============================================================================
// Trapezoid integration
// ============================================================================

template <typename T>
__global__ void trapezoid_kernel_impl(
    const T* __restrict__ y, const T* __restrict__ x,
    T* __restrict__ output, int64_t outer, int64_t inner,
    int64_t n, T dx, bool has_x) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= outer * inner) return;

    int64_t o = idx / inner;
    int64_t i_inner = idx % inner;

    T sum = T(0);
    for (int64_t k = 0; k < n - 1; k++) {
        int64_t idx_k  = (o * n + k) * inner + i_inner;
        int64_t idx_k1 = (o * n + k + 1) * inner + i_inner;
        T h = has_x ? (x[idx_k1] - x[idx_k]) : dx;
        sum += T(0.5) * (y[idx_k] + y[idx_k1]) * h;
    }
    output[idx] = sum;
}

auto trapezoid_kernel(const Tensor& y, int64_t dim, double dx,
                       const Tensor* x_ptr, hipStream_t stream) -> Tensor {
    // Float64 integrates in a native double accumulator; everything else in
    // Float32 (down-converting Float64 to float lost precision previously).
    const DType orig_dtype = y.dtype();
    const bool use_f64 = (orig_dtype == DType::Float64);
    const DType compute_dtype = use_f64 ? DType::Float64 : DType::Float32;

    Tensor yf = (y.dtype() == compute_dtype) ? y.contiguous() : y.contiguous().to(compute_dtype);
    auto shape = yf.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    int64_t n = shape[dim];

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];

    std::vector<int64_t> out_shape;
    for (int64_t d = 0; d < ndim; d++) {
        if (d != dim) out_shape.push_back(shape[d]);
    }
    // Empty out_shape (1-D input) is a true 0-dim scalar, matching
    // CPU/CUDA/OneAPI/Vulkan's convention -- do not force a size-1 dim.

    Tensor result(out_shape, compute_dtype, y.device());
    int64_t total = outer * inner;
    if (n < 2 || total == 0) return result;

    Tensor xf;
    if (x_ptr) {
        xf = (x_ptr->dtype() == compute_dtype) ? x_ptr->contiguous() : x_ptr->contiguous().to(compute_dtype);
    }

    int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    if (use_f64) {
        hipLaunchKernelGGL(trapezoid_kernel_impl<double>, dim3(static_cast<unsigned>(blocks)), dim3(threads), 0, stream,
            yf.data<double>(), x_ptr ? xf.data<double>() : nullptr, result.data<double>(),
            outer, inner, n, dx, x_ptr != nullptr);
    } else {
        hipLaunchKernelGGL(trapezoid_kernel_impl<float>, dim3(static_cast<unsigned>(blocks)), dim3(threads), 0, stream,
            yf.data<float>(), x_ptr ? xf.data<float>() : nullptr, result.data<float>(),
            outer, inner, n, static_cast<float>(dx), x_ptr != nullptr);
    }
    HIP_CHECK(hipGetLastError());
    return (orig_dtype == compute_dtype) ? result : result.to(orig_dtype);
}

// ============================================================================
// Cumulative trapezoid integration
// ============================================================================

template <typename T>
__global__ void cumulative_trapezoid_kernel_impl(
    const T* __restrict__ y, const T* __restrict__ x,
    T* __restrict__ output, int64_t outer, int64_t inner,
    int64_t n, T dx, bool has_x) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= outer * inner) return;

    int64_t o = idx / inner;
    int64_t i_inner = idx % inner;

    T cumsum = T(0);
    for (int64_t k = 0; k < n - 1; k++) {
        int64_t idx_k  = (o * n + k) * inner + i_inner;
        int64_t idx_k1 = (o * n + k + 1) * inner + i_inner;
        T h = has_x ? (x[idx_k1] - x[idx_k]) : dx;
        cumsum += T(0.5) * (y[idx_k] + y[idx_k1]) * h;
        output[(o * (n - 1) + k) * inner + i_inner] = cumsum;
    }
}

auto cumulative_trapezoid_kernel(const Tensor& y, int64_t dim, double dx,
                                  const Tensor* x_ptr, hipStream_t stream) -> Tensor {
    // Float64 integrates in a native double accumulator; everything else in
    // Float32 (down-converting Float64 to float lost precision previously).
    const DType orig_dtype = y.dtype();
    const bool use_f64 = (orig_dtype == DType::Float64);
    const DType compute_dtype = use_f64 ? DType::Float64 : DType::Float32;

    Tensor yf = (y.dtype() == compute_dtype) ? y.contiguous() : y.contiguous().to(compute_dtype);
    auto shape = yf.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    int64_t n = shape[dim];

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[dim] = (n < 2) ? 0 : n - 1;
    Tensor result(out_shape, compute_dtype, y.device());

    int64_t total = outer * inner;
    if (n < 2 || total == 0) return result;

    Tensor xf;
    if (x_ptr) {
        xf = (x_ptr->dtype() == compute_dtype) ? x_ptr->contiguous() : x_ptr->contiguous().to(compute_dtype);
    }

    int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    if (use_f64) {
        hipLaunchKernelGGL(cumulative_trapezoid_kernel_impl<double>, dim3(static_cast<unsigned>(blocks)), dim3(threads), 0, stream,
            yf.data<double>(), x_ptr ? xf.data<double>() : nullptr, result.data<double>(),
            outer, inner, n, dx, x_ptr != nullptr);
    } else {
        hipLaunchKernelGGL(cumulative_trapezoid_kernel_impl<float>, dim3(static_cast<unsigned>(blocks)), dim3(threads), 0, stream,
            yf.data<float>(), x_ptr ? xf.data<float>() : nullptr, result.data<float>(),
            outer, inner, n, static_cast<float>(dx), x_ptr != nullptr);
    }
    HIP_CHECK(hipGetLastError());
    return (orig_dtype == compute_dtype) ? result : result.to(orig_dtype);
}

// ============================================================================
// Numerical gradient
// ============================================================================

template <typename T>
__global__ void gradient_kernel_impl(
    const T* __restrict__ input, T* __restrict__ output,
    int64_t outer, int64_t inner, int64_t n, T spacing) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= outer * inner) return;

    int64_t o = idx / inner;
    int64_t i_inner = idx % inner;

    auto at = [&](int64_t k) -> T {
        return input[(o * n + k) * inner + i_inner];
    };

    output[(o * n + 0) * inner + i_inner] = (at(1) - at(0)) / spacing;
    for (int64_t k = 1; k < n - 1; k++) {
        output[(o * n + k) * inner + i_inner] = (at(k + 1) - at(k - 1)) / (T(2) * spacing);
    }
    output[(o * n + n - 1) * inner + i_inner] = (at(n - 1) - at(n - 2)) / spacing;
}

auto gradient_kernel(const Tensor& input, int64_t dim, double spacing,
                      hipStream_t stream) -> Tensor {
    // Float64 differentiates natively in double; everything else in Float32
    // (down-converting Float64 to float lost precision previously).
    const DType orig_dtype = input.dtype();
    const bool use_f64 = (orig_dtype == DType::Float64);
    const DType compute_dtype = use_f64 ? DType::Float64 : DType::Float32;

    Tensor inf = (input.dtype() == compute_dtype) ? input.contiguous() : input.contiguous().to(compute_dtype);
    auto shape = inf.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    int64_t n = shape[dim];

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];

    Tensor result(std::vector<int64_t>(shape.begin(), shape.end()), compute_dtype, input.device());
    int64_t total = outer * inner;
    if (n < 2 || total == 0) return result;

    int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    if (use_f64) {
        hipLaunchKernelGGL(gradient_kernel_impl<double>, dim3(static_cast<unsigned>(blocks)), dim3(threads), 0, stream,
            inf.data<double>(), result.data<double>(),
            outer, inner, n, spacing);
    } else {
        hipLaunchKernelGGL(gradient_kernel_impl<float>, dim3(static_cast<unsigned>(blocks)), dim3(threads), 0, stream,
            inf.data<float>(), result.data<float>(),
            outer, inner, n, static_cast<float>(spacing));
    }
    HIP_CHECK(hipGetLastError());
    return (orig_dtype == compute_dtype) ? result : result.to(orig_dtype);
}

// ============================================================================
// Pairwise distance
// ============================================================================

template <typename T>
__global__ void pairwise_distance_kernel_impl(
    const T* __restrict__ x1, const T* __restrict__ x2,
    T* __restrict__ output, uint32_t N, uint32_t D, T p) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    T sum = T(0);
    if (p == T(2)) {
        for (uint32_t j = 0; j < D; j++) {
            T diff = x1[i * D + j] - x2[i * D + j];
            sum += diff * diff;
        }
        output[i] = sqrt(sum);
    } else if (p == T(1)) {
        for (uint32_t j = 0; j < D; j++) {
            sum += fabs(x1[i * D + j] - x2[i * D + j]);
        }
        output[i] = sum;
    } else {
        for (uint32_t j = 0; j < D; j++) {
            sum += pow(fabs(x1[i * D + j] - x2[i * D + j]), p);
        }
        output[i] = pow(sum, T(1) / p);
    }
}

auto pairwise_distance_kernel(const Tensor& x1, const Tensor& x2, double p,
                               hipStream_t stream) -> Tensor {
    // Float64 computes in a native double accumulator; everything else Float32
    // (down-converting Float64 to float lost precision previously).
    const DType orig_dtype = x1.dtype();
    const bool use_f64 = (orig_dtype == DType::Float64);
    const DType compute_dtype = use_f64 ? DType::Float64 : DType::Float32;

    Tensor a = (x1.dtype() == compute_dtype) ? x1.contiguous() : x1.contiguous().to(compute_dtype);
    Tensor b = (x2.dtype() == compute_dtype) ? x2.contiguous() : x2.contiguous().to(compute_dtype);

    int64_t N = a.shape()[0], D = a.shape()[1];
    Tensor result({N}, compute_dtype, x1.device());
    if (N == 0) return result;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (use_f64) {
        hipLaunchKernelGGL(pairwise_distance_kernel_impl<double>, dim3(blocks), dim3(threads), 0, stream,
            a.data<double>(), b.data<double>(), result.data<double>(),
            static_cast<uint32_t>(N), static_cast<uint32_t>(D), p);
    } else {
        hipLaunchKernelGGL(pairwise_distance_kernel_impl<float>, dim3(blocks), dim3(threads), 0, stream,
            a.data<float>(), b.data<float>(), result.data<float>(),
            static_cast<uint32_t>(N), static_cast<uint32_t>(D), static_cast<float>(p));
    }
    HIP_CHECK(hipGetLastError());
    return (orig_dtype == compute_dtype) ? result : result.to(orig_dtype);
}

// ============================================================================
// Pdist (all-pairs pairwise distances)
// ============================================================================

template <typename T>
__global__ void pdist_kernel_impl(
    const T* __restrict__ data, T* __restrict__ output,
    int64_t N, int64_t D, int64_t num_pairs, T p) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    int64_t i = 0, offset = 0;
    while (offset + (N - 1 - i) <= idx) {
        offset += (N - 1 - i);
        i++;
    }
    int64_t j = idx - offset + i + 1;

    T sum = T(0);
    if (p == T(2)) {
        for (int64_t d = 0; d < D; d++) {
            T diff = data[i * D + d] - data[j * D + d];
            sum += diff * diff;
        }
        output[idx] = sqrt(sum);
    } else if (p == T(1)) {
        for (int64_t d = 0; d < D; d++) {
            sum += fabs(data[i * D + d] - data[j * D + d]);
        }
        output[idx] = sum;
    } else if (isinf(p)) {
        // p = inf: Chebyshev distance = max_d |x_i,d - x_j,d|. The generic
        // pow() path computes pow(diff, inf) -> inf, then pow(inf, 0) -> 1,
        // returning all-ones instead of the max — matches CPU/CUDA's isinf
        // branch. (PdistChebyshevInfinity expects [4, 2, 6].)
        for (int64_t d = 0; d < D; d++) {
            T a = fabs(data[i * D + d] - data[j * D + d]);
            if (a > sum) sum = a;
        }
        output[idx] = sum;
    } else {
        for (int64_t d = 0; d < D; d++) {
            sum += pow(fabs(data[i * D + d] - data[j * D + d]), p);
        }
        output[idx] = pow(sum, T(1) / p);
    }
}

auto pdist_kernel(const Tensor& input, double p, hipStream_t stream) -> Tensor {
    // Float64 computes in a native double accumulator; everything else Float32
    // (down-converting Float64 to float lost precision previously).
    const DType orig_dtype = input.dtype();
    const bool use_f64 = (orig_dtype == DType::Float64);
    const DType compute_dtype = use_f64 ? DType::Float64 : DType::Float32;

    Tensor inf = (input.dtype() == compute_dtype) ? input.contiguous() : input.contiguous().to(compute_dtype);
    int64_t N = inf.shape()[0], D = inf.shape()[1];
    int64_t num_pairs = N * (N - 1) / 2;

    Tensor result({num_pairs}, compute_dtype, input.device());
    if (num_pairs == 0) return result;

    int threads = 256;
    int64_t blocks = (num_pairs + threads - 1) / threads;
    if (use_f64) {
        hipLaunchKernelGGL(pdist_kernel_impl<double>, dim3(static_cast<unsigned>(blocks)), dim3(threads), 0, stream,
            inf.data<double>(), result.data<double>(),
            N, D, num_pairs, p);
    } else {
        hipLaunchKernelGGL(pdist_kernel_impl<float>, dim3(static_cast<unsigned>(blocks)), dim3(threads), 0, stream,
            inf.data<float>(), result.data<float>(),
            N, D, num_pairs, static_cast<float>(p));
    }
    HIP_CHECK(hipGetLastError());
    return (orig_dtype == compute_dtype) ? result : result.to(orig_dtype);
}

// =========================================================================
// Histogramdd (multi-dimensional histogram) kernel
// =========================================================================

// Device kernel: each thread processes one sample, computing a flat bin index
// and atomically incrementing the counts tensor.
// params_buf layout per dimension d: [min_d, step_d, bins_d, stride_d] (4 doubles each)
template <typename T>
__global__ void histogramdd_kernel_impl(const T* input, int64_t* counts,
                                         const double* params_buf,
                                         int64_t N, int64_t D) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    const T* sample = input + tid * D;
    int64_t flat = 0;
    for (int64_t d = 0; d < D; ++d) {
        int64_t base = d * 4;
        double fmin   = params_buf[base + 0];
        double step   = params_buf[base + 1];
        int64_t nbins = static_cast<int64_t>(params_buf[base + 2]);
        int64_t str   = static_cast<int64_t>(params_buf[base + 3]);

        double v = static_cast<double>(sample[d]);
        double upper = fmin + step * static_cast<double>(nbins);
        if (v < fmin || v > upper) return; // out of range — skip sample

        int64_t b = static_cast<int64_t>((v - fmin) / step);
        if (b >= nbins) b = nbins - 1;
        if (b < 0) b = 0;
        flat += b * str;
    }
    atomicAdd(reinterpret_cast<unsigned long long*>(&counts[flat]),
              static_cast<unsigned long long>(1));
}

// Device kernel: fill edge tensor for one dimension
template <typename T>
__global__ void histogramdd_fill_edges_kernel(T* edges, T fmin, T step, int64_t num_edges) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_edges) return;
    edges[i] = fmin + static_cast<T>(i) * step;
}

// Device kernel: density normalisation — convert int64 counts to float
template <typename T>
__global__ void histogramdd_density_kernel(const int64_t* counts, T* output,
                                            int64_t total_bins, double norm) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_bins) return;
    output[i] = static_cast<T>(static_cast<double>(counts[i]) / norm);
}

// Column min/max reduction: one block per dimension, walks all N rows.
template <typename T>
__global__ void histogramdd_col_minmax_kernel(const T* input, double* min_max_out,
                                               int64_t N, int64_t D) {
    extern __shared__ char smem_raw[];
    double* smin = reinterpret_cast<double*>(smem_raw);
    double* smax = smin + blockDim.x;

    int64_t d = blockIdx.x; // one block per dimension
    int tid = threadIdx.x;

    double lmin = 1e308;
    double lmax = -1e308;
    for (int64_t i = tid; i < N; i += blockDim.x) {
        double v = static_cast<double>(input[i * D + d]);
        if (v < lmin) lmin = v;
        if (v > lmax) lmax = v;
    }
    smin[tid] = lmin;
    smax[tid] = lmax;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (smin[tid + s] < smin[tid]) smin[tid] = smin[tid + s];
            if (smax[tid + s] > smax[tid]) smax[tid] = smax[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        min_max_out[d * 2 + 0] = smin[0];
        min_max_out[d * 2 + 1] = smax[0];
    }
}

auto histogramdd_kernel(const Tensor& input,
                        std::vector<int64_t> bins,
                        std::vector<std::pair<double,double>> ranges,
                        bool density,
                        hipStream_t stream)
    -> std::pair<Tensor, std::vector<Tensor>> {

    if (input.dim() != 2) {
        throw std::runtime_error("histogramdd_kernel: input must be 2-D (N, D)");
    }

    const int64_t N = input.shape()[0];
    const int64_t D = input.shape()[1];

    if (static_cast<int64_t>(bins.size()) != D) {
        throw std::runtime_error("histogramdd_kernel: bins length must equal D");
    }

    const auto orig_dtype = input.dtype();
    const bool use_f64 = (orig_dtype == DType::Float64);
    const auto compute_dtype = use_f64 ? DType::Float64 : DType::Float32;
    auto inp = (input.dtype() != compute_dtype) ? input.to(compute_dtype) : input;
    inp = inp.contiguous();
    const auto& device = input.device();

    bool auto_range = ranges.empty();
    if (auto_range) {
        ranges.resize(static_cast<size_t>(D));
    }

    // Auto-detect ranges on device
    if (auto_range && N > 0) {
        double* d_minmax;
        size_t mm_bytes = static_cast<size_t>(D) * 2 * sizeof(double);
        HIP_CHECK(hipMalloc(&d_minmax, mm_bytes));

        int block_size = 256;
        size_t smem_bytes = 2 * block_size * sizeof(double);
        if (use_f64) {
            hipLaunchKernelGGL(histogramdd_col_minmax_kernel<double>,
                dim3(static_cast<int>(D)), dim3(block_size), smem_bytes, stream,
                inp.data<double>(), d_minmax, N, D);
        } else {
            hipLaunchKernelGGL(histogramdd_col_minmax_kernel<float>,
                dim3(static_cast<int>(D)), dim3(block_size), smem_bytes, stream,
                inp.data<float>(), d_minmax, N, D);
        }

        std::vector<double> h_minmax(static_cast<size_t>(D) * 2);
        HIP_CHECK(hipMemcpyAsync(h_minmax.data(), d_minmax,
                                  mm_bytes, hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipFree(d_minmax));

        for (int64_t d = 0; d < D; ++d) {
            double vmin = h_minmax[static_cast<size_t>(d) * 2 + 0];
            double vmax = h_minmax[static_cast<size_t>(d) * 2 + 1];
            if (vmin == vmax) { vmin -= 0.5; vmax += 0.5; }
            ranges[static_cast<size_t>(d)] = {vmin, vmax};
        }
    } else if (auto_range && N == 0) {
        for (int64_t d = 0; d < D; ++d) {
            ranges[static_cast<size_t>(d)] = {0.0, 1.0};
        }
    }

    // Build per-dimension parameters and edge tensors
    std::vector<double> dim_min(static_cast<size_t>(D));
    std::vector<double> dim_step(static_cast<size_t>(D));
    std::vector<Tensor> edges_vec;
    edges_vec.reserve(static_cast<size_t>(D));

    for (int64_t d = 0; d < D; ++d) {
        auto sd = static_cast<size_t>(d);
        int64_t nb = bins[sd];
        double fmin = ranges[sd].first;
        double fmax = ranges[sd].second;
        // A caller-supplied degenerate range (fmin >= fmax) yields step<=0, so
        // (v - fmin)/step at bin time is NaN/Inf (silently clamped afterward,
        // but the bin assignment would be unspecified). The auto-range path
        // already widens an equal-bounds interval by +-0.5 above; reuse that
        // same widening here — matches CPU's histogramdd_kernel
        // (src/backends/cpu/kernels/reduction.cpp) so step stays strictly
        // positive.
        if (fmin >= fmax) {
            fmin -= 0.5;
            fmax += 0.5;
        }
        double step = (fmax - fmin) / static_cast<double>(nb);
        dim_min[sd] = fmin;
        dim_step[sd] = step;

        Tensor edge({nb + 1}, compute_dtype, device);
        int64_t num_edges = nb + 1;
        int threads_e = 128;
        int blocks_e = static_cast<int>((num_edges + threads_e - 1) / threads_e);
        if (use_f64) {
            hipLaunchKernelGGL(histogramdd_fill_edges_kernel<double>,
                dim3(blocks_e), dim3(threads_e), 0, stream,
                edge.data<double>(), fmin, step, num_edges);
        } else {
            hipLaunchKernelGGL(histogramdd_fill_edges_kernel<float>,
                dim3(blocks_e), dim3(threads_e), 0, stream,
                edge.data<float>(), static_cast<float>(fmin), static_cast<float>(step), num_edges);
        }
        edges_vec.push_back(std::move(edge));
    }

    // Compute output shape and strides (row-major)
    std::vector<int64_t> out_shape(bins.begin(), bins.end());
    std::vector<int64_t> out_strides(static_cast<size_t>(D));
    int64_t stride = 1;
    for (int64_t d = D - 1; d >= 0; --d) {
        out_strides[static_cast<size_t>(d)] = stride;
        stride *= bins[static_cast<size_t>(d)];
    }
    int64_t total_bins = stride;

    // Allocate counts (zero-initialised)
    Tensor counts(out_shape, DType::Int64, device);
    HIP_CHECK(hipMemsetAsync(counts.data_ptr(), 0,
                              static_cast<size_t>(total_bins) * sizeof(int64_t), stream));

    // Upload per-dimension params buffer to device: [min, step, bins, stride] * D
    std::vector<double> h_params(static_cast<size_t>(D) * 4);
    for (int64_t d = 0; d < D; ++d) {
        auto sd = static_cast<size_t>(d);
        h_params[sd * 4 + 0] = dim_min[sd];
        h_params[sd * 4 + 1] = dim_step[sd];
        h_params[sd * 4 + 2] = static_cast<double>(bins[sd]);
        h_params[sd * 4 + 3] = static_cast<double>(out_strides[sd]);
    }
    size_t params_bytes = h_params.size() * sizeof(double);
    // RAII device scratch so an intervening throwing HIP_CHECK does not leak it.
    HipBuffer d_params_buf(params_bytes);
    double* d_params = static_cast<double*>(d_params_buf.ptr);
    HIP_CHECK(hipMemcpyAsync(d_params, h_params.data(), params_bytes,
                              hipMemcpyHostToDevice, stream));

    // Launch histogram kernel
    if (N > 0) {
        int threads = 256;
        int blocks_n = static_cast<int>((N + threads - 1) / threads);
        if (use_f64) {
            hipLaunchKernelGGL(histogramdd_kernel_impl<double>,
                dim3(blocks_n), dim3(threads), 0, stream,
                inp.data<double>(), counts.data<int64_t>(), d_params, N, D);
        } else {
            hipLaunchKernelGGL(histogramdd_kernel_impl<float>,
                dim3(blocks_n), dim3(threads), 0, stream,
                inp.data<float>(), counts.data<int64_t>(), d_params, N, D);
        }
        HIP_CHECK(hipGetLastError());
    }

    HIP_CHECK(hipStreamSynchronize(stream));
    // d_params is freed by HipBuffer's destructor at scope exit.

    // Density normalisation
    Tensor result = counts;
    if (density && N > 0) {
        double bin_volume = 1.0;
        for (int64_t d = 0; d < D; ++d) {
            bin_volume *= dim_step[static_cast<size_t>(d)];
        }
        double norm = static_cast<double>(N) * bin_volume;

        Tensor density_out(out_shape, compute_dtype, device);
        int threads_d = 256;
        int blocks_d = static_cast<int>((total_bins + threads_d - 1) / threads_d);
        if (use_f64) {
            hipLaunchKernelGGL(histogramdd_density_kernel<double>,
                dim3(blocks_d), dim3(threads_d), 0, stream,
                counts.data<int64_t>(), density_out.data<double>(), total_bins, norm);
        } else {
            hipLaunchKernelGGL(histogramdd_density_kernel<float>,
                dim3(blocks_d), dim3(threads_d), 0, stream,
                counts.data<int64_t>(), density_out.data<float>(), total_bins, norm);
        }
        HIP_CHECK(hipGetLastError());
        result = density_out;
    }

    return {std::move(result), std::move(edges_vec)};
}

}  // namespace rocm
}  // namespace tenzor
