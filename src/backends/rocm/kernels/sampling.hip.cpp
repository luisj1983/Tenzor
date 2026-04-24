/**
 * @file sampling.hip.cpp
 * @brief HIP/ROCm port of Bernoulli, Multinomial, Bucketize, Histogram, CDist.
 *
 * Mirrors the equivalent CUDA implementations in src/backends/cuda/kernels/advanced.cu.
 * Replaces the previous CPU-roundtrip fallbacks in rocm_kernel_registry.cpp.
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <hip/hip_runtime.h>
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

#ifndef HIP_CHECK
#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(err)); \
    } \
} while(0)
#endif

// =========================================================================
// Bernoulli sampling
// =========================================================================

__global__ void bernoulli_kernel_impl(const float* probs, float* output,
                                       int64_t n, uint64_t seed) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    uint64_t state = seed + static_cast<uint64_t>(tid) * 6364136223846793005ULL + 1442695040888963407ULL;
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    float u = static_cast<float>(state >> 33) / static_cast<float>(1ULL << 31);
    output[tid] = (u < probs[tid]) ? 1.0f : 0.0f;
}

auto bernoulli_kernel(const Tensor& probs, hipStream_t stream) -> Tensor {
    auto input = probs.contiguous();
    if (input.dtype() != DType::Float32) input = input.to(DType::Float32);

    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, DType::Float32, input.device());
    int64_t n = input.numel();
    if (n == 0) return result;

    int threads = 256;
    int blocks_n = static_cast<int>((n + threads - 1) / threads);
    uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    hipLaunchKernelGGL(bernoulli_kernel_impl,
        dim3(blocks_n), dim3(threads), 0, stream,
        input.data<float>(), result.data<float>(), n, seed);
    HIP_CHECK(hipGetLastError());
    return result;
}

// =========================================================================
// Poisson sampling (Knuth algorithm with LCG PRNG)
// =========================================================================

__global__ void poisson_kernel_impl(const float* rates, int64_t* output,
                                     int64_t n, uint64_t seed) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // LCG state initialisation (same pattern as bernoulli)
    uint64_t state = seed + static_cast<uint64_t>(tid) * 6364136223846793005ULL + 1442695040888963407ULL;

    float lambda = rates[tid];
    float L = expf(-lambda);
    int64_t k = 0;
    float p = 1.0f;

    do {
        ++k;
        // Advance LCG and produce a uniform float in (0, 1)
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        float u = static_cast<float>(state >> 33) / static_cast<float>(1ULL << 31);
        p *= u;
    } while (p > L);

    output[tid] = k - 1;
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
    uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

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

    uint64_t state = seed + tid * 6364136223846793005ULL + 1442695040888963407ULL;

    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    float u1 = static_cast<float>(state >> 33) / static_cast<float>(1ULL << 31);
    u1 = fmaxf(u1, 1.0e-7f);

    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    float u2 = static_cast<float>(state >> 33) / static_cast<float>(1ULL << 31);

    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
    output[tid] = mean[tid] + stddev[tid] * z;
}

auto normal_sample_kernel(const Tensor& mean, const Tensor& stddev, hipStream_t stream) -> Tensor {
    auto m = mean.contiguous();
    auto s = stddev.contiguous();
    if (m.dtype() != DType::Float32) m = m.to(DType::Float32);
    if (s.dtype() != DType::Float32) s = s.to(DType::Float32);

    std::vector<int64_t> shape(m.shape().begin(), m.shape().end());
    Tensor result(shape, DType::Float32, m.device());
    int64_t n = m.numel();
    if (n == 0) return result;

    int threads = 256;
    int blocks_n = static_cast<int>((n + threads - 1) / threads);
    uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    hipLaunchKernelGGL(normal_sample_kernel_impl,
        dim3(blocks_n), dim3(threads), 0, stream,
        m.data<float>(), s.data<float>(), result.data<float>(), n, seed);
    HIP_CHECK(hipGetLastError());
    return result;
}

// =========================================================================
// Exponential sampling (inverse CDF method)
// =========================================================================

__global__ void exponential_sample_kernel_impl(const float* rate, float* output,
                                                int64_t n, uint64_t seed) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    uint64_t state = seed + tid * 6364136223846793005ULL + 1442695040888963407ULL;
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    float u = static_cast<float>(state >> 33) / static_cast<float>(1ULL << 31);
    u = fminf(u, 1.0f - 1.0e-7f);

    output[tid] = -logf(1.0f - u) / rate[tid];
}

auto exponential_sample_kernel(const Tensor& rate, hipStream_t stream) -> Tensor {
    auto input = rate.contiguous();
    if (input.dtype() != DType::Float32) input = input.to(DType::Float32);

    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, DType::Float32, input.device());
    int64_t n = input.numel();
    if (n == 0) return result;

    int threads = 256;
    int blocks_n = static_cast<int>((n + threads - 1) / threads);
    uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    hipLaunchKernelGGL(exponential_sample_kernel_impl,
        dim3(blocks_n), dim3(threads), 0, stream,
        input.data<float>(), result.data<float>(), n, seed);
    HIP_CHECK(hipGetLastError());
    return result;
}

// =========================================================================
// Multinomial sampling
// =========================================================================

__global__ void multinomial_cdf_kernel(const float* probs, float* cdf,
                                        int64_t num_categories) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;

    float val = (tid < num_categories) ? probs[tid] : 0.0f;
    shared[tid] = val;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        float tmp = (tid >= stride) ? shared[tid - stride] : 0.0f;
        __syncthreads();
        shared[tid] += tmp;
        __syncthreads();
    }

    if (tid < num_categories) {
        cdf[tid] = shared[tid];
    }
}

__global__ void multinomial_sample_kernel(const float* cdf, int64_t* output,
                                           int64_t num_categories,
                                           int64_t num_samples, float total,
                                           uint64_t seed) {
    int64_t sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= num_samples) return;
    uint64_t state = seed + static_cast<uint64_t>(sid) * 6364136223846793005ULL + 1442695040888963407ULL;
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    float u = static_cast<float>(state >> 33) / static_cast<float>(1ULL << 31) * total;

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

    uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    if (!replacement) {
        // Gumbel top-k trick: log(p_i) + -log(-log(U_i)), then sort descending.
        // num_samples must be ≤ num_categories; caller is expected to validate.
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

    for (int64_t b = 0; b < batch_size; ++b) {
        const float* prob_ptr = input.data<float>() + b * num_categories;
        float* cdf_ptr = cdf_buf.data<float>() + b * num_categories;
        int64_t* out_ptr = result.data<int64_t>() + b * num_samples;

        int block_size = 1;
        while (block_size < num_categories) block_size *= 2;
        if (block_size > 1024) block_size = 1024;
        hipLaunchKernelGGL(multinomial_cdf_kernel,
            dim3(1), dim3(block_size), block_size * sizeof(float), stream,
            prob_ptr, cdf_ptr, num_categories);

        // Read total (CDF[last]) — single scalar metadata sync, not a CPU compute fallback
        float total = 0.0f;
        HIP_CHECK(hipMemcpyAsync(&total, cdf_ptr + num_categories - 1, sizeof(float),
                                  hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        if (total <= 0.0f) total = 1.0f;

        int threads = 256;
        int blocks = static_cast<int>((num_samples + threads - 1) / threads);
        hipLaunchKernelGGL(multinomial_sample_kernel,
            dim3(blocks), dim3(threads), 0, stream,
            cdf_ptr, out_ptr, num_categories, num_samples, total,
            seed + b * 1000003);
    }

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

__global__ void cdist_l2_kernel_impl(const float* x1, const float* x2,
                                      float* output,
                                      int64_t B, int64_t P, int64_t R, int64_t M) {
    int64_t b = blockIdx.z;
    int64_t p = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || p >= P || r >= R) return;

    const float* a_b = x1 + b * P * M;
    const float* b_b = x2 + b * R * M;
    float sum = 0.0f;
    for (int64_t m = 0; m < M; ++m) {
        float diff = a_b[p * M + m] - b_b[r * M + m];
        sum += diff * diff;
    }
    output[(b * P + p) * R + r] = sqrtf(sum);
}

__global__ void cdist_l1_kernel_impl(const float* x1, const float* x2,
                                      float* output,
                                      int64_t B, int64_t P, int64_t R, int64_t M) {
    int64_t b = blockIdx.z;
    int64_t p = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || p >= P || r >= R) return;

    const float* a_b = x1 + b * P * M;
    const float* b_b = x2 + b * R * M;
    float sum = 0.0f;
    for (int64_t m = 0; m < M; ++m) {
        sum += fabsf(a_b[p * M + m] - b_b[r * M + m]);
    }
    output[(b * P + p) * R + r] = sum;
}

__global__ void cdist_lp_kernel_impl(const float* x1, const float* x2,
                                      float* output, float p,
                                      int64_t B, int64_t P, int64_t R, int64_t M) {
    int64_t b = blockIdx.z;
    int64_t p_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || p_idx >= P || r >= R) return;

    const float* a_b = x1 + b * P * M;
    const float* b_b = x2 + b * R * M;
    float sum = 0.0f;
    for (int64_t m = 0; m < M; ++m) {
        float diff = fabsf(a_b[p_idx * M + m] - b_b[r * M + m]);
        sum += powf(diff, p);
    }
    output[(b * P + p_idx) * R + r] = powf(sum, 1.0f / p);
}

auto cdist_kernel(const Tensor& x1, const Tensor& x2, double p,
                  hipStream_t stream) -> Tensor {
    // Record the original dtype so we can narrow the Float32 result back.
    // Previously the function silently returned Float32 regardless of input
    // dtype, which broke dtype parity for Float64/Float16 tests.
    const DType orig_dtype = x1.dtype();

    auto a = x1.contiguous();
    auto b = x2.contiguous();
    if (a.dtype() != DType::Float32) a = a.to(DType::Float32);
    if (b.dtype() != DType::Float32) b = b.to(DType::Float32);

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
    Tensor result(result_shape, DType::Float32, a.device());
    if (B == 0 || P == 0 || R == 0) {
        return (orig_dtype == DType::Float32) ? result : result.to(orig_dtype);
    }

    dim3 threads(16, 16, 1);
    dim3 blocks(static_cast<unsigned>((R + 15) / 16),
                static_cast<unsigned>((P + 15) / 16),
                static_cast<unsigned>(B));
    if (p == 2.0) {
        hipLaunchKernelGGL(cdist_l2_kernel_impl,
            blocks, threads, 0, stream,
            a.data<float>(), b.data<float>(), result.data<float>(), B, P, R, M);
    } else if (p == 1.0) {
        hipLaunchKernelGGL(cdist_l1_kernel_impl,
            blocks, threads, 0, stream,
            a.data<float>(), b.data<float>(), result.data<float>(), B, P, R, M);
    } else {
        hipLaunchKernelGGL(cdist_lp_kernel_impl,
            blocks, threads, 0, stream,
            a.data<float>(), b.data<float>(), result.data<float>(),
            static_cast<float>(p), B, P, R, M);
    }
    HIP_CHECK(hipGetLastError());
    return (orig_dtype == DType::Float32) ? result : result.to(orig_dtype);
}

// ============================================================================
// Trapezoid integration
// ============================================================================

__global__ void trapezoid_kernel_impl(
    const float* __restrict__ y, const float* __restrict__ x,
    float* __restrict__ output, uint32_t outer, uint32_t inner,
    uint32_t n, float dx, bool has_x) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer * inner) return;

    uint32_t o = idx / inner;
    uint32_t i_inner = idx % inner;

    float sum = 0.0f;
    for (uint32_t k = 0; k < n - 1; k++) {
        uint32_t idx_k  = (o * n + k) * inner + i_inner;
        uint32_t idx_k1 = (o * n + k + 1) * inner + i_inner;
        float h = has_x ? (x[idx_k1] - x[idx_k]) : dx;
        sum += 0.5f * (y[idx_k] + y[idx_k1]) * h;
    }
    output[idx] = sum;
}

auto trapezoid_kernel(const Tensor& y, int64_t dim, double dx,
                       const Tensor* x_ptr, hipStream_t stream) -> Tensor {
    Tensor yf = (y.dtype() == DType::Float32) ? y.contiguous() : y.contiguous().to(DType::Float32);
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
    if (out_shape.empty()) out_shape.push_back(1);

    Tensor result(out_shape, DType::Float32, y.device());
    int64_t total = outer * inner;
    if (n < 2 || total == 0) return result;

    const float* x_data = nullptr;
    Tensor xf;
    if (x_ptr) {
        xf = (x_ptr->dtype() == DType::Float32) ? x_ptr->contiguous() : x_ptr->contiguous().to(DType::Float32);
        x_data = xf.data<float>();
    }

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(trapezoid_kernel_impl, dim3(blocks), dim3(threads), 0, stream,
        yf.data<float>(), x_data, result.data<float>(),
        static_cast<uint32_t>(outer), static_cast<uint32_t>(inner),
        static_cast<uint32_t>(n), static_cast<float>(dx), x_ptr != nullptr);
    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// Cumulative trapezoid integration
// ============================================================================

__global__ void cumulative_trapezoid_kernel_impl(
    const float* __restrict__ y, const float* __restrict__ x,
    float* __restrict__ output, uint32_t outer, uint32_t inner,
    uint32_t n, float dx, bool has_x) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer * inner) return;

    uint32_t o = idx / inner;
    uint32_t i_inner = idx % inner;

    float cumsum = 0.0f;
    for (uint32_t k = 0; k < n - 1; k++) {
        uint32_t idx_k  = (o * n + k) * inner + i_inner;
        uint32_t idx_k1 = (o * n + k + 1) * inner + i_inner;
        float h = has_x ? (x[idx_k1] - x[idx_k]) : dx;
        cumsum += 0.5f * (y[idx_k] + y[idx_k1]) * h;
        output[(o * (n - 1) + k) * inner + i_inner] = cumsum;
    }
}

auto cumulative_trapezoid_kernel(const Tensor& y, int64_t dim, double dx,
                                  const Tensor* x_ptr, hipStream_t stream) -> Tensor {
    Tensor yf = (y.dtype() == DType::Float32) ? y.contiguous() : y.contiguous().to(DType::Float32);
    auto shape = yf.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    int64_t n = shape[dim];

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[dim] = (n < 2) ? 0 : n - 1;
    Tensor result(out_shape, DType::Float32, y.device());

    int64_t total = outer * inner;
    if (n < 2 || total == 0) return result;

    const float* x_data = nullptr;
    Tensor xf;
    if (x_ptr) {
        xf = (x_ptr->dtype() == DType::Float32) ? x_ptr->contiguous() : x_ptr->contiguous().to(DType::Float32);
        x_data = xf.data<float>();
    }

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(cumulative_trapezoid_kernel_impl, dim3(blocks), dim3(threads), 0, stream,
        yf.data<float>(), x_data, result.data<float>(),
        static_cast<uint32_t>(outer), static_cast<uint32_t>(inner),
        static_cast<uint32_t>(n), static_cast<float>(dx), x_ptr != nullptr);
    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// Numerical gradient
// ============================================================================

__global__ void gradient_kernel_impl(
    const float* __restrict__ input, float* __restrict__ output,
    uint32_t outer, uint32_t inner, uint32_t n, float spacing) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer * inner) return;

    uint32_t o = idx / inner;
    uint32_t i_inner = idx % inner;

    auto at = [&](uint32_t k) -> float {
        return input[(o * n + k) * inner + i_inner];
    };

    output[(o * n + 0) * inner + i_inner] = (at(1) - at(0)) / spacing;
    for (uint32_t k = 1; k < n - 1; k++) {
        output[(o * n + k) * inner + i_inner] = (at(k + 1) - at(k - 1)) / (2.0f * spacing);
    }
    output[(o * n + n - 1) * inner + i_inner] = (at(n - 1) - at(n - 2)) / spacing;
}

auto gradient_kernel(const Tensor& input, int64_t dim, double spacing,
                      hipStream_t stream) -> Tensor {
    Tensor inf = (input.dtype() == DType::Float32) ? input.contiguous() : input.contiguous().to(DType::Float32);
    auto shape = inf.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    int64_t n = shape[dim];

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];

    Tensor result(std::vector<int64_t>(shape.begin(), shape.end()), DType::Float32, input.device());
    int64_t total = outer * inner;
    if (n < 2 || total == 0) return result;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(gradient_kernel_impl, dim3(blocks), dim3(threads), 0, stream,
        inf.data<float>(), result.data<float>(),
        static_cast<uint32_t>(outer), static_cast<uint32_t>(inner),
        static_cast<uint32_t>(n), static_cast<float>(spacing));
    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// Pairwise distance
// ============================================================================

__global__ void pairwise_distance_kernel_impl(
    const float* __restrict__ x1, const float* __restrict__ x2,
    float* __restrict__ output, uint32_t N, uint32_t D, float p) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float sum = 0.0f;
    if (p == 2.0f) {
        for (uint32_t j = 0; j < D; j++) {
            float diff = x1[i * D + j] - x2[i * D + j];
            sum += diff * diff;
        }
        output[i] = sqrtf(sum);
    } else if (p == 1.0f) {
        for (uint32_t j = 0; j < D; j++) {
            sum += fabsf(x1[i * D + j] - x2[i * D + j]);
        }
        output[i] = sum;
    } else {
        for (uint32_t j = 0; j < D; j++) {
            sum += powf(fabsf(x1[i * D + j] - x2[i * D + j]), p);
        }
        output[i] = powf(sum, 1.0f / p);
    }
}

auto pairwise_distance_kernel(const Tensor& x1, const Tensor& x2, double p,
                               hipStream_t stream) -> Tensor {
    Tensor a = (x1.dtype() == DType::Float32) ? x1.contiguous() : x1.contiguous().to(DType::Float32);
    Tensor b = (x2.dtype() == DType::Float32) ? x2.contiguous() : x2.contiguous().to(DType::Float32);

    int64_t N = a.shape()[0], D = a.shape()[1];
    Tensor result({N}, DType::Float32, x1.device());
    if (N == 0) return result;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    hipLaunchKernelGGL(pairwise_distance_kernel_impl, dim3(blocks), dim3(threads), 0, stream,
        a.data<float>(), b.data<float>(), result.data<float>(),
        static_cast<uint32_t>(N), static_cast<uint32_t>(D), static_cast<float>(p));
    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// Pdist (all-pairs pairwise distances)
// ============================================================================

__global__ void pdist_kernel_impl(
    const float* __restrict__ data, float* __restrict__ output,
    uint32_t N, uint32_t D, uint32_t num_pairs, float p) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    uint32_t i = 0, offset = 0;
    while (offset + (N - 1 - i) <= idx) {
        offset += (N - 1 - i);
        i++;
    }
    uint32_t j = idx - offset + i + 1;

    float sum = 0.0f;
    if (p == 2.0f) {
        for (uint32_t d = 0; d < D; d++) {
            float diff = data[i * D + d] - data[j * D + d];
            sum += diff * diff;
        }
        output[idx] = sqrtf(sum);
    } else if (p == 1.0f) {
        for (uint32_t d = 0; d < D; d++) {
            sum += fabsf(data[i * D + d] - data[j * D + d]);
        }
        output[idx] = sum;
    } else {
        for (uint32_t d = 0; d < D; d++) {
            sum += powf(fabsf(data[i * D + d] - data[j * D + d]), p);
        }
        output[idx] = powf(sum, 1.0f / p);
    }
}

auto pdist_kernel(const Tensor& input, double p, hipStream_t stream) -> Tensor {
    Tensor inf = (input.dtype() == DType::Float32) ? input.contiguous() : input.contiguous().to(DType::Float32);
    int64_t N = inf.shape()[0], D = inf.shape()[1];
    int64_t num_pairs = N * (N - 1) / 2;

    Tensor result({num_pairs}, DType::Float32, input.device());
    if (num_pairs == 0) return result;

    int threads = 256;
    int blocks = (num_pairs + threads - 1) / threads;
    hipLaunchKernelGGL(pdist_kernel_impl, dim3(blocks), dim3(threads), 0, stream,
        inf.data<float>(), result.data<float>(),
        static_cast<uint32_t>(N), static_cast<uint32_t>(D),
        static_cast<uint32_t>(num_pairs), static_cast<float>(p));
    HIP_CHECK(hipGetLastError());
    return result;
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
    double* d_params;
    size_t params_bytes = h_params.size() * sizeof(double);
    HIP_CHECK(hipMalloc(&d_params, params_bytes));
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

    HIP_CHECK(hipFree(d_params));

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
