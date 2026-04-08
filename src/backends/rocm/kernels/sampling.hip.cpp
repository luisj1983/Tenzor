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
#include <stdexcept>
#include <vector>

namespace tenzor {
namespace rocm {

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

auto multinomial_kernel(const Tensor& probs, int64_t num_samples,
                        bool /*replacement*/, hipStream_t stream) -> Tensor {
    auto input = probs.contiguous();
    if (input.dtype() != DType::Float32) input = input.to(DType::Float32);

    bool was_1d = (input.dim() == 1);
    if (was_1d) input = input.reshape({1, input.numel()});

    int64_t batch_size = input.shape()[0];
    int64_t num_categories = input.shape()[1];
    Tensor result({batch_size, num_samples}, DType::Int64, input.device());
    Tensor cdf_buf({batch_size, num_categories}, DType::Float32, input.device());

    uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

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
                                       float min_val, float bin_width) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    float val = input[tid];
    int64_t bin = static_cast<int64_t>((val - min_val) / bin_width);
    if (bin < 0) bin = 0;
    if (bin >= num_bins) bin = num_bins - 1;
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
    float lmin = (tid < n) ? input[tid] : INFINITY;
    float lmax = (tid < n) ? input[tid] : -INFINITY;
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
            n, bins, static_cast<float>(min_val), bin_width);
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

auto cdist_kernel(const Tensor& x1, const Tensor& x2, double /*p*/,
                  hipStream_t stream) -> Tensor {
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
    if (B == 0 || P == 0 || R == 0) return result;

    dim3 threads(16, 16, 1);
    dim3 blocks(static_cast<unsigned>((R + 15) / 16),
                static_cast<unsigned>((P + 15) / 16),
                static_cast<unsigned>(B));
    hipLaunchKernelGGL(cdist_l2_kernel_impl,
        blocks, threads, 0, stream,
        a.data<float>(), b.data<float>(), result.data<float>(), B, P, R, M);
    HIP_CHECK(hipGetLastError());
    return result;
}

}  // namespace rocm
}  // namespace tenzor
