/**
 * @file sampling.cpp
 * @brief OneAPI/SYCL implementations of Bernoulli, Multinomial, Bucketize,
 *        Histogram, CDist. Replaces the previous CPU-roundtrip fallbacks.
 *
 * Algorithms mirror src/backends/cuda/kernels/advanced.cu line-for-line.
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace tenzor {
namespace oneapi {

namespace {

template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

struct BernoulliKernelTag {};
struct MultinomialCdfKernelTag {};
struct MultinomialSampleKernelTag {};
struct BucketizeKernelTag {};
struct HistogramKernelTag {};
struct HistogramFillEdgesTag {};
struct HistogramMinMaxTag {};
struct CDistKernelTag {};

}  // namespace

// =========================================================================
// Bernoulli sampling
// =========================================================================

auto bernoulli_kernel(const Tensor& probs, sycl::queue& queue) -> Tensor {
    auto input = probs.contiguous();
    if (input.dtype() != DType::Float32) input = input.to(DType::Float32);

    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, DType::Float32, input.device());
    int64_t n = input.numel();
    if (n == 0) return result;

    const float* in_ptr = get_data_ptr<const float>(input);
    float* out_ptr = get_data_ptr<float>(result);
    uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    queue.parallel_for<BernoulliKernelTag>(sycl::range<1>(n), [=](sycl::id<1> idx_) {
        int64_t i = static_cast<int64_t>(idx_);
        uint64_t state = seed + static_cast<uint64_t>(i) * 6364136223846793005ULL + 1442695040888963407ULL;
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        float u = static_cast<float>(state >> 33) / static_cast<float>(1ULL << 31);
        out_ptr[i] = (u < in_ptr[i]) ? 1.0f : 0.0f;
    });
    queue.wait();
    return result;
}

// =========================================================================
// Multinomial sampling
// =========================================================================

auto multinomial_kernel(const Tensor& probs, int64_t num_samples,
                        bool /*replacement*/, sycl::queue& queue) -> Tensor {
    auto input = probs.contiguous();
    if (input.dtype() != DType::Float32) input = input.to(DType::Float32);

    bool was_1d = (input.dim() == 1);
    if (was_1d) input = input.reshape({1, input.numel()});

    int64_t batch_size = input.shape()[0];
    int64_t num_categories = input.shape()[1];
    Tensor result(std::vector<int64_t>{batch_size, num_samples}, DType::Int64, input.device());
    Tensor cdf_buf(std::vector<int64_t>{batch_size, num_categories}, DType::Float32, input.device());

    uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    const float* in_base = get_data_ptr<const float>(input);
    float* cdf_base = get_data_ptr<float>(cdf_buf);
    int64_t* out_base = get_data_ptr<int64_t>(result);

    for (int64_t b = 0; b < batch_size; ++b) {
        const float* prob_ptr = in_base + b * num_categories;
        float* cdf_ptr = cdf_base + b * num_categories;
        int64_t* out_ptr = out_base + b * num_samples;
        const int64_t nc = num_categories;

        // Inclusive prefix sum: simple serial single-task (small num_categories)
        queue.single_task([=]() {
            float acc = 0.0f;
            for (int64_t i = 0; i < nc; ++i) {
                acc += prob_ptr[i];
                cdf_ptr[i] = acc;
            }
        }).wait();

        // Read total (single scalar D2H sync — necessary metadata)
        float total = 0.0f;
        queue.memcpy(&total, cdf_ptr + nc - 1, sizeof(float)).wait();
        if (total <= 0.0f) total = 1.0f;

        // Sample via binary search
        uint64_t batch_seed = seed + b * 1000003;
        const int64_t ns = num_samples;
        queue.parallel_for<MultinomialSampleKernelTag>(sycl::range<1>(ns),
            [=](sycl::id<1> idx_) {
                int64_t sid = static_cast<int64_t>(idx_);
                uint64_t state = batch_seed + static_cast<uint64_t>(sid) * 6364136223846793005ULL + 1442695040888963407ULL;
                state = state * 6364136223846793005ULL + 1442695040888963407ULL;
                float u = static_cast<float>(state >> 33) / static_cast<float>(1ULL << 31) * total;

                int64_t lo = 0, hi = nc - 1;
                while (lo < hi) {
                    int64_t mid = (lo + hi) / 2;
                    if (cdf_ptr[mid] <= u) lo = mid + 1;
                    else hi = mid;
                }
                out_ptr[sid] = lo;
            });
        queue.wait();
    }

    if (was_1d) result = result.reshape({num_samples});
    return result;
}

// =========================================================================
// Bucketize
// =========================================================================

auto bucketize_kernel(const Tensor& input, const Tensor& boundaries,
                      bool right, sycl::queue& queue) -> Tensor {
    auto in_contig = input.contiguous();
    auto bound_contig = boundaries.contiguous();
    if (in_contig.dtype() != DType::Float32)    in_contig = in_contig.to(DType::Float32);
    if (bound_contig.dtype() != DType::Float32) bound_contig = bound_contig.to(DType::Float32);

    std::vector<int64_t> shape(in_contig.shape().begin(), in_contig.shape().end());
    Tensor result(shape, DType::Int64, in_contig.device());
    int64_t n = in_contig.numel();
    if (n == 0) return result;

    const float* in_ptr = get_data_ptr<const float>(in_contig);
    const float* bound_ptr = get_data_ptr<const float>(bound_contig);
    int64_t* out_ptr = get_data_ptr<int64_t>(result);
    int64_t num_boundaries = bound_contig.numel();

    queue.parallel_for<BucketizeKernelTag>(sycl::range<1>(n), [=](sycl::id<1> idx_) {
        int64_t i = static_cast<int64_t>(idx_);
        float val = in_ptr[i];
        int64_t lo = 0, hi = num_boundaries;
        while (lo < hi) {
            int64_t mid = (lo + hi) / 2;
            bool cond = right ? (bound_ptr[mid] <= val) : (bound_ptr[mid] < val);
            if (cond) lo = mid + 1;
            else hi = mid;
        }
        out_ptr[i] = lo;
    });
    queue.wait();
    return result;
}

// =========================================================================
// Histogram (with on-device min/max + bin-edge fill)
// =========================================================================

auto histogram_kernel(const Tensor& input, int64_t bins,
                      double min_val, double max_val,
                      sycl::queue& queue) -> std::pair<Tensor, Tensor> {
    auto in_contig = input.contiguous();
    if (in_contig.dtype() != DType::Float32) in_contig = in_contig.to(DType::Float32);
    int64_t n = in_contig.numel();
    const float* in_ptr = get_data_ptr<const float>(in_contig);

    if (min_val == 0.0 && max_val == 0.0 && n > 0) {
        // On-device parallel min/max via sycl::reduction
        float* d_min = sycl::malloc_device<float>(1, queue);
        float* d_max = sycl::malloc_device<float>(1, queue);
        queue.memcpy(d_min, &in_ptr[0], sizeof(float));  // initialise to first element approx
        queue.memcpy(d_max, &in_ptr[0], sizeof(float));
        queue.wait();

        // Use sycl::reduction with min and max operators
        queue.parallel_for<HistogramMinMaxTag>(sycl::range<1>(n),
            sycl::reduction(d_min, sycl::minimum<float>()),
            sycl::reduction(d_max, sycl::maximum<float>()),
            [=](sycl::id<1> idx_, auto& mn, auto& mx) {
                float v = in_ptr[static_cast<int64_t>(idx_)];
                mn.combine(v);
                mx.combine(v);
            });

        float h_min = 0.0f, h_max = 0.0f;
        queue.memcpy(&h_min, d_min, sizeof(float)).wait();
        queue.memcpy(&h_max, d_max, sizeof(float)).wait();
        sycl::free(d_min, queue);
        sycl::free(d_max, queue);
        min_val = h_min;
        max_val = h_max;
    }
    if (max_val <= min_val) max_val = min_val + 1.0;

    float bin_width = static_cast<float>((max_val - min_val) / bins);

    Tensor counts(std::vector<int64_t>{bins}, DType::Int64, in_contig.device());
    int64_t* counts_ptr = get_data_ptr<int64_t>(counts);
    queue.memset(counts_ptr, 0, static_cast<size_t>(bins) * sizeof(int64_t)).wait();

    if (n > 0) {
        const float local_min = static_cast<float>(min_val);
        const int64_t local_bins = bins;
        queue.parallel_for<HistogramKernelTag>(sycl::range<1>(n), [=](sycl::id<1> idx_) {
            int64_t i = static_cast<int64_t>(idx_);
            float val = in_ptr[i];
            int64_t bin = static_cast<int64_t>((val - local_min) / bin_width);
            if (bin < 0) bin = 0;
            if (bin >= local_bins) bin = local_bins - 1;
            sycl::atomic_ref<int64_t, sycl::memory_order::relaxed,
                              sycl::memory_scope::device,
                              sycl::access::address_space::global_space>
                atomic_count(counts_ptr[bin]);
            atomic_count.fetch_add(int64_t{1});
        });
        queue.wait();
    }

    // Bin edges on-device
    Tensor edges(std::vector<int64_t>{bins + 1}, DType::Float32, in_contig.device());
    {
        float* edge_ptr = get_data_ptr<float>(edges);
        const int64_t num_edges = bins + 1;
        const float local_min2 = static_cast<float>(min_val);
        queue.parallel_for<HistogramFillEdgesTag>(sycl::range<1>(num_edges),
            [=](sycl::id<1> idx_) {
                int64_t i = static_cast<int64_t>(idx_);
                edge_ptr[i] = local_min2 + static_cast<float>(i) * bin_width;
            });
        queue.wait();
    }

    return {counts, edges};
}

// =========================================================================
// CDist (pairwise L2 distance)
// =========================================================================

auto cdist_kernel(const Tensor& x1, const Tensor& x2, double /*p*/,
                  sycl::queue& queue) -> Tensor {
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

    const float* a_ptr = get_data_ptr<const float>(a);
    const float* b_ptr = get_data_ptr<const float>(b);
    float* out_ptr = get_data_ptr<float>(result);

    queue.parallel_for<CDistKernelTag>(sycl::range<3>(B, P, R),
        [=](sycl::id<3> idx) {
            int64_t bi = idx[0];
            int64_t p  = idx[1];
            int64_t r  = idx[2];
            const float* a_b = a_ptr + bi * P * M;
            const float* b_b = b_ptr + bi * R * M;
            float sum = 0.0f;
            for (int64_t m = 0; m < M; ++m) {
                float diff = a_b[p * M + m] - b_b[r * M + m];
                sum += diff * diff;
            }
            out_ptr[(bi * P + p) * R + r] = sycl::sqrt(sum);
        });
    queue.wait();
    return result;
}

}  // namespace oneapi
}  // namespace tenzor
