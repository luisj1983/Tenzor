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
struct HistogramddMinMaxTag {};
struct HistogramddBinTag {};
struct HistogramddEdgesTag {};
struct HistogramddDensityTag {};
struct CDistKernelTag {};
struct CDistL2Tag {};
struct CDistL1Tag {};
struct CDistLpTag {};
struct TrapezoidKernelTag {};
struct CumulativeTrapezoidKernelTag {};
struct GradientKernelTag {};
struct PairwiseDistKernelTag {};
struct PdistKernelTag {};

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
        const float local_max = static_cast<float>(max_val);
        const int64_t local_bins = bins;
        queue.parallel_for<HistogramKernelTag>(sycl::range<1>(n), [=](sycl::id<1> idx_) {
            int64_t i = static_cast<int64_t>(idx_);
            float val = in_ptr[i];
            // Match CPU semantics: drop out-of-range samples.
            if (val < local_min || val > local_max) return;
            int64_t bin = static_cast<int64_t>((val - local_min) / bin_width);
            if (bin >= local_bins) bin = local_bins - 1;
            if (bin < 0) bin = 0;
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
// Histogramdd (multi-dimensional histogram)
// =========================================================================

auto histogramdd_kernel(const Tensor& input, std::vector<int64_t> bins,
                        std::vector<std::pair<double,double>> ranges, bool density,
                        sycl::queue& queue) -> std::pair<Tensor, std::vector<Tensor>> {
    if (input.dim() != 2) {
        throw std::runtime_error("histogramdd_kernel: input must be 2-D (N, D)");
    }

    const int64_t N = input.shape()[0];
    const int64_t D = input.shape()[1];

    if (static_cast<int64_t>(bins.size()) != D) {
        throw std::runtime_error("histogramdd_kernel: bins length must equal D");
    }

    auto in_contig = input.contiguous();
    if (in_contig.dtype() != DType::Float32) in_contig = in_contig.to(DType::Float32);
    const float* in_ptr = get_data_ptr<const float>(in_contig);
    const auto& device = input.device();

    // Auto-detect ranges from data if not provided
    bool auto_range = ranges.empty();
    if (auto_range) {
        ranges.resize(static_cast<size_t>(D));
    }

    if (auto_range && N > 0) {
        // Per-dimension min/max: allocate D min/max pairs on device
        float* d_mins = sycl::malloc_device<float>(static_cast<size_t>(D), queue);
        float* d_maxs = sycl::malloc_device<float>(static_cast<size_t>(D), queue);

        // Initialize from first sample
        queue.memcpy(d_mins, &in_ptr[0], static_cast<size_t>(D) * sizeof(float));
        queue.memcpy(d_maxs, &in_ptr[0], static_cast<size_t>(D) * sizeof(float));
        queue.wait();

        // Parallel min/max over all samples using atomic_ref (no sycl::reduction needed)
        const int64_t local_D = D;
        const int64_t local_N = N;
        queue.parallel_for<HistogramddMinMaxTag>(sycl::range<1>(N * D), [=](sycl::id<1> idx_) {
            int64_t linear = static_cast<int64_t>(idx_);
            int64_t i = linear / local_D;
            int64_t dd = linear % local_D;
            float v = in_ptr[i * local_D + dd];

            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                              sycl::memory_scope::device,
                              sycl::access::address_space::global_space>
                atomic_min(d_mins[dd]);
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                              sycl::memory_scope::device,
                              sycl::access::address_space::global_space>
                atomic_max(d_maxs[dd]);
            atomic_min.fetch_min(v);
            atomic_max.fetch_max(v);
        });
        queue.wait();

        // Copy results back
        std::vector<float> h_mins(static_cast<size_t>(D)), h_maxs(static_cast<size_t>(D));
        queue.memcpy(h_mins.data(), d_mins, static_cast<size_t>(D) * sizeof(float)).wait();
        queue.memcpy(h_maxs.data(), d_maxs, static_cast<size_t>(D) * sizeof(float)).wait();
        sycl::free(d_mins, queue);
        sycl::free(d_maxs, queue);

        for (int64_t d = 0; d < D; ++d) {
            float mn = h_mins[static_cast<size_t>(d)];
            float mx = h_maxs[static_cast<size_t>(d)];
            if (mn == mx) {
                mn -= 0.5f;
                mx += 0.5f;
            }
            ranges[static_cast<size_t>(d)] = {static_cast<double>(mn),
                                                static_cast<double>(mx)};
        }
    } else if (auto_range) {
        for (int64_t d = 0; d < D; ++d) {
            ranges[static_cast<size_t>(d)] = {0.0, 1.0};
        }
    }

    // Build per-dimension parameters
    std::vector<float> dim_min_vec(static_cast<size_t>(D));
    std::vector<float> dim_step_vec(static_cast<size_t>(D));

    for (int64_t d = 0; d < D; ++d) {
        auto sd = static_cast<size_t>(d);
        float fmin = static_cast<float>(ranges[sd].first);
        float fmax = static_cast<float>(ranges[sd].second);
        float step = (fmax - fmin) / static_cast<float>(bins[sd]);
        dim_min_vec[sd] = fmin;
        dim_step_vec[sd] = step;
    }

    // Compute strides (row-major)
    std::vector<int64_t> out_shape(bins.begin(), bins.end());
    std::vector<int64_t> out_strides_vec(static_cast<size_t>(D));
    int64_t stride = 1;
    for (int64_t d = D - 1; d >= 0; --d) {
        out_strides_vec[static_cast<size_t>(d)] = stride;
        stride *= bins[static_cast<size_t>(d)];
    }
    int64_t total_bins = stride;

    // Allocate device buffers for dim parameters and strides
    float* d_dim_min = sycl::malloc_device<float>(static_cast<size_t>(D), queue);
    float* d_dim_step = sycl::malloc_device<float>(static_cast<size_t>(D), queue);
    int64_t* d_strides = sycl::malloc_device<int64_t>(static_cast<size_t>(D), queue);
    int64_t* d_bins = sycl::malloc_device<int64_t>(static_cast<size_t>(D), queue);

    queue.memcpy(d_dim_min, dim_min_vec.data(), static_cast<size_t>(D) * sizeof(float));
    queue.memcpy(d_dim_step, dim_step_vec.data(), static_cast<size_t>(D) * sizeof(float));
    queue.memcpy(d_strides, out_strides_vec.data(), static_cast<size_t>(D) * sizeof(int64_t));
    queue.memcpy(d_bins, bins.data(), static_cast<size_t>(D) * sizeof(int64_t));
    queue.wait();

    // Allocate counts
    Tensor counts(out_shape, DType::Int64, device);
    int64_t* counts_ptr = get_data_ptr<int64_t>(counts);
    queue.memset(counts_ptr, 0, static_cast<size_t>(total_bins) * sizeof(int64_t)).wait();

    // Bin each sample
    if (N > 0) {
        const int64_t local_D = D;
        queue.parallel_for<HistogramddBinTag>(sycl::range<1>(N), [=](sycl::id<1> idx_) {
            int64_t i = static_cast<int64_t>(idx_);
            int64_t flat = 0;
            bool in_range = true;

            for (int64_t dd = 0; dd < local_D; ++dd) {
                float v = in_ptr[i * local_D + dd];
                float fmin_d = d_dim_min[dd];
                float step_d = d_dim_step[dd];
                int64_t nb = d_bins[dd];

                float range_max = fmin_d + step_d * static_cast<float>(nb);
                if (v < fmin_d || v > range_max) {
                    in_range = false;
                    break;
                }

                int64_t b = static_cast<int64_t>((v - fmin_d) / step_d);
                if (b >= nb) b = nb - 1;
                if (b < 0) b = 0;
                flat += b * d_strides[dd];
            }

            if (in_range) {
                sycl::atomic_ref<int64_t, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>
                    atomic_count(counts_ptr[flat]);
                atomic_count.fetch_add(int64_t{1});
            }
        });
        queue.wait();
    }

    // Build edge tensors on-device
    std::vector<Tensor> edges_vec;
    edges_vec.reserve(static_cast<size_t>(D));
    for (int64_t d = 0; d < D; ++d) {
        auto sd = static_cast<size_t>(d);
        int64_t nb = bins[sd];
        int64_t num_edges = nb + 1;
        Tensor edge({num_edges}, DType::Float32, device);
        float* edge_ptr = get_data_ptr<float>(edge);
        const float local_min = dim_min_vec[sd];
        const float local_step = dim_step_vec[sd];
        queue.parallel_for<HistogramddEdgesTag>(sycl::range<1>(num_edges),
            [=](sycl::id<1> idx_) {
                int64_t j = static_cast<int64_t>(idx_);
                edge_ptr[j] = local_min + static_cast<float>(j) * local_step;
            });
        queue.wait();
        edges_vec.push_back(std::move(edge));
    }

    // Density normalization on device
    Tensor result = counts;
    if (density && N > 0) {
        double bin_volume = 1.0;
        for (int64_t d = 0; d < D; ++d) {
            bin_volume *= static_cast<double>(dim_step_vec[static_cast<size_t>(d)]);
        }
        double norm = static_cast<double>(N) * bin_volume;
        float inv_norm = static_cast<float>(1.0 / norm);

        Tensor density_out(out_shape, DType::Float32, device);
        float* ddata = get_data_ptr<float>(density_out);
        const int64_t local_total = total_bins;
        queue.parallel_for<HistogramddDensityTag>(sycl::range<1>(total_bins),
            [=](sycl::id<1> idx_) {
                int64_t j = static_cast<int64_t>(idx_);
                ddata[j] = static_cast<float>(counts_ptr[j]) * inv_norm;
            });
        queue.wait();
        result = density_out;
    }

    // Free device buffers
    sycl::free(d_dim_min, queue);
    sycl::free(d_dim_step, queue);
    sycl::free(d_strides, queue);
    sycl::free(d_bins, queue);

    return {result, std::move(edges_vec)};
}

// =========================================================================
// CDist (pairwise L2 distance)
// =========================================================================

auto cdist_kernel(const Tensor& x1, const Tensor& x2, double p,
                  sycl::queue& queue) -> Tensor {
    // Preserve the input dtype on output. Compute in Float64 when either
    // input is Float64, otherwise compute in Float32 and narrow Float16 /
    // BFloat16 back at the end.
    const DType orig_dtype = (x1.dtype() == DType::Float64 || x2.dtype() == DType::Float64)
                                 ? DType::Float64
                                 : x1.dtype();
    const DType compute_dtype = (orig_dtype == DType::Float64) ? DType::Float64
                                                               : DType::Float32;

    auto a = x1.contiguous();
    auto b = x2.contiguous();
    if (a.dtype() != compute_dtype) a = a.to(compute_dtype);
    if (b.dtype() != compute_dtype) b = b.to(compute_dtype);

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
        return (compute_dtype != orig_dtype) ? result.to(orig_dtype) : result;
    }

    if (compute_dtype == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        float* out_ptr = get_data_ptr<float>(result);

        // Capture p into the kernel with branch specialization at launch time
        // to avoid the pow() cost for the L1 and L2 paths.
        const float p_f = static_cast<float>(p);
        if (p == 2.0) {
            queue.parallel_for<CDistL2Tag>(sycl::range<3>(B, P, R),
                [=](sycl::id<3> idx) {
                    int64_t bi = idx[0];
                    int64_t p_idx = idx[1];
                    int64_t r  = idx[2];
                    const float* a_b = a_ptr + bi * P * M;
                    const float* b_b = b_ptr + bi * R * M;
                    float sum = 0.0f;
                    for (int64_t m = 0; m < M; ++m) {
                        float diff = a_b[p_idx * M + m] - b_b[r * M + m];
                        sum += diff * diff;
                    }
                    out_ptr[(bi * P + p_idx) * R + r] = sycl::sqrt(sum);
                });
        } else if (p == 1.0) {
            queue.parallel_for<CDistL1Tag>(sycl::range<3>(B, P, R),
                [=](sycl::id<3> idx) {
                    int64_t bi = idx[0];
                    int64_t p_idx = idx[1];
                    int64_t r  = idx[2];
                    const float* a_b = a_ptr + bi * P * M;
                    const float* b_b = b_ptr + bi * R * M;
                    float sum = 0.0f;
                    for (int64_t m = 0; m < M; ++m) {
                        sum += sycl::fabs(a_b[p_idx * M + m] - b_b[r * M + m]);
                    }
                    out_ptr[(bi * P + p_idx) * R + r] = sum;
                });
        } else {
            queue.parallel_for<CDistLpTag>(sycl::range<3>(B, P, R),
                [=](sycl::id<3> idx) {
                    int64_t bi = idx[0];
                    int64_t p_idx = idx[1];
                    int64_t r  = idx[2];
                    const float* a_b = a_ptr + bi * P * M;
                    const float* b_b = b_ptr + bi * R * M;
                    float sum = 0.0f;
                    for (int64_t m = 0; m < M; ++m) {
                        float diff = sycl::fabs(a_b[p_idx * M + m] - b_b[r * M + m]);
                        sum += sycl::pow(diff, p_f);
                    }
                    out_ptr[(bi * P + p_idx) * R + r] = sycl::pow(sum, 1.0f / p_f);
                });
        }
    } else {  // Float64
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        double* out_ptr = get_data_ptr<double>(result);

        const double p_d = p;
        if (p == 2.0) {
            queue.parallel_for<class CDistL2F64Tag>(sycl::range<3>(B, P, R),
                [=](sycl::id<3> idx) {
                    int64_t bi = idx[0];
                    int64_t p_idx = idx[1];
                    int64_t r  = idx[2];
                    const double* a_b = a_ptr + bi * P * M;
                    const double* b_b = b_ptr + bi * R * M;
                    double sum = 0.0;
                    for (int64_t m = 0; m < M; ++m) {
                        double diff = a_b[p_idx * M + m] - b_b[r * M + m];
                        sum += diff * diff;
                    }
                    out_ptr[(bi * P + p_idx) * R + r] = sycl::sqrt(sum);
                });
        } else if (p == 1.0) {
            queue.parallel_for<class CDistL1F64Tag>(sycl::range<3>(B, P, R),
                [=](sycl::id<3> idx) {
                    int64_t bi = idx[0];
                    int64_t p_idx = idx[1];
                    int64_t r  = idx[2];
                    const double* a_b = a_ptr + bi * P * M;
                    const double* b_b = b_ptr + bi * R * M;
                    double sum = 0.0;
                    for (int64_t m = 0; m < M; ++m) {
                        sum += sycl::fabs(a_b[p_idx * M + m] - b_b[r * M + m]);
                    }
                    out_ptr[(bi * P + p_idx) * R + r] = sum;
                });
        } else {
            queue.parallel_for<class CDistLpF64Tag>(sycl::range<3>(B, P, R),
                [=](sycl::id<3> idx) {
                    int64_t bi = idx[0];
                    int64_t p_idx = idx[1];
                    int64_t r  = idx[2];
                    const double* a_b = a_ptr + bi * P * M;
                    const double* b_b = b_ptr + bi * R * M;
                    double sum = 0.0;
                    for (int64_t m = 0; m < M; ++m) {
                        double diff = sycl::fabs(a_b[p_idx * M + m] - b_b[r * M + m]);
                        sum += sycl::pow(diff, p_d);
                    }
                    out_ptr[(bi * P + p_idx) * R + r] = sycl::pow(sum, 1.0 / p_d);
                });
        }
    }
    queue.wait();
    return (compute_dtype != orig_dtype) ? result.to(orig_dtype) : result;
}

// =========================================================================
// Poisson sampling (Knuth algorithm per work-item)
// =========================================================================

namespace {
struct PoissonSampleKernelTag {};
}  // namespace

auto poisson_sample_kernel(const Tensor& rates, sycl::queue& queue) -> Tensor {
    auto input = rates.contiguous();
    if (input.dtype() != DType::Float32) input = input.to(DType::Float32);

    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, DType::Int64, input.device());
    int64_t n = input.numel();
    if (n == 0) return result;

    const float* in_ptr = get_data_ptr<const float>(input);
    int64_t* out_ptr = get_data_ptr<int64_t>(result);
    uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    queue.parallel_for<PoissonSampleKernelTag>(sycl::range<1>(n), [=](sycl::id<1> idx_) {
        int64_t i = static_cast<int64_t>(idx_);

        // Per-element LCG PRNG (same constants as bernoulli_kernel)
        uint64_t state = seed + static_cast<uint64_t>(i) * 6364136223846793005ULL + 1442695040888963407ULL;

        float lambda = in_ptr[i];

        // Knuth algorithm: generate Poisson(lambda) by counting uniform
        // samples until their product drops below exp(-lambda).
        float L = sycl::exp(-lambda);
        int64_t k = 0;
        float p = 1.0f;

        do {
            ++k;
            // Advance LCG and produce a uniform in (0, 1)
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            float u = static_cast<float>(state >> 33) / static_cast<float>(1ULL << 31);
            p *= u;
        } while (p > L && k < 1000000);  // guard against infinite loop for huge lambda

        out_ptr[i] = k - 1;
    });
    queue.wait();
    return result;
}

// =========================================================================
// Normal sampling (Box-Muller transform per work-item)
// =========================================================================

namespace {
struct NormalSampleKernelTag {};
}  // namespace

auto normal_sample_kernel(const Tensor& mean, const Tensor& stddev, sycl::queue& queue) -> Tensor {
    auto m = mean.contiguous();
    auto s = stddev.contiguous();
    if (m.dtype() != DType::Float32) m = m.to(DType::Float32);
    if (s.dtype() != DType::Float32) s = s.to(DType::Float32);

    std::vector<int64_t> shape(m.shape().begin(), m.shape().end());
    Tensor result(shape, DType::Float32, m.device());
    int64_t n = m.numel();
    if (n == 0) return result;

    const float* mean_ptr = get_data_ptr<const float>(m);
    const float* std_ptr = get_data_ptr<const float>(s);
    float* out_ptr = get_data_ptr<float>(result);
    uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    queue.parallel_for<NormalSampleKernelTag>(sycl::range<1>(n), [=](sycl::id<1> idx_) {
        int64_t i = static_cast<int64_t>(idx_);

        uint64_t state = seed + static_cast<uint64_t>(i) * 6364136223846793005ULL + 1442695040888963407ULL;

        // Generate two uniforms for Box-Muller
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        float u1 = static_cast<float>(state >> 33) / static_cast<float>(1ULL << 31);
        u1 = sycl::fmax(u1, 1.0e-7f);  // Clamp away from zero

        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        float u2 = static_cast<float>(state >> 33) / static_cast<float>(1ULL << 31);

        // Box-Muller transform
        float z = sycl::sqrt(-2.0f * sycl::log(u1)) *
                  sycl::cos(2.0f * 3.14159265358979323846f * u2);

        out_ptr[i] = mean_ptr[i] + std_ptr[i] * z;
    });
    queue.wait();
    return result;
}

// =========================================================================
// Exponential sampling (inverse CDF per work-item)
// =========================================================================

namespace {
struct ExponentialSampleKernelTag {};
}  // namespace

auto exponential_sample_kernel(const Tensor& rate, sycl::queue& queue) -> Tensor {
    auto input = rate.contiguous();
    if (input.dtype() != DType::Float32) input = input.to(DType::Float32);

    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, DType::Float32, input.device());
    int64_t n = input.numel();
    if (n == 0) return result;

    const float* rate_ptr = get_data_ptr<const float>(input);
    float* out_ptr = get_data_ptr<float>(result);
    uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    queue.parallel_for<ExponentialSampleKernelTag>(sycl::range<1>(n), [=](sycl::id<1> idx_) {
        int64_t i = static_cast<int64_t>(idx_);

        uint64_t state = seed + static_cast<uint64_t>(i) * 6364136223846793005ULL + 1442695040888963407ULL;
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        float u = static_cast<float>(state >> 33) / static_cast<float>(1ULL << 31);
        u = sycl::fmin(u, 1.0f - 1.0e-7f);  // Clamp away from 1.0

        // Inverse CDF: -log(1 - u) / rate
        out_ptr[i] = -sycl::log(1.0f - u) / rate_ptr[i];
    });
    queue.wait();
    return result;
}

// =========================================================================
// Trapezoid integration
// =========================================================================

auto trapezoid_kernel(const Tensor& y, int64_t dim, double dx,
                       const Tensor* x_ptr, sycl::queue& queue) -> Tensor {
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

    const float* y_ptr = get_data_ptr<const float>(yf);
    float* out_ptr = get_data_ptr<float>(result);
    bool has_x = x_ptr != nullptr;
    Tensor xf;
    const float* xd = nullptr;
    if (has_x) {
        xf = (x_ptr->dtype() == DType::Float32) ? x_ptr->contiguous() : x_ptr->contiguous().to(DType::Float32);
        xd = get_data_ptr<const float>(xf);
    }

    float dx_f = static_cast<float>(dx);
    int64_t n_val = n, inner_val = inner;

    queue.parallel_for<TrapezoidKernelTag>(sycl::range<1>(total), [=](sycl::id<1> id) {
        int64_t idx = id[0];
        int64_t o = idx / inner_val;
        int64_t i_inner = idx % inner_val;

        float sum = 0.0f;
        for (int64_t k = 0; k < n_val - 1; k++) {
            int64_t idx_k  = (o * n_val + k) * inner_val + i_inner;
            int64_t idx_k1 = (o * n_val + k + 1) * inner_val + i_inner;
            float h = has_x ? (xd[idx_k1] - xd[idx_k]) : dx_f;
            sum += 0.5f * (y_ptr[idx_k] + y_ptr[idx_k1]) * h;
        }
        out_ptr[idx] = sum;
    });
    queue.wait();
    return result;
}

// =========================================================================
// Cumulative trapezoid integration
// =========================================================================

auto cumulative_trapezoid_kernel(const Tensor& y, int64_t dim, double dx,
                                  const Tensor* x_ptr, sycl::queue& queue) -> Tensor {
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

    const float* y_ptr = get_data_ptr<const float>(yf);
    float* out_ptr = get_data_ptr<float>(result);
    bool has_x = x_ptr != nullptr;
    Tensor xf;
    const float* xd = nullptr;
    if (has_x) {
        xf = (x_ptr->dtype() == DType::Float32) ? x_ptr->contiguous() : x_ptr->contiguous().to(DType::Float32);
        xd = get_data_ptr<const float>(xf);
    }

    float dx_f = static_cast<float>(dx);
    int64_t n_val = n, inner_val = inner;

    queue.parallel_for<CumulativeTrapezoidKernelTag>(sycl::range<1>(total), [=](sycl::id<1> id) {
        int64_t idx = id[0];
        int64_t o = idx / inner_val;
        int64_t i_inner = idx % inner_val;

        float cumsum = 0.0f;
        for (int64_t k = 0; k < n_val - 1; k++) {
            int64_t idx_k  = (o * n_val + k) * inner_val + i_inner;
            int64_t idx_k1 = (o * n_val + k + 1) * inner_val + i_inner;
            float h = has_x ? (xd[idx_k1] - xd[idx_k]) : dx_f;
            cumsum += 0.5f * (y_ptr[idx_k] + y_ptr[idx_k1]) * h;
            out_ptr[(o * (n_val - 1) + k) * inner_val + i_inner] = cumsum;
        }
    });
    queue.wait();
    return result;
}

// =========================================================================
// Numerical gradient (central differences)
// =========================================================================

auto gradient_kernel(const Tensor& input, int64_t dim, double spacing,
                      sycl::queue& queue) -> Tensor {
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

    const float* in_ptr = get_data_ptr<const float>(inf);
    float* out_ptr = get_data_ptr<float>(result);
    float h = static_cast<float>(spacing);
    int64_t n_val = n, inner_val = inner;

    queue.parallel_for<GradientKernelTag>(sycl::range<1>(total), [=](sycl::id<1> id) {
        int64_t idx = id[0];
        int64_t o = idx / inner_val;
        int64_t i_inner = idx % inner_val;

        auto at = [&](int64_t k) -> float {
            return in_ptr[(o * n_val + k) * inner_val + i_inner];
        };

        out_ptr[(o * n_val + 0) * inner_val + i_inner] = (at(1) - at(0)) / h;
        for (int64_t k = 1; k < n_val - 1; k++) {
            out_ptr[(o * n_val + k) * inner_val + i_inner] = (at(k + 1) - at(k - 1)) / (2.0f * h);
        }
        out_ptr[(o * n_val + n_val - 1) * inner_val + i_inner] = (at(n_val - 1) - at(n_val - 2)) / h;
    });
    queue.wait();
    return result;
}

// =========================================================================
// Pairwise distance
// =========================================================================

auto pairwise_distance_kernel(const Tensor& x1, const Tensor& x2, double p,
                               sycl::queue& queue) -> Tensor {
    Tensor a = (x1.dtype() == DType::Float32) ? x1.contiguous() : x1.contiguous().to(DType::Float32);
    Tensor b = (x2.dtype() == DType::Float32) ? x2.contiguous() : x2.contiguous().to(DType::Float32);

    int64_t N = a.shape()[0], D = a.shape()[1];
    Tensor result({N}, DType::Float32, x1.device());
    if (N == 0) return result;

    const float* a_ptr = get_data_ptr<const float>(a);
    const float* b_ptr = get_data_ptr<const float>(b);
    float* out_ptr = get_data_ptr<float>(result);
    float p_f = static_cast<float>(p);

    queue.parallel_for<PairwiseDistKernelTag>(sycl::range<1>(N), [=](sycl::id<1> id) {
        int64_t i = id[0];
        float sum = 0.0f;
        if (p_f == 2.0f) {
            for (int64_t j = 0; j < D; j++) {
                float diff = a_ptr[i * D + j] - b_ptr[i * D + j];
                sum += diff * diff;
            }
            out_ptr[i] = sycl::sqrt(sum);
        } else if (p_f == 1.0f) {
            for (int64_t j = 0; j < D; j++) {
                sum += sycl::fabs(a_ptr[i * D + j] - b_ptr[i * D + j]);
            }
            out_ptr[i] = sum;
        } else {
            for (int64_t j = 0; j < D; j++) {
                sum += sycl::pow(sycl::fabs(a_ptr[i * D + j] - b_ptr[i * D + j]), p_f);
            }
            out_ptr[i] = sycl::pow(sum, 1.0f / p_f);
        }
    });
    queue.wait();
    return result;
}

// =========================================================================
// Pdist (all-pairs pairwise distances)
// =========================================================================

auto pdist_kernel(const Tensor& input, double p, sycl::queue& queue) -> Tensor {
    Tensor inf = (input.dtype() == DType::Float32) ? input.contiguous() : input.contiguous().to(DType::Float32);
    int64_t N = inf.shape()[0], D = inf.shape()[1];
    int64_t num_pairs = N * (N - 1) / 2;

    Tensor result({num_pairs}, DType::Float32, input.device());
    if (num_pairs == 0) return result;

    const float* data = get_data_ptr<const float>(inf);
    float* out_ptr = get_data_ptr<float>(result);
    float p_f = static_cast<float>(p);

    queue.parallel_for<PdistKernelTag>(sycl::range<1>(num_pairs), [=](sycl::id<1> id) {
        int64_t idx = id[0];
        // Map flat index to (i, j) pair where i < j
        int64_t i = 0, offset = 0;
        while (offset + (N - 1 - i) <= idx) {
            offset += (N - 1 - i);
            i++;
        }
        int64_t j = idx - offset + i + 1;

        float sum = 0.0f;
        if (p_f == 2.0f) {
            for (int64_t d = 0; d < D; d++) {
                float diff = data[i * D + d] - data[j * D + d];
                sum += diff * diff;
            }
            out_ptr[idx] = sycl::sqrt(sum);
        } else if (p_f == 1.0f) {
            for (int64_t d = 0; d < D; d++) {
                sum += sycl::fabs(data[i * D + d] - data[j * D + d]);
            }
            out_ptr[idx] = sum;
        } else {
            for (int64_t d = 0; d < D; d++) {
                sum += sycl::pow(sycl::fabs(data[i * D + d] - data[j * D + d]), p_f);
            }
            out_ptr[idx] = sycl::pow(sum, 1.0f / p_f);
        }
    });
    queue.wait();
    return result;
}

}  // namespace oneapi
}  // namespace tenzor
