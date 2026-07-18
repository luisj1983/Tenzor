/**
 * @file sampling.cpp
 * @brief OneAPI/SYCL implementations of Bernoulli, Multinomial, Bucketize,
 *        Histogram, CDist. Replaces the previous CPU-roundtrip fallbacks.
 *
 * Algorithms mirror src/backends/cuda/kernels/advanced.cu line-for-line.
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/creation.hpp"  // for tenzor::get_global_seed
#include "oneapi_kernel_utils.hpp"
#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <utility>

namespace tenzor {
namespace oneapi {

// Forward declaration: topk implementation lives in reduction.cpp.
auto topk_kernel(const Tensor& input, int64_t k, int64_t dim, bool largest,
                 bool sorted, sycl::queue& queue) -> std::pair<Tensor, Tensor>;

namespace {

struct BernoulliKernelTag {};
struct MultinomialCdfKernelTag {};
struct MultinomialSampleKernelTag {};
struct MultinomialEsKeysKernelTag {};
struct BucketizeKernelTag {};
struct HistogramKernelTag {};
struct HistogramFillEdgesTag {};
struct HistogramMinMaxTag {};
template <typename T> struct HistogramddMinMaxTag {};
template <typename T> struct HistogramddBinTag {};
template <typename T> struct HistogramddEdgesTag {};
template <typename T> struct HistogramddDensityTag {};
struct CDistKernelTag {};
struct CDistL2Tag {};
struct CDistL1Tag {};
struct CDistLpTag {};
struct CDistInfTag {};
struct CDistP0Tag {};
struct GammaSampleKernelTag {};
template<typename T> struct TrapezoidKernelTag {};
template<typename T> struct CumulativeTrapezoidKernelTag {};
template<typename T> struct GradientKernelTag {};
template<typename T> struct PairwiseDistKernelTag {};
template<typename T> struct PdistKernelTag {};

}  // namespace

// =========================================================================
// Bernoulli sampling
// =========================================================================

auto bernoulli_kernel(const Tensor& probs, sycl::queue& queue) -> Tensor {
    // Preserve the input dtype on output (matches CPU/other backends and the
    // DtypePreservation contract). Compute in Float32 internally.
    const DType orig_dtype = probs.dtype();
    auto input = probs.contiguous();
    if (input.dtype() != DType::Float32) input = input.to(DType::Float32);

    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, DType::Float32, input.device());
    int64_t n = input.numel();
    if (n == 0) return orig_dtype != DType::Float32 ? result.to(orig_dtype) : result;

    const float* in_ptr = get_data_ptr<const float>(input);
    float* out_ptr = get_data_ptr<float>(result);
    uint64_t seed = ::tenzor::get_global_seed();

    queue.parallel_for<BernoulliKernelTag>(sycl::range<1>(n), [=](sycl::id<1> idx_) {
        int64_t i = static_cast<int64_t>(idx_);
        uint64_t state = seed + static_cast<uint64_t>(i) * 6364136223846793005ULL + 1442695040888963407ULL;
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        float u = static_cast<float>(state >> 33) / static_cast<float>(1ULL << 31);
        out_ptr[i] = (u < in_ptr[i]) ? 1.0f : 0.0f;
    });
    queue.wait_and_throw();
    return orig_dtype != DType::Float32 ? result.to(orig_dtype) : result;
}

// =========================================================================
// Multinomial sampling
// =========================================================================

auto multinomial_kernel(const Tensor& probs, int64_t num_samples,
                        bool replacement, sycl::queue& queue) -> Tensor {
    auto input = probs.contiguous();
    if (input.dtype() != DType::Float32) input = input.to(DType::Float32);

    bool was_1d = (input.dim() == 1);
    if (was_1d) input = input.reshape({1, input.numel()});

    int64_t batch_size = input.shape()[0];
    int64_t num_categories = input.shape()[1];

    if (!replacement && num_samples > num_categories) {
        throw std::runtime_error("multinomial: cannot sample more than "
                                 "num_categories without replacement");
    }

    // Without replacement: Efraimidis-Spirakis weighted reservoir sampling
    // (Gumbel-top-k). For each (b, i) we compute key[b, i] = -log(U) / w
    // and select the `num_samples` smallest keys per row. Equivalent in
    // distribution to repeated weighted sampling without replacement and
    // runs entirely on-device. Mirrors the CUDA / ROCm implementations.
    if (!replacement) {
        Tensor keys({batch_size, num_categories}, DType::Float32, input.device());
        const float* in_ptr = get_data_ptr<const float>(input);
        float* keys_ptr = get_data_ptr<float>(keys);
        const int64_t total_keys = batch_size * num_categories;
        uint64_t seed = ::tenzor::get_global_seed();

        queue.parallel_for<MultinomialEsKeysKernelTag>(
            sycl::range<1>(static_cast<size_t>(total_keys)),
            [=](sycl::id<1> id) {
                int64_t tid = static_cast<int64_t>(id[0]);
                uint64_t state = seed +
                    static_cast<uint64_t>(tid) * 6364136223846793005ULL +
                    1442695040888963407ULL;
                state = state * 6364136223846793005ULL + 1442695040888963407ULL;
                float u = (static_cast<float>(state >> 33) + 1.0f) /
                          static_cast<float>(1ULL << 31);
                if (u >= 1.0f) u = 1.0f - 1.1920929e-07f;
                float w = in_ptr[tid];
                if (!(w > 0.0f)) {
                    keys_ptr[tid] = std::numeric_limits<float>::infinity();
                } else {
                    keys_ptr[tid] = -sycl::log(u) / w;
                }
            }).wait();

        auto [picked_vals, picked_idx] = topk_kernel(
            keys, num_samples, /*dim=*/1,
            /*largest=*/false, /*sorted=*/true, queue);
        Tensor result_idx = picked_idx;
        if (was_1d) result_idx = result_idx.reshape({num_samples});
        return result_idx;
    }

    Tensor result(std::vector<int64_t>{batch_size, num_samples}, DType::Int64, input.device());
    Tensor cdf_buf(std::vector<int64_t>{batch_size, num_categories}, DType::Float32, input.device());

    uint64_t seed = ::tenzor::get_global_seed();

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
        queue.wait_and_throw();
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
    queue.wait_and_throw();
    return result;
}

// =========================================================================
// Histogram (with on-device min/max + bin-edge fill)
// =========================================================================

auto histogram_kernel(const Tensor& input, int64_t bins,
                      double min_val, double max_val,
                      sycl::queue& queue) -> std::pair<Tensor, Tensor> {
    if (bins <= 0) {
        throw std::invalid_argument("histogram: bins must be positive");
    }
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
        queue.wait_and_throw();

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
        queue.wait_and_throw();
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
        queue.wait_and_throw();
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

    // Native-precision path: Float64 input computes and returns edges/density
    // in Float64 instead of being unconditionally downcast to Float32 first.
    // Matches CPU's histogramdd_kernel (src/backends/cpu/kernels/reduction.cpp),
    // CUDA's (src/backends/cuda/kernels/advanced.cu), and ROCm's
    // (src/backends/rocm/kernels/sampling.hip.cpp), all of which are
    // templated on T = float/double and select T from the input dtype.
    const bool use_f64 = (input.dtype() == DType::Float64);
    const DType compute_dtype = use_f64 ? DType::Float64 : DType::Float32;

    auto in_contig = input.contiguous();
    if (in_contig.dtype() != compute_dtype) in_contig = in_contig.to(compute_dtype);
    const auto& device = input.device();

    // Auto-detect ranges from data if not provided
    bool auto_range = ranges.empty();
    if (auto_range) {
        ranges.resize(static_cast<size_t>(D));
    }

    auto launch = [&]<typename T>() -> std::pair<Tensor, std::vector<Tensor>> {
        const T* in_ptr = get_data_ptr<const T>(in_contig);

        if (auto_range && N > 0) {
            // Per-dimension min/max: allocate D min/max pairs on device
            T* d_mins = sycl::malloc_device<T>(static_cast<size_t>(D), queue);
            T* d_maxs = sycl::malloc_device<T>(static_cast<size_t>(D), queue);

            // Initialize from first sample
            queue.memcpy(d_mins, &in_ptr[0], static_cast<size_t>(D) * sizeof(T));
            queue.memcpy(d_maxs, &in_ptr[0], static_cast<size_t>(D) * sizeof(T));
            queue.wait_and_throw();

            // Parallel min/max over all samples using atomic_ref (no sycl::reduction needed)
            const int64_t local_D = D;
            queue.parallel_for<HistogramddMinMaxTag<T>>(sycl::range<1>(N * D), [=](sycl::id<1> idx_) {
                int64_t linear = static_cast<int64_t>(idx_);
                int64_t i = linear / local_D;
                int64_t dd = linear % local_D;
                T v = in_ptr[i * local_D + dd];

                sycl::atomic_ref<T, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>
                    atomic_min(d_mins[dd]);
                sycl::atomic_ref<T, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>
                    atomic_max(d_maxs[dd]);
                atomic_min.fetch_min(v);
                atomic_max.fetch_max(v);
            });
            queue.wait_and_throw();

            // Copy results back
            std::vector<T> h_mins(static_cast<size_t>(D)), h_maxs(static_cast<size_t>(D));
            queue.memcpy(h_mins.data(), d_mins, static_cast<size_t>(D) * sizeof(T)).wait();
            queue.memcpy(h_maxs.data(), d_maxs, static_cast<size_t>(D) * sizeof(T)).wait();
            sycl::free(d_mins, queue);
            sycl::free(d_maxs, queue);

            for (int64_t d = 0; d < D; ++d) {
                T mn = h_mins[static_cast<size_t>(d)];
                T mx = h_maxs[static_cast<size_t>(d)];
                if (mn == mx) {
                    mn -= T(0.5);
                    mx += T(0.5);
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
        std::vector<T> dim_min_vec(static_cast<size_t>(D));
        std::vector<T> dim_step_vec(static_cast<size_t>(D));

        for (int64_t d = 0; d < D; ++d) {
            auto sd = static_cast<size_t>(d);
            T fmin = static_cast<T>(ranges[sd].first);
            T fmax = static_cast<T>(ranges[sd].second);
            // A caller-supplied degenerate range (fmin >= fmax) yields step<=0, so
            // (v - fmin)/step at bin time is NaN/Inf (silently clamped afterward,
            // but the bin assignment would be unspecified). The auto-range path
            // already widens an equal-bounds interval by +-0.5 above; reuse that
            // same widening here — matches CPU's histogramdd_kernel
            // (src/backends/cpu/kernels/reduction.cpp) so step stays strictly
            // positive.
            if (fmin >= fmax) {
                fmin -= T(0.5);
                fmax += T(0.5);
            }
            T step = (fmax - fmin) / static_cast<T>(bins[sd]);
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
        T* d_dim_min = sycl::malloc_device<T>(static_cast<size_t>(D), queue);
        T* d_dim_step = sycl::malloc_device<T>(static_cast<size_t>(D), queue);
        int64_t* d_strides = sycl::malloc_device<int64_t>(static_cast<size_t>(D), queue);
        int64_t* d_bins = sycl::malloc_device<int64_t>(static_cast<size_t>(D), queue);

        queue.memcpy(d_dim_min, dim_min_vec.data(), static_cast<size_t>(D) * sizeof(T));
        queue.memcpy(d_dim_step, dim_step_vec.data(), static_cast<size_t>(D) * sizeof(T));
        queue.memcpy(d_strides, out_strides_vec.data(), static_cast<size_t>(D) * sizeof(int64_t));
        queue.memcpy(d_bins, bins.data(), static_cast<size_t>(D) * sizeof(int64_t));
        queue.wait_and_throw();

        // Allocate counts
        Tensor counts(out_shape, DType::Int64, device);
        int64_t* counts_ptr = get_data_ptr<int64_t>(counts);
        queue.memset(counts_ptr, 0, static_cast<size_t>(total_bins) * sizeof(int64_t)).wait();

        // Bin each sample
        if (N > 0) {
            const int64_t local_D = D;
            queue.parallel_for<HistogramddBinTag<T>>(sycl::range<1>(N), [=](sycl::id<1> idx_) {
                int64_t i = static_cast<int64_t>(idx_);
                int64_t flat = 0;
                bool in_range = true;

                for (int64_t dd = 0; dd < local_D; ++dd) {
                    T v = in_ptr[i * local_D + dd];
                    T fmin_d = d_dim_min[dd];
                    T step_d = d_dim_step[dd];
                    int64_t nb = d_bins[dd];

                    T range_max = fmin_d + step_d * static_cast<T>(nb);
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
            queue.wait_and_throw();
        }

        // Build edge tensors on-device
        std::vector<Tensor> edges_vec;
        edges_vec.reserve(static_cast<size_t>(D));
        for (int64_t d = 0; d < D; ++d) {
            auto sd = static_cast<size_t>(d);
            int64_t nb = bins[sd];
            int64_t num_edges = nb + 1;
            Tensor edge({num_edges}, compute_dtype, device);
            T* edge_ptr = get_data_ptr<T>(edge);
            const T local_min = dim_min_vec[sd];
            const T local_step = dim_step_vec[sd];
            queue.parallel_for<HistogramddEdgesTag<T>>(sycl::range<1>(num_edges),
                [=](sycl::id<1> idx_) {
                    int64_t j = static_cast<int64_t>(idx_);
                    edge_ptr[j] = local_min + static_cast<T>(j) * local_step;
                });
            queue.wait_and_throw();
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

            Tensor density_out(out_shape, compute_dtype, device);
            T* ddata = get_data_ptr<T>(density_out);
            // Direct count/norm division in double (rather than pre-computing
            // a T-precision reciprocal) so Float64 density values keep double
            // precision throughout, matching ROCm's histogramdd_density_kernel.
            queue.parallel_for<HistogramddDensityTag<T>>(sycl::range<1>(total_bins),
                [=](sycl::id<1> idx_) {
                    int64_t j = static_cast<int64_t>(idx_);
                    ddata[j] = static_cast<T>(static_cast<double>(counts_ptr[j]) / norm);
                });
            queue.wait_and_throw();
            result = density_out;
        }

        // Free device buffers
        sycl::free(d_dim_min, queue);
        sycl::free(d_dim_step, queue);
        sycl::free(d_strides, queue);
        sycl::free(d_bins, queue);

        return {result, std::move(edges_vec)};
    };

    if (use_f64) {
        return launch.template operator()<double>();
    }
    return launch.template operator()<float>();
}

// =========================================================================
// CDist (pairwise L2 distance)
// =========================================================================

auto cdist_kernel(const Tensor& x1, const Tensor& x2, double p,
                  sycl::queue& queue) -> Tensor {
    // Preserve the input dtype on output, keyed off x1 only to match the CPU
    // (src/backends/cpu/kernels/math.cpp) and CUDA references. Compute in
    // Float64 when x1 is Float64, otherwise compute in Float32 and narrow
    // Float16 / BFloat16 back at the end.
    const DType orig_dtype = x1.dtype();
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
                    // F130: accumulate squared diffs in double even for float
                    // input so cdist(p=2) matches the CPU reference. Clamp to
                    // >=0 before sqrt like CPU.
                    double sum = 0.0;
                    for (int64_t m = 0; m < M; ++m) {
                        double diff = static_cast<double>(a_b[p_idx * M + m]) - static_cast<double>(b_b[r * M + m]);
                        sum += diff * diff;
                    }
                    out_ptr[(bi * P + p_idx) * R + r] = static_cast<float>(sycl::sqrt(sum > 0.0 ? sum : 0.0));
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
        } else if (std::isinf(p)) {
            // p=inf -> Chebyshev (max-abs); matches CPU/CUDA references.
            queue.parallel_for<CDistInfTag>(sycl::range<3>(B, P, R),
                [=](sycl::id<3> idx) {
                    int64_t bi = idx[0];
                    int64_t p_idx = idx[1];
                    int64_t r  = idx[2];
                    const float* a_b = a_ptr + bi * P * M;
                    const float* b_b = b_ptr + bi * R * M;
                    float max_val = 0.0f;
                    for (int64_t m = 0; m < M; ++m) {
                        max_val = sycl::fmax(max_val,
                            sycl::fabs(a_b[p_idx * M + m] - b_b[r * M + m]));
                    }
                    out_ptr[(bi * P + p_idx) * R + r] = max_val;
                });
        } else if (p == 0.0) {
            // p=0 -> count of unequal components; matches CPU/CUDA references.
            queue.parallel_for<CDistP0Tag>(sycl::range<3>(B, P, R),
                [=](sycl::id<3> idx) {
                    int64_t bi = idx[0];
                    int64_t p_idx = idx[1];
                    int64_t r  = idx[2];
                    const float* a_b = a_ptr + bi * P * M;
                    const float* b_b = b_ptr + bi * R * M;
                    float count = 0.0f;
                    for (int64_t m = 0; m < M; ++m) {
                        if (a_b[p_idx * M + m] != b_b[r * M + m]) count += 1.0f;
                    }
                    out_ptr[(bi * P + p_idx) * R + r] = count;
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
        } else if (std::isinf(p)) {
            // p=inf -> Chebyshev (max-abs); matches CPU/CUDA references.
            queue.parallel_for<class CDistInfF64Tag>(sycl::range<3>(B, P, R),
                [=](sycl::id<3> idx) {
                    int64_t bi = idx[0];
                    int64_t p_idx = idx[1];
                    int64_t r  = idx[2];
                    const double* a_b = a_ptr + bi * P * M;
                    const double* b_b = b_ptr + bi * R * M;
                    double max_val = 0.0;
                    for (int64_t m = 0; m < M; ++m) {
                        max_val = sycl::fmax(max_val,
                            sycl::fabs(a_b[p_idx * M + m] - b_b[r * M + m]));
                    }
                    out_ptr[(bi * P + p_idx) * R + r] = max_val;
                });
        } else if (p == 0.0) {
            // p=0 -> count of unequal components; matches CPU/CUDA references.
            queue.parallel_for<class CDistP0F64Tag>(sycl::range<3>(B, P, R),
                [=](sycl::id<3> idx) {
                    int64_t bi = idx[0];
                    int64_t p_idx = idx[1];
                    int64_t r  = idx[2];
                    const double* a_b = a_ptr + bi * P * M;
                    const double* b_b = b_ptr + bi * R * M;
                    double count = 0.0;
                    for (int64_t m = 0; m < M; ++m) {
                        if (a_b[p_idx * M + m] != b_b[r * M + m]) count += 1.0;
                    }
                    out_ptr[(bi * P + p_idx) * R + r] = count;
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
    queue.wait_and_throw();
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
    uint64_t seed = ::tenzor::get_global_seed();

    queue.parallel_for<PoissonSampleKernelTag>(sycl::range<1>(n), [=](sycl::id<1> idx_) {
        int64_t i = static_cast<int64_t>(idx_);

        // Per-element LCG PRNG (same constants as bernoulli_kernel), threaded
        // through a local `state` via next_uniform() so both the Knuth and
        // Hormann PTRS branches below draw from the same evolving stream.
        uint64_t state = seed + static_cast<uint64_t>(i) * 6364136223846793005ULL + 1442695040888963407ULL;
        auto next_uniform = [&state]() -> float {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            float u = static_cast<float>(state >> 33) / static_cast<float>(1ULL << 31);
            return sycl::fmax(u, 1.0e-7f);  // clamp away from 0 so log() is safe
        };

        float lambda = in_ptr[i];

        if (!(lambda > 0.0f)) {
            out_ptr[i] = 0;
            return;
        }

        if (lambda < 12.0f) {
            // Knuth algorithm: generate Poisson(lambda) by counting uniform
            // samples until their product drops below exp(-lambda). Fine for
            // lambda < ~88 where exp(-lambda) doesn't underflow; capped here
            // at 12 to match the crossover into Hormann PTRS below.
            const double L = sycl::exp(-static_cast<double>(lambda));
            int64_t k = 0;
            double p = 1.0;
            const int64_t max_iter = 1 << 20;
            do {
                k++;
                p *= static_cast<double>(next_uniform());
            } while (p > L && k < max_iter);
            out_ptr[i] = k - 1;
            return;
        }

        // Transformed rejection (Hoermann PTRS) for moderate/large lambda,
        // where Knuth's expected iteration count grows linearly and
        // exp(-lambda) underflows to exactly 0 for lambda gtr ~104 (silently
        // truncating the Knuth loop after ~100-150 iterations regardless of
        // true lambda, which severely low-biases the samples). Ported
        // faithfully from CUDA's reference implementation
        // (src/backends/cuda/kernels/advanced.cu).
        const double dlam = static_cast<double>(lambda);
        const double b = 0.931 + 2.53 * sycl::sqrt(dlam);
        const double a = -0.059 + 0.02483 * b;
        const double inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
        const double v_r = 0.9277 - 3.6224 / (b - 2.0);
        const double loglam = sycl::log(dlam);

        int64_t k = 0;
        for (int iter = 0; iter < 1024; ++iter) {
            double U = static_cast<double>(next_uniform()) - 0.5;
            double V = static_cast<double>(next_uniform());
            double us = 0.5 - sycl::fabs(U);
            double kd = sycl::floor((2.0 * a / us + b) * U + dlam + 0.43);
            if (us >= 0.07 && V <= v_r) {
                k = static_cast<int64_t>(kd);
                break;
            }
            if (kd < 0.0 || (us < 0.013 && V > us)) {
                continue;
            }
            // lgamma(kd+1) = log(kd!)
            double logV = sycl::log(V) + sycl::log(inv_alpha) - sycl::log(a / (us * us) + b);
            double rhs = -dlam + kd * loglam - sycl::lgamma(kd + 1.0);
            if (logV <= rhs) {
                k = static_cast<int64_t>(kd);
                break;
            }
        }
        out_ptr[i] = k;
    });
    queue.wait_and_throw();
    return result;
}

// =========================================================================
// Normal sampling (Box-Muller transform per work-item)
// =========================================================================

namespace {
struct NormalSampleKernelTag {};
}  // namespace

auto normal_sample_kernel(const Tensor& mean, const Tensor& stddev, sycl::queue& queue) -> Tensor {
    // Preserve the input (mean) dtype on output; compute in Float32 internally.
    const DType orig_dtype = mean.dtype();
    auto m = mean.contiguous();
    auto s = stddev.contiguous();
    if (m.dtype() != DType::Float32) m = m.to(DType::Float32);
    if (s.dtype() != DType::Float32) s = s.to(DType::Float32);

    std::vector<int64_t> shape(m.shape().begin(), m.shape().end());
    Tensor result(shape, DType::Float32, m.device());
    int64_t n = m.numel();
    if (n == 0) return orig_dtype != DType::Float32 ? result.to(orig_dtype) : result;

    const float* mean_ptr = get_data_ptr<const float>(m);
    const float* std_ptr = get_data_ptr<const float>(s);
    float* out_ptr = get_data_ptr<float>(result);
    uint64_t seed = ::tenzor::get_global_seed();

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
    queue.wait_and_throw();
    return orig_dtype != DType::Float32 ? result.to(orig_dtype) : result;
}

// =========================================================================
// Exponential sampling (inverse CDF per work-item)
// =========================================================================

namespace {
struct ExponentialSampleKernelTag {};
}  // namespace

namespace {
struct ExponentialValidateRateTag {};
}  // namespace

auto exponential_sample_kernel(const Tensor& rate, sycl::queue& queue) -> Tensor {
    auto input = rate.contiguous();
    if (input.dtype() != DType::Float32) input = input.to(DType::Float32);

    int64_t n = input.numel();
    const float* rate_ptr = get_data_ptr<const float>(input);

    if (n > 0) {
        // The Exponential distribution is only defined for rate > 0 (rate==0
        // gives an undefined +Inf-mean distribution; rate<0 has no valid
        // support at all). Every work-item independently flags its own
        // element into a single atomic int (no reduction-tree ordering to
        // worry about, so NaN is always caught). One scratch int + one
        // readback is an O(1) host<->device sync regardless of tensor size,
        // not an elementwise CPU round-trip.
        int* d_flag = sycl::malloc_device<int>(1, queue);
        queue.memset(d_flag, 0, sizeof(int)).wait();
        queue.parallel_for<ExponentialValidateRateTag>(sycl::range<1>(n), [=](sycl::id<1> idx_) {
            int64_t i = static_cast<int64_t>(idx_);
            if (!(rate_ptr[i] > 0.0f)) {
                sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>
                    flag_ref(*d_flag);
                flag_ref.fetch_or(1);
            }
        });
        queue.wait_and_throw();
        int h_flag = 0;
        queue.memcpy(&h_flag, d_flag, sizeof(int)).wait();
        sycl::free(d_flag, queue);
        if (h_flag != 0) {
            throw std::invalid_argument(
                "exponential: rate must be > 0 (got a non-positive or NaN rate)");
        }
    }

    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, DType::Float32, input.device());
    if (n == 0) return result;

    float* out_ptr = get_data_ptr<float>(result);
    uint64_t seed = ::tenzor::get_global_seed();

    queue.parallel_for<ExponentialSampleKernelTag>(sycl::range<1>(n), [=](sycl::id<1> idx_) {
        int64_t i = static_cast<int64_t>(idx_);

        uint64_t state = seed + static_cast<uint64_t>(i) * 6364136223846793005ULL + 1442695040888963407ULL;
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        float u = static_cast<float>(state >> 33) / static_cast<float>(1ULL << 31);
        u = sycl::fmin(u, 1.0f - 1.0e-7f);  // Clamp away from 1.0

        // Inverse CDF: -log(1 - u) / rate
        out_ptr[i] = -sycl::log(1.0f - u) / rate_ptr[i];
    });
    queue.wait_and_throw();
    return result;
}


// Gamma distribution sampling (Marsaglia-Tsang 2000) — native SYCL device
// kernel, no host fallback. gamma(concentration=alpha, rate=beta); for
// alpha < 1 the alpha+1 draw is boosted by u^(1/alpha). The per-element LCG
// state threads through inline uniform/normal helpers.
auto gamma_sample_kernel(const Tensor& concentration, const Tensor& rate,
                         sycl::queue& queue) -> Tensor {
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
    {
        auto as = a.shape();
        auto bs = b.shape();
        bool same = as.size() == bs.size();
        for (size_t d = 0; same && d < as.size(); ++d) same = (as[d] == bs[d]);
        if (!same) {
            throw std::invalid_argument(
                "gamma_sample: concentration and rate must have the same shape");
        }
    }

    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, DType::Float32, a.device());
    int64_t n = a.numel();
    if (n == 0) return orig_dtype != DType::Float32 ? result.to(orig_dtype) : result;

    const float* a_ptr = get_data_ptr<const float>(a);
    const float* b_ptr = get_data_ptr<const float>(b);
    float* out_ptr = get_data_ptr<float>(result);
    uint64_t seed = ::tenzor::get_global_seed();

    queue.parallel_for<GammaSampleKernelTag>(sycl::range<1>(n), [=](sycl::id<1> idx_) {
        int64_t i = static_cast<int64_t>(idx_);
        uint64_t state = seed + static_cast<uint64_t>(i) * 6364136223846793005ULL + 1442695040888963407ULL;

        auto next_uniform = [&state]() -> float {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            return static_cast<float>(state >> 33) / static_cast<float>(1ULL << 31);
        };
        auto next_normal = [&]() -> float {
            float u1 = sycl::fmax(next_uniform(), 1.0e-7f);
            float u2 = next_uniform();
            return sycl::sqrt(-2.0f * sycl::log(u1)) *
                   sycl::cos(2.0f * 3.14159265358979323846f * u2);
        };

        float alpha = a_ptr[i];
        float beta  = b_ptr[i];
        if (!(alpha > 0.0f)) alpha = 1.1754944e-38f;
        if (!(beta  > 0.0f)) beta  = 1.1754944e-38f;

        float boost = 1.0f;
        if (alpha < 1.0f) {
            float u0 = sycl::fmax(next_uniform(), 1.0e-7f);
            boost = sycl::pow(u0, 1.0f / alpha);
            alpha += 1.0f;
        }

        const float d = alpha - 1.0f / 3.0f;
        const float c = 1.0f / sycl::sqrt(9.0f * d);
        float res;
        for (;;) {
            float x = next_normal();
            float v = 1.0f + c * x;
            if (v <= 0.0f) continue;
            v = v * v * v;
            float u = sycl::fmax(next_uniform(), 1.0e-7f);
            float x2 = x * x;
            if (u < 1.0f - 0.0331f * x2 * x2 ||
                sycl::log(u) < 0.5f * x2 + d * (1.0f - v + sycl::log(v))) {
                res = d * v;
                break;
            }
        }
        out_ptr[i] = boost * res / beta;
    });
    queue.wait_and_throw();
    return orig_dtype != DType::Float32 ? result.to(orig_dtype) : result;
}

// =========================================================================
// Trapezoid integration
// =========================================================================

auto trapezoid_kernel(const Tensor& y, int64_t dim, double dx,
                       const Tensor* x_ptr, sycl::queue& queue) -> Tensor {
    // Compute in Float64 for Float64 input (preserving precision and the
    // output dtype), otherwise compute in Float32 and narrow reduced-precision
    // floats back. Mirrors the CPU/CUDA references.
    const DType orig_dtype = y.dtype();
    const DType compute_dtype =
        (orig_dtype == DType::Float64) ? DType::Float64 : DType::Float32;

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
    // CPU/CUDA/ROCm/Vulkan's convention -- do not force a size-1 dim.

    Tensor result(out_shape, compute_dtype, y.device());
    int64_t total = outer * inner;
    if (n < 2 || total == 0)
        return (compute_dtype != orig_dtype) ? result.to(orig_dtype) : result;

    bool has_x = x_ptr != nullptr;
    Tensor xf;
    if (has_x) {
        xf = (x_ptr->dtype() == compute_dtype) ? x_ptr->contiguous()
                                               : x_ptr->contiguous().to(compute_dtype);
    }

    int64_t n_val = n, inner_val = inner;

    auto launch = [&]<typename T>(T*) {
        const T* y_ptr = get_data_ptr<const T>(yf);
        T* out_ptr = get_data_ptr<T>(result);
        const T* xd = has_x ? get_data_ptr<const T>(xf) : nullptr;
        T dx_f = static_cast<T>(dx);
        queue.parallel_for<TrapezoidKernelTag<T>>(sycl::range<1>(total), [=](sycl::id<1> id) {
            int64_t idx = id[0];
            int64_t o = idx / inner_val;
            int64_t i_inner = idx % inner_val;

            T sum = T(0);
            for (int64_t k = 0; k < n_val - 1; k++) {
                int64_t idx_k  = (o * n_val + k) * inner_val + i_inner;
                int64_t idx_k1 = (o * n_val + k + 1) * inner_val + i_inner;
                T h = has_x ? (xd[idx_k1] - xd[idx_k]) : dx_f;
                sum += T(0.5) * (y_ptr[idx_k] + y_ptr[idx_k1]) * h;
            }
            out_ptr[idx] = sum;
        });
    };
    if (compute_dtype == DType::Float64) launch(static_cast<double*>(nullptr));
    else                                 launch(static_cast<float*>(nullptr));
    queue.wait_and_throw();
    // Narrow the compute-dtype result back to the caller's dtype
    // (Float16 / BFloat16 / Float64) when it differs.
    return (compute_dtype != orig_dtype) ? result.to(orig_dtype) : result;
}

// =========================================================================
// Cumulative trapezoid integration
// =========================================================================

auto cumulative_trapezoid_kernel(const Tensor& y, int64_t dim, double dx,
                                  const Tensor* x_ptr, sycl::queue& queue) -> Tensor {
    // Compute in Float64 for Float64 input, otherwise Float32; mirrors CPU/CUDA.
    const DType orig_dtype = y.dtype();
    const DType compute_dtype =
        (orig_dtype == DType::Float64) ? DType::Float64 : DType::Float32;

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
    if (n < 2 || total == 0)
        return (compute_dtype != orig_dtype) ? result.to(orig_dtype) : result;

    bool has_x = x_ptr != nullptr;
    Tensor xf;
    if (has_x) {
        xf = (x_ptr->dtype() == compute_dtype) ? x_ptr->contiguous()
                                               : x_ptr->contiguous().to(compute_dtype);
    }

    int64_t n_val = n, inner_val = inner;

    auto launch = [&]<typename T>(T*) {
        const T* y_ptr = get_data_ptr<const T>(yf);
        T* out_ptr = get_data_ptr<T>(result);
        const T* xd = has_x ? get_data_ptr<const T>(xf) : nullptr;
        T dx_f = static_cast<T>(dx);
        queue.parallel_for<CumulativeTrapezoidKernelTag<T>>(sycl::range<1>(total), [=](sycl::id<1> id) {
            int64_t idx = id[0];
            int64_t o = idx / inner_val;
            int64_t i_inner = idx % inner_val;

            T cumsum = T(0);
            for (int64_t k = 0; k < n_val - 1; k++) {
                int64_t idx_k  = (o * n_val + k) * inner_val + i_inner;
                int64_t idx_k1 = (o * n_val + k + 1) * inner_val + i_inner;
                T h = has_x ? (xd[idx_k1] - xd[idx_k]) : dx_f;
                cumsum += T(0.5) * (y_ptr[idx_k] + y_ptr[idx_k1]) * h;
                out_ptr[(o * (n_val - 1) + k) * inner_val + i_inner] = cumsum;
            }
        });
    };
    if (compute_dtype == DType::Float64) launch(static_cast<double*>(nullptr));
    else                                 launch(static_cast<float*>(nullptr));
    queue.wait_and_throw();
    return (compute_dtype != orig_dtype) ? result.to(orig_dtype) : result;
}

// =========================================================================
// Numerical gradient (central differences)
// =========================================================================

auto gradient_kernel(const Tensor& input, int64_t dim, double spacing,
                      sycl::queue& queue) -> Tensor {
    // Compute in Float64 for Float64 input, otherwise Float32; mirrors CPU/CUDA.
    const DType orig_dtype = input.dtype();
    const DType compute_dtype =
        (orig_dtype == DType::Float64) ? DType::Float64 : DType::Float32;

    Tensor inf = (input.dtype() == compute_dtype) ? input.contiguous()
                                                  : input.contiguous().to(compute_dtype);
    auto shape = inf.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    int64_t n = shape[dim];

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];

    Tensor result(std::vector<int64_t>(shape.begin(), shape.end()), compute_dtype, input.device());
    int64_t total = outer * inner;
    if (n < 2 || total == 0)
        return (compute_dtype != orig_dtype) ? result.to(orig_dtype) : result;

    int64_t n_val = n, inner_val = inner;

    auto launch = [&]<typename T>(T*) {
        const T* in_ptr = get_data_ptr<const T>(inf);
        T* out_ptr = get_data_ptr<T>(result);
        T h = static_cast<T>(spacing);
        queue.parallel_for<GradientKernelTag<T>>(sycl::range<1>(total), [=](sycl::id<1> id) {
            int64_t idx = id[0];
            int64_t o = idx / inner_val;
            int64_t i_inner = idx % inner_val;

            auto at = [&](int64_t k) -> T {
                return in_ptr[(o * n_val + k) * inner_val + i_inner];
            };

            out_ptr[(o * n_val + 0) * inner_val + i_inner] = (at(1) - at(0)) / h;
            for (int64_t k = 1; k < n_val - 1; k++) {
                out_ptr[(o * n_val + k) * inner_val + i_inner] = (at(k + 1) - at(k - 1)) / (T(2) * h);
            }
            out_ptr[(o * n_val + n_val - 1) * inner_val + i_inner] = (at(n_val - 1) - at(n_val - 2)) / h;
        });
    };
    if (compute_dtype == DType::Float64) launch(static_cast<double*>(nullptr));
    else                                 launch(static_cast<float*>(nullptr));
    queue.wait_and_throw();
    return (compute_dtype != orig_dtype) ? result.to(orig_dtype) : result;
}

// =========================================================================
// Pairwise distance
// =========================================================================

auto pairwise_distance_kernel(const Tensor& x1, const Tensor& x2, double p,
                               sycl::queue& queue) -> Tensor {
    // Compute in Float64 for Float64 input, otherwise Float32; output dtype
    // keyed off x1 to match the CPU/CUDA references.
    const DType orig_dtype = x1.dtype();
    const DType compute_dtype =
        (orig_dtype == DType::Float64) ? DType::Float64 : DType::Float32;

    Tensor a = (x1.dtype() == compute_dtype) ? x1.contiguous() : x1.contiguous().to(compute_dtype);
    Tensor b = (x2.dtype() == compute_dtype) ? x2.contiguous() : x2.contiguous().to(compute_dtype);

    int64_t N = a.shape()[0], D = a.shape()[1];
    Tensor result({N}, compute_dtype, x1.device());
    if (N == 0)
        return (compute_dtype != orig_dtype) ? result.to(orig_dtype) : result;

    auto launch = [&]<typename T>(T*) {
        const T* a_ptr = get_data_ptr<const T>(a);
        const T* b_ptr = get_data_ptr<const T>(b);
        T* out_ptr = get_data_ptr<T>(result);
        T p_f = static_cast<T>(p);
        queue.parallel_for<PairwiseDistKernelTag<T>>(sycl::range<1>(N), [=](sycl::id<1> id) {
            int64_t i = id[0];
            T sum = T(0);
            if (p_f == T(2)) {
                for (int64_t j = 0; j < D; j++) {
                    T diff = a_ptr[i * D + j] - b_ptr[i * D + j];
                    sum += diff * diff;
                }
                out_ptr[i] = sycl::sqrt(sum);
            } else if (p_f == T(1)) {
                for (int64_t j = 0; j < D; j++) {
                    sum += sycl::fabs(a_ptr[i * D + j] - b_ptr[i * D + j]);
                }
                out_ptr[i] = sum;
            } else {
                for (int64_t j = 0; j < D; j++) {
                    sum += sycl::pow(sycl::fabs(a_ptr[i * D + j] - b_ptr[i * D + j]), p_f);
                }
                out_ptr[i] = sycl::pow(sum, T(1) / p_f);
            }
        });
    };
    if (compute_dtype == DType::Float64) launch(static_cast<double*>(nullptr));
    else                                 launch(static_cast<float*>(nullptr));
    queue.wait_and_throw();
    return (compute_dtype != orig_dtype) ? result.to(orig_dtype) : result;
}

// =========================================================================
// Pdist (all-pairs pairwise distances)
// =========================================================================

auto pdist_kernel(const Tensor& input, double p, sycl::queue& queue) -> Tensor {
    // Compute in Float64 for Float64 input, otherwise Float32; mirrors CPU/CUDA.
    const DType orig_dtype = input.dtype();
    const DType compute_dtype =
        (orig_dtype == DType::Float64) ? DType::Float64 : DType::Float32;

    Tensor inf = (input.dtype() == compute_dtype) ? input.contiguous()
                                                  : input.contiguous().to(compute_dtype);
    int64_t N = inf.shape()[0], D = inf.shape()[1];
    int64_t num_pairs = N * (N - 1) / 2;

    Tensor result({num_pairs}, compute_dtype, input.device());
    if (num_pairs == 0)
        return (compute_dtype != orig_dtype) ? result.to(orig_dtype) : result;

    auto launch = [&]<typename T>(T*) {
        const T* data = get_data_ptr<const T>(inf);
        T* out_ptr = get_data_ptr<T>(result);
        T p_f = static_cast<T>(p);
        queue.parallel_for<PdistKernelTag<T>>(sycl::range<1>(num_pairs), [=](sycl::id<1> id) {
            int64_t idx = id[0];
            // Map flat index to (i, j) pair where i < j
            int64_t i = 0, offset = 0;
            while (offset + (N - 1 - i) <= idx) {
                offset += (N - 1 - i);
                i++;
            }
            int64_t j = idx - offset + i + 1;

            T sum = T(0);
            if (p_f == T(2)) {
                for (int64_t d = 0; d < D; d++) {
                    T diff = data[i * D + d] - data[j * D + d];
                    sum += diff * diff;
                }
                out_ptr[idx] = sycl::sqrt(sum);
            } else if (p_f == T(1)) {
                for (int64_t d = 0; d < D; d++) {
                    sum += sycl::fabs(data[i * D + d] - data[j * D + d]);
                }
                out_ptr[idx] = sum;
            } else {
                for (int64_t d = 0; d < D; d++) {
                    sum += sycl::pow(sycl::fabs(data[i * D + d] - data[j * D + d]), p_f);
                }
                out_ptr[idx] = sycl::pow(sum, T(1) / p_f);
            }
        });
    };
    if (compute_dtype == DType::Float64) launch(static_cast<double*>(nullptr));
    else                                 launch(static_cast<float*>(nullptr));
    queue.wait_and_throw();
    return (compute_dtype != orig_dtype) ? result.to(orig_dtype) : result;
}

}  // namespace oneapi
}  // namespace tenzor
