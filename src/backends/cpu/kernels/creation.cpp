#include "tenzor/core/tensor.hpp"
#include "tenzor/core/shape.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/utils/error.hpp"
#include <random>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

// OpenMP parallelization threshold (elements)
static constexpr size_t OMP_THRESHOLD = 65536;

// SIMD intrinsics
#if defined(__AVX512F__)
#include <immintrin.h>
#define TENZOR_HAS_AVX512 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define TENZOR_HAS_AVX2 1
#elif defined(__SSE2__)
#include <emmintrin.h>
#define TENZOR_HAS_SSE2 1
#endif

namespace tenzor {
namespace cpu {

// ============================================================================
// Zeros Kernel - Create tensor filled with zeros
// ============================================================================

auto zeros_kernel(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor {
    Tensor result(shape, dtype, device);
    size_t n = static_cast<size_t>(result.numel());

    // IEEE 754 float/double zero and integer zero are all-bits-zero,
    // so memset is correct and fastest for all supported dtypes.
    if (dtype == DType::Float32) {
        std::memset(result.data<float>(), 0, n * sizeof(float));
    } else if (dtype == DType::Float64) {
        std::memset(result.data<double>(), 0, n * sizeof(double));
    } else if (dtype == DType::Int32) {
        std::memset(result.data<int32_t>(), 0, n * sizeof(int32_t));
    } else if (dtype == DType::Int64) {
        std::memset(result.data<int64_t>(), 0, n * sizeof(int64_t));
    } else {
        throw std::runtime_error("zeros operation: unsupported dtype");
    }

    return result;
}

// ============================================================================
// Ones Kernel - Create tensor filled with ones
// ============================================================================

auto ones_kernel(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor {
    Tensor result(shape, dtype, device);
    int64_t n = result.numel();

    if (dtype == DType::Float32) {
        float* data = result.data<float>();
        #pragma omp parallel for schedule(static) if(n > static_cast<int64_t>(OMP_THRESHOLD))
        for (int64_t i = 0; i < n; ++i) {
            data[i] = 1.0f;
        }
    } else if (dtype == DType::Float64) {
        double* data = result.data<double>();
        #pragma omp parallel for schedule(static) if(n > static_cast<int64_t>(OMP_THRESHOLD))
        for (int64_t i = 0; i < n; ++i) {
            data[i] = 1.0;
        }
    } else if (dtype == DType::Int32) {
        int32_t* data = result.data<int32_t>();
        #pragma omp parallel for schedule(static) if(n > static_cast<int64_t>(OMP_THRESHOLD))
        for (int64_t i = 0; i < n; ++i) {
            data[i] = 1;
        }
    } else if (dtype == DType::Int64) {
        int64_t* data = result.data<int64_t>();
        #pragma omp parallel for schedule(static) if(n > static_cast<int64_t>(OMP_THRESHOLD))
        for (int64_t i = 0; i < n; ++i) {
            data[i] = 1;
        }
    } else {
        throw std::runtime_error("ones operation: unsupported dtype");
    }

    return result;
}

// ============================================================================
// Random Number Generator helpers
// ============================================================================

namespace detail {
    // Get a base seed from the global seed (for reproducibility via manual_seed())
    static unsigned int get_base_seed() {
        return static_cast<unsigned int>(tenzor::get_global_seed());
    }
}

// ============================================================================
// Rand Kernel - Create tensor with uniform random values in [0, 1)
// ============================================================================

auto rand_kernel(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor {
    Tensor result(shape, dtype, device);
    int64_t n = result.numel();

    if (dtype == DType::Float32) {
        float* data = result.data<float>();

        if (n > static_cast<int64_t>(OMP_THRESHOLD)) {
            unsigned int base_seed = detail::get_base_seed();
            #pragma omp parallel
            {
#ifdef _OPENMP
                int tid = omp_get_thread_num();
#else
                int tid = 0;
#endif
                std::mt19937 local_rng(base_seed + static_cast<unsigned int>(tid));
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                #pragma omp for schedule(static)
                for (int64_t i = 0; i < n; ++i) {
                    data[i] = dist(local_rng);
                }
            }
        } else {
            std::mt19937 rng(detail::get_base_seed());
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            for (int64_t i = 0; i < n; ++i) {
                data[i] = dist(rng);
            }
        }

    } else if (dtype == DType::Float64) {
        double* data = result.data<double>();

        if (n > static_cast<int64_t>(OMP_THRESHOLD)) {
            unsigned int base_seed = detail::get_base_seed();
            #pragma omp parallel
            {
#ifdef _OPENMP
                int tid = omp_get_thread_num();
#else
                int tid = 0;
#endif
                std::mt19937 local_rng(base_seed + static_cast<unsigned int>(tid));
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                #pragma omp for schedule(static)
                for (int64_t i = 0; i < n; ++i) {
                    data[i] = dist(local_rng);
                }
            }
        } else {
            std::mt19937 rng(detail::get_base_seed());
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            for (int64_t i = 0; i < n; ++i) {
                data[i] = dist(rng);
            }
        }

    } else {
        throw std::runtime_error("rand operation only supports Float32 and Float64 dtypes");
    }

    return result;
}

// ============================================================================
// Randn Kernel - Create tensor with standard normal distribution N(0, 1)
// ============================================================================

auto randn_kernel(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor {
    Tensor result(shape, dtype, device);
    int64_t n = result.numel();

    if (dtype == DType::Float32) {
        float* data = result.data<float>();

        if (n > static_cast<int64_t>(OMP_THRESHOLD)) {
            unsigned int base_seed = detail::get_base_seed();
            #pragma omp parallel
            {
#ifdef _OPENMP
                int tid = omp_get_thread_num();
#else
                int tid = 0;
#endif
                std::mt19937 local_rng(base_seed + static_cast<unsigned int>(tid));
                std::normal_distribution<float> dist(0.0f, 1.0f);
                #pragma omp for schedule(static)
                for (int64_t i = 0; i < n; ++i) {
                    data[i] = dist(local_rng);
                }
            }
        } else {
            std::mt19937 rng(detail::get_base_seed());
            std::normal_distribution<float> dist(0.0f, 1.0f);
            for (int64_t i = 0; i < n; ++i) {
                data[i] = dist(rng);
            }
        }

    } else if (dtype == DType::Float64) {
        double* data = result.data<double>();

        if (n > static_cast<int64_t>(OMP_THRESHOLD)) {
            unsigned int base_seed = detail::get_base_seed();
            #pragma omp parallel
            {
#ifdef _OPENMP
                int tid = omp_get_thread_num();
#else
                int tid = 0;
#endif
                std::mt19937 local_rng(base_seed + static_cast<unsigned int>(tid));
                std::normal_distribution<double> dist(0.0, 1.0);
                #pragma omp for schedule(static)
                for (int64_t i = 0; i < n; ++i) {
                    data[i] = dist(local_rng);
                }
            }
        } else {
            std::mt19937 rng(detail::get_base_seed());
            std::normal_distribution<double> dist(0.0, 1.0);
            for (int64_t i = 0; i < n; ++i) {
                data[i] = dist(rng);
            }
        }

    } else {
        throw std::runtime_error("randn operation only supports Float32 and Float64 dtypes");
    }

    return result;
}

// ============================================================================
// Randint Kernel - Create tensor with random integers in [low, high)
// ============================================================================

auto randint_kernel(int64_t low, int64_t high, const std::vector<int64_t>& shape,
                    DType dtype, const Device& device) -> Tensor {
    if (dtype != DType::Int32 && dtype != DType::Int64) {
        throw std::runtime_error("randint operation only supports Int32 and Int64 dtypes");
    }

    Tensor result(shape, dtype, device);
    int64_t n = result.numel();

    if (dtype == DType::Int32) {
        int32_t* data = result.data<int32_t>();
        std::uniform_int_distribution<int32_t> dist(
            static_cast<int32_t>(low), static_cast<int32_t>(high - 1));

        if (n > static_cast<int64_t>(OMP_THRESHOLD)) {
            unsigned int base_seed = detail::get_base_seed();
            #pragma omp parallel
            {
#ifdef _OPENMP
                int tid = omp_get_thread_num();
#else
                int tid = 0;
#endif
                std::mt19937 local_rng(base_seed + static_cast<unsigned int>(tid));
                std::uniform_int_distribution<int32_t> local_dist(
                    static_cast<int32_t>(low), static_cast<int32_t>(high - 1));
                #pragma omp for schedule(static)
                for (int64_t i = 0; i < n; ++i) {
                    data[i] = local_dist(local_rng);
                }
            }
        } else {
            std::mt19937 rng(detail::get_base_seed());
            for (int64_t i = 0; i < n; ++i) {
                data[i] = dist(rng);
            }
        }
    } else {  // Int64
        int64_t* data = result.data<int64_t>();
        std::uniform_int_distribution<int64_t> dist(low, high - 1);

        if (n > static_cast<int64_t>(OMP_THRESHOLD)) {
            unsigned int base_seed = detail::get_base_seed();
            #pragma omp parallel
            {
#ifdef _OPENMP
                int tid = omp_get_thread_num();
#else
                int tid = 0;
#endif
                std::mt19937 local_rng(base_seed + static_cast<unsigned int>(tid));
                std::uniform_int_distribution<int64_t> local_dist(low, high - 1);
                #pragma omp for schedule(static)
                for (int64_t i = 0; i < n; ++i) {
                    data[i] = local_dist(local_rng);
                }
            }
        } else {
            std::mt19937 rng(detail::get_base_seed());
            for (int64_t i = 0; i < n; ++i) {
                data[i] = dist(rng);
            }
        }
    }

    return result;
}

// ============================================================================
// Full Kernel - Create tensor filled with a specific value
// ============================================================================

auto full_kernel(const std::vector<int64_t>& shape, float value, DType dtype, const Device& device) -> Tensor {
    Tensor result(shape, dtype, device);
    int64_t n = result.numel();

    if (dtype == DType::Float32) {
        float* data = result.data<float>();
        #pragma omp parallel for schedule(static) if(n > static_cast<int64_t>(OMP_THRESHOLD))
        for (int64_t i = 0; i < n; ++i) {
            data[i] = value;
        }
    } else if (dtype == DType::Float64) {
        double* data = result.data<double>();
        double dval = static_cast<double>(value);
        #pragma omp parallel for schedule(static) if(n > static_cast<int64_t>(OMP_THRESHOLD))
        for (int64_t i = 0; i < n; ++i) {
            data[i] = dval;
        }
    } else if (dtype == DType::Int32) {
        int32_t* data = result.data<int32_t>();
        int32_t ival = static_cast<int32_t>(value);
        #pragma omp parallel for schedule(static) if(n > static_cast<int64_t>(OMP_THRESHOLD))
        for (int64_t i = 0; i < n; ++i) {
            data[i] = ival;
        }
    } else if (dtype == DType::Int64) {
        int64_t* data = result.data<int64_t>();
        int64_t ival = static_cast<int64_t>(value);
        #pragma omp parallel for schedule(static) if(n > static_cast<int64_t>(OMP_THRESHOLD))
        for (int64_t i = 0; i < n; ++i) {
            data[i] = ival;
        }
    } else {
        throw std::runtime_error("full operation: unsupported dtype");
    }

    return result;
}

// ============================================================================
// Arange Kernel - Create tensor with evenly spaced values
// ============================================================================

auto arange_kernel(float start, float end, float step, DType dtype, const Device& device) -> Tensor {
    if (step == 0.0f) {
        throw std::runtime_error("arange: step must be non-zero");
    }
    if ((step > 0 && start >= end) || (step < 0 && start <= end)) {
        // Return empty tensor
        return Tensor({0}, dtype, device);
    }

    int64_t numel = static_cast<int64_t>(std::ceil((end - start) / step));
    Tensor result({numel}, dtype, device);

    if (dtype == DType::Float32) {
        float* data = result.data<float>();
        for (int64_t i = 0; i < numel; ++i) {
            data[i] = start + static_cast<float>(i) * step;
        }
    } else if (dtype == DType::Float64) {
        double* data = result.data<double>();
        for (int64_t i = 0; i < numel; ++i) {
            data[i] = static_cast<double>(start) + static_cast<double>(i) * static_cast<double>(step);
        }
    } else if (dtype == DType::Int32) {
        int32_t* data = result.data<int32_t>();
        for (int64_t i = 0; i < numel; ++i) {
            data[i] = static_cast<int32_t>(start + static_cast<float>(i) * step);
        }
    } else if (dtype == DType::Int64) {
        int64_t* data = result.data<int64_t>();
        for (int64_t i = 0; i < numel; ++i) {
            data[i] = static_cast<int64_t>(start + static_cast<float>(i) * step);
        }
    } else {
        throw std::runtime_error("arange operation: unsupported dtype");
    }

    return result;
}

// ============================================================================
// Linspace Kernel - Create tensor with linearly spaced values
// ============================================================================

auto linspace_kernel(float start, float end, int64_t steps, DType dtype, const Device& device) -> Tensor {
    if (steps < 0) {
        throw std::runtime_error("linspace: number of steps must be non-negative");
    }
    if (steps == 0) {
        return Tensor({0}, dtype, device);
    }

    Tensor result({steps}, dtype, device);

    if (dtype == DType::Float32) {
        float* data = result.data<float>();
        if (steps == 1) {
            data[0] = start;
        } else {
            float step = (end - start) / static_cast<float>(steps - 1);
            for (int64_t i = 0; i < steps; ++i) {
                data[i] = start + static_cast<float>(i) * step;
            }
            data[steps - 1] = end;  // Exact endpoint
        }
    } else if (dtype == DType::Float64) {
        double* data = result.data<double>();
        if (steps == 1) {
            data[0] = static_cast<double>(start);
        } else {
            double step = (static_cast<double>(end) - static_cast<double>(start)) / static_cast<double>(steps - 1);
            for (int64_t i = 0; i < steps; ++i) {
                data[i] = static_cast<double>(start) + static_cast<double>(i) * step;
            }
            data[steps - 1] = static_cast<double>(end);  // Exact endpoint
        }
    } else {
        throw std::runtime_error("linspace operation: only Float32 and Float64 supported");
    }

    return result;
}

// ============================================================================
// Eye Kernel - Create identity matrix
// ============================================================================

auto eye_kernel(int64_t n, int64_t m, DType dtype, const Device& device) -> Tensor {
    if (m < 0) m = n;  // Square matrix by default

    Tensor result({n, m}, dtype, device);

    // Zero-initialize first
    size_t total = static_cast<size_t>(n * m);

    if (dtype == DType::Float32) {
        float* data = result.data<float>();
        std::fill_n(data, total, 0.0f);
        int64_t diag_len = std::min(n, m);
        for (int64_t i = 0; i < diag_len; ++i) {
            data[i * m + i] = 1.0f;
        }
    } else if (dtype == DType::Float64) {
        double* data = result.data<double>();
        std::fill_n(data, total, 0.0);
        int64_t diag_len = std::min(n, m);
        for (int64_t i = 0; i < diag_len; ++i) {
            data[i * m + i] = 1.0;
        }
    } else if (dtype == DType::Int32) {
        int32_t* data = result.data<int32_t>();
        std::fill_n(data, total, 0);
        int64_t diag_len = std::min(n, m);
        for (int64_t i = 0; i < diag_len; ++i) {
            data[i * m + i] = 1;
        }
    } else if (dtype == DType::Int64) {
        int64_t* data = result.data<int64_t>();
        std::fill_n(data, total, static_cast<int64_t>(0));
        int64_t diag_len = std::min(n, m);
        for (int64_t i = 0; i < diag_len; ++i) {
            data[i * m + i] = 1;
        }
    } else {
        throw std::runtime_error("eye operation: unsupported dtype");
    }

    return result;
}

// ============================================================================
// Multinomial Kernel - Weighted random sampling
// ============================================================================

auto multinomial_kernel(const Tensor& probs, int64_t num_samples, bool replacement) -> Tensor {
    // probs: (N, C) or (C,) - probability weights (not necessarily normalized)
    // Returns: (N, num_samples) or (num_samples,) of Int64 indices
    auto shape = probs.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());

    bool batched = (ndim == 2);
    int64_t batch_size = batched ? shape[0] : 1;
    int64_t num_categories = batched ? shape[1] : shape[0];

    if (num_samples <= 0) {
        throw std::runtime_error("multinomial: num_samples must be > 0");
    }
    if (!replacement && num_samples > num_categories) {
        throw std::runtime_error("multinomial: cannot sample more than num_categories without replacement");
    }

    Tensor probs_f32 = (probs.dtype() != DType::Float32) ? probs.to(DType::Float32) : probs;
    const float* p_data = probs_f32.data<float>();

    std::vector<int64_t> out_shape;
    if (batched) out_shape.push_back(batch_size);
    out_shape.push_back(num_samples);

    Tensor result(out_shape, DType::Int64, probs.device());
    int64_t* out_data = result.data<int64_t>();

    std::mt19937 rng(detail::get_base_seed());
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    for (int64_t b = 0; b < batch_size; ++b) {
        const float* row = p_data + b * num_categories;

        // Compute cumulative sum (unnormalized CDF)
        std::vector<float> cumsum(static_cast<size_t>(num_categories));
        cumsum[0] = row[0];
        for (int64_t i = 1; i < num_categories; ++i) {
            cumsum[static_cast<size_t>(i)] = cumsum[static_cast<size_t>(i - 1)] + row[i];
        }
        float total = cumsum[static_cast<size_t>(num_categories - 1)];
        if (total <= 0.0f) {
            throw std::runtime_error("multinomial: sum of probabilities must be > 0");
        }

        // Normalize
        for (int64_t i = 0; i < num_categories; ++i) {
            cumsum[static_cast<size_t>(i)] /= total;
        }

        int64_t* out_row = out_data + b * num_samples;

        if (replacement) {
            for (int64_t s = 0; s < num_samples; ++s) {
                float u = uniform(rng);
                // Binary search in cumsum
                auto it = std::lower_bound(cumsum.begin(), cumsum.end(), u);
                int64_t idx = static_cast<int64_t>(std::distance(cumsum.begin(), it));
                if (idx >= num_categories) idx = num_categories - 1;
                out_row[s] = idx;
            }
        } else {
            // Without replacement: use modified cumsum, zero out sampled
            std::vector<float> weights(row, row + num_categories);
            for (int64_t s = 0; s < num_samples; ++s) {
                // Recompute cumsum from current weights
                std::vector<float> cs(static_cast<size_t>(num_categories));
                cs[0] = weights[0];
                for (int64_t i = 1; i < num_categories; ++i) {
                    cs[static_cast<size_t>(i)] = cs[static_cast<size_t>(i - 1)] + weights[static_cast<size_t>(i)];
                }
                float t = cs[static_cast<size_t>(num_categories - 1)];
                if (t <= 0.0f) {
                    throw std::runtime_error("multinomial: ran out of positive-weight categories");
                }

                float u = uniform(rng) * t;
                auto it = std::lower_bound(cs.begin(), cs.end(), u);
                int64_t idx = static_cast<int64_t>(std::distance(cs.begin(), it));
                if (idx >= num_categories) idx = num_categories - 1;
                out_row[s] = idx;
                weights[static_cast<size_t>(idx)] = 0.0f; // Remove sampled index
            }
        }
    }

    return result;
}

// ============================================================================
// Bernoulli Kernel - Bernoulli distribution sampling
// ============================================================================

auto bernoulli_kernel(const Tensor& probs) -> Tensor {
    // probs: any shape tensor of probabilities in [0, 1]
    // Returns: same shape tensor of 0.0 or 1.0 (Float32)
    Tensor probs_f32 = (probs.dtype() != DType::Float32) ? probs.to(DType::Float32) : probs;
    int64_t n = probs_f32.numel();

    Tensor result(std::vector<int64_t>(probs.shape().begin(), probs.shape().end()),
                  DType::Float32, probs.device());
    const float* p_data = probs_f32.data<float>();
    float* out_data = result.data<float>();

    if (n > static_cast<int64_t>(OMP_THRESHOLD)) {
        unsigned int base_seed = detail::get_base_seed();
        #pragma omp parallel
        {
#ifdef _OPENMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            std::mt19937 local_rng(base_seed + static_cast<unsigned int>(tid));
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            #pragma omp for schedule(static)
            for (int64_t i = 0; i < n; ++i) {
                out_data[i] = (dist(local_rng) < p_data[i]) ? 1.0f : 0.0f;
            }
        }
    } else {
        std::mt19937 rng(detail::get_base_seed());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (int64_t i = 0; i < n; ++i) {
            out_data[i] = (dist(rng) < p_data[i]) ? 1.0f : 0.0f;
        }
    }

    return result;
}

auto normal_sample_kernel(const Tensor& mean, const Tensor& std) -> Tensor {
    Tensor mean_f32 = (mean.dtype() != DType::Float32) ? mean.to(DType::Float32) : mean;
    Tensor std_f32 = (std.dtype() != DType::Float32) ? std.to(DType::Float32) : std;
    int64_t n = mean_f32.numel();

    Tensor result(std::vector<int64_t>(mean.shape().begin(), mean.shape().end()),
                  DType::Float32, mean.device());
    const float* m_data = mean_f32.data<float>();
    const float* s_data = std_f32.data<float>();
    float* out_data = result.data<float>();

    if (n > static_cast<int64_t>(OMP_THRESHOLD)) {
        unsigned int base_seed = detail::get_base_seed();
        #pragma omp parallel
        {
#ifdef _OPENMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            std::mt19937 local_rng(base_seed + static_cast<unsigned int>(tid));
            std::normal_distribution<float> dist(0.0f, 1.0f);
            #pragma omp for schedule(static)
            for (int64_t i = 0; i < n; ++i) {
                out_data[i] = m_data[i] + s_data[i] * dist(local_rng);
            }
        }
    } else {
        std::mt19937 rng(detail::get_base_seed());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (int64_t i = 0; i < n; ++i) {
            out_data[i] = m_data[i] + s_data[i] * dist(rng);
        }
    }

    return result;
}

auto poisson_sample_kernel(const Tensor& rates) -> Tensor {
    Tensor rates_f32 = (rates.dtype() != DType::Float32) ? rates.to(DType::Float32) : rates;
    int64_t n = rates_f32.numel();

    Tensor result(std::vector<int64_t>(rates.shape().begin(), rates.shape().end()),
                  DType::Int64, rates.device());
    const float* r_data = rates_f32.data<float>();
    int64_t* out_data = result.data<int64_t>();

    if (n > static_cast<int64_t>(OMP_THRESHOLD)) {
        unsigned int base_seed = detail::get_base_seed();
        #pragma omp parallel
        {
#ifdef _OPENMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            std::mt19937 local_rng(base_seed + static_cast<unsigned int>(tid));
            #pragma omp for schedule(static)
            for (int64_t i = 0; i < n; ++i) {
                std::poisson_distribution<int64_t> dist(static_cast<double>(r_data[i]));
                out_data[i] = dist(local_rng);
            }
        }
    } else {
        std::mt19937 rng(detail::get_base_seed());
        for (int64_t i = 0; i < n; ++i) {
            std::poisson_distribution<int64_t> dist(static_cast<double>(r_data[i]));
            out_data[i] = dist(rng);
        }
    }

    return result;
}

auto exponential_sample_kernel(const Tensor& rate) -> Tensor {
    Tensor rate_f32 = (rate.dtype() != DType::Float32) ? rate.to(DType::Float32) : rate;
    int64_t n = rate_f32.numel();

    Tensor result(std::vector<int64_t>(rate.shape().begin(), rate.shape().end()),
                  DType::Float32, rate.device());
    const float* r_data = rate_f32.data<float>();
    float* out_data = result.data<float>();

    if (n > static_cast<int64_t>(OMP_THRESHOLD)) {
        unsigned int base_seed = detail::get_base_seed();
        #pragma omp parallel
        {
#ifdef _OPENMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            std::mt19937 local_rng(base_seed + static_cast<unsigned int>(tid));
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            #pragma omp for schedule(static)
            for (int64_t i = 0; i < n; ++i) {
                // Inverse CDF: -ln(1-U) / rate
                float u = dist(local_rng);
                out_data[i] = -std::log(1.0f - u) / r_data[i];
            }
        }
    } else {
        std::mt19937 rng(detail::get_base_seed());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (int64_t i = 0; i < n; ++i) {
            float u = dist(rng);
            out_data[i] = -std::log(1.0f - u) / r_data[i];
        }
    }

    return result;
}

} // namespace cpu
} // namespace tenzor
