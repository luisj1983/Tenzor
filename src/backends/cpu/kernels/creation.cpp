#include "tenzor/core/tensor.hpp"
#include "tenzor/core/shape.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/utils/error.hpp"
#include "philox.hpp"
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
    // IEEE 754 float/double zero, all integer zeros, bool false, complex zero,
    // FP8 zero, and quantized raw-zero are all represented as all-bits-zero.
    // A single memset covers every advertised dtype correctly.
    size_t nbytes = static_cast<size_t>(result.numel()) * dtype_size(dtype);
    std::memset(result.data_ptr(), 0, nbytes);
    return result;
}

// Helper: fill n elements in parallel if n >= OMP_THRESHOLD, else serial.
template <typename T>
static inline void parallel_fill_n(T* dst, int64_t n, T value) {
    if (n < static_cast<int64_t>(OMP_THRESHOLD)) {
        std::fill_n(dst, static_cast<size_t>(n), value);
    } else {
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i) dst[i] = value;
    }
}

// ============================================================================
// Ones Kernel - Create tensor filled with ones
// ============================================================================

auto ones_kernel(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor {
    Tensor result(shape, dtype, device);
    int64_t n = result.numel();

    switch (dtype) {
        case DType::Float16: {
            parallel_fill_n(result.data<Float16>(), n, Float16(1.0f));
            break;
        }
        case DType::BFloat16: {
            parallel_fill_n(result.data<BFloat16>(), n, BFloat16(1.0f));
            break;
        }
        case DType::Float32: {
            parallel_fill_n(result.data<float>(), n, 1.0f);
            break;
        }
        case DType::Float64: {
            parallel_fill_n(result.data<double>(), n, 1.0);
            break;
        }
        case DType::Int8: {
            parallel_fill_n(result.data<int8_t>(), n, static_cast<int8_t>(1));
            break;
        }
        case DType::Int16: {
            parallel_fill_n(result.data<int16_t>(), n, static_cast<int16_t>(1));
            break;
        }
        case DType::Int32: {
            parallel_fill_n(result.data<int32_t>(), n, static_cast<int32_t>(1));
            break;
        }
        case DType::Int64: {
            parallel_fill_n(result.data<int64_t>(), n, static_cast<int64_t>(1));
            break;
        }
        case DType::UInt8: {
            parallel_fill_n(result.data<uint8_t>(), n, static_cast<uint8_t>(1));
            break;
        }
        case DType::UInt16: {
            parallel_fill_n(result.data<uint16_t>(), n, static_cast<uint16_t>(1));
            break;
        }
        case DType::UInt32: {
            parallel_fill_n(result.data<uint32_t>(), n, static_cast<uint32_t>(1));
            break;
        }
        case DType::UInt64: {
            parallel_fill_n(result.data<uint64_t>(), n, static_cast<uint64_t>(1));
            break;
        }
        case DType::Bool: {
            parallel_fill_n(result.data<bool>(), n, true);
            break;
        }
        case DType::Complex64: {
            parallel_fill_n(result.data<std::complex<float>>(), n,
                            std::complex<float>(1.0f, 0.0f));
            break;
        }
        case DType::Complex128: {
            parallel_fill_n(result.data<std::complex<double>>(), n,
                            std::complex<double>(1.0, 0.0));
            break;
        }
        case DType::FP8_E4M3: {
            parallel_fill_n(result.data<FP8_E4M3>(), n, FP8_E4M3(1.0f));
            break;
        }
        case DType::FP8_E5M2: {
            parallel_fill_n(result.data<FP8_E5M2>(), n, FP8_E5M2(1.0f));
            break;
        }
        default:
            throw std::runtime_error("ones operation: unsupported dtype " +
                std::to_string(static_cast<int>(dtype)));
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
        uint64_t seed = static_cast<uint64_t>(detail::get_base_seed());

        // Philox keyed by (seed, element_index) — result is identical regardless
        // of OMP_NUM_THREADS, satisfying reproducibility requirements.
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            data[i] = philox::philox_uniform_f32(seed, i);
        }

    } else if (dtype == DType::Float64) {
        double* data = result.data<double>();
        uint64_t seed = static_cast<uint64_t>(detail::get_base_seed());

        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            data[i] = philox::philox_uniform_f64(seed, i);
        }

    } else if (dtype == DType::Float16 || dtype == DType::BFloat16) {
        // Audit J14: widen to Float32, generate, then narrow to half. The
        // standard library has no `<random>` distribution that accepts
        // half-precision types directly. Output values stay in [0, 1).
        Tensor tmp = rand_kernel(shape, DType::Float32, device);
        return tmp.to(dtype);
    } else {
        throw std::runtime_error("rand operation supports Float32/Float64/Float16/BFloat16");
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
        uint64_t seed = static_cast<uint64_t>(detail::get_base_seed());

        // Philox keyed by (seed, element_index) — identical output regardless
        // of OMP_NUM_THREADS, satisfying reproducibility requirements.
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            data[i] = philox::philox_normal_f32(seed, i);
        }

    } else if (dtype == DType::Float64) {
        double* data = result.data<double>();
        uint64_t seed = static_cast<uint64_t>(detail::get_base_seed());

        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            data[i] = philox::philox_normal_f64(seed, i);
        }

    } else if (dtype == DType::Float16 || dtype == DType::BFloat16) {
        // Audit J14: widen to Float32, generate normals, then narrow.
        Tensor tmp = randn_kernel(shape, DType::Float32, device);
        return tmp.to(dtype);
    } else {
        throw std::runtime_error("randn operation supports Float32/Float64/Float16/BFloat16");
    }

    return result;
}

// ============================================================================
// Randint Kernel - Create tensor with random integers in [low, high)
// ============================================================================

auto randint_kernel(int64_t low, int64_t high, const std::vector<int64_t>& shape,
                    DType dtype, const Device& device) -> Tensor {
    // Audit J14: widen smaller integer dtypes to Int32, generate, then narrow.
    // Range validation: caller must pass `low`/`high` that fit the target.
    if (dtype == DType::Int8 || dtype == DType::UInt8 ||
        dtype == DType::Int16 || dtype == DType::UInt16) {
        Tensor tmp = randint_kernel(low, high, shape, DType::Int32, device);
        return tmp.to(dtype);
    }
    if (dtype != DType::Int32 && dtype != DType::Int64) {
        throw std::runtime_error(
            "randint operation supports Int8/UInt8/Int16/UInt16/Int32/Int64 dtypes");
    }

    Tensor result(shape, dtype, device);
    int64_t n = result.numel();
    const uint64_t seed = static_cast<uint64_t>(detail::get_base_seed());
    const double range = static_cast<double>(high - low);

    // Philox keyed by (seed, element_index): output is identical regardless of
    // OMP_NUM_THREADS, matching the rand/randn reproducibility guarantee. The
    // previous per-thread mt19937 seeding made results depend on thread count.
    if (dtype == DType::Int32) {
        int32_t* data = result.data<int32_t>();
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            int64_t v = low + static_cast<int64_t>(philox::philox_uniform_f64(seed, i) * range);
            if (v >= high) v = high - 1;  // guard against u -> 1-eps rounding up
            data[i] = static_cast<int32_t>(v);
        }
    } else {  // Int64
        int64_t* data = result.data<int64_t>();
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            int64_t v = low + static_cast<int64_t>(philox::philox_uniform_f64(seed, i) * range);
            if (v >= high) v = high - 1;
            data[i] = v;
        }
    }

    return result;
}

// ============================================================================
// Full Kernel - Create tensor filled with a specific value
// ============================================================================

auto full_kernel(const std::vector<int64_t>& shape, double value, DType dtype, const Device& device) -> Tensor {
    Tensor result(shape, dtype, device);
    int64_t n = result.numel();

    switch (dtype) {
        case DType::Float16: {
            Float16* data = result.data<Float16>();
            std::fill_n(data, n, Float16(static_cast<float>(value)));
            break;
        }
        case DType::BFloat16: {
            BFloat16* data = result.data<BFloat16>();
            std::fill_n(data, n, BFloat16(static_cast<float>(value)));
            break;
        }
        case DType::Float32: {
            float* data = result.data<float>();
            std::fill_n(data, n, static_cast<float>(value));
            break;
        }
        case DType::Float64: {
            double* data = result.data<double>();
            std::fill_n(data, n, value);
            break;
        }
        case DType::Int8: {
            int8_t* data = result.data<int8_t>();
            std::fill_n(data, n, static_cast<int8_t>(value));
            break;
        }
        case DType::Int16: {
            int16_t* data = result.data<int16_t>();
            std::fill_n(data, n, static_cast<int16_t>(value));
            break;
        }
        case DType::Int32: {
            int32_t* data = result.data<int32_t>();
            std::fill_n(data, n, static_cast<int32_t>(value));
            break;
        }
        case DType::Int64: {
            int64_t* data = result.data<int64_t>();
            std::fill_n(data, n, static_cast<int64_t>(value));
            break;
        }
        case DType::UInt8: {
            uint8_t* data = result.data<uint8_t>();
            std::fill_n(data, n, static_cast<uint8_t>(value));
            break;
        }
        case DType::UInt16: {
            uint16_t* data = result.data<uint16_t>();
            std::fill_n(data, n, static_cast<uint16_t>(value));
            break;
        }
        case DType::UInt32: {
            uint32_t* data = result.data<uint32_t>();
            std::fill_n(data, n, static_cast<uint32_t>(value));
            break;
        }
        case DType::UInt64: {
            uint64_t* data = result.data<uint64_t>();
            std::fill_n(data, n, static_cast<uint64_t>(value));
            break;
        }
        case DType::Bool: {
            bool* data = result.data<bool>();
            std::fill_n(data, n, value != 0.0);
            break;
        }
        case DType::Complex64: {
            auto* data = result.data<std::complex<float>>();
            std::fill_n(data, n, std::complex<float>(static_cast<float>(value), 0.0f));
            break;
        }
        case DType::Complex128: {
            auto* data = result.data<std::complex<double>>();
            std::fill_n(data, n, std::complex<double>(value, 0.0));
            break;
        }
        case DType::FP8_E4M3: {
            FP8_E4M3* data = result.data<FP8_E4M3>();
            std::fill_n(data, n, FP8_E4M3(static_cast<float>(value)));
            break;
        }
        case DType::FP8_E5M2: {
            FP8_E5M2* data = result.data<FP8_E5M2>();
            std::fill_n(data, n, FP8_E5M2(static_cast<float>(value)));
            break;
        }
        default:
            throw std::runtime_error("full operation: unsupported dtype " +
                std::to_string(static_cast<int>(dtype)));
    }

    return result;
}

// ============================================================================
// Arange Kernel - Create tensor with evenly spaced values
// ============================================================================

auto arange_kernel(double start, double end, double step, DType dtype, const Device& device) -> Tensor {
    if (step == 0.0) {
        throw std::runtime_error("arange: step must be non-zero");
    }
    if ((step > 0 && start >= end) || (step < 0 && start <= end)) {
        return Tensor({0}, dtype, device);
    }

    int64_t numel = static_cast<int64_t>(std::ceil((end - start) / step));
    if (numel < 0) numel = 0;
    Tensor result({numel}, dtype, device);

    switch (dtype) {
        case DType::Float16: {
            Float16* data = result.data<Float16>();
            for (int64_t i = 0; i < numel; ++i)
                data[i] = Float16(static_cast<float>(start + i * step));
            break;
        }
        case DType::BFloat16: {
            BFloat16* data = result.data<BFloat16>();
            for (int64_t i = 0; i < numel; ++i)
                data[i] = BFloat16(static_cast<float>(start + i * step));
            break;
        }
        case DType::Float32: {
            float* data = result.data<float>();
            for (int64_t i = 0; i < numel; ++i)
                data[i] = static_cast<float>(start + i * step);
            break;
        }
        case DType::Float64: {
            double* data = result.data<double>();
            for (int64_t i = 0; i < numel; ++i)
                data[i] = start + i * step;
            break;
        }
        case DType::Int8: {
            int8_t* data = result.data<int8_t>();
            for (int64_t i = 0; i < numel; ++i)
                data[i] = static_cast<int8_t>(start + i * step);
            break;
        }
        case DType::Int16: {
            int16_t* data = result.data<int16_t>();
            for (int64_t i = 0; i < numel; ++i)
                data[i] = static_cast<int16_t>(start + i * step);
            break;
        }
        case DType::Int32: {
            int32_t* data = result.data<int32_t>();
            for (int64_t i = 0; i < numel; ++i)
                data[i] = static_cast<int32_t>(start + i * step);
            break;
        }
        case DType::Int64: {
            int64_t* data = result.data<int64_t>();
            for (int64_t i = 0; i < numel; ++i)
                data[i] = static_cast<int64_t>(start + i * step);
            break;
        }
        case DType::UInt8: {
            uint8_t* data = result.data<uint8_t>();
            for (int64_t i = 0; i < numel; ++i)
                data[i] = static_cast<uint8_t>(start + i * step);
            break;
        }
        case DType::UInt16: {
            uint16_t* data = result.data<uint16_t>();
            for (int64_t i = 0; i < numel; ++i)
                data[i] = static_cast<uint16_t>(start + i * step);
            break;
        }
        case DType::UInt32: {
            uint32_t* data = result.data<uint32_t>();
            for (int64_t i = 0; i < numel; ++i)
                data[i] = static_cast<uint32_t>(start + i * step);
            break;
        }
        case DType::UInt64: {
            uint64_t* data = result.data<uint64_t>();
            for (int64_t i = 0; i < numel; ++i)
                data[i] = static_cast<uint64_t>(start + i * step);
            break;
        }
        case DType::Bool: {
            bool* data = result.data<bool>();
            for (int64_t i = 0; i < numel; ++i)
                data[i] = (start + i * step) != 0.0;
            break;
        }
        case DType::Complex64: {
            auto* data = result.data<std::complex<float>>();
            for (int64_t i = 0; i < numel; ++i)
                data[i] = std::complex<float>(static_cast<float>(start + i * step), 0.0f);
            break;
        }
        case DType::Complex128: {
            auto* data = result.data<std::complex<double>>();
            for (int64_t i = 0; i < numel; ++i)
                data[i] = std::complex<double>(start + i * step, 0.0);
            break;
        }
        default:
            throw std::runtime_error("arange operation: unsupported dtype " +
                std::to_string(static_cast<int>(dtype)));
    }

    return result;
}

// ============================================================================
// Linspace Kernel - Create tensor with linearly spaced values
// ============================================================================

auto linspace_kernel(double start, double end, int64_t steps, DType dtype, const Device& device) -> Tensor {
    if (steps < 0) {
        throw std::runtime_error("linspace: number of steps must be non-negative");
    }
    if (steps == 0) {
        return Tensor({0}, dtype, device);
    }

    Tensor result({steps}, dtype, device);

    double step_size = (steps > 1)
        ? (end - start) / static_cast<double>(steps - 1)
        : 0.0;

    auto fill = [&](auto* ptr, auto cast) {
        if (steps == 1) {
            ptr[0] = cast(start);
            return;
        }
        for (int64_t i = 0; i < steps; ++i) {
            ptr[i] = cast(start + static_cast<double>(i) * step_size);
        }
        ptr[steps - 1] = cast(end);
    };

    switch (dtype) {
        case DType::Float32:
            fill(result.data<float>(), [](double v) { return static_cast<float>(v); });
            break;
        case DType::Float64:
            fill(result.data<double>(), [](double v) { return v; });
            break;
        case DType::Float16:
            fill(result.data<Float16>(),
                 [](double v) { return Float16(static_cast<float>(v)); });
            break;
        case DType::BFloat16:
            fill(result.data<BFloat16>(),
                 [](double v) { return BFloat16(static_cast<float>(v)); });
            break;
        case DType::Int8:
            fill(result.data<int8_t>(), [](double v) { return static_cast<int8_t>(v); });
            break;
        case DType::Int16:
            fill(result.data<int16_t>(), [](double v) { return static_cast<int16_t>(v); });
            break;
        case DType::Int32:
            fill(result.data<int32_t>(), [](double v) { return static_cast<int32_t>(v); });
            break;
        case DType::Int64:
            fill(result.data<int64_t>(), [](double v) { return static_cast<int64_t>(v); });
            break;
        case DType::Complex64:
            fill(result.data<std::complex<float>>(),
                 [](double v) { return std::complex<float>(static_cast<float>(v), 0.0f); });
            break;
        case DType::Complex128:
            fill(result.data<std::complex<double>>(),
                 [](double v) { return std::complex<double>(v, 0.0); });
            break;
        default:
            throw std::runtime_error("linspace operation: dtype not supported " +
                std::to_string(static_cast<int>(dtype)));
    }

    return result;
}

// ============================================================================
// Eye Kernel - Create identity matrix
// ============================================================================

// Zero-fill an (n x m) buffer and set its main diagonal to `one`.  T{} is the
// additive zero for every supported dtype (int 0, +0.0 float/half/fp8 bit
// pattern, (0,0) complex, false bool), so this works across the full matrix.
template <typename T>
static void fill_eye(Tensor& result, int64_t n, int64_t m, T one) {
    T* data = result.data<T>();
    const size_t total = static_cast<size_t>(n) * static_cast<size_t>(m);
    parallel_fill_n(data, total, T{});
    const int64_t diag_len = std::min(n, m);
    for (int64_t i = 0; i < diag_len; ++i) {
        data[i * m + i] = one;
    }
}

auto eye_kernel(int64_t n, int64_t m, DType dtype, const Device& device) -> Tensor {
    if (m < 0) m = n;  // Square matrix by default

    Tensor result({n, m}, dtype, device);

    switch (dtype) {
        case DType::Float16:    fill_eye<Float16>(result, n, m, Float16(1.0f)); break;
        case DType::BFloat16:   fill_eye<BFloat16>(result, n, m, BFloat16(1.0f)); break;
        case DType::Float32:    fill_eye<float>(result, n, m, 1.0f); break;
        case DType::Float64:    fill_eye<double>(result, n, m, 1.0); break;
        case DType::Int8:       fill_eye<int8_t>(result, n, m, static_cast<int8_t>(1)); break;
        case DType::Int16:      fill_eye<int16_t>(result, n, m, static_cast<int16_t>(1)); break;
        case DType::Int32:      fill_eye<int32_t>(result, n, m, static_cast<int32_t>(1)); break;
        case DType::Int64:      fill_eye<int64_t>(result, n, m, static_cast<int64_t>(1)); break;
        case DType::UInt8:      fill_eye<uint8_t>(result, n, m, static_cast<uint8_t>(1)); break;
        case DType::UInt16:     fill_eye<uint16_t>(result, n, m, static_cast<uint16_t>(1)); break;
        case DType::UInt32:     fill_eye<uint32_t>(result, n, m, static_cast<uint32_t>(1)); break;
        case DType::UInt64:     fill_eye<uint64_t>(result, n, m, static_cast<uint64_t>(1)); break;
        case DType::Bool:       fill_eye<bool>(result, n, m, true); break;
        case DType::Complex64:  fill_eye<std::complex<float>>(result, n, m, std::complex<float>(1.0f, 0.0f)); break;
        case DType::Complex128: fill_eye<std::complex<double>>(result, n, m, std::complex<double>(1.0, 0.0)); break;
        case DType::FP8_E4M3:   fill_eye<FP8_E4M3>(result, n, m, FP8_E4M3(1.0f)); break;
        case DType::FP8_E5M2:   fill_eye<FP8_E5M2>(result, n, m, FP8_E5M2(1.0f)); break;
        default:
            throw std::runtime_error("eye operation: unsupported dtype " +
                std::to_string(static_cast<int>(dtype)));
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
    // Returns: same shape tensor of 0.0 or 1.0 with the SAME dtype as `probs`
    // (S13 fix — dtype-preservation: previously hard-wired to Float32, which
    // silently downcast Float64 probability tensors).
    //
    // Sampling itself runs at Float32 (the RNG distributions are Float32);
    // the {0,1} output values are exactly representable in every supported
    // float dtype, so the final cast is lossless.
    const DType out_dtype = probs.dtype();
    Tensor probs_f32 = (out_dtype != DType::Float32) ? probs.to(DType::Float32) : probs;
    int64_t n = probs_f32.numel();

    Tensor result_f32(std::vector<int64_t>(probs.shape().begin(), probs.shape().end()),
                      DType::Float32, probs.device());
    const float* p_data = probs_f32.data<float>();
    float* out_data = result_f32.data<float>();

    const uint64_t seed = static_cast<uint64_t>(detail::get_base_seed());
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        out_data[i] = (philox::philox_uniform_f32(seed, i) < p_data[i]) ? 1.0f : 0.0f;
    }

    if (out_dtype == DType::Float32) {
        return result_f32;
    }
    return result_f32.to(out_dtype);
}

auto normal_sample_kernel(const Tensor& mean, const Tensor& std) -> Tensor {
    // S13 fix — dtype-preservation: output dtype tracks `mean.dtype()` rather
    // than being hard-wired to Float32. Internal compute stays in Float32
    // (std::normal_distribution<float> is the supported path); we cast back.
    // If `mean` and `std` disagree (e.g. promoted), `mean.dtype()` wins —
    // the lower-rank parameter has already been broadcast/promoted upstream.
    const DType out_dtype = mean.dtype();
    Tensor mean_f32 = (mean.dtype() != DType::Float32) ? mean.to(DType::Float32) : mean;
    Tensor std_f32 = (std.dtype() != DType::Float32) ? std.to(DType::Float32) : std;
    int64_t n = mean_f32.numel();

    Tensor result_f32(std::vector<int64_t>(mean.shape().begin(), mean.shape().end()),
                      DType::Float32, mean.device());
    const float* m_data = mean_f32.data<float>();
    const float* s_data = std_f32.data<float>();
    float* out_data = result_f32.data<float>();

    const uint64_t seed = static_cast<uint64_t>(detail::get_base_seed());
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        out_data[i] = m_data[i] + s_data[i] * philox::philox_normal_f32(seed, i);
    }

    if (out_dtype == DType::Float32) {
        return result_f32;
    }
    return result_f32.to(out_dtype);
}

auto poisson_sample_kernel(const Tensor& rates) -> Tensor {
    Tensor rates_f32 = (rates.dtype() != DType::Float32) ? rates.to(DType::Float32) : rates;
    int64_t n = rates_f32.numel();

    Tensor result(std::vector<int64_t>(rates.shape().begin(), rates.shape().end()),
                  DType::Int64, rates.device());
    const float* r_data = rates_f32.data<float>();
    int64_t* out_data = result.data<int64_t>();

    const uint64_t seed = static_cast<uint64_t>(detail::get_base_seed());
    // Poisson draws a variable number of uniforms per element, so a single
    // philox(seed,i) value is insufficient. Seed a per-element RNG stream
    // deterministically from (seed, i) via splitmix64 — thread-count
    // independent, unlike the previous base_seed+tid scheme.
    auto mix64 = [](uint64_t z) {
        z += 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    };
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        std::mt19937_64 local(mix64(seed ^ (static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ULL)));
        std::poisson_distribution<int64_t> dist(static_cast<double>(r_data[i]));
        out_data[i] = dist(local);
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

    const uint64_t seed = static_cast<uint64_t>(detail::get_base_seed());
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        // Inverse CDF: -ln(1-U) / rate. philox_uniform_f32 is in [0,1) so
        // (1-u) is in (0,1] and the log is finite.
        float u = philox::philox_uniform_f32(seed, i);
        out_data[i] = -std::log(1.0f - u) / r_data[i];
    }

    return result;
}

} // namespace cpu
} // namespace tenzor
