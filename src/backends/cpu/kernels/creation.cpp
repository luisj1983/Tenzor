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
#include <cstdint>
#include <limits>

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
        case DType::FP8_E4M3FNUZ: {
            parallel_fill_n(result.data<FP8_E4M3FNUZ>(), n, FP8_E4M3FNUZ(1.0f));
            break;
        }
        case DType::FP8_E5M2FNUZ: {
            parallel_fill_n(result.data<FP8_E5M2FNUZ>(), n, FP8_E5M2FNUZ(1.0f));
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
        #pragma omp parallel for schedule(static) if(n > static_cast<int64_t>(OMP_THRESHOLD))
        for (int64_t i = 0; i < n; ++i) {
            data[i] = philox::philox_uniform_f32(seed, i);
        }

    } else if (dtype == DType::Float64) {
        double* data = result.data<double>();
        uint64_t seed = static_cast<uint64_t>(detail::get_base_seed());

        #pragma omp parallel for schedule(static) if(n > static_cast<int64_t>(OMP_THRESHOLD))
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
        #pragma omp parallel for schedule(static) if(n > static_cast<int64_t>(OMP_THRESHOLD))
        for (int64_t i = 0; i < n; ++i) {
            data[i] = philox::philox_normal_f32(seed, i);
        }

    } else if (dtype == DType::Float64) {
        double* data = result.data<double>();
        uint64_t seed = static_cast<uint64_t>(detail::get_base_seed());

        #pragma omp parallel for schedule(static) if(n > static_cast<int64_t>(OMP_THRESHOLD))
        for (int64_t i = 0; i < n; ++i) {
            data[i] = philox::philox_normal_f64(seed, i);
        }

    } else if (dtype == DType::Float16 || dtype == DType::BFloat16) {
        // Audit J14: widen to Float32, generate normals, then narrow.
        Tensor tmp = randn_kernel(shape, DType::Float32, device);
        return tmp.to(dtype);

    } else if (dtype == DType::Complex64) {
        // Standard complex normal: total variance 1 => variance 1/2 per real
        // and imaginary component (std = sqrt(0.5)), matching torch.randn for
        // complex dtypes. Filling each component with N(0,1) would give
        // E[|z|^2] = 2 (a factor-sqrt(2) too large in magnitude). Real and
        // imaginary parts draw from disjoint Philox counter slots (2*i, 2*i+1)
        // so they're independent and the stream stays reproducible/thread-safe.
        std::complex<float>* data = result.data<std::complex<float>>();
        uint64_t seed = static_cast<uint64_t>(detail::get_base_seed());
        constexpr float kStd = 0.7071067811865476f;  // sqrt(0.5)

        #pragma omp parallel for schedule(static) if(n > static_cast<int64_t>(OMP_THRESHOLD))
        for (int64_t i = 0; i < n; ++i) {
            float re = kStd * philox::philox_normal_f32(seed, 2 * i);
            float im = kStd * philox::philox_normal_f32(seed, 2 * i + 1);
            data[i] = {re, im};
        }

    } else if (dtype == DType::Complex128) {
        std::complex<double>* data = result.data<std::complex<double>>();
        uint64_t seed = static_cast<uint64_t>(detail::get_base_seed());
        constexpr double kStd = 0.70710678118654752440;  // sqrt(0.5)

        #pragma omp parallel for schedule(static) if(n > static_cast<int64_t>(OMP_THRESHOLD))
        for (int64_t i = 0; i < n; ++i) {
            double re = kStd * philox::philox_normal_f64(seed, 2 * i);
            double im = kStd * philox::philox_normal_f64(seed, 2 * i + 1);
            data[i] = {re, im};
        }

    } else {
        throw std::runtime_error(
            "randn operation supports Float32/Float64/Float16/BFloat16/Complex64/Complex128");
    }

    return result;
}

// ============================================================================
// Randint Kernel - Create tensor with random integers in [low, high)
// ============================================================================

auto randint_kernel(int64_t low, int64_t high, const std::vector<int64_t>& shape,
                    DType dtype, const Device& device) -> Tensor {
    // Audit J14: widen smaller integer dtypes to Int32, generate, then narrow.
    // Validate that the generated range [low, high) fits the target dtype's
    // representable range before widening, otherwise `.to(dtype)` would wrap
    // modulo the target width and silently produce out-of-range values.
    if (dtype == DType::Int8 || dtype == DType::UInt8 ||
        dtype == DType::Int16 || dtype == DType::UInt16) {
        int64_t type_min = 0, type_max = 0;
        switch (dtype) {
            case DType::Int8:   type_min = INT8_MIN;  type_max = INT8_MAX;  break;
            case DType::UInt8:  type_min = 0;         type_max = UINT8_MAX; break;
            case DType::Int16:  type_min = INT16_MIN; type_max = INT16_MAX; break;
            case DType::UInt16: type_min = 0;         type_max = UINT16_MAX; break;
            default: break;  // unreachable
        }
        // Values are generated in [low, high-1]; require both endpoints to fit.
        if (high <= low || low < type_min || (high - 1) > type_max) {
            throw std::runtime_error(
                "randint: range [" + std::to_string(low) + ", " + std::to_string(high) +
                ") does not fit the requested dtype's representable range [" +
                std::to_string(type_min) + ", " + std::to_string(type_max) + "]");
        }
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
        #pragma omp parallel for schedule(static) if(n > static_cast<int64_t>(OMP_THRESHOLD))
        for (int64_t i = 0; i < n; ++i) {
            int64_t v = low + static_cast<int64_t>(philox::philox_uniform_f64(seed, i) * range);
            if (v >= high) v = high - 1;  // guard against u -> 1-eps rounding up
            data[i] = static_cast<int32_t>(v);
        }
    } else {  // Int64
        int64_t* data = result.data<int64_t>();
        #pragma omp parallel for schedule(static) if(n > static_cast<int64_t>(OMP_THRESHOLD))
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
        case DType::FP8_E4M3FNUZ: {
            FP8_E4M3FNUZ* data = result.data<FP8_E4M3FNUZ>();
            std::fill_n(data, n, FP8_E4M3FNUZ(static_cast<float>(value)));
            break;
        }
        case DType::FP8_E5M2FNUZ: {
            FP8_E5M2FNUZ* data = result.data<FP8_E5M2FNUZ>();
            std::fill_n(data, n, FP8_E5M2FNUZ(static_cast<float>(value)));
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

    // Length matches PyTorch's torch.arange: ceil((end - start) / step), but a
    // naive ceil over a floating-point ratio rounds an exact-integer quotient
    // (e.g. (1.0 - 0.0) / 0.1 = 9.999999999999998 or 10.000000000000002) up by
    // one, yielding a spurious final element whose value can also drift past
    // `end`. Snap a ratio that is integral within a relative epsilon to that
    // integer before applying ceil, so exact ranges produce the exact count.
    const double ratio = (end - start) / step;
    double rounded = std::round(ratio);
    double count_d;
    if (std::abs(ratio - rounded) < std::numeric_limits<double>::epsilon() *
                                    std::max(1.0, std::abs(ratio)) * 4.0) {
        count_d = rounded;  // exact integer ratio: half-open interval => `rounded` elements
    } else {
        count_d = std::ceil(ratio);
    }
    int64_t numel = static_cast<int64_t>(count_d);
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
        case DType::UInt8:
            fill(result.data<uint8_t>(), [](double v) { return static_cast<uint8_t>(v); });
            break;
        case DType::UInt16:
            fill(result.data<uint16_t>(), [](double v) { return static_cast<uint16_t>(v); });
            break;
        case DType::UInt32:
            fill(result.data<uint32_t>(), [](double v) { return static_cast<uint32_t>(v); });
            break;
        case DType::UInt64:
            fill(result.data<uint64_t>(), [](double v) { return static_cast<uint64_t>(v); });
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
        case DType::FP8_E4M3FNUZ: fill_eye<FP8_E4M3FNUZ>(result, n, m, FP8_E4M3FNUZ(1.0f)); break;
        case DType::FP8_E5M2FNUZ: fill_eye<FP8_E5M2FNUZ>(result, n, m, FP8_E5M2FNUZ(1.0f)); break;
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
    // Cannot sample from an empty category set. Validate BEFORE the per-row loop
    // and any cumsum access: with replacement=true the !replacement guard below
    // is skipped, so cumsum[0]=row[0] / cumsum[num_categories-1] would OOB
    // read/write on a shape {0} or {N,0} input (F079).
    if (num_categories <= 0) {
        throw std::invalid_argument("multinomial: cannot sample from an empty category set (num_categories == 0)");
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

    // Reproducibility contract (matches rand/randn/randint/poisson_sample in this
    // file): the RNG stream is keyed deterministically from (seed, batch, sample)
    // via splitmix64, so the output is identical regardless of thread count and
    // regardless of how many draws sibling rows consume. The previous single
    // serial std::mt19937 broke that contract (output depended on iteration
    // order / draw counts of earlier rows).
    const uint64_t seed = static_cast<uint64_t>(detail::get_base_seed());
    auto mix64 = [](uint64_t z) {
        z += 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    };

    for (int64_t b = 0; b < batch_size; ++b) {
        const float* row = p_data + b * num_categories;

        // Per-row RNG stream keyed by (seed, b): each row is independent and
        // reproducible. A draw advances this local generator, so within a row
        // the draws are correlation-free; across rows the keys differ.
        std::mt19937_64 local(mix64(seed ^ (static_cast<uint64_t>(b) * 0x9E3779B97F4A7C15ULL)));
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

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
                float u = uniform(local);
                // Binary search in cumsum. Use upper_bound (first index with
                // cdf[idx] > u) rather than lower_bound (first index with
                // cdf[idx] >= u) to match the semantics used by every GPU
                // backend (CUDA/ROCm/OneAPI/Vulkan hand-roll `cdf[mid] <= u`
                // loops, i.e. upper_bound). This matters when u exactly ties
                // a CDF breakpoint: lower_bound could select a zero-weight
                // category whose cumulative sum equals u, while upper_bound
                // correctly skips past it to the next positive-weight
                // category.
                auto it = std::upper_bound(cumsum.begin(), cumsum.end(), u);
                int64_t idx = static_cast<int64_t>(std::distance(cumsum.begin(), it));
                if (idx >= num_categories) idx = num_categories - 1;
                out_row[s] = idx;
            }
        } else {
            // Without replacement: sample proportional to the remaining weights,
            // then remove the sampled category. A Fenwick (binary-indexed) tree
            // over the weights gives O(log C) prefix-sum queries and O(log C)
            // point updates, so the whole row is O((C + num_samples) log C)
            // instead of the previous O(C * num_samples) cumsum-per-draw.
            //
            // The tree is 1-indexed; tree[k] holds a partial sum and prefix(i)
            // returns sum(weights[0..i]). Sampling draws u in [0, total) and
            // finds the smallest index whose inclusive prefix sum exceeds u,
            // matching the lower_bound semantics of the previous code.
            const auto C = static_cast<size_t>(num_categories);
            std::vector<double> tree(C + 1, 0.0);  // double accumulator for stability
            double total_w = 0.0;
            for (size_t i = 0; i < C; ++i) {
                double w = static_cast<double>(row[i]);
                total_w += w;
                // Point-add w at 1-indexed position i+1.
                for (size_t k = i + 1; k <= C; k += k & (~k + 1)) {
                    tree[k] += w;
                }
            }

            for (int64_t s = 0; s < num_samples; ++s) {
                if (!(total_w > 0.0)) {
                    throw std::runtime_error("multinomial: ran out of positive-weight categories");
                }
                double u = static_cast<double>(uniform(local)) * total_w;

                // Fenwick lower_bound: smallest 1-indexed pos whose prefix sum
                // is strictly greater than u (i.e. first index where the
                // inclusive cumulative sum exceeds the draw).
                size_t pos = 0;
                double acc = 0.0;
                size_t step = 1;
                while ((step << 1) <= C) step <<= 1;
                for (; step > 0; step >>= 1) {
                    size_t next = pos + step;
                    if (next <= C && acc + tree[next] <= u) {
                        pos = next;
                        acc += tree[next];
                    }
                }
                // `pos` is the count of categories whose cumulative sum is <= u,
                // so the sampled 0-indexed category is `pos` (clamped).
                int64_t idx = static_cast<int64_t>(pos);
                if (idx >= num_categories) idx = num_categories - 1;
                out_row[s] = idx;

                // Remove the sampled category: subtract its weight from the tree
                // and from total_w so it cannot be drawn again.
                double removed = static_cast<double>(row[static_cast<size_t>(idx)]);
                // Re-derive the live weight via a 1-element prefix difference in
                // case duplicates/zeros were involved; row[idx] is the original
                // weight and each index is sampled at most once, so this is exact.
                for (size_t k = static_cast<size_t>(idx) + 1; k <= C; k += k & (~k + 1)) {
                    tree[k] -= removed;
                }
                total_w -= removed;
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
    #pragma omp parallel for schedule(static) if(n > static_cast<int64_t>(OMP_THRESHOLD))
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
    #pragma omp parallel for schedule(static) if(n > static_cast<int64_t>(OMP_THRESHOLD))
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
    #pragma omp parallel for schedule(static) if(n > static_cast<int64_t>(OMP_THRESHOLD))
    for (int64_t i = 0; i < n; ++i) {
        // std::poisson_distribution requires mean > 0; a negative or NaN mean is
        // undefined behavior and mean == 0 has no well-defined draw. Emit 0 for
        // any non-positive / non-finite rate (the Poisson(0) limit), and only
        // construct the distribution for valid means.
        const double mean = static_cast<double>(r_data[i]);
        if (!(mean > 0.0) || !std::isfinite(mean)) {
            out_data[i] = 0;
            continue;
        }
        std::mt19937_64 local(mix64(seed ^ (static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ULL)));
        std::poisson_distribution<int64_t> dist(mean);
        out_data[i] = dist(local);
    }

    return result;
}

auto exponential_sample_kernel(const Tensor& rate) -> Tensor {
    Tensor rate_f32 = (rate.dtype() != DType::Float32) ? rate.to(DType::Float32) : rate;
    int64_t n = rate_f32.numel();
    const float* r_data = rate_f32.data<float>();

    // The Exponential distribution is only defined for rate > 0 (rate==0
    // gives an undefined +Inf-mean distribution; rate<0 has no valid support
    // at all). Throw rather than silently clamping to FLT_MIN, which produced
    // a ~1e38 "sample" for any non-positive/NaN rate — a placeholder value
    // just as invalid as +Inf or a negative sample would have been. This
    // matches the GPU backends (CUDA/ROCm/OneAPI/Vulkan), which all validate
    // host-side before sampling instead of returning a numeric sentinel.
    for (int64_t i = 0; i < n; ++i) {
        if (!(r_data[i] > 0.0f)) {
            throw std::invalid_argument(
                "exponential: rate must be > 0 (got a non-positive or NaN rate)");
        }
    }

    Tensor result(std::vector<int64_t>(rate.shape().begin(), rate.shape().end()),
                  DType::Float32, rate.device());
    float* out_data = result.data<float>();

    const uint64_t seed = static_cast<uint64_t>(detail::get_base_seed());
    #pragma omp parallel for schedule(static) if(n > static_cast<int64_t>(OMP_THRESHOLD))
    for (int64_t i = 0; i < n; ++i) {
        // Inverse CDF: -ln(1-U) / rate. philox_uniform_f32 is in [0,1) so
        // (1-u) is in (0,1] and the log is finite. rate > 0 is guaranteed by
        // the validation pass above.
        float u = philox::philox_uniform_f32(seed, i);
        out_data[i] = -std::log(1.0f - u) / r_data[i];
    }

    return result;
}


// Native Gamma sampler. concentration (shape parameter alpha) and rate (beta);
// the sample is gamma(alpha, scale=1/beta). std::gamma_distribution implements
// the Marsaglia-Tsang method internally (with Marsaglia's squeeze for alpha<1),
// so this is a real device-side draw with no NumPy round-trip. Both inputs must
// have the same shape (the Python layer broadcasts before dispatch, matching
// normal_sample_kernel).
auto gamma_sample_kernel(const Tensor& concentration, const Tensor& rate) -> Tensor {
    {
        auto as = concentration.shape();
        auto bs = rate.shape();
        bool same = as.size() == bs.size();
        for (size_t d = 0; same && d < as.size(); ++d) same = (as[d] == bs[d]);
        if (!same) {
            throw std::invalid_argument(
                "gamma_sample: concentration and rate must have the same shape");
        }
    }
    Tensor result(std::vector<int64_t>(concentration.shape().begin(),
                                       concentration.shape().end()),
                  concentration.dtype(), concentration.device());
    const uint64_t seed = static_cast<uint64_t>(detail::get_base_seed());
    int64_t n = concentration.numel();

    if (concentration.dtype() == DType::Float32) {
        const float* a_data = concentration.data<float>();
        const float* b_data = rate.data<float>();
        float* out = result.data<float>();
        #pragma omp parallel for schedule(static) if(n > static_cast<int64_t>(OMP_THRESHOLD))
        for (int64_t i = 0; i < n; ++i) {
            // Per-element RNG so the draw is deterministic and parallel-safe.
            std::mt19937_64 gen(seed ^ (static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ULL));
            float alpha = a_data[i] > 0.0f ? a_data[i] : std::numeric_limits<float>::min();
            float beta  = b_data[i] > 0.0f ? b_data[i] : std::numeric_limits<float>::min();
            std::gamma_distribution<float> dist(alpha, 1.0f / beta);
            out[i] = dist(gen);
        }
    } else if (concentration.dtype() == DType::Float64) {
        const double* a_data = concentration.data<double>();
        const double* b_data = rate.data<double>();
        double* out = result.data<double>();
        #pragma omp parallel for schedule(static) if(n > static_cast<int64_t>(OMP_THRESHOLD))
        for (int64_t i = 0; i < n; ++i) {
            std::mt19937_64 gen(seed ^ (static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ULL));
            double alpha = a_data[i] > 0.0 ? a_data[i] : std::numeric_limits<double>::min();
            double beta  = b_data[i] > 0.0 ? b_data[i] : std::numeric_limits<double>::min();
            std::gamma_distribution<double> dist(alpha, 1.0 / beta);
            out[i] = dist(gen);
        }
    } else if (concentration.dtype() == DType::Float16) {
        // Widen to Float32, sample, narrow back — mirrors unary_math_kernel's
        // Float16 path in math.cpp (std::gamma_distribution has no half-
        // precision specialization, and half-precision RNG state would just
        // lose entropy for no benefit).
        const Float16* a_data = concentration.data<Float16>();
        const Float16* b_data = rate.data<Float16>();
        Float16* out = result.data<Float16>();
        #pragma omp parallel for schedule(static) if(n > static_cast<int64_t>(OMP_THRESHOLD))
        for (int64_t i = 0; i < n; ++i) {
            std::mt19937_64 gen(seed ^ (static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ULL));
            float a_f = static_cast<float>(a_data[i]);
            float b_f = static_cast<float>(b_data[i]);
            float alpha = a_f > 0.0f ? a_f : std::numeric_limits<float>::min();
            float beta  = b_f > 0.0f ? b_f : std::numeric_limits<float>::min();
            std::gamma_distribution<float> dist(alpha, 1.0f / beta);
            out[i] = Float16(dist(gen));
        }
    } else if (concentration.dtype() == DType::BFloat16) {
        const BFloat16* a_data = concentration.data<BFloat16>();
        const BFloat16* b_data = rate.data<BFloat16>();
        BFloat16* out = result.data<BFloat16>();
        #pragma omp parallel for schedule(static) if(n > static_cast<int64_t>(OMP_THRESHOLD))
        for (int64_t i = 0; i < n; ++i) {
            std::mt19937_64 gen(seed ^ (static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ULL));
            float a_f = static_cast<float>(a_data[i]);
            float b_f = static_cast<float>(b_data[i]);
            float alpha = a_f > 0.0f ? a_f : std::numeric_limits<float>::min();
            float beta  = b_f > 0.0f ? b_f : std::numeric_limits<float>::min();
            std::gamma_distribution<float> dist(alpha, 1.0f / beta);
            out[i] = BFloat16(dist(gen));
        }
    } else {
        throw std::runtime_error("gamma_sample: unsupported dtype (Float32/Float64/Float16/BFloat16 only)");
    }
    return result;
}

} // namespace cpu
} // namespace tenzor
