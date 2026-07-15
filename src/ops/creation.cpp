#include "tenzor/ops/creation.hpp"
#include "tenzor/core/generator.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"  // argsort (GPU randperm)
#include "tenzor/ops/transform.hpp"  // reshape/expand (device-agnostic meshgrid)
#include "tenzor/distributions/distribution.hpp"
#include "tenzor/backend/dispatch.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include <random>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <algorithm>
#include <sstream>
#include <chrono>
#include <complex>

namespace tenzor {

// Saturating double->integer conversion. A plain static_cast<intT>(double)
// is undefined behaviour when the source is NaN/Inf or outside the target
// range (platform-dependent garbage). This clamps to the target's
// representable range and maps NaN to 0, matching a defined fill semantics
// for full()/arange()/linspace() integer outputs.
template <typename IntT>
static inline IntT saturate_to_int(double value) {
    if (std::isnan(value)) return IntT(0);
    constexpr double lo = static_cast<double>(std::numeric_limits<IntT>::lowest());
    constexpr double hi = static_cast<double>(std::numeric_limits<IntT>::max());
    if (value <= lo) return std::numeric_limits<IntT>::lowest();
    if (value >= hi) return std::numeric_limits<IntT>::max();
    return static_cast<IntT>(value);
}

// Thread-local RNG — each thread has independent random state
static thread_local std::mt19937 global_rng(std::random_device{}());
static thread_local bool manual_seed_set = false;
static thread_local uint64_t manual_seed_value = 0;

namespace detail {

// Audit D.3: expose the thread-local global RNG state to other TUs (in
// particular core/generator.cpp) so checkpoint save/restore can snapshot
// it without making global_rng a header-visible symbol.
auto get_global_rng_engine() -> std::mt19937& { return global_rng; }
auto get_global_manual_seed_set() -> bool { return manual_seed_set; }
auto set_global_manual_seed_set(bool v) -> void { manual_seed_set = v; }
auto get_global_manual_seed_value() -> uint64_t { return manual_seed_value; }
auto set_global_manual_seed_value(uint64_t v) -> void { manual_seed_value = v; }

} // namespace detail

// Function to access the thread-local RNG
static std::mt19937& get_rng() {
    return global_rng;
}

// Public function to set the random seed (affects calling thread)
void manual_seed(unsigned int seed) {
    global_rng.seed(seed);
    manual_seed_set = true;
    manual_seed_value = seed;
}

// Get a seed for backend RNG: returns the manual seed if set, otherwise a
// time-based seed.
//
// CONTRACT (consume-on-read): when a manual seed is active this function
// CONSUMES one seed value per call and advances the stored manual seed. This
// keeps successive backend RNG draws deterministic-but-distinct, but it also
// means the stored manual seed is mutated as a side effect: a snapshot taken
// after manual_seed(s) plus some GPU creation ops will NOT reproduce the
// original state s. Callers that need to reproduce a specific manual_seed
// snapshot must re-issue manual_seed() rather than relying on the stored
// value remaining equal to s. CPU and GPU runs seeded identically diverge for
// the same reason; this is intentional, not a bug.
uint64_t get_global_seed() {
    if (manual_seed_set) {
        // Increment so successive calls get different but deterministic seeds
        return manual_seed_value++;
    }
    auto now = std::chrono::high_resolution_clock::now();
    return static_cast<uint64_t>(now.time_since_epoch().count());
}

// Helper function to convert shape vector to comma-separated string
static auto shape_to_string(const std::vector<int64_t>& shape) -> std::string {
    std::ostringstream oss;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) oss << ",";
        oss << shape[i];
    }
    return oss.str();
}

// Helper function to convert DType to string
static auto dtype_to_string(DType dtype) -> std::string {
    switch (dtype) {
        case DType::Float32: return "float32";
        case DType::Float64: return "float64";
        case DType::Float16: return "float16";
        case DType::BFloat16: return "bfloat16";
        case DType::Int8: return "int8";
        case DType::Int16: return "int16";
        case DType::Int32: return "int32";
        case DType::Int64: return "int64";
        case DType::UInt8: return "uint8";
        case DType::UInt16: return "uint16";
        case DType::UInt32: return "uint32";
        case DType::UInt64: return "uint64";
        case DType::Bool: return "bool";
        case DType::Complex64: return "complex64";
        case DType::Complex128: return "complex128";
        case DType::FP8_E4M3: return "fp8_e4m3";
        case DType::FP8_E5M2: return "fp8_e5m2";
        case DType::FP8_E4M3FNUZ: return "fp8_e4m3fnuz";
        case DType::FP8_E5M2FNUZ: return "fp8_e5m2fnuz";
        case DType::QInt8: return "qint8";
        case DType::QUInt8: return "quint8";
        case DType::QInt4x2: return "qint4x2";
        default: return "unknown";
    }
}

auto zeros(std::vector<int64_t> shape, DType dtype, Device device) -> Tensor {
    // Use OpId dispatch for non-CPU devices
    if (device.type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::Shape, shape_to_string(shape));
        attrs.set(AttrKey::Dtype, dtype_to_string(dtype));
        attrs.set(AttrKey::Device, static_cast<int64_t>(device.index));

        return dispatch_to_device(OpId::Zeros, device.type, {}, attrs)[0];
    }

    // CPU path: validate device index (only index 0 is valid for CPU)
    if (device.index != 0) {
        throw std::runtime_error("Invalid device index " + std::to_string(device.index) +
            " for CPU (only index 0 is valid)");
    }

    // CPU path: use zero-initialized constructor directly
    return Tensor(std::move(shape), dtype, device);
}

// Helper: fill n elements in parallel if n >= threshold, else serial.
// Matches the OMP_THRESHOLD=65536 convention used in the CPU kernel files.
template <typename T>
static inline void parallel_fill_n(T* dst, int64_t n, T value,
                                    int64_t threshold = 65536) {
    if (n < threshold) {
        std::fill_n(dst, static_cast<size_t>(n), value);
    } else {
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i) dst[i] = value;
    }
}

auto ones(std::vector<int64_t> shape, DType dtype, Device device) -> Tensor {
    // Use OpId dispatch for non-CPU devices
    if (device.type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::Shape, shape_to_string(shape));
        attrs.set(AttrKey::Dtype, dtype_to_string(dtype));
        attrs.set(AttrKey::Device, static_cast<int64_t>(device.index));

        return dispatch_to_device(OpId::Ones, device.type, {}, attrs)[0];
    }

    // CPU path: validate device index (only index 0 is valid for CPU)
    if (device.index != 0) {
        throw std::runtime_error("Invalid device index " + std::to_string(device.index) +
            " for CPU (only index 0 is valid)");
    }

    // CPU path: use uninitialized allocation (avoid wasteful zeroing before fill)
    auto tensor = Tensor::empty_uninitialized(std::move(shape), dtype, device);
    if (!tensor.impl() || !tensor.storage()) return tensor;

    int64_t numel = static_cast<int64_t>(tensor.numel());
    void* data = tensor.storage()->data();

    // Fill with ones based on dtype (OMP-parallel for large tensors)
    switch (dtype) {
        case DType::Float16: {
            parallel_fill_n(static_cast<Float16*>(data), numel, Float16(1.0f));
            break;
        }
        case DType::BFloat16: {
            parallel_fill_n(static_cast<BFloat16*>(data), numel, BFloat16(1.0f));
            break;
        }
        case DType::Float32: {
            parallel_fill_n(static_cast<float*>(data), numel, 1.0f);
            break;
        }
        case DType::Float64: {
            parallel_fill_n(static_cast<double*>(data), numel, 1.0);
            break;
        }
        case DType::Int32: {
            parallel_fill_n(static_cast<int32_t*>(data), numel, static_cast<int32_t>(1));
            break;
        }
        case DType::Int64: {
            parallel_fill_n(static_cast<int64_t*>(data), numel, static_cast<int64_t>(1));
            break;
        }
        case DType::UInt8: {
            parallel_fill_n(static_cast<uint8_t*>(data), numel, static_cast<uint8_t>(1));
            break;
        }
        case DType::UInt16: {
            parallel_fill_n(static_cast<uint16_t*>(data), numel, static_cast<uint16_t>(1));
            break;
        }
        case DType::UInt32: {
            parallel_fill_n(static_cast<uint32_t*>(data), numel, static_cast<uint32_t>(1));
            break;
        }
        case DType::UInt64: {
            parallel_fill_n(static_cast<uint64_t*>(data), numel, static_cast<uint64_t>(1));
            break;
        }
        case DType::Int8: {
            parallel_fill_n(static_cast<int8_t*>(data), numel, static_cast<int8_t>(1));
            break;
        }
        case DType::Int16: {
            parallel_fill_n(static_cast<int16_t*>(data), numel, static_cast<int16_t>(1));
            break;
        }
        case DType::Bool: {
            parallel_fill_n(static_cast<bool*>(data), numel, true);
            break;
        }
        case DType::Complex64: {
            parallel_fill_n(static_cast<std::complex<float>*>(data), numel,
                            std::complex<float>(1.0f, 0.0f));
            break;
        }
        case DType::Complex128: {
            parallel_fill_n(static_cast<std::complex<double>*>(data), numel,
                            std::complex<double>(1.0, 0.0));
            break;
        }
        case DType::FP8_E4M3: {
            parallel_fill_n(static_cast<FP8_E4M3*>(data), numel, FP8_E4M3(1.0f));
            break;
        }
        case DType::FP8_E5M2: {
            parallel_fill_n(static_cast<FP8_E5M2*>(data), numel, FP8_E5M2(1.0f));
            break;
        }
        case DType::FP8_E4M3FNUZ: {
            parallel_fill_n(static_cast<FP8_E4M3FNUZ*>(data), numel, FP8_E4M3FNUZ(1.0f));
            break;
        }
        case DType::FP8_E5M2FNUZ: {
            parallel_fill_n(static_cast<FP8_E5M2FNUZ*>(data), numel, FP8_E5M2FNUZ(1.0f));
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for ones()");
    }
    return tensor;
}

// full(..., float value, ...) collapsed into full(..., double value, ...)
// per 2026-05-19 cleanup. float→double promotion + the dtype-cast happen
// downstream so no behaviour changes; the two-overload ambiguity that broke
// `full({3}, 42, ...)` (with an int literal) is gone.

auto full(std::vector<int64_t> shape, double value, DType dtype, Device device) -> Tensor {
    // Use OpId dispatch for non-CPU devices
    if (device.type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::Shape, shape_to_string(shape));
        attrs.set(AttrKey::Value, value);
        attrs.set(AttrKey::Dtype, dtype_to_string(dtype));
        attrs.set(AttrKey::Device, static_cast<int64_t>(device.index));

        return dispatch_to_device(OpId::Full, device.type, {}, attrs)[0];
    }

    // CPU path: use uninitialized allocation (avoid wasteful zeroing before fill)
    auto tensor = Tensor::empty_uninitialized(std::move(shape), dtype, device);
    if (!tensor.impl() || !tensor.storage()) return tensor;

    size_t numel = tensor.numel();
    void* data = tensor.storage()->data();

    // Fill with value based on dtype
    switch (dtype) {
        case DType::Float16: {
            Float16* ptr = static_cast<Float16*>(data);
            std::fill(ptr, ptr + numel, Float16(static_cast<float>(value)));
            break;
        }
        case DType::BFloat16: {
            BFloat16* ptr = static_cast<BFloat16*>(data);
            std::fill(ptr, ptr + numel, BFloat16(static_cast<float>(value)));
            break;
        }
        case DType::Float32: {
            float* ptr = static_cast<float*>(data);
            std::fill(ptr, ptr + numel, static_cast<float>(value));
            break;
        }
        case DType::Float64: {
            double* ptr = static_cast<double*>(data);
            std::fill(ptr, ptr + numel, value);  // No cast needed, already double
            break;
        }
        case DType::Int32: {
            int32_t* ptr = static_cast<int32_t*>(data);
            std::fill(ptr, ptr + numel, saturate_to_int<int32_t>(value));
            break;
        }
        case DType::Int64: {
            int64_t* ptr = static_cast<int64_t*>(data);
            std::fill(ptr, ptr + numel, saturate_to_int<int64_t>(value));
            break;
        }
        case DType::UInt8: {
            uint8_t* ptr = static_cast<uint8_t*>(data);
            std::fill(ptr, ptr + numel, saturate_to_int<uint8_t>(value));
            break;
        }
        case DType::UInt16: {
            uint16_t* ptr = static_cast<uint16_t*>(data);
            std::fill(ptr, ptr + numel, saturate_to_int<uint16_t>(value));
            break;
        }
        case DType::UInt32: {
            uint32_t* ptr = static_cast<uint32_t*>(data);
            std::fill(ptr, ptr + numel, saturate_to_int<uint32_t>(value));
            break;
        }
        case DType::UInt64: {
            uint64_t* ptr = static_cast<uint64_t*>(data);
            std::fill(ptr, ptr + numel, saturate_to_int<uint64_t>(value));
            break;
        }
        case DType::Int8: {
            int8_t* ptr = static_cast<int8_t*>(data);
            std::fill(ptr, ptr + numel, saturate_to_int<int8_t>(value));
            break;
        }
        case DType::Int16: {
            int16_t* ptr = static_cast<int16_t*>(data);
            std::fill(ptr, ptr + numel, saturate_to_int<int16_t>(value));
            break;
        }
        case DType::Bool: {
            bool* ptr = static_cast<bool*>(data);
            std::fill(ptr, ptr + numel, value != 0.0);
            break;
        }
        case DType::Complex64: {
            auto* ptr = static_cast<std::complex<float>*>(data);
            std::fill(ptr, ptr + numel, std::complex<float>(static_cast<float>(value), 0.0f));
            break;
        }
        case DType::Complex128: {
            auto* ptr = static_cast<std::complex<double>*>(data);
            std::fill(ptr, ptr + numel, std::complex<double>(value, 0.0));
            break;
        }
        case DType::FP8_E4M3: {
            FP8_E4M3* ptr = static_cast<FP8_E4M3*>(data);
            std::fill(ptr, ptr + numel, FP8_E4M3(static_cast<float>(value)));
            break;
        }
        case DType::FP8_E5M2: {
            FP8_E5M2* ptr = static_cast<FP8_E5M2*>(data);
            std::fill(ptr, ptr + numel, FP8_E5M2(static_cast<float>(value)));
            break;
        }
        case DType::FP8_E4M3FNUZ: {
            FP8_E4M3FNUZ* ptr = static_cast<FP8_E4M3FNUZ*>(data);
            std::fill(ptr, ptr + numel, FP8_E4M3FNUZ(static_cast<float>(value)));
            break;
        }
        case DType::FP8_E5M2FNUZ: {
            FP8_E5M2FNUZ* ptr = static_cast<FP8_E5M2FNUZ*>(data);
            std::fill(ptr, ptr + numel, FP8_E5M2FNUZ(static_cast<float>(value)));
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for full()");
    }
    return tensor;
}

auto empty(std::vector<int64_t> shape, DType dtype, Device device) -> Tensor {
    // empty() returns truly uninitialized memory - no zeroing
    return Tensor::empty_uninitialized(std::move(shape), dtype, device);
}

auto rand(std::vector<int64_t> shape, DType dtype, Device device) -> Tensor {
    // F042: dispatch to the backend for ALL devices (including CPU) so CPU and
    // GPU share the same Philox RNG stream. The CPU backend rand kernel is
    // Philox-based and bit-identical to the CUDA device Philox for a given seed.
    OpAttributes attrs;
    attrs.set(AttrKey::Shape, shape_to_string(shape));
    attrs.set(AttrKey::Dtype, dtype_to_string(dtype));
    attrs.set(AttrKey::Device, static_cast<int64_t>(device.index));

    return dispatch_to_device(OpId::Rand, device.type, {}, attrs)[0];
}

auto randn(std::vector<int64_t> shape, DType dtype, Device device) -> Tensor {
    // F042: dispatch to the backend for ALL devices so CPU (Philox) and GPU
    // (device Philox) share the same RNG stream; normal samples match to ~ULP
    // (Box-Muller transcendentals differ between host libm and device intrinsics).
    OpAttributes attrs;
    attrs.set(AttrKey::Shape, shape_to_string(shape));
    attrs.set(AttrKey::Dtype, dtype_to_string(dtype));
    attrs.set(AttrKey::Device, static_cast<int64_t>(device.index));

    return dispatch_to_device(OpId::Randn, device.type, {}, attrs)[0];
}

auto randint(int64_t low, int64_t high, std::vector<int64_t> shape, DType dtype, Device device) -> Tensor {
    if (low >= high) {
        throw std::invalid_argument("randint: low must be less than high");
    }

    // F042: dispatch to the backend for ALL devices so CPU and GPU share the
    // same Philox stream (bit-identical for integers).
    OpAttributes attrs;
    attrs.set(AttrKey::Start, static_cast<int64_t>(low));
    attrs.set(AttrKey::End, static_cast<int64_t>(high));
    attrs.set(AttrKey::Shape, shape_to_string(shape));
    attrs.set(AttrKey::Dtype, dtype_to_string(dtype));
    attrs.set(AttrKey::Device, static_cast<int64_t>(device.index));

    return dispatch_to_device(OpId::Randint, device.type, {}, attrs)[0];
}

auto arange(double start, double end, double step, DType dtype, Device device) -> Tensor {
    if (step == 0.0) {
        throw std::invalid_argument("step cannot be zero");
    }
    // Empty half-open range: NumPy/PyTorch return an empty tensor rather than
    // throwing when start>=end with step>0 (or start<=end with step<0).
    if ((step > 0 && start >= end) || (step < 0 && start <= end)) {
        return Tensor::empty_uninitialized({0}, dtype, device);
    }

    // Use OpId dispatch for non-CPU devices
    if (device.type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::Start, start);
        attrs.set(AttrKey::End, end);
        attrs.set(AttrKey::Step, step);
        attrs.set(AttrKey::Dtype, dtype_to_string(dtype));
        attrs.set(AttrKey::Device, static_cast<int64_t>(device.index));

        return dispatch_to_device(OpId::Arange, device.type, {}, attrs)[0];
    }

    // Calculate number of elements with sign-aware tolerant rounding (matching
    // PyTorch). A naive ceil((end-start)/step) over-counts when the exact
    // quotient lands a hair above an integer due to floating-point error: e.g.
    // arange(0.1, 0.4, 0.1) yields a quotient of 3.0000000000000004 -> ceil 4,
    // erroneously emitting the excluded half-open end value 0.4. Subtracting a
    // relative epsilon from the quotient (with the same sign as step so the
    // half-open range is respected for both ascending and descending ranges)
    // collapses these spurious extra elements. Exact integer counts (e.g.
    // arange(0, 5, 1) -> 5) are unaffected because the epsilon is far smaller
    // than 1.
    double range = end - start;
    double raw = range / step;
    // Robust element count. The quotient carries FP error that grows with its
    // magnitude (~|raw|*eps_machine), so a fixed absolute epsilon is wrong for
    // large ranges while a large relative epsilon (the previous 1e-9*|raw|, which
    // reaches 1.0 once |raw|>=1e9) silently drops whole elements — e.g.
    // arange(0, 1e9, 1) returned 999,999,999. Instead: if the quotient is within
    // a magnitude-scaled but strictly sub-half tolerance of an integer, the final
    // grid point coincides with the excluded half-open end, so the count is that
    // integer; otherwise round up to include the trailing partial step. Exact
    // integer counts (arange(0,5,1)=5, arange(0,1e9,1)=1e9) are preserved, and
    // FP-overshoot cases (arange(0.1,0.4,0.1)=3) still collapse correctly.
    double rounded = std::round(raw);
    double tol = std::min(0.25, std::max(1e-9, std::fabs(raw) * 1e-12));
    int64_t numel = (std::fabs(raw - rounded) < tol)
                        ? static_cast<int64_t>(rounded)
                        : static_cast<int64_t>(std::ceil(raw));
    if (numel < 0) numel = 0;

    // Use uninitialized allocation (avoid wasteful zeroing before fill)
    auto tensor = Tensor::empty_uninitialized({numel}, dtype, device);
    if (!tensor.impl() || !tensor.storage() || numel == 0) return tensor;

    void* data = tensor.storage()->data();

    // Fill with range values using start + i * step to avoid accumulation drift
    switch (dtype) {
        case DType::Float32: {
            float* ptr = static_cast<float*>(data);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = static_cast<float>(start + i * step);
            }
            break;
        }
        case DType::Float64: {
            double* ptr = static_cast<double*>(data);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = start + i * step;
            }
            break;
        }
        case DType::Int32: {
            int32_t* ptr = static_cast<int32_t*>(data);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = saturate_to_int<int32_t>(start + i * step);
            }
            break;
        }
        case DType::Int64: {
            int64_t* ptr = static_cast<int64_t*>(data);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = saturate_to_int<int64_t>(start + i * step);
            }
            break;
        }
        case DType::Float16: {
            Float16* ptr = static_cast<Float16*>(data);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = Float16(static_cast<float>(start + i * step));
            }
            break;
        }
        case DType::Int8: {
            int8_t* ptr = static_cast<int8_t*>(data);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = saturate_to_int<int8_t>(start + i * step);
            }
            break;
        }
        case DType::UInt8: {
            uint8_t* ptr = static_cast<uint8_t*>(data);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = saturate_to_int<uint8_t>(start + i * step);
            }
            break;
        }
        case DType::Int16: {
            int16_t* ptr = static_cast<int16_t*>(data);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = saturate_to_int<int16_t>(start + i * step);
            }
            break;
        }
        case DType::BFloat16: {
            BFloat16* ptr = static_cast<BFloat16*>(data);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = BFloat16(static_cast<float>(start + i * step));
            }
            break;
        }
        case DType::UInt16: {
            uint16_t* ptr = static_cast<uint16_t*>(data);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = saturate_to_int<uint16_t>(start + i * step);
            }
            break;
        }
        case DType::UInt32: {
            uint32_t* ptr = static_cast<uint32_t*>(data);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = saturate_to_int<uint32_t>(start + i * step);
            }
            break;
        }
        case DType::UInt64: {
            uint64_t* ptr = static_cast<uint64_t*>(data);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = saturate_to_int<uint64_t>(start + i * step);
            }
            break;
        }
        case DType::Bool: {
            bool* ptr = static_cast<bool*>(data);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = (start + i * step) != 0.0;
            }
            break;
        }
        case DType::Complex64: {
            auto* ptr = static_cast<std::complex<float>*>(data);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = std::complex<float>(static_cast<float>(start + i * step), 0.0f);
            }
            break;
        }
        case DType::Complex128: {
            auto* ptr = static_cast<std::complex<double>*>(data);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = std::complex<double>(start + i * step, 0.0);
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for arange()");
    }
    return tensor;
}

// linspace(float, float, ...) collapsed into linspace(double, double, ...)
// per 2026-05-19 cleanup: float→double promotion is implicit, eliminating
// the int-literal ambiguity for callers like linspace(0, 1, 5, ...).

auto linspace(double start, double end, int64_t steps, DType dtype, Device device) -> Tensor {
    if (steps <= 0) {
        throw std::invalid_argument("steps must be positive");
    }

    // Use OpId dispatch for non-CPU devices
    if (device.type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::Start, start);
        attrs.set(AttrKey::End, end);
        attrs.set(AttrKey::Steps, steps);
        attrs.set(AttrKey::Dtype, dtype_to_string(dtype));
        attrs.set(AttrKey::Device, static_cast<int64_t>(device.index));

        return dispatch_to_device(OpId::Linspace, device.type, {}, attrs)[0];
    }

    // Use uninitialized allocation (avoid wasteful zeroing before fill)
    auto tensor = Tensor::empty_uninitialized({steps}, dtype, device);
    if (!tensor.impl() || !tensor.storage()) return tensor;

    void* data = tensor.storage()->data();

    // Calculate step size in double precision — preserves all significant digits.
    double step_size = (steps > 1) ? (end - start) / static_cast<double>(steps - 1) : 0.0;

    // Fill with linearly spaced values based on dtype
    switch (dtype) {
        case DType::Float32: {
            float* ptr = static_cast<float*>(data);
            if (steps == 1) {
                ptr[0] = static_cast<float>(start);
            } else {
                for (int64_t i = 0; i < steps; ++i) {
                    ptr[i] = static_cast<float>(start + i * step_size);
                }
                ptr[steps - 1] = static_cast<float>(end);
            }
            break;
        }
        case DType::Float64: {
            double* ptr = static_cast<double*>(data);
            if (steps == 1) {
                ptr[0] = start;
            } else {
                for (int64_t i = 0; i < steps; ++i) {
                    ptr[i] = start + i * step_size;
                }
                // Pin exact endpoints to avoid accumulated fp error.
                ptr[steps - 1] = end;
            }
            break;
        }
        case DType::Float16: {
            Float16* ptr = static_cast<Float16*>(data);
            if (steps == 1) {
                ptr[0] = Float16(static_cast<float>(start));
            } else {
                for (int64_t i = 0; i < steps; ++i) {
                    ptr[i] = Float16(static_cast<float>(start + i * step_size));
                }
                ptr[steps - 1] = Float16(static_cast<float>(end));
            }
            break;
        }
        case DType::BFloat16: {
            BFloat16* ptr = static_cast<BFloat16*>(data);
            if (steps == 1) {
                ptr[0] = BFloat16(static_cast<float>(start));
            } else {
                for (int64_t i = 0; i < steps; ++i) {
                    ptr[i] = BFloat16(static_cast<float>(start + i * step_size));
                }
                ptr[steps - 1] = BFloat16(static_cast<float>(end));
            }
            break;
        }
        case DType::Int8: {
            int8_t* ptr = static_cast<int8_t*>(data);
            for (int64_t i = 0; i < steps; ++i) {
                ptr[i] = saturate_to_int<int8_t>(start + i * step_size);
            }
            if (steps > 1) ptr[steps - 1] = saturate_to_int<int8_t>(end);
            break;
        }
        case DType::Int16: {
            int16_t* ptr = static_cast<int16_t*>(data);
            for (int64_t i = 0; i < steps; ++i) {
                ptr[i] = saturate_to_int<int16_t>(start + i * step_size);
            }
            if (steps > 1) ptr[steps - 1] = saturate_to_int<int16_t>(end);
            break;
        }
        case DType::Int32: {
            int32_t* ptr = static_cast<int32_t*>(data);
            for (int64_t i = 0; i < steps; ++i) {
                ptr[i] = saturate_to_int<int32_t>(start + i * step_size);
            }
            if (steps > 1) ptr[steps - 1] = saturate_to_int<int32_t>(end);
            break;
        }
        case DType::Int64: {
            int64_t* ptr = static_cast<int64_t*>(data);
            for (int64_t i = 0; i < steps; ++i) {
                ptr[i] = saturate_to_int<int64_t>(start + i * step_size);
            }
            if (steps > 1) ptr[steps - 1] = saturate_to_int<int64_t>(end);
            break;
        }
        case DType::UInt8: {
            uint8_t* ptr = static_cast<uint8_t*>(data);
            for (int64_t i = 0; i < steps; ++i) {
                ptr[i] = saturate_to_int<uint8_t>(start + i * step_size);
            }
            if (steps > 1) ptr[steps - 1] = saturate_to_int<uint8_t>(end);
            break;
        }
        case DType::UInt16: {
            uint16_t* ptr = static_cast<uint16_t*>(data);
            for (int64_t i = 0; i < steps; ++i) {
                ptr[i] = saturate_to_int<uint16_t>(start + i * step_size);
            }
            if (steps > 1) ptr[steps - 1] = saturate_to_int<uint16_t>(end);
            break;
        }
        case DType::UInt32: {
            uint32_t* ptr = static_cast<uint32_t*>(data);
            for (int64_t i = 0; i < steps; ++i) {
                ptr[i] = saturate_to_int<uint32_t>(start + i * step_size);
            }
            if (steps > 1) ptr[steps - 1] = saturate_to_int<uint32_t>(end);
            break;
        }
        case DType::UInt64: {
            uint64_t* ptr = static_cast<uint64_t*>(data);
            for (int64_t i = 0; i < steps; ++i) {
                ptr[i] = saturate_to_int<uint64_t>(start + i * step_size);
            }
            if (steps > 1) ptr[steps - 1] = saturate_to_int<uint64_t>(end);
            break;
        }
        case DType::Complex64: {
            auto* ptr = static_cast<std::complex<float>*>(data);
            for (int64_t i = 0; i < steps; ++i) {
                ptr[i] = std::complex<float>(static_cast<float>(start + i * step_size), 0.0f);
            }
            if (steps > 1) ptr[steps - 1] = std::complex<float>(static_cast<float>(end), 0.0f);
            break;
        }
        case DType::Complex128: {
            auto* ptr = static_cast<std::complex<double>*>(data);
            for (int64_t i = 0; i < steps; ++i) {
                ptr[i] = std::complex<double>(start + i * step_size, 0.0);
            }
            if (steps > 1) ptr[steps - 1] = std::complex<double>(end, 0.0);
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for linspace()");
    }
    return tensor;
}

auto eye(int64_t n, std::optional<int64_t> m, DType dtype, Device device) -> Tensor {
    int64_t cols = m.value_or(n);

    // Use OpId dispatch for non-CPU devices
    if (device.type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::N, n);
        attrs.set(AttrKey::M, cols);
        attrs.set(AttrKey::Dtype, dtype_to_string(dtype));
        attrs.set(AttrKey::Device, static_cast<int64_t>(device.index));

        return dispatch_to_device(OpId::Eye, device.type, {}, attrs)[0];
    }

    // Start with zeros (now safe since we're on CPU)
    auto tensor = zeros({n, cols}, dtype, device);
    if (!tensor.impl() || !tensor.storage()) return tensor;

    void* data = tensor.storage()->data();
    int64_t min_dim = std::min(n, cols);

    // Set diagonal to ones based on dtype
    switch (dtype) {
        case DType::Float32: {
            float* ptr = static_cast<float*>(data);
            for (int64_t i = 0; i < min_dim; ++i) {
                ptr[i * cols + i] = 1.0f;
            }
            break;
        }
        case DType::Float64: {
            double* ptr = static_cast<double*>(data);
            for (int64_t i = 0; i < min_dim; ++i) {
                ptr[i * cols + i] = 1.0;
            }
            break;
        }
        case DType::Int32: {
            int32_t* ptr = static_cast<int32_t*>(data);
            for (int64_t i = 0; i < min_dim; ++i) {
                ptr[i * cols + i] = 1;
            }
            break;
        }
        case DType::Int64: {
            int64_t* ptr = static_cast<int64_t*>(data);
            for (int64_t i = 0; i < min_dim; ++i) {
                ptr[i * cols + i] = 1;
            }
            break;
        }
        case DType::UInt8: {
            uint8_t* ptr = static_cast<uint8_t*>(data);
            for (int64_t i = 0; i < min_dim; ++i) {
                ptr[i * cols + i] = 1;
            }
            break;
        }
        case DType::Bool: {
            bool* ptr = static_cast<bool*>(data);
            for (int64_t i = 0; i < min_dim; ++i) {
                ptr[i * cols + i] = true;
            }
            break;
        }
        case DType::Float16: {
            Float16* ptr = static_cast<Float16*>(data);
            Float16 one(1.0f);
            for (int64_t i = 0; i < min_dim; ++i) {
                ptr[i * cols + i] = one;
            }
            break;
        }
        case DType::Int8: {
            int8_t* ptr = static_cast<int8_t*>(data);
            for (int64_t i = 0; i < min_dim; ++i) {
                ptr[i * cols + i] = 1;
            }
            break;
        }
        case DType::Int16: {
            int16_t* ptr = static_cast<int16_t*>(data);
            for (int64_t i = 0; i < min_dim; ++i) ptr[i * cols + i] = 1;
            break;
        }
        case DType::UInt16: {
            uint16_t* ptr = static_cast<uint16_t*>(data);
            for (int64_t i = 0; i < min_dim; ++i) ptr[i * cols + i] = 1;
            break;
        }
        case DType::UInt32: {
            uint32_t* ptr = static_cast<uint32_t*>(data);
            for (int64_t i = 0; i < min_dim; ++i) ptr[i * cols + i] = 1;
            break;
        }
        case DType::UInt64: {
            uint64_t* ptr = static_cast<uint64_t*>(data);
            for (int64_t i = 0; i < min_dim; ++i) ptr[i * cols + i] = 1;
            break;
        }
        case DType::BFloat16: {
            BFloat16* ptr = static_cast<BFloat16*>(data);
            BFloat16 one(1.0f);
            for (int64_t i = 0; i < min_dim; ++i) ptr[i * cols + i] = one;
            break;
        }
        case DType::Complex64: {
            auto* ptr = static_cast<std::complex<float>*>(data);
            for (int64_t i = 0; i < min_dim; ++i) ptr[i * cols + i] = std::complex<float>(1.0f, 0.0f);
            break;
        }
        case DType::Complex128: {
            auto* ptr = static_cast<std::complex<double>*>(data);
            for (int64_t i = 0; i < min_dim; ++i) ptr[i * cols + i] = std::complex<double>(1.0, 0.0);
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for eye()");
    }
    return tensor;
}

auto zeros_like(const Tensor& tensor) -> Tensor {
    std::vector<int64_t> shape(tensor.shape().begin(), tensor.shape().end());
    return zeros(shape, tensor.dtype(), tensor.device());
}

auto ones_like(const Tensor& tensor) -> Tensor {
    std::vector<int64_t> shape(tensor.shape().begin(), tensor.shape().end());
    return ones(shape, tensor.dtype(), tensor.device());
}

auto rand_like(const Tensor& tensor) -> Tensor {
    std::vector<int64_t> shape(tensor.shape().begin(), tensor.shape().end());
    return rand(shape, tensor.dtype(), tensor.device());
}

auto randn_like(const Tensor& tensor) -> Tensor {
    std::vector<int64_t> shape(tensor.shape().begin(), tensor.shape().end());
    return randn(shape, tensor.dtype(), tensor.device());
}

auto randperm(int64_t n, Device device) -> Tensor {
    if (device.type == Device::Type::CPU) {
        // CPU: sequential range shuffled in place with the global RNG.
        // Use double endpoints: a float endpoint cannot represent every
        // integer beyond 2^24, which would yield a permutation with
        // duplicate/missing indices for large n (the generator overload
        // already uses double).
        auto tensor = arange(0.0, static_cast<double>(n), 1.0, DType::Int64, device);
        if (n > 1) {
            auto data = tensor.data<int64_t>();
            auto& gen = get_rng();
            std::shuffle(data, data + n, gen);
        }
        return tensor;
    }

    // GPU: a uniformly random permutation of [0, n) is the argsort of n uniform
    // random keys. Both the key generation (rand) and the sort (argsort) run on
    // the target device — no host shuffle / CPU fallback, and never the silent
    // unshuffled identity the old guard returned for non-CPU devices.
    if (n <= 1) {
        return arange(0.0, static_cast<double>(n), 1.0, DType::Int64, device);
    }
    // Use Float64 keys: Float32's 24-bit mantissa gives only ~2^24 distinct
    // values in [0,1), so key collisions (broken by argsort's stable index
    // tie-break, which biases toward the identity order) become common well
    // before n reaches that range. Float64's 53-bit mantissa pushes the
    // collision floor (~2^53 distinct values) past any practical n, keeping the
    // permutation effectively uniform while staying fully on-device.
    Tensor keys = rand({n}, DType::Float64, device);
    return argsort(keys, /*dim=*/0, /*descending=*/false).to(DType::Int64);
}

auto meshgrid(const std::vector<Tensor>& tensors, const std::string& indexing) -> std::vector<Tensor> {
    if (tensors.empty()) return {};
    if (indexing != "ij" && indexing != "xy")
        throw std::invalid_argument("meshgrid: indexing must be 'ij' or 'xy'");

    size_t ndim = tensors.size();
    for (size_t i = 0; i < ndim; ++i) {
        if (tensors[i].ndim() != 1)
            throw std::invalid_argument("meshgrid: all input tensors must be 1-D");
    }

    // Output shape: one axis per input (length = that input's numel). For "xy"
    // indexing with >= 2 inputs the first two axes are swapped.
    std::vector<int64_t> out_shape;
    out_shape.reserve(ndim);
    for (size_t i = 0; i < ndim; ++i) out_shape.push_back(tensors[i].numel());
    if (indexing == "xy" && ndim >= 2) std::swap(out_shape[0], out_shape[1]);

    std::vector<Tensor> result;
    result.reserve(ndim);
    for (size_t ti = 0; ti < ndim; ++ti) {
        // Output axis along which this tensor's values vary.
        size_t dim_idx = ti;
        if (indexing == "xy" && ndim >= 2) {
            if (ti == 0) dim_idx = 1;
            else if (ti == 1) dim_idx = 0;
        }

        // Reshape the 1-D input to place its values on `dim_idx` (size 1 on
        // every other axis), then broadcast to the full grid shape. reshape and
        // expand both dispatch to the input's device, so meshgrid now works on
        // every backend. (The previous host std::memcpy loop dereferenced
        // grid.data_ptr()/src.data_ptr() directly — device pointers for a GPU
        // tensor — and crashed / produced garbage off-CPU.)
        std::vector<int64_t> view_shape(ndim, 1);
        view_shape[dim_idx] = tensors[ti].numel();
        Tensor reshaped = reshape(tensors[ti], view_shape);
        result.push_back(expand(reshaped, out_shape).contiguous());
    }

    return result;
}

auto multinomial(const Tensor& input, int64_t num_samples, bool replacement) -> Tensor {
    if (input.ndim() < 1 || input.ndim() > 2) {
        throw std::invalid_argument("multinomial: input must be 1D or 2D");
    }
    if (num_samples <= 0) {
        throw std::invalid_argument("multinomial: num_samples must be positive");
    }

    auto inp = input.contiguous();
    std::array<Tensor, 1> inputs = {inp};
    NewOpAttributes attrs;
    attrs.set(AttrKey::NumSamples, num_samples);
    attrs.set(AttrKey::Replacement, replacement);
    return dispatch<OpId::Multinomial>(inputs, attrs)[0];
}

auto bernoulli(const Tensor& probs) -> Tensor {
    auto inp = probs.contiguous();
    std::array<Tensor, 1> inputs = {inp};
    return dispatch<OpId::Bernoulli>(inputs)[0];
}

auto normal(const Tensor& mean, const Tensor& std) -> Tensor {
    auto m = mean.contiguous();
    auto s = std.contiguous();
    std::array<Tensor, 2> inputs = {m, s};
    return dispatch<OpId::NormalSample>(inputs)[0];
}

auto poisson(const Tensor& rates) -> Tensor {
    // Dispatch to backend-specific kernel for proper Poisson sampling
    auto inp = rates.contiguous();
    std::array<Tensor, 1> inputs = {inp};
    return dispatch<OpId::PoissonSample>(inputs)[0];
}

auto exponential(const Tensor& rate) -> Tensor {
    auto r = rate.contiguous();
    std::array<Tensor, 1> inputs = {r};
    return dispatch<OpId::ExponentialSample>(inputs)[0];
}

auto gamma_sample(const Tensor& concentration, const Tensor& rate) -> Tensor {
    // Native gamma(concentration=alpha, rate=beta) sampler. Concentration and
    // rate must broadcast to a common shape; we broadcast here so every backend
    // kernel can assume equal element-aligned inputs.
    auto bshape = broadcast_shapes(concentration.shape(), rate.shape());
    auto a = broadcast_to(concentration, bshape).contiguous();
    auto b = broadcast_to(rate, bshape).contiguous();
    std::array<Tensor, 2> inputs = {a, b};
    return dispatch<OpId::GammaSample>(inputs)[0];
}

// ============================================================================
// Phase 11: New distribution samplers — delegate to C++ distribution classes
// ============================================================================

auto weibull(const Tensor& scale, const Tensor& concentration,
             std::vector<int64_t> shape) -> Tensor {
    distributions::Weibull dist(scale.contiguous(), concentration.contiguous());
    return dist.sample(std::move(shape));
}

auto laplace(const Tensor& loc, const Tensor& scale,
             std::vector<int64_t> shape) -> Tensor {
    distributions::Laplace dist(loc.contiguous(), scale.contiguous());
    return dist.sample(std::move(shape));
}

auto dirichlet(const Tensor& concentration,
               std::vector<int64_t> shape) -> Tensor {
    distributions::Dirichlet dist(concentration.contiguous());
    return dist.sample(std::move(shape));
}

auto half_normal(const Tensor& scale,
                 std::vector<int64_t> shape) -> Tensor {
    distributions::HalfNormal dist(scale.contiguous());
    return dist.sample(std::move(shape));
}

auto von_mises(const Tensor& loc, const Tensor& concentration,
               std::vector<int64_t> shape) -> Tensor {
    distributions::VonMises dist(loc.contiguous(), concentration.contiguous());
    return dist.sample(std::move(shape));
}

auto student_t(const Tensor& df, const Tensor& loc, const Tensor& scale,
               std::vector<int64_t> shape) -> Tensor {
    distributions::StudentT dist(df.contiguous(), loc.contiguous(), scale.contiguous());
    return dist.sample(std::move(shape));
}

auto negative_binomial(const Tensor& total_count, const Tensor& probs,
                       std::vector<int64_t> shape) -> Tensor {
    distributions::NegativeBinomial dist(total_count.contiguous(), probs.contiguous());
    return dist.sample(std::move(shape));
}

auto binomial(int64_t total_count, const Tensor& probs,
              std::vector<int64_t> shape) -> Tensor {
    distributions::Binomial dist(total_count, probs.contiguous());
    return dist.sample(std::move(shape));
}

// ============================================================================
// Generator-aware overloads
// ============================================================================

auto rand(std::vector<int64_t> shape, DType dtype, Device device,
         Generator& generator) -> Tensor {
    if (device.type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::Shape, shape_to_string(shape));
        attrs.set(AttrKey::Dtype, dtype_to_string(dtype));
        attrs.set(AttrKey::Device, static_cast<int64_t>(device.index));
        attrs.set(AttrKey::Seed, static_cast<int64_t>(generator.next_seed()));
        return dispatch_to_device(OpId::Rand, device.type, {}, attrs)[0];
    }

    auto tensor = Tensor::empty_uninitialized(std::move(shape), dtype, device);
    if (!tensor.impl() || !tensor.storage()) return tensor;

    size_t numel = tensor.numel();
    void* data = tensor.storage()->data();
    auto& eng = generator.engine();

    switch (dtype) {
        case DType::Float32: {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            auto* ptr = static_cast<float*>(data);
            for (size_t i = 0; i < numel; ++i) ptr[i] = dist(eng);
            break;
        }
        case DType::Float64: {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            auto* ptr = static_cast<double*>(data);
            for (size_t i = 0; i < numel; ++i) ptr[i] = dist(eng);
            break;
        }
        case DType::Float16: {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            auto* ptr = static_cast<Float16*>(data);
            for (size_t i = 0; i < numel; ++i) ptr[i] = Float16(dist(eng));
            break;
        }
        case DType::BFloat16: {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            auto* ptr = static_cast<BFloat16*>(data);
            for (size_t i = 0; i < numel; ++i) ptr[i] = BFloat16(dist(eng));
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for rand() with Generator");
    }
    return tensor;
}

auto randn(std::vector<int64_t> shape, DType dtype, Device device,
          Generator& generator) -> Tensor {
    if (device.type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::Shape, shape_to_string(shape));
        attrs.set(AttrKey::Dtype, dtype_to_string(dtype));
        attrs.set(AttrKey::Device, static_cast<int64_t>(device.index));
        attrs.set(AttrKey::Seed, static_cast<int64_t>(generator.next_seed()));
        return dispatch_to_device(OpId::Randn, device.type, {}, attrs)[0];
    }

    auto tensor = Tensor::empty_uninitialized(std::move(shape), dtype, device);
    if (!tensor.impl() || !tensor.storage()) return tensor;

    size_t numel = tensor.numel();
    void* data = tensor.storage()->data();
    auto& eng = generator.engine();

    switch (dtype) {
        case DType::Float32: {
            std::normal_distribution<float> dist(0.0f, 1.0f);
            auto* ptr = static_cast<float*>(data);
            for (size_t i = 0; i < numel; ++i) ptr[i] = dist(eng);
            break;
        }
        case DType::Float64: {
            std::normal_distribution<double> dist(0.0, 1.0);
            auto* ptr = static_cast<double*>(data);
            for (size_t i = 0; i < numel; ++i) ptr[i] = dist(eng);
            break;
        }
        case DType::Float16: {
            std::normal_distribution<float> dist(0.0f, 1.0f);
            auto* ptr = static_cast<Float16*>(data);
            for (size_t i = 0; i < numel; ++i) ptr[i] = Float16(dist(eng));
            break;
        }
        case DType::BFloat16: {
            std::normal_distribution<float> dist(0.0f, 1.0f);
            auto* ptr = static_cast<BFloat16*>(data);
            for (size_t i = 0; i < numel; ++i) ptr[i] = BFloat16(dist(eng));
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for randn() with Generator");
    }
    return tensor;
}

auto randint(int64_t low, int64_t high, std::vector<int64_t> shape,
            DType dtype, Device device, Generator& generator) -> Tensor {
    if (low >= high) {
        throw std::invalid_argument("randint: low must be less than high");
    }

    if (device.type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::Shape, shape_to_string(shape));
        attrs.set(AttrKey::Dtype, dtype_to_string(dtype));
        attrs.set(AttrKey::Device, static_cast<int64_t>(device.index));
        attrs.set(AttrKey::Start, low);
        attrs.set(AttrKey::End, high);
        attrs.set(AttrKey::Seed, static_cast<int64_t>(generator.next_seed()));
        return dispatch_to_device(OpId::Randint, device.type, {}, attrs)[0];
    }

    auto tensor = Tensor::empty_uninitialized(std::move(shape), dtype, device);
    if (!tensor.impl() || !tensor.storage()) return tensor;

    size_t numel = tensor.numel();
    void* data = tensor.storage()->data();
    auto& eng = generator.engine();
    // Reject ranges that do not fit the requested integer dtype; otherwise the
    // per-dtype static_cast below would silently wrap (PyTorch raises here).
    {
        const int64_t hi_incl = high - 1;
        bool fits = true;
        switch (dtype) {
            case DType::Int64: break;
            case DType::Int32: fits = (low >= -2147483648LL && hi_incl <= 2147483647LL); break;
            case DType::Int16: fits = (low >= -32768 && hi_incl <= 32767); break;
            case DType::Int8:  fits = (low >= -128 && hi_incl <= 127); break;
            case DType::UInt8: fits = (low >= 0 && hi_incl <= 255); break;
            default: break;  // unsupported dtype reported by the switch below
        }
        if (!fits) {
            throw std::out_of_range(
                "randint: range [" + std::to_string(low) + ", " + std::to_string(high) +
                ") does not fit in the requested integer dtype");
        }
    }
    std::uniform_int_distribution<int64_t> dist(low, high - 1);

    switch (dtype) {
        case DType::Int32: {
            auto* ptr = static_cast<int32_t*>(data);
            for (size_t i = 0; i < numel; ++i) ptr[i] = static_cast<int32_t>(dist(eng));
            break;
        }
        case DType::Int64: {
            auto* ptr = static_cast<int64_t*>(data);
            for (size_t i = 0; i < numel; ++i) ptr[i] = dist(eng);
            break;
        }
        case DType::Int16: {
            auto* ptr = static_cast<int16_t*>(data);
            for (size_t i = 0; i < numel; ++i) ptr[i] = static_cast<int16_t>(dist(eng));
            break;
        }
        case DType::Int8: {
            auto* ptr = static_cast<int8_t*>(data);
            for (size_t i = 0; i < numel; ++i) ptr[i] = static_cast<int8_t>(dist(eng));
            break;
        }
        case DType::UInt8: {
            auto* ptr = static_cast<uint8_t*>(data);
            for (size_t i = 0; i < numel; ++i) ptr[i] = static_cast<uint8_t>(dist(eng));
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for randint() with Generator");
    }
    return tensor;
}

auto multinomial(const Tensor& input, int64_t num_samples, bool replacement,
                Generator& generator) -> Tensor {
    if (input.ndim() < 1 || input.ndim() > 2) {
        throw std::invalid_argument("multinomial: input must be 1D or 2D");
    }
    if (num_samples <= 0) {
        throw std::invalid_argument("multinomial: num_samples must be positive");
    }

    auto inp = input.contiguous();
    std::array<Tensor, 1> inputs = {inp};
    NewOpAttributes attrs;
    attrs.set(AttrKey::NumSamples, num_samples);
    attrs.set(AttrKey::Replacement, replacement);
    attrs.set(AttrKey::Seed, static_cast<int64_t>(generator.next_seed()));
    return dispatch<OpId::Multinomial>(inputs, attrs)[0];
}

auto bernoulli(const Tensor& probs, Generator& generator) -> Tensor {
    auto inp = probs.contiguous();
    std::array<Tensor, 1> inputs = {inp};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Seed, static_cast<int64_t>(generator.next_seed()));
    return dispatch<OpId::Bernoulli>(inputs, attrs)[0];
}

auto normal(const Tensor& mean, const Tensor& std, Generator& generator) -> Tensor {
    auto m = mean.contiguous();
    auto s = std.contiguous();
    std::array<Tensor, 2> inputs = {m, s};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Seed, static_cast<int64_t>(generator.next_seed()));
    return dispatch<OpId::NormalSample>(inputs, attrs)[0];
}

auto poisson(const Tensor& rates, Generator& generator) -> Tensor {
    auto inp = rates.contiguous();
    std::array<Tensor, 1> inputs = {inp};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Seed, static_cast<int64_t>(generator.next_seed()));
    return dispatch<OpId::PoissonSample>(inputs, attrs)[0];
}

auto exponential(const Tensor& rate, Generator& generator) -> Tensor {
    auto r = rate.contiguous();
    std::array<Tensor, 1> inputs = {r};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Seed, static_cast<int64_t>(generator.next_seed()));
    return dispatch<OpId::ExponentialSample>(inputs, attrs)[0];
}

auto randperm(int64_t n, Device device, Generator& generator) -> Tensor {
    if (device.type == Device::Type::CPU) {
        // CPU: Fisher-Yates shuffle driven by the supplied Generator's engine.
        auto result = tenzor::arange(0.0, static_cast<double>(n), 1.0, DType::Int64, device);
        if (n > 1) {
            auto& eng = generator.engine();
            auto* data = result.data<int64_t>();
            for (int64_t i = n - 1; i > 0; --i) {
                std::uniform_int_distribution<int64_t> dist(0, i);
                int64_t j = dist(eng);
                std::swap(data[i], data[j]);
            }
        }
        return result;
    }

    // GPU: generator-seeded uniform keys + on-device argsort. The Generator
    // overload of rand() forwards generator.next_seed() to the device kernel,
    // so this is reproducible and stays on-device (never the silent unshuffled
    // identity the old CPU-only guard returned for non-CPU devices).
    if (n <= 1) {
        return tenzor::arange(0.0, static_cast<double>(n), 1.0, DType::Int64, device);
    }
    // Float64 keys (53-bit mantissa, ~2^53 distinct values) avoid the frequent
    // collisions Float32's 24-bit mantissa would produce; collisions are broken
    // by argsort's stable index tie-break and would otherwise bias the
    // permutation toward the identity order. See the plain overload above.
    Tensor keys = rand({n}, DType::Float64, device, generator);
    return argsort(keys, /*dim=*/0, /*descending=*/false).to(DType::Int64);
}

auto logspace(double start, double end, int64_t steps, double base,
              DType dtype, Device device) -> Tensor {
    // Carry the intermediates (linspace, log(base), exp) in a floating compute
    // dtype so integer/half outputs are not corrupted: an integer dtype would
    // truncate the exponents and log(base) before exp() (and exp() on an
    // integer tensor is unsupported). Only cast to the requested dtype at the
    // end.
    DType compute = (dtype == DType::Float64) ? DType::Float64 : DType::Float32;
    auto exponents = tenzor::linspace(start, end, steps, compute, device);
    // base^exponents = exp(log(base) * exponents)
    double log_base = std::log(base);
    auto scaled = tenzor::mul(exponents, tenzor::full({1}, log_base, compute, device));
    auto result = tenzor::exp(scaled);
    return (dtype != compute) ? result.to(dtype) : result;
}

// =========================================================================
// New creation operations for PyTorch parity (compositions)
// =========================================================================

auto full_like(const Tensor& tensor, double fill_value) -> Tensor {
    std::vector<int64_t> shape(tensor.shape().begin(), tensor.shape().end());
    return full(shape, fill_value, tensor.dtype(), tensor.device());
}

auto empty_like(const Tensor& tensor) -> Tensor {
    std::vector<int64_t> shape(tensor.shape().begin(), tensor.shape().end());
    return empty(shape, tensor.dtype(), tensor.device());
}

auto randint_like(const Tensor& tensor, int64_t low, int64_t high) -> Tensor {
    std::vector<int64_t> shape(tensor.shape().begin(), tensor.shape().end());
    return randint(low, high, shape, tensor.dtype(), tensor.device());
}

auto tril_indices(int64_t row, int64_t col, int64_t offset,
                  DType dtype, Device device) -> Tensor {
    NewOpAttributes attrs;
    attrs.set(AttrKey::M, row);
    attrs.set(AttrKey::N, col);
    attrs.set(AttrKey::Diagonal, offset);
    std::vector<Tensor> inputs;  // no input tensors
    // Kernels emit Int64; honor the caller's requested index dtype (PyTorch
    // defaults to Int64 but allows other integer types).
    auto result = dispatch_to_device(OpId::TrilIndices, device.type, inputs, attrs)[0];
    if (result.dtype() != dtype) result = result.to(dtype);
    return result;
}

auto triu_indices(int64_t row, int64_t col, int64_t offset,
                  DType dtype, Device device) -> Tensor {
    NewOpAttributes attrs;
    attrs.set(AttrKey::M, row);
    attrs.set(AttrKey::N, col);
    attrs.set(AttrKey::Diagonal, offset);
    std::vector<Tensor> inputs;  // no input tensors
    auto result = dispatch_to_device(OpId::TriuIndices, device.type, inputs, attrs)[0];
    if (result.dtype() != dtype) result = result.to(dtype);
    return result;
}

auto complex(const Tensor& real, const Tensor& imag) -> Tensor {
    // Validate inputs
    auto real_shape = real.shape();
    auto imag_shape = imag.shape();
    if (real_shape.size() != imag_shape.size()) {
        throw std::invalid_argument("complex: real and imag must have the same number of dimensions");
    }
    for (size_t i = 0; i < real_shape.size(); ++i) {
        if (real_shape[i] != imag_shape[i]) {
            throw std::invalid_argument("complex: real and imag must have the same shape");
        }
    }
    if (real.device().type != imag.device().type) {
        throw std::invalid_argument("complex: real and imag must be on the same device");
    }

    // Dispatch through backend for GPU tensors (backends already cover both
    // Float32→Complex64 and Float64→Complex128 paths). Contiguify first: the
    // GPU ComplexTensor kernels index real/imag flat, so a non-contiguous
    // (e.g. transposed) input would be read with the wrong layout — the CPU
    // path below already contiguifies.
    if (real.device().type != Device::Type::CPU) {
        Tensor r = real.is_contiguous() ? real : real.contiguous();
        Tensor im = imag.is_contiguous() ? imag : imag.contiguous();
        std::array<Tensor, 2> inputs = {r, im};
        return dispatch_single(OpId::ComplexTensor, inputs);
    }

    // CPU path: decide output dtype from the inputs.  Float64 inputs build a
    // Complex128 tensor (required by audit item A.10 so the EigBackward
    // complex pullback retains Float64 precision).  Everything else widens
    // through Float32 → Complex64.
    const bool use_double =
        (real.dtype() == DType::Float64) || (imag.dtype() == DType::Float64);

    std::vector<int64_t> shape_vec(real_shape.begin(), real_shape.end());

    if (use_double) {
        auto r = real.to(DType::Float64).contiguous();
        auto im = imag.to(DType::Float64).contiguous();
        auto result = empty(shape_vec, DType::Complex128, Device::cpu());

        const double* r_data = r.data<double>();
        const double* i_data = im.data<double>();
        // Write through data<>() which already applies the tensor's offset,
        // consistent with the offset-applied reads above (the previous
        // raw-storage + manual offset write was fragile if empty() ever
        // returned a non-zero-offset view).
        auto* c_data = result.data<std::complex<double>>();

        int64_t numel = r.numel();
        for (int64_t idx = 0; idx < numel; ++idx) {
            c_data[idx] = std::complex<double>(r_data[idx], i_data[idx]);
        }
        return result;
    }

    // Float32 (and any lower-precision dtype) → Complex64.
    auto r = real.to(DType::Float32).contiguous();
    auto im = imag.to(DType::Float32).contiguous();
    auto result = empty(shape_vec, DType::Complex64, Device::cpu());

    const float* r_data = r.data<float>();
    const float* i_data = im.data<float>();
    // Write through data<>() which already applies the tensor's offset,
    // consistent with the offset-applied reads above.
    auto* c_data = result.data<std::complex<float>>();

    int64_t numel = r.numel();
    for (int64_t idx = 0; idx < numel; ++idx) {
        c_data[idx] = std::complex<float>(r_data[idx], i_data[idx]);
    }

    return result;
}

} // namespace tenzor
