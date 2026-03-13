#include "tenzor/core/tensor.hpp"
#include "tenzor/core/shape.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <stdexcept>

#ifdef TENZOR_HAS_ONEMKL
#include <oneapi/mkl.hpp>
#endif

// Forward declaration for broadcasting fallback
namespace tenzor {
    auto add(const Tensor& a, const Tensor& b) -> Tensor;
    auto sub(const Tensor& a, const Tensor& b) -> Tensor;
    auto mul(const Tensor& a, const Tensor& b) -> Tensor;
    auto div(const Tensor& a, const Tensor& b) -> Tensor;
}

// Forward declaration for contiguous kernel
namespace tenzor {
namespace oneapi {
    auto contiguous_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
}
}

namespace tenzor {
namespace oneapi {

// SYCL Kernel name classes (using functors for SYCL 2025.2 compatibility)
struct AddKernelFloat32 {};
struct AddKernelFloat64 {};
struct AddKernelFloat16 {};
struct AddKernelInt8 {};
struct AddKernelInt32 {};
struct AddKernelInt64 {};
struct AddKernelUInt8 {};
struct AddKernelBool {};
struct SubKernelFloat32 {};
struct SubKernelFloat64 {};
struct SubKernelFloat16 {};
struct SubKernelInt8 {};
struct SubKernelInt32 {};
struct SubKernelInt64 {};
struct SubKernelUInt8 {};
struct MulKernelFloat32 {};
struct MulKernelFloat64 {};
struct MulKernelFloat16 {};
struct MulKernelInt32 {};
struct MulKernelBool {};
struct DivKernelFloat32 {};
struct DivKernelFloat64 {};
struct DivKernelFloat16 {};
struct DivKernelInt32 {};
struct CheckZerosKernel {};
struct MatMulKernelFloat32 {};
struct MatMulKernelFloat64 {};
struct MatMulKernelFloat16 {};
struct MatMulKernelInt32 {};
struct BmmKernelFloat32 {};
struct BmmKernelFloat64 {};
struct BmmKernelFloat16 {};
struct SqrtKernelFloat32 {};
struct SqrtKernelFloat64 {};
struct SqrtKernelFloat16 {};
struct NegKernelFloat32 {};
struct NegKernelFloat64 {};
struct NegKernelFloat16 {};
struct NegKernelInt32 {};
struct AbsKernelFloat32 {};
struct AbsKernelFloat64 {};
struct AbsKernelFloat16 {};
struct AbsKernelInt32 {};
struct LogKernelFloat32 {};
struct LogKernelFloat64 {};
struct LogKernelFloat16 {};
struct ExpKernelFloat32 {};
struct ExpKernelFloat64 {};
struct ExpKernelFloat16 {};
struct PowKernelFloat32 {};
struct PowKernelFloat64 {};
struct PowKernelFloat16 {};

// Trigonometric kernel name classes
struct SinKernelFloat32 {};
struct SinKernelFloat64 {};
struct SinKernelFloat16 {};
struct SinKernelBFloat16 {};
struct CosKernelFloat32 {};
struct CosKernelFloat64 {};
struct CosKernelFloat16 {};
struct CosKernelBFloat16 {};
struct TanKernelFloat32 {};
struct TanKernelFloat64 {};
struct TanKernelFloat16 {};
struct TanKernelBFloat16 {};
struct AsinKernelFloat32 {};
struct AsinKernelFloat64 {};
struct AsinKernelFloat16 {};
struct AsinKernelBFloat16 {};
struct AcosKernelFloat32 {};
struct AcosKernelFloat64 {};
struct AcosKernelFloat16 {};
struct AcosKernelBFloat16 {};
struct AtanKernelFloat32 {};
struct AtanKernelFloat64 {};
struct AtanKernelFloat16 {};
struct AtanKernelBFloat16 {};
struct SinhKernelFloat32 {};
struct SinhKernelFloat64 {};
struct SinhKernelFloat16 {};
struct SinhKernelBFloat16 {};
struct CoshKernelFloat32 {};
struct CoshKernelFloat64 {};
struct CoshKernelFloat16 {};
struct CoshKernelBFloat16 {};
struct Atan2KernelFloat32 {};
struct Atan2KernelFloat64 {};
struct Atan2KernelFloat16 {};
struct Atan2KernelBFloat16 {};

// Extended math kernel name classes
struct Log2KernelFloat32 {};
struct Log2KernelFloat64 {};
struct Log2KernelFloat16 {};
struct Log2KernelBFloat16 {};
struct Log10KernelFloat32 {};
struct Log10KernelFloat64 {};
struct Log10KernelFloat16 {};
struct Log10KernelBFloat16 {};
struct Log1pKernelFloat32 {};
struct Log1pKernelFloat64 {};
struct Log1pKernelFloat16 {};
struct Log1pKernelBFloat16 {};
struct Exp2KernelFloat32 {};
struct Exp2KernelFloat64 {};
struct Exp2KernelFloat16 {};
struct Exp2KernelBFloat16 {};
struct Expm1KernelFloat32 {};
struct Expm1KernelFloat64 {};
struct Expm1KernelFloat16 {};
struct Expm1KernelBFloat16 {};
struct ErfKernelFloat32 {};
struct ErfKernelFloat64 {};
struct ErfKernelFloat16 {};
struct ErfKernelBFloat16 {};
struct ErfcKernelFloat32 {};
struct ErfcKernelFloat64 {};
struct ErfcKernelFloat16 {};
struct ErfcKernelBFloat16 {};

// Bool predicate kernel name classes
struct IsNanKernelFloat32 {};
struct IsNanKernelFloat64 {};
struct IsNanKernelFloat16 {};
struct IsNanKernelBFloat16 {};
struct IsInfKernelFloat32 {};
struct IsInfKernelFloat64 {};
struct IsInfKernelFloat16 {};
struct IsInfKernelBFloat16 {};
struct IsFiniteKernelFloat32 {};
struct IsFiniteKernelFloat64 {};
struct IsFiniteKernelFloat16 {};
struct IsFiniteKernelBFloat16 {};

// Binary math kernel name classes
struct FmodKernelFloat32 {};
struct FmodKernelFloat64 {};
struct FmodKernelFloat16 {};
struct FmodKernelBFloat16 {};
struct RemainderKernelFloat32 {};
struct RemainderKernelFloat64 {};
struct RemainderKernelFloat16 {};
struct RemainderKernelBFloat16 {};

// Ternary kernel name classes
struct LerpKernelFloat32 {};
struct LerpKernelFloat64 {};
struct LerpKernelFloat16 {};
struct LerpKernelBFloat16 {};

// Logical kernel name classes
struct LogicalAndKernelFloat32 {};
struct LogicalAndKernelFloat64 {};
struct LogicalAndKernelFloat16 {};
struct LogicalAndKernelBFloat16 {};
struct LogicalAndKernelInt32 {};
struct LogicalAndKernelInt64 {};
struct LogicalOrKernelFloat32 {};
struct LogicalOrKernelFloat64 {};
struct LogicalOrKernelFloat16 {};
struct LogicalOrKernelBFloat16 {};
struct LogicalOrKernelInt32 {};
struct LogicalOrKernelInt64 {};
struct LogicalNotKernelFloat32 {};
struct LogicalNotKernelFloat64 {};
struct LogicalNotKernelFloat16 {};
struct LogicalNotKernelBFloat16 {};
struct LogicalNotKernelInt32 {};
struct LogicalNotKernelInt64 {};
struct LogicalNotKernelBool {};
struct LogicalXorKernelFloat32 {};
struct LogicalXorKernelFloat64 {};
struct LogicalXorKernelFloat16 {};
struct LogicalXorKernelBFloat16 {};
struct LogicalXorKernelInt32 {};
struct LogicalXorKernelInt64 {};

// Element-wise min/max kernel name classes
struct MinimumKernelFloat32 {};
struct MinimumKernelFloat64 {};
struct MinimumKernelFloat16 {};
struct MinimumKernelBFloat16 {};
struct MaximumKernelFloat32 {};
struct MaximumKernelFloat64 {};
struct MaximumKernelFloat16 {};
struct MaximumKernelBFloat16 {};

// Complex number kernel name classes
struct ConjKernelComplex64 {};
struct ConjKernelComplex128 {};
struct RealKernelComplex64 {};
struct RealKernelComplex128 {};
struct ImagKernelComplex64 {};
struct ImagKernelComplex128 {};
struct AngleKernelComplex64 {};
struct AngleKernelComplex128 {};
struct AngleKernelFloat32 {};
struct AngleKernelFloat64 {};
struct PolarKernelFloat32 {};
struct PolarKernelFloat64 {};

// Rounding kernel name classes
struct RoundKernelFloat32 {};
struct RoundKernelFloat64 {};
struct RoundKernelFloat16 {};
struct RoundKernelBFloat16 {};
struct FloorKernelFloat32 {};
struct FloorKernelFloat64 {};
struct FloorKernelFloat16 {};
struct FloorKernelBFloat16 {};
struct CeilKernelFloat32 {};
struct CeilKernelFloat64 {};
struct CeilKernelFloat16 {};
struct CeilKernelBFloat16 {};
struct TruncKernelFloat32 {};
struct TruncKernelFloat64 {};
struct TruncKernelFloat16 {};
struct TruncKernelBFloat16 {};
struct ReciprocalKernelFloat32 {};
struct ReciprocalKernelFloat64 {};
struct ReciprocalKernelFloat16 {};
struct ReciprocalKernelBFloat16 {};

// Utility kernel name classes
struct ClampMinKernelFloat32 {};
struct ClampMinKernelFloat64 {};
struct ClampMinKernelFloat16 {};
struct ClampMinKernelBFloat16 {};
struct ClampMaxKernelFloat32 {};
struct ClampMaxKernelFloat64 {};
struct ClampMaxKernelFloat16 {};
struct ClampMaxKernelBFloat16 {};
struct WhereKernelFloat32 {};
struct WhereKernelFloat64 {};
struct WhereKernelFloat16 {};
struct WhereKernelBFloat16 {};
struct AddKernelBFloat16 {};
struct SubKernelBFloat16 {};
struct MulKernelBFloat16 {};
struct DivKernelBFloat16 {};
struct AddKernelComplex64 {};
struct AddKernelComplex128 {};
struct SubKernelComplex64 {};
struct SubKernelComplex128 {};
struct MulKernelComplex64 {};
struct MulKernelComplex128 {};
struct DivKernelComplex64 {};
struct DivKernelComplex128 {};
struct MatMulKernelBFloat16 {};
struct BmmKernelBFloat16 {};
struct SqrtKernelBFloat16 {};
struct NegKernelBFloat16 {};
struct AbsKernelBFloat16 {};
struct LogKernelBFloat16 {};
struct ExpKernelBFloat16 {};
struct PowKernelBFloat16 {};
struct WhereKernelInt64 {};
struct RepeatKernelFloat32 {};
struct RepeatKernelFloat64 {};
struct RepeatKernelFloat16 {};

// Helper function to get typed pointer from tensor
template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

// BFloat16 <-> Float32 conversion helpers (device-compatible)
inline float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    __builtin_memcpy(&result, &bits, sizeof(float));
    return result;
}

inline uint16_t f32_to_bf16(float f32) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f32, sizeof(uint32_t));
    return static_cast<uint16_t>(bits >> 16);
}

// Helper to calculate total elements
inline auto calculate_numel(const std::vector<int64_t>& shape) -> int64_t {
    int64_t numel = 1;
    for (auto s : shape) {
        numel *= s;
    }
    return numel;
}

// ============================================================================
// SYCL Broadcasting Support
// ============================================================================

// Kernel name structs for broadcast ops
struct BroadcastAddFloat32 {};
struct BroadcastAddFloat64 {};
struct BroadcastAddInt32 {};
struct BroadcastAddInt64 {};
struct BroadcastAddFloat16 {};
struct BroadcastAddBFloat16 {};
struct BroadcastAddInt8 {};
struct BroadcastAddUInt8 {};
struct BroadcastSubFloat32 {};
struct BroadcastSubFloat64 {};
struct BroadcastSubInt32 {};
struct BroadcastSubInt64 {};
struct BroadcastSubFloat16 {};
struct BroadcastSubBFloat16 {};
struct BroadcastSubInt8 {};
struct BroadcastSubUInt8 {};
struct BroadcastMulFloat32 {};
struct BroadcastMulFloat64 {};
struct BroadcastMulInt32 {};
struct BroadcastMulInt64 {};
struct BroadcastMulFloat16 {};
struct BroadcastMulBFloat16 {};
struct BroadcastMulInt8 {};
struct BroadcastMulUInt8 {};
struct BroadcastDivFloat32 {};
struct BroadcastDivFloat64 {};
struct BroadcastDivInt32 {};
struct BroadcastDivInt64 {};
struct BroadcastDivFloat16 {};
struct BroadcastDivBFloat16 {};
struct BroadcastDivInt8 {};
struct BroadcastDivUInt8 {};
struct BroadcastAddComplex64 {};
struct BroadcastAddComplex128 {};
struct BroadcastSubComplex64 {};
struct BroadcastSubComplex128 {};
struct BroadcastMulComplex64 {};
struct BroadcastMulComplex128 {};
struct BroadcastDivComplex64 {};
struct BroadcastDivComplex128 {};

constexpr int MAX_BROADCAST_DIMS = 16;

// Pre-computed broadcast strides for mapping flat output index to input indices.
// stride=0 for broadcast dimensions (input dim == 1).
struct BroadcastInfo {
    int64_t out_strides[MAX_BROADCAST_DIMS];
    int64_t a_strides[MAX_BROADCAST_DIMS];
    int64_t b_strides[MAX_BROADCAST_DIMS];
    int ndim;
    int64_t out_numel;
};

static auto compute_broadcast_info(
    std::span<const int64_t> a_shape,
    std::span<const int64_t> b_shape,
    const std::vector<int64_t>& out_shape) -> BroadcastInfo {

    BroadcastInfo info{};
    info.ndim = static_cast<int>(out_shape.size());
    if (out_shape.size() > MAX_BROADCAST_DIMS)
        throw std::runtime_error("OneAPI: max 16 broadcast dimensions supported");
    info.out_numel = 1;
    for (auto s : out_shape) info.out_numel *= s;

    // Compute output strides (row-major)
    for (int i = info.ndim - 1; i >= 0; --i) {
        info.out_strides[i] = (i == info.ndim - 1) ? 1 : info.out_strides[i + 1] * out_shape[i + 1];
    }

    // Compute input strides with broadcasting (stride=0 for broadcast dims)
    auto compute_input_strides = [&](std::span<const int64_t> shape, int64_t* strides) {
        int offset = info.ndim - static_cast<int>(shape.size());
        // Leading dimensions not present in input → broadcast (stride=0)
        for (int i = 0; i < offset; ++i) strides[i] = 0;
        // Trailing dimensions
        int64_t stride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            strides[i + offset] = (shape[i] == 1) ? 0 : stride;
            stride *= shape[i];
        }
    };

    compute_input_strides(a_shape, info.a_strides);
    compute_input_strides(b_shape, info.b_strides);

    return info;
}

// Generic SYCL broadcast binary operation
template<typename T, typename KernelName, typename Op>
static void sycl_broadcast_binary(
    const Tensor& a, const Tensor& b, Tensor& output,
    const BroadcastInfo& info, sycl::queue& queue, Op op) {

    const T* a_ptr = get_data_ptr<const T>(a);
    const T* b_ptr = get_data_ptr<const T>(b);
    T* out_ptr = get_data_ptr<T>(output);

    // Copy strides to local arrays for SYCL capture
    int ndim = info.ndim;
    int64_t out_numel = info.out_numel;

    // Copy strides to fixed-size arrays for SYCL kernel capture
    int64_t os[MAX_BROADCAST_DIMS], as_[MAX_BROADCAST_DIMS], bs_[MAX_BROADCAST_DIMS];
    for (int i = 0; i < ndim; ++i) {
        os[i] = info.out_strides[i];
        as_[i] = info.a_strides[i];
        bs_[i] = info.b_strides[i];
    }

    queue.parallel_for<KernelName>(sycl::range<1>(out_numel), [=](sycl::id<1> gid) {
        int64_t flat = gid[0];
        int64_t a_idx = 0, b_idx = 0;
        int64_t remaining = flat;
        for (int d = 0; d < ndim; ++d) {
            int64_t coord = remaining / os[d];
            remaining %= os[d];
            a_idx += coord * as_[d];
            b_idx += coord * bs_[d];
        }
        out_ptr[gid] = op(a_ptr[a_idx], b_ptr[b_idx]);
    });
}

// Generic SYCL broadcast binary operation for complex types
// Strides/indices are in complex element units; data is interleaved real/imag pairs
template<typename T, typename KernelName, typename Op>
static void sycl_broadcast_complex_binary(
    const Tensor& a, const Tensor& b, Tensor& output,
    const BroadcastInfo& info, sycl::queue& queue, Op op) {

    const T* a_ptr = get_data_ptr<const T>(a);
    const T* b_ptr = get_data_ptr<const T>(b);
    T* out_ptr = get_data_ptr<T>(output);

    int ndim = info.ndim;
    int64_t out_numel = info.out_numel;

    int64_t os[MAX_BROADCAST_DIMS], as_[MAX_BROADCAST_DIMS], bs_[MAX_BROADCAST_DIMS];
    for (int i = 0; i < ndim; ++i) {
        os[i] = info.out_strides[i];
        as_[i] = info.a_strides[i];
        bs_[i] = info.b_strides[i];
    }

    queue.parallel_for<KernelName>(sycl::range<1>(out_numel), [=](sycl::id<1> gid) {
        int64_t flat = gid[0];
        int64_t a_idx = 0, b_idx = 0;
        int64_t remaining = flat;
        for (int d = 0; d < ndim; ++d) {
            int64_t coord = remaining / os[d];
            remaining %= os[d];
            a_idx += coord * as_[d];
            b_idx += coord * bs_[d];
        }
        // Indices are in complex element units, multiply by 2 for real/imag pair access
        op(a_ptr, b_ptr, out_ptr, a_idx * 2, b_idx * 2, flat * 2);
    });
}

// Element-wise addition kernel
// IMPORTANT: Must ensure contiguous inputs for direct memory access
auto add_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    // Ensure inputs are contiguous for correct memory access
    Tensor a_cont = a.is_contiguous() ? a : contiguous_kernel(a, queue);
    Tensor b_cont = b.is_contiguous() ? b : contiguous_kernel(b, queue);

    auto a_shape = a_cont.shape();
    auto b_shape = b_cont.shape();

    // Check if shapes match exactly
    bool same_shape = std::equal(a_shape.begin(), a_shape.end(), b_shape.begin(), b_shape.end());

    if (a_cont.dtype() != b_cont.dtype()) {
        throw std::invalid_argument("Tensor dtypes must match for addition");
    }

    // Handle broadcasting on device via SYCL kernel
    if (!same_shape) {
        auto out_shape = broadcast_shapes(a_shape, b_shape);
        auto info = compute_broadcast_info(a_shape, b_shape, out_shape);
        Tensor output(out_shape, a_cont.dtype(), a_cont.device());
        if (a_cont.dtype() == DType::Float32) {
            sycl_broadcast_binary<float, BroadcastAddFloat32>(
                a_cont, b_cont, output, info, queue,
                [](float x, float y) { return x + y; });
        } else if (a_cont.dtype() == DType::Float64) {
            sycl_broadcast_binary<double, BroadcastAddFloat64>(
                a_cont, b_cont, output, info, queue,
                [](double x, double y) { return x + y; });
        } else if (a_cont.dtype() == DType::Int32) {
            sycl_broadcast_binary<int32_t, BroadcastAddInt32>(
                a_cont, b_cont, output, info, queue,
                [](int32_t x, int32_t y) { return x + y; });
        } else if (a_cont.dtype() == DType::Int64) {
            sycl_broadcast_binary<int64_t, BroadcastAddInt64>(
                a_cont, b_cont, output, info, queue,
                [](int64_t x, int64_t y) { return x + y; });
        } else if (a_cont.dtype() == DType::Float16) {
            sycl_broadcast_binary<sycl::half, BroadcastAddFloat16>(
                a_cont, b_cont, output, info, queue,
                [](sycl::half x, sycl::half y) { return x + y; });
        } else if (a_cont.dtype() == DType::BFloat16) {
            sycl_broadcast_binary<uint16_t, BroadcastAddBFloat16>(
                a_cont, b_cont, output, info, queue,
                [](uint16_t x, uint16_t y) { return f32_to_bf16(bf16_to_f32(x) + bf16_to_f32(y)); });
        } else if (a_cont.dtype() == DType::Int8) {
            sycl_broadcast_binary<int8_t, BroadcastAddInt8>(
                a_cont, b_cont, output, info, queue,
                [](int8_t x, int8_t y) { return static_cast<int8_t>(x + y); });
        } else if (a_cont.dtype() == DType::UInt8) {
            sycl_broadcast_binary<uint8_t, BroadcastAddUInt8>(
                a_cont, b_cont, output, info, queue,
                [](uint8_t x, uint8_t y) { return static_cast<uint8_t>(x + y); });
        } else if (a_cont.dtype() == DType::Complex64) {
            sycl_broadcast_complex_binary<float, BroadcastAddComplex64>(
                a_cont, b_cont, output, info, queue,
                [](const float* a, const float* b, float* c, int64_t ai, int64_t bi, int64_t ci) {
                    c[ci]     = a[ai]     + b[bi];
                    c[ci + 1] = a[ai + 1] + b[bi + 1];
                });
        } else if (a_cont.dtype() == DType::Complex128) {
            sycl_broadcast_complex_binary<double, BroadcastAddComplex128>(
                a_cont, b_cont, output, info, queue,
                [](const double* a, const double* b, double* c, int64_t ai, int64_t bi, int64_t ci) {
                    c[ci]     = a[ai]     + b[bi];
                    c[ci + 1] = a[ai + 1] + b[bi + 1];
                });
        } else {
            throw std::runtime_error("add broadcast: unsupported dtype");
        }
        return output;
    }

    // Create output tensor
    Tensor output(std::vector<int64_t>(a_cont.shape().begin(), a_cont.shape().end()),
                  a_cont.dtype(), a_cont.device());

    const int64_t numel = a_cont.numel();

    // Dispatch based on dtype
    if (a_cont.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a_cont);
        const float* b_ptr = get_data_ptr<const float>(b_cont);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<AddKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] + b_ptr[idx];
        });
    }
    else if (a_cont.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a_cont);
        const double* b_ptr = get_data_ptr<const double>(b_cont);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<AddKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] + b_ptr[idx];
        });
    }
    else if (a_cont.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a_cont);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<AddKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            // Use float accumulation for precision
            out_ptr[idx] = sycl::half(static_cast<float>(a_ptr[idx]) + static_cast<float>(b_ptr[idx]));
        });
    }
    else if (a_cont.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a_cont);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<AddKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(bf16_to_f32(a_ptr[idx]) + bf16_to_f32(b_ptr[idx]));
        });
    }
    else if (a_cont.dtype() == DType::Int8) {
        const int8_t* a_ptr = get_data_ptr<const int8_t>(a_cont);
        const int8_t* b_ptr = get_data_ptr<const int8_t>(b_cont);
        int8_t* out_ptr = get_data_ptr<int8_t>(output);

        queue.parallel_for<AddKernelInt8>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] + b_ptr[idx];
        });
    }
    else if (a_cont.dtype() == DType::Int32) {
        const int32_t* a_ptr = get_data_ptr<const int32_t>(a_cont);
        const int32_t* b_ptr = get_data_ptr<const int32_t>(b_cont);
        int32_t* out_ptr = get_data_ptr<int32_t>(output);

        queue.parallel_for<AddKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] + b_ptr[idx];
        });
    }
    else if (a_cont.dtype() == DType::Int64) {
        const int64_t* a_ptr = get_data_ptr<const int64_t>(a_cont);
        const int64_t* b_ptr = get_data_ptr<const int64_t>(b_cont);
        int64_t* out_ptr = get_data_ptr<int64_t>(output);

        queue.parallel_for<AddKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] + b_ptr[idx];
        });
    }
    else if (a_cont.dtype() == DType::UInt8) {
        const uint8_t* a_ptr = get_data_ptr<const uint8_t>(a_cont);
        const uint8_t* b_ptr = get_data_ptr<const uint8_t>(b_cont);
        uint8_t* out_ptr = get_data_ptr<uint8_t>(output);

        queue.parallel_for<AddKernelUInt8>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] + b_ptr[idx];
        });
    }
    else if (a_cont.dtype() == DType::Bool) {
        // Bool addition acts as logical OR
        const bool* a_ptr = get_data_ptr<const bool>(a_cont);
        const bool* b_ptr = get_data_ptr<const bool>(b_cont);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<AddKernelBool>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] || b_ptr[idx];
        });
    }
    else if (a_cont.dtype() == DType::Complex64) {
        const float* a_ptr = get_data_ptr<const float>(a_cont);
        const float* b_ptr = get_data_ptr<const float>(b_cont);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<AddKernelComplex64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int64_t base = idx[0] * 2;
            out_ptr[base]     = a_ptr[base]     + b_ptr[base];
            out_ptr[base + 1] = a_ptr[base + 1] + b_ptr[base + 1];
        });
    }
    else if (a_cont.dtype() == DType::Complex128) {
        const double* a_ptr = get_data_ptr<const double>(a_cont);
        const double* b_ptr = get_data_ptr<const double>(b_cont);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<AddKernelComplex128>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int64_t base = idx[0] * 2;
            out_ptr[base]     = a_ptr[base]     + b_ptr[base];
            out_ptr[base + 1] = a_ptr[base + 1] + b_ptr[base + 1];
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for addition");
    }

    return output;
}

// Element-wise subtraction kernel
// IMPORTANT: Must ensure contiguous inputs for direct memory access
auto sub_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    // Ensure inputs are contiguous for correct memory access
    Tensor a_cont = a.is_contiguous() ? a : contiguous_kernel(a, queue);
    Tensor b_cont = b.is_contiguous() ? b : contiguous_kernel(b, queue);

    auto a_shape = a_cont.shape();
    auto b_shape = b_cont.shape();

    // Check if shapes match exactly
    bool same_shape = std::equal(a_shape.begin(), a_shape.end(), b_shape.begin(), b_shape.end());

    if (!same_shape) {
        auto out_shape = broadcast_shapes(a_shape, b_shape);
        auto info = compute_broadcast_info(a_shape, b_shape, out_shape);
        Tensor output(out_shape, a_cont.dtype(), a_cont.device());
        if (a_cont.dtype() == DType::Float32) {
            sycl_broadcast_binary<float, BroadcastSubFloat32>(
                a_cont, b_cont, output, info, queue,
                [](float x, float y) { return x - y; });
        } else if (a_cont.dtype() == DType::Float64) {
            sycl_broadcast_binary<double, BroadcastSubFloat64>(
                a_cont, b_cont, output, info, queue,
                [](double x, double y) { return x - y; });
        } else if (a_cont.dtype() == DType::Int32) {
            sycl_broadcast_binary<int32_t, BroadcastSubInt32>(
                a_cont, b_cont, output, info, queue,
                [](int32_t x, int32_t y) { return x - y; });
        } else if (a_cont.dtype() == DType::Int64) {
            sycl_broadcast_binary<int64_t, BroadcastSubInt64>(
                a_cont, b_cont, output, info, queue,
                [](int64_t x, int64_t y) { return x - y; });
        } else if (a_cont.dtype() == DType::Float16) {
            sycl_broadcast_binary<sycl::half, BroadcastSubFloat16>(
                a_cont, b_cont, output, info, queue,
                [](sycl::half x, sycl::half y) { return x - y; });
        } else if (a_cont.dtype() == DType::BFloat16) {
            sycl_broadcast_binary<uint16_t, BroadcastSubBFloat16>(
                a_cont, b_cont, output, info, queue,
                [](uint16_t x, uint16_t y) { return f32_to_bf16(bf16_to_f32(x) - bf16_to_f32(y)); });
        } else if (a_cont.dtype() == DType::Int8) {
            sycl_broadcast_binary<int8_t, BroadcastSubInt8>(
                a_cont, b_cont, output, info, queue,
                [](int8_t x, int8_t y) { return static_cast<int8_t>(x - y); });
        } else if (a_cont.dtype() == DType::UInt8) {
            sycl_broadcast_binary<uint8_t, BroadcastSubUInt8>(
                a_cont, b_cont, output, info, queue,
                [](uint8_t x, uint8_t y) { return static_cast<uint8_t>(x - y); });
        } else if (a_cont.dtype() == DType::Complex64) {
            sycl_broadcast_complex_binary<float, BroadcastSubComplex64>(
                a_cont, b_cont, output, info, queue,
                [](const float* a, const float* b, float* c, int64_t ai, int64_t bi, int64_t ci) {
                    c[ci]     = a[ai]     - b[bi];
                    c[ci + 1] = a[ai + 1] - b[bi + 1];
                });
        } else if (a_cont.dtype() == DType::Complex128) {
            sycl_broadcast_complex_binary<double, BroadcastSubComplex128>(
                a_cont, b_cont, output, info, queue,
                [](const double* a, const double* b, double* c, int64_t ai, int64_t bi, int64_t ci) {
                    c[ci]     = a[ai]     - b[bi];
                    c[ci + 1] = a[ai + 1] - b[bi + 1];
                });
        } else {
            throw std::runtime_error("sub broadcast: unsupported dtype");
        }
        return output;
    }

    Tensor output(std::vector<int64_t>(a_cont.shape().begin(), a_cont.shape().end()),
                  a_cont.dtype(), a_cont.device());

    const int64_t numel = a_cont.numel();

    if (a_cont.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a_cont);
        const float* b_ptr = get_data_ptr<const float>(b_cont);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<SubKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] - b_ptr[idx];
        });
    }
    else if (a_cont.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a_cont);
        const double* b_ptr = get_data_ptr<const double>(b_cont);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<SubKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] - b_ptr[idx];
        });
    }
    else if (a_cont.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a_cont);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<SubKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(static_cast<float>(a_ptr[idx]) - static_cast<float>(b_ptr[idx]));
        });
    }
    else if (a_cont.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a_cont);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<SubKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(bf16_to_f32(a_ptr[idx]) - bf16_to_f32(b_ptr[idx]));
        });
    }
    else if (a_cont.dtype() == DType::Int8) {
        const int8_t* a_ptr = get_data_ptr<const int8_t>(a_cont);
        const int8_t* b_ptr = get_data_ptr<const int8_t>(b_cont);
        int8_t* out_ptr = get_data_ptr<int8_t>(output);

        queue.parallel_for<SubKernelInt8>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] - b_ptr[idx];
        });
    }
    else if (a_cont.dtype() == DType::Int32) {
        const int32_t* a_ptr = get_data_ptr<const int32_t>(a_cont);
        const int32_t* b_ptr = get_data_ptr<const int32_t>(b_cont);
        int32_t* out_ptr = get_data_ptr<int32_t>(output);

        queue.parallel_for<SubKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] - b_ptr[idx];
        });
    }
    else if (a_cont.dtype() == DType::Int64) {
        const int64_t* a_ptr = get_data_ptr<const int64_t>(a_cont);
        const int64_t* b_ptr = get_data_ptr<const int64_t>(b_cont);
        int64_t* out_ptr = get_data_ptr<int64_t>(output);

        queue.parallel_for<SubKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] - b_ptr[idx];
        });
    }
    else if (a_cont.dtype() == DType::UInt8) {
        const uint8_t* a_ptr = get_data_ptr<const uint8_t>(a_cont);
        const uint8_t* b_ptr = get_data_ptr<const uint8_t>(b_cont);
        uint8_t* out_ptr = get_data_ptr<uint8_t>(output);

        queue.parallel_for<SubKernelUInt8>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] - b_ptr[idx];
        });
    }
    else if (a_cont.dtype() == DType::Complex64) {
        const float* a_ptr = get_data_ptr<const float>(a_cont);
        const float* b_ptr = get_data_ptr<const float>(b_cont);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<SubKernelComplex64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int64_t base = idx[0] * 2;
            out_ptr[base]     = a_ptr[base]     - b_ptr[base];
            out_ptr[base + 1] = a_ptr[base + 1] - b_ptr[base + 1];
        });
    }
    else if (a_cont.dtype() == DType::Complex128) {
        const double* a_ptr = get_data_ptr<const double>(a_cont);
        const double* b_ptr = get_data_ptr<const double>(b_cont);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<SubKernelComplex128>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int64_t base = idx[0] * 2;
            out_ptr[base]     = a_ptr[base]     - b_ptr[base];
            out_ptr[base + 1] = a_ptr[base + 1] - b_ptr[base + 1];
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for subtraction");
    }

    return output;
}

// Element-wise multiplication kernel
// IMPORTANT: Must ensure contiguous inputs for direct memory access
auto mul_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    // Ensure inputs are contiguous for correct memory access
    Tensor a_cont = a.is_contiguous() ? a : contiguous_kernel(a, queue);
    Tensor b_cont = b.is_contiguous() ? b : contiguous_kernel(b, queue);

    auto a_shape = a_cont.shape();
    auto b_shape = b_cont.shape();

    // Check if shapes match exactly
    bool same_shape = std::equal(a_shape.begin(), a_shape.end(), b_shape.begin(), b_shape.end());

    if (!same_shape) {
        auto out_shape = broadcast_shapes(a_shape, b_shape);
        auto info = compute_broadcast_info(a_shape, b_shape, out_shape);
        Tensor output(out_shape, a_cont.dtype(), a_cont.device());
        if (a_cont.dtype() == DType::Float32) {
            sycl_broadcast_binary<float, BroadcastMulFloat32>(
                a_cont, b_cont, output, info, queue,
                [](float x, float y) { return x * y; });
        } else if (a_cont.dtype() == DType::Float64) {
            sycl_broadcast_binary<double, BroadcastMulFloat64>(
                a_cont, b_cont, output, info, queue,
                [](double x, double y) { return x * y; });
        } else if (a_cont.dtype() == DType::Int32) {
            sycl_broadcast_binary<int32_t, BroadcastMulInt32>(
                a_cont, b_cont, output, info, queue,
                [](int32_t x, int32_t y) { return x * y; });
        } else if (a_cont.dtype() == DType::Int64) {
            sycl_broadcast_binary<int64_t, BroadcastMulInt64>(
                a_cont, b_cont, output, info, queue,
                [](int64_t x, int64_t y) { return x * y; });
        } else if (a_cont.dtype() == DType::Float16) {
            sycl_broadcast_binary<sycl::half, BroadcastMulFloat16>(
                a_cont, b_cont, output, info, queue,
                [](sycl::half x, sycl::half y) { return x * y; });
        } else if (a_cont.dtype() == DType::BFloat16) {
            sycl_broadcast_binary<uint16_t, BroadcastMulBFloat16>(
                a_cont, b_cont, output, info, queue,
                [](uint16_t x, uint16_t y) { return f32_to_bf16(bf16_to_f32(x) * bf16_to_f32(y)); });
        } else if (a_cont.dtype() == DType::Int8) {
            sycl_broadcast_binary<int8_t, BroadcastMulInt8>(
                a_cont, b_cont, output, info, queue,
                [](int8_t x, int8_t y) { return static_cast<int8_t>(x * y); });
        } else if (a_cont.dtype() == DType::UInt8) {
            sycl_broadcast_binary<uint8_t, BroadcastMulUInt8>(
                a_cont, b_cont, output, info, queue,
                [](uint8_t x, uint8_t y) { return static_cast<uint8_t>(x * y); });
        } else if (a_cont.dtype() == DType::Complex64) {
            sycl_broadcast_complex_binary<float, BroadcastMulComplex64>(
                a_cont, b_cont, output, info, queue,
                [](const float* a, const float* b, float* c, int64_t ai, int64_t bi, int64_t ci) {
                    float ar = a[ai], ai_ = a[ai + 1], br = b[bi], bi_ = b[bi + 1];
                    c[ci]     = ar * br - ai_ * bi_;
                    c[ci + 1] = ar * bi_ + ai_ * br;
                });
        } else if (a_cont.dtype() == DType::Complex128) {
            sycl_broadcast_complex_binary<double, BroadcastMulComplex128>(
                a_cont, b_cont, output, info, queue,
                [](const double* a, const double* b, double* c, int64_t ai, int64_t bi, int64_t ci) {
                    double ar = a[ai], ai_ = a[ai + 1], br = b[bi], bi_ = b[bi + 1];
                    c[ci]     = ar * br - ai_ * bi_;
                    c[ci + 1] = ar * bi_ + ai_ * br;
                });
        } else {
            throw std::runtime_error("mul broadcast: unsupported dtype");
        }
        return output;
    }

    Tensor output(std::vector<int64_t>(a_cont.shape().begin(), a_cont.shape().end()),
                  a_cont.dtype(), a_cont.device());

    const int64_t numel = a_cont.numel();

    if (a_cont.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a_cont);
        const float* b_ptr = get_data_ptr<const float>(b_cont);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<MulKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] * b_ptr[idx];
        });
    }
    else if (a_cont.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a_cont);
        const double* b_ptr = get_data_ptr<const double>(b_cont);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<MulKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] * b_ptr[idx];
        });
    }
    else if (a_cont.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a_cont);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<MulKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(static_cast<float>(a_ptr[idx]) * static_cast<float>(b_ptr[idx]));
        });
    }
    else if (a_cont.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a_cont);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<MulKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(bf16_to_f32(a_ptr[idx]) * bf16_to_f32(b_ptr[idx]));
        });
    }
    else if (a_cont.dtype() == DType::Int32) {
        const int32_t* a_ptr = get_data_ptr<const int32_t>(a_cont);
        const int32_t* b_ptr = get_data_ptr<const int32_t>(b_cont);
        int32_t* out_ptr = get_data_ptr<int32_t>(output);

        queue.parallel_for<MulKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] * b_ptr[idx];
        });
    }
    else if (a_cont.dtype() == DType::Bool) {
        // Bool multiplication acts as logical AND
        const bool* a_ptr = get_data_ptr<const bool>(a_cont);
        const bool* b_ptr = get_data_ptr<const bool>(b_cont);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<MulKernelBool>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] && b_ptr[idx];
        });
    }
    else if (a_cont.dtype() == DType::Complex64) {
        const float* a_ptr = get_data_ptr<const float>(a_cont);
        const float* b_ptr = get_data_ptr<const float>(b_cont);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<MulKernelComplex64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int64_t base = idx[0] * 2;
            float ar = a_ptr[base], ai = a_ptr[base + 1];
            float br = b_ptr[base], bi = b_ptr[base + 1];
            out_ptr[base]     = ar * br - ai * bi;
            out_ptr[base + 1] = ar * bi + ai * br;
        });
    }
    else if (a_cont.dtype() == DType::Complex128) {
        const double* a_ptr = get_data_ptr<const double>(a_cont);
        const double* b_ptr = get_data_ptr<const double>(b_cont);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<MulKernelComplex128>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int64_t base = idx[0] * 2;
            double ar = a_ptr[base], ai = a_ptr[base + 1];
            double br = b_ptr[base], bi = b_ptr[base + 1];
            out_ptr[base]     = ar * br - ai * bi;
            out_ptr[base + 1] = ar * bi + ai * br;
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for multiplication");
    }

    return output;
}

// Helper function to check for zeros in integer array (for division by zero check)
template<typename T>
inline void check_integer_divisor_for_zeros(const T* data, int64_t n, sycl::queue& queue) {
    // Use atomic to track if any zero is found
    int* has_zero = sycl::malloc_shared<int>(1, queue);
    *has_zero = 0;

    queue.parallel_for<CheckZerosKernel>(sycl::range<1>(n), [=](sycl::id<1> idx) {
        if (data[idx] == T(0)) {
            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                            sycl::access::address_space::global_space> flag(*has_zero);
            flag.store(1);
        }
    });

    bool found_zero = (*has_zero != 0);
    sycl::free(has_zero, queue);

    if (found_zero) {
        throw std::runtime_error("Integer division by zero");
    }
}

// Element-wise division kernel
// IMPORTANT: Must ensure contiguous inputs for direct memory access
auto div_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    // Ensure inputs are contiguous for correct memory access
    Tensor a_cont = a.is_contiguous() ? a : contiguous_kernel(a, queue);
    Tensor b_cont = b.is_contiguous() ? b : contiguous_kernel(b, queue);

    auto a_shape = a_cont.shape();
    auto b_shape = b_cont.shape();

    // Check if shapes match exactly
    bool same_shape = std::equal(a_shape.begin(), a_shape.end(), b_shape.begin(), b_shape.end());

    if (!same_shape) {
        auto out_shape = broadcast_shapes(a_shape, b_shape);
        auto info = compute_broadcast_info(a_shape, b_shape, out_shape);
        Tensor output(out_shape, a_cont.dtype(), a_cont.device());
        if (a_cont.dtype() == DType::Float32) {
            sycl_broadcast_binary<float, BroadcastDivFloat32>(
                a_cont, b_cont, output, info, queue,
                [](float x, float y) { return x / y; });
        } else if (a_cont.dtype() == DType::Float64) {
            sycl_broadcast_binary<double, BroadcastDivFloat64>(
                a_cont, b_cont, output, info, queue,
                [](double x, double y) { return x / y; });
        } else if (a_cont.dtype() == DType::Int32) {
            sycl_broadcast_binary<int32_t, BroadcastDivInt32>(
                a_cont, b_cont, output, info, queue,
                [](int32_t x, int32_t y) { return x / y; });
        } else if (a_cont.dtype() == DType::Int64) {
            sycl_broadcast_binary<int64_t, BroadcastDivInt64>(
                a_cont, b_cont, output, info, queue,
                [](int64_t x, int64_t y) { return x / y; });
        } else if (a_cont.dtype() == DType::Float16) {
            sycl_broadcast_binary<sycl::half, BroadcastDivFloat16>(
                a_cont, b_cont, output, info, queue,
                [](sycl::half x, sycl::half y) { return x / y; });
        } else if (a_cont.dtype() == DType::BFloat16) {
            sycl_broadcast_binary<uint16_t, BroadcastDivBFloat16>(
                a_cont, b_cont, output, info, queue,
                [](uint16_t x, uint16_t y) { return f32_to_bf16(bf16_to_f32(x) / bf16_to_f32(y)); });
        } else if (a_cont.dtype() == DType::Int8) {
            sycl_broadcast_binary<int8_t, BroadcastDivInt8>(
                a_cont, b_cont, output, info, queue,
                [](int8_t x, int8_t y) { return y != 0 ? static_cast<int8_t>(x / y) : int8_t(0); });
        } else if (a_cont.dtype() == DType::UInt8) {
            sycl_broadcast_binary<uint8_t, BroadcastDivUInt8>(
                a_cont, b_cont, output, info, queue,
                [](uint8_t x, uint8_t y) { return y != 0 ? static_cast<uint8_t>(x / y) : uint8_t(0); });
        } else if (a_cont.dtype() == DType::Complex64) {
            sycl_broadcast_complex_binary<float, BroadcastDivComplex64>(
                a_cont, b_cont, output, info, queue,
                [](const float* a, const float* b, float* c, int64_t ai, int64_t bi, int64_t ci) {
                    float ar = a[ai], ai_ = a[ai + 1], br = b[bi], bi_ = b[bi + 1];
                    float denom = br * br + bi_ * bi_;
                    c[ci]     = (ar * br + ai_ * bi_) / denom;
                    c[ci + 1] = (ai_ * br - ar * bi_) / denom;
                });
        } else if (a_cont.dtype() == DType::Complex128) {
            sycl_broadcast_complex_binary<double, BroadcastDivComplex128>(
                a_cont, b_cont, output, info, queue,
                [](const double* a, const double* b, double* c, int64_t ai, int64_t bi, int64_t ci) {
                    double ar = a[ai], ai_ = a[ai + 1], br = b[bi], bi_ = b[bi + 1];
                    double denom = br * br + bi_ * bi_;
                    c[ci]     = (ar * br + ai_ * bi_) / denom;
                    c[ci + 1] = (ai_ * br - ar * bi_) / denom;
                });
        } else {
            throw std::runtime_error("div broadcast: unsupported dtype");
        }
        return output;
    }

    Tensor output(std::vector<int64_t>(a_cont.shape().begin(), a_cont.shape().end()),
                  a_cont.dtype(), a_cont.device());

    const int64_t numel = a_cont.numel();

    if (a_cont.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a_cont);
        const float* b_ptr = get_data_ptr<const float>(b_cont);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<DivKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] / b_ptr[idx];
        });
    }
    else if (a_cont.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a_cont);
        const double* b_ptr = get_data_ptr<const double>(b_cont);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<DivKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] / b_ptr[idx];
        });
    }
    else if (a_cont.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a_cont);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<DivKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(static_cast<float>(a_ptr[idx]) / static_cast<float>(b_ptr[idx]));
        });
    }
    else if (a_cont.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a_cont);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<DivKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(bf16_to_f32(a_ptr[idx]) / bf16_to_f32(b_ptr[idx]));
        });
    }
    else if (a_cont.dtype() == DType::Int32) {
        const int32_t* a_ptr = get_data_ptr<const int32_t>(a_cont);
        const int32_t* b_ptr = get_data_ptr<const int32_t>(b_cont);
        int32_t* out_ptr = get_data_ptr<int32_t>(output);

        // Check for division by zero before executing kernel
        check_integer_divisor_for_zeros(b_ptr, numel, queue);

        queue.parallel_for<DivKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] / b_ptr[idx];
        });
    }
    else if (a_cont.dtype() == DType::Complex64) {
        const float* a_ptr = get_data_ptr<const float>(a_cont);
        const float* b_ptr = get_data_ptr<const float>(b_cont);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<DivKernelComplex64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int64_t base = idx[0] * 2;
            float ar = a_ptr[base], ai = a_ptr[base + 1];
            float br = b_ptr[base], bi = b_ptr[base + 1];
            float denom = br * br + bi * bi;
            out_ptr[base]     = (ar * br + ai * bi) / denom;
            out_ptr[base + 1] = (ai * br - ar * bi) / denom;
        });
    }
    else if (a_cont.dtype() == DType::Complex128) {
        const double* a_ptr = get_data_ptr<const double>(a_cont);
        const double* b_ptr = get_data_ptr<const double>(b_cont);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<DivKernelComplex128>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int64_t base = idx[0] * 2;
            double ar = a_ptr[base], ai = a_ptr[base + 1];
            double br = b_ptr[base], bi = b_ptr[base + 1];
            double denom = br * br + bi * bi;
            out_ptr[base]     = (ar * br + ai * bi) / denom;
            out_ptr[base + 1] = (ai * br - ar * bi) / denom;
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for division");
    }

    return output;
}

// Matrix multiplication kernel
// IMPORTANT: oneMKL GEMM assumes contiguous row-major layout
// Non-contiguous tensors (e.g., from transpose) must be made contiguous first
auto matmul_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    // Ensure inputs are contiguous - oneMKL GEMM requires contiguous memory layout
    Tensor a_cont = a.is_contiguous() ? a : contiguous_kernel(a, queue);
    Tensor b_cont = b.is_contiguous() ? b : contiguous_kernel(b, queue);

    auto a_shape = a_cont.shape();
    auto b_shape = b_cont.shape();

    // Handle 1D vector × 2D matrix (vector-matrix multiplication)
    if (a_shape.size() == 1 && b_shape.size() == 2) {
        const int64_t n = a_shape[0];  // Vector size
        const int64_t k = b_shape[0];  // Matrix rows
        const int64_t m = b_shape[1];  // Matrix cols

        if (n != k) {
            throw std::invalid_argument(
                "matmul dimension mismatch: vector(" + std::to_string(n) +
                ") @ matrix(" + std::to_string(k) + "×" + std::to_string(m) + ")"
            );
        }

        // Treat 1D vector as row vector (1, n) and perform matmul to get (1, m), then return as (m,)
        Tensor output({m}, a_cont.dtype(), a_cont.device());

#ifdef TENZOR_HAS_ONEMKL
        if (a_cont.dtype() == DType::Float32) {
            const float* a_ptr = get_data_ptr<const float>(a_cont);
            const float* b_ptr = get_data_ptr<const float>(b_cont);
            float* out_ptr = get_data_ptr<float>(output);

            const float alpha = 1.0f;
            const float beta = 0.0f;

            // For row-major: result[m] = vector[n] × matrix[n, m]
            // In oneMKL column-major: C^T[m] = B^T[m, n] × A^T[n]
            try {
                ::oneapi::mkl::blas::column_major::gemm(
                    queue,
                    ::oneapi::mkl::transpose::nontrans,
                    ::oneapi::mkl::transpose::nontrans,
                    m,        // rows of result
                    1,        // cols of result (single vector)
                    n,        // inner dimension
                    alpha,
                    b_ptr, m, // B matrix
                    a_ptr, n, // A vector
                    beta,
                    out_ptr, m // output
                );
                queue.wait_and_throw();
            } catch (const ::oneapi::mkl::exception& e) {
                throw std::runtime_error(std::string("oneMKL GEMM (F32 vec) failed: ") + e.what());
            } catch (const sycl::exception& e) {
                throw std::runtime_error(std::string("SYCL error in GEMM (F32 vec): ") + e.what());
            }
        }
        else if (a_cont.dtype() == DType::Float64) {
            const double* a_ptr = get_data_ptr<const double>(a_cont);
            const double* b_ptr = get_data_ptr<const double>(b_cont);
            double* out_ptr = get_data_ptr<double>(output);

            const double alpha = 1.0;
            const double beta = 0.0;

            try {
                ::oneapi::mkl::blas::column_major::gemm(
                    queue,
                    ::oneapi::mkl::transpose::nontrans,
                    ::oneapi::mkl::transpose::nontrans,
                    m, 1, n,
                    alpha,
                    b_ptr, m,
                    a_ptr, n,
                    beta,
                    out_ptr, m
                );
                queue.wait_and_throw();
            } catch (const ::oneapi::mkl::exception& e) {
                throw std::runtime_error(std::string("oneMKL GEMM (F64 vec) failed: ") + e.what());
            } catch (const sycl::exception& e) {
                throw std::runtime_error(std::string("SYCL error in GEMM (F64 vec): ") + e.what());
            }
        }
        else if (a_cont.dtype() == DType::BFloat16) {
            const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a_cont);
            const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b_cont);
            uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

            queue.parallel_for<class MatMulVecBFloat16>(sycl::range<1>(m), [=](sycl::id<1> idx) {
                const int64_t j = idx[0];
                float sum = 0.0f;
                for (int64_t p = 0; p < n; ++p) {
                    sum += bf16_to_f32(a_ptr[p]) * bf16_to_f32(b_ptr[p * m + j]);
                }
                out_ptr[j] = f32_to_bf16(sum);
            });
        }
        else {
            throw std::runtime_error("Unsupported dtype for 1D×2D matmul with oneMKL");
        }
#else
        // Fallback naive implementation
        if (a_cont.dtype() == DType::Float32) {
            const float* a_ptr = get_data_ptr<const float>(a_cont);
            const float* b_ptr = get_data_ptr<const float>(b_cont);
            float* out_ptr = get_data_ptr<float>(output);

            queue.parallel_for<class MatMulKernelVector>(sycl::range<1>(m), [=](sycl::id<1> idx) {
                const int64_t j = idx[0];
                float sum = 0.0f;
                for (int64_t p = 0; p < n; ++p) {
                    sum += a_ptr[p] * b_ptr[p * m + j];
                }
                out_ptr[j] = sum;
            });
        }
        else {
            throw std::runtime_error("Unsupported dtype for 1D×2D matmul fallback");
        }
#endif
        return output;
    }

    // Validate dimensions
    if (a_shape.size() < 2 || b_shape.size() < 2) {
        throw std::invalid_argument("matmul requires at least 2D tensors");
    }

    const int64_t m = a_shape[a_shape.size() - 2];
    const int64_t k = a_shape[a_shape.size() - 1];
    const int64_t k2 = b_shape[b_shape.size() - 2];
    const int64_t n = b_shape[b_shape.size() - 1];

    if (k != k2) {
        throw std::runtime_error("Inner dimensions must match for matmul");
    }

    // Create output shape
    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < a_shape.size() - 2; ++i) {
        out_shape.push_back(a_shape[i]);
    }
    out_shape.push_back(m);
    out_shape.push_back(n);

    Tensor output(out_shape, a_cont.dtype(), a_cont.device());

#ifdef TENZOR_HAS_ONEMKL
    // oneMKL GEMM has issues with sycl::malloc_shared memory on some devices/configurations
    // (8x multiplier bug when k>16). Use sycl::malloc_device with explicit transfers instead.
    if (a_cont.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a_cont);
        const float* b_ptr = get_data_ptr<const float>(b_cont);
        float* out_ptr = get_data_ptr<float>(output);

        auto deleter = [&queue](float* p) { sycl::free(p, queue); };
        std::unique_ptr<float, decltype(deleter)> d_a(
            static_cast<float*>(sycl::malloc_device(m * k * sizeof(float), queue)), deleter);
        std::unique_ptr<float, decltype(deleter)> d_b(
            static_cast<float*>(sycl::malloc_device(k * n * sizeof(float), queue)), deleter);
        std::unique_ptr<float, decltype(deleter)> d_c(
            static_cast<float*>(sycl::malloc_device(m * n * sizeof(float), queue)), deleter);

        queue.memcpy(d_a.get(), a_ptr, m * k * sizeof(float));
        queue.memcpy(d_b.get(), b_ptr, k * n * sizeof(float));
        queue.wait();

        const float alpha = 1.0f;
        const float beta = 0.0f;

        try {
            // Row-major C(m,n) = A(m,k) * B(k,n)
            // In column-major: C^T(n,m) = B^T(n,k) * A^T(k,m)
            ::oneapi::mkl::blas::column_major::gemm(
                queue,
                ::oneapi::mkl::transpose::nontrans,
                ::oneapi::mkl::transpose::nontrans,
                n, m, k,
                alpha,
                d_b.get(), n,
                d_a.get(), k,
                beta,
                d_c.get(), n
            );
            queue.wait_and_throw();
        } catch (const ::oneapi::mkl::exception& e) {
            throw std::runtime_error(std::string("oneMKL GEMM (F32) failed: ") + e.what());
        } catch (const sycl::exception& e) {
            throw std::runtime_error(std::string("SYCL error in GEMM (F32): ") + e.what());
        }

        queue.memcpy(out_ptr, d_c.get(), m * n * sizeof(float));
        queue.wait();
    }
    else if (a_cont.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a_cont);
        const double* b_ptr = get_data_ptr<const double>(b_cont);
        double* out_ptr = get_data_ptr<double>(output);

        auto deleter = [&queue](double* p) { sycl::free(p, queue); };
        std::unique_ptr<double, decltype(deleter)> d_a(
            static_cast<double*>(sycl::malloc_device(m * k * sizeof(double), queue)), deleter);
        std::unique_ptr<double, decltype(deleter)> d_b(
            static_cast<double*>(sycl::malloc_device(k * n * sizeof(double), queue)), deleter);
        std::unique_ptr<double, decltype(deleter)> d_c(
            static_cast<double*>(sycl::malloc_device(m * n * sizeof(double), queue)), deleter);

        queue.memcpy(d_a.get(), a_ptr, m * k * sizeof(double));
        queue.memcpy(d_b.get(), b_ptr, k * n * sizeof(double));
        queue.wait();

        const double alpha = 1.0;
        const double beta = 0.0;

        try {
            ::oneapi::mkl::blas::column_major::gemm(
                queue,
                ::oneapi::mkl::transpose::nontrans,
                ::oneapi::mkl::transpose::nontrans,
                n, m, k,
                alpha,
                d_b.get(), n,
                d_a.get(), k,
                beta,
                d_c.get(), n
            );
            queue.wait_and_throw();
        } catch (const ::oneapi::mkl::exception& e) {
            throw std::runtime_error(std::string("oneMKL GEMM (F64) failed: ") + e.what());
        } catch (const sycl::exception& e) {
            throw std::runtime_error(std::string("SYCL error in GEMM (F64): ") + e.what());
        }

        queue.memcpy(out_ptr, d_c.get(), m * n * sizeof(double));
        queue.wait();
    }
    else if (a_cont.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a_cont);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        // Use Float32 device buffers for accumulation precision
        auto half_deleter = [&queue](sycl::half* p) { sycl::free(p, queue); };
        auto float_deleter = [&queue](float* p) { sycl::free(p, queue); };
        std::unique_ptr<float, decltype(float_deleter)> d_a(
            static_cast<float*>(sycl::malloc_device(m * k * sizeof(float), queue)), float_deleter);
        std::unique_ptr<float, decltype(float_deleter)> d_b(
            static_cast<float*>(sycl::malloc_device(k * n * sizeof(float), queue)), float_deleter);
        std::unique_ptr<float, decltype(float_deleter)> d_c(
            static_cast<float*>(sycl::malloc_device(m * n * sizeof(float), queue)), float_deleter);

        // Convert FP16 inputs to FP32 on device
        std::unique_ptr<sycl::half, decltype(half_deleter)> d_a_h(
            static_cast<sycl::half*>(sycl::malloc_device(m * k * sizeof(sycl::half), queue)), half_deleter);
        std::unique_ptr<sycl::half, decltype(half_deleter)> d_b_h(
            static_cast<sycl::half*>(sycl::malloc_device(k * n * sizeof(sycl::half), queue)), half_deleter);
        std::unique_ptr<sycl::half, decltype(half_deleter)> d_c_h(
            static_cast<sycl::half*>(sycl::malloc_device(m * n * sizeof(sycl::half), queue)), half_deleter);

        queue.memcpy(d_a_h.get(), a_ptr, m * k * sizeof(sycl::half));
        queue.memcpy(d_b_h.get(), b_ptr, k * n * sizeof(sycl::half));
        queue.wait();

        // Upcast FP16 → FP32 on device
        const int64_t a_count = m * k;
        const int64_t b_count = k * n;
        auto* d_a_h_raw = d_a_h.get();
        auto* d_a_raw = d_a.get();
        queue.parallel_for(sycl::range<1>(a_count), [=](sycl::id<1> i) {
            d_a_raw[i] = static_cast<float>(d_a_h_raw[i]);
        });
        auto* d_b_h_raw = d_b_h.get();
        auto* d_b_raw = d_b.get();
        queue.parallel_for(sycl::range<1>(b_count), [=](sycl::id<1> i) {
            d_b_raw[i] = static_cast<float>(d_b_h_raw[i]);
        });
        queue.wait();

        const float alpha = 1.0f;
        const float beta = 0.0f;

        try {
            ::oneapi::mkl::blas::column_major::gemm(
                queue,
                ::oneapi::mkl::transpose::nontrans,
                ::oneapi::mkl::transpose::nontrans,
                n, m, k,
                alpha,
                d_b.get(), n,
                d_a.get(), k,
                beta,
                d_c.get(), n
            );
            queue.wait_and_throw();
        } catch (const ::oneapi::mkl::exception& e) {
            throw std::runtime_error(std::string("oneMKL GEMM (F16) failed: ") + e.what());
        } catch (const sycl::exception& e) {
            throw std::runtime_error(std::string("SYCL error in GEMM (F16): ") + e.what());
        }

        // Downcast FP32 → FP16 on device, then copy back
        const int64_t c_count = m * n;
        auto* d_c_raw = d_c.get();
        auto* d_c_h_raw = d_c_h.get();
        queue.parallel_for(sycl::range<1>(c_count), [=](sycl::id<1> i) {
            d_c_h_raw[i] = sycl::half(d_c_raw[i]);
        });
        queue.wait();

        queue.memcpy(out_ptr, d_c_h.get(), m * n * sizeof(sycl::half));
        queue.wait();
    }
    else if (a_cont.dtype() == DType::Int32) {
        const int32_t* a_ptr = get_data_ptr<const int32_t>(a_cont);
        const int32_t* b_ptr = get_data_ptr<const int32_t>(b_cont);
        int32_t* out_ptr = get_data_ptr<int32_t>(output);

        // Use int64 accumulation to avoid overflow
        queue.parallel_for<MatMulKernelInt32>(sycl::range<2>(m, n), [=](sycl::id<2> idx) {
            const int64_t i = idx[0];
            const int64_t j = idx[1];

            int64_t sum = 0;
            for (int64_t p = 0; p < k; ++p) {
                sum += static_cast<int64_t>(a_ptr[i * k + p]) * static_cast<int64_t>(b_ptr[p * n + j]);
            }
            out_ptr[i * n + j] = static_cast<int32_t>(sum);
        });
    }
    else if (a_cont.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a_cont);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<MatMulKernelBFloat16>(sycl::range<2>(m, n), [=](sycl::id<2> idx) {
            const int64_t i = idx[0];
            const int64_t j = idx[1];

            float sum = 0.0f;
            for (int64_t p = 0; p < k; ++p) {
                sum += bf16_to_f32(a_ptr[i * k + p]) * bf16_to_f32(b_ptr[p * n + j]);
            }
            out_ptr[i * n + j] = f32_to_bf16(sum);
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for matmul with oneMKL");
    }
#else
    // Fallback naive implementation for Float32
    if (a_cont.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a_cont);
        const float* b_ptr = get_data_ptr<const float>(b_cont);
        float* out_ptr = get_data_ptr<float>(output);

        // Simple parallel matrix multiplication
        queue.parallel_for<MatMulKernelFloat32>(sycl::range<2>(m, n), [=](sycl::id<2> idx) {
            const int64_t i = idx[0];
            const int64_t j = idx[1];

            float sum = 0.0f;
            for (int64_t p = 0; p < k; ++p) {
                sum += a_ptr[i * k + p] * b_ptr[p * n + j];
            }
            out_ptr[i * n + j] = sum;
        });
    }
    else if (a_cont.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a_cont);
        const double* b_ptr = get_data_ptr<const double>(b_cont);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<MatMulKernelFloat64>(sycl::range<2>(m, n), [=](sycl::id<2> idx) {
            const int64_t i = idx[0];
            const int64_t j = idx[1];

            double sum = 0.0;
            for (int64_t p = 0; p < k; ++p) {
                sum += a_ptr[i * k + p] * b_ptr[p * n + j];
            }
            out_ptr[i * n + j] = sum;
        });
    }
    else if (a_cont.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a_cont);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        // Use float accumulation for precision
        queue.parallel_for<MatMulKernelFloat16>(sycl::range<2>(m, n), [=](sycl::id<2> idx) {
            const int64_t i = idx[0];
            const int64_t j = idx[1];

            float sum = 0.0f;
            for (int64_t p = 0; p < k; ++p) {
                sum += static_cast<float>(a_ptr[i * k + p]) * static_cast<float>(b_ptr[p * n + j]);
            }
            out_ptr[i * n + j] = sycl::half(sum);
        });
    }
    else if (a_cont.dtype() == DType::Int32) {
        const int32_t* a_ptr = get_data_ptr<const int32_t>(a_cont);
        const int32_t* b_ptr = get_data_ptr<const int32_t>(b_cont);
        int32_t* out_ptr = get_data_ptr<int32_t>(output);

        // Use int64 accumulation to avoid overflow
        queue.parallel_for<MatMulKernelInt32>(sycl::range<2>(m, n), [=](sycl::id<2> idx) {
            const int64_t i = idx[0];
            const int64_t j = idx[1];

            int64_t sum = 0;
            for (int64_t p = 0; p < k; ++p) {
                sum += static_cast<int64_t>(a_ptr[i * k + p]) * static_cast<int64_t>(b_ptr[p * n + j]);
            }
            out_ptr[i * n + j] = static_cast<int32_t>(sum);
        });
    }
    else if (a_cont.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a_cont);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<MatMulKernelBFloat16>(sycl::range<2>(m, n), [=](sycl::id<2> idx) {
            const int64_t i = idx[0];
            const int64_t j = idx[1];

            float sum = 0.0f;
            for (int64_t p = 0; p < k; ++p) {
                sum += bf16_to_f32(a_ptr[i * k + p]) * bf16_to_f32(b_ptr[p * n + j]);
            }
            out_ptr[i * n + j] = f32_to_bf16(sum);
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for matmul");
    }
#endif

    return output;
}

// Batched matrix multiplication kernel
// For 3D tensors [batch, M, K] x [batch, K, N] -> [batch, M, N]
auto bmm_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    // Ensure inputs are contiguous
    Tensor a_cont = a.is_contiguous() ? a : contiguous_kernel(a, queue);
    Tensor b_cont = b.is_contiguous() ? b : contiguous_kernel(b, queue);

    auto a_shape = a_cont.shape();
    auto b_shape = b_cont.shape();

    // Validate inputs are 3D
    if (a_shape.size() != 3 || b_shape.size() != 3) {
        throw std::runtime_error(
            "bmm_kernel requires 3D tensors, got shapes: [" +
            std::to_string(a_shape.size()) + "D] and [" +
            std::to_string(b_shape.size()) + "D]");
    }

    const int64_t batch_size = a_shape[0];
    const int64_t M = a_shape[1];  // rows of A
    const int64_t K = a_shape[2];  // cols of A = rows of B
    const int64_t N = b_shape[2];  // cols of B

    // Validate dimensions
    if (b_shape[0] != batch_size || b_shape[1] != K) {
        throw std::runtime_error(
            "bmm_kernel dimension mismatch: expected b.shape=[" +
            std::to_string(batch_size) + ", " + std::to_string(K) + ", *], got [" +
            std::to_string(b_shape[0]) + ", " + std::to_string(b_shape[1]) + ", " +
            std::to_string(b_shape[2]) + "]");
    }

    // Output shape: [batch_size, M, N]
    Tensor output({batch_size, M, N}, a_cont.dtype(), a_cont.device());

    if (a_cont.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a_cont);
        const float* b_ptr = get_data_ptr<const float>(b_cont);
        float* out_ptr = get_data_ptr<float>(output);

        // Parallelize over batch and output elements
        // Each work item computes one element of the output
        queue.parallel_for<BmmKernelFloat32>(
            sycl::range<3>(batch_size, M, N),
            [=](sycl::id<3> idx) {
                const int64_t batch = idx[0];
                const int64_t i = idx[1];
                const int64_t j = idx[2];

                const int64_t a_batch_offset = batch * M * K;
                const int64_t b_batch_offset = batch * K * N;
                const int64_t out_batch_offset = batch * M * N;

                float sum = 0.0f;
                for (int64_t p = 0; p < K; ++p) {
                    sum += a_ptr[a_batch_offset + i * K + p] *
                           b_ptr[b_batch_offset + p * N + j];
                }
                out_ptr[out_batch_offset + i * N + j] = sum;
            });
    }
    else if (a_cont.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a_cont);
        const double* b_ptr = get_data_ptr<const double>(b_cont);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<BmmKernelFloat64>(
            sycl::range<3>(batch_size, M, N),
            [=](sycl::id<3> idx) {
                const int64_t batch = idx[0];
                const int64_t i = idx[1];
                const int64_t j = idx[2];

                const int64_t a_batch_offset = batch * M * K;
                const int64_t b_batch_offset = batch * K * N;
                const int64_t out_batch_offset = batch * M * N;

                double sum = 0.0;
                for (int64_t p = 0; p < K; ++p) {
                    sum += a_ptr[a_batch_offset + i * K + p] *
                           b_ptr[b_batch_offset + p * N + j];
                }
                out_ptr[out_batch_offset + i * N + j] = sum;
            });
    }
    else if (a_cont.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a_cont);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        // Use float accumulation for precision
        queue.parallel_for<BmmKernelFloat16>(
            sycl::range<3>(batch_size, M, N),
            [=](sycl::id<3> idx) {
                const int64_t batch = idx[0];
                const int64_t i = idx[1];
                const int64_t j = idx[2];

                const int64_t a_batch_offset = batch * M * K;
                const int64_t b_batch_offset = batch * K * N;
                const int64_t out_batch_offset = batch * M * N;

                float sum = 0.0f;
                for (int64_t p = 0; p < K; ++p) {
                    sum += static_cast<float>(a_ptr[a_batch_offset + i * K + p]) *
                           static_cast<float>(b_ptr[b_batch_offset + p * N + j]);
                }
                out_ptr[out_batch_offset + i * N + j] = sycl::half(sum);
            });
    }
    else if (a_cont.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a_cont);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<BmmKernelBFloat16>(sycl::range<3>(batch_size, M, N),
            [=](sycl::id<3> idx) {
                const int64_t batch = idx[0];
                const int64_t i = idx[1];
                const int64_t j = idx[2];

                const int64_t a_batch_offset = batch * M * K;
                const int64_t b_batch_offset = batch * K * N;
                const int64_t out_batch_offset = batch * M * N;

                float sum = 0.0f;
                for (int64_t p = 0; p < K; ++p) {
                    sum += bf16_to_f32(a_ptr[a_batch_offset + i * K + p]) *
                           bf16_to_f32(b_ptr[b_batch_offset + p * N + j]);
                }
                out_ptr[out_batch_offset + i * N + j] = f32_to_bf16(sum);
            });
    }
    else {
        throw std::runtime_error("Unsupported dtype for bmm: " +
            std::string(dtype_name(a_cont.dtype())));
    }

    return output;
}

// Square root kernel
auto sqrt_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<SqrtKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::sqrt(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<SqrtKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::sqrt(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<SqrtKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::sqrt(static_cast<float>(in_ptr[idx])));
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<SqrtKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::sqrt(bf16_to_f32(in_ptr[idx])));
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for sqrt");
    }

    return output;
}

// Negation kernel
auto neg_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<NegKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = -in_ptr[idx];
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<NegKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = -in_ptr[idx];
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<NegKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(-static_cast<float>(in_ptr[idx]));
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<NegKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(-bf16_to_f32(in_ptr[idx]));
        });
    }
    else if (input.dtype() == DType::Int32) {
        const int32_t* in_ptr = get_data_ptr<const int32_t>(input);
        int32_t* out_ptr = get_data_ptr<int32_t>(output);

        queue.parallel_for<NegKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = -in_ptr[idx];
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for negation");
    }

    return output;
}

// Absolute value kernel
auto abs_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<AbsKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fabs(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<AbsKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fabs(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<AbsKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::fabs(static_cast<float>(in_ptr[idx])));
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<AbsKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::fabs(bf16_to_f32(in_ptr[idx])));
        });
    }
    else if (input.dtype() == DType::Int32) {
        const int32_t* in_ptr = get_data_ptr<const int32_t>(input);
        int32_t* out_ptr = get_data_ptr<int32_t>(output);

        queue.parallel_for<AbsKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::abs(in_ptr[idx]);
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for abs");
    }

    return output;
}

// Natural logarithm kernel
auto log_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<LogKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::log(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<LogKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::log(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<LogKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::log(static_cast<float>(in_ptr[idx])));
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<LogKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::log(bf16_to_f32(in_ptr[idx])));
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for log");
    }

    return output;
}

// Exponential kernel
auto exp_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<ExpKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::exp(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<ExpKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::exp(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<ExpKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            // Convert to float, compute exp, convert back to half
            out_ptr[idx] = sycl::half(sycl::exp(static_cast<float>(in_ptr[idx])));
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<ExpKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::exp(bf16_to_f32(in_ptr[idx])));
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for exp");
    }

    return output;
}

// Power kernel
auto pow_kernel(const Tensor& input, float exponent, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<PowKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::pow(in_ptr[idx], exponent);
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        const double exp_d = static_cast<double>(exponent);

        queue.parallel_for<PowKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::pow(in_ptr[idx], exp_d);
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        // Use float accumulation for numerical stability
        queue.parallel_for<PowKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float val = static_cast<float>(in_ptr[idx]);
            out_ptr[idx] = sycl::half(sycl::pow(val, exponent));
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<PowKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::pow(bf16_to_f32(in_ptr[idx]), exponent));
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for pow");
    }

    return output;
}

// Dot product kernel - element-wise multiply then sum
auto dot_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    // Verify both tensors are 1D
    if (a.ndim() != 1 || b.ndim() != 1) {
        throw std::invalid_argument("dot: inputs must be 1D tensors");
    }

    // Verify same shape
    if (a.shape()[0] != b.shape()[0]) {
        throw std::invalid_argument("dot: inputs must have the same length");
    }

    // Verify same dtype
    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument("dot: inputs must have the same dtype");
    }

    int64_t n = a.shape()[0];

    // Create scalar output tensor
    Tensor output({1}, a.dtype(), a.device());

    if (a.dtype() == DType::Float32) {
        const float* a_data = get_data_ptr<const float>(a);
        const float* b_data = get_data_ptr<const float>(b);
        float* out_ptr = get_data_ptr<float>(output);

        auto sum_buf = sycl::malloc_shared<float>(1, queue);
        sum_buf[0] = 0.0f;

        queue.parallel_for(sycl::range<1>(n), sycl::reduction(sum_buf, sycl::plus<float>()),
                          [=](sycl::id<1> i, auto& s) {
            s += a_data[i] * b_data[i];
        });
        queue.wait();
        out_ptr[0] = sum_buf[0];
        sycl::free(sum_buf, queue);

        return output;
    }
    else if (a.dtype() == DType::Float64) {
        const double* a_data = get_data_ptr<const double>(a);
        const double* b_data = get_data_ptr<const double>(b);
        double* out_ptr = get_data_ptr<double>(output);

        auto sum_buf = sycl::malloc_shared<double>(1, queue);
        sum_buf[0] = 0.0;

        queue.parallel_for(sycl::range<1>(n), sycl::reduction(sum_buf, sycl::plus<double>()),
                          [=](sycl::id<1> i, auto& s) {
            s += a_data[i] * b_data[i];
        });
        queue.wait();
        out_ptr[0] = sum_buf[0];
        sycl::free(sum_buf, queue);

        return output;
    }
    else {
        throw std::runtime_error("dot: only Float32 and Float64 are supported");
    }
}

// ============================================================================
// Trigonometric Functions
// ============================================================================

// Sine kernel
auto sin_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<SinKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::sin(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<SinKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::sin(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<SinKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::sin(static_cast<float>(in_ptr[idx])));
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<SinKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::sin(bf16_to_f32(in_ptr[idx])));
        });
    }
    else {
        throw std::runtime_error("sin: unsupported dtype");
    }

    return output;
}

// Cosine kernel
auto cos_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<CosKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::cos(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<CosKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::cos(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<CosKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::cos(static_cast<float>(in_ptr[idx])));
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<CosKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::cos(bf16_to_f32(in_ptr[idx])));
        });
    }
    else {
        throw std::runtime_error("cos: unsupported dtype");
    }

    return output;
}

// Tangent kernel
auto tan_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<TanKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::tan(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<TanKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::tan(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<TanKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::tan(static_cast<float>(in_ptr[idx])));
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<TanKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::tan(bf16_to_f32(in_ptr[idx])));
        });
    }
    else {
        throw std::runtime_error("tan: unsupported dtype");
    }

    return output;
}

// Arc sine kernel
auto asin_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<AsinKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::asin(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<AsinKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::asin(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<AsinKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::asin(static_cast<float>(in_ptr[idx])));
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<AsinKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::asin(bf16_to_f32(in_ptr[idx])));
        });
    }
    else {
        throw std::runtime_error("asin: unsupported dtype");
    }

    return output;
}

// Arc cosine kernel
auto acos_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<AcosKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::acos(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<AcosKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::acos(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<AcosKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::acos(static_cast<float>(in_ptr[idx])));
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<AcosKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::acos(bf16_to_f32(in_ptr[idx])));
        });
    }
    else {
        throw std::runtime_error("acos: unsupported dtype");
    }

    return output;
}

// Arc tangent kernel
auto atan_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<AtanKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::atan(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<AtanKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::atan(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<AtanKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::atan(static_cast<float>(in_ptr[idx])));
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<AtanKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::atan(bf16_to_f32(in_ptr[idx])));
        });
    }
    else {
        throw std::runtime_error("atan: unsupported dtype");
    }

    return output;
}

// Hyperbolic sine kernel
auto sinh_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<SinhKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::sinh(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<SinhKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::sinh(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<SinhKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::sinh(static_cast<float>(in_ptr[idx])));
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<SinhKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::sinh(bf16_to_f32(in_ptr[idx])));
        });
    }
    else {
        throw std::runtime_error("sinh: unsupported dtype");
    }

    return output;
}

// Hyperbolic cosine kernel
auto cosh_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<CoshKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::cosh(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<CoshKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::cosh(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<CoshKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::cosh(static_cast<float>(in_ptr[idx])));
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<CoshKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::cosh(bf16_to_f32(in_ptr[idx])));
        });
    }
    else {
        throw std::runtime_error("cosh: unsupported dtype");
    }

    return output;
}

// Arc tangent 2 (atan2) kernel - binary operation
auto atan2_kernel(const Tensor& y, const Tensor& x, sycl::queue& queue) -> Tensor {
    if (y.dtype() != x.dtype()) {
        throw std::invalid_argument("atan2: input dtypes must match");
    }

    Tensor output(std::vector<int64_t>(y.shape().begin(), y.shape().end()),
                  y.dtype(), y.device());

    const int64_t numel = y.numel();

    if (y.dtype() == DType::Float32) {
        const float* y_ptr = get_data_ptr<const float>(y);
        const float* x_ptr = get_data_ptr<const float>(x);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<Atan2KernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::atan2(y_ptr[idx], x_ptr[idx]);
        });
    }
    else if (y.dtype() == DType::Float64) {
        const double* y_ptr = get_data_ptr<const double>(y);
        const double* x_ptr = get_data_ptr<const double>(x);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<Atan2KernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::atan2(y_ptr[idx], x_ptr[idx]);
        });
    }
    else if (y.dtype() == DType::Float16) {
        const sycl::half* y_ptr = get_data_ptr<const sycl::half>(y);
        const sycl::half* x_ptr = get_data_ptr<const sycl::half>(x);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<Atan2KernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::atan2(static_cast<float>(y_ptr[idx]), static_cast<float>(x_ptr[idx])));
        });
    }
    else if (y.dtype() == DType::BFloat16) {
        const uint16_t* y_ptr = get_data_ptr<const uint16_t>(y);
        const uint16_t* x_ptr = get_data_ptr<const uint16_t>(x);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<Atan2KernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::atan2(bf16_to_f32(y_ptr[idx]), bf16_to_f32(x_ptr[idx])));
        });
    }
    else {
        throw std::runtime_error("atan2: unsupported dtype");
    }

    return output;
}

// ============================================================================
// Rounding Functions
// ============================================================================

// Round kernel
auto round_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<RoundKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::round(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<RoundKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::round(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<RoundKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::round(static_cast<float>(in_ptr[idx])));
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<RoundKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::round(bf16_to_f32(in_ptr[idx])));
        });
    }
    else {
        throw std::runtime_error("round: unsupported dtype");
    }

    return output;
}

// Floor kernel
auto floor_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<FloorKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::floor(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<FloorKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::floor(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<FloorKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::floor(static_cast<float>(in_ptr[idx])));
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<FloorKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::floor(bf16_to_f32(in_ptr[idx])));
        });
    }
    else {
        throw std::runtime_error("floor: unsupported dtype");
    }

    return output;
}

// Ceil kernel
auto ceil_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<CeilKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::ceil(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<CeilKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::ceil(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<CeilKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::ceil(static_cast<float>(in_ptr[idx])));
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<CeilKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::ceil(bf16_to_f32(in_ptr[idx])));
        });
    }
    else {
        throw std::runtime_error("ceil: unsupported dtype");
    }

    return output;
}

// Trunc kernel
auto trunc_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<TruncKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::trunc(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<TruncKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::trunc(in_ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<TruncKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::trunc(static_cast<float>(in_ptr[idx])));
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<TruncKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::trunc(bf16_to_f32(in_ptr[idx])));
        });
    }
    else {
        throw std::runtime_error("trunc: unsupported dtype");
    }

    return output;
}

// Reciprocal kernel
auto reciprocal_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<ReciprocalKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = 1.0f / in_ptr[idx];
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<ReciprocalKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = 1.0 / in_ptr[idx];
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<ReciprocalKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(1.0f / static_cast<float>(in_ptr[idx]));
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<ReciprocalKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(1.0f / bf16_to_f32(in_ptr[idx]));
        });
    }
    else {
        throw std::runtime_error("reciprocal: unsupported dtype");
    }

    return output;
}

// ============================================================================
// Utility Functions
// ============================================================================

// Clamp min kernel
auto clamp_min_kernel(const Tensor& input, float min_val, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<ClampMinKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fmax(in_ptr[idx], min_val);
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        const double min_d = static_cast<double>(min_val);

        queue.parallel_for<ClampMinKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fmax(in_ptr[idx], min_d);
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<ClampMinKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::fmax(static_cast<float>(in_ptr[idx]), min_val));
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<ClampMinKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::fmax(bf16_to_f32(in_ptr[idx]), min_val));
        });
    }
    else {
        throw std::runtime_error("clamp_min: unsupported dtype");
    }

    return output;
}

// Clamp max kernel
auto clamp_max_kernel(const Tensor& input, float max_val, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<ClampMaxKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fmin(in_ptr[idx], max_val);
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        const double max_d = static_cast<double>(max_val);

        queue.parallel_for<ClampMaxKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fmin(in_ptr[idx], max_d);
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<ClampMaxKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::fmin(static_cast<float>(in_ptr[idx]), max_val));
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<ClampMaxKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::fmin(bf16_to_f32(in_ptr[idx]), max_val));
        });
    }
    else {
        throw std::runtime_error("clamp_max: unsupported dtype");
    }

    return output;
}

// Where kernel (conditional selection)
auto where_kernel(const Tensor& condition, const Tensor& x, const Tensor& y, sycl::queue& queue) -> Tensor {
    if (x.dtype() != y.dtype()) {
        throw std::invalid_argument("where: x and y must have the same dtype");
    }

    Tensor output(std::vector<int64_t>(x.shape().begin(), x.shape().end()),
                  x.dtype(), x.device());

    const int64_t numel = x.numel();

    if (x.dtype() == DType::Float32) {
        const bool* cond_ptr = get_data_ptr<const bool>(condition);
        const float* x_ptr = get_data_ptr<const float>(x);
        const float* y_ptr = get_data_ptr<const float>(y);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<WhereKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = cond_ptr[idx] ? x_ptr[idx] : y_ptr[idx];
        });
    }
    else if (x.dtype() == DType::Float64) {
        const bool* cond_ptr = get_data_ptr<const bool>(condition);
        const double* x_ptr = get_data_ptr<const double>(x);
        const double* y_ptr = get_data_ptr<const double>(y);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<WhereKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = cond_ptr[idx] ? x_ptr[idx] : y_ptr[idx];
        });
    }
    else if (x.dtype() == DType::Float16) {
        const bool* cond_ptr = get_data_ptr<const bool>(condition);
        const sycl::half* x_ptr = get_data_ptr<const sycl::half>(x);
        const sycl::half* y_ptr = get_data_ptr<const sycl::half>(y);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<WhereKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = cond_ptr[idx] ? x_ptr[idx] : y_ptr[idx];
        });
    }
    else if (x.dtype() == DType::BFloat16) {
        const bool* cond_ptr = get_data_ptr<const bool>(condition);
        const uint16_t* x_ptr = get_data_ptr<const uint16_t>(x);
        const uint16_t* y_ptr = get_data_ptr<const uint16_t>(y);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<WhereKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = cond_ptr[idx] ? x_ptr[idx] : y_ptr[idx];
        });
    }
    else if (x.dtype() == DType::Int64) {
        const bool* cond_ptr = get_data_ptr<const bool>(condition);
        const int64_t* x_ptr = get_data_ptr<const int64_t>(x);
        const int64_t* y_ptr = get_data_ptr<const int64_t>(y);
        int64_t* out_ptr = get_data_ptr<int64_t>(output);

        queue.parallel_for<WhereKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = cond_ptr[idx] ? x_ptr[idx] : y_ptr[idx];
        });
    }
    else {
        throw std::runtime_error("where: unsupported dtype");
    }

    return output;
}

// Repeat kernel - repeats tensor along specified dimensions
auto repeat_kernel(const Tensor& input, const std::vector<int64_t>& repeats, sycl::queue& queue) -> Tensor {
    auto shape = input.shape();
    if (repeats.size() != shape.size()) {
        throw std::invalid_argument("repeat: repeats size must match tensor dimensions");
    }

    // Calculate output shape
    std::vector<int64_t> out_shape(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        out_shape[i] = shape[i] * repeats[i];
    }

    Tensor output(out_shape, input.dtype(), input.device());

    // For simplicity, use a strided copy approach
    // This handles general N-dimensional repeat
    const int64_t ndim = shape.size();
    const int64_t out_numel = output.numel();

    // Pre-compute strides
    std::vector<int64_t> in_strides(ndim);
    std::vector<int64_t> out_strides(ndim);
    int64_t in_stride = 1;
    int64_t out_stride = 1;
    for (int64_t i = ndim - 1; i >= 0; --i) {
        in_strides[i] = in_stride;
        out_strides[i] = out_stride;
        in_stride *= shape[i];
        out_stride *= out_shape[i];
    }

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        // Copy shape and strides to device-accessible memory
        auto shape_buf = sycl::buffer<int64_t, 1>(shape.data(), sycl::range<1>(ndim));
        auto out_shape_buf = sycl::buffer<int64_t, 1>(out_shape.data(), sycl::range<1>(ndim));
        auto in_strides_buf = sycl::buffer<int64_t, 1>(in_strides.data(), sycl::range<1>(ndim));
        auto out_strides_buf = sycl::buffer<int64_t, 1>(out_strides.data(), sycl::range<1>(ndim));

        queue.submit([&](sycl::handler& h) {
            auto shape_acc = shape_buf.get_access<sycl::access::mode::read>(h);
            auto out_shape_acc = out_shape_buf.get_access<sycl::access::mode::read>(h);
            auto in_strides_acc = in_strides_buf.get_access<sycl::access::mode::read>(h);
            auto out_strides_acc = out_strides_buf.get_access<sycl::access::mode::read>(h);

            h.parallel_for<RepeatKernelFloat32>(sycl::range<1>(out_numel), [=](sycl::id<1> idx) {
                int64_t out_idx = idx[0];
                int64_t in_idx = 0;

                // Convert output index to input index using modular arithmetic
                for (int64_t d = 0; d < ndim; ++d) {
                    int64_t coord = (out_idx / out_strides_acc[d]) % out_shape_acc[d];
                    int64_t in_coord = coord % shape_acc[d];
                    in_idx += in_coord * in_strides_acc[d];
                }

                out_ptr[out_idx] = in_ptr[in_idx];
            });
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        auto shape_buf = sycl::buffer<int64_t, 1>(shape.data(), sycl::range<1>(ndim));
        auto out_shape_buf = sycl::buffer<int64_t, 1>(out_shape.data(), sycl::range<1>(ndim));
        auto in_strides_buf = sycl::buffer<int64_t, 1>(in_strides.data(), sycl::range<1>(ndim));
        auto out_strides_buf = sycl::buffer<int64_t, 1>(out_strides.data(), sycl::range<1>(ndim));

        queue.submit([&](sycl::handler& h) {
            auto shape_acc = shape_buf.get_access<sycl::access::mode::read>(h);
            auto out_shape_acc = out_shape_buf.get_access<sycl::access::mode::read>(h);
            auto in_strides_acc = in_strides_buf.get_access<sycl::access::mode::read>(h);
            auto out_strides_acc = out_strides_buf.get_access<sycl::access::mode::read>(h);

            h.parallel_for<RepeatKernelFloat64>(sycl::range<1>(out_numel), [=](sycl::id<1> idx) {
                int64_t out_idx = idx[0];
                int64_t in_idx = 0;

                for (int64_t d = 0; d < ndim; ++d) {
                    int64_t coord = (out_idx / out_strides_acc[d]) % out_shape_acc[d];
                    int64_t in_coord = coord % shape_acc[d];
                    in_idx += in_coord * in_strides_acc[d];
                }

                out_ptr[out_idx] = in_ptr[in_idx];
            });
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        auto shape_buf = sycl::buffer<int64_t, 1>(shape.data(), sycl::range<1>(ndim));
        auto out_shape_buf = sycl::buffer<int64_t, 1>(out_shape.data(), sycl::range<1>(ndim));
        auto in_strides_buf = sycl::buffer<int64_t, 1>(in_strides.data(), sycl::range<1>(ndim));
        auto out_strides_buf = sycl::buffer<int64_t, 1>(out_strides.data(), sycl::range<1>(ndim));

        queue.submit([&](sycl::handler& h) {
            auto shape_acc = shape_buf.get_access<sycl::access::mode::read>(h);
            auto out_shape_acc = out_shape_buf.get_access<sycl::access::mode::read>(h);
            auto in_strides_acc = in_strides_buf.get_access<sycl::access::mode::read>(h);
            auto out_strides_acc = out_strides_buf.get_access<sycl::access::mode::read>(h);

            h.parallel_for<RepeatKernelFloat16>(sycl::range<1>(out_numel), [=](sycl::id<1> idx) {
                int64_t out_idx = idx[0];
                int64_t in_idx = 0;

                for (int64_t d = 0; d < ndim; ++d) {
                    int64_t coord = (out_idx / out_strides_acc[d]) % out_shape_acc[d];
                    int64_t in_coord = coord % shape_acc[d];
                    in_idx += in_coord * in_strides_acc[d];
                }

                out_ptr[out_idx] = in_ptr[in_idx];
            });
        });
    }
    else {
        throw std::runtime_error("repeat: unsupported dtype");
    }

    return output;
}

// In-place operation kernel name classes
struct AddInplaceKernelFloat32 {};
struct AddInplaceKernelFloat64 {};
struct AddInplaceKernelFloat16 {};
struct AddInplaceKernelBFloat16 {};
struct SubInplaceKernelFloat32 {};
struct SubInplaceKernelFloat64 {};
struct SubInplaceKernelFloat16 {};
struct SubInplaceKernelBFloat16 {};
struct MulInplaceKernelFloat32 {};
struct MulInplaceKernelFloat64 {};
struct MulInplaceKernelFloat16 {};
struct MulInplaceKernelBFloat16 {};
struct DivInplaceKernelFloat32 {};
struct DivInplaceKernelFloat64 {};
struct DivInplaceKernelFloat16 {};
struct DivInplaceKernelBFloat16 {};
struct AddInplaceBcastKernelFloat32 {};
struct AddInplaceBcastKernelFloat64 {};
struct AddInplaceBcastKernelFloat16 {};
struct AddInplaceBcastKernelBFloat16 {};
struct SubInplaceBcastKernelFloat32 {};
struct SubInplaceBcastKernelFloat64 {};
struct SubInplaceBcastKernelFloat16 {};
struct SubInplaceBcastKernelBFloat16 {};
struct MulInplaceBcastKernelFloat32 {};
struct MulInplaceBcastKernelFloat64 {};
struct MulInplaceBcastKernelFloat16 {};
struct MulInplaceBcastKernelBFloat16 {};
struct DivInplaceBcastKernelFloat32 {};
struct DivInplaceBcastKernelFloat64 {};
struct DivInplaceBcastKernelFloat16 {};
struct DivInplaceBcastKernelBFloat16 {};

// MAX_BROADCAST_DIMS defined above (line ~170)

// Structure to hold broadcast strides for SYCL kernel capture
struct BroadcastStrides {
    int64_t self_strides[MAX_BROADCAST_DIMS];
    int64_t other_strides[MAX_BROADCAST_DIMS];
    int64_t shape[MAX_BROADCAST_DIMS];
    int64_t ndim;
};

/**
 * @brief Check if shapes are broadcastable for in-place operation.
 *
 * For in-place ops, the result shape must match self's shape.
 * Other's shape must be broadcastable to self's shape.
 */
inline auto validate_inplace_broadcast(
    std::span<const int64_t> self_shape,
    std::span<const int64_t> other_shape
) -> bool {
    int64_t self_ndim = self_shape.size();
    int64_t other_ndim = other_shape.size();

    // other cannot have more dims than self (would expand self's shape)
    if (other_ndim > self_ndim) return false;

    // Check each dimension from the right
    for (int64_t i = 0; i < other_ndim; ++i) {
        int64_t self_dim = self_shape[self_ndim - 1 - i];
        int64_t other_dim = other_shape[other_ndim - 1 - i];

        if (other_dim != 1 && other_dim != self_dim) {
            return false;  // Not broadcastable
        }
    }
    return true;
}

/**
 * @brief Compute broadcast strides for mapping self linear index to other linear index.
 */
inline auto compute_broadcast_strides(
    std::span<const int64_t> self_shape,
    std::span<const int64_t> other_shape
) -> BroadcastStrides {
    BroadcastStrides bs{};
    int64_t self_ndim = self_shape.size();
    int64_t other_ndim = other_shape.size();
    bs.ndim = self_ndim;

    // Compute self strides (row-major)
    int64_t stride = 1;
    for (int64_t i = self_ndim - 1; i >= 0; --i) {
        bs.self_strides[i] = stride;
        bs.shape[i] = self_shape[i];
        stride *= self_shape[i];
    }

    // Compute other strides with broadcasting (stride=0 for broadcast dims)
    stride = 1;
    for (int64_t i = self_ndim - 1; i >= 0; --i) {
        int64_t other_idx = i - (self_ndim - other_ndim);
        if (other_idx < 0) {
            // other doesn't have this dim - broadcast
            bs.other_strides[i] = 0;
        } else {
            int64_t other_dim = other_shape[other_idx];
            if (other_dim == 1) {
                bs.other_strides[i] = 0;  // broadcast
            } else {
                bs.other_strides[i] = stride;
            }
            stride *= other_dim;
        }
    }

    return bs;
}

/**
 * @brief Compute linear index into other tensor given linear index into self.
 * Uses broadcast strides: stride=0 means that dimension is broadcast.
 */
inline int64_t broadcast_other_idx(int64_t self_idx, const BroadcastStrides& bs) {
    int64_t other_idx = 0;
    int64_t remaining = self_idx;
    for (int64_t d = 0; d < bs.ndim; ++d) {
        int64_t coord = remaining / bs.self_strides[d];
        remaining %= bs.self_strides[d];
        other_idx += coord * bs.other_strides[d];
    }
    return other_idx;
}

// In-place add kernel with broadcasting support
auto add_inplace_kernel(Tensor& inout, const Tensor& other, sycl::queue& queue) -> Tensor {
    auto self_shape = inout.shape();
    auto other_shape = other.shape();
    bool same_shape = std::equal(self_shape.begin(), self_shape.end(), other_shape.begin(), other_shape.end());

    if (!same_shape) {
        if (!validate_inplace_broadcast(self_shape, other_shape)) {
            throw std::runtime_error("In-place add: shapes are not broadcastable");
        }

        BroadcastStrides bs = compute_broadcast_strides(self_shape, other_shape);
        const int64_t n = inout.numel();

        if (inout.dtype() == DType::Float32) {
            float* data = get_data_ptr<float>(inout);
            const float* other_ptr = get_data_ptr<const float>(other);
            queue.parallel_for<AddInplaceBcastKernelFloat32>(sycl::range<1>(n), [=](sycl::id<1> idx) {
                int64_t oidx = broadcast_other_idx(idx, bs);
                data[idx] += other_ptr[oidx];
            });
        } else if (inout.dtype() == DType::Float64) {
            double* data = get_data_ptr<double>(inout);
            const double* other_ptr = get_data_ptr<const double>(other);
            queue.parallel_for<AddInplaceBcastKernelFloat64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
                int64_t oidx = broadcast_other_idx(idx, bs);
                data[idx] += other_ptr[oidx];
            });
        } else if (inout.dtype() == DType::Float16) {
            sycl::half* data = get_data_ptr<sycl::half>(inout);
            const sycl::half* other_ptr = get_data_ptr<const sycl::half>(other);
            queue.parallel_for<AddInplaceBcastKernelFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
                int64_t oidx = broadcast_other_idx(idx, bs);
                data[idx] = sycl::half(float(data[idx]) + float(other_ptr[oidx]));
            });
        } else if (inout.dtype() == DType::BFloat16) {
            uint16_t* data = get_data_ptr<uint16_t>(inout);
            const uint16_t* other_ptr = get_data_ptr<const uint16_t>(other);
            queue.parallel_for<AddInplaceBcastKernelBFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
                int64_t oidx = broadcast_other_idx(idx, bs);
                data[idx] = f32_to_bf16(bf16_to_f32(data[idx]) + bf16_to_f32(other_ptr[oidx]));
            });
        } else {
            throw std::runtime_error("add_inplace: unsupported dtype");
        }
        return inout;
    }

    const int64_t n = inout.numel();

    if (inout.dtype() == DType::Float32) {
        float* data = get_data_ptr<float>(inout);
        const float* other_ptr = get_data_ptr<const float>(other);
        queue.parallel_for<AddInplaceKernelFloat32>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] += other_ptr[idx];
        });
    } else if (inout.dtype() == DType::Float64) {
        double* data = get_data_ptr<double>(inout);
        const double* other_ptr = get_data_ptr<const double>(other);
        queue.parallel_for<AddInplaceKernelFloat64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] += other_ptr[idx];
        });
    } else if (inout.dtype() == DType::Float16) {
        sycl::half* data = get_data_ptr<sycl::half>(inout);
        const sycl::half* other_ptr = get_data_ptr<const sycl::half>(other);
        queue.parallel_for<AddInplaceKernelFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] = sycl::half(float(data[idx]) + float(other_ptr[idx]));
        });
    } else if (inout.dtype() == DType::BFloat16) {
        uint16_t* data = get_data_ptr<uint16_t>(inout);
        const uint16_t* other_ptr = get_data_ptr<const uint16_t>(other);
        queue.parallel_for<AddInplaceKernelBFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] = f32_to_bf16(bf16_to_f32(data[idx]) + bf16_to_f32(other_ptr[idx]));
        });
    } else {
        throw std::runtime_error("add_inplace: unsupported dtype");
    }

    return inout;
}

// In-place sub kernel with broadcasting support
auto sub_inplace_kernel(Tensor& inout, const Tensor& other, sycl::queue& queue) -> Tensor {
    auto self_shape = inout.shape();
    auto other_shape = other.shape();
    bool same_shape = std::equal(self_shape.begin(), self_shape.end(), other_shape.begin(), other_shape.end());

    if (!same_shape) {
        if (!validate_inplace_broadcast(self_shape, other_shape)) {
            throw std::runtime_error("In-place sub: shapes are not broadcastable");
        }

        BroadcastStrides bs = compute_broadcast_strides(self_shape, other_shape);
        const int64_t n = inout.numel();

        if (inout.dtype() == DType::Float32) {
            float* data = get_data_ptr<float>(inout);
            const float* other_ptr = get_data_ptr<const float>(other);
            queue.parallel_for<SubInplaceBcastKernelFloat32>(sycl::range<1>(n), [=](sycl::id<1> idx) {
                int64_t oidx = broadcast_other_idx(idx, bs);
                data[idx] -= other_ptr[oidx];
            });
        } else if (inout.dtype() == DType::Float64) {
            double* data = get_data_ptr<double>(inout);
            const double* other_ptr = get_data_ptr<const double>(other);
            queue.parallel_for<SubInplaceBcastKernelFloat64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
                int64_t oidx = broadcast_other_idx(idx, bs);
                data[idx] -= other_ptr[oidx];
            });
        } else if (inout.dtype() == DType::Float16) {
            sycl::half* data = get_data_ptr<sycl::half>(inout);
            const sycl::half* other_ptr = get_data_ptr<const sycl::half>(other);
            queue.parallel_for<SubInplaceBcastKernelFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
                int64_t oidx = broadcast_other_idx(idx, bs);
                data[idx] = sycl::half(float(data[idx]) - float(other_ptr[oidx]));
            });
        } else if (inout.dtype() == DType::BFloat16) {
            uint16_t* data = get_data_ptr<uint16_t>(inout);
            const uint16_t* other_ptr = get_data_ptr<const uint16_t>(other);
            queue.parallel_for<SubInplaceBcastKernelBFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
                int64_t oidx = broadcast_other_idx(idx, bs);
                data[idx] = f32_to_bf16(bf16_to_f32(data[idx]) - bf16_to_f32(other_ptr[oidx]));
            });
        } else {
            throw std::runtime_error("sub_inplace: unsupported dtype");
        }
        return inout;
    }

    const int64_t n = inout.numel();

    if (inout.dtype() == DType::Float32) {
        float* data = get_data_ptr<float>(inout);
        const float* other_ptr = get_data_ptr<const float>(other);
        queue.parallel_for<SubInplaceKernelFloat32>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] -= other_ptr[idx];
        });
    } else if (inout.dtype() == DType::Float64) {
        double* data = get_data_ptr<double>(inout);
        const double* other_ptr = get_data_ptr<const double>(other);
        queue.parallel_for<SubInplaceKernelFloat64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] -= other_ptr[idx];
        });
    } else if (inout.dtype() == DType::Float16) {
        sycl::half* data = get_data_ptr<sycl::half>(inout);
        const sycl::half* other_ptr = get_data_ptr<const sycl::half>(other);
        queue.parallel_for<SubInplaceKernelFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] = sycl::half(float(data[idx]) - float(other_ptr[idx]));
        });
    } else if (inout.dtype() == DType::BFloat16) {
        uint16_t* data = get_data_ptr<uint16_t>(inout);
        const uint16_t* other_ptr = get_data_ptr<const uint16_t>(other);
        queue.parallel_for<SubInplaceKernelBFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] = f32_to_bf16(bf16_to_f32(data[idx]) - bf16_to_f32(other_ptr[idx]));
        });
    } else {
        throw std::runtime_error("sub_inplace: unsupported dtype");
    }

    return inout;
}

// In-place mul kernel with broadcasting support
auto mul_inplace_kernel(Tensor& inout, const Tensor& other, sycl::queue& queue) -> Tensor {
    auto self_shape = inout.shape();
    auto other_shape = other.shape();
    bool same_shape = std::equal(self_shape.begin(), self_shape.end(), other_shape.begin(), other_shape.end());

    if (!same_shape) {
        if (!validate_inplace_broadcast(self_shape, other_shape)) {
            throw std::runtime_error("In-place mul: shapes are not broadcastable");
        }

        BroadcastStrides bs = compute_broadcast_strides(self_shape, other_shape);
        const int64_t n = inout.numel();

        if (inout.dtype() == DType::Float32) {
            float* data = get_data_ptr<float>(inout);
            const float* other_ptr = get_data_ptr<const float>(other);
            queue.parallel_for<MulInplaceBcastKernelFloat32>(sycl::range<1>(n), [=](sycl::id<1> idx) {
                int64_t oidx = broadcast_other_idx(idx, bs);
                data[idx] *= other_ptr[oidx];
            });
        } else if (inout.dtype() == DType::Float64) {
            double* data = get_data_ptr<double>(inout);
            const double* other_ptr = get_data_ptr<const double>(other);
            queue.parallel_for<MulInplaceBcastKernelFloat64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
                int64_t oidx = broadcast_other_idx(idx, bs);
                data[idx] *= other_ptr[oidx];
            });
        } else if (inout.dtype() == DType::Float16) {
            sycl::half* data = get_data_ptr<sycl::half>(inout);
            const sycl::half* other_ptr = get_data_ptr<const sycl::half>(other);
            queue.parallel_for<MulInplaceBcastKernelFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
                int64_t oidx = broadcast_other_idx(idx, bs);
                data[idx] = sycl::half(float(data[idx]) * float(other_ptr[oidx]));
            });
        } else if (inout.dtype() == DType::BFloat16) {
            uint16_t* data = get_data_ptr<uint16_t>(inout);
            const uint16_t* other_ptr = get_data_ptr<const uint16_t>(other);
            queue.parallel_for<MulInplaceBcastKernelBFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
                int64_t oidx = broadcast_other_idx(idx, bs);
                data[idx] = f32_to_bf16(bf16_to_f32(data[idx]) * bf16_to_f32(other_ptr[oidx]));
            });
        } else {
            throw std::runtime_error("mul_inplace: unsupported dtype");
        }
        return inout;
    }

    const int64_t n = inout.numel();

    if (inout.dtype() == DType::Float32) {
        float* data = get_data_ptr<float>(inout);
        const float* other_ptr = get_data_ptr<const float>(other);
        queue.parallel_for<MulInplaceKernelFloat32>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] *= other_ptr[idx];
        });
    } else if (inout.dtype() == DType::Float64) {
        double* data = get_data_ptr<double>(inout);
        const double* other_ptr = get_data_ptr<const double>(other);
        queue.parallel_for<MulInplaceKernelFloat64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] *= other_ptr[idx];
        });
    } else if (inout.dtype() == DType::Float16) {
        sycl::half* data = get_data_ptr<sycl::half>(inout);
        const sycl::half* other_ptr = get_data_ptr<const sycl::half>(other);
        queue.parallel_for<MulInplaceKernelFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] = sycl::half(float(data[idx]) * float(other_ptr[idx]));
        });
    } else if (inout.dtype() == DType::BFloat16) {
        uint16_t* data = get_data_ptr<uint16_t>(inout);
        const uint16_t* other_ptr = get_data_ptr<const uint16_t>(other);
        queue.parallel_for<MulInplaceKernelBFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] = f32_to_bf16(bf16_to_f32(data[idx]) * bf16_to_f32(other_ptr[idx]));
        });
    } else {
        throw std::runtime_error("mul_inplace: unsupported dtype");
    }

    return inout;
}

// In-place div kernel with broadcasting support
auto div_inplace_kernel(Tensor& inout, const Tensor& other, sycl::queue& queue) -> Tensor {
    auto self_shape = inout.shape();
    auto other_shape = other.shape();
    bool same_shape = std::equal(self_shape.begin(), self_shape.end(), other_shape.begin(), other_shape.end());

    if (!same_shape) {
        if (!validate_inplace_broadcast(self_shape, other_shape)) {
            throw std::runtime_error("In-place div: shapes are not broadcastable");
        }

        BroadcastStrides bs = compute_broadcast_strides(self_shape, other_shape);
        const int64_t n = inout.numel();

        if (inout.dtype() == DType::Float32) {
            float* data = get_data_ptr<float>(inout);
            const float* other_ptr = get_data_ptr<const float>(other);
            queue.parallel_for<DivInplaceBcastKernelFloat32>(sycl::range<1>(n), [=](sycl::id<1> idx) {
                int64_t oidx = broadcast_other_idx(idx, bs);
                data[idx] /= other_ptr[oidx];
            });
        } else if (inout.dtype() == DType::Float64) {
            double* data = get_data_ptr<double>(inout);
            const double* other_ptr = get_data_ptr<const double>(other);
            queue.parallel_for<DivInplaceBcastKernelFloat64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
                int64_t oidx = broadcast_other_idx(idx, bs);
                data[idx] /= other_ptr[oidx];
            });
        } else if (inout.dtype() == DType::Float16) {
            sycl::half* data = get_data_ptr<sycl::half>(inout);
            const sycl::half* other_ptr = get_data_ptr<const sycl::half>(other);
            queue.parallel_for<DivInplaceBcastKernelFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
                int64_t oidx = broadcast_other_idx(idx, bs);
                data[idx] = sycl::half(float(data[idx]) / float(other_ptr[oidx]));
            });
        } else if (inout.dtype() == DType::BFloat16) {
            uint16_t* data = get_data_ptr<uint16_t>(inout);
            const uint16_t* other_ptr = get_data_ptr<const uint16_t>(other);
            queue.parallel_for<DivInplaceBcastKernelBFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
                int64_t oidx = broadcast_other_idx(idx, bs);
                data[idx] = f32_to_bf16(bf16_to_f32(data[idx]) / bf16_to_f32(other_ptr[oidx]));
            });
        } else {
            throw std::runtime_error("div_inplace: unsupported dtype");
        }
        return inout;
    }

    const int64_t n = inout.numel();

    if (inout.dtype() == DType::Float32) {
        float* data = get_data_ptr<float>(inout);
        const float* other_ptr = get_data_ptr<const float>(other);
        queue.parallel_for<DivInplaceKernelFloat32>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] /= other_ptr[idx];
        });
    } else if (inout.dtype() == DType::Float64) {
        double* data = get_data_ptr<double>(inout);
        const double* other_ptr = get_data_ptr<const double>(other);
        queue.parallel_for<DivInplaceKernelFloat64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] /= other_ptr[idx];
        });
    } else if (inout.dtype() == DType::Float16) {
        sycl::half* data = get_data_ptr<sycl::half>(inout);
        const sycl::half* other_ptr = get_data_ptr<const sycl::half>(other);
        queue.parallel_for<DivInplaceKernelFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] = sycl::half(float(data[idx]) / float(other_ptr[idx]));
        });
    } else if (inout.dtype() == DType::BFloat16) {
        uint16_t* data = get_data_ptr<uint16_t>(inout);
        const uint16_t* other_ptr = get_data_ptr<const uint16_t>(other);
        queue.parallel_for<DivInplaceKernelBFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] = f32_to_bf16(bf16_to_f32(data[idx]) / bf16_to_f32(other_ptr[idx]));
        });
    } else {
        throw std::runtime_error("div_inplace: unsupported dtype");
    }

    return inout;
}

// ============================================================================
// HasInfNan kernel - check if tensor contains Inf or NaN
// ============================================================================
class HasInfNanKernelF32;
class HasInfNanKernelF64;

auto has_inf_nan_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    // Returns a scalar Bool tensor: true if any Inf or NaN
    Tensor output({1}, DType::Bool, input.device());
    bool* out_ptr = static_cast<bool*>(const_cast<void*>(output.data_ptr()));
    int64_t numel = input.numel();

    if (numel == 0) {
        bool false_val = false;
        queue.memcpy(out_ptr, &false_val, sizeof(bool)).wait();
        return output;
    }

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = static_cast<const float*>(input.data_ptr());
        auto flag_buf = sycl::malloc_shared<int32_t>(1, queue);
        flag_buf[0] = 0;

        queue.parallel_for(sycl::range<1>(numel), sycl::reduction(flag_buf, sycl::bit_or<int32_t>()),
                          [=](sycl::id<1> i, auto& f) {
            float val = in_ptr[i];
            if (sycl::isinf(val) || sycl::isnan(val)) f |= 1;
        });
        queue.wait();
        bool found = (flag_buf[0] != 0);
        queue.memcpy(out_ptr, &found, sizeof(bool)).wait();
        sycl::free(flag_buf, queue);
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = static_cast<const double*>(input.data_ptr());
        auto flag_buf = sycl::malloc_shared<int32_t>(1, queue);
        flag_buf[0] = 0;

        queue.parallel_for(sycl::range<1>(numel), sycl::reduction(flag_buf, sycl::bit_or<int32_t>()),
                          [=](sycl::id<1> i, auto& f) {
            double val = in_ptr[i];
            if (sycl::isinf(val) || sycl::isnan(val)) f |= 1;
        });
        queue.wait();
        bool found = (flag_buf[0] != 0);
        queue.memcpy(out_ptr, &found, sizeof(bool)).wait();
        sycl::free(flag_buf, queue);
    } else {
        // Integer/bool types never have inf/nan
        bool false_val = false;
        queue.memcpy(out_ptr, &false_val, sizeof(bool)).wait();
    }

    return output;
}

// ============================================================================
// CumSum kernel - cumulative sum along a dimension
// ============================================================================
auto cumsum_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> Tensor {
    auto shape_span = input.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;

    Tensor output(shape, input.dtype(), input.device());
    int64_t numel = input.numel();

    if (numel == 0) return output;

    // Device-side per-line parallel scan along dim
    int64_t outer_size = 1, inner_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];
    int64_t dim_size = shape[dim];

    int64_t num_lines = outer_size * inner_size;

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for(sycl::range<1>(num_lines), [=](sycl::id<1> idx) {
            int64_t o = idx / inner_size;
            int64_t i = idx % inner_size;
            float sum = 0.0f;
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t flat = o * dim_size * inner_size + d * inner_size + i;
                sum += in_ptr[flat];
                out_ptr[flat] = sum;
            }
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for(sycl::range<1>(num_lines), [=](sycl::id<1> idx) {
            int64_t o = idx / inner_size;
            int64_t i = idx % inner_size;
            double sum = 0.0;
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t flat = o * dim_size * inner_size + d * inner_size + i;
                sum += in_ptr[flat];
                out_ptr[flat] = sum;
            }
        }).wait();
    } else if (input.dtype() == DType::Int32) {
        const int32_t* in_ptr = get_data_ptr<const int32_t>(input);
        int32_t* out_ptr = get_data_ptr<int32_t>(output);
        queue.parallel_for(sycl::range<1>(num_lines), [=](sycl::id<1> idx) {
            int64_t o = idx / inner_size;
            int64_t i = idx % inner_size;
            int32_t sum = 0;
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t flat = o * dim_size * inner_size + d * inner_size + i;
                sum += in_ptr[flat];
                out_ptr[flat] = sum;
            }
        }).wait();
    } else if (input.dtype() == DType::Int64) {
        const int64_t* in_ptr = get_data_ptr<const int64_t>(input);
        int64_t* out_ptr = get_data_ptr<int64_t>(output);
        queue.parallel_for(sycl::range<1>(num_lines), [=](sycl::id<1> idx) {
            int64_t o = idx / inner_size;
            int64_t i = idx % inner_size;
            int64_t sum = 0;
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t flat = o * dim_size * inner_size + d * inner_size + i;
                sum += in_ptr[flat];
                out_ptr[flat] = sum;
            }
        }).wait();
    } else {
        throw std::runtime_error("cumsum: unsupported dtype");
    }

    return output;
}

// ============================================================================
// CumProd kernel - cumulative product along a dimension
// ============================================================================
auto cumprod_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> Tensor {
    auto shape_span = input.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;

    Tensor output(shape, input.dtype(), input.device());
    int64_t numel = input.numel();

    if (numel == 0) return output;

    int64_t outer_size = 1, inner_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];
    int64_t dim_size = shape[dim];

    int64_t num_lines = outer_size * inner_size;

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for(sycl::range<1>(num_lines), [=](sycl::id<1> idx) {
            int64_t o = idx / inner_size;
            int64_t i = idx % inner_size;
            float prod = 1.0f;
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t flat = o * dim_size * inner_size + d * inner_size + i;
                prod *= in_ptr[flat];
                out_ptr[flat] = prod;
            }
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for(sycl::range<1>(num_lines), [=](sycl::id<1> idx) {
            int64_t o = idx / inner_size;
            int64_t i = idx % inner_size;
            double prod = 1.0;
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t flat = o * dim_size * inner_size + d * inner_size + i;
                prod *= in_ptr[flat];
                out_ptr[flat] = prod;
            }
        }).wait();
    } else if (input.dtype() == DType::Int32) {
        const int32_t* in_ptr = get_data_ptr<const int32_t>(input);
        int32_t* out_ptr = get_data_ptr<int32_t>(output);
        queue.parallel_for(sycl::range<1>(num_lines), [=](sycl::id<1> idx) {
            int64_t o = idx / inner_size;
            int64_t i = idx % inner_size;
            int32_t prod = 1;
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t flat = o * dim_size * inner_size + d * inner_size + i;
                prod *= in_ptr[flat];
                out_ptr[flat] = prod;
            }
        }).wait();
    } else if (input.dtype() == DType::Int64) {
        const int64_t* in_ptr = get_data_ptr<const int64_t>(input);
        int64_t* out_ptr = get_data_ptr<int64_t>(output);
        queue.parallel_for(sycl::range<1>(num_lines), [=](sycl::id<1> idx) {
            int64_t o = idx / inner_size;
            int64_t i = idx % inner_size;
            int64_t prod = 1;
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t flat = o * dim_size * inner_size + d * inner_size + i;
                prod *= in_ptr[flat];
                out_ptr[flat] = prod;
            }
        }).wait();
    } else {
        throw std::runtime_error("cumprod: unsupported dtype");
    }

    return output;
}

// ============================================================================
// Extended Math Functions (log2, log10, log1p, exp2, expm1, erf, erfc)
// ============================================================================

auto log2_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());
    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<Log2KernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::log2(in_ptr[idx]);
        });
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<Log2KernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::log2(in_ptr[idx]);
        });
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<Log2KernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::log2(static_cast<float>(in_ptr[idx])));
        });
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<Log2KernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::log2(bf16_to_f32(in_ptr[idx])));
        });
    } else {
        throw std::runtime_error("log2: unsupported dtype");
    }
    return output;
}

auto log10_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());
    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<Log10KernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::log10(in_ptr[idx]);
        });
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<Log10KernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::log10(in_ptr[idx]);
        });
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<Log10KernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::log10(static_cast<float>(in_ptr[idx])));
        });
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<Log10KernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::log10(bf16_to_f32(in_ptr[idx])));
        });
    } else {
        throw std::runtime_error("log10: unsupported dtype");
    }
    return output;
}

auto log1p_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());
    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<Log1pKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::log1p(in_ptr[idx]);
        });
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<Log1pKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::log1p(in_ptr[idx]);
        });
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<Log1pKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::log1p(static_cast<float>(in_ptr[idx])));
        });
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<Log1pKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::log1p(bf16_to_f32(in_ptr[idx])));
        });
    } else {
        throw std::runtime_error("log1p: unsupported dtype");
    }
    return output;
}

auto exp2_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());
    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<Exp2KernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::exp2(in_ptr[idx]);
        });
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<Exp2KernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::exp2(in_ptr[idx]);
        });
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<Exp2KernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::exp2(static_cast<float>(in_ptr[idx])));
        });
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<Exp2KernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::exp2(bf16_to_f32(in_ptr[idx])));
        });
    } else {
        throw std::runtime_error("exp2: unsupported dtype");
    }
    return output;
}

auto expm1_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());
    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<Expm1KernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::expm1(in_ptr[idx]);
        });
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<Expm1KernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::expm1(in_ptr[idx]);
        });
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<Expm1KernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::expm1(static_cast<float>(in_ptr[idx])));
        });
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<Expm1KernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::expm1(bf16_to_f32(in_ptr[idx])));
        });
    } else {
        throw std::runtime_error("expm1: unsupported dtype");
    }
    return output;
}

auto erf_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());
    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<ErfKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::erf(in_ptr[idx]);
        });
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<ErfKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::erf(in_ptr[idx]);
        });
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<ErfKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::erf(static_cast<float>(in_ptr[idx])));
        });
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<ErfKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::erf(bf16_to_f32(in_ptr[idx])));
        });
    } else {
        throw std::runtime_error("erf: unsupported dtype");
    }
    return output;
}

auto erfc_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());
    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<ErfcKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::erfc(in_ptr[idx]);
        });
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<ErfcKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::erfc(in_ptr[idx]);
        });
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<ErfcKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::erfc(static_cast<float>(in_ptr[idx])));
        });
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<ErfcKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::erfc(bf16_to_f32(in_ptr[idx])));
        });
    } else {
        throw std::runtime_error("erfc: unsupported dtype");
    }
    return output;
}

// ============================================================================
// Bool Predicates (isnan, isinf, isfinite)
// ============================================================================

auto isnan_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  DType::Bool, input.device());
    const int64_t numel = input.numel();
    uint8_t* out_ptr = get_data_ptr<uint8_t>(output);

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        queue.parallel_for<IsNanKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(sycl::isnan(in_ptr[idx]) ? 1 : 0);
        });
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        queue.parallel_for<IsNanKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(sycl::isnan(in_ptr[idx]) ? 1 : 0);
        });
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        queue.parallel_for<IsNanKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(sycl::isnan(static_cast<float>(in_ptr[idx])) ? 1 : 0);
        });
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        queue.parallel_for<IsNanKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(sycl::isnan(bf16_to_f32(in_ptr[idx])) ? 1 : 0);
        });
    } else {
        // Integer types are never NaN
        queue.memset(out_ptr, 0, numel * sizeof(uint8_t));
    }
    return output;
}

auto isinf_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  DType::Bool, input.device());
    const int64_t numel = input.numel();
    uint8_t* out_ptr = get_data_ptr<uint8_t>(output);

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        queue.parallel_for<IsInfKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(sycl::isinf(in_ptr[idx]) ? 1 : 0);
        });
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        queue.parallel_for<IsInfKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(sycl::isinf(in_ptr[idx]) ? 1 : 0);
        });
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        queue.parallel_for<IsInfKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(sycl::isinf(static_cast<float>(in_ptr[idx])) ? 1 : 0);
        });
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        queue.parallel_for<IsInfKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(sycl::isinf(bf16_to_f32(in_ptr[idx])) ? 1 : 0);
        });
    } else {
        // Integer types are never infinite
        queue.memset(out_ptr, 0, numel * sizeof(uint8_t));
    }
    return output;
}

auto isfinite_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  DType::Bool, input.device());
    const int64_t numel = input.numel();
    uint8_t* out_ptr = get_data_ptr<uint8_t>(output);

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        queue.parallel_for<IsFiniteKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(sycl::isfinite(in_ptr[idx]) ? 1 : 0);
        });
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        queue.parallel_for<IsFiniteKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(sycl::isfinite(in_ptr[idx]) ? 1 : 0);
        });
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        queue.parallel_for<IsFiniteKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(sycl::isfinite(static_cast<float>(in_ptr[idx])) ? 1 : 0);
        });
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        queue.parallel_for<IsFiniteKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(sycl::isfinite(bf16_to_f32(in_ptr[idx])) ? 1 : 0);
        });
    } else {
        // Integer types are always finite
        queue.memset(out_ptr, 1, numel * sizeof(uint8_t));
    }
    return output;
}

// ============================================================================
// Binary Math (fmod, remainder)
// ============================================================================

auto fmod_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument("fmod: input dtypes must match");
    }

    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  a.dtype(), a.device());
    const int64_t numel = a.numel();

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<FmodKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fmod(a_ptr[idx], b_ptr[idx]);
        });
    } else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<FmodKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fmod(a_ptr[idx], b_ptr[idx]);
        });
    } else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<FmodKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::fmod(static_cast<float>(a_ptr[idx]), static_cast<float>(b_ptr[idx])));
        });
    } else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<FmodKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::fmod(bf16_to_f32(a_ptr[idx]), bf16_to_f32(b_ptr[idx])));
        });
    } else {
        throw std::runtime_error("fmod: unsupported dtype");
    }
    return output;
}

auto remainder_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument("remainder: input dtypes must match");
    }

    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  a.dtype(), a.device());
    const int64_t numel = a.numel();

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<RemainderKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::remainder(a_ptr[idx], b_ptr[idx]);
        });
    } else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<RemainderKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::remainder(a_ptr[idx], b_ptr[idx]);
        });
    } else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<RemainderKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::remainder(static_cast<float>(a_ptr[idx]), static_cast<float>(b_ptr[idx])));
        });
    } else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<RemainderKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::remainder(bf16_to_f32(a_ptr[idx]), bf16_to_f32(b_ptr[idx])));
        });
    } else {
        throw std::runtime_error("remainder: unsupported dtype");
    }
    return output;
}

// ============================================================================
// Ternary: Lerp (linear interpolation)
// ============================================================================

auto lerp_kernel(const Tensor& start, const Tensor& end, const Tensor& weight, sycl::queue& queue) -> Tensor {
    if (start.dtype() != end.dtype() || start.dtype() != weight.dtype()) {
        throw std::invalid_argument("lerp: input dtypes must match");
    }

    Tensor output(std::vector<int64_t>(start.shape().begin(), start.shape().end()),
                  start.dtype(), start.device());
    const int64_t numel = start.numel();

    if (start.dtype() == DType::Float32) {
        const float* s_ptr = get_data_ptr<const float>(start);
        const float* e_ptr = get_data_ptr<const float>(end);
        const float* w_ptr = get_data_ptr<const float>(weight);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<LerpKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = s_ptr[idx] + w_ptr[idx] * (e_ptr[idx] - s_ptr[idx]);
        });
    } else if (start.dtype() == DType::Float64) {
        const double* s_ptr = get_data_ptr<const double>(start);
        const double* e_ptr = get_data_ptr<const double>(end);
        const double* w_ptr = get_data_ptr<const double>(weight);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<LerpKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = s_ptr[idx] + w_ptr[idx] * (e_ptr[idx] - s_ptr[idx]);
        });
    } else if (start.dtype() == DType::Float16) {
        const sycl::half* s_ptr = get_data_ptr<const sycl::half>(start);
        const sycl::half* e_ptr = get_data_ptr<const sycl::half>(end);
        const sycl::half* w_ptr = get_data_ptr<const sycl::half>(weight);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<LerpKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float s = static_cast<float>(s_ptr[idx]);
            float e = static_cast<float>(e_ptr[idx]);
            float w = static_cast<float>(w_ptr[idx]);
            out_ptr[idx] = sycl::half(s + w * (e - s));
        });
    } else if (start.dtype() == DType::BFloat16) {
        const uint16_t* s_ptr = get_data_ptr<const uint16_t>(start);
        const uint16_t* e_ptr = get_data_ptr<const uint16_t>(end);
        const uint16_t* w_ptr = get_data_ptr<const uint16_t>(weight);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<LerpKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float s = bf16_to_f32(s_ptr[idx]);
            float e = bf16_to_f32(e_ptr[idx]);
            float w = bf16_to_f32(w_ptr[idx]);
            out_ptr[idx] = f32_to_bf16(s + w * (e - s));
        });
    } else {
        throw std::runtime_error("lerp: unsupported dtype");
    }
    return output;
}

// ============================================================================
// Logical Operations
// ============================================================================

auto logical_and_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  DType::Bool, a.device());
    const int64_t numel = a.numel();
    uint8_t* out_ptr = get_data_ptr<uint8_t>(output);

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        queue.parallel_for<LogicalAndKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>((a_ptr[idx] != 0.0f && b_ptr[idx] != 0.0f) ? 1 : 0);
        });
    } else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        queue.parallel_for<LogicalAndKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>((a_ptr[idx] != 0.0 && b_ptr[idx] != 0.0) ? 1 : 0);
        });
    } else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        queue.parallel_for<LogicalAndKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>((static_cast<float>(a_ptr[idx]) != 0.0f && static_cast<float>(b_ptr[idx]) != 0.0f) ? 1 : 0);
        });
    } else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        queue.parallel_for<LogicalAndKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>((bf16_to_f32(a_ptr[idx]) != 0.0f && bf16_to_f32(b_ptr[idx]) != 0.0f) ? 1 : 0);
        });
    } else if (a.dtype() == DType::Int32) {
        const int32_t* a_ptr = get_data_ptr<const int32_t>(a);
        const int32_t* b_ptr = get_data_ptr<const int32_t>(b);
        queue.parallel_for<LogicalAndKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>((a_ptr[idx] != 0 && b_ptr[idx] != 0) ? 1 : 0);
        });
    } else if (a.dtype() == DType::Int64) {
        const int64_t* a_ptr = get_data_ptr<const int64_t>(a);
        const int64_t* b_ptr = get_data_ptr<const int64_t>(b);
        queue.parallel_for<LogicalAndKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>((a_ptr[idx] != 0 && b_ptr[idx] != 0) ? 1 : 0);
        });
    } else {
        throw std::runtime_error("logical_and: unsupported dtype");
    }
    return output;
}

auto logical_or_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  DType::Bool, a.device());
    const int64_t numel = a.numel();
    uint8_t* out_ptr = get_data_ptr<uint8_t>(output);

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        queue.parallel_for<LogicalOrKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>((a_ptr[idx] != 0.0f || b_ptr[idx] != 0.0f) ? 1 : 0);
        });
    } else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        queue.parallel_for<LogicalOrKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>((a_ptr[idx] != 0.0 || b_ptr[idx] != 0.0) ? 1 : 0);
        });
    } else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        queue.parallel_for<LogicalOrKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>((static_cast<float>(a_ptr[idx]) != 0.0f || static_cast<float>(b_ptr[idx]) != 0.0f) ? 1 : 0);
        });
    } else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        queue.parallel_for<LogicalOrKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>((bf16_to_f32(a_ptr[idx]) != 0.0f || bf16_to_f32(b_ptr[idx]) != 0.0f) ? 1 : 0);
        });
    } else if (a.dtype() == DType::Int32) {
        const int32_t* a_ptr = get_data_ptr<const int32_t>(a);
        const int32_t* b_ptr = get_data_ptr<const int32_t>(b);
        queue.parallel_for<LogicalOrKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>((a_ptr[idx] != 0 || b_ptr[idx] != 0) ? 1 : 0);
        });
    } else if (a.dtype() == DType::Int64) {
        const int64_t* a_ptr = get_data_ptr<const int64_t>(a);
        const int64_t* b_ptr = get_data_ptr<const int64_t>(b);
        queue.parallel_for<LogicalOrKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>((a_ptr[idx] != 0 || b_ptr[idx] != 0) ? 1 : 0);
        });
    } else {
        throw std::runtime_error("logical_or: unsupported dtype");
    }
    return output;
}

auto logical_not_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  DType::Bool, input.device());
    const int64_t numel = input.numel();
    uint8_t* out_ptr = get_data_ptr<uint8_t>(output);

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        queue.parallel_for<LogicalNotKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(in_ptr[idx] == 0.0f ? 1 : 0);
        });
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        queue.parallel_for<LogicalNotKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(in_ptr[idx] == 0.0 ? 1 : 0);
        });
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        queue.parallel_for<LogicalNotKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(static_cast<float>(in_ptr[idx]) == 0.0f ? 1 : 0);
        });
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        queue.parallel_for<LogicalNotKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(bf16_to_f32(in_ptr[idx]) == 0.0f ? 1 : 0);
        });
    } else if (input.dtype() == DType::Int32) {
        const int32_t* in_ptr = get_data_ptr<const int32_t>(input);
        queue.parallel_for<LogicalNotKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(in_ptr[idx] == 0 ? 1 : 0);
        });
    } else if (input.dtype() == DType::Int64) {
        const int64_t* in_ptr = get_data_ptr<const int64_t>(input);
        queue.parallel_for<LogicalNotKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(in_ptr[idx] == 0 ? 1 : 0);
        });
    } else if (input.dtype() == DType::Bool) {
        const uint8_t* in_ptr = get_data_ptr<const uint8_t>(input);
        queue.parallel_for<LogicalNotKernelBool>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(in_ptr[idx] == 0 ? 1 : 0);
        });
    } else {
        throw std::runtime_error("logical_not: unsupported dtype");
    }
    return output;
}

auto logical_xor_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  DType::Bool, a.device());
    const int64_t numel = a.numel();
    uint8_t* out_ptr = get_data_ptr<uint8_t>(output);

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        queue.parallel_for<LogicalXorKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            bool va = a_ptr[idx] != 0.0f;
            bool vb = b_ptr[idx] != 0.0f;
            out_ptr[idx] = static_cast<uint8_t>((va != vb) ? 1 : 0);
        });
    } else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        queue.parallel_for<LogicalXorKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            bool va = a_ptr[idx] != 0.0;
            bool vb = b_ptr[idx] != 0.0;
            out_ptr[idx] = static_cast<uint8_t>((va != vb) ? 1 : 0);
        });
    } else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        queue.parallel_for<LogicalXorKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            bool va = static_cast<float>(a_ptr[idx]) != 0.0f;
            bool vb = static_cast<float>(b_ptr[idx]) != 0.0f;
            out_ptr[idx] = static_cast<uint8_t>((va != vb) ? 1 : 0);
        });
    } else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        queue.parallel_for<LogicalXorKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            bool va = bf16_to_f32(a_ptr[idx]) != 0.0f;
            bool vb = bf16_to_f32(b_ptr[idx]) != 0.0f;
            out_ptr[idx] = static_cast<uint8_t>((va != vb) ? 1 : 0);
        });
    } else if (a.dtype() == DType::Int32) {
        const int32_t* a_ptr = get_data_ptr<const int32_t>(a);
        const int32_t* b_ptr = get_data_ptr<const int32_t>(b);
        queue.parallel_for<LogicalXorKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            bool va = a_ptr[idx] != 0;
            bool vb = b_ptr[idx] != 0;
            out_ptr[idx] = static_cast<uint8_t>((va != vb) ? 1 : 0);
        });
    } else if (a.dtype() == DType::Int64) {
        const int64_t* a_ptr = get_data_ptr<const int64_t>(a);
        const int64_t* b_ptr = get_data_ptr<const int64_t>(b);
        queue.parallel_for<LogicalXorKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            bool va = a_ptr[idx] != 0;
            bool vb = b_ptr[idx] != 0;
            out_ptr[idx] = static_cast<uint8_t>((va != vb) ? 1 : 0);
        });
    } else {
        throw std::runtime_error("logical_xor: unsupported dtype");
    }
    return output;
}

// ============================================================================
// Element-wise Minimum / Maximum
// ============================================================================

auto minimum_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument("minimum: input dtypes must match");
    }

    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  a.dtype(), a.device());
    const int64_t numel = a.numel();

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<MinimumKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fmin(a_ptr[idx], b_ptr[idx]);
        });
    } else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<MinimumKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fmin(a_ptr[idx], b_ptr[idx]);
        });
    } else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<MinimumKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::fmin(static_cast<float>(a_ptr[idx]), static_cast<float>(b_ptr[idx])));
        });
    } else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<MinimumKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::fmin(bf16_to_f32(a_ptr[idx]), bf16_to_f32(b_ptr[idx])));
        });
    } else {
        throw std::runtime_error("minimum: unsupported dtype");
    }
    return output;
}

auto maximum_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument("maximum: input dtypes must match");
    }

    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  a.dtype(), a.device());
    const int64_t numel = a.numel();

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<MaximumKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fmax(a_ptr[idx], b_ptr[idx]);
        });
    } else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<MaximumKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fmax(a_ptr[idx], b_ptr[idx]);
        });
    } else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<MaximumKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::fmax(static_cast<float>(a_ptr[idx]), static_cast<float>(b_ptr[idx])));
        });
    } else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<MaximumKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::fmax(bf16_to_f32(a_ptr[idx]), bf16_to_f32(b_ptr[idx])));
        });
    } else {
        throw std::runtime_error("maximum: unsupported dtype");
    }
    return output;
}

// =========================================================================
// Complex Number Operations
// =========================================================================

auto conj_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());

    if (input.dtype() == DType::Complex64) {
        Tensor result(shape, DType::Complex64, input.device());
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(result);
        queue.parallel_for<ConjKernelComplex64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            out_ptr[2 * idx]     =  in_ptr[2 * idx];
            out_ptr[2 * idx + 1] = -in_ptr[2 * idx + 1];
        });
        return result;
    } else if (input.dtype() == DType::Complex128) {
        Tensor result(shape, DType::Complex128, input.device());
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(result);
        queue.parallel_for<ConjKernelComplex128>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            out_ptr[2 * idx]     =  in_ptr[2 * idx];
            out_ptr[2 * idx + 1] = -in_ptr[2 * idx + 1];
        });
        return result;
    }
    // For real dtypes, conjugate is identity
    Tensor result(shape, input.dtype(), input.device());
    queue.memcpy(result.data_ptr(), input.data_ptr(), n * dtype_size(input.dtype()));
    return result;
}

auto real_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());

    if (input.dtype() == DType::Complex64) {
        Tensor result(shape, DType::Float32, input.device());
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(result);
        queue.parallel_for<RealKernelComplex64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            out_ptr[idx] = in_ptr[2 * idx];
        });
        return result;
    } else if (input.dtype() == DType::Complex128) {
        Tensor result(shape, DType::Float64, input.device());
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(result);
        queue.parallel_for<RealKernelComplex128>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            out_ptr[idx] = in_ptr[2 * idx];
        });
        return result;
    }
    // For real dtypes, real() is identity
    Tensor result(shape, input.dtype(), input.device());
    queue.memcpy(result.data_ptr(), input.data_ptr(), n * dtype_size(input.dtype()));
    return result;
}

auto imag_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());

    if (input.dtype() == DType::Complex64) {
        Tensor result(shape, DType::Float32, input.device());
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(result);
        queue.parallel_for<ImagKernelComplex64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            out_ptr[idx] = in_ptr[2 * idx + 1];
        });
        return result;
    } else if (input.dtype() == DType::Complex128) {
        Tensor result(shape, DType::Float64, input.device());
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(result);
        queue.parallel_for<ImagKernelComplex128>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            out_ptr[idx] = in_ptr[2 * idx + 1];
        });
        return result;
    }
    // For real dtypes, imaginary part is zero
    Tensor result(shape, input.dtype(), input.device());
    queue.memset(result.data_ptr(), 0, n * dtype_size(input.dtype()));
    return result;
}

auto angle_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());

    if (input.dtype() == DType::Complex64) {
        Tensor result(shape, DType::Float32, input.device());
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(result);
        queue.parallel_for<AngleKernelComplex64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::atan2(in_ptr[2 * idx + 1], in_ptr[2 * idx]);
        });
        return result;
    } else if (input.dtype() == DType::Complex128) {
        Tensor result(shape, DType::Float64, input.device());
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(result);
        queue.parallel_for<AngleKernelComplex128>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::atan2(in_ptr[2 * idx + 1], in_ptr[2 * idx]);
        });
        return result;
    } else if (input.dtype() == DType::Float32) {
        Tensor result(shape, DType::Float32, input.device());
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(result);
        queue.parallel_for<AngleKernelFloat32>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::atan2(0.0f, in_ptr[idx]);
        });
        return result;
    } else if (input.dtype() == DType::Float64) {
        Tensor result(shape, DType::Float64, input.device());
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(result);
        queue.parallel_for<AngleKernelFloat64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::atan2(0.0, in_ptr[idx]);
        });
        return result;
    }
    throw std::runtime_error("angle: unsupported dtype");
}

auto polar_kernel(const Tensor& abs_t, const Tensor& angle_t, sycl::queue& queue) -> Tensor {
    if (abs_t.dtype() != angle_t.dtype()) {
        throw std::runtime_error("polar: abs and angle must have the same dtype");
    }
    auto shape_a = abs_t.shape();
    auto shape_b = angle_t.shape();
    if (!std::equal(shape_a.begin(), shape_a.end(), shape_b.begin(), shape_b.end())) {
        throw std::runtime_error("polar: abs and angle must have the same shape");
    }

    int64_t n = abs_t.numel();
    std::vector<int64_t> shape(shape_a.begin(), shape_a.end());

    if (abs_t.dtype() == DType::Float32) {
        Tensor result(shape, DType::Complex64, abs_t.device());
        const float* r_ptr = get_data_ptr<const float>(abs_t);
        const float* theta_ptr = get_data_ptr<const float>(angle_t);
        float* out_ptr = get_data_ptr<float>(result);
        queue.parallel_for<PolarKernelFloat32>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            float r = r_ptr[idx];
            float theta = theta_ptr[idx];
            out_ptr[2 * idx]     = r * sycl::cos(theta);
            out_ptr[2 * idx + 1] = r * sycl::sin(theta);
        });
        return result;
    } else if (abs_t.dtype() == DType::Float64) {
        Tensor result(shape, DType::Complex128, abs_t.device());
        const double* r_ptr = get_data_ptr<const double>(abs_t);
        const double* theta_ptr = get_data_ptr<const double>(angle_t);
        double* out_ptr = get_data_ptr<double>(result);
        queue.parallel_for<PolarKernelFloat64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            double r = r_ptr[idx];
            double theta = theta_ptr[idx];
            out_ptr[2 * idx]     = r * sycl::cos(theta);
            out_ptr[2 * idx + 1] = r * sycl::sin(theta);
        });
        return result;
    }
    throw std::runtime_error("polar: only Float32 and Float64 inputs are supported");
}

} // namespace oneapi
} // namespace tenzor
