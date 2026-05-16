#include "tenzor/core/tensor.hpp"
#include "tenzor/core/shape.hpp"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

#ifdef TENZOR_HAS_ONEMKL
#include <oneapi/mkl.hpp>
#endif

#ifdef TENZOR_HAS_ONEDPL
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/numeric>
#endif

#include "sycl_sort_utils.hpp"

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
struct MulKernelInt64 {};
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

// New unary kernel name classes
struct RsqrtKernelFloat32 {};
struct RsqrtKernelFloat64 {};
struct RsqrtKernelFloat16 {};
struct RsqrtKernelBFloat16 {};
struct SquareKernelFloat32 {};
struct SquareKernelFloat64 {};
struct SquareKernelFloat16 {};
struct SquareKernelBFloat16 {};
struct AsinhKernelFloat32 {};
struct AsinhKernelFloat64 {};
struct AsinhKernelFloat16 {};
struct AsinhKernelBFloat16 {};
struct AcoshKernelFloat32 {};
struct AcoshKernelFloat64 {};
struct AcoshKernelFloat16 {};
struct AcoshKernelBFloat16 {};
struct AtanhKernelFloat32 {};
struct AtanhKernelFloat64 {};
struct AtanhKernelFloat16 {};
struct AtanhKernelBFloat16 {};

// New binary kernel name classes
struct HypotKernelFloat32 {};
struct HypotKernelFloat64 {};
struct HypotKernelFloat16 {};
struct HypotKernelBFloat16 {};
struct CopysignKernelFloat32 {};
struct CopysignKernelFloat64 {};
struct CopysignKernelFloat16 {};
struct CopysignKernelBFloat16 {};
struct NextafterKernelFloat32 {};
struct NextafterKernelFloat64 {};
struct NextafterKernelFloat16 {};
struct NextafterKernelBFloat16 {};
struct GcdKernelInt32 {};
struct GcdKernelInt64 {};
struct LcmKernelInt32 {};
struct LcmKernelInt64 {};
struct IgammaKernelFloat32 {};
struct IgammaKernelFloat64 {};
struct IgammaKernelFloat16 {};
struct IgammaKernelBFloat16 {};
struct IgammacKernelFloat32 {};
struct IgammacKernelFloat64 {};
struct IgammacKernelFloat16 {};
struct IgammacKernelBFloat16 {};

// Ternary kernel name classes
struct LerpKernelFloat32 {};
struct LerpKernelFloat64 {};
struct LerpKernelFloat16 {};
struct LerpKernelBFloat16 {};
struct AddcmulKernelFloat32 {};
struct AddcmulKernelFloat64 {};
struct AddcmulKernelFloat16 {};
struct AddcmulKernelBFloat16 {};
struct AddcdivKernelFloat32 {};
struct AddcdivKernelFloat64 {};
struct AddcdivKernelFloat16 {};
struct AddcdivKernelBFloat16 {};

// Logical kernel name classes
struct LogicalAndKernelFloat32 {};
struct LogicalAndKernelFloat64 {};
struct LogicalAndKernelFloat16 {};
struct LogicalAndKernelBFloat16 {};
struct LogicalAndKernelInt32 {};
struct LogicalAndKernelInt64 {};
struct LogicalAndKernelBool {};
struct LogicalOrKernelFloat32 {};
struct LogicalOrKernelFloat64 {};
struct LogicalOrKernelFloat16 {};
struct LogicalOrKernelBFloat16 {};
struct LogicalOrKernelInt32 {};
struct LogicalOrKernelInt64 {};
struct LogicalOrKernelBool {};
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
struct LogicalXorKernelBool {};

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
struct AngleKernelFloat16 {};
struct AngleKernelBFloat16 {};
struct PolarKernelFloat32 {};
struct PolarKernelFloat64 {};
struct PolarKernelFloat16 {};
struct PolarKernelBFloat16 {};
struct ComplexTensorKernelFloat32 {};
struct ComplexTensorKernelFloat64 {};
struct ComplexTensorKernelFloat16 {};
struct ComplexTensorKernelBFloat16 {};

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
    // Round to nearest even (banker's rounding) for BFloat16
    uint32_t lsb = (bits >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    bits += rounding_bias;
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
    }).wait();
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
    }).wait();
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
        }).wait();
    }
    else if (a_cont.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a_cont);
        const double* b_ptr = get_data_ptr<const double>(b_cont);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<AddKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] + b_ptr[idx];
        }).wait();
    }
    else if (a_cont.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a_cont);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<AddKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            // Use float accumulation for precision
            out_ptr[idx] = sycl::half(static_cast<float>(a_ptr[idx]) + static_cast<float>(b_ptr[idx]));
        }).wait();
    }
    else if (a_cont.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a_cont);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<AddKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(bf16_to_f32(a_ptr[idx]) + bf16_to_f32(b_ptr[idx]));
        }).wait();
    }
    else if (a_cont.dtype() == DType::Int8) {
        const int8_t* a_ptr = get_data_ptr<const int8_t>(a_cont);
        const int8_t* b_ptr = get_data_ptr<const int8_t>(b_cont);
        int8_t* out_ptr = get_data_ptr<int8_t>(output);

        queue.parallel_for<AddKernelInt8>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] + b_ptr[idx];
        }).wait();
    }
    else if (a_cont.dtype() == DType::Int32) {
        const int32_t* a_ptr = get_data_ptr<const int32_t>(a_cont);
        const int32_t* b_ptr = get_data_ptr<const int32_t>(b_cont);
        int32_t* out_ptr = get_data_ptr<int32_t>(output);

        queue.parallel_for<AddKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] + b_ptr[idx];
        }).wait();
    }
    else if (a_cont.dtype() == DType::Int64) {
        const int64_t* a_ptr = get_data_ptr<const int64_t>(a_cont);
        const int64_t* b_ptr = get_data_ptr<const int64_t>(b_cont);
        int64_t* out_ptr = get_data_ptr<int64_t>(output);

        queue.parallel_for<AddKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] + b_ptr[idx];
        }).wait();
    }
    else if (a_cont.dtype() == DType::UInt8) {
        const uint8_t* a_ptr = get_data_ptr<const uint8_t>(a_cont);
        const uint8_t* b_ptr = get_data_ptr<const uint8_t>(b_cont);
        uint8_t* out_ptr = get_data_ptr<uint8_t>(output);

        queue.parallel_for<AddKernelUInt8>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] + b_ptr[idx];
        }).wait();
    }
    else if (a_cont.dtype() == DType::Bool) {
        // Bool addition acts as logical OR
        const bool* a_ptr = get_data_ptr<const bool>(a_cont);
        const bool* b_ptr = get_data_ptr<const bool>(b_cont);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<AddKernelBool>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] || b_ptr[idx];
        }).wait();
    }
    else if (a_cont.dtype() == DType::Complex64) {
        const float* a_ptr = get_data_ptr<const float>(a_cont);
        const float* b_ptr = get_data_ptr<const float>(b_cont);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<AddKernelComplex64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int64_t base = idx[0] * 2;
            out_ptr[base]     = a_ptr[base]     + b_ptr[base];
            out_ptr[base + 1] = a_ptr[base + 1] + b_ptr[base + 1];
        }).wait();
    }
    else if (a_cont.dtype() == DType::Complex128) {
        const double* a_ptr = get_data_ptr<const double>(a_cont);
        const double* b_ptr = get_data_ptr<const double>(b_cont);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<AddKernelComplex128>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int64_t base = idx[0] * 2;
            out_ptr[base]     = a_ptr[base]     + b_ptr[base];
            out_ptr[base + 1] = a_ptr[base + 1] + b_ptr[base + 1];
        }).wait();
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
        }).wait();
    }
    else if (a_cont.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a_cont);
        const double* b_ptr = get_data_ptr<const double>(b_cont);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<SubKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] - b_ptr[idx];
        }).wait();
    }
    else if (a_cont.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a_cont);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<SubKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(static_cast<float>(a_ptr[idx]) - static_cast<float>(b_ptr[idx]));
        }).wait();
    }
    else if (a_cont.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a_cont);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<SubKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(bf16_to_f32(a_ptr[idx]) - bf16_to_f32(b_ptr[idx]));
        }).wait();
    }
    else if (a_cont.dtype() == DType::Int8) {
        const int8_t* a_ptr = get_data_ptr<const int8_t>(a_cont);
        const int8_t* b_ptr = get_data_ptr<const int8_t>(b_cont);
        int8_t* out_ptr = get_data_ptr<int8_t>(output);

        queue.parallel_for<SubKernelInt8>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] - b_ptr[idx];
        }).wait();
    }
    else if (a_cont.dtype() == DType::Int32) {
        const int32_t* a_ptr = get_data_ptr<const int32_t>(a_cont);
        const int32_t* b_ptr = get_data_ptr<const int32_t>(b_cont);
        int32_t* out_ptr = get_data_ptr<int32_t>(output);

        queue.parallel_for<SubKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] - b_ptr[idx];
        }).wait();
    }
    else if (a_cont.dtype() == DType::Int64) {
        const int64_t* a_ptr = get_data_ptr<const int64_t>(a_cont);
        const int64_t* b_ptr = get_data_ptr<const int64_t>(b_cont);
        int64_t* out_ptr = get_data_ptr<int64_t>(output);

        queue.parallel_for<SubKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] - b_ptr[idx];
        }).wait();
    }
    else if (a_cont.dtype() == DType::UInt8) {
        const uint8_t* a_ptr = get_data_ptr<const uint8_t>(a_cont);
        const uint8_t* b_ptr = get_data_ptr<const uint8_t>(b_cont);
        uint8_t* out_ptr = get_data_ptr<uint8_t>(output);

        queue.parallel_for<SubKernelUInt8>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] - b_ptr[idx];
        }).wait();
    }
    else if (a_cont.dtype() == DType::Complex64) {
        const float* a_ptr = get_data_ptr<const float>(a_cont);
        const float* b_ptr = get_data_ptr<const float>(b_cont);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<SubKernelComplex64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int64_t base = idx[0] * 2;
            out_ptr[base]     = a_ptr[base]     - b_ptr[base];
            out_ptr[base + 1] = a_ptr[base + 1] - b_ptr[base + 1];
        }).wait();
    }
    else if (a_cont.dtype() == DType::Complex128) {
        const double* a_ptr = get_data_ptr<const double>(a_cont);
        const double* b_ptr = get_data_ptr<const double>(b_cont);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<SubKernelComplex128>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int64_t base = idx[0] * 2;
            out_ptr[base]     = a_ptr[base]     - b_ptr[base];
            out_ptr[base + 1] = a_ptr[base + 1] - b_ptr[base + 1];
        }).wait();
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
        }).wait();
    }
    else if (a_cont.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a_cont);
        const double* b_ptr = get_data_ptr<const double>(b_cont);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<MulKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] * b_ptr[idx];
        }).wait();
    }
    else if (a_cont.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a_cont);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<MulKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(static_cast<float>(a_ptr[idx]) * static_cast<float>(b_ptr[idx]));
        }).wait();
    }
    else if (a_cont.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a_cont);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<MulKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(bf16_to_f32(a_ptr[idx]) * bf16_to_f32(b_ptr[idx]));
        }).wait();
    }
    else if (a_cont.dtype() == DType::Int32) {
        const int32_t* a_ptr = get_data_ptr<const int32_t>(a_cont);
        const int32_t* b_ptr = get_data_ptr<const int32_t>(b_cont);
        int32_t* out_ptr = get_data_ptr<int32_t>(output);

        queue.parallel_for<MulKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] * b_ptr[idx];
        }).wait();
    }
    else if (a_cont.dtype() == DType::Int64) {
        const int64_t* a_ptr = get_data_ptr<const int64_t>(a_cont);
        const int64_t* b_ptr = get_data_ptr<const int64_t>(b_cont);
        int64_t* out_ptr = get_data_ptr<int64_t>(output);

        queue.parallel_for<MulKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] * b_ptr[idx];
        }).wait();
    }
    else if (a_cont.dtype() == DType::Bool) {
        // Bool multiplication acts as logical AND
        const bool* a_ptr = get_data_ptr<const bool>(a_cont);
        const bool* b_ptr = get_data_ptr<const bool>(b_cont);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<MulKernelBool>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] && b_ptr[idx];
        }).wait();
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
        }).wait();
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
        }).wait();
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
    }).wait();

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
        }).wait();
    }
    else if (a_cont.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a_cont);
        const double* b_ptr = get_data_ptr<const double>(b_cont);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<DivKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] / b_ptr[idx];
        }).wait();
    }
    else if (a_cont.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a_cont);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<DivKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(static_cast<float>(a_ptr[idx]) / static_cast<float>(b_ptr[idx]));
        }).wait();
    }
    else if (a_cont.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a_cont);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<DivKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(bf16_to_f32(a_ptr[idx]) / bf16_to_f32(b_ptr[idx]));
        }).wait();
    }
    else if (a_cont.dtype() == DType::Int32) {
        const int32_t* a_ptr = get_data_ptr<const int32_t>(a_cont);
        const int32_t* b_ptr = get_data_ptr<const int32_t>(b_cont);
        int32_t* out_ptr = get_data_ptr<int32_t>(output);

        // Check for division by zero before executing kernel
        check_integer_divisor_for_zeros(b_ptr, numel, queue);

        queue.parallel_for<DivKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] / b_ptr[idx];
        }).wait();
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
        }).wait();
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
        }).wait();
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
    // Handle empty tensors: if any dimension is 0, return zero-filled output
    if (a.numel() == 0 || b.numel() == 0) {
        auto a_shape = a.shape();
        auto b_shape = b.shape();
        if (a_shape.size() == 2 && b_shape.size() == 2) {
            // (M, 0) x (0, N) -> zeros(M, N)
            Tensor result({a_shape[0], b_shape[1]}, a.dtype(), a.device());
            queue.memset(const_cast<void*>(result.data_ptr()), 0,
                        result.numel() * result.dtype_size()).wait();
            return result;
        } else if (a_shape.size() == 1 && b_shape.size() == 2) {
            Tensor result({b_shape[1]}, a.dtype(), a.device());
            queue.memset(const_cast<void*>(result.data_ptr()), 0,
                        result.numel() * result.dtype_size()).wait();
            return result;
        }
        // Other empty cases: return empty tensor
        return Tensor({0}, a.dtype(), a.device());
    }

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
        // Fallback naive implementation. Wave F7 (deferred → landed):
        // extended from F32-only to F64/F16/BF16. All paths accumulate
        // in F32 inside the lambda (correct precision for all half-types).
        if (a_cont.dtype() == DType::Float32) {
            const float* a_ptr = get_data_ptr<const float>(a_cont);
            const float* b_ptr = get_data_ptr<const float>(b_cont);
            float* out_ptr = get_data_ptr<float>(output);
            queue.parallel_for<class MatMulKernelVectorF32>(sycl::range<1>(m), [=](sycl::id<1> idx) {
                const int64_t j = idx[0];
                float sum = 0.0f;
                for (int64_t p = 0; p < n; ++p) {
                    sum += a_ptr[p] * b_ptr[p * m + j];
                }
                out_ptr[j] = sum;
            });
        }
        else if (a_cont.dtype() == DType::Float64) {
            const double* a_ptr = get_data_ptr<const double>(a_cont);
            const double* b_ptr = get_data_ptr<const double>(b_cont);
            double* out_ptr = get_data_ptr<double>(output);
            queue.parallel_for<class MatMulKernelVectorF64>(sycl::range<1>(m), [=](sycl::id<1> idx) {
                const int64_t j = idx[0];
                double sum = 0.0;
                for (int64_t p = 0; p < n; ++p) {
                    sum += a_ptr[p] * b_ptr[p * m + j];
                }
                out_ptr[j] = sum;
            });
        }
        else if (a_cont.dtype() == DType::Float16) {
            const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a_cont);
            const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b_cont);
            sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
            queue.parallel_for<class MatMulKernelVectorF16>(sycl::range<1>(m), [=](sycl::id<1> idx) {
                const int64_t j = idx[0];
                float sum = 0.0f;
                for (int64_t p = 0; p < n; ++p) {
                    sum += static_cast<float>(a_ptr[p])
                         * static_cast<float>(b_ptr[p * m + j]);
                }
                out_ptr[j] = sycl::half(sum);
            });
        }
        else if (a_cont.dtype() == DType::BFloat16) {
            const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a_cont);
            const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b_cont);
            uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
            queue.parallel_for<class MatMulKernelVectorBF16>(sycl::range<1>(m), [=](sycl::id<1> idx) {
                const int64_t j = idx[0];
                float sum = 0.0f;
                for (int64_t p = 0; p < n; ++p) {
                    sum += bf16_to_f32(a_ptr[p]) * bf16_to_f32(b_ptr[p * m + j]);
                }
                out_ptr[j] = f32_to_bf16(sum);
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
    int64_t batch_count = 1;
    for (size_t i = 0; i < a_shape.size() - 2; ++i) {
        out_shape.push_back(a_shape[i]);
        batch_count *= a_shape[i];
    }
    out_shape.push_back(m);
    out_shape.push_back(n);

    Tensor output(out_shape, a_cont.dtype(), a_cont.device());

    // Audit-surfaced bug fix (M9 OneAPI 4D parity): the >=2D path below only
    // allocated/copied a single m*k submatrix and called gemm once, so for
    // any rank > 2 input the leading batch dimensions were silently ignored
    // — output beyond the first 2D submatrix was uninitialized garbage.
    // Loop over the leading batch dims so each 2D slice gets its own gemm.
    // (oneMKL gemm_batch would be faster but loop is correct and unblocks
    // the parity sweep; perf optimization tracked separately.)
    const int64_t a_batch_stride = m * k;
    const int64_t b_batch_stride = k * n;
    const int64_t c_batch_stride = m * n;

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

        const float alpha = 1.0f;
        const float beta = 0.0f;

        for (int64_t bi = 0; bi < batch_count; ++bi) {
            queue.memcpy(d_a.get(), a_ptr + bi * a_batch_stride, m * k * sizeof(float));
            queue.memcpy(d_b.get(), b_ptr + bi * b_batch_stride, k * n * sizeof(float));
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
            } catch (const ::oneapi::mkl::exception& e) {
                throw std::runtime_error(std::string("oneMKL GEMM (F32) failed: ") + e.what());
            } catch (const sycl::exception& e) {
                throw std::runtime_error(std::string("SYCL error in GEMM (F32): ") + e.what());
            }
            queue.memcpy(out_ptr + bi * c_batch_stride, d_c.get(), m * n * sizeof(float));
        }
        queue.wait_and_throw();
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

        // Per-batch loop (audit-surfaced bug fix — same as F32 path above).
        const double alpha = 1.0;
        const double beta = 0.0;
        for (int64_t bi = 0; bi < batch_count; ++bi) {
            queue.memcpy(d_a.get(), a_ptr + bi * a_batch_stride, m * k * sizeof(double));
            queue.memcpy(d_b.get(), b_ptr + bi * b_batch_stride, k * n * sizeof(double));
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
            } catch (const ::oneapi::mkl::exception& e) {
                throw std::runtime_error(std::string("oneMKL GEMM (F64) failed: ") + e.what());
            } catch (const sycl::exception& e) {
                throw std::runtime_error(std::string("SYCL error in GEMM (F64): ") + e.what());
            }
            queue.memcpy(out_ptr + bi * c_batch_stride, d_c.get(), m * n * sizeof(double));
        }
        queue.wait_and_throw();
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
        }).wait();
        auto* d_b_h_raw = d_b_h.get();
        auto* d_b_raw = d_b.get();
        queue.parallel_for(sycl::range<1>(b_count), [=](sycl::id<1> i) {
            d_b_raw[i] = static_cast<float>(d_b_h_raw[i]);
        }).wait();
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
        }).wait();
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
        }).wait();
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
        }).wait();
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
        }).wait();
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
        }).wait();
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
        }).wait();
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
        }).wait();
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
        }).wait();
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
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<SqrtKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::sqrt(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<SqrtKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::sqrt(static_cast<float>(in_ptr[idx])));
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<SqrtKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::sqrt(bf16_to_f32(in_ptr[idx])));
        }).wait();
    }
    else if (input.dtype() == DType::Complex64) {
        const auto* in_ptr = get_data_ptr<const std::complex<float>>(input);
        auto* out_ptr = get_data_ptr<std::complex<float>>(output);
        queue.parallel_for<class SqrtKernelComplex64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float a = in_ptr[idx].real();
            float b = in_ptr[idx].imag();
            if (a == 0.0f && b == 0.0f) { out_ptr[idx] = std::complex<float>(0.0f, 0.0f); return; }
            float s = sycl::sqrt(0.5f * (sycl::fabs(a) + sycl::hypot(a, b)));
            float re, im;
            if (a >= 0.0f) { re = s;                     im = b / (2.0f * s); }
            else            { re = sycl::fabs(b) / (2.0f * s); im = sycl::copysign(s, b); }
            out_ptr[idx] = std::complex<float>(re, im);
        }).wait();
    }
    else if (input.dtype() == DType::Complex128) {
        const auto* in_ptr = get_data_ptr<const std::complex<double>>(input);
        auto* out_ptr = get_data_ptr<std::complex<double>>(output);
        queue.parallel_for<class SqrtKernelComplex128>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double a = in_ptr[idx].real();
            double b = in_ptr[idx].imag();
            if (a == 0.0 && b == 0.0) { out_ptr[idx] = std::complex<double>(0.0, 0.0); return; }
            double s = sycl::sqrt(0.5 * (sycl::fabs(a) + sycl::hypot(a, b)));
            double re, im;
            if (a >= 0.0) { re = s;                  im = b / (2.0 * s); }
            else           { re = sycl::fabs(b) / (2.0 * s); im = sycl::copysign(s, b); }
            out_ptr[idx] = std::complex<double>(re, im);
        }).wait();
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
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<NegKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = -in_ptr[idx];
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<NegKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(-static_cast<float>(in_ptr[idx]));
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<NegKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(-bf16_to_f32(in_ptr[idx]));
        }).wait();
    }
    else if (input.dtype() == DType::Int32) {
        const int32_t* in_ptr = get_data_ptr<const int32_t>(input);
        int32_t* out_ptr = get_data_ptr<int32_t>(output);

        queue.parallel_for<NegKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = -in_ptr[idx];
        }).wait();
    }
    else if (input.dtype() == DType::Complex64) {
        const auto* in_ptr  = get_data_ptr<const std::complex<float>>(input);
        auto*       out_ptr = get_data_ptr<std::complex<float>>(output);
        queue.parallel_for<class NegKernelComplex64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = -in_ptr[idx];
        }).wait();
    }
    else if (input.dtype() == DType::Complex128) {
        const auto* in_ptr  = get_data_ptr<const std::complex<double>>(input);
        auto*       out_ptr = get_data_ptr<std::complex<double>>(output);
        queue.parallel_for<class NegKernelComplex128>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = -in_ptr[idx];
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for negation");
    }

    return output;
}

// Absolute value kernel
auto abs_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    const int64_t numel = input.numel();
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    // Complex → real magnitude: allocate output with reduced dtype.
    if (input.dtype() == DType::Complex64) {
        Tensor output(shape_vec, DType::Float32, input.device());
        const auto* in_ptr  = get_data_ptr<const std::complex<float>>(input);
        float*      out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<class AbsKernelComplex64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::hypot(in_ptr[idx].real(), in_ptr[idx].imag());
        }).wait();
        return output;
    }
    if (input.dtype() == DType::Complex128) {
        Tensor output(shape_vec, DType::Float64, input.device());
        const auto* in_ptr  = get_data_ptr<const std::complex<double>>(input);
        double*     out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<class AbsKernelComplex128>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::hypot(in_ptr[idx].real(), in_ptr[idx].imag());
        }).wait();
        return output;
    }

    Tensor output(shape_vec, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<AbsKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fabs(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<AbsKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fabs(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<AbsKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::fabs(static_cast<float>(in_ptr[idx])));
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<AbsKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::fabs(bf16_to_f32(in_ptr[idx])));
        }).wait();
    }
    else if (input.dtype() == DType::Int32) {
        const int32_t* in_ptr = get_data_ptr<const int32_t>(input);
        int32_t* out_ptr = get_data_ptr<int32_t>(output);

        queue.parallel_for<AbsKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::abs(in_ptr[idx]);
        }).wait();
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
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<LogKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::log(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<LogKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::log(static_cast<float>(in_ptr[idx])));
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<LogKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::log(bf16_to_f32(in_ptr[idx])));
        }).wait();
    }
    else if (input.dtype() == DType::Complex64) {
        const auto* in_ptr = get_data_ptr<const std::complex<float>>(input);
        auto* out_ptr = get_data_ptr<std::complex<float>>(output);
        queue.parallel_for<class LogKernelComplex64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float a = in_ptr[idx].real();
            float b = in_ptr[idx].imag();
            out_ptr[idx] = std::complex<float>(sycl::log(sycl::hypot(a, b)), sycl::atan2(b, a));
        }).wait();
    }
    else if (input.dtype() == DType::Complex128) {
        const auto* in_ptr = get_data_ptr<const std::complex<double>>(input);
        auto* out_ptr = get_data_ptr<std::complex<double>>(output);
        queue.parallel_for<class LogKernelComplex128>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double a = in_ptr[idx].real();
            double b = in_ptr[idx].imag();
            out_ptr[idx] = std::complex<double>(sycl::log(sycl::hypot(a, b)), sycl::atan2(b, a));
        }).wait();
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
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<ExpKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::exp(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<ExpKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            // Convert to float, compute exp, convert back to half
            out_ptr[idx] = sycl::half(sycl::exp(static_cast<float>(in_ptr[idx])));
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<ExpKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::exp(bf16_to_f32(in_ptr[idx])));
        }).wait();
    }
    else if (input.dtype() == DType::Complex64) {
        const auto* in_ptr = get_data_ptr<const std::complex<float>>(input);
        auto* out_ptr = get_data_ptr<std::complex<float>>(output);
        queue.parallel_for<class ExpKernelComplex64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float a = in_ptr[idx].real();
            float b = in_ptr[idx].imag();
            float ea = sycl::exp(a);
            out_ptr[idx] = std::complex<float>(ea * sycl::cos(b), ea * sycl::sin(b));
        }).wait();
    }
    else if (input.dtype() == DType::Complex128) {
        const auto* in_ptr = get_data_ptr<const std::complex<double>>(input);
        auto* out_ptr = get_data_ptr<std::complex<double>>(output);
        queue.parallel_for<class ExpKernelComplex128>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double a = in_ptr[idx].real();
            double b = in_ptr[idx].imag();
            double ea = sycl::exp(a);
            out_ptr[idx] = std::complex<double>(ea * sycl::cos(b), ea * sycl::sin(b));
        }).wait();
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
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        const double exp_d = static_cast<double>(exponent);

        queue.parallel_for<PowKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::pow(in_ptr[idx], exp_d);
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        // Use float accumulation for numerical stability
        queue.parallel_for<PowKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float val = static_cast<float>(in_ptr[idx]);
            out_ptr[idx] = sycl::half(sycl::pow(val, exponent));
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<PowKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::pow(bf16_to_f32(in_ptr[idx]), exponent));
        }).wait();
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
        }).wait();
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
        }).wait();
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
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<SinKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::sin(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<SinKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::sin(static_cast<float>(in_ptr[idx])));
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<SinKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::sin(bf16_to_f32(in_ptr[idx])));
        }).wait();
    }
    else if (input.dtype() == DType::Complex64) {
        const auto* in_ptr = get_data_ptr<const std::complex<float>>(input);
        auto* out_ptr = get_data_ptr<std::complex<float>>(output);
        queue.parallel_for<class SinKernelComplex64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float a = in_ptr[idx].real();
            float b = in_ptr[idx].imag();
            out_ptr[idx] = std::complex<float>(
                sycl::sin(a) * sycl::cosh(b), sycl::cos(a) * sycl::sinh(b));
        }).wait();
    }
    else if (input.dtype() == DType::Complex128) {
        const auto* in_ptr = get_data_ptr<const std::complex<double>>(input);
        auto* out_ptr = get_data_ptr<std::complex<double>>(output);
        queue.parallel_for<class SinKernelComplex128>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double a = in_ptr[idx].real();
            double b = in_ptr[idx].imag();
            out_ptr[idx] = std::complex<double>(
                sycl::sin(a) * sycl::cosh(b), sycl::cos(a) * sycl::sinh(b));
        }).wait();
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
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<CosKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::cos(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<CosKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::cos(static_cast<float>(in_ptr[idx])));
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<CosKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::cos(bf16_to_f32(in_ptr[idx])));
        }).wait();
    }
    else if (input.dtype() == DType::Complex64) {
        const auto* in_ptr = get_data_ptr<const std::complex<float>>(input);
        auto* out_ptr = get_data_ptr<std::complex<float>>(output);
        queue.parallel_for<class CosKernelComplex64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float a = in_ptr[idx].real();
            float b = in_ptr[idx].imag();
            out_ptr[idx] = std::complex<float>(
                sycl::cos(a) * sycl::cosh(b), -sycl::sin(a) * sycl::sinh(b));
        }).wait();
    }
    else if (input.dtype() == DType::Complex128) {
        const auto* in_ptr = get_data_ptr<const std::complex<double>>(input);
        auto* out_ptr = get_data_ptr<std::complex<double>>(output);
        queue.parallel_for<class CosKernelComplex128>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double a = in_ptr[idx].real();
            double b = in_ptr[idx].imag();
            out_ptr[idx] = std::complex<double>(
                sycl::cos(a) * sycl::cosh(b), -sycl::sin(a) * sycl::sinh(b));
        }).wait();
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
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<TanKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::tan(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<TanKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::tan(static_cast<float>(in_ptr[idx])));
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<TanKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::tan(bf16_to_f32(in_ptr[idx])));
        }).wait();
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
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<AsinKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::asin(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<AsinKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::asin(static_cast<float>(in_ptr[idx])));
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<AsinKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::asin(bf16_to_f32(in_ptr[idx])));
        }).wait();
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
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<AcosKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::acos(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<AcosKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::acos(static_cast<float>(in_ptr[idx])));
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<AcosKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::acos(bf16_to_f32(in_ptr[idx])));
        }).wait();
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
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<AtanKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::atan(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<AtanKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::atan(static_cast<float>(in_ptr[idx])));
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<AtanKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::atan(bf16_to_f32(in_ptr[idx])));
        }).wait();
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
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<SinhKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::sinh(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<SinhKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::sinh(static_cast<float>(in_ptr[idx])));
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<SinhKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::sinh(bf16_to_f32(in_ptr[idx])));
        }).wait();
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
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<CoshKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::cosh(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<CoshKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::cosh(static_cast<float>(in_ptr[idx])));
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<CoshKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::cosh(bf16_to_f32(in_ptr[idx])));
        }).wait();
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
        }).wait();
    }
    else if (y.dtype() == DType::Float64) {
        const double* y_ptr = get_data_ptr<const double>(y);
        const double* x_ptr = get_data_ptr<const double>(x);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<Atan2KernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::atan2(y_ptr[idx], x_ptr[idx]);
        }).wait();
    }
    else if (y.dtype() == DType::Float16) {
        const sycl::half* y_ptr = get_data_ptr<const sycl::half>(y);
        const sycl::half* x_ptr = get_data_ptr<const sycl::half>(x);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<Atan2KernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::atan2(static_cast<float>(y_ptr[idx]), static_cast<float>(x_ptr[idx])));
        }).wait();
    }
    else if (y.dtype() == DType::BFloat16) {
        const uint16_t* y_ptr = get_data_ptr<const uint16_t>(y);
        const uint16_t* x_ptr = get_data_ptr<const uint16_t>(x);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<Atan2KernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::atan2(bf16_to_f32(y_ptr[idx]), bf16_to_f32(x_ptr[idx])));
        }).wait();
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
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<RoundKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::round(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<RoundKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::round(static_cast<float>(in_ptr[idx])));
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<RoundKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::round(bf16_to_f32(in_ptr[idx])));
        }).wait();
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
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<FloorKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::floor(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<FloorKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::floor(static_cast<float>(in_ptr[idx])));
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<FloorKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::floor(bf16_to_f32(in_ptr[idx])));
        }).wait();
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
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<CeilKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::ceil(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<CeilKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::ceil(static_cast<float>(in_ptr[idx])));
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<CeilKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::ceil(bf16_to_f32(in_ptr[idx])));
        }).wait();
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
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<TruncKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::trunc(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<TruncKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::trunc(static_cast<float>(in_ptr[idx])));
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<TruncKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::trunc(bf16_to_f32(in_ptr[idx])));
        }).wait();
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
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<ReciprocalKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = 1.0 / in_ptr[idx];
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<ReciprocalKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(1.0f / static_cast<float>(in_ptr[idx]));
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<ReciprocalKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(1.0f / bf16_to_f32(in_ptr[idx]));
        }).wait();
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
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        const double min_d = static_cast<double>(min_val);

        queue.parallel_for<ClampMinKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fmax(in_ptr[idx], min_d);
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<ClampMinKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::fmax(static_cast<float>(in_ptr[idx]), min_val));
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<ClampMinKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::fmax(bf16_to_f32(in_ptr[idx]), min_val));
        }).wait();
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
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        const double max_d = static_cast<double>(max_val);

        queue.parallel_for<ClampMaxKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fmin(in_ptr[idx], max_d);
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<ClampMaxKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::fmin(static_cast<float>(in_ptr[idx]), max_val));
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<ClampMaxKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::fmin(bf16_to_f32(in_ptr[idx]), max_val));
        }).wait();
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
        }).wait();
    }
    else if (x.dtype() == DType::Float64) {
        const bool* cond_ptr = get_data_ptr<const bool>(condition);
        const double* x_ptr = get_data_ptr<const double>(x);
        const double* y_ptr = get_data_ptr<const double>(y);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<WhereKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = cond_ptr[idx] ? x_ptr[idx] : y_ptr[idx];
        }).wait();
    }
    else if (x.dtype() == DType::Float16) {
        const bool* cond_ptr = get_data_ptr<const bool>(condition);
        const sycl::half* x_ptr = get_data_ptr<const sycl::half>(x);
        const sycl::half* y_ptr = get_data_ptr<const sycl::half>(y);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<WhereKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = cond_ptr[idx] ? x_ptr[idx] : y_ptr[idx];
        }).wait();
    }
    else if (x.dtype() == DType::BFloat16) {
        const bool* cond_ptr = get_data_ptr<const bool>(condition);
        const uint16_t* x_ptr = get_data_ptr<const uint16_t>(x);
        const uint16_t* y_ptr = get_data_ptr<const uint16_t>(y);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<WhereKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = cond_ptr[idx] ? x_ptr[idx] : y_ptr[idx];
        }).wait();
    }
    else if (x.dtype() == DType::Int64) {
        const bool* cond_ptr = get_data_ptr<const bool>(condition);
        const int64_t* x_ptr = get_data_ptr<const int64_t>(x);
        const int64_t* y_ptr = get_data_ptr<const int64_t>(y);
        int64_t* out_ptr = get_data_ptr<int64_t>(output);

        queue.parallel_for<WhereKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = cond_ptr[idx] ? x_ptr[idx] : y_ptr[idx];
        }).wait();
    }
    else {
        throw std::runtime_error("where: unsupported dtype");
    }

    return output;
}

// Repeat kernel - repeats tensor along specified dimensions
auto repeat_kernel(const Tensor& input_in, const std::vector<int64_t>& repeats, sycl::queue& queue) -> Tensor {
    // Kernel below computes contiguous strides from shape; a non-contiguous
    // input view would be read at wrong offsets. Materialize to contiguous.
    Tensor input = input_in.is_contiguous() ? input_in : input_in.contiguous();
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
                    // Interleave semantics: match CPU/CUDA/Vulkan. Using
                    // `coord % input_shape[d]` (tile) diverges from CPU.
                    int64_t in_coord = coord / (out_shape_acc[d] / shape_acc[d]);
                    in_idx += in_coord * in_strides_acc[d];
                }

                out_ptr[out_idx] = in_ptr[in_idx];
            });
        }).wait();
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
                    // Interleave semantics: match CPU/CUDA/Vulkan. Using
                    // `coord % input_shape[d]` (tile) diverges from CPU.
                    int64_t in_coord = coord / (out_shape_acc[d] / shape_acc[d]);
                    in_idx += in_coord * in_strides_acc[d];
                }

                out_ptr[out_idx] = in_ptr[in_idx];
            });
        }).wait();
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
                    // Interleave semantics: match CPU/CUDA/Vulkan. Using
                    // `coord % input_shape[d]` (tile) diverges from CPU.
                    int64_t in_coord = coord / (out_shape_acc[d] / shape_acc[d]);
                    in_idx += in_coord * in_strides_acc[d];
                }

                out_ptr[out_idx] = in_ptr[in_idx];
            });
        }).wait();
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
        }).wait();
    } else if (inout.dtype() == DType::Float64) {
        double* data = get_data_ptr<double>(inout);
        const double* other_ptr = get_data_ptr<const double>(other);
        queue.parallel_for<AddInplaceKernelFloat64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] += other_ptr[idx];
        }).wait();
    } else if (inout.dtype() == DType::Float16) {
        sycl::half* data = get_data_ptr<sycl::half>(inout);
        const sycl::half* other_ptr = get_data_ptr<const sycl::half>(other);
        queue.parallel_for<AddInplaceKernelFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] = sycl::half(float(data[idx]) + float(other_ptr[idx]));
        }).wait();
    } else if (inout.dtype() == DType::BFloat16) {
        uint16_t* data = get_data_ptr<uint16_t>(inout);
        const uint16_t* other_ptr = get_data_ptr<const uint16_t>(other);
        queue.parallel_for<AddInplaceKernelBFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] = f32_to_bf16(bf16_to_f32(data[idx]) + bf16_to_f32(other_ptr[idx]));
        }).wait();
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
        }).wait();
    } else if (inout.dtype() == DType::Float64) {
        double* data = get_data_ptr<double>(inout);
        const double* other_ptr = get_data_ptr<const double>(other);
        queue.parallel_for<SubInplaceKernelFloat64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] -= other_ptr[idx];
        }).wait();
    } else if (inout.dtype() == DType::Float16) {
        sycl::half* data = get_data_ptr<sycl::half>(inout);
        const sycl::half* other_ptr = get_data_ptr<const sycl::half>(other);
        queue.parallel_for<SubInplaceKernelFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] = sycl::half(float(data[idx]) - float(other_ptr[idx]));
        }).wait();
    } else if (inout.dtype() == DType::BFloat16) {
        uint16_t* data = get_data_ptr<uint16_t>(inout);
        const uint16_t* other_ptr = get_data_ptr<const uint16_t>(other);
        queue.parallel_for<SubInplaceKernelBFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] = f32_to_bf16(bf16_to_f32(data[idx]) - bf16_to_f32(other_ptr[idx]));
        }).wait();
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
        }).wait();
    } else if (inout.dtype() == DType::Float64) {
        double* data = get_data_ptr<double>(inout);
        const double* other_ptr = get_data_ptr<const double>(other);
        queue.parallel_for<MulInplaceKernelFloat64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] *= other_ptr[idx];
        }).wait();
    } else if (inout.dtype() == DType::Float16) {
        sycl::half* data = get_data_ptr<sycl::half>(inout);
        const sycl::half* other_ptr = get_data_ptr<const sycl::half>(other);
        queue.parallel_for<MulInplaceKernelFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] = sycl::half(float(data[idx]) * float(other_ptr[idx]));
        }).wait();
    } else if (inout.dtype() == DType::BFloat16) {
        uint16_t* data = get_data_ptr<uint16_t>(inout);
        const uint16_t* other_ptr = get_data_ptr<const uint16_t>(other);
        queue.parallel_for<MulInplaceKernelBFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] = f32_to_bf16(bf16_to_f32(data[idx]) * bf16_to_f32(other_ptr[idx]));
        }).wait();
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
        }).wait();
    } else if (inout.dtype() == DType::Float64) {
        double* data = get_data_ptr<double>(inout);
        const double* other_ptr = get_data_ptr<const double>(other);
        queue.parallel_for<DivInplaceKernelFloat64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] /= other_ptr[idx];
        }).wait();
    } else if (inout.dtype() == DType::Float16) {
        sycl::half* data = get_data_ptr<sycl::half>(inout);
        const sycl::half* other_ptr = get_data_ptr<const sycl::half>(other);
        queue.parallel_for<DivInplaceKernelFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] = sycl::half(float(data[idx]) / float(other_ptr[idx]));
        }).wait();
    } else if (inout.dtype() == DType::BFloat16) {
        uint16_t* data = get_data_ptr<uint16_t>(inout);
        const uint16_t* other_ptr = get_data_ptr<const uint16_t>(other);
        queue.parallel_for<DivInplaceKernelBFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] = f32_to_bf16(bf16_to_f32(data[idx]) / bf16_to_f32(other_ptr[idx]));
        }).wait();
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
        }).wait();
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
        }).wait();
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
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<Log2KernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::log2(in_ptr[idx]);
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<Log2KernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::log2(static_cast<float>(in_ptr[idx])));
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<Log2KernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::log2(bf16_to_f32(in_ptr[idx])));
        }).wait();
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
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<Log10KernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::log10(in_ptr[idx]);
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<Log10KernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::log10(static_cast<float>(in_ptr[idx])));
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<Log10KernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::log10(bf16_to_f32(in_ptr[idx])));
        }).wait();
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
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<Log1pKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::log1p(in_ptr[idx]);
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<Log1pKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::log1p(static_cast<float>(in_ptr[idx])));
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<Log1pKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::log1p(bf16_to_f32(in_ptr[idx])));
        }).wait();
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
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<Exp2KernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::exp2(in_ptr[idx]);
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<Exp2KernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::exp2(static_cast<float>(in_ptr[idx])));
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<Exp2KernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::exp2(bf16_to_f32(in_ptr[idx])));
        }).wait();
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
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<Expm1KernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::expm1(in_ptr[idx]);
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<Expm1KernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::expm1(static_cast<float>(in_ptr[idx])));
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<Expm1KernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::expm1(bf16_to_f32(in_ptr[idx])));
        }).wait();
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
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<ErfKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::erf(in_ptr[idx]);
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<ErfKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::erf(static_cast<float>(in_ptr[idx])));
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<ErfKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::erf(bf16_to_f32(in_ptr[idx])));
        }).wait();
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
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<ErfcKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::erfc(in_ptr[idx]);
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<ErfcKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::erfc(static_cast<float>(in_ptr[idx])));
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<ErfcKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::erfc(bf16_to_f32(in_ptr[idx])));
        }).wait();
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
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        queue.parallel_for<IsNanKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(sycl::isnan(in_ptr[idx]) ? 1 : 0);
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        queue.parallel_for<IsNanKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(sycl::isnan(static_cast<float>(in_ptr[idx])) ? 1 : 0);
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        queue.parallel_for<IsNanKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(sycl::isnan(bf16_to_f32(in_ptr[idx])) ? 1 : 0);
        }).wait();
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
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        queue.parallel_for<IsInfKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(sycl::isinf(in_ptr[idx]) ? 1 : 0);
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        queue.parallel_for<IsInfKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(sycl::isinf(static_cast<float>(in_ptr[idx])) ? 1 : 0);
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        queue.parallel_for<IsInfKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(sycl::isinf(bf16_to_f32(in_ptr[idx])) ? 1 : 0);
        }).wait();
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
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        queue.parallel_for<IsFiniteKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(sycl::isfinite(in_ptr[idx]) ? 1 : 0);
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        queue.parallel_for<IsFiniteKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(sycl::isfinite(static_cast<float>(in_ptr[idx])) ? 1 : 0);
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        queue.parallel_for<IsFiniteKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(sycl::isfinite(bf16_to_f32(in_ptr[idx])) ? 1 : 0);
        }).wait();
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
        }).wait();
    } else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<FmodKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fmod(a_ptr[idx], b_ptr[idx]);
        }).wait();
    } else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<FmodKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::fmod(static_cast<float>(a_ptr[idx]), static_cast<float>(b_ptr[idx])));
        }).wait();
    } else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<FmodKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::fmod(bf16_to_f32(a_ptr[idx]), bf16_to_f32(b_ptr[idx])));
        }).wait();
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
        }).wait();
    } else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<RemainderKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::remainder(a_ptr[idx], b_ptr[idx]);
        }).wait();
    } else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<RemainderKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::remainder(static_cast<float>(a_ptr[idx]), static_cast<float>(b_ptr[idx])));
        }).wait();
    } else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<RemainderKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::remainder(bf16_to_f32(a_ptr[idx]), bf16_to_f32(b_ptr[idx])));
        }).wait();
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
        }).wait();
    } else if (start.dtype() == DType::Float64) {
        const double* s_ptr = get_data_ptr<const double>(start);
        const double* e_ptr = get_data_ptr<const double>(end);
        const double* w_ptr = get_data_ptr<const double>(weight);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<LerpKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = s_ptr[idx] + w_ptr[idx] * (e_ptr[idx] - s_ptr[idx]);
        }).wait();
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
        }).wait();
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
        }).wait();
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
        }).wait();
    } else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        queue.parallel_for<LogicalAndKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>((a_ptr[idx] != 0.0 && b_ptr[idx] != 0.0) ? 1 : 0);
        }).wait();
    } else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        queue.parallel_for<LogicalAndKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>((static_cast<float>(a_ptr[idx]) != 0.0f && static_cast<float>(b_ptr[idx]) != 0.0f) ? 1 : 0);
        }).wait();
    } else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        queue.parallel_for<LogicalAndKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>((bf16_to_f32(a_ptr[idx]) != 0.0f && bf16_to_f32(b_ptr[idx]) != 0.0f) ? 1 : 0);
        }).wait();
    } else if (a.dtype() == DType::Int32) {
        const int32_t* a_ptr = get_data_ptr<const int32_t>(a);
        const int32_t* b_ptr = get_data_ptr<const int32_t>(b);
        queue.parallel_for<LogicalAndKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>((a_ptr[idx] != 0 && b_ptr[idx] != 0) ? 1 : 0);
        }).wait();
    } else if (a.dtype() == DType::Int64) {
        const int64_t* a_ptr = get_data_ptr<const int64_t>(a);
        const int64_t* b_ptr = get_data_ptr<const int64_t>(b);
        queue.parallel_for<LogicalAndKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>((a_ptr[idx] != 0 && b_ptr[idx] != 0) ? 1 : 0);
        }).wait();
    } else if (a.dtype() == DType::Bool) {
        const uint8_t* a_ptr = get_data_ptr<const uint8_t>(a);
        const uint8_t* b_ptr = get_data_ptr<const uint8_t>(b);
        queue.parallel_for<LogicalAndKernelBool>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>((a_ptr[idx] != 0 && b_ptr[idx] != 0) ? 1 : 0);
        }).wait();
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
        }).wait();
    } else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        queue.parallel_for<LogicalOrKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>((a_ptr[idx] != 0.0 || b_ptr[idx] != 0.0) ? 1 : 0);
        }).wait();
    } else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        queue.parallel_for<LogicalOrKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>((static_cast<float>(a_ptr[idx]) != 0.0f || static_cast<float>(b_ptr[idx]) != 0.0f) ? 1 : 0);
        }).wait();
    } else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        queue.parallel_for<LogicalOrKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>((bf16_to_f32(a_ptr[idx]) != 0.0f || bf16_to_f32(b_ptr[idx]) != 0.0f) ? 1 : 0);
        }).wait();
    } else if (a.dtype() == DType::Int32) {
        const int32_t* a_ptr = get_data_ptr<const int32_t>(a);
        const int32_t* b_ptr = get_data_ptr<const int32_t>(b);
        queue.parallel_for<LogicalOrKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>((a_ptr[idx] != 0 || b_ptr[idx] != 0) ? 1 : 0);
        }).wait();
    } else if (a.dtype() == DType::Int64) {
        const int64_t* a_ptr = get_data_ptr<const int64_t>(a);
        const int64_t* b_ptr = get_data_ptr<const int64_t>(b);
        queue.parallel_for<LogicalOrKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>((a_ptr[idx] != 0 || b_ptr[idx] != 0) ? 1 : 0);
        }).wait();
    } else if (a.dtype() == DType::Bool) {
        const uint8_t* a_ptr = get_data_ptr<const uint8_t>(a);
        const uint8_t* b_ptr = get_data_ptr<const uint8_t>(b);
        queue.parallel_for<LogicalOrKernelBool>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>((a_ptr[idx] != 0 || b_ptr[idx] != 0) ? 1 : 0);
        }).wait();
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
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        queue.parallel_for<LogicalNotKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(in_ptr[idx] == 0.0 ? 1 : 0);
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        queue.parallel_for<LogicalNotKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(static_cast<float>(in_ptr[idx]) == 0.0f ? 1 : 0);
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        queue.parallel_for<LogicalNotKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(bf16_to_f32(in_ptr[idx]) == 0.0f ? 1 : 0);
        }).wait();
    } else if (input.dtype() == DType::Int32) {
        const int32_t* in_ptr = get_data_ptr<const int32_t>(input);
        queue.parallel_for<LogicalNotKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(in_ptr[idx] == 0 ? 1 : 0);
        }).wait();
    } else if (input.dtype() == DType::Int64) {
        const int64_t* in_ptr = get_data_ptr<const int64_t>(input);
        queue.parallel_for<LogicalNotKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(in_ptr[idx] == 0 ? 1 : 0);
        }).wait();
    } else if (input.dtype() == DType::Bool) {
        const uint8_t* in_ptr = get_data_ptr<const uint8_t>(input);
        queue.parallel_for<LogicalNotKernelBool>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(in_ptr[idx] == 0 ? 1 : 0);
        }).wait();
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
        }).wait();
    } else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        queue.parallel_for<LogicalXorKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            bool va = a_ptr[idx] != 0.0;
            bool vb = b_ptr[idx] != 0.0;
            out_ptr[idx] = static_cast<uint8_t>((va != vb) ? 1 : 0);
        }).wait();
    } else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        queue.parallel_for<LogicalXorKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            bool va = static_cast<float>(a_ptr[idx]) != 0.0f;
            bool vb = static_cast<float>(b_ptr[idx]) != 0.0f;
            out_ptr[idx] = static_cast<uint8_t>((va != vb) ? 1 : 0);
        }).wait();
    } else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        queue.parallel_for<LogicalXorKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            bool va = bf16_to_f32(a_ptr[idx]) != 0.0f;
            bool vb = bf16_to_f32(b_ptr[idx]) != 0.0f;
            out_ptr[idx] = static_cast<uint8_t>((va != vb) ? 1 : 0);
        }).wait();
    } else if (a.dtype() == DType::Int32) {
        const int32_t* a_ptr = get_data_ptr<const int32_t>(a);
        const int32_t* b_ptr = get_data_ptr<const int32_t>(b);
        queue.parallel_for<LogicalXorKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            bool va = a_ptr[idx] != 0;
            bool vb = b_ptr[idx] != 0;
            out_ptr[idx] = static_cast<uint8_t>((va != vb) ? 1 : 0);
        }).wait();
    } else if (a.dtype() == DType::Int64) {
        const int64_t* a_ptr = get_data_ptr<const int64_t>(a);
        const int64_t* b_ptr = get_data_ptr<const int64_t>(b);
        queue.parallel_for<LogicalXorKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            bool va = a_ptr[idx] != 0;
            bool vb = b_ptr[idx] != 0;
            out_ptr[idx] = static_cast<uint8_t>((va != vb) ? 1 : 0);
        }).wait();
    } else if (a.dtype() == DType::Bool) {
        const uint8_t* a_ptr = get_data_ptr<const uint8_t>(a);
        const uint8_t* b_ptr = get_data_ptr<const uint8_t>(b);
        queue.parallel_for<LogicalXorKernelBool>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            bool va = a_ptr[idx] != 0;
            bool vb = b_ptr[idx] != 0;
            out_ptr[idx] = static_cast<uint8_t>((va != vb) ? 1 : 0);
        }).wait();
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
        }).wait();
    } else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<MinimumKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fmin(a_ptr[idx], b_ptr[idx]);
        }).wait();
    } else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<MinimumKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::fmin(static_cast<float>(a_ptr[idx]), static_cast<float>(b_ptr[idx])));
        }).wait();
    } else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<MinimumKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::fmin(bf16_to_f32(a_ptr[idx]), bf16_to_f32(b_ptr[idx])));
        }).wait();
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
        }).wait();
    } else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<MaximumKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fmax(a_ptr[idx], b_ptr[idx]);
        }).wait();
    } else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<MaximumKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::fmax(static_cast<float>(a_ptr[idx]), static_cast<float>(b_ptr[idx])));
        }).wait();
    } else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<MaximumKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::fmax(bf16_to_f32(a_ptr[idx]), bf16_to_f32(b_ptr[idx])));
        }).wait();
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
        }).wait();
        return result;
    } else if (input.dtype() == DType::Complex128) {
        Tensor result(shape, DType::Complex128, input.device());
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(result);
        queue.parallel_for<ConjKernelComplex128>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            out_ptr[2 * idx]     =  in_ptr[2 * idx];
            out_ptr[2 * idx + 1] = -in_ptr[2 * idx + 1];
        }).wait();
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
        }).wait();
        return result;
    } else if (input.dtype() == DType::Complex128) {
        Tensor result(shape, DType::Float64, input.device());
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(result);
        queue.parallel_for<RealKernelComplex128>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            out_ptr[idx] = in_ptr[2 * idx];
        }).wait();
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
        }).wait();
        return result;
    } else if (input.dtype() == DType::Complex128) {
        Tensor result(shape, DType::Float64, input.device());
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(result);
        queue.parallel_for<ImagKernelComplex128>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            out_ptr[idx] = in_ptr[2 * idx + 1];
        }).wait();
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
        }).wait();
        return result;
    } else if (input.dtype() == DType::Complex128) {
        Tensor result(shape, DType::Float64, input.device());
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(result);
        queue.parallel_for<AngleKernelComplex128>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::atan2(in_ptr[2 * idx + 1], in_ptr[2 * idx]);
        }).wait();
        return result;
    } else if (input.dtype() == DType::Float32) {
        Tensor result(shape, DType::Float32, input.device());
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(result);
        queue.parallel_for<AngleKernelFloat32>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::atan2(0.0f, in_ptr[idx]);
        }).wait();
        return result;
    } else if (input.dtype() == DType::Float64) {
        Tensor result(shape, DType::Float64, input.device());
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(result);
        queue.parallel_for<AngleKernelFloat64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::atan2(0.0, in_ptr[idx]);
        }).wait();
        return result;
    } else if (input.dtype() == DType::Float16) {
        Tensor result(shape, DType::Float16, input.device());
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(result);
        queue.parallel_for<AngleKernelFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::atan2(0.0f, static_cast<float>(in_ptr[idx])));
        }).wait();
        return result;
    } else if (input.dtype() == DType::BFloat16) {
        Tensor result(shape, DType::BFloat16, input.device());
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(result);
        queue.parallel_for<AngleKernelBFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::atan2(0.0f, bf16_to_f32(in_ptr[idx])));
        }).wait();
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
        }).wait();
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
        }).wait();
        return result;
    } else if (abs_t.dtype() == DType::Float16) {
        Tensor result(shape, DType::Complex64, abs_t.device());
        const sycl::half* r_ptr = get_data_ptr<const sycl::half>(abs_t);
        const sycl::half* theta_ptr = get_data_ptr<const sycl::half>(angle_t);
        float* out_ptr = get_data_ptr<float>(result);
        queue.parallel_for<PolarKernelFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            float r = static_cast<float>(r_ptr[idx]);
            float theta = static_cast<float>(theta_ptr[idx]);
            out_ptr[2 * idx]     = r * sycl::cos(theta);
            out_ptr[2 * idx + 1] = r * sycl::sin(theta);
        }).wait();
        return result;
    } else if (abs_t.dtype() == DType::BFloat16) {
        Tensor result(shape, DType::Complex64, abs_t.device());
        const uint16_t* r_ptr = get_data_ptr<const uint16_t>(abs_t);
        const uint16_t* theta_ptr = get_data_ptr<const uint16_t>(angle_t);
        float* out_ptr = get_data_ptr<float>(result);
        queue.parallel_for<PolarKernelBFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            float r = bf16_to_f32(r_ptr[idx]);
            float theta = bf16_to_f32(theta_ptr[idx]);
            out_ptr[2 * idx]     = r * sycl::cos(theta);
            out_ptr[2 * idx + 1] = r * sycl::sin(theta);
        }).wait();
        return result;
    }
    throw std::runtime_error("polar: unsupported input dtype");
}

// ============================================================================
// ComplexTensor Kernel — interleave real + imag into Complex64/Complex128
// ============================================================================

auto complex_tensor_kernel(const Tensor& real_t, const Tensor& imag_t, sycl::queue& queue) -> Tensor {
    if (real_t.dtype() != imag_t.dtype()) {
        throw std::runtime_error("complex: real and imag must have the same dtype");
    }
    auto shape_r = real_t.shape();
    auto shape_i = imag_t.shape();
    if (!std::equal(shape_r.begin(), shape_r.end(), shape_i.begin(), shape_i.end())) {
        throw std::runtime_error("complex: real and imag must have the same shape");
    }

    int64_t n = real_t.numel();
    std::vector<int64_t> shape(shape_r.begin(), shape_r.end());

    if (real_t.dtype() == DType::Float32) {
        Tensor result(shape, DType::Complex64, real_t.device());
        const float* r_ptr = get_data_ptr<const float>(real_t);
        const float* i_ptr = get_data_ptr<const float>(imag_t);
        float* out_ptr = get_data_ptr<float>(result);
        queue.parallel_for<ComplexTensorKernelFloat32>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            out_ptr[2 * idx]     = r_ptr[idx];
            out_ptr[2 * idx + 1] = i_ptr[idx];
        }).wait();
        return result;
    } else if (real_t.dtype() == DType::Float64) {
        Tensor result(shape, DType::Complex128, real_t.device());
        const double* r_ptr = get_data_ptr<const double>(real_t);
        const double* i_ptr = get_data_ptr<const double>(imag_t);
        double* out_ptr = get_data_ptr<double>(result);
        queue.parallel_for<ComplexTensorKernelFloat64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            out_ptr[2 * idx]     = r_ptr[idx];
            out_ptr[2 * idx + 1] = i_ptr[idx];
        }).wait();
        return result;
    } else if (real_t.dtype() == DType::Float16) {
        Tensor result(shape, DType::Complex64, real_t.device());
        const sycl::half* r_ptr = get_data_ptr<const sycl::half>(real_t);
        const sycl::half* i_ptr = get_data_ptr<const sycl::half>(imag_t);
        float* out_ptr = get_data_ptr<float>(result);
        queue.parallel_for<ComplexTensorKernelFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            out_ptr[2 * idx]     = static_cast<float>(r_ptr[idx]);
            out_ptr[2 * idx + 1] = static_cast<float>(i_ptr[idx]);
        }).wait();
        return result;
    } else if (real_t.dtype() == DType::BFloat16) {
        Tensor result(shape, DType::Complex64, real_t.device());
        const uint16_t* r_ptr = get_data_ptr<const uint16_t>(real_t);
        const uint16_t* i_ptr = get_data_ptr<const uint16_t>(imag_t);
        float* out_ptr = get_data_ptr<float>(result);
        queue.parallel_for<ComplexTensorKernelBFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            out_ptr[2 * idx]     = bf16_to_f32(r_ptr[idx]);
            out_ptr[2 * idx + 1] = bf16_to_f32(i_ptr[idx]);
        }).wait();
        return result;
    }
    throw std::runtime_error("complex: unsupported input dtype");
}

// ============================================================================
// Cross Product Kernel
// ============================================================================

struct CrossKernelFloat32 {};
struct CrossKernelFloat64 {};
struct CrossKernelFloat16 {};
struct CrossKernelBFloat16 {};

auto cross_kernel(const Tensor& a, const Tensor& b, int64_t dim, sycl::queue& queue) -> Tensor {
    auto shape = a.shape();
    int64_t ndim = shape.size();
    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    Tensor output(out_shape, a.dtype(), a.device());

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; ++d) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; ++d) inner *= shape[d];
    int64_t num_pairs = outer * inner;

    if (num_pairs == 0) return output;

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<CrossKernelFloat32>(sycl::range<1>(num_pairs), [=](sycl::id<1> idx) {
            int64_t o = idx / inner;
            int64_t i = idx % inner;
            int64_t base = o * 3 * inner + i;
            float a0 = a_ptr[base], a1 = a_ptr[base + inner], a2 = a_ptr[base + 2*inner];
            float b0 = b_ptr[base], b1 = b_ptr[base + inner], b2 = b_ptr[base + 2*inner];
            out_ptr[base]            = a1*b2 - a2*b1;
            out_ptr[base + inner]    = a2*b0 - a0*b2;
            out_ptr[base + 2*inner]  = a0*b1 - a1*b0;
        }).wait();
    } else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<CrossKernelFloat64>(sycl::range<1>(num_pairs), [=](sycl::id<1> idx) {
            int64_t o = idx / inner;
            int64_t i = idx % inner;
            int64_t base = o * 3 * inner + i;
            double a0 = a_ptr[base], a1 = a_ptr[base + inner], a2 = a_ptr[base + 2*inner];
            double b0 = b_ptr[base], b1 = b_ptr[base + inner], b2 = b_ptr[base + 2*inner];
            out_ptr[base]            = a1*b2 - a2*b1;
            out_ptr[base + inner]    = a2*b0 - a0*b2;
            out_ptr[base + 2*inner]  = a0*b1 - a1*b0;
        }).wait();
    } else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<CrossKernelFloat16>(sycl::range<1>(num_pairs), [=](sycl::id<1> idx) {
            int64_t o = idx / inner;
            int64_t i = idx % inner;
            int64_t base = o * 3 * inner + i;
            float a0 = float(a_ptr[base]), a1 = float(a_ptr[base + inner]), a2 = float(a_ptr[base + 2*inner]);
            float b0 = float(b_ptr[base]), b1 = float(b_ptr[base + inner]), b2 = float(b_ptr[base + 2*inner]);
            out_ptr[base]            = sycl::half(a1*b2 - a2*b1);
            out_ptr[base + inner]    = sycl::half(a2*b0 - a0*b2);
            out_ptr[base + 2*inner]  = sycl::half(a0*b1 - a1*b0);
        }).wait();
    } else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<CrossKernelBFloat16>(sycl::range<1>(num_pairs), [=](sycl::id<1> idx) {
            int64_t o = idx / inner;
            int64_t i = idx % inner;
            int64_t base = o * 3 * inner + i;
            float a0 = bf16_to_f32(a_ptr[base]), a1 = bf16_to_f32(a_ptr[base + inner]), a2 = bf16_to_f32(a_ptr[base + 2*inner]);
            float b0 = bf16_to_f32(b_ptr[base]), b1 = bf16_to_f32(b_ptr[base + inner]), b2 = bf16_to_f32(b_ptr[base + 2*inner]);
            out_ptr[base]            = f32_to_bf16(a1*b2 - a2*b1);
            out_ptr[base + inner]    = f32_to_bf16(a2*b0 - a0*b2);
            out_ptr[base + 2*inner]  = f32_to_bf16(a0*b1 - a1*b0);
        }).wait();
    } else {
        throw std::runtime_error("cross: unsupported dtype");
    }
    return output;
}

// ============================================================================
// New Phase 4 ops: Frac, Heaviside, NanToNum, LogSigmoid, Bitwise
// ============================================================================

struct FracKernelF32 {};
struct FracKernelF64 {};

auto frac_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto f32 = input.to(DType::Float32);
        return frac_kernel(f32, queue).to(input.dtype());
    }
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    int64_t n = input.numel();
    if (n == 0) return result;

    // frac is sign-preserving (x - trunc(x)), not x - floor(x). See the CUDA
    // kernel's comment in src/backends/cuda/kernels/math.cu — the same bug
    // was present here.
    if (input.dtype() == DType::Float32) {
        const float* in = input.data<float>(); float* out = result.data<float>();
        queue.parallel_for<FracKernelF32>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            float x = in[idx]; out[idx] = x - sycl::trunc(x);
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in = input.data<double>(); double* out = result.data<double>();
        queue.parallel_for<FracKernelF64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            double x = in[idx]; out[idx] = x - sycl::trunc(x);
        }).wait();
    } else { throw std::runtime_error("frac: unsupported dtype"); }
    return result;
}

struct HeavisideKernelF32 {};
struct HeavisideKernelF64 {};

auto heaviside_kernel(const Tensor& input, const Tensor& values, sycl::queue& queue) -> Tensor {
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto f32_in = input.to(DType::Float32); auto f32_val = values.to(DType::Float32);
        return heaviside_kernel(f32_in, f32_val, queue).to(input.dtype());
    }
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    int64_t n = input.numel();
    if (n == 0) return result;

    if (input.dtype() == DType::Float32) {
        const float* in = input.data<float>(); const float* val = values.data<float>(); float* out = result.data<float>();
        queue.parallel_for<HeavisideKernelF32>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            float x = in[idx];
            out[idx] = (x < 0.0f) ? 0.0f : (x == 0.0f ? val[idx] : 1.0f);
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in = input.data<double>(); const double* val = values.data<double>(); double* out = result.data<double>();
        queue.parallel_for<HeavisideKernelF64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            double x = in[idx];
            out[idx] = (x < 0.0) ? 0.0 : (x == 0.0 ? val[idx] : 1.0);
        }).wait();
    } else { throw std::runtime_error("heaviside: unsupported dtype"); }
    return result;
}

struct NanToNumKernelF32 {};
struct NanToNumKernelF64 {};

auto nan_to_num_kernel(const Tensor& input, double nan_v, double posinf_v, double neginf_v, sycl::queue& queue) -> Tensor {
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto f32 = input.to(DType::Float32);
        return nan_to_num_kernel(f32, nan_v, posinf_v, neginf_v, queue).to(input.dtype());
    }
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    int64_t n = input.numel();
    if (n == 0) return result;

    if (input.dtype() == DType::Float32) {
        const float* in = input.data<float>(); float* out = result.data<float>();
        float nv = static_cast<float>(nan_v);
        float pv = (posinf_v >= static_cast<double>(std::numeric_limits<float>::max())) ? std::numeric_limits<float>::max() : static_cast<float>(posinf_v);
        float nf = (neginf_v <= static_cast<double>(std::numeric_limits<float>::lowest())) ? std::numeric_limits<float>::lowest() : static_cast<float>(neginf_v);
        queue.parallel_for<NanToNumKernelF32>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            float x = in[idx];
            if (sycl::isnan(x)) out[idx] = nv;
            else if (sycl::isinf(x) && x > 0) out[idx] = pv;
            else if (sycl::isinf(x) && x < 0) out[idx] = nf;
            else out[idx] = x;
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in = input.data<double>(); double* out = result.data<double>();
        double nv = nan_v;
        double pv = posinf_v;
        double nf = neginf_v;
        queue.parallel_for<NanToNumKernelF64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            double x = in[idx];
            if (sycl::isnan(x)) out[idx] = nv;
            else if (sycl::isinf(x) && x > 0) out[idx] = pv;
            else if (sycl::isinf(x) && x < 0) out[idx] = nf;
            else out[idx] = x;
        }).wait();
    } else { throw std::runtime_error("nan_to_num: unsupported dtype"); }
    return result;
}

struct LogSigmoidKernelF32 {};
struct LogSigmoidKernelF64 {};

auto log_sigmoid_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto f32 = input.to(DType::Float32);
        return log_sigmoid_kernel(f32, queue).to(input.dtype());
    }
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    int64_t n = input.numel();
    if (n == 0) return result;

    if (input.dtype() == DType::Float32) {
        const float* in = input.data<float>(); float* out = result.data<float>();
        queue.parallel_for<LogSigmoidKernelF32>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            float x = in[idx];
            out[idx] = (x >= 0.0f) ? -sycl::log1p(sycl::exp(-x)) : x - sycl::log1p(sycl::exp(x));
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in = input.data<double>(); double* out = result.data<double>();
        queue.parallel_for<LogSigmoidKernelF64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            double x = in[idx];
            out[idx] = (x >= 0.0) ? -sycl::log1p(sycl::exp(-x)) : x - sycl::log1p(sycl::exp(x));
        }).wait();
    } else { throw std::runtime_error("log_sigmoid: unsupported dtype"); }
    return result;
}

struct BitwiseAndI32 {};
struct BitwiseOrI32 {};
struct BitwiseXorI32 {};
struct BitwiseNotI32 {};
struct BitwiseLShiftI32 {};
struct BitwiseRShiftI32 {};

// SYCL requires distinct kernel name types per (op, dtype). Tags below cover
// Int8/Int16/Int64 in addition to the original Int32 tags above so the bitwise
// op family supports the same integer dtypes as CPU/CUDA/ROCm.
struct BitwiseAndI8 {}; struct BitwiseAndI16 {}; struct BitwiseAndI64 {};
struct BitwiseOrI8 {};  struct BitwiseOrI16 {};  struct BitwiseOrI64 {};
struct BitwiseXorI8 {}; struct BitwiseXorI16 {}; struct BitwiseXorI64 {};
struct BitwiseNotI8 {}; struct BitwiseNotI16 {}; struct BitwiseNotI64 {};
struct BitwiseLShiftI8 {}; struct BitwiseLShiftI16 {}; struct BitwiseLShiftI64 {};
struct BitwiseRShiftI8 {}; struct BitwiseRShiftI16 {}; struct BitwiseRShiftI64 {};

auto bitwise_and_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device()); int64_t n = a.numel();
    if (n == 0) return result;
    if (a.dtype() == DType::Int8) {
        const int8_t* ad = a.data<int8_t>(); const int8_t* bd = b.data<int8_t>(); int8_t* od = result.data<int8_t>();
        queue.parallel_for<BitwiseAndI8>(sycl::range<1>(n), [=](sycl::id<1> i) { od[i] = ad[i] & bd[i]; }).wait();
    } else if (a.dtype() == DType::Int16) {
        const int16_t* ad = a.data<int16_t>(); const int16_t* bd = b.data<int16_t>(); int16_t* od = result.data<int16_t>();
        queue.parallel_for<BitwiseAndI16>(sycl::range<1>(n), [=](sycl::id<1> i) { od[i] = ad[i] & bd[i]; }).wait();
    } else if (a.dtype() == DType::Int32) {
        const int32_t* ad = a.data<int32_t>(); const int32_t* bd = b.data<int32_t>(); int32_t* od = result.data<int32_t>();
        queue.parallel_for<BitwiseAndI32>(sycl::range<1>(n), [=](sycl::id<1> i) { od[i] = ad[i] & bd[i]; }).wait();
    } else if (a.dtype() == DType::Int64) {
        const int64_t* ad = a.data<int64_t>(); const int64_t* bd = b.data<int64_t>(); int64_t* od = result.data<int64_t>();
        queue.parallel_for<BitwiseAndI64>(sycl::range<1>(n), [=](sycl::id<1> i) { od[i] = ad[i] & bd[i]; }).wait();
    } else { throw std::runtime_error("bitwise_and: unsupported dtype (expected Int8/16/32/64)"); }
    return result;
}
auto bitwise_or_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device()); int64_t n = a.numel();
    if (n == 0) return result;
    if (a.dtype() == DType::Int8) {
        const int8_t* ad = a.data<int8_t>(); const int8_t* bd = b.data<int8_t>(); int8_t* od = result.data<int8_t>();
        queue.parallel_for<BitwiseOrI8>(sycl::range<1>(n), [=](sycl::id<1> i) { od[i] = ad[i] | bd[i]; }).wait();
    } else if (a.dtype() == DType::Int16) {
        const int16_t* ad = a.data<int16_t>(); const int16_t* bd = b.data<int16_t>(); int16_t* od = result.data<int16_t>();
        queue.parallel_for<BitwiseOrI16>(sycl::range<1>(n), [=](sycl::id<1> i) { od[i] = ad[i] | bd[i]; }).wait();
    } else if (a.dtype() == DType::Int32) {
        const int32_t* ad = a.data<int32_t>(); const int32_t* bd = b.data<int32_t>(); int32_t* od = result.data<int32_t>();
        queue.parallel_for<BitwiseOrI32>(sycl::range<1>(n), [=](sycl::id<1> i) { od[i] = ad[i] | bd[i]; }).wait();
    } else if (a.dtype() == DType::Int64) {
        const int64_t* ad = a.data<int64_t>(); const int64_t* bd = b.data<int64_t>(); int64_t* od = result.data<int64_t>();
        queue.parallel_for<BitwiseOrI64>(sycl::range<1>(n), [=](sycl::id<1> i) { od[i] = ad[i] | bd[i]; }).wait();
    } else { throw std::runtime_error("bitwise_or: unsupported dtype (expected Int8/16/32/64)"); }
    return result;
}
auto bitwise_xor_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device()); int64_t n = a.numel();
    if (n == 0) return result;
    if (a.dtype() == DType::Int8) {
        const int8_t* ad = a.data<int8_t>(); const int8_t* bd = b.data<int8_t>(); int8_t* od = result.data<int8_t>();
        queue.parallel_for<BitwiseXorI8>(sycl::range<1>(n), [=](sycl::id<1> i) { od[i] = ad[i] ^ bd[i]; }).wait();
    } else if (a.dtype() == DType::Int16) {
        const int16_t* ad = a.data<int16_t>(); const int16_t* bd = b.data<int16_t>(); int16_t* od = result.data<int16_t>();
        queue.parallel_for<BitwiseXorI16>(sycl::range<1>(n), [=](sycl::id<1> i) { od[i] = ad[i] ^ bd[i]; }).wait();
    } else if (a.dtype() == DType::Int32) {
        const int32_t* ad = a.data<int32_t>(); const int32_t* bd = b.data<int32_t>(); int32_t* od = result.data<int32_t>();
        queue.parallel_for<BitwiseXorI32>(sycl::range<1>(n), [=](sycl::id<1> i) { od[i] = ad[i] ^ bd[i]; }).wait();
    } else if (a.dtype() == DType::Int64) {
        const int64_t* ad = a.data<int64_t>(); const int64_t* bd = b.data<int64_t>(); int64_t* od = result.data<int64_t>();
        queue.parallel_for<BitwiseXorI64>(sycl::range<1>(n), [=](sycl::id<1> i) { od[i] = ad[i] ^ bd[i]; }).wait();
    } else { throw std::runtime_error("bitwise_xor: unsupported dtype (expected Int8/16/32/64)"); }
    return result;
}
auto bitwise_not_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device()); int64_t n = input.numel();
    if (n == 0) return result;
    if (input.dtype() == DType::Int8) {
        const int8_t* id = input.data<int8_t>(); int8_t* od = result.data<int8_t>();
        queue.parallel_for<BitwiseNotI8>(sycl::range<1>(n), [=](sycl::id<1> i) { od[i] = ~id[i]; }).wait();
    } else if (input.dtype() == DType::Int16) {
        const int16_t* id = input.data<int16_t>(); int16_t* od = result.data<int16_t>();
        queue.parallel_for<BitwiseNotI16>(sycl::range<1>(n), [=](sycl::id<1> i) { od[i] = ~id[i]; }).wait();
    } else if (input.dtype() == DType::Int32) {
        const int32_t* id = input.data<int32_t>(); int32_t* od = result.data<int32_t>();
        queue.parallel_for<BitwiseNotI32>(sycl::range<1>(n), [=](sycl::id<1> i) { od[i] = ~id[i]; }).wait();
    } else if (input.dtype() == DType::Int64) {
        const int64_t* id = input.data<int64_t>(); int64_t* od = result.data<int64_t>();
        queue.parallel_for<BitwiseNotI64>(sycl::range<1>(n), [=](sycl::id<1> i) { od[i] = ~id[i]; }).wait();
    } else { throw std::runtime_error("bitwise_not: unsupported dtype (expected Int8/16/32/64)"); }
    return result;
}
auto bitwise_left_shift_kernel(const Tensor& input, const Tensor& shift, sycl::queue& queue) -> Tensor {
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device()); int64_t n = input.numel();
    if (n == 0) return result;
    if (input.dtype() == DType::Int8) {
        const int8_t* id = input.data<int8_t>(); const int8_t* sd = shift.data<int8_t>(); int8_t* od = result.data<int8_t>();
        queue.parallel_for<BitwiseLShiftI8>(sycl::range<1>(n), [=](sycl::id<1> i) { od[i] = id[i] << sd[i]; }).wait();
    } else if (input.dtype() == DType::Int16) {
        const int16_t* id = input.data<int16_t>(); const int16_t* sd = shift.data<int16_t>(); int16_t* od = result.data<int16_t>();
        queue.parallel_for<BitwiseLShiftI16>(sycl::range<1>(n), [=](sycl::id<1> i) { od[i] = id[i] << sd[i]; }).wait();
    } else if (input.dtype() == DType::Int32) {
        const int32_t* id = input.data<int32_t>(); const int32_t* sd = shift.data<int32_t>(); int32_t* od = result.data<int32_t>();
        queue.parallel_for<BitwiseLShiftI32>(sycl::range<1>(n), [=](sycl::id<1> i) { od[i] = id[i] << sd[i]; }).wait();
    } else if (input.dtype() == DType::Int64) {
        const int64_t* id = input.data<int64_t>(); const int64_t* sd = shift.data<int64_t>(); int64_t* od = result.data<int64_t>();
        queue.parallel_for<BitwiseLShiftI64>(sycl::range<1>(n), [=](sycl::id<1> i) { od[i] = id[i] << sd[i]; }).wait();
    } else { throw std::runtime_error("bitwise_left_shift: unsupported dtype (expected Int8/16/32/64)"); }
    return result;
}
auto bitwise_right_shift_kernel(const Tensor& input, const Tensor& shift, sycl::queue& queue) -> Tensor {
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device()); int64_t n = input.numel();
    if (n == 0) return result;
    if (input.dtype() == DType::Int8) {
        const int8_t* id = input.data<int8_t>(); const int8_t* sd = shift.data<int8_t>(); int8_t* od = result.data<int8_t>();
        queue.parallel_for<BitwiseRShiftI8>(sycl::range<1>(n), [=](sycl::id<1> i) { od[i] = id[i] >> sd[i]; }).wait();
    } else if (input.dtype() == DType::Int16) {
        const int16_t* id = input.data<int16_t>(); const int16_t* sd = shift.data<int16_t>(); int16_t* od = result.data<int16_t>();
        queue.parallel_for<BitwiseRShiftI16>(sycl::range<1>(n), [=](sycl::id<1> i) { od[i] = id[i] >> sd[i]; }).wait();
    } else if (input.dtype() == DType::Int32) {
        const int32_t* id = input.data<int32_t>(); const int32_t* sd = shift.data<int32_t>(); int32_t* od = result.data<int32_t>();
        queue.parallel_for<BitwiseRShiftI32>(sycl::range<1>(n), [=](sycl::id<1> i) { od[i] = id[i] >> sd[i]; }).wait();
    } else if (input.dtype() == DType::Int64) {
        const int64_t* id = input.data<int64_t>(); const int64_t* sd = shift.data<int64_t>(); int64_t* od = result.data<int64_t>();
        queue.parallel_for<BitwiseRShiftI64>(sycl::range<1>(n), [=](sycl::id<1> i) { od[i] = id[i] >> sd[i]; }).wait();
    } else { throw std::runtime_error("bitwise_right_shift: unsupported dtype (expected Int8/16/32/64)"); }
    return result;
}

// ============================================================================
// Logcumsumexp kernel - log-cumulative-sum-exp along a dimension
// ============================================================================
auto logcumsumexp_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> Tensor {
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
            // First element
            int64_t flat0 = o * dim_size * inner_size + 0 * inner_size + i;
            float running = in_ptr[flat0];
            out_ptr[flat0] = running;
            // Scan: log(exp(running) + exp(x)) using numerically stable formulation
            for (int64_t d = 1; d < dim_size; ++d) {
                int64_t flat = o * dim_size * inner_size + d * inner_size + i;
                float x = in_ptr[flat];
                float mx = sycl::fmax(running, x);
                running = sycl::log(sycl::exp(running - mx) + sycl::exp(x - mx)) + mx;
                out_ptr[flat] = running;
            }
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for(sycl::range<1>(num_lines), [=](sycl::id<1> idx) {
            int64_t o = idx / inner_size;
            int64_t i = idx % inner_size;
            int64_t flat0 = o * dim_size * inner_size + 0 * inner_size + i;
            double running = in_ptr[flat0];
            out_ptr[flat0] = running;
            for (int64_t d = 1; d < dim_size; ++d) {
                int64_t flat = o * dim_size * inner_size + d * inner_size + i;
                double x = in_ptr[flat];
                double mx = sycl::fmax(running, x);
                running = sycl::log(sycl::exp(running - mx) + sycl::exp(x - mx)) + mx;
                out_ptr[flat] = running;
            }
        }).wait();
    } else {
        throw std::runtime_error("logcumsumexp: unsupported dtype (need Float32 or Float64)");
    }

    return output;
}

// ============================================================================
// Bincount kernel - count occurrences of each value in an integer tensor
// ============================================================================
auto bincount_kernel(const Tensor& input, const Tensor* weights,
                     int64_t minlength, sycl::queue& queue) -> Tensor {
    int64_t numel = input.numel();

    // Determine output size: max(max_val + 1, minlength)
    // We need the max value from the input. Use a device-side reduction.
    int64_t output_size = minlength;
    if (numel > 0) {
        const int64_t* in_ptr = get_data_ptr<const int64_t>(input);
        // Allocate a single int64_t on device for the max
        int64_t* d_max = sycl::malloc_device<int64_t>(1, queue);
        queue.memset(d_max, 0, sizeof(int64_t)).wait();
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int64_t val = in_ptr[idx];
            sycl::atomic_ref<int64_t, sycl::memory_order::relaxed,
                              sycl::memory_scope::device,
                              sycl::access::address_space::global_space>
                atomic_max(d_max[0]);
            // atomic max via compare-and-swap loop
            int64_t old = atomic_max.load();
            while (val > old) {
                if (atomic_max.compare_exchange_weak(old, val)) break;
            }
        }).wait();
        int64_t max_val = 0;
        queue.memcpy(&max_val, d_max, sizeof(int64_t)).wait();
        sycl::free(d_max, queue);
        output_size = std::max(output_size, max_val + 1);
    }

    bool has_weights = (weights != nullptr);
    DType out_dtype = has_weights ? weights->dtype() : DType::Int64;

    Tensor output({output_size}, out_dtype, input.device());
    // Zero-initialize the output
    queue.memset(const_cast<void*>(output.data_ptr()), 0,
                 static_cast<size_t>(output_size) * output.dtype_size()).wait();

    if (numel == 0) return output;

    const int64_t* in_ptr = get_data_ptr<const int64_t>(input);

    if (has_weights && weights->dtype() == DType::Float64) {
        const double* w_ptr = get_data_ptr<const double>(*weights);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int64_t bin = in_ptr[idx];
            double w = w_ptr[idx];
            sycl::atomic_ref<double, sycl::memory_order::relaxed,
                              sycl::memory_scope::device,
                              sycl::access::address_space::global_space>
                atomic_bin(out_ptr[bin]);
            atomic_bin.fetch_add(w);
        }).wait();
    } else if (has_weights && weights->dtype() == DType::Float32) {
        const float* w_ptr = get_data_ptr<const float>(*weights);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int64_t bin = in_ptr[idx];
            float w = w_ptr[idx];
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                              sycl::memory_scope::device,
                              sycl::access::address_space::global_space>
                atomic_bin(out_ptr[bin]);
            atomic_bin.fetch_add(w);
        }).wait();
    } else {
        // No weights: count with int64_t atomics
        int64_t* out_ptr = get_data_ptr<int64_t>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int64_t bin = in_ptr[idx];
            sycl::atomic_ref<int64_t, sycl::memory_order::relaxed,
                              sycl::memory_scope::device,
                              sycl::access::address_space::global_space>
                atomic_bin(out_ptr[bin]);
            atomic_bin.fetch_add(int64_t{1});
        }).wait();
    }

    return output;
}

// ============================================================================
// Rsqrt kernel: output = 1/sqrt(input)
// ============================================================================

auto rsqrt_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());
    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<RsqrtKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::rsqrt(in_ptr[idx]);
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<RsqrtKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::rsqrt(in_ptr[idx]);
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<RsqrtKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::rsqrt(static_cast<float>(in_ptr[idx])));
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<RsqrtKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::rsqrt(bf16_to_f32(in_ptr[idx])));
        }).wait();
    } else {
        throw std::runtime_error("Unsupported dtype for rsqrt");
    }
    return output;
}

// ============================================================================
// Square kernel: output = input * input
// ============================================================================

auto square_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());
    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<SquareKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float v = in_ptr[idx]; out_ptr[idx] = v * v;
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<SquareKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double v = in_ptr[idx]; out_ptr[idx] = v * v;
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<SquareKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float v = static_cast<float>(in_ptr[idx]); out_ptr[idx] = sycl::half(v * v);
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<SquareKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float v = bf16_to_f32(in_ptr[idx]); out_ptr[idx] = f32_to_bf16(v * v);
        }).wait();
    } else {
        throw std::runtime_error("Unsupported dtype for square");
    }
    return output;
}

// ============================================================================
// Asinh kernel
// ============================================================================

auto asinh_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());
    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<AsinhKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::asinh(in_ptr[idx]);
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<AsinhKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::asinh(in_ptr[idx]);
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<AsinhKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::asinh(static_cast<float>(in_ptr[idx])));
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<AsinhKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::asinh(bf16_to_f32(in_ptr[idx])));
        }).wait();
    } else {
        throw std::runtime_error("Unsupported dtype for asinh");
    }
    return output;
}

// ============================================================================
// Acosh kernel
// ============================================================================

auto acosh_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());
    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<AcoshKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::acosh(in_ptr[idx]);
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<AcoshKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::acosh(in_ptr[idx]);
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<AcoshKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::acosh(static_cast<float>(in_ptr[idx])));
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<AcoshKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::acosh(bf16_to_f32(in_ptr[idx])));
        }).wait();
    } else {
        throw std::runtime_error("Unsupported dtype for acosh");
    }
    return output;
}

// ============================================================================
// Atanh kernel
// ============================================================================

auto atanh_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());
    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<AtanhKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::atanh(in_ptr[idx]);
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<AtanhKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::atanh(in_ptr[idx]);
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<AtanhKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::atanh(static_cast<float>(in_ptr[idx])));
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<AtanhKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::atanh(bf16_to_f32(in_ptr[idx])));
        }).wait();
    } else {
        throw std::runtime_error("Unsupported dtype for atanh");
    }
    return output;
}

// ============================================================================
// Hypot kernel: output = sqrt(a*a + b*b)
// ============================================================================

auto hypot_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    if (a.dtype() != b.dtype()) throw std::invalid_argument("hypot: input dtypes must match");
    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  a.dtype(), a.device());
    const int64_t numel = a.numel();

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<HypotKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::hypot(a_ptr[idx], b_ptr[idx]);
        }).wait();
    } else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<HypotKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::hypot(a_ptr[idx], b_ptr[idx]);
        }).wait();
    } else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<HypotKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::hypot(static_cast<float>(a_ptr[idx]), static_cast<float>(b_ptr[idx])));
        }).wait();
    } else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<HypotKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::hypot(bf16_to_f32(a_ptr[idx]), bf16_to_f32(b_ptr[idx])));
        }).wait();
    } else {
        throw std::runtime_error("hypot: unsupported dtype");
    }
    return output;
}

// ============================================================================
// Copysign kernel
// ============================================================================

auto copysign_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    if (a.dtype() != b.dtype()) throw std::invalid_argument("copysign: input dtypes must match");
    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  a.dtype(), a.device());
    const int64_t numel = a.numel();

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<CopysignKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::copysign(a_ptr[idx], b_ptr[idx]);
        }).wait();
    } else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<CopysignKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::copysign(a_ptr[idx], b_ptr[idx]);
        }).wait();
    } else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<CopysignKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::copysign(static_cast<float>(a_ptr[idx]), static_cast<float>(b_ptr[idx])));
        }).wait();
    } else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<CopysignKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::copysign(bf16_to_f32(a_ptr[idx]), bf16_to_f32(b_ptr[idx])));
        }).wait();
    } else {
        throw std::runtime_error("copysign: unsupported dtype");
    }
    return output;
}

// ============================================================================
// Nextafter kernel
// ============================================================================

auto nextafter_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    if (a.dtype() != b.dtype()) throw std::invalid_argument("nextafter: input dtypes must match");
    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  a.dtype(), a.device());
    const int64_t numel = a.numel();

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<NextafterKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::nextafter(a_ptr[idx], b_ptr[idx]);
        }).wait();
    } else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<NextafterKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::nextafter(a_ptr[idx], b_ptr[idx]);
        }).wait();
    } else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<NextafterKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float va = static_cast<float>(a_ptr[idx]);
            float vb = static_cast<float>(b_ptr[idx]);
            out_ptr[idx] = sycl::half(sycl::nextafter(va, vb));
        }).wait();
    } else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<NextafterKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::nextafter(bf16_to_f32(a_ptr[idx]), bf16_to_f32(b_ptr[idx])));
        }).wait();
    } else {
        throw std::runtime_error("nextafter: unsupported dtype");
    }
    return output;
}

// ============================================================================
// Gcd kernel (integer types only)
// ============================================================================

auto gcd_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    if (a.dtype() != b.dtype()) throw std::invalid_argument("gcd: input dtypes must match");
    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  a.dtype(), a.device());
    const int64_t numel = a.numel();

    if (a.dtype() == DType::Int32) {
        const int32_t* a_ptr = get_data_ptr<const int32_t>(a);
        const int32_t* b_ptr = get_data_ptr<const int32_t>(b);
        int32_t* out_ptr = get_data_ptr<int32_t>(output);
        queue.parallel_for<GcdKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int32_t x = a_ptr[idx] < 0 ? -a_ptr[idx] : a_ptr[idx];
            int32_t y = b_ptr[idx] < 0 ? -b_ptr[idx] : b_ptr[idx];
            while (y != 0) { int32_t t = y; y = x % y; x = t; }
            out_ptr[idx] = x;
        }).wait();
    } else if (a.dtype() == DType::Int64) {
        const int64_t* a_ptr = get_data_ptr<const int64_t>(a);
        const int64_t* b_ptr = get_data_ptr<const int64_t>(b);
        int64_t* out_ptr = get_data_ptr<int64_t>(output);
        queue.parallel_for<GcdKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int64_t x = a_ptr[idx] < 0 ? -a_ptr[idx] : a_ptr[idx];
            int64_t y = b_ptr[idx] < 0 ? -b_ptr[idx] : b_ptr[idx];
            while (y != 0) { int64_t t = y; y = x % y; x = t; }
            out_ptr[idx] = x;
        }).wait();
    } else {
        throw std::runtime_error("gcd: only Int32 and Int64 dtypes supported");
    }
    return output;
}

// ============================================================================
// Lcm kernel (integer types only)
// ============================================================================

auto lcm_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    if (a.dtype() != b.dtype()) throw std::invalid_argument("lcm: input dtypes must match");
    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  a.dtype(), a.device());
    const int64_t numel = a.numel();

    if (a.dtype() == DType::Int32) {
        const int32_t* a_ptr = get_data_ptr<const int32_t>(a);
        const int32_t* b_ptr = get_data_ptr<const int32_t>(b);
        int32_t* out_ptr = get_data_ptr<int32_t>(output);
        queue.parallel_for<LcmKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int32_t x = a_ptr[idx] < 0 ? -a_ptr[idx] : a_ptr[idx];
            int32_t y = b_ptr[idx] < 0 ? -b_ptr[idx] : b_ptr[idx];
            int32_t gx = x, gy = y;
            while (gy != 0) { int32_t t = gy; gy = gx % gy; gx = t; }
            out_ptr[idx] = gx == 0 ? 0 : (x / gx) * y;
        }).wait();
    } else if (a.dtype() == DType::Int64) {
        const int64_t* a_ptr = get_data_ptr<const int64_t>(a);
        const int64_t* b_ptr = get_data_ptr<const int64_t>(b);
        int64_t* out_ptr = get_data_ptr<int64_t>(output);
        queue.parallel_for<LcmKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int64_t x = a_ptr[idx] < 0 ? -a_ptr[idx] : a_ptr[idx];
            int64_t y = b_ptr[idx] < 0 ? -b_ptr[idx] : b_ptr[idx];
            int64_t gx = x, gy = y;
            while (gy != 0) { int64_t t = gy; gy = gx % gy; gx = t; }
            out_ptr[idx] = gx == 0 ? 0 : (x / gx) * y;
        }).wait();
    } else {
        throw std::runtime_error("lcm: only Int32 and Int64 dtypes supported");
    }
    return output;
}

// ============================================================================
// Igamma kernel (lower regularized incomplete gamma)
// ============================================================================

auto igamma_kernel(const Tensor& a, const Tensor& x, sycl::queue& queue) -> Tensor {
    if (a.dtype() != x.dtype()) throw std::invalid_argument("igamma: input dtypes must match");
    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  a.dtype(), a.device());
    const int64_t numel = a.numel();

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* x_ptr = get_data_ptr<const float>(x);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<IgammaKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float av = a_ptr[idx], xv = x_ptr[idx];
            if (xv <= 0.0f) { out_ptr[idx] = 0.0f; return; }
            float term = 1.0f / av, sum = term;
            for (int n = 1; n < 200; ++n) {
                term *= xv / (av + static_cast<float>(n));
                sum += term;
                if (sycl::fabs(term) < sycl::fabs(sum) * 1e-7f) break;
            }
            out_ptr[idx] = sycl::exp(-xv + av * sycl::log(xv) - sycl::lgamma(av)) * sum;
        }).wait();
    } else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* x_ptr = get_data_ptr<const double>(x);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<IgammaKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double av = a_ptr[idx], xv = x_ptr[idx];
            if (xv <= 0.0) { out_ptr[idx] = 0.0; return; }
            double term = 1.0 / av, sum = term;
            for (int n = 1; n < 500; ++n) {
                term *= xv / (av + static_cast<double>(n));
                sum += term;
                if (sycl::fabs(term) < sycl::fabs(sum) * 1e-15) break;
            }
            out_ptr[idx] = sycl::exp(-xv + av * sycl::log(xv) - sycl::lgamma(av)) * sum;
        }).wait();
    } else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* x_ptr = get_data_ptr<const sycl::half>(x);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<IgammaKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float av = static_cast<float>(a_ptr[idx]), xv = static_cast<float>(x_ptr[idx]);
            if (xv <= 0.0f) { out_ptr[idx] = sycl::half(0.0f); return; }
            float term = 1.0f / av, sum = term;
            for (int n = 1; n < 200; ++n) {
                term *= xv / (av + static_cast<float>(n));
                sum += term;
                if (sycl::fabs(term) < sycl::fabs(sum) * 1e-7f) break;
            }
            out_ptr[idx] = sycl::half(sycl::exp(-xv + av * sycl::log(xv) - sycl::lgamma(av)) * sum);
        }).wait();
    } else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* x_ptr = get_data_ptr<const uint16_t>(x);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<IgammaKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float av = bf16_to_f32(a_ptr[idx]), xv = bf16_to_f32(x_ptr[idx]);
            if (xv <= 0.0f) { out_ptr[idx] = f32_to_bf16(0.0f); return; }
            float term = 1.0f / av, sum = term;
            for (int n = 1; n < 200; ++n) {
                term *= xv / (av + static_cast<float>(n));
                sum += term;
                if (sycl::fabs(term) < sycl::fabs(sum) * 1e-7f) break;
            }
            out_ptr[idx] = f32_to_bf16(sycl::exp(-xv + av * sycl::log(xv) - sycl::lgamma(av)) * sum);
        }).wait();
    } else {
        throw std::runtime_error("igamma: unsupported dtype");
    }
    return output;
}

// ============================================================================
// Igammac kernel (upper regularized incomplete gamma = 1 - igamma)
// ============================================================================

auto igammac_kernel(const Tensor& a, const Tensor& x, sycl::queue& queue) -> Tensor {
    if (a.dtype() != x.dtype()) throw std::invalid_argument("igammac: input dtypes must match");
    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  a.dtype(), a.device());
    const int64_t numel = a.numel();

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* x_ptr = get_data_ptr<const float>(x);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<IgammacKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float av = a_ptr[idx], xv = x_ptr[idx];
            if (xv <= 0.0f) { out_ptr[idx] = 1.0f; return; }
            float term = 1.0f / av, sum = term;
            for (int n = 1; n < 200; ++n) {
                term *= xv / (av + static_cast<float>(n));
                sum += term;
                if (sycl::fabs(term) < sycl::fabs(sum) * 1e-7f) break;
            }
            out_ptr[idx] = 1.0f - sycl::exp(-xv + av * sycl::log(xv) - sycl::lgamma(av)) * sum;
        }).wait();
    } else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* x_ptr = get_data_ptr<const double>(x);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<IgammacKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double av = a_ptr[idx], xv = x_ptr[idx];
            if (xv <= 0.0) { out_ptr[idx] = 1.0; return; }
            double term = 1.0 / av, sum = term;
            for (int n = 1; n < 500; ++n) {
                term *= xv / (av + static_cast<double>(n));
                sum += term;
                if (sycl::fabs(term) < sycl::fabs(sum) * 1e-15) break;
            }
            out_ptr[idx] = 1.0 - sycl::exp(-xv + av * sycl::log(xv) - sycl::lgamma(av)) * sum;
        }).wait();
    } else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* x_ptr = get_data_ptr<const sycl::half>(x);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<IgammacKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float av = static_cast<float>(a_ptr[idx]), xv = static_cast<float>(x_ptr[idx]);
            if (xv <= 0.0f) { out_ptr[idx] = sycl::half(1.0f); return; }
            float term = 1.0f / av, sum = term;
            for (int n = 1; n < 200; ++n) {
                term *= xv / (av + static_cast<float>(n));
                sum += term;
                if (sycl::fabs(term) < sycl::fabs(sum) * 1e-7f) break;
            }
            out_ptr[idx] = sycl::half(1.0f - sycl::exp(-xv + av * sycl::log(xv) - sycl::lgamma(av)) * sum);
        }).wait();
    } else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* x_ptr = get_data_ptr<const uint16_t>(x);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<IgammacKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float av = bf16_to_f32(a_ptr[idx]), xv = bf16_to_f32(x_ptr[idx]);
            if (xv <= 0.0f) { out_ptr[idx] = f32_to_bf16(1.0f); return; }
            float term = 1.0f / av, sum = term;
            for (int n = 1; n < 200; ++n) {
                term *= xv / (av + static_cast<float>(n));
                sum += term;
                if (sycl::fabs(term) < sycl::fabs(sum) * 1e-7f) break;
            }
            out_ptr[idx] = f32_to_bf16(1.0f - sycl::exp(-xv + av * sycl::log(xv) - sycl::lgamma(av)) * sum);
        }).wait();
    } else {
        throw std::runtime_error("igammac: unsupported dtype");
    }
    return output;
}

// ============================================================================
// Addcmul kernel: output = input + value * tensor1 * tensor2
// ============================================================================

auto addcmul_kernel(const Tensor& input, const Tensor& tensor1, const Tensor& tensor2,
                    float value, sycl::queue& queue) -> Tensor {
    if (input.dtype() != tensor1.dtype() || input.dtype() != tensor2.dtype())
        throw std::invalid_argument("addcmul: all input dtypes must match");
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());
    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* i_ptr = get_data_ptr<const float>(input);
        const float* t1_ptr = get_data_ptr<const float>(tensor1);
        const float* t2_ptr = get_data_ptr<const float>(tensor2);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<AddcmulKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = i_ptr[idx] + value * t1_ptr[idx] * t2_ptr[idx];
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* i_ptr = get_data_ptr<const double>(input);
        const double* t1_ptr = get_data_ptr<const double>(tensor1);
        const double* t2_ptr = get_data_ptr<const double>(tensor2);
        double* out_ptr = get_data_ptr<double>(output);
        double dvalue = static_cast<double>(value);
        queue.parallel_for<AddcmulKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = i_ptr[idx] + dvalue * t1_ptr[idx] * t2_ptr[idx];
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* i_ptr = get_data_ptr<const sycl::half>(input);
        const sycl::half* t1_ptr = get_data_ptr<const sycl::half>(tensor1);
        const sycl::half* t2_ptr = get_data_ptr<const sycl::half>(tensor2);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<AddcmulKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float vi = static_cast<float>(i_ptr[idx]);
            float v1 = static_cast<float>(t1_ptr[idx]);
            float v2 = static_cast<float>(t2_ptr[idx]);
            out_ptr[idx] = sycl::half(vi + value * v1 * v2);
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* i_ptr = get_data_ptr<const uint16_t>(input);
        const uint16_t* t1_ptr = get_data_ptr<const uint16_t>(tensor1);
        const uint16_t* t2_ptr = get_data_ptr<const uint16_t>(tensor2);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<AddcmulKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float vi = bf16_to_f32(i_ptr[idx]);
            float v1 = bf16_to_f32(t1_ptr[idx]);
            float v2 = bf16_to_f32(t2_ptr[idx]);
            out_ptr[idx] = f32_to_bf16(vi + value * v1 * v2);
        }).wait();
    } else {
        throw std::runtime_error("addcmul: unsupported dtype");
    }
    return output;
}

// ============================================================================
// Addcdiv kernel: output = input + value * tensor1 / tensor2
// ============================================================================

auto addcdiv_kernel(const Tensor& input, const Tensor& tensor1, const Tensor& tensor2,
                    float value, sycl::queue& queue) -> Tensor {
    if (input.dtype() != tensor1.dtype() || input.dtype() != tensor2.dtype())
        throw std::invalid_argument("addcdiv: all input dtypes must match");
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());
    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* i_ptr = get_data_ptr<const float>(input);
        const float* t1_ptr = get_data_ptr<const float>(tensor1);
        const float* t2_ptr = get_data_ptr<const float>(tensor2);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<AddcdivKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = i_ptr[idx] + value * t1_ptr[idx] / t2_ptr[idx];
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* i_ptr = get_data_ptr<const double>(input);
        const double* t1_ptr = get_data_ptr<const double>(tensor1);
        const double* t2_ptr = get_data_ptr<const double>(tensor2);
        double* out_ptr = get_data_ptr<double>(output);
        double dvalue = static_cast<double>(value);
        queue.parallel_for<AddcdivKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = i_ptr[idx] + dvalue * t1_ptr[idx] / t2_ptr[idx];
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* i_ptr = get_data_ptr<const sycl::half>(input);
        const sycl::half* t1_ptr = get_data_ptr<const sycl::half>(tensor1);
        const sycl::half* t2_ptr = get_data_ptr<const sycl::half>(tensor2);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<AddcdivKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float vi = static_cast<float>(i_ptr[idx]);
            float v1 = static_cast<float>(t1_ptr[idx]);
            float v2 = static_cast<float>(t2_ptr[idx]);
            out_ptr[idx] = sycl::half(vi + value * v1 / v2);
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* i_ptr = get_data_ptr<const uint16_t>(input);
        const uint16_t* t1_ptr = get_data_ptr<const uint16_t>(tensor1);
        const uint16_t* t2_ptr = get_data_ptr<const uint16_t>(tensor2);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<AddcdivKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float vi = bf16_to_f32(i_ptr[idx]);
            float v1 = bf16_to_f32(t1_ptr[idx]);
            float v2 = bf16_to_f32(t2_ptr[idx]);
            out_ptr[idx] = f32_to_bf16(vi + value * v1 / v2);
        }).wait();
    } else {
        throw std::runtime_error("addcdiv: unsupported dtype");
    }
    return output;
}

// ============================================================================
// CumMax kernel — cumulative maximum along a dimension (returns values + indices)
// ============================================================================

struct CumMaxKernelFloat32 {};
struct CumMaxKernelFloat64 {};
struct CumMinKernelFloat32 {};
struct CumMinKernelFloat64 {};
struct FmaxKernelFloat32 {};
struct FmaxKernelFloat64 {};
struct FmaxKernelInt32 {};
struct FminKernelFloat32 {};
struct FminKernelFloat64 {};
struct FminKernelInt32 {};
struct IsinKernelFloat32 {};
struct IsinKernelFloat64 {};
struct IsinKernelInt32 {};
struct IsinKernelInt64 {};
struct IsinSortKernelFloat32 {};
struct IsinSortKernelFloat64 {};
struct IsinSortKernelInt32 {};
struct IsinSortKernelInt64 {};
struct KthvalueSliceKernelFloat32 {};
struct KthvalueSliceKernelFloat64 {};
struct KthvalueSliceKernelInt32 {};
struct KthvalueSliceKernelInt64 {};
struct KthvalueIdxInitKernelFloat32 {};
struct KthvalueIdxInitKernelFloat64 {};
struct KthvalueIdxInitKernelInt32 {};
struct KthvalueIdxInitKernelInt64 {};
struct QuantileInterpKernelFloat32 {};
struct QuantileInterpKernelFloat64 {};
struct QuantileNanFilterFloat32 {};
struct QuantileNanFilterFloat64 {};
struct HistcKernelFloat32 {};
struct HistcBinKernelFloat32 {};
struct HistcBinKernelFloat64 {};
struct UniqueConsecutiveMaskFloat32 {};
struct UniqueConsecutiveMaskFloat64 {};
struct UniqueConsecutiveMaskInt32 {};
struct UniqueConsecutiveMaskInt64 {};
struct UniqueConsecutiveGatherFloat32 {};
struct UniqueConsecutiveGatherFloat64 {};
struct UniqueConsecutiveGatherInt32 {};
struct UniqueConsecutiveGatherInt64 {};
struct UniqueConsecutiveCountsFloat32 {};
struct UniqueConsecutiveCountsFloat64 {};
struct UniqueConsecutiveCountsInt32 {};
struct UniqueConsecutiveCountsInt64 {};

auto cummax_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> std::pair<Tensor, Tensor>
{
    Tensor input_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);
    const auto& shape = input_cont.shape();
    const int64_t ndim = input_cont.ndim();
    const int64_t dim_size = shape[dim];
    const auto dtype = input_cont.dtype();
    const auto device = input_cont.device();

    Tensor values(std::vector<int64_t>(shape.begin(), shape.end()), dtype, device);
    Tensor indices_out(std::vector<int64_t>(shape.begin(), shape.end()), DType::Int64, device);

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];
    int64_t total_slices = outer_size * inner_size;

    auto launch = [&]<typename T, typename KernelName>() {
        const T* in_ptr = get_data_ptr<const T>(input_cont);
        T* val_ptr = get_data_ptr<T>(values);
        int64_t* idx_ptr = get_data_ptr<int64_t>(indices_out);

        queue.parallel_for<KernelName>(sycl::range<1>(total_slices), [=](sycl::id<1> id) {
            int64_t slice = id[0];
            int64_t outer = slice / inner_size;
            int64_t inner = slice % inner_size;

            T running_max = in_ptr[outer * dim_size * inner_size + inner];
            int64_t running_idx = 0;
            val_ptr[outer * dim_size * inner_size + inner] = running_max;
            idx_ptr[outer * dim_size * inner_size + inner] = 0;

            for (int64_t i = 1; i < dim_size; ++i) {
                int64_t offset = outer * dim_size * inner_size + i * inner_size + inner;
                T val = in_ptr[offset];
                if (val > running_max) {
                    running_max = val;
                    running_idx = i;
                }
                val_ptr[offset] = running_max;
                idx_ptr[offset] = running_idx;
            }
        }).wait();
    };

    switch (dtype) {
        case DType::Float32: launch.template operator()<float, CumMaxKernelFloat32>(); break;
        case DType::Float64: launch.template operator()<double, CumMaxKernelFloat64>(); break;
        default: throw std::runtime_error("cummax OneAPI: unsupported dtype");
    }
    return {values, indices_out};
}

// ============================================================================
// CumMin kernel — cumulative minimum along a dimension (returns values + indices)
// ============================================================================

auto cummin_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> std::pair<Tensor, Tensor>
{
    Tensor input_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);
    const auto& shape = input_cont.shape();
    const int64_t ndim = input_cont.ndim();
    const int64_t dim_size = shape[dim];
    const auto dtype = input_cont.dtype();
    const auto device = input_cont.device();

    Tensor values(std::vector<int64_t>(shape.begin(), shape.end()), dtype, device);
    Tensor indices_out(std::vector<int64_t>(shape.begin(), shape.end()), DType::Int64, device);

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];
    int64_t total_slices = outer_size * inner_size;

    auto launch = [&]<typename T, typename KernelName>() {
        const T* in_ptr = get_data_ptr<const T>(input_cont);
        T* val_ptr = get_data_ptr<T>(values);
        int64_t* idx_ptr = get_data_ptr<int64_t>(indices_out);

        queue.parallel_for<KernelName>(sycl::range<1>(total_slices), [=](sycl::id<1> id) {
            int64_t slice = id[0];
            int64_t outer = slice / inner_size;
            int64_t inner = slice % inner_size;

            T running_min = in_ptr[outer * dim_size * inner_size + inner];
            int64_t running_idx = 0;
            val_ptr[outer * dim_size * inner_size + inner] = running_min;
            idx_ptr[outer * dim_size * inner_size + inner] = 0;

            for (int64_t i = 1; i < dim_size; ++i) {
                int64_t offset = outer * dim_size * inner_size + i * inner_size + inner;
                T val = in_ptr[offset];
                if (val < running_min) {
                    running_min = val;
                    running_idx = i;
                }
                val_ptr[offset] = running_min;
                idx_ptr[offset] = running_idx;
            }
        }).wait();
    };

    switch (dtype) {
        case DType::Float32: launch.template operator()<float, CumMinKernelFloat32>(); break;
        case DType::Float64: launch.template operator()<double, CumMinKernelFloat64>(); break;
        default: throw std::runtime_error("cummin OneAPI: unsupported dtype");
    }
    return {values, indices_out};
}

// ============================================================================
// Fmax kernel — NaN-aware element-wise maximum (sycl::fmax handles NaN per IEEE 754)
// ============================================================================

auto fmax_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor
{
    // Float16 / BFloat16: widen to Float32, compute, narrow back.
    // sycl::fmax is not overloaded for sycl::half in all toolchain versions,
    // so route through the Float32 path.
    if (a.dtype() == DType::Float16 || a.dtype() == DType::BFloat16) {
        const DType orig_dtype = a.dtype();
        Tensor result = fmax_kernel(a.to(DType::Float32), b.to(DType::Float32), queue);
        return result.to(orig_dtype);
    }

    Tensor a_cont = a.is_contiguous() ? a : contiguous_kernel(a, queue);
    Tensor b_cont = b.is_contiguous() ? b : contiguous_kernel(b, queue);
    int64_t n = a_cont.numel();
    Tensor output(std::vector<int64_t>(a_cont.shape().begin(), a_cont.shape().end()),
                  a_cont.dtype(), a_cont.device());

    switch (a_cont.dtype()) {
        case DType::Float32: {
            const float* ap = get_data_ptr<const float>(a_cont);
            const float* bp = get_data_ptr<const float>(b_cont);
            float* op = get_data_ptr<float>(output);
            queue.parallel_for<FmaxKernelFloat32>(sycl::range<1>(n), [=](sycl::id<1> idx) {
                op[idx] = sycl::fmax(ap[idx], bp[idx]);
            }).wait();
            break;
        }
        case DType::Float64: {
            const double* ap = get_data_ptr<const double>(a_cont);
            const double* bp = get_data_ptr<const double>(b_cont);
            double* op = get_data_ptr<double>(output);
            queue.parallel_for<FmaxKernelFloat64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
                op[idx] = sycl::fmax(ap[idx], bp[idx]);
            }).wait();
            break;
        }
        case DType::Int32: {
            const int32_t* ap = get_data_ptr<const int32_t>(a_cont);
            const int32_t* bp = get_data_ptr<const int32_t>(b_cont);
            int32_t* op = get_data_ptr<int32_t>(output);
            queue.parallel_for<FmaxKernelInt32>(sycl::range<1>(n), [=](sycl::id<1> idx) {
                op[idx] = sycl::max(ap[idx], bp[idx]);
            }).wait();
            break;
        }
        default:
            throw std::runtime_error("fmax OneAPI: unsupported dtype");
    }
    return output;
}

// ============================================================================
// Fmin kernel — NaN-aware element-wise minimum (sycl::fmin handles NaN per IEEE 754)
// ============================================================================

auto fmin_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor
{
    // Float16 / BFloat16: widen to Float32, compute, narrow back.
    if (a.dtype() == DType::Float16 || a.dtype() == DType::BFloat16) {
        const DType orig_dtype = a.dtype();
        Tensor result = fmin_kernel(a.to(DType::Float32), b.to(DType::Float32), queue);
        return result.to(orig_dtype);
    }

    Tensor a_cont = a.is_contiguous() ? a : contiguous_kernel(a, queue);
    Tensor b_cont = b.is_contiguous() ? b : contiguous_kernel(b, queue);
    int64_t n = a_cont.numel();
    Tensor output(std::vector<int64_t>(a_cont.shape().begin(), a_cont.shape().end()),
                  a_cont.dtype(), a_cont.device());

    switch (a_cont.dtype()) {
        case DType::Float32: {
            const float* ap = get_data_ptr<const float>(a_cont);
            const float* bp = get_data_ptr<const float>(b_cont);
            float* op = get_data_ptr<float>(output);
            queue.parallel_for<FminKernelFloat32>(sycl::range<1>(n), [=](sycl::id<1> idx) {
                op[idx] = sycl::fmin(ap[idx], bp[idx]);
            }).wait();
            break;
        }
        case DType::Float64: {
            const double* ap = get_data_ptr<const double>(a_cont);
            const double* bp = get_data_ptr<const double>(b_cont);
            double* op = get_data_ptr<double>(output);
            queue.parallel_for<FminKernelFloat64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
                op[idx] = sycl::fmin(ap[idx], bp[idx]);
            }).wait();
            break;
        }
        case DType::Int32: {
            const int32_t* ap = get_data_ptr<const int32_t>(a_cont);
            const int32_t* bp = get_data_ptr<const int32_t>(b_cont);
            int32_t* op = get_data_ptr<int32_t>(output);
            queue.parallel_for<FminKernelInt32>(sycl::range<1>(n), [=](sycl::id<1> idx) {
                op[idx] = sycl::min(ap[idx], bp[idx]);
            }).wait();
            break;
        }
        default:
            throw std::runtime_error("fmin OneAPI: unsupported dtype");
    }
    return output;
}

// ============================================================================
// Isin kernel — set membership test using sorted + binary search
// ============================================================================

auto isin_kernel(const Tensor& elements, const Tensor& test_elements, sycl::queue& queue) -> Tensor
{
    Tensor elem_cont = elements.is_contiguous() ? elements : contiguous_kernel(elements, queue);
    Tensor test_cont = test_elements.is_contiguous() ? test_elements : contiguous_kernel(test_elements, queue);

    int64_t num_elements = elem_cont.numel();
    int64_t num_test = test_cont.numel();
    Tensor output(std::vector<int64_t>(elem_cont.shape().begin(), elem_cont.shape().end()),
                  DType::Bool, elem_cont.device());

    auto launch = [&]<typename T, typename SearchKernel>() {
        // Sort test_elements on device
        T* sorted_buf = sycl::malloc_device<T>(num_test, queue);
        queue.memcpy(sorted_buf, get_data_ptr<const T>(test_cont),
                     num_test * sizeof(T)).wait();

#ifdef TENZOR_HAS_ONEDPL
        auto policy = ::oneapi::dpl::execution::make_device_policy(queue);
        ::oneapi::dpl::sort(policy, sorted_buf, sorted_buf + num_test);
#else
        // Fallback: device-side bitonic sort when oneDPL unavailable
        sycl_bitonic_sort(sorted_buf, num_test, queue);
#endif

        const T* elem_ptr = get_data_ptr<const T>(elem_cont);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<SearchKernel>(sycl::range<1>(num_elements), [=](sycl::id<1> idx) {
            T val = elem_ptr[idx];
            int64_t lo = 0, hi = num_test - 1;
            bool found = false;
            while (lo <= hi) {
                int64_t mid = lo + (hi - lo) / 2;
                T mid_val = sorted_buf[mid];
                if (mid_val == val) { found = true; break; }
                else if (mid_val < val) lo = mid + 1;
                else hi = mid - 1;
            }
            out_ptr[idx] = found;
        }).wait();

        sycl::free(sorted_buf, queue);
    };

    switch (elem_cont.dtype()) {
        case DType::Float32: launch.template operator()<float, IsinKernelFloat32>(); break;
        case DType::Float64: launch.template operator()<double, IsinKernelFloat64>(); break;
        case DType::Int32:   launch.template operator()<int32_t, IsinKernelInt32>(); break;
        case DType::Int64:   launch.template operator()<int64_t, IsinKernelInt64>(); break;
        default: throw std::runtime_error("isin OneAPI: unsupported dtype");
    }
    return output;
}

// ============================================================================
// Kthvalue kernel — k-th smallest value along a dimension
// ============================================================================

auto kthvalue_kernel(const Tensor& input, int64_t k, int64_t dim, bool keepdim,
                     sycl::queue& queue) -> std::pair<Tensor, Tensor>
{
    Tensor input_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);
    const auto& shape = input_cont.shape();
    const int64_t ndim = input_cont.ndim();
    // Normalize negative dim (dim=-1 ⇒ last axis). The kernel-registry default
    // for these reductions is -1, meaning "no dim was specified" from the
    // caller; for 1D inputs this is the only reduction dim, for higher-rank
    // inputs it maps to the last axis.
    if (dim < 0) dim += ndim;
    const int64_t dim_size = shape[dim];
    const auto dtype = input_cont.dtype();
    const auto device = input_cont.device();

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];
    int64_t total_slices = outer_size * inner_size;

    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(shape[i]);
        }
    }
    if (out_shape.empty()) out_shape.push_back(1);

    Tensor values(out_shape, dtype, device);
    Tensor indices_out(out_shape, DType::Int64, device);

    auto launch = [&]<typename T, typename IdxInitKernel>() {
        int64_t in_numel = input_cont.numel();

        if (inner_size == 1) {
            // Contiguous slices along last dims — sort per slice on device
#ifdef TENZOR_HAS_ONEDPL
            auto policy = ::oneapi::dpl::execution::make_device_policy(queue);

            T* tmp_vals = sycl::malloc_device<T>(in_numel, queue);
            int64_t* tmp_idx = sycl::malloc_device<int64_t>(in_numel, queue);

            queue.memcpy(tmp_vals, get_data_ptr<const T>(input_cont),
                         in_numel * sizeof(T)).wait();

            // Initialize indices: each slice gets 0..dim_size-1
            queue.parallel_for<IdxInitKernel>(sycl::range<1>(in_numel),
                [=](sycl::id<1> gid) {
                    tmp_idx[gid] = static_cast<int64_t>(gid[0]) % dim_size;
                }).wait();

            T* out_val_ptr = get_data_ptr<T>(values);
            int64_t* out_idx_ptr = get_data_ptr<int64_t>(indices_out);

            for (int64_t o = 0; o < outer_size; ++o) {
                T* slice_vals = tmp_vals + o * dim_size;
                int64_t* slice_idx = tmp_idx + o * dim_size;
                ::oneapi::dpl::sort_by_key(policy, slice_vals, slice_vals + dim_size, slice_idx);
                // Copy k-th element (0-indexed: k-1)
                queue.memcpy(out_val_ptr + o, slice_vals + (k - 1), sizeof(T)).wait();
                queue.memcpy(out_idx_ptr + o, slice_idx + (k - 1), sizeof(int64_t)).wait();
            }

            sycl::free(tmp_vals, queue);
            sycl::free(tmp_idx, queue);
#else
            // Fallback: device-side bitonic sort when oneDPL unavailable
            T* tmp_vals = sycl::malloc_device<T>(in_numel, queue);
            int64_t* tmp_idx = sycl::malloc_device<int64_t>(in_numel, queue);

            queue.memcpy(tmp_vals, get_data_ptr<const T>(input_cont),
                         in_numel * sizeof(T)).wait();

            // Initialize indices: each slice gets 0..dim_size-1
            queue.parallel_for(sycl::range<1>(in_numel),
                [=](sycl::id<1> gid) {
                    tmp_idx[gid] = static_cast<int64_t>(gid[0]) % dim_size;
                }).wait();

            T* out_val_ptr = get_data_ptr<T>(values);
            int64_t* out_idx_ptr = get_data_ptr<int64_t>(indices_out);

            for (int64_t o = 0; o < outer_size; ++o) {
                T* slice_vals = tmp_vals + o * dim_size;
                int64_t* slice_idx = tmp_idx + o * dim_size;
                sycl_bitonic_sort_by_key(slice_vals, slice_idx, dim_size, queue);
                // Copy k-th element (0-indexed: k-1)
                queue.memcpy(out_val_ptr + o, slice_vals + (k - 1), sizeof(T)).wait();
                queue.memcpy(out_idx_ptr + o, slice_idx + (k - 1), sizeof(int64_t)).wait();
            }

            sycl::free(tmp_vals, queue);
            sycl::free(tmp_idx, queue);
#endif
        } else {
            // Non-contiguous slices: gather per slice, sort on device, scatter result
#ifdef TENZOR_HAS_ONEDPL
            auto policy = ::oneapi::dpl::execution::make_device_policy(queue);
            T* slice_buf = sycl::malloc_device<T>(dim_size, queue);
            int64_t* idx_buf = sycl::malloc_device<int64_t>(dim_size, queue);
            const T* in_ptr = get_data_ptr<const T>(input_cont);
            T* out_val_ptr = get_data_ptr<T>(values);
            int64_t* out_idx_ptr = get_data_ptr<int64_t>(indices_out);

            for (int64_t s = 0; s < total_slices; ++s) {
                int64_t outer = s / inner_size;
                int64_t inner = s % inner_size;
                int64_t base = outer * dim_size * inner_size + inner;

                // Gather slice to contiguous buffer on device
                queue.parallel_for(sycl::range<1>(dim_size), [=](sycl::id<1> i) {
                    slice_buf[i] = in_ptr[base + i[0] * inner_size];
                    idx_buf[i] = static_cast<int64_t>(i[0]);
                }).wait();

                ::oneapi::dpl::sort_by_key(policy, slice_buf, slice_buf + dim_size, idx_buf);
                queue.memcpy(out_val_ptr + s, slice_buf + (k - 1), sizeof(T)).wait();
                queue.memcpy(out_idx_ptr + s, idx_buf + (k - 1), sizeof(int64_t)).wait();
            }

            sycl::free(slice_buf, queue);
            sycl::free(idx_buf, queue);
#else
            // Fallback: device-side bitonic sort when oneDPL unavailable
            T* slice_buf = sycl::malloc_device<T>(dim_size, queue);
            int64_t* idx_buf = sycl::malloc_device<int64_t>(dim_size, queue);
            const T* in_ptr = get_data_ptr<const T>(input_cont);
            T* out_val_ptr = get_data_ptr<T>(values);
            int64_t* out_idx_ptr = get_data_ptr<int64_t>(indices_out);

            for (int64_t s = 0; s < total_slices; ++s) {
                int64_t outer = s / inner_size;
                int64_t inner = s % inner_size;
                int64_t base = outer * dim_size * inner_size + inner;

                // Gather slice to contiguous buffer on device
                queue.parallel_for(sycl::range<1>(dim_size), [=](sycl::id<1> i) {
                    slice_buf[i] = in_ptr[base + i[0] * inner_size];
                    idx_buf[i] = static_cast<int64_t>(i[0]);
                }).wait();

                sycl_bitonic_sort_by_key(slice_buf, idx_buf, dim_size, queue);
                queue.memcpy(out_val_ptr + s, slice_buf + (k - 1), sizeof(T)).wait();
                queue.memcpy(out_idx_ptr + s, idx_buf + (k - 1), sizeof(int64_t)).wait();
            }

            sycl::free(slice_buf, queue);
            sycl::free(idx_buf, queue);
#endif
        }
    };

    switch (dtype) {
        case DType::Float32: launch.template operator()<float, KthvalueIdxInitKernelFloat32>(); break;
        case DType::Float64: launch.template operator()<double, KthvalueIdxInitKernelFloat64>(); break;
        case DType::Int32:   launch.template operator()<int32_t, KthvalueIdxInitKernelInt32>(); break;
        case DType::Int64:   launch.template operator()<int64_t, KthvalueIdxInitKernelInt64>(); break;
        default: throw std::runtime_error("kthvalue OneAPI: unsupported dtype");
    }
    return {values, indices_out};
}

// ============================================================================
// Quantile/Nanquantile/Nanmedian — quantile with optional NaN handling
// ============================================================================

static auto quantile_impl(const Tensor& input, double q, int64_t dim, bool keepdim,
                           bool ignore_nan, sycl::queue& queue) -> Tensor
{
    Tensor input_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);
    const auto& shape = input_cont.shape();
    const int64_t ndim = input_cont.ndim();
    // Normalize negative dim (dim=-1 ⇒ last axis). The kernel-registry default
    // for these reductions is -1, meaning "no dim was specified" from the
    // caller; for 1D inputs this is the only reduction dim, for higher-rank
    // inputs it maps to the last axis.
    if (dim < 0) dim += ndim;
    const int64_t dim_size = shape[dim];
    const auto dtype = input_cont.dtype();
    const auto device = input_cont.device();

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];
    int64_t total_slices = outer_size * inner_size;

    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(shape[i]);
        }
    }
    if (out_shape.empty()) out_shape.push_back(1);

    Tensor output(out_shape, dtype, device);

    auto launch = [&]<typename T>() {
#ifdef TENZOR_HAS_ONEDPL
        auto policy = ::oneapi::dpl::execution::make_device_policy(queue);
        T* slice_buf = sycl::malloc_device<T>(dim_size, queue);
        const T* in_ptr = get_data_ptr<const T>(input_cont);
        T* out_ptr = get_data_ptr<T>(output);

        for (int64_t s = 0; s < total_slices; ++s) {
            int64_t outer = s / inner_size;
            int64_t inner = s % inner_size;
            int64_t base = outer * dim_size * inner_size + inner;

            // Gather slice to contiguous device buffer
            queue.parallel_for(sycl::range<1>(dim_size), [=](sycl::id<1> i) {
                slice_buf[i] = in_ptr[base + i[0] * inner_size];
            }).wait();

            int64_t valid_count = dim_size;
            if (ignore_nan) {
                // Partition non-NaN values to the front on device
                auto end_it = ::oneapi::dpl::partition(policy, slice_buf, slice_buf + dim_size,
                    [](T val) { return !sycl::isnan(static_cast<float>(val)); });
                valid_count = end_it - slice_buf;
            }

            if (valid_count == 0) {
                T nan_val = static_cast<T>(NAN);
                queue.memcpy(out_ptr + s, &nan_val, sizeof(T)).wait();
                continue;
            }

            // Sort valid elements on device
            ::oneapi::dpl::sort(policy, slice_buf, slice_buf + valid_count);

            // Compute interpolated quantile value on device
            double pos = q * (static_cast<double>(valid_count) - 1.0);
            int64_t lo = static_cast<int64_t>(pos);
            int64_t hi = lo + 1;
            if (hi >= valid_count) hi = valid_count - 1;
            double frac = pos - lo;

            queue.single_task([=]() {
                out_ptr[s] = static_cast<T>(static_cast<double>(slice_buf[lo]) * (1.0 - frac) +
                                            static_cast<double>(slice_buf[hi]) * frac);
            }).wait();
        }

        sycl::free(slice_buf, queue);
#else
        // Fallback: device-side bitonic sort + interpolate when oneDPL unavailable
        T* slice_buf = sycl::malloc_device<T>(dim_size, queue);
        const T* in_ptr = get_data_ptr<const T>(input_cont);
        T* out_ptr = get_data_ptr<T>(output);

        for (int64_t s = 0; s < total_slices; ++s) {
            int64_t outer = s / inner_size;
            int64_t inner = s % inner_size;
            int64_t base = outer * dim_size * inner_size + inner;

            // Gather slice to contiguous device buffer
            queue.parallel_for(sycl::range<1>(dim_size), [=](sycl::id<1> i) {
                slice_buf[i] = in_ptr[base + i[0] * inner_size];
            }).wait();

            int64_t valid_count = dim_size;
            if (ignore_nan) {
                // Partition non-NaN values to front: move NaNs to end on device
                // Use a simple compact pass with device-side count
                int64_t* d_count = sycl::malloc_device<int64_t>(1, queue);
                T* compact_buf = sycl::malloc_device<T>(dim_size, queue);
                queue.memset(d_count, 0, sizeof(int64_t)).wait();

                // Count and compact non-NaN values
                queue.single_task([=]() {
                    int64_t c = 0;
                    for (int64_t i = 0; i < dim_size; ++i) {
                        if (!sycl::isnan(static_cast<float>(slice_buf[i]))) {
                            compact_buf[c++] = slice_buf[i];
                        }
                    }
                    d_count[0] = c;
                }).wait();

                queue.memcpy(&valid_count, d_count, sizeof(int64_t)).wait();

                // Copy compacted data back to slice_buf
                queue.memcpy(slice_buf, compact_buf, valid_count * sizeof(T)).wait();

                sycl::free(d_count, queue);
                sycl::free(compact_buf, queue);
            }

            if (valid_count == 0) {
                T nan_val = static_cast<T>(NAN);
                queue.memcpy(out_ptr + s, &nan_val, sizeof(T)).wait();
                continue;
            }

            // Sort valid elements on device
            sycl_bitonic_sort(slice_buf, valid_count, queue);

            // Compute interpolated quantile value on device
            double pos = q * (static_cast<double>(valid_count) - 1.0);
            int64_t lo = static_cast<int64_t>(pos);
            int64_t hi = lo + 1;
            if (hi >= valid_count) hi = valid_count - 1;
            double frac = pos - lo;

            queue.single_task([=]() {
                out_ptr[s] = static_cast<T>(static_cast<double>(slice_buf[lo]) * (1.0 - frac) +
                                            static_cast<double>(slice_buf[hi]) * frac);
            }).wait();
        }

        sycl::free(slice_buf, queue);
#endif
    };

    switch (dtype) {
        case DType::Float32: launch.template operator()<float>(); break;
        case DType::Float64: launch.template operator()<double>(); break;
        default: throw std::runtime_error("quantile OneAPI: unsupported dtype");
    }
    return output;
}

auto quantile_kernel(const Tensor& input, double q, int64_t dim, bool keepdim,
                     sycl::queue& queue) -> Tensor
{
    return quantile_impl(input, q, dim, keepdim, false, queue);
}

auto nanquantile_kernel(const Tensor& input, double q, int64_t dim, bool keepdim,
                        sycl::queue& queue) -> Tensor
{
    return quantile_impl(input, q, dim, keepdim, true, queue);
}

auto nanmedian_kernel(const Tensor& input, int64_t dim, bool keepdim,
                      sycl::queue& queue) -> Tensor
{
    return nanquantile_kernel(input, 0.5, dim, keepdim, queue);
}

// ============================================================================
// Histc kernel — fixed-bin histogram
// ============================================================================

auto histc_kernel(const Tensor& input, int64_t bins, double min_val, double max_val,
                  sycl::queue& queue) -> Tensor
{
    Tensor input_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);
    int64_t n = input_cont.numel();
    const auto device = input_cont.device();

    // Work in float32 for the histogram accumulation
    Tensor output({bins}, DType::Float32, device);

    // Zero-initialize output on device
    queue.memset(get_data_ptr<float>(output), 0, bins * sizeof(float)).wait();

    // Get input as float32 on device
    const float* in_ptr = nullptr;
    Tensor f32_input;
    if (input_cont.dtype() == DType::Float32) {
        in_ptr = get_data_ptr<const float>(input_cont);
    } else if (input_cont.dtype() == DType::Float64) {
        f32_input = input_cont.to(DType::Float32);
        in_ptr = get_data_ptr<const float>(f32_input);
    } else {
        throw std::runtime_error("histc OneAPI: unsupported dtype");
    }

    // Auto-detect range on device
    if (min_val >= max_val) {
#ifdef TENZOR_HAS_ONEDPL
        auto policy = ::oneapi::dpl::execution::make_device_policy(queue);
        auto minmax = ::oneapi::dpl::minmax_element(policy, in_ptr, in_ptr + n);
        // Read two scalars from device (minimal transfer — just two floats for range)
        float h_min, h_max;
        queue.memcpy(&h_min, &(*minmax.first), sizeof(float)).wait();
        queue.memcpy(&h_max, &(*minmax.second), sizeof(float)).wait();
        min_val = h_min;
        max_val = h_max;
#else
        // Fallback: reduction via parallel_for
        float* d_min = sycl::malloc_device<float>(1, queue);
        float* d_max = sycl::malloc_device<float>(1, queue);
        queue.single_task([=]() { d_min[0] = in_ptr[0]; d_max[0] = in_ptr[0]; }).wait();
        queue.parallel_for(sycl::range<1>(n), sycl::reduction(d_min, sycl::minimum<float>()),
                           sycl::reduction(d_max, sycl::maximum<float>()),
                           [=](sycl::id<1> idx, auto& mn, auto& mx) {
                               mn.combine(in_ptr[idx]);
                               mx.combine(in_ptr[idx]);
                           }).wait();
        float h_min, h_max;
        queue.memcpy(&h_min, d_min, sizeof(float)).wait();
        queue.memcpy(&h_max, d_max, sizeof(float)).wait();
        sycl::free(d_min, queue);
        sycl::free(d_max, queue);
        min_val = h_min;
        max_val = h_max;
#endif
    }

    // Compute histogram on device using atomic operations
    float* out_ptr = get_data_ptr<float>(output);
    float f_min = static_cast<float>(min_val);
    float f_max = static_cast<float>(max_val);
    float bin_width = (f_max - f_min) / static_cast<float>(bins);

    queue.parallel_for<HistcBinKernelFloat32>(sycl::range<1>(n), [=](sycl::id<1> idx) {
        float val = in_ptr[idx];
        if (val >= f_min && val <= f_max) {
            int64_t bin = static_cast<int64_t>((val - f_min) / bin_width);
            if (bin >= bins) bin = bins - 1;
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space> atom(out_ptr[bin]);
            atom += 1.0f;
        }
    }).wait();

    return output;
}

// ============================================================================
// UniqueConsecutive kernel — deduplicate consecutive equal elements
// ============================================================================

auto unique_consecutive_kernel(const Tensor& input, bool return_inverse,
                                sycl::queue& queue) -> std::tuple<Tensor, Tensor, Tensor>
{
    Tensor input_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);
    int64_t n = input_cont.numel();
    const auto dtype = input_cont.dtype();
    const auto device = input_cont.device();

    if (n == 0) {
        return {Tensor({0}, dtype, device), Tensor({0}, DType::Int64, device),
                Tensor({0}, DType::Int64, device)};
    }

    auto launch = [&]<typename T, typename MaskKernel, typename GatherKernel, typename CountsKernel>() {
        const T* in_ptr = get_data_ptr<const T>(input_cont);

        // Step 1: compute adjacency mask on device (1 where value differs from predecessor)
        int32_t* d_mask = sycl::malloc_device<int32_t>(n, queue);
        queue.parallel_for<MaskKernel>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            if (idx[0] == 0) {
                d_mask[0] = 1;  // First element is always unique
            } else {
                d_mask[idx] = (in_ptr[idx] != in_ptr[idx[0] - 1]) ? 1 : 0;
            }
        }).wait();

        // Step 2: inclusive prefix sum on mask for scatter offsets
        int32_t* d_prefix = sycl::malloc_device<int32_t>(n, queue);

#ifdef TENZOR_HAS_ONEDPL
        auto policy = ::oneapi::dpl::execution::make_device_policy(queue);
        ::oneapi::dpl::inclusive_scan(policy, d_mask, d_mask + n, d_prefix);
#else
        // No oneDPL: run an inclusive scan on-device via single_task. The
        // scan itself is sequential but stays in device memory — no host
        // roundtrip on the (potentially large) mask array.
        {
            int32_t* mask_in = d_mask;
            int32_t* pref_out = d_prefix;
            int64_t n_local = n;
            queue.single_task([=]() {
                int32_t acc = 0;
                for (int64_t i = 0; i < n_local; ++i) {
                    acc += mask_in[i];
                    pref_out[i] = acc;
                }
            }).wait();
        }
#endif

        // Read total unique count from last element of prefix sum (single scalar readback)
        int32_t num_unique_h;
        queue.memcpy(&num_unique_h, d_prefix + n - 1, sizeof(int32_t)).wait();
        int64_t num_unique = num_unique_h;

        // Step 3: gather unique elements and compute inverse indices on device
        Tensor unique_out({num_unique}, dtype, device);
        Tensor inverse_out({n}, DType::Int64, device);

        T* unique_ptr = get_data_ptr<T>(unique_out);
        int64_t* inv_ptr = get_data_ptr<int64_t>(inverse_out);

        queue.parallel_for<GatherKernel>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            int32_t out_idx = d_prefix[idx] - 1;  // prefix_sum is 1-based
            if (d_mask[idx]) {
                unique_ptr[out_idx] = in_ptr[idx];
            }
            inv_ptr[idx] = static_cast<int64_t>(out_idx);
        }).wait();

        // Step 4: compute counts on device using atomics
        Tensor counts_out({num_unique}, DType::Int64, device);
        queue.memset(get_data_ptr<int64_t>(counts_out), 0, num_unique * sizeof(int64_t)).wait();

        int64_t* counts_ptr = get_data_ptr<int64_t>(counts_out);
        queue.parallel_for<CountsKernel>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            sycl::atomic_ref<int64_t, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space> atom(counts_ptr[inv_ptr[idx]]);
            atom += 1;
        }).wait();

        sycl::free(d_mask, queue);
        sycl::free(d_prefix, queue);

        return std::make_tuple(unique_out, inverse_out, counts_out);
    };

    switch (dtype) {
        case DType::Float32: return launch.template operator()<float, UniqueConsecutiveMaskFloat32, UniqueConsecutiveGatherFloat32, UniqueConsecutiveCountsFloat32>();
        case DType::Float64: return launch.template operator()<double, UniqueConsecutiveMaskFloat64, UniqueConsecutiveGatherFloat64, UniqueConsecutiveCountsFloat64>();
        case DType::Int32:   return launch.template operator()<int32_t, UniqueConsecutiveMaskInt32, UniqueConsecutiveGatherInt32, UniqueConsecutiveCountsInt32>();
        case DType::Int64:   return launch.template operator()<int64_t, UniqueConsecutiveMaskInt64, UniqueConsecutiveGatherInt64, UniqueConsecutiveCountsInt64>();
        default: throw std::runtime_error("unique_consecutive OneAPI: unsupported dtype");
    }
}

// ============================================================================
// Deg2Rad: x * (π / 180)
// ============================================================================
class Deg2RadKernelF32;
class Deg2RadKernelF64;
class Deg2RadKernelF16;
class Deg2RadKernelBF16;

auto deg2rad_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    int64_t numel = input.numel();
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor result(shape_vec, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(result);
        queue.parallel_for<Deg2RadKernelF32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = in_ptr[idx] * (3.14159265358979323846f / 180.0f);
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(result);
        queue.parallel_for<Deg2RadKernelF64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = in_ptr[idx] * (3.14159265358979323846 / 180.0);
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(result);
        queue.parallel_for<Deg2RadKernelF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(static_cast<float>(in_ptr[idx]) * (3.14159265358979323846f / 180.0f));
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(result);
        queue.parallel_for<Deg2RadKernelBF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(bf16_to_f32(in_ptr[idx]) * (3.14159265358979323846f / 180.0f));
        }).wait();
    } else {
        throw std::runtime_error("deg2rad: unsupported dtype");
    }
    return result;
}

// ============================================================================
// Rad2Deg: x * (180 / π)
// ============================================================================
class Rad2DegKernelF32;
class Rad2DegKernelF64;
class Rad2DegKernelF16;
class Rad2DegKernelBF16;

auto rad2deg_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    int64_t numel = input.numel();
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor result(shape_vec, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(result);
        queue.parallel_for<Rad2DegKernelF32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = in_ptr[idx] * (180.0f / 3.14159265358979323846f);
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(result);
        queue.parallel_for<Rad2DegKernelF64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = in_ptr[idx] * (180.0 / 3.14159265358979323846);
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(result);
        queue.parallel_for<Rad2DegKernelF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(static_cast<float>(in_ptr[idx]) * (180.0f / 3.14159265358979323846f));
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(result);
        queue.parallel_for<Rad2DegKernelBF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(bf16_to_f32(in_ptr[idx]) * (180.0f / 3.14159265358979323846f));
        }).wait();
    } else {
        throw std::runtime_error("rad2deg: unsupported dtype");
    }
    return result;
}

// ============================================================================
// Logit: log(clamp(x, eps, 1-eps) / (1 - clamp(x, eps, 1-eps)))
// ============================================================================
class LogitKernelF32;
class LogitKernelF64;
class LogitKernelF16;
class LogitKernelBF16;

auto logit_kernel(const Tensor& input, float eps, sycl::queue& queue) -> Tensor {
    int64_t numel = input.numel();
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor result(shape_vec, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(result);
        float lo = eps;
        float hi = 1.0f - eps;
        queue.parallel_for<LogitKernelF32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = in_ptr[idx];
            float c = x < lo ? lo : (x > hi ? hi : x);
            out_ptr[idx] = sycl::log(c / (1.0f - c));
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(result);
        double lo = static_cast<double>(eps);
        double hi = 1.0 - lo;
        queue.parallel_for<LogitKernelF64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double x = in_ptr[idx];
            double c = x < lo ? lo : (x > hi ? hi : x);
            out_ptr[idx] = sycl::log(c / (1.0 - c));
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(result);
        float lo = eps;
        float hi = 1.0f - eps;
        queue.parallel_for<LogitKernelF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = static_cast<float>(in_ptr[idx]);
            float c = x < lo ? lo : (x > hi ? hi : x);
            out_ptr[idx] = sycl::half(sycl::log(c / (1.0f - c)));
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(result);
        float lo = eps;
        float hi = 1.0f - eps;
        queue.parallel_for<LogitKernelBF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = bf16_to_f32(in_ptr[idx]);
            float c = x < lo ? lo : (x > hi ? hi : x);
            out_ptr[idx] = f32_to_bf16(sycl::log(c / (1.0f - c)));
        }).wait();
    } else {
        throw std::runtime_error("logit: unsupported dtype");
    }
    return result;
}

// ============================================================================
// Signbit: returns Bool tensor indicating sign bit
// ============================================================================
class SignbitKernelF32;
class SignbitKernelF64;
class SignbitKernelF16;
class SignbitKernelBF16;
class SignbitKernelI32;
class SignbitKernelI64;

auto signbit_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    int64_t numel = input.numel();
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor output(shape_vec, DType::Bool, input.device());
    uint8_t* out_ptr = get_data_ptr<uint8_t>(output);

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        queue.parallel_for<SignbitKernelF32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(sycl::signbit(in_ptr[idx]) ? 1 : 0);
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        queue.parallel_for<SignbitKernelF64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(sycl::signbit(in_ptr[idx]) ? 1 : 0);
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        queue.parallel_for<SignbitKernelF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(sycl::signbit(static_cast<float>(in_ptr[idx])) ? 1 : 0);
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        queue.parallel_for<SignbitKernelBF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(sycl::signbit(bf16_to_f32(in_ptr[idx])) ? 1 : 0);
        }).wait();
    } else if (input.dtype() == DType::Int32) {
        const int32_t* in_ptr = get_data_ptr<const int32_t>(input);
        queue.parallel_for<SignbitKernelI32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(in_ptr[idx] < 0 ? 1 : 0);
        }).wait();
    } else if (input.dtype() == DType::Int64) {
        const int64_t* in_ptr = get_data_ptr<const int64_t>(input);
        queue.parallel_for<SignbitKernelI64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>(in_ptr[idx] < 0 ? 1 : 0);
        }).wait();
    } else {
        // Unsigned types: sign bit is never set
        queue.memset(out_ptr, 0, numel * sizeof(uint8_t)).wait();
    }
    return output;
}

// ============================================================================
// IsPosInf: isinf(x) && x > 0 → Bool
// ============================================================================
class IsPosInfKernelF32;
class IsPosInfKernelF64;
class IsPosInfKernelF16;
class IsPosInfKernelBF16;

auto isposinf_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    int64_t numel = input.numel();
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor output(shape_vec, DType::Bool, input.device());
    uint8_t* out_ptr = get_data_ptr<uint8_t>(output);

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        queue.parallel_for<IsPosInfKernelF32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>((sycl::isinf(in_ptr[idx]) && in_ptr[idx] > 0.0f) ? 1 : 0);
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        queue.parallel_for<IsPosInfKernelF64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>((sycl::isinf(in_ptr[idx]) && in_ptr[idx] > 0.0) ? 1 : 0);
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        queue.parallel_for<IsPosInfKernelF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float v = static_cast<float>(in_ptr[idx]);
            out_ptr[idx] = static_cast<uint8_t>((sycl::isinf(v) && v > 0.0f) ? 1 : 0);
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        queue.parallel_for<IsPosInfKernelBF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float v = bf16_to_f32(in_ptr[idx]);
            out_ptr[idx] = static_cast<uint8_t>((sycl::isinf(v) && v > 0.0f) ? 1 : 0);
        }).wait();
    } else {
        // Integer types are never infinite
        queue.memset(out_ptr, 0, numel * sizeof(uint8_t)).wait();
    }
    return output;
}

// ============================================================================
// IsNegInf: isinf(x) && x < 0 → Bool
// ============================================================================
class IsNegInfKernelF32;
class IsNegInfKernelF64;
class IsNegInfKernelF16;
class IsNegInfKernelBF16;

auto isneginf_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    int64_t numel = input.numel();
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor output(shape_vec, DType::Bool, input.device());
    uint8_t* out_ptr = get_data_ptr<uint8_t>(output);

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        queue.parallel_for<IsNegInfKernelF32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>((sycl::isinf(in_ptr[idx]) && in_ptr[idx] < 0.0f) ? 1 : 0);
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        queue.parallel_for<IsNegInfKernelF64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<uint8_t>((sycl::isinf(in_ptr[idx]) && in_ptr[idx] < 0.0) ? 1 : 0);
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        queue.parallel_for<IsNegInfKernelF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float v = static_cast<float>(in_ptr[idx]);
            out_ptr[idx] = static_cast<uint8_t>((sycl::isinf(v) && v < 0.0f) ? 1 : 0);
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        queue.parallel_for<IsNegInfKernelBF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float v = bf16_to_f32(in_ptr[idx]);
            out_ptr[idx] = static_cast<uint8_t>((sycl::isinf(v) && v < 0.0f) ? 1 : 0);
        }).wait();
    } else {
        // Integer types are never infinite
        queue.memset(out_ptr, 0, numel * sizeof(uint8_t)).wait();
    }
    return output;
}

// ============================================================================
// FloatPower: promote to Float64, compute pow(x, y)
// ============================================================================
class FloatPowerKernelF64;

auto float_power_kernel(const Tensor& base, const Tensor& exp, sycl::queue& queue) -> Tensor {
    int64_t numel = base.numel();
    auto shape_vec = std::vector<int64_t>(base.shape().begin(), base.shape().end());
    Tensor result(shape_vec, DType::Float64, base.device());
    double* out_ptr = get_data_ptr<double>(result);

    // Allocate temporary Float64 buffers if needed
    auto cast_to_f64 = [&](const Tensor& t) -> Tensor {
        if (t.dtype() == DType::Float64) return t;
        Tensor f64_t(std::vector<int64_t>(t.shape().begin(), t.shape().end()),
                     DType::Float64, t.device());
        double* dst = get_data_ptr<double>(f64_t);
        if (t.dtype() == DType::Float32) {
            const float* src = get_data_ptr<const float>(t);
            queue.parallel_for(sycl::range<1>(t.numel()), [=](sycl::id<1> i) {
                dst[i] = static_cast<double>(src[i]);
            }).wait();
        } else if (t.dtype() == DType::Float16) {
            const sycl::half* src = get_data_ptr<const sycl::half>(t);
            queue.parallel_for(sycl::range<1>(t.numel()), [=](sycl::id<1> i) {
                dst[i] = static_cast<double>(static_cast<float>(src[i]));
            }).wait();
        } else if (t.dtype() == DType::BFloat16) {
            const uint16_t* src = get_data_ptr<const uint16_t>(t);
            queue.parallel_for(sycl::range<1>(t.numel()), [=](sycl::id<1> i) {
                dst[i] = static_cast<double>(bf16_to_f32(src[i]));
            }).wait();
        } else if (t.dtype() == DType::Int32) {
            const int32_t* src = get_data_ptr<const int32_t>(t);
            queue.parallel_for(sycl::range<1>(t.numel()), [=](sycl::id<1> i) {
                dst[i] = static_cast<double>(src[i]);
            }).wait();
        } else if (t.dtype() == DType::Int64) {
            const int64_t* src = get_data_ptr<const int64_t>(t);
            queue.parallel_for(sycl::range<1>(t.numel()), [=](sycl::id<1> i) {
                dst[i] = static_cast<double>(src[i]);
            }).wait();
        } else {
            throw std::runtime_error("float_power: unsupported input dtype");
        }
        return f64_t;
    };

    Tensor base_f64 = cast_to_f64(base);
    Tensor exp_f64 = cast_to_f64(exp);
    const double* base_ptr = get_data_ptr<const double>(base_f64);
    const double* exp_ptr = get_data_ptr<const double>(exp_f64);

    queue.parallel_for<FloatPowerKernelF64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
        out_ptr[idx] = sycl::pow(base_ptr[idx], exp_ptr[idx]);
    }).wait();

    return result;
}

// ============================================================================
// Xlog1py: x == 0 ? 0 : x * log1p(y)
// ============================================================================
class Xlog1pyKernelF32;
class Xlog1pyKernelF64;
class Xlog1pyKernelF16;
class Xlog1pyKernelBF16;

auto xlog1py_kernel(const Tensor& x, const Tensor& y, sycl::queue& queue) -> Tensor {
    if (x.dtype() != y.dtype()) {
        throw std::invalid_argument("xlog1py: input dtypes must match");
    }
    int64_t numel = x.numel();
    auto shape_vec = std::vector<int64_t>(x.shape().begin(), x.shape().end());
    Tensor result(shape_vec, x.dtype(), x.device());

    if (x.dtype() == DType::Float32) {
        const float* x_ptr = get_data_ptr<const float>(x);
        const float* y_ptr = get_data_ptr<const float>(y);
        float* out_ptr = get_data_ptr<float>(result);
        queue.parallel_for<Xlog1pyKernelF32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float xv = x_ptr[idx];
            out_ptr[idx] = (xv == 0.0f) ? 0.0f : xv * sycl::log1p(y_ptr[idx]);
        }).wait();
    } else if (x.dtype() == DType::Float64) {
        const double* x_ptr = get_data_ptr<const double>(x);
        const double* y_ptr = get_data_ptr<const double>(y);
        double* out_ptr = get_data_ptr<double>(result);
        queue.parallel_for<Xlog1pyKernelF64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double xv = x_ptr[idx];
            out_ptr[idx] = (xv == 0.0) ? 0.0 : xv * sycl::log1p(y_ptr[idx]);
        }).wait();
    } else if (x.dtype() == DType::Float16) {
        const sycl::half* x_ptr = get_data_ptr<const sycl::half>(x);
        const sycl::half* y_ptr = get_data_ptr<const sycl::half>(y);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(result);
        queue.parallel_for<Xlog1pyKernelF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float xv = static_cast<float>(x_ptr[idx]);
            out_ptr[idx] = sycl::half((xv == 0.0f) ? 0.0f : xv * sycl::log1p(static_cast<float>(y_ptr[idx])));
        }).wait();
    } else if (x.dtype() == DType::BFloat16) {
        const uint16_t* x_ptr = get_data_ptr<const uint16_t>(x);
        const uint16_t* y_ptr = get_data_ptr<const uint16_t>(y);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(result);
        queue.parallel_for<Xlog1pyKernelBF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float xv = bf16_to_f32(x_ptr[idx]);
            out_ptr[idx] = f32_to_bf16((xv == 0.0f) ? 0.0f : xv * sycl::log1p(bf16_to_f32(y_ptr[idx])));
        }).wait();
    } else {
        throw std::runtime_error("xlog1py: unsupported dtype");
    }
    return result;
}

// ============================================================================
// Ldexp: x * 2^n
// ============================================================================
class LdexpKernelF32;
class LdexpKernelF64;
class LdexpKernelF16;
class LdexpKernelBF16;

auto ldexp_kernel(const Tensor& x, const Tensor& n, sycl::queue& queue) -> Tensor {
    int64_t numel = x.numel();
    auto shape_vec = std::vector<int64_t>(x.shape().begin(), x.shape().end());
    Tensor result(shape_vec, x.dtype(), x.device());

    // n is expected to be Int32 (exponent tensor)
    const int32_t* n_ptr = get_data_ptr<const int32_t>(n);

    if (x.dtype() == DType::Float32) {
        const float* x_ptr = get_data_ptr<const float>(x);
        float* out_ptr = get_data_ptr<float>(result);
        queue.parallel_for<LdexpKernelF32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::ldexp(x_ptr[idx], n_ptr[idx]);
        }).wait();
    } else if (x.dtype() == DType::Float64) {
        const double* x_ptr = get_data_ptr<const double>(x);
        double* out_ptr = get_data_ptr<double>(result);
        queue.parallel_for<LdexpKernelF64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::ldexp(x_ptr[idx], n_ptr[idx]);
        }).wait();
    } else if (x.dtype() == DType::Float16) {
        const sycl::half* x_ptr = get_data_ptr<const sycl::half>(x);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(result);
        queue.parallel_for<LdexpKernelF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::ldexp(static_cast<float>(x_ptr[idx]), n_ptr[idx]));
        }).wait();
    } else if (x.dtype() == DType::BFloat16) {
        const uint16_t* x_ptr = get_data_ptr<const uint16_t>(x);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(result);
        queue.parallel_for<LdexpKernelBF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::ldexp(bf16_to_f32(x_ptr[idx]), n_ptr[idx]));
        }).wait();
    } else {
        throw std::runtime_error("ldexp: unsupported dtype");
    }
    return result;
}

// ============================================================================
// Frexp: decompose into mantissa (same dtype) and exponent (Int32)
// ============================================================================
class FrexpKernelF32;
class FrexpKernelF64;
class FrexpKernelF16;
class FrexpKernelBF16;

auto frexp_kernel(const Tensor& input, sycl::queue& queue) -> std::vector<Tensor> {
    int64_t numel = input.numel();
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor mantissa(shape_vec, input.dtype(), input.device());
    Tensor exponent(shape_vec, DType::Int32, input.device());
    int32_t* exp_ptr = get_data_ptr<int32_t>(exponent);

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* man_ptr = get_data_ptr<float>(mantissa);
        queue.parallel_for<FrexpKernelF32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int e;
            man_ptr[idx] = sycl::frexp(in_ptr[idx], &e);
            exp_ptr[idx] = e;
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* man_ptr = get_data_ptr<double>(mantissa);
        queue.parallel_for<FrexpKernelF64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int e;
            man_ptr[idx] = sycl::frexp(in_ptr[idx], &e);
            exp_ptr[idx] = e;
        }).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* man_ptr = get_data_ptr<sycl::half>(mantissa);
        queue.parallel_for<FrexpKernelF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int e;
            float m = sycl::frexp(static_cast<float>(in_ptr[idx]), &e);
            man_ptr[idx] = sycl::half(m);
            exp_ptr[idx] = e;
        }).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* man_ptr = get_data_ptr<uint16_t>(mantissa);
        queue.parallel_for<FrexpKernelBF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int e;
            float m = sycl::frexp(bf16_to_f32(in_ptr[idx]), &e);
            man_ptr[idx] = f32_to_bf16(m);
            exp_ptr[idx] = e;
        }).wait();
    } else {
        throw std::runtime_error("frexp: unsupported dtype (requires floating point)");
    }
    return {mantissa, exponent};
}

// ============================================================================
// DiagEmbed: embed a vector on the diagonal of a matrix
// Input shape: (..., N), output shape: (..., N+|offset|, N+|offset|)
// ============================================================================
class DiagEmbedKernelF32;
class DiagEmbedKernelF64;
class DiagEmbedKernelF16;
class DiagEmbedKernelBF16;
class DiagEmbedKernelI32;
class DiagEmbedKernelI64;

auto diag_embed_kernel(const Tensor& input, int64_t offset, int64_t dim1, int64_t dim2,
                       sycl::queue& queue) -> Tensor {
    auto in_shape = input.shape();
    int64_t ndim = static_cast<int64_t>(in_shape.size());
    int64_t N = in_shape[ndim - 1]; // last dim is the vector length
    int64_t abs_offset = offset < 0 ? -offset : offset;
    int64_t mat_size = N + abs_offset;

    // Build output shape: batch dims + (mat_size, mat_size)
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim - 1; i++) out_shape.push_back(in_shape[i]);
    out_shape.push_back(mat_size);
    out_shape.push_back(mat_size);

    int64_t batch = 1;
    for (int64_t i = 0; i < ndim - 1; i++) batch *= in_shape[i];

    Tensor output(out_shape, input.dtype(), input.device());
    int64_t out_numel = output.numel();

    // Zero the output first
    queue.memset(const_cast<void*>(output.data_ptr()), 0, out_numel * dtype_size(input.dtype())).wait();

    int64_t row_off = offset >= 0 ? 0 : -offset;
    int64_t col_off = offset >= 0 ? offset : 0;

    auto launch = [&]<typename T>() {
        const T* in_ptr = get_data_ptr<const T>(input);
        T* out_ptr = get_data_ptr<T>(output);
        int64_t ms = mat_size;
        queue.parallel_for(sycl::range<1>(batch * N), [=](sycl::id<1> id) {
            int64_t idx = id[0];
            int64_t b = idx / N;
            int64_t n = idx % N;
            int64_t r = n + row_off;
            int64_t c = n + col_off;
            out_ptr[b * ms * ms + r * ms + c] = in_ptr[b * N + n];
        }).wait();
    };

    switch (input.dtype()) {
        case DType::Float32:  launch.template operator()<float>(); break;
        case DType::Float64:  launch.template operator()<double>(); break;
        case DType::Int32:    launch.template operator()<int32_t>(); break;
        case DType::Int64:    launch.template operator()<int64_t>(); break;
        case DType::Float16:  launch.template operator()<sycl::half>(); break;
        case DType::BFloat16: launch.template operator()<uint16_t>(); break;
        default: throw std::runtime_error("diag_embed: unsupported dtype");
    }
    return output;
}

// ============================================================================
// Diagflat: flatten input, then create diagonal matrix
// ============================================================================
auto diagflat_kernel(const Tensor& input, int64_t offset, sycl::queue& queue) -> Tensor {
    // Flatten input to 1D
    int64_t N = input.numel();
    int64_t abs_offset = offset < 0 ? -offset : offset;
    int64_t mat_size = N + abs_offset;

    Tensor output({mat_size, mat_size}, input.dtype(), input.device());
    int64_t out_numel = output.numel();

    // Zero the output
    queue.memset(const_cast<void*>(output.data_ptr()), 0, out_numel * dtype_size(input.dtype())).wait();

    int64_t row_off = offset >= 0 ? 0 : -offset;
    int64_t col_off = offset >= 0 ? offset : 0;

    auto launch = [&]<typename T>() {
        const T* in_ptr = get_data_ptr<const T>(input);
        T* out_ptr = get_data_ptr<T>(output);
        int64_t ms = mat_size;
        queue.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
            int64_t n = id[0];
            int64_t r = n + row_off;
            int64_t c = n + col_off;
            out_ptr[r * ms + c] = in_ptr[n];
        }).wait();
    };

    switch (input.dtype()) {
        case DType::Float32:  launch.template operator()<float>(); break;
        case DType::Float64:  launch.template operator()<double>(); break;
        case DType::Int32:    launch.template operator()<int32_t>(); break;
        case DType::Int64:    launch.template operator()<int64_t>(); break;
        case DType::Float16:  launch.template operator()<sycl::half>(); break;
        case DType::BFloat16: launch.template operator()<uint16_t>(); break;
        default: throw std::runtime_error("diagflat: unsupported dtype");
    }
    return output;
}

// ============================================================================
// IsReal: dtype check — all non-complex types are real
// ============================================================================
auto isreal_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    // All dtypes in Tenzor are real (no complex support), so return all-true
    int64_t numel = input.numel();
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor output(shape_vec, DType::Bool, input.device());
    uint8_t* out_ptr = get_data_ptr<uint8_t>(output);
    queue.memset(out_ptr, 1, numel * sizeof(uint8_t)).wait();
    return output;
}

// ============================================================================
// SegmentReduce — reduce over segments defined by offsets (SYCL)
// ============================================================================

template<typename T> struct SegmentReduceKernelTag {};

template<typename T>
static auto segment_reduce_sycl_impl(const Tensor& data, const Tensor& offsets,
                                      int64_t num_segments, int64_t outer_size,
                                      int64_t axis_size, int64_t inner_size,
                                      int mode, sycl::queue& queue) -> Tensor {
    int64_t out_numel = outer_size * num_segments * inner_size;
    Tensor output({out_numel}, data.dtype(), data.device());

    const T* data_ptr = get_data_ptr<const T>(data);
    const int64_t* offsets_ptr = get_data_ptr<const int64_t>(offsets);
    T* out_ptr = get_data_ptr<T>(output);

    int64_t total_work = outer_size * num_segments * inner_size;

    queue.submit([&](sycl::handler& h) {
        h.parallel_for<SegmentReduceKernelTag<T>>(
            sycl::range<1>(total_work), [=](sycl::id<1> idx) {
                int64_t id = idx[0];
                int64_t inner = id % inner_size;
                int64_t seg = (id / inner_size) % num_segments;
                int64_t outer = id / (inner_size * num_segments);

                int64_t seg_start = offsets_ptr[seg];
                int64_t seg_end = offsets_ptr[seg + 1];
                int64_t seg_len = seg_end - seg_start;

                T identity;
                if (mode == 0 || mode == 1) identity = T(0);
                else if (mode == 4) identity = T(1);
                else if (mode == 2) identity = T(-1e38);
                else identity = T(1e38);

                T acc = identity;
                for (int64_t d = seg_start; d < seg_end; ++d) {
                    int64_t in_idx = (outer * axis_size + d) * inner_size + inner;
                    T val = data_ptr[in_idx];
                    if (mode == 0 || mode == 1) acc += val;
                    else if (mode == 4) acc *= val;
                    else if (mode == 2) acc = acc > val ? acc : val;
                    else acc = acc < val ? acc : val;
                }

                if (mode == 1 && seg_len > 0) {
                    acc /= static_cast<T>(seg_len);
                }
                if (seg_len == 0) {
                    acc = (mode == 0 || mode == 1) ? T(0) : identity;
                }

                out_ptr[(outer * num_segments + seg) * inner_size + inner] = acc;
            });
    }).wait();

    return output;
}

auto segment_reduce_kernel(const Tensor& data, const Tensor& offsets,
                           const std::string& reduce, int64_t axis,
                           sycl::queue& queue) -> Tensor {
    Tensor cont = data.is_contiguous() ? data : oneapi::contiguous_kernel(data, queue);
    Tensor offs = offsets.is_contiguous() ? offsets : oneapi::contiguous_kernel(offsets, queue);

    int64_t ndim = cont.ndim();
    if (axis < 0) axis += ndim;

    const auto& shape = cont.shape();
    int64_t axis_size = shape[axis];
    int64_t num_segments = offs.numel() - 1;

    int64_t outer_size = 1;
    for (int64_t i = 0; i < axis; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = axis + 1; i < ndim; ++i) inner_size *= shape[i];

    int mode = 0;
    if (reduce == "sum") mode = 0;
    else if (reduce == "mean") mode = 1;
    else if (reduce == "max") mode = 2;
    else if (reduce == "min") mode = 3;
    else if (reduce == "prod") mode = 4;

    auto dtype = cont.dtype();
    Tensor result;
    switch (dtype) {
        case DType::Float32:
            result = segment_reduce_sycl_impl<float>(cont, offs, num_segments,
                                                      outer_size, axis_size, inner_size, mode, queue);
            break;
        case DType::Float64:
            result = segment_reduce_sycl_impl<double>(cont, offs, num_segments,
                                                       outer_size, axis_size, inner_size, mode, queue);
            break;
        case DType::Int32:
            result = segment_reduce_sycl_impl<int32_t>(cont, offs, num_segments,
                                                        outer_size, axis_size, inner_size, mode, queue);
            break;
        case DType::Int64:
            result = segment_reduce_sycl_impl<int64_t>(cont, offs, num_segments,
                                                        outer_size, axis_size, inner_size, mode, queue);
            break;
        default: {
            DType orig = dtype;
            Tensor cont_f32 = cont.to(DType::Float32);
            auto res_f32 = segment_reduce_sycl_impl<float>(cont_f32, offs, num_segments,
                                                             outer_size, axis_size, inner_size, mode, queue);
            result = res_f32.to(orig);
        }
    }

    // Reshape to correct output shape
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        out_shape.push_back(i == axis ? num_segments : shape[i]);
    }
    return result.reshape(out_shape);
}

} // namespace oneapi
} // namespace tenzor
