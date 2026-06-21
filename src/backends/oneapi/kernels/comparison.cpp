#include "tenzor/core/tensor.hpp"
#include "oneapi_kernel_utils.hpp"
#include "tenzor/core/shape.hpp"
#include <sycl/sycl.hpp>
#include <stdexcept>

namespace tenzor {
namespace oneapi {

// SYCL Kernel name classes (using functors for SYCL 2025.2 compatibility)
struct EqKernelFloat32 {};
struct EqKernelFloat64 {};
struct EqKernelInt32 {};
struct EqKernelInt64 {};
struct EqKernelFloat16 {};
struct EqKernelBool {};
struct NeKernelFloat32 {};
struct NeKernelFloat64 {};
struct NeKernelInt32 {};
struct NeKernelInt64 {};
struct NeKernelFloat16 {};
struct NeKernelBool {};
struct LtKernelFloat32 {};
struct LtKernelFloat64 {};
struct LtKernelInt32 {};
struct LtKernelInt64 {};
struct LtKernelFloat16 {};
struct LtKernelBool {};
struct LeKernelFloat32 {};
struct LeKernelFloat64 {};
struct LeKernelInt32 {};
struct LeKernelInt64 {};
struct LeKernelFloat16 {};
struct LeKernelBool {};
struct GtKernelFloat32 {};
struct GtKernelFloat64 {};
struct GtKernelInt32 {};
struct GtKernelInt64 {};
struct GtKernelFloat16 {};
struct GtKernelBool {};
struct GeKernelFloat32 {};
struct GeKernelFloat64 {};
struct GeKernelInt32 {};
struct GeKernelInt64 {};
struct GeKernelFloat16 {};
struct GeKernelBool {};
struct EqKernelBFloat16 {};
struct NeKernelBFloat16 {};
struct LtKernelBFloat16 {};
struct LeKernelBFloat16 {};
struct GtKernelBFloat16 {};
struct GeKernelBFloat16 {};

// Broadcast comparison kernel name structs
struct BroadcastEqFloat32 {}; struct BroadcastEqFloat64 {}; struct BroadcastEqInt32 {};
struct BroadcastEqInt64 {};   struct BroadcastEqFloat16 {}; struct BroadcastEqBFloat16 {};
struct BroadcastNeFloat32 {}; struct BroadcastNeFloat64 {}; struct BroadcastNeInt32 {};
struct BroadcastNeInt64 {};   struct BroadcastNeFloat16 {}; struct BroadcastNeBFloat16 {};
struct BroadcastLtFloat32 {}; struct BroadcastLtFloat64 {}; struct BroadcastLtInt32 {};
struct BroadcastLtInt64 {};   struct BroadcastLtFloat16 {}; struct BroadcastLtBFloat16 {};
struct BroadcastLeFloat32 {}; struct BroadcastLeFloat64 {}; struct BroadcastLeInt32 {};
struct BroadcastLeInt64 {};   struct BroadcastLeFloat16 {}; struct BroadcastLeBFloat16 {};
struct BroadcastGtFloat32 {}; struct BroadcastGtFloat64 {}; struct BroadcastGtInt32 {};
struct BroadcastGtInt64 {};   struct BroadcastGtFloat16 {}; struct BroadcastGtBFloat16 {};
struct BroadcastGeFloat32 {}; struct BroadcastGeFloat64 {}; struct BroadcastGeInt32 {};
struct BroadcastGeInt64 {};   struct BroadcastGeFloat16 {}; struct BroadcastGeBFloat16 {};
struct BroadcastEqBool {};    struct BroadcastNeBool {};
struct BroadcastLtBool {};    struct BroadcastLeBool {};
struct BroadcastGtBool {};    struct BroadcastGeBool {};


// Helper to check if shapes match
inline auto shapes_match(std::span<const int64_t> a, std::span<const int64_t> b) -> bool {
    return std::equal(a.begin(), a.end(), b.begin(), b.end());
}


// ============================================================================
// Broadcasting Support for Comparisons
// ============================================================================

// Match math.cpp's binary-op broadcast cap (16) so a 9–16D broadcasting
// comparison succeeds where the equivalent add/mul does, instead of throwing.
constexpr int MAX_BROADCAST_DIMS = 16;

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

    if (out_shape.size() > MAX_BROADCAST_DIMS) {
        throw std::runtime_error("OneAPI comparison: max 16 broadcast dimensions supported");
    }

    BroadcastInfo info{};
    info.ndim = static_cast<int>(out_shape.size());
    info.out_numel = 1;
    for (auto d : out_shape) info.out_numel *= d;

    // Compute row-major strides for output
    int64_t stride = 1;
    for (int i = info.ndim - 1; i >= 0; --i) {
        info.out_strides[i] = stride;
        stride *= out_shape[i];
    }

    // Input strides: 0 for broadcast dims
    auto compute_strides = [&](std::span<const int64_t> shape, int64_t* strides) {
        int offset = info.ndim - static_cast<int>(shape.size());
        int64_t s = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            strides[i + offset] = (shape[i] == 1) ? 0 : s;
            s *= shape[i];
        }
        for (int i = 0; i < offset; ++i) strides[i] = 0;
    };
    compute_strides(a_shape, info.a_strides);
    compute_strides(b_shape, info.b_strides);
    return info;
}

// Generic broadcast comparison kernel
template<typename T, typename KernelName, typename Op>
static void sycl_broadcast_compare(
    const Tensor& a, const Tensor& b, Tensor& output,
    const BroadcastInfo& info, sycl::queue& queue, Op op) {

    const T* a_ptr = get_data_ptr<const T>(a);
    const T* b_ptr = get_data_ptr<const T>(b);
    bool* out_ptr = get_data_ptr<bool>(output);

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
        out_ptr[gid] = op(a_ptr[a_idx], b_ptr[b_idx]);
    });
}

// Dispatch broadcast comparison across dtypes
// Returns true if handled on GPU, false for unsupported dtype
template<typename F32K, typename F64K, typename I32K, typename I64K,
         typename F16K, typename BF16K, typename BoolK, typename Op>
static auto dispatch_broadcast_compare(
    const Tensor& a, const Tensor& b, const BroadcastInfo& info,
    const std::vector<int64_t>& out_shape, sycl::queue& queue, Op op) -> Tensor {

    Tensor output(out_shape, DType::Bool, a.device());

    if (a.dtype() == DType::Float32) {
        sycl_broadcast_compare<float, F32K>(a, b, output, info, queue,
            [op](float x, float y) { return op(x, y); });
    } else if (a.dtype() == DType::Float64) {
        sycl_broadcast_compare<double, F64K>(a, b, output, info, queue,
            [op](double x, double y) { return op(x, y); });
    } else if (a.dtype() == DType::Int32) {
        sycl_broadcast_compare<int32_t, I32K>(a, b, output, info, queue,
            [op](int32_t x, int32_t y) { return op(x, y); });
    } else if (a.dtype() == DType::Int64) {
        sycl_broadcast_compare<int64_t, I64K>(a, b, output, info, queue,
            [op](int64_t x, int64_t y) { return op(x, y); });
    } else if (a.dtype() == DType::Float16) {
        sycl_broadcast_compare<sycl::half, F16K>(a, b, output, info, queue,
            [op](sycl::half x, sycl::half y) { return op(x, y); });
    } else if (a.dtype() == DType::BFloat16) {
        sycl_broadcast_compare<uint16_t, BF16K>(a, b, output, info, queue,
            [op](uint16_t x, uint16_t y) { return op(bf16_to_f32(x), bf16_to_f32(y)); });
    } else if (a.dtype() == DType::Bool) {
        sycl_broadcast_compare<bool, BoolK>(a, b, output, info, queue,
            [op](bool x, bool y) { return op(x, y); });
    } else {
        throw std::runtime_error("Unsupported dtype for broadcast comparison");
    }
    // Drain before the host reads the USM-shared Bool output.
    queue.wait_and_throw();
    return output;
}

// ============================================================================
// Element-wise Equal (==)
// ============================================================================

auto eq_kernel(const Tensor& a_in, const Tensor& b_in, sycl::queue& queue) -> Tensor {
    // Both the same-shape path (linear a_ptr[idx]) and the broadcast path
    // (compute_broadcast_info derives strides from shape, assuming contiguous
    // row-major physical layout) require contiguous operands; a transpose/slice
    // view would otherwise read the wrong physical elements.
    const Tensor a = a_in.is_contiguous() ? a_in : a_in.contiguous();
    const Tensor b = b_in.is_contiguous() ? b_in : b_in.contiguous();
    // Validate inputs
    auto a_shape = a.shape();
    auto b_shape = b.shape();

    // Check if shapes match exactly
    bool same_shape = shapes_match(a_shape, b_shape);

    // GPU-side broadcasting for mismatched shapes
    if (!same_shape) {
        auto out_shape = broadcast_shapes(a_shape, b_shape);
        auto info = compute_broadcast_info(a_shape, b_shape, out_shape);
        return dispatch_broadcast_compare<
            BroadcastEqFloat32, BroadcastEqFloat64, BroadcastEqInt32, BroadcastEqInt64,
            BroadcastEqFloat16, BroadcastEqBFloat16, BroadcastEqBool>(
            a, b, info, out_shape, queue, [](auto x, auto y) { return x == y; });
    }

    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument("Tensor dtypes must match for comparison");
    }

    // Create output tensor with Bool dtype
    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  DType::Bool, a.device());

    const int64_t numel = a.numel();

    // Dispatch based on dtype
    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<EqKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] == b_ptr[idx]);
        });
    }
    else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<EqKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] == b_ptr[idx]);
        });
    }
    else if (a.dtype() == DType::Int32) {
        const int32_t* a_ptr = get_data_ptr<const int32_t>(a);
        const int32_t* b_ptr = get_data_ptr<const int32_t>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<EqKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] == b_ptr[idx]);
        });
    }
    else if (a.dtype() == DType::Int64) {
        const int64_t* a_ptr = get_data_ptr<const int64_t>(a);
        const int64_t* b_ptr = get_data_ptr<const int64_t>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<EqKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] == b_ptr[idx]);
        });
    }
    else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<EqKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] == b_ptr[idx]);
        });
    }
    else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<EqKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (bf16_to_f32(a_ptr[idx]) == bf16_to_f32(b_ptr[idx]));
        });
    }
    else if (a.dtype() == DType::Bool) {
        const bool* a_ptr = get_data_ptr<const bool>(a);
        const bool* b_ptr = get_data_ptr<const bool>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<EqKernelBool>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] == b_ptr[idx]);
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for eq comparison");
    }

    queue.wait_and_throw();
    return output;
}

// ============================================================================
// Element-wise Not Equal (!=)
// ============================================================================

auto ne_kernel(const Tensor& a_in, const Tensor& b_in, sycl::queue& queue) -> Tensor {
    const Tensor a = a_in.is_contiguous() ? a_in : a_in.contiguous();
    const Tensor b = b_in.is_contiguous() ? b_in : b_in.contiguous();
    auto a_shape = a.shape();
    auto b_shape = b.shape();

    bool same_shape = shapes_match(a_shape, b_shape);

    if (!same_shape) {
        auto out_shape = broadcast_shapes(a_shape, b_shape);
        auto info = compute_broadcast_info(a_shape, b_shape, out_shape);
        return dispatch_broadcast_compare<
            BroadcastNeFloat32, BroadcastNeFloat64, BroadcastNeInt32, BroadcastNeInt64,
            BroadcastNeFloat16, BroadcastNeBFloat16, BroadcastNeBool>(
            a, b, info, out_shape, queue, [](auto x, auto y) { return x != y; });
    }

    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument("Tensor dtypes must match for comparison");
    }

    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  DType::Bool, a.device());

    const int64_t numel = a.numel();

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<NeKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] != b_ptr[idx]);
        });
    }
    else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<NeKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] != b_ptr[idx]);
        });
    }
    else if (a.dtype() == DType::Int32) {
        const int32_t* a_ptr = get_data_ptr<const int32_t>(a);
        const int32_t* b_ptr = get_data_ptr<const int32_t>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<NeKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] != b_ptr[idx]);
        });
    }
    else if (a.dtype() == DType::Int64) {
        const int64_t* a_ptr = get_data_ptr<const int64_t>(a);
        const int64_t* b_ptr = get_data_ptr<const int64_t>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<NeKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] != b_ptr[idx]);
        });
    }
    else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<NeKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] != b_ptr[idx]);
        });
    }
    else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<NeKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (bf16_to_f32(a_ptr[idx]) != bf16_to_f32(b_ptr[idx]));
        });
    }
    else if (a.dtype() == DType::Bool) {
        const bool* a_ptr = get_data_ptr<const bool>(a);
        const bool* b_ptr = get_data_ptr<const bool>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<NeKernelBool>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] != b_ptr[idx]);
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for ne comparison");
    }

    queue.wait_and_throw();
    return output;
}

// ============================================================================
// Element-wise Less Than (<)
// ============================================================================

auto lt_kernel(const Tensor& a_in, const Tensor& b_in, sycl::queue& queue) -> Tensor {
    const Tensor a = a_in.is_contiguous() ? a_in : a_in.contiguous();
    const Tensor b = b_in.is_contiguous() ? b_in : b_in.contiguous();
    auto a_shape = a.shape();
    auto b_shape = b.shape();

    bool same_shape = shapes_match(a_shape, b_shape);

    if (!same_shape) {
        auto out_shape = broadcast_shapes(a_shape, b_shape);
        auto info = compute_broadcast_info(a_shape, b_shape, out_shape);
        return dispatch_broadcast_compare<
            BroadcastLtFloat32, BroadcastLtFloat64, BroadcastLtInt32, BroadcastLtInt64,
            BroadcastLtFloat16, BroadcastLtBFloat16, BroadcastLtBool>(
            a, b, info, out_shape, queue, [](auto x, auto y) { return x < y; });
    }

    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument("Tensor dtypes must match for comparison");
    }

    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  DType::Bool, a.device());

    const int64_t numel = a.numel();

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<LtKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] < b_ptr[idx]);
        });
    }
    else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<LtKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] < b_ptr[idx]);
        });
    }
    else if (a.dtype() == DType::Int32) {
        const int32_t* a_ptr = get_data_ptr<const int32_t>(a);
        const int32_t* b_ptr = get_data_ptr<const int32_t>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<LtKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] < b_ptr[idx]);
        });
    }
    else if (a.dtype() == DType::Int64) {
        const int64_t* a_ptr = get_data_ptr<const int64_t>(a);
        const int64_t* b_ptr = get_data_ptr<const int64_t>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<LtKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] < b_ptr[idx]);
        });
    }
    else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<LtKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] < b_ptr[idx]);
        });
    }
    else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<LtKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (bf16_to_f32(a_ptr[idx]) < bf16_to_f32(b_ptr[idx]));
        });
    }
    else if (a.dtype() == DType::Bool) {
        const bool* a_ptr = get_data_ptr<const bool>(a);
        const bool* b_ptr = get_data_ptr<const bool>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        // For bools: false < true (0 < 1)
        queue.parallel_for<LtKernelBool>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (!a_ptr[idx] && b_ptr[idx]);
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for lt comparison");
    }

    queue.wait_and_throw();
    return output;
}

// ============================================================================
// Element-wise Less Than or Equal (<=)
// ============================================================================

auto le_kernel(const Tensor& a_in, const Tensor& b_in, sycl::queue& queue) -> Tensor {
    const Tensor a = a_in.is_contiguous() ? a_in : a_in.contiguous();
    const Tensor b = b_in.is_contiguous() ? b_in : b_in.contiguous();
    auto a_shape = a.shape();
    auto b_shape = b.shape();

    bool same_shape = shapes_match(a_shape, b_shape);

    if (!same_shape) {
        auto out_shape = broadcast_shapes(a_shape, b_shape);
        auto info = compute_broadcast_info(a_shape, b_shape, out_shape);
        return dispatch_broadcast_compare<
            BroadcastLeFloat32, BroadcastLeFloat64, BroadcastLeInt32, BroadcastLeInt64,
            BroadcastLeFloat16, BroadcastLeBFloat16, BroadcastLeBool>(
            a, b, info, out_shape, queue, [](auto x, auto y) { return x <= y; });
    }

    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument("Tensor dtypes must match for comparison");
    }

    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  DType::Bool, a.device());

    const int64_t numel = a.numel();

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<LeKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] <= b_ptr[idx]);
        });
    }
    else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<LeKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] <= b_ptr[idx]);
        });
    }
    else if (a.dtype() == DType::Int32) {
        const int32_t* a_ptr = get_data_ptr<const int32_t>(a);
        const int32_t* b_ptr = get_data_ptr<const int32_t>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<LeKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] <= b_ptr[idx]);
        });
    }
    else if (a.dtype() == DType::Int64) {
        const int64_t* a_ptr = get_data_ptr<const int64_t>(a);
        const int64_t* b_ptr = get_data_ptr<const int64_t>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<LeKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] <= b_ptr[idx]);
        });
    }
    else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<LeKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] <= b_ptr[idx]);
        });
    }
    else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<LeKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (bf16_to_f32(a_ptr[idx]) <= bf16_to_f32(b_ptr[idx]));
        });
    }
    else if (a.dtype() == DType::Bool) {
        const bool* a_ptr = get_data_ptr<const bool>(a);
        const bool* b_ptr = get_data_ptr<const bool>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        // For bools: a <= b means !a || b (false <= anything, or true <= true)
        queue.parallel_for<LeKernelBool>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (!a_ptr[idx] || b_ptr[idx]);
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for le comparison");
    }

    queue.wait_and_throw();
    return output;
}

// ============================================================================
// Element-wise Greater Than (>)
// ============================================================================

auto gt_kernel(const Tensor& a_in, const Tensor& b_in, sycl::queue& queue) -> Tensor {
    const Tensor a = a_in.is_contiguous() ? a_in : a_in.contiguous();
    const Tensor b = b_in.is_contiguous() ? b_in : b_in.contiguous();
    auto a_shape = a.shape();
    auto b_shape = b.shape();

    bool same_shape = shapes_match(a_shape, b_shape);

    if (!same_shape) {
        auto out_shape = broadcast_shapes(a_shape, b_shape);
        auto info = compute_broadcast_info(a_shape, b_shape, out_shape);
        return dispatch_broadcast_compare<
            BroadcastGtFloat32, BroadcastGtFloat64, BroadcastGtInt32, BroadcastGtInt64,
            BroadcastGtFloat16, BroadcastGtBFloat16, BroadcastGtBool>(
            a, b, info, out_shape, queue, [](auto x, auto y) { return x > y; });
    }

    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument("Tensor dtypes must match for comparison");
    }

    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  DType::Bool, a.device());

    const int64_t numel = a.numel();

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<GtKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] > b_ptr[idx]);
        });
    }
    else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<GtKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] > b_ptr[idx]);
        });
    }
    else if (a.dtype() == DType::Int32) {
        const int32_t* a_ptr = get_data_ptr<const int32_t>(a);
        const int32_t* b_ptr = get_data_ptr<const int32_t>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<GtKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] > b_ptr[idx]);
        });
    }
    else if (a.dtype() == DType::Int64) {
        const int64_t* a_ptr = get_data_ptr<const int64_t>(a);
        const int64_t* b_ptr = get_data_ptr<const int64_t>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<GtKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] > b_ptr[idx]);
        });
    }
    else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<GtKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] > b_ptr[idx]);
        });
    }
    else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<GtKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (bf16_to_f32(a_ptr[idx]) > bf16_to_f32(b_ptr[idx]));
        });
    }
    else if (a.dtype() == DType::Bool) {
        const bool* a_ptr = get_data_ptr<const bool>(a);
        const bool* b_ptr = get_data_ptr<const bool>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        // For bools: a > b means a && !b (true > false only)
        queue.parallel_for<GtKernelBool>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] && !b_ptr[idx]);
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for gt comparison");
    }

    queue.wait_and_throw();
    return output;
}

// ============================================================================
// Element-wise Greater Than or Equal (>=)
// ============================================================================

auto ge_kernel(const Tensor& a_in, const Tensor& b_in, sycl::queue& queue) -> Tensor {
    const Tensor a = a_in.is_contiguous() ? a_in : a_in.contiguous();
    const Tensor b = b_in.is_contiguous() ? b_in : b_in.contiguous();
    auto a_shape = a.shape();
    auto b_shape = b.shape();

    bool same_shape = shapes_match(a_shape, b_shape);

    if (!same_shape) {
        auto out_shape = broadcast_shapes(a_shape, b_shape);
        auto info = compute_broadcast_info(a_shape, b_shape, out_shape);
        return dispatch_broadcast_compare<
            BroadcastGeFloat32, BroadcastGeFloat64, BroadcastGeInt32, BroadcastGeInt64,
            BroadcastGeFloat16, BroadcastGeBFloat16, BroadcastGeBool>(
            a, b, info, out_shape, queue, [](auto x, auto y) { return x >= y; });
    }

    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument("Tensor dtypes must match for comparison");
    }

    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  DType::Bool, a.device());

    const int64_t numel = a.numel();

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<GeKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] >= b_ptr[idx]);
        });
    }
    else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<GeKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] >= b_ptr[idx]);
        });
    }
    else if (a.dtype() == DType::Int32) {
        const int32_t* a_ptr = get_data_ptr<const int32_t>(a);
        const int32_t* b_ptr = get_data_ptr<const int32_t>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<GeKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] >= b_ptr[idx]);
        });
    }
    else if (a.dtype() == DType::Int64) {
        const int64_t* a_ptr = get_data_ptr<const int64_t>(a);
        const int64_t* b_ptr = get_data_ptr<const int64_t>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<GeKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] >= b_ptr[idx]);
        });
    }
    else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<GeKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] >= b_ptr[idx]);
        });
    }
    else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<GeKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (bf16_to_f32(a_ptr[idx]) >= bf16_to_f32(b_ptr[idx]));
        });
    }
    else if (a.dtype() == DType::Bool) {
        const bool* a_ptr = get_data_ptr<const bool>(a);
        const bool* b_ptr = get_data_ptr<const bool>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        // For bools: a >= b means a || !b (anything >= false, or true >= true)
        queue.parallel_for<GeKernelBool>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] || !b_ptr[idx]);
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for ge comparison");
    }

    queue.wait_and_throw();
    return output;
}

} // namespace oneapi
} // namespace tenzor
