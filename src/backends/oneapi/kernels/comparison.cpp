#include "tenzor/core/tensor.hpp"
#include <sycl/sycl.hpp>
#include <stdexcept>

// Forward declaration for broadcasting fallback
namespace tenzor {
    auto eq(const Tensor& a, const Tensor& b) -> Tensor;
    auto ne(const Tensor& a, const Tensor& b) -> Tensor;
    auto lt(const Tensor& a, const Tensor& b) -> Tensor;
    auto le(const Tensor& a, const Tensor& b) -> Tensor;
    auto gt(const Tensor& a, const Tensor& b) -> Tensor;
    auto ge(const Tensor& a, const Tensor& b) -> Tensor;
}

namespace tenzor {
namespace oneapi {

// SYCL Kernel name classes (using functors for SYCL 2025.2 compatibility)
struct EqKernelFloat32 {};
struct EqKernelFloat64 {};
struct EqKernelInt32 {};
struct EqKernelInt64 {};
struct EqKernelBool {};
struct NeKernelFloat32 {};
struct NeKernelFloat64 {};
struct NeKernelInt32 {};
struct NeKernelInt64 {};
struct NeKernelBool {};
struct LtKernelFloat32 {};
struct LtKernelFloat64 {};
struct LtKernelInt32 {};
struct LtKernelInt64 {};
struct LtKernelBool {};
struct LeKernelFloat32 {};
struct LeKernelFloat64 {};
struct LeKernelInt32 {};
struct LeKernelInt64 {};
struct LeKernelBool {};
struct GtKernelFloat32 {};
struct GtKernelFloat64 {};
struct GtKernelInt32 {};
struct GtKernelInt64 {};
struct GtKernelBool {};
struct GeKernelFloat32 {};
struct GeKernelFloat64 {};
struct GeKernelInt32 {};
struct GeKernelInt64 {};
struct GeKernelBool {};

// Helper function to get typed pointer from tensor
template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

// Helper to check if shapes match
inline auto shapes_match(std::span<const int64_t> a, std::span<const int64_t> b) -> bool {
    return std::equal(a.begin(), a.end(), b.begin(), b.end());
}

// ============================================================================
// Element-wise Equal (==)
// ============================================================================

auto eq_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    // Validate inputs
    auto a_shape = a.shape();
    auto b_shape = b.shape();

    // Check if shapes match exactly
    bool same_shape = shapes_match(a_shape, b_shape);

    // If shapes don't match, fall back to CPU for broadcasting
    if (!same_shape) {
        auto a_cpu = a.to(Device::cpu());
        auto b_cpu = b.to(Device::cpu());
        auto result_cpu = tenzor::eq(a_cpu, b_cpu);
        return result_cpu.to(a.device());
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
        }).wait();
    }
    else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<EqKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] == b_ptr[idx]);
        }).wait();
    }
    else if (a.dtype() == DType::Int32) {
        const int32_t* a_ptr = get_data_ptr<const int32_t>(a);
        const int32_t* b_ptr = get_data_ptr<const int32_t>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<EqKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] == b_ptr[idx]);
        }).wait();
    }
    else if (a.dtype() == DType::Int64) {
        const int64_t* a_ptr = get_data_ptr<const int64_t>(a);
        const int64_t* b_ptr = get_data_ptr<const int64_t>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<EqKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] == b_ptr[idx]);
        }).wait();
    }
    else if (a.dtype() == DType::Bool) {
        const bool* a_ptr = get_data_ptr<const bool>(a);
        const bool* b_ptr = get_data_ptr<const bool>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<EqKernelBool>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] == b_ptr[idx]);
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for eq comparison");
    }

    return output;
}

// ============================================================================
// Element-wise Not Equal (!=)
// ============================================================================

auto ne_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    auto a_shape = a.shape();
    auto b_shape = b.shape();

    bool same_shape = shapes_match(a_shape, b_shape);

    if (!same_shape) {
        auto a_cpu = a.to(Device::cpu());
        auto b_cpu = b.to(Device::cpu());
        auto result_cpu = tenzor::ne(a_cpu, b_cpu);
        return result_cpu.to(a.device());
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
        }).wait();
    }
    else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<NeKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] != b_ptr[idx]);
        }).wait();
    }
    else if (a.dtype() == DType::Int32) {
        const int32_t* a_ptr = get_data_ptr<const int32_t>(a);
        const int32_t* b_ptr = get_data_ptr<const int32_t>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<NeKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] != b_ptr[idx]);
        }).wait();
    }
    else if (a.dtype() == DType::Int64) {
        const int64_t* a_ptr = get_data_ptr<const int64_t>(a);
        const int64_t* b_ptr = get_data_ptr<const int64_t>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<NeKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] != b_ptr[idx]);
        }).wait();
    }
    else if (a.dtype() == DType::Bool) {
        const bool* a_ptr = get_data_ptr<const bool>(a);
        const bool* b_ptr = get_data_ptr<const bool>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<NeKernelBool>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] != b_ptr[idx]);
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for ne comparison");
    }

    return output;
}

// ============================================================================
// Element-wise Less Than (<)
// ============================================================================

auto lt_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    auto a_shape = a.shape();
    auto b_shape = b.shape();

    bool same_shape = shapes_match(a_shape, b_shape);

    if (!same_shape) {
        auto a_cpu = a.to(Device::cpu());
        auto b_cpu = b.to(Device::cpu());
        auto result_cpu = tenzor::lt(a_cpu, b_cpu);
        return result_cpu.to(a.device());
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
        }).wait();
    }
    else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<LtKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] < b_ptr[idx]);
        }).wait();
    }
    else if (a.dtype() == DType::Int32) {
        const int32_t* a_ptr = get_data_ptr<const int32_t>(a);
        const int32_t* b_ptr = get_data_ptr<const int32_t>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<LtKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] < b_ptr[idx]);
        }).wait();
    }
    else if (a.dtype() == DType::Int64) {
        const int64_t* a_ptr = get_data_ptr<const int64_t>(a);
        const int64_t* b_ptr = get_data_ptr<const int64_t>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<LtKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] < b_ptr[idx]);
        }).wait();
    }
    else if (a.dtype() == DType::Bool) {
        const bool* a_ptr = get_data_ptr<const bool>(a);
        const bool* b_ptr = get_data_ptr<const bool>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        // For bools: false < true (0 < 1)
        queue.parallel_for<LtKernelBool>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (!a_ptr[idx] && b_ptr[idx]);
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for lt comparison");
    }

    return output;
}

// ============================================================================
// Element-wise Less Than or Equal (<=)
// ============================================================================

auto le_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    auto a_shape = a.shape();
    auto b_shape = b.shape();

    bool same_shape = shapes_match(a_shape, b_shape);

    if (!same_shape) {
        auto a_cpu = a.to(Device::cpu());
        auto b_cpu = b.to(Device::cpu());
        auto result_cpu = tenzor::le(a_cpu, b_cpu);
        return result_cpu.to(a.device());
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
        }).wait();
    }
    else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<LeKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] <= b_ptr[idx]);
        }).wait();
    }
    else if (a.dtype() == DType::Int32) {
        const int32_t* a_ptr = get_data_ptr<const int32_t>(a);
        const int32_t* b_ptr = get_data_ptr<const int32_t>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<LeKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] <= b_ptr[idx]);
        }).wait();
    }
    else if (a.dtype() == DType::Int64) {
        const int64_t* a_ptr = get_data_ptr<const int64_t>(a);
        const int64_t* b_ptr = get_data_ptr<const int64_t>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<LeKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] <= b_ptr[idx]);
        }).wait();
    }
    else if (a.dtype() == DType::Bool) {
        const bool* a_ptr = get_data_ptr<const bool>(a);
        const bool* b_ptr = get_data_ptr<const bool>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        // For bools: a <= b means !a || b (false <= anything, or true <= true)
        queue.parallel_for<LeKernelBool>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (!a_ptr[idx] || b_ptr[idx]);
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for le comparison");
    }

    return output;
}

// ============================================================================
// Element-wise Greater Than (>)
// ============================================================================

auto gt_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    auto a_shape = a.shape();
    auto b_shape = b.shape();

    bool same_shape = shapes_match(a_shape, b_shape);

    if (!same_shape) {
        auto a_cpu = a.to(Device::cpu());
        auto b_cpu = b.to(Device::cpu());
        auto result_cpu = tenzor::gt(a_cpu, b_cpu);
        return result_cpu.to(a.device());
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
        }).wait();
    }
    else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<GtKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] > b_ptr[idx]);
        }).wait();
    }
    else if (a.dtype() == DType::Int32) {
        const int32_t* a_ptr = get_data_ptr<const int32_t>(a);
        const int32_t* b_ptr = get_data_ptr<const int32_t>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<GtKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] > b_ptr[idx]);
        }).wait();
    }
    else if (a.dtype() == DType::Int64) {
        const int64_t* a_ptr = get_data_ptr<const int64_t>(a);
        const int64_t* b_ptr = get_data_ptr<const int64_t>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<GtKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] > b_ptr[idx]);
        }).wait();
    }
    else if (a.dtype() == DType::Bool) {
        const bool* a_ptr = get_data_ptr<const bool>(a);
        const bool* b_ptr = get_data_ptr<const bool>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        // For bools: a > b means a && !b (true > false only)
        queue.parallel_for<GtKernelBool>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] && !b_ptr[idx]);
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for gt comparison");
    }

    return output;
}

// ============================================================================
// Element-wise Greater Than or Equal (>=)
// ============================================================================

auto ge_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    auto a_shape = a.shape();
    auto b_shape = b.shape();

    bool same_shape = shapes_match(a_shape, b_shape);

    if (!same_shape) {
        auto a_cpu = a.to(Device::cpu());
        auto b_cpu = b.to(Device::cpu());
        auto result_cpu = tenzor::ge(a_cpu, b_cpu);
        return result_cpu.to(a.device());
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
        }).wait();
    }
    else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<GeKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] >= b_ptr[idx]);
        }).wait();
    }
    else if (a.dtype() == DType::Int32) {
        const int32_t* a_ptr = get_data_ptr<const int32_t>(a);
        const int32_t* b_ptr = get_data_ptr<const int32_t>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<GeKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] >= b_ptr[idx]);
        }).wait();
    }
    else if (a.dtype() == DType::Int64) {
        const int64_t* a_ptr = get_data_ptr<const int64_t>(a);
        const int64_t* b_ptr = get_data_ptr<const int64_t>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<GeKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] >= b_ptr[idx]);
        }).wait();
    }
    else if (a.dtype() == DType::Bool) {
        const bool* a_ptr = get_data_ptr<const bool>(a);
        const bool* b_ptr = get_data_ptr<const bool>(b);
        bool* out_ptr = get_data_ptr<bool>(output);

        // For bools: a >= b means a || !b (anything >= false, or true >= true)
        queue.parallel_for<GeKernelBool>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = (a_ptr[idx] || !b_ptr[idx]);
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for ge comparison");
    }

    return output;
}

} // namespace oneapi
} // namespace tenzor
