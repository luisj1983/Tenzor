#include "tenzor/core/tensor.hpp"
#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include <sycl/sycl.hpp>
#include <limits>
#include <stdexcept>
#include <cmath>

namespace tenzor {
namespace oneapi {

// RAII wrapper for sycl::malloc_device allocations
struct SyclDeviceGuard {
    void* ptr; sycl::queue& q;
    SyclDeviceGuard(void* p, sycl::queue& q) : ptr(p), q(q) {}
    ~SyclDeviceGuard() { if (ptr) sycl::free(ptr, q); }
    SyclDeviceGuard(const SyclDeviceGuard&) = delete;
    SyclDeviceGuard& operator=(const SyclDeviceGuard&) = delete;
};

// Kernel class declarations for SYCL
class StdAllKernelFloat32;
class StdAllKernelFloat64;
class StdDimKernelFloat32;
class StdDimKernelFloat64;
class VarKernelFloat32;
class VarKernelFloat64;
class VarDimKernelFloat32;
class VarDimKernelFloat64;
class ProdKernelFloat32;
class ProdKernelFloat64;
class ProdReducePhase1Float32;
class ProdReducePhase2Float32;
class ProdReducePhase1Float64;
class ProdReducePhase2Float64;
class ProdDimKernelFloat32;
class ProdDimKernelFloat64;
class NormReducePhase1Float32;
class NormReducePhase2Float32;
class NormReducePhase1Float64;
class NormReducePhase2Float64;
class NormDimKernelFloat32;
class NormDimKernelFloat64;

// Helper function to get typed pointer from tensor
template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

/**
 * @brief Standard deviation operation wrapper.
 *
 * Computes the standard deviation of tensor elements.
 * Currently supports reduction over all dimensions (dim=-1).
 * OpAttributes wrapper for std operation.
 *
 * Formula: std = sqrt(sum((x - mean)^2) / (n - correction))
 *
 * @param input Input tensor
 * @param attrs Operation attributes containing dim, keepdim, correction
 * @param queue SYCL queue for execution
 * @return Tensor Standard deviation tensor
 */
auto std_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor {
    // INT64_MIN is the project-wide "all dims" sentinel. Treat any other
    // negative dim as a back-from-end axis (e.g. -1 = last dim).
    int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
    bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
    int64_t correction = attrs.get_int(AttrKey::Correction, 1);

    auto shape_span = input.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
    const int64_t ndim = static_cast<int64_t>(shape.size());

    if (dim != INT64_MIN && dim < 0) dim += ndim;

    bool full_reduction = (dim == INT64_MIN || ndim == 1);

    if (full_reduction) {
        // Compute total number of elements
        int64_t total_size = 1;
        for (auto s : shape) total_size *= s;

        std::vector<int64_t> out_shape;
        if (keepdim) {
            out_shape.resize(shape.size(), 1);
        } else {
            // Full reduction without keepdim yields a zero-dim scalar
            // (shape {}), not a length-1 vector. This matches the CPU
            // backend and PyTorch conventions.
            out_shape = {};
        }

        if (input.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(input);
            const float f_total = static_cast<float>(total_size);
            int64_t divisor = total_size - correction;
            if (divisor <= 0) divisor = 1;
            const float f_divisor = static_cast<float>(divisor);

            float* mean_device = sycl::malloc_device<float>(1, queue);
            SyclDeviceGuard mean_guard(mean_device, queue);
            queue.fill(mean_device, 0.0f, 1);

            queue.parallel_for<class MeanAllKernelFloat32>(sycl::range<1>(total_size),
                [=](sycl::id<1> idx) {
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device> atomic_mean(mean_device[0]);
                    atomic_mean += in_ptr[idx];
            });

            float* var_device = sycl::malloc_device<float>(1, queue);
            SyclDeviceGuard var_guard(var_device, queue);
            queue.fill(var_device, 0.0f, 1);

            // Read mean from device memory directly — avoids D-to-H round-trip
            queue.parallel_for<class VarAllKernelFloat32>(sycl::range<1>(total_size),
                [=](sycl::id<1> idx) {
                    float mean_val = mean_device[0] / f_total;
                    float diff = in_ptr[idx] - mean_val;
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device> atomic_var(var_device[0]);
                    atomic_var += diff * diff;
            });

            // Compute final std on device and write directly to output
            Tensor output(out_shape, input.dtype(), input.device());
            float* out_ptr = get_data_ptr<float>(output);
            queue.single_task<class StdFinalizeFloat32>([=]() {
                out_ptr[0] = sycl::sqrt(var_device[0] / f_divisor + 1e-8f);
            }).wait();
            return output;
        }
        else if (input.dtype() == DType::Float64) {
            const double* in_ptr = get_data_ptr<const double>(input);
            const double d_total = static_cast<double>(total_size);
            int64_t divisor = total_size - correction;
            if (divisor <= 0) divisor = 1;
            const double d_divisor = static_cast<double>(divisor);

            double* mean_device = sycl::malloc_device<double>(1, queue);
            SyclDeviceGuard mean_guard(mean_device, queue);
            queue.fill(mean_device, 0.0, 1);

            queue.parallel_for<class MeanAllKernelFloat64>(sycl::range<1>(total_size),
                [=](sycl::id<1> idx) {
                    sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device> atomic_mean(mean_device[0]);
                    atomic_mean += in_ptr[idx];
            });

            double* var_device = sycl::malloc_device<double>(1, queue);
            SyclDeviceGuard var_guard(var_device, queue);
            queue.fill(var_device, 0.0, 1);

            // Read mean from device memory directly — avoids D-to-H round-trip
            queue.parallel_for<class VarAllKernelFloat64>(sycl::range<1>(total_size),
                [=](sycl::id<1> idx) {
                    double mean_val = mean_device[0] / d_total;
                    double diff = in_ptr[idx] - mean_val;
                    sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device> atomic_var(var_device[0]);
                    atomic_var += diff * diff;
            });

            // Compute final std on device and write directly to output
            Tensor output(out_shape, input.dtype(), input.device());
            double* out_ptr = get_data_ptr<double>(output);
            queue.single_task<class StdFinalizeFloat64>([=]() {
                out_ptr[0] = sycl::sqrt(var_device[0] / d_divisor + 1e-8);
            }).wait();
            return output;
        }
        else {
            throw std::runtime_error("Unsupported dtype for std_kernel");
        }
    } else {
        // Dimensional reduction: flatten to (outer_size x dim_size x inner_size)
        if (dim < 0 || dim >= ndim) {
            throw std::invalid_argument("std: dim out of range");
        }

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
        int64_t dim_size = shape[dim];
        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

        int64_t output_size = outer_size * inner_size;

        // Compute output shape
        std::vector<int64_t> out_shape;
        out_shape.reserve(keepdim ? shape.size() : shape.size() - 1);
        for (size_t i = 0; i < shape.size(); ++i) {
            if (static_cast<int64_t>(i) == dim) {
                if (keepdim) out_shape.push_back(1);
            } else {
                out_shape.push_back(shape[i]);
            }
        }

        if (input.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(input);
            Tensor output(out_shape, input.dtype(), input.device());
            float* out_ptr = get_data_ptr<float>(output);
            int64_t corr = correction;

            queue.parallel_for<StdDimKernelFloat32>(
                sycl::range<1>(output_size), [=](sycl::id<1> gid) {
                    int64_t idx = gid[0];
                    int64_t outer = idx / inner_size;
                    int64_t inner = idx % inner_size;

                    // Compute mean along dim
                    float sum = 0.0f;
                    for (int64_t d = 0; d < dim_size; ++d) {
                        sum += in_ptr[outer * dim_size * inner_size + d * inner_size + inner];
                    }
                    float mean = sum / static_cast<float>(dim_size);

                    // Compute variance along dim
                    float var_sum = 0.0f;
                    for (int64_t d = 0; d < dim_size; ++d) {
                        float diff = in_ptr[outer * dim_size * inner_size + d * inner_size + inner] - mean;
                        var_sum += diff * diff;
                    }
                    int64_t divisor = dim_size - corr;
                    if (divisor <= 0) divisor = 1;
                    out_ptr[idx] = sycl::sqrt(var_sum / static_cast<float>(divisor));
                }
            );
            queue.wait_and_throw();
            return output;
        }
        else if (input.dtype() == DType::Float64) {
            const double* in_ptr = get_data_ptr<const double>(input);
            Tensor output(out_shape, input.dtype(), input.device());
            double* out_ptr = get_data_ptr<double>(output);
            int64_t corr = correction;

            queue.parallel_for<StdDimKernelFloat64>(
                sycl::range<1>(output_size), [=](sycl::id<1> gid) {
                    int64_t idx = gid[0];
                    int64_t outer = idx / inner_size;
                    int64_t inner = idx % inner_size;

                    double sum = 0.0;
                    for (int64_t d = 0; d < dim_size; ++d) {
                        sum += in_ptr[outer * dim_size * inner_size + d * inner_size + inner];
                    }
                    double mean = sum / static_cast<double>(dim_size);

                    double var_sum = 0.0;
                    for (int64_t d = 0; d < dim_size; ++d) {
                        double diff = in_ptr[outer * dim_size * inner_size + d * inner_size + inner] - mean;
                        var_sum += diff * diff;
                    }
                    int64_t divisor = dim_size - corr;
                    if (divisor <= 0) divisor = 1;
                    out_ptr[idx] = sycl::sqrt(var_sum / static_cast<double>(divisor));
                }
            );
            queue.wait_and_throw();
            return output;
        }
        else {
            throw std::runtime_error("Unsupported dtype for std_kernel");
        }
    }
}

/**
 * @brief Variance operation wrapper.
 *
 * Computes the variance of tensor elements.
 * Currently supports reduction over all dimensions (dim=-1).
 * OpAttributes wrapper for var operation.
 *
 * Formula: var = sum((x - mean)^2) / (n - correction)
 *
 * @param input Input tensor
 * @param attrs Operation attributes containing dim, keepdim, correction
 * @param queue SYCL queue for execution
 * @return Tensor Variance tensor
 */
auto var_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor {
    // INT64_MIN is the project-wide "all dims" sentinel. The legacy
    // OneAPI convention also accepted -1 as full reduction; that
    // ambiguates `var(v, dim=-1)` (last dim) with full reduction. Treat
    // INT64_MIN as full reduction and normalize any other negative dim to
    // positive.
    int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
    bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
    int64_t correction = attrs.get_int(AttrKey::Correction, 1);

    auto shape_span = input.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
    const int64_t ndim = static_cast<int64_t>(shape.size());

    if (dim != INT64_MIN && dim < 0) dim += ndim;

    bool full_reduction = (dim == INT64_MIN || ndim == 1);

    if (full_reduction) {
        int64_t total_size = 1;
        for (auto s : shape) total_size *= s;

        std::vector<int64_t> out_shape;
        if (keepdim) {
            out_shape.resize(shape.size(), 1);
        } else {
            // Full reduction without keepdim yields a zero-dim scalar
            // (shape {}), not a length-1 vector. This matches the CPU
            // backend and PyTorch conventions.
            out_shape = {};
        }

        if (input.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(input);
            const float f_total = static_cast<float>(total_size);
            int64_t divisor = total_size - correction;
            if (divisor <= 0) divisor = 1;
            const float f_divisor = static_cast<float>(divisor);

            float* mean_device = sycl::malloc_device<float>(1, queue);
            SyclDeviceGuard mean_guard(mean_device, queue);
            queue.fill(mean_device, 0.0f, 1);

            queue.parallel_for<class MeanVarKernelFloat32>(sycl::range<1>(total_size),
                [=](sycl::id<1> idx) {
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device> atomic_mean(mean_device[0]);
                    atomic_mean += in_ptr[idx];
            });

            float* var_device = sycl::malloc_device<float>(1, queue);
            SyclDeviceGuard var_guard(var_device, queue);
            queue.fill(var_device, 0.0f, 1);

            // Read mean from device memory directly — avoids D-to-H round-trip
            queue.parallel_for<class VarKernelFloat32>(sycl::range<1>(total_size),
                [=](sycl::id<1> idx) {
                    float mean_val = mean_device[0] / f_total;
                    float diff = in_ptr[idx] - mean_val;
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device> atomic_var(var_device[0]);
                    atomic_var += diff * diff;
            });

            // Compute final variance on device and write directly to output
            Tensor output(out_shape, input.dtype(), input.device());
            float* out_ptr = get_data_ptr<float>(output);
            queue.single_task<class VarFinalizeFloat32>([=]() {
                out_ptr[0] = var_device[0] / f_divisor;
            }).wait();
            return output;
        }
        else if (input.dtype() == DType::Float64) {
            const double* in_ptr = get_data_ptr<const double>(input);
            const double d_total = static_cast<double>(total_size);
            int64_t divisor = total_size - correction;
            if (divisor <= 0) divisor = 1;
            const double d_divisor = static_cast<double>(divisor);

            double* mean_device = sycl::malloc_device<double>(1, queue);
            SyclDeviceGuard mean_guard(mean_device, queue);
            queue.fill(mean_device, 0.0, 1);

            queue.parallel_for<class MeanVarKernelFloat64>(sycl::range<1>(total_size),
                [=](sycl::id<1> idx) {
                    sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device> atomic_mean(mean_device[0]);
                    atomic_mean += in_ptr[idx];
            });

            double* var_device = sycl::malloc_device<double>(1, queue);
            SyclDeviceGuard var_guard(var_device, queue);
            queue.fill(var_device, 0.0, 1);

            // Read mean from device memory directly — avoids D-to-H round-trip
            queue.parallel_for<class VarKernelFloat64>(sycl::range<1>(total_size),
                [=](sycl::id<1> idx) {
                    double mean_val = mean_device[0] / d_total;
                    double diff = in_ptr[idx] - mean_val;
                    sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device> atomic_var(var_device[0]);
                    atomic_var += diff * diff;
            });

            // Compute final variance on device and write directly to output
            Tensor output(out_shape, input.dtype(), input.device());
            double* out_ptr = get_data_ptr<double>(output);
            queue.single_task<class VarFinalizeFloat64>([=]() {
                out_ptr[0] = var_device[0] / d_divisor;
            }).wait();
            return output;
        }
        else {
            throw std::runtime_error("Unsupported dtype for var_kernel");
        }
    } else {
        // Dimensional reduction: flatten to (outer_size x dim_size x inner_size)
        if (dim < 0 || dim >= ndim) {
            throw std::invalid_argument("var: dim out of range");
        }

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
        int64_t dim_size = shape[dim];
        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

        int64_t output_size = outer_size * inner_size;

        std::vector<int64_t> out_shape;
        out_shape.reserve(keepdim ? shape.size() : shape.size() - 1);
        for (size_t i = 0; i < shape.size(); ++i) {
            if (static_cast<int64_t>(i) == dim) {
                if (keepdim) out_shape.push_back(1);
            } else {
                out_shape.push_back(shape[i]);
            }
        }

        if (input.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(input);
            Tensor output(out_shape, input.dtype(), input.device());
            float* out_ptr = get_data_ptr<float>(output);
            int64_t corr = correction;

            queue.parallel_for<VarDimKernelFloat32>(
                sycl::range<1>(output_size), [=](sycl::id<1> gid) {
                    int64_t idx = gid[0];
                    int64_t outer = idx / inner_size;
                    int64_t inner = idx % inner_size;

                    // Compute mean along dim
                    float sum = 0.0f;
                    for (int64_t d = 0; d < dim_size; ++d) {
                        sum += in_ptr[outer * dim_size * inner_size + d * inner_size + inner];
                    }
                    float mean = sum / static_cast<float>(dim_size);

                    // Compute variance along dim
                    float var_sum = 0.0f;
                    for (int64_t d = 0; d < dim_size; ++d) {
                        float diff = in_ptr[outer * dim_size * inner_size + d * inner_size + inner] - mean;
                        var_sum += diff * diff;
                    }
                    int64_t divisor = dim_size - corr;
                    if (divisor <= 0) divisor = 1;
                    out_ptr[idx] = var_sum / static_cast<float>(divisor);
                }
            );
            queue.wait_and_throw();
            return output;
        }
        else if (input.dtype() == DType::Float64) {
            const double* in_ptr = get_data_ptr<const double>(input);
            Tensor output(out_shape, input.dtype(), input.device());
            double* out_ptr = get_data_ptr<double>(output);
            int64_t corr = correction;

            queue.parallel_for<VarDimKernelFloat64>(
                sycl::range<1>(output_size), [=](sycl::id<1> gid) {
                    int64_t idx = gid[0];
                    int64_t outer = idx / inner_size;
                    int64_t inner = idx % inner_size;

                    double sum = 0.0;
                    for (int64_t d = 0; d < dim_size; ++d) {
                        sum += in_ptr[outer * dim_size * inner_size + d * inner_size + inner];
                    }
                    double mean = sum / static_cast<double>(dim_size);

                    double var_sum = 0.0;
                    for (int64_t d = 0; d < dim_size; ++d) {
                        double diff = in_ptr[outer * dim_size * inner_size + d * inner_size + inner] - mean;
                        var_sum += diff * diff;
                    }
                    int64_t divisor = dim_size - corr;
                    if (divisor <= 0) divisor = 1;
                    out_ptr[idx] = var_sum / static_cast<double>(divisor);
                }
            );
            queue.wait_and_throw();
            return output;
        }
        else {
            throw std::runtime_error("Unsupported dtype for var_kernel");
        }
    }
}

/**
 * @brief Product operation wrapper.
 *
 * Computes the product of all tensor elements.
 * Currently supports reduction over all dimensions (dim=-1).
 * OpAttributes wrapper for prod operation.
 *
 * Formula: prod = x[0] * x[1] * x[2] * ... * x[n-1]
 *
 * @param input Input tensor
 * @param attrs Operation attributes containing dim, keepdim
 * @param queue SYCL queue for execution
 * @return Tensor Product tensor
 */
auto prod_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor {
    // Float16/BFloat16: compute in Float32 for numerical stability and
    // because the SYCL kernels below are only specialised for Float32/
    // Float64/Int32/Int64. Matches the widen-narrow pattern used by the
    // norm/var/std kernels in this file.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        auto result = prod_kernel(input.to(DType::Float32), attrs, queue);
        return result.to(orig);
    }

    // INT64_MIN is the project-wide "all dims" sentinel. Treat any other
    // negative dim as a back-from-end axis.
    int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
    bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);

    auto shape_span = input.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
    auto strides_span = input.strides();
    std::vector<int64_t> strides(strides_span.begin(), strides_span.end());

    if (dim != INT64_MIN && dim < 0) dim += static_cast<int64_t>(shape.size());

    // Compute output shape
    std::vector<int64_t> out_shape;
    if (dim == INT64_MIN) {
        // Full reduction — scalar (shape {}) without keepdim, matches
        // CPU backend and PyTorch conventions.
        out_shape = keepdim ? std::vector<int64_t>(shape.size(), 1) : std::vector<int64_t>{};
    } else {
        // Dimensional reduction
        if (dim < 0 || dim >= static_cast<int64_t>(shape.size())) {
            throw std::invalid_argument("prod: dim out of range");
        }
        out_shape.reserve(keepdim ? shape.size() : shape.size() - 1);
        for (size_t i = 0; i < shape.size(); ++i) {
            if (static_cast<int64_t>(i) == dim) {
                if (keepdim) out_shape.push_back(1);
            } else {
                out_shape.push_back(shape[i]);
            }
        }
    }

    // Compute total number of elements
    int64_t total_size = 1;
    for (auto s : shape) {
        total_size *= s;
    }

    // Compute output size
    int64_t output_size = 1;
    for (auto s : out_shape) {
        output_size *= s;
    }

    // Device-side product reduction using two-phase SYCL work-group reduction
    // Phase 1: nd_range kernel with work-groups of 256, grid-stride accumulation,
    //          work-group reduce to partial buffer (identity = 1, not 0)
    // Phase 2: Single work-group reduces partial results

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        Tensor output(out_shape, input.dtype(), input.device());
        float* out_ptr = get_data_ptr<float>(output);

        if (dim == INT64_MIN) {
            // Full reduction on device
            constexpr int64_t WG_SIZE = 256;
            int64_t num_wgs = (total_size + WG_SIZE - 1) / WG_SIZE;
            auto partial_buf = sycl::malloc_device<float>(num_wgs, queue);
            SyclDeviceGuard partial_guard(partial_buf, queue);

            // Phase 1: each work-group computes partial product
            queue.parallel_for<ProdReducePhase1Float32>(
                sycl::nd_range<1>(num_wgs * WG_SIZE, WG_SIZE),
                [=](sycl::nd_item<1> item) {
                    int64_t gid = item.get_global_id(0);
                    float val = (gid < total_size) ? in_ptr[gid] : 1.0f;
                    float prod = sycl::reduce_over_group(
                        item.get_group(), val, 1.0f, std::multiplies<float>());
                    if (item.get_local_id(0) == 0) {
                        partial_buf[item.get_group(0)] = prod;
                    }
                }
            );

            // Phase 2: single work-group reduces partial products
            queue.parallel_for<ProdReducePhase2Float32>(
                sycl::nd_range<1>(WG_SIZE, WG_SIZE),
                [=](sycl::nd_item<1> item) {
                    int64_t lid = item.get_local_id(0);
                    float val = (lid < num_wgs) ? partial_buf[lid] : 1.0f;
                    float prod = sycl::reduce_over_group(
                        item.get_group(), val, 1.0f, std::multiplies<float>());
                    if (lid == 0) out_ptr[0] = prod;
                }
            );
            queue.wait_and_throw();
        } else {
            // Dimensional reduction on device
            const int64_t ndim = shape.size();
            const int64_t dim_size = shape[dim];

            // Precompute strides on device
            auto d_strides = sycl::malloc_device<int64_t>(ndim, queue);
            SyclDeviceGuard strides_guard(d_strides, queue);
            auto d_shape = sycl::malloc_device<int64_t>(ndim, queue);
            SyclDeviceGuard shape_guard(d_shape, queue);
            queue.memcpy(d_strides, strides.data(), ndim * sizeof(int64_t));
            queue.memcpy(d_shape, shape.data(), ndim * sizeof(int64_t));
            // In-order queue guarantees ordering — no wait needed

            queue.parallel_for<ProdDimKernelFloat32>(
                sycl::range<1>(output_size), [=](sycl::id<1> gid) {
                    int64_t out_idx = gid[0];
                    int64_t tmp = out_idx;

                    // Compute base input index (with dim-th index = 0)
                    int64_t base_in_idx = 0;
                    for (int64_t d = ndim - 1; d >= 0; --d) {
                        if (d == dim) continue;
                        int64_t s = d_shape[d];
                        int64_t coord = tmp % s;
                        tmp /= s;
                        base_in_idx += coord * d_strides[d];
                    }

                    // Accumulate product along reduction dimension
                    float prod_value = 1.0f;
                    for (int64_t i = 0; i < dim_size; ++i) {
                        prod_value *= in_ptr[base_in_idx + i * d_strides[dim]];
                    }
                    out_ptr[out_idx] = prod_value;
                }
            );
            queue.wait_and_throw();
        }

        return output;
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        Tensor output(out_shape, input.dtype(), input.device());
        double* out_ptr = get_data_ptr<double>(output);

        if (dim == INT64_MIN) {
            // Full reduction on device
            constexpr int64_t WG_SIZE = 256;
            int64_t num_wgs = (total_size + WG_SIZE - 1) / WG_SIZE;
            auto partial_buf = sycl::malloc_device<double>(num_wgs, queue);
            SyclDeviceGuard partial_guard(partial_buf, queue);

            // Phase 1: each work-group computes partial product
            queue.parallel_for<ProdReducePhase1Float64>(
                sycl::nd_range<1>(num_wgs * WG_SIZE, WG_SIZE),
                [=](sycl::nd_item<1> item) {
                    int64_t gid = item.get_global_id(0);
                    double val = (gid < total_size) ? in_ptr[gid] : 1.0;
                    double prod = sycl::reduce_over_group(
                        item.get_group(), val, 1.0, std::multiplies<double>());
                    if (item.get_local_id(0) == 0) {
                        partial_buf[item.get_group(0)] = prod;
                    }
                }
            );

            // Phase 2: single work-group reduces partial products
            queue.parallel_for<ProdReducePhase2Float64>(
                sycl::nd_range<1>(WG_SIZE, WG_SIZE),
                [=](sycl::nd_item<1> item) {
                    int64_t lid = item.get_local_id(0);
                    double val = (lid < num_wgs) ? partial_buf[lid] : 1.0;
                    double prod = sycl::reduce_over_group(
                        item.get_group(), val, 1.0, std::multiplies<double>());
                    if (lid == 0) out_ptr[0] = prod;
                }
            );
            queue.wait_and_throw();
        } else {
            // Dimensional reduction on device
            const int64_t ndim = shape.size();
            const int64_t dim_size = shape[dim];

            auto d_strides = sycl::malloc_device<int64_t>(ndim, queue);
            SyclDeviceGuard strides_guard(d_strides, queue);
            auto d_shape = sycl::malloc_device<int64_t>(ndim, queue);
            SyclDeviceGuard shape_guard(d_shape, queue);
            queue.memcpy(d_strides, strides.data(), ndim * sizeof(int64_t));
            queue.memcpy(d_shape, shape.data(), ndim * sizeof(int64_t));
            // In-order queue guarantees ordering — no wait needed

            queue.parallel_for<ProdDimKernelFloat64>(
                sycl::range<1>(output_size), [=](sycl::id<1> gid) {
                    int64_t out_idx = gid[0];
                    int64_t tmp = out_idx;

                    int64_t base_in_idx = 0;
                    for (int64_t d = ndim - 1; d >= 0; --d) {
                        if (d == dim) continue;
                        int64_t s = d_shape[d];
                        int64_t coord = tmp % s;
                        tmp /= s;
                        base_in_idx += coord * d_strides[d];
                    }

                    double prod_value = 1.0;
                    for (int64_t i = 0; i < dim_size; ++i) {
                        prod_value *= in_ptr[base_in_idx + i * d_strides[dim]];
                    }
                    out_ptr[out_idx] = prod_value;
                }
            );
            queue.wait_and_throw();
        }

        return output;
    }
    else if (input.dtype() == DType::Int32) {
        const int32_t* in_ptr = get_data_ptr<const int32_t>(input);
        // Promote to Int64 to prevent overflow (matches PyTorch behavior)
        Tensor output(out_shape, DType::Int64, input.device());
        int64_t* out_ptr = get_data_ptr<int64_t>(output);

        if (dim == INT64_MIN) {
            // Full reduction — use shared memory for int64 accumulator
            auto prod_buf = sycl::malloc_shared<int64_t>(1, queue);
            SyclDeviceGuard prod_guard(prod_buf, queue);
            prod_buf[0] = 1;

            queue.parallel_for(sycl::range<1>(total_size),
                sycl::reduction(prod_buf, int64_t(1), std::multiplies<int64_t>()),
                [=](sycl::id<1> idx, auto& prod) {
                    prod.combine(static_cast<int64_t>(in_ptr[idx]));
                }
            );
            queue.wait_and_throw();
            out_ptr[0] = prod_buf[0];
        } else {
            // Dimensional reduction on device
            const int64_t ndim = shape.size();
            const int64_t dim_size = shape[dim];

            auto d_strides = sycl::malloc_device<int64_t>(ndim, queue);
            SyclDeviceGuard strides_guard(d_strides, queue);
            auto d_shape = sycl::malloc_device<int64_t>(ndim, queue);
            SyclDeviceGuard shape_guard(d_shape, queue);
            queue.memcpy(d_strides, strides.data(), ndim * sizeof(int64_t));
            queue.memcpy(d_shape, shape.data(), ndim * sizeof(int64_t));
            // In-order queue guarantees ordering — no wait needed

            queue.parallel_for(
                sycl::range<1>(output_size), [=](sycl::id<1> gid) {
                    int64_t out_idx = gid[0];
                    int64_t tmp = out_idx;

                    int64_t base_in_idx = 0;
                    for (int64_t d = ndim - 1; d >= 0; --d) {
                        if (d == dim) continue;
                        int64_t s = d_shape[d];
                        int64_t coord = tmp % s;
                        tmp /= s;
                        base_in_idx += coord * d_strides[d];
                    }

                    int64_t prod_value = 1;
                    for (int64_t i = 0; i < dim_size; ++i) {
                        prod_value *= static_cast<int64_t>(
                            in_ptr[base_in_idx + i * d_strides[dim]]);
                    }
                    out_ptr[out_idx] = prod_value;
                }
            );
            queue.wait_and_throw();
        }

        return output;
    }
    else if (input.dtype() == DType::Int64) {
        const int64_t* in_ptr = get_data_ptr<const int64_t>(input);
        Tensor output(out_shape, DType::Int64, input.device());
        int64_t* out_ptr = get_data_ptr<int64_t>(output);

        if (dim == INT64_MIN) {
            // Full reduction using shared accumulator
            auto prod_buf = sycl::malloc_shared<int64_t>(1, queue);
            SyclDeviceGuard prod_guard(prod_buf, queue);
            prod_buf[0] = 1;

            queue.parallel_for(sycl::range<1>(total_size),
                sycl::reduction(prod_buf, int64_t(1), std::multiplies<int64_t>()),
                [=](sycl::id<1> idx, auto& prod) {
                    prod.combine(in_ptr[idx]);
                }
            );
            queue.wait_and_throw();
            out_ptr[0] = prod_buf[0];
        } else {
            // Dimensional reduction on device
            const int64_t ndim = shape.size();
            const int64_t dim_size = shape[dim];

            auto d_strides = sycl::malloc_device<int64_t>(ndim, queue);
            SyclDeviceGuard strides_guard(d_strides, queue);
            auto d_shape = sycl::malloc_device<int64_t>(ndim, queue);
            SyclDeviceGuard shape_guard(d_shape, queue);
            queue.memcpy(d_strides, strides.data(), ndim * sizeof(int64_t));
            queue.memcpy(d_shape, shape.data(), ndim * sizeof(int64_t));

            queue.parallel_for(
                sycl::range<1>(output_size), [=](sycl::id<1> gid) {
                    int64_t out_idx = gid[0];
                    int64_t tmp = out_idx;

                    int64_t base_in_idx = 0;
                    for (int64_t d = ndim - 1; d >= 0; --d) {
                        if (d == dim) continue;
                        int64_t s = d_shape[d];
                        int64_t coord = tmp % s;
                        tmp /= s;
                        base_in_idx += coord * d_strides[d];
                    }

                    int64_t prod_value = 1;
                    for (int64_t i = 0; i < dim_size; ++i) {
                        prod_value *= in_ptr[base_in_idx + i * d_strides[dim]];
                    }
                    out_ptr[out_idx] = prod_value;
                }
            );
            queue.wait_and_throw();
        }

        return output;
    }
    else {
        throw std::runtime_error("Unsupported dtype for prod_kernel");
    }
}

// Norm kernel - compute Lp norm
auto norm_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor {
    // Float16/BFloat16: compute in Float32 for numerical stability
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        auto result = norm_kernel(input.to(DType::Float32), attrs, queue);
        return result.to(orig);
    }

    float p = static_cast<float>(attrs.get_float(AttrKey::P, 2.0));
    // INT64_MIN is the project-wide "all dims" sentinel. Treat any other
    // negative dim as a back-from-end axis.
    int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
    bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);

    auto shape_span = input.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
    const int64_t ndim = static_cast<int64_t>(shape.size());

    if (dim != INT64_MIN && dim < 0) dim += ndim;

    bool full_reduction = (dim == INT64_MIN || ndim == 1);

    if (full_reduction) {
        // Compute output shape
        std::vector<int64_t> out_shape;
        if (keepdim) {
            out_shape.resize(shape.size(), 1);
        } else {
            // Full reduction without keepdim yields a zero-dim scalar
            // (shape {}), not a length-1 vector. This matches the CPU
            // backend and PyTorch conventions.
            out_shape = {};
        }

        int64_t n = input.numel();
        if (n == 0) {
            throw std::runtime_error("norm: input tensor is empty");
        }

        // Device-side norm reduction using two-phase SYCL work-group reduction
        if (input.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(input);
            Tensor output(out_shape, input.dtype(), input.device());
            float* out_ptr = get_data_ptr<float>(output);

            constexpr int64_t WG_SIZE = 256;
            int64_t num_wgs = (n + WG_SIZE - 1) / WG_SIZE;
            auto partial_buf = sycl::malloc_device<float>(num_wgs, queue);
            SyclDeviceGuard partial_guard(partial_buf, queue);
            float p_val = p;

            queue.parallel_for<NormReducePhase1Float32>(
                sycl::nd_range<1>(num_wgs * WG_SIZE, WG_SIZE),
                [=](sycl::nd_item<1> item) {
                    int64_t gid = item.get_global_id(0);
                    float val = 0.0f;
                    if (gid < n) {
                        float x = in_ptr[gid];
                        if (p_val == 1.0f) {
                            val = sycl::fabs(x);
                        } else if (p_val == 2.0f) {
                            val = x * x;
                        } else {
                            val = sycl::pow(sycl::fabs(x), p_val);
                        }
                    }
                    float sum = sycl::reduce_over_group(
                        item.get_group(), val, sycl::plus<float>());
                    if (item.get_local_id(0) == 0) {
                        partial_buf[item.get_group(0)] = sum;
                    }
                }
            );

            queue.parallel_for<NormReducePhase2Float32>(
                sycl::nd_range<1>(WG_SIZE, WG_SIZE),
                [=](sycl::nd_item<1> item) {
                    int64_t lid = item.get_local_id(0);
                    float val = (lid < num_wgs) ? partial_buf[lid] : 0.0f;
                    float sum = sycl::reduce_over_group(
                        item.get_group(), val, sycl::plus<float>());
                    if (lid == 0) {
                        if (p_val == 2.0f) {
                            out_ptr[0] = sycl::sqrt(sum);
                        } else if (p_val == 1.0f) {
                            out_ptr[0] = sum;
                        } else {
                            out_ptr[0] = sycl::pow(sum, 1.0f / p_val);
                        }
                    }
                }
            );
            queue.wait_and_throw();
            return output;
        }
        else if (input.dtype() == DType::Float64) {
            const double* in_ptr = get_data_ptr<const double>(input);
            Tensor output(out_shape, input.dtype(), input.device());
            double* out_ptr = get_data_ptr<double>(output);

            constexpr int64_t WG_SIZE = 256;
            int64_t num_wgs = (n + WG_SIZE - 1) / WG_SIZE;
            auto partial_buf = sycl::malloc_device<double>(num_wgs, queue);
            SyclDeviceGuard partial_guard(partial_buf, queue);
            double p_val = static_cast<double>(p);

            queue.parallel_for<NormReducePhase1Float64>(
                sycl::nd_range<1>(num_wgs * WG_SIZE, WG_SIZE),
                [=](sycl::nd_item<1> item) {
                    int64_t gid = item.get_global_id(0);
                    double val = 0.0;
                    if (gid < n) {
                        double x = in_ptr[gid];
                        if (p_val == 1.0) {
                            val = sycl::fabs(x);
                        } else if (p_val == 2.0) {
                            val = x * x;
                        } else {
                            val = sycl::pow(sycl::fabs(x), p_val);
                        }
                    }
                    double sum = sycl::reduce_over_group(
                        item.get_group(), val, sycl::plus<double>());
                    if (item.get_local_id(0) == 0) {
                        partial_buf[item.get_group(0)] = sum;
                    }
                }
            );

            queue.parallel_for<NormReducePhase2Float64>(
                sycl::nd_range<1>(WG_SIZE, WG_SIZE),
                [=](sycl::nd_item<1> item) {
                    int64_t lid = item.get_local_id(0);
                    double val = (lid < num_wgs) ? partial_buf[lid] : 0.0;
                    double sum = sycl::reduce_over_group(
                        item.get_group(), val, sycl::plus<double>());
                    if (lid == 0) {
                        if (p_val == 2.0) {
                            out_ptr[0] = sycl::sqrt(sum);
                        } else if (p_val == 1.0) {
                            out_ptr[0] = sum;
                        } else {
                            out_ptr[0] = sycl::pow(sum, 1.0 / p_val);
                        }
                    }
                }
            );
            queue.wait_and_throw();
            return output;
        }
        else {
            throw std::runtime_error("Unsupported dtype for norm_kernel");
        }
    } else {
        // Dimensional reduction: flatten to (outer_size x dim_size x inner_size)
        if (dim < 0 || dim >= ndim) {
            throw std::invalid_argument("norm: dim out of range");
        }

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
        int64_t dim_size = shape[dim];
        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

        int64_t output_size = outer_size * inner_size;

        if (dim_size == 0) {
            throw std::runtime_error("norm: reduction dimension is empty");
        }

        std::vector<int64_t> out_shape;
        out_shape.reserve(keepdim ? shape.size() : shape.size() - 1);
        for (size_t i = 0; i < shape.size(); ++i) {
            if (static_cast<int64_t>(i) == dim) {
                if (keepdim) out_shape.push_back(1);
            } else {
                out_shape.push_back(shape[i]);
            }
        }

        if (input.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(input);
            Tensor output(out_shape, input.dtype(), input.device());
            float* out_ptr = get_data_ptr<float>(output);
            float p_val = p;

            queue.parallel_for<NormDimKernelFloat32>(
                sycl::range<1>(output_size), [=](sycl::id<1> gid) {
                    int64_t idx = gid[0];
                    int64_t outer = idx / inner_size;
                    int64_t inner = idx % inner_size;

                    float acc = 0.0f;
                    for (int64_t d = 0; d < dim_size; ++d) {
                        float x = in_ptr[outer * dim_size * inner_size + d * inner_size + inner];
                        if (p_val == 1.0f) {
                            acc += sycl::fabs(x);
                        } else if (p_val == 2.0f) {
                            acc += x * x;
                        } else {
                            acc += sycl::pow(sycl::fabs(x), p_val);
                        }
                    }

                    if (p_val == 2.0f) {
                        out_ptr[idx] = sycl::sqrt(acc);
                    } else if (p_val == 1.0f) {
                        out_ptr[idx] = acc;
                    } else {
                        out_ptr[idx] = sycl::pow(acc, 1.0f / p_val);
                    }
                }
            );
            queue.wait_and_throw();
            return output;
        }
        else if (input.dtype() == DType::Float64) {
            const double* in_ptr = get_data_ptr<const double>(input);
            Tensor output(out_shape, input.dtype(), input.device());
            double* out_ptr = get_data_ptr<double>(output);
            double p_val = static_cast<double>(p);

            queue.parallel_for<NormDimKernelFloat64>(
                sycl::range<1>(output_size), [=](sycl::id<1> gid) {
                    int64_t idx = gid[0];
                    int64_t outer = idx / inner_size;
                    int64_t inner = idx % inner_size;

                    double acc = 0.0;
                    for (int64_t d = 0; d < dim_size; ++d) {
                        double x = in_ptr[outer * dim_size * inner_size + d * inner_size + inner];
                        if (p_val == 1.0) {
                            acc += sycl::fabs(x);
                        } else if (p_val == 2.0) {
                            acc += x * x;
                        } else {
                            acc += sycl::pow(sycl::fabs(x), p_val);
                        }
                    }

                    if (p_val == 2.0) {
                        out_ptr[idx] = sycl::sqrt(acc);
                    } else if (p_val == 1.0) {
                        out_ptr[idx] = acc;
                    } else {
                        out_ptr[idx] = sycl::pow(acc, 1.0 / p_val);
                    }
                }
            );
            queue.wait_and_throw();
            return output;
        }
        else {
            throw std::runtime_error("Unsupported dtype for norm_kernel");
        }
    }
}

} // namespace oneapi
} // namespace tenzor
