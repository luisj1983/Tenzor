// Backend kernels need the view-creating mutation API (mutable_shape /
// mutable_strides). Match the CUDA backend, which defines this at the CMake
// level — here we scope the define to this translation unit.
#ifndef TENZOR_INTERNAL
#define TENZOR_INTERNAL
#endif
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/shape.hpp"
#include "oneapi_kernel_utils.hpp"
#include <sycl/sycl.hpp>
#include <cstring>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

// Forward declarations for kernels in other files
namespace tenzor {
namespace oneapi {
    auto repeat_kernel(const Tensor& input, const std::vector<int64_t>& repeats, sycl::queue& queue) -> Tensor;
    // Forward declaration for contiguous_kernel (defined later in this file)
    auto contiguous_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    // Forward declaration for fill_kernel (defined later in this file)
    auto fill_kernel(const Tensor& tensor, double value, sycl::queue& queue) -> Tensor;
}
}

namespace tenzor {
namespace oneapi {

// SYCL Kernel name classes
class TransposeKernelFloat32;
class TransposeKernelFloat64;
class TransposeKernelFloat16;
class TransposeKernelBFloat16;
class TransposeKernelUInt8;
class TransposeKernelBool;
class PermuteKernelFloat32;
class PermuteKernelFloat64;
class PermuteKernelFloat16;
class PermuteKernelBFloat16;
class PermuteKernelUInt8;
class PermuteKernelBool;
class OnesKernelBFloat16;
class FullKernelBFloat16;
class FillKernelFloat16;
class FillKernelBFloat16;


// Helper to compute flat index from multi-dimensional indices
inline auto compute_flat_index(const std::vector<int64_t>& indices,
                                const std::vector<int64_t>& strides) -> int64_t {
    int64_t flat_idx = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        flat_idx += indices[i] * strides[i];
    }
    return flat_idx;
}

// Reshape kernel — returns a view sharing storage when the input is
// contiguous, and materializes a contiguous copy otherwise. Previous
// implementation always memcpy'd to a new buffer, which broke alias
// detection (`may_alias(a, a.reshape(...))` returned false) and is the
// wrong semantics for reshape-as-a-view.
auto reshape_kernel(const Tensor& input, const std::vector<int64_t>& new_shape, sycl::queue& queue) -> Tensor {
    int64_t input_numel = input.numel();
    int64_t output_numel = 1;
    for (auto dim : new_shape) {
        output_numel *= dim;
    }

    if (input_numel != output_numel) {
        throw std::invalid_argument("Reshape: total number of elements must remain constant");
    }

    // Non-contiguous input: materialize then reshape the contiguous copy
    // (still returns a view of the newly-materialized storage).
    Tensor src = input.is_contiguous() ? input : contiguous_kernel(input, queue);

    // Create a view sharing storage: copy the TensorImpl (shares `storage`
    // intrusive_ptr, copies shape/strides by value), then overwrite shape
    // and strides for the new shape.
    Tensor result;
    TensorAccessor::get_impl_mutable(result) =
        make_intrusive<TensorImpl>(*TensorAccessor::get_impl(src));
    result.mutable_shape() = new_shape;
    result.mutable_strides() = compute_strides(new_shape);
    return result;
}

// Transpose kernel - swap two dimensions
auto transpose_kernel(const Tensor& input, int64_t dim0, int64_t dim1, sycl::queue& queue) -> Tensor {
    auto shape_span = input.shape();
    auto input_strides_span = input.strides();
    const size_t ndim = shape_span.size();

    // Handle negative dimensions
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;

    if (dim0 < 0 || dim0 >= static_cast<int64_t>(ndim) ||
        dim1 < 0 || dim1 >= static_cast<int64_t>(ndim)) {
        throw std::invalid_argument("Transpose: invalid dimensions");
    }

    if (ndim > 8) {
        throw std::invalid_argument("oneapi transpose: tensor rank > 8 is unsupported (on-device stride arrays are fixed at 8 dims)");
    }

    // Convert spans to vectors
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
    std::vector<int64_t> in_actual_strides(input_strides_span.begin(), input_strides_span.end());

    // Create output shape by swapping dimensions
    std::vector<int64_t> out_shape = shape;
    std::swap(out_shape[dim0], out_shape[dim1]);

    Tensor output(out_shape, input.dtype(), input.device());

    // Calculate output strides (always contiguous for output)
    auto out_strides = calculate_strides(out_shape);

    // For iterating, we need contiguous strides to decompose the flat index into coordinates
    auto iter_strides = calculate_strides(shape);

    const int64_t numel = input.numel();

    // Convert to device-copyable arrays
    int64_t shape_arr[8] = {0};
    int64_t in_actual_strides_arr[8] = {0};  // Actual input strides (may be non-contiguous)
    int64_t iter_strides_arr[8] = {0};        // For decomposing flat index into coordinates
    int64_t out_strides_arr[8] = {0};
    for (size_t i = 0; i < ndim && i < 8; ++i) {
        shape_arr[i] = shape[i];
        in_actual_strides_arr[i] = in_actual_strides[i];
        iter_strides_arr[i] = iter_strides[i];
        out_strides_arr[i] = out_strides[i];
    }

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<TransposeKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            // Decompose flat output index into coordinates using contiguous strides
            int64_t remaining = idx;
            int64_t in_idx = 0;
            int64_t out_idx = 0;

            for (size_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / iter_strides_arr[d];
                remaining %= iter_strides_arr[d];

                // Input index uses actual (possibly non-contiguous) strides
                in_idx += coord * in_actual_strides_arr[d];

                // Map coordinate to output dimension (swap dim0 and dim1)
                size_t out_d = (d == static_cast<size_t>(dim0)) ? dim1 :
                              (d == static_cast<size_t>(dim1)) ? dim0 : d;
                out_idx += coord * out_strides_arr[out_d];
            }

            out_ptr[out_idx] = in_ptr[in_idx];
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<TransposeKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int64_t remaining = idx;
            int64_t in_idx = 0;
            int64_t out_idx = 0;

            for (size_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / iter_strides_arr[d];
                remaining %= iter_strides_arr[d];

                in_idx += coord * in_actual_strides_arr[d];

                size_t out_d = (d == static_cast<size_t>(dim0)) ? dim1 :
                              (d == static_cast<size_t>(dim1)) ? dim0 : d;
                out_idx += coord * out_strides_arr[out_d];
            }

            out_ptr[out_idx] = in_ptr[in_idx];
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<TransposeKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int64_t remaining = idx;
            int64_t in_idx = 0;
            int64_t out_idx = 0;

            for (size_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / iter_strides_arr[d];
                remaining %= iter_strides_arr[d];

                in_idx += coord * in_actual_strides_arr[d];

                size_t out_d = (d == static_cast<size_t>(dim0)) ? dim1 :
                              (d == static_cast<size_t>(dim1)) ? dim0 : d;
                out_idx += coord * out_strides_arr[out_d];
            }

            out_ptr[out_idx] = in_ptr[in_idx];
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        // BFloat16 stored as uint16_t — pure copy, no conversion needed
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<TransposeKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int64_t remaining = idx;
            int64_t in_idx = 0;
            int64_t out_idx = 0;

            for (size_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / iter_strides_arr[d];
                remaining %= iter_strides_arr[d];

                in_idx += coord * in_actual_strides_arr[d];

                size_t out_d = (d == static_cast<size_t>(dim0)) ? dim1 :
                              (d == static_cast<size_t>(dim1)) ? dim0 : d;
                out_idx += coord * out_strides_arr[out_d];
            }

            out_ptr[out_idx] = in_ptr[in_idx];
        });
    }
    else if (input.dtype() == DType::UInt8) {
        const uint8_t* in_ptr = get_data_ptr<const uint8_t>(input);
        uint8_t* out_ptr = get_data_ptr<uint8_t>(output);

        queue.parallel_for<TransposeKernelUInt8>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int64_t remaining = idx;
            int64_t in_idx = 0;
            int64_t out_idx = 0;

            for (size_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / iter_strides_arr[d];
                remaining %= iter_strides_arr[d];

                in_idx += coord * in_actual_strides_arr[d];

                size_t out_d = (d == static_cast<size_t>(dim0)) ? dim1 :
                              (d == static_cast<size_t>(dim1)) ? dim0 : d;
                out_idx += coord * out_strides_arr[out_d];
            }

            out_ptr[out_idx] = in_ptr[in_idx];
        });
    }
    else if (input.dtype() == DType::Bool) {
        const bool* in_ptr = get_data_ptr<const bool>(input);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<TransposeKernelBool>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int64_t remaining = idx;
            int64_t in_idx = 0;
            int64_t out_idx = 0;

            for (size_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / iter_strides_arr[d];
                remaining %= iter_strides_arr[d];

                in_idx += coord * in_actual_strides_arr[d];

                size_t out_d = (d == static_cast<size_t>(dim0)) ? dim1 :
                              (d == static_cast<size_t>(dim1)) ? dim0 : d;
                out_idx += coord * out_strides_arr[out_d];
            }

            out_ptr[out_idx] = in_ptr[in_idx];
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for transpose");
    }

    return output;
}

// Permute kernel - reorder dimensions
// IMPORTANT: Must handle non-contiguous inputs correctly by using actual strides
auto permute_kernel(const Tensor& input, const std::vector<int64_t>& dims, sycl::queue& queue) -> Tensor {
    auto shape_span = input.shape();
    auto input_strides_span = input.strides();
    const size_t ndim = shape_span.size();

    if (dims.size() != ndim) {
        throw std::invalid_argument("Permute: number of dimensions must match");
    }

    // Convert spans to vectors
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
    std::vector<int64_t> in_actual_strides(input_strides_span.begin(), input_strides_span.end());

    // Validate and handle negative dimensions
    std::vector<int64_t> perm_dims = dims;
    for (auto& d : perm_dims) {
        if (d < 0) d += ndim;
        if (d < 0 || d >= static_cast<int64_t>(ndim)) {
            throw std::invalid_argument("Permute: invalid dimension");
        }
    }

    // Create output shape
    std::vector<int64_t> out_shape(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        out_shape[i] = shape[perm_dims[i]];
    }

    Tensor output(out_shape, input.dtype(), input.device());

    // Calculate contiguous iteration strides for coordinate decomposition
    auto iter_strides = calculate_strides(shape);
    // Output is always contiguous
    auto out_strides = calculate_strides(out_shape);

    const int64_t numel = input.numel();

    // Convert vectors to device-copyable arrays
    int64_t iter_strides_arr[8] = {0};      // For decomposing flat index into coordinates
    int64_t in_actual_strides_arr[8] = {0}; // Actual input strides (may be non-contiguous)
    int64_t out_strides_arr[8] = {0};
    int64_t perm_dims_arr[8] = {0};
    for (size_t i = 0; i < ndim && i < 8; ++i) {
        iter_strides_arr[i] = iter_strides[i];
        in_actual_strides_arr[i] = in_actual_strides[i];
        out_strides_arr[i] = out_strides[i];
        perm_dims_arr[i] = perm_dims[i];
    }

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<PermuteKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            // Compute multi-dimensional coordinates from flat index using iteration strides
            int64_t coords[8];
            int64_t temp = flat_idx;
            for (size_t d = 0; d < ndim; ++d) {
                coords[d] = temp / iter_strides_arr[d];
                temp %= iter_strides_arr[d];
            }

            // Compute input index using ACTUAL strides (may be non-contiguous)
            int64_t in_idx = 0;
            for (size_t d = 0; d < ndim; ++d) {
                in_idx += coords[d] * in_actual_strides_arr[d];
            }

            // Compute output index with permuted dimensions
            int64_t out_idx = 0;
            for (size_t d = 0; d < ndim; ++d) {
                out_idx += coords[perm_dims_arr[d]] * out_strides_arr[d];
            }

            out_ptr[out_idx] = in_ptr[in_idx];
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<PermuteKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t coords[8];
            int64_t temp = flat_idx;
            for (size_t d = 0; d < ndim; ++d) {
                coords[d] = temp / iter_strides_arr[d];
                temp %= iter_strides_arr[d];
            }

            int64_t in_idx = 0;
            for (size_t d = 0; d < ndim; ++d) {
                in_idx += coords[d] * in_actual_strides_arr[d];
            }

            int64_t out_idx = 0;
            for (size_t d = 0; d < ndim; ++d) {
                out_idx += coords[perm_dims_arr[d]] * out_strides_arr[d];
            }

            out_ptr[out_idx] = in_ptr[in_idx];
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<PermuteKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t coords[8];
            int64_t temp = flat_idx;
            for (size_t d = 0; d < ndim; ++d) {
                coords[d] = temp / iter_strides_arr[d];
                temp %= iter_strides_arr[d];
            }

            int64_t in_idx = 0;
            for (size_t d = 0; d < ndim; ++d) {
                in_idx += coords[d] * in_actual_strides_arr[d];
            }

            int64_t out_idx = 0;
            for (size_t d = 0; d < ndim; ++d) {
                out_idx += coords[perm_dims_arr[d]] * out_strides_arr[d];
            }

            out_ptr[out_idx] = in_ptr[in_idx];
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        // BFloat16 stored as uint16_t — pure copy, no conversion needed
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<PermuteKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t coords[8];
            int64_t temp = flat_idx;
            for (size_t d = 0; d < ndim; ++d) {
                coords[d] = temp / iter_strides_arr[d];
                temp %= iter_strides_arr[d];
            }

            int64_t in_idx = 0;
            for (size_t d = 0; d < ndim; ++d) {
                in_idx += coords[d] * in_actual_strides_arr[d];
            }

            int64_t out_idx = 0;
            for (size_t d = 0; d < ndim; ++d) {
                out_idx += coords[perm_dims_arr[d]] * out_strides_arr[d];
            }

            out_ptr[out_idx] = in_ptr[in_idx];
        });
    }
    else if (input.dtype() == DType::UInt8) {
        const uint8_t* in_ptr = get_data_ptr<const uint8_t>(input);
        uint8_t* out_ptr = get_data_ptr<uint8_t>(output);

        queue.parallel_for<PermuteKernelUInt8>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t coords[8];
            int64_t temp = flat_idx;
            for (size_t d = 0; d < ndim; ++d) {
                coords[d] = temp / iter_strides_arr[d];
                temp %= iter_strides_arr[d];
            }

            int64_t in_idx = 0;
            for (size_t d = 0; d < ndim; ++d) {
                in_idx += coords[d] * in_actual_strides_arr[d];
            }

            int64_t out_idx = 0;
            for (size_t d = 0; d < ndim; ++d) {
                out_idx += coords[perm_dims_arr[d]] * out_strides_arr[d];
            }

            out_ptr[out_idx] = in_ptr[in_idx];
        });
    }
    else if (input.dtype() == DType::Bool) {
        const bool* in_ptr = get_data_ptr<const bool>(input);
        bool* out_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<PermuteKernelBool>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t coords[8];
            int64_t temp = flat_idx;
            for (size_t d = 0; d < ndim; ++d) {
                coords[d] = temp / iter_strides_arr[d];
                temp %= iter_strides_arr[d];
            }

            int64_t in_idx = 0;
            for (size_t d = 0; d < ndim; ++d) {
                in_idx += coords[d] * in_actual_strides_arr[d];
            }

            int64_t out_idx = 0;
            for (size_t d = 0; d < ndim; ++d) {
                out_idx += coords[perm_dims_arr[d]] * out_strides_arr[d];
            }

            out_ptr[out_idx] = in_ptr[in_idx];
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for permute");
    }

    return output;
}

// Squeeze kernel - remove dimensions of size 1
auto squeeze_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> Tensor {
    auto shape = input.shape();

    std::vector<int64_t> out_shape;

    if (dim == -1) {
        // Squeeze all dimensions of size 1
        for (auto s : shape) {
            if (s != 1) {
                out_shape.push_back(s);
            }
        }
    } else {
        // Squeeze specific dimension
        if (dim < 0) dim += shape.size();
        if (dim < 0 || dim >= static_cast<int64_t>(shape.size())) {
            throw std::invalid_argument("Squeeze: invalid dimension");
        }

        if (shape[dim] != 1) {
            throw std::invalid_argument("Squeeze: dimension must be size 1");
        }

        for (size_t i = 0; i < shape.size(); ++i) {
            if (static_cast<int64_t>(i) != dim) {
                out_shape.push_back(shape[i]);
            }
        }
    }

    if (out_shape.empty()) {
        out_shape.push_back(1);
    }

    // Squeeze is just a view change, copy data - works for all dtypes
    Tensor output(out_shape, input.dtype(), input.device());
    const size_t bytes = input.numel() * input.dtype_size();
    const void* in_ptr = input.data_ptr();
    void* out_ptr = const_cast<void*>(output.data_ptr());
    queue.memcpy(out_ptr, in_ptr, bytes);

    return output;
}

// Unsqueeze kernel - add dimension of size 1
auto unsqueeze_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> Tensor {
    auto shape = input.shape();
    const size_t ndim = shape.size();

    // Handle negative dimension
    if (dim < 0) dim += ndim + 1;

    if (dim < 0 || dim > static_cast<int64_t>(ndim)) {
        throw std::invalid_argument("Unsqueeze: invalid dimension");
    }

    // Create output shape with new dimension
    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < ndim; ++i) {
        if (static_cast<int64_t>(i) == dim) {
            out_shape.push_back(1);
        }
        out_shape.push_back(shape[i]);
    }

    if (dim == static_cast<int64_t>(ndim)) {
        out_shape.push_back(1);
    }

    // Unsqueeze is just a view change, copy data - works for all dtypes
    Tensor output(out_shape, input.dtype(), input.device());
    const size_t bytes = input.numel() * input.dtype_size();
    const void* in_ptr = input.data_ptr();
    void* out_ptr = const_cast<void*>(output.data_ptr());
    queue.memcpy(out_ptr, in_ptr, bytes);

    return output;
}

// Contiguous kernel - ensure tensor data is laid out contiguously
// IMPORTANT: Must handle non-contiguous tensors correctly by respecting strides
auto contiguous_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    // If already contiguous, just clone
    if (input.is_contiguous()) {
        Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                      input.dtype(), input.device());
        const size_t bytes = input.numel() * input.dtype_size();
        const void* in_ptr = input.data_ptr();
        void* out_ptr = const_cast<void*>(output.data_ptr());
        queue.memcpy(out_ptr, in_ptr, bytes);
        return output;
    }

    // Non-contiguous tensor: need to respect strides
    auto shape_span = input.shape();
    auto strides_span = input.strides();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
    std::vector<int64_t> in_strides(strides_span.begin(), strides_span.end());

    Tensor output(shape, input.dtype(), input.device());
    const int64_t numel = output.numel();
    const size_t ndim = shape.size();

    // Calculate output strides (contiguous layout)
    std::vector<int64_t> out_strides(ndim);
    int64_t stride = 1;
    for (int64_t i = ndim - 1; i >= 0; --i) {
        out_strides[i] = stride;
        stride *= shape[i];
    }

    // Convert to device-copyable arrays
    int64_t shape_arr[8] = {0};
    int64_t in_strides_arr[8] = {0};
    int64_t out_strides_arr[8] = {0};
    for (size_t i = 0; i < ndim && i < 8; ++i) {
        shape_arr[i] = shape[i];
        in_strides_arr[i] = in_strides[i];
        out_strides_arr[i] = out_strides[i];
    }

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            // Convert flat output index to multi-dimensional indices
            int64_t remaining = flat_idx;
            int64_t in_idx = 0;

            for (size_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / out_strides_arr[d];
                remaining %= out_strides_arr[d];
                in_idx += coord * in_strides_arr[d];
            }

            out_ptr[flat_idx] = in_ptr[in_idx];
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t remaining = flat_idx;
            int64_t in_idx = 0;

            for (size_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / out_strides_arr[d];
                remaining %= out_strides_arr[d];
                in_idx += coord * in_strides_arr[d];
            }

            out_ptr[flat_idx] = in_ptr[in_idx];
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t remaining = flat_idx;
            int64_t in_idx = 0;

            for (size_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / out_strides_arr[d];
                remaining %= out_strides_arr[d];
                in_idx += coord * in_strides_arr[d];
            }

            out_ptr[flat_idx] = in_ptr[in_idx];
        });
    }
    else if (input.dtype() == DType::Int32) {
        const int32_t* in_ptr = get_data_ptr<const int32_t>(input);
        int32_t* out_ptr = get_data_ptr<int32_t>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t remaining = flat_idx;
            int64_t in_idx = 0;

            for (size_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / out_strides_arr[d];
                remaining %= out_strides_arr[d];
                in_idx += coord * in_strides_arr[d];
            }

            out_ptr[flat_idx] = in_ptr[in_idx];
        });
    }
    else if (input.dtype() == DType::Int64) {
        const int64_t* in_ptr = get_data_ptr<const int64_t>(input);
        int64_t* out_ptr = get_data_ptr<int64_t>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t remaining = flat_idx;
            int64_t in_idx = 0;

            for (size_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / out_strides_arr[d];
                remaining %= out_strides_arr[d];
                in_idx += coord * in_strides_arr[d];
            }

            out_ptr[flat_idx] = in_ptr[in_idx];
        });
    }
    else if (input.dtype() == DType::BFloat16 || input.dtype() == DType::Int16 || input.dtype() == DType::UInt16) {
        // All 2-byte types: use uint16_t for raw copy
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t remaining = flat_idx;
            int64_t in_idx = 0;

            for (size_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / out_strides_arr[d];
                remaining %= out_strides_arr[d];
                in_idx += coord * in_strides_arr[d];
            }

            out_ptr[flat_idx] = in_ptr[in_idx];
        });
    }
    else if (input.dtype() == DType::UInt8 || input.dtype() == DType::Int8 || input.dtype() == DType::Bool) {
        // All 1-byte types: use uint8_t for raw copy
        const uint8_t* in_ptr = get_data_ptr<const uint8_t>(input);
        uint8_t* out_ptr = get_data_ptr<uint8_t>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t remaining = flat_idx;
            int64_t in_idx = 0;

            for (size_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / out_strides_arr[d];
                remaining %= out_strides_arr[d];
                in_idx += coord * in_strides_arr[d];
            }

            out_ptr[flat_idx] = in_ptr[in_idx];
        });
    }
    else if (input.dtype() == DType::Complex64) {
        // 8-byte complex: treat as 2 floats but use a 64-bit raw integer for the copy
        const uint64_t* in_ptr = get_data_ptr<const uint64_t>(input);
        uint64_t* out_ptr = get_data_ptr<uint64_t>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t remaining = flat_idx;
            int64_t in_idx = 0;
            for (size_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / out_strides_arr[d];
                remaining %= out_strides_arr[d];
                in_idx += coord * in_strides_arr[d];
            }
            out_ptr[flat_idx] = in_ptr[in_idx];
        });
    }
    else if (input.dtype() == DType::Complex128) {
        // 16-byte complex: copy in two halves via uint64_t pair
        const uint64_t* in_ptr = get_data_ptr<const uint64_t>(input);
        uint64_t* out_ptr = get_data_ptr<uint64_t>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t remaining = flat_idx;
            int64_t in_idx = 0;
            for (size_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / out_strides_arr[d];
                remaining %= out_strides_arr[d];
                in_idx += coord * in_strides_arr[d];
            }
            // 2 uint64s per element
            out_ptr[flat_idx * 2 + 0] = in_ptr[in_idx * 2 + 0];
            out_ptr[flat_idx * 2 + 1] = in_ptr[in_idx * 2 + 1];
        });
    }
    else if (input.dtype() != DType::QInt4x2 &&
             (input.dtype_size() == 1 || input.dtype_size() == 2 ||
              input.dtype_size() == 4 || input.dtype_size() == 8)) {
        // Generic gather for any remaining fixed-width POD dtype (UInt32/UInt64,
        // FP8, etc.): the copy is pure byte movement keyed by element width.
        auto copy_w = [&]<typename U>() {
            const U* in_ptr = get_data_ptr<const U>(input);
            U* out_ptr = get_data_ptr<U>(output);
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
                int64_t remaining = flat_idx, in_idx = 0;
                for (size_t d = 0; d < ndim; ++d) {
                    int64_t coord = remaining / out_strides_arr[d];
                    remaining %= out_strides_arr[d];
                    in_idx += coord * in_strides_arr[d];
                }
                out_ptr[flat_idx] = in_ptr[in_idx];
            });
        };
        switch (input.dtype_size()) {
            case 1: copy_w.template operator()<uint8_t>();  break;
            case 2: copy_w.template operator()<uint16_t>(); break;
            case 4: copy_w.template operator()<uint32_t>(); break;
            case 8: copy_w.template operator()<uint64_t>(); break;
        }
    }
    else {
        throw std::runtime_error("Unsupported dtype for contiguous kernel");
    }

    return output;
}

// Clone kernel - create a copy of the tensor
auto clone_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    // Works for all dtypes
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const size_t bytes = input.numel() * input.dtype_size();
    const void* in_ptr = input.data_ptr();
    void* out_ptr = const_cast<void*>(output.data_ptr());
    queue.memcpy(out_ptr, in_ptr, bytes);

    return output;
}

// Fill operations

// Zeros kernel
auto zeros_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, sycl::queue& queue) -> Tensor {
    Tensor output(shape, dtype, device);
    const size_t bytes = output.numel() * output.dtype_size();

    void* ptr = const_cast<void*>(output.data_ptr());
    queue.memset(ptr, 0, bytes);

    return output;
}

// Ones kernel - use simpler memcpy approach for compatibility
auto ones_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, sycl::queue& queue) -> Tensor {
    Tensor output(shape, dtype, device);
    const int64_t numel = output.numel();

    if (dtype == DType::Float32) {
        queue.fill(get_data_ptr<float>(output), 1.0f, numel).wait();
    }
    else if (dtype == DType::Float64) {
        queue.fill(get_data_ptr<double>(output), 1.0, numel).wait();
    }
    else if (dtype == DType::Float16) {
        sycl::half* device_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            device_ptr[i] = sycl::half(1.0f);
        }).wait();
    }
    else if (dtype == DType::BFloat16) {
        const uint16_t one_bf16 = f32_to_bf16(1.0f);
        uint16_t* device_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<OnesKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            device_ptr[i] = one_bf16;
        }).wait();
    }
    else if (dtype == DType::Int32) {
        queue.fill(get_data_ptr<int32_t>(output), static_cast<int32_t>(1), numel).wait();
    }
    else if (dtype == DType::Int64) {
        queue.fill(get_data_ptr<int64_t>(output), static_cast<int64_t>(1), numel).wait();
    }
    else if (dtype == DType::Int8) {
        queue.fill(get_data_ptr<int8_t>(output), static_cast<int8_t>(1), numel).wait();
    }
    else if (dtype == DType::Int16) {
        queue.fill(get_data_ptr<int16_t>(output), static_cast<int16_t>(1), numel).wait();
    }
    else if (dtype == DType::UInt8) {
        queue.fill(get_data_ptr<uint8_t>(output), static_cast<uint8_t>(1), numel).wait();
    }
    else if (dtype == DType::UInt16) {
        queue.fill(get_data_ptr<uint16_t>(output), static_cast<uint16_t>(1), numel).wait();
    }
    else if (dtype == DType::UInt32) {
        queue.fill(get_data_ptr<uint32_t>(output), static_cast<uint32_t>(1), numel).wait();
    }
    else if (dtype == DType::UInt64) {
        queue.fill(get_data_ptr<uint64_t>(output), static_cast<uint64_t>(1), numel).wait();
    }
    else if (dtype == DType::Bool) {
        queue.fill(get_data_ptr<uint8_t>(output), static_cast<uint8_t>(1), numel).wait();
    }
    else {
        // Complex/FP8/quantized: defer to fill_kernel(value=1), which handles
        // those representations (and throws for quantized dtypes that lack
        // quantization params, matching the CPU backend's "ones on a quantized
        // dtype needs params" contract).
        return fill_kernel(output, 1.0, queue);
    }

    return output;
}

// Full kernel - fill with specific value using memcpy
auto full_kernel(const std::vector<int64_t>& shape, double value, DType dtype, Device device, sycl::queue& queue) -> Tensor {
    Tensor output(shape, dtype, device);
    const int64_t numel = output.numel();

    if (dtype == DType::Float32) {
        float val = static_cast<float>(value);
        float* device_ptr = get_data_ptr<float>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            device_ptr[i] = val;
        }).wait();
    }
    else if (dtype == DType::Float64) {
        // Pass through the double value unchanged — Float64 subnormals
        // (~5e-324) would be lost if we narrowed through float first.
        double val = value;
        double* device_ptr = get_data_ptr<double>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            device_ptr[i] = val;
        }).wait();
    }
    else if (dtype == DType::Float16) {
        const sycl::half value_h(static_cast<float>(value));
        sycl::half* device_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            device_ptr[i] = value_h;
        }).wait();
    }
    else if (dtype == DType::BFloat16) {
        const uint16_t value_bf16 = f32_to_bf16(static_cast<float>(value));
        uint16_t* device_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<FullKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            device_ptr[i] = value_bf16;
        }).wait();
    }
    else if (dtype == DType::Int32) {
        int32_t val = static_cast<int32_t>(value);
        int32_t* device_ptr = get_data_ptr<int32_t>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            device_ptr[i] = val;
        }).wait();
    }
    else if (dtype == DType::Int64) {
        int64_t val = static_cast<int64_t>(value);
        int64_t* device_ptr = get_data_ptr<int64_t>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            device_ptr[i] = val;
        }).wait();
    }
    else if (dtype == DType::Int8) {
        int8_t val = static_cast<int8_t>(value);
        int8_t* device_ptr = get_data_ptr<int8_t>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            device_ptr[i] = val;
        }).wait();
    }
    else if (dtype == DType::Int16) {
        int16_t val = static_cast<int16_t>(value);
        int16_t* device_ptr = get_data_ptr<int16_t>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            device_ptr[i] = val;
        }).wait();
    }
    else if (dtype == DType::UInt8) {
        uint8_t val = static_cast<uint8_t>(value);
        uint8_t* device_ptr = get_data_ptr<uint8_t>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            device_ptr[i] = val;
        }).wait();
    }
    else if (dtype == DType::UInt16) {
        uint16_t val = static_cast<uint16_t>(value);
        uint16_t* device_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            device_ptr[i] = val;
        }).wait();
    }
    else if (dtype == DType::UInt32) {
        uint32_t val = static_cast<uint32_t>(value);
        uint32_t* device_ptr = get_data_ptr<uint32_t>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            device_ptr[i] = val;
        }).wait();
    }
    else if (dtype == DType::UInt64) {
        uint64_t val = static_cast<uint64_t>(value);
        uint64_t* device_ptr = get_data_ptr<uint64_t>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            device_ptr[i] = val;
        }).wait();
    }
    else if (dtype == DType::Bool) {
        uint8_t val = (value != 0.0f) ? 1 : 0;
        uint8_t* device_ptr = get_data_ptr<uint8_t>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            device_ptr[i] = val;
        }).wait();
    }
    else {
        // Complex/FP8/quantized: defer to fill_kernel.
        return fill_kernel(output, value, queue);
    }

    return output;
}

// Fill kernel - fill output tensor with `value` entirely on device.
// Earlier versions built a host std::vector<T>(numel, value) and uploaded it
// via queue.memcpy — that's host compute followed by an H2D transfer, i.e.
// silent CPU work. SYCL's queue.fill() does the same thing as a device-side
// parallel_for and avoids the host loop / staging buffer.
auto fill_kernel(const Tensor& tensor, double value, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(tensor.shape().begin(), tensor.shape().end()),
                  tensor.dtype(), tensor.device());

    const int64_t numel = tensor.numel();
    const size_t count = static_cast<size_t>(numel);

    auto fill_simple = [&]<typename T>(T val) {
        T* device_ptr = get_data_ptr<T>(output);
        queue.fill(device_ptr, val, count).wait();
    };

    if (tensor.dtype() == DType::Float32) {
        fill_simple(static_cast<float>(value));
    }
    else if (tensor.dtype() == DType::Float64) {
        // Pass the double directly — Float64 subnormals would not survive
        // a float intermediate.
        fill_simple(value);
    }
    else if (tensor.dtype() == DType::Float16) {
        const sycl::half value_h(static_cast<float>(value));
        sycl::half* device_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<FillKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            device_ptr[i] = value_h;
        }).wait();
    }
    else if (tensor.dtype() == DType::BFloat16) {
        // BFloat16 is stored as raw uint16_t; fill the bit pattern directly.
        fill_simple(f32_to_bf16(static_cast<float>(value)));
    }
    else if (tensor.dtype() == DType::Int32) {
        fill_simple(static_cast<int32_t>(value));
    }
    else if (tensor.dtype() == DType::Int64) {
        fill_simple(static_cast<int64_t>(value));
    }
    else if (tensor.dtype() == DType::Int8) {
        fill_simple(static_cast<int8_t>(value));
    }
    else if (tensor.dtype() == DType::Int16) {
        fill_simple(static_cast<int16_t>(value));
    }
    else if (tensor.dtype() == DType::UInt8) {
        fill_simple(static_cast<uint8_t>(value));
    }
    else if (tensor.dtype() == DType::UInt16) {
        fill_simple(static_cast<uint16_t>(value));
    }
    else if (tensor.dtype() == DType::UInt32) {
        fill_simple(static_cast<uint32_t>(value));
    }
    else if (tensor.dtype() == DType::UInt64) {
        fill_simple(static_cast<uint64_t>(value));
    }
    else if (tensor.dtype() == DType::Bool) {
        bool* p = get_data_ptr<bool>(output);
        queue.fill(p, (value != 0.0), count).wait();
    }
    else if (tensor.dtype() == DType::Complex64) {
        float* p = get_data_ptr<float>(output);
        const float re = static_cast<float>(value);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            p[2 * i[0]] = re; p[2 * i[0] + 1] = 0.0f;
        }).wait();
    }
    else if (tensor.dtype() == DType::Complex128) {
        double* p = get_data_ptr<double>(output);
        const double re = value;
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            p[2 * i[0]] = re; p[2 * i[0] + 1] = 0.0;
        }).wait();
    }
    else if (tensor.dtype() == DType::FP8_E4M3) {
        FP8_E4M3 v(static_cast<float>(value));
        uint8_t byte; std::memcpy(&byte, &v, sizeof(uint8_t));
        queue.fill(get_data_ptr<uint8_t>(output), byte, count).wait();
    }
    else if (tensor.dtype() == DType::FP8_E5M2) {
        FP8_E5M2 v(static_cast<float>(value));
        uint8_t byte; std::memcpy(&byte, &v, sizeof(uint8_t));
        queue.fill(get_data_ptr<uint8_t>(output), byte, count).wait();
    }
    else if (tensor.dtype() == DType::FP8_E4M3FNUZ) {
        FP8_E4M3FNUZ v(static_cast<float>(value));
        uint8_t byte; std::memcpy(&byte, &v, sizeof(uint8_t));
        queue.fill(get_data_ptr<uint8_t>(output), byte, count).wait();
    }
    else if (tensor.dtype() == DType::FP8_E5M2FNUZ) {
        FP8_E5M2FNUZ v(static_cast<float>(value));
        uint8_t byte; std::memcpy(&byte, &v, sizeof(uint8_t));
        queue.fill(get_data_ptr<uint8_t>(output), byte, count).wait();
    }
    else if (tensor.dtype() == DType::QInt8 || tensor.dtype() == DType::QUInt8 ||
             tensor.dtype() == DType::QInt4x2) {
        if (tensor.q_scale() == 0.0) {
            throw std::runtime_error(
                "fill_kernel: fill on quantized tensor requires quantization params: "
                "call set_quantization_params(scale, zero_point) first");
        }
        const int64_t qval = static_cast<int64_t>(std::llround(value / tensor.q_scale()))
                             + tensor.q_zero_point();
        uint8_t byte;
        if (tensor.dtype() == DType::QInt8) {
            byte = static_cast<uint8_t>(static_cast<int8_t>(
                std::clamp<int64_t>(qval, -128, 127)));
        } else if (tensor.dtype() == DType::QUInt8) {
            byte = static_cast<uint8_t>(std::clamp<int64_t>(qval, 0, 255));
        } else { // QInt4x2: pack identical 4-bit nibbles
            const int64_t c = std::clamp<int64_t>(qval, -8, 7);
            byte = static_cast<uint8_t>((c & 0xF) | ((c & 0xF) << 4));
        }
        queue.fill(get_data_ptr<uint8_t>(output), byte, count).wait();
        output.set_quantization_params(tensor.q_scale(), tensor.q_zero_point());
    }
    else {
        throw std::runtime_error("Unsupported dtype for fill");
    }

    return output;
}

// ============================================================================
// Flatten kernel - reshape dims [start_dim..end_dim] into a single dimension
// ============================================================================

auto flatten_kernel(const Tensor& input, int64_t start_dim, int64_t end_dim, sycl::queue& queue) -> Tensor {
    auto shape = input.shape();
    int64_t ndim = shape.size();

    if (start_dim < 0) start_dim += ndim;
    if (end_dim < 0) end_dim += ndim;

    int64_t flat_size = 1;
    for (int64_t d = start_dim; d <= end_dim; ++d) {
        flat_size *= shape[d];
    }

    std::vector<int64_t> new_shape;
    for (int64_t d = 0; d < start_dim; ++d) new_shape.push_back(shape[d]);
    new_shape.push_back(flat_size);
    for (int64_t d = end_dim + 1; d < ndim; ++d) new_shape.push_back(shape[d]);

    return reshape_kernel(input, new_shape, queue);
}

// ============================================================================
// Slice kernel - extract a slice along dimensions
// ============================================================================

class SliceKernelCopy;

auto slice_kernel(const Tensor& input,
                  const std::vector<int64_t>& starts,
                  const std::vector<int64_t>& ends,
                  const std::vector<int64_t>& steps, sycl::queue& queue) -> Tensor {
    // Apply slice iteratively along each dimension
    Tensor result = input;
    auto shape = input.shape();
    int64_t ndim = shape.size();

    for (size_t d = 0; d < starts.size() && d < static_cast<size_t>(ndim); ++d) {
        auto cur_shape = result.shape();
        int64_t dim_size = cur_shape[d];
        int64_t start = starts[d];
        int64_t end = ends[d];
        int64_t step = steps[d];

        if (start < 0) start += dim_size;
        if (end < 0) end += dim_size;
        start = std::max(int64_t(0), std::min(start, dim_size));
        end = std::max(int64_t(0), std::min(end, dim_size));

        int64_t slice_size = (end - start + step - 1) / step;
        if (slice_size < 0) slice_size = 0;

        if (slice_size == dim_size && step == 1 && start == 0) continue;  // No-op for this dim

        std::vector<int64_t> new_shape(cur_shape.begin(), cur_shape.end());
        new_shape[d] = slice_size;

        int64_t outer_size = 1;
        for (int64_t i = 0; i < static_cast<int64_t>(d); ++i) outer_size *= cur_shape[i];
        int64_t inner_size = 1;
        for (int64_t i = d + 1; i < static_cast<int64_t>(cur_shape.size()); ++i) inner_size *= cur_shape[i];

        Tensor next(new_shape, result.dtype(), result.device());
        size_t elem_size = result.dtype_size();

        // Ensure contiguous for memcpy
        Tensor cont = result.is_contiguous() ? result : contiguous_kernel(result, queue);

        if (step == 1 && elem_size <= 8) {
            // Bulk copy per outer iteration
            size_t chunk_bytes = slice_size * inner_size * elem_size;
            const uint8_t* src = static_cast<const uint8_t*>(cont.data_ptr());
            uint8_t* dst = static_cast<uint8_t*>(const_cast<void*>(next.data_ptr()));
            for (int64_t o = 0; o < outer_size; ++o) {
                size_t src_off = (o * dim_size + start) * inner_size * elem_size;
                size_t dst_off = o * slice_size * inner_size * elem_size;
                queue.memcpy(dst + dst_off, src + src_off, chunk_bytes);
            }
        } else {
            // General strided copy via multiple device memcpy commands
            const uint8_t* src = static_cast<const uint8_t*>(cont.data_ptr());
            uint8_t* dst = static_cast<uint8_t*>(const_cast<void*>(next.data_ptr()));

            int64_t dst_idx = 0;
            for (int64_t o = 0; o < outer_size; ++o) {
                for (int64_t s = start; s < end; s += step) {
                    size_t src_off = (o * dim_size + s) * inner_size * elem_size;
                    size_t dst_off = dst_idx * inner_size * elem_size;
                    queue.memcpy(dst + dst_off, src + src_off, inner_size * elem_size);
                    dst_idx += 1;
                }
            }
        }

        result = next;
    }

    return result;
}

// ============================================================================
// Split kernel - split tensor into chunks along a dimension
// ============================================================================

auto split_kernel(const Tensor& input, int64_t split_size, int64_t dim, sycl::queue& queue) -> std::vector<Tensor> {
    auto shape = input.shape();
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;

    int64_t dim_size = shape[dim];
    Tensor cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);

    int64_t outer_size = 1;
    for (int64_t d = 0; d < dim; ++d) outer_size *= shape[d];
    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) inner_size *= shape[d];

    size_t elem_size = input.dtype_size();
    const uint8_t* src = static_cast<const uint8_t*>(cont.data_ptr());

    std::vector<Tensor> result;
    for (int64_t offset = 0; offset < dim_size; offset += split_size) {
        int64_t chunk_size = std::min(split_size, dim_size - offset);
        std::vector<int64_t> out_shape(shape.begin(), shape.end());
        out_shape[dim] = chunk_size;

        Tensor chunk(out_shape, input.dtype(), input.device());
        uint8_t* dst = static_cast<uint8_t*>(const_cast<void*>(chunk.data_ptr()));

        for (int64_t o = 0; o < outer_size; ++o) {
            size_t src_off = (o * dim_size + offset) * inner_size * elem_size;
            size_t dst_off = o * chunk_size * inner_size * elem_size;
            queue.memcpy(dst + dst_off, src + src_off, chunk_size * inner_size * elem_size);
        }

        result.push_back(std::move(chunk));
    }

    return result;
}

// ============================================================================
// Chunk kernel - split into n chunks
// ============================================================================

auto chunk_kernel(const Tensor& input, int64_t chunks, int64_t dim, sycl::queue& queue) -> std::vector<Tensor> {
    auto shape = input.shape();
    if (dim < 0) dim += shape.size();
    int64_t dim_size = shape[dim];
    int64_t split_size = (dim_size + chunks - 1) / chunks;
    return split_kernel(input, split_size, dim, queue);
}

// ============================================================================
// Tile kernel - repeat tensor along each dimension
// ============================================================================

auto tile_kernel(const Tensor& input, const std::vector<int64_t>& reps, sycl::queue& queue) -> Tensor {
    // tile (numpy.tile / torch.tile) is BLOCK replication: tile([1,2,3],{2}) =
    // [1,2,3,1,2,3]. It is NOT repeat() — Tenzor's repeat is element-wise
    // interleave (repeat([1,2,3],{2}) = [1,1,2,2,3,3]). Delegating to
    // repeat_kernel produced the interleaved (wrong) result on oneAPI. Implement
    // proper per-dimension block tiling with right-aligned reps (numpy semantics).
    auto in_shape = input.shape();
    const int64_t ndim = static_cast<int64_t>(in_shape.size());
    const int64_t out_ndim = std::max(ndim, static_cast<int64_t>(reps.size()));

    std::vector<int64_t> cur(out_ndim, 1), rps(out_ndim, 1);
    for (int64_t i = 0; i < ndim; ++i) cur[out_ndim - ndim + i] = in_shape[i];
    for (size_t i = 0; i < reps.size(); ++i)
        rps[out_ndim - static_cast<int64_t>(reps.size()) + i] = reps[i];

    const size_t elem = input.dtype_size();
    Tensor result = input.is_contiguous() ? input : contiguous_kernel(input, queue);

    for (int64_t d = out_ndim - 1; d >= 0; --d) {
        if (rps[d] == 1) continue;
        const int64_t D = cur[d];
        const int64_t R = rps[d];
        int64_t outer = 1; for (int64_t i = 0; i < d; ++i) outer *= cur[i];
        int64_t inner = 1; for (int64_t i = d + 1; i < out_ndim; ++i) inner *= cur[i];

        std::vector<int64_t> new_shape(cur);
        new_shape[d] = D * R;
        Tensor next(new_shape, input.dtype(), input.device());

        Tensor cont = result.is_contiguous() ? result : contiguous_kernel(result, queue);
        const uint8_t* src = static_cast<const uint8_t*>(cont.data_ptr());
        uint8_t* dst = static_cast<uint8_t*>(const_cast<void*>(next.data_ptr()));
        const size_t block = static_cast<size_t>(D) * inner * elem;  // one full dim-d block

        for (int64_t o = 0; o < outer; ++o) {
            for (int64_t r = 0; r < R; ++r) {
                queue.memcpy(dst + (static_cast<size_t>(o) * R + r) * block,
                             src + static_cast<size_t>(o) * block, block);
            }
        }
        queue.wait();
        result = next;
        cur = new_shape;
    }

    // Carry the full out_ndim shape even when no dim was tiled (right-aligned
    // padding adds leading singleton dims).
    if (static_cast<int64_t>(result.shape().size()) != out_ndim) {
        result = reshape_kernel(result, cur, queue);
    }
    return result;
}

// ============================================================================
// Take kernel - flattened index selection
// ============================================================================

class TakeKernelFloat32;
class TakeKernelFloat64;
class TakeKernelFloat16;
class TakeKernelBFloat16;
class TakeKernelInt32;
class TakeKernelInt64;

auto take_kernel(const Tensor& input, const Tensor& indices, sycl::queue& queue) -> Tensor {
    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);
    Tensor idx_cont = indices.is_contiguous() ? indices : contiguous_kernel(indices, queue);

    int64_t num_indices = idx_cont.numel();
    std::vector<int64_t> out_shape(idx_cont.shape().begin(), idx_cont.shape().end());

    Tensor output(out_shape, in_cont.dtype(), in_cont.device());

    // Indices can be Int32 or Int64
    bool is_int64 = (idx_cont.dtype() == DType::Int64);

    if (in_cont.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(in_cont);
        float* out_ptr = get_data_ptr<float>(output);

        if (is_int64) {
            const int64_t* idx_ptr = get_data_ptr<const int64_t>(idx_cont);
            queue.parallel_for<TakeKernelFloat32>(sycl::range<1>(num_indices), [=](sycl::id<1> i) {
                out_ptr[i] = in_ptr[idx_ptr[i]];
            });
        } else {
            const int32_t* idx_ptr = get_data_ptr<const int32_t>(idx_cont);
            queue.parallel_for(sycl::range<1>(num_indices), [=](sycl::id<1> i) {
                out_ptr[i] = in_ptr[idx_ptr[i]];
            });
        }
    }
    else if (in_cont.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(in_cont);
        double* out_ptr = get_data_ptr<double>(output);

        if (is_int64) {
            const int64_t* idx_ptr = get_data_ptr<const int64_t>(idx_cont);
            queue.parallel_for<TakeKernelFloat64>(sycl::range<1>(num_indices), [=](sycl::id<1> i) {
                out_ptr[i] = in_ptr[idx_ptr[i]];
            });
        } else {
            const int32_t* idx_ptr = get_data_ptr<const int32_t>(idx_cont);
            queue.parallel_for(sycl::range<1>(num_indices), [=](sycl::id<1> i) {
                out_ptr[i] = in_ptr[idx_ptr[i]];
            });
        }
    }
    else if (in_cont.dtype() == DType::Float16 || in_cont.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        if (is_int64) {
            const int64_t* idx_ptr = get_data_ptr<const int64_t>(idx_cont);
            queue.parallel_for<TakeKernelFloat16>(sycl::range<1>(num_indices), [=](sycl::id<1> i) {
                out_ptr[i] = in_ptr[idx_ptr[i]];
            });
        } else {
            const int32_t* idx_ptr = get_data_ptr<const int32_t>(idx_cont);
            queue.parallel_for<TakeKernelBFloat16>(sycl::range<1>(num_indices), [=](sycl::id<1> i) {
                out_ptr[i] = in_ptr[idx_ptr[i]];
            });
        }
    }
    else if (in_cont.dtype() == DType::Int32) {
        const int32_t* in_ptr = get_data_ptr<const int32_t>(in_cont);
        int32_t* out_ptr = get_data_ptr<int32_t>(output);

        if (is_int64) {
            const int64_t* idx_ptr = get_data_ptr<const int64_t>(idx_cont);
            queue.parallel_for<TakeKernelInt32>(sycl::range<1>(num_indices), [=](sycl::id<1> i) {
                out_ptr[i] = in_ptr[idx_ptr[i]];
            });
        } else {
            const int32_t* idx_ptr = get_data_ptr<const int32_t>(idx_cont);
            queue.parallel_for(sycl::range<1>(num_indices), [=](sycl::id<1> i) {
                out_ptr[i] = in_ptr[idx_ptr[i]];
            });
        }
    }
    else if (in_cont.dtype() == DType::Int64) {
        const int64_t* in_ptr = get_data_ptr<const int64_t>(in_cont);
        int64_t* out_ptr = get_data_ptr<int64_t>(output);

        if (is_int64) {
            const int64_t* idx_ptr = get_data_ptr<const int64_t>(idx_cont);
            queue.parallel_for<TakeKernelInt64>(sycl::range<1>(num_indices), [=](sycl::id<1> i) {
                out_ptr[i] = in_ptr[idx_ptr[i]];
            });
        } else {
            const int32_t* idx_ptr = get_data_ptr<const int32_t>(idx_cont);
            queue.parallel_for(sycl::range<1>(num_indices), [=](sycl::id<1> i) {
                out_ptr[i] = in_ptr[idx_ptr[i]];
            });
        }
    }
    else {
        throw std::runtime_error("take_kernel: unsupported dtype " +
                                 std::to_string(static_cast<int>(in_cont.dtype())));
    }

    return output;
}

// ============================================================================
// Unfold kernel - extract sliding local blocks (im2col-like for 1D)
// ============================================================================

class UnfoldIm2colFloat32;
class UnfoldIm2colFloat64;
class FoldCol2imFloat32;
class FoldCol2imFloat64;

auto unfold_kernel(const Tensor& input,
                    int64_t kH, int64_t kW,
                    int64_t sH, int64_t sW,
                    int64_t pH, int64_t pW,
                    int64_t dH, int64_t dW,
                    sycl::queue& queue) -> Tensor {
    // Y.12: per-axis 2D unfold: input [N, C, H, W] -> output [N, C*kH*kW, L].
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::runtime_error("unfold_kernel: expected 4D input [N, C, H, W]");
    }

    // Float16 / BFloat16: widen to Float32, compute, narrow back.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        const DType orig_dtype = input.dtype();
        Tensor widened = input.to(DType::Float32);
        Tensor result = unfold_kernel(widened, kH, kW, sH, sW, pH, pW, dH, dW,
                                      queue);
        return result.to(orig_dtype);
    }

    int64_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    int64_t H_out = (H + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
    int64_t W_out = (W + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
    int64_t L = H_out * W_out;
    int64_t channels_col = C * kH * kW;

    Tensor output({N, channels_col, L}, input.dtype(), input.device());

    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);

    // GPU im2col: each work-item computes one output element per batch
    int64_t col_size_per_batch = channels_col * L;

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = in_cont.data<float>();
        float* out_ptr = output.data<float>();
        for (int64_t n = 0; n < N; ++n) {
            const float* batch_in = in_ptr + n * C * H * W;
            float* batch_out = out_ptr + n * col_size_per_batch;
            queue.parallel_for<UnfoldIm2colFloat32>(sycl::range<1>(col_size_per_batch),
                [=](sycl::id<1> index) {
                    int64_t w_o = index % W_out;
                    int64_t idx = index / W_out;
                    int64_t h_o = idx % H_out;
                    idx /= H_out;
                    int64_t kw = idx % kW;
                    idx /= kW;
                    int64_t kh = idx % kH;
                    int64_t c = idx / kH;
                    int64_t h_in = h_o * sH - pH + kh * dH;
                    int64_t w_in = w_o * sW - pW + kw * dW;
                    batch_out[index] = (h_in >= 0 && w_in >= 0 && h_in < H && w_in < W) ?
                        batch_in[(c * H + h_in) * W + w_in] : 0.0f;
                });
        }
        queue.wait_and_throw();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = in_cont.data<double>();
        double* out_ptr = output.data<double>();
        for (int64_t n = 0; n < N; ++n) {
            const double* batch_in = in_ptr + n * C * H * W;
            double* batch_out = out_ptr + n * col_size_per_batch;
            queue.parallel_for<UnfoldIm2colFloat64>(sycl::range<1>(col_size_per_batch),
                [=](sycl::id<1> index) {
                    int64_t w_o = index % W_out;
                    int64_t idx = index / W_out;
                    int64_t h_o = idx % H_out;
                    idx /= H_out;
                    int64_t kw = idx % kW;
                    idx /= kW;
                    int64_t kh = idx % kH;
                    int64_t c = idx / kH;
                    int64_t h_in = h_o * sH - pH + kh * dH;
                    int64_t w_in = w_o * sW - pW + kw * dW;
                    batch_out[index] = (h_in >= 0 && w_in >= 0 && h_in < H && w_in < W) ?
                        batch_in[(c * H + h_in) * W + w_in] : 0.0;
                });
        }
        queue.wait_and_throw();
    } else {
        throw std::runtime_error("unfold_kernel: unsupported dtype (expected Float32 or Float64)");
    }

    return output;
}

// ============================================================================
// Fold kernel - inverse of unfold (col2im)
// ============================================================================

auto fold_kernel(const Tensor& input, const std::vector<int64_t>& output_size,
                  int64_t kH, int64_t kW,
                  int64_t sH, int64_t sW,
                  int64_t pH, int64_t pW,
                  int64_t dH, int64_t dW,
                  sycl::queue& queue) -> Tensor {
    // Y.12: per-axis 2D fold (col2im).
    auto shape = input.shape();
    if (shape.size() != 3) {
        throw std::runtime_error("fold_kernel: expected 3D input [N, C*kH*kW, L]");
    }

    // Float16 / BFloat16: widen to Float32, compute, narrow back.
    // sycl::atomic_ref used by the col2im accumulator does not support half
    // types. Widen/narrow around the Float32 path instead.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        const DType orig_dtype = input.dtype();
        Tensor widened = input.to(DType::Float32);
        Tensor result = fold_kernel(widened, output_size,
                                    kH, kW, sH, sW, pH, pW, dH, dW, queue);
        return result.to(orig_dtype);
    }

    int64_t N = shape[0];
    int64_t channels_col = shape[1];
    int64_t L = shape[2];
    int64_t H_out = output_size[0];
    int64_t W_out = output_size[1];
    int64_t C = channels_col / (kH * kW);

    Tensor output({N, C, H_out, W_out}, input.dtype(), input.device());

    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);

    int64_t H_col = (H_out + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
    int64_t W_col = (W_out + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
    int64_t col_size_per_batch = channels_col * L;
    int64_t im_size_per_batch = C * H_out * W_out;

    // GPU col2im: each work-item handles one column entry, atomically accumulates into output
    int64_t col2im_work_size = C * kH * kW * H_col * W_col;

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = in_cont.data<float>();
        float* out_ptr = output.data<float>();
        for (int64_t n = 0; n < N; ++n) {
            const float* batch_col = in_ptr + n * col_size_per_batch;
            float* batch_im = out_ptr + n * im_size_per_batch;
            queue.fill(batch_im, 0.0f, im_size_per_batch);
            queue.parallel_for<FoldCol2imFloat32>(sycl::range<1>(col2im_work_size),
                [=](sycl::id<1> index) {
                    int64_t w_c = index % W_col;
                    int64_t idx = index / W_col;
                    int64_t h_c = idx % H_col;
                    idx /= H_col;
                    int64_t kw = idx % kW;
                    idx /= kW;
                    int64_t kh = idx % kH;
                    int64_t c = idx / kH;
                    int64_t h = h_c * sH - pH + kh * dH;
                    int64_t w = w_c * sW - pW + kw * dW;
                    if (h >= 0 && w >= 0 && h < H_out && w < W_out) {
                        int64_t col_idx = ((c * kH + kh) * kW + kw);
                        int64_t l = h_c * W_col + w_c;
                        int64_t im_idx = (c * H_out + h) * W_out + w;
                        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>
                            atomic_val(batch_im[im_idx]);
                        atomic_val.fetch_add(batch_col[col_idx * L + l]);
                    }
                });
        }
        queue.wait_and_throw();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = in_cont.data<double>();
        double* out_ptr = output.data<double>();
        for (int64_t n = 0; n < N; ++n) {
            const double* batch_col = in_ptr + n * col_size_per_batch;
            double* batch_im = out_ptr + n * im_size_per_batch;
            queue.fill(batch_im, 0.0, im_size_per_batch);
            queue.parallel_for<FoldCol2imFloat64>(sycl::range<1>(col2im_work_size),
                [=](sycl::id<1> index) {
                    int64_t w_c = index % W_col;
                    int64_t idx = index / W_col;
                    int64_t h_c = idx % H_col;
                    idx /= H_col;
                    int64_t kw = idx % kW;
                    idx /= kW;
                    int64_t kh = idx % kH;
                    int64_t c = idx / kH;
                    int64_t h = h_c * sH - pH + kh * dH;
                    int64_t w = w_c * sW - pW + kw * dW;
                    if (h >= 0 && w >= 0 && h < H_out && w < W_out) {
                        int64_t col_idx = ((c * kH + kh) * kW + kw);
                        int64_t l = h_c * W_col + w_c;
                        int64_t im_idx = (c * H_out + h) * W_out + w;
                        sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device>
                            atomic_val(batch_im[im_idx]);
                        atomic_val.fetch_add(batch_col[col_idx * L + l]);
                    }
                });
        }
        queue.wait_and_throw();
    } else {
        throw std::runtime_error("fold_kernel: unsupported dtype (expected Float32 or Float64)");
    }

    return output;
}

// ============================================================================
// Roll kernel - shift elements along a dimension with wraparound
// ============================================================================

class RollKernelFloat32;
class RollKernelFloat64;
class RollKernelFloat16;
class RollKernelBFloat16;
class RollKernelInt32;
class RollKernelInt64;

auto roll_kernel(const Tensor& input, int64_t shift, int64_t dim, sycl::queue& queue) -> Tensor {
    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);
    auto shape = in_cont.shape();
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;

    int64_t dim_size = shape[dim];
    // Normalize shift to [0, dim_size)
    shift = ((shift % dim_size) + dim_size) % dim_size;

    // Return independent storage (matching the CPU reference), not an alias of
    // the input: an in-place mutation of the result must not corrupt the input.
    if (shift == 0) return clone_kernel(in_cont, queue);

    Tensor output(std::vector<int64_t>(shape.begin(), shape.end()),
                  in_cont.dtype(), in_cont.device());

    int64_t outer_size = 1;
    for (int64_t d = 0; d < dim; ++d) outer_size *= shape[d];
    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) inner_size *= shape[d];

    int64_t numel = in_cont.numel();

    if (in_cont.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(in_cont);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<RollKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int64_t i = idx;
            int64_t inner_idx = i % inner_size;
            int64_t dim_idx = (i / inner_size) % dim_size;
            int64_t outer_idx = i / (inner_size * dim_size);
            int64_t new_dim_idx = (dim_idx + shift) % dim_size;
            int64_t out_i = (outer_idx * dim_size + new_dim_idx) * inner_size + inner_idx;
            out_ptr[out_i] = in_ptr[i];
        });
    }
    else if (in_cont.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(in_cont);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<RollKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int64_t i = idx;
            int64_t inner_idx = i % inner_size;
            int64_t dim_idx = (i / inner_size) % dim_size;
            int64_t outer_idx = i / (inner_size * dim_size);
            int64_t new_dim_idx = (dim_idx + shift) % dim_size;
            int64_t out_i = (outer_idx * dim_size + new_dim_idx) * inner_size + inner_idx;
            out_ptr[out_i] = in_ptr[i];
        });
    }
    else if (in_cont.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<RollKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int64_t i = idx;
            int64_t inner_idx = i % inner_size;
            int64_t dim_idx = (i / inner_size) % dim_size;
            int64_t outer_idx = i / (inner_size * dim_size);
            int64_t new_dim_idx = (dim_idx + shift) % dim_size;
            int64_t out_i = (outer_idx * dim_size + new_dim_idx) * inner_size + inner_idx;
            out_ptr[out_i] = in_ptr[i];
        });
    }
    else if (in_cont.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<RollKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int64_t i = idx;
            int64_t inner_idx = i % inner_size;
            int64_t dim_idx = (i / inner_size) % dim_size;
            int64_t outer_idx = i / (inner_size * dim_size);
            int64_t new_dim_idx = (dim_idx + shift) % dim_size;
            int64_t out_i = (outer_idx * dim_size + new_dim_idx) * inner_size + inner_idx;
            out_ptr[out_i] = in_ptr[i];
        });
    }
    else if (in_cont.dtype() == DType::Int32) {
        const int32_t* in_ptr = get_data_ptr<const int32_t>(in_cont);
        int32_t* out_ptr = get_data_ptr<int32_t>(output);
        queue.parallel_for<RollKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int64_t i = idx;
            int64_t inner_idx = i % inner_size;
            int64_t dim_idx = (i / inner_size) % dim_size;
            int64_t outer_idx = i / (inner_size * dim_size);
            int64_t new_dim_idx = (dim_idx + shift) % dim_size;
            int64_t out_i = (outer_idx * dim_size + new_dim_idx) * inner_size + inner_idx;
            out_ptr[out_i] = in_ptr[i];
        });
    }
    else if (in_cont.dtype() == DType::Int64) {
        const int64_t* in_ptr = get_data_ptr<const int64_t>(in_cont);
        int64_t* out_ptr = get_data_ptr<int64_t>(output);
        queue.parallel_for<RollKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int64_t i = idx;
            int64_t inner_idx = i % inner_size;
            int64_t dim_idx = (i / inner_size) % dim_size;
            int64_t outer_idx = i / (inner_size * dim_size);
            int64_t new_dim_idx = (dim_idx + shift) % dim_size;
            int64_t out_i = (outer_idx * dim_size + new_dim_idx) * inner_size + inner_idx;
            out_ptr[out_i] = in_ptr[i];
        });
    }
    else {
        throw std::runtime_error("roll_kernel: unsupported dtype " +
                                 std::to_string(static_cast<int>(in_cont.dtype())));
    }

    return output;
}

// ============================================================================
// Cast kernel - type conversion
// ============================================================================
class CastF32ToF64;
class CastF64ToF32;
class CastF32ToF16;
class CastF16ToF32;
class CastF32ToI32;
class CastI32ToF32;
class CastF32ToI64;
class CastI64ToF32;
class CastF32ToBF16;
class CastBF16ToF32;
class CastF32ToU8;
class CastU8ToF32;
class CastF32ToI8;
class CastI8ToF32;
class CastF32ToBool;
class CastBoolToF32;
class CastI64ToF64;
class CastF64ToI64;

auto cast_kernel(const Tensor& input, DType target_dtype, sycl::queue& queue) -> Tensor {
    if (input.dtype() == target_dtype) {
        return clone_kernel(input, queue);
    }

    // Intel oneAPI CPU runtime bug: device casts touching 16-bit integer
    // dtypes SEGV inside the runtime (QueryMaxMemAllocSize during an
    // enqueue-triggered internal flush; "PLEASE submit a bug report" banner).
    // Host<->device memcpy of 16-bit data is safe, so do the cast losslessly
    // on the host and upload the result. Same proven detour as the oneAPI
    // argmax/argmin Int16 path (see feedback_rocm_intrinsic_nan-adjacent
    // notes / grind session 2).
    {
        auto runtime_unsafe = [](DType dt) {
            // UInt32 casts crash the same way (verified via SumDtypeGap.UInt32
            // backtrace), so it takes the host detour too.
            return dt == DType::Int16 || dt == DType::UInt16 || dt == DType::UInt32;
        };
        if (runtime_unsafe(input.dtype()) || runtime_unsafe(target_dtype)) {
            return input.to(Device::cpu()).to(target_dtype).to(input.device());
        }
    }

    auto shape_span = input.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
    Tensor output(shape, target_dtype, input.device());
    const int64_t numel = input.numel();
    if (numel == 0) return output;

    DType src = input.dtype();
    DType dst = target_dtype;

    if (src == DType::Float32 && dst == DType::Float64) {
        const float* in = get_data_ptr<const float>(input);
        double* out = get_data_ptr<double>(output);
        queue.parallel_for<CastF32ToF64>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            out[i] = static_cast<double>(in[i]);
        });
    } else if (src == DType::Float64 && dst == DType::Float32) {
        const double* in = get_data_ptr<const double>(input);
        float* out = get_data_ptr<float>(output);
        queue.parallel_for<CastF64ToF32>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            out[i] = static_cast<float>(in[i]);
        });
    } else if (src == DType::Float32 && dst == DType::Float16) {
        const float* in = get_data_ptr<const float>(input);
        sycl::half* out = get_data_ptr<sycl::half>(output);
        queue.parallel_for<CastF32ToF16>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            out[i] = sycl::half(in[i]);
        });
    } else if (src == DType::Float16 && dst == DType::Float32) {
        const sycl::half* in = get_data_ptr<const sycl::half>(input);
        float* out = get_data_ptr<float>(output);
        queue.parallel_for<CastF16ToF32>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            out[i] = static_cast<float>(in[i]);
        });
    } else if (src == DType::Float32 && dst == DType::BFloat16) {
        const float* in = get_data_ptr<const float>(input);
        uint16_t* out = get_data_ptr<uint16_t>(output);
        queue.parallel_for<CastF32ToBF16>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            out[i] = f32_to_bf16(in[i]);
        });
    } else if (src == DType::BFloat16 && dst == DType::Float32) {
        const uint16_t* in = get_data_ptr<const uint16_t>(input);
        float* out = get_data_ptr<float>(output);
        queue.parallel_for<CastBF16ToF32>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            out[i] = bf16_to_f32(in[i]);
        });
    } else if (src == DType::Float32 && dst == DType::Int32) {
        const float* in = get_data_ptr<const float>(input);
        int32_t* out = get_data_ptr<int32_t>(output);
        queue.parallel_for<CastF32ToI32>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            out[i] = static_cast<int32_t>(in[i]);
        });
    } else if (src == DType::Int32 && dst == DType::Float32) {
        const int32_t* in = get_data_ptr<const int32_t>(input);
        float* out = get_data_ptr<float>(output);
        queue.parallel_for<CastI32ToF32>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            out[i] = static_cast<float>(in[i]);
        });
    } else if (src == DType::Float32 && dst == DType::Int64) {
        const float* in = get_data_ptr<const float>(input);
        int64_t* out = get_data_ptr<int64_t>(output);
        queue.parallel_for<CastF32ToI64>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            out[i] = static_cast<int64_t>(in[i]);
        });
    } else if (src == DType::Int64 && dst == DType::Float32) {
        const int64_t* in = get_data_ptr<const int64_t>(input);
        float* out = get_data_ptr<float>(output);
        queue.parallel_for<CastI64ToF32>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            out[i] = static_cast<float>(in[i]);
        });
    } else if (src == DType::Float32 && dst == DType::UInt8) {
        const float* in = get_data_ptr<const float>(input);
        uint8_t* out = get_data_ptr<uint8_t>(output);
        queue.parallel_for<CastF32ToU8>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            out[i] = static_cast<uint8_t>(sycl::clamp(in[i], 0.0f, 255.0f));
        });
    } else if (src == DType::UInt8 && dst == DType::Float32) {
        const uint8_t* in = get_data_ptr<const uint8_t>(input);
        float* out = get_data_ptr<float>(output);
        queue.parallel_for<CastU8ToF32>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            out[i] = static_cast<float>(in[i]);
        });
    } else if (src == DType::Float32 && dst == DType::Int8) {
        const float* in = get_data_ptr<const float>(input);
        int8_t* out = get_data_ptr<int8_t>(output);
        queue.parallel_for<CastF32ToI8>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            out[i] = static_cast<int8_t>(sycl::clamp(in[i], -128.0f, 127.0f));
        });
    } else if (src == DType::Int8 && dst == DType::Float32) {
        const int8_t* in = get_data_ptr<const int8_t>(input);
        float* out = get_data_ptr<float>(output);
        queue.parallel_for<CastI8ToF32>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            out[i] = static_cast<float>(in[i]);
        });
    } else if (src == DType::Float32 && dst == DType::Bool) {
        const float* in = get_data_ptr<const float>(input);
        bool* out = get_data_ptr<bool>(output);
        queue.parallel_for<CastF32ToBool>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            out[i] = (in[i] != 0.0f);
        });
    } else if (src == DType::Bool && dst == DType::Float32) {
        const bool* in = get_data_ptr<const bool>(input);
        float* out = get_data_ptr<float>(output);
        queue.parallel_for<CastBoolToF32>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            out[i] = in[i] ? 1.0f : 0.0f;
        });
    } else if (src == DType::Int64 && dst == DType::Float64) {
        const int64_t* in = get_data_ptr<const int64_t>(input);
        double* out = get_data_ptr<double>(output);
        queue.parallel_for<CastI64ToF64>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            out[i] = static_cast<double>(in[i]);
        });
    } else if (src == DType::Float64 && dst == DType::Int64) {
        const double* in = get_data_ptr<const double>(input);
        int64_t* out = get_data_ptr<int64_t>(output);
        queue.parallel_for<CastF64ToI64>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            out[i] = static_cast<int64_t>(in[i]);
        });
    } else if ((src == DType::FP8_E4M3 || src == DType::FP8_E5M2) && dst == DType::Float32) {
        // FP8 → Float32: device-side bit manipulation
        const uint8_t* in = get_data_ptr<const uint8_t>(input);
        float* out = get_data_ptr<float>(output);
        bool is_e4m3 = (src == DType::FP8_E4M3);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            uint8_t bits = in[idx];
            uint32_t f_sign, f_exp, f_mantissa;
            if (is_e4m3) {
                uint32_t sign = (bits >> 7) & 0x1;
                uint32_t exp = (bits >> 3) & 0xF;
                uint32_t mantissa = bits & 0x7;
                f_sign = sign;
                if (exp == 0) {
                    if (mantissa == 0) { f_exp = 0; f_mantissa = 0; }
                    else {
                        int e = -1; uint32_t m = mantissa;
                        do { e++; m <<= 1; } while ((m & 0x8) == 0);
                        f_exp = 127 - 7 - e; f_mantissa = (m & 0x7) << 20;
                    }
                } else if (exp == 0xF && mantissa != 0) {
                    f_exp = 0xFF; f_mantissa = mantissa << 20;
                } else {
                    f_exp = exp - 7 + 127; f_mantissa = mantissa << 20;
                }
            } else {
                uint32_t sign = (bits >> 7) & 0x1;
                uint32_t exp = (bits >> 2) & 0x1F;
                uint32_t mantissa = bits & 0x3;
                f_sign = sign;
                if (exp == 0) {
                    if (mantissa == 0) { f_exp = 0; f_mantissa = 0; }
                    else {
                        int e = -1; uint32_t m = mantissa;
                        do { e++; m <<= 1; } while ((m & 0x4) == 0);
                        f_exp = 127 - 15 - e; f_mantissa = (m & 0x3) << 21;
                    }
                } else if (exp == 0x1F) {
                    f_exp = 0xFF; f_mantissa = mantissa << 21;
                } else {
                    f_exp = exp - 15 + 127; f_mantissa = mantissa << 21;
                }
            }
            uint32_t f_bits = (f_sign << 31) | (f_exp << 23) | f_mantissa;
            out[idx] = sycl::bit_cast<float>(f_bits);
        });
        queue.wait_and_throw();
    } else if (src == DType::Float32 && (dst == DType::FP8_E4M3 || dst == DType::FP8_E5M2)) {
        // Float32 → FP8: device-side bit manipulation
        const float* in = get_data_ptr<const float>(input);
        uint8_t* out = get_data_ptr<uint8_t>(output);
        bool is_e4m3 = (dst == DType::FP8_E4M3);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            uint32_t f_bits = sycl::bit_cast<uint32_t>(in[idx]);
            uint32_t sign = (f_bits >> 31) & 0x1;
            uint32_t exp = (f_bits >> 23) & 0xFF;
            uint32_t mantissa = f_bits & 0x7FFFFF;
            uint8_t h_sign = static_cast<uint8_t>(sign);
            uint8_t h_exp, h_mantissa;
            if (is_e4m3) {
                if (exp == 0xFF) { h_exp = 0xF; h_mantissa = 0x7; }
                else if (exp == 0) { h_exp = 0; h_mantissa = 0; }
                else {
                    int32_t new_exp = static_cast<int32_t>(exp) - 127 + 7;
                    if (new_exp >= 0xF) { h_exp = 0xE; h_mantissa = 0x7; }
                    else if (new_exp <= 0) {
                        if (new_exp >= -3) {
                            uint32_t m = (mantissa | 0x800000) >> (1 - new_exp);
                            h_mantissa = static_cast<uint8_t>((m >> 20) & 0x7); h_exp = 0;
                        } else { h_exp = 0; h_mantissa = 0; }
                    } else {
                        h_exp = static_cast<uint8_t>(new_exp);
                        h_mantissa = static_cast<uint8_t>((mantissa >> 20) & 0x7);
                    }
                }
                out[idx] = (h_sign << 7) | (h_exp << 3) | h_mantissa;
            } else {
                if (exp == 0xFF) { h_exp = 0x1F; h_mantissa = mantissa ? 0x3 : 0; }
                else if (exp == 0) { h_exp = 0; h_mantissa = 0; }
                else {
                    int32_t new_exp = static_cast<int32_t>(exp) - 127 + 15;
                    if (new_exp >= 0x1F) { h_exp = 0x1F; h_mantissa = 0; }
                    else if (new_exp <= 0) {
                        if (new_exp >= -2) {
                            uint32_t m = (mantissa | 0x800000) >> (1 - new_exp);
                            h_mantissa = static_cast<uint8_t>((m >> 21) & 0x3); h_exp = 0;
                        } else { h_exp = 0; h_mantissa = 0; }
                    } else {
                        h_exp = static_cast<uint8_t>(new_exp);
                        h_mantissa = static_cast<uint8_t>((mantissa >> 21) & 0x3);
                    }
                }
                out[idx] = (h_sign << 7) | (h_exp << 2) | h_mantissa;
            }
        });
        queue.wait_and_throw();
    } else if ((src == DType::FP8_E4M3FNUZ || src == DType::FP8_E5M2FNUZ) && dst == DType::Float32) {
        // FP8 FNUZ → Float32: device-side bit manipulation. FNUZ (AMD/GraphCore,
        // ONNX FLOAT8E*FNUZ) differs from IEEE FP8: exponent bias is 8 (E4M3FNUZ)
        // / 16 (E5M2FNUZ) instead of 7 / 15, there are NO infinities, and the
        // only NaN is the encoding 0x80 (sign=1, exp=0, mantissa=0). There is no
        // negative zero. A top exponent is a normal number, not inf/NaN.
        const uint8_t* in = get_data_ptr<const uint8_t>(input);
        float* out = get_data_ptr<float>(output);
        bool is_e4m3 = (src == DType::FP8_E4M3FNUZ);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            uint8_t bits = in[idx];
            uint32_t f_sign, f_exp, f_mantissa;
            if (is_e4m3) {
                uint32_t sign = (bits >> 7) & 0x1;
                uint32_t exp = (bits >> 3) & 0xF;
                uint32_t mantissa = bits & 0x7;
                f_sign = sign;
                if (exp == 0) {
                    if (mantissa == 0) {
                        // 0x00 → +0. The 0x80 encoding (sign=1) is NaN.
                        if (sign) { f_exp = 0xFF; f_mantissa = 0x400000; }
                        else      { f_exp = 0;    f_mantissa = 0; }
                    } else {
                        // Subnormal: bias 8 → leading-1 position normalized.
                        int e = -1; uint32_t m = mantissa;
                        do { e++; m <<= 1; } while ((m & 0x8) == 0);
                        f_exp = 127 - 8 - e; f_mantissa = (m & 0x7) << 20;
                    }
                } else {
                    // Normal: no all-ones special case for FNUZ.
                    f_exp = exp - 8 + 127; f_mantissa = mantissa << 20;
                }
            } else {
                uint32_t sign = (bits >> 7) & 0x1;
                uint32_t exp = (bits >> 2) & 0x1F;
                uint32_t mantissa = bits & 0x3;
                f_sign = sign;
                if (exp == 0) {
                    if (mantissa == 0) {
                        if (sign) { f_exp = 0xFF; f_mantissa = 0x400000; }
                        else      { f_exp = 0;    f_mantissa = 0; }
                    } else {
                        // Subnormal: bias 16.
                        int e = -1; uint32_t m = mantissa;
                        do { e++; m <<= 1; } while ((m & 0x4) == 0);
                        f_exp = 127 - 16 - e; f_mantissa = (m & 0x3) << 21;
                    }
                } else {
                    f_exp = exp - 16 + 127; f_mantissa = mantissa << 21;
                }
            }
            uint32_t f_bits = (f_sign << 31) | (f_exp << 23) | f_mantissa;
            out[idx] = sycl::bit_cast<float>(f_bits);
        });
        queue.wait_and_throw();
    } else if (src == DType::Float32 && (dst == DType::FP8_E4M3FNUZ || dst == DType::FP8_E5M2FNUZ)) {
        // Float32 → FP8 FNUZ: device-side bit manipulation. Bias 8 (E4M3FNUZ) /
        // 16 (E5M2FNUZ); no inf (overflow saturates to max finite); the only NaN
        // is 0x80, so NaN/Inf inputs map to 0x80. No negative zero (a signed zero
        // result collapses to +0 = 0x00).
        const float* in = get_data_ptr<const float>(input);
        uint8_t* out = get_data_ptr<uint8_t>(output);
        bool is_e4m3 = (dst == DType::FP8_E4M3FNUZ);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            uint32_t f_bits = sycl::bit_cast<uint32_t>(in[idx]);
            uint32_t sign = (f_bits >> 31) & 0x1;
            uint32_t exp = (f_bits >> 23) & 0xFF;
            uint32_t mantissa = f_bits & 0x7FFFFF;
            if (is_e4m3) {
                if (exp == 0xFF) {
                    // Inf or NaN both map to the single FNUZ NaN encoding 0x80.
                    out[idx] = 0x80;
                    return;
                }
                uint8_t h_sign = static_cast<uint8_t>(sign);
                uint8_t h_exp, h_mantissa;
                if (exp == 0) { h_exp = 0; h_mantissa = 0; }
                else {
                    int32_t new_exp = static_cast<int32_t>(exp) - 127 + 8;
                    // Max finite E4M3FNUZ exponent field is 0xF (15) with mantissa
                    // 0x7 — there is no inf reserve, so saturate to that.
                    if (new_exp >= 0xF) { h_exp = 0xF; h_mantissa = 0x7; }
                    else if (new_exp <= 0) {
                        if (new_exp >= -3) {
                            uint32_t m = (mantissa | 0x800000) >> (1 - new_exp);
                            h_mantissa = static_cast<uint8_t>((m >> 20) & 0x7); h_exp = 0;
                        } else { h_exp = 0; h_mantissa = 0; }
                    } else {
                        h_exp = static_cast<uint8_t>(new_exp);
                        h_mantissa = static_cast<uint8_t>((mantissa >> 20) & 0x7);
                    }
                }
                uint8_t res = (h_sign << 7) | (h_exp << 3) | h_mantissa;
                // No negative zero in FNUZ: collapse 0x80 (sign=1, zero) to +0.
                out[idx] = (res == 0x80) ? 0x00 : res;
            } else {
                if (exp == 0xFF) {
                    out[idx] = 0x80;
                    return;
                }
                uint8_t h_sign = static_cast<uint8_t>(sign);
                uint8_t h_exp, h_mantissa;
                if (exp == 0) { h_exp = 0; h_mantissa = 0; }
                else {
                    int32_t new_exp = static_cast<int32_t>(exp) - 127 + 16;
                    // Max finite E5M2FNUZ: exponent field 0x1F (31), mantissa 0x3.
                    if (new_exp >= 0x1F) { h_exp = 0x1F; h_mantissa = 0x3; }
                    else if (new_exp <= 0) {
                        if (new_exp >= -2) {
                            uint32_t m = (mantissa | 0x800000) >> (1 - new_exp);
                            h_mantissa = static_cast<uint8_t>((m >> 21) & 0x3); h_exp = 0;
                        } else { h_exp = 0; h_mantissa = 0; }
                    } else {
                        h_exp = static_cast<uint8_t>(new_exp);
                        h_mantissa = static_cast<uint8_t>((mantissa >> 21) & 0x3);
                    }
                }
                uint8_t res = (h_sign << 7) | (h_exp << 2) | h_mantissa;
                out[idx] = (res == 0x80) ? 0x00 : res;
            }
        });
        queue.wait_and_throw();
    } else if (src == DType::Float32 && dst == DType::Complex64) {
        // Real → complex: fill imaginary part with zeros. Complex64
        // storage is interleaved (re, im) float pairs.
        const float* in = get_data_ptr<const float>(input);
        float* out = reinterpret_cast<float*>(const_cast<void*>(output.data_ptr()));
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            int64_t k = i[0];
            out[2 * k]     = in[k];
            out[2 * k + 1] = 0.0f;
        });
    } else if (src == DType::Float64 && dst == DType::Complex128) {
        const double* in = get_data_ptr<const double>(input);
        double* out = reinterpret_cast<double*>(const_cast<void*>(output.data_ptr()));
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            int64_t k = i[0];
            out[2 * k]     = in[k];
            out[2 * k + 1] = 0.0;
        });
    } else if (src == DType::Complex64 && dst == DType::Float32) {
        // Complex → real: drop the imaginary part.
        const float* in = reinterpret_cast<const float*>(input.data_ptr());
        float* out = get_data_ptr<float>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            int64_t k = i[0];
            out[k] = in[2 * k];
        });
    } else if (src == DType::Complex128 && dst == DType::Float64) {
        const double* in = reinterpret_cast<const double*>(input.data_ptr());
        double* out = get_data_ptr<double>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            int64_t k = i[0];
            out[k] = in[2 * k];
        });
    } else if (src == DType::Complex64 && dst == DType::Complex128) {
        const float* in = reinterpret_cast<const float*>(input.data_ptr());
        double* out = reinterpret_cast<double*>(const_cast<void*>(output.data_ptr()));
        queue.parallel_for(sycl::range<1>(numel * 2), [=](sycl::id<1> i) {
            out[i[0]] = static_cast<double>(in[i[0]]);
        });
    } else if (src == DType::Complex128 && dst == DType::Complex64) {
        const double* in = reinterpret_cast<const double*>(input.data_ptr());
        float* out = reinterpret_cast<float*>(const_cast<void*>(output.data_ptr()));
        queue.parallel_for(sycl::range<1>(numel * 2), [=](sycl::id<1> i) {
            out[i[0]] = static_cast<float>(in[i[0]]);
        });
    } else if (src == DType::Float64 && dst == DType::Complex64) {
        // Direct Float64 -> Complex64: narrow the real value ONCE. The generic
        // two-hop fallback would go Float64 -> Float32 -> Complex64, discarding
        // ~29 mantissa bits before the complex result is even formed.
        const double* in = get_data_ptr<const double>(input);
        float* out = reinterpret_cast<float*>(const_cast<void*>(output.data_ptr()));
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            int64_t k = i[0];
            out[2 * k]     = static_cast<float>(in[k]);
            out[2 * k + 1] = 0.0f;
        });
    } else if (src == DType::Float32 && dst == DType::Complex128) {
        // Direct Float32 -> Complex128: widen the real value, no lossy hop.
        const float* in = get_data_ptr<const float>(input);
        double* out = reinterpret_cast<double*>(const_cast<void*>(output.data_ptr()));
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            int64_t k = i[0];
            out[2 * k]     = static_cast<double>(in[k]);
            out[2 * k + 1] = 0.0;
        });
    } else {
        // Two-hop: src -> Float32 -> dst
        if (src != DType::Float32) {
            Tensor as_f32 = cast_kernel(input, DType::Float32, queue);
            return cast_kernel(as_f32, target_dtype, queue);
        }
        throw std::runtime_error("cast_kernel: unsupported dtype conversion");
    }

    return output;
}

// ============================================================================
// StridedFill kernel - fill non-contiguous tensor in-place
// ============================================================================
class StridedFillKernelF32;
class StridedFillKernelF64;
class StridedFillKernelF16;
class StridedFillKernelBF16;
class StridedFillKernelI32;
class StridedFillKernelI64;

auto strided_fill_kernel(Tensor& self, double value, sycl::queue& queue) -> void {
    int64_t numel = self.numel();
    if (numel == 0) return;

    auto shape_span = self.shape();
    auto strides_span = self.strides();
    size_t ndim = shape_span.size();

    // The strided path uses fixed-size 8-element stack arrays for shape/strides;
    // the device loop indexes them with d < ndim, so >8-D would read/write OOB.
    if (ndim > 8) {
        throw std::runtime_error("strided_fill: tensors with more than 8 dimensions are not supported");
    }

    if (self.is_contiguous()) {
        if (self.dtype() == DType::Float32) {
            float val = static_cast<float>(value);
            float* ptr = get_data_ptr<float>(self);
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) { ptr[i] = val; }).wait();
        } else if (self.dtype() == DType::Float64) {
            double* ptr = get_data_ptr<double>(self);
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) { ptr[i] = value; }).wait();
        } else {
            Tensor filled = fill_kernel(self, value, queue);
            queue.memcpy(const_cast<void*>(self.data_ptr()), filled.data_ptr(),
                         numel * self.dtype_size()).wait();
        }
        return;
    }

    int64_t shape_arr[8] = {0}, strides_arr[8] = {0}, cont_strides_arr[8] = {0};
    for (size_t i = 0; i < ndim && i < 8; ++i) {
        shape_arr[i] = shape_span[i];
        strides_arr[i] = strides_span[i];
    }
    { int64_t s = 1; for (int64_t i = static_cast<int64_t>(ndim) - 1; i >= 0; --i) { cont_strides_arr[i] = s; s *= shape_span[i]; } }

    if (self.dtype() == DType::Float32) {
        float val = static_cast<float>(value);
        float* ptr = get_data_ptr<float>(self);
        queue.parallel_for<StridedFillKernelF32>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t remaining = flat_idx;
            int64_t offset = 0;
            for (size_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / cont_strides_arr[d];
                remaining %= cont_strides_arr[d];
                offset += coord * strides_arr[d];
            }
            ptr[offset] = val;
        }).wait();
    } else if (self.dtype() == DType::Float64) {
        double* ptr = get_data_ptr<double>(self);
        queue.parallel_for<StridedFillKernelF64>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t remaining = flat_idx;
            int64_t offset = 0;
            for (size_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / cont_strides_arr[d];
                remaining %= cont_strides_arr[d];
                offset += coord * strides_arr[d];
            }
            ptr[offset] = value;
        }).wait();
    } else if (self.dtype() == DType::Float16) {
        sycl::half val = sycl::half(static_cast<float>(value));
        sycl::half* ptr = get_data_ptr<sycl::half>(self);
        queue.parallel_for<StridedFillKernelF16>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t remaining = flat_idx;
            int64_t offset = 0;
            for (size_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / cont_strides_arr[d];
                remaining %= cont_strides_arr[d];
                offset += coord * strides_arr[d];
            }
            ptr[offset] = val;
        }).wait();
    } else if (self.dtype() == DType::BFloat16) {
        uint16_t val = f32_to_bf16(static_cast<float>(value));
        uint16_t* ptr = get_data_ptr<uint16_t>(self);
        queue.parallel_for<StridedFillKernelBF16>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t remaining = flat_idx;
            int64_t offset = 0;
            for (size_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / cont_strides_arr[d];
                remaining %= cont_strides_arr[d];
                offset += coord * strides_arr[d];
            }
            ptr[offset] = val;
        }).wait();
    } else if (self.dtype() == DType::Int32) {
        int32_t val = static_cast<int32_t>(value);
        int32_t* ptr = get_data_ptr<int32_t>(self);
        queue.parallel_for<StridedFillKernelI32>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t remaining = flat_idx;
            int64_t offset = 0;
            for (size_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / cont_strides_arr[d];
                remaining %= cont_strides_arr[d];
                offset += coord * strides_arr[d];
            }
            ptr[offset] = val;
        }).wait();
    } else if (self.dtype() == DType::Int64) {
        int64_t val = static_cast<int64_t>(value);
        int64_t* ptr = get_data_ptr<int64_t>(self);
        queue.parallel_for<StridedFillKernelI64>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t remaining = flat_idx;
            int64_t offset = 0;
            for (size_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / cont_strides_arr[d];
                remaining %= cont_strides_arr[d];
                offset += coord * strides_arr[d];
            }
            ptr[offset] = val;
        }).wait();
    } else {
        // Generic strided fill for the remaining fixed-width dtypes: compute the
        // scalar's storage representation on host and scatter it to the strided
        // offsets (offsets are in element units).
        auto strided_write = [&]<typename T>(T fillv) {
            T* ptr = get_data_ptr<T>(self);
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
                int64_t remaining = flat_idx, offset = 0;
                for (size_t d = 0; d < ndim; ++d) {
                    int64_t coord = remaining / cont_strides_arr[d];
                    remaining %= cont_strides_arr[d];
                    offset += coord * strides_arr[d];
                }
                ptr[offset] = fillv;
            }).wait();
        };
        auto strided_write_complex = [&]<typename R>(R re) {
            R* ptr = get_data_ptr<R>(self);
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
                int64_t remaining = flat_idx, offset = 0;
                for (size_t d = 0; d < ndim; ++d) {
                    int64_t coord = remaining / cont_strides_arr[d];
                    remaining %= cont_strides_arr[d];
                    offset += coord * strides_arr[d];
                }
                ptr[offset * 2] = re; ptr[offset * 2 + 1] = R(0);
            }).wait();
        };
        switch (self.dtype()) {
            case DType::Int8:   strided_write(static_cast<int8_t>(value)); break;
            case DType::Int16:  strided_write(static_cast<int16_t>(value)); break;
            case DType::UInt8:  strided_write(static_cast<uint8_t>(value)); break;
            case DType::UInt16: strided_write(static_cast<uint16_t>(value)); break;
            case DType::UInt32: strided_write(static_cast<uint32_t>(value)); break;
            case DType::UInt64: strided_write(static_cast<uint64_t>(value)); break;
            case DType::Bool:   strided_write(static_cast<bool>(value != 0.0)); break;
            case DType::Complex64:  strided_write_complex(static_cast<float>(value)); break;
            case DType::Complex128: strided_write_complex(static_cast<double>(value)); break;
            case DType::FP8_E4M3: { FP8_E4M3 v(static_cast<float>(value)); uint8_t b; std::memcpy(&b,&v,1); strided_write(b); break; }
            case DType::FP8_E5M2: { FP8_E5M2 v(static_cast<float>(value)); uint8_t b; std::memcpy(&b,&v,1); strided_write(b); break; }
            case DType::FP8_E4M3FNUZ: { FP8_E4M3FNUZ v(static_cast<float>(value)); uint8_t b; std::memcpy(&b,&v,1); strided_write(b); break; }
            case DType::FP8_E5M2FNUZ: { FP8_E5M2FNUZ v(static_cast<float>(value)); uint8_t b; std::memcpy(&b,&v,1); strided_write(b); break; }
            case DType::QInt8:
            case DType::QUInt8: {
                if (self.q_scale() == 0.0)
                    throw std::runtime_error("strided_fill: quantized tensor requires quantization params");
                const int64_t qv = static_cast<int64_t>(std::llround(value / self.q_scale())) + self.q_zero_point();
                const uint8_t b = (self.dtype() == DType::QInt8)
                    ? static_cast<uint8_t>(static_cast<int8_t>(std::clamp<int64_t>(qv, -128, 127)))
                    : static_cast<uint8_t>(std::clamp<int64_t>(qv, 0, 255));
                strided_write(b);
                break;
            }
            case DType::QInt4x2: {
                // Packed two-nibble byte. numel()/shape()/strides() are at byte
                // (packed) granularity, so every visited element is a whole byte
                // whose two logical 4-bit values both belong to the view — write
                // identical nibbles, mirroring the CPU StridedFill kernel.
                if (self.q_scale() == 0.0)
                    throw std::runtime_error("strided_fill: quantized tensor requires quantization params");
                const int64_t qv = static_cast<int64_t>(std::llround(value / self.q_scale())) + self.q_zero_point();
                const int64_t clamped = std::clamp<int64_t>(qv, -8, 7);
                const uint8_t b = static_cast<uint8_t>((clamped & 0xF) | ((clamped & 0xF) << 4));
                strided_write(b);
                break;
            }
            default:
                throw std::runtime_error("strided_fill: unsupported dtype for non-contiguous fill");
        }
    }
}

// ============================================================================
// ToMemoryFormat kernel
// ============================================================================
auto to_memory_format_kernel(const Tensor& input, int format_int, sycl::queue& queue) -> Tensor {
    if (format_int == 0) {
        return contiguous_kernel(input, queue);
    }
    // ChannelsLast: NCHW -> NHWC permutation (dim order: 0,2,3,1)
    if (input.ndim() != 4) {
        throw std::runtime_error("to_memory_format: ChannelsLast requires 4D tensor");
    }
    std::vector<int64_t> perm_dims = {0, 2, 3, 1};
    return permute_kernel(input, perm_dims, queue);
}

// ============================================================================
// Triu kernel - Upper triangular matrix
// ============================================================================

class TriuKernelFloat32;
class TriuKernelFloat64;
class TriuKernelFloat16;
class TriuKernelBFloat16;

auto triu_kernel(const Tensor& input, int64_t diagonal, sycl::queue& queue) -> Tensor {
    if (input.ndim() < 2) {
        throw std::invalid_argument("triu: input must be at least 2-D");
    }

    auto shape_span = input.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
    const int64_t rows = shape[shape.size() - 2];
    const int64_t cols = shape[shape.size() - 1];

    // Calculate batch size (product of all dims except last two)
    int64_t batch_size = 1;
    for (size_t i = 0; i < shape.size() - 2; ++i) {
        batch_size *= shape[i];
    }

    Tensor output(shape, input.dtype(), input.device());
    const int64_t matrix_size = rows * cols;
    const int64_t total = batch_size * matrix_size;

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<TriuKernelFloat32>(sycl::range<1>(total), [=](sycl::id<1> idx) {
            const int64_t mat_idx = idx % matrix_size;
            const int64_t batch_idx = idx / matrix_size;
            const int64_t row = mat_idx / cols;
            const int64_t col = mat_idx % cols;
            out_ptr[idx] = (col >= row + diagonal) ? in_ptr[idx] : 0.0f;
        });
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<TriuKernelFloat64>(sycl::range<1>(total), [=](sycl::id<1> idx) {
            const int64_t mat_idx = idx % matrix_size;
            const int64_t row = mat_idx / cols;
            const int64_t col = mat_idx % cols;
            out_ptr[idx] = (col >= row + diagonal) ? in_ptr[idx] : 0.0;
        });
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<TriuKernelFloat16>(sycl::range<1>(total), [=](sycl::id<1> idx) {
            const int64_t mat_idx = idx % matrix_size;
            const int64_t row = mat_idx / cols;
            const int64_t col = mat_idx % cols;
            out_ptr[idx] = (col >= row + diagonal) ? in_ptr[idx] : sycl::half(0.0f);
        });
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        const uint16_t zero_bf16 = f32_to_bf16(0.0f);
        queue.parallel_for<TriuKernelBFloat16>(sycl::range<1>(total), [=](sycl::id<1> idx) {
            const int64_t mat_idx = idx % matrix_size;
            const int64_t row = mat_idx / cols;
            const int64_t col = mat_idx % cols;
            out_ptr[idx] = (col >= row + diagonal) ? in_ptr[idx] : zero_bf16;
        });
    } else {
        throw std::runtime_error("triu: unsupported dtype");
    }

    return output;
}

// ============================================================================
// Tril kernel - Lower triangular matrix
// ============================================================================

class TrilKernelFloat32;
class TrilKernelFloat64;
class TrilKernelFloat16;
class TrilKernelBFloat16;

auto tril_kernel(const Tensor& input, int64_t diagonal, sycl::queue& queue) -> Tensor {
    if (input.ndim() < 2) {
        throw std::invalid_argument("tril: input must be at least 2-D");
    }

    auto shape_span = input.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
    const int64_t rows = shape[shape.size() - 2];
    const int64_t cols = shape[shape.size() - 1];

    int64_t batch_size = 1;
    for (size_t i = 0; i < shape.size() - 2; ++i) {
        batch_size *= shape[i];
    }

    Tensor output(shape, input.dtype(), input.device());
    const int64_t matrix_size = rows * cols;
    const int64_t total = batch_size * matrix_size;

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<TrilKernelFloat32>(sycl::range<1>(total), [=](sycl::id<1> idx) {
            const int64_t mat_idx = idx % matrix_size;
            const int64_t row = mat_idx / cols;
            const int64_t col = mat_idx % cols;
            out_ptr[idx] = (col <= row + diagonal) ? in_ptr[idx] : 0.0f;
        });
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<TrilKernelFloat64>(sycl::range<1>(total), [=](sycl::id<1> idx) {
            const int64_t mat_idx = idx % matrix_size;
            const int64_t row = mat_idx / cols;
            const int64_t col = mat_idx % cols;
            out_ptr[idx] = (col <= row + diagonal) ? in_ptr[idx] : 0.0;
        });
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<TrilKernelFloat16>(sycl::range<1>(total), [=](sycl::id<1> idx) {
            const int64_t mat_idx = idx % matrix_size;
            const int64_t row = mat_idx / cols;
            const int64_t col = mat_idx % cols;
            out_ptr[idx] = (col <= row + diagonal) ? in_ptr[idx] : sycl::half(0.0f);
        });
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        const uint16_t zero_bf16 = f32_to_bf16(0.0f);
        queue.parallel_for<TrilKernelBFloat16>(sycl::range<1>(total), [=](sycl::id<1> idx) {
            const int64_t mat_idx = idx % matrix_size;
            const int64_t row = mat_idx / cols;
            const int64_t col = mat_idx % cols;
            out_ptr[idx] = (col <= row + diagonal) ? in_ptr[idx] : zero_bf16;
        });
    } else {
        throw std::runtime_error("tril: unsupported dtype");
    }

    return output;
}

// ============================================================================
// Diag kernel - Extract diagonal (2D->1D) or construct diagonal matrix (1D->2D)
// ============================================================================

class DiagExtractKernelFloat32;
class DiagExtractKernelFloat64;
class DiagExtractKernelFloat16;
class DiagExtractKernelBFloat16;
class DiagConstructKernelFloat32;
class DiagConstructKernelFloat64;
class DiagConstructKernelFloat16;
class DiagConstructKernelBFloat16;

auto diag_kernel(const Tensor& input, int64_t diagonal, sycl::queue& queue) -> Tensor {
    const int64_t ndim = input.ndim();

    if (ndim == 2) {
        // Extract diagonal from 2D matrix -> 1D
        auto shape_span = input.shape();
        const int64_t rows = shape_span[0];
        const int64_t cols = shape_span[1];

        // Compute diagonal length
        int64_t diag_start_row = (diagonal >= 0) ? 0 : -diagonal;
        int64_t diag_start_col = (diagonal >= 0) ? diagonal : 0;
        int64_t diag_len = std::min(rows - diag_start_row, cols - diag_start_col);
        if (diag_len <= 0) diag_len = 0;

        Tensor output({diag_len}, input.dtype(), input.device());

        if (diag_len == 0) return output;

        if (input.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(input);
            float* out_ptr = get_data_ptr<float>(output);
            queue.parallel_for<DiagExtractKernelFloat32>(sycl::range<1>(diag_len), [=](sycl::id<1> idx) {
                int64_t r = diag_start_row + static_cast<int64_t>(idx);
                int64_t c = diag_start_col + static_cast<int64_t>(idx);
                out_ptr[idx] = in_ptr[r * cols + c];
            });
        } else if (input.dtype() == DType::Float64) {
            const double* in_ptr = get_data_ptr<const double>(input);
            double* out_ptr = get_data_ptr<double>(output);
            queue.parallel_for<DiagExtractKernelFloat64>(sycl::range<1>(diag_len), [=](sycl::id<1> idx) {
                int64_t r = diag_start_row + static_cast<int64_t>(idx);
                int64_t c = diag_start_col + static_cast<int64_t>(idx);
                out_ptr[idx] = in_ptr[r * cols + c];
            });
        } else if (input.dtype() == DType::Float16) {
            const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
            sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
            queue.parallel_for<DiagExtractKernelFloat16>(sycl::range<1>(diag_len), [=](sycl::id<1> idx) {
                int64_t r = diag_start_row + static_cast<int64_t>(idx);
                int64_t c = diag_start_col + static_cast<int64_t>(idx);
                out_ptr[idx] = in_ptr[r * cols + c];
            });
        } else if (input.dtype() == DType::BFloat16) {
            const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
            uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
            queue.parallel_for<DiagExtractKernelBFloat16>(sycl::range<1>(diag_len), [=](sycl::id<1> idx) {
                int64_t r = diag_start_row + static_cast<int64_t>(idx);
                int64_t c = diag_start_col + static_cast<int64_t>(idx);
                out_ptr[idx] = in_ptr[r * cols + c];
            });
        } else {
            throw std::runtime_error("diag: unsupported dtype");
        }

        return output;
    } else if (ndim == 1) {
        // Construct diagonal matrix from 1D vector -> 2D
        const int64_t n = input.shape()[0];
        const int64_t abs_diag = std::abs(diagonal);
        const int64_t size = n + abs_diag;

        Tensor output({size, size}, input.dtype(), input.device());

        // Zero-fill the output first
        const size_t total_bytes = size * size * input.dtype_size();
        queue.memset(const_cast<void*>(output.data_ptr()), 0, total_bytes);

        if (n == 0) return output;

        const int64_t diag_start_row = (diagonal >= 0) ? 0 : -diagonal;
        const int64_t diag_start_col = (diagonal >= 0) ? diagonal : 0;

        if (input.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(input);
            float* out_ptr = get_data_ptr<float>(output);
            queue.parallel_for<DiagConstructKernelFloat32>(sycl::range<1>(n), [=](sycl::id<1> idx) {
                int64_t r = diag_start_row + static_cast<int64_t>(idx);
                int64_t c = diag_start_col + static_cast<int64_t>(idx);
                out_ptr[r * size + c] = in_ptr[idx];
            });
        } else if (input.dtype() == DType::Float64) {
            const double* in_ptr = get_data_ptr<const double>(input);
            double* out_ptr = get_data_ptr<double>(output);
            queue.parallel_for<DiagConstructKernelFloat64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
                int64_t r = diag_start_row + static_cast<int64_t>(idx);
                int64_t c = diag_start_col + static_cast<int64_t>(idx);
                out_ptr[r * size + c] = in_ptr[idx];
            });
        } else if (input.dtype() == DType::Float16) {
            const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
            sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
            queue.parallel_for<DiagConstructKernelFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
                int64_t r = diag_start_row + static_cast<int64_t>(idx);
                int64_t c = diag_start_col + static_cast<int64_t>(idx);
                out_ptr[r * size + c] = in_ptr[idx];
            });
        } else if (input.dtype() == DType::BFloat16) {
            const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
            uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
            queue.parallel_for<DiagConstructKernelBFloat16>(sycl::range<1>(n), [=](sycl::id<1> idx) {
                int64_t r = diag_start_row + static_cast<int64_t>(idx);
                int64_t c = diag_start_col + static_cast<int64_t>(idx);
                out_ptr[r * size + c] = in_ptr[idx];
            });
        } else {
            throw std::runtime_error("diag: unsupported dtype");
        }

        return output;
    } else {
        throw std::invalid_argument("diag: input must be 1-D or 2-D, got " + std::to_string(ndim) + "-D");
    }
}

// ============================================================================
// Trace kernel - Sum of diagonal elements -> scalar
// ============================================================================

class TraceKernelFloat32;
class TraceKernelFloat64;

auto trace_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    if (input.ndim() != 2) {
        throw std::invalid_argument("trace: input must be 2-D");
    }

    auto shape_span = input.shape();
    const int64_t rows = shape_span[0];
    const int64_t cols = shape_span[1];
    const int64_t diag_len = std::min(rows, cols);

    Tensor output({}, input.dtype(), input.device());

    if (diag_len == 0) {
        // Trace of empty diagonal is 0
        if (input.dtype() == DType::Float32) {
            float zero = 0.0f;
            queue.memcpy(const_cast<void*>(output.data_ptr()), &zero, sizeof(float)).wait();
        } else if (input.dtype() == DType::Float64) {
            double zero = 0.0;
            queue.memcpy(const_cast<void*>(output.data_ptr()), &zero, sizeof(double)).wait();
        } else if (input.dtype() == DType::Float16) {
            sycl::half zero(0.0f);
            queue.memcpy(const_cast<void*>(output.data_ptr()), &zero, sizeof(sycl::half)).wait();
        } else if (input.dtype() == DType::BFloat16) {
            uint16_t zero = f32_to_bf16(0.0f);
            queue.memcpy(const_cast<void*>(output.data_ptr()), &zero, sizeof(uint16_t)).wait();
        }
        return output;
    }

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        auto sum_buf = sycl::malloc_shared<float>(1, queue);
        sum_buf[0] = 0.0f;

        queue.parallel_for(sycl::range<1>(diag_len), sycl::reduction(sum_buf, sycl::plus<float>()),
            [=](sycl::id<1> idx, auto& sum) {
                sum += in_ptr[static_cast<int64_t>(idx) * cols + static_cast<int64_t>(idx)];
            });

        queue.memcpy(const_cast<void*>(output.data_ptr()), sum_buf, sizeof(float)).wait();
        sycl::free(sum_buf, queue);
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        auto sum_buf = sycl::malloc_shared<double>(1, queue);
        sum_buf[0] = 0.0;

        queue.parallel_for(sycl::range<1>(diag_len), sycl::reduction(sum_buf, sycl::plus<double>()),
            [=](sycl::id<1> idx, auto& sum) {
                sum += in_ptr[static_cast<int64_t>(idx) * cols + static_cast<int64_t>(idx)];
            });

        queue.memcpy(const_cast<void*>(output.data_ptr()), sum_buf, sizeof(double)).wait();
        sycl::free(sum_buf, queue);
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        auto sum_buf = sycl::malloc_shared<float>(1, queue);
        sum_buf[0] = 0.0f;

        queue.parallel_for(sycl::range<1>(diag_len), sycl::reduction(sum_buf, sycl::plus<float>()),
            [=](sycl::id<1> idx, auto& sum) {
                sum += static_cast<float>(in_ptr[static_cast<int64_t>(idx) * cols + static_cast<int64_t>(idx)]);
            });

        sycl::half result(sum_buf[0]);
        queue.memcpy(const_cast<void*>(output.data_ptr()), &result, sizeof(sycl::half)).wait();
        sycl::free(sum_buf, queue);
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        auto sum_buf = sycl::malloc_shared<float>(1, queue);
        sum_buf[0] = 0.0f;

        queue.parallel_for(sycl::range<1>(diag_len), sycl::reduction(sum_buf, sycl::plus<float>()),
            [=](sycl::id<1> idx, auto& sum) {
                sum += bf16_to_f32(in_ptr[static_cast<int64_t>(idx) * cols + static_cast<int64_t>(idx)]);
            });

        uint16_t result = f32_to_bf16(sum_buf[0]);
        queue.memcpy(const_cast<void*>(output.data_ptr()), &result, sizeof(uint16_t)).wait();
        sycl::free(sum_buf, queue);
    } else {
        throw std::runtime_error("trace: unsupported dtype");
    }

    return output;
}

// ============================================================================
// Flip kernel - Reverse elements along a dimension
// ============================================================================

class FlipKernelFloat32;
class FlipKernelFloat64;
class FlipKernelFloat16;
class FlipKernelBFloat16;

auto flip_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> Tensor {
    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);

    auto shape_span = in_cont.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
    const int64_t ndim = static_cast<int64_t>(shape.size());

    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("flip: dimension " + std::to_string(dim) + " out of range");
    }

    Tensor output(shape, input.dtype(), input.device());
    const int64_t total = in_cont.numel();

    if (total == 0) return output;

    // Compute outer_size, dim_size, inner_size for the flipped dimension
    const int64_t outer_size = std::accumulate(shape.begin(), shape.begin() + dim, 1LL, std::multiplies<>());
    const int64_t dim_size = shape[dim];
    const int64_t inner_size = std::accumulate(shape.begin() + dim + 1, shape.end(), 1LL, std::multiplies<>());

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(in_cont);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<FlipKernelFloat32>(sycl::range<1>(total), [=](sycl::id<1> idx) {
            const int64_t flat = idx;
            const int64_t outer_idx = flat / (dim_size * inner_size);
            const int64_t remainder = flat % (dim_size * inner_size);
            const int64_t dim_idx = remainder / inner_size;
            const int64_t inner_idx = remainder % inner_size;

            const int64_t flipped_dim_idx = dim_size - 1 - dim_idx;
            const int64_t src_idx = outer_idx * dim_size * inner_size + flipped_dim_idx * inner_size + inner_idx;
            out_ptr[flat] = in_ptr[src_idx];
        });
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(in_cont);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<FlipKernelFloat64>(sycl::range<1>(total), [=](sycl::id<1> idx) {
            const int64_t flat = idx;
            const int64_t outer_idx = flat / (dim_size * inner_size);
            const int64_t remainder = flat % (dim_size * inner_size);
            const int64_t dim_idx = remainder / inner_size;
            const int64_t inner_idx = remainder % inner_size;

            const int64_t flipped_dim_idx = dim_size - 1 - dim_idx;
            const int64_t src_idx = outer_idx * dim_size * inner_size + flipped_dim_idx * inner_size + inner_idx;
            out_ptr[flat] = in_ptr[src_idx];
        });
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<FlipKernelFloat16>(sycl::range<1>(total), [=](sycl::id<1> idx) {
            const int64_t flat = idx;
            const int64_t outer_idx = flat / (dim_size * inner_size);
            const int64_t remainder = flat % (dim_size * inner_size);
            const int64_t dim_idx = remainder / inner_size;
            const int64_t inner_idx = remainder % inner_size;

            const int64_t flipped_dim_idx = dim_size - 1 - dim_idx;
            const int64_t src_idx = outer_idx * dim_size * inner_size + flipped_dim_idx * inner_size + inner_idx;
            out_ptr[flat] = in_ptr[src_idx];
        });
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<FlipKernelBFloat16>(sycl::range<1>(total), [=](sycl::id<1> idx) {
            const int64_t flat = idx;
            const int64_t outer_idx = flat / (dim_size * inner_size);
            const int64_t remainder = flat % (dim_size * inner_size);
            const int64_t dim_idx = remainder / inner_size;
            const int64_t inner_idx = remainder % inner_size;

            const int64_t flipped_dim_idx = dim_size - 1 - dim_idx;
            const int64_t src_idx = outer_idx * dim_size * inner_size + flipped_dim_idx * inner_size + inner_idx;
            out_ptr[flat] = in_ptr[src_idx];
        });
    } else {
        throw std::runtime_error("flip: unsupported dtype");
    }

    return output;
}

// ============================================================================
// repeat_interleave — repeat each element along a dimension
// ============================================================================

class RepeatInterleaveScalarF32;
class RepeatInterleaveScalarF64;
class RepeatInterleaveScalarF16;
class RepeatInterleaveScalarBF16;
class RepeatInterleaveScalarI32;
class RepeatInterleaveScalarI64;

auto repeat_interleave_scalar_kernel(const Tensor& input, int64_t repeats, int64_t dim,
                                     sycl::queue& queue) -> Tensor {
    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);
    auto shape = in_cont.shape();
    int64_t ndim = shape.size();

    int64_t in_dim_size = shape[dim];
    int64_t out_dim_size = in_dim_size * repeats;

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[dim] = out_dim_size;

    Tensor output(out_shape, in_cont.dtype(), in_cont.device());

    int64_t total = 1;
    for (auto s : out_shape) total *= s;
    if (total == 0) return output;

    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) inner_size *= shape[d];

    if (in_cont.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(in_cont);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<RepeatInterleaveScalarF32>(sycl::range<1>(total), [=](sycl::id<1> idx) {
            int64_t i = idx;
            int64_t inner_idx = i % inner_size;
            int64_t out_dim_idx = (i / inner_size) % out_dim_size;
            int64_t outer_idx = i / (inner_size * out_dim_size);
            int64_t src_dim_idx = out_dim_idx / repeats;
            int64_t src_idx = (outer_idx * in_dim_size + src_dim_idx) * inner_size + inner_idx;
            out_ptr[i] = in_ptr[src_idx];
        });
    } else if (in_cont.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(in_cont);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<RepeatInterleaveScalarF64>(sycl::range<1>(total), [=](sycl::id<1> idx) {
            int64_t i = idx;
            int64_t inner_idx = i % inner_size;
            int64_t out_dim_idx = (i / inner_size) % out_dim_size;
            int64_t outer_idx = i / (inner_size * out_dim_size);
            int64_t src_dim_idx = out_dim_idx / repeats;
            int64_t src_idx = (outer_idx * in_dim_size + src_dim_idx) * inner_size + inner_idx;
            out_ptr[i] = in_ptr[src_idx];
        });
    } else if (in_cont.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<RepeatInterleaveScalarF16>(sycl::range<1>(total), [=](sycl::id<1> idx) {
            int64_t i = idx;
            int64_t inner_idx = i % inner_size;
            int64_t out_dim_idx = (i / inner_size) % out_dim_size;
            int64_t outer_idx = i / (inner_size * out_dim_size);
            int64_t src_dim_idx = out_dim_idx / repeats;
            int64_t src_idx = (outer_idx * in_dim_size + src_dim_idx) * inner_size + inner_idx;
            out_ptr[i] = in_ptr[src_idx];
        });
    } else if (in_cont.dtype() == DType::Int32) {
        const int32_t* in_ptr = get_data_ptr<const int32_t>(in_cont);
        int32_t* out_ptr = get_data_ptr<int32_t>(output);
        queue.parallel_for<RepeatInterleaveScalarI32>(sycl::range<1>(total), [=](sycl::id<1> idx) {
            int64_t i = idx;
            int64_t inner_idx = i % inner_size;
            int64_t out_dim_idx = (i / inner_size) % out_dim_size;
            int64_t outer_idx = i / (inner_size * out_dim_size);
            int64_t src_dim_idx = out_dim_idx / repeats;
            int64_t src_idx = (outer_idx * in_dim_size + src_dim_idx) * inner_size + inner_idx;
            out_ptr[i] = in_ptr[src_idx];
        });
    } else if (in_cont.dtype() == DType::Int64) {
        const int64_t* in_ptr = get_data_ptr<const int64_t>(in_cont);
        int64_t* out_ptr = get_data_ptr<int64_t>(output);
        queue.parallel_for<RepeatInterleaveScalarI64>(sycl::range<1>(total), [=](sycl::id<1> idx) {
            int64_t i = idx;
            int64_t inner_idx = i % inner_size;
            int64_t out_dim_idx = (i / inner_size) % out_dim_size;
            int64_t outer_idx = i / (inner_size * out_dim_size);
            int64_t src_dim_idx = out_dim_idx / repeats;
            int64_t src_idx = (outer_idx * in_dim_size + src_dim_idx) * inner_size + inner_idx;
            out_ptr[i] = in_ptr[src_idx];
        });
    } else {
        throw std::runtime_error("repeat_interleave: unsupported dtype");
    }

    return output;
}

class RepeatInterleaveTensorF32;
class RepeatInterleaveTensorF64;
class RepeatInterleaveTensorI32;
class RepeatInterleaveTensorI64;

// SYCL kernel name tags for repeat_interleave cast
class RICastToI64FromI32;
class RICastToI64FromF32;
class RICastToI64FromF64;
class RIPrefixSumI64;

auto repeat_interleave_tensor_kernel(const Tensor& input, const Tensor& repeats_tensor,
                                     int64_t dim, sycl::queue& queue) -> Tensor {
    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);
    auto shape = in_cont.shape();
    int64_t ndim = shape.size();
    int64_t in_dim_size = shape[dim];

    // Convert repeats to int64 on device (no CPU roundtrip)
    int64_t* d_repeats_i64 = sycl::malloc_device<int64_t>(in_dim_size, queue);
    auto repeats_cont = repeats_tensor.is_contiguous() ? repeats_tensor : repeats_tensor.contiguous();

    if (repeats_cont.dtype() == DType::Int64) {
        queue.memcpy(d_repeats_i64, repeats_cont.data<int64_t>(),
                     in_dim_size * sizeof(int64_t)).wait();
    } else if (repeats_cont.dtype() == DType::Int32) {
        const int32_t* src = repeats_cont.data<int32_t>();
        queue.parallel_for<RICastToI64FromI32>(sycl::range<1>(in_dim_size), [=](sycl::id<1> i) {
            d_repeats_i64[i] = static_cast<int64_t>(src[i]);
        }).wait();
    } else if (repeats_cont.dtype() == DType::Float32) {
        const float* src = repeats_cont.data<float>();
        queue.parallel_for<RICastToI64FromF32>(sycl::range<1>(in_dim_size), [=](sycl::id<1> i) {
            d_repeats_i64[i] = static_cast<int64_t>(src[i]);
        }).wait();
    } else if (repeats_cont.dtype() == DType::Float64) {
        const double* src = repeats_cont.data<double>();
        queue.parallel_for<RICastToI64FromF64>(sycl::range<1>(in_dim_size), [=](sycl::id<1> i) {
            d_repeats_i64[i] = static_cast<int64_t>(src[i]);
        }).wait();
    } else {
        sycl::free(d_repeats_i64, queue);
        throw std::runtime_error("repeat_interleave: unsupported repeats dtype");
    }

    // Compute exclusive prefix sum on device
    int64_t* d_prefix = sycl::malloc_device<int64_t>(in_dim_size + 1, queue);
    int64_t N = in_dim_size;
    int64_t* rp = d_repeats_i64;
    int64_t* pfx = d_prefix;
    queue.single_task<RIPrefixSumI64>([=]() {
        pfx[0] = 0;
        for (int64_t i = 0; i < N; ++i) {
            pfx[i + 1] = pfx[i] + rp[i];
        }
    }).wait();

    // Read only the total from device
    int64_t out_dim_size = 0;
    queue.memcpy(&out_dim_size, d_prefix + in_dim_size, sizeof(int64_t)).wait();

    sycl::free(d_repeats_i64, queue);

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[dim] = out_dim_size;

    Tensor output(out_shape, in_cont.dtype(), in_cont.device());

    int64_t total = 1;
    for (auto s : out_shape) total *= s;
    if (total == 0) {
        sycl::free(d_prefix, queue);
        return output;
    }

    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) inner_size *= shape[d];

    auto launch_tensor_ri = [&]<typename T, typename KernelName>() {
        const T* in_ptr = get_data_ptr<const T>(in_cont);
        T* out_ptr = get_data_ptr<T>(output);
        const int64_t* pfx = d_prefix;
        queue.parallel_for<KernelName>(sycl::range<1>(total), [=](sycl::id<1> idx) {
            int64_t i = idx;
            int64_t inner_idx = i % inner_size;
            int64_t out_dim_idx2 = (i / inner_size) % out_dim_size;
            int64_t outer_idx = i / (inner_size * out_dim_size);

            // Binary search
            int64_t lo = 0, hi = in_dim_size;
            while (lo < hi) {
                int64_t mid = lo + (hi - lo) / 2;
                if (pfx[mid + 1] <= out_dim_idx2) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            int64_t src_dim_idx = lo;
            int64_t src_idx = (outer_idx * in_dim_size + src_dim_idx) * inner_size + inner_idx;
            out_ptr[i] = in_ptr[src_idx];
        });
    };

    if (in_cont.dtype() == DType::Float32) {
        launch_tensor_ri.template operator()<float, RepeatInterleaveTensorF32>();
    } else if (in_cont.dtype() == DType::Float64) {
        launch_tensor_ri.template operator()<double, RepeatInterleaveTensorF64>();
    } else if (in_cont.dtype() == DType::Int32) {
        launch_tensor_ri.template operator()<int32_t, RepeatInterleaveTensorI32>();
    } else if (in_cont.dtype() == DType::Int64) {
        launch_tensor_ri.template operator()<int64_t, RepeatInterleaveTensorI64>();
    } else {
        sycl::free(d_prefix, queue);
        throw std::runtime_error("repeat_interleave_tensor: unsupported dtype");
    }

    queue.wait_and_throw();
    sycl::free(d_prefix, queue);
    return output;
}

} // namespace oneapi
} // namespace tenzor
