#include "tenzor/core/tensor.hpp"
#include <sycl/sycl.hpp>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace tenzor {
namespace adaptivecpp {

// SYCL Kernel name classes
class CatKernelFloat32 {};
class CatKernelFloat64 {};
class CatKernelFloat16 {};
class CatKernelInt32 {};
class ClampKernelFloat32 {};
class ClampKernelFloat64 {};
class ClampKernelFloat16 {};
class ClampKernelInt32 {};
class SignKernelFloat32 {};
class SignKernelFloat64 {};
class SignKernelFloat16 {};
class SignKernelInt32 {};

// Helper function to get typed pointer from tensor
template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

// Helper to calculate total elements
inline auto calculate_numel(const std::vector<int64_t>& shape) -> int64_t {
    int64_t numel = 1;
    for (auto s : shape) {
        numel *= s;
    }
    return numel;
}

// Helper to calculate strides from shape
inline auto calculate_strides(const std::vector<int64_t>& shape) -> std::vector<int64_t> {
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (int64_t i = shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

/**
 * @brief Concatenate tensors along a specified dimension
 *
 * @param tensors Span of input tensors to concatenate
 * @param dim Dimension along which to concatenate
 * @param queue SYCL queue for kernel execution
 * @return Concatenated output tensor
 *
 * The operation concatenates multiple tensors along the specified dimension.
 * All input tensors must have the same shape except in the concatenation dimension.
 *
 * Example:
 *   A = [[1, 2], [3, 4]]  shape: (2, 2)
 *   B = [[5, 6]]          shape: (1, 2)
 *   cat([A, B], dim=0) = [[1, 2], [3, 4], [5, 6]]  shape: (3, 2)
 */
auto cat_kernel(std::span<const Tensor> tensors, int64_t dim, sycl::queue& queue) -> Tensor {
    // Validate inputs
    if (tensors.empty()) {
        throw std::invalid_argument("cat_kernel requires at least one input tensor");
    }

    if (tensors.size() == 1) {
        // Single tensor, just clone it
        const void* in_ptr = tensors[0].data_ptr();
        const size_t bytes = tensors[0].numel() * tensors[0].dtype_size();

        Tensor output(std::vector<int64_t>(tensors[0].shape().begin(), tensors[0].shape().end()),
                      tensors[0].dtype(), tensors[0].device());
        void* out_ptr = const_cast<void*>(output.data_ptr());
        queue.memcpy(out_ptr, in_ptr, bytes).wait();
        return output;
    }

    // Get reference shape and validate all tensors
    auto ref_shape_span = tensors[0].shape();
    std::vector<int64_t> ref_shape(ref_shape_span.begin(), ref_shape_span.end());
    const size_t ndim = ref_shape.size();
    const DType dtype = tensors[0].dtype();
    const Device device = tensors[0].device();

    // Normalize dimension
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= static_cast<int64_t>(ndim)) {
        throw std::invalid_argument("cat_kernel: invalid dimension " + std::to_string(dim));
    }

    // Validate all tensors have compatible shapes and same dtype
    for (size_t i = 1; i < tensors.size(); ++i) {
        auto shape_span = tensors[i].shape();
        if (shape_span.size() != ndim) {
            throw std::invalid_argument("cat_kernel: all tensors must have same number of dimensions");
        }
        if (tensors[i].dtype() != dtype) {
            throw std::invalid_argument("cat_kernel: all tensors must have same dtype");
        }

        // Check all dimensions except concat dimension match
        for (size_t d = 0; d < ndim; ++d) {
            if (d != static_cast<size_t>(dim) && shape_span[d] != ref_shape[d]) {
                throw std::invalid_argument("cat_kernel: tensor shapes must match except in concat dimension");
            }
        }
    }

    // Compute output shape
    std::vector<int64_t> out_shape = ref_shape;
    out_shape[dim] = 0;
    for (const auto& tensor : tensors) {
        out_shape[dim] += tensor.shape()[dim];
    }

    // Create output tensor
    Tensor output(out_shape, dtype, device);

    // Calculate strides for efficient indexing
    auto out_strides = calculate_strides(out_shape);

    // For each input tensor, copy its data to the appropriate offset in output
    int64_t offset_in_concat_dim = 0;

    for (const auto& tensor : tensors) {
        auto tensor_shape = tensor.shape();
        const int64_t tensor_size_in_dim = tensor_shape[dim];

        // Calculate how many elements to copy
        // Elements per "slice" in the concatenation dimension
        int64_t elements_per_slice = 1;
        for (size_t d = dim + 1; d < ndim; ++d) {
            elements_per_slice *= tensor_shape[d];
        }

        // Number of slices to copy
        int64_t num_slices = 1;
        for (size_t d = 0; d < static_cast<size_t>(dim); ++d) {
            num_slices *= tensor_shape[d];
        }

        const int64_t tensor_numel = tensor.numel();

        // Dispatch based on dtype
        if (dtype == DType::Float32) {
            const float* src_ptr = get_data_ptr<const float>(tensor);
            float* dst_ptr = get_data_ptr<float>(output);

            // Calculate stride sizes
            const int64_t src_slice_stride = elements_per_slice * tensor_size_in_dim;
            const int64_t dst_slice_stride = elements_per_slice * out_shape[dim];
            const int64_t current_offset = offset_in_concat_dim;

            queue.parallel_for<CatKernelFloat32>(
                sycl::range<1>(num_slices * tensor_size_in_dim * elements_per_slice),
                [=](sycl::id<1> idx) {
                    // Decompose index into slice, concat_idx, and element
                    const int64_t flat_idx = idx[0];
                    const int64_t slice_idx = flat_idx / src_slice_stride;
                    const int64_t remainder = flat_idx % src_slice_stride;
                    const int64_t concat_idx = remainder / elements_per_slice;
                    const int64_t elem_idx = remainder % elements_per_slice;

                    // Source index
                    const int64_t src_idx = slice_idx * src_slice_stride + concat_idx * elements_per_slice + elem_idx;

                    // Destination index
                    const int64_t dst_idx = slice_idx * dst_slice_stride +
                                           (current_offset + concat_idx) * elements_per_slice +
                                           elem_idx;

                    dst_ptr[dst_idx] = src_ptr[src_idx];
                }
            ).wait();
        }
        else if (dtype == DType::Float64) {
            const double* src_ptr = get_data_ptr<const double>(tensor);
            double* dst_ptr = get_data_ptr<double>(output);

            const int64_t src_slice_stride = elements_per_slice * tensor_size_in_dim;
            const int64_t dst_slice_stride = elements_per_slice * out_shape[dim];
            const int64_t current_offset = offset_in_concat_dim;

            queue.parallel_for<CatKernelFloat64>(
                sycl::range<1>(num_slices * tensor_size_in_dim * elements_per_slice),
                [=](sycl::id<1> idx) {
                    const int64_t flat_idx = idx[0];
                    const int64_t slice_idx = flat_idx / src_slice_stride;
                    const int64_t remainder = flat_idx % src_slice_stride;
                    const int64_t concat_idx = remainder / elements_per_slice;
                    const int64_t elem_idx = remainder % elements_per_slice;

                    const int64_t src_idx = slice_idx * src_slice_stride + concat_idx * elements_per_slice + elem_idx;
                    const int64_t dst_idx = slice_idx * dst_slice_stride +
                                           (current_offset + concat_idx) * elements_per_slice +
                                           elem_idx;

                    dst_ptr[dst_idx] = src_ptr[src_idx];
                }
            ).wait();
        }
        else if (dtype == DType::Float16) {
            const sycl::half* src_ptr = get_data_ptr<const sycl::half>(tensor);
            sycl::half* dst_ptr = get_data_ptr<sycl::half>(output);

            const int64_t src_slice_stride = elements_per_slice * tensor_size_in_dim;
            const int64_t dst_slice_stride = elements_per_slice * out_shape[dim];
            const int64_t current_offset = offset_in_concat_dim;

            queue.parallel_for<CatKernelFloat16>(
                sycl::range<1>(num_slices * tensor_size_in_dim * elements_per_slice),
                [=](sycl::id<1> idx) {
                    const int64_t flat_idx = idx[0];
                    const int64_t slice_idx = flat_idx / src_slice_stride;
                    const int64_t remainder = flat_idx % src_slice_stride;
                    const int64_t concat_idx = remainder / elements_per_slice;
                    const int64_t elem_idx = remainder % elements_per_slice;

                    const int64_t src_idx = slice_idx * src_slice_stride + concat_idx * elements_per_slice + elem_idx;
                    const int64_t dst_idx = slice_idx * dst_slice_stride +
                                           (current_offset + concat_idx) * elements_per_slice +
                                           elem_idx;

                    dst_ptr[dst_idx] = src_ptr[src_idx];
                }
            ).wait();
        }
        else if (dtype == DType::Int32) {
            const int32_t* src_ptr = get_data_ptr<const int32_t>(tensor);
            int32_t* dst_ptr = get_data_ptr<int32_t>(output);

            const int64_t src_slice_stride = elements_per_slice * tensor_size_in_dim;
            const int64_t dst_slice_stride = elements_per_slice * out_shape[dim];
            const int64_t current_offset = offset_in_concat_dim;

            queue.parallel_for<CatKernelInt32>(
                sycl::range<1>(num_slices * tensor_size_in_dim * elements_per_slice),
                [=](sycl::id<1> idx) {
                    const int64_t flat_idx = idx[0];
                    const int64_t slice_idx = flat_idx / src_slice_stride;
                    const int64_t remainder = flat_idx % src_slice_stride;
                    const int64_t concat_idx = remainder / elements_per_slice;
                    const int64_t elem_idx = remainder % elements_per_slice;

                    const int64_t src_idx = slice_idx * src_slice_stride + concat_idx * elements_per_slice + elem_idx;
                    const int64_t dst_idx = slice_idx * dst_slice_stride +
                                           (current_offset + concat_idx) * elements_per_slice +
                                           elem_idx;

                    dst_ptr[dst_idx] = src_ptr[src_idx];
                }
            ).wait();
        }
        else {
            throw std::runtime_error("cat_kernel: unsupported dtype");
        }

        offset_in_concat_dim += tensor_size_in_dim;
    }

    return output;
}

/**
 * @brief Clamp tensor values to a specified range [min_val, max_val]
 *
 * @param input Input tensor
 * @param min_val Minimum value (values below this are set to min_val)
 * @param max_val Maximum value (values above this are set to max_val)
 * @param queue SYCL queue for kernel execution
 * @return Output tensor with clamped values
 *
 * Each element x in the input is transformed to:
 *   clamp(x, min, max) = min if x < min
 *                      = max if x > max
 *                      = x   otherwise
 *
 * Example:
 *   input = [-2.0, -1.0, 0.0, 1.0, 2.0]
 *   clamp(input, -1.0, 1.0) = [-1.0, -1.0, 0.0, 1.0, 1.0]
 */
auto clamp_kernel(const Tensor& input, float min_val, float max_val, sycl::queue& queue) -> Tensor {
    // Validate min <= max
    if (min_val > max_val) {
        throw std::invalid_argument("clamp_kernel: min_val must be <= max_val");
    }

    // Create output tensor
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    // Dispatch based on dtype
    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<ClampKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            const float val = in_ptr[idx];
            out_ptr[idx] = sycl::fmin(sycl::fmax(val, min_val), max_val);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        const double min_d = static_cast<double>(min_val);
        const double max_d = static_cast<double>(max_val);

        queue.parallel_for<ClampKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            const double val = in_ptr[idx];
            out_ptr[idx] = sycl::fmin(sycl::fmax(val, min_d), max_d);
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<ClampKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            const float val = static_cast<float>(in_ptr[idx]);
            out_ptr[idx] = sycl::half(sycl::fmin(sycl::fmax(val, min_val), max_val));
        }).wait();
    }
    else if (input.dtype() == DType::Int32) {
        const int32_t* in_ptr = get_data_ptr<const int32_t>(input);
        int32_t* out_ptr = get_data_ptr<int32_t>(output);

        const int32_t min_i = static_cast<int32_t>(min_val);
        const int32_t max_i = static_cast<int32_t>(max_val);

        queue.parallel_for<ClampKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            const int32_t val = in_ptr[idx];
            out_ptr[idx] = sycl::min(sycl::max(val, min_i), max_i);
        }).wait();
    }
    else {
        throw std::runtime_error("clamp_kernel: unsupported dtype");
    }

    return output;
}

/**
 * @brief Compute the sign of tensor elements
 *
 * @param input Input tensor
 * @param queue SYCL queue for kernel execution
 * @return Output tensor with sign values
 *
 * Each element x in the input is transformed to:
 *   sign(x) = -1 if x < 0
 *           =  0 if x == 0
 *           = +1 if x > 0
 *
 * Note: For floating-point, handles -0.0 correctly (returns 0.0)
 *
 * Example:
 *   input = [-2.5, -0.0, 0.0, 3.7]
 *   sign(input) = [-1.0, 0.0, 0.0, 1.0]
 */
auto sign_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    // Create output tensor with same shape and dtype
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    // Dispatch based on dtype
    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<SignKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            const float val = in_ptr[idx];
            // Use copysign for IEEE 754 correctness, handles -0.0 properly
            if (val == 0.0f) {
                out_ptr[idx] = 0.0f;
            } else if (val > 0.0f) {
                out_ptr[idx] = 1.0f;
            } else {
                out_ptr[idx] = -1.0f;
            }
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<SignKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            const double val = in_ptr[idx];
            if (val == 0.0) {
                out_ptr[idx] = 0.0;
            } else if (val > 0.0) {
                out_ptr[idx] = 1.0;
            } else {
                out_ptr[idx] = -1.0;
            }
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<SignKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            const float val = static_cast<float>(in_ptr[idx]);
            if (val == 0.0f) {
                out_ptr[idx] = sycl::half(0.0f);
            } else if (val > 0.0f) {
                out_ptr[idx] = sycl::half(1.0f);
            } else {
                out_ptr[idx] = sycl::half(-1.0f);
            }
        }).wait();
    }
    else if (input.dtype() == DType::Int32) {
        const int32_t* in_ptr = get_data_ptr<const int32_t>(input);
        int32_t* out_ptr = get_data_ptr<int32_t>(output);

        queue.parallel_for<SignKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            const int32_t val = in_ptr[idx];
            if (val == 0) {
                out_ptr[idx] = 0;
            } else if (val > 0) {
                out_ptr[idx] = 1;
            } else {
                out_ptr[idx] = -1;
            }
        }).wait();
    }
    else {
        throw std::runtime_error("sign_kernel: unsupported dtype");
    }

    return output;
}

} // namespace adaptivecpp
} // namespace tenzor
