#include "tenzor/core/tensor.hpp"
#include <sycl/sycl.hpp>
#include <cstring>
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

// BFloat16 conversion helpers (BFloat16 stored as uint16_t)
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

// Helper function to get typed pointer from tensor
template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
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

// Helper to compute flat index from multi-dimensional indices
inline auto compute_flat_index(const std::vector<int64_t>& indices,
                                const std::vector<int64_t>& strides) -> int64_t {
    int64_t flat_idx = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        flat_idx += indices[i] * strides[i];
    }
    return flat_idx;
}

// Reshape kernel - just validates and creates view (no data copy)
auto reshape_kernel(const Tensor& input, const std::vector<int64_t>& new_shape, sycl::queue& queue) -> Tensor {
    // Calculate total elements
    int64_t input_numel = input.numel();
    int64_t output_numel = 1;
    for (auto dim : new_shape) {
        output_numel *= dim;
    }

    if (input_numel != output_numel) {
        throw std::invalid_argument("Reshape: total number of elements must remain constant");
    }

    // Ensure input is contiguous before memcpy (non-contiguous layout would corrupt data)
    Tensor src = input.is_contiguous() ? input : contiguous_kernel(input, queue);

    Tensor output(new_shape, src.dtype(), src.device());

    // Simple memory copy since data layout is preserved - works for all dtypes
    const size_t bytes = input_numel * src.dtype_size();
    const void* in_ptr = src.data_ptr();
    void* out_ptr = const_cast<void*>(output.data_ptr());
    queue.memcpy(out_ptr, in_ptr, bytes);

    return output;
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
        std::vector<float> host_data(numel, 1.0f);
        float* device_ptr = get_data_ptr<float>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(float)).wait();
    }
    else if (dtype == DType::Float64) {
        std::vector<double> host_data(numel, 1.0);
        double* device_ptr = get_data_ptr<double>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(double)).wait();
    }
    else if (dtype == DType::Float16) {
        // sycl::half doesn't work well with std::vector, so fill via kernel
        sycl::half* device_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            device_ptr[i] = sycl::half(1.0f);
        });
    }
    else if (dtype == DType::BFloat16) {
        // BFloat16: convert 1.0f to bf16 representation, fill via kernel
        const uint16_t one_bf16 = f32_to_bf16(1.0f);
        uint16_t* device_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<OnesKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            device_ptr[i] = one_bf16;
        });
    }
    else if (dtype == DType::Int32) {
        std::vector<int32_t> host_data(numel, 1);
        int32_t* device_ptr = get_data_ptr<int32_t>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(int32_t)).wait();
    }
    else if (dtype == DType::Int64) {
        std::vector<int64_t> host_data(numel, 1);
        int64_t* device_ptr = get_data_ptr<int64_t>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(int64_t)).wait();
    }
    else if (dtype == DType::Int8) {
        std::vector<int8_t> host_data(numel, 1);
        int8_t* device_ptr = get_data_ptr<int8_t>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(int8_t)).wait();
    }
    else if (dtype == DType::Int16) {
        std::vector<int16_t> host_data(numel, 1);
        int16_t* device_ptr = get_data_ptr<int16_t>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(int16_t)).wait();
    }
    else if (dtype == DType::UInt8) {
        std::vector<uint8_t> host_data(numel, 1);
        uint8_t* device_ptr = get_data_ptr<uint8_t>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(uint8_t)).wait();
    }
    else if (dtype == DType::UInt16) {
        std::vector<uint16_t> host_data(numel, 1);
        uint16_t* device_ptr = get_data_ptr<uint16_t>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(uint16_t)).wait();
    }
    else if (dtype == DType::UInt32) {
        std::vector<uint32_t> host_data(numel, 1);
        uint32_t* device_ptr = get_data_ptr<uint32_t>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(uint32_t)).wait();
    }
    else if (dtype == DType::UInt64) {
        std::vector<uint64_t> host_data(numel, 1);
        uint64_t* device_ptr = get_data_ptr<uint64_t>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(uint64_t)).wait();
    }
    else if (dtype == DType::Bool) {
        std::vector<bool> host_data(numel, true);
        bool* device_ptr = get_data_ptr<bool>(output);
        // std::vector<bool> is special, need to copy element by element
        std::vector<uint8_t> temp_data(numel, 1);
        queue.memcpy(device_ptr, temp_data.data(), numel * sizeof(bool)).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for ones");
    }

    return output;
}

// Full kernel - fill with specific value using memcpy
auto full_kernel(const std::vector<int64_t>& shape, float value, DType dtype, Device device, sycl::queue& queue) -> Tensor {
    Tensor output(shape, dtype, device);
    const int64_t numel = output.numel();

    if (dtype == DType::Float32) {
        std::vector<float> host_data(numel, value);
        float* device_ptr = get_data_ptr<float>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(float)).wait();
    }
    else if (dtype == DType::Float64) {
        const double value_d = static_cast<double>(value);
        std::vector<double> host_data(numel, value_d);
        double* device_ptr = get_data_ptr<double>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(double)).wait();
    }
    else if (dtype == DType::Float16) {
        const sycl::half value_h(value);
        sycl::half* device_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            device_ptr[i] = value_h;
        });
    }
    else if (dtype == DType::BFloat16) {
        const uint16_t value_bf16 = f32_to_bf16(value);
        uint16_t* device_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<FullKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            device_ptr[i] = value_bf16;
        });
    }
    else if (dtype == DType::Int32) {
        const int32_t value_i = static_cast<int32_t>(value);
        std::vector<int32_t> host_data(numel, value_i);
        int32_t* device_ptr = get_data_ptr<int32_t>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(int32_t)).wait();
    }
    else if (dtype == DType::Int64) {
        const int64_t value_i = static_cast<int64_t>(value);
        std::vector<int64_t> host_data(numel, value_i);
        int64_t* device_ptr = get_data_ptr<int64_t>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(int64_t)).wait();
    }
    else if (dtype == DType::Int8) {
        const int8_t value_i = static_cast<int8_t>(value);
        std::vector<int8_t> host_data(numel, value_i);
        int8_t* device_ptr = get_data_ptr<int8_t>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(int8_t)).wait();
    }
    else if (dtype == DType::Int16) {
        const int16_t value_i = static_cast<int16_t>(value);
        std::vector<int16_t> host_data(numel, value_i);
        int16_t* device_ptr = get_data_ptr<int16_t>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(int16_t)).wait();
    }
    else if (dtype == DType::UInt8) {
        const uint8_t value_i = static_cast<uint8_t>(value);
        std::vector<uint8_t> host_data(numel, value_i);
        uint8_t* device_ptr = get_data_ptr<uint8_t>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(uint8_t)).wait();
    }
    else if (dtype == DType::UInt16) {
        const uint16_t value_i = static_cast<uint16_t>(value);
        std::vector<uint16_t> host_data(numel, value_i);
        uint16_t* device_ptr = get_data_ptr<uint16_t>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(uint16_t)).wait();
    }
    else if (dtype == DType::UInt32) {
        const uint32_t value_i = static_cast<uint32_t>(value);
        std::vector<uint32_t> host_data(numel, value_i);
        uint32_t* device_ptr = get_data_ptr<uint32_t>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(uint32_t)).wait();
    }
    else if (dtype == DType::UInt64) {
        const uint64_t value_i = static_cast<uint64_t>(value);
        std::vector<uint64_t> host_data(numel, value_i);
        uint64_t* device_ptr = get_data_ptr<uint64_t>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(uint64_t)).wait();
    }
    else if (dtype == DType::Bool) {
        const uint8_t value_b = (value != 0.0f) ? 1 : 0;
        std::vector<uint8_t> host_data(numel, value_b);
        bool* device_ptr = get_data_ptr<bool>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(bool)).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for full");
    }

    return output;
}

// Fill kernel - fill existing tensor with value using memcpy
auto fill_kernel(const Tensor& tensor, float value, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(tensor.shape().begin(), tensor.shape().end()),
                  tensor.dtype(), tensor.device());

    const int64_t numel = tensor.numel();

    if (tensor.dtype() == DType::Float32) {
        std::vector<float> host_data(numel, value);
        float* device_ptr = get_data_ptr<float>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(float)).wait();
    }
    else if (tensor.dtype() == DType::Float64) {
        const double value_d = static_cast<double>(value);
        std::vector<double> host_data(numel, value_d);
        double* device_ptr = get_data_ptr<double>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(double)).wait();
    }
    else if (tensor.dtype() == DType::Float16) {
        const sycl::half value_h(value);
        sycl::half* device_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<FillKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            device_ptr[i] = value_h;
        });
    }
    else if (tensor.dtype() == DType::BFloat16) {
        const uint16_t value_bf16 = f32_to_bf16(value);
        uint16_t* device_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<FillKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            device_ptr[i] = value_bf16;
        });
    }
    else if (tensor.dtype() == DType::Int32) {
        const int32_t value_i = static_cast<int32_t>(value);
        std::vector<int32_t> host_data(numel, value_i);
        int32_t* device_ptr = get_data_ptr<int32_t>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(int32_t)).wait();
    }
    else if (tensor.dtype() == DType::Int64) {
        const int64_t value_i = static_cast<int64_t>(value);
        std::vector<int64_t> host_data(numel, value_i);
        int64_t* device_ptr = get_data_ptr<int64_t>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(int64_t)).wait();
    }
    else if (tensor.dtype() == DType::Int8) {
        const int8_t value_i = static_cast<int8_t>(value);
        std::vector<int8_t> host_data(numel, value_i);
        int8_t* device_ptr = get_data_ptr<int8_t>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(int8_t)).wait();
    }
    else if (tensor.dtype() == DType::Int16) {
        const int16_t value_i = static_cast<int16_t>(value);
        std::vector<int16_t> host_data(numel, value_i);
        int16_t* device_ptr = get_data_ptr<int16_t>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(int16_t)).wait();
    }
    else if (tensor.dtype() == DType::UInt8) {
        const uint8_t value_i = static_cast<uint8_t>(value);
        std::vector<uint8_t> host_data(numel, value_i);
        uint8_t* device_ptr = get_data_ptr<uint8_t>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(uint8_t)).wait();
    }
    else if (tensor.dtype() == DType::UInt16) {
        const uint16_t value_i = static_cast<uint16_t>(value);
        std::vector<uint16_t> host_data(numel, value_i);
        uint16_t* device_ptr = get_data_ptr<uint16_t>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(uint16_t)).wait();
    }
    else if (tensor.dtype() == DType::UInt32) {
        const uint32_t value_i = static_cast<uint32_t>(value);
        std::vector<uint32_t> host_data(numel, value_i);
        uint32_t* device_ptr = get_data_ptr<uint32_t>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(uint32_t)).wait();
    }
    else if (tensor.dtype() == DType::UInt64) {
        const uint64_t value_i = static_cast<uint64_t>(value);
        std::vector<uint64_t> host_data(numel, value_i);
        uint64_t* device_ptr = get_data_ptr<uint64_t>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(uint64_t)).wait();
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
            // General strided copy via host
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
    return repeat_kernel(input, reps, queue);
}

// ============================================================================
// Take kernel - flattened index selection
// ============================================================================

class TakeKernelFloat32;
class TakeKernelFloat64;
class TakeKernelFloat16;
class TakeKernelBFloat16;

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
    else {
        // Generic byte copy via host for other dtypes
        size_t elem_size = in_cont.dtype_size();
        int64_t input_numel = in_cont.numel();

        std::vector<uint8_t> in_host(input_numel * elem_size);
        std::vector<uint8_t> out_host(num_indices * elem_size);
        queue.memcpy(in_host.data(), in_cont.data_ptr(), in_host.size());
        // No wait — in-order queue guarantees next memcpy completes after this one

        // Read indices to host
        if (is_int64) {
            std::vector<int64_t> idx_host(num_indices);
            queue.memcpy(idx_host.data(), idx_cont.data_ptr(), num_indices * sizeof(int64_t)).wait();
            for (int64_t i = 0; i < num_indices; ++i) {
                std::memcpy(out_host.data() + i * elem_size,
                           in_host.data() + idx_host[i] * elem_size, elem_size);
            }
        } else {
            std::vector<int32_t> idx_host(num_indices);
            queue.memcpy(idx_host.data(), idx_cont.data_ptr(), num_indices * sizeof(int32_t)).wait();
            for (int64_t i = 0; i < num_indices; ++i) {
                std::memcpy(out_host.data() + i * elem_size,
                           in_host.data() + idx_host[i] * elem_size, elem_size);
            }
        }

        queue.memcpy(const_cast<void*>(output.data_ptr()), out_host.data(), out_host.size()).wait();
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

auto unfold_kernel(const Tensor& input, int64_t kernel_size, int64_t stride,
                    int64_t padding, int64_t dilation, sycl::queue& queue) -> Tensor {
    // 2D unfold: input [N, C, H, W] -> output [N, C*kH*kW, L]
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::runtime_error("unfold_kernel: expected 4D input [N, C, H, W]");
    }

    int64_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    int64_t H_out = (H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int64_t W_out = (W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int64_t L = H_out * W_out;
    int64_t channels_col = C * kernel_size * kernel_size;

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
                    int64_t kw = idx % kernel_size;
                    idx /= kernel_size;
                    int64_t kh = idx % kernel_size;
                    int64_t c = idx / kernel_size;
                    int64_t h_in = h_o * stride - padding + kh * dilation;
                    int64_t w_in = w_o * stride - padding + kw * dilation;
                    batch_out[index] = (h_in >= 0 && w_in >= 0 && h_in < H && w_in < W) ?
                        batch_in[(c * H + h_in) * W + w_in] : 0.0f;
                });
        }
        queue.wait();
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
                    int64_t kw = idx % kernel_size;
                    idx /= kernel_size;
                    int64_t kh = idx % kernel_size;
                    int64_t c = idx / kernel_size;
                    int64_t h_in = h_o * stride - padding + kh * dilation;
                    int64_t w_in = w_o * stride - padding + kw * dilation;
                    batch_out[index] = (h_in >= 0 && w_in >= 0 && h_in < H && w_in < W) ?
                        batch_in[(c * H + h_in) * W + w_in] : 0.0;
                });
        }
        queue.wait();
    } else {
        throw std::runtime_error("unfold_kernel: unsupported dtype (expected Float32 or Float64)");
    }

    return output;
}

// ============================================================================
// Fold kernel - inverse of unfold (col2im)
// ============================================================================

auto fold_kernel(const Tensor& input, const std::vector<int64_t>& output_size,
                  int64_t kernel_size, int64_t stride, int64_t padding,
                  int64_t dilation, sycl::queue& queue) -> Tensor {
    auto shape = input.shape();
    if (shape.size() != 3) {
        throw std::runtime_error("fold_kernel: expected 3D input [N, C*kH*kW, L]");
    }

    int64_t N = shape[0];
    int64_t channels_col = shape[1];
    int64_t L = shape[2];
    int64_t H_out = output_size[0];
    int64_t W_out = output_size[1];
    int64_t C = channels_col / (kernel_size * kernel_size);

    Tensor output({N, C, H_out, W_out}, input.dtype(), input.device());

    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);

    int64_t H_col = (H_out + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int64_t W_col = (W_out + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int64_t col_size_per_batch = channels_col * L;
    int64_t im_size_per_batch = C * H_out * W_out;

    // GPU col2im: each work-item handles one column entry, atomically accumulates into output
    int64_t col2im_work_size = C * kernel_size * kernel_size * H_col * W_col;

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
                    int64_t kw = idx % kernel_size;
                    idx /= kernel_size;
                    int64_t kh = idx % kernel_size;
                    int64_t c = idx / kernel_size;
                    int64_t h = h_c * stride - padding + kh * dilation;
                    int64_t w = w_c * stride - padding + kw * dilation;
                    if (h >= 0 && w >= 0 && h < H_out && w < W_out) {
                        int64_t col_idx = ((c * kernel_size + kh) * kernel_size + kw);
                        int64_t l = h_c * W_col + w_c;
                        int64_t im_idx = (c * H_out + h) * W_out + w;
                        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>
                            atomic_val(batch_im[im_idx]);
                        atomic_val.fetch_add(batch_col[col_idx * L + l]);
                    }
                });
        }
        queue.wait();
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
                    int64_t kw = idx % kernel_size;
                    idx /= kernel_size;
                    int64_t kh = idx % kernel_size;
                    int64_t c = idx / kernel_size;
                    int64_t h = h_c * stride - padding + kh * dilation;
                    int64_t w = w_c * stride - padding + kw * dilation;
                    if (h >= 0 && w >= 0 && h < H_out && w < W_out) {
                        int64_t col_idx = ((c * kernel_size + kh) * kernel_size + kw);
                        int64_t l = h_c * W_col + w_c;
                        int64_t im_idx = (c * H_out + h) * W_out + w;
                        sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device>
                            atomic_val(batch_im[im_idx]);
                        atomic_val.fetch_add(batch_col[col_idx * L + l]);
                    }
                });
        }
        queue.wait();
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

auto roll_kernel(const Tensor& input, int64_t shift, int64_t dim, sycl::queue& queue) -> Tensor {
    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);
    auto shape = in_cont.shape();
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;

    int64_t dim_size = shape[dim];
    // Normalize shift to [0, dim_size)
    shift = ((shift % dim_size) + dim_size) % dim_size;

    if (shift == 0) return in_cont;

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
    else {
        // Generic copy via host for other dtypes
        size_t elem_size = in_cont.dtype_size();
        std::vector<uint8_t> in_host(numel * elem_size);
        std::vector<uint8_t> out_host(numel * elem_size);
        queue.memcpy(in_host.data(), in_cont.data_ptr(), in_host.size()).wait();

        for (int64_t i = 0; i < numel; ++i) {
            int64_t inner_idx = i % inner_size;
            int64_t dim_idx = (i / inner_size) % dim_size;
            int64_t outer_idx = i / (inner_size * dim_size);
            int64_t new_dim_idx = (dim_idx + shift) % dim_size;
            int64_t out_i = (outer_idx * dim_size + new_dim_idx) * inner_size + inner_idx;
            std::memcpy(out_host.data() + out_i * elem_size,
                       in_host.data() + i * elem_size, elem_size);
        }

        queue.memcpy(const_cast<void*>(output.data_ptr()), out_host.data(), out_host.size()).wait();
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

auto strided_fill_kernel(Tensor& self, double value, sycl::queue& queue) -> void {
    int64_t numel = self.numel();
    if (numel == 0) return;

    auto shape_span = self.shape();
    auto strides_span = self.strides();
    size_t ndim = shape_span.size();

    if (self.is_contiguous()) {
        if (self.dtype() == DType::Float32) {
            float val = static_cast<float>(value);
            float* ptr = get_data_ptr<float>(self);
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) { ptr[i] = val; });
        } else if (self.dtype() == DType::Float64) {
            double* ptr = get_data_ptr<double>(self);
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) { ptr[i] = value; });
        } else {
            Tensor filled = fill_kernel(self, static_cast<float>(value), queue);
            queue.memcpy(const_cast<void*>(self.data_ptr()), filled.data_ptr(),
                         numel * self.dtype_size());
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
        });
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
        });
    } else {
        throw std::runtime_error("strided_fill: unsupported dtype for non-contiguous fill");
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

} // namespace oneapi
} // namespace tenzor
