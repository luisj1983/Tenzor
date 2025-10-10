#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <cstring>

namespace tenzor {
namespace cpu {

// CPU transform kernels

auto transpose_kernel(const Tensor& input, int64_t dim0, int64_t dim1) -> Tensor {
    // TODO: Implement cache-optimized transpose
    return input;
}

auto contiguous_kernel(const Tensor& input) -> Tensor {
    // If already contiguous, return as-is
    if (input.is_contiguous()) {
        return input;
    }

    // Create new contiguous tensor with same shape, dtype, device
    Tensor result(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    // Copy data using strides to access elements in correct order
    const int64_t total_elements = input.numel();
    const size_t element_size = dtype_size(input.dtype());

    // Get raw pointers
    auto* src = static_cast<uint8_t*>(const_cast<void*>(
        static_cast<const void*>(input.data<uint8_t>())));
    auto* dst = static_cast<uint8_t*>(static_cast<void*>(result.data<uint8_t>()));

    const int64_t ndims = input.ndim();

    if (ndims == 0) {
        // Scalar tensor - direct copy
        std::memcpy(dst, src, element_size);
        return result;
    }

    // Multi-dimensional copy using stride calculations
    std::vector<int64_t> indices(ndims, 0);
    int64_t dst_offset = 0;

    auto strides = input.strides();
    auto shape = input.shape();

    for (int64_t i = 0; i < total_elements; ++i) {
        // Calculate source offset using strides
        int64_t src_offset = 0;
        for (int64_t dim = 0; dim < ndims; ++dim) {
            src_offset += indices[dim] * strides[dim];
        }

        // Copy single element
        std::memcpy(dst + dst_offset * element_size,
                    src + src_offset * element_size,
                    element_size);

        // Increment destination offset (contiguous layout)
        ++dst_offset;

        // Increment indices (row-major order)
        for (int64_t dim = ndims - 1; dim >= 0; --dim) {
            if (++indices[dim] < shape[dim]) {
                break;
            }
            indices[dim] = 0;
        }
    }

    return result;
}

} // namespace cpu
} // namespace tenzor
