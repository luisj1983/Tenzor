#include "tenzor/core/tensor.hpp"
#include "tenzor/backend/backend.hpp"
#include <sycl/sycl.hpp>
#include <stdexcept>
#include <sstream>

namespace tenzor {
namespace oneapi {

// Kernel class declarations for expand operation (separate classes per dtype to avoid ODR violations)
struct ExpandKernelFloat32 {};
struct ExpandKernelFloat64 {};

// Helper function to get typed pointer from tensor
template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

/**
 * @brief Parse comma-separated shape string into vector.
 *
 * @param shape_str Comma-separated shape string (e.g., "2,4,3")
 * @return std::vector<int64_t> Parsed shape vector
 */
auto parse_shape_string(const std::string& shape_str) -> std::vector<int64_t> {
    std::vector<int64_t> shape;
    std::stringstream ss(shape_str);
    std::string item;

    while (std::getline(ss, item, ',')) {
        shape.push_back(std::stoll(item));
    }

    return shape;
}

/**
 * @brief Compute strides for a given shape (row-major order).
 *
 * @param shape Shape vector
 * @return std::vector<int64_t> Strides for each dimension
 */
auto compute_strides(const std::vector<int64_t>& shape) -> std::vector<int64_t> {
    std::vector<int64_t> strides(shape.size());
    if (shape.empty()) {
        return strides;
    }

    strides[shape.size() - 1] = 1;
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    return strides;
}

/**
 * @brief Expand kernel implementation template.
 *
 * Broadcasts a tensor to a larger size by repeating elements along dimensions
 * where the input size is 1. For example, a tensor of shape [2, 1, 3] can be
 * expanded to [2, 4, 3] by repeating the middle dimension.
 *
 * Broadcasting rules:
 * - Input dimension must be 1 or equal to target dimension
 * - If input dimension is 1, all indices map to 0 in that dimension
 * - Otherwise, indices map directly
 *
 * @tparam T Data type (float or double)
 * @param input_data Input tensor data
 * @param output_data Output tensor data
 * @param input_shape Input tensor shape
 * @param output_shape Output tensor shape
 * @param input_strides Input tensor strides
 * @param output_size Total number of elements in output
 * @param ndim Number of dimensions
 * @param queue SYCL queue for execution
 */
template<typename T>
void expand_kernel_impl(const T* input_data, T* output_data,
                        const std::vector<int64_t>& input_shape,
                        const std::vector<int64_t>& output_shape,
                        const std::vector<int64_t>& input_strides,
                        int64_t output_size, int64_t ndim,
                        sycl::queue& queue) {

    // Use unique kernel class per dtype to avoid ODR violations
    using KernelClass = std::conditional_t<std::is_same_v<T, float>,
                                            ExpandKernelFloat32,
                                            ExpandKernelFloat64>;

    // Copy shape and stride info to arrays for kernel capture
    int64_t input_shape_arr[16];   // Maximum 16 dimensions
    int64_t output_shape_arr[16];
    int64_t input_strides_arr[16];

    for (int64_t i = 0; i < ndim; ++i) {
        input_shape_arr[i] = input_shape[i];
        output_shape_arr[i] = output_shape[i];
        input_strides_arr[i] = input_strides[i];
    }

    // Launch kernel
    queue.parallel_for<KernelClass>(sycl::range<1>(output_size), [=](sycl::id<1> idx) {
        int64_t output_idx = idx[0];
        int64_t input_idx = 0;
        int64_t remaining = output_idx;

        // Compute output coordinates and map to input index
        for (int64_t dim = ndim - 1; dim >= 0; --dim) {
            int64_t output_coord = remaining % output_shape_arr[dim];
            remaining /= output_shape_arr[dim];

            // Broadcasting: if input dimension is 1, use index 0
            int64_t input_coord = (input_shape_arr[dim] == 1) ? 0 : output_coord;
            input_idx += input_coord * input_strides_arr[dim];
        }

        output_data[output_idx] = input_data[input_idx];
    }).wait();
}

/**
 * @brief Expand operation for broadcasting tensors.
 *
 * Expands a tensor to a larger size by broadcasting along dimensions where
 * the input has size 1. This is a zero-copy operation that repeats elements
 * without allocating additional storage for the repeated data in memory.
 *
 * Example:
 *   Input shape:  [2, 1, 3]
 *   Target shape: [2, 4, 3]
 *   Result: The middle dimension is repeated 4 times
 *
 * Broadcasting rules:
 * - Each dimension in input must be 1 or match the target dimension
 * - Dimensions of size 1 can be expanded to any size
 * - Other dimensions must match exactly
 *
 * @param input Input tensor to expand
 * @param attrs Operation attributes containing:
 *   - shape: std::string - Comma-separated target shape (e.g., "2,4,3")
 * @param queue SYCL queue for execution
 * @return Tensor Expanded tensor with target shape
 * @throws std::invalid_argument If shapes are incompatible or invalid
 */
auto expand_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor {
    // Extract target shape from attributes
    if (!attrs.contains("shape")) {
        throw std::invalid_argument("expand: 'shape' attribute is required");
    }

    std::vector<int64_t> target_shape = parse_shape_string(attrs.at("shape"));
    auto input_shape_span = input.shape();
    std::vector<int64_t> input_shape(input_shape_span.begin(), input_shape_span.end());

    // Validate dimensions match
    if (input_shape.size() != target_shape.size()) {
        std::ostringstream oss;
        oss << "expand: input and target must have same number of dimensions (input: "
            << input_shape.size() << ", target: " << target_shape.size() << ")";
        throw std::invalid_argument(oss.str());
    }

    // Validate broadcasting rules
    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (input_shape[i] != 1 && input_shape[i] != target_shape[i]) {
            std::ostringstream oss;
            oss << "expand: cannot expand dimension " << i << " from size "
                << input_shape[i] << " to " << target_shape[i]
                << " (dimension must be 1 or match target)";
            throw std::invalid_argument(oss.str());
        }
    }

    // Create output tensor
    Tensor output(target_shape, input.dtype(), input.device());

    // Compute input strides
    std::vector<int64_t> input_strides = compute_strides(input_shape);

    // Calculate total output size
    int64_t output_size = 1;
    for (int64_t dim : target_shape) {
        output_size *= dim;
    }

    int64_t ndim = static_cast<int64_t>(input_shape.size());

    // Dispatch based on dtype
    if (input.dtype() == DType::Float32) {
        const float* input_ptr = get_data_ptr<const float>(input);
        float* output_ptr = get_data_ptr<float>(output);

        expand_kernel_impl<float>(input_ptr, output_ptr, input_shape, target_shape,
                                 input_strides, output_size, ndim, queue);
    }
    else if (input.dtype() == DType::Float64) {
        const double* input_ptr = get_data_ptr<const double>(input);
        double* output_ptr = get_data_ptr<double>(output);

        expand_kernel_impl<double>(input_ptr, output_ptr, input_shape, target_shape,
                                  input_strides, output_size, ndim, queue);
    }
    else {
        throw std::runtime_error("expand: Unsupported data type (only Float32 and Float64 supported)");
    }

    return output;
}

} // namespace oneapi
} // namespace tenzor
