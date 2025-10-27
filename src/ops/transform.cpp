#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/backend/dispatch.hpp"
#include <sstream>

namespace tenzor {

// Helper function to convert shape vector to comma-separated string
static auto shape_to_string(const std::vector<int64_t>& shape) -> std::string {
    std::ostringstream oss;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) oss << ",";
        oss << shape[i];
    }
    return oss.str();
}

// Shape transformations (delegate to Tensor methods)

auto reshape(const Tensor& input, std::vector<int64_t> shape) -> Tensor {
    return input.reshape(std::move(shape));
}

auto view(const Tensor& input, std::vector<int64_t> shape) -> Tensor {
    return input.view(std::move(shape));
}

auto transpose(const Tensor& input, int64_t dim0, int64_t dim1) -> Tensor {
    return input.transpose(dim0, dim1);
}

auto permute(const Tensor& input, std::vector<int64_t> dims) -> Tensor {
    return input.permute(std::move(dims));
}

auto squeeze(const Tensor& input, std::optional<int64_t> dim) -> Tensor {
    return input.squeeze(dim);
}

auto unsqueeze(const Tensor& input, int64_t dim) -> Tensor {
    return input.unsqueeze(dim);
}

auto flatten(const Tensor& input, int64_t start_dim, int64_t end_dim) -> Tensor {
    return input.flatten(start_dim, end_dim);
}

auto contiguous(const Tensor& input) -> Tensor {
    return input.contiguous();
}

// Additional stub implementations
auto cat(std::span<const Tensor> tensors, int64_t dim) -> Tensor {
    if (tensors.empty()) {
        throw std::invalid_argument("Cannot concatenate empty tensor list");
    }

    if (tensors.size() == 1) {
        return tensors[0];
    }

    // Validate all tensors have compatible shapes
    auto first_shape = tensors[0].shape();
    int64_t ndim = first_shape.size();

    // Normalize dim
    if (dim < 0) {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("Dimension out of range for concatenation");
    }

    // Check all tensors have same ndim and same shape except at dim
    int64_t total_size_at_dim = 0;
    for (const auto& t : tensors) {
        auto shape = t.shape();
        if (shape.size() != static_cast<size_t>(ndim)) {
            throw std::invalid_argument("All tensors must have the same number of dimensions");
        }
        for (int64_t i = 0; i < ndim; ++i) {
            if (i != dim && shape[i] != first_shape[i]) {
                throw std::invalid_argument("All tensors must have the same shape except in the concatenation dimension");
            }
        }
        total_size_at_dim += shape[dim];
    }

    // Create output shape
    std::vector<int64_t> out_shape(first_shape.begin(), first_shape.end());
    out_shape[dim] = total_size_at_dim;

    // For non-CPU devices, use dispatcher to route to backend-specific implementation
    if (tensors[0].device().type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs["dim"] = std::to_string(dim);

        // Convert span to vector for dispatch
        std::vector<Tensor> tensor_vec(tensors.begin(), tensors.end());
        return Dispatcher::dispatch("cat", std::span<const Tensor>(tensor_vec), attrs)[0];
    }

    // For CPU, manually concatenate
    if (tensors[0].device().type == Device::Type::CPU) {
        auto output = zeros(out_shape, tensors[0].dtype(), Device::cpu());
        float* out_data = output.data<float>();

        // Calculate output strides
        std::vector<int64_t> out_strides(ndim);
        int64_t stride = 1;
        for (int64_t i = ndim - 1; i >= 0; --i) {
            out_strides[i] = stride;
            stride *= out_shape[i];
        }

        int64_t offset_at_dim = 0;
        for (const auto& t : tensors) {
            // Make tensor contiguous to ensure proper data layout
            auto t_cont = t.is_contiguous() ? t : t.contiguous();
            const float* in_data = t_cont.data<float>();
            auto in_shape = t_cont.shape();

            // Calculate input strides (contiguous)
            std::vector<int64_t> in_strides(ndim);
            int64_t in_stride = 1;
            for (int64_t i = ndim - 1; i >= 0; --i) {
                in_strides[i] = in_stride;
                in_stride *= in_shape[i];
            }

            // Copy data element by element
            int64_t in_size = 1;
            for (auto s : in_shape) {
                in_size *= s;
            }

            for (int64_t in_idx = 0; in_idx < in_size; ++in_idx) {
                // Calculate coordinates in input (row-major order)
                int64_t temp = in_idx;
                std::vector<int64_t> coords(ndim);
                for (int64_t i = ndim - 1; i >= 0; --i) {
                    coords[i] = temp % in_shape[i];
                    temp /= in_shape[i];
                }

                // Adjust coordinate at concatenation dimension
                coords[dim] += offset_at_dim;

                // Calculate output index using row-major strides
                int64_t out_idx = 0;
                for (int64_t i = 0; i < ndim; ++i) {
                    out_idx += coords[i] * out_strides[i];
                }

                out_data[out_idx] = in_data[in_idx];
            }

            offset_at_dim += in_shape[dim];
        }

        return output;
    }

    // Should never reach here - all device types are handled above
    throw std::runtime_error("Unsupported device type for concatenation");
}

auto stack(std::span<const Tensor> tensors, int64_t dim) -> Tensor {
    if (tensors.empty()) {
        throw std::invalid_argument("Cannot stack empty tensor list");
    }

    // All tensors must have the same shape
    auto first_shape = tensors[0].shape();
    for (size_t i = 1; i < tensors.size(); ++i) {
        auto shape = tensors[i].shape();
        if (shape.size() != first_shape.size()) {
            throw std::invalid_argument("All tensors must have the same number of dimensions for stacking");
        }
        for (size_t j = 0; j < shape.size(); ++j) {
            if (shape[j] != first_shape[j]) {
                throw std::invalid_argument("All tensors must have the same shape for stacking");
            }
        }
    }

    // Normalize dim
    int64_t ndim = first_shape.size();
    if (dim < 0) {
        dim += ndim + 1;
    }
    if (dim < 0 || dim > ndim) {
        throw std::invalid_argument("Dimension out of range for stack");
    }

    // Stack is equivalent to: unsqueeze each tensor at dim, then concatenate
    std::vector<Tensor> unsqueezed;
    unsqueezed.reserve(tensors.size());
    for (const auto& t : tensors) {
        unsqueezed.push_back(t.unsqueeze(dim));
    }

    return cat(std::span<const Tensor>(unsqueezed), dim);
}

auto split(const Tensor& input, int64_t split_size, int64_t dim) -> std::vector<Tensor> {
    if (split_size <= 0) {
        throw std::invalid_argument("Split size must be positive");
    }

    // Get input shape and normalize dimension
    auto shape = input.shape();
    int64_t ndim = shape.size();

    if (dim < 0) {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("Dimension out of range for split");
    }

    int64_t dim_size = shape[dim];

    // Calculate number of splits needed
    int64_t num_splits = (dim_size + split_size - 1) / split_size;  // Ceiling division

    // Split tensor into chunks using slice (creates views, no copying)
    std::vector<Tensor> result;
    result.reserve(num_splits);

    for (int64_t i = 0; i < num_splits; ++i) {
        int64_t start = i * split_size;
        int64_t end = std::min(start + split_size, dim_size);

        // Stop if we've exhausted the dimension (shouldn't happen with ceiling division)
        if (start >= dim_size) {
            break;
        }

        // Use slice to create a view (zero-copy)
        result.push_back(input.slice(dim, start, end));
    }

    return result;
}

auto chunk(const Tensor& input, int64_t chunks, int64_t dim) -> std::vector<Tensor> {
    if (chunks <= 0) {
        throw std::invalid_argument("Number of chunks must be positive");
    }

    // Get input shape and normalize dimension
    auto shape = input.shape();
    int64_t ndim = shape.size();

    if (dim < 0) {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("Dimension out of range for chunk");
    }

    int64_t dim_size = shape[dim];

    // Calculate chunk size (ceiling division)
    int64_t chunk_size = (dim_size + chunks - 1) / chunks;

    // Split tensor into chunks
    std::vector<Tensor> result;
    result.reserve(chunks);

    for (int64_t i = 0; i < chunks; ++i) {
        int64_t start = i * chunk_size;
        int64_t end = std::min(start + chunk_size, dim_size);

        // Stop if we've exhausted the dimension
        if (start >= dim_size) {
            break;
        }

        result.push_back(input.slice(dim, start, end));
    }

    return result;
}

auto repeat(const Tensor& input, std::vector<int64_t> repeats) -> Tensor {
    // TODO: Implement repeat
    return input;
}

auto tile(const Tensor& input, std::vector<int64_t> reps) -> Tensor {
    // TODO: Implement tile
    return input;
}

auto expand(const Tensor& input, std::vector<int64_t> shape) -> Tensor {
    auto input_shape = input.shape();

    // Validate expansion is possible
    if (shape.size() < input_shape.size()) {
        throw std::invalid_argument("Expanded shape must have at least as many dimensions as input");
    }

    // Check if already the right shape
    bool same_shape = (shape.size() == input_shape.size());
    if (same_shape) {
        for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] != input_shape[i]) {
                same_shape = false;
                break;
            }
        }
    }
    if (same_shape) {
        return input;
    }

    // Use dispatcher for non-CPU devices (expand takes an input tensor)
    if (input.device().type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs["shape"] = shape_to_string(shape);

        std::vector<Tensor> inputs = {input};
        return Dispatcher::dispatch("expand", inputs, attrs)[0];
    }

    // CPU path: Manual implementation
    auto input_shape_vec = std::vector<int64_t>(input_shape.begin(), input_shape.end());
    const float* input_data = input.data<float>();

    // Create output tensor on CPU
    auto output = zeros(shape, input.dtype(), Device::cpu());
    float* output_data = output.data<float>();

    // Calculate strides
    std::vector<int64_t> input_strides(input_shape_vec.size());
    int64_t input_stride = 1;
    for (int i = input_shape_vec.size() - 1; i >= 0; --i) {
        input_strides[i] = input_stride;
        input_stride *= input_shape_vec[i];
    }

    // Calculate total output elements
    int64_t total_elements = 1;
    for (auto s : shape) {
        total_elements *= s;
    }

    // Fill output by replicating input
    for (int64_t out_idx = 0; out_idx < total_elements; ++out_idx) {
        int64_t temp = out_idx;
        int64_t in_idx = 0;

        int input_dim_offset = shape.size() - input_shape_vec.size();

        for (int i = shape.size() - 1; i >= 0; --i) {
            int64_t coord = temp % shape[i];
            temp /= shape[i];

            int input_dim = i - input_dim_offset;
            if (input_dim >= 0 && input_dim < static_cast<int>(input_shape_vec.size())) {
                // Map to input dimension (handle size-1 dimensions)
                if (input_shape_vec[input_dim] == 1) {
                    coord = 0;
                } else if (input_shape_vec[input_dim] != shape[i]) {
                    throw std::invalid_argument("Cannot expand dimension from non-1 size to different size");
                }
                in_idx += coord * input_strides[input_dim];
            }
        }

        output_data[out_idx] = input_data[in_idx];
    }

    return output;
}

auto roll(const Tensor& input, int64_t shifts, int64_t dim) -> Tensor {
    // Normalize dimension
    int64_t ndim = input.ndim();
    if (dim < 0) {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("Dimension out of range for roll");
    }

    int64_t dim_size = input.shape()[dim];
    if (dim_size == 0) {
        return input.clone();
    }

    // Normalize shifts to [0, dim_size)
    shifts = shifts % dim_size;
    if (shifts < 0) {
        shifts += dim_size;
    }

    if (shifts == 0) {
        return input.clone();
    }

    // Roll by splitting at the shift point and concatenating in reverse order
    // For positive shift: [a, b, c, d] shift by 2 => [c, d, a, b]
    // Split: [a, b] and [c, d], then cat([c, d], [a, b])
    Tensor part1 = input.slice(dim, 0, dim_size - shifts);
    Tensor part2 = input.slice(dim, dim_size - shifts, dim_size);

    return cat({part2, part1}, dim);
}

} // namespace tenzor
