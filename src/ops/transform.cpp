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
    // TODO: Implement concatenation
    if (tensors.empty()) {
        throw std::invalid_argument("Cannot concatenate empty tensor list");
    }
    return tensors[0];
}

auto stack(std::span<const Tensor> tensors, int64_t dim) -> Tensor {
    // TODO: Implement stack
    if (tensors.empty()) {
        throw std::invalid_argument("Cannot stack empty tensor list");
    }
    return tensors[0];
}

auto split(const Tensor& input, int64_t split_size, int64_t dim) -> std::vector<Tensor> {
    // TODO: Implement split
    return {input};
}

auto chunk(const Tensor& input, int64_t chunks, int64_t dim) -> std::vector<Tensor> {
    // TODO: Implement chunk
    return {input};
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

} // namespace tenzor
