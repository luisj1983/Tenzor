#include "tenzor/ops/transform.hpp"
#include "tenzor/backend/dispatch.hpp"

namespace tenzor {

auto reshape(const Tensor& input, std::vector<int64_t> shape) -> Tensor {
    return input.reshape(std::move(shape));
}

auto transpose(const Tensor& input, int64_t dim0, int64_t dim1) -> Tensor {
    return input.transpose(dim0, dim1);
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

} // namespace tenzor
