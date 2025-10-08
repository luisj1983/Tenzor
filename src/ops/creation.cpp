#include "tenzor/ops/creation.hpp"
#include <random>
#include <cstring>
#include <stdexcept>

namespace tenzor {

auto zeros(std::vector<int64_t> shape, DType dtype, Device device) -> Tensor {
    auto tensor = empty(std::move(shape), dtype, device);
    if (tensor.impl() && tensor.impl()->storage) {
        std::memset(tensor.impl()->storage->data(), 0, tensor.numel() * dtype_size(dtype));
    }
    return tensor;
}

auto ones(std::vector<int64_t> shape, DType dtype, Device device) -> Tensor {
    auto tensor = empty(std::move(shape), dtype, device);
    // TODO: Fill with ones based on dtype
    return tensor;
}

auto full(std::vector<int64_t> shape, float value, DType dtype, Device device) -> Tensor {
    auto tensor = empty(std::move(shape), dtype, device);
    // TODO: Fill with value
    return tensor;
}

auto empty(std::vector<int64_t> shape, DType dtype, Device device) -> Tensor {
    return Tensor(std::move(shape), dtype, device);
}

auto rand(std::vector<int64_t> shape, DType dtype, Device device) -> Tensor {
    // TODO: Implement random uniform
    return empty(std::move(shape), dtype, device);
}

auto randn(std::vector<int64_t> shape, DType dtype, Device device) -> Tensor {
    // TODO: Implement random normal
    return empty(std::move(shape), dtype, device);
}

auto arange(float start, float end, float step, DType dtype, Device device) -> Tensor {
    // TODO: Implement arange
    return empty({static_cast<int64_t>((end - start) / step)}, dtype, device);
}

auto linspace(float start, float end, int64_t steps, DType dtype, Device device) -> Tensor {
    // TODO: Implement linspace
    return empty({steps}, dtype, device);
}

auto eye(int64_t n, std::optional<int64_t> m, DType dtype, Device device) -> Tensor {
    int64_t cols = m.value_or(n);
    // TODO: Implement identity matrix
    return zeros({n, cols}, dtype, device);
}

auto zeros_like(const Tensor& tensor) -> Tensor {
    std::vector<int64_t> shape(tensor.shape().begin(), tensor.shape().end());
    return zeros(shape, tensor.dtype(), tensor.device());
}

auto ones_like(const Tensor& tensor) -> Tensor {
    std::vector<int64_t> shape(tensor.shape().begin(), tensor.shape().end());
    return ones(shape, tensor.dtype(), tensor.device());
}

auto rand_like(const Tensor& tensor) -> Tensor {
    std::vector<int64_t> shape(tensor.shape().begin(), tensor.shape().end());
    return rand(shape, tensor.dtype(), tensor.device());
}

auto randn_like(const Tensor& tensor) -> Tensor {
    std::vector<int64_t> shape(tensor.shape().begin(), tensor.shape().end());
    return randn(shape, tensor.dtype(), tensor.device());
}

} // namespace tenzor
