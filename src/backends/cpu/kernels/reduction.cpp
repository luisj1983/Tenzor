#include "tenzor/core/tensor.hpp"

namespace tenzor {
namespace cpu {

// CPU reduction kernels (stub implementations)

auto sum_kernel(const Tensor& input, int64_t dim) -> Tensor {
    // TODO: Implement parallel reduction
    return input;
}

auto mean_kernel(const Tensor& input, int64_t dim) -> Tensor {
    // TODO: Implement parallel mean reduction
    return input;
}

} // namespace cpu
} // namespace tenzor
