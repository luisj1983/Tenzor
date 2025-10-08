#include "tenzor/ops/indexing.hpp"
#include "tenzor/backend/dispatch.hpp"

namespace tenzor {

auto slice(const Tensor& input, int64_t dim, int64_t start,
          int64_t end, int64_t step) -> Tensor {
    return input.slice(dim, start, end, step);
}

auto index_select(const Tensor& input, int64_t dim, const Tensor& index) -> Tensor {
    OpAttributes attrs;
    attrs["dim"] = std::to_string(dim);
    std::vector<Tensor> inputs = {input, index};
    return Dispatcher::dispatch("index_select", inputs, attrs)[0];
}

} // namespace tenzor
