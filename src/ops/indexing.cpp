#include "tenzor/ops/indexing.hpp"
#include "tenzor/backend/dispatch.hpp"
#include "tenzor/ops/creation.hpp"

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

auto gather(const Tensor& input, int64_t dim, const Tensor& index) -> Tensor {
    OpAttributes attrs;
    attrs["dim"] = std::to_string(dim);
    std::vector<Tensor> inputs = {input, index};
    return Dispatcher::dispatch("gather", inputs, attrs)[0];
}

auto scatter(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& src) -> Tensor {
    OpAttributes attrs;
    attrs["dim"] = std::to_string(dim);
    std::vector<Tensor> inputs = {input, index, src};
    return Dispatcher::dispatch("scatter", inputs, attrs)[0];
}

auto masked_select(const Tensor& input, const Tensor& mask) -> Tensor {
    std::vector<Tensor> inputs = {input, mask};
    return Dispatcher::dispatch("masked_select", inputs)[0];
}

auto masked_fill(const Tensor& input, const Tensor& mask, float value) -> Tensor {
    OpAttributes attrs;
    char value_buf[32];
    snprintf(value_buf, sizeof(value_buf), "%.9e", value);
    attrs["value"] = std::string(value_buf);
    std::vector<Tensor> inputs = {input, mask};
    return Dispatcher::dispatch("masked_fill", inputs, attrs)[0];
}

auto where(const Tensor& condition, const Tensor& x, const Tensor& y) -> Tensor {
    std::vector<Tensor> inputs = {condition, x, y};
    return Dispatcher::dispatch("where", inputs)[0];
}

auto take(const Tensor& input, const Tensor& index) -> Tensor {
    std::vector<Tensor> inputs = {input, index};
    return Dispatcher::dispatch("take", inputs)[0];
}

auto put(const Tensor& input, const Tensor& index, const Tensor& source) -> Tensor {
    std::vector<Tensor> inputs = {input, index, source};
    return Dispatcher::dispatch("put", inputs)[0];
}

} // namespace tenzor
