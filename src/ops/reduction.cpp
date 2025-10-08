#include "tenzor/ops/reduction.hpp"
#include "tenzor/backend/dispatch.hpp"

namespace tenzor {

auto sum(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    OpAttributes attrs;
    if (dim.has_value()) attrs["dim"] = std::to_string(*dim);
    attrs["keepdim"] = keepdim ? "1" : "0";
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("sum", inputs, attrs)[0];
}

auto mean(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    OpAttributes attrs;
    if (dim.has_value()) attrs["dim"] = std::to_string(*dim);
    attrs["keepdim"] = keepdim ? "1" : "0";
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("mean", inputs, attrs)[0];
}

auto max(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    OpAttributes attrs;
    if (dim.has_value()) attrs["dim"] = std::to_string(*dim);
    attrs["keepdim"] = keepdim ? "1" : "0";
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("max", inputs, attrs)[0];
}

auto min(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    OpAttributes attrs;
    if (dim.has_value()) attrs["dim"] = std::to_string(*dim);
    attrs["keepdim"] = keepdim ? "1" : "0";
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("min", inputs, attrs)[0];
}

} // namespace tenzor
