#include "tenzor/ops/math.hpp"
#include "tenzor/backend/dispatch.hpp"

namespace tenzor {

// Stub implementations - will be dispatched to backend kernels

auto add(const Tensor& a, const Tensor& b) -> Tensor {
    std::vector<Tensor> inputs = {a, b};
    return Dispatcher::dispatch("add", inputs)[0];
}

auto sub(const Tensor& a, const Tensor& b) -> Tensor {
    std::vector<Tensor> inputs = {a, b};
    return Dispatcher::dispatch("sub", inputs)[0];
}

auto mul(const Tensor& a, const Tensor& b) -> Tensor {
    std::vector<Tensor> inputs = {a, b};
    return Dispatcher::dispatch("mul", inputs)[0];
}

auto div(const Tensor& a, const Tensor& b) -> Tensor {
    std::vector<Tensor> inputs = {a, b};
    return Dispatcher::dispatch("div", inputs)[0];
}

auto matmul(const Tensor& a, const Tensor& b) -> Tensor {
    std::vector<Tensor> inputs = {a, b};
    return Dispatcher::dispatch("matmul", inputs)[0];
}

auto dot(const Tensor& a, const Tensor& b) -> Tensor {
    std::vector<Tensor> inputs = {a, b};
    return Dispatcher::dispatch("dot", inputs)[0];
}

auto pow(const Tensor& input, float exponent) -> Tensor {
    OpAttributes attrs;
    attrs["exponent"] = std::to_string(exponent);
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("pow", inputs, attrs)[0];
}

auto exp(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("exp", inputs)[0];
}

auto log(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("log", inputs)[0];
}

auto sqrt(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("sqrt", inputs)[0];
}

auto sin(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("sin", inputs)[0];
}

auto cos(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("cos", inputs)[0];
}

auto tan(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("tan", inputs)[0];
}

auto tanh(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("tanh", inputs)[0];
}

auto abs(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("abs", inputs)[0];
}

} // namespace tenzor
