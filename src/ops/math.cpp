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
    // Use scientific notation to preserve precision
    char exp_buf[32];
    snprintf(exp_buf, sizeof(exp_buf), "%.9e", exponent);
    attrs["exponent"] = std::string(exp_buf);
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

auto neg(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("neg", inputs)[0];
}

auto reciprocal(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("reciprocal", inputs)[0];
}

auto sign(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("sign", inputs)[0];
}

auto floor(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("floor", inputs)[0];
}

auto ceil(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("ceil", inputs)[0];
}

auto round(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("round", inputs)[0];
}

auto clamp(const Tensor& input, float min, float max) -> Tensor {
    OpAttributes attrs;
    // Use scientific notation to preserve precision
    char min_buf[32], max_buf[32];
    snprintf(min_buf, sizeof(min_buf), "%.9e", min);
    snprintf(max_buf, sizeof(max_buf), "%.9e", max);
    attrs["min"] = std::string(min_buf);
    attrs["max"] = std::string(max_buf);
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("clamp", inputs, attrs)[0];
}

auto clamp_min(const Tensor& input, float min) -> Tensor {
    OpAttributes attrs;
    char min_buf[32];
    snprintf(min_buf, sizeof(min_buf), "%.9e", min);
    attrs["min"] = std::string(min_buf);
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("clamp_min", inputs, attrs)[0];
}

auto clamp_max(const Tensor& input, float max) -> Tensor {
    OpAttributes attrs;
    char max_buf[32];
    snprintf(max_buf, sizeof(max_buf), "%.9e", max);
    attrs["max"] = std::string(max_buf);
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("clamp_max", inputs, attrs)[0];
}

auto sinh(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("sinh", inputs)[0];
}

auto cosh(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("cosh", inputs)[0];
}

} // namespace tenzor
