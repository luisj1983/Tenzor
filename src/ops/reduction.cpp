#include "tenzor/ops/reduction.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/ops/op_id.hpp"

namespace tenzor {

auto sum(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    OpAttributes attrs;
    if (dim.has_value()) attrs["dim"] = std::to_string(*dim);
    attrs["keepdim"] = keepdim ? "1" : "0";
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Sum, inputs, attrs)[0];
}

auto mean(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    OpAttributes attrs;
    if (dim.has_value()) attrs["dim"] = std::to_string(*dim);
    attrs["keepdim"] = keepdim ? "1" : "0";
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Mean, inputs, attrs)[0];
}

auto max(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    OpAttributes attrs;
    if (dim.has_value()) attrs["dim"] = std::to_string(*dim);
    attrs["keepdim"] = keepdim ? "1" : "0";
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Max, inputs, attrs)[0];
}

auto min(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    OpAttributes attrs;
    if (dim.has_value()) attrs["dim"] = std::to_string(*dim);
    attrs["keepdim"] = keepdim ? "1" : "0";
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Min, inputs, attrs)[0];
}

auto argmax(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    OpAttributes attrs;
    if (dim.has_value()) attrs["dim"] = std::to_string(*dim);
    attrs["keepdim"] = keepdim ? "1" : "0";
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::ArgMax, inputs, attrs)[0];
}

auto argmin(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    OpAttributes attrs;
    if (dim.has_value()) attrs["dim"] = std::to_string(*dim);
    attrs["keepdim"] = keepdim ? "1" : "0";
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::ArgMin, inputs, attrs)[0];
}

auto argsort(const Tensor& input, int64_t dim, bool descending) -> Tensor {
    OpAttributes attrs;
    attrs["dim"] = std::to_string(dim);
    attrs["descending"] = descending ? "1" : "0";
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::ArgSort, inputs, attrs)[0];
}

auto prod(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    OpAttributes attrs;
    if (dim.has_value()) attrs["dim"] = std::to_string(*dim);
    attrs["keepdim"] = keepdim ? "1" : "0";
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Prod, inputs, attrs)[0];
}

auto std(const Tensor& input, std::optional<int64_t> dim, bool keepdim, bool unbiased) -> Tensor {
    OpAttributes attrs;
    if (dim.has_value()) attrs["dim"] = std::to_string(*dim);
    attrs["keepdim"] = keepdim ? "1" : "0";
    attrs["unbiased"] = unbiased ? "1" : "0";
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Std, inputs, attrs)[0];
}

auto var(const Tensor& input, std::optional<int64_t> dim, bool keepdim, bool unbiased) -> Tensor {
    OpAttributes attrs;
    if (dim.has_value()) attrs["dim"] = std::to_string(*dim);
    attrs["keepdim"] = keepdim ? "1" : "0";
    attrs["unbiased"] = unbiased ? "1" : "0";
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Var, inputs, attrs)[0];
}

auto norm(const Tensor& input, float p, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    OpAttributes attrs;
    char p_buf[32];
    snprintf(p_buf, sizeof(p_buf), "%.9e", p);
    attrs["p"] = std::string(p_buf);
    if (dim.has_value()) attrs["dim"] = std::to_string(*dim);
    attrs["keepdim"] = keepdim ? "1" : "0";
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Norm, inputs, attrs)[0];
}

} // namespace tenzor
