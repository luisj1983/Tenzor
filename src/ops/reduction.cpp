#include "tenzor/ops/reduction.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"

namespace tenzor {

auto sum(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, *dim);
    attrs.set(AttrKey::Keepdim, keepdim);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Sum, inputs, attrs)[0];
}

auto mean(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, *dim);
    attrs.set(AttrKey::Keepdim, keepdim);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Mean, inputs, attrs)[0];
}

auto max(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, *dim);
    attrs.set(AttrKey::Keepdim, keepdim);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Max, inputs, attrs)[0];
}

auto min(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, *dim);
    attrs.set(AttrKey::Keepdim, keepdim);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Min, inputs, attrs)[0];
}

auto argmax(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, *dim);
    attrs.set(AttrKey::Keepdim, keepdim);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::ArgMax, inputs, attrs)[0];
}

auto argmin(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, *dim);
    attrs.set(AttrKey::Keepdim, keepdim);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::ArgMin, inputs, attrs)[0];
}

auto argsort(const Tensor& input, int64_t dim, bool descending) -> Tensor {
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    attrs.set(AttrKey::Descending, descending);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::ArgSort, inputs, attrs)[0];
}

auto prod(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, *dim);
    attrs.set(AttrKey::Keepdim, keepdim);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Prod, inputs, attrs)[0];
}

auto std(const Tensor& input, std::optional<int64_t> dim, bool keepdim, bool unbiased) -> Tensor {
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, *dim);
    attrs.set(AttrKey::Keepdim, keepdim);
    attrs.set(AttrKey::Unbiased, unbiased);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Std, inputs, attrs)[0];
}

auto var(const Tensor& input, std::optional<int64_t> dim, bool keepdim, bool unbiased) -> Tensor {
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, *dim);
    attrs.set(AttrKey::Keepdim, keepdim);
    attrs.set(AttrKey::Unbiased, unbiased);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Var, inputs, attrs)[0];
}

auto norm(const Tensor& input, float p, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    NewOpAttributes attrs;
    attrs.set(AttrKey::P, static_cast<double>(p));
    if (dim.has_value()) attrs.set(AttrKey::Dim, *dim);
    attrs.set(AttrKey::Keepdim, keepdim);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Norm, inputs, attrs)[0];
}

} // namespace tenzor
