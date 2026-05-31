#include "tenzor/ops/type_promotion.hpp"

// promote_types() is now a constexpr function in include/tenzor/core/dtype.hpp.
// This file only provides the Tensor-aware helpers that need the full Tensor
// header.

namespace tenzor {

auto promote_inputs(const Tensor& a, const Tensor& b) -> std::pair<Tensor, Tensor> {
    DType target = promote_types(a.dtype(), b.dtype());
    Tensor a_out = (a.dtype() == target) ? a : a.to(target);
    Tensor b_out = (b.dtype() == target) ? b : b.to(target);
    return {a_out, b_out};
}

} // namespace tenzor
