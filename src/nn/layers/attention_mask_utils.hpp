#pragma once

#include <limits>
#include <vector>

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/creation.hpp"  // full, zeros
#include "tenzor/ops/math.hpp"      // gt
#include "tenzor/ops/indexing.hpp"  // where

namespace tenzor {
namespace nn {

// Y.19 / EE.11 / EE.12: normalise a Bool or integer attention mask to a float
// additive mask (-inf where True/non-zero, 0 elsewhere). PyTorch's MHA accepts
// Bool masks where True = "ignore"; without this widening the downstream
// `scores + mask` adds 1.0 (Bool->float of true) at masked positions instead of
// -inf, leaking attention. Shared by attention.cpp and gqa_attention.cpp.
// Returns the normalised Tensor; passes through float dtypes unchanged.
inline auto normalize_attn_mask(const Tensor& attn_mask) -> Tensor {
    const DType am_dtype = attn_mask.dtype();
    if (am_dtype == DType::Float32 || am_dtype == DType::Float64 ||
        am_dtype == DType::Float16 || am_dtype == DType::BFloat16) {
        return attn_mask;
    }
    auto pm_shape = std::vector<int64_t>(attn_mask.shape().begin(),
                                         attn_mask.shape().end());
    if (am_dtype == DType::Bool) {
        Tensor neg_inf_tensor = full(pm_shape, -std::numeric_limits<float>::infinity(),
                                     DType::Float32, attn_mask.device());
        Tensor zero_tensor = zeros(pm_shape, DType::Float32, attn_mask.device());
        return Tensor(where(attn_mask, neg_inf_tensor, zero_tensor));
    }
    // Integer mask: treat as 0/1 indicator, widen to a float -inf/0 mask.
    Tensor as_float = attn_mask.to(DType::Float32);
    Tensor threshold = full(pm_shape, 0.5f, DType::Float32, attn_mask.device());
    Tensor neg_inf_tensor = full(pm_shape, -std::numeric_limits<float>::infinity(),
                                 DType::Float32, attn_mask.device());
    Tensor zero_tensor = zeros(pm_shape, DType::Float32, attn_mask.device());
    Tensor mask_gt = Tensor(gt(as_float, threshold));
    return Tensor(where(mask_gt, neg_inf_tensor, zero_tensor));
}

}  // namespace nn
}  // namespace tenzor
