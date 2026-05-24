// SPDX-License-Identifier: Apache-2.0
//
// Shared R.16 master-weights helpers for half-precision (Float16/BFloat16)
// parameters. Optimiser state (momentum, second-moment, accumulators) must
// live at Float32 when the parameter dtype is half precision; storing state
// at the param dtype underflows for eps=1e-8 in Float16 and erodes BFloat16
// momentum to ~3 bits of effective precision after a few thousand steps.
//
// Mirrors PyTorch's master-weights convention used by AMP: the parameter
// stays in low precision, the optimiser state is upcast. Each optimiser
// step upcasts param/grad to the state dtype for arithmetic, then casts
// back to the param dtype on assignment.
//
// Originally inlined as an anonymous namespace in adam.cpp; extracted here
// so all nine optimisers (audit-4 U.9) and ZeRO (audit-4 V.30) share one
// definition.

#pragma once

#include "tenzor/core/dtype.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"

#include <cstdint>
#include <vector>

namespace tenzor::optim {

/// Return the dtype an optimiser should keep its state buffers in for a
/// parameter of dtype `param_dtype`. Half-precision params (Float16,
/// BFloat16) get Float32 state; everything else gets the param dtype.
inline auto optim_state_dtype(DType param_dtype) -> DType {
    if (param_dtype == DType::Float16 || param_dtype == DType::BFloat16) {
        return DType::Float32;
    }
    return param_dtype;
}

/// Allocate a zero-initialised state buffer for `param`, honouring the
/// R.16 master-weights rule: same shape/device as the param, but at the
/// state dtype returned by `optim_state_dtype`.
inline auto make_optim_state(const Tensor& param) -> Tensor {
    DType state_dt = optim_state_dtype(param.dtype());
    if (state_dt == param.dtype()) {
        return zeros_like(param);
    }
    std::vector<int64_t> shape(param.shape().begin(), param.shape().end());
    return zeros(shape, state_dt, param.device());
}

} // namespace tenzor::optim
