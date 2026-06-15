/**
 * @file autocast_interceptor.hpp
 * @brief Autocast as a composable dispatch interceptor.
 *
 * Converts the autocast logic (previously hardcoded in fast_dispatch.hpp)
 * into a DispatchInterceptor that can be pushed/popped via the interceptor
 * stack. This enables autocast to compose naturally with other interceptors
 * (profiling, tracing, quantization, etc.).
 */

#pragma once

#include "tenzor/backend/dispatch_interceptor.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/nn/amp/autocast.hpp"

namespace tenzor::nn::amp {

/**
 * @brief Create a dispatch interceptor that applies autocast input casting.
 *
 * The returned interceptor:
 * 1. Checks if autocast is enabled (thread-local)
 * 2. For compute-heavy ops: casts Float32 inputs to the target half dtype
 * 3. For stability-critical ops: promotes half inputs to Float32
 * 4. Calls next() with the (possibly modified) inputs
 *
 * @return DispatchInterceptor that applies autocast logic
 */
inline DispatchInterceptor make_autocast_interceptor() {
    return [](OpId op, std::span<const Tensor> inputs,
              const OpAttributes& attrs, DispatchNext next) -> std::vector<Tensor> {
        // Honour Autocast's device targeting (mirrors should_autocast): a
        // CUDA-targeted autocast scope must not cast CPU ops. The interceptor is
        // pushed unconditionally for any enabled scope, so without this gate a
        // matmul on CPU tensors inside a CUDA autocast scope would still be
        // cast Float32->Float16, contradicting the device-targeting contract.
        auto target_device = Autocast::get_device_type();
        if (target_device.has_value() && !inputs.empty() &&
            inputs[0].device().type != *target_device) {
            return next(op, inputs, attrs);
        }

        // Reuse the existing autocast_inputs logic from fast_dispatch.hpp detail
        auto casted = detail::autocast_inputs(op, inputs);
        std::span<const Tensor> effective = casted.empty()
            ? inputs : std::span<const Tensor>(casted);
        return next(op, effective, attrs);
    };
}

} // namespace tenzor::nn::amp
