/**
 * @file widen_narrow.hpp
 * @brief Float16/BFloat16 widen-compute-narrow helper.
 *
 * Many CPU kernels (transcendentals, accumulators, log/exp variants) are
 * mathematically correct only at Float32+ precision but get called on
 * Float16 / BFloat16 tensors. The standard workaround is to cast the input
 * to Float32, run the kernel, and cast the result back. Before this header
 * each call site inlined that idiom by hand (see e.g. `nn/init.cpp`, RNN
 * kernels, fused_ops); duplications make it easy to forget the cast-back
 * step and silently change the output dtype.
 *
 * `widen_narrow_compute(x, fn)` centralises the pattern: for half-precision
 * inputs it widens to Float32 once, runs `fn`, then narrows the result back
 * to the original dtype. For Float32 / Float64 inputs it calls `fn` directly
 * on the input — but still narrows the result back to the input dtype if
 * `fn` widened internally, so callers get a consistent dtype contract.
 */

#pragma once

#include <concepts>
#include <type_traits>
#include <utility>

#include "tenzor/core/dtype.hpp"
#include "tenzor/core/tensor.hpp"

namespace tenzor::utils {

/// True for the dtypes that need widening to Float32 before kernel compute.
inline constexpr auto is_half_precision(DType dt) noexcept -> bool {
    return dt == DType::Float16 || dt == DType::BFloat16;
}

/**
 * @brief Run @p fn at Float32 precision when @p x is half-precision; narrow
 *        the result back to @p x's original dtype.
 *
 * For Float32/Float64/complex/integer inputs, @p fn is invoked directly on
 * @p x. If @p fn happens to widen on its own (e.g. always returns Float32),
 * the result is narrowed back to the original dtype for caller consistency.
 *
 * @param x   Input tensor whose dtype controls the widen decision.
 * @param fn  Callable invocable as `Tensor(const Tensor&)`.
 * @return Tensor with the same dtype as @p x.
 */
template <typename Fn>
    requires std::invocable<Fn, const Tensor&>
auto widen_narrow_compute(const Tensor& x, Fn&& fn) -> Tensor {
    const DType orig = x.dtype();
    if (is_half_precision(orig)) {
        Tensor wide = x.to(DType::Float32);
        Tensor result = std::forward<Fn>(fn)(wide);
        if (result.dtype() != orig) {
            return result.to(orig);
        }
        return result;
    }
    Tensor result = std::forward<Fn>(fn)(x);
    if (result.dtype() != orig) {
        return result.to(orig);
    }
    return result;
}

/**
 * @brief Two-input variant. Widens both inputs to Float32 if either is
 *        half-precision; result narrows back to @p a's dtype.
 *
 * Behaviour mirrors the single-arg form: if neither input is half, @p fn
 * is invoked directly on @p a, @p b. The result is narrowed to @p a's
 * original dtype regardless.
 */
template <typename Fn>
    requires std::invocable<Fn, const Tensor&, const Tensor&>
auto widen_narrow_compute(const Tensor& a, const Tensor& b, Fn&& fn) -> Tensor {
    const DType orig = a.dtype();
    const bool widen = is_half_precision(orig) || is_half_precision(b.dtype());
    if (widen) {
        // Widen to the HIGHEST-precision float among the inputs, not always
        // Float32. A Float64 operand paired with a half-precision operand must
        // compute at Float64, otherwise the result is a Float64-typed tensor
        // carrying only Float32 precision (the classic silent-accumulator bug).
        const DType wide = (a.dtype() == DType::Float64 || b.dtype() == DType::Float64)
                               ? DType::Float64
                               : DType::Float32;
        Tensor wa = (a.dtype() == wide) ? a : a.to(wide);
        Tensor wb = (b.dtype() == wide) ? b : b.to(wide);
        Tensor result = std::forward<Fn>(fn)(wa, wb);
        if (result.dtype() != orig) {
            return result.to(orig);
        }
        return result;
    }
    Tensor result = std::forward<Fn>(fn)(a, b);
    if (result.dtype() != orig) {
        return result.to(orig);
    }
    return result;
}

} // namespace tenzor::utils
