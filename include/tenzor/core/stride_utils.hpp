/// \file stride_utils.hpp
/// \brief Shared stride helpers for backend kernels.
///
/// The audit (2026-05-17) found that several OneAPI and Vulkan kernels
/// reinvent `strides[i] = strides[i+1] * shape[i+1]` inline and then
/// *read* that synthesized stride array instead of the tensor's actual
/// strides. This silently miscomputes whenever the kernel is handed a
/// non-contiguous view (e.g. the output of `permute(0,2,1,3)`).
///
/// This header consolidates the canonical helpers and adds
/// `ensure_contiguous` so kernel entry points have a single, clear
/// expression of "make this tensor contiguous if it isn't already".
///
/// **Rule for backend kernel authors:**
/// - If your kernel *reads* multi-D data using stride arithmetic, accept
///   the tensor's *actual* strides via `tensor.strides()`.
/// - If your kernel assumes contiguous layout for SIMD/vectorisation,
///   call `tenzor::ensure_contiguous(t)` at entry. This is a *no-op*
///   when the tensor is already contiguous, so it is free in the
///   common case.
/// - Never synthesize strides via `strides[i] = strides[i+1] * shape[i+1]`
///   and then read the synthesized strides for an input tensor — use
///   `tensor.strides()`. The synthesis is fine for *output* allocation
///   because freshly-allocated buffers are contiguous by construction
///   (see `tenzor::contiguous_strides` below).

#pragma once

#include "tenzor/core/shape.hpp"
#include "tenzor/core/tensor.hpp"

#include <span>
#include <vector>

namespace tenzor {

/// Compute the row-major (C-order) contiguous strides for a given shape.
///
/// Identical to `tenzor::compute_strides` (which see) — re-exported here
/// so backend kernels need only include one header. The output is in
/// element units, not bytes.
[[nodiscard]] inline auto
contiguous_strides(std::span<const int64_t> shape) -> std::vector<int64_t> {
    return tenzor::compute_strides(shape);
}

/// Returns a contiguous view of `t`. Cheap no-op when `t` is already
/// contiguous (which is the common case for fresh allocations and
/// outputs of element-wise ops).
///
/// Equivalent to writing
///
///     auto t_c = t.is_contiguous() ? t : t.contiguous();
///
/// at every backend kernel entry point. Use this helper so the intent
/// is grep-able: searching for `ensure_contiguous` finds every backend
/// kernel that explicitly requires contiguous input.
[[nodiscard]] inline auto
ensure_contiguous(const Tensor& t) -> Tensor {
    return t.is_contiguous() ? t : t.contiguous();
}

/// Same as `ensure_contiguous` but returns the original tensor unchanged
/// when contiguous, otherwise materialises a contiguous copy. This is the
/// canonical form for kernels that need to hold a const-reference to the
/// input for the duration of the kernel launch (e.g. async streams).
[[nodiscard]] inline auto
ensure_contiguous_or_copy(const Tensor& t) -> Tensor {
    return t.is_contiguous() ? t : t.contiguous();
}

/// True iff `t` has the row-major contiguous stride pattern computed
/// from its own shape. Use as an assertion guard at the top of a kernel
/// that does NOT call `ensure_contiguous` but still requires
/// contiguity (e.g. shader buffer bindings).
[[nodiscard]] inline auto
has_row_major_strides(const Tensor& t) noexcept -> bool {
    return t.is_contiguous();
}

}  // namespace tenzor
