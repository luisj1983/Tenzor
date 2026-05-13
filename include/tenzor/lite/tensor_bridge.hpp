/**
 * @file tensor_bridge.hpp
 * @brief Non-owning views between LiteTensor and the main Tenzor Tensor.
 *
 * The Lite runtime stores intermediate buffers in a `LiteAllocator` arena and
 * the user-facing API exposes `LiteTensor` (a flat struct with fixed shape[8]
 * and an optional `owns_data` flag). The kernels in the main dispatch table,
 * however, consume `Tensor` (with refcounted `Storage`). These helpers bridge
 * the two without copying.
 *
 * `view_as_tensor` constructs a `Tensor` whose `Storage` references the raw
 * pointer inside a `LiteTensor` with a no-op deleter — lifetime of the
 * underlying buffer is the caller's responsibility. The returned Tensor is
 * therefore only valid while the source LiteTensor (or its backing arena)
 * remains alive.
 *
 * `to_lite_tensor` is the inverse for the API boundary: it copies a Tensor's
 * data into a freshly allocated LiteTensor that owns its buffer. Phase 2
 * replaces this with arena-backed zero-copy outputs once the memory planner
 * lands.
 */

#pragma once

#include "runtime.hpp"
#include "../core/tensor.hpp"

namespace tenzor::lite {

/** Non-owning Tensor view over the data pointer of a LiteTensor. */
auto view_as_tensor(const LiteTensor& lt) -> Tensor;

/** Owning LiteTensor copy of a CPU Tensor (data malloc'd + memcpy'd). */
auto to_lite_tensor(const Tensor& t) -> LiteTensor;

}  // namespace tenzor::lite
