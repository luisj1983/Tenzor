/**
 * @file tracing_interceptor.hpp
 * @brief Dispatch interceptor that records operations for automatic graph capture
 *
 * Pushes onto the DispatchInterceptorStack to transparently record all
 * dispatched operations into the Tracer while letting them execute eagerly.
 * This enables torch.compile-style automatic tracing without explicit
 * jit.trace() calls.
 */

#pragma once

#include "../backend/dispatch_interceptor.hpp"
#include "../ops/op_id.hpp"
#include "tracer.hpp"
#include <optional>

namespace tenzor {
namespace jit {

/**
 * @brief Maps OpId (dispatch-level) to OpType (IR-level).
 *
 * Returns nullopt for ops that cannot be represented in the IR,
 * indicating a graph break.
 *
 * @param op Dispatch-level operation ID
 * @return IR-level operation type, or nullopt for unsupported ops
 */
auto opid_to_optype(OpId op) -> std::optional<OpType>;

/**
 * @brief Creates a dispatch interceptor that records operations to the Tracer.
 *
 * When installed on the DispatchInterceptorStack, this interceptor:
 * 1. Maps the OpId to an OpType (graph break if unmapped)
 * 2. Registers input/output tensors with the Tracer
 * 3. Records the operation as a TracedOp
 * 4. Calls next() to execute the operation eagerly
 *
 * @param tracer Tracer instance to record into
 * @param on_graph_break Callback invoked when an unmappable op is encountered.
 *                       If nullptr, unmapped ops are silently skipped.
 * @return DispatchInterceptor suitable for DispatchInterceptorStack::push()
 */
auto make_tracing_interceptor(
    Tracer& tracer,
    std::function<void(OpId)> on_graph_break = nullptr)
    -> DispatchInterceptor;

} // namespace jit
} // namespace tenzor
