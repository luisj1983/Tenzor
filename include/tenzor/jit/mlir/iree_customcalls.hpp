// Phase 13 / Task A.9 — IREE custom_call registrar (public surface).
//
// See src/jit/mlir/iree_customcalls.cpp for the environmental note explaining
// why this is currently driven by compile-time rejection of unregistered
// custom_call targets rather than by runtime-resolved C callbacks. The real
// callback bindings light up once the iree/runtime/api.h headers are restored
// in the distribution.

#pragma once

namespace tenzor::jit::mlir_jit::placeholder_messages {

/// Each function returns the human-readable error message emitted when a
/// resolver for the corresponding tenzor_* custom_call is invoked before
/// Group D wires up a real implementation.
auto flash_attention() -> const char*;
auto gqa() -> const char*;
auto rope_apply() -> const char*;
auto rms_norm() -> const char*;

}  // namespace tenzor::jit::mlir_jit::placeholder_messages
