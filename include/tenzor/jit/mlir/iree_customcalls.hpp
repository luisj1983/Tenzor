// Phase 13 / Group A.9 + Group D — IREE custom_call registrar (public surface).
//
// See src/jit/mlir/iree_customcalls.cpp for the environmental note explaining
// why this is currently driven by compile-time rejection of unregistered
// custom_call targets rather than by runtime-resolved C callbacks. The real
// runtime binding lights up once the iree/runtime/api.h headers are restored
// in the distribution.
//
// Group D wires up the per-op *dispatcher* functions (declared in this header
// under namespace `customcalls`) that take Tenzor tensors + a backend_config
// string, parse the scalar attributes, and dispatch to the existing OpId
// kernel. These are unit-testable today; the iree-run-module side will call
// them once the runtime plugin shared library is built.

#pragma once

#include <string>
#include <vector>

namespace tenzor {
class Tensor;
}

namespace tenzor::jit::mlir_jit::customcalls {

/// `stablehlo.custom_call @tenzor_flash_attention(%q, %k, %v) {backend_config
/// = "causal=<b>,scale=<f>"}` dispatcher. Returns the FlashAttention output
/// tensor with the same shape as Q.
auto dispatch_flash_attention(const std::vector<::tenzor::Tensor>& inputs,
                              const std::string& backend_config)
    -> ::tenzor::Tensor;

/// `@tenzor_gqa` dispatcher. Implements the GQA broadcast-then-FA pipeline
/// (KV-head replication then FlashAttention).
auto dispatch_gqa(const std::vector<::tenzor::Tensor>& inputs,
                  const std::string& backend_config)
    -> ::tenzor::Tensor;

/// `@tenzor_rope_apply` dispatcher. Inputs: (x, cos_table, sin_table).
/// backend_config: `"offset=<i>"` (currently unused — the offset is baked
/// into the cos/sin tables by the eager precomputation).
auto dispatch_rope_apply(const std::vector<::tenzor::Tensor>& inputs,
                         const std::string& backend_config)
    -> ::tenzor::Tensor;

/// `@tenzor_rms_norm` dispatcher. Inputs: (x[, weight]). backend_config:
/// `"eps=<f>"`.
auto dispatch_rms_norm(const std::vector<::tenzor::Tensor>& inputs,
                       const std::string& backend_config)
    -> ::tenzor::Tensor;

}  // namespace tenzor::jit::mlir_jit::customcalls

namespace tenzor::jit::mlir_jit::placeholder_messages {

/// Each function returns the human-readable error message that would be
/// emitted by the corresponding IREE-side stub callback while the
/// iree/runtime/api.h headers remain incomplete in the distribution. The
/// dispatcher logic itself (above, in `customcalls`) is wired up by Group D.
auto flash_attention() -> const char*;
auto gqa() -> const char*;
auto rope_apply() -> const char*;
auto rms_norm() -> const char*;

}  // namespace tenzor::jit::mlir_jit::placeholder_messages
