// Phase 13 / Task B.1 — Graph → MLIR (StableHLO) text lowering.
//
// `GraphToMLIR` walks a topologically-sorted `tenzor::jit::Graph` and emits
// a complete MLIR module (`module { func.func @main(...) -> ... { ... } }`)
// as plain text. Per the 2026-05-19 amendment, this is the input to
// `compile_mlir(text, opts)` which drives the IREE compiler embedding API.
//
// Group B / B.1 implements `OpType::Add` only. All other op types throw
// `std::runtime_error("OpType <name> not yet supported")` to make the
// next-task surface obvious. Subsequent tasks (B.2-B.6 then Group C/D)
// fill in the remaining handlers.

#pragma once

#include "tenzor/jit/graph.hpp"

#include <string>

namespace tenzor::jit::mlir_jit {

/// Visitor that walks a `tenzor::jit::Graph` in topological order and emits
/// a complete StableHLO MLIR module text. Stateful (carries an SSA value-name
/// map and a monotonic counter) but not thread-safe; one instance per lower
/// call.
class GraphToMLIR {
public:
    GraphToMLIR();

    /// When true (the default), the 4 Tenzor dialect ops (FlashAttention,
    /// GQA, RoPE, RMSNorm) are lowered to `stablehlo.custom_call
    /// @tenzor_<name>` text. When false, they are decomposed into pure
    /// StableHLO primitives (the "expand" path used for deploy targets
    /// without the Tenzor runtime plugin). All non-dialect ops are
    /// unaffected.
    auto set_plugin_enabled(bool enabled) -> void { plugin_enabled_ = enabled; }
    auto plugin_enabled() const -> bool { return plugin_enabled_; }

    /// Walk the topologically-sorted Graph and emit a complete MLIR module
    /// (module { func.func @main(...) -> ... { ... } }) as text.
    ///
    /// Throws std::runtime_error if the graph contains an `OpType` that has
    /// not yet been wired into this lowerer. The exception message includes
    /// the unsupported `OpType` name so callers can route to the eager path.
    auto lower(const ::tenzor::jit::Graph& g) -> std::string;

private:
    bool plugin_enabled_ = true;
};

}  // namespace tenzor::jit::mlir_jit
