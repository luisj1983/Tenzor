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

#include <cstddef>
#include <string>
#include <vector>

namespace tenzor::jit::mlir_jit {

/// One closure-captured parameter/buffer leaf lowered as an extra @main
/// function argument beyond the graph's declared inputs (JIT-R109). Order
/// matches the extra `%argN`s lower() appended: `is_buffer=false` entries
/// index into `Graph::parameters()`, `is_buffer=true` into
/// `Graph::buffers()`.
struct ParamBufferLeafArg {
    bool is_buffer;
    std::size_t index;
};

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

    /// Valid after lower() returns: the extra @main arguments appended after
    /// `g.inputs()` for the graph's parameter/buffer leaves (JIT-R109), in
    /// the exact order lower() bound them to `%argN`s. Callers invoking the
    /// compiled module must append each leaf's CURRENT tensor value, in this
    /// same order, after the regular inputs -- these are never baked as
    /// constants, so the compiled module has no value for them otherwise.
    auto leaf_args() const -> const std::vector<ParamBufferLeafArg>& {
        return leaf_args_;
    }

private:
    bool plugin_enabled_ = true;
    std::vector<ParamBufferLeafArg> leaf_args_;
};

}  // namespace tenzor::jit::mlir_jit
