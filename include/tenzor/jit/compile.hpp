/**
 * @file compile.hpp
 * @brief Automatic graph capture and compilation (torch.compile equivalent)
 *
 * Provides tenzor::jit::compile() which wraps a callable and automatically
 * traces + compiles it on first invocation. Subsequent calls with matching
 * shapes use the cached compiled graph. Shape mismatches trigger recompilation.
 *
 * Usage:
 * @code
 * auto model = std::make_shared<MyModel>();
 * auto compiled = tenzor::jit::compile([&](const Variable& x) {
 *     return model->forward(x);
 * });
 *
 * // First call: traces and compiles
 * auto out1 = compiled(input1);
 *
 * // Second call: uses cached compiled graph (same shape)
 * auto out2 = compiled(input2);
 *
 * // Different shape: re-traces and caches new graph
 * auto out3 = compiled(input_different_shape);
 * @endcode
 */

#pragma once

#include <functional>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include "../autograd/variable.hpp"
#include "compiler.hpp"
#include "tracer.hpp"
#include "tracing_interceptor.hpp"

namespace tenzor {
namespace jit {

// Forward declaration
class CompiledModule;

namespace mlir_detail {
/// Pimpl holder for the MLIR backend's per-shape cache of IreeInvoker
/// instances. Defined in compile.cpp so that compile.hpp does not pull
/// in `iree_runtime.hpp` (which transitively requires the IREE headers).
struct MlirInvokerCache;
}  // namespace mlir_detail


/**
 * @brief Configuration for compiled functions.
 */
struct CompileConfig {
    bool fullgraph{false};       ///< If true, error on graph break; if false, allow partial tracing
    int max_retraces{8};         ///< Maximum number of shape specializations to cache
    bool enable_fusion{true};    ///< Run fusion passes on compiled graph
    std::string mode{"default"}; ///< Compilation mode: "default", "reduce-overhead", "max-autotune"

    /// Compiler backend selection.
    ///   "nvrtc" (default, legacy CUDA C++ codegen via NVRTC),
    ///   "mlir"  (Phase 13: Graph → StableHLO text → iree-compile → IreeInvoker).
    std::string backend{"nvrtc"};

    /// Target for the MLIR backend. Ignored when backend != "mlir".
    /// One of: "auto" (pick from input device), "llvm-cpu", "cuda",
    /// "vulkan-spirv", "rocm".
    std::string target{"auto"};

    /// When true, a compilation failure inside `CompiledFunction::operator()`
    /// is re-thrown rather than silently degrading to eager execution. The
    /// environment variable `TENZOR_JIT_STRICT=1` flips this on for any
    /// caller that did not set it explicitly. Defaults to false to preserve
    /// historical "compile-or-eager" behaviour for callers that opt out
    /// (`TENZOR_JIT_STRICT=0`).
    bool strict{false};
};

/**
 * @brief A compiled function that auto-traces on first call.
 *
 * Wraps a user function and manages the trace/compile/cache lifecycle.
 * Thread-safe for concurrent calls with different shapes.
 */
class CompiledFunction {
public:
    /// Single-input function signature (legacy / common case). Retained as
    /// the public type alias so callers like `tz::jit::compile(...)` keep
    /// working unchanged when they pass a `Variable(const Variable&)` lambda.
    using FnType = std::function<Variable(const Variable&)>;

    /// N-input function signature. The captured callable receives a span of
    /// inputs and returns a single Variable. Construction from a single-input
    /// FnType wraps the original callable into this form.
    using FnTypeN = std::function<Variable(std::span<const Variable>)>;

    /**
     * @brief Construct a compiled function wrapper from a single-input fn.
     *
     * Retained for backward compatibility: existing callers that pass a
     * `Variable(const Variable&)` callable continue to work via this
     * overload, which wraps the function into the N-input form internally.
     *
     * @param fn The function to compile
     * @param config Compilation configuration
     */
    CompiledFunction(FnType fn, CompileConfig config = {});

    /**
     * @brief Construct a compiled function wrapper from an N-input fn.
     */
    CompiledFunction(FnTypeN fn, CompileConfig config = {});

    /// Destructor declared out-of-line because `mlir_cache_` holds a
    /// std::unique_ptr to an incomplete type.
    ~CompiledFunction();

    // Move / copy are implicitly deleted by the std::mutex member; callers
    // construct via guaranteed-RVO from `jit::compile(...)`. Keep
    // implicit-delete in place rather than declaring defaults that would
    // be ill-formed.

    /**
     * @brief Execute the compiled function (single-input convenience).
     *
     * Forwards to the N-input overload with a 1-element span. Preserves
     * the historical `compiled_fn(x)` call site.
     *
     * @param input Input variable
     * @return Output variable
     */
    auto operator()(const Variable& input) -> Variable;

    /**
     * @brief Execute the compiled function with N inputs.
     *
     * On first call (or cache miss) for a given input-shape tuple, traces
     * the function with the supplied inputs and compiles the resulting
     * graph. Subsequent calls with matching shapes reuse the cached
     * compilation. Shape mismatches trigger a re-trace.
     *
     * @param inputs Input variables (any positive arity)
     * @return Output variable
     */
    auto operator()(std::span<const Variable> inputs) -> Variable;

    /**
     * @brief Variadic convenience for `compiled(x, y, z, ...)` calls.
     *
     * Packs the arguments into a fixed-size array and forwards to the
     * span-based overload. SFINAE-restricted to `Variable` arguments so
     * the single-Variable overload above is still selected unambiguously.
     */
    template <class... Vs,
              class = std::enable_if_t<(sizeof...(Vs) >= 2) &&
                  (std::is_same_v<std::decay_t<Vs>, Variable> && ...)>>
    auto operator()(Vs&&... inputs) -> Variable {
        const Variable arr[] = {static_cast<const Variable&>(inputs)...};
        return (*this)(std::span<const Variable>(arr, sizeof...(Vs)));
    }

    /**
     * @brief Get the number of cached shape specializations.
     */
    auto num_cached() const -> size_t;

    /**
     * @brief Clear all cached compilations.
     */
    auto clear_cache() -> void;

    /**
     * @brief Check if a graph break occurred during the last trace.
     */
    auto had_graph_break() const -> bool { return had_graph_break_; }

    /**
     * @brief Trace the function (if needed) and dump the post-optimization
     *        tenzor::jit::Graph IR as text. Used by tz.jit.show_graph().
     *
     * Forces a trace at the given input shape/dtype/device, runs the same
     * fusion pass mlir_invoke would, and returns Graph::to_string(). Does
     * not compile or invoke the MLIR pipeline.
     */
    auto dump_graph(const Variable& input) -> std::string;
    auto dump_graph(std::span<const Variable> inputs) -> std::string;

    /**
     * @brief Trace + lower the function and return the StableHLO MLIR text
     *        that would be passed to iree-compile. Used by
     *        tz.jit.show_mlir().
     *
     * Mirrors the MLIR path: same trace, same fusion pass, same lowering
     * with plugin_enabled=true (so Tenzor custom_calls survive in the text).
     */
    auto dump_mlir(const Variable& input) -> std::string;
    auto dump_mlir(std::span<const Variable> inputs) -> std::string;

    /**
     * @brief Like dump_mlir, but lowers with plugin_enabled=false so all
     *        Tenzor custom_call ops are expanded to pure StableHLO. Used by
     *        tz.jit.show_stablehlo().
     */
    auto dump_stablehlo(const Variable& input) -> std::string;
    auto dump_stablehlo(std::span<const Variable> inputs) -> std::string;

    /**
     * @brief Run iree-compile with --mlir-print-ir-after-all on the lowered
     *        MLIR text and return the captured pipeline IR dump as a single
     *        string. Used by tz.jit.show_iree().
     */
    auto dump_iree(const Variable& input) -> std::string;
    auto dump_iree(std::span<const Variable> inputs) -> std::string;

private:
    /// Always stored as the N-input form. Single-input constructors wrap.
    FnTypeN fn_;
    CompileConfig config_;
    mutable std::mutex mutex_;
    bool had_graph_break_{false};

    /// Track warmup calls for reduce-overhead mode (CUDA graph capture)
    int warmup_count_{0};
    static constexpr int kReduceOverheadWarmupCalls = 2;


    /// Break positions recorded during tracing
    std::vector<size_t> break_positions_;

    /// Cache key: shape signature (e.g., "4x3x224x224_f32_cpu")
    std::unordered_map<std::string, std::shared_ptr<CompiledModule>> cache_;

    /// Per-shape cache of IreeInvoker instances for the MLIR backend.
    /// Pimpl-owned so that this header does not pull in the IREE headers.
    /// Empty/null when `config_.backend != "mlir"`.
    std::unique_ptr<mlir_detail::MlirInvokerCache> mlir_cache_;

    /// Compute cache key from input tensor properties.
    static auto shape_key(const Variable& input) -> std::string;
    static auto shape_key(std::span<const Variable> inputs) -> std::string;

    /// Trace and compile the function for the given inputs.
    auto trace_and_compile(std::span<const Variable> inputs)
        -> std::shared_ptr<CompiledModule>;

    /// MLIR backend invoke path. Routes inputs → trace → lower → iree-compile →
    /// IreeInvoker. Cached on a shape-key basis the same way the NVRTC path
    /// is. Falls back to eager when the inputs are not on a device the MLIR
    /// pipeline supports yet.
    auto mlir_invoke(std::span<const Variable> inputs) -> Variable;
};

/**
 * @brief Compile a function for automatic graph capture.
 *
 * Returns a CompiledFunction that traces on first call and caches
 * the compiled graph for subsequent calls.
 *
 * @param fn Function taking a Variable and returning a Variable
 * @param config Optional compilation configuration
 * @return Compiled function wrapper
 *
 * @code
 * auto fast_fn = tenzor::jit::compile([](const Variable& x) {
 *     return autograd::relu(autograd::matmul(x, weights));
 * });
 * auto result = fast_fn(input);
 * @endcode
 */
auto compile(CompiledFunction::FnType fn, CompileConfig config = {})
    -> CompiledFunction;

/// N-input overload of compile().
auto compile(CompiledFunction::FnTypeN fn, CompileConfig config = {})
    -> CompiledFunction;

} // namespace jit
} // namespace tenzor
