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
#include <memory>
#include <mutex>
#include <string>
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
 * @brief A segment of a compiled graph, split at graph breaks.
 */
struct GraphSegment {
    /// Compiled sub-graph (nullptr for eager/break segments)
    std::shared_ptr<CompiledModule> compiled;

    /// Node index in original graph (for break segments)
    size_t break_node_index{0};

    /// Whether this is a compiled segment or an eager break
    bool is_compiled{true};
};

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
};

/**
 * @brief A compiled function that auto-traces on first call.
 *
 * Wraps a user function and manages the trace/compile/cache lifecycle.
 * Thread-safe for concurrent calls with different shapes.
 */
class CompiledFunction {
public:
    using FnType = std::function<Variable(const Variable&)>;

    /**
     * @brief Construct a compiled function wrapper.
     *
     * @param fn The function to compile
     * @param config Compilation configuration
     */
    CompiledFunction(FnType fn, CompileConfig config = {});

    /// Destructor declared out-of-line because `mlir_cache_` holds a
    /// std::unique_ptr to an incomplete type.
    ~CompiledFunction();

    // Move / copy are implicitly deleted by the std::mutex member; callers
    // construct via guaranteed-RVO from `jit::compile(...)`. Keep
    // implicit-delete in place rather than declaring defaults that would
    // be ill-formed.

    /**
     * @brief Execute the compiled function.
     *
     * On first call (or cache miss), traces the function and compiles
     * the resulting graph. Returns the eager execution result.
     *
     * @param input Input variable
     * @return Output variable
     */
    auto operator()(const Variable& input) -> Variable;

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

    /**
     * @brief Trace + lower the function and return the StableHLO MLIR text
     *        that would be passed to iree-compile. Used by
     *        tz.jit.show_mlir().
     *
     * Mirrors the MLIR path: same trace, same fusion pass, same lowering
     * with plugin_enabled=true (so Tenzor custom_calls survive in the text).
     */
    auto dump_mlir(const Variable& input) -> std::string;

    /**
     * @brief Like dump_mlir, but lowers with plugin_enabled=false so all
     *        Tenzor custom_call ops are expanded to pure StableHLO. Used by
     *        tz.jit.show_stablehlo().
     */
    auto dump_stablehlo(const Variable& input) -> std::string;

    /**
     * @brief Run iree-compile with --mlir-print-ir-after-all on the lowered
     *        MLIR text and return the captured pipeline IR dump as a single
     *        string. Used by tz.jit.show_iree().
     */
    auto dump_iree(const Variable& input) -> std::string;

private:
    FnType fn_;
    CompileConfig config_;
    mutable std::mutex mutex_;
    bool had_graph_break_{false};

    /// Track warmup calls for reduce-overhead mode (CUDA graph capture)
    int warmup_count_{0};
    static constexpr int kReduceOverheadWarmupCalls = 2;

    /// Segmented execution for graphs with breaks
    std::vector<GraphSegment> segments_;
    bool has_segments_{false};

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

    /// Trace and compile the function for the given input.
    auto trace_and_compile(const Variable& input) -> std::shared_ptr<CompiledModule>;

    /// MLIR backend invoke path. Routes input → trace → lower → iree-compile →
    /// IreeInvoker. Cached on a shape-key basis the same way the NVRTC path
    /// is. Falls back to eager when the input is not on a device the MLIR
    /// pipeline supports yet.
    auto mlir_invoke(const Variable& input) -> Variable;
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

} // namespace jit
} // namespace tenzor
