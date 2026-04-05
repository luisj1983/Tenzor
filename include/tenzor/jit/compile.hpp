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

/**
 * @brief Configuration for compiled functions.
 */
struct CompileConfig {
    bool fullgraph{false};       ///< If true, error on graph break; if false, allow partial tracing
    int max_retraces{8};         ///< Maximum number of shape specializations to cache
    bool enable_fusion{true};    ///< Run fusion passes on compiled graph
    std::string mode{"default"}; ///< Compilation mode: "default", "reduce-overhead", "max-autotune"
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

private:
    FnType fn_;
    CompileConfig config_;
    mutable std::mutex mutex_;
    bool had_graph_break_{false};

    /// Cache key: shape signature (e.g., "4x3x224x224_f32_cpu")
    std::unordered_map<std::string, std::shared_ptr<CompiledModule>> cache_;

    /// Compute cache key from input tensor properties.
    static auto shape_key(const Variable& input) -> std::string;

    /// Trace and compile the function for the given input.
    auto trace_and_compile(const Variable& input) -> std::shared_ptr<CompiledModule>;
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
