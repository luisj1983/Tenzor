/**
 * @file compile.cpp
 * @brief Implementation of automatic graph capture and compilation
 */

#include "tenzor/jit/compile.hpp"
#include "tenzor/backend/dispatch_interceptor.hpp"
#include <sstream>

namespace tenzor {
namespace jit {

// ============================================================================
// CompiledFunction
// ============================================================================

CompiledFunction::CompiledFunction(FnType fn, CompileConfig config)
    : fn_(std::move(fn)), config_(std::move(config)) {}

auto CompiledFunction::operator()(const Variable& input) -> Variable {
    auto key = shape_key(input);

    // Fast path: check cache without lock
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            // Cache hit: execute compiled graph
            return it->second->forward(input);
        }
    }

    // Cache miss: trace, compile, and cache
    // Always execute eagerly first for correctness
    auto result = fn_(input);

    // Attempt compilation in the background
    try {
        auto compiled = trace_and_compile(input);
        if (compiled) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (cache_.size() < static_cast<size_t>(config_.max_retraces)) {
                cache_[key] = compiled;
            }
        }
    } catch (const std::exception& e) {
        // Compilation failed — fall back to eager execution
        // This is expected for certain dynamic patterns
    }

    return result;
}

auto CompiledFunction::num_cached() const -> size_t {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.size();
}

auto CompiledFunction::clear_cache() -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
}

auto CompiledFunction::shape_key(const Variable& input) -> std::string {
    std::ostringstream ss;
    auto shape = input.tensor().shape();
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) ss << "x";
        ss << shape[i];
    }
    ss << "_" << static_cast<int>(input.tensor().dtype());
    ss << "_" << static_cast<int>(input.tensor().device().type);
    return ss.str();
}

auto CompiledFunction::trace_and_compile(const Variable& input)
    -> std::shared_ptr<CompiledModule> {
    had_graph_break_ = false;

    auto& tracer = Tracer::get_instance();
    tracer.start_trace();

    // Push a tracing interceptor onto the dispatch stack
    auto interceptor = make_tracing_interceptor(tracer,
        [this](OpId op) {
            had_graph_break_ = true;
            if (config_.fullgraph) {
                throw std::runtime_error(
                    "tenzor.compile(fullgraph=True): graph break at OpId "
                    + std::to_string(static_cast<int>(op)));
            }
        });

    DispatchInterceptorStack::push(std::move(interceptor));

    // Execute the function (ops are recorded by the interceptor)
    Variable output;
    try {
        output = fn_(input);
    } catch (...) {
        DispatchInterceptorStack::pop();
        tracer.clear();
        throw;
    }

    DispatchInterceptorStack::pop();

    if (had_graph_break_ && config_.fullgraph) {
        tracer.clear();
        return nullptr;
    }

    // Build graph from traced ops
    auto graph = tracer.end_trace({input}, {output});
    if (!graph || graph->num_nodes() == 0) {
        return nullptr;
    }

    // Optimize the graph
    if (config_.enable_fusion) {
        Compiler compiler(true);
        compiler.optimize(*graph);
    }

    // Wrap in a CompiledModule
    auto compiled = std::make_shared<CompiledModule>();
    compiled->set_graph(graph);

    return compiled;
}

// ============================================================================
// Free function
// ============================================================================

auto compile(CompiledFunction::FnType fn, CompileConfig config) -> CompiledFunction {
    return CompiledFunction(std::move(fn), std::move(config));
}

} // namespace jit
} // namespace tenzor
