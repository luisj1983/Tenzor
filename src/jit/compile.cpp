/**
 * @file compile.cpp
 * @brief Implementation of automatic graph capture and compilation
 */

#include "tenzor/jit/compile.hpp"
#include "tenzor/jit/compiler.hpp"
#include "tenzor/backend/dispatch_interceptor.hpp"

#ifdef TENZOR_HAS_MLIR_JIT
#include "tenzor/jit/mlir/iree_compile.hpp"
#include "tenzor/jit/mlir/iree_runtime.hpp"
#include "tenzor/jit/mlir/lowering.hpp"
#endif

#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace tenzor {
namespace jit {

// ============================================================================
// MLIR backend cache (Pimpl)
// ============================================================================

namespace mlir_detail {

#ifdef TENZOR_HAS_MLIR_JIT
struct MlirInvokerCache {
    std::unordered_map<std::string,
                       std::unique_ptr<::tenzor::jit::mlir_jit::IreeInvoker>>
        invokers;
};
#else
struct MlirInvokerCache {};
#endif

}  // namespace mlir_detail

// ============================================================================
// CompiledFunction
// ============================================================================

CompiledFunction::CompiledFunction(FnType fn, CompileConfig config)
    : fn_(std::move(fn)), config_(std::move(config)) {
    if (config_.mode != "default" && config_.mode != "reduce-overhead" && config_.mode != "max-autotune") {
        throw std::invalid_argument("Unknown compilation mode: " + config_.mode +
            ". Valid modes: \"default\", \"reduce-overhead\", \"max-autotune\"");
    }
    if (config_.backend != "nvrtc" && config_.backend != "mlir") {
        throw std::invalid_argument(
            "Unknown compile backend: " + config_.backend +
            R"_(. Valid backends: "nvrtc", "mlir")_");
    }
    if (config_.backend == "mlir") {
#ifndef TENZOR_HAS_MLIR_JIT
        throw std::runtime_error(
            "CompiledFunction: backend=\"mlir\" requested but Tenzor was "
            "built without TENZOR_USE_MLIR_JIT=ON");
#else
        mlir_cache_ = std::make_unique<mlir_detail::MlirInvokerCache>();
#endif
    }
}

CompiledFunction::~CompiledFunction() = default;

auto CompiledFunction::operator()(const Variable& input) -> Variable {
    if (config_.backend == "mlir") {
        return mlir_invoke(input);
    }

    auto key = shape_key(input);

    // Fast path: check cache without lock
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            auto& compiled_module = it->second;

            // reduce-overhead mode: capture and replay CUDA graphs
            if (config_.mode == "reduce-overhead") {
                if (compiled_module->has_cuda_graph()) {
                    // Replay captured graph
                    std::vector<Tensor> inputs = {input.tensor()};
                    compiled_module->replay_cuda_graph(inputs);
                    return compiled_module->forward(input);
                }
                ++warmup_count_;
                if (warmup_count_ >= kReduceOverheadWarmupCalls) {
                    // Capture CUDA graph on this execution
                    auto result = compiled_module->forward(input);
                    compiled_module->capture_cuda_graph({input.tensor()});
                    return result;
                }
            }

            // Cache hit: execute compiled graph
            return compiled_module->forward(input);
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
    break_positions_.clear();
    size_t traced_op_count = 0;

    auto& tracer = Tracer::get_instance();
    tracer.start_trace();

    // Push a tracing interceptor onto the dispatch stack
    auto interceptor = make_tracing_interceptor(tracer,
        [this, &traced_op_count](OpId op) {
            had_graph_break_ = true;
            break_positions_.push_back(traced_op_count);
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

    // Mode-specific optimizations
    if (config_.mode == "max-autotune") {
        // Enable layout and dtype optimization passes
        Compiler autotune_compiler(false);  // no default passes
        autotune_compiler.add_pass(std::make_unique<LayoutOptimizationPass>());

        auto dtype_pass = std::make_unique<DTypeOptimizationPass>();
        dtype_pass->set_target_dtype(DType::Float16);
        autotune_compiler.add_pass(std::move(dtype_pass));

        autotune_compiler.optimize(*graph, 1);
    }

    if (had_graph_break_ && !config_.fullgraph) {
        // Graph breaks occurred - compile partial graph
        // Future: segment into sub-graphs for mixed compiled/eager execution
    }

    // Wrap in a CompiledModule
    auto compiled = std::make_shared<CompiledModule>();
    compiled->set_graph(graph);

    return compiled;
}

// ============================================================================
// MLIR backend invoke
// ============================================================================

#ifdef TENZOR_HAS_MLIR_JIT
namespace {

/// Resolve `cfg.target == "auto"` based on the input tensor's device.
auto resolve_target(const std::string& cfg_target,
                    const ::tenzor::Device& dev) -> std::string {
    if (cfg_target != "auto") return cfg_target;
    switch (dev.type) {
        case ::tenzor::Device::Type::CPU:    return "llvm-cpu";
        case ::tenzor::Device::Type::CUDA:   return "cuda";
        case ::tenzor::Device::Type::ROCm:   return "rocm";
        case ::tenzor::Device::Type::Vulkan: return "vulkan-spirv";
        default:
            throw std::runtime_error(
                "MLIR backend: no IREE target mapped for device type " +
                std::to_string(static_cast<int>(dev.type)));
    }
}

/// Build the Graph for a single-input function by running the tracer.
/// Mirrors `CompiledFunction::trace_and_compile` (the NVRTC path): pushes
/// a tracing interceptor onto the dispatch stack so every backend dispatch
/// is recorded as a TracedOp. Without the interceptor, the tracer sees no
/// ops and `end_trace` returns an empty graph.
auto trace_single_input_graph(CompiledFunction::FnType& fn,
                              const Variable& input) -> std::shared_ptr<Graph> {
    auto& tracer = Tracer::get_instance();
    tracer.start_trace();
    auto interceptor = make_tracing_interceptor(tracer, [](OpId /*op*/) {
        // No graph-break handling here — the MLIR path treats any
        // break as "fall back to eager" by returning a partial/empty
        // graph from mlir_invoke().
    });
    DispatchInterceptorStack::push(std::move(interceptor));

    Variable output;
    try {
        output = fn(input);
    } catch (...) {
        DispatchInterceptorStack::pop();
        tracer.clear();
        throw;
    }
    DispatchInterceptorStack::pop();
    return tracer.end_trace({input}, {output});
}

}  // namespace

auto CompiledFunction::mlir_invoke(const Variable& input) -> Variable {
    namespace mj = ::tenzor::jit::mlir_jit;

    const auto key = shape_key(input);

    // Cache-hit fast path.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = mlir_cache_->invokers.find(key);
        if (it != mlir_cache_->invokers.end()) {
            auto outs = it->second->invoke({input.tensor()});
            if (outs.size() != 1) {
                throw std::runtime_error(
                    "MLIR backend: expected exactly 1 output tensor from "
                    "@main, got " +
                    std::to_string(outs.size()));
            }
            return Variable(std::move(outs[0]), input.requires_grad());
        }
    }

    // Cache miss: trace → lower → compile → load invoker → cache.
    auto graph = trace_single_input_graph(fn_, input);
    if (!graph || graph->num_nodes() == 0) {
        // Trace failed or produced nothing: fall back to eager so the
        // caller still gets a result for this invocation. Subsequent
        // calls will retry.
        return fn_(input);
    }

    if (config_.enable_fusion) {
        Compiler compiler(true);
        compiler.optimize(*graph);
    }

    mj::GraphToMLIR lowerer;
    const std::string mlir_text = lowerer.lower(*graph);

    mj::CompileOptions opts;
    opts.target          = resolve_target(config_.target, input.tensor().device());
    opts.plugin_enabled  = false;  // No tenzor_* custom_calls in B.2.

    auto artifact = mj::compile_mlir(mlir_text, opts);
    auto invoker  = mj::IreeInvoker::load(artifact);

    auto outs = invoker->invoke({input.tensor()});
    if (outs.size() != 1) {
        throw std::runtime_error(
            "MLIR backend: expected exactly 1 output tensor from @main, got " +
            std::to_string(outs.size()));
    }
    Variable result(std::move(outs[0]), input.requires_grad());

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (mlir_cache_->invokers.size() <
            static_cast<size_t>(config_.max_retraces)) {
            mlir_cache_->invokers.emplace(key, std::move(invoker));
        }
    }
    return result;
}

#else  // TENZOR_HAS_MLIR_JIT

auto CompiledFunction::mlir_invoke(const Variable& /*input*/) -> Variable {
    throw std::runtime_error(
        "CompiledFunction::mlir_invoke: Tenzor was built without "
        "TENZOR_USE_MLIR_JIT=ON");
}

#endif  // TENZOR_HAS_MLIR_JIT

// ============================================================================
// Free function
// ============================================================================

auto compile(CompiledFunction::FnType fn, CompileConfig config) -> CompiledFunction {
    return CompiledFunction(std::move(fn), std::move(config));
}

} // namespace jit
} // namespace tenzor
