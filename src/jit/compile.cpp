/**
 * @file compile.cpp
 * @brief Implementation of automatic graph capture and compilation
 */

#include "tenzor/jit/compile.hpp"
#include "tenzor/jit/compiler.hpp"
#include "tenzor/backend/dispatch_interceptor.hpp"

#ifdef TENZOR_HAS_MLIR_JIT
#include "tenzor/jit/mlir/iree_compile.hpp"
#include "tenzor/jit/mlir/iree_paths.hpp"
#include "tenzor/jit/mlir/iree_runtime.hpp"
#include "tenzor/jit/mlir/lowering.hpp"
#endif

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unistd.h>
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

namespace {

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

// ============================================================================
// Debug / introspection APIs (Group F.1 — show_graph)
// ============================================================================

auto CompiledFunction::dump_graph(const Variable& input) -> std::string {
    auto graph = trace_single_input_graph(fn_, input);
    if (!graph || graph->num_nodes() == 0) {
        return "<empty graph>\n";
    }
    if (config_.enable_fusion) {
        Compiler compiler(true);
        compiler.optimize(*graph);
    }
    return graph->to_string();
}

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

}  // namespace

auto CompiledFunction::mlir_invoke(const Variable& input) -> Variable {
    namespace mj = ::tenzor::jit::mlir_jit;

    const auto key = shape_key(input);

    // Cache-hit fast path.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = mlir_cache_->invokers.find(key);
        if (it != mlir_cache_->invokers.end()) {
            mj::internal::record_cache_hit();
            auto outs = it->second->invoke({input.tensor()});
            if (outs.size() != 1) {
                throw std::runtime_error(
                    "MLIR backend: expected exactly 1 output tensor from "
                    "@main, got " +
                    std::to_string(outs.size()));
            }
            return Variable(std::move(outs[0]), input.requires_grad());
        }
        // Miss. If we've already compiled something else for a different
        // shape, this counts as a retrace (a new shape forces a fresh trace).
        mj::internal::record_cache_miss();
        if (!mlir_cache_->invokers.empty()) {
            mj::internal::record_retrace();
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
    // Plugin path is the default — the lowering emits call @tenzor_plugin.<op>
    // for the 4 dialect ops and the in-process IreeInvoker registers a VM
    // native module that resolves them. Graphs that contain none of those ops
    // are equally compatible (no extern decls are produced).
    lowerer.set_plugin_enabled(true);
    const std::string mlir_text = lowerer.lower(*graph);

    mj::CompileOptions opts;
    opts.target          = resolve_target(config_.target, input.tensor().device());
    opts.plugin_enabled  = true;

    auto artifact = mj::compile_mlir(mlir_text, opts);
    // Mode::InProcess registers the tenzor_plugin VM module before loading
    // the bytecode — required whenever the compiled .vmfb references any
    // tenzor_plugin.<op> import. For targets where the linked runtime lacks
    // the HAL driver (e.g. distributions without cuda/vulkan compiled in)
    // we fall back to the subprocess driver, which carries its own driver
    // set; this preserves the GPU end-to-end path without a CPU fallback.
    std::unique_ptr<mj::IreeInvoker> invoker;
    try {
        invoker = mj::IreeInvoker::load(artifact,
                                        mj::IreeInvoker::Mode::InProcess);
    } catch (const mj::JitInvokeError& e) {
        const std::string what = e.what();
        if (what.find("NOT_FOUND") != std::string::npos &&
            what.find("driver") != std::string::npos) {
            invoker = mj::IreeInvoker::load(artifact,
                                            mj::IreeInvoker::Mode::Subprocess);
        } else {
            throw;
        }
    }

    // Optional artifact dumping: when TENZOR_JIT_DUMP=<dir> is set, write the
    // full pipeline (graph text + MLIR + expanded StableHLO + iree-compile
    // pipeline dump + path to the vmfb) into a hash-keyed subdirectory.
    if (const char* dump_dir = std::getenv("TENZOR_JIT_DUMP"); dump_dir && *dump_dir) {
        try {
            namespace fs = std::filesystem;
            const std::string key = mj::compute_cache_key(mlir_text, opts.target);
            const fs::path subdir = fs::path(dump_dir) / key;
            fs::create_directories(subdir);

            {
                std::ofstream f(subdir / "graph.txt", std::ios::trunc);
                f << graph->to_string();
            }
            {
                std::ofstream f(subdir / "mlir.txt",
                                std::ios::binary | std::ios::trunc);
                f.write(mlir_text.data(),
                        static_cast<std::streamsize>(mlir_text.size()));
            }
            {
                // Re-lower with plugin_enabled=false to dump the deploy form.
                mj::GraphToMLIR sh_lowerer;
                sh_lowerer.set_plugin_enabled(false);
                const std::string sh_text = sh_lowerer.lower(*graph);
                std::ofstream f(subdir / "stablehlo.txt",
                                std::ios::binary | std::ios::trunc);
                f.write(sh_text.data(),
                        static_cast<std::streamsize>(sh_text.size()));
            }
            {
                // Best-effort iree-compile pipeline dump. Shares the same
                // logic as dump_iree() but inlined so a single compile
                // produces the complete dump set without an extra trace.
                // Use the same discovery chain compile_mlir() uses so the
                // dump always lines up with what actually ran.
                const std::string& iree_compile =
                    ::tenzor::jit::mlir_jit::resolve_iree_compile();
                const fs::path tmp_in =
                    fs::temp_directory_path() /
                    ("tz_jit_dump_iree_" +
                     std::to_string(::getpid()) + "_" + key.substr(0, 8) +
                     ".mlir");
                {
                    std::ofstream f(tmp_in,
                                    std::ios::binary | std::ios::trunc);
                    f.write(mlir_text.data(),
                            static_cast<std::streamsize>(mlir_text.size()));
                }
                const std::string cmd =
                    iree_compile + " --iree-input-type=stablehlo" +
                    " --iree-hal-target-backends=" + opts.target +
                    " --mlir-disable-threading" +
                    " --mlir-print-ir-after-all" +
                    " --compile-to=vm" +
                    " -o /dev/null " + tmp_in.string() + " 2>&1";
                std::string captured;
                if (FILE* pipe = ::popen(cmd.c_str(), "r")) {
                    char buf[4096];
                    while (auto n = std::fread(buf, 1, sizeof(buf), pipe)) {
                        captured.append(buf, n);
                    }
                    ::pclose(pipe);
                }
                std::ofstream f(subdir / "iree.log",
                                std::ios::binary | std::ios::trunc);
                f.write(captured.data(),
                        static_cast<std::streamsize>(captured.size()));
                std::error_code _ec;
                fs::remove(tmp_in, _ec);
            }
            {
                // Record where the cached vmfb lives so the dump dir is
                // self-describing without copying the bytecode itself.
                std::ofstream f(subdir / "vmfb.path", std::ios::trunc);
                f << artifact.vmfb_path.string() << "\n";
            }
        } catch (const std::exception&) {
            // Dumping is best-effort: never fail compilation because of it.
        }
    }

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
        } else {
            // Hit the per-function cache ceiling — count this insertion
            // attempt as an eviction (the new shape's invoker is dropped on
            // the floor, so the next call for it will re-trace).
            mj::internal::record_eviction();
        }
    }
    return result;
}

// ============================================================================
// Debug / introspection APIs (Group F.2 / F.3 — show_mlir / show_stablehlo)
// ============================================================================

auto CompiledFunction::dump_mlir(const Variable& input) -> std::string {
    namespace mj = ::tenzor::jit::mlir_jit;
    auto graph = trace_single_input_graph(fn_, input);
    if (!graph || graph->num_nodes() == 0) {
        return "<empty graph: nothing to lower>\n";
    }
    if (config_.enable_fusion) {
        Compiler compiler(true);
        compiler.optimize(*graph);
    }
    mj::GraphToMLIR lowerer;
    lowerer.set_plugin_enabled(true);
    return lowerer.lower(*graph);
}

auto CompiledFunction::dump_stablehlo(const Variable& input) -> std::string {
    namespace mj = ::tenzor::jit::mlir_jit;
    auto graph = trace_single_input_graph(fn_, input);
    if (!graph || graph->num_nodes() == 0) {
        return "<empty graph: nothing to lower>\n";
    }
    if (config_.enable_fusion) {
        Compiler compiler(true);
        compiler.optimize(*graph);
    }
    mj::GraphToMLIR lowerer;
    // The defining difference vs. dump_mlir: expand all `tenzor.*` custom_call
    // ops into pure StableHLO primitives. The text returned here is the form
    // shipped to deploy targets that do not link the Tenzor runtime plugin.
    lowerer.set_plugin_enabled(false);
    return lowerer.lower(*graph);
}

auto CompiledFunction::dump_iree(const Variable& input) -> std::string {
    const std::string mlir_text = dump_mlir(input);
    if (mlir_text.rfind("<empty", 0) == 0) {
        return mlir_text;
    }

    // Resolve the IREE target the same way mlir_invoke does so the dump
    // exactly mirrors what the runtime compiles.
    const std::string target =
        resolve_target(config_.target, input.tensor().device());

    // Write the MLIR text to a temp file and shell out to iree-compile with
    // --compile-to=vm (the fast pipeline that produces no .vmfb, just runs
    // up to the VM stage) plus --mlir-print-ir-after-all. The full pipeline
    // IR is printed to stderr; we capture both streams via 2>&1.
    namespace fs = std::filesystem;
    const fs::path tmp_in =
        fs::temp_directory_path() /
        ("tz_jit_dump_iree_" + std::to_string(::getpid()) + ".mlir");
    {
        std::ofstream f(tmp_in, std::ios::binary | std::ios::trunc);
        f.write(mlir_text.data(),
                static_cast<std::streamsize>(mlir_text.size()));
    }

    // Use the same discovery chain compile_mlir() uses (env override → pip
    // venv → CMake-found dist → $PATH). show_iree must dump the IR for the
    // exact compiler that actually ran during compile_mlir(), so any
    // divergence here would be a bug.
    const std::string& iree_compile =
        ::tenzor::jit::mlir_jit::resolve_iree_compile();

    const std::string cmd =
        iree_compile + " --iree-input-type=stablehlo" +
        " --iree-hal-target-backends=" + target +
        " --mlir-disable-threading" +
        " --mlir-print-ir-after-all" +
        " --compile-to=vm" +
        " -o /dev/null " + tmp_in.string() + " 2>&1";

    std::string captured;
    if (FILE* pipe = ::popen(cmd.c_str(), "r")) {
        char buf[4096];
        while (auto n = std::fread(buf, 1, sizeof(buf), pipe)) {
            captured.append(buf, n);
        }
        ::pclose(pipe);
    } else {
        captured = "<failed to spawn iree-compile via popen()>\n";
    }
    std::error_code _ec;
    fs::remove(tmp_in, _ec);
    return captured;
}

#else  // TENZOR_HAS_MLIR_JIT

auto CompiledFunction::mlir_invoke(const Variable& /*input*/) -> Variable {
    throw std::runtime_error(
        "CompiledFunction::mlir_invoke: Tenzor was built without "
        "TENZOR_USE_MLIR_JIT=ON");
}

auto CompiledFunction::dump_mlir(const Variable& /*input*/) -> std::string {
    throw std::runtime_error(
        "CompiledFunction::dump_mlir: Tenzor was built without "
        "TENZOR_USE_MLIR_JIT=ON");
}

auto CompiledFunction::dump_stablehlo(const Variable& /*input*/) -> std::string {
    throw std::runtime_error(
        "CompiledFunction::dump_stablehlo: Tenzor was built without "
        "TENZOR_USE_MLIR_JIT=ON");
}

auto CompiledFunction::dump_iree(const Variable& /*input*/) -> std::string {
    throw std::runtime_error(
        "CompiledFunction::dump_iree: Tenzor was built without "
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
