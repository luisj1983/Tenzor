/**
 * @file compile.cpp
 * @brief Implementation of automatic graph capture and compilation
 */

#include "tenzor/jit/compile.hpp"
#include "tenzor/jit/compiler.hpp"
#include "tenzor/backend/dispatch_interceptor.hpp"
#include "tenzor/utils/log.hpp"

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
    : CompiledFunction(
          FnTypeN([fn = std::move(fn)](std::span<const Variable> inputs) {
              if (inputs.size() != 1) {
                  throw std::invalid_argument(
                      "CompiledFunction: this function was constructed from a "
                      "single-input callable but was invoked with " +
                      std::to_string(inputs.size()) + " inputs");
              }
              return fn(inputs[0]);
          }),
          std::move(config)) {}

CompiledFunction::CompiledFunction(FnTypeN fn, CompileConfig config)
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
    const Variable arr[1] = {input};
    return (*this)(std::span<const Variable>(arr, 1));
}

auto CompiledFunction::operator()(std::span<const Variable> inputs) -> Variable {
    if (inputs.empty()) {
        throw std::invalid_argument(
            "CompiledFunction::operator(): expected at least one input "
            "Variable, got zero");
    }

    if (config_.backend == "mlir") {
        return mlir_invoke(inputs);
    }

    auto key = shape_key(inputs);

    // Fast path: check cache without lock
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            auto& compiled_module = it->second;

            // reduce-overhead mode: capture and replay CUDA graphs
            if (config_.mode == "reduce-overhead") {
                std::vector<Tensor> input_tensors;
                input_tensors.reserve(inputs.size());
                for (const auto& v : inputs) input_tensors.push_back(v.tensor());

                if (compiled_module->has_cuda_graph()) {
                    // Replay captured graph
                    compiled_module->replay_cuda_graph(input_tensors);
                    if (inputs.size() == 1) {
                        return compiled_module->forward(inputs[0]);
                    }
                    std::vector<Variable> input_vars(inputs.begin(), inputs.end());
                    auto outs = compiled_module->forward(input_vars);
                    if (outs.empty()) {
                        throw std::runtime_error(
                            "CompiledFunction: compiled graph produced no outputs");
                    }
                    return outs[0];
                }
                ++warmup_count_;
                if (warmup_count_ >= kReduceOverheadWarmupCalls) {
                    // Capture CUDA graph on this execution
                    if (inputs.size() == 1) {
                        auto result = compiled_module->forward(inputs[0]);
                        compiled_module->capture_cuda_graph(input_tensors);
                        return result;
                    }
                    std::vector<Variable> input_vars(inputs.begin(), inputs.end());
                    auto outs = compiled_module->forward(input_vars);
                    compiled_module->capture_cuda_graph(input_tensors);
                    if (outs.empty()) {
                        throw std::runtime_error(
                            "CompiledFunction: compiled graph produced no outputs");
                    }
                    return outs[0];
                }
            }

            // Cache hit: execute compiled graph. Prefer the single-input
            // `forward(Variable)` overload when arity is 1 — it carries
            // additional dtype/device-mismatch retrace machinery that the
            // vector overload does not.
            if (inputs.size() == 1) {
                return compiled_module->forward(inputs[0]);
            }
            std::vector<Variable> input_vars(inputs.begin(), inputs.end());
            auto outs = compiled_module->forward(input_vars);
            if (outs.empty()) {
                throw std::runtime_error(
                    "CompiledFunction: compiled graph produced no outputs");
            }
            return outs[0];
        }
    }

    // Cache miss: trace, compile, and cache
    // Always execute eagerly first for correctness
    auto result = fn_(inputs);

    // Attempt compilation in the background
    try {
        auto compiled = trace_and_compile(inputs);
        if (compiled) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (cache_.size() < static_cast<size_t>(config_.max_retraces)) {
                cache_[key] = compiled;
            }
        }
    } catch (const std::exception& e) {
        // Compilation failed. The eager result above is already correct, but we
        // do not want this to be silent: it would hide regressions in the JIT
        // pipeline. Honour strict mode either from the per-call config or from
        // the global TENZOR_JIT_STRICT env var. In non-strict mode log a WARN
        // every time so callers can see exactly which graphs fail.
        bool strict = config_.strict;
        if (!strict) {
            if (const char* s = std::getenv("TENZOR_JIT_STRICT"); s && *s && *s != '0') {
                strict = true;
            }
        }
        if (strict) {
            throw;
        }
        TENZOR_LOG_WARN("JIT compilation failed ({} backend, target={}): {}. "
                        "Falling back to eager execution.",
                        config_.backend, config_.target, e.what());
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
    const Variable arr[1] = {input};
    return shape_key(std::span<const Variable>(arr, 1));
}

auto CompiledFunction::shape_key(std::span<const Variable> inputs) -> std::string {
    std::ostringstream ss;
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (i > 0) ss << "|";
        auto shape = inputs[i].tensor().shape();
        for (size_t j = 0; j < shape.size(); ++j) {
            if (j > 0) ss << "x";
            ss << shape[j];
        }
        ss << "_" << static_cast<int>(inputs[i].tensor().dtype());
        // Encode the FULL device (type AND ordinal index), not just the type.
        // On a multi-GPU box a module captured on cuda:0 must not be reused for
        // a cuda:1 input under the same shape/dtype key — that returns a graph
        // whose NVRTC module / CUDA-graph capture / cached device pointers are
        // bound to GPU 0, causing an illegal cross-device access on GPU 1.
        ss << "_" << inputs[i].tensor().device().to_string();
    }
    return ss.str();
}

auto CompiledFunction::trace_and_compile(std::span<const Variable> inputs)
    -> std::shared_ptr<CompiledModule> {
    had_graph_break_ = false;
    size_t traced_op_count = 0;

    auto& tracer = Tracer::get_instance();
    tracer.start_trace();

    // Push a tracing interceptor onto the dispatch stack
    auto interceptor = make_tracing_interceptor(tracer,
        [this, &traced_op_count](OpId op) {
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
        output = fn_(inputs);
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
    std::vector<Variable> input_vec(inputs.begin(), inputs.end());
    auto graph = tracer.end_trace(input_vec, {output});
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
auto trace_single_input_graph(CompiledFunction::FnTypeN& fn,
                              std::span<const Variable> inputs)
    -> std::shared_ptr<Graph> {
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
        output = fn(inputs);
    } catch (...) {
        DispatchInterceptorStack::pop();
        tracer.clear();
        throw;
    }
    DispatchInterceptorStack::pop();
    std::vector<Variable> input_vec(inputs.begin(), inputs.end());
    return tracer.end_trace(input_vec, {output});
}

}  // namespace

// ============================================================================
// Debug / introspection APIs (Group F.1 — show_graph)
// ============================================================================

auto CompiledFunction::dump_graph(const Variable& input) -> std::string {
    const Variable arr[1] = {input};
    return dump_graph(std::span<const Variable>(arr, 1));
}

auto CompiledFunction::dump_graph(std::span<const Variable> inputs) -> std::string {
    auto graph = trace_single_input_graph(fn_, inputs);
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

auto CompiledFunction::mlir_invoke(std::span<const Variable> inputs) -> Variable {
    namespace mj = ::tenzor::jit::mlir_jit;

    if (inputs.empty()) {
        throw std::invalid_argument(
            "CompiledFunction::mlir_invoke: expected at least one input");
    }

    const auto key = shape_key(inputs);

    // Materialize input tensors once; reused on hit path and after compile.
    std::vector<::tenzor::Tensor> input_tensors;
    input_tensors.reserve(inputs.size());
    for (const auto& v : inputs) input_tensors.push_back(v.tensor());

    bool any_requires_grad = false;
    for (const auto& v : inputs) {
        if (v.requires_grad()) { any_requires_grad = true; break; }
    }

    // Cache-hit fast path.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = mlir_cache_->invokers.find(key);
        if (it != mlir_cache_->invokers.end()) {
            mj::internal::record_cache_hit();
            auto outs = it->second->invoke(input_tensors);
            if (outs.size() != 1) {
                throw std::runtime_error(
                    "MLIR backend: expected exactly 1 output tensor from "
                    "@main, got " +
                    std::to_string(outs.size()));
            }
            return Variable(std::move(outs[0]), any_requires_grad);
        }
        // Miss. If we've already compiled something else for a different
        // shape, this counts as a retrace (a new shape forces a fresh trace).
        mj::internal::record_cache_miss();
        if (!mlir_cache_->invokers.empty()) {
            mj::internal::record_retrace();
        }
    }

    // Cache miss: trace → lower → compile → load invoker → cache.
    auto graph = trace_single_input_graph(fn_, inputs);
    if (!graph || graph->num_nodes() == 0) {
        // Trace produced nothing (graph break in fullgraph mode, or the
        // function only ran ops the interceptor doesn't capture). Either
        // surface a hard error (strict) or log + degrade to eager.
        bool strict = config_.strict;
        if (!strict) {
            if (const char* s = std::getenv("TENZOR_JIT_STRICT"); s && *s && *s != '0') {
                strict = true;
            }
        }
        if (strict) {
            throw std::runtime_error(
                "CompiledFunction::mlir_invoke: trace produced no graph "
                "(TENZOR_JIT_STRICT=1).");
        }
        TENZOR_LOG_WARN("MLIR JIT: trace produced no graph; falling back to "
                        "eager execution for this invocation.");
        return fn_(inputs);
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
    opts.target          = resolve_target(config_.target, inputs[0].tensor().device());
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
                    iree_compile +
                    // --iree-input-type=auto picks up StableHLO on both old
                    // (3.0–3.10) and new (3.11+) IREE. Older accepted
                    // =stablehlo but 3.11+ rejects it; "auto" is the safe
                    // common value.
                    " --iree-input-type=auto" +
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

    auto outs = invoker->invoke(input_tensors);
    if (outs.size() != 1) {
        throw std::runtime_error(
            "MLIR backend: expected exactly 1 output tensor from @main, got " +
            std::to_string(outs.size()));
    }
    Variable result(std::move(outs[0]), any_requires_grad);

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
    const Variable arr[1] = {input};
    return dump_mlir(std::span<const Variable>(arr, 1));
}

auto CompiledFunction::dump_mlir(std::span<const Variable> inputs) -> std::string {
    namespace mj = ::tenzor::jit::mlir_jit;
    auto graph = trace_single_input_graph(fn_, inputs);
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
    const Variable arr[1] = {input};
    return dump_stablehlo(std::span<const Variable>(arr, 1));
}

auto CompiledFunction::dump_stablehlo(std::span<const Variable> inputs) -> std::string {
    namespace mj = ::tenzor::jit::mlir_jit;
    auto graph = trace_single_input_graph(fn_, inputs);
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
    const Variable arr[1] = {input};
    return dump_iree(std::span<const Variable>(arr, 1));
}

auto CompiledFunction::dump_iree(std::span<const Variable> inputs) -> std::string {
    if (inputs.empty()) {
        throw std::invalid_argument(
            "CompiledFunction::dump_iree: expected at least one input");
    }
    const std::string mlir_text = dump_mlir(inputs);
    if (mlir_text.rfind("<empty", 0) == 0) {
        return mlir_text;
    }

    // Resolve the IREE target the same way mlir_invoke does so the dump
    // exactly mirrors what the runtime compiles.
    const std::string target =
        resolve_target(config_.target, inputs[0].tensor().device());

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
        iree_compile +
        // --iree-input-type=auto picks up StableHLO on both old and new
        // IREE (3.0–3.10 also accepted =stablehlo; 3.11+ replaced it).
        " --iree-input-type=auto" +
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

auto CompiledFunction::mlir_invoke(std::span<const Variable> /*inputs*/) -> Variable {
    throw std::runtime_error(
        "CompiledFunction::mlir_invoke: Tenzor was built without "
        "TENZOR_USE_MLIR_JIT=ON");
}

auto CompiledFunction::dump_mlir(const Variable& /*input*/) -> std::string {
    throw std::runtime_error(
        "CompiledFunction::dump_mlir: Tenzor was built without "
        "TENZOR_USE_MLIR_JIT=ON");
}

auto CompiledFunction::dump_mlir(std::span<const Variable> /*inputs*/) -> std::string {
    throw std::runtime_error(
        "CompiledFunction::dump_mlir: Tenzor was built without "
        "TENZOR_USE_MLIR_JIT=ON");
}

auto CompiledFunction::dump_stablehlo(const Variable& /*input*/) -> std::string {
    throw std::runtime_error(
        "CompiledFunction::dump_stablehlo: Tenzor was built without "
        "TENZOR_USE_MLIR_JIT=ON");
}

auto CompiledFunction::dump_stablehlo(std::span<const Variable> /*inputs*/) -> std::string {
    throw std::runtime_error(
        "CompiledFunction::dump_stablehlo: Tenzor was built without "
        "TENZOR_USE_MLIR_JIT=ON");
}

auto CompiledFunction::dump_iree(const Variable& /*input*/) -> std::string {
    throw std::runtime_error(
        "CompiledFunction::dump_iree: Tenzor was built without "
        "TENZOR_USE_MLIR_JIT=ON");
}

auto CompiledFunction::dump_iree(std::span<const Variable> /*inputs*/) -> std::string {
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

auto compile(CompiledFunction::FnTypeN fn, CompileConfig config) -> CompiledFunction {
    return CompiledFunction(std::move(fn), std::move(config));
}

} // namespace jit
} // namespace tenzor
