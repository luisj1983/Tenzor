// Phase 13 / Task B.2–B.6 — End-to-end Add via the MLIR backend.
//
// Pipeline under test:
//   `Variable x` → tracer.start_trace() → fn(x) [records Add]
//     → tracer.end_trace() → Graph
//     → GraphToMLIR::lower(g) → MLIR text
//     → compile_mlir(text, opts) → .vmfb
//     → IreeInvoker::invoke({x.tensor()}) → output tensor
//
// All four IREE targets (llvm-cpu, cuda, vulkan-spirv, rocm) get an
// instance; non-CPU ones skip when the hardware backend is unavailable.

#include "tenzor/autograd/variable.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/jit/compile.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/tenzor.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <set>
#include <string>

namespace {

auto ensure_core_init() -> void {
    static const bool inited = []() {
        ::tenzor::initialize();
        return true;
    }();
    (void)inited;
}

auto backend_present(const std::string& name) -> bool {
    auto* be = ::tenzor::backend_registry().get_backend(name);
    if (be == nullptr) return false;
    try {
        return be->device_count() > 0;
    } catch (...) {
        return false;
    }
}

auto target_hw_present(const std::string& target) -> bool {
    if (target == "llvm-cpu")     return true;
    if (target == "cuda")         return backend_present("cuda");
    if (target == "rocm")         return backend_present("rocm");
    if (target == "vulkan-spirv") return backend_present("vulkan");
    return false;
}

/// Probe the IREE compiler dist for whether `target` is in its registered
/// HAL target backends list. Required for tests that compile-then-run: even
/// if the Tenzor backend is loaded for a target (e.g. CUDA RTX 5070), the
/// local IREE dist may not have been built with that target's compiler
/// support. iree-compile gates target availability at the bytecode-emit
/// stage, so we have to ask it.
auto iree_target_supported(const std::string& target) -> bool {
    // llvm-cpu is the IREE default; always present in any dist.
    if (target == "llvm-cpu") return true;
    static const auto supported = []() -> std::set<std::string> {
        std::set<std::string> out;
        FILE* p = popen("iree-compile --iree-hal-list-target-backends 2>/dev/null", "r");
        if (!p) return out;
        char buf[256];
        while (fgets(buf, sizeof(buf), p)) {
            std::string line(buf);
            // The output prefixes each backend with whitespace.
            auto first = line.find_first_not_of(" \t");
            if (first == std::string::npos) continue;
            auto last = line.find_first_of(" \t\n\r", first);
            if (last == std::string::npos) last = line.size();
            out.insert(line.substr(first, last - first));
        }
        pclose(p);
        return out;
    }();
    return supported.count(target) > 0;
}

auto device_for_target(const std::string& target) -> ::tenzor::Device {
    if (target == "cuda")         return ::tenzor::Device::cuda(0);
    if (target == "rocm")         return ::tenzor::Device::rocm(0);
    if (target == "vulkan-spirv") return ::tenzor::Device::vulkan(0);
    return ::tenzor::Device::cpu();
}

/// `x + x` via the autograd Variable API.
auto add_self_fn() -> ::tenzor::jit::CompiledFunction::FnType {
    return [](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        return x + x;
    };
}

void run_add_on_target(const std::string& target) {
    ensure_core_init();
    if (!target_hw_present(target)) {
        GTEST_SKIP() << "no hardware for target=" << target;
    }
    if (!iree_target_supported(target)) {
        GTEST_SKIP() << "iree-compile dist lacks target=" << target;
    }

    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target  = target;

    auto compiled =
        ::tenzor::jit::CompiledFunction(add_self_fn(), cfg);

    const auto dev = device_for_target(target);
    auto x_tensor = ::tenzor::full({4}, 1.5F, ::tenzor::DType::Float32, dev);
    auto x = ::tenzor::Variable(x_tensor, /*requires_grad=*/false);

    const auto fn = add_self_fn();
    auto eager  = fn(x);
    auto jitted = compiled(x);

    ASSERT_EQ(jitted.tensor().numel(), eager.tensor().numel())
        << "numel mismatch on target=" << target;
    // iree-run-module always returns its output on CPU. Pull the eager
    // tensor back to CPU before the subtract so we don't trip the
    // dispatch's same-device-type guard for GPU targets.
    auto eager_cpu  = eager.tensor().to(::tenzor::Device::cpu());
    auto jitted_cpu = jitted.tensor().to(::tenzor::Device::cpu());
    auto diff_t =
        ::tenzor::max(::tenzor::abs(eager_cpu - jitted_cpu));
    auto diff = diff_t.item<float>();
    EXPECT_LT(diff, 1e-6F) << "target=" << target;
}

}  // namespace

TEST(EndToEndAdd, MLIRBackendMatchesEager_CPU) {
    run_add_on_target("llvm-cpu");
}

TEST(EndToEndAdd, MLIRCudaMatchesEager) {
    run_add_on_target("cuda");
}

TEST(EndToEndAdd, MLIRVulkanMatchesEager) {
    run_add_on_target("vulkan-spirv");
}

TEST(EndToEndAdd, MLIRRocmMatchesEager) {
    run_add_on_target("rocm");
}

TEST(EndToEndAdd, SecondInvocationHitsInProcessCache) {
    ensure_core_init();
    if (!target_hw_present("llvm-cpu")) {
        GTEST_SKIP() << "no llvm-cpu";
    }

    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target  = "llvm-cpu";
    auto compiled =
        ::tenzor::jit::CompiledFunction(add_self_fn(), cfg);

    auto x_tensor = ::tenzor::full({4}, 1.5F, ::tenzor::DType::Float32);
    auto x = ::tenzor::Variable(x_tensor, false);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto y1 = compiled(x);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto y2 = compiled(x);
    auto t2 = std::chrono::high_resolution_clock::now();

    const double ms1 =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double ms2 =
        std::chrono::duration<double, std::milli>(t2 - t1).count();
    // The IreeInvoker subprocess `iree-run-module` invoke takes ~10-15 ms
    // on a warm machine; the cache savings on second call are everything
    // ABOVE that: trace + Compiler::optimize + lower + iree-compile +
    // IreeInvoker::load. Use absolute savings rather than ratio to avoid
    // false positives when the subprocess fluctuates: first call should be
    // ≥ ms2 + 5 ms (i.e. we save at least 5ms of compile+load work).
    EXPECT_GE(ms1, ms2 + 5.0)
        << "no cache hit: first=" << ms1 << " ms second=" << ms2 << " ms "
        << "(expected first ≥ second + 5ms of saved compile/load work)";
}
