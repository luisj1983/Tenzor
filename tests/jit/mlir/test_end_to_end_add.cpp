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
#include "tenzor/jit/mlir/iree_compile.hpp"
#include "tenzor/jit/mlir/iree_paths.hpp"
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
    // Delegate to the same discovery + probe used by the in-process
    // compile_mlir() subprocess fallback so test SKIP / RUN decisions
    // stay consistent with what the runtime actually does.
    return ::tenzor::jit::mlir_jit::iree_compile_supports(target);
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

    // Counter-based signal: every CompiledFunction call goes through the
    // shape-keyed in-process cache; the second invoke with the same shape
    // must bump hits without bumping misses. Wall-time on subprocess
    // iree-run-module is too noisy (range 30-90 ms on a warm laptop) to be
    // a reliable cache indicator.
    ::tenzor::jit::mlir_jit::reset_cache_stats();

    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target  = "llvm-cpu";
    auto compiled =
        ::tenzor::jit::CompiledFunction(add_self_fn(), cfg);

    auto x_tensor = ::tenzor::full({4}, 1.5F, ::tenzor::DType::Float32);
    auto x = ::tenzor::Variable(x_tensor, false);

    auto y1 = compiled(x);
    const auto s1 = ::tenzor::jit::mlir_jit::cache_stats();
    auto y2 = compiled(x);
    const auto s2 = ::tenzor::jit::mlir_jit::cache_stats();

    EXPECT_EQ(s1.hits, 0u)   << "first invocation should have been a miss";
    EXPECT_GE(s1.misses, 1u) << "first invocation should record a miss";
    EXPECT_GE(s2.hits, s1.hits + 1u)
        << "second invocation should hit the in-process compile cache "
           "(hits went " << s1.hits << " -> " << s2.hits << ")";
    EXPECT_EQ(s2.misses, s1.misses)
        << "second invocation should not produce a new miss "
           "(misses went " << s1.misses << " -> " << s2.misses << ")";
}
