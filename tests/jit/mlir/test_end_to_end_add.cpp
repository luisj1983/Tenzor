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
#include "tenzor/jit/mlir/iree_runtime.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/tenzor.hpp"

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <limits>
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
    // cuda/vulkan gate on IREE device-init capability, NOT on the Tenzor
    // backend being loaded — identical to the rocm rationale below. IREE drives
    // the GPU directly via its own CUDA/Vulkan HAL driver, and run_add_on_target
    // keeps the eager input on CPU (Path C.2) when the Tenzor backend is absent,
    // so the JIT path still exercises the GPU. Gating on backend_present() here
    // wrongly SKIPPED cuda/vulkan JIT whenever the Tenzor backend .so wasn't
    // loaded (e.g. the MLIR-only build), leaving those paths untested.
    if (target == "cuda")         return ::tenzor::jit::mlir_jit::iree_can_initialize_default_device("cuda");
    // rocm gating: Path C.2 (see docs/superpowers/plans/
    // 2026-05-19-tz-jit-mlir-phase1a.md). We don't require the Tenzor
    // ROCm backend to be loaded — IREE drives the GPU directly via its
    // HIP HAL driver. As long as the IREE runtime can dlopen libamdhip64
    // and create a default device, the JIT path works end-to-end on real
    // ROCm hardware. The compile-time-discovered ROCm runtime library
    // directory is auto-prepended to LD_LIBRARY_PATH inside the probe.
    if (target == "rocm")
        return ::tenzor::jit::mlir_jit::iree_can_initialize_default_device("hip");
    if (target == "vulkan-spirv") return ::tenzor::jit::mlir_jit::iree_can_initialize_default_device("vulkan");
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
        SKIP_WITH_REASON(tenzor::testing::SkipReason::BackendUnavailable,
            "no hardware for target=" << target);
        return;
    }
    if (!iree_target_supported(target)) {
        SKIP_WITH_REASON(tenzor::testing::SkipReason::BackendUnavailable,
            "iree-compile dist lacks target=" << target);
        return;
    }

    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target  = target;

    auto compiled =
        ::tenzor::jit::CompiledFunction(add_self_fn(), cfg);

    // Path C.2: if the Tenzor backend for this target isn't loaded
    // (e.g. rocm on a host with corrupt /opt/rocm), allocate the
    // eager input on CPU. IREE copies the host buffer into its own
    // device-side buffer during marshaling regardless of where the
    // input lives, so the JIT path still runs on the GPU. The eager
    // x+x evaluation then runs on CPU — identical numerics for Add.
    const std::string be_name =
        target == "vulkan-spirv" ? "vulkan" : target;
    const auto dev = (target != "llvm-cpu" && backend_present(be_name))
                         ? device_for_target(target)
                         : ::tenzor::Device::cpu();
    auto x_tensor = ::tenzor::full({4}, 1.5F, ::tenzor::DType::Float32, dev);
    auto x = ::tenzor::Variable(x_tensor, /*requires_grad=*/false);

    const auto fn = add_self_fn();
    auto eager  = fn(x);
    ::tenzor::jit::mlir_jit::reset_cache_stats();
    auto jitted = compiled(x);
    // Prove the IREE compile+run path executed for this target rather than a
    // silent eager fallback (which would make the diff below eager-vs-eager and
    // pass vacuously). This is what makes the GPU-target assertions meaningful.
    ASSERT_GE(::tenzor::jit::mlir_jit::cache_stats().misses, 1u)
        << "target=" << target
        << " did NOT run through IREE (silent eager fallback)";

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

// A complex64 constant captured by the traced fn is baked into the graph as a
// `stablehlo.constant dense<(re,im)...>`. Before complex-constant support in
// the lowering, extract_scalar_value/emit_tensor_constant threw "unsupported
// DType", which (in non-strict mode) silently fell back to eager. Here we run
// STRICT so a lowering/compile failure throws instead of masking a regression,
// giving a real end-to-end check that complex constants compile through IREE.
TEST(EndToEndComplexConstant, MLIRBackendMatchesEager_CPU) {
    ensure_core_init();
    if (!iree_target_supported("llvm-cpu")) {
        SKIP_WITH_REASON(tenzor::testing::SkipReason::BackendUnavailable,
            "iree-compile dist lacks target=llvm-cpu");
        return;
    }

    ::tenzor::Tensor c({4}, ::tenzor::DType::Complex64, ::tenzor::Device::cpu());
    {
        auto* cd = c.data<std::complex<float>>();
        cd[0] = {1.0F, -2.0F};
        cd[1] = {0.5F, 3.0F};
        cd[2] = {-4.0F, 0.25F};
        cd[3] = {2.0F, 2.0F};
    }
    ::tenzor::Variable c_var(c, /*requires_grad=*/false);

    // Captured complex constant -> baked; input is complex too.
    ::tenzor::jit::CompiledFunction::FnType fn =
        [c_var](const ::tenzor::Variable& x) -> ::tenzor::Variable {
            return x + c_var;
        };

    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target  = "llvm-cpu";
    cfg.strict  = true;  // fail loudly instead of eager-fallback masking a bug
    auto compiled = ::tenzor::jit::CompiledFunction(fn, cfg);

    ::tenzor::Tensor x({4}, ::tenzor::DType::Complex64, ::tenzor::Device::cpu());
    {
        auto* xd = x.data<std::complex<float>>();
        xd[0] = {2.0F, 1.0F};
        xd[1] = {-1.0F, -1.0F};
        xd[2] = {3.0F, 3.0F};
        xd[3] = {0.0F, 5.0F};
    }
    ::tenzor::Variable xv(x, /*requires_grad=*/false);

    auto eager  = fn(xv).tensor().to(::tenzor::Device::cpu());
    auto jitted = compiled(xv).tensor().to(::tenzor::Device::cpu());

    ASSERT_EQ(jitted.numel(), eager.numel());
    ASSERT_EQ(jitted.dtype(), ::tenzor::DType::Complex64);
    const auto* ed = eager.data<std::complex<float>>();
    const auto* jd = jitted.data<std::complex<float>>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(ed[i].real(), jd[i].real(), 1e-5F) << "real i=" << i;
        EXPECT_NEAR(ed[i].imag(), jd[i].imag(), 1e-5F) << "imag i=" << i;
    }
}

// Non-finite float constants (+inf/-inf/NaN) baked into the graph must be
// emitted as width-correct hex bit patterns — MLIR's dense<> parser rejects the
// textual `inf`/`nan` that emit_float_literal produced before the fix. Runs
// STRICT so a bad emission fails iree-compile (throws) instead of silently
// falling back to eager. Verifies the compiled kernel preserves inf/-inf/NaN.
TEST(EndToEndNonFiniteConstant, MLIRBackendMatchesEager_CPU) {
    ensure_core_init();
    if (!iree_target_supported("llvm-cpu")) {
        SKIP_WITH_REASON(tenzor::testing::SkipReason::BackendUnavailable,
            "iree-compile dist lacks target=llvm-cpu");
        return;
    }

    ::tenzor::Tensor c({4}, ::tenzor::DType::Float32, ::tenzor::Device::cpu());
    {
        auto* cd = c.data<float>();
        cd[0] = std::numeric_limits<float>::infinity();
        cd[1] = -std::numeric_limits<float>::infinity();
        cd[2] = std::numeric_limits<float>::quiet_NaN();
        cd[3] = 2.5F;
    }
    ::tenzor::Variable c_var(c, /*requires_grad=*/false);

    ::tenzor::jit::CompiledFunction::FnType fn =
        [c_var](const ::tenzor::Variable& x) -> ::tenzor::Variable {
            return x + c_var;
        };

    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target  = "llvm-cpu";
    cfg.strict  = true;  // a textual inf/nan emission would fail compile loudly
    auto compiled = ::tenzor::jit::CompiledFunction(fn, cfg);

    auto x = ::tenzor::full({4}, 1.0F, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    auto jitted =
        compiled(::tenzor::Variable(x, false)).tensor().to(::tenzor::Device::cpu());

    ASSERT_EQ(jitted.numel(), 4);
    const float* p = jitted.data<float>();
    EXPECT_TRUE(std::isinf(p[0]) && p[0] > 0.0F) << "1+inf -> +inf, got " << p[0];
    EXPECT_TRUE(std::isinf(p[1]) && p[1] < 0.0F) << "1-inf -> -inf, got " << p[1];
    EXPECT_TRUE(std::isnan(p[2])) << "1+NaN -> NaN, got " << p[2];
    EXPECT_FLOAT_EQ(p[3], 3.5F);
}

TEST(EndToEndAdd, SecondInvocationHitsInProcessCache) {
    ensure_core_init();
    if (!target_hw_present("llvm-cpu")) {
        SKIP_WITH_REASON(tenzor::testing::SkipReason::BackendUnavailable,
            "no llvm-cpu");
        return;
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
