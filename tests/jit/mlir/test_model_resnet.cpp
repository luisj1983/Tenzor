// Phase 13 / Task E.4, E.5, E.6 — ResNet18 end-to-end through @tz.jit.
//
//   E.4 — eager forward sanity (shape + finite + non-zero).
//   E.5 — jit-compile via the MLIR backend on llvm-cpu and verify the
//         compiled forward matches the eager forward elementwise.
//   E.6 — same comparison on cuda / vulkan-spirv / rocm IREE targets
//         when the corresponding tenzor backend is present AND the
//         locally installed iree-compile has the matching HAL target.
//
// Uses `tenzor::models::resnet18()` (the project's built-in
// BasicBlock implementation) with the standard 1000-class head.
// Input shape `{1, 3, 32, 32}` keeps tests fast while still exercising
// every distinct op (conv2d / batchnorm2d / relu / maxpool / residual
// add / adaptive_avgpool / linear).

#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/jit/compile.hpp"
#include "tenzor/jit/mlir/iree_paths.hpp"
#include "tenzor/jit/mlir/iree_runtime.hpp"
#include "tenzor/models/resnet.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/tenzor.hpp"

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"

#include <cstdio>
#include <memory>
#include <set>
#include <string>
#include <vector>

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
    // cuda/vulkan gate on IREE device-init capability (like rocm below), NOT on
    // the Tenzor backend .so — IREE drives the GPU directly and run_jit_match
    // keeps eager on CPU (Path C.2) when the backend is absent. Gating on
    // backend_present() wrongly skipped cuda/vulkan JIT in the MLIR-only build.
    if (target == "cuda")
        return ::tenzor::jit::mlir_jit::iree_can_initialize_default_device("cuda");
    // rocm gating: Path C.2 (see docs/superpowers/plans/
    // 2026-05-19-tz-jit-mlir-phase1a.md). The Tenzor ROCm backend is
    // not required — IREE drives the GPU via its HIP HAL. We probe
    // for the IREE HIP driver instead.
    if (target == "rocm")
        return ::tenzor::jit::mlir_jit::iree_can_initialize_default_device("hip");
    if (target == "vulkan-spirv")
        return ::tenzor::jit::mlir_jit::iree_can_initialize_default_device("vulkan");
    return false;
}

// Whether the Tenzor backend for `target` is loaded (i.e. eager tensors
// can be allocated on that device). For Path C.2's rocm case this is
// usually false even when target_hw_present(rocm) is true: the IREE HIP
// HAL works without the Tenzor ROCm backend. Eager-only paths then have
// to stay on CPU.
auto tenzor_backend_for_target_loaded(const std::string& target) -> bool {
    if (target == "llvm-cpu")     return backend_present("cpu");
    if (target == "cuda")         return backend_present("cuda");
    if (target == "rocm")         return backend_present("rocm");
    if (target == "vulkan-spirv") return backend_present("vulkan");
    return false;
}

auto device_for_target(const std::string& target) -> ::tenzor::Device {
    if (target == "cuda")         return ::tenzor::Device::cuda(0);
    if (target == "rocm")         return ::tenzor::Device::rocm(0);
    if (target == "vulkan-spirv") return ::tenzor::Device::vulkan(0);
    return ::tenzor::Device::cpu();
}

/// Probe whether the local iree-compile has a given HAL target backend
/// registered. Skips the test cleanly when the iree-dist isn't built
/// with support for cuda / rocm / etc.
auto iree_target_supported(const std::string& target) -> bool {
    if (target == "llvm-cpu") return true;
    // Defer to the shared discovery + probe so test SKIP/RUN decisions
    // stay consistent with what compile_mlir() does at runtime.
    return ::tenzor::jit::mlir_jit::iree_compile_supports(target);
}

/// Build a ResNet18 in eval mode so BatchNorm uses running stats and the
/// forward is deterministic. We materialise the running stats by calling
/// `eval()` and resetting BN running mean/var to a sane (zero/one)
/// distribution; the constructed model already does this on
/// initialisation.
auto build_resnet18() -> std::shared_ptr<::tenzor::models::ResNet> {
    auto m = ::tenzor::models::resnet18(/*num_classes=*/1000,
                                        /*pretrained=*/false);
    m->eval();
    return m;
}

void run_jit_match(const std::string& target) {
    ensure_core_init();
    if (!target_hw_present(target)) {
        SKIP_WITH_REASON(tenzor::testing::SkipReason::BackendUnavailable,
            "no hardware for target=" << target);
        return;
    }
    if (!iree_target_supported(target)) {
        SKIP_WITH_REASON(tenzor::testing::SkipReason::BackendUnavailable,
            "iree-compile does not have HAL target backend '"
            << target << "' registered (rebuild iree-dist with -DIREE_HAL_DRIVER_"
            << target << "=ON)");
        return;
    }

    auto m = build_resnet18();
    // Path C.2: when the Tenzor backend for this target isn't loaded
    // (rocm on a host with a corrupt /opt/rocm — see Phase 1A plan)
    // keep eager on CPU. IREE marshals host buffers into its own
    // device-side allocations during invoke so the JIT path still runs
    // on the GPU.
    const bool tenzor_be = tenzor_backend_for_target_loaded(target);
    const auto dev = tenzor_be ? device_for_target(target)
                               : ::tenzor::Device::cpu();
    if (dev.type != ::tenzor::Device::Type::CPU) {
        m->to(dev);
    }
    auto x_t = ::tenzor::randn({1, 3, 32, 32}, ::tenzor::DType::Float32,
                               dev);
    ::tenzor::Variable x(x_t, /*requires_grad=*/false);

    auto eager = m->forward(x);

    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target  = target;
    auto fn = [&m](const ::tenzor::Variable& in) -> ::tenzor::Variable {
        return m->forward(in);
    };
    auto compiled = ::tenzor::jit::CompiledFunction(fn, cfg);
    ::tenzor::jit::mlir_jit::reset_cache_stats();
    auto jit_out  = compiled(x);
    // Prove the IREE compile+run path actually executed. mlir_invoke() falls
    // back to eager (non-strict) if any op fails to lower, which would make the
    // diff below eager-vs-eager == 0 and pass vacuously. A recorded compile miss
    // means the compiled StableHLO ran, so the comparison has teeth.
    ASSERT_GE(::tenzor::jit::mlir_jit::cache_stats().misses, 1u)
        << "ResNet18 did NOT run through IREE (silent eager fallback would make "
           "the parity check vacuous); target=" << target;

    const auto e_shape = eager.tensor().shape();
    const auto j_shape = jit_out.tensor().shape();
    ASSERT_EQ(std::vector<int64_t>(j_shape.begin(), j_shape.end()),
              std::vector<int64_t>(e_shape.begin(), e_shape.end()))
        << "shape mismatch on target=" << target;

    const auto e_f64 = eager.tensor().to(::tenzor::Device::cpu())
                           .to(::tenzor::DType::Float64);
    const auto j_f64 = jit_out.tensor().to(::tenzor::Device::cpu())
                           .to(::tenzor::DType::Float64);
    const auto diff  = ::tenzor::max(::tenzor::abs(e_f64 - j_f64))
                           .template item<double>();
    EXPECT_LT(diff, 1e-3)
        << "ResNet18 @tz.jit diverges from eager (target=" << target
        << ", max-abs-diff=" << diff << ")";
}

}  // namespace

TEST(ResNet18, EagerForwardShapeAndFiniteness) {
    ensure_core_init();
    auto m = build_resnet18();
    auto x = ::tenzor::Variable(
        ::tenzor::randn({1, 3, 32, 32}), /*requires_grad=*/false);
    auto logits = m->forward(x);
    const auto logits_shape = logits.tensor().shape();
    const std::vector<int64_t> got_shape(logits_shape.begin(),
                                         logits_shape.end());
    ASSERT_EQ(got_shape, (std::vector<int64_t>{1, 1000}));
    auto f = logits.tensor().to(::tenzor::DType::Float64);
    const auto maxabs =
        ::tenzor::max(::tenzor::abs(f)).template item<double>();
    EXPECT_LT(maxabs, 1e10)  << "logits not finite";
    EXPECT_GT(maxabs, 1e-10) << "logits collapsed to zero";
}

TEST(ResNet18, JitMatchesEager_LlvmCpu) {
    run_jit_match("llvm-cpu");
}

TEST(ResNet18, JitMatchesEager_Cuda) {
    run_jit_match("cuda");
}

TEST(ResNet18, JitMatchesEager_Vulkan) {
    run_jit_match("vulkan-spirv");
}

TEST(ResNet18, JitMatchesEager_Rocm) {
    run_jit_match("rocm");
}
