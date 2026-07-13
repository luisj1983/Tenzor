// MLIR/IREE JIT backend-parity fixes — regression tests.
//
// Covers the fixes for:
//   C1 — devices with no native IREE target (OneAPI/MPS) must degrade to eager
//        instead of throwing uncaught. No OneAPI/MPS runtime exists on this
//        host, so the device-agnostic realization is exercised: an explicit
//        target with no valid IREE mapping takes the same eager path.
//   C2 — the mlir path wraps lower/compile/resolve/run in the SAME strict-aware
//        eager fallback the nvrtc path uses. An op that fails to lower
//        (OpType::Tanh has no StableHLO lowering) degrades to eager in
//        non-strict mode and rethrows under strict mode, on cpu/cuda/rocm.
//   H6 — the documented target alias "vulkan" is normalized to the
//        IREE-required "vulkan-spirv" at the compile boundary; a bogus target
//        yields a clear error.
//   M1 — the ROCm gfx arch is derived from the actual device, and the derived
//        arch compiles.
//   M3 — the HAL device ordinal is threaded into the device URI.

#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/jit/compile.hpp"
#include "tenzor/jit/graph.hpp"
#include "tenzor/jit/mlir/iree_compile.hpp"
#include "tenzor/jit/mlir/iree_paths.hpp"
#include "tenzor/jit/mlir/iree_runtime.hpp"
#include "tenzor/jit/mlir/lowering.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/tenzor.hpp"
#include "../../backend_parity/parity_test_utils.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace tj = ::tenzor::jit::mlir_jit;
namespace fs = std::filesystem;

auto ensure_core_init() -> void {
    static const bool inited = []() {
        ::tenzor::initialize();
        return true;
    }();
    (void)inited;
}

auto make_tmp_dir() -> fs::path {
    const auto now = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 rng(static_cast<uint64_t>(now));
    fs::path dir = fs::temp_directory_path() /
                   ("tenzor_jit_fallback_test_" + std::to_string(rng()));
    fs::create_directories(dir);
    return dir;
}

/// Whether the IREE runtime + compiler can drive `target` end-to-end here.
/// CPU is always available; GPU targets require BOTH an IREE HAL device and
/// iree-compile support for the target. This gates on IREE's own HAL (like
/// the ROCm path in test_end_to_end_add) rather than on the Tenzor GPU
/// backend, so CUDA is exercised even though this build has no CUDA backend.
auto target_runnable(const std::string& target) -> bool {
    if (target == "llvm-cpu") return true;
    if (!tj::iree_compile_supports(target)) return false;
    if (target == "cuda") return tj::iree_can_initialize_default_device("cuda");
    if (target == "rocm") return tj::iree_can_initialize_default_device("hip");
    if (target == "vulkan-spirv")
        return tj::iree_can_initialize_default_device("vulkan");
    return false;
}

constexpr const char* kTrivialModule = R"MLIR(module {
  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }
}
)MLIR";

// ── C2: an unlowered op (Tanh) degrades to eager (non-strict) / rethrows
//    (strict) on each available target. ──────────────────────────────────────

// JIT-R027: the C2 eager-fallback path dispatches on the INPUT TENSOR's
// device (compile.cpp), not on CompileConfig::target -- so a CPU-resident
// input exercises only CPU's own eager tanh kernel regardless of which
// target string this function is called with, giving _Cuda/_Rocm/_Vulkan
// zero coverage of their OWN backend's eager-dispatch numerics (exactly the
// scenario most likely to hide a real per-backend divergence, e.g. an
// edge-value tanh difference). Map target -> the actual device it names so
// each variant's eager fallback genuinely runs on that backend.
auto device_for_target(const std::string& target) -> ::tenzor::Device {
    if (target == "cuda") return ::tenzor::Device::cuda(0);
    if (target == "rocm") return ::tenzor::Device::rocm(0);
    if (target == "vulkan-spirv" || target == "vulkan") return ::tenzor::Device::vulkan(0);
    return ::tenzor::Device::cpu();
}

void run_c2_unlowered_op(const std::string& target) {
    ensure_core_init();
    // R2-03: escalate to FAIL() under TENZOR_REQUIRE_MULTI_BACKEND=1 instead
    // of a bare skip with no signal — this project's convention for an
    // environment that's supposed to have GPU hardware but doesn't.
    if (!target_runnable(target)) {
        if (::tenzor::testing::golden::require_multi_backend() && target != "llvm-cpu") {
            FAIL() << "target " << target << " required by "
                      "TENZOR_REQUIRE_MULTI_BACKEND but not runnable here";
        }
        GTEST_SKIP() << "target not runnable here: " << target;
    }

    const auto dev = device_for_target(target);
    // A GPU device named by `target` may have no ACTUAL Tenzor backend
    // registered on this machine (target_runnable only checks IREE's own
    // HAL/compiler support, deliberately independent of the Tenzor GPU
    // backend -- see its own comment) -- constructing a tensor on an
    // unregistered backend would throw before ever reaching the code this
    // test means to exercise. Skip cleanly rather than fail in that case.
    if (dev.type != ::tenzor::Device::Type::CPU) {
        const auto backends = ::tenzor::testing::get_available_backends();
        const bool have_backend = std::any_of(
            backends.begin(), backends.end(),
            [&](const ::tenzor::Device& d) { return d.type == dev.type; });
        if (!have_backend) {
            if (::tenzor::testing::golden::require_multi_backend()) {
                FAIL() << "Tenzor backend for target " << target
                       << " required by TENZOR_REQUIRE_MULTI_BACKEND but not "
                          "registered";
            }
            GTEST_SKIP() << "no Tenzor backend registered for target: " << target;
        }
    }

    auto tanh_fn = [](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        return ::tenzor::tanh(x);  // OpType::Tanh has no StableHLO lowering
    };

    auto x_tensor = ::tenzor::full({4}, 0.3F, ::tenzor::DType::Float32, dev);
    auto x = ::tenzor::Variable(x_tensor, /*requires_grad=*/false);
    const auto eager = tanh_fn(x).tensor();

    // Non-strict: must NOT throw and must equal the eager result (the
    // fallback runs fn_ eagerly on the input's backend).
    {
        ::tenzor::jit::CompileConfig cfg;
        cfg.backend       = "mlir";
        cfg.target        = target;
        cfg.strict        = false;
        cfg.enable_fusion = false;  // keep the Tanh node intact
        ::tenzor::jit::CompiledFunction compiled(
            ::tenzor::jit::CompiledFunction::FnType(tanh_fn), cfg);

        ::tenzor::Variable out;
        ASSERT_NO_THROW({ out = compiled(x); })
            << "non-strict mlir path must degrade to eager, not throw "
               "(target=" << target << ")";
        const auto got = out.tensor().to(::tenzor::Device::cpu());
        const auto ref = eager.to(::tenzor::Device::cpu());
        const auto diff = ::tenzor::max(::tenzor::abs(ref - got)).item<float>();
        EXPECT_LT(diff, 1e-6F) << "eager-fallback result mismatch (target="
                               << target << ")";
    }

    // Strict: the same lowering gap must rethrow.
    {
        ::tenzor::jit::CompileConfig cfg;
        cfg.backend       = "mlir";
        cfg.target        = target;
        cfg.strict        = true;
        cfg.enable_fusion = false;
        ::tenzor::jit::CompiledFunction compiled(
            ::tenzor::jit::CompiledFunction::FnType(tanh_fn), cfg);
        EXPECT_THROW({ (void)compiled(x); }, std::exception)
            << "strict mode must rethrow the lowering failure (target="
            << target << ")";
    }
}

}  // namespace

TEST(JitEagerFallback, UnloweredOpDegradesToEager_Cpu) {
    run_c2_unlowered_op("llvm-cpu");
}

// F021: a JIT eager-fallback must run the user function EXACTLY once — the trace
// runs it, and the fallback must REUSE that captured result (out-param capture)
// rather than re-running it. The documented double-exec regression
// (mlir_invoke/grad_invoke replaying fn_ twice) produced CORRECT numbers but ran
// fn_ twice, so only an invocation counter catches it. Runs on llvm-cpu where the
// unlowered Tanh op forces the C2 eager-fallback path.
TEST(JitEagerFallback, TracedFnRunsExactlyOnceOnEagerFallback) {
    ensure_core_init();
    auto counter = std::make_shared<int>(0);
    auto counting_fn =
        [counter](const ::tenzor::Variable& x) -> ::tenzor::Variable {
            ++(*counter);
            return ::tenzor::tanh(x);  // no StableHLO lowering -> eager fallback
        };
    auto x = ::tenzor::Variable(
        ::tenzor::full({4}, 0.3F, ::tenzor::DType::Float32), false);

    ::tenzor::jit::CompileConfig cfg;
    cfg.backend       = "mlir";
    cfg.target        = "llvm-cpu";
    cfg.strict        = false;
    cfg.enable_fusion = false;  // keep the Tanh node intact so lowering fails
    ::tenzor::jit::CompiledFunction compiled(
        ::tenzor::jit::CompiledFunction::FnType(counting_fn), cfg);

    (void)compiled(x);
    // Trace-once + reuse == 1 total across construction + first invoke. A
    // double-exec regression would make this 2.
    EXPECT_EQ(*counter, 1)
        << "traced fn ran " << *counter << " times on the eager-fallback path; "
           "it must run EXACTLY once (double-exec regression)";
}
TEST(JitEagerFallback, UnloweredOpDegradesToEager_Cuda) {
    run_c2_unlowered_op("cuda");
}
TEST(JitEagerFallback, UnloweredOpDegradesToEager_Rocm) {
    run_c2_unlowered_op("rocm");
}
// JIT-R028: no _Vulkan variant existed at all, despite target_runnable()
// explicitly handling "vulkan-spirv" and Vulkan being a fully exercised IREE
// HAL target elsewhere in this same file (JitVulkanAlias). Combined with
// JIT-R027's fix above, this closes Vulkan's eager-fallback numeric coverage
// gap alongside CUDA/ROCm's.
TEST(JitEagerFallback, UnloweredOpDegradesToEager_Vulkan) {
    run_c2_unlowered_op("vulkan-spirv");
}

// ── C1: a target with no valid IREE mapping (the OneAPI/MPS class of device,
//    realized here via an explicit unmappable target) degrades to eager in
//    non-strict mode and rethrows under strict mode — never an uncaught crash.
//    This is the same C2 safety net that catches resolve_target's throw for a
//    OneAPI tensor. ─────────────────────────────────────────────────────────

TEST(JitEagerFallback, UnmappableTargetDegradesToEager) {
    ensure_core_init();
    auto add_fn = [](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        return x + x;
    };
    auto x_tensor = ::tenzor::full({4}, 1.5F, ::tenzor::DType::Float32);
    auto x = ::tenzor::Variable(x_tensor, /*requires_grad=*/false);
    const auto eager = add_fn(x).tensor();

    // Non-strict: no throw, eager result.
    {
        ::tenzor::jit::CompileConfig cfg;
        cfg.backend = "mlir";
        cfg.target  = "no-such-iree-target";  // stands in for OneAPI/MPS
        cfg.strict  = false;
        ::tenzor::jit::CompiledFunction compiled(
            ::tenzor::jit::CompiledFunction::FnType(add_fn), cfg);
        ::tenzor::Variable out;
        ASSERT_NO_THROW({ out = compiled(x); });
        const auto diff =
            ::tenzor::max(::tenzor::abs(eager - out.tensor())).item<float>();
        EXPECT_LT(diff, 1e-6F);
    }
    // Strict: rethrows.
    {
        ::tenzor::jit::CompileConfig cfg;
        cfg.backend = "mlir";
        cfg.target  = "no-such-iree-target";
        cfg.strict  = true;
        ::tenzor::jit::CompiledFunction compiled(
            ::tenzor::jit::CompiledFunction::FnType(add_fn), cfg);
        EXPECT_THROW({ (void)compiled(x); }, std::exception);
    }
}

// JIT-R022: the test above stands in for the OneAPI/MPS scenario with a
// bogus explicit target, which hits the generic C2 safety net (a lowering/
// resolve_target failure caught around the whole compile section) — NOT the
// dedicated C1 branch (compile.cpp's mlir_invoke, "C1: a device with no IREE
// HAL target") that actually fires for target="auto" on a real OneAPI/MPS
// device, including that branch's distinct message text and the
// per-CompiledFunction warned_no_accel_ once-flag. This drives the real C1
// branch directly on genuine OneAPI hardware instead of a stand-in.
TEST(JitEagerFallback, UnmappableTargetDegradesToEager_RealOneapiC1) {
    ensure_core_init();
    REQUIRE_BACKEND_OR_SKIP("oneapi");
    auto add_fn = [](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        return x + x;
    };
    auto x_tensor = ::tenzor::full({4}, 1.5F, ::tenzor::DType::Float32,
                                    ::tenzor::Device::oneapi(0));
    auto x = ::tenzor::Variable(x_tensor, /*requires_grad=*/false);
    const auto eager = add_fn(x).tensor();

    // Non-strict (target="auto", the @tz.jit default): C1 fires, warns once,
    // and executes eagerly on the SAME (OneAPI) backend — never CPU.
    {
        ::tenzor::jit::CompileConfig cfg;
        cfg.backend = "mlir";
        cfg.target  = "auto";
        cfg.strict  = false;
        ::tenzor::jit::CompiledFunction compiled(
            ::tenzor::jit::CompiledFunction::FnType(add_fn), cfg);
        ::tenzor::Variable out;
        ASSERT_NO_THROW({ out = compiled(x); });
        EXPECT_EQ(out.tensor().device().type, ::tenzor::Device::Type::OneAPI)
            << "C1 eager fallback must stay on the input's own backend, "
               "never silently move to CPU";
        const auto diff = ::tenzor::max(::tenzor::abs(
            eager.to(::tenzor::Device::cpu()) -
            out.tensor().to(::tenzor::Device::cpu()))).item<float>();
        EXPECT_LT(diff, 1e-6F);
    }
    // Strict: C1 rethrows with its own distinct message (not the generic C2
    // "iree-compile subprocess" text).
    {
        ::tenzor::jit::CompileConfig cfg;
        cfg.backend = "mlir";
        cfg.target  = "auto";
        cfg.strict  = true;
        ::tenzor::jit::CompiledFunction compiled(
            ::tenzor::jit::CompiledFunction::FnType(add_fn), cfg);
        EXPECT_THROW(
            {
                try {
                    (void)compiled(x);
                } catch (const std::exception& e) {
                    EXPECT_NE(std::string(e.what()).find("no IREE target backend"),
                              std::string::npos)
                        << "expected the C1-specific message, got: " << e.what();
                    throw;
                }
            },
            std::exception);
    }
}

// ── H6: "vulkan" is accepted and normalized to "vulkan-spirv" at the compile
//    boundary; a bogus target gives a clear, listing error. ─────────────────

TEST(JitVulkanAlias, CompileMlirNormalizesVulkanAlias) {
    ensure_core_init();
    if (!tj::iree_compile_supports("vulkan-spirv")) {
        if (::tenzor::testing::golden::require_multi_backend()) {
            FAIL() << "vulkan-spirv iree-compile support required by "
                      "TENZOR_REQUIRE_MULTI_BACKEND but unavailable";
        }
        GTEST_SKIP() << "iree-compile dist lacks vulkan-spirv";
    }
    const fs::path tmp = make_tmp_dir();
    tj::CompileOptions opts;
    opts.target    = "vulkan";  // documented alias — must be accepted
    opts.cache_dir = tmp;

    tj::CompiledArtifact artifact;
    ASSERT_NO_THROW({ artifact = tj::compile_mlir(kTrivialModule, opts); })
        << "target=\"vulkan\" must compile (normalized to vulkan-spirv)";
    EXPECT_TRUE(fs::exists(artifact.vmfb_path));
    EXPECT_GT(fs::file_size(artifact.vmfb_path), 0U);
    EXPECT_EQ(artifact.target, "vulkan-spirv")
        << "artifact target must report the canonical name";
    fs::remove_all(tmp);
}

TEST(JitVulkanAlias, CompileMlirRejectsUnknownTargetClearly) {
    ensure_core_init();
    const fs::path tmp = make_tmp_dir();
    tj::CompileOptions opts;
    opts.target    = "totally-bogus-target";
    opts.cache_dir = tmp;
    try {
        (void)tj::compile_mlir(kTrivialModule, opts);
        FAIL() << "expected a JitCompileError for an unknown target";
    } catch (const tj::JitCompileError& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("totally-bogus-target"), std::string::npos);
        // The error should list the valid canonical names.
        EXPECT_NE(msg.find("vulkan-spirv"), std::string::npos) << msg;
    }
    fs::remove_all(tmp);
}

// ── M3: HAL device URIs carry the requested ordinal; CPU has none. ──────────

TEST(JitDeviceOrdinal, HalDeviceUriHonorsOrdinal) {
    EXPECT_EQ(tj::hal_device_uri("cuda", 0), "cuda://0");
    EXPECT_EQ(tj::hal_device_uri("cuda", 1), "cuda://1");
    EXPECT_EQ(tj::hal_device_uri("hip", 2), "hip://2");
    EXPECT_EQ(tj::hal_device_uri("vulkan", 3), "vulkan://3");
    // CPU HAL drivers have no ordinal.
    EXPECT_EQ(tj::hal_device_uri("local-task", 5), "local-task");
    EXPECT_EQ(tj::hal_device_uri("local-sync", 2), "local-sync");
    // Defensive: negative ordinal clamps to 0.
    EXPECT_EQ(tj::hal_device_uri("hip", -1), "hip://0");
}

// R1-02/JIT-055 regression: resolve_hal_ordinal must carry the input
// tensor's device index over to the HAL ordinal ONLY when the device's
// backend family matches the resolved compile target -- an explicit
// cross-family target override (e.g. target="rocm" while the tensor lives
// on cuda:1) must default to ordinal 0, not silently carry cuda:1's index
// over to the ROCm HAL device (which would target the wrong physical GPU
// with the arch correctly derived for ROCm device 0).
TEST(JitDeviceOrdinal, ResolveHalOrdinalIsFamilyAware) {
    namespace tjd = ::tenzor::jit::mlir_detail;

    // Same-family: the device's own ordinal is honored.
    EXPECT_EQ(tjd::resolve_hal_ordinal("cuda", ::tenzor::Device::cuda(0)), 0);
    EXPECT_EQ(tjd::resolve_hal_ordinal("cuda", ::tenzor::Device::cuda(1)), 1);
    EXPECT_EQ(tjd::resolve_hal_ordinal("rocm", ::tenzor::Device::rocm(2)), 2);
    EXPECT_EQ(tjd::resolve_hal_ordinal("vulkan-spirv", ::tenzor::Device::vulkan(3)), 3);
    EXPECT_EQ(tjd::resolve_hal_ordinal("llvm-cpu", ::tenzor::Device::cpu()), 0);

    // Cross-family: the mismatched device's ordinal must NOT be carried over.
    EXPECT_EQ(tjd::resolve_hal_ordinal("rocm", ::tenzor::Device::cuda(1)), 0);
    EXPECT_EQ(tjd::resolve_hal_ordinal("cuda", ::tenzor::Device::rocm(1)), 0);
    EXPECT_EQ(tjd::resolve_hal_ordinal("vulkan-spirv", ::tenzor::Device::cuda(1)), 0);
    EXPECT_EQ(tjd::resolve_hal_ordinal("cuda", ::tenzor::Device::vulkan(1)), 0);
    EXPECT_EQ(tjd::resolve_hal_ordinal("llvm-cpu", ::tenzor::Device::cuda(1)), 0);

    // Defensive: negative ordinal on a matching family clamps to 0.
    EXPECT_EQ(tjd::resolve_hal_ordinal("cuda", ::tenzor::Device(::tenzor::Device::Type::CUDA, -1)), 0);
}

// JIT-R131 regression: AMD/Intel Vulkan --iree-vulkan-target selection used to
// be a single hardcoded value per vendor ("rdna3" for any AMD GPU, "arc" for
// any Intel GPU) with no GPU-generation check, unlike the NVIDIA path (fixed
// as JIT-R108). Verified against the actual installed IREE toolchain: "arc"
// is not merely stale, it CRASHES iree-compile outright (SIGABRT inside
// ExecutableTargetAttr::getBackend(), exit code 245) -- every Intel Vulkan
// JIT compile would crash the compiler, not just silently target an old
// profile. Tests the extracted pure classifier directly (see compile.hpp's
// mlir_detail::classify_vulkan_target doc comment) with synthetic vendor/
// name strings, so every generation is covered without needing that
// generation's real hardware on hand.
TEST(JitVulkanArch, ClassifyVulkanTargetCoversAllKnownGenerations) {
    namespace tjd = ::tenzor::jit::mlir_detail;

    // NVIDIA (JIT-R108): generation-specific, falls back to empty (safe
    // default) for anything unrecognized.
    EXPECT_EQ(tjd::classify_vulkan_target("nvidia", "nvidia geforce rtx 5070 laptop gpu"), "ada");
    EXPECT_EQ(tjd::classify_vulkan_target("nvidia", "nvidia rtx pro 6000"), "ada");
    EXPECT_EQ(tjd::classify_vulkan_target("nvidia", "nvidia geforce rtx 4090"), "ada");
    EXPECT_EQ(tjd::classify_vulkan_target("nvidia", "nvidia geforce rtx 3090"), "ampere");
    EXPECT_EQ(tjd::classify_vulkan_target("nvidia", "nvidia a100"), "ampere");
    EXPECT_EQ(tjd::classify_vulkan_target("nvidia", "nvidia geforce gtx 1080"), "");

    // AMD (JIT-R131): generation-specific by RX-series number; falls back to
    // "rdna3" (not empty) since that default is empirically verified working
    // on real RDNA3.5 (gfx1150) integrated-GPU hardware, whose marketing
    // name carries no RX number to match against.
    EXPECT_EQ(tjd::classify_vulkan_target("amd", "amd radeon rx 9070 xt"), "rdna4");
    EXPECT_EQ(tjd::classify_vulkan_target("amd", "amd radeon rx 7900 xtx"), "rdna3");
    EXPECT_EQ(tjd::classify_vulkan_target("amd", "amd radeon rx 6800 xt"), "rdna2");
    EXPECT_EQ(tjd::classify_vulkan_target("amd", "amd radeon rx 5700 xt"), "rdna1");
    EXPECT_EQ(tjd::classify_vulkan_target("amd", "amd radeon 890m graphics"), "rdna3")
        << "an APU with no RX-series marketing number must fall back to the "
           "verified-working default, not an empty/unresolved target";
    EXPECT_EQ(tjd::classify_vulkan_target("radeon", "amd radeon 890m graphics"), "rdna3")
        << "vendor string may report 'radeon' rather than 'amd'";

    // JIT-R112 regression: a bare "rx 5" substring used to match BOTH the
    // intended RDNA1 "RX 5000-series" (4-digit) AND the unrelated pre-RDNA
    // Polaris/GCN4 "RX 500-series" (3-digit), silently misrouting Polaris
    // cards to an RDNA1 target they don't support. Polaris/GCN4 must now
    // fall through to the safe empty default, like NVIDIA/Intel do for
    // anything unrecognized -- NOT the iGPU "rdna3" catch-all, since an
    // explicit-but-unmatched RX number is a different case from "no RX
    // number present at all".
    EXPECT_EQ(tjd::classify_vulkan_target("amd", "amd radeon rx 580"), "")
        << "Polaris/GCN4 RX 500-series (3-digit) must not collide with "
           "RDNA1's RX 5000-series (4-digit) via a bare \"rx 5\" substring";
    EXPECT_EQ(tjd::classify_vulkan_target("amd", "amd radeon rx 570"), "");
    EXPECT_EQ(tjd::classify_vulkan_target("amd", "amd radeon rx 590"), "");

    // JIT-R112 regression: CDNA/Instinct datacenter GPUs are an entirely
    // different architecture family from consumer RDNA and must never
    // silently receive a consumer RDNA target guess (the old code's
    // blanket "return rdna3" catch-all covered these too, since they carry
    // no RX-series number either -- same shape as the iGPU case, wrong
    // architecture family).
    EXPECT_EQ(tjd::classify_vulkan_target("amd", "amd instinct mi300x"), "")
        << "CDNA/Instinct must not receive the RDNA-targeted iGPU fallback";
    EXPECT_EQ(tjd::classify_vulkan_target("amd", "amd instinct mi210"), "");
    EXPECT_EQ(tjd::classify_vulkan_target("amd", "amd instinct mi100"), "");

    // Intel (JIT-R131): no known-working IREE Vulkan target profile exists
    // for Intel at all in this toolchain -- must return empty (safe
    // default), NEVER the old "arc" value, which crashes iree-compile.
    EXPECT_EQ(tjd::classify_vulkan_target("intel", "intel arc a770 graphics"), "");
    EXPECT_EQ(tjd::classify_vulkan_target("intel", "intel iris xe graphics"), "");

    // Unknown vendor: safe default.
    EXPECT_EQ(tjd::classify_vulkan_target("qualcomm", "adreno 730"), "");
}

// ── M1: the ROCm arch is read from the actual device and the derived arch
//    compiles. ──────────────────────────────────────────────────────────────

TEST(JitRocmArch, DetectedArchMatchesDeviceAndCompiles) {
    ensure_core_init();
    if (!tj::iree_can_initialize_default_device("hip")) {
        if (::tenzor::testing::golden::require_multi_backend()) {
            FAIL() << "ROCm/HIP device required by TENZOR_REQUIRE_MULTI_BACKEND "
                      "but unavailable";
        }
        GTEST_SKIP() << "no ROCm/HIP device";
    }
    const std::string arch = tj::detect_rocm_gfx_arch(0);
    ASSERT_FALSE(arch.empty())
        << "arch must be derived from the physical ROCm device, not a "
           "build constant";
    EXPECT_EQ(arch.rfind("gfx", 0), 0U)
        << "expected a gfx ISA name, got: " << arch;

    if (!tj::iree_compile_supports("rocm")) {
        if (::tenzor::testing::golden::require_multi_backend()) {
            FAIL() << "rocm iree-compile support required by "
                      "TENZOR_REQUIRE_MULTI_BACKEND but unavailable (arch "
                      "derivation still verified above)";
        }
        GTEST_SKIP() << "iree-compile dist lacks rocm (arch derivation still "
                        "verified above)";
    }
    // The device-derived arch must actually compile.
    const fs::path tmp = make_tmp_dir();
    tj::CompileOptions opts;
    opts.target    = "rocm";
    opts.rocm_arch = arch;
    opts.cache_dir = tmp;
    tj::CompiledArtifact artifact;
    ASSERT_NO_THROW({ artifact = tj::compile_mlir(kTrivialModule, opts); })
        << "device-derived arch " << arch << " must compile";
    EXPECT_TRUE(fs::exists(artifact.vmfb_path));
    EXPECT_GT(fs::file_size(artifact.vmfb_path), 0U);
    fs::remove_all(tmp);
}

// ── R1-08: HIP_VISIBLE_DEVICES remap for ROCm arch detection ───────────────
//
// amdgpu-arch/rocm_agent_enumerator report devices in HSA/KFD order (honoring
// ROCR_VISIBLE_DEVICES/GPU_DEVICE_ORDINAL) but do NOT honor HIP_VISIBLE_DEVICES,
// a HIP-runtime-only env var. detect_rocm_gfx_arch() takes a HIP ordinal (the
// same numbering IREE's hip://N HAL driver and Tenzor's own ROCm backend use),
// so it must remap that ordinal into HSA order before indexing the
// enumerator's output. This is pure env-var parsing, testable without ROCm
// hardware.
TEST(JitRocmArch, RemapHipVisibleDeviceIndex) {
    // No HIP_VISIBLE_DEVICES set (or empty): identity mapping.
    EXPECT_EQ(tj::remap_hip_visible_device_index(0, nullptr), 0);
    EXPECT_EQ(tj::remap_hip_visible_device_index(2, nullptr), 2);
    EXPECT_EQ(tj::remap_hip_visible_device_index(0, ""), 0);

    // HIP_VISIBLE_DEVICES="1,0" reverses a 2-GPU system: HIP ordinal 0 is
    // physical/HSA ordinal 1, HIP ordinal 1 is physical/HSA ordinal 0.
    EXPECT_EQ(tj::remap_hip_visible_device_index(0, "1,0"), 1);
    EXPECT_EQ(tj::remap_hip_visible_device_index(1, "1,0"), 0);

    // HIP_VISIBLE_DEVICES="2" restricts to a single physical device: only
    // HIP ordinal 0 is valid and maps to physical ordinal 2.
    EXPECT_EQ(tj::remap_hip_visible_device_index(0, "2"), 2);

    // Requesting a HIP ordinal beyond the visible-device list falls back to
    // identity (out-of-range; caller's bounds check on `archs` catches it).
    EXPECT_EQ(tj::remap_hip_visible_device_index(1, "2"), 1);

    // Whitespace around entries is tolerated (matches HIP runtime leniency).
    EXPECT_EQ(tj::remap_hip_visible_device_index(1, " 3, 1 , 0"), 1);

    // Malformed entries stop parsing at the first bad token; ordinals parsed
    // before it stay valid, matching HIP runtime behavior.
    EXPECT_EQ(tj::remap_hip_visible_device_index(0, "1,garbage,0"), 1);
    EXPECT_EQ(tj::remap_hip_visible_device_index(1, "1,garbage,0"), 1);

    // Negative ordinal input is passed through unchanged (defensive).
    EXPECT_EQ(tj::remap_hip_visible_device_index(-1, "1,0"), -1);
}

// JIT-R129 regression: detect_rocm_gfx_arch()'s process-lifetime cache used
// to be keyed ONLY by HIP device_index, not by the HIP_VISIBLE_DEVICES value
// the remap actually depends on -- so if the env var changed between two
// calls for the SAME ordinal (test harnesses using setenv/unsetenv across
// cases, or any embedding scenario mutating it between JIT compiles), the
// second call would silently return the FIRST call's cached result instead
// of re-deriving it. Demonstrated with a single real GPU (this box reports
// exactly one, gfx1150): HIP ordinal 3 doesn't exist with no
// HIP_VISIBLE_DEVICES override (identity remap => out-of-range => empty).
// Setting HIP_VISIBLE_DEVICES to a 4-entry list whose index 3 is "0" remaps
// that SAME ordinal 3 to physical/HSA ordinal 0, which DOES exist. The old
// index-only cache would incorrectly hand back the first call's cached
// empty string; the fixed env-folded cache key re-derives and returns the
// real arch.
TEST(JitRocmArch, CacheKeyRespectsHipVisibleDevicesChanges) {
    ensure_core_init();
    if (!tj::iree_can_initialize_default_device("hip")) {
        if (::tenzor::testing::golden::require_multi_backend()) {
            FAIL() << "ROCm/HIP device required by TENZOR_REQUIRE_MULTI_BACKEND "
                      "but unavailable";
        }
        GTEST_SKIP() << "no ROCm/HIP device";
    }

    const char* saved_raw = std::getenv("HIP_VISIBLE_DEVICES");
    const bool had_saved = saved_raw != nullptr;
    const std::string saved = had_saved ? saved_raw : std::string{};
    struct EnvGuard {
        bool had;
        std::string val;
        ~EnvGuard() {
            if (had) setenv("HIP_VISIBLE_DEVICES", val.c_str(), 1);
            else unsetenv("HIP_VISIBLE_DEVICES");
        }
    } env_guard{had_saved, saved};

    // A HIP ordinal never queried by any other test in this binary, so its
    // cache entries can't collide with e.g. DetectedArchMatchesDeviceAndCompiles's
    // ordinal-0 entry.
    constexpr int kOrdinal = 3;

    unsetenv("HIP_VISIBLE_DEVICES");
    const std::string arch_before = tj::detect_rocm_gfx_arch(kOrdinal);
    EXPECT_TRUE(arch_before.empty())
        << "HIP ordinal " << kOrdinal << " should not exist on this box "
           "with no HIP_VISIBLE_DEVICES remap, got: " << arch_before;

    setenv("HIP_VISIBLE_DEVICES", "9,9,9,0", 1);
    const std::string arch_after = tj::detect_rocm_gfx_arch(kOrdinal);
    EXPECT_FALSE(arch_after.empty())
        << "HIP ordinal " << kOrdinal << " remaps to physical ordinal 0 "
           "under HIP_VISIBLE_DEVICES=9,9,9,0 and should now resolve to a "
           "real gfx arch -- got empty, which means the cache incorrectly "
           "reused the pre-remap result for the same ordinal under a "
           "different HIP_VISIBLE_DEVICES value";
    EXPECT_EQ(arch_after.rfind("gfx", 0), 0U)
        << "expected a gfx ISA name, got: " << arch_after;
}

// JIT-R118 regression: shared_iree_hal_device()'s cache used to be keyed
// ONLY by device_uri (e.g. "hip://0"/"cuda://0"), not by the
// HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES value that actually determines
// which PHYSICAL GPU that URI's ordinal binds to at device-creation time --
// mirroring JIT-R129's fix to the sibling arch-detection cache one function
// away (see CacheKeyRespectsHipVisibleDevicesChanges above). The real-world
// motivation is CUDA/ROCm-specific, but this host's linked-in IREE runtime
// (like most Tenzor distributions) has no in-process cuda/hip HAL driver at
// all -- IreeInvoker::load(..., Mode::InProcess, ...) for those drivers
// always throws "no driver registered" before ever reaching the cache, so
// the stale-cache-hit scenario can't be exercised with real cuda/hip
// hardware here. shared_iree_hal_device()'s cache-key construction is
// driver-agnostic, so this test instead exercises the SAME code path via
// Vulkan (which this host does have an in-process driver for) and toggles
// CUDA_VISIBLE_DEVICES (a no-op for Vulkan device SELECTION, but exactly
// the value the fixed cache key folds in for any non-"hip" driver) between
// two loads for the same URI/ordinal: pre-fix, the cache is keyed by
// device_uri alone, so it unconditionally hits on the second call and
// returns the SAME device object regardless of any env change; post-fix,
// the differing env value changes the cache key, forcing a fresh device.
TEST(JitRocmArch, SharedHalDeviceCacheRespectsVisibleDevicesEnvChanges) {
    ensure_core_init();

    ::tenzor::jit::Graph g;
    auto x_v = g.create_value("x", {4}, ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    g.set_inputs({x_v});
    auto add_node_ = g.create_node(::tenzor::jit::OpType::Add);
    add_node_->add_input(x_v);
    add_node_->add_input(x_v);
    auto out = g.create_value("out", {4}, ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    add_node_->add_output(out);
    g.add_node(add_node_);
    g.set_outputs({out});

    tj::GraphToMLIR lowerer;
    const std::string mlir = lowerer.lower(g);

    tj::CompileOptions opts;
    opts.target    = "vulkan-spirv";
    opts.cache_dir = make_tmp_dir();
    tj::CompiledArtifact artifact;
    ASSERT_NO_THROW({ artifact = tj::compile_mlir(mlir, opts); });

    std::unique_ptr<tj::IreeInvoker> invoker_a;
    try {
        invoker_a =
            tj::IreeInvoker::load(artifact, tj::IreeInvoker::Mode::InProcess, 0);
    } catch (const std::exception&) {
        GTEST_SKIP() << "no in-process Vulkan device available";
    }
    if (invoker_a->raw_device_handle_for_testing() == nullptr) {
        GTEST_SKIP() << "no in-process Vulkan device available";
    }

    const char* saved_raw = std::getenv("CUDA_VISIBLE_DEVICES");
    const bool had_saved = saved_raw != nullptr;
    const std::string saved = had_saved ? saved_raw : std::string{};
    struct EnvGuard {
        bool had;
        std::string val;
        ~EnvGuard() {
            if (had) setenv("CUDA_VISIBLE_DEVICES", val.c_str(), 1);
            else unsetenv("CUDA_VISIBLE_DEVICES");
        }
    } env_guard{had_saved, saved};

    unsetenv("CUDA_VISIBLE_DEVICES");
    auto invoker_before =
        tj::IreeInvoker::load(artifact, tj::IreeInvoker::Mode::InProcess, 0);
    const void* device_before = invoker_before->raw_device_handle_for_testing();
    ASSERT_NE(device_before, nullptr);

    setenv("CUDA_VISIBLE_DEVICES", "0", 1);
    auto invoker_after =
        tj::IreeInvoker::load(artifact, tj::IreeInvoker::Mode::InProcess, 0);
    const void* device_after = invoker_after->raw_device_handle_for_testing();
    ASSERT_NE(device_after, nullptr);

    EXPECT_NE(device_before, device_after)
        << "shared_iree_hal_device()'s cache reused the SAME device object "
           "for the same driver/ordinal across a visible-devices env change "
           "-- must create a fresh device per distinct visible-devices value";

    std::error_code ec;
    fs::remove_all(opts.cache_dir, ec);
}

// JIT-R123 regression: iree_can_initialize_default_device()'s cache used to
// be keyed ONLY by driver_name, not by the HIP_VISIBLE_DEVICES/
// CUDA_VISIBLE_DEVICES value that determines which PHYSICAL GPU (if any)
// ordinal 0 actually binds to -- the sibling of JIT-R118's fix to
// shared_iree_hal_device (which this function calls internally). Demonstrated
// by probing "hip" with HIP_VISIBLE_DEVICES unset (available on this host)
// vs set to an out-of-range ordinal that hides every real GPU (unavailable):
// the old cache would return the FIRST probe's answer for BOTH calls; the
// fixed cache must re-probe and return a DIFFERENT (correct) answer.
TEST(JitRocmArch, CanInitializeDefaultDeviceCacheRespectsHipVisibleDevicesChanges) {
    ensure_core_init();

    const char* saved_raw = std::getenv("HIP_VISIBLE_DEVICES");
    const bool had_saved = saved_raw != nullptr;
    const std::string saved = had_saved ? saved_raw : std::string{};
    struct EnvGuard {
        bool had;
        std::string val;
        ~EnvGuard() {
            if (had) setenv("HIP_VISIBLE_DEVICES", val.c_str(), 1);
            else unsetenv("HIP_VISIBLE_DEVICES");
        }
    } env_guard{had_saved, saved};

    unsetenv("HIP_VISIBLE_DEVICES");
    const bool ok_before = tj::iree_can_initialize_default_device("hip");
    if (!ok_before) {
        if (::tenzor::testing::golden::require_multi_backend()) {
            FAIL() << "ROCm/HIP device required by TENZOR_REQUIRE_MULTI_BACKEND "
                      "but unavailable";
        }
        GTEST_SKIP() << "no ROCm/HIP device";
    }

    // An out-of-range ordinal hides every real GPU on any host with fewer
    // than 100 devices.
    setenv("HIP_VISIBLE_DEVICES", "99", 1);
    const bool ok_after = tj::iree_can_initialize_default_device("hip");
    EXPECT_FALSE(ok_after)
        << "HIP_VISIBLE_DEVICES=99 hides every real GPU and "
           "iree_can_initialize_default_device(\"hip\") should now report "
           "false -- true means the cache incorrectly reused the pre-remap "
           "availability answer instead of re-probing under the new value";
}

// ── Fix #3: a graph break / empty trace makes trace_and_compile() return
//    nullptr (NOT an exception). On the nvrtc inference path this previously
//    slipped past the strict check and silently degraded to eager, while the
//    mlir path threw — an inconsistent strict contract across backends and
//    between inference and training. Under strict it must now throw; under
//    non-strict it degrades to eager (correct result). An identity function
//    produces an empty trace (num_nodes()==0 -> nullptr). ────────────────────
TEST(JitEagerFallback, EmptyTraceHonorsStrictNvrtc) {
    ensure_core_init();
    auto identity = [](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        return x;  // no traceable ops -> empty graph -> trace_and_compile==nullptr
    };
    auto x_t = ::tenzor::full({4}, 1.5F, ::tenzor::DType::Float32);
    ::tenzor::Variable x(x_t, /*requires_grad=*/false);

    // Strict: must THROW rather than silently degrade to eager.
    {
        ::tenzor::jit::CompileConfig cfg;
        cfg.backend = "nvrtc";
        cfg.strict  = true;
        ::tenzor::jit::CompiledFunction compiled(
            ::tenzor::jit::CompiledFunction::FnType(identity), cfg);
        EXPECT_THROW({ (void)compiled(x); }, std::exception)
            << "strict nvrtc must throw on an empty-trace/graph-break nullptr";
    }
    // Non-strict: degrades to eager, no throw, correct result.
    {
        ::tenzor::jit::CompileConfig cfg;
        cfg.backend = "nvrtc";
        cfg.strict  = false;
        ::tenzor::jit::CompiledFunction compiled(
            ::tenzor::jit::CompiledFunction::FnType(identity), cfg);
        ::tenzor::Variable out;
        ASSERT_NO_THROW({ out = compiled(x); });
        const auto diff =
            ::tenzor::max(::tenzor::abs(x.tensor() - out.tensor())).item<float>();
        EXPECT_LT(diff, 1e-6F);
    }
}

// ── R1-12: InProcess vs Subprocess HAL-driver parity for llvm-cpu ──────────
//
// InProcess mode drives llvm-cpu through the "local-task" HAL driver (IREE's
// multithreaded task system, created once and reused for the runtime
// instance's lifetime); invoke_subprocess() unconditionally rewrites
// "local-task" to "local-sync" (a single-threaded inline queue) for the
// `iree-run-module` CLI path. Both are real, independently-selectable IREE
// HAL drivers (confirmed via `iree-run-module --list_drivers`); local-sync
// is kept for the CLI specifically to avoid paying a full worker-thread-pool
// spin-up/teardown on every short-lived one-shot subprocess invocation, and
// to keep that path single-threaded and deterministic for debugging.
// Because compile.cpp can silently fall back from InProcess to Subprocess
// for a target whose HAL driver isn't statically linked into the runtime,
// the SAME compiled artifact can execute through either path across
// different runs -- so the two drivers must agree numerically. This test
// compiles a genuine reduction (the operation class most sensitive to
// worker-thread partitioning/combine order) over a tensor large enough for
// IREE's llvm-cpu codegen to tile it, and asserts InProcess and Subprocess
// produce the same result.
TEST(JitRocmArch, InProcessSubprocessParity_ReductionOnLlvmCpu) {
    ensure_core_init();

    constexpr int64_t kRows = 256;
    constexpr int64_t kCols = 512;
    auto x_t = ::tenzor::full({kRows, kCols}, 0.0F, ::tenzor::DType::Float32);
    {
        auto* xp = x_t.data<float>();
        std::mt19937_64 rng(12345);
        std::uniform_real_distribution<float> dist(-2.0F, 2.0F);
        for (int64_t i = 0; i < x_t.numel(); ++i) xp[i] = dist(rng);
    }

    ::tenzor::jit::Graph g;
    auto x_v = g.create_value("x", {kRows, kCols}, ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    g.set_inputs({x_v});
    auto node = g.create_node(::tenzor::jit::OpType::Sum);
    node->add_input(x_v);
    auto out_v = g.create_value("o", {kRows}, ::tenzor::DType::Float32,
                                ::tenzor::Device::cpu());
    node->add_output(out_v);
    node->set_int_attr("dim", 1);
    g.add_node(node);
    g.set_outputs({out_v});

    tj::GraphToMLIR lowerer;
    const std::string mlir = lowerer.lower(g);
    ASSERT_NE(mlir.find("stablehlo.reduce"), std::string::npos) << mlir;

    tj::CompileOptions opts;
    opts.target    = "llvm-cpu";
    opts.cache_dir = make_tmp_dir();
    tj::CompiledArtifact artifact;
    ASSERT_NO_THROW({ artifact = tj::compile_mlir(mlir, opts); });

    auto in_proc_invoker =
        tj::IreeInvoker::load(artifact, tj::IreeInvoker::Mode::InProcess);
    const auto in_proc_out = in_proc_invoker->invoke({x_t});
    ASSERT_EQ(in_proc_out.size(), 1u);

    auto sub_proc_invoker =
        tj::IreeInvoker::load(artifact, tj::IreeInvoker::Mode::Subprocess);
    const auto sub_proc_out = sub_proc_invoker->invoke({x_t});
    ASSERT_EQ(sub_proc_out.size(), 1u);

    const auto diff =
        ::tenzor::max(::tenzor::abs(in_proc_out[0] - sub_proc_out[0]))
            .item<float>();
    EXPECT_LT(diff, 1e-6F)
        << "InProcess (local-task, multithreaded) and Subprocess "
           "(local-sync, single-threaded) diverged on the same compiled "
           "llvm-cpu reduction; diff=" << diff;

    std::error_code ec;
    fs::remove_all(opts.cache_dir, ec);
}

// R1-06 regression: a multi-output subprocess invocation where one output is
// BFloat16 (which has no numpy encoding in iree-run-module) used to silently
// fall back to lossy ~6-significant-figure ASCII parsing for ALL outputs,
// including a co-produced Float64 output that needs full precision. This is
// currently unreachable via CompiledFunction (which enforces exactly one
// @main output), so this test drives IreeInvoker directly with
// set_expected_outputs(2) to exercise the guard added in invoke_subprocess.
TEST(JitEagerFallback, MultiOutputSubprocessBf16WithFloat64ThrowsInsteadOfSilentlyDegrading) {
    ensure_core_init();

    ::tenzor::jit::Graph g;
    auto x_v = g.create_value("x", {4}, ::tenzor::DType::Float64,
                              ::tenzor::Device::cpu());
    g.set_inputs({x_v});

    auto cast_node = g.create_node(::tenzor::jit::OpType::Cast);
    cast_node->add_input(x_v);
    auto bf16_out = g.create_value("bf16_out", {4}, ::tenzor::DType::BFloat16,
                                   ::tenzor::Device::cpu());
    cast_node->add_output(bf16_out);
    g.add_node(cast_node);

    auto add_node_ = g.create_node(::tenzor::jit::OpType::Add);
    add_node_->add_input(x_v);
    add_node_->add_input(x_v);
    auto f64_out = g.create_value("f64_out", {4}, ::tenzor::DType::Float64,
                                  ::tenzor::Device::cpu());
    add_node_->add_output(f64_out);
    g.add_node(add_node_);

    g.set_outputs({bf16_out, f64_out});

    tj::GraphToMLIR lowerer;
    const std::string mlir = lowerer.lower(g);
    ASSERT_NE(mlir.find("func.func @main"), std::string::npos) << mlir;

    tj::CompileOptions opts;
    opts.target    = "llvm-cpu";
    opts.cache_dir = make_tmp_dir();
    tj::CompiledArtifact artifact;
    ASSERT_NO_THROW({ artifact = tj::compile_mlir(mlir, opts); });

    auto invoker =
        tj::IreeInvoker::load(artifact, tj::IreeInvoker::Mode::Subprocess);
    invoker->set_expected_outputs(2);

    auto x_t = ::tenzor::full({4}, 0.1, ::tenzor::DType::Float64);
    EXPECT_THROW({ (void)invoker->invoke({x_t}); }, std::exception)
        << "a multi-output subprocess invocation with a co-produced BFloat16 "
           "output must not silently return a precision-degraded Float64 "
           "result";

    std::error_code ec;
    fs::remove_all(opts.cache_dir, ec);
}

// JIT-R128 regression: the ASCII-stdout fallback's precision-loss guard used
// to only fire `if (outs.size() > 1)`, reasoning that in the single-output
// case "that one output is whatever triggered this fallback" -- true only
// when the fallback was reached via the BFloat16-numpy-encoding-gap path
// (safe, since the one output IS the bf16 one). That reasoning breaks when
// npy_path itself is empty because mkstemps() failed to create the .npy
// temp file BEFORE the bit-exact path was ever attempted -- a dtype-
// independent cause (temp dir full/inaccessible/restrictive sandbox) that
// can strand a LONE Float64 output on lossy ~6-sig-fig ASCII parsing with
// no encoding-gap excuse at all. Force that exact failure with a real,
// existing-but-unwritable TMPDIR (so std::filesystem::temp_directory_path()
// still succeeds but mkstemps() gets EACCES), and confirm the single-output
// subprocess invocation now throws instead of silently returning a
// precision-degraded Float64 result.
TEST(JitEagerFallback, SingleOutputSubprocessThrowsWhenNpyTempFileUnavailable) {
    ensure_core_init();

    ::tenzor::jit::Graph g;
    auto x_v = g.create_value("x", {4}, ::tenzor::DType::Float64,
                              ::tenzor::Device::cpu());
    g.set_inputs({x_v});

    auto add_node_ = g.create_node(::tenzor::jit::OpType::Add);
    add_node_->add_input(x_v);
    add_node_->add_input(x_v);
    auto f64_out = g.create_value("f64_out", {4}, ::tenzor::DType::Float64,
                                  ::tenzor::Device::cpu());
    add_node_->add_output(f64_out);
    g.add_node(add_node_);
    g.set_outputs({f64_out});

    tj::GraphToMLIR lowerer;
    const std::string mlir = lowerer.lower(g);
    ASSERT_NE(mlir.find("func.func @main"), std::string::npos) << mlir;

    tj::CompileOptions opts;
    opts.target    = "llvm-cpu";
    opts.cache_dir = make_tmp_dir();
    tj::CompiledArtifact artifact;
    ASSERT_NO_THROW({ artifact = tj::compile_mlir(mlir, opts); });

    auto invoker =
        tj::IreeInvoker::load(artifact, tj::IreeInvoker::Mode::Subprocess);
    auto x_t = ::tenzor::full({4}, 0.1, ::tenzor::DType::Float64);

    // Create a real, existing directory with no write permission, and point
    // TMPDIR at it: temp_directory_path() still succeeds (the path exists
    // and is a directory), but mkstemps() inside invoke_subprocess fails
    // with EACCES trying to create the .npy file in it, leaving npy_path
    // empty -- exactly the fault this guard must catch.
    const fs::path unwritable_dir = make_tmp_dir() / "unwritable";
    fs::create_directories(unwritable_dir);
    fs::permissions(unwritable_dir, fs::perms::owner_read | fs::perms::owner_exec,
                    fs::perm_options::replace);

    const char* saved_tmpdir_raw = std::getenv("TMPDIR");
    const bool had_saved = saved_tmpdir_raw != nullptr;
    const std::string saved = had_saved ? saved_tmpdir_raw : std::string{};
    struct EnvGuard {
        bool had;
        std::string val;
        ~EnvGuard() {
            if (had) setenv("TMPDIR", val.c_str(), 1);
            else unsetenv("TMPDIR");
        }
    } env_guard{had_saved, saved};
    setenv("TMPDIR", unwritable_dir.string().c_str(), 1);

    try {
        (void)invoker->invoke({x_t});
        FAIL() << "expected invoke() to throw when the .npy temp file "
                  "cannot be created for a precision-sensitive output";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("no bit-exact"), std::string::npos)
            << "expected the JIT-R128 precision-loss guard's message, got: "
            << e.what();
    }

    // Restore permissions before cleanup so remove_all can actually delete it.
    fs::permissions(unwritable_dir, fs::perms::owner_all, fs::perm_options::replace);
    std::error_code ec;
    fs::remove_all(opts.cache_dir, ec);
    fs::remove_all(unwritable_dir.parent_path(), ec);
}

// JIT-R125 regression (multi-output subprocess mkstemps partial failure):
// see tests/jit/mlir/test_jit_r125_output_temp_exhaustion.cpp -- moved to
// its own executable because the repro needs unshare(CLONE_NEWUSER), which
// only succeeds in a single-threaded process; this suite's other tests
// (backend loading, ConcurrentSharedHalDeviceCreationIsRaceFree, etc.)
// leave the process multi-threaded for the rest of its life.

// JIT-R130 regression: shared_iree_hal_device() used to hold
// g_shared_iree_device_mu for the ENTIRE device-creation call, serializing
// device creation across every distinct driver/URI. The fix moved creation
// outside the lock with a double-checked pattern (look up under the lock,
// create outside it, then insert-or-discard under the lock again) -- the
// risk that refactor introduces is a race when multiple threads miss the
// cache for the SAME URI simultaneously and each create their own device:
// exactly one must win the cache and the rest must be released, with every
// caller ending up with a handle to the SAME winning device (not a stale
// or double-freed one). Stress this directly: many threads concurrently
// load a fresh IreeInvoker for the identical llvm-cpu artifact/ordinal
// (guaranteeing a cold cache the first time this test runs) and invoke it,
// asserting every thread gets the numerically correct result with no
// crash, deadlock, or corruption -- which would only happen if the
// insert-or-discard logic mishandled the race.
TEST(JitRocmArch, ConcurrentSharedHalDeviceCreationIsRaceFree) {
    ensure_core_init();

    ::tenzor::jit::Graph g;
    auto x_v = g.create_value("x", {4}, ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    g.set_inputs({x_v});
    auto add_node_ = g.create_node(::tenzor::jit::OpType::Add);
    add_node_->add_input(x_v);
    add_node_->add_input(x_v);
    auto out_v = g.create_value("out", {4}, ::tenzor::DType::Float32,
                                ::tenzor::Device::cpu());
    add_node_->add_output(out_v);
    g.add_node(add_node_);
    g.set_outputs({out_v});

    tj::GraphToMLIR lowerer;
    const std::string mlir = lowerer.lower(g);
    ASSERT_NE(mlir.find("func.func @main"), std::string::npos) << mlir;

    tj::CompileOptions opts;
    opts.target    = "llvm-cpu";
    opts.cache_dir = make_tmp_dir();
    tj::CompiledArtifact artifact;
    ASSERT_NO_THROW({ artifact = tj::compile_mlir(mlir, opts); });

    auto x_t = ::tenzor::full({4}, 2.5F, ::tenzor::DType::Float32);

    constexpr int kThreads = 16;
    std::vector<std::thread> threads;
    std::vector<bool> ok(kThreads, false);
    std::vector<std::string> errors(kThreads);
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&, i] {
            try {
                auto invoker = tj::IreeInvoker::load(
                    artifact, tj::IreeInvoker::Mode::InProcess);
                const auto out = invoker->invoke({x_t});
                if (out.size() == 1u) {
                    const auto diff =
                        ::tenzor::max(::tenzor::abs(out[0] - x_t - x_t))
                            .item<float>();
                    ok[i] = diff < 1e-6F;
                    if (!ok[i]) errors[i] = "numeric mismatch, diff=" + std::to_string(diff);
                } else {
                    errors[i] = "unexpected output count: " + std::to_string(out.size());
                }
            } catch (const std::exception& e) {
                errors[i] = e.what();
            }
        });
    }
    for (auto& t : threads) t.join();

    for (int i = 0; i < kThreads; ++i) {
        EXPECT_TRUE(ok[i]) << "thread " << i << " failed: " << errors[i];
    }

    std::error_code ec;
    fs::remove_all(opts.cache_dir, ec);
}
