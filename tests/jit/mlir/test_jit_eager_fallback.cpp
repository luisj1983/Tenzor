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
#include "tenzor/jit/mlir/iree_compile.hpp"
#include "tenzor/jit/mlir/iree_paths.hpp"
#include "tenzor/jit/mlir/iree_runtime.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/tenzor.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <random>
#include <string>

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

void run_c2_unlowered_op(const std::string& target) {
    ensure_core_init();
    if (!target_runnable(target)) {
        GTEST_SKIP() << "target not runnable here: " << target;
    }

    auto tanh_fn = [](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        return ::tenzor::tanh(x);  // OpType::Tanh has no StableHLO lowering
    };

    auto x_tensor = ::tenzor::full({4}, 0.3F, ::tenzor::DType::Float32);
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

// ── H6: "vulkan" is accepted and normalized to "vulkan-spirv" at the compile
//    boundary; a bogus target gives a clear, listing error. ─────────────────

TEST(JitVulkanAlias, CompileMlirNormalizesVulkanAlias) {
    ensure_core_init();
    if (!tj::iree_compile_supports("vulkan-spirv")) {
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

// ── M1: the ROCm arch is read from the actual device and the derived arch
//    compiles. ──────────────────────────────────────────────────────────────

TEST(JitRocmArch, DetectedArchMatchesDeviceAndCompiles) {
    ensure_core_init();
    if (!tj::iree_can_initialize_default_device("hip")) {
        GTEST_SKIP() << "no ROCm/HIP device";
    }
    const std::string arch = tj::detect_rocm_gfx_arch(0);
    ASSERT_FALSE(arch.empty())
        << "arch must be derived from the physical ROCm device, not a "
           "build constant";
    EXPECT_EQ(arch.rfind("gfx", 0), 0U)
        << "expected a gfx ISA name, got: " << arch;

    if (!tj::iree_compile_supports("rocm")) {
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
