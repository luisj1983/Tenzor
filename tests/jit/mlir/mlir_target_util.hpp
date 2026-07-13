// Shared IREE-target fan-out helpers for the MLIR JIT numeric tests.
//
// Motivation (audit F9): the per-op numeric tests (test_op_*.cpp) historically
// compiled every case with a hardcoded target="llvm-cpu", so on a host with
// CUDA / ROCm / Vulkan present the IREE GPU code paths got ZERO numeric
// coverage. These helpers let each test iterate over EVERY IREE target that is
// actually usable on this host (hardware present AND the local iree-compile
// dist supports it), comparing the JIT result of each target to the eager
// reference. llvm-cpu is always available, so the tests never silently vanish.
//
// Gating mirrors test_end_to_end_add.cpp exactly so RUN/SKIP decisions stay
// consistent with the end-to-end add test.

#pragma once

#include "tenzor/backend/loader.hpp"
#include "tenzor/jit/mlir/iree_compile.hpp"
#include "tenzor/jit/mlir/iree_paths.hpp"
#include "tenzor/jit/mlir/iree_runtime.hpp"
#include "tenzor/tenzor.hpp"

#include <gtest/gtest.h>

#include <functional>
#include <string>
#include <vector>

namespace tenzor::testing::mlir {

// audit Tier-1 #1: a compiled-vs-eager numeric check is VACUOUS if the compiled
// call silently fell back to eager (mlir_invoke does this, non-strict, whenever
// an op fails to lower) — the diff is then eager-vs-eager == 0 and the test
// passes proving nothing. Bracket each compiled(x) with reset_jit_stats()
// before and assert_jit_used(...) after: a recorded compile miss proves the
// IREE StableHLO path actually executed, so the numeric comparison has teeth.
inline auto reset_jit_stats() -> void {
    ::tenzor::jit::mlir_jit::reset_cache_stats();
}
// JIT-R117: misses>=1 alone cannot distinguish "compiled and ran" from
// "attempted a compile, failed, and quietly ran eager" -- a target that
// fails to lower/compile/invoke still increments misses via the initial
// attempt, then silently falls back to eager (mlir_invoke's non-strict
// C2 safety net). test_jit_mlir_numeric_parity.cpp's own local
// assert_parity_over_targets() was independently hardened with this exact
// check after JIT-R012/JIT-R120 found a real bug it would have caught (F16
// SDPA on cuda failed to bufferize and fell back to eager with zero test-
// visible signal). This is that same check, folded into the canonical
// shared helper so the 10 files using assert_jit_used() get it too instead
// of only the one file that happened to get hardened first.
// allow_eager_fallback exists ONLY for the narrow, already-understood case
// where a target genuinely lacks the dtype/op support being exercised --
// eager IS the correct reference there. It must NOT be used to paper over
// a genuine compile/lower/invoke failure.
inline auto assert_jit_used(const std::string& what, const std::string& target,
                            bool allow_eager_fallback = false) -> void {
    EXPECT_GE(::tenzor::jit::mlir_jit::cache_stats().misses, 1u)
        << what << " did NOT run through IREE on target=" << target
        << " (silent eager fallback makes the numeric check vacuous)";
    if (!allow_eager_fallback) {
        EXPECT_EQ(::tenzor::jit::mlir_jit::cache_stats().eager_fallbacks, 0u)
            << what << " silently fell back to eager on target=" << target
            << " (compiled path never actually ran; misses>=1 alone made "
               "this look like real JIT activity -- the numeric check is "
               "vacuous for this target)";
    }
}

inline auto ensure_core_init() -> void {
    static const bool inited = []() {
        ::tenzor::initialize();
        return true;
    }();
    (void)inited;
}

inline auto backend_present(const std::string& name) -> bool {
    auto* be = ::tenzor::backend_registry().get_backend(name);
    if (be == nullptr) return false;
    try {
        return be->device_count() > 0;
    } catch (...) {
        return false;
    }
}

// Is there hardware for this IREE target on the host? llvm-cpu is always yes;
// rocm is gated on IREE being able to initialize its HIP HAL device directly
// (it drives the GPU itself, independent of the Tenzor ROCm backend load).
inline auto target_hw_present(const std::string& target) -> bool {
    if (target == "llvm-cpu")     return true;
    if (target == "cuda")         return backend_present("cuda");
    if (target == "rocm")
        return ::tenzor::jit::mlir_jit::iree_can_initialize_default_device("hip");
    if (target == "vulkan-spirv") return backend_present("vulkan");
    return false;
}

// Does the local iree-compile dist actually support emitting for this target?
inline auto iree_target_supported(const std::string& target) -> bool {
    if (target == "llvm-cpu") return true;
    return ::tenzor::jit::mlir_jit::iree_compile_supports(target);
}

// The full IREE target set the MLIR backend can lower to.
inline auto all_iree_targets() -> std::vector<std::string> {
    // Intel OneAPI (and Apple MPS) are intentionally absent (JIT-F030): IREE has
    // no HAL driver / compile target for them (see CompiledFunction::mlir_invoke_impl,
    // which throws in strict mode and otherwise runs eagerly for those devices).
    // The MLIR/IREE JIT path therefore does not cover OneAPI; on that backend a
    // model runs eagerly (correct, unaccelerated). CPU/CUDA/ROCm/Vulkan are the
    // IREE-supported targets and are all exercised here when the hardware is present.
    return {"llvm-cpu", "cuda", "rocm", "vulkan-spirv"};
}

// R2-03: if TENZOR_REQUIRE_MULTI_BACKEND=1 is set (an environment that is
// SUPPOSED to have GPU hardware for the IREE targets), available_iree_targets()
// silently shrinking to just {"llvm-cpu"} used to have no signal at all — no
// skip marker, no warning, no failure — strictly worse than the project's
// established REQUIRE_MULTI_BACKEND_OR_SKIP convention (parity_test_utils.hpp),
// under which a "should have multiple backends but doesn't" environment always
// gets a visible, attributable failure instead of quietly running reduced
// coverage. Every caller of available_iree_targets() gets this check for free.
inline auto require_multi_backend_for_iree_targets() -> bool {
    const char* v = std::getenv("TENZOR_REQUIRE_MULTI_BACKEND");
    return v && *v && *v != '0';
}

// Targets that are actually usable on THIS host (hardware + iree-compile
// support). Always contains at least "llvm-cpu".
inline auto available_iree_targets() -> std::vector<std::string> {
    std::vector<std::string> out;
    for (const auto& t : all_iree_targets()) {
        if (target_hw_present(t) && iree_target_supported(t)) out.push_back(t);
    }
    if (out.size() <= 1 && require_multi_backend_for_iree_targets()) {
        ADD_FAILURE() << "TENZOR_REQUIRE_MULTI_BACKEND=1 but available_iree_targets() "
                         "found no usable GPU IREE target (only "
                      << (out.empty() ? "none" : out.front())
                      << ") -- this environment is supposed to have GPU hardware; "
                         "a silent shrink to CPU-only coverage here would hide a "
                         "broken environment instead of failing loudly.";
    }
    return out;
}

}  // namespace tenzor::testing::mlir
