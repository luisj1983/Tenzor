// Phase 13 / Task A.7 — IREE Compiler embedding API integration.
//
// Per the 2026-05-19 amendment, Tenzor compiles MLIR text in-process via
// the IREE Compiler embedding API (iree/compiler/embedding_api.h). The
// resulting vmfb bytecode is cached on disk keyed by (sha256(text), target,
// iree_compiler_version).

#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace tenzor::jit::mlir_jit {

/// Options controlling a single compile invocation.
struct CompileOptions {
    /// IREE HAL target backend identifier: "llvm-cpu", "cuda", "vulkan-spirv",
    /// "rocm".
    std::string target;

    /// On-disk cache directory. If empty, defaults to
    /// $XDG_CACHE_HOME/tenzor/jit_mlir (falling back to $HOME/.cache/...).
    std::filesystem::path cache_dir;

    /// When true, the fused dialect ops (FlashAttention/GQA/RoPE/RMSNorm) lower
    /// to `stablehlo.custom_call @tenzor_*` and IreeInvoker registers the custom
    /// resolvers — but those resolver shims marshal buffers to CPU and run the
    /// CPU kernel regardless of the compiled HAL target, which breaks
    /// cross-backend parity. Default OFF so those ops instead lower via the
    /// expand-to-StableHLO path and execute natively on the selected target.
    /// Only enable for IR-only tests that assert on the custom_call form.
    bool plugin_enabled = false;

    /// For target=="rocm": the AMD GPU ISA to compile for (e.g. "gfx1150"),
    /// derived from the *actual* ROCm device rather than a build-time
    /// constant. When empty, compile_mlir() falls back to the
    /// TENZOR_DEFAULT_ROCM_TARGET build constant / "gfx1150". Ignored for
    /// non-rocm targets. The value participates in the on-disk cache key so
    /// artifacts for different GPU ISAs never alias.
    std::string rocm_arch;

    /// For target=="vulkan-spirv": the IREE Vulkan target controlling the SPIR-V
    /// environment (e.g. "ampere", "rdna3", "sm_80"), derived from the actual
    /// Vulkan device's vendor. REQUIRED for correctness: without it IREE compiles
    /// against a conservative default SPIR-V env that lacks shaderFloat16 / 16-bit
    /// storage, so an F16/BF16 graph silently produces garbage/NaN on the GPU
    /// (while F32 works). When empty, compile_mlir() consults the
    /// TENZOR_VULKAN_TARGET env var, then the TENZOR_DEFAULT_VULKAN_TARGET build
    /// constant. Ignored for non-vulkan targets. Participates in the on-disk cache
    /// key so artifacts for different Vulkan devices never alias.
    std::string vulkan_arch;

    /// For target=="cuda": the CUDA arch (e.g. "sm_90"), derived from the actual
    /// device's compute capability. Without it the CUDA arch tracked only
    /// TENZOR_CUDA_TARGET / the build default / sm_80, so a Hopper/Blackwell GPU
    /// still got sm_80 PTX and any sm_90+-only op forced eager fallback (F004).
    /// When empty, compile_mlir() consults TENZOR_CUDA_TARGET, then the
    /// TENZOR_DEFAULT_CUDA_TARGET build constant, then sm_80. Ignored for
    /// non-cuda targets. Participates in the on-disk cache key.
    std::string cuda_arch;
};

/// Result of a successful compile invocation.
struct CompiledArtifact {
    /// Absolute path to the vmfb bytecode file inside `CompileOptions::cache_dir`.
    std::filesystem::path vmfb_path;
    /// Echo of the target backend the artifact was compiled for.
    std::string target;
};

/// Thrown when the IREE compiler reports any error during a compile run. The
/// `.what()` message contains the full IREE diagnostic. The original MLIR text
/// is preserved in `mlir_dump_path` for offline debugging.
class JitCompileError : public std::runtime_error {
public:
    JitCompileError(std::string msg, std::filesystem::path mlir_dump_path)
        : std::runtime_error(std::move(msg)),
          mlir_dump_path_(std::move(mlir_dump_path)) {}

    auto mlir_dump_path() const -> const std::filesystem::path& {
        return mlir_dump_path_;
    }

private:
    std::filesystem::path mlir_dump_path_;
};

/// Compile an MLIR module (text bytes) for the given target.
///
/// The output is cached on disk keyed by `compute_cache_key(text, target)`
/// concatenated with the IREE compiler revision string. Repeated calls with
/// the same triple reuse the cached vmfb without re-invoking the compiler.
///
/// Throws `JitCompileError` on any compile failure with the full IREE
/// diagnostic text and a path to the dumped MLIR for debugging.
auto compile_mlir(const std::string& mlir_text,
                  const CompileOptions& opts) -> CompiledArtifact;

/// Deterministic cache key over (text, target, iree_compiler_version()).
/// The returned string is hex-encoded and safe for use as a filename.
auto compute_cache_key(const std::string& mlir_text,
                       const std::string& target) -> std::string;

/// Returns the IREE compiler revision string as reported by
/// `ireeCompilerGetRevision()`. Result is the empty string only if the
/// compiler library is broken; otherwise it is some non-empty identifier.
auto iree_compiler_version() -> std::string;


/// Counters reported by `cache_stats()`. All values are sticky for the
/// lifetime of the process unless `reset_cache_stats()` is called.
struct CacheStats {
    /// In-process IreeInvoker cache hits (same shape+dtype+device seen).
    std::uint64_t hits = 0;
    /// In-process IreeInvoker cache misses (forced a trace+compile).
    std::uint64_t misses = 0;
    /// Number of times a function was re-traced because the shape signature
    /// changed (a miss that landed on an already-populated CompiledFunction).
    std::uint64_t retraces = 0;
    /// Number of cache evictions (max_retraces ceiling hit).
    std::uint64_t evictions = 0;
    /// Cumulative wall-clock time spent inside compile_mlir(), milliseconds.
    double total_compile_ms = 0.0;
};

/// Snapshot the cache stats. Thread-safe — reads atomics under the hood.
auto cache_stats() -> CacheStats;

/// Reset all counters back to zero. Test-only helper; production builds may
/// call it after warmup to measure steady-state behavior.
auto reset_cache_stats() -> void;

/// Internal: bump counters from the compile/invoker paths.
namespace internal {
auto record_cache_hit() -> void;
auto record_cache_miss() -> void;
auto record_retrace() -> void;
auto record_eviction() -> void;
auto record_compile_ms(double ms) -> void;
}  // namespace internal

}  // namespace tenzor::jit::mlir_jit
