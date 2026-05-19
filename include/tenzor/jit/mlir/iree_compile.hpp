// Phase 13 / Task A.7 — IREE Compiler embedding API integration.
//
// Per the 2026-05-19 amendment, Tenzor compiles MLIR text in-process via
// the IREE Compiler embedding API (iree/compiler/embedding_api.h). The
// resulting vmfb bytecode is cached on disk keyed by (sha256(text), target,
// iree_compiler_version).

#pragma once

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

    /// When true, downstream IreeInvoker will register the Tenzor custom_call
    /// resolvers. When false, the compiled module is expected to be free of
    /// `stablehlo.custom_call @tenzor_*` invocations (used by IR-only tests).
    bool plugin_enabled = true;
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

}  // namespace tenzor::jit::mlir_jit
