// Phase 13 — IREE binary discovery + capability probing.
//
// Centralises the priority chain used by every IREE consumer (compile_mlir
// subprocess fallback, iree-run-module driver, end-to-end tests). The chain
// is:
//
//   1. $TENZOR_IREE_COMPILE / $TENZOR_IREE_RUN_MODULE   (explicit user override)
//   2. /home/lee/.venvs/tenzor-jit/bin/<binary>          (pip venv with full GPU)
//   3. /home/lee/venv-iree/bin/<binary>                  (alt pip venv with full GPU)
//   4. ${IREE_COMPILE_EXECUTABLE} / its sibling          (CMake-found build dist)
//   5. `<binary>` on $PATH                                (last resort)
//
// The first binary the chain finds is cached for the process lifetime and
// probed once with `--iree-hal-list-target-backends` (compiler) /
// `--list_drivers` (runtime) so target-mismatch failures fail fast with a
// clear error rather than producing a cryptic `iree-compile: unknown target`.

#pragma once

#include <set>
#include <string>

namespace tenzor::jit::mlir_jit {

/// Discover the most-capable `iree-compile` binary available to this process.
/// Result is cached the first time it is called.
auto resolve_iree_compile() -> const std::string&;

/// Discover the most-capable `iree-run-module` binary available to this
/// process. Result is cached the first time it is called.
auto resolve_iree_run_module() -> const std::string&;

/// Probe `resolve_iree_compile()` for `--iree-hal-list-target-backends` and
/// return the parsed set. Cached after first call.
auto iree_compile_supported_targets() -> const std::set<std::string>&;

/// Probe `resolve_iree_run_module()` for `--list_drivers` and return the
/// parsed set of HAL driver names. Cached after first call.
auto iree_runtime_supported_drivers() -> const std::set<std::string>&;

/// True iff `target` (a `--iree-hal-target-backends=` value: `llvm-cpu`,
/// `cuda`, `vulkan-spirv`, `rocm`) is in the supported set.
auto iree_compile_supports(const std::string& target) -> bool;

/// Map an IREE HAL target backend (compile-time) to the IREE HAL driver
/// name (runtime). Throws std::invalid_argument for unknown targets.
auto driver_for_target(const std::string& target) -> std::string;

}  // namespace tenzor::jit::mlir_jit
