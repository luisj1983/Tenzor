// Phase 13 / Task A.8 — IREE Runtime invocation wrapper.
//
// Environmental note (2026-05-19 amendment, item §):
// The local IREE distributions on this host (iree-dist + pip-installed
// iree-base-{compiler,runtime}) ship complete iree-compile / iree-run-module
// binaries but neither ships the full iree/base/*.h header set that the
// in-process iree/runtime/api.h depends on (9 missing base headers — see
// src/jit/mlir/iree_customcalls.cpp for the list). IreeInvoker therefore
// drives the runtime via the `iree-run-module` subprocess discovered by
// tenzor::jit::mlir_jit::resolve_iree_run_module() (env override → pip venv
// → CMake-found dist → $PATH). This is a real invocation — the subprocess
// loads the same .vmfb bytecode the embedding-API compile produced and runs
// it on the same HAL backends — not a stub.
//
// The in-process runtime + custom_call plugin (Path A of the 2026-05-19
// amendment) lights up automatically once a header-complete IREE C SDK is
// installed; the runtime callbacks in iree_customcalls.cpp are wired up to
// dispatch into existing kernels, but currently return UNIMPLEMENTED
// because the C glue cannot compile without the base headers.

#pragma once

#include "tenzor/core/tensor.hpp"
#include "tenzor/jit/mlir/iree_compile.hpp"

#include <memory>
#include <string>
#include <vector>

namespace tenzor::jit::mlir_jit {

/// Loads a compiled vmfb artifact and exposes a synchronous `invoke()` that
/// marshals tensors into the IREE runtime and unmarshals the outputs back.
///
/// One IreeInvoker corresponds to one vmfb at one target. The default HAL
/// device is selected by IREE based on the target backend in the artifact
/// (e.g. `local-sync` for `llvm-cpu`).
class IreeInvoker {
public:
    static auto load(const CompiledArtifact& artifact)
        -> std::unique_ptr<IreeInvoker>;

    /// Invoke the `@main` function with the given input tensors. Returns the
    /// output tensors in the order they were returned by the MLIR function.
    /// Float32 and Float64 dtypes are supported; other dtypes throw
    /// std::invalid_argument until extended in later tasks.
    auto invoke(const std::vector<::tenzor::Tensor>& inputs)
        -> std::vector<::tenzor::Tensor>;

    ~IreeInvoker();

    // Non-copyable, non-movable to keep the underlying device handle pinned.
    IreeInvoker(const IreeInvoker&)            = delete;
    auto operator=(const IreeInvoker&) -> IreeInvoker& = delete;
    IreeInvoker(IreeInvoker&&)                 = delete;
    auto operator=(IreeInvoker&&) -> IreeInvoker&     = delete;

private:
    IreeInvoker() = default;

    std::string vmfb_path_;
    std::string target_;
    std::string device_;
    std::string iree_run_module_;
};

/// Thrown when iree-run-module reports a runtime failure. Carries the full
/// stderr text for debugging.
class JitInvokeError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

}  // namespace tenzor::jit::mlir_jit
