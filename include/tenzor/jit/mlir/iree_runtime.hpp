// Phase 13 / Task A.8 + X.2 — IREE Runtime invocation wrapper.
//
// Two operating modes (Path A of the 2026-05-19 amendment):
//
//   - Mode::InProcess  → drives iree_runtime_instance / session / call via
//                        the in-process IREE C API. Registers the Tenzor VM
//                        native module ("tenzor_plugin") that supplies the
//                        4 MVP-1 dialect-op callbacks before loading the
//                        bytecode module. Required whenever the compiled
//                        vmfb references the tenzor_plugin module
//                        (i.e. plugin_enabled=true on the lowering side).
//   - Mode::Subprocess → invokes iree-run-module on the cached .vmfb. Used
//                        for targets where the in-process HAL driver is
//                        unavailable, or for the expand path where the
//                        compiled vmfb is plugin-free.
//
// The in-process path uses the authentic IREE 3.11.0 headers staged in
// third_party/iree_compat (see CMakeLists.txt). The runtime static
// archives at third_party/iree_dist/lib/libiree_runtime_unified.a (and
// friends) are the actual implementation; the headers and archives ship
// from the same upstream tag so ABI matches.
//
// One IreeInvoker corresponds to one vmfb at one target.
//
// Custom_call wiring (Group D): the 4 dialect ops (flash_attention, gqa,
// rope_apply, rms_norm) are lowered to a `call @tenzor_plugin.<op>` against
// a `func.func private` declaration in the @main module. The InProcess
// invoker registers an iree_vm_native_module_t for the "tenzor_plugin"
// module that satisfies these imports before loading the bytecode module,
// so the calls reach customcalls::dispatch_<op>() in iree_customcalls.cpp.

#pragma once

#include "tenzor/core/tensor.hpp"
#include "tenzor/jit/mlir/iree_compile.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace tenzor::jit::mlir_jit {

/// Loads a compiled vmfb artifact and exposes a synchronous `invoke()` that
/// marshals tensors into the IREE runtime and unmarshals the outputs back.
///
/// One IreeInvoker corresponds to one vmfb at one target.
class IreeInvoker {
public:
    enum class Mode {
        Subprocess,  ///< Drive via the iree-run-module CLI.
        InProcess,   ///< Drive via the in-process iree_runtime_* C API.
    };

    /// Construct an invoker for an artifact. Defaults to InProcess so the
    /// plugin path light up automatically; callers needing the subprocess
    /// path (e.g. targets without a registered HAL driver in the linked
    /// runtime) can pass Mode::Subprocess explicitly.
    ///
    /// `device_index` is the HAL device ordinal to run on (M3). On a
    /// multi-GPU host a `cuda:1` model must execute on GPU 1, not the driver
    /// default (GPU 0). The ordinal is threaded into the HAL device URI
    /// (`cuda://1`, `hip://1`, `vulkan://1`) for both the in-process and
    /// subprocess paths. CPU (`local-task`) has no ordinal and ignores it.
    static auto load(const CompiledArtifact& artifact,
                     Mode mode = Mode::InProcess,
                     int device_index = 0)
        -> std::unique_ptr<IreeInvoker>;

    /// The HAL device URI this invoker selects (e.g. "hip://0", "cuda://1",
    /// or "local-task" for CPU). Reflects the ordinal that `load()` resolved.
    auto device_uri() const -> const std::string& { return device_uri_; }

    /// The underlying iree_hal_device_t* (opaque here to avoid leaking the
    /// IREE C API into this header) this invoker's device_uri()/device_index
    /// resolved to via the shared, process-lifetime device cache
    /// (InProcess mode only -- nullptr in Subprocess mode, which has no
    /// in-process device handle). Exposed purely for identity comparison in
    /// tests (JIT-R118: verifying two invokers for the same driver/ordinal
    /// under DIFFERENT CUDA_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES values get
    /// DISTINCT cached device objects rather than silently sharing one bound
    /// to the wrong physical GPU) -- not meant for production use.
    auto raw_device_handle_for_testing() const -> const void* { return device_handle_; }

    /// Invoke the `@main` function with the given input tensors. Returns the
    /// output tensors in the order they were returned by the MLIR function.
    /// Float32 and Float64 dtypes are supported; other dtypes throw
    /// std::invalid_argument until extended in later tasks.
    auto invoke(const std::vector<::tenzor::Tensor>& inputs)
        -> std::vector<::tenzor::Tensor>;

    /// The mode this invoker is operating in.
    auto mode() const -> Mode { return mode_; }

    /// Number of outputs the compiled @main returns. The subprocess path needs
    /// this to write one --output=@file per result (the in-process path pops
    /// results dynamically). Set by the caller from the graph's output count so a
    /// multi-output graph returns ALL results on the subprocess path, not just the
    /// first (JIT-F003). Defaults to 1.
    void set_expected_outputs(int n) { expected_outputs_ = n > 0 ? n : 1; }

    ~IreeInvoker();

    // Non-copyable, non-movable to keep the underlying device handle pinned.
    IreeInvoker(const IreeInvoker&)                          = delete;
    auto operator=(const IreeInvoker&) -> IreeInvoker&       = delete;
    IreeInvoker(IreeInvoker&&)                               = delete;
    auto operator=(IreeInvoker&&) -> IreeInvoker&            = delete;

private:
    IreeInvoker() = default;

    Mode mode_ = Mode::InProcess;

    // Subprocess fields (always set so the destructor and resolver can read
    // them without checking which mode was used).
    std::string vmfb_path_;
    std::string target_;
    std::string device_;            ///< HAL driver name ("local-task" etc.)
    std::string device_uri_;        ///< HAL device URI incl. ordinal (M3).
    int         device_index_ = 0;  ///< Requested HAL device ordinal (M3).
    std::string iree_run_module_;   ///< Subprocess CLI path (subprocess mode).
    int         expected_outputs_ = 1;  ///< @main result count (JIT-F003).

    // In-process fields. Stored as void* in the header so we don't drag
    // <iree/runtime/api.h> through every translation unit. The .cpp
    // unifies these via static_cast<iree_runtime_*_t*> at use sites.
    void* instance_       = nullptr;  ///< iree_runtime_instance_t*
    void* device_handle_  = nullptr;  ///< iree_hal_device_t*
    void* session_        = nullptr;  ///< iree_runtime_session_t*
    void* plugin_module_  = nullptr;  ///< iree_vm_module_t* (tenzor_plugin)

    // An IreeInvoker is cached and shared across callers, but an IREE
    // runtime session is NOT concurrency-safe. Serialize invoke() so
    // concurrent calls on the same cached invoker can't race the session.
    std::mutex invoke_mutex_;
};

/// Probe whether the IREE runtime can create a default device for the
/// given driver (e.g. "hip", "cuda", "vulkan"). On hosts where the
/// Tenzor backend for a target isn't loaded but the IREE runtime can
/// still dlopen the driver's underlying vendor library (libamdhip64,
/// libcuda, libvulkan), this returns true and the JIT-test gating in
/// test_end_to_end_add.cpp / test_model_resnet.cpp / test_model_llama
/// .cpp uses it in place of `backend_present(...)`.
///
/// Side effects: on first call for "hip", if the compile-time
/// `TENZOR_ROCM_RUNTIME_LIB_DIR` was set, this prepends that directory
/// to LD_LIBRARY_PATH (via setenv) so the IREE HIP HAL driver's
/// dlopen("libamdhip64.so") finds the working library set.
///
/// Caches the result per-driver to avoid repeated dlopen+device-create
/// probes (each takes ~30ms on a warm laptop).
auto iree_can_initialize_default_device(const std::string& driver_name)
    -> bool;

/// Build the IREE HAL device URI for a driver + ordinal (M3). GPU drivers
/// ("cuda", "hip", "vulkan") get a `driver://<index>` path so the requested
/// ordinal is honored; the CPU driver ("local-task"/"local-sync") has no
/// ordinal and is returned unchanged. Exposed for direct testing of the
/// ordinal-threading contract.
auto hal_device_uri(const std::string& driver, int device_index)
    -> std::string;

/// Query the AMD GPU ISA (e.g. "gfx1150") of the physical ROCm device at the
/// given ordinal (M1), by running the ROCm toolchain's device enumerator
/// (`amdgpu-arch` / `rocm_agent_enumerator`). This reads the *actual* device
/// arch rather than a build-time constant, so a JIT compile targets the ISA
/// the code will run on. Returns an empty string if no enumerator is found or
/// the ordinal is out of range — callers then fall back to the build default.
/// Results are cached per ordinal.
///
/// `device_index` is a HIP ordinal (the same numbering IREE's `hip://N` HAL
/// driver and Tenzor's own ROCm backend use, both resolved through the real
/// HIP runtime). The enumerator tools instead report devices in HSA/KFD
/// order, honoring `ROCR_VISIBLE_DEVICES`/`GPU_DEVICE_ORDINAL` but *not*
/// `HIP_VISIBLE_DEVICES` (a HIP-runtime-only env var). When
/// `HIP_VISIBLE_DEVICES` reorders or restricts devices relative to that HSA
/// order, this function remaps `device_index` via
/// `remap_hip_visible_device_index` before indexing the enumerator's list,
/// so the returned arch matches the physical device HIP ordinal
/// `device_index` actually refers to.
auto detect_rocm_gfx_arch(int device_index) -> std::string;

/// Remap a HIP ordinal to the ordinal it corresponds to in the HSA/KFD
/// device order (i.e. the order `amdgpu-arch`/`rocm_agent_enumerator` and
/// `ROCR_VISIBLE_DEVICES` use), by parsing `hip_visible_devices` the same
/// way the HIP runtime parses `HIP_VISIBLE_DEVICES`: a comma-separated list
/// of ordinals, where HIP ordinal `i` refers to physical ordinal
/// `list[i]`. If `hip_visible_devices` is null/empty, or `hip_index` is out
/// of range of the parsed list, or the list fails to parse, returns
/// `hip_index` unchanged (no remap). Exposed for direct unit testing without
/// requiring ROCm hardware.
auto remap_hip_visible_device_index(int hip_index,
                                     const char* hip_visible_devices) -> int;

/// Computes the device-selection-env-var portion of the process-lifetime HAL
/// device cache key (shared_iree_hal_device) / default-device-probe cache key
/// (iree_can_initialize_default_device) for the given driver. Physical GPU
/// selection for "hip"/"cuda" is controlled by HIP_VISIBLE_DEVICES /
/// CUDA_VISIBLE_DEVICES; for "vulkan" it's controlled by VK_ICD_FILENAMES
/// (which ICD, hence which GPU(s), gets enumerated), MESA_VK_DEVICE_SELECT
/// (Mesa's device-select layer), and DRI_PRIME (Mesa's discrete-vs-integrated
/// switch on hybrid-graphics laptops) -- NOT CUDA_VISIBLE_DEVICES, which has
/// no bearing on Vulkan device selection (JIT-R141). Any other driver (the
/// CPU task-queue drivers local-sync/local-task) has no device-remapping env
/// var and returns an empty string. Exposed for direct unit testing of the
/// cache-key derivation without requiring the corresponding GPU hardware.
auto device_selection_env_key(const std::string& driver) -> std::string;

/// Force the shared IREE runtime instance / HAL device cache teardown that
/// normally only runs once, via std::atexit, at real process exit. Exposed
/// purely so a test can deterministically exercise the exact teardown-vs-
/// concurrent-use race that JIT-R109 fixed, without needing to actually
/// terminate the process. PERMANENTLY disables shared_iree_runtime_instance()/
/// shared_iree_hal_device() for the remaining lifetime of the process (they
/// correctly throw afterward rather than reviving a freed pointer) -- must
/// only be called from a test that is the last (or only) use of the MLIR JIT
/// InProcess path in its process.
auto testing_force_shared_iree_state_cleanup() -> void;

/// Thrown when iree-run-module reports a runtime failure. Carries the full
/// stderr text for debugging.
class JitInvokeError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

}  // namespace tenzor::jit::mlir_jit
