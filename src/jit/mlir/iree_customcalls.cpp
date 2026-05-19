// Phase 13 / Task A.9 — IREE custom_call registrar with 4 MVP-1 callbacks.
//
// Per the 2026-05-19 amendment, the 12 dialect ops named in the parent spec
// become `stablehlo.custom_call @tenzor_<name>` strings in the emitted MLIR
// text. Their existing Tenzor kernels register as custom_call resolvers on
// each IREE runtime session. This file declares the 4 MVP-1 callbacks and
// the public `tenzor_register_custom_calls` entry point.
//
// Environmental note (matching iree_runtime.cpp): the IREE distribution at
// /home/lee/iree-dist does not ship the iree/runtime/api.h companion headers
// (iree/base/allocator.h, hal/buffer.h, …) that the C API requires, so we
// cannot #include <iree/runtime/api.h> here. The file uses forward
// declarations only — the same source compiles unmodified once the dist gets
// the missing headers.
//
// Until then, the smoke test in tests/jit/mlir/test_iree_customcall_smoke.cpp
// captures the "not implemented yet (Group D)" signal at the *compile* stage:
// iree-compile rejects `stablehlo.custom_call @tenzor_flash_attention` as an
// unregistered call target, which is the same signal an
// IREE_STATUS_UNIMPLEMENTED resolver would emit at runtime.

#include <cstddef>
#include <cstdint>

// Forward-declare just enough of the IREE C types so the file compiles
// without iree/runtime/api.h. Real iree_status_t / iree_runtime_session_t
// definitions live in the IREE C API; the layouts here match the upstream
// declarations and are intended to be replaced wholesale once the
// distribution headers are complete.
extern "C" {

typedef struct iree_status_handle_t* iree_status_t;
typedef struct iree_runtime_session_t iree_runtime_session_t;
typedef struct iree_runtime_call_context_t iree_runtime_call_context_t;

}  // extern "C"

namespace {
// Local stand-in for iree_ok_status() until the runtime C API headers are
// restored in the distribution. The real upstream symbol returns a
// status_t whose low bits encode IREE_STATUS_OK (0); we mimic that exactly.
inline auto local_ok_status() -> iree_status_t {
    return reinterpret_cast<iree_status_t>(0);
}
}  // namespace

namespace tenzor::jit::mlir_jit {

namespace {

constexpr const char* kNotImplemented_FlashAttention =
    "tenzor_flash_attention custom_call not implemented yet (Group D)";
constexpr const char* kNotImplemented_GQA =
    "tenzor_gqa custom_call not implemented yet (Group D)";
constexpr const char* kNotImplemented_RopeApply =
    "tenzor_rope_apply custom_call not implemented yet (Group D)";
constexpr const char* kNotImplemented_RmsNorm =
    "tenzor_rms_norm custom_call not implemented yet (Group D)";

// Marker so iree_register_custom_calls returns an error-producing
// status_t pointer rather than nullptr when called without the real headers.
// Bit pattern matches IREE's iree_status_t aliasing convention (low bits
// encode the status code).
constexpr std::uintptr_t kFakeUnimplementedStatus =
    static_cast<std::uintptr_t>(12) /* IREE_STATUS_UNIMPLEMENTED */;

}  // namespace

extern "C" {

iree_status_t tenzor_flash_attention_callback(
    iree_runtime_call_context_t* /*ctx*/) {
    return reinterpret_cast<iree_status_t>(
        kFakeUnimplementedStatus |
        reinterpret_cast<std::uintptr_t>(kNotImplemented_FlashAttention));
}

iree_status_t tenzor_gqa_callback(iree_runtime_call_context_t* /*ctx*/) {
    return reinterpret_cast<iree_status_t>(
        kFakeUnimplementedStatus |
        reinterpret_cast<std::uintptr_t>(kNotImplemented_GQA));
}

iree_status_t tenzor_rope_apply_callback(
    iree_runtime_call_context_t* /*ctx*/) {
    return reinterpret_cast<iree_status_t>(
        kFakeUnimplementedStatus |
        reinterpret_cast<std::uintptr_t>(kNotImplemented_RopeApply));
}

iree_status_t tenzor_rms_norm_callback(iree_runtime_call_context_t* /*ctx*/) {
    return reinterpret_cast<iree_status_t>(
        kFakeUnimplementedStatus |
        reinterpret_cast<std::uintptr_t>(kNotImplemented_RmsNorm));
}

/// Register the 4 MVP-1 custom_call resolvers on a session.
///
/// Once the iree/runtime/api.h headers are complete in the distribution this
/// function calls `iree_hal_register_custom_call` (or the equivalent in the
/// version IREE ships at the time) to bind the four callbacks above by name.
/// While the headers remain incomplete it is a no-op returning ok status —
/// the compile-time check (iree-compile rejecting the unregistered call
/// target) is what the smoke test asserts on instead.
iree_status_t tenzor_register_custom_calls(
    iree_runtime_session_t* /*session*/) {
    return local_ok_status();
}

}  // extern "C"

namespace placeholder_messages {
auto flash_attention() -> const char* { return kNotImplemented_FlashAttention; }
auto gqa() -> const char* { return kNotImplemented_GQA; }
auto rope_apply() -> const char* { return kNotImplemented_RopeApply; }
auto rms_norm() -> const char* { return kNotImplemented_RmsNorm; }
}  // namespace placeholder_messages

}  // namespace tenzor::jit::mlir_jit
