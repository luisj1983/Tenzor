// Phase 13 / Task A.9 + Group D — IREE custom_call registrar with per-op
// dispatchers.
//
// Per the 2026-05-19 amendment, the 4 MVP-1 dialect ops become
// `stablehlo.custom_call @tenzor_<name>` strings in the emitted MLIR text.
// At runtime the Tenzor plugin resolves each name to one of the four
// `customcalls::dispatch_<op>` helper functions below, which take a
// vector of Tensors plus the `backend_config` string and dispatch to the
// existing OpId kernel.
//
// Environmental note (matching iree_runtime.cpp): the IREE distribution at
// /home/lee/iree-dist does not ship the iree/runtime/api.h companion headers
// (iree/base/allocator.h, hal/buffer.h, …) that the C API requires, so we
// cannot #include <iree/runtime/api.h> here. The file therefore declares
// only the *dispatchers* — the C glue that reads
// iree_hal_buffer_view_t inputs, calls a dispatcher, and writes the output
// back into iree_hal_buffer_view_t, will be added once the headers are
// restored. The dispatchers are unit-testable today (and covered by the
// per-op end-to-end-from-tensors tests in tests/jit/mlir/).

#include "tenzor/jit/mlir/iree_customcalls.hpp"

#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/op_id.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

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

namespace tenzor::jit::mlir_jit::customcalls {

namespace {

/// Parse a `backend_config` string of the form "k1=v1,k2=v2,..." into a
/// flat key→value map. Both keys and values are returned as string_views
/// of the original buffer-backed text (caller keeps the source string
/// alive while the map is used).
auto parse_backend_config(const std::string& cfg)
    -> std::unordered_map<std::string, std::string> {
    std::unordered_map<std::string, std::string> kv;
    std::string current;
    for (char c : cfg) {
        if (c == ',') {
            if (!current.empty()) {
                const auto eq = current.find('=');
                if (eq != std::string::npos) {
                    kv[current.substr(0, eq)] = current.substr(eq + 1);
                }
                current.clear();
            }
        } else {
            current.push_back(c);
        }
    }
    if (!current.empty()) {
        const auto eq = current.find('=');
        if (eq != std::string::npos) {
            kv[current.substr(0, eq)] = current.substr(eq + 1);
        }
    }
    return kv;
}

auto parse_bool(const std::string& s, bool dflt) -> bool {
    if (s == "true" || s == "1")  return true;
    if (s == "false" || s == "0") return false;
    return dflt;
}

auto parse_float(const std::string& s, float dflt) -> float {
    if (s.empty()) return dflt;
    try {
        return std::stof(s);
    } catch (...) {
        return dflt;
    }
}

auto parse_int(const std::string& s, int64_t dflt) -> int64_t {
    if (s.empty()) return dflt;
    try {
        return static_cast<int64_t>(std::stoll(s));
    } catch (...) {
        return dflt;
    }
}

}  // namespace

auto dispatch_flash_attention(const std::vector<::tenzor::Tensor>& inputs,
                              const std::string& backend_config)
    -> ::tenzor::Tensor {
    if (inputs.size() != 3) {
        throw std::runtime_error(
            "tenzor_flash_attention: expected 3 inputs (Q, K, V), got " +
            std::to_string(inputs.size()));
    }
    const auto kv     = parse_backend_config(backend_config);
    const bool causal = parse_bool (kv.count("causal") ? kv.at("causal") : "",
                                    false);
    float      scale  = parse_float(kv.count("scale")  ? kv.at("scale")  : "",
                                    0.0f);
    if (scale == 0.0f) {
        const auto& q     = inputs[0];
        const auto rank   = q.shape().size();
        if (rank >= 1) {
            const auto D = q.shape()[rank - 1];
            scale = 1.0f / std::sqrt(static_cast<float>(D));
        }
    }

    ::tenzor::OpAttributes attrs;
    attrs.set(::tenzor::AttrKey::Scale,    static_cast<double>(scale));
    attrs.set(::tenzor::AttrKey::Causal,   causal);
    attrs.set(::tenzor::AttrKey::DropoutP, 0.0);
    attrs.set(::tenzor::AttrKey::IsTraining, false);

    auto outs = ::tenzor::dispatch<::tenzor::OpId::FlashAttention>(
        inputs, attrs);
    if (outs.empty()) {
        throw std::runtime_error(
            "tenzor_flash_attention: dispatch returned no outputs");
    }
    return outs[0];
}

// The remaining 3 dispatchers (gqa, rope_apply, rms_norm) are filled in by
// Group D.2.2 / D.3.2 / D.4.2. Until then they throw a clear error so
// callers (and the IREE-side glue) get an informative failure.
auto dispatch_gqa(const std::vector<::tenzor::Tensor>& /*inputs*/,
                  const std::string& /*backend_config*/)
    -> ::tenzor::Tensor {
    throw std::runtime_error(
        "tenzor_gqa: dispatcher not yet implemented (Group D.2.2)");
}

auto dispatch_rope_apply(const std::vector<::tenzor::Tensor>& /*inputs*/,
                         const std::string& /*backend_config*/)
    -> ::tenzor::Tensor {
    throw std::runtime_error(
        "tenzor_rope_apply: dispatcher not yet implemented (Group D.3.2)");
}

auto dispatch_rms_norm(const std::vector<::tenzor::Tensor>& /*inputs*/,
                       const std::string& /*backend_config*/)
    -> ::tenzor::Tensor {
    throw std::runtime_error(
        "tenzor_rms_norm: dispatcher not yet implemented (Group D.4.2)");
}

}  // namespace tenzor::jit::mlir_jit::customcalls

namespace tenzor::jit::mlir_jit {

namespace {

constexpr const char* kMessage_FlashAttention =
    "tenzor_flash_attention: dispatcher wired (Group D.1.2); IREE-side "
    "runtime binding pending iree/runtime/api.h restoration in the dist";
constexpr const char* kMessage_GQA =
    "tenzor_gqa: dispatcher pending (Group D.2.2); placeholder text emitted "
    "while the iree/runtime/api.h headers remain incomplete";
constexpr const char* kMessage_RopeApply =
    "tenzor_rope_apply: dispatcher pending (Group D.3.2); placeholder text "
    "emitted while the iree/runtime/api.h headers remain incomplete";
constexpr const char* kMessage_RmsNorm =
    "tenzor_rms_norm: dispatcher pending (Group D.4.2); placeholder text "
    "emitted while the iree/runtime/api.h headers remain incomplete";

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
    // Real implementation will read inputs from `ctx` via the IREE runtime
    // call API, call customcalls::dispatch_flash_attention, and write the
    // output buffer view back. Until iree/runtime/api.h is restored in the
    // dist this is a stub that returns IREE_STATUS_UNIMPLEMENTED.
    return reinterpret_cast<iree_status_t>(
        kFakeUnimplementedStatus |
        reinterpret_cast<std::uintptr_t>(kMessage_FlashAttention));
}

iree_status_t tenzor_gqa_callback(iree_runtime_call_context_t* /*ctx*/) {
    return reinterpret_cast<iree_status_t>(
        kFakeUnimplementedStatus |
        reinterpret_cast<std::uintptr_t>(kMessage_GQA));
}

iree_status_t tenzor_rope_apply_callback(
    iree_runtime_call_context_t* /*ctx*/) {
    return reinterpret_cast<iree_status_t>(
        kFakeUnimplementedStatus |
        reinterpret_cast<std::uintptr_t>(kMessage_RopeApply));
}

iree_status_t tenzor_rms_norm_callback(iree_runtime_call_context_t* /*ctx*/) {
    return reinterpret_cast<iree_status_t>(
        kFakeUnimplementedStatus |
        reinterpret_cast<std::uintptr_t>(kMessage_RmsNorm));
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
auto flash_attention() -> const char* { return kMessage_FlashAttention; }
auto gqa() -> const char* { return kMessage_GQA; }
auto rope_apply() -> const char* { return kMessage_RopeApply; }
auto rms_norm() -> const char* { return kMessage_RmsNorm; }
}  // namespace placeholder_messages

}  // namespace tenzor::jit::mlir_jit
