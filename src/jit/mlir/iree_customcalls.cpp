// Phase 13 / Task A.9 + Group D — Tenzor IREE plugin (in-process).
//
// Per the 2026-05-19 amendment, the 4 MVP-1 dialect ops (FlashAttention,
// GQA, RoPE, RMSNorm) are lowered to a `call @tenzor_plugin.<op>(...)`
// against a `func.func private` declaration in @main. At runtime an
// in-process IREE VM native module named `tenzor_plugin` exports those
// 4 functions; loading the bytecode module resolves the imports against
// it. Each callback unpacks the VM ABI arguments (HAL buffer_view refs +
// i32 scalars), copies the buffer_views to host-side ::tenzor::Tensors,
// runs the existing OpId kernel via `customcalls::dispatch_<op>`, and
// wraps the output Tensor in a fresh buffer_view ref for the VM to
// return.
//
// The dispatchers themselves are unchanged from Group D and remain
// unit-testable in isolation (Tensor → Tensor); the VM-side glue is the
// only new code path.

#include "tenzor/jit/mlir/iree_customcalls.hpp"

#include "_iree_marshal.hpp"
#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/transform.hpp"

#include <iree/runtime/api.h>
#include <iree/vm/api.h>
#include <iree/vm/native_module.h>
#include <iree/modules/hal/types.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace tenzor::jit::mlir_jit::customcalls {

namespace {

/// Parse a `backend_config` string of the form "k1=v1,k2=v2,..." into a
/// flat key→value map.
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
    float      scale  = 0.0f;
    if (kv.count("scale")) {
        // Caller supplied an explicit scale (including a legitimate 0).
        scale = parse_float(kv.at("scale"), 0.0f);
    } else {
        // Not supplied: default to 1/sqrt(D) over the last (head) dim.
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

auto dispatch_gqa(const std::vector<::tenzor::Tensor>& inputs,
                  const std::string& backend_config)
    -> ::tenzor::Tensor {
    if (inputs.size() != 3) {
        throw std::runtime_error(
            "tenzor_gqa: expected 3 inputs (Q, K, V), got " +
            std::to_string(inputs.size()));
    }
    const auto& q = inputs[0];
    const auto& k = inputs[1];
    const auto& v = inputs[2];
    if (q.shape().size() != 4 || k.shape().size() != 4 ||
        v.shape().size() != 4) {
        throw std::runtime_error("tenzor_gqa: requires 4-D Q/K/V (B,H,S,D)");
    }
    const int64_t B   = q.shape()[0];
    const int64_t Hq  = q.shape()[1];
    const int64_t Hkv = k.shape()[1];
    const int64_t Sk  = k.shape()[2];
    const int64_t D   = q.shape()[3];
    if (Hkv == 0 || Hq % Hkv != 0) {
        throw std::runtime_error("tenzor_gqa: requires H_kv | H_q");
    }
    const int64_t G = Hq / Hkv;

    auto repeat_kv = [&](const ::tenzor::Tensor& x) -> ::tenzor::Tensor {
        if (Hq == Hkv) return x;  // MHA degenerate case
        auto u   = ::tenzor::unsqueeze(x, 2);          // (B,Hkv,1,Sk,D)
        auto e   = ::tenzor::expand(u, {B, Hkv, G, Sk, D});
        return ::tenzor::reshape(e, {B, Hq, Sk, D});
    };
    auto k_full = repeat_kv(k);
    auto v_full = repeat_kv(v);

    return dispatch_flash_attention({q, k_full, v_full}, backend_config);
}

auto dispatch_rope_apply(const std::vector<::tenzor::Tensor>& inputs,
                         const std::string& backend_config)
    -> ::tenzor::Tensor {
    if (inputs.size() != 3) {
        throw std::runtime_error(
            "tenzor_rope_apply: expected 3 inputs (x, cos, sin), got " +
            std::to_string(inputs.size()));
    }
    const auto& x   = inputs[0];
    const auto& cos = inputs[1];
    const auto& sin = inputs[2];
    const auto x_shape = x.shape();
    const int64_t rank = static_cast<int64_t>(x_shape.size());
    if (rank < 1) {
        throw std::runtime_error("tenzor_rope_apply: x must have rank >= 1");
    }
    const int64_t D = x_shape[rank - 1];
    if (D % 2 != 0) {
        throw std::runtime_error(
            "tenzor_rope_apply: last dim of x must be even");
    }
    const int64_t H = D / 2;
    const int64_t last_dim = rank - 1;

    const auto kv = parse_backend_config(backend_config);
    (void)parse_int(kv.count("offset") ? kv.at("offset") : "", 0);

    auto x1 = ::tenzor::narrow(x, last_dim, 0, H).contiguous();
    auto x2 = ::tenzor::narrow(x, last_dim, H, H).contiguous();
    auto neg_x2 = ::tenzor::neg(x2);
    auto rotated = ::tenzor::cat({neg_x2, x1}, last_dim);

    auto align_trailing = [&](const ::tenzor::Tensor& t) -> ::tenzor::Tensor {
        const auto sh = t.shape();
        if (static_cast<int64_t>(sh.size()) == rank) return t;
        ::tenzor::Tensor out = t;
        while (static_cast<int64_t>(out.shape().size()) < rank) {
            out = ::tenzor::unsqueeze(out, 0);
        }
        return out;
    };
    auto cos_b = align_trailing(cos);
    auto sin_b = align_trailing(sin);

    return x * cos_b + rotated * sin_b;
}

auto dispatch_rms_norm(const std::vector<::tenzor::Tensor>& inputs,
                       const std::string& backend_config)
    -> ::tenzor::Tensor {
    if (inputs.empty()) {
        throw std::runtime_error(
            "tenzor_rms_norm: expected 1+ inputs (x[, weight])");
    }
    const auto kv  = parse_backend_config(backend_config);
    const float eps = parse_float(kv.count("eps") ? kv.at("eps") : "", 1e-6f);

    std::vector<::tenzor::Tensor> in_with_weight;
    in_with_weight.reserve(2);
    in_with_weight.push_back(inputs[0]);
    if (inputs.size() >= 2) {
        in_with_weight.push_back(inputs[1]);
    } else {
        const auto x_shape = inputs[0].shape();
        if (x_shape.empty()) {
            throw std::runtime_error(
                "tenzor_rms_norm: x must have rank >= 1");
        }
        const int64_t D = x_shape.back();
        in_with_weight.push_back(::tenzor::full({D}, 1.0f, inputs[0].dtype(),
                                                inputs[0].device()));
    }

    ::tenzor::OpAttributes attrs;
    attrs.set(::tenzor::AttrKey::Eps, static_cast<double>(eps));

    auto outs = ::tenzor::dispatch<::tenzor::OpId::RMSNorm>(in_with_weight,
                                                            attrs);
    if (outs.empty()) {
        throw std::runtime_error(
            "tenzor_rms_norm: dispatch returned no outputs");
    }
    return outs[0];
}

}  // namespace tenzor::jit::mlir_jit::customcalls

// ─── Plugin VM native module ──────────────────────────────────────────────
//
// Module name: `tenzor_plugin`. Exports (matching the MLIR-side
// `call @tenzor_plugin.<op>` sites):
//   - flash_attention(Q, K, V, scale_bits: i32, causal: i32) -> Out
//   - gqa            (Q, K, V, scale_bits: i32, causal: i32) -> Out
//   - rope_apply    (x, cos, sin) -> Out
//   - rms_norm      (x, weight, eps_bits: i32) -> Out
//
// The IREE VM calling convention chars only support i/I/r — there's no
// 'f' for float — so scalar f32 attrs travel as i32 bit-pattern args
// (reinterpret_cast<float>(bits)). The lowering side encodes the constants
// the same way.

namespace tenzor::jit::mlir_jit {

namespace plugin {

namespace cc = ::tenzor::jit::mlir_jit::customcalls;
namespace marshal = ::tenzor::jit::mlir_jit::marshal;

// Module-instance state. The module is stateless across contexts, but the
// IREE VM still allocates a per-context state object — we use it solely so
// the module integrates with VM lifecycle hooks.
struct PluginState {
    iree_allocator_t allocator;
};

extern "C" {

static void IREE_API_PTR plugin_destroy(void* /*self*/) {
    // Module-level "self" is null (we pass nullptr to iree_vm_module_initialize)
    // so nothing to free here. Per-context state is freed via plugin_free_state.
}

static iree_status_t IREE_API_PTR
plugin_alloc_state(void* /*self*/, iree_allocator_t allocator,
                   iree_vm_module_state_t** out_state) {
    PluginState* state = nullptr;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        allocator, sizeof(PluginState), reinterpret_cast<void**>(&state)));
    std::memset(state, 0, sizeof(*state));
    state->allocator = allocator;
    *out_state = reinterpret_cast<iree_vm_module_state_t*>(state);
    return iree_ok_status();
}

static void IREE_API_PTR
plugin_free_state(void* /*self*/, iree_vm_module_state_t* module_state) {
    if (!module_state) return;
    PluginState* state = reinterpret_cast<PluginState*>(module_state);
    iree_allocator_free(state->allocator, state);
}

}  // extern "C"

// Helper: extract HAL buffer_view ref from a vm.ref slot.
static iree_status_t deref_buffer_view(const iree_vm_ref_t& ref,
                                       iree_hal_buffer_view_t** out_bv) {
    return iree_hal_buffer_view_check_deref(ref, out_bv);
}

// Helper: convert a tensor result into a buffer_view ref to be returned.
// The active device for this call is captured from `template_view`'s
// underlying allocation placement — all inputs to a call share the same
// device, so any one of them suffices.
// Helper: convert a tensor result into a buffer_view ref to be returned.
// The active device for this call is passed in explicitly — it is recovered
// by each shim from the `module` parameter (set by
// create_tenzor_plugin_module via interface.self).
static iree_status_t tensor_to_result_ref(iree_hal_device_t* device,
                                          const ::tenzor::Tensor& out,
                                          iree_vm_ref_t* out_ref) {
    if (!device) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
            "tenzor_plugin: no active HAL device for result marshalling");
    }
    iree_hal_allocator_t* allocator = iree_hal_device_allocator(device);
    if (!allocator) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
            "tenzor_plugin: HAL device has no allocator");
    }

    const ::tenzor::Tensor cpu = out.cpu().contiguous();
    std::vector<iree_hal_dim_t> dims;
    dims.reserve(cpu.shape().size());
    for (auto d : cpu.shape()) {
        dims.push_back(static_cast<iree_hal_dim_t>(d));
    }
    iree_hal_element_type_t etype = marshal::dtype_to_iree(cpu.dtype());
    iree_hal_buffer_params_t params = {};
    params.usage  = IREE_HAL_BUFFER_USAGE_DEFAULT;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.type   = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;

    const std::size_t bytes =
        static_cast<std::size_t>(cpu.numel()) * cpu.element_size();
    iree_hal_buffer_view_t* result_bv = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer_copy(
        device, allocator, dims.size(), dims.data(), etype,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, params,
        iree_make_const_byte_span(cpu.data_ptr(), bytes), &result_bv));

    *out_ref = iree_hal_buffer_view_move_ref(result_bv);
    return iree_ok_status();
}

// ─── Shims for the 4 ops ─────────────────────────────────────────────────

// flash_attention(Q, K, V, scale_bits: i32, causal: i32) -> Out
// CC: "0rrrii_r"
struct FlashAttentionArgs {
    iree_vm_ref_t q;
    iree_vm_ref_t k;
    iree_vm_ref_t v;
    int32_t       scale_bits;
    int32_t       causal;
};
struct OneRefResult {
    iree_vm_ref_t r0;
};

extern "C" {

static iree_status_t IREE_API_PTR call_shim_flash_attention(
    iree_vm_stack_t* /*stack*/,
    iree_vm_native_function_flags_t /*flags*/,
    iree_byte_span_t args_storage,
    iree_byte_span_t rets_storage,
    iree_vm_native_function_target_t /*target_fn*/,
    void* module,
    void* /*module_state*/) {
    auto* device = reinterpret_cast<iree_hal_device_t*>(module);
    const auto* args = reinterpret_cast<const FlashAttentionArgs*>(args_storage.data);
    auto* rets       = reinterpret_cast<OneRefResult*>(rets_storage.data);

    iree_hal_buffer_view_t *q_bv = nullptr, *k_bv = nullptr, *v_bv = nullptr;
    IREE_RETURN_IF_ERROR(deref_buffer_view(args->q, &q_bv));
    IREE_RETURN_IF_ERROR(deref_buffer_view(args->k, &k_bv));
    IREE_RETURN_IF_ERROR(deref_buffer_view(args->v, &v_bv));

    try {
        auto qt = marshal::buffer_view_to_tensor(q_bv);
        auto kt = marshal::buffer_view_to_tensor(k_bv);
        auto vt = marshal::buffer_view_to_tensor(v_bv);

        float scale;
        std::memcpy(&scale, &args->scale_bits, sizeof(scale));
        const bool causal = args->causal != 0;

        std::ostringstream cfg;
        cfg << "causal=" << (causal ? "true" : "false")
            << ",scale=" << scale;
        auto out = cc::dispatch_flash_attention({qt, kt, vt}, cfg.str());

        std::memset(&rets->r0, 0, sizeof(rets->r0));
        IREE_RETURN_IF_ERROR(tensor_to_result_ref(device, out, &rets->r0));
    } catch (const std::exception& e) {
        return iree_make_status(IREE_STATUS_INTERNAL,
                                "tenzor_plugin.flash_attention: %s", e.what());
    }
    return iree_ok_status();
}

// gqa(Q, K, V, scale_bits: i32, causal: i32) -> Out, same cc as flash_attention.
static iree_status_t IREE_API_PTR call_shim_gqa(
    iree_vm_stack_t* /*stack*/,
    iree_vm_native_function_flags_t /*flags*/,
    iree_byte_span_t args_storage,
    iree_byte_span_t rets_storage,
    iree_vm_native_function_target_t /*target_fn*/,
    void* module,
    void* /*module_state*/) {
    auto* device = reinterpret_cast<iree_hal_device_t*>(module);
    const auto* args = reinterpret_cast<const FlashAttentionArgs*>(args_storage.data);
    auto* rets       = reinterpret_cast<OneRefResult*>(rets_storage.data);

    iree_hal_buffer_view_t *q_bv = nullptr, *k_bv = nullptr, *v_bv = nullptr;
    IREE_RETURN_IF_ERROR(deref_buffer_view(args->q, &q_bv));
    IREE_RETURN_IF_ERROR(deref_buffer_view(args->k, &k_bv));
    IREE_RETURN_IF_ERROR(deref_buffer_view(args->v, &v_bv));

    try {
        auto qt = marshal::buffer_view_to_tensor(q_bv);
        auto kt = marshal::buffer_view_to_tensor(k_bv);
        auto vt = marshal::buffer_view_to_tensor(v_bv);

        float scale;
        std::memcpy(&scale, &args->scale_bits, sizeof(scale));
        const bool causal = args->causal != 0;

        std::ostringstream cfg;
        cfg << "causal=" << (causal ? "true" : "false")
            << ",scale=" << scale;
        auto out = cc::dispatch_gqa({qt, kt, vt}, cfg.str());

        std::memset(&rets->r0, 0, sizeof(rets->r0));
        IREE_RETURN_IF_ERROR(tensor_to_result_ref(device, out, &rets->r0));
    } catch (const std::exception& e) {
        return iree_make_status(IREE_STATUS_INTERNAL,
                                "tenzor_plugin.gqa: %s", e.what());
    }
    return iree_ok_status();
}

// rope_apply(x, cos, sin) -> Out
// CC: "0rrr_r"
struct RopeArgs {
    iree_vm_ref_t x;
    iree_vm_ref_t cos;
    iree_vm_ref_t sin;
};

static iree_status_t IREE_API_PTR call_shim_rope_apply(
    iree_vm_stack_t* /*stack*/,
    iree_vm_native_function_flags_t /*flags*/,
    iree_byte_span_t args_storage,
    iree_byte_span_t rets_storage,
    iree_vm_native_function_target_t /*target_fn*/,
    void* module,
    void* /*module_state*/) {
    auto* device = reinterpret_cast<iree_hal_device_t*>(module);
    const auto* args = reinterpret_cast<const RopeArgs*>(args_storage.data);
    auto* rets       = reinterpret_cast<OneRefResult*>(rets_storage.data);

    iree_hal_buffer_view_t *x_bv = nullptr, *cos_bv = nullptr, *sin_bv = nullptr;
    IREE_RETURN_IF_ERROR(deref_buffer_view(args->x,   &x_bv));
    IREE_RETURN_IF_ERROR(deref_buffer_view(args->cos, &cos_bv));
    IREE_RETURN_IF_ERROR(deref_buffer_view(args->sin, &sin_bv));

    try {
        auto xt   = marshal::buffer_view_to_tensor(x_bv);
        auto cost = marshal::buffer_view_to_tensor(cos_bv);
        auto sint = marshal::buffer_view_to_tensor(sin_bv);
        auto out  = cc::dispatch_rope_apply({xt, cost, sint}, "offset=0");

        std::memset(&rets->r0, 0, sizeof(rets->r0));
        IREE_RETURN_IF_ERROR(tensor_to_result_ref(device, out, &rets->r0));
    } catch (const std::exception& e) {
        return iree_make_status(IREE_STATUS_INTERNAL,
                                "tenzor_plugin.rope_apply: %s", e.what());
    }
    return iree_ok_status();
}

// rms_norm(x, weight, eps_bits: i32) -> Out
// CC: "0rri_r"
struct RmsNormArgs {
    iree_vm_ref_t x;
    iree_vm_ref_t weight;
    int32_t       eps_bits;
};

static iree_status_t IREE_API_PTR call_shim_rms_norm(
    iree_vm_stack_t* /*stack*/,
    iree_vm_native_function_flags_t /*flags*/,
    iree_byte_span_t args_storage,
    iree_byte_span_t rets_storage,
    iree_vm_native_function_target_t /*target_fn*/,
    void* module,
    void* /*module_state*/) {
    auto* device = reinterpret_cast<iree_hal_device_t*>(module);
    const auto* args = reinterpret_cast<const RmsNormArgs*>(args_storage.data);
    auto* rets       = reinterpret_cast<OneRefResult*>(rets_storage.data);

    iree_hal_buffer_view_t *x_bv = nullptr, *w_bv = nullptr;
    IREE_RETURN_IF_ERROR(deref_buffer_view(args->x,      &x_bv));
    IREE_RETURN_IF_ERROR(deref_buffer_view(args->weight, &w_bv));

    try {
        auto xt = marshal::buffer_view_to_tensor(x_bv);
        auto wt = marshal::buffer_view_to_tensor(w_bv);

        float eps;
        std::memcpy(&eps, &args->eps_bits, sizeof(eps));

        std::ostringstream cfg;
        cfg << "eps=" << eps;
        auto out = cc::dispatch_rms_norm({xt, wt}, cfg.str());

        std::memset(&rets->r0, 0, sizeof(rets->r0));
        IREE_RETURN_IF_ERROR(tensor_to_result_ref(device, out, &rets->r0));
    } catch (const std::exception& e) {
        return iree_make_status(IREE_STATUS_INTERNAL,
                                "tenzor_plugin.rms_norm: %s", e.what());
    }
    return iree_ok_status();
}

}  // extern "C"

// ─── Module descriptor tables ────────────────────────────────────────────

static const iree_vm_native_export_descriptor_t kExports[] = {
    {IREE_SV("flash_attention"), IREE_SV("0rrrii_r"), 0, nullptr},
    {IREE_SV("gqa"),             IREE_SV("0rrrii_r"), 0, nullptr},
    {IREE_SV("rms_norm"),        IREE_SV("0rri_r"),   0, nullptr},
    {IREE_SV("rope_apply"),      IREE_SV("0rrr_r"),   0, nullptr},
};

// Exports must be in lexicographic sorted order (per
// iree_string_view_compare). Above: flash_attention < gqa < rms_norm <
// rope_apply.

static const iree_vm_native_function_ptr_t kFunctions[] = {
    {call_shim_flash_attention, nullptr},
    {call_shim_gqa,             nullptr},
    {call_shim_rms_norm,        nullptr},
    {call_shim_rope_apply,      nullptr},
};

static_assert(IREE_ARRAYSIZE(kExports) == IREE_ARRAYSIZE(kFunctions),
              "tenzor_plugin: exports and functions tables must match 1:1");

static const iree_vm_native_module_descriptor_t kDescriptor = {
    /*name=*/IREE_SV("tenzor_plugin"),
    /*version=*/0,
    /*attr_count=*/0,
    /*attrs=*/nullptr,
    /*dependency_count=*/0,
    /*dependencies=*/nullptr,
    /*import_count=*/0,
    /*imports=*/nullptr,
    /*export_count=*/IREE_ARRAYSIZE(kExports),
    /*exports=*/kExports,
    /*function_count=*/IREE_ARRAYSIZE(kFunctions),
    /*functions=*/kFunctions,
};

}  // namespace plugin

/// Create the tenzor_plugin VM native module to be appended to the runtime
/// session before loading the bytecode .vmfb. The returned module reference
/// is owned by the caller and must be released via iree_vm_module_release()
/// after the session takes its own reference.
auto create_tenzor_plugin_module(iree_vm_instance_t* instance,
                                 iree_allocator_t allocator,
                                 iree_hal_device_t* device,
                                 iree_vm_module_t** out_module)
    -> iree_status_t {
    iree_vm_module_t interface;
    IREE_RETURN_IF_ERROR(iree_vm_module_initialize(&interface, nullptr));
    // Stash the device on the module so each shim can recover it as the
    // `module` parameter to `iree_vm_native_function_shim_t` (IREE assigns
    // `module->self = user_interface.self` at create time).
    interface.self        = device;
    interface.destroy     = plugin::plugin_destroy;
    interface.alloc_state = plugin::plugin_alloc_state;
    interface.free_state  = plugin::plugin_free_state;
    return iree_vm_native_module_create(
        &interface, &plugin::kDescriptor, instance, allocator, out_module);
}

// Audit item I.6: placeholder_messages namespace deleted along with the
// smoke test that exercised it.

}  // namespace tenzor::jit::mlir_jit
