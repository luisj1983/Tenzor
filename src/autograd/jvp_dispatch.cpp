/**
 * @file jvp_dispatch.cpp
 * @brief JVP dispatch table implementation.
 */

#include "tenzor/autograd/jvp_dispatch.hpp"

#include <array>
#include <atomic>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>

namespace tenzor::detail {
/// Forward declaration: implemented in jvp_rules.cpp. The function
/// registers every built-in JVP rule by calling register_jvp_rule()
/// for each supported OpId. Invoked exactly once via std::call_once.
void register_builtin_jvp_rules();
} // namespace tenzor::detail

namespace tenzor {

namespace {

/// One-shot registry of JVP rules indexed by OpId.
struct JvpDispatchTable {
    std::array<JvpRuleFn, OP_COUNT> rules{};
};

JvpDispatchTable& jvp_table() noexcept {
    static JvpDispatchTable table;
    return table;
}

/// Idempotent flag for built-in registration.
std::once_flag& jvp_registration_flag() noexcept {
    static std::once_flag flag;
    return flag;
}

} // anonymous namespace

void register_jvp_rule(OpId op, JvpRuleFn fn) {
    auto idx = static_cast<size_t>(op);
    if (idx >= OP_COUNT) {
        throw std::out_of_range("register_jvp_rule: OpId out of range");
    }
    auto& slot = jvp_table().rules[idx];
    if (slot != nullptr && slot != fn) {
        std::fprintf(stderr,
                     "[jvp_dispatch] WARNING: overwriting JVP rule for OpId %zu\n",
                     idx);
    }
    slot = fn;
}

bool has_jvp_rule(OpId op) noexcept {
    ensure_jvp_rules_registered();
    auto idx = static_cast<size_t>(op);
    if (idx >= OP_COUNT) {
        return false;
    }
    return jvp_table().rules[idx] != nullptr;
}

JvpRuleFn get_jvp_rule(OpId op) noexcept {
    ensure_jvp_rules_registered();
    auto idx = static_cast<size_t>(op);
    if (idx >= OP_COUNT) {
        return nullptr;
    }
    return jvp_table().rules[idx];
}

JvpResult dispatch_jvp(OpId op,
                       std::span<const Tensor> primals,
                       std::span<const Tensor> tangents,
                       const OpAttributes& attrs) {
    if (primals.size() != tangents.size()) {
        throw std::runtime_error(
            "dispatch_jvp: primals and tangents must have the same length");
    }
    auto fn = get_jvp_rule(op);
    if (fn == nullptr) {
        std::string msg = "dispatch_jvp: no JVP rule registered for OpId ";
        msg += std::to_string(static_cast<int>(op));
        msg += " (";
        msg += op_id_to_name(op);
        msg += "). Forward-mode AD is not implemented for this op; "
               "fall back to backward-mode AD or finite differences.";
        throw std::runtime_error(std::move(msg));
    }
    return fn(primals, tangents, attrs);
}

void ensure_jvp_rules_registered() {
    std::call_once(jvp_registration_flag(), [] {
        detail::register_builtin_jvp_rules();
    });
}

} // namespace tenzor
