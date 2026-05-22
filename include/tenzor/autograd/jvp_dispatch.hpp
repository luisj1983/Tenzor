/**
 * @file jvp_dispatch.hpp
 * @brief Forward-mode AD dispatch table: per-OpId JVP rules.
 *
 * Provides a registration mechanism for JVP (Jacobian-Vector Product) rules
 * indexed by OpId, mirroring the forward BackendDispatchTable. A JVP rule
 * takes the primal inputs and tangent inputs for an op plus its OpAttributes,
 * and produces the primal output together with the tangent output in a single
 * pass.
 *
 * Design:
 *  - Rule signature: `(primals, tangents, attrs) -> (primal_out, tangent_out)`
 *    All tensors are device-agnostic; rules internally call `tenzor::*`
 *    Tensor-level ops which route to the appropriate backend.
 *  - Registration is static / one-shot at startup, similar to the kernel
 *    registry. Use the `register_jvp_rule(OpId, JvpRuleFn)` API or the
 *    `TENZOR_REGISTER_JVP_RULE` macro for compile-time registration.
 *  - Lookup is an O(1) array index by OpId.
 *  - Coverage is partial: ops without a registered rule return false from
 *    `has_jvp_rule(op)`; callers can fall back to finite differences or
 *    raise an error.
 */

#pragma once

#include <span>
#include <utility>
#include <vector>

#include "../core/tensor.hpp"
#include "../ops/op_id.hpp"
#include "../backend/op_attributes.hpp"
#include "../backend/backend.hpp"  // pulls in `using OpAttributes = NewOpAttributes;`

namespace tenzor {

/**
 * @brief JVP rule output: primal value and tangent value.
 *
 * The first element is f(primal); the second is J_f(primal) * tangent where
 * J_f is the Jacobian of the op at the given primal inputs.
 */
struct JvpResult {
    Tensor primal;
    Tensor tangent;
};

/**
 * @brief Multi-output JVP rule result.
 *
 * For ops that return multiple tensors (e.g. BatchNorm2dForwardAffine returns
 * {output, mean, rstd}; LinalgEigh returns {eigenvalues, eigenvectors};
 * LayerNorm returns {output, mean, rstd}). `primals[i]` is f_i(primal) and
 * `tangents[i]` is J_{f_i}(primal) * tangent_in. The two vectors must have
 * identical length matching the op's output arity.
 */
struct JvpMultiResult {
    std::vector<Tensor> primals;
    std::vector<Tensor> tangents;
};

/**
 * @brief JVP rule function pointer.
 *
 * Takes parallel spans of primal and tangent input tensors plus the op's
 * attributes, returns the dual output. The two spans must have the same
 * length; entries in `tangents` may be undefined (default-constructed)
 * Tensors when the corresponding input has no associated tangent (treated
 * as zero tangent by the rule).
 */
using JvpRuleFn = JvpResult (*)(std::span<const Tensor> primals,
                                std::span<const Tensor> tangents,
                                const OpAttributes& attrs);

/**
 * @brief Multi-output JVP rule function pointer.
 *
 * Same input contract as `JvpRuleFn` but returns `JvpMultiResult` carrying
 * one (primal, tangent) pair per op output. Use this for ops whose forward
 * kernel returns multiple tensors (e.g. LayerNorm → {y, mean, rstd}).
 */
using JvpRuleFnMulti = JvpMultiResult (*)(std::span<const Tensor> primals,
                                          std::span<const Tensor> tangents,
                                          const OpAttributes& attrs);

/**
 * @brief Register a JVP rule for an OpId.
 *
 * Must be called during library/program startup, before any concurrent
 * dispatch_jvp() calls. Overwrites any prior rule with a warning.
 */
void register_jvp_rule(OpId op, JvpRuleFn fn);

/**
 * @brief Register a multi-output JVP rule for an OpId.
 *
 * Separate table from single-output rules. An OpId may have at most one
 * single-output rule and/or at most one multi-output rule registered; the
 * tables are independent so a single-output adapter does not displace a
 * multi-output rule for the same OpId (callers pick the entry point that
 * matches the op's output arity).
 */
void register_jvp_rule_multi(OpId op, JvpRuleFnMulti fn);

/**
 * @brief Check whether a JVP rule is registered for the given OpId.
 *
 * Returns true if a single-output rule is registered. For multi-output
 * coverage use `has_jvp_rule_multi`.
 */
[[nodiscard]] bool has_jvp_rule(OpId op) noexcept;

/**
 * @brief Check whether a multi-output JVP rule is registered for the OpId.
 */
[[nodiscard]] bool has_jvp_rule_multi(OpId op) noexcept;

/**
 * @brief Look up the registered JVP rule for an OpId.
 * @return Function pointer or nullptr if none registered.
 */
[[nodiscard]] JvpRuleFn get_jvp_rule(OpId op) noexcept;

/**
 * @brief Look up the registered multi-output JVP rule for an OpId.
 * @return Function pointer or nullptr if none registered.
 */
[[nodiscard]] JvpRuleFnMulti get_jvp_rule_multi(OpId op) noexcept;

/**
 * @brief Dispatch forward-mode AD for a single op.
 *
 * Looks up the JVP rule for `op` and invokes it with the supplied primals,
 * tangents, and attributes. Throws std::runtime_error if no rule is
 * registered for the OpId.
 *
 * @param op       Operation identifier.
 * @param primals  Primal input tensors.
 * @param tangents Tangent input tensors (same length as primals).
 * @param attrs    Op attributes (forwarded verbatim to the rule).
 * @return         {primal_out, tangent_out}
 */
[[nodiscard]] JvpResult dispatch_jvp(OpId op,
                                     std::span<const Tensor> primals,
                                     std::span<const Tensor> tangents,
                                     const OpAttributes& attrs = {});

/**
 * @brief Dispatch multi-output forward-mode AD for a single op.
 *
 * Looks up the multi-output JVP rule for `op` and invokes it. Throws
 * std::runtime_error if no multi-output rule is registered for the OpId.
 */
[[nodiscard]] JvpMultiResult dispatch_jvp_multi(OpId op,
                                                std::span<const Tensor> primals,
                                                std::span<const Tensor> tangents,
                                                const OpAttributes& attrs = {});

/**
 * @brief Trigger registration of all built-in JVP rules.
 *
 * Idempotent. Called automatically on first dispatch_jvp() or
 * has_jvp_rule() lookup; may be called explicitly during library init to
 * front-load the cost.
 */
void ensure_jvp_rules_registered();

} // namespace tenzor
