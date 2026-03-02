/**
 * @file kernel_registry.hpp
 * @brief Macros and utilities for kernel registration
 *
 * Provides convenient macros for registering kernel implementations with
 * the dispatch table. Handles common patterns like unary/binary operations.
 */

#pragma once

#include <span>
#include <vector>
#include <string>
#include <stdexcept>
#include <climits>
#include "../core/tensor.hpp"
#include "../ops/op_id.hpp"
#include "dispatch_table.hpp"

namespace tenzor {

// ============================================================================
// Registration Macros
// ============================================================================

/**
 * @brief Register a kernel with custom wrapper.
 *
 * @param table BackendDispatchTable reference
 * @param op_id OpId enum value (without OpId:: prefix)
 * @param kernel_fn Kernel function or lambda
 */
#define TENZOR_REGISTER_KERNEL(table, op_id, kernel_fn) \
    (table).register_kernel(::tenzor::OpId::op_id, (kernel_fn))

/**
 * @brief Register a unary operation kernel.
 *
 * For kernels with signature: Tensor(const Tensor&)
 *
 * @param table BackendDispatchTable reference
 * @param op_id OpId enum value (without OpId:: prefix)
 * @param kernel_fn Unary kernel function
 */
#define TENZOR_REGISTER_UNARY_KERNEL(table, op_id, kernel_fn) \
    (table).register_kernel(::tenzor::OpId::op_id, \
        [](std::span<const ::tenzor::Tensor> inputs, const ::tenzor::OpAttributes&) \
            -> std::vector<::tenzor::Tensor> { \
            return {(kernel_fn)(inputs[0])}; \
        })

/**
 * @brief Register a binary operation kernel.
 *
 * For kernels with signature: Tensor(const Tensor&, const Tensor&)
 *
 * @param table BackendDispatchTable reference
 * @param op_id OpId enum value (without OpId:: prefix)
 * @param kernel_fn Binary kernel function
 */
#define TENZOR_REGISTER_BINARY_KERNEL(table, op_id, kernel_fn) \
    (table).register_kernel(::tenzor::OpId::op_id, \
        [](std::span<const ::tenzor::Tensor> inputs, const ::tenzor::OpAttributes&) \
            -> std::vector<::tenzor::Tensor> { \
            return {(kernel_fn)(inputs[0], inputs[1])}; \
        })

/**
 * @brief Register a unary operation as single-output kernel.
 * Avoids vector<Tensor> allocation overhead for ops returning one Tensor.
 */
#define TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, op_id, kernel_fn) \
    (table).register_single_output_kernel(::tenzor::OpId::op_id, \
        [](std::span<const ::tenzor::Tensor> inputs, const ::tenzor::OpAttributes&) \
            -> ::tenzor::Tensor { \
            return (kernel_fn)(inputs[0]); \
        })

/**
 * @brief Register a binary operation as single-output kernel.
 * Avoids vector<Tensor> allocation overhead for ops returning one Tensor.
 */
#define TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, op_id, kernel_fn) \
    (table).register_single_output_kernel(::tenzor::OpId::op_id, \
        [](std::span<const ::tenzor::Tensor> inputs, const ::tenzor::OpAttributes&) \
            -> ::tenzor::Tensor { \
            return (kernel_fn)(inputs[0], inputs[1]); \
        })

/**
 * @brief Register a ternary operation kernel.
 *
 * For kernels with signature: Tensor(const Tensor&, const Tensor&, const Tensor&)
 *
 * @param table BackendDispatchTable reference
 * @param op_id OpId enum value (without OpId:: prefix)
 * @param kernel_fn Ternary kernel function
 */
#define TENZOR_REGISTER_TERNARY_KERNEL(table, op_id, kernel_fn) \
    (table).register_kernel(::tenzor::OpId::op_id, \
        [](std::span<const ::tenzor::Tensor> inputs, const ::tenzor::OpAttributes&) \
            -> std::vector<::tenzor::Tensor> { \
            return {(kernel_fn)(inputs[0], inputs[1], inputs[2])}; \
        })

/**
 * @brief Register an inplace operation kernel.
 *
 * For kernels with signature: void(Tensor&, const Tensor&)
 * Returns the modified first input.
 *
 * @param table BackendDispatchTable reference
 * @param op_id OpId enum value (without OpId:: prefix)
 * @param kernel_fn Inplace kernel function
 */
#define TENZOR_REGISTER_INPLACE_KERNEL(table, op_id, kernel_fn) \
    (table).register_inplace_kernel(::tenzor::OpId::op_id, \
        [](::tenzor::Tensor& target, std::span<const ::tenzor::Tensor> others, \
           const ::tenzor::OpAttributes&) -> ::tenzor::Tensor& { \
            (kernel_fn)(target, others[0]); \
            return target; \
        })

/**
 * @brief Register a reduction operation kernel.
 *
 * For kernels with signature: Tensor(const Tensor&, int64_t dim, bool keepdim)
 * Reads "dim" and "keepdim" from typed attributes.
 *
 * @param table BackendDispatchTable reference
 * @param op_id OpId enum value (without OpId:: prefix)
 * @param kernel_fn Reduction kernel function
 */
#define TENZOR_REGISTER_REDUCTION_KERNEL(table, op_id, kernel_fn) \
    (table).register_kernel(::tenzor::OpId::op_id, \
        [](std::span<const ::tenzor::Tensor> inputs, const ::tenzor::OpAttributes& attrs) \
            -> std::vector<::tenzor::Tensor> { \
            int64_t dim = attrs.get_int(::tenzor::AttrKey::Dim, LLONG_MIN); \
            bool keepdim = attrs.get_bool(::tenzor::AttrKey::Keepdim, false); \
            return {(kernel_fn)(inputs[0], dim, keepdim)}; \
        })

} // namespace tenzor
