/**
 * @file fast_dispatch.hpp
 * @brief Fast O(1) dispatch interface for tensor operations
 *
 * Provides the primary dispatch interface using OpId for direct table lookup.
 * This replaces the string-based Dispatcher::dispatch() with ~50-100x faster
 * dispatch overhead.
 *
 * Usage:
 * @code
 * // Compile-time OpId (preferred - enables optimizations)
 * auto result = dispatch<OpId::MatMul>({a, b});
 *
 * // Runtime OpId (for JIT/dynamic scenarios)
 * OpId op = get_op_from_somewhere();
 * auto result = dispatch(op, {a, b});
 * @endcode
 */

#pragma once

#include <span>
#include <vector>
#include <stdexcept>
#include "../core/tensor.hpp"
#include "../ops/op_id.hpp"
#include "../nn/amp/autocast.hpp"
#include "dispatch_table.hpp"
#include "dispatch_interceptor.hpp"
#include "op_attributes.hpp"

namespace tenzor {

// =============================================================================
// Autocast Op Classification (O(1) lookup via bitset)
// =============================================================================

namespace detail {

/// Compute-heavy ops that benefit from lower precision (Float16/BFloat16).
/// These are typically matmul-like or convolution-like operations.
inline bool is_autocast_compute_heavy(OpId op) {
    switch (op) {
        case OpId::MatMul:
        case OpId::Bmm:
        case OpId::Dot:
        case OpId::Linear:
        case OpId::Conv2dForward:
        case OpId::Conv3dForward:
        case OpId::ConvTranspose2dForward:
        case OpId::DepthwiseConv2d:
        case OpId::LSTMForward:
        case OpId::LSTMCellForward:
        case OpId::LSTMMultiLayerForward:
        case OpId::BiLSTMForward:
        case OpId::GRUForward:
        case OpId::GRUCellForward:
        case OpId::GRUMultiLayerForward:
        case OpId::FusedLinearReLU:
        case OpId::FusedConv2dReLU:
        case OpId::FusedConv2dBnReLU:
        case OpId::FusedAttention:
        case OpId::FlashAttention:
            return true;
        default:
            return false;
    }
}

/// Stability-critical ops that must stay in Float32 for numerical correctness.
inline bool is_autocast_stability_critical(OpId op) {
    switch (op) {
        case OpId::Softmax:
        case OpId::LogSoftmax:
        case OpId::BatchNorm2dForward:
        case OpId::BatchNorm2dFusedTraining:
        case OpId::LayerNorm:
        case OpId::GroupNorm:
        case OpId::InstanceNorm:
        case OpId::FusedLayerNorm:
        case OpId::FusedRMSNorm:
        case OpId::FusedSoftmaxCrossEntropy:
            return true;
        default:
            return false;
    }
}

/// Apply autocast to inputs: returns a vector of (possibly cast) tensors.
/// Only allocates if casting is actually needed.
inline std::vector<Tensor> autocast_inputs(
    OpId op, std::span<const Tensor> inputs)
{
    if (!nn::amp::Autocast::is_enabled() || inputs.empty()) {
        return {};  // empty = no casting needed, use originals
    }

    auto target_dtype = nn::amp::Autocast::get_dtype();
    if (!target_dtype.has_value()) {
        return {};
    }

    // Stability-critical: promote all half-precision inputs to Float32
    if (is_autocast_stability_critical(op)) {
        bool needs_promote = false;
        for (const auto& t : inputs) {
            if (t.dtype() == DType::Float16 || t.dtype() == DType::BFloat16) {
                needs_promote = true;
                break;
            }
        }
        if (!needs_promote) return {};

        std::vector<Tensor> promoted;
        promoted.reserve(inputs.size());
        for (const auto& t : inputs) {
            if (t.dtype() == DType::Float16 || t.dtype() == DType::BFloat16) {
                promoted.push_back(t.to(DType::Float32));
            } else {
                promoted.push_back(t);
            }
        }
        return promoted;
    }

    // Compute-heavy: cast Float32 inputs to target half-precision dtype
    if (is_autocast_compute_heavy(op)) {
        DType target = target_dtype.value();
        bool needs_cast = false;
        for (const auto& t : inputs) {
            if (t.dtype() == DType::Float32) {
                needs_cast = true;
                break;
            }
        }
        if (!needs_cast) return {};

        std::vector<Tensor> casted;
        casted.reserve(inputs.size());
        for (const auto& t : inputs) {
            if (t.dtype() == DType::Float32) {
                casted.push_back(t.to(target));
            } else {
                casted.push_back(t);
            }
        }
        return casted;
    }

    return {};  // Other ops: no casting
}

} // namespace detail

/**
 * @brief Determine device type from input tensors.
 *
 * All input tensors must be on the same device. Returns the common device type.
 *
 * @param inputs Input tensors
 * @return Device type of inputs
 * @throws std::runtime_error if tensors are on different devices
 */
inline Device::Type get_dispatch_device(std::span<const Tensor> inputs) {
    if (inputs.empty()) {
        return Device::Type::CPU;  // Default for creation ops
    }

    Device::Type device_type = inputs[0].device().type;

    // Verify all inputs are on the same device type
    for (size_t i = 1; i < inputs.size(); ++i) {
        if (inputs[i].device().type != device_type) [[unlikely]] {
            throw std::runtime_error(
                "All input tensors must be on the same device type. "
                "Got " + std::string(device_type_to_string(device_type)) +
                " and " + std::string(device_type_to_string(inputs[i].device().type))
            );
        }
    }

    return device_type;
}

/**
 * @brief Runtime dispatch with OpId.
 *
 * Single O(1) dispatch path:
 * 1. Determine device type from inputs
 * 2. Look up table[device_type]
 * 3. Look up kernel = table.kernels[op_id]
 * 4. Call kernel(inputs, attrs)
 *
 * @param op Operation identifier
 * @param inputs Input tensors
 * @param attrs Operation attributes
 * @return Output tensors
 * @throws std::runtime_error if operation not supported on device
 *
 * @code
 * auto result = dispatch(OpId::Add, {a, b}, {});
 * @endcode
 */
inline std::vector<Tensor> dispatch(
    OpId op,
    std::span<const Tensor> inputs,
    const OpAttributes& attrs = {})
{
    // Terminal: pure backend dispatch (autocast is handled by its own interceptor)
    DispatchNext terminal = [](OpId o, std::span<const Tensor> ins,
                               const OpAttributes& a) -> std::vector<Tensor> {
        Device::Type device_type = get_dispatch_device(ins);
        const auto& table = DispatchTableRegistry::get_table_const(device_type);
        return table.dispatch(o, ins, a);
    };

    // Run through interceptor stack (zero overhead when stack is empty)
    // When autocast is enabled, its interceptor is in the stack and handles casting.
    return DispatchInterceptorStack::run(op, inputs, attrs, std::move(terminal));
}

/**
 * @brief Compile-time dispatch with OpId template parameter.
 *
 * Same as runtime dispatch but with compile-time OpId for potential
 * compiler optimizations (inlining, branch prediction hints).
 *
 * @tparam Op Operation identifier (compile-time constant)
 * @param inputs Input tensors
 * @param attrs Operation attributes
 * @return Output tensors
 *
 * @code
 * auto result = dispatch<OpId::MatMul>({a, b});
 * @endcode
 */
template<OpId Op>
inline std::vector<Tensor> dispatch(
    std::span<const Tensor> inputs,
    const OpAttributes& attrs = {})
{
    return dispatch(Op, inputs, attrs);
}

// =============================================================================
// Single-Output Optimized Dispatch
// =============================================================================
// These functions avoid vector allocation for operations that return exactly
// one tensor (most operations). Use these for performance-critical paths.

/**
 * @brief Optimized dispatch for single-output operations.
 *
 * Returns Tensor directly without vector allocation overhead.
 * For operations like matmul, linear, add, etc. that produce one output.
 *
 * Performance: ~10x faster than regular dispatch for small operations
 * where vector allocation dominates.
 *
 * @param op Operation identifier
 * @param inputs Input tensors
 * @param attrs Operation attributes
 * @return Single output tensor
 */
inline Tensor dispatch_single(
    OpId op,
    std::span<const Tensor> inputs,
    const OpAttributes& attrs = {})
{
    // Terminal: pure backend dispatch (autocast handled by interceptor stack)
    DispatchNextSingle terminal = [](OpId o, std::span<const Tensor> ins,
                                     const OpAttributes& a) -> Tensor {
        Device::Type device_type = get_dispatch_device(ins);
        const auto& table = DispatchTableRegistry::get_table_const(device_type);
        return table.dispatch_single(o, ins, a);
    };

    // Run through interceptor stack (zero overhead when stack is empty)
    return DispatchInterceptorStack::run_single(op, inputs, attrs, std::move(terminal));
}

/**
 * @brief Compile-time optimized dispatch for single-output operations.
 *
 * Template version for potential compiler optimizations.
 *
 * @tparam Op Operation identifier (compile-time constant)
 * @param inputs Input tensors
 * @param attrs Operation attributes
 * @return Single output tensor
 *
 * @code
 * auto result = dispatch_single<OpId::Linear>({x, w, b});
 * @endcode
 */
template<OpId Op>
inline Tensor dispatch_single(
    std::span<const Tensor> inputs,
    const OpAttributes& attrs = {})
{
    return dispatch_single(Op, inputs, attrs);
}

/**
 * @brief Dispatch with explicit device type (for creation operations).
 *
 * Used when inputs are empty and device must be specified explicitly.
 *
 * @param op Operation identifier
 * @param device_type Target device type
 * @param inputs Input tensors (may be empty)
 * @param attrs Operation attributes
 * @return Output tensors
 *
 * @code
 * // Create zeros on CUDA
 * auto zeros = dispatch_to_device(OpId::Zeros, Device::Type::CUDA, {}, {{"shape", "3,4"}});
 * @endcode
 */
inline std::vector<Tensor> dispatch_to_device(
    OpId op,
    Device::Type device_type,
    std::span<const Tensor> inputs,
    const OpAttributes& attrs = {})
{
    const auto& table = DispatchTableRegistry::get_table_const(device_type);
    return table.dispatch(op, inputs, attrs);
}

/**
 * @brief Dispatch inplace operation on target tensor.
 *
 * Routes to the inplace kernel directly, avoiding const_cast.
 * The target tensor is modified in-place.
 *
 * @param op Operation identifier (should be an inplace op)
 * @param target The tensor to modify in-place
 * @param others Additional input tensors
 * @param attrs Operation attributes
 * @return Reference to the modified target tensor
 */
inline Tensor& dispatch_inplace(
    OpId op,
    Tensor& target,
    std::span<const Tensor> others,
    const OpAttributes& attrs = {})
{
    Device::Type device_type = target.device().type;
    auto& table = DispatchTableRegistry::get_table(device_type);
    return table.dispatch_inplace(op, target, others, attrs);
}

/**
 * @brief Check if an operation is supported on a device type.
 *
 * @param op Operation identifier
 * @param device_type Device type to check
 * @return true if operation has a kernel registered
 */
inline bool is_op_supported(OpId op, Device::Type device_type) noexcept {
    return DispatchTableRegistry::get_table_const(device_type).has_kernel(op);
}

} // namespace tenzor
