#pragma once

// Internal helpers shared across split function_*.cpp files.
// NOT part of the public API.

#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/reduction.hpp"
#include <stdexcept>
#include <string>
#include <vector>

namespace tenzor {

// Reduce gradient Variable along broadcasted dimensions (for create_graph)
inline auto reduce_grad_var_for_broadcasting(const Variable& grad, const std::vector<int64_t>& target_shape) -> Variable {
    auto grad_shape_vec = std::vector<int64_t>(grad.shape().begin(), grad.shape().end());

    if (grad_shape_vec == target_shape) {
        return grad;
    }

    // Handle empty tensors: reshape directly to target
    if (grad.tensor().numel() == 0) {
        return tenzor::reshape(grad, target_shape);
    }

    // Handle scalar target (empty shape): sum all dimensions
    if (target_shape.empty()) {
        auto result = grad;
        for (int64_t d = static_cast<int64_t>(grad_shape_vec.size()) - 1; d >= 0; --d) {
            result = tenzor::sum(result, 0, false);
        }
        return result;
    }

    auto result = grad;

    int64_t ndim_diff = static_cast<int64_t>(grad_shape_vec.size()) - static_cast<int64_t>(target_shape.size());

    if (ndim_diff > 0) {
        for (int64_t i = 0; i < ndim_diff; ++i) {
            result = tenzor::sum(result, 0, false);
        }
    } else if (ndim_diff < 0) {
        throw std::runtime_error(
            "Autograd bug: gradient has fewer dimensions (" +
            std::to_string(grad_shape_vec.size()) + ") than target shape (" +
            std::to_string(target_shape.size()) + ")");
    }

    auto result_shape_vec = std::vector<int64_t>(result.shape().begin(), result.shape().end());
    for (size_t i = 0; i < target_shape.size(); ++i) {
        if (target_shape[i] == 1 && result_shape_vec[i] > 1) {
            result = tenzor::sum(result, static_cast<int64_t>(i), true);
            result_shape_vec = std::vector<int64_t>(result.shape().begin(), result.shape().end());
        }
    }

    if (result_shape_vec != target_shape) {
        result = tenzor::reshape(result, target_shape);
    }

    // Final validation: shape must match target after all reductions
    auto final_shape = std::vector<int64_t>(result.shape().begin(), result.shape().end());
    if (final_shape != target_shape) {
        throw std::runtime_error(
            "Autograd bug: reduce_grad_var_for_broadcasting failed to produce target shape");
    }

    return result;
}

// Reduce gradient along broadcasted dimensions
inline auto reduce_grad_for_broadcasting(const Tensor& grad, const std::vector<int64_t>& target_shape) -> Tensor {
    auto grad_shape_vec = std::vector<int64_t>(grad.shape().begin(), grad.shape().end());

    if (grad_shape_vec == target_shape) {
        return grad;
    }

    // Handle empty tensors: reshape directly to target
    if (grad.numel() == 0) {
        return reshape(grad, target_shape);
    }

    // Handle scalar target (empty shape): sum all dimensions
    if (target_shape.empty()) {
        auto result = grad;
        while (result.dim() > 0) {
            result = tenzor::sum(result, 0, false);
        }
        return result;
    }

    auto result = grad;

    int64_t ndim_diff = static_cast<int64_t>(grad_shape_vec.size()) - static_cast<int64_t>(target_shape.size());

    if (ndim_diff > 0) {
        for (int64_t i = 0; i < ndim_diff; ++i) {
            result = tenzor::sum(result, 0, false);
        }
    } else if (ndim_diff < 0) {
        throw std::runtime_error(
            "Autograd bug: gradient has fewer dimensions (" +
            std::to_string(grad_shape_vec.size()) + ") than target shape (" +
            std::to_string(target_shape.size()) + ")");
    }

    auto result_shape_vec = std::vector<int64_t>(result.shape().begin(), result.shape().end());
    for (size_t i = 0; i < target_shape.size(); ++i) {
        if (target_shape[i] == 1 && result_shape_vec[i] > 1) {
            result = tenzor::sum(result, static_cast<int64_t>(i), true);
            result_shape_vec = std::vector<int64_t>(result.shape().begin(), result.shape().end());
        }
    }

    if (result_shape_vec != target_shape) {
        result = reshape(result, target_shape);
    }

    // Final validation: shape must match target after all reductions
    auto final_shape = std::vector<int64_t>(result.shape().begin(), result.shape().end());
    if (final_shape != target_shape) {
        throw std::runtime_error(
            "Autograd bug: reduce_grad_for_broadcasting failed to produce target shape");
    }

    return result;
}

} // namespace tenzor
