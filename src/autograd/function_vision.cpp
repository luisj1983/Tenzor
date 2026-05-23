/**
 * @file function_vision.cpp
 * @brief Autograd backward functions for vision operations (grid_sample, affine_grid).
 *
 * Audit Q.4: the previous implementation did `.to(cpu()).to(Float32)` round-trips
 * inside backward(), violating the no-CPU-fallback rule and silently narrowing
 * Float64 to Float32. The backward math is now lifted into dedicated backend
 * kernels (OpId::GridSampleBackward, OpId::AffineGridBackward) and dispatched
 * through the regular fast-dispatch path. CPU/CUDA/ROCm/OneAPI/Vulkan all
 * implement the backward natively; if a backend without a registered backward
 * is hit the dispatcher throws — there is no silent fallback.
 */

#include "tenzor/autograd/function.hpp"
#include "tenzor/ops/vision.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/utils/logging.hpp"
#include <cmath>
#include <algorithm>

namespace tenzor {

// ============================================================================
// GridSampleBackward
// ============================================================================

auto GridSampleBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    save_for_backward({inputs[0].tensor(), inputs[1].tensor()});

    auto result = ops::grid_sample(inputs[0].tensor(), inputs[1].tensor(),
                                   mode_, padding_mode_, align_corners_);
    return {Variable(result, inputs[0].requires_grad() || inputs[1].requires_grad())};
}

auto GridSampleBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    const Tensor& input = saved_tensors_[0];
    const Tensor& grid = saved_tensors_[1];
    const Tensor& grad_output = grad_outputs[0];

    OpAttributes attrs;
    attrs.set(AttrKey::Mode, mode_);
    attrs.set(AttrKey::PaddingMode, padding_mode_);
    attrs.set(AttrKey::AlignCorners, align_corners_);

    std::vector<Tensor> dispatch_inputs = {input, grid, grad_output};
    auto outs = tenzor::dispatch(OpId::GridSampleBackward, dispatch_inputs, attrs);
    if (outs.size() != 2) {
        throw std::runtime_error(
            "GridSampleBackward: backend kernel must return {grad_input, grad_grid}");
    }
    return {outs[0], outs[1]};
}

// ============================================================================
// AffineGridBackward
// ============================================================================

auto AffineGridBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    save_for_backward({inputs[0].tensor()});

    auto result = ops::affine_grid(inputs[0].tensor(), size_, align_corners_);
    return {Variable(result, inputs[0].requires_grad())};
}

auto AffineGridBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const Tensor& grad_grid = grad_outputs[0];

    OpAttributes attrs;
    // OutputSize is the {N, C, H, W} layout from the forward, serialized as
    // a comma-separated string per the OpAttributes convention.
    std::string size_str;
    for (size_t i = 0; i < size_.size(); ++i) {
        if (i > 0) size_str += ",";
        size_str += std::to_string(size_[i]);
    }
    attrs.set(AttrKey::OutputSize, size_str);
    attrs.set(AttrKey::AlignCorners, align_corners_);

    std::vector<Tensor> dispatch_inputs = {grad_grid};
    auto results = tenzor::dispatch(OpId::AffineGridBackward, dispatch_inputs, attrs);
    return {results[0]};
}

auto GridSampleBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // R.5 — honest higher-order stub. The first-order backward is an
    // opaque per-backend kernel (OpId::GridSampleBackward) taking
    // (input, grid, grad_output) and returning (grad_input, grad_grid).
    // No Variable-level composition expresses the second derivative
    // without dedicated `GridSampleBackwardBackward` kernels per backend
    // (not yet shipped). Surface limitation via WARN_ONCE +
    // is_higher_order_stub() so the engine counter fires instead of
    // silently zeroing higher-order grads.
    TENZOR_WARN_ONCE(
        "[GridSampleBackward] higher-order backward is a stub — no "
        "Variable-level composition of OpId::GridSampleBackward exists; "
        "second-order grads will be zero.");
    auto result_tensors = backward({grad_outputs[0].tensor()});
    std::vector<Variable> results;
    results.reserve(result_tensors.size());
    for (auto& t : result_tensors) {
        results.emplace_back(std::move(t), grad_outputs[0].requires_grad());
    }
    return results;
}

auto AffineGridBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // R.5 — honest higher-order stub. AffineGrid's first-order backward is
    // a per-backend opaque kernel (OpId::AffineGridBackward); no
    // Variable-level decomposition is available without dedicated
    // second-order kernels per backend.
    TENZOR_WARN_ONCE(
        "[AffineGridBackward] higher-order backward is a stub — no "
        "Variable-level composition of OpId::AffineGridBackward exists; "
        "second-order grads will be zero.");
    auto result_tensors = backward({grad_outputs[0].tensor()});
    std::vector<Variable> results;
    results.reserve(result_tensors.size());
    for (auto& t : result_tensors) {
        results.emplace_back(std::move(t), grad_outputs[0].requires_grad());
    }
    return results;
}

} // namespace tenzor
