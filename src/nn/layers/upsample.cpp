#include "tenzor/nn/layers/upsample.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/vision.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tenzor::nn {

// ============================================================================
// Upsample backward: gradient flows through interpolate backward
// ============================================================================

class UpsampleBackward : public Function {
public:
    UpsampleBackward(std::vector<int64_t> input_spatial_size,
                     std::string mode, bool align_corners)
        : input_spatial_size_(std::move(input_spatial_size)),
          mode_(std::move(mode)), align_corners_(align_corners) {}

    auto forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> override {
        throw std::runtime_error("UpsampleBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        // Downsample gradient back to input size using the same interpolation mode
        auto grad_input = ops::interpolate(grad_outputs[0], input_spatial_size_,
                                           mode_, align_corners_);
        return {grad_input};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        auto grad_tensor = ops::interpolate(grad_outputs[0].tensor(), input_spatial_size_,
                                            mode_, align_corners_);
        return {Variable(grad_tensor, true)};
    }

    // P4.2d: upsample is linear (fixed interpolation weights); second
    // derivative is structurally zero.
    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return true; }

private:
    std::vector<int64_t> input_spatial_size_;
    std::string mode_;
    bool align_corners_;
};

// ============================================================================
// Upsample implementation
// ============================================================================

Upsample::Upsample(std::optional<std::vector<int64_t>> size,
                    std::optional<double> scale_factor,
                    const std::string& mode,
                    bool align_corners)
    : size_(std::move(size)), scale_factor_(scale_factor),
      mode_(mode), align_corners_(align_corners) {
    if (!size_.has_value() && !scale_factor_.has_value()) {
        throw std::invalid_argument(
            "Upsample: exactly one of 'size' or 'scale_factor' must be provided");
    }
    if (size_.has_value() && scale_factor_.has_value()) {
        throw std::invalid_argument(
            "Upsample: only one of 'size' or 'scale_factor' should be provided");
    }
    if (scale_factor_.has_value() && scale_factor_.value() <= 0.0) {
        throw std::invalid_argument(
            "Upsample: scale_factor must be positive, got " +
            std::to_string(scale_factor_.value()));
    }
    if (mode != "nearest" && mode != "bilinear" && mode != "trilinear") {
        throw std::invalid_argument(
            "Upsample: mode must be 'nearest', 'bilinear', or 'trilinear', got '" +
            mode + "'");
    }
    if (align_corners && mode == "nearest") {
        throw std::invalid_argument(
            "Upsample: align_corners option can only be set with 'bilinear' or 'trilinear' modes");
    }
}

auto Upsample::forward_impl(const Variable& input) -> Variable {
    auto shape = input.tensor().shape();
    auto ndim = shape.size();

    if (ndim != 4 && ndim != 5) {
        throw std::invalid_argument(
            "Upsample: input must be 4D (N,C,H,W) or 5D (N,C,D,H,W), got " +
            std::to_string(ndim) + "D");
    }

    // Compute target size
    std::vector<int64_t> target_size;
    if (size_.has_value()) {
        target_size = size_.value();
    } else {
        // Compute from scale_factor; clamp to at least 1 so a small scale
        // factor cannot collapse a spatial dimension to zero.
        double sf = scale_factor_.value();
        size_t num_spatial = ndim - 2;
        for (size_t i = 0; i < num_spatial; ++i) {
            target_size.push_back(std::max<int64_t>(
                1, static_cast<int64_t>(std::floor(shape[2 + i] * sf))));
        }
    }

    // Save input spatial size for backward
    std::vector<int64_t> input_spatial_size;
    for (size_t i = 2; i < ndim; ++i) {
        input_spatial_size.push_back(shape[i]);
    }

    // Perform interpolation
    auto output_tensor = ops::interpolate(input.tensor(), target_size,
                                          mode_, align_corners_);
    Variable output(output_tensor, input.requires_grad());

    if (input.requires_grad()) {
        auto upsample_fn = std::make_shared<UpsampleBackward>(
            input_spatial_size, mode_, align_corners_);

        std::vector<Variable> input_vars{input};
        upsample_fn->set_input_variables(input_vars);

        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) {
            next_funcs.push_back(input.grad_fn());
        }
        upsample_fn->set_next_functions(next_funcs);

        output.set_grad_fn(upsample_fn);
    }

    return output;
}

} // namespace tenzor::nn
