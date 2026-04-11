/**
 * @file conv_autograd.hpp
 * @brief Shared Conv2dBackward autograd function.
 *
 * The autograd Function wiring for conv2d lives here so both
 * `nn::Conv2d` (the module) and `nn::functional::conv2d` (the
 * stateless functional entry point) can use it. Previously only the
 * module set up the backward graph, so F::conv2d silently produced
 * tensors with no grad_fn and gradient tests on F::conv2d failed with
 * `input.has_grad() == false`.
 *
 * Internal to nn/ — not part of the public API.
 */

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/backend/dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/op_id.hpp"

namespace tenzor::nn::internal {

class Conv2dBackward : public ::tenzor::Function {
public:
    Conv2dBackward(int64_t stride_h, int64_t stride_w,
                   int64_t padding_h, int64_t padding_w,
                   int64_t dilation_h, int64_t dilation_w,
                   int64_t groups,
                   std::vector<Tensor> tensors_to_save)
        : stride_h_(stride_h), stride_w_(stride_w),
          padding_h_(padding_h), padding_w_(padding_w),
          dilation_h_(dilation_h), dilation_w_(dilation_w),
          groups_(groups) {
        save_for_backward(std::move(tensors_to_save));
    }

    auto forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> override {
        throw std::runtime_error("Conv2dBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        const Tensor& grad_output = grad_outputs[0];
        const Tensor& input = saved_tensors_[0];
        const Tensor& weight = saved_tensors_[1];

        bool has_bias = saved_tensors_.size() > 2;

        OpAttributes backward_attrs = build_attrs(input.shape(), weight.shape());

        std::vector<Tensor> grad_input_inputs = {grad_output, input, weight};
        auto grad_input_result =
            dispatch(OpId::Conv2dBackwardInput, grad_input_inputs, backward_attrs);

        std::vector<Tensor> grad_weight_inputs = {grad_output, input, weight};
        auto grad_weight_result =
            dispatch(OpId::Conv2dBackwardWeight, grad_weight_inputs, backward_attrs);

        if (has_bias) {
            std::vector<Tensor> grad_bias_inputs = {grad_output};
            auto grad_bias_result =
                dispatch(OpId::Conv2dBackwardBias, grad_bias_inputs, backward_attrs);
            return {grad_input_result[0], grad_weight_result[0], grad_bias_result[0]};
        }
        return {grad_input_result[0], grad_weight_result[0]};
    }

    auto supports_higher_order() const -> bool override { return false; }

private:
    auto build_attrs(std::span<const int64_t> input_shape,
                     std::span<const int64_t> weight_shape) const -> OpAttributes {
        OpAttributes a;
        a.set(AttrKey::Stride, stride_h_);
        a.set(AttrKey::Padding, padding_h_);
        a.set(AttrKey::Dilation, dilation_h_);
        a.set(AttrKey::StrideH, stride_h_);
        a.set(AttrKey::StrideW, stride_w_);
        a.set(AttrKey::PaddingH, padding_h_);
        a.set(AttrKey::PaddingW, padding_w_);
        a.set(AttrKey::DilationH, dilation_h_);
        a.set(AttrKey::DilationW, dilation_w_);
        a.set(AttrKey::Groups, groups_);
        a.set(AttrKey::InputShape, shape_to_string(input_shape));
        a.set(AttrKey::WeightShape, shape_to_string(weight_shape));
        return a;
    }

    static auto shape_to_string(std::span<const int64_t> s) -> std::string {
        std::string out;
        for (size_t i = 0; i < s.size(); ++i) {
            if (i > 0) out += ',';
            out += std::to_string(s[i]);
        }
        return out;
    }

    int64_t stride_h_, stride_w_;
    int64_t padding_h_, padding_w_;
    int64_t dilation_h_, dilation_w_;
    int64_t groups_;
};

} // namespace tenzor::nn::internal
