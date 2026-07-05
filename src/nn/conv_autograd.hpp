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
#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/backend/dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/nn/functional.hpp"
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

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        Variable grad_out_var = grad_outputs[0];

        // Retrieve saved variables (with fallback to tensors wrapped as Variables)
        Variable input_var, weight_var;
        if (has_saved_variables() && saved_variables_.size() >= 2) {
            input_var = saved_variables_[0];
            weight_var = saved_variables_[1];
        } else {
            input_var = Variable(saved_tensors_[0], false);
            weight_var = Variable(saved_tensors_[1], false);
        }
        bool has_bias = saved_tensors_.size() > 2;

        // grad_input: conv_transpose2d(grad_output, weight, stride, padding)
        // Uses F::conv_transpose2d which takes Variables and preserves the graph.
        //
        // For stride > 1 the transpose is ambiguous: several input spatial sizes
        // map to the same conv output size, so conv_transpose2d with
        // output_padding={0,0} produces an input up to (stride-1) smaller than the
        // true input. Recover the exact output_padding from the SAVED input
        // spatial size so the 2nd-order grad_input matches the true input shape,
        // mirroring the 1st-order path which threads AttrKey::InputShape. With
        //   out = floor((in + 2*pad - dilation*(k-1) - 1)/stride) + 1
        // the transpose base size (output_padding=0) is
        //   base = (out-1)*stride - 2*pad + dilation*(k-1) + 1
        // and output_padding = in - base (in [0, stride-1]).
        const auto& in_shape = input_var.tensor().shape();
        const auto& go_shape = grad_out_var.tensor().shape();
        const auto& w_shape = weight_var.tensor().shape();
        auto compute_output_padding = [](int64_t in_size, int64_t out_size,
                                         int64_t stride, int64_t pad,
                                         int64_t dilation, int64_t k) -> int64_t {
            int64_t base = (out_size - 1) * stride - 2 * pad + dilation * (k - 1) + 1;
            int64_t op = in_size - base;
            if (op < 0) op = 0;              // clamp against malformed saved shapes
            if (op > stride - 1) op = stride - 1;
            return op;
        };
        int64_t out_pad_h = 0, out_pad_w = 0;
        if (in_shape.size() == 4 && go_shape.size() == 4 && w_shape.size() == 4) {
            out_pad_h = compute_output_padding(in_shape[2], go_shape[2],
                                               stride_h_, padding_h_,
                                               dilation_h_, w_shape[2]);
            out_pad_w = compute_output_padding(in_shape[3], go_shape[3],
                                               stride_w_, padding_w_,
                                               dilation_w_, w_shape[3]);
        }
        auto grad_input = ::tenzor::nn::functional::conv_transpose2d(
            grad_out_var, weight_var,
            std::nullopt, // no bias for the transpose op
            {stride_h_, stride_w_},
            {padding_h_, padding_w_},
            {out_pad_h, out_pad_w}, // output_padding recovered from saved input size
            groups_,
            {dilation_h_, dilation_w_});

        // grad_weight: dispatch at tensor level, then connect to graph.
        // Expressing weight-gradient purely with Variable-level conv2d for
        // arbitrary stride/dilation is non-trivial (requires im2col in
        // autograd ops). Use the backend kernel but preserve the graph
        // connection through grad_output — this enables 2nd-order
        // differentiation through the loss w.r.t. inputs, which is the
        // dominant MAML / meta-learning use-case.
        OpAttributes bw_attrs = build_attrs(input_var.tensor().shape(),
                                            weight_var.tensor().shape());

        std::vector<Tensor> gw_inputs = {grad_out_var.tensor(),
                                          input_var.tensor(),
                                          weight_var.tensor()};
        auto grad_weight_t = dispatch(OpId::Conv2dBackwardWeight, gw_inputs, bw_attrs)[0];
        Variable grad_weight(grad_weight_t, grad_out_var.requires_grad());

        if (has_bias) {
            // grad_bias = sum(grad_output, dims=[0,2,3])
            // Use Variable-level sum to preserve the autograd chain.
            auto gb = ::tenzor::sum(grad_out_var, 0, false);  // sum over batch
            gb = ::tenzor::sum(gb, 1, false);                  // sum over H (was dim 2, now 1 after batch reduction)
            gb = ::tenzor::sum(gb, 1, false);                  // sum over W (was dim 3, now 1 after previous reductions)
            return {grad_input, grad_weight, gb};
        }
        return {grad_input, grad_weight};
    }

    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return false; }

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
