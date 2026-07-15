#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/functional.hpp"
#include "tenzor/nn/utils/variable_cast.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/backend/dispatch.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/utils/autograd_wrap.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <omp.h>

namespace tenzor::nn {

// Helper namespace for convolution operations
namespace {

// JIT-R073: Tensor::unsqueeze()/squeeze() are pure TensorImpl metadata
// tricks with ZERO dispatch() calls, invisible to both the tracer's generic
// dispatch<OpId> interception and its own tensor lineage map — the SAME bug
// class as JIT-R052/R058a. Currently masked (OpId::DepthwiseConv1d is
// unmapped, so the dispatch() call these wrap graph-breaks first regardless
// of whether the operands were traced), but this is the correct fix ready
// for when depthwise Conv1d JIT tracing support is added. Mirrors quantized_
// layers.cpp's traced_unsqueeze/traced_squeeze exactly: routes through the
// Variable-level autograd wrappers, which DO call jit_record_shape_op() when
// tracing is active. Safe to wrap in Variable(t, false) — every call site is
// inference-only glue around a dispatch() call, not user-facing autograd.
inline auto traced_unsqueeze(const Tensor& t, int64_t dim) -> Tensor {
    return ::tenzor::unsqueeze(::tenzor::Variable(t, false), dim).tensor();
}
inline auto traced_squeeze(const Tensor& t, int64_t dim) -> Tensor {
    return ::tenzor::squeeze(::tenzor::Variable(t, false), dim).tensor();
}

// Calculate output size for convolution
auto calculate_output_size(int64_t input_size, int64_t kernel_size,
                           int64_t stride, int64_t padding, int64_t dilation) -> int64_t {
    // Numerator can be negative when the effective dilated kernel exceeds the
    // padded input. C++ integer division truncates toward zero, so for a
    // numerator in (-stride, 0) it would yield 0 and produce a spurious
    // out_size == 1 that slips past the downstream out_size <= 0 guard. Use
    // floor-division semantics (matching PyTorch) so an invalid geometry maps
    // to a non-positive size that the caller's guard rejects.
    int64_t numerator = input_size + 2 * padding - dilation * (kernel_size - 1) - 1;
    if (numerator < 0) {
        // floor(numerator / stride) for numerator < 0 and stride > 0.
        int64_t q = -((-numerator + stride - 1) / stride);
        return q + 1;
    }
    return numerator / stride + 1;
}

// Pad a 3D tensor [N, C, L] in the last dimension
auto pad_1d(const Tensor& input, int64_t padding) -> Tensor {
    if (padding <= 0) return input;

    auto shape = input.shape();
    int64_t batch = shape[0];
    int64_t channels = shape[1];

    // Create zero padding tensors
    auto left_pad = zeros({batch, channels, padding}, input.dtype(), input.device());
    auto right_pad = zeros({batch, channels, padding}, input.dtype(), input.device());

    // Concatenate: [left_pad, input, right_pad]
    return cat({left_pad, input, right_pad}, 2);
}

} // anonymous namespace

// ============================================================================
// Conv2dBackward - Autograd Function
// ============================================================================

// ============================================================================
// Conv2dGradWeightBackward - JIT-R067
//
// grad_weight[oc,ic,kh,kw] = sum_{b,oh,ow} grad_out[b,oc,oh,ow] *
// input[b,ic, oh*sH-pH+kh*dH, ow*sW-pW+kw*dW] is BILINEAR in (grad_out,
// input). Given upstream gradient G (same shape as weight), the two partial
// derivatives collapse onto the SAME primitives Conv2dBackward's own
// grad_input already uses:
//   d(Loss)/d(grad_out) = conv2d(input, G, stride, padding, dilation, groups)
//   d(Loss)/d(input)    = conv_transpose2d(grad_out, G, stride, padding,
//                                           output_padding=0, groups, dilation)
// (the second is literally grad_input's own formula with G substituted for
// weight — falls out of bilinearity). Both route through the already-
// differentiable, already-multi-backend-verified functional::conv2d /
// conv_transpose2d, so this needs no new backend kernel work and inherits
// correctness from those existing, tested entry points. This closes the
// gap where a 3rd-order gradient (a loss depending on grad_weight from a
// create_graph=true backward, differentiated again) silently returned zero.
// ============================================================================
class Conv2dGradWeightBackward : public Function {
public:
    Conv2dGradWeightBackward(int64_t stride_h, int64_t stride_w,
                              int64_t padding_h, int64_t padding_w,
                              int64_t dilation_h, int64_t dilation_w,
                              int64_t groups,
                              std::vector<Tensor> tensors_to_save)
        : sH_(stride_h), sW_(stride_w), pH_(padding_h), pW_(padding_w),
          dH_(dilation_h), dW_(dilation_w), groups_(groups) {
        save_for_backward(std::move(tensors_to_save));
    }

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("Conv2dGradWeightBackward::forward should not be called");
    }

    // input_variables()/next_functions() order is {grad_out_var, input_var};
    // grad_outputs[0] = G (upstream gradient w.r.t. grad_weight, same shape
    // as weight). Returns {d(Loss)/d(grad_out), d(Loss)/d(input)}.
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        const Tensor& G = grad_outputs[0];
        const Tensor& grad_out_t = saved_tensors_[0];
        const Tensor& input_t = saved_tensors_[1];

        Variable Gv(G, false);
        Variable d_grad_out = ::tenzor::nn::functional::conv2d(
            Variable(input_t, false), Gv, std::nullopt,
            {sH_, sW_}, {pH_, pW_}, {dH_, dW_}, groups_);
        Variable d_input = ::tenzor::nn::functional::conv_transpose2d(
            Variable(grad_out_t, false), Gv, std::nullopt,
            {sH_, sW_}, {pH_, pW_}, {0, 0}, groups_, {dH_, dW_});
        return {d_grad_out.tensor(), d_input.tensor()};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        Variable G = grad_outputs[0];
        Variable grad_out_var, input_var;
        if (has_saved_variables() && saved_variables_.size() >= 2) {
            grad_out_var = saved_variables_[0];
            input_var = saved_variables_[1];
        } else {
            grad_out_var = Variable(saved_tensors_[0], false);
            input_var = Variable(saved_tensors_[1], false);
        }
        Variable d_grad_out = ::tenzor::nn::functional::conv2d(
            input_var, G, std::nullopt, {sH_, sW_}, {pH_, pW_}, {dH_, dW_}, groups_);
        Variable d_input = ::tenzor::nn::functional::conv_transpose2d(
            grad_out_var, G, std::nullopt, {sH_, sW_}, {pH_, pW_}, {0, 0}, groups_, {dH_, dW_});
        return {d_grad_out, d_input};
    }

    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return false; }

private:
    int64_t sH_, sW_, pH_, pW_, dH_, dW_, groups_;
};

class Conv2dBackward : public Function {
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

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("Conv2dBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        const Tensor& grad_output = grad_outputs[0];
        const Tensor& input = saved_tensors_[0];
        const Tensor& weight = saved_tensors_[1];

        bool has_bias = saved_tensors_.size() > 2;

        // Use OpId-based dispatch for each gradient component
        OpAttributes backward_attrs;
        backward_attrs.set(AttrKey::Stride, stride_h_);
        backward_attrs.set(AttrKey::Padding, padding_h_);
        backward_attrs.set(AttrKey::Dilation, dilation_h_);
        backward_attrs.set(AttrKey::StrideH, stride_h_);
        backward_attrs.set(AttrKey::StrideW, stride_w_);
        backward_attrs.set(AttrKey::PaddingH, padding_h_);
        backward_attrs.set(AttrKey::PaddingW, padding_w_);
        backward_attrs.set(AttrKey::DilationH, dilation_h_);
        backward_attrs.set(AttrKey::DilationW, dilation_w_);
        backward_attrs.set(AttrKey::Groups, groups_);

        // Set shape attributes required by backward kernels (as comma-separated strings)
        {
            auto is = input.shape();
            std::string is_str;
            for (size_t i = 0; i < is.size(); ++i) {
                if (i > 0) is_str += ',';
                is_str += std::to_string(is[i]);
            }
            backward_attrs.set(AttrKey::InputShape, is_str);

            auto ws = weight.shape();
            std::string ws_str;
            for (size_t i = 0; i < ws.size(); ++i) {
                if (i > 0) ws_str += ',';
                ws_str += std::to_string(ws[i]);
            }
            backward_attrs.set(AttrKey::WeightShape, ws_str);
        }

        // Conv2dBackwardInput: inputs = {grad_output, input, weight}
        // All 3 tensors needed for cuDNN backward (grad_output for both, weight for grad_input, input for grad_weight)
        std::vector<Tensor> grad_input_inputs = {grad_output, input, weight};
        auto grad_input_result = dispatch(OpId::Conv2dBackwardInput, grad_input_inputs, backward_attrs);

        // Conv2dBackwardWeight: inputs = {grad_output, input, weight}
        std::vector<Tensor> grad_weight_inputs = {grad_output, input, weight};
        auto grad_weight_result = dispatch(OpId::Conv2dBackwardWeight, grad_weight_inputs, backward_attrs);

        if (has_bias) {
            // Conv2dBackwardBias: inputs = {grad_output}
            std::vector<Tensor> grad_bias_inputs = {grad_output};
            auto grad_bias_result = dispatch(OpId::Conv2dBackwardBias, grad_bias_inputs, backward_attrs);
            return {grad_input_result[0], grad_weight_result[0], grad_bias_result[0]};
        }
        return {grad_input_result[0], grad_weight_result[0]};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        // Build computation graph for higher-order gradients (MAML, Reptile, etc.)
        // Uses Variable-level ops so second derivatives flow through.
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
        // H18: for stride > 1 the transpose is ambiguous — several input
        // spatial sizes map to the same conv output size, so
        // output_padding={0,0} produces an input up to (stride-1) smaller
        // than the true input. Recover the exact output_padding from the
        // SAVED input spatial size, mirroring the already-fixed
        // internal::Conv2dBackward (conv_autograd.hpp, used only by
        // nn::functional::conv2d) so the module's own second-order path
        // matches it. With out = floor((in + 2*pad - dilation*(k-1) - 1)/stride) + 1,
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

        // grad_weight: dispatch at tensor level. Expressing weight-gradient
        // purely with Variable-level conv2d for arbitrary stride/dilation is
        // non-trivial. Use the backend kernel but preserve the graph
        // connection through grad_output — this enables 2nd-order
        // differentiation for the dominant MAML / meta-learning use-case.
        OpAttributes backward_attrs;
        backward_attrs.set(AttrKey::Stride, stride_h_);
        backward_attrs.set(AttrKey::Padding, padding_h_);
        backward_attrs.set(AttrKey::Dilation, dilation_h_);
        backward_attrs.set(AttrKey::StrideH, stride_h_);
        backward_attrs.set(AttrKey::StrideW, stride_w_);
        backward_attrs.set(AttrKey::PaddingH, padding_h_);
        backward_attrs.set(AttrKey::PaddingW, padding_w_);
        backward_attrs.set(AttrKey::DilationH, dilation_h_);
        backward_attrs.set(AttrKey::DilationW, dilation_w_);
        backward_attrs.set(AttrKey::Groups, groups_);
        {
            auto is = input_var.tensor().shape();
            std::string is_str;
            for (size_t i = 0; i < is.size(); ++i) {
                if (i > 0) is_str += ',';
                is_str += std::to_string(is[i]);
            }
            backward_attrs.set(AttrKey::InputShape, is_str);

            auto ws = weight_var.tensor().shape();
            std::string ws_str;
            for (size_t i = 0; i < ws.size(); ++i) {
                if (i > 0) ws_str += ',';
                ws_str += std::to_string(ws[i]);
            }
            backward_attrs.set(AttrKey::WeightShape, ws_str);
        }

        std::vector<Tensor> gw_inputs = {grad_out_var.tensor(),
                                           input_var.tensor(),
                                           weight_var.tensor()};
        auto grad_weight_t = dispatch(OpId::Conv2dBackwardWeight, gw_inputs, backward_attrs)[0];

        // JIT-R067: attach a differentiable grad_fn (Conv2dGradWeightBackward,
        // defined above) so a 3rd-order gradient — a loss that depends on
        // grad_weight from this create_graph=true backward, differentiated
        // again — flows correctly back to grad_out_var/input_var instead of
        // silently stopping here. grad_input's path already supports this;
        // this closes the matching gap on the weight-gradient path.
        bool gw_requires_grad = ::tenzor::is_grad_enabled() &&
            (grad_out_var.requires_grad() || input_var.requires_grad());
        Variable grad_weight(grad_weight_t, gw_requires_grad);
        if (gw_requires_grad) {
            auto gw_fn = std::make_shared<Conv2dGradWeightBackward>(
                stride_h_, stride_w_, padding_h_, padding_w_,
                dilation_h_, dilation_w_, groups_,
                std::vector<Tensor>{grad_out_var.tensor(), input_var.tensor()});
            if (::tenzor::is_creating_graph()) {
                gw_fn->save_variables_for_backward({grad_out_var, input_var});
            }
            gw_fn->set_next_functions({grad_out_var.grad_fn(), input_var.grad_fn()});
            gw_fn->set_input_variables({grad_out_var, input_var});
            grad_weight.set_grad_fn(gw_fn);
        }

        if (has_bias) {
            // grad_bias = sum(grad_output, dims=[0,2,3])
            // Use Variable-level sum to preserve the autograd chain.
            auto gb = ::tenzor::sum(grad_out_var, 0, false);  // sum over batch
            gb = ::tenzor::sum(gb, 1, false);                  // sum over H (was dim 2, now 1)
            gb = ::tenzor::sum(gb, 1, false);                  // sum over W (was dim 3, now 1)
            return {grad_input, grad_weight, gb};
        }
        return {grad_input, grad_weight};
    }

    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return false; }

private:
    int64_t stride_h_, stride_w_;
    int64_t padding_h_, padding_w_;
    int64_t dilation_h_, dilation_w_;
    int64_t groups_;
};

// ============================================================================
// Conv2d Implementation
// ============================================================================

Conv2d::Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
              int64_t stride, int64_t padding, int64_t dilation,
              int64_t groups, bool bias)
    : Conv2d(in_channels, out_channels,
             {kernel_size, kernel_size}, {stride, stride},
             {padding, padding}, {dilation, dilation},
             groups, bias) {}

Conv2d::Conv2d(int64_t in_channels, int64_t out_channels,
              std::pair<int64_t, int64_t> kernel_size,
              std::pair<int64_t, int64_t> stride,
              std::pair<int64_t, int64_t> padding,
              std::pair<int64_t, int64_t> dilation,
              int64_t groups, bool bias)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_h_(kernel_size.first), kernel_w_(kernel_size.second),
      stride_h_(stride.first), stride_w_(stride.second),
      padding_h_(padding.first), padding_w_(padding.second),
      dilation_h_(dilation.first), dilation_w_(dilation.second),
      groups_(groups <= 0 ? 1 : groups) {

    if (groups_ <= 0) {
        groups_ = 1;
    }
    if (in_channels % groups_ != 0) {
        throw std::invalid_argument("in_channels must be divisible by groups");
    }
    if (out_channels % groups_ != 0) {
        throw std::invalid_argument("out_channels must be divisible by groups");
    }

    std::vector<int64_t> weight_shape = {out_channels, in_channels / groups_, kernel_h_, kernel_w_};
    int64_t fan_in = (in_channels / groups_) * kernel_h_ * kernel_w_;
    float std_init = std::sqrt(2.0f / fan_in);
    auto weight_tensor = randn(weight_shape) * std_init;
    auto weight_init = Variable(weight_tensor, true);
    register_parameter("weight", weight_init);

    if (bias) {
        // PyTorch convention (and the sibling Conv1d/ConvTranspose/Deformable
        // layers): bias ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)). The previous
        // zero-init diverged from every other conv layer and from PyTorch.
        std::vector<int64_t> bias_shape = {out_channels};
        float bound = 1.0f / std::sqrt(static_cast<float>(fan_in));
        auto bias_tensor = (rand(bias_shape) * 2.0f * bound) - bound;
        auto bias_init = Variable(bias_tensor, true);
        register_parameter("bias", bias_init);
    }
}

auto Conv2d::forward_impl(const Variable& input_orig) -> Variable {
    auto input_shape = input_orig.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("Conv2d expects 4D input [batch, channels, height, width]");
    }

    int64_t in_channels = input_shape[1];

    if (in_channels != in_channels_) {
        throw std::invalid_argument("Input channels mismatch");
    }

    // F.4: all backends now read per-axis padding (AttrKey::PaddingH /
    // AttrKey::PaddingW) and thread it through to their conv kernel
    // descriptors. CUDA: cuDNN per-axis. ROCm: MIOpen per-axis. OneAPI:
    // oneDNN per-axis. Vulkan: shader push-constants per-axis. CPU: native.
    // We pass (padding_h_, padding_w_) directly into the forward attrs.
    const Variable& input = input_orig;

    int64_t height = input_shape[2];
    int64_t width = input_shape[3];
    int64_t out_h = calculate_output_size(height, kernel_h_, stride_h_, padding_h_, dilation_h_);
    int64_t out_w = calculate_output_size(width, kernel_w_, stride_w_, padding_w_, dilation_w_);

    if (out_h <= 0 || out_w <= 0) {
        throw std::runtime_error(
            "Invalid Conv2d configuration: output dimensions are non-positive (out_h=" +
            std::to_string(out_h) + ", out_w=" + std::to_string(out_w) + ")"
        );
    }

    auto& weight = *parameters_["weight"];
    auto bias_it = parameters_.find("bias");
    Device original_device = input.tensor().device();

    // Handle dtype and device mismatch: convert weight and bias to input's dtype and device if needed
    Variable weight_matched = variable_cast(weight, input.dtype());
    if (input.tensor().device().type != weight.tensor().device().type) {
        tenzor::utils::wrap_preserving_grad(weight_matched, weight_matched.tensor().to(original_device));
    }

    const Tensor* bias_ptr = nullptr;
    Variable bias_matched;
    if (bias_it != parameters_.end()) {
        auto& bias = *bias_it->second;
        bias_matched = variable_cast(bias, input.dtype());
        if (input.tensor().device().type != bias.tensor().device().type) {
            tenzor::utils::wrap_preserving_grad(bias_matched, bias_matched.tensor().to(original_device));
        }
        bias_ptr = &bias_matched.tensor();
    }

    Tensor output;

    // Use backend dispatcher (routes to CUDA/cuDNN/CPU automatically based on tensor device)
    std::vector<Tensor> inputs_vec = {input.tensor(), weight_matched.tensor()};
    if (bias_ptr != nullptr) {
        inputs_vec.push_back(*bias_ptr);
    }

    // Pass both paired and single-value keys for backward compat with backends.
    // The real per-axis padding is threaded straight through; pair-aware
    // backends read PaddingH/PaddingW while legacy ones read the scalar Padding.
    NewOpAttributes forward_attrs;
    forward_attrs.set(AttrKey::Stride, stride_h_);
    forward_attrs.set(AttrKey::Padding, padding_h_);
    forward_attrs.set(AttrKey::Dilation, dilation_h_);
    forward_attrs.set(AttrKey::StrideH, stride_h_);
    forward_attrs.set(AttrKey::StrideW, stride_w_);
    forward_attrs.set(AttrKey::PaddingH, padding_h_);
    forward_attrs.set(AttrKey::PaddingW, padding_w_);
    forward_attrs.set(AttrKey::DilationH, dilation_h_);
    forward_attrs.set(AttrKey::DilationW, dilation_w_);
    forward_attrs.set(AttrKey::Groups, groups_);
    // Kernel size is informational for the conv kernels (they read it from the
    // weight shape) but the JIT tracer / ONNX exporter need it as an attribute
    // to emit a faithful Conv node (kernel_shape).
    forward_attrs.set(AttrKey::KernelSizeH, kernel_h_);
    forward_attrs.set(AttrKey::KernelSizeW, kernel_w_);
    DType original_dtype = input.dtype();
    auto output_result = dispatch_to_device(OpId::Conv2dForward, input.tensor().device().type,
        inputs_vec, forward_attrs);
    output = output_result[0];
    if (output.dtype() != original_dtype) {
        output = output.to(original_dtype);
    }

    auto result = Variable(output, input.requires_grad() || weight.requires_grad());

    if (input.requires_grad() || weight.requires_grad()) {
        std::vector<Tensor> tensors_to_save;
        if (bias_ptr != nullptr) {
            tensors_to_save = {input.tensor(), weight_matched.tensor(), *bias_ptr};
        } else {
            tensors_to_save = {input.tensor(), weight_matched.tensor()};
        }

        auto backward_fn = std::make_shared<Conv2dBackward>(
            stride_h_, stride_w_, padding_h_, padding_w_,
            dilation_h_, dilation_w_, groups_, std::move(tensors_to_save)
        );

        // Save Variables for higher-order gradient support (create_graph=true)
        if (::tenzor::is_creating_graph()) {
            std::vector<Variable> vars_to_save = {input, weight_matched};
            if (bias_it != parameters_.end()) {
                vars_to_save.push_back(bias_matched);
            }
            backward_fn->save_variables_for_backward(std::move(vars_to_save));
        }

        result.set_grad_fn(backward_fn);

        std::vector<Variable> input_vars = {input, *parameters_["weight"]};
        if (bias_it != parameters_.end()) {
            input_vars.push_back(*bias_it->second);
        }
        backward_fn->set_input_variables(input_vars);

        // next_functions must have one slot per input variable in the SAME
        // order as input_vars so the engine's positional routing reaches
        // non-leaf weight/bias gradients (grad_fn() is nullptr for leaves).
        std::vector<std::shared_ptr<Function>> next_funcs;
        next_funcs.reserve(input_vars.size());
        for (const auto& var : input_vars) {
            next_funcs.push_back(var.grad_fn());
        }
        backward_fn->set_next_functions(next_funcs);
    }

    return result;
}

auto Conv2d::reset_parameters() -> void {
    int64_t fan_in = in_channels_ / groups_ * kernel_h_ * kernel_w_;
    float std = std::sqrt(2.0f / fan_in);

    std::vector<int64_t> weight_shape = {out_channels_, in_channels_ / groups_, kernel_h_, kernel_w_};
    auto new_weight_tensor = randn(weight_shape) * std;
    parameters_["weight"] = std::make_shared<Variable>(new_weight_tensor, true);

    auto bias_it = parameters_.find("bias");
    if (bias_it != parameters_.end()) {
        // Match the constructor (and every sibling conv layer): bias ~
        // U(-1/sqrt(fan_in), 1/sqrt(fan_in)). Zero-init contradicted the
        // constructor and broke symmetry-breaking parity after a re-init.
        std::vector<int64_t> bias_shape = {out_channels_};
        float bound = 1.0f / std::sqrt(static_cast<float>(fan_in));
        auto bias_tensor = (rand(bias_shape) * 2.0f * bound) - bound;
        bias_it->second = std::make_shared<Variable>(bias_tensor, true);
    }
}

// ============================================================================
// Conv1d Implementation (similar structure with 1D operations)
// ============================================================================

// ============================================================================
// Conv1dGradWeightBackward - JIT-R067 (see Conv2dGradWeightBackward above for
// the full derivation; identical bilinear structure, 1D functional entry
// points). d(Loss)/d(grad_out) = conv1d(input, G, ...); d(Loss)/d(input) =
// conv_transpose1d(grad_out, G, ...). Uses the ORIGINAL (unpadded) input_var
// and the real padding_ attribute via the native functional::conv1d /
// conv_transpose1d entry points, rather than replaying the manual
// pad_1d+unsqueeze-to-4D shoehorn the tensor-level kernel dispatch uses for
// computing grad_weight's forward VALUE — that shoehorn exists only because
// the raw backend kernel needs a 4D shape, not because the math needs it.
// ============================================================================
class Conv1dGradWeightBackward : public Function {
public:
    Conv1dGradWeightBackward(int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                              std::vector<Tensor> tensors_to_save)
        : stride_(stride), padding_(padding), dilation_(dilation), groups_(groups) {
        save_for_backward(std::move(tensors_to_save));
    }

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("Conv1dGradWeightBackward::forward should not be called");
    }

    // input_variables()/next_functions() order is {grad_out_var, input_var}.
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        const Tensor& G = grad_outputs[0];
        const Tensor& grad_out_t = saved_tensors_[0];
        const Tensor& input_t = saved_tensors_[1];

        Variable Gv(G, false);
        Variable d_grad_out = ::tenzor::nn::functional::conv1d(
            Variable(input_t, false), Gv, std::nullopt, stride_, padding_, dilation_, groups_);
        Variable d_input = ::tenzor::nn::functional::conv_transpose1d(
            Variable(grad_out_t, false), Gv, std::nullopt, stride_, padding_,
            /*output_padding=*/0, groups_, dilation_);
        return {d_grad_out.tensor(), d_input.tensor()};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        Variable G = grad_outputs[0];
        Variable grad_out_var, input_var;
        if (has_saved_variables() && saved_variables_.size() >= 2) {
            grad_out_var = saved_variables_[0];
            input_var = saved_variables_[1];
        } else {
            grad_out_var = Variable(saved_tensors_[0], false);
            input_var = Variable(saved_tensors_[1], false);
        }
        Variable d_grad_out = ::tenzor::nn::functional::conv1d(
            input_var, G, std::nullopt, stride_, padding_, dilation_, groups_);
        Variable d_input = ::tenzor::nn::functional::conv_transpose1d(
            grad_out_var, G, std::nullopt, stride_, padding_, /*output_padding=*/0, groups_, dilation_);
        return {d_grad_out, d_input};
    }

    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return false; }

private:
    int64_t stride_, padding_, dilation_, groups_;
};

class Conv1dBackward : public Function {
public:
    Conv1dBackward(int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                   std::vector<Tensor> tensors_to_save)
        : stride_(stride), padding_(padding), dilation_(dilation), groups_(groups) {
        save_for_backward(std::move(tensors_to_save));
    }

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("Conv1dBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        const Tensor& grad_output = grad_outputs[0];
        const Tensor& input = saved_tensors_[0];
        const Tensor& weight = saved_tensors_[1];

        // Manually pad the input in the length dimension (same as forward)
        Tensor input_padded = input;
        if (padding_ > 0) {
            input_padded = pad_1d(input, padding_);
        }

        // Add height dimension of 1 to convert to 4D for Conv2d operations
        auto grad_4d = grad_output.unsqueeze(2);
        auto input_4d = input_padded.unsqueeze(2);
        auto weight_4d = weight.unsqueeze(2);

        bool has_bias = saved_tensors_.size() > 2;

        OpAttributes backward_attrs;
        backward_attrs.set(AttrKey::Stride, stride_);
        backward_attrs.set(AttrKey::Padding, static_cast<int64_t>(0));
        backward_attrs.set(AttrKey::Dilation, dilation_);
        backward_attrs.set(AttrKey::Groups, groups_);

        // Set 4D shape strings required by backward kernels
        {
            auto is = input_4d.shape();
            std::string is_str;
            for (size_t i = 0; i < is.size(); ++i) {
                if (i > 0) is_str += ',';
                is_str += std::to_string(is[i]);
            }
            backward_attrs.set(AttrKey::InputShape, is_str);

            auto ws = weight_4d.shape();
            std::string ws_str;
            for (size_t i = 0; i < ws.size(); ++i) {
                if (i > 0) ws_str += ',';
                ws_str += std::to_string(ws[i]);
            }
            backward_attrs.set(AttrKey::WeightShape, ws_str);
        }

        std::vector<Tensor> backward_inputs = {grad_4d, input_4d, weight_4d};

        // Use OpId-based dispatch for each gradient component
        auto grad_input_result = dispatch(OpId::Conv2dBackwardInput, backward_inputs, backward_attrs);
        auto grad_weight_result = dispatch(OpId::Conv2dBackwardWeight, backward_inputs, backward_attrs);

        // Squeeze height dimension: [N,C,1,L] -> [N,C,L]
        Tensor grad_input_padded = grad_input_result[0].squeeze(2);
        Tensor grad_weight = grad_weight_result[0].squeeze(2);

        // Remove padding from grad_input to match original input shape
        Tensor grad_input = grad_input_padded;
        if (padding_ > 0) {
            int64_t length = input.shape()[2];
            // Slice to remove padding: [N, C, L + 2*padding] -> [N, C, L]
            grad_input = grad_input_padded.slice(2, padding_, padding_ + length);
        }

        if (has_bias) {
            auto grad_bias_result = dispatch(OpId::Conv2dBackwardBias, backward_inputs, backward_attrs);
            return {grad_input, grad_weight, grad_bias_result[0]};
        }
        return {grad_input, grad_weight};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        // Higher-order gradient support for Conv1d (mirrors Conv2dBackward pattern).
        Variable grad_out_var = grad_outputs[0];

        Variable input_var, weight_var;
        if (has_saved_variables() && saved_variables_.size() >= 2) {
            input_var = saved_variables_[0];
            weight_var = saved_variables_[1];
        } else {
            input_var = Variable(saved_tensors_[0], false);
            weight_var = Variable(saved_tensors_[1], false);
        }
        bool has_bias = saved_tensors_.size() > 2;

        // grad_input via conv_transpose1d (preserves computation graph)
        //
        // H23: for stride > 1 the transpose is ambiguous — several input
        // lengths map to the same conv output length, so output_padding=0
        // produces an input up to (stride-1) shorter than the true input.
        // Recover the exact output_padding from the SAVED input length,
        // mirroring the already-fixed Conv2dBackward/Conv3dBackward. With
        // out = floor((in + 2*pad - dilation*(k-1) - 1)/stride) + 1, the
        // transpose base length (output_padding=0) is
        //   base = (out-1)*stride - 2*pad + dilation*(k-1) + 1
        // and output_padding = in - base (clamped to [0, stride-1]).
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
        int64_t out_pad = 0;
        if (in_shape.size() == 3 && go_shape.size() == 3 && w_shape.size() == 3) {
            out_pad = compute_output_padding(in_shape[2], go_shape[2],
                                             stride_, padding_,
                                             dilation_, w_shape[2]);
        }
        auto grad_input = ::tenzor::nn::functional::conv_transpose1d(
            grad_out_var, weight_var,
            std::nullopt,
            {stride_}, {padding_}, {out_pad}, groups_, {dilation_});

        // grad_weight via dispatch (unsqueeze to 4D, use Conv2dBackwardWeight)
        Tensor grad_4d = grad_out_var.tensor().unsqueeze(2);
        Tensor input_padded = input_var.tensor();
        if (padding_ > 0) {
            input_padded = pad_1d(input_padded, padding_);
        }
        Tensor input_4d = input_padded.unsqueeze(2);
        Tensor weight_4d = weight_var.tensor().unsqueeze(2);

        OpAttributes backward_attrs;
        backward_attrs.set(AttrKey::Stride, stride_);
        backward_attrs.set(AttrKey::Padding, static_cast<int64_t>(0));
        backward_attrs.set(AttrKey::Dilation, dilation_);
        backward_attrs.set(AttrKey::Groups, groups_);

        // Set 4D shape strings required by conv2d_backward_weight_kernel
        // (mirrors this class's own backward() above, and the equivalent
        // Conv2dBackward::backward_with_variables setup) -- without these,
        // the kernel reads an empty weight_shape/input_shape and crashes
        // via an out-of-bounds vector access. This path was previously
        // unexercised (no create_graph=true test existed for nn::Conv1d).
        {
            auto is = input_4d.shape();
            std::string is_str;
            for (size_t i = 0; i < is.size(); ++i) {
                if (i > 0) is_str += ',';
                is_str += std::to_string(is[i]);
            }
            backward_attrs.set(AttrKey::InputShape, is_str);

            auto ws = weight_4d.shape();
            std::string ws_str;
            for (size_t i = 0; i < ws.size(); ++i) {
                if (i > 0) ws_str += ',';
                ws_str += std::to_string(ws[i]);
            }
            backward_attrs.set(AttrKey::WeightShape, ws_str);
        }

        std::vector<Tensor> gw_inputs = {grad_4d, input_4d, weight_4d};
        auto grad_weight_t = dispatch(OpId::Conv2dBackwardWeight, gw_inputs, backward_attrs)[0];

        // JIT-R067: attach a differentiable grad_fn (Conv1dGradWeightBackward,
        // defined above) — see Conv2dGradWeightBackward's comment for the
        // full derivation. Closes the 3rd-order-gradient gap on the
        // weight-gradient path.
        bool gw_requires_grad = ::tenzor::is_grad_enabled() &&
            (grad_out_var.requires_grad() || input_var.requires_grad());
        Variable grad_weight(grad_weight_t.squeeze(2), gw_requires_grad);
        if (gw_requires_grad) {
            auto gw_fn = std::make_shared<Conv1dGradWeightBackward>(
                stride_, padding_, dilation_, groups_,
                std::vector<Tensor>{grad_out_var.tensor(), input_var.tensor()});
            if (::tenzor::is_creating_graph()) {
                gw_fn->save_variables_for_backward({grad_out_var, input_var});
            }
            gw_fn->set_next_functions({grad_out_var.grad_fn(), input_var.grad_fn()});
            gw_fn->set_input_variables({grad_out_var, input_var});
            grad_weight.set_grad_fn(gw_fn);
        }

        if (has_bias) {
            auto gb = ::tenzor::sum(grad_out_var, 0, false);  // sum over batch
            gb = ::tenzor::sum(gb, 1, false);                  // sum over L
            return {grad_input, grad_weight, gb};
        }
        return {grad_input, grad_weight};
    }

    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return false; }

private:
    int64_t stride_;
    int64_t padding_;
    int64_t dilation_;
    int64_t groups_;
};

Conv1d::Conv1d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
               int64_t stride, int64_t padding, int64_t dilation,
               int64_t groups, bool bias)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride),
      padding_(padding), dilation_(dilation), groups_(groups) {

    if (in_channels % groups != 0) {
        throw std::invalid_argument("in_channels must be divisible by groups");
    }
    if (out_channels % groups != 0) {
        throw std::invalid_argument("out_channels must be divisible by groups");
    }

    std::vector<int64_t> weight_shape = {out_channels, in_channels / groups, kernel_size};
    int64_t fan_in = (in_channels / groups) * kernel_size;
    float std_init = std::sqrt(2.0f / fan_in);
    auto weight_tensor = randn(weight_shape) * std_init;
    auto weight_var = Variable(weight_tensor, true);
    register_parameter("weight", weight_var);

    if (bias) {
        std::vector<int64_t> bias_shape = {out_channels};
        float bound = 1.0f / std::sqrt(static_cast<float>(fan_in));
        auto bias_tensor = (rand(bias_shape) * 2.0f * bound) - bound;
        auto bias_var = Variable(bias_tensor, true);
        register_parameter("bias", bias_var);
    }
}

auto Conv1d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 3) {
        throw std::invalid_argument("Conv1d expects 3D input [batch, channels, length]");
    }

    int64_t in_channels = input_shape[1];
    int64_t length = input_shape[2];

    if (in_channels != in_channels_) {
        throw std::invalid_argument("Input channels mismatch");
    }

    int64_t out_len = calculate_output_size(length, kernel_size_, stride_, padding_, dilation_);
    if (out_len <= 0) {
        throw std::runtime_error(
            "Invalid Conv1d configuration: output length is non-positive (out_len=" +
            std::to_string(out_len) + ")"
        );
    }

    auto& weight = *parameters_["weight"];
    auto bias_it = parameters_.find("bias");
    Device original_device = input.tensor().device();

    // Handle dtype and device mismatch for weight. H23 follow-up: this used
    // to promote via a raw Tensor (weight.tensor().to(...)), which (a) is
    // never wired back into a differentiable Variable, so it cannot be
    // saved for higher-order gradients, and (b) was what got saved into
    // tensors_to_save below -- but only when NO conversion was needed did
    // that saved tensor actually match the device backend_with_variables'
    // grad_weight dispatch expects; whenever weight lived on a different
    // device than input (the common case for a freshly-constructed module
    // never explicitly moved via .to()), backward_with_variables crashed
    // with a device-mismatch (or, pre-H23, out-of-bounds) failure the first
    // time create_graph=true exercised it. variable_cast + wrap_preserving_grad
    // (the pattern Conv2d/Conv3d already use) keeps the promotion
    // differentiable AND ensures the promoted, device/dtype-matched tensor
    // is what's actually saved for backward.
    Variable weight_matched = variable_cast(weight, input.dtype());
    if (input.tensor().device().type != weight.tensor().device().type) {
        tenzor::utils::wrap_preserving_grad(weight_matched, weight_matched.tensor().to(original_device));
    }

    const Tensor* bias_ptr = nullptr;
    Variable bias_matched;
    if (bias_it != parameters_.end()) {
        auto& bias = *bias_it->second;
        bias_matched = variable_cast(bias, input.dtype());
        if (input.tensor().device().type != bias.tensor().device().type) {
            tenzor::utils::wrap_preserving_grad(bias_matched, bias_matched.tensor().to(original_device));
        }
        bias_ptr = &bias_matched.tensor();
    }

    Tensor input_tensor = input.tensor();   // 3-D [N, C, L], unpadded
    DType original_dtype = input.dtype();

    // CC.5: depthwise fast-path dispatch surface. When `groups_ == in_channels_`
    // and a native DepthwiseConv1d kernel is registered, dispatch the
    // specialised OpId instead of routing through the generic pipeline. Its
    // throw-stub fires loudly on backends without a real kernel (intended).
    const bool depthwise_conv1d_kernel_available =
        ::tenzor::is_op_supported(
            OpId::DepthwiseConv1d, input.tensor().device().type);
    const bool eligible_depthwise =
        (groups_ == in_channels_) && (groups_ > 1);

    Tensor output;
    if (depthwise_conv1d_kernel_available && eligible_depthwise) {
        // Depthwise path keeps the proven manual-pad + unsqueeze-to-4D form.
        Tensor padded = (padding_ > 0) ? pad_1d(input_tensor, padding_) : input_tensor;
        std::vector<Tensor> dw_inputs = {traced_unsqueeze(padded, 2),
                                         traced_unsqueeze(weight_matched.tensor(), 2)};
        if (bias_ptr != nullptr) {
            dw_inputs.push_back(*bias_ptr);
        }
        NewOpAttributes dw_attrs;
        dw_attrs.set(AttrKey::Stride, stride_);
        dw_attrs.set(AttrKey::Padding, static_cast<int64_t>(0));
        dw_attrs.set(AttrKey::Dilation, dilation_);
        dw_attrs.set(AttrKey::Groups, groups_);
        auto r = dispatch(OpId::DepthwiseConv1d,
                          std::span<const Tensor>(dw_inputs), dw_attrs);
        output = traced_squeeze(r[0], 2);
    } else {
        // Native 1-D conv: dispatch OpId::Conv1dForward with 3-D operands and
        // the real padding. The backend kernel pads the length axis only
        // (per-axis), which is numerically identical to the previous
        // manual-pad + Conv2dForward(pad=0) path. Using 3-D operands keeps the
        // op faithful for JIT tracing / ONNX export (3-D weight, rank-1 attrs)
        // instead of leaking a degenerate 4-D Conv2d into the traced graph.
        std::vector<Tensor> inputs_vec = {input_tensor, weight_matched.tensor()};
        if (bias_ptr != nullptr) {
            inputs_vec.push_back(*bias_ptr);
        }
        NewOpAttributes forward_attrs;
        forward_attrs.set(AttrKey::KernelSize, kernel_size_);
        forward_attrs.set(AttrKey::Stride, stride_);
        forward_attrs.set(AttrKey::Padding, padding_);
        forward_attrs.set(AttrKey::Dilation, dilation_);
        forward_attrs.set(AttrKey::Groups, groups_);
        auto r = dispatch(OpId::Conv1dForward,
                          std::span<const Tensor>(inputs_vec), forward_attrs);
        output = r[0];
    }
    if (output.dtype() != original_dtype) {
        output = output.to(original_dtype);
    }

    auto result = Variable(output, input.requires_grad() || weight.requires_grad());

    if (input.requires_grad() || weight.requires_grad()) {
        std::vector<Tensor> tensors_to_save;
        if (bias_ptr != nullptr) {
            tensors_to_save = {input.tensor(), weight_matched.tensor(), *bias_ptr};
        } else {
            tensors_to_save = {input.tensor(), weight_matched.tensor()};
        }

        auto backward_fn = std::make_shared<Conv1dBackward>(
            stride_, padding_, dilation_, groups_, std::move(tensors_to_save)
        );

        // Save Variables for higher-order gradient support (create_graph=true).
        // Previously missing entirely on Conv1d (unlike Conv2d/Conv3d), so
        // Conv1dBackward::backward_with_variables's has_saved_variables()
        // check always fell through to the detached-Tensor fallback --
        // silently severing the second-order gradient chain back to
        // input/weight through grad_input's/grad_weight's cross-dependence,
        // the same bug class as M17-M24.
        if (::tenzor::is_creating_graph()) {
            std::vector<Variable> vars_to_save = {input, weight_matched};
            if (bias_it != parameters_.end()) {
                vars_to_save.push_back(bias_matched);
            }
            backward_fn->save_variables_for_backward(std::move(vars_to_save));
        }

        result.set_grad_fn(backward_fn);

        std::vector<Variable> input_vars = {input, *parameters_["weight"]};
        if (bias_it != parameters_.end()) {
            input_vars.push_back(*bias_it->second);
        }
        backward_fn->set_input_variables(input_vars);

        // next_functions must have one slot per input variable in the SAME
        // order as input_vars so the engine's positional routing reaches
        // non-leaf weight/bias gradients (grad_fn() is nullptr for leaves).
        std::vector<std::shared_ptr<Function>> next_funcs;
        next_funcs.reserve(input_vars.size());
        for (const auto& var : input_vars) {
            next_funcs.push_back(var.grad_fn());
        }
        backward_fn->set_next_functions(next_funcs);
    }

    return result;
}

// ============================================================================
// ConvTranspose2d Implementation
// ============================================================================

// Audit I5-followup: per-axis ConvTranspose2dBackward. Stores sH/sW, pH/pW,
// opH/opW, dH/dW; ctor with scalar args delegates with identical values.
// Both backward() and backward_with_variables() pack per-axis attrs so the
// adjoint Conv2d dispatch (Conv2dForward + Conv2dBackwardWeight via swapped
// roles) sees the anisotropic config in its registered kernel.
// ============================================================================
// ConvTranspose2dGradWeightBackward - JIT-R067 (see Conv2dGradWeightBackward
// and ConvTranspose1dGradWeightBackward above for the full derivation and
// role-swap rationale). A=input_var, B=grad_out_var:
//   d(Loss)/d(input_var)    = conv2d(grad_out_var, G, ...) [cropped to
//                              input_shape, mirroring this class's own
//                              grad_input correction]
//   d(Loss)/d(grad_out_var) = conv_transpose2d(input_var, G, ..., opH_, opW_)
//                              [defensively cropped to grad_out_var's shape]
// ============================================================================
class ConvTranspose2dGradWeightBackward : public Function {
public:
    ConvTranspose2dGradWeightBackward(int64_t sH, int64_t sW, int64_t pH, int64_t pW,
                                       int64_t opH, int64_t opW, int64_t dH, int64_t dW,
                                       int64_t groups,
                                       std::vector<Tensor> tensors_to_save)
        : sH_(sH), sW_(sW), pH_(pH), pW_(pW), opH_(opH), opW_(opW),
          dH_(dH), dW_(dW), groups_(groups) {
        save_for_backward(std::move(tensors_to_save));
    }

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("ConvTranspose2dGradWeightBackward::forward should not be called");
    }

    static auto crop_to(const Tensor& t, std::span<const int64_t> target_shape) -> Tensor {
        auto s = t.shape();
        if (s.size() != 4 || target_shape.size() != 4) return t;
        Tensor out = t;
        if (s[2] != target_shape[2]) out = tenzor::slice(out, 2, 0, target_shape[2]);
        if (s[3] != target_shape[3]) out = tenzor::slice(out, 3, 0, target_shape[3]);
        return out;
    }

    static auto crop_to_var(const Variable& v, std::span<const int64_t> target_shape) -> Variable {
        auto s = v.tensor().shape();
        if (s.size() != 4 || target_shape.size() != 4) return v;
        Variable out = v;
        if (s[2] != target_shape[2]) out = ::tenzor::narrow(out, 2, 0, target_shape[2]);
        if (s[3] != target_shape[3]) out = ::tenzor::narrow(out, 3, 0, target_shape[3]);
        return out;
    }

    // input_variables()/next_functions() order is {input_var, grad_out_var}.
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        const Tensor& G = grad_outputs[0];
        const Tensor& input_t = saved_tensors_[0];
        const Tensor& grad_out_t = saved_tensors_[1];

        Variable Gv(G, false);
        Variable d_input_var = ::tenzor::nn::functional::conv2d(
            Variable(grad_out_t, false), Gv, std::nullopt,
            {sH_, sW_}, {pH_, pW_}, {dH_, dW_}, groups_);
        Tensor d_input = crop_to(d_input_var.tensor(), input_t.shape());

        Variable d_grad_out_var = ::tenzor::nn::functional::conv_transpose2d(
            Variable(input_t, false), Gv, std::nullopt,
            {sH_, sW_}, {pH_, pW_}, {opH_, opW_}, groups_, {dH_, dW_});
        Tensor d_grad_out = crop_to(d_grad_out_var.tensor(), grad_out_t.shape());
        return {d_input, d_grad_out};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        Variable G = grad_outputs[0];
        Variable input_var, grad_out_var;
        if (has_saved_variables() && saved_variables_.size() >= 2) {
            input_var = saved_variables_[0];
            grad_out_var = saved_variables_[1];
        } else {
            input_var = Variable(saved_tensors_[0], false);
            grad_out_var = Variable(saved_tensors_[1], false);
        }

        Variable d_input = ::tenzor::nn::functional::conv2d(
            grad_out_var, G, std::nullopt, {sH_, sW_}, {pH_, pW_}, {dH_, dW_}, groups_);
        d_input = crop_to_var(d_input, input_var.tensor().shape());

        Variable d_grad_out = ::tenzor::nn::functional::conv_transpose2d(
            input_var, G, std::nullopt, {sH_, sW_}, {pH_, pW_}, {opH_, opW_}, groups_, {dH_, dW_});
        d_grad_out = crop_to_var(d_grad_out, grad_out_var.tensor().shape());
        return {d_input, d_grad_out};
    }

    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return false; }

private:
    int64_t sH_, sW_, pH_, pW_, opH_, opW_, dH_, dW_, groups_;
};

class ConvTranspose2dBackward : public Function {
public:
    ConvTranspose2dBackward(int64_t sH, int64_t sW,
                            int64_t pH, int64_t pW,
                            int64_t opH, int64_t opW,
                            int64_t dH, int64_t dW,
                            int64_t groups,
                            std::vector<Tensor> tensors_to_save)
        : sH_(sH), sW_(sW), pH_(pH), pW_(pW),
          opH_(opH), opW_(opW), dH_(dH), dW_(dW), groups_(groups) {
        save_for_backward(std::move(tensors_to_save));
    }

    // Scalar ctor — delegates to per-axis with identical H/W values.
    ConvTranspose2dBackward(int64_t stride, int64_t padding, int64_t output_padding,
                            int64_t dilation, int64_t groups,
                            std::vector<Tensor> tensors_to_save)
        : ConvTranspose2dBackward(stride, stride, padding, padding,
                                  output_padding, output_padding,
                                  dilation, dilation, groups,
                                  std::move(tensors_to_save)) {}

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("ConvTranspose2dBackward::forward should not be called");
    }

    // Helper to pack per-axis attrs (with scalar fallback) for Conv2d adjoint
    // dispatch. The Conv2d backend (E1-E5) reads StrideH/W, PaddingH/W,
    // DilationH/W with scalar fallback.
    void pack_conv2d_attrs(NewOpAttributes& a) const {
        a.set(AttrKey::Stride,    sH_);
        a.set(AttrKey::Padding,   pH_);
        a.set(AttrKey::Dilation,  dH_);
        a.set(AttrKey::StrideH,   sH_);
        a.set(AttrKey::StrideW,   sW_);
        a.set(AttrKey::PaddingH,  pH_);
        a.set(AttrKey::PaddingW,  pW_);
        a.set(AttrKey::DilationH, dH_);
        a.set(AttrKey::DilationW, dW_);
        a.set(AttrKey::Groups,    groups_);
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        const Tensor& grad_output = grad_outputs[0];
        const Tensor& input = saved_tensors_[0];
        const Tensor& weight = saved_tensors_[1];
        bool has_bias = saved_tensors_.size() > 2;

        auto input_shape = input.shape();
        auto weight_shape = weight.shape();

        // grad_input via Conv2d adjoint.
        NewOpAttributes conv_attrs;
        pack_conv2d_attrs(conv_attrs);
        std::vector<Tensor> conv_inputs = {grad_output, weight};
        Tensor grad_input = dispatch(OpId::Conv2dForward,
            std::span<const Tensor>(conv_inputs), conv_attrs)[0];

        // Slice if output_padding caused a shape difference.
        if (!std::equal(grad_input.shape().begin(), grad_input.shape().end(),
                        input_shape.begin(), input_shape.end())) {
            auto gi_shape = grad_input.shape();
            if (gi_shape.size() == 4 && input_shape.size() == 4 &&
                gi_shape[0] == input_shape[0] && gi_shape[1] == input_shape[1]) {
                grad_input = tenzor::slice(grad_input, 2, 0, input_shape[2]);
                grad_input = tenzor::slice(grad_input, 3, 0, input_shape[3]);
            }
        }

        // grad_weight via Conv2dBackwardWeight with role-swapped inputs.
        std::string ws_str = std::to_string(weight_shape[0]) + "," +
                             std::to_string(weight_shape[1]) + "," +
                             std::to_string(weight_shape[2]) + "," +
                             std::to_string(weight_shape[3]);
        NewOpAttributes weight_grad_attrs;
        pack_conv2d_attrs(weight_grad_attrs);
        weight_grad_attrs.set(AttrKey::WeightShape, std::string_view(ws_str));

        std::vector<Tensor> weight_grad_inputs = {input, grad_output, weight};
        Tensor grad_weight = dispatch(OpId::Conv2dBackwardWeight,
            std::span<const Tensor>(weight_grad_inputs), weight_grad_attrs)[0];

        if (has_bias) {
            Tensor grad_bias = tenzor::sum(tenzor::sum(tenzor::sum(grad_output, 0, false), 1, false), 1, false);
            return {grad_input, grad_weight, grad_bias};
        }
        return {grad_input, grad_weight};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        Variable grad_out_var = grad_outputs[0];

        Variable input_var, weight_var;
        if (has_saved_variables() && saved_variables_.size() >= 2) {
            input_var = saved_variables_[0];
            weight_var = saved_variables_[1];
        } else {
            input_var = Variable(saved_tensors_[0], false);
            weight_var = Variable(saved_tensors_[1], false);
        }
        bool has_bias = saved_tensors_.size() > 2;

        // grad_input: regular conv2d with per-axis params.
        auto grad_input = ::tenzor::nn::functional::conv2d(
            grad_out_var, weight_var,
            std::nullopt,
            {sH_, sW_},
            {pH_, pW_},
            {dH_, dW_},
            groups_);

        auto input_shape = input_var.tensor().shape();
        auto gi_shape = grad_input.tensor().shape();
        if (gi_shape.size() == 4 && input_shape.size() == 4 &&
            gi_shape[0] == input_shape[0] && gi_shape[1] == input_shape[1]) {
            if (gi_shape[2] != input_shape[2] || gi_shape[3] != input_shape[3]) {
                auto sliced = tenzor::slice(grad_input.tensor(), 2, 0, input_shape[2]);
                sliced = tenzor::slice(sliced, 3, 0, input_shape[3]);
                grad_input = Variable(sliced, grad_out_var.requires_grad());
            }
        }

        auto weight_shape = weight_var.tensor().shape();
        std::string ws_str = std::to_string(weight_shape[0]) + "," +
                             std::to_string(weight_shape[1]) + "," +
                             std::to_string(weight_shape[2]) + "," +
                             std::to_string(weight_shape[3]);
        NewOpAttributes weight_grad_attrs;
        pack_conv2d_attrs(weight_grad_attrs);
        weight_grad_attrs.set(AttrKey::WeightShape, std::string_view(ws_str));

        std::vector<Tensor> weight_grad_inputs = {input_var.tensor(), grad_out_var.tensor(), weight_var.tensor()};
        auto grad_weight_t = dispatch(OpId::Conv2dBackwardWeight,
            std::span<const Tensor>(weight_grad_inputs), weight_grad_attrs)[0];

        // JIT-R067: attach a differentiable grad_fn
        // (ConvTranspose2dGradWeightBackward, defined above) so a 3rd-order
        // gradient flows correctly back to input_var/grad_out_var instead of
        // silently stopping here.
        bool gw_requires_grad = ::tenzor::is_grad_enabled() &&
            (input_var.requires_grad() || grad_out_var.requires_grad());
        Variable grad_weight(grad_weight_t, gw_requires_grad);
        if (gw_requires_grad) {
            auto gw_fn = std::make_shared<ConvTranspose2dGradWeightBackward>(
                sH_, sW_, pH_, pW_, opH_, opW_, dH_, dW_, groups_,
                std::vector<Tensor>{input_var.tensor(), grad_out_var.tensor()});
            if (::tenzor::is_creating_graph()) {
                gw_fn->save_variables_for_backward({input_var, grad_out_var});
            }
            gw_fn->set_next_functions({input_var.grad_fn(), grad_out_var.grad_fn()});
            gw_fn->set_input_variables({input_var, grad_out_var});
            grad_weight.set_grad_fn(gw_fn);
        }

        if (has_bias) {
            auto gb = ::tenzor::sum(grad_out_var, 0, false);
            gb = ::tenzor::sum(gb, 1, false);
            gb = ::tenzor::sum(gb, 1, false);
            return {grad_input, grad_weight, gb};
        }
        return {grad_input, grad_weight};
    }

    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return false; }

private:
    int64_t sH_, sW_;
    int64_t pH_, pW_;
    int64_t opH_, opW_;
    int64_t dH_, dW_;
    int64_t groups_;
};

// Audit I5: per-axis ConvTranspose2d ctor.
ConvTranspose2d::ConvTranspose2d(int64_t in_channels, int64_t out_channels,
                                 std::pair<int64_t, int64_t> kernel_size,
                                 std::pair<int64_t, int64_t> stride,
                                 std::pair<int64_t, int64_t> padding,
                                 std::pair<int64_t, int64_t> output_padding,
                                 std::pair<int64_t, int64_t> dilation,
                                 int64_t groups, bool bias)
    : in_channels_(in_channels), out_channels_(out_channels),
      kH_(kernel_size.first), kW_(kernel_size.second),
      sH_(stride.first),      sW_(stride.second),
      pH_(padding.first),     pW_(padding.second),
      opH_(output_padding.first), opW_(output_padding.second),
      dH_(dilation.first),    dW_(dilation.second),
      groups_(groups) {

    if (in_channels % groups != 0)  throw std::invalid_argument("in_channels must be divisible by groups");
    if (out_channels % groups != 0) throw std::invalid_argument("out_channels must be divisible by groups");
    if (opH_ >= sH_ && opH_ != 0)   throw std::invalid_argument("output_padding (H) must be smaller than stride (H)");
    if (opW_ >= sW_ && opW_ != 0)   throw std::invalid_argument("output_padding (W) must be smaller than stride (W)");

    std::vector<int64_t> weight_shape = {in_channels, out_channels / groups, kH_, kW_};
    // Transposed-conv weight is (in, out/groups, kH, kW); its fan_in is the
    // number of input connections per output element = (out/groups)*kH*kW, NOT
    // in_channels*kH*kW (that is fan_out). Matches PyTorch's
    // _calculate_fan_in_and_fan_out on weight.size(1).
    int64_t fan_in = (out_channels / groups) * kH_ * kW_;
    float std_init = std::sqrt(2.0f / fan_in);
    auto weight_tensor = randn(weight_shape) * std_init;
    register_parameter("weight", Variable(weight_tensor, true));

    if (bias) {
        std::vector<int64_t> bias_shape = {out_channels};
        float bound = 1.0f / std::sqrt(static_cast<float>(fan_in));
        auto bias_tensor = (rand(bias_shape) * 2.0f * bound) - bound;
        register_parameter("bias", Variable(bias_tensor, true));
    }
}

// Scalar ctor — delegates to per-axis with identical H/W values.
ConvTranspose2d::ConvTranspose2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
                                 int64_t stride, int64_t padding, int64_t output_padding,
                                 int64_t groups, bool bias)
    : ConvTranspose2d(in_channels, out_channels,
                      std::make_pair(kernel_size, kernel_size),
                      std::make_pair(stride, stride),
                      std::make_pair(padding, padding),
                      std::make_pair(output_padding, output_padding),
                      std::make_pair<int64_t, int64_t>(1, 1),  // dilation default
                      groups, bias) {}

auto ConvTranspose2d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("ConvTranspose2d expects 4D input [batch, channels, height, width]");
    }

    int64_t in_channels = input_shape[1];
    if (in_channels != in_channels_) {
        throw std::invalid_argument("Input channels mismatch");
    }

    auto& weight = *parameters_["weight"];
    auto bias_it = parameters_.find("bias");
    Device original_device = input.tensor().device();

    // Handle dtype and device mismatch
    Variable weight_matched = variable_cast(weight, input.dtype());
    if (input.tensor().device().type != weight.tensor().device().type) {
        tenzor::utils::wrap_preserving_grad(weight_matched, weight_matched.tensor().to(original_device));
    }

    const Tensor* bias_ptr = nullptr;
    Variable bias_matched;
    if (bias_it != parameters_.end()) {
        auto& bias = *bias_it->second;
        bias_matched = variable_cast(bias, input.dtype());
        if (input.tensor().device().type != bias.tensor().device().type) {
            tenzor::utils::wrap_preserving_grad(bias_matched, bias_matched.tensor().to(original_device));
        }
        bias_ptr = &bias_matched.tensor();
    }

    Tensor output;

    std::vector<Tensor> inputs_vec = {input.tensor(), weight_matched.tensor()};
    if (bias_ptr != nullptr) {
        inputs_vec.push_back(*bias_ptr);
    }

    // Audit I5: pack per-axis attrs + scalar fallback. Both H/W are forwarded
    // so backends that haven't been updated to per-axis still get a sensible
    // (D-axis) scalar value.
    NewOpAttributes forward_attrs;
    forward_attrs.set(AttrKey::Stride,         sH_);
    forward_attrs.set(AttrKey::Padding,        pH_);
    forward_attrs.set(AttrKey::OutputPadding,  opH_);
    forward_attrs.set(AttrKey::Dilation,       dH_);
    forward_attrs.set(AttrKey::KernelSizeH,    kH_);
    forward_attrs.set(AttrKey::KernelSizeW,    kW_);
    forward_attrs.set(AttrKey::StrideH,        sH_);
    forward_attrs.set(AttrKey::StrideW,        sW_);
    forward_attrs.set(AttrKey::PaddingH,       pH_);
    forward_attrs.set(AttrKey::PaddingW,       pW_);
    forward_attrs.set(AttrKey::OutputPaddingH, opH_);
    forward_attrs.set(AttrKey::OutputPaddingW, opW_);
    forward_attrs.set(AttrKey::DilationH,      dH_);
    forward_attrs.set(AttrKey::DilationW,      dW_);
    forward_attrs.set(AttrKey::Groups,         groups_);
    DType original_dtype = input.dtype();
    auto output_result = dispatch(OpId::ConvTranspose2dForward,
        std::span<const Tensor>(inputs_vec),
        forward_attrs);
    output = output_result[0];
    if (output.dtype() != original_dtype) {
        output = output.to(original_dtype);
    }

    auto result = Variable(output, input.requires_grad() || weight.requires_grad());

    // Set up autograd if needed
    if (input.requires_grad() || weight.requires_grad()) {
        std::vector<Tensor> tensors_to_save;
        if (bias_ptr != nullptr) {
            tensors_to_save = {input.tensor(), weight_matched.tensor(), *bias_ptr};
        } else {
            tensors_to_save = {input.tensor(), weight_matched.tensor()};
        }

        // Audit I5-followup: per-axis ConvTranspose2dBackward (the H/W ctor
        // overload). Asymmetric ConvT2d training now flows per-axis through
        // the adjoint Conv2d dispatch.
        auto backward_fn = std::make_shared<ConvTranspose2dBackward>(
            sH_, sW_, pH_, pW_, opH_, opW_, dH_, dW_, groups_,
            std::move(tensors_to_save)
        );

        result.set_grad_fn(backward_fn);

        // MUST include all inputs to maintain 1:1 index correspondence with gradients
        // The engine correctly skips variables that don't require grad
        std::vector<Variable> input_vars = {input, *parameters_["weight"]};
        if (bias_it != parameters_.end()) {
            input_vars.push_back(*bias_it->second);
        }
        backward_fn->set_input_variables(input_vars);

        // next_functions must have one slot per input variable in the SAME
        // order as input_vars so the engine's positional routing reaches
        // non-leaf weight/bias gradients (grad_fn() is nullptr for leaves).
        std::vector<std::shared_ptr<Function>> next_funcs;
        next_funcs.reserve(input_vars.size());
        for (const auto& var : input_vars) {
            next_funcs.push_back(var.grad_fn());
        }
        backward_fn->set_next_functions(next_funcs);
    }

    return result;
}

auto ConvTranspose2d::reset_parameters() -> void {
    // fan_in for transposed weight (in, out/groups, kH, kW) is (out/groups)*kH*kW.
    int64_t fan_in = (out_channels_ / groups_) * kH_ * kW_;
    float std = std::sqrt(2.0f / fan_in);

    std::vector<int64_t> weight_shape = {in_channels_, out_channels_ / groups_, kH_, kW_};
    auto new_weight_tensor = randn(weight_shape) * std;
    parameters_["weight"] = std::make_shared<Variable>(new_weight_tensor, true);

    auto bias_it = parameters_.find("bias");
    if (bias_it != parameters_.end()) {
        std::vector<int64_t> bias_shape = {out_channels_};
        float bound = 1.0f / std::sqrt(static_cast<float>(fan_in));
        auto new_bias_tensor = (rand(bias_shape) * 2.0f * bound) - bound;
        bias_it->second = std::make_shared<Variable>(new_bias_tensor, true);
    }
}

// ============================================================================
// Conv3d Implementation
// ============================================================================

// Audit I5: Conv3dBackward now carries per-axis stride/padding/dilation. The
// scalar-arg ctor delegates to the per-axis ctor with identical D/H/W values
// to keep legacy callers compiling.
// ============================================================================
// Conv3dGradWeightBackward - JIT-R067 (see Conv2dGradWeightBackward above for
// the full derivation; identical bilinear structure, 3D functional entries).
// d(Loss)/d(grad_out) = conv3d(input, G, ...); d(Loss)/d(input) =
// conv_transpose3d(grad_out, G, ...) — the latter is Conv3dBackward's own
// grad_input formula with G substituted for weight.
// ============================================================================
class Conv3dGradWeightBackward : public Function {
public:
    Conv3dGradWeightBackward(int64_t sD, int64_t sH, int64_t sW,
                              int64_t pD, int64_t pH, int64_t pW,
                              int64_t dD, int64_t dH, int64_t dW,
                              int64_t groups,
                              std::vector<Tensor> tensors_to_save)
        : sD_(sD), sH_(sH), sW_(sW), pD_(pD), pH_(pH), pW_(pW),
          dD_(dD), dH_(dH), dW_(dW), groups_(groups) {
        save_for_backward(std::move(tensors_to_save));
    }

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("Conv3dGradWeightBackward::forward should not be called");
    }

    // input_variables()/next_functions() order is {grad_out_var, input_var}.
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        const Tensor& G = grad_outputs[0];
        const Tensor& grad_out_t = saved_tensors_[0];
        const Tensor& input_t = saved_tensors_[1];

        Variable Gv(G, false);
        Variable d_grad_out = ::tenzor::nn::functional::conv3d(
            Variable(input_t, false), Gv, std::nullopt,
            {sD_, sH_, sW_}, {pD_, pH_, pW_}, {dD_, dH_, dW_}, groups_);
        Variable d_input = ::tenzor::nn::functional::conv_transpose3d(
            Variable(grad_out_t, false), Gv, std::nullopt,
            {sD_, sH_, sW_}, {pD_, pH_, pW_}, {0, 0, 0}, groups_, {dD_, dH_, dW_});
        return {d_grad_out.tensor(), d_input.tensor()};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        Variable G = grad_outputs[0];
        Variable grad_out_var, input_var;
        if (has_saved_variables() && saved_variables_.size() >= 2) {
            grad_out_var = saved_variables_[0];
            input_var = saved_variables_[1];
        } else {
            grad_out_var = Variable(saved_tensors_[0], false);
            input_var = Variable(saved_tensors_[1], false);
        }
        Variable d_grad_out = ::tenzor::nn::functional::conv3d(
            input_var, G, std::nullopt, {sD_, sH_, sW_}, {pD_, pH_, pW_}, {dD_, dH_, dW_}, groups_);
        Variable d_input = ::tenzor::nn::functional::conv_transpose3d(
            grad_out_var, G, std::nullopt, {sD_, sH_, sW_}, {pD_, pH_, pW_}, {0, 0, 0}, groups_, {dD_, dH_, dW_});
        return {d_grad_out, d_input};
    }

    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return false; }

private:
    int64_t sD_, sH_, sW_, pD_, pH_, pW_, dD_, dH_, dW_, groups_;
};

class Conv3dBackward : public Function {
public:
    Conv3dBackward(int64_t sD, int64_t sH, int64_t sW,
                   int64_t pD, int64_t pH, int64_t pW,
                   int64_t dD, int64_t dH, int64_t dW,
                   int64_t groups,
                   std::vector<Tensor> tensors_to_save)
        : sD_(sD), sH_(sH), sW_(sW),
          pD_(pD), pH_(pH), pW_(pW),
          dD_(dD), dH_(dH), dW_(dW),
          groups_(groups) {
        save_for_backward(std::move(tensors_to_save));
    }

    Conv3dBackward(int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                   std::vector<Tensor> tensors_to_save)
        : Conv3dBackward(stride, stride, stride,
                         padding, padding, padding,
                         dilation, dilation, dilation,
                         groups, std::move(tensors_to_save)) {}

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("Conv3dBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        const Tensor& grad_output = grad_outputs[0];
        const Tensor& input = saved_tensors_[0];
        const Tensor& weight = saved_tensors_[1];
        bool has_bias = saved_tensors_.size() > 2;

        auto shape_to_str = [](std::span<const int64_t> s) {
            std::string r;
            for (size_t i = 0; i < s.size(); ++i) { if (i > 0) r += ","; r += std::to_string(s[i]); }
            return r;
        };

        NewOpAttributes common_attrs;
        // Set per-axis AttrKeys (the CPU registry now reads these with scalar
        // fallback, so this remains backward-compat for backends that only
        // see the scalar Stride/Padding/Dilation keys).
        common_attrs.set(AttrKey::StrideD,   sD_);
        common_attrs.set(AttrKey::StrideH,   sH_);
        common_attrs.set(AttrKey::StrideW,   sW_);
        common_attrs.set(AttrKey::PaddingD,  pD_);
        common_attrs.set(AttrKey::PaddingH,  pH_);
        common_attrs.set(AttrKey::PaddingW,  pW_);
        common_attrs.set(AttrKey::DilationD, dD_);
        common_attrs.set(AttrKey::DilationH, dH_);
        common_attrs.set(AttrKey::DilationW, dW_);
        // Scalar fallback (taken from D axis) for backends that only read scalar.
        common_attrs.set(AttrKey::Stride,    sD_);
        common_attrs.set(AttrKey::Padding,   pD_);
        common_attrs.set(AttrKey::Dilation,  dD_);
        common_attrs.set(AttrKey::Groups,    groups_);

        std::string input_shape_str = shape_to_str(input.shape());
        std::string weight_shape_str = shape_to_str(weight.shape());

        NewOpAttributes bi_attrs = common_attrs;
        bi_attrs.set(AttrKey::InputShape, std::string_view(input_shape_str));
        std::vector<Tensor> bi_inputs = {grad_output, input, weight};
        auto grad_input = dispatch<OpId::Conv3dBackwardInput>(bi_inputs, bi_attrs)[0];

        NewOpAttributes bw_attrs = common_attrs;
        bw_attrs.set(AttrKey::WeightShape, std::string_view(weight_shape_str));
        std::vector<Tensor> bw_inputs = {grad_output, input, weight};
        auto grad_weight = dispatch<OpId::Conv3dBackwardWeight>(bw_inputs, bw_attrs)[0];

        if (has_bias) {
            std::vector<Tensor> bb_inputs = {grad_output};
            auto grad_bias = dispatch<OpId::Conv3dBackwardBias>(bb_inputs)[0];
            return {grad_input, grad_weight, grad_bias};
        }
        return {grad_input, grad_weight};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        Variable grad_out_var = grad_outputs[0];

        Variable input_var, weight_var;
        if (has_saved_variables() && saved_variables_.size() >= 2) {
            input_var = saved_variables_[0];
            weight_var = saved_variables_[1];
        } else {
            input_var = Variable(saved_tensors_[0], false);
            weight_var = Variable(saved_tensors_[1], false);
        }
        bool has_bias = saved_tensors_.size() > 2;

        // H19: same output_padding-recovery bug as Conv2dBackward (H18), but
        // unconditional here — Conv3dBackward is the ONLY backward path,
        // shared by both nn::Conv3d and nn::functional::conv3d (no separate
        // fixed implementation like Conv2d's conv_autograd.hpp exists for
        // 3D). Recover the exact output_padding from the saved input
        // spatial size so create_graph=true grad_input matches the true
        // input shape for stride > 1.
        const auto& in_shape3 = input_var.tensor().shape();
        const auto& go_shape3 = grad_out_var.tensor().shape();
        const auto& w_shape3 = weight_var.tensor().shape();
        auto compute_output_padding3 = [](int64_t in_size, int64_t out_size,
                                          int64_t stride, int64_t pad,
                                          int64_t dilation, int64_t k) -> int64_t {
            int64_t base = (out_size - 1) * stride - 2 * pad + dilation * (k - 1) + 1;
            int64_t op = in_size - base;
            if (op < 0) op = 0;
            if (op > stride - 1) op = stride - 1;
            return op;
        };
        int64_t out_pad_d = 0, out_pad_h3 = 0, out_pad_w3 = 0;
        if (in_shape3.size() == 5 && go_shape3.size() == 5 && w_shape3.size() == 5) {
            out_pad_d = compute_output_padding3(in_shape3[2], go_shape3[2],
                                                sD_, pD_, dD_, w_shape3[2]);
            out_pad_h3 = compute_output_padding3(in_shape3[3], go_shape3[3],
                                                 sH_, pH_, dH_, w_shape3[3]);
            out_pad_w3 = compute_output_padding3(in_shape3[4], go_shape3[4],
                                                 sW_, pW_, dW_, w_shape3[4]);
        }
        auto grad_input = ::tenzor::nn::functional::conv_transpose3d(
            grad_out_var, weight_var,
            std::nullopt,
            {sD_, sH_, sW_},
            {pD_, pH_, pW_},
            {out_pad_d, out_pad_h3, out_pad_w3},
            groups_,
            {dD_, dH_, dW_});

        auto shape_to_str = [](std::span<const int64_t> s) {
            std::string r;
            for (size_t i = 0; i < s.size(); ++i) { if (i > 0) r += ","; r += std::to_string(s[i]); }
            return r;
        };

        std::string ws_str = shape_to_str(weight_var.tensor().shape());

        NewOpAttributes bw_attrs;
        bw_attrs.set(AttrKey::StrideD,   sD_);
        bw_attrs.set(AttrKey::StrideH,   sH_);
        bw_attrs.set(AttrKey::StrideW,   sW_);
        bw_attrs.set(AttrKey::PaddingD,  pD_);
        bw_attrs.set(AttrKey::PaddingH,  pH_);
        bw_attrs.set(AttrKey::PaddingW,  pW_);
        bw_attrs.set(AttrKey::DilationD, dD_);
        bw_attrs.set(AttrKey::DilationH, dH_);
        bw_attrs.set(AttrKey::DilationW, dW_);
        bw_attrs.set(AttrKey::Stride,    sD_);
        bw_attrs.set(AttrKey::Padding,   pD_);
        bw_attrs.set(AttrKey::Dilation,  dD_);
        bw_attrs.set(AttrKey::Groups,    groups_);
        bw_attrs.set(AttrKey::WeightShape, std::string_view(ws_str));

        std::vector<Tensor> bw_inputs = {grad_out_var.tensor(), input_var.tensor(), weight_var.tensor()};
        auto grad_weight_t = dispatch<OpId::Conv3dBackwardWeight>(bw_inputs, bw_attrs)[0];

        // JIT-R067: attach a differentiable grad_fn (Conv3dGradWeightBackward,
        // defined above) — see Conv2dGradWeightBackward's comment for the
        // full derivation. Closes the 3rd-order-gradient gap on the
        // weight-gradient path.
        bool gw_requires_grad = ::tenzor::is_grad_enabled() &&
            (grad_out_var.requires_grad() || input_var.requires_grad());
        Variable grad_weight(grad_weight_t, gw_requires_grad);
        if (gw_requires_grad) {
            auto gw_fn = std::make_shared<Conv3dGradWeightBackward>(
                sD_, sH_, sW_, pD_, pH_, pW_, dD_, dH_, dW_, groups_,
                std::vector<Tensor>{grad_out_var.tensor(), input_var.tensor()});
            if (::tenzor::is_creating_graph()) {
                gw_fn->save_variables_for_backward({grad_out_var, input_var});
            }
            gw_fn->set_next_functions({grad_out_var.grad_fn(), input_var.grad_fn()});
            gw_fn->set_input_variables({grad_out_var, input_var});
            grad_weight.set_grad_fn(gw_fn);
        }

        if (has_bias) {
            auto gb = ::tenzor::sum(grad_out_var, 0, false);
            gb = ::tenzor::sum(gb, 1, false);
            gb = ::tenzor::sum(gb, 1, false);
            gb = ::tenzor::sum(gb, 1, false);
            return {grad_input, grad_weight, gb};
        }
        return {grad_input, grad_weight};
    }

    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return false; }

private:
    int64_t sD_, sH_, sW_;
    int64_t pD_, pH_, pW_;
    int64_t dD_, dH_, dW_;
    int64_t groups_;
};

namespace internal {
auto make_conv3d_backward(int64_t stride, int64_t padding, int64_t dilation,
                          int64_t groups,
                          std::vector<::tenzor::Tensor> tensors_to_save)
    -> std::shared_ptr<::tenzor::Function> {
    return std::make_shared<Conv3dBackward>(
        stride, padding, dilation, groups, std::move(tensors_to_save));
}
auto make_conv_transpose2d_backward(int64_t stride, int64_t padding,
                                    int64_t output_padding, int64_t dilation,
                                    int64_t groups,
                                    std::vector<::tenzor::Tensor> tensors_to_save)
    -> std::shared_ptr<::tenzor::Function> {
    return std::make_shared<ConvTranspose2dBackward>(
        stride, padding, output_padding, dilation, groups,
        std::move(tensors_to_save));
}

// F127: anisotropic (per-axis) factory overloads. Conv3dBackward and
// ConvTranspose2dBackward already carry per-axis members and use them
// throughout their backward math, so F::conv3d / F::conv_transpose2d can wire
// true per-axis stride/padding/dilation/output_padding instead of throwing on
// asymmetric values.
auto make_conv3d_backward(int64_t sD, int64_t sH, int64_t sW,
                          int64_t pD, int64_t pH, int64_t pW,
                          int64_t dD, int64_t dH, int64_t dW,
                          int64_t groups,
                          std::vector<::tenzor::Tensor> tensors_to_save)
    -> std::shared_ptr<::tenzor::Function> {
    return std::make_shared<Conv3dBackward>(
        sD, sH, sW, pD, pH, pW, dD, dH, dW, groups,
        std::move(tensors_to_save));
}
auto make_conv_transpose2d_backward(int64_t sH, int64_t sW,
                                    int64_t pH, int64_t pW,
                                    int64_t opH, int64_t opW,
                                    int64_t dH, int64_t dW,
                                    int64_t groups,
                                    std::vector<::tenzor::Tensor> tensors_to_save)
    -> std::shared_ptr<::tenzor::Function> {
    return std::make_shared<ConvTranspose2dBackward>(
        sH, sW, pH, pW, opH, opW, dH, dW, groups,
        std::move(tensors_to_save));
}
} // namespace internal

// Audit I5: per-axis Conv3d ctor. Each spatial axis (D/H/W) gets its own
// kernel/stride/padding/dilation.
Conv3d::Conv3d(int64_t in_channels, int64_t out_channels,
               std::tuple<int64_t, int64_t, int64_t> kernel_size,
               std::tuple<int64_t, int64_t, int64_t> stride,
               std::tuple<int64_t, int64_t, int64_t> padding,
               std::tuple<int64_t, int64_t, int64_t> dilation,
               int64_t groups, bool bias)
    : in_channels_(in_channels), out_channels_(out_channels),
      kD_(std::get<0>(kernel_size)), kH_(std::get<1>(kernel_size)), kW_(std::get<2>(kernel_size)),
      sD_(std::get<0>(stride)),      sH_(std::get<1>(stride)),      sW_(std::get<2>(stride)),
      pD_(std::get<0>(padding)),     pH_(std::get<1>(padding)),     pW_(std::get<2>(padding)),
      dD_(std::get<0>(dilation)),    dH_(std::get<1>(dilation)),    dW_(std::get<2>(dilation)),
      groups_(groups <= 0 ? 1 : groups) {

    if (in_channels % groups_ != 0) {
        throw std::invalid_argument("in_channels must be divisible by groups");
    }
    if (out_channels % groups_ != 0) {
        throw std::invalid_argument("out_channels must be divisible by groups");
    }

    // Weight shape: (C_out, C_in/groups, kD, kH, kW)
    std::vector<int64_t> weight_shape = {out_channels, in_channels / groups_, kD_, kH_, kW_};
    int64_t fan_in = (in_channels / groups_) * kD_ * kH_ * kW_;
    float std_init = std::sqrt(2.0f / fan_in);
    auto weight_tensor = randn(weight_shape) * std_init;
    auto weight_init = Variable(weight_tensor, true);
    register_parameter("weight", weight_init);

    if (bias) {
        // PyTorch convention (and sibling conv layers): bias is uniform on
        // [-1/sqrt(fan_in), 1/sqrt(fan_in)], not zero.
        float bound = 1.0f / std::sqrt(static_cast<float>(fan_in));
        auto bias_tensor = (rand({out_channels}) * 2.0f * bound) - bound;
        auto bias_init = Variable(bias_tensor, true);
        register_parameter("bias", bias_init);
    }
}

// Scalar ctor — delegates to per-axis with identical D/H/W values.
Conv3d::Conv3d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
              int64_t stride, int64_t padding, int64_t dilation,
              int64_t groups, bool bias)
    : Conv3d(in_channels, out_channels,
             std::make_tuple(kernel_size, kernel_size, kernel_size),
             std::make_tuple(stride, stride, stride),
             std::make_tuple(padding, padding, padding),
             std::make_tuple(dilation, dilation, dilation),
             groups, bias) {}

auto Conv3d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 5) {
        throw std::invalid_argument("Conv3d expects 5D input [batch, channels, depth, height, width]");
    }

    int64_t in_channels = input_shape[1];
    int64_t depth = input_shape[2];
    int64_t height = input_shape[3];
    int64_t width = input_shape[4];

    if (in_channels != in_channels_) {
        throw std::invalid_argument("Input channels mismatch");
    }

    // Audit I5: per-axis output sizing.
    int64_t out_d = calculate_output_size(depth,  kD_, sD_, pD_, dD_);
    int64_t out_h = calculate_output_size(height, kH_, sH_, pH_, dH_);
    int64_t out_w = calculate_output_size(width,  kW_, sW_, pW_, dW_);

    if (out_d <= 0 || out_h <= 0 || out_w <= 0) {
        throw std::runtime_error(
            "Invalid Conv3d configuration: output dimensions are non-positive (out_d=" +
            std::to_string(out_d) + ", out_h=" + std::to_string(out_h) +
            ", out_w=" + std::to_string(out_w) + ")");
    }

    auto& weight = *parameters_["weight"];
    auto bias_it = parameters_.find("bias");
    Device original_device = input.tensor().device();

    Variable weight_matched = variable_cast(weight, input.dtype());
    if (input.tensor().device().type != weight.tensor().device().type) {
        tenzor::utils::wrap_preserving_grad(weight_matched, weight_matched.tensor().to(original_device));
    }

    const Tensor* bias_ptr = nullptr;
    Variable bias_matched;
    if (bias_it != parameters_.end()) {
        auto& bias = *bias_it->second;
        bias_matched = variable_cast(bias, input.dtype());
        if (input.tensor().device().type != bias.tensor().device().type) {
            tenzor::utils::wrap_preserving_grad(bias_matched, bias_matched.tensor().to(original_device));
        }
        bias_ptr = &bias_matched.tensor();
    }

    std::vector<Tensor> inputs_vec = {input.tensor(), weight_matched.tensor()};
    if (bias_ptr != nullptr) inputs_vec.push_back(*bias_ptr);

    // Audit I5: pack BOTH scalar (back-compat for backends that haven't read
    // per-axis yet) AND per-axis attribute keys. The CPU registry reads the
    // per-axis keys with scalar fallback.
    NewOpAttributes forward_attrs;
    forward_attrs.set(AttrKey::Stride,    sD_);
    forward_attrs.set(AttrKey::Padding,   pD_);
    forward_attrs.set(AttrKey::Dilation,  dD_);
    forward_attrs.set(AttrKey::StrideD,   sD_);
    forward_attrs.set(AttrKey::StrideH,   sH_);
    forward_attrs.set(AttrKey::StrideW,   sW_);
    forward_attrs.set(AttrKey::PaddingD,  pD_);
    forward_attrs.set(AttrKey::PaddingH,  pH_);
    forward_attrs.set(AttrKey::PaddingW,  pW_);
    forward_attrs.set(AttrKey::DilationD, dD_);
    forward_attrs.set(AttrKey::DilationH, dH_);
    forward_attrs.set(AttrKey::DilationW, dW_);
    forward_attrs.set(AttrKey::Groups,    groups_);
    // Kernel size as attributes so the JIT tracer / ONNX exporter emit a
    // faithful 3D Conv node (kernel_shape); the kernels read it from the weight.
    forward_attrs.set(AttrKey::KernelSizeD, kD_);
    forward_attrs.set(AttrKey::KernelSizeH, kH_);
    forward_attrs.set(AttrKey::KernelSizeW, kW_);
    DType original_dtype = input.dtype();
    // CC.5: depthwise fast-path dispatch surface for 3D (groups == in_channels).
    // FF.3: promoted from a `constexpr bool ... = false` gate to a runtime
    // query so the dispatch surface is live. Eligible depthwise-3d inputs now
    // route to OpId::DepthwiseConv3d; backends without a real kernel fire
    // their CC.5 throw-stub (intentional — loud failure over silent
    // generic-Conv3d miscomputation). See Conv1d::forward_impl for rationale.
    const bool depthwise_conv3d_kernel_available =
        ::tenzor::is_op_supported(
            OpId::DepthwiseConv3d, input.tensor().device().type);
    [[maybe_unused]] const bool eligible_depthwise_3d =
        (groups_ == in_channels_) && (groups_ > 1);
    auto output_result = (depthwise_conv3d_kernel_available && eligible_depthwise_3d)
        ? dispatch(OpId::DepthwiseConv3d, std::span<const Tensor>(inputs_vec), forward_attrs)
        : dispatch<OpId::Conv3dForward>(inputs_vec, forward_attrs);
    auto output = output_result[0];
    if (output.dtype() != original_dtype) {
        output = output.to(original_dtype);
    }

    auto result = Variable(output, input.requires_grad() || weight.requires_grad());

    if (input.requires_grad() || weight.requires_grad()) {
        std::vector<Tensor> tensors_to_save;
        if (bias_ptr != nullptr) {
            tensors_to_save = {input.tensor(), weight_matched.tensor(), *bias_ptr};
        } else {
            tensors_to_save = {input.tensor(), weight_matched.tensor()};
        }

        auto backward_fn = std::make_shared<Conv3dBackward>(
            sD_, sH_, sW_, pD_, pH_, pW_, dD_, dH_, dW_, groups_,
            std::move(tensors_to_save));

        result.set_grad_fn(backward_fn);

        std::vector<Variable> input_vars = {input, *parameters_["weight"]};
        if (bias_it != parameters_.end()) input_vars.push_back(*bias_it->second);
        backward_fn->set_input_variables(input_vars);

        // next_functions must have one slot per input variable in the SAME
        // order as input_vars so the engine's positional routing reaches
        // non-leaf weight/bias gradients (grad_fn() is nullptr for leaves).
        std::vector<std::shared_ptr<Function>> next_funcs;
        next_funcs.reserve(input_vars.size());
        for (const auto& var : input_vars) next_funcs.push_back(var.grad_fn());
        backward_fn->set_next_functions(next_funcs);
    }

    return result;
}

auto Conv3d::reset_parameters() -> void {
    int64_t fan_in = in_channels_ / groups_ * kD_ * kH_ * kW_;
    float std = std::sqrt(2.0f / fan_in);
    auto new_weight = randn({out_channels_, in_channels_ / groups_, kD_, kH_, kW_}) * std;
    auto weight_ = Variable(new_weight, true);
    parameters_["weight"] = std::make_shared<Variable>(weight_);
}

// ============================================================================
// ConvTranspose3d Implementation
// ============================================================================

// Audit I5-followup: per-axis ConvTranspose3dBackward.
// ============================================================================
// ConvTranspose3dGradWeightBackward - JIT-R067 (see Conv2dGradWeightBackward
// and ConvTranspose1dGradWeightBackward above for the full derivation and
// role-swap rationale). A=input_var, B=grad_out_var:
//   d(Loss)/d(input_var)    = conv3d(grad_out_var, G, ...) [cropped to
//                              input_shape, mirroring this class's own
//                              grad_input correction]
//   d(Loss)/d(grad_out_var) = conv_transpose3d(input_var, G, ..., opD_/opH_/opW_)
//                              [defensively cropped to grad_out_var's shape]
// ============================================================================
class ConvTranspose3dGradWeightBackward : public Function {
public:
    ConvTranspose3dGradWeightBackward(int64_t sD, int64_t sH, int64_t sW,
                                       int64_t pD, int64_t pH, int64_t pW,
                                       int64_t opD, int64_t opH, int64_t opW,
                                       int64_t dD, int64_t dH, int64_t dW,
                                       int64_t groups,
                                       std::vector<Tensor> tensors_to_save)
        : sD_(sD), sH_(sH), sW_(sW), pD_(pD), pH_(pH), pW_(pW),
          opD_(opD), opH_(opH), opW_(opW), dD_(dD), dH_(dH), dW_(dW), groups_(groups) {
        save_for_backward(std::move(tensors_to_save));
    }

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("ConvTranspose3dGradWeightBackward::forward should not be called");
    }

    static auto crop_to(const Tensor& t, std::span<const int64_t> target_shape) -> Tensor {
        auto s = t.shape();
        if (s.size() != 5 || target_shape.size() != 5) return t;
        Tensor out = t;
        if (s[2] != target_shape[2]) out = tenzor::slice(out, 2, 0, target_shape[2]);
        if (s[3] != target_shape[3]) out = tenzor::slice(out, 3, 0, target_shape[3]);
        if (s[4] != target_shape[4]) out = tenzor::slice(out, 4, 0, target_shape[4]);
        return out;
    }

    static auto crop_to_var(const Variable& v, std::span<const int64_t> target_shape) -> Variable {
        auto s = v.tensor().shape();
        if (s.size() != 5 || target_shape.size() != 5) return v;
        Variable out = v;
        if (s[2] != target_shape[2]) out = ::tenzor::narrow(out, 2, 0, target_shape[2]);
        if (s[3] != target_shape[3]) out = ::tenzor::narrow(out, 3, 0, target_shape[3]);
        if (s[4] != target_shape[4]) out = ::tenzor::narrow(out, 4, 0, target_shape[4]);
        return out;
    }

    // input_variables()/next_functions() order is {input_var, grad_out_var}.
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        const Tensor& G = grad_outputs[0];
        const Tensor& input_t = saved_tensors_[0];
        const Tensor& grad_out_t = saved_tensors_[1];

        Variable Gv(G, false);
        Variable d_input_var = ::tenzor::nn::functional::conv3d(
            Variable(grad_out_t, false), Gv, std::nullopt,
            {sD_, sH_, sW_}, {pD_, pH_, pW_}, {dD_, dH_, dW_}, groups_);
        Tensor d_input = crop_to(d_input_var.tensor(), input_t.shape());

        Variable d_grad_out_var = ::tenzor::nn::functional::conv_transpose3d(
            Variable(input_t, false), Gv, std::nullopt,
            {sD_, sH_, sW_}, {pD_, pH_, pW_}, {opD_, opH_, opW_}, groups_, {dD_, dH_, dW_});
        Tensor d_grad_out = crop_to(d_grad_out_var.tensor(), grad_out_t.shape());
        return {d_input, d_grad_out};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        Variable G = grad_outputs[0];
        Variable input_var, grad_out_var;
        if (has_saved_variables() && saved_variables_.size() >= 2) {
            input_var = saved_variables_[0];
            grad_out_var = saved_variables_[1];
        } else {
            input_var = Variable(saved_tensors_[0], false);
            grad_out_var = Variable(saved_tensors_[1], false);
        }

        Variable d_input = ::tenzor::nn::functional::conv3d(
            grad_out_var, G, std::nullopt, {sD_, sH_, sW_}, {pD_, pH_, pW_}, {dD_, dH_, dW_}, groups_);
        d_input = crop_to_var(d_input, input_var.tensor().shape());

        Variable d_grad_out = ::tenzor::nn::functional::conv_transpose3d(
            input_var, G, std::nullopt, {sD_, sH_, sW_}, {pD_, pH_, pW_}, {opD_, opH_, opW_}, groups_, {dD_, dH_, dW_});
        d_grad_out = crop_to_var(d_grad_out, grad_out_var.tensor().shape());
        return {d_input, d_grad_out};
    }

    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return false; }

private:
    int64_t sD_, sH_, sW_, pD_, pH_, pW_, opD_, opH_, opW_, dD_, dH_, dW_, groups_;
};

class ConvTranspose3dBackward : public Function {
public:
    ConvTranspose3dBackward(int64_t sD, int64_t sH, int64_t sW,
                            int64_t pD, int64_t pH, int64_t pW,
                            int64_t opD, int64_t opH, int64_t opW,
                            int64_t dD, int64_t dH, int64_t dW,
                            int64_t groups,
                            std::vector<Tensor> tensors_to_save)
        : sD_(sD), sH_(sH), sW_(sW),
          pD_(pD), pH_(pH), pW_(pW),
          opD_(opD), opH_(opH), opW_(opW),
          dD_(dD), dH_(dH), dW_(dW),
          groups_(groups) {
        save_for_backward(std::move(tensors_to_save));
    }

    // Scalar ctor — delegates to per-axis with identical D/H/W values.
    ConvTranspose3dBackward(int64_t stride, int64_t padding, int64_t output_padding,
                            int64_t dilation, int64_t groups,
                            std::vector<Tensor> tensors_to_save)
        : ConvTranspose3dBackward(stride, stride, stride,
                                  padding, padding, padding,
                                  output_padding, output_padding, output_padding,
                                  dilation, dilation, dilation,
                                  groups, std::move(tensors_to_save)) {}

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("ConvTranspose3dBackward::forward should not be called");
    }

    // Helper to pack ConvT3d per-axis attrs (with scalar fallback).
    void pack_t3d_attrs(NewOpAttributes& a) const {
        a.set(AttrKey::Stride,         sD_);
        a.set(AttrKey::Padding,        pD_);
        a.set(AttrKey::OutputPadding,  opD_);
        a.set(AttrKey::Dilation,       dD_);
        a.set(AttrKey::StrideD,        sD_);
        a.set(AttrKey::StrideH,        sH_);
        a.set(AttrKey::StrideW,        sW_);
        a.set(AttrKey::PaddingD,       pD_);
        a.set(AttrKey::PaddingH,       pH_);
        a.set(AttrKey::PaddingW,       pW_);
        a.set(AttrKey::OutputPaddingD, opD_);
        a.set(AttrKey::OutputPaddingH, opH_);
        a.set(AttrKey::OutputPaddingW, opW_);
        a.set(AttrKey::DilationD,      dD_);
        a.set(AttrKey::DilationH,      dH_);
        a.set(AttrKey::DilationW,      dW_);
        a.set(AttrKey::Groups,         groups_);
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        Tensor grad_output = grad_outputs[0];
        auto saved = saved_tensors();
        Tensor input = saved[0];
        Tensor weight = saved[1];
        bool has_bias = saved.size() > 2;

        DType orig_dtype = grad_output.dtype();
        bool needs_upcast = (orig_dtype == DType::Float16);
        if (needs_upcast) {
            grad_output = grad_output.to(DType::Float32);
            input = input.to(DType::Float32);
            weight = weight.to(DType::Float32);
        }

        auto shape_to_str = [](std::span<const int64_t> s) {
            std::string r;
            for (size_t i = 0; i < s.size(); ++i) { if (i > 0) r += ","; r += std::to_string(s[i]); }
            return r;
        };

        NewOpAttributes common_attrs;
        pack_t3d_attrs(common_attrs);

        std::string input_shape_str = shape_to_str(input.shape());
        std::string weight_shape_str = shape_to_str(weight.shape());

        NewOpAttributes bi_attrs = common_attrs;
        bi_attrs.set(AttrKey::InputShape, std::string_view(input_shape_str));
        std::vector<Tensor> bi_inputs = {grad_output, input, weight};
        auto grad_input = dispatch<OpId::ConvTranspose3dBackwardInput>(bi_inputs, bi_attrs)[0];

        NewOpAttributes bw_attrs = common_attrs;
        bw_attrs.set(AttrKey::WeightShape, std::string_view(weight_shape_str));
        std::vector<Tensor> bw_inputs = {grad_output, input, weight};
        auto grad_weight = dispatch<OpId::ConvTranspose3dBackwardWeight>(bw_inputs, bw_attrs)[0];

        if (needs_upcast) {
            grad_input = grad_input.to(orig_dtype);
            grad_weight = grad_weight.to(orig_dtype);
        }

        if (has_bias) {
            std::vector<Tensor> bb_inputs = {grad_output};
            auto grad_bias = dispatch<OpId::ConvTranspose3dBackwardBias>(bb_inputs)[0];
            if (needs_upcast) grad_bias = grad_bias.to(orig_dtype);
            return {grad_input, grad_weight, grad_bias};
        }
        return {grad_input, grad_weight};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        Variable grad_out_var = grad_outputs[0];

        Variable input_var, weight_var;
        if (has_saved_variables() && saved_variables_.size() >= 2) {
            input_var = saved_variables_[0];
            weight_var = saved_variables_[1];
        } else {
            input_var = Variable(saved_tensors_[0], false);
            weight_var = Variable(saved_tensors_[1], false);
        }
        bool has_bias = saved_tensors_.size() > 2;

        auto grad_input = ::tenzor::nn::functional::conv3d(
            grad_out_var, weight_var,
            std::nullopt,
            {sD_, sH_, sW_},
            {pD_, pH_, pW_},
            {dD_, dH_, dW_},
            groups_);

        auto input_shape = input_var.tensor().shape();
        auto gi_shape = grad_input.tensor().shape();
        if (gi_shape.size() == 5 && input_shape.size() == 5 &&
            gi_shape[0] == input_shape[0] && gi_shape[1] == input_shape[1]) {
            if (gi_shape[2] != input_shape[2] || gi_shape[3] != input_shape[3] ||
                gi_shape[4] != input_shape[4]) {
                auto sliced = tenzor::slice(grad_input.tensor(), 2, 0, input_shape[2]);
                sliced = tenzor::slice(sliced, 3, 0, input_shape[3]);
                sliced = tenzor::slice(sliced, 4, 0, input_shape[4]);
                grad_input = Variable(sliced, grad_out_var.requires_grad());
            }
        }

        auto shape_to_str = [](std::span<const int64_t> s) {
            std::string r;
            for (size_t i = 0; i < s.size(); ++i) { if (i > 0) r += ","; r += std::to_string(s[i]); }
            return r;
        };

        std::string ws_str = shape_to_str(weight_var.tensor().shape());

        NewOpAttributes bw_attrs;
        pack_t3d_attrs(bw_attrs);
        bw_attrs.set(AttrKey::WeightShape, std::string_view(ws_str));

        // Pre-existing bug fixed alongside JIT-R067: this previously passed
        // {input, grad_out, weight}, but ConvTranspose3dBackward::backward()
        // (the tensor-level sibling) and the CUDA cuDNN kernel wrapper both
        // expect {grad_out, input, weight} — dispatch<OpId::
        // ConvTranspose3dBackwardWeight> reads in_channels from inputs[1]'s
        // channel dim, so the swapped order silently fed it grad_out's
        // channel count instead of input's. On CPU/Vulkan/ROCm this went
        // undetected (their kernels don't validate against it the same way);
        // on CUDA it threw a shape-mismatch in validate_grad_shape_or_throw
        // for any ConvTranspose3d with in_channels != out_channels/groups —
        // discovered via JIT-R067's gradgrad-check testing, unreachable
        // before because nothing previously built a real 2nd-order graph
        // through ConvTranspose3d with asymmetric channels.
        std::vector<Tensor> bw_inputs = {grad_out_var.tensor(), input_var.tensor(), weight_var.tensor()};
        auto grad_weight_t = dispatch<OpId::ConvTranspose3dBackwardWeight>(bw_inputs, bw_attrs)[0];

        // JIT-R067: attach a differentiable grad_fn
        // (ConvTranspose3dGradWeightBackward, defined above) so a 3rd-order
        // gradient flows correctly back to input_var/grad_out_var instead of
        // silently stopping here.
        bool gw_requires_grad = ::tenzor::is_grad_enabled() &&
            (input_var.requires_grad() || grad_out_var.requires_grad());
        Variable grad_weight(grad_weight_t, gw_requires_grad);
        if (gw_requires_grad) {
            auto gw_fn = std::make_shared<ConvTranspose3dGradWeightBackward>(
                sD_, sH_, sW_, pD_, pH_, pW_, opD_, opH_, opW_, dD_, dH_, dW_, groups_,
                std::vector<Tensor>{input_var.tensor(), grad_out_var.tensor()});
            if (::tenzor::is_creating_graph()) {
                gw_fn->save_variables_for_backward({input_var, grad_out_var});
            }
            gw_fn->set_next_functions({input_var.grad_fn(), grad_out_var.grad_fn()});
            gw_fn->set_input_variables({input_var, grad_out_var});
            grad_weight.set_grad_fn(gw_fn);
        }

        if (has_bias) {
            auto gb = ::tenzor::sum(grad_out_var, 0, false);
            gb = ::tenzor::sum(gb, 1, false);
            gb = ::tenzor::sum(gb, 1, false);
            gb = ::tenzor::sum(gb, 1, false);
            return {grad_input, grad_weight, gb};
        }
        return {grad_input, grad_weight};
    }

    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return false; }

private:
    int64_t sD_, sH_, sW_;
    int64_t pD_, pH_, pW_;
    int64_t opD_, opH_, opW_;
    int64_t dD_, dH_, dW_;
    int64_t groups_;
};

// Audit I5: per-axis ConvTranspose3d ctor.
ConvTranspose3d::ConvTranspose3d(int64_t in_channels, int64_t out_channels,
                                  std::tuple<int64_t, int64_t, int64_t> kernel_size,
                                  std::tuple<int64_t, int64_t, int64_t> stride,
                                  std::tuple<int64_t, int64_t, int64_t> padding,
                                  std::tuple<int64_t, int64_t, int64_t> output_padding,
                                  std::tuple<int64_t, int64_t, int64_t> dilation,
                                  int64_t groups, bool bias)
    : in_channels_(in_channels), out_channels_(out_channels),
      kD_(std::get<0>(kernel_size)), kH_(std::get<1>(kernel_size)), kW_(std::get<2>(kernel_size)),
      sD_(std::get<0>(stride)),      sH_(std::get<1>(stride)),      sW_(std::get<2>(stride)),
      pD_(std::get<0>(padding)),     pH_(std::get<1>(padding)),     pW_(std::get<2>(padding)),
      opD_(std::get<0>(output_padding)), opH_(std::get<1>(output_padding)), opW_(std::get<2>(output_padding)),
      dD_(std::get<0>(dilation)),    dH_(std::get<1>(dilation)),    dW_(std::get<2>(dilation)),
      groups_(groups) {

    if (in_channels % groups != 0)  throw std::invalid_argument("in_channels must be divisible by groups");
    if (out_channels % groups != 0) throw std::invalid_argument("out_channels must be divisible by groups");
    if (opD_ >= sD_ && opD_ != 0)   throw std::invalid_argument("output_padding (D) must be smaller than stride (D)");
    if (opH_ >= sH_ && opH_ != 0)   throw std::invalid_argument("output_padding (H) must be smaller than stride (H)");
    if (opW_ >= sW_ && opW_ != 0)   throw std::invalid_argument("output_padding (W) must be smaller than stride (W)");

    std::vector<int64_t> weight_shape = {in_channels, out_channels / groups, kD_, kH_, kW_};
    // Transposed-conv fan_in = (out/groups)*kD*kH*kW (weight.size(1)*receptive).
    int64_t fan_in = (out_channels / groups) * kD_ * kH_ * kW_;
    float std_init = std::sqrt(2.0f / fan_in);
    auto weight_tensor = randn(weight_shape) * std_init;
    register_parameter("weight", Variable(weight_tensor, true));

    if (bias) {
        std::vector<int64_t> bias_shape = {out_channels};
        float bound = 1.0f / std::sqrt(static_cast<float>(fan_in));
        auto bias_tensor = (rand(bias_shape) * 2.0f * bound) - bound;
        register_parameter("bias", Variable(bias_tensor, true));
    }
}

// Scalar ctor — delegates to per-axis.
ConvTranspose3d::ConvTranspose3d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
                                  int64_t stride, int64_t padding, int64_t output_padding,
                                  int64_t dilation, int64_t groups, bool bias)
    : ConvTranspose3d(in_channels, out_channels,
                      std::make_tuple(kernel_size, kernel_size, kernel_size),
                      std::make_tuple(stride, stride, stride),
                      std::make_tuple(padding, padding, padding),
                      std::make_tuple(output_padding, output_padding, output_padding),
                      std::make_tuple(dilation, dilation, dilation),
                      groups, bias) {}

auto ConvTranspose3d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 5) {
        throw std::invalid_argument("ConvTranspose3d expects 5D input [batch, channels, depth, height, width]");
    }

    int64_t in_channels = input_shape[1];
    if (in_channels != in_channels_) {
        throw std::invalid_argument("Input channels mismatch");
    }

    auto& weight = *parameters_["weight"];
    auto bias_it = parameters_.find("bias");
    Device original_device = input.tensor().device();

    // Handle dtype and device mismatch
    Variable weight_matched = variable_cast(weight, input.dtype());
    if (input.tensor().device().type != weight.tensor().device().type) {
        tenzor::utils::wrap_preserving_grad(weight_matched, weight_matched.tensor().to(original_device));
    }

    const Tensor* bias_ptr = nullptr;
    Variable bias_matched;
    if (bias_it != parameters_.end()) {
        auto& bias = *bias_it->second;
        bias_matched = variable_cast(bias, input.dtype());
        if (input.tensor().device().type != bias.tensor().device().type) {
            tenzor::utils::wrap_preserving_grad(bias_matched, bias_matched.tensor().to(original_device));
        }
        bias_ptr = &bias_matched.tensor();
    }

    // Dispatch forward via OpId
    std::vector<Tensor> inputs_vec = {input.tensor(), weight_matched.tensor()};
    if (bias_ptr != nullptr) {
        inputs_vec.push_back(*bias_ptr);
    }

    // Audit I5: per-axis attrs (with D-axis scalar fallback for back-compat).
    NewOpAttributes forward_attrs;
    forward_attrs.set(AttrKey::Stride,         sD_);
    forward_attrs.set(AttrKey::Padding,        pD_);
    forward_attrs.set(AttrKey::OutputPadding,  opD_);
    forward_attrs.set(AttrKey::Dilation,       dD_);
    forward_attrs.set(AttrKey::StrideD,        sD_);
    forward_attrs.set(AttrKey::StrideH,        sH_);
    forward_attrs.set(AttrKey::StrideW,        sW_);
    forward_attrs.set(AttrKey::PaddingD,       pD_);
    forward_attrs.set(AttrKey::PaddingH,       pH_);
    forward_attrs.set(AttrKey::PaddingW,       pW_);
    forward_attrs.set(AttrKey::OutputPaddingD, opD_);
    forward_attrs.set(AttrKey::OutputPaddingH, opH_);
    forward_attrs.set(AttrKey::OutputPaddingW, opW_);
    forward_attrs.set(AttrKey::DilationD,      dD_);
    forward_attrs.set(AttrKey::DilationH,      dH_);
    forward_attrs.set(AttrKey::DilationW,      dW_);
    forward_attrs.set(AttrKey::Groups,         groups_);

    DType original_dtype = input.dtype();
    auto output_result = dispatch<OpId::ConvTranspose3dForward>(inputs_vec, forward_attrs);
    auto output = output_result[0];
    if (output.dtype() != original_dtype) {
        output = output.to(original_dtype);
    }

    auto result = Variable(output, input.requires_grad() || weight.requires_grad());

    if (input.requires_grad() || weight.requires_grad()) {
        std::vector<Tensor> tensors_to_save;
        if (bias_ptr != nullptr) {
            tensors_to_save = {input.tensor(), weight_matched.tensor(), *bias_ptr};
        } else {
            tensors_to_save = {input.tensor(), weight_matched.tensor()};
        }

        // Audit I5-followup: per-axis ConvTranspose3dBackward.
        auto backward_fn = std::make_shared<ConvTranspose3dBackward>(
            sD_, sH_, sW_, pD_, pH_, pW_, opD_, opH_, opW_, dD_, dH_, dW_, groups_,
            std::move(tensors_to_save)
        );

        result.set_grad_fn(backward_fn);

        std::vector<Variable> input_vars = {input, *parameters_["weight"]};
        if (bias_it != parameters_.end()) {
            input_vars.push_back(*bias_it->second);
        }
        backward_fn->set_input_variables(input_vars);

        // next_functions must have one slot per input variable in the SAME
        // order as input_vars so the engine's positional routing reaches
        // non-leaf weight/bias gradients (grad_fn() is nullptr for leaves).
        std::vector<std::shared_ptr<Function>> next_funcs;
        next_funcs.reserve(input_vars.size());
        for (const auto& var : input_vars) {
            next_funcs.push_back(var.grad_fn());
        }
        backward_fn->set_next_functions(next_funcs);
    }

    return result;
}

auto ConvTranspose3d::reset_parameters() -> void {
    // fan_in for transposed weight = (out/groups)*kD*kH*kW.
    int64_t fan_in = (out_channels_ / groups_) * kD_ * kH_ * kW_;
    float std = std::sqrt(2.0f / fan_in);

    std::vector<int64_t> weight_shape = {in_channels_, out_channels_ / groups_, kD_, kH_, kW_};
    auto new_weight_tensor = randn(weight_shape) * std;
    parameters_["weight"] = std::make_shared<Variable>(new_weight_tensor, true);

    auto bias_it = parameters_.find("bias");
    if (bias_it != parameters_.end()) {
        std::vector<int64_t> bias_shape = {out_channels_};
        float bound = 1.0f / std::sqrt(static_cast<float>(fan_in));
        auto new_bias_tensor = (rand(bias_shape) * 2.0f * bound) - bound;
        bias_it->second = std::make_shared<Variable>(new_bias_tensor, true);
    }
}

// ============================================================================
// ConvTranspose1d Implementation
// ============================================================================

// ============================================================================
// ConvTranspose1dGradWeightBackward - JIT-R067 (see Conv2dGradWeightBackward
// above for the full derivation). ConvTranspose1dBackward's grad_weight
// dispatch uses "swapped roles" — input_var plays the role Conv's grad_out
// plays, grad_out_var plays the role Conv's input plays (per the existing
// class's own "Swap roles" comment) — so with A=input_var, B=grad_out_var:
//   d(Loss)/d(input_var)    = conv1d(grad_out_var, G, ...)
//                              [same formula as this class's own grad_input,
//                               with G substituted for weight — needs the
//                               SAME crop-to-input_shape correction]
//   d(Loss)/d(grad_out_var) = conv_transpose1d(input_var, G, ..., output_padding_)
//                              [uses the REAL output_padding_ so the result
//                               naturally reproduces grad_out_var's shape;
//                               defensively cropped as a safety net]
// ============================================================================
class ConvTranspose1dGradWeightBackward : public Function {
public:
    ConvTranspose1dGradWeightBackward(int64_t stride, int64_t padding, int64_t output_padding,
                                       int64_t dilation, int64_t groups,
                                       std::vector<Tensor> tensors_to_save)
        : stride_(stride), padding_(padding), output_padding_(output_padding),
          dilation_(dilation), groups_(groups) {
        save_for_backward(std::move(tensors_to_save));
    }

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("ConvTranspose1dGradWeightBackward::forward should not be called");
    }

    // input_variables()/next_functions() order is {input_var, grad_out_var}.
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        const Tensor& G = grad_outputs[0];
        const Tensor& input_t = saved_tensors_[0];
        const Tensor& grad_out_t = saved_tensors_[1];
        auto input_shape = input_t.shape();
        auto grad_out_shape = grad_out_t.shape();

        Variable Gv(G, false);
        Variable d_input_var = ::tenzor::nn::functional::conv1d(
            Variable(grad_out_t, false), Gv, std::nullopt, stride_, padding_, dilation_, groups_);
        Tensor d_input = d_input_var.tensor();
        if (d_input.shape().size() == 3 && d_input.shape()[2] != input_shape[2]) {
            d_input = tenzor::slice(d_input, 2, 0, input_shape[2]);
        }

        Variable d_grad_out_var = ::tenzor::nn::functional::conv_transpose1d(
            Variable(input_t, false), Gv, std::nullopt, stride_, padding_,
            output_padding_, groups_, dilation_);
        Tensor d_grad_out = d_grad_out_var.tensor();
        if (d_grad_out.shape().size() == 3 && d_grad_out.shape()[2] != grad_out_shape[2]) {
            d_grad_out = tenzor::slice(d_grad_out, 2, 0, grad_out_shape[2]);
        }
        return {d_input, d_grad_out};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        Variable G = grad_outputs[0];
        Variable input_var, grad_out_var;
        if (has_saved_variables() && saved_variables_.size() >= 2) {
            input_var = saved_variables_[0];
            grad_out_var = saved_variables_[1];
        } else {
            input_var = Variable(saved_tensors_[0], false);
            grad_out_var = Variable(saved_tensors_[1], false);
        }
        auto input_shape = input_var.tensor().shape();
        auto grad_out_shape = grad_out_var.tensor().shape();

        Variable d_input = ::tenzor::nn::functional::conv1d(
            grad_out_var, G, std::nullopt, stride_, padding_, dilation_, groups_);
        if (d_input.tensor().shape().size() == 3 && d_input.tensor().shape()[2] != input_shape[2]) {
            d_input = ::tenzor::narrow(d_input, 2, 0, input_shape[2]);
        }

        Variable d_grad_out = ::tenzor::nn::functional::conv_transpose1d(
            input_var, G, std::nullopt, stride_, padding_, output_padding_, groups_, dilation_);
        if (d_grad_out.tensor().shape().size() == 3 && d_grad_out.tensor().shape()[2] != grad_out_shape[2]) {
            d_grad_out = ::tenzor::narrow(d_grad_out, 2, 0, grad_out_shape[2]);
        }
        return {d_input, d_grad_out};
    }

    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return false; }

private:
    int64_t stride_, padding_, output_padding_, dilation_, groups_;
};

class ConvTranspose1dBackward : public Function {
public:
    ConvTranspose1dBackward(int64_t stride, int64_t padding, int64_t output_padding,
                            int64_t dilation, int64_t groups,
                            std::vector<Tensor> tensors_to_save)
        : stride_(stride), padding_(padding), output_padding_(output_padding),
          dilation_(dilation), groups_(groups) {
        save_for_backward(std::move(tensors_to_save));
    }

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("ConvTranspose1dBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        const Tensor& grad_output = grad_outputs[0];
        const Tensor& input = saved_tensors_[0];
        const Tensor& weight = saved_tensors_[1];
        bool has_bias = saved_tensors_.size() > 2;

        auto input_shape = input.shape();

        // Convert to 4D for ConvTranspose2d backward dispatch
        auto grad_4d = grad_output.unsqueeze(2);     // [N, C_out, L] -> [N, C_out, 1, L]
        auto input_4d = input.unsqueeze(2);           // [N, C_in, L] -> [N, C_in, 1, L]
        auto weight_4d = weight.unsqueeze(2);         // [C_in, C_out/g, K] -> [C_in, C_out/g, 1, K]

        // grad_input: backward of ConvTranspose w.r.t. input is regular Conv2d
        //
        // Audit F.17 ConvTranspose1d dilation: when ConvTranspose1d uses
        // dilation > 1, its conv-adjoint (Conv2d on the unsqueezed tensors)
        // must use the same dilation on the W axis. H is the unsqueezed
        // singleton so dH=1.
        NewOpAttributes conv_attrs;
        conv_attrs.set(AttrKey::Stride, stride_);
        conv_attrs.set(AttrKey::Padding, padding_);
        conv_attrs.set(AttrKey::Dilation, dilation_);
        conv_attrs.set(AttrKey::DilationH, static_cast<int64_t>(1));
        conv_attrs.set(AttrKey::DilationW, dilation_);
        conv_attrs.set(AttrKey::Groups, groups_);

        std::vector<Tensor> conv_inputs = {grad_4d, weight_4d};
        auto conv_result = dispatch(OpId::Conv2dForward, std::span<const Tensor>(conv_inputs), conv_attrs);
        Tensor grad_input_4d = conv_result[0];

        // Handle potential shape mismatch from output_padding
        auto gi_shape = grad_input_4d.shape();
        if (gi_shape.size() == 4 && gi_shape[2] != 1) {
            grad_input_4d = tenzor::slice(grad_input_4d, 2, 0, 1);
        }
        if (gi_shape[3] != input_shape[2]) {
            grad_input_4d = tenzor::slice(grad_input_4d, 3, 0, input_shape[2]);
        }

        Tensor grad_input = grad_input_4d.squeeze(2);

        // grad_weight: swap roles of input and grad_output
        auto weight_4d_shape = weight_4d.shape();
        std::string ws_str = std::to_string(weight_4d_shape[0]) + "," +
                             std::to_string(weight_4d_shape[1]) + "," +
                             std::to_string(weight_4d_shape[2]) + "," +
                             std::to_string(weight_4d_shape[3]);
        NewOpAttributes weight_grad_attrs;
        weight_grad_attrs.set(AttrKey::Stride, stride_);
        weight_grad_attrs.set(AttrKey::Padding, padding_);
        weight_grad_attrs.set(AttrKey::Dilation, dilation_);
        weight_grad_attrs.set(AttrKey::DilationH, static_cast<int64_t>(1));
        weight_grad_attrs.set(AttrKey::DilationW, dilation_);
        weight_grad_attrs.set(AttrKey::Groups, groups_);
        weight_grad_attrs.set(AttrKey::WeightShape, std::string_view(ws_str));

        // Conv2dBackwardWeight expects [grad_output, input, weight]
        std::vector<Tensor> weight_grad_inputs = {input_4d, grad_4d, weight_4d};
        auto weight_grad_result = dispatch(OpId::Conv2dBackwardWeight,
            std::span<const Tensor>(weight_grad_inputs), weight_grad_attrs);
        Tensor grad_weight = weight_grad_result[0].squeeze(2);

        if (has_bias) {
            // grad_bias = sum(grad_output, dims=[0,2])
            Tensor grad_bias = tenzor::sum(tenzor::sum(grad_output, 0, false), 1, false);
            return {grad_input, grad_weight, grad_bias};
        }
        return {grad_input, grad_weight};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        // Higher-order gradient support for ConvTranspose1d.
        // The backward of ConvTranspose1d w.r.t. input is a regular Conv1d.
        Variable grad_out_var = grad_outputs[0];

        Variable input_var, weight_var;
        if (has_saved_variables() && saved_variables_.size() >= 2) {
            input_var = saved_variables_[0];
            weight_var = saved_variables_[1];
        } else {
            input_var = Variable(saved_tensors_[0], false);
            weight_var = Variable(saved_tensors_[1], false);
        }
        bool has_bias = saved_tensors_.size() > 2;

        // grad_input: regular conv1d(grad_output, weight) — preserves graph.
        // Audit F.17 ConvTranspose1d dilation: the regular Conv1d adjoint
        // must match the forward dilation.
        auto grad_input = ::tenzor::nn::functional::conv1d(
            grad_out_var, weight_var,
            std::nullopt,
            stride_, padding_,
            dilation_,
            groups_);

        // Handle potential shape mismatch due to output_padding
        auto input_shape = input_var.tensor().shape();
        auto gi_shape = grad_input.tensor().shape();
        if (gi_shape.size() == 3 && input_shape.size() == 3 &&
            gi_shape[0] == input_shape[0] && gi_shape[1] == input_shape[1]) {
            if (gi_shape[2] != input_shape[2]) {
                auto sliced = tenzor::slice(grad_input.tensor(), 2, 0, input_shape[2]);
                grad_input = Variable(sliced, grad_out_var.requires_grad());
            }
        }

        // grad_weight: dispatch at tensor level via 4D (same as tensor backward)
        auto weight_4d = weight_var.tensor().unsqueeze(2);
        auto grad_4d = grad_out_var.tensor().unsqueeze(2);
        auto input_4d = input_var.tensor().unsqueeze(2);

        auto weight_4d_shape = weight_4d.shape();
        std::string ws_str = std::to_string(weight_4d_shape[0]) + "," +
                             std::to_string(weight_4d_shape[1]) + "," +
                             std::to_string(weight_4d_shape[2]) + "," +
                             std::to_string(weight_4d_shape[3]);
        NewOpAttributes weight_grad_attrs;
        weight_grad_attrs.set(AttrKey::Stride, stride_);
        weight_grad_attrs.set(AttrKey::Padding, padding_);
        weight_grad_attrs.set(AttrKey::Dilation, dilation_);
        weight_grad_attrs.set(AttrKey::DilationH, static_cast<int64_t>(1));
        weight_grad_attrs.set(AttrKey::DilationW, dilation_);
        weight_grad_attrs.set(AttrKey::Groups, groups_);
        weight_grad_attrs.set(AttrKey::WeightShape, std::string_view(ws_str));

        // Swap roles: input acts as grad_output, grad_output acts as input
        std::vector<Tensor> weight_grad_inputs = {input_4d, grad_4d, weight_4d};
        auto weight_grad_result = dispatch(OpId::Conv2dBackwardWeight,
            std::span<const Tensor>(weight_grad_inputs), weight_grad_attrs);
        Tensor grad_weight_t = weight_grad_result[0].squeeze(2);

        // JIT-R067: attach a differentiable grad_fn
        // (ConvTranspose1dGradWeightBackward, defined above) so a 3rd-order
        // gradient flows correctly back to input_var/grad_out_var instead of
        // silently stopping here.
        bool gw_requires_grad = ::tenzor::is_grad_enabled() &&
            (input_var.requires_grad() || grad_out_var.requires_grad());
        Variable grad_weight(grad_weight_t, gw_requires_grad);
        if (gw_requires_grad) {
            auto gw_fn = std::make_shared<ConvTranspose1dGradWeightBackward>(
                stride_, padding_, output_padding_, dilation_, groups_,
                std::vector<Tensor>{input_var.tensor(), grad_out_var.tensor()});
            if (::tenzor::is_creating_graph()) {
                gw_fn->save_variables_for_backward({input_var, grad_out_var});
            }
            gw_fn->set_next_functions({input_var.grad_fn(), grad_out_var.grad_fn()});
            gw_fn->set_input_variables({input_var, grad_out_var});
            grad_weight.set_grad_fn(gw_fn);
        }

        if (has_bias) {
            // grad_bias = sum(grad_output, dims=[0,2])
            auto gb = ::tenzor::sum(grad_out_var, 0, false);  // sum over batch
            gb = ::tenzor::sum(gb, 1, false);                  // sum over L
            return {grad_input, grad_weight, gb};
        }
        return {grad_input, grad_weight};
    }

    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return false; }

private:
    int64_t stride_;
    int64_t padding_;
    int64_t output_padding_;
    int64_t dilation_;
    int64_t groups_;
};

ConvTranspose1d::ConvTranspose1d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
                                 int64_t stride, int64_t padding, int64_t output_padding,
                                 int64_t groups, bool bias, int64_t dilation)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride),
      padding_(padding), output_padding_(output_padding), groups_(groups),
      dilation_(dilation) {

    if (in_channels % groups != 0) {
        throw std::invalid_argument("in_channels must be divisible by groups");
    }
    if (out_channels % groups != 0) {
        throw std::invalid_argument("out_channels must be divisible by groups");
    }
    if (dilation < 1) {
        throw std::invalid_argument("dilation must be >= 1");
    }
    // Audit F.17 ConvTranspose1d dilation: PyTorch's rule is
    // output_padding < max(stride, dilation), since increasing dilation
    // grows the effective kernel span (k_eff = (k-1)*dilation + 1) just
    // as stride > 1 grows the output strided spacing.
    if (output_padding >= std::max(stride, dilation)) {
        throw std::invalid_argument(
            "output_padding must be smaller than max(stride, dilation)");
    }

    // Weight shape: [in_channels, out_channels/groups, kernel_size]
    std::vector<int64_t> weight_shape = {in_channels, out_channels / groups, kernel_size};
    // Transposed-conv fan_in = (out/groups)*kernel_size (weight.size(1)*receptive).
    int64_t fan_in = (out_channels / groups) * kernel_size;
    float std_init = std::sqrt(2.0f / fan_in);
    auto weight_tensor = randn(weight_shape) * std_init;
    register_parameter("weight", Variable(weight_tensor, true));

    if (bias) {
        std::vector<int64_t> bias_shape = {out_channels};
        float bound = 1.0f / std::sqrt(static_cast<float>(fan_in));
        auto bias_tensor = (rand(bias_shape) * 2.0f * bound) - bound;
        register_parameter("bias", Variable(bias_tensor, true));
    }
}

auto ConvTranspose1d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 3) {
        throw std::invalid_argument("ConvTranspose1d expects 3D input [batch, channels, length]");
    }

    int64_t in_channels = input_shape[1];
    if (in_channels != in_channels_) {
        throw std::invalid_argument("Input channels mismatch");
    }

    auto& weight = *parameters_["weight"];
    auto bias_it = parameters_.find("bias");
    Device original_device = input.tensor().device();

    // Handle dtype and device mismatch
    Variable weight_matched = variable_cast(weight, input.dtype());
    if (input.tensor().device().type != weight.tensor().device().type) {
        tenzor::utils::wrap_preserving_grad(weight_matched, weight_matched.tensor().to(original_device));
    }

    const Tensor* bias_ptr = nullptr;
    Variable bias_matched;
    if (bias_it != parameters_.end()) {
        auto& bias = *bias_it->second;
        bias_matched = variable_cast(bias, input.dtype());
        if (input.tensor().device().type != bias.tensor().device().type) {
            tenzor::utils::wrap_preserving_grad(bias_matched, bias_matched.tensor().to(original_device));
        }
        bias_ptr = &bias_matched.tensor();
    }

    DType original_dtype = input.dtype();
    Tensor output;

    {
        // F039: unify the forward code path across ALL backends. Previously only
        // CPU registered a native ConvTranspose1dForward kernel (others fell
        // through to this 2-D-lowered path), so CPU ran different code than the
        // GPU backends and any output_padding/accumulator difference surfaced as
        // CPU-vs-GPU drift. The backward (ConvTranspose1dBackward) is already
        // 2-D-lowered on every backend, so lowering the forward everywhere makes
        // forward and backward consistent AND identical across backends.
        //
        // Unsqueeze to 4-D, dispatch ConvTranspose2d with scalar pad=0 (the
        // kernel would otherwise pad BOTH H and W), squeeze, then trim the W
        // axis by padding_ (ConvTranspose "padding" trims output edges). The H
        // axis is a singleton so dH must stay 1.
        auto input_4d = tenzor::unsqueeze(Variable(input.tensor(), false), 2).tensor();
        auto weight_4d = tenzor::unsqueeze(Variable(weight_matched.tensor(), false), 2).tensor();
        std::vector<Tensor> inputs_vec = {input_4d, weight_4d};
        if (bias_ptr != nullptr) {
            inputs_vec.push_back(*bias_ptr);
        }
        NewOpAttributes forward_attrs;
        forward_attrs.set(AttrKey::Stride, stride_);
        forward_attrs.set(AttrKey::Padding, static_cast<int64_t>(0));
        forward_attrs.set(AttrKey::OutputPadding, output_padding_);
        // The synthetic H axis is a singleton (length 1), so its output_padding
        // MUST be 0 — otherwise the 2D kernel reads OutputPaddingH (which falls
        // back to the scalar OutputPadding=output_padding_) and produces an H of
        // output_padding_+1, leaving a stray 4-D tensor after squeeze(2) and
        // trimming the wrong axis. Pin H=0, real length on W. (Same reason
        // DilationH is pinned to 1 below.)
        forward_attrs.set(AttrKey::OutputPaddingH, static_cast<int64_t>(0));
        forward_attrs.set(AttrKey::OutputPaddingW, output_padding_);
        forward_attrs.set(AttrKey::Dilation, dilation_);
        forward_attrs.set(AttrKey::DilationH, static_cast<int64_t>(1));
        forward_attrs.set(AttrKey::DilationW, dilation_);
        forward_attrs.set(AttrKey::Groups, groups_);
        auto output_result = dispatch(OpId::ConvTranspose2dForward,
            std::span<const Tensor>(inputs_vec), forward_attrs);
        output = tenzor::squeeze(Variable(output_result[0], false), 2).tensor();

        if (padding_ > 0) {
            int64_t L_full = output.shape()[2];
            int64_t L_trimmed = L_full - 2 * padding_;
            if (L_trimmed <= 0) {
                throw std::runtime_error(
                    "Invalid ConvTranspose1d configuration: output length after "
                    "padding trim is non-positive (L_full=" + std::to_string(L_full) +
                    ", padding=" + std::to_string(padding_) + ")");
            }
            output = tenzor::slice(Variable(output, false), /*dim=*/2, /*start=*/padding_,
                                   /*end=*/padding_ + L_trimmed).tensor();
        }
    }
    if (output.dtype() != original_dtype) {
        output = output.to(original_dtype);
    }

    auto result = Variable(output, input.requires_grad() || weight.requires_grad());

    if (input.requires_grad() || weight.requires_grad()) {
        std::vector<Tensor> tensors_to_save;
        if (bias_ptr != nullptr) {
            tensors_to_save = {input.tensor(), weight_matched.tensor(), *bias_ptr};
        } else {
            tensors_to_save = {input.tensor(), weight_matched.tensor()};
        }

        auto backward_fn = std::make_shared<ConvTranspose1dBackward>(
            stride_, padding_, output_padding_, dilation_, groups_,
            std::move(tensors_to_save)
        );

        result.set_grad_fn(backward_fn);

        std::vector<Variable> input_vars = {input, *parameters_["weight"]};
        if (bias_it != parameters_.end()) {
            input_vars.push_back(*bias_it->second);
        }
        backward_fn->set_input_variables(input_vars);

        // next_functions must have one slot per input variable in the SAME
        // order as input_vars so the engine's positional routing reaches
        // non-leaf weight/bias gradients (grad_fn() is nullptr for leaves).
        std::vector<std::shared_ptr<Function>> next_funcs;
        next_funcs.reserve(input_vars.size());
        for (const auto& var : input_vars) {
            next_funcs.push_back(var.grad_fn());
        }
        backward_fn->set_next_functions(next_funcs);
    }

    return result;
}

auto ConvTranspose1d::reset_parameters() -> void {
    // fan_in for transposed weight = (out/groups)*kernel_size.
    int64_t fan_in = (out_channels_ / groups_) * kernel_size_;
    float std = std::sqrt(2.0f / fan_in);

    std::vector<int64_t> weight_shape = {in_channels_, out_channels_ / groups_, kernel_size_};
    auto new_weight_tensor = randn(weight_shape) * std;
    parameters_["weight"] = std::make_shared<Variable>(new_weight_tensor, true);

    auto bias_it = parameters_.find("bias");
    if (bias_it != parameters_.end()) {
        std::vector<int64_t> bias_shape = {out_channels_};
        float bound = 1.0f / std::sqrt(static_cast<float>(fan_in));
        auto new_bias_tensor = (rand(bias_shape) * 2.0f * bound) - bound;
        bias_it->second = std::make_shared<Variable>(new_bias_tensor, true);
    }
}

// ============================================================================
// DeformableConv2dBackward - Autograd Function
// ============================================================================

class DeformableConv2dBackward : public Function {
public:
    DeformableConv2dBackward(int64_t stride, int64_t padding, int64_t dilation,
                              int64_t groups, int64_t offset_groups, bool use_mask,
                              std::vector<Tensor> tensors_to_save)
        : stride_(stride), padding_(padding), dilation_(dilation),
          groups_(groups), offset_groups_(offset_groups), use_mask_(use_mask) {
        save_for_backward(std::move(tensors_to_save));
    }

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("DeformableConv2dBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        const Tensor& grad_output = grad_outputs[0];
        const Tensor& input = saved_tensors_[0];
        const Tensor& offset = saved_tensors_[1];
        const Tensor& weight = saved_tensors_[2];
        const Tensor& mask = saved_tensors_[3];
        bool has_bias = saved_tensors_.size() > 4;

        OpAttributes backward_attrs;
        backward_attrs.set(AttrKey::StrideH, stride_);
        backward_attrs.set(AttrKey::StrideW, stride_);
        backward_attrs.set(AttrKey::PaddingH, padding_);
        backward_attrs.set(AttrKey::PaddingW, padding_);
        backward_attrs.set(AttrKey::DilationH, dilation_);
        backward_attrs.set(AttrKey::DilationW, dilation_);
        backward_attrs.set(AttrKey::Groups, groups_);
        backward_attrs.set(AttrKey::OffsetGroups, offset_groups_);
        backward_attrs.set(AttrKey::UseMask, use_mask_ ? 1 : 0);

        // Set shape attributes for backward kernels
        {
            auto is = input.shape();
            std::string is_str;
            for (size_t i = 0; i < is.size(); ++i) {
                if (i > 0) is_str += ',';
                is_str += std::to_string(is[i]);
            }
            backward_attrs.set(AttrKey::InputShape, is_str);

            auto ws = weight.shape();
            std::string ws_str;
            for (size_t i = 0; i < ws.size(); ++i) {
                if (i > 0) ws_str += ',';
                ws_str += std::to_string(ws[i]);
            }
            backward_attrs.set(AttrKey::WeightShape, ws_str);
        }

        // BackwardInput: returns {grad_input, grad_offset, grad_mask}
        std::vector<Tensor> bwd_input_inputs = {grad_output, input, offset, weight, mask};
        auto bwd_input_result = dispatch(OpId::DeformableConv2dBackwardInput,
                                          bwd_input_inputs, backward_attrs);

        // BackwardWeight: returns {grad_weight}
        std::vector<Tensor> bwd_weight_inputs = {grad_output, input, offset, mask};
        auto bwd_weight_result = dispatch(OpId::DeformableConv2dBackwardWeight,
                                           bwd_weight_inputs, backward_attrs);

        // Result order: grad_input, grad_offset, grad_weight, grad_bias, grad_mask
        std::vector<Tensor> grads;
        grads.push_back(bwd_input_result[0]);   // grad_input
        grads.push_back(bwd_input_result[1]);   // grad_offset
        grads.push_back(bwd_weight_result[0]);  // grad_weight

        if (has_bias) {
            std::vector<Tensor> bwd_bias_inputs = {grad_output};
            auto bwd_bias_result = dispatch(OpId::DeformableConv2dBackwardBias,
                                             bwd_bias_inputs, backward_attrs);
            grads.push_back(bwd_bias_result[0]); // grad_bias
        }

        if (use_mask_ && bwd_input_result.size() > 2) {
            grads.push_back(bwd_input_result[2]); // grad_mask
        }

        return grads;
    }

    TENZOR_HIGHER_ORDER_STRUCTURAL_ZERO_STUB()

private:
    int64_t stride_;
    int64_t padding_;
    int64_t dilation_;
    int64_t groups_;
    int64_t offset_groups_;
    bool use_mask_;
};

// ============================================================================
// DeformableConv2d - NN Layer
// ============================================================================

DeformableConv2d::DeformableConv2d(int64_t in_channels, int64_t out_channels,
                                     int64_t kernel_size, int64_t stride,
                                     int64_t padding, int64_t dilation,
                                     int64_t groups, int64_t offset_groups,
                                     bool bias)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride), padding_(padding),
      dilation_(dilation), groups_(groups), offset_groups_(offset_groups) {

    if (in_channels % groups != 0) {
        throw std::invalid_argument("in_channels must be divisible by groups");
    }
    if (out_channels % groups != 0) {
        throw std::invalid_argument("out_channels must be divisible by groups");
    }
    if (in_channels % offset_groups != 0) {
        throw std::invalid_argument("in_channels must be divisible by offset_groups");
    }

    std::vector<int64_t> weight_shape = {out_channels, in_channels / groups,
                                          kernel_size, kernel_size};
    auto weight_init = Variable(zeros(weight_shape, DType::Float32, Device::cpu()), true);
    register_parameter("weight", weight_init);

    if (bias) {
        std::vector<int64_t> bias_shape = {out_channels};
        auto bias_init = Variable(zeros(bias_shape, DType::Float32, Device::cpu()), true);
        register_parameter("bias", bias_init);
    }

    reset_parameters();
}

auto DeformableConv2d::forward(const Variable& input, const Variable& offset,
                                const Variable& mask) -> Variable {
    auto& weight = *parameters_["weight"];
    auto bias_it = parameters_.find("bias");
    bool has_bias = bias_it != parameters_.end();

    Tensor bias_t = has_bias ? bias_it->second->tensor() : zeros({0}, input.dtype(), input.tensor().device());
    bool use_mask = mask.is_initialized() && mask.tensor().numel() > 0;
    Tensor mask_t = use_mask ? mask.tensor() : zeros({0}, input.dtype(), input.tensor().device());

    OpAttributes forward_attrs;
    forward_attrs.set(AttrKey::StrideH, stride_);
    forward_attrs.set(AttrKey::StrideW, stride_);
    forward_attrs.set(AttrKey::PaddingH, padding_);
    forward_attrs.set(AttrKey::PaddingW, padding_);
    forward_attrs.set(AttrKey::DilationH, dilation_);
    forward_attrs.set(AttrKey::DilationW, dilation_);
    forward_attrs.set(AttrKey::Groups, groups_);
    forward_attrs.set(AttrKey::OffsetGroups, offset_groups_);
    forward_attrs.set(AttrKey::UseMask, use_mask ? 1 : 0);

    std::vector<Tensor> inputs_vec = {input.tensor(), offset.tensor(),
                                       weight.tensor(), bias_t, mask_t};

    auto output_result = dispatch_to_device(OpId::DeformableConv2dForward,
        input.tensor().device().type, inputs_vec, forward_attrs);
    Tensor output = output_result[0];

    bool needs_grad = input.requires_grad() || offset.requires_grad() ||
                      weight.requires_grad() ||
                      (use_mask && mask.requires_grad());
    auto result = Variable(output, needs_grad);

    if (needs_grad) {
        std::vector<Tensor> tensors_to_save = {input.tensor(), offset.tensor(),
                                                weight.tensor(), mask_t};
        if (has_bias) {
            tensors_to_save.push_back(bias_it->second->tensor());
        }

        auto backward_fn = std::make_shared<DeformableConv2dBackward>(
            stride_, padding_, dilation_, groups_, offset_groups_, use_mask,
            std::move(tensors_to_save)
        );

        result.set_grad_fn(backward_fn);

        std::vector<Variable> input_vars = {input, offset, weight};
        if (has_bias) {
            input_vars.push_back(*bias_it->second);
        }
        if (use_mask) {
            input_vars.push_back(mask);
        }
        backward_fn->set_input_variables(input_vars);

        // next_functions must have one slot per input variable in the SAME
        // order as input_vars (input, offset, weight[, bias][, mask]) so the
        // engine's positional routing reaches non-leaf weight/bias/mask
        // gradients (grad_fn() is nullptr for leaves).
        std::vector<std::shared_ptr<Function>> next_funcs;
        next_funcs.reserve(input_vars.size());
        for (const auto& var : input_vars) next_funcs.push_back(var.grad_fn());
        backward_fn->set_next_functions(next_funcs);
    }

    return result;
}

auto DeformableConv2d::forward_impl(const Variable& /*input*/) -> Variable {
    throw std::runtime_error(
        "DeformableConv2d::forward_impl: use forward(input, offset, mask) instead");
}

auto DeformableConv2d::reset_parameters() -> void {
    int64_t fan_in = in_channels_ * kernel_size_ * kernel_size_;
    float std = std::sqrt(2.0f / fan_in);

    std::vector<int64_t> weight_shape = {out_channels_, in_channels_ / groups_,
                                          kernel_size_, kernel_size_};
    auto new_weight_tensor = randn(weight_shape) * std;
    parameters_["weight"] = std::make_shared<Variable>(new_weight_tensor, true);

    auto bias_it = parameters_.find("bias");
    if (bias_it != parameters_.end()) {
        std::vector<int64_t> bias_shape = {out_channels_};
        float bound = 1.0f / std::sqrt(static_cast<float>(fan_in));
        auto new_bias_tensor = (rand(bias_shape) * 2.0f * bound) - bound;
        bias_it->second = std::make_shared<Variable>(new_bias_tensor, true);
    }
}

namespace internal {
auto make_conv_transpose3d_backward(int64_t stride, int64_t padding,
                                    int64_t output_padding, int64_t dilation,
                                    int64_t groups,
                                    std::vector<::tenzor::Tensor> tensors_to_save)
    -> std::shared_ptr<::tenzor::Function> {
    return std::make_shared<ConvTranspose3dBackward>(
        stride, padding, output_padding, dilation, groups,
        std::move(tensors_to_save));
}

// F127: anisotropic (per-axis) overload. ConvTranspose3dBackward already stores
// and uses per-axis stride/padding/output_padding/dilation in its backward
// math, so F::conv_transpose3d can wire asymmetric values directly.
auto make_conv_transpose3d_backward(int64_t sD, int64_t sH, int64_t sW,
                                    int64_t pD, int64_t pH, int64_t pW,
                                    int64_t opD, int64_t opH, int64_t opW,
                                    int64_t dD, int64_t dH, int64_t dW,
                                    int64_t groups,
                                    std::vector<::tenzor::Tensor> tensors_to_save)
    -> std::shared_ptr<::tenzor::Function> {
    return std::make_shared<ConvTranspose3dBackward>(
        sD, sH, sW, pD, pH, pW, opD, opH, opW, dD, dH, dW, groups,
        std::move(tensors_to_save));
}
} // namespace internal

} // namespace tenzor::nn
