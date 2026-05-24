/**
 * @file functional.cpp
 * @brief Implementations for non-trivial nn::functional operations
 *
 * Conv, pooling, normalization, dropout, and padding functional wrappers
 * that dispatch directly to backend kernels.
 */

#include "tenzor/nn/functional.hpp"
#include "tenzor/nn/layers/normalization.hpp"  // for internal::make_layer_norm_backward
#include "tenzor/nn/layers/embedding.hpp"      // for internal::make_embedding_backward
#include "tenzor/nn/layers/pooling.hpp"        // J7: delegate pool functional to Module
#include "tenzor/nn/activations/activations.hpp" // U.13: nn::relu for prelu composition
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/autograd/function.hpp"
#include "conv_autograd.hpp"
#include "conv3d_autograd.hpp"
#include <stdexcept>
#include <cmath>
#include <cstring>

namespace tenzor::nn::functional {

namespace {

// Serialize a shape vector as the comma-separated int-list the backward
// kernel registries expect via AttrKey::InputShape.
inline std::string shape_to_csv(const std::vector<int64_t>& shape) {
    std::string s;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) s += ',';
        s += std::to_string(shape[i]);
    }
    return s;
}

// Backward node for ops whose backward kernel takes
// [grad_output, indices] and attr InputShape, returning grad_input.
// Covers MaxUnpool2d/3dBackward and FractionalMaxPool2d/3dBackward.
template <OpId BackwardOp>
class IndexedPoolBackward : public Function {
public:
    IndexedPoolBackward(std::vector<int64_t> input_shape, Tensor indices)
        : input_shape_(std::move(input_shape)) {
        save_for_backward({std::move(indices)});
    }

    auto forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> override {
        throw std::runtime_error("IndexedPoolBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        OpAttributes attrs;
        attrs.set(AttrKey::InputShape, shape_to_csv(input_shape_));
        std::vector<Tensor> inputs = {grad_outputs[0], saved_tensors_[0]};
        auto result = dispatch_to_device(BackwardOp,
            grad_outputs[0].device().type, inputs, attrs);
        return {result[0]};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs)
        -> std::vector<Variable> override {
        std::vector<Tensor> tensor_grads;
        for (auto& v : grad_outputs) tensor_grads.push_back(v.tensor());
        auto results = backward(std::move(tensor_grads));
        std::vector<Variable> var_results;
        for (auto& t : results) var_results.emplace_back(t, false);
        return var_results;
    }

    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return true; }

private:
    std::vector<int64_t> input_shape_;
};

inline void wire_pool_grad_fn(Variable& result, const Variable& input,
                              std::shared_ptr<Function> backward_fn) {
    result.set_grad_fn(backward_fn);
    std::vector<Variable> input_vars{input};
    backward_fn->set_input_variables(input_vars);
    std::vector<std::shared_ptr<Function>> next_funcs;
    if (input.grad_fn()) next_funcs.push_back(input.grad_fn());
    backward_fn->set_next_functions(next_funcs);
}

}  // anonymous namespace

// ============================================================================
// Convolution
// ============================================================================

auto conv2d(const Variable& input, const Variable& weight,
            const std::optional<Variable>& bias,
            std::pair<int64_t, int64_t> stride,
            std::pair<int64_t, int64_t> padding,
            std::pair<int64_t, int64_t> dilation,
            int64_t groups) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument(
            "F::conv2d expects 4D input [N, C_in, H, W], got " +
            std::to_string(input_shape.size()) + "D");
    }

    std::vector<Tensor> inputs_vec = {input.tensor(), weight.tensor()};
    if (bias.has_value()) {
        inputs_vec.push_back(bias->tensor());
    }

    // Audit E5: asymmetric stride/padding/dilation guard removed. Each
    // backend now has an honest per-axis contract:
    //   - CPU + CUDA: produce correct output for asymmetric values (E1).
    //   - ROCm / OneAPI / Vulkan: throw a clear backend-level error when
    //     the per-axis values differ from each other (E2/E3/E4), pointing
    //     at the kernel-side refactor still pending for each. Removing the
    //     functional-level guard lets the backend error message — which
    //     names the specific backend — surface to the user.
    NewOpAttributes attrs;
    attrs.set(AttrKey::Stride, stride.first);
    attrs.set(AttrKey::Padding, padding.first);
    attrs.set(AttrKey::Dilation, dilation.first);
    attrs.set(AttrKey::StrideH, stride.first);
    attrs.set(AttrKey::StrideW, stride.second);
    attrs.set(AttrKey::PaddingH, padding.first);
    attrs.set(AttrKey::PaddingW, padding.second);
    attrs.set(AttrKey::DilationH, dilation.first);
    attrs.set(AttrKey::DilationW, dilation.second);
    attrs.set(AttrKey::Groups, groups);

    auto result = dispatch_to_device(OpId::Conv2dForward,
        input.tensor().device().type, inputs_vec, attrs);

    bool requires_grad = input.requires_grad() || weight.requires_grad() ||
                         (bias.has_value() && bias->requires_grad());
    Variable output(result[0], requires_grad);

    // Wire the autograd graph so backward() through F::conv2d actually
    // populates input/weight/bias grads. Previously this path returned a
    // Variable with no grad_fn, so anything using the functional API
    // (not the Conv2d module) silently had no gradient.
    if (requires_grad && ::tenzor::is_grad_enabled()) {
        std::vector<Tensor> tensors_to_save = {input.tensor(), weight.tensor()};
        if (bias.has_value()) {
            tensors_to_save.push_back(bias->tensor());
        }
        auto grad_fn = std::make_shared<internal::Conv2dBackward>(
            stride.first, stride.second,
            padding.first, padding.second,
            dilation.first, dilation.second,
            groups,
            std::move(tensors_to_save));

        // Save Variables for higher-order gradient support (create_graph=true)
        if (::tenzor::is_creating_graph()) {
            std::vector<Variable> vars_to_save = {input, weight};
            if (bias.has_value()) vars_to_save.push_back(*bias);
            grad_fn->save_variables_for_backward(std::move(vars_to_save));
        }

        std::vector<Variable> input_vars = {input, weight};
        if (bias.has_value()) input_vars.push_back(*bias);
        grad_fn->set_input_variables(input_vars);

        std::vector<std::shared_ptr<::tenzor::Function>> next_funcs;
        next_funcs.push_back(input.grad_fn());
        next_funcs.push_back(weight.grad_fn());
        if (bias.has_value()) next_funcs.push_back(bias->grad_fn());
        grad_fn->set_next_functions(next_funcs);

        output.set_grad_fn(grad_fn);
    }

    return output;
}

auto conv1d(const Variable& input, const Variable& weight,
            const std::optional<Variable>& bias,
            int64_t stride, int64_t padding, int64_t dilation,
            int64_t groups) -> Variable {
    if (input.shape().size() != 3) {
        throw std::invalid_argument(
            "F::conv1d expects 3D input [N, C_in, L], got " +
            std::to_string(input.shape().size()) + "D");
    }

    // Wrap the 3D→4D unsqueeze / forward / 4D→3D squeeze as autograd ops so
    // backward() propagates through all three steps instead of stopping at
    // the raw dispatch_to_device and leaving input/weight with no grad.
    // Previously this path returned a Variable with no grad_fn — gradcheck
    // saw zero analytical gradient vs a real numerical gradient and failed.
    auto input_4d = ::tenzor::unsqueeze(input, 2);
    auto weight_4d = ::tenzor::unsqueeze(weight, 2);
    auto out_4d = conv2d(input_4d, weight_4d, bias,
                         /*stride=*/ {int64_t{1}, stride},
                         /*padding=*/{int64_t{0}, padding},
                         /*dilation=*/{int64_t{1}, dilation},
                         groups);
    return ::tenzor::squeeze(out_4d, 2);
}

// Audit-4 U.13: free-function PReLU built as a pure Variable-level
// composition (the same identity the ``nn::PReLU`` module evaluates), so
// grad flows back to the caller's ``input`` and ``weight`` Variables rather
// than to a throwaway layer's internal Parameter.
//
//   prelu(x, w) = relu(x) + w * (x - relu(x))
//                = max(0, x) + w * min(0, x)
//
// ``weight`` is broadcast across input.  When ``weight.numel() == 1`` it
// applies as a scalar slope; when ``weight.numel() == C`` it is reshaped to
// ``[1, C, 1, ..., 1]`` to align with input's channel dim before the
// elementwise multiply, matching PyTorch / nn::PReLU semantics.
auto prelu(const Variable& input, const Variable& weight) -> Variable {
    const int64_t w_numel = weight.tensor().numel();
    Variable broadcast_weight = weight;
    if (w_numel != 1) {
        const auto& in_shape = input.shape();
        if (in_shape.size() < 2) {
            throw std::invalid_argument(
                "F::prelu per-channel weight requires input with at least 2 dims "
                "(got " + std::to_string(in_shape.size()) + "D); use a 1-element "
                "weight for scalar slope on lower-rank inputs.");
        }
        const int64_t c = in_shape[1];
        if (w_numel != c) {
            throw std::invalid_argument(
                "F::prelu weight numel (" + std::to_string(w_numel) +
                ") must equal 1 or input channel dim (" + std::to_string(c) + ")");
        }
        // Reshape weight to [1, C, 1, ..., 1] so the broadcast lines up
        // with input's channel axis under standard NCHW / NCL / NC layouts.
        std::vector<int64_t> w_shape(in_shape.size(), int64_t{1});
        w_shape[1] = c;
        broadcast_weight = ::tenzor::reshape(weight, w_shape);
    }
    auto relu_x = ::tenzor::nn::relu(input);
    auto neg_part = input - relu_x;       // min(0, x), autograd-safe via Variable op
    return relu_x + broadcast_weight * neg_part;
}

auto conv3d(const Variable& input, const Variable& weight,
            const std::optional<Variable>& bias,
            std::tuple<int64_t, int64_t, int64_t> stride,
            std::tuple<int64_t, int64_t, int64_t> padding,
            std::tuple<int64_t, int64_t, int64_t> dilation,
            int64_t groups) -> Variable {
    if (input.shape().size() != 5) {
        throw std::invalid_argument(
            "F::conv3d expects 5D input [N, C_in, D, H, W], got " +
            std::to_string(input.shape().size()) + "D");
    }

    std::vector<Tensor> inputs_vec = {input.tensor(), weight.tensor()};
    if (bias.has_value()) {
        inputs_vec.push_back(bias->tensor());
    }

    auto [sd, sh, sw] = stride;
    auto [pd, ph, pw] = padding;
    auto [dd, dh, dw] = dilation;

    NewOpAttributes attrs;
    attrs.set(AttrKey::Stride, sd);
    attrs.set(AttrKey::Padding, pd);
    attrs.set(AttrKey::Dilation, dd);
    attrs.set(AttrKey::StrideH, sh);
    attrs.set(AttrKey::StrideW, sw);
    attrs.set(AttrKey::PaddingH, ph);
    attrs.set(AttrKey::PaddingW, pw);
    attrs.set(AttrKey::DilationH, dh);
    attrs.set(AttrKey::DilationW, dw);
    attrs.set(AttrKey::Groups, groups);

    auto result = dispatch_to_device(OpId::Conv3dForward,
        input.tensor().device().type, inputs_vec, attrs);

    bool requires_grad = input.requires_grad() || weight.requires_grad() ||
                         (bias.has_value() && bias->requires_grad());
    Variable output(result[0], requires_grad);

    // Wire autograd using the existing Conv3dBackward from the nn::Conv3d
    // module. That class takes isotropic stride/padding/dilation (it was
    // built for the module API), so F::conv3d autograd here also requires
    // isotropic parameters. Anisotropic conv3d backward support requires
    // extending Conv3dBackward to per-axis params and is tracked separately.
    if (requires_grad && ::tenzor::is_grad_enabled()) {
        if (sd != sh || sd != sw || pd != ph || pd != pw || dd != dh || dd != dw) {
            throw std::runtime_error(
                "F::conv3d autograd currently requires isotropic "
                "stride/padding/dilation (got asymmetric values). "
                "Use nn::Conv3d module or file an issue for anisotropic support.");
        }
        std::vector<Tensor> tensors_to_save = {input.tensor(), weight.tensor()};
        if (bias.has_value()) tensors_to_save.push_back(bias->tensor());
        auto grad_fn = internal::make_conv3d_backward(sd, pd, dd, groups, std::move(tensors_to_save));
        std::vector<Variable> input_vars = {input, weight};
        if (bias.has_value()) input_vars.push_back(*bias);
        grad_fn->set_input_variables(input_vars);
        std::vector<std::shared_ptr<::tenzor::Function>> next_funcs;
        next_funcs.push_back(input.grad_fn());
        next_funcs.push_back(weight.grad_fn());
        if (bias.has_value()) next_funcs.push_back(bias->grad_fn());
        grad_fn->set_next_functions(std::move(next_funcs));
        output.set_grad_fn(std::move(grad_fn));
    }

    return output;
}

auto conv_transpose1d(const Variable& input, const Variable& weight,
                      const std::optional<Variable>& bias,
                      int64_t stride, int64_t padding, int64_t output_padding,
                      int64_t groups, int64_t dilation) -> Variable {
    if (input.shape().size() != 3) {
        throw std::invalid_argument(
            "F::conv_transpose1d expects 3D input [N, C_in, L], got " +
            std::to_string(input.shape().size()) + "D");
    }

    // Go through the autograd-aware conv_transpose2d so backward flows.
    // ConvTranspose2dBackward only supports isotropic params, but here
    // the H dim is a fake singleton so feeding it the scalar 1D params is
    // safe — H/W scalars happen to match (stride_h=1=stride itself when
    // stride=1; padding_h=0=padding itself when padding=0; etc.). For
    // non-unit stride we require H-direction params to match the W-direction
    // params, which is trivially true because H is of length 1.
    auto input_4d = ::tenzor::unsqueeze(input, 2);
    auto weight_4d = ::tenzor::unsqueeze(weight, 2);
    auto out_4d = conv_transpose2d(
        input_4d, weight_4d, bias,
        /*stride=*/  {stride,   stride},
        /*padding=*/ {padding,  padding},
        /*output_padding=*/ {output_padding, output_padding},
        groups,
        /*dilation=*/{dilation, dilation});
    return ::tenzor::squeeze(out_4d, 2);
}

auto conv_transpose2d(const Variable& input, const Variable& weight,
                      const std::optional<Variable>& bias,
                      std::pair<int64_t, int64_t> stride,
                      std::pair<int64_t, int64_t> padding,
                      [[maybe_unused]] std::pair<int64_t, int64_t> output_padding,
                      int64_t groups,
                      std::pair<int64_t, int64_t> dilation) -> Variable {
    if (input.shape().size() != 4) {
        throw std::invalid_argument(
            "F::conv_transpose2d expects 4D input [N, C_in, H, W], got " +
            std::to_string(input.shape().size()) + "D");
    }

    std::vector<Tensor> inputs_vec = {input.tensor(), weight.tensor()};
    if (bias.has_value()) {
        inputs_vec.push_back(bias->tensor());
    }

    NewOpAttributes attrs;
    attrs.set(AttrKey::Stride, stride.first);
    attrs.set(AttrKey::Padding, padding.first);
    attrs.set(AttrKey::Dilation, dilation.first);
    attrs.set(AttrKey::StrideH, stride.first);
    attrs.set(AttrKey::StrideW, stride.second);
    attrs.set(AttrKey::PaddingH, padding.first);
    attrs.set(AttrKey::PaddingW, padding.second);
    attrs.set(AttrKey::Groups, groups);

    auto result = dispatch_to_device(OpId::ConvTranspose2dForward,
        input.tensor().device().type, inputs_vec, attrs);

    bool requires_grad = input.requires_grad() || weight.requires_grad() ||
                         (bias.has_value() && bias->requires_grad());
    Variable output(result[0], requires_grad);

    // Wire autograd via the ConvTranspose2dBackward class already living in
    // nn::Conv2dTranspose. Previously F::conv_transpose2d returned a
    // Variable with no grad_fn, so any backward through it silently dropped
    // gradients. ConvTranspose2dBackward is isotropic; we enforce that here.
    if (requires_grad && ::tenzor::is_grad_enabled()) {
        if (stride.first != stride.second || padding.first != padding.second ||
            dilation.first != dilation.second) {
            throw std::runtime_error(
                "F::conv_transpose2d autograd currently requires isotropic "
                "stride/padding/dilation (got asymmetric values).");
        }
        std::vector<Tensor> tensors_to_save = {input.tensor(), weight.tensor()};
        if (bias.has_value()) tensors_to_save.push_back(bias->tensor());
        auto grad_fn = internal::make_conv_transpose2d_backward(
            stride.first, padding.first, /*output_padding=*/output_padding.first,
            dilation.first, groups, std::move(tensors_to_save));
        std::vector<Variable> input_vars = {input, weight};
        if (bias.has_value()) input_vars.push_back(*bias);
        grad_fn->set_input_variables(input_vars);
        std::vector<std::shared_ptr<::tenzor::Function>> next_funcs;
        next_funcs.push_back(input.grad_fn());
        next_funcs.push_back(weight.grad_fn());
        if (bias.has_value()) next_funcs.push_back(bias->grad_fn());
        grad_fn->set_next_functions(std::move(next_funcs));
        output.set_grad_fn(std::move(grad_fn));
    }
    return output;
}

auto conv_transpose3d(const Variable& input, const Variable& weight,
                      const std::optional<Variable>& bias,
                      std::tuple<int64_t, int64_t, int64_t> stride,
                      std::tuple<int64_t, int64_t, int64_t> padding,
                      [[maybe_unused]] std::tuple<int64_t, int64_t, int64_t> output_padding,
                      int64_t groups,
                      std::tuple<int64_t, int64_t, int64_t> dilation) -> Variable {
    if (input.shape().size() != 5) {
        throw std::invalid_argument(
            "F::conv_transpose3d expects 5D input [N, C_in, D, H, W], got " +
            std::to_string(input.shape().size()) + "D");
    }

    std::vector<Tensor> inputs_vec = {input.tensor(), weight.tensor()};
    if (bias.has_value()) {
        inputs_vec.push_back(bias->tensor());
    }

    auto [sd, sh, sw] = stride;
    auto [pd, ph, pw] = padding;
    auto [dd, dh, dw] = dilation;

    NewOpAttributes attrs;
    attrs.set(AttrKey::Stride, sd);
    attrs.set(AttrKey::Padding, pd);
    attrs.set(AttrKey::Dilation, dd);
    attrs.set(AttrKey::StrideH, sh);
    attrs.set(AttrKey::StrideW, sw);
    attrs.set(AttrKey::PaddingH, ph);
    attrs.set(AttrKey::PaddingW, pw);
    attrs.set(AttrKey::DilationH, dh);
    attrs.set(AttrKey::DilationW, dw);
    attrs.set(AttrKey::Groups, groups);

    auto result = dispatch_to_device(OpId::ConvTranspose3dForward,
        input.tensor().device().type, inputs_vec, attrs);

    bool requires_grad = input.requires_grad() || weight.requires_grad() ||
                         (bias.has_value() && bias->requires_grad());
    Variable output(result[0], requires_grad);

    // audit-2026-05-03 — wire autograd grad_fn so backward through F::
    // conv_transpose3d propagates gradients (previously this returned a
    // Variable with no grad_fn → silent zero gradients).
    if (requires_grad && ::tenzor::is_grad_enabled()) {
        if (sd != sh || sd != sw || pd != ph || pd != pw || dd != dh || dd != dw) {
            throw std::runtime_error(
                "F::conv_transpose3d autograd currently requires isotropic "
                "stride/padding/dilation (got asymmetric values).");
        }
        auto [opd, oph, opw] = output_padding;
        std::vector<Tensor> tensors_to_save = {input.tensor(), weight.tensor()};
        if (bias.has_value()) tensors_to_save.push_back(bias->tensor());
        auto grad_fn = internal::make_conv_transpose3d_backward(
            sd, pd, /*output_padding=*/opd, dd, groups, std::move(tensors_to_save));
        std::vector<Variable> input_vars = {input, weight};
        if (bias.has_value()) input_vars.push_back(*bias);
        grad_fn->set_input_variables(input_vars);
        std::vector<std::shared_ptr<::tenzor::Function>> next_funcs;
        next_funcs.push_back(input.grad_fn());
        next_funcs.push_back(weight.grad_fn());
        if (bias.has_value()) next_funcs.push_back(bias->grad_fn());
        grad_fn->set_next_functions(std::move(next_funcs));
        output.set_grad_fn(std::move(grad_fn));
    }
    return output;
}

// ============================================================================
// Pooling
// ============================================================================

auto max_pool2d(const Variable& input,
                std::pair<int64_t, int64_t> kernel_size,
                std::pair<int64_t, int64_t> stride,
                std::pair<int64_t, int64_t> padding) -> Variable {
    if (input.shape().size() != 4) {
        throw std::invalid_argument(
            "F::max_pool2d expects 4D input [N, C, H, W]");
    }

    // Default stride = kernel_size
    if (stride.first < 0) stride.first = kernel_size.first;
    if (stride.second < 0) stride.second = kernel_size.second;

    // J7: Delegate to the Module so the backward (MaxPool2dBackward) is
    // wired up. The previous direct-dispatch path built a leaf Variable
    // without a grad_fn and silently produced zero gradients.
    //
    // MaxPool2d Module supports square pooling; this functional signature
    // takes pairs but the previous code already ignored `.second`, so the
    // behavior is preserved — non-square pooling was never actually
    // supported here despite the API shape.
    ::tenzor::nn::MaxPool2d pool(kernel_size.first,
                                 stride.first,
                                 padding.first,
                                 /*ceil_mode=*/false,
                                 /*return_indices=*/false);
    return pool.forward(input);
}

auto avg_pool2d(const Variable& input,
                std::pair<int64_t, int64_t> kernel_size,
                std::pair<int64_t, int64_t> stride,
                std::pair<int64_t, int64_t> padding) -> Variable {
    if (input.shape().size() != 4) {
        throw std::invalid_argument(
            "F::avg_pool2d expects 4D input [N, C, H, W]");
    }

    // Default stride = kernel_size
    if (stride.first < 0) stride.first = kernel_size.first;
    if (stride.second < 0) stride.second = kernel_size.second;

    // J7: Delegate to the Module (same rationale as max_pool2d above).
    ::tenzor::nn::AvgPool2d pool(kernel_size.first,
                                 stride.first,
                                 padding.first);
    return pool.forward(input);
}

auto adaptive_avg_pool2d(const Variable& input,
                         std::pair<int64_t, int64_t> output_size) -> Variable {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument(
            "F::adaptive_avg_pool2d expects 4D input [N, C, H, W]");
    }

    int64_t in_h = shape[2];
    int64_t in_w = shape[3];
    int64_t out_h = output_size.first;
    int64_t out_w = output_size.second;

    // Compute adaptive kernel_size and stride
    int64_t kernel_h = in_h - (out_h - 1) * (in_h / out_h);
    int64_t kernel_w = in_w - (out_w - 1) * (in_w / out_w);
    int64_t stride_h = in_h / out_h;
    int64_t stride_w = in_w / out_w;

    // Use kernel_size = ceil(in/out), stride = floor(in/out) for exact output size
    if (out_h > 0) {
        stride_h = in_h / out_h;
        kernel_h = in_h - (out_h - 1) * stride_h;
    }
    if (out_w > 0) {
        stride_w = in_w / out_w;
        kernel_w = in_w - (out_w - 1) * stride_w;
    }

    return avg_pool2d(input, {kernel_h, kernel_w}, {stride_h, stride_w}, {0, 0});
}

// ============================================================================
// Normalization
// ============================================================================

auto batch_norm(const Variable& input,
                const Tensor& running_mean,
                const Tensor& running_var,
                const std::optional<Variable>& weight,
                const std::optional<Variable>& bias,
                bool training,
                [[maybe_unused]] double momentum,
                double eps) -> Variable {
    if (training) {
        // Training mode: compute batch statistics using dispatch
        auto input_t = input.tensor();

        std::vector<Tensor> norm_inputs = {input_t};
        NewOpAttributes mean_var_attrs;
        auto stats = dispatch(OpId::BatchNorm2dMeanVar, norm_inputs, mean_var_attrs);

        Tensor batch_mean = stats[0];
        Tensor batch_var = stats[1];

        // Normalize
        std::vector<Tensor> fwd_inputs = {input_t, batch_mean, batch_var};
        if (weight.has_value()) fwd_inputs.push_back(weight->tensor());
        if (bias.has_value()) fwd_inputs.push_back(bias->tensor());

        NewOpAttributes fwd_attrs;
        fwd_attrs.set(AttrKey::Eps, eps);
        auto output = dispatch(OpId::BatchNorm2dForwardAffine, fwd_inputs, fwd_attrs);

        return Variable(output[0], input.requires_grad());
    } else {
        // Eval mode: use running statistics
        std::vector<Tensor> inputs_vec = {input.tensor(), running_mean, running_var};
        if (weight.has_value()) inputs_vec.push_back(weight->tensor());
        if (bias.has_value()) inputs_vec.push_back(bias->tensor());

        NewOpAttributes attrs;
        attrs.set(AttrKey::Eps, eps);
        auto output = dispatch(OpId::BatchNorm2dForwardAffine, inputs_vec, attrs);

        return Variable(output[0], input.requires_grad());
    }
}

auto layer_norm(const Variable& input,
                std::vector<int64_t> normalized_shape,
                const std::optional<Variable>& weight,
                const std::optional<Variable>& bias,
                double eps) -> Variable {
    // Backend FusedLayerNorm kernels (CPU/CUDA/ROCm/Vulkan/OneAPI) all read
    // inputs[1] (weight) and inputs[2] (bias) unconditionally. Synthesize an
    // identity affine here when weight/bias are absent so the inputs span
    // always has 3 elements regardless of which backend dispatches.
    //
    // Every FusedLayerNorm kernel reads the input with shape-derived flat
    // offsets, so it requires contiguous storage. Materialize a contiguous
    // copy when the caller passed a non-contig view (slice / permute /
    // narrow / transpose). The LayerNorm Module's forward_impl does the
    // same — keep this functional entry-point consistent.
    Tensor input_tensor = input.tensor();
    if (!input_tensor.is_contiguous()) {
        input_tensor = input_tensor.contiguous();
    }
    std::vector<Tensor> inputs_vec = {input_tensor};
    if (weight.has_value()) {
        Tensor w = weight->tensor();
        if (!w.is_contiguous()) w = w.contiguous();
        inputs_vec.push_back(w);
    } else {
        inputs_vec.push_back(::tenzor::ones(
            normalized_shape, input_tensor.dtype(), input_tensor.device()));
    }
    if (bias.has_value()) {
        Tensor b = bias->tensor();
        if (!b.is_contiguous()) b = b.contiguous();
        inputs_vec.push_back(b);
    } else {
        inputs_vec.push_back(::tenzor::zeros(
            normalized_shape, input_tensor.dtype(), input_tensor.device()));
    }

    // Build normalized_shape as comma-separated string attribute
    std::string shape_str;
    for (size_t i = 0; i < normalized_shape.size(); ++i) {
        if (i > 0) shape_str += ",";
        shape_str += std::to_string(normalized_shape[i]);
    }

    NewOpAttributes attrs;
    attrs.set(AttrKey::NormalizedShape, shape_str);
    attrs.set(AttrKey::Eps, eps);

    auto result = dispatch(OpId::FusedLayerNorm, inputs_vec, attrs);
    Variable output(result[0], input.requires_grad());

    // Phase 24-followup #38 fix: previously this functional returned a
    // Variable with no grad_fn — backward through F::layer_norm produced
    // zero gradients silently. Wire up nn::LayerNormBackward (defined in
    // src/nn/layers/normalization.cpp) so backward flows. FusedLayerNorm
    // returns 3 tensors on CPU/CUDA/OneAPI ([output, mean, inv_std]) but
    // only 1 on Vulkan/ROCm. For the latter, compute mean/inv_std via
    // tensor ops so the backward path still has the saved stats it needs.
    if (input.requires_grad() && ::tenzor::is_grad_enabled()) {
        Tensor mean_t, rstd_t;
        if (result.size() >= 3) {
            mean_t = result[1];
            rstd_t = result[2];
        } else {
            // Compute mean/inv_std manually for backends whose FusedLayerNorm
            // doesn't return them (Vulkan, ROCm). Reduce over last
            // normalized_shape.size() dims.
            const Tensor& inp_t = input.tensor();
            int64_t input_ndim = inp_t.ndim();
            int64_t reduce_start = input_ndim - static_cast<int64_t>(normalized_shape.size());
            // Compute over the trailing dims by flattening to [batch, N].
            int64_t N = 1;
            for (auto d : normalized_shape) N *= d;
            int64_t batch = inp_t.numel() / N;
            std::vector<int64_t> flat_shape{batch, N};
            Tensor inp_flat = ::tenzor::reshape(inp_t, flat_shape);
            // mean over last dim, keeping shape [batch]
            Tensor mean_flat = ::tenzor::mean(inp_flat, /*dim=*/1, /*keepdim=*/false);
            // var = mean(x^2) - mean^2
            Tensor sq = ::tenzor::mul(inp_flat, inp_flat);
            Tensor mean_sq = ::tenzor::mean(sq, /*dim=*/1, /*keepdim=*/false);
            Tensor mu_sq = ::tenzor::mul(mean_flat, mean_flat);
            Tensor var = ::tenzor::sub(mean_sq, mu_sq);
            Tensor var_eps = ::tenzor::add(var, ::tenzor::full(
                std::vector<int64_t>{batch}, eps, var.dtype(), var.device()));
            Tensor std_t = ::tenzor::sqrt(var_eps);
            Tensor ones_b = ::tenzor::ones({batch}, std_t.dtype(), std_t.device());
            Tensor rstd_flat = ::tenzor::div(ones_b, std_t);
            // Reshape to original batch dims (input's leading dims minus normalized).
            std::vector<int64_t> stat_shape;
            for (int64_t i = 0; i < reduce_start; ++i) stat_shape.push_back(inp_t.shape()[i]);
            if (stat_shape.empty()) stat_shape.push_back(1);
            mean_t = ::tenzor::reshape(mean_flat, stat_shape);
            rstd_t = ::tenzor::reshape(rstd_flat, stat_shape);
            (void)reduce_start;  // suppress unused warning if loop body skipped
        }

        bool elementwise_affine = weight.has_value();
        int64_t normalized_size = 1;
        for (auto d : normalized_shape) normalized_size *= d;
        std::vector<Tensor> tensors_to_save = {
            input.tensor(), mean_t, rstd_t, inputs_vec[1]
        };
        auto grad_fn = internal::make_layer_norm_backward(
            elementwise_affine, eps, normalized_size,
            std::vector<int64_t>(normalized_shape.begin(), normalized_shape.end()),
            std::move(tensors_to_save));
        output.set_grad_fn(grad_fn);
        std::vector<std::shared_ptr<::tenzor::Function>> next_funcs;
        if (auto fn = input.grad_fn()) next_funcs.push_back(fn);
        if (weight.has_value()) {
            if (auto fn = weight->grad_fn()) next_funcs.push_back(fn);
        }
        grad_fn->set_next_functions(std::move(next_funcs));
        std::vector<Variable> input_vars{input};
        if (weight.has_value()) input_vars.push_back(*weight);
        if (bias.has_value()) input_vars.push_back(*bias);
        grad_fn->set_input_variables(std::move(input_vars));
    }
    return output;
}

// ============================================================================
// Dropout
// ============================================================================

auto dropout(const Variable& input, double p, bool training) -> Variable {
    if (!training || p == 0.0) {
        return input;
    }
    if (p == 1.0) {
        // p=1 zeros every element. Multiply by a non-grad zero Variable so
        // the autograd graph still records a Mul (whose backward returns
        // zero w.r.t. input) — keeps types/usage uniform with the p<1 path.
        auto s = input.tensor().shape();
        std::vector<int64_t> sv(s.begin(), s.end());
        Variable zero_var(zeros(sv, input.dtype(), input.tensor().device()),
                          /*requires_grad=*/false);
        return input * zero_var;
    }

    // Generate the Bernoulli mask as a raw Tensor (no autograd needed for
    // the mask itself), then wrap as a non-grad Variable and use the
    // Variable-level operator* so backward flows through input. The
    // earlier implementation called tenzor::mul on raw tensors and wrapped
    // the result in a Variable with no grad_fn — silently zeroing
    // input.grad() (raw-tensor-op breaks autograd graph pattern).
    auto shape_span = input.tensor().shape();
    std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
    auto rand_t = tenzor::rand(shape_vec, input.dtype(), input.tensor().device());
    auto threshold = tenzor::full(shape_vec, static_cast<float>(p),
                                  input.dtype(), input.tensor().device());
    auto mask_bool = tenzor::gt(rand_t, threshold);
    auto ones_t = tenzor::ones(shape_vec, input.dtype(), input.tensor().device());
    auto zeros_t = tenzor::zeros(shape_vec, input.dtype(), input.tensor().device());
    auto mask_tensor = tenzor::where(mask_bool, ones_t, zeros_t);
    Variable mask_var(mask_tensor, /*requires_grad=*/false);

    // Inverted dropout: scale by 1/(1-p)
    float scale = static_cast<float>(1.0 / (1.0 - p));
    return (input * mask_var) * scale;
}

// ============================================================================
// Group Normalization
// ============================================================================

auto group_norm(const Variable& input, int64_t num_groups,
                const std::optional<Variable>& weight,
                const std::optional<Variable>& bias,
                double eps) -> Variable {
    // The dispatch path expects a [C]-shaped weight and bias tensor in all
    // cases. Synthesize ones/zeros when the caller didn't supply them so the
    // dispatch shape contract stays consistent with nn::GroupNorm and the
    // backward grad_fn can save the (synthetic) weight tensor uniformly.
    auto in_shape = input.tensor().shape();
    if (in_shape.size() < 2) {
        throw std::runtime_error("F::group_norm: input must have at least 2 dims");
    }
    int64_t num_channels = in_shape[1];
    auto compute_dtype = input.tensor().dtype();
    auto compute_device = input.tensor().device();

    Tensor weight_tensor = weight.has_value()
        ? weight->tensor()
        : ones({num_channels}, compute_dtype, compute_device);
    Tensor bias_tensor = bias.has_value()
        ? bias->tensor()
        : zeros({num_channels}, compute_dtype, compute_device);

    std::vector<Tensor> inputs_vec = {input.tensor(), weight_tensor, bias_tensor};

    NewOpAttributes attrs;
    // Use NumGroups (matches GroupNorm layer convention in normalization.cpp).
    // Backends accept either NumGroups or Groups for compatibility, but the
    // layer-side dispatch path (which is the common case) sets NumGroups.
    attrs.set(AttrKey::NumGroups, num_groups);
    attrs.set(AttrKey::Eps, eps);

    auto results = dispatch(OpId::GroupNorm, inputs_vec, attrs);
    Tensor output_t = results[0];
    Tensor saved_mean = results.size() > 1 ? results[1] : Tensor();
    Tensor saved_rstd = results.size() > 2 ? results[2] : Tensor();

    bool affine = weight.has_value() || bias.has_value();
    bool needs_grad =
        input.requires_grad() ||
        (weight.has_value() && weight->requires_grad()) ||
        (bias.has_value() && bias->requires_grad());

    if (!needs_grad) {
        return Variable(output_t, false);
    }

    Variable output(output_t, true);

    int64_t group_size = num_channels / num_groups;
    std::vector<Tensor> tensors_to_save = {
        input.tensor(), saved_mean, saved_rstd, weight_tensor
    };

    auto grad_fn = internal::make_group_norm_backward(
        affine, eps, num_groups, num_channels, group_size,
        std::move(tensors_to_save));
    output.set_grad_fn(grad_fn);

    std::vector<std::shared_ptr<::tenzor::Function>> next_funcs;
    if (auto fn = input.grad_fn()) next_funcs.push_back(fn);
    if (weight.has_value()) {
        if (auto fn = weight->grad_fn()) next_funcs.push_back(fn);
    }
    if (bias.has_value()) {
        if (auto fn = bias->grad_fn()) next_funcs.push_back(fn);
    }
    grad_fn->set_next_functions(std::move(next_funcs));

    std::vector<Variable> input_vars{input};
    if (weight.has_value()) input_vars.push_back(*weight);
    if (bias.has_value()) input_vars.push_back(*bias);
    grad_fn->set_input_variables(std::move(input_vars));

    return output;
}

// ============================================================================
// Instance Normalization
// ============================================================================

auto instance_norm(const Variable& input,
                   [[maybe_unused]] const std::optional<Tensor>& running_mean,
                   [[maybe_unused]] const std::optional<Tensor>& running_var,
                   const std::optional<Variable>& weight,
                   const std::optional<Variable>& bias,
                   [[maybe_unused]] bool training,
                   [[maybe_unused]] double momentum,
                   double eps) -> Variable {
    auto in_shape = input.tensor().shape();
    if (in_shape.size() < 2) {
        throw std::runtime_error("F::instance_norm: input must have at least 2 dims");
    }
    int64_t num_features = in_shape[1];
    auto compute_dtype = input.tensor().dtype();
    auto compute_device = input.tensor().device();

    // Synthesize a unit weight / zero bias if caller didn't supply them so
    // the dispatch contract matches nn::InstanceNorm and the backward
    // grad_fn can save the weight tensor uniformly.
    Tensor weight_tensor = weight.has_value()
        ? weight->tensor()
        : ones({num_features}, compute_dtype, compute_device);
    Tensor bias_tensor = bias.has_value()
        ? bias->tensor()
        : zeros({num_features}, compute_dtype, compute_device);

    std::vector<Tensor> inputs_vec = {input.tensor(), weight_tensor, bias_tensor};

    NewOpAttributes attrs;
    attrs.set(AttrKey::Eps, eps);

    auto results = dispatch(OpId::InstanceNorm, inputs_vec, attrs);
    Tensor output_t = results[0];
    Tensor saved_mean = results.size() > 1 ? results[1] : Tensor();
    Tensor saved_rstd = results.size() > 2 ? results[2] : Tensor();

    bool affine = weight.has_value() || bias.has_value();
    bool needs_grad =
        input.requires_grad() ||
        (weight.has_value() && weight->requires_grad()) ||
        (bias.has_value() && bias->requires_grad());

    if (!needs_grad) {
        return Variable(output_t, false);
    }

    Variable output(output_t, true);

    std::vector<Tensor> tensors_to_save = {
        input.tensor(), saved_mean, saved_rstd, weight_tensor
    };

    auto grad_fn = internal::make_instance_norm_backward(
        affine, eps, num_features, std::move(tensors_to_save));
    output.set_grad_fn(grad_fn);

    std::vector<std::shared_ptr<::tenzor::Function>> next_funcs;
    if (auto fn = input.grad_fn()) next_funcs.push_back(fn);
    if (weight.has_value()) {
        if (auto fn = weight->grad_fn()) next_funcs.push_back(fn);
    }
    if (bias.has_value()) {
        if (auto fn = bias->grad_fn()) next_funcs.push_back(fn);
    }
    grad_fn->set_next_functions(std::move(next_funcs));

    std::vector<Variable> input_vars{input};
    if (weight.has_value()) input_vars.push_back(*weight);
    if (bias.has_value()) input_vars.push_back(*bias);
    grad_fn->set_input_variables(std::move(input_vars));

    return output;
}

// ============================================================================
// Embedding
// ============================================================================

auto embedding(const Tensor& input, const Variable& weight) -> Variable {
    std::vector<Tensor> inputs_vec = {weight.tensor(), input};
    auto result = dispatch(OpId::Embedding, inputs_vec, {});
    Tensor output_t = result[0];

    if (!weight.requires_grad()) {
        return Variable(output_t, false);
    }

    auto weight_shape = weight.tensor().shape();
    int64_t num_embeddings = weight_shape.size() >= 1 ? weight_shape[0] : 0;
    int64_t embedding_dim = weight_shape.size() >= 2 ? weight_shape[1] : 0;

    // Defaults match PyTorch F.embedding when extra kwargs aren't provided.
    auto grad_fn = internal::make_embedding_backward(
        input, num_embeddings, embedding_dim,
        /*padding_idx=*/-1,
        /*scale_grad_by_freq=*/false,
        /*sparse=*/false);

    Variable output(output_t, true);
    output.set_grad_fn(grad_fn);

    std::vector<std::shared_ptr<::tenzor::Function>> next_funcs;
    if (auto fn = weight.grad_fn()) next_funcs.push_back(fn);
    grad_fn->set_next_functions(std::move(next_funcs));
    grad_fn->set_input_variables({weight});

    return output;
}

// ============================================================================
// Interpolation
// ============================================================================

auto interpolate(const Variable& input,
                 std::pair<int64_t, int64_t> size,
                 const std::string& mode,
                 bool align_corners) -> Variable {
    std::vector<Tensor> inputs_vec = {input.tensor()};

    std::string size_str = std::to_string(size.first) + "," + std::to_string(size.second);
    NewOpAttributes attrs;
    attrs.set(AttrKey::OutputSize, size_str);
    attrs.set(AttrKey::Mode, mode);
    attrs.set(AttrKey::AlignCorners, align_corners);

    auto result = dispatch(OpId::Interpolate, inputs_vec, attrs);
    Tensor output = result[0];

    // Phase 7.1 of the test-coverage campaign: wire backward through the
    // existing `UpsampleBilinearBackward` autograd Function for the bilinear
    // case (which is also what `nn::Upsample` uses). Other modes do not
    // currently have a backward implementation; rather than silently
    // returning zero gradients (the previous behaviour) we attach the
    // bilinear-backward grad_fn only when the mode supports it and otherwise
    // explicitly produce a detached Variable.
    //
    // Adding nearest/bicubic/trilinear/area backwards is tracked as the
    // remaining followup work for Phase 7.1; the bilinear path covers the
    // overwhelming majority of users (segmentation/super-resolution/etc.)
    // and matches PyTorch's grad-flow semantics for that mode.
    if (input.requires_grad() && is_grad_enabled() && mode == "bilinear") {
        const auto in_shape = input.tensor().shape();
        if (in_shape.size() != 4) {
            throw std::runtime_error(
                "interpolate(bilinear): backward requires 4D input (N,C,H,W); "
                "got " + std::to_string(in_shape.size()) + "D");
        }
        int64_t H_in = in_shape[2];
        int64_t W_in = in_shape[3];
        auto grad_fn = std::make_shared<UpsampleBilinearBackward>(
            H_in, W_in, size.first, size.second);
        grad_fn->save_for_backward({input.tensor()});
        std::vector<std::shared_ptr<Function>> next_funcs;
        next_funcs.push_back(input.grad_fn());
        grad_fn->set_next_functions(next_funcs);
        std::vector<Variable> input_vars{input};
        grad_fn->set_input_variables(input_vars);

        Variable result_var(output, /*requires_grad=*/true);
        result_var.set_grad_fn(grad_fn);
        return result_var;
    }

    // Non-bilinear modes (nearest / bicubic / trilinear / area) — no
    // backward yet. Returning a detached Variable makes the missing-
    // gradient semantics explicit. Callers that need autograd through a
    // non-bilinear interpolate should use `nn::Upsample` (bilinear) or
    // wait for the per-mode backwards to land.
    return Variable(output, false);
}

// ============================================================================
// NLL Loss
// ============================================================================

auto nll_loss(const Variable& input, const Tensor& target,
              Reduction reduction, int64_t ignore_index) -> Variable {
    // NLL loss: -sum(input[i, target[i]]) / N
    //
    // Use Variable-level ops throughout so backward propagates through the
    // gather/neg/reduce chain. The previous implementation called the
    // raw-Tensor overloads of gather/mean/sum/reshape and wrapped the
    // result in a Variable with no grad_fn — silently zeroing input.grad()
    // on backward (raw-tensor-op breaks autograd graph pattern). The
    // bound Python helper functional_nll_loss exposes this directly.
    //
    // J.5: honour ignore_index. Targets equal to ignore_index are clamped to
    // 0 *before* gather so it never throws / reads garbage, then the
    // per-sample loss is multiplied by a (target != ignore_index) mask so
    // those samples contribute zero. For Reduction::Mean we divide by the
    // unmasked count (= mask.sum()) instead of N — matching PyTorch.
    auto shape = input.tensor().shape();
    int64_t N = shape[0];

    // Ensure target lives on the input's device for gather + comparisons.
    Tensor target_dev = (target.device() == input.tensor().device())
                            ? target
                            : target.to(input.tensor().device());

    // Build the keep/drop mask on the input's device.
    Tensor ignore_t   = tenzor::full_like(target_dev,
                                          static_cast<double>(ignore_index));
    Tensor is_ignored = tenzor::eq(target_dev, ignore_t);  // bool, shape [N]
    Tensor keep_mask  = tenzor::ne(target_dev, ignore_t);  // bool, shape [N]

    // Clamp out-of-range / sentinel targets to 0 so gather succeeds even
    // when ignore_index is negative or >= num_classes.
    Tensor zero_t = tenzor::full_like(target_dev, 0.0);
    Tensor safe_target = tenzor::where(is_ignored, zero_t, target_dev);

    // Reshape safe target to [N, 1] for column gather. The index is plain
    // index data (Int64) — keep as a Tensor; Variable gather takes
    // (Variable, dim, Tensor index).
    Tensor target_2d = tenzor::reshape(safe_target, {N, 1});

    // Variable gather → Variable reshape → Variable neg.
    Variable gathered = tenzor::gather(input, /*dim=*/1, target_2d);
    Variable loss_per_sample =
        tenzor::neg(tenzor::reshape(gathered, std::vector<int64_t>{N}));

    // Mask out ignored samples in input dtype (already on input device).
    Tensor mask_in_dtype = keep_mask.to(input.tensor().dtype());
    Variable mask_var(mask_in_dtype, /*requires_grad=*/false);
    Variable masked_loss = loss_per_sample * mask_var;

    if (reduction == Reduction::Mean) {
        // Denominator is the number of non-ignored samples.  Clamp to >= 1
        // so an all-ignored batch yields 0 rather than NaN (matches
        // PyTorch's behaviour for the common-sense edge case).  Sum
        // through Float32 because clamp_min lacks BF16 dispatch; the
        // count is a tiny scalar so no precision concern.
        const DType out_dtype = input.tensor().dtype();
        Tensor mask_f32 = mask_in_dtype.dtype() == DType::Float32
                              ? mask_in_dtype
                              : mask_in_dtype.to(DType::Float32);
        Tensor denom_f32 = tenzor::clamp_min(tenzor::sum(mask_f32), 1.0);
        Tensor denom = (denom_f32.dtype() == out_dtype)
                           ? denom_f32
                           : denom_f32.to(out_dtype);
        Variable denom_var(denom, /*requires_grad=*/false);
        Variable total = tenzor::sum(masked_loss);
        return total / denom_var;
    }
    if (reduction == Reduction::Sum) return tenzor::sum(masked_loss);
    return masked_loss;
}

// ============================================================================
// Smooth L1 Loss
// ============================================================================

auto smooth_l1_loss(const Variable& input, const Variable& target,
                    Reduction reduction, double beta) -> Variable {
    // Branchless Smooth-L1 (Huber) loss so autograd flows through.
    // The previous implementation used a raw tenzor::where on Tensors and
    // wrapped the result in a fresh Variable, which silently dropped the
    // grad_fn chain and left pred.grad() empty after backward().
    //
    //   loss = 0.5 * min(|diff|, beta)^2 / beta + max(|diff| - beta, 0)
    // When |diff| < beta : 0.5 * |diff|^2 / beta + 0
    // When |diff| ≥ beta : 0.5 * beta + |diff| - beta = |diff| - 0.5*beta
    //
    // All operations below are Variable ops with registered grad_fns, so
    // backward() propagates correctly into both input and target.
    auto diff = input - target;
    auto abs_diff = tenzor::abs(diff);
    auto clamped_abs = tenzor::clamp(abs_diff, 0.0f, static_cast<float>(beta));
    auto excess = abs_diff - clamped_abs;
    auto loss_unreduced =
        (clamped_abs * clamped_abs * 0.5f) / static_cast<float>(beta) + excess;

    if (reduction == Reduction::Mean) return tenzor::mean(loss_unreduced);
    if (reduction == Reduction::Sum) return tenzor::sum(loss_unreduced);
    return loss_unreduced;
}

// ============================================================================
// Cosine Similarity
// ============================================================================

auto cosine_similarity(const Variable& x1, const Variable& x2,
                       int64_t dim, double eps) -> Variable {
    // cos_sim = (x1 . x2) / (||x1|| * ||x2|| + eps)
    auto dot = tenzor::sum(x1 * x2, dim);
    auto norm1 = tenzor::sqrt(tenzor::sum(x1 * x1, dim));
    auto norm2 = tenzor::sqrt(tenzor::sum(x2 * x2, dim));
    auto denom = norm1 * norm2 + static_cast<float>(eps);
    return dot / denom;
}

// ============================================================================
// Attention
// ============================================================================

auto scaled_dot_product_attention(
    const Variable& query, const Variable& key,
    const Variable& value, const SDPAOptions& opts) -> Variable {

    auto d_k = static_cast<float>(query.shape().back());
    float scale = 1.0f / std::sqrt(d_k);

    // Try FlashAttention dispatch for eligible inputs:
    // 4D tensors, no explicit attn_mask, supported dtype
    bool is_4d = query.tensor().ndim() == 4;
    bool no_mask = !opts.attn_mask.has_value();
    bool supported_dtype = (query.tensor().dtype() == DType::Float32 ||
                            query.tensor().dtype() == DType::Float16 ||
                            query.tensor().dtype() == DType::BFloat16);

    // CUDA fused_attention_cuda and ROCm fused_attention_hip kernels read the
    // Causal attr but discard it — they don't apply triangular masking. So for
    // is_causal=true on those backends, fall through to the manual BMM path
    // which builds an explicit triu mask. CPU and Vulkan FlashAttention do
    // honor causal. (#46/#49)
    auto dev_t = query.tensor().device().type;
    bool device_supports_causal = (dev_t == Device::Type::CPU ||
                                   dev_t == Device::Type::Vulkan);
    bool causal_path_ok = !opts.is_causal || device_supports_causal;

    // FlashAttention dropout is currently only honored by the CPU kernel
    // (the GPU FlashAttention/FusedAttention kernels ignore DropoutP/IsTraining).
    // For dropout_p > 0 on a GPU backend, fall through to the manual BMM path
    // which applies dropout as a separate Variable-level op. (#49)
    bool device_supports_dropout = (dev_t == Device::Type::CPU);
    bool dropout_path_ok = (opts.dropout_p <= 0.0) || device_supports_dropout;

    if (is_4d && no_mask && supported_dtype && causal_path_ok && dropout_path_ok) {
        try {
            // Backend FlashAttention kernels disagree on input rank: CPU's
            // flash_attention_forward expects 4D [B, H, L, E], but CUDA's
            // fused_attention_cuda and ROCm's fused_attention_hip expect 3D
            // [B*H, L, E] (collapsed batch+heads). Normalize to 3D for GPU
            // dispatch and reshape the output back. Use the Variable-overload
            // reshape from autograd::ops so the grad chain is preserved
            // through the FlashAttentionBackward Function below.
            auto q_shape = query.shape();
            int64_t B = q_shape[0], H = q_shape[1], L = q_shape[2], E = q_shape[3];
            auto dev = query.tensor().device().type;
            bool needs_3d_collapse = (dev != Device::Type::CPU);

            Variable q_var = query;
            Variable k_var = key;
            Variable v_var = value;
            if (needs_3d_collapse) {
                int64_t Lk = k_var.shape()[2];
                int64_t Ek = k_var.shape()[3];
                int64_t Lv = v_var.shape()[2];
                int64_t Ev = v_var.shape()[3];
                q_var = ::tenzor::reshape(q_var, std::vector<int64_t>{B * H, L, E});
                k_var = ::tenzor::reshape(k_var, std::vector<int64_t>{B * H, Lk, Ek});
                v_var = ::tenzor::reshape(v_var, std::vector<int64_t>{B * H, Lv, Ev});
            }

            // Route through the autograd FlashAttentionBackward Function so
            // the resulting Variable carries a grad_fn. The previous direct
            // dispatch wrapped output as Variable(t, requires_grad=true) with
            // no grad_fn — silently zeroing Q/K/V gradients (audit C1).
            Variable out_var = ::tenzor::flash_attention(
                q_var, k_var, v_var,
                static_cast<float>(scale),
                /*causal=*/opts.is_causal,
                /*dropout_p=*/static_cast<float>(opts.dropout_p),
                /*is_training=*/opts.dropout_p > 0.0);

            if (needs_3d_collapse) {
                out_var = ::tenzor::reshape(out_var, std::vector<int64_t>{B, H, L, E});
            }
            return out_var;
        } catch (const std::exception&) {
            // Fall through to manual path if FlashAttention kernel unavailable
        }
    }

    // Manual attention path: Q @ K^T / sqrt(d_k)
    // Use Variable-level operators throughout so backward propagates through
    // transpose, scale, and mask additions. Previously these wrapped raw
    // tensor results which silently severed the autograd chain.
    auto kt = ::tenzor::transpose(key, -2, -1);
    auto scores = tenzor::matmul(query, kt);
    Variable scale_var(::tenzor::full({1}, scale,
                                       scores.tensor().dtype(), scores.tensor().device()), false);
    auto scaled = scores * scale_var;

    // Causal mask: use triu to build mask on-device (no CPU fallback).
    // Always use Float32 for the mask to avoid overflow in half-precision
    // types (Float16 max ~65504, -1e9 would overflow).
    if (opts.is_causal) {
        auto L = query.shape()[query.shape().size() - 2];
        auto S = key.shape()[key.shape().size() - 2];
        auto mask = tenzor::triu(
            tenzor::ones({L, S}, DType::Float32, query.tensor().device()),
            /*diagonal=*/1) * -1e9f;
        if (query.tensor().dtype() != DType::Float32) {
            mask = mask.to(query.tensor().dtype());
        }
        Variable mask_var(mask, false);
        scaled = scaled + mask_var;
    }

    // Optional attention mask
    if (opts.attn_mask.has_value()) {
        scaled = scaled + *opts.attn_mask;
    }

    // Softmax along last dimension
    auto attn = tenzor::softmax(scaled, -1);

    // Dropout (if requested). Callers should set dropout_p = 0 during
    // inference — this is a free function without training-mode state.
    // Use Variable-level operators here so backward propagates through the
    // mask multiplication (the previous `Variable(attn.tensor()*..., rg)`
    // pattern would silently sever the autograd chain — see the
    // raw-tensor-op-breaks-autograd-graph memory).
    if (opts.dropout_p > 0.0) {
        auto drop_mask = tenzor::rand(
            std::vector<int64_t>(attn.tensor().shape().begin(), attn.tensor().shape().end()),
            attn.tensor().dtype(), attn.tensor().device());
        auto threshold = tenzor::full({1}, static_cast<float>(opts.dropout_p),
                                      attn.tensor().dtype(), attn.tensor().device());
        Variable keep(tenzor::gt(drop_mask, threshold).to(attn.tensor().dtype()), false);
        Variable scale_var(tenzor::full({1}, static_cast<float>(1.0 / (1.0 - opts.dropout_p)),
                                        attn.tensor().dtype(), attn.tensor().device()), false);
        attn = attn * keep * scale_var;
    }

    // attn @ V
    return tenzor::matmul(attn, value);
}

// ============================================================================
// Normalize
// ============================================================================

auto normalize(const Variable& input, double p, int64_t dim,
               double eps) -> Variable {
    // Compute L_p norm along dim using Variable-level ops so backward
    // flows through. The previous implementation called tenzor::norm on
    // a raw Tensor and wrapped the division result in a Variable with no
    // grad_fn — silently zeroing input.grad() on backward.
    //
    // ||x||_p = ( sum(|x|^p, dim, keepdim=true) )^(1/p)
    // Add eps before division (instead of clamp_min) since clamp_min
    // doesn't have a Variable overload yet; this matches PyTorch's
    // alternate "softer" normalize formula and is identical to clamp_min
    // for nonzero norms (the only relevant case at runtime).
    Variable abs_x = tenzor::abs(input);
    Variable powered = tenzor::pow(abs_x, static_cast<double>(p));
    Variable summed = tenzor::sum(powered, dim, /*keepdim=*/true);
    Variable norm_v = tenzor::pow(summed, 1.0 / static_cast<double>(p));
    return input / (norm_v + static_cast<float>(eps));
}

// ============================================================================
// Pad (supports constant / reflect / replicate / circular modes)
// ============================================================================

// Helper: build index tensor that maps padded output positions to input positions
static auto build_pad_indices(int64_t dim_size, int64_t pad_before, int64_t pad_after,
                               const std::string& mode, Device device) -> Tensor {
    int64_t out_size = dim_size + pad_before + pad_after;
    std::vector<int64_t> indices(out_size);

    for (int64_t i = 0; i < out_size; ++i) {
        int64_t src = i - pad_before;  // position relative to input

        if (mode == "reflect") {
            // Reflect: bounce off boundaries (excluding edge element)
            // For input [0..N-1], reflect at 0 and N-1
            if (dim_size == 1) {
                src = 0;
            } else {
                // Map into [0, 2*(N-1)) cycle, then fold back
                if (src < 0) src = -src;
                int64_t period = 2 * (dim_size - 1);
                if (period > 0) {
                    src = src % period;
                    if (src >= dim_size) src = period - src;
                }
            }
        } else if (mode == "replicate") {
            // Replicate: clamp to valid range
            if (src < 0) src = 0;
            else if (src >= dim_size) src = dim_size - 1;
        } else if (mode == "circular") {
            // Circular: wrap around using modulo
            src = ((src % dim_size) + dim_size) % dim_size;
        }

        indices[i] = src;
    }

    Tensor idx_tensor({out_size}, DType::Int64, Device::cpu());
    std::memcpy(idx_tensor.data<int64_t>(), indices.data(), out_size * sizeof(int64_t));
    if (device != Device::cpu()) {
        idx_tensor = idx_tensor.to(device);
    }
    return idx_tensor;
}

auto pad(const Variable& input, const std::vector<int64_t>& pad_sizes,
         const std::string& mode, double value) -> Variable {
    if (pad_sizes.size() % 2 != 0) {
        throw std::invalid_argument("F::pad: padding must have even number of elements");
    }
    if (mode != "constant" && mode != "reflect" && mode != "replicate" && mode != "circular") {
        throw std::invalid_argument(
            "F::pad: unsupported mode '" + mode + "', expected constant/reflect/replicate/circular");
    }

    auto shape = input.shape();
    auto ndim = static_cast<int64_t>(shape.size());
    auto n_pad_dims = static_cast<int64_t>(pad_sizes.size()) / 2;

    if (n_pad_dims > ndim) {
        throw std::invalid_argument("F::pad: too many padding dimensions");
    }

    // Validate reflect mode constraints: pad size must be < dim size
    if (mode == "reflect") {
        for (int64_t i = 0; i < n_pad_dims; ++i) {
            auto dim_idx = ndim - 1 - i;
            auto dim_size = shape[dim_idx];
            if (pad_sizes[2 * i] >= dim_size || pad_sizes[2 * i + 1] >= dim_size) {
                throw std::invalid_argument(
                    "F::pad reflect: padding size must be less than dimension size");
            }
        }
    }

    auto& inp = input.tensor();

    if (mode == "constant") {
        // Build constant padding as a sequence of cats: pre-pad zeros on
        // each axis, then the input, then post-pad zeros. Each cat is an
        // autograd-aware op, so gradients flow back through the input
        // Variable. The earlier memcpy-through-narrow path dropped grad_fn
        // and silently zeroed any tensor that consumed this output during
        // backward — same pattern as our known "raw-tensor-op breaks
        // autograd graph" bugs.
        auto dtype = inp.dtype();
        auto device = inp.device();

        Variable result = input;
        for (int64_t i = 0; i < n_pad_dims; ++i) {
            auto dim_idx = ndim - 1 - i;
            auto pad_before = pad_sizes[2 * i];
            auto pad_after  = pad_sizes[2 * i + 1];
            if (pad_before == 0 && pad_after == 0) continue;

            auto cur_shape = result.tensor().shape();
            std::vector<Variable> parts;
            if (pad_before > 0) {
                auto pad_shape = std::vector<int64_t>(cur_shape.begin(), cur_shape.end());
                pad_shape[dim_idx] = pad_before;
                parts.push_back(Variable(
                    tenzor::full(std::move(pad_shape),
                                 static_cast<float>(value), dtype, device),
                    false));
            }
            parts.push_back(result);
            if (pad_after > 0) {
                auto pad_shape = std::vector<int64_t>(cur_shape.begin(), cur_shape.end());
                pad_shape[dim_idx] = pad_after;
                parts.push_back(Variable(
                    tenzor::full(std::move(pad_shape),
                                 static_cast<float>(value), dtype, device),
                    false));
            }
            result = cat(parts, dim_idx);
        }
        return result;
    }

    // reflect / replicate / circular: use Variable-level index_select per
    // padded dimension so backward flows back to input. The previous
    // implementation called the raw-Tensor index_select and wrapped the
    // result in a Variable with no grad_fn — silently zeroing input.grad()
    // (raw-tensor-op breaks autograd graph pattern).
    Variable result = input;
    for (int64_t i = 0; i < n_pad_dims; ++i) {
        auto dim_idx = ndim - 1 - i;
        auto pad_before = pad_sizes[2 * i];
        auto pad_after = pad_sizes[2 * i + 1];

        if (pad_before == 0 && pad_after == 0) continue;

        auto current_dim_size = result.tensor().shape()[dim_idx];
        auto idx = build_pad_indices(current_dim_size, pad_before, pad_after,
                                     mode, result.tensor().device());
        result = tenzor::index_select(result, dim_idx, idx);
    }

    return result;
}

// ============================================================================
// Phase 9: Lp Pooling (compositions)
// ============================================================================

auto lp_pool1d(const Variable& input, double norm_type, int64_t kernel_size,
               int64_t stride, [[maybe_unused]] bool ceil_mode) -> Variable {
    if (input.shape().size() != 3) {
        throw std::invalid_argument(
            "F::lp_pool1d expects 3D input [N, C, L]");
    }

    if (stride <= 0) stride = kernel_size;

    // |input|^p — Variable-aware so autograd chain is preserved.
    auto abs_pow = tenzor::pow(tenzor::abs(input), static_cast<float>(norm_type));

    // B4b: Delegate to nn::AvgPool1d so the AvgPool1dBackward grad_fn is
    // wired. The previous direct-dispatch path built a leaf Variable that
    // silently dropped autograd for the entire lp_pool1d chain (mirror of
    // the J7 bug in functional::avg_pool2d, fixed the same way).
    ::tenzor::nn::AvgPool1d pool(kernel_size, stride, /*padding=*/0);
    auto pooled = pool.forward(abs_pow);

    // (mean)^(1/p)
    return tenzor::pow(pooled, static_cast<float>(1.0 / norm_type));
}

auto lp_pool2d(const Variable& input, double norm_type,
               std::pair<int64_t, int64_t> kernel_size,
               std::pair<int64_t, int64_t> stride,
               [[maybe_unused]] bool ceil_mode) -> Variable {
    if (input.shape().size() != 4) {
        throw std::invalid_argument(
            "F::lp_pool2d expects 4D input [N, C, H, W]");
    }

    if (stride.first <= 0) stride.first = kernel_size.first;
    if (stride.second <= 0) stride.second = kernel_size.second;

    // |input|^p — Variable-aware so autograd chain is preserved end-to-end.
    auto abs_pow = tenzor::pow(tenzor::abs(input), static_cast<float>(norm_type));

    // avg_pool2d on the powered values; pass the Variable directly so its
    // grad_fn is retained. The previous code wrapped abs_pow.tensor() in a
    // fresh Variable(t, false) which discarded the graph and zeroed grads.
    auto pooled = avg_pool2d(abs_pow, kernel_size, stride, {0, 0});

    // (mean)^(1/p) — pooled is already a Variable with the correct grad_fn.
    return tenzor::pow(pooled, static_cast<float>(1.0 / norm_type));
}

// ============================================================================
// Phase 9: Local Response Normalization (composition)
// ============================================================================

auto local_response_norm(const Variable& input, int64_t size,
                         double alpha, double beta,
                         double k) -> Variable {
    auto shape_span = input.shape();
    std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
    if (shape_vec.size() < 3) {
        throw std::invalid_argument(
            "F::local_response_norm expects at least 3D input [N, C, ...]");
    }

    // Compute squared values
    auto squared = input * input;

    // Sum across neighboring channels using a sliding window approach.
    // Reshape so channel dimension becomes a spatial dimension, then use
    // avg_pool1d with appropriate kernel, and scale to get sum.
    int64_t N = shape_vec[0];
    int64_t C = shape_vec[1];
    int64_t spatial = 1;
    for (size_t i = 2; i < shape_vec.size(); ++i) spatial *= shape_vec[i];

    // Reshape squared to [N, C, S] where S = product of spatial dims
    std::vector<int64_t> flat_shape = {N, C, spatial};
    auto sq_flat = tenzor::reshape(squared, flat_shape);

    // Transpose to [N, S, C] then reshape to [N*S, 1, C]
    auto sq_transposed = tenzor::transpose(sq_flat, 1, 2);  // [N, S, C]
    std::vector<int64_t> pool_shape = {N * spatial, int64_t{1}, C};
    auto sq_for_pool = tenzor::reshape(sq_transposed, pool_shape);

    // Pad the channel dimension with zeros on both sides
    int64_t pad_size = (size - 1) / 2;

    // Use avg_pool1d with kernel=size, stride=1, padding=pad_size to compute channel sum
    NewOpAttributes pool_attrs;
    pool_attrs.set(AttrKey::KernelSize, size);
    pool_attrs.set(AttrKey::Stride, int64_t{1});
    pool_attrs.set(AttrKey::Padding, pad_size);

    std::vector<Tensor> pool_inputs = {sq_for_pool.tensor()};
    auto channel_avg = dispatch_to_device(OpId::AvgPool1dForward,
        input.tensor().device().type, pool_inputs, pool_attrs);

    // avg_pool1d gives sum/size, we want alpha/size * sum = alpha * (sum/size)
    auto channel_sum_scaled = Variable(channel_avg[0], input.requires_grad());

    // Reshape back: [N*S, 1, C] -> [N, S, C] -> [N, C, S] -> original shape
    std::vector<int64_t> nsc_shape = {N, spatial, C};
    auto reshaped = tenzor::reshape(channel_sum_scaled, nsc_shape);
    auto transposed_back = tenzor::transpose(reshaped, 1, 2);  // [N, C, S]
    auto final_sum = tenzor::reshape(transposed_back, shape_vec);

    // div_factor = (k + alpha * avg_across_channels)^beta
    // Since avg_pool gave us sum/size and we want alpha/size * sum = alpha * (sum/size)
    auto scaled_sum = final_sum * static_cast<float>(alpha) + static_cast<float>(k);
    auto div_factor = tenzor::pow(scaled_sum, static_cast<float>(beta));

    return input / div_factor;
}

// ============================================================================
// Phase 9: Fractional Max Pool (dispatch to backend)
// ============================================================================

auto fractional_max_pool2d(const Variable& input,
                           std::pair<int64_t, int64_t> kernel_size,
                           std::pair<int64_t, int64_t> output_size,
                           const std::optional<Tensor>& random_samples)
    -> std::pair<Variable, Tensor> {
    if (input.shape().size() != 4) {
        throw std::invalid_argument(
            "F::fractional_max_pool2d expects 4D input [N, C, H, W]");
    }

    NewOpAttributes attrs;
    attrs.set(AttrKey::KernelSize, kernel_size.first);
    attrs.set(AttrKey::OutputSizeH, output_size.first);
    attrs.set(AttrKey::OutputSizeW, output_size.second);

    std::vector<Tensor> inputs_vec = {input.tensor()};
    if (random_samples.has_value()) {
        inputs_vec.push_back(random_samples.value());
    }

    auto result = dispatch_to_device(OpId::FractionalMaxPool2dForward,
        input.tensor().device().type, inputs_vec, attrs);

    Variable out(result[0], input.requires_grad());
    Tensor indices = result[1];
    if (input.requires_grad() && ::tenzor::is_grad_enabled()) {
        std::vector<int64_t> in_shape(input.shape().begin(), input.shape().end());
        auto backward_fn = std::make_shared<
            IndexedPoolBackward<OpId::FractionalMaxPool2dBackward>>(
                std::move(in_shape), indices);
        wire_pool_grad_fn(out, input, backward_fn);
    }
    return {std::move(out), indices};
}

auto fractional_max_pool3d(const Variable& input,
                           std::tuple<int64_t, int64_t, int64_t> kernel_size,
                           std::tuple<int64_t, int64_t, int64_t> output_size,
                           const std::optional<Tensor>& random_samples)
    -> std::pair<Variable, Tensor> {
    if (input.shape().size() != 5) {
        throw std::invalid_argument(
            "F::fractional_max_pool3d expects 5D input [N, C, D, H, W]");
    }

    NewOpAttributes attrs;
    attrs.set(AttrKey::KernelSize, std::get<0>(kernel_size));
    attrs.set(AttrKey::OutputSizeD, std::get<0>(output_size));
    attrs.set(AttrKey::OutputSizeH, std::get<1>(output_size));
    attrs.set(AttrKey::OutputSizeW, std::get<2>(output_size));

    std::vector<Tensor> inputs_vec = {input.tensor()};
    if (random_samples.has_value()) {
        inputs_vec.push_back(random_samples.value());
    }

    auto result = dispatch_to_device(OpId::FractionalMaxPool3dForward,
        input.tensor().device().type, inputs_vec, attrs);

    Variable out(result[0], input.requires_grad());
    Tensor indices = result[1];
    if (input.requires_grad() && ::tenzor::is_grad_enabled()) {
        std::vector<int64_t> in_shape(input.shape().begin(), input.shape().end());
        auto backward_fn = std::make_shared<
            IndexedPoolBackward<OpId::FractionalMaxPool3dBackward>>(
                std::move(in_shape), indices);
        wire_pool_grad_fn(out, input, backward_fn);
    }
    return {std::move(out), indices};
}

// ============================================================================
// Phase 9: Max Unpool (dispatch to backend)
// ============================================================================

auto max_unpool2d(const Variable& input, const Tensor& indices,
                  std::pair<int64_t, int64_t> kernel_size,
                  std::pair<int64_t, int64_t> stride,
                  std::pair<int64_t, int64_t> padding,
                  std::optional<std::pair<int64_t, int64_t>> output_size) -> Variable {
    if (input.shape().size() != 4) {
        throw std::invalid_argument(
            "F::max_unpool2d expects 4D input [N, C, H_pool, W_pool]");
    }

    // Default stride = kernel_size
    if (stride.first < 0) stride.first = kernel_size.first;
    if (stride.second < 0) stride.second = kernel_size.second;

    // Compute output size if not specified
    int64_t out_h, out_w;
    if (output_size.has_value()) {
        out_h = output_size->first;
        out_w = output_size->second;
    } else {
        auto shape = input.shape();
        out_h = (shape[2] - 1) * stride.first - 2 * padding.first + kernel_size.first;
        out_w = (shape[3] - 1) * stride.second - 2 * padding.second + kernel_size.second;
    }

    NewOpAttributes attrs;
    attrs.set(AttrKey::KernelSize, kernel_size.first);
    attrs.set(AttrKey::Stride, stride.first);
    attrs.set(AttrKey::Padding, padding.first);
    attrs.set(AttrKey::OutputSizeH, out_h);
    attrs.set(AttrKey::OutputSizeW, out_w);

    std::vector<Tensor> inputs_vec = {input.tensor(), indices};
    auto result = dispatch_to_device(OpId::MaxUnpool2dForward,
        input.tensor().device().type, inputs_vec, attrs);

    Variable out(result[0], input.requires_grad());
    if (input.requires_grad() && ::tenzor::is_grad_enabled()) {
        std::vector<int64_t> in_shape(input.shape().begin(), input.shape().end());
        auto backward_fn = std::make_shared<
            IndexedPoolBackward<OpId::MaxUnpool2dBackward>>(
                std::move(in_shape), indices);
        wire_pool_grad_fn(out, input, backward_fn);
    }
    return out;
}



// =====================================================================
// Phase A.1 — Functional max_unpool1d.
// =====================================================================

auto max_unpool1d(const Variable& input, const Tensor& indices,
                  int64_t kernel_size,
                  int64_t stride,
                  int64_t padding,
                  std::optional<int64_t> output_size) -> Variable {
    if (input.shape().size() != 3) {
        throw std::invalid_argument(
            "F::max_unpool1d expects 3D input [N, C, L_pool]");
    }

    if (stride < 0) stride = kernel_size;

    int64_t out_l;
    if (output_size.has_value()) {
        out_l = *output_size;
    } else {
        auto shape = input.shape();
        out_l = (shape[2] - 1) * stride - 2 * padding + kernel_size;
    }

    NewOpAttributes attrs;
    attrs.set(AttrKey::KernelSize, kernel_size);
    attrs.set(AttrKey::Stride, stride);
    attrs.set(AttrKey::Padding, padding);
    // Reuse OutputSizeW for the 1-D output length (avoids a new AttrKey).
    attrs.set(AttrKey::OutputSizeW, out_l);

    std::vector<Tensor> inputs_vec = {input.tensor(), indices};
    auto result = dispatch_to_device(OpId::MaxUnpool1dForward,
        input.tensor().device().type, inputs_vec, attrs);

    Variable out(result[0], input.requires_grad());
    if (input.requires_grad() && ::tenzor::is_grad_enabled()) {
        std::vector<int64_t> in_shape(input.shape().begin(), input.shape().end());
        auto backward_fn = std::make_shared<
            IndexedPoolBackward<OpId::MaxUnpool1dBackward>>(
                std::move(in_shape), indices);
        wire_pool_grad_fn(out, input, backward_fn);
    }
    return out;
}

auto max_unpool3d(const Variable& input, const Tensor& indices,
                  std::tuple<int64_t, int64_t, int64_t> kernel_size,
                  std::tuple<int64_t, int64_t, int64_t> stride,
                  std::tuple<int64_t, int64_t, int64_t> padding,
                  std::optional<std::tuple<int64_t, int64_t, int64_t>> output_size) -> Variable {
    if (input.shape().size() != 5) {
        throw std::invalid_argument(
            "F::max_unpool3d expects 5D input [N, C, D_pool, H_pool, W_pool]");
    }

    // Default stride = kernel_size
    auto [sk_d, sk_h, sk_w] = stride;
    auto [kk_d, kk_h, kk_w] = kernel_size;
    auto [pk_d, pk_h, pk_w] = padding;
    if (sk_d < 0) sk_d = kk_d;
    if (sk_h < 0) sk_h = kk_h;
    if (sk_w < 0) sk_w = kk_w;

    int64_t out_d, out_h, out_w;
    if (output_size.has_value()) {
        out_d = std::get<0>(output_size.value());
        out_h = std::get<1>(output_size.value());
        out_w = std::get<2>(output_size.value());
    } else {
        auto shape = input.shape();
        out_d = (shape[2] - 1) * sk_d - 2 * pk_d + kk_d;
        out_h = (shape[3] - 1) * sk_h - 2 * pk_h + kk_h;
        out_w = (shape[4] - 1) * sk_w - 2 * pk_w + kk_w;
    }

    NewOpAttributes attrs;
    attrs.set(AttrKey::KernelSize, kk_d);
    attrs.set(AttrKey::Stride, sk_d);
    attrs.set(AttrKey::Padding, pk_d);
    attrs.set(AttrKey::OutputSizeD, out_d);
    attrs.set(AttrKey::OutputSizeH, out_h);
    attrs.set(AttrKey::OutputSizeW, out_w);

    std::vector<Tensor> inputs_vec = {input.tensor(), indices};
    auto result = dispatch_to_device(OpId::MaxUnpool3dForward,
        input.tensor().device().type, inputs_vec, attrs);

    Variable out(result[0], input.requires_grad());
    if (input.requires_grad() && ::tenzor::is_grad_enabled()) {
        std::vector<int64_t> in_shape(input.shape().begin(), input.shape().end());
        auto backward_fn = std::make_shared<
            IndexedPoolBackward<OpId::MaxUnpool3dBackward>>(
                std::move(in_shape), indices);
        wire_pool_grad_fn(out, input, backward_fn);
    }
    return out;
}

// ============================================================================
// Multi-Head Attention
// ============================================================================

auto multi_head_attention_forward(
    const Tensor& query, const Tensor& key, const Tensor& value,
    int64_t num_heads,
    const Tensor& in_proj_weight, const Tensor& in_proj_bias,
    const Tensor& out_proj_weight, const Tensor& out_proj_bias,
    std::optional<Tensor> attn_mask,
    double dropout_p,
    bool training,
    bool need_weights
) -> std::pair<Tensor, Tensor> {

    // Validate input shapes
    auto q_shape = query.shape();
    if (q_shape.size() != 3) {
        throw std::invalid_argument(
            "F::multi_head_attention_forward: query must be 3D [batch, seq_q, embed_dim], got " +
            std::to_string(q_shape.size()) + "D");
    }
    auto k_shape = key.shape();
    auto v_shape = value.shape();
    if (k_shape.size() != 3) {
        throw std::invalid_argument(
            "F::multi_head_attention_forward: key must be 3D [batch, seq_k, embed_dim], got " +
            std::to_string(k_shape.size()) + "D");
    }
    if (v_shape.size() != 3) {
        throw std::invalid_argument(
            "F::multi_head_attention_forward: value must be 3D [batch, seq_k, embed_dim], got " +
            std::to_string(v_shape.size()) + "D");
    }

    int64_t batch_size = q_shape[0];
    int64_t seq_len_q = q_shape[1];
    int64_t embed_dim = q_shape[2];
    int64_t seq_len_k = k_shape[1];

    if (embed_dim % num_heads != 0) {
        throw std::invalid_argument(
            "F::multi_head_attention_forward: embed_dim (" + std::to_string(embed_dim) +
            ") must be divisible by num_heads (" + std::to_string(num_heads) + ")");
    }
    int64_t head_dim = embed_dim / num_heads;

    // Validate projection weight shapes
    auto w_shape = in_proj_weight.shape();
    if (w_shape.size() != 2 || w_shape[0] != 3 * embed_dim || w_shape[1] != embed_dim) {
        throw std::invalid_argument(
            "F::multi_head_attention_forward: in_proj_weight must be [3*embed_dim, embed_dim], got [" +
            std::to_string(w_shape[0]) + ", " + std::to_string(w_shape[1]) + "]");
    }
    auto b_shape = in_proj_bias.shape();
    if (b_shape.size() != 1 || b_shape[0] != 3 * embed_dim) {
        throw std::invalid_argument(
            "F::multi_head_attention_forward: in_proj_bias must be [3*embed_dim]");
    }

    // 1. Input projection: project Q, K, V using combined weight
    //    in_proj_weight is [3*E, E], in_proj_bias is [3*E]
    //    We split the weight into three [E, E] chunks for Q, K, V

    // Slice projection weight and bias for Q, K, V
    auto w_q = tenzor::slice(in_proj_weight, 0, 0, embed_dim);           // [E, E]
    auto w_k = tenzor::slice(in_proj_weight, 0, embed_dim, 2 * embed_dim);   // [E, E]
    auto w_v = tenzor::slice(in_proj_weight, 0, 2 * embed_dim, 3 * embed_dim); // [E, E]

    auto b_q = tenzor::slice(in_proj_bias, 0, 0, embed_dim);             // [E]
    auto b_k = tenzor::slice(in_proj_bias, 0, embed_dim, 2 * embed_dim);     // [E]
    auto b_v = tenzor::slice(in_proj_bias, 0, 2 * embed_dim, 3 * embed_dim);   // [E]

    // Project: Q = query @ w_q^T + b_q, etc.
    // query is [B, Sq, E], w_q^T is [E, E] => result is [B, Sq, E]
    auto w_q_t = tenzor::transpose(w_q, 0, 1);
    auto w_k_t = tenzor::transpose(w_k, 0, 1);
    auto w_v_t = tenzor::transpose(w_v, 0, 1);

    auto Q = tenzor::matmul(query, w_q_t) + b_q;   // [B, Sq, E]
    auto K = tenzor::matmul(key, w_k_t) + b_k;      // [B, Sk, E]
    auto V = tenzor::matmul(value, w_v_t) + b_v;     // [B, Sk, E]

    // 2. Reshape to (batch, num_heads, seq_len, head_dim)
    Q = tenzor::permute(tenzor::reshape(Q, {batch_size, seq_len_q, num_heads, head_dim}), {0, 2, 1, 3});
    K = tenzor::permute(tenzor::reshape(K, {batch_size, seq_len_k, num_heads, head_dim}), {0, 2, 1, 3});
    V = tenzor::permute(tenzor::reshape(V, {batch_size, seq_len_k, num_heads, head_dim}), {0, 2, 1, 3});

    // 3. Scaled dot-product attention via existing functional
    //    scaled_dot_product_attention expects Variables with shape [B, H, L, E]
    Variable q_var(Q, false);
    Variable k_var(K, false);
    Variable v_var(V, false);

    SDPAOptions opts;
    if (attn_mask.has_value()) {
        opts.attn_mask = Variable(attn_mask.value(), false);
    }
    opts.dropout_p = (training ? dropout_p : 0.0);

    auto attended = scaled_dot_product_attention(q_var, k_var, v_var, opts);
    // attended shape: [B, H, Sq, head_dim]

    // Compute attention weights if requested (before merge)
    Tensor attn_weights_out;
    if (need_weights) {
        // Recompute weights: softmax(Q @ K^T / sqrt(d_k))
        auto d_k_f = static_cast<float>(head_dim);
        float scale = 1.0f / std::sqrt(d_k_f);

        auto kt = tenzor::transpose(K, -2, -1);  // [B, H, head_dim, Sk]
        auto scores = tenzor::matmul(Q, kt);      // [B, H, Sq, Sk]
        scores = scores * scale;

        if (attn_mask.has_value()) {
            scores = scores + attn_mask.value();
        }

        attn_weights_out = tenzor::softmax(Variable(scores, false), -1).tensor();
        // [B, H, Sq, Sk]
    }

    // 4. Reshape back to (batch, seq_len, embed_dim)
    //    attended: [B, H, Sq, head_dim] -> permute -> [B, Sq, H, head_dim] -> reshape -> [B, Sq, E]
    auto merged = tenzor::reshape(
        tenzor::permute(attended.tensor(), {0, 2, 1, 3}),
        {batch_size, seq_len_q, embed_dim});

    // 5. Output projection: output = merged @ out_proj_weight^T + out_proj_bias
    auto out_w_t = tenzor::transpose(out_proj_weight, 0, 1);
    auto output = tenzor::matmul(merged, out_w_t) + out_proj_bias;

    // Return (output, attention_weights)
    if (!need_weights) {
        // Return empty tensor for weights
        attn_weights_out = Tensor({0}, query.dtype(), query.device());
    }

    return {output, attn_weights_out};
}

} // namespace tenzor::nn::functional
