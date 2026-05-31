#include "tenzor/nn/layers/pooling.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/backend/dispatch.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/jit/tracer.hpp"
#include "tenzor/utils/autograd_wrap.hpp"
#include <cmath>
#include <stdexcept>
#include <limits>
#include <vector>

namespace tenzor::nn {

// =======================
// MaxPool2d Implementation
// =======================

// MaxPool2d autograd function
class MaxPool2dBackward : public Function {
public:
    MaxPool2dBackward(int64_t kernel_size, int64_t stride, int64_t padding,
                     std::vector<Tensor> tensors_to_save)
        : kernel_size_(kernel_size), stride_(stride), padding_(padding) {
        save_for_backward(std::move(tensors_to_save));
    }

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("MaxPool2dBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        // grad_outputs[0]: gradient w.r.t output [N, C, H_out, W_out]
        // saved_tensors_[0]: input [N, C, H_in, W_in]
        // saved_tensors_[1]: indices [N, C, H_out, W_out] (2D-local, format
        //                    used by the kernel registry).
        // saved_tensors_[2]: output [N, C, H_out, W_out] (for cuDNN backward)
        const auto& grad_output = grad_outputs[0];
        const auto& input       = saved_tensors_[0];
        const auto& indices     = saved_tensors_[1];
        const auto& output      = saved_tensors_[2];

        // Always route through the OpId dispatch table — the registry's
        // backward kernel knows the index format produced by its own
        // forward kernel (2D-local on CPU). The inline nn-layer scatter
        // path that used to live here assumed 4D-flat indices and broke
        // once forward was unified through the registry.
        std::vector<Tensor> inputs = {grad_output, indices, input, output};
        OpAttributes bwd_attrs;
        bwd_attrs.set(AttrKey::KernelSize, kernel_size_);
        bwd_attrs.set(AttrKey::Stride,     stride_);
        bwd_attrs.set(AttrKey::Padding,    padding_);

        auto input_shape = input.shape();
        bwd_attrs.set(AttrKey::InputShape,
            std::to_string(input_shape[0]) + "," +
            std::to_string(input_shape[1]) + "," +
            std::to_string(input_shape[2]) + "," +
            std::to_string(input_shape[3]));

        auto result = dispatch_to_device(OpId::MaxPool2dBackward,
            grad_output.device().type, inputs, bwd_attrs);
        return {result[0]};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        // Pooling backward uses index-based scatter -- delegate to tensor backward and wrap results
        std::vector<Tensor> tensor_grads;
        for (auto& v : grad_outputs) tensor_grads.push_back(v.tensor());
        auto results = backward(std::move(tensor_grads));
        std::vector<Variable> var_results;
        for (auto& t : results) var_results.emplace_back(t, false);
        return var_results;
    }

    // P4.2d: the passthrough above returns Variables without grad_fn,
    // so higher-order gradients through this op are structurally zero.
    // Declare that honestly so the engine's disconnection counter
    // reports accurately in Warn mode and throws in Error mode.
    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return true; }

private:
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
};

MaxPool2d::MaxPool2d(int64_t kernel_size, int64_t stride, int64_t padding,
                     bool ceil_mode, bool return_indices)
    : MaxPool2d(std::array<int64_t, 2>{kernel_size, kernel_size},
                std::array<int64_t, 2>{stride, stride},
                std::array<int64_t, 2>{padding, padding},
                ceil_mode, return_indices) {
}

MaxPool2d::MaxPool2d(std::array<int64_t, 2> kernel_size,
                     std::array<int64_t, 2> stride,
                     std::array<int64_t, 2> padding,
                     bool ceil_mode, bool return_indices)
    : kernel_size_h_(kernel_size[0]), kernel_size_w_(kernel_size[1]),
      stride_h_(stride[0]  < 0 ? kernel_size[0] : stride[0]),
      stride_w_(stride[1]  < 0 ? kernel_size[1] : stride[1]),
      padding_h_(padding[0]), padding_w_(padding[1]),
      kernel_size_(kernel_size[0]),   // legacy scalar = H-axis
      stride_(stride[0] < 0 ? kernel_size[0] : stride[0]),
      padding_(padding[0]),
      ceil_mode_(ceil_mode), return_indices_(return_indices) {
    for (int i = 0; i < 2; ++i) {
        if (kernel_size[i] <= 0) {
            throw std::runtime_error("MaxPool2d: kernel_size must be positive (axis " +
                std::to_string(i) + "), got " + std::to_string(kernel_size[i]));
        }
        if (padding[i] < 0) {
            throw std::runtime_error("MaxPool2d: padding must be non-negative (axis " +
                std::to_string(i) + "), got " + std::to_string(padding[i]));
        }
    }
}

auto MaxPool2d::forward_impl(const Variable& input) -> Variable {
    // Input shape: [N, C, H_in, W_in]
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("MaxPool2d expects 4D input [batch, channels, height, width]");
    }

    // Contiguous on entry — pool kernels expect NCHW contiguous layout.
    Variable contig_input = input;
    if (!input.tensor().is_contiguous()) {
        tenzor::utils::wrap_preserving_grad(contig_input, input.tensor().contiguous());
    }

    // Always route through OpId dispatch — the kernel registry handles every
    // backend (CPU, CUDA, ROCm, Vulkan, OneAPI, MPS) and reads per-axis
    // attrs with scalar fallback. This eliminates inline-CPU/inline-GPU
    // duplication in the nn layer.
    std::vector<Tensor> inputs = {contig_input.tensor()};
    OpAttributes fwd_attrs;
    // Set per-axis (registry reads per-axis first, scalar as fallback).
    fwd_attrs.set(AttrKey::KernelSizeH, kernel_size_h_);
    fwd_attrs.set(AttrKey::KernelSizeW, kernel_size_w_);
    fwd_attrs.set(AttrKey::StrideH, stride_h_);
    fwd_attrs.set(AttrKey::StrideW, stride_w_);
    fwd_attrs.set(AttrKey::PaddingH, padding_h_);
    fwd_attrs.set(AttrKey::PaddingW, padding_w_);
    // Set scalar as compat for backends that read scalar only (none should,
    // but defensive). Use H-axis as the canonical scalar.
    fwd_attrs.set(AttrKey::KernelSize, kernel_size_h_);
    fwd_attrs.set(AttrKey::Stride,     stride_h_);
    fwd_attrs.set(AttrKey::Padding,    padding_h_);

    auto dispatch_result = dispatch_to_device(OpId::MaxPool2dForward,
        contig_input.tensor().device().type, inputs, fwd_attrs);
    Tensor output  = dispatch_result[0];
    Tensor indices = dispatch_result[1];

    // Record into the active JIT trace if any.
    {
        auto& tracer = ::tenzor::jit::Tracer::get_instance();
        if (tracer.is_tracing()) {
            auto in_id  = tracer.register_tensor(contig_input.tensor());
            auto out_id = tracer.register_new_tensor(output);
            ::tenzor::jit::TracedOp op(::tenzor::jit::OpType::MaxPool2d,
                                       {in_id}, {out_id});
            op.int_attrs["kernel_size"] = kernel_size_h_;
            op.int_attrs["stride"]      = stride_h_;
            op.int_attrs["padding"]     = padding_h_;
            tracer.record_op(std::move(op));
        }
    }

    auto result = Variable(output, input.requires_grad());

    if (input.requires_grad()) {
        std::vector<Tensor> tensors_to_save = {contig_input.tensor(), indices, output};
        auto backward_fn = std::make_shared<MaxPool2dBackward>(
            kernel_size_h_, stride_h_, padding_h_, std::move(tensors_to_save));
        result.set_grad_fn(backward_fn);
        std::vector<Variable> input_vars;
        input_vars.push_back(input);
        backward_fn->set_input_variables(input_vars);
        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) next_funcs.push_back(input.grad_fn());
        backward_fn->set_next_functions(next_funcs);
    }

    return result;
}

// =======================
// AvgPool2d Implementation
// =======================

// AvgPool2d autograd function
class AvgPool2dBackward : public Function {
public:
    AvgPool2dBackward(int64_t kernel_size, int64_t stride, int64_t padding,
                     int64_t H_in, int64_t W_in,
                     bool count_include_pad,
                     std::vector<Tensor> tensors_to_save)
        : kernel_size_(kernel_size), stride_(stride), padding_(padding),
          H_in_(H_in), W_in_(W_in),
          count_include_pad_(count_include_pad) {
        save_for_backward(std::move(tensors_to_save));
    }

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("AvgPool2dBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        // grad_outputs[0]: gradient w.r.t output [N, C, H_out, W_out]

        const auto& grad_output = grad_outputs[0];
        auto grad_shape = grad_output.shape();
        int64_t N = grad_shape[0];
        int64_t C = grad_shape[1];
        int64_t H_out = grad_shape[2];
        int64_t W_out = grad_shape[3];

        // Use OpId dispatch for non-CPU devices (CUDA, Vulkan, etc.)
        if (grad_output.device().type != Device::Type::CPU) {
            // cuDNN path needs [grad_output, input]; non-cuDNN needs InputShape attr
            // Save input in saved_tensors_[0] for this purpose
            std::vector<Tensor> inputs = {grad_output, saved_tensors_[0]};
            OpAttributes bwd_attrs;
            bwd_attrs.set(AttrKey::InputShape, std::to_string(N) + "," + std::to_string(C) + "," + std::to_string(H_in_) + "," + std::to_string(W_in_));
            bwd_attrs.set(AttrKey::KernelSize, kernel_size_);
            bwd_attrs.set(AttrKey::Stride, stride_);
            bwd_attrs.set(AttrKey::Padding, padding_);
            bwd_attrs.set(AttrKey::CountIncludePad, count_include_pad_ ? int64_t{1} : int64_t{0});
            auto result = dispatch_to_device(OpId::AvgPool2dBackward, grad_output.device().type,
                inputs, bwd_attrs);
            return {result[0]};
        }

        // CPU path - Initialize gradient w.r.t input with zeros on same device
        auto grad_input = zeros({N, C, H_in_, W_in_}, grad_output.dtype(), grad_output.device());

        auto dtype = grad_output.dtype();
        if (dtype == DType::Float32) {
            float* grad_input_data = grad_input.data<float>();
            const float* grad_output_data = grad_output.data<float>();

            // Distribute gradients evenly across pooling windows
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                        for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                            int64_t h_start = h_out * stride_ - padding_;
                            int64_t w_start = w_out * stride_ - padding_;
                            int64_t h_end = h_start + kernel_size_;
                            int64_t w_end = w_start + kernel_size_;

                            int64_t valid_count = 0;
                            for (int64_t h = h_start; h < h_end; ++h) {
                                for (int64_t w = w_start; w < w_end; ++w) {
                                    if (h >= 0 && h < H_in_ && w >= 0 && w < W_in_) {
                                        valid_count++;
                                    }
                                }
                            }
                            int64_t divisor = count_include_pad_
                                ? (kernel_size_ * kernel_size_)
                                : valid_count;
                            if (divisor <= 0) continue;

                            int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
                            float grad_val = grad_output_data[out_idx] / static_cast<float>(divisor);

                            for (int64_t h = h_start; h < h_end; ++h) {
                                for (int64_t w = w_start; w < w_end; ++w) {
                                    if (h >= 0 && h < H_in_ && w >= 0 && w < W_in_) {
                                        int64_t input_idx = ((n * C + c) * H_in_ + h) * W_in_ + w;
                                        grad_input_data[input_idx] += grad_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else if (dtype == DType::Float64) {
            double* grad_input_data = grad_input.data<double>();
            const double* grad_output_data = grad_output.data<double>();

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                        for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                            int64_t h_start = h_out * stride_ - padding_;
                            int64_t w_start = w_out * stride_ - padding_;
                            int64_t h_end = h_start + kernel_size_;
                            int64_t w_end = w_start + kernel_size_;

                            int64_t valid_count = 0;
                            for (int64_t h = h_start; h < h_end; ++h) {
                                for (int64_t w = w_start; w < w_end; ++w) {
                                    if (h >= 0 && h < H_in_ && w >= 0 && w < W_in_) {
                                        valid_count++;
                                    }
                                }
                            }
                            int64_t divisor = count_include_pad_
                                ? (kernel_size_ * kernel_size_)
                                : valid_count;
                            if (divisor <= 0) continue;

                            int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
                            double grad_val = grad_output_data[out_idx] / static_cast<double>(divisor);

                            for (int64_t h = h_start; h < h_end; ++h) {
                                for (int64_t w = w_start; w < w_end; ++w) {
                                    if (h >= 0 && h < H_in_ && w >= 0 && w < W_in_) {
                                        int64_t input_idx = ((n * C + c) * H_in_ + h) * W_in_ + w;
                                        grad_input_data[input_idx] += grad_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else if (dtype == DType::Float16) {
            Float16* grad_input_data = grad_input.data<Float16>();
            const Float16* grad_output_data = grad_output.data<Float16>();

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                        for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                            int64_t h_start = h_out * stride_ - padding_;
                            int64_t w_start = w_out * stride_ - padding_;
                            int64_t h_end = h_start + kernel_size_;
                            int64_t w_end = w_start + kernel_size_;

                            int64_t valid_count = 0;
                            for (int64_t h = h_start; h < h_end; ++h) {
                                for (int64_t w = w_start; w < w_end; ++w) {
                                    if (h >= 0 && h < H_in_ && w >= 0 && w < W_in_) {
                                        valid_count++;
                                    }
                                }
                            }
                            int64_t divisor = count_include_pad_
                                ? (kernel_size_ * kernel_size_)
                                : valid_count;
                            if (divisor <= 0) continue;

                            int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
                            float grad_val = static_cast<float>(grad_output_data[out_idx]) / static_cast<float>(divisor);

                            for (int64_t h = h_start; h < h_end; ++h) {
                                for (int64_t w = w_start; w < w_end; ++w) {
                                    if (h >= 0 && h < H_in_ && w >= 0 && w < W_in_) {
                                        int64_t input_idx = ((n * C + c) * H_in_ + h) * W_in_ + w;
                                        grad_input_data[input_idx] = Float16(static_cast<float>(grad_input_data[input_idx]) + grad_val);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return {grad_input};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        // Pooling backward uses scatter operations -- delegate to tensor backward and wrap results
        std::vector<Tensor> tensor_grads;
        for (auto& v : grad_outputs) tensor_grads.push_back(v.tensor());
        auto results = backward(std::move(tensor_grads));
        std::vector<Variable> var_results;
        for (auto& t : results) var_results.emplace_back(t, false);
        return var_results;
    }

    // P4.2d: the passthrough above returns Variables without grad_fn,
    // so higher-order gradients through this op are structurally zero.
    // Declare that honestly so the engine's disconnection counter
    // reports accurately in Warn mode and throws in Error mode.
    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return true; }

private:
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
    int64_t H_in_;
    int64_t W_in_;
    bool count_include_pad_;
};

AvgPool2d::AvgPool2d(int64_t kernel_size, int64_t stride, int64_t padding,
                     bool count_include_pad)
    : AvgPool2d(std::array<int64_t, 2>{kernel_size, kernel_size},
                std::array<int64_t, 2>{stride, stride},
                std::array<int64_t, 2>{padding, padding},
                count_include_pad) {}

AvgPool2d::AvgPool2d(std::array<int64_t, 2> kernel_size,
                     std::array<int64_t, 2> stride,
                     std::array<int64_t, 2> padding,
                     bool count_include_pad)
    : kernel_size_h_(kernel_size[0]), kernel_size_w_(kernel_size[1]),
      stride_h_(stride[0] < 0 ? kernel_size[0] : stride[0]),
      stride_w_(stride[1] < 0 ? kernel_size[1] : stride[1]),
      padding_h_(padding[0]), padding_w_(padding[1]),
      kernel_size_(kernel_size[0]),
      stride_(stride[0] < 0 ? kernel_size[0] : stride[0]),
      padding_(padding[0]),
      count_include_pad_(count_include_pad) {
    for (int i = 0; i < 2; ++i) {
        if (kernel_size[i] <= 0) {
            throw std::runtime_error("AvgPool2d: kernel_size must be positive (axis " +
                std::to_string(i) + "), got " + std::to_string(kernel_size[i]));
        }
        if (padding[i] < 0) {
            throw std::runtime_error("AvgPool2d: padding must be non-negative (axis " +
                std::to_string(i) + "), got " + std::to_string(padding[i]));
        }
    }
}

auto AvgPool2d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::runtime_error("AvgPool2d expects 4D input [batch, channels, height, width]");
    }

    Variable contig_input = input;
    if (!input.tensor().is_contiguous()) {
        tenzor::utils::wrap_preserving_grad(contig_input, input.tensor().contiguous());
    }

    int64_t N    = input_shape[0];
    int64_t C    = input_shape[1];
    int64_t H_in = input_shape[2];
    int64_t W_in = input_shape[3];

    std::vector<Tensor> inputs = {contig_input.tensor()};
    OpAttributes fwd_attrs;
    fwd_attrs.set(AttrKey::KernelSize,  kernel_size_h_);
    fwd_attrs.set(AttrKey::KernelSizeH, kernel_size_h_);
    fwd_attrs.set(AttrKey::KernelSizeW, kernel_size_w_);
    fwd_attrs.set(AttrKey::Stride,  stride_h_);
    fwd_attrs.set(AttrKey::StrideH, stride_h_);
    fwd_attrs.set(AttrKey::StrideW, stride_w_);
    fwd_attrs.set(AttrKey::Padding,  padding_h_);
    fwd_attrs.set(AttrKey::PaddingH, padding_h_);
    fwd_attrs.set(AttrKey::PaddingW, padding_w_);
    // S22: route count_include_pad through to the kernel.
    fwd_attrs.set(AttrKey::CountIncludePad, count_include_pad_ ? int64_t{1} : int64_t{0});

    auto dispatch_result = dispatch_to_device(OpId::AvgPool2dForward,
        contig_input.tensor().device().type, inputs, fwd_attrs);
    Tensor output = dispatch_result[0];

    {
        auto& tracer = ::tenzor::jit::Tracer::get_instance();
        if (tracer.is_tracing()) {
            auto in_id  = tracer.register_tensor(contig_input.tensor());
            auto out_id = tracer.register_new_tensor(output);
            ::tenzor::jit::TracedOp op(::tenzor::jit::OpType::AvgPool2d,
                                       {in_id}, {out_id});
            op.int_attrs["kernel_size"] = kernel_size_h_;
            op.int_attrs["stride"]      = stride_h_;
            op.int_attrs["padding"]     = padding_h_;
            tracer.record_op(std::move(op));
        }
    }

    auto result = Variable(output, input.requires_grad());
    if (input.requires_grad()) {
        std::vector<Tensor> tensors_to_save = {contig_input.tensor()};
        auto backward_fn = std::make_shared<AvgPool2dBackward>(
            kernel_size_h_, stride_h_, padding_h_, H_in, W_in,
            count_include_pad_,
            std::move(tensors_to_save));
        result.set_grad_fn(backward_fn);
        std::vector<Variable> input_vars;
        input_vars.push_back(input);
        backward_fn->set_input_variables(input_vars);
        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) next_funcs.push_back(input.grad_fn());
        backward_fn->set_next_functions(next_funcs);
    }
    (void)N; (void)C;
    return result;
}

// =======================
// AdaptiveAvgPool2d Implementation
// =======================

// AdaptiveAvgPool2d autograd function
class AdaptiveAvgPool2dBackward : public Function {
public:
    AdaptiveAvgPool2dBackward(int64_t H_in, int64_t W_in,
                             int64_t H_out, int64_t W_out,
                             std::vector<Tensor> tensors_to_save)
        : H_in_(H_in), W_in_(W_in), H_out_(H_out), W_out_(W_out) {
        save_for_backward(std::move(tensors_to_save));
    }

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("AdaptiveAvgPool2dBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        // grad_outputs[0]: gradient w.r.t output [N, C, H_out, W_out]

        const auto& grad_output = grad_outputs[0];
        auto grad_shape = grad_output.shape();
        int64_t N = grad_shape[0];
        int64_t C = grad_shape[1];

        // Save target device for later
        auto target_device = grad_output.device();
        auto target_dtype = grad_output.dtype();

        // Use operation registry to dispatch to appropriate backend
        if (target_device.type != Device::Type::CPU) {
            OpAttributes attrs;
            attrs.set(AttrKey::InputH, H_in_);
            attrs.set(AttrKey::InputW, W_in_);
            std::vector<Tensor> inputs = {grad_output, saved_tensors_[0]};
            return dispatch<OpId::AdaptiveAvgPool2dBackward>(inputs, attrs);
        }

        // CPU fallback
        auto grad_output_cpu = grad_output.to(Device::cpu());
        auto grad_input = zeros({N, C, H_in_, W_in_}, target_dtype, Device::cpu());

        auto dtype = target_dtype;
        if (dtype == DType::Float32) {
            float* grad_input_data = grad_input.data<float>();
            const float* grad_output_data = grad_output_cpu.data<float>();

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t h_out = 0; h_out < H_out_; ++h_out) {
                        for (int64_t w_out = 0; w_out < W_out_; ++w_out) {
                            int64_t h_start = (h_out * H_in_) / H_out_;
                            int64_t h_end = ((h_out + 1) * H_in_) / H_out_;
                            int64_t w_start = (w_out * W_in_) / W_out_;
                            int64_t w_end = ((w_out + 1) * W_in_) / W_out_;

                            int64_t count = (h_end - h_start) * (w_end - w_start);
                            int64_t out_idx = ((n * C + c) * H_out_ + h_out) * W_out_ + w_out;
                            float grad_val = grad_output_data[out_idx] / static_cast<float>(count);

                            for (int64_t h = h_start; h < h_end; ++h) {
                                for (int64_t w = w_start; w < w_end; ++w) {
                                    int64_t input_idx = ((n * C + c) * H_in_ + h) * W_in_ + w;
                                    grad_input_data[input_idx] += grad_val;
                                }
                            }
                        }
                    }
                }
            }
        } else if (dtype == DType::Float64) {
            double* grad_input_data = grad_input.data<double>();
            const double* grad_output_data = grad_output_cpu.data<double>();

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t h_out = 0; h_out < H_out_; ++h_out) {
                        for (int64_t w_out = 0; w_out < W_out_; ++w_out) {
                            int64_t h_start = (h_out * H_in_) / H_out_;
                            int64_t h_end = ((h_out + 1) * H_in_) / H_out_;
                            int64_t w_start = (w_out * W_in_) / W_out_;
                            int64_t w_end = ((w_out + 1) * W_in_) / W_out_;

                            int64_t count = (h_end - h_start) * (w_end - w_start);
                            int64_t out_idx = ((n * C + c) * H_out_ + h_out) * W_out_ + w_out;
                            double grad_val = grad_output_data[out_idx] / static_cast<double>(count);

                            for (int64_t h = h_start; h < h_end; ++h) {
                                for (int64_t w = w_start; w < w_end; ++w) {
                                    int64_t input_idx = ((n * C + c) * H_in_ + h) * W_in_ + w;
                                    grad_input_data[input_idx] += grad_val;
                                }
                            }
                        }
                    }
                }
            }
        } else if (dtype == DType::Float16) {
            auto grad_output_f32 = grad_output_cpu.to(DType::Float32);
            auto grad_input_f32 = zeros({N, C, H_in_, W_in_}, DType::Float32, Device::cpu());

            float* grad_input_data = grad_input_f32.data<float>();
            const float* grad_output_data = grad_output_f32.data<float>();

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t h_out = 0; h_out < H_out_; ++h_out) {
                        for (int64_t w_out = 0; w_out < W_out_; ++w_out) {
                            int64_t h_start = (h_out * H_in_) / H_out_;
                            int64_t h_end = ((h_out + 1) * H_in_) / H_out_;
                            int64_t w_start = (w_out * W_in_) / W_out_;
                            int64_t w_end = ((w_out + 1) * W_in_) / W_out_;

                            int64_t count = (h_end - h_start) * (w_end - w_start);
                            int64_t out_idx = ((n * C + c) * H_out_ + h_out) * W_out_ + w_out;
                            float grad_val = grad_output_data[out_idx] / static_cast<float>(count);

                            for (int64_t h = h_start; h < h_end; ++h) {
                                for (int64_t w = w_start; w < w_end; ++w) {
                                    int64_t input_idx = ((n * C + c) * H_in_ + h) * W_in_ + w;
                                    grad_input_data[input_idx] += grad_val;
                                }
                            }
                        }
                    }
                }
            }

            grad_input = grad_input_f32.to(DType::Float16);
        } else {
            throw std::runtime_error("AdaptiveAvgPool2dBackward: Unsupported dtype");
        }

        return {grad_input};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        // Adaptive pooling backward uses scatter operations -- delegate to tensor backward and wrap results
        std::vector<Tensor> tensor_grads;
        for (auto& v : grad_outputs) tensor_grads.push_back(v.tensor());
        auto results = backward(std::move(tensor_grads));
        std::vector<Variable> var_results;
        for (auto& t : results) var_results.emplace_back(t, false);
        return var_results;
    }

    // P4.2d: the passthrough above returns Variables without grad_fn,
    // so higher-order gradients through this op are structurally zero.
    // Declare that honestly so the engine's disconnection counter
    // reports accurately in Warn mode and throws in Error mode.
    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return true; }

private:
    int64_t H_in_;
    int64_t W_in_;
    int64_t H_out_;
    int64_t W_out_;
};

AdaptiveAvgPool2d::AdaptiveAvgPool2d(int64_t output_h, int64_t output_w)
    : output_h_(output_h), output_w_(output_w) {}

auto AdaptiveAvgPool2d::forward_impl(const Variable& input) -> Variable {
    // Input shape: [N, C, H_in, W_in]
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::runtime_error("AdaptiveAvgPool2d expects 4D input [batch, channels, height, width]");
    }

    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t H_in = input_shape[2];
    int64_t W_in = input_shape[3];

    // Output dimensions are fixed
    int64_t H_out = output_h_;
    int64_t W_out = output_w_;

    // Save original device and dtype for later
    auto target_device = input.tensor().device();
    auto target_dtype = input.tensor().dtype();

    Tensor output;

    // Use operation registry to dispatch to appropriate backend
    if (target_device.type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::OutputSizeH, H_out);
        attrs.set(AttrKey::OutputSizeW, W_out);
        std::vector<Tensor> inputs = {input.tensor()};
        auto results = dispatch<OpId::AdaptiveAvgPool2d>(inputs, attrs);
        output = results[0];
    } else {
        // CPU fallback
        auto input_cpu = input.tensor().to(Device::cpu());
        output = zeros({N, C, H_out, W_out}, target_dtype, Device::cpu());

        auto dtype = target_dtype;
        if (dtype == DType::Float32) {
            const float* input_data = input_cpu.data<float>();
            float* output_data = output.data<float>();

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                        for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                            int64_t h_start = (h_out * H_in) / H_out;
                            int64_t h_end = ((h_out + 1) * H_in) / H_out;
                            int64_t w_start = (w_out * W_in) / W_out;
                            int64_t w_end = ((w_out + 1) * W_in) / W_out;

                            float sum = 0.0f;
                            int64_t count = 0;

                            for (int64_t h = h_start; h < h_end; ++h) {
                                for (int64_t w = w_start; w < w_end; ++w) {
                                    int64_t input_idx = ((n * C + c) * H_in + h) * W_in + w;
                                    sum += input_data[input_idx];
                                    count++;
                                }
                            }

                            int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
                            output_data[out_idx] = sum / static_cast<float>(count);
                        }
                    }
                }
            }
        } else if (dtype == DType::Float64) {
            const double* input_data = input_cpu.data<double>();
            double* output_data = output.data<double>();

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                        for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                            int64_t h_start = (h_out * H_in) / H_out;
                            int64_t h_end = ((h_out + 1) * H_in) / H_out;
                            int64_t w_start = (w_out * W_in) / W_out;
                            int64_t w_end = ((w_out + 1) * W_in) / W_out;

                            double sum = 0.0;
                            int64_t count = 0;

                            for (int64_t h = h_start; h < h_end; ++h) {
                                for (int64_t w = w_start; w < w_end; ++w) {
                                    int64_t input_idx = ((n * C + c) * H_in + h) * W_in + w;
                                    sum += input_data[input_idx];
                                    count++;
                                }
                            }

                            int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
                            output_data[out_idx] = sum / static_cast<double>(count);
                        }
                    }
                }
            }
        } else if (dtype == DType::Float16) {
            auto input_f32 = input_cpu.to(DType::Float32);
            auto output_f32 = zeros({N, C, H_out, W_out}, DType::Float32, Device::cpu());

            const float* input_data = input_f32.data<float>();
            float* output_data = output_f32.data<float>();

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                        for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                            int64_t h_start = (h_out * H_in) / H_out;
                            int64_t h_end = ((h_out + 1) * H_in) / H_out;
                            int64_t w_start = (w_out * W_in) / W_out;
                            int64_t w_end = ((w_out + 1) * W_in) / W_out;

                            float sum = 0.0f;
                            int64_t count = 0;

                            for (int64_t h = h_start; h < h_end; ++h) {
                                for (int64_t w = w_start; w < w_end; ++w) {
                                    int64_t input_idx = ((n * C + c) * H_in + h) * W_in + w;
                                    sum += input_data[input_idx];
                                    count++;
                                }
                            }

                            int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
                            output_data[out_idx] = sum / static_cast<float>(count);
                        }
                    }
                }
            }

            output = output_f32.to(DType::Float16);
        } else if (dtype == DType::BFloat16) {
            auto input_f32 = input_cpu.to(DType::Float32);
            auto output_f32 = zeros({N, C, H_out, W_out}, DType::Float32, Device::cpu());

            const float* input_data = input_f32.data<float>();
            float* output_data = output_f32.data<float>();

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                        for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                            int64_t h_start = (h_out * H_in) / H_out;
                            int64_t h_end = ((h_out + 1) * H_in) / H_out;
                            int64_t w_start = (w_out * W_in) / W_out;
                            int64_t w_end = ((w_out + 1) * W_in) / W_out;

                            float sum = 0.0f;
                            int64_t count = 0;

                            for (int64_t h = h_start; h < h_end; ++h) {
                                for (int64_t w = w_start; w < w_end; ++w) {
                                    int64_t input_idx = ((n * C + c) * H_in + h) * W_in + w;
                                    sum += input_data[input_idx];
                                    count++;
                                }
                            }

                            int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
                            output_data[out_idx] = sum / static_cast<float>(count);
                        }
                    }
                }
            }

            output = output_f32.to(DType::BFloat16);
        } else {
            throw std::runtime_error("AdaptiveAvgPool2d: Unsupported dtype");
        }
    }

    // JIT trace recording for the CPU path (which bypasses dispatch).
    {
        auto& tracer = ::tenzor::jit::Tracer::get_instance();
        if (tracer.is_tracing()) {
            auto in_id  = tracer.register_tensor(input.tensor());
            auto out_id = tracer.register_new_tensor(output);
            ::tenzor::jit::TracedOp op(
                ::tenzor::jit::OpType::AdaptiveAvgPool2d,
                {in_id}, {out_id});
            op.vec_attrs["output_size"] = {H_out, W_out};
            tracer.record_op(std::move(op));
        }
    }

    // Create output variable with autograd support
    auto result = Variable(output, input.requires_grad());

    // Setup backward function if gradient is required
    if (input.requires_grad()) {
        std::vector<Tensor> tensors_to_save = {input.tensor()};

        auto backward_fn = std::make_shared<AdaptiveAvgPool2dBackward>(
            H_in, W_in, H_out, W_out, std::move(tensors_to_save)
        );

        result.set_grad_fn(backward_fn);

        std::vector<Variable> input_vars;
        input_vars.push_back(input);
        backward_fn->set_input_variables(input_vars);

        // CRITICAL: Connect to input's grad_fn to continue the backward chain
        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) {
            next_funcs.push_back(input.grad_fn());
        }
        backward_fn->set_next_functions(next_funcs);
    }

    return result;
}

// ============================================================================
// Backward Function classes for 1D and 3D pooling
// ============================================================================
// These wrap the registered OpId backward kernels to wire pool ops into the
// autograd graph. Each Function class:
//   - Saves the input shape (and indices for max-pool variants)
//   - On backward(), dispatches to the registered backend kernel via OpId
//   - Returns the resulting input gradient
//
// The backward kernels are registered in src/backends/*/cpu_kernel_registry.cpp
// (and equivalent for other backends), so this dispatcher pattern works on all
// backends without needing CPU-specific code paths in this file.

namespace {

inline std::string pool_shape_to_string(const std::vector<int64_t>& shape) {
    std::string s;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) s += ',';
        s += std::to_string(shape[i]);
    }
    return s;
}

// Generic backward Function for pool ops that need an input shape attribute.
// Used for AvgPool variants (no indices needed).
template <OpId BackwardOp>
class AvgPoolNdBackward : public Function {
public:
    AvgPoolNdBackward(std::vector<int64_t> input_shape,
                      int64_t kernel_size, int64_t stride, int64_t padding,
                      bool count_include_pad = true)
        : input_shape_(std::move(input_shape)),
          kernel_size_(kernel_size), stride_(stride), padding_(padding),
          count_include_pad_(count_include_pad) {}

    auto forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> override {
        throw std::runtime_error("AvgPoolNdBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        OpAttributes attrs;
        attrs.set(AttrKey::InputShape, pool_shape_to_string(input_shape_));
        attrs.set(AttrKey::KernelSize, kernel_size_);
        attrs.set(AttrKey::Stride, stride_);
        attrs.set(AttrKey::Padding, padding_);
        // S22: forward kernel divisor depends on this flag; the backward
        // must use the same divisor or grads will be off-scale at the
        // padding borders.
        attrs.set(AttrKey::CountIncludePad, count_include_pad_ ? int64_t{1} : int64_t{0});
        std::vector<Tensor> inputs = {grad_outputs[0]};
        auto result = dispatch_to_device(BackwardOp, grad_outputs[0].device().type, inputs, attrs);
        return {result[0]};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        std::vector<Tensor> tensor_grads;
        for (auto& v : grad_outputs) tensor_grads.push_back(v.tensor());
        auto results = backward(std::move(tensor_grads));
        std::vector<Variable> var_results;
        for (auto& t : results) var_results.emplace_back(t, false);
        return var_results;
    }

    // P4.2d: the passthrough above returns Variables without grad_fn,
    // so higher-order gradients through this op are structurally zero.
    // Declare that honestly so the engine's disconnection counter
    // reports accurately in Warn mode and throws in Error mode.
    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return true; }

private:
    std::vector<int64_t> input_shape_;
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
    bool count_include_pad_;
};

// Generic backward Function for max-pool variants (needs saved indices).
template <OpId BackwardOp>
class MaxPoolNdBackward : public Function {
public:
    MaxPoolNdBackward(std::vector<int64_t> input_shape, Tensor indices)
        : input_shape_(std::move(input_shape)) {
        save_for_backward({std::move(indices)});
    }

    auto forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> override {
        throw std::runtime_error("MaxPoolNdBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        OpAttributes attrs;
        attrs.set(AttrKey::InputShape, pool_shape_to_string(input_shape_));
        std::vector<Tensor> inputs = {grad_outputs[0], saved_tensors_[0]};
        auto result = dispatch_to_device(BackwardOp, grad_outputs[0].device().type, inputs, attrs);
        return {result[0]};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        std::vector<Tensor> tensor_grads;
        for (auto& v : grad_outputs) tensor_grads.push_back(v.tensor());
        auto results = backward(std::move(tensor_grads));
        std::vector<Variable> var_results;
        for (auto& t : results) var_results.emplace_back(t, false);
        return var_results;
    }

    // P4.2d: the passthrough above returns Variables without grad_fn,
    // so higher-order gradients through this op are structurally zero.
    // Declare that honestly so the engine's disconnection counter
    // reports accurately in Warn mode and throws in Error mode.
    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return true; }

private:
    std::vector<int64_t> input_shape_;
};

// Helper to wire a backward Function as the grad_fn of a result Variable.
inline void wire_grad_fn(Variable& result, const Variable& input,
                         std::shared_ptr<Function> backward_fn) {
    result.set_grad_fn(backward_fn);
    std::vector<Variable> input_vars{input};
    backward_fn->set_input_variables(input_vars);
    std::vector<std::shared_ptr<Function>> next_funcs;
    if (input.grad_fn()) next_funcs.push_back(input.grad_fn());
    backward_fn->set_next_functions(next_funcs);
}

} // anonymous namespace

// ============================================================================
// MaxPool3d Implementation
// ============================================================================

MaxPool3d::MaxPool3d(int64_t kernel_size, int64_t stride, int64_t padding,
                     bool ceil_mode, bool return_indices)
    : MaxPool3d(std::array<int64_t, 3>{kernel_size, kernel_size, kernel_size},
                std::array<int64_t, 3>{stride, stride, stride},
                std::array<int64_t, 3>{padding, padding, padding},
                ceil_mode, return_indices) {
}

MaxPool3d::MaxPool3d(std::array<int64_t, 3> kernel_size,
                     std::array<int64_t, 3> stride,
                     std::array<int64_t, 3> padding,
                     bool ceil_mode, bool return_indices)
    : kernel_size_d_(kernel_size[0]), kernel_size_h_(kernel_size[1]), kernel_size_w_(kernel_size[2]),
      stride_d_(stride[0] < 0 ? kernel_size[0] : stride[0]),
      stride_h_(stride[1] < 0 ? kernel_size[1] : stride[1]),
      stride_w_(stride[2] < 0 ? kernel_size[2] : stride[2]),
      padding_d_(padding[0]), padding_h_(padding[1]), padding_w_(padding[2]),
      kernel_size_(kernel_size[0]),
      stride_(stride[0] < 0 ? kernel_size[0] : stride[0]),
      padding_(padding[0]),
      ceil_mode_(ceil_mode), return_indices_(return_indices) {
    for (int i = 0; i < 3; ++i) {
        if (kernel_size[i] <= 0) {
            throw std::runtime_error("MaxPool3d: kernel_size must be positive (axis " +
                std::to_string(i) + "), got " + std::to_string(kernel_size[i]));
        }
        if (padding[i] < 0) {
            throw std::runtime_error("MaxPool3d: padding must be non-negative (axis " +
                std::to_string(i) + "), got " + std::to_string(padding[i]));
        }
    }
}

auto MaxPool3d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 5) {
        throw std::invalid_argument("MaxPool3d expects 5D input [batch, channels, depth, height, width]");
    }

    Device device = input.tensor().device();
    std::vector<int64_t> in_shape_vec(input_shape.begin(), input_shape.end());

    OpAttributes fwd_attrs;
    fwd_attrs.set(AttrKey::KernelSize,  kernel_size_d_);
    fwd_attrs.set(AttrKey::KernelSizeD, kernel_size_d_);
    fwd_attrs.set(AttrKey::KernelSizeH, kernel_size_h_);
    fwd_attrs.set(AttrKey::KernelSizeW, kernel_size_w_);
    fwd_attrs.set(AttrKey::Stride,  stride_d_);
    fwd_attrs.set(AttrKey::StrideD, stride_d_);
    fwd_attrs.set(AttrKey::StrideH, stride_h_);
    fwd_attrs.set(AttrKey::StrideW, stride_w_);
    fwd_attrs.set(AttrKey::Padding,  padding_d_);
    fwd_attrs.set(AttrKey::PaddingD, padding_d_);
    fwd_attrs.set(AttrKey::PaddingH, padding_h_);
    fwd_attrs.set(AttrKey::PaddingW, padding_w_);

    std::vector<Tensor> inputs = {input.tensor()};
    auto fwd_result = dispatch_to_device(OpId::MaxPool3dForward, device.type, inputs, fwd_attrs);
    Tensor output = fwd_result[0];
    Tensor indices = fwd_result[1];

    auto result = Variable(output, input.requires_grad());

    if (input.requires_grad()) {
        auto backward_fn = std::make_shared<MaxPoolNdBackward<OpId::MaxPool3dBackward>>(
            in_shape_vec, indices);
        wire_grad_fn(result, input, backward_fn);
    }

    return result;
}

// ============================================================================
// AvgPool3d Implementation
// ============================================================================

AvgPool3d::AvgPool3d(int64_t kernel_size, int64_t stride, int64_t padding,
                     bool count_include_pad)
    : AvgPool3d(std::array<int64_t, 3>{kernel_size, kernel_size, kernel_size},
                std::array<int64_t, 3>{stride, stride, stride},
                std::array<int64_t, 3>{padding, padding, padding},
                count_include_pad) {}

AvgPool3d::AvgPool3d(std::array<int64_t, 3> kernel_size,
                     std::array<int64_t, 3> stride,
                     std::array<int64_t, 3> padding,
                     bool count_include_pad)
    : kernel_size_d_(kernel_size[0]), kernel_size_h_(kernel_size[1]), kernel_size_w_(kernel_size[2]),
      stride_d_(stride[0] < 0 ? kernel_size[0] : stride[0]),
      stride_h_(stride[1] < 0 ? kernel_size[1] : stride[1]),
      stride_w_(stride[2] < 0 ? kernel_size[2] : stride[2]),
      padding_d_(padding[0]), padding_h_(padding[1]), padding_w_(padding[2]),
      kernel_size_(kernel_size[0]),
      stride_(stride[0] < 0 ? kernel_size[0] : stride[0]),
      padding_(padding[0]),
      count_include_pad_(count_include_pad) {
    for (int i = 0; i < 3; ++i) {
        if (kernel_size[i] <= 0) {
            throw std::runtime_error("AvgPool3d: kernel_size must be positive (axis " +
                std::to_string(i) + "), got " + std::to_string(kernel_size[i]));
        }
        if (padding[i] < 0) {
            throw std::runtime_error("AvgPool3d: padding must be non-negative (axis " +
                std::to_string(i) + "), got " + std::to_string(padding[i]));
        }
    }
}

auto AvgPool3d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 5) {
        throw std::invalid_argument("AvgPool3d expects 5D input [batch, channels, depth, height, width]");
    }

    Device device = input.tensor().device();
    std::vector<int64_t> in_shape_vec(input_shape.begin(), input_shape.end());

    OpAttributes fwd_attrs;
    fwd_attrs.set(AttrKey::KernelSize,  kernel_size_d_);
    fwd_attrs.set(AttrKey::KernelSizeD, kernel_size_d_);
    fwd_attrs.set(AttrKey::KernelSizeH, kernel_size_h_);
    fwd_attrs.set(AttrKey::KernelSizeW, kernel_size_w_);
    fwd_attrs.set(AttrKey::Stride,  stride_d_);
    fwd_attrs.set(AttrKey::StrideD, stride_d_);
    fwd_attrs.set(AttrKey::StrideH, stride_h_);
    fwd_attrs.set(AttrKey::StrideW, stride_w_);
    fwd_attrs.set(AttrKey::Padding,  padding_d_);
    fwd_attrs.set(AttrKey::PaddingD, padding_d_);
    fwd_attrs.set(AttrKey::PaddingH, padding_h_);
    fwd_attrs.set(AttrKey::PaddingW, padding_w_);
    // S22: route count_include_pad through to the kernel.
    fwd_attrs.set(AttrKey::CountIncludePad, count_include_pad_ ? int64_t{1} : int64_t{0});

    std::vector<Tensor> inputs = {input.tensor()};
    auto fwd_result = dispatch_to_device(OpId::AvgPool3dForward, device.type, inputs, fwd_attrs);
    Tensor output = fwd_result[0];

    auto result = Variable(output, input.requires_grad());

    if (input.requires_grad()) {
        auto backward_fn = std::make_shared<AvgPoolNdBackward<OpId::AvgPool3dBackward>>(
            in_shape_vec, kernel_size_d_, stride_d_, padding_d_,
            count_include_pad_);
        wire_grad_fn(result, input, backward_fn);
    }

    return result;
}

// ============================================================================
// MaxPool1d implementation
// ============================================================================

MaxPool1d::MaxPool1d(int64_t kernel_size, int64_t stride, int64_t padding,
                     bool ceil_mode, bool return_indices)
    : kernel_size_(kernel_size), stride_(stride < 0 ? kernel_size : stride),
      padding_(padding), ceil_mode_(ceil_mode), return_indices_(return_indices) {
    if (kernel_size <= 0) throw std::runtime_error("MaxPool1d: kernel_size must be positive");
    if (padding < 0) throw std::runtime_error("MaxPool1d: padding must be non-negative");
}

auto MaxPool1d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 3) {
        throw std::invalid_argument("MaxPool1d expects 3D input [batch, channels, length]");
    }

    Device device = input.tensor().device();
    std::vector<int64_t> in_shape_vec(input_shape.begin(), input_shape.end());

    OpAttributes fwd_attrs;
    fwd_attrs.set(AttrKey::KernelSize, kernel_size_);
    fwd_attrs.set(AttrKey::Stride, stride_);
    fwd_attrs.set(AttrKey::Padding, padding_);

    std::vector<Tensor> inputs = {input.tensor()};
    auto fwd_result = dispatch_to_device(OpId::MaxPool1dForward, device.type, inputs, fwd_attrs);
    Tensor output = fwd_result[0];
    Tensor indices = fwd_result[1];

    auto result = Variable(output, input.requires_grad());

    if (input.requires_grad()) {
        auto backward_fn = std::make_shared<MaxPoolNdBackward<OpId::MaxPool1dBackward>>(
            in_shape_vec, indices);
        wire_grad_fn(result, input, backward_fn);
    }

    return result;
}

// ============================================================================
// AvgPool1d implementation
// ============================================================================

AvgPool1d::AvgPool1d(int64_t kernel_size, int64_t stride, int64_t padding,
                     bool count_include_pad)
    : kernel_size_(kernel_size), stride_(stride < 0 ? kernel_size : stride),
      padding_(padding),
      count_include_pad_(count_include_pad) {
    if (kernel_size <= 0) throw std::runtime_error("AvgPool1d: kernel_size must be positive");
    if (padding < 0) throw std::runtime_error("AvgPool1d: padding must be non-negative");
}

auto AvgPool1d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 3) {
        throw std::invalid_argument("AvgPool1d expects 3D input [batch, channels, length]");
    }

    Device device = input.tensor().device();
    std::vector<int64_t> in_shape_vec(input_shape.begin(), input_shape.end());

    OpAttributes fwd_attrs;
    fwd_attrs.set(AttrKey::KernelSize, kernel_size_);
    fwd_attrs.set(AttrKey::Stride, stride_);
    fwd_attrs.set(AttrKey::Padding, padding_);
    // S22: route count_include_pad through to the kernel.
    fwd_attrs.set(AttrKey::CountIncludePad, count_include_pad_ ? int64_t{1} : int64_t{0});

    std::vector<Tensor> inputs = {input.tensor()};
    auto fwd_result = dispatch_to_device(OpId::AvgPool1dForward, device.type, inputs, fwd_attrs);
    Tensor output = fwd_result[0];

    auto result = Variable(output, input.requires_grad());

    if (input.requires_grad()) {
        auto backward_fn = std::make_shared<AvgPoolNdBackward<OpId::AvgPool1dBackward>>(
            in_shape_vec, kernel_size_, stride_, padding_,
            count_include_pad_);
        wire_grad_fn(result, input, backward_fn);
    }

    return result;
}

// ============================================================================
// AdaptiveAvgPool1d implementation
// ============================================================================

AdaptiveAvgPool1d::AdaptiveAvgPool1d(int64_t output_size)
    : output_size_(output_size) {
    if (output_size <= 0) throw std::runtime_error("AdaptiveAvgPool1d: output_size must be positive");
}

auto AdaptiveAvgPool1d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 3) {
        throw std::invalid_argument("AdaptiveAvgPool1d expects 3D input [batch, channels, length]");
    }

    Device device = input.tensor().device();
    std::vector<int64_t> in_shape_vec(input_shape.begin(), input_shape.end());

    OpAttributes fwd_attrs;
    fwd_attrs.set(AttrKey::OutputSize, output_size_);

    std::vector<Tensor> inputs = {input.tensor()};
    auto fwd_result = dispatch_to_device(OpId::AdaptiveAvgPool1d, device.type, inputs, fwd_attrs);
    Tensor output = fwd_result[0];

    auto result = Variable(output, input.requires_grad());

    if (input.requires_grad()) {
        // AdaptiveAvgPool backward only needs the input shape (no kernel/stride/padding).
        auto backward_fn = std::make_shared<AvgPoolNdBackward<OpId::AdaptiveAvgPool1dBackward>>(
            in_shape_vec, /*kernel_size=*/0, /*stride=*/0, /*padding=*/0);
        wire_grad_fn(result, input, backward_fn);
    }

    return result;
}

// ============================================================================
// AdaptiveMaxPool2d implementation
// ============================================================================

AdaptiveMaxPool2d::AdaptiveMaxPool2d(int64_t output_h, int64_t output_w)
    : output_h_(output_h), output_w_(output_w) {
    if (output_h <= 0 || output_w <= 0) {
        throw std::runtime_error("AdaptiveMaxPool2d: output dimensions must be positive");
    }
}

auto AdaptiveMaxPool2d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("AdaptiveMaxPool2d expects 4D input [batch, channels, height, width]");
    }

    Device device = input.tensor().device();
    std::vector<int64_t> in_shape_vec(input_shape.begin(), input_shape.end());

    OpAttributes fwd_attrs;
    fwd_attrs.set(AttrKey::OutputSizeH, output_h_);
    fwd_attrs.set(AttrKey::OutputSizeW, output_w_);

    // Dispatch to backend kernel (CPU kernel supports Float32/Float64/Float16/BFloat16,
    // matching the other adaptive-pool layers).  The inlined CPU path that only
    // handled Float32/Float64 has been removed.
    std::vector<Tensor> inputs = {input.tensor()};
    auto fwd_result = dispatch_to_device(OpId::AdaptiveMaxPool2d, device.type, inputs, fwd_attrs);
    Tensor output = fwd_result[0];
    Tensor indices = fwd_result[1];

    auto result = Variable(output, input.requires_grad());

    if (input.requires_grad()) {
        auto backward_fn = std::make_shared<MaxPoolNdBackward<OpId::AdaptiveMaxPool2dBackward>>(
            in_shape_vec, indices);
        wire_grad_fn(result, input, backward_fn);
    }

    return result;
}

// ============================================================================
// AdaptiveMaxPool1d implementation
// ============================================================================

AdaptiveMaxPool1d::AdaptiveMaxPool1d(int64_t output_size)
    : output_size_(output_size) {
    if (output_size <= 0) throw std::runtime_error("AdaptiveMaxPool1d: output_size must be positive");
}

auto AdaptiveMaxPool1d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 3) {
        throw std::invalid_argument("AdaptiveMaxPool1d expects 3D input [batch, channels, length]");
    }

    Device device = input.tensor().device();
    std::vector<int64_t> in_shape_vec(input_shape.begin(), input_shape.end());

    OpAttributes fwd_attrs;
    fwd_attrs.set(AttrKey::OutputSize, output_size_);

    std::vector<Tensor> inputs = {input.tensor()};
    auto fwd_result = dispatch_to_device(OpId::AdaptiveMaxPool1d, device.type, inputs, fwd_attrs);
    Tensor output = fwd_result[0];
    Tensor indices = fwd_result[1];

    auto result = Variable(output, input.requires_grad());

    if (input.requires_grad()) {
        auto backward_fn = std::make_shared<MaxPoolNdBackward<OpId::AdaptiveMaxPool1dBackward>>(
            in_shape_vec, indices);
        wire_grad_fn(result, input, backward_fn);
    }

    return result;
}

// ============================================================================
// AdaptiveMaxPool3d implementation
// ============================================================================

AdaptiveMaxPool3d::AdaptiveMaxPool3d(int64_t output_d, int64_t output_h, int64_t output_w)
    : output_d_(output_d), output_h_(output_h), output_w_(output_w) {
    if (output_d <= 0 || output_h <= 0 || output_w <= 0) {
        throw std::runtime_error("AdaptiveMaxPool3d: output dimensions must be positive");
    }
}

auto AdaptiveMaxPool3d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 5) {
        throw std::invalid_argument("AdaptiveMaxPool3d expects 5D input [batch, channels, depth, height, width]");
    }

    Device device = input.tensor().device();
    std::vector<int64_t> in_shape_vec(input_shape.begin(), input_shape.end());

    OpAttributes fwd_attrs;
    fwd_attrs.set(AttrKey::OutputSizeD, output_d_);
    fwd_attrs.set(AttrKey::OutputSizeH, output_h_);
    fwd_attrs.set(AttrKey::OutputSizeW, output_w_);

    std::vector<Tensor> inputs = {input.tensor()};
    auto fwd_result = dispatch_to_device(OpId::AdaptiveMaxPool3d, device.type, inputs, fwd_attrs);
    Tensor output = fwd_result[0];
    Tensor indices = fwd_result[1];

    auto result = Variable(output, input.requires_grad());

    if (input.requires_grad()) {
        auto backward_fn = std::make_shared<MaxPoolNdBackward<OpId::AdaptiveMaxPool3dBackward>>(
            in_shape_vec, indices);
        wire_grad_fn(result, input, backward_fn);
    }

    return result;
}

// ============================================================================
// AdaptiveAvgPool3d implementation
// ============================================================================

AdaptiveAvgPool3d::AdaptiveAvgPool3d(int64_t output_d, int64_t output_h, int64_t output_w)
    : output_d_(output_d), output_h_(output_h), output_w_(output_w) {
    if (output_d <= 0 || output_h <= 0 || output_w <= 0) {
        throw std::runtime_error("AdaptiveAvgPool3d: output dimensions must be positive");
    }
}

auto AdaptiveAvgPool3d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 5) {
        throw std::invalid_argument("AdaptiveAvgPool3d expects 5D input [batch, channels, depth, height, width]");
    }

    Device device = input.tensor().device();
    std::vector<int64_t> in_shape_vec(input_shape.begin(), input_shape.end());

    OpAttributes fwd_attrs;
    fwd_attrs.set(AttrKey::OutputSizeD, output_d_);
    fwd_attrs.set(AttrKey::OutputSizeH, output_h_);
    fwd_attrs.set(AttrKey::OutputSizeW, output_w_);

    std::vector<Tensor> inputs = {input.tensor()};
    auto fwd_result = dispatch_to_device(OpId::AdaptiveAvgPool3d, device.type, inputs, fwd_attrs);
    Tensor output = fwd_result[0];

    auto result = Variable(output, input.requires_grad());

    if (input.requires_grad()) {
        // Adaptive pool backward only needs the input shape.
        auto backward_fn = std::make_shared<AvgPoolNdBackward<OpId::AdaptiveAvgPool3dBackward>>(
            in_shape_vec, /*kernel_size=*/0, /*stride=*/0, /*padding=*/0);
        wire_grad_fn(result, input, backward_fn);
    }

    return result;
}

// ============================================================================
// LPPool1d implementation
// ============================================================================

LPPool1d::LPPool1d(int64_t norm_type, int64_t kernel_size, int64_t stride)
    : norm_type_(norm_type), kernel_size_(kernel_size),
      stride_(stride == -1 ? kernel_size : stride) {
    if (norm_type < 1) throw std::runtime_error("LPPool1d: norm_type must be >= 1");
    if (kernel_size <= 0) throw std::runtime_error("LPPool1d: kernel_size must be positive");
}

auto LPPool1d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 3) {
        throw std::invalid_argument("LPPool1d expects 3D input [batch, channels, length]");
    }

    // LPPool: (avg_pool(|x|^p))^(1/p) — Variable-level throughout so
    // backward() actually populates input.grad. The prior implementation
    // called input.tensor() up front and re-wrapped intermediate Variables
    // with requires_grad=false, severing the graph.
    auto x_abs = ::tenzor::abs(input);
    auto x_pow = ::tenzor::pow(x_abs, static_cast<double>(norm_type_));

    AvgPool1d avg_pool(kernel_size_, stride_, /*padding=*/0);
    auto pooled = avg_pool.forward(x_pow);

    double inv_p = 1.0 / static_cast<double>(norm_type_);
    return ::tenzor::pow(pooled, inv_p);
}

// ============================================================================
// LPPool2d implementation
// ============================================================================

LPPool2d::LPPool2d(int64_t norm_type,
                   std::pair<int64_t, int64_t> kernel_size,
                   std::pair<int64_t, int64_t> stride)
    : norm_type_(norm_type), kernel_size_(kernel_size),
      stride_(stride.first == -1
              ? kernel_size
              : stride) {
    if (norm_type < 1) throw std::runtime_error("LPPool2d: norm_type must be >= 1");
    if (kernel_size.first <= 0 || kernel_size.second <= 0)
        throw std::runtime_error("LPPool2d: kernel_size must be positive");
}

auto LPPool2d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("LPPool2d expects 4D input [batch, channels, height, width]");
    }

    // LPPool: (avg_pool(|x|^p))^(1/p) — Variable-level (see LPPool1d comment).
    auto x_abs = ::tenzor::abs(input);
    auto x_pow = ::tenzor::pow(x_abs, static_cast<double>(norm_type_));

    AvgPool2d avg_pool(kernel_size_.first, stride_.first, /*padding=*/0);
    auto pooled = avg_pool.forward(x_pow);

    double inv_p = 1.0 / static_cast<double>(norm_type_);
    return ::tenzor::pow(pooled, inv_p);
}

} // namespace tenzor::nn
