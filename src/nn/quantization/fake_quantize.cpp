/**
 * @file fake_quantize.cpp
 * @brief Implementation of fake quantization for QAT
 */

#include "tenzor/nn/quantization/fake_quantize.hpp"
#include "tenzor/nn/quantization/quantized_layers.hpp"
#include "tenzor/nn/quantization/qconfig.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/layers/batchnorm.hpp"
#include "tenzor/nn/module.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/jit/tracer.hpp"
#include <cmath>
#include <stdexcept>

namespace tenzor {
namespace nn {
namespace quantization {

using tenzor::nn::Linear;
using tenzor::nn::Conv2d;
using tenzor::nn::Sequential;

// ============================================================================
// FakeQuantize
// ============================================================================

FakeQuantize::FakeQuantize(
    QuantDType dtype,
    QuantizationScheme scheme,
    bool learnable,
    bool observer_enabled,
    int64_t axis
) : dtype_(dtype),
    scheme_(scheme),
    learnable_(learnable),
    observer_enabled_(observer_enabled),
    axis_(axis) {

    // Create observer
    bool use_histogram = false;  // Can be made configurable
    observer_ = make_observer(scheme, use_histogram, axis);

    // Initialize quantization params with dummy values
    Tensor scale({1}, DType::Float32, Device::cpu());
    Tensor zero_point({1}, DType::Int32, Device::cpu());
    scale.fill_(1.0f);
    zero_point.fill_(0);

    qparams_ = std::make_unique<QuantizationParams>(
        scale, zero_point, dtype, scheme, axis
    );

    if (learnable_) {
        // Register scale and zero_point as parameters
        Variable scale_var(qparams_->scale, true);
        Variable zp_var(qparams_->zero_point, false);  // Zero point typically not learnable
        register_parameter("scale", scale_var);
        register_parameter("zero_point", zp_var);
    }
}

auto FakeQuantize::forward_impl(const Variable& input) -> Variable {
    if (!fake_quant_enabled_) {
        return input;  // Pass through
    }

    // JIT-R065: whenever this call could recompute qparams_ (observer
    // enabled, non-learnable, training mode — the exact condition guarding
    // calculate_qparams() below), qparams_->scale/zero_point are read as
    // plain host floats a few lines down and baked into the STE/observer
    // paths as trace-time constants (fake_quantize_with_grad's `scale`
    // parameter, apply_fake_quantization's qparams_ read). A trace can only
    // ever capture whichever value happened to be resident at trace time,
    // so every replay would silently freeze QAT calibration at that one
    // snapshot forever — even though eager execution keeps refining scale/
    // zero_point every mini-batch. Not gated on observer_->has_data() at
    // trace time: even a not-yet-calibrated FIRST call would freeze the
    // dummy initial scale=1.0/zero_point=0 for all future replays. Mirrors
    // JIT-R063 (VariationalDropout)'s identical cached-member-state class.
    if (observer_enabled_ && !learnable_ && training_ &&
        ::tenzor::jit::Tracer::get_instance().is_tracing()) {
        throw std::runtime_error(
            "FakeQuantize::forward: cannot be JIT-traced while actively "
            "calibrating (observer enabled, non-learnable, training mode) "
            "— scale/zero_point are recomputed from observed statistics "
            "and baked into the compiled graph as constants, so QAT "
            "calibration would silently freeze at the trace-time snapshot "
            "instead of progressively refining. Call outside "
            "jit.compile()/jit.trace() during calibration, or switch to "
            "the learnable (LSQ) path once calibration is complete.");
    }

    // Update observer if enabled (even in eval mode for calibration)
    if (observer_enabled_) {
        observer_->observe(input.tensor());

        // Update qparams if observer has data and in training mode
        if (observer_->has_data() && !learnable_ && training_) {
            calculate_qparams();
        }
    }

    // Quantized value range (shared by the learnable LSQ path and the
    // non-learnable STE paths).
    float quant_min = 0.0f, quant_max = 0.0f;
    switch (dtype_) {
        case QuantDType::INT8:  quant_min = -128.0f; quant_max = 127.0f; break;
        case QuantDType::UINT8: quant_min =    0.0f; quant_max = 255.0f; break;
        case QuantDType::INT4:  quant_min =   -8.0f; quant_max =   7.0f; break;
        case QuantDType::UINT4: quant_min =    0.0f; quant_max =  15.0f; break;
    }

    // Match quantize_tensor's symmetric range: INT8 symmetric uses [-127,127]
    // and INT4 symmetric uses [-7,7] (compute_symmetric_scale divides by the
    // positive max). Without this, the STE could emit -128/-8, a level the
    // real quantized inference path clamps away — a train/inference mismatch.
    const bool symmetric = (scheme_ == QuantizationScheme::PerTensorSymmetric ||
                            scheme_ == QuantizationScheme::PerChannelSymmetric);
    if (symmetric && dtype_ == QuantDType::INT8) {
        quant_min = -127.0f;
    } else if (symmetric && dtype_ == QuantDType::INT4) {
        quant_min = -7.0f;
    }

    // Choose the STE branch from the ACTUAL qparams, not just scheme_. The
    // scheme and the qparams' channel axis are tracked independently and can
    // disagree (e.g. a per-tensor-scheme FakeQuantize that later receives a
    // length-C per-channel scale via set_qparams()/calculate_qparams()).
    // Keying solely on scheme_ would feed a per-channel scale vector into the
    // scalar path and silently quantize every channel with channel 0's
    // scale. Use the per-tensor scalar path only when the params are truly
    // scalar (one scale element); otherwise use the per-channel path along
    // the qparams' recorded axis (falling back to the module's axis_).
    const bool params_per_channel =
        (qparams_->scale.numel() > 1) || (qparams_->axis >= 0);
    // Per-channel params are broadcast along the qparams' channel axis
    // (fall back to the module's axis_ if unset).
    const int64_t channel_axis = (qparams_->axis >= 0) ? qparams_->axis : axis_;

    // LSQ (Esser et al. 2020, "Learned Step Size Quantization"): when this
    // module is learnable, the step size `scale` must receive a gradient so the
    // optimizer actually updates it — the plain STE paths below only
    // differentiate the activation and leave scale/zero_point frozen. Thread the
    // REGISTERED `scale` parameter Variable through a dedicated autograd Function
    // so its gradient accumulates into the leaf the optimizer holds. Read the
    // live value from the registered param (the optimizer rebinds the param
    // tensor each step, so qparams_->scale would be stale). Reachable even when
    // `input` itself does not require grad (e.g. a QuantStub on the raw network
    // input): the scale leaf still requires grad, so the loss depends on it and
    // it must receive the LSQ gradient. This is the QuantStub/QATHelper
    // (learnable=true) QAT path.
    if (learnable_) {
        if (auto scale_param = get_parameter("scale");
            scale_param && scale_param->requires_grad()) {
            auto zp_param = get_parameter("zero_point");
            Tensor zp_const =
                zp_param ? zp_param->tensor() : qparams_->zero_point;
            return lsq_fake_quantize(input, scale_param, zp_const,
                                     quant_min, quant_max,
                                     params_per_channel, channel_axis);
        }
    }

    // Both per-tensor and per-channel schemes use the autograd-enabled
    // straight-through estimator so QAT gradients thread through the
    // quantize→dequantize correctly. (Previously per-channel fell through to a
    // bare Tensor path that severed the graph and silently zeroed weight
    // gradients — QATHelper defaults weights to per-channel, so that was the
    // common QAT case.)
    if (input.requires_grad()) {
        if (!params_per_channel) {
            Tensor scale_cpu = (qparams_->scale.device() == Device::cpu())
                ? qparams_->scale : qparams_->scale.to(Device::cpu());
            Tensor zp_cpu = (qparams_->zero_point.device() == Device::cpu())
                ? qparams_->zero_point : qparams_->zero_point.to(Device::cpu());
            float scale = scale_cpu.data<float>()[0];
            // zero_point's dtype is not fixed by the API (callers may supply
            // Int64, Int32, or other int widths); read the scalar according to
            // its actual dtype instead of assuming Int32 (which throws
            // DTypeException for an Int64 zero_point).
            float zero_point = [&] () -> float {
                switch (zp_cpu.dtype()) {
                    case DType::Int32:    return static_cast<float>(zp_cpu.data<int32_t>()[0]);
                    case DType::Int64:    return static_cast<float>(zp_cpu.data<int64_t>()[0]);
                    case DType::Int16:    return static_cast<float>(zp_cpu.data<int16_t>()[0]);
                    case DType::Int8:     return static_cast<float>(zp_cpu.data<int8_t>()[0]);
                    case DType::UInt8:    return static_cast<float>(zp_cpu.data<uint8_t>()[0]);
                    case DType::Float32:  return zp_cpu.data<float>()[0];
                    default:
                        throw std::runtime_error(
                            "FakeQuantize::forward_impl: unsupported zero_point dtype " +
                            std::to_string(static_cast<int>(zp_cpu.dtype())));
                }
            }();
            return fake_quantize_with_grad(input, scale, zero_point, quant_min, quant_max);
        }

        // Per-channel STE: scale/zero_point are per-channel tensors broadcast
        // along the qparams' channel axis.
        return fake_quantize_per_channel_with_grad(
            input, qparams_->scale, qparams_->zero_point,
            quant_min, quant_max, channel_axis);
    }

    // JIT-R066: apply_fake_quantization() -> quantize_tensor()/
    // dequantize_tensor() is a raw host-pointer computation with zero
    // dispatch() calls (D2H, hand-rolled loop, H2D). Regardless of whether
    // qparams_ is actively changing (the narrower JIT-R065 guard above),
    // ANY use of this path is invisible to the tracer's tensor-lineage map
    // — the result would be frozen as a trace-time constant, silently
    // discarding the real (replayed) input activation on every later call.
    if (::tenzor::jit::Tracer::get_instance().is_tracing()) {
        throw std::runtime_error(
            "FakeQuantize::forward: cannot be JIT-traced on the no-grad "
            "fallback path (see JIT-R066 — apply_fake_quantization's core "
            "(de)quantize math bypasses dispatch/tracing entirely, so the "
            "output would be frozen as a trace-time constant). Call "
            "outside jit.compile()/jit.trace(), or ensure the input "
            "requires_grad so the differentiable STE path is used instead.");
    }

    // Fall-through: non-autograd path (eval calibration or
    // input.requires_grad() == false).
    Tensor quantized = apply_fake_quantization(input.tensor());
    return Variable(quantized, input.requires_grad());
}

auto FakeQuantize::apply_fake_quantization(const Tensor& input) const -> Tensor {
    // Quantize then immediately dequantize
    QuantizedTensor q_tensor = quantize_tensor(input, *qparams_);
    Tensor output = dequantize_tensor(q_tensor);
    // Preserve original dtype
    if (output.dtype() != input.dtype()) {
        output = output.to(input.dtype());
    }
    return output;
}

auto FakeQuantize::set_qparams(const QuantizationParams& params) -> void {
    qparams_ = std::make_unique<QuantizationParams>(params);

    if (learnable_) {
        // Update registered parameters
        Variable scale_var(qparams_->scale, true);
        register_parameter("scale", scale_var);
    }
}

auto FakeQuantize::calculate_qparams() -> void {
    if (!observer_->has_data()) {
        throw std::runtime_error("Observer has no data to calculate qparams");
    }

    *qparams_ = observer_->calculate_qparams(dtype_, scheme_);

    if (learnable_) {
        // Update registered parameters
        Variable scale_var(qparams_->scale, true);
        register_parameter("scale", scale_var);
    }
}

auto FakeQuantize::reset_observer() -> void {
    observer_->reset();
}

// ============================================================================
// LearnableFakeQuantize
// ============================================================================

LearnableFakeQuantize::LearnableFakeQuantize(
    QuantDType dtype,
    QuantizationScheme scheme,
    int64_t axis
) : FakeQuantize(dtype, scheme, true, true, axis) {}

auto LearnableFakeQuantize::init_from_observer() -> void {
    if (!observer()->has_data()) {
        throw std::runtime_error("Observer has no data for initialization");
    }

    calculate_qparams();
}

// ============================================================================
// Functional Interface
// ============================================================================





// ============================================================================
// QATHelper
// ============================================================================

auto QATHelper::prepare_qat(
    Module& model,
    QuantDType dtype,
    QuantizationScheme scheme,
    bool learnable
) -> void {
    // Clear any previously tracked modules
    fake_quant_modules_.clear();

    // Traverse the model's named parameters to find quantizable layers
    // For each quantizable layer, create and track a FakeQuantize module
    auto params = model.named_parameters();

    // Create FakeQuantize modules for weights and activations
    // Weight observer uses per-channel symmetric for better accuracy
    // Activation observer uses per-tensor symmetric for simplicity

    // Create activation fake quantize
    auto activation_fq = std::make_shared<FakeQuantize>(
        dtype, scheme, learnable, true, -1
    );
    fake_quant_modules_.push_back(activation_fq);

    // Create weight fake quantize (per-channel on output channel axis)
    auto weight_scheme = (scheme == QuantizationScheme::PerTensorSymmetric)
        ? QuantizationScheme::PerChannelSymmetric
        : scheme;
    auto weight_fq = std::make_shared<FakeQuantize>(
        dtype, weight_scheme, learnable, true, 0
    );
    fake_quant_modules_.push_back(weight_fq);

    // If model is Sequential, we can insert FakeQuantize modules
    // For generic modules, we just track the fake quant modules
    // The caller is responsible for applying them during forward pass
}

auto QATHelper::enable_observer() -> void {
    for (auto& fq : fake_quant_modules_) {
        fq->enable_observer(true);
    }
}

auto QATHelper::disable_observer() -> void {
    for (auto& fq : fake_quant_modules_) {
        fq->disable_observer();
    }
}

auto QATHelper::freeze_bn_stats() -> void {
    // Freeze batch norm statistics by disabling observer updates
    // and calculating final quantization parameters
    disable_observer();

    // Calculate and freeze quantization parameters from collected statistics
    for (auto& fq : fake_quant_modules_) {
        if (fq->observer() && fq->observer()->has_data()) {
            fq->calculate_qparams();
        }
    }
}

auto QATHelper::convert_to_quantized(Module& model) -> std::shared_ptr<Module> {
    // Convert fake quantization to actual quantized operations
    // This replaces FakeQuantize modules with actual quantized layers
    // and converts floating-point layers to quantized variants

    // Build a QConfig from the FakeQuantize modules' parameters
    auto qconfig = DefaultQConfigs::default_qconfig();

    // If model is a Sequential, convert each layer
    // Skip FakeQuantize modules (they were only for training)
    auto* seq = dynamic_cast<Sequential*>(&model);
    if (!seq) {
        // For non-Sequential models, try direct conversion
        // Check if it's a quantizable layer type
        if (auto* linear = dynamic_cast<Linear*>(&model)) {
            return QuantizedLinear::from_float(*linear, qconfig);
        }
        if (auto* conv = dynamic_cast<Conv2d*>(&model)) {
            return QuantizedConv2d::from_float(*conv, qconfig);
        }
        // Non-quantizable module, return a clone via shared_ptr
        // (caller should handle this case)
        return std::make_shared<Sequential>();
    }

    auto quantized_seq = std::make_shared<Sequential>();

    for (const auto& module : seq->modules()) {
        // Skip FakeQuantize modules - they were only for training simulation
        if (std::dynamic_pointer_cast<FakeQuantize>(module)) {
            continue;
        }

        // Try to convert quantizable layers
        if (auto linear = std::dynamic_pointer_cast<Linear>(module)) {
            auto q_linear = QuantizedLinear::from_float(*linear, qconfig);
            quantized_seq->add_module(q_linear);
        } else if (auto conv = std::dynamic_pointer_cast<Conv2d>(module)) {
            auto q_conv = QuantizedConv2d::from_float(*conv, qconfig);
            quantized_seq->add_module(q_conv);
        } else {
            // Keep non-quantizable modules as-is (ReLU, MaxPool, etc.)
            quantized_seq->add_module(module);
        }
    }

    return quantized_seq;
}

// ============================================================================
// FakeQuantizeFunction (Autograd)
// ============================================================================

namespace {
// Reshape a per-channel parameter (1-D, one entry per channel) so it
// broadcasts against a rank-`ndim` tensor along `axis`, converting to the
// target dtype/device. A scalar (numel == 1) parameter broadcasts across all
// channels (used before the observer has produced per-channel scales).
Tensor broadcast_channel_param(const Tensor& p, int64_t ndim, int64_t axis,
                               int64_t channels, DType dtype, const Device& device) {
    Tensor q = p;
    if (q.device() != device) q = q.to(device);
    if (q.dtype() != dtype) q = q.to(dtype);
    std::vector<int64_t> view_shape(static_cast<size_t>(std::max<int64_t>(ndim, 1)), 1);
    if (ndim > 0 && q.numel() == channels) {
        view_shape[static_cast<size_t>(axis)] = channels;
    }
    return q.reshape(view_shape);
}

// Round-half-to-even (banker's rounding), matching std::nearbyint under the
// default FE_TONEAREST mode used by the REAL quantizer in quantize.cpp. QAT
// fake-quant MUST round identically to deployment so training and inference
// agree at .5 boundaries; tenzor::round() is half-away-from-zero and would
// diverge by one quantization step on exact ties. Both fake-quant forwards
// (FakeQuantizeFunction and LSQFakeQuantizeFunction) route through this helper.
//
// Base = floor(x + 0.5) (half-up). This overshoots by exactly 1 only when x is
// an exact half-integer whose rounded value is odd; subtracting 1 there lands
// on the even neighbour (verified for both signs, e.g. 0.5->0, 2.5->2,
// -1.5->-2, -2.5->-2).
Tensor round_half_to_even(const Tensor& x) {
    Tensor r = tenzor::floor(tenzor::add(x, 0.5));
    // Tie mask: exactly halfway means (r - x) == 0.5.
    Tensor is_tie = tenzor::eq(tenzor::sub(r, x),
                               full({1}, 0.5, x.dtype(), x.device()));
    // Odd mask: r is odd iff r - 2*floor(r/2) != 0.
    Tensor r_is_odd = tenzor::ne(
        tenzor::sub(r, tenzor::mul(tenzor::floor(tenzor::mul(r, 0.5)), 2.0)),
        full({1}, 0.0, x.dtype(), x.device()));
    // On an odd tie, step down by one to reach the even neighbour.
    Tensor even_on_tie = tenzor::where(r_is_odd, tenzor::sub(r, 1.0), r);
    return tenzor::where(is_tie, even_on_tie, r);
}
}  // namespace

FakeQuantizeFunction::FakeQuantizeFunction(float scale, float zero_point,
                                           float quant_min, float quant_max)
    : scale_(scale), zero_point_(zero_point),
      quant_min_(quant_min), quant_max_(quant_max) {}

FakeQuantizeFunction::FakeQuantizeFunction(Tensor scale, Tensor zero_point,
                                           float quant_min, float quant_max,
                                           int64_t axis)
    : scale_(1.0f), zero_point_(0.0f),
      quant_min_(quant_min), quant_max_(quant_max),
      per_channel_(true), scale_t_(std::move(scale)),
      zero_point_t_(std::move(zero_point)), axis_(axis) {}

auto FakeQuantizeFunction::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    auto& input = inputs[0];
    auto x = input.tensor();

    // Save input for backward (STE needs to know which values were in-range)
    save_for_backward({x});

    // Fake quantize: quantize then immediately dequantize
    //   scaled  = x / scale + zero_point
    //   clamped = clamp(round(scaled), quant_min, quant_max)
    //   output  = (clamped - zero_point) * scale
    // scale/zero_point are scalars per-tensor, or per-channel tensors broadcast
    // along axis_ in the per-channel case.
    Tensor scaled, zp_use, scale_use;
    if (per_channel_) {
        int64_t ndim = x.ndim();
        int64_t axis = axis_ < 0 ? axis_ + ndim : axis_;
        int64_t channels = ndim > 0 ? x.shape()[static_cast<size_t>(axis)] : 1;
        scale_use = broadcast_channel_param(scale_t_, ndim, axis, channels, x.dtype(), x.device());
        zp_use = broadcast_channel_param(zero_point_t_, ndim, axis, channels, x.dtype(), x.device());
        scaled = x / scale_use + zp_use;
    } else {
        Tensor inv_scale = full({1}, 1.0f / scale_, x.dtype(), x.device());
        zp_use = full({1}, zero_point_, x.dtype(), x.device());
        scale_use = full({1}, scale_, x.dtype(), x.device());
        scaled = x * inv_scale + zp_use;
    }

    Tensor rounded = round_half_to_even(scaled);
    Tensor clamped = tenzor::clamp(rounded, quant_min_, quant_max_);
    Tensor output = (clamped - zp_use) * scale_use;

    return {Variable(output, input.requires_grad())};
}

auto FakeQuantizeFunction::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    auto& grad_output = grad_outputs[0];
    auto& input = saved_tensors()[0];

    // STE: pass gradients through for values within the quantizable range,
    // zero gradients for values that would be clamped by quantization.
    // The quantizable range in float space is:
    //   [(quant_min - zero_point) * scale, (quant_max - zero_point) * scale]
    // For per-channel quant the bounds are per-channel tensors broadcast
    // along axis_; otherwise they are scalars.
    // Build broadcast scale/zero_point tensors for both per-channel and
    // per-tensor cases so the mask can be computed on the ROUNDED integer level.
    Tensor scale_bc, zp_bc;
    if (per_channel_) {
        int64_t ndim = input.ndim();
        int64_t axis = axis_ < 0 ? axis_ + ndim : axis_;
        int64_t channels = ndim > 0 ? input.shape()[static_cast<size_t>(axis)] : 1;
        scale_bc = broadcast_channel_param(scale_t_, ndim, axis, channels, input.dtype(), input.device());
        zp_bc = broadcast_channel_param(zero_point_t_, ndim, axis, channels, input.dtype(), input.device());
    } else {
        scale_bc = full({1}, scale_, input.dtype(), input.device());
        zp_bc = full({1}, static_cast<float>(zero_point_), input.dtype(), input.device());
    }
    Tensor qmin_t = full({1}, static_cast<float>(quant_min_), input.dtype(), input.device());
    Tensor qmax_t = full({1}, static_cast<float>(quant_max_), input.dtype(), input.device());

    // STE mask on the ROUNDED integer level q = round(x/scale + zp), not on the
    // float range (qmin..qmax)*scale. A value whose x/scale+zp lands in
    // (qmax, qmax+0.5] rounds to qmax — genuinely in range, not clamped — so its
    // gradient must pass. Masking on the float range zeroes it, narrowing the
    // pass-through region by ~half a step. This matches the LSQ backward, which
    // masks on round(v).
    Tensor q = round_half_to_even(input / scale_bc + zp_bc);
    Tensor mask = ge(q, qmin_t) * le(q, qmax_t);

    // Zero out gradients for out-of-range values
    auto shape = grad_output.shape();
    Tensor zero_grad = zeros(std::vector<int64_t>(shape.begin(), shape.end()),
                              grad_output.dtype(), grad_output.device());
    return {tenzor::where(mask, grad_output, zero_grad)};
}

auto fake_quantize_with_grad(
    const Variable& input,
    float scale,
    float zero_point,
    float quant_min,
    float quant_max
) -> Variable {
    auto fn = std::make_shared<FakeQuantizeFunction>(scale, zero_point, quant_min, quant_max);
    auto outputs = fn->forward({input});
    if (input.requires_grad()) {
        // Wire the autograd graph edges so the engine can route the STE
        // gradient back to `input`. Setting grad_fn alone is insufficient:
        // without next_functions + input_variables the backward engine has
        // no input edge to accumulate into, so input.grad() stays empty.
        // (Mirrors the standard pattern in nested_autograd_ops.cpp.)
        fn->set_next_functions({input.grad_fn()});
        fn->set_input_variables({input});
        outputs[0].set_grad_fn(fn);
    }
    return outputs[0];
}

auto fake_quantize_per_channel_with_grad(
    const Variable& input,
    const Tensor& scale,
    const Tensor& zero_point,
    float quant_min,
    float quant_max,
    int64_t axis
) -> Variable {
    auto fn = std::make_shared<FakeQuantizeFunction>(scale, zero_point, quant_min, quant_max, axis);
    auto outputs = fn->forward({input});
    if (input.requires_grad()) {
        // Same STE graph wiring as the per-tensor path: attach next_functions
        // + input_variables so the engine routes the per-channel STE gradient
        // back into `input`.
        fn->set_next_functions({input.grad_fn()});
        fn->set_input_variables({input});
        outputs[0].set_grad_fn(fn);
    }
    return outputs[0];
}

// ============================================================================
// LSQ (Learned Step Size Quantization) Function
// ============================================================================

LSQFakeQuantizeFunction::LSQFakeQuantizeFunction(Tensor zero_point,
                                                 float quant_min, float quant_max,
                                                 bool per_channel, int64_t axis)
    : quant_min_(quant_min), quant_max_(quant_max),
      per_channel_(per_channel), axis_(axis),
      zero_point_(std::move(zero_point)) {}

namespace {
// Reduce a per-element scale-gradient contribution down to the scale parameter
// shape: a scalar {1} for per-tensor, or {C} along `axis` for per-channel
// (sum over every other dimension).
Tensor reduce_to_scale(const Tensor& g_in, int64_t numel_scale, bool per_channel,
                       int64_t axis, DType out_dtype) {
    // The scale-gradient sum is the ENTIRE gradient of the learned step size, so
    // widen the per-element contribution to at least Float32 (Float64 for Float64
    // scale params) BEFORE the sum reduction. Reducing in a half dtype (F16/BF16)
    // accumulates the whole sum in half and loses precision / overflows.
    DType compute_dtype = (out_dtype == DType::Float64) ? DType::Float64 : DType::Float32;
    Tensor g = (g_in.dtype() == compute_dtype) ? g_in : g_in.to(compute_dtype);
    Tensor r;
    if (!per_channel || numel_scale <= 1) {
        r = tenzor::sum(g).reshape({1});
    } else {
        int64_t nd = g.ndim();
        int64_t ax = axis < 0 ? axis + nd : axis;
        Tensor moved = (ax == 0) ? g : g.transpose(0, ax);
        moved = moved.contiguous();
        int64_t rest = moved.numel() / numel_scale;
        moved = moved.reshape({numel_scale, rest});
        r = tenzor::sum(moved, /*dim=*/1, /*keepdim=*/false);  // -> {C}
    }
    if (r.dtype() != out_dtype) r = r.to(out_dtype);
    return r;
}

// Broadcast the scale and zero-point to x's shape for the forward/backward math.
void lsq_broadcast(const Tensor& s, const Tensor& z, const Tensor& x,
                   bool per_channel, int64_t axis_in,
                   Tensor& s_bc, Tensor& z_bc) {
    if (per_channel) {
        int64_t ndim = x.ndim();
        int64_t axis = axis_in < 0 ? axis_in + ndim : axis_in;
        int64_t channels = ndim > 0 ? x.shape()[static_cast<size_t>(axis)] : 1;
        s_bc = broadcast_channel_param(s, ndim, axis, channels, x.dtype(), x.device());
        z_bc = broadcast_channel_param(z, ndim, axis, channels, x.dtype(), x.device());
    } else {
        s_bc = s;
        if (s_bc.device() != x.device()) s_bc = s_bc.to(x.device());
        if (s_bc.dtype() != x.dtype()) s_bc = s_bc.to(x.dtype());
        z_bc = z;
        if (z_bc.device() != x.device()) z_bc = z_bc.to(x.device());
        if (z_bc.dtype() != x.dtype()) z_bc = z_bc.to(x.dtype());
    }
}
}  // namespace

auto LSQFakeQuantizeFunction::forward(std::vector<Variable> inputs)
    -> std::vector<Variable> {
    auto x = inputs[0].tensor();
    auto s = inputs[1].tensor();

    Tensor s_bc, z_bc;
    lsq_broadcast(s, zero_point_, x, per_channel_, axis_, s_bc, z_bc);

    // Fake quantize: x_hat = (clamp(round(x/s + z), qmin, qmax) - z) * s
    Tensor v = x / s_bc + z_bc;
    Tensor q = tenzor::clamp(round_half_to_even(v), quant_min_, quant_max_);
    Tensor output = (q - z_bc) * s_bc;

    // Save raw x and s; z is a constant member captured at construction.
    save_for_backward({x, s});

    bool req = inputs[0].requires_grad() || inputs[1].requires_grad();
    return {Variable(output, req)};
}

auto LSQFakeQuantizeFunction::backward(std::vector<Tensor> grad_outputs)
    -> std::vector<Tensor> {
    auto g = grad_outputs[0];
    auto x = saved_tensors()[0];
    auto s = saved_tensors()[1];

    Tensor s_bc, z_bc;
    lsq_broadcast(s, zero_point_, x, per_channel_, axis_, s_bc, z_bc);

    Tensor v = x / s_bc + z_bc;           // v = x/s + z
    // Use the same rounding as the forward (round-half-to-even); tenzor::round
    // is half-away-from-zero and would disagree with the forward-emitted level
    // on exact .5 ties, misclassifying the region and the (r - v) scale grad.
    Tensor r = round_half_to_even(v);     // pre-clamp integer level

    Tensor qmin_t = full({1}, static_cast<double>(quant_min_), x.dtype(), x.device());
    Tensor qmax_t = full({1}, static_cast<double>(quant_max_), x.dtype(), x.device());

    // Region masks from the pre-clamp level (cast to float so the arithmetic
    // below stays in x's dtype regardless of the comparison op's result dtype).
    Tensor small  = lt(r, qmin_t).to(x.dtype());                  // r < qmin  (clamp low)
    Tensor big    = gt(r, qmax_t).to(x.dtype());                  // r > qmax  (clamp high)
    Tensor middle = (ge(r, qmin_t) * le(r, qmax_t)).to(x.dtype()); // in range

    // grad wrt input: straight-through estimator — identity in range, 0 clamped.
    Tensor grad_input = g * middle;

    // grad wrt scale (LSQ), per element:
    //   in range : d/ds [ s*round(x/s+z) - s*z ] = round(v) - v   ( = round(x/s+z) - (x/s+z) )
    //   clamp low : d/ds [ (qmin - z)*s ]        = qmin - z
    //   clamp high: d/ds [ (qmax - z)*s ]        = qmax - z
    Tensor ds = middle * (r - v)
              + small  * (qmin_t - z_bc)
              + big    * (qmax_t - z_bc);
    Tensor grad_scale =
        reduce_to_scale(g * ds, s.numel(), per_channel_, axis_, s.dtype());

    // LSQ gradient-scale normalization (Esser et al. 2020): scale dL/ds by
    //   g = 1 / sqrt(N * Q_p)
    // where N is the number of elements each step size is applied to (total for
    // per-tensor, per-channel element count for per-channel) and Q_p = quant_max.
    // Without this the effective LR on the step size is ~sqrt(N*Q_p) too large
    // vs the weight LR, changing the joint weight+step QAT trajectory.
    int64_t numel_scale = s.numel();
    int64_t n_per_scale = (numel_scale > 0) ? (x.numel() / numel_scale) : x.numel();
    if (n_per_scale > 0 && quant_max_ > 0.0f) {
        double grad_scale_factor =
            1.0 / std::sqrt(static_cast<double>(n_per_scale) *
                            static_cast<double>(quant_max_));
        grad_scale = grad_scale * grad_scale_factor;
    }

    return {grad_input, grad_scale};
}

auto lsq_fake_quantize(
    const Variable& input,
    const std::shared_ptr<Variable>& scale,
    const Tensor& zero_point,
    float quant_min,
    float quant_max,
    bool per_channel,
    int64_t axis
) -> Variable {
    auto fn = std::make_shared<LSQFakeQuantizeFunction>(
        zero_point, quant_min, quant_max, per_channel, axis);
    auto outputs = fn->forward({input, *scale});
    // Wire graph edges for BOTH differentiable inputs so the engine routes the
    // STE gradient back to `input` and the LSQ step-size gradient into the
    // registered `scale` leaf. grad_fn() is null for a leaf parameter; the
    // engine then accumulates into the Variable identified via input_variables.
    fn->set_next_functions({input.grad_fn(), scale->grad_fn()});
    fn->set_input_variables({input, *scale});
    outputs[0].set_grad_fn(fn);
    return outputs[0];
}

// ============================================================================
// BN Folding
// ============================================================================

auto fold_bn(Module& model) -> void {
    // Look for Conv2d -> BatchNorm2d patterns in Sequential containers
    auto* seq = dynamic_cast<Sequential*>(&model);
    if (!seq) {
        // For non-Sequential models, we cannot iterate children generically.
        // Only Sequential models are supported for automatic BN folding.
        return;
    }

    auto& modules = seq->modules();
    if (modules.size() < 2) return;

    // Identify Conv2d -> BatchNorm2d pairs and fold BN into Conv2d weights.
    // We mark BN indices as "folded" so we can skip them when rebuilding.
    std::vector<bool> skip(modules.size(), false);

    for (size_t i = 0; i + 1 < modules.size(); ++i) {
        if (skip[i]) continue;

        auto conv = std::dynamic_pointer_cast<Conv2d>(modules[i]);
        auto* bn_module = dynamic_cast<BatchNorm2d*>(modules[i + 1].get());
        if (!conv || !bn_module) continue;

        // Extract BN parameters
        auto bn_state = bn_module->state_dict();
        Tensor gamma = bn_state.at("weight");
        Tensor beta = bn_state.at("bias");
        Tensor running_mean = bn_state.at("running_mean");
        Tensor running_var = bn_state.at("running_var");
        double eps = bn_module->eps();

        // Compute folding factors: bn_scale = gamma / sqrt(var + eps)
        Tensor sqrt_var = sqrt(running_var + eps);
        Tensor bn_scale = gamma / sqrt_var;  // [C]

        // Get conv weight and bias
        auto conv_state = conv->state_dict();
        Tensor fp_weight = conv_state.at("weight");  // [out_ch, in_ch/groups, kH, kW]

        auto weight_shape = fp_weight.shape();
        int64_t out_channels = weight_shape[0];

        // Preserve the conv's original dtype. The host-side folding math runs in
        // Float32, but the folded weight/bias must be cast back so a Float16 /
        // BFloat16 conv keeps its dtype (Tensor::data<float>() throws on non-F32,
        // so every parameter is widened to Float32 on CPU before raw access).
        DType conv_dtype = fp_weight.dtype();

        // Fold BN scale into conv weights: w_new[c] = bn_scale[c] * w_old[c]
        auto fw_cpu = (fp_weight.device() == Device::cpu()) ? fp_weight.clone()
                                                            : fp_weight.to(Device::cpu());
        if (fw_cpu.dtype() != DType::Float32) fw_cpu = fw_cpu.to(DType::Float32);
        auto bs_cpu = (bn_scale.device() == Device::cpu()) ? bn_scale : bn_scale.to(Device::cpu());
        if (bs_cpu.dtype() != DType::Float32) bs_cpu = bs_cpu.to(DType::Float32);

        float* w_data = fw_cpu.data<float>();
        const float* s_data = bs_cpu.data<const float>();
        int64_t channel_size = fw_cpu.numel() / out_channels;

        for (int64_t c = 0; c < out_channels; ++c) {
            float s = s_data[c];
            for (int64_t j = 0; j < channel_size; ++j) {
                w_data[c * channel_size + j] *= s;
            }
        }

        // Fold BN into bias: b_new = bn_scale * (b_old - mean) + beta
        std::optional<Tensor> conv_bias;
        if (conv_state.find("bias") != conv_state.end()) {
            conv_bias = conv_state.at("bias");
        }

        Tensor folded_bias({out_channels}, DType::Float32, Device::cpu());
        float* fb_data = folded_bias.data<float>();

        auto rm_cpu = (running_mean.device() == Device::cpu()) ? running_mean : running_mean.to(Device::cpu());
        if (rm_cpu.dtype() != DType::Float32) rm_cpu = rm_cpu.to(DType::Float32);
        auto bt_cpu = (beta.device() == Device::cpu()) ? beta : beta.to(Device::cpu());
        if (bt_cpu.dtype() != DType::Float32) bt_cpu = bt_cpu.to(DType::Float32);
        const float* mean_data = rm_cpu.data<const float>();
        const float* beta_data = bt_cpu.data<const float>();

        for (int64_t c = 0; c < out_channels; ++c) {
            float b_old = 0.0f;
            if (conv_bias.has_value()) {
                auto cb_cpu = (*conv_bias);
                if (cb_cpu.device() != Device::cpu()) cb_cpu = cb_cpu.to(Device::cpu());
                if (cb_cpu.dtype() != DType::Float32) cb_cpu = cb_cpu.to(DType::Float32);
                b_old = cb_cpu.data<const float>()[c];
            }
            fb_data[c] = s_data[c] * (b_old - mean_data[c]) + beta_data[c];
        }

        // Cast the folded parameters back to the conv's original dtype so a
        // non-Float32 conv keeps its dtype after folding.
        Tensor folded_weight_out =
            (conv_dtype == DType::Float32) ? fw_cpu : fw_cpu.to(conv_dtype);
        Tensor folded_bias_out =
            (conv_dtype == DType::Float32) ? folded_bias : folded_bias.to(conv_dtype);

        // Folding BN always produces a bias term. If the source conv was built
        // with bias=false it has no "bias" parameter registered, so a strict
        // load_state_dict would reject "bias" as an unexpected key. Register the
        // bias parameter first so the folded conv can carry it.
        if (conv_state.find("bias") == conv_state.end()) {
            conv->register_parameter_shared(
                "bias", std::make_shared<Variable>(folded_bias_out, /*requires_grad=*/true));
        }

        // Update conv with folded parameters via state_dict
        std::unordered_map<std::string, Tensor> new_state;
        new_state["weight"] = folded_weight_out;
        new_state["bias"] = folded_bias_out;
        conv->load_state_dict(new_state);

        // Mark BN module for skipping in rebuild
        skip[i + 1] = true;
    }

    // Reset the folded BN modules to a true identity transform.
    bool any_folded = false;
    for (bool s : skip) { if (s) { any_folded = true; break; } }
    if (!any_folded) return;

    // We can't swap the internal modules_ vector directly (it's private), and the
    // Conv2d weights are already updated in-place above, so the folding is effective
    // even without removing the BN modules. The BN modules remain but must become a
    // genuine identity in eval mode. Eval BN computes y = (x - mean) / sqrt(var + eps),
    // so to get y == x we need mean=0, weight=1, bias=0, and sqrt(var + eps) == 1,
    // i.e. running_var = 1 - eps (NOT 1, which would scale by 1/sqrt(1 + eps)).
    for (size_t i = 0; i < modules.size(); ++i) {
        if (skip[i]) {
            auto* bn = dynamic_cast<BatchNorm2d*>(modules[i].get());
            if (bn) {
                auto bn_sd = bn->state_dict();
                int64_t num_features = bn_sd.at("weight").shape()[0];
                double eps = bn->eps();
                std::unordered_map<std::string, Tensor> identity_state;
                identity_state["weight"] = ones({num_features}, DType::Float32, Device::cpu());
                identity_state["bias"] = zeros({num_features}, DType::Float32, Device::cpu());
                identity_state["running_mean"] = zeros({num_features}, DType::Float32, Device::cpu());
                // running_var = 1 - eps so that sqrt(running_var + eps) == 1 exactly.
                identity_state["running_var"] = full(
                    {num_features}, static_cast<float>(1.0 - eps),
                    DType::Float32, Device::cpu());
                bn->load_state_dict(identity_state);
            }
        }
    }
}

} // namespace quantization
} // namespace nn
} // namespace tenzor
