/**
 * @file quantized_layers.cpp
 * @brief Implementation of quantized neural network layers
 */

#include "tenzor/nn/quantization/quantized_layers.hpp"
#include "tenzor/nn/quantization/fake_quantize.hpp"  // QAT STE path (B.1)
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/layers/batchnorm.hpp"
#include "tenzor/nn/layers/normalization.hpp"
#include "tenzor/nn/layers/embedding.hpp"
#include "tenzor/nn/layers/rnn.hpp"
#include "tenzor/nn/functional.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/autograd/ops.hpp"
#include "../../backends/cpu/kernels/fused_quantized_ops.hpp"
#include "tenzor/jit/tracer.hpp"
#include <stdexcept>

namespace tenzor {
namespace nn {

namespace {
/// Record a per-tensor (non-per-channel, non-INT4, INT8-activation)
/// `QuantizedLinearStatic` op into the active JIT trace, if any
/// (JIT-R058b). QuantizedLinear::forward_quantized's CPU path (a raw host
/// pointer loop, zero dispatch() calls) AND its GPU fast path (dispatches,
/// but the activation tensor it dispatches is the ALREADY-quantized int8
/// data — produced by quantize_tensor()'s own raw host pointer loop, so
/// disconnected from any traced lineage back to the float `input` Variable
/// regardless of device) are both invisible or lineage-broken for JIT
/// tracing on their own. `float_input_2d` is therefore the RAW FLOAT
/// activation (2D [batch, in_features], pre-quantization) — registering
/// THIS (not the int8 result) keeps the node's input correctly tied to the
/// traced parameter; execute_node re-quantizes the (possibly NEW, replayed)
/// float input using the fixed calibrated scale/zero_point before
/// dispatching the matmul, exactly mirroring what eager forward_impl does
/// on every call. Only called when activation_qparams_ is set (static/
/// calibrated quantization) — the dynamic (uncalibrated) path is refused at
/// trace time instead (see forward_impl's is_tracing() guard), since
/// quantize_per_tensor_symmetric recomputes scale/zero_point from the
/// input's live min/max via untraced raw-Tensor code (JIT-R066); baking
/// THAT as a fixed constant would be a silent wrong-numerics bug, not just
/// staleness. `weight`/`bias` are the frozen (correctly constant across
/// calls) int8 weight and FP32 bias; `bias` may be an invalid/empty Tensor.
inline auto jit_record_quantized_linear_static(
        const ::tenzor::Tensor& float_input_2d,
        const ::tenzor::Tensor& weight,
        const ::tenzor::Tensor& bias,
        const ::tenzor::Tensor& output,
        float input_scale, float weight_scale,
        int32_t input_zp, int32_t weight_zp,
        bool input_asymmetric) -> void {
    auto& tracer = ::tenzor::jit::Tracer::get_instance();
    if (!tracer.is_tracing()) return;
    auto in_id = tracer.register_tensor(float_input_2d);
    auto w_id = tracer.register_tensor(weight);
    std::vector<std::string> input_ids = {in_id, w_id};
    if (bias.is_valid() && bias.numel() > 0) {
        input_ids.push_back(tracer.register_tensor(bias));
    } else {
        // execute_node requires >= 3 inputs (input, weight, bias slot always
        // present, matching the GPU dispatch path's convention of an
        // always-occupied — possibly empty — bias slot at index 2).
        input_ids.push_back(tracer.register_tensor(
            ::tenzor::zeros({0}, DType::Float32, float_input_2d.device())));
    }
    auto out_id = tracer.register_new_tensor(output);
    ::tenzor::jit::TracedOp op(::tenzor::jit::OpType::QuantizedLinearStatic,
                               input_ids, {out_id});
    op.attrs["input_scale"] = static_cast<double>(input_scale);
    op.attrs["weight_scale"] = static_cast<double>(weight_scale);
    op.attrs["output_scale"] = 1.0;
    op.int_attrs["input_zero_point"] = static_cast<int64_t>(input_zp);
    op.int_attrs["weight_zero_point"] = static_cast<int64_t>(weight_zp);
    op.int_attrs["input_asymmetric"] = input_asymmetric ? 1 : 0;
    tracer.record_op(std::move(op));
}

/// Thin wrappers routing raw-Tensor view ops through the tracer-aware
/// Variable-level autograd functions (JIT-R058a). Tensor::slice()/squeeze()/
/// unsqueeze()/permute() are pure TensorImpl metadata tricks with ZERO
/// dispatch() calls, so they're invisible to both the JIT tracer's generic
/// dispatch<OpId> interception (which only sees ops that actually call
/// dispatch(), e.g. matmul/sigmoid/tanh/+/*/cat) AND to the tracer's own
/// tensor lineage map (register_tensor never sees an untraced view op's
/// output, so it gets treated as a brand-new, disconnected leaf and frozen
/// as a trace-time constant on replay). The Variable-level ::tenzor::slice/
/// squeeze/unsqueeze/permute wrappers explicitly call jit_record_shape_op()
/// when tracing is active, which these raw calls never would. Wrapping in
/// Variable(t, false) is safe here — every call site below is inference-
/// only (int8 RNN weights are not differentiable, matching this class's own
/// "Returning a Variable with requires_grad=false" contract) — and, when
/// `t` is itself an already-registered traced tensor (a function argument
/// like h0/c0/hx/cx, or the dispatched output of a prior op like `gates`),
/// register_tensor's fingerprint match (ptr+dtype+shape+strides) correctly
/// resolves to the SAME existing graph value rather than minting a new one.
inline auto traced_slice(const ::tenzor::Tensor& t, int64_t dim, int64_t start,
                          int64_t end, int64_t step = 1) -> ::tenzor::Tensor {
    return ::tenzor::slice(::tenzor::Variable(t, false), dim, start, end, step).tensor();
}
inline auto traced_squeeze(const ::tenzor::Tensor& t, int64_t dim) -> ::tenzor::Tensor {
    return ::tenzor::squeeze(::tenzor::Variable(t, false), dim).tensor();
}
inline auto traced_unsqueeze(const ::tenzor::Tensor& t, int64_t dim) -> ::tenzor::Tensor {
    return ::tenzor::unsqueeze(::tenzor::Variable(t, false), dim).tensor();
}
inline auto traced_permute(const ::tenzor::Tensor& t, std::vector<int64_t> dims) -> ::tenzor::Tensor {
    return ::tenzor::permute(::tenzor::Variable(t, false), dims).tensor();
}
}  // namespace
namespace quantization {

// Forward declare kernel functions
namespace kernels {
    auto quantized_linear_kernel(
        const int8_t* input, const int8_t* weight, const float* bias,
        float* output, int64_t batch_size, int64_t in_features, int64_t out_features,
        float input_scale, float weight_scale, float output_scale,
        int32_t input_zp, int32_t weight_zp
    ) -> void;

    auto quantized_conv2d_kernel(
        const int8_t* input, const int8_t* weight, const float* bias,
        float* output, int64_t batch, int64_t in_channels, int64_t out_channels,
        int64_t h_in, int64_t w_in, int64_t h_out, int64_t w_out,
        int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,
        int64_t pad_h, int64_t pad_w,
        float input_scale, float weight_scale, int32_t input_zp, int32_t weight_zp,
        int64_t dil_h = 1, int64_t dil_w = 1, int64_t groups = 1
    ) -> void;

    // Per-channel quantized variants
    auto quantized_linear_per_channel_kernel(
        const int8_t* input, const int8_t* weight, const float* bias,
        float* output, int64_t batch_size, int64_t in_features, int64_t out_features,
        float input_scale, const float* weight_scales, float output_scale,
        int32_t input_zp, const int32_t* weight_zps
    ) -> void;

    auto quantized_conv2d_per_channel_kernel(
        const int8_t* input, const int8_t* weight, const float* bias,
        float* output, int64_t batch, int64_t in_channels, int64_t out_channels,
        int64_t h_in, int64_t w_in, int64_t h_out, int64_t w_out,
        int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,
        int64_t pad_h, int64_t pad_w,
        float input_scale, const float* weight_scales, int32_t input_zp, const int32_t* weight_zps,
        int64_t dil_h = 1, int64_t dil_w = 1, int64_t groups = 1
    ) -> void;
}

// ============================================================================
// QuantizedLinear
// ============================================================================

QuantizedLinear::QuantizedLinear(
    int64_t in_features,
    int64_t out_features,
    QuantizationParams weight_qparams,
    float bias_scale
) : in_features_(in_features),
    out_features_(out_features),
    weight_(Tensor({out_features, in_features}, DType::Int8, Device::cpu()), weight_qparams),
    bias_scale_(bias_scale) {}

auto QuantizedLinear::forward_impl(const Variable& input) -> Variable {
    // QuantizedLinear is an inference-only int8 layer (analogous to
    // torch.ao.nn.quantized.Linear). The int8 matmul is not differentiable;
    // gradients cannot flow through. For QAT training, compose a regular
    // Linear with a FakeQuantize module, which uses the STE backward path.
    // Returning a Variable with requires_grad=false makes this honest.
    // Static quantization: quantize the input with the calibrated activation
    // scale/zero_point; otherwise fall back to dynamic per-call computation.
    if (!activation_qparams_.has_value() &&
        ::tenzor::jit::Tracer::get_instance().is_tracing()) {
        // JIT-R058b/R066: quantize_per_tensor_symmetric recomputes scale/
        // zero_point from the input's live min/max via raw-Tensor (non-
        // Variable) code that is invisible to the tracer regardless of any
        // dispatch-level fix. A cached graph would silently reuse the
        // trace-time scale/zero_point (and even the trace-time quantized
        // DATA) on every later replay call, regardless of the actual input
        // — a silent wrong-numerics bug, not just staleness. Fail loudly
        // and force an eager fallback instead (matches RoPE/KVCache/
        // WindowAttention's established "fail loudly instead of producing
        // wrong numerics" pattern). Calibrate via set_activation_qparams()
        // to make this layer JIT-traceable.
        throw std::runtime_error(
            "QuantizedLinear::forward_impl: dynamic (uncalibrated) "
            "activation quantization cannot be correctly JIT-traced (see "
            "JIT-R066 — the core (de)quantize math bypasses dispatch/"
            "tracing entirely). Call set_activation_qparams() with "
            "statically calibrated qparams first, or keep this call "
            "outside jit.compile()/jit.trace().");
    }
    auto q_input = activation_qparams_.has_value()
        ? quantize_tensor(input.tensor(), *activation_qparams_)
        : quantize_per_tensor_symmetric(input.tensor());
    Tensor output = forward_quantized(q_input);

    // JIT-R058b: record a trace node for the common per-tensor INT8/INT8
    // static-quant configuration (weight per-tensor INT8, activation
    // per-tensor INT8) — the case OpType::QuantizedLinearStatic's convention
    // matches exactly.
    if (activation_qparams_.has_value()) {
        const auto& wparams = weight_.params();
        bool weight_per_tensor =
            (wparams.scheme == QuantizationScheme::PerTensorSymmetric ||
             wparams.scheme == QuantizationScheme::PerTensorAsymmetric);
        bool weight_int8 = weight_.data().dtype() == DType::Int8;
        bool act_int8 = activation_qparams_->dtype == QuantDType::INT8;
        bool act_per_tensor =
            (activation_qparams_->scheme == QuantizationScheme::PerTensorSymmetric ||
             activation_qparams_->scheme == QuantizationScheme::PerTensorAsymmetric);
        // JIT-R172: per-channel weight quantization (the library's DEFAULT
        // preset — see DefaultQConfigs::default_qconfig() — and INT4)
        // previously fell through this guard with no jit_record call and no
        // error, so a trace silently froze `output` as a trace-time
        // constant: every later replay call with different input data
        // returned the first call's stale value, and this never tripped
        // cache_stats().eager_fallbacks since it isn't a fallback -- it's a
        // silently-wrong compiled node. QuantizedConv2d already refuses to
        // be traced at all for the equivalent reason (see its
        // forward_impl); apply the same fail-loudly discipline here rather
        // than let an untraceable configuration silently corrupt a compiled
        // graph's output.
        if (!(weight_per_tensor && weight_int8 && act_int8 && act_per_tensor) &&
            ::tenzor::jit::Tracer::get_instance().is_tracing()) {
            throw std::runtime_error(
                "QuantizedLinear::forward_impl: this weight/activation "
                "quantization configuration (per-channel weight and/or "
                "non-INT8 weight/activation dtype) cannot be correctly "
                "JIT-traced -- OpType::QuantizedLinearStatic's replay node "
                "only supports per-tensor INT8/INT8 static quantization. "
                "Tracing it anyway would silently freeze the output as a "
                "trace-time constant (JIT-R172). Use a per-tensor INT8 "
                "QConfig for layers that must be JIT-compiled, or keep "
                "this call outside jit.compile()/jit.trace().");
        }
        if (weight_per_tensor && weight_int8 && act_int8 && act_per_tensor) {
            int64_t rows = in_features_ > 0 ? input.tensor().numel() / in_features_ : 0;
            Tensor input_2d = (input.tensor().shape().size() == 2)
                ? input.tensor()
                : input.tensor().reshape({rows, in_features_});

            Tensor weight_scale_cpu = wparams.scale;
            if (weight_scale_cpu.device() != Device::cpu()) weight_scale_cpu = weight_scale_cpu.to(Device::cpu());
            Tensor weight_zp_cpu = wparams.zero_point;
            if (weight_zp_cpu.device() != Device::cpu()) weight_zp_cpu = weight_zp_cpu.to(Device::cpu());
            Tensor act_scale_cpu = activation_qparams_->scale;
            if (act_scale_cpu.device() != Device::cpu()) act_scale_cpu = act_scale_cpu.to(Device::cpu());
            Tensor act_zp_cpu = activation_qparams_->zero_point;
            if (act_zp_cpu.device() != Device::cpu()) act_zp_cpu = act_zp_cpu.to(Device::cpu());

            Tensor bias_for_trace;
            if (bias_.has_value()) {
                bias_for_trace = *bias_;
                if (bias_for_trace.dtype() != DType::Float32) {
                    bias_for_trace = bias_for_trace.to(DType::Float32);
                }
                // JIT-R103: also reconcile device — bias_ stays wherever the
                // module was constructed (CPU by default) until moved.
                if (bias_for_trace.device() != input.tensor().device()) {
                    bias_for_trace = bias_for_trace.to(input.tensor().device());
                }
            }

            // JIT-R103: weight_ likewise stays wherever the module was
            // constructed until moved; this trace-node's inputs must all
            // agree with `output`/`input_2d` (already on the input's real
            // device) or node registration silently graph-breaks.
            Tensor weight_for_trace = weight_.data();
            if (weight_for_trace.device() != input.tensor().device()) {
                weight_for_trace = weight_for_trace.to(input.tensor().device());
            }

            bool asymmetric = activation_qparams_->scheme == QuantizationScheme::PerTensorAsymmetric;
            jit_record_quantized_linear_static(
                input_2d, weight_for_trace, bias_for_trace, output,
                act_scale_cpu.data<const float>()[0],
                weight_scale_cpu.data<const float>()[0],
                act_zp_cpu.data<int32_t>()[0],
                weight_zp_cpu.data<int32_t>()[0],
                asymmetric);
        }
    }
    return Variable(output, /*requires_grad=*/false);
}

auto QuantizedLinear::forward_quantized(const QuantizedTensor& input) -> Tensor {
    auto input_shape = input.shape();
    // Support rank>=2 activations (e.g. transformer [B, S, in_features]) by
    // flattening all leading dims into the effective batch dimension. The
    // kernels iterate `batch_size` rows of `in_features` contiguous elements,
    // so rows = numel()/in_features is the correct row count. The trailing
    // feature dimension must match in_features_.
    if (input_shape.empty()) {
        throw std::runtime_error(
            "QuantizedLinear::forward_quantized: input must have rank >= 1");
    }
    if (input_shape.back() != in_features_) {
        throw std::runtime_error(
            "QuantizedLinear::forward_quantized: last input dim (" +
            std::to_string(input_shape.back()) + ") != in_features (" +
            std::to_string(in_features_) + ")");
    }
    int64_t batch_size = 1;
    for (size_t i = 0; i + 1 < input_shape.size(); ++i) {
        batch_size *= input_shape[i];
    }
    // Output shape preserves the leading dims and replaces in_features with
    // out_features: input_shape[:-1] + [out_features].
    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end() - 1);
    out_shape.push_back(out_features_);

    auto original_device = input.device();

    // GPU dispatch fast-path: for the common per-tensor INT8 case on a
    // non-CPU device, route through the registered OpId::QuantizedLinear
    // (every backend has a native kernel registered). Per-channel and
    // INT4 paths still fall through to the host loop below — those would
    // need additional dispatch attributes / kernels to support on-device.
    {
        const auto& input_params = input.params();
        const auto& weight_params = weight_.params();
        bool wt_per_channel = (weight_params.scheme == QuantizationScheme::PerChannelSymmetric ||
                               weight_params.scheme == QuantizationScheme::PerChannelAsymmetric);
        bool wt_int4 = (weight_.data().dtype() == DType::QInt4x2);
        // F032/F031: per-channel INT8 AND packed INT4 (QInt4x2) now dispatch to
        // their on-device CUDA kernels (no silent host round-trip). INT8 reads
        // weight scale/zp as inputs[3]/[4]; INT4 is symmetric per-tensor (scalar
        // weight scale), matching the CPU int4 kernel.
        bool dispatch_eligible =
            original_device.type != Device::Type::CPU &&
            input.data().dtype() == DType::Int8 &&
            (weight_.data().dtype() == DType::Int8 ||
             weight_.data().dtype() == DType::QInt4x2);

        if (dispatch_eligible) {
            // The OpId::QuantizedLinear contract is per-tensor scalar attrs,
            // so we read four small Int32 / Float32 scalars to host. The
            // tensor data (input / weight / bias) stays on the device.
            Tensor in_scale_cpu  = input_params.scale.to(Device::cpu());
            Tensor wt_scale_cpu  = weight_params.scale.to(Device::cpu());
            Tensor in_zp_cpu     = input_params.zero_point.to(Device::cpu());
            Tensor wt_zp_cpu     = weight_params.zero_point.to(Device::cpu());

            float in_scale  = in_scale_cpu.data<const float>()[0];
            float wt_scale  = wt_scale_cpu.data<const float>()[0];
            int32_t in_zp   = in_zp_cpu.data<int32_t>()[0];
            int32_t wt_zp   = wt_zp_cpu.data<int32_t>()[0];

            NewOpAttributes attrs;
            attrs.set(AttrKey::InputScale,      static_cast<double>(in_scale));
            attrs.set(AttrKey::WeightScaleQ,    static_cast<double>(wt_scale));
            // forward_quantized returns the dequantized FP32 result, so the
            // kernel's output_scale must be 1.0 (combined_scale =
            // input_scale * weight_scale / output_scale). Any output
            // requantization is handled by forward_quantized_output.
            attrs.set(AttrKey::OutputScale,     1.0);
            attrs.set(AttrKey::InputZeroPoint,  static_cast<int64_t>(in_zp));
            attrs.set(AttrKey::WeightZeroPoint, static_cast<int64_t>(wt_zp));

            // Flatten leading dims to a 2D [rows, in_features] view so the
            // backend kernel processes every row of a rank>=3 activation.
            Tensor input_2d = (input.data().shape().size() == 2)
                ? input.data()
                : input.data().reshape({batch_size, in_features_});

            std::vector<Tensor> inputs_vec;
            inputs_vec.push_back(input_2d);
            // JIT-R103: weight_ stays wherever the module was constructed
            // (CPU by default) until moved, unlike bias_dev/wscale/wzp below
            // which already reconcile — a lazily-placed module (constructed
            // on CPU, called with a GPU input) hit a device-mismatch
            // dispatch error here.
            Tensor weight_dev = weight_.data();
            if (weight_dev.device() != original_device) weight_dev = weight_dev.to(original_device);
            inputs_vec.push_back(weight_dev);
            // Always occupy the bias slot (inputs[2]) — empty when absent — so the
            // per-channel weight scale/zp land at the fixed inputs[3]/[4] positions
            // the kernel expects.
            if (bias_.has_value()) {
                Tensor bias_dev = *bias_;
                if (bias_dev.dtype() != DType::Float32) bias_dev = bias_dev.to(DType::Float32);
                if (bias_dev.device() != original_device) bias_dev = bias_dev.to(original_device);
                inputs_vec.push_back(bias_dev);
            } else {
                inputs_vec.push_back(Tensor({0}, DType::Float32, original_device));
            }
            if (wt_per_channel && !wt_int4) {
                // Per-channel weight scale ([out_features] Float32) and zero-point
                // ([out_features] Int32) as device tensors; the kernel dequantizes
                // each output channel with its own scale/zp.
                Tensor wscale = weight_params.scale;
                if (wscale.dtype() != DType::Float32) wscale = wscale.to(DType::Float32);
                if (wscale.device() != original_device) wscale = wscale.to(original_device);
                Tensor wzp = weight_params.zero_point;
                if (wzp.dtype() != DType::Int32) wzp = wzp.to(DType::Int32);
                if (wzp.device() != original_device) wzp = wzp.to(original_device);
                inputs_vec.push_back(wscale.contiguous());
                inputs_vec.push_back(wzp.contiguous());
            }
            // JIT-R058b/R103: OpId::QuantizedLinear is deliberately left
            // unmapped in tracing_interceptor.cpp (see that file's comment)
            // so a raw reach through it safely aborts the trace rather than
            // recording a wrong node. But forward_impl relies on manually
            // recording the CORRECT node afterward (jit_record_quantized_
            // linear_static, from the raw float input) for exactly the
            // per-tensor/INT8-both case this dispatch_eligible block
            // handles — and Tracer::abort_trace_unmappable() THROWS and
            // sets tracing_=false unconditionally, so the free dispatch<>()
            // call here (tracer-visible via DispatchInterceptorStack) was
            // aborting the WHOLE trace before that manual recording ever
            // ran, making jit_record_quantized_linear_static's GPU path
            // unreachable dead code. Call the backend table directly
            // instead — bypasses DispatchInterceptorStack entirely (mirrors
            // JIT-R098's sparse_add fix: src/autograd/ops.cpp), so this
            // computation is invisible to the interceptor and the manual
            // recording below is what makes it tracer-visible instead.
            auto& table = ::tenzor::DispatchTableRegistry::get_table(original_device.type);
            Tensor disp_out = table.dispatch(OpId::QuantizedLinear, inputs_vec, attrs)[0];
            // Restore the original leading dims: [rows, out_features] ->
            // input_shape[:-1] + [out_features].
            if (out_shape.size() != 2) {
                disp_out = disp_out.reshape(out_shape);
            }
            return disp_out;
        }
    }

    // Allocate output on CPU for computation
    Tensor output({batch_size, out_features_}, DType::Float32, Device::cpu());

    // Get quantization parameters - move to CPU for data access
    const auto& input_params = input.params();
    const auto& weight_params = weight_.params();

    Tensor input_scale_cpu = input_params.scale;
    Tensor weight_scale_cpu = weight_params.scale;
    Tensor input_zp_cpu = input_params.zero_point;
    Tensor weight_zp_cpu = weight_params.zero_point;

    if (input_scale_cpu.device() != Device::cpu()) {
        input_scale_cpu = input_scale_cpu.to(Device::cpu());
    }
    if (weight_scale_cpu.device() != Device::cpu()) {
        weight_scale_cpu = weight_scale_cpu.to(Device::cpu());
    }
    if (input_zp_cpu.device() != Device::cpu()) {
        input_zp_cpu = input_zp_cpu.to(Device::cpu());
    }
    if (weight_zp_cpu.device() != Device::cpu()) {
        weight_zp_cpu = weight_zp_cpu.to(Device::cpu());
    }

    float input_scale = input_scale_cpu.data<const float>()[0];
    int32_t input_zp = input_zp_cpu.data<int32_t>()[0];

    bool is_per_channel = (weight_params.scheme == QuantizationScheme::PerChannelSymmetric ||
                           weight_params.scheme == QuantizationScheme::PerChannelAsymmetric);

    // Move data to CPU for computation
    Tensor input_data_cpu = input.data();
    Tensor weight_data_cpu = weight_.data();
    if (input_data_cpu.device() != Device::cpu()) {
        input_data_cpu = input_data_cpu.to(Device::cpu());
    }
    if (weight_data_cpu.device() != Device::cpu()) {
        weight_data_cpu = weight_data_cpu.to(Device::cpu());
    }

    // Perform quantized matrix multiplication
    // uint8 asymmetric activations are stored as UInt8; remap to the signed
    // int8 the kernel expects (q_i8 = q_u8 - 128, zp_i8 = zp_u8 - 128). The
    // dequant value scale*(q - zp) is invariant under shifting q and zp by the
    // same constant, so the signed kernel yields identical results. Reading
    // UInt8 storage straight through data<int8_t>() reinterpreted codes >= 128
    // as negative and silently corrupted the output.
    if (input_data_cpu.dtype() == DType::UInt8) {
        input_data_cpu = sub(input_data_cpu.to(DType::Float32), 128.0).to(DType::Int8);
        input_zp -= 128;
    }
    const int8_t* input_data = input_data_cpu.data<int8_t>();
    // Convert bias to Float32 and CPU if needed
    std::optional<Tensor> bias_f32;
    const float* bias_data = nullptr;
    if (bias_.has_value()) {
        Tensor bias_cpu = *bias_;
        if (bias_cpu.dtype() != DType::Float32) {
            bias_cpu = bias_cpu.to(DType::Float32);
        }
        if (bias_cpu.device() != Device::cpu()) {
            bias_cpu = bias_cpu.to(Device::cpu());
        }
        bias_f32 = bias_cpu;
        bias_data = bias_f32->data<const float>();
    }
    float* output_data = output.data<float>();

    // INT4 (QInt4x2) weight path: use fused dequantizing matmul
    if (weight_data_cpu.dtype() == DType::QInt4x2) {
        const uint8_t* weight_packed = weight_data_cpu.data<uint8_t>();
        float weight_scale = weight_scale_cpu.data<const float>()[0];
        // F073: input_zp was already computed above (asymmetric activation
        // zero-point) and IS passed to the int8 kernels below — pass it here
        // too so asymmetric activation quantization is correctly applied for
        // INT4 weights as well, instead of silently defaulting act_zp to 0.
        cpu::fused_qlinear_dequant(
            input_data, weight_packed, bias_data, output_data,
            batch_size, out_features_, in_features_,
            input_scale, weight_scale, input_zp);
        if (out_shape.size() != 2) output = output.reshape(out_shape);
        return output.to(original_device);
    }

    const int8_t* weight_data = weight_data_cpu.data<int8_t>();

    if (is_per_channel) {
        // Per-channel: weight_scale_cpu has [out_features] scales
        const float* weight_scales = weight_scale_cpu.data<const float>();
        const int32_t* weight_zps = weight_zp_cpu.data<int32_t>();
        kernels::quantized_linear_per_channel_kernel(
            input_data, weight_data, bias_data, output_data,
            batch_size, in_features_, out_features_,
            input_scale, weight_scales, /*output_scale=*/1.0f,
            input_zp, weight_zps
        );
    } else {
        // Per-tensor: scalar scale and zero point
        float weight_scale = weight_scale_cpu.data<const float>()[0];
        int32_t weight_zp = weight_zp_cpu.data<int32_t>()[0];
        kernels::quantized_linear_kernel(
            input_data, weight_data, bias_data, output_data,
            batch_size, in_features_, out_features_,
            input_scale, weight_scale, /*output_scale=*/1.0f,
            input_zp, weight_zp
        );
        // JIT-R058b: tracing (when the static/calibrated activation case
        // qualifies) is recorded up in forward_impl instead of here — this
        // function only ever sees the ALREADY-quantized int8 QuantizedTensor
        // (no lineage back to the original float Variable, since
        // quantize_tensor's int8 write is a raw host loop with zero
        // dispatch() calls), so a jit_record call at this level could only
        // ever produce a lineage-broken, wrongly-frozen node. See
        // jit_record_quantized_linear_static's doc comment.
    }

    // Restore original leading dims, then move output back to original device.
    if (out_shape.size() != 2) output = output.reshape(out_shape);
    return output.to(original_device);
}

auto QuantizedLinear::forward_quantized_output(
    const QuantizedTensor& input,
    const QuantizationParams& output_qparams
) -> QuantizedTensor {
    // Similar to forward_quantized but quantizes the output
    Tensor fp_output = forward_quantized(input);
    return quantize_tensor(fp_output, output_qparams);
}

auto QuantizedLinear::set_weight(const QuantizedTensor& weights) -> void {
    weight_ = weights;
}

auto QuantizedLinear::set_bias(const Tensor& bias) -> void {
    bias_ = bias;
}

auto QuantizedLinear::from_float(const Linear& fp_linear, const QConfig& qconfig)
    -> std::shared_ptr<QuantizedLinear> {
    // forward_quantized reads the weight buffer as contiguous int8. INT4/UINT4
    // pack two values per byte (ceil(numel/2) bytes) stored as Int8/UInt8, so an
    // INT4 weight qconfig would cause out-of-bounds reads at inference (the
    // QInt4x2 dequant path never triggers because quantize_tensor emits packed
    // Int8/UInt8, not QInt4x2). Reject it here, matching Conv1d/ConvTranspose2d.
    if (qconfig.weight_dtype() != QuantDType::INT8 &&
        qconfig.weight_dtype() != QuantDType::UINT8) {
        throw std::invalid_argument(
            "QuantizedLinear::from_float: only INT8/UINT8 weight quantization is "
            "supported; INT4/UINT4 packing is incompatible with the int8 weight reader");
    }

    // Quantize weights
    auto weight_tensor = fp_linear.weight()->tensor();
    auto weight_observer = qconfig.create_weight_observer();
    weight_observer->observe(weight_tensor);
    auto weight_qparams = weight_observer->calculate_qparams(
        qconfig.weight_dtype(), qconfig.weight_scheme()
    );

    auto q_weight = quantize_tensor(weight_tensor, weight_qparams);

    // Create quantized layer
    auto q_layer = std::make_shared<QuantizedLinear>(
        weight_tensor.shape()[1],  // in_features
        weight_tensor.shape()[0],  // out_features
        weight_qparams
    );

    q_layer->set_weight(q_weight);

    // Set bias if present
    if (fp_linear.has_bias()) {
        q_layer->set_bias(fp_linear.bias()->tensor());
    }

    return q_layer;
}

auto quantized_linear_dynamic(const Tensor& input, const Tensor& weight,
                              const std::optional<Tensor>& bias) -> Tensor {
    // Dynamic per-tensor symmetric INT8 quantized linear used by the JIT
    // QuantizationPass interpreter. Mirrors from_float + forward EXACTLY:
    // quantize the fp32 weight to int8 (symmetric max(|w|)/127, matching the
    // pass's compute_scale_and_zero), build a QuantizedLinear, and run its
    // forward — which dynamically quantizes the activation and dispatches the
    // int8 matmul (OpId::QuantizedLinear, registered on every backend). Sharing
    // this code path guarantees JIT quantized == eager quantized on all backends.
    if (weight.shape().size() != 2) {
        throw std::runtime_error(
            "quantized_linear_dynamic: weight must be 2D [out_features, in_features]");
    }
    const int64_t out_features = weight.shape()[0];
    const int64_t in_features  = weight.shape()[1];
    QuantizedTensor q_weight = quantize_per_tensor_symmetric(weight);
    QuantizedLinear layer(in_features, out_features, q_weight.params());
    layer.set_weight(q_weight);
    if (bias.has_value()) {
        layer.set_bias(*bias);
    }
    return layer.forward(Variable(input, /*requires_grad=*/false)).tensor();
}

// ============================================================================
// QuantizedConv2d
// ============================================================================

auto quantized_conv2d_dynamic(const Tensor& input, const Tensor& weight,
                              const std::optional<Tensor>& bias, int64_t stride,
                              int64_t padding, int64_t dilation, int64_t groups)
    -> Tensor {
    // Dynamic per-tensor symmetric INT8 quantized conv2d, mirroring
    // QuantizedConv2d::from_float + forward (see quantized_linear_dynamic).
    // weight: [out_channels, in_channels/groups, kH, kW] with a square kernel
    // (the eager QuantizedConv2d supports square kernels + symmetric configs).
    if (weight.shape().size() != 4) {
        throw std::runtime_error(
            "quantized_conv2d_dynamic: weight must be 4D "
            "[out_channels, in_channels/groups, kH, kW]");
    }
    const int64_t out_channels = weight.shape()[0];
    const int64_t in_channels  = weight.shape()[1] * groups;
    const int64_t kernel_size  = weight.shape()[2];
    QuantizedTensor q_weight = quantize_per_tensor_symmetric(weight);
    QuantizedConv2d layer(in_channels, out_channels, kernel_size, stride, padding,
                          dilation, groups, q_weight.params());
    layer.set_weight(q_weight);
    if (bias.has_value()) {
        layer.set_bias(*bias);
    }
    return layer.forward(Variable(input, /*requires_grad=*/false)).tensor();
}

QuantizedConv2d::QuantizedConv2d(
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    QuantizationParams weight_qparams,
    float bias_scale
) : in_channels_(in_channels),
    out_channels_(out_channels),
    kernel_size_(kernel_size),
    stride_(stride),
    padding_(padding),
    dilation_(dilation),
    groups_(groups),
    weight_(Tensor({out_channels, in_channels / groups, kernel_size, kernel_size},
                   DType::Int8, Device::cpu()), weight_qparams),
    bias_scale_(bias_scale) {
    if (groups <= 0) {
        throw std::invalid_argument(
            "QuantizedConv2d: groups must be positive, got " + std::to_string(groups));
    }
    if (in_channels % groups != 0) {
        throw std::invalid_argument(
            "QuantizedConv2d: in_channels (" + std::to_string(in_channels) +
            ") must be divisible by groups (" + std::to_string(groups) + ")");
    }
    if (out_channels % groups != 0) {
        throw std::invalid_argument(
            "QuantizedConv2d: out_channels (" + std::to_string(out_channels) +
            ") must be divisible by groups (" + std::to_string(groups) + ")");
    }
}

auto QuantizedConv2d::forward_impl(const Variable& input) -> Variable {
    // Inference-only: see QuantizedLinear::forward_impl for rationale.
    // Compose Conv2d + FakeQuantize for QAT training instead.
    // JIT-R066: unlike QuantizedLinear, forward_quantized's im2col/kernel
    // path (kernels::quantized_conv2d_kernel/quantized_conv2d_per_channel_
    // kernel) is a 100% raw host-pointer computation with ZERO dispatch()
    // calls for either the static (calibrated) or dynamic activation-qparams
    // case — there is no partial dispatch lineage to build a manual
    // jit_record_quantized_linear_static-style replay node on top of, the
    // way QuantizedLinear's GPU fast path allows. Any trace would silently
    // freeze `output` as a trace-time constant regardless of the real
    // (replayed) input. Fail loudly instead (matches QuantizedLinear's
    // dynamic-path treatment, just unconditional here since neither
    // qparams path is trace-safe).
    if (::tenzor::jit::Tracer::get_instance().is_tracing()) {
        throw std::runtime_error(
            "QuantizedConv2d::forward_impl: cannot be JIT-traced (see "
            "JIT-R066 — forward_quantized's INT8 kernel is a raw host "
            "computation with no dispatch()-visible lineage back to the "
            "input, so a trace would silently freeze the output as a "
            "constant). Call outside jit.compile()/jit.trace().");
    }
    // Static quantization: quantize input with calibrated activation qparams.
    auto q_input = activation_qparams_.has_value()
        ? quantize_tensor(input.tensor(), *activation_qparams_)
        : quantize_per_tensor_symmetric(input.tensor());
    Tensor output = forward_quantized(q_input);
    return Variable(output, /*requires_grad=*/false);
}

auto QuantizedConv2d::forward_quantized(const QuantizedTensor& input) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::runtime_error(
            "QuantizedConv2d::forward_quantized: expected rank-4 NCHW input, got rank " +
            std::to_string(input_shape.size()));
    }
    int64_t batch = input_shape[0];
    int64_t h_in = input_shape[2];
    int64_t w_in = input_shape[3];

    // Remember original device
    auto original_device = input.device();

    // F075: reject a non-positive stride BEFORE it is used as a divisor below
    // — a zero stride is an integer divide-by-zero (UB/SIGFPE). Previously
    // this was only caught AFTER the fact by the h_out<=0/w_out<=0 check
    // below, by which point the division had already executed. Mirrors the
    // identical guard in cpu_kernel_registry.cpp / cuda_kernel_registry.cpp's
    // OpId::QuantizedConv2d entries.
    if (stride_ < 1) {
        throw std::invalid_argument(
            "QuantizedConv2d::forward_quantized: stride must be >= 1, got " +
            std::to_string(stride_));
    }

    // Compute output dimensions
    int64_t h_out = (h_in + 2 * padding_ - dilation_ * (kernel_size_ - 1) - 1) / stride_ + 1;
    int64_t w_out = (w_in + 2 * padding_ - dilation_ * (kernel_size_ - 1) - 1) / stride_ + 1;
    if (h_out <= 0 || w_out <= 0) {
        throw std::runtime_error(
            "QuantizedConv2d: computed output dims h_out=" + std::to_string(h_out) +
            " w_out=" + std::to_string(w_out) +
            " are non-positive; check kernel/stride/padding/dilation");
    }

    // GPU dispatch fast-path (F077): for the common per-tensor INT8 case on a
    // non-CPU device, route through the registered OpId::QuantizedConv2d
    // kernel (every backend has a native kernel registered — see
    // src/backends/cuda/kernels/quantization/quantized_conv2d.cu) instead of
    // unconditionally round-tripping input/weight/bias through host memory.
    // Mirrors QuantizedLinear::forward_quantized's dispatch_eligible block
    // above exactly (same eligibility checks, same fallback-to-CPU-round-trip
    // structure for anything that doesn't qualify). Unlike Linear there is no
    // INT4 branch to gate on: QuantizedConv2d::from_float rejects INT4/UINT4
    // weight quantization outright, so weight_ is always Int8/UInt8 here.
    {
        const auto& input_params = input.params();
        const auto& weight_params = weight_.params();
        bool wt_per_channel = (weight_params.scheme == QuantizationScheme::PerChannelSymmetric ||
                               weight_params.scheme == QuantizationScheme::PerChannelAsymmetric);
        bool dispatch_eligible =
            original_device.type != Device::Type::CPU &&
            input.data().dtype() == DType::Int8 &&
            weight_.data().dtype() == DType::Int8;

        if (dispatch_eligible) {
            // The OpId::QuantizedConv2d contract is per-tensor scalar attrs
            // (plus optional per-channel weight scale/zp tensors), so read
            // four small Int32/Float32 scalars to host. The tensor data
            // (input/weight/bias) stays on the device.
            Tensor in_scale_cpu  = input_params.scale.to(Device::cpu());
            Tensor wt_scale_cpu  = weight_params.scale.to(Device::cpu());
            Tensor in_zp_cpu     = input_params.zero_point.to(Device::cpu());
            Tensor wt_zp_cpu     = weight_params.zero_point.to(Device::cpu());

            float in_scale  = in_scale_cpu.data<const float>()[0];
            float wt_scale  = wt_scale_cpu.data<const float>()[0];
            int32_t in_zp   = in_zp_cpu.data<int32_t>()[0];
            int32_t wt_zp   = wt_zp_cpu.data<int32_t>()[0];

            NewOpAttributes attrs;
            attrs.set(AttrKey::InputScale,      static_cast<double>(in_scale));
            attrs.set(AttrKey::WeightScaleQ,    static_cast<double>(wt_scale));
            attrs.set(AttrKey::InputZeroPoint,  static_cast<int64_t>(in_zp));
            attrs.set(AttrKey::WeightZeroPoint, static_cast<int64_t>(wt_zp));
            attrs.set(AttrKey::Stride,   stride_);
            attrs.set(AttrKey::Padding,  padding_);
            attrs.set(AttrKey::Dilation, dilation_);
            attrs.set(AttrKey::Groups,   groups_);

            std::vector<Tensor> inputs_vec;
            inputs_vec.push_back(input.data());

            // JIT-R103-style device reconciliation: weight_ stays wherever the
            // module was constructed (CPU by default) until moved, unlike
            // input which already carries the real device.
            Tensor weight_dev = weight_.data();
            if (weight_dev.device() != original_device) weight_dev = weight_dev.to(original_device);
            inputs_vec.push_back(weight_dev);

            // Always occupy the bias slot (inputs[2]) — empty when absent —
            // so the per-channel weight scale/zp land at the fixed
            // inputs[3]/[4] positions the kernel expects (mirrors
            // QuantizedLinear's dispatch_eligible block above).
            if (bias_.has_value()) {
                Tensor bias_dev = *bias_;
                if (bias_dev.dtype() != DType::Float32) bias_dev = bias_dev.to(DType::Float32);
                if (bias_dev.device() != original_device) bias_dev = bias_dev.to(original_device);
                inputs_vec.push_back(bias_dev);
            } else {
                inputs_vec.push_back(Tensor({0}, DType::Float32, original_device));
            }
            if (wt_per_channel) {
                Tensor wscale = weight_params.scale;
                if (wscale.dtype() != DType::Float32) wscale = wscale.to(DType::Float32);
                if (wscale.device() != original_device) wscale = wscale.to(original_device);
                Tensor wzp = weight_params.zero_point;
                if (wzp.dtype() != DType::Int32) wzp = wzp.to(DType::Int32);
                if (wzp.device() != original_device) wzp = wzp.to(original_device);
                inputs_vec.push_back(wscale.contiguous());
                inputs_vec.push_back(wzp.contiguous());
            }

            auto& table = ::tenzor::DispatchTableRegistry::get_table(original_device.type);
            return table.dispatch(OpId::QuantizedConv2d, inputs_vec, attrs)[0];
        }
    }

    // Allocate output on CPU
    Tensor output({batch, out_channels_, h_out, w_out}, DType::Float32, Device::cpu());

    // Get quantization parameters - move to CPU for data access
    const auto& input_params = input.params();
    const auto& weight_params = weight_.params();

    Tensor input_scale_cpu = input_params.scale;
    Tensor weight_scale_cpu = weight_params.scale;
    Tensor input_zp_cpu = input_params.zero_point;
    Tensor weight_zp_cpu = weight_params.zero_point;

    if (input_scale_cpu.device() != Device::cpu()) {
        input_scale_cpu = input_scale_cpu.to(Device::cpu());
    }
    if (weight_scale_cpu.device() != Device::cpu()) {
        weight_scale_cpu = weight_scale_cpu.to(Device::cpu());
    }
    if (input_zp_cpu.device() != Device::cpu()) {
        input_zp_cpu = input_zp_cpu.to(Device::cpu());
    }
    if (weight_zp_cpu.device() != Device::cpu()) {
        weight_zp_cpu = weight_zp_cpu.to(Device::cpu());
    }

    float input_scale = input_scale_cpu.data<const float>()[0];
    int32_t input_zp = input_zp_cpu.data<int32_t>()[0];

    bool is_per_channel = (weight_params.scheme == QuantizationScheme::PerChannelSymmetric ||
                           weight_params.scheme == QuantizationScheme::PerChannelAsymmetric);

    // Move data to CPU for computation
    Tensor input_data_cpu = input.data();
    Tensor weight_data_cpu = weight_.data();
    if (input_data_cpu.device() != Device::cpu()) {
        input_data_cpu = input_data_cpu.to(Device::cpu());
    }
    if (weight_data_cpu.device() != Device::cpu()) {
        weight_data_cpu = weight_data_cpu.to(Device::cpu());
    }

    // uint8 asymmetric activations are stored as UInt8; remap to the signed
    // int8 the kernel expects (q_i8 = q_u8 - 128, zp_i8 = zp_u8 - 128). The
    // dequant value scale*(q - zp) is invariant under shifting q and zp by the
    // same constant, so the signed kernel yields identical results. Reading
    // UInt8 storage straight through data<int8_t>() reinterpreted codes >= 128
    // as negative and silently corrupted the output.
    if (input_data_cpu.dtype() == DType::UInt8) {
        input_data_cpu = sub(input_data_cpu.to(DType::Float32), 128.0).to(DType::Int8);
        input_zp -= 128;
    }
    const int8_t* input_data = input_data_cpu.data<int8_t>();
    const int8_t* weight_data = weight_data_cpu.data<int8_t>();
    // Convert bias to Float32 and CPU if needed
    std::optional<Tensor> bias_f32;
    const float* bias_data = nullptr;
    if (bias_.has_value()) {
        Tensor bias_cpu = *bias_;
        if (bias_cpu.dtype() != DType::Float32) {
            bias_cpu = bias_cpu.to(DType::Float32);
        }
        if (bias_cpu.device() != Device::cpu()) {
            bias_cpu = bias_cpu.to(Device::cpu());
        }
        bias_f32 = bias_cpu;
        bias_data = bias_f32->data<const float>();
    }
    float* output_data = output.data<float>();

    if (is_per_channel) {
        const float* weight_scales = weight_scale_cpu.data<const float>();
        const int32_t* weight_zps = weight_zp_cpu.data<int32_t>();
        // QuantizedConv2d uses square kernel/stride/padding/dilation, so the
        // per-axis kernel params take the same scalar value for H and W.
        kernels::quantized_conv2d_per_channel_kernel(
            input_data, weight_data, bias_data, output_data,
            batch, in_channels_, out_channels_,
            h_in, w_in, h_out, w_out,
            kernel_size_, kernel_size_, stride_, stride_, padding_, padding_,
            input_scale, weight_scales, input_zp, weight_zps,
            dilation_, dilation_, groups_
        );
    } else {
        float weight_scale = weight_scale_cpu.data<const float>()[0];
        int32_t weight_zp = weight_zp_cpu.data<int32_t>()[0];
        kernels::quantized_conv2d_kernel(
            input_data, weight_data, bias_data, output_data,
            batch, in_channels_, out_channels_,
            h_in, w_in, h_out, w_out,
            kernel_size_, kernel_size_, stride_, stride_, padding_, padding_,
            input_scale, weight_scale, input_zp, weight_zp,
            dilation_, dilation_, groups_
        );
    }

    // Move output back to original device
    return output.to(original_device);
}

auto QuantizedConv2d::forward_quantized_output(
    const QuantizedTensor& input,
    const QuantizationParams& output_qparams
) -> QuantizedTensor {
    Tensor fp_output = forward_quantized(input);
    return quantize_tensor(fp_output, output_qparams);
}

auto QuantizedConv2d::set_weight(const QuantizedTensor& weights) -> void {
    weight_ = weights;
}

auto QuantizedConv2d::set_bias(const Tensor& bias) -> void {
    bias_ = bias;
}

auto QuantizedConv2d::from_float(const Conv2d& fp_conv, const QConfig& qconfig)
    -> std::shared_ptr<QuantizedConv2d> {
    // forward_quantized reads the weight buffer as contiguous int8. INT4/UINT4
    // pack two values per byte (ceil(numel/2) bytes) stored as Int8/UInt8, so an
    // INT4 weight qconfig would cause out-of-bounds reads at inference. Reject
    // it here, matching Conv1d/ConvTranspose2d.
    if (qconfig.weight_dtype() != QuantDType::INT8 &&
        qconfig.weight_dtype() != QuantDType::UINT8) {
        throw std::invalid_argument(
            "QuantizedConv2d::from_float: only INT8/UINT8 weight quantization is "
            "supported; INT4/UINT4 packing is incompatible with the int8 weight reader");
    }

    // Extract weight and bias from float Conv2d
    auto state_dict = fp_conv.state_dict();

    Tensor fp_weight = state_dict.at("weight");
    std::optional<Tensor> fp_bias;
    if (state_dict.find("bias") != state_dict.end()) {
        fp_bias = state_dict.at("bias");
    }

    // Quantize weights using the weight observer from qconfig
    auto weight_observer = qconfig.create_weight_observer();
    weight_observer->observe(fp_weight);
    auto weight_qparams = weight_observer->calculate_qparams(
        qconfig.weight_dtype(),
        qconfig.weight_scheme()
    );

    // Extract Conv2d parameters from weight shape
    auto weight_shape = fp_weight.shape();
    int64_t out_channels = weight_shape[0];
    // Conv2d weight is [out_channels, in_channels/groups, kH, kW]; recover the
    // true in_channels by multiplying by groups (matches QuantizedConv2dBnReLU).
    int64_t in_channels = weight_shape[1] * fp_conv.groups();
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];
    if (kernel_h != kernel_w) {
        throw std::runtime_error(
            "QuantizedConv2d::from_float: non-square kernels not supported (got " +
            std::to_string(kernel_h) + "x" + std::to_string(kernel_w) + ")");
    }
    // The quantized ctor stores a single scalar stride/padding/dilation; reject
    // asymmetric geometry rather than silently using the H value for both axes
    // (mirrors QuantizedConvTranspose2d::from_float).
    if (fp_conv.stride_h() != fp_conv.stride_w() ||
        fp_conv.padding_h() != fp_conv.padding_w() ||
        fp_conv.dilation_h() != fp_conv.dilation_w()) {
        throw std::runtime_error(
            "QuantizedConv2d::from_float: asymmetric stride/padding/dilation not supported");
    }

    // Create quantized Conv2d with actual parameters from source Conv2d
    auto q_conv = std::make_shared<QuantizedConv2d>(
        in_channels,
        out_channels,
        kernel_h,
        fp_conv.stride_h(),
        fp_conv.padding_h(),
        fp_conv.dilation_h(),
        fp_conv.groups(),
        weight_qparams,
        1.0f       // bias_scale (default)
    );

    // Quantize and set the weights
    QuantizedTensor q_weight = quantize_tensor(fp_weight, weight_qparams);
    q_conv->set_weight(q_weight);

    // Set bias if available
    if (fp_bias.has_value()) {
        q_conv->set_bias(fp_bias.value());
    }

    return q_conv;
}

// ============================================================================
// QuantizedBatchNorm2d
// ============================================================================

QuantizedBatchNorm2d::QuantizedBatchNorm2d(
    int64_t num_features,
    Tensor scale,
    Tensor bias
) : num_features_(num_features),
    scale_(std::move(scale)),
    bias_(std::move(bias)) {}

auto QuantizedBatchNorm2d::forward_impl(const Variable& input) -> Variable {
    // Audit item B.1: QuantizedBatchNorm2d is an INFERENCE-ONLY layer
    // (it has no learnable parameters in its quantized form — scale and
    // bias are fixed quantization-aware folded constants).  Previously
    // returned `Variable(output, input.requires_grad())` which claimed
    // gradient flow that the raw-tensor pipeline could not deliver.
    //
    // Honest contract: build via raw tensor ops, return with
    // requires_grad=false, and refuse to participate in autograd if the
    // caller passed a requires_grad=true input (the result was being
    // silently disconnected from the upstream graph anyway).
    if (input.requires_grad()) {
        throw std::runtime_error(
            "QuantizedBatchNorm2d: input has requires_grad=true but this "
            "is an inference-only quantized layer; gradients cannot flow "
            "through.  Use a non-quantized BatchNorm2d for QAT/training.");
    }
    Tensor scaled = input.tensor() * scale_.unsqueeze(0).unsqueeze(2).unsqueeze(3);
    Tensor output = scaled + bias_.unsqueeze(0).unsqueeze(2).unsqueeze(3);
    return Variable(output, /*requires_grad=*/false);
}

auto QuantizedBatchNorm2d::forward_quantized(const QuantizedTensor& input) -> QuantizedTensor {
    // Dequantize, apply BN, requantize
    Tensor deq = input.dequantize();
    Tensor scaled = deq * scale_.unsqueeze(0).unsqueeze(2).unsqueeze(3);
    Tensor output = scaled + bias_.unsqueeze(0).unsqueeze(2).unsqueeze(3);
    return quantize_per_tensor_symmetric(output);
}

auto QuantizedBatchNorm2d::from_float(const Module& fp_bn, [[maybe_unused]] const QConfig& qconfig)
    -> std::shared_ptr<QuantizedBatchNorm2d> {
    // Extract BatchNorm parameters from state_dict
    auto state_dict = fp_bn.state_dict();

    // BatchNorm parameters: weight (gamma), bias (beta), running_mean, running_var.
    // Fold in Float32 so the derived scale_/bias_ match the dequantized (Float32)
    // input in forward_quantized; a Float16/BFloat16 source BN would otherwise
    // produce half-precision scale_/bias_ and a dtype mismatch at `deq * scale_`.
    Tensor gamma = state_dict.at("weight").to(DType::Float32);
    Tensor beta = state_dict.at("bias").to(DType::Float32);
    Tensor running_mean = state_dict.at("running_mean").to(DType::Float32);
    Tensor running_var = state_dict.at("running_var").to(DType::Float32);

    // Fold BatchNorm parameters into scale and bias
    // Formula:
    //   scale = gamma / sqrt(running_var + eps)
    //   bias = beta - scale * running_mean
    // This allows BN to be applied as: y = scale * x + bias

    // Read epsilon from the source module if possible, otherwise use default
    double eps = 1e-5;
    if (auto* bn2d = dynamic_cast<const BatchNorm2d*>(&fp_bn)) {
        eps = bn2d->eps();
    } else if (auto* bn1d = dynamic_cast<const BatchNorm1d*>(&fp_bn)) {
        eps = bn1d->eps();
    }

    // Compute scale = gamma / sqrt(var + eps)
    Tensor sqrt_var = sqrt(running_var + eps);
    Tensor scale = gamma / sqrt_var;

    // Compute bias = beta - scale * mean
    Tensor bias = beta - scale * running_mean;

    // Get number of features from gamma shape
    int64_t num_features = gamma.shape()[0];

    // Create quantized BatchNorm with folded parameters
    auto q_bn = std::make_shared<QuantizedBatchNorm2d>(
        num_features,
        scale,
        bias
    );

    return q_bn;
}

// ============================================================================
// QuantizedLayerNorm Implementation
// ============================================================================

QuantizedLayerNorm::QuantizedLayerNorm(
    std::vector<int64_t> normalized_shape,
    Tensor weight,
    Tensor bias,
    double eps
) : normalized_shape_(std::move(normalized_shape)),
    weight_(std::move(weight)),
    bias_(std::move(bias)),
    eps_(eps) {}

auto QuantizedLayerNorm::forward_impl(const Variable& input) -> Variable {
    // Audit item B.1: QuantizedLayerNorm is inference-only; previous
    // forward built the output via raw tensor ops and wrapped it in
    // Variable(out, input.requires_grad()) — the raw-tensor pipeline
    // does not flow gradients, so this was a false claim.
    if (input.requires_grad()) {
        throw std::runtime_error(
            "QuantizedLayerNorm: input has requires_grad=true but this is "
            "an inference-only quantized layer; gradients cannot flow "
            "through.  Use a non-quantized LayerNorm for QAT/training.");
    }
    // Layer norm: y = (x - mean) / sqrt(var + eps) * weight + bias
    // For quantized inference, we compute at float precision
    Tensor x = input.tensor();
    auto x_shape = x.shape();
    int64_t norm_size = 1;
    for (auto d : normalized_shape_) norm_size *= d;

    // Compute mean and variance over the last N dimensions
    int64_t outer_size = x.numel() / norm_size;
    Tensor x_flat = x.reshape({outer_size, norm_size});

    Tensor x_mean = ::tenzor::mean(x_flat, 1, true);    // [outer, 1]
    Tensor centered = x_flat - x_mean;
    Tensor var = ::tenzor::mean(centered * centered, 1, true);  // [outer, 1]

    // inv_std = 1 / sqrt(var + eps)
    auto eps_t = ::tenzor::full({outer_size, 1}, static_cast<float>(eps_),
                                 var.dtype(), var.device());
    Tensor std_val = ::tenzor::sqrt(var + eps_t);
    Tensor ones_t = ::tenzor::ones({outer_size, 1}, var.dtype(), var.device());
    Tensor inv_std = ::tenzor::div(ones_t, std_val);
    Tensor normalized = centered * inv_std;

    // Reshape back and apply weight + bias
    normalized = normalized.reshape(std::vector<int64_t>(x_shape.begin(), x_shape.end()));
    Tensor output = normalized * weight_ + bias_;

    // Honest contract — see comment at top of forward_impl.
    return Variable(output, /*requires_grad=*/false);
}

auto QuantizedLayerNorm::forward_quantized(const QuantizedTensor& input) -> QuantizedTensor {
    // Dequantize → layer norm → requantize
    Tensor deq = input.dequantize();
    auto result = forward_impl(Variable(deq, false));
    return quantize_per_tensor_symmetric(result.tensor());
}

auto QuantizedLayerNorm::from_float(const Module& fp_ln, const QConfig& /*qconfig*/)
    -> std::shared_ptr<QuantizedLayerNorm> {
    auto state_dict = fp_ln.state_dict();

    Tensor weight = state_dict.at("weight");
    Tensor bias = state_dict.at("bias");

    // Infer normalized_shape from weight shape
    auto wshape = weight.shape();
    std::vector<int64_t> normalized_shape(wshape.begin(), wshape.end());

    // Read epsilon from the source module if possible, otherwise use default
    double eps = 1e-5;
    if (auto* ln = dynamic_cast<const LayerNorm*>(&fp_ln)) {
        eps = ln->eps();
    }

    return std::make_shared<QuantizedLayerNorm>(
        normalized_shape, weight, bias, eps);
}

// ============================================================================
// QuantStub / DeQuantStub - Full Quantization/Dequantization Implementation
// ============================================================================

QuantStub::QuantStub(QuantizationParams qparams)
    : qparams_(std::move(qparams)) {}

auto QuantStub::set_calibrating(bool calibrating) -> void {
    calibrating_ = calibrating;
    if (calibrating && !observer_) {
        // Lazy: HistogramObserver with default 2048 bins and the standard
        // percentile clipping. dtype/scheme are passed to calculate_qparams
        // at finalisation time (via update_qparams_from_observer), not the
        // constructor.
        observer_ = std::make_unique<HistogramObserver>(/*num_bins=*/2048);
    }
}

auto QuantStub::update_qparams_from_observer() -> void {
    if (!observer_ || !observer_->has_data()) {
        throw std::runtime_error(
            "QuantStub::update_qparams_from_observer: no calibration data "
            "observed. Call set_calibrating(true) and run at least one "
            "forward pass with calibration inputs first.");
    }
    qparams_ = observer_->calculate_qparams(qparams_.dtype, qparams_.scheme);
}

auto QuantStub::forward_impl(const Variable& input) -> Variable {
    // S20: QuantStub now has three honest modes:
    //
    //   (a) calibrating_ == true
    //       Observer collects the input distribution; the Variable passes
    //       through unchanged (grad_fn preserved). This is the
    //       PyTorch-style `prepare` -> calibration step before convert.
    //
    //   (b) calibrating_ == false, !requires_grad / no grad enabled
    //       Real PTQ Q->DQ noise injection on the tensor (real quantise
    //       to INT8 using stored scale/zp, then dequantise back to FP32).
    //       Returns a fresh Variable with grad disabled — honest about the
    //       loss of grad through an integer round-trip.
    //
    //   (c) calibrating_ == false, requires_grad and grad enabled (QAT)
    //       Real STE through `fake_quantize_with_grad`: forward injects
    //       Q/DQ noise, backward passes grad_output through (clipped to
    //       the representable range). This is the autograd-Function path
    //       that lets QAT training actually learn.
    //
    // The pre-S20 implementation lacked (a) entirely and the documentation
    // misleadingly suggested the Variable carried quantisation metadata
    // that downstream layers could read. With observe-mode + Q/DQ-with-STE
    // the API matches what its docstring claims.

    if (calibrating_) {
        if (!observer_) {
            // Defensive: set_calibrating wasn't called via the public API
            // (e.g. constructed via aggregate initialisation). Allocate the
            // observer on first observation so the call still works.
            observer_ = std::make_unique<HistogramObserver>(/*num_bins=*/2048);
        }
        observer_->observe(input.tensor());
        // Passthrough: returning `input` directly preserves grad_fn so a
        // calibration step inside a larger autograd graph doesn't sever
        // upstream training.
        return input;
    }

    if (!input.requires_grad() || !is_grad_enabled()) {
        // (b) PTQ inference: real Q -> DQ on raw tensor.
        auto q_tensor = forward_to_quantized(input.tensor());
        Tensor dequantized = q_tensor.dequantize();
        return Variable(dequantized, /*requires_grad=*/false);
    }

    // (c) QAT: STE through the quantisation boundary.
    if (qparams_.axis >= 0 && qparams_.scale.numel() > 1) {
        throw std::runtime_error(
            "QuantStub: QAT (requires_grad=true) is only supported for "
            "per-tensor quantization; per-channel scale was supplied. "
            "Use FakeQuantize directly for per-channel QAT.");
    }
    // scale / zero_point may live on a non-CPU device (the module was moved to
    // GPU); read their scalar values via a host copy — data<T>()[0] on a device
    // pointer would segfault.
    const float scale = qparams_.scale.numel() > 0
        ? qparams_.scale.cpu().data<float>()[0]
        : 1.0f;
    // zero_point's dtype is NOT fixed by the API: QuantizationParams accepts
    // any integer dtype the caller supplies (the S20 QAT tests use Int64, the
    // FakeQuantize per-channel tests use Int32, quantize.cpp uses Int32).
    // Reading it with a hardcoded data<int32_t>() throws DTypeException when
    // the caller supplied Int64 (or any other int width), deterministically
    // breaking QAT-through-QuantStub for those callers. Read the scalar
    // according to its ACTUAL dtype instead.
    const float zero_point = [&] () -> float {
        if (qparams_.zero_point.numel() == 0) return 0.0f;
        const Tensor zp_cpu = qparams_.zero_point.cpu();
        switch (zp_cpu.dtype()) {
            case DType::Int32:    return static_cast<float>(zp_cpu.data<int32_t>()[0]);
            case DType::Int64:    return static_cast<float>(zp_cpu.data<int64_t>()[0]);
            case DType::Int16:    return static_cast<float>(zp_cpu.data<int16_t>()[0]);
            case DType::Int8:     return static_cast<float>(zp_cpu.data<int8_t>()[0]);
            case DType::UInt8:    return static_cast<float>(zp_cpu.data<uint8_t>()[0]);
            case DType::Float32:  return zp_cpu.data<float>()[0];
            default:
                throw std::runtime_error(
                    "QuantStub::forward_impl: unsupported zero_point dtype " +
                    std::to_string(static_cast<int>(zp_cpu.dtype())));
        }
    }();
    float qmin = -128.0f, qmax = 127.0f;
    switch (qparams_.dtype) {
        case QuantDType::INT8:   qmin = -128.0f; qmax = 127.0f;  break;
        case QuantDType::UINT8:  qmin =    0.0f; qmax = 255.0f;  break;
        case QuantDType::INT4:   qmin =   -8.0f; qmax =   7.0f;  break;
        case QuantDType::UINT4:  qmin =    0.0f; qmax =  15.0f;  break;
    }
    // Match quantize_tensor's symmetric range: INT8 symmetric uses [-127,127]
    // and INT4 symmetric uses [-7,7], so the QAT STE never simulates the -128/-8
    // level that real quantized inference clamps away.
    const bool symmetric = (qparams_.scheme == QuantizationScheme::PerTensorSymmetric ||
                            qparams_.scheme == QuantizationScheme::PerChannelSymmetric);
    if (symmetric && qparams_.dtype == QuantDType::INT8) {
        qmin = -127.0f;
    } else if (symmetric && qparams_.dtype == QuantDType::INT4) {
        qmin = -7.0f;
    }
    return ::tenzor::nn::quantization::fake_quantize_with_grad(
        input, scale, zero_point, qmin, qmax);
}

auto QuantStub::forward_to_quantized(const Tensor& input) -> QuantizedTensor {
    // Perform full quantization:
    // 1. Validate input is floating-point
    if (input.dtype() != DType::Float32 && input.dtype() != DType::Float64) {
        throw std::runtime_error("QuantStub: Input must be floating-point type");
    }

    // 2. Convert to Float32 if needed
    Tensor fp_input = (input.dtype() == DType::Float32) ? input : input.to(DType::Float32);

    // 3. Apply quantization using stored parameters
    // This handles both per-tensor and per-channel quantization
    // Formula: q = clamp(round(x / scale) + zero_point, qmin, qmax)
    return quantize_tensor(fp_input, qparams_);
}

auto DeQuantStub::forward_impl(const Variable& input) -> Variable {
    // Correct by design — NOT a stub. In Tenzor, autograd Variables always carry
    // FP32 data: QuantStub::forward_impl returns FP32 in every mode (calibration
    // passthrough, PTQ Q->DQ which dequantizes back to FP32, and QAT fake-quant
    // which is FP32 + STE). Real INT8 storage lives in QuantizedTensor and inside
    // the QuantizedLinear/QuantizedConv2d ops, never in a Variable. So by the time
    // a value reaches DeQuantStub it is already dequantized FP32; this identity
    // completes the QuantStub->DeQuantStub pair. The genuinely-quantized entry
    // point is forward_from_quantized(QuantizedTensor).
    // (Returning `input` directly preserves grad_fn — an earlier version rebuilt
    // the Variable and severed the autograd chain.)
    return input;
}

auto DeQuantStub::forward_from_quantized(const QuantizedTensor& input) -> Tensor {
    // Perform full dequantization:
    // 1. Extract quantization parameters
    const auto& params = input.params();

    // 2. Track if this was per-channel for debugging
    last_per_channel_ = (params.axis != -1);

    // 3. Apply dequantization based on scheme
    // Formula: x = (q - zero_point) * scale
    // This handles both:
    //   - Per-tensor: Single scale and zero_point for entire tensor
    //   - Per-channel: Different scale/zero_point per channel along specified axis

    return dequantize_tensor(input);
}

// ============================================================================
// Fused Layers
// ============================================================================

auto QuantizedConv2dReLU::forward_quantized(const QuantizedTensor& input) -> Tensor {
    Tensor output = QuantizedConv2d::forward_quantized(input);

    // Remember original device
    auto original_device = output.device();

    // Move to CPU for data access
    Tensor output_cpu = output;
    if (output_cpu.device() != Device::cpu()) {
        output_cpu = output_cpu.to(Device::cpu());
    }

    // Apply ReLU
    float* data = output_cpu.data<float>();
    int64_t n = output_cpu.numel();
    for (int64_t i = 0; i < n; ++i) {
        data[i] = std::max(0.0f, data[i]);
    }

    // Move back to original device
    return output_cpu.to(original_device);
}

auto QuantizedConv2dReLU::forward_quantized_output(
    const QuantizedTensor& input,
    const QuantizationParams& output_qparams
) -> QuantizedTensor {
    Tensor fp_output = forward_quantized(input);
    return quantize_tensor(fp_output, output_qparams);
}

auto QuantizedConv2dReLU::from_float(const Conv2d& fp_conv, const QConfig& qconfig)
    -> std::shared_ptr<QuantizedConv2dReLU> {
    // forward_quantized reads the weight buffer as contiguous int8; INT4/UINT4
    // pack two values per byte (ceil(numel/2) bytes), causing out-of-bounds
    // reads at inference. Reject here, matching the sibling from_float methods
    // (QuantizedLinear/Conv2d/Conv1d/ConvTranspose2d).
    if (qconfig.weight_dtype() != QuantDType::INT8 &&
        qconfig.weight_dtype() != QuantDType::UINT8) {
        throw std::invalid_argument(
            "QuantizedConv2dReLU::from_float: only INT8/UINT8 weight quantization "
            "is supported; INT4/UINT4 packing is incompatible with the int8 weight reader");
    }

    // Extract weight and bias from float Conv2d
    auto state_dict = fp_conv.state_dict();

    Tensor fp_weight = state_dict.at("weight");
    std::optional<Tensor> fp_bias;
    if (state_dict.find("bias") != state_dict.end()) {
        fp_bias = state_dict.at("bias");
    }

    // Quantize weights using the weight observer from qconfig
    auto weight_observer = qconfig.create_weight_observer();
    weight_observer->observe(fp_weight);
    auto weight_qparams = weight_observer->calculate_qparams(
        qconfig.weight_dtype(),
        qconfig.weight_scheme()
    );

    // Extract Conv2d parameters from weight shape
    auto weight_shape = fp_weight.shape();
    int64_t out_channels = weight_shape[0];
    // Conv2d weight is [out_channels, in_channels/groups, kH, kW]; recover the
    // true in_channels by multiplying by groups (matches QuantizedConv2dBnReLU).
    int64_t in_channels = weight_shape[1] * fp_conv.groups();
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];
    // The quantized ctor stores single scalar kernel/stride/padding/dilation;
    // reject non-square kernels and asymmetric geometry rather than silently using
    // the H value for both axes (mirrors QuantizedConv2d::from_float).
    if (kernel_h != kernel_w) {
        throw std::runtime_error(
            "QuantizedConv2dReLU::from_float: non-square kernels not supported (got " +
            std::to_string(kernel_h) + "x" + std::to_string(kernel_w) + ")");
    }
    if (fp_conv.stride_h() != fp_conv.stride_w() ||
        fp_conv.padding_h() != fp_conv.padding_w() ||
        fp_conv.dilation_h() != fp_conv.dilation_w()) {
        throw std::runtime_error(
            "QuantizedConv2dReLU::from_float: asymmetric stride/padding/dilation not supported");
    }

    // Create QuantizedConv2dReLU with actual parameters from source Conv2d
    auto q_conv_relu = std::shared_ptr<QuantizedConv2dReLU>(
        new QuantizedConv2dReLU(
            in_channels,
            out_channels,
            kernel_h,
            fp_conv.stride_h(),
            fp_conv.padding_h(),
            fp_conv.dilation_h(),
            fp_conv.groups(),
            weight_qparams,
            1.0f       // bias_scale
        )
    );

    // Quantize and set the weights
    QuantizedTensor q_weight = quantize_tensor(fp_weight, weight_qparams);
    q_conv_relu->set_weight(q_weight);

    // Set bias if available
    if (fp_bias.has_value()) {
        q_conv_relu->set_bias(fp_bias.value());
    }

    return q_conv_relu;
}

// ============================================================================
// QuantizedConv2dBnReLU
// ============================================================================

QuantizedConv2dBnReLU::QuantizedConv2dBnReLU(
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    QuantizationParams weight_qparams,
    Tensor bn_scale,
    Tensor bn_bias
) : bn_scale_(std::move(bn_scale)),
    bn_bias_(std::move(bn_bias)) {
    // Create internal quantized conv with the folded weights
    conv_ = std::make_shared<QuantizedConv2d>(
        in_channels, out_channels, kernel_size,
        stride, padding, dilation, groups,
        weight_qparams
    );
}

auto QuantizedConv2dBnReLU::forward_impl(const Variable& input) -> Variable {
    // Audit item B.1: inference-only quantized layer.  Previously
    // returned `Variable(output, input.requires_grad())` claiming
    // gradient flow that the quantize→fused-conv-bn-relu→dequant
    // pipeline cannot deliver.
    if (input.requires_grad()) {
        throw std::runtime_error(
            "QuantizedConv2dBnReLU: input has requires_grad=true but this "
            "is an inference-only quantized fused layer.  Use a "
            "non-quantized Conv2d+BatchNorm2d+ReLU for QAT/training.");
    }
    // JIT-R066: see QuantizedConv2d::forward_impl — same raw, dispatch-
    // invisible INT8 kernel path (always dynamic here, no calibrated-qparams
    // branch even), so a trace would silently freeze `output` as a constant.
    if (::tenzor::jit::Tracer::get_instance().is_tracing()) {
        throw std::runtime_error(
            "QuantizedConv2dBnReLU::forward_impl: cannot be JIT-traced (see "
            "JIT-R066 — the fused conv-bn-relu INT8 kernel is a raw host "
            "computation with no dispatch()-visible lineage back to the "
            "input). Call outside jit.compile()/jit.trace().");
    }
    auto q_input = quantize_per_tensor_symmetric(input.tensor());
    Tensor output = forward_quantized(q_input);
    return Variable(output, /*requires_grad=*/false);
}

auto QuantizedConv2dBnReLU::forward_quantized(const QuantizedTensor& input) -> Tensor {
    // Run quantized convolution (with BN folded into weights)
    Tensor output = conv_->forward_quantized(input);

    auto original_device = output.device();

    // Move to CPU for element-wise operations
    Tensor output_cpu = output;
    if (output_cpu.device() != Device::cpu()) {
        output_cpu = output_cpu.to(Device::cpu());
    }

    // Apply BN scale and bias (already folded from running stats)
    // output shape: [N, C, H, W]
    // bn_scale_ and bn_bias_ shape: [C]
    auto output_shape = output_cpu.shape();
    int64_t batch = output_shape[0];
    int64_t channels = output_shape[1];
    int64_t h_out = output_shape[2];
    int64_t w_out = output_shape[3];

    Tensor bn_scale_cpu = bn_scale_;
    Tensor bn_bias_cpu = bn_bias_;
    if (bn_scale_cpu.device() != Device::cpu()) {
        bn_scale_cpu = bn_scale_cpu.to(Device::cpu());
    }
    if (bn_bias_cpu.device() != Device::cpu()) {
        bn_bias_cpu = bn_bias_cpu.to(Device::cpu());
    }

    float* out_data = output_cpu.data<float>();
    const float* scale_data = bn_scale_cpu.data<const float>();
    const float* bias_data = bn_bias_cpu.data<const float>();

    // Apply BN: y = scale * x + bias, then ReLU: y = max(0, y)
    int64_t spatial = h_out * w_out;
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            float s = scale_data[c];
            float bi = bias_data[c];
            float* channel_data = out_data + (b * channels + c) * spatial;
            for (int64_t i = 0; i < spatial; ++i) {
                float val = channel_data[i] * s + bi;
                channel_data[i] = std::max(0.0f, val);  // Fused ReLU
            }
        }
    }

    return output_cpu.to(original_device);
}

auto QuantizedConv2dBnReLU::from_float(
    const Conv2d& fp_conv,
    const Module& fp_bn,
    const QConfig& qconfig
) -> std::shared_ptr<QuantizedConv2dBnReLU> {
    // Extract BatchNorm parameters from state_dict
    auto bn_state = fp_bn.state_dict();

    // Widen to Float32 before the raw float access below — Float16/BFloat16 BN
    // params would otherwise be misread by .data<float>() (matches
    // QuantizedBatchNorm2d::from_float and fake_quantize.cpp::fold_bn).
    Tensor gamma = bn_state.at("weight").to(DType::Float32);
    Tensor beta = bn_state.at("bias").to(DType::Float32);
    Tensor running_mean = bn_state.at("running_mean").to(DType::Float32);
    Tensor running_var = bn_state.at("running_var").to(DType::Float32);

    // Read epsilon from the source module
    double eps = 1e-5;
    if (auto* bn2d = dynamic_cast<const BatchNorm2d*>(&fp_bn)) {
        eps = bn2d->eps();
    }

    // Fold BN into conv weights:
    // w_folded = gamma / sqrt(var + eps) * w_conv
    // b_folded = gamma / sqrt(var + eps) * (b_conv - mean) + beta
    Tensor sqrt_var = sqrt(running_var + eps);
    Tensor bn_scale = gamma / sqrt_var;  // [C]

    // Get conv weights and bias
    auto conv_state = fp_conv.state_dict();
    Tensor fp_weight = conv_state.at("weight").to(DType::Float32);  // [out_channels, in_channels/groups, kH, kW]
    std::optional<Tensor> fp_bias;
    if (conv_state.find("bias") != conv_state.end()) {
        fp_bias = conv_state.at("bias").to(DType::Float32);
    }

    // Fold BN scale into conv weights
    // weight shape: [out_channels, in_channels/groups, kH, kW]
    auto weight_shape = fp_weight.shape();
    int64_t out_channels = weight_shape[0];

    // Scale each output channel's weights by bn_scale[c]
    Tensor folded_weight = fp_weight.clone();
    Tensor folded_weight_cpu = folded_weight;
    if (folded_weight_cpu.device() != Device::cpu()) {
        folded_weight_cpu = folded_weight_cpu.to(Device::cpu());
    }
    Tensor bn_scale_cpu = bn_scale;
    if (bn_scale_cpu.device() != Device::cpu()) {
        bn_scale_cpu = bn_scale_cpu.to(Device::cpu());
    }

    float* w_data = folded_weight_cpu.data<float>();
    const float* s_data = bn_scale_cpu.data<const float>();
    int64_t channel_size = folded_weight_cpu.numel() / out_channels;
    for (int64_t c = 0; c < out_channels; ++c) {
        float s = s_data[c];
        for (int64_t i = 0; i < channel_size; ++i) {
            w_data[c * channel_size + i] *= s;
        }
    }

    // Fold BN into bias
    // b_folded = bn_scale * (b_conv - running_mean) + beta
    Tensor folded_bias({out_channels}, DType::Float32, Device::cpu());
    float* fb_data = folded_bias.data<float>();
    Tensor running_mean_cpu = running_mean;
    Tensor beta_cpu = beta;
    if (running_mean_cpu.device() != Device::cpu()) {
        running_mean_cpu = running_mean_cpu.to(Device::cpu());
    }
    if (beta_cpu.device() != Device::cpu()) {
        beta_cpu = beta_cpu.to(Device::cpu());
    }
    const float* mean_data = running_mean_cpu.data<const float>();
    const float* beta_data = beta_cpu.data<const float>();

    for (int64_t c = 0; c < out_channels; ++c) {
        float conv_bias = 0.0f;
        if (fp_bias.has_value()) {
            Tensor bias_cpu = *fp_bias;
            if (bias_cpu.device() != Device::cpu()) {
                bias_cpu = bias_cpu.to(Device::cpu());
            }
            conv_bias = bias_cpu.data<const float>()[c];
        }
        fb_data[c] = s_data[c] * (conv_bias - mean_data[c]) + beta_data[c];
    }

    // Quantize folded weights
    auto weight_observer = qconfig.create_weight_observer();
    weight_observer->observe(folded_weight_cpu);
    auto weight_qparams = weight_observer->calculate_qparams(
        qconfig.weight_dtype(),
        qconfig.weight_scheme()
    );

    // Extract Conv2d parameters
    int64_t in_channels = weight_shape[1] * fp_conv.groups();
    int64_t kernel_h = weight_shape[2];

    // Create the fused layer - BN scale/bias are identity since folded into weights
    Tensor ones_scale({out_channels}, DType::Float32, Device::cpu());
    Tensor zeros_bias({out_channels}, DType::Float32, Device::cpu());
    ones_scale.fill_(1.0f);
    zeros_bias.fill_(0.0f);

    auto fused = std::make_shared<QuantizedConv2dBnReLU>(
        in_channels, out_channels, kernel_h,
        fp_conv.stride_h(), fp_conv.padding_h(),
        fp_conv.dilation_h(), fp_conv.groups(),
        weight_qparams,
        ones_scale,   // BN already folded into weights, so scale = 1
        zeros_bias    // BN already folded into bias, so bias = 0
    );

    // Set quantized weights with folded BN
    QuantizedTensor q_weight = quantize_tensor(folded_weight_cpu, weight_qparams);
    fused->conv_->set_weight(q_weight);
    fused->conv_->set_bias(folded_bias);

    return fused;
}

// ============================================================================
// QuantizedEmbedding
// ============================================================================

QuantizedEmbedding::QuantizedEmbedding(
    int64_t num_embeddings,
    int64_t embedding_dim,
    QuantizationParams weight_qparams,
    int64_t padding_idx
) : num_embeddings_(num_embeddings),
    embedding_dim_(embedding_dim),
    padding_idx_(padding_idx),
    weight_(Tensor({num_embeddings, embedding_dim}, DType::Int8, Device::cpu()),
            std::move(weight_qparams)) {}

auto QuantizedEmbedding::forward_impl(const Variable& input) -> Variable {
    // JIT-R066: forward_quantized is a raw host loop over `indices` (this
    // call's actual, per-call-varying input) with zero dispatch() calls —
    // same hazard class as the raw quantize_tensor/dequantize_tensor
    // functions, just implemented inline here instead of via those shared
    // helpers. A trace would silently freeze the looked-up/dequantized
    // output as a constant regardless of the real (replayed) indices.
    if (::tenzor::jit::Tracer::get_instance().is_tracing()) {
        throw std::runtime_error(
            "QuantizedEmbedding::forward_impl: cannot be JIT-traced (see "
            "JIT-R066). Call outside jit.compile()/jit.trace().");
    }
    Tensor output = forward_quantized(input.tensor());
    return Variable(output, false);  // Quantized inference, no grad
}

auto QuantizedEmbedding::forward_quantized(const Tensor& indices) -> Tensor {
    // Look up quantized rows and dequantize them on-the-fly
    int64_t num_indices = indices.numel();
    Tensor output({num_indices, embedding_dim_}, DType::Float32, Device::cpu());

    // Get quantization parameters on CPU
    Tensor scale_cpu = weight_.params().scale;
    Tensor zp_cpu = weight_.params().zero_point;
    if (scale_cpu.device() != Device::cpu()) scale_cpu = scale_cpu.to(Device::cpu());
    if (zp_cpu.device() != Device::cpu()) zp_cpu = zp_cpu.to(Device::cpu());

    float scale = scale_cpu.data<float>()[0];
    int32_t zp = zp_cpu.data<int32_t>()[0];

    Tensor weight_cpu = weight_.data();
    if (weight_cpu.device() != Device::cpu()) weight_cpu = weight_cpu.to(Device::cpu());
    const int8_t* w_data = weight_cpu.data<int8_t>();

    Tensor indices_cpu = indices;
    if (indices_cpu.device() != Device::cpu()) indices_cpu = indices_cpu.to(Device::cpu());
    const int64_t* idx_data = indices_cpu.data<int64_t>();

    float* out_data = output.data<float>();

    for (int64_t i = 0; i < num_indices; ++i) {
        int64_t idx = idx_data[i];
        if (idx < 0) idx += num_embeddings_;
        if (idx < 0 || idx >= num_embeddings_) {
            throw std::out_of_range("QuantizedEmbedding: index " +
                std::to_string(idx_data[i]) + " out of range [0, " +
                std::to_string(num_embeddings_) + ")");
        }

        const int8_t* row = w_data + idx * embedding_dim_;
        float* out_row = out_data + i * embedding_dim_;

        // Dequantize: fp = (q - zp) * scale
        for (int64_t j = 0; j < embedding_dim_; ++j) {
            out_row[j] = (static_cast<float>(row[j]) - zp) * scale;
        }

        // Zero out padding index
        if (idx == padding_idx_) {
            for (int64_t j = 0; j < embedding_dim_; ++j) {
                out_row[j] = 0.0f;
            }
        }
    }

    // Transfer to original device if needed
    if (indices.device() != Device::cpu()) {
        output = output.to(indices.device());
    }

    return output;
}

auto QuantizedEmbedding::set_weight(const QuantizedTensor& weights) -> void {
    weight_ = weights;
}

auto QuantizedEmbedding::from_float(Module& fp_embedding,
                                     [[maybe_unused]] const QConfig& qconfig)
    -> std::shared_ptr<QuantizedEmbedding> {
    // Extract weight from the embedding module
    auto params = fp_embedding.named_parameters();
    Tensor weight;
    bool found = false;
    for (auto& [name, var] : params) {
        if (name == "weight") { weight = var->tensor(); found = true; break; }
    }
    if (!found) {
        throw std::runtime_error("QuantizedEmbedding::from_float: module has no 'weight' parameter");
    }
    auto shape = weight.shape();
    int64_t num_embeddings = shape[0];
    int64_t embedding_dim = shape[1];

    // Quantize weight
    auto weight_cpu = (weight.device() == Device::cpu()) ? weight : weight.to(Device::cpu());
    auto q_weight = quantize_per_tensor_symmetric(weight_cpu);

    auto result = std::make_shared<QuantizedEmbedding>(
        num_embeddings, embedding_dim, q_weight.params());
    result->set_weight(q_weight);
    return result;
}

// ============================================================================
// QuantizedLSTM
// ============================================================================

QuantizedLSTM::QuantizedLSTM(
    int64_t input_size,
    int64_t hidden_size,
    int64_t num_layers,
    bool bias,
    bool batch_first,
    bool bidirectional,
    float dropout,
    QuantizationParams weight_qparams
) : input_size_(input_size),
    hidden_size_(hidden_size),
    num_layers_(num_layers),
    bias_(bias),
    batch_first_(batch_first),
    bidirectional_(bidirectional),
    dropout_(dropout) {

    if (dropout_ < 0.0f || dropout_ >= 1.0f) {
        throw std::invalid_argument(
            "QuantizedLSTM: dropout must be in [0, 1), got " + std::to_string(dropout_));
    }

    int64_t num_directions = bidirectional ? 2 : 1;
    layers_.reserve(num_layers * num_directions);

    for (int64_t layer = 0; layer < num_layers_ * num_directions; ++layer) {
        int64_t in_sz = (layer == 0) ? input_size : hidden_size * num_directions;
        // 4 gates: input, forget, cell, output
        QuantizedTensor w_ih(
            Tensor({4 * hidden_size, in_sz}, DType::Int8, Device::cpu()),
            weight_qparams);
        QuantizedTensor w_hh(
            Tensor({4 * hidden_size, hidden_size}, DType::Int8, Device::cpu()),
            weight_qparams);

        LayerWeights lw{std::move(w_ih), std::move(w_hh), std::nullopt, std::nullopt};
        if (bias) {
            lw.bias_ih = zeros({4 * hidden_size}, DType::Float32, Device::cpu());
            lw.bias_hh = zeros({4 * hidden_size}, DType::Float32, Device::cpu());
        }
        layers_.push_back(std::move(lw));
    }
}

auto QuantizedLSTM::forward_impl(const Variable& input) -> Variable {
    // Default: zero initial state
    auto shape = input.shape();
    int64_t batch = batch_first_ ? shape[0] : shape[1];
    int64_t num_directions = bidirectional_ ? 2 : 1;

    // JIT-R103: default initial state must live on the input's actual
    // device — zeros()'s default device is CPU, and h0/c0 get mixed
    // directly with the real (possibly non-CPU) input/weights below.
    auto dev = input.tensor().device();
    auto h0 = Variable(zeros({num_layers_ * num_directions, batch, hidden_size_},
                              DType::Float32, dev), false);
    auto c0 = Variable(zeros({num_layers_ * num_directions, batch, hidden_size_},
                              DType::Float32, dev), false);

    auto [output, hn, cn] = forward_with_state(input, h0, c0);
    return output;
}

auto QuantizedLSTM::forward_with_state(const Variable& input,
                                        const Variable& h0, const Variable& c0)
    -> std::tuple<Variable, Variable, Variable> {
    // Multi-layer (optionally bidirectional) quantized LSTM, mirroring
    // torch.ao.nn.quantized.dynamic.LSTM:
    //   - INT8 weights for every gate, dequantized per-layer before the
    //     per-timestep matmuls.
    //   - Gate non-linearities (sigmoid/tanh) run in FP32 to preserve
    //     accuracy.
    //   - Inter-layer dropout is applied to the full sequence output of
    //     each layer except the last, only when training_ && dropout_ > 0.
    //   - For bidirectional layers, the forward and reverse per-direction
    //     outputs are concatenated along the feature axis and fed to the
    //     next layer.
    //   - Final per-(layer, direction) (h, c) states are stacked along
    //     dim 0 in PyTorch ordering.

    if (input.tensor().dtype() != DType::Float32) {
        throw std::invalid_argument(
            "QuantizedLSTM: input must be Float32; got dtype " +
            std::to_string(static_cast<int>(input.tensor().dtype())));
    }

    // JIT-R058a: `input` is the traced function's actual (live, per-call-
    // varying) parameter — permuting it via the raw Tensor member function
    // would disconnect everything downstream from the traced graph entirely
    // (the whole layer's output would silently freeze as a giant constant
    // on replay). traced_permute records this as a real graph node instead.
    auto inp = input.tensor();
    if (batch_first_) {
        inp = traced_permute(inp, {1, 0, 2});  // -> [seq_len, batch, features]
    }

    int64_t seq_len = inp.shape()[0];
    int64_t num_directions = bidirectional_ ? 2 : 1;

    // Tensor that the next layer consumes. Initial input is the (already
    // permuted) FP32 sequence. Subsequent layers consume the previous
    // layer's [seq_len, batch, hidden_size * num_directions] output.
    Tensor current_input = inp;

    // Final hidden / cell states per (layer, direction), in PyTorch order:
    //   index = layer * num_directions + direction
    std::vector<Tensor> final_h;
    std::vector<Tensor> final_c;
    final_h.reserve(static_cast<size_t>(num_layers_ * num_directions));
    final_c.reserve(static_cast<size_t>(num_layers_ * num_directions));

    for (int64_t layer = 0; layer < num_layers_; ++layer) {
        // Per-direction full-sequence outputs of this layer.
        std::vector<Tensor> direction_outputs;
        direction_outputs.reserve(static_cast<size_t>(num_directions));

        for (int64_t dir = 0; dir < num_directions; ++dir) {
            int64_t idx = layer * num_directions + dir;
            auto& lw = layers_[idx];

            // Dequantize INT8 weights for this layer/direction once per
            // forward (amortised across all timesteps). dequantize()'s raw
            // pointer loop is untraced (JIT-R066) but the WEIGHT is
            // genuinely constant across calls of this compiled layer
            // instance, so freezing it (and its permute below) as a
            // trace-time constant is correct JIT behavior, not a bug.
            Tensor w_ih = lw.weight_ih.dequantize();
            Tensor w_hh = lw.weight_hh.dequantize();
            // JIT-R103: quantized weights stay wherever they were
            // constructed/quantized until moved, unlike current_input/inp
            // which are on the actual call's device — a lazily-placed
            // module (constructed on CPU, called with a GPU input) hit a
            // device-mismatch dispatch error at the matmul below.
            if (w_ih.device() != current_input.device()) w_ih = w_ih.to(current_input.device());
            if (w_hh.device() != current_input.device()) w_hh = w_hh.to(current_input.device());
            Tensor w_ih_t = w_ih.permute({1, 0});
            Tensor w_hh_t = w_hh.permute({1, 0});

            // Initial states for this (layer, direction): h0/c0 are live,
            // call-varying function parameters — must use traced_slice/
            // traced_squeeze so a replay with different initial state
            // actually uses it, instead of freezing the trace-time slice.
            // contiguous() so we don't alias-mutate the caller's h0/c0
            // storage (a metadata-only op — its own output is a fresh copy,
            // not a distinct logical value, so it's fine to leave untraced).
            Tensor h = traced_squeeze(traced_slice(h0.tensor(), 0, idx, idx + 1), 0).contiguous();
            Tensor c = traced_squeeze(traced_slice(c0.tensor(), 0, idx, idx + 1), 0).contiguous();

            std::vector<Tensor> step_outputs;
            step_outputs.reserve(static_cast<size_t>(seq_len));

            int64_t t_start = (dir == 0) ? 0 : seq_len - 1;
            int64_t t_end   = (dir == 0) ? seq_len : -1;
            int64_t t_step  = (dir == 0) ? 1 : -1;

            for (int64_t t = t_start; t != t_end; t += t_step) {
                auto x_t = traced_squeeze(traced_slice(current_input, 0, t, t + 1), 0);

                // gates = x_t @ w_ih^T + h @ w_hh^T + bias
                auto gates = matmul(x_t, w_ih_t);
                gates = gates + matmul(h, w_hh_t);
                if (lw.bias_ih) {
                    Tensor b = (lw.bias_ih->device() != x_t.device())
                        ? lw.bias_ih->to(x_t.device()) : *lw.bias_ih;
                    gates = gates + b;
                }
                if (lw.bias_hh) {
                    Tensor b = (lw.bias_hh->device() != x_t.device())
                        ? lw.bias_hh->to(x_t.device()) : *lw.bias_hh;
                    gates = gates + b;
                }

                // Split gates: [batch, 4*hidden] -> i, f, g, o. `gates` is a
                // live, per-call-varying dispatched value — traced_slice
                // keeps each split tied to it instead of freezing.
                auto i_gate = sigmoid(traced_slice(gates, 1, 0,                 hidden_size_));
                auto f_gate = sigmoid(traced_slice(gates, 1, hidden_size_,     2 * hidden_size_));
                auto g_gate = tanh   (traced_slice(gates, 1, 2 * hidden_size_, 3 * hidden_size_));
                auto o_gate = sigmoid(traced_slice(gates, 1, 3 * hidden_size_, 4 * hidden_size_));

                c = f_gate * c + i_gate * g_gate;
                h = o_gate * tanh(c);

                step_outputs.push_back(traced_unsqueeze(h, 0));  // [1, batch, hidden]
            }

            // Reverse-direction outputs were appended in reverse time order;
            // flip them so dir_output is in forward time order.
            if (dir == 1) {
                std::reverse(step_outputs.begin(), step_outputs.end());
            }

            // [seq_len, batch, hidden]
            Tensor dir_output = cat(step_outputs, 0);
            direction_outputs.push_back(std::move(dir_output));

            final_h.push_back(h);
            final_c.push_back(c);
        }

        // Combine direction outputs for this layer along the feature axis,
        // producing [seq_len, batch, hidden * num_directions].
        Tensor layer_output = (num_directions == 1)
            ? std::move(direction_outputs[0])
            : cat(direction_outputs, /*dim=*/2);

        // Inter-layer dropout (skip after final layer). Inference-only
        // module, so we use functional::dropout under the layer's own
        // training flag — with is_training()=false this is identity,
        // matching PyTorch's behaviour for AO modules.
        if (dropout_ > 0.0f && layer < num_layers_ - 1 && is_training()) {
            Variable v_in(layer_output, /*requires_grad=*/false);
            Variable v_out = ::tenzor::nn::functional::dropout(
                v_in, static_cast<double>(dropout_), /*training=*/true);
            layer_output = v_out.tensor();
        }

        current_input = std::move(layer_output);
    }

    // Stack final (h, c) per (layer, direction) along dim 0. final_h/final_c
    // entries are live per-call values (each layer/direction's actual final
    // state) — traced_unsqueeze keeps h_n/c_n tied to them.
    std::vector<Tensor> h_expanded;
    std::vector<Tensor> c_expanded;
    h_expanded.reserve(final_h.size());
    c_expanded.reserve(final_c.size());
    for (auto& t : final_h) h_expanded.push_back(traced_unsqueeze(t, 0));
    for (auto& t : final_c) c_expanded.push_back(traced_unsqueeze(t, 0));
    Tensor h_n = cat(h_expanded, 0);
    Tensor c_n = cat(c_expanded, 0);

    Tensor output = current_input;
    if (batch_first_) {
        output = traced_permute(output, {1, 0, 2});
    }

    return {Variable(output, false), Variable(h_n, false), Variable(c_n, false)};
}

auto QuantizedLSTM::from_float(Module& fp_lstm, [[maybe_unused]] const QConfig& qconfig)
    -> std::shared_ptr<QuantizedLSTM> {
    // nn::LSTM composes stacked nn::LSTMCell submodules registered as
    // "forward_cell_<i>" / "backward_cell_<i>" (each cell's weight_ih/
    // weight_hh are themselves nested Linear submodules) — NOT flat
    // PyTorch-style "weight_ih_l<i>" parameters. Looking these up by the
    // ACTUAL registered submodule names/types (instead of matching
    // named_parameters() strings against a naming convention this
    // codebase's nn::LSTM never produces) is required for from_float to
    // find anything at all; previously every call threw immediately.
    const auto& submodules = fp_lstm.get_submodules();
    auto get_cell = [&](const std::string& name) -> std::shared_ptr<LSTMCell> {
        auto it = submodules.find(name);
        if (it == submodules.end()) return nullptr;
        return std::dynamic_pointer_cast<LSTMCell>(it->second);
    };

    auto fwd0 = get_cell("forward_cell_0");
    if (!fwd0) {
        throw std::runtime_error(
            "QuantizedLSTM::from_float: fp_lstm has no forward_cell_0 "
            "submodule of type LSTMCell (expected an nn::LSTM instance)");
    }
    int64_t hidden_size = fwd0->weight_ih()->weight()->shape()[0] / 4;
    int64_t input_size = fwd0->weight_ih()->weight()->shape()[1];
    bool has_bias = fwd0->weight_ih()->has_bias();

    // Count layers
    int64_t num_layers = 0;
    while (get_cell("forward_cell_" + std::to_string(num_layers))) {
        num_layers++;
    }
    bool bidirectional = get_cell("backward_cell_0") != nullptr;

    auto result = std::make_shared<QuantizedLSTM>(
        input_size, hidden_size, num_layers, has_bias, true, bidirectional);

    int64_t num_directions = bidirectional ? 2 : 1;
    for (int64_t layer = 0; layer < num_layers; ++layer) {
        for (int64_t dir = 0; dir < num_directions; ++dir) {
            int64_t idx = layer * num_directions + dir;
            auto cell = (dir == 0)
                ? get_cell("forward_cell_" + std::to_string(layer))
                : get_cell("backward_cell_" + std::to_string(layer));
            if (!cell) {
                throw std::runtime_error(
                    "QuantizedLSTM::from_float: missing cell for layer " +
                    std::to_string(layer) + " direction " + std::to_string(dir));
            }

            auto w_ih = cell->weight_ih()->weight()->tensor();
            auto w_hh = cell->weight_hh()->weight()->tensor();
            auto w_ih_cpu = (w_ih.device() == Device::cpu()) ? w_ih : w_ih.to(Device::cpu());
            auto w_hh_cpu = (w_hh.device() == Device::cpu()) ? w_hh : w_hh.to(Device::cpu());

            result->layers_[idx].weight_ih = quantize_per_tensor_symmetric(w_ih_cpu);
            result->layers_[idx].weight_hh = quantize_per_tensor_symmetric(w_hh_cpu);

            // Query each Linear's own has_bias() independently instead of
            // assuming weight_ih/weight_hh match (LSTMCell's weight_hh is
            // constructed with bias=false unconditionally — see
            // LSTMCell::LSTMCell — so bias()->tensor() would null-deref if
            // gated on `has_bias` alone; GRUCell's weight_hh does carry a
            // bias, so this is a no-op there beyond the extra safety).
            if (has_bias && cell->weight_ih()->has_bias()) {
                result->layers_[idx].bias_ih = cell->weight_ih()->bias()->tensor();
            }
            if (cell->weight_hh()->has_bias()) {
                result->layers_[idx].bias_hh = cell->weight_hh()->bias()->tensor();
            }
        }
    }

    return result;
}

// ============================================================================
// QuantizedConv3d
// ============================================================================

QuantizedConv3d::QuantizedConv3d(
    int64_t in_channels,
    int64_t out_channels,
    std::array<int64_t, 3> kernel_size,
    std::array<int64_t, 3> stride,
    std::array<int64_t, 3> padding,
    std::array<int64_t, 3> dilation,
    int64_t groups,
    QuantizationParams weight_qparams
) : in_channels_(in_channels),
    out_channels_(out_channels),
    kernel_size_(kernel_size),
    stride_(stride),
    padding_(padding),
    dilation_(dilation),
    groups_(groups),
    weight_(Tensor({out_channels, in_channels / groups,
                    kernel_size[0], kernel_size[1], kernel_size[2]},
                   DType::Int8, Device::cpu()),
            std::move(weight_qparams)) {}

auto QuantizedConv3d::forward_impl(const Variable& input) -> Variable {
    // Quantized inference contract:
    //   - Float32 inputs are quantised per-tensor symmetric using the
    //     observer-derived qparams (matches QuantizedConv2d).
    //   - Int8 inputs are already in the quantised domain; wrap them in a
    //     QuantizedTensor with unit scale / zero zp so dequantize() is the
    //     identity on values and we land back in FP32 for the conv.
    //   - Any other dtype is a contract violation and we throw.
    const DType in_dtype = input.tensor().dtype();
    if (in_dtype != DType::Float32 && in_dtype != DType::Int8) {
        throw std::invalid_argument(
            "QuantizedConv3d: input dtype must be Float32 or Int8; got " +
            std::to_string(static_cast<int>(in_dtype)));
    }

    // JIT-R066: see QuantizedConv2d::forward_impl — the INT8 conv kernel
    // path is a raw host computation with no dispatch()-visible lineage
    // back to the input, so a trace would silently freeze the output as a
    // constant regardless of the real (replayed) input.
    if (::tenzor::jit::Tracer::get_instance().is_tracing()) {
        throw std::runtime_error(
            "QuantizedConv3d::forward_impl: cannot be JIT-traced (see "
            "JIT-R066). Call outside jit.compile()/jit.trace().");
    }

    QuantizedTensor q_input = (in_dtype == DType::Float32)
        ? quantize_per_tensor_symmetric(input.tensor())
        : QuantizedTensor(
            input.tensor(),
            QuantizationParams(
                ones({1}, DType::Float32, input.tensor().device()),
                zeros({1}, DType::Int32, input.tensor().device()),
                QuantDType::INT8,
                QuantizationScheme::PerTensorSymmetric));

    Tensor output = forward_quantized(q_input);
    return Variable(output, /*requires_grad=*/false);
}

auto QuantizedConv3d::forward_quantized(const QuantizedTensor& input) -> Tensor {
    // INT8 quantized 3D convolution.
    //
    // Strategy: dequantize activations and weights to FP32, then dispatch
    // through nn::functional::conv3d. The functional path runs the
    // backend's production conv3d kernel — cblas_sgemm-on-im2col on CPU
    // (oneDNN where available), cuDNN implicit-GEMM / Winograd on CUDA,
    // MIOpen on ROCm, native compute shaders on Vulkan — so there's no
    // benefit to re-rolling a naive 7-loop nest here.
    //
    // Per-channel symmetric quantisation along the output-channel axis is
    // handled inside QuantizedTensor::dequantize() (the scale vector is
    // broadcast against the weight tensor), so this path covers both
    // PerTensorSymmetric and PerChannelSymmetric weight schemes.
    //
    // A truly-INT8 path on CUDA (cublasGemmEx with CUDA_R_8I × CUDA_R_8I
    // → CUDA_R_32I) would dispatch through OpId::QuantizedConv3d if/when
    // that op id is added to the registry; until then this is the
    // canonical numerically-correct entry point.

    const auto& w_params = weight_.params();
    if (w_params.scheme != QuantizationScheme::PerTensorSymmetric &&
        w_params.scheme != QuantizationScheme::PerChannelSymmetric) {
        throw std::invalid_argument(
            "QuantizedConv3d: weight must use symmetric quantisation "
            "(PerTensorSymmetric or PerChannelSymmetric)");
    }

    Tensor fp_input  = input.dequantize();
    Tensor fp_weight = weight_.dequantize();

    if (fp_weight.device() != fp_input.device()) {
        fp_weight = fp_weight.to(fp_input.device());
    }

    Variable v_input (fp_input,  /*requires_grad=*/false);
    Variable v_weight(fp_weight, /*requires_grad=*/false);

    std::optional<Variable> v_bias;
    if (bias_) {
        Tensor b = *bias_;
        if (b.dtype() != DType::Float32) {
            b = b.to(DType::Float32);
        }
        if (b.device() != fp_input.device()) {
            b = b.to(fp_input.device());
        }
        v_bias = Variable(b, /*requires_grad=*/false);
    }

    Variable v_out = ::tenzor::nn::functional::conv3d(
        v_input, v_weight, v_bias,
        std::make_tuple(stride_[0],   stride_[1],   stride_[2]),
        std::make_tuple(padding_[0],  padding_[1],  padding_[2]),
        std::make_tuple(dilation_[0], dilation_[1], dilation_[2]),
        groups_);

    return v_out.tensor();
}

auto QuantizedConv3d::set_weight(const QuantizedTensor& weights) -> void {
    weight_ = weights;
}

auto QuantizedConv3d::set_bias(const Tensor& bias) -> void {
    bias_ = bias;
}

auto QuantizedConv3d::from_float(Module& fp_conv3d, [[maybe_unused]] const QConfig& qconfig)
    -> std::shared_ptr<QuantizedConv3d> {
    auto params = fp_conv3d.named_parameters();

    Tensor weight;
    bool found_weight = false;
    for (auto& [name, var] : params) {
        if (name == "weight") { weight = var->tensor(); found_weight = true; break; }
    }
    if (!found_weight) {
        throw std::runtime_error("QuantizedConv3d::from_float: module has no 'weight' parameter");
    }
    auto shape = weight.shape();
    int64_t out_ch = shape[0];
    int64_t in_ch_per_group = shape[1];
    std::array<int64_t, 3> ks = {shape[2], shape[3], shape[4]};

    // Extract stride/padding/dilation/groups from the source Conv3d module
    std::array<int64_t, 3> stride = {1, 1, 1};
    std::array<int64_t, 3> padding = {0, 0, 0};
    std::array<int64_t, 3> dilation = {1, 1, 1};
    int64_t groups = 1;
    if (auto* conv3d = dynamic_cast<nn::Conv3d*>(&fp_conv3d)) {
        stride = {conv3d->stride(), conv3d->stride(), conv3d->stride()};
        padding = {conv3d->padding(), conv3d->padding(), conv3d->padding()};
        dilation = {conv3d->dilation(), conv3d->dilation(), conv3d->dilation()};
        groups = conv3d->groups();
        in_ch_per_group = in_ch_per_group * groups;  // Convert per-group to total in_channels
    }

    auto weight_cpu = (weight.device() == Device::cpu()) ? weight : weight.to(Device::cpu());
    auto q_weight = quantize_per_tensor_symmetric(weight_cpu);

    auto result = std::make_shared<QuantizedConv3d>(
        in_ch_per_group, out_ch, ks, stride,
        padding, dilation, groups,
        q_weight.params());
    result->set_weight(q_weight);

    for (auto& [name, var] : params) {
        if (name == "bias") { result->set_bias(var->tensor()); break; }
    }

    return result;
}

// ============================================================================
// QuantizedMultiheadAttention
// ============================================================================

QuantizedMultiheadAttention::QuantizedMultiheadAttention(
    int64_t embed_dim,
    int64_t num_heads,
    QuantizationParams weight_qparams,
    [[maybe_unused]] bool bias,
    float dropout
) : embed_dim_(embed_dim),
    num_heads_(num_heads),
    head_dim_(embed_dim / num_heads),
    dropout_(dropout) {

    if (embed_dim % num_heads != 0) {
        throw std::runtime_error("QuantizedMultiheadAttention: embed_dim must be divisible by num_heads");
    }

    q_proj_ = std::make_shared<QuantizedLinear>(embed_dim, embed_dim, weight_qparams);
    k_proj_ = std::make_shared<QuantizedLinear>(embed_dim, embed_dim, weight_qparams);
    v_proj_ = std::make_shared<QuantizedLinear>(embed_dim, embed_dim, weight_qparams);
    out_proj_ = std::make_shared<QuantizedLinear>(embed_dim, embed_dim, weight_qparams);
}

auto QuantizedMultiheadAttention::forward_impl(const Variable& input) -> Variable {
    // Self-attention: Q=K=V=input
    auto [output, weights] = forward_qkv(input, input, input);
    return output;
}

auto QuantizedMultiheadAttention::forward_qkv(
    const Variable& query, const Variable& key,
    const Variable& value, const Variable& attn_mask)
    -> std::pair<Variable, Variable> {

    auto shape = query.shape();
    int64_t seq_len = shape[0];
    int64_t batch = shape[1];

    // INT8 linear projections, dequantized to FP32
    auto q = q_proj_->forward_impl(query);
    auto k = k_proj_->forward_impl(key);
    auto v = v_proj_->forward_impl(value);

    // Reshape for multi-head: [seq, batch, embed] -> [seq, batch, heads, head_dim] -> [batch*heads, seq, head_dim]
    auto reshape_for_heads = [&](const Variable& x) -> Tensor {
        auto t = x.tensor().reshape({seq_len, batch, num_heads_, head_dim_});
        t = t.permute({1, 2, 0, 3});  // [batch, heads, seq, head_dim]
        return t.reshape({batch * num_heads_, seq_len, head_dim_});
    };

    auto q_heads = reshape_for_heads(q);
    auto k_heads = reshape_for_heads(k);
    auto v_heads = reshape_for_heads(v);

    // Scaled dot-product attention in FP32
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    auto scores = matmul(q_heads, k_heads.permute({0, 2, 1})) * scale;

    if (attn_mask.is_initialized()) {
        scores = scores + attn_mask.tensor();
    }

    // softmax via autograd (no Tensor-level softmax exists)
    auto attn_weights = tenzor::softmax(Variable(scores, false), -1).tensor();
    auto context = matmul(attn_weights, v_heads);

    // Reshape back: [batch*heads, seq, head_dim] -> [seq, batch, embed]
    context = context.reshape({batch, num_heads_, seq_len, head_dim_});
    context = context.permute({2, 0, 1, 3}).reshape({seq_len, batch, embed_dim_});

    // Output projection (INT8)
    auto output = out_proj_->forward_impl(Variable(context, false));

    return {output, Variable(attn_weights.reshape({batch, num_heads_, seq_len, seq_len}), false)};
}

auto QuantizedMultiheadAttention::from_float(Module& fp_mha,
                                              [[maybe_unused]] const QConfig& qconfig)
    -> std::shared_ptr<QuantizedMultiheadAttention> {
    auto params = fp_mha.named_parameters();

    auto find_param = [&](const std::string& name) -> Tensor* {
        for (auto& [pname, var] : params) {
            if (pname == name) return &var->tensor();
        }
        return nullptr;
    };

    // Determine embed_dim and num_heads from q_proj weight shape
    Tensor* q_weight_ptr = find_param("q_proj.weight");
    if (!q_weight_ptr) q_weight_ptr = find_param("in_proj_weight");
    if (!q_weight_ptr) {
        throw std::runtime_error("QuantizedMultiheadAttention::from_float: cannot find projection weights");
    }

    auto q_weight = *q_weight_ptr;
    int64_t embed_dim = q_weight.shape()[1];

    // Determine num_heads (may need to be passed as parameter in practice)
    // Default: try head_dim=64
    int64_t num_heads = embed_dim / 64;
    if (num_heads < 1) num_heads = 1;

    auto weight_cpu = (q_weight.device() == Device::cpu()) ? q_weight : q_weight.to(Device::cpu());
    auto q_params = quantize_per_tensor_symmetric(weight_cpu).params();

    auto result = std::make_shared<QuantizedMultiheadAttention>(
        embed_dim, num_heads, q_params);

    // Quantize individual projection weights
    auto set_proj = [&](const std::string& name, std::shared_ptr<QuantizedLinear>& proj) {
        if (auto* w = find_param(name + ".weight")) {
            auto w_cpu = (w->device() == Device::cpu()) ? *w : w->to(Device::cpu());
            proj->set_weight(quantize_per_tensor_symmetric(w_cpu));
        }
        if (auto* b = find_param(name + ".bias")) {
            proj->set_bias(*b);
        }
    };

    // PyTorch's nn.MultiheadAttention packs q/k/v into a single
    // in_proj_weight of shape [3*embed_dim, embed_dim] (with optional
    // in_proj_bias [3*embed_dim]); separate q_proj/k_proj/v_proj weights only
    // exist when projections are unpacked. If only the combined weight is
    // present, split it into thirds — otherwise q/k/v stay zero-initialised and
    // the attention output is identically zero.
    if (Tensor* in_w = find_param("in_proj_weight")) {
        Tensor in_w_cpu = (in_w->device() == Device::cpu()) ? *in_w : in_w->to(Device::cpu());
        Tensor* in_b = find_param("in_proj_bias");
        std::optional<Tensor> in_b_cpu;
        if (in_b) in_b_cpu = (in_b->device() == Device::cpu()) ? *in_b : in_b->to(Device::cpu());

        auto set_from_chunk = [&](int idx, std::shared_ptr<QuantizedLinear>& proj) {
            Tensor w_chunk = in_w_cpu.narrow(0, idx * embed_dim, embed_dim).contiguous();
            proj->set_weight(quantize_per_tensor_symmetric(w_chunk));
            if (in_b_cpu) {
                proj->set_bias(in_b_cpu->narrow(0, idx * embed_dim, embed_dim).contiguous());
            }
        };
        set_from_chunk(0, result->q_proj_);
        set_from_chunk(1, result->k_proj_);
        set_from_chunk(2, result->v_proj_);
    } else {
        set_proj("q_proj", result->q_proj_);
        set_proj("k_proj", result->k_proj_);
        set_proj("v_proj", result->v_proj_);
    }
    set_proj("out_proj", result->out_proj_);

    return result;
}

// ============================================================================
// QuantizedGRU
// ============================================================================

QuantizedGRU::QuantizedGRU(
    int64_t input_size,
    int64_t hidden_size,
    int64_t num_layers,
    bool bias,
    bool batch_first,
    bool bidirectional,
    [[maybe_unused]] QuantizationParams weight_qparams
) : input_size_(input_size),
    hidden_size_(hidden_size),
    num_layers_(num_layers),
    bias_(bias),
    batch_first_(batch_first),
    bidirectional_(bidirectional) {

    int64_t num_directions = bidirectional ? 2 : 1;
    layers_.reserve(num_layers * num_directions);

    for (int64_t layer = 0; layer < num_layers; ++layer) {
        for (int64_t dir = 0; dir < num_directions; ++dir) {
            int64_t layer_input_size = (layer == 0) ? input_size : hidden_size * num_directions;

            auto w_ih = zeros({3 * hidden_size, layer_input_size}, DType::Float32);
            auto w_hh = zeros({3 * hidden_size, hidden_size}, DType::Float32);

            LayerWeights lw{
                .weight_ih = quantize_per_tensor_symmetric(w_ih),
                .weight_hh = quantize_per_tensor_symmetric(w_hh),
                .bias_ih = bias ? std::optional{zeros({3 * hidden_size}, DType::Float32)} : std::nullopt,
                .bias_hh = bias ? std::optional{zeros({3 * hidden_size}, DType::Float32)} : std::nullopt,
            };
            layers_.push_back(std::move(lw));
        }
    }
}

auto QuantizedGRU::forward_impl(const Variable& input) -> Variable {
    // JIT-R103: default initial state must live on the input's actual
    // device — zeros()'s default device is CPU, and h0 gets mixed directly
    // with the real (possibly non-CPU) input/weights below.
    auto h0 = Variable(zeros({num_layers_ * (bidirectional_ ? 2 : 1),
                               input.tensor().shape()[batch_first_ ? 0 : 1],
                               hidden_size_}, input.tensor().dtype(),
                              input.tensor().device()), false);
    auto [output, h_n] = forward_with_state(input, h0);
    return output;
}

auto QuantizedGRU::forward_with_state(const Variable& input, const Variable& h0)
    -> std::pair<Variable, Variable> {

    // JIT-R058a: `input` is the traced function's live parameter — permute
    // it via traced_permute (not the raw Tensor member function) or
    // everything downstream loses lineage back to it entirely.
    auto x = input.tensor();
    if (batch_first_) {
        x = traced_permute(x, {1, 0, 2}); // [batch, seq, feat] -> [seq, batch, feat]
    }

    int64_t seq_len = x.shape()[0];
    int64_t num_directions = bidirectional_ ? 2 : 1;

    Tensor output;

    // Collect the final hidden state for each (layer, direction), indexed by
    // layer*num_directions+dir so the stacked result matches the standard
    // RNN h_n layout [num_layers*num_directions, batch, hidden].
    std::vector<Tensor> final_h(num_layers_ * num_directions);

    for (int64_t layer = 0; layer < num_layers_; ++layer) {
        Tensor layer_output;

        for (int64_t dir = 0; dir < num_directions; ++dir) {
            int64_t idx = layer * num_directions + dir;
            auto& lw = layers_[idx];

            // Dequantize weights to FP32. dequantize()'s raw pointer loop is
            // untraced (JIT-R066) but the weight is genuinely constant
            // across calls of this compiled layer instance, so freezing it
            // as a trace-time constant is correct JIT behavior, not a bug.
            auto w_ih = lw.weight_ih.dequantize();
            auto w_hh = lw.weight_hh.dequantize();
            // JIT-R103: quantized weights stay wherever they were
            // constructed/quantized until moved, unlike x/output which are
            // on the actual call's device — a lazily-placed module
            // (constructed on CPU, called with a GPU input) hit a
            // device-mismatch dispatch error at the matmul below.
            if (w_ih.device() != x.device()) w_ih = w_ih.to(x.device());
            if (w_hh.device() != x.device()) w_hh = w_hh.to(x.device());

            // h0 is a live, call-varying function parameter.
            auto h = traced_squeeze(traced_slice(h0.tensor(), 0, idx, idx + 1), 0);

            std::vector<Tensor> outputs;
            outputs.reserve(seq_len);

            int64_t t_start = (dir == 0) ? 0 : seq_len - 1;
            int64_t t_end = (dir == 0) ? seq_len : -1;
            int64_t t_step = (dir == 0) ? 1 : -1;

            Tensor current_input = (layer == 0) ? x : output;

            for (int64_t t = t_start; t != t_end; t += t_step) {
                auto x_t = traced_squeeze(traced_slice(current_input, 0, t, t + 1), 0);

                // gates_x = x_t @ w_ih^T + bias_ih
                auto gates_x = matmul(x_t, w_ih.permute({1, 0}));
                if (lw.bias_ih) {
                    Tensor b = (lw.bias_ih->device() != x_t.device())
                        ? lw.bias_ih->to(x_t.device()) : *lw.bias_ih;
                    gates_x = gates_x + b;
                }

                // gates_h = h @ w_hh^T + bias_hh
                auto gates_h = matmul(h, w_hh.permute({1, 0}));
                if (lw.bias_hh) {
                    Tensor b = (lw.bias_hh->device() != h.device())
                        ? lw.bias_hh->to(h.device()) : *lw.bias_hh;
                    gates_h = gates_h + b;
                }

                // Split: [batch, 3*hidden] -> r, z, n. gates_x/gates_h are
                // live dispatched values — traced_slice keeps the splits
                // tied to them instead of freezing.
                auto r_x = traced_slice(gates_x, 1, 0, hidden_size_);
                auto z_x = traced_slice(gates_x, 1, hidden_size_, 2 * hidden_size_);
                auto n_x = traced_slice(gates_x, 1, 2 * hidden_size_, 3 * hidden_size_);

                auto r_h = traced_slice(gates_h, 1, 0, hidden_size_);
                auto z_h = traced_slice(gates_h, 1, hidden_size_, 2 * hidden_size_);
                auto n_h = traced_slice(gates_h, 1, 2 * hidden_size_, 3 * hidden_size_);

                auto r_gate = sigmoid(r_x + r_h);
                auto z_gate = sigmoid(z_x + z_h);
                auto n_gate = tanh(n_x + r_gate * n_h);

                // JIT-R058a: ones_like(z_gate) on CPU is a raw allocate+fill
                // with ZERO dispatch() calls (ones()'s CPU path bypasses
                // dispatch entirely, unlike its GPU path) — invisible to the
                // tracer, so the later Sub's input registration falls back
                // to the fingerprint-dedup lookup and can alias it to an
                // unrelated, coincidentally-same-address earlier tensor
                // (confirmed: silently resolved to a stale n_x slice from
                // this same timestep, corrupting the whole gate chain).
                // `1.0 - z_gate` uses neg()+scalar-add instead, both of
                // which dispatch (OpId::Neg / OpId::Add) and are correctly
                // traced, avoiding the untraced constant entirely.
                h = (1.0 - z_gate) * n_gate + z_gate * h;
                outputs.push_back(traced_unsqueeze(h, 0));
            }

            if (dir == 1) {
                std::reverse(outputs.begin(), outputs.end());
            }

            // Record the final hidden state for this (layer, direction).
            final_h[idx] = traced_unsqueeze(h, 0);  // [1, batch, hidden]

            auto dir_output = cat(outputs, 0);
            if (dir == 0) {
                layer_output = dir_output;
            } else {
                layer_output = cat({layer_output, dir_output}, 2);
            }
        }

        output = layer_output;
    }

    if (batch_first_) {
        output = traced_permute(output, {1, 0, 2});
    }

    // Stack the per-(layer,direction) final hidden states into h_n with shape
    // [num_layers*num_directions, batch, hidden].
    auto h_n = cat(final_h, 0);

    return {Variable(output, false), Variable(h_n, false)};
}

auto QuantizedGRU::from_float(Module& fp_gru, [[maybe_unused]] const QConfig& qconfig)
    -> std::shared_ptr<QuantizedGRU> {
    // See QuantizedLSTM::from_float's comment: nn::GRU composes stacked
    // nn::GRUCell submodules ("forward_cell_<i>" / "backward_cell_<i>",
    // each with nested Linear weight_ih/weight_hh submodules) — not flat
    // PyTorch-style "weight_ih_l<i>" parameters.
    const auto& submodules = fp_gru.get_submodules();
    auto get_cell = [&](const std::string& name) -> std::shared_ptr<GRUCell> {
        auto it = submodules.find(name);
        if (it == submodules.end()) return nullptr;
        return std::dynamic_pointer_cast<GRUCell>(it->second);
    };

    auto fwd0 = get_cell("forward_cell_0");
    if (!fwd0) {
        throw std::runtime_error(
            "QuantizedGRU::from_float: fp_gru has no forward_cell_0 "
            "submodule of type GRUCell (expected an nn::GRU instance)");
    }
    int64_t hidden_size = fwd0->weight_ih()->weight()->shape()[0] / 3;
    int64_t input_size = fwd0->weight_ih()->weight()->shape()[1];
    bool has_bias = fwd0->weight_ih()->has_bias();

    int64_t num_layers = 0;
    while (get_cell("forward_cell_" + std::to_string(num_layers))) {
        num_layers++;
    }
    bool bidirectional = get_cell("backward_cell_0") != nullptr;

    auto result = std::make_shared<QuantizedGRU>(
        input_size, hidden_size, num_layers, has_bias, true, bidirectional);

    int64_t num_directions = bidirectional ? 2 : 1;
    for (int64_t layer = 0; layer < num_layers; ++layer) {
        for (int64_t dir = 0; dir < num_directions; ++dir) {
            int64_t idx = layer * num_directions + dir;
            auto cell = (dir == 0)
                ? get_cell("forward_cell_" + std::to_string(layer))
                : get_cell("backward_cell_" + std::to_string(layer));
            if (!cell) {
                throw std::runtime_error(
                    "QuantizedGRU::from_float: missing cell for layer " +
                    std::to_string(layer) + " direction " + std::to_string(dir));
            }

            auto w_ih = cell->weight_ih()->weight()->tensor();
            auto w_hh = cell->weight_hh()->weight()->tensor();
            auto w_ih_cpu = (w_ih.device() == Device::cpu()) ? w_ih : w_ih.to(Device::cpu());
            auto w_hh_cpu = (w_hh.device() == Device::cpu()) ? w_hh : w_hh.to(Device::cpu());

            result->layers_[idx].weight_ih = quantize_per_tensor_symmetric(w_ih_cpu);
            result->layers_[idx].weight_hh = quantize_per_tensor_symmetric(w_hh_cpu);

            // Query each Linear's own has_bias() independently instead of
            // assuming weight_ih/weight_hh match (LSTMCell's weight_hh is
            // constructed with bias=false unconditionally — see
            // LSTMCell::LSTMCell — so bias()->tensor() would null-deref if
            // gated on `has_bias` alone; GRUCell's weight_hh does carry a
            // bias, so this is a no-op there beyond the extra safety).
            if (has_bias && cell->weight_ih()->has_bias()) {
                result->layers_[idx].bias_ih = cell->weight_ih()->bias()->tensor();
            }
            if (cell->weight_hh()->has_bias()) {
                result->layers_[idx].bias_hh = cell->weight_hh()->bias()->tensor();
            }
        }
    }

    return result;
}

// ============================================================================
// QuantizedConv1d
// ============================================================================

QuantizedConv1d::QuantizedConv1d(
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    QuantizationParams weight_qparams,
    float bias_scale
) : in_channels_(in_channels),
    out_channels_(out_channels),
    kernel_size_(kernel_size),
    stride_(stride),
    padding_(padding),
    dilation_(dilation),
    groups_(groups),
    weight_(Tensor({out_channels, in_channels / groups, kernel_size},
                   DType::Int8, Device::cpu()), std::move(weight_qparams)),
    bias_scale_(bias_scale) {}

auto QuantizedConv1d::forward_impl(const Variable& input) -> Variable {
    // Audit item B.1: inference-only quantized layer; reject
    // requires_grad inputs instead of falsely claiming grad flow.
    if (input.requires_grad()) {
        throw std::runtime_error(
            "QuantizedConv1d: input has requires_grad=true but this is "
            "an inference-only quantized layer.  Use a non-quantized "
            "Conv1d for training.");
    }
    // JIT-R066: see QuantizedConv2d::forward_impl — the im2col + INT8
    // kernel path is a raw host computation with no dispatch()-visible
    // lineage back to the input, so a trace would silently freeze the
    // output as a constant regardless of the real (replayed) input.
    if (::tenzor::jit::Tracer::get_instance().is_tracing()) {
        throw std::runtime_error(
            "QuantizedConv1d::forward_impl: cannot be JIT-traced (see "
            "JIT-R066). Call outside jit.compile()/jit.trace().");
    }
    auto q_input = quantize_per_tensor_symmetric(input.tensor());
    Tensor output = forward_quantized(q_input);
    return Variable(output, /*requires_grad=*/false);
}

auto QuantizedConv1d::forward_quantized(const QuantizedTensor& input) -> Tensor {
    // Real INT8 Conv1d via im2col + quantized_linear_kernel.
    //
    // Pre-S20 this method dequantised both input and weight to FP32 and ran
    // a textbook scalar conv, then returned FP32. The audit calls that "a
    // lie" — the metadata says quantised but no INT8 arithmetic ever runs.
    //
    // Real INT8 path:
    //   1. Build INT8 im2col matrix per (batch, group) of shape
    //      [l_out, in_c_per_group * kernel_size], padding with `input_zp`.
    //   2. Reuse the signed-safe INT8 GEMM kernel from QuantizedLinear
    //      (`quantized_linear_kernel`, post-S3 fix uses _mm256/512_madd_epi16).
    //      That kernel does INT8×INT8 -> INT32 accumulation + zero-point
    //      correction + (input_scale * weight_scale / output_scale)
    //      dequantisation + bias, writing FP32 directly.
    //   3. Output is FP32 in the layer's natural output space (output_scale=1
    //      for the dequantised path; the FP32 result is the dequantised
    //      INT32 accumulator, just like QuantizedConv2d::forward_quantized).
    //
    // Per-channel weight quantisation reuses
    // `quantized_linear_per_channel_kernel` analogously.

    auto input_shape = input.shape();
    if (input_shape.size() != 3) {
        throw std::runtime_error(
            "QuantizedConv1d: expected input of shape (N, C_in, L), got " +
            std::to_string(input_shape.size()) + "D tensor");
    }
    int64_t batch = input_shape[0];
    int64_t in_c_total = input_shape[1];
    int64_t l_in = input_shape[2];
    if (in_c_total != in_channels_) {
        throw std::runtime_error(
            "QuantizedConv1d: input has " + std::to_string(in_c_total) +
            " channels but layer was constructed for " +
            std::to_string(in_channels_));
    }

    int64_t l_out = (l_in + 2 * padding_ - dilation_ * (kernel_size_ - 1) - 1) / stride_ + 1;
    if (l_out <= 0) {
        throw std::runtime_error(
            "QuantizedConv1d: computed l_out=" + std::to_string(l_out) +
            " is non-positive; check kernel/stride/padding/dilation");
    }

    auto original_device = input.device();

    // Pull INT8 tensors + qparams to CPU for the kernel.
    Tensor input_data_cpu = input.data();
    if (input_data_cpu.device() != Device::cpu()) {
        input_data_cpu = input_data_cpu.to(Device::cpu());
    }
    Tensor weight_data_cpu = weight_.data();
    if (weight_data_cpu.device() != Device::cpu()) {
        weight_data_cpu = weight_data_cpu.to(Device::cpu());
    }

    const auto& input_params = input.params();
    const auto& weight_params = weight_.params();

    Tensor input_scale_cpu = input_params.scale;
    Tensor weight_scale_cpu = weight_params.scale;
    Tensor input_zp_cpu = input_params.zero_point;
    Tensor weight_zp_cpu = weight_params.zero_point;
    if (input_scale_cpu.device() != Device::cpu())
        input_scale_cpu = input_scale_cpu.to(Device::cpu());
    if (weight_scale_cpu.device() != Device::cpu())
        weight_scale_cpu = weight_scale_cpu.to(Device::cpu());
    if (input_zp_cpu.device() != Device::cpu())
        input_zp_cpu = input_zp_cpu.to(Device::cpu());
    if (weight_zp_cpu.device() != Device::cpu())
        weight_zp_cpu = weight_zp_cpu.to(Device::cpu());

    const float input_scale = input_scale_cpu.data<const float>()[0];
    int32_t input_zp = input_zp_cpu.data<int32_t>()[0];

    const bool is_per_channel =
        (weight_params.scheme == QuantizationScheme::PerChannelSymmetric ||
         weight_params.scheme == QuantizationScheme::PerChannelAsymmetric);

    // Bias: Float32 on CPU (matches QuantizedLinear / QuantizedConv2d).
    std::optional<Tensor> bias_f32;
    const float* bias_data = nullptr;
    if (bias_.has_value()) {
        Tensor bias_cpu = *bias_;
        if (bias_cpu.dtype() != DType::Float32) {
            bias_cpu = bias_cpu.to(DType::Float32);
        }
        if (bias_cpu.device() != Device::cpu()) {
            bias_cpu = bias_cpu.to(Device::cpu());
        }
        bias_f32 = bias_cpu;
        bias_data = bias_f32->data<const float>();
    }

    // uint8 asymmetric activations are stored as UInt8; remap to the signed
    // int8 the kernel expects (q_i8 = q_u8 - 128, zp_i8 = zp_u8 - 128). The
    // dequant value scale*(q - zp) is invariant under shifting q and zp by the
    // same constant, so the signed kernel yields identical results. Reading
    // UInt8 storage straight through data<int8_t>() reinterpreted codes >= 128
    // as negative and silently corrupted the output.
    if (input_data_cpu.dtype() == DType::UInt8) {
        input_data_cpu = sub(input_data_cpu.to(DType::Float32), 128.0).to(DType::Int8);
        input_zp -= 128;
    }
    const int8_t* input_data = input_data_cpu.data<int8_t>();
    const int8_t* weight_data = weight_data_cpu.data<int8_t>();

    const int64_t in_c_per_group = in_channels_ / groups_;
    const int64_t out_c_per_group = out_channels_ / groups_;
    const int64_t patch_dim = in_c_per_group * kernel_size_;

    // Allocate FP32 output on CPU; the INT8 GEMM dequantises into FP32.
    Tensor output({batch, out_channels_, l_out}, DType::Float32, Device::cpu());
    float* output_data = output.data<float>();

    // im2col buffer: [l_out, patch_dim] INT8, pad with input_zp (which is the
    // quantised representation of 0.0 in the input scale).
    std::vector<int8_t> col_buffer(static_cast<size_t>(l_out) * patch_dim);
    // Temporary GEMM output buffer for one (batch, group) tile:
    // shape [l_out, out_c_per_group] FP32, written column-major-by-row into
    // the final NCL output below.
    std::vector<float> gemm_out(static_cast<size_t>(l_out) * out_c_per_group);
    // No bias inside the GEMM tile; we add bias when scattering to output so
    // it isn't applied multiple times in the grouped case.

    const int8_t pad_value = static_cast<int8_t>(
        std::clamp<int32_t>(input_zp, -128, 127));

    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t g = 0; g < groups_; ++g) {
            // 1. Build im2col INT8 patches for this (batch, group) tile.
            //    Row ol contains kernel_size patches across in_c_per_group
            //    channels, laid out as [ic0_k0, ic0_k1, ..., ic1_k0, ...].
            //    Out-of-range L positions are filled with pad_value.
            std::fill(col_buffer.begin(), col_buffer.end(), pad_value);
            for (int64_t ol = 0; ol < l_out; ++ol) {
                int8_t* row = col_buffer.data() + ol * patch_dim;
                for (int64_t ic = 0; ic < in_c_per_group; ++ic) {
                    int64_t ic_abs = g * in_c_per_group + ic;
                    int64_t in_base = (n * in_channels_ + ic_abs) * l_in;
                    for (int64_t k = 0; k < kernel_size_; ++k) {
                        int64_t il = ol * stride_ - padding_ + k * dilation_;
                        int8_t v = pad_value;
                        if (il >= 0 && il < l_in) {
                            v = input_data[in_base + il];
                        }
                        row[ic * kernel_size_ + k] = v;
                    }
                }
            }

            // 2. Weight tile for this group is rows
            //    [g*out_c_per_group .. (g+1)*out_c_per_group) of weight,
            //    each row already [in_c_per_group * kernel_size] = patch_dim.
            const int8_t* weight_tile =
                weight_data + g * out_c_per_group * patch_dim;

            // 3. INT8 GEMM: gemm_out[l_out, out_c_per_group] =
            //      (col_buffer @ weight_tile^T) * (input_scale * weight_scale)
            //    (no bias here — we add it once during scatter).
            if (is_per_channel) {
                const float* full_weight_scales =
                    weight_scale_cpu.data<const float>();
                const int32_t* full_weight_zps =
                    weight_zp_cpu.data<int32_t>();
                const float* weight_scales =
                    full_weight_scales + g * out_c_per_group;
                const int32_t* weight_zps =
                    full_weight_zps + g * out_c_per_group;
                kernels::quantized_linear_per_channel_kernel(
                    col_buffer.data(), weight_tile,
                    /*bias=*/nullptr, gemm_out.data(),
                    /*batch_size=*/l_out,
                    /*in_features=*/patch_dim,
                    /*out_features=*/out_c_per_group,
                    input_scale, weight_scales,
                    /*output_scale=*/1.0f,
                    input_zp, weight_zps);
            } else {
                const float weight_scale =
                    weight_scale_cpu.data<const float>()[0];
                const int32_t weight_zp =
                    weight_zp_cpu.data<int32_t>()[0];
                kernels::quantized_linear_kernel(
                    col_buffer.data(), weight_tile,
                    /*bias=*/nullptr, gemm_out.data(),
                    /*batch_size=*/l_out,
                    /*in_features=*/patch_dim,
                    /*out_features=*/out_c_per_group,
                    input_scale, weight_scale,
                    /*output_scale=*/1.0f,
                    input_zp, weight_zp);
            }

            // 4. Scatter [l_out, out_c_per_group] FP32 -> output (N, C, L).
            //    Add bias once per output channel here.
            for (int64_t oc = 0; oc < out_c_per_group; ++oc) {
                int64_t oc_abs = g * out_c_per_group + oc;
                float b = (bias_data != nullptr) ? bias_data[oc_abs] : 0.0f;
                int64_t out_base = (n * out_channels_ + oc_abs) * l_out;
                for (int64_t ol = 0; ol < l_out; ++ol) {
                    output_data[out_base + ol] =
                        gemm_out[ol * out_c_per_group + oc] + b;
                }
            }
        }
    }

    return output.to(original_device);
}

auto QuantizedConv1d::forward_quantized_output(
    const QuantizedTensor& input,
    const QuantizationParams& output_qparams
) -> QuantizedTensor {
    Tensor fp_output = forward_quantized(input);
    return quantize_tensor(fp_output, output_qparams);
}

auto QuantizedConv1d::set_weight(const QuantizedTensor& weights) -> void {
    weight_ = weights;
}

auto QuantizedConv1d::set_bias(const Tensor& bias) -> void {
    bias_ = bias;
}

auto QuantizedConv1d::from_float(Module& fp_conv, const QConfig& qconfig)
    -> std::shared_ptr<QuantizedConv1d> {
    // forward_quantized reads the weight buffer as contiguous int8. INT4/UINT4
    // pack two values per byte (ceil(numel/2) bytes), so an INT4 weight qconfig
    // would cause out-of-bounds reads at inference. Reject it here.
    if (qconfig.weight_dtype() != QuantDType::INT8 &&
        qconfig.weight_dtype() != QuantDType::UINT8) {
        throw std::invalid_argument(
            "QuantizedConv1d::from_float: only INT8/UINT8 weight quantization is "
            "supported; INT4/UINT4 packing is incompatible with the int8 weight reader");
    }

    auto params = fp_conv.named_parameters();

    Tensor weight;
    bool found_weight = false;
    for (auto& [name, var] : params) {
        if (name == "weight") { weight = var->tensor(); found_weight = true; break; }
    }
    if (!found_weight) {
        throw std::runtime_error("QuantizedConv1d::from_float: module has no 'weight' parameter");
    }

    auto shape = weight.shape();  // [out_channels, in_channels/groups, kernel_size]
    int64_t out_channels = shape[0];
    int64_t in_c_per_group = shape[1];
    int64_t kernel_size = shape[2];

    // Quantize weights
    auto weight_cpu = (weight.device() == Device::cpu()) ? weight : weight.to(Device::cpu());
    auto weight_observer = qconfig.create_weight_observer();
    weight_observer->observe(weight_cpu);
    auto weight_qparams = weight_observer->calculate_qparams(
        qconfig.weight_dtype(), qconfig.weight_scheme());

    // Read the real stride/padding/dilation/groups from the source Conv1d
    // (matching QuantizedConv3d::from_float) instead of hardcoding defaults,
    // which silently produced the wrong receptive field / output shape for any
    // non-default conv.
    int64_t stride = 1, padding = 0, dilation = 1, groups = 1;
    int64_t in_channels = in_c_per_group;  // fallback assumes groups=1
    if (auto* conv1d = dynamic_cast<nn::Conv1d*>(&fp_conv)) {
        stride = conv1d->stride();
        padding = conv1d->padding();
        dilation = conv1d->dilation();
        groups = conv1d->groups();
        in_channels = in_c_per_group * groups;  // per-group -> total in_channels
    }

    auto result = std::make_shared<QuantizedConv1d>(
        in_channels,
        out_channels, kernel_size,
        stride, padding, dilation, groups,
        weight_qparams);

    QuantizedTensor q_weight = quantize_tensor(weight_cpu, weight_qparams);
    result->set_weight(q_weight);

    for (auto& [name, var] : params) {
        if (name == "bias") { result->set_bias(var->tensor()); break; }
    }

    return result;
}

// ============================================================================
// QuantizedConvTranspose2d
// ============================================================================

QuantizedConvTranspose2d::QuantizedConvTranspose2d(
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups,
    QuantizationParams weight_qparams,
    float bias_scale
) : in_channels_(in_channels),
    out_channels_(out_channels),
    kernel_size_(kernel_size),
    stride_(stride),
    padding_(padding),
    output_padding_(output_padding),
    groups_(groups),
    weight_(Tensor({in_channels, out_channels / groups, kernel_size, kernel_size},
                   DType::Int8, Device::cpu()), std::move(weight_qparams)),
    bias_scale_(bias_scale) {}

auto QuantizedConvTranspose2d::forward_impl(const Variable& input) -> Variable {
    // Audit item B.1: inference-only quantized layer.
    if (input.requires_grad()) {
        throw std::runtime_error(
            "QuantizedConvTranspose2d: input has requires_grad=true but "
            "this is an inference-only quantized layer.  Use a "
            "non-quantized ConvTranspose2d for training.");
    }
    // JIT-R066: forward_quantized's core math (functional::conv_transpose2d)
    // IS dispatch-visible, but it operates on `input.dequantize()`'s raw,
    // untraced-host-loop OUTPUT — a fresh Tensor disconnected from this
    // Variable's traced lineage (same class as QuantizedLinear's dynamic
    // path: quantize_per_tensor_symmetric() here recomputes qparams from
    // this call's live min/max via the same untraced code). A trace would
    // silently freeze the conv_transpose2d output as a constant regardless
    // of the real (replayed) input.
    if (::tenzor::jit::Tracer::get_instance().is_tracing()) {
        throw std::runtime_error(
            "QuantizedConvTranspose2d::forward_impl: cannot be JIT-traced "
            "(see JIT-R066). Call outside jit.compile()/jit.trace().");
    }
    auto q_input = quantize_per_tensor_symmetric(input.tensor());
    Tensor output = forward_quantized(q_input);
    return Variable(output, /*requires_grad=*/false);
}

auto QuantizedConvTranspose2d::forward_quantized(const QuantizedTensor& input) -> Tensor {
    // Dequantize and delegate to the functional transposed-conv2d kernel
    // (im2col + GEMM, parallelized), mirroring QuantizedConv3d, instead of a
    // naive 7-deep scalar scatter loop.
    Tensor fp_input = input.dequantize();
    Tensor fp_weight = weight_.dequantize();
    // JIT-R066 follow-up (found via QuantizedConvTranspose2dRefusesToTrace's
    // eager-path assertion): weight_ defaults to CPU and dequantize_tensor's
    // `.to(original_device)` is a no-op unless weight_ was explicitly moved,
    // so fp_weight silently stayed on CPU while fp_input follows the real
    // input's device — functional::conv_transpose2d then dispatched with
    // mismatched-device operands, which on Vulkan surfaces as "Invalid
    // buffer pointer: buffer not tracked" (the CPU weight's pointer was
    // never Vulkan-allocated). Mirrors the bias reconciliation just below.
    if (fp_weight.device() != fp_input.device()) {
        fp_weight = fp_weight.to(fp_input.device());
    }

    Variable v_input(fp_input, /*requires_grad=*/false);
    Variable v_weight(fp_weight, /*requires_grad=*/false);

    std::optional<Variable> v_bias;
    if (bias_.has_value()) {
        Tensor b = *bias_;
        if (b.dtype() != DType::Float32) {
            b = b.to(DType::Float32);
        }
        if (b.device() != fp_input.device()) {
            b = b.to(fp_input.device());
        }
        v_bias = Variable(b, /*requires_grad=*/false);
    }

    Variable v_out = ::tenzor::nn::functional::conv_transpose2d(
        v_input, v_weight, v_bias,
        {stride_, stride_},
        {padding_, padding_},
        {output_padding_, output_padding_},
        groups_,
        {1, 1});

    return v_out.tensor();
}

auto QuantizedConvTranspose2d::set_weight(const QuantizedTensor& weights) -> void {
    weight_ = weights;
}

auto QuantizedConvTranspose2d::set_bias(const Tensor& bias) -> void {
    bias_ = bias;
}

auto QuantizedConvTranspose2d::from_float(Module& fp_conv, const QConfig& qconfig)
    -> std::shared_ptr<QuantizedConvTranspose2d> {
    // forward_quantized reads the weight buffer as contiguous int8. INT4/UINT4
    // pack two values per byte (ceil(numel/2) bytes), so an INT4 weight qconfig
    // would cause out-of-bounds reads at inference. Reject it here.
    if (qconfig.weight_dtype() != QuantDType::INT8 &&
        qconfig.weight_dtype() != QuantDType::UINT8) {
        throw std::invalid_argument(
            "QuantizedConvTranspose2d::from_float: only INT8/UINT8 weight quantization "
            "is supported; INT4/UINT4 packing is incompatible with the int8 weight reader");
    }

    auto params = fp_conv.named_parameters();

    Tensor weight;
    bool found_weight = false;
    for (auto& [name, var] : params) {
        if (name == "weight") { weight = var->tensor(); found_weight = true; break; }
    }
    if (!found_weight) {
        throw std::runtime_error("QuantizedConvTranspose2d::from_float: module has no 'weight' parameter");
    }

    auto shape = weight.shape();  // [in_channels, out_channels/groups, kH, kW]
    int64_t in_channels = shape[0];
    int64_t out_c_per_group = shape[1];
    int64_t kernel_h = shape[2];
    int64_t kernel_w = shape[3];

    // Read the real stride/padding/output_padding/dilation/groups from the
    // source ConvTranspose2d instead of hardcoding defaults (which silently
    // produced the wrong receptive field / output shape). The quantized layer
    // ctor is square-only, so reject non-square geometry rather than silently
    // collapsing it.
    int64_t stride = 1, padding = 0, output_padding = 0, groups = 1;
    int64_t out_channels = out_c_per_group;  // fallback assumes groups=1
    if (auto* convt = dynamic_cast<nn::ConvTranspose2d*>(&fp_conv)) {
        if (convt->stride_h() != convt->stride_w() ||
            convt->padding_h() != convt->padding_w() ||
            convt->output_padding_h() != convt->output_padding_w() ||
            kernel_h != kernel_w) {
            throw std::runtime_error(
                "QuantizedConvTranspose2d::from_float: only square "
                "kernel/stride/padding/output_padding are supported");
        }
        stride = convt->stride();
        padding = convt->padding();
        output_padding = convt->output_padding();
        groups = convt->groups();
        out_channels = out_c_per_group * groups;  // per-group -> total out_channels
    }

    // Quantize weights
    auto weight_cpu = (weight.device() == Device::cpu()) ? weight : weight.to(Device::cpu());
    auto weight_observer = qconfig.create_weight_observer();
    weight_observer->observe(weight_cpu);
    auto weight_qparams = weight_observer->calculate_qparams(
        qconfig.weight_dtype(), qconfig.weight_scheme());

    auto result = std::make_shared<QuantizedConvTranspose2d>(
        in_channels, out_channels, kernel_h,
        stride, padding, output_padding, groups,
        weight_qparams);

    QuantizedTensor q_weight = quantize_tensor(weight_cpu, weight_qparams);
    result->set_weight(q_weight);

    for (auto& [name, var] : params) {
        if (name == "bias") { result->set_bias(var->tensor()); break; }
    }

    return result;
}

// ============================================================================
// QuantizedLSTMCell
// ============================================================================

QuantizedLSTMCell::QuantizedLSTMCell(
    int64_t input_size,
    int64_t hidden_size,
    bool bias,
    QuantizationParams weight_qparams
) : input_size_(input_size),
    hidden_size_(hidden_size),
    bias_(bias),
    weight_ih_(Tensor({4 * hidden_size, input_size}, DType::Int8, Device::cpu()),
               weight_qparams),
    weight_hh_(Tensor({4 * hidden_size, hidden_size}, DType::Int8, Device::cpu()),
               weight_qparams) {

    if (bias) {
        bias_ih_ = zeros({4 * hidden_size}, DType::Float32, Device::cpu());
        bias_hh_ = zeros({4 * hidden_size}, DType::Float32, Device::cpu());
    }
}

auto QuantizedLSTMCell::forward_impl(const Variable& input) -> Variable {
    // Default: zero initial state
    auto shape = input.shape();
    int64_t batch = shape[0];

    auto hx = Variable(zeros({batch, hidden_size_}, DType::Float32, input.tensor().device()), false);
    auto cx = Variable(zeros({batch, hidden_size_}, DType::Float32, input.tensor().device()), false);

    auto [h, c] = forward_cell(input, hx, cx);
    return h;
}

auto QuantizedLSTMCell::forward_cell(const Variable& input,
                                      const Variable& hx, const Variable& cx)
    -> std::pair<Variable, Variable> {
    // Dequantize weights to FP32 for gate computation
    Tensor w_ih = weight_ih_.dequantize();
    Tensor w_hh = weight_hh_.dequantize();

    auto x = input.tensor();
    auto h = hx.tensor();
    auto c = cx.tensor();

    // JIT-R103: quantized weights (and from_float() always quantizes on
    // CPU) stay wherever they were constructed/quantized until moved,
    // unlike x/h/c which are on the actual call's device — a lazily-placed
    // module (constructed on CPU, called with a GPU input) hit a
    // device-mismatch dispatch error at the matmul below.
    if (w_ih.device() != x.device()) w_ih = w_ih.to(x.device());
    if (w_hh.device() != x.device()) w_hh = w_hh.to(x.device());

    // gates = x @ w_ih^T + h @ w_hh^T + bias
    auto gates = matmul(x, w_ih.permute({1, 0}));
    gates = gates + matmul(h, w_hh.permute({1, 0}));
    if (bias_ih_) {
        Tensor b_ih = (bias_ih_->device() != x.device()) ? bias_ih_->to(x.device()) : *bias_ih_;
        gates = gates + b_ih;
    }
    if (bias_hh_) {
        Tensor b_hh = (bias_hh_->device() != x.device()) ? bias_hh_->to(x.device()) : *bias_hh_;
        gates = gates + b_hh;
    }

    // Split gates: [batch, 4*hidden] -> 4x [batch, hidden]. `gates` is a
    // live, per-call-varying dispatched value — traced_slice (JIT-R058a)
    // keeps each split tied to it instead of freezing at trace time.
    auto i_gate = sigmoid(traced_slice(gates, 1, 0, hidden_size_));
    auto f_gate = sigmoid(traced_slice(gates, 1, hidden_size_, 2 * hidden_size_));
    auto g_gate = tanh(traced_slice(gates, 1, 2 * hidden_size_, 3 * hidden_size_));
    auto o_gate = sigmoid(traced_slice(gates, 1, 3 * hidden_size_, 4 * hidden_size_));

    auto c_new = f_gate * c + i_gate * g_gate;
    auto h_new = o_gate * tanh(c_new);

    return {Variable(h_new, false), Variable(c_new, false)};
}

auto QuantizedLSTMCell::from_float(Module& fp_lstm_cell, [[maybe_unused]] const QConfig& qconfig)
    -> std::shared_ptr<QuantizedLSTMCell> {
    // nn::LSTMCell's weight_ih/weight_hh are nested Linear submodules
    // ("weight_ih.weight" in named_parameters() terms), not flat "weight_ih"
    // parameters — matching QuantizedLSTM::from_float's fix above.
    auto* cell = dynamic_cast<LSTMCell*>(&fp_lstm_cell);
    if (!cell) {
        throw std::runtime_error(
            "QuantizedLSTMCell::from_float: fp_lstm_cell is not an "
            "nn::LSTMCell instance");
    }

    auto w_ih = cell->weight_ih()->weight()->tensor();
    auto w_hh = cell->weight_hh()->weight()->tensor();
    int64_t hidden_size = w_ih.shape()[0] / 4;
    int64_t input_size = w_ih.shape()[1];
    bool has_bias = cell->weight_ih()->has_bias();

    auto w_ih_cpu = (w_ih.device() == Device::cpu()) ? w_ih : w_ih.to(Device::cpu());
    auto w_hh_cpu = (w_hh.device() == Device::cpu()) ? w_hh : w_hh.to(Device::cpu());

    auto result = std::make_shared<QuantizedLSTMCell>(
        input_size, hidden_size, has_bias);

    result->weight_ih_ = quantize_per_tensor_symmetric(w_ih_cpu);
    result->weight_hh_ = quantize_per_tensor_symmetric(w_hh_cpu);

    // weight_hh's Linear is constructed with bias=false unconditionally
    // (see LSTMCell::LSTMCell) — only weight_ih ever carries a bias.
    if (has_bias && cell->weight_ih()->has_bias()) {
        result->bias_ih_ = cell->weight_ih()->bias()->tensor();
    }
    if (cell->weight_hh()->has_bias()) {
        result->bias_hh_ = cell->weight_hh()->bias()->tensor();
    }

    return result;
}

// ============================================================================
// QuantizedEmbeddingBag
// ============================================================================

QuantizedEmbeddingBag::QuantizedEmbeddingBag(
    int64_t num_embeddings,
    int64_t embedding_dim,
    QuantizationParams weight_qparams,
    Mode mode)
    : num_embeddings_(num_embeddings)
    , embedding_dim_(embedding_dim)
    , mode_(mode)
    , weight_(Tensor({num_embeddings, embedding_dim}, DType::Int8, Device::cpu()),
              std::move(weight_qparams)) {}

auto QuantizedEmbeddingBag::forward_impl(const Variable& input) -> Variable {
    // JIT-R066: see QuantizedEmbedding::forward_impl — forward_quantized is
    // a raw host loop over `indices`/`offsets` with zero dispatch() calls;
    // a trace would silently freeze the output as a constant regardless of
    // the real (replayed) input.
    if (::tenzor::jit::Tracer::get_instance().is_tracing()) {
        throw std::runtime_error(
            "QuantizedEmbeddingBag::forward_impl: cannot be JIT-traced (see "
            "JIT-R066). Call outside jit.compile()/jit.trace().");
    }
    // Simple forward: treat entire input as one bag with no offsets
    Tensor offsets = tenzor::zeros({1}, DType::Int64, input.tensor().device());
    Tensor output = forward_quantized(input.tensor(), offsets);
    return Variable(output, false);
}

auto QuantizedEmbeddingBag::forward_quantized(const Tensor& indices,
                                               const Tensor& offsets) -> Tensor {
    int64_t num_bags = offsets.numel();

    Tensor output({num_bags, embedding_dim_}, DType::Float32, Device::cpu());

    // Get quantization parameters
    Tensor scale_cpu = weight_.params().scale;
    Tensor zp_cpu = weight_.params().zero_point;
    if (scale_cpu.device() != Device::cpu()) scale_cpu = scale_cpu.to(Device::cpu());
    if (zp_cpu.device() != Device::cpu()) zp_cpu = zp_cpu.to(Device::cpu());

    float scale = scale_cpu.data<float>()[0];
    int32_t zp = zp_cpu.data<int32_t>()[0];

    Tensor weight_cpu = weight_.data();
    if (weight_cpu.device() != Device::cpu()) weight_cpu = weight_cpu.to(Device::cpu());
    const int8_t* w_data = weight_cpu.data<int8_t>();

    Tensor indices_cpu = (indices.device() != Device::cpu()) ? indices.to(Device::cpu()) : indices;
    Tensor offsets_cpu = (offsets.device() != Device::cpu()) ? offsets.to(Device::cpu()) : offsets;

    const int64_t* idx_data = indices_cpu.data<int64_t>();
    const int64_t* off_data = offsets_cpu.data<int64_t>();
    int64_t total_indices = indices_cpu.numel();
    float* out_data = output.data<float>();

    // Validate offsets before iterating: they index into idx_data and a
    // malformed offsets tensor (out-of-range or non-monotonic) would cause
    // out-of-bounds reads of idx_data in the accumulation loop below.
    for (int64_t b = 0; b < num_bags; ++b) {
        if (off_data[b] < 0 || off_data[b] > total_indices) {
            throw std::out_of_range(
                "QuantizedEmbeddingBag: offset " + std::to_string(off_data[b]) +
                " at bag " + std::to_string(b) + " is out of range [0, " +
                std::to_string(total_indices) + "]");
        }
        if (b + 1 < num_bags && off_data[b + 1] < off_data[b]) {
            throw std::out_of_range(
                "QuantizedEmbeddingBag: offsets are not monotonically "
                "non-decreasing at bag " + std::to_string(b));
        }
    }

    for (int64_t b = 0; b < num_bags; ++b) {
        int64_t start = off_data[b];
        int64_t end = (b + 1 < num_bags) ? off_data[b + 1] : total_indices;
        int64_t bag_size = end - start;

        float* out_row = out_data + b * embedding_dim_;

        // Initialize output
        for (int64_t j = 0; j < embedding_dim_; ++j) {
            out_row[j] = (mode_ == Mode::Max) ? -std::numeric_limits<float>::infinity() : 0.0f;
        }

        // Accumulate embeddings
        for (int64_t i = start; i < end; ++i) {
            int64_t idx = idx_data[i];
            if (idx < 0) idx += num_embeddings_;
            if (idx < 0 || idx >= num_embeddings_) {
                throw std::out_of_range("QuantizedEmbeddingBag: index out of range");
            }

            const int8_t* row = w_data + idx * embedding_dim_;
            for (int64_t j = 0; j < embedding_dim_; ++j) {
                float val = (static_cast<float>(row[j]) - zp) * scale;
                if (mode_ == Mode::Max) {
                    out_row[j] = std::max(out_row[j], val);
                } else {
                    out_row[j] += val;
                }
            }
        }

        // Empty bags return a zero row for all modes (matching PyTorch
        // EmbeddingBag semantics). In particular this overwrites the -inf
        // initialization used by Max mode, which the accumulation loop never
        // touches for an empty bag.
        if (bag_size == 0) {
            for (int64_t j = 0; j < embedding_dim_; ++j) {
                out_row[j] = 0.0f;
            }
            continue;
        }

        // Apply mean reduction
        if (mode_ == Mode::Mean) {
            float inv_size = 1.0f / static_cast<float>(bag_size);
            for (int64_t j = 0; j < embedding_dim_; ++j) {
                out_row[j] *= inv_size;
            }
        }
    }

    if (indices.device() != Device::cpu()) {
        output = output.to(indices.device());
    }

    return output;
}

auto QuantizedEmbeddingBag::set_weight(const QuantizedTensor& weights) -> void {
    weight_ = weights;
}

auto QuantizedEmbeddingBag::from_float(Module& fp_embedding_bag,
                                        [[maybe_unused]] const QConfig& qconfig)
    -> std::shared_ptr<QuantizedEmbeddingBag> {
    auto params = fp_embedding_bag.named_parameters();
    Tensor weight;
    bool found = false;
    for (auto& [name, var] : params) {
        if (name == "weight") { weight = var->tensor(); found = true; break; }
    }
    if (!found) {
        throw std::runtime_error("QuantizedEmbeddingBag::from_float: module has no 'weight' parameter");
    }
    auto shape = weight.shape();
    int64_t num_embeddings = shape[0];
    int64_t embedding_dim = shape[1];

    auto weight_cpu = (weight.device() == Device::cpu()) ? weight : weight.to(Device::cpu());
    auto q_weight = quantize_per_tensor_symmetric(weight_cpu);

    auto result = std::make_shared<QuantizedEmbeddingBag>(
        num_embeddings, embedding_dim, q_weight.params());
    result->set_weight(q_weight);
    return result;
}

} // namespace quantization
} // namespace nn
} // namespace tenzor
