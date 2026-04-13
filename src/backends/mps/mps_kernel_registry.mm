/**
 * @file mps_kernel_registry.mm
 * @brief MPS kernel registration for Tier 1 operations
 *
 * Registers Metal compute shader kernels with the dispatch table.
 * Tier 1 covers the essential ops needed for inference.
 */

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "mps_backend.hpp"
#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/kernel_registry.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/shape.hpp"

namespace tenzor::mps {

// ============================================================================
// Zero-copy shape operations (metadata-only, no Metal dispatch needed)
// ============================================================================

static auto mps_reshape_kernel(const Tensor& input, const std::vector<int64_t>& new_shape) -> Tensor {
    if (!input.is_contiguous()) {
        // Non-contiguous: must materialize via CPU (rare path)
        auto cpu_input = input.to(Device::cpu());
        auto cpu_result = cpu_input.reshape(new_shape);
        return cpu_result.to(input.device());
    }
    Tensor result;
    TensorAccessor::get_impl_mutable(result) = make_intrusive<TensorImpl>(*TensorAccessor::get_impl(input));
    result.mutable_shape() = new_shape;
    result.mutable_strides() = compute_strides(new_shape);
    return result;
}

static auto mps_transpose_kernel(const Tensor& input, int64_t dim0, int64_t dim1) -> Tensor {
    const int64_t ndim = input.ndim();
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;
    Tensor result;
    TensorAccessor::get_impl_mutable(result) = make_intrusive<TensorImpl>(*TensorAccessor::get_impl(input));
    auto& r_shape = result.mutable_shape();
    auto& r_strides = result.mutable_strides();
    std::swap(r_shape[dim0], r_shape[dim1]);
    std::swap(r_strides[dim0], r_strides[dim1]);
    return result;
}

static auto mps_permute_kernel(const Tensor& input, const std::vector<int64_t>& dims) -> Tensor {
    const int64_t ndim = input.ndim();
    Tensor result;
    TensorAccessor::get_impl_mutable(result) = make_intrusive<TensorImpl>(*TensorAccessor::get_impl(input));
    std::vector<int64_t> new_shape(ndim);
    std::vector<int64_t> new_strides(ndim);
    for (int64_t i = 0; i < ndim; ++i) {
        new_shape[i] = input.shape()[dims[i]];
        new_strides[i] = input.strides()[dims[i]];
    }
    result.mutable_shape() = std::move(new_shape);
    result.mutable_strides() = std::move(new_strides);
    return result;
}

static auto mps_squeeze_kernel(const Tensor& input, int64_t dim) -> Tensor {
    Tensor result;
    TensorAccessor::get_impl_mutable(result) = make_intrusive<TensorImpl>(*TensorAccessor::get_impl(input));
    if (dim >= 0) {
        auto& r_shape = result.mutable_shape();
        auto& r_strides = result.mutable_strides();
        r_shape.erase(r_shape.begin() + dim);
        r_strides.erase(r_strides.begin() + dim);
    } else {
        std::vector<int64_t> new_shape;
        std::vector<int64_t> new_strides;
        for (int64_t i = 0; i < input.ndim(); ++i) {
            if (input.shape()[i] != 1) {
                new_shape.push_back(input.shape()[i]);
                new_strides.push_back(input.strides()[i]);
            }
        }
        if (new_shape.empty()) {
            new_shape.push_back(1);
            new_strides.push_back(1);
        }
        result.mutable_shape() = std::move(new_shape);
        result.mutable_strides() = std::move(new_strides);
    }
    return result;
}

static auto mps_unsqueeze_kernel(const Tensor& input, int64_t dim) -> Tensor {
    Tensor result;
    TensorAccessor::get_impl_mutable(result) = make_intrusive<TensorImpl>(*TensorAccessor::get_impl(input));
    auto& r_shape = result.mutable_shape();
    auto& r_strides = result.mutable_strides();
    r_shape.insert(r_shape.begin() + dim, 1);
    int64_t new_stride = (dim < input.ndim()) ? input.strides()[dim] : 1;
    r_strides.insert(r_strides.begin() + dim, new_stride);
    return result;
}

static auto mps_flatten_kernel(const Tensor& input, int64_t start_dim, int64_t end_dim) -> Tensor {
    auto shape = input.shape();
    int64_t ndim = shape.size();
    if (start_dim < 0) start_dim += ndim;
    if (end_dim < 0) end_dim += ndim;
    int64_t flat_size = 1;
    for (int64_t d = start_dim; d <= end_dim; ++d) {
        flat_size *= shape[d];
    }
    std::vector<int64_t> new_shape;
    for (int64_t d = 0; d < start_dim; ++d) new_shape.push_back(shape[d]);
    new_shape.push_back(flat_size);
    for (int64_t d = end_dim + 1; d < ndim; ++d) new_shape.push_back(shape[d]);
    return mps_reshape_kernel(input, new_shape);
}

static auto mps_expand_kernel(const Tensor& input, const std::vector<int64_t>& target_shape) -> Tensor {
    const auto& in_shape = input.shape();
    const auto& in_strides = input.strides();
    int64_t ndim_out = static_cast<int64_t>(target_shape.size());
    int64_t ndim_in = input.ndim();
    int64_t dim_diff = ndim_out - ndim_in;

    std::vector<int64_t> new_strides(ndim_out, 0);
    for (int64_t i = ndim_out - 1; i >= 0; --i) {
        int64_t in_idx = i - dim_diff;
        if (in_idx >= 0) {
            if (in_shape[in_idx] == target_shape[i]) {
                new_strides[i] = in_strides[in_idx];
            } else if (in_shape[in_idx] == 1) {
                new_strides[i] = 0;  // Broadcast
            } else {
                throw std::runtime_error("expand: incompatible shapes");
            }
        }
    }
    Tensor result;
    TensorAccessor::get_impl_mutable(result) = make_intrusive<TensorImpl>(*TensorAccessor::get_impl(input));
    result.mutable_shape() = target_shape;
    result.mutable_strides() = new_strides;
    return result;
}

// Forward declarations of kernel functions (defined in kernels/mps_elementwise.mm)
Tensor mps_add_kernel(const Tensor& a, const Tensor& b);
Tensor mps_sub_kernel(const Tensor& a, const Tensor& b);
Tensor mps_mul_kernel(const Tensor& a, const Tensor& b);
Tensor mps_div_kernel(const Tensor& a, const Tensor& b);
Tensor mps_relu_kernel(const Tensor& input);
Tensor mps_sigmoid_kernel(const Tensor& input);
Tensor mps_neg_kernel(const Tensor& input);
Tensor mps_exp_kernel(const Tensor& input);
Tensor mps_log_kernel(const Tensor& input);
// Phase 3.2 additions — native Metal replacements for unary CPU fallbacks.
Tensor mps_tanh_kernel(const Tensor& input);
Tensor mps_sqrt_kernel(const Tensor& input);
Tensor mps_abs_kernel(const Tensor& input);
Tensor mps_pow_kernel(const Tensor& base, const Tensor& exponent);
Tensor mps_clamp_kernel(const Tensor& input, float min_val, float max_val);
Tensor mps_matmul_kernel(const Tensor& a, const Tensor& b);
Tensor mps_linear_kernel(const Tensor& input, const Tensor& weight, const Tensor& bias);
Tensor mps_embedding_kernel(const Tensor& weight, const Tensor& indices);
Tensor mps_softmax_kernel(const Tensor& input, int64_t dim);
Tensor mps_batch_norm_kernel(const Tensor& input, const Tensor& mean,
                              const Tensor& var, const Tensor& weight,
                              const Tensor& bias, float eps);
Tensor mps_layer_norm_kernel(const Tensor& input, const Tensor& weight,
                              const Tensor& bias, float eps);
Tensor mps_conv2d_kernel(const Tensor& input, const Tensor& weight,
                          int64_t stride_h, int64_t stride_w,
                          int64_t pad_h, int64_t pad_w, int64_t groups);
// Native reduction kernels
Tensor mps_sum_kernel(const Tensor& input, int64_t dim, bool keepdim);
Tensor mps_mean_kernel(const Tensor& input, int64_t dim, bool keepdim);
Tensor mps_max_kernel(const Tensor& input, int64_t dim, bool keepdim, Tensor& out_indices);
// Native comparison kernels (Bool output)
Tensor mps_gt_kernel(const Tensor& a, const Tensor& b);
Tensor mps_eq_kernel(const Tensor& a, const Tensor& b);
Tensor mps_ne_kernel(const Tensor& a, const Tensor& b);
Tensor mps_lt_kernel(const Tensor& a, const Tensor& b);
Tensor mps_le_kernel(const Tensor& a, const Tensor& b);
Tensor mps_ge_kernel(const Tensor& a, const Tensor& b);
// Native backward activation kernels
Tensor mps_relu_backward_kernel(const Tensor& grad, const Tensor& input);
Tensor mps_sigmoid_backward_kernel(const Tensor& grad, const Tensor& sigmoid_out);
Tensor mps_tanh_backward_kernel(const Tensor& grad, const Tensor& tanh_out);
// Native in-place element-wise kernels
Tensor mps_add_inplace_kernel(Tensor& a, const Tensor& b);
Tensor mps_sub_inplace_kernel(Tensor& a, const Tensor& b);
Tensor mps_mul_inplace_kernel(Tensor& a, const Tensor& b);
Tensor mps_div_inplace_kernel(Tensor& a, const Tensor& b);
// Native Cast kernel
Tensor mps_cast_kernel(const Tensor& input, DType target_dtype);
// Native fused optimizer kernels
std::vector<Tensor> mps_fused_sgd_step(const Tensor& param, const Tensor& grad,
                                         const Tensor& momentum_buf,
                                         float lr, float momentum, float weight_decay);
std::vector<Tensor> mps_fused_adam_step(const Tensor& param, const Tensor& grad,
                                         const Tensor& exp_avg, const Tensor& exp_avg_sq,
                                         float lr, float beta1, float beta2,
                                         float eps, float bc1, float bc2,
                                         float weight_decay);

auto register_mps_kernels(BackendDispatchTable& table) -> void {
    // ================================================================
    // Tier 1: Arithmetic operations
    // ================================================================
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Add, mps_add_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Sub, mps_sub_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Mul, mps_mul_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Div, mps_div_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, MatMul, mps_matmul_kernel);

    // ================================================================
    // Tier 1: Activation functions
    // ================================================================
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, ReLU, mps_relu_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Sigmoid, mps_sigmoid_kernel);

    // ================================================================
    // Tier 1: Element-wise math
    // ================================================================
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Neg, mps_neg_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Exp, mps_exp_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Log, mps_log_kernel);

    // ================================================================
    // Tier 1: Linear (matmul + bias add)
    // ================================================================
    table.register_single_output_kernel(OpId::Linear,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            if (inputs.size() >= 3) {
                return mps_linear_kernel(inputs[0], inputs[1], inputs[2]);
            }
            return mps_matmul_kernel(inputs[0], inputs[1]);
        });

    // ================================================================
    // Tier 1: Embedding
    // ================================================================
    table.register_kernel(OpId::Embedding,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {mps_embedding_kernel(inputs[0], inputs[1])};
        });

    // ================================================================
    // Tier 1: Softmax
    // ================================================================
    table.register_kernel(OpId::Softmax,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            return {mps_softmax_kernel(inputs[0], dim)};
        });

    // ================================================================
    // Tier 1: BatchNorm (inference path)
    // ================================================================
    table.register_kernel(OpId::BatchNorm2dForwardAffine,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            return {mps_batch_norm_kernel(inputs[0], inputs[1], inputs[2],
                                          inputs[3], inputs[4], eps)};
        });

    // ================================================================
    // Tier 1: LayerNorm
    // ================================================================
    table.register_kernel(OpId::FusedLayerNorm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            Tensor mean, inv_std; // placeholders for training path
            return {mps_layer_norm_kernel(inputs[0], inputs[1], inputs[2], eps),
                    mean, inv_std};
        });

    // ================================================================
    // Tier 1: Conv2d
    // ================================================================
    table.register_kernel(OpId::Conv2dForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t sh = attrs.get_int(AttrKey::StrideH, 1);
            int64_t sw = attrs.get_int(AttrKey::StrideW, 1);
            int64_t ph = attrs.get_int(AttrKey::PaddingH, 0);
            int64_t pw = attrs.get_int(AttrKey::PaddingW, 0);
            int64_t groups = attrs.get_int(AttrKey::Groups, 1);
            return {mps_conv2d_kernel(inputs[0], inputs[1], sh, sw, ph, pw, groups)};
        });

    // ================================================================
    // Tier 2: CPU-roundtrip fallbacks for training support
    // ================================================================
    // These enable backward pass and optimizer steps on MPS tensors
    // by routing through CPU. Native Metal shaders can replace these
    // incrementally for better performance.

    // Reductions — native Metal kernels (no CPU roundtrip for contiguous last-dim reductions)
    table.register_kernel(OpId::Sum, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{mps_sum_kernel(inputs[0], dim, keepdim)};
    });

    table.register_kernel(OpId::Mean, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{mps_mean_kernel(inputs[0], dim, keepdim)};
    });

    table.register_kernel(OpId::Max, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        Tensor indices;
        auto values = mps_max_kernel(inputs[0], dim, keepdim, indices);
        return std::vector<Tensor>{values, indices};
    });

    // Shape operations — zero-copy metadata ops (no GPU→CPU→GPU roundtrip)
    table.register_single_output_kernel(OpId::Reshape,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            auto shape = attrs.get_int_list(AttrKey::Shape);
            return mps_reshape_kernel(inputs[0], shape);
        });

    table.register_single_output_kernel(OpId::Transpose,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim0 = attrs.get_int(AttrKey::Dim0, 0);
            int64_t dim1 = attrs.get_int(AttrKey::Dim1, 1);
            return mps_transpose_kernel(inputs[0], dim0, dim1);
        });

    // Phase 3.2: native Metal unary / binary kernels (previously CPU
    // fallbacks) for Tanh/Sqrt/Abs and hand-coded CPU lambdas for
    // Pow/Clamp.
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Tanh, mps_tanh_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Sqrt, mps_sqrt_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Abs,  mps_abs_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Pow, mps_pow_kernel);

    // Clamp needs scalar min/max plumbed through OpAttributes, so it
    // can't use the unary register macro directly.
    table.register_single_output_kernel(OpId::Clamp,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            float min_val = static_cast<float>(
                attrs.get_float(AttrKey::Min, std::numeric_limits<double>::lowest()));
            float max_val = static_cast<float>(
                attrs.get_float(AttrKey::Max, std::numeric_limits<double>::max()));
            return mps_clamp_kernel(inputs[0], min_val, max_val);
        });

    // Permute, Squeeze, Unsqueeze, Flatten, Expand — zero-copy metadata ops
    table.register_single_output_kernel(OpId::Permute,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            auto dims = attrs.get_int_list(AttrKey::Dims);
            return mps_permute_kernel(inputs[0], dims);
        });

    table.register_single_output_kernel(OpId::Squeeze,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            return mps_squeeze_kernel(inputs[0], dim);
        });

    table.register_single_output_kernel(OpId::Unsqueeze,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return mps_unsqueeze_kernel(inputs[0], dim);
        });

    table.register_single_output_kernel(OpId::Flatten,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t start_dim = attrs.get_int(AttrKey::StartDim, 0);
            int64_t end_dim = attrs.get_int(AttrKey::EndDim, -1);
            return mps_flatten_kernel(inputs[0], start_dim, end_dim);
        });

    table.register_single_output_kernel(OpId::Expand,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            auto shape = attrs.get_int_list(AttrKey::Shape);
            return mps_expand_kernel(inputs[0], shape);
        });

    // Comparison ops — native Metal kernels with dedicated Bool-output dispatcher
    table.register_kernel(OpId::Gt, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{mps_gt_kernel(inputs[0], inputs[1])};
    });
    table.register_kernel(OpId::Eq, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{mps_eq_kernel(inputs[0], inputs[1])};
    });
    table.register_kernel(OpId::Ne, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{mps_ne_kernel(inputs[0], inputs[1])};
    });
    table.register_kernel(OpId::Lt, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{mps_lt_kernel(inputs[0], inputs[1])};
    });
    table.register_kernel(OpId::Le, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{mps_le_kernel(inputs[0], inputs[1])};
    });
    table.register_kernel(OpId::Ge, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{mps_ge_kernel(inputs[0], inputs[1])};
    });

    // Backward activation kernels — native Metal
    table.register_kernel(OpId::ReLUBackward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{mps_relu_backward_kernel(inputs[0], inputs[1])};
    });
    table.register_kernel(OpId::SigmoidBackward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{mps_sigmoid_backward_kernel(inputs[0], inputs[1])};
    });
    table.register_kernel(OpId::TanhBackward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{mps_tanh_backward_kernel(inputs[0], inputs[1])};
    });

    // In-place arithmetic — native Metal
    table.register_kernel(OpId::AddInplace, [](std::span<const Tensor> inputs, const OpAttributes&) {
        Tensor a = inputs[0];  // copy handle (shared storage — kernel writes in-place)
        return std::vector<Tensor>{mps_add_inplace_kernel(a, inputs[1])};
    });
    table.register_kernel(OpId::SubInplace, [](std::span<const Tensor> inputs, const OpAttributes&) {
        Tensor a = inputs[0];
        return std::vector<Tensor>{mps_sub_inplace_kernel(a, inputs[1])};
    });
    table.register_kernel(OpId::MulInplace, [](std::span<const Tensor> inputs, const OpAttributes&) {
        Tensor a = inputs[0];
        return std::vector<Tensor>{mps_mul_inplace_kernel(a, inputs[1])};
    });
    table.register_kernel(OpId::DivInplace, [](std::span<const Tensor> inputs, const OpAttributes&) {
        Tensor a = inputs[0];
        return std::vector<Tensor>{mps_div_inplace_kernel(a, inputs[1])};
    });

    // Note: zeros_like / ones_like are library-level free functions in
    // tenzor::ops, not dispatch-level OpIds. Autograd code that needs
    // gradient-init scratch tensors should call tenzor::zeros_like(x) /
    // tenzor::ones_like(x), which internally dispatches zeros()/ones()
    // for the tensor's device. No MPS-specific registration is needed.

    // Fused optimizer steps — native Metal kernels (no CPU roundtrip)
    table.register_kernel(OpId::FusedSGDStep, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float lr = static_cast<float>(attrs.get_float(AttrKey::Lr, 0.01));
        float momentum = static_cast<float>(attrs.get_float(AttrKey::Momentum, 0.0));
        float wd = static_cast<float>(attrs.get_float(AttrKey::WeightDecay, 0.0));
        return mps_fused_sgd_step(inputs[0], inputs[1], inputs[2], lr, momentum, wd);
    });

    table.register_kernel(OpId::FusedAdamStep, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float lr = static_cast<float>(attrs.get_float(AttrKey::Lr, 0.001));
        float beta1 = static_cast<float>(attrs.get_float(AttrKey::Beta1, 0.9));
        float beta2 = static_cast<float>(attrs.get_float(AttrKey::Beta2, 0.999));
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-8));
        int64_t step = attrs.get_int(AttrKey::Step, 1);
        float wd = static_cast<float>(attrs.get_float(AttrKey::WeightDecay, 0.0));
        // Compute bias corrections from step count
        float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
        float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));
        return mps_fused_adam_step(inputs[0], inputs[1], inputs[2], inputs[3],
                                    lr, beta1, beta2, eps, bc1, bc2, wd);
    });

    // Cast — native Metal for common pairs, CPU roundtrip for exotic dtypes
    table.register_single_output_kernel(OpId::Cast,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            auto target = static_cast<DType>(attrs.get_int(AttrKey::DType, 0));
            return mps_cast_kernel(inputs[0], target);
        });

    // ================================================================
    // Tier 3: CPU-roundtrip fallbacks for completeness
    // ================================================================
    // These enable MPS models that touch RNN / linalg / FFT / sparse /
    // signal-processing ops to load and run on macOS without crashing
    // with "no kernel registered". They're slow (GPU→CPU→GPU per op)
    // but unblock the Tier-1 scaffold. Native Metal shaders can
    // replace these incrementally as demand warrants.
    //
    // Two helper lambdas capture the shared forward/scatter pattern
    // so each op only has to name its OpId.
    auto mps_roundtrip_multi = [&](OpId op) {
        table.register_kernel(op, [op](std::span<const Tensor> inputs,
                                        const OpAttributes& attrs) {
            auto dev = inputs[0].device();
            std::vector<Tensor> cpu_inputs;
            cpu_inputs.reserve(inputs.size());
            for (const auto& t : inputs) cpu_inputs.push_back(t.to(Device::cpu()));
            auto cpu_result = dispatch(op, cpu_inputs, attrs);
            std::vector<Tensor> gpu_result;
            gpu_result.reserve(cpu_result.size());
            for (auto& t : cpu_result) gpu_result.push_back(t.to(dev));
            return gpu_result;
        });
    };
    auto mps_roundtrip_single = [&](OpId op) {
        table.register_single_output_kernel(op,
            [op](std::span<const Tensor> inputs,
                 const OpAttributes& attrs) -> Tensor {
                auto dev = inputs[0].device();
                std::vector<Tensor> cpu_inputs;
                cpu_inputs.reserve(inputs.size());
                for (const auto& t : inputs) cpu_inputs.push_back(t.to(Device::cpu()));
                auto cpu_result = dispatch(op, cpu_inputs, attrs);
                return cpu_result[0].to(dev);
            });
    };

    // FFT family (1-D and N-D; 2-D uses FFTN internally in Tenzor)
    mps_roundtrip_single(OpId::FFT);
    mps_roundtrip_single(OpId::IFFT);
    mps_roundtrip_single(OpId::RFFT);
    mps_roundtrip_single(OpId::IRFFT);
    mps_roundtrip_single(OpId::FFTN);
    mps_roundtrip_single(OpId::IFFTN);

    // Linalg family — multi-output ops first
    mps_roundtrip_multi(OpId::LinalgSVD);
    mps_roundtrip_multi(OpId::LinalgQR);
    mps_roundtrip_multi(OpId::LinalgEigh);
    mps_roundtrip_multi(OpId::LinalgLU);
    // Single-output linalg
    mps_roundtrip_single(OpId::LinalgDet);
    mps_roundtrip_single(OpId::LinalgInv);
    mps_roundtrip_single(OpId::LinalgSolve);
    mps_roundtrip_single(OpId::LinalgCholesky);
    mps_roundtrip_single(OpId::LinalgLUSolve);

    // Sparse family
    mps_roundtrip_single(OpId::SparseSpMM);
    mps_roundtrip_single(OpId::SparseSpMV);
    mps_roundtrip_single(OpId::SparseToDense);
    mps_roundtrip_multi(OpId::DenseToSparse);
    mps_roundtrip_single(OpId::SparseAdd);

    // Signal processing
    mps_roundtrip_single(OpId::STFT);
    mps_roundtrip_single(OpId::ISTFT);
    mps_roundtrip_single(OpId::CDist);

    // 3D / extended conv variants (most common ones needed for video models)
    mps_roundtrip_single(OpId::Conv3dForward);
    mps_roundtrip_single(OpId::MaxPool3dForward);
    mps_roundtrip_single(OpId::AvgPool3dForward);

    // New Phase 4 ops — CPU roundtrip (native Metal candidates for future)
    mps_roundtrip_single(OpId::Frac);
    mps_roundtrip_single(OpId::Heaviside);
    mps_roundtrip_single(OpId::NanToNum);
    mps_roundtrip_single(OpId::LogSigmoid);
    mps_roundtrip_single(OpId::LogSigmoidBackward);
    mps_roundtrip_single(OpId::RReLU);
    mps_roundtrip_single(OpId::RReLUBackward);
    mps_roundtrip_single(OpId::BitwiseAnd);
    mps_roundtrip_single(OpId::BitwiseOr);
    mps_roundtrip_single(OpId::BitwiseXor);
    mps_roundtrip_single(OpId::BitwiseNot);
    mps_roundtrip_single(OpId::BitwiseLeftShift);
    mps_roundtrip_single(OpId::BitwiseRightShift);
    mps_roundtrip_single(OpId::CountNonzero);
    mps_roundtrip_single(OpId::Nansum);
    mps_roundtrip_single(OpId::Nanmean);
    mps_roundtrip_multi(OpId::Aminmax);
    mps_roundtrip_single(OpId::IndexAdd);
    mps_roundtrip_single(OpId::IndexCopy);
    mps_roundtrip_single(OpId::IndexFill);

    // ================================================================
    // Tier 3 expansion: CPU-roundtrip for ALL remaining ops
    // ================================================================
    // Generated from CPU backend's registered ops minus the MPS ops
    // already registered above. This ensures no "unsupported operation"
    // crashes for any model that runs on CPU.

    // --- Single-output ops ---
    for (auto op : {
        OpId::AdaptiveAvgPool1d, OpId::AdaptiveAvgPool1dBackward,
        OpId::AdaptiveAvgPool2d, OpId::AdaptiveAvgPool2dBackward,
        OpId::AdaptiveAvgPool3d, OpId::AdaptiveAvgPool3dBackward,
        OpId::AdaptiveMaxPool1dBackward, OpId::AdaptiveMaxPool2dBackward,
        OpId::AdaptiveMaxPool3dBackward,
        OpId::AdvancedIndex, OpId::AdvancedIndexPut,
        OpId::AffineGrid, OpId::ArgSort,
        OpId::AvgPool1dBackward, OpId::AvgPool1dForward,
        OpId::AvgPool2dBackward, OpId::AvgPool2dForward, OpId::AvgPool3dBackward,
        OpId::BatchNorm2dForward,
        OpId::Bernoulli, OpId::BoxIoU, OpId::Bucketize,
        OpId::Cat, OpId::ClampMax, OpId::ClampMin, OpId::Cross,
        OpId::CumProd, OpId::CumSum, OpId::Diag,
        OpId::DropoutBackward, OpId::Elu, OpId::EluBackward,
        OpId::EmbeddingBagBackward, OpId::EmbeddingBagForward,
        OpId::FFT2, OpId::Fill, OpId::Flip, OpId::Fold,
        OpId::FusedBatchNormReLU, OpId::FusedConv2dBnReLU,
        OpId::FusedConv2dReLU, OpId::FusedConv2dSigmoid,
        OpId::FusedConv2dSwish, OpId::FusedConv2dTanh, OpId::FusedLinearReLU,
        OpId::Gather, OpId::GridSample, OpId::GumbelSoftmax,
        OpId::IFFT2, OpId::IndexSelect, OpId::Interpolate,
        OpId::LeakyReLU, OpId::LeakyReLUBackward,
        OpId::LogSoftmax, OpId::LogSoftmaxBackward,
        OpId::MaskedFill, OpId::MaxPool1dBackward, OpId::MaxPool2dBackward,
        OpId::MaxPool3dBackward,
        OpId::Multinomial, OpId::Nonzero, OpId::Norm, OpId::OneHot,
        OpId::Polygamma, OpId::Pow, OpId::Put,
        OpId::QuantizedConv2d, OpId::QuantizedLinear,
        OpId::Repeat, OpId::Roll, OpId::Scatter, OpId::ScatterAdd,
        OpId::SearchSorted, OpId::Slice, OpId::SoftmaxBackward,
        OpId::Softplus, OpId::SoftplusBackward,
        OpId::SparseTrsm, OpId::SparseTrsv,
        OpId::Stack, OpId::Std,
        OpId::Take, OpId::Tile, OpId::ToMemoryFormat,
        OpId::Trace, OpId::Tril, OpId::Triu, OpId::Unfold,
        OpId::Var
    }) {
        mps_roundtrip_single(op);
    }

    // --- Multi-output ops ---
    for (auto op : {
        OpId::AdaptiveMaxPool1d, OpId::AdaptiveMaxPool2d, OpId::AdaptiveMaxPool3d,
        OpId::Arange, OpId::BatchNorm2dBackward,
        OpId::BatchNorm2dFusedTraining, OpId::BatchNorm2dMeanVar,
        OpId::BatchNorm2dUpdateRunningStats,
        OpId::BetaInc, OpId::BiLSTMForward, OpId::Chunk,
        OpId::Conv1dForward, OpId::Conv1dBackwardInput,
        OpId::Conv1dBackwardWeight, OpId::Conv1dBackwardBias,
        OpId::Conv2dBackwardBias, OpId::Conv2dBackwardInput, OpId::Conv2dBackwardWeight,
        OpId::Conv3dBackwardBias, OpId::Conv3dBackwardInput, OpId::Conv3dBackwardWeight,
        OpId::ConvTranspose2dForward, OpId::ConvTranspose3dForward,
        OpId::ConvTranspose3dBackwardBias, OpId::ConvTranspose3dBackwardInput,
        OpId::ConvTranspose3dBackwardWeight,
        OpId::DepthwiseConv2d, OpId::Dropout,
        OpId::EmbeddingBackward, OpId::Eye,
        OpId::FlashAttention, OpId::FlashAttentionBackward,
        OpId::Full, OpId::FusedAdadeltaStep, OpId::FusedAdagradStep,
        OpId::FusedAdamAtan2Step, OpId::FusedAttention,
        OpId::FusedLayerNormBackward, OpId::FusedRMSNorm, OpId::FusedRMSPropStep,
        OpId::FusedSoftmaxCrossEntropy, OpId::GatherRelativePositionBias,
        OpId::GroupNorm, OpId::GroupNormBackward,
        OpId::GRUCellBackward, OpId::GRUCellForward,
        OpId::GRUForward, OpId::GRUMultiLayerForward,
        OpId::Histogram, OpId::InstanceNorm, OpId::InstanceNormBackward,
        OpId::LayerNorm, OpId::LayerNormBackward,
        OpId::Lerp, OpId::LinalgEig, OpId::LinearBackward,
        OpId::Linspace,
        OpId::LSTMCellBackward, OpId::LSTMCellForward,
        OpId::LSTMForward, OpId::LSTMMultiLayerForward,
        OpId::MaxPool1dForward, OpId::MaxPool2dForward,
        OpId::Median, OpId::Mode,
        OpId::NMS, OpId::Ones, OpId::Rand, OpId::Randint, OpId::Randn,
        OpId::RMSNorm, OpId::RMSNormBackward,
        OpId::ROIAlignBackward, OpId::ROIAlignForward,
        OpId::Sort, OpId::SparseSpGEMM, OpId::Split,
        OpId::TopK, OpId::Unique, OpId::Zeros
    }) {
        mps_roundtrip_multi(op);
    }
}

} // namespace tenzor::mps

// Export function for dynamic loading
extern "C" {
    void register_kernels(tenzor::BackendDispatchTable* table) {
        if (table) {
            tenzor::mps::register_mps_kernels(*table);
        }
    }
}
