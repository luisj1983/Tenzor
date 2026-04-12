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

namespace tenzor::mps {

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

    // Reductions (needed by backward pass: sum for gradient accumulation, mean for losses)
    table.register_kernel(OpId::Sum, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto dev = inputs[0].device();
        auto cpu_in = inputs[0].to(Device::cpu());
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool keepdim = attrs.get_bool(AttrKey::KeepDim, false);
        if (dim >= 0) {
            return std::vector<Tensor>{tenzor::sum(cpu_in, dim, keepdim).to(dev)};
        }
        return std::vector<Tensor>{tenzor::sum(cpu_in).to(dev)};
    });

    table.register_kernel(OpId::Mean, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto dev = inputs[0].device();
        auto cpu_in = inputs[0].to(Device::cpu());
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool keepdim = attrs.get_bool(AttrKey::KeepDim, false);
        if (dim >= 0) {
            return std::vector<Tensor>{tenzor::mean(cpu_in, dim, keepdim).to(dev)};
        }
        return std::vector<Tensor>{tenzor::mean(cpu_in).to(dev)};
    });

    table.register_kernel(OpId::Max, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto dev = inputs[0].device();
        auto cpu_in = inputs[0].to(Device::cpu());
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool keepdim = attrs.get_bool(AttrKey::KeepDim, false);
        auto [values, indices] = tenzor::max(cpu_in, dim, keepdim);
        return std::vector<Tensor>{values.to(dev), indices.to(dev)};
    });

    // Shape operations (needed by backward: reshape for gradient matching)
    table.register_single_output_kernel(OpId::Reshape,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            // Reshape is a metadata-only op — just do it
            return inputs[0].to(Device::cpu()).reshape(
                std::vector<int64_t>(inputs[0].shape().begin(), inputs[0].shape().end())).to(inputs[0].device());
        });

    table.register_single_output_kernel(OpId::Transpose,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            auto dev = inputs[0].device();
            return inputs[0].to(Device::cpu()).transpose().to(dev);
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

    // Gt stays on the CPU fallback for now: the native gt_kernel would
    // write Float32 0/1 but the rest of Tenzor expects a Bool output
    // tensor, and `dispatch_binary` creates the output with the input
    // dtype. Making GT native needs a dedicated comparison dispatcher
    // that allocates a Bool output. Leaving as CPU fallback until that
    // helper lands.
    table.register_kernel(OpId::Gt, [](std::span<const Tensor> inputs, const OpAttributes&) {
        auto dev = inputs[0].device();
        return std::vector<Tensor>{tenzor::gt(inputs[0].to(Device::cpu()), inputs[1].to(Device::cpu())).to(dev)};
    });

    // Note: zeros_like / ones_like are library-level free functions in
    // tenzor::ops, not dispatch-level OpIds. Autograd code that needs
    // gradient-init scratch tensors should call tenzor::zeros_like(x) /
    // tenzor::ones_like(x), which internally dispatches zeros()/ones()
    // for the tensor's device. No MPS-specific registration is needed.

    // Fused optimizer steps (SGD, Adam) via CPU roundtrip
    table.register_kernel(OpId::FusedSGDStep, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto dev = inputs[0].device();
        std::vector<Tensor> cpu_inputs;
        for (const auto& t : inputs) cpu_inputs.push_back(t.to(Device::cpu()));
        auto result = dispatch(OpId::FusedSGDStep, cpu_inputs, attrs);
        std::vector<Tensor> gpu_result;
        for (const auto& t : result) gpu_result.push_back(t.to(dev));
        return gpu_result;
    });

    table.register_kernel(OpId::FusedAdamStep, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto dev = inputs[0].device();
        std::vector<Tensor> cpu_inputs;
        for (const auto& t : inputs) cpu_inputs.push_back(t.to(Device::cpu()));
        auto result = dispatch(OpId::FusedAdamStep, cpu_inputs, attrs);
        std::vector<Tensor> gpu_result;
        for (const auto& t : result) gpu_result.push_back(t.to(dev));
        return gpu_result;
    });

    // Cast (dtype conversion needed during training)
    table.register_single_output_kernel(OpId::Cast,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            auto dev = inputs[0].device();
            auto cpu_in = inputs[0].to(Device::cpu());
            auto result = dispatch(OpId::Cast, std::vector<Tensor>{cpu_in}, attrs);
            return result[0].to(dev);
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
        OpId::Expand, OpId::FFT2, OpId::Fill, OpId::Flatten, OpId::Flip, OpId::Fold,
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
        OpId::Permute, OpId::Polygamma, OpId::Pow, OpId::Put,
        OpId::QuantizedConv2d, OpId::QuantizedLinear,
        OpId::Repeat, OpId::Roll, OpId::Scatter, OpId::ScatterAdd,
        OpId::SearchSorted, OpId::Slice, OpId::SoftmaxBackward,
        OpId::Softplus, OpId::SoftplusBackward,
        OpId::SparseTrsm, OpId::SparseTrsv,
        OpId::Squeeze, OpId::Stack, OpId::Std,
        OpId::Take, OpId::Tile, OpId::ToMemoryFormat,
        OpId::Trace, OpId::Tril, OpId::Triu, OpId::Unfold,
        OpId::Unsqueeze, OpId::Var
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
