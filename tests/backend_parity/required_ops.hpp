/**
 * @file required_ops.hpp
 * @brief Curated list of OpIds every compute backend MUST register.
 *
 * This header is the single source of truth for the enforced cross-backend
 * kernel coverage floor. It is consumed by:
 *  - tests/backend_parity/test_kernel_completeness.cpp (enforcement — fails
 *    a backend if any listed OpId is missing from its dispatch table).
 *  - New parity tests that want to assert "this op must exist everywhere"
 *    before they run.
 *
 * **How to grow the floor**
 *   When a new OpId gains parity test coverage in a new parity test binary,
 *   append the OpId(s) here in the same PR. This keeps the enforcement test
 *   green while monotonically expanding the guaranteed coverage.
 *
 * **What is NOT here**
 *   Backward-only ops, in-place variants, creation ops (ones/zeros/randn),
 *   and narrowly-specialized or fused ops are intentionally excluded — the
 *   floor is "what does every backend need to be useful", not "every OpId".
 */

#pragma once

#include <tenzor/ops/op_id.hpp>
#include <vector>

namespace tenzor::testing {

/// Returns the required-OpId floor list. See file-level documentation above.
inline std::vector<OpId> get_required_ops() {
    return {
        // Arithmetic (0-6)
        OpId::Add, OpId::Sub, OpId::Mul, OpId::Div,
        OpId::MatMul, OpId::Bmm, OpId::Dot,

        // Reductions (11-20)
        OpId::Sum, OpId::Mean, OpId::Max, OpId::Min,
        OpId::ArgMax, OpId::ArgMin, OpId::Prod,
        OpId::Var, OpId::Std, OpId::Norm,

        // Element-wise math (30-42)
        OpId::Sqrt, OpId::Neg, OpId::Abs, OpId::Sign,
        OpId::Log, OpId::Exp, OpId::Pow, OpId::Clamp,
        OpId::Reciprocal, OpId::Floor, OpId::Ceil, OpId::Round,

        // Trigonometric (50-61)
        OpId::Sin, OpId::Cos, OpId::Tan,
        OpId::Asin, OpId::Acos, OpId::Atan,
        OpId::Sinh, OpId::Cosh, OpId::Tanh,
        OpId::Asinh, OpId::Acosh, OpId::Atanh,

        // Activations — forward only (65-94)
        OpId::ReLU, OpId::Sigmoid, OpId::TanhActivation,
        OpId::Gelu, OpId::Swish, OpId::LeakyReLU,
        OpId::Elu, OpId::Selu, OpId::Mish,
        OpId::Softplus, OpId::Softmax, OpId::LogSoftmax,
        OpId::LogSigmoid,

        // Shape/View (100-114)
        OpId::Reshape, OpId::Transpose, OpId::Permute,
        OpId::Squeeze, OpId::Unsqueeze, OpId::Flatten,
        OpId::Contiguous, OpId::Clone, OpId::Fill,
        OpId::Repeat, OpId::Tile, OpId::Expand,
        OpId::Stack, OpId::Split, OpId::Chunk,

        // Indexing (120-130)
        OpId::IndexSelect, OpId::Gather, OpId::Scatter,
        OpId::MaskedSelect, OpId::MaskedFill, OpId::Where,
        OpId::Slice, OpId::Cat, OpId::Take, OpId::Put,
        OpId::Nonzero,

        // Comparison (140-145)
        OpId::Eq, OpId::Ne, OpId::Lt, OpId::Le, OpId::Gt, OpId::Ge,

        // Conv2d forward (170)
        OpId::Conv2dForward,

        // Pooling forward (190-196)
        OpId::MaxPool2dForward, OpId::AvgPool2dForward,
        OpId::AdaptiveAvgPool2d, OpId::AdaptiveMaxPool2d,

        // Embedding (260)
        OpId::Embedding,

        // Linear (270)
        OpId::Linear,

        // Cast (316)
        OpId::Cast,

        // Extended math (320-335)
        OpId::Log2, OpId::Log10, OpId::Log1p,
        OpId::Exp2, OpId::Expm1,
        OpId::Erf, OpId::Erfc,
        OpId::Atan2, OpId::Fmod, OpId::Remainder,
        OpId::Hypot, OpId::Copysign,

        // Manipulation (340-345)
        OpId::Triu, OpId::Tril, OpId::Diag,
        OpId::Trace, OpId::Flip, OpId::Roll,

        // Logical (350-353)
        OpId::LogicalAnd, OpId::LogicalOr,
        OpId::LogicalNot, OpId::LogicalXor,

        // Complex (440-444)
        OpId::Conj, OpId::Real, OpId::Imag,
        OpId::Angle, OpId::Polar,

        // ====================================================================
        // Training correctness floor (added Phase 1.4)
        // Without the backward kernels below, a model cannot train on the
        // affected backend — forward-only registration is insufficient.
        // ====================================================================

        // Activation backwards (pair with the forwards above)
        OpId::ReLUBackward, OpId::SigmoidBackward, OpId::GeluBackward,
        OpId::SwishBackward, OpId::LeakyReLUBackward, OpId::EluBackward,
        OpId::SeluBackward, OpId::MishBackward, OpId::SoftplusBackward,
        OpId::SoftmaxBackward, OpId::LogSoftmaxBackward, OpId::LogSigmoidBackward,

        // Normalization backwards
        OpId::BatchNorm2dBackward, OpId::LayerNormBackward,
        OpId::GroupNormBackward, OpId::InstanceNormBackward,
        OpId::RMSNormBackward,

        // RNN cell + full-sequence (fwd + bwd where a dedicated OpId exists)
        // Note: only cell-level backward OpIds exist — full-sequence backward
        // goes through cell-level autograd. See op_id.hpp lines 489-491.
        OpId::LSTMCellForward, OpId::LSTMCellBackward,
        OpId::GRUCellForward,  OpId::GRUCellBackward,
        OpId::LSTMForward, OpId::GRUForward,

        // Embedding backward (forward is already in the floor above)
        OpId::EmbeddingBackward,

        // ====================================================================
        // Extended training floor (added Phase 1.5)
        // ====================================================================

        // Conv2d backward (training requires these — forward-only is insufficient)
        OpId::Conv2dBackwardInput, OpId::Conv2dBackwardWeight,
        OpId::Conv2dBackwardBias,

        // Dropout (every training model uses this)
        OpId::Dropout, OpId::DropoutBackward,

        // Scatter (critical for embedding gradients and GNN workloads)
        OpId::ScatterAdd,

        // TopK and Sort (beam search, NMS, metric computation)
        OpId::TopK, OpId::Sort,

        // Creation ops (fundamental — backend is broken without these)
        OpId::Zeros, OpId::Ones, OpId::Full, OpId::Rand, OpId::Randn,
        OpId::Arange, OpId::Eye,

        // ====================================================================
        // Phase 1 coverage floor (added Phase 3.3 of the test-coverage plan)
        //
        // These ops have multidtype parity test coverage in Phase 1 and are
        // verified registered on all 5 backends per
        // tests/backend_parity/baselines/registration_report.txt.
        // ====================================================================

        // Conv1d — forward + 3 backward variants
        OpId::Conv1dForward,
        OpId::Conv1dBackwardInput, OpId::Conv1dBackwardWeight,
        OpId::Conv1dBackwardBias,

        // ConvTranspose3d backward (forward is reduced via im2col +
        // Conv2dForward on several backends; the backward OpIds are distinct)
        OpId::ConvTranspose3dBackwardInput,
        OpId::ConvTranspose3dBackwardWeight,
        OpId::ConvTranspose3dBackwardBias,

        // Fractional + MaxUnpool pooling variants — all 5 backends
        OpId::FractionalMaxPool2dForward,  OpId::FractionalMaxPool2dBackward,
        OpId::FractionalMaxPool3dForward,  OpId::FractionalMaxPool3dBackward,
        OpId::MaxUnpool2dForward,

        // Signal processing — STFT/ISTFT/DCT all have cross-backend coverage
        OpId::STFT, OpId::ISTFT, OpId::DCT,

        // Vision detection / sampling — NMS and GridSample are universally
        // supported and covered by vision_fused_parity.
        OpId::NMS, OpId::GridSample,
        OpId::ROIAlignForward, OpId::ROIAlignBackward,

        // Sampling primitives — Bernoulli and Multinomial are in the
        // training-critical path for dropout masks and categorical policies.
        OpId::Bernoulli, OpId::Multinomial,

        // Reduction extensions covered by parity tests
        OpId::Aminmax,

        // Normalization forward (backward already in Phase 1.4 floor above)
        OpId::RMSNorm, OpId::FusedRMSNorm,

        // Histogram family (test_histogramdd_multidtype + test_histogram)
        OpId::Histogram, OpId::Histogramdd,

        // ====================================================================
        // Audit-driven floor expansion (Phase 3 of test-coverage plan)
        //
        // After the 2026-05-02 audit confirmed every backend (CPU/CUDA/ROCm/
        // Vulkan/OneAPI) registers all 462 valid OpIds (RegistrationReport
        // shows 100% coverage on each), we promote every previously-skipped
        // family to the enforced floor. This means a backend cannot silently
        // drop one of these kernels without test_kernel_completeness failing.
        //
        // Excludes only:
        //  - MPS (out of scope per the audit plan)
        //  - Specialized fused ops (FusedConv2d* variants), which are perf
        //    optimizations rather than a correctness floor
        // ====================================================================

        // Linalg (420-427, 510-518, 600-607) — full coverage now confirmed
        OpId::LinalgDet, OpId::LinalgInv, OpId::LinalgSolve,
        OpId::LinalgSVD, OpId::LinalgQR, OpId::LinalgEigh, OpId::LinalgEig,
        OpId::LinalgCholesky,
        OpId::LinalgLU, OpId::LinalgLUSolve, OpId::LinalgHouseholder,
        OpId::LinalgLDLFactor, OpId::LinalgLDLSolve,
        OpId::LinalgVectorNorm, OpId::LinalgMatrixNorm,
        OpId::LinalgVecdot, OpId::LinalgCholeskySolve,
        OpId::DiagEmbed, OpId::Diagflat,
        OpId::SolveTriangular, OpId::CholeskyInverse,
        OpId::TensorInv, OpId::TensorSolve,
        OpId::Ormqr, OpId::Geqrf,

        // FFT family (400-407)
        OpId::FFT, OpId::IFFT, OpId::RFFT, OpId::IRFFT,
        OpId::FFT2, OpId::IFFT2, OpId::FFTN, OpId::IFFTN,

        // Sparse family (460-469)
        OpId::SparseSpMM, OpId::SparseSpMV,
        OpId::SparseToDense, OpId::DenseToSparse,
        OpId::SparseAdd, OpId::SparseSpGEMM,
        OpId::SparseTrsv, OpId::SparseTrsm,
        OpId::SparseSoftmax, OpId::SparseLogSoftmax,

        // Index family (132-133, 411-416, 610-611)
        OpId::AdvancedIndex, OpId::AdvancedIndexPut,
        OpId::IndexAdd, OpId::IndexCopy, OpId::IndexFill,
        OpId::SelectScatter, OpId::SliceScatter, OpId::DiagonalScatter,
        OpId::TakeAlongDim, OpId::MaskedScatter,

        // Cumulative / scan (292-293, 544, 560, 580-581)
        OpId::CumSum, OpId::CumProd,
        OpId::Logcumsumexp,
        OpId::CumMax, OpId::CumMin,
        OpId::CumulativeTrapezoid,

        // Bitwise (520-525)
        OpId::BitwiseAnd, OpId::BitwiseOr, OpId::BitwiseXor,
        OpId::BitwiseNot, OpId::BitwiseLeftShift, OpId::BitwiseRightShift,

        // Special math (490-504, 565, 685-688)
        OpId::Gamma, OpId::Lgamma, OpId::Digamma, OpId::Polygamma,
        OpId::Beta, OpId::BetaInc,
        OpId::BesselJ0, OpId::BesselJ1, OpId::BesselY0, OpId::BesselY1,
        OpId::BesselI0, OpId::BesselI1,
        OpId::ErfInv, OpId::Sinc, OpId::Zeta,
        OpId::Multigammaln,
        OpId::I0e, OpId::I1e, OpId::Entr, OpId::SphericalBesselJ0,

        // Forward+backward pairs for newer attentions (295-296, 693-694, 675-676)
        OpId::FlashAttention, OpId::FlashAttentionBackward,
        OpId::FlexAttention, OpId::FlexAttentionBackward,
        OpId::NestedAttention, OpId::NestedAttentionBackward,

        // EmbeddingBag forward + backward (435-436)
        OpId::EmbeddingBagForward, OpId::EmbeddingBagBackward,

        // DeformableConv2d full set (184-187)
        OpId::DeformableConv2dForward,
        OpId::DeformableConv2dBackwardInput,
        OpId::DeformableConv2dBackwardWeight,
        OpId::DeformableConv2dBackwardBias,

        // Inplace ops (7-10, 89-93)
        OpId::AddInplace, OpId::SubInplace, OpId::MulInplace, OpId::DivInplace,
        OpId::ReLUInplace, OpId::SigmoidInplace, OpId::TanhInplace,
        OpId::LeakyReLUInplace, OpId::GeluInplace,

        // Signal processing tail (474-476)
        OpId::IDCT, OpId::MelScale, OpId::MFCC,

        // Phase-5 PyTorch-parity math
        OpId::Deg2Rad, OpId::Rad2Deg, OpId::Logit, OpId::Signbit,
        OpId::FloatPower, OpId::Xlog1py, OpId::Ldexp,
        OpId::IsReal, OpId::IsPosInf, OpId::IsNegInf, OpId::Frexp,

        // Numerically stable math (680-684)
        OpId::LogAddExp, OpId::LogAddExp2,
        OpId::XLogY, OpId::CosineSimilarity, OpId::Renorm,

        // NaN-aware statistics + quantiles + Cov/Corrcoef
        OpId::NanVar, OpId::NanStd, OpId::Nanmedian,
        OpId::Quantile, OpId::Nanquantile,
        OpId::Histc, OpId::Kthvalue, OpId::UniqueConsecutive,
        OpId::Cov, OpId::Corrcoef,
    };
}

}  // namespace tenzor::testing
