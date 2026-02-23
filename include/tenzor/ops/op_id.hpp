/**
 * @file op_id.hpp
 * @brief Unified operation identifier enum for O(1) dispatch
 *
 * This enum replaces string-based operation dispatch with numeric IDs.
 * Each operation has a unique ID that maps directly to a function pointer
 * in the dispatch table, enabling single-dispatch O(1) kernel lookup.
 *
 * Design principles:
 * - Contiguous values for array indexing
 * - Grouped by category for cache locality
 * - OP_COUNT sentinel for compile-time array sizing
 */

#pragma once

#include <cstdint>
#include <string_view>

namespace tenzor {

/**
 * @brief Enumeration of all supported operations.
 *
 * Used for O(1) dispatch via function pointer table lookup.
 * Unifies the previous jit::OpType with all backend operations.
 */
enum class OpId : uint16_t {
    // =========================================================================
    // Arithmetic Operations (0-9)
    // =========================================================================
    Add = 0,
    Sub,
    Mul,
    Div,
    MatMul,
    Bmm,
    Dot,
    AddInplace,
    SubInplace,
    MulInplace,
    DivInplace,

    // =========================================================================
    // Reduction Operations (11-29)
    // =========================================================================
    Sum = 11,
    Mean,
    Max,
    Min,
    ArgMax,
    ArgMin,
    Prod,
    Var,
    Std,
    Norm,
    ArgSort,

    // =========================================================================
    // Element-wise Math (30-49)
    // =========================================================================
    Sqrt = 30,
    Neg,
    Abs,
    Sign,
    Log,
    Exp,
    Pow,
    Clamp,
    ClampMin,
    ClampMax,
    Reciprocal,
    Floor,
    Ceil,
    Round,

    // =========================================================================
    // Trigonometric Operations (50-64)
    // =========================================================================
    Sin = 50,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,  // Note: distinct from TanhActivation for autograd

    // =========================================================================
    // Activation Functions (65-99)
    // =========================================================================
    ReLU = 65,
    ReLUBackward,
    Sigmoid,
    SigmoidBackward,
    TanhActivation,
    TanhBackward,
    Gelu,
    GeluBackward,
    Swish,
    SwishBackward,
    LeakyReLU,
    LeakyReLUBackward,
    Elu,
    EluBackward,
    Selu,
    SeluBackward,
    Mish,
    MishBackward,
    Softplus,
    SoftplusBackward,
    Softmax,
    SoftmaxBackward,
    LogSoftmax,
    LogSoftmaxBackward,
    // In-place activation variants
    ReLUInplace,
    SigmoidInplace,
    TanhInplace,
    LeakyReLUInplace,
    GeluInplace,

    // =========================================================================
    // Shape/View Operations (100-119)
    // =========================================================================
    Reshape = 100,
    Transpose,
    Permute,
    Squeeze,
    Unsqueeze,
    Flatten,
    Contiguous,
    Clone,
    Fill,
    Repeat,
    Tile,
    Expand,
    Stack,
    Split,
    Chunk,
    ToMemoryFormat,  // Memory format conversion (NCHW <-> NHWC)

    // =========================================================================
    // Indexing Operations (120-139)
    // =========================================================================
    IndexSelect = 120,
    Gather,
    Scatter,
    MaskedSelect,
    MaskedFill,
    Where,
    Slice,
    Cat,
    Take,
    Put,
    Nonzero,
    OneHot,

    // =========================================================================
    // Comparison Operations (140-149)
    // =========================================================================
    Eq = 140,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // =========================================================================
    // Normalization Operations (150-169)
    // =========================================================================
    BatchNorm2dMeanVar = 150,
    BatchNorm2dForward,
    BatchNorm2dForwardAffine,
    BatchNorm2dUpdateRunningStats,
    BatchNorm2dBackward,
    BatchNorm2dFusedTraining,  // cuDNN fused forward training
    LayerNorm,
    LayerNormBackward,
    GroupNorm,
    GroupNormBackward,
    InstanceNorm,
    InstanceNormBackward,
    RMSNormBackward,

    // =========================================================================
    // Convolution Operations (170-189)
    // =========================================================================
    Conv2dForward = 170,
    Conv2dBackwardInput,
    Conv2dBackwardWeight,
    Conv2dBackwardBias,
    ConvTranspose2dForward,
    DepthwiseConv2d,

    // =========================================================================
    // Pooling Operations (190-209)
    // =========================================================================
    MaxPool2dForward = 190,
    MaxPool2dBackward,
    AvgPool2dForward,
    AvgPool2dBackward,
    AdaptiveAvgPool2d,
    AdaptiveAvgPool2dBackward,
    AdaptiveMaxPool2d,
    AdaptiveMaxPool2dBackward,

    // =========================================================================
    // Vision Operations (200-209)
    // =========================================================================
    Unfold = 200,
    Fold,
    Interpolate,
    ROIAlignForward,
    ROIAlignBackward,
    BoxIoU,

    // =========================================================================
    // Fused Operations (210-229)
    // =========================================================================
    FusedLinearReLU = 210,
    FusedConv2dReLU,
    FusedBatchNormReLU,
    FusedSoftmaxCrossEntropy,
    FusedAddReLU,
    FusedGelu,
    FusedLayerNorm,
    FusedRMSNorm,
    FusedAttention,
    FusedSGDStep,
    FusedAdamStep,
    FusedRMSPropStep,
    FusedAdadeltaStep,
    FusedAdagradStep,
    FusedAdamAtan2Step,
    FusedConv2dSigmoid,
    FusedConv2dTanh,
    FusedConv2dSwish,
    FusedConv2dBnReLU,
    FusedLayerNormBackward,

    // =========================================================================
    // Creation Operations (230-249)
    // =========================================================================
    Zeros = 230,
    Ones,
    Full,
    Rand,
    Randn,
    Arange,
    Linspace,
    Eye,

    // =========================================================================
    // RNN Operations (250-259)
    // =========================================================================
    LSTMCellForward = 250,
    LSTMCellBackward,
    GRUCellForward,
    GRUCellBackward,
    LSTMForward,              // Full sequence LSTM forward (single layer)
    GRUForward,               // Full sequence GRU forward (single layer)
    LSTMMultiLayerForward,    // Fused multi-layer LSTM forward
    GRUMultiLayerForward,     // Fused multi-layer GRU forward
    BiLSTMForward,            // Bidirectional LSTM forward (single layer)

    // =========================================================================
    // Embedding Operations (260-269)
    // =========================================================================
    Embedding = 260,
    EmbeddingBackward,

    // =========================================================================
    // Linear/FC Operations (270-279)
    // =========================================================================
    Linear = 270,
    LinearBackward,

    // =========================================================================
    // Dropout (280-284)
    // =========================================================================
    Dropout = 280,
    DropoutBackward,

    // =========================================================================
    // JIT-specific markers (285-289)
    // =========================================================================
    Constant = 285,
    Input,
    Output,

    // =========================================================================
    // Advanced Operations (290-299)
    // =========================================================================
    TopK = 290,
    Sort,
    CumSum,
    CumProd,
    Unique,
    FlashAttention = 295,
    FlashAttentionBackward,

    // =========================================================================
    // 3D Convolution and Pooling Operations (301-315)
    // =========================================================================
    Conv3dForward = 301,
    Conv3dBackwardInput,
    Conv3dBackwardWeight,
    Conv3dBackwardBias,
    MaxPool3dForward,
    MaxPool3dBackward,
    AvgPool3dForward,
    AvgPool3dBackward,

    // =========================================================================
    // Type Conversion Operations (316-319)
    // =========================================================================
    Cast = 316,

    // =========================================================================
    // Extended Math Operations (320-339)
    // =========================================================================
    Log2 = 320,
    Log10,
    Log1p,
    Exp2,
    Expm1,
    Erf,
    Erfc,
    IsNan,
    IsInf,
    IsFinite,
    Atan2,       // binary: atan2(y, x)
    Fmod,        // binary: fmod(a, b)
    Remainder,   // binary: remainder(a, b)
    Lerp,        // ternary: lerp(start, end, weight)

    // =========================================================================
    // Sentinel (MUST BE LAST)
    // =========================================================================
    OP_COUNT
};

/// Compile-time constant for dispatch table sizing
inline constexpr size_t OP_COUNT = static_cast<size_t>(OpId::OP_COUNT);

/**
 * @brief Convert OpId to string for error messages and debugging.
 *
 * @param id Operation identifier
 * @return String representation of the operation
 */
auto op_id_to_name(OpId id) noexcept -> std::string_view;

/**
 * @brief Check if an OpId is valid (within range).
 *
 * @param id Operation identifier to check
 * @return true if valid, false otherwise
 */
inline constexpr bool is_valid_op_id(OpId id) noexcept {
    return static_cast<uint16_t>(id) < static_cast<uint16_t>(OpId::OP_COUNT);
}

} // namespace tenzor
