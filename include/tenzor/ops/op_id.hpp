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
 *
 * Layout: Values are intentionally sparse with range-based allocation per
 * category (e.g., Arithmetic 0-9, Reductions 11-29, Activations 65-99).
 * Gaps between ranges allow adding new ops to a category without renumbering.
 * The dispatch table is sized by OP_COUNT (~380 entries, ~3.5KB/device),
 * so the memory overhead of gaps is negligible.
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
    Any,     // = 22, Boolean any reduction
    All,     // = 23, Boolean all reduction
    Median,  // = 24
    Mode,    // = 25
    CountNonzero,  // = 26
    Nansum,        // = 27
    Nanmean,       // = 28
    Aminmax,       // = 29, simultaneous min+max in single pass (returns tuple)

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
    Trunc,
    Frac,                // = 38, fractional part: x - floor(x)
    Heaviside,           // = 39, step function: 0 if x<0, val if x==0, 1 if x>0
    NanToNum,            // = 40, replace NaN/Inf with specified values
    Rsqrt,               // = 41, reciprocal square root: 1/sqrt(x)
    Square,              // = 42, element-wise square: x*x

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
    Asinh,         // = 59, inverse hyperbolic sine
    Acosh,         // = 60, inverse hyperbolic cosine
    Atanh,         // = 61, inverse hyperbolic tangent

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
    LogSigmoid,          // = 94, log(sigmoid(x)) = -softplus(-x), numerically stable
    LogSigmoidBackward,  // = 95
    RReLU,               // = 96, randomized leaky ReLU
    RReLUBackward,       // = 97

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
    AdvancedIndex,    // = 132, NumPy-style fancy indexing (gather with multiple index tensors)
    AdvancedIndexPut, // = 133, In-place scatter with multiple index tensors

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
    RMSNorm,
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
    ConvTranspose3dForward,      // 176
    ConvTranspose3dBackwardInput,
    ConvTranspose3dBackwardWeight,
    ConvTranspose3dBackwardBias,
    Conv1dForward = 180,             // 1D convolution forward pass
    Conv1dBackwardInput,             // 1D convolution gradient w.r.t. input
    Conv1dBackwardWeight,            // 1D convolution gradient w.r.t. weight
    Conv1dBackwardBias,              // 1D convolution gradient w.r.t. bias

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
    GatherRelativePositionBias,
    NMS,                       // Non-Maximum Suppression
    GridSample,                // F.grid_sample spatial transformer
    AffineGrid,                // F.affine_grid for grid generation

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
    Randint,

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
    AdaptiveMaxPool3d,       // 309
    AdaptiveMaxPool3dBackward,
    AdaptiveAvgPool3d,
    AdaptiveAvgPool3dBackward,

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
    Hypot,       // binary: sqrt(x*x + y*y) overflow-safe
    Copysign,    // binary: copysign(magnitude, sign)
    Nextafter,   // binary: next representable float
    Gcd,         // binary: greatest common divisor (integer)
    Lcm,         // binary: least common multiple (integer)
    Addcmul,     // ternary: input + value * tensor1 * tensor2

    // =========================================================================
    // Tensor Manipulation Operations (340-349)
    // =========================================================================
    Triu = 340,      // Upper triangular
    Tril,            // Lower triangular
    Diag,            // Extract/construct diagonal
    Trace,           // Sum of diagonal elements
    Flip,            // Reverse along dimension
    Roll,            // Shift elements along dimension

    // =========================================================================
    // Logical Operations (350-359)
    // =========================================================================
    LogicalAnd = 350,
    LogicalOr,
    LogicalNot,
    LogicalXor,

    // Element-wise binary ops
    // =========================================================================
    Minimum = 360,
    Maximum,
    Cross,           // 3D cross product along a dimension

    // =========================================================================
    // 1D Pooling Operations (370-379)
    // =========================================================================
    MaxPool1dForward = 370,
    MaxPool1dBackward,
    AvgPool1dForward,
    AvgPool1dBackward,
    AdaptiveMaxPool1d,
    AdaptiveMaxPool1dBackward,
    AdaptiveAvgPool1d,
    AdaptiveAvgPool1dBackward,

    // =========================================================================
    // Search and Sampling Operations (390-399)
    // =========================================================================
    SearchSorted = 390,
    GumbelSoftmax,

    // =========================================================================
    // FFT Operations (400-409)
    // =========================================================================
    FFT = 400,
    IFFT,
    RFFT,
    IRFFT,
    FFT2,
    IFFT2,
    FFTN,
    IFFTN,

    // =========================================================================
    // Indexing Operations (410-419)
    // =========================================================================
    ScatterAdd = 410,
    IndexAdd,            // = 411, self[index] += source along dim
    IndexCopy,           // = 412, self[index] = source along dim
    IndexFill,           // = 413, self[index] = value along dim

    // =========================================================================
    // Linear Algebra Operations (420-439)
    // =========================================================================
    LinalgDet = 420,
    LinalgInv,
    LinalgSolve,
    LinalgSVD,
    LinalgQR,
    LinalgEigh,
    LinalgEig,
    LinalgCholesky,

    // =========================================================================
    // Extended Operations (428-439)
    // =========================================================================
    StridedFill = 428,
    QuantizedLinear,           // 429
    QuantizedConv2d,           // 430
    EmbeddingWithBoundsCheck,  // 431
    LogSumExp = 433,           // 433 — (432 retired: WinogradConv2d was an unused enum slot; CPU conv uses winograd_conv2d_f4x3 as an internal helper, not via dispatch)
    HasInfNan,                 // 434
    EmbeddingBagForward,       // 435
    EmbeddingBagBackward,      // 436

    // =========================================================================
    // Complex Number Operations (440-449)
    // =========================================================================
    Conj = 440,                // Complex conjugate
    Real,                      // Extract real part
    Imag,                      // Extract imaginary part
    Angle,                     // Argument (phase angle)
    Polar,                     // Construct complex from magnitude and phase

    // =========================================================================
    // (450-459 previously reserved LSTMBackward/GRUBackward/BiLSTMBackward/
    // RNNForward — removed as unused. Autograd uses cell-level backward ops
    // LSTMCellBackward / GRUCellBackward / RNNCellBackward instead.)
    // =========================================================================

    // =========================================================================
    // Sparse Tensor Operations (460-469)
    //
    // These OpIds are registered via wrapper kernels that reconstruct a
    // SparseTensor from CSR components (crow_indices, col_indices, values)
    // passed as plain Tensors, then delegate to the existing sparse::
    // functions in src/sparse/sparse_ops.cpp. SpMM/SpMV on GPU are guarded
    // by TENZOR_HAS_CUSPARSE / TENZOR_HAS_ROCSPARSE / TENZOR_HAS_ONEMKL.
    // Vulkan uses its own native dispatch (registered separately).
    //
    // Per-backend coverage matrix:
    //
    //                 CPU  CUDA  ROCm  Vulkan  OneAPI
    //   SparseSpMM    yes  yes   yes   yes     yes
    //   SparseSpMV    yes  yes   yes   yes     yes
    //   SparseToDense yes  yes   yes   yes     yes
    //   DenseToSparse yes  yes   yes   yes     yes
    //   SparseAdd     yes  yes   yes   yes     yes
    //   SparseSpGEMM  yes  yes   yes   yes     yes
    //   SparseTrsv    yes  yes   yes   yes     yes
    //   SparseTrsm    yes  yes   yes   yes     yes
    // =========================================================================
    SparseSpMM = 460,          // Sparse-Dense matrix multiplication
    SparseSpMV,                // Sparse-Dense matrix-vector multiplication
    SparseToDense,             // Convert sparse tensor to dense
    DenseToSparse,             // Convert dense tensor to sparse (CSR)
    SparseAdd,                 // Sparse + Dense addition
    SparseSpGEMM = 465,        // Sparse × Sparse → Sparse (CSR × CSR → CSR)
    SparseTrsv,                // Sparse lower/upper triangular solve: L*x = b
    SparseTrsm,                // Sparse triangular multi-RHS solve: L*X = B

    // =========================================================================
    // Signal Processing Operations (470-479)
    // =========================================================================
    STFT = 470,                // Short-time Fourier transform
    ISTFT,                     // Inverse short-time Fourier transform
    CDist,                     // Pairwise distance computation

    // =========================================================================
    // Sampling and Statistics Operations (480-489)
    // =========================================================================
    Multinomial = 480,         // Weighted random sampling
    Bernoulli,                 // Bernoulli distribution sampling
    Histogram,                 // Histogram computation
    Bucketize,                 // Bucket assignment via binary search

    // =========================================================================
    // Special Math Functions (490-509)
    // =========================================================================
    Gamma = 490,               // Gamma function: Γ(x)
    Lgamma,                    // Log-gamma: ln|Γ(x)|
    Digamma,                   // Digamma (psi): ψ(x) = d/dx ln Γ(x)
    Polygamma,                 // Polygamma: ψ^(n)(x), n-th derivative of digamma
    Beta,                      // Beta function: B(a,b) = Γ(a)Γ(b)/Γ(a+b)
    BetaInc,                   // Regularized incomplete beta function: I_x(a,b)
    BesselJ0,                  // Bessel function of first kind, order 0
    BesselJ1,                  // Bessel function of first kind, order 1
    BesselY0,                  // Bessel function of second kind, order 0
    BesselY1,                  // Bessel function of second kind, order 1
    BesselI0,                  // Modified Bessel function of first kind, order 0
    BesselI1,                  // Modified Bessel function of first kind, order 1
    ErfInv,                    // Inverse error function: erfinv(x)
    Sinc,                      // Normalized sinc: sin(πx)/(πx)
    Zeta,                      // Hurwitz zeta function: ζ(x, q)

    // =========================================================================
    // Additional Linear Algebra Operations (510-519)
    // =========================================================================
    LinalgLU = 510,            // LU factorization with partial pivoting
    LinalgLUSolve,             // Solve via pre-computed LU factors

    // =========================================================================
    // Bitwise Operations (520-529)
    // =========================================================================
    BitwiseAnd = 520,          // Element-wise bitwise AND (integer types)
    BitwiseOr,                 // Element-wise bitwise OR
    BitwiseXor,                // Element-wise bitwise XOR
    BitwiseNot,                // Element-wise bitwise NOT (unary)
    BitwiseLeftShift,          // Element-wise left shift
    BitwiseRightShift,         // Element-wise right shift

    // =========================================================================
    // Scatter-Reduce Operations (530-539)
    // =========================================================================
    ScatterReduce = 530,       // Scatter with reduction (sum/prod/mean/amax/amin)

    // =========================================================================
    // Fused GEMM Operations (540-549)
    // =========================================================================
    Addmm = 540,               // beta*input + alpha*(mat1 @ mat2)
    Addmv = 541,               // beta*input + alpha*(mat @ vec)
    Baddbmm = 542,             // Batched: beta*input + alpha*(batch1 @ batch2)

    // =========================================================================
    // Repeat/Interleave Operations (550-559)
    // =========================================================================
    RepeatInterleave = 550,    // Repeat each element along a dimension

    // =========================================================================
    // Cumulative Scan Operations (560-569)
    // =========================================================================
    Logcumsumexp = 560,        // Log-cumulative-sum-exp along a dimension
    Bincount = 561,            // Count occurrences of each value in integer tensor

    // =========================================================================
    // New Element-wise Math (570-579)
    // =========================================================================
    Igamma = 570,              // Lower regularized incomplete gamma
    Igammac,                   // Upper regularized incomplete gamma (1 - igamma)
    Addcdiv,                   // ternary: input + value * tensor1 / tensor2

    // =========================================================================
    // New Reduction Operations (580-599)
    // =========================================================================
    CumMax = 580,              // Cumulative max (returns values, indices)
    CumMin,                    // Cumulative min (returns values, indices)
    Isin,                      // Set membership test
    Kthvalue,                  // k-th smallest value along dim
    Fmax,                      // Element-wise max, NaN-propagating per IEEE 754-2008
    Fmin,                      // Element-wise min, NaN-propagating per IEEE 754-2008
    Quantile,                  // Interpolated quantile along dim
    Nanquantile,               // NaN-ignoring quantile
    Nanmedian,                 // NaN-ignoring median
    Histc,                     // Fixed-bin histogram
    UniqueConsecutive,         // Deduplicate consecutive equal elements

    // =========================================================================
    // New Linear Algebra (600-609)
    // =========================================================================
    DiagEmbed = 600,           // Embed vector as batch diagonal
    Diagflat,                  // Flat input to diagonal matrix
    SolveTriangular,           // Triangular system solve

    // =========================================================================
    // New Shape/Indexing Operations (610-619)
    // =========================================================================
    TakeAlongDim = 610,        // Gather along specific dim
    MaskedScatter,             // Scatter into masked positions
    TrilIndices,               // Lower-triangular index pairs
    TriuIndices,               // Upper-triangular index pairs

    // =========================================================================
    // New Pooling Operations (620-629)
    // =========================================================================
    FractionalMaxPool2dForward = 620,
    FractionalMaxPool2dBackward,
    FractionalMaxPool3dForward,
    FractionalMaxPool3dBackward,
    MaxUnpool2dForward,
    MaxUnpool2dBackward,
    MaxUnpool3dForward,
    MaxUnpool3dBackward,

    // =========================================================================
    // Sentinel (MUST BE LAST)
    // =========================================================================
    OP_COUNT
};

/// Compile-time constant for dispatch table sizing
inline constexpr size_t OP_COUNT = static_cast<size_t>(OpId::OP_COUNT);

/// Guard against runaway enum growth — dispatch tables are stack-allocated in some paths
static_assert(OP_COUNT <= 1024, "OpId::OP_COUNT exceeds 1024; review sparse layout or increase limit");

/**
 * @brief Convert OpId to string for error messages and debugging.
 *
 * @param id Operation identifier
 * @return String representation of the operation
 */
auto op_id_to_name(OpId id) noexcept -> std::string_view;

/**
 * @brief Convert operation name string to OpId.
 *
 * Performs reverse lookup from the name table built by op_id_to_name.
 * Returns OpId::OP_COUNT if the name is not recognized.
 *
 * @param name Operation name (e.g. "matmul", "relu")
 * @return Corresponding OpId, or OpId::OP_COUNT if unknown
 */
auto string_to_op_id(std::string_view name) noexcept -> OpId;

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
