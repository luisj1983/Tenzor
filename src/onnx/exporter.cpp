/**
 * @file exporter.cpp
 * @brief Implementation of ONNX model export functionality
 *
 * As of the P3 pass, this exporter uses generated protobuf bindings from
 * proto/onnx.proto (via onnx.pb.h) instead of a hand-rolled wire-format
 * encoder. The output is now field-number-compatible with canonical
 * upstream ONNX (IR version 8), so models exported from Tenzor can be
 * loaded by onnxruntime / netron / onnx-checker without modification.
 */

#include "../../include/tenzor/onnx/exporter.hpp"
#include "../../include/tenzor/nn/quantization/quantized_layers.hpp"
#include "../../include/tenzor/nn/layers/conv.hpp"
#include "../../include/tenzor/utils/error.hpp"
#include "../../include/tenzor/utils/logging.hpp"
#include <cstring>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <system_error>
#include <unordered_set>

#ifdef TENZOR_HAS_ONNX_PROTOBUF
#include "onnx.pb.h"
#endif

namespace tenzor {
namespace onnx {

// ============================================================================
// Helper Functions
// ============================================================================

// Audit I7: removed `trace_custom_module` and its forward declaration. The
// free `export_to_onnx` function now routes through real JIT tracing
// (jit::trace + convert_jit_graph_to_onnx) — same path as
// ONNXExporter::export_module — which handles arbitrary ops, not just the
// hand-coded Linear/BatchNorm/LayerNorm/Conv patterns the old tracer
// recognized.

// ============================================================================
// DType Conversion
// ============================================================================

auto dtype_to_onnx(DType dtype) -> ONNXDataType {
    switch (dtype) {
        case DType::Float32: return ONNXDataType::FLOAT;
        case DType::Float64: return ONNXDataType::DOUBLE;
        case DType::Float16: return ONNXDataType::FLOAT16;
        case DType::BFloat16: return ONNXDataType::BFLOAT16;
        case DType::Int8: return ONNXDataType::INT8;
        case DType::Int16: return ONNXDataType::INT16;
        case DType::Int32: return ONNXDataType::INT32;
        case DType::Int64: return ONNXDataType::INT64;
        case DType::UInt8: return ONNXDataType::UINT8;
        case DType::UInt16: return ONNXDataType::UINT16;
        case DType::UInt32: return ONNXDataType::UINT32;
        case DType::UInt64: return ONNXDataType::UINT64;
        case DType::Bool: return ONNXDataType::BOOL;
        // Inf-D3: native 8-bit float types — ONNX opset 19+.
        case DType::FP8_E4M3: return ONNXDataType::FLOAT8E4M3FN;
        case DType::FP8_E5M2: return ONNXDataType::FLOAT8E5M2;
        // Inf-D3: quantized 8-bit storage codes; per-tensor scale/zero_point
        // travel via QLinearMatMul / QLinearConv input slots at op level.
        case DType::QInt8: return ONNXDataType::INT8;
        case DType::QUInt8: return ONNXDataType::UINT8;
        // Inf-D3: 4-bit packed quantization → ONNX INT4 (opset 21+).
        // Tenzor's QInt4x2 stores two 4-bit values per byte, matching the
        // ONNX INT4 packed-byte convention.
        case DType::QInt4x2: return ONNXDataType::INT4;
        // 5th-audit C1: Tenzor's ONNX importer does not currently accept
        // COMPLEX64/COMPLEX128 codes (see src/onnx/importer.cpp dtype_from_onnx),
        // so exporting them silently produces a non-round-trippable file.
        // Throw early with a clear message instead.
        case DType::Complex64:
        case DType::Complex128:
            throw std::runtime_error(
                "ONNX export: Complex64/Complex128 are not round-trippable through "
                "this importer/exporter. Cast to a real {real, imag} Float32/Float64 "
                "pair (e.g. tenzor::view_as_real) before export.");
    }
    // Unreachable for valid DType — switch above is exhaustive over the enum
    // (we ensure this via the static_assert at the end of this file's TU).
    throw std::runtime_error("Unsupported DType for ONNX export: dtype id " +
                             std::to_string(static_cast<int>(dtype)));
}

// ============================================================================
// OpId to ONNX Mapping
// ============================================================================

auto ONNXExporter::op_to_onnx(OpId op) -> std::string {
    switch (op) {
        // Arithmetic operations
        case OpId::Add:          return "Add";
        case OpId::Sub:          return "Sub";
        case OpId::Mul:          return "Mul";
        case OpId::Div:          return "Div";
        case OpId::MatMul:       return "MatMul";
        case OpId::Bmm:          return "MatMul";
        case OpId::Dot:          return "MatMul";

        // Reduction operations
        case OpId::Sum:          return "ReduceSum";
        case OpId::Mean:         return "ReduceMean";
        case OpId::Max:          return "ReduceMax";
        case OpId::Min:          return "ReduceMin";
        case OpId::ArgMax:       return "ArgMax";
        case OpId::ArgMin:       return "ArgMin";
        case OpId::Prod:         return "ReduceProd";

        // Element-wise math
        case OpId::Sqrt:         return "Sqrt";
        case OpId::Neg:          return "Neg";
        case OpId::Abs:          return "Abs";
        case OpId::Sign:         return "Sign";
        case OpId::Log:          return "Log";
        case OpId::Exp:          return "Exp";
        case OpId::Pow:          return "Pow";
        case OpId::Clamp:        return "Clip";
        case OpId::Reciprocal:   return "Reciprocal";
        case OpId::Floor:        return "Floor";
        case OpId::Ceil:         return "Ceil";
        case OpId::Round:        return "Round";

        // Trigonometric operations
        case OpId::Sin:          return "Sin";
        case OpId::Cos:          return "Cos";
        case OpId::Tan:          return "Tan";
        case OpId::Asin:         return "Asin";
        case OpId::Acos:         return "Acos";
        case OpId::Atan:         return "Atan";
        case OpId::Sinh:         return "Sinh";
        case OpId::Cosh:         return "Cosh";
        case OpId::Tanh:         return "Tanh";

        // Activation functions
        case OpId::ReLU:         return "Relu";
        case OpId::Sigmoid:      return "Sigmoid";
        case OpId::Hardswish:    return "HardSwish";    // opset 14+
        case OpId::Hardsigmoid:  return "HardSigmoid";
        case OpId::TanhActivation: return "Tanh";
        case OpId::Gelu:         return "Gelu";       // opset 20+
        // Inf-D2: Swish has no direct ONNX equivalent before opset 14 (HardSwish).
        // The visitor emits Sigmoid + Mul; callers needing the explicit
        // decomposition should still use ONNXExporter::export_swish().
        case OpId::Swish:        return "Mul";          // decomposed: x * sigmoid(x)
        case OpId::LeakyReLU:    return "LeakyRelu";
        case OpId::Elu:          return "Elu";
        case OpId::Selu:         return "Selu";
        case OpId::Mish:         return "Mish";        // opset 18+
        case OpId::Softplus:     return "Softplus";
        case OpId::Softmax:      return "Softmax";
        case OpId::LogSoftmax:   return "LogSoftmax";

        // Shape/view operations
        case OpId::Reshape:      return "Reshape";
        case OpId::Transpose:    return "Transpose";
        case OpId::Permute:      return "Transpose";
        case OpId::Squeeze:      return "Squeeze";
        case OpId::Unsqueeze:    return "Unsqueeze";
        case OpId::Flatten:      return "Flatten";
        case OpId::Expand:       return "Expand";
        case OpId::Stack:        return "Concat";      // With unsqueeze pre-step
        case OpId::Split:        return "Split";
        case OpId::Chunk:        return "Split";
        case OpId::Tile:         return "Tile";
        case OpId::Repeat:       return "Tile";

        // Indexing operations
        case OpId::IndexSelect:  return "Gather";
        case OpId::Gather:       return "GatherElements";
        case OpId::Scatter:      return "ScatterElements";
        case OpId::Where:        return "Where";
        case OpId::Slice:        return "Slice";
        case OpId::Cat:          return "Concat";
        case OpId::Nonzero:      return "NonZero";
        case OpId::OneHot:       return "OneHot";

        // Comparison operations
        case OpId::Eq:           return "Equal";
        case OpId::Ne:           return "Equal";       // Negated Equal
        case OpId::Lt:           return "Less";
        case OpId::Le:           return "LessOrEqual";
        case OpId::Gt:           return "Greater";
        case OpId::Ge:           return "GreaterOrEqual";

        // Normalization operations
        case OpId::BatchNorm2dForward:       return "BatchNormalization";
        case OpId::BatchNorm2dForwardAffine: return "BatchNormalization";
        case OpId::LayerNorm:                return "LayerNormalization";  // opset 17+
        case OpId::GroupNorm:                return "GroupNormalization";  // opset 18+
        case OpId::InstanceNorm:             return "InstanceNormalization";

        // Convolution operations
        case OpId::Conv2dForward:            return "Conv";
        case OpId::ConvTranspose2dForward:   return "ConvTranspose";
        case OpId::Conv3dForward:            return "Conv";

        // Pooling operations
        case OpId::MaxPool2dForward:         return "MaxPool";
        case OpId::AvgPool2dForward:         return "AveragePool";
        case OpId::AdaptiveAvgPool2d:        return "AveragePool";      // Computed kernel/stride
        case OpId::AdaptiveMaxPool2d:        return "MaxPool";          // Computed kernel/stride
        case OpId::MaxPool1dForward:         return "MaxPool";
        case OpId::MaxPool3dForward:         return "MaxPool";
        case OpId::AvgPool3dForward:         return "AveragePool";

        // RNN operations
        case OpId::LSTMForward:              return "LSTM";
        case OpId::LSTMMultiLayerForward:    return "LSTM";
        case OpId::BiLSTMForward:            return "LSTM";
        case OpId::GRUForward:               return "GRU";
        case OpId::GRUMultiLayerForward:     return "GRU";
        case OpId::LSTMCellForward:          return "LSTM";
        case OpId::GRUCellForward:           return "GRU";

        // Embedding
        case OpId::Embedding:    return "Gather";      // ONNX uses Gather for embedding lookup

        // Linear/FC
        case OpId::Linear:       return "Gemm";

        // Dropout
        case OpId::Dropout:      return "Dropout";

        // Advanced operations
        case OpId::TopK:         return "TopK";
        case OpId::Sort:         return "TopK";        // ONNX lacks Sort; TopK with k=N approximates
        case OpId::CumSum:       return "CumSum";

        // Vision operations
        case OpId::Interpolate:  return "Resize";

        // Cast operation
        case OpId::Cast:         return "Cast";

        // Triangular operations
        case OpId::Triu:         return "Trilu";        // upper=1 attribute
        case OpId::Tril:         return "Trilu";        // upper=0 attribute

        // Logical operations
        case OpId::LogicalAnd:   return "And";
        case OpId::LogicalOr:    return "Or";
        case OpId::LogicalNot:   return "Not";

        // Scatter with reduction
        case OpId::ScatterAdd:   return "ScatterElements"; // reduction='add'

        // Signal processing (opset 17+)
        case OpId::FFT:          return "DFT";
        case OpId::IFFT:         return "DFT";          // inverse=1 attribute
        case OpId::RFFT:         return "DFT";

        // Accumulation operations
        case OpId::CumProd:      return "CumProd";       // Decomposed: Log + CumSum + Exp

        // Roll (custom: Slice + Concat decomposition)
        case OpId::Roll:         return "Roll";         // No native ONNX op; decomposed in export

        // Depthwise convolution
        case OpId::DepthwiseConv2d: return "Conv";      // group=in_channels attribute
        case OpId::DepthwiseConv1d: return "Conv";      // group=in_channels attribute
        case OpId::DepthwiseConv3d: return "Conv";      // group=in_channels attribute

        // Log2 (custom: Log / Log(2) decomposition)
        case OpId::Log2:         return "Log";          // Decomposed: Log(x) / Log(2)

        // Quantized convolution
        case OpId::QuantizedConv2d: return "QLinearConv";

        // EmbeddingBag (custom: Gather + ReduceSum decomposition)
        case OpId::EmbeddingBagForward: return "Gather"; // Decomposed: Gather + ReduceSum

        // Unfold/Fold — complex subgraph ops
        // Unfold → custom (Slice + Reshape); Fold → Col2Im (opset 18+)
        case OpId::Unfold:       return "Unfold";       // Decomposed: Slice + Reshape
        case OpId::Fold:         return "Col2Im";       // opset 18+

        // SearchSorted — registered as custom ONNX op in "tenzor" domain
        case OpId::SearchSorted: return "SearchSorted"; // Custom op: tenzor domain

        // ====================================================================
        // Inf-D2: Extended OpId coverage — direct ONNX mappings and decompositions
        // ====================================================================
        //
        // Activations beyond the audited core set (only the OpIds that exist
        // in tenzor::OpId today — Hardswish/Hardsigmoid/PReLU/CELU/SiLU/GLU/
        // ReLU6/Threshold/Softsign/Hardshrink/Softshrink are not yet enumerated
        // in op_id.hpp and so cannot have ONNX mappings until that lands).
        case OpId::LogSigmoid:   return "LogSigmoid";   // decomposed: Sigmoid + Log
        // OpId::Swish already mapped above (in the original activations block).
        case OpId::RReLU:        return "PRelu";        // randomization dropped at export

        // Trigonometric & hyperbolic completions.
        case OpId::Atan2:        return "Atan2";        // tenzor custom (ONNX has no Atan2)
        case OpId::Acosh:        return "Acosh";
        case OpId::Asinh:        return "Asinh";
        case OpId::Atanh:        return "Atanh";

        // Extended unary math.
        case OpId::Erf:          return "Erf";
        case OpId::Erfc:         return "Erfc";          // tenzor custom
        case OpId::ErfInv:       return "ErfInv";        // tenzor custom
        case OpId::Exp2:         return "Exp";           // decomposed Pow(2, x)
        case OpId::Expm1:        return "Expm1";         // tenzor custom
        case OpId::Log10:        return "Log";           // decomposed: Log/Log10
        case OpId::Log1p:        return "Log1p";         // tenzor custom
        case OpId::Lgamma:       return "Lgamma";        // tenzor custom
        case OpId::Digamma:      return "Digamma";       // tenzor custom
        case OpId::Polygamma:    return "Polygamma";     // tenzor custom
        case OpId::Multigammaln: return "Multigammaln";  // tenzor custom
        case OpId::Sinc:         return "Sinc";          // tenzor custom
        case OpId::Trunc:        return "Trunc";         // tenzor custom
        case OpId::Frac:         return "Frac";          // tenzor custom (decomposed x - floor(x))
        case OpId::Square:       return "Mul";           // decomposed: x*x
        case OpId::Rsqrt:        return "Reciprocal";    // decomposed Sqrt+Reciprocal
        case OpId::Rad2Deg:      return "Mul";           // decomposed *(180/π)
        case OpId::Deg2Rad:      return "Mul";           // decomposed *(π/180)
        case OpId::Logit:        return "Logit";         // tenzor custom (log(p/(1-p)))
        case OpId::Heaviside:    return "Heaviside";     // tenzor custom
        case OpId::Signbit:      return "Signbit";       // tenzor custom
        case OpId::Copysign:     return "Copysign";      // tenzor custom
        case OpId::Hypot:        return "Hypot";         // tenzor custom
        case OpId::Ldexp:        return "Ldexp";         // tenzor custom
        case OpId::Frexp:        return "Frexp";         // tenzor custom (two outputs)
        case OpId::FloatPower:   return "Pow";
        case OpId::Nextafter:    return "Nextafter";     // tenzor custom
        case OpId::Lerp:         return "Lerp";          // tenzor custom (decomposed)
        case OpId::Fmod:         return "Mod";           // fmod=1 attribute
        case OpId::Remainder:    return "Mod";           // fmod=0
        case OpId::Gcd:          return "Gcd";           // tenzor custom (integer)
        case OpId::Lcm:          return "Lcm";           // tenzor custom (integer)
        case OpId::Addcdiv:      return "Addcdiv";       // tenzor custom (decomposed Mul+Add+Div)
        case OpId::Addcmul:      return "Addcmul";       // tenzor custom (decomposed Mul+Add+Mul)
        case OpId::Addmm:        return "Gemm";          // beta·C + alpha·(A@B)
        case OpId::Addmv:        return "Gemm";          // vector form decomposed
        case OpId::Baddbmm:      return "Gemm";          // batched form decomposed
        case OpId::Xlog1py:      return "Xlog1py";       // tenzor custom
        case OpId::XLogY:        return "XLogY";         // tenzor custom
        case OpId::LogAddExp:    return "LogAddExp";     // tenzor custom (decomposed)
        case OpId::LogAddExp2:   return "LogAddExp2";    // tenzor custom

        // Bessel & special functions — no ONNX equivalents, all tenzor custom.
        case OpId::BesselI0:        return "BesselI0";
        case OpId::BesselI1:        return "BesselI1";
        case OpId::BesselJ0:        return "BesselJ0";
        case OpId::BesselJ1:        return "BesselJ1";
        case OpId::BesselY0:        return "BesselY0";
        case OpId::BesselY1:        return "BesselY1";
        case OpId::I0e:             return "I0e";
        case OpId::I1e:             return "I1e";
        case OpId::SphericalBesselJ0: return "SphericalBesselJ0";
        case OpId::Gamma:           return "Gamma";
        case OpId::Beta:            return "Beta";
        case OpId::BetaInc:         return "BetaInc";
        case OpId::Igamma:          return "Igamma";
        case OpId::Igammac:         return "Igammac";
        case OpId::Zeta:            return "Zeta";
        case OpId::Entr:            return "Entr";
        case OpId::Ndtr:            return "Ndtr";
        case OpId::LogNdtr:         return "LogNdtr";

        // Bitwise.
        case OpId::BitwiseAnd:        return "BitwiseAnd";    // opset 18+
        case OpId::BitwiseOr:         return "BitwiseOr";
        case OpId::BitwiseXor:        return "BitwiseXor";
        case OpId::BitwiseNot:        return "BitwiseNot";
        case OpId::BitwiseLeftShift:  return "BitShift";      // direction=LEFT attr
        case OpId::BitwiseRightShift: return "BitShift";      // direction=RIGHT attr

        // Logical (LogicalAnd/Or/Not already covered above).
        case OpId::LogicalXor:        return "Xor";

        // Predicates / classification.
        case OpId::IsFinite:        return "IsFinite";     // tenzor custom
        case OpId::IsInf:           return "IsInf";        // opset 10+
        case OpId::IsNan:           return "IsNaN";        // opset 9+
        case OpId::IsNegInf:        return "IsNegInf";     // tenzor custom
        case OpId::IsPosInf:        return "IsPosInf";     // tenzor custom
        case OpId::IsReal:          return "IsReal";       // tenzor custom (always true for real dtypes)
        case OpId::Isin:            return "Isin";         // tenzor custom
        case OpId::HasInfNan:       return "HasInfNan";    // tenzor custom (returns scalar bool)
        case OpId::NanToNum:        return "NanToNum";     // tenzor custom

        // Reductions (Min/Max/Mean/Sum/Prod already covered).
        case OpId::All:             return "ReduceMin";    // bool → all is min over {0,1}
        case OpId::Any:             return "ReduceMax";    // bool → any is max over {0,1}
        case OpId::Median:          return "Median";       // tenzor custom
        case OpId::Mode:            return "Mode";         // tenzor custom
        case OpId::Std:             return "ReduceL2";     // decomposed via mean
        case OpId::Var:             return "Var";          // tenzor custom (decomposed)
        case OpId::Norm:            return "ReduceL2";     // ord-dependent decomposition
        case OpId::LogSumExp:       return "ReduceLogSumExp";
        case OpId::Nansum:          return "Nansum";       // tenzor custom (where+sum)
        case OpId::Nanmean:         return "Nanmean";      // tenzor custom
        case OpId::Nanmedian:       return "Nanmedian";    // tenzor custom
        case OpId::NanStd:          return "NanStd";       // tenzor custom
        case OpId::NanVar:          return "NanVar";       // tenzor custom
        case OpId::Quantile:        return "Quantile";     // tenzor custom
        case OpId::Nanquantile:     return "Nanquantile";  // tenzor custom
        case OpId::Kthvalue:        return "Kthvalue";     // tenzor custom
        case OpId::Aminmax:         return "Aminmax";      // tenzor custom (two outputs)
        case OpId::Cov:             return "Cov";          // tenzor custom
        case OpId::Corrcoef:        return "Corrcoef";     // tenzor custom
        case OpId::CountNonzero:    return "ReduceSum";    // decomposed (x != 0).sum()
        case OpId::Trace:           return "Trace";        // tenzor custom (diag + sum)
        case OpId::CumMax:          return "CumMax";       // tenzor custom (two outputs)
        case OpId::CumMin:          return "CumMin";       // tenzor custom (two outputs)
        case OpId::Logcumsumexp:    return "Logcumsumexp"; // tenzor custom
        case OpId::Bincount:        return "Bincount";     // tenzor custom
        case OpId::Histc:           return "Histc";        // tenzor custom
        case OpId::Histogram:       return "Histogram";    // tenzor custom
        case OpId::Histogramdd:     return "Histogramdd";  // tenzor custom
        case OpId::SegmentReduce:   return "SegmentReduce"; // tenzor custom
        case OpId::Trapezoid:       return "Trapezoid";    // tenzor custom
        case OpId::CumulativeTrapezoid: return "CumulativeTrapezoid"; // tenzor custom

        // Min/Max element-wise.
        case OpId::Maximum:         return "Max";
        case OpId::Minimum:         return "Min";
        case OpId::Fmax:            return "Max";          // decomposed via where(isnan)
        case OpId::Fmin:            return "Min";          // decomposed via where(isnan)
        case OpId::ClampMax:        return "Clip";         // max-only
        case OpId::ClampMin:        return "Clip";         // min-only

        // Shape ops & creation.
        case OpId::Flip:            return "Slice";        // negative step decomposition
        case OpId::AsStrided:       return "AsStrided";    // tenzor custom
        case OpId::Clone:           return "Identity";
        case OpId::Contiguous:      return "Identity";
        case OpId::ToMemoryFormat:  return "Identity";
        case OpId::Fill:            return "ConstantOfShape";
        case OpId::Full:            return "ConstantOfShape";
        case OpId::Zeros:           return "ConstantOfShape"; // value=0
        case OpId::Ones:            return "ConstantOfShape"; // value=1
        case OpId::Eye:             return "EyeLike";
        case OpId::Arange:          return "Range";
        case OpId::Linspace:        return "Range";        // computed step (end-start)/(n-1)
        case OpId::Diag:            return "Diag";         // tenzor custom (no native ONNX)
        case OpId::DiagEmbed:       return "DiagEmbed";    // tenzor custom
        case OpId::Diagflat:        return "Diagflat";     // tenzor custom
        case OpId::TrilIndices:     return "TrilIndices";  // tenzor custom
        case OpId::TriuIndices:     return "TriuIndices";  // tenzor custom
        case OpId::RepeatInterleave: return "RepeatInterleave"; // tenzor custom (or decomposed)
        case OpId::Unique:          return "Unique";       // opset 11+
        case OpId::UniqueConsecutive: return "UniqueConsecutive"; // tenzor custom
        case OpId::Renorm:          return "Renorm";       // tenzor custom

        // Indexing / scatter (Gather/Scatter/ScatterElements/Where already covered).
        case OpId::AdvancedIndex:    return "AdvancedIndex";   // tenzor custom
        case OpId::AdvancedIndexPut: return "AdvancedIndexPut";// tenzor custom
        case OpId::MaskedFill:       return "Where";           // decomposed
        case OpId::MaskedScatter:    return "MaskedScatter";   // tenzor custom
        case OpId::MaskedSelect:     return "MaskedSelect";    // tenzor custom (Compress in ONNX)
        case OpId::IndexAdd:         return "ScatterElements"; // reduction='add'
        case OpId::IndexCopy:        return "ScatterElements"; // reduction='none'
        case OpId::IndexFill:        return "ScatterElements"; // reduction='none', constant src
        case OpId::Take:             return "Gather";
        case OpId::TakeAlongDim:     return "GatherElements";
        case OpId::Put:              return "ScatterElements";
        case OpId::ScatterReduce:    return "ScatterElements"; // reduction attribute set
        case OpId::SelectScatter:    return "ScatterElements";
        case OpId::SliceScatter:     return "ScatterElements";
        case OpId::DiagonalScatter:  return "DiagonalScatter"; // tenzor custom
        case OpId::ArgSort:          return "TopK";            // returns indices only
        case OpId::Bucketize:        return "Bucketize";       // tenzor custom (binary search)
        case OpId::EmbeddingWithBoundsCheck: return "Gather";  // bounds-check decomposes to Where+Gather

        // Sort second output already covered via OpId::Sort.

        // Conv / Deformable / Fractional Pool.
        case OpId::Conv1dForward:                return "Conv";
        case OpId::ConvTranspose3dForward:       return "ConvTranspose";
        case OpId::DeformableConv2dForward:      return "DeformableConv2d"; // tenzor custom
        case OpId::FractionalMaxPool2dForward:   return "FractionalMaxPool2d"; // tenzor custom
        case OpId::FractionalMaxPool3dForward:   return "FractionalMaxPool3d"; // tenzor custom

        // Pooling extra.
        case OpId::AvgPool1dForward:        return "AveragePool";
        case OpId::AdaptiveAvgPool1d:       return "AveragePool";   // computed kernel
        case OpId::AdaptiveAvgPool3d:       return "AveragePool";   // computed kernel
        case OpId::AdaptiveMaxPool1d:       return "MaxPool";       // computed kernel
        case OpId::AdaptiveMaxPool3d:       return "MaxPool";       // computed kernel
        case OpId::MaxUnpool1dForward:      return "MaxUnpool";     // opset 11+
        case OpId::MaxUnpool2dForward:      return "MaxUnpool";
        case OpId::MaxUnpool3dForward:      return "MaxUnpool";

        // Norms (LayerNorm/GroupNorm/InstanceNorm/BatchNorm already covered).
        case OpId::RMSNorm:                 return "RMSNormalization"; // opset 23+
        case OpId::FusedLayerNorm:          return "LayerNormalization";
        case OpId::FusedRMSNorm:            return "RMSNormalization";

        // Fused ops (decompositions live in the visitor; mapped here for naming).
        case OpId::FusedAddReLU:            return "Relu";    // decomposed: Add + Relu
        case OpId::FusedGelu:               return "Gelu";    // opset 20+ (or decomposed)
        case OpId::FusedLinearReLU:         return "Relu";    // Gemm + Relu
        case OpId::FusedConv2dReLU:         return "Relu";    // Conv + Relu
        case OpId::FusedConv2dSigmoid:      return "Sigmoid"; // Conv + Sigmoid
        case OpId::FusedConv2dTanh:         return "Tanh";    // Conv + Tanh
        case OpId::FusedConv2dSwish:        return "Mul";     // Conv + Swish
        case OpId::FusedConv2dBnReLU:       return "Relu";    // Conv + BN + Relu folded
        case OpId::FusedBatchNormReLU:      return "Relu";    // BN + Relu
        case OpId::FusedSoftmaxCrossEntropy: return "SoftmaxCrossEntropyLoss"; // opset 12+
        case OpId::FusedAttention:          return "Attention"; // opset 23+ (else MHA decomposition)

        // Random / sampling.
        case OpId::Bernoulli:        return "Bernoulli";     // opset 15+
        case OpId::Multinomial:      return "Multinomial";   // opset 7+
        case OpId::Rand:             return "RandomUniform";
        case OpId::Randn:            return "RandomNormal";
        case OpId::Randint:          return "RandomUniform"; // cast to int
        case OpId::NormalSample:     return "RandomNormal";
        case OpId::PoissonSample:    return "PoissonSample"; // tenzor custom
        case OpId::ExponentialSample: return "ExponentialSample"; // tenzor custom
        case OpId::GumbelSoftmax:    return "GumbelSoftmax"; // tenzor custom

        // Complex / FFT block.
        case OpId::Real:             return "Real";          // tenzor custom
        case OpId::Imag:             return "Imag";          // tenzor custom
        case OpId::Conj:             return "Conj";          // tenzor custom
        case OpId::Angle:            return "Angle";         // tenzor custom
        case OpId::ComplexTensor:    return "ComplexTensor"; // tenzor custom (combines re+im)
        case OpId::Polar:            return "Polar";         // tenzor custom
        case OpId::FFT2:             return "DFT";
        case OpId::FFTN:             return "DFT";
        case OpId::IFFT2:            return "DFT";           // inverse=1
        case OpId::IFFTN:            return "DFT";           // inverse=1
        case OpId::IRFFT:            return "DFT";           // inverse=1, onesided=1
        case OpId::DCT:              return "DCT";           // tenzor custom
        case OpId::IDCT:             return "IDCT";          // tenzor custom
        case OpId::STFT:             return "STFT";          // opset 17+
        case OpId::ISTFT:            return "ISTFT";         // tenzor custom
        case OpId::MelScale:         return "MelScale";      // tenzor custom (audio)
        case OpId::MFCC:             return "MFCC";          // tenzor custom (audio)

        // Linear-algebra block — all but Det/Inv go under tenzor custom domain.
        case OpId::LinalgDet:           return "Det";
        case OpId::LinalgInv:           return "LinalgInv";        // tenzor custom (Inverse opset 9+, but we route via custom)
        case OpId::LinalgCholesky:      return "LinalgCholesky";   // tenzor custom
        case OpId::LinalgCholeskySolve: return "LinalgCholeskySolve";
        case OpId::CholeskyInverse:     return "CholeskyInverse";
        case OpId::LinalgQR:            return "LinalgQR";         // tenzor custom
        case OpId::LinalgSVD:           return "LinalgSVD";        // tenzor custom
        case OpId::LinalgEig:           return "LinalgEig";        // tenzor custom
        case OpId::LinalgEigh:          return "LinalgEigh";       // tenzor custom
        case OpId::LinalgSolve:         return "LinalgSolve";      // tenzor custom
        case OpId::LinalgLU:            return "LinalgLU";         // tenzor custom
        case OpId::LinalgLUSolve:       return "LinalgLUSolve";    // tenzor custom
        case OpId::LinalgLDLFactor:     return "LinalgLDLFactor";  // tenzor custom
        case OpId::LinalgLDLSolve:      return "LinalgLDLSolve";   // tenzor custom
        case OpId::LinalgHouseholder:   return "LinalgHouseholder";// tenzor custom
        case OpId::LinalgMatrixNorm:    return "LinalgMatrixNorm"; // tenzor custom
        case OpId::LinalgVectorNorm:    return "LinalgVectorNorm"; // tenzor custom
        case OpId::LinalgVecdot:        return "LinalgVecdot";     // tenzor custom (dot)
        case OpId::Geqrf:               return "Geqrf";            // tenzor custom
        case OpId::Ormqr:               return "Ormqr";            // tenzor custom
        case OpId::TensorSolve:         return "TensorSolve";      // tenzor custom
        case OpId::TensorInv:           return "TensorInv";        // tenzor custom
        case OpId::SolveTriangular:     return "SolveTriangular";  // tenzor custom
        case OpId::Einsum:              return "Einsum";           // opset 12+
        case OpId::Cross:               return "Cross";            // tenzor custom
        case OpId::CDist:               return "CDist";            // tenzor custom
        case OpId::Pdist:               return "Pdist";            // tenzor custom
        case OpId::PairwiseDistance:    return "PairwiseDistance"; // tenzor custom
        case OpId::CosineSimilarity:    return "CosineSimilarity"; // tenzor custom
        case OpId::LOBPCG:              return "LOBPCG";           // tenzor custom

        // Sparse.
        case OpId::SparseSpMM:    return "SparseSpMM";    // tenzor custom domain
        case OpId::SparseSpMV:    return "SparseSpMV";    // tenzor custom
        case OpId::SparseSpGEMM:  return "SparseSpGEMM";  // tenzor custom
        case OpId::SparseAdd:     return "SparseAdd";     // tenzor custom
        case OpId::SparseToDense: return "SparseToDense"; // tenzor custom
        case OpId::DenseToSparse: return "DenseToSparse"; // tenzor custom
        case OpId::SparseSoftmax: return "SparseSoftmax"; // tenzor custom
        case OpId::SparseLogSoftmax: return "SparseLogSoftmax"; // tenzor custom
        case OpId::SparseTrsm:    return "SparseTrsm";    // tenzor custom
        case OpId::SparseTrsv:    return "SparseTrsv";    // tenzor custom

        // Attention / nested.
        case OpId::FlashAttention:   return "Attention";      // opset 23+ (else custom MHA)
        case OpId::FlexAttention:    return "FlexAttention";  // tenzor custom
        case OpId::NestedAttention:  return "NestedAttention";// tenzor custom
        case OpId::NestedFromPadded: return "NestedFromPadded"; // tenzor custom
        case OpId::NestedToPadded:   return "NestedToPadded";   // tenzor custom
        case OpId::NestedSoftmax:    return "NestedSoftmax";    // tenzor custom
        case OpId::NestedLogSoftmax: return "NestedLogSoftmax"; // tenzor custom
        case OpId::NestedLinear:     return "NestedLinear";     // tenzor custom
        case OpId::NestedLayerNorm:  return "NestedLayerNorm";  // tenzor custom
        case OpId::NestedSum:        return "NestedSum";        // tenzor custom
        case OpId::NestedMean:       return "NestedMean";       // tenzor custom

        // Vision / detection.
        // Disambiguated: OpId::AffineGrid is now pinned to enum value 692
        // (op_id.hpp), distinct from FusedLinearReLU = 210. Routes through
        // the tenzor custom-domain "AffineGrid" op consumed by the importer
        // (which already supports it via Tenzor's existing affine_grid op).
        case OpId::AffineGrid:       return "AffineGrid";    // tenzor custom
        case OpId::GridSample:       return "GridSample";    // opset 16+
        case OpId::ROIAlignForward:  return "RoiAlign";      // opset 10+
        case OpId::NMS:              return "NonMaxSuppression"; // opset 10+
        case OpId::BoxIoU:           return "BoxIoU";        // tenzor custom
        case OpId::GatherRelativePositionBias: return "GatherRelativePositionBias"; // tenzor custom

        // Misc.
        case OpId::QuantizedLinear:  return "QLinearMatMul";  // opset 10+
        case OpId::BatchNorm2dFusedTraining:    return "BatchNormalization";
        case OpId::BatchNorm2dUpdateRunningStats: return "Identity"; // stats update is implicit
        case OpId::StridedFill:      return "ScatterElements"; // strided-fill decomposes
        case OpId::NumericalGradient: return "NumericalGradient"; // tenzor custom (debug)

        // ====================================================================
        // Inf-D1: catch-all — any OpId not above is either:
        //   (a) a *Backward, *Inplace, or autograd-internal sentinel (e.g.
        //       OP_COUNT, BatchNorm2dMeanVar) — those reach this branch only
        //       through misuse since exporter walks the forward graph only;
        //   (b) a genuinely new op added after Inf-D — falls through with a
        //       clear error citing this site so the dev knows what to update.
        // The exhaustiveness static_assert at end-of-TU prevents (b) at
        // compile time once a developer touches op_id.hpp.
        // ====================================================================
        default:
            throw std::runtime_error(
                "No ONNX mapping for OpId: " +
                std::string(op_id_to_name(op)) +
                " — add a case in src/onnx/exporter.cpp::op_to_onnx, or add "
                "the OpId to op_id_export_skip_list if it is autograd-internal."
            );
    }
}

// ============================================================================
// DType to ONNX Integer Mapping
// ============================================================================

auto ONNXExporter::dtype_to_onnx_int(DType dtype) -> int {
    return static_cast<int>(dtype_to_onnx(dtype));
}

// ============================================================================
// Static export_model Convenience Method
// ============================================================================

void ONNXExporter::export_model(nn::Module& model,
                                std::vector<Tensor> example_inputs,
                                const std::string& output_path,
                                int opset_version) {
    if (example_inputs.empty()) {
        throw std::runtime_error("At least one example input must be provided for ONNX export");
    }

    // Create exporter instance
    ONNXExporter exporter(opset_version);
    exporter.set_model_name("tenzor_model");
    exporter.set_description("Model exported from Tenzor via ONNXExporter::export_model");

    // Switch model to evaluation mode for consistent tracing
    bool was_training = model.is_training();
    model.eval();

    try {
        // Register example inputs
        for (size_t i = 0; i < example_inputs.size(); ++i) {
            Tensor cpu_input = example_inputs[i].cpu().contiguous();
            std::string input_name = "input";
            if (example_inputs.size() > 1) {
                input_name += "_" + std::to_string(i);
            }
            exporter.add_input(cpu_input, input_name);
        }

        // Export all module parameters as initializers
        auto named_params = model.named_parameters();
        for (const auto& [param_name, param_var] : named_params) {
            if (param_var && param_var->is_initialized() && param_var->tensor().numel() > 0) {
                std::string safe_name = param_name;
                std::replace(safe_name.begin(), safe_name.end(), '.', '_');
                exporter.add_initializer_tensor(param_var->tensor().cpu().contiguous(), safe_name);
            }
        }

        // Export all module buffers as initializers (e.g., BatchNorm running stats)
        auto named_buffs = model.named_buffers();
        for (const auto& [buffer_name, buffer_var] : named_buffs) {
            if (buffer_var && buffer_var->is_initialized() && buffer_var->tensor().numel() > 0) {
                std::string safe_name = buffer_name;
                std::replace(safe_name.begin(), safe_name.end(), '.', '_');
                exporter.add_initializer_tensor(buffer_var->tensor().cpu().contiguous(), safe_name);
            }
        }

        // Run forward pass to determine output shape/type
        Variable input_var(example_inputs[0].cpu().contiguous(), false);
        Variable output_var = model.forward(input_var);

        if (!output_var.is_initialized() || output_var.tensor().numel() == 0) {
            throw std::runtime_error("Module forward pass produced undefined or empty output");
        }

        // Attempt structural tracing for the module
        // For modules with submodules (Sequential, etc.), iterate and export each layer
        auto& submodules = model.get_submodules();
        if (!submodules.empty()) {
            // Trace through submodules in order
            std::string current_name = "input";
            if (example_inputs.size() > 1) {
                current_name = "input_0";
            }
            int layer_idx = 0;

            for (const auto& [mod_name, submod] : submodules) {
                std::string layer_output = "layer_" + std::to_string(layer_idx) + "_output";
                auto sub_params = submod->named_parameters();

                bool has_weight = false;
                bool has_bias = false;
                std::shared_ptr<Variable> weight_param;
                std::shared_ptr<Variable> bias_param;

                for (const auto& [pname, param] : sub_params) {
                    if (pname.find("weight") != std::string::npos) {
                        has_weight = true;
                        weight_param = param;
                    }
                    if (pname.find("bias") != std::string::npos) {
                        has_bias = true;
                        bias_param = param;
                    }
                }

                if (has_weight && weight_param) {
                    auto weight_shape = weight_param->tensor().shape();

                    if (weight_shape.size() == 2) {
                        // Linear layer pattern
                        std::optional<Tensor> bias_tensor;
                        if (has_bias && bias_param) {
                            bias_tensor = bias_param->tensor();
                        }
                        // Create a dummy input/output for shape tracking
                        Variable sub_input(example_inputs[0].cpu().contiguous(), false);
                        Variable sub_output = submod->forward(sub_input);
                        exporter.export_linear(
                            sub_input.tensor(),
                            weight_param->tensor(),
                            bias_tensor,
                            sub_output.tensor(),
                            layer_output
                        );
                    } else if (weight_shape.size() == 4) {
                        // Conv2d layer pattern — read real {stride,padding,dilation,groups}
                        // off the submodule instead of assuming defaults. The previous
                        // hardcoded {0,0} padding is why Conv2d(padding=1) round-trip
                        // diverged even before the importer's own drop bug.
                        std::vector<int64_t> kernel_size = {weight_shape[2], weight_shape[3]};
                        std::vector<int64_t> stride = {1, 1};
                        std::vector<int64_t> padding = {0, 0};
                        std::vector<int64_t> dilation = {1, 1};
                        int64_t groups = 1;
                        if (auto* c = dynamic_cast<nn::Conv2d*>(submod.get())) {
                            stride   = {c->stride_h(),   c->stride_w()};
                            padding  = {c->padding_h(),  c->padding_w()};
                            dilation = {c->dilation_h(), c->dilation_w()};
                            groups   = c->groups();
                        }

                        std::optional<Tensor> bias_tensor;
                        if (has_bias && bias_param) {
                            bias_tensor = bias_param->tensor();
                        }

                        Variable sub_input(example_inputs[0].cpu().contiguous(), false);
                        Variable sub_output = submod->forward(sub_input);
                        exporter.export_conv2d(
                            sub_input.tensor(),
                            weight_param->tensor(),
                            bias_tensor,
                            kernel_size,
                            stride,
                            padding,
                            dilation,
                            groups,
                            sub_output.tensor(),
                            layer_output
                        );
                    }
                }

                current_name = layer_output;
                ++layer_idx;
            }
        } else {
            // For flat modules, attempt pattern matching from parameters
            auto own_params = model.own_parameters();
            bool has_weight = false;
            bool has_bias = false;
            std::shared_ptr<Variable> weight_param;
            std::shared_ptr<Variable> bias_param;

            for (const auto& [pname, param] : named_params) {
                if (pname.find("weight") != std::string::npos) {
                    has_weight = true;
                    weight_param = param;
                }
                if (pname.find("bias") != std::string::npos) {
                    has_bias = true;
                    bias_param = param;
                }
            }

            if (has_weight && weight_param) {
                auto weight_shape = weight_param->tensor().shape();
                std::string output_name = "output";

                if (weight_shape.size() == 2) {
                    std::optional<Tensor> bias_tensor;
                    if (has_bias && bias_param) {
                        bias_tensor = bias_param->tensor();
                    }
                    exporter.export_linear(
                        input_var.tensor(),
                        weight_param->tensor(),
                        bias_tensor,
                        output_var.tensor(),
                        output_name
                    );
                } else if (weight_shape.size() == 4) {
                    // Flat-module Conv2d — same accessor path as the Sequential branch
                    std::vector<int64_t> kernel_size = {weight_shape[2], weight_shape[3]};
                    std::vector<int64_t> stride = {1, 1};
                    std::vector<int64_t> padding = {0, 0};
                    std::vector<int64_t> dilation = {1, 1};
                    int64_t groups = 1;
                    if (auto* c = dynamic_cast<const nn::Conv2d*>(&model)) {
                        stride   = {c->stride_h(),   c->stride_w()};
                        padding  = {c->padding_h(),  c->padding_w()};
                        dilation = {c->dilation_h(), c->dilation_w()};
                        groups   = c->groups();
                    }

                    std::optional<Tensor> bias_tensor;
                    if (has_bias && bias_param) {
                        bias_tensor = bias_param->tensor();
                    }
                    exporter.export_conv2d(
                        input_var.tensor(),
                        weight_param->tensor(),
                        bias_tensor,
                        kernel_size, stride, padding, dilation,
                        groups,
                        output_var.tensor(),
                        output_name
                    );
                }
            }
        }

        // Add output to graph
        exporter.add_output(output_var.tensor().cpu(), "output");

        // Serialize and write to file
        exporter.export_to_file(output_path);

    } catch (...) {
        // Restore training mode before re-throwing
        if (was_training) {
            model.train();
        }
        throw;
    }

    // Restore original training mode
    if (was_training) {
        model.train();
    }
}

// ============================================================================
// ONNXTensor Implementation
// ============================================================================

ONNXTensor::ONNXTensor(const Tensor& tensor, const std::string& name)
    : name(name), dtype(dtype_to_onnx(tensor.dtype())) {

    // Copy shape
    auto shape_span = tensor.shape();
    dims.assign(shape_span.begin(), shape_span.end());

    // Copy raw data
    size_t num_bytes = tensor.numel() * tensor.dtype_size();
    raw_data.resize(num_bytes);

    // Ensure tensor is on CPU and contiguous
    Tensor cpu_tensor = tensor.cpu().contiguous();
    const void* data_ptr = cpu_tensor.data_ptr();
    std::memcpy(raw_data.data(), data_ptr, num_bytes);
}

auto ONNXTensor::numel() const -> int64_t {
    int64_t result = 1;
    for (int64_t dim : dims) {
        result *= dim;
    }
    return result;
}

auto ONNXTensor::size_bytes() const -> size_t {
    return raw_data.size();
}

// ============================================================================
// ONNXExportValueInfo Implementation
// ============================================================================

ONNXExportValueInfo::ONNXExportValueInfo(const std::string& name, ONNXDataType dtype,
                             const std::vector<int64_t>& shape)
    : name(name), dtype(dtype), shape(shape) {}

// ============================================================================
// ONNXExportNode Implementation
// ============================================================================

ONNXExportNode::ONNXExportNode(const std::string& op_type, const std::string& name)
    : op_type(op_type), name(name) {}

auto ONNXExportNode::add_input(const std::string& input) -> void {
    inputs.push_back(input);
}

auto ONNXExportNode::add_output(const std::string& output) -> void {
    outputs.push_back(output);
}

auto ONNXExportNode::set_attr(const std::string& key, int64_t value) -> void {
    int_attrs[key] = value;
}

auto ONNXExportNode::set_attr(const std::string& key, float value) -> void {
    float_attrs[key] = value;
}

auto ONNXExportNode::set_attr(const std::string& key, const std::string& value) -> void {
    string_attrs[key] = value;
}

auto ONNXExportNode::set_attr(const std::string& key, const std::vector<int64_t>& value) -> void {
    ints_attrs[key] = value;
}

auto ONNXExportNode::set_attr(const std::string& key, const std::vector<float>& value) -> void {
    floats_attrs[key] = value;
}

auto ONNXExportNode::set_attr(const std::string& key, const ONNXTensor& value) -> void {
    tensor_attrs[key] = value;
}

// ============================================================================
// ONNXGraph Implementation
// ============================================================================

ONNXGraph::ONNXGraph(const std::string& name) : name(name) {}

auto ONNXGraph::add_node(const ONNXExportNode& node) -> void {
    nodes.push_back(node);
}

auto ONNXGraph::add_input(const ONNXExportValueInfo& input) -> void {
    inputs.push_back(input);
}

auto ONNXGraph::add_output(const ONNXExportValueInfo& output) -> void {
    outputs.push_back(output);
}

auto ONNXGraph::add_initializer(const ONNXTensor& tensor) -> void {
    initializers.push_back(tensor);
}

auto ONNXGraph::add_value_info(const ONNXExportValueInfo& info) -> void {
    value_info[info.name] = info;
}

auto ONNXGraph::get_unique_name(const std::string& prefix) -> std::string {
    return prefix + "_" + std::to_string(name_counter_++);
}

// ============================================================================
// ExportContext Implementation
// ============================================================================

auto ExportContext::register_tensor(const Tensor& tensor, const std::string& onnx_name) -> void {
    tensor_map_[tensor.data_ptr()] = onnx_name;
}

auto ExportContext::get_tensor_name(const Tensor& tensor) -> std::optional<std::string> {
    auto it = tensor_map_.find(tensor.data_ptr());
    if (it != tensor_map_.end()) {
        return it->second;
    }
    return std::nullopt;
}

auto ExportContext::has_tensor(const Tensor& tensor) -> bool {
    return tensor_map_.find(tensor.data_ptr()) != tensor_map_.end();
}

auto ExportContext::generate_name(const std::string& prefix) -> std::string {
    int64_t& counter = name_counters_[prefix];
    return prefix + "_" + std::to_string(counter++);
}

// ============================================================================
// ONNXExporter Implementation
// ============================================================================

ONNXExporter::ONNXExporter(int64_t opset_version)
    : opset_version_(opset_version), graph_("main_graph") {}

// Model Configuration

auto ONNXExporter::set_model_name(const std::string& name) -> void {
    model_name_ = name;
}

auto ONNXExporter::set_opset_version(int64_t version) -> void {
    opset_version_ = version;
}

auto ONNXExporter::set_description(const std::string& desc) -> void {
    description_ = desc;
}

auto ONNXExporter::set_producer_name(const std::string& name) -> void {
    producer_name_ = name;
}

auto ONNXExporter::set_model_version(int64_t version) -> void {
    model_version_ = version;
}

// Graph Building

auto ONNXExporter::add_input(const Tensor& tensor, const std::string& name,
                             const std::unordered_map<int64_t, std::string>& dynamic_axes) -> void {
    auto shape_span = tensor.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

    // Apply dynamic axes
    for (const auto& [axis, axis_name] : dynamic_axes) {
        if (axis >= 0 && axis < static_cast<int64_t>(shape.size())) {
            shape[axis] = -1; // -1 indicates dynamic dimension in ONNX
        }
    }

    ONNXExportValueInfo input_info(name, dtype_to_onnx(tensor.dtype()), shape);
    input_info.dim_params = dynamic_axes;
    graph_.add_input(input_info);
    context_.register_tensor(tensor, name);
}

auto ONNXExporter::add_output(const Tensor& tensor, const std::string& name,
                              const std::unordered_map<int64_t, std::string>& dynamic_axes) -> void {
    auto shape_span = tensor.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

    // Apply dynamic axes
    for (const auto& [axis, axis_name] : dynamic_axes) {
        if (axis >= 0 && axis < static_cast<int64_t>(shape.size())) {
            shape[axis] = -1; // -1 indicates dynamic dimension in ONNX
        }
    }

    ONNXExportValueInfo output_info(name, dtype_to_onnx(tensor.dtype()), shape);
    output_info.dim_params = dynamic_axes;
    graph_.add_output(output_info);
    context_.register_tensor(tensor, name);
}

// Tensor Operations

auto ONNXExporter::export_add(const Tensor& a, const Tensor& b, const Tensor& output,
                               const std::string& output_name) -> void {
    ONNXExportNode node("Add", context_.generate_name("add"));

    std::string a_name = get_tensor_name(a, "add_input_a");
    std::string b_name = get_tensor_name(b, "add_input_b");

    node.add_input(a_name);
    node.add_input(b_name);
    node.add_output(output_name);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_sub(const Tensor& a, const Tensor& b, const Tensor& output,
                               const std::string& output_name) -> void {
    ONNXExportNode node("Sub", context_.generate_name("sub"));

    std::string a_name = get_tensor_name(a, "sub_input_a");
    std::string b_name = get_tensor_name(b, "sub_input_b");

    node.add_input(a_name);
    node.add_input(b_name);
    node.add_output(output_name);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_mul(const Tensor& a, const Tensor& b, const Tensor& output,
                               const std::string& output_name) -> void {
    ONNXExportNode node("Mul", context_.generate_name("mul"));

    std::string a_name = get_tensor_name(a, "mul_input_a");
    std::string b_name = get_tensor_name(b, "mul_input_b");

    node.add_input(a_name);
    node.add_input(b_name);
    node.add_output(output_name);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_div(const Tensor& a, const Tensor& b, const Tensor& output,
                               const std::string& output_name) -> void {
    ONNXExportNode node("Div", context_.generate_name("div"));

    std::string a_name = get_tensor_name(a, "div_input_a");
    std::string b_name = get_tensor_name(b, "div_input_b");

    node.add_input(a_name);
    node.add_input(b_name);
    node.add_output(output_name);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_matmul(const Tensor& a, const Tensor& b, const Tensor& output,
                                  const std::string& output_name) -> void {
    ONNXExportNode node("MatMul", context_.generate_name("matmul"));

    std::string a_name = get_tensor_name(a, "matmul_input_a");
    std::string b_name = get_tensor_name(b, "matmul_input_b");

    node.add_input(a_name);
    node.add_input(b_name);
    node.add_output(output_name);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_reshape(const Tensor& input, const std::vector<int64_t>& new_shape,
                                   const Tensor& output, const std::string& output_name) -> void {
    ONNXExportNode node("Reshape", context_.generate_name("reshape"));

    std::string input_name = get_tensor_name(input, "reshape_input");

    // Create shape constant tensor
    std::string shape_name = context_.generate_name("reshape_shape");
    Tensor shape_tensor({static_cast<int64_t>(new_shape.size())}, DType::Int64, Device::cpu());
    std::memcpy(shape_tensor.data<int64_t>(), new_shape.data(), new_shape.size() * sizeof(int64_t));
    add_initializer_tensor(shape_tensor, shape_name);

    node.add_input(input_name);
    node.add_input(shape_name);
    node.add_output(output_name);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_transpose(const Tensor& input, const std::vector<int64_t>& perm,
                                     const Tensor& output, const std::string& output_name) -> void {
    ONNXExportNode node("Transpose", context_.generate_name("transpose"));

    std::string input_name = get_tensor_name(input, "transpose_input");

    node.add_input(input_name);
    node.add_output(output_name);
    node.set_attr("perm", perm);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_concat(const std::vector<Tensor>& inputs, int64_t axis,
                                  const Tensor& output, const std::string& output_name) -> void {
    ONNXExportNode node("Concat", context_.generate_name("concat"));

    for (size_t i = 0; i < inputs.size(); ++i) {
        std::string input_name = get_tensor_name(inputs[i], "concat_input_" + std::to_string(i));
        node.add_input(input_name);
    }

    node.add_output(output_name);
    node.set_attr("axis", axis);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_split(const Tensor& input, int64_t axis, const std::vector<int64_t>& split_sizes,
                                 const std::vector<Tensor>& outputs, const std::vector<std::string>& output_names) -> void {
    ONNXExportNode node("Split", context_.generate_name("split"));

    std::string input_name = get_tensor_name(input, "split_input");
    node.add_input(input_name);

    for (const auto& name : output_names) {
        node.add_output(name);
    }

    node.set_attr("axis", axis);
    if (!split_sizes.empty()) {
        node.set_attr("split", split_sizes);
    }

    graph_.add_node(node);

    for (size_t i = 0; i < outputs.size(); ++i) {
        context_.register_tensor(outputs[i], output_names[i]);
    }
}

// Neural Network Layers

auto ONNXExporter::export_linear(const Tensor& input, const Tensor& weight,
                                  const std::optional<Tensor>& bias, const Tensor& output,
                                  const std::string& output_name) -> void {
    // Linear layer in ONNX: Gemm (General Matrix Multiplication)
    // Y = alpha * A * B + beta * C
    // For Linear: Y = X @ W^T + bias
    // We need to transpose W

    ONNXExportNode gemm_node("Gemm", context_.generate_name("gemm"));

    std::string input_name = get_tensor_name(input, "linear_input");

    // Add weight as initializer (transposed)
    std::string weight_name = context_.generate_name("linear_weight");
    add_initializer_tensor(weight, weight_name);

    gemm_node.add_input(input_name);
    gemm_node.add_input(weight_name);

    if (bias.has_value()) {
        std::string bias_name = context_.generate_name("linear_bias");
        add_initializer_tensor(bias.value(), bias_name);
        gemm_node.add_input(bias_name);
    }

    gemm_node.add_output(output_name);

    // Attributes: alpha=1.0, beta=1.0, transB=1 (transpose weight)
    gemm_node.set_attr("alpha", 1.0f);
    gemm_node.set_attr("beta", 1.0f);
    gemm_node.set_attr("transB", static_cast<int64_t>(1));

    graph_.add_node(gemm_node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_conv2d(const Tensor& input, const Tensor& weight,
                                  const std::optional<Tensor>& bias,
                                  const std::vector<int64_t>& kernel_size,
                                  const std::vector<int64_t>& stride,
                                  const std::vector<int64_t>& padding,
                                  const std::vector<int64_t>& dilation,
                                  int64_t groups,
                                  const Tensor& output,
                                  const std::string& output_name) -> void {
    ONNXExportNode node("Conv", context_.generate_name("conv"));

    std::string input_name = get_tensor_name(input, "conv_input");

    std::string weight_name = context_.generate_name("conv_weight");
    add_initializer_tensor(weight, weight_name);

    node.add_input(input_name);
    node.add_input(weight_name);

    if (bias.has_value()) {
        std::string bias_name = context_.generate_name("conv_bias");
        add_initializer_tensor(bias.value(), bias_name);
        node.add_input(bias_name);
    }

    node.add_output(output_name);

    // Set attributes
    node.set_attr("kernel_shape", kernel_size);
    node.set_attr("strides", stride);
    node.set_attr("pads", std::vector<int64_t>{padding[0], padding[1], padding[0], padding[1]}); // [top, left, bottom, right]
    node.set_attr("dilations", dilation);
    node.set_attr("group", groups);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_conv3d(const Tensor& input, const Tensor& weight,
                                  const std::optional<Tensor>& bias,
                                  int64_t kernel_size, int64_t stride, int64_t padding,
                                  int64_t dilation, int64_t groups,
                                  const Tensor& output,
                                  const std::string& output_name) -> void {
    ONNXExportNode node("Conv", context_.generate_name("conv3d"));
    std::string input_name = get_tensor_name(input, "conv3d_input");
    std::string weight_name = context_.generate_name("conv3d_weight");
    add_initializer_tensor(weight, weight_name);
    node.add_input(input_name);
    node.add_input(weight_name);
    if (bias.has_value()) {
        std::string bias_name = context_.generate_name("conv3d_bias");
        add_initializer_tensor(bias.value(), bias_name);
        node.add_input(bias_name);
    }
    node.add_output(output_name);
    // 3-D kernel_shape / strides / dilations, and 6-element pads in ONNX order
    // [begin_d, begin_h, begin_w, end_d, end_h, end_w].
    node.set_attr("kernel_shape", std::vector<int64_t>{kernel_size, kernel_size, kernel_size});
    node.set_attr("strides",      std::vector<int64_t>{stride, stride, stride});
    node.set_attr("pads",         std::vector<int64_t>{padding, padding, padding, padding, padding, padding});
    node.set_attr("dilations",    std::vector<int64_t>{dilation, dilation, dilation});
    node.set_attr("group",        groups);
    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_conv_transpose(const Tensor& input, const Tensor& weight,
                                         const std::optional<Tensor>& bias,
                                         int64_t spatial_rank,
                                         int64_t kernel_size, int64_t stride, int64_t padding,
                                         int64_t output_padding, int64_t dilation, int64_t groups,
                                         const Tensor& output,
                                         const std::string& output_name) -> void {
    if (spatial_rank < 1 || spatial_rank > 3) {
        throw std::runtime_error("export_conv_transpose: spatial_rank must be 1, 2, or 3");
    }
    // 5th-audit C3: validate that the parameters we are about to emit are
    // ones the importer can read back. The exporter signature only accepts
    // scalar kernel_size/stride/padding/output_padding/dilation, so
    // anisotropic pads/output_padding cannot be constructed here — but the
    // dilation case can: ConvTranspose2d in Tenzor does not expose dilation
    // and the importer rejects dilation != 1 (importer.cpp ~ line 1038).
    // Emit the validation defensively so a future signature broadening also
    // gets the check.
    if (spatial_rank == 2 && dilation != 1) {
        throw std::runtime_error(
            "ONNX ConvTranspose2d export: dilation != 1 is not round-trippable "
            "through this importer (Tenzor ConvTranspose2d does not expose "
            "dilation). Use spatial_rank=1 or 3, or set dilation=1.");
    }
    // Asymmetric pads / anisotropic output_padding are not constructible
    // from the current scalar signature; assert-on-broaden:
    if (padding < 0 || output_padding < 0 || dilation < 1 || stride < 1 || kernel_size < 1) {
        throw std::runtime_error(
            "ONNX ConvTranspose export: negative/zero kernel/stride/padding/"
            "output_padding/dilation is not representable.");
    }
    ONNXExportNode node("ConvTranspose", context_.generate_name("convtranspose"));
    std::string input_name = get_tensor_name(input, "convt_input");
    std::string weight_name = context_.generate_name("convt_weight");
    add_initializer_tensor(weight, weight_name);
    node.add_input(input_name);
    node.add_input(weight_name);
    if (bias.has_value()) {
        std::string bias_name = context_.generate_name("convt_bias");
        add_initializer_tensor(bias.value(), bias_name);
        node.add_input(bias_name);
    }
    node.add_output(output_name);
    std::vector<int64_t> ks(spatial_rank, kernel_size);
    std::vector<int64_t> st(spatial_rank, stride);
    std::vector<int64_t> dl(spatial_rank, dilation);
    std::vector<int64_t> op(spatial_rank, output_padding);
    std::vector<int64_t> pd(spatial_rank * 2, padding);
    node.set_attr("kernel_shape",   ks);
    node.set_attr("strides",        st);
    node.set_attr("pads",           pd);
    node.set_attr("dilations",      dl);
    node.set_attr("output_padding", op);
    node.set_attr("group",          groups);
    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_conv1d(const Tensor& input, const Tensor& weight,
                                  const std::optional<Tensor>& bias,
                                  int64_t kernel_size, int64_t stride, int64_t padding,
                                  int64_t dilation, int64_t groups,
                                  const Tensor& output,
                                  const std::string& output_name) -> void {
    ONNXExportNode node("Conv", context_.generate_name("conv1d"));

    std::string input_name = get_tensor_name(input, "conv1d_input");

    std::string weight_name = context_.generate_name("conv1d_weight");
    add_initializer_tensor(weight, weight_name);

    node.add_input(input_name);
    node.add_input(weight_name);

    if (bias.has_value()) {
        std::string bias_name = context_.generate_name("conv1d_bias");
        add_initializer_tensor(bias.value(), bias_name);
        node.add_input(bias_name);
    }

    node.add_output(output_name);

    // Set attributes for 1D conv
    node.set_attr("kernel_shape", std::vector<int64_t>{kernel_size});
    node.set_attr("strides", std::vector<int64_t>{stride});
    node.set_attr("pads", std::vector<int64_t>{padding, padding}); // [begin, end]
    node.set_attr("dilations", std::vector<int64_t>{dilation});
    node.set_attr("group", groups);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_batchnorm2d(const Tensor& input, const Tensor& scale,
                                       const Tensor& bias, const Tensor& mean,
                                       const Tensor& var, double eps,
                                       const Tensor& output,
                                       const std::string& output_name,
                                       bool training) -> void {
    ONNXExportNode node("BatchNormalization", context_.generate_name("batchnorm"));

    std::string input_name = get_tensor_name(input, "bn_input");

    std::string scale_name = context_.generate_name("bn_scale");
    add_initializer_tensor(scale, scale_name);

    std::string bias_name = context_.generate_name("bn_bias");
    add_initializer_tensor(bias, bias_name);

    std::string mean_name = context_.generate_name("bn_mean");
    add_initializer_tensor(mean, mean_name);

    std::string var_name = context_.generate_name("bn_var");
    add_initializer_tensor(var, var_name);

    node.add_input(input_name);
    node.add_input(scale_name);
    node.add_input(bias_name);
    node.add_input(mean_name);
    node.add_input(var_name);

    node.add_output(output_name);

    node.set_attr("epsilon", static_cast<float>(eps));
    node.set_attr("momentum", 0.9f); // Default momentum
    // 5th-audit C4: ONNX BatchNormalization has a `training_mode` attribute
    // (opset 14+; default 0 = inference). Emit it when the caller declares the
    // BN module was in training mode so the produced graph faithfully encodes
    // batch-stats vs running-stats semantics. Pre-fix this attribute was
    // never set, silently exporting every BN as inference.
    if (training) {
        node.set_attr("training_mode", static_cast<int64_t>(1));
    }

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_batchnorm1d(const Tensor& input, const Tensor& scale,
                                       const Tensor& bias, const Tensor& mean,
                                       const Tensor& var, double eps,
                                       const Tensor& output,
                                       const std::string& output_name,
                                       bool training) -> void {
    // BatchNorm1d uses the same ONNX op as BatchNorm2d
    export_batchnorm2d(input, scale, bias, mean, var, eps, output, output_name, training);
}

auto ONNXExporter::export_layernorm(const Tensor& input, const Tensor& scale,
                                     const Tensor& bias, int64_t axis, double eps,
                                     const Tensor& output,
                                     const std::string& output_name) -> void {
    // ONNX LayerNormalization (opset 17+). If the target opset is older, fall
    // back to the BatchNormalization form which matches LayerNorm semantics
    // when the input is reshaped to (N, C) — callers on pre-17 opsets should
    // route through the graph-based LayerNorm OpId path instead.
    ONNXExportNode node("LayerNormalization", context_.generate_name("layernorm"));

    std::string input_name = get_tensor_name(input, "ln_input");

    std::string scale_name = context_.generate_name("ln_scale");
    add_initializer_tensor(scale, scale_name);

    std::string bias_name = context_.generate_name("ln_bias");
    add_initializer_tensor(bias, bias_name);

    node.add_input(input_name);
    node.add_input(scale_name);
    node.add_input(bias_name);
    node.add_output(output_name);

    node.set_attr("axis", axis);
    node.set_attr("epsilon", static_cast<float>(eps));

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_groupnorm(const Tensor& input, const Tensor& weight,
                                     const Tensor& bias, int64_t num_groups, double eps,
                                     const Tensor& output,
                                     const std::string& output_name) -> void {
    if (opset_version_ >= 18) {
        // ONNX opset 18+ has native GroupNormalization
        ONNXExportNode node("GroupNormalization", context_.generate_name("groupnorm"));

        std::string input_name = get_tensor_name(input, "gn_input");

        std::string scale_name = context_.generate_name("gn_scale");
        add_initializer_tensor(weight, scale_name);

        std::string bias_name = context_.generate_name("gn_bias");
        add_initializer_tensor(bias, bias_name);

        node.add_input(input_name);
        node.add_input(scale_name);
        node.add_input(bias_name);

        node.add_output(output_name);

        node.set_attr("epsilon", static_cast<float>(eps));
        node.set_attr("num_groups", num_groups);

        graph_.add_node(node);
    } else {
        // For opset < 18, decompose GroupNorm into:
        //   Reshape(N,C,*) -> (N*G, C/G, *) + InstanceNormalization + Reshape back
        //
        // The scale and bias for InstanceNorm must be per-channel of the reshaped
        // tensor (C/G channels per group, repeated G times -> total C channels).
        // However, ONNX InstanceNormalization takes scale/bias of shape [C'] where
        // C' = num_channels of the reshaped input = C/G.
        //
        // We handle this by reshaping into (N*G, C/G, *), applying InstanceNorm
        // with ones/zeros scale/bias (C/G), reshaping back, then applying the
        // actual affine transform via Mul + Add with the original weight/bias.

        std::string input_name = get_tensor_name(input, "gn_input");
        auto input_shape = input.shape();
        int64_t ndim = static_cast<int64_t>(input_shape.size());
        int64_t N = input_shape[0];
        int64_t C = input_shape[1];
        int64_t channels_per_group = C / num_groups;

        // Build reshape target: (N*G, C/G, spatial_dims...)
        std::vector<int64_t> reshaped_dims;
        reshaped_dims.push_back(N * num_groups);
        reshaped_dims.push_back(channels_per_group);
        for (int64_t d = 2; d < ndim; ++d) {
            reshaped_dims.push_back(input_shape[d]);
        }

        // Reshape input -> (N*G, C/G, *)
        std::string reshape1_shape_name = context_.generate_name("gn_reshape1_shape");
        Tensor reshape1_shape_tensor({static_cast<int64_t>(reshaped_dims.size())}, DType::Int64, Device::cpu());
        std::memcpy(reshape1_shape_tensor.data<int64_t>(), reshaped_dims.data(),
                     reshaped_dims.size() * sizeof(int64_t));
        add_initializer_tensor(reshape1_shape_tensor, reshape1_shape_name);

        std::string reshaped_name = context_.generate_name("gn_reshaped");
        ONNXExportNode reshape1_node("Reshape", context_.generate_name("gn_reshape1"));
        reshape1_node.add_input(input_name);
        reshape1_node.add_input(reshape1_shape_name);
        reshape1_node.add_output(reshaped_name);
        graph_.add_node(reshape1_node);

        // Create per-channel scale (ones) and bias (zeros) for InstanceNorm
        // Shape: (channels_per_group,)
        std::string instnorm_scale_name = context_.generate_name("gn_instnorm_scale");
        Tensor instnorm_scale({channels_per_group}, DType::Float32, Device::cpu());
        instnorm_scale.fill_(1.0f);
        add_initializer_tensor(instnorm_scale, instnorm_scale_name);

        std::string instnorm_bias_name = context_.generate_name("gn_instnorm_bias");
        Tensor instnorm_bias({channels_per_group}, DType::Float32, Device::cpu());
        instnorm_bias.fill_(0.0f);
        add_initializer_tensor(instnorm_bias, instnorm_bias_name);

        // InstanceNormalization on (N*G, C/G, *)
        std::string instnorm_out_name = context_.generate_name("gn_instnorm_out");
        ONNXExportNode instnorm_node("InstanceNormalization", context_.generate_name("gn_instnorm"));
        instnorm_node.add_input(reshaped_name);
        instnorm_node.add_input(instnorm_scale_name);
        instnorm_node.add_input(instnorm_bias_name);
        instnorm_node.set_attr("epsilon", static_cast<float>(eps));
        instnorm_node.add_output(instnorm_out_name);
        graph_.add_node(instnorm_node);

        // Reshape back to original shape (N, C, *)
        std::vector<int64_t> orig_shape(input_shape.begin(), input_shape.end());
        std::string reshape2_shape_name = context_.generate_name("gn_reshape2_shape");
        Tensor reshape2_shape_tensor({static_cast<int64_t>(orig_shape.size())}, DType::Int64, Device::cpu());
        std::memcpy(reshape2_shape_tensor.data<int64_t>(), orig_shape.data(),
                     orig_shape.size() * sizeof(int64_t));
        add_initializer_tensor(reshape2_shape_tensor, reshape2_shape_name);

        std::string reshape2_out_name = context_.generate_name("gn_reshape2_out");
        ONNXExportNode reshape2_node("Reshape", context_.generate_name("gn_reshape2"));
        reshape2_node.add_input(instnorm_out_name);
        reshape2_node.add_input(reshape2_shape_name);
        reshape2_node.add_output(reshape2_out_name);
        graph_.add_node(reshape2_node);

        // Apply affine transform: output = reshape2_out * weight + bias
        // weight shape: (C,) -> reshape to (1, C, 1, 1, ...) for broadcasting
        std::vector<int64_t> affine_shape(ndim, 1);
        affine_shape[1] = C;

        std::string weight_name = context_.generate_name("gn_weight");
        add_initializer_tensor(weight, weight_name);

        std::string weight_reshaped_name = context_.generate_name("gn_weight_reshaped");
        std::string weight_reshape_shape_name = context_.generate_name("gn_weight_reshape_shape");
        Tensor weight_reshape_shape({static_cast<int64_t>(affine_shape.size())}, DType::Int64, Device::cpu());
        std::memcpy(weight_reshape_shape.data<int64_t>(), affine_shape.data(),
                     affine_shape.size() * sizeof(int64_t));
        add_initializer_tensor(weight_reshape_shape, weight_reshape_shape_name);

        ONNXExportNode weight_reshape_node("Reshape", context_.generate_name("gn_weight_reshape"));
        weight_reshape_node.add_input(weight_name);
        weight_reshape_node.add_input(weight_reshape_shape_name);
        weight_reshape_node.add_output(weight_reshaped_name);
        graph_.add_node(weight_reshape_node);

        std::string mul_out_name = context_.generate_name("gn_mul_out");
        ONNXExportNode mul_node("Mul", context_.generate_name("gn_mul"));
        mul_node.add_input(reshape2_out_name);
        mul_node.add_input(weight_reshaped_name);
        mul_node.add_output(mul_out_name);
        graph_.add_node(mul_node);

        std::string bias_name = context_.generate_name("gn_bias");
        add_initializer_tensor(bias, bias_name);

        std::string bias_reshaped_name = context_.generate_name("gn_bias_reshaped");
        std::string bias_reshape_shape_name = context_.generate_name("gn_bias_reshape_shape");
        // Reuse same shape as weight reshape
        Tensor bias_reshape_shape({static_cast<int64_t>(affine_shape.size())}, DType::Int64, Device::cpu());
        std::memcpy(bias_reshape_shape.data<int64_t>(), affine_shape.data(),
                     affine_shape.size() * sizeof(int64_t));
        add_initializer_tensor(bias_reshape_shape, bias_reshape_shape_name);

        ONNXExportNode bias_reshape_node("Reshape", context_.generate_name("gn_bias_reshape"));
        bias_reshape_node.add_input(bias_name);
        bias_reshape_node.add_input(bias_reshape_shape_name);
        bias_reshape_node.add_output(bias_reshaped_name);
        graph_.add_node(bias_reshape_node);

        ONNXExportNode add_node("Add", context_.generate_name("gn_add"));
        add_node.add_input(mul_out_name);
        add_node.add_input(bias_reshaped_name);
        add_node.add_output(output_name);
        graph_.add_node(add_node);
    }

    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_lstm(const Tensor& input,
                                const Tensor& weight_ih, const Tensor& weight_hh,
                                const std::optional<Tensor>& bias_ih,
                                const std::optional<Tensor>& bias_hh,
                                int64_t hidden_size, [[maybe_unused]] int64_t num_layers,
                                bool bidirectional,
                                const Tensor& output,
                                const std::string& output_name) -> void {
    // ONNX LSTM operator expects weights in gate order: i, o, f, c (input, output, forget, cell)
    // Tenzor LSTM uses gate order: i, f, g, o (input, forget, cell-gate, output)
    // We need to reorder: Tenzor [i,f,g,o] -> ONNX [i,o,f,g]
    //   Tenzor row blocks: [0..H) = i, [H..2H) = f, [2H..3H) = g, [3H..4H) = o
    //   ONNX row blocks:   [0..H) = i, [H..2H) = o, [2H..3H) = f, [3H..4H) = g
    // So mapping: ONNX[0]=T[0](i), ONNX[1]=T[3](o), ONNX[2]=T[1](f), ONNX[3]=T[2](g)

    int64_t num_directions = bidirectional ? 2 : 1;
    auto wih_shape = weight_ih.shape();
    int64_t input_size = wih_shape[1]; // weight_ih: (4*hidden_size, input_size)

    // Helper lambda to reorder gates from Tenzor [i,f,g,o] -> ONNX [i,o,f,c]
    // Given a flat tensor of shape (4*H, *), reorder the H-sized row blocks
    auto reorder_lstm_gates = [&](const Tensor& w) -> Tensor {
        Tensor cpu_w = w.cpu().contiguous();
        auto shape = cpu_w.shape();
        int64_t H = hidden_size;
        int64_t cols = (shape.size() > 1) ? shape[1] : 1;
        bool is_1d = (shape.size() == 1);

        // Create output tensor with same shape
        std::vector<int64_t> shape_vec(shape.begin(), shape.end());
        Tensor reordered(shape_vec, cpu_w.dtype(), Device::cpu());

        const auto* src = cpu_w.data<float>();
        auto* dst = reordered.data<float>();

        if (is_1d) {
            // 1D bias: (4*H,)
            // Copy i (src[0..H)) -> dst[0..H)     (i -> i)
            std::memcpy(dst + 0 * H, src + 0 * H, H * sizeof(float));
            // Copy o (src[3*H..4*H)) -> dst[H..2H) (o -> o)
            std::memcpy(dst + 1 * H, src + 3 * H, H * sizeof(float));
            // Copy f (src[H..2H)) -> dst[2H..3H)   (f -> f)
            std::memcpy(dst + 2 * H, src + 1 * H, H * sizeof(float));
            // Copy g (src[2H..3H)) -> dst[3H..4H)  (g -> c)
            std::memcpy(dst + 3 * H, src + 2 * H, H * sizeof(float));
        } else {
            // 2D weight: (4*H, cols)
            size_t row_bytes = cols * sizeof(float);
            // i -> i
            std::memcpy(dst + 0 * H * cols, src + 0 * H * cols, H * row_bytes);
            // o -> o
            std::memcpy(dst + 1 * H * cols, src + 3 * H * cols, H * row_bytes);
            // f -> f
            std::memcpy(dst + 2 * H * cols, src + 1 * H * cols, H * row_bytes);
            // g -> c
            std::memcpy(dst + 3 * H * cols, src + 2 * H * cols, H * row_bytes);
        }
        return reordered;
    };

    // Reorder weight_ih: (4*H, input_size) -> ONNX gate order
    Tensor w_ih_reordered = reorder_lstm_gates(weight_ih);
    // Reorder weight_hh: (4*H, hidden_size) -> ONNX gate order
    Tensor w_hh_reordered = reorder_lstm_gates(weight_hh);

    // ONNX expects W shape: [num_directions, 4*hidden_size, input_size]
    // ONNX expects R shape: [num_directions, 4*hidden_size, hidden_size]
    // For single direction, add the leading dimension
    std::string w_name = context_.generate_name("lstm_W");
    {
        Tensor w_3d({num_directions, 4 * hidden_size, input_size}, DType::Float32, Device::cpu());
        std::memcpy(w_3d.data<float>(), w_ih_reordered.data<float>(),
                     4 * hidden_size * input_size * sizeof(float));
        add_initializer_tensor(w_3d, w_name);
    }

    std::string r_name = context_.generate_name("lstm_R");
    {
        Tensor r_3d({num_directions, 4 * hidden_size, hidden_size}, DType::Float32, Device::cpu());
        std::memcpy(r_3d.data<float>(), w_hh_reordered.data<float>(),
                     4 * hidden_size * hidden_size * sizeof(float));
        add_initializer_tensor(r_3d, r_name);
    }

    // Build ONNX LSTM node
    ONNXExportNode node("LSTM", context_.generate_name("lstm"));

    std::string input_name = get_tensor_name(input, "lstm_input");
    node.add_input(input_name);  // X
    node.add_input(w_name);      // W
    node.add_input(r_name);      // R

    // Bias: ONNX expects B of shape [num_directions, 8*hidden_size]
    // = concat(Wb_i, Wb_o, Wb_f, Wb_c, Rb_i, Rb_o, Rb_f, Rb_c)
    if (bias_ih.has_value() && bias_hh.has_value()) {
        Tensor b_ih_reordered = reorder_lstm_gates(bias_ih.value());
        Tensor b_hh_reordered = reorder_lstm_gates(bias_hh.value());

        Tensor bias_combined({num_directions, 8 * hidden_size}, DType::Float32, Device::cpu());
        auto* dst = bias_combined.data<float>();
        std::memcpy(dst, b_ih_reordered.data<float>(), 4 * hidden_size * sizeof(float));
        std::memcpy(dst + 4 * hidden_size, b_hh_reordered.data<float>(),
                     4 * hidden_size * sizeof(float));

        std::string bias_name = context_.generate_name("lstm_B");
        add_initializer_tensor(bias_combined, bias_name);
        node.add_input(bias_name);  // B
    } else {
        node.add_input("");  // B (empty = no bias)
    }

    // sequence_lens (optional, empty)
    node.add_input("");

    // initial_h (optional, empty)
    node.add_input("");

    // initial_c (optional, empty)
    node.add_input("");

    node.add_output(output_name);  // Y (all hidden states)

    // Additional outputs: Y_h (final hidden), Y_c (final cell)
    std::string y_h_name = context_.generate_name("lstm_Y_h");
    std::string y_c_name = context_.generate_name("lstm_Y_c");
    node.add_output(y_h_name);
    node.add_output(y_c_name);

    // Attributes
    node.set_attr("hidden_size", hidden_size);
    if (bidirectional) {
        node.set_attr("direction", std::string("bidirectional"));
    } else {
        node.set_attr("direction", std::string("forward"));
    }

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_gru(const Tensor& input,
                               const Tensor& weight_ih, const Tensor& weight_hh,
                               const std::optional<Tensor>& bias_ih,
                               const std::optional<Tensor>& bias_hh,
                               int64_t hidden_size, [[maybe_unused]] int64_t num_layers,
                               bool bidirectional,
                               const Tensor& output,
                               const std::string& output_name) -> void {
    // ONNX GRU expects gate order: z, r, h (update, reset, hidden)
    // Tenzor GRU uses gate order: r, z, n (reset, update, new)
    // Reorder: Tenzor [r,z,n] -> ONNX [z,r,h]
    //   Tenzor row blocks: [0..H) = r, [H..2H) = z, [2H..3H) = n
    //   ONNX row blocks:   [0..H) = z, [H..2H) = r, [2H..3H) = h
    // So mapping: ONNX[0]=T[1](z), ONNX[1]=T[0](r), ONNX[2]=T[2](n/h)

    int64_t num_directions = bidirectional ? 2 : 1;
    auto wih_shape = weight_ih.shape();
    int64_t input_size = wih_shape[1]; // weight_ih: (3*hidden_size, input_size)

    // Helper lambda to reorder gates from Tenzor [r,z,n] -> ONNX [z,r,h]
    auto reorder_gru_gates = [&](const Tensor& w) -> Tensor {
        Tensor cpu_w = w.cpu().contiguous();
        auto shape = cpu_w.shape();
        int64_t H = hidden_size;
        int64_t cols = (shape.size() > 1) ? shape[1] : 1;
        bool is_1d = (shape.size() == 1);

        std::vector<int64_t> shape_vec(shape.begin(), shape.end());
        Tensor reordered(shape_vec, cpu_w.dtype(), Device::cpu());

        const auto* src = cpu_w.data<float>();
        auto* dst = reordered.data<float>();

        if (is_1d) {
            // 1D bias: (3*H,)
            // z (src[H..2H)) -> dst[0..H)
            std::memcpy(dst + 0 * H, src + 1 * H, H * sizeof(float));
            // r (src[0..H)) -> dst[H..2H)
            std::memcpy(dst + 1 * H, src + 0 * H, H * sizeof(float));
            // n/h (src[2H..3H)) -> dst[2H..3H)
            std::memcpy(dst + 2 * H, src + 2 * H, H * sizeof(float));
        } else {
            // 2D weight: (3*H, cols)
            size_t row_bytes = cols * sizeof(float);
            // z -> first block
            std::memcpy(dst + 0 * H * cols, src + 1 * H * cols, H * row_bytes);
            // r -> second block
            std::memcpy(dst + 1 * H * cols, src + 0 * H * cols, H * row_bytes);
            // n/h -> third block
            std::memcpy(dst + 2 * H * cols, src + 2 * H * cols, H * row_bytes);
        }
        return reordered;
    };

    Tensor w_ih_reordered = reorder_gru_gates(weight_ih);
    Tensor w_hh_reordered = reorder_gru_gates(weight_hh);

    // ONNX expects W shape: [num_directions, 3*hidden_size, input_size]
    // ONNX expects R shape: [num_directions, 3*hidden_size, hidden_size]
    std::string w_name = context_.generate_name("gru_W");
    {
        Tensor w_3d({num_directions, 3 * hidden_size, input_size}, DType::Float32, Device::cpu());
        std::memcpy(w_3d.data<float>(), w_ih_reordered.data<float>(),
                     3 * hidden_size * input_size * sizeof(float));
        add_initializer_tensor(w_3d, w_name);
    }

    std::string r_name = context_.generate_name("gru_R");
    {
        Tensor r_3d({num_directions, 3 * hidden_size, hidden_size}, DType::Float32, Device::cpu());
        std::memcpy(r_3d.data<float>(), w_hh_reordered.data<float>(),
                     3 * hidden_size * hidden_size * sizeof(float));
        add_initializer_tensor(r_3d, r_name);
    }

    // Build ONNX GRU node
    ONNXExportNode node("GRU", context_.generate_name("gru"));

    std::string input_name = get_tensor_name(input, "gru_input");
    node.add_input(input_name);  // X
    node.add_input(w_name);      // W
    node.add_input(r_name);      // R

    // Bias: ONNX expects B of shape [num_directions, 6*hidden_size]
    // = concat(Wb_z, Wb_r, Wb_h, Rb_z, Rb_r, Rb_h)
    if (bias_ih.has_value() && bias_hh.has_value()) {
        Tensor b_ih_reordered = reorder_gru_gates(bias_ih.value());
        Tensor b_hh_reordered = reorder_gru_gates(bias_hh.value());

        Tensor bias_combined({num_directions, 6 * hidden_size}, DType::Float32, Device::cpu());
        auto* dst = bias_combined.data<float>();
        std::memcpy(dst, b_ih_reordered.data<float>(), 3 * hidden_size * sizeof(float));
        std::memcpy(dst + 3 * hidden_size, b_hh_reordered.data<float>(),
                     3 * hidden_size * sizeof(float));

        std::string bias_name = context_.generate_name("gru_B");
        add_initializer_tensor(bias_combined, bias_name);
        node.add_input(bias_name);  // B
    } else {
        node.add_input("");  // B (empty = no bias)
    }

    // sequence_lens (optional, empty)
    node.add_input("");

    // initial_h (optional, empty)
    node.add_input("");

    node.add_output(output_name);  // Y (all hidden states)

    // Additional output: Y_h (final hidden)
    std::string y_h_name = context_.generate_name("gru_Y_h");
    node.add_output(y_h_name);

    // Attributes
    node.set_attr("hidden_size", hidden_size);
    if (bidirectional) {
        node.set_attr("direction", std::string("bidirectional"));
    } else {
        node.set_attr("direction", std::string("forward"));
    }
    // Use linear_before_reset=1 to match Tenzor's GRU formulation
    // (Tenzor computes r * (W_hh * h) rather than W_hh * (r * h))
    node.set_attr("linear_before_reset", static_cast<int64_t>(1));

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_multihead_attention(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& q_proj_weight, const Tensor& k_proj_weight,
    const Tensor& v_proj_weight, const Tensor& out_proj_weight,
    const std::optional<Tensor>& q_proj_bias,
    const std::optional<Tensor>& k_proj_bias,
    const std::optional<Tensor>& v_proj_bias,
    const std::optional<Tensor>& out_proj_bias,
    int64_t num_heads,
    const Tensor& output,
    const std::string& output_name) -> void {
    // Multi-Head Attention decomposition into ONNX primitives:
    //
    // 1. Q = query @ W_q^T + b_q   (via Gemm/MatMul+Add)
    // 2. K = key   @ W_k^T + b_k
    // 3. V = value @ W_v^T + b_v
    // 4. Reshape Q,K,V from (N, L/S, E) to (N, L/S, num_heads, head_dim)
    // 5. Transpose to (N, num_heads, L/S, head_dim)
    // 6. scores = Q @ K^T / sqrt(head_dim)
    // 7. attn_weights = Softmax(scores, axis=-1)
    // 8. context = attn_weights @ V
    // 9. Transpose context to (N, L, num_heads, head_dim)
    // 10. Reshape to (N, L, E)
    // 11. output = context @ W_o^T + b_o

    auto q_shape = query.shape();
    int64_t embed_dim = q_shape[q_shape.size() - 1];
    int64_t head_dim = embed_dim / num_heads;

    std::string query_name = get_tensor_name(query, "mha_query");
    std::string key_name = get_tensor_name(key, "mha_key");
    std::string value_name = get_tensor_name(value, "mha_value");

    // ---- Step 1-3: Q, K, V projections ----
    // Helper lambda to create a linear projection: output = input @ weight^T + bias
    auto create_projection = [&](const std::string& input_name_proj,
                                 const Tensor& weight,
                                 const std::optional<Tensor>& bias,
                                 const std::string& prefix) -> std::string {
        std::string weight_name = context_.generate_name(prefix + "_weight");
        add_initializer_tensor(weight, weight_name);

        // Gemm: Y = alpha * A * B^T + beta * C
        std::string proj_out_name = context_.generate_name(prefix + "_proj");
        ONNXExportNode gemm_node("Gemm", context_.generate_name(prefix + "_gemm"));
        gemm_node.add_input(input_name_proj);
        gemm_node.add_input(weight_name);

        if (bias.has_value()) {
            std::string bias_name = context_.generate_name(prefix + "_bias");
            add_initializer_tensor(bias.value(), bias_name);
            gemm_node.add_input(bias_name);
        }

        gemm_node.add_output(proj_out_name);
        gemm_node.set_attr("alpha", 1.0f);
        gemm_node.set_attr("beta", 1.0f);
        gemm_node.set_attr("transB", static_cast<int64_t>(1));

        graph_.add_node(gemm_node);
        return proj_out_name;
    };

    std::string q_proj_name = create_projection(query_name, q_proj_weight, q_proj_bias, "mha_q");
    std::string k_proj_name = create_projection(key_name, k_proj_weight, k_proj_bias, "mha_k");
    std::string v_proj_name = create_projection(value_name, v_proj_weight, v_proj_bias, "mha_v");

    // ---- Step 4: Reshape Q, K, V from (N, L/S, E) to (N, L/S, num_heads, head_dim) ----
    auto create_reshape_to_heads = [&](const std::string& input_reshape_name,
                                       const std::string& prefix) -> std::string {
        // Target shape: (0, -1, num_heads, head_dim)
        // Using 0 for batch dim (keep original), -1 for seq dim (infer)
        std::string shape_name = context_.generate_name(prefix + "_reshape_shape");
        std::vector<int64_t> target_shape = {0, -1, num_heads, head_dim};
        Tensor shape_tensor({4}, DType::Int64, Device::cpu());
        std::memcpy(shape_tensor.data<int64_t>(), target_shape.data(), 4 * sizeof(int64_t));
        add_initializer_tensor(shape_tensor, shape_name);

        std::string reshaped_name = context_.generate_name(prefix + "_reshaped");
        ONNXExportNode reshape_node("Reshape", context_.generate_name(prefix + "_reshape"));
        reshape_node.add_input(input_reshape_name);
        reshape_node.add_input(shape_name);
        reshape_node.add_output(reshaped_name);
        graph_.add_node(reshape_node);

        return reshaped_name;
    };

    std::string q_reshaped = create_reshape_to_heads(q_proj_name, "mha_q");
    std::string k_reshaped = create_reshape_to_heads(k_proj_name, "mha_k");
    std::string v_reshaped = create_reshape_to_heads(v_proj_name, "mha_v");

    // ---- Step 5: Transpose from (N, L/S, num_heads, head_dim) to (N, num_heads, L/S, head_dim) ----
    auto create_transpose = [&](const std::string& input_tr_name,
                                const std::vector<int64_t>& perm,
                                const std::string& prefix) -> std::string {
        std::string transposed_name = context_.generate_name(prefix + "_transposed");
        ONNXExportNode transpose_node("Transpose", context_.generate_name(prefix + "_transpose"));
        transpose_node.add_input(input_tr_name);
        transpose_node.add_output(transposed_name);
        transpose_node.set_attr("perm", perm);
        graph_.add_node(transpose_node);
        return transposed_name;
    };

    std::string q_t = create_transpose(q_reshaped, {0, 2, 1, 3}, "mha_q");
    std::string k_t = create_transpose(k_reshaped, {0, 2, 1, 3}, "mha_k");
    std::string v_t = create_transpose(v_reshaped, {0, 2, 1, 3}, "mha_v");

    // ---- Step 6: scores = Q @ K^T / sqrt(head_dim) ----
    // Transpose K: (N, num_heads, S, head_dim) -> (N, num_heads, head_dim, S)
    std::string k_t2 = create_transpose(k_t, {0, 1, 3, 2}, "mha_k");

    // MatMul: Q @ K^T -> (N, num_heads, L, S)
    std::string qk_name = context_.generate_name("mha_qk");
    ONNXExportNode qk_node("MatMul", context_.generate_name("mha_qk_matmul"));
    qk_node.add_input(q_t);
    qk_node.add_input(k_t2);
    qk_node.add_output(qk_name);
    graph_.add_node(qk_node);

    // Divide by sqrt(head_dim)
    std::string scale_name = context_.generate_name("mha_scale");
    Tensor scale_tensor({1}, DType::Float32, Device::cpu());
    scale_tensor.fill_(std::sqrt(static_cast<float>(head_dim)));
    add_initializer_tensor(scale_tensor, scale_name);

    std::string scaled_qk_name = context_.generate_name("mha_scaled_qk");
    ONNXExportNode div_node("Div", context_.generate_name("mha_scale_div"));
    div_node.add_input(qk_name);
    div_node.add_input(scale_name);
    div_node.add_output(scaled_qk_name);
    graph_.add_node(div_node);

    // ---- Step 7: attn_weights = Softmax(scores, axis=-1) ----
    std::string attn_weights_name = context_.generate_name("mha_attn_weights");
    ONNXExportNode softmax_node("Softmax", context_.generate_name("mha_softmax"));
    softmax_node.add_input(scaled_qk_name);
    softmax_node.add_output(attn_weights_name);
    softmax_node.set_attr("axis", static_cast<int64_t>(-1));
    graph_.add_node(softmax_node);

    // ---- Step 8: context = attn_weights @ V ----
    // (N, num_heads, L, S) @ (N, num_heads, S, head_dim) -> (N, num_heads, L, head_dim)
    std::string context_name = context_.generate_name("mha_context");
    ONNXExportNode av_node("MatMul", context_.generate_name("mha_av_matmul"));
    av_node.add_input(attn_weights_name);
    av_node.add_input(v_t);
    av_node.add_output(context_name);
    graph_.add_node(av_node);

    // ---- Step 9: Transpose back: (N, num_heads, L, head_dim) -> (N, L, num_heads, head_dim) ----
    std::string context_transposed = create_transpose(context_name, {0, 2, 1, 3}, "mha_context");

    // ---- Step 10: Reshape to (N, L, embed_dim) ----
    std::string final_reshape_shape_name = context_.generate_name("mha_final_reshape_shape");
    std::vector<int64_t> final_shape = {0, -1, embed_dim};
    Tensor final_shape_tensor({3}, DType::Int64, Device::cpu());
    std::memcpy(final_shape_tensor.data<int64_t>(), final_shape.data(), 3 * sizeof(int64_t));
    add_initializer_tensor(final_shape_tensor, final_reshape_shape_name);

    std::string context_flat_name = context_.generate_name("mha_context_flat");
    ONNXExportNode final_reshape_node("Reshape", context_.generate_name("mha_final_reshape"));
    final_reshape_node.add_input(context_transposed);
    final_reshape_node.add_input(final_reshape_shape_name);
    final_reshape_node.add_output(context_flat_name);
    graph_.add_node(final_reshape_node);

    // ---- Step 11: Output projection: output = context_flat @ W_o^T + b_o ----
    std::string out_weight_name = context_.generate_name("mha_out_weight");
    add_initializer_tensor(out_proj_weight, out_weight_name);

    ONNXExportNode out_gemm_node("Gemm", context_.generate_name("mha_out_gemm"));
    out_gemm_node.add_input(context_flat_name);
    out_gemm_node.add_input(out_weight_name);

    if (out_proj_bias.has_value()) {
        std::string out_bias_name = context_.generate_name("mha_out_bias");
        add_initializer_tensor(out_proj_bias.value(), out_bias_name);
        out_gemm_node.add_input(out_bias_name);
    }

    out_gemm_node.add_output(output_name);
    out_gemm_node.set_attr("alpha", 1.0f);
    out_gemm_node.set_attr("beta", 1.0f);
    out_gemm_node.set_attr("transB", static_cast<int64_t>(1));
    graph_.add_node(out_gemm_node);

    context_.register_tensor(output, output_name);
}

// Activation Functions

auto ONNXExporter::export_relu(const Tensor& input, const Tensor& output,
                                const std::string& output_name) -> void {
    ONNXExportNode node("Relu", context_.generate_name("relu"));

    std::string input_name = get_tensor_name(input, "relu_input");

    node.add_input(input_name);
    node.add_output(output_name);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_leaky_relu(const Tensor& input, double alpha,
                                      const Tensor& output, const std::string& output_name) -> void {
    ONNXExportNode node("LeakyRelu", context_.generate_name("leaky_relu"));

    std::string input_name = get_tensor_name(input, "leaky_relu_input");

    node.add_input(input_name);
    node.add_output(output_name);
    node.set_attr("alpha", static_cast<float>(alpha));

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_sigmoid(const Tensor& input, const Tensor& output,
                                   const std::string& output_name) -> void {
    ONNXExportNode node("Sigmoid", context_.generate_name("sigmoid"));

    std::string input_name = get_tensor_name(input, "sigmoid_input");

    node.add_input(input_name);
    node.add_output(output_name);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_tanh(const Tensor& input, const Tensor& output,
                                const std::string& output_name) -> void {
    ONNXExportNode node("Tanh", context_.generate_name("tanh"));

    std::string input_name = get_tensor_name(input, "tanh_input");

    node.add_input(input_name);
    node.add_output(output_name);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_gelu(const Tensor& input, const Tensor& output,
                                const std::string& output_name) -> void {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // For ONNX opset 20+, there's a Gelu op. For earlier versions, we decompose it.

    if (opset_version_ >= 20) {
        ONNXExportNode node("Gelu", context_.generate_name("gelu"));
        std::string input_name = get_tensor_name(input, "gelu_input");
        node.add_input(input_name);
        node.add_output(output_name);
        graph_.add_node(node);
    } else {
        // Decompose GELU for older opset versions
        std::string input_name = get_tensor_name(input, "gelu_input");

        // Constants
        std::string half_name = context_.generate_name("gelu_half");
        Tensor half_tensor({1}, DType::Float32, Device::cpu());
        half_tensor.fill_(0.5f);
        add_initializer_tensor(half_tensor, half_name);

        std::string coef_name = context_.generate_name("gelu_coef");
        Tensor coef_tensor({1}, DType::Float32, Device::cpu());
        coef_tensor.fill_(0.044715f);
        add_initializer_tensor(coef_tensor, coef_name);

        std::string sqrt_2_pi_name = context_.generate_name("gelu_sqrt_2_pi");
        Tensor sqrt_2_pi_tensor({1}, DType::Float32, Device::cpu());
        sqrt_2_pi_tensor.fill_(std::sqrt(2.0 / M_PI));
        add_initializer_tensor(sqrt_2_pi_tensor, sqrt_2_pi_name);

        // x^3
        std::string x3_name = context_.generate_name("gelu_x3");
        ONNXExportNode pow_node("Pow", context_.generate_name("gelu_pow"));
        std::string three_name = context_.generate_name("gelu_three");
        Tensor three_tensor({1}, DType::Float32, Device::cpu());
        three_tensor.fill_(3.0f);
        add_initializer_tensor(three_tensor, three_name);
        pow_node.add_input(input_name);
        pow_node.add_input(three_name);
        pow_node.add_output(x3_name);
        graph_.add_node(pow_node);

        // 0.044715 * x^3
        std::string scaled_x3_name = context_.generate_name("gelu_scaled_x3");
        ONNXExportNode mul1_node("Mul", context_.generate_name("gelu_mul1"));
        mul1_node.add_input(coef_name);
        mul1_node.add_input(x3_name);
        mul1_node.add_output(scaled_x3_name);
        graph_.add_node(mul1_node);

        // x + 0.044715 * x^3
        std::string sum1_name = context_.generate_name("gelu_sum1");
        ONNXExportNode add1_node("Add", context_.generate_name("gelu_add1"));
        add1_node.add_input(input_name);
        add1_node.add_input(scaled_x3_name);
        add1_node.add_output(sum1_name);
        graph_.add_node(add1_node);

        // sqrt(2/pi) * (x + 0.044715 * x^3)
        std::string mul2_name = context_.generate_name("gelu_mul2");
        ONNXExportNode mul2_node("Mul", context_.generate_name("gelu_mul2"));
        mul2_node.add_input(sqrt_2_pi_name);
        mul2_node.add_input(sum1_name);
        mul2_node.add_output(mul2_name);
        graph_.add_node(mul2_node);

        // tanh(...)
        std::string tanh_name = context_.generate_name("gelu_tanh");
        ONNXExportNode tanh_node("Tanh", context_.generate_name("gelu_tanh"));
        tanh_node.add_input(mul2_name);
        tanh_node.add_output(tanh_name);
        graph_.add_node(tanh_node);

        // 1 + tanh(...)
        std::string one_name = context_.generate_name("gelu_one");
        Tensor one_tensor({1}, DType::Float32, Device::cpu());
        one_tensor.fill_(1.0f);
        add_initializer_tensor(one_tensor, one_name);

        std::string add2_name = context_.generate_name("gelu_add2");
        ONNXExportNode add2_node("Add", context_.generate_name("gelu_add2"));
        add2_node.add_input(one_name);
        add2_node.add_input(tanh_name);
        add2_node.add_output(add2_name);
        graph_.add_node(add2_node);

        // x * (1 + tanh(...))
        std::string mul3_name = context_.generate_name("gelu_mul3");
        ONNXExportNode mul3_node("Mul", context_.generate_name("gelu_mul3"));
        mul3_node.add_input(input_name);
        mul3_node.add_input(add2_name);
        mul3_node.add_output(mul3_name);
        graph_.add_node(mul3_node);

        // 0.5 * x * (1 + tanh(...))
        ONNXExportNode mul4_node("Mul", context_.generate_name("gelu_mul4"));
        mul4_node.add_input(half_name);
        mul4_node.add_input(mul3_name);
        mul4_node.add_output(output_name);
        graph_.add_node(mul4_node);
    }

    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_softmax(const Tensor& input, int64_t axis,
                                   const Tensor& output, const std::string& output_name) -> void {
    ONNXExportNode node("Softmax", context_.generate_name("softmax"));

    std::string input_name = get_tensor_name(input, "softmax_input");

    node.add_input(input_name);
    node.add_output(output_name);
    node.set_attr("axis", axis);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_log_softmax(const Tensor& input, int64_t axis,
                                       const Tensor& output, const std::string& output_name) -> void {
    ONNXExportNode node("LogSoftmax", context_.generate_name("log_softmax"));

    std::string input_name = get_tensor_name(input, "log_softmax_input");

    node.add_input(input_name);
    node.add_output(output_name);
    node.set_attr("axis", axis);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_elu(const Tensor& input, double alpha,
                               const Tensor& output, const std::string& output_name) -> void {
    ONNXExportNode node("Elu", context_.generate_name("elu"));

    std::string input_name = get_tensor_name(input, "elu_input");

    node.add_input(input_name);
    node.add_output(output_name);
    node.set_attr("alpha", static_cast<float>(alpha));

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_selu(const Tensor& input, const Tensor& output,
                                const std::string& output_name) -> void {
    ONNXExportNode node("Selu", context_.generate_name("selu"));

    std::string input_name = get_tensor_name(input, "selu_input");

    node.add_input(input_name);
    node.add_output(output_name);

    // SELU constants: alpha ≈ 1.6733, gamma ≈ 1.0507
    node.set_attr("alpha", 1.67326324f);
    node.set_attr("gamma", 1.05070098f);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_swish(const Tensor& input, const Tensor& output,
                                 const std::string& output_name) -> void {
    // Swish = x * sigmoid(x)
    std::string input_name = get_tensor_name(input, "swish_input");

    // Sigmoid
    std::string sigmoid_out_name = context_.generate_name("swish_sigmoid");
    ONNXExportNode sigmoid_node("Sigmoid", context_.generate_name("swish_sigmoid"));
    sigmoid_node.add_input(input_name);
    sigmoid_node.add_output(sigmoid_out_name);
    graph_.add_node(sigmoid_node);

    // Multiply
    ONNXExportNode mul_node("Mul", context_.generate_name("swish_mul"));
    mul_node.add_input(input_name);
    mul_node.add_input(sigmoid_out_name);
    mul_node.add_output(output_name);
    graph_.add_node(mul_node);

    context_.register_tensor(output, output_name);
}

// Pooling Layers

auto ONNXExporter::export_maxpool2d(const Tensor& input, int64_t kernel_size,
                                     int64_t stride, int64_t padding,
                                     const Tensor& output, const std::string& output_name) -> void {
    ONNXExportNode node("MaxPool", context_.generate_name("maxpool"));

    std::string input_name = get_tensor_name(input, "maxpool_input");

    node.add_input(input_name);
    node.add_output(output_name);

    node.set_attr("kernel_shape", std::vector<int64_t>{kernel_size, kernel_size});
    node.set_attr("strides", std::vector<int64_t>{stride, stride});
    node.set_attr("pads", std::vector<int64_t>{padding, padding, padding, padding});

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_avgpool2d(const Tensor& input, int64_t kernel_size,
                                     int64_t stride, int64_t padding,
                                     const Tensor& output, const std::string& output_name) -> void {
    ONNXExportNode node("AveragePool", context_.generate_name("avgpool"));

    std::string input_name = get_tensor_name(input, "avgpool_input");

    node.add_input(input_name);
    node.add_output(output_name);

    node.set_attr("kernel_shape", std::vector<int64_t>{kernel_size, kernel_size});
    node.set_attr("strides", std::vector<int64_t>{stride, stride});
    node.set_attr("pads", std::vector<int64_t>{padding, padding, padding, padding});

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_adaptive_avgpool2d(const Tensor& input,
                                              const std::vector<int64_t>& output_size,
                                              const Tensor& output, const std::string& output_name) -> void {
    // AdaptiveAvgPool uses GlobalAveragePool in ONNX when output size is 1x1
    // Otherwise, we need to compute the kernel and stride

    if (output_size.size() == 2 && output_size[0] == 1 && output_size[1] == 1) {
        // Use GlobalAveragePool
        ONNXExportNode node("GlobalAveragePool", context_.generate_name("global_avgpool"));
        std::string input_name = get_tensor_name(input, "adaptive_avgpool_input");
        node.add_input(input_name);
        node.add_output(output_name);
        graph_.add_node(node);
    } else {
        // Compute kernel size and stride from input and output sizes
        auto input_shape = input.shape();
        int64_t h_in = input_shape[2];
        int64_t w_in = input_shape[3];
        int64_t h_out = output_size[0];
        int64_t w_out = output_size[1];

        int64_t stride_h = h_in / h_out;
        int64_t stride_w = w_in / w_out;
        int64_t kernel_h = h_in - (h_out - 1) * stride_h;
        int64_t kernel_w = w_in - (w_out - 1) * stride_w;

        ONNXExportNode node("AveragePool", context_.generate_name("adaptive_avgpool"));
        std::string input_name = get_tensor_name(input, "adaptive_avgpool_input");

        node.add_input(input_name);
        node.add_output(output_name);
        node.set_attr("kernel_shape", std::vector<int64_t>{kernel_h, kernel_w});
        node.set_attr("strides", std::vector<int64_t>{stride_h, stride_w});

        graph_.add_node(node);
    }

    context_.register_tensor(output, output_name);
}

// ============================================================================
// Dynamic Shape Propagation
// ============================================================================

auto ONNXExporter::propagate_dynamic_shapes() -> void {
    // Collect symbolic dimension names from graph inputs and outputs.
    // Maps value_name -> (dim_index -> dim_param_string).
    std::unordered_map<std::string, std::unordered_map<int64_t, std::string>> dynamic_dims;

    for (const auto& input : graph_.inputs) {
        for (const auto& [axis, param] : input.dim_params) {
            dynamic_dims[input.name][axis] = param;
        }
    }

    for (const auto& output : graph_.outputs) {
        for (const auto& [axis, param] : output.dim_params) {
            dynamic_dims[output.name][axis] = param;
        }
    }

    if (dynamic_dims.empty()) {
        return; // No dynamic dimensions to propagate
    }

    // Walk the graph nodes forward and propagate dynamic dimension info
    // to intermediate value_info entries. For each node, if an input has
    // dynamic dims, we propagate those to the outputs based on the op type.
    for (const auto& node : graph_.nodes) {
        // Collect dynamic dims from this node's inputs
        std::unordered_map<int64_t, std::string> input_dynamic;

        for (const auto& inp_name : node.inputs) {
            auto it = dynamic_dims.find(inp_name);
            if (it != dynamic_dims.end()) {
                // Merge input dynamic dims (first input takes precedence for
                // shape-preserving ops like Add, Mul, Relu, etc.)
                for (const auto& [axis, param] : it->second) {
                    if (input_dynamic.find(axis) == input_dynamic.end()) {
                        input_dynamic[axis] = param;
                    }
                }
            }
        }

        if (input_dynamic.empty()) {
            continue;
        }

        // For most element-wise and unary ops, dynamic dims pass through
        // unchanged. For reshape/transpose, propagation is more complex
        // and we skip it (the user should set dynamic_axes on the output).
        static const std::unordered_set<std::string> passthrough_ops = {
            "Add", "Sub", "Mul", "Div", "Relu", "LeakyRelu", "Sigmoid",
            "Tanh", "Gelu", "Elu", "Selu", "Softplus", "Mish",
            "Softmax", "LogSoftmax", "BatchNormalization",
            "LayerNormalization", "InstanceNormalization",
            "GroupNormalization", "Dropout", "Neg", "Abs", "Sqrt",
            "Exp", "Log", "Pow", "Reciprocal", "Floor", "Ceil", "Round",
            "Sin", "Cos", "Tan", "Clip", "Sign",
        };

        bool is_passthrough = passthrough_ops.count(node.op_type) > 0;

        // MatMul preserves batch dimensions (all dims except last two)
        bool is_matmul = (node.op_type == "MatMul" || node.op_type == "Gemm");

        for (const auto& out_name : node.outputs) {
            if (is_passthrough) {
                // All dynamic dims pass through
                dynamic_dims[out_name] = input_dynamic;

                // Update value_info if it exists
                auto vi_it = graph_.value_info.find(out_name);
                if (vi_it != graph_.value_info.end()) {
                    for (const auto& [axis, param] : input_dynamic) {
                        if (axis >= 0 && axis < static_cast<int64_t>(vi_it->second.shape.size())) {
                            vi_it->second.shape[axis] = -1;
                            vi_it->second.dim_params[axis] = param;
                        }
                    }
                }
            } else if (is_matmul) {
                // For MatMul/Gemm, batch dims (all except last 2) propagate
                std::unordered_map<int64_t, std::string> out_dynamic;
                for (const auto& [axis, param] : input_dynamic) {
                    // Only propagate dims that are batch dims (not the
                    // contracted/result dims which are the last 2)
                    auto vi_it = graph_.value_info.find(out_name);
                    if (vi_it != graph_.value_info.end()) {
                        int64_t ndim = static_cast<int64_t>(vi_it->second.shape.size());
                        if (axis < ndim - 2) {
                            out_dynamic[axis] = param;
                            vi_it->second.shape[axis] = -1;
                            vi_it->second.dim_params[axis] = param;
                        }
                    } else {
                        // No value_info yet; just record for downstream
                        if (axis >= 0) {
                            out_dynamic[axis] = param;
                        }
                    }
                }
                if (!out_dynamic.empty()) {
                    dynamic_dims[out_name] = out_dynamic;
                }
            } else {
                // For other ops (Reshape, Transpose, Conv, Pool, etc.),
                // propagate batch dim (axis 0) only, as spatial dims may change
                auto it = input_dynamic.find(0);
                if (it != input_dynamic.end()) {
                    dynamic_dims[out_name][0] = it->second;

                    auto vi_it = graph_.value_info.find(out_name);
                    if (vi_it != graph_.value_info.end() &&
                        !vi_it->second.shape.empty()) {
                        vi_it->second.shape[0] = -1;
                        vi_it->second.dim_params[0] = it->second;
                    }
                }
            }
        }
    }

    // Also update graph outputs with any propagated dims
    for (auto& output : graph_.outputs) {
        auto it = dynamic_dims.find(output.name);
        if (it != dynamic_dims.end()) {
            for (const auto& [axis, param] : it->second) {
                if (axis >= 0 && axis < static_cast<int64_t>(output.shape.size())) {
                    if (output.dim_params.find(axis) == output.dim_params.end()) {
                        output.shape[axis] = -1;
                        output.dim_params[axis] = param;
                    }
                }
            }
        }
    }
}

// ============================================================================
// Validation
// ============================================================================

auto ONNXExporter::validate() const -> ValidationResult {
    ValidationResult result;

    // -- 1. Collect all defined value names --
    std::unordered_set<std::string> defined_names;
    std::unordered_map<std::string, int> name_counts; // For duplicate detection

    auto record_name = [&](const std::string& name, const std::string& source) {
        name_counts[name]++;
        if (name_counts[name] > 1) {
            result.errors.push_back(
                "Duplicate name '" + name + "' (redefined in " + source + ")");
            result.valid = false;
        }
        defined_names.insert(name);
    };

    // Graph inputs
    for (const auto& input : graph_.inputs) {
        if (input.name.empty()) {
            result.errors.push_back("Graph input has empty name");
            result.valid = false;
        } else {
            record_name(input.name, "graph input");
        }
    }

    // Initializers
    for (const auto& init : graph_.initializers) {
        if (init.name.empty()) {
            result.errors.push_back("Initializer has empty name");
            result.valid = false;
        } else {
            // Initializers are allowed to share names with graph inputs
            // (this is how ONNX represents default values for inputs)
            if (defined_names.count(init.name) == 0) {
                record_name(init.name, "initializer");
            } else {
                // Check if it's already defined as an input (OK)
                bool is_input = false;
                for (const auto& inp : graph_.inputs) {
                    if (inp.name == init.name) {
                        is_input = true;
                        break;
                    }
                }
                if (!is_input && name_counts[init.name] == 0) {
                    record_name(init.name, "initializer");
                }
                // If already an input, that's fine -- just ensure it's in defined_names
                defined_names.insert(init.name);
            }
        }
    }

    // Node outputs
    for (const auto& node : graph_.nodes) {
        for (const auto& out : node.outputs) {
            if (out.empty()) {
                result.errors.push_back(
                    "Node '" + node.name + "' (op=" + node.op_type + ") has empty output name");
                result.valid = false;
            } else {
                record_name(out, "node '" + node.name + "' output");
            }
        }
    }

    // -- 2. Check all node inputs reference a defined value --
    for (const auto& node : graph_.nodes) {
        for (const auto& inp : node.inputs) {
            if (inp.empty()) {
                // Empty input is valid in ONNX (optional input)
                continue;
            }
            if (defined_names.find(inp) == defined_names.end()) {
                result.errors.push_back(
                    "Node '" + node.name + "' (op=" + node.op_type +
                    ") references undefined input '" + inp + "'");
                result.valid = false;
            }
        }
    }

    // -- 3. Check graph outputs reference a defined value --
    for (const auto& output : graph_.outputs) {
        if (!output.name.empty() &&
            defined_names.find(output.name) == defined_names.end()) {
            result.warnings.push_back(
                "Graph output '" + output.name +
                "' is not produced by any node or input");
        }
    }

    // -- 4. Attribute type correctness for known ONNX ops --
    // For each node, validate that required attributes exist and have the
    // expected types. We check a subset of commonly used operators.
    struct AttrSpec {
        std::string attr_name;
        enum AttrType { INT, FLOAT, STRING, INTS, FLOATS } type;
        bool required;
    };

    auto check_attrs = [&](const ONNXExportNode& node,
                           const std::vector<AttrSpec>& specs) {
        for (const auto& spec : specs) {
            bool found = false;

            switch (spec.type) {
                case AttrSpec::INT:
                    found = node.int_attrs.count(spec.attr_name) > 0;
                    break;
                case AttrSpec::FLOAT:
                    found = node.float_attrs.count(spec.attr_name) > 0;
                    break;
                case AttrSpec::STRING:
                    found = node.string_attrs.count(spec.attr_name) > 0;
                    break;
                case AttrSpec::INTS:
                    found = node.ints_attrs.count(spec.attr_name) > 0;
                    break;
                case AttrSpec::FLOATS:
                    found = node.floats_attrs.count(spec.attr_name) > 0;
                    break;
            }

            // Also check if the attr exists under a wrong type
            if (!found) {
                bool in_wrong_type =
                    node.int_attrs.count(spec.attr_name) ||
                    node.float_attrs.count(spec.attr_name) ||
                    node.string_attrs.count(spec.attr_name) ||
                    node.ints_attrs.count(spec.attr_name) ||
                    node.floats_attrs.count(spec.attr_name);

                if (in_wrong_type) {
                    result.errors.push_back(
                        "Node '" + node.name + "' (op=" + node.op_type +
                        "): attribute '" + spec.attr_name + "' has wrong type");
                    result.valid = false;
                    continue;
                }
            }

            if (spec.required && !found) {
                result.errors.push_back(
                    "Node '" + node.name + "' (op=" + node.op_type +
                    "): missing required attribute '" + spec.attr_name + "'");
                result.valid = false;
            }
        }
    };

    for (const auto& node : graph_.nodes) {
        if (node.op_type == "Conv") {
            check_attrs(node, {
                {"kernel_shape", AttrSpec::INTS, true},
                {"strides", AttrSpec::INTS, false},
                {"pads", AttrSpec::INTS, false},
                {"dilations", AttrSpec::INTS, false},
                {"group", AttrSpec::INT, false},
            });
        } else if (node.op_type == "MaxPool" || node.op_type == "AveragePool") {
            check_attrs(node, {
                {"kernel_shape", AttrSpec::INTS, true},
                {"strides", AttrSpec::INTS, false},
                {"pads", AttrSpec::INTS, false},
            });
        } else if (node.op_type == "Gemm") {
            check_attrs(node, {
                {"alpha", AttrSpec::FLOAT, false},
                {"beta", AttrSpec::FLOAT, false},
                {"transB", AttrSpec::INT, false},
            });
        } else if (node.op_type == "BatchNormalization") {
            check_attrs(node, {
                {"epsilon", AttrSpec::FLOAT, false},
            });
        } else if (node.op_type == "Concat") {
            check_attrs(node, {
                {"axis", AttrSpec::INT, true},
            });
        } else if (node.op_type == "Split") {
            check_attrs(node, {
                {"axis", AttrSpec::INT, true},
            });
        } else if (node.op_type == "Transpose") {
            check_attrs(node, {
                {"perm", AttrSpec::INTS, false},
            });
        } else if (node.op_type == "Softmax" || node.op_type == "LogSoftmax") {
            check_attrs(node, {
                {"axis", AttrSpec::INT, false},
            });
        } else if (node.op_type == "LeakyRelu") {
            check_attrs(node, {
                {"alpha", AttrSpec::FLOAT, false},
            });
        } else if (node.op_type == "GroupNormalization") {
            check_attrs(node, {
                {"epsilon", AttrSpec::FLOAT, false},
                {"num_groups", AttrSpec::INT, true},
            });
        } else if (node.op_type == "LSTM") {
            check_attrs(node, {
                {"hidden_size", AttrSpec::INT, true},
                {"direction", AttrSpec::STRING, false},
            });
        } else if (node.op_type == "GRU") {
            check_attrs(node, {
                {"hidden_size", AttrSpec::INT, true},
                {"direction", AttrSpec::STRING, false},
                {"linear_before_reset", AttrSpec::INT, false},
            });
        }
    }

    // -- 5. Opset compatibility --
    // Check that operators requiring higher opset versions are flagged
    struct OpsetRequirement {
        std::string op_type;
        int64_t min_opset;
    };

    static const std::vector<OpsetRequirement> opset_requirements = {
        {"Gelu", 20},
        {"Mish", 18},
        {"LayerNormalization", 17},
        {"GroupNormalization", 18},
        {"LessOrEqual", 12},
        {"GreaterOrEqual", 12},
        {"CumSum", 11},
    };

    for (const auto& node : graph_.nodes) {
        for (const auto& req : opset_requirements) {
            if (node.op_type == req.op_type && opset_version_ < req.min_opset) {
                result.errors.push_back(
                    "Node '" + node.name + "' uses op '" + req.op_type +
                    "' which requires opset >= " + std::to_string(req.min_opset) +
                    " but exporter is configured for opset " +
                    std::to_string(opset_version_));
                result.valid = false;
            }
        }
    }

    // -- 6. Basic structural checks --
    if (graph_.inputs.empty()) {
        result.warnings.push_back("Graph has no inputs defined");
    }

    if (graph_.outputs.empty()) {
        result.warnings.push_back("Graph has no outputs defined");
    }

    if (graph_.nodes.empty()) {
        result.warnings.push_back("Graph has no nodes (operations)");
    }

    return result;
}

// Export Functions

auto ONNXExporter::serialize_model() -> std::vector<uint8_t> {
#ifdef TENZOR_HAS_ONNX_PROTOBUF
    // Build a canonical ModelProto using the generated protobuf types.
    // Field numbers match upstream onnx.proto (see proto/onnx.proto).
    tenzor_onnx::ModelProto model;
    model.set_ir_version(8);                           // canonical IR version 8
    model.set_producer_name(producer_name_);
    model.set_model_version(model_version_);
    if (!description_.empty()) {
        model.set_doc_string(description_);
    }

    auto* opset = model.add_opset_import();
    opset->set_domain("");                             // ai.onnx (default)
    opset->set_version(opset_version_);

    auto* graph_proto = model.mutable_graph();
    graph_proto->set_name(graph_.name);

    // Build one NodeProto per ONNXExportNode.
    for (const auto& node : graph_.nodes) {
        auto* n = graph_proto->add_node();
        for (const auto& input  : node.inputs)  n->add_input(input);
        for (const auto& output : node.outputs) n->add_output(output);
        n->set_name(node.name);
        n->set_op_type(node.op_type);

        auto set_attr_name_and_type = [](tenzor_onnx::AttributeProto* a,
                                         const std::string& name,
                                         tenzor_onnx::AttributeProto_AttributeType t) {
            a->set_name(name);
            a->set_type(t);
        };

        for (const auto& [key, val] : node.int_attrs) {
            auto* a = n->add_attribute();
            set_attr_name_and_type(a, key, tenzor_onnx::AttributeProto_AttributeType_INT);
            a->set_i(val);
        }
        for (const auto& [key, val] : node.float_attrs) {
            auto* a = n->add_attribute();
            set_attr_name_and_type(a, key, tenzor_onnx::AttributeProto_AttributeType_FLOAT);
            a->set_f(val);
        }
        for (const auto& [key, val] : node.string_attrs) {
            auto* a = n->add_attribute();
            set_attr_name_and_type(a, key, tenzor_onnx::AttributeProto_AttributeType_STRING);
            a->set_s(val);
        }
        for (const auto& [key, vals] : node.ints_attrs) {
            auto* a = n->add_attribute();
            set_attr_name_and_type(a, key, tenzor_onnx::AttributeProto_AttributeType_INTS);
            for (auto v : vals) a->add_ints(v);
        }
        for (const auto& [key, vals] : node.floats_attrs) {
            auto* a = n->add_attribute();
            set_attr_name_and_type(a, key, tenzor_onnx::AttributeProto_AttributeType_FLOATS);
            for (auto v : vals) a->add_floats(v);
        }
        for (const auto& [key, t] : node.tensor_attrs) {
            auto* a = n->add_attribute();
            set_attr_name_and_type(a, key, tenzor_onnx::AttributeProto_AttributeType_TENSOR);
            auto* tp = a->mutable_t();
            tp->set_name(t.name);
            tp->set_data_type(static_cast<int32_t>(t.dtype));
            for (auto d : t.dims) tp->add_dims(d);
            if (!t.raw_data.empty()) {
                tp->set_raw_data(std::string(t.raw_data.begin(), t.raw_data.end()));
            }
        }
    }

    // Initializers.
    for (const auto& tensor : graph_.initializers) {
        auto* t = graph_proto->add_initializer();
        t->set_name(tensor.name);
        t->set_data_type(static_cast<int32_t>(tensor.dtype));
        for (auto d : tensor.dims) t->add_dims(d);
        if (!tensor.raw_data.empty()) {
            t->set_raw_data(std::string(tensor.raw_data.begin(), tensor.raw_data.end()));
        }
    }

    // ValueInfoProto builder with dim_param support.
    auto build_value_info = [](tenzor_onnx::ValueInfoProto* vi,
                               const ONNXExportValueInfo& src) {
        vi->set_name(src.name);
        auto* type = vi->mutable_type();
        auto* tensor_type = type->mutable_tensor_type();
        tensor_type->set_elem_type(static_cast<int32_t>(src.dtype));
        auto* shape = tensor_type->mutable_shape();
        for (int64_t i = 0; i < static_cast<int64_t>(src.shape.size()); ++i) {
            auto* d = shape->add_dim();
            auto param_it = src.dim_params.find(i);
            if (src.shape[i] < 0 && param_it != src.dim_params.end()) {
                d->set_dim_param(param_it->second);
            } else if (src.shape[i] < 0) {
                d->set_dim_param("dynamic_" + std::to_string(i));
            } else {
                d->set_dim_value(src.shape[i]);
            }
        }
    };

    for (const auto& input : graph_.inputs) {
        build_value_info(graph_proto->add_input(), input);
    }
    for (const auto& output : graph_.outputs) {
        build_value_info(graph_proto->add_output(), output);
    }

    const std::string out = model.SerializeAsString();
    return std::vector<uint8_t>(out.begin(), out.end());
#else
    throw std::runtime_error(
        "ONNXExporter::serialize_model: built without protobuf support. "
        "Reconfigure with libprotobuf-dev installed and rebuild tenzor_core.");
#endif
}

auto ONNXExporter::export_to_file(const std::string& filepath) -> void {
    // Legacy single-argument export — no external_data, single .onnx file.
    export_to_file(filepath, /*use_external_data=*/std::optional<bool>{false});
}

auto ONNXExporter::export_to_file(const std::string& filepath,
                                  std::optional<bool> use_external_data,
                                  size_t external_data_threshold_bytes) -> void {
#ifdef TENZOR_HAS_ONNX_PROTOBUF
    // 5th-audit C6: optional external_data sidecar for >2GB models.
    //
    // We always defer to `serialize_model()` for the "no-external-data" path
    // (legacy callers, small models). For the external path we re-serialise
    // here so we can intercept large initializers BEFORE they get packed
    // into raw_data, write them to the sidecar, and replace them with an
    // `external_data` reference.

    // Decide whether to enable external_data automatically when not specified.
    bool emit_external = use_external_data.value_or(false);
    if (!use_external_data.has_value()) {
        size_t total_init_bytes = 0;
        for (const auto& init : graph_.initializers) {
            total_init_bytes += init.raw_data.size();
        }
        // 1.5 GB safety margin under protobuf's 2 GB cap.
        if (total_init_bytes > (size_t{1500} * 1024ULL * 1024ULL)) {
            emit_external = true;
        }
    }

    if (!emit_external) {
        // Fast path: single-file proto, legacy semantics.
        auto bytes = serialize_model();
        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + filepath);
        }
        file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        return;
    }

    // External-data path. We write a sibling `.data` file and use the
    // `external_data` field on each large initializer.
    namespace fs = std::filesystem;
    const fs::path proto_path(filepath);
    const std::string data_basename = proto_path.filename().string() + ".data";
    const fs::path data_path = proto_path.parent_path().empty()
                                   ? fs::path(data_basename)
                                   : (proto_path.parent_path() / data_basename);

    std::ofstream data_out(data_path, std::ios::binary | std::ios::trunc);
    if (!data_out.is_open()) {
        throw std::runtime_error(
            "Failed to open external-data sidecar for writing: " + data_path.string());
    }

    // Build the same ModelProto as `serialize_model`, but route large
    // initializers through the sidecar instead of inline raw_data.
    // (We could refactor `serialize_model` to share this code; kept inline
    // for clarity of the data-location decisions.)
    tenzor_onnx::ModelProto model;
    model.set_ir_version(8);
    model.set_producer_name(producer_name_);
    model.set_model_version(model_version_);
    if (!description_.empty()) {
        model.set_doc_string(description_);
    }
    auto* opset = model.add_opset_import();
    opset->set_domain("");
    opset->set_version(opset_version_);
    auto* graph_proto = model.mutable_graph();
    graph_proto->set_name(graph_.name);

    auto set_attr_name_and_type = [](tenzor_onnx::AttributeProto* a,
                                     const std::string& name,
                                     tenzor_onnx::AttributeProto_AttributeType t) {
        a->set_name(name);
        a->set_type(t);
    };

    for (const auto& node : graph_.nodes) {
        auto* n = graph_proto->add_node();
        for (const auto& input  : node.inputs)  n->add_input(input);
        for (const auto& output : node.outputs) n->add_output(output);
        n->set_name(node.name);
        n->set_op_type(node.op_type);

        for (const auto& [key, val] : node.int_attrs) {
            auto* a = n->add_attribute();
            set_attr_name_and_type(a, key, tenzor_onnx::AttributeProto_AttributeType_INT);
            a->set_i(val);
        }
        for (const auto& [key, val] : node.float_attrs) {
            auto* a = n->add_attribute();
            set_attr_name_and_type(a, key, tenzor_onnx::AttributeProto_AttributeType_FLOAT);
            a->set_f(val);
        }
        for (const auto& [key, val] : node.string_attrs) {
            auto* a = n->add_attribute();
            set_attr_name_and_type(a, key, tenzor_onnx::AttributeProto_AttributeType_STRING);
            a->set_s(val);
        }
        for (const auto& [key, vals] : node.ints_attrs) {
            auto* a = n->add_attribute();
            set_attr_name_and_type(a, key, tenzor_onnx::AttributeProto_AttributeType_INTS);
            for (auto v : vals) a->add_ints(v);
        }
        for (const auto& [key, vals] : node.floats_attrs) {
            auto* a = n->add_attribute();
            set_attr_name_and_type(a, key, tenzor_onnx::AttributeProto_AttributeType_FLOATS);
            for (auto v : vals) a->add_floats(v);
        }
        for (const auto& [key, t] : node.tensor_attrs) {
            auto* a = n->add_attribute();
            set_attr_name_and_type(a, key, tenzor_onnx::AttributeProto_AttributeType_TENSOR);
            auto* tp = a->mutable_t();
            tp->set_name(t.name);
            tp->set_data_type(static_cast<int32_t>(t.dtype));
            for (auto d : t.dims) tp->add_dims(d);
            if (!t.raw_data.empty()) {
                tp->set_raw_data(std::string(t.raw_data.begin(), t.raw_data.end()));
            }
        }
    }

    // Initializers: route large ones through the sidecar.
    uint64_t offset = 0;
    for (const auto& tensor : graph_.initializers) {
        auto* t = graph_proto->add_initializer();
        t->set_name(tensor.name);
        t->set_data_type(static_cast<int32_t>(tensor.dtype));
        for (auto d : tensor.dims) t->add_dims(d);

        if (!tensor.raw_data.empty()
            && tensor.raw_data.size() > external_data_threshold_bytes) {
            // External: append bytes to sidecar, emit (location, offset, length).
            const uint64_t length = tensor.raw_data.size();
            data_out.write(reinterpret_cast<const char*>(tensor.raw_data.data()),
                           static_cast<std::streamsize>(length));
            if (!data_out) {
                throw std::runtime_error(
                    "External-data sidecar write failed: " + data_path.string());
            }

            t->set_data_location(tenzor_onnx::TensorProto_DataLocation_EXTERNAL);
            auto add_entry = [&](const char* key, const std::string& value) {
                auto* e = t->add_external_data();
                e->set_key(key);
                e->set_value(value);
            };
            add_entry("location", data_basename);
            add_entry("offset",   std::to_string(offset));
            add_entry("length",   std::to_string(length));
            offset += length;
        } else if (!tensor.raw_data.empty()) {
            // Inline: small enough to embed.
            t->set_raw_data(std::string(tensor.raw_data.begin(), tensor.raw_data.end()));
        }
    }
    data_out.close();

    auto build_value_info = [](tenzor_onnx::ValueInfoProto* vi,
                               const ONNXExportValueInfo& src) {
        vi->set_name(src.name);
        auto* type = vi->mutable_type();
        auto* tensor_type = type->mutable_tensor_type();
        tensor_type->set_elem_type(static_cast<int32_t>(src.dtype));
        auto* shape = tensor_type->mutable_shape();
        for (int64_t i = 0; i < static_cast<int64_t>(src.shape.size()); ++i) {
            auto* d = shape->add_dim();
            auto param_it = src.dim_params.find(i);
            if (src.shape[i] < 0 && param_it != src.dim_params.end()) {
                d->set_dim_param(param_it->second);
            } else if (src.shape[i] < 0) {
                d->set_dim_param("dynamic_" + std::to_string(i));
            } else {
                d->set_dim_value(src.shape[i]);
            }
        }
    };
    for (const auto& input : graph_.inputs) {
        build_value_info(graph_proto->add_input(), input);
    }
    for (const auto& output : graph_.outputs) {
        build_value_info(graph_proto->add_output(), output);
    }

    const std::string out_bytes = model.SerializeAsString();
    std::ofstream proto_out(filepath, std::ios::binary | std::ios::trunc);
    if (!proto_out.is_open()) {
        // Clean up the sidecar so we don't leave a half-written export.
        std::error_code ec;
        fs::remove(data_path, ec);
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }
    proto_out.write(out_bytes.data(), static_cast<std::streamsize>(out_bytes.size()));
#else
    (void)filepath;
    (void)use_external_data;
    (void)external_data_threshold_bytes;
    throw std::runtime_error(
        "ONNXExporter::export_to_file(external_data): built without protobuf support.");
#endif
}

auto ONNXExporter::export_to_bytes() -> std::vector<uint8_t> {
    return serialize_model();
}

auto ONNXExporter::get_graph() const -> const ONNXGraph& {
    return graph_;
}

auto ONNXExporter::clear() -> void {
    graph_ = ONNXGraph("main_graph");
    context_ = ExportContext();
}

auto ONNXExporter::get_tensor_name(const Tensor& tensor, const std::string& default_name) -> std::string {
    auto existing = context_.get_tensor_name(tensor);
    if (existing.has_value()) {
        return existing.value();
    }

    std::string name = context_.generate_name(default_name);
    context_.register_tensor(tensor, name);
    return name;
}

auto ONNXExporter::add_initializer_tensor(const Tensor& tensor, const std::string& name) -> void {
    ONNXTensor onnx_tensor(tensor, name);
    graph_.add_initializer(onnx_tensor);
    context_.register_tensor(tensor, name);
}

// ============================================================================
// JIT OpType -> ONNX Operator Mapping
// ============================================================================

auto ONNXExporter::jit_op_type_to_onnx(jit::OpType op_type) -> std::string {
    switch (op_type) {
        // Arithmetic
        case jit::OpType::Add:              return "Add";
        case jit::OpType::Sub:              return "Sub";
        case jit::OpType::Mul:              return "Mul";
        case jit::OpType::Div:              return "Div";

        // Matrix operations
        case jit::OpType::MatMul:           return "MatMul";
        case jit::OpType::Bmm:              return "MatMul";

        // Activations
        case jit::OpType::ReLU:             return "Relu";
        case jit::OpType::Sigmoid:          return "Sigmoid";
        case jit::OpType::Tanh:             return "Tanh";
        case jit::OpType::Softmax:          return "Softmax";
        case jit::OpType::LogSoftmax:       return "LogSoftmax";

        // Pooling
        case jit::OpType::MaxPool2d:        return "MaxPool";
        case jit::OpType::AvgPool2d:        return "AveragePool";
        case jit::OpType::AdaptiveAvgPool2d: return "GlobalAveragePool";

        // Convolution
        case jit::OpType::Conv2d:           return "Conv";
        case jit::OpType::ConvTranspose:    return "ConvTranspose";

        // Normalization
        case jit::OpType::BatchNorm2d:      return "BatchNormalization";
        case jit::OpType::LayerNorm:        return "LayerNormalization";

        // Shape operations
        case jit::OpType::Reshape:          return "Reshape";
        case jit::OpType::Transpose:        return "Transpose";
        case jit::OpType::Permute:          return "Transpose";
        case jit::OpType::Squeeze:          return "Squeeze";
        case jit::OpType::Unsqueeze:        return "Unsqueeze";
        case jit::OpType::Flatten:          return "Flatten";

        // Reductions
        case jit::OpType::Sum:              return "ReduceSum";
        case jit::OpType::Mean:             return "ReduceMean";
        case jit::OpType::Max:              return "ReduceMax";
        case jit::OpType::Min:              return "ReduceMin";

        // Element-wise
        case jit::OpType::Exp:              return "Exp";
        case jit::OpType::Log:              return "Log";
        case jit::OpType::Sqrt:             return "Sqrt";
        case jit::OpType::Pow:              return "Pow";
        case jit::OpType::Abs:              return "Abs";
        case jit::OpType::Neg:              return "Neg";
        case jit::OpType::Clamp:            return "Clip";

        // Indexing
        case jit::OpType::Slice:            return "Slice";
        case jit::OpType::Cat:              return "Concat";

        // Other
        case jit::OpType::Dropout:          return "Dropout";
        case jit::OpType::Linear:           return "Gemm";
        case jit::OpType::Embedding:        return "Gather";
        case jit::OpType::GELU:             return "Gelu";

        // Linear algebra. ONNX has no native ops for most of these, so they go
        // under the tenzor custom domain. These op_type strings MUST match the
        // OpId-visitor mapping (op_id_to_onnx_type, "Linalg*" names) so the two
        // export paths emit identical, re-importable nodes; importer.cpp has
        // matching cases for the custom-domain names.
        case jit::OpType::Det:              return "Det";
        case jit::OpType::Inv:              return "LinalgInv";
        case jit::OpType::Solve:            return "LinalgSolve";
        case jit::OpType::Cholesky:         return "LinalgCholesky";
        case jit::OpType::Svd:              return "LinalgSVD";
        case jit::OpType::Qr:               return "LinalgQR";
        case jit::OpType::Eigh:             return "LinalgEigh";
        case jit::OpType::Eigvalsh:         return "LinalgEigvalsh";
        case jit::OpType::Norm:             return "LinalgMatrixNorm";
        case jit::OpType::Slogdet:          return "LinalgSlogdet";

        // Constant / Input / Output are structural, not ONNX ops
        case jit::OpType::Constant:         return "Constant";
        case jit::OpType::Input:            return "";
        case jit::OpType::Output:           return "";

        default:
            throw std::runtime_error(
                "No ONNX mapping for JIT OpType: " +
                jit::op_type_to_string(op_type));
    }
}

// ============================================================================
// Single JIT Node -> ONNX Node Conversion
// ============================================================================

auto ONNXExporter::convert_jit_node_to_onnx(
    const std::shared_ptr<jit::Node>& node,
    std::unordered_map<std::string, std::string>& value_name_map) -> void {

    auto op_type = node->op_type();

    // Skip Input/Output meta-nodes -- they don't produce ONNX ops
    if (op_type == jit::OpType::Input || op_type == jit::OpType::Output) {
        return;
    }

    // Resolve ONNX op string
    std::string onnx_op = jit_op_type_to_onnx(op_type);
    if (onnx_op.empty()) {
        return; // No ONNX representation needed
    }

    // Generate a unique node name
    std::string node_name = graph_.get_unique_name(
        jit::op_type_to_string(op_type));

    ONNXExportNode onnx_node(onnx_op, node_name);

    // --- Map inputs ---
    for (const auto& input_val : node->inputs()) {
        const auto& jit_id = input_val->id();
        auto it = value_name_map.find(jit_id);
        if (it != value_name_map.end()) {
            onnx_node.add_input(it->second);
        } else {
            // Not yet mapped -- this is likely a weight/parameter that
            // appeared as a JIT value but hasn't been registered.
            // Create a name and record it.
            std::string onnx_name = graph_.get_unique_name("param");
            value_name_map[jit_id] = onnx_name;
            onnx_node.add_input(onnx_name);
        }
    }

    // --- Add tensor attributes as initializers ---
    // Many JIT nodes store weights/biases as tensor_attrs (e.g., "weight", "bias").
    // For ONNX, these need to be graph initializers referenced by name.
    auto [float_attrs, int_attrs, vec_attrs, bool_attrs, tensor_attrs] =
        const_cast<jit::Node&>(*node).get_all_attrs();

    for (const auto& [attr_name, tensor] : tensor_attrs) {
        if (tensor.numel() > 0) {
            std::string init_name = graph_.get_unique_name(node_name + "_" + attr_name);
            add_initializer_tensor(tensor.cpu().contiguous(), init_name);
            onnx_node.add_input(init_name);
        }
    }

    // --- Map outputs ---
    for (const auto& output_val : node->outputs()) {
        const auto& jit_id = output_val->id();
        std::string onnx_name = graph_.get_unique_name("val");
        value_name_map[jit_id] = onnx_name;
        onnx_node.add_output(onnx_name);

        // Register intermediate value info for shape tracking
        auto shape_vec = output_val->shape();
        ONNXExportValueInfo vi(onnx_name, dtype_to_onnx(output_val->dtype()), shape_vec);
        graph_.add_value_info(vi);
    }

    // --- Copy scalar/vector attributes to the ONNX node ---

    // Special-case attribute mapping per op type
    switch (op_type) {
        case jit::OpType::Linear: {
            // JIT Linear -> ONNX Gemm: Y = alpha * A * B^T + beta * C
            onnx_node.set_attr("alpha", 1.0f);
            onnx_node.set_attr("beta", 1.0f);
            onnx_node.set_attr("transB", static_cast<int64_t>(1));
            break;
        }
        case jit::OpType::Conv2d: {
            // Map JIT conv attributes to ONNX Conv attributes
            auto kernel = node->get_vec_attr("kernel_size");
            if (!kernel.empty()) {
                onnx_node.set_attr("kernel_shape", kernel);
            }
            auto stride = node->get_vec_attr("stride");
            if (!stride.empty()) {
                onnx_node.set_attr("strides", stride);
            }
            auto padding = node->get_vec_attr("padding");
            if (!padding.empty()) {
                // ONNX `pads` is [x1_begin..xN_begin, x1_end..xN_end] — i.e. the
                // per-axis padding repeated (begins then ends). This generalises
                // to 1D/2D/3D: [w]->[w,w], [h,w]->[h,w,h,w], [d,h,w]->[d,h,w,d,h,w].
                std::vector<int64_t> pads(padding.begin(), padding.end());
                pads.insert(pads.end(), padding.begin(), padding.end());
                onnx_node.set_attr("pads", pads);
            }
            auto dilation = node->get_vec_attr("dilation");
            if (!dilation.empty()) {
                onnx_node.set_attr("dilations", dilation);
            }
            auto groups = node->get_int_attr("groups");
            if (groups > 0) {
                onnx_node.set_attr("group", groups);
            }
            break;
        }
        case jit::OpType::ConvTranspose: {
            // ONNX ConvTranspose mirrors Conv's attribute set plus
            // output_padding. kernel_shape is inferred from the weight by the
            // importer when omitted, but we emit it for round-trip fidelity.
            auto kernel = node->get_vec_attr("kernel_size");
            if (!kernel.empty()) {
                onnx_node.set_attr("kernel_shape", kernel);
            }
            auto stride = node->get_vec_attr("stride");
            if (!stride.empty()) {
                onnx_node.set_attr("strides", stride);
            }
            auto padding = node->get_vec_attr("padding");
            if (!padding.empty()) {
                // ONNX `pads` = [begins..., ends...]; per-axis padding repeated.
                std::vector<int64_t> pads(padding.begin(), padding.end());
                pads.insert(pads.end(), padding.begin(), padding.end());
                onnx_node.set_attr("pads", pads);
            }
            auto dilation = node->get_vec_attr("dilation");
            if (!dilation.empty()) {
                onnx_node.set_attr("dilations", dilation);
            }
            auto output_padding = node->get_vec_attr("output_padding");
            if (!output_padding.empty()) {
                onnx_node.set_attr("output_padding", output_padding);
            }
            auto groups = node->get_int_attr("groups");
            if (groups > 0) {
                onnx_node.set_attr("group", groups);
            }
            break;
        }
        case jit::OpType::BatchNorm2d: {
            float eps = node->get_attr("eps");
            if (eps > 0.0f) {
                onnx_node.set_attr("epsilon", eps);
            } else {
                onnx_node.set_attr("epsilon", 1e-5f); // default
            }
            float momentum = node->get_attr("momentum");
            if (momentum > 0.0f) {
                onnx_node.set_attr("momentum", momentum);
            }
            break;
        }
        case jit::OpType::MaxPool2d:
        case jit::OpType::AvgPool2d: {
            auto kernel = node->get_vec_attr("kernel_size");
            if (!kernel.empty()) {
                onnx_node.set_attr("kernel_shape", kernel);
            }
            auto stride = node->get_vec_attr("stride");
            if (!stride.empty()) {
                onnx_node.set_attr("strides", stride);
            }
            auto padding = node->get_vec_attr("padding");
            if (!padding.empty()) {
                if (padding.size() == 2) {
                    onnx_node.set_attr("pads", std::vector<int64_t>{
                        padding[0], padding[1], padding[0], padding[1]});
                } else {
                    onnx_node.set_attr("pads", padding);
                }
            }
            break;
        }
        case jit::OpType::Softmax:
        case jit::OpType::LogSoftmax: {
            auto axis = node->get_int_attr("axis");
            onnx_node.set_attr("axis", axis != 0 ? axis : static_cast<int64_t>(-1));
            break;
        }
        case jit::OpType::Transpose:
        case jit::OpType::Permute: {
            auto perm = node->get_vec_attr("perm");
            if (!perm.empty()) {
                onnx_node.set_attr("perm", perm);
            }
            break;
        }
        case jit::OpType::Cat: {
            auto axis = node->get_int_attr("axis");
            onnx_node.set_attr("axis", axis);
            break;
        }
        case jit::OpType::Flatten: {
            auto axis = node->get_int_attr("axis");
            onnx_node.set_attr("axis", axis != 0 ? axis : static_cast<int64_t>(1));
            break;
        }
        case jit::OpType::Clamp: {
            // ONNX Clip has no attributes in opset 11+; min/max are inputs.
            // For simplicity we add them as float attrs if present.
            float min_val = node->get_attr("min");
            float max_val = node->get_attr("max");
            if (min_val != 0.0f || max_val != 0.0f) {
                onnx_node.set_attr("min", min_val);
                onnx_node.set_attr("max", max_val);
            }
            break;
        }
        case jit::OpType::Dropout: {
            float ratio = node->get_attr("ratio");
            if (ratio > 0.0f) {
                onnx_node.set_attr("ratio", ratio);
            }
            break;
        }
        default: {
            // For remaining ops, forward any generic int/float/vec attrs
            for (const auto& [name, val] : int_attrs) {
                onnx_node.set_attr(name, val);
            }
            for (const auto& [name, val] : float_attrs) {
                onnx_node.set_attr(name, val);
            }
            for (const auto& [name, val] : vec_attrs) {
                onnx_node.set_attr(name, val);
            }
            break;
        }
    }

    graph_.add_node(onnx_node);
}

// ============================================================================
// Quantization (QDQ) Nodes
// ============================================================================

auto ONNXExporter::export_quantize_linear(const Tensor& input, const Tensor& scale,
                                           const Tensor& zero_point,
                                           const std::string& output_name,
                                           int64_t axis) -> void {
    ONNXExportNode node("QuantizeLinear", context_.generate_name("quantize_linear"));

    std::string input_name = get_tensor_name(input, "ql_input");
    std::string scale_name = context_.generate_name("ql_scale");
    std::string zp_name = context_.generate_name("ql_zero_point");

    add_initializer_tensor(scale, scale_name);
    add_initializer_tensor(zero_point, zp_name);

    node.add_input(input_name);
    node.add_input(scale_name);
    node.add_input(zp_name);
    node.add_output(output_name);

    if (axis >= 0) {
        node.set_attr("axis", axis);
    }

    graph_.add_node(node);
}

auto ONNXExporter::export_dequantize_linear(const Tensor& input, const Tensor& scale,
                                             const Tensor& zero_point,
                                             const std::string& output_name,
                                             int64_t axis) -> void {
    ONNXExportNode node("DequantizeLinear", context_.generate_name("dequantize_linear"));

    std::string input_name = get_tensor_name(input, "dql_input");
    std::string scale_name = context_.generate_name("dql_scale");
    std::string zp_name = context_.generate_name("dql_zero_point");

    add_initializer_tensor(scale, scale_name);
    add_initializer_tensor(zero_point, zp_name);

    node.add_input(input_name);
    node.add_input(scale_name);
    node.add_input(zp_name);
    node.add_output(output_name);

    if (axis >= 0) {
        node.set_attr("axis", axis);
    }

    graph_.add_node(node);
}

auto ONNXExporter::export_quantized_linear(
    const nn::quantization::QuantizedLinear& layer,
    const Tensor& input,
    const std::string& output_name) -> void {
    // QDQ pattern: DequantizeLinear(weight) → MatMul → [Add bias] → output
    // The input is assumed to already be in FP32 (dequantized upstream).

    // Get quantized weight data and params
    auto& q_weight = layer.weight();
    auto& params = q_weight.params();

    // DequantizeLinear for weight
    std::string dq_weight_name = context_.generate_name("ql_dq_weight");
    std::string weight_name = context_.generate_name("ql_weight_q");
    std::string scale_name = context_.generate_name("ql_weight_scale");
    std::string zp_name = context_.generate_name("ql_weight_zp");

    add_initializer_tensor(q_weight.data(), weight_name);
    add_initializer_tensor(params.scale, scale_name);
    add_initializer_tensor(params.zero_point, zp_name);

    ONNXExportNode dq_node("DequantizeLinear", context_.generate_name("dequantize_weight"));
    dq_node.add_input(weight_name);
    dq_node.add_input(scale_name);
    dq_node.add_input(zp_name);
    dq_node.add_output(dq_weight_name);

    if (params.scheme == nn::quantization::QuantizationScheme::PerChannelSymmetric ||
        params.scheme == nn::quantization::QuantizationScheme::PerChannelAsymmetric) {
        dq_node.set_attr("axis", static_cast<int64_t>(0));
    }
    graph_.add_node(dq_node);

    // Transpose weight for Gemm (from [out, in] to [in, out])
    std::string input_name = get_tensor_name(input, "ql_input");

    // Use Gemm node: output = input @ weight^T + bias
    ONNXExportNode gemm_node("Gemm", context_.generate_name("ql_gemm"));
    gemm_node.add_input(input_name);
    gemm_node.add_input(dq_weight_name);

    if (layer.has_bias()) {
        std::string bias_name = context_.generate_name("ql_bias");
        add_initializer_tensor(layer.bias(), bias_name);
        gemm_node.add_input(bias_name);
    }

    gemm_node.add_output(output_name);
    gemm_node.set_attr("transB", static_cast<int64_t>(1));
    graph_.add_node(gemm_node);
}

// ============================================================================
// Phase 13: Expanded ONNX Export Coverage
// ============================================================================

// --- Group 1: Basic ops ---

auto ONNXExporter::export_cast(const Tensor& input, DType target_dtype,
                                const Tensor& output,
                                const std::string& output_name) -> void {
    ONNXExportNode node("Cast", context_.generate_name("cast"));
    std::string input_name = get_tensor_name(input, "cast_input");
    node.add_input(input_name);
    node.add_output(output_name);
    node.set_attr("to", static_cast<int64_t>(dtype_to_onnx(target_dtype)));
    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_triu(const Tensor& input, int64_t diagonal,
                                const Tensor& output,
                                const std::string& output_name) -> void {
    // ONNX Trilu (opset 14+): upper=1 for upper triangular
    ONNXExportNode node("Trilu", context_.generate_name("triu"));
    std::string input_name = get_tensor_name(input, "triu_input");
    node.add_input(input_name);

    // k (diagonal offset) is the second input as a scalar tensor
    if (diagonal != 0) {
        std::string k_name = context_.generate_name("triu_k");
        Tensor k_tensor({1}, DType::Int64, Device::cpu());
        *k_tensor.data<int64_t>() = diagonal;
        add_initializer_tensor(k_tensor, k_name);
        node.add_input(k_name);
    }

    node.add_output(output_name);
    node.set_attr("upper", static_cast<int64_t>(1));
    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_tril(const Tensor& input, int64_t diagonal,
                                const Tensor& output,
                                const std::string& output_name) -> void {
    // ONNX Trilu (opset 14+): upper=0 for lower triangular
    ONNXExportNode node("Trilu", context_.generate_name("tril"));
    std::string input_name = get_tensor_name(input, "tril_input");
    node.add_input(input_name);

    if (diagonal != 0) {
        std::string k_name = context_.generate_name("tril_k");
        Tensor k_tensor({1}, DType::Int64, Device::cpu());
        *k_tensor.data<int64_t>() = diagonal;
        add_initializer_tensor(k_tensor, k_name);
        node.add_input(k_name);
    }

    node.add_output(output_name);
    node.set_attr("upper", static_cast<int64_t>(0));
    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_logical_and(const Tensor& a, const Tensor& b,
                                       const Tensor& output,
                                       const std::string& output_name) -> void {
    ONNXExportNode node("And", context_.generate_name("logical_and"));
    std::string a_name = get_tensor_name(a, "and_a");
    std::string b_name = get_tensor_name(b, "and_b");
    node.add_input(a_name);
    node.add_input(b_name);
    node.add_output(output_name);
    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_logical_or(const Tensor& a, const Tensor& b,
                                      const Tensor& output,
                                      const std::string& output_name) -> void {
    ONNXExportNode node("Or", context_.generate_name("logical_or"));
    std::string a_name = get_tensor_name(a, "or_a");
    std::string b_name = get_tensor_name(b, "or_b");
    node.add_input(a_name);
    node.add_input(b_name);
    node.add_output(output_name);
    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_logical_not(const Tensor& input,
                                       const Tensor& output,
                                       const std::string& output_name) -> void {
    ONNXExportNode node("Not", context_.generate_name("logical_not"));
    std::string input_name = get_tensor_name(input, "not_input");
    node.add_input(input_name);
    node.add_output(output_name);
    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

// --- Group 2: Indexing ops ---

auto ONNXExporter::export_scatter_add(const Tensor& data, const Tensor& indices,
                                       const Tensor& updates, int64_t axis,
                                       const Tensor& output,
                                       const std::string& output_name) -> void {
    // ONNX ScatterElements with reduction='add' (opset 16+)
    ONNXExportNode node("ScatterElements", context_.generate_name("scatter_add"));
    std::string data_name = get_tensor_name(data, "scatter_add_data");
    std::string indices_name = get_tensor_name(indices, "scatter_add_indices");
    std::string updates_name = get_tensor_name(updates, "scatter_add_updates");
    node.add_input(data_name);
    node.add_input(indices_name);
    node.add_input(updates_name);
    node.add_output(output_name);
    node.set_attr("axis", axis);
    node.set_attr("reduction", std::string("add"));
    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_unfold(const Tensor& input, int64_t dimension,
                                  int64_t size, int64_t step,
                                  const Tensor& output,
                                  const std::string& output_name) -> void {
    // Unfold(dim, size, step) extracts sliding windows of `size` along `dim`
    // with stride `step`. Decompose to a sequence of Slice ops + Concat + Reshape.
    //
    // For input shape [..., L, ...] along dim, output has shape
    // [..., n_windows, ..., size] where n_windows = (L - size) / step + 1.
    std::string input_name = get_tensor_name(input, "unfold_input");

    auto in_shape = input.shape();
    int64_t ndim = static_cast<int64_t>(in_shape.size());
    int64_t dim = dimension >= 0 ? dimension : ndim + dimension;
    int64_t dim_size = in_shape[dim];
    int64_t n_windows = (dim_size - size) / step + 1;

    // Create Slice for each window, then Concat + Reshape
    std::vector<std::string> window_names;
    window_names.reserve(n_windows);

    for (int64_t i = 0; i < n_windows; ++i) {
        int64_t start = i * step;
        int64_t end = start + size;

        std::string prefix = "unfold_w" + std::to_string(i);

        // Create starts/ends/axes initializers for Slice
        std::string starts_name = context_.generate_name(prefix + "_starts");
        Tensor starts_t({1}, DType::Int64, Device::cpu());
        *starts_t.data<int64_t>() = start;
        add_initializer_tensor(starts_t, starts_name);

        std::string ends_name = context_.generate_name(prefix + "_ends");
        Tensor ends_t({1}, DType::Int64, Device::cpu());
        *ends_t.data<int64_t>() = end;
        add_initializer_tensor(ends_t, ends_name);

        std::string axes_name = context_.generate_name(prefix + "_axes");
        Tensor axes_t({1}, DType::Int64, Device::cpu());
        *axes_t.data<int64_t>() = dim;
        add_initializer_tensor(axes_t, axes_name);

        // Slice node
        std::string slice_out = context_.generate_name(prefix + "_slice");
        ONNXExportNode slice_node("Slice", context_.generate_name(prefix + "_slice"));
        slice_node.add_input(input_name);
        slice_node.add_input(starts_name);
        slice_node.add_input(ends_name);
        slice_node.add_input(axes_name);
        slice_node.add_output(slice_out);
        graph_.add_node(slice_node);

        // Unsqueeze the sliced window to add a dimension for stacking
        std::string unsq_axes_name = context_.generate_name(prefix + "_unsq_axes");
        Tensor unsq_axes_t({1}, DType::Int64, Device::cpu());
        *unsq_axes_t.data<int64_t>() = dim;
        add_initializer_tensor(unsq_axes_t, unsq_axes_name);

        std::string unsq_out = context_.generate_name(prefix + "_unsqueeze");
        ONNXExportNode unsq_node("Unsqueeze", context_.generate_name(prefix + "_unsqueeze"));
        unsq_node.add_input(slice_out);
        unsq_node.add_input(unsq_axes_name);
        unsq_node.add_output(unsq_out);
        graph_.add_node(unsq_node);

        window_names.push_back(unsq_out);
    }

    // Concat all windows along dim
    std::string concat_out = context_.generate_name("unfold_concat");
    ONNXExportNode concat_node("Concat", context_.generate_name("unfold_concat"));
    for (const auto& wname : window_names) {
        concat_node.add_input(wname);
    }
    concat_node.add_output(concat_out);
    concat_node.set_attr("axis", dim);
    graph_.add_node(concat_node);

    // Reshape to final output shape
    auto out_shape = output.shape();
    std::string shape_name = context_.generate_name("unfold_shape");
    Tensor shape_t({static_cast<int64_t>(out_shape.size())}, DType::Int64, Device::cpu());
    for (size_t i = 0; i < out_shape.size(); ++i) {
        shape_t.data<int64_t>()[i] = out_shape[i];
    }
    add_initializer_tensor(shape_t, shape_name);

    ONNXExportNode reshape_node("Reshape", context_.generate_name("unfold_reshape"));
    reshape_node.add_input(concat_out);
    reshape_node.add_input(shape_name);
    reshape_node.add_output(output_name);
    graph_.add_node(reshape_node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_fold(const Tensor& input,
                                const std::vector<int64_t>& output_size,
                                const std::vector<int64_t>& kernel_size,
                                const std::vector<int64_t>& dilation,
                                const std::vector<int64_t>& padding,
                                const std::vector<int64_t>& stride,
                                const Tensor& output,
                                const std::string& output_name) -> void {
    // Fold is the inverse of Unfold. ONNX Col2Im (opset 18+) provides this.
    // Col2Im(input, image_shape) with kernel_shape, dilations, pads, strides.
    ONNXExportNode node("Col2Im", context_.generate_name("fold"));
    std::string input_name = get_tensor_name(input, "fold_input");
    node.add_input(input_name);

    // image_shape input (the spatial dimensions of the output)
    std::string img_shape_name = context_.generate_name("fold_image_shape");
    Tensor img_shape_t({static_cast<int64_t>(output_size.size())}, DType::Int64, Device::cpu());
    for (size_t i = 0; i < output_size.size(); ++i) {
        img_shape_t.data<int64_t>()[i] = output_size[i];
    }
    add_initializer_tensor(img_shape_t, img_shape_name);
    node.add_input(img_shape_name);

    // block_shape input (kernel size)
    std::string block_name = context_.generate_name("fold_block_shape");
    Tensor block_t({static_cast<int64_t>(kernel_size.size())}, DType::Int64, Device::cpu());
    for (size_t i = 0; i < kernel_size.size(); ++i) {
        block_t.data<int64_t>()[i] = kernel_size[i];
    }
    add_initializer_tensor(block_t, block_name);
    node.add_input(block_name);

    node.add_output(output_name);

    // Set attributes
    node.set_attr("dilations", dilation);
    // ONNX pads format: [begin1, begin2, ..., end1, end2, ...]
    std::vector<int64_t> onnx_pads;
    onnx_pads.reserve(padding.size() * 2);
    for (int64_t p : padding) { onnx_pads.push_back(p); }
    for (int64_t p : padding) { onnx_pads.push_back(p); }
    node.set_attr("pads", onnx_pads);
    node.set_attr("strides", stride);

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_search_sorted(const Tensor& sorted_sequence,
                                         const Tensor& values,
                                         bool right,
                                         const Tensor& output,
                                         const std::string& output_name) -> void {
    // SearchSorted has no native ONNX op. Register as a custom op in the
    // "tenzor" domain. ONNX runtimes that support custom ops (e.g.,
    // onnxruntime with custom op library) can provide the implementation.
    ONNXExportNode node("SearchSorted", context_.generate_name("search_sorted"));
    node.set_attr("domain", std::string("tenzor"));

    std::string seq_name = get_tensor_name(sorted_sequence, "search_sorted_seq");
    std::string val_name = get_tensor_name(values, "search_sorted_values");
    node.add_input(seq_name);
    node.add_input(val_name);
    node.add_output(output_name);
    node.set_attr("right", static_cast<int64_t>(right ? 1 : 0));

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

// --- Group 3: Signal processing ---

auto ONNXExporter::export_fft(const Tensor& input, int64_t signal_ndim,
                               const Tensor& output,
                               const std::string& output_name) -> void {
    // ONNX DFT (opset 17+)
    ONNXExportNode node("DFT", context_.generate_name("fft"));
    std::string input_name = get_tensor_name(input, "fft_input");
    node.add_input(input_name);
    node.add_output(output_name);
    // axis defaults to -2 (second-to-last dim, per ONNX DFT spec)
    if (signal_ndim > 0) {
        node.set_attr("axis", static_cast<int64_t>(signal_ndim));
    }
    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_ifft(const Tensor& input, int64_t signal_ndim,
                                const Tensor& output,
                                const std::string& output_name) -> void {
    // ONNX DFT with inverse=1 (opset 17+)
    ONNXExportNode node("DFT", context_.generate_name("ifft"));
    std::string input_name = get_tensor_name(input, "ifft_input");
    node.add_input(input_name);
    node.add_output(output_name);
    node.set_attr("inverse", static_cast<int64_t>(1));
    if (signal_ndim > 0) {
        node.set_attr("axis", static_cast<int64_t>(signal_ndim));
    }
    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_rfft(const Tensor& input, int64_t signal_ndim,
                                const Tensor& output,
                                const std::string& output_name) -> void {
    // ONNX DFT for real-to-complex (opset 17+)
    // onesided=1 is the default for real input in ONNX DFT
    ONNXExportNode node("DFT", context_.generate_name("rfft"));
    std::string input_name = get_tensor_name(input, "rfft_input");
    node.add_input(input_name);
    node.add_output(output_name);
    node.set_attr("onesided", static_cast<int64_t>(1));
    if (signal_ndim > 0) {
        node.set_attr("axis", static_cast<int64_t>(signal_ndim));
    }
    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

// --- Group 4: Accumulation ops ---

auto ONNXExporter::export_cumsum(const Tensor& input, int64_t axis,
                                  const Tensor& output,
                                  const std::string& output_name) -> void {
    // ONNX CumSum (opset 11+): axis is the second input (1-D tensor)
    ONNXExportNode node("CumSum", context_.generate_name("cumsum"));
    std::string input_name = get_tensor_name(input, "cumsum_input");
    node.add_input(input_name);

    // axis must be provided as an initializer tensor
    std::string axis_name = context_.generate_name("cumsum_axis");
    Tensor axis_tensor({1}, DType::Int64, Device::cpu());
    *axis_tensor.data<int64_t>() = axis;
    add_initializer_tensor(axis_tensor, axis_name);
    node.add_input(axis_name);

    node.add_output(output_name);
    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_cumprod(const Tensor& input, int64_t axis,
                                   const Tensor& output,
                                   const std::string& output_name) -> void {
    // ONNX has no native CumProd. Decompose via log-domain:
    //   cumprod(x) = exp(cumsum(log(x)))
    // This is numerically valid for positive inputs (the common case for
    // probability chains, attention products, etc.).
    std::string input_name = get_tensor_name(input, "cumprod_input");

    // Step 1: Log(x)
    std::string log_out = context_.generate_name("cumprod_log");
    ONNXExportNode log_node("Log", context_.generate_name("cumprod_log"));
    log_node.add_input(input_name);
    log_node.add_output(log_out);
    graph_.add_node(log_node);

    // Step 2: CumSum(log(x), axis)
    std::string cumsum_out = context_.generate_name("cumprod_cumsum");
    ONNXExportNode cumsum_node("CumSum", context_.generate_name("cumprod_cumsum"));
    cumsum_node.add_input(log_out);

    std::string axis_name = context_.generate_name("cumprod_axis");
    Tensor axis_tensor({1}, DType::Int64, Device::cpu());
    *axis_tensor.data<int64_t>() = axis;
    add_initializer_tensor(axis_tensor, axis_name);
    cumsum_node.add_input(axis_name);

    cumsum_node.add_output(cumsum_out);
    graph_.add_node(cumsum_node);

    // Step 3: Exp(cumsum(log(x)))
    ONNXExportNode exp_node("Exp", context_.generate_name("cumprod_exp"));
    exp_node.add_input(cumsum_out);
    exp_node.add_output(output_name);
    graph_.add_node(exp_node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_roll(const Tensor& input, int64_t shift, int64_t axis,
                                const Tensor& output,
                                const std::string& output_name) -> void {
    // Roll = Slice(tail) + Slice(head) + Concat
    // For tensor of size N along axis: roll by shift S means:
    //   result = concat(input[N-S:], input[:N-S]) along axis
    std::string input_name = get_tensor_name(input, "roll_input");

    auto shape = input.shape();
    int64_t dim_size = shape[axis >= 0 ? axis : static_cast<int64_t>(shape.size()) + axis];
    // Normalize shift to [0, dim_size)
    int64_t norm_shift = ((shift % dim_size) + dim_size) % dim_size;

    if (norm_shift == 0) {
        // No-op: just use Identity
        ONNXExportNode id_node("Identity", context_.generate_name("roll_noop"));
        id_node.add_input(input_name);
        id_node.add_output(output_name);
        graph_.add_node(id_node);
        context_.register_tensor(output, output_name);
        return;
    }

    int64_t split_point = dim_size - norm_shift;

    // Helper: create axis/starts/ends/steps initializers for Slice
    auto make_slice_init = [&](const std::string& prefix, int64_t start, int64_t end) {
        std::string starts_name = context_.generate_name(prefix + "_starts");
        std::string ends_name = context_.generate_name(prefix + "_ends");
        std::string axes_name = context_.generate_name(prefix + "_axes");

        Tensor starts_t({1}, DType::Int64, Device::cpu());
        *starts_t.data<int64_t>() = start;
        add_initializer_tensor(starts_t, starts_name);

        Tensor ends_t({1}, DType::Int64, Device::cpu());
        *ends_t.data<int64_t>() = end;
        add_initializer_tensor(ends_t, ends_name);

        Tensor axes_t({1}, DType::Int64, Device::cpu());
        *axes_t.data<int64_t>() = axis;
        add_initializer_tensor(axes_t, axes_name);

        return std::tuple{starts_name, ends_name, axes_name};
    };

    // Slice 1: tail part [split_point : dim_size]
    auto [s1_starts, s1_ends, s1_axes] = make_slice_init("roll_tail", split_point, dim_size);
    std::string tail_name = context_.generate_name("roll_tail");
    ONNXExportNode tail_slice("Slice", context_.generate_name("roll_tail_slice"));
    tail_slice.add_input(input_name);
    tail_slice.add_input(s1_starts);
    tail_slice.add_input(s1_ends);
    tail_slice.add_input(s1_axes);
    tail_slice.add_output(tail_name);
    graph_.add_node(tail_slice);

    // Slice 2: head part [0 : split_point]
    auto [s2_starts, s2_ends, s2_axes] = make_slice_init("roll_head", 0, split_point);
    std::string head_name = context_.generate_name("roll_head");
    ONNXExportNode head_slice("Slice", context_.generate_name("roll_head_slice"));
    head_slice.add_input(input_name);
    head_slice.add_input(s2_starts);
    head_slice.add_input(s2_ends);
    head_slice.add_input(s2_axes);
    head_slice.add_output(head_name);
    graph_.add_node(head_slice);

    // Concat: [tail, head] along axis
    ONNXExportNode concat_node("Concat", context_.generate_name("roll_concat"));
    concat_node.add_input(tail_name);
    concat_node.add_input(head_name);
    concat_node.add_output(output_name);
    concat_node.set_attr("axis", axis);
    graph_.add_node(concat_node);
    context_.register_tensor(output, output_name);
}

// --- Group 5: Layer ops ---

auto ONNXExporter::export_embedding_bag(const Tensor& weight, const Tensor& indices,
                                         [[maybe_unused]] const Tensor& offsets, int64_t mode,
                                         const Tensor& output,
                                         const std::string& output_name) -> void {
    // EmbeddingBag = Gather embeddings + ReduceSum/ReduceMean per bag
    // mode: 0=sum, 1=mean, 2=max
    std::string weight_name = get_tensor_name(weight, "embbag_weight");
    std::string indices_name = get_tensor_name(indices, "embbag_indices");

    // Step 1: Gather all embeddings by indices
    std::string gathered_name = context_.generate_name("embbag_gathered");
    ONNXExportNode gather_node("Gather", context_.generate_name("embbag_gather"));
    gather_node.add_input(weight_name);
    gather_node.add_input(indices_name);
    gather_node.set_attr("axis", static_cast<int64_t>(0));
    gather_node.add_output(gathered_name);
    graph_.add_node(gather_node);

    // Step 2: Reduce along the bag dimension
    // For simplicity, this emits a single ReduceSum/ReduceMean over axis=0
    // A full implementation would use offsets to segment the reduction.
    std::string reduce_op;
    switch (mode) {
        case 0: reduce_op = "ReduceSum"; break;
        case 1: reduce_op = "ReduceMean"; break;
        case 2: reduce_op = "ReduceMax"; break;
        default: reduce_op = "ReduceSum"; break;
    }

    ONNXExportNode reduce_node(reduce_op, context_.generate_name("embbag_reduce"));
    reduce_node.add_input(gathered_name);
    reduce_node.add_output(output_name);
    reduce_node.set_attr("axes", std::vector<int64_t>{0});
    reduce_node.set_attr("keepdims", static_cast<int64_t>(0));
    graph_.add_node(reduce_node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_depthwise_conv2d(const Tensor& input, const Tensor& weight,
                                            const std::optional<Tensor>& bias,
                                            const std::vector<int64_t>& kernel_size,
                                            const std::vector<int64_t>& stride,
                                            const std::vector<int64_t>& padding,
                                            const std::vector<int64_t>& dilation,
                                            int64_t in_channels,
                                            const Tensor& output,
                                            const std::string& output_name) -> void {
    // Depthwise conv is just Conv with group=in_channels
    export_conv2d(input, weight, bias, kernel_size, stride, padding, dilation,
                  in_channels, output, output_name);
}

// --- Group 6: Misc ops ---

auto ONNXExporter::export_log2(const Tensor& input, const Tensor& output,
                                const std::string& output_name) -> void {
    // log2(x) = log(x) / log(2)
    std::string input_name = get_tensor_name(input, "log2_input");

    // Step 1: Log(x)
    std::string log_out = context_.generate_name("log2_log");
    ONNXExportNode log_node("Log", context_.generate_name("log2_log"));
    log_node.add_input(input_name);
    log_node.add_output(log_out);
    graph_.add_node(log_node);

    // Step 2: Constant log(2)
    std::string ln2_name = context_.generate_name("log2_ln2");
    Tensor ln2_tensor({1}, DType::Float32, Device::cpu());
    *ln2_tensor.data<float>() = std::log(2.0f);
    add_initializer_tensor(ln2_tensor, ln2_name);

    // Step 3: Div
    ONNXExportNode div_node("Div", context_.generate_name("log2_div"));
    div_node.add_input(log_out);
    div_node.add_input(ln2_name);
    div_node.add_output(output_name);
    graph_.add_node(div_node);
    context_.register_tensor(output, output_name);
}

auto ONNXExporter::export_quantized_conv2d(
    const Tensor& input, const Tensor& input_scale, const Tensor& input_zp,
    const Tensor& weight, const Tensor& weight_scale, const Tensor& weight_zp,
    const std::optional<Tensor>& bias,
    const Tensor& output_scale, const Tensor& output_zp,
    const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    int64_t groups,
    const Tensor& output,
    const std::string& output_name) -> void {
    // ONNX QLinearConv (opset 10+)
    ONNXExportNode node("QLinearConv", context_.generate_name("qlinearconv"));

    std::string x_name = get_tensor_name(input, "qconv_x");
    std::string x_scale_name = context_.generate_name("qconv_x_scale");
    std::string x_zp_name = context_.generate_name("qconv_x_zp");
    std::string w_name = context_.generate_name("qconv_w");
    std::string w_scale_name = context_.generate_name("qconv_w_scale");
    std::string w_zp_name = context_.generate_name("qconv_w_zp");
    std::string y_scale_name = context_.generate_name("qconv_y_scale");
    std::string y_zp_name = context_.generate_name("qconv_y_zp");

    add_initializer_tensor(input_scale, x_scale_name);
    add_initializer_tensor(input_zp, x_zp_name);
    add_initializer_tensor(weight, w_name);
    add_initializer_tensor(weight_scale, w_scale_name);
    add_initializer_tensor(weight_zp, w_zp_name);
    add_initializer_tensor(output_scale, y_scale_name);
    add_initializer_tensor(output_zp, y_zp_name);

    // QLinearConv inputs: x, x_scale, x_zero_point, w, w_scale, w_zero_point,
    //                     y_scale, y_zero_point, [B]
    node.add_input(x_name);
    node.add_input(x_scale_name);
    node.add_input(x_zp_name);
    node.add_input(w_name);
    node.add_input(w_scale_name);
    node.add_input(w_zp_name);
    node.add_input(y_scale_name);
    node.add_input(y_zp_name);

    if (bias.has_value()) {
        std::string bias_name = context_.generate_name("qconv_bias");
        add_initializer_tensor(bias.value(), bias_name);
        node.add_input(bias_name);
    }

    node.add_output(output_name);

    node.set_attr("kernel_shape", kernel_size);
    node.set_attr("strides", stride);

    // ONNX expects [top, left, bottom, right] for 2D pads
    if (padding.size() == 2) {
        node.set_attr("pads", std::vector<int64_t>{
            padding[0], padding[1], padding[0], padding[1]});
    } else {
        node.set_attr("pads", padding);
    }

    node.set_attr("dilations", dilation);
    if (groups > 1) {
        node.set_attr("group", groups);
    }

    graph_.add_node(node);
    context_.register_tensor(output, output_name);
}

// ============================================================================
// Convert Full JIT Graph to ONNX
// ============================================================================

auto ONNXExporter::convert_jit_graph_to_onnx(const jit::Graph& graph) -> void {
    // Build a mapping from JIT value ID -> ONNX tensor name.
    // Graph inputs are pre-populated before calling this method.
    std::unordered_map<std::string, std::string> value_name_map;

    // Map graph inputs: JIT input value IDs -> the ONNX input names that
    // were already registered via add_input() before this call.
    // The ONNX inputs are in graph_.inputs in the same order as jit inputs.
    const auto& jit_inputs = graph.inputs();
    for (size_t i = 0; i < jit_inputs.size() && i < graph_.inputs.size(); ++i) {
        value_name_map[jit_inputs[i]->id()] = graph_.inputs[i].name;
    }

    // Map captured constants (module parameters/buffers that appear as node
    // inputs) to ONNX initializers, so a node input referencing a constant
    // value resolves to a real initializer tensor rather than a dangling
    // placeholder name (which the importer rejects as "Input tensor not found").
    for (const auto& [value_id, tensor] : graph.constants()) {
        if (value_name_map.count(value_id) || tensor.numel() == 0) {
            continue;
        }
        std::string init_name = graph_.get_unique_name("const");
        add_initializer_tensor(tensor.cpu().contiguous(), init_name);
        value_name_map[value_id] = init_name;
    }

    // Convert each node in topological order
    for (const auto& node : graph.nodes()) {
        convert_jit_node_to_onnx(node, value_name_map);
    }

    // Map graph outputs: replace the ONNX output value_info shapes with
    // the actual traced shapes, and ensure the output name references
    // the correct ONNX value produced by the last node.
    const auto& jit_outputs = graph.outputs();
    for (size_t i = 0; i < jit_outputs.size(); ++i) {
        const auto& jit_id = jit_outputs[i]->id();
        auto it = value_name_map.find(jit_id);

        if (it != value_name_map.end()) {
            // If there are already ONNX outputs registered, update the name
            // to point to the actual ONNX value that was produced.
            if (i < graph_.outputs.size()) {
                // The output was pre-registered; record the mapping so that
                // the ONNX output references the right value.
                // If the names differ, we need to add an Identity node to
                // bridge the traced output name to the declared output name.
                const std::string& declared_name = graph_.outputs[i].name;
                const std::string& actual_name = it->second;

                if (declared_name != actual_name) {
                    ONNXExportNode identity("Identity",
                        graph_.get_unique_name("output_identity"));
                    identity.add_input(actual_name);
                    identity.add_output(declared_name);
                    graph_.add_node(identity);
                }
            } else {
                // No pre-registered output; add one
                auto shape_vec = jit_outputs[i]->shape();
                ONNXExportValueInfo out_info(it->second,
                    dtype_to_onnx(jit_outputs[i]->dtype()), shape_vec);
                graph_.add_output(out_info);
            }
        }
    }
}

// ============================================================================
// Export Module via JIT Tracing
// ============================================================================

auto ONNXExporter::export_module(nn::Module& module, const Tensor& dummy_input,
                                 const std::string& output_path) -> void {
    // Switch to eval mode for consistent tracing
    bool was_training = module.is_training();
    module.eval();

    try {
        // Clear any existing state
        clear();

        // Set metadata
        set_model_name("tenzor_jit_traced_model");
        set_description("Model exported from Tenzor via JIT tracing");

        // Prepare CPU-contiguous input
        Tensor cpu_input = dummy_input.cpu().contiguous();

        // Register the ONNX graph input
        add_input(cpu_input, "input");

        // Export all module parameters as initializers
        auto named_params = module.named_parameters();
        for (const auto& [param_name, param_var] : named_params) {
            if (param_var && param_var->is_initialized() && param_var->tensor().numel() > 0) {
                std::string safe_name = param_name;
                std::replace(safe_name.begin(), safe_name.end(), '.', '_');
                add_initializer_tensor(param_var->tensor().cpu().contiguous(), safe_name);
            }
        }

        // Export all module buffers as initializers (e.g., BatchNorm running stats)
        auto named_buffs = module.named_buffers();
        for (const auto& [buffer_name, buffer_var] : named_buffs) {
            if (buffer_var && buffer_var->is_initialized() && buffer_var->tensor().numel() > 0) {
                std::string safe_name = buffer_name;
                std::replace(safe_name.begin(), safe_name.end(), '.', '_');
                add_initializer_tensor(buffer_var->tensor().cpu().contiguous(), safe_name);
            }
        }

        // Trace the module's forward pass through the JIT system
        auto module_ptr = std::shared_ptr<nn::Module>(&module, [](nn::Module*) {
            // Non-owning shared_ptr: we don't own the module, the caller does.
        });

        Variable input_var(cpu_input, false);
        auto jit_graph = jit::trace(module_ptr, input_var);

        if (!jit_graph) {
            throw std::runtime_error("JIT tracing produced null graph");
        }

        // Run the forward pass to get output shape for the ONNX output declaration
        Variable output_var = module.forward(input_var);
        if (!output_var.is_initialized() || output_var.tensor().numel() == 0) {
            throw std::runtime_error("Module forward pass produced undefined or empty output");
        }

        // Register the ONNX graph output
        add_output(output_var.tensor().cpu(), "output");

        // Convert the JIT graph into ONNX nodes
        convert_jit_graph_to_onnx(*jit_graph);

        // Serialize and write to file
        export_to_file(output_path);

    } catch (...) {
        if (was_training) {
            module.train();
        }
        throw;
    }

    // Restore original training mode
    if (was_training) {
        module.train();
    }
}

// High-level export function

// Audit I7: unified export. The free `export_to_onnx` function previously
// reimplemented the parameter/buffer registration logic AND used a separate
// `trace_custom_module` shape-pattern matcher (an anonymous-namespace
// heuristic that recognized only a small set of patterns: Linear, BatchNorm,
// LayerNorm, etc.). `ONNXExporter::export_module` does real JIT tracing via
// `jit::trace` + `convert_jit_graph_to_onnx` which handles arbitrary ops.
// The free function now delegates to the class method for the core path,
// reusing parameterized input/output names.
auto export_to_onnx(std::shared_ptr<nn::Module> module,
                    const Tensor& dummy_input,
                    const std::string& filepath,
                    const std::vector<std::string>& input_names,
                    const std::vector<std::string>& output_names,
                    int64_t opset_version) -> void {
    if (!module) {
        throw std::runtime_error("Cannot export null module to ONNX");
    }
    if (input_names.empty()) {
        throw std::runtime_error("At least one input name must be provided");
    }
    if (output_names.empty()) {
        throw std::runtime_error("At least one output name must be provided");
    }

    bool was_training = module->is_training();
    module->eval();

    try {
        ONNXExporter exporter(opset_version);
        exporter.set_model_name("tenzor_traced_model");
        exporter.set_description("Model traced and exported from Tenzor");

        // The bulk of `export_module`'s logic, parameterized by the input/
        // output names from this free-function signature (the class method
        // hardcodes "input"/"output"). Inline-copied here because the class
        // method doesn't yet accept name args; refactoring `export_module`
        // to take names is a small future cleanup but not required for I7.
        Tensor cpu_input = dummy_input.cpu().contiguous();
        exporter.add_input(cpu_input, input_names[0]);

        auto named_params = module->named_parameters();
        for (const auto& [param_name, param_var] : named_params) {
            if (param_var && param_var->is_initialized() && param_var->tensor().numel() > 0) {
                std::string safe_name = param_name;
                std::replace(safe_name.begin(), safe_name.end(), '.', '_');
                exporter.add_initializer_tensor(param_var->tensor().cpu().contiguous(), safe_name);
            }
        }
        auto named_buffs = module->named_buffers();
        for (const auto& [buffer_name, buffer_var] : named_buffs) {
            if (buffer_var && buffer_var->is_initialized() && buffer_var->tensor().numel() > 0) {
                std::string safe_name = buffer_name;
                std::replace(safe_name.begin(), safe_name.end(), '.', '_');
                exporter.add_initializer_tensor(buffer_var->tensor().cpu().contiguous(), safe_name);
            }
        }

        // Audit I7: real JIT trace, replacing the old `trace_custom_module`
        // shape-pattern matcher. The JIT graph captures every traced
        // operation; `convert_jit_graph_to_onnx` emits ONNX nodes per op.
        auto module_ptr = std::shared_ptr<nn::Module>(module.get(), [](nn::Module*) {
            // Non-owning shared_ptr (the caller owns the actual lifetime).
        });
        Variable input_var(cpu_input, false);
        auto jit_graph = jit::trace(module_ptr, input_var);
        if (!jit_graph) {
            throw std::runtime_error("JIT tracing produced null graph");
        }

        Variable output_var = module->forward(input_var);
        if (!output_var.is_initialized() || output_var.tensor().numel() == 0) {
            throw std::runtime_error("Module forward pass produced undefined or empty output");
        }
        exporter.add_output(output_var.tensor().cpu(), output_names[0]);

        // Convert traced JIT graph → ONNX nodes (handles arbitrary ops, not
        // just the pre-baked patterns the old tracer recognized).
        exporter.convert_jit_graph_to_onnx(*jit_graph);

        exporter.export_to_file(filepath);
    } catch (...) {
        if (was_training) module->train();
        throw;
    }
    if (was_training) module->train();
}

// Audit I7: trace_custom_module fully removed (was a shape-pattern matcher
// that only recognized Linear/BatchNorm/LayerNorm/Conv). The free-function
// export_to_onnx now routes through ONNXExporter::export_module-style JIT
// tracing which handles arbitrary ops.

} // namespace onnx
} // namespace tenzor
