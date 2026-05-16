// test_onnx_exporter_coverage.cpp
//
// Wave Inf-D: enforces ONNX exporter coverage across the OpId enum + the
// extended DType set (Float8E4M3FN, Float8E5M2, INT4-packed). Acts as the
// runtime exhaustiveness check that compile-time static_assert cannot
// express against a 470-entry switch.
//
// What it verifies:
//
//   1. Every forward OpId (not *Backward / *Inplace / known autograd-internal
//      sentinel) has a non-empty mapping in `op_to_onnx`, OR is listed in
//      the skip list documented below.
//
//   2. Each extended DType maps to its expected ONNX type code, and
//      Complex64/128 still throw with the documented round-trip message.
//
//   3. A spot-check that representative Inf-D additions (FFT block,
//      Pooling extras, Norms, Linalg-as-custom, Random with seed,
//      Bessel/special) actually return the names this wave intended.

#include <gtest/gtest.h>
#include <tenzor/onnx/exporter.hpp>
#include <tenzor/onnx/types.hpp>
#include <tenzor/ops/op_id.hpp>
#include <tenzor/core/dtype.hpp>
#include <set>
#include <string>
#include <string_view>

using tenzor::OpId;
using tenzor::DType;
using tenzor::onnx::ONNXExporter;
using tenzor::onnx::ONNXDataType;
using tenzor::onnx::dtype_to_onnx;
using tenzor::op_id_to_name;

namespace {

// OpIds that are intentionally NOT exportable to ONNX. Anything reaching
// `op_to_onnx` for one of these is misuse — the exporter walks the
// *forward* graph only, while these are autograd-internal, allocator
// sentinels, or backward ops the graph tracer drops.
//
// Anything outside this list must either return a non-empty mapping or
// be flagged by this test so the developer adds a case.
const std::set<OpId> kExportSkipList = {
    OpId::OP_COUNT,                  // sentinel — not a real op
    OpId::BatchNorm2dMeanVar,        // intermediate stats output
    // Optimizer fused-step ops are training-internal — they update parameters
    // in place during training and never appear in an inference graph that
    // gets exported. Explicitly skip-listed so the coverage test stays
    // strict about everything else.
    OpId::FusedSGDStep,
    OpId::FusedAdamStep,
    OpId::FusedRMSPropStep,
    OpId::FusedAdadeltaStep,
    OpId::FusedAdagradStep,
    OpId::FusedAdamAtan2Step,
};

// Backward / Inplace OpIds and unnamed enum-gap entries are filtered out
// programmatically. `op_id_to_name` returns snake_case strings, and "unknown"
// for OpId values that are valid enum slots but lack a name entry (e.g.
// numeric gaps from explicit `= 210` assignments). The exporter is not
// expected to handle these.
bool is_autograd_internal(std::string_view name) {
    if (name == "unknown") return true;
    auto ends_with = [](std::string_view s, std::string_view suf) {
        return s.size() >= suf.size() &&
               s.substr(s.size() - suf.size()) == suf;
    };
    if (ends_with(name, "_backward")) return true;
    if (ends_with(name, "_inplace")) return true;
    if (ends_with(name, "Backward")) return true;  // defensive: any PascalCase
    if (ends_with(name, "Inplace"))  return true;
    // conv2d_backward_input / _weight / _bias style — backward gradient
    // splits. Anything with `_backward_` in the middle of the name.
    if (name.find("_backward_") != std::string_view::npos) return true;
    return false;
}

}  // namespace

// ----------------------------------------------------------------------------
// D4.1: every forward OpId has a mapping or is on the skip list.
// ----------------------------------------------------------------------------
TEST(ONNXExporterCoverage, EveryForwardOpIdHasMapping) {
    int total_checked = 0;
    int total_mapped = 0;
    std::vector<std::string> missing;

    for (int v = 0; v < static_cast<int>(OpId::OP_COUNT); ++v) {
        OpId op = static_cast<OpId>(v);
        std::string name(op_id_to_name(op));

        // Skip autograd-internal (backward/inplace).
        if (is_autograd_internal(name)) continue;
        // Skip-listed sentinels.
        if (kExportSkipList.count(op)) continue;

        ++total_checked;
        try {
            std::string mapped = ONNXExporter::op_to_onnx(op);
            if (mapped.empty()) {
                missing.emplace_back(name + " → empty string");
            } else {
                ++total_mapped;
            }
        } catch (const std::exception& e) {
            missing.emplace_back(name + " → throws: " + e.what());
        }
    }

    EXPECT_GT(total_checked, 0);
    if (!missing.empty()) {
        std::string msg = "Forward OpIds missing ONNX mapping (" +
                          std::to_string(missing.size()) + "):\n";
        for (const auto& m : missing) msg += "  - " + m + "\n";
        msg += "\nAdd a case in src/onnx/exporter.cpp::op_to_onnx, OR add the "
               "OpId to kExportSkipList in this test file with justification.";
        ADD_FAILURE() << msg;
    }
}

// ----------------------------------------------------------------------------
// D4.2: extended DType mappings (BF16, FP8, INT4).
// ----------------------------------------------------------------------------
TEST(ONNXExporterCoverage, BFloat16MapsCorrectly) {
    EXPECT_EQ(dtype_to_onnx(DType::BFloat16), ONNXDataType::BFLOAT16);
}

TEST(ONNXExporterCoverage, Float8E4M3FNMapsCorrectly) {
    EXPECT_EQ(dtype_to_onnx(DType::FP8_E4M3), ONNXDataType::FLOAT8E4M3FN);
}

TEST(ONNXExporterCoverage, Float8E5M2MapsCorrectly) {
    EXPECT_EQ(dtype_to_onnx(DType::FP8_E5M2), ONNXDataType::FLOAT8E5M2);
}

TEST(ONNXExporterCoverage, QInt4x2MapsToInt4) {
    EXPECT_EQ(dtype_to_onnx(DType::QInt4x2), ONNXDataType::INT4);
}

TEST(ONNXExporterCoverage, QInt8MapsToInt8) {
    EXPECT_EQ(dtype_to_onnx(DType::QInt8), ONNXDataType::INT8);
}

TEST(ONNXExporterCoverage, QUInt8MapsToUInt8) {
    EXPECT_EQ(dtype_to_onnx(DType::QUInt8), ONNXDataType::UINT8);
}

TEST(ONNXExporterCoverage, Complex64StillThrowsWithRoundTripMessage) {
    EXPECT_THROW({
        try {
            dtype_to_onnx(DType::Complex64);
        } catch (const std::exception& e) {
            EXPECT_NE(std::string(e.what()).find("round-trippable"), std::string::npos);
            throw;
        }
    }, std::runtime_error);
}

TEST(ONNXExporterCoverage, Complex128StillThrowsWithRoundTripMessage) {
    EXPECT_THROW({
        try {
            dtype_to_onnx(DType::Complex128);
        } catch (const std::exception& e) {
            EXPECT_NE(std::string(e.what()).find("round-trippable"), std::string::npos);
            throw;
        }
    }, std::runtime_error);
}

// ----------------------------------------------------------------------------
// D4.3: spot-check key Inf-D2 additions hit the right ONNX names.
// ----------------------------------------------------------------------------
TEST(ONNXExporterCoverage, FFTBlockMapsToDFT) {
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::FFT),   "DFT");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::IFFT),  "DFT");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::RFFT),  "DFT");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::IRFFT), "DFT");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::FFT2),  "DFT");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::FFTN),  "DFT");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::IFFT2), "DFT");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::IFFTN), "DFT");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::STFT),  "STFT");
}

TEST(ONNXExporterCoverage, PoolingExtrasMap) {
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::AvgPool1dForward),  "AveragePool");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::AdaptiveAvgPool1d), "AveragePool");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::AdaptiveAvgPool3d), "AveragePool");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::AdaptiveMaxPool1d), "MaxPool");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::AdaptiveMaxPool3d), "MaxPool");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::MaxUnpool1dForward), "MaxUnpool");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::MaxUnpool2dForward), "MaxUnpool");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::MaxUnpool3dForward), "MaxUnpool");
}

TEST(ONNXExporterCoverage, NormExtrasMap) {
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::RMSNorm),        "RMSNormalization");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::FusedLayerNorm), "LayerNormalization");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::FusedRMSNorm),   "RMSNormalization");
}

TEST(ONNXExporterCoverage, BesselSpecialMapsToCustomDomain) {
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::BesselI0), "BesselI0");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::BesselI1), "BesselI1");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::Lgamma),   "Lgamma");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::Digamma),  "Digamma");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::Igamma),   "Igamma");
}

TEST(ONNXExporterCoverage, RandomOpsMap) {
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::Bernoulli),    "Bernoulli");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::Multinomial),  "Multinomial");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::Rand),         "RandomUniform");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::Randn),        "RandomNormal");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::NormalSample), "RandomNormal");
}

TEST(ONNXExporterCoverage, BitwiseAndIsCheckOpsMap) {
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::BitwiseAnd),       "BitwiseAnd");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::BitwiseOr),        "BitwiseOr");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::BitwiseXor),       "BitwiseXor");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::BitwiseNot),       "BitwiseNot");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::BitwiseLeftShift), "BitShift");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::IsNan),            "IsNaN");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::IsInf),            "IsInf");
}

TEST(ONNXExporterCoverage, CreationOpsMap) {
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::Eye),       "EyeLike");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::Arange),    "Range");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::Linspace),  "Range");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::Zeros),     "ConstantOfShape");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::Ones),      "ConstantOfShape");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::Full),      "ConstantOfShape");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::Fill),      "ConstantOfShape");
}

TEST(ONNXExporterCoverage, LinalgOpsMap) {
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::LinalgDet),      "Det");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::LinalgCholesky), "LinalgCholesky");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::LinalgSVD),      "LinalgSVD");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::LinalgQR),       "LinalgQR");
    EXPECT_EQ(ONNXExporter::op_to_onnx(OpId::Einsum),         "Einsum");
}
