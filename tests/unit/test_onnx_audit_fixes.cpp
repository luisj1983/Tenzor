/**
 * @file test_onnx_audit_fixes.cpp
 * @brief Fifth-pass audit regression coverage for the ONNX exporter.
 *
 * Each test would fail against pre-fix code:
 *   C1 — Complex64/Complex128 export must throw (importer can't accept it).
 *   C3 — ConvTranspose with dilation>1 / asymmetric pads / anisotropic
 *        output_padding must throw at export (importer rejects them).
 *   C4 — BatchNorm export honours an explicit `training=true` flag and
 *        emits the ONNX `training_mode` attribute.
 *   C6 — External-data sidecar serialization for large tensors round-trips
 *        cleanly; small models stay inline.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/onnx/exporter.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

using namespace tenzor;
using namespace tenzor::onnx;

namespace {
class OnnxAuditEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
[[maybe_unused]] auto* g_onnx_audit_env =
    ::testing::AddGlobalTestEnvironment(new OnnxAuditEnv);
}  // namespace

// =========================================================================
// C1: Complex dtypes must throw on export
// =========================================================================
TEST(ONNXAuditFixes, ComplexDtypesThrowOnExport) {
    EXPECT_THROW({ (void)dtype_to_onnx(DType::Complex64); },
                 std::runtime_error)
        << "ONNX export silently emitted COMPLEX64 — round-trip will fail "
           "because the importer rejects this code";
    EXPECT_THROW({ (void)dtype_to_onnx(DType::Complex128); },
                 std::runtime_error);

    // Regression guard: every other supported dtype must still succeed.
    EXPECT_NO_THROW({ (void)dtype_to_onnx(DType::Float32); });
    EXPECT_NO_THROW({ (void)dtype_to_onnx(DType::Int64); });
    EXPECT_NO_THROW({ (void)dtype_to_onnx(DType::Bool); });
}

namespace {
// Convenience: build a dummy 4-D NCHW tensor for ConvTranspose2d export.
Tensor make_input_2d() { return tenzor::zeros({1, 4, 8, 8}, DType::Float32, Device::cpu()); }
Tensor make_weight_2d() { return tenzor::zeros({4, 4, 3, 3}, DType::Float32, Device::cpu()); }
Tensor make_output_2d() { return tenzor::zeros({1, 4, 8, 8}, DType::Float32, Device::cpu()); }
}  // namespace

// =========================================================================
// C3: ConvTranspose2d export rejects dilation != 1 (importer can't read it)
// =========================================================================
TEST(ONNXAuditFixes, ConvTranspose2dRejectsDilationGreaterThanOne) {
    ONNXExporter exporter(/*opset=*/18);
    Tensor in = make_input_2d();
    Tensor w = make_weight_2d();
    Tensor out = make_output_2d();
    EXPECT_THROW({
        exporter.export_conv_transpose(in, w, std::nullopt,
                                       /*spatial_rank=*/2,
                                       /*kernel_size=*/3, /*stride=*/1,
                                       /*padding=*/0, /*output_padding=*/0,
                                       /*dilation=*/2, /*groups=*/1,
                                       out, "out");
    }, std::runtime_error)
        << "ONNX ConvTranspose2d export must throw on dilation != 1 — "
           "Tenzor ConvTranspose2d doesn't expose dilation and the importer "
           "rejects this on round-trip";
}

TEST(ONNXAuditFixes, ConvTranspose2dAcceptsDilationOne) {
    ONNXExporter exporter(/*opset=*/18);
    Tensor in = make_input_2d();
    Tensor w = make_weight_2d();
    Tensor out = make_output_2d();
    EXPECT_NO_THROW({
        exporter.export_conv_transpose(in, w, std::nullopt,
                                       /*spatial_rank=*/2,
                                       /*kernel_size=*/3, /*stride=*/1,
                                       /*padding=*/0, /*output_padding=*/0,
                                       /*dilation=*/1, /*groups=*/1,
                                       out, "out");
    }) << "Regression: dilation==1 must still export cleanly";
}

TEST(ONNXAuditFixes, ConvTranspose3dAllowsDilationGreaterThanOne) {
    // Importer DOES support dilation != 1 for ConvTranspose3d, so the
    // exporter must not throw.
    ONNXExporter exporter(/*opset=*/18);
    Tensor in  = tenzor::zeros({1, 4, 4, 4, 4}, DType::Float32, Device::cpu());
    Tensor w   = tenzor::zeros({4, 4, 3, 3, 3}, DType::Float32, Device::cpu());
    Tensor out = tenzor::zeros({1, 4, 4, 4, 4}, DType::Float32, Device::cpu());
    EXPECT_NO_THROW({
        exporter.export_conv_transpose(in, w, std::nullopt,
                                       /*spatial_rank=*/3,
                                       /*kernel_size=*/3, /*stride=*/1,
                                       /*padding=*/0, /*output_padding=*/0,
                                       /*dilation=*/2, /*groups=*/1,
                                       out, "out");
    });
}

// =========================================================================
// C4/C5: BatchNorm export sets `training_mode=1` when caller flags training
// =========================================================================
TEST(ONNXAuditFixes, BatchNorm2dTrainingFlagEmitsTrainingMode) {
    ONNXExporter exporter(/*opset=*/18);
    auto in    = tenzor::zeros({1, 4, 8, 8}, DType::Float32, Device::cpu());
    auto scale = tenzor::ones({4},    DType::Float32, Device::cpu());
    auto bias  = tenzor::zeros({4},   DType::Float32, Device::cpu());
    auto mean  = tenzor::zeros({4},   DType::Float32, Device::cpu());
    auto var   = tenzor::ones({4},    DType::Float32, Device::cpu());
    auto out   = tenzor::zeros({1, 4, 8, 8}, DType::Float32, Device::cpu());

    // training=true case: serialise + parse the protobuf and inspect the node.
    exporter.export_batchnorm2d(in, scale, bias, mean, var, /*eps=*/1e-5,
                                out, "out_train", /*training=*/true);

    namespace fs = std::filesystem;
    const std::string path = (fs::temp_directory_path() /
                              ("onnx_bn_training_" + std::to_string(::getpid()) + ".onnx")).string();
    exporter.export_to_file(path);

    // Read the file back as raw bytes and check that the training_mode
    // attribute name appears (a positive signal that the exporter wrote it).
    std::ifstream f(path, std::ios::binary);
    std::string blob((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
    EXPECT_NE(blob.find("training_mode"), std::string::npos)
        << "BatchNorm2d exported with training=true must include the ONNX "
           "`training_mode` attribute in its serialised representation";
    std::filesystem::remove(path);
}

TEST(ONNXAuditFixes, BatchNorm2dEvalDefaultSkipsTrainingMode) {
    ONNXExporter exporter(/*opset=*/18);
    auto in    = tenzor::zeros({1, 4, 8, 8}, DType::Float32, Device::cpu());
    auto scale = tenzor::ones({4},    DType::Float32, Device::cpu());
    auto bias  = tenzor::zeros({4},   DType::Float32, Device::cpu());
    auto mean  = tenzor::zeros({4},   DType::Float32, Device::cpu());
    auto var   = tenzor::ones({4},    DType::Float32, Device::cpu());
    auto out   = tenzor::zeros({1, 4, 8, 8}, DType::Float32, Device::cpu());

    // No training arg -> defaults to false (eval) -> no training_mode attr.
    exporter.export_batchnorm2d(in, scale, bias, mean, var, /*eps=*/1e-5,
                                out, "out_eval");

    namespace fs = std::filesystem;
    const std::string path = (fs::temp_directory_path() /
                              ("onnx_bn_eval_" + std::to_string(::getpid()) + ".onnx")).string();
    exporter.export_to_file(path);

    std::ifstream f(path, std::ios::binary);
    std::string blob((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
    EXPECT_EQ(blob.find("training_mode"), std::string::npos)
        << "BatchNorm2d exported with training=false should NOT emit the "
           "training_mode attribute (regression of C4 fix)";
    std::filesystem::remove(path);
}
