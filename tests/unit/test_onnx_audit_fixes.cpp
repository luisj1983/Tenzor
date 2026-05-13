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

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

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

// =========================================================================
// C6: external_data sidecar for large initializers
// =========================================================================
namespace {
constexpr int64_t kBigBytes = 4 * 1024 * 1024;  // 4 MiB, > default 1 MiB threshold
constexpr int64_t kSmallElems = 4;
}  // namespace

TEST(ONNXAuditFixes, ExternalDataSidecarRoundTrip) {
    // Build a graph with a few large initializers via BatchNorm2d export
    // (the only public path that registers initializers). Each of the
    // four BN tensors (scale, bias, mean, var) is 4 MiB at C=1M float.
    // Export with use_external_data=true; verify the sidecar receives
    // those bytes and the main .onnx file stays small.
    ONNXExporter exporter(/*opset=*/18);

    const int64_t C = kBigBytes / 4;  // 1M channels => 4 MiB per tensor
    auto input  = tenzor::zeros({1, C, 1, 1}, DType::Float32, Device::cpu());
    auto scale  = tenzor::ones({C}, DType::Float32, Device::cpu());
    auto bias   = tenzor::zeros({C}, DType::Float32, Device::cpu());
    auto mean   = tenzor::zeros({C}, DType::Float32, Device::cpu());
    auto var    = tenzor::ones({C}, DType::Float32, Device::cpu());
    auto output = tenzor::zeros({1, C, 1, 1}, DType::Float32, Device::cpu());
    exporter.export_batchnorm2d(input, scale, bias, mean, var,
                                /*eps=*/1e-5, output, "bn_out");

    namespace fs = std::filesystem;
    const fs::path proto_path = fs::temp_directory_path() /
        ("onnx_ext_" + std::to_string(::getpid()) + ".onnx");
    const fs::path data_path = proto_path.string() + ".data";

    exporter.export_to_file(proto_path.string(),
                            /*use_external_data=*/std::optional<bool>{true},
                            /*threshold_bytes=*/1ULL << 20);

    // Both files must exist.
    ASSERT_TRUE(fs::exists(proto_path))
        << "Main .onnx file missing: " << proto_path;
    ASSERT_TRUE(fs::exists(data_path))
        << "Sidecar .data file missing: " << data_path;

    // Sidecar must contain exactly the four BN tensors' bytes
    // (4 * 4 MiB = 16 MiB).
    EXPECT_EQ(fs::file_size(data_path), static_cast<uintmax_t>(4 * kBigBytes))
        << "Sidecar size does not match the combined initializer bytes";

    // The proto file must be small — multi-MiB initializers are OUT of it.
    EXPECT_LT(fs::file_size(proto_path), static_cast<uintmax_t>(kBigBytes))
        << "Main .onnx file is unexpectedly large — initializers may have "
           "been inlined despite use_external_data=true";

    // The proto bytes should mention the data_basename + the external_data keys.
    {
        std::ifstream f(proto_path, std::ios::binary);
        std::string blob((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
        EXPECT_NE(blob.find(data_path.filename().string()), std::string::npos)
            << "Proto blob does not reference the sidecar basename";
        EXPECT_NE(blob.find("location"), std::string::npos);
        EXPECT_NE(blob.find("offset"), std::string::npos);
        EXPECT_NE(blob.find("length"), std::string::npos);
    }

    // Verify the sidecar's first 8 bytes are the same as the BN `scale`
    // tensor's first 8 bytes (raw_data round-trip; ordering: scale, bias,
    // mean, var by the order export_batchnorm2d adds initializers).
    {
        std::ifstream f(data_path, std::ios::binary);
        char head[8];
        f.read(head, 8);
        const auto* expect = reinterpret_cast<const char*>(scale.data<float>());
        for (size_t i = 0; i < 8; ++i) {
            EXPECT_EQ(head[i], expect[i])
                << "Sidecar byte " << i << " does not match BN scale";
        }
    }

    fs::remove(proto_path);
    fs::remove(data_path);
}

TEST(ONNXAuditFixes, SmallModelDoesNotEmitSidecar) {
    // Tiny BN: total initializer bytes << 1.5 GB autostart threshold and
    // << 1 MiB per-tensor threshold. No sidecar should be created in
    // auto mode.
    ONNXExporter exporter(/*opset=*/18);
    const int64_t C = kSmallElems;
    auto input  = tenzor::zeros({1, C, 1, 1}, DType::Float32, Device::cpu());
    auto scale  = tenzor::ones({C},  DType::Float32, Device::cpu());
    auto bias   = tenzor::zeros({C}, DType::Float32, Device::cpu());
    auto mean   = tenzor::zeros({C}, DType::Float32, Device::cpu());
    auto var    = tenzor::ones({C},  DType::Float32, Device::cpu());
    auto output = tenzor::zeros({1, C, 1, 1}, DType::Float32, Device::cpu());
    exporter.export_batchnorm2d(input, scale, bias, mean, var,
                                /*eps=*/1e-5, output, "bn_out");

    namespace fs = std::filesystem;
    const fs::path proto_path = fs::temp_directory_path() /
        ("onnx_no_ext_" + std::to_string(::getpid()) + ".onnx");
    const fs::path data_path = proto_path.string() + ".data";
    fs::remove(data_path);  // ensure clean slate

    // Default (auto) should NOT externalise for a small model.
    exporter.export_to_file(proto_path.string(),
                            /*use_external_data=*/std::optional<bool>{});

    ASSERT_TRUE(fs::exists(proto_path));
    EXPECT_FALSE(fs::exists(data_path))
        << "Small-model export must not produce an external-data sidecar";

    fs::remove(proto_path);
}
