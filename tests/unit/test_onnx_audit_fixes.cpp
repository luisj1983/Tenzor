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
#include <tenzor/onnx/importer.hpp>
#include <tenzor/nn/layers/linear.hpp>

#include <algorithm>
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

// =========================================================================
// 6th-audit Fix #1: importer must read external_data sidecars.
// =========================================================================

TEST(ONNXAuditFixes, ImporterReadsExternalDataRoundTrip) {
    using namespace tenzor::onnx;
    namespace fs = std::filesystem;

    // Export a BN model with external_data, then re-parse it via the importer
    // and verify the initializer bytes were loaded from the sidecar (not
    // silently treated as empty).
    ONNXExporter exporter(/*opset=*/18);
    const int64_t C = (4 * 1024 * 1024) / 4;  // 4 MiB per BN tensor (Float32)

    auto input  = tenzor::zeros({1, C, 1, 1}, DType::Float32, Device::cpu());
    auto scale  = tenzor::ones({C}, DType::Float32, Device::cpu());
    auto bias   = tenzor::zeros({C}, DType::Float32, Device::cpu());
    auto mean   = tenzor::zeros({C}, DType::Float32, Device::cpu());
    auto var    = tenzor::ones({C}, DType::Float32, Device::cpu());
    auto output = tenzor::zeros({1, C, 1, 1}, DType::Float32, Device::cpu());
    exporter.export_batchnorm2d(input, scale, bias, mean, var,
                                /*eps=*/1e-5, output, "bn_out");

    const fs::path proto_path = fs::temp_directory_path() /
        ("onnx_ext_rt_" + std::to_string(::getpid()) + ".onnx");
    const fs::path data_path = proto_path.string() + ".data";

    exporter.export_to_file(proto_path.string(),
                            /*use_external_data=*/std::optional<bool>{true},
                            /*threshold_bytes=*/1ULL << 20);

    ASSERT_TRUE(fs::exists(proto_path));
    ASSERT_TRUE(fs::exists(data_path));

    // Importer side: parse the file and check that the initializers
    // round-tripped correctly. Pre-fix the importer ignored external_data
    // and surfaced an "expected N bytes, got 0" exception when downstream
    // code tried to materialise a tensor from empty raw_data.
    ONNXImporter importer;
    auto model = importer.import_from_file(proto_path.string());
    const auto& graph = importer.get_model_data().graph;

    // BN export adds 4 initializers (scale, bias, mean, var). Each is
    // 4 MiB of Float32 == C floats.
    EXPECT_EQ(graph.initializers.size(), 4u)
        << "Importer dropped or duplicated initializers from the round-trip";

    bool found_at_least_one_loaded = false;
    for (const auto& [name, init] : graph.initializers) {
        EXPECT_EQ(init.raw_data.size(),
                  static_cast<size_t>(C * sizeof(float)))
            << "Initializer '" << name << "' raw_data size mismatch — "
               "external_data was not loaded (pre-Fix#1 bug)";
        if (init.raw_data.size() == static_cast<size_t>(C * sizeof(float))) {
            found_at_least_one_loaded = true;
        }
    }
    EXPECT_TRUE(found_at_least_one_loaded)
        << "No initializers were loaded from the sidecar — importer ignored "
           "data_location=EXTERNAL";

    fs::remove(proto_path);
    fs::remove(data_path);
}

TEST(ONNXAuditFixes, ImporterRejectsPathTraversalInExternalData) {
    // Hand-build a TensorProto pointing the sidecar at "../../../etc/passwd".
    // The importer must refuse to open it.
    using namespace tenzor::onnx;
    namespace fs = std::filesystem;

    ONNXExporter exporter(/*opset=*/18);
    auto t = tenzor::ones({1024}, DType::Float32, Device::cpu());
    // Force-emit through BN (small initializers, so the auto path won't
    // externalise — but use_external_data=true threshold=0 forces it).
    auto in     = tenzor::zeros({1, 1024, 1, 1}, DType::Float32, Device::cpu());
    auto bias   = tenzor::zeros({1024}, DType::Float32, Device::cpu());
    auto mean   = tenzor::zeros({1024}, DType::Float32, Device::cpu());
    auto var    = tenzor::ones({1024},  DType::Float32, Device::cpu());
    auto out    = tenzor::zeros({1, 1024, 1, 1}, DType::Float32, Device::cpu());
    exporter.export_batchnorm2d(in, t, bias, mean, var, /*eps=*/1e-5, out, "y");

    const fs::path proto_path = fs::temp_directory_path() /
        ("onnx_pt_" + std::to_string(::getpid()) + ".onnx");
    const fs::path data_path = proto_path.string() + ".data";
    exporter.export_to_file(proto_path.string(),
                            /*use_external_data=*/std::optional<bool>{true},
                            /*threshold_bytes=*/0);  // force all-external

    // Hand-edit the proto to replace the sidecar filename with a relative
    // traversal. We rely on the basename appearing as a length-prefixed
    // string inside the proto; replace its content in-place with a string
    // of equal length but the leading characters set to "../".
    std::ifstream in_f(proto_path, std::ios::binary);
    std::string blob((std::istreambuf_iterator<char>(in_f)),
                      std::istreambuf_iterator<char>());
    in_f.close();
    const std::string needle = data_path.filename().string();
    auto pos = blob.find(needle);
    ASSERT_NE(pos, std::string::npos)
        << "Could not locate sidecar basename in proto blob";
    // Patch: "../X" of the same length so the protobuf wire encoding stays
    // valid (length prefix unchanged).
    blob[pos]     = '.';
    blob[pos + 1] = '.';
    blob[pos + 2] = '/';
    std::ofstream out_f(proto_path, std::ios::binary | std::ios::trunc);
    out_f.write(blob.data(), static_cast<std::streamsize>(blob.size()));
    out_f.close();

    ONNXImporter importer;
    EXPECT_THROW(importer.import_from_file(proto_path.string()),
                 std::runtime_error)
        << "Importer should reject external_data locations containing '..' "
           "(path-traversal hardening)";

    fs::remove(proto_path);
    fs::remove(data_path);
}

// =========================================================================
// 7th-audit Fix #2: importer rejects absolute external_data location.
// =========================================================================
TEST(ONNXAuditFixes, ImporterRejectsAbsolutePathInExternalData) {
    using namespace tenzor::onnx;
    namespace fs = std::filesystem;

    // Same setup as the path-traversal test, but patch the location to
    // an absolute path. Pre-Fix#2 std::filesystem::path::operator/ would
    // silently REPLACE the base anchor, allowing any-file read.
    ONNXExporter exporter(/*opset=*/18);
    auto in     = tenzor::zeros({1, 1024, 1, 1}, DType::Float32, Device::cpu());
    auto scale  = tenzor::ones({1024},  DType::Float32, Device::cpu());
    auto bias   = tenzor::zeros({1024}, DType::Float32, Device::cpu());
    auto mean   = tenzor::zeros({1024}, DType::Float32, Device::cpu());
    auto var    = tenzor::ones({1024},  DType::Float32, Device::cpu());
    auto out    = tenzor::zeros({1, 1024, 1, 1}, DType::Float32, Device::cpu());
    exporter.export_batchnorm2d(in, scale, bias, mean, var,
                                /*eps=*/1e-5, out, "y");

    const fs::path proto_path = fs::temp_directory_path() /
        ("onnx_abs_" + std::to_string(::getpid()) + ".onnx");
    const fs::path data_path = proto_path.string() + ".data";
    exporter.export_to_file(proto_path.string(),
                            /*use_external_data=*/std::optional<bool>{true},
                            /*threshold_bytes=*/0);

    // Patch the proto: find the sidecar basename and rewrite it to begin
    // with a '/' (POSIX absolute). Replace exactly the basename length so
    // the proto length-prefix stays valid.
    std::ifstream in_f(proto_path, std::ios::binary);
    std::string blob((std::istreambuf_iterator<char>(in_f)),
                      std::istreambuf_iterator<char>());
    in_f.close();
    const std::string needle = data_path.filename().string();
    auto pos = blob.find(needle);
    ASSERT_NE(pos, std::string::npos);
    // Patch to "/tmp/X" (same length as the basename — we just overwrite
    // the leading character with '/'; rest of the bytes remain valid path
    // characters). Importer's absolute-path check fires before any open.
    blob[pos] = '/';
    std::ofstream out_f(proto_path, std::ios::binary | std::ios::trunc);
    out_f.write(blob.data(), static_cast<std::streamsize>(blob.size()));
    out_f.close();

    ONNXImporter importer;
    EXPECT_THROW(importer.import_from_file(proto_path.string()),
                 std::runtime_error)
        << "Importer must reject absolute paths in external_data location "
           "(7th-audit Fix #2 — std::filesystem operator/ replaces base "
           "when RHS is absolute, bypassing the anchor)";

    fs::remove(proto_path);
    fs::remove(data_path);
}

// =========================================================================
// 7th-audit Fix #3: external_data_dir_ does not leak across calls.
// =========================================================================
TEST(ONNXAuditFixes, ImporterClearsExternalDataDirOnBytesEntry) {
    using namespace tenzor::onnx;
    namespace fs = std::filesystem;

    // Export a model WITH external_data so the .onnx file references a
    // sidecar relative to its directory.
    ONNXExporter exporter(/*opset=*/18);
    const int64_t C = (4 * 1024 * 1024) / 4;
    auto in     = tenzor::zeros({1, C, 1, 1}, DType::Float32, Device::cpu());
    auto scale  = tenzor::ones({C},  DType::Float32, Device::cpu());
    auto bias   = tenzor::zeros({C}, DType::Float32, Device::cpu());
    auto mean   = tenzor::zeros({C}, DType::Float32, Device::cpu());
    auto var    = tenzor::ones({C},  DType::Float32, Device::cpu());
    auto out    = tenzor::zeros({1, C, 1, 1}, DType::Float32, Device::cpu());
    exporter.export_batchnorm2d(in, scale, bias, mean, var,
                                /*eps=*/1e-5, out, "y");

    const fs::path proto_path = fs::temp_directory_path() /
        ("onnx_dir_leak_" + std::to_string(::getpid()) + ".onnx");
    const fs::path data_path = proto_path.string() + ".data";
    exporter.export_to_file(proto_path.string(),
                            /*use_external_data=*/std::optional<bool>{true},
                            /*threshold_bytes=*/1ULL << 20);
    ASSERT_TRUE(fs::exists(data_path));

    ONNXImporter importer;
    // First import via file path — sets the anchor.
    ASSERT_NO_THROW(importer.import_from_file(proto_path.string()));

    // Now slurp the same bytes and import via the bytes entry point.
    // Pre-Fix#3 the leftover external_data_dir_ would silently resolve
    // the sidecar against the previous file's directory. Post-fix the
    // anchor is cleared and the call must throw "no anchor".
    std::ifstream f(proto_path, std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());
    f.close();

    EXPECT_THROW(importer.import_from_bytes(bytes), std::runtime_error)
        << "import_from_bytes inherited an external_data anchor from a "
           "prior import_from_file call (7th-audit Fix #3 regression)";

    fs::remove(proto_path);
    fs::remove(data_path);
}

// =========================================================================
// C7: JIT-traced module export must not serialize each weight twice.
//
// export_module() emits every module parameter/buffer as a named initializer
// up-front, and convert_jit_graph_to_onnx() then maps the same tensors when
// they reappear as captured graph constants. Before the dedup fix, the
// constants loop created a SECOND `const_N` initializer per weight, doubling
// the .onnx file size. Guard: initializer count equals the parameter/buffer
// count, and every node input still resolves (no dangling references).
// =========================================================================
TEST(ONNXAuditFixes, JitExportDoesNotDuplicateWeights) {
    auto model = std::make_shared<nn::Linear>(4, 3, /*bias=*/true);

    size_t expected = 0;
    for (const auto& [name, p] : model->named_parameters()) {
        if (p && p->is_initialized() && p->tensor().numel() > 0) ++expected;
    }
    for (const auto& [name, b] : model->named_buffers()) {
        if (b && b->is_initialized() && b->tensor().numel() > 0) ++expected;
    }
    ASSERT_GT(expected, 0u);

    ONNXExporter exporter(/*opset=*/17);
    Tensor input = tenzor::randn({2, 4});
    const auto tmp = std::filesystem::temp_directory_path() /
                     "tenzor_jit_export_dedup.onnx";
    exporter.export_module(*model, input, tmp.string());

    const auto& g = exporter.get_graph();
    EXPECT_EQ(g.initializers.size(), expected)
        << "Each weight was serialized more than once — the JIT constants "
           "loop duplicated the up-front named initializers";

    // No node input may dangle after dedup: it must be a graph input, an
    // initializer, or another node's output.
    std::vector<std::string> known;
    for (const auto& in : g.inputs) known.push_back(in.name);
    for (const auto& init : g.initializers) known.push_back(init.name);
    for (const auto& n : g.nodes)
        for (const auto& out : n.outputs) known.push_back(out);
    for (const auto& n : g.nodes) {
        for (const auto& in : n.inputs) {
            if (in.empty()) continue;
            EXPECT_NE(std::find(known.begin(), known.end(), in), known.end())
                << "Node '" << n.name << "' references unknown input '" << in
                << "' after weight dedup";
        }
    }

    std::filesystem::remove(tmp);
}

// =========================================================================
// SECURITY: per-axis Conv/Pool attribute vectors must be length-validated
// before being indexed, or a crafted model triggers an out-of-bounds
// std::vector::operator[] read. We hand-build malformed ModelProtos here.
// =========================================================================
#ifdef TENZOR_HAS_ONNX_PROTOBUF
#include "onnx.pb.h"

namespace {
// Serialize a single-node graph whose op carries the given INTS attribute(s)
// to a byte buffer the importer can ingest. The weight initializer fixes the
// spatial rank the importer infers (rank-4 weight => spatial_dims == 2).
std::vector<uint8_t> serialize_model(const tenzor_onnx::ModelProto& m) {
    std::string s;
    m.SerializeToString(&s);
    return std::vector<uint8_t>(s.begin(), s.end());
}

// Add a rank-4 Float32 weight initializer named `name` with the given shape.
void add_weight_initializer(tenzor_onnx::GraphProto* g, const std::string& name,
                            std::vector<int64_t> shape) {
    auto* init = g->add_initializer();
    init->set_name(name);
    init->set_data_type(tenzor_onnx::TensorProto::FLOAT);
    int64_t numel = 1;
    for (auto d : shape) { init->add_dims(d); numel *= d; }
    std::string raw(static_cast<size_t>(numel) * sizeof(float), '\0');
    init->set_raw_data(raw);
}

void add_ints_attr(tenzor_onnx::NodeProto* n, const std::string& name,
                   std::vector<int64_t> vals) {
    auto* a = n->add_attribute();
    a->set_name(name);
    a->set_type(tenzor_onnx::AttributeProto::INTS);
    for (auto v : vals) a->add_ints(v);
}
}  // namespace

TEST(ONNXAuditFixes, ConvRejectsShortStridesAttribute) {
    // 2-D conv (rank-4 weight) but strides has only ONE entry. Pre-fix the
    // importer read strides[1] past the end (UB); post-fix it must throw.
    tenzor_onnx::ModelProto m;
    m.set_ir_version(7);
    auto* op = m.add_opset_import();
    op->set_domain("");
    op->set_version(18);
    auto* g = m.mutable_graph();
    g->set_name("malformed_conv");

    auto* node = g->add_node();
    node->set_op_type("Conv");
    node->add_input("x");
    node->add_input("w");
    node->add_output("y");
    add_ints_attr(node, "strides", {1});          // too short: needs 2
    add_weight_initializer(g, "w", {4, 4, 3, 3}); // spatial_dims == 2

    ONNXImporter importer;
    EXPECT_THROW(importer.import_from_bytes(serialize_model(m)),
                 std::runtime_error)
        << "Conv with a 1-entry strides attribute on a rank-4 weight must be "
           "rejected, not indexed out of bounds";
}

TEST(ONNXAuditFixes, ConvRejectsShortDilationsAttribute) {
    tenzor_onnx::ModelProto m;
    m.set_ir_version(7);
    auto* op = m.add_opset_import();
    op->set_domain("");
    op->set_version(18);
    auto* g = m.mutable_graph();
    auto* node = g->add_node();
    node->set_op_type("Conv");
    node->add_input("x");
    node->add_input("w");
    node->add_output("y");
    add_ints_attr(node, "dilations", {1});        // too short: needs 2
    add_weight_initializer(g, "w", {4, 4, 3, 3});

    ONNXImporter importer;
    EXPECT_THROW(importer.import_from_bytes(serialize_model(m)),
                 std::runtime_error);
}

TEST(ONNXAuditFixes, ConvTransposeRejectsShortKernelShape) {
    // kernel_shape from get_ints() has NO size default, so a 1-entry value on a
    // rank-4 weight would be indexed [0],[1] out of bounds.
    tenzor_onnx::ModelProto m;
    m.set_ir_version(7);
    auto* op = m.add_opset_import();
    op->set_domain("");
    op->set_version(18);
    auto* g = m.mutable_graph();
    auto* node = g->add_node();
    node->set_op_type("ConvTranspose");
    node->add_input("x");
    node->add_input("w");
    node->add_output("y");
    add_ints_attr(node, "kernel_shape", {3});     // too short: needs 2
    add_weight_initializer(g, "w", {4, 4, 3, 3}); // [in, out/groups, kH, kW]

    ONNXImporter importer;
    EXPECT_THROW(importer.import_from_bytes(serialize_model(m)),
                 std::runtime_error);
}

TEST(ONNXAuditFixes, MaxPoolRejectsShortStridesAttribute) {
    // 2-D MaxPool: kernel_shape has 2 entries but strides has 1, so strides[1]
    // would read past the end.
    tenzor_onnx::ModelProto m;
    m.set_ir_version(7);
    auto* op = m.add_opset_import();
    op->set_domain("");
    op->set_version(18);
    auto* g = m.mutable_graph();
    auto* node = g->add_node();
    node->set_op_type("MaxPool");
    node->add_input("x");
    node->add_output("y");
    add_ints_attr(node, "kernel_shape", {2, 2});
    add_ints_attr(node, "strides", {1});          // too short: needs 2

    ONNXImporter importer;
    EXPECT_THROW(importer.import_from_bytes(serialize_model(m)),
                 std::runtime_error)
        << "MaxPool with a 1-entry strides attribute on a 2-D kernel must be "
           "rejected, not indexed out of bounds";
}
#endif  // TENZOR_HAS_ONNX_PROTOBUF
