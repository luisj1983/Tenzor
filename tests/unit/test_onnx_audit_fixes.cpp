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
