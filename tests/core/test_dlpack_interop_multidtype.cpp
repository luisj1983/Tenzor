// Multi-backend multi-dtype tests for DLPack import/export.
//
// Verifies that the DLPack round-trip (Tensor -> DLManagedTensor -> Tensor)
// preserves metadata and data across backends and dtypes, and that
// cross-device compatibility is verified.

#include "../multi_backend_dtype_fixture.hpp"

#include <tenzor/core/dlpack.hpp>
#include <tenzor/ops/creation.hpp>

#include <atomic>

namespace tenzor {
namespace testing {

class DLPackInteropMultiDTypeTest : public MultiBackendDTypeTest {};

// ---------------------------------------------------------------------------
// Export -> Import round trip on the parameterized device/dtype.
// The imported tensor must have the same shape, dtype, and device.
// ---------------------------------------------------------------------------

TEST_P(DLPackInteropMultiDTypeTest, RoundtripPreservesMetadata) {
    auto original = tenzor::zeros({3, 4}, dtype(), device());

    DLManagedTensor* managed = to_dlpack(original);
    ASSERT_NE(managed, nullptr);
    ASSERT_EQ(managed->dl_tensor.ndim, 2);
    ASSERT_EQ(managed->dl_tensor.shape[0], 3);
    ASSERT_EQ(managed->dl_tensor.shape[1], 4);

    Tensor imported = from_dlpack(managed);
    EXPECT_EQ(imported.shape().size(), 2u);
    EXPECT_EQ(imported.shape()[0], 3);
    EXPECT_EQ(imported.shape()[1], 4);
    EXPECT_EQ(imported.dtype(), dtype());
    EXPECT_EQ(imported.device().type, device().type);
}

// ---------------------------------------------------------------------------
// Round-trip preserves data values (zero-copy on same device)
// ---------------------------------------------------------------------------

TEST_P(DLPackInteropMultiDTypeTest, RoundtripPreservesData) {
    auto original = tenzor::ones({6}, dtype(), device());

    DLManagedTensor* managed = to_dlpack(original);
    ASSERT_NE(managed, nullptr);

    Tensor imported = from_dlpack(managed);

    // Verify data by converting both to CPU Float32
    auto orig_cpu = original.to(Device::cpu()).to(DType::Float32);
    auto imp_cpu = imported.to(Device::cpu()).to(DType::Float32);

    auto* orig_data = orig_cpu.data<float>();
    auto* imp_data = imp_cpu.data<float>();

    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(imp_data[i], orig_data[i], atol())
            << "Mismatch at index " << i;
    }
}

// ---------------------------------------------------------------------------
// Zero-copy verification: imported tensor shares data with original
// (only for same-device tensors)
// ---------------------------------------------------------------------------

TEST_P(DLPackInteropMultiDTypeTest, ZeroCopyOnSameDevice) {
    auto original = tenzor::zeros({4}, dtype(), device());

    DLManagedTensor* managed = to_dlpack(original);
    Tensor imported = from_dlpack(managed);

    // On same device, should be zero-copy
    EXPECT_EQ(imported.data_ptr(), original.data_ptr());
}

// ---------------------------------------------------------------------------
// DLPack dtype code is correct for the parameterized dtype
// ---------------------------------------------------------------------------

TEST_P(DLPackInteropMultiDTypeTest, DLPackDTypeCodeCorrect) {
    auto original = tenzor::zeros({2}, dtype(), device());
    DLManagedTensor* managed = to_dlpack(original);
    ASSERT_NE(managed, nullptr);

    auto dl_dtype = managed->dl_tensor.dtype;

    if (dtype() == DType::Float32) {
        EXPECT_EQ(dl_dtype.code, kDLFloat);
        EXPECT_EQ(dl_dtype.bits, 32u);
    } else if (dtype() == DType::Float64) {
        EXPECT_EQ(dl_dtype.code, kDLFloat);
        EXPECT_EQ(dl_dtype.bits, 64u);
    } else if (dtype() == DType::Float16) {
        EXPECT_EQ(dl_dtype.code, kDLFloat);
        EXPECT_EQ(dl_dtype.bits, 16u);
    }

    // Clean up by importing (transfers ownership)
    from_dlpack(managed);
}

// ---------------------------------------------------------------------------
// Non-contiguous import must throw (pinned behavior)
// ---------------------------------------------------------------------------

TEST_P(DLPackInteropMultiDTypeTest, NonContiguousImportThrows) {
    static int64_t shape[] = {2, 3};
    static int64_t bad_strides[] = {1, 2};
    static float buffer[6] = {};

    auto* managed = new DLManagedTensor{};
    managed->dl_tensor.data = buffer;
    managed->dl_tensor.device = DLDevice{kDLCPU, 0};
    managed->dl_tensor.ndim = 2;
    managed->dl_tensor.dtype = DLDataType{kDLFloat, 32, 1};
    managed->dl_tensor.shape = shape;
    managed->dl_tensor.strides = bad_strides;
    managed->dl_tensor.byte_offset = 0;
    managed->manager_ctx = nullptr;
    managed->deleter = [](DLManagedTensor* self) { delete self; };

    EXPECT_THROW(from_dlpack(managed), std::runtime_error);
    managed->deleter(managed);
}

// ---------------------------------------------------------------------------
// Quantized export must throw cleanly
// ---------------------------------------------------------------------------

TEST_P(DLPackInteropMultiDTypeTest, QuantizedExportThrows) {
    auto original = tenzor::zeros({4}, DType::QInt8, Device::cpu());
    EXPECT_THROW(to_dlpack(original), std::runtime_error);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(DLPackInteropMultiDTypeTest);

} // namespace testing
} // namespace tenzor
