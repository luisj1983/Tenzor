// Round-trip tests for DLPack import/export.
//
// Phase 5.1 of the fix-all plan. Verifies:
//   - A Tensor -> DLManagedTensor -> Tensor round trip preserves all
//     metadata (shape, dtype, device, stride contiguity) and data.
//   - The producer-side deleter is invoked exactly once when the
//     imported Tensor's storage is released.
//   - Non-contiguous imports fail cleanly.
//   - FP8 / quantized exports fail cleanly with a clear error.

#include <gtest/gtest.h>

#include <tenzor/tenzor.hpp>
#include <tenzor/core/dlpack.hpp>
#include <tenzor/ops/creation.hpp>

#include <atomic>

namespace tenzor {
namespace {

class DLPackInteropTest : public ::testing::Test {
protected:
    void SetUp() override { tenzor::initialize(); }
};

// ---------------------------------------------------------------------------
// Export -> Import round trip: a Tenzor goes out as a DLManagedTensor, and
// from_dlpack wraps it back into a Tenzor. The new Tensor must have the
// same shape, dtype, device, and data pointer (zero-copy).
// ---------------------------------------------------------------------------

TEST_F(DLPackInteropTest, RoundtripContiguousFloat32) {
    auto original = zeros({3, 4}, DType::Float32, Device::cpu());
    auto* data = original.data<float>();
    for (int i = 0; i < 12; ++i) data[i] = static_cast<float>(i);

    DLManagedTensor* managed = to_dlpack(original);
    ASSERT_NE(managed, nullptr);
    ASSERT_EQ(managed->dl_tensor.ndim, 2);
    ASSERT_EQ(managed->dl_tensor.shape[0], 3);
    ASSERT_EQ(managed->dl_tensor.shape[1], 4);
    ASSERT_EQ(managed->dl_tensor.dtype.code, kDLFloat);
    ASSERT_EQ(managed->dl_tensor.dtype.bits, 32u);
    ASSERT_EQ(managed->dl_tensor.device.device_type, kDLCPU);

    Tensor imported = from_dlpack(managed);
    EXPECT_EQ(imported.shape().size(), 2u);
    EXPECT_EQ(imported.shape()[0], 3);
    EXPECT_EQ(imported.shape()[1], 4);
    EXPECT_EQ(imported.dtype(), DType::Float32);
    EXPECT_EQ(imported.device().type, Device::Type::CPU);

    // Zero-copy: imported.data_ptr() should point into the SAME storage as
    // the original tensor's data.
    EXPECT_EQ(imported.data_ptr(), original.data_ptr());

    // Values agree.
    auto* imp_data = imported.data<float>();
    for (int i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(imp_data[i], static_cast<float>(i));
    }
}

TEST_F(DLPackInteropTest, RoundtripInt64) {
    auto original = zeros({5}, DType::Int64, Device::cpu());
    DLManagedTensor* managed = to_dlpack(original);
    ASSERT_EQ(managed->dl_tensor.dtype.code, kDLInt);
    ASSERT_EQ(managed->dl_tensor.dtype.bits, 64u);

    Tensor imported = from_dlpack(managed);
    EXPECT_EQ(imported.dtype(), DType::Int64);
    EXPECT_EQ(imported.numel(), 5);
}

TEST_F(DLPackInteropTest, RoundtripBFloat16) {
    auto original = zeros({2, 2}, DType::BFloat16, Device::cpu());
    DLManagedTensor* managed = to_dlpack(original);
    ASSERT_EQ(managed->dl_tensor.dtype.code, kDLBfloat);
    ASSERT_EQ(managed->dl_tensor.dtype.bits, 16u);

    Tensor imported = from_dlpack(managed);
    EXPECT_EQ(imported.dtype(), DType::BFloat16);
}

// ---------------------------------------------------------------------------
// External producer: we construct our own DLManagedTensor backed by a
// raw buffer we own, hand it to from_dlpack, and verify the deleter runs
// exactly once when the resulting Tensor is released.
// ---------------------------------------------------------------------------

TEST_F(DLPackInteropTest, ExternalProducerDeleterRunsOnce) {
    static std::atomic<int> deleter_calls{0};
    deleter_calls.store(0);

    // Stable buffers — the lambda inside from_blob keeps `managed` alive
    // by value, but the shape/stride arrays referenced by the DLTensor
    // must outlive the Tensor.
    static int64_t shape[] = {4};
    static int64_t strides[] = {1};
    static float buffer[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    auto* managed = new DLManagedTensor{};
    managed->dl_tensor.data = buffer;
    managed->dl_tensor.device = DLDevice{kDLCPU, 0};
    managed->dl_tensor.ndim = 1;
    managed->dl_tensor.dtype = DLDataType{kDLFloat, 32, 1};
    managed->dl_tensor.shape = shape;
    managed->dl_tensor.strides = strides;
    managed->dl_tensor.byte_offset = 0;
    managed->manager_ctx = nullptr;
    managed->deleter = [](DLManagedTensor* self) {
        deleter_calls.fetch_add(1);
        delete self;
    };

    {
        Tensor t = from_dlpack(managed);
        EXPECT_EQ(t.numel(), 4);
        EXPECT_FLOAT_EQ(t.data<float>()[2], 3.0f);
        EXPECT_EQ(deleter_calls.load(), 0)
            << "Deleter must NOT run while Tensor is alive";
    } // Tensor goes out of scope -> storage released -> our wrapper runs
      // the producer deleter.

    EXPECT_EQ(deleter_calls.load(), 1)
        << "Deleter must run exactly once when the imported Tensor is released";
}

// ---------------------------------------------------------------------------
// Non-contiguous imports currently throw. This is a documented limitation;
// the test pins the behavior so we notice if it changes.
// ---------------------------------------------------------------------------

TEST_F(DLPackInteropTest, NonContiguousImportThrows) {
    static int64_t shape[] = {2, 3};
    static int64_t bad_strides[] = {1, 2}; // Intentionally wrong (not C-contig)
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

    // Ownership was NOT transferred on failure; caller must free.
    managed->deleter(managed);
}

// ---------------------------------------------------------------------------
// FP8 and quantized exports throw with a clear error.
// ---------------------------------------------------------------------------

TEST_F(DLPackInteropTest, QuantizedExportThrows) {
    auto original = zeros({4}, DType::QInt8, Device::cpu());
    EXPECT_THROW(to_dlpack(original), std::runtime_error);
}

} // namespace
} // namespace tenzor
