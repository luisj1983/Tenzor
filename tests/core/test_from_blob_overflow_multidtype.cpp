// Multi-backend multi-dtype tests for Tensor::from_blob overflow and
// negative-dim validation.
//
// Verifies that from_blob works correctly with various dtypes and that
// overflow/validation checks are consistent across backends.

#include "../multi_backend_dtype_fixture.hpp"

#include <tenzor/core/tensor.hpp>

#include <array>
#include <limits>

namespace tenzor {
namespace testing {

class FromBlobOverflowMultiDTypeTest : public MultiBackendDTypeTest {};

// ---------------------------------------------------------------------------
// Valid from_blob with parameterized dtype on CPU
// (from_blob requires a raw pointer, so we use CPU buffers)
// ---------------------------------------------------------------------------

TEST_P(FromBlobOverflowMultiDTypeTest, ValidShapeWorks) {
    // Allocate buffer large enough for any dtype (12 * 8 bytes)
    alignas(16) char buf[96] = {};
    auto t = Tensor::from_blob(buf, {3, 4}, dtype(), Device::cpu());
    EXPECT_EQ(t.numel(), 12);
    EXPECT_EQ(t.shape()[0], 3);
    EXPECT_EQ(t.shape()[1], 4);
    EXPECT_EQ(t.dtype(), dtype());
}

// ---------------------------------------------------------------------------
// Empty shape allows null data pointer
// ---------------------------------------------------------------------------

TEST_P(FromBlobOverflowMultiDTypeTest, EmptyShapeAllowsNullData) {
    auto t = Tensor::from_blob(nullptr, {0, 4}, dtype(), Device::cpu());
    EXPECT_EQ(t.numel(), 0);
    EXPECT_EQ(t.dtype(), dtype());
}

// ---------------------------------------------------------------------------
// Null data for non-empty tensor must throw
// ---------------------------------------------------------------------------

TEST_P(FromBlobOverflowMultiDTypeTest, NullDataForNonEmptyThrows) {
    EXPECT_THROW(
        Tensor::from_blob(nullptr, {3, 4}, dtype(), Device::cpu()),
        std::runtime_error);
}

// ---------------------------------------------------------------------------
// Negative dimensions must throw
// ---------------------------------------------------------------------------

TEST_P(FromBlobOverflowMultiDTypeTest, NegativeDimThrows) {
    alignas(16) char buf[64] = {};
    EXPECT_THROW(
        Tensor::from_blob(buf, {-1, 4}, dtype(), Device::cpu()),
        std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Element count overflow must throw
// ---------------------------------------------------------------------------

TEST_P(FromBlobOverflowMultiDTypeTest, ElementCountOverflowThrows) {
    alignas(16) char buf[8] = {};
    const int64_t big = std::numeric_limits<int64_t>::max();
    EXPECT_THROW(
        Tensor::from_blob(buf, {big, 2}, dtype(), Device::cpu()),
        std::overflow_error);
}

// ---------------------------------------------------------------------------
// Verify from_blob tensor values can be read back correctly
// ---------------------------------------------------------------------------

TEST_P(FromBlobOverflowMultiDTypeTest, DataReadBack) {
    // Skip Float16 - no direct data<float16>() access pattern
    if (dtype() == DType::Float16) {
        GTEST_SKIP() << "Float16 direct data access not portable";
    }

    // Use Float32 buffer with known values, then verify via dtype conversion
    std::array<float, 4> buf = {1.0f, 2.0f, 3.0f, 4.0f};
    auto t = Tensor::from_blob(buf.data(), {4}, DType::Float32, Device::cpu());

    // Convert to test dtype and back
    auto converted = t.to(dtype());
    EXPECT_EQ(converted.dtype(), dtype());

    auto back = converted.to(Device::cpu()).to(DType::Float32);
    auto* data = back.data<float>();
    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(data[i], static_cast<float>(i + 1), atol());
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(FromBlobOverflowMultiDTypeTest);

} // namespace testing
} // namespace tenzor
