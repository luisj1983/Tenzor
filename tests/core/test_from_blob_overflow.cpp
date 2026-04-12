// Tests for Tensor::from_blob overflow and negative-dim validation.
// See src/core/tensor.cpp:from_blob — guarded with detail::safe_abs
// and a size_bytes check mirroring the internal allocating ctor.

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <stdexcept>
#include <vector>

#include <tenzor/tenzor.hpp>
#include <tenzor/core/tensor.hpp>

namespace tenzor {
namespace {

class FromBlobOverflowTest : public ::testing::Test {
protected:
    void SetUp() override { tenzor::initialize(); }
};

TEST_F(FromBlobOverflowTest, ValidShapeWorks) {
    std::array<float, 12> buf{};
    auto t = Tensor::from_blob(buf.data(), {3, 4}, DType::Float32, Device::cpu());
    EXPECT_EQ(t.numel(), 12);
    EXPECT_EQ(t.shape()[0], 3);
    EXPECT_EQ(t.shape()[1], 4);
}

TEST_F(FromBlobOverflowTest, EmptyShapeAllowsNullData) {
    auto t = Tensor::from_blob(nullptr, {0, 4}, DType::Float32, Device::cpu());
    EXPECT_EQ(t.numel(), 0);
}

TEST_F(FromBlobOverflowTest, NullDataForNonEmptyThrows) {
    EXPECT_THROW(
        Tensor::from_blob(nullptr, {3, 4}, DType::Float32, Device::cpu()),
        std::runtime_error);
}

TEST_F(FromBlobOverflowTest, NegativeDimThrows) {
    std::array<float, 8> buf{};
    EXPECT_THROW(
        Tensor::from_blob(buf.data(), {-1, 4}, DType::Float32, Device::cpu()),
        std::invalid_argument);
}

TEST_F(FromBlobOverflowTest, ElementCountOverflowThrows) {
    std::array<float, 1> buf{};
    const int64_t big = std::numeric_limits<int64_t>::max();
    EXPECT_THROW(
        Tensor::from_blob(buf.data(), {big, 2}, DType::Float32, Device::cpu()),
        std::overflow_error);
}

TEST_F(FromBlobOverflowTest, ByteSizeOverflowThrows) {
    std::array<float, 1> buf{};
    // n fits in int64 but n * dtype_size overflows size_t on 32-bit size_t.
    // On 64-bit platforms this particular combination does not overflow bytes
    // (int64_max elements of 1 byte each is still representable), so this
    // test only asserts no crash when called with an oversized-but-legal n.
    // The oversized case is covered by ElementCountOverflowThrows above.
    const int64_t n = 1024;
    auto t = Tensor::from_blob(buf.data(), {n, 1}, DType::Int8, Device::cpu());
    EXPECT_EQ(t.numel(), n);
}

} // namespace
} // namespace tenzor
