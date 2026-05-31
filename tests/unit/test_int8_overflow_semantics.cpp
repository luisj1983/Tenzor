/**
 * @file test_int8_overflow_semantics.cpp
 * @brief Verify Int8/UInt8 arithmetic wraps on overflow (PyTorch-compatible).
 *
 * PyTorch: int8(100) + int8(100) = -56 (wrap), int8(100) * int8(2) = -56 (wrap).
 * All of add/sub/mul must use the same semantics — two's-complement wrap.
 */

#include <gtest/gtest.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include "../backend_test_fixture.hpp"

namespace tz = ::tenzor;

class Int8OverflowSemantics : public ::tenzor::testing::BackendTest {};
class UInt8OverflowSemantics : public ::tenzor::testing::BackendTest {};

TEST_P(Int8OverflowSemantics, AddWraps) {
    auto a = tz::full({16}, 100.0, tz::DType::Int8).to(device);
    auto b = tz::full({16}, 100.0, tz::DType::Int8).to(device);
    auto c = a + b;
    auto host = c.cpu().to(tz::DType::Int32);
    auto p = host.data<int32_t>();
    for (int i = 0; i < 16; ++i) EXPECT_EQ(p[i], -56) << "int8 add not wrapping at i=" << i;
}

TEST_P(Int8OverflowSemantics, MulWraps) {
    auto a = tz::full({16}, 100.0, tz::DType::Int8).to(device);
    auto b = tz::full({16},   2.0, tz::DType::Int8).to(device);
    auto c = a * b;  // 100 * 2 = 200 -> wrap to -56
    auto host = c.cpu().to(tz::DType::Int32);
    auto p = host.data<int32_t>();
    for (int i = 0; i < 16; ++i) EXPECT_EQ(p[i], -56) << "int8 mul not wrapping at i=" << i;
}

TEST_P(Int8OverflowSemantics, SubWraps) {
    auto a = tz::full({16}, 100.0, tz::DType::Int8).to(device);
    auto b = tz::full({16}, -100.0, tz::DType::Int8).to(device);
    auto c = a - b;  // 100 - (-100) = 200 -> wrap to -56
    auto host = c.cpu().to(tz::DType::Int32);
    auto p = host.data<int32_t>();
    for (int i = 0; i < 16; ++i) EXPECT_EQ(p[i], -56);
}

TEST_P(UInt8OverflowSemantics, AddWraps) {
    auto a = tz::full({16}, 200.0, tz::DType::UInt8).to(device);
    auto b = tz::full({16}, 200.0, tz::DType::UInt8).to(device);
    auto c = a + b;  // 200+200 = 400 -> wrap to 144 (400 mod 256 = 144)
    auto host = c.cpu().to(tz::DType::Int32);
    auto p = host.data<int32_t>();
    for (int i = 0; i < 16; ++i) EXPECT_EQ(p[i], 144);
}

TEST_P(UInt8OverflowSemantics, MulWraps) {
    // 200 * 2 = 400 -> wrap to 144
    auto a = tz::full({16}, 200.0, tz::DType::UInt8).to(device);
    auto b = tz::full({16},   2.0, tz::DType::UInt8).to(device);
    auto c = a * b;
    auto host = c.cpu().to(tz::DType::Int32);
    auto p = host.data<int32_t>();
    for (int i = 0; i < 16; ++i) EXPECT_EQ(p[i], 144) << "uint8 mul not wrapping at i=" << i;
}

// Larger tensor (>32 elements) to exercise the AVX2 SIMD path
TEST_P(Int8OverflowSemantics, MulWrapsAVX2Path) {
    auto a = tz::full({64}, 100.0, tz::DType::Int8).to(device);
    auto b = tz::full({64},   2.0, tz::DType::Int8).to(device);
    auto c = a * b;
    auto host = c.cpu().to(tz::DType::Int32);
    auto p = host.data<int32_t>();
    for (int i = 0; i < 64; ++i) EXPECT_EQ(p[i], -56) << "int8 mul SIMD path not wrapping at i=" << i;
}

TEST_P(Int8OverflowSemantics, AddWrapsAVX2Path) {
    auto a = tz::full({64}, 100.0, tz::DType::Int8).to(device);
    auto b = tz::full({64}, 100.0, tz::DType::Int8).to(device);
    auto c = a + b;
    auto host = c.cpu().to(tz::DType::Int32);
    auto p = host.data<int32_t>();
    for (int i = 0; i < 64; ++i) EXPECT_EQ(p[i], -56) << "int8 add SIMD path not wrapping at i=" << i;
}

INSTANTIATE_BACKEND_TESTS(Int8OverflowSemantics);
INSTANTIATE_BACKEND_TESTS(UInt8OverflowSemantics);
