/**
 * @file test_lite_runtime.cpp
 * @brief Tests for LiteRuntime and LiteAllocator
 */

#include <gtest/gtest.h>
#include <tenzor/lite/runtime.hpp>
#include <cstdint>

using namespace tenzor::lite;

TEST(LiteRuntimeExpandedTest, AllocatorAlignment64) {
    std::vector<size_t> pools = {256};
    LiteAllocator allocator(pools, 64);

    void* ptr = allocator.get_buffer(0, 0);
    EXPECT_NE(ptr, nullptr);
    // Pointer should be 64-byte aligned
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0u);
}

TEST(LiteRuntimeExpandedTest, AllocatorAlignment16) {
    std::vector<size_t> pools = {128};
    LiteAllocator allocator(pools, 16);

    void* ptr = allocator.get_buffer(0, 0);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 16, 0u);
}

TEST(LiteRuntimeExpandedTest, AllocatorMultiplePools) {
    std::vector<size_t> pools = {512, 1024, 2048};
    LiteAllocator allocator(pools, 64);
    EXPECT_EQ(allocator.total_bytes(), 512 + 1024 + 2048);

    // Each pool should be independently addressable
    void* p0 = allocator.get_buffer(0, 0);
    void* p1 = allocator.get_buffer(1, 0);
    void* p2 = allocator.get_buffer(2, 0);
    EXPECT_NE(p0, p1);
    EXPECT_NE(p1, p2);
    EXPECT_NE(p0, p2);
}

TEST(LiteRuntimeExpandedTest, AllocatorOffsetWithinPool) {
    std::vector<size_t> pools = {1024};
    LiteAllocator allocator(pools, 64);

    void* base = allocator.get_buffer(0, 0);
    void* offset = allocator.get_buffer(0, 128);
    auto diff = static_cast<uint8_t*>(offset) - static_cast<uint8_t*>(base);
    EXPECT_EQ(diff, 128);
}

TEST(LiteRuntimeExpandedTest, RuntimeLoadFromMagic) {
    uint8_t magic[] = {0x54, 0x4C, 0x5A, 0x54};  // "TZLT"
    auto runtime = LiteRuntime::load(magic, sizeof(magic));
    ASSERT_NE(runtime, nullptr);
}

TEST(LiteRuntimeExpandedTest, RuntimeCreateInputShape) {
    uint8_t magic[] = {0x54, 0x4C, 0x5A, 0x54};
    auto runtime = LiteRuntime::load(magic, sizeof(magic));
    ASSERT_NE(runtime, nullptr);

    auto input = runtime->create_input({1, 16});
    EXPECT_EQ(input.ndim, 2);
    EXPECT_EQ(input.shape[0], 1);
    EXPECT_EQ(input.shape[1], 16);
    EXPECT_EQ(input.numel(), 16);
    EXPECT_TRUE(input.owns_data);
    EXPECT_NE(input.data, nullptr);
}

TEST(LiteRuntimeExpandedTest, RuntimeCreateInputDtype) {
    uint8_t magic[] = {0x54, 0x4C, 0x5A, 0x54};
    auto runtime = LiteRuntime::load(magic, sizeof(magic));
    ASSERT_NE(runtime, nullptr);

    auto input_f32 = runtime->create_input({2, 2}, tenzor::DType::Float32);
    EXPECT_EQ(input_f32.dtype, tenzor::DType::Float32);
}

TEST(LiteRuntimeExpandedTest, RuntimeLoadInvalidPath) {
    EXPECT_THROW(LiteRuntime::load("/nonexistent/model.tzlite"), std::runtime_error);
}

TEST(LiteRuntimeExpandedTest, RuntimeLoadNullData) {
    EXPECT_THROW(LiteRuntime::load(nullptr, 0), std::runtime_error);
}

TEST(LiteRuntimeExpandedTest, MaxDimsConstant) {
    EXPECT_EQ(kMaxDims, 8);
}
