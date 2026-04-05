/**
 * @file test_lite_ops.cpp
 * @brief Tests for lite runtime (LiteTensor, LiteAllocator, LiteRuntime)
 */

#include <gtest/gtest.h>
#include <tenzor/lite/runtime.hpp>
#include <cstring>

using namespace tenzor::lite;

TEST(LiteTensorTest, DefaultConstruction) {
    LiteTensor tensor;
    EXPECT_EQ(tensor.data, nullptr);
    EXPECT_EQ(tensor.ndim, 0);
    EXPECT_EQ(tensor.dtype, tenzor::DType::Float32);
    EXPECT_FALSE(tensor.owns_data);
}

TEST(LiteTensorTest, Numel) {
    LiteTensor tensor;
    tensor.ndim = 3;
    tensor.shape[0] = 2;
    tensor.shape[1] = 3;
    tensor.shape[2] = 4;
    EXPECT_EQ(tensor.numel(), 24);
}

TEST(LiteTensorTest, NumelEmpty) {
    LiteTensor tensor;
    tensor.ndim = 0;
    EXPECT_EQ(tensor.numel(), 0);
}

TEST(LiteAllocatorTest, Construction) {
    std::vector<size_t> pools = {1024, 2048};
    LiteAllocator allocator(pools, 64);
    EXPECT_EQ(allocator.total_bytes(), 3072);
}

TEST(LiteAllocatorTest, GetBuffer) {
    std::vector<size_t> pools = {1024};
    LiteAllocator allocator(pools, 64);

    void* ptr = allocator.get_buffer(0, 0);
    EXPECT_NE(ptr, nullptr);

    void* ptr2 = allocator.get_buffer(0, 512);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_NE(ptr, ptr2);
}

TEST(LiteAllocatorTest, OutOfRange) {
    std::vector<size_t> pools = {1024};
    LiteAllocator allocator(pools, 64);

    EXPECT_THROW(allocator.get_buffer(5, 0), std::out_of_range);
}

TEST(LiteAllocatorTest, EmptyPools) {
    std::vector<size_t> pools = {};
    LiteAllocator allocator(pools, 64);
    EXPECT_EQ(allocator.total_bytes(), 0);
}

TEST(LiteRuntimeTest, LoadFromInvalidPath) {
    EXPECT_THROW(LiteRuntime::load("/nonexistent/path.tzlite"), std::runtime_error);
}

TEST(LiteRuntimeTest, LoadFromNullData) {
    EXPECT_THROW(LiteRuntime::load(nullptr, 0), std::runtime_error);
}

TEST(LiteRuntimeTest, CreateInput) {
    // Create a dummy runtime from minimal data
    uint8_t dummy_data[] = {0x54, 0x4C, 0x5A, 0x54};  // "TZLT" magic
    auto runtime = LiteRuntime::load(dummy_data, sizeof(dummy_data));
    ASSERT_NE(runtime, nullptr);

    auto input = runtime->create_input({1, 3, 224, 224});
    EXPECT_NE(input.data, nullptr);
    EXPECT_EQ(input.ndim, 4);
    EXPECT_EQ(input.shape[0], 1);
    EXPECT_EQ(input.shape[1], 3);
    EXPECT_EQ(input.shape[2], 224);
    EXPECT_EQ(input.shape[3], 224);
    EXPECT_EQ(input.numel(), 1 * 3 * 224 * 224);
    EXPECT_TRUE(input.owns_data);
}

TEST(LiteRuntimeTest, ForwardPassthrough) {
    uint8_t dummy_data[] = {0x54, 0x4C, 0x5A, 0x54};
    auto runtime = LiteRuntime::load(dummy_data, sizeof(dummy_data));
    ASSERT_NE(runtime, nullptr);

    auto input = runtime->create_input({2, 4});
    // Fill with known values
    float* data = input.data_as<float>();
    for (int i = 0; i < 8; ++i) data[i] = static_cast<float>(i);

    auto output = runtime->forward(input);
    EXPECT_EQ(output.numel(), 8);
    // Placeholder forward returns a copy
    float* out_data = output.data_as<float>();
    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(out_data[i], static_cast<float>(i));
    }
}

TEST(LiteRuntimeTest, MaxDims) {
    EXPECT_EQ(kMaxDims, 8);
}
