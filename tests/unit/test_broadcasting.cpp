#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

//==============================================================================
// Broadcasting Tests
//==============================================================================

class BroadcastingTest : public BackendTest {};

TEST_P(BroadcastingTest, AddBroadcast_ScalarToVector) {
    // Test: (3,) + (1,) -> (3,)
    auto a = ones({3}, DType::Float32, device);
    auto b = ones({1}, DType::Float32, device);

    auto a_cpu = a.to(Device::cpu());
    auto b_cpu = b.to(Device::cpu());
    auto a_data = a_cpu.data<float>();
    auto b_data = b_cpu.data<float>();

    a_data[0] = 1.0f; a_data[1] = 2.0f; a_data[2] = 3.0f;
    b_data[0] = 10.0f;

    // Copy modified data back to device
    a = a_cpu.to(device);
    b = b_cpu.to(device);

    auto c = add(a, b);

    EXPECT_EQ(c.shape()[0], 3) << "Failed on " << device.to_string();

    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();
    EXPECT_FLOAT_EQ(c_data[0], 11.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(c_data[1], 12.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(c_data[2], 13.0f) << "Failed on " << device.to_string();
}

TEST_P(BroadcastingTest, AddBroadcast_RowToMatrix) {
    // Test: (2, 3) + (1, 3) -> (2, 3)
    auto a = ones({2, 3}, DType::Float32, device);
    auto b = ones({1, 3}, DType::Float32, device);

    auto a_cpu = a.to(Device::cpu());
    auto b_cpu = b.to(Device::cpu());
    auto a_data = a_cpu.data<float>();
    auto b_data = b_cpu.data<float>();

    // A = [[1, 2, 3], [4, 5, 6]]
    for (int i = 0; i < 6; i++) {
        a_data[i] = static_cast<float>(i + 1);
    }

    // B = [[10, 20, 30]]
    b_data[0] = 10.0f; b_data[1] = 20.0f; b_data[2] = 30.0f;

    // Copy modified data back to device
    a = a_cpu.to(device);
    b = b_cpu.to(device);

    auto c = add(a, b);

    EXPECT_EQ(c.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(c.shape()[1], 3) << "Failed on " << device.to_string();

    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();

    // Expected: [[11, 22, 33], [14, 25, 36]]
    EXPECT_FLOAT_EQ(c_data[0], 11.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(c_data[1], 22.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(c_data[2], 33.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(c_data[3], 14.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(c_data[4], 25.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(c_data[5], 36.0f) << "Failed on " << device.to_string();
}

TEST_P(BroadcastingTest, AddBroadcast_ColumnToMatrix) {
    // Test: (2, 3) + (2, 1) -> (2, 3)
    auto a = ones({2, 3}, DType::Float32, device);
    auto b = ones({2, 1}, DType::Float32, device);

    auto a_cpu = a.to(Device::cpu());
    auto b_cpu = b.to(Device::cpu());
    auto a_data = a_cpu.data<float>();
    auto b_data = b_cpu.data<float>();

    // A = [[1, 2, 3], [4, 5, 6]]
    for (int i = 0; i < 6; i++) {
        a_data[i] = static_cast<float>(i + 1);
    }

    // B = [[100], [200]]
    b_data[0] = 100.0f;
    b_data[1] = 200.0f;

    // Copy modified data back to device
    a = a_cpu.to(device);
    b = b_cpu.to(device);

    auto c = add(a, b);

    EXPECT_EQ(c.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(c.shape()[1], 3) << "Failed on " << device.to_string();

    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();

    // Expected: [[101, 102, 103], [204, 205, 206]]
    EXPECT_FLOAT_EQ(c_data[0], 101.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(c_data[1], 102.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(c_data[2], 103.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(c_data[3], 204.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(c_data[4], 205.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(c_data[5], 206.0f) << "Failed on " << device.to_string();
}

TEST_P(BroadcastingTest, AddBroadcast_ScalarToMatrix) {
    // Test: (2, 3) + (1, 1) -> (2, 3)
    auto a = ones({2, 3}, DType::Float32, device);
    auto b = ones({1, 1}, DType::Float32, device);

    auto a_cpu = a.to(Device::cpu());
    auto b_cpu = b.to(Device::cpu());
    auto a_data = a_cpu.data<float>();
    auto b_data = b_cpu.data<float>();

    for (int i = 0; i < 6; i++) {
        a_data[i] = static_cast<float>(i + 1);
    }

    b_data[0] = 1000.0f;

    // Copy modified data back to device
    a = a_cpu.to(device);
    b = b_cpu.to(device);

    auto c = add(a, b);

    EXPECT_EQ(c.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(c.shape()[1], 3) << "Failed on " << device.to_string();

    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();

    // Expected: [[1001, 1002, 1003], [1004, 1005, 1006]]
    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(c_data[i], 1000.0f + static_cast<float>(i + 1))
            << "Failed on " << device.to_string();
    }
}

TEST_P(BroadcastingTest, AddBroadcast_DifferentDimensions) {
    // Test: (3, 2) + (2,) -> (3, 2)
    auto a = ones({3, 2}, DType::Float32, device);
    auto b = ones({2}, DType::Float32, device);

    auto a_cpu = a.to(Device::cpu());
    auto b_cpu = b.to(Device::cpu());
    auto a_data = a_cpu.data<float>();
    auto b_data = b_cpu.data<float>();

    // A = [[1, 2], [3, 4], [5, 6]]
    for (int i = 0; i < 6; i++) {
        a_data[i] = static_cast<float>(i + 1);
    }

    // B = [10, 20]
    b_data[0] = 10.0f;
    b_data[1] = 20.0f;

    // Copy modified data back to device
    a = a_cpu.to(device);
    b = b_cpu.to(device);

    auto c = add(a, b);

    EXPECT_EQ(c.shape()[0], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(c.shape()[1], 2) << "Failed on " << device.to_string();

    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();

    // Expected: [[11, 22], [13, 24], [15, 26]]
    EXPECT_FLOAT_EQ(c_data[0], 11.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(c_data[1], 22.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(c_data[2], 13.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(c_data[3], 24.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(c_data[4], 15.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(c_data[5], 26.0f) << "Failed on " << device.to_string();
}

TEST_P(BroadcastingTest, AddBroadcast_Int32) {
    // Test broadcasting with Int32
    auto a = ones({2, 3}, DType::Int32, device);
    auto b = ones({1, 3}, DType::Int32, device);

    auto a_cpu = a.to(Device::cpu());
    auto b_cpu = b.to(Device::cpu());
    auto a_data = a_cpu.data<int32_t>();
    auto b_data = b_cpu.data<int32_t>();

    for (int i = 0; i < 6; i++) {
        a_data[i] = i + 1;
    }

    b_data[0] = 10; b_data[1] = 20; b_data[2] = 30;

    // Copy modified data back to device
    a = a_cpu.to(device);
    b = b_cpu.to(device);

    auto c = add(a, b);

    EXPECT_EQ(c.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(c.shape()[1], 3) << "Failed on " << device.to_string();

    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<int32_t>();

    // Expected: [[11, 22, 33], [14, 25, 36]]
    EXPECT_EQ(c_data[0], 11) << "Failed on " << device.to_string();
    EXPECT_EQ(c_data[1], 22) << "Failed on " << device.to_string();
    EXPECT_EQ(c_data[2], 33) << "Failed on " << device.to_string();
    EXPECT_EQ(c_data[3], 14) << "Failed on " << device.to_string();
    EXPECT_EQ(c_data[4], 25) << "Failed on " << device.to_string();
    EXPECT_EQ(c_data[5], 36) << "Failed on " << device.to_string();
}

TEST_P(BroadcastingTest, AddNoBroadcast_SameShape) {
    // Test that same-shape operations still work (fast path)
    auto a = ones({2, 3}, DType::Float32, device);
    auto b = ones({2, 3}, DType::Float32, device);

    auto a_cpu = a.to(Device::cpu());
    auto b_cpu = b.to(Device::cpu());
    auto a_data = a_cpu.data<float>();
    auto b_data = b_cpu.data<float>();

    for (int i = 0; i < 6; i++) {
        a_data[i] = static_cast<float>(i + 1);
        b_data[i] = static_cast<float>((i + 1) * 10);
    }

    // Copy modified data back to device
    a = a_cpu.to(device);
    b = b_cpu.to(device);

    auto c = add(a, b);

    EXPECT_EQ(c.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(c.shape()[1], 3) << "Failed on " << device.to_string();

    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();

    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(c_data[i], static_cast<float>((i + 1) + (i + 1) * 10))
            << "Failed on " << device.to_string();
    }
}

INSTANTIATE_BACKEND_TESTS(BroadcastingTest);

// ============================================================================
// broadcast_strides() unit tests (CPU-only, tests shape utility)
// ============================================================================

#include "tenzor/core/shape.hpp"

namespace {

// Compute strides for a tensor after broadcasting to a target shape.
//
// Given a tensor's original shape and strides, returns the strides that
// correspond to the broadcast shape. Dimensions where the original size was 1
// (broadcast dimensions) get stride 0. This helper previously lived in
// include/tenzor/core/shape.hpp but its only consumer was this test, so it was
// relocated here to shrink the public surface.
static std::vector<int64_t> broadcast_strides(
    std::span<const int64_t> shape,
    std::span<const int64_t> strides,
    std::span<const int64_t> broadcast_shape) {
    size_t ndim = broadcast_shape.size();
    std::vector<int64_t> result(ndim, 0);

    // Use signed arithmetic to avoid size_t underflow when ndim > shape.size()
    ptrdiff_t offset = static_cast<ptrdiff_t>(shape.size()) - static_cast<ptrdiff_t>(ndim);

    for (size_t i = 0; i < ndim; ++i) {
        ptrdiff_t orig_idx = static_cast<ptrdiff_t>(i) + offset;
        if (orig_idx >= 0 && orig_idx < static_cast<ptrdiff_t>(shape.size())) {
            // This dimension exists in the original tensor
            int64_t orig_dim = shape[static_cast<size_t>(orig_idx)];
            if (orig_dim == broadcast_shape[i]) {
                result[i] = strides[static_cast<size_t>(orig_idx)];
            } else {
                // orig_dim must be 1 (broadcast) — stride is 0
                result[i] = 0;
            }
        }
        // Dimensions that don't exist in original (prepended) get stride 0
    }

    return result;
}

}  // namespace

TEST(BroadcastStrides, ScalarBroadcast) {
    // Scalar (shape {1}) broadcast to {4}: stride should be 0
    std::vector<int64_t> shape = {1};
    std::vector<int64_t> strides = {1};
    std::vector<int64_t> target = {4};
    auto result = broadcast_strides(shape, strides, target);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], 0);  // broadcast dim gets stride 0
}

TEST(BroadcastStrides, DimensionExpansion) {
    // Shape {3} broadcast to {4, 3}: prepended dim gets stride 0
    std::vector<int64_t> shape = {3};
    std::vector<int64_t> strides = {1};
    std::vector<int64_t> target = {4, 3};
    auto result = broadcast_strides(shape, strides, target);
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], 0);  // prepended dim
    EXPECT_EQ(result[1], 1);  // original dim kept
}

TEST(BroadcastStrides, MultipleBroadcastDims) {
    // Shape {1, 3, 1} broadcast to {4, 3, 5}
    std::vector<int64_t> shape = {1, 3, 1};
    std::vector<int64_t> strides = {3, 1, 1};
    std::vector<int64_t> target = {4, 3, 5};
    auto result = broadcast_strides(shape, strides, target);
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0], 0);  // dim 0: was 1, broadcast to 4
    EXPECT_EQ(result[1], 1);  // dim 1: was 3, stays 3
    EXPECT_EQ(result[2], 0);  // dim 2: was 1, broadcast to 5
}

TEST(BroadcastStrides, NoBroadcast) {
    // Shape {2, 3} broadcast to {2, 3}: all strides preserved
    std::vector<int64_t> shape = {2, 3};
    std::vector<int64_t> strides = {3, 1};
    std::vector<int64_t> target = {2, 3};
    auto result = broadcast_strides(shape, strides, target);
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], 3);
    EXPECT_EQ(result[1], 1);
}

TEST(BroadcastStrides, PrependMultipleDims) {
    // Shape {3} broadcast to {2, 4, 3}: two prepended dims
    std::vector<int64_t> shape = {3};
    std::vector<int64_t> strides = {1};
    std::vector<int64_t> target = {2, 4, 3};
    auto result = broadcast_strides(shape, strides, target);
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0], 0);
    EXPECT_EQ(result[1], 0);
    EXPECT_EQ(result[2], 1);
}
