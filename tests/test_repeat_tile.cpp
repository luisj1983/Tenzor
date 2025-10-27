#include <gtest/gtest.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include <iostream>
#include <vector>

using namespace tenzor;

// Helper function to print tensor values
template<typename T>
void print_tensor(const Tensor& t, const std::string& name) {
    std::cout << name << " (shape: [";
    auto shape = t.shape();
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << shape[i];
    }
    std::cout << "]): ";

    const T* data = t.data<T>();
    int64_t numel = t.numel();
    std::cout << "[";
    for (int64_t i = 0; i < std::min(numel, int64_t(20)); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << data[i];
    }
    if (numel > 20) std::cout << ", ...";
    std::cout << "]" << std::endl;
}

TEST(RepeatTileTest, Repeat1D) {
    // Test repeat on 1D tensor: [1, 2, 3] with repeats={2}
    // Expected: [1, 1, 2, 2, 3, 3]
    auto input = from_data<float>(std::vector<float>{1.0f, 2.0f, 3.0f}.data(), {3});
    auto output = repeat(input, {2});

    print_tensor<float>(input, "Input");
    print_tensor<float>(output, "Output (repeat)");

    ASSERT_EQ(output.shape()[0], 6);

    const float* data = output.data<float>();
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 1.0f);
    EXPECT_FLOAT_EQ(data[2], 2.0f);
    EXPECT_FLOAT_EQ(data[3], 2.0f);
    EXPECT_FLOAT_EQ(data[4], 3.0f);
    EXPECT_FLOAT_EQ(data[5], 3.0f);
}

TEST(RepeatTileTest, Tile1D) {
    // Test tile on 1D tensor: [1, 2, 3] with reps={2}
    // Expected: [1, 2, 3, 1, 2, 3]
    auto input = from_data<float>(std::vector<float>{1.0f, 2.0f, 3.0f}.data(), {3});
    auto output = tile(input, {2});

    print_tensor<float>(input, "Input");
    print_tensor<float>(output, "Output (tile)");

    ASSERT_EQ(output.shape()[0], 6);

    const float* data = output.data<float>();
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 2.0f);
    EXPECT_FLOAT_EQ(data[2], 3.0f);
    EXPECT_FLOAT_EQ(data[3], 1.0f);
    EXPECT_FLOAT_EQ(data[4], 2.0f);
    EXPECT_FLOAT_EQ(data[5], 3.0f);
}

TEST(RepeatTileTest, Repeat2D) {
    // Test repeat on 2D tensor: [[1, 2], [3, 4]] with repeats={2, 3}
    // Each element repeated 2 times in dim 0, 3 times in dim 1
    std::vector<float> data_vec = {1.0f, 2.0f, 3.0f, 4.0f};
    auto input = from_data<float>(data_vec.data(), {2, 2});
    auto output = repeat(input, {2, 3});

    print_tensor<float>(input, "Input");
    print_tensor<float>(output, "Output (repeat)");

    ASSERT_EQ(output.shape()[0], 4);  // 2 * 2
    ASSERT_EQ(output.shape()[1], 6);  // 2 * 3

    const float* out_data = output.data<float>();
    // First row repeated: [1, 1, 1, 2, 2, 2]
    EXPECT_FLOAT_EQ(out_data[0], 1.0f);
    EXPECT_FLOAT_EQ(out_data[1], 1.0f);
    EXPECT_FLOAT_EQ(out_data[2], 1.0f);
    EXPECT_FLOAT_EQ(out_data[3], 2.0f);
    EXPECT_FLOAT_EQ(out_data[4], 2.0f);
    EXPECT_FLOAT_EQ(out_data[5], 2.0f);
}

TEST(RepeatTileTest, Tile2D) {
    // Test tile on 2D tensor: [[1, 2], [3, 4]] with reps={2, 3}
    // Entire tensor tiled 2 times in dim 0, 3 times in dim 1
    std::vector<float> data_vec = {1.0f, 2.0f, 3.0f, 4.0f};
    auto input = from_data<float>(data_vec.data(), {2, 2});
    auto output = tile(input, {2, 3});

    print_tensor<float>(input, "Input");
    print_tensor<float>(output, "Output (tile)");

    ASSERT_EQ(output.shape()[0], 4);  // 2 * 2
    ASSERT_EQ(output.shape()[1], 6);  // 2 * 3

    const float* out_data = output.data<float>();
    // First row tiled: [1, 2, 1, 2, 1, 2]
    EXPECT_FLOAT_EQ(out_data[0], 1.0f);
    EXPECT_FLOAT_EQ(out_data[1], 2.0f);
    EXPECT_FLOAT_EQ(out_data[2], 1.0f);
    EXPECT_FLOAT_EQ(out_data[3], 2.0f);
    EXPECT_FLOAT_EQ(out_data[4], 1.0f);
    EXPECT_FLOAT_EQ(out_data[5], 2.0f);
}

TEST(RepeatTileTest, TileWithBroadcast) {
    // Test tile with broadcasting: [1, 2, 3] with reps={2, 1}
    // Should expand to shape (2, 3)
    auto input = from_data<float>(std::vector<float>{1.0f, 2.0f, 3.0f}.data(), {3});
    auto output = tile(input, {2, 1});

    print_tensor<float>(input, "Input");
    print_tensor<float>(output, "Output (tile with broadcast)");

    ASSERT_EQ(output.shape()[0], 2);
    ASSERT_EQ(output.shape()[1], 3);

    const float* out_data = output.data<float>();
    // First row: [1, 2, 3]
    EXPECT_FLOAT_EQ(out_data[0], 1.0f);
    EXPECT_FLOAT_EQ(out_data[1], 2.0f);
    EXPECT_FLOAT_EQ(out_data[2], 3.0f);
    // Second row: [1, 2, 3]
    EXPECT_FLOAT_EQ(out_data[3], 1.0f);
    EXPECT_FLOAT_EQ(out_data[4], 2.0f);
    EXPECT_FLOAT_EQ(out_data[5], 3.0f);
}

TEST(RepeatTileTest, RepeatPartialDimensions) {
    // Test repeat with fewer repeats than dimensions
    // Input shape: (2, 3), repeats: {2} should apply only to last dimension
    std::vector<float> data_vec = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto input = from_data<float>(data_vec.data(), {2, 3});
    auto output = repeat(input, {2});

    print_tensor<float>(input, "Input");
    print_tensor<float>(output, "Output (repeat partial)");

    ASSERT_EQ(output.shape()[0], 2);
    ASSERT_EQ(output.shape()[1], 6);  // 3 * 2
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
