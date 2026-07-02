#include <gtest/gtest.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/tenzor.hpp"
#include "backend_test_fixture.hpp"
#include <iostream>
#include <vector>

using namespace tenzor;

// Helper function to print tensor values (reads on host; caller passes a CPU tensor)
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

class RepeatTileTest : public ::tenzor::testing::BackendTest {};
class RepeatInterleaveTest : public ::tenzor::testing::BackendTest {};

TEST_P(RepeatTileTest, Repeat1D) {
    // repeat() is torch.Tensor.repeat (tiling), so [1, 2, 3].repeat(2) tiles the
    // whole vector -> [1, 2, 3, 1, 2, 3] (element-wise repetition is
    // repeat_interleave). For 1D this coincides with tile({2}).
    auto input = from_data<float>(std::vector<float>{1.0f, 2.0f, 3.0f}.data(), {3}, device);
    auto output = repeat(input, {2});

    auto input_cpu = input.cpu();
    auto output_cpu = output.cpu();
    print_tensor<float>(input_cpu, "Input");
    print_tensor<float>(output_cpu, "Output (repeat)");

    ASSERT_EQ(output.shape()[0], 6);

    const float* data = output_cpu.data<float>();
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 2.0f);
    EXPECT_FLOAT_EQ(data[2], 3.0f);
    EXPECT_FLOAT_EQ(data[3], 1.0f);
    EXPECT_FLOAT_EQ(data[4], 2.0f);
    EXPECT_FLOAT_EQ(data[5], 3.0f);
}

TEST_P(RepeatTileTest, Tile1D) {
    // Test tile on 1D tensor: [1, 2, 3] with reps={2}
    // Expected: [1, 2, 3, 1, 2, 3]
    auto input = from_data<float>(std::vector<float>{1.0f, 2.0f, 3.0f}.data(), {3}, device);
    auto output = tile(input, {2});

    auto input_cpu = input.cpu();
    auto output_cpu = output.cpu();
    print_tensor<float>(input_cpu, "Input");
    print_tensor<float>(output_cpu, "Output (tile)");

    ASSERT_EQ(output.shape()[0], 6);

    const float* data = output_cpu.data<float>();
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 2.0f);
    EXPECT_FLOAT_EQ(data[2], 3.0f);
    EXPECT_FLOAT_EQ(data[3], 1.0f);
    EXPECT_FLOAT_EQ(data[4], 2.0f);
    EXPECT_FLOAT_EQ(data[5], 3.0f);
}

TEST_P(RepeatTileTest, Repeat2D) {
    // repeat() implements torch.Tensor.repeat(*sizes): it TILES the whole tensor
    // along each dim (element-wise repetition is repeat_interleave instead). For
    // [[1, 2], [3, 4]] with repeats={2, 3} the (2x3)-tiled result's first row is
    // the row [1, 2] tiled 3 times -> [1, 2, 1, 2, 1, 2].
    std::vector<float> data_vec = {1.0f, 2.0f, 3.0f, 4.0f};
    auto input = from_data<float>(data_vec.data(), {2, 2}, device);
    auto output = repeat(input, {2, 3});

    auto input_cpu = input.cpu();
    auto output_cpu = output.cpu();
    print_tensor<float>(input_cpu, "Input");
    print_tensor<float>(output_cpu, "Output (repeat)");

    ASSERT_EQ(output.shape()[0], 4);  // 2 * 2
    ASSERT_EQ(output.shape()[1], 6);  // 2 * 3

    const float* out_data = output_cpu.data<float>();
    // First row: row [1, 2] tiled 3 times -> [1, 2, 1, 2, 1, 2]
    EXPECT_FLOAT_EQ(out_data[0], 1.0f);
    EXPECT_FLOAT_EQ(out_data[1], 2.0f);
    EXPECT_FLOAT_EQ(out_data[2], 1.0f);
    EXPECT_FLOAT_EQ(out_data[3], 2.0f);
    EXPECT_FLOAT_EQ(out_data[4], 1.0f);
    EXPECT_FLOAT_EQ(out_data[5], 2.0f);
}

TEST_P(RepeatTileTest, Tile2D) {
    // Test tile on 2D tensor: [[1, 2], [3, 4]] with reps={2, 3}
    // Entire tensor tiled 2 times in dim 0, 3 times in dim 1
    std::vector<float> data_vec = {1.0f, 2.0f, 3.0f, 4.0f};
    auto input = from_data<float>(data_vec.data(), {2, 2}, device);
    auto output = tile(input, {2, 3});

    auto input_cpu = input.cpu();
    auto output_cpu = output.cpu();
    print_tensor<float>(input_cpu, "Input");
    print_tensor<float>(output_cpu, "Output (tile)");

    ASSERT_EQ(output.shape()[0], 4);  // 2 * 2
    ASSERT_EQ(output.shape()[1], 6);  // 2 * 3

    const float* out_data = output_cpu.data<float>();
    // First row tiled: [1, 2, 1, 2, 1, 2]
    EXPECT_FLOAT_EQ(out_data[0], 1.0f);
    EXPECT_FLOAT_EQ(out_data[1], 2.0f);
    EXPECT_FLOAT_EQ(out_data[2], 1.0f);
    EXPECT_FLOAT_EQ(out_data[3], 2.0f);
    EXPECT_FLOAT_EQ(out_data[4], 1.0f);
    EXPECT_FLOAT_EQ(out_data[5], 2.0f);
}

TEST_P(RepeatTileTest, TileWithBroadcast) {
    // Test tile with broadcasting: [1, 2, 3] with reps={2, 1}
    // Should expand to shape (2, 3)
    auto input = from_data<float>(std::vector<float>{1.0f, 2.0f, 3.0f}.data(), {3}, device);
    auto output = tile(input, {2, 1});

    auto input_cpu = input.cpu();
    auto output_cpu = output.cpu();
    print_tensor<float>(input_cpu, "Input");
    print_tensor<float>(output_cpu, "Output (tile with broadcast)");

    ASSERT_EQ(output.shape()[0], 2);
    ASSERT_EQ(output.shape()[1], 3);

    const float* out_data = output_cpu.data<float>();
    // First row: [1, 2, 3]
    EXPECT_FLOAT_EQ(out_data[0], 1.0f);
    EXPECT_FLOAT_EQ(out_data[1], 2.0f);
    EXPECT_FLOAT_EQ(out_data[2], 3.0f);
    // Second row: [1, 2, 3]
    EXPECT_FLOAT_EQ(out_data[3], 1.0f);
    EXPECT_FLOAT_EQ(out_data[4], 2.0f);
    EXPECT_FLOAT_EQ(out_data[5], 3.0f);
}

TEST_P(RepeatTileTest, RepeatPartialDimensions) {
    // Test repeat with fewer repeats than dimensions
    // Input shape: (2, 3), repeats: {2} should apply only to last dimension
    std::vector<float> data_vec = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto input = from_data<float>(data_vec.data(), {2, 3}, device);
    auto output = repeat(input, {2});

    auto input_cpu = input.cpu();
    auto output_cpu = output.cpu();
    print_tensor<float>(input_cpu, "Input");
    print_tensor<float>(output_cpu, "Output (repeat partial)");

    ASSERT_EQ(output.shape()[0], 2);
    ASSERT_EQ(output.shape()[1], 6);  // 3 * 2
}

// =========================================================================
// repeat_interleave tests
// =========================================================================

TEST_P(RepeatInterleaveTest, Scalar1D) {
    // [1, 2, 3] with repeats=2 -> [1, 1, 2, 2, 3, 3]
    auto input = from_data<float>(std::vector<float>{1.0f, 2.0f, 3.0f}.data(), {3}, device);
    auto output = repeat_interleave(input, 2);

    auto output_cpu = output.cpu();
    print_tensor<float>(output_cpu, "repeat_interleave scalar 1D");

    ASSERT_EQ(output.ndim(), 1);
    ASSERT_EQ(output.shape()[0], 6);

    const float* data = output_cpu.data<float>();
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 1.0f);
    EXPECT_FLOAT_EQ(data[2], 2.0f);
    EXPECT_FLOAT_EQ(data[3], 2.0f);
    EXPECT_FLOAT_EQ(data[4], 3.0f);
    EXPECT_FLOAT_EQ(data[5], 3.0f);
}

TEST_P(RepeatInterleaveTest, Scalar1DWithDim) {
    // Same as above but with explicit dim=0
    auto input = from_data<float>(std::vector<float>{1.0f, 2.0f, 3.0f}.data(), {3}, device);
    auto output = repeat_interleave(input, 3, 0);

    auto output_cpu = output.cpu();
    print_tensor<float>(output_cpu, "repeat_interleave scalar 1D dim=0");

    ASSERT_EQ(output.ndim(), 1);
    ASSERT_EQ(output.shape()[0], 9);

    const float* data = output_cpu.data<float>();
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 1.0f);
    EXPECT_FLOAT_EQ(data[2], 1.0f);
    EXPECT_FLOAT_EQ(data[3], 2.0f);
    EXPECT_FLOAT_EQ(data[4], 2.0f);
    EXPECT_FLOAT_EQ(data[5], 2.0f);
    EXPECT_FLOAT_EQ(data[6], 3.0f);
    EXPECT_FLOAT_EQ(data[7], 3.0f);
    EXPECT_FLOAT_EQ(data[8], 3.0f);
}

TEST_P(RepeatInterleaveTest, Scalar2D_Dim0) {
    // [[1, 2], [3, 4]] with repeats=2, dim=0 -> [[1,2],[1,2],[3,4],[3,4]]
    std::vector<float> data_vec = {1.0f, 2.0f, 3.0f, 4.0f};
    auto input = from_data<float>(data_vec.data(), {2, 2}, device);
    auto output = repeat_interleave(input, 2, 0);

    auto output_cpu = output.cpu();
    print_tensor<float>(output_cpu, "repeat_interleave scalar 2D dim=0");

    ASSERT_EQ(output.shape()[0], 4);
    ASSERT_EQ(output.shape()[1], 2);

    const float* data = output_cpu.data<float>();
    // Row 0: [1, 2] (original row 0)
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 2.0f);
    // Row 1: [1, 2] (repeated row 0)
    EXPECT_FLOAT_EQ(data[2], 1.0f);
    EXPECT_FLOAT_EQ(data[3], 2.0f);
    // Row 2: [3, 4] (original row 1)
    EXPECT_FLOAT_EQ(data[4], 3.0f);
    EXPECT_FLOAT_EQ(data[5], 4.0f);
    // Row 3: [3, 4] (repeated row 1)
    EXPECT_FLOAT_EQ(data[6], 3.0f);
    EXPECT_FLOAT_EQ(data[7], 4.0f);
}

TEST_P(RepeatInterleaveTest, Scalar2D_Dim1) {
    // [[1, 2], [3, 4]] with repeats=2, dim=1 -> [[1,1,2,2],[3,3,4,4]]
    std::vector<float> data_vec = {1.0f, 2.0f, 3.0f, 4.0f};
    auto input = from_data<float>(data_vec.data(), {2, 2}, device);
    auto output = repeat_interleave(input, 2, 1);

    auto output_cpu = output.cpu();
    print_tensor<float>(output_cpu, "repeat_interleave scalar 2D dim=1");

    ASSERT_EQ(output.shape()[0], 2);
    ASSERT_EQ(output.shape()[1], 4);

    const float* data = output_cpu.data<float>();
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 1.0f);
    EXPECT_FLOAT_EQ(data[2], 2.0f);
    EXPECT_FLOAT_EQ(data[3], 2.0f);
    EXPECT_FLOAT_EQ(data[4], 3.0f);
    EXPECT_FLOAT_EQ(data[5], 3.0f);
    EXPECT_FLOAT_EQ(data[6], 4.0f);
    EXPECT_FLOAT_EQ(data[7], 4.0f);
}

// Regression: scalar repeat_interleave with a negative dim must normalize the
// dim before indexing the shape. The oneAPI scalar kernel read shape[dim] with
// a raw negative dim (OOB); dim=-1 must behave like dim=1 for a 2D input.
TEST_P(RepeatInterleaveTest, Scalar2D_DimNegative) {
    std::vector<float> data_vec = {1.0f, 2.0f, 3.0f, 4.0f};
    auto input = from_data<float>(data_vec.data(), {2, 2}, device);
    auto out_neg = repeat_interleave(input, 2, -1).cpu();
    auto out_pos = repeat_interleave(input, 2, 1).cpu();
    ASSERT_EQ(out_neg.ndim(), 2);
    ASSERT_EQ(out_neg.shape()[0], 2);
    ASSERT_EQ(out_neg.shape()[1], 4);
    ASSERT_EQ(out_neg.numel(), out_pos.numel());
    const float* n = out_neg.data<float>();
    const float* p = out_pos.data<float>();
    for (int64_t i = 0; i < out_neg.numel(); ++i)
        EXPECT_FLOAT_EQ(n[i], p[i]) << "dim=-1 != dim=1 at " << i << " on " << device.to_string();
}

TEST_P(RepeatInterleaveTest, Tensor1D) {
    // [1, 2, 3] with repeats=[1, 2, 3] -> [1, 2, 2, 3, 3, 3]
    auto input = from_data<float>(std::vector<float>{1.0f, 2.0f, 3.0f}.data(), {3}, device);
    auto repeats = from_data<int64_t>(std::vector<int64_t>{1, 2, 3}.data(), {3}, device);
    auto output = repeat_interleave(input, repeats);

    auto output_cpu = output.cpu();
    print_tensor<float>(output_cpu, "repeat_interleave tensor 1D");

    ASSERT_EQ(output.ndim(), 1);
    ASSERT_EQ(output.shape()[0], 6);

    const float* data = output_cpu.data<float>();
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 2.0f);
    EXPECT_FLOAT_EQ(data[2], 2.0f);
    EXPECT_FLOAT_EQ(data[3], 3.0f);
    EXPECT_FLOAT_EQ(data[4], 3.0f);
    EXPECT_FLOAT_EQ(data[5], 3.0f);
}

TEST_P(RepeatInterleaveTest, Tensor2D_Dim0) {
    // [[1, 2], [3, 4]] with repeats=[1, 3], dim=0
    // -> [[1,2], [3,4], [3,4], [3,4]]
    std::vector<float> data_vec = {1.0f, 2.0f, 3.0f, 4.0f};
    auto input = from_data<float>(data_vec.data(), {2, 2}, device);
    auto repeats = from_data<int64_t>(std::vector<int64_t>{1, 3}.data(), {2}, device);
    auto output = repeat_interleave(input, repeats, 0);

    auto output_cpu = output.cpu();
    print_tensor<float>(output_cpu, "repeat_interleave tensor 2D dim=0");

    ASSERT_EQ(output.shape()[0], 4);
    ASSERT_EQ(output.shape()[1], 2);

    const float* data = output_cpu.data<float>();
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 2.0f);
    EXPECT_FLOAT_EQ(data[2], 3.0f);
    EXPECT_FLOAT_EQ(data[3], 4.0f);
    EXPECT_FLOAT_EQ(data[4], 3.0f);
    EXPECT_FLOAT_EQ(data[5], 4.0f);
    EXPECT_FLOAT_EQ(data[6], 3.0f);
    EXPECT_FLOAT_EQ(data[7], 4.0f);
}

TEST_P(RepeatInterleaveTest, FlattenNoDim) {
    // [[1, 2], [3, 4]] with repeats=2, no dim -> flatten first then repeat
    // flatten: [1, 2, 3, 4] -> [1, 1, 2, 2, 3, 3, 4, 4]
    std::vector<float> data_vec = {1.0f, 2.0f, 3.0f, 4.0f};
    auto input = from_data<float>(data_vec.data(), {2, 2}, device);
    auto output = repeat_interleave(input, 2);

    auto output_cpu = output.cpu();
    print_tensor<float>(output_cpu, "repeat_interleave flatten (no dim)");

    ASSERT_EQ(output.ndim(), 1);
    ASSERT_EQ(output.shape()[0], 8);

    const float* data = output_cpu.data<float>();
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 1.0f);
    EXPECT_FLOAT_EQ(data[2], 2.0f);
    EXPECT_FLOAT_EQ(data[3], 2.0f);
    EXPECT_FLOAT_EQ(data[4], 3.0f);
    EXPECT_FLOAT_EQ(data[5], 3.0f);
    EXPECT_FLOAT_EQ(data[6], 4.0f);
    EXPECT_FLOAT_EQ(data[7], 4.0f);
}

TEST_P(RepeatInterleaveTest, ZeroRepeats) {
    // [1, 2, 3] with repeats=0 -> empty tensor
    auto input = from_data<float>(std::vector<float>{1.0f, 2.0f, 3.0f}.data(), {3}, device);
    auto output = repeat_interleave(input, 0);

    ASSERT_EQ(output.ndim(), 1);
    ASSERT_EQ(output.shape()[0], 0);
    ASSERT_EQ(output.numel(), 0);
}

INSTANTIATE_BACKEND_TESTS(RepeatTileTest);
INSTANTIATE_BACKEND_TESTS(RepeatInterleaveTest);
