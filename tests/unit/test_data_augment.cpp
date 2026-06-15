// Regression tests for data-augmentation transforms.
//
// These augmentations (Cutout, RandomErasing, CutMix) must NOT mutate the
// caller's input tensors in place: assigning `Tensor out = input;` shares
// storage, so an in-place write corrupts the caller's data. The fix clones the
// input; these tests guard against a regression by asserting the inputs are
// byte-for-byte unchanged after the transform runs.

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "tenzor/tenzor.hpp"
#include "tenzor/data/transforms.hpp"

#include <vector>

namespace {

using tenzor::Tensor;

// Fixture that brings the runtime up exactly once (CPU backend registration).
class DataAugmentInPlace : public ::testing::Test {
protected:
    void SetUp() override { tenzor::testing::EnsureInitialized(); }
};

std::vector<float> read_floats(const Tensor& t) {
    const float* p = static_cast<const float*>(t.data_ptr());
    return std::vector<float>(p, p + t.numel());
}

Tensor make_filled(const std::vector<int64_t>& shape, float value) {
    Tensor t(shape, tenzor::DType::Float32, tenzor::Device::cpu());
    float* p = static_cast<float*>(t.data_ptr());
    for (int64_t i = 0; i < t.numel(); ++i) {
        p[i] = value;
    }
    return t;
}

} // namespace

TEST_F(DataAugmentInPlace, CutoutDoesNotMutateInput) {
    tenzor::manual_seed(123);
    Tensor input = make_filled({1, 8, 8}, 1.0f);
    Tensor target = make_filled({1}, 0.0f);
    auto before = read_floats(input);

    tenzor::data::transforms::Cutout cutout(/*num_holes=*/2, /*hole_size=*/4);
    auto [output, out_target] = cutout(input, target);

    auto after = read_floats(input);
    EXPECT_EQ(before, after) << "Cutout mutated the caller's input tensor";
    // And the output must actually have been modified (holes zeroed).
    auto out = read_floats(output);
    bool any_zero = false;
    for (float v : out) any_zero = any_zero || (v == 0.0f);
    EXPECT_TRUE(any_zero) << "Cutout produced no holes; test is not exercising the write path";
}

TEST_F(DataAugmentInPlace, RandomErasingDoesNotMutateInput) {
    tenzor::manual_seed(123);
    Tensor input = make_filled({3, 8, 8}, 1.0f);
    Tensor target = make_filled({1}, 0.0f);
    auto before = read_floats(input);

    // p = 1.0 forces the erase path to run.
    tenzor::data::transforms::RandomErasing erasing(/*p=*/1.0f);
    auto [output, out_target] = erasing(input, target);

    auto after = read_floats(input);
    EXPECT_EQ(before, after) << "RandomErasing mutated the caller's input tensor";
}

TEST_F(DataAugmentInPlace, CutMixDoesNotMutateInputs) {
    tenzor::manual_seed(123);
    Tensor input1 = make_filled({3, 8, 8}, 1.0f);
    Tensor input2 = make_filled({3, 8, 8}, 2.0f);
    Tensor target1 = make_filled({2}, 0.0f);
    Tensor target2 = make_filled({2}, 1.0f);
    auto before1 = read_floats(input1);
    auto before2 = read_floats(input2);

    tenzor::data::transforms::CutMix cutmix(/*alpha=*/1.0f);
    auto [mixed_input, mixed_target] = cutmix(input1, target1, input2, target2);

    EXPECT_EQ(before1, read_floats(input1)) << "CutMix mutated input1 in place";
    EXPECT_EQ(before2, read_floats(input2)) << "CutMix mutated input2 in place";
}
