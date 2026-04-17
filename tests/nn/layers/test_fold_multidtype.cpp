/**
 * @file test_fold_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for Fold (col2im) layer
 */

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/vision.hpp>

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::nn;

class FoldMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(FoldMultiDTypeTest, ForwardShape) {
    Fold fold({4, 4}, 2, 1, 0, 1);
    convert_model(fold);
    auto input = createInput({1, 4, 9}, false);
    auto output = fold.forward(input);
    auto shape = output.tensor().shape();
    ASSERT_EQ(shape[0], 1);
    ASSERT_EQ(shape[2], 4);
    ASSERT_EQ(shape[3], 4);
}

TEST_P(FoldMultiDTypeTest, FoldUnfoldInverse) {
    Unfold unfold(3, 1, 0, 1);
    Fold fold({6, 6}, 3, 1, 0, 1);
    convert_model(unfold);
    convert_model(fold);

    auto input = createInput({1, 1, 6, 6}, false);
    auto unfolded = unfold.forward(input);
    auto refolded = fold.forward(unfolded);
    ASSERT_EQ(refolded.tensor().shape()[2], 6);
    ASSERT_EQ(refolded.tensor().shape()[3], 6);
}

TEST_P(FoldMultiDTypeTest, Backward) {
    Fold fold({4, 4}, 2, 1, 0, 1);
    convert_model(fold);
    auto input = createInput({1, 4, 9}, true);
    auto output = fold.forward(input);
    auto loss = sum(output);
    loss.backward();
    ASSERT_TRUE(input.grad().has_value());
}

TEST_P(FoldMultiDTypeTest, DTypePreserved) {
    Fold fold({4, 4}, 2, 1, 0, 1);
    convert_model(fold);
    auto input = createInput({1, 4, 9}, false);
    auto output = fold.forward(input);
    expectDType(output.tensor());
}

TEST_P(FoldMultiDTypeTest, DevicePreserved) {
    Fold fold({4, 4}, 2, 1, 0, 1);
    convert_model(fold);
    auto input = createInput({1, 4, 9}, false);
    auto output = fold.forward(input);
    expectDevice(output.tensor());
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(FoldMultiDTypeTest);
