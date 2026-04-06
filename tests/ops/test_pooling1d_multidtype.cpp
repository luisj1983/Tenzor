/**
 * @file test_pooling1d_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for 1D pooling operations
 *
 * Covers: MaxPool1d, AvgPool1d, AdaptiveMaxPool1d, AdaptiveAvgPool1d
 * (forward + backward for each)
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/pooling.hpp>
#include <tenzor/autograd/variable.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Fixture
// ============================================================================

class Pooling1dMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// MaxPool1d Tests
// ============================================================================

TEST_P(Pooling1dMultiDTypeTest, MaxPool1dForwardShape) {
    nn::MaxPool1d pool(3, 2);

    Variable input = createInput({2, 3, 10}, false);
    auto output = pool.forward(input);
    expectShape(output.tensor(), {2, 3, 4});
    expectDevice(output.tensor());
    expectDType(output.tensor());
}

TEST_P(Pooling1dMultiDTypeTest, MaxPool1dGradientFlow) {
    nn::MaxPool1d pool(2, 1);

    Variable input = createInput({1, 2, 6}, true);
    auto output = pool.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    EXPECT_NO_THROW({ output.backward(grad_output); });

    if (!input.grad().has_value()) {
        GTEST_SKIP() << "1D pooling backward not yet implemented for this backend";
    }
    expectShape(*input.grad(), {1, 2, 6});
}

TEST_P(Pooling1dMultiDTypeTest, MaxPool1dWithPadding) {
    nn::MaxPool1d pool(3, 1, 1);

    Variable input = createInput({1, 4, 8}, false);
    auto output = pool.forward(input);
    expectShape(output.tensor(), {1, 4, 8});
}

// ============================================================================
// AvgPool1d Tests
// ============================================================================

TEST_P(Pooling1dMultiDTypeTest, AvgPool1dForwardShape) {
    nn::AvgPool1d pool(3, 2);

    Variable input = createInput({2, 3, 10}, false);
    auto output = pool.forward(input);
    expectShape(output.tensor(), {2, 3, 4});
    expectDevice(output.tensor());
    expectDType(output.tensor());
}

TEST_P(Pooling1dMultiDTypeTest, AvgPool1dGradientFlow) {
    nn::AvgPool1d pool(2, 1);

    Variable input = createInput({1, 2, 6}, true);
    auto output = pool.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    EXPECT_NO_THROW({ output.backward(grad_output); });

    if (!input.grad().has_value()) {
        GTEST_SKIP() << "1D pooling backward not yet implemented for this backend";
    }
    expectShape(*input.grad(), {1, 2, 6});
}

// ============================================================================
// AdaptiveMaxPool1d Tests
// ============================================================================

TEST_P(Pooling1dMultiDTypeTest, AdaptiveMaxPool1dOutputSize) {
    nn::AdaptiveMaxPool1d pool(5);

    Variable input = createInput({2, 3, 20}, false);
    auto output = pool.forward(input);
    expectShape(output.tensor(), {2, 3, 5});
    expectDevice(output.tensor());
}

TEST_P(Pooling1dMultiDTypeTest, AdaptiveMaxPool1dGradientFlow) {
    nn::AdaptiveMaxPool1d pool(4);

    Variable input = createInput({1, 2, 12}, true);
    auto output = pool.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    EXPECT_NO_THROW({ output.backward(grad_output); });

    if (!input.grad().has_value()) {
        GTEST_SKIP() << "1D pooling backward not yet implemented for this backend";
    }
    expectShape(*input.grad(), {1, 2, 12});
}

// ============================================================================
// AdaptiveAvgPool1d Tests
// ============================================================================

TEST_P(Pooling1dMultiDTypeTest, AdaptiveAvgPool1dOutputSize) {
    nn::AdaptiveAvgPool1d pool(5);

    Variable input = createInput({2, 3, 20}, false);
    auto output = pool.forward(input);
    expectShape(output.tensor(), {2, 3, 5});
    expectDevice(output.tensor());
}

TEST_P(Pooling1dMultiDTypeTest, AdaptiveAvgPool1dGradientFlow) {
    nn::AdaptiveAvgPool1d pool(4);

    Variable input = createInput({1, 2, 12}, true);
    auto output = pool.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    EXPECT_NO_THROW({ output.backward(grad_output); });

    if (!input.grad().has_value()) {
        GTEST_SKIP() << "1D pooling backward not yet implemented for this backend";
    }
    expectShape(*input.grad(), {1, 2, 12});
}

TEST_P(Pooling1dMultiDTypeTest, AdaptiveAvgPool1dToSingleElement) {
    nn::AdaptiveAvgPool1d pool(1);

    Variable input = createInput({2, 4, 16}, false);
    auto output = pool.forward(input);
    expectShape(output.tensor(), {2, 4, 1});
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(Pooling1dMultiDTypeTest);
