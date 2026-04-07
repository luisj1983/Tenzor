/**
 * @file test_pooling3d_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for 3D pooling operations
 *
 * Covers: MaxPool3d, AvgPool3d, AdaptiveMaxPool3d, AdaptiveAvgPool3d
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

class Pooling3dMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// MaxPool3d Tests
// ============================================================================

TEST_P(Pooling3dMultiDTypeTest, MaxPool3dForwardShape) {
    nn::MaxPool3d pool(2, 2);

    Variable input = createInput({1, 2, 8, 8, 8}, false);
    auto output = pool.forward(input);
    expectShape(output.tensor(), {1, 2, 4, 4, 4});
    expectDevice(output.tensor());
    expectDType(output.tensor());
}

TEST_P(Pooling3dMultiDTypeTest, MaxPool3dGradientFlow) {
    nn::MaxPool3d pool(2, 2);

    Variable input = createInput({1, 1, 4, 4, 4}, true);
    auto output = pool.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    EXPECT_NO_THROW({ output.backward(grad_output); });

    ASSERT_TRUE(input.grad().has_value())
        << "MaxPool3d backward did not produce gradient on " << device().to_string();
    expectShape(*input.grad(), {1, 1, 4, 4, 4});
}

// ============================================================================
// AvgPool3d Tests
// ============================================================================

TEST_P(Pooling3dMultiDTypeTest, AvgPool3dForwardShape) {
    nn::AvgPool3d pool(2, 2);

    Variable input = createInput({1, 2, 8, 8, 8}, false);
    auto output = pool.forward(input);
    expectShape(output.tensor(), {1, 2, 4, 4, 4});
    expectDevice(output.tensor());
    expectDType(output.tensor());
}

TEST_P(Pooling3dMultiDTypeTest, AvgPool3dGradientFlow) {
    nn::AvgPool3d pool(2, 2);

    Variable input = createInput({1, 1, 4, 4, 4}, true);
    auto output = pool.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    EXPECT_NO_THROW({ output.backward(grad_output); });

    ASSERT_TRUE(input.grad().has_value())
        << "AvgPool3d backward did not produce gradient on " << device().to_string();
    expectShape(*input.grad(), {1, 1, 4, 4, 4});
}

// ============================================================================
// AdaptiveMaxPool3d Tests
// ============================================================================

TEST_P(Pooling3dMultiDTypeTest, AdaptiveMaxPool3dOutputSize) {
    nn::AdaptiveMaxPool3d pool({2, 2, 2});

    Variable input = createInput({1, 3, 8, 8, 8}, false);
    auto output = pool.forward(input);
    expectShape(output.tensor(), {1, 3, 2, 2, 2});
    expectDevice(output.tensor());
}

TEST_P(Pooling3dMultiDTypeTest, AdaptiveMaxPool3dGradientFlow) {
    nn::AdaptiveMaxPool3d pool({2, 2, 2});

    Variable input = createInput({1, 1, 6, 6, 6}, true);
    auto output = pool.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    EXPECT_NO_THROW({ output.backward(grad_output); });

    ASSERT_TRUE(input.grad().has_value())
        << "AdaptiveMaxPool3d backward did not produce gradient on " << device().to_string();
    expectShape(*input.grad(), {1, 1, 6, 6, 6});
}

// ============================================================================
// AdaptiveAvgPool3d Tests
// ============================================================================

TEST_P(Pooling3dMultiDTypeTest, AdaptiveAvgPool3dOutputSize) {
    nn::AdaptiveAvgPool3d pool({2, 2, 2});

    Variable input = createInput({1, 3, 8, 8, 8}, false);
    auto output = pool.forward(input);
    expectShape(output.tensor(), {1, 3, 2, 2, 2});
    expectDevice(output.tensor());
}

TEST_P(Pooling3dMultiDTypeTest, AdaptiveAvgPool3dGradientFlow) {
    nn::AdaptiveAvgPool3d pool({2, 2, 2});

    Variable input = createInput({1, 1, 6, 6, 6}, true);
    auto output = pool.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    EXPECT_NO_THROW({ output.backward(grad_output); });

    ASSERT_TRUE(input.grad().has_value())
        << "AdaptiveAvgPool3d backward did not produce gradient on " << device().to_string();
    expectShape(*input.grad(), {1, 1, 6, 6, 6});
}

TEST_P(Pooling3dMultiDTypeTest, AdaptiveAvgPool3dGlobalPooling) {
    nn::AdaptiveAvgPool3d pool({1, 1, 1});

    Variable input = createInput({2, 4, 6, 6, 6}, false);
    auto output = pool.forward(input);
    expectShape(output.tensor(), {2, 4, 1, 1, 1});
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(Pooling3dMultiDTypeTest);
