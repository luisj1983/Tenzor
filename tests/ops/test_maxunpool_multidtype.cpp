/**
 * @file test_maxunpool_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for MaxUnpool2d / MaxUnpool3d
 *
 * Covers forward correctness of max_unpool2d / max_unpool3d across all
 * registered backends × {Float32, Float64, Float16}. Uses synthetic indices
 * (as in tests/backend_parity/test_nn_pooling_parity.cpp) since the library
 * does not expose a public max_pool2d_with_indices.
 *
 * Backward path: src/nn/functional.cpp wires an IndexedPoolBackward grad_fn
 * that dispatches to OpId::MaxUnpool{2,3}dBackward on the appropriate
 * backend. Tests below include GradientFlow assertions.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/functional.hpp>
#include <tenzor/autograd/variable.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class MaxUnpoolMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// MaxUnpool2d
// ============================================================================

TEST_P(MaxUnpoolMultiDTypeTest, MaxUnpool2dForwardShape) {
    auto pooled_t = tenzor::randn({1, 2, 2, 2}, DType::Float32, device_);
    if (dtype_ != DType::Float32) pooled_t = pooled_t.to(dtype_);
    Variable pooled(pooled_t, false);

    auto indices_cpu = zeros({1, 2, 2, 2}, DType::Int64, Device::cpu());
    auto* idx = indices_cpu.data<int64_t>();
    int64_t pattern[] = {0, 2, 8, 10};
    for (int64_t c = 0; c < 2; ++c) {
        for (int64_t k = 0; k < 4; ++k) {
            idx[c * 4 + k] = pattern[k];
        }
    }
    auto indices = indices_cpu.to(device_);

    auto out = nn::functional::max_unpool2d(
        pooled, indices, std::make_pair<int64_t, int64_t>(2, 2));

    expectShape(out.tensor(), {1, 2, 4, 4});
    expectDevice(out.tensor());
    expectDType(out.tensor());
}

TEST_P(MaxUnpoolMultiDTypeTest, MaxUnpool2dGradientFlow) {
    auto pooled_t = tenzor::randn({1, 2, 2, 2}, DType::Float32, device_);
    if (dtype_ != DType::Float32) pooled_t = pooled_t.to(dtype_);
    Variable pooled(pooled_t, true);

    auto indices_cpu = zeros({1, 2, 2, 2}, DType::Int64, Device::cpu());
    auto* idx = indices_cpu.data<int64_t>();
    int64_t pattern[] = {0, 2, 8, 10};
    for (int64_t c = 0; c < 2; ++c) {
        for (int64_t k = 0; k < 4; ++k) idx[c * 4 + k] = pattern[k];
    }
    auto indices = indices_cpu.to(device_);

    auto out = nn::functional::max_unpool2d(
        pooled, indices, std::make_pair<int64_t, int64_t>(2, 2));
    auto grad = tenzor::ones({1, 2, 4, 4}, dtype_, device_);
    out.backward(grad);

    ASSERT_TRUE(pooled.grad().has_value())
        << "MaxUnpool2d backward did not populate pooled.grad on "
        << device().to_string();
    expectShape(*pooled.grad(), {1, 2, 2, 2});
}

TEST_P(MaxUnpoolMultiDTypeTest, MaxUnpool2dPlacesValuesAtIndices) {
    // Known pooled values placed at known flat indices → verify the output
    // actually carries those values at those positions and zeros elsewhere.
    auto pooled_cpu = tenzor::ones({1, 1, 2, 2}, DType::Float32, Device::cpu());
    auto* pd = pooled_cpu.data<float>();
    pd[0] = 1.0f; pd[1] = 2.0f; pd[2] = 3.0f; pd[3] = 4.0f;
    if (dtype_ != DType::Float32) pooled_cpu = pooled_cpu.to(dtype_);
    Variable pooled(pooled_cpu.to(device_), false);

    auto indices_cpu = zeros({1, 1, 2, 2}, DType::Int64, Device::cpu());
    auto* idx = indices_cpu.data<int64_t>();
    idx[0] = 0; idx[1] = 2; idx[2] = 8; idx[3] = 10;
    auto indices = indices_cpu.to(device_);

    auto out = nn::functional::max_unpool2d(
        pooled, indices, std::make_pair<int64_t, int64_t>(2, 2));

    auto out_cpu = out.tensor().to(Device::cpu());
    if (out_cpu.dtype() != DType::Float32) out_cpu = out_cpu.to(DType::Float32);
    const auto* op = out_cpu.data<float>();
    EXPECT_NEAR(op[0],  1.0f, atol_);
    EXPECT_NEAR(op[2],  2.0f, atol_);
    EXPECT_NEAR(op[8],  3.0f, atol_);
    EXPECT_NEAR(op[10], 4.0f, atol_);
    // A few off-pattern positions must be zero.
    EXPECT_NEAR(op[1], 0.0f, atol_);
    EXPECT_NEAR(op[5], 0.0f, atol_);
    EXPECT_NEAR(op[15], 0.0f, atol_);
}

// ============================================================================
// MaxUnpool3d
// ============================================================================

TEST_P(MaxUnpoolMultiDTypeTest, MaxUnpool3dForwardShape) {
    auto pooled_t = tenzor::randn({1, 1, 2, 2, 2}, DType::Float32, device_);
    if (dtype_ != DType::Float32) pooled_t = pooled_t.to(dtype_);
    Variable pooled(pooled_t, false);

    auto indices_cpu = zeros({1, 1, 2, 2, 2}, DType::Int64, Device::cpu());
    auto* idx = indices_cpu.data<int64_t>();
    int64_t pattern[] = {0, 2, 8, 10, 32, 34, 40, 42};
    for (int64_t k = 0; k < 8; ++k) idx[k] = pattern[k];
    auto indices = indices_cpu.to(device_);

    auto out = nn::functional::max_unpool3d(
        pooled, indices,
        std::make_tuple<int64_t, int64_t, int64_t>(2, 2, 2));

    expectShape(out.tensor(), {1, 1, 4, 4, 4});
    expectDevice(out.tensor());
    expectDType(out.tensor());
}

TEST_P(MaxUnpoolMultiDTypeTest, MaxUnpool3dGradientFlow) {
    auto pooled_t = tenzor::randn({1, 1, 2, 2, 2}, DType::Float32, device_);
    if (dtype_ != DType::Float32) pooled_t = pooled_t.to(dtype_);
    Variable pooled(pooled_t, true);

    auto indices_cpu = zeros({1, 1, 2, 2, 2}, DType::Int64, Device::cpu());
    auto* idx = indices_cpu.data<int64_t>();
    int64_t pattern[] = {0, 2, 8, 10, 32, 34, 40, 42};
    for (int64_t k = 0; k < 8; ++k) idx[k] = pattern[k];
    auto indices = indices_cpu.to(device_);

    auto out = nn::functional::max_unpool3d(
        pooled, indices,
        std::make_tuple<int64_t, int64_t, int64_t>(2, 2, 2));
    auto grad = tenzor::ones({1, 1, 4, 4, 4}, dtype_, device_);
    out.backward(grad);

    ASSERT_TRUE(pooled.grad().has_value())
        << "MaxUnpool3d backward did not populate pooled.grad on "
        << device().to_string();
    expectShape(*pooled.grad(), {1, 1, 2, 2, 2});
}

TEST_P(MaxUnpoolMultiDTypeTest, MaxUnpool3dPlacesValuesAtIndices) {
    auto pooled_cpu = tenzor::ones({1, 1, 2, 2, 2}, DType::Float32, Device::cpu());
    auto* pd = pooled_cpu.data<float>();
    for (int i = 0; i < 8; ++i) pd[i] = static_cast<float>(i + 1);
    if (dtype_ != DType::Float32) pooled_cpu = pooled_cpu.to(dtype_);
    Variable pooled(pooled_cpu.to(device_), false);

    auto indices_cpu = zeros({1, 1, 2, 2, 2}, DType::Int64, Device::cpu());
    auto* idx = indices_cpu.data<int64_t>();
    int64_t pattern[] = {0, 2, 8, 10, 32, 34, 40, 42};
    for (int64_t k = 0; k < 8; ++k) idx[k] = pattern[k];
    auto indices = indices_cpu.to(device_);

    auto out = nn::functional::max_unpool3d(
        pooled, indices,
        std::make_tuple<int64_t, int64_t, int64_t>(2, 2, 2));

    auto out_cpu = out.tensor().to(Device::cpu());
    if (out_cpu.dtype() != DType::Float32) out_cpu = out_cpu.to(DType::Float32);
    const auto* op = out_cpu.data<float>();
    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(op[pattern[i]], static_cast<float>(i + 1), atol_);
    }
    // Some off-pattern positions must be zero.
    EXPECT_NEAR(op[1], 0.0f, atol_);
    EXPECT_NEAR(op[16], 0.0f, atol_);
    EXPECT_NEAR(op[63], 0.0f, atol_);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(MaxUnpoolMultiDTypeTest);
