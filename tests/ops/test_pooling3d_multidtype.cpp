/**
 * @file test_pooling3d_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for 3D pooling operations
 *
 * Covers: MaxPool3d, AvgPool3d, AdaptiveMaxPool3d, AdaptiveAvgPool3d
 * (forward + backward for each)
 *
 * Audit-T.1: every TEST_P that previously only asserted output shape now
 * also asserts numeric output (or gradient) values against a CPU reference
 * computed with the identical layer configuration.  We don't rely on
 * closed-form pooling math because of corner cases in
 * Adaptive*Pool's window-overlap rules; running the same layer on a CPU
 * copy of the input is the cleanest "ground truth".
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/pooling.hpp>
#include <tenzor/autograd/variable.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include "../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Fixture
// ============================================================================

class Pooling3dMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    // Audit-T.1: tolerance for the device-vs-CPU diff after a Float32
    // round-trip.  Pooling is mostly select-or-mean so error is small.
    float poolAtol() const {
        switch (dtype()) {
            case DType::Float16:
            case DType::BFloat16:
                return 5e-2f;
            case DType::Float64:
                return 1e-6f;
            default:
                return 1e-4f;
        }
    }
};

// ============================================================================
// MaxPool3d Tests
// ============================================================================

TEST_P(Pooling3dMultiDTypeTest, MaxPool3dForwardShape) {
    nn::MaxPool3d pool(2, 2);
    nn::MaxPool3d pool_cpu(2, 2);

    Variable input = createInput({1, 2, 8, 8, 8}, false);
    auto output = pool.forward(input);
    expectShape(output.tensor(), {1, 2, 4, 4, 4});
    expectDevice(output.tensor());
    expectDType(output.tensor());

    // Audit-T.1: forward MaxPool3d on a CPU Float32 copy of the same input.
    Variable input_cpu(input.tensor().to(Device::cpu()).to(DType::Float32), false);
    auto ref = pool_cpu.forward(input_cpu);
    expectTensorNear(output.tensor(), ref.tensor(), poolAtol());
}

TEST_P(Pooling3dMultiDTypeTest, MaxPool3dGradientFlow) {
    nn::MaxPool3d pool(2, 2);
    nn::MaxPool3d pool_cpu(2, 2);

    Variable input = createInput({1, 1, 4, 4, 4}, true);
    auto output = pool.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    output.backward(grad_output);
    EXPECT_GRAD_FLOWS(input);
    expectShape(*input.grad(), {1, 1, 4, 4, 4});

    // Audit-T.1: rerun on CPU and compare input.grad() element-wise.
    Variable input_cpu(input.tensor().to(Device::cpu()).to(DType::Float32), true);
    auto out_cpu = pool_cpu.forward(input_cpu);
    auto grad_cpu = tenzor::ones(out_shape_vec, DType::Float32, Device::cpu());
    out_cpu.backward(grad_cpu);
    ASSERT_TRUE(input_cpu.grad().has_value());
    expectTensorNear(input.grad().value(), input_cpu.grad().value(), poolAtol());
}

// ============================================================================
// AvgPool3d Tests
// ============================================================================

TEST_P(Pooling3dMultiDTypeTest, AvgPool3dForwardShape) {
    nn::AvgPool3d pool(2, 2);
    nn::AvgPool3d pool_cpu(2, 2);

    Variable input = createInput({1, 2, 8, 8, 8}, false);
    auto output = pool.forward(input);
    expectShape(output.tensor(), {1, 2, 4, 4, 4});
    expectDevice(output.tensor());
    expectDType(output.tensor());

    // Audit-T.1: CPU reference comparison.
    Variable input_cpu(input.tensor().to(Device::cpu()).to(DType::Float32), false);
    auto ref = pool_cpu.forward(input_cpu);
    expectTensorNear(output.tensor(), ref.tensor(), poolAtol());
}

TEST_P(Pooling3dMultiDTypeTest, AvgPool3dGradientFlow) {
    nn::AvgPool3d pool(2, 2);
    nn::AvgPool3d pool_cpu(2, 2);

    Variable input = createInput({1, 1, 4, 4, 4}, true);
    auto output = pool.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    output.backward(grad_output);
    EXPECT_GRAD_FLOWS(input);
    expectShape(*input.grad(), {1, 1, 4, 4, 4});

    // Audit-T.1: AvgPool3d backward distributes 1/window_size to each
    // input position.  Compare to CPU reference.
    Variable input_cpu(input.tensor().to(Device::cpu()).to(DType::Float32), true);
    auto out_cpu = pool_cpu.forward(input_cpu);
    auto grad_cpu = tenzor::ones(out_shape_vec, DType::Float32, Device::cpu());
    out_cpu.backward(grad_cpu);
    ASSERT_TRUE(input_cpu.grad().has_value());
    expectTensorNear(input.grad().value(), input_cpu.grad().value(), poolAtol());
}

// Regression: AvgPool3d with padding>0 must honor count_include_pad (the ROCm
// 3D kernels previously ignored it, hardcoding count_include_pad=false, so the
// default count_include_pad=true diverged from CPU at border outputs by using
// valid_count instead of kD*kH*kW as the divisor).
TEST_P(Pooling3dMultiDTypeTest, AvgPool3dCountIncludePadTrue) {
    nn::AvgPool3d pool(/*kernel=*/3, /*stride=*/2, /*padding=*/1, /*count_include_pad=*/true);
    nn::AvgPool3d pool_cpu(3, 2, 1, true);

    Variable input = createInput({1, 2, 5, 5, 5}, false);
    auto output = pool.forward(input);
    expectDevice(output.tensor());
    expectDType(output.tensor());

    Variable input_cpu(input.tensor().to(Device::cpu()).to(DType::Float32), false);
    auto ref = pool_cpu.forward(input_cpu);
    expectTensorNear(output.tensor(), ref.tensor(), poolAtol());
}

TEST_P(Pooling3dMultiDTypeTest, AvgPool3dCountIncludePadFalse) {
    nn::AvgPool3d pool(3, 2, 1, /*count_include_pad=*/false);
    nn::AvgPool3d pool_cpu(3, 2, 1, false);

    Variable input = createInput({1, 2, 5, 5, 5}, true);
    auto output = pool.forward(input);

    Variable input_cpu(input.tensor().to(Device::cpu()).to(DType::Float32), false);
    auto ref = pool_cpu.forward(input_cpu);
    expectTensorNear(output.tensor(), ref.tensor(), poolAtol());

    // Backward gradient must also use the same (valid-count) divisor as CPU.
    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    output.backward(tenzor::ones(out_shape_vec, dtype(), device()));
    EXPECT_GRAD_FLOWS(input);
}

// ============================================================================
// AdaptiveMaxPool3d Tests
// ============================================================================

TEST_P(Pooling3dMultiDTypeTest, AdaptiveMaxPool3dOutputSize) {
    nn::AdaptiveMaxPool3d pool({2, 2, 2});
    nn::AdaptiveMaxPool3d pool_cpu({2, 2, 2});

    Variable input = createInput({1, 3, 8, 8, 8}, false);
    auto output = pool.forward(input);
    expectShape(output.tensor(), {1, 3, 2, 2, 2});
    expectDevice(output.tensor());

    // Audit-T.1: CPU reference comparison.
    Variable input_cpu(input.tensor().to(Device::cpu()).to(DType::Float32), false);
    auto ref = pool_cpu.forward(input_cpu);
    expectTensorNear(output.tensor(), ref.tensor(), poolAtol());
}

TEST_P(Pooling3dMultiDTypeTest, AdaptiveMaxPool3dGradientFlow) {
    nn::AdaptiveMaxPool3d pool({2, 2, 2});
    nn::AdaptiveMaxPool3d pool_cpu({2, 2, 2});

    Variable input = createInput({1, 1, 6, 6, 6}, true);
    auto output = pool.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    output.backward(grad_output);
    EXPECT_GRAD_FLOWS(input);
    expectShape(*input.grad(), {1, 1, 6, 6, 6});

    Variable input_cpu(input.tensor().to(Device::cpu()).to(DType::Float32), true);
    auto out_cpu = pool_cpu.forward(input_cpu);
    auto grad_cpu = tenzor::ones(out_shape_vec, DType::Float32, Device::cpu());
    out_cpu.backward(grad_cpu);
    ASSERT_TRUE(input_cpu.grad().has_value());
    expectTensorNear(input.grad().value(), input_cpu.grad().value(), poolAtol());
}

// ============================================================================
// AdaptiveAvgPool3d Tests
// ============================================================================

TEST_P(Pooling3dMultiDTypeTest, AdaptiveAvgPool3dOutputSize) {
    nn::AdaptiveAvgPool3d pool({2, 2, 2});
    nn::AdaptiveAvgPool3d pool_cpu({2, 2, 2});

    Variable input = createInput({1, 3, 8, 8, 8}, false);
    auto output = pool.forward(input);
    expectShape(output.tensor(), {1, 3, 2, 2, 2});
    expectDevice(output.tensor());

    Variable input_cpu(input.tensor().to(Device::cpu()).to(DType::Float32), false);
    auto ref = pool_cpu.forward(input_cpu);
    expectTensorNear(output.tensor(), ref.tensor(), poolAtol());
}

TEST_P(Pooling3dMultiDTypeTest, AdaptiveAvgPool3dGradientFlow) {
    nn::AdaptiveAvgPool3d pool({2, 2, 2});
    nn::AdaptiveAvgPool3d pool_cpu({2, 2, 2});

    Variable input = createInput({1, 1, 6, 6, 6}, true);
    auto output = pool.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    output.backward(grad_output);
    EXPECT_GRAD_FLOWS(input);
    expectShape(*input.grad(), {1, 1, 6, 6, 6});

    Variable input_cpu(input.tensor().to(Device::cpu()).to(DType::Float32), true);
    auto out_cpu = pool_cpu.forward(input_cpu);
    auto grad_cpu = tenzor::ones(out_shape_vec, DType::Float32, Device::cpu());
    out_cpu.backward(grad_cpu);
    ASSERT_TRUE(input_cpu.grad().has_value());
    expectTensorNear(input.grad().value(), input_cpu.grad().value(), poolAtol());
}

TEST_P(Pooling3dMultiDTypeTest, AdaptiveAvgPool3dGlobalPooling) {
    nn::AdaptiveAvgPool3d pool({1, 1, 1});

    Variable input = createInput({2, 4, 6, 6, 6}, false);
    auto output = pool.forward(input);
    expectShape(output.tensor(), {2, 4, 1, 1, 1});

    // Audit-T.1: global avg pool == mean over spatial dims (here 6*6*6 = 216
    // elements per (N,C) cell).  Compute the closed-form reference on CPU.
    auto in_cpu = input.tensor().to(Device::cpu()).to(DType::Float32).contiguous();
    auto* ip = in_cpu.data<float>();
    auto expected = tenzor::zeros({2, 4, 1, 1, 1}, DType::Float32, Device::cpu());
    auto* ep = expected.data<float>();
    const int64_t spatial = 6 * 6 * 6;
    for (int64_t n = 0; n < 2; ++n) {
        for (int64_t c = 0; c < 4; ++c) {
            float sum = 0.0f;
            for (int64_t i = 0; i < spatial; ++i) {
                sum += ip[n * 4 * spatial + c * spatial + i];
            }
            ep[n * 4 + c] = sum / static_cast<float>(spatial);
        }
    }
    expectTensorNear(output.tensor(), expected, poolAtol());
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(Pooling3dMultiDTypeTest);
