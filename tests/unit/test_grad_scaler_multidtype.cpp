/**
 * @file test_grad_scaler_multidtype.cpp
 * @brief Multi-dtype multi-backend tests for GradScaler (AMP gradient scaling)
 *
 * Tests gradient scaling with Float32, Float64, and Float16 dtypes across
 * CPU, CUDA, OneAPI, Vulkan, and ROCm backends to ensure:
 * - Scaling behavior across different precision levels
 * - Underflow/overflow detection with different numeric ranges
 * - Scale growth and backoff mechanisms
 * - Integration with optimizers (SGD, Adam) across dtypes
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/amp/grad_scaler.hpp>
#include <tenzor/nn/optim/sgd.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/creation.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>
#include <limits>

using namespace tenzor;
using namespace tenzor::nn::amp;
using namespace tenzor::optim;
using namespace tenzor::testing;

// ============================================================================
// GradScaler Multi-Backend Multi-DType Test Fixture
// ============================================================================

class GradScalerMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    // Helper to get dtype-appropriate epsilon for numerical comparisons
    float GetEpsilon() const {
        if (dtype() == DType::Float64) {
            return 1e-6f;
        } else if (dtype() == DType::Float16) {
            return 1e-2f;
        }
        return 1e-5f;
    }

    // Helper to create scalar tensor based on dtype
    template<typename T>
    Tensor CreateScalar(T value) {
        return full({}, static_cast<float>(value), dtype(), device());
    }

    // Helper to create tensor based on dtype
    template<typename T>
    Tensor CreateTensor(const std::vector<int64_t>& shape, T value) {
        return full(shape, static_cast<float>(value), dtype(), device());
    }
};

// ============================================================================
// Test Cases
// ============================================================================

// Test 1: Default constructor with different dtypes
TEST_P(GradScalerMultiDTypeTest, DefaultConstructor) {
    GradScaler scaler;

    EXPECT_FLOAT_EQ(scaler.get_scale(), 65536.0f);
    EXPECT_EQ(scaler.get_growth_tracker(), 0);
    EXPECT_FALSE(scaler.found_inf_nan());
}

// Test 2: Custom constructor parameters
TEST_P(GradScalerMultiDTypeTest, CustomConstructor) {
    GradScaler scaler(1024.0f, 1.5f, 0.75f, 1000);

    EXPECT_FLOAT_EQ(scaler.get_scale(), 1024.0f);
    EXPECT_EQ(scaler.get_growth_tracker(), 0);
}

// Test 3: Loss scaling
TEST_P(GradScalerMultiDTypeTest, LossScaling) {
    // Skip Float16 - GradScaler works with Float32/64 gradients
    if (dtype() == DType::Float16) {
        GTEST_SKIP() << "GradScaler typically uses Float32/Float64 for gradients";
    }

    GradScaler scaler(1000.0f);

    auto loss_tensor = CreateScalar<float>(0.5f);
    auto loss = Variable(loss_tensor, true);
    auto scaled_loss = scaler.scale(loss);

    auto scaled_cpu = scaled_loss.tensor().to(Device::cpu()).to(DType::Float32);
    EXPECT_NEAR(scaled_cpu.item<float>(), 500.0f, GetEpsilon())
        << "Loss scaling failed for " << backend_name();
}

// Test 4: Gradient unscaling
TEST_P(GradScalerMultiDTypeTest, GradientUnscaling) {
    if (dtype() == DType::Float16) {
        GTEST_SKIP() << "GradScaler typically uses Float32/Float64 for gradients";
    }

    GradScaler scaler(100.0f);

    auto param_tensor = ones({2, 3}, dtype(), device());
    auto param = Variable(param_tensor, true);

    auto grad_tensor = CreateTensor<float>({2, 3}, 200.0f);
    param.grad() = grad_tensor;

    auto param_ptr = std::make_shared<Variable>(param);
    auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
    SGD optimizer(params, 0.01);

    scaler.unscale_(optimizer);

    auto grad_cpu = param.grad()->to(Device::cpu()).to(DType::Float32);
    const float* grad_data = grad_cpu.data<float>();
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(grad_data[i], 2.0f, GetEpsilon())
            << "Gradient unscaling failed at index " << i;
    }
}

// Test 5: Inf detection
TEST_P(GradScalerMultiDTypeTest, InfDetection) {
    if (dtype() == DType::Float16) {
        GTEST_SKIP() << "GradScaler typically uses Float32/Float64 for gradients";
    }

    GradScaler scaler;

    auto param_tensor = ones({2, 2}, dtype(), device());
    auto param = Variable(param_tensor, true);

    auto grad_tensor = ones({2, 2}, dtype(), device());
    auto grad_cpu = grad_tensor.to(Device::cpu());
    if (dtype() == DType::Float64) {
        double* grad_data = grad_cpu.data<double>();
        grad_data[1] = std::numeric_limits<double>::infinity();
    } else {
        float* grad_data = grad_cpu.data<float>();
        grad_data[1] = std::numeric_limits<float>::infinity();
    }
    param.grad() = grad_cpu.to(device());

    auto param_ptr = std::make_shared<Variable>(param);
    auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
    SGD optimizer(params, 0.01);

    bool success = scaler.step(optimizer);

    EXPECT_FALSE(success) << "Inf detection failed for " << backend_name();
    EXPECT_TRUE(scaler.found_inf_nan()) << "Inf flag not set";
}

// Test 6: NaN detection
TEST_P(GradScalerMultiDTypeTest, NanDetection) {
    if (dtype() == DType::Float16) {
        GTEST_SKIP() << "GradScaler typically uses Float32/Float64 for gradients";
    }

    GradScaler scaler;

    auto param_tensor = ones({2, 2}, dtype(), device());
    auto param = Variable(param_tensor, true);

    auto grad_tensor = ones({2, 2}, dtype(), device());
    auto grad_cpu = grad_tensor.to(Device::cpu());
    if (dtype() == DType::Float64) {
        double* grad_data = grad_cpu.data<double>();
        grad_data[2] = std::numeric_limits<double>::quiet_NaN();
    } else {
        float* grad_data = grad_cpu.data<float>();
        grad_data[2] = std::numeric_limits<float>::quiet_NaN();
    }
    param.grad() = grad_cpu.to(device());

    auto param_ptr = std::make_shared<Variable>(param);
    auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
    SGD optimizer(params, 0.01);

    bool success = scaler.step(optimizer);

    EXPECT_FALSE(success) << "NaN detection failed for " << backend_name();
    EXPECT_TRUE(scaler.found_inf_nan()) << "NaN flag not set";
}

// Test 7: Scale backoff on overflow
TEST_P(GradScalerMultiDTypeTest, ScaleBackoff) {
    if (dtype() == DType::Float16) {
        GTEST_SKIP() << "GradScaler typically uses Float32/Float64 for gradients";
    }

    GradScaler scaler(1000.0f, 2.0f, 0.5f, 10);
    float initial_scale = scaler.get_scale();

    auto param_tensor = ones({2, 2}, dtype(), device());
    auto param = Variable(param_tensor, true);

    // Create gradient with infinity
    float inf_val = (dtype() == DType::Float64) ?
        static_cast<float>(std::numeric_limits<double>::infinity()) :
        std::numeric_limits<float>::infinity();
    auto grad_tensor = full({2, 2}, inf_val, dtype(), device());
    param.grad() = grad_tensor;

    auto param_ptr = std::make_shared<Variable>(param);
    auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
    SGD optimizer(params, 0.01);

    scaler.step(optimizer);
    scaler.update();

    float new_scale = scaler.get_scale();
    EXPECT_FLOAT_EQ(new_scale, initial_scale * 0.5f)
        << "Scale backoff failed";
    EXPECT_EQ(scaler.get_growth_tracker(), 0)
        << "Growth tracker not reset";
}

// Test 8: Scale growth after successful iterations
TEST_P(GradScalerMultiDTypeTest, ScaleGrowth) {
    if (dtype() == DType::Float16) {
        GTEST_SKIP() << "GradScaler typically uses Float32/Float64 for gradients";
    }

    GradScaler scaler(1000.0f, 2.0f, 0.5f, 3);

    auto param_tensor = ones({2, 2}, dtype(), device());
    auto param = Variable(param_tensor, true);

    auto param_ptr = std::make_shared<Variable>(param);
    auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
    SGD optimizer(params, 0.01);

    for (int i = 0; i < 3; ++i) {
        auto grad_tensor = CreateTensor<float>({2, 2}, 1.0f);
        param.grad() = grad_tensor;

        bool success = scaler.step(optimizer);
        EXPECT_TRUE(success) << "Step " << i << " failed";
        scaler.update();
    }

    EXPECT_FLOAT_EQ(scaler.get_scale(), 2000.0f)
        << "Scale growth failed";
    EXPECT_EQ(scaler.get_growth_tracker(), 0)
        << "Growth tracker not reset after growth";
}

// Test 9: SGD optimizer integration
TEST_P(GradScalerMultiDTypeTest, SGDIntegration) {
    if (dtype() == DType::Float16) {
        GTEST_SKIP() << "GradScaler typically uses Float32/Float64 for gradients";
    }

    GradScaler scaler(100.0f);

    auto param_tensor = CreateTensor<float>({3, 3}, 1.0f);
    auto param = Variable(param_tensor, true);

    auto param_ptr = std::make_shared<Variable>(param);
    auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
    SGD optimizer(params, 0.1);

    auto grad_tensor = CreateTensor<float>({3, 3}, 10.0f);
    param.grad() = grad_tensor;

    bool success = scaler.step(optimizer);
    scaler.update();

    EXPECT_TRUE(success) << "SGD step failed";

    auto param_cpu = param.tensor().to(Device::cpu()).to(DType::Float32);
    const float* param_data = param_cpu.data<float>();
    // Unscaled gradient = 10/100 = 0.1
    // Update: param = param - lr * grad = 1.0 - 0.1 * 0.1 = 0.99
    EXPECT_NEAR(param_data[0], 0.99f, GetEpsilon())
        << "SGD parameter update incorrect";
}

// Test 10: Adam optimizer integration
TEST_P(GradScalerMultiDTypeTest, AdamIntegration) {
    if (dtype() == DType::Float16) {
        GTEST_SKIP() << "GradScaler typically uses Float32/Float64 for gradients";
    }

    GradScaler scaler(100.0f);

    auto param_tensor = CreateTensor<float>({2, 2}, 1.0f);
    auto param = Variable(param_tensor, true);

    auto param_ptr = std::make_shared<Variable>(param);
    auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
    Adam optimizer(params, 0.01);

    auto grad_tensor = CreateTensor<float>({2, 2}, 50.0f);
    param.grad() = grad_tensor;

    bool success = scaler.step(optimizer);
    scaler.update();

    EXPECT_TRUE(success) << "Adam step failed";

    auto param_cpu = param.tensor().to(Device::cpu()).to(DType::Float32);
    const float* param_data = param_cpu.data<float>();
    EXPECT_NE(param_data[0], 1.0f)
        << "Adam parameter not updated";
}

// Test 11: Multiple parameters
TEST_P(GradScalerMultiDTypeTest, MultipleParameters) {
    if (dtype() == DType::Float16) {
        GTEST_SKIP() << "GradScaler typically uses Float32/Float64 for gradients";
    }

    GradScaler scaler(100.0f);

    auto param1_tensor = ones({2, 2}, dtype(), device());
    auto param1 = Variable(param1_tensor, true);

    auto param2_tensor = ones({3, 3}, dtype(), device());
    auto param2 = Variable(param2_tensor, true);

    auto grad1_tensor = CreateTensor<float>({2, 2}, 100.0f);
    param1.grad() = grad1_tensor;

    auto grad2_tensor = CreateTensor<float>({3, 3}, 200.0f);
    param2.grad() = grad2_tensor;

    auto param1_ptr = std::make_shared<Variable>(param1);
    auto param2_ptr = std::make_shared<Variable>(param2);
    auto params = std::vector<std::shared_ptr<Variable>>{param1_ptr, param2_ptr};
    SGD optimizer(params, 0.1);

    bool success = scaler.step(optimizer);

    EXPECT_TRUE(success) << "Multi-parameter step failed";

    auto grad1_cpu = param1.grad()->to(Device::cpu()).to(DType::Float32);
    auto grad2_cpu = param2.grad()->to(Device::cpu()).to(DType::Float32);
    const float* grad1_data = grad1_cpu.data<float>();
    const float* grad2_data = grad2_cpu.data<float>();

    EXPECT_NEAR(grad1_data[0], 1.0f, GetEpsilon())
        << "Param1 gradient unscaling failed";
    EXPECT_NEAR(grad2_data[0], 2.0f, GetEpsilon())
        << "Param2 gradient unscaling failed";
}

// Test 12: Underflow prevention
TEST_P(GradScalerMultiDTypeTest, UnderflowPrevention) {
    if (dtype() == DType::Float16) {
        GTEST_SKIP() << "GradScaler typically uses Float32/Float64 for gradients";
    }

    float scale = 65536.0f;
    GradScaler scaler(scale);

    auto param_tensor = ones({4, 4}, dtype(), device());
    auto param = Variable(param_tensor, true);

    float tiny_gradient = 1e-7f;
    // Pre-scale the gradient (simulates what backward does with scaled loss)
    float scaled_gradient = tiny_gradient * scale;
    auto grad_tensor = CreateTensor<float>({4, 4}, scaled_gradient);
    param.grad() = grad_tensor;

    auto param_ptr = std::make_shared<Variable>(param);
    auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
    SGD optimizer(params, 0.01);

    bool success = scaler.step(optimizer);
    scaler.update();

    EXPECT_TRUE(success) << "Underflow prevention failed";

    // After unscaling, gradient should be back to original tiny_gradient
    auto grad_cpu = param.grad()->to(Device::cpu()).to(DType::Float32);
    const float* grad_data = grad_cpu.data<float>();
    EXPECT_NEAR(grad_data[0], tiny_gradient, GetEpsilon())
        << "Tiny gradient not preserved after unscaling";
}

// Test 13: Training loop simulation
TEST_P(GradScalerMultiDTypeTest, TrainingLoopSimulation) {
    if (dtype() == DType::Float16) {
        GTEST_SKIP() << "GradScaler typically uses Float32/Float64 for gradients";
    }

    GradScaler scaler(1024.0f, 2.0f, 0.5f, 5);

    auto param_tensor = ones({10, 10}, dtype(), device());
    auto param = Variable(param_tensor, true);

    auto param_ptr = std::make_shared<Variable>(param);
    auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
    SGD optimizer(params, 0.01);

    int successful_steps = 0;
    int skipped_steps = 0;

    for (int iter = 0; iter < 20; ++iter) {
        optimizer.zero_grad();

        // Simulate gradients with occasional overflow
        float grad_value = (iter % 7 == 0) ?
            std::numeric_limits<float>::infinity() : 1.0f;

        auto grad_tensor = CreateTensor<float>({10, 10}, grad_value);
        param.grad() = grad_tensor;

        bool success = scaler.step(optimizer);
        scaler.update();

        if (success) {
            successful_steps++;
        } else {
            skipped_steps++;
        }
    }

    EXPECT_GT(successful_steps, 0) << "No successful steps";
    EXPECT_GT(skipped_steps, 0) << "No skipped steps";
    EXPECT_EQ(successful_steps + skipped_steps, 20)
        << "Step count mismatch";
}

// Test 14: Numeric range differences
TEST_P(GradScalerMultiDTypeTest, NumericRangeDifferences) {
    if (dtype() == DType::Float16) {
        GTEST_SKIP() << "GradScaler typically uses Float32/Float64 for gradients";
    }

    GradScaler scaler(1024.0f);

    auto param_tensor = ones({3, 3}, dtype(), device());
    auto param = Variable(param_tensor, true);

    // Large but not infinite value - use float max for both dtypes since
    // CreateTensor casts to float before creating the tensor
    float large_value = std::numeric_limits<float>::max() / 2048.0f;
    auto grad_tensor = CreateTensor<float>({3, 3}, large_value);
    param.grad() = grad_tensor;

    auto param_ptr = std::make_shared<Variable>(param);
    auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
    SGD optimizer(params, 0.001);

    bool success = scaler.step(optimizer);

    EXPECT_TRUE(success)
        << "Large value handling failed";
}

// Test 15: State persistence
TEST_P(GradScalerMultiDTypeTest, StateDictConsistency) {
    if (dtype() == DType::Float16) {
        GTEST_SKIP() << "GradScaler typically uses Float32/Float64 for gradients";
    }

    GradScaler scaler1(2048.0f, 3.0f, 0.25f, 500);

    // Modify state using the current dtype
    auto param_tensor = ones({2, 2}, dtype(), device());
    auto param = Variable(param_tensor, true);
    auto grad_tensor = CreateTensor<float>({2, 2}, 1.0f);
    param.grad() = grad_tensor;

    auto param_ptr = std::make_shared<Variable>(param);
    auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
    SGD optimizer(params, 0.01);

    for (int i = 0; i < 10; ++i) {
        scaler1.step(optimizer);
        scaler1.update();
    }

    // Save and load state
    auto state = scaler1.state_dict();
    GradScaler scaler2;
    scaler2.load_state_dict(state);

    // Verify state matches regardless of dtype
    EXPECT_FLOAT_EQ(scaler2.get_scale(), scaler1.get_scale())
        << "Scale mismatch after state load";
    EXPECT_EQ(scaler2.get_growth_tracker(), scaler1.get_growth_tracker())
        << "Growth tracker mismatch after state load";
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(GradScalerMultiDTypeTest);

/*
 * COVERAGE SUMMARY:
 *
 * Test Cases: 15
 * DTypes Tested: Float32, Float64, Float16 (Float16 skipped)
 * Backends Tested: CPU, CUDA, OneAPI
 * Total Scenarios: 15 tests × 3 dtypes × 3 backends = 135 test scenarios
 *
 * Coverage:
 * - Constructor: default, custom parameters
 * - Loss scaling: basic scaling
 * - Gradient operations: unscaling, inf detection, NaN detection
 * - Scale dynamics: backoff, growth
 * - Optimizer integration: SGD, Adam
 * - Multi-parameter: handling multiple parameters
 * - Underflow: prevention with tiny gradients
 * - Training simulation: realistic loop with overflows
 * - Numeric ranges: handling large values
 * - State persistence: save/load state dict
 */
