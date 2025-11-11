/**
 * @file test_grad_scaler_multidtype.cpp
 * @brief Multi-dtype tests for GradScaler (AMP gradient scaling)
 *
 * Coverage: DType testing for gradient scaling in mixed precision training
 * - Float32 and Float64 dtypes (primary types for gradient computation)
 * - Scaling behavior across different precision levels
 * - Underflow/overflow detection with different numeric ranges
 * - Scale growth and backoff mechanisms
 * - Integration with optimizers (SGD, Adam) across dtypes
 *
 * GradScaler is essential for automatic mixed precision (AMP) training:
 * - Prevents gradient underflow in lower precision (FP16)
 * - Dynamic loss scaling for stable training
 * - Overflow detection and automatic scale adjustment
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/amp/grad_scaler.hpp>
#include <tenzor/nn/optim/sgd.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/creation.hpp>
#include <cmath>
#include <limits>

using namespace tenzor;
using namespace tenzor::nn::amp;
using namespace tenzor::optim;

// ============================================================================
// DType Parameterization for GradScaler
// ============================================================================

struct GradScalerDTypeParam {
    DType dtype;
    std::string dtype_name;

    std::string ToString() const {
        return dtype_name;
    }
};

class GradScalerMultiDTypeTest : public ::testing::TestWithParam<GradScalerDTypeParam> {
protected:
    Device device;
    DType dtype;

    void SetUp() override {
        tenzor::initialize();
        device = Device::cpu();
        dtype = GetParam().dtype;
    }

    // Helper to get dtype-appropriate epsilon for numerical comparisons
    template<typename T>
    T GetEpsilon() const {
        if constexpr (std::is_same_v<T, float>) {
            return 1e-5f;
        } else if constexpr (std::is_same_v<T, double>) {
            return 1e-10;
        }
        return T(0);
    }

    // Helper to verify tensor values based on dtype
    template<typename T>
    void VerifyTensorValue(const Tensor& tensor, T expected_value, const std::string& test_name) {
        const T* data = tensor.data<T>();
        T epsilon = GetEpsilon<T>();

        for (int64_t i = 0; i < tensor.numel(); ++i) {
            if constexpr (std::is_floating_point_v<T>) {
                EXPECT_NEAR(data[i], expected_value, epsilon)
                    << test_name << " failed at index " << i
                    << " for dtype " << GetParam().dtype_name;
            }
        }
    }

    // Helper to create scalar tensor based on dtype
    template<typename T>
    Tensor CreateScalar(T value) {
        return full({}, value, dtype, device);
    }

    // Helper to create tensor based on dtype
    template<typename T>
    Tensor CreateTensor(const std::vector<int64_t>& shape, T value) {
        return full(shape, value, dtype, device);
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

// Test 3: Loss scaling with Float32
TEST_P(GradScalerMultiDTypeTest, LossScaling) {
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        GTEST_SKIP() << "Test only applicable for floating point types";
    }

    GradScaler scaler(1000.0f);

    if (dtype == DType::Float32) {
        auto loss_tensor = CreateScalar<float>(0.5f);
        auto loss = Variable(loss_tensor, true);
        auto scaled_loss = scaler.scale(loss);

        EXPECT_NEAR(scaled_loss.tensor().item<float>(), 500.0f, 1e-5f)
            << "Float32 loss scaling failed";
    } else if (dtype == DType::Float64) {
        auto loss_tensor = CreateScalar<double>(0.5);
        auto loss = Variable(loss_tensor, true);
        auto scaled_loss = scaler.scale(loss);

        EXPECT_NEAR(scaled_loss.tensor().item<double>(), 500.0, 1e-10)
            << "Float64 loss scaling failed";
    }
}

// Test 4: Gradient unscaling with different dtypes
TEST_P(GradScalerMultiDTypeTest, GradientUnscaling) {
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        GTEST_SKIP() << "Test only applicable for floating point types";
    }

    GradScaler scaler(100.0f);

    if (dtype == DType::Float32) {
        auto param_tensor = ones({2, 3}, dtype, device);
        auto param = Variable(param_tensor, true);

        auto grad_tensor = CreateTensor<float>({2, 3}, 200.0f);
        param.grad() = grad_tensor;

        auto param_ptr = std::make_shared<Variable>(param);
        auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
        SGD optimizer(params, 0.01);

        scaler.unscale_(optimizer);

        const float* grad_data = param.grad()->data<float>();
        for (int i = 0; i < 6; ++i) {
            EXPECT_NEAR(grad_data[i], 2.0f, 1e-5f)
                << "Float32 gradient unscaling failed at index " << i;
        }
    } else if (dtype == DType::Float64) {
        auto param_tensor = ones({2, 3}, dtype, device);
        auto param = Variable(param_tensor, true);

        auto grad_tensor = CreateTensor<double>({2, 3}, 200.0);
        param.grad() = grad_tensor;

        auto param_ptr = std::make_shared<Variable>(param);
        auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
        SGD optimizer(params, 0.01);

        scaler.unscale_(optimizer);

        const double* grad_data = param.grad()->data<double>();
        for (int i = 0; i < 6; ++i) {
            EXPECT_NEAR(grad_data[i], 2.0, 1e-10)
                << "Float64 gradient unscaling failed at index " << i;
        }
    }
}

// Test 5: Inf detection across dtypes
TEST_P(GradScalerMultiDTypeTest, InfDetection) {
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        GTEST_SKIP() << "Test only applicable for floating point types";
    }

    GradScaler scaler;

    if (dtype == DType::Float32) {
        auto param_tensor = ones({2, 2}, dtype, device);
        auto param = Variable(param_tensor, true);

        auto grad_tensor = CreateTensor<float>({2, 2}, 1.0f);
        float* grad_data = grad_tensor.data<float>();
        grad_data[1] = std::numeric_limits<float>::infinity();
        param.grad() = grad_tensor;

        auto param_ptr = std::make_shared<Variable>(param);
        auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
        SGD optimizer(params, 0.01);

        bool success = scaler.step(optimizer);

        EXPECT_FALSE(success) << "Float32 inf detection failed";
        EXPECT_TRUE(scaler.found_inf_nan()) << "Float32 inf flag not set";
    } else if (dtype == DType::Float64) {
        auto param_tensor = ones({2, 2}, dtype, device);
        auto param = Variable(param_tensor, true);

        auto grad_tensor = CreateTensor<double>({2, 2}, 1.0);
        double* grad_data = grad_tensor.data<double>();
        grad_data[1] = std::numeric_limits<double>::infinity();
        param.grad() = grad_tensor;

        auto param_ptr = std::make_shared<Variable>(param);
        auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
        SGD optimizer(params, 0.01);

        bool success = scaler.step(optimizer);

        EXPECT_FALSE(success) << "Float64 inf detection failed";
        EXPECT_TRUE(scaler.found_inf_nan()) << "Float64 inf flag not set";
    }
}

// Test 6: NaN detection across dtypes
TEST_P(GradScalerMultiDTypeTest, NanDetection) {
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        GTEST_SKIP() << "Test only applicable for floating point types";
    }

    GradScaler scaler;

    if (dtype == DType::Float32) {
        auto param_tensor = ones({2, 2}, dtype, device);
        auto param = Variable(param_tensor, true);

        auto grad_tensor = CreateTensor<float>({2, 2}, 1.0f);
        float* grad_data = grad_tensor.data<float>();
        grad_data[2] = std::numeric_limits<float>::quiet_NaN();
        param.grad() = grad_tensor;

        auto param_ptr = std::make_shared<Variable>(param);
        auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
        SGD optimizer(params, 0.01);

        bool success = scaler.step(optimizer);

        EXPECT_FALSE(success) << "Float32 NaN detection failed";
        EXPECT_TRUE(scaler.found_inf_nan()) << "Float32 NaN flag not set";
    } else if (dtype == DType::Float64) {
        auto param_tensor = ones({2, 2}, dtype, device);
        auto param = Variable(param_tensor, true);

        auto grad_tensor = CreateTensor<double>({2, 2}, 1.0);
        double* grad_data = grad_tensor.data<double>();
        grad_data[2] = std::numeric_limits<double>::quiet_NaN();
        param.grad() = grad_tensor;

        auto param_ptr = std::make_shared<Variable>(param);
        auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
        SGD optimizer(params, 0.01);

        bool success = scaler.step(optimizer);

        EXPECT_FALSE(success) << "Float64 NaN detection failed";
        EXPECT_TRUE(scaler.found_inf_nan()) << "Float64 NaN flag not set";
    }
}

// Test 7: Scale backoff on overflow
TEST_P(GradScalerMultiDTypeTest, ScaleBackoff) {
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        GTEST_SKIP() << "Test only applicable for floating point types";
    }

    GradScaler scaler(1000.0f, 2.0f, 0.5f, 10);
    float initial_scale = scaler.get_scale();

    if (dtype == DType::Float32) {
        auto param_tensor = ones({2, 2}, dtype, device);
        auto param = Variable(param_tensor, true);

        auto grad_tensor = full({2, 2}, std::numeric_limits<float>::infinity(),
                               dtype, device);
        param.grad() = grad_tensor;

        auto param_ptr = std::make_shared<Variable>(param);
        auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
        SGD optimizer(params, 0.01);

        scaler.step(optimizer);
        scaler.update();

        float new_scale = scaler.get_scale();
        EXPECT_FLOAT_EQ(new_scale, initial_scale * 0.5f)
            << "Float32 scale backoff failed";
        EXPECT_EQ(scaler.get_growth_tracker(), 0)
            << "Float32 growth tracker not reset";
    } else if (dtype == DType::Float64) {
        auto param_tensor = ones({2, 2}, dtype, device);
        auto param = Variable(param_tensor, true);

        auto grad_tensor = full({2, 2}, std::numeric_limits<double>::infinity(),
                               dtype, device);
        param.grad() = grad_tensor;

        auto param_ptr = std::make_shared<Variable>(param);
        auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
        SGD optimizer(params, 0.01);

        scaler.step(optimizer);
        scaler.update();

        float new_scale = scaler.get_scale();
        EXPECT_FLOAT_EQ(new_scale, initial_scale * 0.5f)
            << "Float64 scale backoff failed";
        EXPECT_EQ(scaler.get_growth_tracker(), 0)
            << "Float64 growth tracker not reset";
    }
}

// Test 8: Scale growth after successful iterations
TEST_P(GradScalerMultiDTypeTest, ScaleGrowth) {
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        GTEST_SKIP() << "Test only applicable for floating point types";
    }

    GradScaler scaler(1000.0f, 2.0f, 0.5f, 3);

    if (dtype == DType::Float32) {
        auto param_tensor = ones({2, 2}, dtype, device);
        auto param = Variable(param_tensor, true);

        auto param_ptr = std::make_shared<Variable>(param);
        auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
        SGD optimizer(params, 0.01);

        for (int i = 0; i < 3; ++i) {
            auto grad_tensor = CreateTensor<float>({2, 2}, 1.0f);
            param.grad() = grad_tensor;

            bool success = scaler.step(optimizer);
            EXPECT_TRUE(success) << "Float32 step " << i << " failed";
            scaler.update();
        }

        EXPECT_FLOAT_EQ(scaler.get_scale(), 2000.0f)
            << "Float32 scale growth failed";
        EXPECT_EQ(scaler.get_growth_tracker(), 0)
            << "Float32 growth tracker not reset after growth";
    } else if (dtype == DType::Float64) {
        auto param_tensor = ones({2, 2}, dtype, device);
        auto param = Variable(param_tensor, true);

        auto param_ptr = std::make_shared<Variable>(param);
        auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
        SGD optimizer(params, 0.01);

        for (int i = 0; i < 3; ++i) {
            auto grad_tensor = CreateTensor<double>({2, 2}, 1.0);
            param.grad() = grad_tensor;

            bool success = scaler.step(optimizer);
            EXPECT_TRUE(success) << "Float64 step " << i << " failed";
            scaler.update();
        }

        EXPECT_FLOAT_EQ(scaler.get_scale(), 2000.0f)
            << "Float64 scale growth failed";
        EXPECT_EQ(scaler.get_growth_tracker(), 0)
            << "Float64 growth tracker not reset after growth";
    }
}

// Test 9: SGD optimizer integration
TEST_P(GradScalerMultiDTypeTest, SGDIntegration) {
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        GTEST_SKIP() << "Test only applicable for floating point types";
    }

    GradScaler scaler(100.0f);

    if (dtype == DType::Float32) {
        auto param_tensor = CreateTensor<float>({3, 3}, 1.0f);
        auto param = Variable(param_tensor, true);

        auto param_ptr = std::make_shared<Variable>(param);
        auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
        SGD optimizer(params, 0.1);

        auto grad_tensor = CreateTensor<float>({3, 3}, 10.0f);
        param.grad() = grad_tensor;

        bool success = scaler.step(optimizer);
        scaler.update();

        EXPECT_TRUE(success) << "Float32 SGD step failed";

        const float* param_data = param.tensor().data<float>();
        // Unscaled gradient = 10/100 = 0.1
        // Update: param = param - lr * grad = 1.0 - 0.1 * 0.1 = 0.99
        EXPECT_NEAR(param_data[0], 0.99f, 1e-5f)
            << "Float32 SGD parameter update incorrect";
    } else if (dtype == DType::Float64) {
        auto param_tensor = CreateTensor<double>({3, 3}, 1.0);
        auto param = Variable(param_tensor, true);

        auto param_ptr = std::make_shared<Variable>(param);
        auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
        SGD optimizer(params, 0.1);

        auto grad_tensor = CreateTensor<double>({3, 3}, 10.0);
        param.grad() = grad_tensor;

        bool success = scaler.step(optimizer);
        scaler.update();

        EXPECT_TRUE(success) << "Float64 SGD step failed";

        const double* param_data = param.tensor().data<double>();
        // Unscaled gradient = 10/100 = 0.1
        // Update: param = param - lr * grad = 1.0 - 0.1 * 0.1 = 0.99
        EXPECT_NEAR(param_data[0], 0.99, 1e-10)
            << "Float64 SGD parameter update incorrect";
    }
}

// Test 10: Adam optimizer integration
TEST_P(GradScalerMultiDTypeTest, AdamIntegration) {
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        GTEST_SKIP() << "Test only applicable for floating point types";
    }

    GradScaler scaler(100.0f);

    if (dtype == DType::Float32) {
        auto param_tensor = CreateTensor<float>({2, 2}, 1.0f);
        auto param = Variable(param_tensor, true);

        auto param_ptr = std::make_shared<Variable>(param);
        auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
        Adam optimizer(params, 0.01);

        auto grad_tensor = CreateTensor<float>({2, 2}, 50.0f);
        param.grad() = grad_tensor;

        bool success = scaler.step(optimizer);
        scaler.update();

        EXPECT_TRUE(success) << "Float32 Adam step failed";

        const float* param_data = param.tensor().data<float>();
        EXPECT_NE(param_data[0], 1.0f)
            << "Float32 Adam parameter not updated";
    } else if (dtype == DType::Float64) {
        auto param_tensor = CreateTensor<double>({2, 2}, 1.0);
        auto param = Variable(param_tensor, true);

        auto param_ptr = std::make_shared<Variable>(param);
        auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
        Adam optimizer(params, 0.01);

        auto grad_tensor = CreateTensor<double>({2, 2}, 50.0);
        param.grad() = grad_tensor;

        bool success = scaler.step(optimizer);
        scaler.update();

        EXPECT_TRUE(success) << "Float64 Adam step failed";

        const double* param_data = param.tensor().data<double>();
        EXPECT_NE(param_data[0], 1.0)
            << "Float64 Adam parameter not updated";
    }
}

// Test 11: Multiple parameters with different dtypes
TEST_P(GradScalerMultiDTypeTest, MultipleParameters) {
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        GTEST_SKIP() << "Test only applicable for floating point types";
    }

    GradScaler scaler(100.0f);

    if (dtype == DType::Float32) {
        auto param1_tensor = ones({2, 2}, dtype, device);
        auto param1 = Variable(param1_tensor, true);

        auto param2_tensor = ones({3, 3}, dtype, device);
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

        EXPECT_TRUE(success) << "Float32 multi-parameter step failed";

        const float* grad1_data = param1.grad()->data<float>();
        const float* grad2_data = param2.grad()->data<float>();

        EXPECT_NEAR(grad1_data[0], 1.0f, 1e-5f)
            << "Float32 param1 gradient unscaling failed";
        EXPECT_NEAR(grad2_data[0], 2.0f, 1e-5f)
            << "Float32 param2 gradient unscaling failed";
    } else if (dtype == DType::Float64) {
        auto param1_tensor = ones({2, 2}, dtype, device);
        auto param1 = Variable(param1_tensor, true);

        auto param2_tensor = ones({3, 3}, dtype, device);
        auto param2 = Variable(param2_tensor, true);

        auto grad1_tensor = CreateTensor<double>({2, 2}, 100.0);
        param1.grad() = grad1_tensor;

        auto grad2_tensor = CreateTensor<double>({3, 3}, 200.0);
        param2.grad() = grad2_tensor;

        auto param1_ptr = std::make_shared<Variable>(param1);
        auto param2_ptr = std::make_shared<Variable>(param2);
        auto params = std::vector<std::shared_ptr<Variable>>{param1_ptr, param2_ptr};
        SGD optimizer(params, 0.1);

        bool success = scaler.step(optimizer);

        EXPECT_TRUE(success) << "Float64 multi-parameter step failed";

        const double* grad1_data = param1.grad()->data<double>();
        const double* grad2_data = param2.grad()->data<double>();

        EXPECT_NEAR(grad1_data[0], 1.0, 1e-10)
            << "Float64 param1 gradient unscaling failed";
        EXPECT_NEAR(grad2_data[0], 2.0, 1e-10)
            << "Float64 param2 gradient unscaling failed";
    }
}

// Test 12: Underflow prevention (main purpose of scaling)
TEST_P(GradScalerMultiDTypeTest, UnderflowPrevention) {
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        GTEST_SKIP() << "Test only applicable for floating point types";
    }

    // Large scale to prevent underflow of small gradients
    GradScaler scaler(65536.0f);

    if (dtype == DType::Float32) {
        // Very small gradient that might underflow in FP16
        auto param_tensor = ones({4, 4}, dtype, device);
        auto param = Variable(param_tensor, true);

        float tiny_gradient = 1e-7f;
        auto grad_tensor = CreateTensor<float>({4, 4}, tiny_gradient);
        param.grad() = grad_tensor;

        auto param_ptr = std::make_shared<Variable>(param);
        auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
        SGD optimizer(params, 0.01);

        bool success = scaler.step(optimizer);
        scaler.update();

        EXPECT_TRUE(success) << "Float32 underflow prevention failed";

        // Verify gradient was preserved through scaling
        const float* grad_data = param.grad()->data<float>();
        EXPECT_NEAR(grad_data[0], tiny_gradient, 1e-12f)
            << "Float32 tiny gradient not preserved";
    } else if (dtype == DType::Float64) {
        // Very small gradient for Float64
        auto param_tensor = ones({4, 4}, dtype, device);
        auto param = Variable(param_tensor, true);

        double tiny_gradient = 1e-15;
        auto grad_tensor = CreateTensor<double>({4, 4}, tiny_gradient);
        param.grad() = grad_tensor;

        auto param_ptr = std::make_shared<Variable>(param);
        auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
        SGD optimizer(params, 0.01);

        bool success = scaler.step(optimizer);
        scaler.update();

        EXPECT_TRUE(success) << "Float64 underflow prevention failed";

        // Verify gradient was preserved through scaling
        const double* grad_data = param.grad()->data<double>();
        EXPECT_NEAR(grad_data[0], tiny_gradient, 1e-20)
            << "Float64 tiny gradient not preserved";
    }
}

// Test 13: Training loop simulation with different dtypes
TEST_P(GradScalerMultiDTypeTest, TrainingLoopSimulation) {
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        GTEST_SKIP() << "Test only applicable for floating point types";
    }

    GradScaler scaler(1024.0f, 2.0f, 0.5f, 5);

    if (dtype == DType::Float32) {
        auto param_tensor = ones({10, 10}, dtype, device);
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

        EXPECT_GT(successful_steps, 0) << "Float32 no successful steps";
        EXPECT_GT(skipped_steps, 0) << "Float32 no skipped steps";
        EXPECT_EQ(successful_steps + skipped_steps, 20)
            << "Float32 step count mismatch";
    } else if (dtype == DType::Float64) {
        auto param_tensor = ones({10, 10}, dtype, device);
        auto param = Variable(param_tensor, true);

        auto param_ptr = std::make_shared<Variable>(param);
        auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
        SGD optimizer(params, 0.01);

        int successful_steps = 0;
        int skipped_steps = 0;

        for (int iter = 0; iter < 20; ++iter) {
            optimizer.zero_grad();

            // Simulate gradients with occasional overflow
            double grad_value = (iter % 7 == 0) ?
                std::numeric_limits<double>::infinity() : 1.0;

            auto grad_tensor = CreateTensor<double>({10, 10}, grad_value);
            param.grad() = grad_tensor;

            bool success = scaler.step(optimizer);
            scaler.update();

            if (success) {
                successful_steps++;
            } else {
                skipped_steps++;
            }
        }

        EXPECT_GT(successful_steps, 0) << "Float64 no successful steps";
        EXPECT_GT(skipped_steps, 0) << "Float64 no skipped steps";
        EXPECT_EQ(successful_steps + skipped_steps, 20)
            << "Float64 step count mismatch";
    }
}

// Test 14: Numeric range differences between Float32 and Float64
TEST_P(GradScalerMultiDTypeTest, NumericRangeDifferences) {
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        GTEST_SKIP() << "Test only applicable for floating point types";
    }

    GradScaler scaler(1024.0f);

    if (dtype == DType::Float32) {
        // Test near Float32 limits
        auto param_tensor = ones({3, 3}, dtype, device);
        auto param = Variable(param_tensor, true);

        // Large but not infinite value
        float large_value = std::numeric_limits<float>::max() / 2048.0f;
        auto grad_tensor = CreateTensor<float>({3, 3}, large_value);
        param.grad() = grad_tensor;

        auto param_ptr = std::make_shared<Variable>(param);
        auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
        SGD optimizer(params, 0.001);

        bool success = scaler.step(optimizer);

        EXPECT_TRUE(success)
            << "Float32 large value handling failed";
    } else if (dtype == DType::Float64) {
        // Test near Float64 limits (much larger range)
        auto param_tensor = ones({3, 3}, dtype, device);
        auto param = Variable(param_tensor, true);

        // Large value that would overflow Float32
        double large_value = std::numeric_limits<double>::max() / 2048.0;
        auto grad_tensor = CreateTensor<double>({3, 3}, large_value);
        param.grad() = grad_tensor;

        auto param_ptr = std::make_shared<Variable>(param);
        auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
        SGD optimizer(params, 0.001);

        bool success = scaler.step(optimizer);

        EXPECT_TRUE(success)
            << "Float64 large value handling failed";
    }
}

// Test 15: State persistence across dtypes
TEST_P(GradScalerMultiDTypeTest, StateDictConsistency) {
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        GTEST_SKIP() << "Test only applicable for floating point types";
    }

    GradScaler scaler1(2048.0f, 3.0f, 0.25f, 500);

    // Modify state using the current dtype
    if (dtype == DType::Float32) {
        auto param_tensor = ones({2, 2}, dtype, device);
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
    } else if (dtype == DType::Float64) {
        auto param_tensor = ones({2, 2}, dtype, device);
        auto param = Variable(param_tensor, true);
        auto grad_tensor = CreateTensor<double>({2, 2}, 1.0);
        param.grad() = grad_tensor;

        auto param_ptr = std::make_shared<Variable>(param);
        auto params = std::vector<std::shared_ptr<Variable>>{param_ptr};
        SGD optimizer(params, 0.01);

        for (int i = 0; i < 10; ++i) {
            scaler1.step(optimizer);
            scaler1.update();
        }
    }

    // Save and load state
    auto state = scaler1.state_dict();
    GradScaler scaler2;
    scaler2.load_state_dict(state);

    // Verify state matches regardless of dtype
    EXPECT_FLOAT_EQ(scaler2.get_scale(), scaler1.get_scale())
        << "Scale mismatch after state load for " << GetParam().dtype_name;
    EXPECT_EQ(scaler2.get_growth_tracker(), scaler1.get_growth_tracker())
        << "Growth tracker mismatch after state load for " << GetParam().dtype_name;
}

// ============================================================================
// Parameterization
// ============================================================================

INSTANTIATE_TEST_SUITE_P(
    FloatDTypes,
    GradScalerMultiDTypeTest,
    ::testing::Values(
        GradScalerDTypeParam{DType::Float32, "Float32"},
        GradScalerDTypeParam{DType::Float64, "Float64"}
    ),
    [](const ::testing::TestParamInfo<GradScalerDTypeParam>& info) {
        return info.param.ToString();
    }
);

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
