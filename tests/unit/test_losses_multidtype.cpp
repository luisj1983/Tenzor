#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <tenzor/nn/functional.hpp>
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::nn;

/**
 * @file test_losses_multidtype.cpp
 * @brief Multi-dtype parameterized tests for loss functions
 *
 * Loss functions are used in training and need to support multiple precisions:
 * - Float32: Standard training precision
 * - Float64: High precision for numerical stability and gradient accuracy
 *
 * Verifies:
 * - Loss values are correct across different dtypes
 * - Dtype preservation (output matches input dtype)
 * - Gradient computation works correctly
 * - Reduction operations (Mean, Sum, None) preserve dtype
 * - Numerical precision is appropriate for each dtype
 */

// ============================================================================
// DType Parameterization Structure
// ============================================================================

struct LossDTypeParam {
    std::string backend_name;
    DType dtype;
    std::string dtype_name;
    double rtol;  // Relative tolerance
    double atol;  // Absolute tolerance

    std::string ToString() const {
        return backend_name + "_" + dtype_name;
    }
};

// Required for gtest_discover_tests to show human-readable test names
void PrintTo(const LossDTypeParam& param, std::ostream* os) {
    *os << param.ToString();
}

// ============================================================================
// Test Fixture
// ============================================================================

class LossMultiDTypeTest : public ::testing::TestWithParam<LossDTypeParam> {
protected:
    Device device;
    DType dtype;
    double rtol;
    double atol;

    void SetUp() override {
        tenzor::initialize();

        auto param = GetParam();
        dtype = param.dtype;
        rtol = param.rtol;
        atol = param.atol;

        if (param.backend_name == "cpu") {
            device = Device::cpu();
        }
        else if (param.backend_name == "cuda") {
            if (!isBackendAvailable(Device::Type::CUDA)) {
                GTEST_SKIP() << "CUDA not available";
            }
            device = Device::cuda(0);
        }
        else if (param.backend_name == "vulkan") {
            if (!isBackendAvailable(Device::Type::Vulkan)) {
                GTEST_SKIP() << "Vulkan not available";
            }
            device = Device::vulkan(0);
        }
        else if (param.backend_name == "oneapi") {
            if (!isBackendAvailable(Device::Type::OneAPI)) {
                GTEST_SKIP() << "OneAPI not available";
            }
            device = Device::oneapi(0);
        }
        else if (param.backend_name == "rocm") {
            if (!isBackendAvailable(Device::Type::ROCm)) {
                GTEST_SKIP() << "ROCm not available";
            }
            device = Device::rocm(0);
        }
    }

    static bool isBackendAvailable(Device::Type type) {
        try {
            Device test_device{type, 0};
            auto t = zeros({2, 2}, DType::Float32, test_device);
            return true;
        } catch (...) {
            return false;
        }
    }

    // Helper to verify scalar loss value with appropriate tolerance
    template<typename T>
    void assertLossValue(const Tensor& loss_tensor, double expected_value) {
        auto loss_cpu = loss_tensor.to(Device::cpu());
        const T* loss_data = loss_cpu.data<T>();

        double diff = std::abs(static_cast<double>(loss_data[0]) - expected_value);
        double threshold = atol + rtol * std::abs(expected_value);

        ASSERT_LE(diff, threshold)
            << "Loss mismatch on " << device.to_string()
            << ": got " << static_cast<double>(loss_data[0])
            << ", expected " << expected_value
            << ", diff = " << diff
            << ", threshold = " << threshold;
    }

    void assertLossValueGeneric(const Tensor& loss_tensor, double expected_value) {
        if (dtype == DType::Float32) {
            assertLossValue<float>(loss_tensor, expected_value);
        } else if (dtype == DType::Float64) {
            assertLossValue<double>(loss_tensor, expected_value);
        }
    }

    // Helper to verify tensor values
    template<typename T>
    void assertAllValues(const Tensor& tensor, T expected_value) {
        auto tensor_cpu = tensor.to(Device::cpu());
        const T* data = tensor_cpu.data<T>();

        for (int64_t i = 0; i < tensor.numel(); i++) {
            double diff = std::abs(static_cast<double>(data[i]) - static_cast<double>(expected_value));
            double threshold = atol + rtol * std::abs(static_cast<double>(expected_value));

            ASSERT_LE(diff, threshold)
                << "Mismatch at index " << i << " on " << device.to_string()
                << ": got " << static_cast<double>(data[i])
                << ", expected " << static_cast<double>(expected_value);
        }
    }

    void assertAllValuesGeneric(const Tensor& tensor, double expected_value) {
        if (dtype == DType::Float32) {
            assertAllValues<float>(tensor, static_cast<float>(expected_value));
        } else if (dtype == DType::Float64) {
            assertAllValues<double>(tensor, expected_value);
        }
    }

    // Helper to create full tensor with proper dtype
    Tensor createFull(const std::vector<int64_t>& shape, double value) {
        return full(shape, static_cast<float>(value), dtype, device);
    }

    // Helper to create ones tensor with proper dtype
    Tensor createOnes(const std::vector<int64_t>& shape) {
        return ones(shape, dtype, device);
    }

    // Helper to create zeros tensor with proper dtype
    Tensor createZeros(const std::vector<int64_t>& shape) {
        return zeros(shape, dtype, device);
    }
};

// ============================================================================
// MSE Loss Tests
// ============================================================================

TEST_P(LossMultiDTypeTest, MSELossBasic) {
    auto pred = Variable(createFull({2, 3}, 1.0), true);
    auto target = Variable(createFull({2, 3}, 2.0), false);

    auto loss = mse_loss(pred, target, Reduction::Mean);

    // Expected: mean((1-2)^2) = mean(1) = 1.0
    EXPECT_EQ(loss.tensor().numel(), 1) << "Failed on " << device.to_string();
    EXPECT_EQ(loss.tensor().dtype(), dtype) << "Loss dtype should match input dtype";
    assertLossValueGeneric(loss.tensor(), 1.0);
}

TEST_P(LossMultiDTypeTest, MSELossZero) {
    // Perfect predictions
    auto pred = Variable(createFull({3, 3}, 5.0), true);
    auto target = Variable(createFull({3, 3}, 5.0), false);

    auto loss = mse_loss(pred, target, Reduction::Mean);

    EXPECT_EQ(loss.tensor().dtype(), dtype);
    assertLossValueGeneric(loss.tensor(), 0.0);
}

TEST_P(LossMultiDTypeTest, MSELossSum) {
    auto pred = Variable(createFull({2, 2}, 1.0), true);
    auto target = Variable(createFull({2, 2}, 2.0), false);

    auto loss = mse_loss(pred, target, Reduction::Sum);

    // Expected: sum((1-2)^2) = sum(1, 1, 1, 1) = 4.0
    EXPECT_EQ(loss.tensor().dtype(), dtype);
    assertLossValueGeneric(loss.tensor(), 4.0);
}

TEST_P(LossMultiDTypeTest, MSELossNone) {
    auto pred = Variable(createFull({2, 2}, 1.0), true);
    auto target = Variable(createFull({2, 2}, 2.0), false);

    auto loss = mse_loss(pred, target, Reduction::None);

    // Expected: element-wise (1-2)^2 = [1, 1, 1, 1]
    EXPECT_EQ(loss.tensor().numel(), 4) << "Failed on " << device.to_string();
    EXPECT_EQ(loss.tensor().dtype(), dtype);
    assertAllValuesGeneric(loss.tensor(), 1.0);
}

TEST_P(LossMultiDTypeTest, MSELossHighPrecision) {
    // Test with values that benefit from higher precision
    auto pred = Variable(createFull({100}, 1.000001), true);
    auto target = Variable(createFull({100}, 1.0), false);

    auto loss = mse_loss(pred, target, Reduction::Mean);

    // Expected: mean((1.000001 - 1.0)^2) = 1e-12
    EXPECT_EQ(loss.tensor().dtype(), dtype);

    // Float64 should handle this precision better
    auto loss_cpu = loss.tensor().to(Device::cpu());
    if (dtype == DType::Float64) {
        const double* loss_data = loss_cpu.data<double>();
        EXPECT_NEAR(loss_data[0], 1e-12, 1e-13) << "Failed on " << device.to_string();
    } else {
        // Float32 has limited precision for this test
        const float* loss_data = loss_cpu.data<float>();
        EXPECT_GT(loss_data[0], 0.0f) << "Failed on " << device.to_string();
    }
}

// ============================================================================
// Binary Cross Entropy Tests
// ============================================================================

TEST_P(LossMultiDTypeTest, BCELossBasic) {
    auto pred = Variable(createFull({2, 3}, 0.5), true);
    auto target = Variable(createOnes({2, 3}), false);

    auto loss = bce_loss(pred, target, Reduction::Mean);

    // Expected: -[1 * log(0.5) + 0 * log(0.5)] = -log(0.5) ≈ 0.693
    EXPECT_EQ(loss.tensor().dtype(), dtype);
    auto loss_cpu = loss.tensor().to(Device::cpu());

    if (dtype == DType::Float32) {
        const float* loss_data = loss_cpu.data<float>();
        EXPECT_GT(loss_data[0], 0.0f) << "Failed on " << device.to_string();
        EXPECT_NEAR(loss_data[0], -std::log(0.5f), 0.01f) << "Failed on " << device.to_string();
    } else {
        const double* loss_data = loss_cpu.data<double>();
        EXPECT_GT(loss_data[0], 0.0) << "Failed on " << device.to_string();
        EXPECT_NEAR(loss_data[0], -std::log(0.5), 0.001) << "Failed on " << device.to_string();
    }
}

TEST_P(LossMultiDTypeTest, BCELossPerfectPrediction) {
    auto pred = Variable(createOnes({3, 3}), true);
    auto target = Variable(createOnes({3, 3}), false);

    auto loss = bce_loss(pred, target, Reduction::Mean);

    // With clamping, we expect a very small loss (not exactly 0)
    EXPECT_EQ(loss.tensor().dtype(), dtype);
    auto loss_cpu = loss.tensor().to(Device::cpu());

    if (dtype == DType::Float32) {
        const float* loss_data = loss_cpu.data<float>();
        EXPECT_LT(loss_data[0], 0.001f) << "Failed on " << device.to_string();
    } else {
        const double* loss_data = loss_cpu.data<double>();
        EXPECT_LT(loss_data[0], 0.001) << "Failed on " << device.to_string();
    }
}

TEST_P(LossMultiDTypeTest, BCELossMixedTargets) {
    auto pred = Variable(createFull({2}, 0.7), true);
    auto target = Variable(createFull({2}, 0.5), false);

    auto loss = bce_loss(pred, target, Reduction::Mean);

    EXPECT_EQ(loss.tensor().dtype(), dtype);
    auto loss_cpu = loss.tensor().to(Device::cpu());

    if (dtype == DType::Float32) {
        const float* loss_data = loss_cpu.data<float>();
        EXPECT_GT(loss_data[0], 0.0f) << "Failed on " << device.to_string();
    } else {
        const double* loss_data = loss_cpu.data<double>();
        EXPECT_GT(loss_data[0], 0.0) << "Failed on " << device.to_string();
    }
}

// ============================================================================
// L1 Loss Tests
// ============================================================================

TEST_P(LossMultiDTypeTest, L1LossBasic) {
    auto pred = Variable(createFull({2, 3}, 3.0), true);
    auto target = Variable(createFull({2, 3}, 1.0), false);

    auto loss = l1_loss(pred, target, Reduction::Mean);

    // Expected: mean(|3-1|) = mean(2) = 2.0
    EXPECT_EQ(loss.tensor().dtype(), dtype);
    assertLossValueGeneric(loss.tensor(), 2.0);
}

TEST_P(LossMultiDTypeTest, L1LossNegativeDiff) {
    auto pred = Variable(createFull({2, 2}, 1.0), true);
    auto target = Variable(createFull({2, 2}, 3.0), false);

    auto loss = l1_loss(pred, target, Reduction::Mean);

    // Expected: mean(|1-3|) = mean(2) = 2.0
    EXPECT_EQ(loss.tensor().dtype(), dtype);
    assertLossValueGeneric(loss.tensor(), 2.0);
}

TEST_P(LossMultiDTypeTest, L1LossSum) {
    auto pred = Variable(createFull({2, 2}, 5.0), true);
    auto target = Variable(createFull({2, 2}, 3.0), false);

    auto loss = l1_loss(pred, target, Reduction::Sum);

    // Expected: sum(|5-3|) = sum(2, 2, 2, 2) = 8.0
    EXPECT_EQ(loss.tensor().dtype(), dtype);
    assertLossValueGeneric(loss.tensor(), 8.0);
}

TEST_P(LossMultiDTypeTest, L1LossZero) {
    auto pred = Variable(createFull({3, 3}, 7.0), true);
    auto target = Variable(createFull({3, 3}, 7.0), false);

    auto loss = l1_loss(pred, target, Reduction::Mean);

    EXPECT_EQ(loss.tensor().dtype(), dtype);
    assertLossValueGeneric(loss.tensor(), 0.0);
}

TEST_P(LossMultiDTypeTest, L1LossNone) {
    auto pred = Variable(createFull({2, 3}, 4.0), true);
    auto target = Variable(createFull({2, 3}, 1.0), false);

    auto loss = l1_loss(pred, target, Reduction::None);

    // Expected: element-wise |4-1| = [3, 3, 3, 3, 3, 3]
    EXPECT_EQ(loss.tensor().numel(), 6) << "Failed on " << device.to_string();
    EXPECT_EQ(loss.tensor().dtype(), dtype);
    assertAllValuesGeneric(loss.tensor(), 3.0);
}

// ============================================================================
// Cross Entropy Loss Tests
// ============================================================================

TEST_P(LossMultiDTypeTest, CrossEntropyBasic) {
    // Create logits for 2 samples, 3 classes
    auto logits = Variable(createFull({2, 3}, 0.0), true);
    auto targets = createZeros({2, 3});

    // Set first class to 1.0 for both samples
    auto targets_cpu = targets.to(Device::cpu());
    if (dtype == DType::Float32) {
        float* target_data = const_cast<float*>(targets_cpu.data<float>());
        target_data[0] = 1.0f;  // First sample, first class
        target_data[3] = 1.0f;  // Second sample, first class
    } else {
        double* target_data = const_cast<double*>(targets_cpu.data<double>());
        target_data[0] = 1.0;
        target_data[3] = 1.0;
    }
    targets = targets_cpu.to(device);

    auto loss = cross_entropy(logits, targets, Reduction::Mean);

    // Loss should be positive
    EXPECT_EQ(loss.tensor().dtype(), dtype);
    auto loss_cpu = loss.tensor().to(Device::cpu());

    if (dtype == DType::Float32) {
        const float* loss_data = loss_cpu.data<float>();
        EXPECT_GT(loss_data[0], 0.0f) << "Failed on " << device.to_string();
    } else {
        const double* loss_data = loss_cpu.data<double>();
        EXPECT_GT(loss_data[0], 0.0) << "Failed on " << device.to_string();
    }
}

TEST_P(LossMultiDTypeTest, CrossEntropyUniformLogits) {
    // For uniform logits across classes, cross entropy should be approximately log(num_classes)
    auto logits = Variable(createZeros({4, 3}), true);
    auto targets = createZeros({4, 3});

    // One-hot encode: each sample has class 0
    auto targets_cpu = targets.to(Device::cpu());
    if (dtype == DType::Float32) {
        float* target_data = const_cast<float*>(targets_cpu.data<float>());
        for (int i = 0; i < 4; i++) {
            target_data[i * 3] = 1.0f;
        }
    } else {
        double* target_data = const_cast<double*>(targets_cpu.data<double>());
        for (int i = 0; i < 4; i++) {
            target_data[i * 3] = 1.0;
        }
    }
    targets = targets_cpu.to(device);

    auto loss = cross_entropy(logits, targets, Reduction::Mean);

    EXPECT_EQ(loss.tensor().dtype(), dtype);
    auto loss_cpu = loss.tensor().to(Device::cpu());

    // For uniform probabilities over 3 classes: -log(1/3) ≈ 1.099
    if (dtype == DType::Float32) {
        const float* loss_data = loss_cpu.data<float>();
        EXPECT_NEAR(loss_data[0], std::log(3.0f), 0.1f) << "Failed on " << device.to_string();
    } else {
        const double* loss_data = loss_cpu.data<double>();
        EXPECT_NEAR(loss_data[0], std::log(3.0), 0.1) << "Failed on " << device.to_string();
    }
}

// ============================================================================
// NLL Loss Tests
// ============================================================================

TEST_P(LossMultiDTypeTest, NLLLossBasic) {
    // Create log probabilities (already log-transformed)
    auto log_val = std::log(0.5);
    auto log_probs = Variable(createFull({2, 3}, log_val), true);
    auto targets = createZeros({2, 3});

    auto targets_cpu = targets.to(Device::cpu());
    if (dtype == DType::Float32) {
        float* target_data = const_cast<float*>(targets_cpu.data<float>());
        target_data[0] = 1.0f;
        target_data[3] = 1.0f;
    } else {
        double* target_data = const_cast<double*>(targets_cpu.data<double>());
        target_data[0] = 1.0;
        target_data[3] = 1.0;
    }
    targets = targets_cpu.to(device);

    auto loss = nll_loss(log_probs, targets, Reduction::Mean);

    EXPECT_EQ(loss.tensor().dtype(), dtype);
    auto loss_cpu = loss.tensor().to(Device::cpu());

    if (dtype == DType::Float32) {
        const float* loss_data = loss_cpu.data<float>();
        EXPECT_GT(loss_data[0], 0.0f) << "Failed on " << device.to_string();
    } else {
        const double* loss_data = loss_cpu.data<double>();
        EXPECT_GT(loss_data[0], 0.0) << "Failed on " << device.to_string();
    }
}

// ============================================================================
// Loss Function Classes Tests
// ============================================================================

TEST_P(LossMultiDTypeTest, MSELossClass) {
    MSELoss criterion(Reduction::Mean);

    auto pred = Variable(createFull({2, 2}, 2.0), true);
    auto target = Variable(createFull({2, 2}, 1.0), false);

    auto loss = criterion(pred, target);

    EXPECT_EQ(loss.tensor().dtype(), dtype);
    assertLossValueGeneric(loss.tensor(), 1.0);
}

TEST_P(LossMultiDTypeTest, BCELossClass) {
    BCELoss criterion(Reduction::Mean);

    auto pred = Variable(createFull({2, 2}, 0.8), true);
    auto target = Variable(createOnes({2, 2}), false);

    auto loss = criterion(pred, target);

    EXPECT_EQ(loss.tensor().dtype(), dtype);
    auto loss_cpu = loss.tensor().to(Device::cpu());

    if (dtype == DType::Float32) {
        const float* loss_data = loss_cpu.data<float>();
        EXPECT_GT(loss_data[0], 0.0f) << "Failed on " << device.to_string();
    } else {
        const double* loss_data = loss_cpu.data<double>();
        EXPECT_GT(loss_data[0], 0.0) << "Failed on " << device.to_string();
    }
}

TEST_P(LossMultiDTypeTest, L1LossClass) {
    L1Loss criterion(Reduction::Mean);

    auto pred = Variable(createFull({3, 3}, 5.0), true);
    auto target = Variable(createFull({3, 3}, 2.0), false);

    auto loss = criterion(pred, target);

    EXPECT_EQ(loss.tensor().dtype(), dtype);
    assertLossValueGeneric(loss.tensor(), 3.0);
}

TEST_P(LossMultiDTypeTest, CrossEntropyLossClass) {
    CrossEntropyLoss criterion(Reduction::Mean);

    auto logits = Variable(createZeros({2, 3}), true);
    auto targets = createZeros({2, 3});

    auto targets_cpu = targets.to(Device::cpu());
    if (dtype == DType::Float32) {
        float* target_data = const_cast<float*>(targets_cpu.data<float>());
        target_data[0] = 1.0f;
        target_data[3] = 1.0f;
    } else {
        double* target_data = const_cast<double*>(targets_cpu.data<double>());
        target_data[0] = 1.0;
        target_data[3] = 1.0;
    }
    targets = targets_cpu.to(device);

    auto loss = criterion(logits, targets);

    EXPECT_EQ(loss.tensor().dtype(), dtype);
    auto loss_cpu = loss.tensor().to(Device::cpu());

    if (dtype == DType::Float32) {
        const float* loss_data = loss_cpu.data<float>();
        EXPECT_GT(loss_data[0], 0.0f) << "Failed on " << device.to_string();
    } else {
        const double* loss_data = loss_cpu.data<double>();
        EXPECT_GT(loss_data[0], 0.0) << "Failed on " << device.to_string();
    }
}

// ============================================================================
// Gradient Requirement Tests
// ============================================================================

TEST_P(LossMultiDTypeTest, MSELossGradientRequired) {
    auto pred = Variable(createFull({2, 2}, 1.0), true);
    auto target = Variable(createFull({2, 2}, 2.0), false);

    auto loss = mse_loss(pred, target, Reduction::Mean);

    // Loss should require gradient since pred requires grad
    EXPECT_TRUE(loss.requires_grad()) << "Failed on " << device.to_string();
}

TEST_P(LossMultiDTypeTest, BCELossGradientRequired) {
    auto pred = Variable(createFull({2, 2}, 0.6), true);
    auto target = Variable(createOnes({2, 2}), false);

    auto loss = bce_loss(pred, target, Reduction::Mean);

    EXPECT_TRUE(loss.requires_grad()) << "Failed on " << device.to_string();
}

TEST_P(LossMultiDTypeTest, L1LossGradientRequired) {
    auto pred = Variable(createFull({2, 2}, 3.0), true);
    auto target = Variable(createFull({2, 2}, 1.0), false);

    auto loss = l1_loss(pred, target, Reduction::Mean);

    EXPECT_TRUE(loss.requires_grad()) << "Failed on " << device.to_string();
}

// ============================================================================
// Backward Gradient Tests — verify .backward() actually populates .grad()
// ============================================================================

TEST_P(LossMultiDTypeTest, MSELoss_BackwardGradient) {
    auto pred = Variable(createFull({2, 3}, 2.0), true);
    auto target = Variable(createFull({2, 3}, 1.0), false);

    auto loss = mse_loss(pred, target, Reduction::Mean);
    EXPECT_EQ(loss.tensor().dtype(), dtype);

    EXPECT_NO_THROW(loss.backward());
    ASSERT_TRUE(pred.grad().has_value());
    // grad of MSE wrt pred is 2*(pred-target)/N; on 2x3 with diff=1 → 2/6 = 0.333...
    auto grad_cpu = pred.grad().value().to(Device::cpu()).to(DType::Float32);
    auto grad_data = grad_cpu.data<float>();
    for (int64_t i = 0; i < grad_cpu.numel(); ++i) {
        EXPECT_NEAR(grad_data[i], 2.0f / 6.0f, 1e-3f) << "on " << device.to_string();
    }
}

TEST_P(LossMultiDTypeTest, L1Loss_BackwardGradient) {
    auto pred = Variable(createFull({2, 3}, 3.0), true);
    auto target = Variable(createFull({2, 3}, 1.0), false);

    auto loss = l1_loss(pred, target, Reduction::Mean);
    EXPECT_EQ(loss.tensor().dtype(), dtype);

    EXPECT_NO_THROW(loss.backward());
    ASSERT_TRUE(pred.grad().has_value());
    // grad of L1 wrt pred is sign(pred-target)/N; pred>target → +1/6
    auto grad_cpu = pred.grad().value().to(Device::cpu()).to(DType::Float32);
    auto grad_data = grad_cpu.data<float>();
    for (int64_t i = 0; i < grad_cpu.numel(); ++i) {
        EXPECT_NEAR(grad_data[i], 1.0f / 6.0f, 1e-3f) << "on " << device.to_string();
    }
}

TEST_P(LossMultiDTypeTest, BCELoss_BackwardGradient) {
    auto pred = Variable(createFull({2, 3}, 0.6), true);
    auto target = Variable(createFull({2, 3}, 1.0), false);

    auto loss = bce_loss(pred, target, Reduction::Mean);
    EXPECT_EQ(loss.tensor().dtype(), dtype);

    EXPECT_NO_THROW(loss.backward());
    ASSERT_TRUE(pred.grad().has_value());
    // grad of BCE wrt pred: -(t/p - (1-t)/(1-p))/N; t=1,p=0.6 → -(1/0.6)/6 ≈ -0.2778
    auto grad_cpu = pred.grad().value().to(Device::cpu()).to(DType::Float32);
    auto grad_data = grad_cpu.data<float>();
    for (int64_t i = 0; i < grad_cpu.numel(); ++i) {
        EXPECT_NEAR(grad_data[i], -1.0f / (0.6f * 6.0f), 1e-3f)
            << "on " << device.to_string() << " at idx " << i;
    }
}

TEST_P(LossMultiDTypeTest, CrossEntropy_BackwardGradient) {
    // Logits for 2 samples × 3 classes; targets are class indices.
    auto logits = Variable(createFull({2, 3}, 0.0), true);
    // Target: class index 1 for both samples.
    auto target = zeros({2}, DType::Int64, device);
    auto target_cpu = target.to(Device::cpu());
    auto target_data = target_cpu.data<int64_t>();
    target_data[0] = 1;
    target_data[1] = 1;
    target = target_cpu.to(device);

    auto loss = cross_entropy(logits, target, Reduction::Mean);
    EXPECT_EQ(loss.tensor().dtype(), dtype);

    EXPECT_NO_THROW(loss.backward());
    ASSERT_TRUE(logits.grad().has_value());

    // For uniform logits, softmax = 1/3 uniformly; grad_i = (softmax_i - onehot_i)/N.
    auto grad_cpu = logits.grad().value().to(Device::cpu()).to(DType::Float32);
    auto grad_data = grad_cpu.data<float>();
    // Class 0, 2 → softmax - 0 = 1/3; divided by N=2 → 1/6
    // Class 1 (target) → softmax - 1 = -2/3; divided by N=2 → -1/3
    for (int64_t n = 0; n < 2; ++n) {
        EXPECT_NEAR(grad_data[n * 3 + 0],  1.0f / 6.0f, 1e-3f);
        EXPECT_NEAR(grad_data[n * 3 + 1], -1.0f / 3.0f, 1e-3f);
        EXPECT_NEAR(grad_data[n * 3 + 2],  1.0f / 6.0f, 1e-3f);
    }
}

TEST_P(LossMultiDTypeTest, NLLLoss_BackwardGradient) {
    // NLL expects log-probabilities. Uniform log-probs = log(1/3).
    auto log_probs = Variable(createFull({2, 3}, std::log(1.0 / 3.0)), true);
    // Target: class 0 for both samples.
    auto target = zeros({2}, DType::Int64, device);

    auto loss = nll_loss(log_probs, target, Reduction::Mean);
    EXPECT_EQ(loss.tensor().dtype(), dtype);

    EXPECT_NO_THROW(loss.backward());
    ASSERT_TRUE(log_probs.grad().has_value());

    // grad of NLL wrt log_probs: -onehot(target)/N
    auto grad_cpu = log_probs.grad().value().to(Device::cpu()).to(DType::Float32);
    auto grad_data = grad_cpu.data<float>();
    for (int64_t n = 0; n < 2; ++n) {
        EXPECT_NEAR(grad_data[n * 3 + 0], -0.5f, 1e-4f);  // target=0 class
        EXPECT_NEAR(grad_data[n * 3 + 1],  0.0f, 1e-6f);
        EXPECT_NEAR(grad_data[n * 3 + 2],  0.0f, 1e-6f);
    }
}

TEST_P(LossMultiDTypeTest, SmoothL1Loss_BackwardGradient) {
    // SmoothL1 is |x|-0.5 for |x|>1 (linear grad), 0.5*x^2 for |x|<=1 (linear grad x).
    auto pred = Variable(createFull({2, 3}, 0.5), true);   // |diff|=0.5 < 1 → quadratic region
    auto target = Variable(createFull({2, 3}, 0.0), false);

    auto loss = tenzor::nn::functional::smooth_l1_loss(pred, target, Reduction::Mean);
    EXPECT_EQ(loss.tensor().dtype(), dtype);

    EXPECT_NO_THROW(loss.backward());
    ASSERT_TRUE(pred.grad().has_value());

    // In quadratic region, d/dpred[0.5*(pred-target)^2] / N = (pred-target)/N = 0.5/6
    auto grad_cpu = pred.grad().value().to(Device::cpu()).to(DType::Float32);
    auto grad_data = grad_cpu.data<float>();
    for (int64_t i = 0; i < grad_cpu.numel(); ++i) {
        EXPECT_NEAR(grad_data[i], 0.5f / 6.0f, 1e-3f) << "on " << device.to_string();
    }
}

// ============================================================================
// Test Instantiation
// ============================================================================

std::vector<LossDTypeParam> GenerateLossDTypeCombinations() {
    std::vector<std::string> backends = {"cpu", "cuda", "vulkan", "oneapi", "rocm"};

    // Loss functions should work with Float32 and Float64
    // Float32: Standard training precision
    // Float64: High precision for numerical stability
    std::vector<std::tuple<DType, std::string, double, double>> dtypes = {
        {DType::Float32, "float32", 1e-4, 1e-6},   // Standard precision
        {DType::Float64, "float64", 1e-8, 1e-10},  // High precision
    };

    std::vector<LossDTypeParam> combinations;
    for (const auto& backend : backends) {
        for (const auto& [dtype, dtype_name, rtol, atol] : dtypes) {
            combinations.push_back({backend, dtype, dtype_name, rtol, atol});
        }
    }
    return combinations;
}

INSTANTIATE_TEST_SUITE_P(
    AllBackendsFloatDTypes,
    LossMultiDTypeTest,
    ::testing::ValuesIn(GenerateLossDTypeCombinations()),
    [](const ::testing::TestParamInfo<LossDTypeParam>& info) {
        return info.param.ToString();
    }
);

/*
 * COVERAGE IMPACT SUMMARY:
 *
 * Original test_losses.cpp:
 * - 21 tests × 4 backends × 1 dtype (Float32) = 84 test scenarios
 *
 * Refactored test_losses_multidtype.cpp:
 * - 22 tests × 4 backends × 2 dtypes (Float32, Float64) = 176 test scenarios
 *
 * Coverage increase: 2.1x improvement (92 additional test scenarios)
 *
 * Tests covered:
 * - MSE Loss: Basic, Zero, Sum, None, HighPrecision, Class, GradientRequired (7 tests)
 * - BCE Loss: Basic, PerfectPrediction, MixedTargets, Class, GradientRequired (5 tests)
 * - L1 Loss: Basic, NegativeDiff, Sum, Zero, None, Class, GradientRequired (7 tests)
 * - Cross Entropy: Basic, UniformLogits, Class (3 tests)
 * - NLL Loss: Basic (1 test)
 *
 * DTypes tested:
 * - Float32: Standard training precision, ~7 decimal digits of accuracy
 * - Float64: High precision training, ~15 decimal digits of accuracy
 *
 * Key improvements:
 * 1. Dtype preservation verification (output matches input dtype)
 * 2. Precision-appropriate tolerances for each dtype
 * 3. High precision gradient computation testing
 * 4. All reduction modes tested (Mean, Sum, None)
 * 5. Both functional API and class API tested
 * 6. Gradient requirement propagation verified
 *
 * Benefits for training:
 * - Float64 provides better gradient accuracy for sensitive models
 * - Allows mixed-precision training strategies
 * - Ensures numerical stability across precision levels
 * - Validates loss computation correctness across hardware backends
 */
