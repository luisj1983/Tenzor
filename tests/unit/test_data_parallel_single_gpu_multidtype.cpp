/**
 * @file test_data_parallel_single_gpu_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for single-GPU DataParallel
 *
 * Tests DataParallel wrapping, forward pass, output shape, gradient flow,
 * and mode switching across backends and dtypes.
 * CPU is skipped since DataParallel requires a GPU backend.
 */

#include "../multi_backend_dtype_fixture.hpp"
#include "tenzor/nn/parallel/data_parallel.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/module.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/ops/creation.hpp"

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

// ============================================================================
// Helper Module
// ============================================================================

class ScaleModuleMD : public Module {
public:
    explicit ScaleModuleMD(float scale = 2.0f) : scale_(scale) {}

    auto forward_impl(const Variable& input) -> Variable override {
        return Variable(input.tensor() * scale_, input.requires_grad());
    }

    auto parameters() -> std::vector<std::shared_ptr<Variable>> override {
        return {};
    }

    auto named_parameters() -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>> override {
        return {};
    }

private:
    float scale_;
};

// ============================================================================
// Fixture
// ============================================================================

class DataParallelSingleGPUMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();
        if (IsSkipped()) return;

        // DataParallel is wired to CUDA-specific runtime calls
        // (cudaGetDeviceCount, cudaSetDevice, .cuda()). Other GPU backends
        // currently throw "DataParallel: CUDA support not enabled". Until
        // DataParallel grows a generic backend path, restrict to CUDA.
        if (device().type != Device::Type::CUDA) {
            GTEST_SKIP() << "DataParallel currently requires the CUDA backend";
        }

        // Verify device is usable
        try {
            auto t = zeros({2, 2}, dtype(), device());
        } catch (...) {
            GTEST_SKIP() << "Cannot allocate on " << backend_name();
        }
    }

    int dev_idx() const { return static_cast<int>(device().index); }
};;

// ============================================================================
// Tests
// ============================================================================

/// Wrapping a module in DataParallel should succeed.
TEST_P(DataParallelSingleGPUMultiDTypeTest, ModelWrapping) {
    auto module = std::make_shared<ScaleModuleMD>(3.0f);
    EXPECT_NO_THROW({
        DataParallel dp(module, {dev_idx()}, dev_idx());
    });
}

/// Forward pass produces correct scaled values.
TEST_P(DataParallelSingleGPUMultiDTypeTest, ForwardPassValues) {
    auto module = std::make_shared<ScaleModuleMD>(3.0f);
    DataParallel dp(module, {dev_idx()}, dev_idx());

    auto input_tensor = ones({4, 8}, DType::Float32, device()).to(dtype());
    auto output = dp.forward(Variable(input_tensor));

    auto out_f32 = output.tensor().to(DType::Float32).to(Device::cpu());
    auto* data = out_f32.data<float>();
    for (int i = 0; i < 32; ++i) {
        EXPECT_NEAR(data[i], 3.0f, atol()) << "Mismatch at index " << i;
    }
}

/// Output shape matches input shape for identity-like modules.
TEST_P(DataParallelSingleGPUMultiDTypeTest, OutputShapePreserved) {
    auto module = std::make_shared<ScaleModuleMD>(1.0f);
    DataParallel dp(module, {dev_idx()}, dev_idx());

    auto input_tensor = ones({8, 16}, DType::Float32, device()).to(dtype());
    auto output = dp.forward(Variable(input_tensor));

    auto out_shape = output.tensor().shape();
    EXPECT_EQ(out_shape[0], 8);
    EXPECT_EQ(out_shape[1], 16);
}

/// Output shape correct for 4D input (batch, channels, H, W).
TEST_P(DataParallelSingleGPUMultiDTypeTest, OutputShape4D) {
    auto module = std::make_shared<ScaleModuleMD>();
    DataParallel dp(module, {dev_idx()}, dev_idx());

    auto input_tensor = ones({2, 3, 8, 8}, DType::Float32, device()).to(dtype());
    auto output = dp.forward(Variable(input_tensor));

    auto out_shape = output.tensor().shape();
    EXPECT_EQ(out_shape[0], 2);
    EXPECT_EQ(out_shape[1], 3);
    EXPECT_EQ(out_shape[2], 8);
    EXPECT_EQ(out_shape[3], 8);
}

/// Output of forward with requires_grad=true should require grad.
TEST_P(DataParallelSingleGPUMultiDTypeTest, GradientFlowForward) {
    auto module = std::make_shared<ScaleModuleMD>(2.0f);
    DataParallel dp(module, {dev_idx()}, dev_idx());

    auto input_tensor = ones({2, 4}, DType::Float32, device()).to(dtype());
    auto output = dp.forward(Variable(input_tensor, true));

    EXPECT_TRUE(output.requires_grad()) << "Output should require gradients";
}

/// Train/eval mode transitions propagate to wrapped module.
TEST_P(DataParallelSingleGPUMultiDTypeTest, TrainEvalModeSync) {
    auto module = std::make_shared<ScaleModuleMD>();
    DataParallel dp(module, {dev_idx()}, dev_idx());

    EXPECT_TRUE(module->is_training());
    dp.eval();
    EXPECT_FALSE(module->is_training());
    dp.train();
    EXPECT_TRUE(module->is_training());
}

/// Device IDs accessor returns what was passed to constructor.
TEST_P(DataParallelSingleGPUMultiDTypeTest, DeviceIDsAccessor) {
    auto module = std::make_shared<ScaleModuleMD>();
    std::vector<int> device_ids = {dev_idx()};
    DataParallel dp(module, device_ids, dev_idx());

    EXPECT_EQ(dp.device_ids(), device_ids);
    EXPECT_EQ(dp.output_device(), dev_idx());
}

/// Null module should throw.
TEST_P(DataParallelSingleGPUMultiDTypeTest, NullModuleThrows) {
    EXPECT_THROW(
        DataParallel(nullptr, {dev_idx()}, dev_idx()),
        std::invalid_argument);
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(DataParallelSingleGPUMultiDTypeTest);
