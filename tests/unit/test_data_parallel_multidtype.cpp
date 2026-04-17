/**
 * @file test_data_parallel_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for DataParallel multi-device training
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

/**
 * @brief Simple test module that doubles its input
 */
class DoubleModuleMD : public Module {
public:
    DoubleModuleMD() = default;

    auto forward_impl(const Variable& input) -> Variable override {
        return Variable(input.tensor() * 2.0f);
    }

    auto parameters() -> std::vector<std::shared_ptr<Variable>> override {
        return {};
    }

    auto named_parameters() -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>> override {
        return {};
    }
};

class DataParallelMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();

        // DataParallel requires CUDA (or similar multi-device backend)
        if (device().type == Device::Type::CPU) {
            GTEST_SKIP() << "DataParallel requires a GPU backend";
        }

        // Check if at least 1 device is available for this backend
        try {
            auto t = zeros({2, 2}, dtype(), device());
        } catch (...) {
            GTEST_SKIP() << "Cannot allocate on " << backend_name();
        }
    }
};

TEST_P(DataParallelMultiDTypeTest, ConstructionWithValidDevice) {
    auto module = std::make_shared<DoubleModuleMD>();

    EXPECT_NO_THROW({
        DataParallel dp(module, {static_cast<int>(device().index)},
                        static_cast<int>(device().index));
    });
}

TEST_P(DataParallelMultiDTypeTest, ConstructionWithNullModule) {
    EXPECT_THROW({
        DataParallel dp(nullptr, {static_cast<int>(device().index)},
                        static_cast<int>(device().index));
    }, std::invalid_argument);
}

TEST_P(DataParallelMultiDTypeTest, SingleDeviceForwardPass) {
    auto module = std::make_shared<DoubleModuleMD>();
    DataParallel dp(module, {static_cast<int>(device().index)},
                    static_cast<int>(device().index));

    auto input_tensor = ones({4, 10}, DType::Float32, device()).to(dtype());
    Variable input(input_tensor);

    EXPECT_NO_THROW({
        auto output = dp.forward(input);
        auto out_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
        auto* data = out_cpu.data<float>();
        for (int i = 0; i < 40; ++i) {
            EXPECT_NEAR(data[i], 2.0f, atol());
        }
    });
}

TEST_P(DataParallelMultiDTypeTest, DeviceIDsAccessor) {
    auto module = std::make_shared<DoubleModuleMD>();
    int dev_idx = static_cast<int>(device().index);
    std::vector<int> device_ids = {dev_idx};
    DataParallel dp(module, device_ids, dev_idx);

    auto returned_ids = dp.device_ids();
    EXPECT_EQ(returned_ids, device_ids);
}

TEST_P(DataParallelMultiDTypeTest, TrainAndEvalMode) {
    auto module = std::make_shared<DoubleModuleMD>();
    DataParallel dp(module, {static_cast<int>(device().index)},
                    static_cast<int>(device().index));

    EXPECT_NO_THROW({ dp.train(true); });
    EXPECT_NO_THROW({ dp.eval(); });
}

TEST_P(DataParallelMultiDTypeTest, MakeDataParallelHelper) {
    auto module = std::make_shared<DoubleModuleMD>();

    EXPECT_NO_THROW({
        auto parallel_model = make_data_parallel(
            module, {static_cast<int>(device().index)},
            static_cast<int>(device().index));
        EXPECT_NE(parallel_model, nullptr);
    });
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(DataParallelMultiDTypeTest);
