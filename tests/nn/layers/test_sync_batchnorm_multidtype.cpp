/**
 * @file test_sync_batchnorm_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for SyncBatchNorm layer
 */

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/sync_batchnorm.hpp>
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

class SyncBatchNormMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    SyncBatchNorm createSyncBN(int64_t num_features) {
        AllReduceFn noop_allreduce = [](Tensor&) {};
        // intentionally exercising deprecated legacy SyncBatchNorm ctor
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        return SyncBatchNorm(num_features, noop_allreduce, /*world_size=*/1);
#pragma GCC diagnostic pop
    }
};

TEST_P(SyncBatchNormMultiDTypeTest, BasicForward) {
    auto sbn = createSyncBN(8);
    convert_model(sbn);
    sbn.train();

    auto input = createInput({4, 8, 4, 4}, false);
    auto output = sbn.forward(input);

    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 8);
    EXPECT_EQ(output.shape()[2], 4);
    EXPECT_EQ(output.shape()[3], 4);

    auto output_f32 = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto* data = output_f32.data<float>();
    for (int64_t i = 0; i < output_f32.numel(); ++i) {
        EXPECT_FALSE(std::isnan(data[i]));
        EXPECT_FALSE(std::isinf(data[i]));
    }
}

TEST_P(SyncBatchNormMultiDTypeTest, RunningStats) {
    auto sbn = createSyncBN(8);
    convert_model(sbn);
    sbn.train();

    auto input = createInput({4, 8, 4, 4}, false);
    sbn.forward(input);

    sbn.eval();
    auto input2 = createInput({2, 8, 4, 4}, false);
    EXPECT_NO_THROW({
        auto output = sbn.forward(input2);
        EXPECT_EQ(output.shape()[0], 2);
        EXPECT_EQ(output.shape()[1], 8);
    });
}

TEST_P(SyncBatchNormMultiDTypeTest, EvalModeDeterministic) {
    auto sbn = createSyncBN(8);
    convert_model(sbn);

    sbn.train();
    auto train_input = createInput({4, 8, 4, 4}, false);
    sbn.forward(train_input);

    sbn.eval();
    auto test_t = createRandn({4, 8, 4, 4});
    auto out1 = sbn.forward(Variable(test_t, false));
    auto out2 = sbn.forward(Variable(test_t, false));

    auto o1 = out1.tensor().to(Device::cpu()).to(DType::Float32);
    auto o2 = out2.tensor().to(Device::cpu()).to(DType::Float32);
    auto* d1 = o1.data<float>();
    auto* d2 = o2.data<float>();

    for (int64_t i = 0; i < o1.numel(); ++i) {
        EXPECT_NEAR(d1[i], d2[i], atol())
            << "Eval mode outputs should be deterministic at index " << i;
    }
}

TEST_P(SyncBatchNormMultiDTypeTest, GradientFlow) {
    auto sbn = createSyncBN(8);
    convert_model(sbn);
    sbn.train();

    auto input = createInput({4, 8, 4, 4}, true);
    auto output = sbn.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = createRandn(shape_vec);
    output.backward(grad_output);

    EXPECT_TRUE(input.has_grad());

    auto grad_f32 = input.grad()->to(Device::cpu()).to(DType::Float32);
    auto* grad_data = grad_f32.data<float>();

    bool has_nonzero = false;
    for (int64_t i = 0; i < grad_f32.numel(); ++i) {
        EXPECT_FALSE(std::isnan(grad_data[i]));
        if (std::abs(grad_data[i]) > 1e-7f) has_nonzero = true;
    }
    EXPECT_TRUE(has_nonzero) << "Gradients should not all be zero";
}

TEST_P(SyncBatchNormMultiDTypeTest, ParametersNonEmpty) {
    auto sbn = createSyncBN(8);
    convert_model(sbn);

    auto params = sbn.parameters();
    EXPECT_FALSE(params.empty());
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(SyncBatchNormMultiDTypeTest);
