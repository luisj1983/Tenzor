/**
 * @file test_gradient_utils_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for gradient utility functions
 */

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/optim/gradient_utils.hpp"
#include "gradient_utils_test_support.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/autograd/variable.hpp"

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::optim;

class GradientUtilsMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(GradientUtilsMultiDTypeTest, FlattenSingleTensor) {
    // Gradient utils operate on CPU tensors
    Tensor t({10, 20}, DType::Float32, Device::cpu());
    t.fill_(1.5f);

    auto flat = flatten_tensors({t});
    EXPECT_EQ(flat.ndim(), 1);
    EXPECT_EQ(flat.numel(), 200);

    auto* data = flat.data<float>();
    for (int i = 0; i < 200; ++i) {
        EXPECT_FLOAT_EQ(data[i], 1.5f);
    }
}

TEST_P(GradientUtilsMultiDTypeTest, FlattenMultipleTensors) {
    Tensor t1({10, 20}, DType::Float32, Device::cpu());
    Tensor t2({5, 5}, DType::Float32, Device::cpu());
    t1.fill_(1.0f);
    t2.fill_(2.0f);

    auto flat = flatten_tensors({t1, t2});
    EXPECT_EQ(flat.numel(), 225);
}

TEST_P(GradientUtilsMultiDTypeTest, FlattenDifferentDtypesFails) {
    Tensor t1({10}, DType::Float32, Device::cpu());
    Tensor t2({10}, DType::Float64, Device::cpu());
    EXPECT_THROW(flatten_tensors({t1, t2}), std::invalid_argument);
}

TEST_P(GradientUtilsMultiDTypeTest, UnflattenIntoShapes) {
    Tensor flat({225}, DType::Float32, Device::cpu());
    auto* data = flat.data<float>();
    for (int i = 0; i < 225; ++i) data[i] = static_cast<float>(i);

    std::vector<std::vector<int64_t>> shapes = {{10, 20}, {5, 5}};
    auto tensors = unflatten_into(flat, shapes);

    ASSERT_EQ(tensors.size(), 2);
    EXPECT_EQ(tensors[0].shape()[0], 10);
    EXPECT_EQ(tensors[0].shape()[1], 20);
    EXPECT_EQ(tensors[1].shape()[0], 5);
    EXPECT_EQ(tensors[1].shape()[1], 5);
}

TEST_P(GradientUtilsMultiDTypeTest, FlattenUnflattenRoundtrip) {
    Tensor t1({10, 20}, DType::Float32, Device::cpu());
    Tensor t2({5, 5}, DType::Float32, Device::cpu());
    auto* d1 = t1.data<float>();
    for (int i = 0; i < 200; ++i) d1[i] = static_cast<float>(i);
    auto* d2 = t2.data<float>();
    for (int i = 0; i < 25; ++i) d2[i] = static_cast<float>(200 + i);

    auto flat = flatten_tensors({t1, t2});
    std::vector<std::vector<int64_t>> shapes = {{10, 20}, {5, 5}};
    auto tensors = unflatten_into(flat, shapes);

    ASSERT_EQ(tensors.size(), 2);
    auto* r1 = tensors[0].data<float>();
    for (int i = 0; i < 200; ++i) {
        EXPECT_FLOAT_EQ(r1[i], d1[i]);
    }
}

TEST_P(GradientUtilsMultiDTypeTest, ComputeBucketsSingleTensor) {
    Tensor t({1024 * 1024}, DType::Float32, Device::cpu());
    BucketConfig config;
    config.max_bucket_size_mb = 25;
    auto buckets = compute_bucket_sizes({t}, config);
    ASSERT_EQ(buckets.size(), 1);
    EXPECT_EQ(buckets[0].num_elements, 1024 * 1024);
}

TEST_P(GradientUtilsMultiDTypeTest, HelperFunctions) {
    Tensor t1({10}, DType::Float32, Device::cpu());
    Tensor t2({10}, DType::Float32, Device::cpu());
    EXPECT_TRUE(same_dtype({t1, t2}));
    EXPECT_TRUE(same_device({t1, t2}));
    EXPECT_EQ(total_numel({t1, t2}), 20u);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(GradientUtilsMultiDTypeTest);
