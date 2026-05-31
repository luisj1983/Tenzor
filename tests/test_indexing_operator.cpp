#include <tenzor/core/tensor.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/tenzor.hpp>
#include <gtest/gtest.h>
#include <iostream>

#include "backend_test_fixture.hpp"

using namespace tenzor;

class TensorIndexingTest : public ::tenzor::testing::BackendTest {};

TEST_P(TensorIndexingTest, OneDimensionalIndexing) {
    // Create a 1D tensor [0, 1, 2, 3, 4]
    Tensor t = arange(0.0f, 5.0f, 1.0f, DType::Float32, device);

    // Test positive indexing
    Tensor elem0 = t[0];
    EXPECT_EQ(elem0.ndim(), 0) << "Indexing 1D tensor should return 0D scalar";
    auto elem0_cpu = elem0.cpu();
    EXPECT_FLOAT_EQ(elem0_cpu.item<float>(), 0.0f);

    Tensor elem2 = t[2];
    EXPECT_EQ(elem2.ndim(), 0);
    auto elem2_cpu = elem2.cpu();
    EXPECT_FLOAT_EQ(elem2_cpu.item<float>(), 2.0f);

    // Test negative indexing
    Tensor elem_last = t[-1];
    EXPECT_EQ(elem_last.ndim(), 0);
    auto elem_last_cpu = elem_last.cpu();
    EXPECT_FLOAT_EQ(elem_last_cpu.item<float>(), 4.0f);

    Tensor elem_neg2 = t[-2];
    EXPECT_EQ(elem_neg2.ndim(), 0);
    auto elem_neg2_cpu = elem_neg2.cpu();
    EXPECT_FLOAT_EQ(elem_neg2_cpu.item<float>(), 3.0f);
}

TEST_P(TensorIndexingTest, MultiDimensionalIndexing) {
    // Create a 3D tensor with shape [4, 3, 2]
    Tensor host({4, 3, 2}, DType::Float32, Device::cpu());
    // Fill with sequential values
    float* data = host.data<float>();
    for (int i = 0; i < 24; ++i) {
        data[i] = static_cast<float>(i);
    }
    Tensor t = host.to(device);

    // Test indexing returns a slice along first dimension
    Tensor slice0 = t[0];
    EXPECT_EQ(slice0.ndim(), 2) << "Indexing 3D tensor should return 2D slice";
    EXPECT_EQ(slice0.shape()[0], 3);
    EXPECT_EQ(slice0.shape()[1], 2);

    Tensor slice1 = t[1];
    EXPECT_EQ(slice1.ndim(), 2);
    EXPECT_EQ(slice1.shape()[0], 3);
    EXPECT_EQ(slice1.shape()[1], 2);

    // Test negative indexing
    Tensor slice_last = t[-1];
    EXPECT_EQ(slice_last.ndim(), 2);
    EXPECT_EQ(slice_last.shape()[0], 3);
    EXPECT_EQ(slice_last.shape()[1], 2);

    // Verify it's a view, not a copy (shares storage)
    EXPECT_EQ(t.data_ptr(), slice0.data_ptr()) << "Indexing should return a view";
}

TEST_P(TensorIndexingTest, TwoDimensionalIndexing) {
    // Create a 2D tensor [3, 4]
    Tensor host({3, 4}, DType::Float32, Device::cpu());
    // Fill with sequential values [0, 1, 2, ..., 11]
    float* data = host.data<float>();
    for (int i = 0; i < 12; ++i) {
        data[i] = static_cast<float>(i);
    }
    Tensor t = host.to(device);

    // Indexing should return 1D tensor (row)
    Tensor row0 = t[0];
    EXPECT_EQ(row0.ndim(), 1);
    EXPECT_EQ(row0.shape()[0], 4);
    auto row0_0_cpu = row0[0].cpu();
    EXPECT_FLOAT_EQ(row0_0_cpu.item<float>(), 0.0f);
    auto row0_1_cpu = row0[1].cpu();
    EXPECT_FLOAT_EQ(row0_1_cpu.item<float>(), 1.0f);

    Tensor row1 = t[1];
    EXPECT_EQ(row1.ndim(), 1);
    auto row1_0_cpu = row1[0].cpu();
    EXPECT_FLOAT_EQ(row1_0_cpu.item<float>(), 4.0f);

    Tensor row_last = t[-1];
    EXPECT_EQ(row_last.ndim(), 1);
    auto row_last_0_cpu = row_last[0].cpu();
    EXPECT_FLOAT_EQ(row_last_0_cpu.item<float>(), 8.0f);
}

TEST_P(TensorIndexingTest, OutOfBoundsPositive) {
    Tensor t = arange(0.0f, 5.0f, 1.0f, DType::Float32, device);

    EXPECT_THROW({
        Tensor elem = t[5];
    }, std::runtime_error);

    EXPECT_THROW({
        Tensor elem = t[100];
    }, std::runtime_error);
}

TEST_P(TensorIndexingTest, OutOfBoundsNegative) {
    Tensor t = arange(0.0f, 5.0f, 1.0f, DType::Float32, device);

    EXPECT_THROW({
        Tensor elem = t[-6];
    }, std::runtime_error);

    EXPECT_THROW({
        Tensor elem = t[-100];
    }, std::runtime_error);
}

TEST_P(TensorIndexingTest, NullTensorError) {
    Tensor t;  // Null tensor

    EXPECT_THROW({
        Tensor elem = t[0];
    }, std::runtime_error);
}

TEST_P(TensorIndexingTest, ScalarTensorError) {
    Tensor t({}, DType::Float32, device);  // 0D scalar
    t.fill_(5.0f);

    EXPECT_THROW({
        Tensor elem = t[0];
    }, std::runtime_error);
}

INSTANTIATE_BACKEND_TESTS(TensorIndexingTest);
