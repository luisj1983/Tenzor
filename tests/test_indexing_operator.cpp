#include <tenzor/core/tensor.hpp>
#include <tenzor/ops/creation.hpp>
#include <gtest/gtest.h>
#include <iostream>

using namespace tenzor;

TEST(TensorIndexingTest, OneDimensionalIndexing) {
    // Create a 1D tensor [0, 1, 2, 3, 4]
    Tensor t = arange(0.0f, 5.0f, 1.0f, DType::Float32, Device::cpu());

    // Test positive indexing
    Tensor elem0 = t[0];
    EXPECT_EQ(elem0.ndim(), 0) << "Indexing 1D tensor should return 0D scalar";
    EXPECT_FLOAT_EQ(elem0.item<float>(), 0.0f);

    Tensor elem2 = t[2];
    EXPECT_EQ(elem2.ndim(), 0);
    EXPECT_FLOAT_EQ(elem2.item<float>(), 2.0f);

    // Test negative indexing
    Tensor elem_last = t[-1];
    EXPECT_EQ(elem_last.ndim(), 0);
    EXPECT_FLOAT_EQ(elem_last.item<float>(), 4.0f);

    Tensor elem_neg2 = t[-2];
    EXPECT_EQ(elem_neg2.ndim(), 0);
    EXPECT_FLOAT_EQ(elem_neg2.item<float>(), 3.0f);
}

TEST(TensorIndexingTest, MultiDimensionalIndexing) {
    // Create a 3D tensor with shape [4, 3, 2]
    Tensor t({4, 3, 2}, DType::Float32, Device::cpu());
    // Fill with sequential values
    float* data = t.data<float>();
    for (int i = 0; i < 24; ++i) {
        data[i] = static_cast<float>(i);
    }

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

TEST(TensorIndexingTest, TwoDimensionalIndexing) {
    // Create a 2D tensor [3, 4]
    Tensor t({3, 4}, DType::Float32, Device::cpu());
    // Fill with sequential values [0, 1, 2, ..., 11]
    float* data = t.data<float>();
    for (int i = 0; i < 12; ++i) {
        data[i] = static_cast<float>(i);
    }

    // Indexing should return 1D tensor (row)
    Tensor row0 = t[0];
    EXPECT_EQ(row0.ndim(), 1);
    EXPECT_EQ(row0.shape()[0], 4);
    EXPECT_FLOAT_EQ(row0[0].item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(row0[1].item<float>(), 1.0f);

    Tensor row1 = t[1];
    EXPECT_EQ(row1.ndim(), 1);
    EXPECT_FLOAT_EQ(row1[0].item<float>(), 4.0f);

    Tensor row_last = t[-1];
    EXPECT_EQ(row_last.ndim(), 1);
    EXPECT_FLOAT_EQ(row_last[0].item<float>(), 8.0f);
}

TEST(TensorIndexingTest, OutOfBoundsPositive) {
    Tensor t = arange(0.0f, 5.0f, 1.0f, DType::Float32, Device::cpu());

    EXPECT_THROW({
        Tensor elem = t[5];
    }, std::out_of_range);

    EXPECT_THROW({
        Tensor elem = t[100];
    }, std::out_of_range);
}

TEST(TensorIndexingTest, OutOfBoundsNegative) {
    Tensor t = arange(0.0f, 5.0f, 1.0f, DType::Float32, Device::cpu());

    EXPECT_THROW({
        Tensor elem = t[-6];
    }, std::out_of_range);

    EXPECT_THROW({
        Tensor elem = t[-100];
    }, std::out_of_range);
}

TEST(TensorIndexingTest, NullTensorError) {
    Tensor t;  // Null tensor

    EXPECT_THROW({
        Tensor elem = t[0];
    }, std::runtime_error);
}

TEST(TensorIndexingTest, ScalarTensorError) {
    Tensor t({}, DType::Float32, Device::cpu());  // 0D scalar
    t.fill_(5.0f);

    EXPECT_THROW({
        Tensor elem = t[0];
    }, std::runtime_error);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
