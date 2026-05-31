/**
 * @file test_nested_tensor.cpp
 * @brief Tests for NestedTensor construction, ops, device transfer, and padding.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nested/nested_tensor.hpp>
#include <tenzor/nested/nested_ops.hpp>
#include <tenzor/ops/creation.hpp>
#include "../backend_test_fixture.hpp"
#include <vector>
#include <cmath>
#include <numeric>

namespace {
// Helper to create an Int64 tensor from a vector on a given device
tenzor::Tensor make_int64_tensor(const std::vector<int64_t>& data,
                                 tenzor::Device device) {
    return tenzor::from_data(data.data(),
                             {static_cast<int64_t>(data.size())}, device);
}

class NestedTensorTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

// ============================================================================
// Construction
// ============================================================================

TEST_P(NestedTensorTest, FromTensorList) {
    auto t1 = tenzor::randn({3, 4}, tenzor::DType::Float32, device);
    auto t2 = tenzor::randn({5, 4}, tenzor::DType::Float32, device);
    auto t3 = tenzor::randn({2, 4}, tenzor::DType::Float32, device);

    auto nt = tenzor::NestedTensor::from_tensor_list({t1, t2, t3});

    EXPECT_EQ(nt.batch_size(), 3);
    EXPECT_EQ(nt.dtype(), tenzor::DType::Float32);
    EXPECT_TRUE(nt.device().type == device.type);

    // Total length should be 3 + 5 + 2 = 10
    EXPECT_EQ(nt.values().shape()[0], 10);
    EXPECT_EQ(nt.values().shape()[1], 4);

    // Offsets should be [0, 3, 8, 10]
    auto off_cpu = nt.offsets().cpu();
    EXPECT_EQ(off_cpu.numel(), 4);
    EXPECT_EQ(off_cpu.data<int64_t>()[0], 0);
    EXPECT_EQ(off_cpu.data<int64_t>()[1], 3);
    EXPECT_EQ(off_cpu.data<int64_t>()[2], 8);
    EXPECT_EQ(off_cpu.data<int64_t>()[3], 10);
}

TEST_P(NestedTensorTest, Unbind) {
    auto t1 = tenzor::randn({3, 4}, tenzor::DType::Float32, device);
    auto t2 = tenzor::randn({5, 4}, tenzor::DType::Float32, device);

    auto nt = tenzor::NestedTensor::from_tensor_list({t1, t2});
    auto unbound = nt.unbind();

    ASSERT_EQ(unbound.size(), 2u);
    EXPECT_EQ(unbound[0].shape()[0], 3);
    EXPECT_EQ(unbound[0].shape()[1], 4);
    EXPECT_EQ(unbound[1].shape()[0], 5);
    EXPECT_EQ(unbound[1].shape()[1], 4);

    // Values should match originals
    auto unbound0_cpu = unbound[0].cpu();
    auto t1_cpu = t1.cpu();
    for (int64_t i = 0; i < 3; ++i) {
        for (int64_t j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(unbound0_cpu.data<float>()[i * 4 + j],
                            t1_cpu.data<float>()[i * 4 + j]);
        }
    }
}

TEST_P(NestedTensorTest, Select) {
    auto t1 = tenzor::randn({3, 4}, tenzor::DType::Float32, device);
    auto t2 = tenzor::randn({5, 4}, tenzor::DType::Float32, device);

    auto nt = tenzor::NestedTensor::from_tensor_list({t1, t2});
    auto selected = nt.select(1);

    EXPECT_EQ(selected.shape()[0], 5);
    EXPECT_EQ(selected.shape()[1], 4);
}

// ============================================================================
// Padding roundtrip
// ============================================================================

TEST_P(NestedTensorTest, ToPaddedRoundtrip) {
    auto t1 = tenzor::ones({3, 4}, tenzor::DType::Float32, device);
    auto t2 = tenzor::ones({5, 4}, tenzor::DType::Float32, device);
    // Scale t2 so we can distinguish
    t2 = tenzor::mul(t2, tenzor::full({1}, 2.0f, tenzor::DType::Float32, device));

    auto nt = tenzor::NestedTensor::from_tensor_list({t1, t2});

    // Convert to padded
    auto padded = nt.to_padded_tensor(0.0);

    EXPECT_EQ(padded.shape()[0], 2);   // batch
    EXPECT_EQ(padded.shape()[1], 5);   // max_len
    EXPECT_EQ(padded.shape()[2], 4);   // D

    auto padded_cpu = padded.cpu();
    // First sequence: rows 0-2 should be 1.0, rows 3-4 should be 0.0 (padding)
    EXPECT_FLOAT_EQ(padded_cpu.data<float>()[0], 1.0f);
    EXPECT_FLOAT_EQ(padded_cpu.data<float>()[3 * 4], 0.0f);  // padding row

    // Second sequence: all rows should be 2.0
    EXPECT_FLOAT_EQ(padded_cpu.data<float>()[5 * 4], 2.0f);  // row 0 of seq 2
}

TEST_P(NestedTensorTest, FromPadded) {
    // Create a padded tensor [2, 5, 4] and lengths [3, 5]
    auto padded = tenzor::randn({2, 5, 4}, tenzor::DType::Float32, device);
    auto lengths = make_int64_tensor({3, 5}, device);

    auto nt = tenzor::NestedTensor::from_padded(padded, lengths);

    EXPECT_EQ(nt.batch_size(), 2);
    EXPECT_EQ(nt.values().shape()[0], 8);  // 3 + 5
    EXPECT_EQ(nt.values().shape()[1], 4);
}

// ============================================================================
// Element-wise ops
// ============================================================================

TEST_P(NestedTensorTest, ElementWiseAdd) {
    auto t1 = tenzor::ones({3, 4}, tenzor::DType::Float32, device);
    auto t2 = tenzor::ones({5, 4}, tenzor::DType::Float32, device);

    auto nt1 = tenzor::NestedTensor::from_tensor_list({t1, t2});
    auto nt2 = tenzor::NestedTensor::from_tensor_list({t1, t2});

    auto result = tenzor::nested_add(nt1, nt2);

    // All values should be 2.0
    auto vals = result.values().cpu();
    for (int64_t i = 0; i < vals.numel(); ++i) {
        EXPECT_FLOAT_EQ(vals.data<float>()[i], 2.0f);
    }
}

TEST_P(NestedTensorTest, ElementWiseMul) {
    auto t1 = tenzor::full({3, 4}, 3.0f, tenzor::DType::Float32, device);
    auto t2 = tenzor::full({5, 4}, 2.0f, tenzor::DType::Float32, device);

    auto nt1 = tenzor::NestedTensor::from_tensor_list({t1, t2});
    auto nt2 = tenzor::NestedTensor::from_tensor_list({t1, t2});

    auto result = tenzor::nested_mul(nt1, nt2);

    // First segment: 3*3 = 9, second: 2*2 = 4
    auto vals = result.values().cpu();
    EXPECT_FLOAT_EQ(vals.data<float>()[0], 9.0f);
    EXPECT_FLOAT_EQ(vals.data<float>()[3 * 4], 4.0f);
}

// ============================================================================
// Softmax
// ============================================================================

TEST_P(NestedTensorTest, NestedSoftmax) {
    auto t1 = tenzor::randn({3, 4}, tenzor::DType::Float32, device);
    auto t2 = tenzor::randn({5, 4}, tenzor::DType::Float32, device);

    auto nt = tenzor::NestedTensor::from_tensor_list({t1, t2});
    auto result = tenzor::nested_softmax(nt, /*dim=*/-1);

    // Output should be a NestedTensor with same structure
    EXPECT_EQ(result.batch_size(), 2);
    EXPECT_EQ(result.values().shape()[0], 8);  // 3 + 5
    EXPECT_EQ(result.values().shape()[1], 4);
}

// ============================================================================
// Reduction
// ============================================================================

TEST_P(NestedTensorTest, NestedSum) {
    auto t1 = tenzor::ones({3, 4}, tenzor::DType::Float32, device);
    auto t2 = tenzor::ones({5, 4}, tenzor::DType::Float32, device);

    auto nt = tenzor::NestedTensor::from_tensor_list({t1, t2});
    // Sum along ragged dim (dim=0 within each segment)
    auto result = tenzor::nested_sum(nt, /*dim=*/1, /*keepdim=*/false);

    // Result should be a NestedTensor (or dense depending on implementation)
    // Each batch element gets reduced to a single row
    auto vals = result.values();
    EXPECT_EQ(result.batch_size(), 2);
}

TEST_P(NestedTensorTest, NestedMean) {
    auto t1 = tenzor::full({3, 4}, 6.0f, tenzor::DType::Float32, device);
    auto t2 = tenzor::full({5, 4}, 10.0f, tenzor::DType::Float32, device);

    auto nt = tenzor::NestedTensor::from_tensor_list({t1, t2});
    auto result = tenzor::nested_mean(nt, /*dim=*/1, /*keepdim=*/false);

    EXPECT_EQ(result.batch_size(), 2);
}

// ============================================================================
// Clone
// ============================================================================

TEST_P(NestedTensorTest, Clone) {
    auto t1 = tenzor::randn({3, 4}, tenzor::DType::Float32, device);
    auto nt = tenzor::NestedTensor::from_tensor_list({t1});
    auto cloned = nt.clone();

    EXPECT_EQ(cloned.batch_size(), 1);
    EXPECT_EQ(cloned.values().shape()[0], 3);
    EXPECT_EQ(cloned.values().shape()[1], 4);

    // Should be a deep copy (different pointers)
    EXPECT_NE(cloned.values().data_ptr(), nt.values().data_ptr());
}

INSTANTIATE_BACKEND_TESTS(NestedTensorTest);

} // namespace
