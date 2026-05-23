/**
 * @file test_split_operation_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for split and chunk operations
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/transform.hpp>
#include <tenzor/ops/creation.hpp>
#include "multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class SplitOpMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(SplitOpMultiDTypeTest, SplitEven) {
    auto t_cpu = tenzor::randn({6, 4}, DType::Float32, Device::cpu());
    auto t = t_cpu.to(dtype_).to(device_);
    auto parts = tenzor::split(t, 2, 0);  // split into size-2 chunks
    EXPECT_EQ(parts.size(), 3);
    for (auto& p : parts) {
        expectShape(p, {2, 4});
        expectDevice(p);
        expectDType(p);
    }
    // Concatenating the parts back along dim 0 must equal the original.
    std::vector<Tensor> parts_vec(parts.begin(), parts.end());
    auto rejoined = tenzor::cat(std::span<const Tensor>(parts_vec.data(),
                                                         parts_vec.size()), 0);
    expectTensorNear(rejoined, t, std::max(atol_, 1e-3f));
    // Per-part value check against CPU reference.
    auto parts_cpu = tenzor::split(t_cpu, 2, 0);
    for (size_t i = 0; i < parts.size(); ++i) {
        expectTensorNear(parts[i], parts_cpu[i], std::max(atol_, 1e-3f));
    }
}

TEST_P(SplitOpMultiDTypeTest, SplitDim1) {
    auto t_cpu = tenzor::randn({3, 8}, DType::Float32, Device::cpu());
    auto t = t_cpu.to(dtype_).to(device_);
    auto parts = tenzor::split(t, 4, 1);
    EXPECT_EQ(parts.size(), 2);
    expectShape(parts[0], {3, 4});
    expectShape(parts[1], {3, 4});
    auto parts_cpu = tenzor::split(t_cpu, 4, 1);
    expectTensorNear(parts[0], parts_cpu[0], std::max(atol_, 1e-3f));
    expectTensorNear(parts[1], parts_cpu[1], std::max(atol_, 1e-3f));
}

TEST_P(SplitOpMultiDTypeTest, ChunkEven) {
    auto t_cpu = tenzor::randn({12, 4}, DType::Float32, Device::cpu());
    auto t = t_cpu.to(dtype_).to(device_);
    auto parts = tenzor::chunk(t, 3, 0);
    EXPECT_EQ(parts.size(), 3);
    for (auto& p : parts) {
        expectShape(p, {4, 4});
    }
    auto parts_cpu = tenzor::chunk(t_cpu, 3, 0);
    for (size_t i = 0; i < parts.size(); ++i) {
        expectTensorNear(parts[i], parts_cpu[i], std::max(atol_, 1e-3f));
    }
}

TEST_P(SplitOpMultiDTypeTest, ChunkUneven) {
    // 10 elements split into 3 chunks: 4, 4, 2
    auto t_cpu = tenzor::randn({10, 3}, DType::Float32, Device::cpu());
    auto t = t_cpu.to(dtype_).to(device_);
    auto parts = tenzor::chunk(t, 3, 0);
    EXPECT_EQ(parts.size(), 3);
    expectShape(parts[0], {4, 3});
    expectShape(parts[1], {4, 3});
    expectShape(parts[2], {2, 3});
    auto parts_cpu = tenzor::chunk(t_cpu, 3, 0);
    for (size_t i = 0; i < parts.size(); ++i) {
        expectTensorNear(parts[i], parts_cpu[i], std::max(atol_, 1e-3f));
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(SplitOpMultiDTypeTest);
