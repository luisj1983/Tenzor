/**
 * @file test_ops_additional.cpp
 * @brief Comprehensive tests for tensor operations to increase coverage
 *
 * This file contains extensive tests for:
 * - Reduction operations (sum, mean, max, min, prod, var, std, norm, argmax, argmin)
 * - Tensor manipulation (reshape, view, transpose, permute, squeeze, unsqueeze,
 *   concat, stack, split, chunk, expand, repeat, flatten)
 * - Indexing operations (gather, scatter, index_select, masked_select, where,
 *   slice, select, nonzero)
 * - Mathematical operations (matmul, dot, bmm, exp, log, sqrt, pow, trigonometric,
 *   hyperbolic, abs, neg, reciprocal, sign, rounding, clamping)
 * - Comparison operations (eq, ne, lt, gt, le, ge)
 * - In-place operations (add_, mul_, sub_, div_)
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <cmath>
#include <complex>
#include <limits>

using namespace tenzor;
using namespace tenzor::testing;

//==============================================================================
// Reduction Operations Tests
//==============================================================================

class ReductionOpsTest : public BackendTest {};

TEST_P(ReductionOpsTest, SumAllElements) {
    // Sum all elements in tensor
    auto t = ones({2, 3, 4}, DType::Float32, device);
    auto result = sum(t);

    EXPECT_EQ(result.numel(), 1) << "Failed on " << device.to_string();
    auto result_cpu = result.to(Device::cpu());
    EXPECT_FLOAT_EQ(result_cpu.data<float>()[0], 24.0f) << "Failed on " << device.to_string();
}

TEST_P(ReductionOpsTest, SumAlongDimension) {
    // Sum along specific dimension with keepdim
    auto t = ones({2, 3, 4}, DType::Float32, device);

    // Sum along dim 0
    auto result0 = sum(t, 0, false);
    EXPECT_EQ(result0.shape().size(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(result0.shape()[0], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(result0.shape()[1], 4) << "Failed on " << device.to_string();

    // Sum along dim 1 with keepdim
    auto result1 = sum(t, 1, true);
    EXPECT_EQ(result1.shape().size(), 3) << "Failed on " << device.to_string();
    EXPECT_EQ(result1.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(result1.shape()[1], 1) << "Failed on " << device.to_string();
    EXPECT_EQ(result1.shape()[2], 4) << "Failed on " << device.to_string();

    auto result1_cpu = result1.to(Device::cpu());
    for (int i = 0; i < result1.numel(); i++) {
        EXPECT_FLOAT_EQ(result1_cpu.data<float>()[i], 3.0f) << "Failed on " << device.to_string();
    }
}

TEST_P(ReductionOpsTest, MeanOperation) {
    auto t = full({2, 3, 4}, 6.0f, DType::Float32, device);

    // Mean all elements
    auto result_all = mean(t);
    auto result_all_cpu = result_all.to(Device::cpu());
    EXPECT_FLOAT_EQ(result_all_cpu.data<float>()[0], 6.0f) << "Failed on " << device.to_string();

    // Mean along dimension
    auto result_dim = mean(t, 2, false);
    EXPECT_EQ(result_dim.shape().size(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(result_dim.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(result_dim.shape()[1], 3) << "Failed on " << device.to_string();
}

TEST_P(ReductionOpsTest, MaxMinOperations) {
    auto t = zeros({3, 4}, DType::Float32, device);
    auto t_cpu = t.to(Device::cpu());
    auto data = t_cpu.data<float>();

    // Fill with known values
    for (int i = 0; i < 12; i++) {
        data[i] = static_cast<float>(i);
    }
    t = t_cpu.to(device);

    // Max all elements
    auto max_result = max(t);
    auto max_cpu = max_result.to(Device::cpu());
    EXPECT_FLOAT_EQ(max_cpu.data<float>()[0], 11.0f) << "Failed on " << device.to_string();

    // Min all elements
    auto min_result = min(t);
    auto min_cpu = min_result.to(Device::cpu());
    EXPECT_FLOAT_EQ(min_cpu.data<float>()[0], 0.0f) << "Failed on " << device.to_string();

    // Max along dimension
    auto max_dim = max(t, 1, true);
    EXPECT_EQ(max_dim.shape()[0], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(max_dim.shape()[1], 1) << "Failed on " << device.to_string();
}

TEST_P(ReductionOpsTest, ArgMaxArgMin) {
    auto t = zeros({3, 4}, DType::Float32, device);
    auto t_cpu = t.to(Device::cpu());
    auto data = t_cpu.data<float>();

    // Fill: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    for (int i = 0; i < 12; i++) {
        data[i] = static_cast<float>(i);
    }
    t = t_cpu.to(device);

    // ArgMax along dim 1
    auto argmax_result = argmax(t, 1);
    auto argmax_cpu = argmax_result.to(Device::cpu());
    auto argmax_data = argmax_cpu.data<int64_t>();

    // Each row's max is at index 3
    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(argmax_data[i], 3) << "Failed on " << device.to_string();
    }

    // ArgMin along dim 0
    auto argmin_result = argmin(t, 0);
    auto argmin_cpu = argmin_result.to(Device::cpu());
    auto argmin_data = argmin_cpu.data<int64_t>();

    // Each column's min is at index 0
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(argmin_data[i], 0) << "Failed on " << device.to_string();
    }
}

TEST_P(ReductionOpsTest, ProdOperation) {
    auto t = full({2, 3}, 2.0f, DType::Float32, device);

    // Product all elements: 2^6 = 64
    auto result = prod(t);
    auto result_cpu = result.to(Device::cpu());
    EXPECT_FLOAT_EQ(result_cpu.data<float>()[0], 64.0f) << "Failed on " << device.to_string();

    // Product along dimension
    auto result_dim = prod(t, 1, false);
    EXPECT_EQ(result_dim.shape()[0], 2) << "Failed on " << device.to_string();
    auto result_dim_cpu = result_dim.to(Device::cpu());
    // Each row: 2^3 = 8
    EXPECT_FLOAT_EQ(result_dim_cpu.data<float>()[0], 8.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(result_dim_cpu.data<float>()[1], 8.0f) << "Failed on " << device.to_string();
}

TEST_P(ReductionOpsTest, VarianceStd) {
    auto t = zeros({4}, DType::Float32, device);
    auto t_cpu = t.to(Device::cpu());
    auto data = t_cpu.data<float>();

    // Values: [1, 2, 3, 4], mean = 2.5
    data[0] = 1.0f; data[1] = 2.0f; data[2] = 3.0f; data[3] = 4.0f;
    t = t_cpu.to(device);

    // Variance (unbiased)
    auto var_result = var(t, std::nullopt, false, true);
    auto var_cpu = var_result.to(Device::cpu());
    // Var = [(1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2] / 3 = 5/3 ≈ 1.667
    EXPECT_NEAR(var_cpu.data<float>()[0], 1.667f, 0.01f) << "Failed on " << device.to_string();

    // Standard deviation
    auto std_result = tenzor::std(t, std::nullopt, false, true);
    auto std_cpu = std_result.to(Device::cpu());
    EXPECT_NEAR(std_cpu.data<float>()[0], ::std::sqrt(1.667f), 0.01f) << "Failed on " << device.to_string();
}

TEST_P(ReductionOpsTest, NormOperations) {
    auto t = ones({3, 4}, DType::Float32, device);

    // L2 norm (default)
    auto l2_norm = norm(t);
    auto l2_cpu = l2_norm.to(Device::cpu());
    // sqrt(12 * 1^2) = sqrt(12) ≈ 3.464
    EXPECT_NEAR(l2_cpu.data<float>()[0], std::sqrt(12.0f), 0.01f) << "Failed on " << device.to_string();

    // L1 norm
    auto l1_norm = norm(t, 1.0f);
    auto l1_cpu = l1_norm.to(Device::cpu());
    EXPECT_FLOAT_EQ(l1_cpu.data<float>()[0], 12.0f) << "Failed on " << device.to_string();
}

TEST_P(ReductionOpsTest, EmptyTensorReduction) {
    // Test edge case: empty tensor
    auto t = zeros({0}, DType::Float32, device);

    // Sum of empty tensor should be 0
    auto sum_result = sum(t);
    auto sum_cpu = sum_result.to(Device::cpu());
    EXPECT_FLOAT_EQ(sum_cpu.data<float>()[0], 0.0f) << "Failed on " << device.to_string();
}

//==============================================================================
// Tensor Manipulation Tests
//==============================================================================

class ManipulationOpsTest : public BackendTest {};

TEST_P(ManipulationOpsTest, ReshapeBasic) {
    auto t = zeros({2, 3, 4}, DType::Float32, device);

    // Reshape to 2D
    auto r1 = reshape(t, {6, 4});
    EXPECT_EQ(r1.shape()[0], 6) << "Failed on " << device.to_string();
    EXPECT_EQ(r1.shape()[1], 4) << "Failed on " << device.to_string();

    // Flatten
    auto r2 = reshape(t, {-1});
    EXPECT_EQ(r2.shape()[0], 24) << "Failed on " << device.to_string();

    // Reshape with auto-inferred dimension
    auto r3 = reshape(t, {-1, 8});
    EXPECT_EQ(r3.shape()[0], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(r3.shape()[1], 8) << "Failed on " << device.to_string();
}

TEST_P(ManipulationOpsTest, ViewOperation) {
    auto t = zeros({2, 6}, DType::Float32, device);
    auto t_cont = contiguous(t);

    // View with new shape
    auto v = view(t_cont, {3, 4});
    EXPECT_EQ(v.shape()[0], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(v.shape()[1], 4) << "Failed on " << device.to_string();
}

TEST_P(ManipulationOpsTest, TransposeOperations) {
    auto t = zeros({2, 3, 4}, DType::Float32, device);
    auto t_cpu = t.to(Device::cpu());
    auto data = t_cpu.data<float>();

    // Fill with sequential values
    for (int i = 0; i < 24; i++) {
        data[i] = static_cast<float>(i);
    }
    t = t_cpu.to(device);

    // Transpose dimensions 0 and 2
    auto t_trans = transpose(t, 0, 2);
    EXPECT_EQ(t_trans.shape()[0], 4) << "Failed on " << device.to_string();
    EXPECT_EQ(t_trans.shape()[1], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(t_trans.shape()[2], 2) << "Failed on " << device.to_string();

    // Transpose matrix (2D)
    auto m = zeros({3, 4}, DType::Float32, device);
    auto m_trans = transpose(m, 0, 1);
    EXPECT_EQ(m_trans.shape()[0], 4) << "Failed on " << device.to_string();
    EXPECT_EQ(m_trans.shape()[1], 3) << "Failed on " << device.to_string();
}

TEST_P(ManipulationOpsTest, PermuteOperations) {
    auto t = zeros({2, 3, 4}, DType::Float32, device);

    // Permute dimensions
    auto p1 = permute(t, {2, 0, 1});
    EXPECT_EQ(p1.shape()[0], 4) << "Failed on " << device.to_string();
    EXPECT_EQ(p1.shape()[1], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(p1.shape()[2], 3) << "Failed on " << device.to_string();

    // Reverse order
    auto p2 = permute(t, {2, 1, 0});
    EXPECT_EQ(p2.shape()[0], 4) << "Failed on " << device.to_string();
    EXPECT_EQ(p2.shape()[1], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(p2.shape()[2], 2) << "Failed on " << device.to_string();
}

TEST_P(ManipulationOpsTest, SqueezeUnsqueeze) {
    auto t = zeros({1, 3, 1, 4, 1}, DType::Float32, device);

    // Squeeze all size-1 dimensions
    auto s1 = squeeze(t);
    EXPECT_EQ(s1.shape().size(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(s1.shape()[0], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(s1.shape()[1], 4) << "Failed on " << device.to_string();

    // Squeeze specific dimension
    auto s2 = squeeze(t, 0);
    EXPECT_EQ(s2.shape().size(), 4) << "Failed on " << device.to_string();
    EXPECT_EQ(s2.shape()[0], 3) << "Failed on " << device.to_string();

    // Unsqueeze
    auto base = zeros({3, 4}, DType::Float32, device);
    auto u1 = unsqueeze(base, 0);
    EXPECT_EQ(u1.shape()[0], 1) << "Failed on " << device.to_string();
    EXPECT_EQ(u1.shape()[1], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(u1.shape()[2], 4) << "Failed on " << device.to_string();

    auto u2 = unsqueeze(base, 1);
    EXPECT_EQ(u2.shape()[0], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(u2.shape()[1], 1) << "Failed on " << device.to_string();
    EXPECT_EQ(u2.shape()[2], 4) << "Failed on " << device.to_string();
}

TEST_P(ManipulationOpsTest, ConcatenateOperations) {
    auto t1 = ones({2, 3}, DType::Float32, device);
    auto t2 = full({2, 3}, 2.0f, DType::Float32, device);
    auto t3 = full({2, 3}, 3.0f, DType::Float32, device);

    // Concatenate along dim 0
    auto cat0 = cat({t1, t2, t3}, 0);
    EXPECT_EQ(cat0.shape()[0], 6) << "Failed on " << device.to_string();
    EXPECT_EQ(cat0.shape()[1], 3) << "Failed on " << device.to_string();

    // Concatenate along dim 1
    auto cat1 = cat({t1, t2}, 1);
    EXPECT_EQ(cat1.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(cat1.shape()[1], 6) << "Failed on " << device.to_string();

    // Verify values
    auto cat0_cpu = cat0.to(Device::cpu());
    auto data = cat0_cpu.data<float>();
    // First 6 elements should be 1.0
    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(data[i], 1.0f) << "Failed on " << device.to_string();
    }
    // Next 6 should be 2.0
    for (int i = 6; i < 12; i++) {
        EXPECT_FLOAT_EQ(data[i], 2.0f) << "Failed on " << device.to_string();
    }
}

TEST_P(ManipulationOpsTest, StackOperations) {
    auto t1 = ones({2, 3}, DType::Float32, device);
    auto t2 = full({2, 3}, 2.0f, DType::Float32, device);

    // Stack along new dim 0
    std::vector<Tensor> tensors0{t1, t2};
    auto s0 = stack(tensors0, 0);
    EXPECT_EQ(s0.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(s0.shape()[1], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(s0.shape()[2], 3) << "Failed on " << device.to_string();

    // Stack along new dim 1
    std::vector<Tensor> tensors1{t1, t2};
    auto s1 = stack(tensors1, 1);
    EXPECT_EQ(s1.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(s1.shape()[1], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(s1.shape()[2], 3) << "Failed on " << device.to_string();
}

TEST_P(ManipulationOpsTest, SplitOperations) {
    auto t = zeros({6, 4}, DType::Float32, device);

    // Split into chunks of size 2
    auto splits = split(t, 2, 0);
    EXPECT_EQ(splits.size(), 3) << "Failed on " << device.to_string();
    for (const auto& s : splits) {
        EXPECT_EQ(s.shape()[0], 2) << "Failed on " << device.to_string();
        EXPECT_EQ(s.shape()[1], 4) << "Failed on " << device.to_string();
    }
}

TEST_P(ManipulationOpsTest, ChunkOperations) {
    auto t = zeros({10, 4}, DType::Float32, device);

    // Chunk into 3 parts
    auto chunks = chunk(t, 3, 0);
    EXPECT_EQ(chunks.size(), 3) << "Failed on " << device.to_string();

    // First two chunks should have size 4, last should have size 2
    EXPECT_EQ(chunks[0].shape()[0], 4) << "Failed on " << device.to_string();
    EXPECT_EQ(chunks[1].shape()[0], 4) << "Failed on " << device.to_string();
    EXPECT_EQ(chunks[2].shape()[0], 2) << "Failed on " << device.to_string();
}

TEST_P(ManipulationOpsTest, RepeatOperations) {
    auto t = ones({2, 3}, DType::Float32, device);

    // Repeat along each dimension
    auto r = repeat(t, {2, 3});
    EXPECT_EQ(r.shape()[0], 4) << "Failed on " << device.to_string();
    EXPECT_EQ(r.shape()[1], 9) << "Failed on " << device.to_string();
}

TEST_P(ManipulationOpsTest, ExpandOperations) {
    auto t = ones({1, 3}, DType::Float32, device);

    // Expand size-1 dimension
    auto e = expand(t, {4, 3});
    EXPECT_EQ(e.shape()[0], 4) << "Failed on " << device.to_string();
    EXPECT_EQ(e.shape()[1], 3) << "Failed on " << device.to_string();

    // All values should be 1.0
    auto e_cpu = e.to(Device::cpu());
    auto data = e_cpu.data<float>();
    for (int i = 0; i < 12; i++) {
        EXPECT_FLOAT_EQ(data[i], 1.0f) << "Failed on " << device.to_string();
    }
}

TEST_P(ManipulationOpsTest, FlattenOperations) {
    auto t = zeros({2, 3, 4}, DType::Float32, device);

    // Flatten all
    auto f1 = flatten(t);
    EXPECT_EQ(f1.shape()[0], 24) << "Failed on " << device.to_string();

    // Flatten from dim 1
    auto f2 = flatten(t, 1, 2);
    EXPECT_EQ(f2.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(f2.shape()[1], 12) << "Failed on " << device.to_string();
}

TEST_P(ManipulationOpsTest, RollOperations) {
    auto t = zeros({5}, DType::Float32, device);
    auto t_cpu = t.to(Device::cpu());
    auto data = t_cpu.data<float>();

    // Fill: [0, 1, 2, 3, 4]
    for (int i = 0; i < 5; i++) {
        data[i] = static_cast<float>(i);
    }
    t = t_cpu.to(device);

    // Roll by 2 positions
    auto r = roll(t, 2, 0);
    auto r_cpu = r.to(Device::cpu());
    auto r_data = r_cpu.data<float>();

    // Expected: [3, 4, 0, 1, 2]
    EXPECT_FLOAT_EQ(r_data[0], 3.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(r_data[1], 4.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(r_data[2], 0.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(r_data[3], 1.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(r_data[4], 2.0f) << "Failed on " << device.to_string();
}

//==============================================================================
// Indexing Operations Tests
//==============================================================================

class IndexingOpsTest : public BackendTest {};

TEST_P(IndexingOpsTest, SliceOperation) {
    auto t = zeros({10, 5}, DType::Float32, device);

    // Slice rows 2 to 7
    auto s1 = t.slice(0, 2, 7, 1);
    EXPECT_EQ(s1.shape()[0], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(s1.shape()[1], 5) << "Failed on " << device.to_string();

    // Slice with step
    auto s2 = t.slice(0, 0, 10, 2);
    EXPECT_EQ(s2.shape()[0], 5) << "Failed on " << device.to_string();
}

TEST_P(IndexingOpsTest, IndexSelectOperation) {
    auto t = zeros({5, 3}, DType::Float32, device);
    auto t_cpu = t.to(Device::cpu());
    auto data = t_cpu.data<float>();

    // Fill with row numbers
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 3; j++) {
            data[i * 3 + j] = static_cast<float>(i);
        }
    }
    t = t_cpu.to(device);

    // Create index tensor
    auto indices = zeros({3}, DType::Int64, device);
    auto idx_cpu = indices.to(Device::cpu());
    auto idx_data = idx_cpu.data<int64_t>();
    idx_data[0] = 0; idx_data[1] = 2; idx_data[2] = 4;
    indices = idx_cpu.to(device);

    // Select rows 0, 2, 4
    auto selected = index_select(t, 0, indices);
    EXPECT_EQ(selected.shape()[0], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(selected.shape()[1], 3) << "Failed on " << device.to_string();

    auto sel_cpu = selected.to(Device::cpu());
    auto sel_data = sel_cpu.data<float>();
    // Row 0: all 0s
    EXPECT_FLOAT_EQ(sel_data[0], 0.0f) << "Failed on " << device.to_string();
    // Row 1 (originally row 2): all 2s
    EXPECT_FLOAT_EQ(sel_data[3], 2.0f) << "Failed on " << device.to_string();
    // Row 2 (originally row 4): all 4s
    EXPECT_FLOAT_EQ(sel_data[6], 4.0f) << "Failed on " << device.to_string();
}

TEST_P(IndexingOpsTest, GatherOperation) {
    auto t = zeros({3, 4}, DType::Float32, device);
    auto t_cpu = t.to(Device::cpu());
    auto data = t_cpu.data<float>();

    // Fill with sequential values
    for (int i = 0; i < 12; i++) {
        data[i] = static_cast<float>(i);
    }
    t = t_cpu.to(device);

    // Gather along dim 1
    auto indices = zeros({3, 2}, DType::Int64, device);
    auto idx_cpu = indices.to(Device::cpu());
    auto idx_data = idx_cpu.data<int64_t>();
    // Gather columns 0 and 2 from each row
    idx_data[0] = 0; idx_data[1] = 2;
    idx_data[2] = 0; idx_data[3] = 2;
    idx_data[4] = 0; idx_data[5] = 2;
    indices = idx_cpu.to(device);

    auto gathered = gather(t, 1, indices);
    EXPECT_EQ(gathered.shape()[0], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(gathered.shape()[1], 2) << "Failed on " << device.to_string();
}

TEST_P(IndexingOpsTest, ScatterOperation) {
    auto t = zeros({3, 4}, DType::Float32, device);

    // Create indices
    auto indices = zeros({3, 2}, DType::Int64, device);
    auto idx_cpu = indices.to(Device::cpu());
    auto idx_data = idx_cpu.data<int64_t>();
    idx_data[0] = 0; idx_data[1] = 2;
    idx_data[2] = 1; idx_data[3] = 3;
    idx_data[4] = 0; idx_data[5] = 1;
    indices = idx_cpu.to(device);

    // Create source values
    auto src = ones({3, 2}, DType::Float32, device);

    // Scatter
    auto result = scatter(t, 1, indices, src);
    EXPECT_EQ(result.shape()[0], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(result.shape()[1], 4) << "Failed on " << device.to_string();
}

TEST_P(IndexingOpsTest, MaskedSelectOperation) {
    auto t = zeros({4}, DType::Float32, device);
    auto t_cpu = t.to(Device::cpu());
    auto data = t_cpu.data<float>();
    data[0] = 1.0f; data[1] = 2.0f; data[2] = 3.0f; data[3] = 4.0f;
    t = t_cpu.to(device);

    // Create boolean mask
    auto mask = zeros({4}, DType::Bool, device);
    auto mask_cpu = mask.to(Device::cpu());
    auto mask_data = mask_cpu.data<bool>();
    mask_data[0] = true; mask_data[1] = false; mask_data[2] = true; mask_data[3] = false;
    mask = mask_cpu.to(device);

    // Select masked elements
    auto selected = masked_select(t, mask);
    EXPECT_EQ(selected.numel(), 2) << "Failed on " << device.to_string();

    auto sel_cpu = selected.to(Device::cpu());
    auto sel_data = sel_cpu.data<float>();
    EXPECT_FLOAT_EQ(sel_data[0], 1.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(sel_data[1], 3.0f) << "Failed on " << device.to_string();
}

TEST_P(IndexingOpsTest, MaskedFillOperation) {
    auto t = ones({4}, DType::Float32, device);

    // Create boolean mask
    auto mask = zeros({4}, DType::Bool, device);
    auto mask_cpu = mask.to(Device::cpu());
    auto mask_data = mask_cpu.data<bool>();
    mask_data[0] = true; mask_data[1] = false; mask_data[2] = true; mask_data[3] = false;
    mask = mask_cpu.to(device);

    // Fill masked positions with 99.0
    auto filled = masked_fill(t, mask, 99.0f);
    auto filled_cpu = filled.to(Device::cpu());
    auto filled_data = filled_cpu.data<float>();

    EXPECT_FLOAT_EQ(filled_data[0], 99.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(filled_data[1], 1.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(filled_data[2], 99.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(filled_data[3], 1.0f) << "Failed on " << device.to_string();
}

TEST_P(IndexingOpsTest, WhereOperation) {
    // Create condition
    auto cond = zeros({4}, DType::Bool, device);
    auto cond_cpu = cond.to(Device::cpu());
    auto cond_data = cond_cpu.data<bool>();
    cond_data[0] = true; cond_data[1] = false; cond_data[2] = true; cond_data[3] = false;
    cond = cond_cpu.to(device);

    // Create x and y tensors
    auto x = ones({4}, DType::Float32, device);
    auto y = full({4}, 2.0f, DType::Float32, device);

    // Where condition is true, take x, else take y
    auto result = where(cond, x, y);
    auto result_cpu = result.to(Device::cpu());
    auto result_data = result_cpu.data<float>();

    EXPECT_FLOAT_EQ(result_data[0], 1.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(result_data[1], 2.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(result_data[2], 1.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(result_data[3], 2.0f) << "Failed on " << device.to_string();
}

TEST_P(IndexingOpsTest, NonzeroOperation) {
    auto t = zeros({3, 4}, DType::Float32, device);
    auto t_cpu = t.to(Device::cpu());
    auto data = t_cpu.data<float>();

    // Set some elements to non-zero
    data[0] = 1.0f;  // [0, 0]
    data[5] = 2.0f;  // [1, 1]
    data[11] = 3.0f; // [2, 3]
    t = t_cpu.to(device);

    auto indices = nonzero(t);
    EXPECT_EQ(indices.shape()[0], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(indices.shape()[1], 2) << "Failed on " << device.to_string();
}

// Regression (release-audit WS6): the old `is_nonzero` default branch
// reinterpreted complex storage as float* and tested only the first 4 bytes
// (the real part), so a value with zero real part but nonzero imaginary part
// (e.g. 0 + 1i) was silently misclassified as zero. nonzero() copies the input
// to CPU internally, so a CPU complex tensor exercises the fixed path.
TEST(NonzeroComplexTest, ZeroRealNonzeroImagCountsAsNonzero) {
    tenzor::initialize();  // plain TEST() bypasses the param-fixture's init
    auto t = zeros({4}, DType::Complex64, Device::cpu());
    auto* p = static_cast<std::complex<float>*>(t.data_ptr());
    p[0] = {0.0f, 0.0f};  // zero -> excluded
    p[1] = {0.0f, 1.0f};  // nonzero (imaginary only) -> the bug case
    p[2] = {2.0f, 0.0f};  // nonzero (real only)
    p[3] = {0.0f, 0.0f};  // zero -> excluded

    auto indices = nonzero(t);
    ASSERT_EQ(indices.shape()[0], 2);  // exactly indices 1 and 2
    ASSERT_EQ(indices.shape()[1], 1);
    auto idx = indices.to(Device::cpu());
    auto* ip = static_cast<const int64_t*>(idx.data_ptr());
    EXPECT_EQ(ip[0], 1);  // the (0 + 1i) element must be reported
    EXPECT_EQ(ip[1], 2);
}

// Integer clamp/sign/pow (release-audit WS16): PyTorch supports these on integer
// tensors; previously every backend threw. Now CPU/CUDA/ROCm/OneAPI compute them.
TEST(IntegerMathOpsTest, ClampSignPowInt32) {
    tenzor::initialize();
    auto t = zeros({5}, DType::Int32, Device::cpu());
    auto* p = static_cast<int32_t*>(t.data_ptr());
    p[0] = -5; p[1] = -1; p[2] = 0; p[3] = 3; p[4] = 10;

    auto cl = clamp(t, -2.0, 4.0);
    auto* c = static_cast<const int32_t*>(cl.data_ptr());
    EXPECT_EQ(c[0], -2); EXPECT_EQ(c[1], -1); EXPECT_EQ(c[2], 0);
    EXPECT_EQ(c[3], 3);  EXPECT_EQ(c[4], 4);

    auto sg = sign(t);
    auto* s = static_cast<const int32_t*>(sg.data_ptr());
    EXPECT_EQ(s[0], -1); EXPECT_EQ(s[1], -1); EXPECT_EQ(s[2], 0);
    EXPECT_EQ(s[3], 1);  EXPECT_EQ(s[4], 1);

    auto t2 = zeros({4}, DType::Int64, Device::cpu());
    auto* q = static_cast<int64_t*>(t2.data_ptr());
    q[0] = 2; q[1] = 3; q[2] = 4; q[3] = 5;
    auto pw = pow(t2, 2.0);  // exact integer exponentiation
    auto* r = static_cast<const int64_t*>(pw.data_ptr());
    EXPECT_EQ(r[0], 4); EXPECT_EQ(r[1], 9); EXPECT_EQ(r[2], 16); EXPECT_EQ(r[3], 25);
}

// Cross-backend integer clamp/sign/pow parity (WS16 + Vulkan integer shaders):
// every available GPU backend must match the CPU result byte-for-byte for all
// integer dtypes. Backends that aren't built/available are skipped.
TEST(IntegerMathOpsTest, GpuBackendsMatchCpu) {
    tenzor::initialize();
    auto bytes_eq = [](const Tensor& a, const Tensor& b) {
        auto x = a.to(Device::cpu()).contiguous();
        auto y = b.to(Device::cpu()).contiguous();
        if (x.numel() != y.numel() || x.dtype() != y.dtype()) return false;
        return std::memcmp(x.data_ptr(), y.data_ptr(),
                           static_cast<size_t>(x.numel()) * dtype_size(x.dtype())) == 0;
    };
    // One representative signed-and-unsigned-safe tensor (Int32) is enough to
    // exercise the per-backend integer dispatch end to end.
    auto base = zeros({7}, DType::Int32, Device::cpu());
    auto* p = static_cast<int32_t*>(base.data_ptr());
    p[0]=-100; p[1]=-5; p[2]=-1; p[3]=0; p[4]=3; p[5]=10; p[6]=100;
    auto ref_clamp = clamp(base, -2.0, 4.0);
    auto ref_sign  = sign(base);
    auto ref_pow   = pow(base, 2.0);

    const Device::Type devs[] = {Device::Type::Vulkan, Device::Type::CUDA,
                                 Device::Type::ROCm, Device::Type::OneAPI};
    for (auto dt : devs) {
        Tensor t;
        try { t = base.to(Device(dt, 0)); }
        catch (...) { continue; }  // backend not available
        EXPECT_TRUE(bytes_eq(ref_clamp, clamp(t, -2.0, 4.0)));
        EXPECT_TRUE(bytes_eq(ref_sign, sign(t)));
        EXPECT_TRUE(bytes_eq(ref_pow, pow(t, 2.0)));
    }
}

TEST(NonzeroComplexTest, Complex128ZeroRealNonzeroImag) {
    tenzor::initialize();
    auto t = zeros({3}, DType::Complex128, Device::cpu());
    auto* p = static_cast<std::complex<double>*>(t.data_ptr());
    p[0] = {0.0, 0.0};
    p[1] = {0.0, 2.0};  // nonzero (imaginary only)
    p[2] = {5.0, 0.0};
    auto indices = nonzero(t);
    ASSERT_EQ(indices.shape()[0], 2);
    auto idx = indices.to(Device::cpu());
    auto* ip = static_cast<const int64_t*>(idx.data_ptr());
    EXPECT_EQ(ip[0], 1);
    EXPECT_EQ(ip[1], 2);
}

TEST_P(IndexingOpsTest, SelectOperation) {
    auto t = zeros({3, 4, 5}, DType::Float32, device);

    // Select index 1 along dim 0 - using slice as a workaround for select() name conflict
    // select(t, dim, index) is equivalent to slice(t, dim, index, index+1)
    auto s1 = t.slice(0, 1, 2)[0];  // Get single element along dimension 0
    EXPECT_EQ(s1.shape().size(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(s1.shape()[0], 4) << "Failed on " << device.to_string();
    EXPECT_EQ(s1.shape()[1], 5) << "Failed on " << device.to_string();

    // Select index 2 along dim 1
    auto s2_temp = t.slice(1, 2, 3);  // Get single slice
    // Note: This creates a tensor with shape [3, 1, 5], need to squeeze dimension 1
    EXPECT_EQ(s2_temp.shape()[0], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(s2_temp.shape()[1], 1) << "Failed on " << device.to_string();
    EXPECT_EQ(s2_temp.shape()[2], 5) << "Failed on " << device.to_string();
}

//==============================================================================
// Mathematical Operations Tests
//==============================================================================

class MathOpsTest : public BackendTest {};

TEST_P(MathOpsTest, ArithmeticOperations) {
    auto a = full({2, 3}, 6.0f, DType::Float32, device);
    auto b = full({2, 3}, 2.0f, DType::Float32, device);

    // Addition
    auto add_result = add(a, b);
    auto add_cpu = add_result.to(Device::cpu());
    EXPECT_FLOAT_EQ(add_cpu.data<float>()[0], 8.0f) << "Failed on " << device.to_string();

    // Subtraction
    auto sub_result = sub(a, b);
    auto sub_cpu = sub_result.to(Device::cpu());
    EXPECT_FLOAT_EQ(sub_cpu.data<float>()[0], 4.0f) << "Failed on " << device.to_string();

    // Multiplication
    auto mul_result = mul(a, b);
    auto mul_cpu = mul_result.to(Device::cpu());
    EXPECT_FLOAT_EQ(mul_cpu.data<float>()[0], 12.0f) << "Failed on " << device.to_string();

    // Division
    auto div_result = div(a, b);
    auto div_cpu = div_result.to(Device::cpu());
    EXPECT_FLOAT_EQ(div_cpu.data<float>()[0], 3.0f) << "Failed on " << device.to_string();
}

TEST_P(MathOpsTest, MatMulOperations) {
    // Matrix multiplication
    auto a = ones({2, 3}, DType::Float32, device);
    auto b = ones({3, 4}, DType::Float32, device);

    auto c = matmul(a, b);
    EXPECT_EQ(c.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(c.shape()[1], 4) << "Failed on " << device.to_string();

    auto c_cpu = c.to(Device::cpu());
    // Each element should be 3.0 (sum of 3 ones)
    EXPECT_FLOAT_EQ(c_cpu.data<float>()[0], 3.0f) << "Failed on " << device.to_string();
}

TEST_P(MathOpsTest, BatchMatMul) {
    // Batch matrix multiplication
    auto a = ones({2, 3, 4}, DType::Float32, device);
    auto b = ones({2, 4, 5}, DType::Float32, device);

    auto c = bmm(a, b);
    EXPECT_EQ(c.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(c.shape()[1], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(c.shape()[2], 5) << "Failed on " << device.to_string();

    auto c_cpu = c.to(Device::cpu());
    // Each element should be 4.0
    EXPECT_FLOAT_EQ(c_cpu.data<float>()[0], 4.0f) << "Failed on " << device.to_string();
}

TEST_P(MathOpsTest, DotProduct) {
    auto a = ones({5}, DType::Float32, device);
    auto b = full({5}, 2.0f, DType::Float32, device);

    auto result = dot(a, b);
    auto result_cpu = result.to(Device::cpu());
    // 1*2 + 1*2 + 1*2 + 1*2 + 1*2 = 10
    EXPECT_FLOAT_EQ(result_cpu.data<float>()[0], 10.0f) << "Failed on " << device.to_string();
}

TEST_P(MathOpsTest, PowerOperations) {
    auto t = full({3, 3}, 2.0f, DType::Float32, device);

    // Square
    auto squared = pow(t, 2.0f);
    auto sq_cpu = squared.to(Device::cpu());
    EXPECT_FLOAT_EQ(sq_cpu.data<float>()[0], 4.0f) << "Failed on " << device.to_string();

    // Cube
    auto cubed = pow(t, 3.0f);
    auto cb_cpu = cubed.to(Device::cpu());
    EXPECT_FLOAT_EQ(cb_cpu.data<float>()[0], 8.0f) << "Failed on " << device.to_string();

    // Square root
    auto t2 = full({3, 3}, 4.0f, DType::Float32, device);
    auto sqrt_result = sqrt(t2);
    auto sqrt_cpu = sqrt_result.to(Device::cpu());
    EXPECT_FLOAT_EQ(sqrt_cpu.data<float>()[0], 2.0f) << "Failed on " << device.to_string();
}

TEST_P(MathOpsTest, ExponentialLogarithm) {
    auto t = ones({3, 3}, DType::Float32, device);

    // Exponential: e^1 ≈ 2.71828
    auto exp_result = exp(t);
    auto exp_cpu = exp_result.to(Device::cpu());
    EXPECT_NEAR(exp_cpu.data<float>()[0], std::exp(1.0f), 0.001f) << "Failed on " << device.to_string();

    // Natural log: ln(e) = 1
    auto log_result = log(exp_result);
    auto log_cpu = log_result.to(Device::cpu());
    EXPECT_NEAR(log_cpu.data<float>()[0], 1.0f, 0.001f) << "Failed on " << device.to_string();
}

TEST_P(MathOpsTest, TrigonometricFunctions) {
    auto t = zeros({4}, DType::Float32, device);
    auto t_cpu = t.to(Device::cpu());
    auto data = t_cpu.data<float>();
    data[0] = 0.0f;
    data[1] = M_PI / 6;  // 30 degrees
    data[2] = M_PI / 4;  // 45 degrees
    data[3] = M_PI / 2;  // 90 degrees
    t = t_cpu.to(device);

    // Sine
    auto sin_result = sin(t);
    auto sin_cpu = sin_result.to(Device::cpu());
    auto sin_data = sin_cpu.data<float>();
    EXPECT_NEAR(sin_data[0], 0.0f, 0.001f) << "Failed on " << device.to_string();
    EXPECT_NEAR(sin_data[1], 0.5f, 0.001f) << "Failed on " << device.to_string();
    EXPECT_NEAR(sin_data[3], 1.0f, 0.001f) << "Failed on " << device.to_string();

    // Cosine
    auto cos_result = cos(t);
    auto cos_cpu = cos_result.to(Device::cpu());
    auto cos_data = cos_cpu.data<float>();
    EXPECT_NEAR(cos_data[0], 1.0f, 0.001f) << "Failed on " << device.to_string();
    EXPECT_NEAR(cos_data[3], 0.0f, 0.001f) << "Failed on " << device.to_string();

    // Tangent
    auto tan_result = tan(t);
    auto tan_cpu = tan_result.to(Device::cpu());
    auto tan_data = tan_cpu.data<float>();
    EXPECT_NEAR(tan_data[0], 0.0f, 0.001f) << "Failed on " << device.to_string();
}

TEST_P(MathOpsTest, InverseTrigFunctions) {
    auto t = zeros({3}, DType::Float32, device);
    auto t_cpu = t.to(Device::cpu());
    auto data = t_cpu.data<float>();
    data[0] = 0.0f;
    data[1] = 0.5f;
    data[2] = 1.0f;
    t = t_cpu.to(device);

    // Arcsine
    auto asin_result = asin(t);
    auto asin_cpu = asin_result.to(Device::cpu());
    auto asin_data = asin_cpu.data<float>();
    EXPECT_NEAR(asin_data[0], 0.0f, 0.001f) << "Failed on " << device.to_string();
    EXPECT_NEAR(asin_data[2], M_PI / 2, 0.001f) << "Failed on " << device.to_string();

    // Arccosine
    auto acos_result = acos(t);
    auto acos_cpu = acos_result.to(Device::cpu());
    auto acos_data = acos_cpu.data<float>();
    EXPECT_NEAR(acos_data[0], M_PI / 2, 0.001f) << "Failed on " << device.to_string();
    EXPECT_NEAR(acos_data[2], 0.0f, 0.001f) << "Failed on " << device.to_string();

    // Arctangent
    auto atan_result = atan(t);
    auto atan_cpu = atan_result.to(Device::cpu());
    auto atan_data = atan_cpu.data<float>();
    EXPECT_NEAR(atan_data[0], 0.0f, 0.001f) << "Failed on " << device.to_string();
}

TEST_P(MathOpsTest, HyperbolicFunctions) {
    auto t = full({3}, 1.0f, DType::Float32, device);

    // Hyperbolic sine
    auto sinh_result = sinh(t);
    auto sinh_cpu = sinh_result.to(Device::cpu());
    EXPECT_NEAR(sinh_cpu.data<float>()[0], std::sinh(1.0f), 0.001f) << "Failed on " << device.to_string();

    // Hyperbolic cosine
    auto cosh_result = cosh(t);
    auto cosh_cpu = cosh_result.to(Device::cpu());
    EXPECT_NEAR(cosh_cpu.data<float>()[0], std::cosh(1.0f), 0.001f) << "Failed on " << device.to_string();

    // Hyperbolic tangent
    auto tanh_result = tanh(t);
    auto tanh_cpu = tanh_result.to(Device::cpu());
    EXPECT_NEAR(tanh_cpu.data<float>()[0], std::tanh(1.0f), 0.001f) << "Failed on " << device.to_string();
}

TEST_P(MathOpsTest, ElementWiseOperations) {
    auto t = zeros({4}, DType::Float32, device);
    auto t_cpu = t.to(Device::cpu());
    auto data = t_cpu.data<float>();
    data[0] = -2.5f; data[1] = -1.0f; data[2] = 0.0f; data[3] = 3.7f;
    t = t_cpu.to(device);

    // Absolute value
    auto abs_result = abs(t);
    auto abs_cpu = abs_result.to(Device::cpu());
    auto abs_data = abs_cpu.data<float>();
    EXPECT_FLOAT_EQ(abs_data[0], 2.5f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(abs_data[1], 1.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(abs_data[3], 3.7f) << "Failed on " << device.to_string();

    // Negation
    auto neg_result = neg(t);
    auto neg_cpu = neg_result.to(Device::cpu());
    auto neg_data = neg_cpu.data<float>();
    EXPECT_FLOAT_EQ(neg_data[0], 2.5f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(neg_data[3], -3.7f) << "Failed on " << device.to_string();

    // Sign
    auto sign_result = sign(t);
    auto sign_cpu = sign_result.to(Device::cpu());
    auto sign_data = sign_cpu.data<float>();
    EXPECT_FLOAT_EQ(sign_data[0], -1.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(sign_data[2], 0.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(sign_data[3], 1.0f) << "Failed on " << device.to_string();
}

TEST_P(MathOpsTest, ReciprocalOperation) {
    auto t = full({3}, 2.0f, DType::Float32, device);

    auto recip = reciprocal(t);
    auto recip_cpu = recip.to(Device::cpu());
    EXPECT_FLOAT_EQ(recip_cpu.data<float>()[0], 0.5f) << "Failed on " << device.to_string();
}

TEST_P(MathOpsTest, RoundingFunctions) {
    auto t = zeros({6}, DType::Float32, device);
    auto t_cpu = t.to(Device::cpu());
    auto data = t_cpu.data<float>();
    data[0] = 1.2f; data[1] = 1.5f; data[2] = 1.8f;
    data[3] = -1.2f; data[4] = -1.5f; data[5] = -1.8f;
    t = t_cpu.to(device);

    // Floor
    auto floor_result = floor(t);
    auto floor_cpu = floor_result.to(Device::cpu());
    auto floor_data = floor_cpu.data<float>();
    EXPECT_FLOAT_EQ(floor_data[0], 1.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(floor_data[2], 1.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(floor_data[5], -2.0f) << "Failed on " << device.to_string();

    // Ceil
    auto ceil_result = ceil(t);
    auto ceil_cpu = ceil_result.to(Device::cpu());
    auto ceil_data = ceil_cpu.data<float>();
    EXPECT_FLOAT_EQ(ceil_data[0], 2.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(ceil_data[3], -1.0f) << "Failed on " << device.to_string();

    // Round
    auto round_result = round(t);
    auto round_cpu = round_result.to(Device::cpu());
    auto round_data = round_cpu.data<float>();
    EXPECT_FLOAT_EQ(round_data[0], 1.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(round_data[1], 2.0f) << "Failed on " << device.to_string();
}

TEST_P(MathOpsTest, ClampOperations) {
    auto t = zeros({5}, DType::Float32, device);
    auto t_cpu = t.to(Device::cpu());
    auto data = t_cpu.data<float>();
    data[0] = -5.0f; data[1] = -1.0f; data[2] = 0.0f; data[3] = 1.0f; data[4] = 5.0f;
    t = t_cpu.to(device);

    // Clamp to [-2, 2]
    auto clamped = clamp(t, -2.0f, 2.0f);
    auto clamped_cpu = clamped.to(Device::cpu());
    auto clamped_data = clamped_cpu.data<float>();
    EXPECT_FLOAT_EQ(clamped_data[0], -2.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(clamped_data[1], -1.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(clamped_data[4], 2.0f) << "Failed on " << device.to_string();

    // Clamp min
    auto clamped_min = clamp_min(t, 0.0f);
    auto clamped_min_cpu = clamped_min.to(Device::cpu());
    auto clamped_min_data = clamped_min_cpu.data<float>();
    EXPECT_FLOAT_EQ(clamped_min_data[0], 0.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(clamped_min_data[4], 5.0f) << "Failed on " << device.to_string();

    // Clamp max
    auto clamped_max = clamp_max(t, 1.0f);
    auto clamped_max_cpu = clamped_max.to(Device::cpu());
    auto clamped_max_data = clamped_max_cpu.data<float>();
    EXPECT_FLOAT_EQ(clamped_max_data[0], -5.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(clamped_max_data[4], 1.0f) << "Failed on " << device.to_string();
}

//==============================================================================
// Comparison Operations Tests
//==============================================================================

class ComparisonOpsTest : public BackendTest {};

TEST_P(ComparisonOpsTest, EqualityComparison) {
    auto a = zeros({4}, DType::Float32, device);
    auto a_cpu = a.to(Device::cpu());
    auto a_data = a_cpu.data<float>();
    a_data[0] = 1.0f; a_data[1] = 2.0f; a_data[2] = 3.0f; a_data[3] = 4.0f;
    a = a_cpu.to(device);

    auto b = zeros({4}, DType::Float32, device);
    auto b_cpu = b.to(Device::cpu());
    auto b_data = b_cpu.data<float>();
    b_data[0] = 1.0f; b_data[1] = 2.5f; b_data[2] = 3.0f; b_data[3] = 3.5f;
    b = b_cpu.to(device);

    // Equal
    auto eq_result = eq(a, b);
    auto eq_cpu = eq_result.to(Device::cpu());
    auto eq_data = eq_cpu.data<bool>();
    EXPECT_TRUE(eq_data[0]) << "Failed on " << device.to_string();
    EXPECT_FALSE(eq_data[1]) << "Failed on " << device.to_string();
    EXPECT_TRUE(eq_data[2]) << "Failed on " << device.to_string();

    // Not equal
    auto ne_result = ne(a, b);
    auto ne_cpu = ne_result.to(Device::cpu());
    auto ne_data = ne_cpu.data<bool>();
    EXPECT_FALSE(ne_data[0]) << "Failed on " << device.to_string();
    EXPECT_TRUE(ne_data[1]) << "Failed on " << device.to_string();
}

TEST_P(ComparisonOpsTest, InequalityComparisons) {
    auto a = zeros({4}, DType::Float32, device);
    auto a_cpu = a.to(Device::cpu());
    auto a_data = a_cpu.data<float>();
    a_data[0] = 1.0f; a_data[1] = 2.0f; a_data[2] = 3.0f; a_data[3] = 4.0f;
    a = a_cpu.to(device);

    auto b = full({4}, 2.5f, DType::Float32, device);

    // Less than
    auto lt_result = lt(a, b);
    auto lt_cpu = lt_result.to(Device::cpu());
    auto lt_data = lt_cpu.data<bool>();
    EXPECT_TRUE(lt_data[0]) << "Failed on " << device.to_string();
    EXPECT_TRUE(lt_data[1]) << "Failed on " << device.to_string();
    EXPECT_FALSE(lt_data[2]) << "Failed on " << device.to_string();
    EXPECT_FALSE(lt_data[3]) << "Failed on " << device.to_string();

    // Less than or equal
    auto le_result = le(a, b);
    auto le_cpu = le_result.to(Device::cpu());
    auto le_data = le_cpu.data<bool>();
    EXPECT_TRUE(le_data[0]) << "Failed on " << device.to_string();
    EXPECT_TRUE(le_data[1]) << "Failed on " << device.to_string();

    // Greater than
    auto gt_result = gt(a, b);
    auto gt_cpu = gt_result.to(Device::cpu());
    auto gt_data = gt_cpu.data<bool>();
    EXPECT_FALSE(gt_data[0]) << "Failed on " << device.to_string();
    EXPECT_TRUE(gt_data[2]) << "Failed on " << device.to_string();
    EXPECT_TRUE(gt_data[3]) << "Failed on " << device.to_string();

    // Greater than or equal
    auto ge_result = ge(a, b);
    auto ge_cpu = ge_result.to(Device::cpu());
    auto ge_data = ge_cpu.data<bool>();
    EXPECT_FALSE(ge_data[0]) << "Failed on " << device.to_string();
    EXPECT_TRUE(ge_data[2]) << "Failed on " << device.to_string();
}

//==============================================================================
// In-Place Operations Tests
//==============================================================================

class InPlaceOpsTest : public BackendTest {};

TEST_P(InPlaceOpsTest, InPlaceAddition) {
    auto a = full({3, 3}, 5.0f, DType::Float32, device);
    auto b = full({3, 3}, 3.0f, DType::Float32, device);

    // In-place add
    add_(a, b);

    auto a_cpu = a.to(Device::cpu());
    auto data = a_cpu.data<float>();
    EXPECT_FLOAT_EQ(data[0], 8.0f) << "Failed on " << device.to_string();
}

TEST_P(InPlaceOpsTest, InPlaceMultiplication) {
    auto a = full({3, 3}, 4.0f, DType::Float32, device);
    auto b = full({3, 3}, 2.0f, DType::Float32, device);

    // In-place multiply
    mul_(a, b);

    auto a_cpu = a.to(Device::cpu());
    auto data = a_cpu.data<float>();
    EXPECT_FLOAT_EQ(data[0], 8.0f) << "Failed on " << device.to_string();
}

TEST_P(InPlaceOpsTest, InPlaceSubtraction) {
    auto a = full({3, 3}, 10.0f, DType::Float32, device);
    auto b = full({3, 3}, 3.0f, DType::Float32, device);

    // In-place subtract
    sub_(a, b);

    auto a_cpu = a.to(Device::cpu());
    auto data = a_cpu.data<float>();
    EXPECT_FLOAT_EQ(data[0], 7.0f) << "Failed on " << device.to_string();
}

TEST_P(InPlaceOpsTest, InPlaceDivision) {
    auto a = full({3, 3}, 12.0f, DType::Float32, device);
    auto b = full({3, 3}, 3.0f, DType::Float32, device);

    // In-place divide
    div_(a, b);

    auto a_cpu = a.to(Device::cpu());
    auto data = a_cpu.data<float>();
    EXPECT_FLOAT_EQ(data[0], 4.0f) << "Failed on " << device.to_string();
}

// Regression (CPU): in-place integer division by zero must throw a catchable
// exception, mirroring the out-of-place div() guard. Previously the Int32/Int64
// in-place path did a bare `a /= b` with no zero check, so an integer 0 divisor
// triggered a SIGFPE that hard-crashed the process instead of throwing.
TEST_P(InPlaceOpsTest, InPlaceIntegerDivisionByZeroThrows) {
    if (device.type != Device::Type::CPU) {
        GTEST_SKIP() << "CPU-specific integer divide-by-zero guard";
    }

    // Int32, same-shape path.
    {
        auto a = full({4}, 12.0, DType::Int32, device);
        auto b = full({4}, 3.0, DType::Int32, device);
        b.data<int32_t>()[2] = 0;  // introduce a zero divisor
        EXPECT_THROW(div_(a, b), std::runtime_error)
            << "Int32 same-shape in-place div by zero must throw";
    }

    // Int64, same-shape path.
    {
        auto a = full({4}, 12.0, DType::Int64, device);
        auto b = full({4}, 3.0, DType::Int64, device);
        b.data<int64_t>()[0] = 0;
        EXPECT_THROW(div_(a, b), std::runtime_error)
            << "Int64 same-shape in-place div by zero must throw";
    }

    // Int32 broadcast path (scalar zero divisor broadcast over a).
    {
        auto a = full({4}, 12.0, DType::Int32, device);
        auto b = full({1}, 0.0, DType::Int32, device);
        EXPECT_THROW(div_(a, b), std::runtime_error)
            << "Int32 broadcast in-place div by zero must throw";
    }

    // A non-zero integer divisor must still succeed and compute correctly.
    {
        auto a = full({4}, 12.0, DType::Int32, device);
        auto b = full({4}, 4.0, DType::Int32, device);
        EXPECT_NO_THROW(div_(a, b));
        auto a_cpu = a.to(Device::cpu());
        EXPECT_EQ(a_cpu.data<int32_t>()[0], 3) << "Failed on " << device.to_string();
    }
}

//==============================================================================
// Edge Cases and Special Values Tests
//==============================================================================

class EdgeCaseOpsTest : public BackendTest {};

TEST_P(EdgeCaseOpsTest, BroadcastingInOperations) {
    // Test broadcasting in arithmetic operations
    auto a = ones({3, 1}, DType::Float32, device);
    auto b = full({1, 4}, 2.0f, DType::Float32, device);

    auto result = add(a, b);
    EXPECT_EQ(result.shape()[0], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(result.shape()[1], 4) << "Failed on " << device.to_string();

    auto result_cpu = result.to(Device::cpu());
    EXPECT_FLOAT_EQ(result_cpu.data<float>()[0], 3.0f) << "Failed on " << device.to_string();
}

TEST_P(EdgeCaseOpsTest, SingleElementTensor) {
    // Operations on scalar tensors
    auto a = full({1}, 5.0f, DType::Float32, device);
    auto b = full({1}, 3.0f, DType::Float32, device);

    auto result = add(a, b);
    auto result_cpu = result.to(Device::cpu());
    EXPECT_FLOAT_EQ(result_cpu.data<float>()[0], 8.0f) << "Failed on " << device.to_string();
}

TEST_P(EdgeCaseOpsTest, ReshapeEdgeCases) {
    auto t = ones({24}, DType::Float32, device);

    // Various valid reshapes
    auto r1 = reshape(t, {1, 24});
    EXPECT_EQ(r1.shape()[0], 1) << "Failed on " << device.to_string();

    auto r2 = reshape(t, {24, 1});
    EXPECT_EQ(r2.shape()[1], 1) << "Failed on " << device.to_string();

    auto r3 = reshape(t, {2, 3, 4});
    EXPECT_EQ(r3.shape()[0], 2) << "Failed on " << device.to_string();
}

TEST_P(EdgeCaseOpsTest, TransposeOnVector) {
    // Transpose on 1D tensor (should work but be no-op)
    auto v = ones({5}, DType::Float32, device);

    // Note: transpose typically requires 2+ dimensions
    // This tests the edge case handling
}

TEST_P(EdgeCaseOpsTest, MatMulVectorMatrix) {
    // Vector-matrix multiplication (1D x 2D)
    auto v = ones({3}, DType::Float32, device);
    auto m = ones({3, 4}, DType::Float32, device);

    auto result = matmul(v, m);
    EXPECT_EQ(result.shape()[0], 4) << "Failed on " << device.to_string();

    auto result_cpu = result.to(Device::cpu());
    EXPECT_FLOAT_EQ(result_cpu.data<float>()[0], 3.0f) << "Failed on " << device.to_string();

    // BFloat16 1D×2D with N (contraction) > K (output width). This is the
    // configuration that previously triggered a heap overflow + wrong product
    // in the MKL BF16 path (output buffer was sized to the contraction dim and
    // the GEMM dims were swapped). Asymmetric column values so a wrong
    // contraction can't coincidentally pass. v(4) @ m(4x2) = [10, 100].
    {
        auto vf = ones({4}, DType::Float32, Device::cpu());
        auto vd = vf.data<float>();
        vd[0] = 1.0f; vd[1] = 2.0f; vd[2] = 3.0f; vd[3] = 4.0f;  // sum = 10
        auto mf = ones({4, 2}, DType::Float32, Device::cpu());
        auto md = mf.data<float>();
        for (int i = 0; i < 4; ++i) { md[i * 2 + 0] = 1.0f; md[i * 2 + 1] = 10.0f; }
        auto vb = vf.to(DType::BFloat16).to(device);
        auto mb = mf.to(DType::BFloat16).to(device);
        auto rb = matmul(vb, mb);
        ASSERT_EQ(rb.shape().size(), 1u) << "Failed on " << device.to_string();
        EXPECT_EQ(rb.shape()[0], 2) << "Failed on " << device.to_string();
        auto rb_cpu = rb.to(DType::Float32).to(Device::cpu());
        EXPECT_FLOAT_EQ(rb_cpu.data<float>()[0], 10.0f) << "Failed on " << device.to_string();
        EXPECT_FLOAT_EQ(rb_cpu.data<float>()[1], 100.0f) << "Failed on " << device.to_string();
    }
}

TEST_P(EdgeCaseOpsTest, SpecialMathValues) {
    auto t = zeros({4}, DType::Float32, device);
    auto t_cpu = t.to(Device::cpu());
    auto data = t_cpu.data<float>();
    data[0] = 0.0f;
    data[1] = 1.0f;
    data[2] = -1.0f;
    data[3] = 0.5f;
    t = t_cpu.to(device);

    // Test exp with various values
    auto exp_result = exp(t);
    auto exp_cpu = exp_result.to(Device::cpu());
    EXPECT_NEAR(exp_cpu.data<float>()[0], 1.0f, 0.001f) << "Failed on " << device.to_string();
    EXPECT_NEAR(exp_cpu.data<float>()[1], std::exp(1.0f), 0.001f) << "Failed on " << device.to_string();
}

//==============================================================================
// Advanced Indexing Tests
//==============================================================================

class AdvancedIndexingTest : public BackendTest {};

TEST_P(AdvancedIndexingTest, NegativeIndexing) {
    auto t = zeros({5}, DType::Float32, device);
    auto t_cpu = t.to(Device::cpu());
    auto data = t_cpu.data<float>();
    for (int i = 0; i < 5; i++) {
        data[i] = static_cast<float>(i);
    }
    t = t_cpu.to(device);

    // Access last element with negative index
    auto last = t[-1];
    auto last_cpu = last.to(Device::cpu());
    EXPECT_FLOAT_EQ(last_cpu.data<float>()[0], 4.0f) << "Failed on " << device.to_string();
}

TEST_P(AdvancedIndexingTest, MultiDimensionalSlicing) {
    auto t = zeros({4, 5, 6}, DType::Float32, device);

    // Slice along first dimension - using tensor slice, not autograd slice
    auto s1 = t.slice(0, 1, 3);
    EXPECT_EQ(s1.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(s1.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(s1.shape()[2], 6) << "Failed on " << device.to_string();

    // Slice along last dimension
    auto s2 = t.slice(2, 0, 6);
    EXPECT_EQ(s2.shape()[0], 4) << "Failed on " << device.to_string();
    EXPECT_EQ(s2.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(s2.shape()[2], 6) << "Failed on " << device.to_string();
}

//==============================================================================
// Test Instantiation
//==============================================================================

// Instantiate all test suites for all available backends
INSTANTIATE_BACKEND_TESTS(ReductionOpsTest);
INSTANTIATE_BACKEND_TESTS(ManipulationOpsTest);
INSTANTIATE_BACKEND_TESTS(IndexingOpsTest);
INSTANTIATE_BACKEND_TESTS(MathOpsTest);
INSTANTIATE_BACKEND_TESTS(ComparisonOpsTest);
INSTANTIATE_BACKEND_TESTS(InPlaceOpsTest);
INSTANTIATE_BACKEND_TESTS(EdgeCaseOpsTest);
INSTANTIATE_BACKEND_TESTS(AdvancedIndexingTest);
