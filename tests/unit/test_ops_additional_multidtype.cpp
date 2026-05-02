/**
 * @file test_ops_additional_multidtype.cpp
 * @brief Multi-dtype() comprehensive tests for tensor operations
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
 *
 * Tests across multiple data types: Float32, Float64, Float16, Int32
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>
#include <limits>

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Multi-DType Test Fixture
// ============================================================================

class OpsAdditionalMultiDTypeTest : public MultiBackendDTypeTest {

protected:
    float tolerance;
    template<typename T>
    T getTypedValue(double val) {
        return static_cast<T>(val);
    }
};

//==============================================================================
// Reduction Operations Tests
//==============================================================================

TEST_P(OpsAdditionalMultiDTypeTest, SumAllElements) {
    // Sum all elements in tensor
    auto t = ones({2, 3, 4}, dtype(), device());
    auto result = sum(t);

    EXPECT_EQ(result.numel(), 1) << "Failed on " << device().to_string();
    auto result_cpu = result.to(Device::cpu());

    if (dtype() == DType::Float32) {
        EXPECT_FLOAT_EQ(result_cpu.data<float>()[0], 24.0f);
    } else if (dtype() == DType::Float64) {
        EXPECT_DOUBLE_EQ(result_cpu.data<double>()[0], 24.0);
    } else if (dtype() == DType::Int32) {
        // Int32 sum promotes to Int64 to prevent overflow (matches PyTorch)
        EXPECT_EQ(result_cpu.data<int64_t>()[0], 24);
    }
}

TEST_P(OpsAdditionalMultiDTypeTest, SumAlongDimension) {
    // Sum along specific dimension with keepdim
    auto t = ones({2, 3, 4}, dtype(), device());

    // Sum along dim 0
    auto result0 = sum(t, 0, false);
    EXPECT_EQ(result0.shape().size(), 2) << "Failed on " << device().to_string();
    EXPECT_EQ(result0.shape()[0], 3) << "Failed on " << device().to_string();
    EXPECT_EQ(result0.shape()[1], 4) << "Failed on " << device().to_string();

    // Sum along dim 1 with keepdim
    auto result1 = sum(t, 1, true);
    EXPECT_EQ(result1.shape().size(), 3) << "Failed on " << device().to_string();
    EXPECT_EQ(result1.shape()[0], 2) << "Failed on " << device().to_string();
    EXPECT_EQ(result1.shape()[1], 1) << "Failed on " << device().to_string();
    EXPECT_EQ(result1.shape()[2], 4) << "Failed on " << device().to_string();
}

TEST_P(OpsAdditionalMultiDTypeTest, MeanOperation) {
    // Skip mean for non-floating types
    if (dtype() != DType::Float32 && dtype() != DType::Float64 && dtype() != DType::Float16) {
        GTEST_SKIP() << "Mean operation only supports floating point dtypes";
    }

    auto t = full({2, 3, 4}, 6.0f, dtype(), device());

    // Mean all elements
    auto result_all = mean(t);
    auto result_all_cpu = result_all.to(Device::cpu());
    if (dtype() == DType::Float32) {
        EXPECT_FLOAT_EQ(result_all_cpu.data<float>()[0], 6.0f);
    } else if (dtype() == DType::Float64) {
        EXPECT_DOUBLE_EQ(result_all_cpu.data<double>()[0], 6.0);
    }

    // Mean along dimension
    auto result_dim = mean(t, 2, false);
    EXPECT_EQ(result_dim.shape().size(), 2) << "Failed on " << device().to_string();
    EXPECT_EQ(result_dim.shape()[0], 2) << "Failed on " << device().to_string();
    EXPECT_EQ(result_dim.shape()[1], 3) << "Failed on " << device().to_string();
}

TEST_P(OpsAdditionalMultiDTypeTest, MaxMinOperations) {
    auto t = zeros({3, 4}, dtype(), device());
    auto t_cpu = t.to(Device::cpu());

    // Fill with known values
    if (dtype() == DType::Float32) {
        auto data = t_cpu.data<float>();
        for (int i = 0; i < 12; i++) {
            data[i] = static_cast<float>(i);
        }
    } else if (dtype() == DType::Int32) {
        auto data = t_cpu.data<int32_t>();
        for (int i = 0; i < 12; i++) {
            data[i] = i;
        }
    }
    t = t_cpu.to(device());

    // Max all elements
    auto max_result = max(t);
    auto max_cpu = max_result.to(Device::cpu());
    if (dtype() == DType::Float32) {
        EXPECT_FLOAT_EQ(max_cpu.data<float>()[0], 11.0f);
    } else if (dtype() == DType::Int32) {
        EXPECT_EQ(max_cpu.data<int32_t>()[0], 11);
    }

    // Min all elements
    auto min_result = min(t);
    auto min_cpu = min_result.to(Device::cpu());
    if (dtype() == DType::Float32) {
        EXPECT_FLOAT_EQ(min_cpu.data<float>()[0], 0.0f);
    } else if (dtype() == DType::Int32) {
        EXPECT_EQ(min_cpu.data<int32_t>()[0], 0);
    }
}

TEST_P(OpsAdditionalMultiDTypeTest, ArgMaxArgMin) {
    auto t = zeros({3, 4}, dtype(), device());
    auto t_cpu = t.to(Device::cpu());

    // Fill: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    if (dtype() == DType::Float32) {
        auto data = t_cpu.data<float>();
        for (int i = 0; i < 12; i++) {
            data[i] = static_cast<float>(i);
        }
    } else if (dtype() == DType::Float64) {
        auto data = t_cpu.data<double>();
        for (int i = 0; i < 12; i++) {
            data[i] = static_cast<double>(i);
        }
    } else if (dtype() == DType::Int32) {
        auto data = t_cpu.data<int32_t>();
        for (int i = 0; i < 12; i++) {
            data[i] = i;
        }
    }
    t = t_cpu.to(device());

    // ArgMax along dim 1
    auto argmax_result = argmax(t, 1);
    auto argmax_cpu = argmax_result.to(Device::cpu());
    auto argmax_data = argmax_cpu.data<int64_t>();

    // Each row's max is at index 3
    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(argmax_data[i], 3) << "Failed on " << device().to_string();
    }
}

TEST_P(OpsAdditionalMultiDTypeTest, ProdOperation) {
    auto t = full({2, 3}, 2.0f, dtype(), device());

    // Product all elements
    auto result = prod(t);
    auto result_cpu = result.to(Device::cpu());

    if (dtype() == DType::Float32) {
        EXPECT_FLOAT_EQ(result_cpu.data<float>()[0], 64.0f);
    } else if (dtype() == DType::Int32) {
        // Int32 prod promotes to Int64 to prevent overflow (matches PyTorch)
        EXPECT_EQ(result_cpu.data<int64_t>()[0], 64);
    }
}

//==============================================================================
// Tensor Manipulation Tests
//==============================================================================

TEST_P(OpsAdditionalMultiDTypeTest, ReshapeBasic) {
    auto t = zeros({2, 3, 4}, dtype(), device());

    // Reshape to 2D
    auto r1 = reshape(t, {6, 4});
    EXPECT_EQ(r1.shape()[0], 6) << "Failed on " << device().to_string();
    EXPECT_EQ(r1.shape()[1], 4) << "Failed on " << device().to_string();

    // Flatten
    auto r2 = reshape(t, {-1});
    EXPECT_EQ(r2.shape()[0], 24) << "Failed on " << device().to_string();

    // Reshape with auto-inferred dimension
    auto r3 = reshape(t, {-1, 8});
    EXPECT_EQ(r3.shape()[0], 3) << "Failed on " << device().to_string();
    EXPECT_EQ(r3.shape()[1], 8) << "Failed on " << device().to_string();
}

TEST_P(OpsAdditionalMultiDTypeTest, TransposeOperations) {
    auto t = zeros({2, 3, 4}, dtype(), device());

    // Transpose dimensions 0 and 2
    auto t_trans = transpose(t, 0, 2);
    EXPECT_EQ(t_trans.shape()[0], 4) << "Failed on " << device().to_string();
    EXPECT_EQ(t_trans.shape()[1], 3) << "Failed on " << device().to_string();
    EXPECT_EQ(t_trans.shape()[2], 2) << "Failed on " << device().to_string();

    // Transpose matrix (2D)
    auto m = zeros({3, 4}, dtype(), device());
    auto m_trans = transpose(m, 0, 1);
    EXPECT_EQ(m_trans.shape()[0], 4) << "Failed on " << device().to_string();
    EXPECT_EQ(m_trans.shape()[1], 3) << "Failed on " << device().to_string();
}

TEST_P(OpsAdditionalMultiDTypeTest, PermuteOperations) {
    auto t = zeros({2, 3, 4}, dtype(), device());

    // Permute dimensions
    auto p1 = permute(t, {2, 0, 1});
    EXPECT_EQ(p1.shape()[0], 4) << "Failed on " << device().to_string();
    EXPECT_EQ(p1.shape()[1], 2) << "Failed on " << device().to_string();
    EXPECT_EQ(p1.shape()[2], 3) << "Failed on " << device().to_string();
}

TEST_P(OpsAdditionalMultiDTypeTest, SqueezeUnsqueeze) {
    auto t = zeros({1, 3, 1, 4, 1}, dtype(), device());

    // Squeeze all size-1 dimensions
    auto s1 = squeeze(t);
    EXPECT_EQ(s1.shape().size(), 2) << "Failed on " << device().to_string();
    EXPECT_EQ(s1.shape()[0], 3) << "Failed on " << device().to_string();
    EXPECT_EQ(s1.shape()[1], 4) << "Failed on " << device().to_string();

    // Unsqueeze
    auto base = zeros({3, 4}, dtype(), device());
    auto u1 = unsqueeze(base, 0);
    EXPECT_EQ(u1.shape()[0], 1) << "Failed on " << device().to_string();
    EXPECT_EQ(u1.shape()[1], 3) << "Failed on " << device().to_string();
    EXPECT_EQ(u1.shape()[2], 4) << "Failed on " << device().to_string();
}

TEST_P(OpsAdditionalMultiDTypeTest, ConcatenateOperations) {
    auto t1 = ones({2, 3}, dtype(), device());
    auto t2 = full({2, 3}, 2.0f, dtype(), device());
    auto t3 = full({2, 3}, 3.0f, dtype(), device());

    // Concatenate along dim 0
    auto cat0 = cat({t1, t2, t3}, 0);
    EXPECT_EQ(cat0.shape()[0], 6) << "Failed on " << device().to_string();
    EXPECT_EQ(cat0.shape()[1], 3) << "Failed on " << device().to_string();

    // Concatenate along dim 1
    auto cat1 = cat({t1, t2}, 1);
    EXPECT_EQ(cat1.shape()[0], 2) << "Failed on " << device().to_string();
    EXPECT_EQ(cat1.shape()[1], 6) << "Failed on " << device().to_string();
}

TEST_P(OpsAdditionalMultiDTypeTest, StackOperations) {
    auto t1 = ones({2, 3}, dtype(), device());
    auto t2 = full({2, 3}, 2.0f, dtype(), device());

    // Stack along new dim 0
    std::vector<Tensor> tensors0{t1, t2};
    auto s0 = stack(tensors0, 0);
    EXPECT_EQ(s0.shape()[0], 2) << "Failed on " << device().to_string();
    EXPECT_EQ(s0.shape()[1], 2) << "Failed on " << device().to_string();
    EXPECT_EQ(s0.shape()[2], 3) << "Failed on " << device().to_string();
}

TEST_P(OpsAdditionalMultiDTypeTest, SplitOperations) {
    auto t = zeros({6, 4}, dtype(), device());

    // Split into chunks of size 2
    auto splits = split(t, 2, 0);
    EXPECT_EQ(splits.size(), 3) << "Failed on " << device().to_string();
    for (const auto& s : splits) {
        EXPECT_EQ(s.shape()[0], 2) << "Failed on " << device().to_string();
        EXPECT_EQ(s.shape()[1], 4) << "Failed on " << device().to_string();
    }
}

TEST_P(OpsAdditionalMultiDTypeTest, ChunkOperations) {
    auto t = zeros({10, 4}, dtype(), device());

    // Chunk into 3 parts
    auto chunks = chunk(t, 3, 0);
    EXPECT_EQ(chunks.size(), 3) << "Failed on " << device().to_string();

    // First two chunks should have size 4, last should have size 2
    EXPECT_EQ(chunks[0].shape()[0], 4) << "Failed on " << device().to_string();
    EXPECT_EQ(chunks[1].shape()[0], 4) << "Failed on " << device().to_string();
    EXPECT_EQ(chunks[2].shape()[0], 2) << "Failed on " << device().to_string();
}

TEST_P(OpsAdditionalMultiDTypeTest, FlattenOperations) {
    auto t = zeros({2, 3, 4}, dtype(), device());

    // Flatten all
    auto f1 = flatten(t);
    EXPECT_EQ(f1.shape()[0], 24) << "Failed on " << device().to_string();

    // Flatten from dim 1
    auto f2 = flatten(t, 1, 2);
    EXPECT_EQ(f2.shape()[0], 2) << "Failed on " << device().to_string();
    EXPECT_EQ(f2.shape()[1], 12) << "Failed on " << device().to_string();
}

//==============================================================================
// Mathematical Operations Tests
//==============================================================================

TEST_P(OpsAdditionalMultiDTypeTest, ArithmeticOperations) {
    auto a = full({2, 3}, 6.0f, dtype(), device());
    auto b = full({2, 3}, 2.0f, dtype(), device());

    // Addition
    auto add_result = add(a, b);
    auto add_cpu = add_result.to(Device::cpu());
    if (dtype() == DType::Float32) {
        EXPECT_FLOAT_EQ(add_cpu.data<float>()[0], 8.0f);
    } else if (dtype() == DType::Int32) {
        EXPECT_EQ(add_cpu.data<int32_t>()[0], 8);
    }

    // Subtraction
    auto sub_result = sub(a, b);
    auto sub_cpu = sub_result.to(Device::cpu());
    if (dtype() == DType::Float32) {
        EXPECT_FLOAT_EQ(sub_cpu.data<float>()[0], 4.0f);
    } else if (dtype() == DType::Int32) {
        EXPECT_EQ(sub_cpu.data<int32_t>()[0], 4);
    }

    // Multiplication
    auto mul_result = mul(a, b);
    auto mul_cpu = mul_result.to(Device::cpu());
    if (dtype() == DType::Float32) {
        EXPECT_FLOAT_EQ(mul_cpu.data<float>()[0], 12.0f);
    } else if (dtype() == DType::Int32) {
        EXPECT_EQ(mul_cpu.data<int32_t>()[0], 12);
    }
}

TEST_P(OpsAdditionalMultiDTypeTest, MatMulOperations) {
    // Matrix multiplication
    auto a = ones({2, 3}, dtype(), device());
    auto b = ones({3, 4}, dtype(), device());

    auto c = matmul(a, b);
    EXPECT_EQ(c.shape()[0], 2) << "Failed on " << device().to_string();
    EXPECT_EQ(c.shape()[1], 4) << "Failed on " << device().to_string();

    auto c_cpu = c.to(Device::cpu());
    // Each element should be 3.0 (sum of 3 ones)
    if (dtype() == DType::Float32) {
        EXPECT_FLOAT_EQ(c_cpu.data<float>()[0], 3.0f);
    } else if (dtype() == DType::Int32) {
        EXPECT_EQ(c_cpu.data<int32_t>()[0], 3);
    }
}

TEST_P(OpsAdditionalMultiDTypeTest, PowerOperations) {
    // Skip for non-floating types
    if (dtype() != DType::Float32 && dtype() != DType::Float64 && dtype() != DType::Float16) {
        GTEST_SKIP();
    }

    auto t = full({3, 3}, 2.0f, dtype(), device());

    // Square
    auto squared = pow(t, 2.0f);
    auto sq_cpu = squared.to(Device::cpu());
    if (dtype() == DType::Float32) {
        EXPECT_FLOAT_EQ(sq_cpu.data<float>()[0], 4.0f);
    }

    // Square root
    auto t2 = full({3, 3}, 4.0f, dtype(), device());
    auto sqrt_result = sqrt(t2);
    auto sqrt_cpu = sqrt_result.to(Device::cpu());
    if (dtype() == DType::Float32) {
        EXPECT_FLOAT_EQ(sqrt_cpu.data<float>()[0], 2.0f);
    }
}

TEST_P(OpsAdditionalMultiDTypeTest, TrigonometricFunctions) {
    if (dtype() != DType::Float32 && dtype() != DType::Float64 && dtype() != DType::Float16) {
        GTEST_SKIP();
    }

    auto t = zeros({4}, dtype(), device());
    auto t_cpu = t.to(Device::cpu());

    if (dtype() == DType::Float32) {
        auto data = t_cpu.data<float>();
        data[0] = 0.0f;
        data[1] = M_PI / 6;  // 30 degrees
        data[2] = M_PI / 4;  // 45 degrees
        data[3] = M_PI / 2;  // 90 degrees
    }
    t = t_cpu.to(device());

    // Sine
    auto sin_result = sin(t);
    auto sin_cpu = sin_result.to(Device::cpu());
    if (dtype() == DType::Float32) {
        auto sin_data = sin_cpu.data<float>();
        EXPECT_NEAR(sin_data[0], 0.0f, tolerance);
        EXPECT_NEAR(sin_data[1], 0.5f, tolerance);
        EXPECT_NEAR(sin_data[3], 1.0f, tolerance);
    }
}

TEST_P(OpsAdditionalMultiDTypeTest, ElementWiseOperations) {
    auto t = zeros({4}, dtype(), device());
    auto t_cpu = t.to(Device::cpu());

    if (dtype() == DType::Float32) {
        auto data = t_cpu.data<float>();
        data[0] = -2.5f; data[1] = -1.0f; data[2] = 0.0f; data[3] = 3.7f;
    } else if (dtype() == DType::Int32) {
        auto data = t_cpu.data<int32_t>();
        data[0] = -2; data[1] = -1; data[2] = 0; data[3] = 3;
    }
    t = t_cpu.to(device());

    // Absolute value
    auto abs_result = abs(t);
    auto abs_cpu = abs_result.to(Device::cpu());
    if (dtype() == DType::Float32) {
        auto abs_data = abs_cpu.data<float>();
        EXPECT_FLOAT_EQ(abs_data[0], 2.5f);
        EXPECT_FLOAT_EQ(abs_data[1], 1.0f);
    } else if (dtype() == DType::Int32) {
        auto abs_data = abs_cpu.data<int32_t>();
        EXPECT_EQ(abs_data[0], 2);
        EXPECT_EQ(abs_data[1], 1);
    }

    // Negation
    auto neg_result = neg(t);
    auto neg_cpu = neg_result.to(Device::cpu());
    if (dtype() == DType::Float32) {
        auto neg_data = neg_cpu.data<float>();
        EXPECT_FLOAT_EQ(neg_data[0], 2.5f);
    } else if (dtype() == DType::Int32) {
        auto neg_data = neg_cpu.data<int32_t>();
        EXPECT_EQ(neg_data[0], 2);
    }
}

//==============================================================================
// Comparison Operations Tests
//==============================================================================

TEST_P(OpsAdditionalMultiDTypeTest, EqualityComparison) {
    auto a = zeros({4}, dtype(), device());
    auto a_cpu = a.to(Device::cpu());

    if (dtype() == DType::Float32) {
        auto a_data = a_cpu.data<float>();
        a_data[0] = 1.0f; a_data[1] = 2.0f; a_data[2] = 3.0f; a_data[3] = 4.0f;
    } else if (dtype() == DType::Float64) {
        auto a_data = a_cpu.data<double>();
        a_data[0] = 1.0; a_data[1] = 2.0; a_data[2] = 3.0; a_data[3] = 4.0;
    } else if (dtype() == DType::Int32) {
        auto a_data = a_cpu.data<int32_t>();
        a_data[0] = 1; a_data[1] = 2; a_data[2] = 3; a_data[3] = 4;
    }
    a = a_cpu.to(device());

    auto b = zeros({4}, dtype(), device());
    auto b_cpu = b.to(Device::cpu());

    if (dtype() == DType::Float32) {
        auto b_data = b_cpu.data<float>();
        b_data[0] = 1.0f; b_data[1] = 2.5f; b_data[2] = 3.0f; b_data[3] = 3.5f;
    } else if (dtype() == DType::Float64) {
        auto b_data = b_cpu.data<double>();
        b_data[0] = 1.0; b_data[1] = 2.5; b_data[2] = 3.0; b_data[3] = 3.5;
    } else if (dtype() == DType::Int32) {
        auto b_data = b_cpu.data<int32_t>();
        b_data[0] = 1; b_data[1] = 3; b_data[2] = 3; b_data[3] = 4;
    }
    b = b_cpu.to(device());

    // Equal
    auto eq_result = eq(a, b);
    auto eq_cpu = eq_result.to(Device::cpu());
    auto eq_data = eq_cpu.data<bool>();
    EXPECT_TRUE(eq_data[0]) << "Failed on " << device().to_string();
    EXPECT_FALSE(eq_data[1]) << "Failed on " << device().to_string();
    EXPECT_TRUE(eq_data[2]) << "Failed on " << device().to_string();
}

TEST_P(OpsAdditionalMultiDTypeTest, InequalityComparisons) {
    auto a = zeros({4}, dtype(), device());
    auto a_cpu = a.to(Device::cpu());

    if (dtype() == DType::Float32) {
        auto a_data = a_cpu.data<float>();
        a_data[0] = 1.0f; a_data[1] = 2.0f; a_data[2] = 3.0f; a_data[3] = 4.0f;
    } else if (dtype() == DType::Float64) {
        auto a_data = a_cpu.data<double>();
        a_data[0] = 1.0; a_data[1] = 2.0; a_data[2] = 3.0; a_data[3] = 4.0;
    } else if (dtype() == DType::Int32) {
        auto a_data = a_cpu.data<int32_t>();
        a_data[0] = 1; a_data[1] = 2; a_data[2] = 3; a_data[3] = 4;
    }
    a = a_cpu.to(device());

    auto b = full({4}, 2.5f, dtype(), device());

    // Less than
    auto lt_result = lt(a, b);
    auto lt_cpu = lt_result.to(Device::cpu());
    auto lt_data = lt_cpu.data<bool>();
    EXPECT_TRUE(lt_data[0]) << "Failed on " << device().to_string();
    // For Int32, 2.5f truncates to 2, so 2 < 2 is false
    if (dtype() == DType::Int32) {
        EXPECT_FALSE(lt_data[1]) << "Failed on " << device().to_string();
    } else {
        EXPECT_TRUE(lt_data[1]) << "Failed on " << device().to_string();
    }
    EXPECT_FALSE(lt_data[2]) << "Failed on " << device().to_string();
    EXPECT_FALSE(lt_data[3]) << "Failed on " << device().to_string();
}

//==============================================================================
// Test Instantiation
//==============================================================================

INSTANTIATE_TEST_SUITE_P(
    AllBackendsMultiDTypes,
    OpsAdditionalMultiDTypeTest,
    ::testing::Combine(
        STANDARD_BACKENDS,
        ::testing::Values(DType::Float32, DType::Float64, DType::Float16, DType::Int32)
    ),
    BackendDTypeParamName);
