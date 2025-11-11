/**
 * @file test_advanced_ops_multidtype.cpp
 * @brief DType-parameterized tests for advanced tensor operations
 *
 * Tests advanced operations (topk, sort, unique, cumsum, cumprod, expand)
 * across multiple dtypes to ensure correct behavior.
 *
 * OPERATION-SPECIFIC DTYPE SUPPORT:
 * - Expand: Float32, Float64, Float16, Int32, Int64 (broadcasting)
 * - TopK: Float32, Float64, Float16, Int32, Int64 (comparisons)
 * - Sort: Float32, Float64, Float16, Int32, Int64 (comparisons)
 * - Unique: Float32, Float64, Int32, Int64, Bool (equality)
 * - Cumsum: Float32, Float64, Int32, Int64 (accumulation)
 * - Cumprod: Float32, Float64 (product can overflow for integers)
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/advanced.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <cmath>
#include <type_traits>

using namespace tenzor;

// ============================================================================
// DType Parameter Structure
// ============================================================================

struct DTypeParam {
    DType dtype;
    std::string name;

    std::string ToString() const { return name; }
};

// ============================================================================
// Helper Templates for Type-Specific Operations
// ============================================================================

template<typename T>
struct TypedVerifier {
    static void expectEq(T actual, T expected, const std::string& context = "") {
        if constexpr (std::is_floating_point_v<T>) {
            EXPECT_NEAR(actual, expected, static_cast<T>(1e-4))
                << context;
        } else {
            EXPECT_EQ(actual, expected) << context;
        }
    }

    static void expectNear(T actual, T expected, T tolerance, const std::string& context = "") {
        if constexpr (std::is_floating_point_v<T>) {
            EXPECT_NEAR(actual, expected, tolerance) << context;
        } else {
            EXPECT_EQ(actual, expected) << context;
        }
    }
};

// ============================================================================
// Expand Tests (Broadcasting Operations)
// ============================================================================

class ExpandMultiDTypeTest : public ::testing::TestWithParam<DTypeParam> {
protected:
    Device cpu = Device::cpu();
    DType dtype;

    void SetUp() override {
        tenzor::initialize();
        dtype = GetParam().dtype;
    }

    template<typename T>
    void testExpandBroadcast() {
        auto t = ones({1, 3}, dtype, cpu);
        auto expanded = expand(t, {4, 3});

        EXPECT_EQ(expanded.shape()[0], 4);
        EXPECT_EQ(expanded.shape()[1], 3);

        // Verify all values are 1
        auto data = expanded.contiguous().data<T>();
        for (int64_t i = 0; i < 12; ++i) {
            TypedVerifier<T>::expectEq(data[i], static_cast<T>(1),
                "Expand broadcast value mismatch at index " + std::to_string(i));
        }
    }
};

TEST_P(ExpandMultiDTypeTest, ExpandBroadcast) {
    auto param = GetParam();

    if (dtype == DType::Float32) {
        testExpandBroadcast<float>();
    } else if (dtype == DType::Float64) {
        testExpandBroadcast<double>();
    } else if (dtype == DType::Int32) {
        testExpandBroadcast<int32_t>();
    } else if (dtype == DType::Int64) {
        testExpandBroadcast<int64_t>();
    }
}

TEST_P(ExpandMultiDTypeTest, ExpandInvalidShape) {
    auto t = ones({2, 3}, dtype, cpu);

    // Cannot expand dimension 2 to 4 (not singleton)
    EXPECT_THROW(expand(t, {4, 3}), std::runtime_error);
}

std::vector<DTypeParam> GetExpandDTypes() {
    return {
        {DType::Float32, "float32"},
        {DType::Float64, "float64"},
        {DType::Int32, "int32"},
        {DType::Int64, "int64"},
    };
}

INSTANTIATE_TEST_SUITE_P(
    AllDTypes,
    ExpandMultiDTypeTest,
    ::testing::ValuesIn(GetExpandDTypes()),
    [](const ::testing::TestParamInfo<DTypeParam>& info) {
        return info.param.ToString();
    }
);

// ============================================================================
// TopK Tests (Comparison Operations)
// ============================================================================

class TopKMultiDTypeTest : public ::testing::TestWithParam<DTypeParam> {
protected:
    Device cpu = Device::cpu();
    DType dtype;

    void SetUp() override {
        tenzor::initialize();
        dtype = GetParam().dtype;
    }

    template<typename T>
    void testTopKLargest() {
        auto t = Tensor({5}, dtype, cpu);
        auto data = t.data<T>();
        data[0] = static_cast<T>(3);
        data[1] = static_cast<T>(1);
        data[2] = static_cast<T>(4);
        data[3] = static_cast<T>(2);
        data[4] = static_cast<T>(5);

        auto [values, indices] = topk(t, 3, 0, true, true);

        EXPECT_EQ(values.shape()[0], 3);
        EXPECT_EQ(indices.shape()[0], 3);

        auto vals = values.data<T>();
        auto idxs = indices.data<int64_t>();

        TypedVerifier<T>::expectEq(vals[0], static_cast<T>(5), "TopK value 0");
        TypedVerifier<T>::expectEq(vals[1], static_cast<T>(4), "TopK value 1");
        TypedVerifier<T>::expectEq(vals[2], static_cast<T>(3), "TopK value 2");

        EXPECT_EQ(idxs[0], 4);
        EXPECT_EQ(idxs[1], 2);
        EXPECT_EQ(idxs[2], 0);
    }

    template<typename T>
    void testTopKSmallest() {
        auto t = Tensor({5}, dtype, cpu);
        auto data = t.data<T>();
        data[0] = static_cast<T>(3);
        data[1] = static_cast<T>(1);
        data[2] = static_cast<T>(4);
        data[3] = static_cast<T>(2);
        data[4] = static_cast<T>(5);

        auto [values, indices] = topk(t, 2, 0, false, true);

        auto vals = values.data<T>();
        TypedVerifier<T>::expectEq(vals[0], static_cast<T>(1), "TopK smallest value 0");
        TypedVerifier<T>::expectEq(vals[1], static_cast<T>(2), "TopK smallest value 1");
    }
};

TEST_P(TopKMultiDTypeTest, TopKLargest) {
    auto param = GetParam();

    if (dtype == DType::Float32) {
        testTopKLargest<float>();
    } else if (dtype == DType::Float64) {
        testTopKLargest<double>();
    } else if (dtype == DType::Int32) {
        testTopKLargest<int32_t>();
    } else if (dtype == DType::Int64) {
        testTopKLargest<int64_t>();
    }
}

TEST_P(TopKMultiDTypeTest, TopKSmallest) {
    auto param = GetParam();

    if (dtype == DType::Float32) {
        testTopKSmallest<float>();
    } else if (dtype == DType::Float64) {
        testTopKSmallest<double>();
    } else if (dtype == DType::Int32) {
        testTopKSmallest<int32_t>();
    } else if (dtype == DType::Int64) {
        testTopKSmallest<int64_t>();
    }
}

std::vector<DTypeParam> GetTopKDTypes() {
    return {
        {DType::Float32, "float32"},
        {DType::Float64, "float64"},
        {DType::Int32, "int32"},
        {DType::Int64, "int64"},
    };
}

INSTANTIATE_TEST_SUITE_P(
    AllDTypes,
    TopKMultiDTypeTest,
    ::testing::ValuesIn(GetTopKDTypes()),
    [](const ::testing::TestParamInfo<DTypeParam>& info) {
        return info.param.ToString();
    }
);

// ============================================================================
// Sort Tests (Comparison Operations)
// ============================================================================

class SortMultiDTypeTest : public ::testing::TestWithParam<DTypeParam> {
protected:
    Device cpu = Device::cpu();
    DType dtype;

    void SetUp() override {
        tenzor::initialize();
        dtype = GetParam().dtype;
    }

    template<typename T>
    void testSortAscending() {
        auto t = Tensor({4}, dtype, cpu);
        auto data = t.data<T>();
        data[0] = static_cast<T>(3);
        data[1] = static_cast<T>(1);
        data[2] = static_cast<T>(4);
        data[3] = static_cast<T>(2);

        auto [sorted, indices] = sort(t, 0, false);

        auto vals = sorted.data<T>();
        TypedVerifier<T>::expectEq(vals[0], static_cast<T>(1), "Sort ascending value 0");
        TypedVerifier<T>::expectEq(vals[1], static_cast<T>(2), "Sort ascending value 1");
        TypedVerifier<T>::expectEq(vals[2], static_cast<T>(3), "Sort ascending value 2");
        TypedVerifier<T>::expectEq(vals[3], static_cast<T>(4), "Sort ascending value 3");

        auto idxs = indices.data<int64_t>();
        EXPECT_EQ(idxs[0], 1);
        EXPECT_EQ(idxs[1], 3);
        EXPECT_EQ(idxs[2], 0);
        EXPECT_EQ(idxs[3], 2);
    }

    template<typename T>
    void testSortDescending() {
        auto t = Tensor({4}, dtype, cpu);
        auto data = t.data<T>();
        data[0] = static_cast<T>(1);
        data[1] = static_cast<T>(4);
        data[2] = static_cast<T>(2);
        data[3] = static_cast<T>(3);

        auto [sorted, indices] = sort(t, 0, true);

        auto vals = sorted.data<T>();
        TypedVerifier<T>::expectEq(vals[0], static_cast<T>(4), "Sort descending value 0");
        TypedVerifier<T>::expectEq(vals[1], static_cast<T>(3), "Sort descending value 1");
        TypedVerifier<T>::expectEq(vals[2], static_cast<T>(2), "Sort descending value 2");
        TypedVerifier<T>::expectEq(vals[3], static_cast<T>(1), "Sort descending value 3");
    }
};

TEST_P(SortMultiDTypeTest, SortAscending) {
    auto param = GetParam();

    if (dtype == DType::Float32) {
        testSortAscending<float>();
    } else if (dtype == DType::Float64) {
        testSortAscending<double>();
    } else if (dtype == DType::Int32) {
        testSortAscending<int32_t>();
    } else if (dtype == DType::Int64) {
        testSortAscending<int64_t>();
    }
}

TEST_P(SortMultiDTypeTest, SortDescending) {
    auto param = GetParam();

    if (dtype == DType::Float32) {
        testSortDescending<float>();
    } else if (dtype == DType::Float64) {
        testSortDescending<double>();
    } else if (dtype == DType::Int32) {
        testSortDescending<int32_t>();
    } else if (dtype == DType::Int64) {
        testSortDescending<int64_t>();
    }
}

std::vector<DTypeParam> GetSortDTypes() {
    return {
        {DType::Float32, "float32"},
        {DType::Float64, "float64"},
        {DType::Int32, "int32"},
        {DType::Int64, "int64"},
    };
}

INSTANTIATE_TEST_SUITE_P(
    AllDTypes,
    SortMultiDTypeTest,
    ::testing::ValuesIn(GetSortDTypes()),
    [](const ::testing::TestParamInfo<DTypeParam>& info) {
        return info.param.ToString();
    }
);

// ============================================================================
// Unique Tests (Equality Operations)
// ============================================================================

class UniqueMultiDTypeTest : public ::testing::TestWithParam<DTypeParam> {
protected:
    Device cpu = Device::cpu();
    DType dtype;

    void SetUp() override {
        tenzor::initialize();
        dtype = GetParam().dtype;
    }

    template<typename T>
    void testUniqueBasic() {
        auto t = Tensor({6}, dtype, cpu);
        auto data = t.data<T>();
        data[0] = static_cast<T>(1);
        data[1] = static_cast<T>(2);
        data[2] = static_cast<T>(1);
        data[3] = static_cast<T>(3);
        data[4] = static_cast<T>(2);
        data[5] = static_cast<T>(3);

        auto [unique_vals, inverse, counts] = unique(t, true, false, false);

        EXPECT_EQ(unique_vals.numel(), 3);

        auto vals = unique_vals.data<T>();
        TypedVerifier<T>::expectEq(vals[0], static_cast<T>(1), "Unique value 0");
        TypedVerifier<T>::expectEq(vals[1], static_cast<T>(2), "Unique value 1");
        TypedVerifier<T>::expectEq(vals[2], static_cast<T>(3), "Unique value 2");
    }

    template<typename T>
    void testUniqueWithCounts() {
        auto t = Tensor({5}, dtype, cpu);
        auto data = t.data<T>();
        data[0] = static_cast<T>(1);
        data[1] = static_cast<T>(2);
        data[2] = static_cast<T>(1);
        data[3] = static_cast<T>(1);
        data[4] = static_cast<T>(2);

        auto [unique_vals, inverse, counts_tensor] = unique(t, true, false, true);

        EXPECT_EQ(unique_vals.numel(), 2);

        auto counts_data = counts_tensor.data<int64_t>();
        EXPECT_EQ(counts_data[0], 3);  // 1 appears 3 times
        EXPECT_EQ(counts_data[1], 2);  // 2 appears 2 times
    }
};

TEST_P(UniqueMultiDTypeTest, UniqueBasic) {
    auto param = GetParam();

    if (dtype == DType::Float32) {
        testUniqueBasic<float>();
    } else if (dtype == DType::Float64) {
        testUniqueBasic<double>();
    } else if (dtype == DType::Int32) {
        testUniqueBasic<int32_t>();
    } else if (dtype == DType::Int64) {
        testUniqueBasic<int64_t>();
    } else if (dtype == DType::Bool) {
        testUniqueBasic<bool>();
    }
}

TEST_P(UniqueMultiDTypeTest, UniqueWithCounts) {
    auto param = GetParam();

    if (dtype == DType::Float32) {
        testUniqueWithCounts<float>();
    } else if (dtype == DType::Float64) {
        testUniqueWithCounts<double>();
    } else if (dtype == DType::Int32) {
        testUniqueWithCounts<int32_t>();
    } else if (dtype == DType::Int64) {
        testUniqueWithCounts<int64_t>();
    } else if (dtype == DType::Bool) {
        testUniqueWithCounts<bool>();
    }
}

std::vector<DTypeParam> GetUniqueDTypes() {
    return {
        {DType::Float32, "float32"},
        {DType::Float64, "float64"},
        {DType::Int32, "int32"},
        {DType::Int64, "int64"},
        {DType::Bool, "bool"},
    };
}

INSTANTIATE_TEST_SUITE_P(
    AllDTypes,
    UniqueMultiDTypeTest,
    ::testing::ValuesIn(GetUniqueDTypes()),
    [](const ::testing::TestParamInfo<DTypeParam>& info) {
        return info.param.ToString();
    }
);

// ============================================================================
// Cumsum Tests (Accumulation Operations)
// ============================================================================

class CumsumMultiDTypeTest : public ::testing::TestWithParam<DTypeParam> {
protected:
    Device cpu = Device::cpu();
    DType dtype;

    void SetUp() override {
        tenzor::initialize();
        dtype = GetParam().dtype;
    }

    template<typename T>
    void testCumsumBasic() {
        auto t = Tensor({4}, dtype, cpu);
        auto data = t.data<T>();
        data[0] = static_cast<T>(1);
        data[1] = static_cast<T>(2);
        data[2] = static_cast<T>(3);
        data[3] = static_cast<T>(4);

        auto result = cumsum(t, 0);

        auto res_data = result.data<T>();
        TypedVerifier<T>::expectEq(res_data[0], static_cast<T>(1), "Cumsum value 0");
        TypedVerifier<T>::expectEq(res_data[1], static_cast<T>(3), "Cumsum value 1");
        TypedVerifier<T>::expectEq(res_data[2], static_cast<T>(6), "Cumsum value 2");
        TypedVerifier<T>::expectEq(res_data[3], static_cast<T>(10), "Cumsum value 3");
    }

    template<typename T>
    void testCumsum2D() {
        auto t = Tensor({2, 3}, dtype, cpu);
        auto data = t.data<T>();
        for (int i = 0; i < 6; ++i) {
            data[i] = static_cast<T>(i + 1);
        }

        auto result = cumsum(t, 1);

        auto res_data = result.data<T>();
        TypedVerifier<T>::expectEq(res_data[0], static_cast<T>(1), "Cumsum 2D [0,0]");
        TypedVerifier<T>::expectEq(res_data[1], static_cast<T>(3), "Cumsum 2D [0,1]");
        TypedVerifier<T>::expectEq(res_data[2], static_cast<T>(6), "Cumsum 2D [0,2]");
        TypedVerifier<T>::expectEq(res_data[3], static_cast<T>(4), "Cumsum 2D [1,0]");
        TypedVerifier<T>::expectEq(res_data[4], static_cast<T>(9), "Cumsum 2D [1,1]");
        TypedVerifier<T>::expectEq(res_data[5], static_cast<T>(15), "Cumsum 2D [1,2]");
    }
};

TEST_P(CumsumMultiDTypeTest, CumsumBasic) {
    auto param = GetParam();

    if (dtype == DType::Float32) {
        testCumsumBasic<float>();
    } else if (dtype == DType::Float64) {
        testCumsumBasic<double>();
    } else if (dtype == DType::Int32) {
        testCumsumBasic<int32_t>();
    } else if (dtype == DType::Int64) {
        testCumsumBasic<int64_t>();
    }
}

TEST_P(CumsumMultiDTypeTest, Cumsum2D) {
    auto param = GetParam();

    if (dtype == DType::Float32) {
        testCumsum2D<float>();
    } else if (dtype == DType::Float64) {
        testCumsum2D<double>();
    } else if (dtype == DType::Int32) {
        testCumsum2D<int32_t>();
    } else if (dtype == DType::Int64) {
        testCumsum2D<int64_t>();
    }
}

std::vector<DTypeParam> GetCumsumDTypes() {
    return {
        {DType::Float32, "float32"},
        {DType::Float64, "float64"},
        {DType::Int32, "int32"},
        {DType::Int64, "int64"},
    };
}

INSTANTIATE_TEST_SUITE_P(
    AllDTypes,
    CumsumMultiDTypeTest,
    ::testing::ValuesIn(GetCumsumDTypes()),
    [](const ::testing::TestParamInfo<DTypeParam>& info) {
        return info.param.ToString();
    }
);

// ============================================================================
// Cumprod Tests (Product Accumulation - Float Only)
// ============================================================================

class CumprodMultiDTypeTest : public ::testing::TestWithParam<DTypeParam> {
protected:
    Device cpu = Device::cpu();
    DType dtype;

    void SetUp() override {
        tenzor::initialize();
        dtype = GetParam().dtype;
    }

    template<typename T>
    void testCumprodBasic() {
        auto t = Tensor({4}, dtype, cpu);
        auto data = t.data<T>();
        data[0] = static_cast<T>(2);
        data[1] = static_cast<T>(3);
        data[2] = static_cast<T>(4);
        data[3] = static_cast<T>(5);

        auto result = cumprod(t, 0);

        auto res_data = result.data<T>();

        // Use appropriate tolerance for floating point
        T tol = std::is_floating_point_v<T> ? static_cast<T>(1e-4) : static_cast<T>(0);

        TypedVerifier<T>::expectNear(res_data[0], static_cast<T>(2), tol, "Cumprod value 0");
        TypedVerifier<T>::expectNear(res_data[1], static_cast<T>(6), tol, "Cumprod value 1");
        TypedVerifier<T>::expectNear(res_data[2], static_cast<T>(24), tol, "Cumprod value 2");
        TypedVerifier<T>::expectNear(res_data[3], static_cast<T>(120), tol, "Cumprod value 3");
    }
};

TEST_P(CumprodMultiDTypeTest, CumprodBasic) {
    auto param = GetParam();

    if (dtype == DType::Float32) {
        testCumprodBasic<float>();
    } else if (dtype == DType::Float64) {
        testCumprodBasic<double>();
    }
}

std::vector<DTypeParam> GetCumprodDTypes() {
    // Only float types - integer cumprod can easily overflow
    return {
        {DType::Float32, "float32"},
        {DType::Float64, "float64"},
    };
}

INSTANTIATE_TEST_SUITE_P(
    AllDTypes,
    CumprodMultiDTypeTest,
    ::testing::ValuesIn(GetCumprodDTypes()),
    [](const ::testing::TestParamInfo<DTypeParam>& info) {
        return info.param.ToString();
    }
);

/*
 * COVERAGE SUMMARY:
 *
 * Expand:    4 dtypes × 2 tests = 8 test scenarios
 * TopK:      4 dtypes × 2 tests = 8 test scenarios
 * Sort:      4 dtypes × 2 tests = 8 test scenarios
 * Unique:    5 dtypes × 2 tests = 10 test scenarios (includes Bool)
 * Cumsum:    4 dtypes × 2 tests = 8 test scenarios
 * Cumprod:   2 dtypes × 1 test = 2 test scenarios (float only)
 *
 * Total: 44 test scenarios from 11 test cases
 *
 * DTYPE SELECTION RATIONALE:
 * - Expand: All numeric types (broadcasting is type-agnostic)
 * - TopK/Sort: All numeric types (need comparison operators)
 * - Unique: All types including Bool (only needs equality)
 * - Cumsum: All numeric types (accumulation is well-defined)
 * - Cumprod: Float only (integer overflow risk: 5! = 120, 10! = 3628800)
 *
 * EXCLUDED DTYPES:
 * - Float16: Not tested (requires special handling, limited precision)
 * - Int8/Int16: Not tested (would overflow very quickly in cumulative ops)
 * - Integers in Cumprod: Overflow risk (120 fits in int32, but grows fast)
 */
