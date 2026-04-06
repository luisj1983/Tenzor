/**
 * @file test_advanced_ops_multidtype.cpp
 * @brief Multi-backend, DType-parameterized tests for advanced tensor operations
 *
 * Tests advanced operations (topk, sort, unique, cumsum, cumprod, expand)
 * across multiple backends and dtypes to ensure correct behavior.
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
using namespace tenzor::testing;

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
// Helper to create a tensor with specific values on the test device
// ============================================================================

template<typename T>
Tensor make_tensor(std::initializer_list<int64_t> shape, DType dtype, Device target_device, std::vector<T> values) {
    auto t = Tensor(shape, dtype, Device::cpu());
    auto data = t.data<T>();
    for (size_t i = 0; i < values.size(); ++i) {
        data[i] = values[i];
    }
    return (target_device.type == Device::Type::CPU) ? t : t.to(target_device);
}

// ============================================================================
// Expand Tests (Broadcasting Operations)
// ============================================================================

class ExpandMultiDTypeTest : public BackendTest {
protected:
    template<typename T>
    void testExpandBroadcast(DType dtype) {
        auto t = ones({1, 3}, dtype, device);
        auto expanded = expand(t, {4, 3});

        EXPECT_EQ(expanded.shape()[0], 4);
        EXPECT_EQ(expanded.shape()[1], 3);

        auto cpu_expanded = expanded.contiguous().to(Device::cpu());
        auto data = cpu_expanded.template data<T>();
        for (int64_t i = 0; i < 12; ++i) {
            TypedVerifier<T>::expectEq(data[i], static_cast<T>(1),
                "Expand broadcast value mismatch at index " + std::to_string(i));
        }
    }
};

TEST_P(ExpandMultiDTypeTest, ExpandBroadcastFloat32) {
    testExpandBroadcast<float>(DType::Float32);
}

TEST_P(ExpandMultiDTypeTest, ExpandBroadcastFloat64) {
    testExpandBroadcast<double>(DType::Float64);
}

TEST_P(ExpandMultiDTypeTest, ExpandBroadcastInt32) {
    testExpandBroadcast<int32_t>(DType::Int32);
}

TEST_P(ExpandMultiDTypeTest, ExpandBroadcastInt64) {
    testExpandBroadcast<int64_t>(DType::Int64);
}

TEST_P(ExpandMultiDTypeTest, ExpandInvalidShape) {
    auto t = ones({2, 3}, DType::Float32, device);
    EXPECT_THROW(expand(t, {4, 3}), std::runtime_error);
}

INSTANTIATE_BACKEND_TESTS(ExpandMultiDTypeTest);

// ============================================================================
// TopK Tests (Comparison Operations)
// ============================================================================

class TopKMultiDTypeTest : public BackendTest {
protected:
    template<typename T>
    void testTopKLargest(DType dtype) {
        auto t = make_tensor<T>({5}, dtype, device,
            {static_cast<T>(3), static_cast<T>(1), static_cast<T>(4), static_cast<T>(2), static_cast<T>(5)});

        auto [values, indices] = topk(t, 3, 0, true, true);

        EXPECT_EQ(values.shape()[0], 3);
        EXPECT_EQ(indices.shape()[0], 3);

        auto vals = values.to(Device::cpu()).template data<T>();
        auto idxs = indices.to(Device::cpu()).template data<int64_t>();

        TypedVerifier<T>::expectEq(vals[0], static_cast<T>(5), "TopK value 0");
        TypedVerifier<T>::expectEq(vals[1], static_cast<T>(4), "TopK value 1");
        TypedVerifier<T>::expectEq(vals[2], static_cast<T>(3), "TopK value 2");

        EXPECT_EQ(idxs[0], 4);
        EXPECT_EQ(idxs[1], 2);
        EXPECT_EQ(idxs[2], 0);
    }

    template<typename T>
    void testTopKSmallest(DType dtype) {
        auto t = make_tensor<T>({5}, dtype, device,
            {static_cast<T>(3), static_cast<T>(1), static_cast<T>(4), static_cast<T>(2), static_cast<T>(5)});

        auto [values, indices] = topk(t, 2, 0, false, true);

        auto vals = values.to(Device::cpu()).template data<T>();
        TypedVerifier<T>::expectEq(vals[0], static_cast<T>(1), "TopK smallest value 0");
        TypedVerifier<T>::expectEq(vals[1], static_cast<T>(2), "TopK smallest value 1");
    }
};

TEST_P(TopKMultiDTypeTest, TopKLargestFloat32) { testTopKLargest<float>(DType::Float32); }
TEST_P(TopKMultiDTypeTest, TopKLargestFloat64) { testTopKLargest<double>(DType::Float64); }
TEST_P(TopKMultiDTypeTest, TopKLargestInt32) { testTopKLargest<int32_t>(DType::Int32); }
TEST_P(TopKMultiDTypeTest, TopKLargestInt64) { testTopKLargest<int64_t>(DType::Int64); }
TEST_P(TopKMultiDTypeTest, TopKSmallestFloat32) { testTopKSmallest<float>(DType::Float32); }
TEST_P(TopKMultiDTypeTest, TopKSmallestFloat64) { testTopKSmallest<double>(DType::Float64); }
TEST_P(TopKMultiDTypeTest, TopKSmallestInt32) { testTopKSmallest<int32_t>(DType::Int32); }
TEST_P(TopKMultiDTypeTest, TopKSmallestInt64) { testTopKSmallest<int64_t>(DType::Int64); }

INSTANTIATE_BACKEND_TESTS(TopKMultiDTypeTest);

// ============================================================================
// Sort Tests (Comparison Operations)
// ============================================================================

class SortMultiDTypeTest : public BackendTest {
protected:
    template<typename T>
    void testSortAscending(DType dtype) {
        auto t = make_tensor<T>({4}, dtype, device,
            {static_cast<T>(3), static_cast<T>(1), static_cast<T>(4), static_cast<T>(2)});

        auto [sorted, indices] = sort(t, 0, false);

        auto vals = sorted.to(Device::cpu()).template data<T>();
        TypedVerifier<T>::expectEq(vals[0], static_cast<T>(1), "Sort ascending value 0");
        TypedVerifier<T>::expectEq(vals[1], static_cast<T>(2), "Sort ascending value 1");
        TypedVerifier<T>::expectEq(vals[2], static_cast<T>(3), "Sort ascending value 2");
        TypedVerifier<T>::expectEq(vals[3], static_cast<T>(4), "Sort ascending value 3");

        auto idxs = indices.to(Device::cpu()).template data<int64_t>();
        EXPECT_EQ(idxs[0], 1);
        EXPECT_EQ(idxs[1], 3);
        EXPECT_EQ(idxs[2], 0);
        EXPECT_EQ(idxs[3], 2);
    }

    template<typename T>
    void testSortDescending(DType dtype) {
        auto t = make_tensor<T>({4}, dtype, device,
            {static_cast<T>(1), static_cast<T>(4), static_cast<T>(2), static_cast<T>(3)});

        auto [sorted, indices] = sort(t, 0, true);

        auto vals = sorted.to(Device::cpu()).template data<T>();
        TypedVerifier<T>::expectEq(vals[0], static_cast<T>(4), "Sort descending value 0");
        TypedVerifier<T>::expectEq(vals[1], static_cast<T>(3), "Sort descending value 1");
        TypedVerifier<T>::expectEq(vals[2], static_cast<T>(2), "Sort descending value 2");
        TypedVerifier<T>::expectEq(vals[3], static_cast<T>(1), "Sort descending value 3");
    }
};

TEST_P(SortMultiDTypeTest, SortAscendingFloat32) { testSortAscending<float>(DType::Float32); }
TEST_P(SortMultiDTypeTest, SortAscendingFloat64) { testSortAscending<double>(DType::Float64); }
TEST_P(SortMultiDTypeTest, SortAscendingInt32) { testSortAscending<int32_t>(DType::Int32); }
TEST_P(SortMultiDTypeTest, SortAscendingInt64) { testSortAscending<int64_t>(DType::Int64); }
TEST_P(SortMultiDTypeTest, SortDescendingFloat32) { testSortDescending<float>(DType::Float32); }
TEST_P(SortMultiDTypeTest, SortDescendingFloat64) { testSortDescending<double>(DType::Float64); }
TEST_P(SortMultiDTypeTest, SortDescendingInt32) { testSortDescending<int32_t>(DType::Int32); }
TEST_P(SortMultiDTypeTest, SortDescendingInt64) { testSortDescending<int64_t>(DType::Int64); }

INSTANTIATE_BACKEND_TESTS(SortMultiDTypeTest);

// ============================================================================
// Unique Tests (Equality Operations)
// ============================================================================

class UniqueMultiDTypeTest : public BackendTest {
protected:
    template<typename T>
    void testUniqueBasic(DType dtype) {
        auto t = make_tensor<T>({6}, dtype, device,
            {static_cast<T>(1), static_cast<T>(2), static_cast<T>(1),
             static_cast<T>(3), static_cast<T>(2), static_cast<T>(3)});

        auto [unique_vals, inverse, counts] = unique(t, true, false, false);

        EXPECT_EQ(unique_vals.numel(), 3);

        auto vals = unique_vals.to(Device::cpu()).template data<T>();
        TypedVerifier<T>::expectEq(vals[0], static_cast<T>(1), "Unique value 0");
        TypedVerifier<T>::expectEq(vals[1], static_cast<T>(2), "Unique value 1");
        TypedVerifier<T>::expectEq(vals[2], static_cast<T>(3), "Unique value 2");
    }

    void testUniqueBasicBool() {
        // Bool tensors: create on CPU, move to device
        auto t_cpu = Tensor({6}, DType::Bool, Device::cpu());
        auto data = t_cpu.data<bool>();
        data[0] = false; data[1] = true; data[2] = false;
        data[3] = true; data[4] = true; data[5] = false;
        auto t = (device.type == Device::Type::CPU) ? t_cpu : t_cpu.to(device);

        auto [unique_vals, inverse, counts] = unique(t, true, false, false);

        EXPECT_EQ(unique_vals.numel(), 2);

        auto vals = unique_vals.to(Device::cpu()).template data<bool>();
        EXPECT_EQ(vals[0], false);
        EXPECT_EQ(vals[1], true);
    }

    template<typename T>
    void testUniqueWithCounts(DType dtype) {
        auto t = make_tensor<T>({5}, dtype, device,
            {static_cast<T>(1), static_cast<T>(2), static_cast<T>(1),
             static_cast<T>(1), static_cast<T>(2)});

        auto [unique_vals, inverse, counts_tensor] = unique(t, true, false, true);

        EXPECT_EQ(unique_vals.numel(), 2);

        auto counts_data = counts_tensor.to(Device::cpu()).template data<int64_t>();
        EXPECT_EQ(counts_data[0], 3);
        EXPECT_EQ(counts_data[1], 2);
    }

    void testUniqueWithCountsBool() {
        auto t_cpu = Tensor({5}, DType::Bool, Device::cpu());
        auto data = t_cpu.data<bool>();
        data[0] = false; data[1] = true; data[2] = false;
        data[3] = false; data[4] = true;
        auto t = (device.type == Device::Type::CPU) ? t_cpu : t_cpu.to(device);

        auto [unique_vals, inverse, counts_tensor] = unique(t, true, false, true);

        EXPECT_EQ(unique_vals.numel(), 2);

        auto counts_data = counts_tensor.to(Device::cpu()).template data<int64_t>();
        EXPECT_EQ(counts_data[0], 3);
        EXPECT_EQ(counts_data[1], 2);
    }
};

TEST_P(UniqueMultiDTypeTest, UniqueBasicFloat32) { testUniqueBasic<float>(DType::Float32); }
TEST_P(UniqueMultiDTypeTest, UniqueBasicFloat64) { testUniqueBasic<double>(DType::Float64); }
TEST_P(UniqueMultiDTypeTest, UniqueBasicInt32) { testUniqueBasic<int32_t>(DType::Int32); }
TEST_P(UniqueMultiDTypeTest, UniqueBasicInt64) { testUniqueBasic<int64_t>(DType::Int64); }
TEST_P(UniqueMultiDTypeTest, UniqueBasicBool) { testUniqueBasicBool(); }
TEST_P(UniqueMultiDTypeTest, UniqueWithCountsFloat32) { testUniqueWithCounts<float>(DType::Float32); }
TEST_P(UniqueMultiDTypeTest, UniqueWithCountsFloat64) { testUniqueWithCounts<double>(DType::Float64); }
TEST_P(UniqueMultiDTypeTest, UniqueWithCountsInt32) { testUniqueWithCounts<int32_t>(DType::Int32); }
TEST_P(UniqueMultiDTypeTest, UniqueWithCountsInt64) { testUniqueWithCounts<int64_t>(DType::Int64); }
TEST_P(UniqueMultiDTypeTest, UniqueWithCountsBool) { testUniqueWithCountsBool(); }

INSTANTIATE_BACKEND_TESTS(UniqueMultiDTypeTest);

// ============================================================================
// Cumsum Tests (Accumulation Operations)
// ============================================================================

class CumsumMultiDTypeTest : public BackendTest {
protected:
    template<typename T>
    void testCumsumBasic(DType dtype) {
        auto t = make_tensor<T>({4}, dtype, device,
            {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4)});

        auto result = cumsum(t, 0);

        auto res_data = result.to(Device::cpu()).template data<T>();
        TypedVerifier<T>::expectEq(res_data[0], static_cast<T>(1), "Cumsum value 0");
        TypedVerifier<T>::expectEq(res_data[1], static_cast<T>(3), "Cumsum value 1");
        TypedVerifier<T>::expectEq(res_data[2], static_cast<T>(6), "Cumsum value 2");
        TypedVerifier<T>::expectEq(res_data[3], static_cast<T>(10), "Cumsum value 3");
    }

    template<typename T>
    void testCumsum2D(DType dtype) {
        std::vector<T> vals;
        for (int i = 0; i < 6; ++i) vals.push_back(static_cast<T>(i + 1));
        auto t = make_tensor<T>({2, 3}, dtype, device, vals);

        auto result = cumsum(t, 1);

        auto res_data = result.to(Device::cpu()).template data<T>();
        TypedVerifier<T>::expectEq(res_data[0], static_cast<T>(1), "Cumsum 2D [0,0]");
        TypedVerifier<T>::expectEq(res_data[1], static_cast<T>(3), "Cumsum 2D [0,1]");
        TypedVerifier<T>::expectEq(res_data[2], static_cast<T>(6), "Cumsum 2D [0,2]");
        TypedVerifier<T>::expectEq(res_data[3], static_cast<T>(4), "Cumsum 2D [1,0]");
        TypedVerifier<T>::expectEq(res_data[4], static_cast<T>(9), "Cumsum 2D [1,1]");
        TypedVerifier<T>::expectEq(res_data[5], static_cast<T>(15), "Cumsum 2D [1,2]");
    }
};

TEST_P(CumsumMultiDTypeTest, CumsumBasicFloat32) { testCumsumBasic<float>(DType::Float32); }
TEST_P(CumsumMultiDTypeTest, CumsumBasicFloat64) { testCumsumBasic<double>(DType::Float64); }
TEST_P(CumsumMultiDTypeTest, CumsumBasicInt32) { testCumsumBasic<int32_t>(DType::Int32); }
TEST_P(CumsumMultiDTypeTest, CumsumBasicInt64) { testCumsumBasic<int64_t>(DType::Int64); }
TEST_P(CumsumMultiDTypeTest, Cumsum2DFloat32) { testCumsum2D<float>(DType::Float32); }
TEST_P(CumsumMultiDTypeTest, Cumsum2DFloat64) { testCumsum2D<double>(DType::Float64); }
TEST_P(CumsumMultiDTypeTest, Cumsum2DInt32) { testCumsum2D<int32_t>(DType::Int32); }
TEST_P(CumsumMultiDTypeTest, Cumsum2DInt64) { testCumsum2D<int64_t>(DType::Int64); }

INSTANTIATE_BACKEND_TESTS(CumsumMultiDTypeTest);

// ============================================================================
// Cumprod Tests (Product Accumulation - Float Only)
// ============================================================================

class CumprodMultiDTypeTest : public BackendTest {
protected:
    template<typename T>
    void testCumprodBasic(DType dtype) {
        auto t = make_tensor<T>({4}, dtype, device,
            {static_cast<T>(2), static_cast<T>(3), static_cast<T>(4), static_cast<T>(5)});

        auto result = cumprod(t, 0);

        auto res_data = result.to(Device::cpu()).template data<T>();
        T tol = std::is_floating_point_v<T> ? static_cast<T>(1e-4) : static_cast<T>(0);

        TypedVerifier<T>::expectNear(res_data[0], static_cast<T>(2), tol, "Cumprod value 0");
        TypedVerifier<T>::expectNear(res_data[1], static_cast<T>(6), tol, "Cumprod value 1");
        TypedVerifier<T>::expectNear(res_data[2], static_cast<T>(24), tol, "Cumprod value 2");
        TypedVerifier<T>::expectNear(res_data[3], static_cast<T>(120), tol, "Cumprod value 3");
    }
};

TEST_P(CumprodMultiDTypeTest, CumprodBasicFloat32) { testCumprodBasic<float>(DType::Float32); }
TEST_P(CumprodMultiDTypeTest, CumprodBasicFloat64) { testCumprodBasic<double>(DType::Float64); }

INSTANTIATE_BACKEND_TESTS(CumprodMultiDTypeTest);
