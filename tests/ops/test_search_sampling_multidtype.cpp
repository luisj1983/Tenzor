/**
 * @file test_search_sampling_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for search and sampling operations
 *
 * Covers: SearchSorted, GumbelSoftmax
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/advanced.hpp>
#include <tenzor/ops/creation.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Fixture
// ============================================================================

class SearchSamplingMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    Tensor fromValues(const std::vector<float>& vals) {
        auto t = tenzor::full({static_cast<int64_t>(vals.size())},
                              0.0f, DType::Float32, Device::cpu());
        auto* d = t.data<float>();
        for (size_t i = 0; i < vals.size(); ++i) d[i] = vals[i];
        return t.to(device()).to(dtype());
    }
};

// ============================================================================
// SearchSorted Tests
// ============================================================================

TEST_P(SearchSamplingMultiDTypeTest, SearchSortedBasic) {
    // sorted_sequence = [1, 3, 5, 7, 9], values = [2, 4, 6]
    // Expected indices: [1, 2, 3] (left insertion)
    auto sorted_seq = fromValues({1.0f, 3.0f, 5.0f, 7.0f, 9.0f});
    auto values = fromValues({2.0f, 4.0f, 6.0f});

    auto result = tenzor::searchsorted(sorted_seq, values);
    auto result_cpu = result.to(Device::cpu());

    // Indices should be Int64 on CPU
    if (result_cpu.dtype() == DType::Int64) {
        auto* d = result_cpu.data<int64_t>();
        EXPECT_EQ(d[0], 1);
        EXPECT_EQ(d[1], 2);
        EXPECT_EQ(d[2], 3);
    } else {
        auto result_f32 = result_cpu.to(DType::Float32);
        auto* d = result_f32.data<float>();
        EXPECT_NEAR(d[0], 1.0f, 0.5f);
        EXPECT_NEAR(d[1], 2.0f, 0.5f);
        EXPECT_NEAR(d[2], 3.0f, 0.5f);
    }
    expectDevice(result);
}

TEST_P(SearchSamplingMultiDTypeTest, SearchSortedEdgeCases) {
    // Value below min, above max, equal to elements
    auto sorted_seq = fromValues({1.0f, 3.0f, 5.0f});
    auto values = fromValues({0.0f, 5.0f, 10.0f});

    auto result = tenzor::searchsorted(sorted_seq, values);
    auto result_cpu = result.to(Device::cpu());

    if (result_cpu.dtype() == DType::Int64) {
        auto* d = result_cpu.data<int64_t>();
        EXPECT_EQ(d[0], 0);  // before everything
        EXPECT_EQ(d[1], 2);  // at boundary (left insertion)
        EXPECT_EQ(d[2], 3);  // after everything
    }
}

// ============================================================================
// GumbelSoftmax Tests
// ============================================================================

TEST_P(SearchSamplingMultiDTypeTest, GumbelSoftmaxShape) {
    // Output shape should match logits shape
    auto logits = createRandn({4, 10});
    auto result = tenzor::gumbel_softmax(logits, /*tau=*/1.0, /*hard=*/false);

    expectShape(result, {4, 10});
    expectDevice(result);
}

TEST_P(SearchSamplingMultiDTypeTest, GumbelSoftmaxSumToOne) {
    // Soft samples should sum to ~1 along the last dimension
    auto logits = createRandn({8, 5});
    auto result = tenzor::gumbel_softmax(logits, /*tau=*/1.0, /*hard=*/false);

    auto sum = tenzor::sum(result, /*dim=*/-1);
    auto sum_f32 = sum.to(Device::cpu()).to(DType::Float32);
    auto* d = sum_f32.data<float>();
    for (int64_t i = 0; i < sum_f32.numel(); ++i) {
        EXPECT_NEAR(d[i], 1.0f, std::max(atol(), 1e-3f))
            << "Row " << i << " sum != 1";
    }
}

TEST_P(SearchSamplingMultiDTypeTest, GumbelSoftmaxHard) {
    // hard=true should produce approximately one-hot outputs
    auto logits = createRandn({4, 6});
    auto result = tenzor::gumbel_softmax(logits, /*tau=*/1.0, /*hard=*/true);

    auto result_f32 = result.to(Device::cpu()).to(DType::Float32);
    auto* d = result_f32.data<float>();

    for (int64_t row = 0; row < 4; ++row) {
        float row_sum = 0.0f;
        int ones_count = 0;
        for (int64_t col = 0; col < 6; ++col) {
            float val = d[row * 6 + col];
            row_sum += val;
            if (val > 0.5f) ones_count++;
        }
        EXPECT_NEAR(row_sum, 1.0f, std::max(atol(), 1e-3f));
        EXPECT_EQ(ones_count, 1) << "Row " << row << " is not one-hot";
    }
}

TEST_P(SearchSamplingMultiDTypeTest, GumbelSoftmaxLowTau) {
    // Low temperature should approach argmax behavior
    auto logits = createRandn({2, 5});
    auto result = tenzor::gumbel_softmax(logits, /*tau=*/0.01, /*hard=*/false);

    // Max value in each row should be close to 1 (sharp distribution)
    auto max_vals = tenzor::max(result, /*dim=*/-1);
    float max_min = compute_min(max_vals);
    EXPECT_GT(max_min, 0.8f) << "Low temperature should produce sharp distributions";
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(SearchSamplingMultiDTypeTest);
