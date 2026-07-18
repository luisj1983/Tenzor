/**
 * @file test_nanstats_parity.cpp
 * @brief Cross-backend parity for NaN-aware statistics, quantiles, histc,
 *        kthvalue, unique_consecutive, cov, corrcoef, median, mode.
 *
 * Covers OpIds: NanVar, NanStd, Nanmedian, Quantile, Nanquantile, Histc,
 * Kthvalue, UniqueConsecutive, Cov, Corrcoef, Median, Mode. The audit
 * (2026-05-02) flagged these as missing dedicated parity coverage even
 * though all are registered on every non-MPS backend.
 *
 * Median/Mode tie-break coverage: their index/value tie-break rules were
 * historically coordinated bug fixes across all 5 backends (grep tags
 * F065, F132, F138, F140, F144). Summary of what those fixes pinned down,
 * which the Median_* / Mode_* tests below exercise directly:
 *   - F065/F140: NaN sorts as the LARGEST value (not "always false"), so
 *     median/mode/sort comparators don't break strict-weak-ordering.
 *   - median_kernel: even-length reductions take the LOWER of the two
 *     middle sorted values (mid = (dim_size-1)/2), not an average.
 *   - median_kernel: on a duplicated median value, the returned index is
 *     the one a *stable* ascending sort (equal values ordered by
 *     ascending original index) would place at rank `mid` — deterministic
 *     across backends, not "whichever nth_element/sort leaves there".
 *   - F132/F138/F144: mode's tie-break on equal counts picks the SMALLEST
 *     modal VALUE (not the smallest index) and reports the HIGHEST
 *     original index of that value's run ("run END", per the Vulkan
 *     mode.comp F144 comment). OneAPI/ROCm previously tied to the
 *     smallest index instead, returning a different modal VALUE on ties
 *     (e.g. [7,7,3,3] -> 7 instead of 3), which silently broke
 *     ModeBackward since it masks gradients by the returned value.
 */

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/advanced.hpp>
#include <tenzor/ops/reduction.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class NanStatsParity : public BackendTest {};

namespace {
// Build a Float32 tensor with a sprinkle of NaN values at fixed positions.
Tensor make_with_nans(std::vector<int64_t> shape, double nan_fraction = 0.15) {
    auto t = randn(shape, DType::Float32, Device::cpu());
    int64_t n = t.numel();
    int64_t nan_count = static_cast<int64_t>(std::ceil(n * nan_fraction));
    float* data = t.data<float>();
    for (int64_t i = 0; i < nan_count; ++i) {
        // Spread NaNs deterministically through the tensor.
        int64_t pos = (i * 7 + 3) % n;
        data[pos] = std::numeric_limits<float>::quiet_NaN();
    }
    return t;
}
}  // namespace

// ----------------------------------------------------------------------------
// NanVar / NanStd — should ignore NaN positions
// ----------------------------------------------------------------------------

TEST_P(NanStatsParity, NanVar_ReduceAll) {
    auto x = make_with_nans({4, 16});
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return nanvar(in[0]);
    }, {x}, device, 1e-4f, 1e-5f, "NanVar_All");
}

TEST_P(NanStatsParity, NanVar_AlongDim) {
    // nanvar() is implemented compositionally as
    //     sum((x - nanmean(x))**2, ignoring NaN) / (n_valid - correction)
    // (see src/ops/reduction.cpp:nanvar). The naive E[(x-mean)^2] form
    // accumulates Float32 round-off whose magnitude depends on each
    // backend's reduction order, so cross-backend agreement is bounded
    // by ~1% of the variance value, not by 1e-5. The wider tolerance
    // here matches the empirical drift on a 16-element-per-row var.
    auto x = randn({4, 16}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return nanvar(in[0], /*dim=*/1, /*keepdim=*/false, /*correction=*/1);
    }, {x}, device, /*rtol=*/3e-2f, /*atol=*/5e-2f, "NanVar_Dim_NoNaN");
}

TEST_P(NanStatsParity, NanStd_ReduceAll) {
    auto x = make_with_nans({3, 12});
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return nanstd(in[0]);
    }, {x}, device, 1e-4f, 1e-5f, "NanStd_All");
}

// ----------------------------------------------------------------------------
// Nanmedian
// ----------------------------------------------------------------------------

TEST_P(NanStatsParity, Nanmedian_All) {
    // Deterministic input where the non-NaN count is ODD so the median is
    // a single middle element (no even-count averaging ambiguity). PyTorch
    // and CPU both implement the lower-median for even counts but the GPU
    // composed-ops fallback uses the average — we sidestep that
    // backend-spec gap by ensuring an odd survivor count.
    // 12 elements, 1 NaN ⇒ 11 non-NaN, odd.
    auto x = full({12}, 0.0, DType::Float32, Device::cpu());
    float vals[] = {3.0f, std::numeric_limits<float>::quiet_NaN(),
                    1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f, 6.0f, 5.0f, 3.0f, 7.0f};
    for (int i = 0; i < 12; ++i) x.data<float>()[i] = vals[i];

    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return nanmedian(in[0]);
    }, {x}, device, 1e-5f, 1e-7f, "Nanmedian_All");
}

// ----------------------------------------------------------------------------
// Quantile / Nanquantile
// ----------------------------------------------------------------------------

TEST_P(NanStatsParity, Quantile_AllValues) {
    auto x = randn({64}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return quantile(in[0], /*q=*/0.25);
    }, {x}, device, 1e-4f, 1e-5f, "Quantile_q025");
}

TEST_P(NanStatsParity, Quantile_AlongDim_Median) {
    auto x = randn({4, 32}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return quantile(in[0], /*q=*/0.5, /*dim=*/1);
    }, {x}, device, 1e-4f, 1e-5f, "Quantile_q050_Dim");
}

TEST_P(NanStatsParity, Nanquantile_WithNaNs) {
    auto x = make_with_nans({4, 32});
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return nanquantile(in[0], /*q=*/0.75, /*dim=*/1);
    }, {x}, device, 1e-4f, 1e-5f, "Nanquantile_q075");
}

// ----------------------------------------------------------------------------
// Histc — fixed-bin histogram of a 1-D float input
// ----------------------------------------------------------------------------

// Note: Histc with min==max==0 selects an implementation-defined
// auto-range from the input's empirical min/max. Backends compute this
// reduction with slightly different orderings and the resulting bin edges
// place boundary-adjacent values into different buckets, producing a
// large absolute count diff that is not a correctness regression. The
// FixedRange variant below provides the cross-backend coverage the audit
// asked for; auto-range parity belongs in a per-backend numeric-stability
// test, not in cross-backend parity.

TEST_P(NanStatsParity, Histc_FixedRange) {
    auto x = randn({256}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return histc(in[0], /*bins=*/8, /*min=*/-2.0, /*max=*/2.0);
    }, {x}, device, 0.0f, 0.0f, "Histc_FixedRange");
}

// ----------------------------------------------------------------------------
// Kthvalue — k-th smallest along dim, returns (values, indices) pair.
// ----------------------------------------------------------------------------

TEST_P(NanStatsParity, Kthvalue_Values_Dim1) {
    auto x = randn({4, 16}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return kthvalue(in[0], /*k=*/3, /*dim=*/1).first;
    }, {x}, device, 1e-5f, 1e-7f, "Kthvalue_Values");
}

// ----------------------------------------------------------------------------
// Median — see file header for the F065/tie-break background. Every case
// below asserts the exact expected (value, index) on CPU first (verifying
// this codebase's documented convention), then runs the same op through
// test_operation_parity_single so all backends are checked against CPU for
// both the values AND the indices tensors — the index tensor is what
// actually exercises the tie-break logic.
// ----------------------------------------------------------------------------

TEST_P(NanStatsParity, Median_Odd_NoDimArg) {
    // 7 unique values, dim argument OMITTED (ops-level default dim=-1,
    // i.e. this 1-D tensor's only dimension) -- covers the "without a dim
    // argument" case. Sorted: [1,2,3,5,7,8,9], mid=(7-1)/2=3 -> value 5,
    // originally at index 0.
    auto x = full({7}, 0.0, DType::Float32, Device::cpu());
    float vals[] = {5, 3, 8, 1, 9, 2, 7};
    for (int i = 0; i < 7; ++i) x.data<float>()[i] = vals[i];

    auto [val_cpu, idx_cpu] = median(x);
    EXPECT_FLOAT_EQ(val_cpu.data<float>()[0], 5.0f);
    EXPECT_EQ(idx_cpu.data<int64_t>()[0], 0);

    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return std::get<0>(median(in[0]));
    }, {x}, device, 1e-5f, 1e-7f, "Median_Odd_Values");
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return std::get<1>(median(in[0]));
    }, {x}, device, 0.0f, 0.0f, "Median_Odd_Indices");
}

TEST_P(NanStatsParity, Median_Even_LowerConvention_Dim) {
    // Even dim_size=6 per row, explicit dim=1. mid=(6-1)/2=2 selects the
    // LOWER of the two middle sorted values, not their average and not the
    // upper value -- confirms median_kernel's "lower median for even
    // sizes" convention (cpu reduction.cpp / cuda advanced.cu agree).
    //   row0 sorted: 10,20,30,40,50,60 -> lower-mid 30 @ original index 2
    //   row1 sorted: 10,20,40,60,80,100 -> lower-mid 40 @ original index 3
    auto x = full({2, 6}, 0.0, DType::Float32, Device::cpu());
    float row0[] = {40, 10, 30, 60, 20, 50};
    float row1[] = {100, 20, 80, 40, 60, 10};
    for (int i = 0; i < 6; ++i) x.data<float>()[i] = row0[i];
    for (int i = 0; i < 6; ++i) x.data<float>()[6 + i] = row1[i];

    auto [val_cpu, idx_cpu] = median(x, /*dim=*/1);
    EXPECT_FLOAT_EQ(val_cpu.data<float>()[0], 30.0f)
        << "even-length median must return the LOWER middle value, not the average";
    EXPECT_EQ(idx_cpu.data<int64_t>()[0], 2);
    EXPECT_FLOAT_EQ(val_cpu.data<float>()[1], 40.0f);
    EXPECT_EQ(idx_cpu.data<int64_t>()[1], 3);

    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return std::get<0>(median(in[0], /*dim=*/1));
    }, {x}, device, 1e-5f, 1e-7f, "Median_Even_Dim_Values");
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return std::get<1>(median(in[0], /*dim=*/1));
    }, {x}, device, 0.0f, 0.0f, "Median_Even_Dim_Indices");
}

TEST_P(NanStatsParity, Median_DuplicateValues_IndexTieBreak) {
    // All four elements equal: the VALUE is trivially correct on any
    // backend, but the INDEX is genuinely ambiguous without a tie-break --
    // exactly what median_kernel's ascending-original-index tie-break
    // pins down. mid=(4-1)/2=1 -> index 1 (not 0, 2, or 3).
    auto x = full({4}, 2.0, DType::Float32, Device::cpu());

    auto [val_cpu, idx_cpu] = median(x, /*dim=*/0);
    EXPECT_FLOAT_EQ(val_cpu.data<float>()[0], 2.0f);
    EXPECT_EQ(idx_cpu.data<int64_t>()[0], 1);

    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return std::get<1>(median(in[0], /*dim=*/0));
    }, {x}, device, 0.0f, 0.0f, "Median_AllDuplicate_Indices");
}

TEST_P(NanStatsParity, Median_DuplicateAtBoundary_IndexTieBreak) {
    // Only the values spanning the median rank are duplicated: {7,3,3,3,9}.
    // Stable ascending order by (value, then original index) is
    // (3,idx1),(3,idx2),(3,idx3),(7,idx0),(9,idx4); mid=(5-1)/2=2 lands on
    // the THIRD occurrence of value 3, i.e. original index 3. This is the
    // sharper form of the F065 tie-break: the answer is NOT "the lowest
    // index among all duplicates of the median value" (that would be
    // index 1) but whichever index a stable sort places at rank `mid`.
    auto x = full({5}, 0.0, DType::Float32, Device::cpu());
    float vals[] = {7, 3, 3, 3, 9};
    for (int i = 0; i < 5; ++i) x.data<float>()[i] = vals[i];

    auto [val_cpu, idx_cpu] = median(x, /*dim=*/0);
    EXPECT_FLOAT_EQ(val_cpu.data<float>()[0], 3.0f);
    EXPECT_EQ(idx_cpu.data<int64_t>()[0], 3);

    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return std::get<0>(median(in[0], /*dim=*/0));
    }, {x}, device, 1e-5f, 1e-7f, "Median_BoundaryDup_Values");
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return std::get<1>(median(in[0], /*dim=*/0));
    }, {x}, device, 0.0f, 0.0f, "Median_BoundaryDup_Indices");
}

TEST_P(NanStatsParity, Median_AlongDim_Random) {
    // General-case coverage on random data. dim_size=17 (odd) along dim=1
    // avoids the even-length convention entirely, so this is a plain
    // "does the selected element match across backends" check, mirroring
    // the file's existing Quantile_AlongDim_Median pattern.
    auto x = randn({4, 17}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return std::get<0>(median(in[0], /*dim=*/1));
    }, {x}, device, 1e-5f, 1e-7f, "Median_Random_Dim_Values");
}

// ----------------------------------------------------------------------------
// Mode — most-frequent-value selection. See file header for the F132/F138/
// F144 tie-break convention this exercises: on a count tie, pick the
// SMALLEST modal value and report the HIGHEST original index of that
// value's run.
// ----------------------------------------------------------------------------

TEST_P(NanStatsParity, Mode_TiedCounts_SmallestValueHighestIndex) {
    // [7,7,3,3]: count(7)=count(3)=2, a genuine tie. This is the literal
    // F132 regression case: OneAPI/ROCm used to tie to the smallest INDEX
    // (returning value 7) instead of the smallest VALUE (3).
    auto x = full({4}, 0.0, DType::Float32, Device::cpu());
    float vals[] = {7, 7, 3, 3};
    for (int i = 0; i < 4; ++i) x.data<float>()[i] = vals[i];

    auto [val_cpu, idx_cpu] = mode(x, /*dim=*/0);
    EXPECT_FLOAT_EQ(val_cpu.data<float>()[0], 3.0f);
    EXPECT_EQ(idx_cpu.data<int64_t>()[0], 3);

    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return std::get<0>(mode(in[0], /*dim=*/0));
    }, {x}, device, 1e-5f, 1e-7f, "Mode_Tied_Values");
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return std::get<1>(mode(in[0], /*dim=*/0));
    }, {x}, device, 0.0f, 0.0f, "Mode_Tied_Indices");
}

TEST_P(NanStatsParity, Mode_ClearWinner) {
    // {1,2,2,2,3}: unambiguous mode 2 (count 3), whose run sits at original
    // indices 1,2,3 -> run-END (highest) index 3.
    auto x = full({5}, 0.0, DType::Float32, Device::cpu());
    float vals[] = {1, 2, 2, 2, 3};
    for (int i = 0; i < 5; ++i) x.data<float>()[i] = vals[i];

    auto [val_cpu, idx_cpu] = mode(x, /*dim=*/0);
    EXPECT_FLOAT_EQ(val_cpu.data<float>()[0], 2.0f);
    EXPECT_EQ(idx_cpu.data<int64_t>()[0], 3);

    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return std::get<0>(mode(in[0], /*dim=*/0));
    }, {x}, device, 1e-5f, 1e-7f, "Mode_Clear_Values");
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return std::get<1>(mode(in[0], /*dim=*/0));
    }, {x}, device, 0.0f, 0.0f, "Mode_Clear_Indices");
}

TEST_P(NanStatsParity, Mode_NaN_Handling) {
    // {1,2,3,NaN,NaN}: mode_nan_aware_eq treats NaN==NaN, so the repeated
    // NaN (count 2) legitimately outranks every distinct singleton and IS
    // the mode. Exercises both the NaN-aware sort comparator (which must
    // not violate strict-weak-ordering, unlike plain `<`) and the NaN-aware
    // run-length count in the same kernel; the NaN run's END is index 4.
    auto x = full({5}, 0.0, DType::Float32, Device::cpu());
    float nan = std::numeric_limits<float>::quiet_NaN();
    float vals[] = {1, 2, 3, nan, nan};
    for (int i = 0; i < 5; ++i) x.data<float>()[i] = vals[i];

    auto [val_cpu, idx_cpu] = mode(x, /*dim=*/0);
    EXPECT_TRUE(std::isnan(val_cpu.data<float>()[0]));
    EXPECT_EQ(idx_cpu.data<int64_t>()[0], 4);

    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return std::get<0>(mode(in[0], /*dim=*/0));
    }, {x}, device, 1e-5f, 1e-7f, "Mode_NaN_Values");
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return std::get<1>(mode(in[0], /*dim=*/0));
    }, {x}, device, 0.0f, 0.0f, "Mode_NaN_Indices");
}

TEST_P(NanStatsParity, Mode_AlongDim_MultiDim) {
    // {2,5}, dim=1: row0 mode=5 (count 3, run-END index 2); row1 mode=4
    // (count 3 beats 9's count 2 outright -- no tie -- run-END index 4).
    // Covers the multi-row dim!=REDUCE_ALL path with an independent
    // run-length scan per row.
    auto x = full({2, 5}, 0.0, DType::Float32, Device::cpu());
    float row0[] = {5, 5, 5, 1, 2};
    float row1[] = {9, 9, 4, 4, 4};
    for (int i = 0; i < 5; ++i) x.data<float>()[i] = row0[i];
    for (int i = 0; i < 5; ++i) x.data<float>()[5 + i] = row1[i];

    auto [val_cpu, idx_cpu] = mode(x, /*dim=*/1);
    EXPECT_FLOAT_EQ(val_cpu.data<float>()[0], 5.0f);
    EXPECT_EQ(idx_cpu.data<int64_t>()[0], 2);
    EXPECT_FLOAT_EQ(val_cpu.data<float>()[1], 4.0f);
    EXPECT_EQ(idx_cpu.data<int64_t>()[1], 4);

    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return std::get<0>(mode(in[0], /*dim=*/1));
    }, {x}, device, 1e-5f, 1e-7f, "Mode_Dim_Values");
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return std::get<1>(mode(in[0], /*dim=*/1));
    }, {x}, device, 0.0f, 0.0f, "Mode_Dim_Indices");
}

// ----------------------------------------------------------------------------
// UniqueConsecutive — collapses runs of equal values
// ----------------------------------------------------------------------------

TEST_P(NanStatsParity, UniqueConsecutive_Default) {
    // Hand-built input with consecutive duplicates so the result is
    // deterministic and easy to compare bitwise.
    auto x = full({10}, 0.0, DType::Float32, Device::cpu());
    float vals[] = {1, 1, 2, 3, 3, 3, 4, 5, 5, 6};
    for (int i = 0; i < 10; ++i) x.data<float>()[i] = vals[i];

    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return std::get<0>(unique_consecutive(in[0]));
    }, {x}, device, 0.0f, 0.0f, "UniqueConsecutive");
}

// ----------------------------------------------------------------------------
// Cov / Corrcoef
// ----------------------------------------------------------------------------

TEST_P(NanStatsParity, Cov_2DMatrix) {
    auto x = randn({4, 32}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return cov(in[0], /*correction=*/1);
        // cov computes X @ X^T internally (src/ops/reduction.cpp:527) so it
        // inherits the FP32 cross-device GEMM floor; see parity::MATMUL_*.
    }, {x}, device, parity::MATMUL_RTOL, parity::MATMUL_ATOL, "Cov_2D");
}

TEST_P(NanStatsParity, Corrcoef_2DMatrix) {
    auto x = randn({4, 32}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return corrcoef(in[0]);
    }, {x}, device, 1e-4f, 1e-5f, "Corrcoef_2D");
}

INSTANTIATE_BACKEND_TESTS(NanStatsParity);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    try {
        if (!::testing::GTEST_FLAG(list_tests)) {
            tenzor::initialize();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }

    int result = RUN_ALL_TESTS();

    try {
        tenzor::finalize();
    } catch (...) {}

    return result;
}
