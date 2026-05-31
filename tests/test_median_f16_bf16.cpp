/**
 * @file test_median_f16_bf16.cpp
 * @brief Phase P0 / Fix 5: `median` Float16/BFloat16 correctness across backends.
 *
 * `median_kernel` in src/backends/cuda/kernels/advanced.cu around line 1012
 * used to write a hard-coded `0` to the indices output for half-precision
 * inputs, and reconstruct the value by converting the sorted float back
 * (losing the original half-precision representation). The fix mirrors
 * `mode_kernel`'s widen-sort-gather pattern: sort a float copy of the
 * slice with parallel index tracking, then use the resulting sorted index
 * to gather the original half value AND populate the indices output.
 *
 * Ported to the cross-backend BackendTest fixture: each test runs on every
 * available backend, and the half-precision cases iterate Float16/BFloat16
 * inside the body.
 */

#include "backend_test_fixture.hpp"
#include <tenzor/ops/advanced.hpp>
#include <tenzor/ops/creation.hpp>

#include <algorithm>
#include <vector>

using namespace tenzor;

namespace {

struct MedianResult { std::vector<float> values; std::vector<int64_t> indices; };

// Build a half-precision tensor on `device` from a Float32 host slice, run
// median, and pull values/indices back to the host.
auto run_median(const std::vector<float>& host_data,
                const std::vector<int64_t>& shape,
                int64_t dim,
                DType dtype,
                const tenzor::Device& device) -> MedianResult {
    auto staging = Tensor::from_blob(const_cast<float*>(host_data.data()),
                                     shape, DType::Float32, Device::cpu())
                       .clone();
    auto half_dev = staging.to(dtype).to(device);
    auto [vals, idx] = tenzor::median(half_dev, dim, /*keepdim=*/false);

    auto vals_cpu = vals.cpu().to(DType::Float32).contiguous();
    auto idx_cpu = idx.cpu().contiguous();

    MedianResult r;
    r.values.assign(static_cast<size_t>(vals_cpu.numel()), 0.0f);
    r.indices.assign(static_cast<size_t>(idx_cpu.numel()), 0);
    for (int64_t i = 0; i < vals_cpu.numel(); ++i) {
        r.values[static_cast<size_t>(i)] = vals_cpu.data<float>()[i];
    }
    for (int64_t i = 0; i < idx_cpu.numel(); ++i) {
        r.indices[static_cast<size_t>(i)] = idx_cpu.data<int64_t>()[i];
    }
    return r;
}

class MedianHalfPrecision : public ::tenzor::testing::BackendTest {};

TEST_P(MedianHalfPrecision, ReturnsCorrectValueAndIndex) {
    // 5-element slice with distinct values. After sorting, median is at
    // index 2 (the middle). Pre-shuffle so the median index in the ORIGINAL
    // slice is unambiguous and not 0 (the old placeholder).
    //
    // Sorted ascending:        [-3.0, -1.0,  0.5,  2.0,  4.0]
    // Original (shuffled):     [ 2.0, -3.0,  0.5, -1.0,  4.0]
    // Original-index of median (0.5): 2.
    std::vector<float> data = {2.0f, -3.0f, 0.5f, -1.0f, 4.0f};
    for (DType dtype : {DType::Float16, DType::BFloat16}) {
        auto r = run_median(data, {5}, /*dim=*/0, dtype, device);
        ASSERT_EQ(r.values.size(), 1u);
        ASSERT_EQ(r.indices.size(), 1u);
        EXPECT_NEAR(r.values[0], 0.5f, dtype == DType::Float16 ? 1e-3f : 1e-2f)
            << "median value wrong for dtype="
            << static_cast<int>(dtype);
        EXPECT_EQ(r.indices[0], 2)
            << "median index wrong (returned the pre-fix placeholder 0?) "
               "for dtype=" << static_cast<int>(dtype);
    }
}

TEST_P(MedianHalfPrecision, IndexNotPlaceholderZero) {
    // Specifically guards against the old placeholder behaviour: the
    // median element here is NOT at position 0, so a returned index of 0
    // would indicate the bug is back.
    //
    // Median (sorted: -5, -2, 1, 3, 4, 7, 9) -> 3 at sorted index 3.
    // Original positions:   [ 4, -5, 9, -2, 1, 3, 7]
    //                      idx  0   1  2   3  4  5  6
    // Sorted from these:    [-5,-2,1,3,4,7,9]; original positions
    //                       [ 1, 3,4,5,0,6,2]; mid (sorted idx 3) -> orig idx 5.
    std::vector<float> data = {4.0f, -5.0f, 9.0f, -2.0f, 1.0f, 3.0f, 7.0f};
    for (DType dtype : {DType::Float16, DType::BFloat16}) {
        auto r = run_median(data, {7}, /*dim=*/0, dtype, device);
        ASSERT_EQ(r.values.size(), 1u);
        ASSERT_EQ(r.indices.size(), 1u);
        EXPECT_NEAR(r.values[0], 3.0f, dtype == DType::Float16 ? 1e-3f : 1e-2f)
            << "for dtype=" << static_cast<int>(dtype);
        EXPECT_EQ(r.indices[0], 5)
            << "median index should be 5 (the original slice position of 3.0). "
               "A return of 0 indicates the pre-fix placeholder is back. "
               "dtype=" << static_cast<int>(dtype);
    }
}

// Regression guard: Float32/Float64 paths must keep working (the rewrite
// only touched the half-precision branch).
TEST_P(MedianHalfPrecision, Float32PathUnchanged) {
    std::vector<float> data = {2.0f, -3.0f, 0.5f, -1.0f, 4.0f};
    auto host = Tensor::from_blob(data.data(), {5}, DType::Float32, Device::cpu())
                    .clone();
    auto dev_t = host.to(device);
    auto [vals, idx] = tenzor::median(dev_t, /*dim=*/0, /*keepdim=*/false);
    auto v = vals.cpu().contiguous();
    auto i = idx.cpu().contiguous();
    EXPECT_FLOAT_EQ(v.data<float>()[0], 0.5f);
    EXPECT_EQ(i.data<int64_t>()[0], 2);
}

INSTANTIATE_BACKEND_TESTS(MedianHalfPrecision);

}  // namespace
