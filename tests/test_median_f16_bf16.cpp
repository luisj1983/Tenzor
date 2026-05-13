/**
 * @file test_median_f16_bf16.cpp
 * @brief Phase P0 / Fix 5: CUDA `median` Float16/BFloat16 correctness.
 *
 * `median_kernel` in src/backends/cuda/kernels/advanced.cu around line 1012
 * used to write a hard-coded `0` to the indices output for half-precision
 * inputs, and reconstruct the value by converting the sorted float back
 * (losing the original half-precision representation). The fix mirrors
 * `mode_kernel`'s widen-sort-gather pattern: sort a float copy of the
 * slice with parallel index tracking, then use the resulting sorted index
 * to gather the original half value AND populate the indices output.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/advanced.hpp>
#include <tenzor/ops/creation.hpp>

#include <algorithm>
#include <vector>

namespace tenzor { void initialize(); }

namespace {
class MedianHalfEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
[[maybe_unused]] auto* g_env =
    ::testing::AddGlobalTestEnvironment(new MedianHalfEnv);
}  // namespace

using namespace tenzor;

namespace {

auto cuda_available() -> bool {
    return Device::cuda().type == Device::Type::CUDA;
}

// Build a slice of distinct sorted-but-shuffled float values (so the median
// index is unambiguous), upload to CUDA, run median, return (values, idx)
// pulled back to CPU.
struct MedianResult { std::vector<float> values; std::vector<int64_t> indices; };

template <typename T>
auto run_median_cuda(const std::vector<float>& host_data,
                     const std::vector<int64_t>& shape,
                     int64_t dim,
                     DType dtype) -> MedianResult {
    auto cpu_t = tenzor::zeros(shape, dtype, Device::cpu());
    // Fill via from_blob on a Float32 staging buffer, then .to(dtype)
    // works for half types because Tensor::to handles narrowing.
    auto staging = Tensor::from_blob(const_cast<float*>(host_data.data()),
                                     shape, DType::Float32, Device::cpu())
                       .clone();
    auto half_cpu = staging.to(dtype);
    auto cuda_t = half_cpu.to(Device::cuda());
    auto [vals, idx] = tenzor::median(cuda_t, dim, /*keepdim=*/false);
    auto vals_cpu = vals.to(Device::cpu()).contiguous();
    auto idx_cpu  = idx.to(Device::cpu()).contiguous();

    // Convert values back to float for comparison.
    auto vals_f32 = vals_cpu.to(DType::Float32).contiguous();
    MedianResult r;
    r.values.assign(static_cast<size_t>(vals_f32.numel()), 0.0f);
    r.indices.assign(static_cast<size_t>(idx_cpu.numel()), 0);
    std::memcpy(r.values.data(), vals_f32.data_ptr(),
                r.values.size() * sizeof(float));
    std::memcpy(r.indices.data(), idx_cpu.data_ptr(),
                r.indices.size() * sizeof(int64_t));
    return r;
}

}  // namespace

class MedianHalfPrecision
    : public ::testing::TestWithParam<DType> {};

TEST_P(MedianHalfPrecision, ReturnsCorrectValueAndIndex) {
    if (!cuda_available()) GTEST_SKIP() << "CUDA not available";
    const DType dtype = GetParam();

    // 5-element slice with distinct values. After sorting, median is at
    // index 2 (the middle). Pre-shuffle so the median index in the ORIGINAL
    // slice is unambiguous and not 0 (the old placeholder).
    //
    // Sorted ascending:        [-3.0, -1.0,  0.5,  2.0,  4.0]
    // Original (shuffled):     [ 2.0, -3.0,  0.5, -1.0,  4.0]
    // Original-index of median (0.5): 2.
    std::vector<float> data = {2.0f, -3.0f, 0.5f, -1.0f, 4.0f};
    auto r = run_median_cuda<float>(data, {5}, /*dim=*/0, dtype);
    ASSERT_EQ(r.values.size(), 1u);
    ASSERT_EQ(r.indices.size(), 1u);
    EXPECT_NEAR(r.values[0], 0.5f, dtype == DType::Float16 ? 1e-3f : 1e-2f)
        << "median value wrong for dtype="
        << static_cast<int>(dtype);
    EXPECT_EQ(r.indices[0], 2)
        << "median index wrong (returned the pre-fix placeholder 0?)";
}

TEST_P(MedianHalfPrecision, IndexNotPlaceholderZero) {
    // Specifically guards against the old placeholder behaviour: the
    // median element here is NOT at position 0, so a returned index of 0
    // would indicate the bug is back.
    if (!cuda_available()) GTEST_SKIP() << "CUDA not available";
    const DType dtype = GetParam();

    // Median (sorted: -5, -2, 1, 3, 4, 7, 9) -> 3 at sorted index 3.
    // Original positions:   [ 4, -5, 9, -2, 1, 3, 7]
    //                      idx  0   1  2   3  4  5  6
    // Sorted from these:    [-5,-2,1,3,4,7,9]; original positions
    //                       [ 1, 3,4,5,0,6,2]; mid (sorted idx 3) -> orig idx 5.
    std::vector<float> data = {4.0f, -5.0f, 9.0f, -2.0f, 1.0f, 3.0f, 7.0f};
    auto r = run_median_cuda<float>(data, {7}, /*dim=*/0, dtype);
    ASSERT_EQ(r.values.size(), 1u);
    ASSERT_EQ(r.indices.size(), 1u);
    EXPECT_NEAR(r.values[0], 3.0f, dtype == DType::Float16 ? 1e-3f : 1e-2f);
    EXPECT_EQ(r.indices[0], 5)
        << "median index should be 5 (the original slice position of 3.0). "
           "A return of 0 indicates the pre-fix placeholder is back.";
}

INSTANTIATE_TEST_SUITE_P(
    HalfTypes, MedianHalfPrecision,
    ::testing::Values(DType::Float16, DType::BFloat16),
    [](const ::testing::TestParamInfo<DType>& info) {
        return info.param == DType::Float16 ? "Float16" : "BFloat16";
    });

// Regression guard: Float32/Float64 paths must keep working (the rewrite
// only touched the half-precision branch).
TEST(MedianHalfPrecision, Float32PathUnchanged) {
    if (!cuda_available()) GTEST_SKIP() << "CUDA not available";
    auto staging = tenzor::zeros({5}, DType::Float32, Device::cpu());
    std::vector<float> data = {2.0f, -3.0f, 0.5f, -1.0f, 4.0f};
    std::memcpy(staging.data_ptr(), data.data(), data.size() * sizeof(float));
    auto cuda_t = staging.to(Device::cuda());
    auto [vals, idx] = tenzor::median(cuda_t, /*dim=*/0, /*keepdim=*/false);
    auto v = vals.to(Device::cpu()).contiguous();
    auto i = idx.to(Device::cpu()).contiguous();
    EXPECT_FLOAT_EQ(v.data<float>()[0], 0.5f);
    EXPECT_EQ(i.data<int64_t>()[0], 2);
}
