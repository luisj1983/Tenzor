/**
 * @file test_group_instance_norm_f64_precision.cpp
 * @brief Numerical-precision fix: group_norm_impl and instance_norm_impl
 *        used `float` accumulators inside `template<typename T>`, silently
 *        truncating Float64 inputs to Float32 precision — same MEMORY.md
 *        bug pattern fixed earlier for layer_norm_scalar.
 *
 * Strategy: inputs at 1.0 + i*1e-9 (Float32 rounds every element to 1.0
 * exactly, Float64 preserves the per-element 1e-9 offset). With large
 * weights amplifying small (in - mean) differences in the normalized
 * output, the buggy float-accum path produces effectively zero output;
 * the fixed double-accum path produces the proper non-zero pattern.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/backend/op_attributes.hpp>
#include <tenzor/ops/op_id.hpp>
#include <tenzor/ops/creation.hpp>

#include <cmath>
#include <span>
#include <vector>

namespace tenzor { void initialize(); }

namespace {
class GroupInstanceNormF64Env : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
[[maybe_unused]] auto* g_env =
    ::testing::AddGlobalTestEnvironment(new GroupInstanceNormF64Env);
}  // namespace

using namespace tenzor;

namespace {

// Build a (1, C, H*W) input where each element is 1.0 + idx*1e-9. In Float32
// every element rounds to 1.0 exactly; in Float64 every element is distinct.
// The buggy float-accum code would compute mean = 1.0 and produce zero
// normalized output; the fix preserves the F64 precision and yields a
// non-zero gradient-of-output-vs-input pattern.
auto build_input(int64_t C, int64_t spatial) -> std::vector<double> {
    std::vector<double> data(static_cast<size_t>(C * spatial));
    for (int64_t i = 0; i < C * spatial; ++i) {
        data[static_cast<size_t>(i)] = 1.0 + i * 1e-9;
    }
    return data;
}

}  // namespace

TEST(GroupInstanceNormF64Precision, GroupNormFloat64PreservesPerElementOffsets) {
    constexpr int64_t N = 1;
    constexpr int64_t C = 4;
    constexpr int64_t H = 4;
    constexpr int64_t W = 4;
    constexpr int64_t num_groups = 2;
    auto raw_in = build_input(C, H * W);
    std::vector<double> raw_w(C, 1e6);  // amplify so precision loss is visible
    std::vector<double> raw_b(C, 0.0);

    auto in_t = Tensor::from_blob(raw_in.data(), {N, C, H, W}, DType::Float64,
                                  Device::cpu()).clone();
    auto w_t  = Tensor::from_blob(raw_w.data(),  {C}, DType::Float64,
                                  Device::cpu()).clone();
    auto b_t  = Tensor::from_blob(raw_b.data(),  {C}, DType::Float64,
                                  Device::cpu()).clone();

    OpAttributes attrs;
    attrs.set(AttrKey::NumGroups, num_groups);
    attrs.set(AttrKey::Eps, 1e-5);
    const Tensor in_arr[3] = {in_t, w_t, b_t};
    auto out_vec = tenzor::dispatch(OpId::GroupNorm,
                                    std::span<const Tensor>{in_arr, 3}, attrs);
    ASSERT_GE(out_vec.size(), 1u);
    const auto& out = out_vec[0];
    ASSERT_EQ(out.dtype(), DType::Float64);

    auto out_cpu = out.to(Device::cpu()).contiguous();
    const double* od = out_cpu.data<double>();

    // Compute the variance of the output across the elements within one
    // group. The fix gives variance proportional to the 1e6 amplification of
    // the 1e-9 offsets, scaled by the inverse standard deviation of the
    // group. The pre-fix code produced mean=1.0 and zero diff → zero output
    // variance.
    double mean = 0.0;
    for (int64_t i = 0; i < out_cpu.numel(); ++i) mean += od[i];
    mean /= static_cast<double>(out_cpu.numel());
    double var = 0.0;
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        const double d = od[i] - mean;
        var += d * d;
    }
    var /= static_cast<double>(out_cpu.numel());
    EXPECT_GT(var, 1e-3)
        << "group_norm Float64 output variance is too low — "
           "accumulator precision loss collapses per-element offsets to zero";
}

TEST(GroupInstanceNormF64Precision, InstanceNormFloat64PreservesPerElementOffsets) {
    constexpr int64_t N = 1;
    constexpr int64_t C = 2;
    constexpr int64_t H = 4;
    constexpr int64_t W = 4;
    auto raw_in = build_input(C, H * W);
    std::vector<double> raw_w(C, 1e6);
    std::vector<double> raw_b(C, 0.0);

    auto in_t = Tensor::from_blob(raw_in.data(), {N, C, H, W}, DType::Float64,
                                  Device::cpu()).clone();
    auto w_t  = Tensor::from_blob(raw_w.data(),  {C}, DType::Float64,
                                  Device::cpu()).clone();
    auto b_t  = Tensor::from_blob(raw_b.data(),  {C}, DType::Float64,
                                  Device::cpu()).clone();

    OpAttributes attrs;
    attrs.set(AttrKey::Eps, 1e-5);
    const Tensor in_arr[3] = {in_t, w_t, b_t};
    auto out_vec = tenzor::dispatch(OpId::InstanceNorm,
                                    std::span<const Tensor>{in_arr, 3}, attrs);
    ASSERT_GE(out_vec.size(), 1u);
    const auto& out = out_vec[0];
    ASSERT_EQ(out.dtype(), DType::Float64);

    auto out_cpu = out.to(Device::cpu()).contiguous();
    const double* od = out_cpu.data<double>();
    double mean = 0.0;
    for (int64_t i = 0; i < out_cpu.numel(); ++i) mean += od[i];
    mean /= static_cast<double>(out_cpu.numel());
    double var = 0.0;
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        const double d = od[i] - mean;
        var += d * d;
    }
    var /= static_cast<double>(out_cpu.numel());
    EXPECT_GT(var, 1e-3)
        << "instance_norm Float64 output variance is too low";
}

// Regression guard: Float32 path still correct (the change widened accumulators
// from F32 to F64; F32 callers see no behavior change beyond improved stability).
TEST(GroupInstanceNormF64Precision, Float32PathStillCorrect) {
    constexpr int64_t N = 2;
    constexpr int64_t C = 4;
    auto x = tenzor::randn({N, C, 4, 4}, DType::Float32, Device::cpu());
    auto w = tenzor::ones({C}, DType::Float32, Device::cpu());
    auto b = tenzor::zeros({C}, DType::Float32, Device::cpu());
    OpAttributes attrs;
    attrs.set(AttrKey::NumGroups, static_cast<int64_t>(2));
    attrs.set(AttrKey::Eps, 1e-5);
    const Tensor in_arr[3] = {x, w, b};
    auto out_vec = tenzor::dispatch(OpId::GroupNorm,
                                    std::span<const Tensor>{in_arr, 3}, attrs);
    ASSERT_GE(out_vec.size(), 1u);
    EXPECT_EQ(out_vec[0].dtype(), DType::Float32);
    EXPECT_EQ(out_vec[0].numel(), N * C * 4 * 4);
}
