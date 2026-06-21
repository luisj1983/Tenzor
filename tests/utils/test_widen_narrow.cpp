// Tests for include/tenzor/utils/widen_narrow.hpp
//
// Verify the helper:
//   - widens Float16/BFloat16 to Float32 before invoking the callable
//   - narrows the result back to the input dtype
//   - is a passthrough for Float32/Float64 inputs
//   - still narrows when the callable widens internally (consistency)
//   - works correctly in the two-input form when either input is half
//   - PRESERVES NUMERIC VALUES through widen->compute->narrow (not just dtype)
//   - computes at Float64 (not Float32) when a Float64 operand is paired with a
//     half operand — the documented silent-accumulator branch in the header.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "tenzor/core/dtype.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/tenzor.hpp"
#include "tenzor/utils/widen_narrow.hpp"

namespace {

using tenzor::DType;
using tenzor::Tensor;
using tenzor::utils::is_half_precision;
using tenzor::utils::widen_narrow_compute;

class WidenNarrowEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
static auto* const env =
    ::testing::AddGlobalTestEnvironment(new WidenNarrowEnv);

// Read every element of a tensor as double (converts via .to(Float64) so the
// returned values are exact for whatever the tensor actually holds).
std::vector<double> read_f64(const Tensor& t) {
    Tensor f = t.to(DType::Float64);
    const double* p = f.data<double>();
    return std::vector<double>(p, p + f.numel());
}

// Build a Float32 tensor from explicit values then cast to the target dtype.
Tensor make_tensor(const std::vector<float>& vals, DType dt) {
    Tensor f = tenzor::from_data(vals.data(),
                                 {static_cast<int64_t>(vals.size())});
    return (dt == DType::Float32) ? f : f.to(dt);
}

TEST(WidenNarrow, IsHalfPrecisionPredicate) {
    EXPECT_TRUE(is_half_precision(DType::Float16));
    EXPECT_TRUE(is_half_precision(DType::BFloat16));
    EXPECT_FALSE(is_half_precision(DType::Float32));
    EXPECT_FALSE(is_half_precision(DType::Float64));
    EXPECT_FALSE(is_half_precision(DType::Int32));
}

TEST(WidenNarrow, Float32IsPassthrough) {
    Tensor x = make_tensor({1.0f, 2.0f, 3.0f, 4.0f}, DType::Float32);
    DType inner_seen = DType::Bool; // sentinel
    Tensor out = widen_narrow_compute(x, [&](const Tensor& w) {
        inner_seen = w.dtype();
        // Square so a dropped/garbled value would show numerically.
        return w * w;
    });
    EXPECT_EQ(inner_seen, DType::Float32);
    EXPECT_EQ(out.dtype(), DType::Float32);
    auto v = read_f64(out);
    ASSERT_EQ(v.size(), 4u);
    EXPECT_DOUBLE_EQ(v[0], 1.0);
    EXPECT_DOUBLE_EQ(v[1], 4.0);
    EXPECT_DOUBLE_EQ(v[2], 9.0);
    EXPECT_DOUBLE_EQ(v[3], 16.0);
}

TEST(WidenNarrow, Float64IsPassthrough) {
    std::vector<double> dvals = {1.5, 2.5, 3.5, 4.5};
    Tensor x = tenzor::from_data(dvals.data(), {4});
    ASSERT_EQ(x.dtype(), DType::Float64);
    DType inner_seen = DType::Bool;
    Tensor out = widen_narrow_compute(x, [&](const Tensor& w) {
        inner_seen = w.dtype();
        return w * w;
    });
    EXPECT_EQ(inner_seen, DType::Float64);
    EXPECT_EQ(out.dtype(), DType::Float64);
    auto v = read_f64(out);
    ASSERT_EQ(v.size(), 4u);
    EXPECT_DOUBLE_EQ(v[0], 2.25);
    EXPECT_DOUBLE_EQ(v[1], 6.25);
    EXPECT_DOUBLE_EQ(v[2], 12.25);
    EXPECT_DOUBLE_EQ(v[3], 20.25);
}

TEST(WidenNarrow, Float16WidensThenNarrowsAndPreservesValues) {
    // Values exactly representable in Float16 so the only thing under test is
    // value preservation through widen->compute->narrow, not half rounding.
    Tensor x = make_tensor({1.0f, 2.0f, 4.0f, 8.0f}, DType::Float16);
    DType inner_seen = DType::Bool;
    Tensor out = widen_narrow_compute(x, [&](const Tensor& w) {
        inner_seen = w.dtype();
        return w + w;  // doubling: 2,4,8,16 — all exact in Float16
    });
    EXPECT_EQ(inner_seen, DType::Float32);   // widened for compute
    EXPECT_EQ(out.dtype(), DType::Float16);  // narrowed back
    auto v = read_f64(out);
    ASSERT_EQ(v.size(), 4u);
    EXPECT_DOUBLE_EQ(v[0], 2.0);
    EXPECT_DOUBLE_EQ(v[1], 4.0);
    EXPECT_DOUBLE_EQ(v[2], 8.0);
    EXPECT_DOUBLE_EQ(v[3], 16.0);
}

TEST(WidenNarrow, BFloat16WidensThenNarrowsAndPreservesValues) {
    Tensor x = make_tensor({1.0f, 2.0f, 4.0f, 8.0f}, DType::BFloat16);
    DType inner_seen = DType::Bool;
    Tensor out = widen_narrow_compute(x, [&](const Tensor& w) {
        inner_seen = w.dtype();
        return w + w;
    });
    EXPECT_EQ(inner_seen, DType::Float32);
    EXPECT_EQ(out.dtype(), DType::BFloat16);
    auto v = read_f64(out);
    ASSERT_EQ(v.size(), 4u);
    EXPECT_DOUBLE_EQ(v[0], 2.0);
    EXPECT_DOUBLE_EQ(v[1], 4.0);
    EXPECT_DOUBLE_EQ(v[2], 8.0);
    EXPECT_DOUBLE_EQ(v[3], 16.0);
}

TEST(WidenNarrow, NarrowsWhenFnWidensInternally) {
    // Caller passes Float16 input; the lambda widens to Float32 itself and
    // returns a Float32 tensor. The helper must still narrow back to Float16,
    // and the value must survive.
    Tensor x = make_tensor({3.0f, 5.0f, 7.0f, 9.0f}, DType::Float16);
    Tensor out = widen_narrow_compute(x, [](const Tensor& w) -> Tensor {
        return w.to(DType::Float32);
    });
    EXPECT_EQ(out.dtype(), DType::Float16);
    auto v = read_f64(out);
    ASSERT_EQ(v.size(), 4u);
    EXPECT_DOUBLE_EQ(v[0], 3.0);
    EXPECT_DOUBLE_EQ(v[1], 5.0);
    EXPECT_DOUBLE_EQ(v[2], 7.0);
    EXPECT_DOUBLE_EQ(v[3], 9.0);
}

TEST(WidenNarrow, ShapePreserved) {
    Tensor x = tenzor::ones({3, 5, 7}, DType::Float16);
    Tensor out = widen_narrow_compute(x, [](const Tensor& w) { return w; });
    ASSERT_EQ(out.shape().size(), 3u);
    EXPECT_EQ(out.shape()[0], 3);
    EXPECT_EQ(out.shape()[1], 5);
    EXPECT_EQ(out.shape()[2], 7);
}

TEST(WidenNarrow, TwoInputBothHalfWidensAndPreservesValues) {
    Tensor a = make_tensor({1.0f, 2.0f, 4.0f, 8.0f}, DType::Float16);
    Tensor b = make_tensor({2.0f, 2.0f, 2.0f, 2.0f}, DType::Float16);
    DType seen_a = DType::Bool, seen_b = DType::Bool;
    Tensor out = widen_narrow_compute(a, b, [&](const Tensor& wa, const Tensor& wb) {
        seen_a = wa.dtype();
        seen_b = wb.dtype();
        return wa * wb;  // 2,4,8,16
    });
    EXPECT_EQ(seen_a, DType::Float32);
    EXPECT_EQ(seen_b, DType::Float32);
    EXPECT_EQ(out.dtype(), DType::Float16);
    auto v = read_f64(out);
    ASSERT_EQ(v.size(), 4u);
    EXPECT_DOUBLE_EQ(v[0], 2.0);
    EXPECT_DOUBLE_EQ(v[1], 4.0);
    EXPECT_DOUBLE_EQ(v[2], 8.0);
    EXPECT_DOUBLE_EQ(v[3], 16.0);
}

TEST(WidenNarrow, TwoInputMixedHalfAndF32WidensBoth) {
    Tensor a = make_tensor({1.0f, 2.0f, 4.0f, 8.0f}, DType::BFloat16);
    Tensor b = make_tensor({1.0f, 1.0f, 1.0f, 1.0f}, DType::Float32);
    DType seen_a = DType::Bool, seen_b = DType::Bool;
    Tensor out = widen_narrow_compute(a, b, [&](const Tensor& wa, const Tensor& wb) {
        seen_a = wa.dtype();
        seen_b = wb.dtype();
        return wa + wb;  // 2,3,5,9
    });
    EXPECT_EQ(seen_a, DType::Float32);
    EXPECT_EQ(seen_b, DType::Float32);
    EXPECT_EQ(out.dtype(), DType::BFloat16);  // first input's dtype wins for narrow
    auto v = read_f64(out);
    ASSERT_EQ(v.size(), 4u);
    EXPECT_DOUBLE_EQ(v[0], 2.0);
    EXPECT_DOUBLE_EQ(v[1], 3.0);
    EXPECT_DOUBLE_EQ(v[2], 5.0);
    EXPECT_DOUBLE_EQ(v[3], 9.0);
}

// The documented silent-accumulator branch (widen_narrow.hpp:85-87): a Float64
// operand paired with a half operand MUST compute at Float64, otherwise the
// result is a Float64-typed tensor carrying only Float32 precision. This test
// uses a value that survives only at Float64 precision and fails at Float32.
TEST(WidenNarrow, TwoInputF64PlusHalfComputesAtFloat64) {
    // a is Float64 holding 1 + 2^-30 ≈ 1.0000000009313226, which rounds to
    // exactly 1.0 in Float32 but is exactly representable in Float64.
    const double epsilon = std::ldexp(1.0, -30);  // 2^-30
    std::vector<double> avals = {1.0 + epsilon};
    Tensor a = tenzor::from_data(avals.data(), {1});
    ASSERT_EQ(a.dtype(), DType::Float64);

    // b is a half-precision 1.0 (exact in Float16).
    Tensor b = make_tensor({1.0f}, DType::Float16);

    DType seen_a = DType::Bool, seen_b = DType::Bool;
    Tensor out = widen_narrow_compute(a, b, [&](const Tensor& wa, const Tensor& wb) {
        seen_a = wa.dtype();
        seen_b = wb.dtype();
        // Subtract 1 then the residual is 2^-30, which only survives if the
        // compute happened at Float64. At Float32 the residual is 0.
        return wa * wb;
    });

    // Inner compute MUST be Float64 (the highest-precision operand), not F32.
    EXPECT_EQ(seen_a, DType::Float64)
        << "Float64 operand was downcast to Float32 — silent-accumulator bug";
    EXPECT_EQ(seen_b, DType::Float64)
        << "half operand was widened only to Float32 alongside a Float64 operand";

    // Output narrows back to a's dtype (Float64) and must retain the residual.
    EXPECT_EQ(out.dtype(), DType::Float64);
    auto v = read_f64(out);
    ASSERT_EQ(v.size(), 1u);
    // The full Float64 reference value, computed independently here.
    EXPECT_NEAR(v[0], 1.0 + epsilon, 1e-15);
    // And critically: the residual above 1.0 is non-zero (it would be exactly
    // 0 if the compute had been done at Float32).
    EXPECT_GT(v[0] - 1.0, 0.0)
        << "residual lost — compute was performed at Float32 precision";
}

TEST(WidenNarrow, TwoInputNeitherHalfIsPassthrough) {
    Tensor a = make_tensor({2.0f, 3.0f, 4.0f, 5.0f}, DType::Float32);
    Tensor b = make_tensor({2.0f, 2.0f, 2.0f, 2.0f}, DType::Float32);
    DType seen_a = DType::Bool;
    Tensor out = widen_narrow_compute(a, b, [&](const Tensor& wa, const Tensor& wb) {
        seen_a = wa.dtype();
        return wa * wb;  // 4,6,8,10
    });
    EXPECT_EQ(seen_a, DType::Float32);
    EXPECT_EQ(out.dtype(), DType::Float32);
    auto v = read_f64(out);
    ASSERT_EQ(v.size(), 4u);
    EXPECT_DOUBLE_EQ(v[0], 4.0);
    EXPECT_DOUBLE_EQ(v[1], 6.0);
    EXPECT_DOUBLE_EQ(v[2], 8.0);
    EXPECT_DOUBLE_EQ(v[3], 10.0);
}

} // namespace
