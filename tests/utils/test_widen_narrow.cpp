// Tests for include/tenzor/utils/widen_narrow.hpp
//
// Verify the helper:
//   - widens Float16/BFloat16 to Float32 before invoking the callable
//   - narrows the result back to the input dtype
//   - is a passthrough for Float32/Float64 inputs
//   - still narrows when the callable widens internally (consistency)
//   - works correctly in the two-input form when either input is half

#include <gtest/gtest.h>

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

TEST(WidenNarrow, IsHalfPrecisionPredicate) {
    EXPECT_TRUE(is_half_precision(DType::Float16));
    EXPECT_TRUE(is_half_precision(DType::BFloat16));
    EXPECT_FALSE(is_half_precision(DType::Float32));
    EXPECT_FALSE(is_half_precision(DType::Float64));
    EXPECT_FALSE(is_half_precision(DType::Int32));
}

TEST(WidenNarrow, Float32IsPassthrough) {
    Tensor x = tenzor::ones({4}, DType::Float32);
    DType inner_seen = DType::Bool; // sentinel
    Tensor out = widen_narrow_compute(x, [&](const Tensor& w) {
        inner_seen = w.dtype();
        return w;
    });
    EXPECT_EQ(inner_seen, DType::Float32);
    EXPECT_EQ(out.dtype(), DType::Float32);
}

TEST(WidenNarrow, Float64IsPassthrough) {
    Tensor x = tenzor::ones({4}, DType::Float64);
    DType inner_seen = DType::Bool;
    Tensor out = widen_narrow_compute(x, [&](const Tensor& w) {
        inner_seen = w.dtype();
        return w;
    });
    EXPECT_EQ(inner_seen, DType::Float64);
    EXPECT_EQ(out.dtype(), DType::Float64);
}

TEST(WidenNarrow, Float16WidensThenNarrows) {
    Tensor x = tenzor::ones({4}, DType::Float16);
    DType inner_seen = DType::Bool;
    Tensor out = widen_narrow_compute(x, [&](const Tensor& w) {
        inner_seen = w.dtype();
        return w;
    });
    EXPECT_EQ(inner_seen, DType::Float32);   // widened for compute
    EXPECT_EQ(out.dtype(), DType::Float16);  // narrowed back
}

TEST(WidenNarrow, BFloat16WidensThenNarrows) {
    Tensor x = tenzor::ones({4}, DType::BFloat16);
    DType inner_seen = DType::Bool;
    Tensor out = widen_narrow_compute(x, [&](const Tensor& w) {
        inner_seen = w.dtype();
        return w;
    });
    EXPECT_EQ(inner_seen, DType::Float32);
    EXPECT_EQ(out.dtype(), DType::BFloat16);
}

TEST(WidenNarrow, NarrowsWhenFnWidensInternally) {
    // Caller passes Float16 input; the lambda widens to Float32 itself and
    // returns a Float32 tensor. The helper must still narrow back to Float16.
    Tensor x = tenzor::ones({4}, DType::Float16);
    Tensor out = widen_narrow_compute(x, [](const Tensor& w) -> Tensor {
        return w.to(DType::Float32);
    });
    EXPECT_EQ(out.dtype(), DType::Float16);
}

TEST(WidenNarrow, ShapePreserved) {
    Tensor x = tenzor::ones({3, 5, 7}, DType::Float16);
    Tensor out = widen_narrow_compute(x, [](const Tensor& w) { return w; });
    ASSERT_EQ(out.shape().size(), 3u);
    EXPECT_EQ(out.shape()[0], 3);
    EXPECT_EQ(out.shape()[1], 5);
    EXPECT_EQ(out.shape()[2], 7);
}

TEST(WidenNarrow, TwoInputBothHalfWidens) {
    Tensor a = tenzor::ones({4}, DType::Float16);
    Tensor b = tenzor::ones({4}, DType::Float16);
    DType seen_a = DType::Bool, seen_b = DType::Bool;
    Tensor out = widen_narrow_compute(a, b, [&](const Tensor& wa, const Tensor& wb) {
        seen_a = wa.dtype();
        seen_b = wb.dtype();
        return wa;
    });
    EXPECT_EQ(seen_a, DType::Float32);
    EXPECT_EQ(seen_b, DType::Float32);
    EXPECT_EQ(out.dtype(), DType::Float16);
}

TEST(WidenNarrow, TwoInputMixedHalfAndF32WidensBoth) {
    Tensor a = tenzor::ones({4}, DType::BFloat16);
    Tensor b = tenzor::ones({4}, DType::Float32);
    DType seen_a = DType::Bool, seen_b = DType::Bool;
    Tensor out = widen_narrow_compute(a, b, [&](const Tensor& wa, const Tensor& wb) {
        seen_a = wa.dtype();
        seen_b = wb.dtype();
        return wa;
    });
    EXPECT_EQ(seen_a, DType::Float32);
    EXPECT_EQ(seen_b, DType::Float32);
    EXPECT_EQ(out.dtype(), DType::BFloat16);  // first input's dtype wins for narrow
}

TEST(WidenNarrow, TwoInputNeitherHalfIsPassthrough) {
    Tensor a = tenzor::ones({4}, DType::Float32);
    Tensor b = tenzor::ones({4}, DType::Float32);
    DType seen_a = DType::Bool;
    Tensor out = widen_narrow_compute(a, b, [&](const Tensor& wa, const Tensor& /*wb*/) {
        seen_a = wa.dtype();
        return wa;
    });
    EXPECT_EQ(seen_a, DType::Float32);
    EXPECT_EQ(out.dtype(), DType::Float32);
}

} // namespace
