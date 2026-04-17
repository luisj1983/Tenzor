// Multi-backend multi-dtype tests for dtype promotion rules.
//
// Verifies that promote_types() results are consistent and that actual
// tensor operations produce correctly promoted dtypes across all backends.

#include "../multi_backend_dtype_fixture.hpp"

#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/type_promotion.hpp"

#include <array>

namespace tenzor {
namespace testing {

class DTypePromotionMultiDTypeTest : public MultiBackendDTypeTest {};

// ---------------------------------------------------------------------------
// Verify promote_types() symmetry holds regardless of backend/dtype context
// ---------------------------------------------------------------------------

TEST_P(DTypePromotionMultiDTypeTest, SymmetryInvariant) {
    constexpr std::array<DType, 8> dtypes = {
        DType::Float32, DType::Float64, DType::Float16, DType::BFloat16,
        DType::Int8, DType::Int16, DType::Int32, DType::Int64};

    for (DType a : dtypes) {
        for (DType b : dtypes) {
            EXPECT_EQ(promote_types(a, b), promote_types(b, a))
                << "asymmetric: " << dtype_name(a) << " + " << dtype_name(b);
        }
    }
}

TEST_P(DTypePromotionMultiDTypeTest, IdentityInvariant) {
    constexpr std::array<DType, 8> dtypes = {
        DType::Float32, DType::Float64, DType::Float16, DType::BFloat16,
        DType::Int8, DType::Int16, DType::Int32, DType::Int64};

    for (DType a : dtypes) {
        EXPECT_EQ(promote_types(a, a), a)
            << "identity failed: " << dtype_name(a);
    }
}

// ---------------------------------------------------------------------------
// Verify float hierarchy on actual tensors created on the test device
// ---------------------------------------------------------------------------

TEST_P(DTypePromotionMultiDTypeTest, FloatHierarchyOnDevice) {
    EXPECT_EQ(promote_types(DType::Float16, DType::Float32), DType::Float32);
    EXPECT_EQ(promote_types(DType::Float32, DType::Float64), DType::Float64);
    EXPECT_EQ(promote_types(DType::Float16, DType::Float64), DType::Float64);
    EXPECT_EQ(promote_types(DType::BFloat16, DType::Float32), DType::Float32);
}

// ---------------------------------------------------------------------------
// Verify that adding tensors of different dtypes on the test device produces
// the promoted dtype.
// ---------------------------------------------------------------------------

TEST_P(DTypePromotionMultiDTypeTest, BinaryOpPromotionOnDevice) {
    // Create a Float32 tensor and the parameterized-dtype tensor, add them.
    auto a = tenzor::ones({4}, DType::Float32, device());
    auto b = tenzor::ones({4}, dtype(), device());
    auto result = a + b;

    DType expected = promote_types(DType::Float32, dtype());
    EXPECT_EQ(result.dtype(), expected)
        << "Expected promoted dtype " << static_cast<int>(expected)
        << " but got " << static_cast<int>(result.dtype());
    EXPECT_EQ(result.device().type, device().type);
}

// ---------------------------------------------------------------------------
// Float vs int: Float16/BFloat16 + int must widen to Float32
// ---------------------------------------------------------------------------

TEST_P(DTypePromotionMultiDTypeTest, FloatWinsOverInt) {
    constexpr std::array<DType, 4> ints = {
        DType::Int8, DType::Int16, DType::Int32, DType::Int64};

    for (DType i : ints) {
        EXPECT_EQ(promote_types(DType::Float32, i), DType::Float32);
        EXPECT_EQ(promote_types(DType::Float64, i), DType::Float64);
        EXPECT_EQ(promote_types(DType::Float16, i), DType::Float32)
            << "Float16 + " << dtype_name(i) << " should widen to Float32";
        EXPECT_EQ(promote_types(DType::BFloat16, i), DType::Float32)
            << "BFloat16 + " << dtype_name(i) << " should widen to Float32";
    }
}

// ---------------------------------------------------------------------------
// Bool is weakest: Bool + any float -> that float
// ---------------------------------------------------------------------------

TEST_P(DTypePromotionMultiDTypeTest, BoolIsWeakest) {
    EXPECT_EQ(promote_types(DType::Bool, dtype()), dtype());
}

// ---------------------------------------------------------------------------
// Tensor creation with parameterized dtype on parameterized device
// ---------------------------------------------------------------------------

TEST_P(DTypePromotionMultiDTypeTest, TensorCreationPreservesDType) {
    auto t = tenzor::zeros({3, 4}, dtype(), device());
    EXPECT_EQ(t.dtype(), dtype());
    EXPECT_EQ(t.device().type, device().type);
    EXPECT_EQ(t.numel(), 12);
}

// ---------------------------------------------------------------------------
// dtype conversion round-trip on device
// ---------------------------------------------------------------------------

TEST_P(DTypePromotionMultiDTypeTest, DTypeConversionRoundTrip) {
    // Skip Float16 for this precision-sensitive test
    if (dtype() == DType::Float16) {
        GTEST_SKIP() << "Float16 round-trip loses too much precision";
    }

    auto original = tenzor::ones({4}, DType::Float32, device());
    auto converted = original.to(dtype());
    EXPECT_EQ(converted.dtype(), dtype());

    auto back = converted.to(DType::Float32);
    EXPECT_EQ(back.dtype(), DType::Float32);

    // Check values survived the round-trip
    auto cpu = back.to(Device::cpu()).to(DType::Float32);
    auto* data = cpu.data<float>();
    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(data[i], 1.0f, atol());
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(DTypePromotionMultiDTypeTest);

} // namespace testing
} // namespace tenzor
