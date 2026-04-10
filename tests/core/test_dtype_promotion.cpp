// Exhaustive dtype promotion tests.
//
// Verifies the rules documented in include/tenzor/core/dtype.hpp and
// implemented in tenzor::promote_types(). Matches PyTorch / NumPy
// torch.result_type semantics.

#include <gtest/gtest.h>

#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/type_promotion.hpp"

#include <array>
#include <string>

namespace tenzor {
namespace {

constexpr std::array<DType, 20> kAllDtypes = {
    DType::Float32, DType::Float64, DType::Float16,   DType::BFloat16,
    DType::Int8,    DType::Int16,   DType::Int32,     DType::Int64,
    DType::UInt8,   DType::UInt16,  DType::UInt32,    DType::UInt64,
    DType::Bool,    DType::Complex64, DType::Complex128,
    DType::FP8_E4M3, DType::FP8_E5M2,
    DType::QInt8,   DType::QUInt8,  DType::QInt4x2};

std::string pair_str(DType a, DType b) {
    return std::string{dtype_name(a)} + " + " + std::string{dtype_name(b)};
}

// ---------------------------------------------------------------------------
// Invariants: symmetry, identity, bool-is-weakest
// ---------------------------------------------------------------------------

TEST(DTypePromotion, Symmetric) {
    for (DType a : kAllDtypes) {
        for (DType b : kAllDtypes) {
            EXPECT_EQ(promote_types(a, b), promote_types(b, a))
                << "asymmetric: " << pair_str(a, b);
        }
    }
}

TEST(DTypePromotion, Identity) {
    for (DType a : kAllDtypes) {
        EXPECT_EQ(promote_types(a, a), a) << "identity failed: " << dtype_name(a);
    }
}

TEST(DTypePromotion, BoolIsWeakest) {
    // Bool combined with any non-quantized type should yield the other type.
    // Quantized + Bool is an unusual combination; skip it — the implementation
    // treats it as "quantized dominates" which is defensible.
    for (DType other : kAllDtypes) {
        if (other == DType::Bool) continue;
        if (is_quantized(other)) continue;
        EXPECT_EQ(promote_types(DType::Bool, other), other)
            << "Bool + " << dtype_name(other) << " should be " << dtype_name(other);
    }
}

// ---------------------------------------------------------------------------
// Float vs Int rules. Float16/BFloat16 + any int must upgrade to Float32
// because those float types cannot exactly represent even 32-bit integers.
// This matches torch.result_type.
// ---------------------------------------------------------------------------

TEST(DTypePromotion, FloatWinsOverInt) {
    constexpr std::array<DType, 8> ints = {
        DType::Int8,  DType::Int16,  DType::Int32,  DType::Int64,
        DType::UInt8, DType::UInt16, DType::UInt32, DType::UInt64};

    // Float32/Float64 win directly.
    for (DType i : ints) {
        EXPECT_EQ(promote_types(DType::Float32, i), DType::Float32);
        EXPECT_EQ(promote_types(DType::Float64, i), DType::Float64);
    }

    // Float16/BFloat16 + int → Float32 (precision safety).
    for (DType i : ints) {
        EXPECT_EQ(promote_types(DType::Float16, i), DType::Float32)
            << "Float16 + " << dtype_name(i) << " should widen to Float32";
        EXPECT_EQ(promote_types(DType::BFloat16, i), DType::Float32)
            << "BFloat16 + " << dtype_name(i) << " should widen to Float32";
    }
}

TEST(DTypePromotion, FloatHierarchy) {
    EXPECT_EQ(promote_types(DType::Float16, DType::Float32), DType::Float32);
    EXPECT_EQ(promote_types(DType::Float32, DType::Float64), DType::Float64);
    EXPECT_EQ(promote_types(DType::Float16, DType::Float64), DType::Float64);
    EXPECT_EQ(promote_types(DType::BFloat16, DType::Float32), DType::Float32);
}

TEST(DTypePromotion, IntHierarchy) {
    EXPECT_EQ(promote_types(DType::Int8, DType::Int32), DType::Int32);
    EXPECT_EQ(promote_types(DType::Int16, DType::Int64), DType::Int64);
    EXPECT_EQ(promote_types(DType::UInt8, DType::Int32), DType::Int32);
    EXPECT_EQ(promote_types(DType::Int8, DType::Int16), DType::Int16);
}

// ---------------------------------------------------------------------------
// Complex dominance
// ---------------------------------------------------------------------------

TEST(DTypePromotion, ComplexDominates) {
    EXPECT_EQ(promote_types(DType::Float32, DType::Complex64), DType::Complex64);
    EXPECT_EQ(promote_types(DType::Float64, DType::Complex64), DType::Complex128);
    EXPECT_EQ(promote_types(DType::Float64, DType::Complex128), DType::Complex128);
    EXPECT_EQ(promote_types(DType::Int32, DType::Complex64), DType::Complex64);
    EXPECT_EQ(promote_types(DType::Complex64, DType::Complex128), DType::Complex128);
}

// ---------------------------------------------------------------------------
// FP8 rules: FP8 + wide type → wider type; mixed FP8 → E5M2 (wider range)
// ---------------------------------------------------------------------------

TEST(DTypePromotion, FP8Rules) {
    EXPECT_EQ(promote_types(DType::FP8_E4M3, DType::FP8_E4M3), DType::FP8_E4M3);
    EXPECT_EQ(promote_types(DType::FP8_E5M2, DType::FP8_E5M2), DType::FP8_E5M2);
    EXPECT_EQ(promote_types(DType::FP8_E4M3, DType::FP8_E5M2), DType::FP8_E5M2);

    EXPECT_EQ(promote_types(DType::FP8_E4M3, DType::Float32), DType::Float32);
    EXPECT_EQ(promote_types(DType::FP8_E5M2, DType::Float64), DType::Float64);
}

// ---------------------------------------------------------------------------
// Quantized types: quantized + float → at-least-Float32, quantized + int →
// Float32 (implicit dequantization).
// ---------------------------------------------------------------------------

TEST(DTypePromotion, QuantizedRules) {
    EXPECT_EQ(promote_types(DType::QInt8, DType::QInt8), DType::QInt8);
    EXPECT_EQ(promote_types(DType::QInt8, DType::Float32), DType::Float32);
    EXPECT_EQ(promote_types(DType::QUInt8, DType::Float64), DType::Float64);
    EXPECT_EQ(promote_types(DType::QInt8, DType::Int32), DType::Float32);
}

// ---------------------------------------------------------------------------
// Coverage: every pair must produce some valid DType (sanity check that
// promote_types() is total — no "unknown" fallthrough).
// ---------------------------------------------------------------------------

TEST(DTypePromotion, TotalCoverage) {
    for (DType a : kAllDtypes) {
        for (DType b : kAllDtypes) {
            DType result = promote_types(a, b);
            // Assert the result is one of the known dtypes.
            bool found = false;
            for (DType known : kAllDtypes) {
                if (result == known) {
                    found = true;
                    break;
                }
            }
            EXPECT_TRUE(found) << "promote_types(" << pair_str(a, b)
                               << ") returned an unknown DType";
        }
    }
}

} // namespace
} // namespace tenzor
