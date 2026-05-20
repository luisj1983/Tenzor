/**
 * @file test_masked_tensor.cpp
 * @brief Regression tests for MaskedTensor — audit item A.7.
 *
 * Two correctness gaps in the previous implementation:
 *  - max() / min() always filled masked positions with float infinities,
 *    overflowing or wrapping on integer dtypes.
 *  - Binary plus / minus / mul / div computed @code data_ op other.data_
 *    @endcode over masked positions too, allowing NaN/Inf to bleed into
 *    the result's data buffer (visible to a user calling data()).
 */

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "tenzor/core/dtype.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/core/masked_tensor.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/tenzor.hpp"

using namespace tenzor;

namespace {

class MaskedTensorTest : public ::testing::Test {
protected:
    void SetUp() override { tenzor::initialize(); }

    static Tensor make_bool(std::vector<int64_t> shape,
                            std::vector<bool> vals) {
        Tensor t(shape, DType::Bool, Device::cpu());
        // bool storage convention: 1 byte per element on CPU.
        auto* p = static_cast<uint8_t*>(t.data_ptr());
        for (size_t i = 0; i < vals.size(); ++i) {
            p[i] = vals[i] ? 1 : 0;
        }
        return t;
    }
};

// ---------------------------------------------------------------------------
// A.7 — Integer-dtype max() must not overflow on the ±inf sentinel.
// ---------------------------------------------------------------------------
TEST_F(MaskedTensorTest, IntegerMaxRespectsDtype) {
    Tensor data({4}, DType::Int32, Device::cpu());
    auto* dp = data.data<int32_t>();
    dp[0] = 7; dp[1] = -5; dp[2] = 12; dp[3] = -100;

    Tensor mask = make_bool({4}, {true, true, false, true});

    MaskedTensor mt(data, mask);
    Tensor maxv = mt.max();
    ASSERT_EQ(maxv.dtype(), DType::Int32);
    // Max over unmasked positions {7, -5, -100} is 7.
    EXPECT_EQ(maxv.item<int32_t>(), 7);
}

TEST_F(MaskedTensorTest, IntegerMinRespectsDtype) {
    Tensor data({4}, DType::Int32, Device::cpu());
    auto* dp = data.data<int32_t>();
    dp[0] = 7; dp[1] = -5; dp[2] = 12; dp[3] = -100;

    Tensor mask = make_bool({4}, {true, true, false, false});

    MaskedTensor mt(data, mask);
    Tensor minv = mt.min();
    ASSERT_EQ(minv.dtype(), DType::Int32);
    // Min over unmasked positions {7, -5} is -5.
    EXPECT_EQ(minv.item<int32_t>(), -5);
}

// ---------------------------------------------------------------------------
// A.7 — Binary ops must zero out masked positions of the result so
//       NaN/Inf produced at masked positions doesn't bleed into data().
// ---------------------------------------------------------------------------
TEST_F(MaskedTensorTest, BinaryOpsZeroMaskedPositions) {
    // a / b where b[2] = 0 (masked) — division by zero yields ±inf in the
    // raw tensor; the masked-tensor binary op must zero that position so
    // the user-visible data buffer is clean.
    Tensor a({4}, DType::Float32, Device::cpu());
    Tensor b({4}, DType::Float32, Device::cpu());
    auto* ap = a.data<float>();
    auto* bp = b.data<float>();
    ap[0] = 1.0f; ap[1] = 4.0f; ap[2] = 9.0f; ap[3] = 16.0f;
    bp[0] = 1.0f; bp[1] = 2.0f; bp[2] = 0.0f; bp[3] = 4.0f;

    // Mask out position 2 (where b is zero).
    Tensor mask_a = make_bool({4}, {true, true, false, true});
    Tensor mask_b = make_bool({4}, {true, true, false, true});

    MaskedTensor A(a, mask_a);
    MaskedTensor B(b, mask_b);

    MaskedTensor C = A / B;
    const auto& cd = C.data();
    ASSERT_EQ(cd.dtype(), DType::Float32);

    const float* cp = cd.data<float>();
    EXPECT_FLOAT_EQ(cp[0], 1.0f / 1.0f);
    EXPECT_FLOAT_EQ(cp[1], 4.0f / 2.0f);
    // Position 2 was masked in both inputs ⇒ the result data must be 0,
    // not Inf or NaN (and the mask says invalid).
    EXPECT_FALSE(std::isinf(cp[2])) << "masked position leaked Inf: " << cp[2];
    EXPECT_FALSE(std::isnan(cp[2])) << "masked position leaked NaN: " << cp[2];
    EXPECT_FLOAT_EQ(cp[2], 0.0f);
    EXPECT_FLOAT_EQ(cp[3], 16.0f / 4.0f);
}

TEST_F(MaskedTensorTest, BinaryOpsCleanForAllVariants) {
    Tensor a({3}, DType::Float32, Device::cpu());
    Tensor b({3}, DType::Float32, Device::cpu());
    auto* ap = a.data<float>();
    auto* bp = b.data<float>();
    // Position 1 will be masked; put something that would produce NaN
    // under sqrt-then-divide so we are sure the data buffer stays clean.
    ap[0] = 1.0f; ap[1] = std::numeric_limits<float>::infinity(); ap[2] = 3.0f;
    bp[0] = 4.0f; bp[1] = 0.0f; bp[2] = 2.0f;

    Tensor mask = make_bool({3}, {true, false, true});
    MaskedTensor A(a, mask);
    MaskedTensor B(b, mask);

    auto check_clean = [](const MaskedTensor& mt, int64_t i,
                          const char* label) {
        const float v = mt.data().data<float>()[i];
        EXPECT_FALSE(std::isinf(v)) << label << " leaked Inf at i=" << i;
        EXPECT_FALSE(std::isnan(v)) << label << " leaked NaN at i=" << i;
        EXPECT_FLOAT_EQ(v, 0.0f) << label << " masked slot not zeroed";
    };

    check_clean(A + B, 1, "operator+");
    check_clean(A - B, 1, "operator-");
    check_clean(A * B, 1, "operator*");
    check_clean(A / B, 1, "operator/");
}

}  // namespace
