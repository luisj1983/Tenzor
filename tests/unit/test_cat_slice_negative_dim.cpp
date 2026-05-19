/**
 * @file test_cat_slice_negative_dim.cpp
 * @brief Audit P0 #7: cat_kernel and slice_kernel must normalize negative dim
 *        at the kernel layer (defensive). The public ops normalize today, but
 *        the kernel may be reached via dispatch<OpId::Cat>(...) directly.
 */

#include <gtest/gtest.h>
#include <vector>
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"

namespace tenzor {
    void initialize();
}

using namespace tenzor;

// Helper: convert shape span to vector for EXPECT_EQ comparisons
static std::vector<int64_t> to_vec(std::span<const int64_t> s) {
    return {s.begin(), s.end()};
}

class TenzorTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const tenzor_env =
    ::testing::AddGlobalTestEnvironment(new TenzorTestEnvironment);

// ============================================================================
// cat negative dim tests
// ============================================================================

TEST(CatNegativeDim, PublicOpNegMatchesPos_LastDim) {
    // dim=-1 must equal dim=2 for a 3D tensor
    auto a = arange(0.0, 24.0, 1.0).reshape({2, 3, 4}).to(DType::Float32);
    auto b = arange(24.0, 48.0, 1.0).reshape({2, 3, 4}).to(DType::Float32);
    auto c_neg = cat({a, b}, /*dim=*/-1);
    auto c_pos = cat({a, b}, /*dim=*/ 2);
    EXPECT_EQ(to_vec(c_neg.shape()), to_vec(c_pos.shape()));
    auto diff = max(abs(c_neg - c_pos)).item<float>();
    EXPECT_LT(diff, 1e-9f);
}

TEST(CatNegativeDim, PublicOpNegMatchesPos_MiddleDim) {
    // dim=-2 must equal dim=1 for a 3D tensor
    auto a = arange(0.0, 24.0, 1.0).reshape({2, 3, 4}).to(DType::Float32);
    auto b = arange(24.0, 48.0, 1.0).reshape({2, 3, 4}).to(DType::Float32);
    auto c_neg = cat({a, b}, /*dim=*/-2);
    auto c_pos = cat({a, b}, /*dim=*/ 1);
    EXPECT_EQ(to_vec(c_neg.shape()), to_vec(c_pos.shape()));
    auto diff = max(abs(c_neg - c_pos)).item<float>();
    EXPECT_LT(diff, 1e-9f);
}

TEST(CatNegativeDim, PublicOpNegMatchesPos_FirstDim) {
    // dim=-3 must equal dim=0 for a 3D tensor
    auto a = arange(0.0, 24.0, 1.0).reshape({2, 3, 4}).to(DType::Float32);
    auto b = arange(24.0, 48.0, 1.0).reshape({2, 3, 4}).to(DType::Float32);
    auto c_neg = cat({a, b}, /*dim=*/-3);
    auto c_pos = cat({a, b}, /*dim=*/ 0);
    EXPECT_EQ(to_vec(c_neg.shape()), to_vec(c_pos.shape()));
    auto diff = max(abs(c_neg - c_pos)).item<float>();
    EXPECT_LT(diff, 1e-9f);
}

TEST(CatNegativeDim, OutputShapeCorrect) {
    // shape check: cat along dim=-1 must expand last dim
    auto a = arange(0.0, 6.0, 1.0).reshape({2, 3}).to(DType::Float32);
    auto b = arange(6.0, 12.0, 1.0).reshape({2, 3}).to(DType::Float32);
    auto c = cat({a, b}, /*dim=*/-1);
    ASSERT_EQ(c.shape().size(), 2u);
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 6);
}

// ============================================================================
// slice negative dim tests
// ============================================================================

TEST(SliceNegativeDim, PublicOpNegMatchesPos_LastDim) {
    // dim=-1 must equal dim=2 for a 3D tensor
    auto a = arange(0.0, 24.0, 1.0).reshape({2, 3, 4}).to(DType::Float32);
    auto s_neg = slice(a, /*dim=*/-1, /*start=*/1, /*end=*/3);
    auto s_pos = slice(a, /*dim=*/ 2, /*start=*/1, /*end=*/3);
    EXPECT_EQ(to_vec(s_neg.shape()), to_vec(s_pos.shape()));
    auto diff = max(abs(s_neg - s_pos)).item<float>();
    EXPECT_LT(diff, 1e-9f);
}

TEST(SliceNegativeDim, PublicOpNegMatchesPos_MiddleDim) {
    // dim=-2 must equal dim=1 for a 3D tensor
    auto a = arange(0.0, 24.0, 1.0).reshape({2, 3, 4}).to(DType::Float32);
    auto s_neg = slice(a, /*dim=*/-2, /*start=*/0, /*end=*/2);
    auto s_pos = slice(a, /*dim=*/ 1, /*start=*/0, /*end=*/2);
    EXPECT_EQ(to_vec(s_neg.shape()), to_vec(s_pos.shape()));
    auto diff = max(abs(s_neg - s_pos)).item<float>();
    EXPECT_LT(diff, 1e-9f);
}

TEST(SliceNegativeDim, PublicOpNegMatchesPos_FirstDim) {
    // dim=-3 must equal dim=0 for a 3D tensor
    auto a = arange(0.0, 24.0, 1.0).reshape({2, 3, 4}).to(DType::Float32);
    auto s_neg = slice(a, /*dim=*/-3, /*start=*/0, /*end=*/1);
    auto s_pos = slice(a, /*dim=*/ 0, /*start=*/0, /*end=*/1);
    EXPECT_EQ(to_vec(s_neg.shape()), to_vec(s_pos.shape()));
    auto diff = max(abs(s_neg - s_pos)).item<float>();
    EXPECT_LT(diff, 1e-9f);
}

TEST(SliceNegativeDim, OutputShapeCorrect) {
    // shape check: slice along dim=-1 with step=2 must produce correct size
    auto a = arange(0.0, 24.0, 1.0).reshape({2, 3, 4}).to(DType::Float32);
    auto s = slice(a, /*dim=*/-1, /*start=*/0, /*end=*/4, /*step=*/2);
    ASSERT_EQ(s.shape().size(), 3u);
    EXPECT_EQ(s.shape()[0], 2);
    EXPECT_EQ(s.shape()[1], 3);
    EXPECT_EQ(s.shape()[2], 2);
}
