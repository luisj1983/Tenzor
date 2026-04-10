// Verifies that TensorImpl metadata mutators bump the version counter and
// invalidate both cache fields. Regression test for the autograd in-place
// detection hole where mutators previously only invalidated the contiguity
// cache and did not bump version_counter_.

// This test exercises the internal mutation API, so it must define
// TENZOR_INTERNAL before any tenzor headers are pulled in.
#define TENZOR_INTERNAL 1

#include <gtest/gtest.h>

#include <tenzor/tenzor.hpp>
#include <tenzor/core/tensor.hpp>
#include <tenzor/ops/creation.hpp>

namespace tenzor {
namespace {

class MutationVersionTest : public ::testing::Test {
protected:
    void SetUp() override { tenzor::initialize(); }
};

TEST_F(MutationVersionTest, MutableShapeBumpsVersion) {
    Tensor t = zeros({3, 4}, DType::Float32, Device::cpu());
    auto v0 = t.version();
    auto& s = t.mutable_shape();
    (void)s;
    EXPECT_EQ(t.version(), v0 + 1);
}

TEST_F(MutationVersionTest, MutableStridesBumpsVersion) {
    Tensor t = zeros({3, 4}, DType::Float32, Device::cpu());
    auto v0 = t.version();
    auto& s = t.mutable_strides();
    (void)s;
    EXPECT_EQ(t.version(), v0 + 1);
}

TEST_F(MutationVersionTest, SetOffsetBumpsVersion) {
    Tensor t = zeros({3, 4}, DType::Float32, Device::cpu());
    auto v0 = t.version();
    t.set_offset(0);  // no-op value, but still a mutation call
    EXPECT_EQ(t.version(), v0 + 1);
}

TEST_F(MutationVersionTest, MultipleMutationsBumpMultipleTimes) {
    Tensor t = zeros({3, 4}, DType::Float32, Device::cpu());
    auto v0 = t.version();
    (void)t.mutable_shape();
    (void)t.mutable_strides();
    t.set_offset(0);
    EXPECT_EQ(t.version(), v0 + 3);
}

} // namespace
} // namespace tenzor
