/**
 * @file test_mutation_version_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for TensorImpl mutation version tracking
 */

#define TENZOR_INTERNAL 1

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/core/tensor.hpp>
#include <tenzor/ops/creation.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class MutationVersionMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(MutationVersionMultiDTypeTest, MutableShapeBumpsVersion) {
    Tensor t = zeros({3, 4}, dtype(), device());
    auto v0 = t.version();
    auto& s = t.mutable_shape();
    (void)s;
    EXPECT_EQ(t.version(), v0 + 1);
}

TEST_P(MutationVersionMultiDTypeTest, MutableStridesBumpsVersion) {
    Tensor t = zeros({3, 4}, dtype(), device());
    auto v0 = t.version();
    auto& s = t.mutable_strides();
    (void)s;
    EXPECT_EQ(t.version(), v0 + 1);
}

TEST_P(MutationVersionMultiDTypeTest, SetOffsetBumpsVersion) {
    Tensor t = zeros({3, 4}, dtype(), device());
    auto v0 = t.version();
    t.set_offset(0);
    EXPECT_EQ(t.version(), v0 + 1);
}

TEST_P(MutationVersionMultiDTypeTest, MultipleMutationsBumpMultipleTimes) {
    Tensor t = zeros({3, 4}, dtype(), device());
    auto v0 = t.version();
    (void)t.mutable_shape();
    (void)t.mutable_strides();
    t.set_offset(0);
    EXPECT_EQ(t.version(), v0 + 3);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(MutationVersionMultiDTypeTest);
