// test_logical_predicates_nan_parity.cpp
//
// JIT-R192: test_op_logical_predicates.cpp (tests/jit/mlir/) only compares
// CPU-eager against the IREE-MLIR-*compiled* result for the comparison
// predicates (Eq/Ne/Lt/Le/Gt/Ge), and only with plain finite inputs
// ({1,2,3,2} vs 2.0). Neither a NaN-comparison case nor the NATIVE
// per-backend eager comparison kernels (comparison.cpp/comparison.hip.cpp/
// comparison.comp/etc.) were ever exercised cross-backend directly -- a
// real divergence in a native eager comparison kernel (e.g. a backend using
// a bitwise/integer trick that mishandles NaN, or an off-by-one on the
// Le/Ge boundary) would not be caught by that suite.
//
// This test dispatches OpId::Eq/Ne/Lt/Le/Gt/Ge directly (bypassing MLIR/IREE
// entirely) on every available backend, with an input that includes NaN,
// and verifies every backend agrees with CPU on both the ordinary and the
// NaN-involving comparisons. Per IEEE 754, every ordered comparison
// (eq/lt/le/gt/ge) involving NaN must be false, and ne must be true.

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <tenzor/tenzor.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/backend/op_attributes.hpp>
#include <tenzor/ops/op_id.hpp>

#include "parity_test_utils.hpp"

using namespace tenzor;

namespace {

class LogicalPredicatesNanParity : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
};

// x = [1, 2, 3, NaN]; y = [2, 2, 2, 2]. Index 3 exercises every comparison
// op against a NaN operand; indices 0-2 are an ordinary sanity control.
auto make_x() -> Tensor {
    Tensor t({4}, DType::Float32, Device::cpu());
    auto* p = t.data<float>();
    p[0] = 1.0f; p[1] = 2.0f; p[2] = 3.0f;
    p[3] = std::numeric_limits<float>::quiet_NaN();
    return t;
}

auto make_y() -> Tensor {
    return full({4}, 2.0f, DType::Float32, Device::cpu());
}

void check_op_matches_cpu_with_nan(OpId op, const char* name,
                                    bool expected_nan_result) {
    Tensor x_cpu = make_x();
    Tensor y_cpu = make_y();

    OpAttributes attrs;
    std::vector<Tensor> cpu_inputs = {x_cpu, y_cpu};
    auto cpu_out = dispatch(op, cpu_inputs, attrs);
    ASSERT_GE(cpu_out.size(), 1u) << name;
    Tensor cpu_result = cpu_out[0].to(DType::Bool);
    auto* cpu_p = cpu_result.data<bool>();

    // Sanity: confirm the CPU reference itself honors IEEE-754 NaN
    // semantics before using it as ground truth for other backends.
    ASSERT_EQ(cpu_p[3], expected_nan_result)
        << name << ": CPU reference itself has wrong NaN-comparison "
                   "semantics -- test assumption invalid";

    for (auto dev_type : {Device::Type::CUDA, Device::Type::ROCm,
                          Device::Type::Vulkan, Device::Type::OneAPI}) {
        Device dev{dev_type, 0};
        if (!tenzor::testing::is_backend_available(dev_type, 0)) continue;
        std::vector<Tensor> ins = {x_cpu.to(dev), y_cpu.to(dev)};
        std::vector<Tensor> outs;
        try {
            outs = dispatch(op, ins, attrs);
        } catch (const std::exception& e) {
            ADD_FAILURE() << name << " threw on "
                          << tenzor::testing::device_type_to_backend_name(dev_type)
                          << ": " << e.what();
            continue;
        }
        ASSERT_GE(outs.size(), 1u) << name;
        Tensor result = outs[0].to(Device::cpu()).to(DType::Bool);
        auto* rp = result.data<bool>();
        for (int64_t i = 0; i < 4; ++i) {
            EXPECT_EQ(cpu_p[i], rp[i])
                << name << " on " << tenzor::testing::device_type_to_backend_name(dev_type)
                << " diverges from CPU at elem " << i
                << " (elem 3 is the NaN-comparison case)";
        }
    }
}

}  // namespace

// eq(NaN, 2.0) must be false.
TEST_F(LogicalPredicatesNanParity, EqWithNanMatchesAcrossBackends) {
    check_op_matches_cpu_with_nan(OpId::Eq, "eq", /*expected_nan_result=*/false);
}

// ne(NaN, 2.0) must be TRUE -- the one comparison where NaN yields true.
TEST_F(LogicalPredicatesNanParity, NeWithNanMatchesAcrossBackends) {
    check_op_matches_cpu_with_nan(OpId::Ne, "ne", /*expected_nan_result=*/true);
}

TEST_F(LogicalPredicatesNanParity, LtWithNanMatchesAcrossBackends) {
    check_op_matches_cpu_with_nan(OpId::Lt, "lt", /*expected_nan_result=*/false);
}

TEST_F(LogicalPredicatesNanParity, LeWithNanMatchesAcrossBackends) {
    check_op_matches_cpu_with_nan(OpId::Le, "le", /*expected_nan_result=*/false);
}

TEST_F(LogicalPredicatesNanParity, GtWithNanMatchesAcrossBackends) {
    check_op_matches_cpu_with_nan(OpId::Gt, "gt", /*expected_nan_result=*/false);
}

TEST_F(LogicalPredicatesNanParity, GeWithNanMatchesAcrossBackends) {
    check_op_matches_cpu_with_nan(OpId::Ge, "ge", /*expected_nan_result=*/false);
}
