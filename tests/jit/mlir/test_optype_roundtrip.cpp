#include <gtest/gtest.h>
#include "tenzor/jit/tracer.hpp"

namespace tj = ::tenzor::jit;

TEST(OpTypeRoundtrip, NewMVP1Ops) {
    for (auto op : {tj::OpType::SiLU, tj::OpType::Where, tj::OpType::Stack,
                    tj::OpType::Broadcast, tj::OpType::IndexSelect,
                    tj::OpType::RMSNorm, tj::OpType::GQA, tj::OpType::RoPE,
                    tj::OpType::Padding, tj::OpType::Interpolate}) {
        auto s = tj::op_type_to_string(op);
        EXPECT_NE(s, "Unknown") << "op " << static_cast<int>(op) << " has no string";
        EXPECT_EQ(tj::string_to_op_type(s), op);
    }
}

TEST(OpTypeRoundtrip, ElementwiseTrigAndRsqrt) {
    for (auto op : {tj::OpType::Sin, tj::OpType::Cos, tj::OpType::Rsqrt}) {
        auto s = tj::op_type_to_string(op);
        EXPECT_NE(s, "Unknown") << "op " << static_cast<int>(op) << " has no string";
        EXPECT_EQ(tj::string_to_op_type(s), op);
    }
}

TEST(OpTypeRoundtrip, C2MVP2Ops) {
    // These 13 ops are handled by the interceptor/executor but previously had no
    // op_type_to_string case, collapsing to "Unknown" and breaking
    // find_nodes_by_type / DOT / histogram labels (JIT-F007).
    for (auto op : {tj::OpType::Var, tj::OpType::Std, tj::OpType::Prod,
                    tj::OpType::LeakyReLU, tj::OpType::ELU, tj::OpType::Mish,
                    tj::OpType::Softplus, tj::OpType::GroupNorm,
                    tj::OpType::InstanceNorm, tj::OpType::Gather,
                    tj::OpType::Scatter, tj::OpType::Flip, tj::OpType::Roll}) {
        auto s = tj::op_type_to_string(op);
        EXPECT_NE(s, "Unknown") << "op " << static_cast<int>(op) << " has no string";
        EXPECT_EQ(tj::string_to_op_type(s), op);
    }
}
