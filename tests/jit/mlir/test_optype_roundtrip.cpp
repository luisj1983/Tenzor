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
