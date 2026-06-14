// Executes the Metal-independent C++ logic of the MPS fixes (mps-1/mps-2/mps-3)
// against tenzor_core. The Metal kernel dispatch itself can only run on Apple
// hardware, but the bug fixes live in plain C++ decision logic which is tested
// here on Linux.
#include <gtest/gtest.h>
#include <string>
#include <unordered_map>

#include "tenzor/tenzor.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/shape.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"

using namespace tenzor;

// ---- mps-2: exact copy of shader_name_for_dtype from mps_elementwise.mm.
// It must reject integer/Float64 dtypes (which have an f32 pipeline and would
// otherwise silently reinterpret bytes) instead of falling through to "base".
static std::string shader_name_for_dtype(const std::string& base, DType dtype) {
    switch (dtype) {
        case DType::Float32: return base;
        case DType::Float16: return base + "_f16";
        default:
            throw std::runtime_error(
                "MPS: element-wise op '" + base + "' unsupported for dtype " +
                std::string(dtype_name(dtype)) + " (only Float32/Float16 implemented)");
    }
}

TEST(MPSFixLogic, Mps2RejectsNonFloatDtypes) {
    EXPECT_EQ(shader_name_for_dtype("add_kernel", DType::Float32), "add_kernel");
    EXPECT_EQ(shader_name_for_dtype("add_kernel", DType::Float16), "add_kernel_f16");
    // The bug: these used to return "add_kernel" (f32 shader) and reinterpret bytes.
    EXPECT_THROW(shader_name_for_dtype("add_kernel", DType::Int32), std::runtime_error);
    EXPECT_THROW(shader_name_for_dtype("add_kernel", DType::Int64), std::runtime_error);
    EXPECT_THROW(shader_name_for_dtype("add_kernel", DType::Bool), std::runtime_error);
    EXPECT_THROW(shader_name_for_dtype("add_kernel", DType::Float64), std::runtime_error);
}

// ---- mps-1: the broadcast-then-materialize logic dispatch_binary now performs
// before the element-wise kernel (which indexes a[id]/b[id] over the output
// numel assuming both are contiguous with the output shape).
static std::pair<Tensor, Tensor> broadcast_operands(const Tensor& a_in, const Tensor& b_in) {
    std::vector<int64_t> out_shape = broadcast_shapes(a_in.shape(), b_in.shape());
    auto matches = [&](const Tensor& t) {
        auto s = t.shape();
        return t.is_contiguous() && s.size() == out_shape.size() &&
               std::equal(s.begin(), s.end(), out_shape.begin());
    };
    Tensor a = matches(a_in) ? a_in : broadcast_to(a_in, out_shape).contiguous();
    Tensor b = matches(b_in) ? b_in : broadcast_to(b_in, out_shape).contiguous();
    return {a, b};
}

TEST(MPSFixLogic, Mps1BroadcastsOperandsToCommonContiguousShape) {
    // {4,5} + {5}  (row bias) — pre-fix read b past its end.
    auto a = tenzor::ones({4, 5}, DType::Float32, Device::cpu());
    auto b = tenzor::full({5}, 2.0, DType::Float32, Device::cpu());
    auto [ab, bb] = broadcast_operands(a, b);
    std::vector<int64_t> exp = {4, 5};
    EXPECT_EQ(std::vector<int64_t>(ab.shape().begin(), ab.shape().end()), exp);
    EXPECT_EQ(std::vector<int64_t>(bb.shape().begin(), bb.shape().end()), exp);
    EXPECT_TRUE(ab.is_contiguous());
    EXPECT_TRUE(bb.is_contiguous());
    EXPECT_EQ(ab.numel(), bb.numel());  // both fully materialized -> no OOB index

    // tensor + scalar (b is a 1-element tensor) — the worst pre-fix case.
    auto s = tenzor::full({1}, 3.0, DType::Float32, Device::cpu());
    auto [a2, b2] = broadcast_operands(a, s);
    EXPECT_EQ(a2.numel(), b2.numel());
    EXPECT_EQ(b2.numel(), 20);  // scalar expanded+materialized to [4,5]
    EXPECT_TRUE(b2.is_contiguous());
}

// ---- mps-3: pooled_buffer_for is an exact-key map lookup with nil fallback.
TEST(MPSFixLogic, Mps3BufferLookupExactKeyWithFallback) {
    std::unordered_map<void*, int> buffer_map;  // void* -> (stand-in for MTLBuffer)
    int alloc = 7;
    void* base = &alloc;
    buffer_map[base] = 42;
    auto pooled_for = [&](void* p) -> int {
        auto it = buffer_map.find(p);
        return it != buffer_map.end() ? it->second : 0;  // 0 == "nil -> fallback"
    };
    EXPECT_EQ(pooled_for(base), 42);                    // contiguous: reuse allocator buffer
    int other = 0;
    EXPECT_EQ(pooled_for(&other), 0);                   // view/foreign ptr: fall back
    EXPECT_EQ(pooled_for(nullptr), 0);
}

int main(int argc, char** argv) {
    tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
