// Phase 13 / Task A.6′ — tests for MLIR text emission helpers.

#include "tenzor/jit/mlir/mlir_text_emit.hpp"

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

namespace tj = ::tenzor::jit::mlir_jit;
using ::tenzor::DType;

TEST(MlirTextEmit, TypeName_Scalars) {
    EXPECT_EQ(tj::mlir_type_name(DType::Float32), "f32");
    EXPECT_EQ(tj::mlir_type_name(DType::Float64), "f64");
    EXPECT_EQ(tj::mlir_type_name(DType::Float16), "f16");
    EXPECT_EQ(tj::mlir_type_name(DType::BFloat16), "bf16");
    EXPECT_EQ(tj::mlir_type_name(DType::Int8), "i8");
    EXPECT_EQ(tj::mlir_type_name(DType::Int32), "i32");
    EXPECT_EQ(tj::mlir_type_name(DType::Int64), "i64");
    EXPECT_EQ(tj::mlir_type_name(DType::Bool), "i1");
    EXPECT_EQ(tj::mlir_type_name(DType::Complex64), "complex<f32>");
    EXPECT_EQ(tj::mlir_type_name(DType::Complex128), "complex<f64>");
}

TEST(MlirTextEmit, TensorType_Shapes) {
    EXPECT_EQ(tj::mlir_tensor_type({4, 8}, DType::Float32), "tensor<4x8xf32>");
    EXPECT_EQ(tj::mlir_tensor_type({}, DType::Int64), "tensor<i64>");
    EXPECT_EQ(tj::mlir_tensor_type({1}, DType::Float64), "tensor<1xf64>");
    EXPECT_EQ(tj::mlir_tensor_type({2, 3, 5, 7}, DType::BFloat16),
              "tensor<2x3x5x7xbf16>");
}

TEST(MlirTextEmit, BinaryAdd) {
    std::ostringstream os;
    tj::emit_stablehlo_binary(os, "add", "c", "a", "b", {4}, DType::Float32);
    EXPECT_EQ(os.str(), "%c = stablehlo.add %a, %b : tensor<4xf32>");
}

TEST(MlirTextEmit, BinaryMultiplyAndSubtract) {
    {
        std::ostringstream os;
        tj::emit_stablehlo_binary(os, "multiply", "out", "x", "y", {2, 3},
                                  DType::Float64);
        EXPECT_EQ(os.str(),
                  "%out = stablehlo.multiply %x, %y : tensor<2x3xf64>");
    }
    {
        std::ostringstream os;
        tj::emit_stablehlo_binary(os, "subtract", "d", "p", "q", {},
                                  DType::Int32);
        EXPECT_EQ(os.str(), "%d = stablehlo.subtract %p, %q : tensor<i32>");
    }
}

TEST(MlirTextEmit, UnaryNegate) {
    std::ostringstream os;
    tj::emit_stablehlo_unary(os, "negate", "y", "x", {8}, DType::Float32);
    EXPECT_EQ(os.str(), "%y = stablehlo.negate %x : tensor<8xf32>");
}

// JIT-R110 regression: IREE's Vulkan/SPIR-V backend miscomputes a
// dot_general whose operand comes directly from a fused stablehlo.convert
// (verified standalone: a trivial F16->F32-converted 32x32 matmul returned
// 49.0 instead of 8.0 for a uniform 0.5 input on Vulkan; llvm-cpu/cuda/rocm
// computed it correctly). stablehlo.optimization_barrier between each
// operand and the dot_general prevents that harmful fusion and is a
// semantic no-op elsewhere, so emit_stablehlo_dot_general must always emit
// it for both operands, unconditionally of dtype/target.
TEST(MlirTextEmit, DotGeneral_WrapsOperandsInOptimizationBarrier) {
    std::ostringstream os;
    tj::emit_stablehlo_dot_general(os, "c", "a", "b",
                                   /*lhs_batch=*/{0}, /*rhs_batch=*/{0},
                                   /*lhs_contracting=*/{2}, /*rhs_contracting=*/{1},
                                   /*lhs_shape=*/{4, 32, 32},
                                   /*rhs_shape=*/{4, 32, 32},
                                   /*result_shape=*/{4, 32, 32}, DType::Float32);
    EXPECT_EQ(os.str(),
              "%cba = stablehlo.optimization_barrier %a : tensor<4x32x32xf32>\n"
              "%cbb = stablehlo.optimization_barrier %b : tensor<4x32x32xf32>\n"
              "%c = stablehlo.dot_general %cba, %cbb, batching_dims = [0] x "
              "[0], contracting_dims = [2] x [1], precision = [HIGHEST, "
              "HIGHEST] : (tensor<4x32x32xf32>, tensor<4x32x32xf32>) -> "
              "tensor<4x32x32xf32>");
}

TEST(MlirTextEmit, CustomCall_SingleOperand) {
    std::ostringstream os;
    tj::emit_custom_call(os, "tenzor_rms_norm", "y", {"x"}, {{4, 16}},
                         {DType::Float32}, {4, 16}, DType::Float32,
                         "eps=1e-5");
    EXPECT_EQ(
        os.str(),
        "%y = stablehlo.custom_call @tenzor_rms_norm(%x) "
        "{backend_config = \"eps=1e-5\"} : (tensor<4x16xf32>) -> "
        "tensor<4x16xf32>");
}

TEST(MlirTextEmit, CustomCall_MultiOperand) {
    std::ostringstream os;
    tj::emit_custom_call(os, "tenzor_flash_attention", "o",
                         {"q", "k", "v"},
                         {{2, 8, 4, 16}, {2, 8, 4, 16}, {2, 8, 4, 16}},
                         {DType::Float32, DType::Float32, DType::Float32},
                         {2, 8, 4, 16}, DType::Float32, "causal=1;scale=0.5");
    EXPECT_EQ(
        os.str(),
        "%o = stablehlo.custom_call @tenzor_flash_attention(%q, %k, %v) "
        "{backend_config = \"causal=1;scale=0.5\"} : "
        "(tensor<2x8x4x16xf32>, tensor<2x8x4x16xf32>, tensor<2x8x4x16xf32>) "
        "-> tensor<2x8x4x16xf32>");
}

TEST(MlirTextEmit, CustomCall_MismatchedOperandLengths) {
    std::ostringstream os;
    EXPECT_THROW(
        tj::emit_custom_call(os, "tenzor_x", "y", {"a", "b"}, {{4}},
                             {DType::Float32, DType::Float32}, {4},
                             DType::Float32, ""),
        std::invalid_argument);
}

TEST(MlirTextEmit, ModuleWrapper_SingleArgIdentity) {
    std::ostringstream body;
    // No body statements — identity function returns arg0 directly.
    const std::string text = tj::emit_module_wrapper(
        body, {{{4}, DType::Float32}}, {{{4}, DType::Float32}}, {"arg0"});
    EXPECT_NE(text.find("module {"), std::string::npos);
    EXPECT_NE(text.find("func.func @main(%arg0: tensor<4xf32>)"),
              std::string::npos);
    EXPECT_NE(text.find("-> (tensor<4xf32>)"), std::string::npos);
    EXPECT_NE(text.find("return %arg0 : tensor<4xf32>"), std::string::npos);
}

TEST(MlirTextEmit, ModuleWrapper_AddTwoArgs) {
    std::ostringstream body;
    tj::emit_stablehlo_binary(body, "add", "sum", "arg0", "arg1", {2, 3},
                              DType::Float32);
    body << '\n';
    const std::string text = tj::emit_module_wrapper(
        body,
        {{{2, 3}, DType::Float32}, {{2, 3}, DType::Float32}},
        {{{2, 3}, DType::Float32}}, {"sum"});

    // Verify the key MLIR fragments appear in order in the resulting module.
    auto pos_module = text.find("module {");
    auto pos_func   = text.find("func.func @main(");
    auto pos_arg0   = text.find("%arg0: tensor<2x3xf32>");
    auto pos_arg1   = text.find("%arg1: tensor<2x3xf32>");
    auto pos_add    = text.find("%sum = stablehlo.add %arg0, %arg1");
    auto pos_return = text.find("return %sum : tensor<2x3xf32>");
    auto pos_close  = text.find("}");
    EXPECT_NE(pos_module, std::string::npos);
    EXPECT_NE(pos_func, std::string::npos);
    EXPECT_NE(pos_arg0, std::string::npos);
    EXPECT_NE(pos_arg1, std::string::npos);
    EXPECT_NE(pos_add, std::string::npos);
    EXPECT_NE(pos_return, std::string::npos);
    EXPECT_LT(pos_module, pos_func);
    EXPECT_LT(pos_func, pos_arg0);
    EXPECT_LT(pos_arg0, pos_arg1);
    EXPECT_LT(pos_arg1, pos_add);
    EXPECT_LT(pos_add, pos_return);
    EXPECT_LT(pos_return, pos_close);
}

TEST(MlirTextEmit, ModuleWrapper_MismatchedReturns) {
    std::ostringstream body;
    EXPECT_THROW(
        tj::emit_module_wrapper(body, {}, {{{4}, DType::Float32}}, {}),
        std::invalid_argument);
}
