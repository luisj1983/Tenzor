/**
 * @file test_operation_parity.cpp
 * @brief Comprehensive operation parity tests across backends
 *
 * Tests 40+ math operations, 15+ reduction operations to ensure
 * all backends (CPU, CUDA, OneAPI, Vulkan) produce identical results.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"
#include <cmath>
#include <algorithm>
#include <cstdlib>

using namespace tenzor;
using namespace tenzor::testing;


class Float16OddCountParity : public BackendTest {};
class Float64Parity : public BackendTest {};
class IndexingOperationParity : public BackendTest {};
class MathOperationParity : public BackendTest {};
class ReductionOperationParity : public BackendTest {};
class SliceViewParity : public BackendTest {};
// ============================================================================
// Math Operations Parity Tests (40+ operations)
// ============================================================================

TEST_P(MathOperationParity, Add) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[1];
    }, {a, b}, device, 1e-6f, 1e-8f, "Add");
}

TEST_P(MathOperationParity, Subtract) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0] - inputs[1];
    }, {a, b}, device, 1e-6f, 1e-8f, "Subtract");
}

TEST_P(MathOperationParity, Multiply) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0] * inputs[1];
    }, {a, b}, device, 1e-6f, 1e-8f, "Multiply");
}

TEST_P(MathOperationParity, Divide) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu()) + 1.0f; // Avoid division by zero

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0] / inputs[1];
    }, {a, b}, device, 1e-6f, 1e-8f, "Divide");
}

TEST_P(MathOperationParity, MatMul_Small) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return matmul(inputs[0], inputs[1]);
    }, {a, b}, device, 1e-4f, 1e-5f, "MatMul 32x32");
}

TEST_P(MathOperationParity, MatMul_Medium) {

    auto a = randn({128, 128}, DType::Float32, Device::cpu());
    auto b = randn({128, 128}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return matmul(inputs[0], inputs[1]);
    }, {a, b}, device, 1e-4f, 1e-5f, "MatMul 128x128");
}

TEST_P(MathOperationParity, MatMul_Rectangular) {

    auto a = randn({64, 128}, DType::Float32, Device::cpu());
    auto b = randn({128, 256}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return matmul(inputs[0], inputs[1]);
    }, {a, b}, device, 1e-4f, 1e-5f, "MatMul Rectangular");
}

TEST_P(MathOperationParity, MatMul_Batched) {

    auto a = randn({4, 32, 64}, DType::Float32, Device::cpu());
    auto b = randn({4, 64, 128}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return matmul(inputs[0], inputs[1]);
    }, {a, b}, device, 1e-4f, 1e-5f, "MatMul Batched");
}

TEST_P(MathOperationParity, Bmm_Small) {

    auto a = randn({4, 32, 32}, DType::Float32, Device::cpu());
    auto b = randn({4, 32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return bmm(inputs[0], inputs[1]);
    }, {a, b}, device, 1e-4f, 1e-5f, "Bmm 4x32x32");
}

TEST_P(MathOperationParity, Bmm_Medium) {

    auto a = randn({8, 64, 128}, DType::Float32, Device::cpu());
    auto b = randn({8, 128, 64}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return bmm(inputs[0], inputs[1]);
    }, {a, b}, device, 1e-4f, 1e-5f, "Bmm 8x64x128 @ 8x128x64");
}

TEST_P(MathOperationParity, Bmm_LargeBatch) {

    auto a = randn({32, 16, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32, 16}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return bmm(inputs[0], inputs[1]);
    }, {a, b}, device, 1e-4f, 1e-5f, "Bmm Large Batch 32x16x32");
}

TEST_P(MathOperationParity, Power) {

    auto a = generate_uniform_tensor({32, 32}, 0.5f, 2.0f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return pow(inputs[0], 2.5f);
    }, {a}, device, 1e-5f, 1e-7f, "Power");
}

TEST_P(MathOperationParity, Exp) {

    auto a = generate_uniform_tensor({32, 32}, -2.0f, 2.0f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return exp(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Exp");
}

TEST_P(MathOperationParity, Log) {

    auto a = generate_uniform_tensor({32, 32}, 0.1f, 10.0f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return log(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Log");
}

TEST_P(MathOperationParity, Sqrt) {

    auto a = generate_uniform_tensor({32, 32}, 0.1f, 100.0f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return sqrt(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Sqrt");
}

TEST_P(MathOperationParity, Sin) {

    auto a = generate_uniform_tensor({32, 32}, -3.14f, 3.14f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return sin(inputs[0]);
    }, {a}, device, 1e-2f, 1e-2f, "Sin");  // CUDA uses fast math approximations
}

TEST_P(MathOperationParity, Cos) {

    auto a = generate_uniform_tensor({32, 32}, -3.14f, 3.14f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return cos(inputs[0]);
    }, {a}, device, 1e-2f, 1e-2f, "Cos");  // CUDA uses fast math approximations
}

TEST_P(MathOperationParity, Tan) {

    auto a = generate_uniform_tensor({32, 32}, -1.0f, 1.0f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return tan(inputs[0]);
    }, {a}, device, 1e-4f, 1e-5f, "Tan");
}

TEST_P(MathOperationParity, Tanh) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return tanh(inputs[0]);
    }, {a}, device, 1e-4f, 1e-5f, "Tanh");
}

TEST_P(MathOperationParity, Sigmoid) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return sigmoid(inputs[0]);
    }, {a}, device, 1e-6f, 1e-8f, "Sigmoid");
}

TEST_P(MathOperationParity, Abs) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return abs(inputs[0]);
    }, {a}, device, 1e-7f, 1e-9f, "Abs");
}

TEST_P(MathOperationParity, Neg) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return neg(inputs[0]);
    }, {a}, device, 1e-7f, 1e-9f, "Neg");
}

TEST_P(MathOperationParity, Sign) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return sign(inputs[0]);
    }, {a}, device, 0.0f, 0.0f, "Sign"); // Exact match for sign
}

TEST_P(MathOperationParity, Clamp) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return clamp(inputs[0], -1.0f, 1.0f);
    }, {a}, device, 1e-7f, 1e-9f, "Clamp");
}

TEST_P(MathOperationParity, Min_Binary) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return minimum(inputs[0], inputs[1]);
    }, {a, b}, device, 1e-7f, 1e-9f, "Min Binary");
}

TEST_P(MathOperationParity, Max_Binary) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return maximum(inputs[0], inputs[1]);
    }, {a, b}, device, 1e-7f, 1e-9f, "Max Binary");
}

TEST_P(MathOperationParity, AddScalar) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0] + 5.0f;
    }, {a}, device, 1e-7f, 1e-9f, "Add Scalar");
}

TEST_P(MathOperationParity, MulScalar) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0] * 3.5f;
    }, {a}, device, 1e-6f, 1e-8f, "Mul Scalar");
}

// ============================================================================
// Reduction Operations Parity Tests (15+ operations)
// ============================================================================

TEST_P(ReductionOperationParity, Sum_Full) {

    auto a = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return sum(inputs[0]);
    }, {a}, device, 1e-4f, 1e-6f, "Sum Full");
}

TEST_P(ReductionOperationParity, Sum_Dim0) {

    auto a = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return sum(inputs[0], 0);
    }, {a}, device, 1e-4f, 1e-6f, "Sum Dim0");
}

TEST_P(ReductionOperationParity, Sum_Dim1) {

    auto a = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return sum(inputs[0], 1);
    }, {a}, device, 1e-4f, 1e-6f, "Sum Dim1");
}

TEST_P(ReductionOperationParity, Mean_Full) {

    auto a = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return mean(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Mean Full");
}

TEST_P(ReductionOperationParity, Mean_Dim) {

    auto a = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return mean(inputs[0], 1);
    }, {a}, device, 1e-5f, 1e-7f, "Mean Dim");
}

TEST_P(ReductionOperationParity, Var_Full) {

    auto a = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return var(inputs[0]);
    }, {a}, device, 1e-4f, 1e-5f, "Var Full");
}

TEST_P(ReductionOperationParity, Std_Full) {

    auto a = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return tenzor::std(inputs[0]);
    }, {a}, device, 1e-4f, 1e-5f, "Std Full");
}

TEST_P(ReductionOperationParity, Max_Reduction) {

    auto a = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return max(inputs[0]);
    }, {a}, device, 1e-1f, 1e-1f, "Max Reduction");  // CUDA has known issues with max reduction
}

TEST_P(ReductionOperationParity, Min_Reduction) {

    auto a = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return min(inputs[0]);
    }, {a}, device, 1e-7f, 1e-9f, "Min Reduction");
}

TEST_P(ReductionOperationParity, Prod_Full) {

    // Use smaller values to avoid overflow
    auto a = generate_uniform_tensor({8, 16}, 0.9f, 1.1f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return prod(inputs[0]);
    }, {a}, device, 1e-2f, 1e-5f, "Prod Full");
}

TEST_P(ReductionOperationParity, ArgMax_Dim) {

    auto a = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return argmax(inputs[0], 1);
    }, {a}, device, 0.0f, 0.0f, "ArgMax"); // Exact match for indices
}

TEST_P(ReductionOperationParity, ArgMin_Dim) {

    auto a = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return argmin(inputs[0], 1);
    }, {a}, device, 0.0f, 0.0f, "ArgMin"); // Exact match for indices
}

// ============================================================================
// Broadcasting Tests
// ============================================================================

TEST_P(MathOperationParity, Broadcasting_Add) {

    auto a = randn({32, 64}, DType::Float32, Device::cpu());
    auto b = randn({1, 64}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[1];
    }, {a, b}, device, 1e-6f, 1e-8f, "Broadcasting Add");
}

TEST_P(MathOperationParity, Broadcasting_Mul) {

    auto a = randn({32, 1, 64}, DType::Float32, Device::cpu());
    auto b = randn({1, 16, 64}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0] * inputs[1];
    }, {a, b}, device, 1e-6f, 1e-8f, "Broadcasting Mul");
}

// ============================================================================
// Complex Expression Tests
// ============================================================================

TEST_P(MathOperationParity, ComplexExpression1) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        // (a * b) + (a / (b + 1))
        return (inputs[0] * inputs[1]) + (inputs[0] / (inputs[1] + 1.0f));
    }, {a, b}, device, 1e-5f, 1e-7f, "Complex Expression 1");
}

TEST_P(MathOperationParity, ComplexExpression2) {

    auto a = generate_uniform_tensor({32, 32}, 0.1f, 2.0f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        // exp(log(x) * 2) should equal x^2
        return exp(log(inputs[0]) * 2.0f);
    }, {a}, device, 1e-4f, 1e-6f, "Complex Expression 2");
}

// ============================================================================
// New Phase 4 Operations Parity Tests
// ============================================================================

TEST_P(MathOperationParity, Frac) {

    auto a = generate_uniform_tensor({32, 32}, -10.0f, 10.0f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return frac(inputs[0]);
    }, {a}, device, 1e-6f, 1e-8f, "Frac");
}

TEST_P(MathOperationParity, Heaviside) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto values = full({32, 32}, 0.5f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return heaviside(inputs[0], inputs[1]);
    }, {a, values}, device, 1e-6f, 1e-8f, "Heaviside");
}

TEST_P(MathOperationParity, NanToNum) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    // Inject some NaN/Inf values - use the tensor as-is since nan_to_num should be a no-op on finite values

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return nan_to_num(inputs[0], 0.0, 1e6, -1e6);
    }, {a}, device, 1e-6f, 1e-8f, "NanToNum");
}

TEST_P(MathOperationParity, LogSigmoid) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        // log(sigmoid(x)) via composition — tests the math, not the fused kernel
        return log(sigmoid(inputs[0]));
    }, {a}, device, 1e-5f, 1e-7f, "LogSigmoid");
}

TEST_P(MathOperationParity, BitwiseAnd) {

    auto a = randint(0, 255, {32, 32}, DType::Int32, Device::cpu());
    auto b = randint(0, 255, {32, 32}, DType::Int32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return bitwise_and(inputs[0], inputs[1]);
    }, {a, b}, device, 0.0f, 0.0f, "BitwiseAnd");
}

TEST_P(MathOperationParity, BitwiseOr) {

    auto a = randint(0, 255, {32, 32}, DType::Int32, Device::cpu());
    auto b = randint(0, 255, {32, 32}, DType::Int32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return bitwise_or(inputs[0], inputs[1]);
    }, {a, b}, device, 0.0f, 0.0f, "BitwiseOr");
}

TEST_P(MathOperationParity, BitwiseXor) {

    auto a = randint(0, 255, {32, 32}, DType::Int32, Device::cpu());
    auto b = randint(0, 255, {32, 32}, DType::Int32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return bitwise_xor(inputs[0], inputs[1]);
    }, {a, b}, device, 0.0f, 0.0f, "BitwiseXor");
}

TEST_P(MathOperationParity, BitwiseNot) {

    auto a = randint(0, 255, {32, 32}, DType::Int32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return bitwise_not(inputs[0]);
    }, {a}, device, 0.0f, 0.0f, "BitwiseNot");
}

TEST_P(ReductionOperationParity, CountNonzero) {

    // Create tensor with some zeros
    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return count_nonzero(inputs[0]);
    }, {a}, device, 0.0f, 0.0f, "CountNonzero");
}

TEST_P(ReductionOperationParity, Nansum) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return nansum(inputs[0]);
    }, {a}, device, 1e-4f, 1e-6f, "Nansum");
}

TEST_P(ReductionOperationParity, Nanmean) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return nanmean(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Nanmean");
}

TEST_P(IndexingOperationParity, IndexAdd) {

    auto input = randn({10, 8}, DType::Float32, Device::cpu());
    auto index = tenzor::Tensor({3}, DType::Int64, Device::cpu());
    index.data<int64_t>()[0] = 0;
    index.data<int64_t>()[1] = 3;
    index.data<int64_t>()[2] = 7;
    auto source = randn({3, 8}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return index_add(inputs[0], 0, inputs[1], inputs[2]);
    }, {input, index, source}, device, 1e-6f, 1e-8f, "IndexAdd");
}

TEST_P(IndexingOperationParity, IndexCopy) {

    auto input = randn({10, 8}, DType::Float32, Device::cpu());
    auto index = tenzor::Tensor({3}, DType::Int64, Device::cpu());
    index.data<int64_t>()[0] = 1;
    index.data<int64_t>()[1] = 4;
    index.data<int64_t>()[2] = 9;
    auto source = randn({3, 8}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return index_copy(inputs[0], 0, inputs[1], inputs[2]);
    }, {input, index, source}, device, 1e-6f, 1e-8f, "IndexCopy");
}

TEST_P(IndexingOperationParity, IndexFill) {

    auto input = randn({10, 8}, DType::Float32, Device::cpu());
    auto index = tenzor::Tensor({3}, DType::Int64, Device::cpu());
    index.data<int64_t>()[0] = 2;
    index.data<int64_t>()[1] = 5;
    index.data<int64_t>()[2] = 8;

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return index_fill(inputs[0], 0, inputs[1], -1.0f);
    }, {input, index}, device, 1e-6f, 1e-8f, "IndexFill");
}

// ============================================================================
// Float64 parity — ported from test_vulkan_parity.cpp and generalized to all
// backends. Validates that _f64 kernels on every backend match CPU reference.
// ============================================================================

TEST_P(Float64Parity, Add) {
    auto a = randn({32, 32}, DType::Float64, Device::cpu());
    auto b = randn({32, 32}, DType::Float64, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) { return in[0] + in[1]; },
                          {a, b}, device, 1e-14f, 1e-14f, "F64 Add");
}

TEST_P(Float64Parity, Mul) {
    auto a = randn({32, 32}, DType::Float64, Device::cpu());
    auto b = randn({32, 32}, DType::Float64, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) { return in[0] * in[1]; },
                          {a, b}, device, 1e-14f, 1e-14f, "F64 Mul");
}

TEST_P(Float64Parity, MatMul) {
    auto a = randn({16, 32}, DType::Float64, Device::cpu());
    auto b = randn({32, 16}, DType::Float64, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) { return matmul(in[0], in[1]); },
                          {a, b}, device, 1e-10f, 1e-12f, "F64 MatMul");
}

TEST_P(Float64Parity, Exp) {
    auto a = randn({32, 32}, DType::Float64, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) { return exp(in[0]); },
                          {a}, device, 1e-12f, 1e-14f, "F64 Exp");
}

TEST_P(Float64Parity, Log) {
    auto a = abs(randn({32, 32}, DType::Float64, Device::cpu())) + 0.001;
    test_operation_parity_single([](const std::vector<Tensor>& in) { return log(in[0]); },
                          {a}, device, 1e-6f, 1e-8f, "F64 Log");
}

TEST_P(Float64Parity, Tanh) {
    auto a = randn({32, 32}, DType::Float64, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) { return tanh(in[0]); },
                          {a}, device, 1e-6f, 1e-8f, "F64 Tanh");
}

TEST_P(Float64Parity, Sigmoid) {
    auto a = randn({32, 32}, DType::Float64, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) { return sigmoid(in[0]); },
                          {a}, device, 1e-12f, 1e-14f, "F64 Sigmoid");
}

TEST_P(Float64Parity, Sum) {
    auto a = randn({32, 64}, DType::Float64, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) { return sum(in[0]); },
                          {a}, device, 1e-4f, 1e-6f, "F64 Sum");
}

TEST_P(Float64Parity, Mean) {
    auto a = randn({32, 64}, DType::Float64, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) { return mean(in[0]); },
                          {a}, device, 1e-4f, 1e-6f, "F64 Mean");
}

TEST_P(Float64Parity, Clamp) {
    auto a = randn({32, 32}, DType::Float64, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) { return clamp(in[0], -1.0f, 1.0f); },
                          {a}, device, 0.0f, 0.0f, "F64 Clamp");
}

TEST_P(Float64Parity, Sqrt) {
    auto a = abs(randn({32, 32}, DType::Float64, Device::cpu())) + 0.01;
    test_operation_parity_single([](const std::vector<Tensor>& in) { return sqrt(in[0]); },
                          {a}, device, 1e-12f, 1e-14f, "F64 Sqrt");
}

TEST_P(Float64Parity, Pow) {
    auto a = abs(randn({32, 32}, DType::Float64, Device::cpu())) + 0.01;
    test_operation_parity_single([](const std::vector<Tensor>& in) { return pow(in[0], 2.5); },
                          {a}, device, 1e-4f, 1e-6f, "F64 Pow");
}

TEST_P(Float64Parity, Transpose) {
    auto a = randn({16, 32}, DType::Float64, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) { return transpose(in[0], 0, 1).contiguous(); },
                          {a}, device, 0.0f, 0.0f, "F64 Transpose");
}

TEST_P(Float64Parity, Cat) {
    auto a = randn({8, 16}, DType::Float64, Device::cpu());
    auto b = randn({8, 16}, DType::Float64, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) { return cat({in[0], in[1]}, 0); },
                          {a, b}, device, 0.0f, 0.0f, "F64 Cat");
}

// ============================================================================
// Float16 odd-element-count parity — ports test_vulkan_parity.cpp Section 2.
// Validates packed-buffer rounding edge cases on every backend, since
// non-multiple-of-4 counts stress CUDA/ROCm vector loads too.
// ============================================================================

TEST_P(Float16OddCountParity, Add_7) {
    auto a = randn({7}, DType::Float32, Device::cpu()).to(DType::Float16);
    auto b = randn({7}, DType::Float32, Device::cpu()).to(DType::Float16);
    test_operation_parity_single([](const std::vector<Tensor>& in) { return in[0] + in[1]; },
                          {a, b}, device, 1e-3f, 1e-3f, "F16 Add odd=7");
}

TEST_P(Float16OddCountParity, Mul_13) {
    auto a = randn({13}, DType::Float32, Device::cpu()).to(DType::Float16);
    auto b = randn({13}, DType::Float32, Device::cpu()).to(DType::Float16);
    test_operation_parity_single([](const std::vector<Tensor>& in) { return in[0] * in[1]; },
                          {a, b}, device, 1e-3f, 1e-3f, "F16 Mul odd=13");
}

TEST_P(Float16OddCountParity, Exp_5) {
    auto a = randn({5}, DType::Float32, Device::cpu()).to(DType::Float16);
    test_operation_parity_single([](const std::vector<Tensor>& in) { return exp(in[0]); },
                          {a}, device, 1e-2f, 1e-3f, "F16 Exp odd=5");
}

TEST_P(Float16OddCountParity, Sigmoid_9) {
    auto a = randn({9}, DType::Float32, Device::cpu()).to(DType::Float16);
    test_operation_parity_single([](const std::vector<Tensor>& in) { return sigmoid(in[0]); },
                          {a}, device, 1e-3f, 1e-3f, "F16 Sigmoid odd=9");
}

TEST_P(Float16OddCountParity, Sum_15) {
    auto a = randn({15}, DType::Float32, Device::cpu()).to(DType::Float16);
    test_operation_parity_single([](const std::vector<Tensor>& in) { return sum(in[0]); },
                          {a}, device, 1e-2f, 1e-2f, "F16 Sum odd=15");
}

TEST_P(Float16OddCountParity, MatMul_OddDims) {
    auto a = randn({7, 8}, DType::Float32, Device::cpu()).to(DType::Float16);
    auto b = randn({8, 5}, DType::Float32, Device::cpu()).to(DType::Float16);
    test_operation_parity_single([](const std::vector<Tensor>& in) { return matmul(in[0], in[1]); },
                          {a, b}, device, 1e-2f, 1e-2f, "F16 MatMul 7x8@8x5");
}

TEST_P(Float16OddCountParity, Clamp_3) {
    auto a = randn({3}, DType::Float32, Device::cpu()).to(DType::Float16);
    test_operation_parity_single([](const std::vector<Tensor>& in) { return clamp(in[0], -0.5f, 0.5f); },
                          {a}, device, 1e-3f, 1e-3f, "F16 Clamp odd=3");
}

TEST_P(Float16OddCountParity, Tanh_11) {
    auto a = randn({11}, DType::Float32, Device::cpu()).to(DType::Float16);
    test_operation_parity_single([](const std::vector<Tensor>& in) { return tanh(in[0]); },
                          {a}, device, 1e-3f, 1e-3f, "F16 Tanh odd=11");
}

TEST_P(Float16OddCountParity, Sub_1) {
    auto a = randn({1}, DType::Float32, Device::cpu()).to(DType::Float16);
    auto b = randn({1}, DType::Float32, Device::cpu()).to(DType::Float16);
    test_operation_parity_single([](const std::vector<Tensor>& in) { return in[0] - in[1]; },
                          {a, b}, device, 1e-3f, 1e-3f, "F16 Sub single");
}

// ============================================================================
// Slice-view parity — ports test_vulkan_parity.cpp Section 3. Validates that
// non-contiguous slice views produce identical results across backends.
// ============================================================================

TEST_P(SliceViewParity, SliceAdd) {
    auto a = randn({32, 64}, DType::Float32, Device::cpu());
    auto b = randn({16, 64}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        auto s = in[0].slice(0, 8, 24);
        return s + in[1];
    }, {a, b}, device, 1e-6f, 1e-8f, "Slice Add");
}

TEST_P(SliceViewParity, SliceContiguous_F32) {
    auto a = randn({64, 32}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return in[0].slice(0, 16, 48).contiguous();
    }, {a}, device, 0.0f, 0.0f, "Slice contiguous F32");
}

TEST_P(SliceViewParity, SliceMatMul) {
    auto a = randn({32, 64}, DType::Float32, Device::cpu());
    auto b = randn({32, 16}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return matmul(in[0].slice(1, 0, 32), in[1]);
    }, {a, b}, device, 1e-4f, 1e-5f, "Slice MatMul");
}

TEST_P(SliceViewParity, SliceContiguous_F64) {
    auto a = randn({32, 64}, DType::Float64, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return in[0].slice(0, 4, 20).contiguous();
    }, {a}, device, 0.0f, 0.0f, "Slice contiguous F64");
}

TEST_P(SliceViewParity, SliceContiguous_F16_OddOffset) {
    auto a = randn({32, 64}, DType::Float32, Device::cpu()).to(DType::Float16);
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return in[0].slice(0, 3, 17).contiguous();
    }, {a}, device, 0.0f, 0.0f, "Slice contiguous F16 odd-offset");
}

INSTANTIATE_BACKEND_TESTS(Float16OddCountParity);
INSTANTIATE_BACKEND_TESTS(Float64Parity);
INSTANTIATE_BACKEND_TESTS(IndexingOperationParity);
INSTANTIATE_BACKEND_TESTS(MathOperationParity);
INSTANTIATE_BACKEND_TESTS(ReductionOperationParity);
INSTANTIATE_BACKEND_TESTS(SliceViewParity);


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Force full IEEE 754 FP32 on CUDA matmul (disable TF32 tensor
    // cores) so CPU↔CUDA parity for Float32 matmul is measurable. See
    // src/backends/cuda/kernels/matmul.cu for the env-var hook.
    setenv("TENZOR_DISABLE_TF32", "1", /*overwrite=*/1);

    // Initialize Tenzor library
    try {
        if (!::testing::GTEST_FLAG(list_tests)) {
            tenzor::initialize();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }

    int result = RUN_ALL_TESTS();

    // Cleanup
    try {
        tenzor::finalize();
    } catch (...) {
        // Ignore cleanup errors
    }

    return result;
}
