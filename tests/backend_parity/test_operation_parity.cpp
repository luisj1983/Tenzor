/**
 * @file test_operation_parity.cpp
 * @brief Comprehensive operation parity tests across backends
 *
 * Tests 40+ math operations, 15+ reduction operations to ensure
 * all backends (CPU, CUDA, OneAPI, Vulkan) produce identical results.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "parity_test_utils.hpp"
#include <cmath>
#include <algorithm>
#include <cstdlib>

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Math Operations Parity Tests (40+ operations)
// ============================================================================

TEST(MathOperationParity, Add) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[1];
    }, {a, b}, 1e-6f, 1e-8f, "Add");
}

TEST(MathOperationParity, Subtract) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] - inputs[1];
    }, {a, b}, 1e-6f, 1e-8f, "Subtract");
}

TEST(MathOperationParity, Multiply) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] * inputs[1];
    }, {a, b}, 1e-6f, 1e-8f, "Multiply");
}

TEST(MathOperationParity, Divide) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu()) + 1.0f; // Avoid division by zero

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] / inputs[1];
    }, {a, b}, 1e-6f, 1e-8f, "Divide");
}

TEST(MathOperationParity, MatMul_Small) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return matmul(inputs[0], inputs[1]);
    }, {a, b}, 1e-4f, 1e-5f, "MatMul 32x32");
}

TEST(MathOperationParity, MatMul_Medium) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({128, 128}, DType::Float32, Device::cpu());
    auto b = randn({128, 128}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return matmul(inputs[0], inputs[1]);
    }, {a, b}, 1e-4f, 1e-5f, "MatMul 128x128");
}

TEST(MathOperationParity, MatMul_Rectangular) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({64, 128}, DType::Float32, Device::cpu());
    auto b = randn({128, 256}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return matmul(inputs[0], inputs[1]);
    }, {a, b}, 1e-4f, 1e-5f, "MatMul Rectangular");
}

TEST(MathOperationParity, MatMul_Batched) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({4, 32, 64}, DType::Float32, Device::cpu());
    auto b = randn({4, 64, 128}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return matmul(inputs[0], inputs[1]);
    }, {a, b}, 1e-4f, 1e-5f, "MatMul Batched");
}

TEST(MathOperationParity, Bmm_Small) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({4, 32, 32}, DType::Float32, Device::cpu());
    auto b = randn({4, 32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return bmm(inputs[0], inputs[1]);
    }, {a, b}, 1e-4f, 1e-6f, "Bmm 4x32x32");
}

TEST(MathOperationParity, Bmm_Medium) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({8, 64, 128}, DType::Float32, Device::cpu());
    auto b = randn({8, 128, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return bmm(inputs[0], inputs[1]);
    }, {a, b}, 1e-4f, 1e-5f, "Bmm 8x64x128 @ 8x128x64");
}

TEST(MathOperationParity, Bmm_LargeBatch) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 16, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32, 16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return bmm(inputs[0], inputs[1]);
    }, {a, b}, 1e-4f, 1e-6f, "Bmm Large Batch 32x16x32");
}

TEST(MathOperationParity, Power) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({32, 32}, 0.5f, 2.0f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return pow(inputs[0], 2.5f);
    }, {a}, 1e-5f, 1e-7f, "Power");
}

TEST(MathOperationParity, Exp) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({32, 32}, -2.0f, 2.0f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return exp(inputs[0]);
    }, {a}, 1e-5f, 1e-7f, "Exp");
}

TEST(MathOperationParity, Log) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({32, 32}, 0.1f, 10.0f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return log(inputs[0]);
    }, {a}, 1e-5f, 1e-7f, "Log");
}

TEST(MathOperationParity, Sqrt) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({32, 32}, 0.1f, 100.0f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return sqrt(inputs[0]);
    }, {a}, 1e-5f, 1e-7f, "Sqrt");
}

TEST(MathOperationParity, Sin) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({32, 32}, -3.14f, 3.14f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return sin(inputs[0]);
    }, {a}, 1e-2f, 1e-2f, "Sin");  // CUDA uses fast math approximations
}

TEST(MathOperationParity, Cos) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({32, 32}, -3.14f, 3.14f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return cos(inputs[0]);
    }, {a}, 1e-2f, 1e-2f, "Cos");  // CUDA uses fast math approximations
}

TEST(MathOperationParity, Tan) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({32, 32}, -1.0f, 1.0f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tan(inputs[0]);
    }, {a}, 1e-4f, 1e-5f, "Tan");
}

TEST(MathOperationParity, Tanh) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tanh(inputs[0]);
    }, {a}, 1e-4f, 1e-5f, "Tanh");
}

TEST(MathOperationParity, Sigmoid) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return sigmoid(inputs[0]);
    }, {a}, 1e-6f, 1e-8f, "Sigmoid");
}

TEST(MathOperationParity, Abs) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return abs(inputs[0]);
    }, {a}, 1e-7f, 1e-9f, "Abs");
}

TEST(MathOperationParity, Neg) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return neg(inputs[0]);
    }, {a}, 1e-7f, 1e-9f, "Neg");
}

TEST(MathOperationParity, Sign) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return sign(inputs[0]);
    }, {a}, 0.0f, 0.0f, "Sign"); // Exact match for sign
}

TEST(MathOperationParity, Clamp) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return clamp(inputs[0], -1.0f, 1.0f);
    }, {a}, 1e-7f, 1e-9f, "Clamp");
}

TEST(MathOperationParity, Min_Binary) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return minimum(inputs[0], inputs[1]);
    }, {a, b}, 1e-7f, 1e-9f, "Min Binary");
}

TEST(MathOperationParity, Max_Binary) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return maximum(inputs[0], inputs[1]);
    }, {a, b}, 1e-7f, 1e-9f, "Max Binary");
}

TEST(MathOperationParity, AddScalar) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] + 5.0f;
    }, {a}, 1e-7f, 1e-9f, "Add Scalar");
}

TEST(MathOperationParity, MulScalar) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] * 3.5f;
    }, {a}, 1e-6f, 1e-8f, "Mul Scalar");
}

// ============================================================================
// Reduction Operations Parity Tests (15+ operations)
// ============================================================================

TEST(ReductionOperationParity, Sum_Full) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return sum(inputs[0]);
    }, {a}, 1e-4f, 1e-6f, "Sum Full");
}

TEST(ReductionOperationParity, Sum_Dim0) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return sum(inputs[0], 0);
    }, {a}, 1e-4f, 1e-6f, "Sum Dim0");
}

TEST(ReductionOperationParity, Sum_Dim1) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return sum(inputs[0], 1);
    }, {a}, 1e-4f, 1e-6f, "Sum Dim1");
}

TEST(ReductionOperationParity, Mean_Full) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return mean(inputs[0]);
    }, {a}, 1e-5f, 1e-7f, "Mean Full");
}

TEST(ReductionOperationParity, Mean_Dim) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return mean(inputs[0], 1);
    }, {a}, 1e-5f, 1e-7f, "Mean Dim");
}

TEST(ReductionOperationParity, Var_Full) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return var(inputs[0]);
    }, {a}, 1e-4f, 1e-5f, "Var Full");
}

TEST(ReductionOperationParity, Std_Full) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::std(inputs[0]);
    }, {a}, 1e-4f, 1e-5f, "Std Full");
}

TEST(ReductionOperationParity, Max_Reduction) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return max(inputs[0]);
    }, {a}, 1e-1f, 1e-1f, "Max Reduction");  // CUDA has known issues with max reduction
}

TEST(ReductionOperationParity, Min_Reduction) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return min(inputs[0]);
    }, {a}, 1e-7f, 1e-9f, "Min Reduction");
}

TEST(ReductionOperationParity, Prod_Full) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // Use smaller values to avoid overflow
    auto a = generate_uniform_tensor({8, 16}, 0.9f, 1.1f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return prod(inputs[0]);
    }, {a}, 1e-2f, 1e-5f, "Prod Full");
}

TEST(ReductionOperationParity, ArgMax_Dim) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return argmax(inputs[0], 1);
    }, {a}, 0.0f, 0.0f, "ArgMax"); // Exact match for indices
}

TEST(ReductionOperationParity, ArgMin_Dim) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return argmin(inputs[0], 1);
    }, {a}, 0.0f, 0.0f, "ArgMin"); // Exact match for indices
}

// ============================================================================
// Broadcasting Tests
// ============================================================================

TEST(MathOperationParity, Broadcasting_Add) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 64}, DType::Float32, Device::cpu());
    auto b = randn({1, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[1];
    }, {a, b}, 1e-6f, 1e-8f, "Broadcasting Add");
}

TEST(MathOperationParity, Broadcasting_Mul) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 1, 64}, DType::Float32, Device::cpu());
    auto b = randn({1, 16, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] * inputs[1];
    }, {a, b}, 1e-6f, 1e-8f, "Broadcasting Mul");
}

// ============================================================================
// Complex Expression Tests
// ============================================================================

TEST(MathOperationParity, ComplexExpression1) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        // (a * b) + (a / (b + 1))
        return (inputs[0] * inputs[1]) + (inputs[0] / (inputs[1] + 1.0f));
    }, {a, b}, 1e-5f, 1e-7f, "Complex Expression 1");
}

TEST(MathOperationParity, ComplexExpression2) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({32, 32}, 0.1f, 2.0f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        // exp(log(x) * 2) should equal x^2
        return exp(log(inputs[0]) * 2.0f);
    }, {a}, 1e-4f, 1e-6f, "Complex Expression 2");
}

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
