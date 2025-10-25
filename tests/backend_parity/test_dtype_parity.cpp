/**
 * @file test_dtype_parity.cpp
 * @brief Data type parity tests across backends
 *
 * Tests operations with different data types (Float32, Float64, Int32, Int64)
 * to ensure consistent behavior across backends.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Float32 Operations
// ============================================================================

TEST(DTypeParity, Float32_Add) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[1];
    }, {a, b}, 1e-6f, 1e-8f, "Float32 Add");
}

TEST(DTypeParity, Float32_MatMul) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({64, 64}, DType::Float32, Device::cpu());
    auto b = randn({64, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return matmul(inputs[0], inputs[1]);
    }, {a, b}, 1e-4f, 1e-6f, "Float32 MatMul");
}

// ============================================================================
// Float64 Operations
// ============================================================================

TEST(DTypeParity, Float64_Add) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float64, Device::cpu());
    auto b = randn({32, 32}, DType::Float64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[1];
    }, {a, b}, 1e-10f, 1e-12f, "Float64 Add");
}

TEST(DTypeParity, Float64_MatMul) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({64, 64}, DType::Float64, Device::cpu());
    auto b = randn({64, 64}, DType::Float64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return matmul(inputs[0], inputs[1]);
    }, {a, b}, 1e-8f, 1e-10f, "Float64 MatMul");
}

TEST(DTypeParity, Float64_Precision) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        // Test precision-sensitive operation
        return exp(log(inputs[0]));
    }, {a}, 1e-9f, 1e-11f, "Float64 Precision");
}

// ============================================================================
// Integer Operations
// ============================================================================

TEST(DTypeParity, Int32_Add) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = (randn({32, 32}, DType::Float32, Device::cpu()) * 100).to(DType::Int32);
    auto b = (randn({32, 32}, DType::Float32, Device::cpu()) * 100).to(DType::Int32);

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[1];
    }, {a, b}, 0.0f, 0.0f, "Int32 Add"); // Exact match for integers
}

TEST(DTypeParity, Int64_Add) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = (randn({32, 32}, DType::Float32, Device::cpu()) * 1000).to(DType::Int64);
    auto b = (randn({32, 32}, DType::Float32, Device::cpu()) * 1000).to(DType::Int64);

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[1];
    }, {a, b}, 0.0f, 0.0f, "Int64 Add");
}

TEST(DTypeParity, Int32_Mul) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = (randn({32, 32}, DType::Float32, Device::cpu()) * 10).to(DType::Int32);
    auto b = (randn({32, 32}, DType::Float32, Device::cpu()) * 10).to(DType::Int32);

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] * inputs[1];
    }, {a, b}, 0.0f, 0.0f, "Int32 Mul");
}

// ============================================================================
// Type Casting Tests
// ============================================================================

TEST(DTypeParity, Cast_Float32_To_Float64) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0].to(DType::Float64);
    }, {a}, 1e-7f, 1e-9f, "Cast Float32 to Float64");
}

TEST(DTypeParity, Cast_Float64_To_Float32) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0].to(DType::Float32);
    }, {a}, 1e-6f, 1e-8f, "Cast Float64 to Float32");
}

TEST(DTypeParity, Cast_Float_To_Int) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu()) * 10;

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0].to(DType::Int32);
    }, {a}, 0.0f, 0.0f, "Cast Float to Int");
}

TEST(DTypeParity, Cast_Int_To_Float) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = (randn({32, 32}, DType::Float32, Device::cpu()) * 100).to(DType::Int32);

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0].to(DType::Float32);
    }, {a}, 1e-7f, 1e-9f, "Cast Int to Float");
}

// ============================================================================
// Type Promotion Tests
// ============================================================================

TEST(DTypeParity, TypePromotion_Float32_Float64) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        // Should promote to Float64
        return inputs[0] + inputs[1];
    }, {a, b}, 1e-6f, 1e-8f, "Type Promotion Float32+Float64");

    // Verify result dtype is Float64
    auto result = a + b;
    EXPECT_EQ(result.dtype(), DType::Float64);
}

TEST(DTypeParity, TypePromotion_Int_Float) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = (randn({32, 32}, DType::Float32, Device::cpu()) * 10).to(DType::Int32);
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        // Should promote to Float32
        return inputs[0] + inputs[1];
    }, {a, b}, 1e-6f, 1e-8f, "Type Promotion Int+Float");

    auto result = a + b;
    EXPECT_EQ(result.dtype(), DType::Float32);
}

// ============================================================================
// Mixed-Type Operations
// ============================================================================

TEST(DTypeParity, MixedType_MatMul) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a_f32 = randn({32, 64}, DType::Float32, Device::cpu());
    auto b_f32 = randn({64, 128}, DType::Float32, Device::cpu());

    auto a_f64 = a_f32.to(DType::Float64);
    auto b_f64 = b_f32.to(DType::Float64);

    // Test Float32
    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return matmul(inputs[0], inputs[1]);
    }, {a_f32, b_f32}, 1e-4f, 1e-6f, "MatMul Float32");

    // Test Float64
    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return matmul(inputs[0], inputs[1]);
    }, {a_f64, b_f64}, 1e-8f, 1e-10f, "MatMul Float64");
}

TEST(DTypeParity, MixedType_Reduction) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a_f32 = randn({32, 64}, DType::Float32, Device::cpu());
    auto a_f64 = a_f32.to(DType::Float64);

    // Test Float32 sum
    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0].sum();
    }, {a_f32}, 1e-4f, 1e-6f, "Sum Float32");

    // Test Float64 sum
    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0].sum();
    }, {a_f64}, 1e-8f, 1e-10f, "Sum Float64");
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(DTypeParity, EdgeCase_ZeroValues) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = zeros({32, 32}, DType::Float32, Device::cpu());
    auto b = ones({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[1];
    }, {a, b}, 0.0f, 0.0f, "Zero Values");
}

TEST(DTypeParity, EdgeCase_OneValues) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = ones({32, 32}, DType::Float32, Device::cpu());
    auto b = ones({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] * inputs[1];
    }, {a, b}, 0.0f, 0.0f, "One Values");
}

TEST(DTypeParity, EdgeCase_LargeInt) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // Use large integers that are within Int32 range
    auto a = full({32, 32}, 1000000, DType::Int32, Device::cpu());
    auto b = full({32, 32}, 1000, DType::Int32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[1];
    }, {a, b}, 0.0f, 0.0f, "Large Int Values");
}

// ============================================================================
// Precision Comparison Tests
// ============================================================================

TEST(DTypeParity, PrecisionComparison_Exp) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a_f32 = generate_uniform_tensor({32, 32}, -2.0f, 2.0f, DType::Float32, Device::cpu());
    auto a_f64 = a_f32.to(DType::Float64);

    // Float32 should have less precision than Float64
    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return exp(inputs[0]);
    }, {a_f32}, 1e-5f, 1e-7f, "Exp Float32");

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return exp(inputs[0]);
    }, {a_f64}, 1e-9f, 1e-11f, "Exp Float64");
}

TEST(DTypeParity, PrecisionComparison_TrigFunctions) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a_f32 = generate_uniform_tensor({32, 32}, -3.14f, 3.14f, DType::Float32, Device::cpu());
    auto a_f64 = a_f32.to(DType::Float64);

    // Test sin with different precisions
    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return sin(inputs[0]);
    }, {a_f32}, 1e-6f, 1e-8f, "Sin Float32");

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return sin(inputs[0]);
    }, {a_f64}, 1e-10f, 1e-12f, "Sin Float64");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    try {
        tenzor::initialize();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }

    int result = RUN_ALL_TESTS();

    try {
        tenzor::finalize();
    } catch (...) {}

    return result;
}
