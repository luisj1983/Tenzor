/**
 * @file test_dtype_parity.cpp
 * @brief Data type parity tests across backends
 *
 * Tests operations with different data types (Float32, Float64, Int32, Int64)
 * to ensure consistent behavior across backends.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/functional.hpp>
#include <cstdlib>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;


class DTypeParity : public BackendTest {};
// The CUDA backend defaults to TF32 tensor cores for Float32 matmul
// (~3e-4 relative precision) which is well below the parity tolerances
// used here. Setting TENZOR_DISABLE_TF32=1 before initialize() forces
// full IEEE 754 FP32 on cuBLAS so CPU↔CUDA matmul parity is measurable.
// The actual setenv() call lives in main().

// ============================================================================
// Float32 Operations
// ============================================================================

TEST_P(DTypeParity, Float32_Add) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[1];
    }, {a, b}, device, 1e-6f, 1e-8f, "Float32 Add");
}

TEST_P(DTypeParity, Float32_MatMul) {

    auto a = randn({64, 64}, DType::Float32, Device::cpu());
    auto b = randn({64, 64}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return matmul(inputs[0], inputs[1]);
    }, {a, b}, device, parity::MATMUL_RTOL, parity::MATMUL_ATOL, "Float32 MatMul");
}

// ============================================================================
// Float64 Operations
// ============================================================================

TEST_P(DTypeParity, Float64_Add) {

    auto a = randn({32, 32}, DType::Float64, Device::cpu());
    auto b = randn({32, 32}, DType::Float64, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[1];
    }, {a, b}, device, 1e-10f, 1e-12f, "Float64 Add");
}

TEST_P(DTypeParity, Float64_MatMul) {

    auto a = randn({64, 64}, DType::Float64, Device::cpu());
    auto b = randn({64, 64}, DType::Float64, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return matmul(inputs[0], inputs[1]);
    }, {a, b}, device, 1e-8f, 1e-10f, "Float64 MatMul");
}

TEST_P(DTypeParity, Float64_Precision) {

    // exp(log(x)) is only the identity for x > 0. Random normals include
    // negative values, which send log() to NaN (or -inf on some GPU
    // libms), so each backend's undefined-input handling decides the
    // result at roughly half the positions. Shift to a strictly positive
    // distribution so the test actually measures round-trip precision.
    auto a = tenzor::abs(randn({32, 32}, DType::Float64, Device::cpu())) + 0.1;

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return exp(log(inputs[0]));
    }, {a}, device, 1e-9f, 1e-11f, "Float64 Precision");
}

// ============================================================================
// Integer Operations
// ============================================================================

TEST_P(DTypeParity, Int32_Add) {

    auto a = (randn({32, 32}, DType::Float32, Device::cpu()) * 100).to(DType::Int32);
    auto b = (randn({32, 32}, DType::Float32, Device::cpu()) * 100).to(DType::Int32);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[1];
    }, {a, b}, device, 0.0f, 0.0f, "Int32 Add"); // Exact match for integers
}

TEST_P(DTypeParity, Int64_Add) {

    auto a = (randn({32, 32}, DType::Float32, Device::cpu()) * 1000).to(DType::Int64);
    auto b = (randn({32, 32}, DType::Float32, Device::cpu()) * 1000).to(DType::Int64);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[1];
    }, {a, b}, device, 0.0f, 0.0f, "Int64 Add");
}

TEST_P(DTypeParity, Int32_Mul) {

    auto a = (randn({32, 32}, DType::Float32, Device::cpu()) * 10).to(DType::Int32);
    auto b = (randn({32, 32}, DType::Float32, Device::cpu()) * 10).to(DType::Int32);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0] * inputs[1];
    }, {a, b}, device, 0.0f, 0.0f, "Int32 Mul");
}

// ============================================================================
// Type Casting Tests
// ============================================================================

TEST_P(DTypeParity, Cast_Float32_To_Float64) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0].to(DType::Float64);
    }, {a}, device, 1e-7f, 1e-9f, "Cast Float32 to Float64");
}

TEST_P(DTypeParity, Cast_Float64_To_Float32) {

    auto a = randn({32, 32}, DType::Float64, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0].to(DType::Float32);
    }, {a}, device, 1e-6f, 1e-8f, "Cast Float64 to Float32");
}

TEST_P(DTypeParity, Cast_Float_To_Int) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu()) * 10;

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0].to(DType::Int32);
    }, {a}, device, 0.0f, 0.0f, "Cast Float to Int");
}

TEST_P(DTypeParity, Cast_Int_To_Float) {

    auto a = (randn({32, 32}, DType::Float32, Device::cpu()) * 100).to(DType::Int32);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0].to(DType::Float32);
    }, {a}, device, 1e-7f, 1e-9f, "Cast Int to Float");
}

// ============================================================================
// Type Promotion Tests
// ============================================================================

TEST_P(DTypeParity, TypePromotion_Float32_Float64) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float64, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        // Should promote to Float64
        return inputs[0] + inputs[1];
    }, {a, b}, device, 1e-6f, 1e-8f, "Type Promotion Float32+Float64");

    // Verify result dtype is Float64
    auto result = a + b;
    EXPECT_EQ(result.dtype(), DType::Float64);
}

TEST_P(DTypeParity, TypePromotion_Int_Float) {

    auto a = (randn({32, 32}, DType::Float32, Device::cpu()) * 10).to(DType::Int32);
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        // Should promote to Float32
        return inputs[0] + inputs[1];
    }, {a, b}, device, 1e-6f, 1e-8f, "Type Promotion Int+Float");

    auto result = a + b;
    EXPECT_EQ(result.dtype(), DType::Float32);
}

// ============================================================================
// Mixed-Type Operations
// ============================================================================

TEST_P(DTypeParity, MixedType_MatMul) {

    auto a_f32 = randn({32, 64}, DType::Float32, Device::cpu());
    auto b_f32 = randn({64, 128}, DType::Float32, Device::cpu());

    auto a_f64 = a_f32.to(DType::Float64);
    auto b_f64 = b_f32.to(DType::Float64);

    // Test Float32
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return matmul(inputs[0], inputs[1]);
    }, {a_f32, b_f32}, device, 1e-4f, 1e-6f, "MatMul Float32");

    // Test Float64
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return matmul(inputs[0], inputs[1]);
    }, {a_f64, b_f64}, device, 1e-8f, 1e-10f, "MatMul Float64");
}

TEST_P(DTypeParity, MixedType_Reduction) {

    auto a_f32 = randn({32, 64}, DType::Float32, Device::cpu());
    auto a_f64 = a_f32.to(DType::Float64);

    // Test Float32 sum
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return sum(inputs[0]);
    }, {a_f32}, device, 1e-4f, 1e-6f, "Sum Float32");

    // Test Float64 sum
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return sum(inputs[0]);
    }, {a_f64}, device, 1e-8f, 1e-10f, "Sum Float64");
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_P(DTypeParity, EdgeCase_ZeroValues) {

    auto a = zeros({32, 32}, DType::Float32, Device::cpu());
    auto b = ones({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[1];
    }, {a, b}, device, 0.0f, 0.0f, "Zero Values");
}

TEST_P(DTypeParity, EdgeCase_OneValues) {

    auto a = ones({32, 32}, DType::Float32, Device::cpu());
    auto b = ones({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0] * inputs[1];
    }, {a, b}, device, 0.0f, 0.0f, "One Values");
}

TEST_P(DTypeParity, EdgeCase_LargeInt) {

    // Use large integers that are within Int32 range
    auto a = full({32, 32}, 1000000.0f, DType::Int32, Device::cpu());
    auto b = full({32, 32}, 1000.0f, DType::Int32, Device::cpu());

    std::vector<Tensor> inputs = {a, b};
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[1];
    }, inputs, device, 0.0f, 0.0f, "Large Int Values");
}

// ============================================================================
// Precision Comparison Tests
// ============================================================================

TEST_P(DTypeParity, PrecisionComparison_Exp) {

    auto a_f32 = generate_uniform_tensor({32, 32}, -2.0f, 2.0f, DType::Float32, Device::cpu());
    auto a_f64 = a_f32.to(DType::Float64);

    // Float32 should have less precision than Float64
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return exp(inputs[0]);
    }, {a_f32}, device, 1e-5f, 1e-7f, "Exp Float32");

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return exp(inputs[0]);
    }, {a_f64}, device, 1e-9f, 1e-11f, "Exp Float64");
}

TEST_P(DTypeParity, PrecisionComparison_TrigFunctions) {

    auto a_f32 = generate_uniform_tensor({32, 32}, -3.14f, 3.14f, DType::Float32, Device::cpu());
    auto a_f64 = a_f32.to(DType::Float64);

    // Float32: the CPU Taylor-11 polynomial has ~6e-8 max error on [-π/2, π/2],
    // and near-zero sin values land in the atol floor, so atol=1e-6 (~10 ULP)
    // is the realistic Float32 target.
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return sin(inputs[0]);
    }, {a_f32}, device, 1e-5f, 1e-6f, "Sin Float32");

    // Float64: the whole point of this test is that Float64 sin is MORE precise
    // than Float32. Using a Float32-class tolerance (1e-5/1e-6) would let a
    // backend silently degrade Float64 sin to single precision and still pass.
    // The Vulkan trigonometric_f64 shader uses Cody-Waite range reduction + a
    // degree-15 double-precision Taylor series, giving ~1e-15 agreement with the
    // CPU/OneAPI reference, so the strict Float64 tolerance applies to every
    // backend including Vulkan (no skip).
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return sin(inputs[0]);
    }, {a_f64}, device, 1e-9f, 1e-10f, "Sin Float64");
}

// ============================================================================
// C4: Float64 parity for structured ops (Conv, LayerNorm, Pool).
// These are the categories flagged by the audit as most likely to harbor
// Float32-accumulator-in-Float64-codepath bugs. Running the Variable-level
// forward with Float64 inputs forces each backend's kernel to actually
// honor the declared dtype end-to-end.
// ============================================================================

TEST_P(DTypeParity, Float64_Conv2d) {
    // Small shapes keep parity tolerance tight.
    auto input_t = randn({2, 3, 8, 8}, DType::Float64, Device::cpu());
    auto weight_t = randn({4, 3, 3, 3}, DType::Float64, Device::cpu());
    test_operation_parity_single(
        [](const std::vector<Tensor>& inputs) {
            Variable in(inputs[0], false);
            Variable w(inputs[1], false);
            return tenzor::nn::functional::conv2d(in, w, std::nullopt, {1, 1}, {1, 1}).tensor();
        },
        {input_t, weight_t}, device, 1e-8f, 1e-10f, "Float64 Conv2d");
}

TEST_P(DTypeParity, Float64_LayerNorm) {
    auto input_t = randn({4, 16}, DType::Float64, Device::cpu());
    test_operation_parity_single(
        [](const std::vector<Tensor>& inputs) {
            Variable in(inputs[0], false);
            return tenzor::nn::functional::layer_norm(in, std::vector<int64_t>{16}).tensor();
        },
        {input_t}, device, 1e-8f, 1e-10f, "Float64 LayerNorm");
}

TEST_P(DTypeParity, Float64_MaxPool2d) {
    auto input_t = randn({2, 3, 8, 8}, DType::Float64, Device::cpu());
    test_operation_parity_single(
        [](const std::vector<Tensor>& inputs) {
            Variable in(inputs[0], false);
            return tenzor::nn::functional::max_pool2d(in, {2, 2}, {2, 2}).tensor();
        },
        {input_t}, device, 0.0f, 0.0f, "Float64 MaxPool2d");
}

TEST_P(DTypeParity, Float64_AvgPool2d) {
    auto input_t = randn({2, 3, 8, 8}, DType::Float64, Device::cpu());
    test_operation_parity_single(
        [](const std::vector<Tensor>& inputs) {
            Variable in(inputs[0], false);
            return tenzor::nn::functional::avg_pool2d(in, {2, 2}, {2, 2}, {0, 0}).tensor();
        },
        {input_t}, device, 1e-8f, 1e-10f, "Float64 AvgPool2d");
}

INSTANTIATE_BACKEND_TESTS(DTypeParity);


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Disable CUDA TF32 tensor cores so Float32 matmul uses full FP32
    // precision. Without this the CUDA path silently drops ~13 mantissa
    // bits (TF32) and trivially fails Float32 parity against CPU MKL.
    // Must be set before tenzor::initialize() so the CUDA backend picks
    // it up on first use.
    setenv("TENZOR_DISABLE_TF32", "1", /*overwrite=*/1);

    try {
        if (!::testing::GTEST_FLAG(list_tests)) {
            tenzor::initialize();
        }
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
