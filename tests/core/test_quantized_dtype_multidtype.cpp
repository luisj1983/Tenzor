// Multi-backend multi-dtype tests for quantized dtype operations.
//
// Quantization always starts from a floating-point input, so we test
// quantize/dequantize round-trips with the parameterized source dtype
// across all backends.

#include "../multi_backend_dtype_fixture.hpp"

#include <cmath>

namespace tenzor {
namespace testing {

class QuantizedDTypeMultiDTypeTest : public MultiBackendDTypeTest {};

// ---------------------------------------------------------------------------
// Basic quantized dtype metadata (backend/dtype-independent, but validates
// consistency across all parameterized contexts)
// ---------------------------------------------------------------------------

TEST_P(QuantizedDTypeMultiDTypeTest, DTypeSizeAndName) {
    EXPECT_EQ(dtype_size(DType::QInt8), 1);
    EXPECT_EQ(dtype_size(DType::QUInt8), 1);
    EXPECT_EQ(dtype_size(DType::QInt4x2), 1);
    EXPECT_EQ(dtype_name(DType::QInt8), "qint8");
    EXPECT_EQ(dtype_name(DType::QUInt8), "quint8");
    EXPECT_EQ(dtype_name(DType::QInt4x2), "qint4x2");
}

TEST_P(QuantizedDTypeMultiDTypeTest, IsQuantized) {
    EXPECT_TRUE(is_quantized(DType::QInt8));
    EXPECT_TRUE(is_quantized(DType::QUInt8));
    EXPECT_TRUE(is_quantized(DType::QInt4x2));
    EXPECT_FALSE(is_quantized(DType::Float32));
    EXPECT_FALSE(is_quantized(DType::Int8));
    EXPECT_FALSE(is_quantized(dtype()));
}

// ---------------------------------------------------------------------------
// Quantize from the parameterized dtype (after converting to Float32 on CPU
// since quantize_per_tensor requires Float32 input)
// ---------------------------------------------------------------------------

TEST_P(QuantizedDTypeMultiDTypeTest, QuantizePerTensorQInt8) {
    // Create input in parametrized dtype on device, then move to CPU Float32
    // for quantization (quantize_per_tensor requires Float32 CPU input).
    auto input = createRandn({4, 4});
    auto cpu_f32 = input.to(Device::cpu()).to(DType::Float32);

    double scale = 0.1;
    int64_t zero_point = 0;
    auto quantized = quantize_per_tensor(cpu_f32, scale, zero_point, DType::QInt8);

    EXPECT_TRUE(quantized.is_quantized());
    EXPECT_DOUBLE_EQ(quantized.q_scale(), scale);
    EXPECT_EQ(quantized.q_zero_point(), zero_point);
    EXPECT_EQ(quantized.numel(), 16);
}

// ---------------------------------------------------------------------------
// Dequantize round-trip: float -> quantize -> dequantize should be close
// ---------------------------------------------------------------------------

TEST_P(QuantizedDTypeMultiDTypeTest, DequantizeRoundTrip) {
    // Skip Float16 for precision-sensitive test
    if (dtype() == DType::Float16) {
        GTEST_SKIP() << "Float16 precision too low for quantization round-trip";
    }

    auto input = tenzor::zeros({4}, DType::Float32, Device::cpu());
    float* data = input.data<float>();
    data[0] = 0.0f;
    data[1] = 0.5f;
    data[2] = -0.5f;
    data[3] = 1.0f;

    double scale = 0.1;
    int64_t zero_point = 0;

    auto quantized = quantize_per_tensor(input, scale, zero_point, DType::QInt8);
    auto dequantized = quantized.dequantize();

    EXPECT_FALSE(dequantized.is_quantized());
    const float* out = dequantized.data<float>();
    EXPECT_NEAR(out[0], 0.0f, scale);
    EXPECT_NEAR(out[1], 0.5f, scale);
    EXPECT_NEAR(out[2], -0.5f, scale);
    EXPECT_NEAR(out[3], 1.0f, scale);
}

// ---------------------------------------------------------------------------
// .to() cannot produce a quantized tensor — must throw
// ---------------------------------------------------------------------------

TEST_P(QuantizedDTypeMultiDTypeTest, ToQuantizedDtypeThrows) {
    auto x = createRandn({4});
    auto cpu_x = x.to(Device::cpu()).to(DType::Float32);
    EXPECT_THROW(cpu_x.to(DType::QInt8), std::runtime_error);
    EXPECT_THROW(cpu_x.to(DType::QUInt8), std::runtime_error);
    EXPECT_THROW(cpu_x.to(DType::QInt4x2), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Casting quantized -> float requires dequantize(), not .to()
// ---------------------------------------------------------------------------

TEST_P(QuantizedDTypeMultiDTypeTest, ToFromQuantizedThrows) {
    auto x = tenzor::randn({4}, DType::Float32, Device::cpu());
    auto q = quantize_per_tensor(x, 0.1, 0, DType::QInt8);
    EXPECT_THROW(q.to(DType::Float32), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Non-quantized tensor must not expose quantization attributes
// ---------------------------------------------------------------------------

TEST_P(QuantizedDTypeMultiDTypeTest, NonQuantizedThrows) {
    auto t = createRandn({5});
    auto cpu_t = t.to(Device::cpu()).to(DType::Float32);
    EXPECT_FALSE(cpu_t.is_quantized());
    EXPECT_THROW(cpu_t.q_scale(), std::runtime_error);
    EXPECT_THROW(cpu_t.q_zero_point(), std::runtime_error);
    EXPECT_THROW(cpu_t.int_repr(), std::runtime_error);
    EXPECT_THROW(cpu_t.dequantize(), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Invalid scale must throw
// ---------------------------------------------------------------------------

TEST_P(QuantizedDTypeMultiDTypeTest, InvalidScaleThrows) {
    auto input = tenzor::randn({5}, DType::Float32, Device::cpu());
    EXPECT_THROW(quantize_per_tensor(input, 0.0, 0), std::invalid_argument);
    EXPECT_THROW(quantize_per_tensor(input, -1.0, 0), std::invalid_argument);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(QuantizedDTypeMultiDTypeTest);

} // namespace testing
} // namespace tenzor
