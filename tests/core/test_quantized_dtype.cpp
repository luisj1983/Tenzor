#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <cmath>

using namespace tenzor;

class QuantizedDTypeTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }
};

TEST_F(QuantizedDTypeTest, DTypeSizeAndName) {
    EXPECT_EQ(dtype_size(DType::QInt8), 1);
    EXPECT_EQ(dtype_size(DType::QUInt8), 1);
    EXPECT_EQ(dtype_size(DType::QInt4x2), 1);
    EXPECT_EQ(dtype_name(DType::QInt8), "qint8");
    EXPECT_EQ(dtype_name(DType::QUInt8), "quint8");
    EXPECT_EQ(dtype_name(DType::QInt4x2), "qint4x2");
}

TEST_F(QuantizedDTypeTest, IsQuantized) {
    EXPECT_TRUE(is_quantized(DType::QInt8));
    EXPECT_TRUE(is_quantized(DType::QUInt8));
    EXPECT_TRUE(is_quantized(DType::QInt4x2));
    EXPECT_FALSE(is_quantized(DType::Float32));
    EXPECT_FALSE(is_quantized(DType::Int8));
}

TEST_F(QuantizedDTypeTest, QuantizePerTensorQInt8) {
    auto input = randn({4, 4}, DType::Float32, Device::cpu());
    double scale = 0.1;
    int64_t zero_point = 0;

    auto quantized = quantize_per_tensor(input, scale, zero_point, DType::QInt8);

    EXPECT_TRUE(quantized.is_quantized());
    EXPECT_DOUBLE_EQ(quantized.q_scale(), scale);
    EXPECT_EQ(quantized.q_zero_point(), zero_point);
    EXPECT_EQ(quantized.numel(), 16);
}

TEST_F(QuantizedDTypeTest, QuantizePerTensorQUInt8) {
    auto input = randn({10}, DType::Float32, Device::cpu());
    double scale = 0.05;
    int64_t zero_point = 128;

    auto quantized = quantize_per_tensor(input, scale, zero_point, DType::QUInt8);

    EXPECT_TRUE(quantized.is_quantized());
    EXPECT_DOUBLE_EQ(quantized.q_scale(), scale);
    EXPECT_EQ(quantized.q_zero_point(), zero_point);
}

TEST_F(QuantizedDTypeTest, DequantizeRoundTrip) {
    // Create a tensor with known values
    auto input = zeros({4}, DType::Float32, Device::cpu());
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

    // Should be close to original within quantization error
    EXPECT_NEAR(out[0], 0.0f, scale);
    EXPECT_NEAR(out[1], 0.5f, scale);
    EXPECT_NEAR(out[2], -0.5f, scale);
    EXPECT_NEAR(out[3], 1.0f, scale);
}

TEST_F(QuantizedDTypeTest, IntRepr) {
    auto input = zeros({3}, DType::Float32, Device::cpu());
    float* data = input.data<float>();
    data[0] = 0.0f;
    data[1] = 1.0f;
    data[2] = -1.0f;

    auto quantized = quantize_per_tensor(input, 0.1, 0, DType::QInt8);
    auto int_repr = quantized.int_repr();

    EXPECT_FALSE(int_repr.is_quantized());
    const int8_t* idata = int_repr.data<int8_t>();
    EXPECT_EQ(idata[0], 0);    // 0.0 / 0.1 = 0
    EXPECT_EQ(idata[1], 10);   // 1.0 / 0.1 = 10
    EXPECT_EQ(idata[2], -10);  // -1.0 / 0.1 = -10
}

TEST_F(QuantizedDTypeTest, NonQuantizedThrows) {
    auto t = randn({5}, DType::Float32, Device::cpu());
    EXPECT_FALSE(t.is_quantized());
    EXPECT_THROW(t.q_scale(), std::runtime_error);
    EXPECT_THROW(t.q_zero_point(), std::runtime_error);
    EXPECT_THROW(t.int_repr(), std::runtime_error);
    EXPECT_THROW(t.dequantize(), std::runtime_error);
}

TEST_F(QuantizedDTypeTest, InvalidScaleThrows) {
    auto input = randn({5}, DType::Float32, Device::cpu());
    EXPECT_THROW(quantize_per_tensor(input, 0.0, 0), std::invalid_argument);
    EXPECT_THROW(quantize_per_tensor(input, -1.0, 0), std::invalid_argument);
}

TEST_F(QuantizedDTypeTest, NonQuantizedDTypeThrows) {
    auto input = randn({5}, DType::Float32, Device::cpu());
    EXPECT_THROW(quantize_per_tensor(input, 0.1, 0, DType::Float32), std::invalid_argument);
}
