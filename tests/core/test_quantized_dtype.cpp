#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <cmath>
#include "../backend_test_fixture.hpp"

using namespace tenzor;

namespace {

class QuantizedDTypeTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

TEST_P(QuantizedDTypeTest, DTypeSizeAndName) {
    EXPECT_EQ(dtype_size(DType::QInt8), 1);
    EXPECT_EQ(dtype_size(DType::QUInt8), 1);
    EXPECT_EQ(dtype_size(DType::QInt4x2), 1);
    EXPECT_EQ(dtype_name(DType::QInt8), "qint8");
    EXPECT_EQ(dtype_name(DType::QUInt8), "quint8");
    EXPECT_EQ(dtype_name(DType::QInt4x2), "qint4x2");
}

TEST_P(QuantizedDTypeTest, IsQuantized) {
    EXPECT_TRUE(is_quantized(DType::QInt8));
    EXPECT_TRUE(is_quantized(DType::QUInt8));
    EXPECT_TRUE(is_quantized(DType::QInt4x2));
    EXPECT_FALSE(is_quantized(DType::Float32));
    EXPECT_FALSE(is_quantized(DType::Int8));
}

TEST_P(QuantizedDTypeTest, QuantizePerTensorQInt8) {
    auto input = randn({4, 4}, DType::Float32, device);
    double scale = 0.1;
    int64_t zero_point = 0;

    auto quantized = quantize_per_tensor(input, scale, zero_point, DType::QInt8);

    EXPECT_TRUE(quantized.is_quantized());
    EXPECT_DOUBLE_EQ(quantized.q_scale(), scale);
    EXPECT_EQ(quantized.q_zero_point(), zero_point);
    EXPECT_EQ(quantized.numel(), 16);
}

TEST_P(QuantizedDTypeTest, QuantizePerTensorQUInt8) {
    auto input = randn({10}, DType::Float32, device);
    double scale = 0.05;
    int64_t zero_point = 128;

    auto quantized = quantize_per_tensor(input, scale, zero_point, DType::QUInt8);

    EXPECT_TRUE(quantized.is_quantized());
    EXPECT_DOUBLE_EQ(quantized.q_scale(), scale);
    EXPECT_EQ(quantized.q_zero_point(), zero_point);
}

TEST_P(QuantizedDTypeTest, DequantizeRoundTrip) {
    // Create a tensor with known values (host write -> CPU then to(device)).
    auto input_cpu = zeros({4}, DType::Float32, Device::cpu());
    float* data = input_cpu.data<float>();
    data[0] = 0.0f;
    data[1] = 0.5f;
    data[2] = -0.5f;
    data[3] = 1.0f;
    auto input = input_cpu.to(device);

    double scale = 0.1;
    int64_t zero_point = 0;

    auto quantized = quantize_per_tensor(input, scale, zero_point, DType::QInt8);
    auto dequantized = quantized.dequantize();

    EXPECT_FALSE(dequantized.is_quantized());
    auto dequantized_cpu = dequantized.cpu();
    const float* out = dequantized_cpu.data<float>();

    // Should be close to original within quantization error
    EXPECT_NEAR(out[0], 0.0f, scale);
    EXPECT_NEAR(out[1], 0.5f, scale);
    EXPECT_NEAR(out[2], -0.5f, scale);
    EXPECT_NEAR(out[3], 1.0f, scale);
}

TEST_P(QuantizedDTypeTest, IntRepr) {
    auto input_cpu = zeros({3}, DType::Float32, Device::cpu());
    float* data = input_cpu.data<float>();
    data[0] = 0.0f;
    data[1] = 1.0f;
    data[2] = -1.0f;
    auto input = input_cpu.to(device);

    auto quantized = quantize_per_tensor(input, 0.1, 0, DType::QInt8);
    auto int_repr = quantized.int_repr();

    EXPECT_FALSE(int_repr.is_quantized());
    auto int_repr_cpu = int_repr.cpu();
    const int8_t* idata = int_repr_cpu.data<int8_t>();
    EXPECT_EQ(idata[0], 0);    // 0.0 / 0.1 = 0
    EXPECT_EQ(idata[1], 10);   // 1.0 / 0.1 = 10
    EXPECT_EQ(idata[2], -10);  // -1.0 / 0.1 = -10
}

TEST_P(QuantizedDTypeTest, IntReprIsZeroCopyView) {
    // int_repr() must share storage with the quantized parent: writes through
    // the int view must be visible via the quantized tensor's raw bytes.
    auto input_cpu = zeros({4}, DType::Float32, Device::cpu());
    float* data = input_cpu.data<float>();
    data[0] = 0.0f;
    data[1] = 1.0f;
    data[2] = 2.0f;
    data[3] = -1.0f;
    auto input = input_cpu.to(device);
    auto quantized = quantize_per_tensor(input, 0.1, 0, DType::QInt8);

    auto view = quantized.int_repr();
    EXPECT_EQ(view.data_ptr(), quantized.data_ptr());
    EXPECT_EQ(view.dtype(), DType::Int8);

    // Mutate element 1 in-place through the view (device-safe: fill_ on the
    // shared storage) and confirm dequantize() observes it. Writing through a
    // raw host pointer (view.data<int8_t>()[1]) segfaults for a GPU-resident
    // view; fill_ exercises the same zero-copy-mutation property on any backend.
    view.slice(0, 1, 2).fill_(42.0);
    auto deq = quantized.dequantize();
    auto deq_cpu = deq.cpu();
    const float* fdata = deq_cpu.data<float>();
    EXPECT_NEAR(fdata[1], 42 * 0.1f, 1e-6f);
}

TEST_P(QuantizedDTypeTest, IntReprUInt8) {
    auto input = randn({5}, DType::Float32, device);
    auto quantized = quantize_per_tensor(input, 0.05, 128, DType::QUInt8);
    auto view = quantized.int_repr();
    EXPECT_EQ(view.dtype(), DType::UInt8);
    EXPECT_EQ(view.data_ptr(), quantized.data_ptr());
}

TEST_P(QuantizedDTypeTest, ToQuantizedDtypeThrows) {
    // .to() cannot produce a quantized tensor because scale/zero_point are
    // not supplied — the old silent zero fall-through is now an explicit
    // throw pointing the user at quantize_per_tensor.
    auto x = randn({4}, DType::Float32, device);
    EXPECT_THROW(x.to(DType::QInt8), std::runtime_error);
    EXPECT_THROW(x.to(DType::QUInt8), std::runtime_error);
    EXPECT_THROW(x.to(DType::QInt4x2), std::runtime_error);
}

TEST_P(QuantizedDTypeTest, ToFromQuantizedThrows) {
    // Casting a quantized tensor back to float requires dequantize() since
    // the bare bytes have no meaning without scale/zero_point.
    auto x = randn({4}, DType::Float32, device);
    auto q = quantize_per_tensor(x, 0.1, 0, DType::QInt8);
    EXPECT_THROW(q.to(DType::Float32), std::runtime_error);
}

TEST_P(QuantizedDTypeTest, NonQuantizedThrows) {
    auto t = randn({5}, DType::Float32, device);
    EXPECT_FALSE(t.is_quantized());
    EXPECT_THROW(t.q_scale(), std::runtime_error);
    EXPECT_THROW(t.q_zero_point(), std::runtime_error);
    EXPECT_THROW(t.int_repr(), std::runtime_error);
    EXPECT_THROW(t.dequantize(), std::runtime_error);
}

TEST_P(QuantizedDTypeTest, InvalidScaleThrows) {
    auto input = randn({5}, DType::Float32, device);
    EXPECT_THROW(quantize_per_tensor(input, 0.0, 0), std::invalid_argument);
    EXPECT_THROW(quantize_per_tensor(input, -1.0, 0), std::invalid_argument);
}

TEST_P(QuantizedDTypeTest, NonQuantizedDTypeThrows) {
    auto input = randn({5}, DType::Float32, device);
    EXPECT_THROW(quantize_per_tensor(input, 0.1, 0, DType::Float32), std::invalid_argument);
}

INSTANTIATE_BACKEND_TESTS(QuantizedDTypeTest);

}  // namespace
