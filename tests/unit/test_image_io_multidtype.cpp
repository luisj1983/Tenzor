/**
 * @file test_image_io_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for image I/O operations
 *
 * Image I/O (stb_image) operates on host memory and produces UInt8 tensors.
 * This test verifies that image data can be created, transferred to different
 * backends, converted between dtypes, and round-tripped correctly.
 */

#include "../multi_backend_dtype_fixture.hpp"
#include <tenzor/io/image.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Helpers
// ============================================================================

namespace {

/// Build a UInt8 (C, H, W) pattern image on CPU.
Tensor make_pattern_image(int64_t C, int64_t H, int64_t W) {
    auto t = zeros({C, H, W}, DType::UInt8, Device::cpu());
    auto* data = t.data<uint8_t>();
    for (int64_t c = 0; c < C; ++c) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t w = 0; w < W; ++w) {
                uint8_t v = static_cast<uint8_t>((c * 64 + h + w) & 0xFF);
                data[c * H * W + h * W + w] = v;
            }
        }
    }
    return t;
}

std::string temp_path(const std::string& stem, const std::string& ext) {
    std::filesystem::path tmp = std::filesystem::temp_directory_path();
    tmp /= "tenzor_test_image_md_" + stem + "_" +
           std::to_string(::getpid()) + ext;
    return tmp.string();
}

}  // namespace

// ============================================================================
// Fixture
// ============================================================================

class ImageIOMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// Tests
// ============================================================================

/// Create an image tensor, convert to test dtype/device, verify shape preserved.
TEST_P(ImageIOMultiDTypeTest, TensorCreationFromImage) {
    auto img = make_pattern_image(3, 8, 16);  // UInt8 on CPU
    // Convert to test dtype and device
    auto converted = img.to(dtype()).to(device());

    expectShape(converted, {3, 8, 16});
    EXPECT_EQ(converted.dtype(), dtype());
    EXPECT_EQ(converted.device().type, device().type);
}

/// Shape is preserved through dtype conversion round-trip.
TEST_P(ImageIOMultiDTypeTest, ShapePreservationAfterConversion) {
    auto img = make_pattern_image(1, 12, 12);
    auto on_device = img.to(dtype()).to(device());
    // Convert back to CPU Float32
    auto back = on_device.to(DType::Float32).to(Device::cpu());

    expectShape(back, {1, 12, 12});
    EXPECT_EQ(back.numel(), 144);
}

/// DType round-trip: UInt8 -> test dtype -> Float32, values stay consistent.
TEST_P(ImageIOMultiDTypeTest, DTypeRoundtripConsistency) {
    auto img = make_pattern_image(3, 4, 4);
    auto converted = img.to(DType::Float32).to(dtype()).to(device());
    auto back_f32 = converted.to(DType::Float32).to(Device::cpu());
    auto orig_f32 = img.to(DType::Float32);

    auto* a = orig_f32.data<float>();
    auto* b = back_f32.data<float>();
    for (int64_t i = 0; i < img.numel(); ++i) {
        EXPECT_NEAR(a[i], b[i], atol())
            << "Mismatch at index " << i << " on " << backend_name();
    }
}

/// PNG round-trip: write from CPU, read back, convert to test dtype/device.
TEST_P(ImageIOMultiDTypeTest, PNGRoundtripToDevice) {
    auto img = make_pattern_image(3, 8, 8);
    auto path = temp_path("rgb_md", ".png");

    io::write_image(img, path);
    auto loaded = io::read_image(path, io::ImageMode::RGB);

    // Move loaded image to test dtype/device
    auto on_device = loaded.to(dtype()).to(device());
    expectShape(on_device, {3, 8, 8});
    EXPECT_EQ(on_device.dtype(), dtype());

    std::filesystem::remove(path);
}

/// In-memory PNG encode/decode, then convert to test dtype/device.
TEST_P(ImageIOMultiDTypeTest, InMemoryPNGToDevice) {
    auto img = make_pattern_image(3, 16, 16);

    auto encoded = io::encode_png(img);
    EXPECT_GT(encoded.size(), 0u);

    auto decoded = io::decode_png(encoded, io::ImageMode::RGB);
    auto on_device = decoded.to(dtype()).to(device());

    expectShape(on_device, {3, 16, 16});
}

/// RGBA image: 4 channels preserved through dtype conversion.
TEST_P(ImageIOMultiDTypeTest, RGBAChannelPreservation) {
    auto img = make_pattern_image(4, 6, 6);
    auto converted = img.to(dtype()).to(device());

    auto shape = converted.shape();
    EXPECT_EQ(shape[0], 4);
    EXPECT_EQ(shape[1], 6);
    EXPECT_EQ(shape[2], 6);
}

/// Grayscale image: single channel, verify numel after conversion.
TEST_P(ImageIOMultiDTypeTest, GrayscaleConversion) {
    auto img = make_pattern_image(1, 10, 10);
    auto converted = img.to(dtype()).to(device());

    EXPECT_EQ(converted.shape()[0], 1);
    EXPECT_EQ(converted.numel(), 100);
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ImageIOMultiDTypeTest);
