/**
 * @file test_image_io.cpp
 * @brief Unit tests for the stb-based image I/O wrapper (commit ca264523).
 *
 * NOTE: This is a CPU-only test, not a backend parity test. The stb_image /
 * stb_image_write code paths operate on host memory; there is no GPU path.
 * Put it in tests/unit/ and gate by no backend.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/io/image.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

using namespace tenzor;

namespace {

// Build a uint8 (C, H, W) tensor filled with a recognizable pattern so we can
// verify roundtrip correctness bit-for-bit (PNG is lossless).
Tensor make_pattern_image(int64_t C, int64_t H, int64_t W) {
    auto t = zeros({C, H, W}, DType::UInt8, Device::cpu());
    auto* data = t.data<uint8_t>();
    for (int64_t c = 0; c < C; ++c) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t w = 0; w < W; ++w) {
                // Distinct values per channel so a channel-order bug is loud.
                uint8_t v = static_cast<uint8_t>((c * 64 + h + w) & 0xFF);
                data[c * H * W + h * W + w] = v;
            }
        }
    }
    return t;
}

// Generate a unique temp path per test so parallel test discovery (if any)
// doesn't collide even though ctest runs -j1.
std::string temp_path(const std::string& stem, const std::string& ext) {
    std::filesystem::path tmp = std::filesystem::temp_directory_path();
    tmp /= "tenzor_test_image_" + stem + "_" +
           std::to_string(::getpid()) + ext;
    return tmp.string();
}

}  // namespace

TEST(ImageIO, PNG_Roundtrip_RGB) {
    auto in = make_pattern_image(3, 8, 16);
    auto path = temp_path("rgb", ".png");

    io::write_image(in, path);
    auto out = io::read_image(path, io::ImageMode::RGB);

    ASSERT_EQ(out.dtype(), DType::UInt8);
    auto in_shape = in.shape();
    auto out_shape = out.shape();
    ASSERT_EQ(in_shape.size(), out_shape.size());
    for (size_t i = 0; i < in_shape.size(); ++i) {
        EXPECT_EQ(in_shape[i], out_shape[i]) << "shape mismatch at dim " << i;
    }

    // PNG is lossless — every byte should round-trip.
    const auto* a = in.data<uint8_t>();
    const auto* b = out.data<uint8_t>();
    for (int64_t i = 0; i < in.numel(); ++i) {
        ASSERT_EQ(a[i], b[i]) << "byte " << i << " differs after PNG roundtrip";
    }

    std::filesystem::remove(path);
}

TEST(ImageIO, PNG_Roundtrip_Grayscale) {
    auto in = make_pattern_image(1, 12, 12);
    auto path = temp_path("gray", ".png");

    io::write_image(in, path);
    auto out = io::read_image(path, io::ImageMode::GRAYSCALE);

    ASSERT_EQ(out.dtype(), DType::UInt8);
    auto out_shape = out.shape();
    EXPECT_EQ(out_shape[0], 1);
    EXPECT_EQ(out_shape[1], 12);
    EXPECT_EQ(out_shape[2], 12);

    std::filesystem::remove(path);
}

TEST(ImageIO, PNG_Roundtrip_RGBA) {
    auto in = make_pattern_image(4, 6, 6);
    auto path = temp_path("rgba", ".png");

    io::write_image(in, path);
    auto out = io::read_image(path, io::ImageMode::RGBA);

    ASSERT_EQ(out.dtype(), DType::UInt8);
    EXPECT_EQ(out.shape()[0], 4);

    std::filesystem::remove(path);
}

TEST(ImageIO, JPEG_RoundtripQualityTolerance) {
    auto in = make_pattern_image(3, 32, 32);
    auto path = temp_path("jpeg", ".jpg");

    io::write_image(in, path, /*quality=*/95);
    auto out = io::read_image(path, io::ImageMode::RGB);

    ASSERT_EQ(out.dtype(), DType::UInt8);
    auto out_shape = out.shape();
    EXPECT_EQ(out_shape[0], 3);
    EXPECT_EQ(out_shape[1], 32);
    EXPECT_EQ(out_shape[2], 32);

    // JPEG is lossy — compare within a generous tolerance instead of byte-equal.
    // Compute mean absolute difference; at quality=95 it should be tiny.
    const auto* a = in.data<uint8_t>();
    const auto* b = out.data<uint8_t>();
    double total_diff = 0;
    for (int64_t i = 0; i < in.numel(); ++i) {
        total_diff += std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
    }
    double mean_diff = total_diff / static_cast<double>(in.numel());
    // At quality=95 typical mean absolute diff on a smooth pattern is <5.
    EXPECT_LT(mean_diff, 10.0) << "JPEG quality=95 diverged unexpectedly: "
                               << mean_diff;

    std::filesystem::remove(path);
}

TEST(ImageIO, EncodeDecodePNG_InMemoryRoundtrip) {
    auto in = make_pattern_image(3, 16, 16);

    auto encoded = io::encode_png(in);
    EXPECT_GT(encoded.size(), 0u);

    auto out = io::decode_png(encoded, io::ImageMode::RGB);
    ASSERT_EQ(out.dtype(), DType::UInt8);
    EXPECT_EQ(out.shape()[0], 3);
    EXPECT_EQ(out.shape()[1], 16);
    EXPECT_EQ(out.shape()[2], 16);

    const auto* a = in.data<uint8_t>();
    const auto* b = out.data<uint8_t>();
    for (int64_t i = 0; i < in.numel(); ++i) {
        ASSERT_EQ(a[i], b[i]) << "byte " << i
                              << " differs after in-memory PNG roundtrip";
    }
}

TEST(ImageIO, EncodeDecodeJPEG_InMemoryRoundtrip) {
    auto in = make_pattern_image(3, 32, 32);

    auto encoded = io::encode_jpeg(in, /*quality=*/95);
    EXPECT_GT(encoded.size(), 0u);

    auto out = io::decode_jpeg(encoded, io::ImageMode::RGB);
    ASSERT_EQ(out.dtype(), DType::UInt8);
    EXPECT_EQ(out.shape()[0], 3);
}

TEST(ImageIO, ReadNonexistent_Throws) {
    EXPECT_THROW(io::read_image("/nonexistent_/tenzor_not_a_real_file.png"),
                 std::exception);
}

// SECURITY: decode_jpeg / decode_png must enforce the expected container magic
// so a caller asking for one format cannot be silently handed a different
// (content-sniffed) one — and so the GIF/PSD/PIC/HDR/PNM decoders (disabled in
// this TU) are never reached. A real PNG fed to decode_jpeg must be rejected.
TEST(ImageIO, DecodeJpegRejectsNonJpegMagic) {
    auto img = make_pattern_image(3, 16, 16);
    auto png = io::encode_png(img);           // valid PNG bytes
    ASSERT_GT(png.size(), 8u);
    EXPECT_THROW(io::decode_jpeg(png, io::ImageMode::RGB), std::exception)
        << "decode_jpeg accepted PNG-magic bytes";
}

TEST(ImageIO, DecodePngRejectsNonPngMagic) {
    auto img = make_pattern_image(3, 16, 16);
    auto jpg = io::encode_jpeg(img, /*quality=*/90);  // valid JPEG bytes
    ASSERT_GT(jpg.size(), 3u);
    EXPECT_THROW(io::decode_png(jpg, io::ImageMode::RGB), std::exception)
        << "decode_png accepted JPEG-magic bytes";
}

TEST(ImageIO, DecodeRejectsGarbageMagic) {
    std::vector<uint8_t> garbage(64, 0x5A);   // not any known format
    EXPECT_THROW(io::decode_png(garbage, io::ImageMode::RGB), std::exception);
    EXPECT_THROW(io::decode_jpeg(garbage, io::ImageMode::RGB), std::exception);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    try {
        if (!::testing::GTEST_FLAG(list_tests)) {
            tenzor::initialize();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }
    int result = RUN_ALL_TESTS();
    try { tenzor::finalize(); } catch (...) {}
    return result;
}
