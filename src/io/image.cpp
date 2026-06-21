/**
 * @file image.cpp
 * @brief Image I/O implementation using stb_image and stb_image_write.
 */

// This translation unit decodes UNTRUSTED image bytes. stb sniffs the format
// by content (not by the caller's intent), so without these guards a caller
// that thinks it is decoding a PNG would happily run an attacker-supplied
// GIF/PSD/PIC/HDR/PNM through the corresponding (historically memory-unsafe,
// CVE-prone) C decoder. Compile in ONLY the formats this library actually
// supports end-to-end (JPEG/PNG/BMP/TGA — the same set write_image emits) and
// exclude the dangerous decoders entirely so they are not even reachable.
#define STBI_NO_GIF
#define STBI_NO_PSD
#define STBI_NO_PIC
#define STBI_NO_HDR
#define STBI_NO_PNM
// Cap the maximum accepted image dimension well below stb's 32768 default so a
// tiny crafted header cannot declare a multi-gigapixel image and force a
// multi-GB decode/transpose allocation (amplification DoS). 16384 covers any
// realistic photo/dataset image; see decode_from_memory/read_image for the
// application-level pixel-count ceiling that backs this up.
#define STBI_MAX_DIMENSIONS 16384
#define STB_IMAGE_IMPLEMENTATION
#include "tenzor/io/stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tenzor/io/stb/stb_image_write.h"

#include "tenzor/io/image.hpp"
#include "tenzor/ops/creation.hpp"
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>

namespace tenzor::io {

namespace {

int mode_to_channels(ImageMode mode) {
    switch (mode) {
        case ImageMode::RGB:       return 3;
        case ImageMode::RGBA:      return 4;
        case ImageMode::GRAYSCALE: return 1;
        case ImageMode::UNCHANGED: return 0; // stb uses 0 for "keep original"
    }
    return 0;
}

// Convert HWC uint8 buffer to CHW Tensor
auto hwc_to_chw_tensor(const uint8_t* data, int width, int height, int channels) -> Tensor {
    Tensor out = zeros({channels, height, width}, DType::UInt8, Device::cpu());
    auto* out_ptr = out.data<uint8_t>();

    // Transpose HWC -> CHW. Accumulate offsets in size_t so the index
    // arithmetic does not overflow 32-bit int for large decoded images.
    const size_t W = static_cast<size_t>(width);
    const size_t H = static_cast<size_t>(height);
    const size_t C = static_cast<size_t>(channels);
    for (size_t c = 0; c < C; ++c) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t w = 0; w < W; ++w) {
                out_ptr[c * H * W + h * W + w] =
                    data[(h * W + w) * C + c];
            }
        }
    }
    return out;
}

// Convert CHW Tensor to HWC uint8 buffer
auto chw_to_hwc_buffer(const Tensor& tensor) -> std::vector<uint8_t> {
    auto shape = tensor.shape();
    int channels, height, width;

    if (tensor.ndim() == 3) {
        channels = static_cast<int>(shape[0]);
        height = static_cast<int>(shape[1]);
        width = static_cast<int>(shape[2]);
    } else if (tensor.ndim() == 2) {
        channels = 1;
        height = static_cast<int>(shape[0]);
        width = static_cast<int>(shape[1]);
    } else {
        throw std::invalid_argument("write_image: tensor must be 2D (H,W) or 3D (C,H,W)");
    }

    if (height <= 0 || width <= 0 || channels <= 0) {
        throw std::invalid_argument("write_image: dimensions must be positive");
    }

    auto t = tensor.contiguous();
    const auto* src = t.data<uint8_t>();
    // Promote to size_t BEFORE multiplying so the buffer size (and the index
    // arithmetic below) cannot overflow 32-bit int and under-allocate.
    const size_t W = static_cast<size_t>(width);
    const size_t H = static_cast<size_t>(height);
    const size_t C = static_cast<size_t>(channels);
    std::vector<uint8_t> buf(H * W * C);

    if (tensor.ndim() == 2) {
        // Grayscale — direct copy
        std::memcpy(buf.data(), src, buf.size());
    } else {
        // CHW -> HWC
        for (size_t h = 0; h < H; ++h) {
            for (size_t w = 0; w < W; ++w) {
                for (size_t c = 0; c < C; ++c) {
                    buf[(h * W + w) * C + c] =
                        src[c * H * W + h * W + w];
                }
            }
        }
    }
    return buf;
}

// Reject image dimensions whose internal stb size computations would overflow
// 32-bit signed int. stbi_write_png_to_mem allocates the filtered-scanline
// buffer as (x*n+1)*y in `int`; for large images this overflows to an
// undersized allocation and the row loop then writes past the end (heap
// overflow). All sizes here are computed in 64-bit and bounded by INT_MAX.
void validate_image_dims(int width, int height, int channels) {
    if (width <= 0 || height <= 0 || channels <= 0) {
        throw std::invalid_argument("write_image: dimensions must be positive");
    }
    // stb_image_write only writes 1 (grey), 3 (RGB) or 4 (RGBA) channels for the
    // formats we expose. A channel count outside {1,3,4} (e.g. a (5,H,W) tensor)
    // is passed straight through as `comp` and produces a structurally INVALID
    // file while stb still returns success — a silent malformed write. Reject it
    // here, before any encoder is reached. (This also excludes comp==2, which
    // JPEG in particular cannot represent.)
    if (channels != 1 && channels != 3 && channels != 4) {
        throw std::invalid_argument(
            "write_image: channel count must be 1 (grey), 3 (RGB) or 4 (RGBA), "
            "got " + std::to_string(channels));
    }
    const int64_t w = width, h = height, n = channels;
    // PNG filtered-scanline buffer: (x*n + 1) * y.
    const int64_t png_bytes = (w * n + 1) * h;
    // Raw HWC pixel buffer: x * y * n.
    const int64_t raw_bytes = w * h * n;
    // BMP/TGA write the per-pixel source offset as (j*x+i)*comp in int, and the
    // BMP headers compute file-size fields as int: RGB BMP uses 14+40+(x*3+pad)*y
    // and the V4 (RGBA) header uses 14+108+x*y*4. Bound the largest of these so
    // none of stb's 32-bit size fields overflow.
    const int64_t bmp_v4_bytes = 14 + 108 + w * h * 4;       // RGBA header field
    const int64_t bmp_rgb_pad = (-(w * 3)) & 3;
    const int64_t bmp_rgb_bytes = 14 + 40 + (w * 3 + bmp_rgb_pad) * h;
    constexpr int64_t kIntMax = static_cast<int64_t>(std::numeric_limits<int>::max());
    if (w > kIntMax / n ||                 // x*n
        (w * n) > kIntMax - 1 ||           // x*n + 1
        png_bytes > kIntMax ||             // (x*n+1)*y
        (w * h) > kIntMax / n ||           // x*y*n
        raw_bytes > kIntMax ||
        (w * h) > (kIntMax - 122) / 4 ||   // x*y*4 (V4 header) without overflow
        bmp_v4_bytes > kIntMax ||
        bmp_rgb_bytes > kIntMax) {
        throw std::invalid_argument(
            "write_image: image dimensions too large (would overflow encoder)");
    }
}

auto get_extension(const std::string& path) -> std::string {
    auto pos = path.rfind('.');
    if (pos == std::string::npos) return "";
    std::string ext = path.substr(pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext;
}

// Callback for stb_image_write to memory
void stbi_write_to_vec(void* context, void* data, int size) {
    auto* vec = static_cast<std::vector<uint8_t>*>(context);
    auto* bytes = static_cast<uint8_t*>(data);
    vec->insert(vec->end(), bytes, bytes + size);
}

// Application-level ceiling on the total decoded pixel count (across all
// channels). STBI_MAX_DIMENSIONS bounds each axis, but width*height*channels
// can still be large; reject anything beyond this so a crafted header cannot
// force a multi-GB allocation in stb and again in hwc_to_chw_tensor.
constexpr int64_t kMaxDecodedElements = 256LL * 1024 * 1024;  // 256M elements (~256 MB u8)

void validate_decoded_dims(int width, int height, int channels) {
    if (width <= 0 || height <= 0 || channels <= 0) {
        throw std::runtime_error("decode: decoder returned non-positive dimensions");
    }
    const int64_t total =
        static_cast<int64_t>(width) * static_cast<int64_t>(height) * static_cast<int64_t>(channels);
    if (total > kMaxDecodedElements) {
        throw std::runtime_error(
            "decode: decoded image too large (" + std::to_string(total) +
            " elements exceeds limit " + std::to_string(kMaxDecodedElements) + ")");
    }
}

// Enforce the expected container format on the raw bytes so a caller asking for
// a JPEG/PNG cannot be silently handed a different (content-sniffed) format.
enum class ExpectFormat { JPEG, PNG };

void check_magic(const uint8_t* data, size_t len, ExpectFormat fmt) {
    if (fmt == ExpectFormat::JPEG) {
        // SOI marker: FF D8 FF
        if (len < 3 || data[0] != 0xFF || data[1] != 0xD8 || data[2] != 0xFF) {
            throw std::runtime_error("decode_jpeg: input is not a JPEG (bad magic)");
        }
    } else {  // PNG
        static const uint8_t sig[8] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
        if (len < 8 || std::memcmp(data, sig, 8) != 0) {
            throw std::runtime_error("decode_png: input is not a PNG (bad magic)");
        }
    }
}

auto decode_from_memory(const uint8_t* data, size_t len, ImageMode mode) -> Tensor {
    int width, height, actual_channels;
    int desired = mode_to_channels(mode);

    // stbi_load_from_memory takes the length as `int`; narrowing a >= 2^31-byte
    // buffer would yield a negative/truncated length and let stb compute
    // img_buffer_end = buffer + len past the real end (OOB read) or fail
    // spuriously. Reject oversized untrusted payloads instead of narrowing.
    if (len > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "decode_from_memory: image buffer too large (exceeds INT_MAX bytes)");
    }

    uint8_t* pixels = stbi_load_from_memory(
        data, static_cast<int>(len), &width, &height, &actual_channels, desired);

    if (!pixels) {
        throw std::runtime_error(
            std::string("Failed to decode image: ") + stbi_failure_reason());
    }

    int out_channels = (desired > 0) ? desired : actual_channels;
    try {
        validate_decoded_dims(width, height, out_channels);
    } catch (...) {
        stbi_image_free(pixels);
        throw;
    }
    auto tensor = hwc_to_chw_tensor(pixels, width, height, out_channels);
    stbi_image_free(pixels);
    return tensor;
}

} // anonymous namespace

auto read_image(const std::string& path, ImageMode mode) -> Tensor {
    int width, height, actual_channels;
    int desired = mode_to_channels(mode);

    uint8_t* pixels = stbi_load(
        path.c_str(), &width, &height, &actual_channels, desired);

    if (!pixels) {
        throw std::runtime_error(
            "Failed to read image '" + path + "': " + stbi_failure_reason());
    }

    int out_channels = (desired > 0) ? desired : actual_channels;
    try {
        validate_decoded_dims(width, height, out_channels);
    } catch (...) {
        stbi_image_free(pixels);
        throw;
    }
    auto tensor = hwc_to_chw_tensor(pixels, width, height, out_channels);
    stbi_image_free(pixels);
    return tensor;
}

void write_image(const Tensor& tensor, const std::string& path, int quality) {
    if (tensor.dtype() != DType::UInt8) {
        throw std::invalid_argument("write_image: tensor must have DType::UInt8");
    }

    auto buf = chw_to_hwc_buffer(tensor);
    auto shape = tensor.shape();

    int channels, height, width;
    if (tensor.ndim() == 3) {
        channels = static_cast<int>(shape[0]);
        height = static_cast<int>(shape[1]);
        width = static_cast<int>(shape[2]);
    } else {
        channels = 1;
        height = static_cast<int>(shape[0]);
        width = static_cast<int>(shape[1]);
    }

    // Reject dimensions that would overflow stb's 32-bit internal size math
    // before handing them to the encoder.
    validate_image_dims(width, height, channels);

    auto ext = get_extension(path);
    int result = 0;

    if (ext == ".jpg" || ext == ".jpeg") {
        result = stbi_write_jpg(path.c_str(), width, height, channels, buf.data(), quality);
    } else if (ext == ".png") {
        result = stbi_write_png(path.c_str(), width, height, channels, buf.data(), width * channels);
    } else if (ext == ".bmp") {
        result = stbi_write_bmp(path.c_str(), width, height, channels, buf.data());
    } else if (ext == ".tga") {
        result = stbi_write_tga(path.c_str(), width, height, channels, buf.data());
    } else {
        throw std::invalid_argument("write_image: unsupported format '" + ext + "'. Use .jpg, .png, .bmp, or .tga");
    }

    if (!result) {
        throw std::runtime_error("Failed to write image to '" + path + "'");
    }
}

auto decode_jpeg(const std::vector<uint8_t>& data, ImageMode mode) -> Tensor {
    check_magic(data.data(), data.size(), ExpectFormat::JPEG);
    return decode_from_memory(data.data(), data.size(), mode);
}

auto decode_png(const std::vector<uint8_t>& data, ImageMode mode) -> Tensor {
    check_magic(data.data(), data.size(), ExpectFormat::PNG);
    return decode_from_memory(data.data(), data.size(), mode);
}

auto encode_jpeg(const Tensor& tensor, int quality) -> std::vector<uint8_t> {
    if (tensor.dtype() != DType::UInt8) {
        throw std::invalid_argument("encode_jpeg: tensor must have DType::UInt8");
    }

    auto buf = chw_to_hwc_buffer(tensor);
    auto shape = tensor.shape();
    int channels = (tensor.ndim() == 3) ? static_cast<int>(shape[0]) : 1;
    int height = (tensor.ndim() == 3) ? static_cast<int>(shape[1]) : static_cast<int>(shape[0]);
    int width = (tensor.ndim() == 3) ? static_cast<int>(shape[2]) : static_cast<int>(shape[1]);

    validate_image_dims(width, height, channels);

    std::vector<uint8_t> result;
    int ok = stbi_write_jpg_to_func(stbi_write_to_vec, &result,
                                     width, height, channels, buf.data(), quality);
    if (!ok) {
        throw std::runtime_error("Failed to encode JPEG");
    }
    return result;
}

auto encode_png(const Tensor& tensor) -> std::vector<uint8_t> {
    if (tensor.dtype() != DType::UInt8) {
        throw std::invalid_argument("encode_png: tensor must have DType::UInt8");
    }

    auto buf = chw_to_hwc_buffer(tensor);
    auto shape = tensor.shape();
    int channels = (tensor.ndim() == 3) ? static_cast<int>(shape[0]) : 1;
    int height = (tensor.ndim() == 3) ? static_cast<int>(shape[1]) : static_cast<int>(shape[0]);
    int width = (tensor.ndim() == 3) ? static_cast<int>(shape[2]) : static_cast<int>(shape[1]);

    validate_image_dims(width, height, channels);

    std::vector<uint8_t> result;
    int ok = stbi_write_png_to_func(stbi_write_to_vec, &result,
                                     width, height, channels, buf.data(), width * channels);
    if (!ok) {
        throw std::runtime_error("Failed to encode PNG");
    }
    return result;
}

} // namespace tenzor::io
