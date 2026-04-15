/**
 * @file image.cpp
 * @brief Image I/O implementation using stb_image and stb_image_write.
 */

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

    // Transpose HWC -> CHW
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                out_ptr[c * height * width + h * width + w] =
                    data[(h * width + w) * channels + c];
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

    auto t = tensor.contiguous();
    const auto* src = t.data<uint8_t>();
    std::vector<uint8_t> buf(static_cast<size_t>(height * width * channels));

    if (tensor.ndim() == 2) {
        // Grayscale — direct copy
        std::memcpy(buf.data(), src, buf.size());
    } else {
        // CHW -> HWC
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                for (int c = 0; c < channels; ++c) {
                    buf[(h * width + w) * channels + c] =
                        src[c * height * width + h * width + w];
                }
            }
        }
    }
    return buf;
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

auto decode_from_memory(const uint8_t* data, int len, ImageMode mode) -> Tensor {
    int width, height, actual_channels;
    int desired = mode_to_channels(mode);

    uint8_t* pixels = stbi_load_from_memory(
        data, len, &width, &height, &actual_channels, desired);

    if (!pixels) {
        throw std::runtime_error(
            std::string("Failed to decode image: ") + stbi_failure_reason());
    }

    int out_channels = (desired > 0) ? desired : actual_channels;
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
    return decode_from_memory(data.data(), static_cast<int>(data.size()), mode);
}

auto decode_png(const std::vector<uint8_t>& data, ImageMode mode) -> Tensor {
    return decode_from_memory(data.data(), static_cast<int>(data.size()), mode);
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

    std::vector<uint8_t> result;
    int ok = stbi_write_png_to_func(stbi_write_to_vec, &result,
                                     width, height, channels, buf.data(), width * channels);
    if (!ok) {
        throw std::runtime_error("Failed to encode PNG");
    }
    return result;
}

} // namespace tenzor::io
