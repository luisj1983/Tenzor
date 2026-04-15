/**
 * @file image.hpp
 * @brief Image I/O operations for reading and writing common image formats.
 *
 * Supports JPEG, PNG, BMP, and TGA via stb_image/stb_image_write.
 * Images are returned as uint8 tensors in CHW layout by default.
 */

#pragma once

#include "tenzor/core/tensor.hpp"
#include <string>
#include <vector>
#include <cstdint>

namespace tenzor::io {

/**
 * @brief Image reading mode.
 */
enum class ImageMode {
    RGB,        ///< 3-channel RGB
    RGBA,       ///< 4-channel RGBA
    GRAYSCALE,  ///< 1-channel grayscale
    UNCHANGED   ///< Keep original channels
};

/**
 * @brief Read an image file into a Tensor.
 *
 * Returns a uint8 tensor of shape (C, H, W) where C depends on mode.
 * Supported formats: JPEG, PNG, BMP, TGA, GIF, PSD, HDR, PIC, PNM.
 *
 * @param path Path to image file
 * @param mode Desired output channel mode
 * @return Tensor of shape (C, H, W) with DType::UInt8
 */
auto read_image(const std::string& path, ImageMode mode = ImageMode::RGB) -> Tensor;

/**
 * @brief Write a Tensor to an image file.
 *
 * Input should be a uint8 tensor of shape (C, H, W) or (H, W) for grayscale.
 * Format is inferred from file extension (.jpg, .png, .bmp, .tga).
 *
 * @param tensor Image tensor of shape (C, H, W) or (H, W)
 * @param path Output file path
 * @param quality JPEG quality (1-100, default 95). Ignored for non-JPEG.
 */
void write_image(const Tensor& tensor, const std::string& path, int quality = 95);

/**
 * @brief Decode JPEG from memory buffer into a Tensor.
 *
 * @param data Raw JPEG bytes
 * @param mode Desired output channel mode
 * @return Tensor of shape (C, H, W) with DType::UInt8
 */
auto decode_jpeg(const std::vector<uint8_t>& data, ImageMode mode = ImageMode::RGB) -> Tensor;

/**
 * @brief Decode PNG from memory buffer into a Tensor.
 *
 * @param data Raw PNG bytes
 * @param mode Desired output channel mode
 * @return Tensor of shape (C, H, W) with DType::UInt8
 */
auto decode_png(const std::vector<uint8_t>& data, ImageMode mode = ImageMode::RGB) -> Tensor;

/**
 * @brief Encode a Tensor to JPEG bytes.
 *
 * @param tensor Image tensor of shape (C, H, W) with DType::UInt8
 * @param quality JPEG quality (1-100)
 * @return Encoded JPEG bytes
 */
auto encode_jpeg(const Tensor& tensor, int quality = 95) -> std::vector<uint8_t>;

/**
 * @brief Encode a Tensor to PNG bytes.
 *
 * @param tensor Image tensor of shape (C, H, W) with DType::UInt8
 * @return Encoded PNG bytes
 */
auto encode_png(const Tensor& tensor) -> std::vector<uint8_t>;

} // namespace tenzor::io
