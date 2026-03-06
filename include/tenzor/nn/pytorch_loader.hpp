/**
 * @file pytorch_loader.hpp
 * @brief PyTorch .pth/.pt file loader (read-only)
 *
 * Loads PyTorch saved state dictionaries into Tenzor tensor maps.
 * Supports the zip-based format used by torch.save() with protocol 2.
 *
 * Limitations:
 * - Read-only (no writing .pth files)
 * - Only loads tensors (not arbitrary Python objects)
 * - Requires tensors to be stored as torch.FloatTensor, torch.DoubleTensor,
 *   torch.HalfTensor, torch.LongTensor, etc.
 * - Does not execute arbitrary pickle opcodes (safety)
 */

#pragma once

#include "../core/tensor.hpp"
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace tenzor {
namespace nn {

/**
 * @brief Load a PyTorch .pth/.pt state dictionary
 *
 * Reads a PyTorch checkpoint file and extracts the tensor state dictionary.
 * The file must be in PyTorch's zip-based format (torch.save default).
 *
 * @param path Path to .pth or .pt file
 * @return Map of parameter names to tensors
 * @throws std::runtime_error if file format is invalid or unsupported
 *
 * @code
 * auto state = load_pytorch_state_dict("resnet50.pth");
 * model.load_state_dict(state);
 * @endcode
 */
auto load_pytorch_state_dict(const std::string& path)
    -> std::unordered_map<std::string, Tensor>;

/**
 * @brief Check if a file appears to be a PyTorch checkpoint
 *
 * Checks for the ZIP magic number (PK\x03\x04) at the start of the file.
 *
 * @param path File path to check
 * @return true if file starts with ZIP magic number
 */
auto is_pytorch_file(const std::string& path) -> bool;

/**
 * @brief List tensor names in a PyTorch checkpoint without loading data
 *
 * @param path Path to .pth or .pt file
 * @return Vector of tensor parameter names
 */
auto list_pytorch_tensors(const std::string& path) -> std::vector<std::string>;

} // namespace nn
} // namespace tenzor
