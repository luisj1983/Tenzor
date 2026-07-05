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
#include "../core/device.hpp"
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace tenzor {
namespace nn {

/**
 * @brief Controls where loaded tensors are placed (PyTorch's map_location).
 *
 * - `MapLocation{}` / `MapLocation::cpu()` (default): every tensor is
 *   materialized on the CPU, matching the historical loader behavior — existing
 *   callers are unaffected.
 * - `MapLocation::on(device)`: every tensor is placed on the given device.
 * - `MapLocation::preserve()`: each tensor is placed on the device recorded in
 *   the checkpoint's persistent id (F071), falling back to CPU when the saved
 *   device string is missing or unparseable.
 */
struct MapLocation {
    enum class Mode { CPU, Device, Preserve };

    Mode mode{Mode::CPU};
    tenzor::Device device{tenzor::Device::cpu()};

    MapLocation() = default;

    /// Force all tensors to CPU (historical default behavior).
    static auto cpu() -> MapLocation { return MapLocation{}; }

    /// Place all loaded tensors on an explicit target device.
    static auto on(const tenzor::Device& dev) -> MapLocation {
        return MapLocation{Mode::Device, dev};
    }

    /// Honor each tensor's saved device (as recorded in the checkpoint).
    static auto preserve() -> MapLocation {
        return MapLocation{Mode::Preserve, tenzor::Device::cpu()};
    }

private:
    MapLocation(Mode m, const tenzor::Device& d) : mode(m), device(d) {}
};

/**
 * @brief Load a PyTorch .pth/.pt state dictionary
 *
 * Reads a PyTorch checkpoint file and extracts the tensor state dictionary.
 * The file must be in PyTorch's zip-based format (torch.save default).
 *
 * @param path Path to .pth or .pt file
 * @param map_location Where to place loaded tensors (default: CPU, which keeps
 *        the historical behavior so existing callers are unaffected). Use
 *        `MapLocation::on(dev)` to force a device or `MapLocation::preserve()`
 *        to honor each tensor's saved device.
 * @return Map of parameter names to tensors
 * @throws std::runtime_error if file format is invalid or unsupported
 *
 * @code
 * auto state = load_pytorch_state_dict("resnet50.pth");
 * model.load_state_dict(state);
 *
 * // Load straight onto GPU 0:
 * auto gpu_state = load_pytorch_state_dict(
 *     "resnet50.pth", MapLocation::on(Device::cuda(0)));
 * @endcode
 */
auto load_pytorch_state_dict(const std::string& path,
                             const MapLocation& map_location = MapLocation{})
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
