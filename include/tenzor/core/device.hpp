/**
 * @file device.hpp
 * @brief Device specification and management for tensors
 *
 * Defines device types and utilities for managing tensor placement across
 * different hardware backends (CPU, CUDA, ROCm, OneAPI).
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace tenzor {

/**
 * @brief Device specification for tensor placement.
 *
 * Represents a computational device where tensor data resides and
 * operations are executed. Supports multiple backend types with
 * device indexing for multi-GPU systems.
 *
 * @code
 * // Create devices
 * Device cpu_dev = Device::cpu();
 * Device gpu_dev = Device::cuda(0);
 *
 * // Move tensor to device
 * Tensor t = tensor.to(gpu_dev);
 * @endcode
 */
struct Device {
    /**
     * @brief Device backend type enumeration.
     */
    enum class Type : uint8_t {
        CPU,         ///< CPU backend
        CUDA,        ///< NVIDIA CUDA backend
        ROCm,        ///< AMD ROCm backend
        OneAPI,      ///< Intel OneAPI backend
        Vulkan,      ///< Vulkan cross-platform backend
        MPS,         ///< Apple Metal Performance Shaders backend
        COUNT        ///< Sentinel — must be last
    };

    Type type;         ///< Device backend type
    int32_t index{0};  ///< Device index (for multi-device systems)

    /**
     * @brief Create a CPU device.
     *
     * @return Device configured for CPU execution
     *
     * @code
     * Device dev = Device::cpu();
     * @endcode
     */
    static auto cpu() -> Device {
        return Device{Type::CPU, 0};
    }

    /**
     * @brief Create a CUDA device.
     *
     * @param idx GPU device index (default: 0)
     * @return Device configured for CUDA execution
     *
     * @code
     * Device dev = Device::cuda(1);  // Use second GPU
     * @endcode
     */
    static auto cuda(int32_t idx = 0) -> Device {
        return Device{Type::CUDA, idx};
    }

    /**
     * @brief Create a ROCm device.
     *
     * @param idx GPU device index (default: 0)
     * @return Device configured for ROCm execution
     */
    static auto rocm(int32_t idx = 0) -> Device {
        return Device{Type::ROCm, idx};
    }

    /**
     * @brief Create a OneAPI device.
     *
     * @param idx Device index (default: 0)
     * @return Device configured for OneAPI execution
     */
    static auto oneapi(int32_t idx = 0) -> Device {
        return Device{Type::OneAPI, idx};
    }

    /**
     * @brief Create a Vulkan device.
     *
     * @param idx Device index (default: 0)
     * @return Device configured for Vulkan execution
     */
    static auto vulkan(int32_t idx = 0) -> Device {
        return Device{Type::Vulkan, idx};
    }

    /**
     * @brief Create an Apple Metal/MPS device.
     *
     * @param idx Device index (default: 0)
     * @return Device configured for MPS execution
     */
    static auto mps(int32_t idx = 0) -> Device {
        return Device{Type::MPS, idx};
    }

    /**
     * @brief Compare devices for equality.
     *
     * @param other Device to compare with
     * @return true if devices have same type and index
     */
    auto operator==(const Device& other) const -> bool {
        return type == other.type && index == other.index;
    }

    /**
     * @brief Compare devices for inequality.
     *
     * @param other Device to compare with
     * @return true if devices differ in type or index
     */
    auto operator!=(const Device& other) const -> bool {
        return !(*this == other);
    }

    /**
     * @brief Convert device to string representation.
     *
     * @return String like "cpu", "cuda:0", "rocm:1"
     *
     * @code
     * Device dev = Device::cuda(2);
     * std::cout << dev.to_string();  // Prints "cuda:2"
     * @endcode
     */
    auto to_string() const -> std::string {
        switch (type) {
            case Type::CPU: return "cpu";
            case Type::CUDA: return "cuda:" + std::to_string(index);
            case Type::ROCm: return "rocm:" + std::to_string(index);
            case Type::OneAPI: return "oneapi:" + std::to_string(index);
            case Type::Vulkan: return "vulkan:" + std::to_string(index);
            case Type::MPS: return "mps:" + std::to_string(index);
            case Type::COUNT: break;
        }
        return "unknown";
    }

    /**
     * @brief Parse device from string representation.
     *
     * @param str String like "cpu", "cuda:0", "rocm:1"
     * @return Parsed device
     * @throws std::runtime_error if string format is invalid
     *
     * @code
     * Device dev = Device::from_string("cuda:1");
     * @endcode
     */
    static auto from_string(std::string_view str) -> Device;

    /**
     * @brief Synchronize all operations on this device.
     *
     * Blocks until all pending operations on this device have completed.
     * Works uniformly across all backend types (CPU, CUDA, ROCm, OneAPI).
     *
     * @code
     * Device device = Device::cuda(0);
     * // ... perform operations ...
     * device.synchronize();  // Wait for all operations to complete
     * @endcode
     *
     * @note This is the recommended way to synchronize device operations
     *       instead of using backend-specific calls like cudaDeviceSynchronize().
     */
    auto synchronize() const -> void;
};

// Forward declaration — full definition in backend/backend.hpp
struct DeviceInfo;

/**
 * @brief Get hardware properties for a specific device.
 *
 * Queries the backend for detailed device information including
 * memory capacity, compute capability, and feature support.
 *
 * @param device Device to query
 * @return DeviceInfo structure with hardware details
 * @throws std::runtime_error if device backend is not available
 *
 * @code
 * auto props = get_device_properties(Device::cuda(0));
 * std::cout << props.name << ": " << props.total_memory / 1e9 << " GB\n";
 * std::cout << "Compute: " << props.major_version << "." << props.minor_version << "\n";
 * @endcode
 */
auto get_device_properties(const Device& device) -> DeviceInfo;

} // namespace tenzor

// Hash support for std::unordered_map
template<>
struct std::hash<tenzor::Device> {
    auto operator()(const tenzor::Device& device) const -> size_t {
        return std::hash<uint8_t>{}(static_cast<uint8_t>(device.type)) * 31 +
               std::hash<int32_t>{}(device.index);
    }
};
