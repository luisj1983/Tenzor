#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace tenzor {

// Device specification
struct Device {
    enum class Type : uint8_t {
        CPU,
        CUDA,
        ROCm,
        OneAPI
    };

    Type type;
    int32_t index{0};

    // Factory methods
    static auto cpu() -> Device {
        return Device{Type::CPU, 0};
    }

    static auto cuda(int32_t idx = 0) -> Device {
        return Device{Type::CUDA, idx};
    }

    static auto rocm(int32_t idx = 0) -> Device {
        return Device{Type::ROCm, idx};
    }

    static auto oneapi(int32_t idx = 0) -> Device {
        return Device{Type::OneAPI, idx};
    }

    // Comparison
    auto operator==(const Device& other) const -> bool {
        return type == other.type && index == other.index;
    }

    auto operator!=(const Device& other) const -> bool {
        return !(*this == other);
    }

    // String representation
    auto to_string() const -> std::string {
        switch (type) {
            case Type::CPU: return "cpu";
            case Type::CUDA: return "cuda:" + std::to_string(index);
            case Type::ROCm: return "rocm:" + std::to_string(index);
            case Type::OneAPI: return "oneapi:" + std::to_string(index);
        }
        return "unknown";
    }

    static auto from_string(std::string_view str) -> Device;
};

} // namespace tenzor

// Hash support for std::unordered_map
template<>
struct std::hash<tenzor::Device> {
    auto operator()(const tenzor::Device& device) const -> size_t {
        return std::hash<uint8_t>{}(static_cast<uint8_t>(device.type)) ^
               (std::hash<int32_t>{}(device.index) << 1);
    }
};
