#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <span>
#include <functional>
#include <unordered_map>
#include "../core/tensor.hpp"
#include "../core/device.hpp"

namespace tenzor {

// Copy kind enumeration
enum class CopyKind {
    HostToHost,
    HostToDevice,
    DeviceToHost,
    DeviceToDevice
};

// Stream handle for async operations
using StreamHandle = void*;

// Operation attributes (generic parameter passing)
using OpAttributes = std::unordered_map<std::string, std::string>;

// Abstract backend interface
class Backend {
public:
    virtual ~Backend() = default;

    // Metadata
    virtual auto name() const -> std::string_view = 0;
    virtual auto device_count() const -> int32_t = 0;
    virtual auto is_available() const -> bool = 0;

    // Memory management
    virtual auto allocate(size_t bytes, int32_t device_id) -> void* = 0;
    virtual auto deallocate(void* ptr) -> void = 0;
    virtual auto copy(void* dst, const void* src, size_t bytes,
                     CopyKind kind) -> void = 0;

    // Synchronization
    virtual auto synchronize(int32_t device_id) -> void = 0;

    // Stream management
    virtual auto create_stream(int32_t device_id) -> StreamHandle = 0;
    virtual auto destroy_stream(StreamHandle stream) -> void = 0;
    virtual auto synchronize_stream(StreamHandle stream) -> void = 0;

    // Kernel dispatch
    virtual auto dispatch(const std::string& op_name,
                         std::span<const Tensor> inputs,
                         const OpAttributes& attrs) -> std::vector<Tensor> = 0;
};

// Backend factory function signature
using BackendFactory = std::unique_ptr<Backend>(*)();

} // namespace tenzor
