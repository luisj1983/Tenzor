/**
 * @file mps_backend.hpp
 * @brief Apple Metal Performance Shaders backend
 *
 * Implements the Backend interface using Apple's Metal framework for GPU
 * compute on Apple Silicon (M-series) and AMD GPUs on Mac.
 *
 * Key design decisions:
 * - Uses MTLResourceStorageModeShared on Apple Silicon (unified memory)
 * - Command batching (32 ops per batch) to reduce submission overhead
 * - Thread-local device tracking (Metal has no global current device)
 */

#pragma once

#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/dispatch_table.hpp"
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace tenzor {
namespace mps {

/**
 * @brief MPS backend implementation.
 *
 * Metal objects are stored as void* to avoid requiring Objective-C headers
 * in this C++ header. The actual Metal objects (id<MTLDevice>, etc.) are
 * managed in the .mm implementation file.
 */
class MPSBackend : public Backend {
public:
    MPSBackend();
    ~MPSBackend() override;

    // Backend interface
    auto name() const -> std::string override { return "mps"; }
    auto device_count() const -> int override;
    auto is_available() const -> bool override;
    auto get_device_info(int device_id) const -> DeviceInfo override;

    auto allocate(size_t size_bytes, int device_id = 0) -> void* override;
    auto deallocate(void* ptr) -> void override;
    auto copy(void* dst, const void* src, size_t bytes, CopyKind kind) -> void override;
    auto memset(void* ptr, int value, size_t bytes) -> void override;
    auto synchronize(int device_id = 0) -> void override;

    auto set_device(int device_id) -> void override;
    auto get_current_device() const -> int override;

    auto register_kernels(BackendDispatchTable& table) -> void override;

    // Command batching
    static constexpr size_t BATCH_SIZE_THRESHOLD = 32;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Register all MPS kernels with the dispatch table
auto register_mps_kernels(BackendDispatchTable& table) -> void;

} // namespace mps
} // namespace tenzor
