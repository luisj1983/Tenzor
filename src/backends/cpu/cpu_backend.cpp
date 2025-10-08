#include "tenzor/backend/backend.hpp"
#include <cstring>

namespace tenzor {

class CPUBackend : public Backend {
public:
    auto name() const -> std::string_view override {
        return "cpu";
    }

    auto device_count() const -> int32_t override {
        return 1;
    }

    auto is_available() const -> bool override {
        return true;
    }

    auto allocate(size_t bytes, int32_t device_id) -> void* override {
        #ifdef _WIN32
            return _aligned_malloc(bytes, 64);
        #else
            void* ptr = nullptr;
            posix_memalign(&ptr, 64, bytes);
            return ptr;
        #endif
    }

    auto deallocate(void* ptr) -> void override {
        if (!ptr) return;
        #ifdef _WIN32
            _aligned_free(ptr);
        #else
            free(ptr);
        #endif
    }

    auto copy(void* dst, const void* src, size_t bytes, CopyKind kind) -> void override {
        std::memcpy(dst, src, bytes);
    }

    auto synchronize(int32_t device_id) -> void override {
        // CPU is always synchronized
    }

    auto create_stream(int32_t device_id) -> StreamHandle override {
        return nullptr;
    }

    auto destroy_stream(StreamHandle stream) -> void override {
        // No-op for CPU
    }

    auto synchronize_stream(StreamHandle stream) -> void override {
        // No-op for CPU
    }

    auto dispatch(const std::string& op_name,
                 std::span<const Tensor> inputs,
                 const OpAttributes& attrs) -> std::vector<Tensor> override {
        // TODO: Implement CPU kernel dispatch
        return {};
    }
};

// Export factory function
extern "C" {
    auto create_backend() -> std::unique_ptr<Backend> {
        return std::make_unique<CPUBackend>();
    }
}

} // namespace tenzor
