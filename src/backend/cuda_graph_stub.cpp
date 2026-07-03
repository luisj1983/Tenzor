#include "tenzor/backend/cuda_graph.hpp"

#include <array>
#include <mutex>

namespace tenzor {

// ── GPU graph factory registry ──────────────────────────────────────────────
// This TU is compiled into tenzor_core, so BOTH the JIT (which calls
// create_for) and the GPU backend .so's (which call register_factory when they
// load) share this single registry. Backends are dlopen'd with RTLD_LOCAL, so a
// direct symbol link would only ever reach the stub below — routing through a
// registered function pointer is how the JIT reaches the REAL implementation.
namespace {

// Indexed by Device::Type (small enum). 16 slots is comfortably larger than the
// number of device types.
constexpr int kMaxDeviceTypes = 16;

std::mutex& registry_mutex() {
    static std::mutex m;
    return m;
}

std::array<CUDAGraph::Factory, kMaxDeviceTypes>& registry() {
    static std::array<CUDAGraph::Factory, kMaxDeviceTypes> table{};
    return table;
}

}  // namespace

void CUDAGraph::register_factory(int device_type, Factory factory) {
    if (device_type < 0 || device_type >= kMaxDeviceTypes) return;
    std::lock_guard<std::mutex> lock(registry_mutex());
    registry()[device_type] = factory;
}

auto CUDAGraph::create_for(int device_type, int32_t device_id)
    -> std::unique_ptr<CUDAGraph> {
    Factory f = nullptr;
    if (device_type >= 0 && device_type < kMaxDeviceTypes) {
        std::lock_guard<std::mutex> lock(registry_mutex());
        f = registry()[device_type];
    }
    if (!f) return nullptr;  // backend not loaded / no graph support
    return f(device_id);
}

// Weak/default CUDAGraph::create for the CUDA device type. The CUDA backend .so
// registers its real factory via register_factory(); this remains only so the
// core lib (and any legacy caller of create()) links cleanly when no CUDA
// backend is present. Routes through the registry so a registered CUDA factory
// is used when available.
auto CUDAGraph::create(int32_t device_id) -> std::unique_ptr<CUDAGraph> {
    // Device::Type::CUDA — passed as its integer value to avoid a header dep.
    // (Device::Type: CPU=0, CUDA=1, ... — create_for validates the slot.)
    return create_for(/*CUDA*/ 1, device_id);
}

} // namespace tenzor
