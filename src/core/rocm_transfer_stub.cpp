// Stub implementation of the ROCm transfer API, linked into tenzor_core only
// when the ROCm backend is NOT built. transfer_engine.cpp references these
// symbols unconditionally (the ROCm device branch is never taken without a ROCm
// device, but the symbols must resolve at link time). The real HIP-backed
// version lives in rocm_transfer.hip.cpp.
#include "tenzor/core/rocm_transfer.hpp"

namespace tenzor {
namespace rocm_transfer {

auto available() -> bool { return false; }
auto h2d_async(void*, const void*, std::size_t, int) -> void* { return nullptr; }
auto d2h_async(void*, const void*, std::size_t, int) -> void* { return nullptr; }
auto event_sync(void*) -> void {}
auto event_ready(void*) -> bool { return true; }

auto stream_create() -> void* { return nullptr; }
auto stream_destroy(void*) -> void {}
auto event_create() -> void* { return nullptr; }
auto event_destroy(void*) -> void {}
auto event_record(void*, void*) -> void {}
auto stream_wait_event(void*, void*) -> void {}
auto stream_synchronize(void*) -> void {}
auto mem_get_info(int, std::size_t*, std::size_t*) -> bool { return false; }

auto host_register(void*, std::size_t) -> bool { return false; }
auto host_unregister(void*) -> void {}
auto host_malloc(std::size_t) -> void* { return nullptr; }
auto host_free(void*) -> void {}

}  // namespace rocm_transfer
}  // namespace tenzor
