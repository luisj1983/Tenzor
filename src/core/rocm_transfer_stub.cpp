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

}  // namespace rocm_transfer
}  // namespace tenzor
