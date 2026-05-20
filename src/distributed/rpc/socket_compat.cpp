// socket_compat.cpp
// Platform-specific implementations for the cross-platform socket helpers
// declared in socket_compat.hpp. POSIX needs no out-of-line code; Windows
// requires reference-counted WSAStartup/WSACleanup.

#include "socket_compat.hpp"

#if defined(_WIN32)

#include <atomic>
#include <stdexcept>
#include <string>

namespace tenzor {
namespace distributed {
namespace rpc {

namespace {
std::atomic<int> g_winsock_refcount{0};
}

bool tenzor_rpc_socket_init() {
    if (g_winsock_refcount.fetch_add(1, std::memory_order_acq_rel) > 0) {
        return true;  // already initialised by an earlier caller
    }
    WSADATA wsa_data;
    int rc = WSAStartup(MAKEWORD(2, 2), &wsa_data);
    if (rc != 0) {
        g_winsock_refcount.fetch_sub(1, std::memory_order_acq_rel);
        throw std::runtime_error(
            std::string("WSAStartup failed: ") + socket_strerror(rc));
    }
    return true;
}

void tenzor_rpc_socket_shutdown() {
    if (g_winsock_refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        WSACleanup();
    }
}

}  // namespace rpc
}  // namespace distributed
}  // namespace tenzor

#endif  // _WIN32
