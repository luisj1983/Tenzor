// socket_compat.hpp
// Cross-platform TCP-socket abstractions used by the RPC agent.
//
// The POSIX (Linux + macOS) and Windows (winsock2) socket APIs are 95%
// identical in semantics and 100% different in spelling. This header
// gives the rest of the RPC code a single vocabulary so the same
// implementation can compile on every platform.
//
// Usage:
//   #include "socket_compat.hpp"
//
//   socket_t fd = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
//   if (fd == invalid_socket) {
//       throw std::runtime_error("socket(): " + socket_strerror(socket_errno()));
//   }
//   ...
//   close_socket(fd);
//
// On Windows, `tenzor_rpc_socket_init()` must be called once before any
// socket use (it invokes WSAStartup). It is reference-counted and safe
// to call from every TcpRpcAgent constructor.

#pragma once

#include <cerrno>
#include <cstring>
#include <string>

#if defined(_WIN32)
  // Windows must include winsock2 BEFORE windows.h. Define WIN32_LEAN_AND_MEAN
  // to keep the include surface small and avoid clashes with our names.
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #include <windows.h>
  #pragma comment(lib, "Ws2_32.lib")
#else
  #include <sys/socket.h>
  #include <sys/types.h>
  #include <netinet/in.h>
  #include <netinet/tcp.h>
  #include <arpa/inet.h>
  #include <netdb.h>
  #include <unistd.h>
  #include <fcntl.h>
#endif

namespace tenzor {
namespace distributed {
namespace rpc {

#if defined(_WIN32)
using socket_t = SOCKET;
inline constexpr socket_t invalid_socket = INVALID_SOCKET;
inline int socket_errno() { return ::WSAGetLastError(); }
inline int close_socket(socket_t s) { return ::closesocket(s); }
inline std::string socket_strerror(int err) {
    char* msg = nullptr;
    FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                       FORMAT_MESSAGE_IGNORE_INSERTS,
                   nullptr, err,
                   MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                   reinterpret_cast<LPSTR>(&msg), 0, nullptr);
    std::string s = msg ? msg : "(unknown winsock error)";
    if (msg) LocalFree(msg);
    return s;
}
inline bool socket_would_block() {
    int e = socket_errno();
    return e == WSAEWOULDBLOCK || e == WSAEINPROGRESS;
}
inline bool socket_was_interrupted() {
    return socket_errno() == WSAEINTR;
}
// One-time WSAStartup. Safe to call from every constructor; ref-counted.
bool tenzor_rpc_socket_init();
void tenzor_rpc_socket_shutdown();
#else
using socket_t = int;
inline constexpr socket_t invalid_socket = -1;
inline int socket_errno() { return errno; }
inline int close_socket(socket_t s) { return ::close(s); }
inline std::string socket_strerror(int err) { return std::strerror(err); }
inline bool socket_would_block() {
    return errno == EAGAIN || errno == EWOULDBLOCK || errno == EINPROGRESS;
}
inline bool socket_was_interrupted() { return errno == EINTR; }
inline bool tenzor_rpc_socket_init() { return true; }
inline void tenzor_rpc_socket_shutdown() {}
#endif

}  // namespace rpc
}  // namespace distributed
}  // namespace tenzor
