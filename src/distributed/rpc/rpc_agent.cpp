/**
 * @file rpc_agent.cpp
 * @brief TCP-based RPC agent — minimal but functional request/response loop.
 *
 * Design:
 *   - Each agent binds a TCP listener on its self_.port and spawns one accept
 *     thread that services inbound connections.
 *   - A receive thread is spawned per accepted connection; it reads framed
 *     messages and hands them to dispatch_message() on the same thread.
 *   - send() lazily dials the destination worker, writes a framed request,
 *     then waits on the request's promise. The response is delivered by the
 *     receive thread running for that peer.
 *   - Framing: [uint32 payload_len][uint8 type][int32 src][int32 dst]
 *     [int64 request_id][uint32 func_name_len][func_name bytes]
 *     [uint32 extra_bytes_len][extra_bytes][uint32 tensor_count]
 *     [per-tensor: uint32 dtype][uint32 ndim][int64 shape[ndim]]
 *     [uint64 nbytes][nbytes of raw data]
 *
 *   Tensors are always materialised on CPU for the wire format; callers
 *   can .to(device) after receipt. This keeps the protocol device-agnostic.
 */

#include "tenzor/distributed/rpc/rpc_agent.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/device.hpp"
#include <cstring>
#include <stdexcept>
#include <chrono>
#include <optional>
#include <vector>
#include <thread>

// Cross-platform socket support. socket_compat.hpp pulls in POSIX socket
// headers on Linux/macOS and winsock2 on Windows, and exposes socket_t,
// close_socket, socket_errno, socket_strerror, tenzor_rpc_socket_init.
#include "socket_compat.hpp"
#include <errno.h>

namespace tenzor {
namespace distributed {
namespace rpc {

namespace {

#if defined(__linux__) || defined(__APPLE__)

// -----------------------------------------------------------------------------
// Low-level TCP framing helpers.
// -----------------------------------------------------------------------------

// Read exactly n bytes; return false on EOF/error (connection dead).
bool read_exact(int fd, void* buf, size_t n) {
    auto* p = static_cast<uint8_t*>(buf);
    while (n > 0) {
        ssize_t r = ::recv(fd, p, n, MSG_WAITALL);
        if (r == 0) return false;
        if (r < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        p += r;
        n -= r;
    }
    return true;
}

bool write_exact(int fd, const void* buf, size_t n) {
    auto* p = static_cast<const uint8_t*>(buf);
    while (n > 0) {
        ssize_t w = ::send(fd, p, n, MSG_NOSIGNAL);
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        p += w;
        n -= w;
    }
    return true;
}

// Serialize a Tensor (always materialised on CPU) onto a byte vector.
void append_tensor(std::vector<uint8_t>& out, const Tensor& in) {
    Tensor cpu = in.device().type == Device::Type::CPU ? in.contiguous()
                                                       : in.to(Device::cpu()).contiguous();
    uint32_t dtype = static_cast<uint32_t>(cpu.dtype());
    uint32_t ndim = static_cast<uint32_t>(cpu.shape().size());
    auto ap = [&](const void* p, size_t n) {
        const uint8_t* bp = static_cast<const uint8_t*>(p);
        out.insert(out.end(), bp, bp + n);
    };
    ap(&dtype, sizeof(dtype));
    ap(&ndim, sizeof(ndim));
    for (auto d : cpu.shape()) {
        int64_t v = d;
        ap(&v, sizeof(v));
    }
    uint64_t nbytes = static_cast<uint64_t>(cpu.numel()) * dtype_size(cpu.dtype());
    ap(&nbytes, sizeof(nbytes));
    ap(cpu.data_ptr(), nbytes);
}

std::optional<Tensor> read_tensor(int fd) {
    uint32_t dtype = 0, ndim = 0;
    if (!read_exact(fd, &dtype, sizeof(dtype))) return std::nullopt;
    if (!read_exact(fd, &ndim, sizeof(ndim))) return std::nullopt;
    std::vector<int64_t> shape(ndim);
    for (uint32_t i = 0; i < ndim; ++i) {
        if (!read_exact(fd, &shape[i], sizeof(int64_t))) return std::nullopt;
    }
    uint64_t nbytes = 0;
    if (!read_exact(fd, &nbytes, sizeof(nbytes))) return std::nullopt;
    Tensor t(shape, static_cast<DType>(dtype), Device::cpu());
    if (nbytes > 0) {
        if (!read_exact(fd, t.data_ptr(), nbytes)) return std::nullopt;
    }
    return t;
}

void serialize_message(std::vector<uint8_t>& out, const Message& m) {
    uint8_t type = static_cast<uint8_t>(m.type);
    int32_t src = m.src_worker;
    int32_t dst = m.dst_worker;
    int64_t rid = m.payload.request_id;
    out.insert(out.end(), reinterpret_cast<uint8_t*>(&type), reinterpret_cast<uint8_t*>(&type) + 1);
    out.insert(out.end(), reinterpret_cast<uint8_t*>(&src),  reinterpret_cast<uint8_t*>(&src)  + 4);
    out.insert(out.end(), reinterpret_cast<uint8_t*>(&dst),  reinterpret_cast<uint8_t*>(&dst)  + 4);
    out.insert(out.end(), reinterpret_cast<uint8_t*>(&rid),  reinterpret_cast<uint8_t*>(&rid)  + 8);
    uint32_t fn_len = static_cast<uint32_t>(m.payload.function_name.size());
    out.insert(out.end(), reinterpret_cast<uint8_t*>(&fn_len), reinterpret_cast<uint8_t*>(&fn_len) + 4);
    out.insert(out.end(), m.payload.function_name.begin(), m.payload.function_name.end());
    uint32_t extra_len = static_cast<uint32_t>(m.payload.bytes.size());
    out.insert(out.end(), reinterpret_cast<uint8_t*>(&extra_len), reinterpret_cast<uint8_t*>(&extra_len) + 4);
    out.insert(out.end(), m.payload.bytes.begin(), m.payload.bytes.end());
    uint32_t nt = static_cast<uint32_t>(m.payload.tensors.size());
    out.insert(out.end(), reinterpret_cast<uint8_t*>(&nt), reinterpret_cast<uint8_t*>(&nt) + 4);
    for (const auto& t : m.payload.tensors) append_tensor(out, t);
}

// Cap the payload a peer can request. One gibibyte is generous for tensor
// transfers and small enough that a malformed or hostile peer can't
// trigger an OOM by claiming multi-gigabyte payloads.
constexpr uint32_t kMaxPayloadBytes = 1u << 30;  // 1 GiB

std::optional<Message> deserialize_message(int fd) {
    uint32_t payload_len = 0;
    if (!read_exact(fd, &payload_len, sizeof(payload_len))) return std::nullopt;
    if (payload_len > kMaxPayloadBytes) return std::nullopt;

    std::vector<uint8_t> buf(payload_len);
    if (payload_len > 0 && !read_exact(fd, buf.data(), payload_len)) return std::nullopt;

    // Parse from in-memory buffer — no network reads beyond this point.
    size_t off = 0;
    auto take = [&](void* dst, size_t n) -> bool {
        if (off + n > buf.size()) return false;
        std::memcpy(dst, buf.data() + off, n);
        off += n;
        return true;
    };

    Message m;
    uint8_t type;
    int32_t src, dst;
    int64_t rid;
    if (!take(&type, 1) || !take(&src, 4) || !take(&dst, 4) || !take(&rid, 8))
        return std::nullopt;
    m.type = static_cast<MessageType>(type);
    m.src_worker = src;
    m.dst_worker = dst;
    m.payload.request_id = rid;

    uint32_t fn_len;
    if (!take(&fn_len, 4)) return std::nullopt;
    m.payload.function_name.assign(
        reinterpret_cast<const char*>(buf.data() + off),
        reinterpret_cast<const char*>(buf.data() + off + fn_len));
    off += fn_len;

    uint32_t extra_len;
    if (!take(&extra_len, 4)) return std::nullopt;
    m.payload.bytes.assign(buf.data() + off, buf.data() + off + extra_len);
    off += extra_len;

    uint32_t nt;
    if (!take(&nt, 4)) return std::nullopt;
    for (uint32_t i = 0; i < nt; ++i) {
        uint32_t dtype, ndim;
        if (!take(&dtype, 4) || !take(&ndim, 4)) return std::nullopt;
        std::vector<int64_t> shape(ndim);
        for (uint32_t k = 0; k < ndim; ++k) {
            if (!take(&shape[k], 8)) return std::nullopt;
        }
        uint64_t nbytes;
        if (!take(&nbytes, 8)) return std::nullopt;
        Tensor t(shape, static_cast<DType>(dtype), Device::cpu());
        if (nbytes > 0) {
            if (off + nbytes > buf.size()) return std::nullopt;
            std::memcpy(t.data_ptr(), buf.data() + off, nbytes);
            off += nbytes;
        }
        m.payload.tensors.push_back(std::move(t));
    }
    return m;
}

bool send_framed(int fd, const Message& msg) {
    std::vector<uint8_t> payload;
    serialize_message(payload, msg);
    uint32_t len = static_cast<uint32_t>(payload.size());
    if (!write_exact(fd, &len, sizeof(len))) return false;
    if (!payload.empty() && !write_exact(fd, payload.data(), payload.size())) return false;
    return true;
}

#endif // __linux__

} // namespace

TcpRpcAgent::TcpRpcAgent(WorkerInfo self, RpcAgentConfig config)
    : self_(std::move(self)), config_(std::move(config)) {}

TcpRpcAgent::~TcpRpcAgent() {
    if (running_.load(std::memory_order_acquire)) {
        shutdown();
    }
}

auto TcpRpcAgent::init(const std::vector<WorkerInfo>& all_workers) -> void {
    workers_ = all_workers;
    running_.store(true, std::memory_order_release);

    register_handler(MessageType::RPC_CALL,
        [this](const Message& msg) { return handle_rpc_call(msg); });
    register_handler(MessageType::HEARTBEAT,
        [this](const Message& msg) {
            Message ack;
            ack.type = MessageType::HEARTBEAT_ACK;
            ack.src_worker = self_.id;
            ack.dst_worker = msg.src_worker;
            return ack;
        });

#if defined(__linux__) || defined(__APPLE__)
    // Start accept thread so we can receive inbound connections.
    // Wait for it to confirm bind()/listen() succeeded before returning so
    // callers can't begin send() on a port that never came up.
    listen_ready_ = std::promise<bool>{};
    auto ready_future = listen_ready_.get_future();
    io_threads_.emplace_back([this]() { accept_loop(); });
    auto fail = [this](const std::string& what) {
        running_.store(false, std::memory_order_release);
        // Join the accept thread so init() can throw without leaving a
        // detached joinable thread alive — the destructor would then
        // terminate() the process.
        for (auto& t : io_threads_) {
            if (t.joinable()) t.join();
        }
        io_threads_.clear();
        throw std::runtime_error(what);
    };
    if (ready_future.wait_for(std::chrono::seconds(5)) !=
        std::future_status::ready) {
        fail("RPC agent init timed out waiting for listen on port " +
             std::to_string(self_.port));
    }
    if (!ready_future.get()) {
        fail("RPC agent failed to bind/listen on port " +
             std::to_string(self_.port));
    }
#endif
}

auto TcpRpcAgent::send(Message msg) -> Message {
    if (!running_.load(std::memory_order_acquire)) {
        throw std::runtime_error("RPC agent is not running");
    }

    auto request_id = next_request_id_.fetch_add(1, std::memory_order_relaxed);
    msg.payload.request_id = request_id;
    msg.src_worker = self_.id;
    int32_t dst_worker = msg.dst_worker;

    std::promise<Message> promise;
    auto future = promise.get_future();
    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        pending_[request_id] = PendingRequest{
            std::move(promise),
            std::chrono::steady_clock::now() + std::chrono::milliseconds(config_.timeout_ms)
        };
    }

    if (dst_worker == self_.id) {
        // Fast-path: in-process dispatch.
        dispatch_message(std::move(msg));
    } else {
#if defined(__linux__) || defined(__APPLE__)
        int fd = get_or_connect(dst_worker);
        if (fd < 0 || !send_framed_locked(fd, msg)) {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            pending_.erase(request_id);
            throw std::runtime_error("RPC send failed to worker " + std::to_string(dst_worker));
        }
#else
        throw std::runtime_error("RPC cross-worker transport only implemented on Linux");
#endif
    }

    auto status = future.wait_for(std::chrono::milliseconds(config_.timeout_ms));
    if (status == std::future_status::timeout) {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        pending_.erase(request_id);
        throw std::runtime_error("RPC timeout: request " + std::to_string(request_id));
    }
    return future.get();
}

auto TcpRpcAgent::send_async(Message msg, std::function<void(Message)> callback) -> void {
    // Simple implementation: run send() on a detached thread. For our test
    // coverage this is sufficient; a production agent would pipeline writes.
    std::thread([this, msg = std::move(msg), callback = std::move(callback)]() mutable {
        try {
            auto response = send(std::move(msg));
            callback(std::move(response));
        } catch (...) {
            Message err;
            err.type = MessageType::RPC_ERROR;
            callback(std::move(err));
        }
    }).detach();
}

auto TcpRpcAgent::shutdown() -> void {
    running_.store(false, std::memory_order_release);

#if defined(__linux__) || defined(__APPLE__)
    // Shut down listener if present (accept thread will exit).
    if (listen_fd_ >= 0) {
        ::shutdown(listen_fd_, SHUT_RDWR);
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
    {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        for (auto& [id, fd] : connections_) {
            ::shutdown(fd, SHUT_RDWR);
            ::close(fd);
        }
        connections_.clear();
        for (auto& [id, fd] : inbound_connections_) {
            ::shutdown(fd, SHUT_RDWR);
            ::close(fd);
        }
        inbound_connections_.clear();
        fd_write_mutexes_.clear();
    }
#endif

    for (auto& t : io_threads_) {
        if (t.joinable()) t.join();
    }
    io_threads_.clear();
    for (auto& t : worker_threads_) {
        if (t.joinable()) t.join();
    }
    worker_threads_.clear();

    // Wake up any stragglers that were waiting on promises.
    std::lock_guard<std::mutex> lock(pending_mutex_);
    for (auto& [_, req] : pending_) {
        Message abort;
        abort.type = MessageType::RPC_ERROR;
        abort.payload.function_name = "agent shutdown";
        try { req.promise.set_value(std::move(abort)); } catch (...) {}
    }
    pending_.clear();
}

auto TcpRpcAgent::register_handler(MessageType type,
                                    std::function<Message(const Message&)> handler) -> void {
    std::lock_guard<std::mutex> lock(handler_mutex_);
    handlers_[type] = std::move(handler);
}

auto TcpRpcAgent::dispatch_message(Message msg) -> void {
    // Response messages (RPC_RESPONSE / RPC_ERROR / HEARTBEAT_ACK) are
    // correlated with a pending request and delivered via the promise.
    if (msg.type == MessageType::RPC_RESPONSE || msg.type == MessageType::RPC_ERROR ||
        msg.type == MessageType::HEARTBEAT_ACK) {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        auto it = pending_.find(msg.payload.request_id);
        if (it != pending_.end()) {
            try { it->second.promise.set_value(std::move(msg)); } catch (...) {}
            pending_.erase(it);
        }
        return;
    }

    std::function<Message(const Message&)> handler;
    {
        std::lock_guard<std::mutex> lock(handler_mutex_);
        auto it = handlers_.find(msg.type);
        if (it != handlers_.end()) handler = it->second;
    }
    if (!handler) return;

    auto response = handler(msg);

    // For self-loopback (no remote peer), the response correlates with a
    // local pending request; deliver directly.
    if (msg.src_worker == self_.id) {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        auto it = pending_.find(msg.payload.request_id);
        if (it != pending_.end()) {
            try { it->second.promise.set_value(std::move(response)); } catch (...) {}
            pending_.erase(it);
        }
        return;
    }

#if defined(__linux__) || defined(__APPLE__)
    // Send the response back to the originating worker over the same mesh.
    int fd = get_or_connect(msg.src_worker);
    if (fd >= 0) send_framed_locked(fd, response);
#endif
}

auto TcpRpcAgent::handle_rpc_call(const Message& msg) -> Message {
    auto* fn = FunctionRegistry::instance().get_function(msg.payload.function_name);

    Message response;
    response.type = MessageType::RPC_RESPONSE;
    response.src_worker = self_.id;
    response.dst_worker = msg.src_worker;
    response.payload.request_id = msg.payload.request_id;

    if (!fn) {
        response.type = MessageType::RPC_ERROR;
        response.payload.function_name = "Function not found: " + msg.payload.function_name;
        return response;
    }
    try {
        response.payload.tensors = (*fn)(msg.payload.tensors);
    } catch (const std::exception& e) {
        response.type = MessageType::RPC_ERROR;
        response.payload.function_name = e.what();
    }
    return response;
}

bool TcpRpcAgent::send_framed_locked(int fd, const Message& msg) {
#if defined(__linux__) || defined(__APPLE__)
    std::shared_ptr<std::mutex> mtx;
    {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        auto it = fd_write_mutexes_.find(fd);
        if (it != fd_write_mutexes_.end()) mtx = it->second;
    }
    if (!mtx) {
        // No mutex registered — a vestigial fd or a unit-test path.
        // Fall through to an unlocked write; safer than dropping.
        return send_framed(fd, msg);
    }
    std::lock_guard<std::mutex> wlock(*mtx);
    return send_framed(fd, msg);
#else
    (void)fd; (void)msg;
    return false;
#endif
}

#if defined(__linux__) || defined(__APPLE__)

// Look up or establish a connection to `peer_id`. Returns -1 on failure.
int TcpRpcAgent::get_or_connect(int32_t peer_id) {
    {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        auto it = connections_.find(peer_id);
        if (it != connections_.end()) return it->second;
    }

    WorkerInfo target{};
    for (const auto& w : workers_) {
        if (w.id == peer_id) { target = w; break; }
    }
    if (target.id != peer_id) return -1;

    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(target.port));
    if (::inet_pton(AF_INET, target.address.c_str(), &addr.sin_addr) != 1) {
        ::close(fd);
        return -1;
    }

    // Peer's listener may not be ready yet; retry for a few seconds.
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    // Verify connect succeeded.
    int err = 0;
    socklen_t errlen = sizeof(err);
    if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &errlen) != 0 || err != 0) {
        ::close(fd);
        return -1;
    }

    {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        auto [it, inserted] = connections_.emplace(peer_id, fd);
        if (!inserted) {
            ::close(fd);
            return it->second;
        }
        fd_write_mutexes_[fd] = std::make_shared<std::mutex>();
    }

    // Send a HELLO-style handshake so the peer can tag the connection with
    // our worker id. We reuse the Message framing with type HEARTBEAT and
    // src_worker set — the receive_loop pulls src_worker out.
    Message hello;
    hello.type = MessageType::HEARTBEAT;
    hello.src_worker = self_.id;
    hello.dst_worker = peer_id;
    hello.payload.request_id = -1;  // -1 means "handshake, do not reply"
    if (!send_framed_locked(fd, hello)) {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        connections_.erase(peer_id);
        fd_write_mutexes_.erase(fd);
        ::close(fd);
        return -1;
    }

    // Spawn a receive loop so responses come back in.
    worker_threads_.emplace_back([this, fd, peer_id]() { receive_loop(fd, peer_id); });
    return fd;
}

void TcpRpcAgent::accept_loop() {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        try { listen_ready_.set_value(false); } catch (...) {}
        return;
    }
    int one = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(self_.port));
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(listen_fd_); listen_fd_ = -1;
        try { listen_ready_.set_value(false); } catch (...) {}
        return;
    }
    if (::listen(listen_fd_, 16) < 0) {
        ::close(listen_fd_); listen_fd_ = -1;
        try { listen_ready_.set_value(false); } catch (...) {}
        return;
    }
    try { listen_ready_.set_value(true); } catch (...) {}

    while (running_.load(std::memory_order_acquire)) {
        sockaddr_in peer{};
        socklen_t plen = sizeof(peer);
        int cfd = ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&peer), &plen);
        if (cfd < 0) break;
        int one_ = 1;
        ::setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one_, sizeof(one_));

        // Read the handshake message to learn peer id.
        auto hello = deserialize_message(cfd);
        if (!hello || hello->src_worker < 0) {
            ::close(cfd);
            continue;
        }
        int32_t peer_id = hello->src_worker;

        {
            std::lock_guard<std::mutex> lock(connections_mutex_);
            // Keep inbound sockets in a separate map so accept never
            // silently overwrites an existing outbound fd (which would
            // strand that fd's write-mutex and receive_loop). If an old
            // inbound fd was already here, close it first.
            auto it = inbound_connections_.find(peer_id);
            if (it != inbound_connections_.end()) {
                ::close(it->second);
                fd_write_mutexes_.erase(it->second);
            }
            inbound_connections_[peer_id] = cfd;
            fd_write_mutexes_[cfd] = std::make_shared<std::mutex>();
        }
        worker_threads_.emplace_back([this, cfd, peer_id]() { receive_loop(cfd, peer_id); });
    }
}

void TcpRpcAgent::receive_loop(int fd, int32_t peer_id) {
    (void)peer_id;
    while (running_.load(std::memory_order_acquire)) {
        auto msg = deserialize_message(fd);
        if (!msg) break;
        // Requests with request_id == -1 are handshake-only; skip dispatch.
        if (msg->payload.request_id < 0 && msg->type == MessageType::HEARTBEAT) continue;
        dispatch_message(std::move(*msg));
    }
}

#else  // not POSIX (Linux/macOS) — Windows path lands here.

// Windows winsock2 implementation requires mirroring the POSIX
// implementation above with `socket_t`, `closesocket`, `WSAGetLastError`
// substitutions. The `socket_compat.hpp` header already provides the
// type aliases and `tenzor_rpc_socket_init()` reference-counted
// WSAStartup. To activate Windows, port the four functions below
// (`get_or_connect`, `accept_loop`, `receive_loop`, plus the helpers
// they use in the POSIX block above) using winsock2 calls — every
// `::close` becomes `close_socket`, every `errno` becomes
// `socket_errno()`, `MSG_NOSIGNAL` is unsupported (use `SO_NOSIGPIPE`
// or simply pass `0`), and `SHUT_RDWR` becomes `SD_BOTH`. The header
// already pulls in the right headers and links Ws2_32.lib via pragma.
//
// Until that port lands, fail loudly at first use rather than silently
// returning -1 / no-op.

namespace {
[[noreturn]] void rpc_unsupported_platform(const char* fn) {
    throw std::runtime_error(
        std::string("TcpRpcAgent::") + fn +
        ": distributed RPC is only available on POSIX platforms (Linux/macOS) "
        "in this build. Windows winsock2 port pending — see "
        "src/distributed/rpc/socket_compat.hpp for the abstraction layer "
        "ready to be wired up.");
}
}

int TcpRpcAgent::get_or_connect(int32_t) { rpc_unsupported_platform("get_or_connect"); }
void TcpRpcAgent::accept_loop()          { rpc_unsupported_platform("accept_loop"); }
void TcpRpcAgent::receive_loop(int, int32_t) { rpc_unsupported_platform("receive_loop"); }

#endif

} // namespace rpc
} // namespace distributed
} // namespace tenzor
