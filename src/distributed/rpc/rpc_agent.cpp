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
#include "tenzor/distributed/rpc/rref.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/utils/log.hpp"
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
        // send() returning 0 with n>0 makes no progress; treat it as a
        // closed/failed connection rather than spinning forever (symmetric
        // with read_exact's r==0 == EOF handling).
        if (w == 0) return false;
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

// NOTE: a standalone read_tensor(int fd) helper used to live here but had zero
// callers and diverged from the hardened inline parse in deserialize_message()
// (it lacked the rank bound, negative-dim rejection, numel-overflow guard, and
// numel*elem_size == nbytes consistency check). It was deleted so the only
// tensor parser is the bounds-checked path in deserialize_message(); any future
// socket tensor read must go through that function.

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
    // Bound the variable-length field against the remaining buffer BEFORE the
    // assign (subtraction form avoids integer overflow). A peer controls
    // fn_len/extra_len, so without this the assign reads arbitrary heap past buf.
    if (fn_len > buf.size() - off) return std::nullopt;
    m.payload.function_name.assign(
        reinterpret_cast<const char*>(buf.data() + off),
        reinterpret_cast<const char*>(buf.data() + off + fn_len));
    off += fn_len;

    uint32_t extra_len;
    if (!take(&extra_len, 4)) return std::nullopt;
    if (extra_len > buf.size() - off) return std::nullopt;
    m.payload.bytes.assign(buf.data() + off, buf.data() + off + extra_len);
    off += extra_len;

    uint32_t nt;
    if (!take(&nt, 4)) return std::nullopt;
    for (uint32_t i = 0; i < nt; ++i) {
        uint32_t dtype, ndim;
        if (!take(&dtype, 4) || !take(&ndim, 4)) return std::nullopt;
        if (ndim > 4096) return std::nullopt;  // implausible rank
        std::vector<int64_t> shape(ndim);
        for (uint32_t k = 0; k < ndim; ++k) {
            if (!take(&shape[k], 8)) return std::nullopt;
        }
        // Validate dims and compute the expected byte size with checked math so
        // a crafted shape/nbytes can't desync from the allocation.
        int64_t numel = 1;
        for (int64_t d : shape) {
            if (d < 0) return std::nullopt;
            if (__builtin_mul_overflow(numel, d, &numel)) return std::nullopt;
        }
        uint64_t nbytes;
        if (!take(&nbytes, 8)) return std::nullopt;
        // Subtraction form: off <= buf.size(), so this cannot overflow (unlike
        // off + nbytes, which wraps for nbytes near UINT64_MAX and bypassed the
        // bound, then memcpy'd ~exabytes out of buf).
        if (nbytes > buf.size() - off) return std::nullopt;
        // Reject out-of-range dtypes: dtype_size() returns 0 only for an invalid
        // DType. Without this, a peer sending an invalid dtype with nbytes==0
        // would pass the size check below (expected==0) and construct a Tensor
        // with a garbage enum, which then hits UB in downstream dispatch.
        size_t elem_size = dtype_size(static_cast<DType>(dtype));
        if (elem_size == 0) return std::nullopt;
        // Checked multiply: numel is bounded only by int64 overflow above
        // (~9.2e18), so numel * elem_size (elem_size up to 16 for Complex128)
        // can wrap past UINT64_MAX to a small value that would spuriously match
        // a crafted small nbytes and bypass the size check below. Use a
        // checked multiply so an oversized shape is rejected here, before the
        // Tensor allocation, rather than silently desyncing nbytes/expected.
        uint64_t expected;
        if (__builtin_mul_overflow(static_cast<uint64_t>(numel),
                                   static_cast<uint64_t>(elem_size), &expected)) {
            return std::nullopt;
        }
        if (nbytes != expected) return std::nullopt;
        // The TensorImpl ctor throws (overflow_error / runtime_error) for a
        // shape whose allocation is too large; catch it here so a hostile or
        // corrupt frame is dropped (std::nullopt) instead of letting the
        // exception escape the receive/accept thread callable and call
        // std::terminate() (remote DoS). Any allocation/ctor failure means the
        // frame is unusable, so treat it like a malformed message.
        std::optional<Tensor> t;
        try {
            t.emplace(shape, static_cast<DType>(dtype), Device::cpu());
        } catch (const std::exception&) {
            return std::nullopt;
        }
        if (nbytes > 0) {
            std::memcpy(t->data_ptr(), buf.data() + off, nbytes);
            off += nbytes;
        }
        m.payload.tensors.push_back(std::move(*t));
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
            // Echo the request_id so dispatch_message can correlate this ACK
            // with the originating send()'s pending_ entry; otherwise every
            // heartbeat past the first (id 0) misses pending_ and blocks until
            // timeout, falsely reporting live peers dead.
            ack.payload.request_id = msg.payload.request_id;
            return ack;
        });
    // RRef remote-fetch: the rref_id rides in payload.bytes (8 bytes) because
    // the wire's request_id field is reserved for request/response correlation
    // and is rewritten by send(). The owner returns the stored tensor.
    register_handler(MessageType::RREF_FETCH,
        [this](const Message& msg) {
            Message response;
            response.type = MessageType::RPC_RESPONSE;
            response.src_worker = self_.id;
            response.dst_worker = msg.src_worker;
            response.payload.request_id = msg.payload.request_id;
            if (msg.payload.bytes.size() != sizeof(int64_t)) {
                response.type = MessageType::RPC_ERROR;
                response.payload.function_name = "RREF_FETCH: malformed rref id";
                return response;
            }
            int64_t rref_id = 0;
            std::memcpy(&rref_id, msg.payload.bytes.data(), sizeof(int64_t));
            try {
                response.payload.tensors.push_back(RRefStore::instance().fetch(rref_id));
            } catch (const std::exception& e) {
                response.type = MessageType::RPC_ERROR;
                response.payload.function_name = e.what();
            }
            return response;
        });
    // RRef remote GC: delete the owner's stored tensor. Sent fire-and-forget via
    // send_async; we still return an ack so the response path stays uniform.
    register_handler(MessageType::RREF_DELETE,
        [this](const Message& msg) {
            Message response;
            response.type = MessageType::RPC_RESPONSE;
            response.src_worker = self_.id;
            response.dst_worker = msg.src_worker;
            response.payload.request_id = msg.payload.request_id;
            if (msg.payload.bytes.size() == sizeof(int64_t)) {
                int64_t rref_id = 0;
                std::memcpy(&rref_id, msg.payload.bytes.data(), sizeof(int64_t));
                RRefStore::instance().remove(rref_id);
            }
            return response;
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
        pending_[request_id] = PendingRequest{std::move(promise)};
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
    // Run send() on a detached thread. The thread dereferences `this` for the
    // full duration of a blocking send(), so we register it in inflight_async_
    // BEFORE detaching and decrement+notify on exit; shutdown() drains this
    // counter before members are destroyed to avoid a use-after-free.
    {
        std::lock_guard<std::mutex> lock(async_mutex_);
        ++inflight_async_;
    }
    const int64_t request_id = msg.payload.request_id;  // capture before move
    std::thread([this, msg = std::move(msg), callback = std::move(callback),
                 request_id]() mutable {
        // Decrement and notify on every exit path so shutdown()'s drain wakes.
        struct InflightGuard {
            TcpRpcAgent* self;
            ~InflightGuard() {
                {
                    std::lock_guard<std::mutex> lock(self->async_mutex_);
                    --self->inflight_async_;
                }
                self->async_cv_.notify_all();
            }
        } guard{this};
        try {
            auto response = send(std::move(msg));
            callback(std::move(response));
        } catch (const std::exception& e) {
            // Propagate the request id and diagnostic so the callback can
            // correlate and report the failure (the old empty RPC_ERROR lost both).
            // The diagnostic MUST go in function_name: every RPC_ERROR consumer
            // (rpc_async / rpc_sync) reads the error text from function_name, not
            // from payload.bytes, so writing it to bytes silently dropped the cause.
            Message err;
            err.type = MessageType::RPC_ERROR;
            err.payload.request_id = request_id;
            err.payload.function_name = e.what();
            callback(std::move(err));
        } catch (...) {
            Message err;
            err.type = MessageType::RPC_ERROR;
            err.payload.request_id = request_id;
            err.payload.function_name = "unknown RPC send error";
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
        // Collect all fds and their per-fd write-mutexes, then erase the maps,
        // all under connections_mutex_. We must NOT hold connections_mutex_
        // while acquiring a write-mutex below: an in-flight send_framed_locked()
        // holds the write-mutex first and then re-acquires connections_mutex_,
        // so the reverse order here would deadlock.
        std::vector<std::pair<int, std::shared_ptr<std::mutex>>> to_close;
        {
            std::lock_guard<std::mutex> lock(connections_mutex_);
            auto collect = [&](int fd) {
                std::shared_ptr<std::mutex> mtx;
                auto it = fd_write_mutexes_.find(fd);
                if (it != fd_write_mutexes_.end()) mtx = it->second;
                to_close.emplace_back(fd, std::move(mtx));
            };
            for (auto& [id, fd] : connections_) collect(fd);
            for (auto& [id, fd] : inbound_connections_) collect(fd);
            // Erase the maps now: a writer that has not yet acquired its
            // write-mutex will, once it does, find its entry gone and abort
            // before touching the (about-to-be-closed) fd.
            connections_.clear();
            inbound_connections_.clear();
            fd_write_mutexes_.clear();
        }
        // Drain in-flight writers per fd, then close. Locking the write-mutex
        // waits for any writer mid-send_framed() to finish before we close the
        // fd, preventing use-after-close / fd-integer reuse.
        for (auto& [fd, mtx] : to_close) {
            if (mtx) {
                std::lock_guard<std::mutex> wlock(*mtx);
                ::shutdown(fd, SHUT_RDWR);
                ::close(fd);
            } else {
                ::shutdown(fd, SHUT_RDWR);
                ::close(fd);
            }
        }
    }
#endif

    // Wake up any stragglers that were waiting on promises FIRST, so that any
    // in-flight send()/send_async() blocked on future.wait_for() returns
    // immediately rather than blocking shutdown for the full timeout. This must
    // precede the async drain below, otherwise a detached send_async thread
    // waiting on its own pending promise would never complete.
    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        for (auto& [_, req] : pending_) {
            Message abort;
            abort.type = MessageType::RPC_ERROR;
            abort.payload.function_name = "agent shutdown";
            try { req.promise.set_value(std::move(abort)); } catch (...) {}
        }
        pending_.clear();
    }

    // Drain detached async-send threads before destroying members. They
    // dereference `this` until they finish, so this must complete before the
    // destructor proceeds.
    {
        std::unique_lock<std::mutex> lock(async_mutex_);
        async_cv_.wait(lock, [this] { return inflight_async_ == 0; });
    }

    for (auto& t : io_threads_) {
        if (t.joinable()) t.join();
    }
    io_threads_.clear();
    {
        std::lock_guard<std::mutex> lock(threads_mutex_);
        for (auto& t : worker_threads_) {
            if (t.joinable()) t.join();
        }
        worker_threads_.clear();
    }
}

auto TcpRpcAgent::register_handler(MessageType type,
                                    std::function<Message(const Message&)> handler) -> void {
    std::lock_guard<std::mutex> lock(handler_mutex_);
    handlers_[type] = std::move(handler);
}

auto TcpRpcAgent::dispatch_message(Message msg, int inbound_fd) -> void {
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

    // A request frame must be addressed to this worker. Connections are
    // point-to-point and a worker only reads on its own sockets, so a frame
    // carrying a foreign dst_worker indicates a buggy or malicious sender;
    // execute nothing and drop it rather than running the handler and
    // generating a response for an id that is not ours.
    if (msg.dst_worker != self_.id) {
        TENZOR_LOG_WARN(
            "TcpRpcAgent: dropping message addressed to worker {} (self is {})",
            msg.dst_worker, self_.id);
        return;
    }

    std::function<Message(const Message&)> handler;
    {
        std::lock_guard<std::mutex> lock(handler_mutex_);
        auto it = handlers_.find(msg.type);
        if (it != handlers_.end()) handler = it->second;
    }
    if (!handler) return;

    // Run the handler with a catch-all guard. handle_rpc_call already converts
    // exceptions to RPC_ERROR, but other registered handlers (or future ones)
    // may throw; an escaping exception here would unwind into the receive
    // thread callable and call std::terminate(). On a non-std::exception we
    // still send back an RPC_ERROR so the caller's send() unblocks instead of
    // timing out.
    Message response;
    try {
        response = handler(msg);
    } catch (const std::exception& e) {
        response.type = MessageType::RPC_ERROR;
        response.src_worker = self_.id;
        response.dst_worker = msg.src_worker;
        response.payload.request_id = msg.payload.request_id;
        response.payload.function_name = e.what();
    } catch (...) {
        response.type = MessageType::RPC_ERROR;
        response.src_worker = self_.id;
        response.dst_worker = msg.src_worker;
        response.payload.request_id = msg.payload.request_id;
        response.payload.function_name = "RPC handler threw a non-standard exception";
    }

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
    // Reply on the inbound socket that carried the request when available: it is
    // already connected and avoids requiring a symmetric, fully reachable mesh
    // (the requester need not be dialable as an outbound peer). Fall back to
    // dialing the originating worker only if there is no inbound fd or the
    // inbound write fails. If the response cannot be delivered at all, log it so
    // the requester's blocked send() — which will time out — has a diagnostic
    // instead of failing silently.
    bool sent = false;
    if (inbound_fd >= 0) {
        sent = send_framed_locked(inbound_fd, response);
    }
    if (!sent) {
        int fd = get_or_connect(msg.src_worker);
        if (fd >= 0) {
            sent = send_framed_locked(fd, response);
        }
    }
    if (!sent) {
        TENZOR_LOG_ERROR(
            "TcpRpcAgent: failed to deliver RPC response to worker {} "
            "(request {}); peer will block until timeout",
            msg.src_worker, msg.payload.request_id);
    }
#endif
}

auto TcpRpcAgent::handle_rpc_call(const Message& msg) -> Message {
    auto fn = FunctionRegistry::instance().get_function(msg.payload.function_name);

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
    } catch (...) {
        // A registered RpcFunction may throw something not derived from
        // std::exception (int, string literal, custom type). Without this
        // catch-all the exception would escape handle_rpc_call -> the handler
        // lambda -> dispatch_message -> the receive thread callable and call
        // std::terminate(), taking down the whole agent. Convert it to an
        // RPC_ERROR so only the offending call fails.
        response.type = MessageType::RPC_ERROR;
        response.payload.function_name = "RPC function threw a non-standard exception";
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
        // No write-mutex registered for this fd. Either the connection was
        // never tracked, or shutdown() has already closed and erased it. In
        // the latter case the integer fd may have been reused by the OS for an
        // unrelated descriptor, so writing here would corrupt that target and
        // bypass interleave protection. Fail closed rather than write blind.
        return false;
    }
    // Hold the per-fd write-mutex for the whole write. shutdown() acquires the
    // same mutex before ::close()ing the fd, so an in-flight writer either
    // completes first (shutdown waits) or, if shutdown ran first, finds the
    // mutex gone above and aborts — closing the use-after-close/fd-reuse TOCTOU.
    std::lock_guard<std::mutex> wlock(*mtx);
    // Re-verify the fd is still registered now that we hold the write-mutex:
    // shutdown() erases fd_write_mutexes_ under connections_mutex_ before
    // closing, so a vanished entry means the fd is being torn down.
    {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        if (fd_write_mutexes_.find(fd) == fd_write_mutexes_.end()) return false;
    }
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

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(target.port));
    if (::inet_pton(AF_INET, target.address.c_str(), &addr.sin_addr) != 1) {
        return -1;
    }

    // Peer's listener may not be ready yet; retry for a few seconds. Re-issuing
    // ::connect() on the SAME blocking fd after a failed connect has
    // undefined/portability-fragile behavior on Linux (the next call may return
    // EALREADY/EISCONN/EADDRNOTAVAIL rather than cleanly retrying), and the
    // SO_ERROR check can then read 0 on a socket that never finished
    // connecting. Create a fresh socket per attempt and only keep an fd whose
    // connect() actually returned 0 — matching RendezvousStore::connect_to_master
    // and NCCLProcessGroup::bootstrap_unique_id.
    int fd = -1;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    for (;;) {
        fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return -1;

        int one = 1;
        ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

        if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
            break;  // connected
        }

        ::close(fd);
        fd = -1;
        if (std::chrono::steady_clock::now() >= deadline) {
            return -1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
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
    {
        std::lock_guard<std::mutex> lock(threads_mutex_);
        worker_threads_.emplace_back([this, fd, peer_id]() { receive_loop(fd, peer_id); });
    }
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
    // Bind the RPC listener to loopback by default. The RPC protocol is
    // unauthenticated and handle_rpc_call() executes any registered function
    // with attacker-supplied tensors, so binding INADDR_ANY would expose
    // arbitrary remote invocation to every reachable host. init_rpc wires
    // worker addresses to 127.0.0.1, so loopback is sufficient for the
    // intra-host topology this transport targets; if self_.address names a
    // concrete local interface, honor it so multi-host deployments that have
    // arranged their own network isolation/auth can still bind that NIC.
    in_addr bind_addr{};
    if (self_.address.empty() ||
        ::inet_pton(AF_INET, self_.address.c_str(), &bind_addr) != 1) {
        bind_addr.s_addr = htonl(INADDR_LOOPBACK);
    }
    addr.sin_addr = bind_addr;
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

        // Bound the handshake read so a client that connects but sends nothing
        // (or only a partial frame) cannot stall the single accept thread and
        // block all further inbound connections. The per-connection
        // receive_loop runs on its own thread, so this timeout only guards the
        // handshake performed inline here.
        {
            struct timeval tv;
            tv.tv_sec = static_cast<long>(config_.timeout_ms / 1000);
            tv.tv_usec = static_cast<long>((config_.timeout_ms % 1000) * 1000);
            ::setsockopt(cfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        }

        // Read the handshake message to learn peer id.
        auto hello = deserialize_message(cfd);
        if (!hello || hello->src_worker < 0) {
            ::close(cfd);
            continue;
        }
        int32_t peer_id = hello->src_worker;

        // Clear the handshake recv timeout so the long-lived receive_loop can
        // block indefinitely between requests on an idle but healthy peer.
        {
            struct timeval tv0;
            tv0.tv_sec = 0;
            tv0.tv_usec = 0;
            ::setsockopt(cfd, SOL_SOCKET, SO_RCVTIMEO, &tv0, sizeof(tv0));
        }

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
        {
            std::lock_guard<std::mutex> lock(threads_mutex_);
            worker_threads_.emplace_back([this, cfd, peer_id]() { receive_loop(cfd, peer_id); });
        }
    }
}

void TcpRpcAgent::receive_loop(int fd, int32_t peer_id) {
    (void)peer_id;
    while (running_.load(std::memory_order_acquire)) {
        auto msg = deserialize_message(fd);
        if (!msg) break;
        // Requests with request_id == -1 are handshake-only; skip dispatch.
        if (msg->payload.request_id < 0 && msg->type == MessageType::HEARTBEAT) continue;
        // Pass the inbound fd so dispatch_message can reply on this same
        // connection rather than dialing a fresh outbound one.
        dispatch_message(std::move(*msg), fd);
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
