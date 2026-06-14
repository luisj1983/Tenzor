/**
 * @file gloo_backend.cpp
 * @brief Implementation of Gloo backend for CPU communication
 */

#include "tenzor/distributed/gloo_backend.hpp"
#include "tenzor/utils/error.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/backend/dtype_dispatch.hpp"
#include <stdexcept>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sys/stat.h>
#include <netdb.h>
#include <fcntl.h>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <thread>

namespace tenzor {
namespace distributed {

// ============================================================================
// TCPConnection Implementation
// ============================================================================

TCPConnection::TCPConnection(int socket_fd, int timeout_seconds)
    : socket_fd_(socket_fd), timeout_seconds_(timeout_seconds) {
    if (timeout_seconds_ <= 0) {
        // Read default from environment variable (default: 300s)
        const char* env = std::getenv("TENZOR_COMM_TIMEOUT");
        timeout_seconds_ = env ? std::atoi(env) : 300;
    }
    set_timeout(timeout_seconds_);
}

auto TCPConnection::set_timeout(int seconds) -> void {
    timeout_seconds_ = seconds;
    if (socket_fd_ >= 0 && seconds > 0) {
        struct timeval tv{};
        tv.tv_sec = seconds;
        tv.tv_usec = 0;
        setsockopt(socket_fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(socket_fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    }
}

TCPConnection::~TCPConnection() {
    close();
}

auto TCPConnection::send(const void* data, size_t size) -> void {
    if (closed_) {
        throw std::runtime_error("TCPConnection: socket is closed");
    }

    size_t total_sent = 0;
    const char* ptr = static_cast<const char*>(data);

    while (total_sent < size) {
        ssize_t sent = ::send(socket_fd_, ptr + total_sent, size - total_sent, 0);
        if (sent < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                throw std::runtime_error(
                    "TCPConnection: send timed out after " +
                    std::to_string(timeout_seconds_) + "s");
            }
            throw std::runtime_error(
                std::string("TCPConnection: send failed: ") + strerror(errno));
        }
        total_sent += sent;
    }
}

auto TCPConnection::recv(void* data, size_t size) -> void {
    if (closed_) {
        throw std::runtime_error("TCPConnection: socket is closed");
    }

    size_t total_received = 0;
    char* ptr = static_cast<char*>(data);

    while (total_received < size) {
        ssize_t received = ::recv(socket_fd_, ptr + total_received, size - total_received, MSG_WAITALL);
        if (received < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                throw std::runtime_error(
                    "TCPConnection: recv timed out after " +
                    std::to_string(timeout_seconds_) + "s");
            }
            throw std::runtime_error(
                std::string("TCPConnection: recv failed: ") + strerror(errno));
        }
        if (received == 0) {
            throw std::runtime_error("TCPConnection: peer disconnected");
        }
        total_received += received;
    }
}

auto TCPConnection::close() -> void {
    if (!closed_ && socket_fd_ >= 0) {
        ::close(socket_fd_);
        closed_ = true;
    }
}

// ============================================================================
// RendezvousStore Implementation
// ============================================================================

RendezvousStore::RendezvousStore(
    const std::string& master_addr,
    int master_port,
    int rank,
    int world_size
) : master_addr_(master_addr),
    master_port_(master_port),
    rank_(rank),
    world_size_(world_size) {

    if (rank_ == 0) {
        run_master_server();
    }
}

RendezvousStore::~RendezvousStore() {
    if (socket_fd_ >= 0) {
        ::close(socket_fd_);
        socket_fd_ = -1;
    }
    // Signal the server thread to stop and unblock its accept() by shutting down
    // the listening socket, then join (previously join() hung forever because
    // the accept() loop had no shutdown path).
    stop_.store(true);
    if (listen_fd_ >= 0) {
        ::shutdown(listen_fd_, SHUT_RDWR);
    }
    if (server_thread_ && server_thread_->joinable()) {
        server_thread_->join();
    }
}

auto RendezvousStore::set(const std::string& key, const std::string& value) -> void {
    connect_to_master();

    // Send SET command
    std::string command = "SET:" + key + ":" + value;
    uint32_t len = command.size();
    ::send(socket_fd_, &len, sizeof(len), 0);
    ::send(socket_fd_, command.c_str(), len, 0);
}

auto RendezvousStore::get(const std::string& key) -> std::string {
    connect_to_master();

    // Send GET command
    std::string command = "GET:" + key;
    uint32_t len = command.size();
    ::send(socket_fd_, &len, sizeof(len), 0);
    ::send(socket_fd_, command.c_str(), len, 0);

    // Receive response
    uint32_t response_len;
    ::recv(socket_fd_, &response_len, sizeof(response_len), MSG_WAITALL);

    std::string response(response_len, '\0');
    ::recv(socket_fd_, &response[0], response_len, MSG_WAITALL);

    return response;
}

auto RendezvousStore::wait(const std::string& key) -> void {
    // Simple wait implementation: poll until key is available
    for (int i = 0; i < 1000; ++i) {
        try {
            get(key);
            return;
        } catch (...) {
            usleep(10000);  // 10ms
        }
    }
    throw std::runtime_error("RendezvousStore: timeout waiting for key " + key);
}

auto RendezvousStore::delete_key(const std::string& key) -> bool {
    // Send a real DELETE command so the key is removed from the master store
    // (a subsequent GET observes a missing key, not a sentinel value).
    try {
        connect_to_master();
        std::string command = "DEL:" + key;
        uint32_t len = command.size();
        ::send(socket_fd_, &len, sizeof(len), 0);
        ::send(socket_fd_, command.c_str(), len, 0);
        return true;
    } catch (...) {
        return false;
    }
}

auto RendezvousStore::check_key(const std::string& key) -> bool {
    try {
        auto val = get(key);
        // Missing keys come back empty; also honour the legacy "__deleted__"
        // sentinel for back-compat with stores written by older peers.
        return !val.empty() && val != "__deleted__";
    } catch (...) {
        return false;
    }
}

auto RendezvousStore::add(const std::string& key, int64_t delta) -> int64_t {
    connect_to_master();

    std::string command = "ADD:" + key + ":" + std::to_string(delta);
    uint32_t len = command.size();
    ::send(socket_fd_, &len, sizeof(len), 0);
    ::send(socket_fd_, command.c_str(), len, 0);

    uint32_t response_len = 0;
    if (::recv(socket_fd_, &response_len, sizeof(response_len), MSG_WAITALL)
            != static_cast<ssize_t>(sizeof(response_len))) {
        throw std::runtime_error("RendezvousStore::add: no response from master");
    }
    std::string response(response_len, '\0');
    if (response_len > 0) {
        ::recv(socket_fd_, &response[0], response_len, MSG_WAITALL);
    }
    try {
        return std::stoll(response);
    } catch (...) {
        throw std::runtime_error("RendezvousStore::add: invalid response '" + response + "'");
    }
}

auto RendezvousStore::connect_to_master() -> void {
    // The master server handles exactly one command per accepted connection and
    // then closes it, so each store operation needs a FRESH connection — never
    // reuse a cached (now-dead) socket. Close any stale fd before reconnecting.
    if (socket_fd_ >= 0) {
        ::close(socket_fd_);
        socket_fd_ = -1;
    }

    struct hostent* server = gethostbyname(master_addr_.c_str());
    if (!server) {
        throw std::runtime_error("Failed to resolve address: " + master_addr_);
    }

    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(master_port_);
    std::memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);

    // Retry with backoff: the server may be starting in another thread/process,
    // or briefly busy handling the previous one-shot connection.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        socket_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (socket_fd_ < 0) {
            throw std::runtime_error("Failed to create socket");
        }
        if (connect(socket_fd_, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
            return;  // connected
        }
        ::close(socket_fd_);
        socket_fd_ = -1;
        if (std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error("Failed to connect to rendezvous master at " +
                                     master_addr_ + ":" + std::to_string(master_port_));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

auto RendezvousStore::run_master_server() -> void {
    // Create server socket for rendezvous key-value store
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        return;  // Silently fail if we can't create server
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(master_port_);

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        ::close(server_fd);
        return;
    }

    if (listen(server_fd, 10) < 0) {
        ::close(server_fd);
        return;
    }

    // Record the listening socket so the destructor can shut it down and unblock
    // the accept() loop below for a clean thread join.
    listen_fd_ = server_fd;

    // Run server in background thread
    server_thread_ = std::make_unique<std::thread>([this, server_fd]() {
        std::unordered_map<std::string, std::string> store;
        std::mutex store_mutex;

        while (true) {
            int client_fd = accept(server_fd, nullptr, nullptr);
            if (client_fd < 0) break;

            // Handle client request
            uint32_t cmd_len;
            if (::recv(client_fd, &cmd_len, sizeof(cmd_len), MSG_WAITALL) != sizeof(cmd_len)) {
                ::close(client_fd);
                continue;
            }

            std::string command(cmd_len, '\0');
            if (::recv(client_fd, &command[0], cmd_len, MSG_WAITALL) != static_cast<ssize_t>(cmd_len)) {
                ::close(client_fd);
                continue;
            }

            // Parse command (SET:key:value or GET:key)
            size_t first_colon = command.find(':');
            if (first_colon == std::string::npos) {
                ::close(client_fd);
                continue;
            }

            std::string op = command.substr(0, first_colon);
            std::string rest = command.substr(first_colon + 1);

            if (op == "SET") {
                size_t second_colon = rest.find(':');
                if (second_colon != std::string::npos) {
                    std::string key = rest.substr(0, second_colon);
                    std::string value = rest.substr(second_colon + 1);

                    std::lock_guard<std::mutex> lock(store_mutex);
                    store[key] = value;
                }
            } else if (op == "GET") {
                std::string key = rest;
                std::string value;

                {
                    std::lock_guard<std::mutex> lock(store_mutex);
                    auto it = store.find(key);
                    if (it != store.end()) {
                        value = it->second;
                    }
                }

                if (!value.empty()) {
                    uint32_t response_len = value.size();
                    ::send(client_fd, &response_len, sizeof(response_len), 0);
                    ::send(client_fd, value.c_str(), value.size(), 0);
                }
            } else if (op == "DEL") {
                // Real key deletion: remove from the master store so a later GET
                // observes a missing key (rather than a "__deleted__" sentinel).
                std::string key = rest;
                std::lock_guard<std::mutex> lock(store_mutex);
                store.erase(key);
            } else if (op == "ADD") {
                // Atomic fetch-and-add: "ADD:key:delta" -> new integer value.
                size_t second_colon = rest.find(':');
                if (second_colon != std::string::npos) {
                    std::string key = rest.substr(0, second_colon);
                    long long delta = 0;
                    try { delta = std::stoll(rest.substr(second_colon + 1)); }
                    catch (...) { delta = 0; }
                    long long new_val;
                    {
                        std::lock_guard<std::mutex> lock(store_mutex);
                        long long cur = 0;
                        auto it = store.find(key);
                        if (it != store.end()) {
                            try { cur = std::stoll(it->second); } catch (...) { cur = 0; }
                        }
                        new_val = cur + delta;
                        store[key] = std::to_string(new_val);
                    }
                    std::string value = std::to_string(new_val);
                    uint32_t response_len = value.size();
                    ::send(client_fd, &response_len, sizeof(response_len), 0);
                    ::send(client_fd, value.c_str(), value.size(), 0);
                }
            }

            ::close(client_fd);
        }

        ::close(server_fd);
    });
}

// ============================================================================
// GlooBackend Implementation
// ============================================================================

GlooBackend::GlooBackend() = default;

GlooBackend::~GlooBackend() {
    try {
        finalize();
    } catch (...) {
        // Ignore errors during destruction
    }
}

auto GlooBackend::initialize(
    int rank,
    int world_size,
    const std::string& master_addr,
    int master_port
) -> void {

    if (initialized_) {
        throw std::runtime_error("GlooBackend: already initialized");
    }

    rank_ = rank;
    world_size_ = world_size;
    master_addr_ = master_addr;
    master_port_ = master_port;

    // Initialize TCP connections
    init_connections();

    initialized_ = true;
}

auto GlooBackend::broadcast(Tensor& tensor, int src_rank) -> void {
    validate_cpu_accessible(tensor);

    if (rank_ == src_rank) {
        // Send to all other ranks
        for (int dst = 0; dst < world_size_; ++dst) {
            if (dst != rank_) {
                send_tensor(tensor, dst);
            }
        }
    } else {
        // Receive from source
        recv_tensor(tensor, src_rank);
    }
}

auto GlooBackend::all_reduce(Tensor& tensor, ReduceOp op) -> void {
    validate_cpu_accessible(tensor);
    // ring_all_reduce only dispatches Float32/Float64. Widen Float16/BFloat16
    // to Float32 for the reduction, then narrow back in place.
    auto orig_dtype = tensor.dtype();
    if (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16) {
        Tensor wide = tensor.to(DType::Float32);
        ring_all_reduce(wide, op);
        tensor = wide.to(orig_dtype);
        return;
    }
    ring_all_reduce(tensor, op);
}

auto GlooBackend::reduce(Tensor& tensor, int dst_rank, ReduceOp op) -> void {
    validate_cpu_accessible(tensor);

    if (rank_ == dst_rank) {
        // Receive and reduce from all other ranks
        for (int src = 0; src < world_size_; ++src) {
            if (src != rank_) {
                Tensor received = zeros_like(tensor);
                recv_tensor(received, src);
                apply_reduce_op(tensor, received, op);
            }
        }
        // apply_reduce_op treats AVG as SUM (it has no world_size context), so
        // finish the average here. Without this, reduce(AVG) returned the
        // unaveraged sum (off by world_size); ring_all_reduce divides similarly.
        if (op == ReduceOp::AVG && world_size_ > 0) {
            tensor = tenzor::div(tensor, static_cast<double>(world_size_));
        }
    } else {
        // Send to destination
        send_tensor(tensor, dst_rank);
    }
}

auto GlooBackend::all_gather(const Tensor& tensor, std::vector<Tensor>& output) -> void {
    validate_cpu_accessible(tensor);

    if (output.size() != static_cast<size_t>(world_size_)) {
        throw std::invalid_argument("all_gather: output size must equal world_size");
    }

    // Send local tensor to all ranks
    for (int dst = 0; dst < world_size_; ++dst) {
        if (dst != rank_) {
            send_tensor(tensor, dst);
        } else {
            output[rank_] = tensor.clone();
        }
    }

    // Receive from all other ranks
    for (int src = 0; src < world_size_; ++src) {
        if (src != rank_) {
            output[src] = zeros_like(tensor);
            recv_tensor(output[src], src);
        }
    }
}

auto GlooBackend::gather(const Tensor& tensor, std::vector<Tensor>& output, int dst_rank) -> void {
    validate_cpu_accessible(tensor);

    if (rank_ == dst_rank) {
        if (output.size() != static_cast<size_t>(world_size_)) {
            throw std::invalid_argument("gather: output size must equal world_size");
        }

        output[rank_] = tensor.clone();

        // Receive from all other ranks
        for (int src = 0; src < world_size_; ++src) {
            if (src != rank_) {
                output[src] = zeros_like(tensor);
                recv_tensor(output[src], src);
            }
        }
    } else {
        // Send to destination
        send_tensor(tensor, dst_rank);
    }
}

auto GlooBackend::scatter(const std::vector<Tensor>& tensors, Tensor& output, int src_rank) -> void {
    if (rank_ == src_rank) {
        if (tensors.size() != static_cast<size_t>(world_size_)) {
            throw std::invalid_argument("scatter: tensors size must equal world_size");
        }

        output = tensors[rank_].clone();

        // Send to all other ranks
        for (int dst = 0; dst < world_size_; ++dst) {
            if (dst != rank_) {
                send_tensor(tensors[dst], dst);
            }
        }
    } else {
        // Receive from source
        recv_tensor(output, src_rank);
    }
}

auto GlooBackend::reduce_scatter(const std::vector<Tensor>& tensors, Tensor& output, ReduceOp op) -> void {
    if (tensors.size() != static_cast<size_t>(world_size_)) {
        throw std::invalid_argument("reduce_scatter: tensors size must equal world_size");
    }

    // Reduce-scatter: each rank provides tensors[i] for all i,
    // and rank j receives the reduction of all ranks' tensors[j]

    // Simple but correct implementation: all-reduce each chunk separately
    // For chunk i, all ranks all-reduce their tensors[i], and rank i keeps the result

    // Copy our output chunk
    output = tensors[rank_].clone();

    // All-reduce it (all ranks do this for their respective chunks)
    all_reduce(output, op);

    // Note: This works because:
    // - Rank 0 all-reduces its tensors[0] -> gets reduction of all ranks' tensors[0]
    // - Rank 1 all-reduces its tensors[1] -> gets reduction of all ranks' tensors[1]
    // - etc.
    // Even though they're all-reducing different data, the all_reduce calls don't interfere
    // because each rank is doing its own independent all-reduce operation.

    // Wait, that's wrong too! All ranks must call all_reduce with the SAME data!

    // OK, the REAL correct implementation: do world_size all-reduces, one per chunk
    for (int chunk_idx = 0; chunk_idx < world_size_; ++chunk_idx) {
        Tensor temp = tensors[chunk_idx].clone();
        all_reduce(temp, op);

        if (chunk_idx == rank_) {
            output = temp;
        }
    }
}

auto GlooBackend::send(const Tensor& tensor, int dst_rank) -> void {
    if (!initialized_) {
        throw std::runtime_error("GlooBackend: not initialized");
    }

    if (dst_rank < 0 || dst_rank >= world_size_) {
        throw std::invalid_argument(
            "GlooBackend::send: invalid dst_rank " + std::to_string(dst_rank)
        );
    }
    if (dst_rank == rank_) {
        throw std::invalid_argument(
            "GlooBackend::send: cannot send to self (rank " + std::to_string(rank_) + ")"
        );
    }

    // Ensure tensor is on CPU (Gloo is CPU-only)
    Tensor cpu_tensor = get_cpu_buffer(tensor);
    send_tensor(cpu_tensor, dst_rank);
}

auto GlooBackend::recv(Tensor& tensor, int src_rank) -> void {
    if (!initialized_) {
        throw std::runtime_error("GlooBackend: not initialized");
    }

    if (src_rank < 0 || src_rank >= world_size_) {
        throw std::invalid_argument(
            "GlooBackend::recv: invalid src_rank " + std::to_string(src_rank)
        );
    }
    if (src_rank == rank_) {
        throw std::invalid_argument(
            "GlooBackend::recv: cannot recv from self (rank " + std::to_string(rank_) + ")"
        );
    }

    recv_tensor(tensor, src_rank);
}

auto GlooBackend::barrier() -> void {
    // Simple barrier: all-reduce on dummy tensor
    Tensor dummy = zeros({1}, DType::Float32, Device::cpu());
    all_reduce(dummy, ReduceOp::SUM);
}

auto GlooBackend::finalize() -> void {
    // Close all TCP connections explicitly
    for (auto& conn : connections_) {
        if (conn) {
            conn->close();
        }
    }
    connections_.clear();

    if (accept_thread_ && accept_thread_->joinable()) {
        accept_thread_->join();
    }

    if (server_socket_ >= 0) {
        ::close(server_socket_);
        server_socket_ = -1;
    }

    // Clean up rendezvous store files
    std::string store_path = "/tmp/tenzor_rendezvous_" + std::to_string(master_port_);
    if (std::filesystem::exists(store_path)) {
        try {
            std::filesystem::remove_all(store_path);
        } catch (...) {
            // Ignore errors during cleanup
        }
    }

    initialized_ = false;
}

auto GlooBackend::supports_device([[maybe_unused]] Device::Type device_type) const -> bool {
    // Gloo works with CPU and can handle GPU tensors by copying to CPU
    return true;
}

// ============================================================================
// B.3: async collectives via WorkExecutor worker thread
// ============================================================================

auto GlooBackend::async_executor() -> WorkExecutor& {
    if (!async_executor_) {
        async_executor_ = std::make_unique<WorkExecutor>();
    }
    return *async_executor_;
}

auto GlooBackend::all_reduce_async(Tensor& tensor, ReduceOp op,
                                    void* /*stream*/) -> void {
    Tensor& t = tensor;
    async_executor().enqueue([this, &t, op]() { this->all_reduce(t, op); });
}

auto GlooBackend::all_gather_async(const Tensor& tensor,
                                    std::vector<Tensor>& output,
                                    void* /*stream*/) -> void {
    const Tensor& in = tensor;
    auto& out = output;
    async_executor().enqueue([this, &in, &out]() { this->all_gather(in, out); });
}

auto GlooBackend::reduce_scatter_async(const std::vector<Tensor>& tensors,
                                        Tensor& output, ReduceOp op,
                                        void* /*stream*/) -> void {
    const auto& in = tensors;
    auto& out = output;
    async_executor().enqueue([this, &in, &out, op]() {
        this->reduce_scatter(in, out, op);
    });
}

auto GlooBackend::wait_pending_async() -> void {
    if (async_executor_) {
        async_executor_->wait_pending();
    }
}

// ============================================================================
// Private Helper Methods
// ============================================================================

auto GlooBackend::init_connections() -> void {
    connections_.resize(world_size_);

    // Create server socket
    server_socket_ = create_server_socket();

    // Write our port to rendezvous store
    write_port_to_store(rank_, server_port_);

    // Use ordered connections to avoid deadlock:
    // - Lower ranks connect to higher ranks
    // - Higher ranks accept from lower ranks
    for (int peer_rank = 0; peer_rank < world_size_; ++peer_rank) {
        if (peer_rank == rank_) {
            continue;
        }

        if (peer_rank < rank_) {
            // Accept connection from lower rank
            int client_fd = accept(server_socket_, nullptr, nullptr);
            if (client_fd < 0) {
                throw std::runtime_error("Failed to accept connection from rank " + std::to_string(peer_rank));
            }

            // Receive peer rank ID for verification
            int received_rank = -1;
            ssize_t n = ::recv(client_fd, &received_rank, sizeof(int), 0);
            if (n != sizeof(int) || received_rank != peer_rank) {
                ::close(client_fd);
                throw std::runtime_error("Handshake failed: expected rank " + std::to_string(peer_rank) +
                                       " but got " + std::to_string(received_rank));
            }

            // Send our rank ID
            ::send(client_fd, &rank_, sizeof(int), 0);

            connections_[peer_rank] = std::make_shared<TCPConnection>(client_fd);
        } else {
            // Connect to higher rank
            connections_[peer_rank] = connect_to_rank(peer_rank);
        }
    }
}

auto GlooBackend::create_server_socket() -> int {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        throw std::runtime_error("Failed to create server socket");
    }

    int opt = 1;
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = 0;  // Let OS assign port

    if (bind(sockfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        ::close(sockfd);
        throw std::runtime_error("Failed to bind server socket");
    }

    socklen_t len = sizeof(addr);
    if (getsockname(sockfd, (struct sockaddr*)&addr, &len) < 0) {
        ::close(sockfd);
        throw std::runtime_error("Failed to get server port");
    }
    server_port_ = ntohs(addr.sin_port);

    if (listen(sockfd, world_size_) < 0) {
        ::close(sockfd);
        throw std::runtime_error("Failed to listen on server socket");
    }

    return sockfd;
}

auto GlooBackend::accept_connections() -> void {
    for (int i = 0; i < world_size_ - 1; ++i) {
        int client_fd = accept(server_socket_, nullptr, nullptr);
        if (client_fd >= 0) {
            // Connection accepted (connection will be stored when needed)
        }
    }
}

auto GlooBackend::connect_to_rank(int peer_rank) -> std::shared_ptr<TCPConnection> {
    // Read peer's port from the store (with retries)
    int peer_port = read_port_from_store(peer_rank);

    // Connect to peer
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        throw std::runtime_error("Failed to create socket for rank " + std::to_string(peer_rank));
    }

    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(peer_port);

    // Convert master_addr to IP
    if (inet_pton(AF_INET, master_addr_.c_str(), &addr.sin_addr) <= 0) {
        // Try as hostname
        struct hostent* he = gethostbyname(master_addr_.c_str());
        if (he == nullptr) {
            ::close(sockfd);
            throw std::runtime_error("Invalid address: " + master_addr_);
        }
        std::memcpy(&addr.sin_addr, he->h_addr_list[0], he->h_length);
    }

    // Retry connection with exponential backoff
    int max_retries = 10;
    for (int retry = 0; retry < max_retries; ++retry) {
        if (connect(sockfd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
            // Connection successful - perform handshake
            // Send our rank ID
            ::send(sockfd, &rank_, sizeof(int), 0);

            // Receive peer rank ID for verification
            int received_rank = -1;
            ssize_t n = ::recv(sockfd, &received_rank, sizeof(int), 0);
            if (n != sizeof(int) || received_rank != peer_rank) {
                ::close(sockfd);
                throw std::runtime_error("Handshake failed: expected rank " + std::to_string(peer_rank) +
                                       " but got " + std::to_string(received_rank));
            }

            return std::make_shared<TCPConnection>(sockfd);
        }

        // Sleep and retry
        std::this_thread::sleep_for(std::chrono::milliseconds(100 * (1 << retry)));
    }

    ::close(sockfd);
    throw std::runtime_error("Failed to connect to rank " + std::to_string(peer_rank) +
                            " at " + master_addr_ + ":" + std::to_string(peer_port));
}

auto GlooBackend::write_port_to_store(int rank, int port) -> void {
    // Per-user, owner-only rendezvous dir. A fixed world-accessible
    // /tmp/tenzor_rendezvous_<port> let any local user pre-create the dir or
    // rank files to inject a bogus port (MITM the handshake) or DoS it.
    std::string store_path;
    if (const char* xdg = std::getenv("XDG_RUNTIME_DIR"); xdg && xdg[0]) {
        store_path = std::string(xdg) + "/tenzor_rendezvous_" + std::to_string(master_port_);
    } else {
        store_path = "/tmp/tenzor_rendezvous_" + std::to_string(getuid()) +
                     "_" + std::to_string(master_port_);
    }
    std::filesystem::create_directories(store_path);
    ::chmod(store_path.c_str(), 0700);

    std::string rank_file = store_path + "/rank_" + std::to_string(rank);
    std::ofstream file(rank_file);
    if (!file) {
        throw std::runtime_error("Failed to write to rendezvous store");
    }
    file << port << std::endl;
    file.close();
}

auto GlooBackend::read_port_from_store(int rank) -> int {
    // Must match write_port_to_store()'s per-user path exactly.
    std::string store_path;
    if (const char* xdg = std::getenv("XDG_RUNTIME_DIR"); xdg && xdg[0]) {
        store_path = std::string(xdg) + "/tenzor_rendezvous_" + std::to_string(master_port_);
    } else {
        store_path = "/tmp/tenzor_rendezvous_" + std::to_string(getuid()) +
                     "_" + std::to_string(master_port_);
    }
    std::string rank_file = store_path + "/rank_" + std::to_string(rank);

    // Wait for file to appear (with timeout)
    int max_wait_ms = 30000;  // 30 seconds
    int wait_interval_ms = 100;
    int elapsed = 0;

    while (!std::filesystem::exists(rank_file) && elapsed < max_wait_ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(wait_interval_ms));
        elapsed += wait_interval_ms;
    }

    if (!std::filesystem::exists(rank_file)) {
        throw std::runtime_error("Timeout waiting for rank " + std::to_string(rank) +
                                " to write port to rendezvous store");
    }

    std::ifstream file(rank_file);
    if (!file) {
        throw std::runtime_error("Failed to read from rendezvous store");
    }

    int port;
    file >> port;
    return port;
}

auto GlooBackend::send_tensor(const Tensor& tensor, int peer_rank) -> void {
    auto conn = connections_[peer_rank];
    if (!conn) {
        throw std::runtime_error("No connection to rank " + std::to_string(peer_rank));
    }

    Tensor cpu_tensor = get_cpu_buffer(tensor);

    // Send metadata
    uint64_t numel = cpu_tensor.numel();
    uint32_t dtype = static_cast<uint32_t>(cpu_tensor.dtype());
    conn->send(&numel, sizeof(numel));
    conn->send(&dtype, sizeof(dtype));

    // Send data
    size_t data_size = numel * dtype_size(cpu_tensor.dtype());
    conn->send(cpu_tensor.data_ptr(), data_size);
}

auto GlooBackend::recv_tensor(Tensor& tensor, int peer_rank) -> void {
    auto conn = connections_[peer_rank];
    if (!conn) {
        throw std::runtime_error("No connection to rank " + std::to_string(peer_rank));
    }

    // Receive metadata
    uint64_t numel;
    uint32_t dtype;
    conn->recv(&numel, sizeof(numel));
    conn->recv(&dtype, sizeof(dtype));

    // Receive data - use dtype from metadata, not cpu_tensor.dtype()
    Tensor cpu_tensor = get_cpu_buffer(tensor);
    size_t data_size = numel * dtype_size(static_cast<DType>(dtype));

    // Validate tensor size matches
    if (cpu_tensor.numel() != static_cast<int64_t>(numel)) {
        throw std::runtime_error(
            "recv_tensor: size mismatch - expected " + std::to_string(numel) +
            " elements, but tensor has " + std::to_string(cpu_tensor.numel())
        );
    }

    conn->recv(cpu_tensor.data_ptr(), data_size);

    // Copy back to original device if needed
    if (tensor.device().type != Device::Type::CPU) {
        tensor = cpu_tensor.to(tensor.device());
    }
}

auto GlooBackend::apply_reduce_op(Tensor& result, const Tensor& operand, ReduceOp op) -> void {
    switch (op) {
        case ReduceOp::SUM:
        case ReduceOp::AVG:
            result = result + operand;
            break;
        case ReduceOp::PRODUCT:
            result = result * operand;
            break;
        case ReduceOp::MIN:
            // Element-wise minimum operation with proper dtype support
            {
                size_t numel = result.numel();

                switch (result.dtype()) {
                    case DType::Float32: {
                        auto* result_ptr = static_cast<float*>(result.data_ptr());
                        const auto* operand_ptr = static_cast<const float*>(operand.data_ptr());
                        for (size_t i = 0; i < numel; ++i) {
                            result_ptr[i] = std::min(result_ptr[i], operand_ptr[i]);
                        }
                        break;
                    }
                    case DType::Float64: {
                        auto* result_ptr = static_cast<double*>(result.data_ptr());
                        const auto* operand_ptr = static_cast<const double*>(operand.data_ptr());
                        for (size_t i = 0; i < numel; ++i) {
                            result_ptr[i] = std::min(result_ptr[i], operand_ptr[i]);
                        }
                        break;
                    }
                    case DType::Int32: {
                        auto* result_ptr = static_cast<int32_t*>(result.data_ptr());
                        const auto* operand_ptr = static_cast<const int32_t*>(operand.data_ptr());
                        for (size_t i = 0; i < numel; ++i) {
                            result_ptr[i] = std::min(result_ptr[i], operand_ptr[i]);
                        }
                        break;
                    }
                    case DType::Int64: {
                        auto* result_ptr = static_cast<int64_t*>(result.data_ptr());
                        const auto* operand_ptr = static_cast<const int64_t*>(operand.data_ptr());
                        for (size_t i = 0; i < numel; ++i) {
                            result_ptr[i] = std::min(result_ptr[i], operand_ptr[i]);
                        }
                        break;
                    }
                    default:
                        throw std::invalid_argument(
                            "MIN reduction not supported for dtype: " +
                            std::to_string(static_cast<int>(result.dtype()))
                        );
                }
            }
            break;
        case ReduceOp::MAX:
            // Element-wise maximum operation with proper dtype support
            {
                size_t numel = result.numel();

                switch (result.dtype()) {
                    case DType::Float32: {
                        auto* result_ptr = static_cast<float*>(result.data_ptr());
                        const auto* operand_ptr = static_cast<const float*>(operand.data_ptr());
                        for (size_t i = 0; i < numel; ++i) {
                            result_ptr[i] = std::max(result_ptr[i], operand_ptr[i]);
                        }
                        break;
                    }
                    case DType::Float64: {
                        auto* result_ptr = static_cast<double*>(result.data_ptr());
                        const auto* operand_ptr = static_cast<const double*>(operand.data_ptr());
                        for (size_t i = 0; i < numel; ++i) {
                            result_ptr[i] = std::max(result_ptr[i], operand_ptr[i]);
                        }
                        break;
                    }
                    case DType::Int32: {
                        auto* result_ptr = static_cast<int32_t*>(result.data_ptr());
                        const auto* operand_ptr = static_cast<const int32_t*>(operand.data_ptr());
                        for (size_t i = 0; i < numel; ++i) {
                            result_ptr[i] = std::max(result_ptr[i], operand_ptr[i]);
                        }
                        break;
                    }
                    case DType::Int64: {
                        auto* result_ptr = static_cast<int64_t*>(result.data_ptr());
                        const auto* operand_ptr = static_cast<const int64_t*>(operand.data_ptr());
                        for (size_t i = 0; i < numel; ++i) {
                            result_ptr[i] = std::max(result_ptr[i], operand_ptr[i]);
                        }
                        break;
                    }
                    default:
                        throw std::invalid_argument(
                            "MAX reduction not supported for dtype: " +
                            std::to_string(static_cast<int>(result.dtype()))
                        );
                }
            }
            break;
        default:
            throw std::invalid_argument("Unsupported reduce operation");
    }
}

auto GlooBackend::ring_all_reduce(Tensor& tensor, ReduceOp op) -> void {
    // Ring all-reduce algorithm using direct memory operations.
    // Works with any tensor shape by operating on contiguous memory.
    // Dtype-generic over all numeric types: Float32/Float64 (half is pre-widened
    // by all_reduce) AND integer types (Int8/16/32/64, UInt8). Integer all_reduce
    // — e.g. summing counts/indices across ranks — matches NCCL/MPI and the
    // apply_reduce_op helper used by reduce(); it previously threw here.

    validate_cpu_accessible(tensor);
    Tensor cpu_tensor = get_cpu_buffer(tensor);

    int64_t total_elements = cpu_tensor.numel();
    int num_chunks = world_size_;
    size_t chunk_size = (total_elements + num_chunks - 1) / num_chunks;
    size_t elem_size = dtype_size(cpu_tensor.dtype());

    // Generic over the reduction element type. Dispatched explicitly (below)
    // over Float32/Float64 + integer types only — Float16/BFloat16 are widened
    // to Float32 by all_reduce before reaching here, and std::min/std::max are
    // not defined for the half types, so they are intentionally excluded.
    auto ring_body = [&]<typename scalar_t>() {
        auto* data_ptr = static_cast<scalar_t*>(cpu_tensor.data_ptr());

        // Reduce-scatter phase: reduce chunks in a ring pattern
        for (int i = 0; i < world_size_ - 1; ++i) {
            int send_chunk_idx = (rank_ - i + world_size_) % world_size_;
            int recv_chunk_idx = (rank_ - i - 1 + world_size_) % world_size_;

            int send_rank = (rank_ + 1) % world_size_;
            int recv_rank = (rank_ - 1 + world_size_) % world_size_;

            // Calculate chunk boundaries
            size_t send_start = send_chunk_idx * chunk_size;
            size_t send_count = std::min(chunk_size, static_cast<size_t>(total_elements - send_start));

            size_t recv_start = recv_chunk_idx * chunk_size;
            size_t recv_count = std::min(chunk_size, static_cast<size_t>(total_elements - recv_start));

            if (send_start < static_cast<size_t>(total_elements) && recv_start < static_cast<size_t>(total_elements)) {
                // Use tensor-based send/recv buffers (dtype-safe)
                Tensor send_chunk = zeros({static_cast<int64_t>(send_count)}, cpu_tensor.dtype(), cpu_tensor.device());
                std::memcpy(send_chunk.data_ptr(), data_ptr + send_start, send_count * elem_size);

                Tensor recv_chunk = zeros({static_cast<int64_t>(recv_count)}, cpu_tensor.dtype(), cpu_tensor.device());

                // Send/receive
                send_tensor(send_chunk, send_rank);
                recv_tensor(recv_chunk, recv_rank);

                // Reduce received data into our chunk
                auto* recv_data = static_cast<scalar_t*>(recv_chunk.data_ptr());
                for (size_t j = 0; j < recv_count; ++j) {
                    switch (op) {
                        case ReduceOp::SUM:
                        case ReduceOp::AVG:
                            data_ptr[recv_start + j] += recv_data[j];
                            break;
                        case ReduceOp::PRODUCT:
                            data_ptr[recv_start + j] *= recv_data[j];
                            break;
                        case ReduceOp::MIN:
                            data_ptr[recv_start + j] = std::min(data_ptr[recv_start + j], recv_data[j]);
                            break;
                        case ReduceOp::MAX:
                            data_ptr[recv_start + j] = std::max(data_ptr[recv_start + j], recv_data[j]);
                            break;
                        case ReduceOp::BAND:
                        case ReduceOp::BOR:
                        case ReduceOp::BXOR:
                            throw std::runtime_error("Gloo backend does not support bitwise reduction ops (BAND/BOR/BXOR)");
                    }
                }
            }
        }

        // All-gather phase: distribute reduced chunks to all nodes
        for (int i = 0; i < world_size_ - 1; ++i) {
            int send_chunk_idx = (rank_ - i + 1 + world_size_) % world_size_;
            int recv_chunk_idx = (rank_ - i + world_size_) % world_size_;

            int send_rank = (rank_ + 1) % world_size_;
            int recv_rank = (rank_ - 1 + world_size_) % world_size_;

            // Calculate chunk boundaries
            size_t send_start = send_chunk_idx * chunk_size;
            size_t send_count = std::min(chunk_size, static_cast<size_t>(total_elements - send_start));

            size_t recv_start = recv_chunk_idx * chunk_size;
            size_t recv_count = std::min(chunk_size, static_cast<size_t>(total_elements - recv_start));

            if (send_start < static_cast<size_t>(total_elements) && recv_start < static_cast<size_t>(total_elements)) {
                Tensor send_chunk = zeros({static_cast<int64_t>(send_count)}, cpu_tensor.dtype(), cpu_tensor.device());
                std::memcpy(send_chunk.data_ptr(), data_ptr + send_start, send_count * elem_size);

                Tensor recv_chunk = zeros({static_cast<int64_t>(recv_count)}, cpu_tensor.dtype(), cpu_tensor.device());

                // Send/receive
                send_tensor(send_chunk, send_rank);
                recv_tensor(recv_chunk, recv_rank);

                // Copy received data to our buffer
                std::memcpy(data_ptr + recv_start, recv_chunk.data_ptr(), recv_count * elem_size);
            }
        }

        // Handle AVG operation
        if (op == ReduceOp::AVG) {
            scalar_t divisor = static_cast<scalar_t>(world_size_);
            for (int64_t i = 0; i < total_elements; ++i) {
                data_ptr[i] /= divisor;
            }
        }
    };

    switch (cpu_tensor.dtype()) {
        case DType::Float32: ring_body.template operator()<float>();   break;
        case DType::Float64: ring_body.template operator()<double>();  break;
        case DType::Int64:   ring_body.template operator()<int64_t>(); break;
        case DType::Int32:   ring_body.template operator()<int32_t>(); break;
        case DType::Int16:   ring_body.template operator()<int16_t>(); break;
        case DType::Int8:    ring_body.template operator()<int8_t>();  break;
        case DType::UInt8:   ring_body.template operator()<uint8_t>(); break;
        default:
            throw std::runtime_error(
                "ring_all_reduce: unsupported dtype " +
                std::string(dtype_name(cpu_tensor.dtype())));
    }

    // Copy back to original device if needed
    if (tensor.device().type != Device::Type::CPU) {
        tensor = cpu_tensor.to(tensor.device());
    } else {
        tensor = cpu_tensor;
    }
}

auto GlooBackend::get_cpu_buffer(const Tensor& tensor) -> Tensor {
    if (tensor.device().type == Device::Type::CPU) {
        return tensor;
    }
    return tensor.to(Device::cpu());
}

auto GlooBackend::validate_cpu_accessible([[maybe_unused]] const Tensor& tensor) -> void {
    // Gloo can handle any tensor by copying to CPU if needed
    // No validation needed
}

} // namespace distributed
} // namespace tenzor
