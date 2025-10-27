/**
 * @file gloo_backend.cpp
 * @brief Implementation of Gloo backend for CPU communication
 */

#include "tenzor/distributed/gloo_backend.hpp"
#include "tenzor/utils/error.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#include <fcntl.h>

namespace tenzor {
namespace distributed {

// ============================================================================
// TCPConnection Implementation
// ============================================================================

TCPConnection::TCPConnection(int socket_fd)
    : socket_fd_(socket_fd) {
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
            throw std::runtime_error("TCPConnection: send failed");
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
        if (received <= 0) {
            throw std::runtime_error("TCPConnection: recv failed");
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

auto RendezvousStore::connect_to_master() -> void {
    if (socket_fd_ >= 0) {
        return;  // Already connected
    }

    socket_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd_ < 0) {
        throw std::runtime_error("Failed to create socket");
    }

    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(master_port_);

    struct hostent* server = gethostbyname(master_addr_.c_str());
    if (!server) {
        ::close(socket_fd_);
        socket_fd_ = -1;
        throw std::runtime_error("Failed to resolve address: " + master_addr_);
    }

    std::memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);

    if (connect(socket_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        ::close(socket_fd_);
        socket_fd_ = -1;
        throw std::runtime_error("Failed to connect to master");
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

    // Each rank reduces its assigned chunk from all ranks
    output = tensors[rank_].clone();

    for (int src = 0; src < world_size_; ++src) {
        if (src != rank_) {
            Tensor received = zeros_like(output);
            recv_tensor(received, src);
            apply_reduce_op(output, received, op);
        }
    }
}

auto GlooBackend::barrier() -> void {
    // Simple barrier: all-reduce on dummy tensor
    Tensor dummy = zeros({1}, DType::Float32, Device::cpu());
    all_reduce(dummy, ReduceOp::SUM);
}

auto GlooBackend::finalize() -> void {
    connections_.clear();

    if (accept_thread_ && accept_thread_->joinable()) {
        accept_thread_->join();
    }

    if (server_socket_ >= 0) {
        ::close(server_socket_);
        server_socket_ = -1;
    }

    initialized_ = false;
}

auto GlooBackend::supports_device(Device::Type device_type) const -> bool {
    // Gloo works with CPU and can handle GPU tensors by copying to CPU
    return true;
}

// ============================================================================
// Private Helper Methods
// ============================================================================

auto GlooBackend::init_connections() -> void {
    connections_.resize(world_size_);

    // Create server socket
    server_socket_ = create_server_socket();

    // Start accept thread
    accept_thread_ = std::make_unique<std::thread>([this]() {
        accept_connections();
    });

    // Connect to all ranks
    for (int peer_rank = 0; peer_rank < world_size_; ++peer_rank) {
        if (peer_rank != rank_) {
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
    // Simplified: In real implementation, use rendezvous store to exchange addresses
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        throw std::runtime_error("Failed to create socket for rank " + std::to_string(peer_rank));
    }

    return std::make_shared<TCPConnection>(sockfd);
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

    // Receive data
    Tensor cpu_tensor = get_cpu_buffer(tensor);
    size_t data_size = numel * dtype_size(cpu_tensor.dtype());
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
    // Ring all-reduce algorithm
    // Divide tensor into chunks and pipeline communication

    int num_chunks = world_size_;
    size_t chunk_size = (tensor.numel() + num_chunks - 1) / num_chunks;

    // Reduce-scatter phase
    for (int i = 0; i < world_size_ - 1; ++i) {
        int send_chunk = (rank_ - i + world_size_) % world_size_;
        int recv_chunk = (rank_ - i - 1 + world_size_) % world_size_;

        int send_rank = (rank_ + 1) % world_size_;
        int recv_rank = (rank_ - 1 + world_size_) % world_size_;

        // Extract chunk
        size_t chunk_start = recv_chunk * chunk_size;
        size_t chunk_end = std::min(chunk_start + chunk_size, static_cast<size_t>(tensor.numel()));

        if (chunk_start < tensor.numel()) {
            Tensor chunk = tensor.slice(0, chunk_start, chunk_end);
            Tensor recv_chunk_tensor = zeros_like(chunk);

            // Send/receive
            send_tensor(chunk, send_rank);
            recv_tensor(recv_chunk_tensor, recv_rank);

            // Reduce
            apply_reduce_op(chunk, recv_chunk_tensor, op);
        }
    }

    // All-gather phase
    for (int i = 0; i < world_size_ - 1; ++i) {
        int send_chunk = (rank_ - i + 1 + world_size_) % world_size_;
        int recv_chunk = (rank_ - i + world_size_) % world_size_;

        int send_rank = (rank_ + 1) % world_size_;
        int recv_rank = (rank_ - 1 + world_size_) % world_size_;

        size_t chunk_start = recv_chunk * chunk_size;
        size_t chunk_end = std::min(chunk_start + chunk_size, static_cast<size_t>(tensor.numel()));

        if (chunk_start < tensor.numel()) {
            Tensor chunk = tensor.slice(0, chunk_start, chunk_end);
            Tensor recv_chunk_tensor = zeros_like(chunk);

            send_tensor(chunk, send_rank);
            recv_tensor(recv_chunk_tensor, recv_rank);
        }
    }

    // Handle AVG operation
    if (op == ReduceOp::AVG) {
        tensor = tensor / static_cast<float>(world_size_);
    }
}

auto GlooBackend::get_cpu_buffer(const Tensor& tensor) -> Tensor {
    if (tensor.device().type == Device::Type::CPU) {
        return tensor;
    }
    return tensor.to(Device::cpu());
}

auto GlooBackend::validate_cpu_accessible(const Tensor& tensor) -> void {
    // Gloo can handle any tensor by copying to CPU if needed
    // No validation needed
}

} // namespace distributed
} // namespace tenzor
