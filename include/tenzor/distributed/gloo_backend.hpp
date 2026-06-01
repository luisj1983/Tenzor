/**
 * @file gloo_backend.hpp
 * @brief Gloo backend for CPU-based distributed communication
 *
 * Implements CPU collective operations using TCP/IP sockets. Works on any
 * hardware and serves as fallback when GPU communication is unavailable.
 */

#pragma once

#include "distributed.hpp"
#include "work_executor.hpp"  // B.3: async worker thread pool
#include <vector>
#include <memory>
#include <thread>
#include <atomic>

namespace tenzor {
namespace distributed {

/**
 * @brief Simple TCP connection for process communication.
 */
class TCPConnection {
public:
    explicit TCPConnection(int socket_fd, int timeout_seconds = 0);
    ~TCPConnection();

    auto send(const void* data, size_t size) -> void;
    auto recv(void* data, size_t size) -> void;
    auto close() -> void;

    /**
     * @brief Apply send/recv timeouts to the underlying socket.
     * @param seconds Timeout in seconds (0 = no timeout)
     */
    auto set_timeout(int seconds) -> void;

private:
    int socket_fd_{-1};
    bool closed_{false};
    int timeout_seconds_{0};
};

/**
 * @brief Rendezvous store for process coordination.
 *
 * Provides key-value store for exchanging initialization data
 * across processes (e.g., socket addresses).
 */
class RendezvousStore {
public:
    RendezvousStore(const std::string& master_addr, int master_port, int rank, int world_size);
    ~RendezvousStore();

    /**
     * @brief Set key-value pair.
     */
    auto set(const std::string& key, const std::string& value) -> void;

    /**
     * @brief Get value for key.
     */
    auto get(const std::string& key) -> std::string;

    /**
     * @brief Wait for all processes to set a key.
     */
    auto wait(const std::string& key) -> void;

    /**
     * @brief Delete a key from the store.
     *
     * Used by elastic rendezvous to clean up stale worker entries.
     *
     * @param key Key to delete
     * @return true if key existed and was deleted
     */
    auto delete_key(const std::string& key) -> bool;

    /**
     * @brief Check if a key exists without blocking.
     *
     * @param key Key to check
     * @return true if key exists
     */
    auto check_key(const std::string& key) -> bool;

    /**
     * @brief Atomically add `delta` to the integer value at `key` (missing key
     *        treated as 0) and return the new value.
     *
     * Used by the elastic rendezvous to allocate unique, monotonically
     * increasing participant slots without a coordinator-election race. The
     * increment is serialized by the master store's mutex.
     */
    auto add(const std::string& key, int64_t delta) -> int64_t;

private:
    std::string master_addr_;
    int master_port_;
    int rank_;
    int world_size_;
    int socket_fd_{-1};
    int listen_fd_{-1};                  ///< Master server listening socket.
    std::atomic<bool> stop_{false};      ///< Signals the server thread to exit.

    auto connect_to_master() -> void;
    auto run_master_server() -> void;
    std::unique_ptr<std::thread> server_thread_;
};

/**
 * @brief Gloo communication backend for CPU operations.
 *
 * Implements collective operations using TCP/IP sockets. Supports any
 * hardware and works as a fallback when GPU communication is unavailable.
 *
 * Features:
 * - Pure CPU implementation (no GPU required)
 * - TCP/IP based communication
 * - Works on any network topology
 * - Support for CPU tensors and host memory
 * - Automatic buffer management
 *
 * Algorithm: Ring all-reduce for bandwidth-optimal communication
 *
 * Limitations:
 * - Slower than NCCL for GPU tensors
 * - Higher latency due to TCP/IP overhead
 * - Best suited for small-scale CPU training or as fallback
 */
class GlooBackend : public CommunicationBackend {
public:
    /**
     * @brief Construct Gloo backend.
     */
    GlooBackend();

    /**
     * @brief Destructor - cleanup connections.
     */
    ~GlooBackend() override;

    // CommunicationBackend interface

    auto initialize(int rank, int world_size, const std::string& master_addr,
                   int master_port) -> void override;

    auto broadcast(Tensor& tensor, int src_rank) -> void override;

    auto all_reduce(Tensor& tensor, ReduceOp op) -> void override;

    auto reduce(Tensor& tensor, int dst_rank, ReduceOp op) -> void override;

    auto all_gather(const Tensor& tensor, std::vector<Tensor>& output) -> void override;

    auto gather(const Tensor& tensor, std::vector<Tensor>& output, int dst_rank) -> void override;

    auto scatter(const std::vector<Tensor>& tensors, Tensor& output, int src_rank) -> void override;

    auto reduce_scatter(const std::vector<Tensor>& tensors, Tensor& output, ReduceOp op) -> void override;

    auto send(const Tensor& tensor, int dst_rank) -> void override;

    auto recv(Tensor& tensor, int src_rank) -> void override;

    auto barrier() -> void override;

    auto finalize() -> void override;

    auto backend_type() const -> Backend override { return Backend::GLOO; }

    auto supports_device(Device::Type device_type) const -> bool override;

    // B.3: real async via dedicated worker thread. The CommunicationBackend
    // async API returns `void`, so the executor runs each sync collective
    // on a background thread; DDP/FSDP-style overlap is achieved by
    // calling `wait_pending_async()` before depending on the result.
    // supports_async_stream() stays false (no GPU stream concept).
    auto all_reduce_async(Tensor& tensor, ReduceOp op,
                          void* stream) -> void override;
    auto all_gather_async(const Tensor& tensor,
                          std::vector<Tensor>& output,
                          void* stream) -> void override;
    auto reduce_scatter_async(const std::vector<Tensor>& tensors,
                              Tensor& output, ReduceOp op,
                              void* stream) -> void override;

    /// Block the caller until every enqueued async collective has
    /// completed. Re-throws the first exception caught from a worker
    /// task. Called by DDP/FSDP before the optimizer step.
    auto wait_pending_async() -> void;

private:
    int rank_{-1};
    int world_size_{-1};
    std::string master_addr_;
    int master_port_{29500};

    // TCP connections to other ranks
    std::vector<std::shared_ptr<TCPConnection>> connections_;
    std::unique_ptr<RendezvousStore> store_;
    bool initialized_{false};

    // Server socket for accepting connections
    int server_socket_{-1};
    int server_port_{0};
    std::unique_ptr<std::thread> accept_thread_;

    // B.3: worker for async collectives. Lazily constructed on first
    // *_async() call so cost is zero for sync-only deployments.
    std::unique_ptr<WorkExecutor> async_executor_;
    auto async_executor() -> WorkExecutor&;

    // Helper methods

    /**
     * @brief Initialize TCP connections to all other ranks.
     */
    auto init_connections() -> void;

    /**
     * @brief Create server socket for accepting connections.
     */
    auto create_server_socket() -> int;

    /**
     * @brief Accept connections from other ranks.
     */
    auto accept_connections() -> void;

    /**
     * @brief Connect to another rank.
     */
    auto connect_to_rank(int peer_rank) -> std::shared_ptr<TCPConnection>;

    /**
     * @brief Write port to file-based rendezvous store.
     */
    auto write_port_to_store(int rank, int port) -> void;

    /**
     * @brief Read port from file-based rendezvous store.
     */
    auto read_port_from_store(int rank) -> int;

    /**
     * @brief Send tensor data to peer.
     */
    auto send_tensor(const Tensor& tensor, int peer_rank) -> void;

    /**
     * @brief Receive tensor data from peer.
     */
    auto recv_tensor(Tensor& tensor, int peer_rank) -> void;

    /**
     * @brief Apply reduction operation to two tensors.
     */
    auto apply_reduce_op(Tensor& result, const Tensor& operand, ReduceOp op) -> void;

    /**
     * @brief Ring all-reduce implementation.
     *
     * Implements bandwidth-optimal ring all-reduce algorithm.
     */
    auto ring_all_reduce(Tensor& tensor, ReduceOp op) -> void;

    /**
     * @brief Get CPU buffer from tensor (handle GPU tensors).
     */
    auto get_cpu_buffer(const Tensor& tensor) -> Tensor;

    /**
     * @brief Validate tensor is accessible from CPU.
     */
    auto validate_cpu_accessible(const Tensor& tensor) -> void;
};

} // namespace distributed
} // namespace tenzor
