/**
 * @file rpc_agent.hpp
 * @brief TCP-based RPC agent for inter-worker communication
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include "types.hpp"
#include "function_registry.hpp"

namespace tenzor {
namespace distributed {
namespace rpc {

/**
 * @brief Configuration for the RPC agent.
 */
struct RpcAgentConfig {
    int32_t num_io_threads{2};        ///< I/O threads for socket operations
    int32_t num_worker_threads{4};    ///< Worker threads for RPC execution
    int32_t timeout_ms{60000};        ///< RPC timeout in milliseconds
    int32_t heartbeat_interval_ms{5000}; ///< Heartbeat interval
    bool enable_heartbeat{true};      ///< Enable health monitoring
};

/**
 * @brief TCP-based RPC agent with persistent full-mesh connections.
 *
 * Manages inter-worker communication over TCP sockets. Each worker
 * maintains persistent connections to all other workers for low-latency
 * messaging. Separate I/O threads handle socket operations while worker
 * threads execute RPC function bodies.
 *
 * Lifecycle:
 * 1. Construct with local worker info
 * 2. Call init() with all worker info to establish connections
 * 3. Use send() / send_async() for messaging
 * 4. Call shutdown() to cleanly close connections
 */
class TcpRpcAgent {
public:
    /**
     * @brief Construct RPC agent for the local worker.
     *
     * @param self Local worker info
     * @param config Agent configuration
     */
    TcpRpcAgent(WorkerInfo self, RpcAgentConfig config = {});
    ~TcpRpcAgent();

    /**
     * @brief Initialize connections to all workers.
     *
     * Establishes persistent TCP connections in a full-mesh topology.
     * Should be called after all workers are ready.
     *
     * @param all_workers Info for all workers in the group
     */
    auto init(const std::vector<WorkerInfo>& all_workers) -> void;

    /**
     * @brief Send a message and wait for response.
     *
     * @param msg Message to send
     * @return Response message
     * @throws std::runtime_error on timeout or connection failure
     */
    auto send(Message msg) -> Message;

    /**
     * @brief Send a message asynchronously.
     *
     * @param msg Message to send
     * @param callback Called with response when received
     */
    auto send_async(Message msg, std::function<void(Message)> callback) -> void;

    /**
     * @brief Get the local worker info.
     */
    auto self() const -> const WorkerInfo& { return self_; }

    /**
     * @brief Check if the agent is initialized and running.
     */
    auto is_running() const -> bool { return running_.load(std::memory_order_acquire); }

    /**
     * @brief Gracefully shut down the agent.
     */
    auto shutdown() -> void;

    /**
     * @brief Register a message handler for custom message types.
     *
     * @param type Message type to handle
     * @param handler Function called when message of this type is received
     */
    auto register_handler(MessageType type,
                          std::function<Message(const Message&)> handler) -> void;

private:
    WorkerInfo self_;
    RpcAgentConfig config_;
    std::atomic<bool> running_{false};
    std::atomic<int64_t> next_request_id_{0};

    // Connection state (connections_ mutated from multiple threads).
    std::vector<WorkerInfo> workers_;
    std::mutex connections_mutex_;
    std::unordered_map<int32_t, int> connections_;  // worker_id -> outbound socket_fd
    std::unordered_map<int32_t, int> inbound_connections_;  // worker_id -> inbound socket_fd
    // One write-mutex per fd so concurrent send_framed() calls on the same
    // socket can't interleave bytes of two different messages.
    std::unordered_map<int, std::shared_ptr<std::mutex>> fd_write_mutexes_;
    int listen_fd_ = -1;
    // Signals that accept_loop() has finished binding/listening (or
    // failed). Used by init() to surface bind failures synchronously.
    std::promise<bool> listen_ready_;

    // Threading
    std::vector<std::thread> io_threads_;
    std::vector<std::thread> worker_threads_;

    // Request/response tracking
    struct PendingRequest {
        std::promise<Message> promise;
        std::chrono::steady_clock::time_point deadline;
    };
    std::mutex pending_mutex_;
    std::unordered_map<int64_t, PendingRequest> pending_;

    // Message handlers
    std::mutex handler_mutex_;
    std::unordered_map<MessageType, std::function<Message(const Message&)>> handlers_;

    // Internal methods
    void accept_loop();
    void receive_loop(int fd, int32_t peer_id);
    int  get_or_connect(int32_t peer_id);
    // Send a message on fd with per-fd write serialization to prevent
    // interleaving under concurrent send() calls. Returns false on write
    // failure (connection dropped).
    bool send_framed_locked(int fd, const Message& msg);
    auto dispatch_message(Message msg) -> void;
    auto handle_rpc_call(const Message& msg) -> Message;
};

} // namespace rpc
} // namespace distributed
} // namespace tenzor
