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
 *
 * @note Threading model: TcpRpcAgent does NOT use fixed-size I/O or worker
 *       thread pools. It spawns exactly one accept thread plus one
 *       receive_loop thread per peer connection, and runs each send_async()
 *       on its own short-lived thread. The @ref num_io_threads and
 *       @ref num_worker_threads fields are therefore currently advisory only
 *       and are not consumed by the implementation.
 *
 * @note Heartbeats: the agent responds to inbound HEARTBEAT messages
 *       (HEARTBEAT_ACK) but does not run a heartbeat *sender*. Active
 *       liveness probing is performed by HealthMonitor (see
 *       elastic/health_monitor.hpp), which has its own interval/threshold
 *       config. The @ref enable_heartbeat and @ref heartbeat_interval_ms
 *       fields here are not consumed by TcpRpcAgent; configure liveness via
 *       HealthMonitorConfig instead.
 */
struct RpcAgentConfig {
    /// Advisory only; not consumed (one accept thread + one thread per peer).
    int32_t num_io_threads{2};
    /// Advisory only; not consumed (per-connection receive_loop threading).
    int32_t num_worker_threads{4};
    int32_t timeout_ms{60000};        ///< RPC timeout in milliseconds
    /// Advisory only; active probing lives in HealthMonitor, not here.
    int32_t heartbeat_interval_ms{5000};
    /// Advisory only; the agent only ACKs heartbeats, it does not send them.
    bool enable_heartbeat{true};
};

/**
 * @brief TCP-based RPC agent with persistent full-mesh connections.
 *
 * Manages inter-worker communication over TCP sockets. Each worker
 * maintains persistent connections to all other workers for low-latency
 * messaging. A single accept thread handles inbound connections; each
 * established connection gets its own receive_loop thread that executes RPC
 * function bodies. (There are no fixed-size I/O/worker pools — see
 * RpcAgentConfig.)
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

    // Threading. worker_threads_ is mutated from several threads
    // (get_or_connect on the caller thread, accept_loop on the io thread) and
    // iterated/joined by shutdown(), so all access is guarded by threads_mutex_.
    std::mutex threads_mutex_;
    std::vector<std::thread> io_threads_;
    std::vector<std::thread> worker_threads_;

    // In-flight async sends. send_async() spawns detached worker threads that
    // dereference `this` for the duration of a blocking send(); shutdown() must
    // wait for them to drain before members are destroyed or it is a
    // use-after-free. Tracked via a counter drained under async_mutex_.
    std::mutex async_mutex_;
    std::condition_variable async_cv_;
    int64_t inflight_async_{0};

    // Request/response tracking
    struct PendingRequest {
        std::promise<Message> promise;
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
    // Dispatch an inbound/loopback message. `inbound_fd` is the socket the
    // message arrived on (>=0 from receive_loop), so a remote request's response
    // can be written back on that same connection instead of dialing a fresh
    // outbound one; -1 (the default) means no inbound socket (self-loopback).
    auto dispatch_message(Message msg, int inbound_fd = -1) -> void;
    auto handle_rpc_call(const Message& msg) -> Message;
};

} // namespace rpc
} // namespace distributed
} // namespace tenzor
