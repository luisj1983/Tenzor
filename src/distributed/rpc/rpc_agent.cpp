/**
 * @file rpc_agent.cpp
 * @brief Implementation of TCP-based RPC agent
 */

#include "tenzor/distributed/rpc/rpc_agent.hpp"
#include <cstring>
#include <stdexcept>

#ifdef __linux__
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#endif

namespace tenzor {
namespace distributed {
namespace rpc {

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

    // Establish TCP connections to all workers
    // In a production implementation, this would:
    // 1. Start a listening socket on self_.port
    // 2. Accept connections from workers with ID < self_.id
    // 3. Connect to workers with ID > self_.id
    // 4. Launch I/O threads for each connection
    // 5. Launch worker threads for RPC execution

    // Register default handlers
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
}

auto TcpRpcAgent::send(Message msg) -> Message {
    if (!running_.load(std::memory_order_acquire)) {
        throw std::runtime_error("RPC agent is not running");
    }

    auto request_id = next_request_id_.fetch_add(1, std::memory_order_relaxed);
    msg.payload.request_id = request_id;
    msg.src_worker = self_.id;

    // Create a promise/future pair for synchronous waiting
    std::promise<Message> promise;
    auto future = promise.get_future();

    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        pending_[request_id] = PendingRequest{
            std::move(promise),
            std::chrono::steady_clock::now() + std::chrono::milliseconds(config_.timeout_ms)
        };
    }

    // Send the message (in production: serialize and write to socket)
    // For local execution / testing, dispatch directly if dst == self
    if (msg.dst_worker == self_.id) {
        dispatch_message(std::move(msg));
    }

    // Wait for response
    auto status = future.wait_for(std::chrono::milliseconds(config_.timeout_ms));
    if (status == std::future_status::timeout) {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        pending_.erase(request_id);
        throw std::runtime_error("RPC timeout: request " + std::to_string(request_id));
    }

    return future.get();
}

auto TcpRpcAgent::send_async(Message msg, std::function<void(Message)> callback) -> void {
    if (!running_.load(std::memory_order_acquire)) {
        throw std::runtime_error("RPC agent is not running");
    }

    auto request_id = next_request_id_.fetch_add(1, std::memory_order_relaxed);
    msg.payload.request_id = request_id;
    msg.src_worker = self_.id;

    // Store callback for when response arrives
    std::promise<Message> promise;
    auto future = promise.get_future();

    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        pending_[request_id] = PendingRequest{
            std::move(promise),
            std::chrono::steady_clock::now() + std::chrono::milliseconds(config_.timeout_ms)
        };
    }

    // Launch async waiter
    std::thread([future = std::move(future), callback = std::move(callback)]() mutable {
        try {
            auto response = future.get();
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

    // Close all connections
    for (auto& [id, fd] : connections_) {
#ifdef __linux__
        ::close(fd);
#endif
    }
    connections_.clear();

    // Join threads
    for (auto& t : io_threads_) {
        if (t.joinable()) t.join();
    }
    for (auto& t : worker_threads_) {
        if (t.joinable()) t.join();
    }
}

auto TcpRpcAgent::register_handler(MessageType type,
                                    std::function<Message(const Message&)> handler) -> void {
    std::lock_guard<std::mutex> lock(handler_mutex_);
    handlers_[type] = std::move(handler);
}

auto TcpRpcAgent::dispatch_message(Message msg) -> void {
    std::function<Message(const Message&)> handler;

    {
        std::lock_guard<std::mutex> lock(handler_mutex_);
        auto it = handlers_.find(msg.type);
        if (it != handlers_.end()) {
            handler = it->second;
        }
    }

    if (handler) {
        auto response = handler(msg);

        // Deliver response to pending request
        std::lock_guard<std::mutex> lock(pending_mutex_);
        auto it = pending_.find(msg.payload.request_id);
        if (it != pending_.end()) {
            it->second.promise.set_value(std::move(response));
            pending_.erase(it);
        }
    }
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

} // namespace rpc
} // namespace distributed
} // namespace tenzor
