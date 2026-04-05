/**
 * @file types.hpp
 * @brief Core types for the RPC framework
 */

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include "../../core/tensor.hpp"

namespace tenzor {
namespace distributed {
namespace rpc {

/**
 * @brief RPC message types.
 */
enum class MessageType : uint8_t {
    RPC_CALL,          ///< Remote function call request
    RPC_RESPONSE,      ///< Response with result tensors
    RPC_ERROR,         ///< Error response
    RREF_FETCH,        ///< Fetch data from remote reference
    RREF_DELETE,       ///< Delete remote reference (GC)
    HEARTBEAT,         ///< Health check ping
    HEARTBEAT_ACK,     ///< Health check response
    SHUTDOWN,          ///< Graceful shutdown request
};

/**
 * @brief Information about a worker in the RPC group.
 */
struct WorkerInfo {
    std::string name;       ///< Human-readable name
    int32_t id{-1};         ///< Numeric ID (rank)
    std::string address;    ///< Hostname or IP
    int32_t port{0};        ///< Listening port

    auto operator==(const WorkerInfo& other) const -> bool { return id == other.id; }
};

/**
 * @brief Serialized payload for RPC messages.
 *
 * Tensors are separated from byte data to enable zero-copy
 * transfer on RDMA-capable networks.
 */
struct SerializedPayload {
    std::string function_name;       ///< Name of function to call (for RPC_CALL)
    std::vector<uint8_t> bytes;      ///< Serialized non-tensor arguments
    std::vector<Tensor> tensors;     ///< Tensor arguments (zero-copy potential)
    int64_t request_id{0};           ///< Correlation ID for request/response matching
};

/**
 * @brief Complete RPC message (header + payload).
 */
struct Message {
    MessageType type;
    int32_t src_worker{-1};          ///< Sender worker ID
    int32_t dst_worker{-1};          ///< Destination worker ID
    SerializedPayload payload;
};

/**
 * @brief Result type for async RPC calls.
 */
struct RpcResult {
    std::vector<Tensor> tensors;     ///< Result tensors
    std::string error;               ///< Error message (empty on success)

    auto is_error() const -> bool { return !error.empty(); }
};

/**
 * @brief Type signature for registered RPC functions.
 */
using RpcFunction = std::function<std::vector<Tensor>(const std::vector<Tensor>&)>;

} // namespace rpc
} // namespace distributed
} // namespace tenzor
