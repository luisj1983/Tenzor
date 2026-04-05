/**
 * @file rpc.hpp
 * @brief High-level RPC API: rpc_sync, rpc_async, remote
 */

#pragma once

#include <future>
#include <string>
#include <vector>
#include "rpc_agent.hpp"
#include "rref.hpp"
#include "types.hpp"

namespace tenzor {
namespace distributed {
namespace rpc {

/**
 * @brief Initialize the RPC framework.
 *
 * Must be called before any RPC operations. Bootstraps worker
 * discovery using the existing ProcessGroup infrastructure.
 *
 * @param name Local worker name
 * @param rank Local rank
 * @param world_size Total number of workers
 * @param config Agent configuration
 */
auto init_rpc(const std::string& name, int32_t rank, int32_t world_size,
              RpcAgentConfig config = {}) -> void;

/**
 * @brief Shut down the RPC framework.
 */
auto shutdown_rpc() -> void;

/**
 * @brief Get the global RPC agent.
 */
auto get_agent() -> std::shared_ptr<TcpRpcAgent>;

/**
 * @brief Synchronous RPC call.
 *
 * Calls a registered function on a remote worker and blocks
 * until the result is available.
 *
 * @param dst Destination worker ID
 * @param func_name Name of the registered function
 * @param args Tensor arguments
 * @return Result tensors
 */
auto rpc_sync(int32_t dst, const std::string& func_name,
              const std::vector<Tensor>& args) -> std::vector<Tensor>;

/**
 * @brief Asynchronous RPC call.
 *
 * Calls a registered function on a remote worker and returns
 * a future that will hold the result.
 *
 * @param dst Destination worker ID
 * @param func_name Name of the registered function
 * @param args Tensor arguments
 * @return Future holding result tensors
 */
auto rpc_async(int32_t dst, const std::string& func_name,
               const std::vector<Tensor>& args) -> std::future<std::vector<Tensor>>;

/**
 * @brief Create a remote reference.
 *
 * Calls a registered function on a remote worker and returns
 * an RRef that can lazily fetch the result.
 *
 * @param dst Destination worker ID
 * @param func_name Name of the registered function
 * @param args Tensor arguments
 * @return Remote reference to the result
 */
auto remote(int32_t dst, const std::string& func_name,
            const std::vector<Tensor>& args) -> RRef;

} // namespace rpc
} // namespace distributed
} // namespace tenzor
