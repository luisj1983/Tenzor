/**
 * @file rpc.cpp
 * @brief Implementation of high-level RPC API
 */

#include "tenzor/distributed/rpc/rpc.hpp"
#include <stdexcept>

namespace tenzor {
namespace distributed {
namespace rpc {

namespace {
    std::shared_ptr<TcpRpcAgent> g_agent;
}

auto init_rpc(const std::string& name, int32_t rank, int32_t world_size,
              RpcAgentConfig config) -> void {
    WorkerInfo self;
    self.name = name;
    self.id = rank;
    self.address = "127.0.0.1";
    self.port = 29500 + rank;

    g_agent = std::make_shared<TcpRpcAgent>(std::move(self), std::move(config));

    // Build worker list (in production, exchange via ProcessGroup::all_gather)
    std::vector<WorkerInfo> all_workers;
    all_workers.reserve(world_size);
    for (int32_t i = 0; i < world_size; ++i) {
        WorkerInfo w;
        w.name = "worker_" + std::to_string(i);
        w.id = i;
        w.address = "127.0.0.1";
        w.port = 29500 + i;
        all_workers.push_back(std::move(w));
    }

    g_agent->init(all_workers);
}

auto shutdown_rpc() -> void {
    if (g_agent) {
        g_agent->shutdown();
        g_agent.reset();
    }
}

auto get_agent() -> std::shared_ptr<TcpRpcAgent> {
    if (!g_agent) {
        throw std::runtime_error("RPC not initialized. Call init_rpc() first.");
    }
    return g_agent;
}

auto rpc_sync(int32_t dst, const std::string& func_name,
              const std::vector<Tensor>& args) -> std::vector<Tensor> {
    auto agent = get_agent();

    Message msg;
    msg.type = MessageType::RPC_CALL;
    msg.dst_worker = dst;
    msg.payload.function_name = func_name;
    msg.payload.tensors = args;

    auto response = agent->send(std::move(msg));

    if (response.type == MessageType::RPC_ERROR) {
        throw std::runtime_error("RPC error: " + response.payload.function_name);
    }

    return std::move(response.payload.tensors);
}

auto rpc_async(int32_t dst, const std::string& func_name,
               const std::vector<Tensor>& args) -> std::future<std::vector<Tensor>> {
    auto agent = get_agent();

    auto promise = std::make_shared<std::promise<std::vector<Tensor>>>();
    auto future = promise->get_future();

    Message msg;
    msg.type = MessageType::RPC_CALL;
    msg.dst_worker = dst;
    msg.payload.function_name = func_name;
    msg.payload.tensors = args;

    agent->send_async(std::move(msg),
        [promise](Message response) {
            if (response.type == MessageType::RPC_ERROR) {
                promise->set_exception(std::make_exception_ptr(
                    std::runtime_error("RPC error: " + response.payload.function_name)));
            } else {
                promise->set_value(std::move(response.payload.tensors));
            }
        });

    return future;
}

auto remote(int32_t dst, const std::string& func_name,
            const std::vector<Tensor>& args) -> RRef {
    auto agent = get_agent();

    // Execute the function on `dst`. rpc_sync runs it on the destination worker
    // and transfers the result back to this worker.
    auto results = rpc_sync(dst, func_name, args);

    // Materialize the result in the local RRefStore and return an RRef owned by
    // THIS worker — that is where the value now lives, so to_here()/local_value()
    // resolve correctly for any caller. (Eager fetch: the lazy "keep the value on
    // `dst`, fetch on demand" optimization would need a dedicated remote-store RPC
    // message type. The previous code returned a broken RRef with id=0 for any
    // non-self dst, so cross-rank remote() never resolved.)
    Tensor value = results.empty()
        ? Tensor({}, DType::Float32, Device::cpu())
        : results[0];
    int64_t rref_id = RRefStore::instance().store(value);
    return RRef(agent->self().id, rref_id, agent);
}

} // namespace rpc
} // namespace distributed
} // namespace tenzor
