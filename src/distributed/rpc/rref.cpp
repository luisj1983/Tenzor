/**
 * @file rref.cpp
 * @brief Implementation of remote references
 */

#include "tenzor/distributed/rpc/rref.hpp"

namespace tenzor {
namespace distributed {
namespace rpc {

// ============================================================================
// RRef
// ============================================================================

RRef::RRef(int32_t owner_id, int64_t rref_id, std::shared_ptr<TcpRpcAgent> agent)
    : owner_id_(owner_id), rref_id_(rref_id), agent_(std::move(agent)) {}

RRef::RRef(RRef&& other) noexcept
    : owner_id_(other.owner_id_), rref_id_(other.rref_id_),
      agent_(std::move(other.agent_)), valid_(other.valid_) {
    other.valid_ = false;
}

auto RRef::operator=(RRef&& other) noexcept -> RRef& {
    if (this != &other) {
        // Send delete for current ref if valid
        if (valid_ && agent_ && !is_owner()) {
            try {
                Message msg;
                msg.type = MessageType::RREF_DELETE;
                msg.dst_worker = owner_id_;
                msg.payload.request_id = rref_id_;
                agent_->send_async(std::move(msg), [](Message) {});
            } catch (...) {}
        }

        owner_id_ = other.owner_id_;
        rref_id_ = other.rref_id_;
        agent_ = std::move(other.agent_);
        valid_ = other.valid_;
        other.valid_ = false;
    }
    return *this;
}

RRef::~RRef() {
    if (valid_ && agent_ && !is_owner()) {
        try {
            Message msg;
            msg.type = MessageType::RREF_DELETE;
            msg.dst_worker = owner_id_;
            msg.payload.request_id = rref_id_;
            agent_->send_async(std::move(msg), [](Message) {});
        } catch (...) {
            // Suppress exceptions in destructor
        }
    }
}

auto RRef::to_here() -> Tensor {
    if (!valid_) {
        throw std::runtime_error("RRef::to_here() called on invalid RRef");
    }

    if (is_owner()) {
        // Local fetch
        return RRefStore::instance().fetch(rref_id_);
    }

    // Remote fetch
    Message msg;
    msg.type = MessageType::RREF_FETCH;
    msg.dst_worker = owner_id_;
    msg.payload.request_id = rref_id_;

    auto response = agent_->send(std::move(msg));
    if (response.type == MessageType::RPC_ERROR) {
        throw std::runtime_error("RRef fetch failed: " + response.payload.function_name);
    }

    if (response.payload.tensors.empty()) {
        throw std::runtime_error("RRef fetch returned no tensors");
    }

    return response.payload.tensors[0];
}

auto RRef::is_owner() const -> bool {
    return agent_ && owner_id_ == agent_->self().id;
}

// ============================================================================
// RRefStore
// ============================================================================

auto RRefStore::instance() -> RRefStore& {
    static RRefStore store;
    return store;
}

auto RRefStore::store(Tensor tensor) -> int64_t {
    auto id = next_id_.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(mutex_);
    store_[id] = std::move(tensor);
    return id;
}

auto RRefStore::fetch(int64_t rref_id) -> Tensor {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = store_.find(rref_id);
    if (it == store_.end()) {
        throw std::runtime_error("RRef not found: " + std::to_string(rref_id));
    }
    return it->second;
}

auto RRefStore::remove(int64_t rref_id) -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    store_.erase(rref_id);
}

} // namespace rpc
} // namespace distributed
} // namespace tenzor
