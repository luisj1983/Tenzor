/**
 * @file rref.hpp
 * @brief Remote references for distributed tensor access
 */

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include "types.hpp"
#include "rpc_agent.hpp"

namespace tenzor {
namespace distributed {
namespace rpc {

/**
 * @brief Remote reference to a tensor on another worker.
 *
 * RRef provides transparent access to tensors living on remote workers.
 * The owner worker holds the actual data; user RRefs hold a reference
 * that fetches the data on demand.
 *
 * Distributed reference counting ensures the owner doesn't delete
 * the data while any user RRef still exists.
 *
 * Usage:
 * @code
 * // On worker 0: create a remote tensor on worker 1
 * auto rref = remote(1, "create_tensor", input);
 *
 * // Fetch data (blocking)
 * Tensor local = rref.to_here();
 *
 * // RRef destructor sends RREF_DELETE to owner
 * @endcode
 */
class RRef {
public:
    /**
     * @brief Construct an RRef.
     *
     * @param owner_id Worker ID that owns the data
     * @param rref_id Unique reference ID
     * @param agent RPC agent for communication
     */
    RRef(int32_t owner_id, int64_t rref_id, std::shared_ptr<TcpRpcAgent> agent);

    /// Move-only semantics
    RRef(RRef&& other) noexcept;
    auto operator=(RRef&& other) noexcept -> RRef&;
    RRef(const RRef&) = delete;
    auto operator=(const RRef&) -> RRef& = delete;

    ~RRef();

    /**
     * @brief Fetch the referenced tensor to the local worker.
     *
     * Sends RREF_FETCH to the owner and blocks until data arrives.
     *
     * @return Local copy of the tensor
     */
    auto to_here() -> Tensor;

    /**
     * @brief Check if this RRef points to local data.
     */
    auto is_owner() const -> bool;

    /**
     * @brief Get the owner worker ID.
     */
    auto owner_id() const -> int32_t { return owner_id_; }

    /**
     * @brief Get the unique reference ID.
     */
    auto rref_id() const -> int64_t { return rref_id_; }

private:
    /**
     * @brief Release the referenced storage.
     *
     * For an owner RRef, erases the entry from the local RRefStore. For a
     * non-owner RRef, sends RREF_DELETE to the owner for remote GC. Idempotent:
     * a no-op once already released. Called from the destructor and move-assign.
     */
    auto release() noexcept -> void;

    int32_t owner_id_{-1};
    int64_t rref_id_{-1};
    std::shared_ptr<TcpRpcAgent> agent_;
    bool valid_{true};
};

/**
 * @brief Storage for locally-owned RRef data.
 *
 * Each worker maintains a store of tensors that are referenced
 * by remote RRefs. The store handles RREF_FETCH and RREF_DELETE
 * requests from remote workers.
 */
class RRefStore {
public:
    static auto instance() -> RRefStore&;

    /**
     * @brief Store a tensor and get a unique RRef ID.
     *
     * @param tensor       Value to store.
     * @param owner_worker Worker id permitted to fetch/delete this entry over
     *                     the network. RRef ids are sequential and therefore
     *                     predictable, so remote RREF_FETCH/RREF_DELETE requests
     *                     are only served when their src_worker matches this id.
     *                     The default (-1) marks a purely-local entry that no
     *                     remote worker may access.
     */
    auto store(Tensor tensor, int32_t owner_worker = -1) -> int64_t;

    /**
     * @brief Fetch a stored tensor by RRef ID (local, unchecked).
     *
     * Used by the owning worker for in-process access. Network requests must
     * use the requester-checked overload.
     */
    auto fetch(int64_t rref_id) -> Tensor;

    /**
     * @brief Fetch a stored tensor, enforcing ownership.
     *
     * @throws std::runtime_error if the id is unknown or @p requester is not the
     *         worker that owns the entry.
     */
    auto fetch(int64_t rref_id, int32_t requester) -> Tensor;

    /**
     * @brief Delete a stored tensor (local, unchecked).
     */
    auto remove(int64_t rref_id) -> void;

    /**
     * @brief Delete a stored tensor, enforcing ownership (remote GC).
     *
     * A no-op if the id is unknown (idempotent GC).
     * @throws std::runtime_error if the entry exists but @p requester is not its
     *         owner.
     */
    auto remove(int64_t rref_id, int32_t requester) -> void;

private:
    RRefStore() = default;
    struct Entry {
        Tensor tensor;
        int32_t owner{-1};  ///< Worker id permitted to fetch/delete over the wire.
    };
    std::mutex mutex_;
    std::unordered_map<int64_t, Entry> store_;
    std::atomic<int64_t> next_id_{0};
};

} // namespace rpc
} // namespace distributed
} // namespace tenzor
