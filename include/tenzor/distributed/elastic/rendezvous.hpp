/**
 * @file rendezvous.hpp
 * @brief Store-based dynamic membership rendezvous protocol
 *
 * Enables workers to join/leave training dynamically. Uses a
 * key-value store (RendezvousStore from Gloo backend) for coordination
 * without requiring external infrastructure like etcd.
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tenzor {
namespace distributed {

class RendezvousStore;  // shared key-value store (gloo_backend.hpp)

namespace elastic {

/**
 * @brief Configuration for the rendezvous protocol.
 */
struct RendezvousConfig {
    int32_t min_workers{1};                ///< Minimum workers to proceed
    int32_t max_workers{256};              ///< Maximum workers allowed
    std::chrono::seconds timeout{300};     ///< Timeout for gathering workers
    std::string run_id;                    ///< Unique run identifier
    std::string store_addr{"localhost"};   ///< Store address
    int32_t store_port{29400};             ///< Store port
};

/**
 * @brief Result of a rendezvous round.
 */
struct RendezvousResult {
    int32_t rank{-1};                      ///< Assigned rank
    int32_t world_size{0};                 ///< Total number of participants
    std::string store_key;                 ///< Key prefix for this round
};

/**
 * @brief C10d-style store-based rendezvous for elastic training.
 *
 * Workers write join keys to a shared store. A coordinator (typically
 * rank 0 from the previous round, or the first joiner) collects
 * participants and assigns new ranks once [min_workers, max_workers]
 * are assembled.
 *
 * No external dependencies (etcd, ZooKeeper) required — uses the
 * existing RendezvousStore TCP implementation from gloo_backend.hpp.
 */
class C10dRendezvous {
public:
    explicit C10dRendezvous(RendezvousConfig config);

    /**
     * @brief Join the rendezvous and get assigned rank.
     *
     * Blocks until enough workers have joined or timeout expires.
     *
     * @return Rendezvous result with assigned rank and world_size
     * @throws std::runtime_error on timeout or if min_workers not met
     */
    auto join() -> RendezvousResult;

    /**
     * @brief Leave the current rendezvous.
     *
     * Notifies other workers that this worker is departing.
     */
    auto leave() -> void;

    /**
     * @brief Get current world size.
     */
    auto world_size() const -> int32_t { return world_size_; }

    /**
     * @brief Get current rank.
     */
    auto rank() const -> int32_t { return rank_; }

private:
    /// Lazily-created connection to the shared store. Held for the rendezvous
    /// object's lifetime so a coordinator's master server keeps serving across
    /// rounds (and after join() returns, while stragglers read the result).
    auto ensure_store() -> RendezvousStore&;
    /// Per-process unique worker identifier (hostname:pid).
    auto worker_id() const -> std::string;

    RendezvousConfig config_;
    std::unique_ptr<RendezvousStore> store_;
    int32_t rank_{-1};
    int32_t world_size_{0};
    int64_t round_{0};
    /// Round number of the last successful join(). leave() keys its departure
    /// marker off this rather than round_, which join() may have advanced past
    /// (e.g. after a failed re-rendezvous) — writing the marker under the
    /// advanced round would notify peers in a round this worker never joined.
    int64_t joined_round_{-1};

public:
    ~C10dRendezvous();
};

} // namespace elastic
} // namespace distributed
} // namespace tenzor
