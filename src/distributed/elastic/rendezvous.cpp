/**
 * @file rendezvous.cpp
 * @brief Implementation of store-based elastic rendezvous
 */

#include "tenzor/distributed/elastic/rendezvous.hpp"
#include <stdexcept>
#include <thread>

namespace tenzor {
namespace distributed {
namespace elastic {

C10dRendezvous::C10dRendezvous(RendezvousConfig config)
    : config_(std::move(config)) {}

auto C10dRendezvous::join() -> RendezvousResult {
    // In a full implementation, this would:
    // 1. Connect to the RendezvousStore at config_.store_addr:store_port
    // 2. Write a join key: "{run_id}/round_{round}/join_{worker_id}"
    // 3. Wait for the coordinator to collect [min_workers, max_workers] joiners
    // 4. Read assigned rank from store: "{run_id}/round_{round}/rank_{worker_id}"

    ++round_;

    RendezvousResult result;
    result.rank = rank_;
    result.world_size = world_size_;
    result.store_key = config_.run_id + "/round_" + std::to_string(round_);

    return result;
}

auto C10dRendezvous::leave() -> void {
    // Write leave key to store
    // Other workers will detect departure during next rendezvous round
    rank_ = -1;
    world_size_ = 0;
}

} // namespace elastic
} // namespace distributed
} // namespace tenzor
