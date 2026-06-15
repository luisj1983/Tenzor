/**
 * @file rendezvous.cpp
 * @brief Store-based elastic rendezvous: dynamic-membership barrier + rank
 *        assignment on top of the gloo RendezvousStore.
 *
 * Protocol (per round, no external coordinator service):
 *   1. Every worker connects to the shared RendezvousStore. The first worker to
 *      bind the store port runs the master server; the rest are clients.
 *   2. Each worker atomically increments a per-round slot counter, obtaining a
 *      unique 0-based slot (== its rank if the round closes with it inside).
 *   3. Workers barrier until [min_workers, max_workers] have joined (with a
 *      short grace window to admit stragglers), or the timeout expires — in
 *      which case join() throws rather than silently returning rank=-1.
 *   4. The slot-0 worker freezes world_size; everyone reads it. Workers whose
 *      slot landed past world_size missed the round and are told to retry.
 */

#include "tenzor/distributed/elastic/rendezvous.hpp"
#include "tenzor/distributed/gloo_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>

namespace tenzor {
namespace distributed {
namespace elastic {

namespace {
constexpr auto kPollInterval = std::chrono::milliseconds(25);
constexpr auto kGraceWindow  = std::chrono::milliseconds(250);
}  // namespace

C10dRendezvous::C10dRendezvous(RendezvousConfig config)
    : config_(std::move(config)) {}

C10dRendezvous::~C10dRendezvous() = default;

auto C10dRendezvous::ensure_store() -> RendezvousStore& {
    if (!store_) {
        // rank=0 asks the store to run the master server; only the first worker
        // to bind the port succeeds, the rest fall back to client connections
        // to that server (run_master_server fails silently on EADDRINUSE).
        store_ = std::make_unique<RendezvousStore>(
            config_.store_addr, config_.store_port, /*rank=*/0,
            /*world_size=*/config_.max_workers);
    }
    return *store_;
}

auto C10dRendezvous::worker_id() const -> std::string {
    char host[256] = {0};
    if (::gethostname(host, sizeof(host) - 1) != 0) {
        std::snprintf(host, sizeof(host), "unknown");
    }
    return std::string(host) + ":" + std::to_string(static_cast<long long>(::getpid()));
}

auto C10dRendezvous::join() -> RendezvousResult {
    if (config_.min_workers < 1 || config_.max_workers < config_.min_workers) {
        throw std::runtime_error(
            "C10dRendezvous::join: invalid config (need 1 <= min_workers <= max_workers)");
    }

    ++round_;
    const std::string prefix = config_.run_id + "/round_" + std::to_string(round_);
    const std::string counter_key = prefix + "/slot_counter";
    const std::string world_size_key = prefix + "/world_size";

    RendezvousStore& store = ensure_store();

    // Claim a unique slot for this round (0-based). Slot 0 is the coordinator.
    const int64_t slot = store.add(counter_key, 1) - 1;
    store.set(prefix + "/member/" + std::to_string(slot), worker_id());

    // Barrier: gather participants until min_workers (then a grace window for
    // stragglers, capped at max_workers), or throw on timeout.
    const auto deadline = std::chrono::steady_clock::now() + config_.timeout;
    bool have_min = false;
    std::chrono::steady_clock::time_point min_reached_at{};
    int64_t count = 0;
    while (true) {
        count = store.add(counter_key, 0);  // read current participant count
        const auto now = std::chrono::steady_clock::now();

        if (count >= config_.max_workers) break;
        if (count >= config_.min_workers) {
            if (!have_min) { have_min = true; min_reached_at = now; }
            if (now - min_reached_at >= kGraceWindow) break;
        }
        if (now >= deadline) {
            if (count >= config_.min_workers) break;
            throw std::runtime_error(
                "C10dRendezvous::join: timed out after " +
                std::to_string(config_.timeout.count()) + "s with only " +
                std::to_string(count) + " of min_workers=" +
                std::to_string(config_.min_workers) + " joined (run_id='" +
                config_.run_id + "', round " + std::to_string(round_) + ")");
        }
        std::this_thread::sleep_for(kPollInterval);
    }

    // The coordinator (slot 0) freezes world_size for this round; everyone reads
    // it. Read with a parse-retry loop: a transient empty response under
    // concurrent store access must not be mistaken for a missing value.
    if (slot == 0) {
        // Clamp to max_workers: concurrent joiners can claim slots between two
        // 25ms polls, so the live counter can overshoot max_workers before the
        // barrier loop observes it. Freezing the raw count would produce
        // world_size > max_workers, inconsistent with ensure_store() (which
        // sizes the store to max_workers) and the fixed-size downstream
        // collectives.
        const int64_t frozen =
            std::min<int64_t>(count, static_cast<int64_t>(config_.max_workers));
        store.set(world_size_key, std::to_string(frozen));
    }
    int64_t world_size = 0;
    for (;;) {
        std::string ws;
        try { ws = store.get(world_size_key); } catch (...) { ws.clear(); }
        if (!ws.empty()) {
            try { world_size = std::stoll(ws); break; } catch (...) { /* retry */ }
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error(
                "C10dRendezvous::join: coordinator failed to publish world_size "
                "for run_id='" + config_.run_id + "' round " + std::to_string(round_));
        }
        std::this_thread::sleep_for(kPollInterval);
    }

    // A worker whose slot landed at or beyond the frozen world_size joined after
    // the round closed; tell it to retry (the caller starts the next round).
    if (slot >= world_size) {
        throw std::runtime_error(
            "C10dRendezvous::join: joined round " + std::to_string(round_) +
            " after it closed (slot " + std::to_string(slot) + " >= world_size " +
            std::to_string(world_size) + "); retry to enter the next round");
    }

    rank_ = static_cast<int32_t>(slot);
    world_size_ = static_cast<int32_t>(world_size);

    RendezvousResult result;
    result.rank = rank_;
    result.world_size = world_size_;
    result.store_key = prefix;
    return result;
}

auto C10dRendezvous::leave() -> void {
    // Best-effort departure marker so peers can detect the change next round.
    if (rank_ >= 0 && store_) {
        try {
            store_->set(config_.run_id + "/round_" + std::to_string(round_) +
                            "/leave_" + std::to_string(rank_),
                        "1");
        } catch (...) {
            // Departure notification is best-effort; never throw from leave().
        }
    }
    rank_ = -1;
    world_size_ = 0;
}

} // namespace elastic
} // namespace distributed
} // namespace tenzor
