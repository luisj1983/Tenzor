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
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace tenzor {
namespace distributed {

// getpid() has no direct MSVC equivalent (Windows uses GetCurrentProcessId,
// which returns DWORD rather than pid_t).
#ifdef _WIN32
inline unsigned long tenzor_getpid() { return GetCurrentProcessId(); }
#else
inline pid_t tenzor_getpid() { return ::getpid(); }
#endif
namespace elastic {

namespace {
constexpr auto kPollInterval = std::chrono::milliseconds(25);
constexpr auto kGraceWindow  = std::chrono::milliseconds(250);

// Shared key holding the round number that *all* workers — survivors and
// freshly-spawned replacements alike — should currently attempt. Stored once
// in the run's namespace (not per-round) so it survives across rounds.
auto current_round_key(const std::string& run_id) -> std::string {
    return run_id + "/current_round";
}

// Per-round single-fire guard used to advance current_round exactly once even
// when many workers concurrently discover the same round has closed.
auto advance_guard_key(const std::string& run_id, int64_t round) -> std::string {
    return run_id + "/round_" + std::to_string(round) + "/advance_guard";
}

// Read the shared current round from the store (missing/unparseable => 0).
auto read_current_round(RendezvousStore& store, const std::string& run_id)
    -> int64_t {
    std::string v;
    try {
        v = store.get(current_round_key(run_id));
    } catch (...) {
        v.clear();
    }
    if (v.empty()) return 0;
    try {
        const int64_t r = std::stoll(v);
        return r < 0 ? 0 : r;
    } catch (...) {
        return 0;
    }
}

// Atomically advance the shared current round past `closed_round`, exactly once
// across all workers that observed the same closed round. The per-round guard's
// atomic add serialises the winner; losers (guard != 1) leave current_round
// untouched. Returns the round the caller should next attempt.
auto advance_current_round(RendezvousStore& store, const std::string& run_id,
                           int64_t closed_round) -> int64_t {
    const int64_t next = closed_round + 1;
    // First caller to bump the guard from 0 -> 1 publishes the next round.
    if (store.add(advance_guard_key(run_id, closed_round), 1) == 1) {
        // Only ever move forward; never clobber a higher round already
        // published by a later, concurrently-closing round.
        if (read_current_round(store, run_id) < next) {
            store.set(current_round_key(run_id), std::to_string(next));
        }
    }
    // Whether we won or lost the guard, the published current round is now at
    // least `next`; report what is actually in the store so all workers agree.
    return std::max<int64_t>(next, read_current_round(store, run_id));
}
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
    return std::string(host) + ":" + std::to_string(static_cast<long long>(tenzor_getpid()));
}

auto C10dRendezvous::join() -> RendezvousResult {
    if (config_.min_workers < 1 || config_.max_workers < config_.min_workers) {
        throw std::runtime_error(
            "C10dRendezvous::join: invalid config (need 1 <= min_workers <= max_workers)");
    }

    RendezvousStore& store = ensure_store();

    // Discover the active round from the shared store rather than a per-process
    // counter. Survivors (which leave()+join() after a membership change) and
    // freshly-spawned replacement workers must agree on the same round number,
    // otherwise they write/read disjoint key prefixes and can never assemble
    // into the same rendezvous. The coordinator advances current_round once a
    // round closes (see below), so everyone converges on the live round.
    round_ = read_current_round(store, config_.run_id);
    std::string prefix =
        config_.run_id + "/round_" + std::to_string(round_);
    std::string counter_key = prefix + "/slot_counter";
    std::string world_size_key = prefix + "/world_size";

    // Claim a unique slot for this round (0-based). Slot 0 is the coordinator.
    int64_t slot = store.add(counter_key, 1) - 1;
    std::string member_key = prefix + "/member/" + std::to_string(slot);
    store.set(member_key, worker_id());

    // Fast path for survivors rejoining a round that has already closed: if the
    // coordinator has already frozen world_size for this round, a slot claimed
    // now necessarily lands past it. Don't sit through the barrier/grace window
    // — drop the just-claimed (orphan) member key, advance the shared round, and
    // signal the caller to retry into the next round.
    if (store.check_key(world_size_key)) {
        try { store.delete_key(member_key); } catch (...) { /* best-effort */ }
        round_ = advance_current_round(store, config_.run_id, round_);
        throw std::runtime_error(
            "C10dRendezvous::join: round " + std::to_string(round_ - 1) +
            " already closed before slot " + std::to_string(slot) +
            " could join; retry to enter round " + std::to_string(round_));
    }

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
            // Round failed to assemble. Advance the shared round and drop this
            // worker's member key so a caller-driven retry enters a clean round
            // rather than re-claiming slots in the stuck one (which would
            // accumulate phantom counter increments and stale member keys).
            try { store.delete_key(member_key); } catch (...) { /* best-effort */ }
            const int64_t stuck_round = round_;
            round_ = advance_current_round(store, config_.run_id, stuck_round);
            throw std::runtime_error(
                "C10dRendezvous::join: timed out after " +
                std::to_string(config_.timeout.count()) + "s with only " +
                std::to_string(count) + " of min_workers=" +
                std::to_string(config_.min_workers) + " joined (run_id='" +
                config_.run_id + "', round " + std::to_string(stuck_round) +
                "); retry to enter round " + std::to_string(round_));
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
    // Give the world_size read its OWN bounded grace window. The barrier
    // `deadline` above may already be at/past now() (the barrier can break
    // exactly at the deadline once min_workers is met), so reusing it here would
    // spuriously fail before the coordinator has a chance to publish. On timeout
    // we must also drop the just-claimed member slot and advance the shared
    // round — mirroring the sibling failure paths — so a leaked orphan member
    // key and a stuck round don't accumulate across retries.
    const auto ws_deadline = std::chrono::steady_clock::now() + config_.timeout;
    int64_t world_size = 0;
    for (;;) {
        std::string ws;
        try { ws = store.get(world_size_key); } catch (...) { ws.clear(); }
        if (!ws.empty()) {
            try { world_size = std::stoll(ws); break; } catch (...) { /* retry */ }
        }
        if (std::chrono::steady_clock::now() >= ws_deadline) {
            try { store.delete_key(member_key); } catch (...) { /* best-effort */ }
            const int64_t stuck_round = round_;
            round_ = advance_current_round(store, config_.run_id, stuck_round);
            throw std::runtime_error(
                "C10dRendezvous::join: coordinator failed to publish world_size "
                "for run_id='" + config_.run_id + "' round " +
                std::to_string(stuck_round) + "; retry to enter round " +
                std::to_string(round_));
        }
        std::this_thread::sleep_for(kPollInterval);
    }

    // A worker whose slot landed at or beyond the frozen world_size joined after
    // the round closed; tell it to retry. Advance the shared round so the next
    // join() (from this caller and any concurrent late joiner) converges on the
    // same fresh round, and best-effort drop the orphaned member key so rejected
    // entries don't accumulate in the store across retries.
    if (slot >= world_size) {
        try { store.delete_key(member_key); } catch (...) { /* best-effort */ }
        const int64_t closed_round = round_;
        round_ = advance_current_round(store, config_.run_id, closed_round);
        throw std::runtime_error(
            "C10dRendezvous::join: joined round " + std::to_string(closed_round) +
            " after it closed (slot " + std::to_string(slot) + " >= world_size " +
            std::to_string(world_size) + "); retry to enter round " +
            std::to_string(round_));
    }

    rank_ = static_cast<int32_t>(slot);
    world_size_ = static_cast<int32_t>(world_size);
    // Remember the round we actually joined so leave() marks departure under it,
    // even if a later join() advances round_ past this value.
    joined_round_ = round_;

    RendezvousResult result;
    result.rank = rank_;
    result.world_size = world_size_;
    result.store_key = prefix;
    return result;
}

auto C10dRendezvous::leave() -> void {
    // Best-effort departure marker so peers can detect the change next round.
    // Key it off the round this worker actually joined (joined_round_), NOT the
    // current round_ — a failed re-join may have advanced round_ to a round this
    // worker was never a member of, and writing the marker there would mislead
    // that round's peers while leaving the real round's marker unset.
    if (rank_ >= 0 && store_ && joined_round_ >= 0) {
        try {
            store_->set(config_.run_id + "/round_" +
                            std::to_string(joined_round_) +
                            "/leave_" + std::to_string(rank_),
                        "1");
        } catch (...) {
            // Departure notification is best-effort; never throw from leave().
        }
    }
    rank_ = -1;
    world_size_ = 0;
    joined_round_ = -1;
}

} // namespace elastic
} // namespace distributed
} // namespace tenzor
