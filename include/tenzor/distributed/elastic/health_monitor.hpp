/**
 * @file health_monitor.hpp
 * @brief Heartbeat-based worker health monitoring
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tenzor {
namespace distributed {

namespace rpc { class TcpRpcAgent; }

namespace elastic {

/**
 * @brief Worker health state.
 */
enum class WorkerState : uint8_t {
    ALIVE,       ///< Responding to heartbeats
    SUSPECTED,   ///< Missed 1 heartbeat
    DEAD,        ///< Missed 3+ consecutive heartbeats
    UNKNOWN,     ///< Worker id is not (or no longer) being monitored
};

/**
 * @brief Callback for worker state changes.
 */
using HealthCallback = std::function<void(int32_t worker_id, WorkerState new_state)>;

/**
 * @brief Configuration for health monitoring.
 */
struct HealthMonitorConfig {
    std::chrono::milliseconds heartbeat_interval{5000};
    int32_t suspect_threshold{1};      ///< Misses before SUSPECTED
    int32_t dead_threshold{3};         ///< Misses before DEAD
    /**
     * @brief Re-probe DEAD workers once every N cycles (reduced cadence).
     *
     * DEAD is not terminal: a transiently-partitioned worker that recovers
     * can transition back to ALIVE. Set to 0 to disable DEAD re-probing
     * (DEAD becomes terminal until start() is called again).
     */
    int32_t dead_recheck_interval{6};
};

/**
 * @brief Monitors worker health via periodic heartbeats.
 *
 * Sends HEARTBEAT messages via the RPC agent and tracks responses.
 * Workers that miss consecutive heartbeats are marked SUSPECTED
 * then DEAD, triggering registered callbacks.
 */
class HealthMonitor {
public:
    HealthMonitor(std::shared_ptr<rpc::TcpRpcAgent> agent,
                  HealthMonitorConfig config = {});
    ~HealthMonitor();

    /**
     * @brief Start the health monitoring loop.
     *
     * @param worker_ids IDs of workers to monitor
     */
    auto start(const std::vector<int32_t>& worker_ids) -> void;

    /**
     * @brief Stop the monitoring loop.
     */
    auto stop() -> void;

    /**
     * @brief Register a callback for state changes.
     */
    auto on_state_change(HealthCallback callback) -> void;

    /**
     * @brief Get current state of a worker.
     *
     * Returns WorkerState::UNKNOWN for an id that is not being monitored
     * (never registered via start(), out of range, or a typo), so callers
     * can distinguish a never-registered worker from a genuinely DEAD one.
     * This keeps the contract consistent with dead_workers(), which only
     * reports ids present in the monitored set with state == DEAD.
     */
    auto get_state(int32_t worker_id) const -> WorkerState;

    /**
     * @brief Get IDs of all dead workers.
     */
    auto dead_workers() const -> std::vector<int32_t>;

private:
    std::shared_ptr<rpc::TcpRpcAgent> agent_;
    HealthMonitorConfig config_;
    std::atomic<bool> running_{false};
    std::thread monitor_thread_;

    mutable std::mutex state_mutex_;
    std::unordered_map<int32_t, WorkerState> states_;
    std::unordered_map<int32_t, int32_t> miss_counts_;
    std::vector<HealthCallback> callbacks_;

    auto monitor_loop() -> void;
    /**
     * @brief Record a state transition under state_mutex_.
     *
     * The caller must hold state_mutex_. This updates states_ and appends the
     * (worker_id, new_state) change to @p changes; callbacks are invoked by
     * the caller *after* releasing the lock (see monitor_loop) so a callback
     * that re-enters HealthMonitor cannot deadlock on the non-recursive
     * state_mutex_.
     */
    auto update_state(int32_t worker_id, WorkerState new_state,
                      std::vector<std::pair<int32_t, WorkerState>>& changes) -> void;
};

} // namespace elastic
} // namespace distributed
} // namespace tenzor
