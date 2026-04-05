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
    auto update_state(int32_t worker_id, WorkerState new_state) -> void;
};

} // namespace elastic
} // namespace distributed
} // namespace tenzor
