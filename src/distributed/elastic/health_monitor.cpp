/**
 * @file health_monitor.cpp
 * @brief Implementation of heartbeat-based health monitoring
 */

#include "tenzor/distributed/elastic/health_monitor.hpp"
#include "tenzor/distributed/rpc/rpc_agent.hpp"
#include "tenzor/distributed/rpc/types.hpp"

namespace tenzor {
namespace distributed {
namespace elastic {

HealthMonitor::HealthMonitor(std::shared_ptr<rpc::TcpRpcAgent> agent,
                             HealthMonitorConfig config)
    : agent_(std::move(agent)), config_(std::move(config)) {}

HealthMonitor::~HealthMonitor() {
    stop();
}

auto HealthMonitor::start(const std::vector<int32_t>& worker_ids) -> void {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        for (auto id : worker_ids) {
            states_[id] = WorkerState::ALIVE;
            miss_counts_[id] = 0;
        }
    }

    running_.store(true, std::memory_order_release);
    monitor_thread_ = std::thread([this] { monitor_loop(); });
}

auto HealthMonitor::stop() -> void {
    running_.store(false, std::memory_order_release);
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
}

auto HealthMonitor::on_state_change(HealthCallback callback) -> void {
    std::lock_guard<std::mutex> lock(state_mutex_);
    callbacks_.push_back(std::move(callback));
}

auto HealthMonitor::get_state(int32_t worker_id) const -> WorkerState {
    std::lock_guard<std::mutex> lock(state_mutex_);
    auto it = states_.find(worker_id);
    return it != states_.end() ? it->second : WorkerState::DEAD;
}

auto HealthMonitor::dead_workers() const -> std::vector<int32_t> {
    std::lock_guard<std::mutex> lock(state_mutex_);
    std::vector<int32_t> dead;
    for (auto& [id, state] : states_) {
        if (state == WorkerState::DEAD) {
            dead.push_back(id);
        }
    }
    return dead;
}

auto HealthMonitor::monitor_loop() -> void {
    while (running_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(config_.heartbeat_interval);
        if (!running_.load(std::memory_order_acquire)) break;

        std::vector<int32_t> workers_to_check;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            for (auto& [id, state] : states_) {
                if (state != WorkerState::DEAD) {
                    workers_to_check.push_back(id);
                }
            }
        }

        for (auto worker_id : workers_to_check) {
            bool alive = false;

            try {
                rpc::Message heartbeat;
                heartbeat.type = rpc::MessageType::HEARTBEAT;
                heartbeat.dst_worker = worker_id;

                auto response = agent_->send(std::move(heartbeat));
                alive = (response.type == rpc::MessageType::HEARTBEAT_ACK);
            } catch (...) {
                alive = false;
            }

            std::lock_guard<std::mutex> lock(state_mutex_);
            if (alive) {
                miss_counts_[worker_id] = 0;
                if (states_[worker_id] != WorkerState::ALIVE) {
                    update_state(worker_id, WorkerState::ALIVE);
                }
            } else {
                miss_counts_[worker_id]++;
                int32_t misses = miss_counts_[worker_id];

                if (misses >= config_.dead_threshold) {
                    if (states_[worker_id] != WorkerState::DEAD) {
                        update_state(worker_id, WorkerState::DEAD);
                    }
                } else if (misses >= config_.suspect_threshold) {
                    if (states_[worker_id] != WorkerState::SUSPECTED) {
                        update_state(worker_id, WorkerState::SUSPECTED);
                    }
                }
            }
        }
    }
}

auto HealthMonitor::update_state(int32_t worker_id, WorkerState new_state) -> void {
    // Caller must hold state_mutex_
    states_[worker_id] = new_state;
    for (auto& cb : callbacks_) {
        try {
            cb(worker_id, new_state);
        } catch (...) {
            // Don't let callback exceptions break the monitor
        }
    }
}

} // namespace elastic
} // namespace distributed
} // namespace tenzor
