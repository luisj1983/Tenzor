/**
 * @file health_monitor.cpp
 * @brief Implementation of heartbeat-based health monitoring
 */

#include "tenzor/distributed/elastic/health_monitor.hpp"
#include "tenzor/distributed/rpc/rpc_agent.hpp"
#include "tenzor/distributed/rpc/types.hpp"

#include <future>
#include <memory>
#include <utility>

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
    // Ensure any previously running monitor thread is stopped and joined before
    // launching a new one. std::thread::operator= onto a joinable thread calls
    // std::terminate(), so a second start() without an intervening stop() would
    // otherwise abort the process. stop() is idempotent: on the first start()
    // monitor_thread_ is default-constructed (not joinable) and this is a no-op.
    stop();

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        // Reset monitored set to exactly `worker_ids`. After a membership shrink
        // (e.g. elastic re-rendezvous into a smaller world) ranks are reassigned
        // contiguously, so a stale top rank id would otherwise linger here
        // permanently DEAD and keep being probed/reported by dead_workers().
        states_.clear();
        miss_counts_.clear();
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
    return it != states_.end() ? it->second : WorkerState::UNKNOWN;
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
    int64_t cycle = 0;
    while (running_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(config_.heartbeat_interval);
        if (!running_.load(std::memory_order_acquire)) break;

        ++cycle;
        // DEAD workers are re-probed at a reduced cadence so a transiently
        // partitioned worker that recovers can transition back to ALIVE,
        // rather than being permanently excluded from probing.
        const bool recheck_dead =
            config_.dead_recheck_interval > 0 &&
            (cycle % config_.dead_recheck_interval) == 0;

        std::vector<int32_t> workers_to_check;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            for (auto& [id, state] : states_) {
                if (state != WorkerState::DEAD || recheck_dead) {
                    workers_to_check.push_back(id);
                }
            }
        }

        // Fan out all heartbeats concurrently via send_async so one slow or
        // dead peer does not serialize detection of the others. Cycle time is
        // bounded by ~max(timeout) instead of the sum over dead workers.
        std::vector<std::shared_ptr<std::promise<bool>>> promises;
        std::vector<std::future<bool>> futures;
        promises.reserve(workers_to_check.size());
        futures.reserve(workers_to_check.size());

        for (auto worker_id : workers_to_check) {
            auto prom = std::make_shared<std::promise<bool>>();
            futures.push_back(prom->get_future());
            promises.push_back(prom);

            rpc::Message heartbeat;
            heartbeat.type = rpc::MessageType::HEARTBEAT;
            heartbeat.dst_worker = worker_id;

            try {
                agent_->send_async(
                    std::move(heartbeat),
                    [prom](rpc::Message response) {
                        prom->set_value(
                            response.type == rpc::MessageType::HEARTBEAT_ACK);
                    });
            } catch (...) {
                // Failed to even dispatch the probe: treat as not alive.
                prom->set_value(false);
            }
        }

        // Wait on the probe futures WITHOUT holding state_mutex_. Each future
        // is only fulfilled when send_async's callback fires, which for an
        // unresponsive peer happens only after send() blocks for the full RPC
        // timeout. Blocking here under the lock would serialize concurrent
        // get_state()/dead_workers() readers (including check_and_recover()'s
        // poll) behind the worst-case timeout each cycle.
        std::vector<bool> alive_results(workers_to_check.size(), false);
        for (size_t i = 0; i < workers_to_check.size(); ++i) {
            try {
                alive_results[i] = futures[i].get();
            } catch (...) {
                alive_results[i] = false;
            }
        }

        // Apply the state transitions under the lock; this is now non-blocking.
        std::vector<std::pair<int32_t, WorkerState>> changes;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            for (size_t i = 0; i < workers_to_check.size(); ++i) {
                int32_t worker_id = workers_to_check[i];
                bool alive = alive_results[i];

                if (alive) {
                    miss_counts_[worker_id] = 0;
                    if (states_[worker_id] != WorkerState::ALIVE) {
                        update_state(worker_id, WorkerState::ALIVE, changes);
                    }
                } else {
                    miss_counts_[worker_id]++;
                    int32_t misses = miss_counts_[worker_id];

                    if (misses >= config_.dead_threshold) {
                        if (states_[worker_id] != WorkerState::DEAD) {
                            update_state(worker_id, WorkerState::DEAD, changes);
                        }
                    } else if (misses >= config_.suspect_threshold) {
                        if (states_[worker_id] != WorkerState::SUSPECTED) {
                            update_state(worker_id, WorkerState::SUSPECTED,
                                         changes);
                        }
                    }
                }
            }
        }

        // Invoke callbacks OUTSIDE state_mutex_ so a callback that re-enters
        // HealthMonitor (e.g. get_state()/dead_workers()) cannot deadlock on
        // the non-recursive mutex.
        if (!changes.empty()) {
            std::vector<HealthCallback> cbs;
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                cbs = callbacks_;
            }
            for (auto& [id, st] : changes) {
                for (auto& cb : cbs) {
                    try {
                        cb(id, st);
                    } catch (...) {
                        // Don't let callback exceptions break the monitor.
                    }
                }
            }
        }
    }
}

auto HealthMonitor::update_state(
    int32_t worker_id, WorkerState new_state,
    std::vector<std::pair<int32_t, WorkerState>>& changes) -> void {
    // Caller must hold state_mutex_. Callbacks are deferred and invoked by the
    // caller after releasing the lock (see monitor_loop).
    states_[worker_id] = new_state;
    changes.emplace_back(worker_id, new_state);
}

} // namespace elastic
} // namespace distributed
} // namespace tenzor
