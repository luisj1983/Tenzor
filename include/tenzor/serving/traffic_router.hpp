/**
 * @file traffic_router.hpp
 * @brief A/B testing traffic router for inference serving
 */
#pragma once

#include <string>
#include <unordered_map>
#include <mutex>
#include <random>
#include <atomic>
#include <stdexcept>
#include <cmath>
#include <vector>
#include <memory>
#include <utility>
#include <cstdint>

namespace tenzor::serving {

struct TrafficRule {
    std::string model_a;      ///< Primary model name
    std::string model_b;      ///< Variant model name
    double fraction_b{0.1};   ///< Fraction of traffic routed to model_b [0.0, 1.0]
};

struct ExperimentMetrics {
    std::atomic<uint64_t> requests_a{0};
    std::atomic<uint64_t> requests_b{0};
};

class TrafficRouter {
public:
    /// Create or update an A/B experiment
    /// @throws std::invalid_argument if rule.fraction_b is NaN or outside [0.0, 1.0]
    auto set_experiment(const std::string& experiment_name, TrafficRule rule) -> void {
        if (std::isnan(rule.fraction_b) || rule.fraction_b < 0.0 || rule.fraction_b > 1.0) {
            throw std::invalid_argument(
                "TrafficRouter::set_experiment: fraction_b must be in [0.0, 1.0]");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        experiments_[experiment_name] = std::move(rule);
        if (metrics_.find(experiment_name) == metrics_.end()) {
            metrics_[experiment_name] = std::make_unique<ExperimentMetrics>();
        }
    }

    /// Remove an experiment
    auto remove_experiment(const std::string& experiment_name) -> void {
        std::lock_guard<std::mutex> lock(mutex_);
        experiments_.erase(experiment_name);
    }

    /// Select which model to use for a given experiment
    /// Returns the selected model name, or empty string if no experiment found
    auto select_model(const std::string& experiment_name) -> std::string {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = experiments_.find(experiment_name);
        if (it == experiments_.end()) return {};

        const auto& rule = it->second;
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        bool use_b = dist(rng_) < rule.fraction_b;

        auto& m = *metrics_[experiment_name];
        if (use_b) {
            m.requests_b.fetch_add(1, std::memory_order_relaxed);
            return rule.model_b;
        } else {
            m.requests_a.fetch_add(1, std::memory_order_relaxed);
            return rule.model_a;
        }
    }

    /// List all active experiments
    auto list_experiments() const -> std::vector<std::string> {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::string> names;
        names.reserve(experiments_.size());
        for (const auto& [name, _] : experiments_) {
            names.push_back(name);
        }
        return names;
    }

    /// Get experiment metrics
    auto get_metrics(const std::string& name) const -> std::pair<uint64_t, uint64_t> {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = metrics_.find(name);
        if (it == metrics_.end()) return {0, 0};
        return {it->second->requests_a.load(), it->second->requests_b.load()};
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, TrafficRule> experiments_;
    std::unordered_map<std::string, std::unique_ptr<ExperimentMetrics>> metrics_;
    std::mt19937 rng_{std::random_device{}()};
};

} // namespace tenzor::serving
