/**
 * @file server.cpp
 * @brief Implementation of inference serving infrastructure
 */

#include "tenzor/serving/server.hpp"
#include "tenzor/jit/serialization.hpp"
#include "tenzor/autograd/variable.hpp"
#include <algorithm>
#include <iostream>
#include <sstream>

namespace tenzor {
namespace serving {

// ============================================================================
// DynamicBatcher
// ============================================================================

DynamicBatcher::DynamicBatcher(std::shared_ptr<jit::CompiledModule> model,
                               BatchConfig config)
    : model_(std::move(model)), config_(std::move(config)) {}

DynamicBatcher::~DynamicBatcher() {
    stop();
}

auto DynamicBatcher::submit(Tensor input) -> std::future<Tensor> {
    auto req = std::make_shared<InferRequest>(std::move(input));
    auto future = req->result.get_future();

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        queue_.push(std::move(req));
    }
    queue_cv_.notify_one();

    return future;
}

auto DynamicBatcher::start() -> void {
    running_.store(true, std::memory_order_release);
    batch_thread_ = std::thread([this] { batch_loop(); });
}

auto DynamicBatcher::stop() -> void {
    running_.store(false, std::memory_order_release);
    queue_cv_.notify_all();
    if (batch_thread_.joinable()) {
        batch_thread_.join();
    }
}

auto DynamicBatcher::batch_loop() -> void {
    while (running_.load(std::memory_order_acquire)) {
        std::vector<std::shared_ptr<InferRequest>> batch;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);

            // Wait for either batch full or timeout
            queue_cv_.wait_for(lock,
                std::chrono::microseconds(config_.max_latency_us),
                [this] {
                    return !queue_.empty() || !running_.load(std::memory_order_acquire);
                });

            if (!running_.load(std::memory_order_acquire) && queue_.empty()) {
                break;
            }

            // Drain up to max_batch_size requests
            while (!queue_.empty() &&
                   static_cast<int32_t>(batch.size()) < config_.max_batch_size) {
                batch.push_back(std::move(queue_.front()));
                queue_.pop();
            }
        }

        if (!batch.empty()) {
            execute_batch(batch);
        }
    }
}

auto DynamicBatcher::execute_batch(
    std::vector<std::shared_ptr<InferRequest>>& batch) -> void {
    try {
        // Concatenate inputs along batch dimension (dim=0)
        // For simplicity, execute individually if batch size is 1
        // In production: cat(inputs, 0) -> forward -> split
        for (auto& req : batch) {
            try {
                tenzor::Variable input_var(req->input, false);
                auto output = model_->forward(input_var);
                req->result.set_value(output.tensor());
            } catch (const std::exception& e) {
                req->result.set_exception(std::current_exception());
            }
        }
    } catch (const std::exception& e) {
        // If batch execution fails, propagate error to all requests
        for (auto& req : batch) {
            try {
                req->result.set_exception(std::current_exception());
            } catch (...) {}
        }
    }
}

// ============================================================================
// ModelRepository
// ============================================================================

auto ModelRepository::load_model(const std::string& name, const std::string& path,
                                  Device device, BatchConfig batch_config) -> void {
    auto entry = std::make_shared<ModelEntry>();
    entry->name = name;
    entry->device = device;
    entry->state.store(ModelState::LOADING, std::memory_order_release);

    try {
        // Load the compiled module
        auto graph = jit::load_graph(path);
        auto module = std::make_shared<jit::CompiledModule>();
        module->set_graph(graph);
        module->optimize_for_inference();

        entry->module = std::move(module);
        entry->batcher = std::make_unique<DynamicBatcher>(entry->module, batch_config);
        entry->batcher->start();
        entry->state.store(ModelState::READY, std::memory_order_release);
    } catch (...) {
        entry->state.store(ModelState::FAILED, std::memory_order_release);
        throw;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    models_[name] = std::move(entry);
}

auto ModelRepository::unload_model(const std::string& name) -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = models_.find(name);
    if (it != models_.end()) {
        it->second->state.store(ModelState::UNLOADING, std::memory_order_release);
        if (it->second->batcher) {
            it->second->batcher->stop();
        }
        models_.erase(it);
    }
}

auto ModelRepository::get_model(const std::string& name) -> std::shared_ptr<ModelEntry> {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = models_.find(name);
    return it != models_.end() ? it->second : nullptr;
}

auto ModelRepository::list_models() const -> std::vector<std::string> {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> names;
    names.reserve(models_.size());
    for (auto& [name, _] : models_) {
        names.push_back(name);
    }
    return names;
}

// ============================================================================
// MetricsRegistry
// ============================================================================

auto MetricsRegistry::instance() -> MetricsRegistry& {
    static MetricsRegistry registry;
    return registry;
}

auto MetricsRegistry::get_metrics(const std::string& model_name) -> ModelMetrics& {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = metrics_.find(model_name);
    if (it == metrics_.end()) {
        metrics_[model_name] = std::make_unique<ModelMetrics>();
        return *metrics_[model_name];
    }
    return *it->second;
}

auto MetricsRegistry::format_prometheus() const -> std::string {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream ss;

    for (auto& [name, m] : metrics_) {
        auto total = m->total_requests.load(std::memory_order_relaxed);
        auto latency = m->total_latency_us.load(std::memory_order_relaxed);
        auto errors = m->error_count.load(std::memory_order_relaxed);
        auto batches = m->total_batch_count.load(std::memory_order_relaxed);

        ss << "tenzor_requests_total{model=\"" << name << "\"} " << total << "\n";
        ss << "tenzor_errors_total{model=\"" << name << "\"} " << errors << "\n";
        if (total > 0) {
            ss << "tenzor_latency_avg_us{model=\"" << name << "\"} "
               << (latency / total) << "\n";
        }
        if (batches > 0) {
            auto batch_sum = m->total_batch_size.load(std::memory_order_relaxed);
            ss << "tenzor_batch_size_avg{model=\"" << name << "\"} "
               << (batch_sum / batches) << "\n";
        }
    }

    return ss.str();
}

// ============================================================================
// InferenceServer
// ============================================================================

InferenceServer::InferenceServer(ServerConfig config)
    : config_(std::move(config)) {}

InferenceServer::~InferenceServer() {
    stop();
}

auto InferenceServer::start() -> void {
    running_.store(true, std::memory_order_release);
    server_thread_ = std::thread([this] { serve_loop(); });
    std::cout << "[TenzorServing] Server started on port " << config_.http_port << std::endl;
}

auto InferenceServer::stop() -> void {
    running_.store(false, std::memory_order_release);
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    std::cout << "[TenzorServing] Server stopped" << std::endl;
}

auto InferenceServer::wait() -> void {
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
}

auto InferenceServer::serve_loop() -> void {
    // In a full implementation, this would:
    // 1. Start cpp-httplib server on config_.http_port
    // 2. Register routes:
    //    - POST /v1/models/{name}/predict -> parse input -> batcher.submit() -> return result
    //    - GET  /v1/models/{name}/status -> model state
    //    - POST /v1/models/{name}/load -> repository.load_model()
    //    - DELETE /v1/models/{name} -> repository.unload_model()
    //    - GET  /health -> {"status": "ok"}
    //    - GET  /metrics -> MetricsRegistry::format_prometheus()
    // 3. Optionally start gRPC server on config_.grpc_port

    while (running_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

} // namespace serving
} // namespace tenzor
