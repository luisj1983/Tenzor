/**
 * @file server.cpp
 * @brief Implementation of inference serving infrastructure
 */

#include "tenzor/serving/server.hpp"
#include "tenzor/jit/serialization.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/core/dtype.hpp"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <chrono>
#include <unordered_map>
#include <mutex>
#ifdef TENZOR_HAS_HTTPLIB
#include <httplib.h>
#endif

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
        if (batch.size() == 1) {
            // Single request: no batching overhead
            tenzor::Variable input_var(batch[0]->input, false);
            auto output = model_->forward(input_var);
            batch[0]->result.set_value(output.tensor());
        } else {
            // Batch multiple requests: concatenate along dim 0, single forward, split
            std::vector<Tensor> inputs;
            inputs.reserve(batch.size());
            for (auto& req : batch) {
                inputs.push_back(req->input);
            }

            // Concatenate all inputs along batch dimension
            auto batched_input = tenzor::cat(inputs, 0);

            // Single forward pass on the full batch
            tenzor::Variable batched_var(batched_input, false);
            auto batched_output = model_->forward(batched_var);

            // Split output back into individual results
            auto split_outputs = tenzor::split(batched_output.tensor(),
                                               /*split_size=*/1, /*dim=*/0);
            for (size_t i = 0; i < batch.size(); ++i) {
                if (i < split_outputs.size()) {
                    batch[i]->result.set_value(split_outputs[i]);
                } else {
                    batch[i]->result.set_exception(
                        std::make_exception_ptr(std::runtime_error(
                            "Batch output split mismatch")));
                }
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

        // Compute latency percentiles from ring buffer
        if (total > 0) {
            auto count = std::min(total, static_cast<uint64_t>(ModelMetrics::kLatencyWindowSize));
            std::vector<uint64_t> latencies(count);
            for (uint64_t i = 0; i < count; ++i) {
                latencies[i] = m->latency_window[i];
            }
            std::sort(latencies.begin(), latencies.end());
            auto p50 = latencies[static_cast<size_t>(count * 0.50)];
            auto p95 = latencies[static_cast<size_t>(count * 0.95)];
            auto p99 = latencies[static_cast<size_t>(std::min(count - 1, static_cast<uint64_t>(count * 0.99)))];
            ss << "tenzor_latency_p50_us{model=\"" << name << "\"} " << p50 << "\n";
            ss << "tenzor_latency_p95_us{model=\"" << name << "\"} " << p95 << "\n";
            ss << "tenzor_latency_p99_us{model=\"" << name << "\"} " << p99 << "\n";
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
#ifdef TENZOR_HAS_HTTPLIB
    httplib::Server svr;

    // ---------- Authentication middleware ----------
    if (config_.enable_auth && !config_.api_keys.empty()) {
        svr.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) -> httplib::Server::HandlerResponse {
            // Skip auth for health and metrics endpoints
            if (req.path == "/health" || req.path == "/metrics") {
                return httplib::Server::HandlerResponse::Unhandled;
            }
            auto it = req.headers.find(config_.auth_header);
            if (it == req.headers.end()) {
                res.status = 401;
                res.set_content(R"({"error":"missing authentication header"})", "application/json");
                return httplib::Server::HandlerResponse::Handled;
            }
            // Extract Bearer token
            std::string token = it->second;
            if (token.substr(0, 7) == "Bearer ") {
                token = token.substr(7);
            }
            bool valid = false;
            for (const auto& key : config_.api_keys) {
                if (token == key) { valid = true; break; }
            }
            if (!valid) {
                res.status = 403;
                res.set_content(R"({"error":"invalid api key"})", "application/json");
                auto& metrics = MetricsRegistry::instance().get_metrics("_auth");
                metrics.error_count.fetch_add(1, std::memory_order_relaxed);
                return httplib::Server::HandlerResponse::Handled;
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });
    }

    // ---------- Rate limiting (token bucket per client IP) ----------
    struct TokenBucket {
        double tokens;
        std::chrono::steady_clock::time_point last_refill;
    };
    std::mutex rate_limit_mutex;
    std::unordered_map<std::string, TokenBucket> rate_limit_buckets;

    // Health check
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok","version":"0.1.0"})", "application/json");
    });

    // Prometheus metrics
    svr.Get("/metrics", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(MetricsRegistry::instance().format_prometheus(), "text/plain");
    });

    // List models
    svr.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
        auto models = repository_.list_models();
        std::ostringstream oss;
        oss << R"({"models":[)";
        for (size_t i = 0; i < models.size(); ++i) {
            if (i > 0) oss << ",";
            oss << "\"" << models[i] << "\"";
        }
        oss << "]}";
        res.set_content(oss.str(), "application/json");
    });

    // Model status
    svr.Get(R"(/v1/models/([^/]+)/status)", [this](const httplib::Request& req, httplib::Response& res) {
        auto name = req.matches[1].str();
        auto model = repository_.get_model(name);
        if (!model) {
            res.status = 404;
            res.set_content(R"({"error":"model not found"})", "application/json");
            return;
        }
        auto state = model->state.load();
        const char* state_str = "UNKNOWN";
        switch (state) {
            case ModelState::LOADING: state_str = "LOADING"; break;
            case ModelState::READY: state_str = "READY"; break;
            case ModelState::UNLOADING: state_str = "UNLOADING"; break;
            case ModelState::FAILED: state_str = "FAILED"; break;
        }
        std::ostringstream oss;
        oss << R"({"model_name":")" << name
            << R"(","version":)" << model->version
            << R"(,"status":")" << state_str << "\"}";
        res.set_content(oss.str(), "application/json");
    });

    // Load model
    svr.Post(R"(/v1/models/([^/]+)/load)", [this](const httplib::Request& req, httplib::Response& res) {
        auto name = req.matches[1].str();
        // Expect JSON body: {"model_path": "/path/to/model"}
        auto path_pos = req.body.find("\"model_path\"");
        if (path_pos == std::string::npos) {
            res.status = 400;
            res.set_content(R"({"error":"missing model_path"})", "application/json");
            return;
        }
        // Simple JSON value extraction
        auto colon_pos = req.body.find(':', path_pos);
        auto quote1 = req.body.find('"', colon_pos);
        auto quote2 = req.body.find('"', quote1 + 1);
        if (quote1 == std::string::npos || quote2 == std::string::npos) {
            res.status = 400;
            res.set_content(R"({"error":"invalid model_path"})", "application/json");
            return;
        }
        auto model_path = req.body.substr(quote1 + 1, quote2 - quote1 - 1);

        // Parse optional "device" field (defaults to "cpu")
        Device target_device = Device::cpu();
        auto device_pos = req.body.find("\"device\"");
        if (device_pos != std::string::npos) {
            auto dev_colon = req.body.find(':', device_pos);
            auto dev_q1 = req.body.find('"', dev_colon);
            auto dev_q2 = req.body.find('"', dev_q1 + 1);
            if (dev_q1 != std::string::npos && dev_q2 != std::string::npos) {
                auto device_str = req.body.substr(dev_q1 + 1, dev_q2 - dev_q1 - 1);
                target_device = Device::from_string(device_str);
            }
        }

        try {
            repository_.load_model(name, model_path, target_device);
            res.set_content(R"({"success":true})", "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            std::ostringstream oss;
            oss << R"({"success":false,"message":")" << e.what() << "\"}";
            res.set_content(oss.str(), "application/json");
        }
    });

    // Unload model
    svr.Delete(R"(/v1/models/([^/]+))", [this](const httplib::Request& req, httplib::Response& res) {
        auto name = req.matches[1].str();
        try {
            repository_.unload_model(name);
            res.set_content(R"({"success":true})", "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            std::ostringstream oss;
            oss << R"({"success":false,"message":")" << e.what() << "\"}";
            res.set_content(oss.str(), "application/json");
        }
    });

    // Predict (simplified: accepts raw float data, returns raw float data)
    svr.Post(R"(/v1/models/([^/]+)/predict)", [this, &rate_limit_mutex, &rate_limit_buckets](const httplib::Request& req, httplib::Response& res) {
        // Rate limiting check
        if (config_.enable_rate_limit) {
            auto client_ip = req.remote_addr;
            std::lock_guard<std::mutex> lock(rate_limit_mutex);
            auto now = std::chrono::steady_clock::now();
            auto& bucket = rate_limit_buckets[client_ip];
            if (bucket.tokens == 0 && bucket.last_refill == std::chrono::steady_clock::time_point{}) {
                // Initialize new bucket
                bucket.tokens = static_cast<double>(config_.rate_limit_burst);
                bucket.last_refill = now;
            }
            // Refill tokens
            auto elapsed = std::chrono::duration<double>(now - bucket.last_refill).count();
            bucket.tokens = std::min(
                static_cast<double>(config_.rate_limit_burst),
                bucket.tokens + elapsed * config_.rate_limit_rps);
            bucket.last_refill = now;
            if (bucket.tokens < 1.0) {
                res.status = 429;
                res.set_content(R"({"error":"rate limit exceeded"})", "application/json");
                return;
            }
            bucket.tokens -= 1.0;
        }

        auto name = req.matches[1].str();
        auto model = repository_.get_model(name);
        if (!model || model->state.load() != ModelState::READY) {
            res.status = 404;
            res.set_content(R"({"error":"model not ready"})", "application/json");
            return;
        }

        auto start = std::chrono::high_resolution_clock::now();

        try {
            // Parse binary tensor data from body
            // Format: [4 bytes: ndim][ndim * 8 bytes: shape][rest: float32 data]
            if (req.body.size() < 4) {
                res.status = 400;
                res.set_content(R"({"error":"body too short"})", "application/json");
                return;
            }

            const char* body = req.body.data();
            int32_t ndim;
            std::memcpy(&ndim, body, 4);
            body += 4;

            if (req.body.size() < static_cast<size_t>(4 + ndim * 8)) {
                res.status = 400;
                res.set_content(R"({"error":"body too short for shape"})", "application/json");
                return;
            }

            std::vector<int64_t> shape(ndim);
            std::memcpy(shape.data(), body, ndim * 8);
            body += ndim * 8;

            size_t data_bytes = req.body.size() - 4 - ndim * 8;
            auto input = tenzor::from_data(
                reinterpret_cast<const float*>(body),
                shape, model->device
            );

            // Submit to batcher and wait for result
            auto future = model->batcher->submit(input);
            auto output = future.get();

            auto end = std::chrono::high_resolution_clock::now();
            auto latency_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            // Update metrics
            auto& metrics = MetricsRegistry::instance().get_metrics(name);
            metrics.total_requests.fetch_add(1);
            metrics.total_latency_us.fetch_add(latency_us);
            metrics.record_latency(latency_us);

            // Return binary tensor data
            auto output_cont = output.contiguous();
            size_t out_data_bytes = output_cont.numel() * dtype_size(output_cont.dtype());
            auto out_shape = output_cont.shape();
            int32_t out_ndim = static_cast<int32_t>(out_shape.size());

            std::string response;
            response.resize(4 + out_ndim * 8 + out_data_bytes);
            char* rp = response.data();
            std::memcpy(rp, &out_ndim, 4); rp += 4;
            std::memcpy(rp, out_shape.data(), out_ndim * 8); rp += out_ndim * 8;
            std::memcpy(rp, output_cont.data<float>(), out_data_bytes);

            res.set_content(response, "application/octet-stream");
        } catch (const std::exception& e) {
            auto& metrics = MetricsRegistry::instance().get_metrics(name);
            metrics.error_count.fetch_add(1);
            res.status = 500;
            std::ostringstream oss;
            oss << R"({"error":")" << e.what() << "\"}";
            res.set_content(oss.str(), "application/json");
        }
    });

    // A/B experiment management
    svr.Post("/v1/experiments", [this](const httplib::Request& req, httplib::Response& res) {
        // Parse: {"name":"exp1","model_a":"v1","model_b":"v2","fraction_b":0.1}
        auto extract = [&](const std::string& key) -> std::string {
            auto pos = req.body.find("\"" + key + "\"");
            if (pos == std::string::npos) return {};
            auto colon = req.body.find(':', pos);
            auto q1 = req.body.find('"', colon);
            auto q2 = req.body.find('"', q1 + 1);
            if (q1 == std::string::npos || q2 == std::string::npos) return {};
            return req.body.substr(q1 + 1, q2 - q1 - 1);
        };
        auto name = extract("name");
        auto model_a = extract("model_a");
        auto model_b = extract("model_b");
        if (name.empty() || model_a.empty() || model_b.empty()) {
            res.status = 400;
            res.set_content(R"({"error":"missing name, model_a, or model_b"})", "application/json");
            return;
        }
        double fraction_b = 0.1;
        auto frac_pos = req.body.find("\"fraction_b\"");
        if (frac_pos != std::string::npos) {
            auto colon = req.body.find(':', frac_pos);
            fraction_b = std::stod(req.body.substr(colon + 1));
        }
        traffic_router_.set_experiment(name, {model_a, model_b, fraction_b});
        res.set_content(R"({"success":true})", "application/json");
    });

    svr.Get("/v1/experiments", [this](const httplib::Request&, httplib::Response& res) {
        auto experiments = traffic_router_.list_experiments();
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < experiments.size(); ++i) {
            if (i > 0) oss << ",";
            auto [a_count, b_count] = traffic_router_.get_metrics(experiments[i]);
            oss << R"({"name":")" << experiments[i]
                << R"(","requests_a":)" << a_count
                << R"(,"requests_b":)" << b_count << "}";
        }
        oss << "]";
        res.set_content(oss.str(), "application/json");
    });

    svr.Delete(R"(/v1/experiments/([^/]+))", [this](const httplib::Request& req, httplib::Response& res) {
        auto name = req.matches[1].str();
        traffic_router_.remove_experiment(name);
        res.set_content(R"({"success":true})", "application/json");
    });

    std::cout << "[TenzorServing] HTTP server listening on port " << config_.http_port << std::endl;
    svr.listen("0.0.0.0", config_.http_port);
#else
    // Fallback: no HTTP library, just spin
    while (running_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
#endif
}

} // namespace serving
} // namespace tenzor
