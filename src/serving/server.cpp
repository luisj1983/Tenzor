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
#include <cstring>
#include <iostream>
#include <sstream>
#include <chrono>
#include <unordered_map>
#include <mutex>
#ifdef TENZOR_HAS_HTTPLIB
#include <httplib.h>
#endif

// G.12: vendored nlohmann/json (third_party/nlohmann/json.hpp). Used to
// replace the previous ad-hoc std::string::find('"') request parsing in
// the /v1/models/.../{load,predict} and /v1/experiments endpoints.
#include <nlohmann/json.hpp>

// G.12: explicit build-config gate. When TENZOR_BUILD_SERVING is ON,
// the serving subsystem requires at least one HTTP/gRPC transport. We
// don't fire when serving is OFF because this TU is unconditionally
// compiled into tenzor_core (the InferenceServer class needs to exist
// even in trimmed builds for header / ABI compatibility) — the runtime
// guard in serve_loop() handles the OFF case with a typed exception.
#if defined(TENZOR_BUILD_SERVING) && !defined(TENZOR_HAS_HTTPLIB) && !defined(TENZOR_HAS_GRPC)
static_assert(false,
    "tenzor::serving requires at least one HTTP/gRPC transport when "
    "TENZOR_BUILD_SERVING=ON. The httplib FetchContent should run "
    "automatically; if it didn't, check src/CMakeLists.txt.");
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

auto InferenceServer::wait_for(std::chrono::milliseconds timeout) -> bool {
    // Audit-11 QQ.19: bounded wait. We do not have a condition_variable on
    // running_ in this path, so poll on the atomic. Resolution is bounded by
    // a 10 ms tick — sufficient for KeyboardInterrupt latency, well below
    // any realistic server-shutdown deadline.
    using clock = std::chrono::steady_clock;
    const auto deadline = clock::now() + timeout;
    const auto tick = std::chrono::milliseconds(10);
    while (running_.load(std::memory_order_acquire)) {
        if (clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(tick);
    }
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    return true;
}

auto InferenceServer::serve_loop() -> void {
#ifdef TENZOR_HAS_HTTPLIB
    using nlohmann::json;
    httplib::Server svr;

    // ---------- Helpers (JSON dtype <-> Tensor) ----------
    auto dtype_from_request = [](const std::string& s) -> DType {
        // Accept the canonical names emitted by dtype_name() so a client
        // round-trips an output_dtype string back into a predict request.
        if (s == "float32" || s == "f32") return DType::Float32;
        if (s == "float64" || s == "f64") return DType::Float64;
        if (s == "float16" || s == "f16" || s == "half") return DType::Float16;
        if (s == "bfloat16" || s == "bf16") return DType::BFloat16;
        if (s == "int32"   || s == "i32") return DType::Int32;
        if (s == "int64"   || s == "i64") return DType::Int64;
        if (s == "int8"    || s == "i8")  return DType::Int8;
        if (s == "uint8"   || s == "u8")  return DType::UInt8;
        if (s == "bool")                  return DType::Bool;
        throw std::runtime_error("unsupported dtype in request: " + s);
    };

    // Build a host-side Tensor of the given dtype/shape from a JSON array.
    // Numeric arrays are accepted as plain JSON numbers; the request always
    // travels in textual JSON (binary payloads use the gRPC path).
    auto tensor_from_json = [&](const json& shape_j, const json& data_j,
                                DType dtype, Device device) -> Tensor {
        if (!shape_j.is_array()) {
            throw std::runtime_error("\"shape\" must be a JSON array");
        }
        if (!data_j.is_array()) {
            throw std::runtime_error("\"data\" must be a JSON array");
        }
        std::vector<int64_t> shape;
        shape.reserve(shape_j.size());
        int64_t expected = 1;
        for (const auto& d : shape_j) {
            auto v = d.get<int64_t>();
            shape.push_back(v);
            expected *= (v < 0 ? 0 : v);
        }
        if (static_cast<int64_t>(data_j.size()) != expected) {
            throw std::runtime_error("data length does not match shape product");
        }

        // Allocate on host first, then move to target device via .to() if
        // needed. Avoids touching GPU memory from host JSON deserialization
        // code paths.
        auto host = tenzor::empty(shape, dtype, Device::cpu());

        auto write_typed = [&](auto* out_ptr, auto convert) {
            using T = std::remove_pointer_t<decltype(out_ptr)>;
            for (size_t i = 0; i < data_j.size(); ++i) {
                out_ptr[i] = static_cast<T>(convert(data_j[i]));
            }
        };

        switch (dtype) {
            case DType::Float32:
                write_typed(host.data<float>(),
                            [](const json& v){ return v.get<double>(); });
                break;
            case DType::Float64:
                write_typed(host.data<double>(),
                            [](const json& v){ return v.get<double>(); });
                break;
            case DType::Int32:
                write_typed(host.data<int32_t>(),
                            [](const json& v){ return v.get<int64_t>(); });
                break;
            case DType::Int64:
                write_typed(host.data<int64_t>(),
                            [](const json& v){ return v.get<int64_t>(); });
                break;
            case DType::Int8:
                write_typed(host.data<int8_t>(),
                            [](const json& v){ return v.get<int64_t>(); });
                break;
            case DType::UInt8:
                write_typed(host.data<uint8_t>(),
                            [](const json& v){ return v.get<int64_t>(); });
                break;
            case DType::Bool:
                write_typed(host.data<bool>(),
                            [](const json& v){ return v.get<bool>(); });
                break;
            case DType::Float16: {
                auto* out = host.data<Float16>();
                for (size_t i = 0; i < data_j.size(); ++i) {
                    out[i] = Float16(static_cast<float>(data_j[i].get<double>()));
                }
                break;
            }
            case DType::BFloat16: {
                auto* out = host.data<BFloat16>();
                for (size_t i = 0; i < data_j.size(); ++i) {
                    out[i] = BFloat16(static_cast<float>(data_j[i].get<double>()));
                }
                break;
            }
            default:
                throw std::runtime_error(
                    std::string("dtype not supported by JSON predict path: ") +
                    std::string(dtype_name(dtype)));
        }

        // Move to target device (no-op if already on CPU).
        if (device.type != Device::Type::CPU) {
            return host.to(device);
        }
        return host;
    };

    // Serialize a host Tensor (assumed contiguous) into the JSON output_data
    // array. Mirrors the dtype switch above.
    auto json_from_tensor = [](const Tensor& t) -> json {
        json arr = json::array();
        const auto n = static_cast<size_t>(t.numel());
        switch (t.dtype()) {
            case DType::Float32: {
                const auto* p = t.data<float>();
                for (size_t i = 0; i < n; ++i) arr.push_back(p[i]);
                break;
            }
            case DType::Float64: {
                const auto* p = t.data<double>();
                for (size_t i = 0; i < n; ++i) arr.push_back(p[i]);
                break;
            }
            case DType::Int32: {
                const auto* p = t.data<int32_t>();
                for (size_t i = 0; i < n; ++i) arr.push_back(p[i]);
                break;
            }
            case DType::Int64: {
                const auto* p = t.data<int64_t>();
                for (size_t i = 0; i < n; ++i) arr.push_back(p[i]);
                break;
            }
            case DType::Int8: {
                const auto* p = t.data<int8_t>();
                for (size_t i = 0; i < n; ++i) arr.push_back(static_cast<int>(p[i]));
                break;
            }
            case DType::UInt8: {
                const auto* p = t.data<uint8_t>();
                for (size_t i = 0; i < n; ++i) arr.push_back(static_cast<unsigned>(p[i]));
                break;
            }
            case DType::Bool: {
                const auto* p = t.data<bool>();
                for (size_t i = 0; i < n; ++i) arr.push_back(p[i]);
                break;
            }
            case DType::Float16: {
                const auto* p = t.data<Float16>();
                for (size_t i = 0; i < n; ++i) {
                    arr.push_back(static_cast<float>(p[i]));
                }
                break;
            }
            case DType::BFloat16: {
                const auto* p = t.data<BFloat16>();
                for (size_t i = 0; i < n; ++i) {
                    arr.push_back(static_cast<float>(p[i]));
                }
                break;
            }
            default:
                throw std::runtime_error(
                    std::string("dtype not supported by JSON predict response: ") +
                    std::string(dtype_name(t.dtype())));
        }
        return arr;
    };

    auto json_error = [](httplib::Response& res, int status,
                         const std::string& msg) {
        json j;
        j["error"] = msg;
        res.status = status;
        res.set_content(j.dump(), "application/json");
    };

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
                res.set_content(json{{"error","missing authentication header"}}.dump(), "application/json");
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
                res.set_content(json{{"error","invalid api key"}}.dump(), "application/json");
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
        res.set_content(json{{"status","ok"},{"version","0.1.0"}}.dump(),
                        "application/json");
    });

    // Prometheus metrics
    svr.Get("/metrics", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(MetricsRegistry::instance().format_prometheus(), "text/plain");
    });

    // List models
    svr.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
        auto models = repository_.list_models();
        json j;
        j["models"] = models;
        res.set_content(j.dump(), "application/json");
    });

    // Model status
    svr.Get(R"(/v1/models/([^/]+)/status)", [this](const httplib::Request& req, httplib::Response& res) {
        auto name = req.matches[1].str();
        auto model = repository_.get_model(name);
        if (!model) {
            res.status = 404;
            res.set_content(json{{"error","model not found"}}.dump(), "application/json");
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
        json j;
        j["model_name"] = name;
        j["version"]    = model->version;
        j["status"]     = state_str;
        res.set_content(j.dump(), "application/json");
    });

    // Load model
    svr.Post(R"(/v1/models/([^/]+)/load)", [this, &json_error](const httplib::Request& req, httplib::Response& res) {
        auto name = req.matches[1].str();
        // Expect JSON body: {"model_path": "/path/to/model", "device": "cpu"}
        json body;
        try {
            body = json::parse(req.body);
        } catch (const json::parse_error& e) {
            json_error(res, 400, std::string("malformed JSON: ") + e.what());
            return;
        }
        if (!body.contains("model_path") || !body["model_path"].is_string()) {
            json_error(res, 400, "missing or non-string model_path");
            return;
        }
        std::string model_path = body["model_path"].get<std::string>();

        Device target_device = Device::cpu();
        if (body.contains("device") && body["device"].is_string()) {
            target_device = Device::from_string(body["device"].get<std::string>());
        }

        try {
            repository_.load_model(name, model_path, target_device);
            res.set_content(json{{"success",true}}.dump(), "application/json");
        } catch (const std::exception& e) {
            json j;
            j["success"] = false;
            j["message"] = e.what();
            res.status = 500;
            res.set_content(j.dump(), "application/json");
        }
    });

    // Unload model
    svr.Delete(R"(/v1/models/([^/]+))", [this](const httplib::Request& req, httplib::Response& res) {
        auto name = req.matches[1].str();
        try {
            repository_.unload_model(name);
            res.set_content(json{{"success",true}}.dump(), "application/json");
        } catch (const std::exception& e) {
            json j;
            j["success"] = false;
            j["message"] = e.what();
            res.status = 500;
            res.set_content(j.dump(), "application/json");
        }
    });

    // Predict — JSON in, JSON out.
    // Request:  {"shape":[...], "dtype":"float32", "data":[...]}
    // Response: {"output_shape":[...], "output_dtype":"...", "output_data":[...]}
    svr.Post(R"(/v1/models/([^/]+)/predict)",
             [this, &rate_limit_mutex, &rate_limit_buckets,
              &dtype_from_request, &tensor_from_json, &json_from_tensor,
              &json_error]
             (const httplib::Request& req, httplib::Response& res) {
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
                json_error(res, 429, "rate limit exceeded");
                return;
            }
            bucket.tokens -= 1.0;
        }

        auto name = req.matches[1].str();
        auto model = repository_.get_model(name);
        if (!model || model->state.load() != ModelState::READY) {
            json_error(res, 404, "model not ready");
            return;
        }

        auto start = std::chrono::high_resolution_clock::now();

        try {
            json body;
            try {
                body = json::parse(req.body);
            } catch (const json::parse_error& e) {
                json_error(res, 400, std::string("malformed JSON: ") + e.what());
                return;
            }

            if (!body.contains("shape") || !body.contains("data")) {
                json_error(res, 400, "request must contain \"shape\" and \"data\"");
                return;
            }
            // dtype defaults to float32 to preserve the previous ad-hoc
            // float-only behavior when clients omit the field.
            DType dtype = DType::Float32;
            if (body.contains("dtype") && body["dtype"].is_string()) {
                try {
                    dtype = dtype_from_request(body["dtype"].get<std::string>());
                } catch (const std::exception& e) {
                    json_error(res, 400, e.what());
                    return;
                }
            }

            Tensor input;
            try {
                input = tensor_from_json(body["shape"], body["data"], dtype, model->device);
            } catch (const std::exception& e) {
                json_error(res, 400, e.what());
                return;
            }

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

            // Pull output back to host (cheap no-op when already on CPU) and
            // serialize to JSON. contiguous() guarantees the data<T>() reads
            // are tight-packed.
            Tensor output_host = (output.device().type == Device::Type::CPU)
                ? output.contiguous()
                : output.to(Device::cpu()).contiguous();

            json resp;
            resp["output_shape"] = output_host.shape();
            resp["output_dtype"] = std::string(dtype_name(output_host.dtype()));
            resp["output_data"]  = json_from_tensor(output_host);

            res.set_content(resp.dump(), "application/json");
        } catch (const std::exception& e) {
            auto& metrics = MetricsRegistry::instance().get_metrics(name);
            metrics.error_count.fetch_add(1);
            json_error(res, 500, e.what());
        }
    });

    // A/B experiment management
    svr.Post("/v1/experiments", [this, &json_error](const httplib::Request& req, httplib::Response& res) {
        // {"name":"exp1","model_a":"v1","model_b":"v2","fraction_b":0.1}
        json body;
        try {
            body = json::parse(req.body);
        } catch (const json::parse_error& e) {
            json_error(res, 400, std::string("malformed JSON: ") + e.what());
            return;
        }
        auto str_or_empty = [&](const char* k) -> std::string {
            return (body.contains(k) && body[k].is_string())
                ? body[k].get<std::string>()
                : std::string{};
        };
        auto name    = str_or_empty("name");
        auto model_a = str_or_empty("model_a");
        auto model_b = str_or_empty("model_b");
        if (name.empty() || model_a.empty() || model_b.empty()) {
            json_error(res, 400, "missing name, model_a, or model_b");
            return;
        }
        double fraction_b = 0.1;
        if (body.contains("fraction_b") && body["fraction_b"].is_number()) {
            fraction_b = body["fraction_b"].get<double>();
        }
        traffic_router_.set_experiment(name, {model_a, model_b, fraction_b});
        res.set_content(json{{"success",true}}.dump(), "application/json");
    });

    svr.Get("/v1/experiments", [this](const httplib::Request&, httplib::Response& res) {
        auto experiments = traffic_router_.list_experiments();
        json arr = json::array();
        for (const auto& e : experiments) {
            auto [a_count, b_count] = traffic_router_.get_metrics(e);
            arr.push_back({
                {"name",       e},
                {"requests_a", a_count},
                {"requests_b", b_count},
            });
        }
        res.set_content(arr.dump(), "application/json");
    });

    svr.Delete(R"(/v1/experiments/([^/]+))", [this](const httplib::Request& req, httplib::Response& res) {
        auto name = req.matches[1].str();
        traffic_router_.remove_experiment(name);
        res.set_content(json{{"success",true}}.dump(), "application/json");
    });

    std::cout << "[TenzorServing] HTTP server listening on port " << config_.http_port << std::endl;
    svr.listen("0.0.0.0", config_.http_port);
#else
    // G.12: per the audit, the non-httplib fallback (which previously
    // entered a silent 100ms-sleep busy-loop pretending to serve) has
    // been removed. The build-time static_assert above guarantees that
    // some HTTP/gRPC transport is configured, so reaching this branch
    // means the build picked a transport other than httplib — surface a
    // typed runtime error rather than silently no-op'ing.
    throw std::runtime_error(
        "tenzor::serving::InferenceServer::serve_loop: this build has "
        "no httplib transport. Configure with TENZOR_BUILD_SERVING=ON "
        "or call the gRPC server entry point instead.");
#endif
}

} // namespace serving
} // namespace tenzor
